from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_tickers_data.universes import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    DJI_COMPONENTS,
    EUROSTOXX50_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
    SP500_COMPONENTS,
    WORLD_INDEX_COMPONENTS,
)
from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.strategies.common import resolve_allocation_date

from .io import ensure_output_dir, write_json, write_request_json
from .models import MarketSynthesisRequest, MarketSynthesisResult, RunArtifacts

MOMENTUM_WINDOWS = {
    'annual': 252,
    'semiannual': 126,
    'quarterly': 63,
    'monthly': 21,
    'weekly': 5,
    'daily': 1,
}

UNIVERSE_COMPONENTS = {
    'nasdaq100': NASDAQ100_COMPONENTS,
    'cac40': CAC40_COMPONENTS,
    'dji': DJI_COMPONENTS,
    'sp500': SP500_COMPONENTS,
    'eurostoxx50': EUROSTOXX50_COMPONENTS,
    'index': INDEX_COMPONENTS,
    'dataset': DATASET_COMPONENTS,
    'dataset_all': DATASET_COMPONENTS,
    'table8_all': DATASET_COMPONENTS,
    'table_8': DATASET_COMPONENTS,
    'world_index': WORLD_INDEX_COMPONENTS,
}


def _component_metadata(universe: str, tickers: list[str]) -> pd.DataFrame:
    components = UNIVERSE_COMPONENTS.get(universe, {})
    rows: list[dict[str, str]] = []
    for ticker in tickers:
        meta = components.get(str(ticker), {})
        rows.append(
            {
                'ticker': str(ticker),
                'sector': str(meta.get('sector', '') or '').strip(),
                'sub_sector': str(meta.get('sub_sector', '') or '').strip(),
                'category': str(meta.get('category', '') or '').strip(),
                'description': str(meta.get('description', '') or '').strip(),
            }
        )
    return pd.DataFrame(rows).set_index('ticker')


def _hierarchy_mask(metadata: pd.DataFrame) -> pd.Series:
    if metadata.empty:
        return pd.Series(dtype=bool)
    sector_ok = metadata['sector'].astype(str).str.strip().ne('')
    sub_sector_ok = metadata['sub_sector'].astype(str).str.strip().ne('')
    return (sector_ok & sub_sector_ok).astype(bool)


def _category_mask(metadata: pd.DataFrame) -> pd.Series:
    if metadata.empty:
        return pd.Series(dtype=bool)
    return metadata['category'].astype(str).str.strip().ne('').astype(bool)


def _compute_ticker_momentum(history: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for label, periods in MOMENTUM_WINDOWS.items():
        if len(history.index) <= periods:
            rows[label] = pd.Series(index=history.columns, dtype=float)
        else:
            rows[label] = history.pct_change(periods=periods).iloc[-1]
    frame = pd.DataFrame(rows)
    frame.index.name = 'ticker'
    return frame


def _build_consolidated_frame(ticker_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sector, sector_frame in ticker_frame.groupby('sector', sort=True):
        sector_metrics = sector_frame[list(MOMENTUM_WINDOWS)].mean(axis=0)
        rows.append({
            'level': 'sector',
            'sector': sector,
            'sub_sector': '',
            'label': sector,
            'num_tickers': int(len(sector_frame)),
            **sector_metrics.to_dict(),
        })
        for sub_sector, sub_frame in sector_frame.groupby('sub_sector', sort=True):
            sub_metrics = sub_frame[list(MOMENTUM_WINDOWS)].mean(axis=0)
            rows.append({
                'level': 'sub_sector',
                'sector': sector,
                'sub_sector': sub_sector,
                'label': f'  {sub_sector}',
                'num_tickers': int(len(sub_frame)),
                **sub_metrics.to_dict(),
            })
    return pd.DataFrame(rows)


def _build_group_nav_frame(history: pd.DataFrame, metadata: pd.DataFrame, *, group_column: str, mask_column: str) -> pd.DataFrame:
    if history.empty or metadata.empty:
        return pd.DataFrame()
    classified = metadata.loc[metadata[mask_column]].copy()
    if classified.empty:
        return pd.DataFrame()
    group_series: dict[str, pd.Series] = {}
    daily_returns = history.pct_change().dropna(how='all')
    if daily_returns.empty:
        return pd.DataFrame()
    for group, group_meta in classified.groupby(group_column, sort=True):
        tickers = [ticker for ticker in group_meta.index if ticker in daily_returns.columns]
        if not tickers:
            continue
        group_returns = daily_returns[tickers].mean(axis=1)
        group_nav = 100.0 * (1.0 + group_returns.fillna(0.0)).cumprod()
        group_series[group] = group_nav
    if not group_series:
        return pd.DataFrame()
    return pd.DataFrame(group_series).sort_index(axis=1)


def run_market_synthesis(request: MarketSynthesisRequest) -> MarketSynthesisResult:
    universe_cfg, *_ = load_config(request.config_path)
    universe_name = request.universe or universe_cfg.name
    start = request.start or universe_cfg.start

    prices = load_prices_for_universe(universe_name, start=start, refresh_policy=request.refresh_policy)
    as_of_date = resolve_allocation_date(prices.index, as_of_date=request.as_of_date)
    history = prices.loc[prices.index <= as_of_date].ffill()
    if history.empty:
        raise ValueError(f'No price history available on or before {as_of_date.date()}.')

    ticker_momentum = _compute_ticker_momentum(history)
    metadata = _component_metadata(universe_name, list(ticker_momentum.index))
    hierarchy_mask = _hierarchy_mask(metadata)
    category_mask = _category_mask(metadata)
    has_hierarchy = bool(hierarchy_mask.any())
    synthesis_mode = 'none'

    if has_hierarchy:
        synthesis_mode = 'hierarchy'
        metadata = metadata.copy()
        metadata['sector'] = metadata['sector'].where(hierarchy_mask, 'Unclassified')
        metadata['sub_sector'] = metadata['sub_sector'].where(hierarchy_mask, 'Unclassified')
        metadata['hierarchy_complete'] = hierarchy_mask.astype(bool)
        metadata = metadata.sort_values(['hierarchy_complete', 'sector', 'sub_sector'], ascending=[False, True, True])
        sorted_tickers = list(metadata.index)
        ticker_frame = ticker_momentum.loc[sorted_tickers].copy()
        ticker_frame.insert(0, 'sector', metadata.loc[sorted_tickers, 'sector'].to_numpy())
        ticker_frame.insert(1, 'sub_sector', metadata.loc[sorted_tickers, 'sub_sector'].to_numpy())
        ticker_frame.insert(2, 'category', metadata.loc[sorted_tickers, 'category'].to_numpy())
        ticker_frame.insert(3, 'description', metadata.loc[sorted_tickers, 'description'].to_numpy())
        ticker_frame.insert(4, 'hierarchy_complete', metadata.loc[sorted_tickers, 'hierarchy_complete'].to_numpy())
        classified_ticker_frame = ticker_frame.loc[ticker_frame['hierarchy_complete']].drop(columns=['hierarchy_complete'])
        consolidated_frame = _build_consolidated_frame(classified_ticker_frame)
        sector_nav_frame = _build_group_nav_frame(history, metadata, group_column='sector', mask_column='hierarchy_complete')
        index = pd.MultiIndex.from_arrays(
            [
                metadata.loc[sorted_tickers, 'sector'].to_numpy(),
                metadata.loc[sorted_tickers, 'sub_sector'].to_numpy(),
                pd.Index(sorted_tickers, dtype='object').to_numpy(),
            ],
            names=['sector', 'sub_sector', 'ticker'],
        )
        ticker_frame = ticker_frame.drop(columns=['sector', 'sub_sector'])
        ticker_frame.index = index
    elif bool(category_mask.any()):
        synthesis_mode = 'category'
        metadata = metadata.copy()
        metadata['category_complete'] = category_mask.astype(bool)
        metadata['category'] = metadata['category'].where(category_mask, 'category_unclassified')
        metadata = metadata.sort_values(['category_complete', 'category'], ascending=[False, True])
        sorted_tickers = list(metadata.index)
        ticker_frame = ticker_momentum.loc[sorted_tickers].copy()
        ticker_frame.insert(0, 'category', metadata.loc[sorted_tickers, 'category'].to_numpy())
        ticker_frame.insert(1, 'description', metadata.loc[sorted_tickers, 'description'].to_numpy())
        ticker_frame.insert(2, 'category_complete', metadata.loc[sorted_tickers, 'category_complete'].to_numpy())
        consolidated_frame = pd.DataFrame()
        sector_nav_frame = pd.DataFrame()
        index = pd.MultiIndex.from_arrays(
            [
                metadata.loc[sorted_tickers, 'category'].to_numpy(),
                pd.Index(sorted_tickers, dtype='object').to_numpy(),
            ],
            names=['category', 'ticker'],
        )
        ticker_frame = ticker_frame.drop(columns=['category'])
        ticker_frame.index = index
    else:
        consolidated_frame = pd.DataFrame()
        sector_nav_frame = pd.DataFrame()
        ticker_frame = ticker_momentum.sort_index()
        if not metadata.empty:
            ticker_frame.insert(0, 'description', metadata.loc[ticker_frame.index, 'description'].to_numpy())
            ticker_frame.insert(1, 'category', metadata.loc[ticker_frame.index, 'category'].to_numpy())


    outdir = ensure_output_dir(request.output_dir)
    files: dict[str, Path] = {}
    if outdir is not None:
        ticker_path = outdir / 'ticker_momentum.csv'
        ticker_frame.to_csv(ticker_path)
        files['ticker_momentum'] = ticker_path
        if not consolidated_frame.empty:
            consolidated_path = outdir / 'consolidated_momentum.csv'
            consolidated_frame.to_csv(consolidated_path, index=False)
            files['consolidated_momentum'] = consolidated_path
        if not sector_nav_frame.empty:
            sector_nav_path = outdir / 'sector_nav.csv'
            sector_nav_frame.to_csv(sector_nav_path)
            files['sector_nav'] = sector_nav_path
        req = write_request_json(outdir, request)
        if req is not None:
            files['request'] = req
        write_json(
            outdir,
            'summary.json',
            {
                'universe': universe_name,
                'start': start,
                'as_of_date': as_of_date.strftime('%Y-%m-%d'),
                'synthesis_mode': synthesis_mode,
                'has_hierarchy': has_hierarchy,
                'num_tickers': int(len(ticker_momentum)),
                'num_classified_tickers': int(hierarchy_mask.sum()),
                'num_unclassified_tickers': int((~hierarchy_mask).sum()),
                'num_category_classified_tickers': int(category_mask.sum()),
                'num_category_unclassified_tickers': int((~category_mask).sum()),
                'momentum_windows': MOMENTUM_WINDOWS,
            },
        )
        files['summary'] = outdir / 'summary.json'

    return MarketSynthesisResult(
        request=request,
        universe=universe_name,
        start=start,
        as_of_date=as_of_date,
        synthesis_mode=synthesis_mode,
        has_hierarchy=has_hierarchy,
        consolidated_frame=consolidated_frame,
        ticker_frame=ticker_frame,
        sector_nav_frame=sector_nav_frame,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )
