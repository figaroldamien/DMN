from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from market_tickers_data.universes import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    DJI_COMPONENTS,
    EUROSTOXX50_COMPONENTS,
    EUROSTOXX600_COMPONENTS,
    FUTURES_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
    SBF120_COMPONENTS,
    SP500_COMPONENTS,
    WORLD_INDEX_COMPONENTS,
)
from optimal_tf.config_io import load_config
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.strategies.common import resolve_allocation_date

from .io import ensure_output_dir, write_json, write_quality_artifacts, write_request_json
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
    'sbf120': SBF120_COMPONENTS,
    'eurostoxx50': EUROSTOXX50_COMPONENTS,
    'eurostoxx600': EUROSTOXX600_COMPONENTS,
    'index': INDEX_COMPONENTS,
    'futures': FUTURES_COMPONENTS,
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
                'sub_category': str(meta.get('sub_category', '') or '').strip(),
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


def _compute_monthly_return_history(history: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = history.resample('ME').last().ffill()
    monthly_returns = monthly_prices.pct_change().dropna(how='all')
    if monthly_returns.empty:
        return pd.DataFrame(index=history.columns)
    monthly_returns = monthly_returns.T
    monthly_returns.columns = [pd.Timestamp(ts).strftime('%b-%y') for ts in monthly_returns.columns]
    monthly_returns.index.name = 'ticker'
    return monthly_returns


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


def _build_monthly_consolidated_frame(monthly_ticker_frame: pd.DataFrame) -> pd.DataFrame:
    month_columns = [column for column in monthly_ticker_frame.columns if column not in {'sector', 'sub_sector', 'category', 'sub_category', 'description'}]
    rows: list[dict[str, object]] = []
    for sector, sector_frame in monthly_ticker_frame.groupby('sector', sort=True):
        sector_metrics = sector_frame[month_columns].mean(axis=0)
        rows.append({
            'level': 'sector',
            'sector': sector,
            'sub_sector': '',
            'label': sector,
            'num_tickers': int(len(sector_frame)),
            **sector_metrics.to_dict(),
        })
        for sub_sector, sub_frame in sector_frame.groupby('sub_sector', sort=True):
            sub_metrics = sub_frame[month_columns].mean(axis=0)
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


def _build_ticker_nav_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    filled = history.ffill().copy()
    first_valid = filled.apply(
        lambda column: column.dropna().iloc[0] if not column.dropna().empty else np.nan
    )
    rebased = 100.0 * filled.divide(first_valid.replace(0.0, np.nan), axis="columns")
    return rebased.dropna(how='all').sort_index(axis=1)


def run_market_synthesis(request: MarketSynthesisRequest) -> MarketSynthesisResult:
    universe_cfg, *_ = load_config(request.config_path)
    universe_name = request.universe or universe_cfg.name
    start = request.start or universe_cfg.start

    universe_cfg = replace(universe_cfg, name=universe_name, start=start)
    prices, quality_report_obj = load_filtered_prices_for_universe(universe_cfg, refresh_policy=request.refresh_policy)
    quality_report = asdict(quality_report_obj)
    as_of_date = resolve_allocation_date(prices.index, as_of_date=request.as_of_date)
    history = prices.loc[prices.index <= as_of_date].ffill()
    if history.empty:
        raise ValueError(f'No price history available on or before {as_of_date.date()}.')

    ticker_momentum = _compute_ticker_momentum(history)
    monthly_ticker_returns = _compute_monthly_return_history(history)
    ticker_nav_frame = _build_ticker_nav_frame(history)
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
        ticker_frame.insert(3, 'sub_category', metadata.loc[sorted_tickers, 'sub_category'].to_numpy())
        ticker_frame.insert(4, 'description', metadata.loc[sorted_tickers, 'description'].to_numpy())
        ticker_frame.insert(5, 'hierarchy_complete', metadata.loc[sorted_tickers, 'hierarchy_complete'].to_numpy())
        classified_ticker_frame = ticker_frame.loc[ticker_frame['hierarchy_complete']].drop(columns=['hierarchy_complete'])
        consolidated_frame = _build_consolidated_frame(classified_ticker_frame)

        monthly_ticker_frame = monthly_ticker_returns.loc[sorted_tickers].copy()
        monthly_ticker_frame.insert(0, 'sector', metadata.loc[sorted_tickers, 'sector'].to_numpy())
        monthly_ticker_frame.insert(1, 'sub_sector', metadata.loc[sorted_tickers, 'sub_sector'].to_numpy())
        monthly_ticker_frame.insert(2, 'category', metadata.loc[sorted_tickers, 'category'].to_numpy())
        monthly_ticker_frame.insert(3, 'sub_category', metadata.loc[sorted_tickers, 'sub_category'].to_numpy())
        monthly_ticker_frame.insert(4, 'description', metadata.loc[sorted_tickers, 'description'].to_numpy())
        monthly_ticker_frame.insert(5, 'hierarchy_complete', metadata.loc[sorted_tickers, 'hierarchy_complete'].to_numpy())
        classified_monthly_ticker_frame = monthly_ticker_frame.loc[monthly_ticker_frame['hierarchy_complete']].drop(columns=['hierarchy_complete'])
        monthly_consolidated_frame = _build_monthly_consolidated_frame(classified_monthly_ticker_frame)

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
        monthly_ticker_frame = monthly_ticker_frame.drop(columns=['sector', 'sub_sector'])
        monthly_ticker_frame.index = index
    elif bool(category_mask.any()):
        synthesis_mode = 'category'
        metadata = metadata.copy()
        metadata['category_complete'] = category_mask.astype(bool)
        metadata['category'] = metadata['category'].where(category_mask, 'category_unclassified')
        metadata = metadata.sort_values(['category_complete', 'category'], ascending=[False, True])
        sorted_tickers = list(metadata.index)

        ticker_frame = ticker_momentum.loc[sorted_tickers].copy()
        ticker_frame.insert(0, 'category', metadata.loc[sorted_tickers, 'category'].to_numpy())
        ticker_frame.insert(1, 'sub_category', metadata.loc[sorted_tickers, 'sub_category'].to_numpy())
        ticker_frame.insert(2, 'description', metadata.loc[sorted_tickers, 'description'].to_numpy())
        ticker_frame.insert(3, 'category_complete', metadata.loc[sorted_tickers, 'category_complete'].to_numpy())

        monthly_ticker_frame = monthly_ticker_returns.loc[sorted_tickers].copy()
        monthly_ticker_frame.insert(0, 'category', metadata.loc[sorted_tickers, 'category'].to_numpy())
        monthly_ticker_frame.insert(1, 'sub_category', metadata.loc[sorted_tickers, 'sub_category'].to_numpy())
        monthly_ticker_frame.insert(2, 'description', metadata.loc[sorted_tickers, 'description'].to_numpy())
        monthly_ticker_frame.insert(3, 'category_complete', metadata.loc[sorted_tickers, 'category_complete'].to_numpy())

        consolidated_frame = pd.DataFrame()
        monthly_consolidated_frame = pd.DataFrame()
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
        monthly_ticker_frame = monthly_ticker_frame.drop(columns=['category'])
        monthly_ticker_frame.index = index
    else:
        consolidated_frame = pd.DataFrame()
        monthly_consolidated_frame = pd.DataFrame()
        sector_nav_frame = pd.DataFrame()
        ticker_frame = ticker_momentum.sort_index()
        monthly_ticker_frame = monthly_ticker_returns.loc[ticker_frame.index].copy() if not monthly_ticker_returns.empty else pd.DataFrame(index=ticker_frame.index)
        if not metadata.empty:
            ticker_frame.insert(0, 'description', metadata.loc[ticker_frame.index, 'description'].to_numpy())
            ticker_frame.insert(1, 'category', metadata.loc[ticker_frame.index, 'category'].to_numpy())
            ticker_frame.insert(2, 'sub_category', metadata.loc[ticker_frame.index, 'sub_category'].to_numpy())
            monthly_ticker_frame.insert(0, 'description', metadata.loc[ticker_frame.index, 'description'].to_numpy())
            monthly_ticker_frame.insert(1, 'category', metadata.loc[ticker_frame.index, 'category'].to_numpy())
            monthly_ticker_frame.insert(2, 'sub_category', metadata.loc[ticker_frame.index, 'sub_category'].to_numpy())

    outdir = ensure_output_dir(request.output_dir)
    files: dict[str, Path] = {}
    if outdir is not None:
        ticker_path = outdir / 'ticker_momentum.csv'
        ticker_frame.to_csv(ticker_path)
        files['ticker_momentum'] = ticker_path

        monthly_ticker_path = outdir / 'ticker_monthly_returns.csv'
        monthly_ticker_frame.to_csv(monthly_ticker_path)
        files['ticker_monthly_returns'] = monthly_ticker_path

        if not consolidated_frame.empty:
            consolidated_path = outdir / 'consolidated_momentum.csv'
            consolidated_frame.to_csv(consolidated_path, index=False)
            files['consolidated_momentum'] = consolidated_path
        if not monthly_consolidated_frame.empty:
            monthly_consolidated_path = outdir / 'consolidated_monthly_returns.csv'
            monthly_consolidated_frame.to_csv(monthly_consolidated_path, index=False)
            files['consolidated_monthly_returns'] = monthly_consolidated_path
        if not sector_nav_frame.empty:
            sector_nav_path = outdir / 'sector_nav.csv'
            sector_nav_frame.to_csv(sector_nav_path)
            files['sector_nav'] = sector_nav_path
        if not ticker_nav_frame.empty:
            ticker_nav_path = outdir / 'ticker_nav.csv'
            ticker_nav_frame.to_csv(ticker_nav_path)
            files['ticker_nav'] = ticker_nav_path
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
                'monthly_columns': list(monthly_ticker_returns.columns),
                'quality_report': quality_report,
            },
        )
        files['summary'] = outdir / 'summary.json'
        files.update(write_quality_artifacts(outdir, quality_report))

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
        ticker_nav_frame=ticker_nav_frame,
        monthly_consolidated_frame=monthly_consolidated_frame,
        monthly_ticker_frame=monthly_ticker_frame,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )
