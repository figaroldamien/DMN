from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import UniverseConfig
from .data import load_prices_for_universe
from .features import compute_returns


@dataclass(frozen=True)
class PriceQualityFilterReport:
    enabled: bool
    universe: str
    start: str
    reference_start: str
    reference_rows: int
    total_tickers: int
    kept_tickers: tuple[str, ...]
    excluded_tickers: tuple[str, ...]
    excluded_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    min_history_days: int = 0
    min_coverage_ratio: float = 0.0
    max_internal_missing: int = 0
    max_abs_return: float = 0.0
    require_latest_price: bool = True


def _series_internal_missing(series: pd.Series) -> int:
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()
    if first_valid is None or last_valid is None:
        return 0
    return int(series.loc[first_valid:last_valid].isna().sum())


def filter_price_frame_by_quality(
    prices: pd.DataFrame,
    *,
    universe: str,
    start: str,
    evaluation_start: str | None,
    quality_cfg: UniverseConfig,
) -> tuple[pd.DataFrame, PriceQualityFilterReport]:
    reference_start_ts = pd.Timestamp(evaluation_start or start)
    reference_start = reference_start_ts.strftime("%Y-%m-%d")
    reference_frame = prices.loc[prices.index >= reference_start_ts].copy()
    reference_rows = int(len(reference_frame.index))
    if prices.empty:
        return prices.copy(), PriceQualityFilterReport(
            enabled=bool(quality_cfg.quality_filter_enabled),
            universe=universe,
            start=start,
            reference_start=reference_start,
            reference_rows=reference_rows,
            total_tickers=0,
            kept_tickers=(),
            excluded_tickers=(),
            excluded_reasons={},
            min_history_days=int(quality_cfg.quality_min_history_days),
            min_coverage_ratio=float(quality_cfg.quality_min_coverage_ratio),
            max_internal_missing=int(quality_cfg.quality_max_internal_missing),
            max_abs_return=float(quality_cfg.quality_max_abs_return),
            require_latest_price=bool(quality_cfg.quality_require_latest_price),
        )

    if not quality_cfg.quality_filter_enabled:
        kept = tuple(str(column) for column in prices.columns)
        return prices.copy(), PriceQualityFilterReport(
            enabled=False,
            universe=universe,
            start=start,
            reference_start=reference_start,
            reference_rows=reference_rows,
            total_tickers=int(len(prices.columns)),
            kept_tickers=kept,
            excluded_tickers=(),
            excluded_reasons={},
            min_history_days=int(quality_cfg.quality_min_history_days),
            min_coverage_ratio=float(quality_cfg.quality_min_coverage_ratio),
            max_internal_missing=int(quality_cfg.quality_max_internal_missing),
            max_abs_return=float(quality_cfg.quality_max_abs_return),
            require_latest_price=bool(quality_cfg.quality_require_latest_price),
        )

    returns = compute_returns(prices)
    total_days = max(reference_rows, 1)
    last_date = prices.index.max()
    kept: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, list[str]] = {}

    for ticker in prices.columns:
        series = prices[ticker]
        reference_series = reference_frame[ticker] if ticker in reference_frame.columns else series.iloc[0:0]
        ticker_reasons: list[str] = []
        valid_count = int(reference_series.notna().sum())
        coverage_ratio = valid_count / total_days
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        history_days = 0 if first_valid is None or last_valid is None else int(series.loc[first_valid:last_valid].notna().sum())
        internal_missing = _series_internal_missing(series)
        max_abs_return = float(returns[ticker].abs().max()) if returns[ticker].notna().any() else float("nan")

        if quality_cfg.quality_require_latest_price and pd.isna(series.loc[last_date]):
            ticker_reasons.append("missing_latest_price")
        if history_days < int(quality_cfg.quality_min_history_days):
            ticker_reasons.append("history_too_short")
        if coverage_ratio < float(quality_cfg.quality_min_coverage_ratio):
            ticker_reasons.append("coverage_too_low")
        if internal_missing > int(quality_cfg.quality_max_internal_missing):
            ticker_reasons.append("internal_gaps")
        if pd.notna(max_abs_return) and max_abs_return > float(quality_cfg.quality_max_abs_return):
            ticker_reasons.append("return_jump")
        if ticker_reasons:
            excluded.append(str(ticker))
            reasons[str(ticker)] = ticker_reasons
        else:
            kept.append(str(ticker))

    filtered = prices.loc[:, kept].copy()
    if filtered.empty:
        raise ValueError(
            f"All {len(prices.columns)} tickers were excluded by the quality filter for universe '{universe}'."
        )
    report = PriceQualityFilterReport(
        enabled=True,
        universe=universe,
        start=start,
        reference_start=reference_start,
        reference_rows=reference_rows,
        total_tickers=int(len(prices.columns)),
        kept_tickers=tuple(kept),
        excluded_tickers=tuple(excluded),
        excluded_reasons={ticker: tuple(reason_list) for ticker, reason_list in reasons.items()},
        min_history_days=int(quality_cfg.quality_min_history_days),
        min_coverage_ratio=float(quality_cfg.quality_min_coverage_ratio),
        max_internal_missing=int(quality_cfg.quality_max_internal_missing),
        max_abs_return=float(quality_cfg.quality_max_abs_return),
        require_latest_price=bool(quality_cfg.quality_require_latest_price),
    )
    return filtered, report


def load_filtered_prices_for_universe(
    universe_cfg: UniverseConfig,
    *,
    evaluation_start: str | None = None,
    refresh_policy: str = "auto",
) -> tuple[pd.DataFrame, PriceQualityFilterReport]:
    prices = load_prices_for_universe(universe_cfg.name, start=universe_cfg.start, refresh_policy=refresh_policy)
    return filter_price_frame_by_quality(
        prices,
        universe=universe_cfg.name,
        start=universe_cfg.start,
        evaluation_start=evaluation_start,
        quality_cfg=universe_cfg,
    )
