from __future__ import annotations

import pandas as pd

SUMMARY_COLUMNS = [
    "total_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "mdd",
    "avg_turnover",
    "total_cost",
    "final_nav",
]


def build_scenario_summary(frame: pd.DataFrame, scenario_col: str) -> pd.DataFrame:
    available = [scenario_col, *[col for col in SUMMARY_COLUMNS if col in frame.columns]]
    summary = frame.loc[:, available].copy()
    if "sharpe" in summary.columns:
        summary = summary.sort_values("sharpe", ascending=False)
    return summary.reset_index(drop=True)


def build_scenario_highlights(frame: pd.DataFrame, scenario_col: str) -> dict[str, str]:
    highlights: dict[str, str] = {}
    if frame.empty:
        return highlights
    if "sharpe" in frame.columns:
        row = frame.sort_values("sharpe", ascending=False).iloc[0]
        highlights["best_sharpe"] = f"{row[scenario_col]} ({row['sharpe']:.4f})"
    if "total_return" in frame.columns:
        row = frame.sort_values("total_return", ascending=False).iloc[0]
        highlights["best_total_return"] = f"{row[scenario_col]} ({row['total_return']:.4f})"
    if "mdd" in frame.columns:
        row = frame.sort_values("mdd", ascending=False).iloc[0]
        highlights["lowest_drawdown"] = f"{row[scenario_col]} ({row['mdd']:.4f})"
    return highlights


def render_scenario_summary_text(frame: pd.DataFrame, scenario_col: str) -> str:
    summary = build_scenario_summary(frame, scenario_col)
    if summary.empty:
        return "No scenario summary available."
    return summary.to_string(index=False, float_format=lambda value: f"{value:.4f}")
