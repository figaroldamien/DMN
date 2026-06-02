from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def write_evaluation_outputs(result, output_dir: str | None) -> None:
    if output_dir is None:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.weights_by_rebalance.to_csv(out / "weights_by_rebalance.csv")
    if not result.base_weights_by_rebalance.empty:
        result.base_weights_by_rebalance.to_csv(out / "base_weights_by_rebalance.csv")
    if not result.effective_weights_by_rebalance.empty:
        result.effective_weights_by_rebalance.to_csv(out / "effective_weights_by_rebalance.csv")
    if len(result.signal_scale_by_rebalance) > 0:
        result.signal_scale_by_rebalance.rename("signal_scale").to_csv(out / "signal_scale.csv", header=True)
    if len(result.portfolio_vol_scale) > 0:
        result.portfolio_vol_scale.rename("portfolio_vol_scale").to_csv(out / "portfolio_vol_scale.csv", header=True)
    result.daily_returns_gross.rename("gross_return").to_csv(out / "daily_returns_gross.csv", header=True)
    result.daily_returns_net.rename("net_return").to_csv(out / "daily_returns_net.csv", header=True)
    result.turnover_by_rebalance.rename("turnover").to_csv(out / "turnover.csv", header=True)
    result.costs_by_rebalance.rename("cost").to_csv(out / "costs.csv", header=True)
    (out / "summary.json").write_text(json.dumps(asdict(result.summary), indent=2), encoding="utf-8")
