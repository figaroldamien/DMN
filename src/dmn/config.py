from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Perf:
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    mdd: float
    pct_pos: float
    avg_profit_over_avg_loss: float
    avg_turnover: float


@dataclass
class BacktestConfig:
    sigma_target_annual: float = 0.15
    vol_span: int = 60
    cost_bps: float = 0.0
    portfolio_vol_target: bool = True
    min_obs: int = 252 + 60 + 5


@dataclass
class RunConfig:
    market: str = "cac40"
    start: str = "2000-01-01"
    sector: str | None = None
    sub_sector: str | None = None
    run_ml: bool = False
    run_dmn: bool = True
    backtest: BacktestConfig = field(
        default_factory=lambda: BacktestConfig(
            sigma_target_annual=0.15,
            vol_span=60,
            cost_bps=2.0,
            portfolio_vol_target=True,
            min_obs=400,
        )
    )

    def to_dict(self) -> dict:
        return asdict(self)
