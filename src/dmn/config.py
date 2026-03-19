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
class ModelConfig:
    hidden: int = 32
    dropout: float = 0.1
    use_ticker_embedding: bool = True


@dataclass
class OptimizationConfig:
    strategy: str
    metric: str
    hidden_values: list[int]
    dropout_values: list[float]
    batch_size_values: list[int]
    learning_rate_values: list[float]
    epochs_values: list[int]


@dataclass
class RunConfig:
    market: str | None = None
    ticker: str | None = None
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
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig | None = None

    def to_dict(self) -> dict:
        return asdict(self)
