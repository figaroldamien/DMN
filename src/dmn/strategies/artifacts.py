"""Artifact metadata objects used by live training/inference workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DMNLSTMArtifact:
    """Serializable metadata required to reload and reuse a trained LSTM model."""

    tickers: list[str]
    feature_names: list[str]
    seq_len: int
    hidden: int
    dropout: float
    lr: float
    epochs: int
    batch_size: int
    turnover_lambda: float
    seed: int
    vol_span: int
    sigma_target_annual: float
    cutoff_date: str
    train_start_date: str
    train_end_date: str
    n_train_samples: int
    trained_at_utc: str
    cost_bps: float = 0.0
    portfolio_vol_target: bool = True
    min_obs: int = 0

    def to_dict(self) -> dict:
        """Convert artifact metadata to a plain dictionary."""

        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "DMNLSTMArtifact":
        """Build artifact metadata from persisted payload."""

        defaults = {
            "cost_bps": 0.0,
            "portfolio_vol_target": True,
            "min_obs": 0,
        }
        for k, v in defaults.items():
            raw.setdefault(k, v)
        return DMNLSTMArtifact(**raw)
