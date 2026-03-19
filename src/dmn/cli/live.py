from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..config import BacktestConfig, RunConfig
from ..data import load_prices_yf
from ..strategies.live import (
    load_lstm_artifact,
    predict_positions_from_model,
    save_lstm_artifact,
    train_lstm_until_cutoff,
)
from .common import apply_argsets, print_config_payload, resolve_config_tickers


def _resolve_cli_tickers(args: argparse.Namespace) -> list[str]:
    if args.ticker:
        return [args.ticker]

    cfg = RunConfig(
        market=args.market,
        ticker=args.ticker,
        start=args.start,
        sector=args.sector,
        sub_sector=args.sub_sector,
    )
    parser = argparse.ArgumentParser(add_help=False)
    return resolve_config_tickers(cfg, parser)


def _resolve_cutoff_date(
    dates: pd.Index,
    mode: str,
    cutoff_date: str | None = None,
    reference_date: str | None = None,
) -> pd.Timestamp:
    ref = pd.Timestamp(reference_date).normalize() if reference_date else pd.Timestamp.now().normalize()

    if mode == "date":
        if not cutoff_date:
            raise ValueError("--cutoff-date is required when --cutoff-mode=date")
        candidate = pd.Timestamp(cutoff_date).normalize()
    elif mode == "year_end_prev":
        candidate = pd.Timestamp(year=ref.year - 1, month=12, day=31)
    elif mode == "month_end_prev":
        candidate = (ref.replace(day=1) - pd.Timedelta(days=1)).normalize()
    elif mode == "yesterday":
        candidate = (ref - pd.Timedelta(days=1)).normalize()
    else:
        raise ValueError(f"Unknown cutoff mode: {mode}")

    eligible = dates[dates <= candidate]
    if len(eligible) == 0:
        raise ValueError(f"No market data available on or before cutoff candidate {candidate.date()}")
    return pd.Timestamp(eligible[-1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/predict DMN LSTM with persisted model artifacts.")
    parser.add_argument("--no-print-config", dest="print_config", action="store_false", default=True, help="Disable effective configuration display.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a DMN LSTM up to a cutoff date and save artifact")
    existing_train = train.add_argument_group("Existing Arguments (from main)")
    apply_argsets(
        existing_train,
        "universe",
        "backtest",
        overrides={
            "universe": {"mutually_exclusive": True, "start_default": "2000-01-01"},
            "backtest": {
                "sigma_target_annual_default": 0.15,
                "vol_span_default": 60,
                "cost_bps_default": 2.0,
                "min_obs_default": 400,
                "portfolio_vol_target_default": True,
            },
        },
    )
    live_train = train.add_argument_group("New Live Arguments")
    live_train.add_argument("--cutoff-mode", choices=["year_end_prev", "month_end_prev", "yesterday", "date"], default="year_end_prev")
    live_train.add_argument("--cutoff-date", default=None, help="Cutoff date (required with --cutoff-mode=date)")
    live_train.add_argument("--reference-date", default=None, help="Reference date for relative cutoff modes (YYYY-MM-DD)")
    live_train.add_argument("--artifact-dir", default="artifacts/dmn", help="Directory for saved model artifacts")
    live_train.add_argument("--artifact-name", default=None, help="Optional artifact filename stem")
    live_train.add_argument("--seq-len", type=int, default=63)
    live_train.add_argument("--hidden", type=int, default=32)
    live_train.add_argument("--dropout", type=float, default=0.1)
    live_train.add_argument(
        "--use-ticker-embedding",
        dest="use_ticker_embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    live_train.add_argument("--lr", type=float, default=1e-3)
    live_train.add_argument("--epochs", type=int, default=20)
    live_train.add_argument("--batch-size", type=int, default=256)
    live_train.add_argument("--turnover-lambda", type=float, default=0.0)
    live_train.add_argument("--seed", type=int, default=0)
    live_train.add_argument("--min-train-samples", type=int, default=2000)

    predict = sub.add_parser("predict", help="Load a trained DMN artifact and generate future positions")
    existing_predict = predict.add_argument_group("Existing Arguments (from main)")
    apply_argsets(
        existing_predict,
        "universe",
        "backtest",
        overrides={
            "universe": {"mutually_exclusive": True, "start_default": "2000-01-01"},
            "backtest": {
                "sigma_target_annual_default": 0.15,
                "vol_span_default": 60,
                "cost_bps_default": 2.0,
                "min_obs_default": 400,
                "portfolio_vol_target_default": True,
            },
        },
    )
    live_predict = predict.add_argument_group("New Live Arguments")
    live_predict.add_argument("--artifact-path", required=True, help="Path to .pt artifact produced by train")
    live_predict.add_argument("--from-date", default=None, help="First prediction date (default: today)")
    live_predict.add_argument("--to-date", default=None, help="Last prediction date (default: latest available)")
    live_predict.add_argument("--output", default=None, help="Output CSV path for predicted positions")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    tickers = _resolve_cli_tickers(args)
    prices = load_prices_yf(tickers, start=args.start)

    cutoff = _resolve_cutoff_date(
        dates=prices.index,
        mode=args.cutoff_mode,
        cutoff_date=args.cutoff_date,
        reference_date=args.reference_date,
    )

    cfg = BacktestConfig(
        sigma_target_annual=args.sigma_target_annual,
        vol_span=args.vol_span,
        cost_bps=args.cost_bps,
        portfolio_vol_target=args.portfolio_vol_target,
        min_obs=args.min_obs,
    )
    train_config = {
        "command": "train",
        "market": args.market,
        "ticker": args.ticker,
        "resolved_tickers": tickers,
        "start": args.start,
        "sector": args.sector,
        "sub_sector": args.sub_sector,
        "cutoff_mode": args.cutoff_mode,
        "cutoff_date": args.cutoff_date,
        "reference_date": args.reference_date,
        "resolved_cutoff": str(cutoff.date()),
        "artifact_dir": args.artifact_dir,
        "artifact_name": args.artifact_name,
        "backtest": {
            "sigma_target_annual": args.sigma_target_annual,
            "vol_span": args.vol_span,
            "cost_bps": args.cost_bps,
            "portfolio_vol_target": args.portfolio_vol_target,
            "min_obs": args.min_obs,
        },
        "model": {
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "use_ticker_embedding": args.use_ticker_embedding,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "turnover_lambda": args.turnover_lambda,
            "seed": args.seed,
            "min_train_samples": args.min_train_samples,
        },
    }
    if args.print_config:
        print("Effective train configuration:")
        print_config_payload(train_config)

    model, artifact = train_lstm_until_cutoff(
        prices=prices,
        cfg=cfg,
        cutoff_date=cutoff,
        seq_len=args.seq_len,
        hidden=args.hidden,
        dropout=args.dropout,
        use_ticker_embedding=args.use_ticker_embedding,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        turnover_lambda=args.turnover_lambda,
        seed=args.seed,
        min_train_samples=args.min_train_samples,
    )

    # Default artifact naming:
    # - market N -> N_YYYYMMDD
    # - ticker T -> T_YYYYMMDD
    # Explicit --artifact-name keeps priority.
    inferred_prefix = args.ticker if args.ticker else (args.market or "cac40")
    inferred_name = f"{inferred_prefix}_{artifact.cutoff_date.replace('-', '')}"
    out_path = save_lstm_artifact(
        model=model,
        artifact=artifact,
        artifact_dir=args.artifact_dir,
        artifact_name=args.artifact_name or inferred_name,
    )
    print(f"Saved artifact: {out_path}")
    print(f"Cutoff used: {artifact.cutoff_date}")
    print(f"Train samples: {artifact.n_train_samples}")
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    model, artifact = load_lstm_artifact(args.artifact_path)

    if args.ticker or args.market:
        tickers = _resolve_cli_tickers(args)
    else:
        tickers = artifact.tickers

    start = args.start or artifact.train_start_date
    prices = load_prices_yf(tickers, start=start)
    predict_config = {
        "command": "predict",
        "artifact_path": args.artifact_path,
        "artifact_cutoff_date": artifact.cutoff_date,
        "artifact_tickers": artifact.tickers,
        "market": args.market,
        "ticker": args.ticker,
        "resolved_tickers": tickers,
        "start": start,
        "backtest": {
            "sigma_target_annual": args.sigma_target_annual,
            "vol_span": args.vol_span,
            "cost_bps": args.cost_bps,
            "portfolio_vol_target": args.portfolio_vol_target,
            "min_obs": args.min_obs,
        },
        "from_date": args.from_date,
        "to_date": args.to_date,
        "output": args.output,
    }
    positions = predict_positions_from_model(
        prices=prices,
        model=model,
        artifact=artifact,
        from_date=args.from_date,
        to_date=args.to_date,
    )

    inferred_prefix = args.ticker or args.market
    if not inferred_prefix:
        inferred_prefix = artifact.tickers[0] if len(artifact.tickers) == 1 else "N"
    inferred_output = Path("output") / f"{inferred_prefix}_{artifact.cutoff_date.replace('-', '')}.csv"
    out_path = Path(args.output) if args.output else inferred_output
    predict_config["output"] = str(out_path)

    if args.print_config:
        print("Effective predict configuration:")
        print_config_payload(predict_config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(out_path)
    print(f"Saved predictions: {out_path}")
    print(f"Prediction rows: {len(positions)}")
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            return _run_train(args)
        if args.command == "predict":
            return _run_predict(args)
        parser.error(f"Unknown command: {args.command}")
    except ImportError:
        print("Install runtime dependencies: pip install yfinance torch pandas numpy")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
