from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from market_tickers import MARKET_TICKERS

from ..config import BacktestConfig
from ..data import load_prices_yf
from ..strategies.dmn_lstm import (
    load_lstm_artifact,
    predict_positions_from_model,
    save_lstm_artifact,
    train_lstm_until_cutoff,
)
from ..universe import resolve_tickers


def _resolve_cli_tickers(args: argparse.Namespace) -> list[str]:
    if args.ticker:
        return [args.ticker]
    market = args.market or "cac40"
    return resolve_tickers(market, sector=getattr(args, "sector", None), sub_sector=getattr(args, "sub_sector", None))


def _add_existing_main_args(group: argparse._ArgumentGroup, *, include_backtest_params: bool = True) -> None:
    """
    Arguments inherited from `dmn.cli.main` (or close equivalents relevant for live train/predict).
    `run-ml` / `run-dmn` are intentionally excluded because live CLI is dedicated to DMN only.
    """
    sel = group.add_mutually_exclusive_group()
    sel.add_argument("--market", choices=sorted(MARKET_TICKERS.keys()), help="Ticker universe to load")
    sel.add_argument("--ticker", help="Single ticker to load")
    group.add_argument("--start", default="2000-01-01", help="Start date for yfinance download")
    group.add_argument("--sector", default=None, help="Optional sector filter for component-based indices")
    group.add_argument("--sub-sector", dest="sub_sector", default=None, help="Optional sub-sector filter")
    if include_backtest_params:
        group.add_argument("--sigma-target-annual", type=float, default=0.15)
        group.add_argument("--vol-span", type=int, default=60)
        group.add_argument("--cost-bps", type=float, default=2.0)
        group.add_argument("--min-obs", type=int, default=400)
        group.add_argument(
            "--portfolio-vol-target",
            dest="portfolio_vol_target",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable portfolio-level volatility targeting.",
        )


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
    _add_existing_main_args(existing_train, include_backtest_params=True)
    live_train = train.add_argument_group("New Live Arguments")
    live_train.add_argument("--cutoff-mode", choices=["year_end_prev", "month_end_prev", "yesterday", "date"], default="year_end_prev")
    live_train.add_argument("--cutoff-date", default=None, help="Cutoff date (required with --cutoff-mode=date)")
    live_train.add_argument("--reference-date", default=None, help="Reference date for relative cutoff modes (YYYY-MM-DD)")
    live_train.add_argument("--artifact-dir", default="artifacts/dmn", help="Directory for saved model artifacts")
    live_train.add_argument("--artifact-name", default=None, help="Optional artifact filename stem")
    live_train.add_argument("--seq-len", type=int, default=63)
    live_train.add_argument("--hidden", type=int, default=32)
    live_train.add_argument("--dropout", type=float, default=0.1)
    live_train.add_argument("--lr", type=float, default=1e-3)
    live_train.add_argument("--epochs", type=int, default=20)
    live_train.add_argument("--batch-size", type=int, default=256)
    live_train.add_argument("--turnover-lambda", type=float, default=0.0)
    live_train.add_argument("--seed", type=int, default=0)
    live_train.add_argument("--min-train-samples", type=int, default=2000)

    predict = sub.add_parser("predict", help="Load a trained DMN artifact and generate future positions")
    existing_predict = predict.add_argument_group("Existing Arguments (from main)")
    _add_existing_main_args(existing_predict, include_backtest_params=True)
    live_predict = predict.add_argument_group("New Live Arguments")
    live_predict.add_argument("--artifact-path", required=True, help="Path to .pt artifact produced by train")
    live_predict.add_argument("--from-date", default=None, help="First prediction date (default: today)")
    live_predict.add_argument("--to-date", default=None, help="Last prediction date (default: latest available)")
    live_predict.add_argument("--output", default="output/dmn_live_signals.csv", help="Output CSV path for predicted positions")

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
        print(json.dumps(train_config, indent=2))

    model, artifact = train_lstm_until_cutoff(
        prices=prices,
        cfg=cfg,
        cutoff_date=cutoff,
        seq_len=args.seq_len,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        turnover_lambda=args.turnover_lambda,
        seed=args.seed,
        min_train_samples=args.min_train_samples,
    )

    out_path = save_lstm_artifact(
        model=model,
        artifact=artifact,
        artifact_dir=args.artifact_dir,
        artifact_name=args.artifact_name,
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
    if args.print_config:
        print("Effective predict configuration:")
        print(json.dumps(predict_config, indent=2))

    positions = predict_positions_from_model(
        prices=prices,
        model=model,
        artifact=artifact,
        from_date=args.from_date,
        to_date=args.to_date,
    )

    out_path = Path(args.output)
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
