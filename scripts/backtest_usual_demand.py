"""Run rolling time-ordered usual-demand baseline backtests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models.usual_demand import (
    BaselineConfig,
    DEFAULT_HORIZONS_HOURS,
    build_model_ready_dataset,
    compute_usual_demand_baselines,
    evaluate_usual_demand_baseline,
    read_public_frame,
    write_backtest_artifact,
)
from src.public_data.storage import PublicDataStore


DEFAULT_INPUT_DIR = settings.processed_dir / "usual_demand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, default=DEFAULT_INPUT_DIR / "hourly_features.parquet")
    parser.add_argument("--training", type=Path, default=DEFAULT_INPUT_DIR / "baseline_training.parquet")
    parser.add_argument("--input", type=Path, help="Combined normalized public-data parquet/CSV/JSON table; builds features in memory")
    parser.add_argument("--public-root", type=Path, default=PROJECT_ROOT / "data" / "public")
    parser.add_argument("--start", help="Inclusive UTC read boundary when reading public store")
    parser.add_argument("--end", help="Exclusive UTC read boundary when reading public store")
    parser.add_argument("--timezone", default=settings.timezone)
    parser.add_argument("--horizon", type=int, action="append", dest="horizons", help="Forecast horizon in hours; repeatable")
    parser.add_argument("--lookback-days", type=int, default=180, help="Limit evaluated origins to this many latest days")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--recent-days", type=int, default=28)
    parser.add_argument("--max-history-days", type=int)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_INPUT_DIR / "usual_demand_predictions.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_DIR / "usual_demand_backtest.json")
    return parser.parse_args()


def _load_or_build(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizons = tuple(args.horizons or DEFAULT_HORIZONS_HOURS)
    if args.input is not None:
        result = build_model_ready_dataset(
            read_public_frame(args.input),
            horizons_hours=horizons,
            timezone=args.timezone,
        )
        return result.hourly, result.supervised
    if args.hourly.exists() and args.training.exists():
        return pd.read_parquet(args.hourly), pd.read_parquet(args.training)
    records = PublicDataStore(args.public_root, layer="silver").read(start=args.start, end=args.end)
    if records.empty:
        raise SystemExit("No feature files or public records found. Run scripts.build_usual_demand_dataset first.")
    result = build_model_ready_dataset(records, horizons_hours=horizons, timezone=args.timezone)
    return result.hourly, result.supervised


def main() -> int:
    args = parse_args()
    if args.lookback_days < 1:
        raise SystemExit("--lookback-days must be positive")
    hourly, supervised = _load_or_build(args)
    if supervised.empty:
        raise SystemExit("No supervised feature rows are available for backtesting.")
    supervised = supervised.copy()
    supervised["origin_timestamp"] = pd.to_datetime(supervised["origin_timestamp"], utc=True, errors="coerce")
    latest_origin = supervised["origin_timestamp"].max()
    cutoff = latest_origin - pd.Timedelta(days=args.lookback_days)
    supervised = supervised[supervised["origin_timestamp"].ge(cutoff)].copy()

    predictions = compute_usual_demand_baselines(
        supervised,
        hourly,
        config=BaselineConfig(
            min_samples=args.min_samples,
            recent_days=args.recent_days,
            max_history_days=args.max_history_days,
        ),
    )
    metrics = evaluate_usual_demand_baseline(predictions)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.predictions_output, index=False)
    write_backtest_artifact(predictions, metrics, args.output)

    overall = metrics["overall"]
    mae = overall["mae_gw"]
    wape = overall["wape"]
    mae_text = "n/a" if mae is None else f"{mae:.3f} GW"
    wape_text = "n/a" if wape is None else f"{100 * wape:.2f}%"
    print(
        f"Wrote {len(predictions):,} baseline predictions to {args.predictions_output}; "
        f"artifact={args.output}; MAE={mae_text}; WAPE={wape_text}; "
        f"weak_periods={len(metrics['weak_data_periods'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
