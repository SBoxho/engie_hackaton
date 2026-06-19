"""Train the 48-hour probabilistic demand forecast candidate.

Default inputs are the usual-demand dataset artifacts.  A typical run is:

    python -m scripts.build_usual_demand_dataset --horizon 1 ... --horizon 48
    python -m scripts.train_probabilistic_demand_forecast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models.probabilistic_demand import (
    DEFAULT_HORIZONS_HOURS,
    ResidualQuantileConfig,
    save_training_artifacts,
    train_residual_quantile_candidate,
)
from src.models.usual_demand import build_model_ready_dataset, read_public_frame


DEFAULT_USUAL_DIR = settings.processed_dir / "usual_demand"
DEFAULT_OUTPUT_DIR = settings.processed_dir / "demand_forecast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, default=DEFAULT_USUAL_DIR / "hourly_features.parquet")
    parser.add_argument("--training", type=Path, default=DEFAULT_USUAL_DIR / "baseline_training.parquet")
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_USUAL_DIR / "feature_manifest.json")
    parser.add_argument("--input", type=Path, help="Optional normalized public records table; builds usual-demand features in memory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timezone", default=settings.timezone)
    parser.add_argument("--horizon", type=int, action="append", dest="horizons", help="Forecast horizon in hours; repeatable")
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--min-train-samples", type=int, default=168)
    parser.add_argument("--min-validation-samples", type=int, default=48)
    parser.add_argument("--baseline-min-samples", type=int, default=5)
    parser.add_argument("--baseline-recent-days", type=int, default=28)
    parser.add_argument("--baseline-max-history-days", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--no-sklearn-fallback", action="store_true", help="Fail if LightGBM is unavailable")
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    horizons = tuple(args.horizons or DEFAULT_HORIZONS_HOURS)
    if args.input is not None:
        result = build_model_ready_dataset(
            read_public_frame(args.input),
            horizons_hours=horizons,
            timezone=args.timezone,
        )
        return result.hourly, result.supervised, result.feature_manifest
    if not args.hourly.exists() or not args.training.exists():
        raise SystemExit(
            "Usual-demand artifacts were not found. Run scripts.build_usual_demand_dataset first "
            "or pass --input with normalized public records."
        )
    hourly = pd.read_parquet(args.hourly)
    supervised = pd.read_parquet(args.training)
    manifest = {}
    if args.feature_manifest.exists():
        import json

        manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    return hourly, supervised, manifest


def main() -> int:
    args = parse_args()
    horizons = tuple(args.horizons or DEFAULT_HORIZONS_HOURS)
    hourly, supervised, feature_manifest = _load_inputs(args)
    config = ResidualQuantileConfig(
        horizons_hours=horizons,
        timezone=args.timezone,
        random_seed=args.seed,
        validation_folds=args.validation_folds,
        validation_fraction=args.validation_fraction,
        min_train_samples=args.min_train_samples,
        min_validation_samples=args.min_validation_samples,
        baseline_min_samples=args.baseline_min_samples,
        baseline_recent_days=args.baseline_recent_days,
        baseline_max_history_days=args.baseline_max_history_days,
        allow_sklearn_fallback=not args.no_sklearn_fallback,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
    )
    artifact, validation_predictions = train_residual_quantile_candidate(
        supervised,
        hourly,
        feature_manifest=feature_manifest,
        config=config,
    )
    manifest = save_training_artifacts(artifact, validation_predictions, args.output_dir)
    overall = artifact["metrics"]["overall"]
    comparison = artifact["metrics"]["baseline_comparison"]
    mae = overall.get("mae_gw")
    wape = overall.get("wape")
    mae_text = "n/a" if mae is None else f"{mae:.3f} GW"
    wape_text = "n/a" if wape is None else f"{100 * wape:.2f}%"
    print(f"Wrote demand forecast artifacts to {args.output_dir}")
    print(f"Status: {artifact['status']}")
    if artifact.get("rejection_reason"):
        print(f"Rejection reason: {artifact['rejection_reason']}")
    print(
        f"Validation MAE={mae_text}; WAPE={wape_text}; "
        f"strongest_baseline={comparison.get('strongest_baseline')}; "
        f"relative_improvement={comparison.get('relative_improvement')}"
    )
    print(f"Registry manifest: {args.output_dir / 'artifact_manifest.json'}")
    print(f"Artifact checksums: {manifest['artifact_checksums']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
