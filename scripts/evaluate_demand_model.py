"""Evaluate demand model against baselines on the same chronological test origins."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models.demand_model import evaluate_models, load_model_bundle, save_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=settings.processed_dir / "demand_model" / "features.parquet")
    parser.add_argument("--model", type=Path, default=settings.processed_dir / "demand_model" / "demand_hgb_model.pkl")
    parser.add_argument("--output", type=Path, default=settings.processed_dir / "demand_model" / "evaluation.json")
    parser.add_argument("--min-segment-samples", type=int, default=24)
    parser.add_argument(
        "--fail-on-readiness-warning",
        action="store_true",
        help="Exit non-zero when minimum history/weather checks do not pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features = pd.read_parquet(args.features)
    bundle = load_model_bundle(args.model)
    evaluation = evaluate_models(features, bundle, min_segment_samples=args.min_segment_samples)
    save_evaluation(evaluation, args.output)
    print(
        f"Wrote {len(evaluation['predictions']):,} model predictions and "
        f"{len(evaluation['metrics'])} metric rows to {args.output}"
    )
    readiness = evaluation.get("readiness", {})
    failed = [check for check in readiness.get("checks", []) if not check.get("passed")]
    if failed:
        details = ", ".join(
            f"{check['name']} observed={check.get('observed')} threshold={check.get('threshold')}"
            for check in failed
        )
        print(f"Readiness warnings: {details}")
        if args.fail_on_readiness_warning:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
