"""Build hourly analytical and supervised usual-demand feature datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models.usual_demand import (
    DEFAULT_HORIZONS_HOURS,
    build_model_ready_dataset,
    combine_public_inputs,
    read_public_frame,
    write_json,
)
from src.public_data.storage import PublicDataStore


DEFAULT_OUTPUT_DIR = settings.processed_dir / "usual_demand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Combined normalized public-data parquet/CSV/JSON table")
    source.add_argument("--public-root", type=Path, default=PROJECT_ROOT / "data" / "public")
    parser.add_argument("--energy-input", type=Path, help="Optional energy table to combine with --weather-input")
    parser.add_argument("--weather-input", type=Path, help="Optional weather table to combine with energy input")
    parser.add_argument("--public-holidays-input", type=Path, help="Optional French public-holidays table")
    parser.add_argument("--school-holidays-input", type=Path, help="Optional French school-holidays table")
    parser.add_argument("--start", help="Inclusive UTC read boundary for public store")
    parser.add_argument("--end", help="Exclusive UTC read boundary for public store")
    parser.add_argument("--timezone", default=settings.timezone)
    parser.add_argument("--horizon", type=int, action="append", dest="horizons", help="Forecast horizon in hours; repeatable")
    parser.add_argument("--hourly-output", type=Path, default=DEFAULT_OUTPUT_DIR / "hourly_features.parquet")
    parser.add_argument("--training-output", type=Path, default=DEFAULT_OUTPUT_DIR / "baseline_training.parquet")
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_OUTPUT_DIR / "feature_manifest.json")
    parser.add_argument("--quality-output", type=Path, default=DEFAULT_OUTPUT_DIR / "data_quality_report.json")
    parser.add_argument("--coverage-output", type=Path, default=DEFAULT_OUTPUT_DIR / "feature_coverage_report.json")
    return parser.parse_args()


def _read_optional(path: Path | None):
    return read_public_frame(path) if path is not None and path.exists() else None


def _load_records(args: argparse.Namespace):
    if args.input is not None:
        combined = read_public_frame(args.input)
    elif args.energy_input is None and args.weather_input is None and args.public_holidays_input is None and args.school_holidays_input is None:
        combined = PublicDataStore(args.public_root, layer="silver").read(start=args.start, end=args.end)
    else:
        combined = None
    return combine_public_inputs(
        combined=combined,
        energy=_read_optional(args.energy_input),
        weather=_read_optional(args.weather_input),
        public_holidays=_read_optional(args.public_holidays_input),
        school_holidays=_read_optional(args.school_holidays_input),
    )


def main() -> int:
    args = parse_args()
    horizons = tuple(args.horizons or DEFAULT_HORIZONS_HOURS)
    records = _load_records(args)
    if records.empty:
        raise SystemExit("No public records found. Ingest data first or provide --input/--energy-input.")
    result = build_model_ready_dataset(records, horizons_hours=horizons, timezone=args.timezone)

    args.hourly_output.parent.mkdir(parents=True, exist_ok=True)
    result.hourly.to_parquet(args.hourly_output, index=False)
    result.supervised.to_parquet(args.training_output, index=False)
    write_json(result.feature_manifest, args.manifest_output)
    write_json(result.quality_report, args.quality_output)
    write_json(result.coverage_report, args.coverage_output)

    print(
        "Wrote "
        f"{len(result.hourly):,} hourly rows to {args.hourly_output}; "
        f"{len(result.supervised):,} supervised rows to {args.training_output}; "
        f"horizons={list(horizons)}"
    )
    print(f"Feature manifest: {args.manifest_output}")
    print(f"Quality report: {args.quality_output}")
    print(f"Coverage report: {args.coverage_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
