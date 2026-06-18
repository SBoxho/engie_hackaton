"""Backfill multi-year Energy Pulse demand, weather, calendar, and model artifacts."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.data_processing.features import add_time_features
from src.data_processing.storage import PartitionedParquetStore, read_processed_data
from src.data_processing.weather_features import build_national_weather_features, join_energy_weather
from src.data_sources.rte_eco2mix_historical import fetch_historical
from src.data_sources.school_calendar import fetch_school_calendar
from src.data_sources.weather_national import fetch_national_weather, load_city_reference
from src.models.demand_model import (
    FeatureConfig,
    TrainConfig,
    build_feature_frame,
    evaluate_models,
    save_evaluation,
    save_feature_metadata,
    save_model_bundle,
    train_models,
)
from src.utils.io import write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True, help="First calendar year, inclusive")
    parser.add_argument("--end-year", type=int, required=True, help="Last calendar year, inclusive")
    parser.add_argument("--energy-store", type=Path, default=settings.energy_store_dir)
    parser.add_argument("--weather-output", type=Path, default=settings.weather_features_path)
    parser.add_argument("--joined-output", type=Path, default=settings.joined_features_path)
    parser.add_argument("--school-calendar-output", type=Path, default=settings.school_calendar_path)
    parser.add_argument("--features-output", type=Path, default=settings.processed_dir / "demand_model" / "features.parquet")
    parser.add_argument("--metadata-output", type=Path, default=settings.processed_dir / "demand_model" / "feature_metadata.json")
    parser.add_argument("--model-output", type=Path, default=settings.processed_dir / "demand_model" / "demand_hgb_model.pkl")
    parser.add_argument("--evaluation-output", type=Path, default=settings.processed_dir / "demand_model" / "evaluation.json")
    parser.add_argument("--weather-cache-dir", type=Path, default=settings.raw_dir / "weather_national")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-school-calendar", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Do not write immutable ODRÉ raw snapshots")
    parser.add_argument("--force-weather-refresh", action="store_true")
    parser.add_argument("--strict-weather", action="store_true")
    parser.add_argument("--train", action="store_true", help="Train after feature generation")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training/loading the model")
    parser.add_argument("--min-continuous-hours", type=float, default=168.0)
    parser.add_argument("--cadence-minutes", type=int, help="Override inferred demand cadence")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--min-train-samples", type=int, default=96)
    parser.add_argument("--min-test-samples", type=int, default=24)
    parser.add_argument("--min-segment-samples", type=int, default=24)
    parser.add_argument(
        "--fail-on-readiness-warning",
        action="store_true",
        help="Exit non-zero after evaluation if history/weather readiness checks do not pass.",
    )
    return parser.parse_args()


def _date_range(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, date, date]:
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be less than or equal to --end-year")
    start = pd.Timestamp(year=args.start_year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=args.end_year + 1, month=1, day=1, tz="UTC")
    return start, end, start.date(), (end - pd.Timedelta(days=1)).date()


def _fetch_energy_years(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> int:
    store = PartitionedParquetStore(args.energy_store)
    total_rows = 0
    for year in range(args.start_year, args.end_year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        year_end = min(pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"), end)
        frame = fetch_historical(
            year_start.to_pydatetime(),
            year_end.to_pydatetime(),
            cache=not args.no_cache,
        )
        if not frame.empty:
            frame = add_time_features(frame, settings.timezone)
            store.upsert(frame)
        total_rows += len(frame)
        print(f"Energy {year}: {len(frame):,} clean rows")
    return total_rows


def _build_weather(args: argparse.Namespace, start_date: date, end_date: date, start: pd.Timestamp, end: pd.Timestamp) -> int:
    cities, _ = load_city_reference()
    energy = read_processed_data(args.energy_store, start=start, end=end, regions=["France"])
    raw = fetch_national_weather(
        start_date,
        end_date,
        cities=cities,
        cache_dir=args.weather_cache_dir,
        force_refresh=args.force_weather_refresh,
        strict=args.strict_weather,
    )
    if energy.empty:
        targets = pd.date_range(start, end - pd.Timedelta(minutes=15), freq="15min")
    else:
        targets = pd.DatetimeIndex(pd.to_datetime(energy["timestamp"], utc=True)).drop_duplicates().sort_values()
    weather = build_national_weather_features(raw, targets, cities=cities)
    write_dataframe(weather, args.weather_output)
    if not energy.empty:
        joined = join_energy_weather(energy, weather)
        write_dataframe(joined, args.joined_output)
    return len(weather)


def _build_school_calendar(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    if args.skip_school_calendar:
        return None
    calendar = fetch_school_calendar(start.date(), end.date(), cache=not args.no_cache)
    write_dataframe(calendar, args.school_calendar_output)
    return calendar


def _build_features(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp, school_calendar: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    energy = read_processed_data(args.energy_store, start=start, end=end, regions=["France"])
    if energy.empty:
        raise SystemExit("No energy rows were available after backfill; cannot build features.")
    weather = pd.read_parquet(args.weather_output) if args.weather_output.exists() else None
    if school_calendar is None and args.school_calendar_output.exists():
        school_calendar = pd.read_parquet(args.school_calendar_output)
    features, metadata = build_feature_frame(
        energy,
        weather=weather,
        school_calendar=school_calendar,
        config=FeatureConfig(
            min_continuous_hours=args.min_continuous_hours,
            cadence_minutes=args.cadence_minutes,
        ),
        source=f"multi-year-backfill:{args.start_year}-{args.end_year}",
    )
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.features_output, index=False)
    save_feature_metadata(metadata, args.metadata_output)
    return features, metadata


def main() -> int:
    args = parse_args()
    start, end, start_date, end_date = _date_range(args)
    energy_rows = _fetch_energy_years(args, start, end)
    weather_rows = 0
    if not args.skip_weather:
        weather_rows = _build_weather(args, start_date, end_date, start, end)
    school_calendar = _build_school_calendar(args, start, end)
    features = pd.DataFrame()
    metadata: dict = {}
    if not args.skip_features:
        features, metadata = _build_features(args, start, end, school_calendar)
    bundle = None
    if args.train:
        if features.empty:
            features = pd.read_parquet(args.features_output)
        if not metadata:
            from src.models.demand_model import load_feature_metadata

            metadata = load_feature_metadata(args.metadata_output)
        bundle = train_models(
            features,
            metadata,
            config=TrainConfig(
                test_fraction=args.test_fraction,
                validation_folds=args.validation_folds,
                min_train_samples=args.min_train_samples,
                min_test_samples=args.min_test_samples,
            ),
        )
        save_model_bundle(bundle, args.model_output)
    if args.evaluate:
        if features.empty:
            features = pd.read_parquet(args.features_output)
        if bundle is None:
            from src.models.demand_model import load_model_bundle

            bundle = load_model_bundle(args.model_output)
        evaluation = evaluate_models(features, bundle, min_segment_samples=args.min_segment_samples)
        save_evaluation(evaluation, args.evaluation_output)
        failed = [check for check in evaluation.get("readiness", {}).get("checks", []) if not check.get("passed")]
        if failed:
            details = ", ".join(
                f"{check['name']} observed={check.get('observed')} threshold={check.get('threshold')}"
                for check in failed
            )
            print(f"Readiness warnings: {details}")
            if args.fail_on_readiness_warning:
                raise SystemExit(2)
    print(
        f"Backfilled {args.start_year}-{args.end_year}: energy_rows={energy_rows:,}, "
        f"weather_rows={weather_rows:,}, features={len(features):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
