"""Export a small, deployment-safe demo bundle for the Streamlit app.

The bundle is intentionally made from local processed artifacts only. It never
fetches external services and never copies trained model binaries or raw data.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_contract import validate_demo_bundle
from src.config import settings
from src.data_processing.quality import run_quality_checks
from src.data_processing.storage import PartitionedParquetStore
from src.data_sources.ecowatt import EcoWattError, load_cached_ecowatt


MIN_DEMO_DAYS = 7
EMPTY_ECOWATT_COLUMNS = [
    "timestamp",
    "ecowatt_status",
    "ecowatt_label",
    "ecowatt_severity",
    "ecowatt_message",
    "ecowatt_source",
    "ecowatt_dataset_id",
    "ecowatt_source_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=settings.demo_dir)
    parser.add_argument("--days", type=int, default=14, choices=range(7, 15), metavar="[7-14]")
    parser.add_argument("--energy-store", type=Path, default=settings.energy_store_dir)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=settings.processed_dir / "demand_model" / "evaluation.json",
    )
    parser.add_argument("--baseline", type=Path, default=settings.baseline_artifact_path)
    parser.add_argument("--mood", type=Path, default=settings.mood_artifact_path)
    return parser.parse_args()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.copy()
    for column in serializable.select_dtypes(include=["datetimetz", "datetime"]).columns:
        serializable[column] = pd.to_datetime(serializable[column], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    serializable = serializable.where(pd.notna(serializable), None)
    return json.loads(serializable.to_json(orient="records", double_precision=10))


def _load_energy(store: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    try:
        frames.append(PartitionedParquetStore(store).read())
    except (OSError, ValueError):
        pass
    for path in (
        settings.processed_dir / "energy_weather_2024.parquet",
        settings.processed_dir / "energy_weather.parquet",
        settings.processed_dir / "eco2mix_latest.parquet",
    ):
        if path.exists():
            try:
                frames.append(pd.read_parquet(path))
            except (OSError, ValueError):
                pass
    frames = [frame for frame in frames if not frame.empty and "timestamp" in frame]
    if not frames:
        raise FileNotFoundError("No processed energy data is available for a demo export.")
    energy = pd.concat(frames, ignore_index=True, sort=False)
    energy["timestamp"] = pd.to_datetime(energy["timestamp"], utc=True, errors="coerce")
    energy = energy.dropna(subset=["timestamp"])
    if "region" in energy:
        energy = energy.loc[energy["region"].fillna("France").eq("France")].copy()
    keys = ["timestamp", "region"] if "region" in energy else ["timestamp"]
    return energy.sort_values("timestamp").drop_duplicates(keys, keep="last")


def _latest_continuous_window(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    data = frame.sort_values("timestamp").copy()
    gaps = data["timestamp"].diff().gt(pd.Timedelta(hours=2)).fillna(True)
    data["_block_id"] = gaps.cumsum()
    candidates: list[pd.DataFrame] = []
    for _, block in data.groupby("_block_id", sort=False):
        span = block["timestamp"].max() - block["timestamp"].min()
        if span >= pd.Timedelta(days=MIN_DEMO_DAYS):
            candidates.append(block)
    if not candidates:
        raise ValueError(f"No continuous local energy window of at least {MIN_DEMO_DAYS} days was found.")
    selected = max(candidates, key=lambda item: item["timestamp"].max())
    end = selected["timestamp"].max()
    start = end - pd.Timedelta(days=days)
    return selected.loc[selected["timestamp"].between(start, end)].drop(columns=["_block_id"]).copy()


def _export_weather(start: pd.Timestamp, end: pd.Timestamp, output: Path) -> int:
    sources = [
        settings.processed_dir / "weather_national_2024.parquet",
        settings.weather_features_path,
    ]
    for path in sources:
        if not path.exists():
            continue
        weather = pd.read_parquet(path)
        if "timestamp" not in weather:
            continue
        weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
        sample = weather.loc[weather["timestamp"].between(start, end)].copy()
        if not sample.empty:
            sample.to_parquet(output, index=False)
            return len(sample)
    pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}).to_parquet(output, index=False)
    return 0


def _export_ecowatt(start: pd.Timestamp, end: pd.Timestamp, output: Path) -> int:
    try:
        ecowatt = load_cached_ecowatt(timezone_name=settings.timezone)
    except (EcoWattError, FileNotFoundError, OSError, ValueError):
        ecowatt = pd.DataFrame(columns=EMPTY_ECOWATT_COLUMNS)
    if not ecowatt.empty and "timestamp" in ecowatt:
        ecowatt["timestamp"] = pd.to_datetime(ecowatt["timestamp"], utc=True, errors="coerce")
        ecowatt = ecowatt.loc[ecowatt["timestamp"].between(start, end)].copy()
    if ecowatt.empty:
        ecowatt = pd.DataFrame(columns=EMPTY_ECOWATT_COLUMNS)
    ecowatt.to_parquet(output, index=False)
    return len(ecowatt)


def _filter_prediction_payload(payload: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    predictions = pd.DataFrame(payload.get("predictions", []))
    if not predictions.empty and "target_timestamp" in predictions:
        predictions["target_timestamp"] = pd.to_datetime(predictions["target_timestamp"], utc=True, errors="coerce")
        predictions["origin_timestamp"] = pd.to_datetime(
            predictions.get("origin_timestamp"), utc=True, errors="coerce"
        )
        predictions = predictions.loc[predictions["target_timestamp"].between(start, end)].copy()
        predictions = (
            predictions.sort_values(["horizon_hours", "target_timestamp"])
            .groupby("horizon_hours", group_keys=False)
            .tail(800)
        )
    result = dict(payload)
    result["predictions"] = _json_records(predictions) if not predictions.empty else []
    result["demo_bundle"] = {
        "source": "trimmed local evaluation artifact",
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "prediction_rows": len(result["predictions"]),
    }
    return result


def _export_model_forecast(evaluation: dict[str, Any], output: Path) -> int:
    predictions = pd.DataFrame(evaluation.get("predictions", []))
    if predictions.empty:
        _write_json({"schema_version": 1, "forecasts": []}, output)
        return 0
    predictions["origin_timestamp"] = pd.to_datetime(predictions["origin_timestamp"], utc=True, errors="coerce")
    rows = (
        predictions.sort_values(["horizon_hours", "origin_timestamp", "target_timestamp"])
        .groupby("horizon_hours", group_keys=False)
        .tail(1)
        .copy()
    )
    keep = [
        column
        for column in (
            "origin_timestamp",
            "target_timestamp",
            "horizon_hours",
            "model_predicted_mw",
            "model_interval_lower_mw",
            "model_interval_upper_mw",
            "persistence_predicted_mw",
            "day_naive_predicted_mw",
            "week_naive_predicted_mw",
        )
        if column in rows
    ]
    payload = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "source": "trimmed local demand model evaluation",
        "forecasts": _json_records(rows[keep].sort_values("horizon_hours")),
    }
    _write_json(payload, output)
    return len(payload["forecasts"])


def _filter_baseline_payload(payload: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    predictions = pd.DataFrame(payload.get("predictions", []))
    if not predictions.empty and "target" in predictions:
        predictions["target"] = pd.to_datetime(predictions["target"], utc=True, errors="coerce")
        for column in ("origin", "source_timestamp"):
            if column in predictions:
                predictions[column] = pd.to_datetime(predictions[column], utc=True, errors="coerce")
        predictions = predictions.loc[predictions["target"].between(start, end)].copy()
        predictions = predictions.sort_values(["horizon_hours", "baseline", "target"]).tail(5000)
    result = dict(payload)
    result["predictions"] = _json_records(predictions) if not predictions.empty else []
    result["demo_bundle"] = {
        "source": "trimmed local baseline artifact",
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "prediction_rows": len(result["predictions"]),
    }
    return result


def main() -> int:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    energy = _latest_continuous_window(_load_energy(args.energy_store), args.days)
    energy_path = output / "energy_recent.parquet"
    energy.to_parquet(energy_path, index=False)
    start = energy["timestamp"].min()
    end = energy["timestamp"].max()

    weather_rows = _export_weather(start, end, output / "weather_national.parquet")
    ecowatt_rows = _export_ecowatt(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=25), output / "ecowatt.parquet")

    report = run_quality_checks(energy, now=end + pd.Timedelta(minutes=30))
    _write_json(report.to_dict(), output / "quality_report.json")
    if not report.suspicious_rows.empty:
        report.suspicious_rows.to_parquet(output / "quality_suspicious_rows.parquet", index=False)

    evaluation = _filter_prediction_payload(_read_json(args.evaluation), start, end)
    _write_json(evaluation, output / "demand_model_evaluation.json")
    forecast_rows = _export_model_forecast(evaluation, output / "model_forecast.json")

    baseline = _filter_baseline_payload(_read_json(args.baseline), start, end)
    _write_json(baseline, output / "baseline_backtest.json")
    if args.mood.exists():
        shutil.copy2(args.mood, output / "mood_calibration.json")

    manifest = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "app_mode": "demo",
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "artifacts": {
            "energy_recent_rows": len(energy),
            "weather_rows": weather_rows,
            "ecowatt_rows": ecowatt_rows,
            "quality_passed": report.passed,
            "demand_evaluation_prediction_rows": len(evaluation.get("predictions", [])),
            "model_forecast_rows": forecast_rows,
            "baseline_prediction_rows": len(baseline.get("predictions", [])),
        },
    }
    _write_json(manifest, output / "manifest.json")

    checks = validate_demo_bundle(output, log=True)
    blockers = [check for check in checks if check.blocks_demo]
    if blockers:
        details = "; ".join(f"{check.spec.label}: {check.detail}" for check in blockers)
        raise RuntimeError(f"Exported demo bundle failed readiness validation: {details}")
    warnings = [check for check in checks if not check.ok]
    if warnings:
        print("Demo bundle validation warnings: " + "; ".join(f"{check.spec.label}={check.status}" for check in warnings))

    print(
        f"Exported demo bundle to {output} with {len(energy):,} energy rows "
        f"from {start} to {end}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
