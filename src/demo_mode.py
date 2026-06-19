from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.artifact_contract import read_json_object
from src.config import settings


_DEMO_SOURCE_END_UTC = pd.Timestamp("2024-12-31T23:30:00Z")
_DEMO_SHIFT_CUTOFF_UTC = pd.Timestamp("2026-01-01T00:00:00Z")


def external_api_enabled() -> bool:
    """Return whether demo mode may call live APIs."""
    return (not settings.is_demo_mode) or settings.demo_allow_external_api


def demo_time_shift() -> pd.Timedelta:
    """Move the historical demo window onto the presentation date."""
    anchor = pd.Timestamp(settings.demo_anchor_end_utc)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    return anchor.tz_convert("UTC") - _DEMO_SOURCE_END_UTC


def mode_badge_color() -> str:
    return "grey" if settings.is_demo_mode else "blue"


def read_demo_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
    frame = _shift_demo_frame_dates(frame)
    return frame


def read_demo_json(path: Path) -> dict[str, Any]:
    payload, error = read_json_object(path)
    if error:
        return {}
    return _shift_demo_json_dates(payload)


def _shift_demo_frame_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    offset = demo_time_shift()
    shifted = frame.copy()
    for column in shifted.columns:
        lowered = str(column).lower()
        if "timestamp" not in lowered and "date" not in lowered and "time" not in lowered:
            continue
        values = pd.to_datetime(shifted[column], utc=True, errors="coerce")
        if not values.notna().any():
            continue
        mask = values.notna() & values.lt(_DEMO_SHIFT_CUTOFF_UTC)
        shifted.loc[mask, column] = values.loc[mask] + offset
    if "timestamp" in shifted:
        timestamp = pd.to_datetime(shifted["timestamp"], utc=True, errors="coerce")
        local = timestamp.dt.tz_convert(settings.timezone)
        if "hour" in shifted:
            shifted["hour"] = local.dt.hour
        if "day_of_week" in shifted:
            shifted["day_of_week"] = local.dt.dayofweek
        if "month" in shifted:
            shifted["month"] = local.dt.month
        if "is_weekend" in shifted:
            shifted["is_weekend"] = local.dt.dayofweek.ge(5)
    return shifted


def _shift_demo_json_dates(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: _shift_demo_json_dates(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_shift_demo_json_dates(item, key) for item in value]
    if isinstance(value, str):
        return _shift_demo_timestamp_string(value, key=key)
    return value


def _shift_demo_timestamp_string(value: str, *, key: str | None = None) -> str:
    if len(value) < 10 or value[4:5] != "-" or value[7:8] != "-":
        return value
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(timestamp):
        return value
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    timestamp = timestamp.tz_convert("UTC")
    if str(key or "").lower() == "generated_at":
        anchor = pd.Timestamp(settings.demo_anchor_end_utc)
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize("UTC")
        return _format_shifted_timestamp(anchor.tz_convert("UTC"), original=value)
    if timestamp >= _DEMO_SHIFT_CUTOFF_UTC:
        return value
    shifted = timestamp + demo_time_shift()
    return _format_shifted_timestamp(shifted, original=value)


def _format_shifted_timestamp(timestamp: pd.Timestamp, *, original: str) -> str:
    if original.endswith("Z"):
        return timestamp.isoformat().replace("+00:00", "Z")
    return timestamp.isoformat()


def demo_energy() -> pd.DataFrame:
    return read_demo_parquet(settings.demo_energy_path)


def demo_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    weather = read_demo_parquet(settings.demo_weather_path)
    if weather.empty or "timestamp" not in weather:
        return weather
    return weather.loc[weather["timestamp"].between(start, end)].copy()


def demo_ecowatt(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    ecowatt = read_demo_parquet(settings.demo_ecowatt_path)
    if ecowatt.empty or "timestamp" not in ecowatt:
        return pd.DataFrame(), "Demo EcoWatt sample unavailable"
    frame = ecowatt.loc[ecowatt["timestamp"].between(start, end)].copy()
    if frame.empty:
        return frame, "Demo EcoWatt sample unavailable for this window"
    return frame, "Demo EcoWatt sample"


def demo_model_evaluation() -> dict[str, Any]:
    return read_demo_json(settings.demo_model_evaluation_path)


def demo_mood_artifact() -> dict[str, Any]:
    return read_demo_json(settings.demo_mood_artifact_path)


def demo_quality_report() -> dict[str, Any]:
    return read_demo_json(settings.demo_quality_path)
