from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.artifact_contract import read_json_object
from src.config import settings

LOGGER = logging.getLogger(__name__)


def external_api_enabled() -> bool:
    """Return whether demo mode may call live APIs."""
    return (not settings.is_demo_mode) or settings.demo_allow_external_api


def mode_badge_color() -> str:
    return "grey" if settings.is_demo_mode else "blue"


def read_demo_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Demo parquet artifact is missing: %s", path)
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        LOGGER.warning("Demo parquet artifact is unreadable: %s (%s)", path, exc)
        return pd.DataFrame()
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
    return frame


def read_demo_json(path: Path) -> dict[str, Any]:
    payload, error = read_json_object(path)
    if error:
        LOGGER.warning("Demo JSON artifact unavailable: %s (%s)", path, error)
        return {}
    return payload


def demo_energy() -> pd.DataFrame:
    return read_demo_parquet(settings.demo_energy_path)


def demo_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    weather = read_demo_parquet(settings.demo_weather_path)
    if weather.empty or "timestamp" not in weather:
        return weather
    return weather.loc[weather["timestamp"].between(start, end)].copy()


def demo_ecowatt(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    ecowatt = read_demo_parquet(settings.demo_ecowatt_path)
    if ecowatt.empty:
        message = (
            "Demo EcoWatt artifact is present but contains 0 rows"
            if settings.demo_ecowatt_path.exists()
            else "Demo EcoWatt sample unavailable"
        )
        return pd.DataFrame(), message
    if "timestamp" not in ecowatt:
        LOGGER.warning("Demo EcoWatt artifact has no timestamp column: %s", settings.demo_ecowatt_path)
        return pd.DataFrame(), "Demo EcoWatt artifact is malformed"
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
