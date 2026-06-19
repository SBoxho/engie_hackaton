from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import streamlit as st

from app.view_models import synthesize_regional_history
from src.config import settings
from src.data_processing.clean_energy_mix import clean_energy_mix
from src.data_processing.features import add_time_features
from src.data_processing.storage import PartitionedParquetStore
from src.data_sources.ecowatt import load_ecowatt_window
from src.data_sources.rte_eco2mix import Eco2MixError, fetch_eco2mix, load_cached_eco2mix
from src.data_sources.rte_eco2mix_regional import (
    REGION_NAMES,
    RegionalEco2MixError,
    demo_regional_snapshot,
    fallback_region_geojson,
    fetch_regional_eco2mix,
    load_cached_regional_eco2mix,
    load_region_geojson,
    prepare_regional_snapshot,
    region_code,
)
from src.demo_mode import (
    demo_ecowatt,
    demo_energy,
    demo_model_evaluation,
    demo_mood_artifact,
    external_api_enabled,
    read_demo_parquet,
)

REGIONAL_CONTEXT_CACHE_VERSION = 3
REPLAY_TIMEBASE_CACHE_VERSION = 2


def _mode_for_source(*, live: bool) -> str:
    return "LIVE" if live else "REPLAY"


def _replay_anchor() -> pd.Timestamp:
    demo = demo_energy()
    if not demo.empty and "timestamp" in demo:
        latest = pd.to_datetime(demo["timestamp"], utc=True, errors="coerce").dropna()
        if not latest.empty:
            return pd.Timestamp(latest.max()).floor("15min")
    return pd.Timestamp.now(tz="UTC").floor("15min")


@st.cache_data(ttl=900, show_spinner=False)
def load_national_energy(
    hours: int,
    replay_timebase_version: int = REPLAY_TIMEBASE_CACHE_VERSION,
) -> tuple[pd.DataFrame, str, str]:
    _ = replay_timebase_version
    if settings.is_demo_mode and not external_api_enabled():
        demo = demo_energy()
        if not demo.empty:
            return demo.sort_values("timestamp"), "Bundled demo replay sample", _mode_for_source(live=False)
        return pd.DataFrame(), "Historical sample unavailable", _mode_for_source(live=False)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        raw = fetch_eco2mix(start=start, end=end)
        clean = add_time_features(clean_energy_mix(raw), settings.timezone)
        PartitionedParquetStore(settings.energy_store_dir).upsert(clean)
        return clean.sort_values("timestamp"), "RTE eco2mix refreshed", _mode_for_source(live=True)
    except (Eco2MixError, OSError, ValueError):
        try:
            stored = PartitionedParquetStore(settings.energy_store_dir).read(start=start, end=end)
            if not stored.empty:
                return stored.sort_values("timestamp"), "RTE eco2mix cached snapshot", _mode_for_source(live=False)
        except (OSError, ValueError):
            pass
        raw = load_cached_eco2mix()
        return clean_energy_mix(raw).sort_values("timestamp"), "RTE eco2mix cached snapshot", _mode_for_source(live=False)


@st.cache_data(ttl=900, show_spinner=False)
def load_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    replay_timebase_version: int = REPLAY_TIMEBASE_CACHE_VERSION,
) -> pd.DataFrame:
    _ = replay_timebase_version
    if settings.is_demo_mode and not external_api_enabled():
        weather = read_demo_parquet(settings.demo_weather_path)
        if weather.empty:
            return weather
        return weather.loc[weather["timestamp"].between(start, end)].copy()
    if not settings.weather_features_path.exists():
        return pd.DataFrame()
    weather = pd.read_parquet(settings.weather_features_path)
    if "timestamp" not in weather:
        return pd.DataFrame()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
    return weather.loc[weather["timestamp"].between(start, end)].copy()


@st.cache_data(ttl=900, show_spinner=False)
def load_ecowatt(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    if settings.is_demo_mode and not external_api_enabled():
        return demo_ecowatt(start, end)
    return load_ecowatt_window(start, end, timezone_name=settings.timezone)


@st.cache_data(ttl=900, show_spinner=False)
def load_model_evaluation() -> dict[str, Any]:
    if settings.is_demo_mode:
        return demo_model_evaluation()
    path = settings.processed_dir / "demand_model" / "evaluation.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=900, show_spinner=False)
def load_mood_artifact() -> dict[str, Any]:
    if settings.is_demo_mode:
        return demo_mood_artifact()
    if not settings.mood_artifact_path.exists():
        return {}
    try:
        payload = json.loads(settings.mood_artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _regional_history_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    clean = clean_energy_mix(raw)
    if "region" not in clean:
        return pd.DataFrame()
    clean["region_code"] = clean["region"].map(region_code)
    clean = clean.dropna(subset=["region_code"]).copy()
    if clean.empty:
        return clean
    clean["region_code"] = clean["region_code"].astype(str)
    clean["region_display"] = clean["region_code"].map(REGION_NAMES).fillna(clean["region"])
    return clean.sort_values(["region_code", "timestamp"]).reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_regional_energy(
    hours: int,
    cache_version: int = REGIONAL_CONTEXT_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    _ = cache_version
    if settings.is_demo_mode and not external_api_enabled():
        snapshot = demo_regional_snapshot(_replay_anchor().floor("h"))
        history = synthesize_regional_history(snapshot, timezone=settings.timezone)
        return snapshot, history, "Regional historical sample", _mode_for_source(live=False)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        raw = fetch_regional_eco2mix(start=start, end=end)
        snapshot = prepare_regional_snapshot(raw)
        history = _regional_history_from_raw(raw)
        return snapshot, history, "RTE regional eco2mix refreshed", _mode_for_source(live=True)
    except (RegionalEco2MixError, OSError, ValueError):
        try:
            raw = load_cached_regional_eco2mix()
            snapshot = prepare_regional_snapshot(raw)
            history = _regional_history_from_raw(raw)
            return snapshot, history, "RTE regional eco2mix cached snapshot", _mode_for_source(live=False)
        except (RegionalEco2MixError, FileNotFoundError, ValueError, OSError):
            snapshot = demo_regional_snapshot()
            history = synthesize_regional_history(snapshot, timezone=settings.timezone)
            return snapshot, history, "Regional historical sample", _mode_for_source(live=False)


@st.cache_data(ttl=86400, show_spinner=False)
def load_regions_geojson() -> tuple[dict[str, Any], str]:
    if settings.is_demo_mode and not external_api_enabled():
        return fallback_region_geojson(), "Bundled regional boundaries"
    try:
        return load_region_geojson(), "French administrative regional boundaries"
    except (RegionalEco2MixError, AttributeError, TypeError, ValueError, OSError):
        return fallback_region_geojson(), "Bundled regional boundaries"


def load_public_context() -> dict[str, Any]:
    energy, national_source, national_mode = load_national_energy(settings.history_hours)
    if energy.empty:
        return {
            "energy": energy,
            "national_source": national_source,
            "mode": national_mode,
            "weather": pd.DataFrame(),
            "ecowatt": pd.DataFrame(),
            "ecowatt_source": "Unavailable",
            "model_payload": {},
            "regional": pd.DataFrame(),
            "regional_history": pd.DataFrame(),
            "regions_geojson": {"type": "FeatureCollection", "features": []},
            "regional_source": "Unavailable",
        }

    start = pd.to_datetime(energy["timestamp"].min(), utc=True)
    end = pd.to_datetime(energy["timestamp"].max(), utc=True)
    weather = load_weather(start, end)
    ecowatt_start = end.floor("h") - pd.Timedelta(hours=1)
    ecowatt_end = end.floor("h") + pd.Timedelta(hours=49)
    ecowatt, ecowatt_source = load_ecowatt(ecowatt_start, ecowatt_end)
    regional, regional_history, regional_source, regional_mode = load_regional_energy(settings.history_hours)
    regions_geojson, geo_source = load_regions_geojson()
    mode = "LIVE" if national_mode == "LIVE" and regional_mode == "LIVE" else "REPLAY"
    return {
        "energy": energy,
        "national_source": national_source,
        "mode": mode,
        "weather": weather,
        "ecowatt": ecowatt,
        "ecowatt_source": ecowatt_source,
        "model_payload": load_model_evaluation(),
        "mood_artifact": load_mood_artifact(),
        "regional": regional,
        "regional_history": regional_history,
        "regions_geojson": regions_geojson,
        "regional_source": regional_source,
        "geo_source": geo_source,
    }
