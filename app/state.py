from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

import pandas as pd
import streamlit as st

from app.i18n import DEFAULT_LOCALE, LOCALE_QUERY_KEY, LOCALE_SESSION_KEY, normalize_locale
from src.api.current_state import normalize_region_code
from src.contracts.energy_twin import DomainMode


DEFAULT_REGION = "11"
DEFAULT_SCENARIO = "cold_snap"
APP_QUERY_KEYS = ("mode", "region", "t", "run", "scenario", LOCALE_QUERY_KEY)


@dataclass(frozen=True)
class AppState:
    mode: DomainMode
    selected_region: str
    selected_timestamp: datetime | None = None
    selected_forecast_run: str | None = None
    selected_scenario: str | None = DEFAULT_SCENARIO
    locale: str = DEFAULT_LOCALE


def restore_state(
    query_params: Mapping[str, Any],
    session_state: Mapping[str, Any] | None = None,
    *,
    default_mode: DomainMode = DomainMode.REPLAY,
    default_region: str = DEFAULT_REGION,
) -> AppState:
    session_state = session_state or {}
    mode = _mode_value(
        _first(query_params, "mode")
        or session_state.get("app_mode")
        or default_mode.value,
        default=default_mode,
    )
    region = _region_value(
        _first(query_params, "region")
        or session_state.get("selected_region_code")
        or default_region
    )
    timestamp = _timestamp_value(
        _first(query_params, "t")
        or session_state.get("selected_timestamp")
    )
    forecast_run = _text_value(
        _first(query_params, "run")
        or session_state.get("selected_forecast_run")
    )
    scenario = _text_value(
        _first(query_params, "scenario")
        or session_state.get("selected_scenario")
        or DEFAULT_SCENARIO
    )
    locale = normalize_locale(
        _first(query_params, LOCALE_QUERY_KEY)
        or session_state.get(LOCALE_SESSION_KEY)
        or DEFAULT_LOCALE
    )
    return AppState(
        mode=mode,
        selected_region=region,
        selected_timestamp=timestamp,
        selected_forecast_run=forecast_run,
        selected_scenario=scenario,
        locale=locale,
    )


def read_app_state(*, default_mode: DomainMode = DomainMode.REPLAY) -> AppState:
    return restore_state(st.query_params, st.session_state, default_mode=default_mode)


def with_updates(state: AppState, **updates: Any) -> AppState:
    return replace(state, **updates)


def persist_app_state(state: AppState) -> None:
    st.session_state["app_mode"] = state.mode.value
    st.session_state["selected_region_code"] = state.selected_region
    st.session_state["selected_timestamp"] = state.selected_timestamp
    st.session_state["selected_forecast_run"] = state.selected_forecast_run
    st.session_state["selected_scenario"] = state.selected_scenario
    st.session_state[LOCALE_SESSION_KEY] = state.locale
    sync_query_params(state, st.query_params)


def sync_query_params(state: AppState, query_params: MutableMapping[str, Any]) -> None:
    desired = state_to_query_params(state)
    for key in APP_QUERY_KEYS:
        value = desired.get(key)
        if value is None:
            _delete_query_key(query_params, key)
            continue
        if _first(query_params, key) != value:
            query_params[key] = value


def state_to_query_params(state: AppState) -> dict[str, str]:
    result = {
        "mode": state.mode.value,
        "region": state.selected_region,
        LOCALE_QUERY_KEY: state.locale,
    }
    if state.selected_timestamp is not None:
        result["t"] = _utc_iso(state.selected_timestamp)
    if state.selected_forecast_run:
        result["run"] = state.selected_forecast_run
    if state.selected_scenario:
        result["scenario"] = state.selected_scenario
    return result


def mode_for_page(*, replay: bool, page: str) -> DomainMode:
    if replay:
        return DomainMode.REPLAY
    if page == "what_if":
        return DomainMode.SIMULATION
    if page == "next_48h":
        return DomainMode.FORECAST
    return DomainMode.LIVE


def select_timestamp_from_options(
    options: list[pd.Timestamp],
    selected: datetime | None,
) -> int:
    if not options or selected is None:
        return 0
    target = _timestamp_value(selected)
    if target is None:
        return 0
    distances = [abs(pd.Timestamp(option).tz_convert("UTC") - pd.Timestamp(target)) for option in options]
    return int(min(range(len(distances)), key=lambda index: distances[index]))


def forecast_run_id(from_time: datetime | str | None, hours: int) -> str:
    if from_time is None:
        return f"forecast:{hours}h"
    timestamp = _timestamp_value(from_time)
    if timestamp is None:
        return f"forecast:{hours}h"
    return f"forecast:{_utc_iso(timestamp)}:{hours}h"


def _first(values: Mapping[str, Any], key: str) -> str | None:
    if key not in values:
        return None
    value = values[key]
    if isinstance(value, list | tuple):
        return None if not value else str(value[0])
    if value is None:
        return None
    return str(value)


def _delete_query_key(query_params: MutableMapping[str, Any], key: str) -> None:
    try:
        if key in query_params:
            del query_params[key]
    except (KeyError, TypeError):
        return


def _mode_value(value: Any, *, default: DomainMode) -> DomainMode:
    try:
        return value if isinstance(value, DomainMode) else DomainMode(str(value).strip().lower())
    except (TypeError, ValueError):
        return default


def _region_value(value: Any) -> str:
    try:
        return normalize_region_code(str(value))
    except ValueError:
        return DEFAULT_REGION


def _timestamp_value(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.tz_convert("UTC").to_pydatetime()


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_iso(value: datetime) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")
