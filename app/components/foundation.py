from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal

import streamlit as st

from app.formatting import format_age_seconds, format_timestamp
from app.state import AppState
from src.contracts.energy_twin import (
    CurrentStateResponse,
    DataProvenance,
    DomainMode,
    Freshness,
    OperatingState,
    SourceType,
    TwinResponse,
)
from src.data_sources.rte_eco2mix_regional import REGION_NAMES


PROVENANCE_DEFINITIONS = {
    "official": "Official public source or official forecast.",
    "observed": "Observed public measurement.",
    "model": "Model estimate from the Energy Pulse analytical layer.",
    "modelled": "App balance context from the analytical layer.",
    "scenario": "Scenario output from user-selected assumptions.",
    "fallback": "Fallback route used because preferred data was unavailable.",
    "replay": "Historical replay or demo fixture, not live data.",
    "unavailable": "The source or field is unavailable and has not been replaced with zero.",
}

TERM_DEFINITIONS = {
    "usual demand": "A comparable-history baseline for the same season, day type, and local hour.",
    "local generation": "Electricity generated in the selected region; the region remains part of the connected French grid.",
    "likely range": "The p10 to p90 forecast interval. It describes uncertainty and is not used as a balance-pressure input.",
    "modelled balance context": "A documented analytical pressure context, separate from official EcoWatt and not an operational reserve margin.",
}

StateKind = Literal["loading", "empty", "stale", "fallback", "partial", "error"]


@dataclass(frozen=True)
class TrustNotice:
    kind: StateKind
    title: str
    body: str


def provenance_badge_html(kind: str, label: str | None = None) -> str:
    normalized = _normalize_provenance_kind(kind)
    if normalized == "modelled":
        return ""
    text = label or _provenance_label(normalized)
    definition = PROVENANCE_DEFINITIONS[normalized]
    return (
        f'<span class="ep-provenance ep-provenance-{normalized}" role="status" '
        f'aria-label="Provenance: {html.escape(text)}. {html.escape(definition)}">'
        f'<span class="ep-provenance-key">{html.escape(text)}</span>'
        "</span>"
    )


def render_provenance_badges(kinds: Iterable[str]) -> None:
    badges = "".join(provenance_badge_html(kind) for kind in _unique(kinds))
    if badges:
        st.markdown(f'<div class="ep-provenance-row">{badges}</div>', unsafe_allow_html=True)


def provenance_kinds_from_twin(twin: TwinResponse | None) -> list[str]:
    if twin is None or not twin.snapshots:
        return ["fallback"]
    kinds: list[str] = []
    first = twin.snapshots[0]
    for source in first.provenance_chain or [first.source]:
        kinds.append(provenance_kind_for_source(source))
    if first.mode == DomainMode.REPLAY:
        kinds.append("replay")
    return _unique(kinds)


def provenance_kind_for_source(source: DataProvenance) -> str:
    if source.mode == DomainMode.REPLAY or source.is_demo:
        return "replay"
    if source.is_fallback or source.source_type == SourceType.FALLBACK:
        return "fallback"
    if source.source_type == SourceType.OFFICIAL:
        return "official"
    if source.source_type == SourceType.OBSERVED:
        return "observed"
    if source.source_type == SourceType.SCENARIO:
        return "scenario"
    return "model"


def status_text_html(label: str, detail: str, *, status: str = "info") -> str:
    return (
        f'<span class="ep-status-text ep-status-text-{html.escape(status)}" role="status" '
        f'aria-label="Status: {html.escape(label)}. {html.escape(detail)}">'
        f'<strong>{html.escape(label)}</strong><span>{html.escape(detail)}</span></span>'
    )


def trust_notice_html(kind: StateKind, title: str, body: str) -> str:
    role = "alert" if kind == "error" else "status"
    return (
        f'<div class="ep-trust-state ep-trust-{kind}" role="{role}" '
        f'aria-label="{html.escape(title)}: {html.escape(body)}">'
        f'<div class="ep-trust-label">{html.escape(_state_label(kind))}</div>'
        f'<div><div class="ep-trust-title">{html.escape(title)}</div>'
        f'<div class="ep-trust-body">{html.escape(body)}</div></div></div>'
    )


def render_trust_notices(notices: Iterable[TrustNotice]) -> None:
    markup = "".join(trust_notice_html(item.kind, item.title, item.body) for item in notices)
    if markup:
        st.markdown(f'<div class="ep-trust-stack">{markup}</div>', unsafe_allow_html=True)


def loading_state(title: str = "Loading data", body: str = "Retrieving the latest typed contract response.") -> None:
    st.markdown(trust_notice_html("loading", title, body), unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(trust_notice_html("empty", title, body), unsafe_allow_html=True)


def error_state(title: str, body: str) -> None:
    st.markdown(trust_notice_html("error", title, body), unsafe_allow_html=True)


def term_tooltip_html(term: str) -> str:
    key = term.strip().lower()
    definition = TERM_DEFINITIONS.get(key, "Domain term used by this dashboard.")
    return (
        f'<span class="ep-term" tabindex="0" role="note" '
        f'aria-label="{html.escape(term)}: {html.escape(definition)}">'
        f'<abbr title="{html.escape(definition)}">{html.escape(term)}</abbr>'
        f'<span class="ep-term-popover" aria-hidden="true">{html.escape(definition)}</span></span>'
    )


def render_context_bar(
    state: AppState,
    *,
    twin: TwinResponse | None,
    current_state: CurrentStateResponse | None,
    timezone_name: str,
) -> None:
    last_update = _last_update(twin, current_state)
    selected_time = state.selected_timestamp or _selected_time(twin, current_state)
    mode_label = _mode_label(state.mode)
    scope = REGION_NAMES.get(state.selected_region, state.selected_region)
    provenance_kinds = provenance_kinds_from_twin(twin) if twin is not None else provenance_kinds_from_current_state(current_state)
    badges = "".join(provenance_badge_html(kind) for kind in provenance_kinds)
    age = _age_text(current_state)
    st.markdown(
        f"""
        <div class="ep-context-bar" aria-label="Application context">
          <div class="ep-context-main">
            <div class="ep-context-mode">{html.escape(mode_label)}</div>
          </div>
          <div class="ep-context-item"><span>Scope</span><strong>{html.escape(scope)}</strong></div>
          <div class="ep-context-item"><span>Selected time</span><strong>{html.escape(format_timestamp(selected_time, timezone_name=timezone_name))}</strong></div>
          <div class="ep-context-item"><span>Last update</span><strong>{html.escape(format_timestamp(last_update, timezone_name=timezone_name))}</strong><small>{html.escape(age)}</small></div>
          <div class="ep-context-badges">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notices_from_contracts(twin: TwinResponse | None, current_state: CurrentStateResponse | None) -> list[TrustNotice]:
    notices: list[TrustNotice] = []
    if twin is None and current_state is None:
        notices.append(
            TrustNotice(
                "fallback",
                "Typed twin contract unavailable",
                "The page is using the legacy public context while the typed response is unavailable.",
            )
        )
    elif twin is not None and twin.snapshots and twin.snapshots[0].mode == DomainMode.REPLAY:
        notices.append(
            TrustNotice(
                "fallback",
                "Demo fixture mode",
                "This is contract-valid historical sample data and must not be read as live operations.",
            )
        )
    if current_state is not None and current_state.operating_state == OperatingState.HISTORICAL_REPLAY:
        notices.append(
            TrustNotice(
                "fallback",
                "Demo fixture mode",
                "This is contract-valid historical sample data and must not be read as live operations.",
            )
        )
    if current_state is not None and current_state.unavailable_fields:
        notices.append(
            TrustNotice(
                "partial",
                "Partial data",
                f"{len(current_state.unavailable_fields)} field(s) are unavailable and labelled in the typed contract.",
            )
        )
    if current_state is not None and current_state.operating_state == OperatingState.DELAYED_LIVE_DATA:
        notices.append(
            TrustNotice(
                "stale",
                "Delayed source data",
                "The current-state API marks the latest source record as delayed.",
            )
        )
    if current_state is not None and current_state.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK:
        notices.append(
            TrustNotice(
                "fallback",
                "Using last-known-good data",
                "Live refresh did not provide a fresh complete source bundle.",
            )
        )
    if twin is not None and twin.unavailable_fields:
        notices.append(
            TrustNotice(
                "partial",
                "Optional source unavailable",
                f"{len(twin.unavailable_fields)} optional source field(s) are unavailable.",
            )
        )
    freshness = twin.snapshots[0].quality.freshness if twin and twin.snapshots else None
    if freshness == Freshness.STALE:
        notices.append(
            TrustNotice(
                "stale",
                "Stale data",
                "The latest typed contract marks this data as stale.",
            )
        )
    return notices


def responsive_columns(kind: str = "content"):
    if kind == "content":
        return st.columns([1.55, 1], gap="large")
    if kind == "map-detail":
        return st.columns([1.62, 1], gap="large")
    if kind == "metrics":
        return st.columns(4)
    return st.columns(2, gap="large")


def _last_update(twin: TwinResponse | None, current_state: CurrentStateResponse | None) -> datetime | str | None:
    if twin is not None:
        return twin.generated_at
    if current_state is not None:
        return current_state.generated_at
    return None


def _selected_time(twin: TwinResponse | None, current_state: CurrentStateResponse | None) -> datetime | str | None:
    if twin is not None:
        return twin.from_time
    if current_state is not None:
        return current_state.national_context.freshness.timestamp
    return None


def _age_text(current_state: CurrentStateResponse | None) -> str:
    if current_state is None:
        return ""
    return format_age_seconds(current_state.national_context.freshness.age_seconds)


def provenance_kinds_from_current_state(current_state: CurrentStateResponse | None) -> list[str]:
    if current_state is None:
        return ["fallback"]
    if current_state.operating_state == OperatingState.HISTORICAL_REPLAY:
        return ["replay"]
    if current_state.operating_state == OperatingState.SOURCE_UNAVAILABLE:
        return ["unavailable"]
    if current_state.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK:
        return ["fallback"]
    return ["observed"]


def _mode_label(mode: DomainMode) -> str:
    return {
        DomainMode.LIVE: "Live",
        DomainMode.FORECAST: "Forecast",
        DomainMode.SIMULATION: "Simulation",
        DomainMode.REPLAY: "Demo context",
    }.get(mode, mode.value.title())


def _normalize_provenance_kind(kind: str) -> str:
    lowered = str(kind).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "model-estimate": "model",
        "observed-data": "observed",
        "demo": "replay",
        "modelled-estimate": "modelled",
        "unavailable-data": "unavailable",
    }
    normalized = aliases.get(lowered, lowered)
    if normalized not in PROVENANCE_DEFINITIONS:
        return "model"
    return normalized


def _provenance_label(kind: str) -> str:
    return {
        "official": "Official",
        "observed": "Observed",
        "model": "Model estimate",
        "modelled": "Balance context",
        "scenario": "Scenario",
        "fallback": "Fallback",
        "replay": "Replay",
        "unavailable": "Unavailable",
    }[kind]


def _state_label(kind: StateKind) -> str:
    return {
        "loading": "Loading",
        "empty": "Empty",
        "stale": "Stale data",
        "fallback": "Fallback",
        "partial": "Partial data",
        "error": "Error",
    }[kind]


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = _normalize_provenance_kind(value)
        if normalized not in result:
            result.append(normalized)
    return result
