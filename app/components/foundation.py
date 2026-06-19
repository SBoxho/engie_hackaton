from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal

import streamlit as st

from app.formatting import format_age_seconds, format_timestamp
from app.i18n import current_locale, mode_label, normalize_locale, t
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


PROVENANCE_KINDS = {"official", "observed", "model", "modelled", "scenario", "fallback", "replay", "unavailable"}

StateKind = Literal["loading", "empty", "stale", "fallback", "partial", "error"]


@dataclass(frozen=True)
class TrustNotice:
    kind: StateKind
    title: str
    body: str


def provenance_badge_html(kind: str, label: str | None = None, *, locale: str | None = None) -> str:
    normalized = _normalize_provenance_kind(kind)
    if normalized == "modelled":
        return ""
    resolved_locale = _ui_locale(locale)
    text = label or _provenance_label(normalized, locale=resolved_locale)
    definition = t(f"shared.provenance.definition.{normalized}", locale=resolved_locale)
    return (
        f'<span class="ep-provenance ep-provenance-{normalized}" role="status" '
        f'aria-label="{html.escape(t("shared.aria.provenance", locale=resolved_locale))}: {html.escape(text)}. {html.escape(definition)}">'
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


def trust_notice_html(kind: StateKind, title: str, body: str, *, locale: str | None = None) -> str:
    resolved_locale = _ui_locale(locale)
    role = "alert" if kind == "error" else "status"
    return (
        f'<div class="ep-trust-state ep-trust-{kind}" role="{role}" '
        f'aria-label="{html.escape(title)}: {html.escape(body)}">'
        f'<div class="ep-trust-label">{html.escape(_state_label(kind, locale=resolved_locale))}</div>'
        f'<div><div class="ep-trust-title">{html.escape(title)}</div>'
        f'<div class="ep-trust-body">{html.escape(body)}</div></div></div>'
    )


def render_trust_notices(notices: Iterable[TrustNotice], *, locale: str | None = None) -> None:
    resolved_locale = _ui_locale(locale)
    markup = "".join(
        trust_notice_html(item.kind, *_localized_notice_text(item, resolved_locale), locale=resolved_locale)
        for item in notices
    )
    if markup:
        st.markdown(f'<div class="ep-trust-stack">{markup}</div>', unsafe_allow_html=True)


def loading_state(title: str | None = None, body: str | None = None, *, locale: str | None = None) -> None:
    resolved_locale = _ui_locale(locale)
    st.markdown(
        trust_notice_html(
            "loading",
            title or t("shared.states.loading.title", locale=resolved_locale),
            body or t("shared.states.loading.body", locale=resolved_locale),
            locale=resolved_locale,
        ),
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(trust_notice_html("empty", title, body), unsafe_allow_html=True)


def error_state(title: str, body: str, *, locale: str | None = None) -> None:
    st.markdown(trust_notice_html("error", title, body, locale=_ui_locale(locale)), unsafe_allow_html=True)


def term_tooltip_html(term: str, *, locale: str | None = None) -> str:
    resolved_locale = _ui_locale(locale)
    key = term.strip().lower()
    term_key = key.replace(" ", "_").replace("-", "_")
    definition = t(
        f"shared.terms.{term_key}.definition",
        locale=resolved_locale,
        default=t("shared.terms.default.definition", locale=resolved_locale),
    )
    label = t(f"shared.terms.{term_key}.label", locale=resolved_locale, default=term)
    return (
        f'<span class="ep-term" tabindex="0" role="note" '
        f'aria-label="{html.escape(label)}: {html.escape(definition)}">'
        f'<abbr title="{html.escape(definition)}">{html.escape(label)}</abbr>'
        f'<span class="ep-term-popover" aria-hidden="true">{html.escape(definition)}</span></span>'
    )


def render_context_bar(
    state: AppState,
    *,
    twin: TwinResponse | None,
    current_state: CurrentStateResponse | None,
    timezone_name: str,
    weather: dict[str, Any] | None = None,
    hide_replay_badge: bool = False,
    hide_fallback_badge: bool = False,
) -> None:
    last_update = _last_update(twin, current_state)
    selected_time = state.selected_timestamp or _selected_time(twin, current_state)
    mode_text = mode_label(state.mode, locale=state.locale)
    scope = REGION_NAMES.get(state.selected_region, state.selected_region)
    provenance_kinds = provenance_kinds_from_twin(twin) if twin is not None else provenance_kinds_from_current_state(current_state)
    if hide_replay_badge:
        # The "Demo context" mode label already conveys replay status, so the
        # extra Replay pill is redundant on pages that opt in to suppression.
        provenance_kinds = [kind for kind in provenance_kinds if kind != "replay"]
    if hide_fallback_badge:
        # Pages that already surface fallback context inline (e.g. per-card
        # provenance badges) can suppress the bar-level Fallback pill to keep
        # the header focused on the forecast.
        provenance_kinds = [kind for kind in provenance_kinds if kind != "fallback"]
    badges = "".join(provenance_badge_html(kind, locale=state.locale) for kind in provenance_kinds)
    age = _age_text(current_state, locale=state.locale)
    time_label, time_value, time_sublabel = _context_time_slot(
        selected_time, weather, timezone_name=timezone_name, locale=state.locale
    )
    time_sublabel_html = (
        f"<small>{html.escape(time_sublabel)}</small>" if time_sublabel else ""
    )
    st.markdown(
        f"""
        <div class="ep-context-bar" aria-label="{html.escape(t("shared.context.aria_label", locale=state.locale))}">
          <div class="ep-context-main">
            <div class="ep-context-mode">{html.escape(mode_text)}</div>
          </div>
          <div class="ep-context-item"><span>{html.escape(t("shared.context.scope", locale=state.locale))}</span><strong>{html.escape(scope)}</strong></div>
          <div class="ep-context-item"><span>{html.escape(time_label)}</span><strong>{html.escape(time_value)}</strong>{time_sublabel_html}</div>
          <div class="ep-context-item"><span>{html.escape(t("shared.context.last_update", locale=state.locale))}</span><strong>{html.escape(format_timestamp(last_update, timezone_name=timezone_name, locale=state.locale))}</strong><small>{html.escape(age)}</small></div>
          <div class="ep-context-badges">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _context_time_slot(
    selected_time: Any,
    weather: dict[str, Any] | None,
    *,
    timezone_name: str,
    locale: str,
) -> tuple[str, str, str]:
    """Decide the label/value/sublabel for the selected-time slot.

    When a live weather payload is present, the slot is repurposed to
    surface current conditions (the selected timestamp moves into the
    sublabel so it stays visible). Otherwise the original behavior is
    preserved exactly.
    """
    timestamp_text = format_timestamp(selected_time, timezone_name=timezone_name, locale=locale)
    if not weather:
        return t("shared.context.selected_time", locale=locale), timestamp_text, ""
    temp = weather.get("temperature_c")
    wind = weather.get("wind_kmh")
    if temp is None and wind is None:
        return t("shared.context.selected_time", locale=locale), timestamp_text, ""
    parts: list[str] = []
    if temp is not None:
        from app.formatting import format_temperature

        parts.append(format_temperature(temp, locale=locale))
    if wind is not None:
        parts.append(t("shared.context.wind", locale=locale, value=f"{float(wind):.0f}"))
    weather_text = ", ".join(parts)
    is_live = bool(weather.get("is_live"))
    location = weather.get("location") or "Paris"
    sub_bits: list[str] = []
    if is_live:
        source = weather.get("source") or "Open-Meteo"
        sub_bits.append(t("shared.context.live_weather_source", locale=locale, location=location, source=source))
    elif location:
        sub_bits.append(location)
    sub_bits.append(t("shared.context.selected_sublabel", locale=locale, timestamp=timestamp_text))
    return t("shared.context.current_weather", locale=locale), weather_text, " · ".join(sub_bits)


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


def _age_text(current_state: CurrentStateResponse | None, *, locale: str) -> str:
    if current_state is None:
        return ""
    return format_age_seconds(current_state.national_context.freshness.age_seconds, locale=locale)


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
    if normalized not in PROVENANCE_KINDS:
        return "model"
    return normalized


def _provenance_label(kind: str, *, locale: str) -> str:
    return t(f"shared.provenance.label.{kind}", locale=locale)


def _state_label(kind: StateKind, *, locale: str) -> str:
    return t(f"shared.state_kind.{kind}", locale=locale)


def _localized_notice_text(notice: TrustNotice, locale: str) -> tuple[str, str]:
    key = str(notice.title).strip().lower().replace("-", "_").replace(" ", "_")
    title = t(f"shared.notices.{key}.title", locale=locale, default=notice.title)
    body = t(f"shared.notices.{key}.body", locale=locale, default=notice.body)
    if "{count}" in body:
        body = body.format(count=_first_number(notice.body))
    return title, body


def _first_number(text: str) -> str:
    for token in str(text).replace("(", " ").split():
        stripped = token.strip(".,")
        if stripped.isdigit():
            return stripped
    return ""


def _ui_locale(locale: str | None) -> str:
    return normalize_locale(locale or current_locale())


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = _normalize_provenance_kind(value)
        if normalized not in result:
            result.append(normalized)
    return result
