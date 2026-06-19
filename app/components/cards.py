from __future__ import annotations

import html

import streamlit as st

from app.components.foundation import provenance_badge_html
from app.i18n import current_locale, normalize_locale, t


STATUS_ALIASES = {
    "calm": "green",
    "good": "green",
    "green": "green",
    "normal": "green",
    "renewable-rich": "green",
    "watch": "yellow",
    "elevated": "yellow",
    "yellow": "yellow",
    "orange": "orange",
    "tense": "orange",
    "high": "red",
    "carbon-heavy": "red",
    "needs work": "red",
    "red": "red",
    "blue": "blue",
    "info": "blue",
    "simulation": "blue",
    "light": "blue",
    "low-carbon": "blue",
    "model edge detected": "green",
    "beats baseline": "green",
    "experimental horizon": "yellow",
    "does not beat baseline": "yellow",
    "grey": "grey",
    "gray": "grey",
    "unknown": "grey",
    "muted": "grey",
    "unavailable": "grey",
}


def status_key(status: str | None) -> str:
    if not status:
        return "grey"
    return STATUS_ALIASES.get(status.strip().lower(), "grey")


def metric_card(
    label: str,
    value: str,
    detail: str | None = None,
    *,
    icon: str | None = None,
    status: str | None = None,
    provenance: str | None = None,
) -> None:
    """Render a compact product-style metric card."""
    locale = _ui_locale()
    badge = status_badge_html(status, status, locale=locale) if status else ""
    provenance_html = provenance_badge_html(provenance, locale=locale) if provenance else ""
    icon_html = f'<div class="ep-icon">{html.escape(icon)}</div>' if icon else ""
    detail_html = f'<div class="ep-detail">{html.escape(detail)}</div>' if detail else ""
    # Build as a single line: indented multi-line templates can trigger
    # CommonMark's "blank line closes HTML block" + 4-space code-block rule
    # when {badge} or {provenance_html} is empty, surfacing raw HTML as text.
    st.markdown(
        f'<div class="ep-metric-card" aria-label="{html.escape(label)}: {html.escape(value)}">'
        f'{icon_html}'
        f'<div class="ep-card-row">'
        f'<div class="ep-label">{html.escape(label)}</div>'
        f'{badge}'
        f'{provenance_html}'
        f'</div>'
        f'<div class="ep-value">{html.escape(value)}</div>'
        f'{detail_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(label: str, status: str | None = None) -> None:
    st.markdown(status_badge_html(label, status), unsafe_allow_html=True)


def status_badge_html(label: str | None, status: str | None = None, *, locale: str | None = None) -> str:
    resolved_locale = _ui_locale(locale)
    text = _status_text(label, status, locale=resolved_locale)
    return (
        f'<span class="ep-status ep-status-{status_key(status or text)}" role="status" '
        f'aria-label="{html.escape(t("shared.aria.status", locale=resolved_locale))}: {html.escape(text)}">{html.escape(text)}</span>'
    )


def section_header(kicker: str, title: str, copy: str | None = None) -> None:
    copy_html = f'<div class="ep-section-copy">{html.escape(copy)}</div>' if copy else ""
    st.markdown(
        f"""
        <div class="ep-section-kicker">{html.escape(kicker)}</div>
        <div class="ep-section-title">{html.escape(title)}</div>
        {copy_html}
        """,
        unsafe_allow_html=True,
    )


def story_steps(steps: list[tuple[str, str, str]]) -> None:
    items = []
    for index, (verb, title, detail) in enumerate(steps, start=1):
        items.append(
            f"""
            <div class="ep-story-step">
              <div class="ep-story-number">{index}</div>
              <div class="ep-label">{html.escape(verb)}</div>
              <div class="ep-title">{html.escape(title)}</div>
              <div class="ep-detail">{html.escape(detail)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="ep-story-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def source_badges(items: list[tuple[str, str]]) -> None:
    locale = _ui_locale()
    badges = [
        f'<span class="ep-source-badge"><span class="ep-source-dot"></span>{html.escape(label)}'
        f'<small>{html.escape(detail)}</small></span>'
        for label, detail in items
    ]
    st.markdown(
        f"""
        <div class="ep-source-wrap">
          <div class="ep-label">{html.escape(t("shared.sources.visible", locale=locale))}</div>
          <div class="ep-source-row">{"".join(badges)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def viz_note(title: str, detail: str, *, source: str | None = None) -> None:
    source_html = f'<span class="ep-viz-source">{html.escape(source)}</span>' if source else ""
    st.markdown(
        f"""
        <div class="ep-viz-note">
          <div>
            <div class="ep-viz-title">{html.escape(title)}</div>
            <div class="ep-viz-detail">{html.escape(detail)}</div>
          </div>
          {source_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def explanation_card(
    title: str,
    detail: str,
    *,
    label: str | None = None,
    icon: str | None = None,
    status: str | None = None,
    provenance: str | None = None,
    href: str | None = None,
) -> None:
    locale = _ui_locale()
    icon_html = f'<div class="ep-icon">{html.escape(icon)}</div>' if icon else ""
    label_html = f'<div class="ep-label">{html.escape(label)}</div>' if label else ""
    badge = status_badge_html(status, status, locale=locale) if status else ""
    provenance_html = provenance_badge_html(provenance, locale=locale) if provenance else ""
    body = (
        f"{icon_html}"
        f'<div class="ep-card-row">{label_html}{badge}{provenance_html}</div>'
        f'<div class="ep-title">{html.escape(title)}</div>'
        f'<div class="ep-detail">{html.escape(detail)}</div>'
    )
    if href:
        link = (
            f'<a class="ep-card-stretched-link" href="{html.escape(href)}" '
            f'target="_self" aria-label="{html.escape(title)}"></a>'
        )
        st.markdown(
            f'<div class="ep-explanation-card ep-explanation-card-link" '
            f'aria-label="{html.escape(title)}">{link}{body}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ep-explanation-card" aria-label="{html.escape(title)}">{body}</div>',
            unsafe_allow_html=True,
        )


def message_box(title: str, body: str, *, kind: str = "info") -> None:
    box_kind = "warning" if kind == "warning" else "info"
    st.markdown(
        f"""
        <div class="ep-box ep-box-{box_kind}">
          <div class="ep-box-title">{html.escape(title)}</div>
          <div class="ep-box-body">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def horizon_forecast_card(
    horizon: str,
    value: str,
    detail: str,
    *,
    status: str = "unknown",
) -> None:
    key = status_key(status)
    locale = _ui_locale()
    st.markdown(
        f"""
        <div class="ep-horizon-card ep-border-{key}">
          <div class="ep-card-row">
            <div class="ep-label">{html.escape(horizon)}</div>
            {status_badge_html(status, status, locale=locale)}
          </div>
          <div class="ep-value">{html.escape(value)}</div>
          <div class="ep-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def driver_card(
    icon: str,
    title: str,
    detail: str,
    *,
    label: str = "Driver",
    provenance: str | None = None,
) -> None:
    provenance_html = provenance_badge_html(provenance, locale=_ui_locale()) if provenance else ""
    st.markdown(
        f"""
        <div class="ep-driver-card" aria-label="{html.escape(label)}: {html.escape(title)}">
          <div class="ep-icon">{html.escape(icon)}</div>
          <div class="ep-card-row"><div class="ep-label">{html.escape(label)}</div>{provenance_html}</div>
          <div class="ep-title">{html.escape(title)}</div>
          <div class="ep-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hackathon_footer(*, project: str, team: str, detail: str) -> None:
    locale = _ui_locale()
    st.markdown(
        f"""
        <div class="ep-footer">
          <div>
            <div class="ep-label">{html.escape(t("shared.footer.built_for", locale=locale))}</div>
            <div class="ep-footer-title">{html.escape(project)}</div>
            <div class="ep-detail">{html.escape(detail)}</div>
          </div>
          <div class="ep-footer-team">{html.escape(team)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_text(label: str | None, status: str | None, *, locale: str) -> str:
    raw = str(label or status or "unknown").strip()
    normalized_label = raw.lower().replace("-", "_").replace(" ", "_")
    normalized_status = str(status or raw).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_label == normalized_status or normalized_label in STATUS_ALIASES:
        return t(f"shared.status.{normalized_label}", locale=locale, default=raw[:1].upper() + raw[1:])
    return raw


def _ui_locale(locale: str | None = None) -> str:
    return normalize_locale(locale or current_locale())
