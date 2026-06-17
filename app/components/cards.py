from __future__ import annotations

import html

import streamlit as st


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
) -> None:
    """Render a compact product-style metric card."""
    badge = status_badge_html(status, status) if status else ""
    icon_html = f'<div class="ep-icon">{html.escape(icon)}</div>' if icon else ""
    detail_html = f'<div class="ep-detail">{html.escape(detail)}</div>' if detail else ""
    st.markdown(
        f"""
        <div class="ep-metric-card">
          {icon_html}
          <div class="ep-card-row">
            <div class="ep-label">{html.escape(label)}</div>
            {badge}
          </div>
          <div class="ep-value">{html.escape(value)}</div>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, status: str | None = None) -> None:
    st.markdown(status_badge_html(label, status), unsafe_allow_html=True)


def status_badge_html(label: str | None, status: str | None = None) -> str:
    text = label or "Unknown"
    return f'<span class="ep-status ep-status-{status_key(status or text)}">{html.escape(text)}</span>'


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


def explanation_card(
    title: str,
    detail: str,
    *,
    label: str | None = None,
    icon: str | None = None,
    status: str | None = None,
) -> None:
    icon_html = f'<div class="ep-icon">{html.escape(icon)}</div>' if icon else ""
    label_html = f'<div class="ep-label">{html.escape(label)}</div>' if label else ""
    badge = status_badge_html(status, status) if status else ""
    st.markdown(
        f"""
        <div class="ep-explanation-card">
          {icon_html}
          <div class="ep-card-row">{label_html}{badge}</div>
          <div class="ep-title">{html.escape(title)}</div>
          <div class="ep-detail">{html.escape(detail)}</div>
        </div>
        """,
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
    st.markdown(
        f"""
        <div class="ep-horizon-card ep-border-{key}">
          <div class="ep-card-row">
            <div class="ep-label">{html.escape(horizon)}</div>
            {status_badge_html(status, status)}
          </div>
          <div class="ep-value">{html.escape(value)}</div>
          <div class="ep-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def driver_card(icon: str, title: str, detail: str, *, label: str = "Driver") -> None:
    st.markdown(
        f"""
        <div class="ep-driver-card">
          <div class="ep-icon">{html.escape(icon)}</div>
          <div class="ep-label">{html.escape(label)}</div>
          <div class="ep-title">{html.escape(title)}</div>
          <div class="ep-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
