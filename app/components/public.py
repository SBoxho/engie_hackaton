from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.cards import status_badge, status_badge_html
from app.components.charts import COLORS, dark_chart_layout
from app.formatting import format_mw as format_display_mw
from app.i18n import current_locale, normalize_locale, render_language_selector, t
from app.view_models import ForecastPointView, GridSnapshotView, format_mw, pressure_score
from src.contracts.status_thresholds import score_status, status_label


def render_public_header(title: str, subtitle: str, mode: str) -> None:
    _ = mode
    subtitle_markup = f'<div class="ep-section-copy">{html.escape(subtitle)}</div>' if subtitle else ""
    title_col, language_col = st.columns([1, 0.22], gap="large", vertical_alignment="top")
    with title_col:
        st.markdown(
            f"""
            <div class="ep-page-head">
              <div>
                <div class="ep-page-brand">{html.escape(t("shared.app.name"))}</div>
                <div class="ep-page-title">{html.escape(title)}</div>
                {subtitle_markup}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with language_col:
        render_language_selector()


def selected_location(event: object) -> str | None:
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(event, dict):
        points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[0]
    if isinstance(point, dict):
        value = point.get("location")
        if value:
            return str(value)
        customdata = point.get("customdata") or []
        if customdata:
            return str(customdata[0])
    return None


def selected_region(regional: pd.DataFrame, event: object | None = None) -> pd.Series | None:
    if regional.empty or "region_code" not in regional:
        return None
    event_code = selected_location(event)
    codes = regional["region_code"].astype(str).tolist()
    if event_code in set(codes):
        st.session_state["selected_region_code"] = event_code
    stored = st.session_state.get("selected_region_code")
    if stored not in codes:
        score_column = "demand_anomaly_score" if "demand_anomaly_score" in regional else "demand_pressure"
        stored = str(regional.loc[regional[score_column].idxmax(), "region_code"])
        st.session_state["selected_region_code"] = stored
    labels = regional.set_index("region_code")["region_display"].to_dict()
    selected_code = st.selectbox(
        "Selected region",
        codes,
        index=codes.index(stored),
        format_func=lambda code: labels.get(code, code),
    )
    st.session_state["selected_region_code"] = selected_code
    return regional.loc[regional["region_code"].astype(str).eq(selected_code)].iloc[0]


def generation_mix_bar(snapshot: GridSnapshotView) -> go.Figure:
    labels = list(snapshot.generation_by_source)
    values = [snapshot.generation_by_source[label] for label in labels]
    fig = go.Figure()
    for label, value in zip(labels, values):
        fig.add_bar(
            x=[value],
            y=["Generation"],
            orientation="h",
            name=label,
            marker_color=COLORS.get(label, "#94a3b8"),
            hovertemplate=f"{label}: %{{x:,.0f}} MW<extra></extra>",
        )
    fig.update_layout(
        **dark_chart_layout(
            barmode="stack",
            height=150,
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis_title="MW",
            yaxis_title=None,
            legend_title=None,
        )
    )
    return fig


def forecast_ribbon(points: list[ForecastPointView], *, limit: int = 8) -> None:
    if not points:
        st.info("Forecast context is unavailable for this window.")
        return
    columns = st.columns(min(limit, len(points)))
    for point, column in zip(points[:limit], columns):
        local = point.timestamp.tz_convert("Europe/Paris")
        with column:
            status_badge(f"{local:%H:%M}", "blue")
            st.markdown(
                f"""
                <div class="ep-ribbon-cell">
                  <div class="ep-value">{html.escape(point.pressure_label)}</div>
                  <div class="ep-detail">{html.escape(format_mw(point.p50))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    selected_timestamp: pd.Timestamp | None = None,
    now_timestamp: pd.Timestamp | None = None,
    timezone_name: str = "Europe/Paris",
    show_period_bands: bool = True,
    locale: str = "en",
) -> go.Figure:
    fig = go.Figure()
    selected_index: int | None = None
    if not history.empty:
        hist = history.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
        hist = hist.dropna(subset=["timestamp"]).sort_values("timestamp").tail(96)
        hist_customdata = [
            [
                _chart_datetime_label(timestamp, timezone_name=timezone_name, locale=locale),
                format_display_mw(value, locale=locale),
            ]
            for timestamp, value in zip(hist["timestamp"], hist["consumption_mw"])
        ]
        fig.add_trace(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["consumption_mw"],
                name=t("next48h.chart.actual_demand", locale=locale),
                mode="lines",
                line=dict(color="#f8fafc", width=2.2),
                customdata=hist_customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{t('next48h.chart.actual_demand', locale=locale)}: %{{customdata[1]}}<extra></extra>"
                ),
            )
        )
        if now_timestamp is None and not hist.empty:
            now_timestamp = pd.Timestamp(hist["timestamp"].max())
    if not forecast.empty:
        forecast = forecast.copy()
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True, errors="coerce")
        forecast = forecast.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if selected_timestamp is not None and not forecast.empty:
            target = pd.Timestamp(selected_timestamp)
            if target.tzinfo is None:
                target = target.tz_localize("UTC")
            target = target.tz_convert("UTC")
            distances = (forecast["timestamp"] - target).abs()
            selected_index = int(distances.idxmin()) if not distances.empty else None
        if show_period_bands and "pressure_label" in forecast:
            for row in forecast.itertuples(index=False):
                label = str(getattr(row, "pressure_label", ""))
                if label not in {"Watch", "High"}:
                    continue
                color = "rgba(245,158,11,.09)" if label == "Watch" else "rgba(239,68,68,.10)"
                timestamp = pd.Timestamp(getattr(row, "timestamp"))
                fig.add_vrect(
                    x0=timestamp - pd.Timedelta(minutes=30),
                    x1=timestamp + pd.Timedelta(minutes=30),
                    fillcolor=color,
                    line_width=0,
                    layer="below",
                )
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["p90"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                name="p90",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(20,184,166,.16)",
                line=dict(width=0),
                name=t("next48h.chart.likely_range", locale=locale),
            )
        )
        if "usual_demand_mw" in forecast and pd.to_numeric(forecast["usual_demand_mw"], errors="coerce").notna().any():
            usual_customdata = [
                [
                    _chart_datetime_label(timestamp, timezone_name=timezone_name, locale=locale),
                    format_display_mw(value, locale=locale),
                ]
                for timestamp, value in zip(forecast["timestamp"], forecast["usual_demand_mw"])
            ]
            fig.add_trace(
                go.Scatter(
                    x=forecast["timestamp"],
                    y=forecast["usual_demand_mw"],
                    mode="lines",
                    name=t("next48h.chart.usual_baseline", locale=locale),
                    line=dict(color="rgba(226,232,240,.72)", width=1.8, dash="dot"),
                    customdata=usual_customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{t('next48h.chart.usual_baseline', locale=locale)}: %{{customdata[1]}}<extra></extra>"
                    ),
                )
            )
        customdata = [
            [
                pd.Timestamp(row.timestamp).isoformat(),
                _chart_datetime_label(row.timestamp, timezone_name=timezone_name, locale=locale),
                _status_display(row.pressure_label if hasattr(row, "pressure_label") else "", locale=locale),
                _source_display(row.source if hasattr(row, "source") else "", locale=locale),
                format_display_mw(row.p50, locale=locale),
            ]
            for row in forecast.itertuples(index=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["p50"],
                mode="lines+markers",
                name=t("next48h.chart.forecast_demand", locale=locale),
                line=dict(color="#42d6c7", width=2.4),
                marker=dict(size=7, color="#42d6c7", line=dict(width=1, color="#07111d")),
                selectedpoints=[selected_index] if selected_index is not None else None,
                selected=dict(marker=dict(size=13, color="#f8fafc")),
                unselected=dict(marker=dict(opacity=0.82)),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    f"{t('next48h.chart.p50', locale=locale)}: %{{customdata[4]}}<br>"
                    f"{t('next48h.chart.status', locale=locale)}: %{{customdata[2]}}<br>"
                    f"{t('next48h.chart.route', locale=locale)}: %{{customdata[3]}}<extra></extra>"
                ),
            )
        )
        if selected_index is not None:
            selected = forecast.iloc[selected_index]
            fig.add_trace(
                go.Scatter(
                    x=[selected["timestamp"]],
                    y=[selected["p50"]],
                    mode="markers",
                    name=t("next48h.chart.selected_hour", locale=locale),
                    marker=dict(size=16, color="#f8fafc", symbol="circle-open", line=dict(width=3, color="#42d6c7")),
                    hoverinfo="skip",
                )
            )
    if now_timestamp is not None:
        now = pd.Timestamp(now_timestamp)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        now = now.tz_convert("UTC")
        fig.add_vline(
            x=now,
            line=dict(color="rgba(248,250,252,.75)", width=1.4, dash="dash"),
            annotation_text=t("next48h.chart.now", locale=locale),
            annotation_position="top left",
        )
    timestamps: list[pd.Timestamp] = []
    for trace in fig.data:
        x_values = getattr(trace, "x", None)
        if x_values is None:
            continue
        for value in x_values:
            try:
                timestamps.append(_as_utc_chart_timestamp(value))
            except (TypeError, ValueError):
                continue
    if timestamps:
        start_local = min(timestamps).tz_convert(timezone_name)
        end_local = max(timestamps).tz_convert(timezone_name)
        separators = pd.date_range(
            start_local.normalize() + pd.Timedelta(days=1),
            end_local.normalize(),
            freq="D",
            tz=timezone_name,
        )
        for separator in separators:
            fig.add_vline(
                x=separator.tz_convert("UTC"),
                line=dict(color="rgba(148,163,184,.34)", width=1, dash="dot"),
            )
        tickvals, ticktext = _chart_tick_values(timestamps, timezone_name=timezone_name, locale=locale)
    else:
        tickvals, ticktext = [], []
    fig.update_layout(
        **dark_chart_layout(
            height=430,
            margin=dict(l=8, r=8, t=18, b=8),
            xaxis_title=None,
            yaxis_title=t("next48h.chart.axis_y", locale=locale),
            hovermode="x unified",
            legend_title=None,
            clickmode="event+select",
        )
    )
    if tickvals:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig


def _chart_datetime_label(value: Any, *, timezone_name: str, locale: str) -> str:
    local = _as_utc_chart_timestamp(value).tz_convert(timezone_name)
    return t(
        "next48h.time.local_hour",
        locale=locale,
        weekday=t(f"next48h.time.weekdays_short.{local.weekday()}", locale=locale, default=f"{local:%a}"),
        day=f"{local.day:02d}",
        month=t(f"next48h.time.months_short.{local.month}", locale=locale, default=f"{local:%b}"),
        time=f"{local:%H:%M}",
    )


def _chart_tick_values(
    timestamps: list[pd.Timestamp],
    *,
    timezone_name: str,
    locale: str,
) -> tuple[list[pd.Timestamp], list[str]]:
    unique = sorted({_as_utc_chart_timestamp(value) for value in timestamps})
    if not unique:
        return [], []
    selected = [
        value
        for value in unique
        if value.tz_convert(timezone_name).hour in {0, 12}
    ]
    if len(selected) < 3:
        stride = max(1, len(unique) // 6)
        selected = unique[::stride]
    for edge in (unique[0], unique[-1]):
        if edge not in selected:
            selected.append(edge)
    selected = sorted(set(selected))
    return selected, [_chart_datetime_label(value, timezone_name=timezone_name, locale=locale) for value in selected]


def _as_utc_chart_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _status_display(value: Any, *, locale: str) -> str:
    text = str(value or "").strip()
    if not text:
        return t("next48h.status.unknown", locale=locale)
    key = text.lower().replace("-", "_").replace(" ", "_")
    return t(f"next48h.status.{key}", locale=locale, default=text)


def _source_display(value: Any, *, locale: str) -> str:
    text = str(value or "").strip()
    if not text:
        return t("next48h.tables.forecast.unavailable", locale=locale)
    lowered = text.lower()
    if "usual-demand baseline fallback" in lowered or "usual demand baseline fallback" in lowered:
        return t("next48h.sources.usual_fallback", locale=locale)
    if "unsupported" in lowered:
        return t("next48h.sources.unsupported_route", locale=locale)
    if "fallback" in lowered:
        return t("next48h.sources.fallback_route", locale=locale)
    if text == "Demand forecast":
        return t("next48h.sources.demand_forecast", locale=locale)
    return text


def driver_waterfall(point: ForecastPointView) -> go.Figure:
    drivers = [driver for driver in point.drivers if str(driver.get("name", "")).lower() != "uncertainty"]
    labels = [driver["name"] for driver in drivers]
    values = []
    for driver in drivers:
        value = float(driver.get("value", 0) or 0)
        if driver["name"] == "Supply margin":
            value = -value
        values.append(value)
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["relative"] * len(values),
            connector={"line": {"color": "rgba(203,213,225,.55)"}},
            increasing={"marker": {"color": "#f59e0b"}},
            decreasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#38bdf8"}},
        )
    )
    fig.update_layout(
        **dark_chart_layout(
            height=300,
            margin=dict(l=8, r=8, t=14, b=8),
            yaxis_title="Demand and margin contribution",
            showlegend=False,
        )
    )
    return fig


def scenario_comparison_chart(baseline: pd.DataFrame, modified: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not baseline.empty:
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["p50"],
                mode="lines",
                name="Baseline",
                line=dict(color="#f8fafc", width=2),
            )
        )
    if not modified.empty:
        fig.add_trace(
            go.Scatter(
                x=modified["timestamp"],
                y=modified["p50"],
                mode="lines",
                name="Scenario",
                line=dict(color="#42d6c7", width=2.6),
            )
        )
    fig.update_layout(
        **dark_chart_layout(
            height=310,
            margin=dict(l=8, r=8, t=18, b=8),
            xaxis_title=None,
            yaxis_title="Demand MW",
            hovermode="x unified",
            legend_title=None,
        )
    )
    return fig


def regional_pressure_frame(regional: pd.DataFrame) -> pd.DataFrame:
    frame = regional.copy()
    if frame.empty:
        return frame
    if "scenario_score" not in frame:
        frame["scenario_score"] = frame.get("demand_anomaly_score", frame.get("demand_pressure", 0.5))
    frame["pressure_label"] = frame["scenario_score"].map(lambda value: status_label(score_status(float(value))))
    frame["demand_anomaly_label"] = frame["pressure_label"]
    return frame


def about_project_drawer(*, locale: str | None = None) -> None:
    resolved_locale = normalize_locale(locale or current_locale())
    with st.expander(t("shared.about.title", locale=resolved_locale), expanded=False):
        st.write(t("shared.about.body_1", locale=resolved_locale))
        st.write(t("shared.about.body_2", locale=resolved_locale))


def provenance_drawer(*items: tuple[str, str], locale: str | None = None) -> None:
    resolved_locale = normalize_locale(locale or current_locale())
    with st.expander(t("shared.provenance_drawer.title", locale=resolved_locale), expanded=False):
        for label, detail in items:
            st.markdown(f"**{html.escape(label)}**: {html.escape(detail)}")
