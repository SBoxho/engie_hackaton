from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.components.charts import dark_chart_layout


DIVERGING_COLORSCALE = [
    [0.0, "#2563eb"],
    [0.44, "#bfdbfe"],
    [0.5, "#f8fafc"],
    [0.56, "#fecaca"],
    [1.0, "#dc2626"],
]


def diverging_metric_range(values: pd.Series, *, floor: float = 1.0, ceiling: float | None = None) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if numeric.empty:
        return floor
    max_abs = float(numeric.abs().max())
    if ceiling is not None:
        max_abs = min(max_abs, ceiling)
    return max(max_abs, floor)


def clipped_diverging_values(values: pd.Series, max_abs: float) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(-max_abs, max_abs)


def signed_percent_tick(value: float, *, clipped: bool = False) -> str:
    sign = "+" if value > 0 else ""
    prefix = ">=" if clipped and value > 0 else "<=" if clipped and value < 0 else ""
    return f"{prefix}{sign}{value:g}%"


def signed_mw_tick(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f} MW"


def has_meaningful_anomaly_signal(
    frame: pd.DataFrame,
    *,
    min_regions: int = 4,
    ceiling: float = 15.0,
    max_clipped_share: float = 0.75,
) -> bool:
    if frame.empty or "demand_anomaly_pct" not in frame:
        return False
    values = pd.to_numeric(frame["demand_anomaly_pct"], errors="coerce")
    if "availability_flag" in frame:
        availability = frame["availability_flag"].fillna(False).astype(bool)
        values = values.loc[availability]
    values = values.dropna()
    if len(values) < min_regions:
        return False
    if values.round(6).nunique() < 2:
        return False
    return float(values.abs().gt(ceiling).mean()) < max_clipped_share


def _append_geometry_lines(geometry: dict, lon: list[float | None], lat: list[float | None]) -> None:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        return
    if not isinstance(polygons, list):
        return
    for polygon in polygons:
        if not isinstance(polygon, list):
            continue
        for ring in polygon[:1]:
            if not isinstance(ring, list):
                continue
            for point in ring:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    lon.append(float(point[0]))
                    lat.append(float(point[1]))
            lon.append(None)
            lat.append(None)


def _department_boundary_trace(department_geojson: dict) -> go.Scattergeo | None:
    lon: list[float | None] = []
    lat: list[float | None] = []
    for feature in department_geojson.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            _append_geometry_lines(geometry, lon, lat)
    if not lon:
        return None
    return go.Scattergeo(
        lon=lon,
        lat=lat,
        mode="lines",
        line=dict(color="rgba(226,232,240,.45)", width=0.45),
        hoverinfo="skip",
        showlegend=False,
    )


def regional_demand_choropleth(
    frame: pd.DataFrame,
    geojson: dict,
    department_geojson: dict | None = None,
    *,
    department_metrics_available: bool = False,
    value_column: str = "demand_anomaly_score",
    colorbar_title: str = "Demand vs usual",
    hover_value_label: str = "Demand context",
) -> go.Figure:
    value_column = value_column if value_column in frame else "demand_pressure"
    label_column = "demand_anomaly_label" if "demand_anomaly_label" in frame else value_column
    z_values = pd.to_numeric(frame[value_column], errors="coerce").fillna(0.5).clip(0, 1)
    hover = frame.assign(
        demand_label=frame["consumption_mw"].map(lambda value: f"{value:,.0f} MW"),
        renewable_label=frame["renewable_share"].map(lambda value: f"{value:.0%}"),
        production_label=frame["total_production_mw"].map(lambda value: f"{value:,.0f} MW"),
        map_value_label=frame[label_column].map(str),
    )
    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=hover["region_code"],
            z=z_values,
            zmin=0,
            zmax=1,
            featureidkey="properties.code",
            colorscale=[
                [0.0, "#2563eb"],
                [0.18, "#22d3ee"],
                [0.38, "#22c55e"],
                [0.55, "#facc15"],
                [0.74, "#fb923c"],
                [1.0, "#ef4444"],
            ],
            marker_line_color="rgba(219,234,254,.72)",
            marker_line_width=0.7,
            colorbar=dict(
                title=colorbar_title,
                tickmode="array",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=["Low", "25%", "50%", "75%", "High"],
                outlinewidth=0,
                thickness=12,
                len=0.72,
            ),
            customdata=hover[
                [
                    "region_display",
                    "demand_label",
                    "production_label",
                    "renewable_label",
                    "map_value_label",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Demand: %{customdata[1]}<br>"
                "Production: %{customdata[2]}<br>"
                "Renewable share: %{customdata[3]}<br>"
                f"{hover_value_label}: %{{customdata[4]}}<extra></extra>"
            ),
        )
    )
    if department_geojson and department_metrics_available:
        boundary_trace = _department_boundary_trace(department_geojson)
        if boundary_trace is not None:
            fig.add_trace(boundary_trace)
    fig.update_geos(
        scope="europe",
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
    )
    fig.update_layout(
        **dark_chart_layout(
            height=610,
            margin=dict(l=0, r=0, t=12, b=0),
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(14, 165, 233, .18)",
                landcolor="rgba(15, 28, 44, .92)",
            ),
        )
    )
    return fig


def regional_anomaly_choropleth(
    frame: pd.DataFrame,
    geojson: dict,
    department_geojson: dict | None = None,
    *,
    department_metrics_available: bool = False,
    demand_label_title: str = "Observed demand",
    source_label_title: str = "Source",
) -> go.Figure:
    """Render regional demand anomaly on a zero-centred diverging scale."""
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "region_code",
                "region_display",
                "demand_anomaly_pct",
                "demand_label",
                "usual_label",
                "difference_label",
                "freshness_label",
                "source_label",
                "availability_flag",
                "unavailable_reason",
            ]
        )
    frame = frame.copy()
    if "region_code" in frame:
        frame["region_code"] = frame["region_code"].astype(str)
    anomaly_values = pd.to_numeric(frame.get("demand_anomaly_pct"), errors="coerce")
    availability = (
        frame["availability_flag"].fillna(False).astype(bool)
        if "availability_flag" in frame
        else pd.Series(True, index=frame.index)
    )
    available_mask = availability & anomaly_values.notna()
    available = frame.loc[available_mask].copy()
    unavailable = frame.loc[~available_mask].copy()
    fig = go.Figure()
    if not available.empty:
        available_values = anomaly_values.loc[available.index]
        max_abs = diverging_metric_range(available_values, floor=1.0, ceiling=15.0)
        clipped = bool(available_values.abs().max() > max_abs)
        z_values = clipped_diverging_values(available_values, max_abs)
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=available["region_code"],
                z=z_values,
                zmin=-max_abs,
                zmid=0,
                zmax=max_abs,
                featureidkey="properties.code",
                colorscale=DIVERGING_COLORSCALE,
                marker_line_color="rgba(219,234,254,.72)",
                marker_line_width=0.7,
                colorbar=dict(
                    title="Versus usual",
                    tickmode="array",
                    tickvals=[-max_abs, 0, max_abs],
                    ticktext=[
                        signed_percent_tick(-max_abs, clipped=clipped),
                        "0%",
                        signed_percent_tick(max_abs, clipped=clipped),
                    ],
                    outlinewidth=0,
                    thickness=12,
                    len=0.72,
                ),
                customdata=available[
                    [
                        "region_display",
                        "demand_label",
                        "usual_label",
                        "difference_label",
                        "freshness_label",
                        "source_label",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{demand_label_title}: %{{customdata[1]}}<br>"
                    "Usual demand: %{customdata[2]}<br>"
                    "Difference: %{customdata[3]}<br>"
                    "Freshness: %{customdata[4]}<br>"
                    f"{source_label_title}: %{{customdata[5]}}<extra></extra>"
                ),
                name="Available regional anomaly",
            )
        )
    if not unavailable.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=unavailable["region_code"],
                z=[0] * len(unavailable),
                zmin=0,
                zmax=1,
                featureidkey="properties.code",
                colorscale=[[0, "#475569"], [1, "#475569"]],
                showscale=False,
                marker_line_color="rgba(226,232,240,.76)",
                marker_line_width=0.9,
                customdata=unavailable[
                    [
                        "region_display",
                        "unavailable_reason",
                        "freshness_label",
                        "source_label",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Demand anomaly: Unavailable<br>"
                    "Reason: %{customdata[1]}<br>"
                    "Freshness: %{customdata[2]}<br>"
                    "Source: %{customdata[3]}<extra></extra>"
                ),
                name="Unavailable regional data",
            )
        )
    if department_geojson and department_metrics_available:
        boundary_trace = _department_boundary_trace(department_geojson)
        if boundary_trace is not None:
            fig.add_trace(boundary_trace)
    fig.update_geos(
        scope="europe",
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
    )
    fig.update_layout(
        **dark_chart_layout(
            height=610,
            margin=dict(l=0, r=0, t=12, b=0),
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(14, 165, 233, .18)",
                landcolor="rgba(15, 28, 44, .92)",
            ),
            showlegend=False,
        )
    )
    return fig
