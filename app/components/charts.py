from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = {
    "Nuclear": "#a78bfa",
    "Wind": "#42d6c7",
    "Solar": "#ffd166",
    "Hydro": "#38bdf8",
    "Bioenergy": "#84cc16",
    "Gas": "#fb7185",
    "Coal": "#78716c",
    "Oil": "#f97316",
}
MIX_COLUMNS = {
    "nuclear_mw": "Nuclear", "wind_mw": "Wind", "solar_mw": "Solar",
    "hydro_mw": "Hydro", "bioenergy_mw": "Bioenergy", "gas_mw": "Gas",
    "coal_mw": "Coal", "oil_mw": "Oil",
}


def dark_chart_layout(**kwargs) -> dict:
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(7,17,29,0.35)",
        "font": {"color": "#dbeafe"},
    }
    layout.update(kwargs)
    return layout


def consumption_chart(frame: pd.DataFrame) -> go.Figure:
    fig = px.line(frame, x="timestamp", y="consumption_mw")
    fig.update_traces(line_color="#42d6c7", line_width=3, fill="tozeroy", fillcolor="rgba(66,214,199,.08)")
    fig.update_layout(**dark_chart_layout(xaxis_title=None, yaxis_title="MW", hovermode="x unified"))
    return fig


def production_area_chart(frame: pd.DataFrame) -> go.Figure:
    long = frame.melt(id_vars="timestamp", value_vars=list(MIX_COLUMNS), var_name="source", value_name="mw")
    long["source"] = long["source"].map(MIX_COLUMNS)
    fig = px.area(long, x="timestamp", y="mw", color="source", color_discrete_map=COLORS)
    fig.update_layout(**dark_chart_layout(xaxis_title=None, yaxis_title="MW", hovermode="x unified", legend_title=None))
    return fig


def mix_donut(row: pd.Series) -> go.Figure:
    labels = list(MIX_COLUMNS.values())
    values = [max(float(row[column] or 0), 0) for column in MIX_COLUMNS]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.68, marker_colors=[COLORS[x] for x in labels]))
    fig.update_layout(**dark_chart_layout(showlegend=True, legend_title=None, margin=dict(l=10, r=10, t=20, b=10)))
    return fig
