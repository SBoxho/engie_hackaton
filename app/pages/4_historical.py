from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.components.charts import consumption_chart, production_area_chart
from app.components.layout import apply_theme
from src.config import settings
from src.data_processing.features import add_time_features
from src.data_processing.storage import PartitionedParquetStore
from src.data_sources.rte_eco2mix_historical import (
    MAX_RANGE_DAYS,
    fetch_historical,
    load_cached_historical,
)

apply_theme()

st.title("Historical national grid")
st.caption("Consolidated RTE éCO2mix data via the official ODRÉ open-data API")

today = date.today()
left, right = st.columns(2)
start = left.date_input("Start (inclusive)", value=today - timedelta(days=7), max_value=today)
end = right.date_input("End (exclusive)", value=today, max_value=today + timedelta(days=1))


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(start_value: date, end_value: date) -> pd.DataFrame:
    result = add_time_features(fetch_historical(start_value, end_value), settings.timezone)
    PartitionedParquetStore(settings.energy_store_dir).upsert(result)
    return result


frame: pd.DataFrame | None = None
if st.button("Fetch official history", type="primary"):
    try:
        with st.spinner("Fetching ODRÉ history…"):
            frame = _fetch(start, end)
    except Exception as exc:
        st.error(str(exc))
else:
    try:
        frame = load_cached_historical()
        st.info("Showing the latest immutable local snapshot. Choose dates and fetch to refresh.")
    except FileNotFoundError:
        st.info(f"Choose an interval of at most {MAX_RANGE_DAYS} days, then fetch official history.")

if frame is not None:
    if frame.empty:
        st.warning("No consolidated records were published for this interval.")
    else:
        newest = frame.sort_values("timestamp").iloc[-1]
        first, second, third = st.columns(3)
        first.metric("Rows", f"{len(frame):,}")
        second.metric("Latest demand", f"{newest['consumption_mw']:,.0f} MW")
        third.metric("Latest CO₂ intensity", f"{newest['co2_intensity_g_per_kwh']:,.0f} g/kWh")
        st.plotly_chart(consumption_chart(frame), width="stretch")
        st.plotly_chart(production_area_chart(frame), width="stretch")
        st.download_button(
            "Download standardized CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name=f"eco2mix_historical_{start}_{end}.csv",
            mime="text/csv",
        )
        st.caption("Historical consolidated data is distinct from the near-live rolling feed on the home page.")
