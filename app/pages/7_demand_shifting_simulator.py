from __future__ import annotations

import streamlit as st

from app.components.cards import message_box, section_header
from app.components.layout import apply_theme

st.set_page_config(page_title="Demand-shifting simulator", page_icon=":material/tune:", layout="wide")
apply_theme()

section_header(
    "Action simulator",
    "Demand-shifting simulator",
    "Move flexible electricity use away from high-pressure hours and compare the result.",
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Flexible use")
    appliance = st.selectbox("Load type", ["EV charging", "Dishwasher", "Water heater", "Laundry"])
    power_kw = st.slider("Power", min_value=0.5, max_value=11.0, value=3.7, step=0.1, format="%.1f kW")
    duration = st.slider("Duration", min_value=1, max_value=8, value=3, step=1, format="%d h")

with right:
    st.subheader("Shift window")
    current_hour = st.slider("Current start hour", min_value=0, max_value=23, value=19, step=1)
    shifted_hour = st.slider("Shifted start hour", min_value=0, max_value=23, value=2, step=1)
    energy_kwh = power_kw * duration
    st.metric("Flexible energy", f"{energy_kwh:.1f} kWh")

message_box(
    "Next step",
    "Connect this simulator to the 24h pressure timeline and estimate carbon and demand-pressure changes from the same official data pipeline.",
    kind="info",
)

st.link_button("Back to Energy Pulse France", "/")
