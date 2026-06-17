import streamlit as st
import plotly.io as pio

from app.components.theme import build_theme_css


def apply_theme() -> None:
    pio.templates.default = "plotly_dark"
    st.markdown(build_theme_css(), unsafe_allow_html=True)
