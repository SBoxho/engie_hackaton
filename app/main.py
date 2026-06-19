from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import apply_theme
from app.i18n import init_locale, nav_label, set_html_lang

st.set_page_config(page_title="Energy Pulse France", page_icon=":zap:", layout="wide")
apply_theme()
locale = init_locale()
set_html_lang(locale)

now_page = st.Page("pages/now.py", title=nav_label("now", locale=locale), url_path="", default=True)
next_page = st.Page("pages/next_48h.py", title=nav_label("next_48h", locale=locale), url_path="next-48h")
what_if_page = st.Page("pages/what_if.py", title=nav_label("what_if", locale=locale), url_path="what-if")

technical_lab_page = st.Page(
    "pages/technical_lab.py",
    title=nav_label("technical_lab", locale=locale),
    url_path="technical-lab",
    visibility="hidden",
)
technical_pages = [
    st.Page("pages/1_live_grid.py", title="live grid", url_path="technical-live-grid", visibility="hidden"),
    st.Page("pages/4_historical.py", title="historical", url_path="technical-historical", visibility="hidden"),
    st.Page("pages/6_demand_model.py", title="demand model", url_path="technical-demand-model", visibility="hidden"),
    st.Page("pages/technical_data_quality.py", title="data quality", url_path="technical-data-quality", visibility="hidden"),
    st.Page(
        "pages/technical_deployment_health.py",
        title="deployment health",
        url_path="technical-deployment-health",
        visibility="hidden",
    ),
]

current_page = st.navigation(
    [now_page, next_page, what_if_page, technical_lab_page, *technical_pages],
    position="top",
)

with st.sidebar:
    st.page_link(technical_lab_page, label=nav_label("technical_lab", locale=locale), icon=":material/engineering:")

current_page.run()
