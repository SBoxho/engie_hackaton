from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.cards import section_header
from app.components.data_quality import render_data_quality
from app.data_loader import load_public_context

section_header("Technical lab", "Data quality")

context = load_public_context()
energy: pd.DataFrame = context["energy"]
if energy.empty:
    st.warning("No energy data is loaded for quality checks.")
else:
    render_data_quality(energy)
