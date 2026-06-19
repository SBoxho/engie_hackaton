from __future__ import annotations

import streamlit as st

from app.components.cards import explanation_card, section_header
from app.components.public import render_public_header

render_public_header(
    "Technical lab",
    "Engineering workbench",
    "Development, diagnostics, and model audit pages are kept out of the public decision loop.",
)

section_header("Workbench", "Diagnostics and audit pages")

cards = st.columns(3)
with cards[0]:
    explanation_card(
        "live grid",
        "Regional feed, geometry, and raw regional values.",
        label="Data",
        href="technical-live-grid",
    )
    explanation_card(
        "historical",
        "Fetch and inspect consolidated historical eco2mix data.",
        label="Data",
        href="technical-historical",
    )
with cards[1]:
    explanation_card(
        "data quality",
        "Validation findings and suspicious-row evidence.",
        label="Data",
        href="technical-data-quality",
    )
    explanation_card(
        "demand model",
        "Weather-aware model evaluation and segments.",
        label="Model",
        href="technical-demand-model",
    )
with cards[2]:
    explanation_card(
        "deployment health",
        "Artifact and runtime readiness checks.",
        label="Ops",
        href="technical-deployment-health",
    )
