from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.cards import section_header
from app.components.deployment_health import artifact_checks, data_check, mode_check
from app.data_loader import load_public_context
from app.generated.energy_twin_client import EnergyTwinApiClient

section_header("Technical lab", "Deployment health")

context = load_public_context()
energy: pd.DataFrame = context["energy"]
checks = [
    mode_check(),
    data_check(energy, context["national_source"]) if not energy.empty else data_check(pd.DataFrame(), context["national_source"]),
    *artifact_checks(),
]
st.dataframe(
    pd.DataFrame([check.__dict__ for check in checks]).rename(
        columns={"label": "Check", "status": "Status", "detail": "Detail"}
    ),
    width="stretch",
    hide_index=True,
)

try:
    health = EnergyTwinApiClient().get_data_health()
except (OSError, TypeError, ValueError):
    health = None

if health is not None:
    section_header("Source Health", "Freshness, gaps, fallback, and circuit state")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Source": source.source_id,
                    "State": source.operating_state.value,
                    "Latest timestamp": source.freshness.timestamp,
                    "Missing intervals": source.missing_intervals,
                    "Fallback rows": source.fallback_records,
                    "Adapter failures": source.adapter_failures,
                    "Circuit breaker": source.circuit_breaker_state,
                    "Reason": source.reason,
                }
                for source in health.sources
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    section_header("Model And Scenario Health", "Forecast and simulation readiness")
    model = health.model_health
    scenario = health.scenario_engine
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Check": "Demand forecast",
                    "Status": model.status,
                    "Version": model.model_version,
                    "Latest successful run": model.latest_successful_forecast_run_id,
                    "Recent MAE MW": model.recent_forecast_error_mae_mw,
                    "Detail": model.reason or model.fallback_usage,
                },
                {
                    "Check": "Scenario engine",
                    "Status": "available" if scenario.available else "unavailable",
                    "Version": scenario.version,
                    "Latest successful run": scenario.last_successful_scenario_id,
                    "Recent MAE MW": None,
                    "Detail": scenario.reason,
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
