from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from app.data_loader import load_public_context
from app.generated.energy_twin_client import CurrentStateQuery, EnergyTwinApiClient, TwinQuery
from app.state import AppState, forecast_run_id
from src.contracts.energy_twin import CurrentStateResponse, TwinResponse


TYPED_DEMO_CONTEXT_CACHE_VERSION = 2


@dataclass(frozen=True)
class PublicAppContext:
    legacy: dict[str, Any]
    current_state: CurrentStateResponse | None
    twin: TwinResponse | None
    forecast_run_id: str | None

    @property
    def energy(self) -> pd.DataFrame:
        value = self.legacy.get("energy", pd.DataFrame())
        return value if isinstance(value, pd.DataFrame) else pd.DataFrame()

    @property
    def is_replay(self) -> bool:
        if self.twin and self.twin.snapshots:
            return self.twin.snapshots[0].mode.value == "replay"
        return str(self.legacy.get("mode", "")).upper() != "LIVE"


def load_typed_public_context(
    state: AppState,
    *,
    hours: int = 48,
    include_current_state: bool = False,
) -> PublicAppContext:
    legacy = load_public_context()
    if _empty_energy(legacy):
        return PublicAppContext(legacy=legacy, current_state=None, twin=None, forecast_run_id=None)

    client = EnergyTwinApiClient()
    current_state = (
        _typed_current_state(client, state.selected_region, cache_version=TYPED_DEMO_CONTEXT_CACHE_VERSION)
        if include_current_state
        else None
    )
    twin = (
        _typed_twin(client, state.selected_region, hours=hours, cache_version=TYPED_DEMO_CONTEXT_CACHE_VERSION)
        if hours > 0
        else None
    )
    run_id = forecast_run_id(twin.from_time if twin is not None else None, hours) if twin is not None else None
    return PublicAppContext(
        legacy=legacy,
        current_state=current_state,
        twin=twin,
        forecast_run_id=run_id,
    )


@st.cache_data(ttl=900, show_spinner=False)
def _typed_current_state(
    _client: EnergyTwinApiClient,
    region: str,
    *,
    cache_version: int = TYPED_DEMO_CONTEXT_CACHE_VERSION,
) -> CurrentStateResponse | None:
    _ = cache_version
    try:
        return _client.get_current_state(CurrentStateQuery(region=region))
    except (OSError, ValueError, TypeError):
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _typed_twin(
    _client: EnergyTwinApiClient,
    region: str,
    *,
    hours: int,
    cache_version: int = TYPED_DEMO_CONTEXT_CACHE_VERSION,
) -> TwinResponse | None:
    _ = cache_version
    try:
        return _client.get_twin(
            TwinQuery(
                from_timestamp=None,
                hours=hours,
                region=region,
            )
        )
    except (OSError, ValueError, TypeError):
        return None


def _empty_energy(context: dict[str, Any]) -> bool:
    energy = context.get("energy")
    return not isinstance(energy, pd.DataFrame) or energy.empty
