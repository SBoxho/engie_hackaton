"""Generated typed client for Energy Pulse France contracts.

This Streamlit frontend is in-process, so the generated client calls the
framework-neutral backend services directly and round-trips through the
contract serializers to keep UI code aligned with the backend schema.

Generated from src/contracts/energy_twin.py. Do not edit by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.api.current_state import CurrentStateService, default_service
from src.api.scenarios import ScenarioService, default_scenario_service
from src.api.twin import TwinService, default_twin_service
from src.contracts.energy_twin import (
    CurrentStateResponse,
    DataHealthResponse,
    SourcesResponse,
    StatusThresholdsResponse,
    TwinResponse,
    from_dict,
    to_dict,
)


@dataclass(frozen=True)
class CurrentStateQuery:
    region: str


@dataclass(frozen=True)
class TwinQuery:
    from_timestamp: str | datetime | None = None
    hours: int = 48
    region: str | None = None


@dataclass(frozen=True)
class ScenarioRunQuery:
    request: dict
    use_cache: bool = True


class EnergyTwinClientProtocol(Protocol):
    def get_current_state(self, query: CurrentStateQuery) -> CurrentStateResponse:
        ...

    def get_data_health(self) -> DataHealthResponse:
        ...

    def get_sources(self) -> SourcesResponse:
        ...

    def get_status_thresholds(self) -> StatusThresholdsResponse:
        ...

    def get_twin(self, query: TwinQuery) -> TwinResponse:
        ...

    def run_scenario(self, query: ScenarioRunQuery) -> dict:
        ...


class EnergyTwinApiClient:
    """Typed client over the in-process backend services."""

    def __init__(
        self,
        *,
        current_state_service: CurrentStateService | None = None,
        twin_service: TwinService | None = None,
        scenario_service: ScenarioService | None = None,
    ) -> None:
        self._current_state_service = current_state_service or default_service
        self._twin_service = twin_service or default_twin_service
        self._scenario_service = scenario_service or default_scenario_service

    def get_current_state(self, query: CurrentStateQuery) -> CurrentStateResponse:
        payload = self._current_state_service.get_current_state(query.region)
        return from_dict(CurrentStateResponse, to_dict(payload))

    def get_data_health(self) -> DataHealthResponse:
        payload = self._current_state_service.get_data_health()
        return from_dict(DataHealthResponse, to_dict(payload))

    def get_sources(self) -> SourcesResponse:
        payload = self._current_state_service.get_sources()
        return from_dict(SourcesResponse, to_dict(payload))

    def get_status_thresholds(self) -> StatusThresholdsResponse:
        payload = self._current_state_service.get_status_thresholds()
        return from_dict(StatusThresholdsResponse, to_dict(payload))

    def get_twin(self, query: TwinQuery) -> TwinResponse:
        payload = self._twin_service.get_twin(
            from_timestamp=query.from_timestamp,
            hours=query.hours,
            region=query.region,
        )
        return from_dict(TwinResponse, to_dict(payload))

    def run_scenario(self, query: ScenarioRunQuery) -> dict:
        return self._scenario_service.run(query.request, use_cache=query.use_cache)
