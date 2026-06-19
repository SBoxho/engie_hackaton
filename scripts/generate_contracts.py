"""Generate schema and frontend type artifacts from backend contracts."""

from __future__ import annotations

import json
from pathlib import Path

from src.contracts.energy_twin import openapi_document, schema_document, typescript_declarations


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs" / "contracts" / "energy-twin.schema.json"
OPENAPI_PATH = ROOT / "docs" / "contracts" / "energy-twin.openapi.json"
FRONTEND_TYPES_PATH = ROOT / "app" / "generated" / "energy_twin.d.ts"
FRONTEND_CLIENT_PATH = ROOT / "app" / "generated" / "energy_twin_client.py"


def write_artifacts() -> list[Path]:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_TYPES_PATH.parent.mkdir(parents=True, exist_ok=True)

    SCHEMA_PATH.write_text(json.dumps(schema_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OPENAPI_PATH.write_text(json.dumps(openapi_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    FRONTEND_TYPES_PATH.write_text(typescript_declarations(), encoding="utf-8")
    FRONTEND_CLIENT_PATH.write_text(python_client_source(), encoding="utf-8")
    return [SCHEMA_PATH, OPENAPI_PATH, FRONTEND_TYPES_PATH, FRONTEND_CLIENT_PATH]


def python_client_source() -> str:
    return '''"""Generated typed client for Energy Pulse France contracts.

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
'''


def main() -> int:
    for path in write_artifacts():
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
