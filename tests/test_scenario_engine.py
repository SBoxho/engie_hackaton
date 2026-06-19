from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from typing import Any
from wsgiref.util import setup_testing_defaults

import pytest

from src.api.scenarios import ScenarioService, normalize_scenario_request
from src.api.server import create_app
from src.contracts.energy_twin import to_dict
from tests.test_twin_api import NOW, service as twin_fixture_service


def scenario_service() -> ScenarioService:
    return ScenarioService(
        twin_service=twin_fixture_service(),
        now=lambda: NOW + timedelta(minutes=7),
        cache_enabled=True,
    )


def cold_request(**updates: Any) -> dict[str, Any]:
    request = {
        "scenario_type": "cold_snap",
        "magnitude": {"temperature_delta_c": -4.0},
        "scope": "national",
        "start_time": "2026-06-18T14:00:00Z",
        "duration_hours": 3,
        "baseline_from_time": "2026-06-18T12:00:00Z",
        "hours": 12,
        "assumptions": {"source": "pytest", "heating": "directional"},
        "user_label": "test cold snap",
    }
    request.update(updates)
    return request


def outage_request(**updates: Any) -> dict[str, Any]:
    request = {
        "scenario_type": "generation_unavailability",
        "magnitude": {"unavailable_capacity_mw": 2_000.0, "asset_name": "Unit A"},
        "scope": "national",
        "start_time": "2026-06-18T14:00:00Z",
        "end_time": "2026-06-18T17:00:00Z",
        "baseline_from_time": "2026-06-18T12:00:00Z",
        "hours": 12,
        "assumptions": {"source": "pytest", "outage": "capacity unavailable"},
    }
    request.update(updates)
    return request


def ev_request(**updates: Any) -> dict[str, Any]:
    request = {
        "scenario_type": "ev_charging_shift",
        "magnitude": {
            "vehicles": 10_000,
            "average_energy_per_vehicle_kwh": 8,
            "participation_rate": 0.5,
            "original_charging_window": {"start": "18:00", "end": "22:00"},
            "target_charging_window": {"start": "01:00", "end": "05:00"},
        },
        "scope": "national",
        "start_time": "2026-06-18T16:00:00Z",
        "end_time": "2026-06-19T04:00:00Z",
        "baseline_from_time": "2026-06-18T12:00:00Z",
        "hours": 24,
        "timezone": "Europe/Paris",
        "assumptions": {"source": "pytest", "ev": "energy conservation"},
    }
    request.update(updates)
    return request


def test_ev_shift_conserves_total_scenario_energy() -> None:
    result = scenario_service().run(ev_request(), use_cache=False)

    assert result["demand_delta"]["total_mwh"] == pytest.approx(0.0, abs=1e-9)
    negative = [row for row in result["scenario_series"] if row["demand_delta_mw"] < 0]
    positive = [row for row in result["scenario_series"] if row["demand_delta_mw"] > 0]
    assert len(negative) == 4
    assert len(positive) == 4
    assert any(row["rebound_peak_candidate"] for row in positive)


def test_scenario_time_boundaries_are_start_inclusive_end_exclusive() -> None:
    result = scenario_service().run(cold_request(duration_hours=2), use_cache=False)
    changed_timestamps = [row["timestamp"] for row in result["scenario_series"] if row["demand_delta_mw"] != 0]

    assert changed_timestamps == ["2026-06-18T14:00:00Z", "2026-06-18T15:00:00Z"]


def test_cold_snap_negative_temperature_delta_raises_demand() -> None:
    result = scenario_service().run(cold_request(), use_cache=False)
    active_deltas = [row["demand_delta_mw"] for row in result["scenario_series"] if row["demand_delta_mw"] != 0]

    assert active_deltas
    assert all(delta > 0 for delta in active_deltas)
    assert result["peak_demand_delta_mw"] > 0


def test_generation_unavailability_affects_supply_not_demand() -> None:
    result = scenario_service().run(outage_request(), use_cache=False)

    assert all(row["demand_delta_mw"] == 0 for row in result["scenario_series"])
    active = [row for row in result["scenario_series"] if row["generation_availability_delta_mw"] != 0]
    assert active
    assert all(row["generation_availability_delta_mw"] == -2_000 for row in active)
    assert result["regional_deltas"]["supported"] is False


def test_changed_status_hour_calculation_reports_watch_or_high_transitions() -> None:
    request = cold_request(magnitude={"temperature_delta_c": -35.0}, duration_hours=5)
    result = scenario_service().run(request, use_cache=False)

    assert result["changed_watch_or_high_hour_count"] >= 1
    for item in result["changed_watch_or_high_hours"]:
        assert item["baseline_status"] != item["scenario_status"]
        assert {item["baseline_status"], item["scenario_status"]} & {"watch", "high"}


def test_carbon_range_is_ordered_for_positive_and_negative_response() -> None:
    cold = scenario_service().run(cold_request(), use_cache=False)
    ev = scenario_service().run(ev_request(), use_cache=False)

    assert cold["estimated_carbon_range"]["total_tonnes_co2_delta_range"][0] <= cold["estimated_carbon_range"]["total_tonnes_co2_delta_range"][1]
    assert ev["estimated_carbon_range"]["total_tonnes_co2_delta_range"][0] <= ev["estimated_carbon_range"]["total_tonnes_co2_delta_range"][1]


def test_baseline_twin_response_is_not_mutated() -> None:
    twin = twin_fixture_service()
    baseline = twin.get_twin(from_timestamp="2026-06-18T12:00:00Z", hours=12)
    before = to_dict(baseline)

    scenario_service().run(cold_request(), baseline_response=baseline, use_cache=False)

    assert to_dict(baseline) == before


def test_timezone_and_dst_duration_are_explicit() -> None:
    request = cold_request(
        start_time="2026-03-29T01:00:00+01:00",
        duration_hours=3,
        baseline_from_time="2026-03-29T01:00:00+01:00",
        timezone="Europe/Paris",
    )
    normalized = normalize_scenario_request(request)

    assert normalized["start_time"] == "2026-03-29T00:00:00Z"
    assert normalized["end_time"] == "2026-03-29T03:00:00Z"
    assert normalized["duration_hours"] == 3.0

    with pytest.raises(ValueError, match="ambiguous or nonexistent"):
        normalize_scenario_request(
            cold_request(
                start_time="2026-03-29T02:30:00",
                baseline_from_time="2026-03-29T01:00:00+01:00",
                timezone="Europe/Paris",
            )
        )


def test_invalid_magnitude_and_scope_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature_delta_c must be negative"):
        scenario_service().run(cold_request(magnitude={"temperature_delta_c": 2.0}), use_cache=False)

    with pytest.raises(ValueError, match="unsupported scope"):
        scenario_service().run(cold_request(scope="department"), use_cache=False)

    with pytest.raises(ValueError, match="national scope only"):
        scenario_service().run(outage_request(scope={"type": "regional", "regions": ["11"]}), use_cache=False)

    with pytest.raises(ValueError, match="cannot exceed"):
        scenario_service().run(outage_request(magnitude={"unavailable_capacity_mw": 50_000}), use_cache=False)

    with pytest.raises(ValueError, match="assumptions exceed"):
        scenario_service().run(cold_request(assumptions={"blob": "x" * 5000}), use_cache=False)


def test_all_initial_scenarios_can_reuse_same_baseline_forecast() -> None:
    twin = twin_fixture_service()
    baseline = twin.get_twin(from_timestamp="2026-06-18T12:00:00Z", hours=24)
    engine = scenario_service()

    cold = engine.run(cold_request(hours=24), baseline_response=baseline, use_cache=False)
    outage = engine.run(outage_request(hours=24), baseline_response=baseline, use_cache=False)
    ev = engine.run(ev_request(hours=24), baseline_response=baseline, use_cache=False)

    hashes = {
        cold["data_versions"]["baseline_series_hash"],
        outage["data_versions"]["baseline_series_hash"],
        ev["data_versions"]["baseline_series_hash"],
    }
    assert len(hashes) == 1
    assert {cold["scenario_request"]["scenario_type"], outage["scenario_request"]["scenario_type"], ev["scenario_request"]["scenario_type"]} == {
        "cold_snap",
        "generation_unavailability",
        "ev_charging_shift",
    }


def test_wsgi_post_scenarios_run_endpoint_and_cache_hash() -> None:
    engine = scenario_service()
    app = create_app(twin_service=twin_fixture_service(), scenario_service=engine)

    status, payload = request_json(app, "/v1/scenarios/run", "POST", cold_request())
    status_again, payload_again = request_json(app, "/v1/scenarios/run", "POST", cold_request())

    assert status.startswith("200")
    assert status_again.startswith("200")
    assert payload["request_hash"] == payload_again["request_hash"]
    assert payload_again["cache"]["hit"] is True
    for key in [
        "baseline_series",
        "scenario_series",
        "demand_delta",
        "changed_watch_or_high_hours",
        "balance_context_delta",
        "estimated_import_export_delta",
        "estimated_generation_response_range",
        "estimated_carbon_range",
        "regional_deltas",
        "causal_chain",
        "assumptions",
        "caveats",
        "model_versions",
        "data_versions",
    ]:
        assert key in payload
    assert any("not an operator dispatch forecast" in caveat for caveat in payload["caveats"])


def request_json(app: Any, path: str, method: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    environ: dict[str, Any] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = method
    environ["PATH_INFO"] = path
    environ["CONTENT_TYPE"] = "application/json"
    environ["CONTENT_LENGTH"] = str(len(body))
    environ["wsgi.input"] = BytesIO(body)
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    response_body = b"".join(app(environ, start_response))
    return str(captured["status"]), json.loads(response_body.decode("utf-8"))
