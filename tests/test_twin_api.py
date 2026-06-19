from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from wsgiref.util import setup_testing_defaults

import numpy as np
import pandas as pd
import pytest

from src.api.current_state import CurrentStateBundle, SourceTable
from src.api.server import create_app
from src.api.twin import RESIDUAL_BUCKET_NAME, TwinService
from src.contracts.energy_twin import (
    EstimateProvenanceKind,
    OperatingState,
    TwinResponse,
    from_dict,
    to_dict,
)
from src.contracts.status_thresholds import load_status_thresholds, threshold_config_version
from src.data_sources.rte_eco2mix_regional import REGION_NAMES
from src.models.usual_demand import build_hourly_analytical_dataset


NOW = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def hourly_fixture(end: datetime = NOW, days: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range(end=pd.Timestamp(end), periods=days * 24, freq="h")
    rows = []
    for index, timestamp in enumerate(timestamps):
        local = timestamp.tz_convert("Europe/Paris")
        demand = 50_000 + 1_800 * np.sin(2 * np.pi * local.hour / 24) + local.dayofweek * 90
        solar = max(0.0, 2_400 - abs(local.hour - 13) * 340)
        rows.append(
            {
                "timestamp": timestamp,
                "region": "France",
                "consumption_mw": demand,
                "nuclear_mw": 32_000,
                "wind_mw": 4_200 + local.hour * 15,
                "solar_mw": solar,
                "hydro_mw": 4_000,
                "gas_mw": 1_000,
                "coal_mw": 0,
                "oil_mw": 0,
                "bioenergy_mw": 500,
                "net_imports_mw": 300,
                "imports_mw": 300,
                "exports_mw": 0,
                "total_production_mw": 32_000 + 4_200 + solar + 5_500,
                "co2_intensity_g_per_kwh": 42,
                "weather_wind_speed_kmh": 22 + local.hour * 0.2,
                "weather_solar_radiation_wm2": max(0.0, 600 - abs(local.hour - 13) * 90),
            }
        )
    return pd.DataFrame(rows)


def regional_history_fixture(end: datetime = NOW) -> pd.DataFrame:
    national = hourly_fixture(end=end, days=14)
    frames = []
    shares = {"11": 0.18, "84": 0.16, "53": 0.05}
    for code, share in shares.items():
        frame = national.copy()
        frame["region"] = REGION_NAMES[code]
        frame["region_code"] = code
        frame["region_display"] = REGION_NAMES[code]
        frame["consumption_mw"] = frame["consumption_mw"] * share
        frame["total_production_mw"] = frame["total_production_mw"] * share
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def source_bundle() -> CurrentStateBundle:
    hourly = hourly_fixture()
    regional = regional_history_fixture()
    regional_snapshot = (
        regional.sort_values("timestamp")
        .drop_duplicates("region_code", keep="last")
        .reset_index(drop=True)
    )
    return CurrentStateBundle(
        national=SourceTable(
            frame=hourly,
            source_id="odre_eco2mix_national",
            name="RTE eCO2mix national fixture",
            operating_state=OperatingState.FRESH_LIVE_DATA,
            source_quality="validated",
            retrieved_at=NOW,
        ),
        regional_snapshot=SourceTable(
            frame=regional_snapshot,
            source_id="odre_eco2mix_regional",
            name="RTE eCO2mix regional fixture",
            operating_state=OperatingState.FRESH_LIVE_DATA,
            source_quality="validated",
            retrieved_at=NOW,
        ),
        regional_history=regional,
        ecowatt=SourceTable(
            frame=pd.DataFrame(),
            source_id="odre_ecowatt",
            name="EcoWatt unavailable fixture",
            operating_state=OperatingState.SOURCE_UNAVAILABLE,
            source_quality="unavailable",
            reason="Fixture EcoWatt unavailable.",
        ),
    )


def model_hourly_fixture() -> pd.DataFrame:
    return build_hourly_analytical_dataset(hourly_fixture(), derive_national_from_regions=False)


def service(
    *,
    generation_forecast_loader=None,
    unavailability_loader=None,
) -> TwinService:
    return TwinService(
        bundle_loader=lambda _now: source_bundle(),
        hourly_loader=model_hourly_fixture,
        generation_forecast_loader=generation_forecast_loader,
        unavailability_loader=unavailability_loader,
        artifact=None,
        artifact_path=None,
        now=lambda: NOW,
    )


def test_national_regional_reconciliation_and_no_regional_balance_status() -> None:
    response = service().get_twin(from_timestamp="2026-06-18T12:37:00Z", hours=3, region="11")

    assert len(response.snapshots) == 4
    assert response.from_time == datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    for snapshot in response.snapshots:
        regional_sum = sum(region.forecast.p50.value or 0.0 for region in snapshot.regional_demand_context or [])
        national = snapshot.demand_forecast.p50.value or 0.0
        assert regional_sum == pytest.approx(national, abs=1e-6)
        assert (snapshot.regional_demand_context or [])[0].region_code == "11"

    payload = json.dumps(to_dict(response))
    assert "regional_balance_status" not in payload
    assert "adequacy_status" not in payload


def test_generation_totals_residual_bucket_and_fallback_provenance() -> None:
    snapshot = service().get_twin(from_timestamp=NOW, hours=1).snapshots[1]
    mix = snapshot.generation_mix_estimate
    assert mix is not None
    total = sum(component.value.value or 0.0 for component in mix.components if component.included_in_total)

    assert mix.total.value == pytest.approx(total)
    residual = next(component for component in mix.components if component.component == RESIDUAL_BUCKET_NAME)
    assert residual.value.value == pytest.approx(5_800.0)
    assert residual.provenance_kind == EstimateProvenanceKind.RESIDUAL_ESTIMATE
    assert snapshot.wind_estimate.provenance_kind == EstimateProvenanceKind.STATISTICAL_ESTIMATE
    assert snapshot.solar_estimate.provenance_kind == EstimateProvenanceKind.STATISTICAL_ESTIMATE
    assert "weather" in (snapshot.wind_estimate.value.source.name or "").lower()


def test_official_generation_forecast_is_used_when_supplied() -> None:
    def official_generation(_origin: datetime, _hours: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-06-18T13:00:00Z"),
                    "wind_mw": 7_000,
                    "solar_mw": 3_000,
                }
            ]
        )

    snapshot = service(generation_forecast_loader=official_generation).get_twin(from_timestamp=NOW, hours=1).snapshots[1]

    assert snapshot.wind_estimate.value.value == 7_000
    assert snapshot.wind_estimate.provenance_kind == EstimateProvenanceKind.OFFICIAL_FORECAST
    assert snapshot.solar_estimate.provenance_kind == EstimateProvenanceKind.OFFICIAL_FORECAST


def test_unavailable_optional_sources_and_threshold_configuration_are_exposed() -> None:
    response = service().get_twin(from_timestamp=NOW, hours=1)
    snapshot = response.snapshots[1]
    config = load_status_thresholds()

    assert any(item.field == "rte_generation_forecast_optional" for item in response.unavailable_fields)
    assert any(item.field == "rte_asset_unavailability_optional" for item in response.unavailable_fields)
    assert snapshot.modelled_national_balance_context.threshold_config_version == threshold_config_version()
    assert snapshot.modelled_national_balance_context.calculation_inputs == config["modelled_balance_context"]["calculation_inputs"]
    assert [item.component for item in snapshot.modelled_balance_contributions] == [
        "residual_load_percentile",
        "announced_unavailability_ratio",
    ]


def test_announced_unavailability_changes_balance_context_component() -> None:
    def unavailability(_origin: datetime, _hours: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "start_time": "2026-06-18T13:00:00Z",
                    "end_time": "2026-06-18T14:00:00Z",
                    "technology": "nuclear",
                    "unavailable_mw": 4_000,
                }
            ]
        )

    snapshot = service(unavailability_loader=unavailability).get_twin(from_timestamp=NOW, hours=1).snapshots[1]

    assert snapshot.generation_availability_context.announced_unavailable.value == 4_000
    assert snapshot.generation_availability_context.announced_unavailability_components
    contribution = next(
        item for item in snapshot.modelled_balance_contributions if item.component == "announced_unavailability_ratio"
    )
    assert contribution.value.value == pytest.approx(4_000 / (snapshot.demand_forecast.p50.value or 1.0))


def test_timestamp_alignment_contract_round_trip_and_wsgi_endpoint() -> None:
    twin_service = service()
    app = create_app(twin_service=twin_service)
    status, payload = request_json(app, "/v1/twin", "from=2026-06-18T12:37:00Z&hours=2&region=11")
    restored = from_dict(TwinResponse, payload)

    assert status.startswith("200")
    assert restored.from_time == datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    assert [item.event_time.hour for item in restored.snapshots] == [12, 13, 14]
    assert all(item.event_time.minute == 0 for item in restored.snapshots)


def request_json(app: Any, path: str, query: str = "") -> tuple[str, dict[str, Any]]:
    environ: dict[str, Any] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "GET"
    environ["PATH_INFO"] = path
    environ["QUERY_STRING"] = query
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    return str(captured["status"]), json.loads(body.decode("utf-8"))
