from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from typing import Any
from wsgiref.util import setup_testing_defaults

import pandas as pd

from src.api.current_state import CurrentStateBundle, CurrentStateService, SourceTable
from src.api.server import create_app
from src.config import settings
from src.contracts.energy_twin import CurrentStateResponse, OperatingState, from_dict
from src.data_sources.rte_eco2mix_regional import REGION_NAMES


NOW = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def energy_frame(end: datetime = NOW, *, hours: int = 24 * 14, region: str = "France") -> pd.DataFrame:
    timestamps = pd.date_range(end=pd.Timestamp(end), periods=hours, freq="h")
    rows = []
    for index, timestamp in enumerate(timestamps):
        local = timestamp.tz_convert("Europe/Paris")
        demand = 48_000 + local.hour * 220 + local.dayofweek * 80 + index * 2
        rows.append(
            {
                "timestamp": timestamp,
                "region": region,
                "consumption_mw": demand,
                "nuclear_mw": 31_000,
                "wind_mw": 4_000 + local.hour * 10,
                "solar_mw": max(0, 2_500 - abs(local.hour - 13) * 300),
                "hydro_mw": 5_200,
                "gas_mw": 1_400,
                "coal_mw": 0,
                "oil_mw": 40,
                "bioenergy_mw": 750,
                "imports_mw": 900,
                "exports_mw": 120,
                "net_imports_mw": 780,
                "co2_intensity_g_per_kwh": 37,
                "total_production_mw": 44_890,
                "renewable_production_mw": 12_450,
                "renewable_share": 0.277,
                "fossil_production_mw": 1_440,
                "fossil_share": 0.032,
            }
        )
    return pd.DataFrame(rows)


def regional_history(end: datetime = NOW, *, region_codes: tuple[str, ...] = ("11", "84")) -> pd.DataFrame:
    frames = []
    for offset, code in enumerate(region_codes):
        frame = energy_frame(end=end, region=REGION_NAMES[code])
        frame["region_code"] = code
        frame["region_display"] = REGION_NAMES[code]
        frame["consumption_mw"] = frame["consumption_mw"] * (0.12 + offset * 0.03)
        frame["total_production_mw"] = frame["total_production_mw"] * (0.08 + offset * 0.02)
        frame["net_imports_mw"] = 100 + offset * 50
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def regional_snapshot(end: datetime = NOW, *, region_codes: tuple[str, ...] = ("11", "84")) -> pd.DataFrame:
    history = regional_history(end=end, region_codes=region_codes)
    return (
        history.sort_values("timestamp")
        .drop_duplicates("region_code", keep="last")
        .sort_values("region_display")
        .reset_index(drop=True)
    )


def ecowatt_frame(end: datetime = NOW) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(end),
                "ecowatt_status": "green",
                "ecowatt_label": "Normal",
                "ecowatt_severity": 1,
                "ecowatt_message": "Fixture official signal.",
                "ecowatt_source": "EcoWatt fixture",
                "ecowatt_dataset_id": "fixture",
                "ecowatt_source_url": "https://example.test/ecowatt",
            }
        ]
    )


def bundle(
    *,
    end: datetime = NOW,
    national_state: OperatingState = OperatingState.FRESH_LIVE_DATA,
    regional_state: OperatingState = OperatingState.FRESH_LIVE_DATA,
    region_codes: tuple[str, ...] = ("11", "84"),
    ecowatt_available: bool = True,
) -> CurrentStateBundle:
    return CurrentStateBundle(
        national=SourceTable(
            frame=energy_frame(end),
            source_id="odre_eco2mix_national",
            name="RTE eCO2mix national fixture",
            operating_state=national_state,
            source_quality="validated" if national_state == OperatingState.FRESH_LIVE_DATA else "last_known_good",
            retrieved_at=NOW,
            reason="fixture fallback" if national_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK else None,
        ),
        regional_snapshot=SourceTable(
            frame=regional_snapshot(end, region_codes=region_codes),
            source_id="odre_eco2mix_regional",
            name="RTE eCO2mix regional fixture",
            operating_state=regional_state,
            source_quality="validated" if regional_state == OperatingState.FRESH_LIVE_DATA else "last_known_good",
            retrieved_at=NOW,
            reason="fixture fallback" if regional_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK else None,
        ),
        regional_history=regional_history(end, region_codes=region_codes),
        ecowatt=SourceTable(
            frame=ecowatt_frame(end) if ecowatt_available else pd.DataFrame(),
            source_id="odre_ecowatt",
            name="EcoWatt fixture",
            operating_state=(
                OperatingState.FRESH_LIVE_DATA if ecowatt_available else OperatingState.SOURCE_UNAVAILABLE
            ),
            source_quality="validated" if ecowatt_available else "unavailable",
            retrieved_at=NOW if ecowatt_available else None,
            reason=None if ecowatt_available else "Fixture EcoWatt unavailable.",
        ),
    )


def service_for(source_bundle: CurrentStateBundle) -> CurrentStateService:
    return CurrentStateService(loader=lambda _now: source_bundle, now=lambda: NOW, cache_ttl_seconds=900)


def test_current_state_contract_uses_usual_demand_and_separates_signals() -> None:
    response = service_for(bundle()).get_current_state("11")
    payload = response.to_dict()
    restored = from_dict(CurrentStateResponse, payload)

    assert restored.operating_state == OperatingState.FRESH_LIVE_DATA
    assert restored.national_context.demand.current.value is not None
    assert restored.national_context.demand.usual.value is not None
    assert restored.national_context.demand.difference_vs_usual_pct.value is not None
    assert restored.national_context.carbon_estimate.included_in_modelled_status is False
    assert restored.national_context.official_ecowatt_signal.signal_type == "official"
    assert restored.national_context.modelled_status is not None
    assert restored.national_context.modelled_status.signal_type == "modelled"
    assert restored.selected_region_context.local_generation.total.value is not None
    assert "regional_supply" not in json.dumps(payload)
    assert any(item.region_id == "11" and item.availability_flag for item in restored.map)


def test_stale_live_data_returns_delayed_operating_state() -> None:
    delayed = service_for(bundle(end=NOW - timedelta(hours=2))).get_current_state("11")

    assert delayed.operating_state == OperatingState.DELAYED_LIVE_DATA
    assert delayed.national_context.freshness.state == OperatingState.DELAYED_LIVE_DATA
    assert delayed.national_context.freshness.reason is not None


def test_missing_regional_record_returns_nullable_values_with_reason() -> None:
    response = service_for(bundle(region_codes=("84",))).get_current_state("11")
    selected = response.selected_region_context
    map_entry = next(item for item in response.map if item.region_id == "11")

    assert response.operating_state == OperatingState.SOURCE_UNAVAILABLE
    assert selected.freshness.state == OperatingState.SOURCE_UNAVAILABLE
    assert selected.demand.current.value is None
    assert selected.demand.current.reason is not None
    assert map_entry.availability_flag is False
    assert map_entry.observed_demand.value is None
    assert map_entry.observed_demand.reason is not None


def test_unavailable_ecowatt_stays_official_and_nullable() -> None:
    response = service_for(bundle(ecowatt_available=False)).get_current_state("11")
    signal = response.national_context.official_ecowatt_signal

    assert signal.signal_type == "official"
    assert signal.available is False
    assert signal.status is None
    assert signal.reason == "Fixture EcoWatt unavailable."
    assert response.national_context.modelled_status is not None


def test_last_known_good_and_source_bundle_cache_are_explicit() -> None:
    calls = {"count": 0}
    source_bundle = bundle(
        national_state=OperatingState.LAST_KNOWN_GOOD_FALLBACK,
        regional_state=OperatingState.LAST_KNOWN_GOOD_FALLBACK,
    )

    def loader(_now: datetime) -> CurrentStateBundle:
        calls["count"] += 1
        return source_bundle

    service = CurrentStateService(loader=loader, now=lambda: NOW, cache_ttl_seconds=900)
    first = service.get_current_state("11")
    second = service.get_current_state("84")

    assert calls["count"] == 1
    assert first.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK
    assert second.cache.cache_hit is True
    assert second.national_context.freshness.state == OperatingState.LAST_KNOWN_GOOD_FALLBACK


def test_wsgi_endpoints_return_contract_payloads() -> None:
    app = create_app(service_for(bundle()))
    status, payload = request_json(app, "/v1/state/current", "region=11")

    assert status.startswith("200")
    assert payload["region"] == "11"
    assert payload["national_context"]["official_ecowatt_signal"]["signal_type"] == "official"

    status, health = request_json(app, "/v1/data-health")
    assert status.startswith("200")
    assert health["sources"]
    assert health["model_health"]["model_id"] == "demand-forecast"
    assert health["scenario_engine"]["available"] is True

    status, sources = request_json(app, "/v1/sources")
    assert status.startswith("200")
    assert any(source["source_id"] == "usual_demand_baseline" for source in sources["sources"])

    status, thresholds = request_json(app, "/v1/config/status-thresholds")
    assert status.startswith("200")
    assert thresholds["version"]

    status, metrics = request_json(app, "/v1/metrics")
    assert status.startswith("200")
    assert "requests_total" in metrics
    assert "http_latency" in metrics


def test_wsgi_responses_include_request_id_and_restricted_cors() -> None:
    app = create_app(service_for(bundle()))
    status, payload, headers = request_json_full(
        app,
        "/v1/state/current",
        "region=11",
        origin="http://localhost:8501",
        request_id="judge-request-1",
    )
    header_map = dict(headers)

    assert status.startswith("200")
    assert payload["region"] == "11"
    assert header_map["X-Request-ID"] == "judge-request-1"
    assert header_map["Access-Control-Allow-Origin"] == "http://localhost:8501"

    _status, _payload, blocked_headers = request_json_full(
        app,
        "/v1/state/current",
        "region=11",
        origin="https://example.invalid",
    )
    assert "Access-Control-Allow-Origin" not in dict(blocked_headers)


def test_wsgi_rejects_oversized_scenario_payload_safely() -> None:
    app = create_app()
    body = b"{" + b'"x":' + b'"a"' * (settings.api_max_body_bytes + 1) + b"}"
    environ: dict[str, Any] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "POST"
    environ["PATH_INFO"] = "/v1/scenarios/run"
    environ["CONTENT_TYPE"] = "application/json"
    environ["CONTENT_LENGTH"] = str(len(body))
    environ["wsgi.input"] = BytesIO(body)
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    response_body = b"".join(app(environ, start_response))
    payload = json.loads(response_body.decode("utf-8"))

    assert str(captured["status"]).startswith("400")
    assert payload["error"] == "bad_request"
    assert "request_id" in payload
    assert "Traceback" not in response_body.decode("utf-8")


def test_data_health_exposes_source_model_and_scenario_status() -> None:
    response = service_for(bundle()).get_data_health()

    assert response.sources[0].missing_intervals >= 0
    assert response.sources[0].circuit_breaker_state == "closed"
    assert response.model_health.model_id == "demand-forecast"
    assert response.model_health.status
    assert response.scenario_engine.available is True
    assert response.scenario_engine.version.startswith("scenario-engine")


def request_json(app: Any, path: str, query: str = "") -> tuple[str, dict[str, Any]]:
    status, payload, _headers = request_json_full(app, path, query)
    return status, payload


def request_json_full(
    app: Any,
    path: str,
    query: str = "",
    *,
    origin: str | None = None,
    request_id: str | None = None,
) -> tuple[str, dict[str, Any], list[tuple[str, str]]]:
    environ: dict[str, Any] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "GET"
    environ["PATH_INFO"] = path
    environ["QUERY_STRING"] = query
    if origin:
        environ["HTTP_ORIGIN"] = origin
    if request_id:
        environ["HTTP_X_REQUEST_ID"] = request_id
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    return str(captured["status"]), json.loads(body.decode("utf-8")), list(captured["headers"])
