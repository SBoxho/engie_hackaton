from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.public_data.adapters.odre import NationalEco2MixAdapter, RegionalEco2MixAdapter
from src.public_data.adapters.weather_calendar import (
    FrenchPublicHolidayAdapter,
    FrenchSchoolHolidayAdapter,
    OpenMeteoWeatherAdapter,
)
from src.public_data.contracts import (
    AdapterConfig,
    DataWindow,
    FallbackStatus,
    QualityStatus,
    SourceSchemaError,
    SourceUnavailableError,
)
from src.public_data.http import PublicDataHttpClient, SourceCircuitBreaker
from src.public_data.quality import analyze_frame_health
from src.public_data.storage import PublicDataStore, ensure_canonical_columns
from src.public_data.time import local_day_utc_hours, render_paris

FIXTURES = Path(__file__).parent / "fixtures" / "public_data"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Response:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict[str, str] | None = None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FixtureSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("no fixture response queued")
        return self.responses.pop(0)


def fixture_http(*payloads: dict) -> PublicDataHttpClient:
    return PublicDataHttpClient(
        session=FixtureSession([Response(payload) for payload in payloads]),
        config=AdapterConfig(backoff_base_seconds=0, min_interval_seconds=0),
        sleep=lambda _: None,
    )


def test_http_client_retries_429_with_retry_after():
    session = FixtureSession([
        Response({"error": True}, status_code=429, headers={"Retry-After": "0"}),
        Response({"ok": True}),
    ])
    client = PublicDataHttpClient(
        session=session,
        config=AdapterConfig(max_retries=1, backoff_base_seconds=0),
        sleep=lambda _: None,
    )
    assert client.get_json("https://example.test") == {"ok": True}
    assert len(session.calls) == 2


def test_http_client_opens_source_circuit_breaker_after_bounded_failures():
    breaker = SourceCircuitBreaker(
        source_name="fixture_source",
        failure_threshold=1,
        recovery_timeout_seconds=60,
        clock=lambda: 100.0,
    )
    session = FixtureSession([Response({"error": True}, status_code=503)])
    client = PublicDataHttpClient(
        session=session,
        config=AdapterConfig(max_retries=0, backoff_base_seconds=0, min_interval_seconds=0),
        sleep=lambda _: None,
        source_name="fixture_source",
        circuit_breaker=breaker,
    )

    with pytest.raises(SourceUnavailableError):
        client.get_json("https://example.test")
    assert breaker.snapshot()["state"] == "open"

    with pytest.raises(SourceUnavailableError, match="circuit breaker"):
        client.get_json("https://example.test")


def test_odre_national_adapter_normalizes_generation_forecasts_and_exchanges():
    adapter = NationalEco2MixAdapter(http=fixture_http(load_fixture("odre_national.json")))
    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-01-02"))
    frame = result.silver
    assert len(frame) == 2
    assert frame["event_time"].dt.tz is not None
    assert frame["source_name"].unique().tolist() == ["odre_eco2mix_national"]
    assert {"rte_forecast_j_mw", "rte_forecast_j1_mw", "physical_balance_mw"} <= set(frame.columns)
    assert frame["exports_mw"].iloc[0] == 1200
    assert result.provenance.dataset_id == "eco2mix-national-tr"


def test_odre_regional_adapter_preserves_region_dimension():
    adapter = RegionalEco2MixAdapter(http=fixture_http(load_fixture("odre_regional.json")))
    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-01-02"))
    frame = result.silver
    assert frame["source_name"].unique().tolist() == ["odre_eco2mix_regional"]
    assert frame["source_record_id"].tolist() == ["Ile-de-France"]
    assert frame["imports_mw"].iloc[0] == 10410


def test_odre_malformed_response_is_source_schema_error():
    adapter = NationalEco2MixAdapter(http=fixture_http({"total_count": 1, "records": []}))
    with pytest.raises(SourceSchemaError, match="results"):
        adapter.fetch(DataWindow.from_values("2026-01-01", "2026-01-02"))


def test_open_meteo_weather_adapter_supports_current_and_forecast_rows():
    adapter = OpenMeteoWeatherAdapter(http=fixture_http(load_fixture("open_meteo_forecast.json")))
    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-01-02", mode="current"))
    assert {"open_meteo_weather", "open_meteo_current"} == set(result.silver["source_name"])
    assert {"temperature_c", "wind_speed_kmh", "solar_radiation_wm2"} <= set(result.silver.columns)


def test_public_and_school_calendar_adapters_use_public_json_fixtures():
    public = FrenchPublicHolidayAdapter(http=fixture_http(load_fixture("public_holidays_2026.json")))
    holidays = public.fetch(DataWindow.from_values("2026-01-01", "2027-01-01")).silver
    assert holidays["is_public_holiday"].sum() == 2
    assert render_paris(holidays["event_time"].iloc[0]).startswith("2026-01-01T00:00:00")

    school = FrenchSchoolHolidayAdapter(http=fixture_http(load_fixture("school_holidays.json")))
    school_rows = school.fetch(DataWindow.from_values("2026-01-01", "2026-12-31")).silver
    assert school_rows["zone"].tolist() == ["A"]
    assert school_rows["is_school_holiday"].tolist() == [1]


def test_storage_is_idempotent_and_builds_hourly_gold(tmp_path):
    adapter = NationalEco2MixAdapter(http=fixture_http(load_fixture("odre_national.json")))
    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-01-02"))
    store = PublicDataStore(tmp_path / "public")
    first = store.write_result(result)
    second = store.write_result(result)
    assert first.silver.inserted_rows == 2
    assert second.silver.inserted_rows == 0
    assert second.silver.unchanged_rows == 2
    assert len(store.silver.read()) == 2
    assert len(store.gold.read()) == 1
    assert first.bronze_path.exists()


def test_health_report_counts_missing_duplicates_fallback_and_schema_failure():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01T00:00Z",
                    "2026-01-01T00:00Z",
                    "2026-01-01T02:00Z",
                    "2026-01-01T03:00Z",
                ],
                utc=True,
            ),
            "published_at": pd.NaT,
            "ingested_at": pd.Timestamp("2026-01-01T04:00Z"),
            "source_name": "test_source",
            "source_revision": "fixture",
            "quality_status": [
                QualityStatus.OK.value,
                QualityStatus.OK.value,
                QualityStatus.SCHEMA_FAILURE.value,
                QualityStatus.OK.value,
            ],
            "fallback_status": [
                FallbackStatus.NONE.value,
                FallbackStatus.NONE.value,
                FallbackStatus.LAST_KNOWN_GOOD.value,
                FallbackStatus.NONE.value,
            ],
            "source_record_id": "France",
        }
    )
    report = analyze_frame_health(
        ensure_canonical_columns(frame),
        expected_cadences={"test_source": "1h"},
        adapter_failure_counts={"test_source": 2},
        now=pd.Timestamp("2026-01-01T05:00Z"),
    )
    health = report.to_dict()["sources"][0]
    assert health["missing_intervals"] == 1
    assert health["duplicate_intervals"] == 2
    assert health["schema_failures"] == 1
    assert health["fallback_records"] == 1
    assert health["adapter_failures"] == 2


def test_paris_dst_days_have_23_and_25_local_hours():
    assert len(local_day_utc_hours("2024-03-31")) == 23
    assert len(local_day_utc_hours("2024-10-27")) == 25
