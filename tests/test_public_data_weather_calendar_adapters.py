from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.public_data.adapters.weather_calendar import (
    ARCHIVE_URL,
    FORECAST_URL,
    HISTORICAL_FORECAST_URL,
    FrenchPublicHolidayAdapter,
    FrenchPublicHolidayError,
    FrenchSchoolHolidayAdapter,
    OpenMeteoWeatherAdapter,
    SourceValidationError,
    WeatherLocation,
    calendar_year_windows,
    open_meteo_incremental_windows,
    validate_source_url,
)
from src.public_data.contracts import DataWindow


FIXTURES = Path(__file__).parent / "fixtures" / "public_data"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeHttp:
    def __init__(self, *payloads: dict):
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict | None]] = []

    def get_json(self, url: str, *, params: dict | None = None) -> dict:
        self.calls.append((url, params))
        if not self.payloads:
            raise AssertionError("no fixture response queued")
        return self.payloads.pop(0)


class FakeSchoolHttp:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[str, dict | None]] = []

    def get_json(self, url: str, *, params: dict | None = None) -> dict:
        self.calls.append((url, params))
        assert params is not None
        offset = int(params["offset"])
        limit = int(params["limit"])
        return {
            "total_count": self.payload["total_count"],
            "results": self.payload["results"][offset : offset + limit],
        }


def test_open_meteo_current_uses_forecast_endpoint_and_provenance() -> None:
    http = FakeHttp(load_fixture("open_meteo_forecast.json"))
    adapter = OpenMeteoWeatherAdapter(http=http)

    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-01-02", mode="current"))

    assert http.calls[0][0] == FORECAST_URL
    assert http.calls[0][1]["timezone"] == "UTC"
    assert "temperature_2m" in http.calls[0][1]["current"]
    assert {"open_meteo_weather", "open_meteo_current"} == set(result.silver["source_name"])
    current = result.silver[result.silver["source_name"] == "open_meteo_current"].iloc[0]
    assert current["temperature_c"] == pytest.approx(6.5)
    assert result.provenance.source_url == FORECAST_URL
    assert "content_sha256" in result.provenance.query


def test_open_meteo_forecast_filters_window_and_keeps_published_at() -> None:
    http = FakeHttp(load_fixture("open_meteo_forecast.json"))
    adapter = OpenMeteoWeatherAdapter(http=http)

    result = adapter.fetch(
        DataWindow.from_values("2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", mode="forecast")
    )

    params = http.calls[0][1]
    assert http.calls[0][0] == FORECAST_URL
    assert params["start_date"] == "2026-01-01"
    assert params["end_date"] == "2026-01-01"
    assert result.silver["event_time"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T01:00:00Z"),
    ]
    assert result.provenance.published_at == datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    assert result.silver["published_at"].notna().all()


def test_open_meteo_archive_and_historical_forecast_endpoints() -> None:
    archive_http = FakeHttp(load_fixture("open_meteo_archive.json"))
    historical_http = FakeHttp(load_fixture("open_meteo_archive.json"))

    archive = OpenMeteoWeatherAdapter(http=archive_http).fetch(
        DataWindow.from_values("2024-01-15T00:00:00Z", "2024-01-15T02:00:00Z", mode="archive")
    )
    historical = OpenMeteoWeatherAdapter(http=historical_http).fetch(
        DataWindow.from_values("2024-01-15T00:00:00Z", "2024-01-15T02:00:00Z", mode="historical_forecast")
    )

    assert archive_http.calls[0][0] == ARCHIVE_URL
    assert historical_http.calls[0][0] == HISTORICAL_FORECAST_URL
    assert archive.provenance.source_url == ARCHIVE_URL
    assert historical.provenance.source_url == HISTORICAL_FORECAST_URL


def test_incremental_windows_are_half_open_and_source_sized() -> None:
    windows = open_meteo_incremental_windows(
        "2026-01-01T00:00:00Z",
        "2026-02-10T00:00:00Z",
        chunk_days=15,
        mode="archive",
    )

    assert [(window.start, window.end, window.mode) for window in windows] == [
        (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-16T00:00:00Z"), "archive"),
        (pd.Timestamp("2026-01-16T00:00:00Z"), pd.Timestamp("2026-01-31T00:00:00Z"), "archive"),
        (pd.Timestamp("2026-01-31T00:00:00Z"), pd.Timestamp("2026-02-10T00:00:00Z"), "archive"),
    ]
    calendar_windows = calendar_year_windows("2025-07-01", "2027-02-01")
    assert [window.chunk_id for window in calendar_windows] == [
        "2025-07-01_2026-01-01",
        "2026-01-01_2027-01-01",
        "2027-01-01_2027-02-01",
    ]


def test_french_public_holidays_fetch_years_and_normalize_rows() -> None:
    http = FakeHttp(load_fixture("calendar_public_holidays_2026.json"))
    adapter = FrenchPublicHolidayAdapter(http=http)

    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-05-02", mode="calendar"))

    assert http.calls[0][0].endswith("/metropole/2026.json")
    assert result.silver["holiday_name"].tolist() == ["Jour de l'an", "Fete du Travail"]
    assert result.silver["is_public_holiday"].tolist() == [1, 1]
    assert result.provenance.source_name == "french_public_holidays"
    assert result.provenance.query["territory"] == "metropole"


def test_french_school_holidays_paginate_and_expand_zones() -> None:
    http = FakeSchoolHttp(load_fixture("calendar_school_holidays.json"))
    adapter = FrenchSchoolHolidayAdapter(http=http, page_size=1)

    result = adapter.fetch(DataWindow.from_values("2026-01-01", "2026-06-01", mode="calendar"))

    assert len(http.calls) == 2
    assert all(call[1]["limit"] == 1 for call in http.calls)
    assert result.silver["zone"].tolist() == ["A", "B", "C"]
    assert result.silver["is_school_holiday"].eq(1).all()
    assert result.provenance.dataset_id == "fr-en-calendrier-scolaire"
    assert result.provenance.published_at == datetime(2025, 9, 1, tzinfo=timezone.utc)


def test_source_validation_and_source_specific_errors_are_explicit() -> None:
    with pytest.raises(SourceValidationError, match="https"):
        validate_source_url(
            "http://api.open-meteo.com/v1/forecast",
            source_name="open_meteo_weather",
            allowed_hosts=("api.open-meteo.com",),
        )
    with pytest.raises(SourceValidationError, match="latitude"):
        WeatherLocation("bad", "Bad", 91.0, 2.3522)
    with pytest.raises(FrenchPublicHolidayError, match="unsupported public-holiday territory"):
        FrenchPublicHolidayAdapter(territory="unknown", http=FakeHttp({}))
