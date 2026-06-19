from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
import requests

from src.public_data.adapters.odre import (
    NATIONAL_DATASET_ID,
    ODRESourceSchemaError,
    REGIONAL_DATASET_ID,
    NationalEco2MixAdapter,
    RegionalEco2MixAdapter,
)
from src.public_data.contracts import AdapterConfig, DataWindow, SourceSchemaError, SourceUnavailableError
from src.public_data.http import PublicDataHttpClient


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "public_data"


def fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class Response:
    def __init__(
        self,
        payload: dict,
        *,
        status_code: int = 200,
        status_error: Exception | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.payload = payload
        self.status_code = status_code
        self.status_error = status_error
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_error:
            raise self.status_error

    def json(self) -> dict:
        return self.payload


class Session:
    def __init__(self, *items: dict | Response):
        self.items = list(items)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs: object) -> Response:
        self.calls.append((url, kwargs))
        if not self.items:
            raise AssertionError("no fixture response queued")
        item = self.items.pop(0)
        return item if isinstance(item, Response) else Response(item)


def fixture_http(*payloads: dict | Response, session: Session | None = None) -> PublicDataHttpClient:
    return PublicDataHttpClient(
        session=session or Session(*payloads),
        config=AdapterConfig(max_retries=0, backoff_base_seconds=0, min_interval_seconds=0),
        sleep=lambda _: None,
    )


def test_national_adapter_maps_forecasts_exchanges_published_at_and_provenance() -> None:
    session = Session(fixture("odre_eco2mix_national.json"))
    adapter = NationalEco2MixAdapter(http=fixture_http(session=session))

    result = adapter.fetch(DataWindow.from_values("2026-06-17T09:00:00Z", "2026-06-17T11:00:00Z"))
    frame = result.silver

    assert len(frame) == 1
    assert result.provenance.dataset_id == NATIONAL_DATASET_ID
    assert result.provenance.published_at == datetime(2026, 6, 17, 10, 10, tzinfo=timezone.utc)
    assert result.provenance.query["content_sha256"]
    assert result.provenance.query["source_page_url"].endswith(f"/{NATIONAL_DATASET_ID}/")
    assert "consommation is not null" not in result.provenance.query["query_params"]["where"]

    row = frame.iloc[0]
    assert row["source_name"] == "odre_eco2mix_national"
    assert row["dataset_id"] == NATIONAL_DATASET_ID
    assert row["consumption_mw"] == 50000
    assert row["wind_mw"] == 4000
    assert row["exports_mw"] == 6900
    assert row["rte_forecast_j_mw"] == 50500
    assert row["rte_forecast_j1_mw"] == 51000
    assert row["source_page_url"].endswith(f"/{NATIONAL_DATASET_ID}/")

    _, kwargs = session.calls[0]
    assert kwargs["params"]["order_by"] == "date_heure asc"
    assert 'date_heure >= "2026-06-17T09:00:00Z"' in kwargs["params"]["where"]
    assert 'date_heure < "2026-06-17T11:00:00Z"' in kwargs["params"]["where"]


def test_regional_adapter_preserves_local_generation_and_physical_balance() -> None:
    adapter = RegionalEco2MixAdapter(http=fixture_http(fixture("odre_eco2mix_regional.json")))

    frame = adapter.fetch(
        DataWindow.from_values("2026-06-17T09:00:00Z", "2026-06-17T11:00:00Z")
    ).silver

    assert set(frame["source_record_id"]) == {"Ile-de-France", "Bretagne"}
    assert set(frame["dataset_id"]) == {REGIONAL_DATASET_ID}
    brittany = frame.loc[frame["source_record_id"] == "Bretagne"].iloc[0]
    assert brittany["total_production_mw"] == 1420
    assert brittany["imports_mw"] == 1080
    assert brittany["physical_balance_mw"] == 0
    assert frame["physical_balance_mw"].tolist() == [0, 0]


def test_national_forecast_only_record_is_kept_when_observed_demand_is_missing() -> None:
    payload = fixture("odre_eco2mix_national.json")
    record = dict(payload["results"][0])
    record["consommation"] = None
    record["nucleaire"] = None
    record["eolien"] = None
    record["solaire"] = None
    record["hydraulique"] = None
    record["gaz"] = None
    record["charbon"] = None
    record["fioul"] = None
    record["bioenergies"] = None
    record["ech_physiques"] = None
    record["taux_co2"] = None
    payload["results"] = [record]

    frame = NationalEco2MixAdapter(http=fixture_http(payload)).fetch(
        DataWindow.from_values("2026-06-17T09:00:00Z", "2026-06-17T11:00:00Z")
    ).silver

    assert len(frame) == 1
    assert frame["consumption_mw"].isna().all()
    assert frame["rte_forecast_j_mw"].iloc[0] == 50500
    assert set(frame["quality_status"]) == {"ok"}


def test_validation_error_is_odre_specific_and_still_source_schema_error() -> None:
    adapter = NationalEco2MixAdapter(http=fixture_http({"total_count": 1, "records": []}))

    with pytest.raises(ODRESourceSchemaError, match="results") as raised:
        adapter.fetch(DataWindow.from_values("2026-06-17", "2026-06-18"))
    assert isinstance(raised.value, SourceSchemaError)


def test_session_injection_and_http_errors_stay_offline_safe() -> None:
    session = Session(Response({}, status_code=503, status_error=requests.HTTPError("service unavailable")))
    adapter = RegionalEco2MixAdapter(session=session, config=AdapterConfig(max_retries=0, backoff_base_seconds=0))

    with pytest.raises(SourceUnavailableError, match="service unavailable"):
        adapter.fetch(DataWindow.from_values("2026-06-17", "2026-06-18"))
