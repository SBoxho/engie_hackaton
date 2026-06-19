"""ODRE eCO2mix adapters backed by the Opendatasoft public API."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

import pandas as pd
import requests

from src.config import settings
from src.public_data.contracts import (
    AdapterConfig,
    AdapterResult,
    DataWindow,
    FallbackStatus,
    Provenance,
    QualityStatus,
    SourceSchemaError,
    SourceUnavailableError,
)
from src.public_data.http import PublicDataHttpClient

PAGE_SIZE = 100
NATIONAL_DATASET_ID = "eco2mix-national-tr"
NATIONAL_HISTORY_DATASET_ID = "eco2mix-national-cons-def"
REGIONAL_DATASET_ID = "eco2mix-regional-tr"
ODRE_LICENSE = "Licence Ouverte / Etalab 2.0 via ODRE/Opendatasoft"
ODRE_SOURCE_LIMITATIONS = (
    "ODRE Explore API rows may lag RTE production time and do not always expose a record publication timestamp.",
    "Near-real-time eco2mix can include forecast rows where observed demand or production fields are null.",
)
METRIC_DEMAND = "demand_mw"
METRIC_GENERATION = "generation_mw"
METRIC_DEMAND_FORECAST = "demand_forecast_mw"
METRIC_NET_PHYSICAL_EXCHANGE = "net_physical_exchange_mw"
METRIC_REGIONAL_DEMAND = "regional_demand_mw"
METRIC_REGIONAL_GENERATION = "regional_generation_mw"
METRIC_LOCAL_GENERATION = "local_generation_mw"
METRIC_PHYSICAL_BALANCE = "physical_balance_mw"

GENERATION_MAP = {
    "consommation": "consumption_mw",
    "nucleaire": "nuclear_mw",
    "eolien": "wind_mw",
    "solaire": "solar_mw",
    "hydraulique": "hydro_mw",
    "gaz": "gas_mw",
    "charbon": "coal_mw",
    "fioul": "oil_mw",
    "bioenergies": "bioenergy_mw",
    "ech_physiques": "net_imports_mw",
    "taux_co2": "co2_intensity_g_per_kwh",
    "prevision_j": "rte_forecast_j_mw",
    "prevision_j1": "rte_forecast_j1_mw",
}
NATIONAL_REQUIRED = {
    "date_heure",
    "consommation",
    "nucleaire",
    "eolien",
    "solaire",
    "hydraulique",
    "gaz",
    "charbon",
    "bioenergies",
    "ech_physiques",
    "taux_co2",
}
REGIONAL_REQUIRED = NATIONAL_REQUIRED | {"perimetre"}
PUBLISHED_AT_FIELDS = (
    "published_at",
    "date_publication",
    "date_de_publication",
    "date_maj",
    "date_mise_a_jour",
    "last_update",
    "updated_at",
    "record_timestamp",
)


class ODREValidationError(SourceSchemaError):
    """Raised when an ODRE payload is syntactically valid but unusable."""


class ODREHTTPError(SourceUnavailableError):
    """Raised when an ODRE HTTP request fails."""


ODRESourceSchemaError = ODREValidationError


@dataclass(frozen=True)
class ODREMetricRecord:
    event_time: pd.Timestamp
    metric: str
    value: float
    region: str
    dimensions: dict[str, str]


@dataclass(frozen=True)
class ODREBatchProvenance:
    dataset_id: str
    source_url: str
    published_at: datetime | None


@dataclass(frozen=True)
class ODREMetricBatch:
    dataset_id: str
    provenance: ODREBatchProvenance
    records: tuple[ODREMetricRecord, ...]


def _records_url(dataset_id: str, base_url: str | None = None) -> str:
    base = (base_url or settings.odre_base_url).rstrip("/")
    return f"{base}/catalog/datasets/{dataset_id}/records"


def _source_page_url(dataset_id: str) -> str:
    return f"https://odre.opendatasoft.com/explore/dataset/{dataset_id}/"


def _iso(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _published_at(row: Mapping[str, Any]) -> pd.Timestamp | None:
    for key in PUBLISHED_AT_FIELDS:
        value = row.get(key)
        if value:
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.notna(parsed):
                return parsed
    return None


def _max_published_at(items: list[Mapping[str, Any]]) -> datetime | None:
    values = [_published_at(item) for item in items]
    populated = [value for value in values if value is not None]
    if not populated:
        return None
    return max(populated).to_pydatetime()


def _content_sha256(payload: Any) -> str:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class _OpendatasoftEco2MixAdapter:
    dataset_id: str
    name: str
    required_columns: set[str]
    http: PublicDataHttpClient | None = None
    config: AdapterConfig | None = None
    base_url: str | None = None
    session: requests.Session | None = None

    @property
    def source_revision(self) -> str:
        return f"{self.dataset_id}:explore-v2.1"

    @property
    def source_url(self) -> str:
        return _records_url(self.dataset_id, self.base_url)

    @property
    def source_page_url(self) -> str:
        return _source_page_url(self.dataset_id)

    def _where(self, window: DataWindow) -> str:
        clauses: list[str] = []
        if window.start is not None:
            clauses.append(f'date_heure >= "{_iso(window.start)}"')
        if window.end is not None:
            clauses.append(f'date_heure < "{_iso(window.end)}"')
        return " AND ".join(clauses)

    def query_params(self, window: DataWindow, *, offset: int = 0) -> dict[str, Any]:
        params: dict[str, Any] = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "order_by": "date_heure asc",
        }
        where = self._where(window)
        if where:
            params["where"] = where
        return params

    def incremental_query_params(
        self,
        *,
        cursor: str | datetime | pd.Timestamp | None = None,
        until: str | datetime | pd.Timestamp | None = None,
        limit: int = PAGE_SIZE,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        if cursor is not None:
            cursor_ts = pd.Timestamp(cursor)
            if cursor_ts.tzinfo is None:
                cursor_ts = cursor_ts.tz_localize("UTC")
            else:
                cursor_ts = cursor_ts.tz_convert("UTC")
            clauses.append(f'date_heure > "{_iso(cursor_ts)}"')
        if until is not None:
            until_ts = pd.Timestamp(until)
            if until_ts.tzinfo is None:
                until_ts = until_ts.tz_localize("UTC")
            else:
                until_ts = until_ts.tz_convert("UTC")
            clauses.append(f'date_heure < "{_iso(until_ts)}"')
        return {
            "limit": min(max(int(limit), 1), PAGE_SIZE),
            "offset": 0,
            "order_by": "date_heure asc",
            "where": " AND ".join(clauses),
        }

    def _fetch_pages(self, window: DataWindow) -> list[dict[str, Any]]:
        if self.session is not None:
            return self._fetch_pages_with_session(window)
        client = self.http or PublicDataHttpClient(config=self.config, source_name=self.name)
        offset = 0
        pages: list[dict[str, Any]] = []
        while True:
            payload = client.get_json(
                self.source_url,
                params=self.query_params(window, offset=offset),
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ODRESourceSchemaError(f"{self.name} response has no valid results array")
            pages.append(payload)
            batch = payload["results"]
            total = int(payload.get("total_count", len(batch)))
            offset += len(batch)
            if not batch or offset >= total:
                break
        return pages

    def _fetch_pages_with_session(self, window: DataWindow) -> list[dict[str, Any]]:
        offset = 0
        pages: list[dict[str, Any]] = []
        timeout = (self.config or AdapterConfig()).timeout_seconds
        while True:
            params = self.query_params(window, offset=offset)
            try:
                response = self.session.get(self.source_url, params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                raise ODREHTTPError(f"{self.dataset_id}: {exc}") from exc
            except ValueError as exc:
                raise ODREValidationError(f"{self.dataset_id}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ODREValidationError(f"{self.name} response has no valid results array")
            pages.append(payload)
            batch = payload["results"]
            total = int(payload.get("total_count", len(batch)))
            offset += len(batch)
            if not batch or offset >= total:
                break
        return pages

    def _records(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [record for page in pages for record in page.get("results", [])]

    def _validate_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ODRESourceSchemaError(f"{self.name} returned no records")
        available = set().union(*(record.keys() for record in records))
        missing = sorted(self.required_columns.difference(available))
        if missing:
            raise ODRESourceSchemaError(f"{self.name} missing fields: {missing}")

    def _normalize(self, records: list[dict[str, Any]], provenance: Provenance) -> pd.DataFrame:
        self._validate_records(records)
        raw = pd.DataFrame.from_records(records)
        frame = raw.rename(columns={k: v for k, v in GENERATION_MAP.items() if k in raw}).copy()
        frame["event_time"] = pd.to_datetime(raw["date_heure"], utc=True, errors="coerce")
        frame["published_at"] = pd.Series([_published_at(row) for row in records], dtype="object")
        frame["ingested_at"] = pd.Timestamp(provenance.ingested_at)
        frame["source_name"] = self.name
        frame["source_revision"] = provenance.source_revision
        frame["quality_status"] = QualityStatus.OK.value
        frame["fallback_status"] = FallbackStatus.NONE.value
        frame["region"] = raw.get("perimetre", "France")
        frame["source_record_id"] = frame["region"].fillna("France").astype(str)
        frame["dataset_id"] = self.dataset_id
        frame["source_url"] = self.source_url
        frame["source_page_url"] = self.source_page_url
        frame["source_license"] = ODRE_LICENSE
        numeric = [column for column in GENERATION_MAP.values() if column in frame]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "oil_mw" not in frame:
            frame["oil_mw"] = 0.0
        frame["imports_mw"] = frame["net_imports_mw"].clip(lower=0)
        frame["exports_mw"] = (-frame["net_imports_mw"]).clip(lower=0)
        generation_columns = [
            "nuclear_mw",
            "wind_mw",
            "solar_mw",
            "hydro_mw",
            "gas_mw",
            "coal_mw",
            "oil_mw",
            "bioenergy_mw",
        ]
        for column in generation_columns:
            if column not in frame:
                frame[column] = pd.NA
        frame["total_production_mw"] = frame[generation_columns].sum(axis=1, min_count=1)
        frame["physical_balance_mw"] = (
            frame["total_production_mw"]
            + frame["imports_mw"]
            - frame["exports_mw"]
            - frame["consumption_mw"]
        )
        forecast_columns = [column for column in ("rte_forecast_j_mw", "rte_forecast_j1_mw") if column in frame]
        if forecast_columns:
            has_forecast = frame[forecast_columns].notna().any(axis=1)
        else:
            has_forecast = pd.Series(False, index=frame.index)
        invalid = frame["event_time"].isna() | (frame["consumption_mw"].isna() & ~has_forecast)
        frame.loc[invalid, "quality_status"] = QualityStatus.SCHEMA_FAILURE.value
        keep = [
            "event_time",
            "published_at",
            "ingested_at",
            "source_name",
            "source_revision",
            "quality_status",
            "fallback_status",
            "source_record_id",
            "dataset_id",
            "source_url",
            "source_page_url",
            "source_license",
            "region",
            "consumption_mw",
            *generation_columns,
            "total_production_mw",
            "imports_mw",
            "exports_mw",
            "net_imports_mw",
            "physical_balance_mw",
            "co2_intensity_g_per_kwh",
        ]
        keep.extend(column for column in ("rte_forecast_j_mw", "rte_forecast_j1_mw") if column in frame)
        return frame[keep].sort_values(["event_time", "source_record_id"]).reset_index(drop=True)

    def _fetch_result(self, window: DataWindow) -> AdapterResult:
        ingested_at = datetime.now(timezone.utc)
        pages = self._fetch_pages(window)
        records = self._records(pages)
        query = window.query_metadata()
        query.update(
            {
                "api_url": self.source_url,
                "source_page_url": self.source_page_url,
                "query_params": self.query_params(window, offset=0),
                "record_count": len(records),
                "content_sha256": _content_sha256(records),
                "limitations": list(ODRE_SOURCE_LIMITATIONS),
            }
        )
        provenance = Provenance(
            source_name=self.name,
            source_revision=self.source_revision,
            source_url=self.source_url,
            dataset_id=self.dataset_id,
            ingested_at=ingested_at,
            published_at=_max_published_at([*records, *pages]),
            query=query,
            license=ODRE_LICENSE,
        )
        silver = self._normalize(records, provenance)
        return AdapterResult(
            source_name=self.name,
            source_revision=self.source_revision,
            bronze_payload={"pages": pages},
            silver=silver,
            provenance=provenance,
        )

    def fetch(
        self,
        window: DataWindow | datetime | pd.Timestamp | None = None,
        end: datetime | pd.Timestamp | None = None,
    ) -> AdapterResult | ODREMetricBatch:
        if isinstance(window, DataWindow):
            return self._fetch_result(window)
        request_window = DataWindow.from_values(window, end)
        result = self._fetch_result(request_window)
        return self._metric_batch(result)

    def _metric_batch(self, result: AdapterResult) -> ODREMetricBatch:
        rows = result.silver
        records: list[ODREMetricRecord] = []
        source_columns = {
            "nuclear": "nuclear_mw",
            "wind": "wind_mw",
            "solar": "solar_mw",
            "hydro": "hydro_mw",
            "gas": "gas_mw",
            "coal": "coal_mw",
            "oil": "oil_mw",
            "bioenergy": "bioenergy_mw",
        }
        for row in rows.itertuples(index=False):
            event_time = pd.Timestamp(row.event_time)
            region = str(row.region)
            if self.dataset_id == NATIONAL_DATASET_ID:
                demand_metric = METRIC_DEMAND
                generation_metric = METRIC_GENERATION
            else:
                demand_metric = METRIC_REGIONAL_DEMAND
                generation_metric = METRIC_REGIONAL_GENERATION
            if pd.notna(getattr(row, "consumption_mw", pd.NA)):
                records.append(
                    ODREMetricRecord(event_time, demand_metric, float(row.consumption_mw), region, {})
                )
            for source, column in source_columns.items():
                value = getattr(row, column, pd.NA)
                if pd.notna(value):
                    records.append(
                        ODREMetricRecord(
                            event_time,
                            generation_metric,
                            float(value),
                            region,
                            {"source": source},
                        )
                    )
            if self.dataset_id == REGIONAL_DATASET_ID:
                local_generation = sum(
                    float(getattr(row, column, 0) or 0) for column in source_columns.values()
                )
                records.append(
                    ODREMetricRecord(event_time, METRIC_LOCAL_GENERATION, local_generation, region, {})
                )
            for horizon, column in (("D", "rte_forecast_j_mw"), ("D+1", "rte_forecast_j1_mw")):
                value = getattr(row, column, pd.NA)
                if pd.notna(value):
                    records.append(
                        ODREMetricRecord(
                            event_time,
                            METRIC_DEMAND_FORECAST,
                            float(value),
                            region,
                            {"forecast_horizon": horizon},
                        )
                    )
            if pd.notna(getattr(row, "net_imports_mw", pd.NA)):
                records.append(
                    ODREMetricRecord(
                        event_time,
                        METRIC_NET_PHYSICAL_EXCHANGE,
                        float(row.net_imports_mw),
                        region,
                        {},
                    )
                )
            if pd.notna(getattr(row, "physical_balance_mw", pd.NA)):
                records.append(
                    ODREMetricRecord(
                        event_time,
                        METRIC_PHYSICAL_BALANCE,
                        float(row.physical_balance_mw),
                        region,
                        {},
                    )
                )
        return ODREMetricBatch(
            dataset_id=self.dataset_id,
            provenance=ODREBatchProvenance(
                dataset_id=self.dataset_id,
                source_url=self.source_page_url,
                published_at=result.provenance.published_at,
            ),
            records=tuple(records),
        )


class NationalEco2MixAdapter(_OpendatasoftEco2MixAdapter):
    def __init__(
        self,
        *,
        http: PublicDataHttpClient | None = None,
        session: requests.Session | None = None,
        config: AdapterConfig | None = None,
        base_url: str | None = None,
        dataset_id: str = NATIONAL_DATASET_ID,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            name="odre_eco2mix_national",
            required_columns=NATIONAL_REQUIRED,
            http=http,
            config=config,
            base_url=base_url,
            session=session,
        )


class NationalEco2MixHistoryAdapter(_OpendatasoftEco2MixAdapter):
    def __init__(
        self,
        *,
        http: PublicDataHttpClient | None = None,
        session: requests.Session | None = None,
        config: AdapterConfig | None = None,
        base_url: str | None = None,
        dataset_id: str = NATIONAL_HISTORY_DATASET_ID,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            name="odre_eco2mix_national_history",
            required_columns=NATIONAL_REQUIRED,
            http=http,
            config=config,
            base_url=base_url,
            session=session,
        )


class RegionalEco2MixAdapter(_OpendatasoftEco2MixAdapter):
    def __init__(
        self,
        *,
        http: PublicDataHttpClient | None = None,
        session: requests.Session | None = None,
        config: AdapterConfig | None = None,
        base_url: str | None = None,
        dataset_id: str = REGIONAL_DATASET_ID,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            name="odre_eco2mix_regional",
            required_columns=REGIONAL_REQUIRED,
            http=http,
            config=config,
            base_url=base_url,
            session=session,
        )
