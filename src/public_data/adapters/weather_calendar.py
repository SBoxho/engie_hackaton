"""Open-Meteo and French calendar public-data adapters."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from src.public_data.contracts import (
    AdapterConfig,
    AdapterResult,
    DataWindow,
    FallbackStatus,
    Provenance,
    PublicDataError,
    QualityStatus,
    SourceSchemaError,
)
from src.public_data.http import PublicDataHttpClient
from src.public_data.time import local_midnight_to_utc

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PUBLIC_HOLIDAYS_URL_TEMPLATE = "https://calendrier.api.gouv.fr/jours-feries/{territory}/{year}.json"
SCHOOL_CALENDAR_RECORDS_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/records"
)

OPEN_METEO_HOURLY = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
)
OPEN_METEO_CURRENT = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
)
OPEN_METEO_RENAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_kmh",
    "cloud_cover": "cloud_cover_pct",
    "shortwave_radiation": "solar_radiation_wm2",
}
SCHOOL_ZONE_COLUMNS = ("A", "B", "C")
PUBLIC_HOLIDAY_TERRITORIES = {
    "alsace-moselle",
    "guadeloupe",
    "guyane",
    "martinique",
    "mayotte",
    "metropole",
    "nouvelle-caledonie",
    "polynesie-francaise",
    "reunion",
    "saint-barthelemy",
    "saint-martin",
    "saint-pierre-et-miquelon",
    "wallis-et-futuna",
}
SCHOOL_FIELD_ALIASES = {
    "annee_scolaire": "school_year",
    "date_de_debut": "start_date",
    "date_de_fin": "end_date",
    "debut": "start_date",
    "description": "description",
    "end_date": "end_date",
    "fin": "end_date",
    "lieu": "location",
    "location": "location",
    "population": "population",
    "published_at": "published_at",
    "record_timestamp": "published_at",
    "start_date": "start_date",
    "zone": "zones",
    "zones": "zones",
}

OPEN_METEO_LICENSE = "Open-Meteo free public API terms"
FRENCH_OPEN_DATA_LICENSE = "Licence Ouverte / Etalab compatible public open data"


class WeatherCalendarError(PublicDataError):
    """Base class for weather/calendar adapter failures."""


class SourceValidationError(WeatherCalendarError):
    """Raised when a source URL, territory, or location is unsupported."""


class OpenMeteoWeatherError(SourceSchemaError, WeatherCalendarError):
    """Raised for Open-Meteo response or validation failures."""


class FrenchPublicHolidayError(SourceSchemaError, WeatherCalendarError):
    """Raised for French public-holiday response or validation failures."""


class FrenchSchoolHolidayError(SourceSchemaError, WeatherCalendarError):
    """Raised for French school-calendar response or validation failures."""


OpenMeteoAdapterError = OpenMeteoWeatherError
FrenchPublicHolidayAdapterError = FrenchPublicHolidayError


@dataclass(frozen=True)
class FunctionalWindow:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class FunctionalProvenance:
    source_id: str
    record_count: int
    window: FunctionalWindow
    published_at: datetime | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class FunctionalFetchResult:
    records: pd.DataFrame
    provenance: FunctionalProvenance


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= float(latitude) <= 90:
        raise SourceValidationError("latitude must be between -90 and 90")
    if not -180 <= float(longitude) <= 180:
        raise SourceValidationError("longitude must be between -180 and 180")


@dataclass(frozen=True)
class WeatherLocation:
    id: str
    name: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not self.id:
            raise SourceValidationError("weather locations require a stable id")
        _validate_coordinates(self.latitude, self.longitude)


DEFAULT_WEATHER_LOCATIONS = (
    WeatherLocation("paris", "Paris", 48.8566, 2.3522),
)


def validate_source_url(
    url: str,
    *,
    source_name: str | None = None,
    source_id: str | None = None,
    allowed_hosts: Sequence[str],
    allowed_path_prefixes: Sequence[str] = (),
) -> None:
    """Fail closed when an adapter is pointed outside its official public source."""
    label = source_name or source_id or "source"
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SourceValidationError(f"{label} must use https")
    if parsed.username or parsed.password:
        raise SourceValidationError(f"{label} source URL must not include credentials")
    host = parsed.hostname or ""
    if host not in set(allowed_hosts):
        raise SourceValidationError(f"{label} unsupported host: {host}")
    if allowed_path_prefixes and not any(parsed.path.startswith(prefix) for prefix in allowed_path_prefixes):
        raise SourceValidationError(f"{label} unsupported path: {parsed.path}")


def incremental_windows(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    chunk_days: int | None = None,
    max_days: int | None = None,
    mode: str = "window",
) -> tuple[DataWindow, ...]:
    """Build half-open UTC windows for incremental source backfills."""
    chunk_days = chunk_days or max_days or 31
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)
    if start_ts >= end_ts:
        raise ValueError("start must be earlier than end")
    windows: list[DataWindow] = []
    cursor = start_ts
    while cursor < end_ts:
        boundary = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        windows.append(
            DataWindow.from_values(
                cursor,
                boundary,
                mode=mode,
                chunk_id=f"{cursor.date()}_{boundary.date()}",
            )
        )
        cursor = boundary
    return tuple(windows)


def open_meteo_incremental_windows(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    chunk_days: int = 31,
    mode: str = "archive",
) -> tuple[DataWindow, ...]:
    return incremental_windows(start, end, chunk_days=chunk_days, mode=mode)


def calendar_year_windows(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    mode: str = "calendar",
) -> tuple[DataWindow, ...]:
    start_date, end_date = _window_date_values(start, end)
    windows: list[DataWindow] = []
    cursor = start_date
    while cursor < end_date:
        boundary = min(date(cursor.year + 1, 1, 1), end_date)
        windows.append(
            DataWindow.from_values(
                pd.Timestamp(cursor).tz_localize("UTC"),
                pd.Timestamp(boundary).tz_localize("UTC"),
                mode=mode,
                chunk_id=f"{cursor.isoformat()}_{boundary.isoformat()}",
            )
        )
        cursor = boundary
    return tuple(windows)


def _date_string(value: pd.Timestamp | None, default: date) -> str:
    if value is None:
        return default.isoformat()
    return value.tz_convert("UTC").date().isoformat()


def _inclusive_end_date(value: pd.Timestamp | None, default: date) -> str:
    if value is None:
        return default.isoformat()
    return (value.tz_convert("UTC") - pd.Timedelta(nanoseconds=1)).date().isoformat()


class OpenMeteoWeatherAdapter:
    name = "open_meteo_weather"
    source_revision = "open-meteo:v1"
    forecast_url = FORECAST_URL
    archive_url = ARCHIVE_URL
    historical_forecast_url = HISTORICAL_FORECAST_URL

    def __init__(
        self,
        *,
        locations: tuple[WeatherLocation, ...] = DEFAULT_WEATHER_LOCATIONS,
        http: PublicDataHttpClient | None = None,
        config: AdapterConfig | None = None,
        forecast_url: str = FORECAST_URL,
        archive_url: str = ARCHIVE_URL,
        historical_forecast_url: str = HISTORICAL_FORECAST_URL,
    ) -> None:
        if not locations:
            raise SourceValidationError("Open-Meteo adapter requires at least one location")
        if len({location.id for location in locations}) != len(locations):
            raise SourceValidationError("Open-Meteo locations require unique ids")
        self.locations = locations
        self.http = http or PublicDataHttpClient(config=config, source_name=self.name)
        self.forecast_url = forecast_url
        self.archive_url = archive_url
        self.historical_forecast_url = historical_forecast_url
        self._validate_sources()

    def _validate_sources(self) -> None:
        validate_source_url(
            self.forecast_url,
            source_name=self.name,
            allowed_hosts=("api.open-meteo.com",),
            allowed_path_prefixes=("/v1/forecast",),
        )
        validate_source_url(
            self.archive_url,
            source_name=self.name,
            allowed_hosts=("archive-api.open-meteo.com",),
            allowed_path_prefixes=("/v1/archive",),
        )
        validate_source_url(
            self.historical_forecast_url,
            source_name=self.name,
            allowed_hosts=("historical-forecast-api.open-meteo.com",),
            allowed_path_prefixes=("/v1/forecast",),
        )

    def _url(self, window: DataWindow) -> str:
        today = datetime.now(timezone.utc).date()
        if window.mode == "historical_forecast":
            return self.historical_forecast_url
        if window.mode == "archive":
            return self.archive_url
        if window.end is not None and window.end.date() < today and window.mode in {"backfill", "history"}:
            return self.archive_url
        return self.forecast_url

    def _params(self, location: WeatherLocation, window: DataWindow) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date()
        params: dict[str, Any] = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": "UTC",
            "hourly": ",".join(OPEN_METEO_HOURLY),
            "wind_speed_unit": "kmh",
        }
        if window.mode == "current":
            params["current"] = ",".join(OPEN_METEO_CURRENT)
            params["forecast_days"] = 2
            return params
        params["start_date"] = _date_string(window.start, today)
        params["end_date"] = _inclusive_end_date(window.end, today)
        return params

    @staticmethod
    def _normalize_hourly(
        payload: dict[str, Any],
        location: WeatherLocation,
        provenance: Provenance,
        *,
        window: DataWindow | None,
    ) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
            raise OpenMeteoWeatherError("Open-Meteo response has no hourly time array")
        times = hourly["time"]
        rows = pd.DataFrame({"event_time": times})
        for name in OPEN_METEO_HOURLY:
            values = hourly.get(name)
            if values is None:
                raise OpenMeteoWeatherError(f"Open-Meteo response missing hourly field: {name}")
            if not isinstance(values, list) or len(values) != len(times):
                raise OpenMeteoWeatherError(f"Open-Meteo hourly field length mismatch: {name}")
            rows[name] = values
        rows = rows.rename(columns=OPEN_METEO_RENAMES)
        rows["event_time"] = pd.to_datetime(rows["event_time"], utc=True, errors="coerce")
        if window is not None:
            if window.start is not None:
                rows = rows.loc[rows["event_time"] >= window.start]
            if window.end is not None:
                rows = rows.loc[rows["event_time"] < window.end]
        rows = _attach_common_columns(rows, provenance, source_record_id=location.id)
        rows["source_record_id"] = location.id
        rows["location_name"] = location.name
        rows["latitude"] = location.latitude
        rows["longitude"] = location.longitude
        rows.loc[rows["event_time"].isna(), "quality_status"] = QualityStatus.SCHEMA_FAILURE.value
        return rows.reset_index(drop=True)

    @staticmethod
    def _normalize_current(payload: dict[str, Any], location: WeatherLocation, provenance: Provenance) -> pd.DataFrame:
        current = payload.get("current")
        if not isinstance(current, dict) or "time" not in current:
            raise OpenMeteoWeatherError("Open-Meteo current response has no current time")
        missing = [name for name in OPEN_METEO_CURRENT if name not in current]
        if missing:
            raise OpenMeteoWeatherError(f"Open-Meteo current response missing fields: {missing}")
        row = {
            "event_time": pd.to_datetime(current.get("time"), utc=True, errors="coerce"),
            "source_record_id": location.id,
            "location_name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "interval_seconds": current.get("interval"),
        }
        frame = _attach_common_columns(pd.DataFrame([row]), provenance, source_record_id=location.id)
        frame["source_name"] = "open_meteo_current"
        frame.loc[frame["event_time"].isna(), "quality_status"] = QualityStatus.SCHEMA_FAILURE.value
        return frame

    def fetch(self, window: DataWindow) -> AdapterResult:
        ingested_at = datetime.now(timezone.utc)
        payloads: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []
        url = self._url(window)
        for location in self.locations:
            params = self._params(location, window)
            payload = self.http.get_json(url, params=params)
            if not isinstance(payload, dict) or payload.get("error") is True:
                raise OpenMeteoWeatherError(f"Open-Meteo error response: {payload}")
            payloads.append({"location": location.__dict__, "params": params, "response": payload})
            provenance = Provenance(
                source_name=self.name,
                source_revision=self.source_revision,
                source_url=url,
                dataset_id=None,
                ingested_at=ingested_at,
                published_at=_extract_published_at(payload),
                query={**window.query_metadata(), "location_id": location.id, "params": params},
                license=OPEN_METEO_LICENSE,
            )
            hourly_window = None if window.mode == "current" else window
            frames.append(self._normalize_hourly(payload, location, provenance, window=hourly_window))
            if window.mode == "current":
                frames.append(self._normalize_current(payload, location, provenance))
        silver = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        aggregate_query = {
            **window.query_metadata(),
            "location_ids": [location.id for location in self.locations],
            "content_sha256": _content_sha256(payloads),
            "source_limitations": [
                "Current weather is served by the Open-Meteo forecast endpoint.",
                "Forecast rows are target weather times, not measured observations.",
                "Historical forecast and archive availability depends on Open-Meteo endpoint support.",
            ],
        }
        return AdapterResult(
            source_name=self.name,
            source_revision=self.source_revision,
            bronze_payload=payloads,
            silver=silver,
            provenance=Provenance(
                source_name=self.name,
                source_revision=self.source_revision,
                source_url=url,
                dataset_id=None,
                ingested_at=ingested_at,
                published_at=_extract_published_at(payloads),
                query=aggregate_query,
                license=OPEN_METEO_LICENSE,
            ),
        )


class FrenchPublicHolidayAdapter:
    name = "french_public_holidays"
    url_template = PUBLIC_HOLIDAYS_URL_TEMPLATE

    def __init__(
        self,
        *,
        territory: str = "metropole",
        http: PublicDataHttpClient | None = None,
        config: AdapterConfig | None = None,
        url_template: str = PUBLIC_HOLIDAYS_URL_TEMPLATE,
    ) -> None:
        if territory not in PUBLIC_HOLIDAY_TERRITORIES:
            raise FrenchPublicHolidayError(f"unsupported public-holiday territory: {territory}")
        self.territory = territory
        self.http = http or PublicDataHttpClient(config=config, source_name=self.name)
        self.url_template = url_template

    @property
    def source_revision(self) -> str:
        return f"calendrier.api.gouv.fr:{self.territory}"

    @staticmethod
    def _years(window: DataWindow) -> range:
        start_date, end_date = _window_date_bounds(window)
        end_inclusive = end_date - pd.Timedelta(days=1)
        return range(int(start_date.year), int(end_inclusive.year) + 1)

    def _url(self, year: int) -> str:
        url = self.url_template.format(territory=self.territory, year=year)
        validate_source_url(
            url,
            source_name=self.name,
            allowed_hosts=("calendrier.api.gouv.fr",),
            allowed_path_prefixes=("/jours-feries",),
        )
        return url

    def fetch(self, window: DataWindow) -> AdapterResult:
        ingested_at = datetime.now(timezone.utc)
        start_date, end_date = _window_date_bounds(window)
        payloads: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for year in self._years(window):
            url = self._url(year)
            payload = self.http.get_json(url)
            if not isinstance(payload, dict):
                raise FrenchPublicHolidayError("French public holidays response must be a JSON object")
            payloads[str(year)] = payload
            for day, label in payload.items():
                event_date = _parse_source_date(day, FrenchPublicHolidayError)
                if event_date < start_date or event_date >= end_date:
                    continue
                rows.append(
                    {
                        "event_time": local_midnight_to_utc(event_date),
                        "published_at": pd.NaT,
                        "ingested_at": pd.Timestamp(ingested_at),
                        "source_name": self.name,
                        "source_revision": self.source_revision,
                        "quality_status": QualityStatus.OK.value,
                        "fallback_status": FallbackStatus.NONE.value,
                        "source_record_id": f"{self.territory}:{event_date.isoformat()}",
                        "holiday_name": str(label),
                        "is_public_holiday": 1,
                        "territory": self.territory,
                    }
                )
        query = {
            **window.query_metadata(),
            "territory": self.territory,
            "content_sha256": _content_sha256(payloads),
            "source_limitations": [
                "The API returns official holiday dates by territory but no per-response publication timestamp.",
            ],
        }
        provenance = Provenance(
            source_name=self.name,
            source_revision=self.source_revision,
            source_url=self.url_template,
            dataset_id=None,
            ingested_at=ingested_at,
            published_at=_extract_published_at(payloads),
            query=query,
            license=FRENCH_OPEN_DATA_LICENSE,
        )
        return AdapterResult(
            source_name=self.name,
            source_revision=self.source_revision,
            bronze_payload=payloads,
            silver=pd.DataFrame(rows),
            provenance=provenance,
        )


class FrenchSchoolHolidayAdapter:
    name = "french_school_holidays"
    source_revision = "fr-en-calendrier-scolaire:explore-v2.1"
    dataset_id = "fr-en-calendrier-scolaire"
    records_url = SCHOOL_CALENDAR_RECORDS_URL

    def __init__(
        self,
        *,
        http: PublicDataHttpClient | None = None,
        config: AdapterConfig | None = None,
        records_url: str = SCHOOL_CALENDAR_RECORDS_URL,
        page_size: int = 100,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.http = http or PublicDataHttpClient(config=config, source_name=self.name)
        self.records_url = records_url
        self.page_size = page_size
        validate_source_url(
            self.records_url,
            source_name=self.name,
            allowed_hosts=("data.education.gouv.fr",),
            allowed_path_prefixes=("/api/explore/v2.1/catalog/datasets/",),
        )

    def _where(self, window: DataWindow) -> str | None:
        if window.start is None or window.end is None:
            return None
        start, end = _window_date_bounds(window)
        return f'start_date < "{end.isoformat()}" AND end_date >= "{start.isoformat()}"'

    def fetch(self, window: DataWindow) -> AdapterResult:
        ingested_at = datetime.now(timezone.utc)
        start_date, end_date = _window_date_bounds(window)
        offset = 0
        pages: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {"limit": self.page_size, "offset": offset, "order_by": "start_date asc"}
            where = self._where(window)
            if where:
                params["where"] = where
            payload = self.http.get_json(self.records_url, params=params)
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise FrenchSchoolHolidayError("school holiday response has no valid results array")
            pages.append(payload)
            batch = payload["results"]
            for raw_record in batch:
                if not isinstance(raw_record, dict):
                    continue
                record = _canonical_school_record(raw_record)
                school_start = _parse_optional_date(record.get("start_date"))
                school_end = _parse_optional_date(record.get("end_date"))
                if school_start is None or school_end is None:
                    continue
                if school_start >= end_date or school_end <= start_date:
                    continue
                for zone in _school_zones(record.get("zones")):
                    rows.append(
                        {
                            "event_time": local_midnight_to_utc(school_start),
                            "published_at": _extract_published_at(record) or pd.NaT,
                            "ingested_at": pd.Timestamp(ingested_at),
                            "source_name": self.name,
                            "source_revision": self.source_revision,
                            "quality_status": QualityStatus.OK.value,
                            "fallback_status": FallbackStatus.NONE.value,
                            "source_record_id": f"{zone}:{school_start.isoformat()}:{school_end.isoformat()}",
                            "zone": zone,
                            "start_date": school_start.isoformat(),
                            "end_date": school_end.isoformat(),
                            "description": str(record.get("description") or ""),
                            "location": str(record.get("location") or ""),
                            "population": str(record.get("population") or ""),
                            "school_year": str(record.get("school_year") or ""),
                            "is_school_holiday": 1,
                        }
                    )
            total = int(payload.get("total_count", len(batch)))
            offset += len(batch)
            if not batch or offset >= total:
                break
        query = {
            **window.query_metadata(),
            "dataset_id": self.dataset_id,
            "content_sha256": _content_sha256({"pages": pages}),
            "source_limitations": [
                "School holiday intervals are start-inclusive and end-exclusive in downstream feature use.",
                "Zones are expanded from official Zone A/B/C labels when present; missing zone labels imply all zones.",
            ],
        }
        provenance = Provenance(
            source_name=self.name,
            source_revision=self.source_revision,
            source_url=self.records_url,
            dataset_id=self.dataset_id,
            ingested_at=ingested_at,
            published_at=_extract_published_at({"pages": pages}),
            query=query,
            license=FRENCH_OPEN_DATA_LICENSE,
        )
        return AdapterResult(
            source_name=self.name,
            source_revision=self.source_revision,
            bronze_payload={"pages": pages},
            silver=pd.DataFrame(rows),
            provenance=provenance,
        )


def _attach_common_columns(frame: pd.DataFrame, provenance: Provenance, *, source_record_id: str) -> pd.DataFrame:
    rows = frame.copy()
    rows["published_at"] = provenance.published_at or pd.NaT
    rows["ingested_at"] = pd.Timestamp(provenance.ingested_at)
    rows["source_name"] = provenance.source_name
    rows["source_revision"] = provenance.source_revision
    rows["quality_status"] = QualityStatus.OK.value
    rows["fallback_status"] = FallbackStatus.NONE.value
    rows["source_record_id"] = source_record_id
    return rows


def _utc_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _window_date_values(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
) -> tuple[date, date]:
    start_date = _utc_timestamp(start).date()
    end_date = _utc_timestamp(end).date()
    if start_date >= end_date:
        raise ValueError("start date must be earlier than end date")
    return start_date, end_date


def _window_date_bounds(window: DataWindow) -> tuple[date, date]:
    start = window.start or pd.Timestamp(datetime.now(timezone.utc)).floor("D")
    end = window.end or (pd.Timestamp(start) + pd.Timedelta(days=1))
    if start >= end:
        raise ValueError("window start must be earlier than end")
    return start.date(), end.date()


def _parse_source_date(value: Any, error_cls: type[WeatherCalendarError]) -> date:
    parsed = _parse_optional_date(value)
    if parsed is None:
        raise error_cls(f"invalid source date: {value!r}")
    return parsed


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _content_sha256(payload: Any) -> str:
    content = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_published_at(payload: Any) -> datetime | None:
    keys = {
        "data_processed_at",
        "date_maj",
        "date_publication",
        "last_modified",
        "metadata_modified",
        "modified",
        "published_at",
        "record_timestamp",
        "updated_at",
    }
    stack = [payload]
    seen = 0
    while stack and seen < 200:
        seen += 1
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, value in item.items():
                if str(key) in keys:
                    parsed = pd.to_datetime(value, utc=True, errors="coerce")
                    if pd.notna(parsed):
                        return pd.Timestamp(parsed).to_pydatetime()
                elif isinstance(value, (Mapping, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(value for value in item if isinstance(value, (Mapping, list)))
    return None


def _field_id(column: object) -> str:
    text = unicodedata.normalize("NFKD", str(column)).encode("ascii", "ignore").decode()
    return "_".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def _canonical_school_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {SCHOOL_FIELD_ALIASES.get(_field_id(key), _field_id(key)): value for key, value in record.items()}


def _school_zones(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = "" if value is None or pd.isna(value) else str(value)
    normalized = _field_id(text).replace("_", " ")
    zones = tuple(zone for zone in SCHOOL_ZONE_COLUMNS if f"zone {zone.lower()}" in normalized or normalized == zone.lower())
    return zones or SCHOOL_ZONE_COLUMNS


def _response_json(session: Any, url: str, *, params: dict[str, Any] | None = None) -> Any:
    response = session.get(url, params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def _open_meteo_params(latitude: float, longitude: float) -> dict[str, Any]:
    _validate_coordinates(latitude, longitude)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }


def _hourly_records(
    payload: Mapping[str, Any],
    *,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    run_kind: str,
) -> pd.DataFrame:
    window = DataWindow.from_values(start, end)
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping) or not isinstance(hourly.get("time"), list):
        raise OpenMeteoAdapterError("Open-Meteo response has no hourly time array")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True, errors="coerce")})
    for source, target in OPEN_METEO_RENAMES.items():
        values = hourly.get(source)
        if values is None:
            raise OpenMeteoAdapterError(f"Open-Meteo response missing hourly field: {source}")
        frame[target] = values
    frame = frame[(frame["timestamp"] >= window.start) & (frame["timestamp"] < window.end)].reset_index(drop=True)
    frame["published_at"] = _extract_published_at(payload)
    frame["run_kind"] = run_kind
    frame["adapter_run_kind"] = run_kind
    return frame


def fetch_open_meteo_current(
    latitude: float,
    longitude: float,
    *,
    session: Any,
    retrieved_at: datetime | None = None,
) -> FunctionalFetchResult:
    params = _open_meteo_params(latitude, longitude)
    params["current"] = ",".join(OPEN_METEO_CURRENT)
    payload = _response_json(session, FORECAST_URL, params=params)
    current = payload.get("current") if isinstance(payload, Mapping) else None
    if not isinstance(current, Mapping) or "time" not in current:
        raise OpenMeteoAdapterError("Open-Meteo current response has no current time")
    start = pd.to_datetime(current["time"], utc=True)
    interval = int(current.get("interval") or 900)
    records = pd.DataFrame(
        [
            {
                "timestamp": start,
                "temperature_c": current.get("temperature_2m"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "cloud_cover_pct": current.get("cloud_cover"),
                "published_at": _extract_published_at(payload),
                "run_kind": "current",
                "adapter_run_kind": "current",
            }
        ]
    )
    return FunctionalFetchResult(
        records=records,
        provenance=FunctionalProvenance(
            source_id="open_meteo_current",
            record_count=len(records),
            window=FunctionalWindow(start, start + pd.Timedelta(seconds=interval)),
            published_at=_extract_published_at(payload),
            extra={"retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat()},
        ),
    )


def fetch_open_meteo_forecast(
    latitude: float,
    longitude: float,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    *,
    session: Any,
    retrieved_at: datetime | None = None,
) -> FunctionalFetchResult:
    params = _open_meteo_params(latitude, longitude)
    start_ts, end_ts = _utc_timestamp(start), _utc_timestamp(end)
    params.update(
        {
            "hourly": ",".join(OPEN_METEO_HOURLY),
            "start_date": start_ts.date().isoformat(),
            "end_date": (end_ts - pd.Timedelta(nanoseconds=1)).date().isoformat(),
        }
    )
    payload = _response_json(session, FORECAST_URL, params=params)
    records = _hourly_records(payload, start=start, end=end, run_kind="forecast")
    return FunctionalFetchResult(
        records=records,
        provenance=FunctionalProvenance(
            source_id="open_meteo_forecast",
            record_count=len(records),
            window=FunctionalWindow(start_ts, end_ts),
            published_at=_extract_published_at(payload),
            extra={"retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat()},
        ),
    )


def fetch_open_meteo_archive(
    latitude: float,
    longitude: float,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    *,
    session: Any,
    retrieved_at: datetime | None = None,
) -> FunctionalFetchResult:
    payload = _response_json(
        session,
        ARCHIVE_URL,
        params={
            **_open_meteo_params(latitude, longitude),
            "hourly": ",".join(OPEN_METEO_HOURLY),
            "start_date": _utc_timestamp(start).date().isoformat(),
            "end_date": (_utc_timestamp(end) - pd.Timedelta(nanoseconds=1)).date().isoformat(),
        },
    )
    records = _hourly_records(payload, start=start, end=end, run_kind="archive")
    return FunctionalFetchResult(
        records=records,
        provenance=FunctionalProvenance(
            source_id="open_meteo_archive",
            record_count=len(records),
            window=FunctionalWindow(_utc_timestamp(start), _utc_timestamp(end)),
            published_at=_extract_published_at(payload),
            extra={"retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat()},
        ),
    )


def fetch_open_meteo_historical_forecast(
    latitude: float,
    longitude: float,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    *,
    session: Any,
    retrieved_at: datetime | None = None,
) -> FunctionalFetchResult:
    payload = _response_json(
        session,
        HISTORICAL_FORECAST_URL,
        params={
            **_open_meteo_params(latitude, longitude),
            "hourly": ",".join(OPEN_METEO_HOURLY),
            "start_date": _utc_timestamp(start).date().isoformat(),
            "end_date": (_utc_timestamp(end) - pd.Timedelta(nanoseconds=1)).date().isoformat(),
        },
    )
    records = _hourly_records(payload, start=start, end=end, run_kind="historical_forecast")
    return FunctionalFetchResult(
        records=records,
        provenance=FunctionalProvenance(
            source_id="open_meteo_historical_forecast",
            record_count=len(records),
            window=FunctionalWindow(_utc_timestamp(start), _utc_timestamp(end)),
            published_at=_extract_published_at(payload),
            extra={"retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat()},
        ),
    )


def fetch_french_public_holidays(
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    zone: str = "metropole",
    session: Any,
    retrieved_at: datetime | None = None,
) -> FunctionalFetchResult:
    if zone not in PUBLIC_HOLIDAY_TERRITORIES:
        raise FrenchPublicHolidayAdapterError(f"unsupported public-holiday zone: {zone}")
    start_date, end_date = _window_date_values(start, end)
    rows: list[dict[str, Any]] = []
    source_urls: list[str] = []
    for year in range(start_date.year, end_date.year + 1):
        url = PUBLIC_HOLIDAYS_URL_TEMPLATE.format(territory=zone, year=year)
        payload = _response_json(session, url)
        source_urls.append(url)
        if not isinstance(payload, Mapping):
            raise FrenchPublicHolidayAdapterError("French public holidays response must be a JSON object")
        for day, label in payload.items():
            event_date = _parse_source_date(day, FrenchPublicHolidayError)
            if start_date <= event_date < end_date:
                rows.append({"date": event_date, "name": str(label), "is_public_holiday": 1, "territory": zone})
    records = pd.DataFrame(rows)
    return FunctionalFetchResult(
        records=records,
        provenance=FunctionalProvenance(
            source_id="french_public_holidays",
            record_count=len(records),
            window=FunctionalWindow(_utc_timestamp(start), _utc_timestamp(end)),
            extra={
                "source_urls": source_urls,
                "retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat(),
            },
        ),
    )


def fetch_french_school_holidays(
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    page_size: int = 100,
    session: Any,
    retrieved_at: datetime | None = None,
) -> FunctionalFetchResult:
    start_date, end_date = _window_date_values(start, end)
    rows: list[dict[str, Any]] = []
    published_values: list[datetime] = []
    offset = 0
    while True:
        payload = _response_json(
            session,
            SCHOOL_CALENDAR_RECORDS_URL,
            params={
                "limit": page_size,
                "offset": offset,
                "where": f'start_date < "{end_date.isoformat()}" AND end_date >= "{start_date.isoformat()}"',
                "order_by": "start_date asc",
            },
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise FrenchSchoolHolidayError("school holiday response has no valid results array")
        batch = payload["results"]
        for raw in batch:
            record = _canonical_school_record(raw)
            published = _extract_published_at(record)
            if published is not None:
                published_values.append(published)
            holiday_start = _parse_optional_date(record.get("start_date"))
            holiday_end = _parse_optional_date(record.get("end_date"))
            if holiday_start is None or holiday_end is None:
                continue
            if holiday_start >= end_date or holiday_end < start_date:
                continue
            for zone in _school_zones(record.get("zones")):
                rows.append(
                    {
                        "start_date": holiday_start,
                        "end_date": holiday_end,
                        "zone": zone.lower(),
                        "description": str(record.get("description") or ""),
                        "is_school_holiday": 1,
                    }
                )
        offset += len(batch)
        if not batch or offset >= int(payload.get("total_count", len(batch))):
            break
    records = pd.DataFrame(rows)
    return FunctionalFetchResult(
        records=records,
        provenance=FunctionalProvenance(
            source_id="french_school_holidays",
            record_count=len(records),
            window=FunctionalWindow(_utc_timestamp(start), _utc_timestamp(end)),
            published_at=max(published_values) if published_values else None,
            extra={"retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat()},
        ),
    )
