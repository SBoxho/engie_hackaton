"""Shared contracts for public source adapters and ingestion layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Protocol

import pandas as pd


class QualityStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    SCHEMA_FAILURE = "schema_failure"
    SOURCE_FAILURE = "source_failure"


class FallbackStatus(StrEnum):
    NONE = "none"
    LAST_KNOWN_GOOD = "last_known_good"
    EMPTY = "empty"


class PublicDataError(RuntimeError):
    """Base class for source-specific public-data failures."""


class SourceSchemaError(PublicDataError):
    """Raised when a source response is syntactically valid but unusable."""


class SourceUnavailableError(PublicDataError):
    """Raised when a source cannot be reached or rate-limits the client."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class AdapterConfig:
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0
    min_interval_seconds: float = 0.0
    user_agent: str = "EnergyPulseFrance/0.1 public-data-ingestion"


@dataclass(frozen=True)
class DataWindow:
    """Half-open UTC time window requested from a source adapter."""

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    mode: str = "current"
    chunk_id: str | None = None

    @classmethod
    def from_values(
        cls,
        start: str | date | datetime | pd.Timestamp | None = None,
        end: str | date | datetime | pd.Timestamp | None = None,
        *,
        mode: str = "current",
        chunk_id: str | None = None,
    ) -> "DataWindow":
        start_ts = utc_timestamp(start)
        end_ts = utc_timestamp(end)
        if start_ts is not None and end_ts is not None and start_ts >= end_ts:
            raise ValueError("start must be earlier than end")
        return cls(start_ts, end_ts, mode, chunk_id)

    def query_metadata(self) -> dict[str, str | None]:
        return {
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "mode": self.mode,
            "chunk_id": self.chunk_id,
        }

    def as_dict(self) -> dict[str, str | None]:
        def fmt(value: pd.Timestamp | None) -> str | None:
            if value is None:
                return None
            return value.isoformat().replace("+00:00", "Z")

        return {"start": fmt(self.start), "end": fmt(self.end)}


@dataclass(frozen=True)
class Provenance:
    source_name: str
    source_revision: str
    source_url: str
    dataset_id: str | None
    ingested_at: datetime
    query: Mapping[str, Any] = field(default_factory=dict)
    published_at: datetime | None = None
    license: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_revision": self.source_revision,
            "source_url": self.source_url,
            "dataset_id": self.dataset_id,
            "ingested_at": self.ingested_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "query": dict(self.query),
            "license": self.license,
        }


@dataclass
class AdapterResult:
    source_name: str
    source_revision: str
    bronze_payload: Any
    silver: pd.DataFrame
    provenance: Provenance
    failures: tuple[str, ...] = ()
    fallback_status: FallbackStatus = FallbackStatus.NONE


class SourceAdapter(Protocol):
    name: str
    source_revision: str

    def fetch(self, window: DataWindow) -> AdapterResult:
        """Fetch and normalize a public source response."""
