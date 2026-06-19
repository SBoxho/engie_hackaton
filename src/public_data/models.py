"""Contracts shared by public-data adapters, storage, and quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd


class PublicDataLayer(str, Enum):
    """Logical storage layers used by the public-data ingestion foundation."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class QualityStatus(str, Enum):
    """Row-level validation status carried through every layer."""

    VALID = "valid"
    WARNING = "warning"
    SCHEMA_FAILURE = "schema_failure"
    ADAPTER_FAILURE = "adapter_failure"


class FallbackStatus(str, Enum):
    """How a row was produced when the preferred source path was unavailable."""

    PRIMARY = "primary"
    LAST_KNOWN_GOOD = "last_known_good"
    SYNTHETIC = "synthetic"
    UNAVAILABLE = "unavailable"


METADATA_COLUMNS: tuple[str, ...] = (
    "event_time",
    "published_at",
    "ingested_at",
    "source_name",
    "source_revision",
    "quality_status",
    "fallback_status",
)


@dataclass(frozen=True)
class PublicDataSchema:
    """Adapter-declared schema and cadence expectations.

    ``required_columns`` should list payload columns beyond the metadata columns
    defined above. ``key_columns`` must uniquely identify an observation for
    idempotent upserts; adapters with multiple metrics per timestamp should add
    their metric or location dimensions here.
    """

    source_name: str
    source_revision: str
    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
    key_columns: tuple[str, ...] = ("source_name", "event_time")
    cadence: timedelta | str | None = None
    layer: PublicDataLayer = PublicDataLayer.SILVER
    allow_extra_columns: bool = True

    @property
    def required_record_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*METADATA_COLUMNS, *self.required_columns)))


@dataclass(frozen=True)
class AdapterRequest:
    """Time-bounded request passed to source adapters."""

    start: datetime | None = None
    end: datetime | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterBatch:
    """Adapter output before storage.

    ``records`` should already be shaped as a table, but storage and validation
    will canonicalize metadata timestamps to UTC and reject malformed rows.
    """

    records: pd.DataFrame
    schema: PublicDataSchema
    raw_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class AdapterFailure:
    """Offline-safe record of a failed adapter fetch or parse attempt."""

    source_name: str
    source_revision: str
    failed_at: datetime
    error_type: str
    message: str
    layer: PublicDataLayer = PublicDataLayer.BRONZE

    def to_row(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_revision": self.source_revision,
            "failed_at": self.failed_at,
            "error_type": self.error_type,
            "message": self.message,
            "layer": self.layer.value,
        }


class SourceAdapter(Protocol):
    """Protocol implemented by public source adapters."""

    name: str
    revision: str
    schema: PublicDataSchema

    def fetch(self, request: AdapterRequest) -> AdapterBatch:
        """Fetch a bounded offline-testable batch without mutating storage."""

