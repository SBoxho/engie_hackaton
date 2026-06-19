"""Data-health reporting for public-data ingestion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.public_data.contracts import FallbackStatus, QualityStatus
from src.public_data.models import AdapterFailure, PublicDataSchema
from src.public_data.storage import CANONICAL_COLUMNS, PublicDataStore, read_jsonl


@dataclass(frozen=True)
class SourceHealth:
    source_name: str
    most_recent_event_time: str | None
    latest_observation_age_seconds: float | None
    missing_intervals: int
    duplicate_intervals: int
    schema_failures: int
    fallback_records: int
    adapter_failures: int
    missing_interval_details: list[dict[str, Any]] = field(default_factory=list)
    duplicate_interval_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class DataHealthReport:
    generated_at: str
    sources: list[SourceHealth]
    fallback_usage: dict[str, int] = field(default_factory=dict)
    adapter_failure_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "sources": [asdict(source) for source in self.sources],
            "fallback_usage": dict(self.fallback_usage),
            "adapter_failure_details": list(self.adapter_failure_details),
        }

    def source(self, source_name: str) -> SourceHealth:
        for source in self.sources:
            if source.source_name == source_name:
                return source
        raise KeyError(source_name)


@dataclass(frozen=True)
class SchemaFailure:
    check: str
    message: str
    count: int
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    records: pd.DataFrame
    schema_failures: tuple[SchemaFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.schema_failures


def _expected_missing(group: pd.DataFrame, cadence: pd.Timedelta) -> int:
    times = pd.to_datetime(group["event_time"], utc=True, errors="coerce").dropna()
    if len(times) <= 1:
        return 0
    unique = times.drop_duplicates().sort_values()
    deltas = unique.diff().dropna()
    return int(sum(max(int(delta / cadence) - 1, 0) for delta in deltas if delta > cadence))


def validate_public_records(
    frame: pd.DataFrame,
    schema: PublicDataSchema | None = None,
) -> ValidationResult:
    extra_required = tuple(schema.required_columns) if schema is not None else ()
    missing = tuple(sorted(set((*CANONICAL_COLUMNS, *extra_required)).difference(frame.columns)))
    if missing:
        return ValidationResult(
            pd.DataFrame(columns=list(frame.columns)),
            (
                SchemaFailure(
                    check="missing_columns",
                    message=f"records are missing required metadata columns: {list(missing)}",
                    count=len(frame),
                    columns=missing,
                ),
            ),
        )

    result = frame.copy()
    failures: list[SchemaFailure] = []
    for column in ("event_time", "published_at", "ingested_at"):
        parsed = pd.to_datetime(result[column], utc=True, errors="coerce")
        bad = parsed.isna()
        if bad.any():
            failures.append(
                SchemaFailure(
                    check=f"invalid_{column}",
                    message=f"{column} contains invalid or missing timestamps",
                    count=int(bad.sum()),
                    columns=(column,),
                )
            )
        result[column] = parsed

    for column in ("source_name", "source_revision"):
        missing_values = result[column].isna() | result[column].astype("string").str.strip().eq("")
        if missing_values.any():
            failures.append(
                SchemaFailure(
                    check=f"missing_{column}",
                    message=f"{column} contains missing values",
                    count=int(missing_values.sum()),
                    columns=(column,),
                )
            )

    bad_timestamp_columns = {
        column
        for failure in failures
        for column in failure.columns
        if column in {"event_time", "published_at", "ingested_at"}
    }
    if bad_timestamp_columns:
        valid_mask = pd.Series(True, index=result.index)
        for column in bad_timestamp_columns:
            valid_mask &= result[column].notna()
        result = result[valid_mask].reset_index(drop=True)
    return ValidationResult(result.reset_index(drop=True), tuple(failures))


def _missing_interval_details(
    group: pd.DataFrame,
    cadence: pd.Timedelta,
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    times = pd.to_datetime(group["event_time"], utc=True, errors="coerce").dropna()
    unique = times.drop_duplicates().sort_values()
    previous: pd.Timestamp | None = None
    for current in unique:
        if previous is not None:
            delta = current - previous
            if delta > cadence:
                details.append(
                    {
                        "source_name": source_name,
                        "start": (previous + cadence).isoformat(),
                        "end": (current - cadence).isoformat(),
                        "expected_interval_seconds": cadence.total_seconds(),
                        "missing_count": max(int(delta / cadence) - 1, 1),
                    }
                )
        previous = current
    return details


def analyze_frame_health(
    frame: pd.DataFrame,
    *,
    now: datetime | pd.Timestamp | None = None,
    expected_cadences: Mapping[str, str] | None = None,
    adapter_failure_counts: Mapping[str, int] | None = None,
) -> DataHealthReport:
    generated = pd.Timestamp(now or datetime.now(timezone.utc))
    if generated.tzinfo is None:
        generated = generated.tz_localize("UTC")
    else:
        generated = generated.tz_convert("UTC")
    expected_cadences = expected_cadences or {}
    adapter_failure_counts = adapter_failure_counts or {}
    if frame.empty:
        names = sorted(adapter_failure_counts)
        return DataHealthReport(
            generated.isoformat(),
            [
                SourceHealth(name, None, None, 0, 0, 0, 0, adapter_failure_counts.get(name, 0))
                for name in names
            ],
            {},
            [
                {"source_name": name, "count": int(count)}
                for name, count in sorted(adapter_failure_counts.items())
            ],
        )

    source = frame.copy()
    schema_validation = validate_public_records(source)
    for column in CANONICAL_COLUMNS:
        if column not in source:
            source[column] = pd.NA
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="coerce")
    fallback_usage = (
        source["fallback_status"].fillna(FallbackStatus.NONE.value).astype(str).value_counts().astype(int).to_dict()
    )

    reports: list[SourceHealth] = []
    for name, group in source.groupby("source_name", dropna=False):
        source_name = str(name)
        latest = group["event_time"].dropna().max()
        age = (generated - latest).total_seconds() if pd.notna(latest) else None
        duplicate_keys = group.duplicated(
            ["event_time", "source_name", "source_revision", "source_record_id"],
            keep=False,
        )
        cadence = pd.Timedelta(expected_cadences.get(source_name, "1h"))
        missing = _expected_missing(group, cadence)
        missing_details = _missing_interval_details(group, cadence, source_name=source_name)
        duplicate_details = [
            {
                "source_name": source_name,
                "event_time": pd.Timestamp(event_time).isoformat(),
                "count": int(len(duplicate_group)),
            }
            for event_time, duplicate_group in group[duplicate_keys].groupby("event_time", dropna=False)
        ]
        schema_failures = int((group["quality_status"] == QualityStatus.SCHEMA_FAILURE.value).sum())
        schema_failures += sum(failure.count for failure in schema_validation.schema_failures)
        fallback_records = int((group["fallback_status"] != FallbackStatus.NONE.value).sum())
        reports.append(
            SourceHealth(
                source_name=source_name,
                most_recent_event_time=latest.isoformat() if pd.notna(latest) else None,
                latest_observation_age_seconds=age,
                missing_intervals=missing,
                duplicate_intervals=int(duplicate_keys.sum()),
                schema_failures=schema_failures,
                fallback_records=fallback_records,
                adapter_failures=int(adapter_failure_counts.get(source_name, 0)),
                missing_interval_details=missing_details,
                duplicate_interval_details=duplicate_details,
            )
        )
    for source_name, count in adapter_failure_counts.items():
        if source_name not in {item.source_name for item in reports}:
            reports.append(SourceHealth(source_name, None, None, 0, 0, 0, 0, int(count)))
    return DataHealthReport(
        generated.isoformat(),
        sorted(reports, key=lambda item: item.source_name),
        {str(key): int(value) for key, value in fallback_usage.items()},
        [
            {"source_name": name, "count": int(count)}
            for name, count in sorted(adapter_failure_counts.items())
        ],
    )


def build_data_health(
    store: PublicDataStore,
    *,
    expected_cadences: Mapping[str, str] | None = None,
    now: datetime | pd.Timestamp | None = None,
) -> DataHealthReport:
    frame = store.silver.read()
    failures = read_jsonl(Path(store.failures_path))
    failure_counts: dict[str, int] = {}
    for failure in failures:
        source_name = str(failure.get("source_name", "unknown"))
        failure_counts[source_name] = failure_counts.get(source_name, 0) + 1
    report = analyze_frame_health(
        frame,
        now=now,
        expected_cadences=expected_cadences,
        adapter_failure_counts=failure_counts,
    )
    return DataHealthReport(
        generated_at=report.generated_at,
        sources=report.sources,
        fallback_usage=report.fallback_usage,
        adapter_failure_details=failures,
    )


@dataclass(frozen=True)
class DuplicateInterval:
    event_time: pd.Timestamp
    count: int


@dataclass(frozen=True)
class MissingInterval:
    start: pd.Timestamp
    missing_count: int


@dataclass(frozen=True)
class DetailedSourceHealth:
    source_name: str
    latest_event_time: pd.Timestamp | None
    latest_age: pd.Timedelta | None
    missing_intervals: tuple[MissingInterval, ...]
    duplicate_intervals: tuple[DuplicateInterval, ...]
    fallback_rows: int


@dataclass(frozen=True)
class DetailedHealthReport:
    sources: tuple[DetailedSourceHealth, ...]
    fallback_usage: dict[str, int]
    schema_failures: tuple[SchemaFailure, ...]
    adapter_failures: tuple[AdapterFailure, ...]

    def source(self, source_name: str) -> DetailedSourceHealth:
        for source in self.sources:
            if source.source_name == source_name:
                return source
        raise KeyError(source_name)


def build_data_health_report(
    frame: pd.DataFrame,
    *,
    expected_interval: str = "1h",
    now: datetime | pd.Timestamp | None = None,
    schema_failures: list[SchemaFailure] | tuple[SchemaFailure, ...] = (),
    adapter_failures: list[AdapterFailure] | tuple[AdapterFailure, ...] = (),
) -> DetailedHealthReport:
    checked_at = pd.Timestamp(now or datetime.now(timezone.utc))
    if checked_at.tzinfo is None:
        checked_at = checked_at.tz_localize("UTC")
    else:
        checked_at = checked_at.tz_convert("UTC")
    if frame.empty:
        return DetailedHealthReport((), {}, tuple(schema_failures), tuple(adapter_failures))
    source = frame.copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="coerce")
    cadence = pd.Timedelta(expected_interval)
    fallback_usage = (
        source.get("fallback_status", pd.Series(dtype=str)).fillna("primary").astype(str).value_counts().to_dict()
    )
    reports: list[DetailedSourceHealth] = []
    for source_name, group in source.groupby("source_name", dropna=False):
        group = group.sort_values("event_time")
        latest = group["event_time"].dropna().max()
        duplicates = tuple(
            DuplicateInterval(pd.Timestamp(event_time), int(count))
            for event_time, count in group.groupby("event_time", dropna=True).size().items()
            if count > 1
        )
        missing: list[MissingInterval] = []
        times = group["event_time"].dropna().drop_duplicates().sort_values().tolist()
        for previous, current in zip(times[:-1], times[1:]):
            delta = current - previous
            if delta > cadence:
                missing.append(
                    MissingInterval(
                        start=previous + cadence,
                        missing_count=max(int(delta / cadence) - 1, 0),
                    )
                )
        statuses = group.get("fallback_status", pd.Series(FallbackStatus.NONE.value, index=group.index)).fillna(
            FallbackStatus.NONE.value
        )
        fallback_rows = int((~statuses.astype(str).isin([FallbackStatus.NONE.value, "primary"])).sum())
        reports.append(
            DetailedSourceHealth(
                source_name=str(source_name),
                latest_event_time=latest if pd.notna(latest) else None,
                latest_age=(checked_at - latest) if pd.notna(latest) else None,
                missing_intervals=tuple(missing),
                duplicate_intervals=duplicates,
                fallback_rows=fallback_rows,
            )
        )
    return DetailedHealthReport(
        sources=tuple(reports),
        fallback_usage={str(key): int(value) for key, value in fallback_usage.items()},
        schema_failures=tuple(schema_failures),
        adapter_failures=tuple(adapter_failures),
    )
