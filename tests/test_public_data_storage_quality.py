from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.public_data import (
    AdapterResult,
    FallbackStatus,
    Provenance,
    PublicDataStore,
    QualityStatus,
    analyze_frame_health,
    paris_hourly_index_for_day,
    render_paris,
    validate_public_records,
)


def public_rows(times: list[str] | pd.DatetimeIndex, *, source: str = "fixture") -> pd.DataFrame:
    event_times = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "event_time": event_times,
            "published_at": event_times,
            "ingested_at": pd.Timestamp("2026-06-18T10:00:00Z"),
            "source_name": source,
            "source_revision": "rev-1",
            "quality_status": QualityStatus.OK.value,
            "fallback_status": FallbackStatus.NONE.value,
            "source_record_id": [str(index) for index in range(len(event_times))],
            "value_mw": list(range(50_000, 50_000 + len(event_times))),
        }
    )


def adapter_result(frame: pd.DataFrame, *, payload: dict[str, object] | None = None) -> AdapterResult:
    ingested_at = pd.Timestamp("2026-06-18T10:00:00Z").to_pydatetime()
    return AdapterResult(
        source_name="fixture",
        source_revision="rev-1",
        bronze_payload=payload or {"rows": len(frame)},
        silver=frame,
        provenance=Provenance(
            source_name="fixture",
            source_revision="rev-1",
            source_url="https://example.test/fixture",
            dataset_id="fixture",
            ingested_at=ingested_at,
            published_at=ingested_at,
        ),
    )


def test_paris_dst_23_and_25_hour_days_are_utc_stable() -> None:
    spring_forward = paris_hourly_index_for_day(date(2024, 3, 31))
    fall_back = paris_hourly_index_for_day(date(2024, 10, 27))

    assert len(spring_forward) == 23
    assert len(fall_back) == 25
    assert render_paris(spring_forward[0], "%Y-%m-%d %H:%M") == "2024-03-31 00:00"
    assert render_paris(fall_back[0], "%Y-%m-%d %H:%M") == "2024-10-27 00:00"

    report = analyze_frame_health(
        public_rows(spring_forward),
        expected_cadences={"fixture": "1h"},
        now=spring_forward[-1] + pd.Timedelta(hours=1),
    )
    assert report.sources[0].missing_intervals == 0


def test_health_report_detects_duplicates_missing_intervals_and_fallbacks() -> None:
    frame = public_rows(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T03:00:00Z",
        ]
    )
    frame.loc[2, "fallback_status"] = FallbackStatus.LAST_KNOWN_GOOD.value
    frame.loc[2, "source_record_id"] = frame.loc[1, "source_record_id"]

    report = analyze_frame_health(
        frame,
        expected_cadences={"fixture": "1h"},
        now=pd.Timestamp("2026-01-01T04:00:00Z"),
    )
    source = report.sources[0]

    assert source.most_recent_event_time == pd.Timestamp("2026-01-01T03:00:00Z").isoformat()
    assert source.latest_observation_age_seconds == 3600
    assert source.duplicate_intervals == 2
    assert source.duplicate_interval_details[0]["event_time"] == pd.Timestamp("2026-01-01T01:00:00Z").isoformat()
    assert source.duplicate_interval_details[0]["count"] == 2
    assert source.missing_intervals == 1
    assert source.missing_interval_details[0]["start"] == pd.Timestamp("2026-01-01T02:00:00Z").isoformat()
    assert source.missing_interval_details[0]["missing_count"] == 1
    assert source.fallback_records == 1
    assert report.fallback_usage[FallbackStatus.LAST_KNOWN_GOOD.value] == 1


def test_malformed_adapter_response_reports_schema_failures_without_live_api() -> None:
    malformed = pd.DataFrame(
        {
            "event_time": ["not-a-date"],
            "published_at": ["2026-01-01T00:00:00Z"],
            "ingested_at": ["2026-01-01T00:05:00Z"],
            "source_name": ["fixture"],
            "source_revision": ["rev-1"],
            "quality_status": [QualityStatus.OK.value],
            "fallback_status": [FallbackStatus.NONE.value],
            "source_record_id": ["bad"],
        }
    )

    result = validate_public_records(malformed)

    assert not result.ok
    assert result.records.empty
    checks = {failure.check for failure in result.schema_failures}
    assert checks == {"invalid_event_time"}
    assert result.schema_failures[0].columns == ("event_time",)


def test_health_report_carries_schema_and_adapter_failures() -> None:
    schema_failure = validate_public_records(
        public_rows(["2026-01-01T00:00:00Z"]).drop(columns=["published_at"])
    ).schema_failures[0]
    frame = public_rows(["2026-01-01T00:00:00Z"])
    frame.loc[0, "quality_status"] = QualityStatus.SCHEMA_FAILURE.value

    report = analyze_frame_health(
        frame,
        expected_cadences={"fixture": "1h"},
        adapter_failure_counts={"fixture": 1},
        now=pd.Timestamp("2026-01-01T01:00:00Z"),
    )

    assert schema_failure.check == "missing_columns"
    assert report.sources[0].schema_failures == 1
    assert report.sources[0].adapter_failures == 1
    assert report.adapter_failure_details == [{"source_name": "fixture", "count": 1}]


def test_parquet_store_is_layered_idempotent_and_supports_lkg_fallback(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    store = PublicDataStore(tmp_path)
    frame = public_rows(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"])

    first = store.write_result(adapter_result(frame)).silver
    rerun = store.write_result(adapter_result(frame)).silver

    assert first.inserted_rows == 2
    assert first.replaced_rows == 0
    assert rerun.inserted_rows == 0
    assert rerun.replaced_rows == 0
    assert rerun.unchanged_rows == 2
    assert len(store.silver.read()) == 2
    assert (tmp_path / "silver" / "source=fixture" / "year=2026" / "month=01" / "data.parquet").exists()
    assert (tmp_path / "bronze_index" / "source=fixture").exists()

    replacement = frame.iloc[[1]].copy()
    replacement["value_mw"] = 99_999
    changed = store.write_result(adapter_result(replacement, payload={"changed": True})).silver
    assert changed.replaced_rows == 1
    assert store.silver.read().sort_values("event_time")["value_mw"].tolist() == [50_000, 99_999]

    fallback = store.last_known_good_fallback(
        "fixture",
        reason="adapter failed in offline fixture",
        now=pd.Timestamp("2026-01-01T02:00:00Z"),
    )
    assert len(fallback) == 2
    assert set(fallback["fallback_status"]) == {FallbackStatus.LAST_KNOWN_GOOD.value}
    assert set(fallback["quality_status"]) == {QualityStatus.PARTIAL.value}
    assert set(fallback["fallback_reason"]) == {"adapter failed in offline fixture"}
