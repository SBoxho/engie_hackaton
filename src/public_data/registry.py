"""Adapter registry and orchestration helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.public_data.adapters import (
    FrenchPublicHolidayAdapter,
    FrenchSchoolHolidayAdapter,
    NationalEco2MixAdapter,
    NationalEco2MixHistoryAdapter,
    OpenMeteoWeatherAdapter,
    RegionalEco2MixAdapter,
)
from src.public_data.contracts import AdapterResult, DataWindow, PublicDataError, SourceAdapter
from src.public_data.storage import PublicDataStore


def default_adapters(*, mode: str = "current") -> dict[str, SourceAdapter]:
    national = (
        NationalEco2MixHistoryAdapter()
        if mode in {"history", "backfill"}
        else NationalEco2MixAdapter()
    )
    return {
        "odre-national": national,
        "odre-regional": RegionalEco2MixAdapter(),
        "open-meteo": OpenMeteoWeatherAdapter(),
        "public-holidays": FrenchPublicHolidayAdapter(),
        "school-holidays": FrenchSchoolHolidayAdapter(),
    }


def select_adapters(names: Iterable[str] | None = None, *, mode: str = "current") -> dict[str, SourceAdapter]:
    adapters = default_adapters(mode=mode)
    selected = list(names or ["all"])
    if "all" in selected:
        return adapters
    unknown = sorted(set(selected).difference(adapters))
    if unknown:
        raise ValueError(f"unknown public-data source(s): {unknown}")
    return {name: adapters[name] for name in selected}


def current_window(hours: int = 72) -> DataWindow:
    end = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    return DataWindow.from_values(end - pd.Timedelta(hours=hours), end, mode="current")


def history_window(start: str, end: str) -> DataWindow:
    return DataWindow.from_values(start, end, mode="history")


def iter_backfill_windows(start: str, end: str, *, chunk_days: int = 7) -> Iterable[DataWindow]:
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    cursor = start_ts
    while cursor < end_ts:
        boundary = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        yield DataWindow.from_values(cursor, boundary, mode="backfill", chunk_id=f"{cursor.date()}_{boundary.date()}")
        cursor = boundary


def ingest_window(
    store: PublicDataStore,
    window: DataWindow,
    *,
    source_names: Iterable[str] | None = None,
) -> list[tuple[str, AdapterResult | Exception]]:
    results: list[tuple[str, AdapterResult | Exception]] = []
    for name, adapter in select_adapters(source_names, mode=window.mode).items():
        try:
            result = adapter.fetch(window)
            store.write_result(result)
            results.append((name, result))
        except PublicDataError as exc:
            store.record_adapter_failure(name, str(exc))
            results.append((name, exc))
    return results


def default_store(root: str | Path | None = None) -> PublicDataStore:
    return PublicDataStore(root or Path("data") / "public")
