"""EcoWatt grid tension signals from public RTE/ODRE sources.

The current public historical dataset is ``nouveau_signal_ecowatt``. A legacy
regional dataset, ``signal-ecowatt``, is also supported for older demos. The
optional RTE live API path is deliberately lightweight: if it is unavailable or
not authenticated, callers can fall back to ODRE/cache without failing the app.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from src.config import settings
from src.utils.io import latest_file, read_json, timestamped_path, write_json
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

PAGE_SIZE = 100
CURRENT_SOURCE = "EcoWatt ODRE current history"
LEGACY_SOURCE = "EcoWatt ODRE legacy regional history"
LIVE_SOURCE = "RTE EcoWatt API"

STATUS_SEVERITY = {"unknown": 0, "green": 1, "orange": 2, "red": 3}
STATUS_LABELS = {
    "green": "Normal",
    "orange": "Tense",
    "red": "Very tense",
    "unknown": "Unknown",
}
LEGACY_PERIODS = {
    "nuit": range(0, 6),
    "matin": range(6, 12),
    "apres_midi": range(12, 18),
    "soir": range(18, 24),
}


class EcoWattError(RuntimeError):
    """Raised when EcoWatt data cannot be fetched or normalized."""


def records_url(dataset_id: str, base_url: str | None = None) -> str:
    base = (base_url or settings.odre_base_url).rstrip("/")
    return f"{base}/catalog/datasets/{dataset_id}/records"


def source_attribution() -> dict[str, str]:
    return {
        "current_history": settings.ecowatt_current_odre_url,
        "legacy_history": settings.ecowatt_legacy_odre_url,
        "data_gouv": settings.ecowatt_current_data_gouv_url,
        "rte_live_api": settings.rte_ecowatt_api_url,
    }


def _utc(value: str | date | datetime | pd.Timestamp) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _iso_date(value: datetime) -> str:
    return value.date().isoformat()


def _status_from_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    try:
        numeric = int(float(text.replace(",", ".")))
    except ValueError:
        numeric = None
    if numeric is not None:
        if numeric <= 1:
            return "green"
        if numeric == 2:
            return "orange"
        if numeric >= 3:
            return "red"

    if "rouge" in text or "red" in text or "tres tendu" in text or "très tendu" in text:
        return "red"
    if "orange" in text or "tendu" in text or "tense" in text:
        return "orange"
    if "vert" in text or "green" in text or "raisonnable" in text or "pas d" in text:
        return "green"
    return "unknown"


def normalize_ecowatt_status(value: Any) -> tuple[str, str, int]:
    status = _status_from_value(value)
    return status, STATUS_LABELS[status], STATUS_SEVERITY[status]


def _local_hour_to_utc(day: Any, hour: int, timezone_name: str) -> pd.Timestamp:
    local_midnight = pd.Timestamp.combine(pd.Timestamp(day).date(), time.min).tz_localize(
        timezone_name,
        nonexistent="shift_forward",
        ambiguous=True,
    )
    return (local_midnight + pd.Timedelta(hours=hour)).tz_convert("UTC")


def _finalize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "timestamp",
        "ecowatt_status",
        "ecowatt_label",
        "ecowatt_severity",
        "ecowatt_message",
        "ecowatt_source",
        "ecowatt_dataset_id",
        "ecowatt_source_url",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["ecowatt_severity"] = pd.to_numeric(frame["ecowatt_severity"], errors="coerce").fillna(0)
    frame["dataset_priority"] = frame["ecowatt_dataset_id"].map(
        {settings.ecowatt_current_dataset_id: 0, settings.ecowatt_legacy_dataset_id: 1}
    ).fillna(2)
    frame = frame.sort_values(
        ["timestamp", "ecowatt_severity", "dataset_priority"],
        ascending=[True, False, True],
    ).drop_duplicates("timestamp", keep="first")
    return frame[columns].sort_values("timestamp").reset_index(drop=True)


def _normalize_current(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        day = record.get("date")
        if day is None or pd.isna(day):
            continue
        day_status, _, day_severity = normalize_ecowatt_status(record.get("couleur_du_jour"))
        for hour in range(24):
            raw_hour = record.get(f"h{hour}")
            hour_status, _, hour_severity = normalize_ecowatt_status(raw_hour)
            status = hour_status if hour_severity > day_severity else day_status
            if raw_hour is None or str(raw_hour).strip() == "" or pd.isna(raw_hour):
                status = day_status
            label = STATUS_LABELS[status]
            rows.append(
                {
                    "timestamp": _local_hour_to_utc(day, hour, timezone_name),
                    "ecowatt_status": status,
                    "ecowatt_label": label,
                    "ecowatt_severity": STATUS_SEVERITY[status],
                    "ecowatt_message": record.get("message") or "",
                    "ecowatt_source": CURRENT_SOURCE,
                    "ecowatt_dataset_id": settings.ecowatt_current_dataset_id,
                    "ecowatt_source_url": settings.ecowatt_current_odre_url,
                }
            )
    return _finalize(rows)


def _normalize_legacy(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        day = record.get("date")
        if day is None or pd.isna(day):
            continue
        region = record.get("region") or "unknown region"
        for period, hours in LEGACY_PERIODS.items():
            status, label, severity = normalize_ecowatt_status(record.get(period))
            message = record.get(f"situation_{period}") or ""
            for hour in hours:
                rows.append(
                    {
                        "timestamp": _local_hour_to_utc(day, hour, timezone_name),
                        "ecowatt_status": status,
                        "ecowatt_label": label,
                        "ecowatt_severity": severity,
                        "ecowatt_message": f"{region}: {message}".strip(),
                        "ecowatt_source": LEGACY_SOURCE,
                        "ecowatt_dataset_id": settings.ecowatt_legacy_dataset_id,
                        "ecowatt_source_url": settings.ecowatt_legacy_odre_url,
                    }
                )
    return _finalize(rows)


def normalize_ecowatt_records(
    records: Iterable[dict[str, Any]] | pd.DataFrame,
    *,
    dataset_id: str,
    timezone_name: str | None = None,
) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)
    if frame.empty:
        return _finalize([])
    tz = timezone_name or settings.timezone
    if dataset_id == settings.ecowatt_current_dataset_id:
        return _normalize_current(frame, tz)
    if dataset_id == settings.ecowatt_legacy_dataset_id:
        return _normalize_legacy(frame, tz)
    raise EcoWattError(f"unsupported EcoWatt dataset: {dataset_id}")


def _fetch_dataset_records(
    dataset_id: str,
    start: datetime,
    end: datetime,
    *,
    session: requests.Session,
    timeout: int,
) -> list[dict[str, Any]]:
    if start >= end:
        raise ValueError("start must be earlier than end")
    where = f'date >= "{_iso_date(start)}" AND date <= "{_iso_date(end)}"'
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "where": where,
            "order_by": "date asc",
        }
        try:
            response = session.get(records_url(dataset_id), params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise EcoWattError(f"failed to fetch EcoWatt dataset {dataset_id}: {exc}") from exc
        batch = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(batch, list):
            raise EcoWattError(f"EcoWatt dataset {dataset_id} response has no valid results array")
        records.extend(batch)
        total = int(payload.get("total_count", len(batch)))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return records


def _cache_payload(
    payloads: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> Path:
    path = timestamped_path(cache_dir, "ecowatt")
    return write_json(
        {
            "source": source_attribution(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "query": {"start": start.isoformat(), "end": end.isoformat()},
            "datasets": payloads,
        },
        path,
    )


def fetch_historical_ecowatt(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    include_legacy: bool = True,
    cache: bool = True,
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
    timezone_name: str | None = None,
) -> pd.DataFrame:
    """Fetch public EcoWatt history from ODRE and return hourly normalized status."""
    start_utc = _utc(start)
    end_utc = _utc(end)
    client = session or requests.Session()
    datasets = [settings.ecowatt_current_dataset_id]
    if include_legacy:
        datasets.append(settings.ecowatt_legacy_dataset_id)

    payloads: list[dict[str, Any]] = []
    normalized: list[pd.DataFrame] = []
    for dataset_id in datasets:
        records = _fetch_dataset_records(dataset_id, start_utc, end_utc, session=client, timeout=timeout)
        payloads.append({"dataset_id": dataset_id, "results": records})
        normalized.append(
            normalize_ecowatt_records(records, dataset_id=dataset_id, timezone_name=timezone_name)
        )

    if cache:
        target = cache_dir or settings.ecowatt_cache_dir
        path = _cache_payload(payloads, start_utc, end_utc, target)
        LOGGER.info("Cached EcoWatt payload at %s", path)
    return _finalize(pd.concat(normalized, ignore_index=True).to_dict(orient="records"))


def load_cached_ecowatt(
    path: Path | None = None,
    *,
    cache_dir: Path | None = None,
    timezone_name: str | None = None,
) -> pd.DataFrame:
    directory = cache_dir or settings.ecowatt_cache_dir
    path = path or latest_file(directory, "ecowatt_*.json")
    if path is None or not path.exists():
        raise FileNotFoundError("no cached EcoWatt snapshot found")
    payload = read_json(path)
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, list):
        raise EcoWattError("invalid EcoWatt cache payload")
    normalized = [
        normalize_ecowatt_records(
            item.get("results", []),
            dataset_id=str(item.get("dataset_id")),
            timezone_name=timezone_name,
        )
        for item in datasets
    ]
    return _finalize(pd.concat(normalized, ignore_index=True).to_dict(orient="records"))


def _records_from_live_payload(payload: Any) -> list[dict[str, Any]]:
    signals = payload.get("signals", payload) if isinstance(payload, dict) else payload
    if not isinstance(signals, list):
        return []
    records: list[dict[str, Any]] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        record: dict[str, Any] = {
            "date": signal.get("date") or signal.get("jour") or signal.get("day"),
            "couleur_du_jour": signal.get("couleur_du_jour")
            or signal.get("dvalue")
            or signal.get("value"),
            "message": signal.get("message") or signal.get("message_fr") or "",
        }
        values = signal.get("values") or signal.get("hourly")
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, dict):
                    continue
                hour_value = item.get("pas") or item.get("hour") or item.get("start_date")
                try:
                    hour = pd.Timestamp(hour_value).hour if "-" in str(hour_value) else int(str(hour_value).split(":")[0])
                except (TypeError, ValueError):
                    continue
                record[f"h{hour}"] = item.get("hvalue") or item.get("value") or item.get("signal")
        for hour in range(24):
            if f"h{hour}" in signal:
                record[f"h{hour}"] = signal[f"h{hour}"]
        if record["date"] is not None:
            records.append(record)
    return records


def fetch_live_ecowatt(
    *,
    session: requests.Session | None = None,
    timeout: int = 15,
    timezone_name: str | None = None,
) -> pd.DataFrame:
    """Fetch the optional RTE EcoWatt API and normalize it when credentials work."""
    client = session or requests.Session()
    headers = {"Accept": "application/json"}
    if settings.rte_ecowatt_api_token:
        headers["Authorization"] = f"Bearer {settings.rte_ecowatt_api_token}"
    try:
        response = client.get(settings.rte_ecowatt_api_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise EcoWattError(f"failed to fetch live EcoWatt API: {exc}") from exc
    records = _records_from_live_payload(payload)
    return normalize_ecowatt_records(
        records,
        dataset_id=settings.ecowatt_current_dataset_id,
        timezone_name=timezone_name,
    )


def load_ecowatt_window(
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    prefer_live: bool = True,
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
    timezone_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load EcoWatt for a time window with live, ODRE, and cache fallbacks."""
    frames: list[pd.DataFrame] = []
    notes: list[str] = []
    if prefer_live and settings.rte_ecowatt_api_token:
        try:
            live = fetch_live_ecowatt(session=session, timeout=min(timeout, 15), timezone_name=timezone_name)
            if not live.empty:
                frames.append(live)
                notes.append("RTE EcoWatt API")
        except EcoWattError as exc:
            LOGGER.info("EcoWatt live API unavailable: %s", exc)

    try:
        historical = fetch_historical_ecowatt(
            start,
            end,
            cache=True,
            cache_dir=cache_dir,
            session=session,
            timeout=timeout,
            timezone_name=timezone_name,
        )
        if not historical.empty:
            frames.append(historical)
            notes.append("ODRE EcoWatt public history")
    except (EcoWattError, OSError, ValueError) as exc:
        LOGGER.info("EcoWatt ODRE fetch unavailable: %s", exc)
        try:
            cached = load_cached_ecowatt(cache_dir=cache_dir, timezone_name=timezone_name)
            if not cached.empty:
                frames.append(cached)
                notes.append("cached EcoWatt snapshot")
        except (EcoWattError, FileNotFoundError, OSError) as cache_exc:
            LOGGER.info("EcoWatt cache unavailable: %s", cache_exc)

    if not frames:
        return _finalize([]), "EcoWatt unavailable"
    combined = _finalize(pd.concat(frames, ignore_index=True).to_dict(orient="records"))
    start_utc, end_utc = pd.Timestamp(_utc(start)), pd.Timestamp(_utc(end))
    combined = combined.loc[combined["timestamp"].between(start_utc, end_utc)].copy()
    return combined, " + ".join(dict.fromkeys(notes))


def status_at(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=75),
) -> dict[str, Any]:
    """Return the EcoWatt status nearest to a timestamp, or an unknown contract."""
    unknown = {
        "ecowatt_status": "unknown",
        "ecowatt_label": STATUS_LABELS["unknown"],
        "ecowatt_severity": STATUS_SEVERITY["unknown"],
        "ecowatt_message": "No current EcoWatt signal is available.",
        "ecowatt_source": "Unavailable",
        "ecowatt_source_url": settings.ecowatt_current_odre_url,
    }
    if frame.empty or "timestamp" not in frame:
        return unknown
    target = pd.Timestamp(timestamp)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    if work.empty:
        return unknown
    nearest_index = (work["timestamp"] - target).abs().idxmin()
    nearest = work.loc[nearest_index]
    if abs(nearest["timestamp"] - target) > tolerance:
        return unknown
    return nearest.to_dict()
