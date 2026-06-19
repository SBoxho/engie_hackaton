"""RTE éCO2mix data through the public ODRÉ Opendatasoft API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import settings
from src.public_data.contracts import SourceUnavailableError
from src.public_data.http import PublicDataHttpClient
from src.utils.io import latest_file, read_json, timestamped_path, write_json
from src.utils.logging import get_logger
from src.utils.time import iso_utc

LOGGER = get_logger(__name__)
DATASET_ID = "eco2mix-national-tr"
PAGE_SIZE = 100
REQUIRED_COLUMNS = {
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


class Eco2MixError(RuntimeError):
    """Raised when éCO2mix data cannot be fetched or validated."""


def _records_url() -> str:
    return f"{settings.odre_base_url}/catalog/datasets/{DATASET_ID}/records"


def _validate(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise Eco2MixError("The éCO2mix API returned no populated records.")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise Eco2MixError(f"éCO2mix response is missing columns: {sorted(missing)}")


def fetch_eco2mix(
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    history_hours: int | None = None,
    cache: bool = True,
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch populated national records and optionally cache the raw API payload.

    The public real-time dataset also contains future forecast rows whose observed
    fields are null. They are deliberately excluded from this observed-data feed.
    """
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(hours=history_hours or settings.history_hours)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("start must be earlier than end")

    where = (
        f'consommation is not null AND date_heure >= "{iso_utc(start)}" '
        f'AND date_heure <= "{iso_utc(end)}"'
    )
    client = session or requests.Session()
    http_client = None if session is not None else PublicDataHttpClient(source_name="rte_eco2mix_national")
    records: list[dict[str, Any]] = []
    offset = 0
    total = None

    while total is None or offset < total:
        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "where": where,
            "order_by": "date_heure asc",
        }
        try:
            if http_client is None:
                response = client.get(_records_url(), params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
            else:
                payload = http_client.get_json(_records_url(), params=params)
        except (requests.RequestException, SourceUnavailableError, ValueError) as exc:
            raise Eco2MixError(f"Failed to fetch éCO2mix data: {exc}") from exc

        batch = payload.get("results")
        if not isinstance(batch, list):
            raise Eco2MixError("éCO2mix response has no valid 'results' array.")
        total = int(payload.get("total_count", len(batch)))
        records.extend(batch)
        offset += len(batch)
        if not batch:
            break

    frame = pd.DataFrame.from_records(records)
    _validate(frame)
    if cache:
        target_dir = cache_dir or settings.raw_dir / "rte_eco2mix"
        path = timestamped_path(target_dir, "eco2mix_national")
        write_json(
            {
                "source": _records_url(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "query": {"start": iso_utc(start), "end": iso_utc(end)},
                "results": records,
            },
            path,
        )
        LOGGER.info("Cached %s raw éCO2mix records at %s", len(frame), path)
    return frame


def load_cached_eco2mix(path: Path | None = None, cache_dir: Path | None = None) -> pd.DataFrame:
    if path is None:
        directory = cache_dir or settings.raw_dir / "rte_eco2mix"
        path = latest_file(directory, "eco2mix_national_*.json")
    if path is None or not path.exists():
        raise FileNotFoundError("No cached éCO2mix file found. Run the fetch command first.")
    payload = read_json(path)
    records = payload.get("results", payload) if isinstance(payload, dict) else payload
    frame = pd.DataFrame.from_records(records)
    _validate(frame)
    LOGGER.info("Loaded %s cached éCO2mix records from %s", len(frame), path)
    return frame
