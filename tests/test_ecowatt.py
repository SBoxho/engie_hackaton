from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import settings
from src.data_sources.ecowatt import (
    EcoWattError,
    fetch_historical_ecowatt,
    load_cached_ecowatt,
    normalize_ecowatt_records,
    normalize_ecowatt_status,
    status_at,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(next(self.pages))


class FailingSession:
    def get(self, *args, **kwargs):
        raise OSError("offline")


def current_record(date="2026-06-17"):
    return {
        "date": date,
        "couleur_du_jour": 1,
        "h0": 0,
        "h1": 1,
        "h2": 2,
        "h3": 3,
        "message": "Test EcoWatt signal.",
    }


def test_normalizes_ecowatt_values_to_public_statuses():
    assert normalize_ecowatt_status(1) == ("green", "Normal", 1)
    assert normalize_ecowatt_status(2) == ("orange", "Tense", 2)
    assert normalize_ecowatt_status(3) == ("red", "Very tense", 3)
    assert normalize_ecowatt_status(None) == ("unknown", "Unknown", 0)


def test_normalizes_current_hourly_history():
    frame = normalize_ecowatt_records(
        [current_record()],
        dataset_id=settings.ecowatt_current_dataset_id,
        timezone_name="Europe/Paris",
    )

    assert len(frame) == 24
    assert frame["ecowatt_status"].head(4).tolist() == ["green", "green", "orange", "red"]
    assert set(frame["ecowatt_source"]) == {"EcoWatt ODRE current history"}


def test_fetches_and_caches_current_history(tmp_path: Path):
    session = Session([
        {"total_count": 1, "results": [current_record()]},
    ])

    frame = fetch_historical_ecowatt(
        "2026-06-17T00:00:00Z",
        "2026-06-18T00:00:00Z",
        include_legacy=False,
        cache_dir=tmp_path,
        session=session,
        timezone_name="Europe/Paris",
    )

    assert len(frame) == 24
    assert settings.ecowatt_current_dataset_id in session.calls[0][0]
    assert list(tmp_path.glob("ecowatt_*.json"))
    cached = load_cached_ecowatt(cache_dir=tmp_path, timezone_name="Europe/Paris")
    assert len(cached) == 24


def test_status_at_returns_unknown_when_signal_is_stale():
    frame = normalize_ecowatt_records(
        [current_record("2026-06-16")],
        dataset_id=settings.ecowatt_current_dataset_id,
        timezone_name="Europe/Paris",
    )

    signal = status_at(frame, pd.Timestamp("2026-06-17T12:00:00Z"))

    assert signal["ecowatt_status"] == "unknown"


def test_invalid_cache_payload_is_explicit(tmp_path: Path):
    path = tmp_path / "ecowatt_bad.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(EcoWattError, match="invalid EcoWatt cache"):
        load_cached_ecowatt(path)
