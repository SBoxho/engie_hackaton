"""Versioned status-threshold configuration for modelled balance context."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from src.contracts.energy_twin import Status


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data" / "config" / "status_thresholds.json"


@lru_cache(maxsize=1)
def load_status_thresholds() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload.get("version"):
        raise ValueError("status threshold config requires a version")
    return payload


def threshold_config_version() -> str:
    return str(load_status_thresholds()["version"])


def balance_status_for_ratio(ratio: float | None) -> Status:
    if ratio is None:
        return Status.UNKNOWN
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return Status.UNKNOWN
    if value < 0:
        return Status.UNKNOWN
    thresholds = load_status_thresholds()["balance_pressure_ratio"]
    if value >= float(thresholds["high_min"]):
        return Status.HIGH
    if value >= float(thresholds["watch_min"]):
        return Status.WATCH
    return Status.NORMAL


def score_status(score: float | None) -> Status:
    if score is None:
        return Status.UNKNOWN
    try:
        value = float(score)
    except (TypeError, ValueError):
        return Status.UNKNOWN
    if value < 0:
        return Status.UNKNOWN
    thresholds = load_status_thresholds()["visual_pressure_score"]
    if value >= float(thresholds["high_min"]):
        return Status.HIGH
    if value >= float(thresholds["watch_min"]):
        return Status.WATCH
    return Status.NORMAL


def modelled_balance_status_for_score(score: float | None) -> Status:
    if score is None:
        return Status.UNKNOWN
    try:
        value = float(score)
    except (TypeError, ValueError):
        return Status.UNKNOWN
    if value < 0:
        return Status.UNKNOWN
    thresholds = load_status_thresholds().get("modelled_balance_context", {}).get("score_thresholds", {})
    if value >= float(thresholds.get("high_min", 0.82)):
        return Status.HIGH
    if value >= float(thresholds.get("watch_min", 0.62)):
        return Status.WATCH
    return Status.NORMAL


def status_label(status: Status | str) -> str:
    value = status if isinstance(status, Status) else Status(str(status).lower())
    return value.value.title()
