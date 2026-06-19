"""Lightweight in-process metrics for demo and small-host deployments."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class _MetricsState:
    requests_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    forecast_runs_total: int = 0
    scenario_runs_total: int = 0
    source_failures_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_request_at: str | None = None
    last_forecast_run_id: str | None = None
    last_scenario_id: str | None = None


_STATE = _MetricsState()
_LOCK = Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_request(path: str, method: str, status_code: int, duration_ms: float, error_category: str | None = None) -> None:
    key = f"{method} {path} {status_code}"
    with _LOCK:
        _STATE.requests_total[key] += 1
        _STATE.durations_ms.append(float(duration_ms))
        _STATE.last_request_at = _now_iso()
        if error_category:
            _STATE.errors_total[error_category] += 1


def record_forecast_run(run_id: str | None) -> None:
    with _LOCK:
        _STATE.forecast_runs_total += 1
        _STATE.last_forecast_run_id = run_id


def record_scenario_run(scenario_id: str | None) -> None:
    with _LOCK:
        _STATE.scenario_runs_total += 1
        _STATE.last_scenario_id = scenario_id


def record_source_failure(source_name: str) -> None:
    with _LOCK:
        _STATE.source_failures_total[str(source_name or "unknown")] += 1


def metrics_snapshot() -> dict[str, Any]:
    with _LOCK:
        durations = list(_STATE.durations_ms)
        if durations:
            ordered = sorted(durations)
            p95_index = max(min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1), 0)
            latency = {
                "count": len(ordered),
                "avg_ms": round(sum(ordered) / len(ordered), 3),
                "p95_ms": round(ordered[p95_index], 3),
                "max_ms": round(max(ordered), 3),
            }
        else:
            latency = {"count": 0, "avg_ms": None, "p95_ms": None, "max_ms": None}
        return {
            "generated_at": _now_iso(),
            "requests_total": dict(sorted(_STATE.requests_total.items())),
            "errors_total": dict(sorted(_STATE.errors_total.items())),
            "http_latency": latency,
            "forecast_runs_total": _STATE.forecast_runs_total,
            "scenario_runs_total": _STATE.scenario_runs_total,
            "source_failures_total": dict(sorted(_STATE.source_failures_total.items())),
            "last_request_at": _STATE.last_request_at,
            "last_forecast_run_id": _STATE.last_forecast_run_id,
            "last_scenario_id": _STATE.last_scenario_id,
        }
