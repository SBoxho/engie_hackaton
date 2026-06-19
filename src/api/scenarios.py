"""Deterministic first-generation scenario engine.

The engine compares an unmodified TwinSnapshot sequence with a scenario
sequence. It is intentionally bounded: no dispatch optimization, no power-flow
model, no reserve-margin claim, and no verified emissions accounting.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from src.api.twin import TwinService, UNSUPPORTED_PHYSICAL_BEHAVIOURS, default_twin_service
from src.config import settings
from src.contracts.energy_twin import Status, TwinResponse, TwinSnapshot
from src.contracts.status_thresholds import modelled_balance_status_for_score, threshold_config_version
from src.data_sources.rte_eco2mix_regional import REGION_NAMES
from src.observability import record_scenario_run


SCENARIO_ENGINE_VERSION = "scenario-engine.v1"
SCENARIO_ASSUMPTION_VERSION = "scenario-assumptions.v1"

SUPPORTED_SCENARIOS = {
    "cold_snap": "cold_snap",
    "cold-snap": "cold_snap",
    "cold snap": "cold_snap",
    "generation_unavailability": "generation_unavailability",
    "generating_unit_unavailability": "generation_unavailability",
    "generating-unit-unavailability": "generation_unavailability",
    "outage": "generation_unavailability",
    "ev_charging_shift": "ev_charging_shift",
    "ev_shift": "ev_charging_shift",
    "ev charging shift": "ev_charging_shift",
}

DEFAULT_ASSUMPTIONS: dict[str, Any] = {
    "heating_sensitivity_mw_per_c": 900.0,
    "cold_snap_peak_multiplier": 1.15,
    "cold_snap_day_multiplier": 1.0,
    "cold_snap_offpeak_multiplier": 0.72,
    "flexible_generation_response_fraction_range": [0.25, 0.55],
    "import_export_response_fraction_of_remaining_range": [0.20, 0.70],
    "balance_score_pressure_multiplier": 2.0,
    "positive_response_carbon_g_per_kwh_range": [80.0, 550.0],
    "negative_response_avoided_carbon_g_per_kwh_range": [80.0, 550.0],
}

SCENARIO_CAVEATS = [
    "This is a directional public-decision scenario, not an operator dispatch forecast.",
    "Imports, exports, flexible generation, and unresolved residual are represented as ranges to avoid false precision.",
    "The engine does not make import requirement automatically equal to peak-demand change.",
    "Carbon is a plausible response-mix range, not verified emissions accounting.",
]
MAX_SCENARIO_FIELDS = 20
MAX_ASSUMPTIONS_BYTES = 4096
MAX_USER_LABEL_CHARS = 120
MAX_COLD_DELTA_C = 40.0
MAX_UNAVAILABLE_CAPACITY_MW = 30_000.0
MAX_EV_VEHICLES = 1_000_000.0
MAX_EV_ENERGY_KWH = 100.0


@dataclass(frozen=True)
class ScenarioWindow:
    start_time: datetime
    end_time: datetime
    duration_hours: float


def normalize_scenario_request(raw_request: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate a scenario request.

    The normalized request is stable JSON and is used to derive the request
    hash. Timestamps are stored in UTC ISO format.
    """

    if not isinstance(raw_request, Mapping):
        raise ValueError("scenario request must be a JSON object")
    if len(raw_request) > MAX_SCENARIO_FIELDS:
        raise ValueError(f"scenario request has too many top-level fields; maximum is {MAX_SCENARIO_FIELDS}")
    timezone_name = str(raw_request.get("timezone") or settings.timezone)
    scenario_type = _scenario_type(raw_request.get("scenario_type"))
    magnitude = _normalize_magnitude(scenario_type, raw_request.get("magnitude"))
    scope = _normalize_scope(raw_request.get("scope"), scenario_type)
    if "start_time" not in raw_request:
        raise ValueError("scenario request requires start_time")
    window = _scenario_window(raw_request, timezone_name)
    assumptions = raw_request.get("assumptions")
    if assumptions in (None, "", [], {}):
        raise ValueError("scenario request requires non-empty assumptions")
    if len(json.dumps(assumptions, ensure_ascii=True, default=str)) > MAX_ASSUMPTIONS_BYTES:
        raise ValueError(f"scenario assumptions exceed {MAX_ASSUMPTIONS_BYTES} bytes")

    baseline_from = _parse_timestamp(
        raw_request.get("baseline_from_time") or raw_request.get("from_time") or window.start_time,
        timezone_name,
        field_name="baseline_from_time",
    )
    if baseline_from > window.start_time:
        raise ValueError("baseline_from_time must not be after scenario start_time")
    hours = raw_request.get("hours")
    if hours is None:
        extra_hours = 24 if scenario_type == "ev_charging_shift" else 0
        hours = math.ceil((window.end_time - baseline_from).total_seconds() / 3600.0) + extra_hours
    hours = min(max(int(hours), 1), 48)
    if baseline_from + timedelta(hours=hours) < window.start_time:
        raise ValueError("baseline horizon does not cover scenario start_time")

    user_label = raw_request.get("user_label", raw_request.get("label"))
    if user_label is not None and len(str(user_label)) > MAX_USER_LABEL_CHARS:
        raise ValueError(f"user_label must be {MAX_USER_LABEL_CHARS} characters or fewer")
    normalized = {
        "scenario_type": scenario_type,
        "magnitude": magnitude,
        "scope": scope,
        "start_time": _iso_utc(window.start_time),
        "end_time": _iso_utc(window.end_time),
        "duration_hours": window.duration_hours,
        "assumptions": assumptions,
        "user_label": None if user_label is None else str(user_label),
        "timezone": timezone_name,
        "baseline": {
            "from_time": _iso_utc(baseline_from),
            "hours": hours,
            "region": _baseline_region(scope),
        },
    }
    return normalized


def normalized_request_hash(normalized_request: Mapping[str, Any]) -> str:
    payload = json.dumps(normalized_request, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScenarioService:
    """Run initial What If scenarios against a baseline TwinResponse."""

    def __init__(
        self,
        *,
        twin_service: TwinService | None = None,
        now: Any | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self._twin_service = twin_service or default_twin_service
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._cache_enabled = cache_enabled
        self._cache: dict[str, dict[str, Any]] = {}

    def run(
        self,
        raw_request: Mapping[str, Any],
        *,
        baseline_response: TwinResponse | None = None,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_scenario_request(raw_request)
        request_hash = normalized_request_hash(normalized)
        cache_requested = self._cache_enabled if use_cache is None else bool(use_cache)
        if cache_requested and baseline_response is None and request_hash in self._cache:
            cached = deepcopy(self._cache[request_hash])
            cached["cache"] = {"enabled": True, "hit": True, "key": request_hash}
            record_scenario_run(str(cached.get("result_id") or ""))
            return cached

        baseline = baseline_response or self._twin_service.get_twin(
            from_timestamp=normalized["baseline"]["from_time"],
            hours=int(normalized["baseline"]["hours"]),
            region=normalized["baseline"].get("region"),
        )
        result = self._run_normalized(normalized, request_hash, baseline)
        result["cache"] = {"enabled": cache_requested, "hit": False, "key": request_hash}
        if cache_requested and baseline_response is None:
            self._cache[request_hash] = deepcopy(result)
        record_scenario_run(str(result.get("result_id") or ""))
        return result

    def _run_normalized(
        self,
        normalized: Mapping[str, Any],
        request_hash: str,
        baseline: TwinResponse,
    ) -> dict[str, Any]:
        baseline_series = [_baseline_row(snapshot) for snapshot in baseline.snapshots]
        effects = _scenario_effects(normalized, baseline_series)
        scenario_series = [_scenario_row(row, effects.get(row["timestamp"], {})) for row in baseline_series]

        demand_delta_series = [
            {"timestamp": row["timestamp"], "delta_mw": row["demand_delta_mw"]}
            for row in scenario_series
            if abs(float(row["demand_delta_mw"])) > 1e-9
        ]
        peak_demand_delta = _peak_value(scenario_series, "demand_mw") - _peak_value(baseline_series, "demand_mw")
        changed = [
            {
                "timestamp": row["timestamp"],
                "baseline_status": row["baseline_balance_status"],
                "scenario_status": row["balance_status"],
                "balance_score_delta": row["balance_score_delta"],
            }
            for row in scenario_series
            if row["baseline_balance_status"] != row["balance_status"]
            and (row["baseline_balance_status"] in {"watch", "high"} or row["balance_status"] in {"watch", "high"})
        ]
        import_range = _sum_ranges(row["import_export_delta_mw_range"] for row in scenario_series)
        generation_range = _sum_ranges(row["flexible_generation_response_mw_range"] for row in scenario_series)
        carbon_range = _sum_ranges(row["carbon_delta_tonnes_range"] for row in scenario_series)
        balance_deltas = [
            {"timestamp": row["timestamp"], "score_delta": row["balance_score_delta"]}
            for row in scenario_series
            if abs(float(row["balance_score_delta"])) > 1e-12
        ]
        generated_at = _iso_utc(_ensure_utc(self._now()))
        result = {
            "result_id": f"scenario-{request_hash[:16]}",
            "request_hash": request_hash,
            "generated_at": generated_at,
            "scenario_request": deepcopy(dict(normalized)),
            "baseline_forecast_run_id": _baseline_run_id(baseline),
            "baseline_series": baseline_series,
            "scenario_series": scenario_series,
            "demand_delta": {
                "unit": "MW_by_hour_and_MWh_total",
                "series": demand_delta_series,
                "total_mwh": _sum_values(row["demand_delta_mw"] for row in scenario_series),
                "peak_hourly_delta_mw": _peak_abs_delta(scenario_series, "demand_delta_mw"),
            },
            "peak_demand_delta_mw": peak_demand_delta,
            "changed_watch_or_high_hours": changed,
            "changed_watch_or_high_hour_count": len(changed),
            "balance_context_delta": {
                "unit": "score_delta",
                "peak_score_delta": _peak_abs_delta(scenario_series, "balance_score_delta"),
                "series": balance_deltas,
                "method": (
                    "Scenario stress is translated into the documented modelled-balance score using "
                    "scenario-assumptions.v1; this is not an operational reserve-margin calculation."
                ),
            },
            "estimated_import_export_delta": {
                "defensible": any(row["net_imports_mw"] is not None for row in baseline_series),
                "net_import_delta_mwh_range": import_range,
                "peak_net_import_delta_mw_range": _peak_range(
                    row["import_export_delta_mw_range"] for row in scenario_series
                ),
                "method": (
                    "After flexible generation response, a configured fraction of the remaining residual "
                    "is represented as higher net imports, lower exports, lower net imports, or higher exports."
                ),
            },
            "estimated_generation_response_range": {
                "flexible_generation_delta_mwh_range": generation_range,
                "peak_flexible_generation_delta_mw_range": _peak_range(
                    row["flexible_generation_response_mw_range"] for row in scenario_series
                ),
                "method": "Flexible generation absorbs a configured portion of residual change; wind and solar follow baseline forecasts.",
            },
            "estimated_carbon_range": {
                "total_tonnes_co2_delta_range": carbon_range,
                "method": (
                    "Each positive residual MWh is multiplied by a plausible response-mix intensity range; "
                    "each negative residual MWh is treated as avoided response-mix output."
                ),
            },
            "regional_deltas": _regional_delta_summary(normalized, scenario_series),
            "causal_chain": _causal_chain(normalized),
            "assumptions": _assumptions_list(normalized),
            "caveats": list(SCENARIO_CAVEATS),
            "model_versions": {
                "scenario_engine": SCENARIO_ENGINE_VERSION,
                "scenario_assumptions": SCENARIO_ASSUMPTION_VERSION,
                "status_thresholds": threshold_config_version(),
            },
            "data_versions": {
                "baseline_from_time": _iso_utc(baseline.from_time),
                "baseline_generated_at": _iso_utc(baseline.generated_at),
                "baseline_hours": baseline.hours,
                "baseline_snapshot_count": len(baseline.snapshots),
                "baseline_snapshot_ids": [snapshot.snapshot_id for snapshot in baseline.snapshots],
                "baseline_series_hash": _series_hash(baseline_series),
                "provenance_names": _provenance_names(baseline.snapshots),
            },
            "unsupported_grid_behaviours": list(UNSUPPORTED_PHYSICAL_BEHAVIOURS),
        }
        return result


def _scenario_effects(normalized: Mapping[str, Any], baseline_series: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    effects: dict[str, dict[str, Any]] = {row["timestamp"]: _empty_effect() for row in baseline_series}
    scenario_type = str(normalized["scenario_type"])
    if scenario_type == "cold_snap":
        _apply_cold_snap(normalized, baseline_series, effects)
    elif scenario_type == "generation_unavailability":
        _apply_generation_unavailability(normalized, baseline_series, effects)
    elif scenario_type == "ev_charging_shift":
        _apply_ev_shift(normalized, baseline_series, effects)
    return effects


def _apply_cold_snap(
    normalized: Mapping[str, Any],
    baseline_series: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
) -> None:
    magnitude = dict(normalized["magnitude"])
    temp_delta = float(magnitude["temperature_delta_c"])
    heating_delta = -temp_delta * float(DEFAULT_ASSUMPTIONS["heating_sensitivity_mw_per_c"])
    start = _parse_utc_iso(str(normalized["start_time"]))
    end = _parse_utc_iso(str(normalized["end_time"]))
    for row in baseline_series:
        timestamp = _parse_utc_iso(str(row["timestamp"]))
        if not (start <= timestamp < end):
            continue
        local_hour = timestamp.astimezone(ZoneInfo(str(normalized["timezone"]))).hour
        scope_share = _scope_share(row, dict(normalized["scope"]))
        delta = heating_delta * _cold_hour_multiplier(local_hour) * scope_share
        effect = effects[row["timestamp"]]
        effect["demand_delta_mw"] += delta
        effect["regional_deltas"].update(_regional_demand_deltas(row, dict(normalized["scope"]), delta, "cold_snap"))


def _apply_generation_unavailability(
    normalized: Mapping[str, Any],
    baseline_series: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
) -> None:
    capacity = float(dict(normalized["magnitude"])["unavailable_capacity_mw"])
    start = _parse_utc_iso(str(normalized["start_time"]))
    end = _parse_utc_iso(str(normalized["end_time"]))
    for row in baseline_series:
        timestamp = _parse_utc_iso(str(row["timestamp"]))
        if start <= timestamp < end:
            effect = effects[row["timestamp"]]
            effect["generation_availability_delta_mw"] -= capacity
            effect["supply_loss_mw"] += capacity


def _apply_ev_shift(
    normalized: Mapping[str, Any],
    baseline_series: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
) -> None:
    magnitude = dict(normalized["magnitude"])
    timezone_name = str(normalized["timezone"])
    tz = ZoneInfo(timezone_name)
    start = _parse_utc_iso(str(normalized["start_time"]))
    end = _parse_utc_iso(str(normalized["end_time"]))
    original_window = tuple(magnitude["original_charging_window_minutes"])
    target_window = tuple(magnitude["target_charging_window_minutes"])

    original_rows: list[dict[str, Any]] = []
    original_dates: set[date] = set()
    for row in baseline_series:
        timestamp = _parse_utc_iso(str(row["timestamp"]))
        local = timestamp.astimezone(tz)
        if start <= timestamp < end and _local_minute_in_window(_minute_of_day(local), original_window):
            original_rows.append(row)
            original_dates.add(local.date())
    if not original_rows:
        return

    target_dates = {
        item + timedelta(days=1 if target_window[0] < original_window[0] else 0)
        for item in original_dates
    }
    target_rows = [
        row
        for row in baseline_series
        if _parse_utc_iso(str(row["timestamp"])).astimezone(tz).date() in target_dates
        and _local_minute_in_window(_minute_of_day(_parse_utc_iso(str(row["timestamp"])).astimezone(tz)), target_window)
    ]
    if not target_rows:
        return

    total_mwh = (
        float(magnitude["vehicles"])
        * float(magnitude["average_energy_per_vehicle_kwh"])
        * float(magnitude["participation_rate"])
        / 1000.0
        * len(original_dates)
    )
    remove_mw = total_mwh / len(original_rows)
    add_mw = total_mwh / len(target_rows)
    scope = dict(normalized["scope"])
    for row in original_rows:
        effect = effects[row["timestamp"]]
        effect["demand_delta_mw"] -= remove_mw
        effect["regional_deltas"].update(_regional_demand_deltas(row, scope, -remove_mw, "ev_charging_shift"))
    for row in target_rows:
        effect = effects[row["timestamp"]]
        effect["demand_delta_mw"] += add_mw
        effect["regional_deltas"].update(_regional_demand_deltas(row, scope, add_mw, "ev_charging_shift"))
        effect["rebound_peak_candidate"] = True


def _scenario_row(row: Mapping[str, Any], effect: Mapping[str, Any]) -> dict[str, Any]:
    demand_delta = float(effect.get("demand_delta_mw", 0.0))
    availability_delta = float(effect.get("generation_availability_delta_mw", 0.0))
    supply_loss = float(effect.get("supply_loss_mw", 0.0))
    stress = demand_delta + supply_loss
    flex_range = _scaled_range(stress, DEFAULT_ASSUMPTIONS["flexible_generation_response_fraction_range"])
    flex_mid = sum(flex_range) / 2.0
    remaining_after_flex = stress - flex_mid
    import_range = _scaled_range(
        remaining_after_flex,
        DEFAULT_ASSUMPTIONS["import_export_response_fraction_of_remaining_range"],
    )
    import_mid = sum(import_range) / 2.0
    unresolved = stress - flex_mid - import_mid
    demand = float(row["demand_mw"]) + demand_delta
    available_generation = _none_safe(row.get("available_generation_mw")) + availability_delta + flex_mid
    net_imports = _none_safe(row.get("net_imports_mw")) + import_mid
    supply_margin = _none_safe(row.get("supply_margin_mw")) - demand_delta + availability_delta + flex_mid + import_mid
    score_delta = _balance_score_delta(stress, flex_mid, import_mid, float(row["demand_mw"]))
    score = min(max(float(row["balance_score"] or 0.0) + score_delta, 0.0), 1.0)
    status = modelled_balance_status_for_score(score).value
    carbon_range = _carbon_range_for_stress(stress)
    regional_deltas = dict(effect.get("regional_deltas", {}))
    scenario_regional = _scenario_regional(row, regional_deltas)
    result = {
        **dict(row),
        "demand_mw": demand,
        "baseline_demand_mw": row["demand_mw"],
        "demand_delta_mw": demand_delta,
        "generation_availability_delta_mw": availability_delta,
        "stress_change_mw": stress,
        "flexible_generation_response_mw_range": flex_range,
        "flexible_generation_response_mw": flex_mid,
        "import_export_delta_mw_range": import_range,
        "import_export_delta_mw": import_mid,
        "unresolved_residual_mw": unresolved,
        "available_generation_mw": available_generation,
        "net_imports_mw": net_imports,
        "supply_margin_mw": supply_margin,
        "import_requirement_mw": max(-supply_margin, 0.0),
        "balance_score": score,
        "balance_score_delta": score_delta,
        "baseline_balance_status": row["balance_status"],
        "balance_status": status,
        "carbon_delta_tonnes_range": carbon_range,
        "regional_demand_deltas_mw": regional_deltas,
        "regional_demand_context": scenario_regional,
        "rebound_peak_candidate": bool(effect.get("rebound_peak_candidate", False)),
    }
    return result


def _baseline_row(snapshot: TwinSnapshot) -> dict[str, Any]:
    balance = snapshot.modelled_national_balance_context or snapshot.national.balance_context
    demand = snapshot.demand_forecast.p50.value if snapshot.demand_forecast else snapshot.national.demand_context.current.value
    generation_mix = snapshot.generation_mix_estimate
    exchange = snapshot.exchange_estimate
    carbon = snapshot.carbon_estimate
    regional = []
    for region in snapshot.regional_demand_context or []:
        share = region.share_of_national_p50.value
        if share is None and demand:
            share = (region.forecast.p50.value or 0.0) / float(demand)
        regional.append(
            {
                "region_code": region.region_code,
                "region_name": region.region_name,
                "demand_mw": region.forecast.p50.value,
                "share_of_national_p50": share,
            }
        )
    return {
        "timestamp": _iso_utc(snapshot.event_time),
        "snapshot_id": snapshot.snapshot_id,
        "demand_mw": float(demand or 0.0),
        "demand_p10_mw": snapshot.demand_forecast.p10.value if snapshot.demand_forecast else None,
        "demand_p90_mw": snapshot.demand_forecast.p90.value if snapshot.demand_forecast else None,
        "wind_mw": snapshot.wind_estimate.value.value if snapshot.wind_estimate else None,
        "solar_mw": snapshot.solar_estimate.value.value if snapshot.solar_estimate else None,
        "nuclear_available_mw": (
            snapshot.generation_availability_context.nuclear.value.value
            if snapshot.generation_availability_context
            else None
        ),
        "generation_total_mw": generation_mix.total.value if generation_mix else None,
        "available_generation_mw": balance.available_generation.value if balance else None,
        "net_imports_mw": exchange.net_imports.value if exchange else (balance.net_imports.value if balance else None),
        "supply_margin_mw": balance.supply_margin.value if balance else None,
        "import_requirement_mw": balance.import_requirement.value if balance else None,
        "balance_score": balance.pressure_ratio.value if balance else None,
        "balance_status": balance.status.value if balance else Status.UNKNOWN.value,
        "carbon_intensity_g_per_kwh": carbon.intensity.value if carbon else None,
        "regional_demand_context": regional,
    }


def _empty_effect() -> dict[str, Any]:
    return {
        "demand_delta_mw": 0.0,
        "generation_availability_delta_mw": 0.0,
        "supply_loss_mw": 0.0,
        "regional_deltas": {},
        "rebound_peak_candidate": False,
    }


def _scenario_type(raw_type: Any) -> str:
    if raw_type in (None, ""):
        raise ValueError("scenario request requires scenario_type")
    key = str(raw_type).strip().lower()
    if key not in SUPPORTED_SCENARIOS:
        raise ValueError(f"unsupported scenario_type: {raw_type}")
    return SUPPORTED_SCENARIOS[key]


def _normalize_magnitude(scenario_type: str, raw_magnitude: Any) -> dict[str, Any]:
    if raw_magnitude is None:
        raise ValueError("scenario request requires magnitude")
    if scenario_type == "cold_snap":
        value = _field_or_number(raw_magnitude, "temperature_delta_c", "temp_delta_c", "delta_c")
        if value >= 0:
            raise ValueError("cold_snap magnitude.temperature_delta_c must be negative")
        if abs(value) > MAX_COLD_DELTA_C:
            raise ValueError(f"cold_snap magnitude.temperature_delta_c cannot exceed {MAX_COLD_DELTA_C:g} deg C")
        return {"temperature_delta_c": value}
    if scenario_type == "generation_unavailability":
        value = _field_or_number(raw_magnitude, "unavailable_capacity_mw", "capacity_mw")
        if value <= 0:
            raise ValueError("generation_unavailability magnitude.unavailable_capacity_mw must be positive")
        if value > MAX_UNAVAILABLE_CAPACITY_MW:
            raise ValueError(
                f"generation_unavailability magnitude.unavailable_capacity_mw cannot exceed {MAX_UNAVAILABLE_CAPACITY_MW:g}"
            )
        asset_name = raw_magnitude.get("asset_name") if isinstance(raw_magnitude, Mapping) else None
        asset_text = None if asset_name in (None, "") else str(asset_name)[:MAX_USER_LABEL_CHARS]
        return {"unavailable_capacity_mw": value, "asset_name": asset_text}
    if scenario_type == "ev_charging_shift":
        if not isinstance(raw_magnitude, Mapping):
            raise ValueError("ev_charging_shift magnitude must be an object")
        vehicles = _positive_float(raw_magnitude.get("vehicles", raw_magnitude.get("number_of_vehicles")), "vehicles")
        energy = _positive_float(
            raw_magnitude.get("average_energy_per_vehicle_kwh", raw_magnitude.get("avg_energy_per_vehicle_kwh")),
            "average_energy_per_vehicle_kwh",
        )
        if vehicles > MAX_EV_VEHICLES:
            raise ValueError(f"ev_charging_shift vehicles cannot exceed {MAX_EV_VEHICLES:g}")
        if energy > MAX_EV_ENERGY_KWH:
            raise ValueError(f"ev_charging_shift average_energy_per_vehicle_kwh cannot exceed {MAX_EV_ENERGY_KWH:g}")
        participation = _positive_float(raw_magnitude.get("participation_rate"), "participation_rate")
        if participation > 1:
            raise ValueError("ev_charging_shift participation_rate must be between 0 and 1")
        original = _normalize_window(raw_magnitude.get("original_charging_window"), "original_charging_window")
        target = _normalize_window(raw_magnitude.get("target_charging_window"), "target_charging_window")
        return {
            "vehicles": vehicles,
            "average_energy_per_vehicle_kwh": energy,
            "participation_rate": participation,
            "original_charging_window": _window_label(original),
            "target_charging_window": _window_label(target),
            "original_charging_window_minutes": list(original),
            "target_charging_window_minutes": list(target),
        }
    raise ValueError(f"unsupported scenario_type: {scenario_type}")


def _normalize_scope(raw_scope: Any, scenario_type: str) -> dict[str, Any]:
    if raw_scope in (None, ""):
        raise ValueError("scenario request requires scope")
    if isinstance(raw_scope, str):
        key = raw_scope.strip().lower()
        if key in {"national", "france", "metropolitan_france"}:
            return {"type": "national", "regions": []}
        raise ValueError(f"unsupported scope: {raw_scope}")
    if not isinstance(raw_scope, Mapping):
        raise ValueError("scope must be a string or object")
    scope_type = str(raw_scope.get("type", raw_scope.get("scope", ""))).strip().lower()
    if scope_type in {"national", "france"}:
        return {"type": "national", "regions": []}
    if scope_type != "regional":
        raise ValueError(f"unsupported scope: {scope_type}")
    if scenario_type == "generation_unavailability":
        raise ValueError("generation_unavailability currently supports national scope only")
    raw_regions = raw_scope.get("regions", raw_scope.get("region_codes", raw_scope.get("region")))
    if isinstance(raw_regions, str):
        regions = [raw_regions]
    elif isinstance(raw_regions, list | tuple):
        regions = [str(item) for item in raw_regions]
    else:
        raise ValueError("regional scope requires regions")
    normalized = []
    for code in regions:
        item = str(code).strip()
        if item not in REGION_NAMES:
            raise ValueError(f"unsupported regional scope code: {item}")
        normalized.append(item)
    if not normalized:
        raise ValueError("regional scope requires at least one region")
    return {"type": "regional", "regions": sorted(set(normalized))}


def _scenario_window(raw_request: Mapping[str, Any], timezone_name: str) -> ScenarioWindow:
    start = _parse_timestamp(raw_request["start_time"], timezone_name, field_name="start_time")
    end_value = raw_request.get("end_time")
    duration_value = raw_request.get("duration_hours", raw_request.get("duration"))
    if end_value is None and duration_value is None:
        raise ValueError("scenario request requires duration_hours or end_time")
    end_from_duration: datetime | None = None
    duration_hours: float | None = None
    if duration_value is not None:
        duration_hours = _duration_hours(duration_value)
        end_from_duration = start + timedelta(hours=duration_hours)
    end = _parse_timestamp(end_value, timezone_name, field_name="end_time") if end_value is not None else end_from_duration
    if end is None:
        raise ValueError("scenario request requires duration_hours or end_time")
    if end_from_duration is not None and end_value is not None and abs((end - end_from_duration).total_seconds()) > 1:
        raise ValueError("duration_hours and end_time describe different windows")
    if end <= start:
        raise ValueError("scenario end_time must be after start_time")
    if duration_hours is None:
        duration_hours = (end - start).total_seconds() / 3600.0
    return ScenarioWindow(start_time=start, end_time=end, duration_hours=duration_hours)


def _parse_timestamp(value: Any, timezone_name: str, *, field_name: str) -> datetime:
    if value is None:
        raise ValueError(f"{field_name} is required")
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an ISO timestamp") from exc
    if timestamp.tzinfo is None:
        try:
            timestamp = timestamp.tz_localize(timezone_name, ambiguous="raise", nonexistent="raise")
        except Exception as exc:
            raise ValueError(f"{field_name} is ambiguous or nonexistent in timezone {timezone_name}") from exc
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _duration_hours(value: Any) -> float:
    try:
        hours = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_hours must be numeric hours") from exc
    if hours <= 0:
        raise ValueError("duration_hours must be positive")
    return hours


def _field_or_number(raw: Any, *names: str) -> float:
    if isinstance(raw, Mapping):
        for name in names:
            if name in raw:
                return _finite_float(raw[name], name)
        raise ValueError(f"magnitude requires one of: {', '.join(names)}")
    return _finite_float(raw, "magnitude")


def _positive_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _normalize_window(value: Any, field_name: str) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, str):
        if "-" not in value:
            raise ValueError(f"{field_name} string must look like HH:MM-HH:MM")
        start_text, end_text = value.split("-", 1)
    elif isinstance(value, Mapping):
        start_text = value.get("start")
        end_text = value.get("end")
    elif isinstance(value, list | tuple) and len(value) == 2:
        start_text, end_text = value
    else:
        raise ValueError(f"{field_name} must be a window object, pair, or HH:MM-HH:MM string")
    start = _parse_hhmm(start_text, f"{field_name}.start")
    end = _parse_hhmm(end_text, f"{field_name}.end")
    if start == end:
        raise ValueError(f"{field_name} cannot have equal start and end")
    return start, end


def _parse_hhmm(value: Any, field_name: str) -> int:
    try:
        parts = str(value).strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{field_name} must be HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{field_name} must be a valid local time")
    return hour * 60 + minute


def _window_label(window: tuple[int, int]) -> dict[str, str]:
    return {"start": _minutes_label(window[0]), "end": _minutes_label(window[1])}


def _minutes_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _cold_hour_multiplier(local_hour: int) -> float:
    if 7 <= local_hour <= 10 or 17 <= local_hour <= 21:
        return float(DEFAULT_ASSUMPTIONS["cold_snap_peak_multiplier"])
    if 11 <= local_hour <= 16:
        return float(DEFAULT_ASSUMPTIONS["cold_snap_day_multiplier"])
    return float(DEFAULT_ASSUMPTIONS["cold_snap_offpeak_multiplier"])


def _scope_share(row: Mapping[str, Any], scope: Mapping[str, Any]) -> float:
    if scope.get("type") == "national":
        return 1.0
    regions = set(scope.get("regions") or [])
    return sum(float(item.get("share_of_national_p50") or 0.0) for item in row.get("regional_demand_context", []) if item.get("region_code") in regions)


def _regional_demand_deltas(
    row: Mapping[str, Any],
    scope: Mapping[str, Any],
    national_delta_mw: float,
    scenario_type: str,
) -> dict[str, float]:
    regional = list(row.get("regional_demand_context", []))
    if not regional:
        return {}
    if scope.get("type") == "national":
        return {
            str(item["region_code"]): national_delta_mw * float(item.get("share_of_national_p50") or 0.0)
            for item in regional
        }
    selected = set(scope.get("regions") or [])
    selected_share = sum(
        float(item.get("share_of_national_p50") or 0.0)
        for item in regional
        if item.get("region_code") in selected
    )
    if selected_share <= 0:
        return {}
    if scenario_type == "cold_snap":
        full_national_equivalent = national_delta_mw / selected_share
        return {
            str(item["region_code"]): full_national_equivalent * float(item.get("share_of_national_p50") or 0.0)
            for item in regional
            if item.get("region_code") in selected
        }
    return {
        str(item["region_code"]): national_delta_mw * float(item.get("share_of_national_p50") or 0.0) / selected_share
        for item in regional
        if item.get("region_code") in selected
    }


def _scenario_regional(row: Mapping[str, Any], deltas: Mapping[str, float]) -> list[dict[str, Any]]:
    result = []
    for item in row.get("regional_demand_context", []):
        code = str(item["region_code"])
        delta = float(deltas.get(code, 0.0))
        result.append({**dict(item), "demand_delta_mw": delta, "scenario_demand_mw": float(item.get("demand_mw") or 0.0) + delta})
    return result


def _local_minute_in_window(minute: int, window: tuple[int, int]) -> bool:
    start, end = window
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


def _minute_of_day(value: datetime) -> int:
    return value.hour * 60 + value.minute


def _scaled_range(value: float, fractions: Any) -> list[float]:
    low_fraction, high_fraction = (float(fractions[0]), float(fractions[1]))
    values = [value * low_fraction, value * high_fraction]
    return [min(values), max(values)]


def _balance_score_delta(stress: float, flex_mid: float, import_mid: float, baseline_demand: float) -> float:
    if baseline_demand <= 0:
        return 0.0
    effective_stress = stress - 0.40 * flex_mid - 0.15 * import_mid
    return (effective_stress / baseline_demand) * float(DEFAULT_ASSUMPTIONS["balance_score_pressure_multiplier"])


def _carbon_range_for_stress(stress_mw: float) -> list[float]:
    positive_low, positive_high = DEFAULT_ASSUMPTIONS["positive_response_carbon_g_per_kwh_range"]
    avoided_low, avoided_high = DEFAULT_ASSUMPTIONS["negative_response_avoided_carbon_g_per_kwh_range"]
    if stress_mw >= 0:
        return sorted([stress_mw * float(positive_low) / 1000.0, stress_mw * float(positive_high) / 1000.0])
    return sorted([stress_mw * float(avoided_high) / 1000.0, stress_mw * float(avoided_low) / 1000.0])


def _sum_ranges(ranges: Any) -> list[float]:
    low = 0.0
    high = 0.0
    for item in ranges:
        low += float(item[0])
        high += float(item[1])
    return [min(low, high), max(low, high)]


def _peak_range(ranges: Any) -> list[float]:
    lows: list[float] = []
    highs: list[float] = []
    for item in ranges:
        lows.append(float(item[0]))
        highs.append(float(item[1]))
    if not lows:
        return [0.0, 0.0]
    return [min(lows), max(highs)]


def _sum_values(values: Any) -> float:
    return float(sum(float(value) for value in values))


def _peak_value(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return max(float(row[key]) for row in rows)


def _peak_abs_delta(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return max((float(row[key]) for row in rows), key=lambda item: abs(item))


def _regional_delta_summary(normalized: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if normalized["scenario_type"] == "generation_unavailability":
        return {
            "supported": False,
            "reason": "Generating-unit unavailability is represented as national supply availability only in v1.",
            "regions": [],
        }
    totals: dict[str, float] = {}
    peaks: dict[str, float] = {}
    names: dict[str, str] = {}
    for row in rows:
        for item in row.get("regional_demand_context", []):
            code = str(item["region_code"])
            delta = float(item.get("demand_delta_mw", 0.0))
            names[code] = str(item.get("region_name", REGION_NAMES.get(code, code)))
            totals[code] = totals.get(code, 0.0) + delta
            if code not in peaks or abs(delta) > abs(peaks[code]):
                peaks[code] = delta
    return {
        "supported": True,
        "method": "Regional demand deltas are allocated from the reconciled regional demand shares; this is not regional adequacy.",
        "regions": [
            {
                "region_code": code,
                "region_name": names.get(code, REGION_NAMES.get(code, code)),
                "total_delta_mwh": totals.get(code, 0.0),
                "peak_delta_mw": peaks.get(code, 0.0),
            }
            for code in sorted(totals)
            if abs(totals.get(code, 0.0)) > 1e-9 or abs(peaks.get(code, 0.0)) > 1e-9
        ],
    }


def _causal_chain(normalized: Mapping[str, Any]) -> dict[str, Any]:
    scenario_type = str(normalized["scenario_type"])
    if scenario_type == "cold_snap":
        nodes = [
            {"id": "temperature_delta_c", "kind": "input"},
            {"id": "demand_delta_mw", "kind": "demand_feature_rerun"},
            {"id": "residual_change_mw", "kind": "balance_context"},
            {"id": "response_mix_range", "kind": "generation_import_response"},
            {"id": "carbon_range", "kind": "emissions_context"},
        ]
        edges = [
            {
                "from": "temperature_delta_c",
                "to": "demand_delta_mw",
                "formula": "demand_delta_mw = -temperature_delta_c * heating_sensitivity_mw_per_c * hour_multiplier * scope_share",
            },
            {"from": "demand_delta_mw", "to": "residual_change_mw", "formula": "residual_change_mw = demand_delta_mw"},
        ]
    elif scenario_type == "generation_unavailability":
        nodes = [
            {"id": "unavailable_capacity_mw", "kind": "input"},
            {"id": "generation_availability_delta_mw", "kind": "supply_context"},
            {"id": "residual_change_mw", "kind": "balance_context"},
            {"id": "response_mix_range", "kind": "generation_import_response"},
            {"id": "carbon_range", "kind": "emissions_context"},
        ]
        edges = [
            {
                "from": "unavailable_capacity_mw",
                "to": "generation_availability_delta_mw",
                "formula": "generation_availability_delta_mw = -unavailable_capacity_mw",
            },
            {
                "from": "generation_availability_delta_mw",
                "to": "residual_change_mw",
                "formula": "residual_change_mw = -generation_availability_delta_mw",
            },
        ]
    else:
        nodes = [
            {"id": "ev_energy_mwh", "kind": "input"},
            {"id": "original_charging_window", "kind": "load_removal"},
            {"id": "target_charging_window", "kind": "load_addition"},
            {"id": "rebound_peak_candidate", "kind": "demand_feature_rerun"},
            {"id": "carbon_range", "kind": "emissions_context"},
        ]
        edges = [
            {
                "from": "ev_energy_mwh",
                "to": "original_charging_window",
                "formula": "removed_mw_per_hour = vehicles * kwh_per_vehicle * participation_rate / 1000 / original_window_hours",
            },
            {
                "from": "ev_energy_mwh",
                "to": "target_charging_window",
                "formula": "added_mw_per_hour = same_total_mwh / target_window_hours",
            },
        ]
    return {"scenario_type": scenario_type, "nodes": nodes, "edges": edges}


def _assumptions_list(normalized: Mapping[str, Any]) -> list[str]:
    request_assumptions = normalized.get("assumptions")
    assumptions = [
        f"Request assumptions: {json.dumps(request_assumptions, sort_keys=True, ensure_ascii=True)}",
        f"Scenario assumptions artifact: {SCENARIO_ASSUMPTION_VERSION}.",
        "Wind and solar follow their baseline adjusted forecasts.",
        "Nuclear follows availability context; generating-unit unavailability reduces available supply, not demand.",
        (
            "Flexible generation response fraction range: "
            f"{DEFAULT_ASSUMPTIONS['flexible_generation_response_fraction_range']} of residual change."
        ),
        (
            "Import/export response fraction range after flexible response: "
            f"{DEFAULT_ASSUMPTIONS['import_export_response_fraction_of_remaining_range']}."
        ),
    ]
    if normalized["scenario_type"] == "cold_snap":
        assumptions.append(
            "Cold-snap demand rerun uses heating_sensitivity_mw_per_c and deterministic local-hour multipliers."
        )
    if normalized["scenario_type"] == "ev_charging_shift":
        assumptions.append("EV charging shift conserves total modelled scenario energy across complete source and target windows.")
    return assumptions


def _baseline_run_id(baseline: TwinResponse) -> str:
    return f"twin:{_iso_utc(baseline.from_time)}:{baseline.hours}:{baseline.region or 'national'}"


def _series_hash(series: list[dict[str, Any]]) -> str:
    payload = json.dumps(series, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance_names(snapshots: list[TwinSnapshot]) -> list[str]:
    names: set[str] = set()
    for snapshot in snapshots:
        for source in snapshot.provenance_chain or []:
            names.add(source.name)
    return sorted(names)


def _baseline_region(scope: Mapping[str, Any]) -> str | None:
    if scope.get("type") == "regional":
        regions = list(scope.get("regions") or [])
        return str(regions[0]) if regions else None
    return None


def _none_safe(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _parse_utc_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


default_scenario_service = ScenarioService()


__all__ = [
    "SCENARIO_ASSUMPTION_VERSION",
    "SCENARIO_ENGINE_VERSION",
    "ScenarioService",
    "default_scenario_service",
    "normalize_scenario_request",
    "normalized_request_hash",
]
