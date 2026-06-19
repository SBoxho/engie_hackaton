# Scenario Engine

Date: 2026-06-19

`POST /v1/scenarios/run` compares a baseline `TwinSnapshot` sequence with a deterministic scenario sequence. This is a first-generation directional simulator for the What If page. It is not an operator dispatch forecast, an adequacy study, or verified emissions accounting.

## Supported Scenarios

1. Cold snap
   - Input: `temperature_delta_c` as a negative temperature delta.
   - Demand is rerun with a fixed heating sensitivity and local-hour multipliers.
   - Wind and solar are unchanged from the baseline twin sequence.

2. Generating-unit unavailability
   - Input: `unavailable_capacity_mw`, plus optional `asset_name`.
   - Generation availability and modelled balance context change.
   - Demand is not modified.

3. EV charging shift
   - Inputs: `vehicles`, `average_energy_per_vehicle_kwh`, `participation_rate`, `original_charging_window`, `target_charging_window`.
   - Total scenario energy is conserved across complete original and target windows.
   - Target-window load additions are marked as rebound-peak candidates.

## Request Contract

Every request must include:

- `scenario_type`
- `magnitude`
- `scope`
- `start_time`
- `duration_hours` or `end_time`
- `assumptions`
- optional `user_label`

Requests are normalized to UTC and hashed with SHA-256. When caching is enabled, the normalized request hash is the cache key.

## Main Formulas

Cold snap:

```text
demand_delta_mw =
  -temperature_delta_c
  * heating_sensitivity_mw_per_c
  * local_hour_multiplier
  * scope_share
```

Generating-unit unavailability:

```text
generation_availability_delta_mw = -unavailable_capacity_mw
residual_change_mw = unavailable_capacity_mw
demand_delta_mw = 0
```

EV charging shift:

```text
shift_energy_mwh =
  vehicles * average_energy_per_vehicle_kwh * participation_rate / 1000

removed_mw_per_original_hour = shift_energy_mwh / original_window_hours
added_mw_per_target_hour = shift_energy_mwh / target_window_hours
sum(demand_delta_mw over compared hourly sequence) = 0
```

Response heuristic:

```text
stress_change_mw = demand_delta_mw + supply_loss_mw
flexible_generation_response_mw_range =
  stress_change_mw * flexible_generation_response_fraction_range

remaining_after_flexible_response_mw =
  stress_change_mw - midpoint(flexible_generation_response_mw_range)

import_export_delta_mw_range =
  remaining_after_flexible_response_mw
  * import_export_response_fraction_of_remaining_range
```

Modelled balance-context score:

```text
effective_stress_mw =
  stress_change_mw
  - 0.40 * midpoint(flexible_generation_response_mw_range)
  - 0.15 * midpoint(import_export_delta_mw_range)

balance_score_delta =
  effective_stress_mw / baseline_demand_mw * balance_score_pressure_multiplier
```

Carbon:

```text
positive stress:
  carbon_delta_tonnes_range =
    stress_mwh * positive_response_carbon_g_per_kwh_range / 1000

negative stress:
  carbon_delta_tonnes_range =
    stress_mwh * negative_avoided_response_carbon_g_per_kwh_range / 1000
```

Ranges are sorted before being returned.

## Example Requests

Cold snap:

```json
{
  "scenario_type": "cold_snap",
  "magnitude": {"temperature_delta_c": -4},
  "scope": "national",
  "start_time": "2026-06-18T14:00:00Z",
  "duration_hours": 6,
  "baseline_from_time": "2026-06-18T12:00:00Z",
  "hours": 24,
  "assumptions": {"source": "user what-if", "heating": "directional v1"},
  "user_label": "Four-degree cold snap"
}
```

Generating-unit unavailability:

```json
{
  "scenario_type": "generation_unavailability",
  "magnitude": {"unavailable_capacity_mw": 1300, "asset_name": "Example unit"},
  "scope": "national",
  "start_time": "2026-06-18T14:00:00Z",
  "end_time": "2026-06-18T22:00:00Z",
  "baseline_from_time": "2026-06-18T12:00:00Z",
  "hours": 24,
  "assumptions": {"source": "user what-if", "availability": "capacity unavailable in selected window"}
}
```

EV charging shift:

```json
{
  "scenario_type": "ev_charging_shift",
  "magnitude": {
    "vehicles": 100000,
    "average_energy_per_vehicle_kwh": 8,
    "participation_rate": 0.6,
    "original_charging_window": {"start": "18:00", "end": "22:00"},
    "target_charging_window": {"start": "01:00", "end": "05:00"}
  },
  "scope": "national",
  "start_time": "2026-06-18T16:00:00Z",
  "end_time": "2026-06-19T04:00:00Z",
  "baseline_from_time": "2026-06-18T12:00:00Z",
  "hours": 24,
  "timezone": "Europe/Paris",
  "assumptions": {"source": "user what-if", "charging": "energy shifted, not added"}
}
```

## Example Result Shape

```json
{
  "result_id": "scenario-...",
  "request_hash": "...",
  "baseline_series": [{"timestamp": "2026-06-18T12:00:00Z", "demand_mw": 50540.0}],
  "scenario_series": [{"timestamp": "2026-06-18T14:00:00Z", "demand_delta_mw": 4140.0}],
  "demand_delta": {"total_mwh": 12420.0, "series": []},
  "peak_demand_delta_mw": 4140.0,
  "changed_watch_or_high_hours": [],
  "balance_context_delta": {"peak_score_delta": 0.11},
  "estimated_import_export_delta": {"net_import_delta_mwh_range": [1200.0, 4100.0]},
  "estimated_generation_response_range": {"flexible_generation_delta_mwh_range": [3100.0, 6800.0]},
  "estimated_carbon_range": {"total_tonnes_co2_delta_range": [993.6, 6831.0]},
  "regional_deltas": {"supported": true, "regions": []},
  "causal_chain": {"scenario_type": "cold_snap", "nodes": [], "edges": []},
  "assumptions": [],
  "caveats": [],
  "model_versions": {},
  "data_versions": {},
  "unsupported_grid_behaviours": []
}
```

Numeric values above are illustrative. The actual response includes full hourly series, assumptions, caveats, model versions, data versions, and provenance names.

## Unsupported Grid Behaviours

- No AC power-flow, voltage, congestion, or network constraint model is implemented.
- No unit-commitment, dispatch optimization, ramp-rate, storage, or balancing-market model is implemented.
- The residual bucket combines flexible domestic output and imports until a dispatch model exists.
- Regional demand forecasts are allocation context only and do not imply regional adequacy or shortage status.
- Carbon estimates are separate context and are not used in the modelled national balance status.
