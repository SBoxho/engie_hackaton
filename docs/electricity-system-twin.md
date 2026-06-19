# Electricity-System Twin

Date: 2026-06-19

This backend extension builds a time-indexed `TwinSnapshot` for the current or aligned source hour and for every requested forecast hour up to 48 hours:

```text
GET /v1/twin?from=<timestamp>&hours=48&region=<region-code>
```

The service aligns `from` to the latest available UTC hour at or before the requested timestamp. A `hours=48` request returns 49 snapshots: horizon 0 plus horizons 1 through 48.

## Demand Forecast

National demand uses the existing probabilistic demand forecast boundary:

```text
p50 = usual_demand_baseline + residual_model_correction
```

If no champion model artifact is available, the route is an explicit fallback:

```text
p50 = usual_demand_baseline
```

The interval is carried as `p10`, `p50`, and `p90` with route provenance. Current-hour demand uses observed ODRÉ/eCO2mix demand and collapses the interval to the observed value.

## Regional Demand Reconciliation

Regional demand is demand context only. It is not a regional adequacy, reserve, shortage, or balance status.

The initial regional method is a pooled comparable-history allocator:

```text
prelim_region_p50 =
  median(regional_demand_mw for matching local_hour and day_type)

reconciliation_factor =
  national_p50 / sum(prelim_region_p50 across regions)

reconciled_region_p50 =
  prelim_region_p50 * reconciliation_factor
```

Regional interval shape follows the national relative interval:

```text
regional_p10 = reconciled_region_p50 * (national_p10 / national_p50)
regional_p90 = reconciled_region_p50 * (national_p90 / national_p50)
```

Fallbacks are explicit:

- If a region has no comparable history, latest available regional demand is used.
- If no regional history exists, equal-share allocation is used and marked as fallback.

## Generation Context

Wind and solar estimates use this precedence:

1. Optional RTE generation forecast, when the credentialed adapter is configured and returns target-hour data.
2. Public-weather plus recent ODRÉ/eCO2mix statistical fallback.
3. Persistence fallback from recent ODRÉ/eCO2mix generation.

The weather fallback is:

```text
wind_estimate =
  recent_same_hour_odre_wind_median
  * clip(target_public_wind_speed / recent_same_hour_public_wind_speed, 0.25, 2.0)

solar_estimate =
  recent_same_hour_odre_solar_median
  * clip(target_public_solar_radiation / recent_same_hour_public_solar_radiation, 0.0, 1.8)
```

Nuclear is represented separately as expected output or availability:

```text
nuclear_expected_output =
  median_comparable_nuclear_output_mw
  - active_announced_nuclear_unavailability_mw
```

Publicly announced generation unavailability is ingested only when optional RTE credentials and adapter wiring are configured. Without that optional source, the twin records the source as unavailable; it does not invent outage data.

Uncertain flexible domestic output and imports are combined into an explicit residual bucket until a dispatch model exists:

```text
residual_flexible_sources_and_imports =
  median_comparable(hydro + gas + coal + oil + bioenergy + net_imports)

estimated_generation_total =
  nuclear + wind + solar + residual_flexible_sources_and_imports
```

Every component carries one provenance kind:

- `official_forecast`
- `statistical_estimate`
- `persistence_fallback`
- `residual_estimate`
- `observed`
- `unavailable`

## Exchange and Carbon

Exchange is exposed separately when defensible:

```text
net_imports = observed current exchange
           or median comparable recent exchange
           or persistence fallback
```

Carbon intensity is a separate context field. It is never an input to the modelled balance context.

## Modelled National Balance Context

The national balance context is not EcoWatt and does not claim to calculate actual operational reserve margin. It is an analytical context score configured in `data/config/status_thresholds.json`.

Current method:

```text
residual_load = forecast_demand_p50 - wind_estimate - solar_estimate

residual_load_percentile =
  percentile(residual_load within comparable recent historical residual-load values)

announced_unavailability_ratio =
  active_announced_unavailability_mw / forecast_demand_p50

announced_unavailability_score =
  min(announced_unavailability_ratio / announced_unavailability_high_ratio, 1.0)

balance_score =
  residual_load_weight * residual_load_percentile
  + announced_unavailability_weight * announced_unavailability_score
```

Thresholds, weights, comparable keys, and normalizers are versioned in `status_thresholds.json` under `modelled_balance_context`. The snapshot exposes the contributing components and the threshold configuration version.

## Unsupported Physical-Grid Behaviours

The twin does not model:

- AC power flow, voltage, congestion, or network constraints.
- Unit commitment, dispatch optimization, ramp rates, storage, or balancing markets.
- Actual operational reserve margin.
- Foreign generation mix behind imports.
- Regional adequacy or regional shortage status.
- Verified emissions accounting for forecast hours.

