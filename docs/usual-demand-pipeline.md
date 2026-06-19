# Usual-demand feature pipeline

This pre-ML pipeline builds hourly national and regional analytical rows from normalized public data, then evaluates a transparent comparable-history baseline.

## Commands

Build hourly and supervised feature rows:

```powershell
python -m scripts.build_usual_demand_dataset
```

Run the usual-demand rolling backtest:

```powershell
python -m scripts.backtest_usual_demand
```

Default outputs:

- `data/processed/usual_demand/hourly_features.parquet`
- `data/processed/usual_demand/baseline_training.parquet`
- `data/processed/usual_demand/feature_manifest.json`
- `data/processed/usual_demand/data_quality_report.json`
- `data/processed/usual_demand/feature_coverage_report.json`
- `data/processed/usual_demand/usual_demand_predictions.parquet`
- `data/processed/usual_demand/usual_demand_backtest.json`

Both commands can read the normalized public-data store under `data/public`. For offline smoke runs, `build_usual_demand_dataset` also accepts `--input`, `--energy-input`, `--weather-input`, `--public-holidays-input`, and `--school-holidays-input`.

## Resampling Contract

Hourly timestamps are UTC hour ends. A timestamp `T` summarizes source energy records in `[T-1h, T)`, so completed-hour power can be used at forecast origin `T`.

- MW power fields such as demand, generation, imports, exports, and physical exchanges are averaged within the hour.
- Energy quantity columns ending in `_mwh`, `_kwh`, or `_wh` are summed within the hour.
- CO2 intensity and weather fields are averaged.
- Quality and fallback fields are converted to counts, fractions, and max source timestamps.
- Regional MW values are summed across regions only to create a derived national row when no official national row exists for that hour.

## Leakage Controls

Every supervised feature row carries `feature_available_at`. The builder rejects rows where `feature_available_at > origin_timestamp`. Weather is joined backward by public weather event time and availability time. Baseline comparison samples are restricted to observations with `timestamp <= origin_timestamp`.

The target demand is retained as `target_mw` for training/backtesting only. Deterministic target calendar features are allowed because holidays, weekdays, seasons, and local hours are known ahead of the forecast origin.

## Usual-demand Baseline

The baseline uses the median of comparable historical demand for the same geography. It tries:

1. Same local hour, weekday/weekend type, season, and holiday type.
2. Same local hour, weekday/weekend type, and season.
3. Same local hour and weekday/weekend type.
4. Same local hour.
5. Recent rolling seasonal median.

Each prediction exposes `usual_demand_method`, `usual_demand_sample_count`, `usual_demand_fallback_level`, and `actual_above_usual_percent`, enabling statements like "national demand is 8% above usual."

## Reported Metrics

`usual_demand_backtest.json` reports MAE in GW, WAPE, error by horizon, season, weekday/weekend type, region, and fallback level, plus weak-data periods where history is sparse or high fallback levels dominate.
