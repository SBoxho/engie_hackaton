# Experimental weather-aware demand model

This milestone adds an honest demand-model pipeline that is evaluated against the existing baselines. It is not an RTE operational forecast.

## Commands

Run from the repository root:

```powershell
python -m scripts.build_features --min-continuous-hours 48
python -m scripts.train_demand_model
python -m scripts.evaluate_demand_model
```

The defaults read `data/processed/eco2mix`, join `data/processed/weather_national.parquet` when present, and write generated artifacts under `data/processed/demand_model/`. The `data/processed/*` ignore rule keeps feature tables, model binaries, and evaluation outputs out of Git.

For consolidated historical éCO2mix, the observed cadence can be 30 minutes rather than the near-live 15-minute cadence. Feature generation infers the cadence mode by default and records it in metadata. Use `--cadence-minutes` only when you need an explicit override.

## Multi-season backfill

Calendar year 2024 is a practical bounded example because it covers winter, spring, summer, and autumn while staying within the consolidated demand fetcher's one-request limit:

```powershell
python -m scripts.fetch_historical --start 2024-01-01 --end 2025-01-01 --output data/processed/eco2mix_historical_2024_clean.parquet
python -m scripts.fetch_weather --start 2024-01-01 --end 2024-12-31 --output data/processed/weather_national_2024.parquet --joined-output data/processed/energy_weather_2024.parquet --strict
python -m scripts.build_features --start 2024-01-01 --end 2025-01-01 --weather data/processed/weather_national_2024.parquet --min-continuous-hours 168
python -m scripts.train_demand_model
python -m scripts.evaluate_demand_model
python -m scripts.backtest_baselines --start 2024-01-01 --end 2025-01-01
```

The final baseline command writes the dashboard baseline artifact for the same calendar period; the model evaluation artifact also includes baseline metrics recomputed on the model's untouched test origins.

## Data validation

`scripts.build_features` inspects the available processed demand data before modeling. The feature metadata records:

- Resolved demand cadence, expected interval count at that cadence, missing interval count, duplicates, off-grid timestamps, and missing target count.
- Continuous non-missing demand periods. Training rows are allowed only inside periods at least `--min-continuous-hours` long.
- Weather row count, overlap with energy timestamps, population coverage, and missing weather-feature rows.
- Required and extra columns, so schema drift is visible.

Demand targets are never fabricated, interpolated, forward-filled, or backward-filled. Missing target intervals break continuity and are excluded from supervised training/evaluation rows.

## Features

Each supervised row is anchored at a UTC forecast origin and direct horizon. The target timestamp is exactly `origin + horizon`.

Demand features use observations whose source time is at or before the origin:

- Origin demand.
- Exact demand lags at 1h, 3h, 6h, 24h, and 168h when available.
- Shifted rolling demand mean/std over 1h, 4h, and 24h. Rolling windows shift by one 15-minute interval before aggregation.

Calendar features use Europe/Paris local time while storing timestamps in UTC:

- Origin and target hour, weekday, month, season, weekend, French holiday, DST flag, UTC offset, and cyclic hour/weekday terms.
- Target calendar features are deterministic future calendar values, not observed future measurements.

Weather features are population-weighted Open-Meteo fields joined at the origin timestamp:

- Temperature, wind, cloud cover, shortwave radiation, humidity, population coverage, city availability, source age, and missing indicators.
- Weather provenance is rejected when `weather_source_timestamp_max > origin`.

## Model

Training uses scikit-learn `HistGradientBoostingRegressor`, one deterministic direct model per eligible horizon: 1h, 3h, 6h, and 24h. The trainer stores the feature schema used by each horizon because short histories can make long-lag features all-missing or constant. Such non-informative columns are excluded for that horizon instead of being silently imputed.

The trainer also fits lower and upper quantile regressors for each eligible horizon using `loss="quantile"` at the 0.10 and 0.90 quantiles. The evaluation artifact stores these as an 80% central prediction interval for each forecast row. If an older model artifact does not contain quantile models, evaluation falls back to an empirical residual interval so uncertainty is still shown rather than omitted silently.

Default training parameters are CPU-friendly:

- `learning_rate=0.05`
- `max_iter=80`
- `max_leaf_nodes=31`
- `l2_regularization=0.05`
- `early_stopping=False`
- `random_state=42`

## Splits and evaluation

Evaluation is chronological. For each horizon:

- Valid supervised rows are sorted by target timestamp.
- The last chronological fraction is held out as the untouched test period.
- Earlier rows are used for expanding-window validation and final training.
- The model and persistence, previous-day, and previous-week baselines are scored on the exact same test origins and target timestamps.

Metrics are MAE, RMSE, sMAPE, sample count, coverage, and improvement versus the strongest eligible baseline by MAE. Baselines with zero usable samples, such as previous-week on histories shorter than seven days, are reported but not treated as eligible. Hour and season metrics are emitted only when a segment has enough samples.

Each horizon also gets a trust summary: model MAE, strongest-baseline MAE, improvement percentage, test coverage, interval coverage, interval width, and a reliability badge. A positive improvement gets `Model edge detected`; horizons that do not beat the strongest eligible baseline get `Experimental horizon`. Weak horizons remain in the dashboard and artifact.

## Explainability

Evaluation also emits a first local XAI layer for each prediction. The method is a lightweight grouped ablation: for one forecast row, each feature family is replaced with deterministic reference values from the training period, and the model prediction is compared with the original prediction.

Feature families are reported in user-facing language:

- Weather: temperature, humidity, cloud cover, radiation, and wind.
- Calendar: hour, weekday, weekend, holiday, season, DST, and horizon timing.
- Recent demand: current demand, 1h/3h/6h lags, and rolling demand statistics.
- Weekly pattern: the 168h lag.
- Data quality/provenance: weather coverage, missing indicators, city coverage, and source age.

Each prediction record contains 2-4 readable explanation cards plus a compact technical contribution list for the dashboard expander. These explanations are approximate, model-derived sensitivity checks. They describe associations learned by the model and must not be read as causal explanations.

## Artifacts

Generated artifact layout:

- `features.parquet`: supervised feature rows with origin/target timestamps, horizon, target, continuity flags, and model features.
- `feature_metadata.json`: schema version, feature columns, target column, source, coverage audit, weather audit, leakage controls, and source digest.
- `demand_hgb_model.pkl`: pickled model bundle containing schema version, model kind, feature metadata, train config, per-horizon feature schemas, per-horizon point and quantile models, split periods, validation metrics, interval definition, and skipped horizon reasons.
- `evaluation.json`: model metrics, baseline metrics, improvement rows with trust badges, segment metrics, prediction-level records with point forecasts, lower/upper interval bounds, local explanation cards, data audit, training periods, generation timestamp, and the experimental forecast and explanation disclaimers.

Artifact validation rejects unsupported schema versions, missing feature columns, missing per-horizon models, missing per-horizon feature schemas, target/horizon misalignment, and future weather provenance.

## Limitations

The current local cache may contain only short islands of official history. That is enough for smoke testing but not enough to claim robust model skill, especially across seasons, holidays, weather regimes, and previous-week comparisons. Model performance depends on obtaining a sufficiently long, continuous historical/weather dataset.

When the model does not beat the strongest eligible baseline, the result should be read literally. The pipeline is designed to make that visible rather than hide it.
