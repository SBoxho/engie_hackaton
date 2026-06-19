# Probabilistic 48-hour demand forecast

This pipeline forecasts national electricity demand as:

```text
forecast demand = transparent usual-demand baseline + machine-learning residual correction
```

The usual-demand baseline remains the production fallback. A trained residual model is used only when its artifact status is `champion`.

## Retraining

Build the usual-demand dataset with the full 48-hour horizon set, then train the candidate:

```powershell
python -m scripts.build_usual_demand_dataset --horizon 1 --horizon 2 --horizon 3 --horizon 4 --horizon 5 --horizon 6 --horizon 7 --horizon 8 --horizon 9 --horizon 10 --horizon 11 --horizon 12 --horizon 13 --horizon 14 --horizon 15 --horizon 16 --horizon 17 --horizon 18 --horizon 19 --horizon 20 --horizon 21 --horizon 22 --horizon 23 --horizon 24 --horizon 25 --horizon 26 --horizon 27 --horizon 28 --horizon 29 --horizon 30 --horizon 31 --horizon 32 --horizon 33 --horizon 34 --horizon 35 --horizon 36 --horizon 37 --horizon 38 --horizon 39 --horizon 40 --horizon 41 --horizon 42 --horizon 43 --horizon 44 --horizon 45 --horizon 46 --horizon 47 --horizon 48
python -m scripts.train_probabilistic_demand_forecast
```

The trainer prefers LightGBM quantile regression for P10, P50, and P90. If LightGBM is not installed and fallback is allowed, it uses scikit-learn `HistGradientBoostingRegressor(loss="quantile")`, which is the existing deterministic tree-based equivalent used for local reproducibility.

## Validation and promotion

Validation is rolling and time ordered. The candidate is rejected unless:

- P50 MAE improves overall versus the strongest eligible baseline.
- P50 MAE improves in a majority of rolling validation periods.

Baselines are:

- `usual_demand`: comparable-history usual-demand baseline.
- `seasonal_naive`: previous-week same target timestamp.
- `rte_public_forecast`: reported separately when RTE J/J+1 forecast columns exist.

## Reported metrics

The registry and model card include:

- MAE in GW.
- WAPE.
- Pinball loss for P10, P50, and P90.
- P10-P90 empirical coverage.
- Mean interval width.
- Metrics by forecast horizon.
- Metrics by season.
- Metrics for peak periods, defined as validation rows at or above the validation actual-demand P90.
- RTE public forecast comparison when required columns are available.

## Artifacts

Default output directory: `data/processed/demand_forecast/`.

- `demand_residual_quantile_model.pkl`: model bundle, feature schema, imputation values, config, metrics, and model card.
- `validation_predictions.parquet`: rolling validation predictions.
- `model_card.json`: model card data structure.
- `artifact_manifest.json`: lightweight registry manifest with model version, training period, feature and dataset versions, data cutoff, metrics, status, rejection reason, and artifact checksums.

If the manifest status is `rejected`, inference must continue with the usual-demand baseline.

## Inference

Use `DemandForecastService` from `src.models.probabilistic_demand`:

```python
from src.models.probabilistic_demand import DemandForecastService

service = DemandForecastService(artifact_path="data/processed/demand_forecast/demand_residual_quantile_model.pkl")
run = service.forecast("2026-01-10T00:00:00Z", hourly_features)
points = run.to_frame()
```

The service returns 48 hourly points with UTC target timestamps, local `Europe/Paris` timestamps, P10, P50, P90, route, fallback reason, and baseline provenance. When no champion model exists, the route is `baseline_fallback`.

## Leakage policy

Feature rows are anchored at a forecast origin. Demand lags, rolling demand, physical context, and weather fields must be available at or before that origin. Deterministic target calendar fields are allowed. Future observed target demand, future observed weather, target observation timestamps, and comparison-only RTE forecast columns are not model features.

Historical weather forecast fields can be used only when their provenance shows they were available at the historical forecast origin. Otherwise the model uses origin-available weather context and missingness indicators.

## Explainability layer

The API explainability layer applies SHAP `TreeExplainer` to the champion P50
residual tree model when one is available. The usual-demand baseline is shown
separately, so the reconciliation equation is:

```text
p50 = usual-demand baseline + residual expected value + grouped residual SHAP contributions
```

Public payloads group correlated raw inputs into Weather, Calendar, Time of
day, Recent demand, Regional or national pattern, Generation and exchange
context, and Data-quality fallback. Raw SHAP values remain available in the
technical explanation payload for expert inspection. The numerical
reconciliation tolerance is 0.001 GW.

Confidence is not derived from SHAP. It is based on measurable diagnostics such
as horizon, P10-P90 interval width, weather-model disagreement when present,
fallback or missing feature flags, out-of-distribution score when present, and
recent validation error when present.

All SHAP and feature-delta explanations are labelled as model explanations, not
causal proof.

Relevant API routes:

- `GET /v1/forecast?scope=france&hours=48`
- `GET /v1/forecast/{run-id}`
- `GET /v1/explanations?run_id=<id>&timestamp=<time>`
- `GET /v1/model-card`
- `GET /v1/forecast-changes?current=<id>&previous=<id>`
