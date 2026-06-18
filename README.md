# Energy Pulse France

Energy Pulse France is a Python-first Streamlit application for trustworthy French electricity analysis. It preserves official raw payloads, standardizes RTE éCO2mix observations, maintains idempotent Parquet partitions, joins population-weighted weather, reports data quality, evaluates transparent demand baselines, and calibrates an educational grid-mood indicator.

No live, historical, weather, or forecast values are fabricated. Cached values are labelled as cached, and baseline forecasts are explicitly not described as AI.

## UX structure

The Streamlit entry page is now a story-first public demo: a hero, current grid pulse cards, a 24-hour demand-pressure timeline, plain-language driver cards, a demand-shifting simulator link, and a model-honesty box. Raw dataframes, calibration details, data quality checks, historical views, baseline backtests, and the experimental demand model remain available from the **Advanced / Data Science** section so non-technical reviewers see the energy-weather story first while technical reviewers can still inspect the evidence.

## Official data sources

| Data | Access | Dataset / reference | Notes |
|---|---|---|---|
| Near-live national electricity | [ODRÉ/RTE éCO2mix](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/) | `eco2mix-national-tr`, Opendatasoft Explore API v2.1 | Public, no key; nominal 15-minute observations; rolling coverage and publication latency apply. |
| Consolidated national history | [ODRÉ/RTE consolidated éCO2mix](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/) | `eco2mix-national-cons-def`, Opendatasoft Explore API v2.1 | Public, Licence Ouverte 2.0; coverage starts in 2012; historical cadence/schema can differ from near-live data. |
| Weather | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) and [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Hourly temperature, wind, cloud cover, shortwave radiation, humidity | Public, no key; requested in UTC and cached by city/date. |
| School holidays | [data.education.gouv.fr calendrier scolaire](https://data.education.gouv.fr/explore/dataset/fr-en-calendrier-scolaire/) | `fr-en-calendrier-scolaire`, Opendatasoft Explore API v2.1 | Official open school calendar; transformed to deterministic Zone A/B/C flags. |
| City weights | [INSEE legal populations 2022](https://www.insee.fr/fr/statistiques/8290591) | Municipal population, ten major metropolitan-France communes | An auditable urban-demand proxy, not a complete national population model. |

Detailed source contracts and limitations are in `docs/`.

## Install

Python 3.11 or newer is recommended. A fresh clone can run the public demo without private files, raw API caches, model binaries, or credentials.

```powershell
git clone <your-fork-or-repo-url>
cd engie_hackaton
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_app.py
```

For local development, tests, and model training, install the extra developer dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

No credential is needed for RTE/ODRE or Open-Meteo. Optional secrets are only used if you choose to enable live integrations.

## Demo and deployment mode

The app defaults to `APP_MODE=demo` so a clean clone can boot from the small committed `demo_data/` bundle without raw datasets, trained model binaries, or network calls. The UI shows **Demo data mode** when this path is active. Use `APP_MODE=live` for the normal fetch/cache/pipeline behavior.

```powershell
# Reliable hackathon/demo run: uses only demo_data/
$env:APP_MODE="demo"
$env:DEMO_ALLOW_EXTERNAL_API="0"
python run_app.py

# Full local/live run: uses existing data pipelines and live fallbacks.
$env:APP_MODE="live"
python run_app.py
```

Demo mode never calls external APIs unless `DEMO_ALLOW_EXTERNAL_API=1` is explicitly set. Keep that value at `0` for Streamlit Community Cloud, Hugging Face Spaces, and hackathon judging. In this judge-safe state the homepage sidebar reports `APP_MODE=demo, DEMO_ALLOW_EXTERNAL_API=0`, the app shows a **Judge mode: demo data** cue on the forecast, explainability, and simulator pages, and CI smoke tests fail if those pages attempt a network request.

To refresh the committed demo bundle after regenerating local processed artifacts:

```powershell
python -m scripts.export_demo_bundle
```

The exporter writes only small safe artifacts to `demo_data/`: a 7-14 day energy sample, trimmed demand-model evaluation, one model forecast artifact, data-quality report, optional EcoWatt sample, optional weather sample, baseline sample, and mood calibration. It does not copy raw API payloads or `demand_hgb_model.pkl`.

## Public Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub with `demo_data/`, `.streamlit/config.toml`, `requirements.txt`, `runtime.txt`, and `app/main.py` committed.
2. In Streamlit Community Cloud, create a new app from the GitHub repository.
3. Set the main file path to exactly:

```text
app/main.py
```

Do not set `run_app.py` as the hosted entrypoint. Streamlit Cloud already runs the selected file with `streamlit run`; `run_app.py` is only a local convenience wrapper for `python run_app.py`.

4. Set environment variables or app secrets:

```text
APP_MODE="demo"
DEMO_ALLOW_EXTERNAL_API="0"
ENERGY_PULSE_TIMEZONE="Europe/Paris"
ENERGY_PULSE_HISTORY_HOURS="72"
```

5. Deploy. The sidebar should show `Ready for public demo` or only optional missing artifacts.

For final judging, open the homepage, **Forecast Cockpit**, **Forecast Explainability**, and **Flatten the Peak** simulator once after deployment. Each should show the demo-mode cue and load from the committed artifacts without requiring credentials or a warm local cache.

No Streamlit secrets are required for the default demo deployment. Do not upload `.env`, `data/raw/`, `data/processed/`, or local virtual environments.

### Hugging Face Spaces Backup

1. Create a new Hugging Face Space with the Streamlit SDK.
2. Add this repository as the Space content, or mirror the GitHub repository into the Space.
3. Use `requirements.txt` as the Space dependency file.
4. Set `app/main.py` as the Streamlit entrypoint. If a custom command is needed, use:

```bash
streamlit run app/main.py --server.port 7860 --server.address 0.0.0.0
```

5. Add the same environment variables:

```text
APP_MODE=demo
DEMO_ALLOW_EXTERNAL_API=0
ENERGY_PULSE_TIMEZONE=Europe/Paris
ENERGY_PULSE_HISTORY_HOURS=72
```

### Expected Environment Variables

| Variable | Required | Purpose |
|---|---:|---|
| `APP_MODE` | No | `demo` for public deterministic demo, `live` for fetch/cache behavior. Defaults to `demo`. |
| `DEMO_ALLOW_EXTERNAL_API` | No | Set to `0` for offline demo deployments. |
| `ENERGY_PULSE_TIMEZONE` | No | Display timezone. Defaults to `Europe/Paris`. |
| `ENERGY_PULSE_HISTORY_HOURS` | No | Recent window for live/cached data. Defaults to `72`. |
| `ODRE_BASE_URL` | No | Override the public ODRE API base URL for testing. |
| `OPEN_METEO_BASE_URL` | No | Override the Open-Meteo endpoint for testing. |
| `RTE_ECOWATT_API_TOKEN` | No | Optional live EcoWatt token. Not needed for demo. |
| `ENTSOE_API_TOKEN` | No | Reserved for future ENTSO-E integration. Not used by this app. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| App says demo energy sample is unavailable | Commit or regenerate `demo_data/energy_recent.parquet` with `python -m scripts.export_demo_bundle`. |
| Sidebar health shows required artifacts missing | Verify `demo_data/manifest.json`, `quality_report.json`, `baseline_backtest.json`, `demand_model_evaluation.json`, and `mood_calibration.json` are in the repository. |
| Streamlit Cloud fails during dependency install | Use `requirements.txt`, not `requirements-dev.txt`; keep `runtime.txt` committed so the hosted Python version stays on 3.12. |
| Streamlit Cloud was configured with `run_app.py` | Change the app's main file path to `app/main.py`, then reboot or redeploy. |
| Hosted app unexpectedly calls APIs | Set `APP_MODE=demo` and `DEMO_ALLOW_EXTERNAL_API=0`, then clear Streamlit's cache or redeploy. |
| Demand-model page has no predictions | Refresh the demo bundle after creating `data/processed/demand_model/evaluation.json`. The public demo does not require a `.pkl` model file. |
| Live mode has no data | Run `python -m scripts.update_data --hours 72` locally, or switch back to `APP_MODE=demo` for public judging. |

## Reproduce the pipeline

Run commands from the repository root.

```powershell
# 1. Consolidated official history; writes an immutable raw snapshot,
#    a compatibility export, and year/month Parquet partitions.
python -m scripts.fetch_historical --start 2024-01-01 --end 2024-02-01

# 2. Incremental near-live update. Repeating it is idempotent.
python -m scripts.update_data --hours 72
python -m scripts.update_data --hours 72

# A bounded historical update can also use the unified command.
python -m scripts.update_data --start 2024-01-01 --end 2024-02-01

# 3. Multi-city weather plus energy/weather joined Parquet output.
python -m scripts.fetch_weather --start 2024-01-01 --end 2024-01-31

# 4–7. Quality, baselines, and calibrated mood artifacts.
python -m scripts.run_quality_checks
python -m scripts.backtest_baselines
python -m scripts.calibrate_mood

# 8. Experimental weather-aware demand model.
python -m scripts.build_features
python -m scripts.train_demand_model
python -m scripts.evaluate_demand_model

# 9. Dashboard.
python run_app.py
```

Use `python -m scripts.update_data --offline` to process the newest immutable near-live cache without a network call. Add `--strict` to the weather command when any missing city should fail the run.

## Integrated architecture

- `data/raw/`: immutable or source-faithful JSON caches with query and retrieval provenance; ignored by Git.
- `data/processed/eco2mix/year=YYYY/month=MM/data.parquet`: atomic, idempotent processed partitions keyed by UTC timestamp and region.
- `data/processed/weather_national.parquet`: population-weighted weather with source timestamps and coverage diagnostics.
- `data/processed/energy_weather.parquet`: weather joined backward to the electricity timeline, preventing future-data leakage.
- `data/processed/baseline_backtest.json`: deterministic prediction-level and metric artifact.
- `data/processed/demand_model/features.parquet`: supervised demand-model features generated from exact UTC forecast origins; ignored by Git.
- `data/processed/demand_model/feature_metadata.json`: feature schema, leakage controls, source coverage, and weather coverage.
- `data/processed/demand_model/demand_hgb_model.pkl`: generated scikit-learn model bundle; ignored by Git.
- `data/processed/demand_model/evaluation.json`: model-versus-baseline metrics and prediction records for untouched chronological test periods.
- `data/processed/mood_calibration.json`: quantile thresholds, source period, sample sizes, fallback metadata, and generation time.
- `demo_data/`: deployment-safe demo artifacts used when `APP_MODE=demo`; tracked in Git and independent of raw caches or trained model binaries.

The standardized electricity schema includes explicit UTC timestamps, region, consumption, generation by source, imports/exports, source CO₂ intensity, total/renewable/fossil production, and renewable/fossil shares. Suspicious records remain available as quality evidence; quality checks do not silently clean them away.

## Data quality

The report classifies required-schema, timestamp, duplicate, cadence, missing-interval, null, nonnegative-generation, share-range, freshness, extreme-value, and supply/demand residual checks as errors, warnings, or information. Defaults are screening thresholds, not corrections or RTE operational limits.

```powershell
python -m scripts.run_quality_checks --start 2024-01-01 --end 2024-02-01
```

## Forecast baselines

The backtest uses chronological rolling origins and exact target timestamps for persistence, previous-day, and previous-week seasonal-naive forecasts at 1, 3, 6, and 24 hours. It reports MAE, RMSE, sMAPE, sample count, missing targets, and coverage. These are reference rules that a future ML model must beat—not an AI model or production-quality forecast.

## Experimental demand model

The demand model is a CPU-friendly scikit-learn `HistGradientBoostingRegressor`, trained as one direct model per 1, 3, 6, and 24 hour horizon. It uses only features available at the forecast origin: observed demand lags, shifted rolling demand statistics, Europe/Paris calendar features across DST, holiday flags, and population-weighted weather joined with source provenance no later than the origin. Target demand is never interpolated.

```powershell
python -m scripts.build_features --min-continuous-hours 48
python -m scripts.train_demand_model
python -m scripts.evaluate_demand_model
```

Feature generation infers the observed demand cadence, so consolidated 30-minute history and near-live 15-minute history are both handled without target interpolation. The evaluator uses chronological train/test splits and compares the model against persistence, previous-day, and previous-week baselines on the exact same test origins. The Streamlit **Demand model** page shows recent actuals versus model and baseline predictions, horizon metrics, training/evaluation periods, data freshness, weather coverage, and artifact timestamps. The page labels the model as experimental and not an RTE operational forecast.

To backfill a continuous multi-season training set, use a bounded consolidated demand window and the matching historical weather window:

```powershell
python -m scripts.backfill_multiyear --start-year 2019 --end-year 2025 --strict-weather --train --evaluate
```

The command fetches each consolidated éCO2mix year separately to preserve raw ODRÉ cache snapshots, upserts the clean rows into the existing year/month Parquet store, rebuilds weather features on the actual stored energy timestamps, caches the official school calendar, generates supervised features, and optionally trains/evaluates the model. Use `--start-year 2018 --end-year 2025` for a longer window when the external services and local runtime can handle the fetch.

See `docs/demand_model.md` for features, split assumptions, artifact format, limitations, and interpretation guidance. Model performance depends on obtaining a sufficiently long, continuous historical demand, weather, and calendar dataset; short cached slices can train a smoke-test model but must not be read as evidence of forecasting skill.

## Grid mood

Calibration uses Europe/Paris local hour and meteorological season. Transparent historical quantiles define high demand, high/low CO₂, high renewable share, and high fossil share. The fallback order is season/hour, season, hour, global, then explicit fixed thresholds. Decision precedence is Carbon-heavy, Tense, Renewable-rich, Calm. The result exposes its reason, thresholds, segment, sample count, and fallback status.

Grid mood is an educational indicator, not an RTE operational alert or grid-security assessment.

## Test

```powershell
python -m compileall -q app src scripts
python -m pytest -q
```

Real-service smoke tests are opt-in and skip gracefully offline:

```powershell
$env:RUN_REAL_DATA_TESTS="1"
$env:RUN_LIVE_DATA_SMOKE="1"
python -m pytest -q
```

## Known limitations

- ODRÉ and Open-Meteo availability and publication latency are external dependencies; raw caches support explicit offline fallback.
- The consolidated series can differ from the near-live schema and cadence. Quality reports expose cadence gaps; baseline coverage records unavailable exact targets.
- Population weighting covers ten large communes and is an urban exposure proxy, not all residents or regional weather diversity.
- Hourly weather is aligned backward to quarter-hours. It is leakage-safe but does not create new intra-hour observations.
- The experimental demand model is only as good as the continuous history available. It may correctly report that baselines are stronger, that week-naive baselines are ineligible, or that there is insufficient data to train.
- Source CO₂ intensity is retained; imported electricity is not decomposed by foreign generation mix.
- Multi-partition writes are atomic per month, not as one cross-month transaction.

## Next AI tasks

1. Extend the continuous official history and weather backfill, then retrain until the model is evaluated over multiple seasons and holidays.
2. Add probabilistic intervals and calibration diagnostics, especially around holidays and extreme weather.
3. Add regional éCO2mix and weather features with hierarchical validation.
4. Explain validated model predictions with time-safe SHAP analysis and plain-language driver summaries.
5. Monitor schema drift, forecast drift, and segment-level errors before any operational pilot.
