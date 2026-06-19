from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any
from wsgiref.util import setup_testing_defaults

import numpy as np
import pandas as pd
import pytest

from src.api.forecast_explanations import ForecastExplanationApiService
from src.api.server import create_app
from src.models.forecast_explainability import (
    CONCEPTS,
    CONCEPT_DATA_QUALITY,
    CONCEPT_RECENT_DEMAND,
    CONCEPT_WEATHER,
    assess_forecast_confidence,
    build_hourly_explanations,
    concept_for_feature,
    explain_forecast_changes,
)
from src.models.probabilistic_demand import (
    CONFIG_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_FAMILY,
    MODEL_SCHEMA_VERSION,
    forecast_with_artifact,
)
from src.models.usual_demand import build_hourly_analytical_dataset


NOW = datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc)
GENERATED_AT = "2026-02-12T12:00:00Z"


@pytest.fixture(scope="module")
def shap_forecast_fixture() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestRegressor

    feature_columns = [
        "usual_demand_mw",
        "weather_temperature_c",
        "heating_degree_c",
        "target_hour_of_day",
        "target_is_weekend",
        "demand_lag_1h_mw",
        "demand_roll_24h_mean_mw",
        "national_net_imports_mw",
        "source_fallback_record_count",
        "energy_incomplete_hour",
        "horizon_hours",
    ]
    train = _training_matrix(feature_columns)
    residual = (
        -90.0 * train["weather_temperature_c"]
        + 120.0 * train["heating_degree_c"]
        + 45.0 * train["target_hour_of_day"]
        + 0.035 * (train["demand_lag_1h_mw"] - 51_000)
        + 0.10 * train["national_net_imports_mw"]
        + 280.0 * train["target_is_weekend"]
        - 350.0 * train["source_fallback_record_count"]
    )
    models = {}
    for key, offset in {"p10": -850.0, "p50": 0.0, "p90": 850.0}.items():
        model = RandomForestRegressor(n_estimators=32, max_depth=5, random_state=13)
        model.fit(train[feature_columns], residual + offset)
        models[key] = model
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_version": "test-residual-tree-v1",
        "status": "champion",
        "feature_columns": feature_columns,
        "feature_imputation_values": {column: float(train[column].median()) for column in feature_columns},
        "models": models,
        "train_config": {"baseline_min_samples": 1, "baseline_recent_days": 28},
        "metrics": {
            "overall": {"mae_gw": 0.65},
            "by_horizon": [
                {"horizon_hours": 1, "mae_gw": 0.55},
                {"horizon_hours": 24, "mae_gw": 0.95},
                {"horizon_hours": 48, "mae_gw": 1.80},
            ],
        },
        "model_card": {"limitations": []},
    }
    rows = pd.DataFrame(
        [
            _inference_row(hour=1, temperature=4.0, heating=14.0, usual=52_000.0),
            _inference_row(hour=24, temperature=7.0, heating=11.0, usual=50_500.0),
            _inference_row(hour=48, temperature=1.0, heating=17.0, usual=54_000.0),
        ]
    )
    forecast = forecast_with_artifact(artifact, rows)
    points = build_hourly_explanations(
        artifact=artifact,
        feature_rows=rows,
        forecast_frame=forecast,
        run_id="forecast-shap-test",
        generated_at=GENERATED_AT,
        source_freshness={"state": "fresh", "latest_timestamp": "2026-02-12T12:00:00Z"},
    )
    return artifact, rows, forecast, points


def test_shap_additivity_reconciles_p50(shap_forecast_fixture) -> None:
    _, _, _, points = shap_forecast_fixture
    first = points[0]
    reconciliation = first["technical"]["reconciliation"]

    assert first["model_explanation_available"] is True
    assert reconciliation["within_tolerance"] is True
    assert abs(reconciliation["error_gw"]) <= reconciliation["tolerance_gw"]
    assert len(first["technical"]["raw_shap_values"]) > 0


def test_grouped_contribution_additivity(shap_forecast_fixture) -> None:
    _, _, _, points = shap_forecast_fixture
    first = points[0]

    grouped_sum = sum(item["contribution_gw"] for item in first["concept_contributions"])
    reconstructed = (
        first["usual_demand_baseline_gw"]
        + first["residual_model_expected_value_gw"]
        + grouped_sum
    )

    assert reconstructed == pytest.approx(first["expected_demand_gw"], abs=0.001)
    assert {item["concept"] for item in first["concept_contributions"]} == set(CONCEPTS)


def test_gw_unit_conversion_for_raw_and_grouped_values(shap_forecast_fixture) -> None:
    _, _, forecast, points = shap_forecast_fixture
    first = points[0]
    raw = first["technical"]["raw_shap_values"][0]

    assert raw["value_gw"] == pytest.approx(raw["value_mw"] / 1000.0, abs=1e-9)
    assert first["expected_demand_gw"] == pytest.approx(float(forecast["p50"].iloc[0]) / 1000.0)
    assert first["usual_demand_baseline_gw"] == pytest.approx(float(forecast["usual_demand_mw"].iloc[0]) / 1000.0)


def test_correlated_raw_features_are_grouped_for_casual_users(shap_forecast_fixture) -> None:
    _, _, _, points = shap_forecast_fixture
    first = points[0]

    assert concept_for_feature("weather_temperature_c") == CONCEPT_WEATHER
    assert concept_for_feature("heating_degree_c") == CONCEPT_WEATHER
    assert concept_for_feature("demand_lag_1h_mw") == CONCEPT_RECENT_DEMAND
    assert concept_for_feature("demand_roll_24h_mean_mw") == CONCEPT_RECENT_DEMAND
    assert concept_for_feature("source_fallback_record_count") == CONCEPT_DATA_QUALITY
    for driver in first["top_positive_concept_drivers"] + first["top_negative_concept_drivers"]:
        assert driver["concept"] in CONCEPTS
        assert "_mw" not in driver["concept"].lower()
        assert "temperature_c" not in driver["concept"].lower()


def test_confidence_logic_uses_measurable_inputs_not_shap(shap_forecast_fixture) -> None:
    artifact, rows, forecast, _ = shap_forecast_fixture
    high = assess_forecast_confidence(
        rows.iloc[0],
        {**forecast.iloc[0].to_dict(), "p10": 51_700.0, "p90": 52_300.0, "route": "validated_model"},
        artifact=artifact,
        source_freshness={"state": "fresh"},
    )
    low_row = rows.iloc[2].copy()
    low_row["weather_model_disagreement_c"] = 5.0
    low_row["ood_score"] = 0.92
    low_row["weather_temperature_c_missing"] = 1
    low_row["energy_incomplete_hour"] = 1
    low = assess_forecast_confidence(
        low_row,
        {**forecast.iloc[2].to_dict(), "p10": 48_000.0, "p90": 57_000.0, "route": "baseline_fallback"},
        artifact=artifact,
        source_freshness={"state": "fresh"},
    )

    assert high.level == "high"
    assert low.level == "low"
    assert "shap" not in " ".join(low.reasons).lower()
    assert low.inputs["out_of_distribution_score"] == pytest.approx(0.92)


def test_forecast_change_decomposition_mentions_weather_without_causal_claim(shap_forecast_fixture) -> None:
    _, _, _, points = shap_forecast_fixture
    previous_point = copy.deepcopy(points[0])
    current_point = copy.deepcopy(points[0])
    current_point["expected_demand_gw"] = previous_point["expected_demand_gw"] + 0.70
    for item in current_point["concept_contributions"]:
        if item["concept"] == CONCEPT_WEATHER:
            item["contribution_gw"] += 0.70
            item["direction"] = "raises_forecast"
    current_point["technical"]["feature_values"]["weather_temperature_c"] = (
        previous_point["technical"]["feature_values"]["weather_temperature_c"] - 4.0
    )
    current = {"run_id": "current", "points": [current_point]}
    previous = {"run_id": "previous", "points": [previous_point]}

    changes = explain_forecast_changes(current, previous)
    text = changes["changes"][0]["explanation"]

    assert changes["changes"][0]["delta_gw"] == pytest.approx(0.70)
    assert "temperature feature colder" in text
    assert "not causal proof" in text.lower()


def test_missing_model_fallback_still_reconciles_and_explains(shap_forecast_fixture) -> None:
    _, rows, _, _ = shap_forecast_fixture
    fallback_forecast = forecast_with_artifact(None, rows.iloc[:1])
    points = build_hourly_explanations(
        artifact=None,
        feature_rows=rows.iloc[:1],
        forecast_frame=fallback_forecast,
        run_id="forecast-fallback-test",
        generated_at=GENERATED_AT,
        source_freshness={"state": "fresh"},
    )
    point = points[0]

    assert point["route"] == "baseline_fallback"
    assert point["model_explanation_available"] is False
    assert point["technical"]["raw_shap_values"] == []
    assert point["technical"]["reconciliation"]["within_tolerance"] is True
    assert "fallback model explanation" in point["explanation"]


def test_stable_text_generation(shap_forecast_fixture) -> None:
    _, rows, forecast, _ = shap_forecast_fixture
    first = build_hourly_explanations(
        artifact=None,
        feature_rows=rows.iloc[:1],
        forecast_frame=forecast.iloc[:1],
        run_id="stable",
        generated_at=GENERATED_AT,
        source_freshness={"state": "fresh"},
    )[0]["explanation"]
    second = build_hourly_explanations(
        artifact=None,
        feature_rows=rows.iloc[:1],
        forecast_frame=forecast.iloc[:1],
        run_id="stable",
        generated_at=GENERATED_AT,
        source_freshness={"state": "fresh"},
    )[0]["explanation"]

    assert first == second
    assert "caused" not in first.lower()


def test_forecast_explanation_wsgi_endpoints() -> None:
    forecast_service = ForecastExplanationApiService(
        hourly_loader=_api_hourly_features,
        artifact=None,
        now=lambda: NOW,
    )
    app = create_app(forecast_service=forecast_service)

    status, run = request_json(app, "/v1/forecast", "scope=france&hours=3")
    assert status.startswith("200")
    assert run["horizon_hours"] == 3
    assert run["points"][0]["technical"]["reconciliation"]["within_tolerance"] is True

    run_id = run["run_id"]
    status, retrieved = request_json(app, f"/v1/forecast/{run_id}")
    assert status.startswith("200")
    assert retrieved["run_id"] == run_id

    timestamp = run["points"][0]["timestamp"]
    status, explanation = request_json(app, "/v1/explanations", f"run_id={run_id}&timestamp={timestamp}")
    assert status.startswith("200")
    assert explanation["casual"]["expected_demand_gw"] == run["points"][0]["expected_demand_gw"]
    assert explanation["technical"]["raw_shap_values"] == []

    previous = copy.deepcopy(run)
    previous["run_id"] = "previous-run"
    previous["points"][0]["expected_demand_gw"] -= 0.25
    for item in previous["points"][0]["concept_contributions"]:
        if item["concept"] == CONCEPT_DATA_QUALITY:
            item["contribution_gw"] -= 0.25
    forecast_service.store_run(previous)
    status, changes = request_json(app, "/v1/forecast-changes", f"current={run_id}&previous=previous-run")
    assert status.startswith("200")
    assert changes["changes"]
    assert "not causal proof" in changes["caveat"].lower()

    status, card = request_json(app, "/v1/model-card")
    assert status.startswith("200")
    assert card["status"] == "baseline_fallback"
    assert "explainability" in card


def _training_matrix(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for index in range(96):
        hour = index % 24
        temperature = -2.0 + (index % 16) * 1.2
        rows.append(
            {
                "usual_demand_mw": 49_000 + index * 12,
                "weather_temperature_c": temperature,
                "heating_degree_c": max(0.0, 18.0 - temperature),
                "target_hour_of_day": hour,
                "target_is_weekend": int((index // 24) % 7 >= 5),
                "demand_lag_1h_mw": 50_000 + index * 20,
                "demand_roll_24h_mean_mw": 50_500 + index * 10,
                "national_net_imports_mw": -400 + (index % 12) * 90,
                "source_fallback_record_count": int(index % 19 == 0),
                "energy_incomplete_hour": int(index % 23 == 0),
                "horizon_hours": 1 + (index % 48),
            }
        )
    return pd.DataFrame(rows)[feature_columns]


def _inference_row(*, hour: int, temperature: float, heating: float, usual: float) -> dict[str, Any]:
    origin = pd.Timestamp("2026-02-12T12:00:00Z")
    return {
        "origin_timestamp": origin,
        "target_timestamp": origin + pd.Timedelta(hours=hour),
        "target_timestamp_local": (origin + pd.Timedelta(hours=hour)).tz_convert("Europe/Paris").isoformat(),
        "horizon_hours": hour,
        "usual_demand_mw": usual,
        "usual_demand_p10_mw": usual - 900.0,
        "usual_demand_p90_mw": usual + 900.0,
        "usual_demand_method": "fixture usual demand",
        "usual_demand_sample_count": 24,
        "usual_demand_fallback_level": 1,
        "weather_temperature_c": temperature,
        "heating_degree_c": heating,
        "target_hour_of_day": (12 + hour) % 24,
        "target_is_weekend": 0,
        "demand_lag_1h_mw": usual - 450.0,
        "demand_roll_24h_mean_mw": usual - 700.0,
        "national_net_imports_mw": 600.0,
        "source_fallback_record_count": 0,
        "energy_incomplete_hour": 0,
        "weather_temperature_c_missing": 0,
    }


def _api_hourly_features() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=40 * 24, freq="h")
    rows = []
    for index, timestamp in enumerate(timestamps):
        local = timestamp.tz_convert("Europe/Paris")
        demand = 50_000 + 1_400 * np.sin(2 * np.pi * local.hour / 24) + index * 2
        rows.append(
            {
                "event_time": timestamp,
                "published_at": timestamp,
                "ingested_at": timestamp,
                "source_name": "odre_eco2mix_national",
                "source_revision": "test",
                "quality_status": "ok",
                "fallback_status": "none",
                "source_record_id": "France",
                "region": "France",
                "consumption_mw": demand,
                "total_production_mw": demand + 2_000,
                "net_imports_mw": 300,
            }
        )
    return build_hourly_analytical_dataset(pd.DataFrame(rows))


def request_json(app: Any, path: str, query: str = "") -> tuple[str, dict[str, Any]]:
    environ: dict[str, Any] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "GET"
    environ["PATH_INFO"] = path
    environ["QUERY_STRING"] = query
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    return str(captured["status"]), json.loads(body.decode("utf-8"))
