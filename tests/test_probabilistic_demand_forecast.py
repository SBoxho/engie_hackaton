from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.models.probabilistic_demand import (
    DEFAULT_HORIZONS_HOURS,
    MODEL_SCHEMA_VERSION,
    DemandForecastService,
    ResidualQuantileConfig,
    build_inference_feature_rows,
    forecast_with_artifact,
    save_training_artifacts,
    train_residual_quantile_candidate,
    validate_model_artifact,
    validate_no_target_leakage,
)
from src.models.usual_demand import build_model_ready_dataset


def synthetic_public_records(days: int = 10) -> pd.DataFrame:
    start = "2026-01-01T00:00:00Z"
    energy_times = pd.date_range(start, periods=days * 24 * 4, freq="15min", tz="UTC")
    energy_rows = []
    for step, timestamp in enumerate(energy_times):
        local = timestamp.tz_convert("Europe/Paris")
        daily = 1_500 * np.sin(2 * np.pi * local.hour / 24)
        weekly = 650 if local.dayofweek < 5 else -850
        trend = step * 3.5
        demand = 48_000 + daily + weekly + trend
        energy_rows.append(
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
                "total_production_mw": demand + 1_200,
                "net_imports_mw": 0,
                "rte_forecast_j_mw": demand + 50,
                "rte_forecast_j1_mw": demand + 80,
            }
        )
    weather_times = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    weather = pd.DataFrame(
        {
            "event_time": weather_times,
            "published_at": weather_times,
            "ingested_at": weather_times,
            "source_name": "open_meteo_weather",
            "source_revision": "test",
            "quality_status": "ok",
            "fallback_status": "none",
            "source_record_id": "national",
            "temperature_c": 8 + np.sin(np.arange(len(weather_times)) / 24 * 2 * np.pi),
            "humidity_pct": 70,
            "wind_speed_kmh": 20,
            "cloud_cover_pct": 50,
            "solar_radiation_wm2": 0,
        }
    )
    return pd.concat([pd.DataFrame(energy_rows), weather], ignore_index=True, sort=False)


@pytest.fixture(scope="module")
def trained_candidate():
    horizons = (1, 2, 24, 48)
    dataset = build_model_ready_dataset(synthetic_public_records(), horizons_hours=horizons)
    config = ResidualQuantileConfig(
        horizons_hours=horizons,
        min_train_samples=120,
        min_validation_samples=32,
        validation_folds=2,
        baseline_min_samples=1,
        max_iter=5,
        learning_rate=0.08,
    )
    artifact, predictions = train_residual_quantile_candidate(
        dataset.supervised,
        dataset.hourly,
        feature_manifest=dataset.feature_manifest,
        config=config,
    )
    return dataset, artifact, predictions


def test_no_target_leakage_and_metric_driven_champion_decision(trained_candidate):
    _, artifact, predictions = trained_candidate

    validate_no_target_leakage(predictions, artifact["feature_columns"])
    assert "target_mw" not in artifact["feature_columns"]
    assert "target_observation_available_at" not in artifact["feature_columns"]
    assert "rte_forecast_j_mw" not in artifact["feature_columns"]
    assert "shap_values" not in artifact["model_card"]

    metrics = artifact["metrics"]
    assert metrics["overall"]["mae_gw"] is not None
    assert metrics["overall"]["wape"] is not None
    assert metrics["overall"]["pinball_loss_p10_gw"] is not None
    assert metrics["overall"]["p10_p90_empirical_coverage"] is not None
    assert metrics["by_horizon"]
    assert metrics["by_season"]
    assert metrics["peaks"]["sample_count"] > 0
    assert metrics["baselines"]["usual_demand"]["sample_count"] > 0
    assert metrics["baselines"]["seasonal_naive"]["origin_count"] > 0
    assert metrics["baselines"]["rte_public_forecast"]["available"] is True

    decision = metrics["champion_decision"]
    assert artifact["status"] in {"champion", "rejected"}
    assert artifact["status"] == ("champion" if decision["accepted"] else "rejected")
    if artifact["status"] == "rejected":
        assert artifact["rejection_reason"]


def test_deterministic_inference_quantile_order_length_and_timezone(trained_candidate):
    dataset, artifact, _ = trained_candidate
    origin = dataset.hourly["timestamp"].max()
    service = DemandForecastService(artifact=artifact)

    first = service.forecast(origin, dataset.hourly)
    second = service.forecast(origin, dataset.hourly)
    first_frame = first.to_frame()
    second_frame = second.to_frame()

    assert len(first_frame) == 48
    pd.testing.assert_series_equal(first_frame["p50"], second_frame["p50"])
    assert first_frame["target_timestamp"].map(lambda value: pd.Timestamp(value).tzinfo is not None).all()
    assert first_frame["target_timestamp_local"].astype(str).str.contains(r"\+0[12]:00").all()
    assert (first_frame["p10"] <= first_frame["p50"]).all()
    assert (first_frame["p50"] <= first_frame["p90"]).all()
    assert first_frame["horizon_hours"].tolist() == list(DEFAULT_HORIZONS_HOURS)


def test_missing_optional_features_are_imputed(trained_candidate):
    dataset, artifact, _ = trained_candidate
    champion = copy.copy(artifact)
    champion["status"] = "champion"
    champion["rejection_reason"] = None
    rows = build_inference_feature_rows(dataset.hourly, dataset.hourly["timestamp"].max(), horizons_hours=(1, 2))
    optional_feature = next(column for column in champion["feature_columns"] if column not in {"usual_demand_mw", "horizon_hours"})
    rows = rows.drop(columns=[optional_feature], errors="ignore")

    forecast = forecast_with_artifact(champion, rows)

    assert len(forecast) == 2
    assert forecast[["p10", "p50", "p90"]].notna().all().all()
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()


def test_baseline_fallback_without_trained_model(trained_candidate):
    dataset, _, _ = trained_candidate
    service = DemandForecastService()
    run = service.forecast(dataset.hourly["timestamp"].max(), dataset.hourly)
    frame = run.to_frame()

    assert run.route == "baseline_fallback"
    assert run.fallback_reason == "no champion model artifact is available"
    assert len(frame) == 48
    assert frame["route"].eq("baseline_fallback").all()


def test_artifact_compatibility_validation(trained_candidate):
    _, artifact, _ = trained_candidate
    validate_model_artifact(artifact)

    bad_schema = dict(artifact)
    bad_schema["schema_version"] = MODEL_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema"):
        validate_model_artifact(bad_schema)

    missing_quantile = copy.copy(artifact)
    missing_quantile["status"] = "champion"
    missing_quantile["models"] = dict(artifact["models"])
    missing_quantile["models"].pop("p90", None)
    with pytest.raises(ValueError, match="p90"):
        validate_model_artifact(missing_quantile)


def test_registry_manifest_contains_status_reason_and_checksums(tmp_path, trained_candidate):
    _, artifact, predictions = trained_candidate

    manifest = save_training_artifacts(artifact, predictions, tmp_path)

    assert manifest["model_version"] == artifact["model_version"]
    assert manifest["status"] == artifact["status"]
    assert "rejection_reason" in manifest
    assert set(manifest["artifact_checksums"]) == {"model", "validation_predictions", "model_card"}
    assert all(len(value) == 64 for value in manifest["artifact_checksums"].values())
    assert (tmp_path / "artifact_manifest.json").exists()
    assert (tmp_path / "model_card.json").exists()
