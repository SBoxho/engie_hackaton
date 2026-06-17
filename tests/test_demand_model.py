from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.demand_model import (
    FeatureConfig,
    TrainConfig,
    build_feature_frame,
    chronological_split,
    evaluate_models,
    explain_forecast_rows,
    feature_family_columns,
    inspect_demand_dataset,
    load_model_bundle,
    regression_metrics,
    save_feature_metadata,
    train_models,
    validate_feature_metadata,
    validate_model_bundle,
)


def synthetic_energy(days: int = 14, *, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    step = np.arange(len(timestamps), dtype=float)
    local = timestamps.tz_convert("Europe/Paris")
    daily = 1800 * np.sin(2 * np.pi * local.hour / 24)
    weekly = np.where(local.dayofweek >= 5, -900, 400)
    trend = step * 0.8
    temperature_effect = -120 * np.cos(2 * np.pi * step / 96)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "region": "France",
            "consumption_mw": 52_000 + daily + weekly + trend + temperature_effect,
        }
    )


def synthetic_weather(energy: pd.DataFrame, *, missing_every: int | None = None) -> pd.DataFrame:
    timestamps = pd.to_datetime(energy["timestamp"], utc=True)
    step = np.arange(len(timestamps), dtype=float)
    weather = pd.DataFrame(
        {
            "timestamp": timestamps,
            "weather_city_count": 10,
            "weather_expected_city_count": 10,
            "weather_population_coverage": 1.0,
            "weather_missing_cities": "",
            "weather_temperature_c": 12 + 7 * np.sin(2 * np.pi * step / 96),
            "weather_wind_speed_kmh": 20 + np.cos(2 * np.pi * step / 24),
            "weather_cloud_cover_pct": 50,
            "weather_solar_radiation_wm2": np.maximum(0, 300 * np.sin(2 * np.pi * (step % 96) / 96)),
            "weather_humidity_pct": 70,
            "weather_source_timestamp_max": timestamps.copy(),
        }
    )
    if missing_every:
        weather.loc[weather.index % missing_every == 0, "weather_temperature_c"] = np.nan
        weather.loc[weather.index % missing_every == 0, "weather_population_coverage"] = 0.8
    return weather


def build_small_features(days: int = 14, horizons=(1, 3, 6, 24)):
    energy = synthetic_energy(days)
    weather = synthetic_weather(energy, missing_every=37)
    return build_feature_frame(
        energy,
        weather=weather,
        config=FeatureConfig(horizons_hours=tuple(horizons), min_continuous_hours=48),
        source="synthetic",
    )


def test_dataset_audit_reports_coverage_gaps_duplicates_and_weather():
    energy = synthetic_energy(2)
    duplicate = pd.concat([energy, energy.iloc[[0]]], ignore_index=True)
    missing = duplicate.drop(index=[10])
    audit = inspect_demand_dataset(missing, weather=synthetic_weather(energy))

    assert audit["duplicate_timestamp_count"] == 1
    assert audit["missing_interval_count"] == 1
    assert audit["start_utc"] == "2024-01-01T00:00:00Z"
    assert audit["weather"]["overlap_row_count"] > 0
    assert "region" in audit["extra_columns"]


def test_feature_timestamp_alignment_and_leakage_controls():
    features, metadata = build_small_features(days=4, horizons=(1, 3))
    assert set(features["horizon_hours"]) == {1, 3}
    assert (
        pd.to_datetime(features["target_timestamp"], utc=True)
        - pd.to_datetime(features["origin_timestamp"], utc=True)
    ).eq(pd.to_timedelta(features["horizon_hours"], unit="h")).all()
    assert features["weather_source_age_minutes"].dropna().ge(0).all()
    assert "rolling demand statistics are shifted" in " ".join(metadata["leakage_controls"])

    first = features.loc[features["horizon_hours"].eq(1)].iloc[0]
    assert pd.isna(first["demand_roll_1h_mean_mw"])
    later = features.loc[features["horizon_hours"].eq(1)].iloc[4]
    expected = synthetic_energy(4)["consumption_mw"].iloc[:4].mean()
    assert later["demand_roll_1h_mean_mw"] == pytest.approx(expected)


def test_rejects_future_weather_provenance():
    energy = synthetic_energy(3)
    weather = synthetic_weather(energy)
    weather.loc[0, "weather_source_timestamp_max"] = pd.Timestamp("2024-01-01T00:15:00Z")
    with pytest.raises(ValueError, match="later than the forecast origin"):
        build_feature_frame(
            energy,
            weather=weather,
            config=FeatureConfig(horizons_hours=(1,), min_continuous_hours=24),
        )


def test_dst_calendar_features_are_utc_stable_across_fallback():
    energy = synthetic_energy(4, start="2024-10-26T00:00:00Z")
    features, _ = build_feature_frame(
        energy,
        weather=synthetic_weather(energy),
        config=FeatureConfig(horizons_hours=(1,), min_continuous_hours=24),
    )
    dst_slice = features[
        features["origin_timestamp"].between(
            pd.Timestamp("2024-10-27T00:00:00Z"), pd.Timestamp("2024-10-27T03:00:00Z")
        )
    ]
    assert dst_slice["origin_timestamp"].is_unique
    assert set(dst_slice["origin_utc_offset_hours"]) == {1.0, 2.0}
    repeated_local_two = dst_slice.loc[dst_slice["origin_hour"].eq(2)]
    assert set(repeated_local_two["origin_utc_offset_hours"]) == {1.0, 2.0}


def test_feature_builder_infers_consolidated_30_minute_cadence():
    energy = synthetic_energy(10).iloc[::2].reset_index(drop=True)
    weather = synthetic_weather(energy)
    features, metadata = build_feature_frame(
        energy,
        weather=weather,
        config=FeatureConfig(horizons_hours=(1, 24), min_continuous_hours=168),
    )

    assert metadata["audit"]["cadence_minutes"] == 30
    assert metadata["feature_config"]["resolved_cadence_minutes"] == 30
    assert metadata["audit"]["missing_interval_count"] == 0
    usable = features[
        features["target_mw"].notna()
        & features["same_continuous_block"]
        & features["eligible_continuous_period"]
    ]
    assert not usable.empty
    assert (usable["target_timestamp"] - usable["origin_timestamp"]).isin(
        [pd.Timedelta(hours=1), pd.Timedelta(hours=24)]
    ).all()


def test_missing_target_breaks_continuity_without_interpolation_and_weather_is_explicit():
    energy = synthetic_energy(4)
    energy.loc[100, "consumption_mw"] = np.nan
    weather = synthetic_weather(energy, missing_every=5)
    with pytest.raises(ValueError, match="No sufficiently continuous"):
        build_feature_frame(
            energy,
            weather=weather,
            config=FeatureConfig(horizons_hours=(1,), min_continuous_hours=72),
        )
    features, _ = build_feature_frame(
        energy,
        weather=weather,
        config=FeatureConfig(horizons_hours=(1,), min_continuous_hours=24),
    )
    assert features["weather_temperature_c_missing"].sum() > 0
    assert not features["same_continuous_block"].all()


def test_chronological_split_integrity_and_exact_horizon_alignment():
    features, metadata = build_small_features(days=8, horizons=(1,))
    rows = features[
        features["target_mw"].notna()
        & features["same_continuous_block"]
        & features["eligible_continuous_period"]
    ].sort_values("target_timestamp")
    train, test = chronological_split(rows, config=TrainConfig(min_train_samples=96, min_test_samples=24))
    assert train["target_timestamp"].max() < test["target_timestamp"].min()
    assert (test["target_timestamp"] - test["origin_timestamp"]).eq(pd.Timedelta(hours=1)).all()
    validate_feature_metadata(features, metadata)


def test_deterministic_training_and_evaluation_baseline_comparison():
    features, metadata = build_small_features(days=10, horizons=(1, 3))
    config = TrainConfig(
        random_seed=7,
        validation_folds=2,
        min_train_samples=96,
        min_test_samples=24,
        min_validation_samples=24,
    )
    first = train_models(features, metadata, config=config)
    second = train_models(features, metadata, config=config)
    first_eval = evaluate_models(features, first, min_segment_samples=8)
    second_eval = evaluate_models(features, second, min_segment_samples=8)

    first_predictions = pd.DataFrame(first_eval["predictions"])
    second_predictions = pd.DataFrame(second_eval["predictions"])
    np.testing.assert_allclose(
        first_predictions["model_predicted_mw"],
        second_predictions["model_predicted_mw"],
    )
    assert {row["horizon_hours"] for row in first_eval["baseline_comparison"]} == {1, 3}
    for row in first_eval["baseline_comparison"]:
        assert row["strongest_baseline"] in {"persistence", "day_naive", "week_naive"}
        assert row["baseline_eligible_count"] >= 2
        assert row["reliability_badge"] in {"Model edge detected", "Experimental horizon"}
        assert row["model_beats_strongest_baseline"] == (
            row["improvement_vs_strongest_baseline_percent"] > 0
        )
        assert row["prediction_interval_method"] == "hist_gradient_boosting_quantile"
    assert first_eval["segment_metrics"]
    assert first_eval["explanation_disclaimer"]
    assert first_eval["interval_definition"]["coverage_label"] == "80% central prediction interval"
    first_explanation = first_predictions.iloc[0]
    assert first_explanation["explanation_status"] == "ok"
    assert first_explanation["model_interval_lower_mw"] <= first_explanation["model_predicted_mw"]
    assert first_explanation["model_interval_upper_mw"] >= first_explanation["model_predicted_mw"]
    assert 2 <= len(first_explanation["explanation_cards"]) <= 4
    assert first_explanation["technical_contributions"]
    assert all("weather_" not in card["title"] for card in first_explanation["explanation_cards"])


def test_metric_calculation_and_artifact_schema_validation(tmp_path):
    metrics = regression_metrics([100, 200], [110, 180])
    assert metrics["mae_mw"] == pytest.approx(15)
    assert metrics["rmse_mw"] == pytest.approx(np.sqrt((100 + 400) / 2))
    features, metadata = build_small_features(days=8, horizons=(1,))
    bundle = train_models(
        features,
        metadata,
        config=TrainConfig(min_train_samples=96, min_test_samples=24, validation_folds=1),
    )
    validate_model_bundle(bundle)

    bad_metadata = {**metadata, "schema_version": 999}
    with pytest.raises(ValueError, match="Unsupported feature schema"):
        validate_feature_metadata(features, bad_metadata)

    path = tmp_path / "model.pkl"
    from src.models.demand_model import save_model_bundle

    save_model_bundle(bundle, path)
    loaded = load_model_bundle(path)
    assert loaded["schema_version"] == bundle["schema_version"]


def test_explanation_family_mapping_and_fallback_are_readable():
    columns = [
        "weather_temperature_c",
        "origin_hour",
        "target_is_weekend",
        "demand_lag_1h_mw",
        "demand_roll_1h_mean_mw",
        "demand_lag_168h_mw",
        "weather_population_coverage",
        "weather_temperature_c_missing",
    ]
    families = feature_family_columns(columns)

    assert families["weather"] == ["weather_temperature_c"]
    assert "target_is_weekend" in families["calendar"]
    assert "demand_lag_1h_mw" in families["recent_demand"]
    assert families["weekly_pattern"] == ["demand_lag_168h_mw"]
    assert "weather_temperature_c_missing" in families["data_quality"]

    fallback = explain_forecast_rows(
        pd.DataFrame({"origin_hour": [12]}),
        model=object(),
        feature_columns=["missing_column"],
        reference_rows=pd.DataFrame(),
    )
    assert fallback.iloc[0]["explanation_status"] == "fallback"
    assert fallback.iloc[0]["explanation_cards"][0]["title"] == "Explanation could not be computed"


def test_cli_smoke_build_train_evaluate(tmp_path):
    energy = synthetic_energy(8)
    weather = synthetic_weather(energy)
    energy_path = tmp_path / "energy.parquet"
    weather_path = tmp_path / "weather.parquet"
    features_path = tmp_path / "features.parquet"
    metadata_path = tmp_path / "feature_metadata.json"
    model_path = tmp_path / "model.pkl"
    evaluation_path = tmp_path / "evaluation.json"
    energy.to_parquet(energy_path, index=False)
    weather.to_parquet(weather_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_features",
            "--input",
            str(energy_path),
            "--weather",
            str(weather_path),
            "--output",
            str(features_path),
            "--metadata-output",
            str(metadata_path),
            "--min-continuous-hours",
            "48",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_demand_model",
            "--features",
            str(features_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(model_path),
            "--min-train-samples",
            "96",
            "--min-test-samples",
            "24",
            "--validation-folds",
            "1",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_demand_model",
            "--features",
            str(features_path),
            "--model",
            str(model_path),
            "--output",
            str(evaluation_path),
            "--min-segment-samples",
            "8",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert payload["metrics"]
    assert payload["predictions"]


def test_streamlit_demand_model_page_smoke():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parents[1] / "app" / "pages" / "6_demand_model.py"
    app = AppTest.from_file(str(page)).run(timeout=10)
    assert not app.exception
