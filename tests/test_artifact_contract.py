from __future__ import annotations

import json

import pandas as pd

from src.artifact_contract import ArtifactSpec, validate_artifact, validate_demo_bundle


def test_optional_empty_parquet_is_explicit(tmp_path):
    path = tmp_path / "ecowatt.parquet"
    pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}).to_parquet(path, index=False)

    check = validate_artifact(
        ArtifactSpec("ecowatt", "Demo EcoWatt", path, "parquet", required=False, min_rows=1)
    )

    assert check.status == "empty"
    assert check.health_status == "empty"
    assert not check.blocks_demo
    assert "0 rows" in check.detail


def test_required_missing_file_blocks_demo(tmp_path):
    check = validate_artifact(
        ArtifactSpec("energy", "Demo energy", tmp_path / "missing.parquet", "parquet", required=True, min_rows=1)
    )

    assert check.status == "missing"
    assert check.blocks_demo
    assert check.health_status == "missing"


def test_json_missing_keys_is_invalid_and_blocks_when_required(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"predictions": []}), encoding="utf-8")

    check = validate_artifact(
        ArtifactSpec(
            "demand_evaluation",
            "Demand model evaluation",
            path,
            "json",
            required=True,
            required_keys=("predictions", "metrics"),
        )
    )

    assert check.status == "invalid"
    assert check.blocks_demo
    assert "metrics" in check.detail


def test_demo_bundle_allows_optional_ecowatt_zero_rows(tmp_path):
    pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=1, tz="UTC")}).to_parquet(
        tmp_path / "energy_recent.parquet", index=False
    )
    pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}).to_parquet(
        tmp_path / "ecowatt.parquet", index=False
    )
    pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}).to_parquet(
        tmp_path / "weather_national.parquet", index=False
    )
    payloads = {
        "manifest.json": {"schema_version": 1, "generated_at": "2026-06-17T00:00:00Z", "artifacts": {}},
        "quality_report.json": {"findings": [], "passed": True},
        "demand_model_evaluation.json": {"predictions": [], "metrics": []},
        "model_forecast.json": {"schema_version": 1, "forecasts": []},
        "baseline_backtest.json": {"predictions": [], "metrics": []},
        "mood_calibration.json": {"segments": [], "fixed_thresholds": {}},
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    checks = validate_demo_bundle(tmp_path, log=False)
    by_key = {check.spec.key: check for check in checks}

    assert by_key["ecowatt"].status == "empty"
    assert by_key["weather"].status == "empty"
    assert not by_key["ecowatt"].blocks_demo
    assert not [check for check in checks if check.blocks_demo]
