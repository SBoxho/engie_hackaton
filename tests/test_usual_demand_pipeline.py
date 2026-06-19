from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.usual_demand import (
    BaselineConfig,
    build_hourly_analytical_dataset,
    build_model_ready_dataset,
    compute_usual_demand_baselines,
    compute_usual_demand_state,
    evaluate_usual_demand_baseline,
)


def energy_records(
    start: str = "2026-01-01T00:00:00Z",
    *,
    hours: int = 24,
    source_name: str = "odre_eco2mix_national",
    regions: tuple[str, ...] = ("France",),
    published_at: str | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=hours * 4, freq="15min", tz="UTC")
    rows = []
    for region_index, region in enumerate(regions):
        for step, timestamp in enumerate(timestamps):
            local = timestamp.tz_convert("Europe/Paris")
            base = 50_000 + 1_000 * region_index
            demand = base + 80 * local.hour + 12 * local.dayofweek + step * 0.5
            rows.append(
                {
                    "event_time": timestamp,
                    "published_at": pd.Timestamp(published_at) if published_at else pd.NaT,
                    "ingested_at": timestamp,
                    "source_name": source_name,
                    "source_revision": "test",
                    "quality_status": "ok",
                    "fallback_status": "none",
                    "source_record_id": region,
                    "region": region,
                    "consumption_mw": demand,
                    "nuclear_mw": 30_000 + region_index * 100,
                    "wind_mw": 4_000 + step,
                    "solar_mw": max(0, 100 * np.sin(np.pi * local.hour / 24)),
                    "hydro_mw": 6_000,
                    "gas_mw": 2_000,
                    "coal_mw": 0,
                    "oil_mw": 50,
                    "bioenergy_mw": 800,
                    "total_production_mw": 42_850 + region_index * 100 + step,
                    "imports_mw": 500,
                    "exports_mw": 0,
                    "net_imports_mw": 500,
                    "physical_balance_mw": 0,
                    "co2_intensity_g_per_kwh": 35 + region_index,
                }
            )
    return pd.DataFrame(rows)


def weather_records(start: str = "2026-01-01T00:00:00Z", *, hours: int = 24) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "event_time": timestamps,
            "published_at": timestamps,
            "ingested_at": timestamps,
            "source_name": "open_meteo_weather",
            "source_revision": "test",
            "quality_status": "ok",
            "fallback_status": "none",
            "source_record_id": "paris",
            "temperature_c": 6 + np.sin(np.arange(hours) / 24 * 2 * np.pi),
            "humidity_pct": 80,
            "wind_speed_kmh": 12,
            "cloud_cover_pct": 70,
            "solar_radiation_wm2": 0,
        }
    )


def public_holiday_row(day: str) -> pd.DataFrame:
    event_time = pd.Timestamp(day).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.DataFrame(
        [
            {
                "event_time": event_time,
                "published_at": pd.NaT,
                "ingested_at": event_time,
                "source_name": "french_public_holidays",
                "source_revision": "test",
                "quality_status": "ok",
                "fallback_status": "none",
                "source_record_id": f"metropole:{day}",
                "is_public_holiday": 1,
                "holiday_name": "Test holiday",
                "territory": "metropole",
            }
        ]
    )


def school_holiday_row(start: str, end: str, zone: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp(start).tz_localize("UTC"),
                "published_at": pd.NaT,
                "ingested_at": pd.Timestamp(start).tz_localize("UTC"),
                "source_name": "french_school_holidays",
                "source_revision": "test",
                "quality_status": "ok",
                "fallback_status": "none",
                "source_record_id": f"{zone}:{start}:{end}",
                "start_date": start,
                "end_date": end,
                "zone": zone,
                "description": "Test school break",
                "is_school_holiday": 1,
            }
        ]
    )


def test_power_is_averaged_and_energy_is_summed_explicitly():
    frame = energy_records(hours=1)
    frame["consumption_mw"] = [100.0, 200.0, 300.0, 400.0]
    frame["metered_energy_mwh"] = [1.0, 2.0, 3.0, 4.0]

    hourly = build_hourly_analytical_dataset(frame)
    row = hourly.iloc[0]

    assert row["timestamp"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert row["consumption_mw"] == pytest.approx(250.0)
    assert row["metered_energy_mwh"] == pytest.approx(10.0)
    assert row["energy_expected_observation_count"] == 4
    assert row["energy_coverage_ratio"] == pytest.approx(1.0)


def test_no_future_observations_enter_supervised_features():
    delayed = energy_records(hours=1, published_at="2026-01-01T03:00:00Z")
    normal = energy_records(start="2026-01-01T01:00:00Z", hours=4)
    records = pd.concat([delayed, normal, weather_records(hours=6)], ignore_index=True, sort=False)

    result = build_model_ready_dataset(records, horizons_hours=(1,))

    assert result.supervised["feature_available_at"].le(result.supervised["origin_timestamp"]).all()
    assert pd.Timestamp("2026-01-01T01:00:00Z") not in set(result.supervised["origin_timestamp"])


def test_french_public_and_school_holiday_features_are_deterministic():
    energy = energy_records(start="2025-12-31T23:00:00Z", hours=3)
    records = pd.concat(
        [
            energy,
            public_holiday_row("2026-01-01"),
            school_holiday_row("2025-12-20", "2026-01-05", "A"),
        ],
        ignore_index=True,
        sort=False,
    )

    hourly = build_hourly_analytical_dataset(records)
    holiday = hourly.loc[hourly["timestamp"].eq(pd.Timestamp("2026-01-01T00:00:00Z"))].iloc[0]

    assert holiday["is_public_holiday"] == 1
    assert holiday["school_holiday_zone_a"] == 1
    assert holiday["holiday_type"] == "public_and_school_holiday"
    assert holiday["season"] == "winter"


def test_dst_fall_back_keeps_utc_rows_and_repeated_local_hour_offsets():
    hourly = build_hourly_analytical_dataset(
        energy_records(start="2024-10-26T22:00:00Z", hours=6)
    )

    repeated_two = hourly.loc[hourly["hour_of_day"].eq(2)]

    assert hourly["timestamp"].is_unique
    assert len(repeated_two) == 2
    assert set(repeated_two["utc_offset_hours"]) == {1.0, 2.0}


def test_missing_weather_data_is_explicit_and_does_not_drop_energy_rows():
    result = build_model_ready_dataset(energy_records(hours=8), horizons_hours=(1,))

    assert not result.hourly.empty
    assert result.hourly["weather_missing"].eq(1).all()
    assert result.supervised["weather_temperature_c_missing"].eq(1).all()


def test_regional_hourly_rows_and_derived_national_aggregation_are_physical():
    regional = energy_records(
        hours=1,
        source_name="odre_eco2mix_regional",
        regions=("Bretagne", "Ile-de-France"),
    )
    regional.loc[regional["region"].eq("Bretagne"), "consumption_mw"] = [100, 200, 300, 400]
    regional.loc[regional["region"].eq("Ile-de-France"), "consumption_mw"] = [1000, 1100, 1200, 1300]

    hourly = build_hourly_analytical_dataset(regional)

    brittany = hourly.loc[hourly["region"].eq("Bretagne")].iloc[0]
    ile_de_france = hourly.loc[hourly["region"].eq("Ile-de-France")].iloc[0]
    national = hourly.loc[hourly["geographic_scope"].eq("national")].iloc[0]
    assert brittany["consumption_mw"] == pytest.approx(250.0)
    assert ile_de_france["consumption_mw"] == pytest.approx(1150.0)
    assert national["is_derived_national_from_regions"] == 1
    assert national["consumption_mw"] == pytest.approx(1400.0)


def test_fallback_hierarchy_and_metrics_expose_method_sample_and_region():
    energy = energy_records(start="2026-01-01T00:00:00Z", hours=9 * 24)
    records = pd.concat([energy, public_holiday_row("2026-01-08")], ignore_index=True, sort=False)
    result = build_model_ready_dataset(records, horizons_hours=(24,))
    predictions = compute_usual_demand_baselines(
        result.supervised,
        result.hourly,
        config=BaselineConfig(min_samples=2),
    )

    normal = predictions.loc[predictions["target_holiday_type"].eq("normal")]
    holiday = predictions.loc[predictions["target_holiday_type"].eq("public_holiday")]
    assert normal["usual_demand_fallback_level"].min() == 1
    assert holiday["usual_demand_fallback_level"].min() == 2
    assert predictions["usual_demand_sample_count"].ge(1).all()
    assert predictions["usual_demand_source_timestamp_max"].le(predictions["origin_timestamp"]).all()

    metrics = evaluate_usual_demand_baseline(predictions)
    assert metrics["overall"]["mae_gw"] is not None
    assert metrics["overall"]["wape"] is not None
    assert metrics["by_horizon"]
    assert metrics["by_season"]
    assert metrics["by_weekday_type"]
    assert metrics["by_region"]
    assert metrics["by_fallback_level"]


def test_usual_demand_state_provides_percent_above_usual_for_national_and_region():
    regional = energy_records(
        hours=8 * 24,
        source_name="odre_eco2mix_regional",
        regions=("Bretagne", "Ile-de-France"),
    )
    hourly = build_hourly_analytical_dataset(regional)

    state = compute_usual_demand_state(hourly, config=BaselineConfig(min_samples=1))

    assert {"national", "regional"} <= set(state["geographic_scope"])
    assert state["above_usual_percent"].notna().any()
    assert state["usual_demand_sample_count"].ge(1).all()


def test_feature_generation_is_deterministic():
    records = pd.concat(
        [energy_records(hours=4 * 24), weather_records(hours=4 * 24)],
        ignore_index=True,
        sort=False,
    )

    first = build_model_ready_dataset(records, horizons_hours=(1, 24))
    second = build_model_ready_dataset(records, horizons_hours=(1, 24))

    pd.testing.assert_frame_equal(first.hourly, second.hourly)
    pd.testing.assert_frame_equal(first.supervised, second.supervised)
    assert first.feature_manifest == second.feature_manifest


def test_cli_build_and_backtest_smoke(tmp_path):
    records = pd.concat(
        [energy_records(hours=8 * 24), weather_records(hours=8 * 24)],
        ignore_index=True,
        sort=False,
    )
    input_path = tmp_path / "public_records.parquet"
    records.to_parquet(input_path, index=False)
    hourly = tmp_path / "hourly.parquet"
    training = tmp_path / "training.parquet"
    manifest = tmp_path / "manifest.json"
    quality = tmp_path / "quality.json"
    coverage = tmp_path / "coverage.json"
    predictions = tmp_path / "predictions.parquet"
    backtest = tmp_path / "backtest.json"
    root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_usual_demand_dataset",
            "--input",
            str(input_path),
            "--horizon",
            "24",
            "--hourly-output",
            str(hourly),
            "--training-output",
            str(training),
            "--manifest-output",
            str(manifest),
            "--quality-output",
            str(quality),
            "--coverage-output",
            str(coverage),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.backtest_usual_demand",
            "--hourly",
            str(hourly),
            "--training",
            str(training),
            "--min-samples",
            "1",
            "--lookback-days",
            "30",
            "--predictions-output",
            str(predictions),
            "--output",
            str(backtest),
        ],
        cwd=root,
        check=True,
    )

    payload = json.loads(backtest.read_text(encoding="utf-8"))
    assert payload["metrics"]["overall"]["mae_gw"] is not None
    assert payload["fallback_hierarchy"][0]["level"] == 1
    assert manifest.exists()
    assert quality.exists()
    assert coverage.exists()
