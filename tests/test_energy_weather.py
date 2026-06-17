from __future__ import annotations

import pandas as pd

from app.components.energy_weather import (
    build_energy_weather_timeline,
    energy_weather_heatmap,
    summarize_energy_weather,
)
from src.models.mood_calibration import calibrate_mood


def sample_energy() -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-16T12:00:00Z", periods=32, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "region": "France",
            "consumption_mw": [45_000 + (index % 8) * 1_000 for index in range(len(timestamps))],
            "co2_intensity_g_per_kwh": [20 if index % 6 == 0 else 55 for index in range(len(timestamps))],
            "renewable_share": [0.35] * len(timestamps),
            "fossil_share": [0.05] * len(timestamps),
        }
    )


def test_energy_weather_builds_24_hour_fallback_timeline():
    energy = sample_energy()
    artifact = calibrate_mood(energy, min_sample=1, generated_at="2026-06-17T00:00:00Z")

    result = build_energy_weather_timeline(
        energy,
        latest_ts=energy["timestamp"].max(),
        mood_artifact=artifact,
        timezone="Europe/Paris",
    )

    assert len(result.timeline) == 24
    assert result.timeline["target"].is_monotonic_increasing
    assert result.timeline["demand_signal_mw"].notna().all()
    assert set(result.timeline["status"]).issubset(
        {"Comfortable", "Watch", "Tense", "Low-carbon opportunity", "Unknown"}
    )
    assert result.metadata["demand_source_counts"]["Recent same-hour pattern"] == 24


def test_energy_weather_marks_stale_model_without_using_it_as_live_precision():
    energy = sample_energy()
    stale_payload = {
        "predictions": [
            {
                "origin_timestamp": "2024-12-31T00:00:00Z",
                "target_timestamp": "2024-12-31T01:00:00Z",
                "horizon_hours": 1,
                "model_predicted_mw": 99_999,
            }
        ]
    }

    result = build_energy_weather_timeline(
        energy,
        latest_ts=energy["timestamp"].max(),
        model_payload=stale_payload,
        timezone="Europe/Paris",
    )

    assert result.metadata["model"]["status"] == "stale"
    assert result.timeline["model_predicted_mw"].isna().all()
    assert "historical" in summarize_energy_weather(result.timeline, result.metadata)["confidence"]


def test_energy_weather_heatmap_renders_plotly_figure():
    result = build_energy_weather_timeline(sample_energy(), timezone="Europe/Paris")
    figure = energy_weather_heatmap(result.timeline)

    assert figure.data
    assert figure.data[0].type == "heatmap"


def test_energy_weather_overlays_official_ecowatt_signal():
    energy = sample_energy()
    latest = energy["timestamp"].max()
    targets = pd.date_range(latest.floor("h") + pd.Timedelta(hours=1), periods=24, freq="h")
    ecowatt = pd.DataFrame(
        {
            "timestamp": targets,
            "ecowatt_status": ["green"] * 23 + ["red"],
            "ecowatt_label": ["Normal"] * 23 + ["Very tense"],
            "ecowatt_severity": [1] * 23 + [3],
            "ecowatt_message": ["Official signal"] * 24,
            "ecowatt_source": ["EcoWatt test"] * 24,
            "ecowatt_source_url": ["https://example.test"] * 24,
        }
    )

    result = build_energy_weather_timeline(
        energy,
        latest_ts=latest,
        ecowatt=ecowatt,
        timezone="Europe/Paris",
    )
    figure = energy_weather_heatmap(result.timeline)

    assert result.metadata["ecowatt"]["status"] == "available"
    assert result.timeline["ecowatt_status"].iloc[-1] == "red"
    assert figure.data[0].y == ("App outlook", "EcoWatt")
