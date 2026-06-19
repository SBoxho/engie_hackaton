from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    values = [item.strip() for item in os.getenv(name, default).split(",")]
    return tuple(item for item in values if item)


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    app_mode: str = os.getenv("APP_MODE", "demo").strip().lower()
    demo_allow_external_api: bool = _env_bool("DEMO_ALLOW_EXTERNAL_API", "0")
    odre_base_url: str = os.getenv(
        "ODRE_BASE_URL", "https://odre.opendatasoft.com/api/explore/v2.1"
    )
    ecowatt_current_dataset_id: str = os.getenv(
        "ECOWATT_CURRENT_DATASET_ID", "nouveau_signal_ecowatt"
    )
    ecowatt_legacy_dataset_id: str = os.getenv(
        "ECOWATT_LEGACY_DATASET_ID", "signal-ecowatt"
    )
    ecowatt_current_odre_url: str = os.getenv(
        "ECOWATT_CURRENT_ODRE_URL",
        "https://odre.opendatasoft.com/explore/dataset/nouveau_signal_ecowatt/",
    )
    ecowatt_legacy_odre_url: str = os.getenv(
        "ECOWATT_LEGACY_ODRE_URL",
        "https://odre.opendatasoft.com/explore/dataset/signal-ecowatt/",
    )
    ecowatt_current_data_gouv_url: str = os.getenv(
        "ECOWATT_CURRENT_DATA_GOUV_URL",
        "https://www.data.gouv.fr/datasets/donnees-du-signal-ecowatt-a-partir-du-01-09-2022",
    )
    rte_ecowatt_api_url: str = os.getenv(
        "RTE_ECOWATT_API_URL",
        "https://digital.iservices.rte-france.com/open_api/ecowatt/v5/signals",
    )
    rte_ecowatt_api_token: str | None = os.getenv("RTE_ECOWATT_API_TOKEN") or None
    open_meteo_base_url: str = os.getenv(
        "OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast"
    )
    entsoe_api_token: str | None = os.getenv("ENTSOE_API_TOKEN") or None
    timezone: str = os.getenv("ENERGY_PULSE_TIMEZONE", "Europe/Paris")
    history_hours: int = _env_int("ENERGY_PULSE_HISTORY_HOURS", "72")
    api_allowed_origins: tuple[str, ...] = _env_csv(
        "ENERGY_PULSE_ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    )
    api_max_body_bytes: int = _env_int("ENERGY_PULSE_MAX_BODY_BYTES", "65536")
    public_http_timeout_seconds: float = _env_float("ENERGY_PULSE_HTTP_TIMEOUT_SECONDS", "15")
    public_http_max_retries: int = _env_int("ENERGY_PULSE_HTTP_MAX_RETRIES", "2")
    public_http_min_interval_seconds: float = _env_float("ENERGY_PULSE_HTTP_MIN_INTERVAL_SECONDS", "0.25")
    circuit_breaker_failure_threshold: int = _env_int("ENERGY_PULSE_CIRCUIT_BREAKER_FAILURES", "3")
    circuit_breaker_recovery_seconds: float = _env_float("ENERGY_PULSE_CIRCUIT_BREAKER_RECOVERY_SECONDS", "60")
    demo_fixed_date_label: str = os.getenv(
        "ENERGY_PULSE_DEMO_DATE_LABEL",
        "demo replay window anchored to 19 Jun 2026 for presentation",
    )
    demo_anchor_end_utc: str = os.getenv(
        "ENERGY_PULSE_DEMO_ANCHOR_END_UTC",
        "2026-06-19T10:30:00Z",
    )

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def is_demo_mode(self) -> bool:
        return self.app_mode == "demo"

    @property
    def is_live_mode(self) -> bool:
        return self.app_mode == "live"

    @property
    def app_mode_label(self) -> str:
        return "Demo data mode" if self.is_demo_mode else "Live data mode"

    @property
    def demo_dir(self) -> Path:
        return self.project_root / "demo_data"

    @property
    def demo_energy_path(self) -> Path:
        return self.demo_dir / "energy_recent.parquet"

    @property
    def demo_weather_path(self) -> Path:
        return self.demo_dir / "weather_national.parquet"

    @property
    def demo_ecowatt_path(self) -> Path:
        return self.demo_dir / "ecowatt.parquet"

    @property
    def demo_quality_path(self) -> Path:
        return self.demo_dir / "quality_report.json"

    @property
    def demo_model_evaluation_path(self) -> Path:
        return self.demo_dir / "demand_model_evaluation.json"

    @property
    def demo_model_forecast_path(self) -> Path:
        return self.demo_dir / "model_forecast.json"

    @property
    def demo_baseline_artifact_path(self) -> Path:
        return self.demo_dir / "baseline_backtest.json"

    @property
    def demo_mood_artifact_path(self) -> Path:
        return self.demo_dir / "mood_calibration.json"

    @property
    def energy_store_dir(self) -> Path:
        return self.processed_dir / "eco2mix"

    @property
    def ecowatt_cache_dir(self) -> Path:
        return self.raw_dir / "ecowatt"

    @property
    def weather_features_path(self) -> Path:
        return self.processed_dir / "weather_national.parquet"

    @property
    def school_calendar_path(self) -> Path:
        return self.processed_dir / "school_calendar.parquet"

    @property
    def joined_features_path(self) -> Path:
        return self.processed_dir / "energy_weather.parquet"

    @property
    def baseline_artifact_path(self) -> Path:
        return self.processed_dir / "baseline_backtest.json"

    @property
    def mood_artifact_path(self) -> Path:
        return self.processed_dir / "mood_calibration.json"


settings = Settings()
