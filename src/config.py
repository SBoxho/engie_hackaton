from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
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
    history_hours: int = int(os.getenv("ENERGY_PULSE_HISTORY_HOURS", "72"))

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

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
    def joined_features_path(self) -> Path:
        return self.processed_dir / "energy_weather.parquet"

    @property
    def baseline_artifact_path(self) -> Path:
        return self.processed_dir / "baseline_backtest.json"

    @property
    def mood_artifact_path(self) -> Path:
        return self.processed_dir / "mood_calibration.json"


settings = Settings()
