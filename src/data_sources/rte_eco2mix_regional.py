"""Regional RTE eco2mix data and administrative region geometry."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
import unicodedata

import pandas as pd
import requests

from src.config import settings
from src.data_processing.clean_energy_mix import clean_energy_mix
from src.public_data.contracts import SourceUnavailableError
from src.public_data.http import PublicDataHttpClient
from src.utils.io import latest_file, read_json, timestamped_path, write_json
from src.utils.logging import get_logger
from src.utils.time import iso_utc

LOGGER = get_logger(__name__)
DATASET_ID = "eco2mix-regional-tr"
PAGE_SIZE = 100
ADMIN_REGIONS_GEOJSON_URL = (
    "https://geo.api.gouv.fr/regions?geometry=contour&format=geojson"
)
ADMIN_REGIONS_DATA_GOUV_URL = (
    "https://www.data.gouv.fr/fr/datasets/api-decoupage-administratif-api-geo/"
)
REGIONAL_ODRE_URL = f"https://odre.opendatasoft.com/explore/dataset/{DATASET_ID}/"
BUNDLED_REGIONS_GEOJSON_PATH = Path(__file__).with_name("fr_regions_simplified.geojson")
BUNDLED_DEPARTMENTS_GEOJSON_PATH = Path(__file__).with_name(
    "fr_departements_simplified.geojson"
)
BUNDLED_REGIONS_GEOJSON_URL = (
    "https://github.com/gregoiredavid/france-geojson/blob/master/regions-version-simplifiee.geojson"
)
BUNDLED_DEPARTMENTS_GEOJSON_URL = (
    "https://github.com/gregoiredavid/france-geojson/blob/master/departements-version-simplifiee.geojson"
)
REQUIRED_COLUMNS = {
    "date_heure",
    "perimetre",
    "consommation",
    "nucleaire",
    "eolien",
    "solaire",
    "hydraulique",
    "gaz",
    "charbon",
    "bioenergies",
    "ech_physiques",
    "taux_co2",
}

REGION_CODES = {
    "auvergne-rhone-alpes": "84",
    "bourgogne-franche-comte": "27",
    "bretagne": "53",
    "centre-val de loire": "24",
    "corse": "94",
    "grand est": "44",
    "hauts-de-france": "32",
    "ile-de-france": "11",
    "normandie": "28",
    "nouvelle-aquitaine": "75",
    "occitanie": "76",
    "pays de la loire": "52",
    "provence-alpes-cote d azur": "93",
}
REGION_NAMES = {
    "84": "Auvergne-Rhone-Alpes",
    "27": "Bourgogne-Franche-Comte",
    "53": "Bretagne",
    "24": "Centre-Val de Loire",
    "94": "Corse",
    "44": "Grand Est",
    "32": "Hauts-de-France",
    "11": "Ile-de-France",
    "28": "Normandie",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "52": "Pays de la Loire",
    "93": "Provence-Alpes-Cote d'Azur",
}
DEMO_REGION_ROWS = [
    ("11", 11850, 420, 260, 80, 120, 540, 0, 0, 70, 45),
    ("84", 8100, 3860, 620, 420, 1650, 260, 0, 0, 170, 34),
    ("93", 6100, 0, 150, 900, 880, 1180, 0, 80, 210, 58),
    ("76", 5850, 0, 1120, 760, 980, 420, 0, 0, 320, 39),
    ("75", 6200, 1120, 520, 680, 720, 340, 0, 0, 360, 37),
    ("44", 5600, 4100, 840, 210, 720, 260, 0, 0, 180, 26),
    ("32", 5000, 3600, 760, 140, 120, 540, 120, 0, 120, 51),
    ("53", 3000, 0, 1180, 110, 260, 170, 0, 0, 110, 25),
    ("52", 3300, 0, 520, 230, 190, 240, 0, 0, 140, 34),
    ("28", 3900, 2380, 640, 160, 160, 280, 0, 0, 120, 29),
    ("24", 2800, 2400, 220, 230, 140, 150, 0, 0, 80, 24),
    ("27", 3500, 2460, 420, 240, 520, 180, 0, 0, 120, 26),
    ("94", 520, 0, 40, 75, 130, 190, 0, 35, 30, 88),
]
FALLBACK_BOUNDS = {
    "53": [(-5.2, 47.5), (-1.3, 47.5), (-1.3, 48.9), (-5.2, 48.9), (-5.2, 47.5)],
    "28": [(-1.7, 48.4), (1.8, 48.4), (1.8, 50.1), (-1.7, 50.1), (-1.7, 48.4)],
    "32": [(1.4, 49.4), (4.3, 49.4), (4.3, 51.1), (1.4, 51.1), (1.4, 49.4)],
    "44": [(4.0, 47.6), (8.3, 47.6), (8.3, 50.2), (4.0, 50.2), (4.0, 47.6)],
    "11": [(1.4, 48.1), (3.6, 48.1), (3.6, 49.2), (1.4, 49.2), (1.4, 48.1)],
    "52": [(-2.3, 46.3), (0.9, 46.3), (0.9, 48.2), (-2.3, 48.2), (-2.3, 46.3)],
    "24": [(0.0, 46.3), (3.2, 46.3), (3.2, 48.5), (0.0, 48.5), (0.0, 46.3)],
    "27": [(2.5, 46.1), (6.8, 46.1), (6.8, 48.4), (2.5, 48.4), (2.5, 46.1)],
    "75": [(-1.8, 43.1), (2.4, 43.1), (2.4, 46.8), (-1.8, 46.8), (-1.8, 43.1)],
    "76": [(0.0, 42.4), (4.9, 42.4), (4.9, 45.1), (0.0, 45.1), (0.0, 42.4)],
    "84": [(3.6, 44.1), (7.6, 44.1), (7.6, 46.7), (3.6, 46.7), (3.6, 44.1)],
    "93": [(4.3, 43.0), (7.7, 43.0), (7.7, 44.7), (4.3, 44.7), (4.3, 43.0)],
    "94": [(8.55, 41.35), (9.56, 41.35), (9.56, 43.05), (8.55, 43.05), (8.55, 41.35)],
}


class RegionalEco2MixError(RuntimeError):
    """Raised when regional eco2mix data cannot be fetched or validated."""


def _records_url() -> str:
    return f"{settings.odre_base_url}/catalog/datasets/{DATASET_ID}/records"


def normalize_region_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().replace("'", " ").split())


def region_code(value: str) -> str | None:
    return REGION_CODES.get(normalize_region_name(value))


def _validate(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RegionalEco2MixError("The regional eco2mix API returned no populated records.")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RegionalEco2MixError(
            f"Regional eco2mix response is missing columns: {sorted(missing)}"
        )


def fetch_regional_eco2mix(
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    history_hours: int | None = None,
    cache: bool = True,
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch populated regional records from the public ODRE Opendatasoft API."""
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(hours=history_hours or settings.history_hours)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("start must be earlier than end")

    where = (
        f'consommation is not null AND date_heure >= "{iso_utc(start)}" '
        f'AND date_heure <= "{iso_utc(end)}"'
    )
    client = session or requests.Session()
    http_client = None if session is not None else PublicDataHttpClient(source_name="rte_eco2mix_regional")
    records: list[dict[str, Any]] = []
    offset = 0
    total = None

    while total is None or offset < total:
        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "where": where,
            "order_by": "date_heure asc",
        }
        try:
            if http_client is None:
                response = client.get(_records_url(), params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
            else:
                payload = http_client.get_json(_records_url(), params=params)
        except (requests.RequestException, SourceUnavailableError, ValueError) as exc:
            raise RegionalEco2MixError(f"Failed to fetch regional eco2mix data: {exc}") from exc

        batch = payload.get("results")
        if not isinstance(batch, list):
            raise RegionalEco2MixError("Regional eco2mix response has no valid 'results' array.")
        total = int(payload.get("total_count", len(batch)))
        records.extend(batch)
        offset += len(batch)
        if not batch:
            break

    frame = pd.DataFrame.from_records(records)
    _validate(frame)
    if cache:
        target_dir = cache_dir or settings.raw_dir / "rte_eco2mix_regional"
        path = timestamped_path(target_dir, "eco2mix_regional")
        write_json(
            {
                "source": _records_url(),
                "source_page": REGIONAL_ODRE_URL,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "query": {"start": iso_utc(start), "end": iso_utc(end)},
                "results": records,
            },
            path,
        )
        LOGGER.info("Cached %s raw regional eco2mix records at %s", len(frame), path)
    return frame


def load_cached_regional_eco2mix(
    path: Path | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    if path is None:
        directory = cache_dir or settings.raw_dir / "rte_eco2mix_regional"
        path = latest_file(directory, "eco2mix_regional_*.json")
    if path is None or not path.exists():
        raise FileNotFoundError("No cached regional eco2mix file found.")
    payload = read_json(path)
    records = payload.get("results", payload) if isinstance(payload, dict) else payload
    frame = pd.DataFrame.from_records(records)
    _validate(frame)
    LOGGER.info("Loaded %s cached regional eco2mix records from %s", len(frame), path)
    return frame


def demo_regional_snapshot(timestamp: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return a stable demo snapshot shaped like cleaned regional eco2mix data."""
    ts = timestamp or pd.Timestamp.now(tz="UTC").floor("h")
    records = []
    for (
        code,
        consumption,
        nuclear,
        wind,
        solar,
        hydro,
        gas,
        coal,
        oil,
        bioenergy,
        co2,
    ) in DEMO_REGION_ROWS:
        records.append(
            {
                "date_heure": ts.isoformat(),
                "perimetre": REGION_NAMES[code],
                "consommation": consumption,
                "nucleaire": nuclear,
                "eolien": wind,
                "solaire": solar,
                "hydraulique": hydro,
                "gaz": gas,
                "charbon": coal,
                "fioul": oil,
                "bioenergies": bioenergy,
                "ech_physiques": 0,
                "taux_co2": co2,
            }
        )
    return prepare_regional_snapshot(pd.DataFrame.from_records(records))


def prepare_regional_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean regional records and keep the latest row for every known region."""
    clean = clean_energy_mix(raw)
    clean["region_code"] = clean["region"].map(region_code)
    clean = clean.dropna(subset=["region_code"]).copy()
    if clean.empty:
        raise RegionalEco2MixError("No regional records matched French administrative regions.")
    clean["region_code"] = clean["region_code"].astype(str)
    clean["region_display"] = clean["region_code"].map(REGION_NAMES).fillna(clean["region"])
    clean = clean.sort_values("timestamp").drop_duplicates("region_code", keep="last")
    peak_demand = clean["consumption_mw"].max()
    total_demand = clean["consumption_mw"].sum()
    clean["demand_pressure"] = clean["consumption_mw"] / peak_demand if peak_demand else 0
    clean["national_demand_share"] = clean["consumption_mw"] / total_demand if total_demand else 0
    clean["regional_balance_mw"] = clean["total_production_mw"] - clean["consumption_mw"]
    clean["balance_ratio"] = clean["regional_balance_mw"] / clean["consumption_mw"].where(clean["consumption_mw"] > 0)
    clean["demand_rank"] = clean["consumption_mw"].rank(method="min", ascending=False).astype(int)
    clean["renewable_rank"] = clean["renewable_share"].rank(method="min", ascending=False).astype(int)
    clean["pressure_band"] = pd.cut(
        clean["demand_pressure"],
        bins=[-0.01, 0.55, 0.75, 0.88, 1.01],
        labels=["Light", "Visible", "Elevated", "Peak"],
    ).astype(str)
    return clean.sort_values("region_display").reset_index(drop=True)


def load_region_geojson(
    *,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    """Load official French administrative regions as GeoJSON from API Geo."""
    client = session or requests.Session()
    try:
        response = client.get(ADMIN_REGIONS_GEOJSON_URL, timeout=timeout)
        response.raise_for_status()
        geojson = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RegionalEco2MixError(f"Failed to load region GeoJSON: {exc}") from exc
    return normalize_region_geojson(geojson)


def _geometry_from_api_item(item: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("geometry", "contour", "geo_shape", "geom"):
        value = item.get(key)
        if not isinstance(value, dict):
            continue
        if value.get("type") == "Feature":
            geometry = value.get("geometry")
            return geometry if isinstance(geometry, dict) else None
        if value.get("type") in {"Polygon", "MultiPolygon"}:
            return value
        geometry = value.get("geometry")
        if isinstance(geometry, dict):
            return geometry
    return None


def _normalizable_features(geojson: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(geojson, list):
        features = []
        for item in geojson:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "Feature":
                features.append(item)
                continue
            geometry = _geometry_from_api_item(item)
            properties = {
                key: value
                for key, value in item.items()
                if key not in {"geometry", "contour", "geo_shape", "geom"}
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometry,
                }
            )
        return features
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise RegionalEco2MixError("Region GeoJSON is not a FeatureCollection.")
    features = geojson.get("features")
    if not isinstance(features, list) or not features:
        raise RegionalEco2MixError("Region GeoJSON contains no features.")
    return features


def normalize_region_geojson(geojson: dict[str, Any] | list[Any]) -> dict[str, Any]:
    features = _normalizable_features(geojson)
    normalized_features = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = dict(feature.get("properties") or {})
        code = str(properties.get("code") or properties.get("codeRegion") or "")
        name = str(properties.get("nom") or properties.get("name") or "")
        if not code and name:
            code = region_code(name) or ""
        if code not in REGION_NAMES or not feature.get("geometry"):
            continue
        normalized = dict(feature)
        normalized["id"] = code
        normalized["properties"] = {
            **properties,
            "code": code,
            "name": REGION_NAMES[code],
        }
        normalized_features.append(normalized)
    if not normalized_features:
        raise RegionalEco2MixError("Region GeoJSON has no usable metropolitan region features.")
    return {"type": "FeatureCollection", "features": normalized_features}


def _rough_region_box_geojson() -> dict[str, Any]:
    """Emergency last-resort geometry if the bundled asset is missing."""
    features = []
    for code, coordinates in FALLBACK_BOUNDS.items():
        features.append(
            {
                "type": "Feature",
                "id": code,
                "properties": {"code": code, "name": REGION_NAMES[code]},
                "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in coordinates]]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def fallback_region_geojson() -> dict[str, Any]:
    """Bundled offline GeoJSON fallback with real simplified regional boundaries."""
    try:
        with BUNDLED_REGIONS_GEOJSON_PATH.open(encoding="utf-8") as handle:
            return normalize_region_geojson(json.load(handle))
    except (OSError, json.JSONDecodeError, RegionalEco2MixError):
        return _rough_region_box_geojson()


def fallback_department_geojson() -> dict[str, Any]:
    """Bundled simplified department boundaries used as visual map context."""
    with BUNDLED_DEPARTMENTS_GEOJSON_PATH.open(encoding="utf-8") as handle:
        geojson = json.load(handle)
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise RegionalEco2MixError("Department GeoJSON is not a FeatureCollection.")
    if not isinstance(geojson.get("features"), list) or not geojson["features"]:
        raise RegionalEco2MixError("Department GeoJSON contains no features.")
    return geojson


def source_attribution() -> dict[str, str]:
    return {
        "regional_eco2mix": REGIONAL_ODRE_URL,
        "regional_geojson": ADMIN_REGIONS_DATA_GOUV_URL,
        "regional_geojson_api": ADMIN_REGIONS_GEOJSON_URL,
        "regional_geojson_fallback": BUNDLED_REGIONS_GEOJSON_URL,
        "department_geojson_fallback": BUNDLED_DEPARTMENTS_GEOJSON_URL,
    }
