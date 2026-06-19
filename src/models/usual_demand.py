"""Hourly leakage-safe demand features and transparent usual-demand baselines.

This module intentionally stops before machine learning.  It builds model-ready
hourly feature rows from normalized public records and evaluates a comparable
history baseline that always exposes the fallback level and sample size used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.data_sources.school_calendar import school_holiday_features


SCHEMA_VERSION = 1
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_HORIZONS_HOURS = (1, 24, 48)
DEMAND_LAGS_HOURS = (1, 24, 48, 168)
ROLLING_WINDOWS_HOURS = (3, 24, 168)
RECENT_TREND_HOURS = (1, 24)
HEATING_BASE_C = 18.0
COOLING_BASE_C = 22.0
MAX_PLAUSIBLE_PUBLICATION_LAG = pd.Timedelta(days=7)

NATIONAL_SOURCE_NAMES = {
    "odre_eco2mix_national",
    "odre_eco2mix_national_history",
}
REGIONAL_SOURCE_NAMES = {"odre_eco2mix_regional"}
WEATHER_SOURCE_PREFIXES = ("open_meteo_weather", "open_meteo_current")
PUBLIC_HOLIDAY_SOURCE = "french_public_holidays"
SCHOOL_HOLIDAY_SOURCE = "french_school_holidays"

GENERATION_MW_COLUMNS = (
    "nuclear_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "gas_mw",
    "coal_mw",
    "oil_mw",
    "bioenergy_mw",
    "total_production_mw",
)
EXCHANGE_MW_COLUMNS = ("imports_mw", "exports_mw", "net_imports_mw", "physical_balance_mw")
FORECAST_MW_COLUMNS = ("rte_forecast_j_mw", "rte_forecast_j1_mw")
POWER_MW_COLUMNS = (
    "consumption_mw",
    *GENERATION_MW_COLUMNS,
    *EXCHANGE_MW_COLUMNS,
    *FORECAST_MW_COLUMNS,
)
ENERGY_SUM_SUFFIXES = ("_mwh", "_kwh", "_wh")
WEATHER_RENAMES = {
    "weather_temperature_c": "temperature_c",
    "weather_apparent_temperature_c": "apparent_temperature_c",
    "weather_wind_speed_kmh": "wind_speed_kmh",
    "weather_cloud_cover_pct": "cloud_cover_pct",
    "weather_solar_radiation_wm2": "solar_radiation_wm2",
    "weather_humidity_pct": "humidity_pct",
    "relative_humidity_pct": "humidity_pct",
}
WEATHER_VALUE_COLUMNS = (
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "cloud_cover_pct",
    "solar_radiation_wm2",
)
NO_FALLBACK_VALUES = {"none", "primary", "", "nan", "<na>"}
GOOD_QUALITY_VALUES = {"ok", "valid"}


@dataclass(frozen=True)
class BaselineConfig:
    """Controls sparse-history fallback behavior."""

    min_samples: int = 5
    robust_statistic: str = "median"
    recent_days: int = 28
    max_history_days: int | None = None


@dataclass(frozen=True)
class DatasetBuildResult:
    hourly: pd.DataFrame
    supervised: pd.DataFrame
    feature_manifest: dict[str, Any]
    quality_report: dict[str, Any]
    coverage_report: dict[str, Any]


def read_public_frame(path: str | Path) -> pd.DataFrame:
    """Read a parquet/CSV/JSON table used by the feature CLI."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table path: {path}")


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def normalize_public_compat_frame(
    frame: pd.DataFrame,
    *,
    source_name: str | None = None,
) -> pd.DataFrame:
    """Normalize lightly shaped compatibility inputs into public-data columns.

    The primary path is the normalized public-data store.  This adapter exists so
    commands can also be smoke-tested with existing replay parquet files that
    already contain physical fields but not public-data metadata.
    """

    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "event_time" not in result:
        timestamp_col = "timestamp" if "timestamp" in result else "date_heure" if "date_heure" in result else None
        if timestamp_col is None:
            raise ValueError("Input data requires event_time, timestamp, or date_heure.")
        result["event_time"] = result[timestamp_col]
    result["event_time"] = pd.to_datetime(result["event_time"], utc=True, errors="coerce")
    if result["event_time"].isna().any():
        raise ValueError("Input data contains invalid event_time values.")

    for source, target in WEATHER_RENAMES.items():
        if source in result and target not in result:
            result[target] = result[source]

    if "source_name" not in result:
        inferred = source_name
        if inferred is None:
            if any(column in result for column in WEATHER_VALUE_COLUMNS):
                inferred = "open_meteo_weather"
            elif "region" in result and set(result["region"].dropna().astype(str)) - {"France"}:
                inferred = "odre_eco2mix_regional"
            else:
                inferred = "odre_eco2mix_national_history"
        result["source_name"] = inferred
    if "source_revision" not in result:
        result["source_revision"] = "compat"
    if "published_at" not in result:
        result["published_at"] = pd.NaT
    if "ingested_at" not in result:
        result["ingested_at"] = result["event_time"]
    if "quality_status" not in result:
        result["quality_status"] = "ok"
    if "fallback_status" not in result:
        result["fallback_status"] = "none"
    if "region" not in result:
        result["region"] = "France"
    if "source_record_id" not in result:
        if result["source_name"].astype(str).str.startswith(WEATHER_SOURCE_PREFIXES).any():
            result["source_record_id"] = "national"
        else:
            result["source_record_id"] = result["region"].fillna("France").astype(str)
    return result


def combine_public_inputs(
    *,
    combined: pd.DataFrame | None = None,
    energy: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    public_holidays: pd.DataFrame | None = None,
    school_holidays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if combined is not None and not combined.empty:
        frames.append(normalize_public_compat_frame(combined))
    if energy is not None and not energy.empty:
        frames.append(normalize_public_compat_frame(energy))
    if weather is not None and not weather.empty:
        frames.append(normalize_public_compat_frame(weather, source_name="open_meteo_weather"))
    if public_holidays is not None and not public_holidays.empty:
        holidays = public_holidays.copy()
        if "date" in holidays and "event_time" not in holidays:
            holidays["event_time"] = pd.to_datetime(holidays["date"]).dt.tz_localize(DEFAULT_TIMEZONE).dt.tz_convert("UTC")
        holidays = normalize_public_compat_frame(holidays, source_name=PUBLIC_HOLIDAY_SOURCE)
        if "is_public_holiday" not in holidays:
            holidays["is_public_holiday"] = 1
        frames.append(holidays)
    if school_holidays is not None and not school_holidays.empty:
        school = school_holidays.copy()
        if "event_time" not in school and "start_date" in school:
            school["event_time"] = pd.to_datetime(school["start_date"]).dt.tz_localize("UTC")
        school = normalize_public_compat_frame(school, source_name=SCHOOL_HOLIDAY_SOURCE)
        if "is_school_holiday" not in school:
            school["is_school_holiday"] = 1
        frames.append(school)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_model_ready_dataset(
    public_records: pd.DataFrame,
    *,
    horizons_hours: Iterable[int] = DEFAULT_HORIZONS_HOURS,
    timezone: str = DEFAULT_TIMEZONE,
) -> DatasetBuildResult:
    hourly = build_hourly_analytical_dataset(public_records, timezone=timezone)
    supervised = build_baseline_training_dataset(
        hourly,
        horizons_hours=tuple(horizons_hours),
        timezone=timezone,
    )
    validate_no_future_observations(supervised)
    manifest = build_feature_manifest(supervised.columns)
    quality = build_data_quality_report(hourly, supervised)
    coverage = build_feature_coverage_report(supervised)
    return DatasetBuildResult(hourly, supervised, manifest, quality, coverage)


def build_hourly_analytical_dataset(
    public_records: pd.DataFrame,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    derive_national_from_regions: bool = True,
) -> pd.DataFrame:
    """Build one hourly state row per geography.

    The hourly timestamp is the hour end in UTC.  MW fields are averaged within
    the preceding hour because they represent power.  Energy-like fields ending
    in MWh/kWh/Wh are summed.  Regional rows can be summed across simultaneous
    regions to form a derived national total only when an official national row
    is absent for that hour.
    """

    records = normalize_public_compat_frame(public_records) if not public_records.empty else public_records.copy()
    if records.empty:
        return pd.DataFrame()
    energy_hourly = _resample_energy_hourly(records)
    if derive_national_from_regions:
        energy_hourly = _append_derived_national_rows(energy_hourly)
    weather_hourly = _resample_weather_hourly(records)
    hourly = _attach_weather(energy_hourly, weather_hourly)
    public_dates = _public_holiday_dates(records, hourly["timestamp"], timezone)
    school_calendar = _school_calendar(records)
    hourly = _add_calendar_features(
        hourly,
        timestamp_col="timestamp",
        timezone=timezone,
        public_holiday_dates=public_dates,
        school_calendar=school_calendar,
        prefix="",
    )
    hourly = _add_recent_features(hourly)
    hourly = _add_national_recent_features(hourly)
    hourly = _finalize_missing_indicators(hourly)
    hourly = hourly.sort_values(["timestamp", "geographic_scope", "region"], kind="stable").reset_index(drop=True)
    return _stable_column_order(hourly)


def build_baseline_training_dataset(
    hourly: pd.DataFrame,
    *,
    horizons_hours: Iterable[int] = DEFAULT_HORIZONS_HOURS,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Create direct-horizon rows with origin-safe features and future target labels."""

    if hourly.empty:
        return pd.DataFrame()
    source = hourly.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    source["feature_available_at"] = pd.to_datetime(source["feature_available_at"], utc=True, errors="coerce")
    origin_safe = source[source["feature_available_at"].le(source["timestamp"])].copy()
    origin_safe = origin_safe.rename(columns={"timestamp": "origin_timestamp"})
    origin_safe["forecast_origin"] = origin_safe["origin_timestamp"]

    target_columns = [
        "timestamp",
        "geographic_scope",
        "region",
        "consumption_mw",
        "feature_available_at",
        "hour_of_day",
        "weekday",
        "weekday_type",
        "is_weekend",
        "month",
        "season",
        "season_code",
        "is_public_holiday",
        "holiday_type",
        "school_holiday_zone_a",
        "school_holiday_zone_b",
        "school_holiday_zone_c",
        "school_holiday_any_zone",
        "school_holiday_all_zones",
        "utc_offset_hours",
        "is_dst",
    ]
    target_lookup = source[[column for column in target_columns if column in source]].copy()
    target_lookup = target_lookup.rename(
        columns={
            "timestamp": "target_timestamp",
            "consumption_mw": "target_mw",
            "feature_available_at": "target_observation_available_at",
            **{
                column: f"target_{column}"
                for column in target_columns
                if column
                not in {
                    "timestamp",
                    "geographic_scope",
                    "region",
                    "consumption_mw",
                    "feature_available_at",
                }
            },
        }
    )

    parts: list[pd.DataFrame] = []
    horizons = tuple(sorted({int(value) for value in horizons_hours}))
    if not horizons or any(value < 0 for value in horizons):
        raise ValueError("horizons_hours must contain non-negative whole hours")
    for horizon in horizons:
        rows = origin_safe.copy()
        rows["horizon_hours"] = int(horizon)
        rows["target_timestamp"] = rows["origin_timestamp"] + pd.Timedelta(hours=int(horizon))
        rows = rows.merge(
            target_lookup,
            on=["target_timestamp", "geographic_scope", "region"],
            how="left",
            validate="many_to_one",
        )
        parts.append(rows)
    supervised = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    supervised = _fill_missing_target_calendar(supervised, source, timezone)
    supervised = supervised.sort_values(
        ["origin_timestamp", "horizon_hours", "geographic_scope", "region"],
        kind="stable",
    ).reset_index(drop=True)
    supervised = _stable_column_order(supervised)
    validate_no_future_observations(supervised)
    return supervised


def compute_usual_demand_baselines(
    supervised: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    config: BaselineConfig | None = None,
) -> pd.DataFrame:
    """Attach usual-demand predictions to direct-horizon rows."""

    config = config or BaselineConfig()
    if config.min_samples < 1:
        raise ValueError("min_samples must be positive")
    if supervised.empty:
        return supervised.copy()
    history = hourly.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history["consumption_mw"] = pd.to_numeric(history["consumption_mw"], errors="coerce")
    history = history.dropna(subset=["timestamp", "consumption_mw"]).sort_values("timestamp", kind="stable")

    history_by_geo = {
        key: group.reset_index(drop=True)
        for key, group in history.groupby(["geographic_scope", "region"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for item in supervised.itertuples(index=False):
        row = item._asdict()
        key = (str(row["geographic_scope"]), str(row["region"]))
        group = history_by_geo.get(key, pd.DataFrame())
        baseline = _usual_baseline_for_row(row, group, config)
        output = dict(row)
        output.update(baseline)
        target = _finite_float(output.get("target_mw"))
        predicted = _finite_float(output.get("usual_demand_mw"))
        if target is not None and predicted not in (None, 0.0):
            output["actual_above_usual_percent"] = 100.0 * (target - predicted) / predicted
        else:
            output["actual_above_usual_percent"] = np.nan
        rows.append(output)
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["origin_timestamp", "horizon_hours", "geographic_scope", "region"],
        kind="stable",
    ).reset_index(drop=True)
    validate_no_future_observations(result)
    return _stable_column_order(result)


def evaluate_usual_demand_baseline(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return deterministic rolling backtest metrics for usual demand."""

    if predictions.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "method": "usual-demand comparable-history baseline",
            "overall": _empty_metric_row(),
            "by_horizon": [],
            "by_season": [],
            "by_weekday_type": [],
            "by_region": [],
            "by_fallback_level": [],
            "weak_data_periods": [],
        }
    scored = predictions.copy()
    scored["target_mw"] = pd.to_numeric(scored["target_mw"], errors="coerce")
    scored["usual_demand_mw"] = pd.to_numeric(scored["usual_demand_mw"], errors="coerce")
    valid = scored["target_mw"].notna() & scored["usual_demand_mw"].notna()
    scored = scored.loc[valid].copy()
    scored["absolute_error_mw"] = (scored["target_mw"] - scored["usual_demand_mw"]).abs()
    scored["absolute_error_gw"] = scored["absolute_error_mw"] / 1000.0
    scored["signed_error_mw"] = scored["usual_demand_mw"] - scored["target_mw"]

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "usual-demand comparable-history baseline",
        "fallback_hierarchy": fallback_hierarchy(),
        "overall": _metric_row(scored),
        "by_horizon": _group_metric_rows(scored, ["horizon_hours"]),
        "by_season": _group_metric_rows(scored, ["target_season"]),
        "by_weekday_type": _group_metric_rows(scored, ["target_weekday_type"]),
        "by_region": _group_metric_rows(scored, ["geographic_scope", "region"]),
        "by_fallback_level": _group_metric_rows(scored, ["usual_demand_fallback_level", "usual_demand_method"]),
        "weak_data_periods": _weak_data_periods(predictions),
        "prediction_count": int(len(predictions)),
        "scored_prediction_count": int(len(scored)),
    }


def compute_usual_demand_state(
    hourly: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    config: BaselineConfig | None = None,
) -> pd.DataFrame:
    """Compute current actual versus usual percent for each latest geography."""

    if hourly.empty:
        return pd.DataFrame()
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        data = data[data["timestamp"].le(cutoff)]
    latest = (
        data.sort_values("timestamp", kind="stable")
        .dropna(subset=["consumption_mw"])
        .groupby(["geographic_scope", "region"], as_index=False, sort=False)
        .tail(1)
    )
    if latest.empty:
        return pd.DataFrame()
    rows = latest.rename(columns={"timestamp": "target_timestamp", "consumption_mw": "target_mw"}).copy()
    rows["origin_timestamp"] = rows["target_timestamp"]
    rows["horizon_hours"] = 0
    for column in (
        "hour_of_day",
        "weekday",
        "weekday_type",
        "is_weekend",
        "month",
        "season",
        "season_code",
        "is_public_holiday",
        "holiday_type",
    ):
        if column in rows:
            rows[f"target_{column}"] = rows[column]
    latest_keys = set(zip(latest["geographic_scope"], latest["region"], latest["timestamp"]))
    history = data[
        ~data[["geographic_scope", "region", "timestamp"]].apply(tuple, axis=1).isin(latest_keys)
    ].copy()
    predictions = compute_usual_demand_baselines(rows, history, config=config)
    predictions["above_usual_percent"] = predictions["actual_above_usual_percent"]
    return predictions[
        [
            "target_timestamp",
            "geographic_scope",
            "region",
            "target_mw",
            "usual_demand_mw",
            "above_usual_percent",
            "usual_demand_method",
            "usual_demand_sample_count",
            "usual_demand_fallback_level",
        ]
    ].reset_index(drop=True)


def validate_no_future_observations(frame: pd.DataFrame) -> None:
    """Reject supervised feature rows whose observable inputs postdate origin."""

    if frame.empty:
        return
    if {"origin_timestamp", "feature_available_at"}.issubset(frame.columns):
        origin = pd.to_datetime(frame["origin_timestamp"], utc=True, errors="coerce")
        available = pd.to_datetime(frame["feature_available_at"], utc=True, errors="coerce")
        invalid = available.notna() & origin.notna() & available.gt(origin)
        if invalid.any():
            raise ValueError("Feature rows contain source data unavailable at the forecast origin.")
    source_columns = [
        column
        for column in (
            "source_event_time_max",
            "weather_source_event_time",
            "usual_demand_source_timestamp_max",
        )
        if column in frame
    ]
    for column in source_columns:
        source_time = pd.to_datetime(frame[column], utc=True, errors="coerce")
        origin = pd.to_datetime(frame["origin_timestamp"], utc=True, errors="coerce")
        invalid = source_time.notna() & origin.notna() & source_time.gt(origin)
        if invalid.any():
            raise ValueError(f"{column} contains observations after the forecast origin.")
    if {"origin_timestamp", "target_timestamp", "horizon_hours"}.issubset(frame.columns):
        delta = pd.to_datetime(frame["target_timestamp"], utc=True) - pd.to_datetime(
            frame["origin_timestamp"], utc=True
        )
        expected = pd.to_timedelta(frame["horizon_hours"], unit="h")
        if not delta.eq(expected).all():
            raise ValueError("Target timestamp must equal origin plus horizon.")


def fallback_hierarchy() -> list[dict[str, Any]]:
    return [
        {
            "level": 1,
            "method": "same_hour_weekday_type_season_holiday_type",
            "description": "Same local hour, weekday/weekend type, season, holiday type, and geography.",
        },
        {
            "level": 2,
            "method": "same_hour_weekday_type_season",
            "description": "Same local hour, weekday/weekend type, season, and geography.",
        },
        {
            "level": 3,
            "method": "same_hour_weekday_type",
            "description": "Same local hour, weekday/weekend type, and geography.",
        },
        {
            "level": 4,
            "method": "same_hour",
            "description": "Same local hour and geography.",
        },
        {
            "level": 5,
            "method": "recent_rolling_seasonal",
            "description": "Recent historical median for the same season and geography, then recent median.",
        },
    ]


def build_feature_manifest(columns: Iterable[str] | None = None) -> dict[str, Any]:
    selected = sorted(set(columns) if columns is not None else ())
    feature_rows = [_manifest_row(column) for column in selected if _is_feature_column(column)]
    return {
        "schema_version": SCHEMA_VERSION,
        "timezone": DEFAULT_TIMEZONE,
        "target_column": "target_mw",
        "horizons_hours": list(DEFAULT_HORIZONS_HOURS),
        "resampling_rules": {
            "hourly_timestamp": "UTC hour end; energy rows in [T-1h, T) feed timestamp T.",
            "power_mw": "Mean within the hour; MW is power and is not summed over time.",
            "energy_mwh_kwh_wh": "Sum within the hour; these columns are energy quantities.",
            "weather": "Mean across available public weather records at the same event hour, then backward as-of join to origin.",
            "quality": "Counts, fractions, and max source timestamps; quality flags are never averaged into physical values.",
            "geography": "Regional MW values are summed across regions only to derive a national total when no official national row exists.",
        },
        "leakage_controls": [
            "Feature rows keep only origins with feature_available_at <= origin_timestamp.",
            "Weather is joined only from public rows with weather event_time <= origin and availability <= origin.",
            "Demand lags and rolling demand features are computed within each geography from timestamps <= origin.",
            "Target calendar fields are deterministic calendar values; target demand remains the label, not a feature.",
            "Usual-demand comparison samples always have timestamp <= forecast origin.",
        ],
        "features": feature_rows,
    }


def build_data_quality_report(hourly: pd.DataFrame, supervised: pd.DataFrame | None = None) -> dict[str, Any]:
    if hourly.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "row_count": 0,
            "weak_data_periods": [],
        }
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    reports: list[dict[str, Any]] = []
    for (scope, region), group in data.groupby(["geographic_scope", "region"], sort=True):
        group = group.sort_values("timestamp", kind="stable")
        times = pd.DatetimeIndex(group["timestamp"].dropna().drop_duplicates())
        missing = _missing_hour_count(times)
        reports.append(
            {
                "geographic_scope": str(scope),
                "region": str(region),
                "row_count": int(len(group)),
                "start_utc": _iso(group["timestamp"].min()),
                "end_utc": _iso(group["timestamp"].max()),
                "missing_hour_count": int(missing),
                "mean_energy_coverage_ratio": _mean_or_none(group.get("energy_coverage_ratio")),
                "weather_missing_fraction": _mean_or_none(group.get("weather_missing")),
                "source_fallback_record_count": int(pd.to_numeric(group.get("source_fallback_record_count"), errors="coerce").fillna(0).sum())
                if "source_fallback_record_count" in group
                else 0,
            }
        )
    supervised_count = 0 if supervised is None else int(len(supervised))
    return {
        "schema_version": SCHEMA_VERSION,
        "hourly_row_count": int(len(hourly)),
        "supervised_row_count": supervised_count,
        "start_utc": _iso(data["timestamp"].min()),
        "end_utc": _iso(data["timestamp"].max()),
        "geography": reports,
        "weak_data_periods": _weak_hourly_periods(data),
    }


def build_feature_coverage_report(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"schema_version": SCHEMA_VERSION, "row_count": 0, "features": []}
    features = []
    for column in sorted(column for column in frame.columns if _is_feature_column(column)):
        values = frame[column]
        missing = int(values.isna().sum()) if hasattr(values, "isna") else 0
        features.append(
            {
                "feature": column,
                "missing_count": missing,
                "non_null_fraction": float(1.0 - missing / len(frame)) if len(frame) else 0.0,
                "unit": _feature_unit(column),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": int(len(frame)),
        "features": features,
    }


def write_backtest_artifact(predictions: pd.DataFrame, metrics: Mapping[str, Any], path: str | Path) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "usual-demand comparable-history rolling backtest",
        "fallback_hierarchy": fallback_hierarchy(),
        "metrics": metrics,
        "predictions": _json_records(predictions),
    }
    return write_json(payload, path)


def _resample_energy_hourly(records: pd.DataFrame) -> pd.DataFrame:
    has_consumption = (
        records["consumption_mw"].notna()
        if "consumption_mw" in records
        else pd.Series(False, index=records.index)
    )
    energy = records[
        records["source_name"].astype(str).isin(NATIONAL_SOURCE_NAMES | REGIONAL_SOURCE_NAMES)
        | has_consumption
    ].copy()
    if energy.empty:
        raise ValueError("No normalized energy records with consumption_mw were found.")
    energy["event_time"] = pd.to_datetime(energy["event_time"], utc=True, errors="coerce")
    energy = energy.dropna(subset=["event_time"])
    energy["timestamp"] = energy["event_time"].dt.floor("h") + pd.Timedelta(hours=1)
    energy["region"] = energy.get("region", "France").fillna("France").astype(str)
    energy["source_name"] = energy["source_name"].astype(str)
    energy["geographic_scope"] = np.where(
        energy["source_name"].isin(REGIONAL_SOURCE_NAMES),
        "regional",
        np.where(energy["region"].eq("France"), "national", "regional"),
    )
    energy["_availability_time"] = _availability_time(energy)
    rows: list[dict[str, Any]] = []
    group_columns = ["geographic_scope", "region", "timestamp"]
    for key, group in energy.sort_values("event_time", kind="stable").groupby(group_columns, sort=True):
        row: dict[str, Any] = {
            "geographic_scope": str(key[0]),
            "region": str(key[1]),
            "timestamp": pd.Timestamp(key[2]),
            "is_derived_national_from_regions": 0,
            "source_names": ",".join(sorted(set(group["source_name"].astype(str)))),
            "source_event_time_max": group["event_time"].max(),
            "source_availability_time_max": group["_availability_time"].max(),
            "feature_available_at": group["_availability_time"].max(),
            "energy_observation_count": int(len(group)),
            "energy_consumption_observation_count": int(pd.to_numeric(group.get("consumption_mw"), errors="coerce").notna().sum())
            if "consumption_mw" in group
            else 0,
            "energy_expected_observation_count": int(_expected_observations_per_hour(group)),
        }
        for column in _energy_mean_columns(group):
            row[column] = _mean_or_nan(group[column])
        for column in _energy_sum_columns(group):
            row[column] = pd.to_numeric(group[column], errors="coerce").sum(min_count=1)
        row.update(_quality_aggregate(group, prefix="source"))
        if row["energy_expected_observation_count"]:
            row["energy_coverage_ratio"] = min(
                1.0,
                row["energy_consumption_observation_count"] / row["energy_expected_observation_count"],
            )
        else:
            row["energy_coverage_ratio"] = np.nan
        rows.append(row)
    hourly = pd.DataFrame(rows)
    if "consumption_mw" not in hourly:
        hourly["consumption_mw"] = np.nan
    return hourly


def _append_derived_national_rows(hourly: pd.DataFrame) -> pd.DataFrame:
    regional = hourly[hourly["geographic_scope"].eq("regional")].copy()
    if regional.empty:
        return hourly
    official_national_times = set(
        pd.to_datetime(
            hourly.loc[hourly["geographic_scope"].eq("national"), "timestamp"],
            utc=True,
            errors="coerce",
        )
    )
    rows: list[dict[str, Any]] = []
    for timestamp, group in regional.groupby("timestamp", sort=True):
        if pd.Timestamp(timestamp) in official_national_times:
            continue
        row: dict[str, Any] = {
            "geographic_scope": "national",
            "region": "France",
            "timestamp": pd.Timestamp(timestamp),
            "is_derived_national_from_regions": 1,
            "source_names": "regional_sum",
            "source_event_time_max": group["source_event_time_max"].max(),
            "source_availability_time_max": group["source_availability_time_max"].max(),
            "feature_available_at": group["feature_available_at"].max(),
            "energy_observation_count": int(pd.to_numeric(group["energy_observation_count"], errors="coerce").fillna(0).sum()),
            "energy_consumption_observation_count": int(pd.to_numeric(group["energy_consumption_observation_count"], errors="coerce").fillna(0).sum()),
            "energy_expected_observation_count": int(pd.to_numeric(group["energy_expected_observation_count"], errors="coerce").fillna(0).sum()),
        }
        for column in [column for column in POWER_MW_COLUMNS if column in group]:
            if column == "co2_intensity_g_per_kwh":
                continue
            row[column] = pd.to_numeric(group[column], errors="coerce").sum(min_count=1)
        for column in [column for column in group.columns if _is_energy_sum_column(column)]:
            row[column] = pd.to_numeric(group[column], errors="coerce").sum(min_count=1)
        if "co2_intensity_g_per_kwh" in group:
            weights = pd.to_numeric(group.get("consumption_mw"), errors="coerce")
            values = pd.to_numeric(group["co2_intensity_g_per_kwh"], errors="coerce")
            valid = values.notna() & weights.notna() & weights.gt(0)
            row["co2_intensity_g_per_kwh"] = (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any()
                else _mean_or_nan(values)
            )
        row["source_quality_ok_fraction"] = _mean_or_nan(group.get("source_quality_ok_fraction", pd.Series(dtype=float)))
        row["source_schema_failure_count"] = int(pd.to_numeric(group.get("source_schema_failure_count"), errors="coerce").fillna(0).sum())
        row["source_fallback_record_count"] = int(pd.to_numeric(group.get("source_fallback_record_count"), errors="coerce").fillna(0).sum())
        row["energy_coverage_ratio"] = (
            row["energy_consumption_observation_count"] / row["energy_expected_observation_count"]
            if row["energy_expected_observation_count"]
            else np.nan
        )
        rows.append(row)
    if not rows:
        return hourly
    return pd.concat([hourly, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _resample_weather_hourly(records: pd.DataFrame) -> pd.DataFrame:
    source = records.copy()
    for original, renamed in WEATHER_RENAMES.items():
        if original in source and renamed not in source:
            source[renamed] = source[original]
    source_names = source.get("source_name", pd.Series("", index=source.index)).astype(str)
    weather = source[
        source_names.str.startswith(WEATHER_SOURCE_PREFIXES)
        | source[[column for column in WEATHER_VALUE_COLUMNS if column in source]].notna().any(axis=1)
    ].copy()
    if weather.empty:
        return pd.DataFrame()
    weather["event_time"] = pd.to_datetime(weather["event_time"], utc=True, errors="coerce")
    weather = weather.dropna(subset=["event_time"])
    weather["timestamp"] = weather["event_time"].dt.floor("h")
    weather["_availability_time"] = _availability_time(weather)
    rows: list[dict[str, Any]] = []
    for timestamp, group in weather.sort_values("event_time", kind="stable").groupby("timestamp", sort=True):
        row: dict[str, Any] = {
            "weather_timestamp": pd.Timestamp(timestamp),
            "weather_source_event_time": group["event_time"].max(),
            "weather_availability_time_max": group["_availability_time"].max(),
            "weather_location_count": int(group.get("source_record_id", pd.Series(index=group.index)).nunique(dropna=True)),
            "weather_observation_count": int(len(group)),
        }
        for column in WEATHER_VALUE_COLUMNS:
            if column in group:
                row[f"weather_{column}"] = _mean_or_nan(group[column])
        row.update(_quality_aggregate(group, prefix="weather_source"))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    available = pd.concat(
        [result["weather_timestamp"], result["weather_availability_time_max"]],
        axis=1,
    ).max(axis=1)
    result["weather_available_from"] = available
    return result.sort_values("weather_available_from", kind="stable").reset_index(drop=True)


def _attach_weather(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    result = energy.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if weather.empty:
        for column in WEATHER_VALUE_COLUMNS:
            result[f"weather_{column}"] = np.nan
        result["weather_missing"] = 1
        result["weather_location_count"] = 0
        result["weather_observation_count"] = 0
        result["weather_source_age_hours"] = np.nan
        return _add_weather_derived(result)

    left = result.sort_values("timestamp", kind="stable").reset_index()
    right = weather.sort_values("weather_available_from", kind="stable").copy()
    joined = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="weather_available_from",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("index").sort_index()
    for column in weather.columns:
        if column not in result and column in joined:
            result[column] = joined[column]
    result["weather_source_age_hours"] = (
        result["timestamp"] - pd.to_datetime(result["weather_source_event_time"], utc=True, errors="coerce")
    ) / pd.Timedelta(hours=1)
    stale = result["weather_source_age_hours"].isna() | result["weather_source_age_hours"].gt(3)
    for column in WEATHER_VALUE_COLUMNS:
        output = f"weather_{column}"
        if output not in result:
            result[output] = np.nan
        result.loc[stale, output] = np.nan
    result.loc[stale, "weather_location_count"] = 0
    result.loc[stale, "weather_observation_count"] = 0
    result["weather_missing"] = stale.astype(int)
    weather_available = pd.to_datetime(result.get("weather_available_from"), utc=True, errors="coerce")
    result["feature_available_at"] = pd.concat(
        [
            pd.to_datetime(result["feature_available_at"], utc=True, errors="coerce"),
            weather_available,
        ],
        axis=1,
    ).max(axis=1)
    return _add_weather_derived(result)


def _add_weather_derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    temp = pd.to_numeric(result.get("weather_temperature_c"), errors="coerce")
    humidity = pd.to_numeric(result.get("weather_humidity_pct"), errors="coerce")
    wind_kmh = pd.to_numeric(result.get("weather_wind_speed_kmh"), errors="coerce")
    vapor_pressure = (humidity / 100.0) * 6.105 * np.exp((17.27 * temp) / (237.7 + temp))
    result["weather_apparent_temperature_c"] = temp + 0.33 * vapor_pressure - 0.70 * (wind_kmh / 3.6) - 4.0
    result["heating_degree_c"] = (HEATING_BASE_C - temp).clip(lower=0)
    result["cooling_degree_c"] = (temp - COOLING_BASE_C).clip(lower=0)
    for column in (
        "weather_temperature_c",
        "weather_apparent_temperature_c",
        "weather_wind_speed_kmh",
        "weather_cloud_cover_pct",
        "weather_solar_radiation_wm2",
        "heating_degree_c",
        "cooling_degree_c",
    ):
        result[f"{column}_missing"] = pd.to_numeric(result.get(column), errors="coerce").isna().astype(int)
    return result


def _add_calendar_features(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    timezone: str,
    public_holiday_dates: set[Any],
    school_calendar: pd.DataFrame | None,
    prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(result[timestamp_col], utc=True, errors="coerce"))
    local_period_start = pd.Series((timestamps - pd.Timedelta(hours=1)).tz_convert(timezone), index=result.index)
    local_dates = local_period_start.dt.date
    result[f"{prefix}hour_of_day"] = local_period_start.dt.hour.astype("Int64")
    result[f"{prefix}weekday"] = local_period_start.dt.dayofweek.astype("Int64")
    result[f"{prefix}weekday_type"] = np.where(result[f"{prefix}weekday"].ge(5), "weekend", "weekday")
    result[f"{prefix}is_weekend"] = result[f"{prefix}weekday"].ge(5).astype(int)
    result[f"{prefix}month"] = local_period_start.dt.month.astype("Int64")
    result[f"{prefix}season"] = result[f"{prefix}month"].map(_season_label)
    result[f"{prefix}season_code"] = result[f"{prefix}month"].map(_season_code).astype("Int64")
    result[f"{prefix}is_public_holiday"] = local_dates.map(lambda day: int(day in public_holiday_dates)).astype(int)
    school = school_holiday_features(local_period_start, school_calendar)
    for column in school.columns:
        result[f"{prefix}{column}"] = school[column].astype(int)
    result[f"{prefix}holiday_type"] = [
        _holiday_type(public, school_any)
        for public, school_any in zip(
            result[f"{prefix}is_public_holiday"],
            result[f"{prefix}school_holiday_any_zone"],
        )
    ]
    result[f"{prefix}utc_offset_hours"] = local_period_start.map(
        lambda value: value.utcoffset().total_seconds() / 3600 if pd.notna(value) else np.nan
    )
    result[f"{prefix}is_dst"] = local_period_start.map(
        lambda value: int(bool(value.dst() and value.dst().total_seconds())) if pd.notna(value) else 0
    )
    return result


def _fill_missing_target_calendar(supervised: pd.DataFrame, hourly: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if supervised.empty or "target_timestamp" not in supervised:
        return supervised
    public_dates = set()
    if {"timestamp", "is_public_holiday"}.issubset(hourly.columns):
        holiday_rows = hourly[pd.to_numeric(hourly["is_public_holiday"], errors="coerce").fillna(0).eq(1)]
        if not holiday_rows.empty:
            public_dates.update(
                (pd.DatetimeIndex(pd.to_datetime(holiday_rows["timestamp"], utc=True)) - pd.Timedelta(hours=1))
                .tz_convert(timezone)
                .date
            )
    computed = _add_calendar_features(
        supervised[["target_timestamp"]].copy(),
        timestamp_col="target_timestamp",
        timezone=timezone,
        public_holiday_dates=public_dates,
        school_calendar=None,
        prefix="target_",
    )
    result = supervised.copy()
    fill_columns = [column for column in computed.columns if column.startswith("target_") and column != "target_timestamp"]
    for column in fill_columns:
        if column not in result:
            result[column] = computed[column]
        else:
            result[column] = result[column].where(result[column].notna(), computed[column])
    return result


def _add_recent_features(hourly: pd.DataFrame) -> pd.DataFrame:
    result = hourly.sort_values(["geographic_scope", "region", "timestamp"], kind="stable").copy()
    parts: list[pd.DataFrame] = []
    for _, group in result.groupby(["geographic_scope", "region"], sort=False):
        group = group.copy().set_index("timestamp").sort_index()
        demand = pd.to_numeric(group["consumption_mw"], errors="coerce")
        group["origin_demand_mw"] = demand
        for hours in DEMAND_LAGS_HOURS:
            group[f"demand_lag_{hours}h_mw"] = demand.shift(hours)
        for hours in ROLLING_WINDOWS_HOURS:
            group[f"demand_roll_{hours}h_mean_mw"] = demand.rolling(hours, min_periods=max(1, min(hours, 3))).mean()
        for hours in RECENT_TREND_HOURS:
            group[f"demand_change_{hours}h_mw"] = demand - demand.shift(hours)
        group["demand_roll_3h_vs_24h_mw"] = group["demand_roll_3h_mean_mw"] - group["demand_roll_24h_mean_mw"]
        parts.append(group.reset_index())
    return pd.concat(parts, ignore_index=True, sort=False)


def _add_national_recent_features(hourly: pd.DataFrame) -> pd.DataFrame:
    result = hourly.copy()
    national = result[result["geographic_scope"].eq("national") & result["region"].eq("France")].copy()
    if national.empty:
        for column in (
            "national_total_production_mw",
            "national_total_production_lag_1h_mw",
            "national_total_production_roll_24h_mean_mw",
            "national_net_imports_mw",
            "national_net_imports_lag_1h_mw",
            "national_net_imports_roll_24h_mean_mw",
        ):
            result[column] = np.nan
        return result
    national = national.sort_values("timestamp", kind="stable").set_index("timestamp")
    national_features = pd.DataFrame(index=national.index)
    production = pd.to_numeric(national.get("total_production_mw"), errors="coerce")
    exchange = pd.to_numeric(national.get("net_imports_mw"), errors="coerce")
    national_features["national_total_production_mw"] = production
    national_features["national_total_production_lag_1h_mw"] = production.shift(1)
    national_features["national_total_production_roll_24h_mean_mw"] = production.rolling(24, min_periods=3).mean()
    national_features["national_net_imports_mw"] = exchange
    national_features["national_net_imports_lag_1h_mw"] = exchange.shift(1)
    national_features["national_net_imports_roll_24h_mean_mw"] = exchange.rolling(24, min_periods=3).mean()
    result = result.merge(national_features.reset_index(), on="timestamp", how="left", validate="many_to_one")
    return result


def _finalize_missing_indicators(hourly: pd.DataFrame) -> pd.DataFrame:
    result = hourly.copy()
    result["demand_missing"] = pd.to_numeric(result.get("consumption_mw"), errors="coerce").isna().astype(int)
    result["generation_missing"] = pd.to_numeric(result.get("total_production_mw"), errors="coerce").isna().astype(int)
    result["exchange_missing"] = pd.to_numeric(result.get("net_imports_mw"), errors="coerce").isna().astype(int)
    result["energy_incomplete_hour"] = pd.to_numeric(result.get("energy_coverage_ratio"), errors="coerce").lt(1).fillna(True).astype(int)
    return result


def _usual_baseline_for_row(row: Mapping[str, Any], history: pd.DataFrame, config: BaselineConfig) -> dict[str, Any]:
    if history.empty:
        return _empty_baseline("no_history")
    origin = pd.Timestamp(row["origin_timestamp"])
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    target_hour_value = row.get("target_hour_of_day", row.get("hour_of_day", -1))
    if pd.isna(target_hour_value):
        target_timestamp = pd.Timestamp(row.get("target_timestamp", origin))
        target_timestamp = (
            target_timestamp.tz_localize("UTC")
            if target_timestamp.tzinfo is None
            else target_timestamp.tz_convert("UTC")
        )
        target_hour_value = (target_timestamp - pd.Timedelta(hours=1)).tz_convert(DEFAULT_TIMEZONE).hour
    target_hour = int(target_hour_value)
    target_weekday_type = str(row.get("target_weekday_type", row.get("weekday_type", "")))
    target_season = str(row.get("target_season", row.get("season", "")))
    target_holiday_type = str(row.get("target_holiday_type", row.get("holiday_type", "")))

    candidates = history[history["timestamp"].le(origin)].copy()
    if config.max_history_days is not None:
        candidates = candidates[candidates["timestamp"].ge(origin - pd.Timedelta(days=int(config.max_history_days)))]
    if candidates.empty:
        return _empty_baseline("no_history_before_origin")

    rules = (
        (
            1,
            "same_hour_weekday_type_season_holiday_type",
            candidates[
                candidates["hour_of_day"].eq(target_hour)
                & candidates["weekday_type"].eq(target_weekday_type)
                & candidates["season"].eq(target_season)
                & candidates["holiday_type"].eq(target_holiday_type)
            ],
        ),
        (
            2,
            "same_hour_weekday_type_season",
            candidates[
                candidates["hour_of_day"].eq(target_hour)
                & candidates["weekday_type"].eq(target_weekday_type)
                & candidates["season"].eq(target_season)
            ],
        ),
        (
            3,
            "same_hour_weekday_type",
            candidates[
                candidates["hour_of_day"].eq(target_hour)
                & candidates["weekday_type"].eq(target_weekday_type)
            ],
        ),
        (4, "same_hour", candidates[candidates["hour_of_day"].eq(target_hour)]),
    )
    for level, method, sample in rules:
        sample = sample.dropna(subset=["consumption_mw"])
        if len(sample) >= config.min_samples:
            return _baseline_from_sample(sample, level=level, method=method)

    recent_start = origin - pd.Timedelta(days=int(config.recent_days))
    recent = candidates[candidates["timestamp"].ge(recent_start)]
    seasonal = recent[recent["season"].eq(target_season)].dropna(subset=["consumption_mw"])
    sample = seasonal if len(seasonal) >= 1 else recent.dropna(subset=["consumption_mw"])
    if sample.empty:
        sample = candidates.tail(max(1, config.min_samples)).dropna(subset=["consumption_mw"])
    if sample.empty:
        return _empty_baseline("no_non_null_history")
    return _baseline_from_sample(sample, level=5, method="recent_rolling_seasonal")


def _baseline_from_sample(sample: pd.DataFrame, *, level: int, method: str) -> dict[str, Any]:
    values = pd.to_numeric(sample["consumption_mw"], errors="coerce").dropna()
    if values.empty:
        return _empty_baseline("no_non_null_history")
    median = float(values.median())
    q10 = float(values.quantile(0.10))
    q90 = float(values.quantile(0.90))
    return {
        "usual_demand_mw": median,
        "usual_demand_p10_mw": q10,
        "usual_demand_p90_mw": q90,
        "usual_demand_method": method,
        "usual_demand_fallback_level": int(level),
        "usual_demand_sample_count": int(len(values)),
        "usual_demand_source_timestamp_min": sample["timestamp"].min(),
        "usual_demand_source_timestamp_max": sample["timestamp"].max(),
    }


def _empty_baseline(reason: str) -> dict[str, Any]:
    return {
        "usual_demand_mw": np.nan,
        "usual_demand_p10_mw": np.nan,
        "usual_demand_p90_mw": np.nan,
        "usual_demand_method": reason,
        "usual_demand_fallback_level": 99,
        "usual_demand_sample_count": 0,
        "usual_demand_source_timestamp_min": pd.NaT,
        "usual_demand_source_timestamp_max": pd.NaT,
    }


def _metric_row(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return _empty_metric_row()
    actual = pd.to_numeric(group["target_mw"], errors="coerce")
    predicted = pd.to_numeric(group["usual_demand_mw"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    if not valid.any():
        return _empty_metric_row(origin_count=len(group))
    abs_error = (actual[valid] - predicted[valid]).abs()
    denominator = actual[valid].abs().sum()
    return {
        "mae_gw": float(abs_error.mean() / 1000.0),
        "wape": float(abs_error.sum() / denominator) if denominator else None,
        "sample_count": int(valid.sum()),
        "origin_count": int(len(group)),
        "mean_baseline_sample_count": _mean_or_none(group.loc[valid, "usual_demand_sample_count"]),
        "median_fallback_level": _median_or_none(group.loc[valid, "usual_demand_fallback_level"]),
    }


def _empty_metric_row(origin_count: int = 0) -> dict[str, Any]:
    return {
        "mae_gw": None,
        "wape": None,
        "sample_count": 0,
        "origin_count": int(origin_count),
        "mean_baseline_sample_count": None,
        "median_fallback_level": None,
    }


def _group_metric_rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for key, group in frame.groupby(columns, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = {column: _json_scalar(value) for column, value in zip(columns, values)}
        row.update(_metric_row(group))
        rows.append(row)
    return rows


def _weak_data_periods(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    frame = predictions.copy()
    frame["origin_timestamp"] = pd.to_datetime(frame["origin_timestamp"], utc=True, errors="coerce")
    weak = frame[
        pd.to_numeric(frame.get("usual_demand_sample_count"), errors="coerce").fillna(0).lt(5)
        | pd.to_numeric(frame.get("usual_demand_fallback_level"), errors="coerce").fillna(99).ge(4)
    ]
    if weak.empty:
        return []
    rows = []
    for (scope, region), group in weak.groupby(["geographic_scope", "region"], sort=True):
        rows.append(
            {
                "geographic_scope": str(scope),
                "region": str(region),
                "start_utc": _iso(group["origin_timestamp"].min()),
                "end_utc": _iso(group["origin_timestamp"].max()),
                "row_count": int(len(group)),
                "reason": "sparse comparable history or high fallback level",
                "max_fallback_level": int(pd.to_numeric(group["usual_demand_fallback_level"], errors="coerce").max()),
            }
        )
    return rows


def _weak_hourly_periods(hourly: pd.DataFrame) -> list[dict[str, Any]]:
    weak = hourly[
        pd.to_numeric(hourly.get("energy_coverage_ratio"), errors="coerce").lt(1).fillna(True)
        | pd.to_numeric(hourly.get("weather_missing"), errors="coerce").fillna(1).eq(1)
    ]
    rows = []
    for (scope, region), group in weak.groupby(["geographic_scope", "region"], sort=True):
        rows.append(
            {
                "geographic_scope": str(scope),
                "region": str(region),
                "start_utc": _iso(group["timestamp"].min()),
                "end_utc": _iso(group["timestamp"].max()),
                "row_count": int(len(group)),
                "reason": "incomplete energy hour or missing/stale weather",
            }
        )
    return rows


def _public_holiday_dates(records: pd.DataFrame, timestamps: pd.Series, timezone: str) -> set[Any]:
    dates: set[Any] = set()
    holiday_rows = records[
        records.get("source_name", pd.Series("", index=records.index)).astype(str).eq(PUBLIC_HOLIDAY_SOURCE)
        | records.get("is_public_holiday", pd.Series(0, index=records.index)).fillna(0).astype(int).eq(1)
    ].copy()
    if not holiday_rows.empty:
        holiday_rows["event_time"] = pd.to_datetime(holiday_rows["event_time"], utc=True, errors="coerce")
        dates.update(holiday_rows["event_time"].dt.tz_convert(timezone).dt.date.dropna().tolist())
    if dates:
        return dates
    try:
        import holidays

        years = sorted(
            set(
                (pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)) - pd.Timedelta(hours=1))
                .tz_convert(timezone)
                .year
            )
        )
        dates.update(holidays.France(years=years).keys())
    except Exception:
        pass
    return dates


def _school_calendar(records: pd.DataFrame) -> pd.DataFrame | None:
    rows = records[
        records.get("source_name", pd.Series("", index=records.index)).astype(str).eq(SCHOOL_HOLIDAY_SOURCE)
        | records.get("is_school_holiday", pd.Series(0, index=records.index)).fillna(0).astype(int).eq(1)
    ].copy()
    if rows.empty or not {"start_date", "end_date", "zone"}.issubset(rows.columns):
        return None
    return rows[["start_date", "end_date", "zone", *[column for column in ("description", "source") if column in rows]]]


def _availability_time(frame: pd.DataFrame) -> pd.Series:
    event = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    published = pd.to_datetime(frame.get("published_at"), utc=True, errors="coerce")
    plausible = published.notna() & published.le(event + MAX_PLAUSIBLE_PUBLICATION_LAG)
    return published.where(plausible, event)


def _expected_observations_per_hour(group: pd.DataFrame) -> int:
    source_names = set(group["source_name"].astype(str))
    if "odre_eco2mix_national_history" in source_names:
        return 2
    if source_names & {"odre_eco2mix_national", "odre_eco2mix_regional"}:
        return 4
    times = pd.DatetimeIndex(group["event_time"].dropna().drop_duplicates().sort_values())
    if len(times) <= 1:
        return max(1, len(group))
    diffs = times.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return max(1, len(group))
    cadence = diffs.mode().iloc[0]
    if cadence <= pd.Timedelta(0):
        return max(1, len(group))
    return max(1, int(pd.Timedelta(hours=1) / cadence))


def _energy_mean_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in POWER_MW_COLUMNS if column in frame]
    if "co2_intensity_g_per_kwh" in frame:
        columns.append("co2_intensity_g_per_kwh")
    return columns


def _energy_sum_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if _is_energy_sum_column(column) and column not in POWER_MW_COLUMNS
    ]


def _is_energy_sum_column(column: str) -> bool:
    lowered = str(column).lower()
    return lowered.endswith(ENERGY_SUM_SUFFIXES)


def _quality_aggregate(group: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    quality = group.get("quality_status", pd.Series("", index=group.index)).astype(str).str.lower()
    fallback = group.get("fallback_status", pd.Series("", index=group.index)).astype(str).str.lower()
    return {
        f"{prefix}_quality_ok_fraction": float(quality.isin(GOOD_QUALITY_VALUES).mean()) if len(group) else np.nan,
        f"{prefix}_schema_failure_count": int(quality.isin({"schema_failure", "source_failure"}).sum()),
        f"{prefix}_fallback_record_count": int((~fallback.isin(NO_FALLBACK_VALUES)).sum()),
    }


def _season_label(month: int | float | pd.NA) -> str:
    if pd.isna(month):
        return "unknown"
    value = int(month)
    if value in {12, 1, 2}:
        return "winter"
    if value in {3, 4, 5}:
        return "spring"
    if value in {6, 7, 8}:
        return "summer"
    return "autumn"


def _season_code(month: int | float | pd.NA) -> int:
    return {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}.get(_season_label(month), -1)


def _holiday_type(is_public_holiday: Any, is_school_holiday: Any) -> str:
    public = int(is_public_holiday or 0) == 1
    school = int(is_school_holiday or 0) == 1
    if public and school:
        return "public_and_school_holiday"
    if public:
        return "public_holiday"
    if school:
        return "school_holiday"
    return "normal"


def _missing_hour_count(times: pd.DatetimeIndex) -> int:
    if len(times) <= 1:
        return 0
    diffs = times.sort_values().to_series().diff().dropna()
    return int(sum(max(int(delta / pd.Timedelta(hours=1)) - 1, 0) for delta in diffs))


def _is_feature_column(column: str) -> bool:
    excluded = {
        "target_mw",
        "target_timestamp",
        "target_observation_available_at",
        "forecast_origin",
        "usual_demand_mw",
        "usual_demand_p10_mw",
        "usual_demand_p90_mw",
        "actual_above_usual_percent",
    }
    if column in excluded:
        return False
    prefixes = ("origin_", "demand_", "weather_", "heating_", "cooling_", "national_", "source_", "energy_")
    exact = {
        "horizon_hours",
        "hour_of_day",
        "weekday",
        "weekday_type",
        "is_weekend",
        "month",
        "season",
        "season_code",
        "is_public_holiday",
        "holiday_type",
        "school_holiday_zone_a",
        "school_holiday_zone_b",
        "school_holiday_zone_c",
        "school_holiday_any_zone",
        "school_holiday_all_zones",
        "utc_offset_hours",
        "is_dst",
        "demand_missing",
        "generation_missing",
        "exchange_missing",
        "energy_incomplete_hour",
        "consumption_mw",
        "total_production_mw",
        "net_imports_mw",
    }
    return column.startswith(prefixes) or column.startswith("target_") or column in exact


def _manifest_row(column: str) -> dict[str, Any]:
    return {
        "name": column,
        "description": _feature_description(column),
        "unit": _feature_unit(column),
        "leakage_rule": _feature_leakage_rule(column),
    }


def _feature_description(column: str) -> str:
    if column == "horizon_hours":
        return "Forecast horizon from origin to target."
    if column == "consumption_mw":
        return "Hourly mean observed demand for the completed origin hour."
    if column.startswith("demand_lag_"):
        return "Demand from the same geography at the stated lag before the origin."
    if column.startswith("demand_roll_"):
        return "Rolling demand statistic computed from completed hourly demand at or before origin."
    if column.startswith("demand_change_"):
        return "Recent demand difference between origin demand and the lagged demand value."
    if column.startswith("weather_"):
        return "Open-Meteo weather feature joined backward to the forecast origin."
    if column.startswith(("heating_", "cooling_")):
        return "Temperature degree variable derived from origin weather."
    if column.startswith("target_"):
        return "Deterministic target calendar attribute known before the forecast origin."
    if column.startswith("national_"):
        return "Recent national generation or physical exchange context available at origin."
    if "holiday" in column:
        return "French public or school holiday calendar feature."
    if "quality" in column or "missing" in column or "coverage" in column or "fallback" in column:
        return "Source-quality, coverage, or missing-data indicator."
    return column.replace("_", " ").capitalize()


def _feature_unit(column: str) -> str:
    lowered = column.lower()
    if lowered.endswith("_mw") or "_mw_" in lowered:
        return "MW"
    if lowered.endswith("_gw"):
        return "GW"
    if lowered.endswith("_c") or "degree_c" in lowered:
        return "deg C"
    if lowered.endswith("_kmh"):
        return "km/h"
    if lowered.endswith("_pct") or lowered.endswith("_percent") or "fraction" in lowered or "ratio" in lowered:
        return "percent_or_fraction"
    if lowered.endswith("_wm2"):
        return "W/m2"
    if "timestamp" in lowered or lowered.endswith("_at"):
        return "UTC timestamp"
    if lowered.startswith("is_") or lowered.endswith("_missing") or "holiday" in lowered or "weekend" in lowered:
        return "boolean"
    if "count" in lowered or "hour" in lowered or "weekday" in lowered or "month" in lowered or "horizon" in lowered:
        return "count_or_index"
    return "category"


def _feature_leakage_rule(column: str) -> str:
    if column.startswith("target_"):
        return "Deterministic calendar only; no target measurement is used."
    if column.startswith("weather_"):
        return "Weather event and availability timestamps must be at or before origin."
    if column.startswith("demand_") or column == "consumption_mw":
        return "Demand source timestamps are at or before origin within the same geography."
    if column.startswith("national_"):
        return "National context is joined at the origin timestamp only."
    return "Computed from public records observable at or before origin."


def _stable_column_order(frame: pd.DataFrame) -> pd.DataFrame:
    priority = [
        "origin_timestamp",
        "forecast_origin",
        "target_timestamp",
        "horizon_hours",
        "timestamp",
        "geographic_scope",
        "region",
        "target_mw",
        "usual_demand_mw",
        "usual_demand_method",
        "usual_demand_sample_count",
        "usual_demand_fallback_level",
    ]
    ordered = [column for column in priority if column in frame]
    ordered.extend(sorted(column for column in frame.columns if column not in ordered))
    return frame[ordered]


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = pd.to_datetime(serializable[column], utc=True, errors="coerce").map(_iso)
    serializable = serializable.replace({np.nan: None, pd.NaT: None})
    return json.loads(serializable.to_json(orient="records", double_precision=10))


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return _iso(value)
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _mean_or_nan(values: Any) -> float:
    series = pd.to_numeric(values, errors="coerce")
    return float(series.mean()) if series.notna().any() else np.nan


def _mean_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce")
    return float(series.mean()) if series.notna().any() else None


def _median_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce")
    return float(series.median()) if series.notna().any() else None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
