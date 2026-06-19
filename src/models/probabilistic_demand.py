"""Production-oriented probabilistic national demand forecasting.

The forecast structure is deliberately conservative:

    demand forecast = transparent usual-demand baseline + ML residual correction

The usual-demand baseline remains the production fallback.  Candidate residual
models are promoted only when rolling, chronological validation shows an
overall edge and an edge in a majority of validation periods.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.models.usual_demand import (
    BaselineConfig,
    build_baseline_training_dataset,
    compute_usual_demand_baselines,
    validate_no_future_observations,
)


MODEL_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
MODEL_FAMILY = "usual_demand_plus_residual_quantile"
LIGHTGBM_MODEL_KIND = "lightgbm.LGBMRegressor.quantile"
SKLEARN_FALLBACK_MODEL_KIND = "sklearn.HistGradientBoostingRegressor.quantile"
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_QUANTILES = (0.10, 0.50, 0.90)
DEFAULT_HORIZONS_HOURS = tuple(range(1, 49))
MODEL_FILENAME = "demand_residual_quantile_model.pkl"
REGISTRY_FILENAME = "artifact_manifest.json"
MODEL_CARD_FILENAME = "model_card.json"
VALIDATION_PREDICTIONS_FILENAME = "validation_predictions.parquet"
TARGET_COLUMN = "target_mw"
RESIDUAL_COLUMN = "residual_mw"


@dataclass(frozen=True)
class ResidualQuantileConfig:
    """Configuration stored with every candidate model artifact."""

    horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS_HOURS
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    timezone: str = DEFAULT_TIMEZONE
    random_seed: int = 42
    validation_folds: int = 4
    validation_fraction: float = 0.30
    min_train_samples: int = 168
    min_validation_samples: int = 48
    champion_min_relative_improvement: float = 0.0
    champion_majority_ratio: float = 0.50
    baseline_min_samples: int = 5
    baseline_recent_days: int = 28
    baseline_max_history_days: int | None = None
    preferred_engine: str = "lightgbm"
    allow_sklearn_fallback: bool = True
    max_iter: int = 80
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31


@dataclass(frozen=True)
class ForecastRun:
    """Inference response returned by the lightweight service."""

    origin: str
    generated_at: str
    horizon_hours: int
    route: str
    model_version: str | None
    fallback_reason: str | None
    points: list[dict[str, Any]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.points)


def utc_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def generated_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dataframe_digest(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce").map(utc_iso)
    hashed = pd.util.hash_pandas_object(normalized.reset_index(drop=True), index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_residual_training_frame(
    supervised: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    config: ResidualQuantileConfig | None = None,
) -> pd.DataFrame:
    """Attach baselines and residual labels to origin-safe supervised rows."""

    config = config or ResidualQuantileConfig()
    if supervised.empty:
        raise ValueError("No supervised rows are available for residual training.")
    frame = supervised.copy()
    for column in ("origin_timestamp", "target_timestamp"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if "usual_demand_mw" not in frame:
        frame = compute_usual_demand_baselines(
            frame,
            hourly,
            config=BaselineConfig(
                min_samples=config.baseline_min_samples,
                recent_days=config.baseline_recent_days,
                max_history_days=config.baseline_max_history_days,
            ),
        )
    frame = _national_rows(frame)
    frame = add_seasonal_naive_baseline(frame, hourly)
    frame = add_rte_public_forecast_comparison(frame, hourly, timezone_name=config.timezone)
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame["usual_demand_mw"] = pd.to_numeric(frame["usual_demand_mw"], errors="coerce")
    frame[RESIDUAL_COLUMN] = frame[TARGET_COLUMN] - frame["usual_demand_mw"]
    validate_no_target_leakage(frame)
    valid = frame[TARGET_COLUMN].notna() & frame["usual_demand_mw"].notna()
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No rows have both target demand and usual-demand baseline values.")
    return frame.sort_values(["origin_timestamp", "horizon_hours", "target_timestamp"], kind="stable").reset_index(drop=True)


def add_seasonal_naive_baseline(frame: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Add previous-week same-target-hour baseline without using future values."""

    result = frame.copy()
    history = _national_rows(hourly)
    if history.empty or "timestamp" not in history or "consumption_mw" not in history:
        result["seasonal_naive_mw"] = np.nan
        return result
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    values = (
        history.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")["consumption_mw"]
    )
    target = pd.DatetimeIndex(pd.to_datetime(result["target_timestamp"], utc=True, errors="coerce"))
    origin = pd.DatetimeIndex(pd.to_datetime(result["origin_timestamp"], utc=True, errors="coerce"))
    source_times = target - pd.Timedelta(days=7)
    valid_source = source_times <= origin
    result["seasonal_naive_source_timestamp"] = source_times
    result["seasonal_naive_mw"] = values.reindex(source_times).to_numpy()
    result.loc[~valid_source, "seasonal_naive_mw"] = np.nan
    if (source_times[valid_source] > origin[valid_source]).any():
        raise ValueError("seasonal_naive_mw would use observations after the forecast origin.")
    return result


def add_rte_public_forecast_comparison(
    frame: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Attach RTE public demand forecast when J/J+1 columns are present.

    The value is used for comparison reporting only, not as a residual-model
    feature.  It is left missing when the required public forecast column is not
    available for the target day class.
    """

    result = frame.copy()
    history = _national_rows(hourly)
    forecast_columns = [column for column in ("rte_forecast_j_mw", "rte_forecast_j1_mw") if column in history]
    if history.empty or not forecast_columns:
        result["rte_public_forecast_mw"] = np.nan
        return result
    source = (
        history[["timestamp", *forecast_columns]]
        .assign(timestamp=lambda data: pd.to_datetime(data["timestamp"], utc=True, errors="coerce"))
        .dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )
    target = pd.DatetimeIndex(pd.to_datetime(result["target_timestamp"], utc=True, errors="coerce"))
    origin = pd.DatetimeIndex(pd.to_datetime(result["origin_timestamp"], utc=True, errors="coerce"))
    values = pd.Series(np.nan, index=result.index, dtype=float)
    target_local_dates = pd.Series(target.tz_convert(timezone_name).date, index=result.index)
    origin_local_dates = pd.Series(origin.tz_convert(timezone_name).date, index=result.index)
    same_day = target_local_dates.eq(origin_local_dates)
    next_day = target_local_dates.eq(origin_local_dates + pd.Timedelta(days=1))
    if "rte_forecast_j_mw" in source:
        j_values = pd.Series(
            pd.to_numeric(source["rte_forecast_j_mw"].reindex(target), errors="coerce").to_numpy(),
            index=result.index,
        )
        values.loc[same_day] = j_values.loc[same_day]
    if "rte_forecast_j1_mw" in source:
        j1_values = pd.Series(
            pd.to_numeric(source["rte_forecast_j1_mw"].reindex(target), errors="coerce").to_numpy(),
            index=result.index,
        )
        values.loc[next_day] = j1_values.loc[next_day]
    result["rte_public_forecast_mw"] = values.to_numpy()
    return result


def validate_no_target_leakage(frame: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> None:
    """Reject obvious target leakage in rows or selected feature columns."""

    validate_no_future_observations(frame)
    columns = set(feature_columns or [])
    banned_exact = {
        TARGET_COLUMN,
        RESIDUAL_COLUMN,
        "actual_above_usual_percent",
        "target_observation_available_at",
        "seasonal_naive_mw",
        "rte_public_forecast_mw",
    }
    banned = sorted(column for column in columns if column in banned_exact)
    banned.extend(sorted(column for column in columns if column.endswith("_source_timestamp")))
    banned.extend(sorted(column for column in columns if "timestamp" in column or column.endswith("_available_at")))
    if banned:
        raise ValueError(f"Feature columns contain leakage-prone fields: {sorted(set(banned))}")
    origin = pd.to_datetime(frame.get("origin_timestamp"), utc=True, errors="coerce")
    for column in [name for name in frame.columns if name.endswith("_source_timestamp") or name.endswith("_available_at")]:
        if column in {"target_observation_available_at"}:
            continue
        source_time = pd.to_datetime(frame[column], utc=True, errors="coerce")
        invalid = source_time.notna() & origin.notna() & source_time.gt(origin)
        if invalid.any():
            raise ValueError(f"{column} contains values after the forecast origin.")


def train_residual_quantile_candidate(
    supervised: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    feature_manifest: Mapping[str, Any] | None = None,
    config: ResidualQuantileConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train and evaluate a residual quantile candidate with rolling validation."""

    config = config or ResidualQuantileConfig()
    _validate_config(config)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    rows = prepare_residual_training_frame(supervised, hourly, config=config)
    feature_columns = select_model_feature_columns(rows)
    validate_no_target_leakage(rows, feature_columns)
    splits = rolling_validation_splits(rows, config=config)
    if not splits:
        metrics = empty_metrics("no rolling validation periods met the minimum sample requirements")
        artifact = _artifact_shell(
            rows=rows,
            hourly=hourly,
            feature_columns=feature_columns,
            feature_manifest=feature_manifest,
            config=config,
            metrics=metrics,
            status="rejected",
            rejection_reason=metrics["rejection_reason"],
            models={},
            imputation_values={},
            model_kind=None,
        )
        return artifact, pd.DataFrame()

    prediction_parts: list[pd.DataFrame] = []
    model_kind = None
    for fold_index, (train_rows, validation_rows) in enumerate(splits, start=1):
        imputation_values = feature_imputation_values(train_rows, feature_columns)
        models, model_kind = fit_quantile_models(
            train_rows,
            feature_columns,
            imputation_values,
            config=config,
        )
        fold_predictions = predict_with_models(
            validation_rows,
            models=models,
            feature_columns=feature_columns,
            imputation_values=imputation_values,
        )
        fold_predictions["validation_fold"] = fold_index
        prediction_parts.append(fold_predictions)

    validation_predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["origin_timestamp", "horizon_hours"], kind="stable", ignore_index=True
    )
    metrics = evaluate_candidate_predictions(validation_predictions, config=config)
    decision = champion_decision(metrics, config=config)
    status = "champion" if decision["accepted"] else "rejected"
    rejection_reason = None if decision["accepted"] else decision["reason"]

    final_imputation = feature_imputation_values(rows, feature_columns)
    final_models, model_kind = fit_quantile_models(
        rows,
        feature_columns,
        final_imputation,
        config=config,
    )
    artifact = _artifact_shell(
        rows=rows,
        hourly=hourly,
        feature_columns=feature_columns,
        feature_manifest=feature_manifest,
        config=config,
        metrics={**metrics, "champion_decision": decision},
        status=status,
        rejection_reason=rejection_reason,
        models=final_models,
        imputation_values=final_imputation,
        model_kind=model_kind,
    )
    artifact["model_card"] = build_model_card(artifact)
    return artifact, validation_predictions


def select_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric, origin-safe features for residual correction."""

    banned_exact = {
        TARGET_COLUMN,
        RESIDUAL_COLUMN,
        "actual_above_usual_percent",
        "target_observation_available_at",
        "seasonal_naive_mw",
        "seasonal_naive_source_timestamp",
        "rte_public_forecast_mw",
        "rte_forecast_j_mw",
        "rte_forecast_j1_mw",
    }
    banned_fragments = ("timestamp", "available_at", "source_event_time")
    selected: list[str] = []
    for column in frame.columns:
        if column in banned_exact or any(fragment in column for fragment in banned_fragments):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            selected.append(column)
    if "usual_demand_mw" not in selected:
        raise ValueError("The usual-demand baseline must be present as a model feature.")
    if "horizon_hours" not in selected:
        raise ValueError("horizon_hours must be present as a model feature.")
    return selected


def rolling_validation_splits(
    rows: pd.DataFrame,
    *,
    config: ResidualQuantileConfig,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window validation over ordered forecast origins."""

    ordered = rows.sort_values(["origin_timestamp", "horizon_hours"], kind="stable").reset_index(drop=True)
    origins = pd.Index(pd.to_datetime(ordered["origin_timestamp"], utc=True).drop_duplicates().sort_values())
    if len(origins) < 3:
        return []
    fold_count = max(1, int(config.validation_folds))
    validation_origin_count = max(1, int(math.ceil(len(origins) * config.validation_fraction / fold_count)))
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(fold_count, 0, -1):
        valid_end = len(origins) - (fold - 1) * validation_origin_count
        valid_start = max(0, valid_end - validation_origin_count)
        if valid_start <= 0 or valid_start >= valid_end:
            continue
        train_origins = set(origins[:valid_start])
        valid_origins = set(origins[valid_start:valid_end])
        train = ordered[ordered["origin_timestamp"].isin(train_origins)].copy()
        valid = ordered[ordered["origin_timestamp"].isin(valid_origins)].copy()
        if len(train) < config.min_train_samples or len(valid) < config.min_validation_samples:
            continue
        splits.append((train, valid))
    return splits


def feature_imputation_values(frame: pd.DataFrame, feature_columns: Iterable[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in feature_columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)
        clean = series[np.isfinite(series)]
        median = clean.median() if not clean.empty else np.nan
        values[column] = float(median) if pd.notna(median) and np.isfinite(float(median)) else 0.0
    return values


def fit_quantile_models(
    train_rows: pd.DataFrame,
    feature_columns: list[str],
    imputation_values: Mapping[str, float],
    *,
    config: ResidualQuantileConfig,
) -> tuple[dict[str, Any], str]:
    models: dict[str, Any] = {}
    model_kind = None
    x_train = feature_matrix(train_rows, feature_columns, imputation_values)
    y_train = pd.to_numeric(train_rows[RESIDUAL_COLUMN], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_train)
    if not valid.any():
        raise ValueError("Training residuals are all missing.")
    x_train = x_train.loc[valid]
    y_train = y_train[valid]
    for quantile in config.quantiles:
        key = quantile_key(quantile)
        model, model_kind = _fit_single_quantile_model(x_train, y_train, quantile, config=config)
        models[key] = model
    return models, str(model_kind)


def _fit_single_quantile_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    quantile: float,
    *,
    config: ResidualQuantileConfig,
) -> tuple[Any, str]:
    if config.preferred_engine.lower() == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="quantile",
                alpha=float(quantile),
                n_estimators=int(config.max_iter),
                learning_rate=float(config.learning_rate),
                num_leaves=int(config.max_leaf_nodes),
                min_child_samples=20,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=int(config.random_seed),
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            model.fit(x_train, y_train)
            return model, LIGHTGBM_MODEL_KIND
        except ImportError:
            if not config.allow_sklearn_fallback:
                raise RuntimeError("LightGBM is required but is not installed.")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("Training requires LightGBM or scikit-learn.") from exc
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        learning_rate=float(config.learning_rate),
        max_iter=int(config.max_iter),
        max_leaf_nodes=int(config.max_leaf_nodes),
        l2_regularization=0.05,
        early_stopping=False,
        random_state=int(config.random_seed),
    )
    model.fit(x_train, y_train)
    return model, SKLEARN_FALLBACK_MODEL_KIND


def predict_with_models(
    rows: pd.DataFrame,
    *,
    models: Mapping[str, Any],
    feature_columns: list[str],
    imputation_values: Mapping[str, float],
) -> pd.DataFrame:
    result = rows.copy()
    x_values = feature_matrix(result, feature_columns, imputation_values)
    for quantile in DEFAULT_QUANTILES:
        key = quantile_key(quantile)
        if key not in models:
            raise ValueError(f"Model artifact is missing quantile {key}.")
        residual = np.asarray(models[key].predict(x_values), dtype=float)
        result[f"residual_{key}_mw"] = residual
        result[key] = pd.to_numeric(result["usual_demand_mw"], errors="coerce").to_numpy(dtype=float) + residual
    result = correct_quantile_crossing(result)
    return result


def feature_matrix(
    rows: pd.DataFrame,
    feature_columns: Iterable[str],
    imputation_values: Mapping[str, float],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=rows.index)
    for column in feature_columns:
        if column in rows:
            matrix[column] = pd.to_numeric(rows[column], errors="coerce")
        else:
            matrix[column] = np.nan
        matrix[column] = matrix[column].fillna(float(imputation_values.get(column, 0.0)))
    return matrix


def correct_quantile_crossing(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required = ["p10", "p50", "p90"]
    if not set(required).issubset(result.columns):
        return result
    raw = result[required].to_numpy(dtype=float)
    crossing_before = (raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2])
    ordered = np.sort(raw, axis=1)
    result["p10_raw"] = raw[:, 0]
    result["p50_raw"] = raw[:, 1]
    result["p90_raw"] = raw[:, 2]
    result[required] = ordered
    result["quantile_crossing_corrected"] = crossing_before
    return result


def evaluate_candidate_predictions(
    predictions: pd.DataFrame,
    *,
    config: ResidualQuantileConfig,
) -> dict[str, Any]:
    if predictions.empty:
        return empty_metrics("no validation predictions")
    frame = predictions.copy()
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    overall = forecast_metric_row(frame, "p10", "p50", "p90")
    usual = forecast_metric_row(frame, "usual_demand_p10_mw", "usual_demand_mw", "usual_demand_p90_mw")
    seasonal = point_metric_row(frame, "seasonal_naive_mw")
    rte = point_metric_row(frame, "rte_public_forecast_mw")
    strongest_name, strongest = strongest_baseline({"usual_demand": usual, "seasonal_naive": seasonal})
    validation_periods = []
    for fold, group in frame.groupby("validation_fold", sort=True):
        model_fold = point_metric_row(group, "p50")
        usual_fold = point_metric_row(group, "usual_demand_mw")
        seasonal_fold = point_metric_row(group, "seasonal_naive_mw")
        baseline_name, baseline_fold = strongest_baseline({"usual_demand": usual_fold, "seasonal_naive": seasonal_fold})
        model_mae = model_fold.get("mae_gw")
        baseline_mae = baseline_fold.get("mae_gw")
        improved = _finite(model_mae) and _finite(baseline_mae) and float(model_mae) < float(baseline_mae)
        validation_periods.append(
            {
                "fold": int(fold),
                "sample_count": int(len(group)),
                "model_mae_gw": model_mae,
                "strongest_baseline": baseline_name,
                "strongest_baseline_mae_gw": baseline_mae,
                "improved": bool(improved),
            }
        )
    improved_count = sum(1 for row in validation_periods if row["improved"])
    majority_ratio = improved_count / len(validation_periods) if validation_periods else 0.0
    baseline_mae = strongest.get("mae_gw")
    model_mae = overall.get("mae_gw")
    improvement = (
        (float(baseline_mae) - float(model_mae)) / float(baseline_mae)
        if _finite(baseline_mae) and _finite(model_mae) and float(baseline_mae) != 0
        else None
    )
    peak_threshold = frame[TARGET_COLUMN].quantile(0.90)
    peak_rows = frame[frame[TARGET_COLUMN].ge(peak_threshold)].copy()
    by_horizon = [
        {"horizon_hours": int(horizon), **forecast_metric_row(group, "p10", "p50", "p90")}
        for horizon, group in frame.groupby("horizon_hours", sort=True)
    ]
    season_column = "target_season" if "target_season" in frame else "season"
    by_season = [
        {"season": _json_scalar(season), **forecast_metric_row(group, "p10", "p50", "p90")}
        for season, group in frame.groupby(season_column, dropna=False, sort=True)
    ] if season_column in frame else []
    crossing = frame.get("quantile_crossing_corrected", pd.Series(False, index=frame.index)).fillna(False)
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "overall": overall,
        "by_horizon": by_horizon,
        "by_season": by_season,
        "peaks": {
            "definition": "actual demand at or above validation-set p90",
            "threshold_gw": float(peak_threshold / 1000.0) if pd.notna(peak_threshold) else None,
            **forecast_metric_row(peak_rows, "p10", "p50", "p90"),
        },
        "baselines": {
            "usual_demand": usual,
            "seasonal_naive": seasonal,
            "rte_public_forecast": {
                **rte,
                "available": bool(rte.get("sample_count", 0)),
                "note": "RTE public forecast comparison is reported only when forecast columns are available.",
            },
        },
        "baseline_comparison": {
            "strongest_baseline": strongest_name,
            "model_mae_gw": model_mae,
            "strongest_baseline_mae_gw": baseline_mae,
            "relative_improvement": improvement,
        },
        "validation_periods": validation_periods,
        "validation_majority": {
            "improved_period_count": int(improved_count),
            "period_count": int(len(validation_periods)),
            "improved_ratio": float(majority_ratio),
            "required_ratio": float(config.champion_majority_ratio),
        },
        "quantile_crossing": {
            "corrected_count": int(crossing.sum()),
            "corrected_fraction": float(crossing.mean()) if len(crossing) else 0.0,
        },
    }


def forecast_metric_row(frame: pd.DataFrame, p10_col: str, p50_col: str, p90_col: str) -> dict[str, Any]:
    if frame.empty or TARGET_COLUMN not in frame:
        return _empty_forecast_metric()
    actual = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    p10 = pd.to_numeric(frame.get(p10_col), errors="coerce")
    p50 = pd.to_numeric(frame.get(p50_col), errors="coerce")
    p90 = pd.to_numeric(frame.get(p90_col), errors="coerce")
    valid = actual.notna() & p10.notna() & p50.notna() & p90.notna()
    if not valid.any():
        return _empty_forecast_metric(origin_count=len(frame))
    actual = actual[valid]
    p10 = p10[valid]
    p50 = p50[valid]
    p90 = p90[valid]
    abs_error = (actual - p50).abs()
    denominator = actual.abs().sum()
    return {
        "mae_gw": float(abs_error.mean() / 1000.0),
        "wape": float(abs_error.sum() / denominator) if denominator else None,
        "pinball_loss_p10_gw": pinball_loss(actual, p10, 0.10),
        "pinball_loss_p50_gw": pinball_loss(actual, p50, 0.50),
        "pinball_loss_p90_gw": pinball_loss(actual, p90, 0.90),
        "p10_p90_empirical_coverage": float(((actual >= p10) & (actual <= p90)).mean()),
        "interval_width_gw": float((p90 - p10).mean() / 1000.0),
        "sample_count": int(valid.sum()),
        "origin_count": int(len(frame)),
    }


def point_metric_row(frame: pd.DataFrame, prediction_col: str) -> dict[str, Any]:
    if frame.empty or TARGET_COLUMN not in frame or prediction_col not in frame:
        return _empty_point_metric()
    actual = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    predicted = pd.to_numeric(frame[prediction_col], errors="coerce")
    valid = actual.notna() & predicted.notna()
    if not valid.any():
        return _empty_point_metric(origin_count=len(frame))
    abs_error = (actual[valid] - predicted[valid]).abs()
    denominator = actual[valid].abs().sum()
    return {
        "mae_gw": float(abs_error.mean() / 1000.0),
        "wape": float(abs_error.sum() / denominator) if denominator else None,
        "sample_count": int(valid.sum()),
        "origin_count": int(len(frame)),
    }


def pinball_loss(actual: Iterable[float], predicted: Iterable[float], quantile: float) -> float | None:
    y = np.asarray(list(actual), dtype=float)
    yhat = np.asarray(list(predicted), dtype=float)
    valid = np.isfinite(y) & np.isfinite(yhat)
    if not valid.any():
        return None
    residual = y[valid] - yhat[valid]
    losses = np.maximum(quantile * residual, (quantile - 1.0) * residual)
    return float(np.mean(losses) / 1000.0)


def strongest_baseline(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, Mapping[str, Any]]:
    eligible = [
        (name, row)
        for name, row in metrics.items()
        if row.get("sample_count", 0) and _finite(row.get("mae_gw"))
    ]
    if not eligible:
        return None, _empty_point_metric()
    return min(eligible, key=lambda item: float(item[1]["mae_gw"]))


def champion_decision(metrics: Mapping[str, Any], *, config: ResidualQuantileConfig) -> dict[str, Any]:
    comparison = metrics.get("baseline_comparison", {})
    model_mae = comparison.get("model_mae_gw")
    baseline_mae = comparison.get("strongest_baseline_mae_gw")
    improvement = comparison.get("relative_improvement")
    majority = metrics.get("validation_majority", {})
    majority_ratio = majority.get("improved_ratio", 0.0)
    if not (_finite(model_mae) and _finite(baseline_mae)):
        return {"accepted": False, "reason": "model or baseline validation MAE is unavailable"}
    if float(improvement or 0.0) <= float(config.champion_min_relative_improvement):
        return {
            "accepted": False,
            "reason": "candidate did not improve overall p50 MAE versus the strongest baseline",
        }
    if float(majority_ratio) < float(config.champion_majority_ratio):
        return {
            "accepted": False,
            "reason": "candidate did not improve a sufficient majority of rolling validation periods",
        }
    return {"accepted": True, "reason": None}


def build_inference_feature_rows(
    hourly: pd.DataFrame,
    forecast_origin: Any,
    *,
    horizons_hours: Iterable[int] = DEFAULT_HORIZONS_HOURS,
    timezone_name: str = DEFAULT_TIMEZONE,
    baseline_config: BaselineConfig | None = None,
) -> pd.DataFrame:
    """Build 48 hourly national forecast rows and usual-demand fallback."""

    if hourly.empty:
        raise ValueError("Hourly history is required for baseline inference.")
    data = _national_rows(hourly).copy()
    if data.empty:
        raise ValueError("No national France hourly rows are available for inference.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    origin = pd.Timestamp(forecast_origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    available = data[data["timestamp"].le(origin)]
    if available.empty:
        raise ValueError("No hourly observation is available at or before the requested forecast origin.")
    effective_origin = available["timestamp"].max()
    horizons = tuple(sorted({int(value) for value in horizons_hours}))
    supervised = build_baseline_training_dataset(
        data,
        horizons_hours=horizons,
        timezone=timezone_name,
    )
    rows = supervised[
        pd.to_datetime(supervised["origin_timestamp"], utc=True, errors="coerce").eq(effective_origin)
    ].copy()
    if rows.empty:
        raise ValueError("Could not build inference rows for the latest available forecast origin.")
    rows = compute_usual_demand_baselines(
        rows,
        data[data["timestamp"].le(effective_origin)].copy(),
        config=baseline_config or BaselineConfig(),
    )
    rows["requested_forecast_origin"] = origin
    rows["effective_forecast_origin"] = effective_origin
    return rows.sort_values("horizon_hours", kind="stable").reset_index(drop=True)


def forecast_with_artifact(
    artifact: Mapping[str, Any] | None,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return p10/p50/p90 for prepared inference rows.

    Missing trained artifacts, rejected artifacts, incompatible artifacts, or
    per-row prediction failures fall back to the usual-demand baseline.
    """

    if artifact is None or artifact.get("status") != "champion":
        return baseline_forecast_frame(rows, reason="no champion model artifact is available")
    try:
        validate_model_artifact(artifact)
        predicted = predict_with_models(
            rows,
            models=artifact["models"],
            feature_columns=list(artifact["feature_columns"]),
            imputation_values=artifact["feature_imputation_values"],
        )
        predicted["route"] = "validated_model"
        predicted["fallback_reason"] = None
        invalid = predicted[["p10", "p50", "p90"]].isna().any(axis=1)
        if invalid.any():
            fallback = baseline_forecast_frame(predicted.loc[invalid], reason="model prediction missing for row")
            predicted.loc[invalid, ["p10", "p50", "p90", "route", "fallback_reason"]] = fallback[
                ["p10", "p50", "p90", "route", "fallback_reason"]
            ].to_numpy()
        return _forecast_output_columns(predicted, model_version=artifact.get("model_version"))
    except Exception as exc:
        return baseline_forecast_frame(rows, reason=f"model artifact could not be used: {exc}")


def baseline_forecast_frame(rows: pd.DataFrame, *, reason: str) -> pd.DataFrame:
    result = rows.copy()
    result["p10"] = pd.to_numeric(result.get("usual_demand_p10_mw"), errors="coerce")
    result["p50"] = pd.to_numeric(result.get("usual_demand_mw"), errors="coerce")
    result["p90"] = pd.to_numeric(result.get("usual_demand_p90_mw"), errors="coerce")
    result["p10"] = result["p10"].fillna(result["p50"])
    result["p90"] = result["p90"].fillna(result["p50"])
    result = correct_quantile_crossing(result)
    result["route"] = "baseline_fallback"
    result["fallback_reason"] = reason
    return _forecast_output_columns(result, model_version=None)


class DemandForecastService:
    """Small in-process inference boundary used by scripts or the app."""

    def __init__(
        self,
        *,
        artifact: Mapping[str, Any] | None = None,
        artifact_path: str | Path | None = None,
    ) -> None:
        self.artifact = dict(artifact) if artifact is not None else None
        if self.artifact is None and artifact_path is not None and Path(artifact_path).exists():
            self.artifact = load_model_artifact(Path(artifact_path))

    def forecast(
        self,
        forecast_origin: Any,
        hourly: pd.DataFrame,
        *,
        horizons_hours: Iterable[int] = DEFAULT_HORIZONS_HOURS,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> ForecastRun:
        config_payload = (self.artifact or {}).get("train_config", {})
        baseline_config = BaselineConfig(
            min_samples=int(config_payload.get("baseline_min_samples", 5)),
            recent_days=int(config_payload.get("baseline_recent_days", 28)),
            max_history_days=config_payload.get("baseline_max_history_days"),
        )
        rows = build_inference_feature_rows(
            hourly,
            forecast_origin,
            horizons_hours=horizons_hours,
            timezone_name=timezone_name,
            baseline_config=baseline_config,
        )
        forecast = forecast_with_artifact(self.artifact, rows)
        origin = pd.Timestamp(rows["effective_forecast_origin"].iloc[0])
        route = "validated_model" if forecast["route"].eq("validated_model").all() else "baseline_fallback"
        reason = None
        fallback_reasons = forecast["fallback_reason"].dropna().astype(str).unique().tolist()
        if fallback_reasons:
            reason = fallback_reasons[0]
        return ForecastRun(
            origin=utc_iso(origin) or "",
            generated_at=generated_at_utc(),
            horizon_hours=int(len(forecast)),
            route=route,
            model_version=(self.artifact or {}).get("model_version") if route == "validated_model" else None,
            fallback_reason=reason,
            points=_json_records(forecast),
        )


def save_training_artifacts(
    artifact: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / MODEL_FILENAME
    predictions_path = output / VALIDATION_PREDICTIONS_FILENAME
    model_card_path = output / MODEL_CARD_FILENAME
    registry_path = output / REGISTRY_FILENAME

    with model_path.open("wb") as handle:
        pickle.dump(dict(artifact), handle, protocol=pickle.HIGHEST_PROTOCOL)
    validation_predictions.to_parquet(predictions_path, index=False)
    write_json(artifact.get("model_card", build_model_card(artifact)), model_card_path)
    checksums = {
        "model": file_checksum(model_path),
        "validation_predictions": file_checksum(predictions_path),
        "model_card": file_checksum(model_card_path),
    }
    manifest = build_artifact_manifest(
        artifact,
        output_dir=output,
        checksums=checksums,
    )
    write_json(manifest, registry_path)
    return manifest


def build_artifact_manifest(
    artifact: Mapping[str, Any],
    *,
    output_dir: Path,
    checksums: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_version": artifact.get("model_version"),
        "training_period": artifact.get("training_period"),
        "feature_version": artifact.get("feature_version"),
        "dataset_version": artifact.get("dataset_version"),
        "config_version": artifact.get("config_version"),
        "data_cutoff": artifact.get("data_cutoff"),
        "metrics": artifact.get("metrics"),
        "status": artifact.get("status", "candidate"),
        "rejection_reason": artifact.get("rejection_reason"),
        "artifact_checksums": dict(checksums),
        "artifacts": {
            "model": str(output_dir / MODEL_FILENAME),
            "validation_predictions": str(output_dir / VALIDATION_PREDICTIONS_FILENAME),
            "model_card": str(output_dir / MODEL_CARD_FILENAME),
        },
    }


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    validate_model_artifact(artifact)
    return artifact


def validate_model_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("Unsupported demand forecast model schema version.")
    if artifact.get("model_family") != MODEL_FAMILY:
        raise ValueError("Unsupported demand forecast model family.")
    if not artifact.get("feature_columns"):
        raise ValueError("Model artifact is missing feature columns.")
    if artifact.get("status") == "champion":
        models = artifact.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("Champion artifact is missing models.")
        for quantile in DEFAULT_QUANTILES:
            key = quantile_key(quantile)
            if key not in models:
                raise ValueError(f"Champion artifact is missing quantile model {key}.")
    if artifact.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Unsupported feature schema version.")
    if artifact.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema version.")
    if artifact.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported config schema version.")


def build_model_card(artifact: Mapping[str, Any]) -> dict[str, Any]:
    metrics = artifact.get("metrics", {})
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": artifact.get("model_version"),
        "model_family": MODEL_FAMILY,
        "status": artifact.get("status"),
        "intended_use": "48-hour national French electricity-demand forecasting with p10/p50/p90 uncertainty.",
        "forecast_equation": "forecast_demand = usual_demand_baseline + residual_quantile_model",
        "fallback": "The transparent usual-demand baseline is the production fallback.",
        "training_period": artifact.get("training_period"),
        "data_cutoff": artifact.get("data_cutoff"),
        "features": {
            "feature_version": artifact.get("feature_version"),
            "feature_count": len(artifact.get("feature_columns", [])),
            "weather_policy": (
                "Use weather fields available at the forecast origin. "
                "Historical weather forecasts may be included when provenance is origin-safe; "
                "future observed target weather is not used."
            ),
        },
        "validation": {
            "method": "rolling expanding-window chronological validation",
            "metrics": metrics,
        },
        "limitations": [
            "Promotion depends on available public historical demand and weather coverage.",
            "Rejected candidates are stored for audit but are not production champions.",
            "RTE public forecast comparison is present only when required forecast columns exist.",
            "SHAP explanations are generated outside the model artifact for the P50 residual tree model only.",
            "SHAP feature attributions are model explanations, not causal proof.",
        ],
    }


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "overall": _empty_forecast_metric(),
        "by_horizon": [],
        "by_season": [],
        "peaks": _empty_forecast_metric(),
        "baselines": {
            "usual_demand": _empty_forecast_metric(),
            "seasonal_naive": _empty_point_metric(),
            "rte_public_forecast": {**_empty_point_metric(), "available": False},
        },
        "baseline_comparison": {
            "strongest_baseline": None,
            "model_mae_gw": None,
            "strongest_baseline_mae_gw": None,
            "relative_improvement": None,
        },
        "validation_periods": [],
        "validation_majority": {
            "improved_period_count": 0,
            "period_count": 0,
            "improved_ratio": 0.0,
            "required_ratio": None,
        },
        "quantile_crossing": {"corrected_count": 0, "corrected_fraction": 0.0},
        "rejection_reason": reason,
    }


def _artifact_shell(
    *,
    rows: pd.DataFrame,
    hourly: pd.DataFrame,
    feature_columns: list[str],
    feature_manifest: Mapping[str, Any] | None,
    config: ResidualQuantileConfig,
    metrics: Mapping[str, Any],
    status: str,
    rejection_reason: str | None,
    models: Mapping[str, Any],
    imputation_values: Mapping[str, float],
    model_kind: str | None,
) -> dict[str, Any]:
    dataset_version = dataset_digest(rows, hourly)
    config_payload = asdict(config)
    feature_version = feature_digest(feature_columns, feature_manifest)
    model_version = f"demand-rq-v{MODEL_SCHEMA_VERSION}-{dataset_version[:12]}-{feature_version[:12]}"
    training_start = rows["origin_timestamp"].min()
    training_end = rows["origin_timestamp"].max()
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "model_kind": model_kind,
        "model_version": model_version,
        "generated_at": generated_at_utc(),
        "status": status,
        "rejection_reason": rejection_reason,
        "training_period": {
            "start_utc": utc_iso(training_start),
            "end_utc": utc_iso(training_end),
            "target_start_utc": utc_iso(rows["target_timestamp"].min()),
            "target_end_utc": utc_iso(rows["target_timestamp"].max()),
        },
        "data_cutoff": utc_iso(training_end),
        "feature_version": feature_version,
        "dataset_version": dataset_version,
        "config_version": hashlib.sha256(json.dumps(_jsonable(config_payload), sort_keys=True).encode("utf-8")).hexdigest(),
        "train_config": config_payload,
        "quantiles": list(config.quantiles),
        "horizons_hours": list(config.horizons_hours),
        "feature_columns": list(feature_columns),
        "feature_imputation_values": dict(imputation_values),
        "feature_manifest": dict(feature_manifest or {}),
        "models": dict(models),
        "metrics": dict(metrics),
        "determinism": {
            "random_seed": int(config.random_seed),
            "n_jobs": 1,
            "time_ordered_validation": True,
        },
    }
    artifact["model_card"] = build_model_card(artifact)
    return artifact


def dataset_digest(rows: pd.DataFrame, hourly: pd.DataFrame) -> str:
    row_columns = [column for column in ("origin_timestamp", "target_timestamp", "horizon_hours", TARGET_COLUMN, "usual_demand_mw") if column in rows]
    hourly_columns = [column for column in ("timestamp", "geographic_scope", "region", "consumption_mw") if column in hourly]
    payload = dataframe_digest(rows[row_columns]) + dataframe_digest(_national_rows(hourly)[hourly_columns])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_digest(feature_columns: list[str], feature_manifest: Mapping[str, Any] | None) -> str:
    payload = {"feature_columns": feature_columns, "feature_manifest": feature_manifest or {}}
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")).hexdigest()


def quantile_key(quantile: float) -> str:
    return f"p{int(round(float(quantile) * 100)):02d}"


def _forecast_output_columns(frame: pd.DataFrame, *, model_version: str | None) -> pd.DataFrame:
    result = frame.copy()
    result["origin_timestamp"] = pd.to_datetime(result["origin_timestamp"], utc=True, errors="coerce")
    result["target_timestamp"] = pd.to_datetime(result["target_timestamp"], utc=True, errors="coerce")
    if "target_timestamp_local" not in result:
        result["target_timestamp_local"] = result["target_timestamp"].dt.tz_convert(DEFAULT_TIMEZONE).map(
            lambda value: value.isoformat()
        )
    result["model_version"] = model_version
    columns = [
        "origin_timestamp",
        "target_timestamp",
        "target_timestamp_local",
        "horizon_hours",
        "p10",
        "p50",
        "p90",
        "route",
        "fallback_reason",
        "model_version",
        "usual_demand_mw",
        "usual_demand_method",
        "usual_demand_sample_count",
        "usual_demand_fallback_level",
        "quantile_crossing_corrected",
    ]
    return result[[column for column in columns if column in result]].reset_index(drop=True)


def _national_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "geographic_scope" in result:
        result = result[result["geographic_scope"].astype(str).eq("national")]
    if "region" in result:
        result = result[result["region"].astype(str).eq("France")]
    return result


def _validate_config(config: ResidualQuantileConfig) -> None:
    if tuple(config.quantiles) != DEFAULT_QUANTILES:
        raise ValueError("This artifact schema requires p10, p50, and p90 quantiles.")
    if not config.horizons_hours or any(int(value) < 1 or int(value) > 48 for value in config.horizons_hours):
        raise ValueError("horizons_hours must contain whole hours from 1 to 48.")
    if config.validation_folds < 1:
        raise ValueError("validation_folds must be positive.")
    if config.min_train_samples < 1 or config.min_validation_samples < 1:
        raise ValueError("minimum sample counts must be positive.")


def _empty_forecast_metric(origin_count: int = 0) -> dict[str, Any]:
    return {
        "mae_gw": None,
        "wape": None,
        "pinball_loss_p10_gw": None,
        "pinball_loss_p50_gw": None,
        "pinball_loss_p90_gw": None,
        "p10_p90_empirical_coverage": None,
        "interval_width_gw": None,
        "sample_count": 0,
        "origin_count": int(origin_count),
    }


def _empty_point_metric(origin_count: int = 0) -> dict[str, Any]:
    return {
        "mae_gw": None,
        "wape": None,
        "sample_count": 0,
        "origin_count": int(origin_count),
    }


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return list(_jsonable(frame.replace({np.nan: None, pd.NaT: None}).to_dict(orient="records")))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_records(value)
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return utc_iso(value)
    if isinstance(value, datetime):
        return utc_iso(pd.Timestamp(value))
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return utc_iso(value)
    if pd.isna(value):
        return None
    return value
