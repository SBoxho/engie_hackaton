from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from app.formatting import format_carbon, format_gw, format_mw, format_percentage, format_signed_mw, format_timestamp
from app.i18n import t
from app.view_models import ForecastPointView, demand_anomaly_label, pressure_score
from src.contracts.energy_twin import TwinResponse, TwinSnapshot
from src.contracts.status_thresholds import status_label


BEST_WINDOW_OBJECTIVES = {
    "lowest_balance": "Lowest balance context",
    "lowest_carbon": "Lowest carbon",
    "combined": "Best combined window",
}


def best_window_objective_label(objective: str, *, locale: str = "en") -> str:
    key = objective if objective in BEST_WINDOW_OBJECTIVES else "combined"
    return t(f"next48h.objectives.{key}", locale=locale, default=BEST_WINDOW_OBJECTIVES[key])


def local_hour_label(
    timestamp: datetime | pd.Timestamp,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = "en",
) -> str:
    local = _as_utc(timestamp).tz_convert(timezone_name)
    return t(
        "next48h.time.local_hour",
        locale=locale,
        weekday=t(f"next48h.time.weekdays_short.{local.weekday()}", locale=locale, default=f"{local:%a}"),
        day=f"{local.day:02d}",
        month=t(f"next48h.time.months_short.{local.month}", locale=locale, default=f"{local:%b}"),
        time=f"{local:%H:%M}",
    )


def window_label(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = "en",
) -> str:
    start_local = _as_utc(start).tz_convert(timezone_name)
    end_local = _as_utc(end).tz_convert(timezone_name)
    start_weekday = t(f"next48h.time.weekdays_short.{start_local.weekday()}", locale=locale, default=f"{start_local:%a}")
    end_weekday = t(f"next48h.time.weekdays_short.{end_local.weekday()}", locale=locale, default=f"{end_local:%a}")
    if start_local.date() == end_local.date():
        return t(
            "next48h.time.window_same_day",
            locale=locale,
            weekday=start_weekday,
            start_time=f"{start_local:%H:%M}",
            end_time=f"{end_local:%H:%M}",
        )
    return t(
        "next48h.time.window_cross_day",
        locale=locale,
        start_weekday=start_weekday,
        start_time=f"{start_local:%H:%M}",
        end_weekday=end_weekday,
        end_time=f"{end_local:%H:%M}",
    )


def time_delta_text(selected: pd.Timestamp, reference: pd.Timestamp, *, locale: str = "en") -> str:
    hours = int(round((selected - reference).total_seconds() / 3600.0))
    if hours == 0:
        return t("next48h.delta.at_peak", locale=locale)
    if hours > 0:
        return t("next48h.delta.after_peak", locale=locale, hours=hours)
    return t("next48h.delta.before_peak", locale=locale, hours=abs(hours))


@dataclass(frozen=True)
class DemandDriver:
    name: str
    value_mw: float
    detail: str


@dataclass(frozen=True)
class SelectedHourExplanation:
    usual_demand_mw: float
    positive_drivers: list[DemandDriver]
    negative_drivers: list[DemandDriver]
    expected_demand_mw: float
    p10_mw: float
    p90_mw: float
    reconciliation_error_mw: float
    text: str

    @property
    def reconciled_demand_mw(self) -> float:
        positive = sum(driver.value_mw for driver in self.positive_drivers)
        negative = sum(driver.value_mw for driver in self.negative_drivers)
        return self.usual_demand_mw + positive - negative


@dataclass(frozen=True)
class ConfidenceFactor:
    name: str
    value: str
    status: str
    detail: str


@dataclass(frozen=True)
class ConfidenceSummary:
    level: str
    detail: str
    factors: list[ConfidenceFactor]


@dataclass(frozen=True)
class BestWindow:
    objective: str
    start: pd.Timestamp
    end: pd.Timestamp
    score: float
    mean_demand_mw: float
    mean_balance_score: float | None
    mean_carbon_g_per_kwh: float | None
    explanation: str


def forecast_points_from_twin(
    twin: TwinResponse | None,
    *,
    legacy_points: Iterable[ForecastPointView] = (),
) -> list[ForecastPointView]:
    """Build page forecast points from typed twin snapshots when available."""
    if twin is None or not twin.snapshots:
        return []
    legacy_errors = {point.horizon_hours: point.backtest_error for point in legacy_points if point.horizon_hours}
    origin = _as_utc(twin.from_time)
    points: list[ForecastPointView] = []
    for snapshot in twin.snapshots:
        target = _as_utc(snapshot.event_time)
        horizon = int(round((target - origin).total_seconds() / 3600.0))
        if horizon <= 0 or snapshot.demand_forecast is None:
            continue
        interval = snapshot.demand_forecast
        p50 = _number(interval.p50.value)
        if p50 is None:
            continue
        p10 = _number(interval.p10.value, default=p50)
        p90 = _number(interval.p90.value, default=p50)
        usual = _number(snapshot.usual_demand_baseline.value if snapshot.usual_demand_baseline else None, default=p50)
        source_name = interval.p50.source.name or "Demand forecast"
        fallback_reason = interval.p50.source.fallback_reason
        pressure = "Unknown"
        balance = snapshot.modelled_national_balance_context or snapshot.national.balance_context
        if balance is not None:
            pressure = status_label(balance.status)
        backtest = _number(
            interval.confidence.backtest_mae.value if interval.confidence.backtest_mae else None,
            default=legacy_errors.get(horizon, 0.0),
        )
        points.append(
            ForecastPointView(
                timestamp=target,
                p10=float(p10),
                p50=float(p50),
                p90=float(max(p90, p50)),
                source=source_name,
                pressure_label=pressure,
                drivers=[
                    {
                        "name": "Usual demand",
                        "value": float(usual),
                        "unit": "MW",
                        "note": "Comparable-history baseline for this local hour.",
                    },
                    {
                        "name": "Forecast correction",
                        "value": float(p50 - usual),
                        "unit": "MW",
                        "note": "Difference between the selected forecast and usual demand.",
                    },
                ],
                backtest_error=float(backtest),
                horizon_hours=horizon,
                usual_demand_mw=float(usual),
                route_label=source_name,
                fallback_reason=fallback_reason,
            )
        )
    return points


def twin_aligned_to_reference(
    twin: TwinResponse | None,
    reference_timestamp: datetime | pd.Timestamp,
    *,
    tolerance_hours: int = 6,
) -> bool:
    if twin is None or not twin.snapshots:
        return False
    reference = _as_utc(reference_timestamp).floor("h")
    origin = _as_utc(twin.from_time).floor("h")
    return abs(origin - reference) <= pd.Timedelta(hours=tolerance_hours)


def enrich_forecast_frame_with_twin(forecast: pd.DataFrame, twin: TwinResponse | None) -> pd.DataFrame:
    if forecast.empty:
        return forecast.copy()
    result = forecast.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    snapshots = _snapshot_by_timestamp(twin)
    columns: dict[str, list[Any]] = {
        "balance_score": [],
        "balance_label": [],
        "carbon_g_per_kwh": [],
        "generation_total_mw": [],
        "net_imports_mw": [],
        "supply_margin_mw": [],
    }
    usual_values: list[float | None] = []
    for timestamp in result["timestamp"]:
        snapshot = snapshots.get(_timestamp_key(timestamp))
        usual = None
        if snapshot is not None:
            usual = _number(snapshot.usual_demand_baseline.value if snapshot.usual_demand_baseline else None)
            balance = snapshot.modelled_national_balance_context or snapshot.national.balance_context
            carbon = snapshot.carbon_estimate
            mix = snapshot.generation_mix_estimate
            exchange = snapshot.exchange_estimate
            columns["balance_score"].append(_number(balance.pressure_ratio.value if balance else None))
            columns["balance_label"].append(status_label(balance.status) if balance else None)
            columns["carbon_g_per_kwh"].append(_number(carbon.intensity.value if carbon else None))
            columns["generation_total_mw"].append(_number(mix.total.value if mix else None))
            columns["net_imports_mw"].append(_number(exchange.net_imports.value if exchange else None))
            columns["supply_margin_mw"].append(_number(balance.supply_margin.value if balance else None))
        else:
            for values in columns.values():
                values.append(None)
        usual_values.append(usual)
    for column, values in columns.items():
        result[column] = values
    if "usual_demand_mw" not in result:
        result["usual_demand_mw"] = usual_values
    else:
        result["usual_demand_mw"] = pd.to_numeric(result["usual_demand_mw"], errors="coerce")
        result["usual_demand_mw"] = result["usual_demand_mw"].where(result["usual_demand_mw"].notna(), pd.Series(usual_values))
    result["pressure_numeric"] = result.get("pressure_label", pd.Series("Unknown", index=result.index)).map(pressure_score)
    return result


def selected_forecast_point(
    points: list[ForecastPointView],
    selected: datetime | pd.Timestamp | None,
) -> ForecastPointView | None:
    if not points:
        return None
    if selected is None:
        return points[0]
    target = _as_utc(selected)
    return min(points, key=lambda point: abs(_as_utc(point.timestamp) - target))


def selected_twin_snapshot(
    twin: TwinResponse | None,
    selected: datetime | pd.Timestamp | None,
    *,
    future_only: bool = True,
) -> TwinSnapshot | None:
    if twin is None or not twin.snapshots:
        return None
    snapshots = list(twin.snapshots[1:] if future_only else twin.snapshots)
    if not snapshots:
        return None
    if selected is None:
        return snapshots[0]
    target = _as_utc(selected)
    return min(snapshots, key=lambda snapshot: abs(_as_utc(snapshot.event_time) - target))


def selected_timestamp_from_chart_event(event: object | None) -> pd.Timestamp | None:
    points = None
    selection = getattr(event, "selection", None)
    if selection is not None:
        points = getattr(selection, "points", None)
    if points is None and isinstance(event, dict):
        points = event.get("selection", {}).get("points")
    if not points:
        return None
    point = points[0]
    value: Any = None
    if isinstance(point, dict):
        customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple)) and customdata:
            value = customdata[0]
        value = value or point.get("x")
    else:
        customdata = getattr(point, "customdata", None)
        if isinstance(customdata, (list, tuple)) and customdata:
            value = customdata[0]
        value = value or getattr(point, "x", None)
    if value is None:
        return None
    return _as_utc(value)


def peak_forecast_row(forecast: pd.DataFrame) -> pd.Series | None:
    if forecast.empty or "p50" not in forecast:
        return None
    p50 = pd.to_numeric(forecast["p50"], errors="coerce")
    if p50.dropna().empty:
        return None
    return forecast.loc[p50.idxmax()]


def choose_best_window(
    forecast: pd.DataFrame,
    objective: str,
    *,
    window_hours: int = 3,
    locale: str = "en",
) -> BestWindow | None:
    if forecast.empty or "timestamp" not in forecast:
        return None
    objective = objective if objective in BEST_WINDOW_OBJECTIVES else "combined"
    frame = forecast.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["p50"] = pd.to_numeric(frame.get("p50"), errors="coerce")
    frame = frame.dropna(subset=["timestamp", "p50"]).sort_values("timestamp").reset_index(drop=True)
    if "pressure_label" in frame:
        frame = frame.loc[~frame["pressure_label"].eq("No recommendation")].reset_index(drop=True)
    if frame.empty:
        return None

    frame["balance_score_for_window"] = _metric_series(frame, "balance_score", fallback="pressure_numeric")
    frame["carbon_for_window"] = _metric_series(frame, "carbon_g_per_kwh", fallback="p50")
    frame["demand_for_window"] = pd.to_numeric(frame["p50"], errors="coerce")
    balance_norm = _normalised(frame["balance_score_for_window"])
    carbon_norm = _normalised(frame["carbon_for_window"])
    demand_norm = _normalised(frame["demand_for_window"])
    if objective == "lowest_balance":
        frame["objective_score"] = frame["balance_score_for_window"]
        explanation = t("next48h.best_window.explanations.lowest_balance", locale=locale)
    elif objective == "lowest_carbon":
        frame["objective_score"] = frame["carbon_for_window"]
        explanation = t("next48h.best_window.explanations.lowest_carbon", locale=locale)
    else:
        frame["objective_score"] = 0.5 * balance_norm + 0.3 * carbon_norm + 0.2 * demand_norm
        explanation = t("next48h.best_window.explanations.combined", locale=locale)

    window = max(1, min(int(window_hours), len(frame)))
    candidates: list[tuple[float, int, pd.DataFrame]] = []
    for start in range(0, len(frame) - window + 1):
        chunk = frame.iloc[start : start + window]
        candidates.append((float(chunk["objective_score"].mean()), start, chunk))
    score, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    carbon = pd.to_numeric(selected.get("carbon_g_per_kwh"), errors="coerce").mean()
    balance = pd.to_numeric(selected.get("balance_score"), errors="coerce").mean()
    return BestWindow(
        objective=objective,
        start=pd.Timestamp(selected.iloc[0]["timestamp"]),
        end=pd.Timestamp(selected.iloc[-1]["timestamp"]),
        score=score,
        mean_demand_mw=float(pd.to_numeric(selected["p50"], errors="coerce").mean()),
        mean_balance_score=None if pd.isna(balance) else float(balance),
        mean_carbon_g_per_kwh=None if pd.isna(carbon) else float(carbon),
        explanation=explanation,
    )


def selected_hour_explanation(
    point: ForecastPointView,
    snapshot: TwinSnapshot | None = None,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = "en",
) -> SelectedHourExplanation:
    p10, expected, p90 = _demand_interval_values(point, snapshot)
    usual = _usual_value(point, snapshot, default=expected)
    correction = expected - usual
    positive: list[DemandDriver] = []
    negative: list[DemandDriver] = []
    if correction >= 50:
        positive.append(
            DemandDriver(
                t("next48h.selected_explanation.drivers.lift_name", locale=locale),
                correction,
                t("next48h.selected_explanation.drivers.lift_detail", locale=locale),
            )
        )
    elif correction <= -50:
        negative.append(
            DemandDriver(
                t("next48h.selected_explanation.drivers.easing_name", locale=locale),
                abs(correction),
                t("next48h.selected_explanation.drivers.easing_detail", locale=locale),
            )
        )
    reconciled = usual + sum(driver.value_mw for driver in positive) - sum(driver.value_mw for driver in negative)
    delta = expected - usual
    if abs(delta) < 50:
        direction = t("next48h.selected_explanation.direction.near", locale=locale)
    elif delta > 0:
        direction = t(
            "next48h.selected_explanation.direction.above",
            locale=locale,
            delta=format_signed_mw(delta, locale=locale),
        )
    else:
        direction = t(
            "next48h.selected_explanation.direction.below",
            locale=locale,
            delta=format_signed_mw(delta, locale=locale),
        )
    text = t(
        "next48h.selected_explanation.text",
        locale=locale,
        hour=local_hour_label(point.timestamp, timezone_name=timezone_name, locale=locale),
        expected=format_mw(expected, locale=locale),
        direction=direction,
    )
    return SelectedHourExplanation(
        usual_demand_mw=usual,
        positive_drivers=positive,
        negative_drivers=negative,
        expected_demand_mw=expected,
        p10_mw=p10,
        p90_mw=p90,
        reconciliation_error_mw=expected - reconciled,
        text=text,
    )


def confidence_summary(
    point: ForecastPointView,
    snapshot: TwinSnapshot | None = None,
    *,
    locale: str = "en",
) -> ConfidenceSummary:
    p10, expected, p90 = _demand_interval_values(point, snapshot)
    width = max(p90 - p10, 0.0)
    width_ratio = width / expected if expected > 0 else 0.0
    horizon = point.horizon_hours or _horizon_from_snapshot(snapshot)
    source = _source_text(point, snapshot)
    fallback = "fallback" in source.lower() or "unsupported" in source.lower()
    source_display = _source_display(source, locale=locale)
    route_status = "watch" if fallback else "good"
    perf_ratio = point.backtest_error / expected if expected > 0 and point.backtest_error else 0.0
    score = 3.0
    if horizon > 24:
        score -= 0.8
    if horizon > 36:
        score -= 0.5
    if width_ratio > 0.12:
        score -= 0.7
    if width_ratio > 0.18:
        score -= 0.5
    if fallback:
        score -= 0.9
    if perf_ratio > 0.06:
        score -= 0.5
    if snapshot is not None and snapshot.demand_forecast is not None:
        confidence = snapshot.demand_forecast.confidence.confidence.value
        if confidence == "low":
            score -= 0.6
        elif confidence == "high":
            score += 0.3
    if score >= 2.5:
        level_key = "high"
    elif score >= 1.5:
        level_key = "medium"
    else:
        level_key = "low"
    level = t(f"next48h.confidence.levels.{level_key}", locale=locale)
    factors = [
        ConfidenceFactor(
            t("next48h.confidence.factors.horizon.name", locale=locale),
            t("next48h.confidence.factors.horizon.value", locale=locale, hours=horizon),
            "watch" if horizon > 24 else "good",
            t("next48h.confidence.factors.horizon.detail", locale=locale),
        ),
        ConfidenceFactor(
            t("next48h.confidence.factors.interval_width.name", locale=locale),
            t(
                "next48h.confidence.factors.interval_width.value",
                locale=locale,
                width=format_gw(width, locale=locale),
                ratio=format_percentage(width_ratio, locale=locale),
            ),
            "watch" if width_ratio > 0.12 else "good",
            t("next48h.confidence.factors.interval_width.detail", locale=locale),
        ),
        ConfidenceFactor(
            t("next48h.confidence.factors.weather_disagreement.name", locale=locale),
            t("next48h.confidence.factors.weather_disagreement.value", locale=locale),
            "unknown",
            t("next48h.confidence.factors.weather_disagreement.detail", locale=locale),
        ),
        ConfidenceFactor(
            t("next48h.confidence.factors.fallback_sources.name", locale=locale),
            t(
                "next48h.confidence.factors.fallback_sources.fallback_value"
                if fallback
                else "next48h.confidence.factors.fallback_sources.model_value",
                locale=locale,
            ),
            route_status,
            source_display,
        ),
        ConfidenceFactor(
            t("next48h.confidence.factors.recent_performance.name", locale=locale),
            (
                format_mw(point.backtest_error, locale=locale)
                if point.backtest_error
                else t("next48h.confidence.factors.recent_performance.unavailable_value", locale=locale)
            ),
            "watch" if perf_ratio > 0.06 else "good",
            t("next48h.confidence.factors.recent_performance.detail", locale=locale),
        ),
    ]
    detail = t("next48h.confidence.detail", locale=locale, level=level)
    return ConfidenceSummary(level=level, detail=detail, factors=factors)


def future_regional_map_frame(
    snapshot: TwinSnapshot | None,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = "en",
) -> pd.DataFrame:
    if snapshot is None or not snapshot.regional_demand_context:
        return pd.DataFrame()
    event_label = format_timestamp(snapshot.event_time, timezone_name=timezone_name, locale=locale)
    records: list[dict[str, Any]] = []
    for region in snapshot.regional_demand_context:
        demand = _number(region.forecast.p50.value)
        usual = _number(region.usual.value)
        anomaly = None if demand is None or usual in {None, 0} else (float(demand) - float(usual)) / float(usual)
        anomaly_pct = None if anomaly is None else anomaly * 100.0
        difference = None if demand is None or usual is None else float(demand) - float(usual)
        records.append(
            {
                "region_code": region.region_code,
                "region_display": region.region_name,
                "demand_anomaly_pct": anomaly_pct,
                "demand_anomaly_label": demand_anomaly_label(anomaly),
                "demand_anomaly_score": _anomaly_score(anomaly),
                "consumption_mw": demand,
                "usual_demand_mw": usual,
                "difference_mw": difference,
                "demand_label": format_mw(demand, locale=locale),
                "usual_label": format_mw(usual, locale=locale),
                "difference_label": format_signed_mw(difference, locale=locale),
                "freshness_label": t("next48h.regional.typed_freshness", locale=locale, hour=event_label),
                "source_label": t("next48h.regional.typed_source", locale=locale),
                "availability_flag": anomaly_pct is not None,
                "unavailable_reason": t("next48h.regional.typed_unavailable", locale=locale),
                "method": region.method,
                "note": t("next48h.regional.typed_note", locale=locale, default=region.note),
            }
        )
    return pd.DataFrame.from_records(records)


def projected_future_regional_map_frame(
    regional: pd.DataFrame,
    point: ForecastPointView,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = "en",
) -> pd.DataFrame:
    if regional.empty:
        return pd.DataFrame()
    frame = regional.copy()
    event_label = format_timestamp(point.timestamp, timezone_name=timezone_name, locale=locale)
    demand = pd.to_numeric(frame.get("consumption_mw"), errors="coerce")
    usual = pd.to_numeric(frame.get("usual_demand_mw"), errors="coerce")
    demand_total = float(demand.sum()) if demand.notna().any() else 0.0
    usual_total = float(usual.sum()) if usual.notna().any() else 0.0
    if demand_total <= 0:
        demand_share = pd.Series(1.0 / max(len(frame), 1), index=frame.index)
    else:
        demand_share = demand.fillna(0.0) / demand_total
    if usual_total <= 0:
        usual_share = demand_share
    else:
        usual_share = usual.fillna(0.0) / usual_total
    forecast_demand = demand_share * float(point.p50)
    point_usual = float(point.usual_demand_mw or point.p50)
    forecast_usual = usual_share * point_usual
    difference = forecast_demand - forecast_usual
    anomaly = difference / forecast_usual.replace(0, pd.NA)
    result = pd.DataFrame(
        {
            "region_code": frame["region_code"].astype(str),
            "region_display": frame.get("region_display", frame["region_code"].astype(str)),
            "demand_anomaly_pct": anomaly.astype(float) * 100.0,
            "demand_anomaly_label": anomaly.map(demand_anomaly_label),
            "demand_anomaly_score": anomaly.map(_anomaly_score),
            "consumption_mw": forecast_demand.astype(float),
            "usual_demand_mw": forecast_usual.astype(float),
            "difference_mw": difference.astype(float),
        }
    )
    result["demand_label"] = result["consumption_mw"].map(lambda value: format_mw(value, locale=locale))
    result["usual_label"] = result["usual_demand_mw"].map(lambda value: format_mw(value, locale=locale))
    result["difference_label"] = result["difference_mw"].map(lambda value: format_signed_mw(value, locale=locale))
    result["freshness_label"] = t("next48h.regional.typed_freshness", locale=locale, hour=event_label)
    result["source_label"] = t("next48h.regional.projected_source", locale=locale)
    result["availability_flag"] = result["demand_anomaly_pct"].notna()
    result["unavailable_reason"] = t("next48h.regional.projected_unavailable", locale=locale)
    result["method"] = t("next48h.regional.projected_method", locale=locale)
    result["note"] = t("next48h.regional.projected_note", locale=locale)
    return result


def generation_mix_rows(snapshot: TwinSnapshot | None, *, locale: str = "en") -> pd.DataFrame:
    columns = [
        t("next48h.tables.generation.component", locale=locale),
        t("next48h.tables.generation.estimate", locale=locale),
        t("next48h.tables.generation.provenance", locale=locale),
        t("next48h.tables.generation.formula", locale=locale),
    ]
    if snapshot is None or snapshot.generation_mix_estimate is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for component in snapshot.generation_mix_estimate.components:
        rows.append(
            {
                columns[0]: _component_label(component.component, locale=locale),
                columns[1]: format_mw(component.value.value, locale=locale),
                columns[2]: _provenance_display(component.provenance_kind.value, locale=locale),
                columns[3]: component.formula or "",
            }
        )
    return pd.DataFrame.from_records(rows)


def weather_for_hour(
    forecast_frame: pd.DataFrame | None,
    target: datetime | pd.Timestamp | None,
) -> dict[str, Any] | None:
    """Pick the hourly Open-Meteo forecast row closest to ``target``.

    Returns a payload shaped like :func:`app.data_loader.load_live_current_weather`
    so callers can feed it into ``current_weather_summary`` without branching on
    shape. Returns ``None`` when the frame is empty or the target is missing.
    """
    if forecast_frame is None or forecast_frame.empty or target is None:
        return None
    if "timestamp" not in forecast_frame:
        return None
    timestamps = pd.to_datetime(forecast_frame["timestamp"], utc=True, errors="coerce")
    valid = forecast_frame.assign(timestamp=timestamps).dropna(subset=["timestamp"])
    if valid.empty:
        return None
    target_ts = _as_utc(target)
    idx = (valid["timestamp"] - target_ts).abs().idxmin()
    row = valid.loc[idx]

    def _maybe_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "temperature_c": _maybe_float(row.get("temperature_c")),
        "wind_kmh": _maybe_float(row.get("wind_kmh")),
        "cloud_pct": _maybe_float(row.get("cloud_pct")),
        "observed_at": pd.Timestamp(row["timestamp"]).isoformat(),
        "location": str(row.get("location") or "Paris"),
        "source": str(row.get("source") or "Open-Meteo"),
    }


def forecast_display_table(
    forecast: pd.DataFrame,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = "en",
) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame()
    frame = forecast.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return pd.DataFrame(
        {
            t("next48h.tables.forecast.time", locale=locale): frame["timestamp"].map(
                lambda value: format_timestamp(value, timezone_name=timezone_name, locale=locale)
            ),
            t("next48h.tables.forecast.p10", locale=locale): frame["p10"].map(lambda value: format_mw(value, locale=locale)),
            t("next48h.tables.forecast.p50", locale=locale): frame["p50"].map(lambda value: format_mw(value, locale=locale)),
            t("next48h.tables.forecast.p90", locale=locale): frame["p90"].map(lambda value: format_mw(value, locale=locale)),
            t("next48h.tables.forecast.usual_demand", locale=locale): frame.get(
                "usual_demand_mw", pd.Series(index=frame.index)
            ).map(lambda value: format_mw(value, locale=locale)),
            t("next48h.tables.forecast.balance_context", locale=locale): frame.get(
                "balance_label", frame.get("pressure_label", pd.Series(index=frame.index))
            )
            .map(lambda value: _status_display(value, locale=locale))
            .fillna(t("next48h.tables.forecast.unknown", locale=locale)),
            t("next48h.tables.forecast.carbon", locale=locale): frame.get("carbon_g_per_kwh", pd.Series(index=frame.index)).map(
                lambda value: (
                    t("next48h.tables.forecast.unavailable", locale=locale)
                    if pd.isna(value)
                    else format_carbon(float(value), locale=locale)
                )
            ),
            t("next48h.tables.forecast.route", locale=locale): frame.get("source", pd.Series(index=frame.index)).map(
                lambda value: _source_display(value, locale=locale)
            ),
        }
    )


def _demand_interval_values(point: ForecastPointView, snapshot: TwinSnapshot | None) -> tuple[float, float, float]:
    if snapshot is not None and snapshot.demand_forecast is not None:
        interval = snapshot.demand_forecast
        expected = _number(interval.p50.value, default=point.p50)
        p10 = _number(interval.p10.value, default=point.p10)
        p90 = _number(interval.p90.value, default=point.p90)
        return float(p10), float(expected), float(max(p90, expected))
    return float(point.p10), float(point.p50), float(max(point.p90, point.p50))


def _usual_value(point: ForecastPointView, snapshot: TwinSnapshot | None, *, default: float) -> float:
    if snapshot is not None and snapshot.usual_demand_baseline is not None:
        value = _number(snapshot.usual_demand_baseline.value)
        if value is not None:
            return float(value)
    if point.usual_demand_mw is not None:
        return float(point.usual_demand_mw)
    return float(default)


def _source_text(point: ForecastPointView, snapshot: TwinSnapshot | None) -> str:
    if snapshot is not None and snapshot.demand_forecast is not None:
        source = snapshot.demand_forecast.p50.source
        reason = f" ({source.fallback_reason})" if source.fallback_reason else ""
        return f"{source.name}{reason}"
    return point.source


def _snapshot_by_timestamp(twin: TwinResponse | None) -> dict[str, TwinSnapshot]:
    if twin is None:
        return {}
    return {_timestamp_key(snapshot.event_time): snapshot for snapshot in twin.snapshots}


def _timestamp_key(value: Any) -> str:
    return _as_utc(value).isoformat()


def _as_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number) or number in {float("inf"), float("-inf")}:
        return default
    return number


def _metric_series(frame: pd.DataFrame, column: str, *, fallback: str) -> pd.Series:
    values = pd.to_numeric(frame.get(column), errors="coerce")
    if values.notna().any():
        return values
    return pd.to_numeric(frame.get(fallback), errors="coerce").fillna(0.5)


def _normalised(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.5, index=values.index)
    numeric = numeric.fillna(float(numeric.median()))
    low = float(numeric.min())
    high = float(numeric.max())
    if abs(high - low) < 1e-9:
        return pd.Series(0.5, index=values.index)
    return (numeric - low) / (high - low)


def _horizon_from_snapshot(snapshot: TwinSnapshot | None) -> int:
    if snapshot is None:
        return 0
    try:
        return int(round((pd.Timestamp(snapshot.event_time) - pd.Timestamp(snapshot.source.update_time)).total_seconds() / 3600.0))
    except (TypeError, ValueError):
        return 0


def _anomaly_score(anomaly: float | None) -> float:
    if anomaly is None or pd.isna(anomaly):
        return 0.5
    return max(0.0, min(1.0, 0.5 + float(anomaly) / 0.36))


def _component_label(component: str, *, locale: str = "en") -> str:
    labels = {
        "nuclear_expected_output_or_availability": "Nuclear expected output",
        "wind": "Wind",
        "solar": "Solar",
        "residual_flexible_sources_and_imports": "Residual flexible sources and imports",
    }
    return t(f"next48h.components.{component}", locale=locale, default=labels.get(component, component.replace("_", " ").title()))


def _status_display(value: Any, *, locale: str = "en") -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return t("next48h.status.unknown", locale=locale)
    key = text.lower().replace("-", "_").replace(" ", "_")
    return t(f"next48h.status.{key}", locale=locale, default=text)


def _source_display(value: Any, *, locale: str = "en") -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return t("next48h.tables.forecast.unavailable", locale=locale)
    lowered = text.lower()
    if "usual-demand baseline fallback" in lowered or "usual demand baseline fallback" in lowered:
        return t("next48h.sources.usual_fallback", locale=locale)
    if "unsupported" in lowered:
        return t("next48h.sources.unsupported_route", locale=locale)
    if "fallback" in lowered:
        return t("next48h.sources.fallback_route", locale=locale)
    if text == "Demand forecast":
        return t("next48h.sources.demand_forecast", locale=locale)
    return text


def _provenance_display(value: Any, *, locale: str = "en") -> str:
    text = str(value or "").strip()
    key = text.lower().replace("-", "_").replace(" ", "_")
    return t(f"next48h.provenance_kinds.{key}", locale=locale, default=text.replace("_", " "))
