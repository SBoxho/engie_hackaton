"""Hour-indexed electricity-system twin snapshots.

This module composes the existing demand forecast, current-state source bundle,
regional eCO2mix context, and explicit generation fallbacks into one coherent
snapshot per hour.  The balance context is a transparent national analytical
context; it is not an EcoWatt calculation and not an operational reserve margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.api.current_state import CurrentStateBundle, load_current_state_bundle, normalize_region_code
from src.api.forecast_explanations import DEFAULT_FORECAST_MODEL_PATH, load_default_hourly_features
from src.config import settings
from src.contracts.energy_twin import (
    BaselineDefinition,
    CarbonEstimate,
    Confidence,
    ConfidenceAssessment,
    DataProvenance,
    DataQuality,
    DemandContext,
    DomainMode,
    EstimateProvenanceKind,
    EstimatedGenerationMix,
    ExchangeEstimate,
    ForecastInterval,
    Freshness,
    GenerationAvailabilityContext,
    GenerationEstimate,
    GenerationMix,
    ModelledBalanceContext,
    ModelledBalanceContribution,
    NationalState,
    OfficialSignal,
    QuantifiedValue,
    RegionalDemandForecast,
    Scope,
    SourceType,
    Status,
    TwinComponentEstimate,
    TwinResponse,
    TwinSnapshot,
    UnavailableField,
    Unit,
    percentage_value,
    power_value,
)
from src.contracts.status_thresholds import (
    load_status_thresholds,
    modelled_balance_status_for_score,
    status_label,
    threshold_config_version,
)
from src.data_sources.ecowatt import status_at
from src.data_sources.rte_eco2mix_regional import REGION_NAMES
from src.models.probabilistic_demand import (
    DemandForecastService,
    MODEL_FILENAME,
    utc_iso,
)
from src.public_data.adapters.rte import (
    OptionalRteGenerationForecastAdapter,
    OptionalRteUnavailabilityAdapter,
)
from src.public_data.contracts import DataWindow


GENERATION_HISTORY_COLUMNS = (
    "nuclear_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "gas_mw",
    "coal_mw",
    "oil_mw",
    "bioenergy_mw",
    "imports_mw",
    "exports_mw",
    "net_imports_mw",
    "co2_intensity_g_per_kwh",
)
RESIDUAL_BUCKET_NAME = "residual_flexible_sources_and_imports"
UNSUPPORTED_PHYSICAL_BEHAVIOURS = [
    "No AC power-flow, voltage, congestion, or network constraint model is implemented.",
    "No unit-commitment, dispatch optimization, ramp-rate, storage, or balancing-market model is implemented.",
    "The residual bucket combines flexible domestic output and imports until a dispatch model exists.",
    "Regional demand forecasts are allocation context only and do not imply regional adequacy or shortage status.",
    "Carbon estimates are separate context and are not used in the modelled national balance status.",
]


@dataclass(frozen=True)
class OptionalSourceResult:
    frame: pd.DataFrame
    unavailable: list[UnavailableField]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TwinService:
    """Build current and 48-hour forecast TwinSnapshot objects."""

    def __init__(
        self,
        *,
        bundle_loader: Callable[[datetime], CurrentStateBundle] | None = None,
        hourly_loader: Callable[[], pd.DataFrame] | None = None,
        generation_forecast_loader: Callable[[datetime, int], pd.DataFrame] | None = None,
        unavailability_loader: Callable[[datetime, int], pd.DataFrame] | None = None,
        artifact: Mapping[str, Any] | None = None,
        artifact_path: str | Path | None = DEFAULT_FORECAST_MODEL_PATH,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._bundle_loader = bundle_loader or load_current_state_bundle
        self._hourly_loader = hourly_loader or load_default_hourly_features
        self._generation_forecast_loader = generation_forecast_loader
        self._unavailability_loader = unavailability_loader
        self._now = now
        if artifact_path == DEFAULT_FORECAST_MODEL_PATH and artifact_path is not None:
            artifact_path = settings.processed_dir / "demand_forecast" / MODEL_FILENAME
        self._forecast_service = DemandForecastService(artifact=artifact, artifact_path=artifact_path)

    def get_twin(
        self,
        *,
        from_timestamp: str | datetime | None = None,
        hours: int = 48,
        region: str | None = None,
    ) -> TwinResponse:
        hours = min(max(int(hours), 0), 48)
        selected_region = normalize_region_code(region) if region else None
        generated_at = _ensure_utc_datetime(self._now())
        hourly = _prepare_hourly_frame(self._hourly_loader())
        if hourly.empty:
            raise ValueError("No hourly demand features are available for twin snapshots.")
        requested_origin = _requested_origin(from_timestamp, hourly)
        forecast_run = (
            self._forecast_service.forecast(
                requested_origin,
                hourly,
                horizons_hours=tuple(range(1, hours + 1)),
                timezone_name=settings.timezone,
            )
            if hours > 0
            else None
        )
        effective_origin = _effective_origin(requested_origin, hourly, forecast_run)
        bundle = self._bundle_loader(generated_at)
        generation_forecast = self._optional_generation_forecast(effective_origin, hours)
        unavailability = self._optional_unavailability(effective_origin, hours)
        unavailable = [*generation_forecast.unavailable, *unavailability.unavailable]
        generation_history = _generation_history(bundle.national.frame, hourly)
        forecast_points = _forecast_points_by_horizon(forecast_run.points if forecast_run else [])

        snapshots = [
            self._snapshot_for_hour(
                horizon=0,
                event_time=effective_origin,
                generated_at=generated_at,
                hourly=hourly,
                bundle=bundle,
                generation_history=generation_history,
                generation_forecast=generation_forecast.frame,
                unavailability=unavailability.frame,
                optional_unavailable=unavailable,
                forecast_point=None,
                selected_region=selected_region,
            )
        ]
        for horizon in range(1, hours + 1):
            point = forecast_points.get(horizon)
            if point is None:
                continue
            snapshots.append(
                self._snapshot_for_hour(
                    horizon=horizon,
                    event_time=pd.Timestamp(point["target_timestamp"]).to_pydatetime(),
                    generated_at=generated_at,
                    hourly=hourly,
                    bundle=bundle,
                    generation_history=generation_history,
                    generation_forecast=generation_forecast.frame,
                    unavailability=unavailability.frame,
                    optional_unavailable=unavailable,
                    forecast_point=point,
                    selected_region=selected_region,
                )
            )
        return TwinResponse(
            generated_at=generated_at,
            from_time=effective_origin.to_pydatetime(),
            hours=hours,
            region=selected_region,
            snapshots=snapshots,
            unavailable_fields=unavailable,
        )

    def _optional_generation_forecast(self, origin: pd.Timestamp, hours: int) -> OptionalSourceResult:
        if self._generation_forecast_loader is not None:
            return OptionalSourceResult(
                frame=_timestamped_frame(self._generation_forecast_loader(origin.to_pydatetime(), hours)),
                unavailable=[],
            )
        try:
            adapter = OptionalRteGenerationForecastAdapter()
            window = DataWindow.from_values(origin.to_pydatetime(), (origin + pd.Timedelta(hours=hours + 1)).to_pydatetime())
            result = adapter.fetch(window)
            frame = getattr(result, "silver", result)
            return OptionalSourceResult(frame=_timestamped_frame(frame), unavailable=[])
        except Exception as exc:  # Optional credentialed integration, unavailable by default.
            return OptionalSourceResult(
                frame=pd.DataFrame(),
                unavailable=[
                    UnavailableField(
                        field="rte_generation_forecast_optional",
                        reason=f"Optional RTE generation forecast unavailable: {exc}",
                    )
                ],
            )

    def _optional_unavailability(self, origin: pd.Timestamp, hours: int) -> OptionalSourceResult:
        if self._unavailability_loader is not None:
            return OptionalSourceResult(
                frame=_timestamped_unavailability_frame(self._unavailability_loader(origin.to_pydatetime(), hours)),
                unavailable=[],
            )
        try:
            adapter = OptionalRteUnavailabilityAdapter()
            window = DataWindow.from_values(origin.to_pydatetime(), (origin + pd.Timedelta(hours=hours + 1)).to_pydatetime())
            result = adapter.fetch(window)
            frame = getattr(result, "silver", result)
            return OptionalSourceResult(frame=_timestamped_unavailability_frame(frame), unavailable=[])
        except Exception as exc:  # Optional credentialed integration, unavailable by default.
            return OptionalSourceResult(
                frame=pd.DataFrame(),
                unavailable=[
                    UnavailableField(
                        field="rte_asset_unavailability_optional",
                        reason=f"Optional RTE announced unavailability unavailable: {exc}",
                    )
                ],
            )

    def _snapshot_for_hour(
        self,
        *,
        horizon: int,
        event_time: pd.Timestamp | datetime,
        generated_at: datetime,
        hourly: pd.DataFrame,
        bundle: CurrentStateBundle,
        generation_history: pd.DataFrame,
        generation_forecast: pd.DataFrame,
        unavailability: pd.DataFrame,
        optional_unavailable: list[UnavailableField],
        forecast_point: Mapping[str, Any] | None,
        selected_region: str | None,
    ) -> TwinSnapshot:
        event_ts = _to_utc_timestamp(event_time).floor("h")
        event_dt = event_ts.to_pydatetime()
        mode = _snapshot_mode(horizon)
        quality = _quality_for_mode(mode, bool(forecast_point))
        demand_interval = _demand_interval(event_ts, generated_at, hourly, forecast_point, mode, quality)
        usual = _usual_value(event_ts, generated_at, forecast_point, hourly, mode, quality)
        demand_p50 = _value_or_zero(demand_interval.p50.value)
        regional = _regional_demand_context(
            bundle.regional_history,
            bundle.regional_snapshot.frame,
            national_p10=_value_or_zero(demand_interval.p10.value),
            national_p50=demand_p50,
            national_p90=_value_or_zero(demand_interval.p90.value),
            target=event_ts,
            update_time=generated_at,
            mode=mode,
            selected_region=selected_region,
        )
        components = _generation_components(
            event_ts,
            generated_at,
            demand_p50=demand_p50,
            hourly=hourly,
            history=generation_history,
            official_forecast=generation_forecast,
            unavailability=unavailability,
            mode=mode,
        )
        exchange = _exchange_estimate(event_ts, generated_at, generation_history, mode)
        carbon = _carbon_estimate(event_ts, generated_at, generation_history, mode)
        availability = _generation_availability_context(
            components["nuclear"],
            event_ts,
            generated_at,
            unavailability,
            optional_unavailable,
            mode,
        )
        generation_mix = _estimated_generation_mix(
            event_ts,
            generated_at,
            [components["nuclear"], components["wind"], components["solar"], components["residual"]],
            mode,
        )
        balance, balance_contributions = _modelled_balance_context(
            event_ts,
            generated_at,
            demand_p50=demand_p50,
            wind_mw=_value_or_zero(components["wind"].value.value),
            solar_mw=_value_or_zero(components["solar"].value.value),
            generation_mix=generation_mix,
            exchange=exchange,
            unavailability_mw=_value_or_zero(availability.announced_unavailable.value),
            history=generation_history,
            mode=mode,
        )
        official = _official_signal(bundle, event_ts, generated_at, mode)
        demand_context = DemandContext(
            current=demand_interval.p50,
            usual=usual,
            anomaly_percentage=_anomaly_percentage(demand_interval.p50, usual, generated_at, mode, quality),
            baseline_definition=_baseline_definition(),
            scope=Scope.NATIONAL,
            interpretation=(
                "National forecast demand compared with the usual-demand baseline."
                if horizon > 0
                else "National observed demand compared with the usual-demand baseline."
            ),
        )
        national = NationalState(
            scope=Scope.NATIONAL,
            demand_context=demand_context,
            generation_mix=_national_generation_mix(generation_mix, carbon),
            official_signal=official,
            balance_context=balance,
            source=_source(
                SourceType.MODEL if horizon > 0 else SourceType.OBSERVED,
                "Electricity twin national context",
                event_dt,
                generated_at,
                mode,
            ),
            quality=quality,
            regions=None,
        )
        snapshot_source = _source(
            SourceType.MODEL if horizon > 0 else SourceType.OBSERVED,
            "Energy Pulse electricity-system twin",
            event_dt,
            generated_at,
            mode,
            transformation="Composed demand, generation, exchange, balance, signal, and carbon context.",
        )
        return TwinSnapshot(
            snapshot_id=f"twin-{utc_iso(event_ts)}",
            mode=mode,
            event_time=event_dt,
            update_time=generated_at,
            national=national,
            source=snapshot_source,
            quality=quality,
            demand_forecast=demand_interval,
            usual_demand_baseline=usual,
            regional_demand_context=regional,
            wind_estimate=components["wind"],
            solar_estimate=components["solar"],
            generation_availability_context=availability,
            generation_mix_estimate=generation_mix,
            exchange_estimate=exchange,
            modelled_national_balance_context=balance,
            modelled_balance_contributions=balance_contributions,
            official_signal_context=official,
            carbon_estimate=carbon,
            provenance_chain=_provenance_chain(
                demand_interval,
                components["wind"],
                components["solar"],
                components["nuclear"],
                components["residual"],
                exchange,
                carbon,
                balance,
                official,
            ),
            unsupported_physical_behaviours=list(UNSUPPORTED_PHYSICAL_BEHAVIOURS),
        )


def _demand_interval(
    event_time: pd.Timestamp,
    update_time: datetime,
    hourly: pd.DataFrame,
    forecast_point: Mapping[str, Any] | None,
    mode: DomainMode,
    quality: DataQuality,
) -> ForecastInterval:
    if forecast_point is None:
        current = _current_hour_value(hourly, event_time, "consumption_mw")
        source = _source(SourceType.OBSERVED, "Observed ODRÉ eCO2mix demand", event_time, update_time, mode)
        confidence = _confidence(source, quality, "Current observed demand; uncertainty interval collapses to the observed value.")
        return ForecastInterval(
            p10=power_value(current, event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            p50=power_value(current, event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            p90=power_value(current, event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            confidence=confidence,
        )
    route = str(forecast_point.get("route") or "baseline_fallback")
    fallback = route != "validated_model"
    source = _source(
        SourceType.FALLBACK if fallback else SourceType.MODEL,
        "Usual-demand baseline fallback" if fallback else "Demand residual quantile model",
        event_time,
        update_time,
        mode,
        fallback=fallback,
        fallback_reason=str(forecast_point.get("fallback_reason") or "usual-demand baseline fallback") if fallback else None,
        transformation="p50 = usual-demand baseline + residual model correction when a champion model exists.",
    )
    confidence = _confidence(
        source,
        _quality(Freshness.FRESH, Confidence.LOW if fallback else Confidence.MEDIUM, Status.UNKNOWN),
        "Probabilistic demand forecast with explicit route and fallback reason.",
    )
    return ForecastInterval(
        p10=power_value(_finite_float(forecast_point.get("p10")), event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        p50=power_value(_finite_float(forecast_point.get("p50")), event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        p90=power_value(_finite_float(forecast_point.get("p90")), event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        confidence=confidence,
    )


def _usual_value(
    event_time: pd.Timestamp,
    update_time: datetime,
    forecast_point: Mapping[str, Any] | None,
    hourly: pd.DataFrame,
    mode: DomainMode,
    quality: DataQuality,
) -> QuantifiedValue:
    source = _source(
        SourceType.FALLBACK,
        "Usual-demand comparable-history baseline",
        event_time,
        update_time,
        mode,
        fallback=True,
        fallback_reason="Transparent usual-demand baseline is used as the reference level.",
        transformation="Median comparable recent demand by local hour and day type.",
    )
    if forecast_point is not None:
        value = _finite_float(forecast_point.get("usual_demand_mw"))
        if value is not None:
            return power_value(value, event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality)
    usual = _comparable_median(hourly, event_time, "consumption_mw")
    if usual is None:
        usual = _current_hour_value(hourly, event_time, "consumption_mw")
    return power_value(usual, event_time=event_time.to_pydatetime(), update_time=update_time, source=source, quality=quality)


def _regional_demand_context(
    regional_history: pd.DataFrame,
    regional_snapshot: pd.DataFrame,
    *,
    national_p10: float,
    national_p50: float,
    national_p90: float,
    target: pd.Timestamp,
    update_time: datetime,
    mode: DomainMode,
    selected_region: str | None,
) -> list[RegionalDemandForecast]:
    frame = _regional_history(regional_history, regional_snapshot)
    prelim: dict[str, float] = {}
    source_kind = SourceType.MODEL
    fallback = False
    for code, _name in REGION_NAMES.items():
        row_value = _regional_comparable_value(frame, code, target)
        if row_value is None:
            row_value = _regional_latest_value(frame, code)
            fallback = True
        prelim[code] = float(row_value or 0.0)
    total = sum(prelim.values())
    if total <= 0:
        prelim = {code: national_p50 / max(len(REGION_NAMES), 1) for code in REGION_NAMES}
        total = sum(prelim.values())
        source_kind = SourceType.FALLBACK
        fallback = True
    factor = national_p50 / total if total > 0 else 0.0
    low_ratio = national_p10 / national_p50 if national_p50 > 0 else 1.0
    high_ratio = national_p90 / national_p50 if national_p50 > 0 else 1.0
    result: list[RegionalDemandForecast] = []
    ordered_codes = list(REGION_NAMES)
    if selected_region in REGION_NAMES:
        ordered_codes = [selected_region, *[code for code in ordered_codes if code != selected_region]]
    source = _source(
        source_kind,
        "Pooled regional demand allocator",
        target,
        update_time,
        mode,
        fallback=fallback,
        fallback_reason="Regional history unavailable for one or more regions; latest or equal-share fallback used." if fallback else None,
        transformation=(
            "Unreconciled regional comparable-history forecasts are multiplied by a single national "
            "reconciliation factor so regional P50 values sum to the national P50."
        ),
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM if not fallback else Confidence.LOW, Status.UNKNOWN)
    for code in ordered_codes:
        p50 = prelim[code] * factor
        p10 = max(p50 * low_ratio, 0.0)
        p90 = max(p50 * high_ratio, p50)
        interval = ForecastInterval(
            p10=power_value(p10, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            p50=power_value(p50, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            p90=power_value(p90, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            confidence=_confidence(source, quality, "Regional allocation reconciled to the national demand forecast."),
        )
        result.append(
            RegionalDemandForecast(
                region_code=code,
                region_name=REGION_NAMES[code],
                forecast=interval,
                usual=power_value(prelim[code], event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
                unreconciled_p50=power_value(prelim[code], event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
                share_of_national_p50=percentage_value(
                    p50 / national_p50 if national_p50 > 0 else None,
                    event_time=target.to_pydatetime(),
                    update_time=update_time,
                    source=source,
                    quality=quality,
                ),
                reconciliation_factor=float(factor),
                method=(
                    "Pooled regional comparable-history allocator: prelim_region_p50 = median regional demand "
                    "for matching local hour and day type; reconciled_region_p50 = prelim_region_p50 * "
                    "(national_p50 / sum(prelim_region_p50))."
                ),
                source=source,
                quality=quality,
            )
        )
    return result


def _generation_components(
    target: pd.Timestamp,
    update_time: datetime,
    *,
    demand_p50: float,
    hourly: pd.DataFrame,
    history: pd.DataFrame,
    official_forecast: pd.DataFrame,
    unavailability: pd.DataFrame,
    mode: DomainMode,
) -> dict[str, TwinComponentEstimate]:
    wind = _weather_dependent_component(
        "wind",
        "wind_mw",
        target,
        update_time,
        hourly,
        history,
        official_forecast,
        mode,
    )
    solar = _weather_dependent_component(
        "solar",
        "solar_mw",
        target,
        update_time,
        hourly,
        history,
        official_forecast,
        mode,
    )
    nuclear_base = _comparable_median(history, target, "nuclear_mw")
    if nuclear_base is None:
        nuclear_base = _current_hour_value(history, target, "nuclear_mw")
    nuclear_unavailable = _active_unavailable_mw(unavailability, target, technology="nuclear")
    nuclear_value = max(float(nuclear_base or 0.0) - nuclear_unavailable, 0.0)
    nuclear_kind = EstimateProvenanceKind.OBSERVED if _has_exact_observed(history, target) else EstimateProvenanceKind.STATISTICAL_ESTIMATE
    nuclear_source = _source(
        SourceType.OBSERVED if nuclear_kind == EstimateProvenanceKind.OBSERVED else SourceType.MODEL,
        "Nuclear observed output" if nuclear_kind == EstimateProvenanceKind.OBSERVED else "Nuclear expected output from recent ODRÉ history",
        target,
        update_time,
        mode,
        transformation="Comparable recent ODRÉ nuclear output minus active announced unavailability when available.",
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM, Status.UNKNOWN)
    nuclear = TwinComponentEstimate(
        component="nuclear_expected_output_or_availability",
        value=power_value(nuclear_value, event_time=target.to_pydatetime(), update_time=update_time, source=nuclear_source, quality=quality),
        provenance_kind=nuclear_kind,
        formula="median_comparable_nuclear_output_mw - active_announced_nuclear_unavailability_mw",
        note="Nuclear is represented separately from wind and solar.",
    )
    residual_value = _residual_bucket_value(history, target)
    if residual_value is None:
        residual_value = max(demand_p50 - nuclear_value - _value_or_zero(wind.value.value) - _value_or_zero(solar.value.value), 0.0)
    residual_source = _source(
        SourceType.MODEL,
        "Residual flexible sources and imports estimate",
        target,
        update_time,
        mode,
        transformation="Comparable historical hydro, thermal, bioenergy, and net import residual until dispatch model exists.",
    )
    residual = TwinComponentEstimate(
        component=RESIDUAL_BUCKET_NAME,
        value=power_value(residual_value, event_time=target.to_pydatetime(), update_time=update_time, source=residual_source, quality=quality),
        provenance_kind=EstimateProvenanceKind.RESIDUAL_ESTIMATE,
        formula="median_comparable(hydro + gas + coal + oil + bioenergy + net_imports)",
        note="Explicit residual bucket for uncertain flexible sources and imports.",
    )
    return {"nuclear": nuclear, "wind": wind, "solar": solar, "residual": residual}


def _weather_dependent_component(
    component: str,
    column: str,
    target: pd.Timestamp,
    update_time: datetime,
    hourly: pd.DataFrame,
    history: pd.DataFrame,
    official_forecast: pd.DataFrame,
    mode: DomainMode,
) -> TwinComponentEstimate:
    official = _official_component_value(official_forecast, target, column)
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM, Status.UNKNOWN)
    if official is not None:
        source = _source(SourceType.OFFICIAL, f"RTE {component} generation forecast", target, update_time, mode)
        return TwinComponentEstimate(
            component=component,
            value=power_value(official, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            provenance_kind=EstimateProvenanceKind.OFFICIAL_FORECAST,
            formula=f"RTE official {component} forecast at target hour.",
        )
    if _has_exact_observed(history, target):
        value = _current_hour_value(history, target, column)
        source = _source(SourceType.OBSERVED, f"Observed ODRÉ {component} generation", target, update_time, mode)
        return TwinComponentEstimate(
            component=component,
            value=power_value(value, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            provenance_kind=EstimateProvenanceKind.OBSERVED,
            formula=f"Observed ODRÉ {component}_mw at target hour.",
        )
    estimated = _weather_scaled_generation(component, column, target, hourly, history)
    if estimated is not None:
        source = _source(
            SourceType.MODEL,
            f"Public weather plus recent ODRÉ {component} estimate",
            target,
            update_time,
            mode,
            transformation="Weather-scaled comparable-history generation fallback.",
        )
        return TwinComponentEstimate(
            component=component,
            value=power_value(estimated, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            provenance_kind=EstimateProvenanceKind.STATISTICAL_ESTIMATE,
            formula=(
                "recent_same_hour_odre_generation_median scaled by target public weather "
                "relative to recent same-hour public weather"
            ),
            note="Fallback used because optional RTE generation forecast is unavailable.",
        )
    fallback = _comparable_median(history, target, column)
    if fallback is None:
        fallback = _current_hour_value(history, target, column)
    source = _source(
        SourceType.FALLBACK,
        f"Persistence fallback for {component} generation",
        target,
        update_time,
        mode,
        fallback=True,
        fallback_reason=f"No official {component} forecast or weather-scaled estimate is available.",
    )
    return TwinComponentEstimate(
        component=component,
        value=power_value(fallback, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=_quality(Freshness.FRESH, Confidence.LOW, Status.UNKNOWN)),
        provenance_kind=EstimateProvenanceKind.PERSISTENCE_FALLBACK,
        formula=f"latest_or_comparable_recent_odre_{component}_mw",
    )


def _generation_availability_context(
    nuclear: TwinComponentEstimate,
    target: pd.Timestamp,
    update_time: datetime,
    unavailability: pd.DataFrame,
    optional_unavailable: list[UnavailableField],
    mode: DomainMode,
) -> GenerationAvailabilityContext:
    source = _source(
        SourceType.MODEL,
        "Generation availability context",
        target,
        update_time,
        mode,
        transformation="Optional announced unavailability plus nuclear expected-output context.",
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM if not unavailability.empty else Confidence.LOW, Status.UNKNOWN)
    components = _unavailability_components(unavailability, target, update_time, mode)
    total = sum(_value_or_zero(item.value.value) for item in components)
    return GenerationAvailabilityContext(
        nuclear=nuclear,
        announced_unavailable=power_value(total, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        announced_unavailability_components=components,
        unavailable_optional_sources=list(optional_unavailable),
        method=(
            "Nuclear expected output is represented separately; optional public RTE unavailability records "
            "are included when configured, otherwise the source is listed as unavailable."
        ),
        source=source,
        quality=quality,
    )


def _estimated_generation_mix(
    target: pd.Timestamp,
    update_time: datetime,
    components: list[TwinComponentEstimate],
    mode: DomainMode,
) -> EstimatedGenerationMix:
    total = sum(_value_or_zero(item.value.value) for item in components if item.included_in_total)
    source = _source(
        SourceType.MODEL,
        "Estimated generation mix with residual bucket",
        target,
        update_time,
        mode,
        transformation="Component-level mix estimate with official/statistical/persistence/residual provenance.",
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM, Status.UNKNOWN)
    return EstimatedGenerationMix(
        total=power_value(total, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        components=components,
        residual_bucket_name=RESIDUAL_BUCKET_NAME,
        formula="total = nuclear + wind + solar + residual_flexible_sources_and_imports",
        source=source,
        quality=quality,
    )


def _exchange_estimate(
    target: pd.Timestamp,
    update_time: datetime,
    history: pd.DataFrame,
    mode: DomainMode,
) -> ExchangeEstimate:
    observed = _has_exact_observed(history, target)
    kind = EstimateProvenanceKind.OBSERVED if observed else EstimateProvenanceKind.STATISTICAL_ESTIMATE
    source = _source(
        SourceType.OBSERVED if observed else SourceType.MODEL,
        "Observed exchange" if observed else "Comparable-history exchange estimate",
        target,
        update_time,
        mode,
        transformation="Observed current exchange or median comparable recent exchange where defensible.",
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM, Status.UNKNOWN)
    net = _current_hour_value(history, target, "net_imports_mw") if observed else _comparable_median(history, target, "net_imports_mw")
    if net is None:
        net = _current_hour_value(history, target, "net_imports_mw")
        kind = EstimateProvenanceKind.PERSISTENCE_FALLBACK
        source = _source(
            SourceType.FALLBACK,
            "Persistence fallback exchange estimate",
            target,
            update_time,
            mode,
            fallback=True,
            fallback_reason="Comparable exchange history is unavailable.",
        )
    imports = max(float(net or 0.0), 0.0)
    exports = max(-float(net or 0.0), 0.0)
    return ExchangeEstimate(
        net_imports=power_value(net, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        imports=power_value(imports, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        exports=power_value(exports, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        provenance_kind=kind,
        method="Observed current exchange or comparable-history net import estimate.",
        source=source,
        quality=quality,
    )


def _carbon_estimate(target: pd.Timestamp, update_time: datetime, history: pd.DataFrame, mode: DomainMode) -> CarbonEstimate:
    observed = _has_exact_observed(history, target)
    kind = EstimateProvenanceKind.OBSERVED if observed else EstimateProvenanceKind.PERSISTENCE_FALLBACK
    source = _source(
        SourceType.OBSERVED if observed else SourceType.FALLBACK,
        "Observed carbon intensity" if observed else "Persistence fallback carbon intensity",
        target,
        update_time,
        mode,
        fallback=not observed,
        fallback_reason="Future carbon intensity is not forecast by this twin." if not observed else None,
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM if observed else Confidence.LOW, Status.UNKNOWN)
    value = _current_hour_value(history, target, "co2_intensity_g_per_kwh") if observed else _comparable_median(history, target, "co2_intensity_g_per_kwh")
    if value is None:
        value = _current_hour_value(history, target, "co2_intensity_g_per_kwh")
    return CarbonEstimate(
        intensity=QuantifiedValue(
            value=value,
            unit=Unit.GCO2_PER_KWH,
            event_time=target.to_pydatetime(),
            update_time=update_time,
            source=source,
            quality=quality,
            is_fallback=source.is_fallback,
            label="Carbon intensity context",
        ),
        provenance_kind=kind,
        method="Observed current carbon intensity or comparable-history persistence fallback; never a balance input.",
        included_in_balance_context=False,
        source=source,
        quality=quality,
    )


def _modelled_balance_context(
    target: pd.Timestamp,
    update_time: datetime,
    *,
    demand_p50: float,
    wind_mw: float,
    solar_mw: float,
    generation_mix: EstimatedGenerationMix,
    exchange: ExchangeEstimate,
    unavailability_mw: float,
    history: pd.DataFrame,
    mode: DomainMode,
) -> tuple[ModelledBalanceContext, list[ModelledBalanceContribution]]:
    config = load_status_thresholds()
    balance_config = dict(config.get("modelled_balance_context", {}))
    weights = dict(balance_config.get("weights", {}))
    residual_load = demand_p50 - wind_mw - solar_mw
    percentile = _residual_load_percentile(history, target, residual_load)
    unavailable_ratio = unavailability_mw / demand_p50 if demand_p50 > 0 else 0.0
    unavailable_normalizer = float(balance_config.get("normalizers", {}).get("announced_unavailability_high_ratio", 0.08))
    unavailable_score = min(unavailable_ratio / unavailable_normalizer, 1.0) if unavailable_normalizer > 0 else 0.0
    residual_weight = float(weights.get("residual_load_percentile", 0.75))
    unavailable_weight = float(weights.get("announced_unavailability_ratio", 0.25))
    score = residual_weight * percentile + unavailable_weight * unavailable_score
    status = modelled_balance_status_for_score(score)
    source = _source(
        SourceType.MODEL,
        "Modelled national balance context",
        target,
        update_time,
        mode,
        dataset_id=threshold_config_version(),
        transformation=balance_config.get("method", "historical-residual-load-context.v1"),
    )
    quality = _quality(Freshness.FRESH, Confidence.MEDIUM, status)
    total_supply = _value_or_zero(generation_mix.total.value)
    net_imports = _value_or_zero(exchange.net_imports.value)
    available_generation = total_supply - net_imports
    margin = total_supply - demand_p50
    context = ModelledBalanceContext(
        status=status,
        pressure_ratio=percentage_value(score, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality, label="Modelled balance score"),
        available_generation=power_value(available_generation, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        net_imports=exchange.net_imports,
        supply_margin=power_value(margin, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        import_requirement=power_value(max(-margin, 0.0), event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
        threshold_config_version=threshold_config_version(),
        source=source,
        quality=quality,
        calculation_inputs=[str(item) for item in balance_config.get("calculation_inputs", [])],
        method=(
            "Historical residual-load distribution plus announced unavailability context; "
            "not EcoWatt and not an operational reserve-margin calculation."
        ),
    )
    contributions = [
        ModelledBalanceContribution(
            component="residual_load_percentile",
            value=percentage_value(percentile, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            weight=residual_weight,
            contribution=residual_weight * percentile,
            source=source,
            quality=quality,
            note="Percentile of forecast demand minus wind and solar against comparable recent history.",
        ),
        ModelledBalanceContribution(
            component="announced_unavailability_ratio",
            value=percentage_value(unavailable_ratio, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
            weight=unavailable_weight,
            contribution=unavailable_weight * unavailable_score,
            source=source,
            quality=quality,
            note="Active announced unavailable generation normalized by forecast demand.",
        ),
    ]
    return context, contributions


def _official_signal(bundle: CurrentStateBundle, target: pd.Timestamp, update_time: datetime, mode: DomainMode) -> OfficialSignal:
    source = _source(
        SourceType.OFFICIAL,
        "EcoWatt official signal",
        target,
        update_time,
        mode,
        dataset_id=bundle.ecowatt.source_id,
    )
    if bundle.ecowatt.frame.empty:
        return OfficialSignal(
            name="EcoWatt",
            scope=Scope.NATIONAL,
            status=Status.UNKNOWN,
            label="EcoWatt unavailable",
            signal_time=target.to_pydatetime(),
            update_time=update_time,
            source=source,
            quality=_quality(Freshness.UNAVAILABLE, Confidence.UNAVAILABLE, Status.UNKNOWN),
            detail=bundle.ecowatt.reason or "No official signal is available for this timestamp.",
        )
    signal = status_at(bundle.ecowatt.frame, target)
    raw = str(signal.get("ecowatt_status") or "unknown").lower()
    status = {"green": Status.NORMAL, "normal": Status.NORMAL, "orange": Status.WATCH, "red": Status.HIGH}.get(
        raw,
        Status.UNKNOWN,
    )
    label = str(signal.get("ecowatt_label") or raw or "EcoWatt unavailable")
    return OfficialSignal(
        name="EcoWatt",
        scope=Scope.NATIONAL,
        status=status,
        label=label,
        signal_time=_to_utc_timestamp(signal.get("timestamp") or target).to_pydatetime(),
        update_time=update_time,
        source=source,
        quality=_quality(Freshness.FRESH, Confidence.HIGH, status),
        detail=str(signal.get("ecowatt_message") or ""),
    )


def _national_generation_mix(mix: EstimatedGenerationMix, carbon: CarbonEstimate) -> GenerationMix:
    estimates = [
        GenerationEstimate(
            technology=item.component,
            power=item.value,
            share=percentage_value(
                _value_or_zero(item.value.value) / _value_or_zero(mix.total.value) if _value_or_zero(mix.total.value) > 0 else None,
                event_time=item.value.event_time,
                update_time=item.value.update_time,
                source=item.value.source,
                quality=item.value.quality,
            ),
        )
        for item in mix.components
        if item.included_in_total
    ]
    return GenerationMix(total=mix.total, estimates=estimates, co2_intensity=carbon.intensity)


def _baseline_definition() -> BaselineDefinition:
    return BaselineDefinition(
        baseline_id="usual-demand-comparable-history",
        version="usual-demand.v1",
        method="median demand for comparable recent local hour and day type",
        comparison_keys=["local_hour", "day_type"],
        lookback_days=28,
    )


def _anomaly_percentage(
    observed: QuantifiedValue,
    usual: QuantifiedValue,
    update_time: datetime,
    mode: DomainMode,
    quality: DataQuality,
) -> QuantifiedValue:
    source = _source(SourceType.MODEL, "Demand anomaly calculation", observed.event_time, update_time, mode)
    if observed.value is None or usual.value in {None, 0}:
        value = None
    else:
        value = (float(observed.value) - float(usual.value)) / float(usual.value)
    return percentage_value(value, event_time=observed.event_time, update_time=update_time, source=source, quality=quality)


def _confidence(source: DataProvenance, quality: DataQuality, rationale: str) -> ConfidenceAssessment:
    return ConfidenceAssessment(confidence=quality.confidence, rationale=rationale, source=source, quality=quality)


def _provenance_chain(*items: Any) -> list[DataProvenance]:
    result: list[DataProvenance] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        candidates: list[DataProvenance] = []
        if isinstance(item, DataProvenance):
            candidates = [item]
        elif isinstance(item, TwinComponentEstimate):
            candidates = [item.value.source]
        elif isinstance(item, ExchangeEstimate):
            candidates = [item.source]
        elif isinstance(item, CarbonEstimate):
            candidates = [item.source]
        elif isinstance(item, ModelledBalanceContext):
            candidates = [item.source]
        elif isinstance(item, OfficialSignal):
            candidates = [item.source]
        elif isinstance(item, ForecastInterval):
            candidates = [item.p50.source]
        for source in candidates:
            key = (source.name, source.source_type.value, source.mode.value)
            if key not in seen:
                seen.add(key)
                result.append(source)
    return result


def _forecast_points_by_horizon(points: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in points:
        horizon = _finite_float(item.get("horizon_hours"))
        if horizon is not None:
            result[int(horizon)] = dict(item)
    return result


def _prepare_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "timestamp" not in result and "event_time" in result:
        result["timestamp"] = result["event_time"]
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    return result


def _generation_history(national: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    source = national if not national.empty else hourly
    if source.empty:
        return pd.DataFrame()
    frame = source.copy()
    if "timestamp" not in frame and "event_time" in frame:
        frame["timestamp"] = frame["event_time"]
    if "timestamp" not in frame:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    for column in GENERATION_HISTORY_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = [column for column in GENERATION_HISTORY_COLUMNS if column in frame]
    if not numeric:
        return pd.DataFrame(columns=["timestamp"])
    return frame.set_index("timestamp")[numeric].resample("1h").mean().reset_index()


def _regional_history(regional_history: pd.DataFrame, regional_snapshot: pd.DataFrame) -> pd.DataFrame:
    source = regional_history if not regional_history.empty else regional_snapshot
    if source.empty:
        return pd.DataFrame()
    frame = source.copy()
    if "timestamp" not in frame:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if "region_code" not in frame:
        return pd.DataFrame()
    frame["region_code"] = frame["region_code"].astype(str)
    frame["consumption_mw"] = pd.to_numeric(frame.get("consumption_mw"), errors="coerce")
    return frame.dropna(subset=["consumption_mw"]).sort_values(["region_code", "timestamp"], kind="stable")


def _regional_comparable_value(frame: pd.DataFrame, code: str, target: pd.Timestamp) -> float | None:
    if frame.empty:
        return None
    subset = frame[frame["region_code"].astype(str).eq(str(code))].copy()
    if subset.empty:
        return None
    comparable = _comparable_subset(subset, target)
    if comparable.empty:
        return None
    return _finite_float(comparable["consumption_mw"].median())


def _regional_latest_value(frame: pd.DataFrame, code: str) -> float | None:
    if frame.empty:
        return None
    subset = frame[frame["region_code"].astype(str).eq(str(code))].sort_values("timestamp")
    if subset.empty:
        return None
    return _finite_float(subset.iloc[-1].get("consumption_mw"))


def _weather_scaled_generation(
    component: str,
    column: str,
    target: pd.Timestamp,
    hourly: pd.DataFrame,
    history: pd.DataFrame,
) -> float | None:
    base = _comparable_median(history, target, column)
    if base is None:
        return None
    if component == "wind":
        target_weather = _target_feature(hourly, target, ("weather_wind_speed_kmh", "wind_speed_kmh"))
        recent_weather = _comparable_median(hourly, target, "weather_wind_speed_kmh")
        if recent_weather is None:
            recent_weather = _comparable_median(hourly, target, "wind_speed_kmh")
        if target_weather is None or recent_weather in {None, 0}:
            return base
        return max(float(base) * min(max(float(target_weather) / float(recent_weather), 0.25), 2.0), 0.0)
    target_radiation = _target_feature(hourly, target, ("weather_solar_radiation_wm2", "solar_radiation_wm2"))
    recent_radiation = _comparable_median(hourly, target, "weather_solar_radiation_wm2")
    if recent_radiation is None:
        recent_radiation = _comparable_median(hourly, target, "solar_radiation_wm2")
    if target_radiation is None or recent_radiation in {None, 0}:
        return base
    return max(float(base) * min(max(float(target_radiation) / float(recent_radiation), 0.0), 1.8), 0.0)


def _residual_bucket_value(history: pd.DataFrame, target: pd.Timestamp) -> float | None:
    if history.empty:
        return None
    residual_columns = [column for column in ("hydro_mw", "gas_mw", "coal_mw", "oil_mw", "bioenergy_mw", "net_imports_mw") if column in history]
    if not residual_columns:
        return None
    work = history.copy()
    work["residual_bucket_mw"] = work[residual_columns].sum(axis=1, min_count=1)
    comparable = _comparable_subset(work, target)
    if comparable.empty:
        comparable = work
    value = _finite_float(comparable["residual_bucket_mw"].median())
    return None if value is None else max(value, 0.0)


def _residual_load_percentile(history: pd.DataFrame, target: pd.Timestamp, value: float) -> float:
    if history.empty or not {"consumption_mw", "wind_mw", "solar_mw"}.issubset(history.columns):
        return 0.5
    work = history.copy()
    work["residual_load_mw"] = (
        pd.to_numeric(work["consumption_mw"], errors="coerce")
        - pd.to_numeric(work["wind_mw"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["solar_mw"], errors="coerce").fillna(0.0)
    )
    comparable = _comparable_subset(work.dropna(subset=["residual_load_mw"]), target)
    if comparable.empty:
        comparable = work.dropna(subset=["residual_load_mw"])
    if comparable.empty:
        return 0.5
    distribution = pd.to_numeric(comparable["residual_load_mw"], errors="coerce").dropna()
    if distribution.empty:
        return 0.5
    return float((distribution <= value).mean())


def _comparable_subset(frame: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame:
        return pd.DataFrame()
    local = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(settings.timezone)
    target_local = target.tz_convert(settings.timezone)
    same_hour = local.dt.hour.eq(target_local.hour)
    same_day_type = local.dt.dayofweek.ge(5).eq(target_local.dayofweek >= 5)
    return frame.loc[same_hour & same_day_type]


def _comparable_median(frame: pd.DataFrame, target: pd.Timestamp, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    comparable = _comparable_subset(frame, target)
    if comparable.empty:
        comparable = frame
    values = pd.to_numeric(comparable[column], errors="coerce").dropna()
    if values.empty:
        return None
    return _finite_float(values.tail(28).median())


def _current_hour_value(frame: pd.DataFrame, target: pd.Timestamp, column: str) -> float | None:
    if frame.empty or column not in frame or "timestamp" not in frame:
        return None
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    exact = work[work["timestamp"].eq(target)]
    if not exact.empty:
        return _finite_float(exact.iloc[-1].get(column))
    previous = work[work["timestamp"].le(target)]
    if previous.empty:
        return None
    return _finite_float(previous.iloc[-1].get(column))


def _has_exact_observed(frame: pd.DataFrame, target: pd.Timestamp) -> bool:
    if frame.empty or "timestamp" not in frame:
        return False
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return bool(timestamps.eq(target).any())


def _official_component_value(frame: pd.DataFrame, target: pd.Timestamp, column: str) -> float | None:
    if frame.empty or column not in frame or "timestamp" not in frame:
        return None
    work = _timestamped_frame(frame)
    rows = work[work["timestamp"].eq(target)]
    if rows.empty:
        return None
    return _finite_float(rows.iloc[-1].get(column))


def _target_feature(frame: pd.DataFrame, target: pd.Timestamp, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        value = _current_hour_value(frame, target, column)
        if value is not None:
            return value
    return None


def _unavailability_components(
    frame: pd.DataFrame,
    target: pd.Timestamp,
    update_time: datetime,
    mode: DomainMode,
) -> list[TwinComponentEstimate]:
    if frame.empty:
        return []
    active = _active_unavailability_rows(frame, target)
    if active.empty:
        return []
    source = _source(SourceType.OFFICIAL, "RTE announced generation unavailability", target, update_time, mode)
    quality = _quality(Freshness.FRESH, Confidence.HIGH, Status.UNKNOWN)
    components: list[TwinComponentEstimate] = []
    for row in active.itertuples(index=False):
        technology = str(getattr(row, "technology", getattr(row, "component", "generation")))
        value = _finite_float(getattr(row, "unavailable_mw", None)) or 0.0
        components.append(
            TwinComponentEstimate(
                component=f"{technology}_announced_unavailable",
                value=power_value(value, event_time=target.to_pydatetime(), update_time=update_time, source=source, quality=quality),
                provenance_kind=EstimateProvenanceKind.OFFICIAL_FORECAST,
                included_in_total=False,
                formula="Active announced unavailable MW from optional RTE data.",
            )
        )
    return components


def _active_unavailable_mw(frame: pd.DataFrame, target: pd.Timestamp, *, technology: str | None = None) -> float:
    active = _active_unavailability_rows(frame, target)
    if active.empty:
        return 0.0
    if technology:
        tech = active.get("technology", active.get("component", pd.Series("", index=active.index))).astype(str).str.lower()
        active = active[tech.str.contains(technology.lower(), na=False)]
    if active.empty or "unavailable_mw" not in active:
        return 0.0
    return float(pd.to_numeric(active["unavailable_mw"], errors="coerce").fillna(0.0).sum())


def _active_unavailability_rows(frame: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = _timestamped_unavailability_frame(frame)
    if work.empty:
        return work
    start = work.get("start_time", pd.Series(pd.NaT, index=work.index))
    end = work.get("end_time", pd.Series(pd.NaT, index=work.index))
    start = pd.to_datetime(start, utc=True, errors="coerce").fillna(work["timestamp"])
    end = pd.to_datetime(end, utc=True, errors="coerce").fillna(work["timestamp"] + pd.Timedelta(hours=1))
    return work.loc[start.le(target) & end.gt(target)].copy()


def _timestamped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if "timestamp" not in result:
        for column in ("event_time", "target_timestamp", "date_heure"):
            if column in result:
                result["timestamp"] = result[column]
                break
    if "timestamp" not in result:
        return pd.DataFrame()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    return result.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")


def _timestamped_unavailability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if "unavailable_mw" not in result:
        for column in ("capacity_unavailable_mw", "unavailability_mw", "power_mw"):
            if column in result:
                result["unavailable_mw"] = result[column]
                break
    if "technology" not in result and "component" in result:
        result["technology"] = result["component"]
    if "timestamp" not in result:
        for column in ("start_time", "event_time", "target_timestamp"):
            if column in result:
                result["timestamp"] = result[column]
                break
    return _timestamped_frame(result)


def _requested_origin(from_timestamp: str | datetime | None, hourly: pd.DataFrame) -> pd.Timestamp:
    if from_timestamp is None:
        return pd.Timestamp(hourly["timestamp"].max()).floor("h")
    return _to_utc_timestamp(from_timestamp).floor("h")


def _effective_origin(
    requested_origin: pd.Timestamp,
    hourly: pd.DataFrame,
    forecast_run: Any,
) -> pd.Timestamp:
    if forecast_run is not None and forecast_run.origin:
        return _to_utc_timestamp(forecast_run.origin).floor("h")
    available = hourly[pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce").le(requested_origin)]
    if available.empty:
        raise ValueError("No hourly observation is available at or before the requested twin start timestamp.")
    return pd.Timestamp(available["timestamp"].max()).floor("h")


def _snapshot_mode(horizon: int) -> DomainMode:
    if settings.is_demo_mode:
        return DomainMode.REPLAY
    return DomainMode.LIVE if horizon == 0 else DomainMode.FORECAST


def _quality_for_mode(mode: DomainMode, forecast: bool) -> DataQuality:
    if mode == DomainMode.REPLAY:
        return _quality(Freshness.FRESH, Confidence.LOW if forecast else Confidence.MEDIUM, Status.UNKNOWN)
    return _quality(Freshness.FRESH, Confidence.MEDIUM if forecast else Confidence.HIGH, Status.UNKNOWN)


def _quality(freshness: Freshness, confidence: Confidence, status: Status) -> DataQuality:
    return DataQuality(freshness=freshness, confidence=confidence, status=status, checked_at=utc_now())


def _source(
    source_type: SourceType,
    name: str,
    event_time: pd.Timestamp | datetime,
    update_time: datetime,
    mode: DomainMode,
    *,
    fallback: bool = False,
    fallback_reason: str | None = None,
    dataset_id: str | None = None,
    transformation: str | None = None,
) -> DataProvenance:
    is_demo = settings.is_demo_mode
    return DataProvenance(
        source_type=SourceType.FALLBACK if fallback else source_type,
        name=name,
        mode=mode,
        event_time=_to_utc_timestamp(event_time).to_pydatetime(),
        update_time=_ensure_utc_datetime(update_time),
        is_fallback=fallback,
        is_demo=is_demo,
        dataset_id=dataset_id,
        retrieved_at=_ensure_utc_datetime(update_time),
        transformation=transformation,
        fallback_reason=fallback_reason if fallback else None,
        replay_label="historical replay/demo mode" if is_demo or mode == DomainMode.REPLAY else None,
    )


def _ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number in {float("inf"), float("-inf")}:
        return None
    return number


def _value_or_zero(value: Any) -> float:
    number = _finite_float(value)
    return 0.0 if number is None else float(number)


default_twin_service = TwinService()
