"""Current-state backend service for the public Now page.

The service is intentionally framework-neutral. It builds contracts from the
same public-data clients and usual-demand baseline code used by the Streamlit
prototype, while adding an in-process refresh cache so browser traffic does not
translate into one upstream API call per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.config import settings
from src.contracts.energy_twin import (
    CacheInfo,
    CurrentDemandContext,
    CurrentGenerationMix,
    CurrentMapRegion,
    CurrentModelledStatus,
    CurrentOfficialSignal,
    CurrentStateResponse,
    DataHealthResponse,
    EnvironmentalMetric,
    FreshnessStatus,
    GenerationTechnologyMetric,
    ModelHealth,
    NationalCurrentContext,
    NullableMetric,
    OperatingState,
    RegionalCurrentContext,
    SourceHealth,
    SourceMetadata,
    ScenarioEngineHealth,
    SourcesResponse,
    StatusThresholdsResponse,
    Status,
    UnavailableField,
)
from src.contracts.status_thresholds import (
    balance_status_for_ratio,
    load_status_thresholds,
    status_label,
    threshold_config_version,
)
from src.data_processing.clean_energy_mix import clean_energy_mix
from src.data_processing.storage import PartitionedParquetStore
from src.data_sources import ecowatt as ecowatt_source
from src.data_sources import rte_eco2mix, rte_eco2mix_regional
from src.data_sources.ecowatt import load_ecowatt_window, status_at
from src.data_sources.rte_eco2mix import Eco2MixError, fetch_eco2mix, load_cached_eco2mix
from src.data_sources.rte_eco2mix_regional import (
    REGION_NAMES,
    RegionalEco2MixError,
    demo_regional_snapshot,
    fetch_regional_eco2mix,
    load_cached_regional_eco2mix,
    prepare_regional_snapshot,
    region_code as code_for_region_name,
)
from src.demo_mode import demo_energy, external_api_enabled, read_demo_json, read_demo_parquet
from src.models.usual_demand import (
    BaselineConfig,
)


SOURCE_REFRESH_INTERVAL_SECONDS = 900
FRESH_LIVE_MAX_AGE = timedelta(minutes=45)
DELAYED_LIVE_MAX_AGE = timedelta(hours=6)
LAST_KNOWN_GOOD_MAX_AGE = timedelta(hours=24)
SCENARIO_ENGINE_HEALTH_VERSION = "scenario-engine.v1"
SCENARIO_ASSUMPTION_HEALTH_VERSION = "scenario-assumptions.v1"

GENERATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("nuclear", "nuclear_mw"),
    ("wind", "wind_mw"),
    ("solar", "solar_mw"),
    ("hydro", "hydro_mw"),
    ("gas", "gas_mw"),
    ("coal", "coal_mw"),
    ("oil", "oil_mw"),
    ("bioenergy", "bioenergy_mw"),
)


@dataclass(frozen=True)
class SourceTable:
    frame: pd.DataFrame
    source_id: str
    name: str
    operating_state: OperatingState
    source_quality: str
    retrieved_at: datetime | None = None
    reason: str | None = None


@dataclass(frozen=True)
class CurrentStateBundle:
    national: SourceTable
    regional_snapshot: SourceTable
    regional_history: pd.DataFrame
    ecowatt: SourceTable


@dataclass(frozen=True)
class _CachedBundle:
    bundle: CurrentStateBundle
    generated_at: datetime
    expires_at: datetime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CurrentStateService:
    """Build and cache Now-page API contracts."""

    def __init__(
        self,
        *,
        loader: Callable[[datetime], CurrentStateBundle] | None = None,
        now: Callable[[], datetime] = utc_now,
        cache_ttl_seconds: int = SOURCE_REFRESH_INTERVAL_SECONDS,
        baseline_config: BaselineConfig | None = None,
    ) -> None:
        self._loader = loader or load_current_state_bundle
        self._now = now
        self._cache_ttl_seconds = int(cache_ttl_seconds)
        self._baseline_config = baseline_config or BaselineConfig()
        self._cache: _CachedBundle | None = None
        self._last_cache_hit = False

    def clear_cache(self) -> None:
        self._cache = None
        self._last_cache_hit = False

    def get_current_state(self, region: str) -> CurrentStateResponse:
        region_id = normalize_region_code(region)
        now = _ensure_utc_datetime(self._now())
        bundle = self._bundle(now)
        cache = self._cache_info(now)
        usual = _usual_demand_rows(bundle.national.frame, bundle.regional_history, config=self._baseline_config)

        national_freshness = _freshness_for_table(bundle.national, now)
        national = _national_context(bundle, usual, national_freshness, now)
        selected_freshness = _selected_region_freshness(bundle.regional_snapshot, region_id, now)
        selected = _selected_region_context(bundle.regional_snapshot, usual, region_id, selected_freshness)
        map_regions = _map_regions(bundle.regional_snapshot, usual)
        unavailable = _unavailable_fields(national, selected, map_regions)
        operating_state = _response_operating_state(national.freshness, selected.freshness)

        return CurrentStateResponse(
            generated_at=now,
            region=region_id,
            operating_state=operating_state,
            cache=cache,
            national_context=national,
            selected_region_context=selected,
            map=map_regions,
            unavailable_fields=unavailable,
        )

    def get_data_health(self) -> DataHealthResponse:
        now = _ensure_utc_datetime(self._now())
        bundle = self._bundle(now)
        cache = self._cache_info(now)
        sources = [
            _source_health(bundle.national, now),
            _source_health(bundle.regional_snapshot, now),
            _source_health(bundle.ecowatt, now),
        ]
        unavailable = [
            UnavailableField(field=f"sources.{source.source_id}", reason=source.reason or "Source unavailable.")
            for source in sources
            if source.operating_state == OperatingState.SOURCE_UNAVAILABLE
        ]
        operating_state = _health_operating_state(sources)
        return DataHealthResponse(
            generated_at=now,
            operating_state=operating_state,
            cache=cache,
            sources=sources,
            model_health=_model_health(now),
            scenario_engine=_scenario_engine_health(),
            unavailable_fields=unavailable,
        )

    def get_sources(self) -> SourcesResponse:
        now = _ensure_utc_datetime(self._now())
        return SourcesResponse(generated_at=now, sources=source_catalog())

    def get_status_thresholds(self) -> StatusThresholdsResponse:
        now = _ensure_utc_datetime(self._now())
        config = load_status_thresholds()
        return StatusThresholdsResponse(
            generated_at=now,
            version=str(config["version"]),
            thresholds={
                "balance_pressure_ratio": dict(config.get("balance_pressure_ratio", {})),
                "visual_pressure_score": dict(config.get("visual_pressure_score", {})),
            },
            calculation_inputs=[str(item) for item in config.get("calculation_inputs", [])],
            excluded_inputs=[str(item) for item in config.get("excluded_inputs", [])],
            raw_config=config,
        )

    def _bundle(self, now: datetime) -> CurrentStateBundle:
        if self._cache is not None and now < self._cache.expires_at:
            self._last_cache_hit = True
            return self._cache.bundle
        bundle = self._loader(now)
        generated_at = now
        self._cache = _CachedBundle(
            bundle=bundle,
            generated_at=generated_at,
            expires_at=generated_at + timedelta(seconds=self._cache_ttl_seconds),
        )
        self._last_cache_hit = False
        return bundle

    def _cache_info(self, now: datetime) -> CacheInfo:
        if self._cache is None:
            generated_at = now
            expires_at = now + timedelta(seconds=self._cache_ttl_seconds)
        else:
            generated_at = self._cache.generated_at
            expires_at = self._cache.expires_at
        return CacheInfo(
            cache_key="current-state-source-bundle",
            ttl_seconds=self._cache_ttl_seconds,
            generated_at=generated_at,
            expires_at=expires_at,
            cache_hit=self._last_cache_hit,
        )


def load_current_state_bundle(now: datetime | None = None) -> CurrentStateBundle:
    """Load source data for the current-state API with source-level fallbacks."""

    now = _ensure_utc_datetime(now or utc_now())
    if settings.is_demo_mode and not external_api_enabled():
        return _load_demo_bundle(now)
    return _load_live_bundle(now)


def normalize_region_code(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized.startswith("FR-"):
        normalized = normalized[3:]
    if normalized not in REGION_NAMES:
        raise ValueError(f"unknown French region code: {value}")
    return normalized


def source_catalog() -> list[SourceMetadata]:
    return [
        SourceMetadata(
            source_id="odre_eco2mix_national",
            name="RTE eCO2mix national",
            source_type="official_observed",
            dataset_id=rte_eco2mix.DATASET_ID,
            url="https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/",
            required_for_now=True,
            credential_required=False,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            notes="Primary national demand, generation, CO2 intensity, and physical exchanges.",
        ),
        SourceMetadata(
            source_id="odre_eco2mix_regional",
            name="RTE eCO2mix regional",
            source_type="official_observed",
            dataset_id=rte_eco2mix_regional.DATASET_ID,
            url=rte_eco2mix_regional.REGIONAL_ODRE_URL,
            required_for_now=True,
            credential_required=False,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            notes="Regional demand, local generation, and regional physical exchange context.",
        ),
        SourceMetadata(
            source_id="odre_ecowatt",
            name="EcoWatt ODRE public history",
            source_type="official_signal",
            dataset_id=settings.ecowatt_current_dataset_id,
            url=settings.ecowatt_current_odre_url,
            required_for_now=False,
            credential_required=False,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            notes="Official national EcoWatt signal when a current public record is available.",
        ),
        SourceMetadata(
            source_id="rte_ecowatt_api",
            name="RTE EcoWatt live API",
            source_type="optional_official_signal",
            dataset_id=None,
            url=settings.rte_ecowatt_api_url,
            required_for_now=False,
            credential_required=True,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            notes="Optional token-gated enhancement; unavailable values are not replaced with modelled signals.",
        ),
        SourceMetadata(
            source_id="usual_demand_baseline",
            name="Usual-demand comparable-history baseline",
            source_type="derived_baseline",
            dataset_id=None,
            url="docs/usual-demand-pipeline.md",
            required_for_now=True,
            credential_required=False,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            notes="Transparent baseline from normalized public records; exposes fallback level and sample count.",
        ),
        SourceMetadata(
            source_id="status_thresholds",
            name="Modelled status thresholds",
            source_type="configuration",
            dataset_id="balance-pressure-thresholds",
            url="data/config/status_thresholds.json",
            required_for_now=True,
            credential_required=False,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            notes="Documented modelled balance status. It is separate from official EcoWatt.",
        ),
    ]


def _load_demo_bundle(now: datetime) -> CurrentStateBundle:
    energy = demo_energy()
    latest = _latest_timestamp(energy) or pd.Timestamp(now)
    regional = demo_regional_snapshot(latest.floor("h"))
    regional_history = _synthesize_regional_history(regional)
    ecowatt = read_demo_parquet(settings.demo_ecowatt_path)
    ecowatt_state = (
        OperatingState.HISTORICAL_REPLAY
        if not ecowatt.empty
        else OperatingState.SOURCE_UNAVAILABLE
    )
    return CurrentStateBundle(
        national=SourceTable(
            frame=energy,
            source_id="odre_eco2mix_national",
            name="Bundled eCO2mix historical replay",
            operating_state=OperatingState.HISTORICAL_REPLAY if not energy.empty else OperatingState.SOURCE_UNAVAILABLE,
            source_quality="historical_replay" if not energy.empty else "unavailable",
            retrieved_at=now,
            reason="Bundled demo replay sample anchored to the presentation date.",
        ),
        regional_snapshot=SourceTable(
            frame=regional,
            source_id="odre_eco2mix_regional",
            name="Bundled regional eCO2mix historical replay",
            operating_state=OperatingState.HISTORICAL_REPLAY,
            source_quality="historical_replay",
            retrieved_at=now,
            reason="Bundled demo replay sample anchored to the presentation date.",
        ),
        regional_history=regional_history,
        ecowatt=SourceTable(
            frame=ecowatt,
            source_id="odre_ecowatt",
            name="Bundled EcoWatt replay",
            operating_state=ecowatt_state,
            source_quality="historical_replay" if not ecowatt.empty else "unavailable",
            retrieved_at=now if not ecowatt.empty else None,
            reason=None if not ecowatt.empty else "Bundled EcoWatt replay has no current record.",
        ),
    )


def _load_live_bundle(now: datetime) -> CurrentStateBundle:
    start = now - timedelta(hours=settings.history_hours)
    national = _load_live_national(start, now)
    regional, regional_history = _load_live_regional(start, now)
    reference_time = _latest_timestamp(national.frame) or pd.Timestamp(now)
    ecowatt_start = reference_time.floor("h") - pd.Timedelta(hours=1)
    ecowatt_end = reference_time.floor("h") + pd.Timedelta(hours=49)
    ecowatt = _load_live_ecowatt(ecowatt_start, ecowatt_end, now)
    return CurrentStateBundle(
        national=national,
        regional_snapshot=regional,
        regional_history=regional_history,
        ecowatt=ecowatt,
    )


def _load_live_national(start: datetime, end: datetime) -> SourceTable:
    try:
        raw = fetch_eco2mix(start=start, end=end)
        clean = clean_energy_mix(raw).sort_values("timestamp")
        PartitionedParquetStore(settings.energy_store_dir).upsert(clean)
        return SourceTable(
            frame=clean,
            source_id="odre_eco2mix_national",
            name="RTE eCO2mix national",
            operating_state=OperatingState.FRESH_LIVE_DATA,
            source_quality="validated",
            retrieved_at=end,
        )
    except (Eco2MixError, OSError, ValueError):
        pass

    try:
        stored = PartitionedParquetStore(settings.energy_store_dir).read(start=start, end=end)
        if not stored.empty:
            return SourceTable(
                frame=stored.sort_values("timestamp"),
                source_id="odre_eco2mix_national",
                name="RTE eCO2mix national cached snapshot",
                operating_state=OperatingState.LAST_KNOWN_GOOD_FALLBACK,
                source_quality="last_known_good",
                retrieved_at=end,
                reason="Live eCO2mix refresh failed; using processed last-known-good data.",
            )
    except (OSError, ValueError):
        pass

    try:
        cached = clean_energy_mix(load_cached_eco2mix()).sort_values("timestamp")
        return SourceTable(
            frame=cached,
            source_id="odre_eco2mix_national",
            name="RTE eCO2mix raw cached snapshot",
            operating_state=OperatingState.LAST_KNOWN_GOOD_FALLBACK,
            source_quality="last_known_good",
            retrieved_at=end,
            reason="Live eCO2mix refresh failed; using raw last-known-good data.",
        )
    except (Eco2MixError, OSError, ValueError, FileNotFoundError):
        return SourceTable(
            frame=pd.DataFrame(),
            source_id="odre_eco2mix_national",
            name="RTE eCO2mix national",
            operating_state=OperatingState.SOURCE_UNAVAILABLE,
            source_quality="unavailable",
            reason="No live or cached national eCO2mix data is available.",
        )


def _load_live_regional(start: datetime, end: datetime) -> tuple[SourceTable, pd.DataFrame]:
    try:
        raw = fetch_regional_eco2mix(start=start, end=end)
        snapshot = prepare_regional_snapshot(raw)
        return (
            SourceTable(
                frame=snapshot,
                source_id="odre_eco2mix_regional",
                name="RTE eCO2mix regional",
                operating_state=OperatingState.FRESH_LIVE_DATA,
                source_quality="validated",
                retrieved_at=end,
            ),
            _regional_history_from_raw(raw),
        )
    except (RegionalEco2MixError, OSError, ValueError):
        pass

    try:
        raw = load_cached_regional_eco2mix()
        snapshot = prepare_regional_snapshot(raw)
        return (
            SourceTable(
                frame=snapshot,
                source_id="odre_eco2mix_regional",
                name="RTE eCO2mix regional cached snapshot",
                operating_state=OperatingState.LAST_KNOWN_GOOD_FALLBACK,
                source_quality="last_known_good",
                retrieved_at=end,
                reason="Live regional eCO2mix refresh failed; using cached regional data.",
            ),
            _regional_history_from_raw(raw),
        )
    except (RegionalEco2MixError, OSError, ValueError, FileNotFoundError):
        return (
            SourceTable(
                frame=pd.DataFrame(),
                source_id="odre_eco2mix_regional",
                name="RTE eCO2mix regional",
                operating_state=OperatingState.SOURCE_UNAVAILABLE,
                source_quality="unavailable",
                reason="No live or cached regional eCO2mix data is available.",
            ),
            pd.DataFrame(),
        )


def _load_live_ecowatt(start: pd.Timestamp, end: pd.Timestamp, now: datetime) -> SourceTable:
    try:
        frame, label = load_ecowatt_window(start, end, timezone_name=settings.timezone)
    except (OSError, ValueError):
        frame, label = pd.DataFrame(), "EcoWatt unavailable"
    if frame.empty:
        return SourceTable(
            frame=frame,
            source_id="odre_ecowatt",
            name="EcoWatt",
            operating_state=OperatingState.SOURCE_UNAVAILABLE,
            source_quality="unavailable",
            reason="No current official EcoWatt signal is available.",
        )
    return SourceTable(
        frame=frame,
        source_id="odre_ecowatt",
        name=label,
        operating_state=OperatingState.FRESH_LIVE_DATA,
        source_quality="validated",
        retrieved_at=now,
    )


def _national_context(
    bundle: CurrentStateBundle,
    usual: pd.DataFrame,
    freshness: FreshnessStatus,
    now: datetime,
) -> NationalCurrentContext:
    row = _latest_row(bundle.national.frame)
    source_quality = _quality_for_freshness(freshness, bundle.national.source_quality)
    if row is None:
        event_time = now
        demand = _demand_context(None, None, "national", source_quality)
        generation_mix = _generation_mix(None, source_quality)
        imports = _metric(None, "MW", "National physical imports are unavailable.", source_quality)
        exports = _metric(None, "MW", "National physical exports are unavailable.", source_quality)
        net_imports = _metric(None, "MW", "National net imports are unavailable.", source_quality)
        carbon = _carbon_metric(None, source_quality)
    else:
        event_time = _row_timestamp(row) or pd.Timestamp(now)
        usual_row = _usual_row(usual, scope="national", region_code=None)
        demand = _demand_context(row, usual_row, "national", source_quality)
        generation_mix = _generation_mix(row, source_quality)
        imports = _row_metric(row, "imports_mw", "MW", "National physical imports are unavailable.", source_quality)
        exports = _row_metric(row, "exports_mw", "MW", "National physical exports are unavailable.", source_quality)
        net_imports = _row_metric(row, "net_imports_mw", "MW", "National net imports are unavailable.", source_quality)
        carbon = _carbon_metric(row, source_quality)

    ecowatt = _official_ecowatt_signal(bundle.ecowatt, event_time)
    modelled = _modelled_status(row)
    return NationalCurrentContext(
        demand=demand,
        freshness=freshness,
        generation_mix=generation_mix,
        physical_imports=imports,
        physical_exports=exports,
        net_imports=net_imports,
        carbon_estimate=carbon,
        official_ecowatt_signal=ecowatt,
        modelled_status=modelled,
    )


def _selected_region_context(
    regional: SourceTable,
    usual: pd.DataFrame,
    region_id: str,
    freshness: FreshnessStatus,
) -> RegionalCurrentContext:
    region_name = REGION_NAMES[region_id]
    row = _region_row(regional.frame, region_id)
    source_quality = _quality_for_freshness(freshness, regional.source_quality)
    usual_row = _usual_row(usual, scope="regional", region_code=region_id)
    demand = _demand_context(row, usual_row, "regional", source_quality)
    return RegionalCurrentContext(
        region_code=region_id,
        region_name=region_name,
        demand=demand,
        freshness=freshness,
        local_generation=_generation_mix(row, source_quality),
        net_flow=_row_metric(row, "net_imports_mw", "MW", "Regional net physical flow is unavailable.", source_quality),
        physical_balance=_row_metric(
            row,
            "physical_balance_mw",
            "MW",
            "Regional physical balance is not available in this source record.",
            source_quality,
        ),
        connected_grid_note=(
            "Regional context is not an independent shortage warning. Regional values are connected-grid context only; "
            "local generation is not a measure of electricity available only inside the region, and low local generation "
            "does not indicate a regional electricity shortage."
        ),
    )


def _map_regions(regional: SourceTable, usual: pd.DataFrame) -> list[CurrentMapRegion]:
    rows = []
    for region_id, region_name in REGION_NAMES.items():
        row = _region_row(regional.frame, region_id)
        source_quality = regional.source_quality if row is not None else "missing_region_record"
        usual_row = _usual_row(usual, scope="regional", region_code=region_id)
        observed = _row_metric(
            row,
            "consumption_mw",
            "MW",
            "No regional demand record is available for this region.",
            source_quality,
        )
        usual_metric = _usual_metric(usual_row, source_quality)
        anomaly = _anomaly_metric(observed.value, usual_metric.value, source_quality)
        rows.append(
            CurrentMapRegion(
                region_id=region_id,
                region_name=region_name,
                demand_anomaly_pct=anomaly,
                observed_demand=observed,
                usual_demand=usual_metric,
                source_quality=source_quality,
                availability_flag=anomaly.value is not None,
            )
        )
    return rows


def _source_health(table: SourceTable, now: datetime) -> SourceHealth:
    freshness = _freshness_for_table(table, now)
    fallback_records = len(table.frame) if table.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK else 0
    return SourceHealth(
        source_id=table.source_id,
        name=table.name,
        operating_state=freshness.state,
        freshness=freshness,
        source_quality=_quality_for_freshness(freshness, table.source_quality),
        missing_intervals=_missing_interval_count(table.frame, expected_interval=timedelta(minutes=15)),
        fallback_records=fallback_records,
        adapter_failures=1 if table.operating_state == OperatingState.SOURCE_UNAVAILABLE else 0,
        circuit_breaker_state="open_or_fallback" if table.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK else "closed",
        latest_successful_fetch_at=table.retrieved_at if table.operating_state != OperatingState.SOURCE_UNAVAILABLE else None,
        reason=table.reason or freshness.reason,
    )


def _missing_interval_count(frame: pd.DataFrame, *, expected_interval: timedelta) -> int:
    if frame.empty or "timestamp" not in frame:
        return 0
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(timestamps) <= 1:
        return 0
    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(expected_interval)
    positive = deltas[deltas > pd.Timedelta(0)]
    if not positive.empty:
        expected = max(expected, pd.Timedelta(positive.median()))
    return int(sum(max(int(delta / expected) - 1, 0) for delta in deltas if delta > expected))


def _model_health(now: datetime) -> ModelHealth:
    evaluation_path = (
        settings.demo_model_evaluation_path
        if settings.is_demo_mode
        else settings.processed_dir / "demand_model" / "evaluation.json"
    )
    forecast_path = (
        settings.demo_model_forecast_path
        if settings.is_demo_mode
        else settings.processed_dir / "demand_forecast" / "latest_forecast.json"
    )
    registry_path = settings.processed_dir / "demand_forecast" / "artifact_manifest.json"
    evaluation = _read_json_dict(evaluation_path)
    forecast = _read_json_dict(forecast_path)
    registry = _read_json_dict(registry_path)

    model_version = (
        _string_or_none(registry.get("model_version"))
        or _string_or_none(evaluation.get("model_version"))
        or _string_or_none(evaluation.get("model_kind"))
    )
    latest_forecast_at = _datetime_or_none(forecast.get("generated_at")) or _datetime_or_none(
        evaluation.get("generated_at")
    )
    forecast_rows = forecast.get("forecasts") if isinstance(forecast.get("forecasts"), list) else []
    run_id = _forecast_health_run_id(forecast) if forecast_rows else None
    error = _recent_forecast_error(evaluation, registry)
    fallback_usage = None
    status = _string_or_none(registry.get("status"))
    if not status:
        status = "historical_replay" if settings.is_demo_mode and evaluation else "baseline_fallback"
    if not model_version:
        fallback_usage = "usual-demand baseline fallback; no champion model artifact is loaded"
    elif status != "champion":
        fallback_usage = "model artifact is present but not promoted as champion"
    reason = None
    if not evaluation and not registry:
        reason = "No demand-model evaluation or probabilistic forecast registry artifact is available."
    elif settings.is_demo_mode:
        reason = "Model health is read from the presentation-anchored demo replay bundle."
    return ModelHealth(
        model_id="demand-forecast",
        status=status,
        model_version=model_version,
        latest_successful_forecast_at=latest_forecast_at,
        latest_successful_forecast_run_id=run_id,
        recent_forecast_error_mae_mw=error,
        fallback_usage=fallback_usage,
        reason=reason,
    )


def _scenario_engine_health() -> ScenarioEngineHealth:
    return ScenarioEngineHealth(
        available=True,
        version=SCENARIO_ENGINE_HEALTH_VERSION,
        assumption_version=SCENARIO_ASSUMPTION_HEALTH_VERSION,
        cache_enabled=True,
        last_successful_scenario_id=None,
        reason="Scenario engine is deterministic and uses typed twin snapshots as its baseline.",
    )


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if settings.is_demo_mode:
        try:
            path.relative_to(settings.demo_dir)
        except ValueError:
            pass
        else:
            return read_demo_json(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _datetime_or_none(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _forecast_health_run_id(payload: Mapping[str, Any]) -> str:
    stable = json.dumps(payload.get("forecasts", []), sort_keys=True, default=str, separators=(",", ":"))
    return f"forecast-health-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:12]}"


def _recent_forecast_error(evaluation: Mapping[str, Any], registry: Mapping[str, Any]) -> float | None:
    metrics = evaluation.get("metrics")
    if isinstance(metrics, list):
        candidates = [
            item
            for item in metrics
            if isinstance(item, Mapping)
            and str(item.get("model")).lower() in {"demand_hgb", "residual_quantile", "demand_forecast"}
            and item.get("mae_mw") is not None
        ]
        if candidates:
            preferred = min(candidates, key=lambda item: int(float(item.get("horizon_hours") or 999)))
            return _finite_float(preferred.get("mae_mw"))
    overall = registry.get("metrics", {}).get("overall") if isinstance(registry.get("metrics"), Mapping) else {}
    mae_gw = overall.get("mae_gw") if isinstance(overall, Mapping) else None
    mae = _finite_float(mae_gw)
    return None if mae is None else mae * 1000.0


def _usual_demand_rows(national: pd.DataFrame, regional_history: pd.DataFrame, *, config: BaselineConfig) -> pd.DataFrame:
    """Return current comparable-history baselines without building model features."""
    frames: list[pd.DataFrame] = []
    if not national.empty:
        national_frame = national.copy()
        national_frame["geographic_scope"] = "national"
        national_frame["region"] = "France"
        frames.append(national_frame)
    if not regional_history.empty:
        regional_frame = regional_history.copy()
        regional_frame["geographic_scope"] = "regional"
        if "region_display" in regional_frame:
            regional_frame["region"] = regional_frame["region_display"]
        frames.append(regional_frame)
    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "timestamp" not in frame or "consumption_mw" not in frame or "region" not in frame:
        return pd.DataFrame()
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["consumption_mw"] = pd.to_numeric(frame["consumption_mw"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "consumption_mw", "region"])
    if frame.empty:
        return pd.DataFrame()
    local = frame["timestamp"].dt.tz_convert(settings.timezone)
    frame["hour_of_day"] = local.dt.hour.astype(int)
    frame["weekday_type"] = local.dt.dayofweek.map(lambda day: "weekend" if day >= 5 else "weekday")
    frame["season"] = local.dt.month.map(_season_label)
    rows: list[dict[str, Any]] = []
    grouped = frame.sort_values("timestamp", kind="stable").groupby(["geographic_scope", "region"], sort=False)
    for (scope, region), group in grouped:
        latest = group.iloc[-1]
        target = pd.Timestamp(latest["timestamp"])
        history = group.iloc[:-1].copy()
        if config.max_history_days is not None:
            history = history[history["timestamp"].ge(target - pd.Timedelta(days=int(config.max_history_days)))]
        baseline = _current_usual_baseline(latest, history, config)
        usual = _finite_float(baseline["usual_demand_mw"])
        target_mw = _finite_float(latest.get("consumption_mw"))
        if usual is not None and target_mw is not None and usual != 0:
            above = 100.0 * (target_mw - usual) / usual
        else:
            above = None
        rows.append(
            {
                "target_timestamp": target,
                "geographic_scope": str(scope),
                "region": str(region),
                "target_mw": target_mw,
                "usual_demand_mw": usual,
                "above_usual_percent": above,
                "usual_demand_method": baseline["usual_demand_method"],
                "usual_demand_sample_count": baseline["usual_demand_sample_count"],
                "usual_demand_fallback_level": baseline["usual_demand_fallback_level"],
            }
        )
    return pd.DataFrame.from_records(rows)


def _current_usual_baseline(row: pd.Series, history: pd.DataFrame, config: BaselineConfig) -> dict[str, Any]:
    if history.empty:
        return _empty_usual_baseline("no_history")
    rules = (
        (
            1,
            "same_hour_weekday_type_season",
            history[
                history["hour_of_day"].eq(row["hour_of_day"])
                & history["weekday_type"].eq(row["weekday_type"])
                & history["season"].eq(row["season"])
            ],
        ),
        (
            2,
            "same_hour_weekday_type",
            history[
                history["hour_of_day"].eq(row["hour_of_day"])
                & history["weekday_type"].eq(row["weekday_type"])
            ],
        ),
        (3, "same_hour", history[history["hour_of_day"].eq(row["hour_of_day"])]),
    )
    for level, method, sample in rules:
        sample = sample.dropna(subset=["consumption_mw"])
        if len(sample) >= config.min_samples:
            return _usual_from_sample(sample, level=level, method=method)

    target = pd.Timestamp(row["timestamp"])
    recent = history[history["timestamp"].ge(target - pd.Timedelta(days=int(config.recent_days)))]
    sample = recent.dropna(subset=["consumption_mw"])
    if sample.empty:
        sample = history.tail(max(1, config.min_samples)).dropna(subset=["consumption_mw"])
    if sample.empty:
        return _empty_usual_baseline("no_non_null_history")
    return _usual_from_sample(sample, level=4, method="recent_rolling")


def _usual_from_sample(sample: pd.DataFrame, *, level: int, method: str) -> dict[str, Any]:
    values = pd.to_numeric(sample["consumption_mw"], errors="coerce").dropna()
    if values.empty:
        return _empty_usual_baseline("no_non_null_history")
    return {
        "usual_demand_mw": float(values.median()),
        "usual_demand_method": method,
        "usual_demand_sample_count": int(len(values)),
        "usual_demand_fallback_level": int(level),
    }


def _empty_usual_baseline(reason: str) -> dict[str, Any]:
    return {
        "usual_demand_mw": None,
        "usual_demand_method": reason,
        "usual_demand_sample_count": 0,
        "usual_demand_fallback_level": 99,
    }


def _season_label(month: int) -> str:
    if int(month) in {12, 1, 2}:
        return "winter"
    if int(month) in {3, 4, 5}:
        return "spring"
    if int(month) in {6, 7, 8}:
        return "summer"
    return "autumn"


def _demand_context(
    row: pd.Series | None,
    usual_row: pd.Series | None,
    scope: str,
    source_quality: str,
) -> CurrentDemandContext:
    observed = _row_metric(row, "consumption_mw", "MW", f"{scope.title()} observed demand is unavailable.", source_quality)
    usual = _usual_metric(usual_row, source_quality)
    anomaly_pct = _anomaly_metric(observed.value, usual.value, source_quality)
    if observed.value is None or usual.value is None:
        delta_gw = _metric(None, "GW", "Difference versus usual is unavailable because demand or baseline is missing.", source_quality)
    else:
        delta_gw = _metric((observed.value - usual.value) / 1000.0, "GW", source_quality=source_quality)
    return CurrentDemandContext(
        current=observed,
        usual=usual,
        difference_vs_usual_pct=anomaly_pct,
        difference_vs_usual_gw=delta_gw,
        baseline_id="usual-demand-comparable-history",
        baseline_method=_string_or_default(
            None if usual_row is None else usual_row.get("usual_demand_method"),
            "usual-demand baseline unavailable",
        ),
        baseline_sample_count=_int_or_none(None if usual_row is None else usual_row.get("usual_demand_sample_count")),
        baseline_fallback_level=_int_or_none(None if usual_row is None else usual_row.get("usual_demand_fallback_level")),
    )


def _generation_mix(row: pd.Series | None, source_quality: str) -> CurrentGenerationMix:
    total = _row_metric(row, "total_production_mw", "MW", "Generation total is unavailable.", source_quality)
    technologies = []
    for technology, column in GENERATION_COLUMNS:
        power = _row_metric(row, column, "MW", f"{technology} generation is unavailable.", source_quality)
        if total.value is None or total.value <= 0 or power.value is None:
            share = _metric(None, "percent", "Generation share is unavailable because total generation is missing.", source_quality)
        else:
            share = _metric(power.value / total.value * 100.0, "percent", source_quality=source_quality)
        if power.value is not None or row is None:
            technologies.append(GenerationTechnologyMetric(technology=technology, power=power, share=share))
    return CurrentGenerationMix(
        total=total,
        technologies=technologies,
        renewable_share=_share_metric(row, "renewable_share", "Renewable share is unavailable.", source_quality),
        fossil_share=_share_metric(row, "fossil_share", "Fossil share is unavailable.", source_quality),
    )


def _carbon_metric(row: pd.Series | None, source_quality: str) -> EnvironmentalMetric:
    return EnvironmentalMetric(
        metric="carbon_intensity",
        estimate=_row_metric(
            row,
            "co2_intensity_g_per_kwh",
            "gCO2/kWh",
            "Carbon intensity is unavailable from the source record.",
            source_quality,
        ),
        included_in_modelled_status=False,
        note="Environmental metric reported separately; it is not an input to modelled balance status.",
    )


def _official_ecowatt_signal(ecowatt: SourceTable, event_time: pd.Timestamp) -> CurrentOfficialSignal:
    if ecowatt.frame.empty:
        return CurrentOfficialSignal(
            name="EcoWatt",
            signal_type="official",
            available=False,
            status=None,
            label=None,
            timestamp=None,
            source=ecowatt.name,
            reason=ecowatt.reason or "Official EcoWatt signal is unavailable for this window.",
        )
    signal = status_at(ecowatt.frame, event_time)
    available = str(signal.get("ecowatt_status", "unknown")) != "unknown"
    if not available:
        return CurrentOfficialSignal(
            name="EcoWatt",
            signal_type="official",
            available=False,
            status=None,
            label=None,
            timestamp=None,
            source=str(signal.get("ecowatt_source") or ecowatt.name),
            reason="No official EcoWatt signal is available near the current timestamp.",
            detail=str(signal.get("ecowatt_message") or ""),
        )
    return CurrentOfficialSignal(
        name="EcoWatt",
        signal_type="official",
        available=True,
        status=str(signal.get("ecowatt_status")),
        label=str(signal.get("ecowatt_label")),
        timestamp=_timestamp_to_datetime(signal.get("timestamp")),
        source=str(signal.get("ecowatt_source") or ecowatt.name),
        detail=str(signal.get("ecowatt_message") or ""),
    )


def _modelled_status(row: pd.Series | None) -> CurrentModelledStatus | None:
    if row is None:
        return None
    try:
        version = threshold_config_version()
        config = load_status_thresholds()
    except (OSError, ValueError, KeyError):
        return None
    demand = _finite_float(row.get("consumption_mw"))
    total_generation = _finite_float(row.get("total_production_mw"))
    imports = _finite_float(row.get("imports_mw"))
    if imports is None:
        net_imports = _finite_float(row.get("net_imports_mw"))
        imports = max(net_imports, 0.0) if net_imports is not None else None
    available = None if total_generation is None or imports is None else total_generation + imports
    ratio = None if demand is None or available is None or available <= 0 else demand / available
    status = balance_status_for_ratio(ratio)
    return CurrentModelledStatus(
        signal_type="modelled",
        status=status,
        label=status_label(status),
        model_id="documented-balance-thresholds",
        model_version=version,
        calculation_inputs=[str(item) for item in config.get("calculation_inputs", [])],
        threshold_config_version=version,
        reason="Documented balance status from demand divided by generation plus physical imports.",
    )


def _freshness_for_table(table: SourceTable, now: datetime) -> FreshnessStatus:
    timestamp = _latest_timestamp(table.frame)
    if table.operating_state == OperatingState.SOURCE_UNAVAILABLE:
        return FreshnessStatus(
            state=OperatingState.SOURCE_UNAVAILABLE,
            timestamp=None,
            retrieved_at=table.retrieved_at,
            age_seconds=None,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            reason=table.reason or "Source data is unavailable.",
        )
    if timestamp is None:
        return FreshnessStatus(
            state=OperatingState.SOURCE_UNAVAILABLE,
            timestamp=None,
            retrieved_at=table.retrieved_at,
            age_seconds=None,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            reason="Source returned no timestamped records.",
        )
    if table.operating_state == OperatingState.HISTORICAL_REPLAY:
        return FreshnessStatus(
            state=OperatingState.HISTORICAL_REPLAY,
            timestamp=timestamp.to_pydatetime(),
            retrieved_at=table.retrieved_at,
            age_seconds=None,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            reason=f"Historical replay: {settings.demo_fixed_date_label}.",
        )
    age = max((pd.Timestamp(now) - timestamp).total_seconds(), 0.0)
    state = table.operating_state
    reason = table.reason
    if state == OperatingState.FRESH_LIVE_DATA:
        if age <= FRESH_LIVE_MAX_AGE.total_seconds():
            state = OperatingState.FRESH_LIVE_DATA
        else:
            state = OperatingState.DELAYED_LIVE_DATA
            reason = f"Latest source record is {age / 60.0:.0f} minutes old."
    elif state == OperatingState.LAST_KNOWN_GOOD_FALLBACK and age > LAST_KNOWN_GOOD_MAX_AGE.total_seconds():
        reason = f"Last-known-good source record is {age / 3600.0:.1f} hours old."
    return FreshnessStatus(
        state=state,
        timestamp=timestamp.to_pydatetime(),
        retrieved_at=table.retrieved_at,
        age_seconds=age,
        refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
        reason=reason,
    )


def _selected_region_freshness(table: SourceTable, region_id: str, now: datetime) -> FreshnessStatus:
    row = _region_row(table.frame, region_id)
    if row is None:
        return FreshnessStatus(
            state=OperatingState.SOURCE_UNAVAILABLE,
            timestamp=None,
            retrieved_at=table.retrieved_at,
            age_seconds=None,
            refresh_interval_seconds=SOURCE_REFRESH_INTERVAL_SECONDS,
            reason=f"No regional record is available for {REGION_NAMES[region_id]}.",
        )
    region_table = SourceTable(
        frame=pd.DataFrame([row.to_dict()]),
        source_id=table.source_id,
        name=table.name,
        operating_state=table.operating_state,
        source_quality=table.source_quality,
        retrieved_at=table.retrieved_at,
        reason=table.reason,
    )
    return _freshness_for_table(region_table, now)


def _health_operating_state(sources: list[SourceHealth]) -> OperatingState:
    states = [source.operating_state for source in sources]
    if all(state == OperatingState.HISTORICAL_REPLAY for state in states if state != OperatingState.SOURCE_UNAVAILABLE):
        return OperatingState.HISTORICAL_REPLAY
    if OperatingState.SOURCE_UNAVAILABLE in states[:2]:
        return OperatingState.SOURCE_UNAVAILABLE
    if OperatingState.LAST_KNOWN_GOOD_FALLBACK in states:
        return OperatingState.LAST_KNOWN_GOOD_FALLBACK
    if OperatingState.DELAYED_LIVE_DATA in states:
        return OperatingState.DELAYED_LIVE_DATA
    return OperatingState.FRESH_LIVE_DATA


def _response_operating_state(national: FreshnessStatus, selected: FreshnessStatus) -> OperatingState:
    if national.state == OperatingState.HISTORICAL_REPLAY:
        return OperatingState.HISTORICAL_REPLAY
    if national.state == OperatingState.SOURCE_UNAVAILABLE or selected.state == OperatingState.SOURCE_UNAVAILABLE:
        return OperatingState.SOURCE_UNAVAILABLE
    if national.state == OperatingState.LAST_KNOWN_GOOD_FALLBACK or selected.state == OperatingState.LAST_KNOWN_GOOD_FALLBACK:
        return OperatingState.LAST_KNOWN_GOOD_FALLBACK
    if national.state == OperatingState.DELAYED_LIVE_DATA or selected.state == OperatingState.DELAYED_LIVE_DATA:
        return OperatingState.DELAYED_LIVE_DATA
    return OperatingState.FRESH_LIVE_DATA


def _unavailable_fields(
    national: NationalCurrentContext,
    selected: RegionalCurrentContext,
    map_regions: list[CurrentMapRegion],
) -> list[UnavailableField]:
    fields: list[UnavailableField] = []

    def add(field: str, metric: NullableMetric) -> None:
        if metric.value is None:
            fields.append(UnavailableField(field=field, reason=metric.reason or "Value unavailable."))

    add("national_context.demand.current", national.demand.current)
    add("national_context.demand.usual", national.demand.usual)
    add("national_context.physical_imports", national.physical_imports)
    add("national_context.physical_exports", national.physical_exports)
    add("national_context.carbon_estimate", national.carbon_estimate.estimate)
    if not national.official_ecowatt_signal.available:
        fields.append(
            UnavailableField(
                field="national_context.official_ecowatt_signal",
                reason=national.official_ecowatt_signal.reason or "EcoWatt unavailable.",
            )
        )
    add("selected_region_context.demand.current", selected.demand.current)
    add("selected_region_context.demand.usual", selected.demand.usual)
    add("selected_region_context.net_flow", selected.net_flow)
    add("selected_region_context.physical_balance", selected.physical_balance)
    for region in map_regions:
        if not region.availability_flag:
            fields.append(
                UnavailableField(
                    field=f"map.{region.region_id}.demand_anomaly_pct",
                    reason=region.demand_anomaly_pct.reason or "Map demand anomaly unavailable.",
                )
            )
    return fields


def _metric(
    value: Any,
    unit: str,
    reason: str | None = None,
    source_quality: str | None = None,
) -> NullableMetric:
    number = _finite_float(value)
    return NullableMetric(value=number, unit=unit, reason=reason if number is None else None, source_quality=source_quality)


def _row_metric(
    row: pd.Series | None,
    column: str,
    unit: str,
    reason: str,
    source_quality: str,
) -> NullableMetric:
    if row is None or column not in row:
        return _metric(None, unit, reason, source_quality)
    return _metric(row.get(column), unit, reason, source_quality)


def _usual_metric(row: pd.Series | None, source_quality: str) -> NullableMetric:
    if row is None:
        return _metric(None, "MW", "Usual-demand baseline is unavailable for this geography.", source_quality)
    return _metric(
        row.get("usual_demand_mw"),
        "MW",
        "Usual-demand baseline is unavailable for this geography.",
        source_quality,
    )


def _anomaly_metric(observed: float | None, usual: float | None, source_quality: str) -> NullableMetric:
    if observed is None or usual is None or usual == 0:
        return _metric(
            None,
            "percent",
            "Demand anomaly is unavailable because observed or usual demand is missing.",
            source_quality,
        )
    return _metric((observed - usual) / usual * 100.0, "percent", source_quality=source_quality)


def _share_metric(row: pd.Series | None, column: str, reason: str, source_quality: str) -> NullableMetric:
    if row is None or column not in row:
        return _metric(None, "percent", reason, source_quality)
    value = _finite_float(row.get(column))
    return _metric(None if value is None else value * 100.0, "percent", reason, source_quality)


def _usual_row(usual: pd.DataFrame, *, scope: str, region_code: str | None) -> pd.Series | None:
    if usual.empty or "geographic_scope" not in usual or "region" not in usual:
        return None
    if scope == "national":
        matches = usual[usual["geographic_scope"].astype(str).eq("national")]
    else:
        region_name = REGION_NAMES[str(region_code)]
        normalized = _normalize_text(region_name)
        matches = usual[
            usual["geographic_scope"].astype(str).eq("regional")
            & usual["region"].astype(str).map(_normalize_text).eq(normalized)
        ]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty or "timestamp" not in frame:
        return None
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    if work.empty:
        return None
    return work.iloc[-1]


def _region_row(frame: pd.DataFrame, region_id: str) -> pd.Series | None:
    if frame.empty or "region_code" not in frame:
        return None
    rows = frame[frame["region_code"].astype(str).eq(str(region_id))].copy()
    if rows.empty:
        return None
    if "timestamp" in rows:
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
        rows = rows.sort_values("timestamp")
    return rows.iloc[-1]


def _latest_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
    row = _latest_row(frame)
    return None if row is None else _row_timestamp(row)


def _row_timestamp(row: pd.Series) -> pd.Timestamp | None:
    if "timestamp" not in row:
        return None
    try:
        timestamp = pd.Timestamp(row.get("timestamp"))
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _timestamp_to_datetime(value: Any) -> datetime | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


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


def _int_or_none(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(number)


def _string_or_default(value: Any, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value)
    return text if text else default


def _quality_for_freshness(freshness: FreshnessStatus, default: str) -> str:
    if freshness.state == OperatingState.SOURCE_UNAVAILABLE:
        return "unavailable"
    if freshness.state == OperatingState.DELAYED_LIVE_DATA:
        return "delayed"
    if freshness.state == OperatingState.LAST_KNOWN_GOOD_FALLBACK:
        return "last_known_good"
    if freshness.state == OperatingState.HISTORICAL_REPLAY:
        return "historical_replay"
    return default


def _ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _regional_history_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    clean = clean_energy_mix(raw)
    if "region" not in clean:
        return pd.DataFrame()
    clean["region_code"] = clean["region"].map(code_for_region_name)
    clean = clean.dropna(subset=["region_code"]).copy()
    if clean.empty:
        return clean
    clean["region_code"] = clean["region_code"].astype(str)
    clean["region_display"] = clean["region_code"].map(REGION_NAMES).fillna(clean["region"])
    return clean.sort_values(["region_code", "timestamp"]).reset_index(drop=True)


def _synthesize_regional_history(current: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    if current.empty:
        return current.copy()
    latest = _latest_timestamp(current) or pd.Timestamp.now(tz="UTC")
    timestamps = pd.date_range(latest.floor("h") - pd.Timedelta(days=days), latest.floor("h"), freq="h")
    rows: list[dict[str, Any]] = []
    for row in current.itertuples(index=False):
        current_demand = _finite_float(getattr(row, "consumption_mw", None)) or 0.0
        latest_local = latest.tz_convert(settings.timezone)
        current_factor = _hour_factor(latest_local.hour) * _day_factor(latest_local.dayofweek)
        base = current_demand / current_factor if current_factor else current_demand
        for index, timestamp in enumerate(timestamps):
            local = timestamp.tz_convert(settings.timezone)
            drift = 1 + ((index % 11) - 5) * 0.004
            item = row._asdict()
            item["timestamp"] = timestamp
            item["consumption_mw"] = max(base * _hour_factor(local.hour) * _day_factor(local.dayofweek) * drift, 0.0)
            rows.append(item)
    return pd.DataFrame.from_records(rows)


def _hour_factor(hour: int) -> float:
    if 18 <= hour <= 21:
        return 1.13
    if 7 <= hour <= 9:
        return 1.06
    if 1 <= hour <= 5:
        return 0.86
    if 11 <= hour <= 15:
        return 0.96
    return 1.0


def _day_factor(day_of_week: int) -> float:
    return 0.94 if day_of_week >= 5 else 1.0


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("'", " ").split())


default_service = CurrentStateService()
