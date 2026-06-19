"""Typed contracts for the Energy Pulse France digital twin.

The repository is currently an in-process Streamlit app, so these Python
contracts are the backend schema source. The schema and frontend declaration
artifacts are generated from this module by ``scripts/generate_contracts.py``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import types
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints


class ContractValidationError(ValueError):
    """Raised when a domain contract violates an explicit semantic rule."""


class DomainMode(str, Enum):
    LIVE = "live"
    FORECAST = "forecast"
    SIMULATION = "simulation"
    REPLAY = "replay"


class OperatingState(str, Enum):
    FRESH_LIVE_DATA = "fresh_live_data"
    DELAYED_LIVE_DATA = "delayed_live_data"
    LAST_KNOWN_GOOD_FALLBACK = "last_known_good_fallback"
    SOURCE_UNAVAILABLE = "source_unavailable"
    HISTORICAL_REPLAY = "historical_replay"


class Scope(str, Enum):
    NATIONAL = "national"
    REGIONAL = "regional"


class SourceType(str, Enum):
    OFFICIAL = "official"
    OBSERVED = "observed"
    MODEL = "model"
    FALLBACK = "fallback"
    SCENARIO = "scenario"


class EstimateProvenanceKind(str, Enum):
    OBSERVED = "observed"
    OFFICIAL_FORECAST = "official_forecast"
    STATISTICAL_ESTIMATE = "statistical_estimate"
    PERSISTENCE_FALLBACK = "persistence_fallback"
    RESIDUAL_ESTIMATE = "residual_estimate"
    UNAVAILABLE = "unavailable"


class Freshness(str, Enum):
    FRESH = "fresh"
    DELAYED = "delayed"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class Status(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    HIGH = "high"
    UNKNOWN = "unknown"


class Unit(str, Enum):
    MW = "MW"
    GW = "GW"
    MWH = "MWh"
    GWH = "GWh"
    PERCENTAGE = "percentage"
    TONNES_CO2 = "tonnes_CO2"
    GCO2_PER_KWH = "gCO2_per_kWh"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class ContractBase:
    """Base behavior shared by all dataclass contracts."""

    _schema_description: ClassVar[str] = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        result = to_dict(self)
        if not isinstance(result, dict):
            raise ContractValidationError("contract serialization did not produce an object")
        return result


@dataclass(frozen=True)
class DataQuality(ContractBase):
    freshness: Freshness
    confidence: Confidence
    status: Status = Status.UNKNOWN
    checked_at: datetime | None = None
    warnings: list[str] | None = None

    def validate(self) -> None:
        if self.freshness == Freshness.UNAVAILABLE and self.confidence != Confidence.UNAVAILABLE:
            raise ContractValidationError("unavailable data quality must use unavailable confidence")


@dataclass(frozen=True)
class DataProvenance(ContractBase):
    source_type: SourceType
    name: str
    mode: DomainMode
    event_time: datetime
    update_time: datetime
    is_fallback: bool = False
    is_demo: bool = False
    dataset_id: str | None = None
    url: str | None = None
    retrieved_at: datetime | None = None
    license: str | None = None
    transformation: str | None = None
    fallback_reason: str | None = None
    replay_label: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ContractValidationError("data provenance requires a source name")
        if self.mode == DomainMode.REPLAY or self.is_demo:
            if not self.replay_label:
                raise ContractValidationError("demo or replay provenance must carry an explicit replay label")
        if self.source_type == SourceType.FALLBACK and not self.is_fallback:
            raise ContractValidationError("fallback source type must set is_fallback")
        if self.is_fallback and not self.fallback_reason:
            raise ContractValidationError("fallback values must include a fallback reason")


@dataclass(frozen=True)
class QuantifiedValue(ContractBase):
    value: float | None
    unit: Unit
    event_time: datetime
    update_time: datetime
    source: DataProvenance
    quality: DataQuality
    is_fallback: bool = False
    display_value: float | None = None
    display_unit: Unit | None = None
    label: str | None = None

    def validate(self) -> None:
        if self.source.is_fallback != self.is_fallback:
            raise ContractValidationError("value fallback flag must match provenance fallback flag")
        if self.display_unit is not None and self.display_value is None:
            raise ContractValidationError("display_unit requires display_value")


@dataclass(frozen=True)
class BaselineDefinition(ContractBase):
    baseline_id: str
    version: str
    method: str
    comparison_keys: list[str]
    lookback_days: int | None = None
    notes: str | None = None

    def validate(self) -> None:
        if not self.baseline_id or not self.version:
            raise ContractValidationError("usual demand requires a baseline id and version")
        if not self.comparison_keys:
            raise ContractValidationError("usual demand baseline must declare comparison keys")


@dataclass(frozen=True)
class DemandContext(ContractBase):
    current: QuantifiedValue
    usual: QuantifiedValue
    anomaly_percentage: QuantifiedValue
    baseline_definition: BaselineDefinition
    scope: Scope
    interpretation: str

    def validate(self) -> None:
        if self.scope == Scope.REGIONAL and "run out" in self.interpretation.lower():
            raise ContractValidationError("regional demand context must not imply independent outage risk")
        if self.scope == Scope.REGIONAL and "independent" in self.interpretation.lower():
            raise ContractValidationError("regional demand context must avoid independent-system wording")


@dataclass(frozen=True)
class GenerationEstimate(ContractBase):
    technology: str
    power: QuantifiedValue
    share: QuantifiedValue | None = None

    def validate(self) -> None:
        if not self.technology:
            raise ContractValidationError("generation estimate requires a technology name")


@dataclass(frozen=True)
class GenerationMix(ContractBase):
    total: QuantifiedValue
    estimates: list[GenerationEstimate]
    renewable_share: QuantifiedValue | None = None
    fossil_share: QuantifiedValue | None = None
    co2_intensity: QuantifiedValue | None = None

    def validate(self) -> None:
        if not self.estimates:
            raise ContractValidationError("generation mix requires at least one generation estimate")


@dataclass(frozen=True)
class TwinComponentEstimate(ContractBase):
    component: str
    value: QuantifiedValue
    provenance_kind: EstimateProvenanceKind
    included_in_total: bool = True
    formula: str | None = None
    note: str | None = None

    def validate(self) -> None:
        if not self.component:
            raise ContractValidationError("component estimates require a component name")
        if self.provenance_kind == EstimateProvenanceKind.OFFICIAL_FORECAST:
            if self.value.source.source_type != SourceType.OFFICIAL:
                raise ContractValidationError("official forecast estimates require official provenance")
        if self.provenance_kind == EstimateProvenanceKind.PERSISTENCE_FALLBACK:
            if not self.value.source.is_fallback:
                raise ContractValidationError("persistence fallback estimates must use fallback provenance")
        if self.provenance_kind == EstimateProvenanceKind.UNAVAILABLE and self.value.value is not None:
            raise ContractValidationError("unavailable component estimates must have null values")


@dataclass(frozen=True)
class RegionalDemandForecast(ContractBase):
    region_code: str
    region_name: str
    forecast: ForecastInterval
    usual: QuantifiedValue
    unreconciled_p50: QuantifiedValue
    share_of_national_p50: QuantifiedValue
    reconciliation_factor: float
    method: str
    source: DataProvenance
    quality: DataQuality
    note: str = (
        "Regional demand is a reconciled demand forecast only; it is not a regional adequacy "
        "or shortage status."
    )

    def validate(self) -> None:
        if not self.region_code or not self.region_name:
            raise ContractValidationError("regional demand forecasts require a region code and name")
        forbidden = ("adequacy status", "reserve margin", "regional balance status")
        lowered = f"{self.method} {self.note}".lower()
        if any(token in lowered for token in forbidden):
            raise ContractValidationError("regional demand forecast must not represent regional adequacy status")
        if self.reconciliation_factor < 0:
            raise ContractValidationError("regional reconciliation factor cannot be negative")


@dataclass(frozen=True)
class GenerationAvailabilityContext(ContractBase):
    nuclear: TwinComponentEstimate
    announced_unavailable: QuantifiedValue
    announced_unavailability_components: list[TwinComponentEstimate]
    unavailable_optional_sources: list[UnavailableField]
    method: str
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if "nuclear" not in self.nuclear.component.lower():
            raise ContractValidationError("nuclear availability must be represented separately")
        if "wind" in self.nuclear.component.lower() or "solar" in self.nuclear.component.lower():
            raise ContractValidationError("nuclear availability cannot be combined with wind or solar")
        if not self.method:
            raise ContractValidationError("generation availability context requires a method")


@dataclass(frozen=True)
class EstimatedGenerationMix(ContractBase):
    total: QuantifiedValue
    components: list[TwinComponentEstimate]
    residual_bucket_name: str
    formula: str
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if not self.components:
            raise ContractValidationError("estimated generation mix requires components")
        if "residual" not in self.residual_bucket_name.lower():
            raise ContractValidationError("estimated generation mix must name its residual bucket")
        if not any(
            item.component == self.residual_bucket_name
            and item.provenance_kind == EstimateProvenanceKind.RESIDUAL_ESTIMATE
            for item in self.components
        ):
            raise ContractValidationError("estimated generation mix requires a residual-estimate component")
        values = [item.value.value for item in self.components if item.included_in_total]
        if self.total.value is not None and all(value is not None for value in values):
            total = sum(float(value) for value in values if value is not None)
            if abs(total - float(self.total.value)) > 1e-6 * max(abs(float(self.total.value)), 1.0):
                raise ContractValidationError("estimated generation mix total must equal included components")


@dataclass(frozen=True)
class ExchangeEstimate(ContractBase):
    net_imports: QuantifiedValue
    imports: QuantifiedValue
    exports: QuantifiedValue
    provenance_kind: EstimateProvenanceKind
    method: str
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if "exchange" not in self.method.lower() and "import" not in self.method.lower():
            raise ContractValidationError("exchange estimates require an exchange/import method note")


@dataclass(frozen=True)
class CarbonEstimate(ContractBase):
    intensity: QuantifiedValue
    provenance_kind: EstimateProvenanceKind
    method: str
    included_in_balance_context: bool
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if self.included_in_balance_context:
            raise ContractValidationError("carbon estimates must remain separate from modelled balance context")
        if not self.method:
            raise ContractValidationError("carbon estimates require a method")


@dataclass(frozen=True)
class ModelledBalanceContribution(ContractBase):
    component: str
    value: QuantifiedValue
    weight: float
    contribution: float
    source: DataProvenance
    quality: DataQuality
    note: str | None = None

    def validate(self) -> None:
        if not self.component:
            raise ContractValidationError("balance contributions require a component")
        if self.weight < 0:
            raise ContractValidationError("balance contribution weight cannot be negative")
        lowered = self.component.lower()
        if "carbon" in lowered or "co2" in lowered or "uncertainty" in lowered:
            raise ContractValidationError("carbon and uncertainty cannot be balance contributions")


@dataclass(frozen=True)
class OfficialSignal(ContractBase):
    name: str
    scope: Scope
    status: Status
    label: str
    signal_time: datetime
    update_time: datetime
    source: DataProvenance
    quality: DataQuality
    detail: str | None = None

    def validate(self) -> None:
        if self.name.lower() == "ecowatt" and self.scope != Scope.NATIONAL:
            raise ContractValidationError("official EcoWatt status is national-only in this contract")
        if self.source.source_type != SourceType.OFFICIAL:
            raise ContractValidationError("official signals must use official provenance")


@dataclass(frozen=True)
class ModelledBalanceContext(ContractBase):
    status: Status
    pressure_ratio: QuantifiedValue
    available_generation: QuantifiedValue
    net_imports: QuantifiedValue
    supply_margin: QuantifiedValue
    import_requirement: QuantifiedValue
    threshold_config_version: str
    source: DataProvenance
    quality: DataQuality
    calculation_inputs: list[str]
    method: str = "demand divided by available generation plus net imports"

    def validate(self) -> None:
        if self.source.source_type == SourceType.OFFICIAL:
            raise ContractValidationError("modelled balance context must not be labelled official")
        if not self.threshold_config_version:
            raise ContractValidationError("balance status requires a versioned threshold configuration")
        forbidden = ("carbon", "co2", "uncertainty", "interval")
        lowered = [item.lower() for item in self.calculation_inputs]
        if any(any(token in item for token in forbidden) for item in lowered):
            raise ContractValidationError("carbon and uncertainty cannot be balance-pressure inputs")


@dataclass(frozen=True)
class ConfidenceAssessment(ContractBase):
    confidence: Confidence
    rationale: str
    source: DataProvenance
    quality: DataQuality
    backtest_mae: QuantifiedValue | None = None
    interval_calibration: str | None = None

    def validate(self) -> None:
        if not self.rationale:
            raise ContractValidationError("confidence assessment requires a rationale")


@dataclass(frozen=True)
class ExplanationDriver(ContractBase):
    name: str
    contribution: QuantifiedValue
    direction: str
    included_in_balance_pressure: bool
    source: DataProvenance
    quality: DataQuality
    note: str | None = None

    def validate(self) -> None:
        if self.included_in_balance_pressure and self.name.lower() in {"carbon", "co2", "uncertainty"}:
            raise ContractValidationError("carbon and uncertainty are explanation-only, not balance inputs")


@dataclass(frozen=True)
class Explanation(ContractBase):
    explanation_id: str
    method: str
    text: str
    drivers: list[ExplanationDriver]
    confidence: ConfidenceAssessment
    source: DataProvenance
    quality: DataQuality
    caveats: list[str] | None = None

    def validate(self) -> None:
        if not self.explanation_id or not self.method:
            raise ContractValidationError("explanation requires id and method")


@dataclass(frozen=True)
class ForecastInterval(ContractBase):
    p10: QuantifiedValue
    p50: QuantifiedValue
    p90: QuantifiedValue
    confidence: ConfidenceAssessment

    def validate(self) -> None:
        values = (self.p10.value, self.p50.value, self.p90.value)
        if all(value is not None for value in values):
            p10, p50, p90 = (float(value) for value in values if value is not None)
            if not (p10 <= p50 <= p90):
                raise ContractValidationError("forecast interval must satisfy p10 <= p50 <= p90")


@dataclass(frozen=True)
class ForecastPoint(ContractBase):
    target_time: datetime
    horizon_hours: int
    demand: ForecastInterval
    balance_context: ModelledBalanceContext
    route_source_type: SourceType
    route_label: str
    uncertainty: QuantifiedValue
    explanation: Explanation | None = None

    def validate(self) -> None:
        if self.horizon_hours < 0:
            raise ContractValidationError("forecast horizon must be non-negative")
        if self.route_source_type == SourceType.OFFICIAL and self.balance_context.source.source_type == SourceType.OFFICIAL:
            raise ContractValidationError("official forecast routing cannot make balance status official")


@dataclass(frozen=True)
class ModelCard(ContractBase):
    model_id: str
    version: str
    display_name: str
    purpose: str
    source_type: SourceType
    training_period_start: datetime | None
    training_period_end: datetime | None
    evaluation_metrics: dict[str, float]
    promoted_horizons: list[int]
    provenance: DataProvenance
    quality: DataQuality
    limitations: list[str]

    def validate(self) -> None:
        if self.source_type == SourceType.OFFICIAL:
            raise ContractValidationError("model cards cannot describe an official source as a model")
        if not self.model_id or not self.version:
            raise ContractValidationError("model card requires id and version")


@dataclass(frozen=True)
class ForecastRun(ContractBase):
    run_id: str
    mode: DomainMode
    origin_time: datetime
    generated_at: datetime
    horizon_hours: int
    points: list[ForecastPoint]
    source: DataProvenance
    quality: DataQuality
    model_card: ModelCard | None = None

    def validate(self) -> None:
        if self.mode not in {DomainMode.FORECAST, DomainMode.REPLAY}:
            raise ContractValidationError("forecast run mode must be forecast or replay")
        if self.horizon_hours < 0:
            raise ContractValidationError("forecast run horizon must be non-negative")
        if len(self.points) > self.horizon_hours and self.horizon_hours > 0:
            raise ContractValidationError("forecast run has more points than declared horizon")


@dataclass(frozen=True)
class RegionalState(ContractBase):
    region_code: str
    region_name: str
    scope: Scope
    demand_context: DemandContext
    local_generation: GenerationMix
    source: DataProvenance
    quality: DataQuality
    grid_context_note: str = (
        "Regional demand is context within the connected French grid; "
        "this contract does not model a region as electrically isolated."
    )

    def validate(self) -> None:
        if self.scope != Scope.REGIONAL:
            raise ContractValidationError("regional state must use regional scope")
        note = self.grid_context_note.lower()
        if "connected" not in note or "isolated" not in note:
            raise ContractValidationError("regional state must explain connected-grid context")


@dataclass(frozen=True)
class NationalState(ContractBase):
    scope: Scope
    demand_context: DemandContext
    generation_mix: GenerationMix
    official_signal: OfficialSignal
    balance_context: ModelledBalanceContext
    source: DataProvenance
    quality: DataQuality
    regions: list[RegionalState] | None = None

    def validate(self) -> None:
        if self.scope != Scope.NATIONAL:
            raise ContractValidationError("national state must use national scope")
        if self.official_signal.scope != Scope.NATIONAL:
            raise ContractValidationError("national state cannot contain a regional official signal")


@dataclass(frozen=True)
class ScenarioEvent(ContractBase):
    event_id: str
    event_type: str
    start_time: datetime
    end_time: datetime
    affected_value: str
    delta: QuantifiedValue
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if self.source.source_type != SourceType.SCENARIO:
            raise ContractValidationError("scenario events must use scenario provenance")
        if self.end_time < self.start_time:
            raise ContractValidationError("scenario event end_time must be after start_time")


@dataclass(frozen=True)
class ScenarioRequest(ContractBase):
    scenario_id: str
    mode: DomainMode
    created_at: datetime
    baseline_forecast_run_id: str
    assumption_version: str
    events: list[ScenarioEvent]
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if self.mode != DomainMode.SIMULATION:
            raise ContractValidationError("scenario requests must use simulation mode")
        if self.source.source_type != SourceType.SCENARIO:
            raise ContractValidationError("scenario requests must use scenario provenance")
        if not self.assumption_version:
            raise ContractValidationError("scenario request requires an assumption version")


@dataclass(frozen=True)
class ScenarioDelta(ContractBase):
    metric: str
    baseline: QuantifiedValue
    scenario: QuantifiedValue
    delta: QuantifiedValue
    explanation: str

    def validate(self) -> None:
        if not self.metric:
            raise ContractValidationError("scenario delta requires a metric")


@dataclass(frozen=True)
class ScenarioResult(ContractBase):
    result_id: str
    request: ScenarioRequest
    generated_at: datetime
    forecast_points: list[ForecastPoint]
    deltas: list[ScenarioDelta]
    confidence: ConfidenceAssessment
    source: DataProvenance
    quality: DataQuality

    def validate(self) -> None:
        if self.source.source_type != SourceType.SCENARIO:
            raise ContractValidationError("scenario results must use scenario provenance")


@dataclass(frozen=True)
class NullableMetric(ContractBase):
    value: float | None
    unit: str
    reason: str | None = None
    source_quality: str | None = None

    def validate(self) -> None:
        if self.value is None and not self.reason:
            raise ContractValidationError("nullable metrics require a reason when value is null")
        if not self.unit:
            raise ContractValidationError("nullable metrics require a unit")


@dataclass(frozen=True)
class FreshnessStatus(ContractBase):
    state: OperatingState
    timestamp: datetime | None
    retrieved_at: datetime | None
    age_seconds: float | None
    refresh_interval_seconds: int
    reason: str | None = None

    def validate(self) -> None:
        if self.refresh_interval_seconds <= 0:
            raise ContractValidationError("refresh interval must be positive")
        if self.timestamp is None and not self.reason:
            raise ContractValidationError("freshness without a timestamp requires a reason")


@dataclass(frozen=True)
class GenerationTechnologyMetric(ContractBase):
    technology: str
    power: NullableMetric
    share: NullableMetric

    def validate(self) -> None:
        if not self.technology:
            raise ContractValidationError("generation technology metric requires a technology")


@dataclass(frozen=True)
class CurrentGenerationMix(ContractBase):
    total: NullableMetric
    technologies: list[GenerationTechnologyMetric]
    renewable_share: NullableMetric
    fossil_share: NullableMetric


@dataclass(frozen=True)
class EnvironmentalMetric(ContractBase):
    metric: str
    estimate: NullableMetric
    included_in_modelled_status: bool
    note: str

    def validate(self) -> None:
        if not self.metric or not self.note:
            raise ContractValidationError("environmental metrics require a metric name and note")
        if self.included_in_modelled_status:
            raise ContractValidationError("environmental metrics must remain separate from modelled status")


@dataclass(frozen=True)
class CurrentDemandContext(ContractBase):
    current: NullableMetric
    usual: NullableMetric
    difference_vs_usual_pct: NullableMetric
    difference_vs_usual_gw: NullableMetric
    baseline_id: str
    baseline_method: str
    baseline_sample_count: int | None = None
    baseline_fallback_level: int | None = None

    def validate(self) -> None:
        if not self.baseline_id or not self.baseline_method:
            raise ContractValidationError("current demand context requires baseline metadata")


@dataclass(frozen=True)
class CurrentOfficialSignal(ContractBase):
    name: str
    signal_type: str
    available: bool
    status: str | None
    label: str | None
    timestamp: datetime | None
    source: str
    reason: str | None = None
    detail: str | None = None

    def validate(self) -> None:
        if self.signal_type != "official":
            raise ContractValidationError("official signal contract must use signal_type=official")
        if not self.available and not self.reason:
            raise ContractValidationError("unavailable official signal requires a reason")


@dataclass(frozen=True)
class CurrentModelledStatus(ContractBase):
    signal_type: str
    status: Status
    label: str
    model_id: str
    model_version: str
    calculation_inputs: list[str]
    threshold_config_version: str
    reason: str | None = None

    def validate(self) -> None:
        if self.signal_type != "modelled":
            raise ContractValidationError("modelled status contract must use signal_type=modelled")
        if not self.calculation_inputs:
            raise ContractValidationError("modelled status requires calculation inputs")
        forbidden = ("carbon", "co2", "uncertainty")
        lowered = " ".join(self.calculation_inputs).lower()
        if any(token in lowered for token in forbidden):
            raise ContractValidationError("modelled status cannot include carbon or uncertainty inputs")


@dataclass(frozen=True)
class NationalCurrentContext(ContractBase):
    demand: CurrentDemandContext
    freshness: FreshnessStatus
    generation_mix: CurrentGenerationMix
    physical_imports: NullableMetric
    physical_exports: NullableMetric
    net_imports: NullableMetric
    carbon_estimate: EnvironmentalMetric
    official_ecowatt_signal: CurrentOfficialSignal
    modelled_status: CurrentModelledStatus | None


@dataclass(frozen=True)
class RegionalCurrentContext(ContractBase):
    region_code: str
    region_name: str
    demand: CurrentDemandContext
    freshness: FreshnessStatus
    local_generation: CurrentGenerationMix
    net_flow: NullableMetric
    physical_balance: NullableMetric
    connected_grid_note: str

    def validate(self) -> None:
        if not self.region_code or not self.region_name:
            raise ContractValidationError("selected-region context requires a code and name")
        lowered = self.connected_grid_note.lower()
        if "connected" not in lowered or "shortage" not in lowered:
            raise ContractValidationError("regional context note must prevent isolated-shortage interpretation")


@dataclass(frozen=True)
class CurrentMapRegion(ContractBase):
    region_id: str
    region_name: str
    demand_anomaly_pct: NullableMetric
    observed_demand: NullableMetric
    usual_demand: NullableMetric
    source_quality: str
    availability_flag: bool

    def validate(self) -> None:
        if not self.region_id or not self.region_name:
            raise ContractValidationError("map regions require id and name")
        if self.availability_flag and self.demand_anomaly_pct.value is None:
            raise ContractValidationError("available map regions require a demand anomaly value")


@dataclass(frozen=True)
class CacheInfo(ContractBase):
    cache_key: str
    ttl_seconds: int
    generated_at: datetime
    expires_at: datetime
    cache_hit: bool


@dataclass(frozen=True)
class UnavailableField(ContractBase):
    field: str
    reason: str

    def validate(self) -> None:
        if not self.field or not self.reason:
            raise ContractValidationError("unavailable fields require field and reason")


@dataclass(frozen=True)
class CurrentStateResponse(ContractBase):
    generated_at: datetime
    region: str
    operating_state: OperatingState
    cache: CacheInfo
    national_context: NationalCurrentContext
    selected_region_context: RegionalCurrentContext
    map: list[CurrentMapRegion]
    unavailable_fields: list[UnavailableField]


@dataclass(frozen=True)
class SourceHealth(ContractBase):
    source_id: str
    name: str
    operating_state: OperatingState
    freshness: FreshnessStatus
    source_quality: str
    missing_intervals: int = 0
    fallback_records: int = 0
    adapter_failures: int = 0
    circuit_breaker_state: str = "not_configured"
    latest_successful_fetch_at: datetime | None = None
    reason: str | None = None

    def validate(self) -> None:
        if not self.source_id or not self.name:
            raise ContractValidationError("source health requires source_id and name")
        if self.missing_intervals < 0 or self.fallback_records < 0 or self.adapter_failures < 0:
            raise ContractValidationError("source health counters cannot be negative")


@dataclass(frozen=True)
class ModelHealth(ContractBase):
    model_id: str
    status: str
    model_version: str | None
    latest_successful_forecast_at: datetime | None
    latest_successful_forecast_run_id: str | None
    recent_forecast_error_mae_mw: float | None
    fallback_usage: str | None = None
    reason: str | None = None

    def validate(self) -> None:
        if not self.model_id or not self.status:
            raise ContractValidationError("model health requires model_id and status")
        if self.recent_forecast_error_mae_mw is not None and self.recent_forecast_error_mae_mw < 0:
            raise ContractValidationError("forecast error cannot be negative")


@dataclass(frozen=True)
class ScenarioEngineHealth(ContractBase):
    available: bool
    version: str
    assumption_version: str
    cache_enabled: bool
    last_successful_scenario_id: str | None = None
    reason: str | None = None

    def validate(self) -> None:
        if not self.version or not self.assumption_version:
            raise ContractValidationError("scenario engine health requires versions")
        if not self.available and not self.reason:
            raise ContractValidationError("unavailable scenario engine requires a reason")


@dataclass(frozen=True)
class DataHealthResponse(ContractBase):
    generated_at: datetime
    operating_state: OperatingState
    cache: CacheInfo
    sources: list[SourceHealth]
    model_health: ModelHealth
    scenario_engine: ScenarioEngineHealth
    unavailable_fields: list[UnavailableField]


@dataclass(frozen=True)
class SourceMetadata(ContractBase):
    source_id: str
    name: str
    source_type: str
    dataset_id: str | None
    url: str | None
    required_for_now: bool
    credential_required: bool
    refresh_interval_seconds: int
    notes: str

    def validate(self) -> None:
        if not self.source_id or not self.name or not self.source_type:
            raise ContractValidationError("source metadata requires id, name, and source type")


@dataclass(frozen=True)
class SourcesResponse(ContractBase):
    generated_at: datetime
    sources: list[SourceMetadata]


@dataclass(frozen=True)
class StatusThresholdsResponse(ContractBase):
    generated_at: datetime
    version: str
    thresholds: dict[str, Any]
    calculation_inputs: list[str]
    excluded_inputs: list[str]
    raw_config: dict[str, Any]

    def validate(self) -> None:
        if not self.version:
            raise ContractValidationError("status thresholds response requires a version")


@dataclass(frozen=True)
class TwinSnapshot(ContractBase):
    snapshot_id: str
    mode: DomainMode
    event_time: datetime
    update_time: datetime
    national: NationalState
    source: DataProvenance
    quality: DataQuality
    regional_states: list[RegionalState] | None = None
    forecast_run: ForecastRun | None = None
    scenario_result: ScenarioResult | None = None
    explanations: list[Explanation] | None = None
    model_cards: list[ModelCard] | None = None
    demand_forecast: ForecastInterval | None = None
    usual_demand_baseline: QuantifiedValue | None = None
    regional_demand_context: list[RegionalDemandForecast] | None = None
    wind_estimate: TwinComponentEstimate | None = None
    solar_estimate: TwinComponentEstimate | None = None
    generation_availability_context: GenerationAvailabilityContext | None = None
    generation_mix_estimate: EstimatedGenerationMix | None = None
    exchange_estimate: ExchangeEstimate | None = None
    modelled_national_balance_context: ModelledBalanceContext | None = None
    modelled_balance_contributions: list[ModelledBalanceContribution] | None = None
    official_signal_context: OfficialSignal | None = None
    carbon_estimate: CarbonEstimate | None = None
    provenance_chain: list[DataProvenance] | None = None
    unsupported_physical_behaviours: list[str] | None = None

    def validate(self) -> None:
        if self.mode == DomainMode.SIMULATION and self.scenario_result is None:
            raise ContractValidationError("simulation snapshots require a scenario_result")
        if self.mode == DomainMode.FORECAST and self.forecast_run is None and self.demand_forecast is None:
            raise ContractValidationError("forecast snapshots require a forecast_run or demand_forecast")
        if self.source.mode != self.mode and not (self.mode == DomainMode.LIVE and self.source.mode == DomainMode.FORECAST):
            raise ContractValidationError("snapshot mode and provenance mode must align")
        if self.regional_demand_context:
            for region in self.regional_demand_context:
                lowered = region.note.lower()
                if "adequacy status" in lowered or "regional balance status" in lowered:
                    raise ContractValidationError("regional demand context cannot contain regional adequacy status")


@dataclass(frozen=True)
class TwinResponse(ContractBase):
    generated_at: datetime
    from_time: datetime
    hours: int
    region: str | None
    snapshots: list[TwinSnapshot]
    unavailable_fields: list[UnavailableField]

    def validate(self) -> None:
        if self.hours < 0:
            raise ContractValidationError("twin response hours cannot be negative")
        if not self.snapshots:
            raise ContractValidationError("twin response requires at least one snapshot")
        if len(self.snapshots) > self.hours + 1:
            raise ContractValidationError("twin response has more snapshots than current plus requested hours")


CONTRACT_ENUMS: tuple[type[Enum], ...] = (
    DomainMode,
    OperatingState,
    Scope,
    SourceType,
    EstimateProvenanceKind,
    Freshness,
    Status,
    Unit,
    Confidence,
)

CONTRACT_TYPES: tuple[type[Any], ...] = (
    DataQuality,
    DataProvenance,
    QuantifiedValue,
    BaselineDefinition,
    DemandContext,
    GenerationEstimate,
    GenerationMix,
    TwinComponentEstimate,
    RegionalDemandForecast,
    GenerationAvailabilityContext,
    EstimatedGenerationMix,
    ExchangeEstimate,
    CarbonEstimate,
    ModelledBalanceContribution,
    OfficialSignal,
    ModelledBalanceContext,
    ConfidenceAssessment,
    ExplanationDriver,
    Explanation,
    ForecastInterval,
    ForecastPoint,
    ModelCard,
    ForecastRun,
    RegionalState,
    NationalState,
    ScenarioEvent,
    ScenarioRequest,
    ScenarioDelta,
    ScenarioResult,
    NullableMetric,
    FreshnessStatus,
    GenerationTechnologyMetric,
    CurrentGenerationMix,
    EnvironmentalMetric,
    CurrentDemandContext,
    CurrentOfficialSignal,
    CurrentModelledStatus,
    NationalCurrentContext,
    RegionalCurrentContext,
    CurrentMapRegion,
    CacheInfo,
    UnavailableField,
    CurrentStateResponse,
    SourceHealth,
    ModelHealth,
    ScenarioEngineHealth,
    DataHealthResponse,
    SourceMetadata,
    SourcesResponse,
    StatusThresholdsResponse,
    TwinSnapshot,
    TwinResponse,
)


def utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def power_value(
    mw: float | None,
    *,
    event_time: datetime,
    update_time: datetime,
    source: DataProvenance,
    quality: DataQuality,
    label: str | None = None,
) -> QuantifiedValue:
    display = None if mw is None else float(mw) / 1000
    return QuantifiedValue(
        value=None if mw is None else float(mw),
        unit=Unit.MW,
        event_time=event_time,
        update_time=update_time,
        source=source,
        quality=quality,
        is_fallback=source.is_fallback,
        display_value=display,
        display_unit=Unit.GW,
        label=label,
    )


def percentage_value(
    value: float | None,
    *,
    event_time: datetime,
    update_time: datetime,
    source: DataProvenance,
    quality: DataQuality,
    label: str | None = None,
) -> QuantifiedValue:
    return QuantifiedValue(
        value=None if value is None else float(value),
        unit=Unit.PERCENTAGE,
        event_time=event_time,
        update_time=update_time,
        source=source,
        quality=quality,
        is_fallback=source.is_fallback,
        display_value=None if value is None else float(value) * 100,
        display_unit=Unit.PERCENTAGE,
        label=label,
    )


def to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {
            field.name: to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    return value


def from_dict(contract_type: type[Any], data: dict[str, Any]) -> Any:
    value = _from_value(contract_type, data)
    validate_contract(value)
    return value


def validate_contract(value: Any) -> None:
    if isinstance(value, ContractBase):
        value.validate()
        for field in fields(value):
            validate_contract(getattr(value, field.name))
    elif isinstance(value, list | tuple):
        for item in value:
            validate_contract(item)
    elif isinstance(value, dict):
        for item in value.values():
            validate_contract(item)


def schema_document() -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    for enum_type in CONTRACT_ENUMS:
        schemas[enum_type.__name__] = {
            "type": "string",
            "enum": [item.value for item in enum_type],
        }
    for contract_type in CONTRACT_TYPES:
        schemas[contract_type.__name__] = _dataclass_schema(contract_type)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://energy-pulse-france.local/contracts/energy-twin.schema.json",
        "title": "Energy Pulse France Digital Twin Contracts",
        "type": "object",
        "$ref": "#/$defs/TwinSnapshot",
        "$defs": schemas,
    }


def openapi_document() -> dict[str, Any]:
    schemas = _replace_schema_refs(schema_document()["$defs"], "#/$defs/", "#/components/schemas/")
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Energy Pulse France Digital Twin Contracts",
            "version": "1.0.0",
        },
        "paths": {
            "/v1/state/current": {
                "get": {
                    "summary": "Current observed electricity state for the Now page",
                    "parameters": [
                        {
                            "name": "region",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "French administrative region code, for example 11 for Ile-de-France.",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Current national, selected-region, and regional-map context.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CurrentStateResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/v1/data-health": {
                "get": {
                    "summary": "Current data-source health and operating-state report",
                    "responses": {
                        "200": {
                            "description": "Source freshness, fallback, and unavailable-field details.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DataHealthResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/v1/sources": {
                "get": {
                    "summary": "Source catalog for the Now API",
                    "responses": {
                        "200": {
                            "description": "Public data sources and configuration artifacts used by the API.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SourcesResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/v1/config/status-thresholds": {
                "get": {
                    "summary": "Versioned modelled-status threshold configuration",
                    "responses": {
                        "200": {
                            "description": "Thresholds used for modelled balance status. This is not an official signal.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/StatusThresholdsResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/v1/twin": {
                "get": {
                    "summary": "Current and hour-indexed electricity-system twin snapshots",
                    "parameters": [
                        {
                            "name": "from",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "format": "date-time"},
                            "description": "Requested start timestamp. The service aligns to the latest available UTC hour at or before this time.",
                        },
                        {
                            "name": "hours",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0, "maximum": 48},
                            "description": "Number of forecast hours after the current/aligned snapshot.",
                        },
                        {
                            "name": "region",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Optional French administrative region code to place first in regional demand context.",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "One coherent TwinSnapshot for the current/aligned state and each forecast hour.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TwinResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/v1/scenarios/run": {
                "post": {
                    "summary": "Run a deterministic first-generation scenario against a TwinSnapshot baseline",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": [
                                        "scenario_type",
                                        "magnitude",
                                        "scope",
                                        "start_time",
                                        "assumptions",
                                    ],
                                    "properties": {
                                        "scenario_type": {
                                            "type": "string",
                                            "enum": [
                                                "cold_snap",
                                                "generation_unavailability",
                                                "ev_charging_shift",
                                            ],
                                        },
                                        "magnitude": {"type": "object"},
                                        "scope": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                                        "start_time": {"type": "string", "format": "date-time"},
                                        "duration_hours": {"type": "number"},
                                        "end_time": {"type": "string", "format": "date-time"},
                                        "assumptions": {},
                                        "user_label": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": (
                                "Scenario comparison with baseline series, scenario series, deltas, "
                                "causal chain, assumptions, caveats, and model/data versions."
                            ),
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
            "/v1/forecast": {
                "get": {
                    "summary": "Create a 48-hour national demand forecast with grouped explanations",
                    "parameters": [
                        {
                            "name": "scope",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "enum": ["france"]},
                            "description": "Forecast scope. Only national France is currently supported.",
                        },
                        {
                            "name": "hours",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 1, "maximum": 48},
                            "description": "Number of forecast hours to return.",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Forecast run with casual and technical model explanations.",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
            "/v1/forecast/{run-id}": {
                "get": {
                    "summary": "Retrieve a previously generated forecast run",
                    "parameters": [
                        {
                            "name": "run-id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Stored forecast run.",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
            "/v1/explanations": {
                "get": {
                    "summary": "Retrieve casual and technical explanation for one forecast hour",
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "timestamp",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "format": "date-time"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Selected-hour explanation with raw technical SHAP values when available.",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
            "/v1/model-card": {
                "get": {
                    "summary": "Model card for the probabilistic demand forecast and explanation policy",
                    "responses": {
                        "200": {
                            "description": "Forecast model card and explainability limitations.",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
            "/v1/forecast-changes": {
                "get": {
                    "summary": "Explain changes between two forecast runs",
                    "parameters": [
                        {
                            "name": "current",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "previous",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Component comparison across matched forecast timestamps.",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
        },
        "components": {"schemas": schemas},
    }


def typescript_declarations() -> str:
    lines = [
        "/* Generated from src/contracts/energy_twin.py. Do not edit by hand. */",
        "",
    ]
    for enum_type in CONTRACT_ENUMS:
        values = " | ".join(f'"{item.value}"' for item in enum_type)
        lines.append(f"export type {enum_type.__name__} = {values};")
    lines.append("")
    for contract_type in CONTRACT_TYPES:
        hints = get_type_hints(contract_type)
        lines.append(f"export interface {contract_type.__name__} {{")
        for field in fields(contract_type):
            optional = _field_optional(field)
            marker = "?" if optional else ""
            lines.append(f"  {field.name}{marker}: {_ts_type(hints[field.name])};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _from_value(expected_type: Any, value: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return value
    if origin in (Union, types.UnionType):
        non_none = [item for item in args if item is not type(None)]
        if value is None:
            if len(non_none) != len(args):
                return None
            raise ContractValidationError("null is not allowed for this field")
        last_error: Exception | None = None
        for item in non_none:
            try:
                return _from_value(item, value)
            except (TypeError, ValueError, ContractValidationError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    if origin in (list, tuple):
        item_type = args[0] if args else Any
        return [_from_value(item_type, item) for item in (value or [])]
    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {str(key): _from_value(value_type, item) for key, item in (value or {}).items()}
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return value if isinstance(value, expected_type) else expected_type(value)
    if expected_type is datetime:
        return utc_datetime(value)
    if expected_type is float:
        return float(value)
    if expected_type is int:
        return int(value)
    if expected_type is bool:
        return bool(value)
    if expected_type is str:
        return str(value)
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, dict):
            raise ContractValidationError(f"{expected_type.__name__} requires an object")
        field_names = {field.name for field in fields(expected_type)}
        extra_fields = sorted(set(value).difference(field_names))
        if extra_fields:
            raise ContractValidationError(f"{expected_type.__name__} received unknown fields: {extra_fields}")
        hints = get_type_hints(expected_type)
        kwargs: dict[str, Any] = {}
        for field in fields(expected_type):
            if field.name in value:
                kwargs[field.name] = _from_value(hints[field.name], value[field.name])
            elif field.default is not MISSING:
                kwargs[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                kwargs[field.name] = field.default_factory()  # type: ignore[misc]
            else:
                raise ContractValidationError(f"missing required field {expected_type.__name__}.{field.name}")
        return expected_type(**kwargs)
    return value


def _dataclass_schema(contract_type: type[Any]) -> dict[str, Any]:
    hints = get_type_hints(contract_type)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in fields(contract_type):
        properties[field.name] = _type_schema(hints[field.name])
        if not _field_optional(field):
            required.append(field.name)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    if getattr(contract_type, "_schema_description", ""):
        schema["description"] = contract_type._schema_description
    return schema


def _type_schema(type_hint: Any) -> dict[str, Any]:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if type_hint is Any:
        return {}
    if origin in (Union, types.UnionType):
        schemas = [_type_schema(arg) for arg in args if arg is not type(None)]
        if len(schemas) != len(args):
            schemas.append({"type": "null"})
        return {"anyOf": schemas}
    if origin in (list, tuple):
        return {"type": "array", "items": _type_schema(args[0] if args else Any)}
    if origin is dict:
        return {
            "type": "object",
            "additionalProperties": _type_schema(args[1] if len(args) > 1 else Any),
        }
    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        return {"$ref": f"#/$defs/{type_hint.__name__}"}
    if isinstance(type_hint, type) and is_dataclass(type_hint):
        return {"$ref": f"#/$defs/{type_hint.__name__}"}
    if type_hint is datetime:
        return {"type": "string", "format": "date-time"}
    if type_hint is str:
        return {"type": "string"}
    if type_hint is int:
        return {"type": "integer"}
    if type_hint is float:
        return {"type": "number"}
    if type_hint is bool:
        return {"type": "boolean"}
    return {}


def _field_optional(field: Any) -> bool:
    return field.default is not MISSING or field.default_factory is not MISSING  # type: ignore[comparison-overlap]


def _ts_type(type_hint: Any) -> str:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if type_hint is Any:
        return "unknown"
    if origin in (Union, types.UnionType):
        return " | ".join(_ts_type(arg) for arg in args)
    if origin in (list, tuple):
        return f"Array<{_ts_type(args[0] if args else Any)}>"
    if origin is dict:
        return f"Record<string, {_ts_type(args[1] if len(args) > 1 else Any)}>"
    if type_hint is type(None):
        return "null"
    if type_hint is datetime:
        return "string"
    if type_hint in {str, int, float}:
        return "string" if type_hint is str else "number"
    if type_hint is bool:
        return "boolean"
    if isinstance(type_hint, type):
        return type_hint.__name__
    return "unknown"


def _replace_schema_refs(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_schema_refs(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_schema_refs(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(old):
        return new + value[len(old) :]
    return value


def contract_root() -> Path:
    return Path(__file__).resolve().parents[2]
