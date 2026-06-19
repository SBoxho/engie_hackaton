/* Generated from src/contracts/energy_twin.py. Do not edit by hand. */

export type DomainMode = "live" | "forecast" | "simulation" | "replay";
export type OperatingState = "fresh_live_data" | "delayed_live_data" | "last_known_good_fallback" | "source_unavailable" | "historical_replay";
export type Scope = "national" | "regional";
export type SourceType = "official" | "observed" | "model" | "fallback" | "scenario";
export type EstimateProvenanceKind = "observed" | "official_forecast" | "statistical_estimate" | "persistence_fallback" | "residual_estimate" | "unavailable";
export type Freshness = "fresh" | "delayed" | "stale" | "unavailable";
export type Status = "normal" | "watch" | "high" | "unknown";
export type Unit = "MW" | "GW" | "MWh" | "GWh" | "percentage" | "tonnes_CO2" | "gCO2_per_kWh";
export type Confidence = "high" | "medium" | "low" | "unavailable";

export interface DataQuality {
  freshness: Freshness;
  confidence: Confidence;
  status?: Status;
  checked_at?: string | null;
  warnings?: Array<string> | null;
}

export interface DataProvenance {
  source_type: SourceType;
  name: string;
  mode: DomainMode;
  event_time: string;
  update_time: string;
  is_fallback?: boolean;
  is_demo?: boolean;
  dataset_id?: string | null;
  url?: string | null;
  retrieved_at?: string | null;
  license?: string | null;
  transformation?: string | null;
  fallback_reason?: string | null;
  replay_label?: string | null;
}

export interface QuantifiedValue {
  value: number | null;
  unit: Unit;
  event_time: string;
  update_time: string;
  source: DataProvenance;
  quality: DataQuality;
  is_fallback?: boolean;
  display_value?: number | null;
  display_unit?: Unit | null;
  label?: string | null;
}

export interface BaselineDefinition {
  baseline_id: string;
  version: string;
  method: string;
  comparison_keys: Array<string>;
  lookback_days?: number | null;
  notes?: string | null;
}

export interface DemandContext {
  current: QuantifiedValue;
  usual: QuantifiedValue;
  anomaly_percentage: QuantifiedValue;
  baseline_definition: BaselineDefinition;
  scope: Scope;
  interpretation: string;
}

export interface GenerationEstimate {
  technology: string;
  power: QuantifiedValue;
  share?: QuantifiedValue | null;
}

export interface GenerationMix {
  total: QuantifiedValue;
  estimates: Array<GenerationEstimate>;
  renewable_share?: QuantifiedValue | null;
  fossil_share?: QuantifiedValue | null;
  co2_intensity?: QuantifiedValue | null;
}

export interface TwinComponentEstimate {
  component: string;
  value: QuantifiedValue;
  provenance_kind: EstimateProvenanceKind;
  included_in_total?: boolean;
  formula?: string | null;
  note?: string | null;
}

export interface RegionalDemandForecast {
  region_code: string;
  region_name: string;
  forecast: ForecastInterval;
  usual: QuantifiedValue;
  unreconciled_p50: QuantifiedValue;
  share_of_national_p50: QuantifiedValue;
  reconciliation_factor: number;
  method: string;
  source: DataProvenance;
  quality: DataQuality;
  note?: string;
}

export interface GenerationAvailabilityContext {
  nuclear: TwinComponentEstimate;
  announced_unavailable: QuantifiedValue;
  announced_unavailability_components: Array<TwinComponentEstimate>;
  unavailable_optional_sources: Array<UnavailableField>;
  method: string;
  source: DataProvenance;
  quality: DataQuality;
}

export interface EstimatedGenerationMix {
  total: QuantifiedValue;
  components: Array<TwinComponentEstimate>;
  residual_bucket_name: string;
  formula: string;
  source: DataProvenance;
  quality: DataQuality;
}

export interface ExchangeEstimate {
  net_imports: QuantifiedValue;
  imports: QuantifiedValue;
  exports: QuantifiedValue;
  provenance_kind: EstimateProvenanceKind;
  method: string;
  source: DataProvenance;
  quality: DataQuality;
}

export interface CarbonEstimate {
  intensity: QuantifiedValue;
  provenance_kind: EstimateProvenanceKind;
  method: string;
  included_in_balance_context: boolean;
  source: DataProvenance;
  quality: DataQuality;
}

export interface ModelledBalanceContribution {
  component: string;
  value: QuantifiedValue;
  weight: number;
  contribution: number;
  source: DataProvenance;
  quality: DataQuality;
  note?: string | null;
}

export interface OfficialSignal {
  name: string;
  scope: Scope;
  status: Status;
  label: string;
  signal_time: string;
  update_time: string;
  source: DataProvenance;
  quality: DataQuality;
  detail?: string | null;
}

export interface ModelledBalanceContext {
  status: Status;
  pressure_ratio: QuantifiedValue;
  available_generation: QuantifiedValue;
  net_imports: QuantifiedValue;
  supply_margin: QuantifiedValue;
  import_requirement: QuantifiedValue;
  threshold_config_version: string;
  source: DataProvenance;
  quality: DataQuality;
  calculation_inputs: Array<string>;
  method?: string;
}

export interface ConfidenceAssessment {
  confidence: Confidence;
  rationale: string;
  source: DataProvenance;
  quality: DataQuality;
  backtest_mae?: QuantifiedValue | null;
  interval_calibration?: string | null;
}

export interface ExplanationDriver {
  name: string;
  contribution: QuantifiedValue;
  direction: string;
  included_in_balance_pressure: boolean;
  source: DataProvenance;
  quality: DataQuality;
  note?: string | null;
}

export interface Explanation {
  explanation_id: string;
  method: string;
  text: string;
  drivers: Array<ExplanationDriver>;
  confidence: ConfidenceAssessment;
  source: DataProvenance;
  quality: DataQuality;
  caveats?: Array<string> | null;
}

export interface ForecastInterval {
  p10: QuantifiedValue;
  p50: QuantifiedValue;
  p90: QuantifiedValue;
  confidence: ConfidenceAssessment;
}

export interface ForecastPoint {
  target_time: string;
  horizon_hours: number;
  demand: ForecastInterval;
  balance_context: ModelledBalanceContext;
  route_source_type: SourceType;
  route_label: string;
  uncertainty: QuantifiedValue;
  explanation?: Explanation | null;
}

export interface ModelCard {
  model_id: string;
  version: string;
  display_name: string;
  purpose: string;
  source_type: SourceType;
  training_period_start: string | null;
  training_period_end: string | null;
  evaluation_metrics: Record<string, number>;
  promoted_horizons: Array<number>;
  provenance: DataProvenance;
  quality: DataQuality;
  limitations: Array<string>;
}

export interface ForecastRun {
  run_id: string;
  mode: DomainMode;
  origin_time: string;
  generated_at: string;
  horizon_hours: number;
  points: Array<ForecastPoint>;
  source: DataProvenance;
  quality: DataQuality;
  model_card?: ModelCard | null;
}

export interface RegionalState {
  region_code: string;
  region_name: string;
  scope: Scope;
  demand_context: DemandContext;
  local_generation: GenerationMix;
  source: DataProvenance;
  quality: DataQuality;
  grid_context_note?: string;
}

export interface NationalState {
  scope: Scope;
  demand_context: DemandContext;
  generation_mix: GenerationMix;
  official_signal: OfficialSignal;
  balance_context: ModelledBalanceContext;
  source: DataProvenance;
  quality: DataQuality;
  regions?: Array<RegionalState> | null;
}

export interface ScenarioEvent {
  event_id: string;
  event_type: string;
  start_time: string;
  end_time: string;
  affected_value: string;
  delta: QuantifiedValue;
  source: DataProvenance;
  quality: DataQuality;
}

export interface ScenarioRequest {
  scenario_id: string;
  mode: DomainMode;
  created_at: string;
  baseline_forecast_run_id: string;
  assumption_version: string;
  events: Array<ScenarioEvent>;
  source: DataProvenance;
  quality: DataQuality;
}

export interface ScenarioDelta {
  metric: string;
  baseline: QuantifiedValue;
  scenario: QuantifiedValue;
  delta: QuantifiedValue;
  explanation: string;
}

export interface ScenarioResult {
  result_id: string;
  request: ScenarioRequest;
  generated_at: string;
  forecast_points: Array<ForecastPoint>;
  deltas: Array<ScenarioDelta>;
  confidence: ConfidenceAssessment;
  source: DataProvenance;
  quality: DataQuality;
}

export interface NullableMetric {
  value: number | null;
  unit: string;
  reason?: string | null;
  source_quality?: string | null;
}

export interface FreshnessStatus {
  state: OperatingState;
  timestamp: string | null;
  retrieved_at: string | null;
  age_seconds: number | null;
  refresh_interval_seconds: number;
  reason?: string | null;
}

export interface GenerationTechnologyMetric {
  technology: string;
  power: NullableMetric;
  share: NullableMetric;
}

export interface CurrentGenerationMix {
  total: NullableMetric;
  technologies: Array<GenerationTechnologyMetric>;
  renewable_share: NullableMetric;
  fossil_share: NullableMetric;
}

export interface EnvironmentalMetric {
  metric: string;
  estimate: NullableMetric;
  included_in_modelled_status: boolean;
  note: string;
}

export interface CurrentDemandContext {
  current: NullableMetric;
  usual: NullableMetric;
  difference_vs_usual_pct: NullableMetric;
  difference_vs_usual_gw: NullableMetric;
  baseline_id: string;
  baseline_method: string;
  baseline_sample_count?: number | null;
  baseline_fallback_level?: number | null;
}

export interface CurrentOfficialSignal {
  name: string;
  signal_type: string;
  available: boolean;
  status: string | null;
  label: string | null;
  timestamp: string | null;
  source: string;
  reason?: string | null;
  detail?: string | null;
}

export interface CurrentModelledStatus {
  signal_type: string;
  status: Status;
  label: string;
  model_id: string;
  model_version: string;
  calculation_inputs: Array<string>;
  threshold_config_version: string;
  reason?: string | null;
}

export interface NationalCurrentContext {
  demand: CurrentDemandContext;
  freshness: FreshnessStatus;
  generation_mix: CurrentGenerationMix;
  physical_imports: NullableMetric;
  physical_exports: NullableMetric;
  net_imports: NullableMetric;
  carbon_estimate: EnvironmentalMetric;
  official_ecowatt_signal: CurrentOfficialSignal;
  modelled_status: CurrentModelledStatus | null;
}

export interface RegionalCurrentContext {
  region_code: string;
  region_name: string;
  demand: CurrentDemandContext;
  freshness: FreshnessStatus;
  local_generation: CurrentGenerationMix;
  net_flow: NullableMetric;
  physical_balance: NullableMetric;
  connected_grid_note: string;
}

export interface CurrentMapRegion {
  region_id: string;
  region_name: string;
  demand_anomaly_pct: NullableMetric;
  observed_demand: NullableMetric;
  usual_demand: NullableMetric;
  source_quality: string;
  availability_flag: boolean;
}

export interface CacheInfo {
  cache_key: string;
  ttl_seconds: number;
  generated_at: string;
  expires_at: string;
  cache_hit: boolean;
}

export interface UnavailableField {
  field: string;
  reason: string;
}

export interface CurrentStateResponse {
  generated_at: string;
  region: string;
  operating_state: OperatingState;
  cache: CacheInfo;
  national_context: NationalCurrentContext;
  selected_region_context: RegionalCurrentContext;
  map: Array<CurrentMapRegion>;
  unavailable_fields: Array<UnavailableField>;
}

export interface SourceHealth {
  source_id: string;
  name: string;
  operating_state: OperatingState;
  freshness: FreshnessStatus;
  source_quality: string;
  missing_intervals?: number;
  fallback_records?: number;
  adapter_failures?: number;
  circuit_breaker_state?: string;
  latest_successful_fetch_at?: string | null;
  reason?: string | null;
}

export interface ModelHealth {
  model_id: string;
  status: string;
  model_version: string | null;
  latest_successful_forecast_at: string | null;
  latest_successful_forecast_run_id: string | null;
  recent_forecast_error_mae_mw: number | null;
  fallback_usage?: string | null;
  reason?: string | null;
}

export interface ScenarioEngineHealth {
  available: boolean;
  version: string;
  assumption_version: string;
  cache_enabled: boolean;
  last_successful_scenario_id?: string | null;
  reason?: string | null;
}

export interface DataHealthResponse {
  generated_at: string;
  operating_state: OperatingState;
  cache: CacheInfo;
  sources: Array<SourceHealth>;
  model_health: ModelHealth;
  scenario_engine: ScenarioEngineHealth;
  unavailable_fields: Array<UnavailableField>;
}

export interface SourceMetadata {
  source_id: string;
  name: string;
  source_type: string;
  dataset_id: string | null;
  url: string | null;
  required_for_now: boolean;
  credential_required: boolean;
  refresh_interval_seconds: number;
  notes: string;
}

export interface SourcesResponse {
  generated_at: string;
  sources: Array<SourceMetadata>;
}

export interface StatusThresholdsResponse {
  generated_at: string;
  version: string;
  thresholds: Record<string, unknown>;
  calculation_inputs: Array<string>;
  excluded_inputs: Array<string>;
  raw_config: Record<string, unknown>;
}

export interface TwinSnapshot {
  snapshot_id: string;
  mode: DomainMode;
  event_time: string;
  update_time: string;
  national: NationalState;
  source: DataProvenance;
  quality: DataQuality;
  regional_states?: Array<RegionalState> | null;
  forecast_run?: ForecastRun | null;
  scenario_result?: ScenarioResult | null;
  explanations?: Array<Explanation> | null;
  model_cards?: Array<ModelCard> | null;
  demand_forecast?: ForecastInterval | null;
  usual_demand_baseline?: QuantifiedValue | null;
  regional_demand_context?: Array<RegionalDemandForecast> | null;
  wind_estimate?: TwinComponentEstimate | null;
  solar_estimate?: TwinComponentEstimate | null;
  generation_availability_context?: GenerationAvailabilityContext | null;
  generation_mix_estimate?: EstimatedGenerationMix | null;
  exchange_estimate?: ExchangeEstimate | null;
  modelled_national_balance_context?: ModelledBalanceContext | null;
  modelled_balance_contributions?: Array<ModelledBalanceContribution> | null;
  official_signal_context?: OfficialSignal | null;
  carbon_estimate?: CarbonEstimate | null;
  provenance_chain?: Array<DataProvenance> | null;
  unsupported_physical_behaviours?: Array<string> | null;
}

export interface TwinResponse {
  generated_at: string;
  from_time: string;
  hours: number;
  region: string | null;
  snapshots: Array<TwinSnapshot>;
  unavailable_fields: Array<UnavailableField>;
}
