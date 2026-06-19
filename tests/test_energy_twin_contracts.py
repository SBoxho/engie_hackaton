from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from src.contracts.energy_twin import (
    BaselineDefinition,
    Confidence,
    ConfidenceAssessment,
    ContractValidationError,
    DataProvenance,
    DataQuality,
    DemandContext,
    DomainMode,
    Explanation,
    ExplanationDriver,
    ForecastInterval,
    ForecastPoint,
    ForecastRun,
    Freshness,
    GenerationEstimate,
    GenerationMix,
    ModelCard,
    ModelledBalanceContext,
    NationalState,
    OfficialSignal,
    QuantifiedValue,
    RegionalState,
    ScenarioDelta,
    ScenarioEvent,
    ScenarioRequest,
    ScenarioResult,
    Scope,
    SourceType,
    Status,
    TwinSnapshot,
    Unit,
    from_dict,
    percentage_value,
    power_value,
    to_dict,
    validate_contract,
)
from src.contracts.status_thresholds import threshold_config_version


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def quality(
    freshness: Freshness = Freshness.FRESH,
    confidence: Confidence = Confidence.HIGH,
    status: Status = Status.NORMAL,
) -> DataQuality:
    return DataQuality(freshness=freshness, confidence=confidence, status=status, checked_at=NOW)


def source(
    source_type: SourceType,
    *,
    mode: DomainMode = DomainMode.LIVE,
    name: str | None = None,
    fallback: bool = False,
    demo: bool = False,
) -> DataProvenance:
    return DataProvenance(
        source_type=source_type,
        name=name or source_type.value,
        mode=mode,
        event_time=NOW,
        update_time=NOW + timedelta(minutes=5),
        is_fallback=fallback,
        is_demo=demo,
        dataset_id="fixture",
        url="https://example.test/fixture",
        fallback_reason="fixture fallback" if fallback else None,
        replay_label="fixture replay data" if demo or mode == DomainMode.REPLAY else None,
    )


def baseline_definition() -> BaselineDefinition:
    return BaselineDefinition(
        baseline_id="usual-demand-comparable-hour",
        version="usual-demand.v1",
        method="median demand for matching season, day type, and local hour",
        comparison_keys=["season", "day_type", "local_hour"],
        lookback_days=28,
    )


def demand_context(scope: Scope, provenance: DataProvenance) -> DemandContext:
    return DemandContext(
        current=power_value(53_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        usual=power_value(50_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        anomaly_percentage=percentage_value(
            0.06,
            event_time=NOW,
            update_time=NOW,
            source=provenance,
            quality=quality(confidence=Confidence.MEDIUM),
        ),
        baseline_definition=baseline_definition(),
        scope=scope,
        interpretation=(
            "Regional demand is comparable-context information inside the connected French grid."
            if scope == Scope.REGIONAL
            else "National demand compared with a machine-readable usual-demand baseline."
        ),
    )


def generation_mix(provenance: DataProvenance) -> GenerationMix:
    return GenerationMix(
        total=power_value(58_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        estimates=[
            GenerationEstimate(
                technology="nuclear",
                power=power_value(38_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
                share=percentage_value(0.66, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
            ),
            GenerationEstimate(
                technology="wind",
                power=power_value(5_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
                share=percentage_value(0.09, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
            ),
        ],
        renewable_share=percentage_value(0.31, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        fossil_share=percentage_value(0.04, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        co2_intensity=QuantifiedValue(
            value=42,
            unit=Unit.TONNES_CO2,
            event_time=NOW,
            update_time=NOW,
            source=provenance,
            quality=quality(),
            is_fallback=False,
            label="CO2 intensity context",
        ),
    )


def balance_context(provenance: DataProvenance, status: Status = Status.NORMAL) -> ModelledBalanceContext:
    return ModelledBalanceContext(
        status=status,
        pressure_ratio=percentage_value(0.88, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        available_generation=power_value(58_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        net_imports=power_value(2_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        supply_margin=power_value(7_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        import_requirement=power_value(0, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        threshold_config_version=threshold_config_version(),
        source=provenance,
        quality=quality(),
        calculation_inputs=["demand_mw", "available_generation_mw", "net_imports_mw"],
    )


def official_signal() -> OfficialSignal:
    official = source(SourceType.OFFICIAL, name="EcoWatt ODRE")
    return OfficialSignal(
        name="EcoWatt",
        scope=Scope.NATIONAL,
        status=Status.NORMAL,
        label="EcoWatt normal",
        signal_time=NOW,
        update_time=NOW,
        source=official,
        quality=quality(),
    )


def national_state() -> NationalState:
    observed = source(SourceType.OBSERVED, name="RTE eCO2mix")
    return NationalState(
        scope=Scope.NATIONAL,
        demand_context=demand_context(Scope.NATIONAL, observed),
        generation_mix=generation_mix(observed),
        official_signal=official_signal(),
        balance_context=balance_context(observed),
        source=observed,
        quality=quality(),
    )


def regional_state() -> RegionalState:
    observed = source(SourceType.OBSERVED, name="RTE regional eCO2mix")
    return RegionalState(
        region_code="11",
        region_name="Ile-de-France",
        scope=Scope.REGIONAL,
        demand_context=demand_context(Scope.REGIONAL, observed),
        local_generation=generation_mix(observed),
        source=observed,
        quality=quality(),
    )


def confidence(provenance: DataProvenance) -> ConfidenceAssessment:
    return ConfidenceAssessment(
        confidence=Confidence.MEDIUM,
        rationale="Fixture forecast route has representative backtest metadata.",
        source=provenance,
        quality=quality(confidence=Confidence.MEDIUM),
        backtest_mae=power_value(1_800, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        interval_calibration="p10-p90 fixture interval",
    )


def explanation(provenance: DataProvenance) -> Explanation:
    driver = ExplanationDriver(
        name="Demand",
        contribution=power_value(53_000, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        direction="raises_pressure",
        included_in_balance_pressure=True,
        source=provenance,
        quality=quality(),
    )
    uncertainty = ExplanationDriver(
        name="Uncertainty",
        contribution=power_value(1_800, event_time=NOW, update_time=NOW, source=provenance, quality=quality()),
        direction="widens_interval",
        included_in_balance_pressure=False,
        source=provenance,
        quality=quality(),
    )
    return Explanation(
        explanation_id="explain-1",
        method="fixture grouped drivers",
        text="Demand and margin explain the modelled balance context.",
        drivers=[driver, uncertainty],
        confidence=confidence(provenance),
        source=provenance,
        quality=quality(confidence=Confidence.MEDIUM),
        caveats=["Fixture explanation; not causal."],
    )


def forecast_point(hour: int) -> ForecastPoint:
    model = source(SourceType.MODEL, mode=DomainMode.FORECAST, name="Demand model fixture")
    target = NOW + timedelta(hours=hour)
    p50 = 52_000 + hour * 20
    return ForecastPoint(
        target_time=target,
        horizon_hours=hour,
        demand=ForecastInterval(
            p10=power_value(p50 - 1_800, event_time=target, update_time=NOW, source=model, quality=quality()),
            p50=power_value(p50, event_time=target, update_time=NOW, source=model, quality=quality()),
            p90=power_value(p50 + 1_800, event_time=target, update_time=NOW, source=model, quality=quality()),
            confidence=confidence(model),
        ),
        balance_context=balance_context(model),
        route_source_type=SourceType.MODEL,
        route_label="validated_model",
        uncertainty=power_value(1_800, event_time=target, update_time=NOW, source=model, quality=quality()),
        explanation=explanation(model),
    )


def forecast_run(hours: int = 48) -> ForecastRun:
    model = source(SourceType.MODEL, mode=DomainMode.FORECAST, name="Demand model fixture")
    return ForecastRun(
        run_id="forecast-fixture",
        mode=DomainMode.FORECAST,
        origin_time=NOW,
        generated_at=NOW + timedelta(minutes=2),
        horizon_hours=hours,
        points=[forecast_point(hour) for hour in range(1, hours + 1)],
        source=model,
        quality=quality(confidence=Confidence.MEDIUM),
        model_card=ModelCard(
            model_id="demand-fixture",
            version="model-card.v1",
            display_name="Fixture demand model",
            purpose="Contract serialization fixture",
            source_type=SourceType.MODEL,
            training_period_start=NOW - timedelta(days=60),
            training_period_end=NOW - timedelta(days=1),
            evaluation_metrics={"mae_mw": 1800.0},
            promoted_horizons=[1, 2, 3],
            provenance=model,
            quality=quality(confidence=Confidence.MEDIUM),
            limitations=["Representative fixture only."],
        ),
    )


def test_live_national_snapshot_serializes_and_validates() -> None:
    observed = source(SourceType.OBSERVED, name="RTE eCO2mix")
    snapshot = TwinSnapshot(
        snapshot_id="live-national-fixture",
        mode=DomainMode.LIVE,
        event_time=NOW,
        update_time=NOW,
        national=national_state(),
        regional_states=[regional_state()],
        source=observed,
        quality=quality(),
    )

    payload = to_dict(snapshot)
    restored = from_dict(TwinSnapshot, payload)

    validate_contract(restored)
    assert restored.national.official_signal.status == Status.NORMAL
    assert restored.national.balance_context.status == Status.NORMAL
    assert restored.national.official_signal.source.source_type == SourceType.OFFICIAL
    assert restored.national.balance_context.source.source_type == SourceType.OBSERVED


def test_regional_state_uses_local_generation_and_rejects_regional_ecowatt() -> None:
    state = regional_state()
    payload = to_dict(state)
    schema = json.loads((ROOT / "docs" / "contracts" / "energy-twin.schema.json").read_text(encoding="utf-8"))

    assert "local_generation" in payload
    assert "regional_supply" not in json.dumps(schema)
    assert "connected French grid" in state.demand_context.interpretation

    with pytest.raises(ContractValidationError, match="national-only"):
        OfficialSignal(
            name="EcoWatt",
            scope=Scope.REGIONAL,
            status=Status.WATCH,
            label="Invalid regional EcoWatt",
            signal_time=NOW,
            update_time=NOW,
            source=source(SourceType.OFFICIAL, name="EcoWatt ODRE"),
            quality=quality(),
        )


def test_48_hour_forecast_round_trip_keeps_uncertainty_out_of_pressure() -> None:
    run = forecast_run(48)
    payload = to_dict(run)
    restored = from_dict(ForecastRun, payload)

    assert len(restored.points) == 48
    assert restored.points[0].demand.p10.value < restored.points[0].demand.p50.value
    assert restored.points[0].uncertainty.label is None
    for point in restored.points:
        inputs = set(point.balance_context.calculation_inputs)
        assert inputs == {"demand_mw", "available_generation_mw", "net_imports_mw"}
        assert "carbon" not in " ".join(inputs).lower()
        assert "uncertainty" not in " ".join(inputs).lower()


def test_unavailable_source_carries_quality_and_fallback_status() -> None:
    fallback = source(
        SourceType.FALLBACK,
        mode=DomainMode.LIVE,
        name="Unavailable source fallback",
        fallback=True,
    )
    unavailable_quality = quality(Freshness.UNAVAILABLE, Confidence.UNAVAILABLE, Status.UNKNOWN)
    value = QuantifiedValue(
        value=None,
        unit=Unit.MW,
        event_time=NOW,
        update_time=NOW,
        source=fallback,
        quality=unavailable_quality,
        is_fallback=True,
        label="Unavailable MW value",
    )

    restored = from_dict(QuantifiedValue, to_dict(value))

    assert restored.value is None
    assert restored.source.is_fallback
    assert restored.quality.freshness == Freshness.UNAVAILABLE


def test_simulation_result_serializes_with_scenario_provenance() -> None:
    scenario = source(SourceType.SCENARIO, mode=DomainMode.SIMULATION, name="Scenario fixture")
    event = ScenarioEvent(
        event_id="cold-snap",
        event_type="cold_snap",
        start_time=NOW + timedelta(hours=18),
        end_time=NOW + timedelta(hours=22),
        affected_value="demand_mw",
        delta=power_value(2_500, event_time=NOW, update_time=NOW, source=scenario, quality=quality()),
        source=scenario,
        quality=quality(confidence=Confidence.LOW),
    )
    request = ScenarioRequest(
        scenario_id="scenario-fixture",
        mode=DomainMode.SIMULATION,
        created_at=NOW,
        baseline_forecast_run_id="forecast-fixture",
        assumption_version="scenario-assumptions.v1",
        events=[event],
        source=scenario,
        quality=quality(confidence=Confidence.LOW),
    )
    result = ScenarioResult(
        result_id="scenario-result-fixture",
        request=request,
        generated_at=NOW + timedelta(minutes=1),
        forecast_points=[forecast_point(1), forecast_point(2)],
        deltas=[
            ScenarioDelta(
                metric="peak_demand_mw",
                baseline=power_value(53_000, event_time=NOW, update_time=NOW, source=scenario, quality=quality()),
                scenario=power_value(55_500, event_time=NOW, update_time=NOW, source=scenario, quality=quality()),
                delta=power_value(2_500, event_time=NOW, update_time=NOW, source=scenario, quality=quality()),
                explanation="Cold snap raises demand in the selected window.",
            )
        ],
        confidence=confidence(scenario),
        source=scenario,
        quality=quality(confidence=Confidence.LOW),
    )

    restored = from_dict(ScenarioResult, to_dict(result))

    assert restored.request.mode == DomainMode.SIMULATION
    assert restored.source.source_type == SourceType.SCENARIO
    assert restored.deltas[0].delta.display_unit == Unit.GW


def test_invalid_official_modelled_and_balance_inputs_are_rejected() -> None:
    with pytest.raises(ContractValidationError, match="official provenance"):
        OfficialSignal(
            name="EcoWatt",
            scope=Scope.NATIONAL,
            status=Status.WATCH,
            label="Model-generated status",
            signal_time=NOW,
            update_time=NOW,
            source=source(SourceType.MODEL, mode=DomainMode.FORECAST),
            quality=quality(),
        )

    with pytest.raises(ContractValidationError, match="must not be labelled official"):
        balance_context(source(SourceType.OFFICIAL, name="EcoWatt ODRE"))

    with pytest.raises(ContractValidationError, match="carbon and uncertainty"):
        from_dict(
            ModelledBalanceContext,
            {
                **to_dict(balance_context(source(SourceType.MODEL, mode=DomainMode.FORECAST))),
                "source": to_dict(source(SourceType.MODEL, mode=DomainMode.FORECAST)),
                "calculation_inputs": ["demand_mw", "co2_intensity", "forecast_uncertainty"],
            },
        )


def test_generated_contract_artifacts_expose_frontend_types() -> None:
    openapi = json.loads((ROOT / "docs" / "contracts" / "energy-twin.openapi.json").read_text(encoding="utf-8"))
    declarations = (ROOT / "app" / "generated" / "energy_twin.d.ts").read_text(encoding="utf-8")

    assert "TwinSnapshot" in openapi["components"]["schemas"]
    assert "ForecastRun" in openapi["components"]["schemas"]
    assert "export interface TwinSnapshot" in declarations
    assert "export type SourceType" in declarations


def test_quantified_values_carry_event_update_source_quality_and_fallback() -> None:
    snapshot = TwinSnapshot(
        snapshot_id="metadata-fixture",
        mode=DomainMode.LIVE,
        event_time=NOW,
        update_time=NOW,
        national=national_state(),
        regional_states=[regional_state()],
        source=source(SourceType.OBSERVED),
        quality=quality(),
    )
    values = list(_quantified_values(snapshot))

    assert values
    for value in values:
        assert value.event_time
        assert value.update_time
        assert value.source
        assert value.quality
        assert value.is_fallback == value.source.is_fallback


def _quantified_values(value: Any) -> list[QuantifiedValue]:
    if isinstance(value, QuantifiedValue):
        return [value]
    if is_dataclass(value):
        result: list[QuantifiedValue] = []
        for field in fields(value):
            result.extend(_quantified_values(getattr(value, field.name)))
        return result
    if isinstance(value, list | tuple):
        result: list[QuantifiedValue] = []
        for item in value:
            result.extend(_quantified_values(item))
        return result
    if isinstance(value, dict):
        result: list[QuantifiedValue] = []
        for item in value.values():
            result.extend(_quantified_values(item))
        return result
    return []
