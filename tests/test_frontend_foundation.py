from __future__ import annotations

from datetime import datetime, timezone

from app.components.cards import status_badge_html
from app.components.foundation import (
    provenance_badge_html,
    term_tooltip_html,
    trust_notice_html,
)
from app.formatting import (
    format_carbon,
    format_gw,
    format_mw,
    format_percentage,
    format_timestamp,
    format_uncertainty_range,
)
from app.state import restore_state, state_to_query_params
from app.generated.energy_twin_client import EnergyTwinApiClient, TwinQuery
from src.contracts.energy_twin import DomainMode


def test_provenance_badges_render_all_contract_source_labels() -> None:
    labels = {
        "official": "Official",
        "observed": "Observed",
        "model": "Model estimate",
        "scenario": "Scenario",
        "fallback": "Fallback",
        "replay": "Replay",
    }

    markup = "".join(provenance_badge_html(kind) for kind in labels)

    for kind, label in labels.items():
        assert f"ep-provenance-{kind}" in markup
        assert label in markup
        assert "aria-label=\"Provenance:" in markup


def test_stale_data_state_uses_visible_text_and_status_role() -> None:
    markup = trust_notice_html("stale", "Stale data", "Latest contract is older than the freshness window.")

    assert 'role="status"' in markup
    assert "Stale data" in markup
    assert "Latest contract" in markup


def test_unit_formatting_for_power_percent_carbon_and_uncertainty() -> None:
    assert format_mw(1234.4) == "1,234 MW"
    assert format_gw(1234.4) == "1.2 GW"
    assert format_gw(-1300, signed=True) == "-1.3 GW"
    assert format_percentage(0.126) == "13%"
    assert format_percentage(-0.044, signed=True) == "-4%"
    assert format_carbon(42.3) == "42 gCO2/kWh"
    assert format_carbon(1234, unit="tonnes", signed=True) == "+1,234 t CO2"
    assert format_uncertainty_range(50_000, 54_000) == "50,000-54,000 MW"


def test_timezone_formatting_uses_requested_display_zone() -> None:
    value = datetime(2026, 6, 18, 12, 30, tzinfo=timezone.utc)

    assert format_timestamp(value, timezone_name="Europe/Paris") == "18 Jun 2026 14:30 CEST"
    assert format_timestamp(value, timezone_name="UTC") == "18 Jun 2026 12:30 UTC"


def test_url_state_restoration_and_serialization() -> None:
    state = restore_state(
        {
            "mode": ["forecast"],
            "region": ["FR-11"],
            "t": ["2026-06-18T12:30:00Z"],
            "run": ["forecast-1"],
            "scenario": ["low_wind"],
        }
    )

    assert state.mode == DomainMode.FORECAST
    assert state.selected_region == "11"
    assert state.selected_timestamp is not None
    assert state.selected_forecast_run == "forecast-1"
    assert state.selected_scenario == "low_wind"
    assert state_to_query_params(state) == {
        "mode": "forecast",
        "region": "11",
        "t": "2026-06-18T12:30:00Z",
        "run": "forecast-1",
        "scenario": "low_wind",
    }


def test_term_tooltip_is_keyboard_reachable_and_screen_reader_labelled() -> None:
    markup = term_tooltip_html("usual demand")

    assert 'tabindex="0"' in markup
    assert 'role="note"' in markup
    assert "aria-label=\"usual demand:" in markup
    assert "<abbr" in markup
    assert "comparable-history baseline" in markup


def test_status_badge_has_screen_reader_label_and_visible_status_text() -> None:
    markup = status_badge_html("Watch", "watch")

    assert 'role="status"' in markup
    assert 'aria-label="Status: Watch"' in markup
    assert ">Watch<" in markup
    assert "ep-status-watch" not in markup
    assert "ep-status-yellow" in markup


def test_generated_client_fixture_mode_returns_replay_contract() -> None:
    response = EnergyTwinApiClient().get_twin(TwinQuery(hours=1, region="11"))

    assert response.snapshots
    assert response.snapshots[0].mode == DomainMode.REPLAY
    assert response.snapshots[0].source.is_demo
    assert response.snapshots[0].source.replay_label
    assert "Live" not in provenance_badge_html("replay")
