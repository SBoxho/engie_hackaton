from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.components.now_dashboard import build_current_state_map_frame, generation_mix_figure, generation_mix_items
from app.components.regional_map import has_meaningful_anomaly_signal, regional_anomaly_choropleth
from app.components.theme import build_theme_css
from app.components.foundation import notices_from_contracts
from app.i18n import t, translation_keys
from src.data_sources.rte_eco2mix_regional import fallback_region_geojson
from src.contracts.energy_twin import OperatingState
from tests.test_current_state_api import NOW, bundle, service_for


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_now_translation_keys_match_between_supported_locales() -> None:
    assert translation_keys("fr-FR", "now") == translation_keys("en", "now")


def test_unavailable_regions_are_explicit_on_anomaly_map() -> None:
    response = service_for(bundle(region_codes=("84",))).get_current_state("11")
    frame = build_current_state_map_frame(response)
    unavailable = frame[frame["region_code"].eq("11")].iloc[0]

    assert bool(unavailable["availability_flag"]) is False
    assert pd.isna(unavailable["demand_anomaly_pct"])
    assert "No regional demand record" in unavailable["unavailable_reason"]

    figure = regional_anomaly_choropleth(frame, fallback_region_geojson())
    assert figure.data[0].zmin == -1
    assert figure.data[0].zmid == 0
    assert figure.data[0].zmax == 1
    assert list(figure.data[0].z) == pytest.approx([0.7440995233112476])
    assert list(figure.data[0].colorbar.ticktext) == ["-1%", "0%", "+1%"]
    assert any("11" in list(trace.locations) for trace in figure.data if trace.name == "Unavailable regional data")


def test_now_map_frame_and_tooltips_can_render_french_labels() -> None:
    response = service_for(bundle(region_codes=("84",))).get_current_state("11")
    frame = build_current_state_map_frame(response, locale="fr-FR")
    unavailable = frame[frame["region_code"].eq("11")].iloc[0]

    assert "Aucun relev\u00e9 de demande r\u00e9gionale" in unavailable["unavailable_reason"]
    assert "Indisponible" in unavailable["demand_label"]

    figure = regional_anomaly_choropleth(
        frame,
        fallback_region_geojson(),
        demand_label_title=t("now.regions.map_observed_demand", locale="fr-FR"),
        usual_demand_label_title=t("now.regions.map_usual_demand", locale="fr-FR"),
        difference_label_title=t("now.regions.map_difference", locale="fr-FR"),
        freshness_label_title=t("now.regions.map_freshness", locale="fr-FR"),
        source_label_title=t("now.regions.map_source", locale="fr-FR"),
        colorbar_title=t("now.regions.map_colorbar", locale="fr-FR"),
        anomaly_label_title=t("now.regions.map_demand_anomaly", locale="fr-FR"),
        reason_label_title=t("now.regions.map_reason", locale="fr-FR"),
        unavailable_label=t("now.regions.map_unavailable", locale="fr-FR"),
        available_trace_name=t("now.regions.map_available_name", locale="fr-FR"),
        unavailable_trace_name=t("now.regions.map_unavailable_name", locale="fr-FR"),
    )

    available = next(trace for trace in figure.data if trace.name == "\u00c9cart r\u00e9gional disponible")
    unavailable_trace = next(trace for trace in figure.data if trace.name == "Donn\u00e9es r\u00e9gionales indisponibles")
    assert "Demande observ\u00e9e" in available.hovertemplate
    assert "Niveau de demande habituel" in available.hovertemplate
    assert "\u00c9cart de demande: Indisponible" in unavailable_trace.hovertemplate


def test_anomaly_map_uses_finite_available_z_values_and_adaptive_scale() -> None:
    frame = pd.DataFrame(
        [
            {
                "region_code": "11",
                "region_display": "Ile-de-France",
                "demand_anomaly_pct": -0.8,
                "demand_label": "99 MW",
                "usual_label": "100 MW",
                "difference_label": "-1 MW",
                "freshness_label": "Fresh",
                "source_label": "Test",
                "availability_flag": True,
                "unavailable_reason": "",
            },
            {
                "region_code": "84",
                "region_display": "Auvergne-Rhone-Alpes",
                "demand_anomaly_pct": 0.6,
                "demand_label": "101 MW",
                "usual_label": "100 MW",
                "difference_label": "+1 MW",
                "freshness_label": "Fresh",
                "source_label": "Test",
                "availability_flag": True,
                "unavailable_reason": "",
            },
            {
                "region_code": "53",
                "region_display": "Bretagne",
                "demand_anomaly_pct": float("nan"),
                "demand_label": "Unavailable",
                "usual_label": "Unavailable",
                "difference_label": "Unavailable",
                "freshness_label": "Fresh",
                "source_label": "Test",
                "availability_flag": True,
                "unavailable_reason": "No comparable regional demand baseline.",
            },
        ]
    )

    figure = regional_anomaly_choropleth(frame, fallback_region_geojson())
    available = next(trace for trace in figure.data if trace.name == "Available regional anomaly")
    unavailable = next(trace for trace in figure.data if trace.name == "Unavailable regional data")

    assert available.type == "choropleth"
    assert available.featureidkey == "properties.code"
    assert list(available.locations) == ["11", "84"]
    assert list(available.z) == pytest.approx([-0.8, 0.6])
    assert available.zmin == -1
    assert available.zmid == 0
    assert available.zmax == 1
    assert list(unavailable.locations) == ["53"]


def test_sparse_current_state_map_uses_public_regional_fallback_signal() -> None:
    response = service_for(bundle()).get_current_state("11")
    typed_frame = build_current_state_map_frame(response)

    assert not has_meaningful_anomaly_signal(typed_frame)


def test_stale_and_unavailable_source_notices_are_visible() -> None:
    delayed = service_for(bundle(end=NOW - timedelta(hours=2))).get_current_state("11")
    unavailable_ecowatt = service_for(bundle(ecowatt_available=False)).get_current_state("11")

    assert delayed.operating_state == OperatingState.DELAYED_LIVE_DATA
    assert any(notice.kind == "stale" and "Delayed source data" in notice.title for notice in notices_from_contracts(None, delayed))
    assert any(
        notice.kind == "partial" and "unavailable" in notice.body
        for notice in notices_from_contracts(None, unavailable_ecowatt)
    )


def test_generation_mix_uses_direct_labels_other_group_and_demand_marker() -> None:
    response = service_for(bundle()).get_current_state("11")
    national = response.national_context
    labels = [item["label"] for item in generation_mix_items(national.generation_mix)]
    figure = generation_mix_figure(
        national.generation_mix,
        demand_mw=national.demand.current.value,
        net_imports_mw=national.net_imports.value,
    )

    assert "Other" in labels
    assert figure.layout.showlegend is False
    assert figure.layout.shapes
    assert any("Net imports" in annotation.text for annotation in figure.layout.annotations)


def test_generation_mix_can_render_french_labels_and_annotation() -> None:
    response = service_for(bundle()).get_current_state("11")
    national = response.national_context
    labels = [item["label"] for item in generation_mix_items(national.generation_mix, locale="fr-FR")]
    figure = generation_mix_figure(
        national.generation_mix,
        demand_mw=national.demand.current.value,
        net_imports_mw=national.net_imports.value,
        locale="fr-FR",
    )

    assert "Autres" in labels
    assert any("Importations nettes" in annotation.text for annotation in figure.layout.annotations)
    assert any(annotation.text == "Demande" for annotation in figure.layout.annotations)


def test_now_dashboard_css_has_responsive_layout_rules() -> None:
    css = build_theme_css()

    assert ".ep-now-hero" in css
    assert ".ep-next12" in css
    assert "@media (max-width: 760px)" in css
    assert "grid-template-columns: 1fr" in css


def test_theme_css_keeps_long_badges_and_card_text_inside_containers() -> None:
    css = build_theme_css()
    icon_rule = _css_rule(css, ".ep-icon")
    status_rule = _css_rule(css, ".ep-status")
    provenance_rule = _css_rule(css, ".ep-provenance")
    label_rule = _css_rule(css, ".ep-label")
    title_rule = _css_rule(css, ".ep-title")
    value_rule = _css_rule(css, ".ep-value")

    assert "width: fit-content;" in icon_rule
    assert "min-width: 38px;" in icon_rule
    assert re.search(r"^\s*height:\s*38px;", icon_rule, flags=re.MULTILINE) is None
    assert "max-width: 100%;" in icon_rule
    assert "overflow-wrap: anywhere;" in icon_rule
    assert "white-space: normal;" in icon_rule

    for rule in (status_rule, provenance_rule):
        assert "width: fit-content;" in rule
        assert "max-width: 100%;" in rule
        assert "overflow-wrap: anywhere;" in rule

    for rule in (label_rule, title_rule, value_rule):
        assert "overflow-wrap: anywhere;" in rule


def _css_rule(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n    \}}", css, flags=re.DOTALL)
    assert match is not None
    return match.group("body")


def test_now_page_e2e_hierarchy_and_copy(monkeypatch) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    app = AppTest.from_file(APP_MAIN, default_timeout=20)

    app.run(timeout=45)

    assert not app.exception
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert "Que se passe-t-il maintenant ?" in rendered
    # The "Official EcoWatt signal" and "Modelled national balance context"
    # status rows were intentionally removed from the Now page; the same
    # information is summarized in the Drivers section's Eco/Gen cards.
    assert "Official EcoWatt signal" not in rendered
    assert "Modelled national balance context" not in rendered
    assert "Ruban de demande s\u00e9lectionnable" in rendered
    assert "Demande par rapport au niveau habituel" in rendered
    assert "Contexte de la r\u00e9gion s\u00e9lectionn\u00e9e" in rendered
    assert "Production, demande, \u00e9changes et carbone" in rendered
    assert "EcoWatt indisponible" in rendered
    assert "p\u00e9nurie isol\u00e9e" in rendered.lower()
    assert "regional supply" not in rendered.lower()
    assert "regional risk" not in rendered.lower()


def test_now_page_e2e_english_locale_switch(monkeypatch) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    app = AppTest.from_file(APP_MAIN, default_timeout=20)
    app.query_params["lang"] = ["en"]

    app.run(timeout=45)

    assert not app.exception
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert "What is happening now?" in rendered
    assert "Selectable demand ribbon" in rendered
    assert "Generation, demand, exchange, and carbon" in rendered
