from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import streamlit as st


DEFAULT_LOCALE = "fr-FR"
FALLBACK_LOCALE = "en"
SUPPORTED_LOCALES = (DEFAULT_LOCALE, FALLBACK_LOCALE)
LOCALE_QUERY_KEY = "lang"
LOCALE_SESSION_KEY = "locale"
LOCALE_SELECTOR_KEY = "_ep_locale_selector"

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


def normalize_locale(value: Any, *, default: str = DEFAULT_LOCALE) -> str:
    text = str(value or "").strip()
    lowered = text.lower().replace("_", "-")
    if lowered in {"fr", "fr-fr"}:
        return "fr-FR"
    if lowered.startswith("en"):
        return "en"
    return default


def current_locale() -> str:
    query_value = _first(st.query_params, LOCALE_QUERY_KEY)
    if query_value:
        return normalize_locale(query_value)
    return normalize_locale(st.session_state.get(LOCALE_SESSION_KEY))


def init_locale() -> str:
    locale = current_locale()
    st.session_state[LOCALE_SESSION_KEY] = locale
    if st.session_state.get(LOCALE_SELECTOR_KEY) not in SUPPORTED_LOCALES:
        st.session_state[LOCALE_SELECTOR_KEY] = locale
    if _first(st.query_params, LOCALE_QUERY_KEY) != locale:
        st.query_params[LOCALE_QUERY_KEY] = locale
    return locale


def set_locale(locale: str) -> str:
    normalized = normalize_locale(locale)
    st.session_state[LOCALE_SESSION_KEY] = normalized
    st.session_state[LOCALE_SELECTOR_KEY] = normalized
    if _first(st.query_params, LOCALE_QUERY_KEY) != normalized:
        st.query_params[LOCALE_QUERY_KEY] = normalized
    return normalized


def render_language_selector() -> None:
    locale = current_locale()
    st.session_state[LOCALE_SELECTOR_KEY] = locale
    st.radio(
        t("shared.language.label", locale=locale),
        options=list(SUPPORTED_LOCALES),
        format_func=lambda value: t(f"shared.language.options.{locale_key(value)}", locale=locale),
        horizontal=True,
        key=LOCALE_SELECTOR_KEY,
        on_change=_sync_locale_from_selector,
    )


def set_html_lang(locale: str | None = None) -> None:
    normalized = normalize_locale(locale or current_locale())
    st.iframe(
        f"""
        <script>
          const root = window.parent.document.documentElement;
          root.lang = "{html_lang(normalized)}";
        </script>
        """,
        height=1,
        width=1,
        tab_index=-1,
    )


def html_lang(locale: str) -> str:
    return "fr-FR" if normalize_locale(locale) == "fr-FR" else "en"


def locale_key(locale: str) -> str:
    return "fr" if normalize_locale(locale) == "fr-FR" else "en"


def locale_display(locale: str, *, display_locale: str | None = None) -> str:
    return t(f"shared.language.options.{locale_key(locale)}", locale=display_locale or current_locale())


def nav_label(page: str, *, locale: str | None = None) -> str:
    return t(f"shared.nav.{page}", locale=locale)


def mode_label(mode: Any, *, locale: str | None = None) -> str:
    value = getattr(mode, "value", mode)
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return t(f"shared.mode.{key}", locale=locale, default=str(value).title())


def t(key: str, *, locale: str | None = None, default: str | None = None, **params: Any) -> str:
    normalized = normalize_locale(locale or current_locale())
    namespace, path = _split_key(key)
    value = _lookup(load_namespace(normalized, namespace), path)
    if value is None and normalized != FALLBACK_LOCALE:
        value = _lookup(load_namespace(FALLBACK_LOCALE, namespace), path)
    if value is None:
        value = default if default is not None else key
    text = str(value)
    if params:
        try:
            return text.format(**params)
        except (KeyError, ValueError):
            return text
    return text


@lru_cache(maxsize=None)
def load_namespace(locale: str, namespace: str) -> dict[str, Any]:
    normalized = normalize_locale(locale)
    path = _LOCALES_DIR / normalized / f"{namespace}.json"
    if not path.exists() and normalized != FALLBACK_LOCALE:
        path = _LOCALES_DIR / FALLBACK_LOCALE / f"{namespace}.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"translation namespace must be an object: {path}")
    return data


def translation_keys(locale: str, namespace: str) -> set[str]:
    return _flatten_keys(load_namespace(locale, namespace))


def _sync_locale_from_selector() -> None:
    set_locale(str(st.session_state.get(LOCALE_SELECTOR_KEY, DEFAULT_LOCALE)))


def _split_key(key: str) -> tuple[str, str]:
    namespace, separator, path = key.partition(".")
    if not separator:
        return "shared", namespace
    return namespace, path


def _lookup(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _flatten_keys(data: Mapping[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            keys.update(_flatten_keys(value, path))
        else:
            keys.add(path)
    return keys


def _first(values: Mapping[str, Any], key: str) -> str | None:
    if key not in values:
        return None
    value = values[key]
    if isinstance(value, list | tuple):
        return None if not value else str(value[0])
    if value is None:
        return None
    return str(value)
