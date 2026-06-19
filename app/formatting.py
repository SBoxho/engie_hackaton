from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.i18n import DEFAULT_LOCALE, normalize_locale


UNAVAILABLE = "Unavailable"
_FR_NBSP = "\u202f"
_MONTHS_EN_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTHS_EN_FULL = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTHS_FR_FULL = (
    "janvier",
    "f\u00e9vrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "ao\u00fbt",
    "septembre",
    "octobre",
    "novembre",
    "d\u00e9cembre",
)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number in {float("inf"), float("-inf")}:
        return None
    return number


def format_number(
    value: float | int | None,
    *,
    decimals: int = 0,
    signed: bool = False,
    locale: str = "en",
) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    normalized = normalize_locale(locale, default="en")
    sign = ""
    if number < 0:
        sign = "-"
    elif signed:
        sign = "+"
    text = f"{abs(number):,.{max(decimals, 0)}f}"
    if normalized == "fr-FR":
        text = text.replace(",", _FR_NBSP).replace(".", ",")
    return f"{sign}{text}"


def format_mw(value: float | int | None, *, locale: str = "en") -> str:
    text = format_number(value, locale=locale)
    return UNAVAILABLE if text == UNAVAILABLE else f"{text} MW"


def format_signed_mw(value: float | int | None, *, locale: str = "en") -> str:
    text = format_number(value, signed=True, locale=locale)
    return UNAVAILABLE if text == UNAVAILABLE else f"{text} MW"


def format_gw(value_mw: float | int | None, *, signed: bool = False, locale: str = "en") -> str:
    number = _number(value_mw)
    if number is None:
        return UNAVAILABLE
    return f"{format_number(number / 1000.0, decimals=1, signed=signed, locale=locale)} GW"


def format_percentage(
    value: float | int | None,
    *,
    signed: bool = False,
    already_percent: bool = False,
    locale: str = "en",
) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    percent = number if already_percent else number * 100.0
    space = " " if normalize_locale(locale, default="en") == "fr-FR" else ""
    return f"{format_number(percent, signed=signed, locale=locale)}{space}%"


def format_temperature(value_c: float | int | None, *, signed: bool = False, locale: str = "en") -> str:
    text = format_number(value_c, signed=signed, locale=locale)
    return UNAVAILABLE if text == UNAVAILABLE else f"{text} \u00b0C"


def format_energy(value_mwh: float | int | None, *, signed: bool = False, locale: str = "en") -> str:
    text = format_number(value_mwh, signed=signed, locale=locale)
    return UNAVAILABLE if text == UNAVAILABLE else f"{text} MWh"


def format_timestamp(
    value: datetime | str | pd.Timestamp | None,
    *,
    timezone_name: str = "Europe/Paris",
    include_date: bool = True,
    locale: str = "en",
) -> str:
    local = _local_timestamp(value, timezone_name=timezone_name)
    if local is None:
        return UNAVAILABLE
    if not include_date:
        return f"{format_time(local, timezone_name=timezone_name, locale=locale)} {local.tzname()}"
    return f"{format_date(local, timezone_name=timezone_name, locale=locale, abbreviated=True)} {local:%H:%M} {local.tzname()}"


def format_date(
    value: datetime | str | pd.Timestamp | None,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = DEFAULT_LOCALE,
    abbreviated: bool = False,
) -> str:
    local = _local_timestamp(value, timezone_name=timezone_name)
    if local is None:
        return UNAVAILABLE
    normalized = normalize_locale(locale)
    if normalized == "fr-FR":
        month = _MONTHS_FR_FULL[local.month - 1]
    else:
        month = (_MONTHS_EN_ABBR if abbreviated else _MONTHS_EN_FULL)[local.month - 1]
    return f"{local.day} {month} {local.year}"


def format_time(
    value: datetime | str | pd.Timestamp | None,
    *,
    timezone_name: str = "Europe/Paris",
    locale: str = DEFAULT_LOCALE,
    include_timezone: bool = False,
) -> str:
    local = _local_timestamp(value, timezone_name=timezone_name)
    if local is None:
        return UNAVAILABLE
    result = f"{local:%H:%M}"
    return f"{result} {local.tzname()}" if include_timezone else result


def format_carbon(value: float | int | None, *, unit: str = "gCO2/kWh", signed: bool = False, locale: str = "en") -> str:
    text = format_number(value, signed=signed, locale=locale)
    if text == UNAVAILABLE:
        return UNAVAILABLE
    if unit == "tonnes":
        return f"{text} t CO2"
    return f"{text} {unit}"


def format_uncertainty_range(
    lower: float | int | None,
    upper: float | int | None,
    *,
    unit: str = "MW",
    locale: str = "en",
) -> str:
    low = _number(lower)
    high = _number(upper)
    if low is None or high is None:
        return UNAVAILABLE
    if unit.upper() == "GW":
        return f"{format_number(low / 1000.0, decimals=1, locale=locale)}-{format_number(high / 1000.0, decimals=1, locale=locale)} GW"
    return f"{format_number(low, locale=locale)}-{format_number(high, locale=locale)} {unit}"


def format_age_seconds(value: float | int | None, *, locale: str = "en") -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    if number < 90:
        if normalize_locale(locale, default="en") == "fr-FR":
            return f"{format_number(number, locale=locale)} s"
        return f"{number:.0f} sec old"
    minutes = number / 60.0
    if minutes < 90:
        if normalize_locale(locale, default="en") == "fr-FR":
            return f"{format_number(minutes, locale=locale)} min"
        return f"{minutes:.0f} min old"
    if normalize_locale(locale, default="en") == "fr-FR":
        return f"{format_number(minutes / 60.0, decimals=1, locale=locale)} h"
    return f"{minutes / 60.0:.1f} h old"


def _local_timestamp(value: datetime | str | pd.Timestamp | None, *, timezone_name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.tz_convert(timezone_name)
