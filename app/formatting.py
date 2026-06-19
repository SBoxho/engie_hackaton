from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


UNAVAILABLE = "Unavailable"


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


def format_mw(value: float | int | None) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.0f} MW"


def format_signed_mw(value: float | int | None) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:+,.0f} MW"


def format_gw(value_mw: float | int | None, *, signed: bool = False) -> str:
    number = _number(value_mw)
    if number is None:
        return UNAVAILABLE
    sign = "+" if signed else ""
    return f"{number / 1000.0:{sign},.1f} GW"


def format_percentage(value: float | int | None, *, signed: bool = False, already_percent: bool = False) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    percent = number if already_percent else number * 100.0
    sign = "+" if signed else ""
    return f"{percent:{sign}.0f}%"


def format_timestamp(
    value: datetime | str | pd.Timestamp | None,
    *,
    timezone_name: str = "Europe/Paris",
    include_date: bool = True,
) -> str:
    if value is None:
        return UNAVAILABLE
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if pd.isna(timestamp):
        return UNAVAILABLE
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    local = timestamp.tz_convert(timezone_name)
    pattern = "%d %b %Y %H:%M %Z" if include_date else "%H:%M %Z"
    return local.strftime(pattern)


def format_carbon(value: float | int | None, *, unit: str = "gCO2/kWh", signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    sign = "+" if signed else ""
    if unit == "tonnes":
        return f"{number:{sign},.0f} t CO2"
    return f"{number:{sign},.0f} {unit}"


def format_uncertainty_range(
    lower: float | int | None,
    upper: float | int | None,
    *,
    unit: str = "MW",
) -> str:
    low = _number(lower)
    high = _number(upper)
    if low is None or high is None:
        return UNAVAILABLE
    if unit.upper() == "GW":
        return f"{low / 1000.0:,.1f}-{high / 1000.0:,.1f} GW"
    return f"{low:,.0f}-{high:,.0f} {unit}"


def format_age_seconds(value: float | int | None) -> str:
    number = _number(value)
    if number is None:
        return UNAVAILABLE
    if number < 90:
        return f"{number:.0f} sec old"
    minutes = number / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min old"
    return f"{minutes / 60.0:.1f} h old"
