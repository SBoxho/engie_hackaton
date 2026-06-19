"""UTC/local-time helpers for French public energy data."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

PARIS_TZ = "Europe/Paris"


def to_utc(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def as_utc_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return to_utc(value)


def normalize_utc_series(values: object) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def local_midnight_to_utc(day: str | date, timezone_name: str = PARIS_TZ) -> pd.Timestamp:
    local = pd.Timestamp(day).tz_localize(ZoneInfo(timezone_name))
    return local.tz_convert("UTC")


def paris_day_utc_bounds(day: str | date, timezone_name: str = PARIS_TZ) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = local_midnight_to_utc(day, timezone_name)
    end_day = pd.Timestamp(day).date() + pd.Timedelta(days=1)
    end = local_midnight_to_utc(end_day, timezone_name)
    return start, end


def render_paris(value: str | datetime | pd.Timestamp, fmt: str | None = None) -> str:
    local = to_utc(value).tz_convert(PARIS_TZ)
    return local.strftime(fmt) if fmt else local.isoformat()


def paris_hourly_index_for_day(day: str | date) -> pd.DatetimeIndex:
    return local_day_utc_hours(day, PARIS_TZ)


def local_day_utc_hours(day: str | date, timezone_name: str = PARIS_TZ) -> pd.DatetimeIndex:
    """Return UTC hour starts that cover one Europe/Paris local calendar day.

    DST transition days naturally contain 23 or 25 UTC instants.
    """
    tz = ZoneInfo(timezone_name)
    start_local = pd.Timestamp(day).tz_localize(tz)
    end_local = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
    return pd.date_range(
        start_local.tz_convert("UTC"),
        end_local.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
