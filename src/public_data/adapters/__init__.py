"""Source-specific public-data adapters."""
from __future__ import annotations

from src.public_data.adapters.odre import (
    NationalEco2MixAdapter,
    NationalEco2MixHistoryAdapter,
    RegionalEco2MixAdapter,
)
from src.public_data.adapters.weather_calendar import (
    FrenchPublicHolidayAdapter,
    FrenchSchoolHolidayAdapter,
    OpenMeteoWeatherAdapter,
)

__all__ = [
    "FrenchPublicHolidayAdapter",
    "FrenchSchoolHolidayAdapter",
    "NationalEco2MixAdapter",
    "NationalEco2MixHistoryAdapter",
    "OpenMeteoWeatherAdapter",
    "RegionalEco2MixAdapter",
]
