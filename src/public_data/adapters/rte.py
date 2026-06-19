"""Optional credentialed RTE adapters.

These adapters are deliberately inert unless credentials and a feature flag are
provided. The no-secret public ingestion path never depends on them.
"""
from __future__ import annotations

import os

from src.public_data.contracts import DataWindow, PublicDataError


class RteCredentialedAdapterDisabled(PublicDataError):
    """Raised when an optional RTE adapter is requested without credentials."""


class OptionalRteGenerationForecastAdapter:
    name = "rte_generation_forecast_optional"
    source_revision = "rte-open-api:optional"

    def __init__(self, *, token: str | None = None, enabled: bool | None = None) -> None:
        self.token = token or os.getenv("RTE_API_TOKEN")
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("RTE_PUBLIC_DATA_ENABLED", "0").lower() in {"1", "true", "yes"}
        )

    def fetch(self, window: DataWindow):  # pragma: no cover - integration-only placeholder
        if not self.enabled or not self.token:
            raise RteCredentialedAdapterDisabled(
                "RTE generation forecast adapter requires RTE_PUBLIC_DATA_ENABLED=1 and RTE_API_TOKEN"
            )
        raise NotImplementedError("RTE generation forecast API wiring is optional and not enabled yet")


class OptionalRteUnavailabilityAdapter(OptionalRteGenerationForecastAdapter):
    name = "rte_asset_unavailability_optional"
