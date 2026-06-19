"""Shared public-data exception types."""

from __future__ import annotations


class PublicDataError(RuntimeError):
    """Base class for public-data ingestion failures."""


class PublicDataSourceError(PublicDataError):
    """Raised when an external public-data source cannot be fetched or parsed."""


class PublicDataValidationError(PublicDataError, ValueError):
    """Raised when records violate the public-data contract."""

