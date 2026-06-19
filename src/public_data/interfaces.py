"""Adapter-facing interface aliases for the public-data foundation."""

from __future__ import annotations

from .models import AdapterBatch, AdapterFailure, AdapterRequest, PublicDataSchema, SourceAdapter


PublicDataAdapter = SourceAdapter

__all__ = [
    "AdapterBatch",
    "AdapterFailure",
    "AdapterRequest",
    "PublicDataAdapter",
    "PublicDataSchema",
    "SourceAdapter",
]
