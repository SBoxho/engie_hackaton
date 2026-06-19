"""HTTP client policy for public-data adapters."""
from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import random
import time
from typing import Any, Callable, Mapping

import requests

from src.config import settings
from src.observability import record_source_failure
from src.public_data.contracts import AdapterConfig, SourceUnavailableError


@dataclass
class HttpResponseMeta:
    url: str
    status_code: int
    headers: Mapping[str, str]


class SourceCircuitBreaker:
    """Small source-level circuit breaker for repeated upstream failures."""

    def __init__(
        self,
        *,
        source_name: str,
        failure_threshold: int | None = None,
        recovery_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source_name = source_name
        self.failure_threshold = max(int(failure_threshold or settings.circuit_breaker_failure_threshold), 1)
        self.recovery_timeout_seconds = max(
            float(recovery_timeout_seconds or settings.circuit_breaker_recovery_seconds),
            0.0,
        )
        self.clock = clock
        self.failure_count = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.recovery_timeout_seconds:
            return "half_open"
        return "open"

    def before_request(self) -> None:
        if self.state == "open":
            raise SourceUnavailableError(f"{self.source_name} circuit breaker is open")

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        record_source_failure(self.source_name)
        if self.failure_count >= self.failure_threshold:
            self.opened_at = self.clock()

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
        }


class PublicDataHttpClient:
    """Small requests wrapper with timeouts, retries, and rate-limit handling."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        config: AdapterConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
        source_name: str = "public_data",
        circuit_breaker: SourceCircuitBreaker | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.config = config or AdapterConfig(
            timeout_seconds=settings.public_http_timeout_seconds,
            max_retries=settings.public_http_max_retries,
            min_interval_seconds=settings.public_http_min_interval_seconds,
        )
        self.sleep = sleep
        self._next_allowed_at = 0.0
        self.source_name = source_name
        self.circuit_breaker = circuit_breaker or SourceCircuitBreaker(source_name=source_name)

    def _wait_for_client_rate_limit(self) -> None:
        delay = self._next_allowed_at - time.monotonic()
        if delay > 0:
            self.sleep(delay)

    def _mark_request(self) -> None:
        if self.config.min_interval_seconds > 0:
            self._next_allowed_at = time.monotonic() + self.config.min_interval_seconds

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(float(value), 0.0)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            return max(parsed.timestamp() - time.time(), 0.0)

    def _server_rate_limit_delay(self, response: requests.Response) -> float | None:
        retry_after = self._retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is not None:
            return retry_after
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            try:
                return max(float(reset) - time.time(), 0.0)
            except ValueError:
                return None
        return None

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None) -> Any:
        headers = {"User-Agent": self.config.user_agent}
        last_error: Exception | None = None
        attempts = max(self.config.max_retries, 0) + 1
        for attempt in range(attempts):
            self.circuit_breaker.before_request()
            self._wait_for_client_rate_limit()
            try:
                response = self.session.get(
                    url,
                    params=dict(params or {}),
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                self._mark_request()
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    delay = self._server_rate_limit_delay(response)
                    if delay is None:
                        delay = min(
                            self.config.backoff_base_seconds * (2 ** attempt),
                            self.config.backoff_max_seconds,
                        )
                        delay *= 1 + random.random() * 0.1
                    if attempt < attempts - 1:
                        self.sleep(delay)
                        continue
                response.raise_for_status()
                payload = response.json()
                self.circuit_breaker.record_success()
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                delay = min(
                    self.config.backoff_base_seconds * (2 ** attempt),
                    self.config.backoff_max_seconds,
                )
                self.sleep(delay)
        self.circuit_breaker.record_failure()
        raise SourceUnavailableError(f"GET {url} failed after {attempts} attempts: {last_error}")
