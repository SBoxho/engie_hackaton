"""Minimal WSGI server for the Energy Pulse France backend API."""

from __future__ import annotations

from http import HTTPStatus
import json
import time
import uuid
from typing import Any, Callable
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from src.api.current_state import CurrentStateService, default_service
from src.api.forecast_explanations import ForecastExplanationApiService, default_forecast_service
from src.api.scenarios import ScenarioService, default_scenario_service
from src.api.twin import TwinService, default_twin_service
from src.config import settings
from src.contracts.energy_twin import ContractBase, to_dict
from src.observability import metrics_snapshot, record_request
from src.utils.logging import get_logger, log_event


LOGGER = get_logger(__name__)


def create_app(
    service: CurrentStateService | None = None,
    forecast_service: ForecastExplanationApiService | None = None,
    twin_service: TwinService | None = None,
    scenario_service: ScenarioService | None = None,
) -> Callable[..., list[bytes]]:
    api = service or default_service
    forecasts = forecast_service or default_forecast_service
    twin = twin_service or default_twin_service
    scenarios = scenario_service or default_scenario_service

    def application(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        started = time.perf_counter()
        request_id = _request_id(environ)
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", ""))
        query = parse_qs(str(environ.get("QUERY_STRING", "")))
        response_status = HTTPStatus.INTERNAL_SERVER_ERROR
        error_category: str | None = None

        def respond(status: HTTPStatus, payload: Any, *, cache_control: str | None = None) -> list[bytes]:
            nonlocal response_status
            response_status = status
            return _json_response(
                start_response,
                status,
                payload,
                request_id=request_id,
                environ=environ,
                cache_control=cache_control,
            )

        try:
            if method == "OPTIONS":
                response_status = HTTPStatus.NO_CONTENT
                return _empty_response(start_response, request_id=request_id, environ=environ)
            if path == "/v1/metrics":
                if method != "GET":
                    return respond(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed", "request_id": request_id})
                return respond(HTTPStatus.OK, metrics_snapshot(), cache_control="no-store")
            if path == "/v1/scenarios/run":
                if method != "POST":
                    return respond(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed", "request_id": request_id})
                return respond(
                    HTTPStatus.OK,
                    scenarios.run(_read_json_body(environ)),
                    cache_control="no-store",
                )
            if method != "GET":
                return respond(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed", "request_id": request_id})
            if path == "/v1/state/current":
                region_values = query.get("region", [])
                if not region_values:
                    return respond(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "missing_region", "message": "region query parameter is required", "request_id": request_id},
                    )
                return respond(HTTPStatus.OK, api.get_current_state(region_values[0]))
            if path == "/v1/data-health":
                return respond(HTTPStatus.OK, api.get_data_health(), cache_control="no-store")
            if path == "/v1/sources":
                return respond(HTTPStatus.OK, api.get_sources())
            if path == "/v1/config/status-thresholds":
                return respond(HTTPStatus.OK, api.get_status_thresholds())
            if path == "/v1/twin":
                hours = _query_int(query, "hours", default=48, minimum=0, maximum=48)
                from_timestamp = query.get("from", [None])[0]
                region = query.get("region", [None])[0]
                return respond(
                    HTTPStatus.OK,
                    twin.get_twin(from_timestamp=from_timestamp, hours=hours, region=region),
                )
            if path == "/v1/forecast":
                scope = query.get("scope", ["france"])[0]
                hours = _query_int(query, "hours", default=48, minimum=1, maximum=48)
                return respond(HTTPStatus.OK, forecasts.create_forecast(scope=scope, hours=hours), cache_control="no-store")
            if path.startswith("/v1/forecast/"):
                run_id = path.removeprefix("/v1/forecast/").strip("/")
                if not run_id:
                    return respond(HTTPStatus.BAD_REQUEST, {"error": "missing_run_id", "request_id": request_id})
                return respond(HTTPStatus.OK, forecasts.get_forecast(run_id), cache_control="no-store")
            if path == "/v1/explanations":
                run_values = query.get("run_id", [])
                timestamp_values = query.get("timestamp", [])
                if not run_values or not timestamp_values:
                    return respond(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "error": "missing_parameter",
                            "message": "run_id and timestamp query parameters are required",
                            "request_id": request_id,
                        },
                    )
                return respond(
                    HTTPStatus.OK,
                    forecasts.get_explanation(run_id=run_values[0], timestamp=timestamp_values[0]),
                    cache_control="no-store",
                )
            if path == "/v1/model-card":
                return respond(HTTPStatus.OK, forecasts.get_model_card())
            if path == "/v1/forecast-changes":
                current_values = query.get("current", [])
                previous_values = query.get("previous", [])
                if not current_values or not previous_values:
                    return respond(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "error": "missing_parameter",
                            "message": "current and previous query parameters are required",
                            "request_id": request_id,
                        },
                    )
                return respond(
                    HTTPStatus.OK,
                    forecasts.get_forecast_changes(current=current_values[0], previous=previous_values[0]),
                    cache_control="no-store",
                )
            return respond(HTTPStatus.NOT_FOUND, {"error": "not_found", "request_id": request_id}, cache_control="no-store")
        except ValueError as exc:
            error_category = "bad_request"
            return respond(
                HTTPStatus.BAD_REQUEST,
                {"error": "bad_request", "message": str(exc), "request_id": request_id},
                cache_control="no-store",
            )
        except KeyError as exc:
            error_category = "not_found"
            return respond(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "message": str(exc), "request_id": request_id},
                cache_control="no-store",
            )
        except Exception as exc:
            error_category = "internal_error"
            log_event(
                LOGGER,
                "api_unhandled_error",
                request_id=request_id,
                method=method,
                path=path,
                error_type=type(exc).__name__,
            )
            return respond(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal_error", "request_id": request_id},
                cache_control="no-store",
            )
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            record_request(path, method, response_status.value, duration_ms, error_category)
            log_event(
                LOGGER,
                "api_request",
                request_id=request_id,
                method=method,
                path=path,
                status_code=response_status.value,
                duration_ms=round(duration_ms, 3),
                error_category=error_category,
            )

    return application


def _read_json_body(environ: dict[str, Any]) -> dict[str, Any]:
    length_text = str(environ.get("CONTENT_LENGTH") or "0")
    content_type = str(environ.get("CONTENT_TYPE") or "")
    if "application/json" not in content_type.lower():
        raise ValueError("Content-Type must be application/json")
    try:
        length = int(length_text)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length > settings.api_max_body_bytes:
        raise ValueError(f"request body exceeds {settings.api_max_body_bytes} bytes")
    body = environ.get("wsgi.input")
    if body is None or length <= 0:
        raise ValueError("request body must be a JSON object")
    payload = body.read(length)
    try:
        value = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _query_int(
    query: dict[str, list[str]],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = query.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _request_id(environ: dict[str, Any]) -> str:
    incoming = str(environ.get("HTTP_X_REQUEST_ID") or "").strip()
    if incoming and len(incoming) <= 80:
        return incoming
    return uuid.uuid4().hex


def _cors_headers(environ: dict[str, Any]) -> list[tuple[str, str]]:
    origin = str(environ.get("HTTP_ORIGIN") or "").strip()
    if not origin or origin not in settings.api_allowed_origins:
        return [("Vary", "Origin")]
    return [
        ("Access-Control-Allow-Origin", origin),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, X-Request-ID"),
        ("Access-Control-Max-Age", "600"),
        ("Vary", "Origin"),
    ]


def _empty_response(
    start_response: Callable[..., Any],
    *,
    request_id: str,
    environ: dict[str, Any],
) -> list[bytes]:
    headers = [
        ("Content-Length", "0"),
        ("X-Request-ID", request_id),
        ("Cache-Control", "no-store"),
        *_cors_headers(environ),
    ]
    start_response(f"{HTTPStatus.NO_CONTENT.value} {HTTPStatus.NO_CONTENT.phrase}", headers)
    return []


def _json_response(
    start_response: Callable[..., Any],
    status: HTTPStatus,
    payload: Any,
    *,
    request_id: str,
    environ: dict[str, Any],
    cache_control: str | None = None,
) -> list[bytes]:
    body = _json_bytes(payload)
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", cache_control or "public, max-age=60"),
        ("X-Request-ID", request_id),
        *_cors_headers(environ),
    ]
    start_response(f"{status.value} {status.phrase}", headers)
    return [body]


def _json_bytes(payload: Any) -> bytes:
    if isinstance(payload, ContractBase):
        serializable = payload.to_dict()
    else:
        serializable = to_dict(payload)
    return json.dumps(serializable, ensure_ascii=False, allow_nan=False).encode("utf-8")


application = create_app()


def main() -> int:
    host = "127.0.0.1"
    port = 8000
    with make_server(host, port, application) as server:
        print(f"Serving Energy Pulse France API at http://{host}:{port}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
