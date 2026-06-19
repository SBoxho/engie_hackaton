"""Public-data ingestion commands for Energy Pulse France."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.public_data.quality import build_data_health
from src.public_data.registry import (
    current_window,
    default_store,
    history_window,
    ingest_window,
    iter_backfill_windows,
)

DEFAULT_CADENCES = {
    "odre_eco2mix_national": "15min",
    "odre_eco2mix_national_history": "30min",
    "odre_eco2mix_regional": "15min",
    "open_meteo_weather": "1h",
    "open_meteo_current": "15min",
    "french_public_holidays": "1D",
    "french_school_holidays": "1D",
}


def _sources(value: list[str] | None) -> list[str] | None:
    return value if value else None


def _print_ingest_results(results) -> int:
    failures = 0
    for name, result in results:
        if isinstance(result, Exception):
            failures += 1
            print(f"{name}: failed: {result}")
        else:
            print(f"{name}: stored {len(result.silver)} silver rows")
    return 2 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "public")
    sub = parser.add_subparsers(dest="command", required=True)

    current = sub.add_parser("ingest-current", help="Fetch recent/current public data")
    current.add_argument("--hours", type=int, default=72)
    current.add_argument("--source", action="append", dest="sources")

    history = sub.add_parser("ingest-history", help="Fetch a bounded historical window")
    history.add_argument("--start", required=True)
    history.add_argument("--end", required=True)
    history.add_argument("--source", action="append", dest="sources")

    backfill = sub.add_parser("backfill", help="Backfill in repeatable chunks")
    backfill.add_argument("--start", required=True)
    backfill.add_argument("--end", required=True)
    backfill.add_argument("--chunk-days", type=int, default=7)
    backfill.add_argument("--source", action="append", dest="sources")

    validate = sub.add_parser("validate-data", help="Validate stored public-data health")
    validate.add_argument("--fail-on-warning", action="store_true")

    sub.add_parser("show-data-health", help="Print the public-data health report")
    sub.add_parser("manual-integration-test", help="Optional live API smoke for developers")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = default_store(args.root)

    if args.command == "ingest-current":
        return _print_ingest_results(
            ingest_window(store, current_window(args.hours), source_names=_sources(args.sources))
        )
    if args.command == "ingest-history":
        return _print_ingest_results(
            ingest_window(
                store,
                history_window(args.start, args.end),
                source_names=_sources(args.sources),
            )
        )
    if args.command == "backfill":
        status = 0
        for window in iter_backfill_windows(args.start, args.end, chunk_days=args.chunk_days):
            print(f"Backfill window {window.chunk_id}")
            status |= _print_ingest_results(
                ingest_window(store, window, source_names=_sources(args.sources))
            )
        return status
    if args.command in {"validate-data", "show-data-health"}:
        report = build_data_health(store, expected_cadences=DEFAULT_CADENCES)
        payload = report.to_dict()
        print(json.dumps(payload, indent=2))
        if args.command == "show-data-health":
            return 0
        has_failures = any(
            source["schema_failures"] or source["duplicate_intervals"] or source["adapter_failures"]
            for source in payload["sources"]
        )
        has_warnings = any(source["missing_intervals"] or source["fallback_records"] for source in payload["sources"])
        return 2 if has_failures or (args.fail_on_warning and has_warnings) else 0
    if args.command == "manual-integration-test":
        print("Running optional live public API smoke. This command is intentionally not used by tests.")
        return _print_ingest_results(
            ingest_window(store, current_window(3), source_names=["odre-national", "open-meteo", "public-holidays"])
        )
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
