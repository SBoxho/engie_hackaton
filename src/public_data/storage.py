"""Bronze/Silver/Gold Parquet storage with optional DuckDB access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.public_data.contracts import AdapterResult, FallbackStatus, QualityStatus


CANONICAL_COLUMNS = (
    "event_time",
    "published_at",
    "ingested_at",
    "source_name",
    "source_revision",
    "quality_status",
    "fallback_status",
    "source_record_id",
)


@dataclass(frozen=True)
class LayerWriteResult:
    received_rows: int
    stored_rows: int
    inserted_rows: int
    replaced_rows: int
    partitions_written: tuple[str, ...]
    unchanged_rows: int = 0

    @property
    def updated_rows(self) -> int:
        return self.replaced_rows


@dataclass(frozen=True)
class IngestionWriteResult:
    bronze_path: Path
    silver: LayerWriteResult
    gold: LayerWriteResult
    bronze: LayerWriteResult | None = None


class PublicDataStorageError(RuntimeError):
    """Raised when public-data storage cannot be read or written safely."""


class _PartitionLock:
    def __init__(self, path: Path, timeout: float = 10.0) -> None:
        self.path = path
        self.timeout = timeout
        self.fd: int | None = None

    def __enter__(self) -> "_PartitionLock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid={os.getpid()}\n".encode())
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise PublicDataStorageError(f"Timed out waiting for public-data lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _payload_digest(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row_signature(row: pd.Series, columns: Sequence[str]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for column in columns:
        value = row[column] if column in row.index else pd.NA
        if pd.isna(value):
            signature.append("<NA>")
        elif isinstance(value, pd.Timestamp):
            signature.append(value.isoformat())
        else:
            signature.append(value)
    return tuple(signature)


def ensure_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    result["event_time"] = pd.to_datetime(result["event_time"], utc=True, errors="coerce")
    result["published_at"] = pd.to_datetime(result["published_at"], utc=True, errors="coerce")
    result["ingested_at"] = pd.to_datetime(result["ingested_at"], utc=True, errors="coerce")
    missing_published = result["published_at"].isna()
    if missing_published.any():
        result.loc[missing_published, "published_at"] = result.loc[missing_published, "ingested_at"]
    source_name = result["source_name"].astype("string")
    source_revision = result["source_revision"].astype("string")
    missing_source = source_name.isna() | source_name.str.strip().eq("")
    missing_revision = source_revision.isna() | source_revision.str.strip().eq("")
    if missing_source.any() or missing_revision.any():
        raise PublicDataStorageError("public data records require source_name and source_revision")
    result["source_name"] = source_name.astype(str)
    result["source_revision"] = source_revision.astype(str)
    result["quality_status"] = result["quality_status"].fillna(QualityStatus.OK.value).astype(str)
    result["fallback_status"] = result["fallback_status"].fillna(FallbackStatus.NONE.value).astype(str)
    result["source_record_id"] = result["source_record_id"].fillna("").astype(str)
    invalid = result["event_time"].isna() | result["published_at"].isna() | result["ingested_at"].isna()
    if invalid.any():
        raise PublicDataStorageError(
            "public data records require valid event_time, published_at, and ingested_at"
        )
    return result


class PublicParquetLayer:
    """Generic UTC event-time partitioned Parquet upsert layer."""

    filename = "data.parquet"

    def __init__(
        self,
        root: str | Path,
        *,
        key_columns: Sequence[str] = (
            "event_time",
            "source_name",
            "source_revision",
            "source_record_id",
        ),
    ) -> None:
        self.root = Path(root)
        self.key_columns = tuple(key_columns)

    def _partition_dir(self, source_name: str, event_time: pd.Timestamp) -> Path:
        return (
            self.root
            / f"source={source_name}"
            / f"year={event_time.year:04d}"
            / f"month={event_time.month:02d}"
        )

    @staticmethod
    def _recover(directory: Path) -> None:
        target = directory / PublicParquetLayer.filename
        backup = directory / f"{PublicParquetLayer.filename}.bak"
        for temporary in directory.glob(f".{PublicParquetLayer.filename}.*.tmp"):
            temporary.unlink(missing_ok=True)
        if target.exists():
            try:
                pd.read_parquet(target)
                return
            except Exception:
                pass
        if backup.exists():
            pd.read_parquet(backup)
            os.replace(backup, target)
            return
        if target.exists():
            raise PublicDataStorageError(f"Corrupt Parquet partition has no backup: {target}")

    @staticmethod
    def _atomic_write(frame: pd.DataFrame, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / PublicParquetLayer.filename
        backup = directory / f"{PublicParquetLayer.filename}.bak"
        temporary = directory / f".{PublicParquetLayer.filename}.{uuid.uuid4().hex}.tmp"
        backup_tmp = directory / f".{PublicParquetLayer.filename}.backup.{uuid.uuid4().hex}.tmp"
        try:
            frame.to_parquet(temporary, index=False)
            pd.read_parquet(temporary)
            if target.exists():
                shutil.copy2(target, backup_tmp)
                pd.read_parquet(backup_tmp)
                os.replace(backup_tmp, backup)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
            backup_tmp.unlink(missing_ok=True)

    def upsert(self, frame: pd.DataFrame) -> LayerWriteResult:
        incoming = ensure_canonical_columns(frame)
        if incoming.empty:
            return LayerWriteResult(0, 0, 0, 0, ())
        received = len(incoming)
        missing = sorted(set(self.key_columns).difference(incoming.columns))
        if missing:
            raise PublicDataStorageError(f"missing key columns: {missing}")
        incoming = incoming.drop_duplicates(list(self.key_columns), keep="last")
        inserted = replaced = unchanged = stored = 0
        written: list[str] = []
        for (source_name, year, month), batch in incoming.assign(
            _year=incoming["event_time"].dt.year,
            _month=incoming["event_time"].dt.month,
        ).groupby(["source_name", "_year", "_month"], sort=True):
            event_time = pd.Timestamp(year=int(year), month=int(month), day=1, tz="UTC")
            directory = self._partition_dir(str(source_name), event_time)
            directory.mkdir(parents=True, exist_ok=True)
            with _PartitionLock(directory / ".write.lock"):
                self._recover(directory)
                clean_batch = batch.drop(columns=["_year", "_month"])
                target = directory / self.filename
                existing = ensure_canonical_columns(pd.read_parquet(target)) if target.exists() else None
                if existing is None:
                    merged = clean_batch
                    inserted += len(clean_batch)
                else:
                    comparable_columns = tuple(sorted(set(existing.columns).union(clean_batch.columns)))
                    old_by_key = {
                        key: _row_signature(row, comparable_columns)
                        for key, row in existing.set_index(list(self.key_columns), drop=False).iterrows()
                    }
                    for key, row in clean_batch.set_index(list(self.key_columns), drop=False).iterrows():
                        if key not in old_by_key:
                            inserted += 1
                        elif old_by_key[key] == _row_signature(row, comparable_columns):
                            unchanged += 1
                        else:
                            replaced += 1
                    merged = pd.concat([existing, clean_batch], ignore_index=True, sort=False)
                    merged = merged.drop_duplicates(list(self.key_columns), keep="last")
                merged = merged.sort_values(list(self.key_columns)).reset_index(drop=True)
                self._atomic_write(merged, directory)
                stored += len(merged)
                written.append(str(directory.relative_to(self.root)))
        return LayerWriteResult(received, stored, inserted, replaced, tuple(written), unchanged)

    def read(
        self,
        *,
        start: str | datetime | pd.Timestamp | None = None,
        end: str | datetime | pd.Timestamp | None = None,
        sources: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        start_ts = pd.to_datetime(start, utc=True) if start is not None else None
        end_ts = pd.to_datetime(end, utc=True) if end is not None else None
        source_filter = set(sources or [])
        requested = list(columns) if columns is not None else None
        scan_columns = None
        if requested is not None:
            scan_columns = list(dict.fromkeys([*requested, "event_time", "source_name"]))
        frames: list[pd.DataFrame] = []
        for path in sorted(self.root.glob("source=*/year=*/month=*/data.parquet")):
            source_name = path.parent.parent.parent.name.split("=", 1)[1]
            if source_filter and source_name not in source_filter:
                continue
            self._recover(path.parent)
            part = pd.read_parquet(path, columns=scan_columns)
            part["event_time"] = pd.to_datetime(part["event_time"], utc=True, errors="coerce")
            if start_ts is not None:
                part = part[part["event_time"] >= start_ts]
            if end_ts is not None:
                part = part[part["event_time"] < end_ts]
            if source_filter:
                part = part[part["source_name"].isin(source_filter)]
            if requested is not None:
                part = part[requested]
            frames.append(part)
        if not frames:
            return pd.DataFrame(columns=requested or list(CANONICAL_COLUMNS))
        return pd.concat(frames, ignore_index=True, sort=False)


def build_hourly_gold(silver: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_canonical_columns(silver)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["event_time"] = frame["event_time"].dt.floor("h")
    key_columns = ["event_time", "source_name", "source_revision", "source_record_id"]
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {"quality_score"}
    ]
    aggregations: dict[str, str] = {column: "mean" for column in numeric}
    for column in frame.columns:
        if column in key_columns or column in aggregations:
            continue
        aggregations[column] = "last"
    return frame.groupby(key_columns, dropna=False, as_index=False).agg(aggregations)


class PublicDataStore:
    def __init__(self, root: str | Path, *, layer: str = "silver") -> None:
        self.root = Path(root)
        self.bronze_root = self.root / "bronze"
        self.bronze = PublicParquetLayer(self.root / "bronze_index")
        self.silver = PublicParquetLayer(self.root / "silver")
        self.gold = PublicParquetLayer(self.root / "gold")
        self.failures_path = self.root / "adapter_failures.jsonl"
        self.last_known_good_root = self.root / "last_known_good"
        self.layer = layer

    def _selected_layer(self) -> PublicParquetLayer:
        if self.layer == "bronze":
            return self.bronze
        if self.layer == "gold":
            return self.gold
        return self.silver

    def upsert(self, frame: pd.DataFrame) -> LayerWriteResult:
        result = self._selected_layer().upsert(frame)
        sources = sorted(set(ensure_canonical_columns(frame)["source_name"])) if not frame.empty else []
        for source_name in sources:
            self.refresh_last_known_good(str(source_name))
        return result

    def read(self, **filters: Any) -> pd.DataFrame:
        return self._selected_layer().read(**filters)

    def write_bronze(self, result: AdapterResult) -> Path:
        payload = {
            "provenance": result.provenance.to_dict(),
            "failures": list(result.failures),
            "fallback_status": result.fallback_status.value,
            "payload": result.bronze_payload,
        }
        digest = _payload_digest(payload)
        stamp = result.provenance.ingested_at.strftime("%Y%m%dT%H%M%SZ")
        path = self.bronze_root / result.source_name / f"{stamp}_{digest[:16]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        return path

    def write_bronze_index(self, result: AdapterResult, bronze_path: Path) -> LayerWriteResult:
        ingested_at = pd.Timestamp(result.provenance.ingested_at)
        if ingested_at.tzinfo is None:
            ingested_at = ingested_at.tz_localize("UTC")
        else:
            ingested_at = ingested_at.tz_convert("UTC")
        published_at = (
            pd.Timestamp(result.provenance.published_at)
            if result.provenance.published_at is not None
            else ingested_at
        )
        frame = pd.DataFrame(
            [
                {
                    "event_time": ingested_at,
                    "published_at": published_at,
                    "ingested_at": ingested_at,
                    "source_name": result.source_name,
                    "source_revision": result.source_revision,
                    "quality_status": (
                        QualityStatus.SOURCE_FAILURE.value if result.failures else QualityStatus.OK.value
                    ),
                    "fallback_status": result.fallback_status.value,
                    "source_record_id": bronze_path.stem,
                    "bronze_payload_path": str(bronze_path),
                }
            ]
        )
        return self.bronze.upsert(frame)

    def write_result(self, result: AdapterResult) -> IngestionWriteResult:
        bronze_path = self.write_bronze(result)
        bronze_result = self.write_bronze_index(result, bronze_path)
        silver_frame = ensure_canonical_columns(result.silver)
        silver_result = self.silver.upsert(silver_frame)
        gold_result = self.gold.upsert(build_hourly_gold(silver_frame))
        self.refresh_last_known_good(result.source_name)
        if result.failures:
            for message in result.failures:
                self.record_adapter_failure(result.source_name, message)
        return IngestionWriteResult(bronze_path, silver_result, gold_result, bronze_result)

    def record_adapter_failure(self, source_name: str, message: str) -> None:
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "source_name": source_name,
            "message": message,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def latest_event_time(self, source_name: str) -> pd.Timestamp | None:
        frame = self.silver.read(sources=[source_name])
        if frame.empty:
            return None
        return pd.to_datetime(frame["event_time"], utc=True, errors="coerce").max()

    def _last_known_good_layer(self) -> PublicParquetLayer:
        return PublicParquetLayer(self.last_known_good_root / "silver")

    def refresh_last_known_good(self, source_name: str) -> LayerWriteResult | None:
        frame = self.silver.read(sources=[source_name])
        if frame.empty:
            return None
        usable = ensure_canonical_columns(frame)
        usable = usable[
            (usable["fallback_status"].isin([FallbackStatus.NONE.value, "primary"]))
            & (usable["quality_status"].isin([QualityStatus.OK.value, QualityStatus.PARTIAL.value, "valid", "warning"]))
        ]
        if usable.empty:
            return None
        return self._last_known_good_layer().upsert(usable)

    def last_known_good(self, source_name: str) -> pd.DataFrame:
        return self._last_known_good_layer().read(sources=[source_name])

    def last_known_good_fallback(
        self,
        source_name: str,
        *,
        reason: str,
        now: datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self.last_known_good(source_name)
        if frame.empty:
            return frame
        ingested_at = pd.Timestamp(now or datetime.now(timezone.utc))
        if ingested_at.tzinfo is None:
            ingested_at = ingested_at.tz_localize("UTC")
        else:
            ingested_at = ingested_at.tz_convert("UTC")
        fallback = frame.copy()
        fallback["ingested_at"] = ingested_at
        fallback["quality_status"] = QualityStatus.PARTIAL.value
        fallback["fallback_status"] = FallbackStatus.LAST_KNOWN_GOOD.value
        fallback["fallback_reason"] = reason
        return fallback

    def duckdb_connection(self) -> Any:
        try:
            import duckdb
        except ModuleNotFoundError as exc:
            raise PublicDataStorageError(
                "duckdb is not installed; install requirements.txt to use DuckDB queries"
            ) from exc
        self.root.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.root / "catalog.duckdb"))
        for layer, root in (
            ("bronze", self.root / "bronze_index"),
            ("silver", self.root / "silver"),
            ("gold", self.root / "gold"),
        ):
            files = sorted(root.glob("source=*/year=*/month=*/data.parquet"))
            if files:
                connection.execute(
                    f"""
                    CREATE OR REPLACE VIEW {layer}_public_data AS
                    SELECT * FROM read_parquet(?, union_by_name=true)
                    """,
                    [[path.as_posix() for path in files]],
                )
            else:
                connection.execute(
                    f"""
                    CREATE OR REPLACE VIEW {layer}_public_data AS
                    SELECT
                        NULL::TIMESTAMPTZ AS event_time,
                        NULL::TIMESTAMPTZ AS published_at,
                        NULL::TIMESTAMPTZ AS ingested_at,
                        NULL::VARCHAR AS source_name,
                        NULL::VARCHAR AS source_revision,
                        NULL::VARCHAR AS quality_status,
                        NULL::VARCHAR AS fallback_status,
                        NULL::VARCHAR AS source_record_id
                    WHERE FALSE
                    """
                )
        return connection


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
