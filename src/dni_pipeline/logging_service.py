"""Centralised logging utilities with timing support for the DNI pipeline."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    """Stores metadata about a timed pipeline stage."""

    name: str
    status: str
    duration: float
    extra: Dict[str, Any] = field(default_factory=dict)


class LoggingService:
    """Singleton that configures logging and offers helpers for timing stages."""

    _instance: "LoggingService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "LoggingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._configured = False
        self._stage_records: List[StageRecord] = []
        self._records_lock = threading.Lock()

    def configure(self, verbose: bool = False) -> None:
        """Configure the root logger once with the desired verbosity."""
        if self._configured:
            return
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """Return a module-specific logger."""
        return logging.getLogger(name)

    @contextmanager
    def log_stage(
        self,
        stage_name: str,
        *,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager that logs timing information for a pipeline stage.

        Parameters
        ----------
        stage_name:
            Human-readable name for the stage.
        logger:
            Logger to emit messages with. Defaults to the root logger if not provided.
        level:
            Logging level for the start/completion messages (defaults to INFO).
        extra:
            Optional metadata to store alongside the timing result.
        """
        if logger is None:
            logger = logging.getLogger(stage_name)
        metadata = extra.copy() if extra else {}
        start = time.perf_counter()
        logger.log(level, "Stage '%s' started%s", stage_name, self._format_extra(metadata))
        try:
            yield
        except Exception as exc:  # pragma: no cover - defensive logging
            duration = time.perf_counter() - start
            logger.error(
                "Stage '%s' failed after %.3fs%s: %s",
                stage_name,
                duration,
                self._format_extra(metadata),
                exc,
            )
            self._store_stage_record(stage_name, "failed", duration, metadata)
            raise
        else:
            duration = time.perf_counter() - start
            logger.log(
                level,
                "Stage '%s' completed in %.3fs%s",
                stage_name,
                duration,
                self._format_extra(metadata),
            )
            self._store_stage_record(stage_name, "completed", duration, metadata)

    def _store_stage_record(
        self,
        stage_name: str,
        status: str,
        duration: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Persist stage metadata for later inspection."""
        with self._records_lock:
            self._stage_records.append(
                StageRecord(
                    name=stage_name,
                    status=status,
                    duration=duration,
                    extra=dict(metadata),
                )
            )

    def get_stage_records(self) -> List[StageRecord]:
        """Return a snapshot of collected stage timing data."""
        with self._records_lock:
            return list(self._stage_records)

    @staticmethod
    def _format_extra(metadata: Dict[str, Any]) -> str:
        if not metadata:
            return ""
        formatted = ", ".join(f"{key}={value}" for key, value in metadata.items())
        return f" ({formatted})"


logging_service = LoggingService()

