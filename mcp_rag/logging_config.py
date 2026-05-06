"""Structured logging setup."""

import json
import logging
import sys
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extras", {}).items():
            log_data[key] = value
        return json.dumps(log_data, default=str)


def setup_logging(level: str = "INFO", fmt: str = "json", file_path: str | None = None) -> None:
    """Configure root logger for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    if fmt == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (simple append; rotation handled externally or added later)
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
