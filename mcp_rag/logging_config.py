"""Structured logging setup."""

import json
import logging
import logging.config
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
    """Configure all loggers cleanly using dictConfig.

    ``disable_existing_loggers=True`` wipes any handlers added by imported
    libraries before this function is called (pydantic, watchdog, etc.).
    """
    handlers_cfg: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "level": "DEBUG",
            "formatter": fmt,
        },
    }
    handler_names = ["console"]

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        handlers_cfg["file"] = {
            "class": "logging.FileHandler",
            "filename": str(Path(file_path)),
            "encoding": "utf-8",
            "level": "DEBUG",
            "formatter": fmt,
        }
        handler_names.append("file")

    fmt_name = fmt if fmt in ("json", "text") else "json"

    cfg = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "json": {"()": _JsonFormatter},
            "text": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            },
        },
        "handlers": handlers_cfg,
        "root": {
            "level": getattr(logging, level.upper(), logging.INFO),
            "handlers": handler_names,
        },
        "loggers": {
            "uvicorn": {"level": "WARNING", "handlers": [], "propagate": False},
            "uvicorn.access": {"level": "WARNING", "handlers": [], "propagate": False},
            "uvicorn.error": {"level": "WARNING", "handlers": [], "propagate": False},
        },
    }
    logging.config.dictConfig(cfg)

    # Verify we really have exactly one handler per configured stream
    root = logging.getLogger()
    assert len(root.handlers) == len(handler_names), root.handlers
