"""Structured JSON logs. Request text is never logged: embeddings of medical
text are PHI-adjacent, so logs carry counts and ids, not content."""

from __future__ import annotations

import json
import logging
import sys
import time

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "color_message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        doc.update({k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")})
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)


def configure(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).disabled = True  # replaced by our access log
