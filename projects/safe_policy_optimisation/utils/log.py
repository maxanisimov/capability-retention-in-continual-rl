"""Lightweight logging setup for stage scripts.

Stages used to ``print`` progress directly. The pipeline orchestrator captures a
stage's output by ``contextlib.redirect_stdout`` to a per-stage log file, so the
logging handler must follow ``sys.stdout`` *at emit time* rather than binding the
stream when the handler is constructed. :class:`_CurrentStdoutHandler` does that,
which keeps the existing per-stage ``logs/<stage>.log`` capture working while
giving stages a real ``logging`` interface.
"""

from __future__ import annotations

import logging
import sys

PACKAGE_LOGGER = "projects.safe_policy_optimisation"


class _CurrentStdoutHandler(logging.Handler):
    """A stream handler that writes to whatever ``sys.stdout`` is at emit time."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            stream = sys.stdout
            stream.write(message + "\n")
            stream.flush()
        except Exception:  # pragma: no cover - defensive, mirrors logging.StreamHandler
            self.handleError(record)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a stdout handler to the package logger once (idempotent).

    Safe to call at the start of every stage ``run()``: repeated calls do not add
    duplicate handlers, so inline multi-stage runs in one process stay clean.
    """

    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    logger.propagate = False  # don't double-emit through the root logger
    if not any(getattr(handler, "_spo_handler", False) for handler in logger.handlers):
        handler = _CurrentStdoutHandler()
        handler._spo_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the package namespace (propagates to its handler)."""

    if name is None or name == PACKAGE_LOGGER:
        return logging.getLogger(PACKAGE_LOGGER)
    if name.startswith(PACKAGE_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")


def log_info(message: str, *args: object) -> None:
    """Log an informational stage message (ensures logging is configured)."""

    configure_logging()
    get_logger().info(message, *args)


def log_warning(message: str, *args: object) -> None:
    """Log a warning stage message (ensures logging is configured)."""

    configure_logging()
    get_logger().warning(message, *args)
