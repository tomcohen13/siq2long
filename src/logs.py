"""
Console and file logging for evaluation runs.
"""

import logging
import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"

LEVEL_COLORS = {
    logging.DEBUG: "\033[38;5;245m",     # grey
    logging.INFO: "\033[38;5;39m",       # blue
    logging.WARNING: "\033[38;5;214m",   # amber
    logging.ERROR: "\033[38;5;203m",     # red
    logging.CRITICAL: "\033[1;38;5;203m",
}

NOISY_LIBRARIES = ("transformers", "accelerate", "urllib3", "filelock", "httpx", "PIL")

FILE_FORMAT = "%(asctime)s  %(levelname)-8s %(name)s: %(message)s"


def supports_color(stream) -> bool:
    """NO_COLOR is the de facto opt-out; a redirected stream gets no escapes."""
    return (
        hasattr(stream, "isatty")
        and stream.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )


class ColorFormatter(logging.Formatter):
    """`12:04:31  INFO      message  [module]`, with the level tinted by severity."""

    def __init__(self, color: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, self.datefmt)
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if not self.color:
            return f"{stamp}  {record.levelname:<8} {message}  [{record.name}]"

        tint = LEVEL_COLORS.get(record.levelno, "")
        return (
            f"{DIM}{stamp}{RESET}  {tint}{record.levelname:<8}{RESET} "
            f"{message}  {DIM}[{record.name}]{RESET}"
        )


class TqdmHandler(logging.StreamHandler):
    """Route records through `tqdm.write` to keep the progress bar intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def setup_logging(
    log_file: str | Path | None = None,
    level: int | str = logging.INFO,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """
    Install the console handler, and a file handler when `log_file` is given.

    Idempotent: existing root handlers are removed first, so calling this twice
    in a notebook doesn't double every line.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers do the filtering, not the logger
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = TqdmHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(ColorFormatter(color=supports_color(sys.stderr)))
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(log_file, encoding="utf-8")
        to_file.setLevel(logging.DEBUG)  # the file keeps everything
        to_file.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(to_file)

    if quiet_libraries:
        for name in NOISY_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)

    return root


def banner(logger: logging.Logger, title: str, fields: dict) -> None:
    """
    Log a run's configuration as aligned `key: value` lines.

    Plain text on purpose. Messages reach every handler, so tinting here would
    write escape codes into the log file; colour is the console formatter's job.
    Alignment survives in both.
    """
    logger.info(title)
    width = max((len(k) for k in fields), default=0)
    for key, value in fields.items():
        logger.info("  %*s  %s", width, key, value)
