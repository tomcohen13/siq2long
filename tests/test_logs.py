"""The log file is a run artifact, so its content matters as much as the console's."""

import logging

import logs
from logs import ColorFormatter, setup_logging


def _record(level=logging.INFO, msg="hello"):
    return logging.LogRecord("test", level, __file__, 1, msg, None, None)


def test_color_formatter_plain_has_no_escapes():
    line = ColorFormatter(color=False).format(_record())
    assert "\033" not in line
    assert "INFO" in line and "hello" in line


def test_color_formatter_colored_tints_the_level():
    line = ColorFormatter(color=True).format(_record(logging.WARNING))
    assert logs.LEVEL_COLORS[logging.WARNING] in line
    assert line.endswith(logs.RESET)


def test_log_file_is_written_without_escape_codes(tmp_path):
    path = tmp_path / "run.log"
    setup_logging(log_file=path, level=logging.INFO)
    logging.getLogger("run_eval").warning("watch out")

    for handler in logging.getLogger().handlers:
        handler.flush()
    text = path.read_text()
    assert "watch out" in text
    assert "\033" not in text  # colour belongs on the console, never in the artifact


def test_log_file_keeps_debug_below_console_level(tmp_path):
    """Console at INFO, file at DEBUG: the file is the complete record."""
    path = tmp_path / "run.log"
    setup_logging(log_file=path, level=logging.INFO)
    logging.getLogger("inference").debug("per-batch detail")

    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "per-batch detail" in path.read_text()


def test_setup_logging_is_idempotent(tmp_path):
    """Called twice in a notebook, every line would otherwise be duplicated."""
    setup_logging(log_file=tmp_path / "a.log")
    first = len(logging.getLogger().handlers)
    setup_logging(log_file=tmp_path / "b.log")
    assert len(logging.getLogger().handlers) == first


def test_noisy_libraries_are_quieted(tmp_path):
    setup_logging(log_file=tmp_path / "run.log")
    assert logging.getLogger("transformers").level == logging.WARNING


def test_banner_logs_one_line_per_field(tmp_path):
    # Asserted against the file, not caplog: setup_logging strips existing root
    # handlers by design, pytest's capture handler included.
    path = tmp_path / "run.log"
    setup_logging(log_file=path)
    logs.banner(logging.getLogger("run_eval"), "title", {"model": "qwen3-vl", "split": "val"})

    for handler in logging.getLogger().handlers:
        handler.flush()
    text = path.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 3  # title + one per field
    assert "qwen3-vl" in lines[1] and "val" in lines[2]
    # banner used to tint its own message, which put escape codes in the artifact
    assert "\033" not in text
