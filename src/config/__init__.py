# version: 0.1.0
# path: src/config/__init__.py

"""Utility helpers for configuration files."""

import os


def _read_first_value(path: str, default: str) -> str:
    """Return the first non-comment line from ``path`` or ``default``."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except Exception:
        pass
    return default


def get_pilot_name() -> str:
    """Retrieve the configured pilot name."""

    fname = os.path.join(os.path.dirname(__file__), "pilot_name.txt")
    return _read_first_value(fname, "CitizenZero")


def get_window_title() -> str:
    """Return the default EVE window title."""

    return f"EVE - {get_pilot_name()}"

