"""Configuration loading shared by the central BUCToolkit test runners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"


def _merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an override mapping into a copied base mapping.

    Args:
        base: Default configuration mapping.
        override: Mode-specific values that replace or extend defaults.

    Return:
        A new mapping containing both configurations.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one test configuration YAML file.

    Args:
        path: Configuration file path.

    Return:
        Parsed top-level mapping.

    Raises:
        TypeError: If the YAML document is not a mapping.
    """
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Test configuration must be a mapping: {path}")
    return config


def load_test_config(mode: str = "main", path: str | Path | None = None) -> dict[str, Any]:
    """Load the central test configuration and an optional mode override.

    Args:
        mode: Test mode, currently ``main`` or ``fast``.
        path: Optional path to a replacement default YAML file.

    Return:
        Fully merged test configuration mapping.

    Raises:
        ValueError: If ``mode`` is unsupported.
        FileNotFoundError: If an explicitly selected configuration is missing.
    """
    if mode not in {"main", "fast"}:
        raise ValueError(f"Unsupported test mode: {mode!r}")
    base_path = Path(path) if path is not None else CONFIG_DIR / "main_test.yaml"
    config = _read_yaml(base_path)
    if mode == "fast":
        config = _merge_mapping(config, _read_yaml(CONFIG_DIR / "fast_test.yaml"))
    return config


def get_test_section(config: dict[str, Any], test_name: str) -> dict[str, Any]:
    """Return one test function's configuration section.

    Args:
        config: Full central test configuration.
        test_name: Test method name, for example ``test_MD``.

    Return:
        The requested section, or an empty mapping when it is absent.
    """
    section = config.get(test_name, {})
    if not isinstance(section, dict):
        raise TypeError(f"Configuration section {test_name!r} must be a mapping.")
    return section
