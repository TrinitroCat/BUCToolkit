"""Small shared helpers for reading CLI input files."""

import os
from collections.abc import Mapping
from typing import Any

import yaml


_PATH_FIELDS = (
    "MODEL_FILE",
    "DATA_PATH",
    "VAL_SET_PATH",
    "FSDATA_PATH",
    "DISPDATA_PATH",
    "LOAD_CHK_FILE_PATH",
    "OUTPUT_ROOT",
    "OUTPUT_PATH",
    "PREDICTIONS_SAVE_FILE",
    "CONSTRAINTS_FILE",
)
_NESTED_PATH_FIELDS = (
    ("MD", "CONSTRAINTS_FILE"),
    ("TRAIN", "CHK_SAVE_PATH"),
)


def _field_error(error_type, input_path: str, field: str, message: str):
    """Build a configuration error that identifies its field and source.

    Args:
        error_type: Exception class to instantiate.
        input_path: YAML input path containing the invalid field.
        field: Dotted configuration field name.
        message: Description of the violated contract.

    Returns:
        An unraised exception instance of ``error_type``.
    """
    return error_type(f"Invalid `{field}` in input file `{input_path}`: {message}")


def _absolute_path(value: str, input_directory: str) -> str:
    """Resolve one configured path from the YAML input directory.

    Args:
        value: Absolute, relative, or home-relative configured path.
        input_directory: Directory containing the YAML input file.

    Returns:
        An absolute normalized path.
    """
    value = os.path.expanduser(value)
    if not os.path.isabs(value):
        value = os.path.join(input_directory, value)
    return os.path.abspath(value)


def load_input_config(input_path: str, require_output_root: bool = False) -> dict[str, Any]:
    """Load a CLI input file and resolve its known path fields.

    Relative paths owned by the CLI configuration are resolved from the input
    file's directory. Strings inside user-defined mappings such as
    ``MODEL_CONFIG`` and ``DATA_LOADER_KWARGS`` are left unchanged.

    Args:
        input_path: YAML input file to load.
        require_output_root: Whether absence of both ``OUTPUT_ROOT`` and the
            legacy ``OUTPUT_PATH`` is an error.

    Returns:
        A mapping containing validated configuration data, resolved known
        paths, and output-path fallbacks when an output root is available.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        TypeError: If a known typed field has an incompatible type.
        ValueError: If the YAML document is empty, its top level is not a
            mapping, or an output root is required but missing.
        yaml.YAMLError: If the input is not valid YAML.
    """
    input_path = os.path.abspath(input_path)
    with open(input_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if config is None:
        raise _field_error(ValueError, input_path, "<document>", "YAML is empty")
    if not isinstance(config, Mapping):
        raise _field_error(
            ValueError,
            input_path,
            "<document>",
            f"expected a mapping, got {type(config).__name__}",
        )
    config = dict(config)

    for field in ("TASK", "DATA_TYPE"):
        if field in config and not isinstance(config[field], str):
            raise _field_error(
                TypeError,
                input_path,
                field,
                f"expected a string, got {type(config[field]).__name__}",
            )

    input_directory = os.path.dirname(input_path)
    for field in _PATH_FIELDS:
        if field not in config:
            continue
        value = config[field]
        if not isinstance(value, str):
            raise _field_error(
                TypeError,
                input_path,
                field,
                f"expected a string, got {type(value).__name__}",
            )
        config[field] = _absolute_path(value, input_directory)

    for section, field in _NESTED_PATH_FIELDS:
        if section not in config:
            continue
        section_config = config[section]
        if not isinstance(section_config, Mapping):
            raise _field_error(
                TypeError,
                input_path,
                section,
                f"expected a mapping, got {type(section_config).__name__}",
            )
        section_config = dict(section_config)
        config[section] = section_config
        if field not in section_config:
            continue
        value = section_config[field]
        if not isinstance(value, str):
            raise _field_error(
                TypeError,
                input_path,
                f"{section}.{field}",
                f"expected a string, got {type(value).__name__}",
            )
        section_config[field] = _absolute_path(value, input_directory)

    output_root = config.get("OUTPUT_ROOT")
    if output_root is None:
        output_root = config.get("OUTPUT_PATH")
        if output_root is not None:
            config["OUTPUT_ROOT"] = output_root
    if output_root is None:
        if require_output_root:
            raise _field_error(
                ValueError,
                input_path,
                "OUTPUT_ROOT",
                "set OUTPUT_ROOT or the legacy OUTPUT_PATH",
            )
        return config

    config.setdefault("OUTPUT_PATH", os.path.join(output_root, "logs"))
    task = config.get("TASK", "").upper()
    result_name = "result.pt" if task in {"VIB", "VIBRATIONAL_ANALYSIS"} else "result"
    config.setdefault(
        "PREDICTIONS_SAVE_FILE",
        os.path.join(output_root, "results", result_name),
    )
    train_config = config.get("TRAIN")
    if isinstance(train_config, dict):
        train_config.setdefault("CHK_SAVE_PATH", os.path.join(output_root, "chk"))
    return config


def prepare_output_root(output_root: str) -> str:
    """Prepare the root directory used by one CLI task.

    Args:
        output_root: Directory that will own logs, results, and checkpoints.

    Returns:
        The absolute output-root path. A missing directory is created before
        it is returned.

    Raises:
        ValueError: If the path is a symbolic link, a file, or a non-empty
            directory.
        OSError: If the missing directory cannot be created or inspected.
    """
    output_root = os.path.abspath(os.path.expanduser(output_root))
    if os.path.lexists(output_root):
        if os.path.islink(output_root):
            raise ValueError(f"Output root `{output_root}` must not be a symbolic link.")
        if not os.path.isdir(output_root):
            raise ValueError(f"Output root `{output_root}` must be a directory.")
        with os.scandir(output_root) as entries:
            if next(entries, None) is not None:
                raise ValueError(f"Output root `{output_root}` must be empty.")
    else:
        os.makedirs(output_root)
    return output_root
