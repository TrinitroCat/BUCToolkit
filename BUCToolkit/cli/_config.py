"""Small shared helpers for reading CLI input files."""

import os
import sys
import time
import warnings
import copy
from collections.abc import Mapping
from typing import Any

import yaml


_PATH_FIELDS = (
    "MODEL_FILE",
    "MODEL_WRAPPER_FILE",
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


def backup_output(path: str, expected_kind: str) -> str | None:
    """Move an existing output to a timestamped backup path.

    Args:
        path: Output path that may already exist.
        expected_kind: ``"file"`` or ``"directory"``.

    Returns:
        The backup path, or ``None`` when ``path`` does not exist.

    Raises:
        ValueError: If ``path`` is a symbolic link or ``expected_kind`` is invalid.
        IsADirectoryError: If a file output names a directory.
        NotADirectoryError: If a directory output names a file.
    """
    if expected_kind not in {"file", "directory"}:
        raise ValueError(f"Unknown output kind `{expected_kind}`.")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.lexists(path):
        return None
    if os.path.islink(path):
        raise ValueError(f"Output `{path}` must not be a symbolic link.")
    if expected_kind == "file" and not os.path.isfile(path):
        raise IsADirectoryError(f"Output file `{path}` is an existing directory.")
    if expected_kind == "directory" and not os.path.isdir(path):
        raise NotADirectoryError(f"Output directory `{path}` is an existing file.")

    backup_base = f"{path}.bak{time.strftime('%Y%m%d_%H%M%S')}"
    backup_path = backup_base
    suffix = 1
    while os.path.lexists(backup_path):
        backup_path = f"{backup_base}_{suffix}"
        suffix += 1
    os.rename(path, backup_path)
    return backup_path


def raise_output_backup_warning(path: str, backup_path: str) -> str:
    """Announce one output backup on the standard warning channel.

    Args:
        path: Original output path.
        backup_path: Path receiving the previous output.

    Returns:
        The warning message emitted to the user.
    """
    message = f"WARNING: Output `{path}` already exists and was moved to `{backup_path}`. Be sure to make a backup!"
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return message


def load_input_config(input_path: str, require_output_root: bool = False) -> dict[str, Any]:
    """Load a CLI input file and resolve its known path fields.

    Relative paths owned by the CLI configuration are resolved from the input
    file's directory. Strings inside user-defined mappings such as
    ``MODEL_CONFIG`` and ``DATA_LOADER_KWARGS`` are left unchanged.

    Args:
        input_path: YAML input file to load.
        require_output_root: Retained for call compatibility. Missing roots
            use `./output`; legacy `OUTPUT_PATH` remains a fallback.

    Returns:
        A mapping containing validated configuration data, resolved known
        paths, and output-path fallbacks.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        TypeError: If a known typed field has an incompatible type.
        ValueError: If the YAML document is empty or its top level is not a
            mapping.
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

    from BUCToolkit.api._io import CONFIG_DEFAULTS
    for key, value in CONFIG_DEFAULTS.items():
        if key in {"OUTPUT_ROOT", "DATA_READER_KWARGS", "DATA_LOADER_KWARGS"}:
            continue
        config.setdefault(key, copy.deepcopy(value))

    for field in ("TASK", "DATA_TYPE", "MODEL_TYPE", "MODEL_WRAPPER_NAME"):
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
        output_root = _absolute_path("./output", input_directory)
        config["OUTPUT_ROOT"] = output_root

    config.setdefault("OUTPUT_PATH", os.path.join(output_root, "logs"))
    config.setdefault(
        "PREDICTIONS_SAVE_FILE",
        os.path.join(output_root, "results", "result"),
    )
    train_config = config.get("TRAIN")
    if isinstance(train_config, dict):
        train_config.setdefault("CHK_SAVE_PATH", os.path.join(output_root, "chk"))
    return config


def prepare_output_root(output_root: str, backup_existing: bool = True) -> str:
    """Prepare the root directory used by one CLI task.

    Args:
        output_root: Directory that will own logs, results, and checkpoints.
        backup_existing: Move a non-empty existing directory to a timestamped
            backup before recreating the output root.

    Returns:
        The absolute output-root path. A missing directory is created before
        it is returned; non-empty existing roots are backed up by default.

    Raises:
        ValueError: If the path is a symbolic link or a file.
        OSError: If the missing directory cannot be created or inspected.
    """
    output_root = os.path.abspath(os.path.expanduser(output_root))
    if os.path.lexists(output_root):
        if os.path.islink(output_root):
            raise ValueError(f"Output root `{output_root}` must not be a symbolic link.")
        if not os.path.isdir(output_root):
            raise ValueError(f"Output root `{output_root}` must be a directory.")
        with os.scandir(output_root) as entries:
            has_entries = next(entries, None) is not None
        if backup_existing and has_entries:
            backup_path = backup_output(output_root, "directory")
            if backup_path is not None:
                raise_output_backup_warning(output_root, backup_path)
            os.makedirs(output_root)
    else:
        os.makedirs(output_root)
    return output_root
