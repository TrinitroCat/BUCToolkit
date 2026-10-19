"""Private glue used by the ``btvasp`` command-line frontend."""

from __future__ import annotations

import argparse
import os
from BUCToolkit.utils.model_wrappers import VASP_PluginModel


def run_btvasp(
        input_file: str,
        vasp_directory: str,
) -> None:
    """Run an existing BUCToolkit task with a local VASP plugin calculator.

    Args:
        input_file: Standard BUCToolkit YAML task input.
        vasp_directory: VASP cwd containing INCAR, POSCAR, KPOINTS and POTCAR.

    Returns:
        None. The configured BUCToolkit engine writes its normal outputs.
    """
    from BUCToolkit.cli.main import parse_center_input_file
    from BUCToolkit.cli._config import load_input_config

    config = load_input_config(input_file)
    if str(config.get("MODEL_TYPE", "pyg")).lower() != "vasp":
        raise ValueError("btvasp requires `MODEL_TYPE: vasp` in the BUCToolkit input file.")
    if int(config.get("BATCH_SIZE", 1)) != 1:
        raise ValueError("btvasp requires BATCH_SIZE = 1 for one persistent VASP session")
    model_wrapper_config = dict(config.get("MODEL_WRAPPER_CONFIG") or {})
    model_wrapper_config["input_path"] = vasp_directory
    task_type, runner, model = parse_center_input_file(
        input_file,
        model_override=VASP_PluginModel,
        model_wrapper_config_override=model_wrapper_config,
    )
    if task_type == "TRAIN":
        runner.train(model)
    elif task_type == "TS":
        runner.ts(model)
    elif task_type == "OPT":
        runner.relax(model)
    else:
        runner.run(model)


def main_btvasp(argv: list[str] | None = None) -> None:
    """Parse ``btvasp`` arguments and dispatch one BUCToolkit task."""
    parser = argparse.ArgumentParser(description="Run BUCToolkit with a VASP Python-plugin calculator.")
    parser.add_argument("-i", "--input", required=True, help="BUCToolkit YAML input file")
    parser.add_argument("-d", "--vasp-dir", default=".", help="VASP working directory (default: .)")
    args = parser.parse_args(argv)
    run_btvasp(
        os.path.abspath(args.input),
        os.path.abspath(args.vasp_dir),
    )
