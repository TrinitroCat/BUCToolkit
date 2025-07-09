""" check whether imported module exists """
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _CheckModules.py
#  Environment: Python 3.12

import importlib
import warnings


# Check modules func
def check_module(module_name: str, pkg_name: None | str = None):
    """
    Check whether imported module exists.
    Args:
        module_name: imported module name.
        pkg_name: package name. It is required when performing a relative import.
                  It specifies the package to use as the anchor point from which to resolve the relative import to an absolute import.

    Returns:
        the imported module if it exists, else return None.

    """
    try:
        pkg = importlib.import_module(module_name, pkg_name)
        return pkg
    except ImportError:
        warnings.warn(f'Package {module_name} was not found, therefore some related methods would be unavailable.')
        return None
