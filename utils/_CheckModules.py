""" check whether imported module exists """
import importlib
import warnings


# Check modules func
def check_module(module_name: str, pkg_name: None | str = None):
    """
    Check whether imported module exists.
    Args:
        module_name: imported module name.
        pkg_name: package name. it is required when performing a relative import.
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
