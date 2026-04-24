""" Setup logger for all modules """
#  Copyright (c) 2026.3.26, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: setup_loggers.py
#  Environment: Python 3.12

import logging
import warnings


def clear_all_handlers(logger: logging.Logger):
    """
    Clear all handlers of this current logger and its all ancestors until one does not propagate.

    Args:
        logger: the input logger instance. If it is not a logging.Logger, nothing is done.
    Returns:
        the logger instance
    """
    current = logger
    while current:
        for _hdl in list(current.handlers):
            current.removeHandler(_hdl)
            try:
                _hdl.close()
            except Exception as ehdl:
                warnings.warn(f'Failed to close handler {_hdl}: {ehdl}', RuntimeWarning)
        # if propagate=False，stop checking
        if not current.propagate:
            break
        current = current.parent

def has_any_handler(logger: logging.Logger) -> bool:
    """
    Check whether the current logger and its all ancestor loggers have any handlers.
    Args:
        logger: the input logger instance. If it is not a logging.Logger, False is returned.

    Returns: bool, the logger has any handlers.

    """
    if not isinstance(logger, logging.Logger):
        return False
    has_handler = False
    current = logger
    while current:
        if current.handlers:
            has_handler = True
            break
        # if propagate=False，stop checking
        if not current.propagate:
            break
        current = current.parent

    return has_handler
