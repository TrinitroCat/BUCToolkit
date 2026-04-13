""" Setup logger for all modules """
#  Copyright (c) 2026.3.26, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: setup_loggers.py
#  Environment: Python 3.12

import logging
import sys
import warnings
import os
from typing import Any

from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper


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


class BaseIO:
    """
    Base class for logs of all classes.
    """

    def __init__(self, output_file: str|None = None) -> None:
        self.logger = None
        self.log_handler = None
        self.output_file = str(output_file) if output_file is not None else None
        self.dumper = structures_io_dumper(
            path=self.output_file,
            mode='x',
        )

    def init_logger(self, logger_name: str):
        # logging
        if self.logger is None:
            # cut off propagation
            supreme_name = logger_name.split('.')[0]
            top_logger = logging.getLogger(supreme_name)
            top_logger.propagate = False
            # set true logger
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            if not has_any_handler(self.logger):
                self.log_handler = logging.StreamHandler(sys.stdout)
                self.log_handler.setLevel(logging.INFO)
                self.log_handler.setFormatter(formatter)
                top_logger.addHandler(self.log_handler)
        else:
            warnings.warn('Logger has already initialized. Nothing will be done.', RuntimeWarning)

    def reset_logger_handler(self, handler: str|logging.StreamHandler|logging.FileHandler):
        """
        Clear all logging handlers including current logger and its ancestors, and reset one.
        Args:
            handler: the new handler.

        Returns:

        """
        clear_all_handlers(self.logger)
        # redirect to supreme logger
        top_logger = self.logger
        while top_logger.parent and top_logger.propagate:
            top_logger = top_logger.parent

        formatter = logging.Formatter('%(message)s')
        if isinstance(handler, logging.Handler):
            self.log_handler = handler
        elif isinstance(handler, str):
            output_path = os.path.dirname(handler)
            # check whether path exists
            if not os.path.isdir(output_path): os.makedirs(output_path)
            # set log handler
            self.log_handler = logging.FileHandler(handler, 'w', delay=True)
        else:
            raise TypeError("handler must be a string path or a logging.Handler instance")

        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(formatter)
        top_logger.addHandler(self.log_handler)

    def reset_dumper(self, dumper: Any) -> None:
        if self.output_file is not None:
            self.dumper.close()
            del self.dumper
            self.dumper = dumper
        else:
            self.logger.error(
                "ERROR: No output file specified. Hence, resetting dumper is meaningless.\n"
                "'reset_dumper': Operation REFUSED."
            )
