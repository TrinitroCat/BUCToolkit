"""
An Advanced Interactive Command-Line Interface which can run end-to-end tasks of model training, structure optimization, molecular dynamics,
and Monte Carlo simulations.
"""

#  Copyright (c) 2026.3.26, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: __init__.py
#  Environment: Python 3.12

import os
import re
import time
import sys
import logging
from typing import Literal
import shlex

import BUCToolkit as bt
from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.CLI.print_logo import generate_display_art

has_prmt = (check_module('prompt_toolkit') is not None)

if has_prmt:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter


class BaseCLI:

    def __init__(self, *args, **kwargs):
        self._COMMANDS = {
            "/help": (self.print_help, 'Show this help message.'),
            "/exit": (self.do_exit, 'Exit the CLI program. One can also press "Ctrl+C" to exit."'),
            "/verbose": (
                self.set_verbose,
                "Reset the verbosity level. Available values: 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'"
            ),
            "/logto": (
                self.reset_handler,
                "Reset where the log information is printed. If `None` is given, logs will be printed into stdout."
            ),
            "train": (
                cli_train,
                ""
            ),
            "predict": (
                cli_predict,
                ""
            ),
            "opt": (
                cli_opt,
                ""
            ),
            "ts": (
                cli_ts,
                ""
            ),
            "neb": (
                cli_neb,
                ""
            ),
            "md": (
                cli_md,
                ""
            ),
            "mc": (
               cli_mc,
                ""
            )
        }

        self.closed = False

        self.logger = logging.getLogger('CLI')
        self.logger.setLevel(logging.INFO)
        self._log_formatter = logging.Formatter('%(message)s')
        if len(self.logger.handlers) <= 0:
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(self._log_formatter)
            self.logger.addHandler(log_handler)
        self._current_log_level = self.logger.getEffectiveLevel()

        pass

    @staticmethod
    def _input_parser(raw_input: str):
        """
        Parse the input to identify the commands and args
        Args:
            raw_input: raw input string

        Returns:
            Command key: str
            args: tuple
            kwargs: dict

        """
        inp_list = shlex.split(raw_input)
        command_key = inp_list[0]
        _kwargs = dict()
        _args = list()
        skip_times = 0  # times to skip
        for i, inp in enumerate(inp_list[1:]):
            if skip_times > 0:
                skip_times -= 1
                continue
            if inp.startswith('-') or inp.startswith('--'):
                _inp = inp.strip('-')
                if '=' in _inp:  # use '=' to align value
                    _ = _inp.split('=')
                    _kwargs[_[0]] = _[1] if _[1] != '' else None
                else:            # otherwise use space ' '
                    _prefetch = inp_list[i + 1]
                    if not _prefetch.startswith('-'):
                        _kwargs[_inp] = inp_list[i + 1]
                        skip_times += 1
                    else:
                        _kwargs[_inp] = None
                    continue
            else:
                _args.append(inp)

        return command_key, _args, _kwargs

    def print_help(self, *args, **kwargs):
        collect_help_info = list()
        for k, v in self._COMMANDS.items():
            collect_help_info.append(
                f"\"{k}\": {v[1]}\n"
            )
        self.logger.info(
            '\n' + ''.join(collect_help_info)
        )

    def _close(self):
        try:
            self._purge_handlers()
            return 0
        except Exception as e:
            print(f"An exception occurred when closing loggers: {e}. File may not be properly closed.")
            return 1

    def do_exit(self, *args, **kwargs):
        _exit_code = self._close()
        self.closed = True
        self.logger.info('BYE!')
        exit(_exit_code)

    def set_verbose(
            self,
            verbose: int|Literal['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'] = 'INFO',
            *args,
            **kwargs
    ):
        if not isinstance(verbose, int):
            verb = getattr(logging, verbose.upper(), None)
            if verb is None:
                raise ValueError(f"Invalid verbosity level '{verbose}'.")
        else:
            verb = verbose
        self._current_log_level = verb
        self.logger.setLevel(verb)

    def reset_handler(self, handler, *args, **kwargs):
        if (handler is None) or (handler == 'None'):
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(str(handler))
        handler.setFormatter(self._log_formatter)
        handler.setLevel(self._current_log_level)
        # purge old handlers
        self._purge_handlers()
        self.logger.addHandler(handler)

    def _purge_handlers(self):
        """ Remove all handlers """
        while self.logger.handlers:
            try:
                self.logger.handlers[0].close()
            except Exception as e:
                print(f"Failed to close a logger handler: {e}.")
            self.logger.removeHandler(self.logger.handlers[0])

    def run(self):
        try:
            # LOGO
            self.logger.info(generate_display_art())
            self.logger.info('Please type the content below. Type "/help" to show all commands.')
            while True:
                try:
                    if has_prmt:
                        content = prompt('\n>>> ', history=FileHistory('.cli_history'), auto_suggest=AutoSuggestFromHistory())
                    else:
                        content = input('\n>>> ')
                    if len(content) == 0: continue
                    command_key, _args, _kwargs = self._input_parser(content)
                    if command_key in self._COMMANDS:
                        self._COMMANDS[command_key][0](*_args, **_kwargs)
                        continue
                    else:
                        self.logger.error(f"Unknown command: '{command_key}'.")
                        continue

                except KeyboardInterrupt:
                    self.do_exit()

                except Exception as e:
                    self.logger.error(f"An exception occurred in CLI: {e}.")
                    continue
        finally:
            if not self.closed:
                self._close()


def cli_train():
    """

    Returns:

    """
    pass

def cli_predict():
    """

    Returns:

    """
    pass

def cli_opt():
    """
    Do optimization by BUCToolkit.TrainingMethod.StructureOptimization.relax
    Returns:

    """
    pass

def cli_ts():
    """
    Do optimization by BUCToolkit.TrainingMethod.StructureOptimization.ts
    Returns:

    """
    pass

def cli_neb():
    """
    Do optimization by BUCToolkit.TrainingMethod.NEB
    Returns:

    """
    pass

def cli_md():
    """

    Returns:

    """
    pass

def cli_mc():
    """

    Returns:

    """
    pass


