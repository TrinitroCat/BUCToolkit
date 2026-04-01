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
import copy

import yaml

from BUCToolkit.cli.main import launch_task
from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.cli.print_logo import generate_display_art
from BUCToolkit.cli.input_stub import CONFIG_STUB

has_prmt = (check_module('prompt_toolkit') is not None)

if has_prmt:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter  # TODO, auto-completion
    prompt_config = dict(history=FileHistory('.cli_history'), auto_suggest=AutoSuggestFromHistory())
else:
    prompt = input
    prompt_config = dict()

class BaseCLI:

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.INPUT_FILE = None  # current input information
        self._is_config = False
        self._TASK = {
            'TRAIN': 'TRAIN',
            'PREDICT': 'PREDICT',
            'OPT': 'OPT',
            'STRUCTURE_OPTIMIZATION': 'OPT',
            'STRUC_OPT': 'OPT',
            'DIMER': 'TS',
            'TS': 'TS',
            'VIB': 'VIB',
            'VIBRATIONAL_ANALYSIS': 'VIB',
            'NEB': 'NEB',
            'CINEB': 'NEB',
            'CI_NEB': 'NEB',
            'MD': 'MD',
            'MOLECULAR_DYNAMICS': 'MD',
            'CMD': 'CMD',
            'CONSTRAINED_MOLECULAR_DYNAMICS': 'CMD',
            'CONSTR_MD': 'CMD',
            'MC': 'MC',
            'MONTE_CARLO': 'MC',
        }
        self._COMMANDS = {
            "help": (self.print_help, 'Show this help message.'),
            "exit": (self.do_exit, 'Exit the cli program. One can also press "Ctrl+C" to exit.'),
            "verbose": (
                self.set_verbose,
                "Reset the verbosity level. Available values: 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'"
            ),
            "logto": (
                self.reset_handler,
                "Usage: logto `logger`  - Reset where the log information is printed. If `None` is given, logs will be printed into stdout."
            ),
            "show": (
                self.show_current_config,
                "Usage: `show [-v]`  - Show the current input file path. `-v` is to fully print the input file content."
            ),
            "edit": (
                self.edit_sub_cli,
                "Usage: `edit [path]`  - Edit current configuration file. if [path] is set, current file would try to change to the specified path."
            ),
            "task": (
                self.task_sub_cli,
                "usage: task `task_name` [current input file path].  "
                "- Create or edit the configurations of a task. "
                "If [current input file path] is not given, default of './task.inp' will be used.\n"
                f"{' '*12} Available task_name: {", ".join(self._TASK.keys())}"
            ),
            "run": (
                self.run_task,
                "Launch a task the cli.\n"
            )
        }
        # colours
        self._GREEN = '\033[32m'
        self._YELLOW = '\033[33m'
        self._RED = '\033[31m'
        self._RESET = '\033[0m'

        self.logger = logging.getLogger('cli')
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
                    if i + 1 >= len(inp_list):  # the last one
                        _kwargs[_inp] = None
                    else:
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
                f"{self._GREEN}{k: <10s}{self._RESET}: {v[1]}\n"
            )
        self.logger.info(
            '\n'.join(collect_help_info)
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
                self.logger.error(f"ERROR: Invalid verbosity level '{verbose}'.")
                return
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

    def args_exhibitor(self, arg_dict: dict, indent: int = 0):
        """
        Show formatted args dict to the CLI
        Args:
           arg_dict: dict load from YAML.
           indent: the indent for printing formatted args.

        Returns: None

        """
        if indent == 0:  # print a path info as title
            self.logger.info(f"Configurations saving path: {self.INPUT_FILE}\n")

        if indent >= 20:
            self.logger.critical(
                f"How frightful that you set an input containing over 20 times nesting. "
                f"I refuse such a horrible input."
            )
            raise ValueError(
                f"How frightful that you set an input containing over 20 times nesting. "
                f"I refuse such a horrible input."
            )
        for k, v in arg_dict.items():
            if (isinstance(v, dict)) and (len(v) > 1):
                self.logger.info(f"\n{'  '*indent}{self._RED}{k}{self._RESET}:")
                self.args_exhibitor(v, indent + 1)
                #if indent == 1: self.logger.info("\n")  # for 1 level printing, add a line break
            else:
                # print
                self.logger.info(f"{'  '*indent}{self._RED}{k}{self._RESET}: {v}")

    def show_current_config(self, v: str | None = '8Yc3TmYIgdGABhQbtLuL+3RdIjQHY3/eTELeYkkFMQ4='):
        """
        The default `is_v` is a placeholder that No one may precisely match this string. if is_v is set to None, print verbosely.
        """
        if v == '8Yc3TmYIgdGABhQbtLuL+3RdIjQHY3/eTELeYkkFMQ4=':
            self.logger.info(f"Current input file: {self.INPUT_FILE}")
            if self.INPUT_FILE is None: return
            if not os.path.isfile(self.INPUT_FILE):
                self.logger.warning(f"The current input file does not exist yet. YOU MAY CREATE IT BEFORE RUN CALCULATION.")
        elif v is None:
            if (self.INPUT_FILE is not None) and (os.path.isfile(self.INPUT_FILE)):
                with open(self.INPUT_FILE, 'r') as fv:
                    argdict = yaml.safe_load(fv)
                self.args_exhibitor(argdict)
            else:
                self.logger.warning(f"The input file does not exist. Showing details is impossible.")
        else:
            self.logger.error(f"Unknown value of v: {v}.")

    def rec_find_key(self, inp_dict, key, val, n_target: int = 1):
        """
        Recursively find the key in a nested dict and modify its value.
        Args:
            inp_dict: dict to modify.
            key: the key to find.
            val: the value to modify.
            n_target: the times of finding key.

        Returns: the times of finding key.

        """
        n_count = 0
        if n_target < 0: n_target = float('inf')
        for k, v in inp_dict.items():
            if n_count >= n_target:
                break
            if k == key:
                inp_dict[k] = val
                n_count += 1
            elif isinstance(v, dict):
                n_count += self.rec_find_key(v, key, val, n_target - n_count)  # limit to the rest available times
        return n_count

    def rec_modify_val(self, inp_dict: dict, nest_key_list: list, val, k_ptr: int = 0):
        """
        Recursively modify a nested dict by a given key list.
        The i-th elem. in the key list repr. the key of i-level nested dict.
        Args:
            inp_dict: dict to modify.
            nest_key_list: the key list of the nested dict.
            val: the value to modify.
            k_ptr: the current level of nesting, as well as the key index of `nest_key_list`.

        Returns:
            has_matched: bool, whether catches the value to modify

        """
        if len(nest_key_list) == 0: return True
        is_last = (k_ptr == (len(nest_key_list) - 1))
        has_matched = False
        if is_last:
            inp_dict[nest_key_list[k_ptr]] = val  #  compatible with both addition and modification cases.
            return True
        for k in list(inp_dict.keys()):
            v = inp_dict[k]
            if k == nest_key_list[k_ptr]:  # match the i-level nested key
                if isinstance(v, dict):
                    has_matched = self.rec_modify_val(v, nest_key_list, val, k_ptr + 1)
                    if not has_matched:  # failed at deeper level, thus success is impossible
                        return False
                    else: break
                else:  # matched the key but not a dict value. search aborted
                    return False
        # if not matched the key but not at last, adding empty key-dict pairs
        if not has_matched:
            sub_dict_to_update = dict()
            u = {nest_key_list[k_ptr]: dict()}
            v = sub_dict_to_update
            ik = k_ptr
            while ik < (len(nest_key_list) - 1):
                v.update(u)
                v = u[nest_key_list[ik]]
                ik += 1
                u = {nest_key_list[ik]: dict()}
            v[nest_key_list[ik]] = val
            inp_dict.update(sub_dict_to_update)

        return True

    def rec_rm_key(self, inp_dict: dict, nest_key_list: list, k_ptr: int = 0):
        """
        Recursively remove the key given by the key list in a nested dict.
        The i-th elem. in the key list repr. the key of i-level nested dict.
        Args:
            inp_dict: dict to modify.
            nest_key_list: the key list of the nested dict.
            k_ptr: the current level of nesting, as well as the key index of `nest_key_list`.

        Returns:
            has_matched: bool, whether catches the key to remove

        """
        if len(nest_key_list) == 0: return True
        is_last = (k_ptr == (len(nest_key_list) - 1))
        has_matched = False
        if is_last:
            if nest_key_list[k_ptr] in inp_dict:
                inp_dict.pop(nest_key_list[k_ptr])  #  compatible with both addition and modification cases.
            # even if not match, goal of deletion can be viewed as successful, so return True
            return True
        for k in list(inp_dict.keys()):
            v = inp_dict[k]
            if k == nest_key_list[k_ptr]:  # match the i-level nested key
                if isinstance(v, dict):
                    has_matched = self.rec_rm_key(v, nest_key_list, k_ptr + 1)
                    if not has_matched:  # failed at deeper level, thus success is impossible
                        return False
                    else:
                        break
                else:  # matched the key but not a dict value. search aborted
                    return False

        return has_matched  # not matched

    def rec_check_key(self, inp_dict: dict, nest_key_list: list, k_ptr: int = 0):
        """
        Recursively search and show the value of the key given by the key list in a nested dict.
        The i-th elem. in the key list repr. the key of i-level nested dict.
        Args:
            inp_dict: dict to show.
            nest_key_list: the key list of the nested dict.
            k_ptr: the current level of nesting, as well as the key index of `nest_key_list`.

        Returns:
            content: the value of given keys. If failed, content will be `None`.
            has_matched: bool, whether catches the key to show

        """
        if len(nest_key_list) == 0: return None, True
        is_last = (k_ptr == (len(nest_key_list) - 1))
        has_matched = False
        if is_last:
            if nest_key_list[k_ptr] in inp_dict:
                cont = inp_dict[nest_key_list[k_ptr]]
                return cont, True
            else:
                cont = None
                return cont, False
        for k in list(inp_dict.keys()):
            v = inp_dict[k]
            if k == nest_key_list[k_ptr]:  # match the i-level nested key
                if isinstance(v, dict):
                    cont, has_matched = self.rec_check_key(v, nest_key_list, k_ptr + 1)
                    return cont, has_matched
                else:  # matched the key but not a dict value. search aborted
                    return None, False

        return None, has_matched  # not matched

    def task_sub_cli(self, task: str|None = None, inp_file: str|None = None):
        """

        Returns:

        """

        if task is None:
            task = prompt('>>> Enter a task (TRAIN, PREDICT, OPT, TS, VIB, NEB, MD, CMD, MC): ')
        task = task.upper()
        if task not in self._TASK:
            self.logger.error(f"Unknown task: {task}\nAvailable task_name: {", ".join(self._TASK.keys())}")
            return
        else:
            ARGS_WORK = eval(f"ARGS_{self._TASK[task]}")
        if inp_file is None:  # if not input file, use default configs
            AGRS_NOW = '\n'.join([ARGS_GLOBAL, ARGS_IO, ARGS_MODEL, ARGS_WORK])
            inp_args = yaml.safe_load(AGRS_NOW)
        else:
            with open(inp_file, 'r') as fx:
                inp_args = yaml.safe_load(fx)
            self.INPUT_FILE = inp_file
        inp_args['TASK'] = self._TASK[task]
        # Show once
        self.logger.info(f'Current configuration:\n')
        self.args_exhibitor(inp_args)
        self.logger.info("*" * 89)
        # To edit
        self.edit_sub_cli(inp_args)

    def edit_sub_cli(self, inp_args: dict|str|None = None):
        """

        Returns:

        """
        if inp_args is None: # if not given, try to read from self.INPUT_FILE
            if self.INPUT_FILE is None:
                self.logger.warning(f"Nothing to edit. EDIT ABORTED.")
                return
            else:
                self.logger.info(f"Try to load configs from {self.INPUT_FILE} ...")
                try:
                    with open(self.INPUT_FILE, 'r') as fx:
                        inp_args = yaml.safe_load(fx)
                except Exception as e:
                    self.logger.error(f"ERROR: Failed to load configs from {self.INPUT_FILE} due to \"{e}\"")
                    return
        elif isinstance(inp_args, str):  # try to read file
            self.logger.info(f"Try to load configs from {inp_args} ...")
            _inp_args_path = inp_args
            try:
                with open(_inp_args_path, 'r') as fx:
                    inp_args = yaml.safe_load(fx)
                self.INPUT_FILE = _inp_args_path
            except Exception as e:
                self.logger.error(f"ERROR: Failed to load configs from {_inp_args_path} due to \"{e}\"")
                return

        help_info = """
        Commands:
            help: show this help message.
            exit: save & exit edit. 
            load: `load [path]`, load a configuration file from given path. If path is given, current path will be changed synchronously.
            save: `save [path]`, save current configuration. If path is given, current path will be changed synchronously.
            chpt: `chpt [path]`, change the file saving path.
            quit: exit edit without saving
            show: show all current configurations
            list: alias of 'show'
        To show the information of arguments, just directly input the keyword:
            `[Section(s)].KEYWORDS`
            example:
                `MD.THERMOSTAT_CONFIG.TIME_CONST`, which will show the required data type and corresponding docstring of 
                the time constant (of CSVR thermostat) under MD.THERMOSTAT_CONFIG sections, 
                where `THERMOSTAT_CONFIG` is a sub-section of the section `MD`.
        Two ways to change configurations:
            1. `[Section(s)].KEYWORDS = new_value`
            2. `[Section(s)].KEYWORDS: new_value`
            example: 
                `MD.THERMOSTAT_CONFIG.TIME_CONST = 100.`, which modifies the time constant above.
        To delete keywords in current configuration:
            `del [Section(s)].KEYWORDS`
            example:
                `del OPT.USE_BB`, which deletes the `USE_BB` (whether to use Barzilai-Borwein step) argument under section `OPT`.
        One can also use 'Ctrl+C' to quit.
        """
        if 'TASK' not in inp_args:
            self.logger.error(
                f"ERROR: Argument 'TASK' is absent in the given configurations. "
                f"YOU SHOULD SPECIFY ONE."
            )
            _xtask = prompt(f'\n>>> Type the TASK: ')
            if _xtask not in self._TASK:
                self.logger.error(f"Unknown task: {_xtask}\nAvailable task_name: {", ".join(self._TASK.keys())}")
                self.logger.error(f"EDIT ABORTED.")
                return
            else:
                inp_args['TASK'] = self._TASK[_xtask]
        inp_args_bak = copy.deepcopy(inp_args)
        task = inp_args['TASK']
        # main cli loop
        while True:
            try:
                content = prompt(f'\n>>> {task}: ', **prompt_config)
                if len(content) == 0:
                    self.logger.info(f'{task} configuration done.')
                    self.dump_inpfile(inp_args)
                    break
                elif content.lower() == 'help':
                    self.logger.info(f"{help_info}")
                    continue
                elif content.lower() == 'exit':
                    self.logger.info(f'{task} configuration done.')
                    self.dump_inpfile(inp_args)
                    break
                elif content.lower() == 'quit':
                    self.logger.info('All changes have been cancelled.')
                    while True:
                        try:
                            is_save = prompt('>>> Do you want to save current configurations? (y/N): ')
                        except KeyboardInterrupt:
                            is_save = 'n'
                        if len(is_save) == 0:  # viewed as 'n'
                            break
                        elif is_save.lower() == 'y':
                            self.dump_inpfile(inp_args_bak)
                            break
                        elif is_save.lower() == 'n':
                            break
                        else:
                            self.logger.error(f"Please input 'y' or 'n'.")
                    break
                elif content.lower()[:4] == 'save':
                    true_cont = content.lower().split()
                    if len(true_cont) == 2:
                        self.INPUT_FILE = true_cont[1]
                    self.dump_inpfile(inp_args, force=True)
                    continue
                elif content.lower()[:4] == 'load':
                    true_cont = content.split()
                    if len(true_cont) == 2:
                        self.INPUT_FILE = true_cont[1]
                    # load
                    try:
                        with open(self.INPUT_FILE, 'r') as fs:
                            _inp_args = yaml.safe_load(fs)
                        if 'TASK' not in _inp_args:
                            self.logger.error(f"Failed to load: Argument 'TASK' is absent in the given file.")
                            continue
                        elif _inp_args['TASK'] != self._TASK[task]:
                            self.logger.error(f"The task of loaded file does not match current task.")
                            continue
                        inp_args = _inp_args
                        inp_args_bak = copy.deepcopy(inp_args)
                    except FileNotFoundError:
                        self.logger.error(f"File not found: {self.INPUT_FILE}.")
                    continue
                elif (content.lower() == 'show') or (content.lower() == 'list'):
                    self.args_exhibitor(inp_args)
                    continue
                elif content.lower()[:4] == 'chpt':
                    _ = content.split(maxsplit=1)
                    if len(_) == 1:
                        _path = prompt(f">>> I/O: Please enter a path to save current configuration file: ")
                        if len(_path) == 0:
                            self.logger.info(f"Path change cancelled.")
                            continue
                        else:
                            _path = _path.strip()
                    else:
                        _path = str(_[1]).strip()
                    self.INPUT_FILE = _path
                    continue
                elif content.lower()[:3] == 'del':
                    _ = content.split(maxsplit=1)
                    if len(_) != 2:
                        self.logger.error(f"Invalid argument: {content}. Usage: del `KEYWORDS`")
                        continue
                    key = _[1].strip().upper()
                    key_list = key.split('.')
                    is_succ = self.rec_rm_key(inp_args, key_list)
                    if not is_succ:
                        self.logger.error(f"No keyword {key} matched in current configuration.")
                    continue
                else:  # show & change keywords
                    key = None
                    val = None
                    try:
                        # if not alignment symbol, SHOW info.
                        if ('=' not in content) and (':' not in content):
                            key = content.strip().upper()
                            # handle keychains
                            key_list = key.split('.')
                            chk_res, is_succ = self.rec_check_key(CONFIG_STUB, key_list)
                            if not is_succ:
                                self.logger.error(f"Unknown keyword {key}.")
                            else:
                                # print info
                                # normal result: (default value, data type, docstring)
                                self.logger.info(
                                    f"{'.'.join(key_list)}: {chk_res[1]}, {chk_res[2]}. Default: {chk_res[0]}"
                                )
                            continue
                        # else CHANGE value
                        _valid_flag = False
                        for try_delimiter in ['=', ':']:
                            cont = content.split(try_delimiter, maxsplit=1)
                            if len(cont) != 2:
                                continue
                            else:
                                key = cont[0].strip().upper()
                                # handle keychains
                                key_list = key.split('.')
                                # handle values
                                val = cont[1].strip()
                                if val.lower() == 'true':
                                    val = True
                                elif val.lower() == 'false':
                                    val = False
                                elif val.lower() == 'none':
                                    val = None
                                else:
                                    try:  # try to convert to python object
                                        val = eval(val)
                                    except (SyntaxError, NameError, TypeError, ValueError):
                                        val = str(val)
                                is_succ = self.rec_modify_val(inp_args, key_list, val)
                                # double check
                                chk_res, is_succ_ = self.rec_check_key(CONFIG_STUB, key_list)
                                _valid_flag = True
                                if is_succ:
                                    if not is_succ_:
                                        self.logger.warning(
                                            f"WARNING: {key} is not a valid keyword in all possible configurations. "
                                            f"While it still added/modified to current configuration anyway."
                                        )
                                    # check type:
                                    elif not isinstance(val, chk_res[1]):
                                        self.logger.warning(
                                            f"WARNING: The type of keyword's value should be {chk_res[1]}, "
                                            f"but now the type ({type(val)}) of value ({val}) is given. "
                                            f"PLEASE CAREFULLY CHECK!!!"
                                        )
                                    self.logger.info(
                                        f'The value of `{key}` has been successfully changed to `{val}` that belongs to {type(val)}.'
                                    )
                                else:
                                    self.logger.info(f"Failed to change {key} in current configuration. Try again.")
                                break
                        if not _valid_flag:
                            self.logger.error(f'ERROR: Unknown command. help:\n{help_info}')
                        continue
                    except Exception as e:
                        self.logger.error(f'ERROR: Failed to change the key-val pair `{key}:{val}` due to {e}. Try again.')
                        continue

                #self.args_exhibitor(inp_args)
            except KeyboardInterrupt:
                self.logger.info('All changes have been cancelled.')
                while True:
                    try:  # handle Ctrl+C during input cli
                        is_save = prompt('>>> Do you want to save current configurations? (y/N): ')
                    except KeyboardInterrupt:
                        is_save = 'n'
                    if len(is_save) == 0:
                        break
                    elif is_save.lower() == 'y':
                        self.dump_inpfile(inp_args_bak)
                        break
                    elif is_save.lower() == 'n':
                        break
                    else:
                        self.logger.error(f"Please input 'y' or 'n'.")
                break
            except Exception as e:
                self.logger.error(f'ERROR: {e}.\nAll changes have been cancelled without saving.')
                #self.dump_inpfile(inp_args_bak, force=True)
                break

    def dump_inpfile(self, inp, disable = False, force=False):
        """
        Dumping input file
        Args:
            inp: input content
            disable: whether to really dump. A placeholder.
            force: whether to force dump even overwritten.
        Returns:

        """
        if disable:
            return
        if self.INPUT_FILE is None:
            self.INPUT_FILE = prompt(f">>> I/O: Please enter a path to save current configuration file: ")
        if (not force) and os.path.exists(self.INPUT_FILE):
            while True:
                content = prompt(f'\n>>> I/O: Found an existing configuration file. Do you want to overwrite it? (y/N): ')
                if len(content) == 0:
                    content = 'n'
                if content.lower() == 'y':
                    with open(self.INPUT_FILE, 'w') as f:
                        yaml.dump(inp, f)
                    break
                elif content.lower() == 'n':
                    break
                else:
                    self.logger.info('Please enter y or n. Default: n.')
        else:
            with open(self.INPUT_FILE, 'w') as f:
                yaml.dump(inp, f)

    def run_task(self, ):
        """ run """
        if self.INPUT_FILE is None:
            self.logger.error(f'ERROR: No input file was provided, please input `task [name]` to configure it first.')
            return
        elif os.path.exists(self.INPUT_FILE):
            launch_task(self.INPUT_FILE)
        else:
            self.logger.error(f'ERROR: Current input file `{self.INPUT_FILE}` does not exist. Please check or try another one.')
            return

    def run(self):
        try:
            # LOGO
            self.logger.info(generate_display_art())
            self.logger.info('Please type the content below. Type "help" to show all commands.')
            while True:
                try:
                    if has_prmt:
                        content = prompt('\n>>> ', **prompt_config)
                    else:
                        content = input('\n>>> ')
                    if len(content) == 0: continue
                    command_key, _args, _kwargs = self._input_parser(content)
                    if command_key in self._COMMANDS:
                        self._COMMANDS[command_key][0](*_args, **_kwargs)
                        continue
                    else:
                        self.logger.error(f"Unknown command: '{command_key}'. Please input `help` to show all commands.")
                        continue

                except KeyboardInterrupt:
                    self.do_exit()

                except Exception as e:
                    self.logger.error(f"An exception occurred in cli: {e}.")
                    continue
        finally:
            if not self.closed:
                self._close()


ARGS_GLOBAL = """
# global configs
TASK: !!str MD
START: !!int 1          # 0: from scratch; 1: load checkpoint from LOAD_CHK_FILE_PATH
VERBOSE: !!int 1
DEVICE: !!str 'cuda:0'
BATCH_SIZE: !!int 16
"""

ARGS_IO = """
LOAD_CHK_FILE_PATH: !!str your/model/checkpoint/file/path
OUTPUT_PATH: !!str your/log/output/path
OUTPUT_POSTFIX: !!str your_logfile_suffix
PREDICTIONS_SAVE_FILE: !!str your/model/predictions/save/path  # path of saving predictions
STRICT_LOAD: !!bool true  # whether to strictly load model parameter
REDIRECT: !!bool true    # whether output training logs to `OUTPUT_PATH` or directly print on screen.
SAVE_PREDICTIONS: !!bool true  # only for predictions. Whether output predictions to a dump file.
DATA_TYPE: !!str BS  # Literal['POSCAR', 'OUTCAR', 'CIF', 'ASE_TRAJ', 'BS', 'OPT', 'MD']
DATA_PATH: !!str /your/data/path # the path of data used for calculation. if training, it will be viewed as the training set.
DATA_NAME_SELECTOR: !!str ".*$"  # regular express to select data names. Only matched name will be finally load.
FSDATA_PATH: !!str your/final/state/data/path  # used for calc. requiring both initial and final states, e.g., CI-NEB
DISPDATA_PATH: !!str your/displacement/data/path  # used for calc. requiring initial guess of a direction, e.g., Dimer
VAL_SET_PATH: !!str your/validation/set/path  # used for training that requires validation data
VAL_SPLIT_RATIO: !!float 0.1  # the ratio of validation set in the total dataset. if `VAL_SET` is given, this arg will be ignored.
DATA_LOADER_KWARGS: {}      # other kwargs for data loader.
IS_SHUFFLE: !!bool false    # whether to randomly shuffle dataset before calculating.
"""

ARGS_MODEL = """
# model configs
MODEL_FILE: !!str your/model/file/path/template_model.py  # function file path of torch model
MODEL_NAME: !!str YourModel   # the specific name of the model in `MODEL_FILE`
MODEL_CONFIG:   # model hyperparameters used for `MODEL_NAME.__init__(**MODEL_CONFIG)`
  hyperparameter1: 'xxx'
  hyperparameter2: 'xxx'
"""

ARGS_TRAIN = """
# training
TRAIN:
  # epoches & val set
  EPOCH: !!int 10
  VAL_BATCH_SIZE: !!int 20  # batch size for validation. default is the same as BATCH_SIZE
  VAL_PER_STEP: !!int 100   # validate every `VAL_PER_STEP` steps. step = `BATCH_SIZE` * `ACCUMULATE_STEP`
  VAL_IF_TRN_LOSS_BELOW: !!float 1.e5  # only validating after training loss < `VAL_IF_TRN_LOSS_BELOW`
  ACCUMULATE_STEP: !!int 12  # gradient accumulation steps
  # loss configs
  LOSS: !!str Energy_Loss  # 'MSE': nn.MSELoss, 'MAE': nn.L1Loss, 'Hubber': nn.HuberLoss, 'CrossEntropy': nn.CrossEntropyLoss 'Energy_Force_Loss': Energy_Force_Loss, 'Energy_Loss': Energy_Loss
  LOSS_CONFIG:             # other kwargs for loss function
    loss_E: !!str SmoothMAE

  METRICS:  # tuple of ones in [E_MAE, F_MAE, F_MaxE, E_R2, MSE, MAE, R2, RMSE], F_MaxE is the max absolute error of forces.
    - !!str E_MAE
    - !!str E_R2
  METRICS_CONFIG: {}  # other kwargs for metrics
  #  - F_MaxE

  # optimizer configs
  OPTIM: !!str AdamW  # model optimizer. Available values:
                      # 'Adam': th.optim.Adam, 'SGD': th.optim.SGD, 'AdamW': th.optim.AdamW, 'Adadelta': th.optim.Adadelta,
                      # 'Adagrad': th.optim.Adagrad, 'ASGD': th.optim.ASGD, 'Adamax': th.optim.Adamax, 'FIRE': FIRELikeOptimizer,
  OPTIM_CONFIG:       # optimizer kwargs
    lr: !!float 2.e-4
    # ...
  LAYERWISE_OPTIM_CONFIG: # Supporting regular expression to selection layers and set them.
    'force_block.*': { 'lr': 5.e-4 }
    'energy_block.*': { 'lr': 2.e-4 }
    '.*_bias_layer.*': { 'lr': 2.e-4 }

  GRAD_CLIP: !!bool true  # whether to toggle on gradient clip
  GRAD_CLIP_MAX_NORM: !!float 10.  # maximum grad. norm to clip
  GRAD_CLIP_CONFIG: {}    # other kwargs for `nn.utils.clip_grad_norm_` function
  LR_SCHEDULER: !!str None  # learning rate scheduler. Available values:
                            # 'StepLR': StepLR, 'ExponentialLR': ExponentialLR, 'ChainedScheduler': ChainedScheduler,
                            # 'ConstantLR': ConstantLR, 'LambdaLR': LambdaLR, 'LinearLR': LinearLR,
                            # 'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts, 'CyclicLR': CyclicLR,
                            # 'MultiStepLR': MultiStepLR, 'CosineAnnealingLR': CosineAnnealingLR, 'None': None,
  LR_SCHEDULER_CONFIG: {} # kwargs of above `LR_SCHEDULER`
  EMA: !!bool false       # whether to toggle on EMA (Exponential Moving Average)
  EMA_DECAY: !!float 0.999  # EMA decay rate
"""

ARGS_PREDICT = """"""

ARGS_OPT = """
# relaxation
RELAXATION:
  ALGO: !!str 'FIRE'  # CG, BFGS, FIRE
  ITER_SCHEME: !!str 'PR+'  # only for ALGO=CG, 'PR+', 'FR', 'PR', 'WYL'
  E_THRES: !!float 1.e4  # threshold of Energy difference
  F_THRES: !!float 0.05  # threshold of max Force
  MAXITER: !!int 300
  STEPLENGTH: !!float 0.5
  USE_BB: !!bool true
  LINESEARCH: !!str 'B'  # 'Backtrack' with Armijo's cond., 'Golden' for exact line search by golden section algo., 'Wolfe' for advance & retreat algo. with weak Wolfe cond.
  LINESEARCH_MAXITER: !!int 8  # max iterations of linear search.
  LINESEARCH_THRES: !!float 0.02  # only for LINESEARCH = 'Golden', threshold of exact line search.
  LINESEARCH_FACTOR: !!float 0.5  # A factor in linesearch. Shrinkage factor for "Backtrack", scaling factor in interval search for "Golden"
  REQUIRE_GRAD: !!bool False
"""

ARGS_TS = """
# transition state
TRANSITION_STATE:
  ALGO: !!str DIMER
  #X_DIFF: None
  E_THRES: !!float 1.e-4
  TORQ_THRES: !!float 1.e-2
  F_THRES: !!float 5.e-2
  MAXITER_TRANS: !!int 300
  MAXITER_ROT: !!int 5
  MAX_STEPLENGTH: !!float 0.5
  DX: !!float 1.e-1
  REQUIRE_GRAD: !!bool False
"""

ARGS_VIB = """
# vibration analyses (harmonic)
VIBRATION:
  METHOD: !!str 'Coord'  # Coord/Grad corresponding to finite difference and auto-grad scheme.
  BLOCK_SIZE: !!int 1
  DELTA: !!float 1e-2
"""

ARGS_NEB = """
# NEB transition state
NEB:
  ALGO: !!str 'CI-NEB'
  N_IMAGES: !!int 7
  SPRING_CONST: 5.0
  OPTIMIZER: !!str FIRE
  #OPTIMIZER_CONFIGS: Optional[Dict[str, Any]] = None, other kwargs of optimizer.
  STEPLENGTH: !!float 0.2
  E_THRESHOLD: !!float 1.e-3
  F_THRESHOLD: !!float 0.05
  MAXITER: !!int 20
  REQUIRE_GRAD: !!bool False
"""

ARGS_MD = """
# molecular dynamics
MD:
  ENSEMBLE: !!str NVT
  THERMOSTAT: !!str CSVR  # only for ENSEMBLE=NVT, 'Langevin', 'VR', 'Nose-Hoover', 'CSVR'
  THERMOSTAT_CONFIG:
    DAMPING_COEFF: !!float 0.01
    TIME_CONST: !!float 120
  TIME_STEP: !!float 1  # Unit: fs
  MAX_STEP: !!int 100  # total time (fs) = TIME_STEP * MAX_STEP
  T_INIT: !!float 298.15  # Initial Temperature, Unit: K. For ENSEMBLE=NVE, T_INIT is only used to generate ramdom initial velocities by Boltzmann dist.
  OUTPUT_COORDS_PER_STEP: !!int 1  # To control the frequency of outputting atom coordinates. If verbose = 3, atom velocities would also be outputted.
  REQUIRE_GRAD: !!bool False
"""

ARGS_CMD = """
MD:
  ENSEMBLE: !!str NVT
  THERMOSTAT: !!str CSVR  # only for ENSEMBLE=NVT, 'Langevin', 'VR', 'Nose-Hoover', 'CSVR'
  THERMOSTAT_CONFIG:
    DAMPING_COEFF: !!float 0.01
    TIME_CONST: !!float 120
  TIME_STEP: !!float 1  # Unit: fs
  MAX_STEP: !!int 100  # total time (fs) = TIME_STEP * MAX_STEP
  T_INIT: !!float 298.15  # Initial Temperature, Unit: K. For ENSEMBLE=NVE, T_INIT is only used to generate ramdom initial velocities by Boltzmann dist.
  OUTPUT_COORDS_PER_STEP: !!int 1  # To control the frequency of outputting atom coordinates. If verbose = 3, atom velocities would also be outputted.
  REQUIRE_GRAD: !!bool False
  # Optional: constraints
  CONSTRAINTS_FILE: !!str ./constraints.py
  CONSTRAINTS_FUNC: !!str func
"""

if __name__ == '__main__':
    f = BaseCLI()
    f.run()
