#  Copyright (c) 2026.5.19, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: batch_updaters.py
#  Environment: Python 3.12


import torch as th

class PygBatchUpdater:
    """
    batch updater for torch-geometric objects.
    It can be directly called after initialization.
    Examples:
        ```
        updater = PygBatchUpdater()
        updater.initialize()
        optimizer = CG(...)
        optimizer.set_batch_updater(updater)
        optimizer.run(...)
        ```
    One can use `self.initialize()` to reset this updater.
    """

    def __init__(self):
        from BUCToolkit.utils import check_module
        from BUCToolkit.BatchStructures import Batch

        self.__check_old = None
        _pyg = check_module('torch_geometric.data.batch')
        if _pyg is None:
            self.pygData = Batch
        else:
            self.pygData = _pyg.Batch

    def initialize(self):
        self.__check_old = None

    def _reallocate(
            self,
            converge_check: th.Tensor,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs
    ):
        # main
        g = func_args[0]
        g_list = g.index_select(converge_check)
        g_new = self.pygData.from_data_list(g_list)
        self.__check_old = converge_check
        self.__g_old = g_new
        return (g_new,), func_kwargs, (g_new,), grad_func_kwargs

    def __call__(
            self,
            converge_check: th.Tensor,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs
    ):
        # adding a buffer
        is_new = (self.__check_old is None) or (converge_check.shape != self.__check_old.shape)
        if is_new:
            if th.all(converge_check):  # if all are unconverged. usually occurred for new ones.
                self.__check_old = None
                return func_args, func_kwargs, grad_func_args, grad_func_kwargs
            else:
                self.__check_old = converge_check
                self.__g_old = func_args[0]
                return self._reallocate(converge_check, func_args, func_kwargs, grad_func_args, grad_func_kwargs)

        elif th.all(th.eq(self.__check_old, converge_check)):
            return (self.__g_old,), func_kwargs, (self.__g_old,), grad_func_kwargs
        else:
            return self._reallocate(converge_check, func_args, func_kwargs, grad_func_args, grad_func_kwargs)
