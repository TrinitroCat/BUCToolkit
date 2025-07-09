#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: CI_NEB.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time
import warnings
import logging
import sys

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as Fn

from BM4Ckit.utils._print_formatter import GLOBAL_SCIENTIFIC_ARRAY_FORMAT
from ..minimize import CG, QN, FIRE

np.set_printoptions(**GLOBAL_SCIENTIFIC_ARRAY_FORMAT)


class CI_NEB:
    """
    Ref. TODO

    """

    def __init__(
            self,
            N_images: int,
            spring_const: float = 0.1,
            optimizer: Literal['CG', 'QN', 'FIRE'] = 'CG',
            iter_scheme: Literal['BFGS', 'Newton', 'PR', 'PR+', 'SD', 'FR'] = 'PR+',
            linesearch: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'] = 'Backtrack',
            steplength: float = 0.05,
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):

        warnings.filterwarnings('always')
        optimizer_dict = {'CG': CG, 'QN': QN, 'FIRE': FIRE}
        self.Optimizer = optimizer_dict[optimizer](
            iter_scheme = iter_scheme,
            E_threshold = E_threshold,
            F_threshold = F_threshold,
            maxiter = maxiter,
            linesearch = 'None',
            linesearch_maxiter = 10,
            linesearch_thres = 0.02,
            linesearch_factor = 0.6,
            steplength = steplength,
            device = device,
            verbose = verbose
        )
        self.N_images = N_images
        self.spring_const = spring_const
        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter
        self.device = device
        self.verbose = verbose
        # check
        if (not isinstance(self.maxiter, int)) or (self.maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.maxiter}. It would be an integer greater than 0.')

        # logger
        self.logger = logging.getLogger('Main.OPT')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not self.logger.hasHandlers():
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)
        pass

    class __NebWrapper:
        """ function wrapper for NEB, Adding the elastic potential """
        def __init__(self, f: Callable, gf: Callable, k: float, is_grad_contain_y: bool):
            self.ener = None
            self.f = f
            self.gf = gf
            self.k = k
            self.is_grad_contain_y = is_grad_contain_y

        def energy(self, X, *args, **kwargs):
            with th.enable_grad():
                y = self.f(X, *args, **kwargs)
                self.ener = y
            return y

        def grad(self, X, *args, **kwargs):
            assert self.ener is not None, 'Energy is None now, please calculate energy first.'
            with th.enable_grad():
                X.requires_grad_()
                if self.is_grad_contain_y:  # here `X` is actually `y`
                    F_ori = self.gf(X, args[0], *(args[1:]), **kwargs)  # (n_i, n_atom, 3)
                    X = args[0]
                else:
                    F_ori = self.gf(X, *args, **kwargs)  # (n_i, n_atom, 3)
                n_i, n_atom, n_dim = X.shape

            with th.no_grad():
                # projected direction
                _tau = th.diff(X, dim=0)  # (n_i - 1, n_atom, 3)
                ediff = th.diff(self.ener)  # (n_i, )
                ediff_back = ediff[:-1].unsqueeze(-1).unsqueeze(-1)  # (n_i - 2, ), E_(n) - E_(n-1)
                tau_back = _tau[:-1]
                ediff_front  = ediff[1:].unsqueeze(-1).unsqueeze(-1)   # (n_i - 2, ), E_(n+1) - E_(n)
                tau_front = _tau[1:]
                #tau_shift = th.cat((th.zeros((1, n_atom, n_dim)), tau), dim=0)  # (n_i, n_atom, 3)
                delta_Emax = th.where(th.abs(ediff_front) - th.abs(ediff_back) > 0, th.abs(ediff_front), th.abs(ediff_back))
                delta_Emin = th.where(th.abs(ediff_front) - th.abs(ediff_back) < 0, th.abs(ediff_front), th.abs(ediff_back))
                tau_extreme = th.where(
                    ediff_front + ediff_back > 0,  # (E_(n+1) > E_(n-1))
                    tau_front * delta_Emax + tau_back * delta_Emin,
                    tau_front * delta_Emin + tau_back * delta_Emax
                )
                tau = th.where(
                    (ediff_front > 0) * (ediff_back > 0),
                    tau_front,
                    th.where(
                        (ediff_front <= 0) * (ediff_back <= 0),
                        tau_back,
                        tau_extreme
                    )
                )# (n_i - 2, n_atom, n_dim)
                tau = Fn.normalize(tau.flatten(-2, -1), dim=-1)  # (n_i - 2, n_atom*n_dim)
                # forces
                F_ori[0] = 0.
                F_ori[-1] = 0.
                F_image = F_ori[1: -1].flatten(-2, -1)  # (n_i - 2, n_atom*n_dim)
                # climbing
                #max_indx = th.argmax(self.ener[1:-1])
                #maxF = F_image[max_indx]
                #maxF = maxF - 2 * ((maxF.unsqueeze(0) @ tau[max_indx].unsqueeze(-1)).squeeze()) * tau[max_indx]
                # other NEB
                F_vert = F_image - ((F_image.unsqueeze(1) @ tau.unsqueeze(-1)).squeeze(-1)) * tau
                F_hori = self.k * (th.linalg.norm(tau_front, dim=(-2, -1)) - th.linalg.norm(tau_back, dim=(-2, -1))).unsqueeze(-1) * tau
                F_ori[1: -1] = (F_vert + F_hori).reshape(n_i-2, n_atom, n_dim)
                print(f'NEB forces: {F_ori.tolist()}')
                #F_ori[max_indx + 1] = maxF.reshape(n_atom, n_dim)

            return - F_ori

    def run(
            self,
            func: Any | nn.Module,
            X_init: th.Tensor,
            X_fin: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            is_grad_func_contain_y: bool = True,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
    ):
        """
        run Dimer algo.
        Args:
            func: function
            X_init: initial structure coordinates
            X_fin: finale structure coordinates
            grad_func: function of func's gradient
            func_args: function args
            func_kwargs: function kwargs
            grad_func_args: gradient function args
            grad_func_kwargs: gradient function kwargs
            is_grad_func_contain_y: whether gradient function contains dependant various y
            output_grad: whether output gradient
            fixed_atom_tensor: mask of fixed atoms

        Returns:

        """

        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        if func_kwargs is None:
            func_kwargs = dict()

        t_main = time.perf_counter()
        n_batch, n_atom, n_dim = X_init.shape
        if grad_func is None:
            is_grad_func_contain_y = True

            def grad_func_(y, x, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                g = th.autograd.grad(y, x, grad_shape)
                return g[0]
        else:
            grad_func_ = grad_func
        '''# Selective dynamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X_init, device=self.device)
        elif fixed_atom_tensor.shape == X_init.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not have the same shape of X (shape: {X_init.shape}).')
        atom_masks_ = atom_masks.flatten(-2, -1)  # (n_batch, n_atom*n_dim)'''


        # check
        if len(X_init.shape) != 2:
            X_init.squeeze_(0)
            X_fin.squeeze_(0)
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        if X_init.shape != X_fin.shape:
            raise ValueError(f'Shape of X_init {X_init.shape} and X_fin{X_fin.shape} does not match.')
        elif len(X_init.shape) != 2:
            raise ValueError(f'Invalid shape of X_init: {X_init.shape}, it mast be (n_atom, 3).')
        X_init = X_init.to(self.device)
        X_fin = X_fin.to(self.device)

        # interpolation images
        ind = th.linspace(0., 1., self.N_images + 2).tolist()
        X_pnts = [(X_init + _ * (X_fin - X_init)).unsqueeze(0) for _ in ind]  # (N_img, n_atom, n_dim)
        X_pnts = th.cat(X_pnts, dim=0)
        plist = list()  # TEST <<<<
        is_main_loop_converge = False
        # Main Loop
        X_pnts_ = X_pnts.flatten(-2, -1)  # (N_img, n_atom * n_dim). NOTE: '_' means the flatten variables.
        f = self.__NebWrapper(func, grad_func_, self.spring_const, is_grad_func_contain_y)
        self.Optimizer: CG
        with th.no_grad():
            _energy, _X, plist = self.Optimizer.run(
                f.energy,
                X_pnts,
                f.grad,
                func_args,
                func_kwargs,
                grad_func_args,
                grad_func_kwargs,
                is_grad_func_contain_y,
                False,
                fixed_atom_tensor
            )

        return _energy, _X, plist  # TEST <<<<<<

        pass