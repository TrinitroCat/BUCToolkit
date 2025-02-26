#  Copyright (c) 2025.2.13, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: _GPR.py
#  Environment: Python 3.12

"""
GaussianProcessRegressor to implement Bayes optimization
"""
from math import log
from typing import Literal, Any, Sequence, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from BM4Ckit.BatchOptim.minimize.CG import CG


#th.random.manual_seed(151)
class GPR:

    def __init__(self, l: float=1.2, sigma: float=0.5, optimize=False):
        self.is_fit = False
        self.train_X, self.train_y, self.Kff, self.Kff_inv_mul_y = None, None, None, None
        self.params = {"l": l, "sigma_f": sigma}
        self.optimize = optimize

    def fit(self, X: th.Tensor, y: th.Tensor):
        if not (isinstance(X, th.Tensor) and isinstance(y, th.Tensor)):
            raise TypeError(f'Both X, y must be torch.Tensor, not {type(X), type(y)}.')
        X = th.atleast_2d(X)
        y = y.unsqueeze(-1) if len(y.shape) < 2 else y

        # hyperparameter optimization
        def negative_log_likelihood_loss(params: th.Tensor):
            params = th.abs(params.squeeze(0, 1)) + 1e-4
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            self.Kff = self.kernel(X, X)  # (B, B)
            _check: th.Tensor = (self.params["l"] < 1e4) * (self.params["sigma_f"] < 1e4)
            if _check:
                self.Kff_inv_mul_y = th.linalg.solve(self.Kff + 1e-5 * th.eye(self.Kff.shape[0]), y) # (B, B) @ (B, 1) -> (B, 1)
                loss = 0.5 * y.T @ self.Kff_inv_mul_y + 0.5 * th.linalg.slogdet(self.Kff)[1] + 0.5 * len(X) * log(2 * th.pi)
            else:
                #print('ss')
                loss = th.abs(self.params["l"]) + th.abs(self.params["sigma_f"])

            return th.log(loss.ravel())

        #plt.plot(np.arange(-1, 1, 0.01), [th.log(negative_log_likelihood_loss(th.tensor([[[_], [_]]], dtype=th.float32))) for _ in np.arange(-1, 1, 0.01)])
        #plt.show()

        if self.optimize:
            optim = CG(
                'PR+',
                1e-3,
                1e-2,
                100,
                'Backtrack',
                steplength=0.1,
                verbose=0
            )
            loss_, param = optim.run(
                negative_log_likelihood_loss,
                th.tensor([self.params["l"], self.params["sigma_f"]]).unsqueeze(0).unsqueeze(0),
            )
            self.params["l"], self.params["sigma_f"] = th.abs(param.squeeze(0, 1)) + 1e-4
            """res = minimize(negative_log_likelihood_loss, np.asarray([self.params["l"], self.params["sigma_f"]]),
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]"""

        # store train data
        self.train_X = X  # (B, N)
        self.train_y = y  # (B, 1)
        self.Kff = self.kernel(self.train_X, self.train_X)  # (B, B)
        self.Kff_inv_mul_y = th.linalg.solve(self.Kff, self.train_y)  # (B, B) @ (B, 1) -> (B, 1)

        self.is_fit = True

    def predict(self, X: th.Tensor):
        if not isinstance(X, th.Tensor):
            raise TypeError(f'X must be torch.Tensor, not {type(X), type(y)}.')
        if not self.is_fit:
            raise RuntimeError("GPR Model not fit yet.")

        X = th.atleast_2d(X)  # (K, N)
        n_p = X.shape[0]
        Kff = self.Kff
        Kff_inv_mul_y = self.Kff_inv_mul_y
        Kyy = self.kernel(X, X)  # (K, K)
        Kfy = self.kernel(self.train_X, X)  # (B, K)
        Kff_inv_mul_Kfy = th.linalg.solve(Kff, Kfy)  # (B, B) @ (B, K) -> (B, K)
        #Kff_inv = th.linalg.inv(Kff + 1e-8 * th.eye(len(self.train_X)))  # (B, B)

        mu = Kfy.mT @ Kff_inv_mul_y  # (K, B) @ (B, 1) -> (K, 1)
        mu = mu.squeeze() if n_p != 1 else mu.squeeze().unsqueeze(0)
        cov = Kyy - Kfy.mT @ Kff_inv_mul_Kfy  # (K, K) - (K, B) @ (B, K) -> (K, K)
        cov = th.where(th.abs(cov) < 1e-6, 0., cov)
        return mu, cov

    def kernel(self, x1, x2) -> th.Tensor:
        dist_matrix = th.linalg.norm(x1.unsqueeze(1) - x2.unsqueeze(0), dim=-1)**2  # (B, B)
        z = self.params["sigma_f"] ** 2 * th.exp(-(0.5 / self.params["l"] ** 2) * dist_matrix)
        return z

    def __call__(self, X, X_train, y_train):
        if not (isinstance(X, th.Tensor) and isinstance(y_train, th.Tensor)):
            raise TypeError(f'Both X, y must be torch.Tensor, not {type(X), type(y_train)}.')
        X = th.atleast_2d(X)
        X_train = th.atleast_2d(X_train)
        y_train = y_train.unsqueeze(-1) if len(y_train.shape) < 2 else y_train
        # store train data
        self.train_X = X_train  # (B, N)
        self.train_y = y_train  # (B, 1)
        self.Kff = self.kernel(self.train_X, self.train_X)  # (B, B)
        self.Kff_inv_mul_y = th.linalg.solve(self.Kff, self.train_y)  # (B, B) @ (B, 1) -> (B, 1)
        self.is_fit = True
        _mu, _cov = self.predict(X)

        return _mu, _cov


class BayesianOptimization:
    def __init__(
            self,
            acquis_func: Literal['PI', 'EI', 'EI-PI'],
            N_init_points: int = 3,
            eps: float = 1e-5,
            E_threshold: float = 1e-3,
            maxiter: int = 100,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        """

        Args:
            acquis_func: acquisition function
            N_init_points: number of initial sampling points generated by stochastic perturbation.
            E_threshold:
            maxiter:
            device:
            verbose:
        """
        self.acquis_func = acquis_func
        self.N_init_points = N_init_points
        self.E_threshold = E_threshold
        self.maxiter = maxiter
        self.device = device
        self.verbose = verbose
        self.mu = None
        self.sigma = None
        self.eps = eps

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            gpr_param=None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
    ):
        if gpr_param is None:
            gpr_param = {'l': 1.2, 'sigma': 0.5, 'optimize': False}
        if len(X.shape) > 2:
            X = X.flatten(2, -1)
        n_batch, n_dim = X.shape
        X = th.randn((self.N_init_points, n_batch, n_dim), dtype=X.dtype, device=self.device) * 10 + X.unsqueeze(0)  # (n_batch, n_pt, n_atom)
        # func_wrapper
        func_wrap = th.vmap(func, in_dims=0, out_dims=0)
        y_ = func_wrap(X)
        # GP_wrapper
        gpr_ = GPR(**gpr_param)
        gpr_func = th.vmap(gpr_, in_dims=1, out_dims=1)
        # 1D search
        '''_ = [self._acq_func(gpr_func, x[None, None, None].expand(1, n_batch, 1), X, y_).numpy() for x in th.linspace(0, 10, 1000)]
        plt.plot(th.linspace(0, 10, 1000), _)
        plt.show()'''
        pass

    def _acq_func(self, gpr_func, x, x_train, y_train):
        y_min = th.min(y_train, dim=0).values  # (n_batch, )
        _mu, _sig = gpr_func(x, x_train, y_train)  # (n_batch, 1), (1, n_batch, 1)
        _mu.squeeze_()
        _sig.squeeze_()
        b = th.sqrt(_sig)
        if self.acquis_func == 'EI':
            z = (- _mu + y_min - self.eps)/b  # (n_batch, )
            EI = (- _mu + y_min - self.eps) * th.erf(z) + b * 1/(2 * th.pi)**0.5 * th.exp(-0.5 * z**2)
            EI = th.where(th.abs(b) < 1e-6, 0., EI)
            return EI


def y(x, noise_sigma=0.0):
    """th.random.manual_seed(114514)
    w1 = th.rand(1, 10)
    w2 = th.rand(10, 10)
    w3 = th.rand(10, 1)
    y = th.cos(x @ w1)
    y = th.cos(y @ w2)
    y = y @ w3"""
    z = th.cos(0.5 * x) + 2. * th.sin(x) - x
    return z.squeeze()


train_X = th.linspace(0, 10, 2).reshape(-1, 1)
train_y = y(train_X, noise_sigma=1e-4)
test_X = th.arange(-10, 10, 0.02).reshape(-1, 1)

gpr = GPR()
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)
test_y = mu.ravel()
true_y = y(test_X)
uncertainty = 1.96 * (th.diag(cov))**0.5
#print(uncertainty)
plt.figure()
plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, true_y, label="true")
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_y, label="train", c="red", marker="x")
plt.legend()
plt.show()

bo = BayesianOptimization('EI', 5)
bo.run(y, train_X)
