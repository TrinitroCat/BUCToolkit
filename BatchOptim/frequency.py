#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: frequency.py
#  Environment: Python 3.12
"""
Calculating the harmonic frequencies by finite difference algorithm and automatic differentiation.
Also support vibration thermodynamic quantities calculations.
"""
import re
from typing import Literal, Callable, Tuple, Dict, List

import numpy as np
import torch as th



class Frequency:
    """
    Calculate normal mode frequency by finite difference method.

    Args:
        method: 'Coord' for directly calculating 2nd-order deviations, i.e., the Hessian matrix.
                'Grad' for calculating 1st-order deviations of grad to get Hessian matrix.
        block_size: block size to calculate Hessian. `block_size`*N rows would be input at once.
                    `None` for input all (3*N)**2 rows coords at once, which requires enough memory.
        delta: finite difference delta.
    """
    def __init__(self, method: Literal['Coord', 'Grad'] = 'Coord', block_size: None | int = None, delta: float = 1e-2):
        self.hessian = None
        self.method = method
        self.block_size = block_size
        self.delta = delta

    class _func_wrapper:
        """
        Check the function output format
        """
        def __init__(self, func, args: Tuple = tuple(), kwargs: Dict|None = None):
            if kwargs is None: kwargs = dict()
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def __call__(self, x):
            y: th.Tensor = self.func(x, *self.args, **self.kwargs)
            if y.dim() > 1:
                y.squeeze_()
                if y.dim() > 1: raise ValueError(f'Expected a 1-dimensional output, but got {y.dim()} dim.')
            return y

    def _create_finite_diff_tensor(self, coords: th.Tensor, fixed_atom_tensor: th.Tensor | None):
        """
        Generating finite difference tensor with shape (6N+1, N, 3)

        Args:
            coords: (N, 3).
            fixed_atom_tensor: indices of free atoms. generate from `fixed_atom_tensor` in method `normal_mode`.

        Returns：
            positive_difference, negative_difference, origin_point: (3 N, N, 3), (3 N, N, 3), (1, N, 3)
        """

        coords_free = coords[fixed_atom_tensor]
        N = coords_free.shape[0]
        coords_out = coords.unsqueeze(0).expand(3*N, -1, -1)
        device = coords_free.device
        dtype = coords_free.dtype

        # displacement indx
        k_indices = th.arange(6 * N, device=device)
        i = k_indices // 6  # each point corresponding 2 directions, i.e., the center difference algo.
        direction = (k_indices % 6) // 2  # direction：0=x,1=y,2=z
        # odd remainder: negative; even remainder: positive
        sign = th.where((k_indices % 6) % 2 == 0, 1.0, -1.0).to(dtype=dtype)
        # initialize finite difference tensor
        delta_tensor = th.zeros(6 * N, N, 3, device=device, dtype=dtype)
        delta_tensor[k_indices, i, direction] = self.delta * sign
        displaced = coords_free.unsqueeze(0) + delta_tensor  # (6N, N, 3)
        # positive tensor
        pos_indx = th.arange(0, 6 * N, 2)
        # negative tensor
        neg_indx = th.arange(1, 6 * N, 2)
        # split directions
        diff_pos = displaced[pos_indx]  # (3N, N, 3)
        diff_neg = displaced[neg_indx]
        # Adding the original point
        origin = coords.unsqueeze(0)  # (1, N, 3)
        #diff_tensor = th.cat([origin, displaced], dim=0)  # (6N+1, N, 3)
        diff_pos_out = coords_out.clone()
        diff_pos_out[:, fixed_atom_tensor] = diff_pos
        diff_neg_out = coords_out.clone()
        diff_neg_out[:, fixed_atom_tensor] = diff_neg

        return diff_pos_out, diff_neg_out, origin

    def create_hessian(
            self,
            func: Callable,
            coords: th.Tensor,
            func_args: Tuple = tuple(),
            func_kwargs: Dict|None = None,
            save_hessian: bool = False,
            fixed_atom_tensor: th.Tensor | None = None,
    ) -> th.Tensor:
        """
        To calculate Hessian matrix in blocks via finite difference
        Args:
            func:
            coords: (N, 3) shape Tensor
            func_args: other input arguments of `func`.
            func_kwargs: other input keyword arguments of `func`.
            save_hessian: whether save calculated hessian as an attribute.
            fixed_atom_tensor: the indices of X that fixed, i.e., not performing vibration calculation.

        Returns:

        """
        if func_kwargs is None:
            func_kwargs = dict()
        device = coords.device
        dtype = coords.dtype
        func_ = self._func_wrapper(func, func_args, func_kwargs)
        diff_pos, diff_neg, origin = self._create_finite_diff_tensor(coords, fixed_atom_tensor)

        pp = diff_pos.unsqueeze(1) + diff_pos.unsqueeze(0) - origin  # (3N, 3N, N, 3)
        np = diff_neg.unsqueeze(1) + diff_pos.unsqueeze(0) - origin
        pn = diff_pos.unsqueeze(1) + diff_neg.unsqueeze(0) - origin
        nn = diff_neg.unsqueeze(1) + diff_neg.unsqueeze(0) - origin

        # reformat as batch dimension
        n_free = len(diff_pos)
        n_atom = n_free // 3
        all_batch = len(pp)**2
        block_size = all_batch if self.block_size is None else self.block_size
        n_int_block = pp.flatten(0, 1).shape[0] // block_size
        n_rest = pp.flatten(0, 1).shape[0] % block_size
        real_block_size = [block_size] * n_int_block + [n_rest]  # manage the problem that all_batch could not be divided by given block_size.
        pp = th.split(pp.flatten(0, 1), real_block_size)  # (9N**2, N ,3)
        np = th.split(np.flatten(0, 1), real_block_size)
        pn = th.split(pn.flatten(0, 1), real_block_size)
        nn = th.split(nn.flatten(0, 1), real_block_size)
        num_block = len(pp)

        # split blocks
        start = 0
        hessian = th.empty(all_batch, device=device, dtype=dtype)
        for _indx in range(num_block):
            end = start + block_size
            hessian[start:end] = (
                                         func_(pp[_indx]) - func_(np[_indx]) - func_(pn[_indx]) + func_(nn[_indx])
                                 ) / (4 * self.delta ** 2)  # (3N * 3N)
            start = end
        hessian = hessian.reshape(n_free, n_free)
        if save_hessian:
            self.hessian = hessian

        return hessian.squeeze(0)

    def hessian_by_autograd(
            self,
            grad_func,
            X: th.Tensor,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs: Dict | None = None,
            fixed_atom_tensor: th.Tensor | None = None,
            save_hessian: bool = False
    ):
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()

        if X.dim() == 2:
            X = X.unsqueeze(0)
        elif X.dim() != 3:
            raise ValueError(f'Expected X has 2 or 3 dimensions, but got {X.shape}')
        X_flat = X.flatten(-2, -1)
        if fixed_atom_tensor is None:
            fixed_atom_tensor = th.arange(X_flat.size(1), dtype=X.dtype, device=X.device)

        n_batch, n_atom, n_dim = X.shape
        n_free_atom = len(fixed_atom_tensor)
        fixed_atom_list = fixed_atom_tensor.tolist()

        with th.enable_grad():
            X_flat.requires_grad_()
            X_ = X_flat.reshape(n_batch, n_atom, n_dim)
            g: th.Tensor = grad_func(X_, *grad_func_args, **grad_func_kwargs)
            g = g.reshape(n_batch, n_atom * n_dim)
            # Hessian
            hessian = th.zeros((n_batch, n_free_atom, n_free_atom), device=X.device)
            hess_mask = th.zeros(n_batch, n_atom * n_dim, device=X.device)
            for i, indx in enumerate(fixed_atom_list):
                hess_mask[:, indx] = 1.
                H_line = th.autograd.grad(g, X_flat, hess_mask, retain_graph=True)[0]  # (n_batch, n_atom*n_dim, 1)
                hessian[:, :, i] = H_line.squeeze(-1)[:, fixed_atom_tensor].detach()
                hess_mask[:, indx] = 0.
            del H_line, g, X_flat
        if save_hessian:
            self.hessian = hessian.squeeze(0)

        return hessian.squeeze(0)

    def normal_mode(
            self,
            func: Callable,
            coords: th.Tensor,
            masses: th.Tensor|None = None,
            func_args: Tuple = tuple(),
            func_kwargs: Dict|None = None,
            grad_func: Callable | None = None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs: Dict | None = None,
            fixed_atom_tensor: th.Tensor | None = None,
            save_hessian: bool = False
    ):
        """
        Calculating the normal modes and corresponding frequencies.
        Notice: The negative frequencies are actually imaginary frequencies.
        Args:
            func: potential energy function.
            coords: atomic coordinate tensor with shape (N, 3).
            masses: atomic masses tensor with shape (N, 3) or (N, ). `None` for tensor filled with 1.
            func_args: other input arguments of `func`.
            func_kwargs: other input keyword arguments of `func`.
            grad_func: gradient function, only used for self.method = 'Grad'.
            grad_func_args: other input arguments of `grad_func`.
            grad_func_kwargs: other input keyword arguments of `grad_func`.
            fixed_atom_tensor: (N, ) or (N, 3), the indices of X that fixed, i.e., not performing vibration calculation.
                               Only the 1st dimension will be read. Fixing partial degrees of freedom for an atom is not supported.
            save_hessian: whether save calculated hessian as an attribute.

        Returns:
            normal mode frequencies: (3N, ) shape Tensor
            normal mode coordinates: (3N, N, 3) shape Tensor
        """
        # check
        with th.no_grad():
            if fixed_atom_tensor is None:
                fixed_atom_tensor = th.ones_like(coords[:, 0], dtype=th.int)
            elif fixed_atom_tensor.shape == coords.shape:
                fixed_atom_tensor = fixed_atom_tensor[:, 0]
            elif not fixed_atom_tensor.shape == coords[:, 0].shape:
                raise ValueError(f'Invalid shape of fixed_atom_tensor: {fixed_atom_tensor.shape}')
            if th.all(fixed_atom_tensor != 1):
                raise ValueError(f'All atoms are fixed. No vibration will be calculated.')
            fixed_atom_tensor_indx = th.where(fixed_atom_tensor)[0]  # (n_free_atom, )
            fixed_atom_tensor_ex = fixed_atom_tensor.unsqueeze(-1).expand(-1, coords.size(-1)).flatten(0, 1)  # (n_free_atom * 3)
            fixed_atom_tensor_ex = th.where(fixed_atom_tensor_ex)[0]
            device = coords.device
            if (coords.shape[-1] != 3) or (coords.dim() != 2):
                raise ValueError(f'Expected `coords` have shape (N, 3), but got {coords.shape}')
            if masses is None:
                masses = th.ones_like(coords)
            else:
                if (masses.shape[0] != coords.shape[0]) or masses.dim() > 2:
                    raise ValueError(f'Expected `masses` have shape (N, ) or (N, 3), but got {masses.shape}')
                elif masses.dim() < 2:
                    masses = th.broadcast_to(masses.unsqueeze(-1), coords.shape)   # transfer to (N, 3)
            masses = masses.to(device)
            masses = masses[fixed_atom_tensor_indx]
            flat_masses = masses.flatten(0, 1)
            hess_weights = th.sqrt(flat_masses.unsqueeze(-1) * flat_masses.unsqueeze(0))
            # calc hessian
            if self.method == 'Coord':
                hessian = self.create_hessian(func, coords, func_args, func_kwargs, fixed_atom_tensor=fixed_atom_tensor_indx)
            else:
                hessian = self.hessian_by_autograd(grad_func, coords, grad_func_args, grad_func_kwargs, fixed_atom_tensor=fixed_atom_tensor_ex)
            weighted_hessian = hessian/hess_weights
            norm_freq_square, weighted_norm_mode = th.linalg.eigh(weighted_hessian)
            neg_indx = th.where(norm_freq_square < 0.)[0]
            norm_freq = th.sqrt(th.abs(norm_freq_square))
            norm_freq[neg_indx] = - norm_freq[neg_indx]
            norm_mode = (weighted_norm_mode/flat_masses).reshape(norm_freq_square.size(-1), -1, 3)
            if save_hessian:
                self.hessian = hessian

            return norm_freq, norm_mode


def vibrational_thermo(frequencies: th.Tensor, T: float = 298.15):
    r"""
    Thermodynamic quantities calculation for condensed states, i.e., U = H, F = G, and \delta(PV) = 0.
    输入:
        frequencies: (N,), frequencies tensor, unit: cm^-1. Negative values mean imaginary frequencies.
        T: Temperature. unit: K.
    输出:
        S_vib: Entropy (kJ/mol·K)
        H_vib: Enthalpy (eV)
        C_vib: Thermo-capacity (kJ/mol·K)
        F_vib: Free-energy (eV)
    """
    # CONSTANTS
    HC = 1.98630e-23  # h*c (J·cm)
    NA = 6.02214076e23  # Avogadro const. (mol^-1)
    KB_J = 1.380649e-23  # Boltzmann const. (J/K)
    CM_TO_eV = 1.23981e-4  # cm^-1 -> eV (1 cm^-1 = 1.23981e-4 eV)
    CM_TO_kJ_PER_MOL = HC * NA * 1e-3  # cm^-1 → kJ/mol
    CM_TO_J_PER_MOL = HC * NA  # cm^-1 → J/mol
    J_PER_MOL_TO_eV = 1e3 / (NA * 1.60218e-19)  # J/mol → eV (约 1.0364e-2)
    CM_TO_eV_PER_MOL = CM_TO_J_PER_MOL * J_PER_MOL_TO_eV  # cm^-1 → eV
    # remove imag. freq.
    positive_mask = frequencies > 0
    if th.all(~positive_mask):
        raise RuntimeError(f'Input frequencies are all imaginary frequencies. I HOPE YOU KNOW WHAT YOU ARE DOING.')
    nu = frequencies[positive_mask]

    # set freq < 50 cm^-1 to 50 cm^-1 to avoid abnormal entropy contribution.
    nu = th.where(nu < 50., 50., nu)

    # x = hν/kT = ν/(kB * T)
    kB_cm = KB_J / HC
    x = nu / (kB_cm * T)

    # if x > 100, considered T as a small value. e.g., for freq. of 3200 cm^-1 and T = 298 K, x = 5.406 << 100. If x = 100, T is about merely 16 K.
    high_x = x > 100

    # U
    exp_x = th.exp(x)
    zero_point_energy = 0.5 * nu
    thermal_energy = nu / (exp_x - 1 + 1e-16)
    U_vib = th.where(high_x, zero_point_energy, zero_point_energy + thermal_energy)

    # C
    denom = (exp_x - 1) ** 2
    C_factor = (x ** 2) * exp_x / (denom + 1e-16)
    C_vib_cm = th.where(high_x, th.zeros_like(x), kB_cm * C_factor)

    # S
    S1 = x / (exp_x - 1 + 1e-16)
    S2 = th.log(1 - th.exp(-x) + 1e-16)
    S_vib_cm = kB_cm * th.where(high_x, th.zeros_like(x), S1 - S2)

    # G
    F_vib = th.where(high_x, zero_point_energy, zero_point_energy + kB_cm * T * S2)

    # sum
    U_total = th.sum(U_vib)
    C_total = th.sum(C_vib_cm)
    S_total = th.sum(S_vib_cm)
    F_total = th.sum(F_vib)

    # unit conversion. convert to eV, eV/K
    S_ev = S_total * CM_TO_eV
    C_ev = C_total * CM_TO_eV
    U_ev = U_total * CM_TO_eV
    F_ev = F_total * CM_TO_eV

    return S_ev, U_ev, C_ev, F_ev

def extract_freq(path: str) -> List[np.ndarray]:
    """
    Extract frequency from Vibration calc. output files.
    Args:
        path:

    Returns:

    """
    with open(path, 'r') as f:
        data = f.read()

    pattern = re.compile(r'Eigen Vibration Frequency \(cm\^-1\):\n([+.\-0-9\s\n]+)', flags=re.M)
    freqn = re.findall(pattern, data)
    freq_list = [np.asarray(frl.split(), dtype=np.float32) for frl in freqn]

    return freq_list

__all__ = [
    'Frequency',
    'vibrational_thermo',
    'extract_freq'
]

# 示例用法
if __name__ == "__main__":
    # 创建示例频率张量 (包含正负频率和低频)
    frequencies = th.tensor([
        3083.953574,
        3047.925383,
        3027.645431,
        3018.475192,
        3012.926299,
        2927.218652,
        2799.844693,
        1456.985281,
        1446.552789,
        1417.972562,
        1406.900854,
        1357.130821,
        1294.452227,
        1268.645944,
        1210.800793,
        1154.637385,
        1082.020997,
        993.746725,
        969.939002,
        913.057972,
        871.097270,
        818.914607,
        756.614656,
        552.091194,
        414.467590,
        250.240054,
        185.291922,
        116.075049,
        77.272448,
        63.180734,
        39.135462,
        - 99.207538,
        - 1281.881665,
    ])
    freq2 = th.tensor([
        -2677.8059082000, - 718.62634277000, - 515.33343506000,
        -169.42842102000,   128.61962891000,   196.16116333000,
        259.79168701000 ,  416.30462646000 ,  492.31179810000,
        655.95141602000 ,  718.72033691000 ,  841.46276855000,
        940.41229248000 ,  1146.7479248000 ,  1180.0986328100,
        1234.0729980500 ,  1301.6352539100 ,  1432.3217773400,
        1553.4138183600 ,  1575.9421386700 ,  1638.2396240200,
        1678.4561767600 ,  1733.8581543000 ,  1804.6947021500,
        1953.1727294900 ,  2313.4604492200 ,  2356.5280761700,
        2467.3337402300 ,  2573.1855468800 ,  2679.1540527300,
        2702.9921875000 ,  2840.1782226600 ,  3055.0336914100,
    ])# 单位 cm⁻¹
    T = 298.15  # 温度 (K)
    frequencies = th.sort(frequencies).values
    freq2= th.sort(freq2).values

    S_vib, H_vib, C_vib, F_vib = vibrational_thermo(frequencies, T)
    S_vib_ml, H_vib_ml, C_vib_ml, F_vib_ml = vibrational_thermo(freq2, T)

    print("频率对比", frequencies - freq2)  # 原始正频率

    print(f"\n振动熵: {S_vib.item():.6f}, {S_vib_ml.item():.6f} eV/K")
    print(f"振动焓: {H_vib.item():.6f}, {H_vib_ml.item():.6f} eV")
    print(f"振动热容: {C_vib.item():.6f}, {C_vib_ml.item():.6f} eV/K")
    print(f"振动自由能: {F_vib.item():.6f}, {F_vib_ml.item():.6f} eV")
    print(f"振动自由能差: {F_vib.item() - F_vib_ml.item():.6f} eV")


if __name__ == '__main__' and False:
    a = th.rand(6, 6)
    a = a + a.mT

    '''class fn(th.nn.Module):
        def __init__(self):
            super().__init__()
            seed = th.manual_seed(114514)
            self.layers = th.nn.Sequential(
                th.nn.Linear(6, 100, bias=False),
                th.nn.Tanh(),
                th.nn.Linear(100, 100, bias=False),
                th.nn.Tanh(),
                th.nn.Linear(100, 1, bias=False),
            )

        def forward(self, x):
            x = x.flatten(-2, -1)
            return self.layers(x)**2'''

    def f(x):
        """

        Args:
            x:

        Returns:

        """
        x = x.flatten(-2, -1).unsqueeze(-1)
        y = th.exp(x).mT @ a @ th.cos(x)
        return y.squeeze(-1, -2)

    def grad_f(x:th.Tensor):
        with th.enable_grad():
            x.requires_grad_()
            y = f(x)
            g = th.autograd.grad(y, x, th.ones_like(y), create_graph=True)[0]
        return g


    x0 = th.ones(2, 3)
    freq1 = Frequency('Coord')  # FIXME
    ei1, ej1 = freq1.normal_mode(f, x0, grad_func=grad_f, save_hessian=True, fixed_atom_tensor=th.tensor([0, 1]))

    x0.detach_()
    freq0 = Frequency('Grad')
    ei0, ej0 = freq0.normal_mode(f, x0, grad_func=grad_f, save_hessian=True, fixed_atom_tensor=th.tensor([1, 1]))

    x0.detach_()
    freq2 = Frequency('Grad')
    ei2, ej2 = freq2.normal_mode(f, x0, grad_func=grad_f, save_hessian=True, fixed_atom_tensor=th.tensor([0, 1]))
    pass

