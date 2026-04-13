#  Copyright (c) 2026.3.29, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: ModelOptims.py
#  Environment: Python 3.12
import math
from collections import deque

import torch as th
from torch.optim import Optimizer

class FIRELikeOptimizer(Optimizer):
    """
    FIRE-like optimizer based on damped Newton dynamics.
    When power = -grad @ velocities > 0, which means loss descents, n_count += 1;
    when power <= 0, apply an infinite pulse that sets velocity and N_min to zero, and lr to initial lr.
    Once n_count >= N_min, lr *= fi (fi > 1.), i.e., increase learning rate.

    Args:
        params: model parameters.
        lr: initial learning rate, i.e., time steplength.
        mass: hyperparameter of mass.
        N_min: steps that keep lr. If contiguous loss descent step number >= N_min, increase lr.
        tolerance: steps that keep lr & veloc. If contiguous loss increment step number >= tolerance, reset lr to the default and veloc. to 0.
        fi: learning rate increment factor
        max_lr: maximum learning rate.
    """
    def __init__(self, params, lr=1e-2, mass=1.0, N_min=10, tolerance=5, fi=1.1, max_lr=None):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        else:
            lr = float(lr)
        if mass <= 0:
            raise ValueError(f"mass must be positive: {mass}")
        else:
            mass = float(mass)
        if N_min < 1:
            raise ValueError(f"N_min must be >= 1: {N_min}")
        else:
            N_min = int(N_min)
        if tolerance < 0:
            raise ValueError(f"tolerance must be positive: {tolerance}")
        else:
            tolerance = int(tolerance)
        if fi <= 1.0:
            raise ValueError(f"fi must be > 1.0: {fi}")
        else:
            fi = float(fi)
        if max_lr is None:
            max_lr = lr * 10.
        elif max_lr < lr:
            raise ValueError("max_lr must be >= initial lr")
        else:
            max_lr = float(max_lr)

        defaults = dict(lr=lr, mass=mass, N_min=N_min, tolerance=tolerance, fi=fi, max_lr=max_lr)
        super().__init__(params, defaults)

        # store 'lr' as the baseline, and 'lr_now' initialized to current really used lr.
        for group in self.param_groups:
            group['lr_now'] = group['lr']

        # decrement counter
        self.n_count_dec = 0
        # increment counter
        self.n_count_inc = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # ---------- 1. Power P = -sum(grad · velocity) ----------
        power = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # lazy launch initial veloc.
                if 'velocity' not in state:
                    state['velocity'] = th.zeros_like(p.data)
                if power is None: power = th.scalar_tensor(0.0, dtype=p.data.dtype, device=p.device)
                vel = state['velocity']
                #power -= th.dot(grad.view(-1), vel.view(-1)).item()
                power: th.Tensor
                power -= th.sum(vel * grad)  # DO NOT USE `.item()` which causes global synchronization

        # ---------- 2. infinite damp pulse ----------
        # scalar Tensor allows directly comparison
        if power <= 0:  # loss increment
            # sets velocity and N_min to zero, and lr to initial lr.
            self.n_count_dec = 0
            self.n_count_inc += 1
            for group in self.param_groups:
                if self.n_count_inc >= group['tolerance']:
                    group['lr_now'] = group['lr']
                    for p in group['params']:
                        state = self.state.get(p)
                        if (state is not None) and ('velocity' in state):
                            state['velocity'].zero_()
        else:
            self.n_count_dec += 1
            self.n_count_inc = 0
            # check if contiguously decreased at least N_min steps
            for i, group in enumerate(self.param_groups):
                if self.n_count_dec >= self.param_groups[i]['N_min']:
                    group['lr_now'] = min(group['lr_now'] * group['fi'], group['max_lr'])

        # ---------- 3. semi-implicit Euler update ----------
        for group in self.param_groups:
            lr = group['lr_now']
            mass = group['mass']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                vel = state['velocity']
                # a = -grad / mass
                neg_acc = grad / mass  # here avoid one O(n) tensor ops
                # v: v = v + a * dt
                vel.add_(neg_acc, alpha=-lr)
                # p: p = p + v * dt
                p.data.add_(vel, alpha=lr)

        return loss

    # ---------- store/resume states ----------
    def state_dict(self):
        state = super().state_dict()
        state['n'] = self.n_count_dec
        return state

    def load_state_dict(self, state_dict):
        self.n_count_dec = state_dict.pop('n')
        super().load_state_dict(state_dict)


class LangevinOptimizer(Optimizer):
    r"""
    Langevin optimizer based on Newton dynamics with Langevin thermostat.
    Use Equipartition theorem or Virial theorem <x_i \partial H / \partial x_i> == k_B T to
    judge whether equilibrium reached, wherein k_B = 1. If equilibrium reached, temperature decreases.

    Args:
        params: model parameters.
        lr: initial learning rate, i.e., time steplength.
        mass: hyperparameter of mass.
        alpha: damping coefficient.
        temperature: initial virtual temperature.
        anneal_coeff: anneal coefficient to descent temperature.
        balance_tol: relative tolerance of differences between k_B T and the Virial, which calculates as `(Virial - T)/T`
        time_window_size: time window size to count the average <x_i \partial H / \partial x_i>.
        heating_max_steps: the maximum heating steps. Once the number of calling `self.step()` over it, use normal SGD with momenta instead.
        momenta_in_sgd: momentum decay coefficient after annealing.
    """
    def __init__(
            self,
            params,
            lr=1e-3,
            mass=1.0,
            alpha=50.,
            temperature=1.,
            anneal_coeff=0.9,
            balance_tol=0.1,
            time_window_size=500,
            heating_max_steps=100000,
            momenta_in_sgd=0.6,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        else:
            lr = float(lr)
        if mass <= 0:
            raise ValueError(f"mass must be positive: {mass}")
        else:
            mass = th.scalar_tensor(mass, dtype=th.float32)
        if anneal_coeff > 1. or anneal_coeff < 0.:
            raise ValueError(f"anneal_coeff must be in [0, 1]: {anneal_coeff}")
        time_window_size = int(time_window_size)
        if time_window_size < 1:
            raise ValueError(f"time_window_size must be positive: {time_window_size}")

        # Virial queue
        self.virials = deque([0.] * time_window_size, maxlen=time_window_size)
        self.Virial_now = 0.
        self.heating_max_steps = heating_max_steps
        self.heating_step_now = 0

        damp = math.exp(- alpha * lr)
        defaults = dict(
            lr=lr,
            mass=mass,
            damp=damp,
            temperature=temperature,
            anneal_coeff=anneal_coeff,
            balance_tol=balance_tol,
            momenta_in_sgd=momenta_in_sgd
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # ---------- ABOBA update ----------
        with th.no_grad():
            for group in self.param_groups:
                if self.heating_step_now >= self.heating_max_steps:  # enough annealing, use trivial SGD instead
                    lr = group['lr']
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        grad = p.grad.data
                        state = self.state[p]
                        # lazy launch initial veloc.
                        if 'velocity' not in state:
                            state['velocity'] = th.zeros_like(p.data)
                        vel = state['velocity']
                        vel.mul_(group['momenta_in_sgd']).add_(grad)
                        p.add_(vel, alpha=-lr)

                else:  # keep annealing
                    lr = group['lr']
                    mass = group['mass']
                    damp = group['damp']
                    temp = group['temperature']
                    anneal_coeff = group['anneal_coeff']
                    balance_tol = group['balance_tol']
                    xpHpx = 0.
                    sizes = 0
                    _vir = 0.
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]
                        # lazy launch initial veloc.
                        if 'velocity' not in state:
                            state['velocity'] = th.randn_like(p) * th.sqrt(temp / mass)
                        vel = state['velocity']
                        # Calc. Virial
                        #xpHpx += th.sum(p * grad)  # x_i * \partial H / \partial x_i
                        sizes += p.numel()
                        _vir += (vel.pow(2).sum() * 0.5 * mass).item()
                        mass = mass.to(p.dtype).to(p.device)
                        # ABOBA
                        vel.addcdiv_(grad, mass, value=- 0.5 * lr)
                        p.add_(vel, alpha=0.5 * lr)
                        # stochastic update velocity
                        vel.mul_(damp)
                        vel.add_(th.sqrt((temp * (1 - damp ** 2)) / mass) * th.randn_like(vel))
                        # V = damp * V + th.sqrt((self.T_init * (1 - damp ** 2)) / masses) * th.randn_like(V)
                        # the rest half-step
                        p.add_(vel, alpha=0.5 * lr)
                        vel.addcdiv_(grad, mass, value=-0.5 * lr)

                    # Annealing, check the Equipartition theorem
                    self.Virial_now -= self.virials[0]  # sub the head
                    #_vir = xpHpx / sizes
                    _vir /= sizes
                    self.virials.append(_vir)
                    self.Virial_now += _vir
                    #print(f"Virial: {(self.Virial_now/len(self.virials)): <.4e}, T: {temp}")
                    #avg_grad_norm = sum(p.grad.norm().item() for p in group['params'] if p.grad is not None) / len(group['params'])
                    #avg_param_norm = sum(p.norm().item() for p in group['params']) / len(group['params'])
                    #print(f"Step {self.heating_step_now}: Virial={self.Virial_now/(0.5 * len(self.virials)):.4e}, T={temp:.4e}")

                    #if abs((self.Virial_now/len(self.virials) - temp)/temp) <= balance_tol:
                    #    group['temperature'] *= anneal_coeff
                    if abs((self.Virial_now/(0.5 * len(self.virials)) - temp) / temp) <= balance_tol:
                        group['temperature'] *= anneal_coeff

        self.heating_step_now += 1

        return loss

    # ---------- store/resume states ----------
    def state_dict(self):
        state = super().state_dict()
        state.update({
            'virials': self.virials, 'Virial_now': self.Virial_now,
            'heating_step_now': self.heating_step_now,
            'heating_max_steps': self.heating_max_steps,
        })
        return state

    def load_state_dict(self, state_dict):
        self.virials = state_dict.pop('virials')
        self.virials = deque(self.virials, maxlen=len(self.virials))
        self.Virial_now = state_dict.pop('Virial_now')
        self.heating_step_now = state_dict.pop('heating_step_now')
        self.heating_max_steps = state_dict.pop('heating_max_steps')
        super().load_state_dict(state_dict)
