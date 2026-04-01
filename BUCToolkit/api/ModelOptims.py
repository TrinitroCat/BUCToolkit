#  Copyright (c) 2026.3.29, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: ModelOptims.py
#  Environment: Python 3.12

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
