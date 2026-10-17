
import torch as th
import triton
import triton.language as tl


@triton.jit
def CSVR_kernel(
        Ek,
        epsK,
        one_sub_c,
        EK_TARGET,
        Nf,
        sqrt_c,
        _buf_R,
        _buf_S,
):
    K = th.clamp(Ek, min=epsK)
    f = one_sub_c * EK_TARGET / (Nf * K)
    sqrt_f = th.sqrt(th.clamp(f, min=0.0))
    alpha2 = th.addcmul(sqrt_c, sqrt_f, _buf_R) ** 2
    alpha2.addcmul_(f, _buf_S).clamp_min_(epsK)
    alpha = th.sqrt(alpha2)
