import importlib
import importlib.util
import os
import warnings
from typing import List

import torch as th
import triton
import triton.language as tl

_KERNEL_CACHE = {}

_TRITON_ELEMWISE_UNARY_OPS = {
    "abs": "tl.abs",
    "neg": "-",
    "cos": "tl.cos",
    "sin": "tl.sin",
    "exp": "tl.exp",
    "exp2": "tl.exp2",
    "log": "tl.log",
    "log2": "tl.log2",
    "sqrt": "tl.sqrt",
    "sqrt_rn": "tl.sqrt_rn",
    "rsqrt": "tl.rsqrt",
    "erf": "tl.erf",
    "sigmoid": "tl.sigmoid",
    "ceil": "tl.ceil",
    "floor": "tl.floor",
}

_TRITON_ELEMWISE_BINARY_OPS = {
    "add": "tl.add",
    "a": "tl.add",
    "sub": "tl.sub",
    "s": "tl.sub",
    "mul": "tl.mul",
    "m": "tl.mul",
    "fdiv": "tl.fdiv",
    "d": "tl.fdiv",
    "div_rn": "tl.div_rn",
    "cdiv": "tl.cdiv",
    "umulhi": "tl.umulhi",
    "maximum": "tl.maximum",
    "minimum": "tl.minimum",
    # extent set
    #"gemm": "_bgemm_inline",
}

_TRITON_ELEMWISE_TRINARY_OPS = {  # not work now
    "clamp": "tl.clamp",
    "where": "tl.where",
    "fma": "tl.fma",
}

def _gen_macro_source(
        n_vars: int,
        ops_list: List[str],
        ops_signatures: List[int],
        ops_name: str
) -> None:
    """
    A macro to generate the meta code.
    Args:
        n_vars:
        ops_list:
        ops_signatures:

    Returns:

    """
    if n_vars < 0:
        raise ValueError("n_vars must be >= 0")

    ptr_names = [f"x_ptr_{i}" for i in range(n_vars + 1)]
    load_lines = [
        f"x_{i} = tl.load(x_ptr_{i} + offsets, mask=mask, other=0.0)"  # at most 3 register to be used at the same time
        for i in range(n_vars + 1)
    ]
    _expr_list = []
    t_ptr = 0
    for i, op in enumerate(ops_list):
        if ops_signatures[i] == 1:
            _expr_list.append(f"x = {op}(x)")
        elif ops_signatures[i] == 2:
            t_ptr += 1
            _expr_list.append(f"{load_lines[t_ptr]}\n    x = {op}(x, x_{t_ptr})")
        elif ops_signatures[i] == 3:
            t_ptr += 2
            _t_ptr = t_ptr - 1
            _expr_list.append(
                f"{load_lines[_t_ptr]}; {load_lines[t_ptr]}\n    x = {op}(x, x_{_t_ptr}, x_{t_ptr})"
            )

    _expr_str = '\n    '.join(_expr_list)

    macro = f"""
import triton
import triton.language as tl

@triton.jit
def c_elemwise_kernel(
    {", ".join(ptr_names)},
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(x_ptr_0 + offsets, mask=mask, other=0.0)
    {_expr_str}
    output = x
    tl.store(output_ptr + offsets, output, mask=mask)

"""
    uid = os.getuid()
    os.makedirs(f'/dev/shm/.TRITON_OPS_{uid}', exist_ok=True)
    with open(f'/dev/shm/.TRITON_OPS_{uid}/{ops_name}.py', 'w') as f:
        f.write(macro)


def _get_kernel(n_multiplier: int, ops_list, ops_sig_list, ops_chain: str):
    uid = os.getuid()
    if ops_chain not in _KERNEL_CACHE:
        if not os.path.exists(f'/dev/shm/.TRITON_OPS_{uid}/{ops_chain}.py'):
            _gen_macro_source(n_multiplier, ops_list, ops_sig_list, ops_chain)
        else:
            warnings.warn(f"The operator file {ops_chain}.py already exists. Code generation skipped.")

        module_name = f"_dyn_kernel_{ops_chain}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            f'/dev/shm/.TRITON_OPS_{uid}/{ops_chain}.py'
        )
        if spec is None:
            raise RuntimeError(f"There is a mysterious bug when importing {module_name}. Please report this.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, "c_elemwise_kernel")
        _KERNEL_CACHE[ops_chain] = func

    return _KERNEL_CACHE[ops_chain]


def chained_elem_wise(
        x: th.Tensor,
        *ys: th.Tensor,
        ops_chain: str = 'm',
        out: th.Tensor | None = None,
        block_size: int = 256,
):
    """

    Args:
        x: the initial tensor
        *ys: the tensors that will left-apply to `x`
        ops_chain: the operators chain.
            Keywords:
                'm': multiply
                'a': addition
                's': subtraction
                'd': division
                ... and all other triton build-in element-wise ops, e.g. `sin`, `div_rn`.
            format:
                for example, "s*4+d*2+sin" means tl.sin(((x - y1 - y2 - y3 - y4) / y5 / y6)), that is, applying
                    4 times subtractions, and then 2 divisions, and a sin.
        out: the output dst
        block_size: the block size for cuda use.

    Returns:

    """
    # parse the ops
    ops_chain = ops_chain.replace(' ', '').strip()  # standardise
    _ops_list = ops_chain.split('+')
    ops_list = list()
    ops_sig_list = list()
    for op in _ops_list:
        if "*" not in op:
            op = op + "*1"
        _opl = op.split("*")
        if len(_opl) != 2:
            raise ValueError(f"Invalid operators fragment: '{op}'.")
        op, _n = _opl
        _n = int(_n)
        if _n <= 0:
            raise ValueError(f"repeat count must be > 0, got {_n} for op '{op}'")
        if op in _TRITON_ELEMWISE_UNARY_OPS:
            ops_list.extend([_TRITON_ELEMWISE_UNARY_OPS[op]] * _n)
            ops_sig_list.extend([1] * _n)
        elif op in _TRITON_ELEMWISE_BINARY_OPS:
            ops_list.extend([_TRITON_ELEMWISE_BINARY_OPS[op]] * _n)
            ops_sig_list.extend([2] * _n)
        elif op in _TRITON_ELEMWISE_TRINARY_OPS:
            ops_list.extend([_TRITON_ELEMWISE_TRINARY_OPS[op]] * _n)
            ops_sig_list.extend([3] * _n)
        else:
            raise ValueError(f"Unknown operator {op}.")

    # check the correctness of variable numbers and operators
    n_vars = len(ys)
    required_n_vars = 0
    for _ in ops_sig_list:
        required_n_vars += (_ - 1)  # unary requires 0, binary requires 1, n-variates function requires n-1 vars.
    if required_n_vars != n_vars:
        raise ValueError(
            f"ops_chain requires {required_n_vars} variables, "
            f"but {n_vars} extra tensors were provided."
        )
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of 2")
    # check shapes
    _has_noncontiguous = False if x.is_contiguous() else True
    x = x.contiguous()
    if x.device.type != "cuda":
        raise ValueError(f"device must be cuda device, but got {x.device.type}")
    for _, y in enumerate(ys):
        if y.shape != x.shape:
            raise ValueError(f"shape {tuple(y.shape)} != {tuple(x.shape)}")
        if y.device != x.device:
            raise ValueError("device mismatch")
        if y.dtype != x.dtype:
            raise ValueError("dtype mismatch")
        if y.device.type != "cuda":
            raise ValueError(f"device must be cuda device, but got {y.device.type}")
        if not y.is_contiguous():
            _has_noncontiguous = True
            warnings.warn(f"Detected non-contiguous tensor in {_}-th position. Implicit copy will be used.")
    if _has_noncontiguous:
        ys = [_.contiguous() for _ in ys]

    if out is None:
        output = th.empty(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            memory_format=th.contiguous_format,
        )
    elif (out.shape == x.shape) and (out.dtype == x.dtype) and (out.device == x.device):
        if not out.is_contiguous():
            raise ValueError(f"`out` must be a contiguous tensor.")
        output = out
    else:
        raise ValueError(
            f"Expected the `out` has the same shape, dtype, and device as inputs, "
            f"but got ({out.shape}, {out.dtype}, {out.device}) instead of ({x.shape}, {x.dtype}, {x.device})."
        )
    N = x.numel()
    if N == 0: return output

    kernel = _get_kernel(n_vars, ops_list, ops_sig_list, ops_chain)


    grids = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    kernel[grids](x, *ys, output, N, block_size)

    return output


def purge_all_kernels():
    uid = os.getuid()
    if not os.path.isdir(f'/dev/shm/.TRITON_OPS_{uid}'):
        return
    import shutil
    try:
        shutil.rmtree(f'/dev/shm/.TRITON_OPS_{uid}')
        _KERNEL_CACHE.clear()
    except Exception as e:
        warnings.warn(f"Failed to delete operator cache in /dev/shm/.TRITON_OPS_{uid}, due to \"{e}\"")
