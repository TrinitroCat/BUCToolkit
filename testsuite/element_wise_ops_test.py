# test_chained_elemwise.py
import time
from typing import List

import torch as th

from BUCToolkit.utils.operators.element_wise_ops import chained_elem_wise, purge_all_kernels


# ---------- 纯 Torch 参考实现 ----------
_UNARY_TORCH = {
    "abs": th.abs, "neg": th.neg,
    "cos": th.cos, "sin": th.sin,
    "exp": th.exp, "exp2": th.exp2,
    "log": th.log, "log2": th.log2,
    "sqrt": th.sqrt, "rsqrt": th.rsqrt,
    "erf": th.erf, "sigmoid": th.sigmoid,
    "ceil": th.ceil, "floor": th.floor,
}

_BINARY_TORCH = {
    "add": lambda a, b: a + b, "a": lambda a, b: a + b,
    "sub": lambda a, b: a - b, "s": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "m": lambda a, b: a * b,
    "fdiv": lambda a, b: a / b, "d": lambda a, b: a / b,
    "maximum": th.maximum, "minimum": th.minimum,
    "bitwise_and": th.bitwise_and,
    "bitwise_or": th.bitwise_or,
    "bitwise_xor": th.bitwise_xor,
}


def ref_chain(x: th.Tensor, ys: List[th.Tensor], ops_chain: str) -> th.Tensor:
    """用纯 Torch 复现 chained_elem_wise 的语义。"""
    parts = ops_chain.replace(" ", "").split("+")
    ops: List[str] = []
    for p in parts:
        if "*" in p:
            op, n = p.split("*")
            n = int(n)
        else:
            op, n = p, 1
        ops.extend([op] * n)

    out = x
    yi = 0
    for op in ops:
        if op in _UNARY_TORCH:
            out = _UNARY_TORCH[op](out)
        elif op in _BINARY_TORCH:
            out = _BINARY_TORCH[op](out, ys[yi])
            yi += 1
        else:
            raise ValueError(f"Unknown op: {op}")
    return out

def _safe_ys(ys, ops_chain):
    """避免除零和 log/负数溢出。"""
    if any(op in ops_chain for op in ("d", "fdiv")):
        ys = [y + th.sign(y) * 0.5 + 1e-3 for y in ys]
    if ("log" in ops_chain or "log2" in ops_chain or "sqrt" in ops_chain) \
       and "exp" not in ops_chain:
        ys = [y.abs() + 0.1 for y in ys]
    return ys


def test_correctness(device="cuda", dtype=th.float32):
    print("=" * 78)
    print("Correctness: chained_elem_wise vs pure torch")
    print("=" * 78)

    cases = [
        ("neg",            0),
        ("abs",            0),
        ("sin",            0),
        ("cos+sin+exp",   0),
        ("neg*3",          0),
        ("abs+neg+abs",    0),
        ("a",              1),
        ("s",              1),
        ("m",              1),
        ("d",              1),
        ("a*2+m*3",        5),
        ("s+d+m",          3),
        ("cos+m+s+d+exp", 3),
        ("s*4+d*2+sin",    6),
        ("a*4+m*4",        8),
    ]

    n = 8192
    max_err_global = 0.0
    for ops_chain, n_ys in cases:
        x = th.randn(n, device=device, dtype=dtype)
        ys = [th.randn(n, device=device, dtype=dtype) for _ in range(n_ys)]
        ys = _safe_ys(ys, ops_chain)

        out_ref = ref_chain(x, ys, ops_chain)
        out_ker = chained_elem_wise(x, *ys, ops_chain=ops_chain)

        diff = (out_ref - out_ker).abs().max().item()
        denom = out_ref.abs().max().item() + 1e-8
        rel = diff / denom
        ok = rel < 1e-5 or diff < 1e-6
        max_err_global = max(max_err_global, rel)

        print(f"[{'OK ' if ok else 'FAIL'}] "
              f"{ops_chain:25s} max_abs={diff:.3e}  rel={rel:.3e}")
        assert ok, f"Correctness failed for {ops_chain}"

    print(f"\nMax relative error over all cases: {max_err_global:.3e}")


def test_edge_sizes(device="cuda", dtype=th.float32):
    print("=" * 78)
    print("Edge-case sizes")
    print("=" * 78)

    sizes = [1, 2, 3, 1023, 1024, 1025, 4095, 4096, 4097, 100_000, 1_000_003]
    ops_chain = "s+m"                     # 需要 2 个 y
    for n in sizes:
        x  = th.randn(n, device=device, dtype=dtype)
        y1 = th.randn(n, device=device, dtype=dtype)
        y2 = th.randn(n, device=device, dtype=dtype)

        out_ref = ref_chain(x, [y1, y2], ops_chain)
        out_ker = chained_elem_wise(x, y1, y2, ops_chain=ops_chain)

        diff = (out_ref - out_ker).abs().max().item()
        ok = diff < 1e-5
        print(f"[{'OK ' if ok else 'FAIL'}] n={n:9d}  diff={diff:.3e}")
        assert ok

def test_error_handling(device="cuda", dtype=th.float32):
    print("=" * 78)
    print("Error handling")
    print("=" * 78)

    x = th.randn(1024, device=device, dtype=dtype)
    y = th.randn(1024, device=device, dtype=dtype)
    y_wrong_shape = th.randn(512, device=device, dtype=dtype)
    y_wrong_dtype = th.randn(1024, device=device, dtype=th.float64)

    def expect_error(fn, name):
        try:
            fn()
        except Exception as e:
            print(f"[OK ] {name:35s} -> {type(e).__name__}: {e}")
            return True
        print(f"[FAIL] {name:35s} -> no error raised")
        return False

    results = [
        expect_error(
            lambda: chained_elem_wise(x, ops_chain="a"),
            "missing binary operand"),
        expect_error(
            lambda: chained_elem_wise(x, y, ops_chain="neg"),
            "extra operand"),
        expect_error(
            lambda: chained_elem_wise(x, y_wrong_shape, ops_chain="a"),
            "shape mismatch"),
        expect_error(
            lambda: chained_elem_wise(x, y_wrong_dtype, ops_chain="a"),
            "dtype mismatch"),
        expect_error(
            lambda: chained_elem_wise(x, y, ops_chain="not_an_op"),
            "unknown op"),
        expect_error(
            lambda: chained_elem_wise(x, y, ops_chain="a*0"),
            "repeat count 0"),
        expect_error(
            lambda: chained_elem_wise(x, y, ops_chain="a*2"),
            "repeat count vs operand mismatch"),
    ]
    assert all(results), "Some error-handling checks failed"

def _bench(fn, warmup=5, rep=50):
    """用 CUDA event 计时，返回平均毫秒数。"""
    for _ in range(warmup):
        fn()
    th.cuda.synchronize()

    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    t_s = time.perf_counter()
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    th.cuda.synchronize()
    trad_time = time.perf_counter() - t_s
    return start.elapsed_time(end) / rep, trad_time


def test_performance(device="cuda", dtype=th.float32):
    for N in range(5):
        print("=" * 78)
        print("Performance: pure torch chain vs fused triton kernel")
        print(f"TIME {N+1}")
        print("=" * 78)

        cases = [
            # (ops_chain, n_ys)
            ("neg",          0),
            ("s",            1),
            ("s+d+m",        3),
            ("s*4+d*2+sin",  6),
            ("a*4+m*4+s*2", 10),
            ("s*8+d*4+m*4+exp", 16),
        ]
        sizes = [10000, 16_000_000]
        compiled_ref = th.compile(ref_chain, dynamic=True)

        header = f"{'ops_chain':22s} {'N':>12s} " \
                 f"{'torch(ms)':>12s} {'triton(ms)':>12s} {'speedup-eag':>10s} {'speedup-com':>10s}"
        print(header)
        print("-" * len(header))

        for n in sizes:
            for ops_chain, n_ys in cases:
                x = th.randn(n, device=device, dtype=dtype)
                ys = [th.randn(n, device=device, dtype=dtype) for _ in range(n_ys)]
                ys = _safe_ys(ys, ops_chain)

                # 保证两者第一次都在计时前完成 JIT / 分配
                ref_chain(x, ys, ops_chain)
                chained_elem_wise(x, *ys, ops_chain=ops_chain)
                compiled_ref(x, ys, ops_chain=ops_chain)
                th.cuda.synchronize()

                t_ref, t_ref2 = _bench(lambda: ref_chain(x, ys, ops_chain))
                t_ker, t_ker2 = _bench(lambda: chained_elem_wise(x, *ys, ops_chain=ops_chain, block_size=64))
                t_com, t_com2 = _bench(lambda: compiled_ref(x, ys, ops_chain=ops_chain))
                speedup = t_ref / t_ker if t_ker > 0 else float("inf")
                speedup_c = t_com / t_ker if t_ker > 0 else float("inf")
                #speedup2 = t_ref2 / t_ker2 if t_ker2 > 0 else float("inf")

                print(f"{ops_chain:22s} {n:>12d} "
                      f"{t_ref:>12.4f} {t_ker:>12.4f} {speedup:>9.2f}x {speedup_c:>9.2f}x")



if __name__ == "__main__":
    purge_all_kernels()
    if not th.cuda.is_available():
        print("CUDA is required for Triton. Skipping all tests.")
        raise SystemExit

    th.manual_seed(0)
    device = "cuda"
    dtype = th.float32

    test_correctness(device, dtype)
    print()
    test_edge_sizes(device, dtype)
    print()
    test_error_handling(device, dtype)
    print()
    test_performance(device, dtype)

