"""Gradient checks for every op in tensor.py.

Method: pick a random input x, define a scalar f(x) = (op(x) * random_weights).sum(),
compute analytic gradient via backward(), and compare to finite differences:

    df/dx_i  ≈  (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

Tolerance is loose (3e-3) because we run in float32. The point is to catch
incorrect-formula bugs and broadcasting bugs, not floating-point noise.

If you add a new op to Tensor, add a test here. Untested ops are dead ops.
"""

import numpy as np
from tensor import Tensor

RNG = np.random.default_rng(0)
EPS = 1e-3
TOL = 3e-3


def _numerical_grad(f, x: np.ndarray) -> np.ndarray:
    """Central-difference numerical gradient of scalar f(x) w.r.t. each entry of x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + EPS; f_plus = float(f(x))
        x[idx] = orig - EPS; f_minus = float(f(x))
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * EPS)
        it.iternext()
    return grad


def _check(name, build_fn, *input_shapes):
    """Generic gradient check. `build_fn(*tensors) -> scalar Tensor`."""
    global _TOTAL, _FAILURES
    _TOTAL += 1
    inputs_np = [RNG.standard_normal(s).astype(np.float32) for s in input_shapes]

    # Analytic gradient.
    inputs_t = [Tensor(arr.copy()) for arr in inputs_np]
    out = build_fn(*inputs_t)
    assert out.data.ndim == 0 or out.data.size == 1, f"{name}: f must return scalar"
    out.backward()
    ana = [t.grad for t in inputs_t]

    # Numerical gradient per input, holding others fixed.
    for i, arr in enumerate(inputs_np):
        def f(x, i=i):
            args = [Tensor(a if j != i else x) for j, a in enumerate(inputs_np)]
            return build_fn(*args).data
        num = _numerical_grad(f, arr.copy())
        diff = np.max(np.abs(ana[i] - num))
        rel = diff / (np.max(np.abs(num)) + 1e-8)
        ok = diff < TOL or rel < TOL
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}  input{i}: max|ana-num|={diff:.2e}  rel={rel:.2e}")
        if not ok:
            print(f"    ana[:5]={ana[i].flat[:5]}\n    num[:5]={num.flat[:5]}")
            _FAILURES += 1
            return False
    return True


_FAILURES = 0
_TOTAL = 0


def run():
    global _FAILURES, _TOTAL
    _FAILURES = 0
    _TOTAL = 0

    # Random projection turns vector/matrix output into a scalar so we have a
    # well-defined gradient to check. Different shapes per test as needed.
    def proj(t, w): return (t * Tensor(w)).sum()

    print("=== elementwise binary ===")
    w_33 = RNG.standard_normal((3, 3)).astype(np.float32)
    _check("add (same shape)",      lambda a, b: proj(a + b, w_33),                (3, 3), (3, 3))
    _check("sub (same shape)",      lambda a, b: proj(a - b, w_33),                (3, 3), (3, 3))
    _check("mul (same shape)",      lambda a, b: proj(a * b, w_33),                (3, 3), (3, 3))
    _check("div (same shape)",      lambda a, b: proj(a / (b + Tensor(2.0)), w_33),(3, 3), (3, 3))

    print("=== broadcasting (the hard part) ===")
    w_32_10 = RNG.standard_normal((32, 10)).astype(np.float32)
    _check("add bias  (32,10)+(10,)",   lambda a, b: proj(a + b, w_32_10), (32, 10), (10,))
    _check("mul bias  (32,10)*(10,)",   lambda a, b: proj(a * b, w_32_10), (32, 10), (10,))
    _check("add (3,1)+(1,3) -> (3,3)",  lambda a, b: proj(a + b, w_33),    (3, 1),  (1, 3))
    _check("scalar+tensor",             lambda a, b: proj(a + b, w_33),    (),      (3, 3))

    print("=== matmul ===")
    w_4_5 = RNG.standard_normal((4, 5)).astype(np.float32)
    _check("matmul (4,3)@(3,5)", lambda a, b: proj(a @ b, w_4_5), (4, 3), (3, 5))

    print("=== unary ===")
    _check("relu", lambda a: proj(a.relu(), w_33), (3, 3))
    _check("tanh", lambda a: proj(a.tanh(), w_33), (3, 3))
    _check("exp",  lambda a: proj(a.exp(),  w_33), (3, 3))
    _check("log",  lambda a: proj((a * a + Tensor(1.0)).log(), w_33), (3, 3))  # ensure positive
    _check("pow3", lambda a: proj(a ** 3, w_33), (3, 3))
    _check("neg",  lambda a: proj(-a, w_33), (3, 3))

    # NOTE: projection weights MUST be drawn outside the lambda. If we draw
    # them inside, every finite-difference call gets a different `w`, so f
    # becomes non-deterministic and the numerical gradient is garbage. (Found
    # the hard way -- six tests failed before this fix.)

    print("=== reductions ===")
    w_4    = RNG.standard_normal(4).astype(np.float32)
    w_3    = RNG.standard_normal(3).astype(np.float32)
    _check("sum all",                lambda a: a.sum(),                       (3, 4))
    _check("sum axis=0",             lambda a: proj(a.sum(axis=0),  w_4),     (3, 4))
    _check("sum axis=1 keepdims",    lambda a: a.sum(axis=1, keepdims=True).sum(), (3, 4))
    _check("mean all",               lambda a: a.mean(),                      (3, 4))
    _check("mean axis=0",            lambda a: proj(a.mean(axis=0), w_4),     (3, 4))
    _check("max all",                lambda a: a.max(),                       (3, 4))
    _check("max axis=1",             lambda a: proj(a.max(axis=1),  w_3),     (3, 4))

    print("=== shape ===")
    w_4_3   = RNG.standard_normal((4, 3)).astype(np.float32)
    w_4_2_3 = RNG.standard_normal((4, 2, 3)).astype(np.float32)
    _check("reshape",      lambda a: proj(a.reshape(4, 3),       w_4_3),   (3, 4))
    _check("transpose 2D", lambda a: proj(a.T,                   w_4_3),   (3, 4))
    _check("transpose 3D", lambda a: proj(a.transpose(2, 0, 1),  w_4_2_3), (2, 3, 4))

    print("=== indexing ===")
    w_2_4 = RNG.standard_normal((2, 4)).astype(np.float32)
    _check("getitem int",   lambda a: proj(a[1],   w_4),   (3, 4))
    _check("getitem slice", lambda a: proj(a[1:3], w_2_4), (4, 4))

    print("=== losses ===")
    # softmax_cross_entropy: grad w.r.t. logits only.
    targets = np.array([2, 0, 1, 3], dtype=np.int64)
    def xent_fn(logits): return logits.softmax_cross_entropy(targets)
    _check("softmax_cross_entropy (4,5) labels", xent_fn, (4, 5))

    passed = _TOTAL - _FAILURES
    print(f"\n{passed}/{_TOTAL} checks passed.")
    return _FAILURES == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
