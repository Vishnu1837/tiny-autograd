"""Tensor-valued reverse-mode autograd, built on numpy.

A Tensor wraps an ndarray plus a gradient slot. Forward ops build a DAG;
backward() walks it in reverse-topological order, applying each op's local
Jacobian via a registered closure.

The most error-prone piece is un-broadcasting. Numpy happily broadcasts a (10,)
operand against a (32, 10) operand in the forward pass. The backward pass has
to put gradients back where they belong -- which means summing the incoming
gradient along the dims that got broadcast, before accumulating into the
smaller tensor's grad. `_unbroadcast` does exactly that and every binary op
routes its parent gradients through it.
"""

from __future__ import annotations
import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` from a broadcasted shape back down to `shape` by summing.

    Two distinct cases:
      (1) `grad` has extra leading dims that didn't exist in `shape`. These came
          from broadcasting (e.g. (10,) -> (32, 10)). Sum them away.
      (2) Same ndim, but some axes are larger than `shape` because that axis
          was 1 in the original (e.g. (1, 10) -> (32, 10)). Sum with keepdims.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "requires_grad")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        self.data = arr
        self.grad = None
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op
        self.requires_grad = requires_grad

    # --- properties --------------------------------------------------------

    @property
    def shape(self): return self.data.shape

    @property
    def ndim(self): return self.data.ndim

    @property
    def size(self): return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'})"

    # --- internal ----------------------------------------------------------

    def _accum(self, g: np.ndarray):
        # Always reduce to self.shape -- callers may pass a broadcasted grad.
        if g.shape != self.data.shape:
            g = _unbroadcast(g, self.data.shape)
        if self.grad is None:
            self.grad = g.astype(np.float32, copy=True)
        else:
            self.grad += g.astype(np.float32, copy=False)

    def zero_grad(self):
        self.grad = None

    # --- elementwise binary ------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self._accum(out.grad)
            other._accum(out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self._accum(other.data * out.grad)
            other._accum(self.data * out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), "-")

        def _backward():
            self._accum(out.grad)
            other._accum(-out.grad)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            self._accum(out.grad / other.data)
            other._accum(-out.grad * self.data / (other.data ** 2))
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        # Scalar exponent only -- tensor-tensor pow is rarely needed.
        assert isinstance(exponent, (int, float)), "tensor**scalar only"
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self._accum(exponent * (self.data ** (exponent - 1)) * out.grad)
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return Tensor(other) - self
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return Tensor(other) / self

    # --- matmul ------------------------------------------------------------

    def __matmul__(self, other):
        # 2D only (batch, features) @ (features, out). Sufficient for MLPs.
        assert self.ndim == 2 and other.ndim == 2, "matmul: 2D only"
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self._accum(out.grad @ other.data.T)
            other._accum(self.data.T @ out.grad)
        out._backward = _backward
        return out

    # --- unary -------------------------------------------------------------

    def relu(self):
        mask = (self.data > 0).astype(np.float32)
        out = Tensor(self.data * mask, (self,), "relu")

        def _backward():
            self._accum(mask * out.grad)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self._accum((1 - t * t) * out.grad)
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self._accum(e * out.grad)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self._accum(out.grad / self.data)
        out._backward = _backward
        return out

    # --- reductions --------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            # If we collapsed dims (keepdims=False), restore them so broadcast works.
            if axis is not None and not keepdims:
                ax = (axis,) if isinstance(axis, int) else tuple(axis)
                # Normalize negative axes against the *original* ndim.
                ax = tuple(a if a >= 0 else a + self.ndim for a in ax)
                for a in sorted(ax):
                    g = np.expand_dims(g, a)
            elif axis is None and not keepdims:
                # Reduced to scalar; reshape so broadcast back to full shape works.
                g = np.reshape(g, (1,) * self.ndim)
            self._accum(np.broadcast_to(g, self.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        # mean = sum / N, where N is the count of elements collapsed.
        if axis is None:
            n = self.data.size
        else:
            ax = (axis,) if isinstance(axis, int) else tuple(axis)
            n = 1
            for a in ax:
                n *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims=False):
        # Routes gradient to the argmax position(s). Ties split evenly.
        m = self.data.max(axis=axis, keepdims=True)
        out_data = m if keepdims else self.data.max(axis=axis, keepdims=False)
        out = Tensor(out_data, (self,), "max")

        def _backward():
            mask = (self.data == m).astype(np.float32)
            mask /= mask.sum(axis=axis, keepdims=True)  # split ties
            g = out.grad
            if axis is not None and not keepdims:
                ax = (axis,) if isinstance(axis, int) else tuple(axis)
                ax = tuple(a if a >= 0 else a + self.ndim for a in ax)
                for a in sorted(ax):
                    g = np.expand_dims(g, a)
            elif axis is None and not keepdims:
                g = np.reshape(g, (1,) * self.ndim)
            self._accum(mask * g)
        out._backward = _backward
        return out

    # --- shape -------------------------------------------------------------

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original_shape = self.shape
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self._accum(out.grad.reshape(original_shape))
        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        # Inverse permutation for backward.
        inv = tuple(np.argsort(axes))
        out = Tensor(self.data.transpose(axes), (self,), "transpose")

        def _backward():
            self._accum(out.grad.transpose(inv))
        out._backward = _backward
        return out

    @property
    def T(self):
        assert self.ndim == 2, "T: 2D only; use transpose(axes) for >2D"
        return self.transpose(1, 0)

    # --- indexing ----------------------------------------------------------

    def __getitem__(self, key):
        # Supports integer/array/slice indexing. Backward scatters via np.add.at.
        out = Tensor(self.data[key], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, key, out.grad)
            self._accum(g)
        out._backward = _backward
        return out

    # --- losses (fused for stability) --------------------------------------

    def softmax_cross_entropy(self, targets):
        """Fused softmax + NLL. self is logits (N, C); targets is int labels (N,).

        Fused because the unfused gradient `(softmax - one_hot)/N` is much more
        stable than autograd-ing through log(softmax(...)) where exp can blow up.
        """
        logits = self.data
        # log-sum-exp trick for numerical stability.
        m = logits.max(axis=1, keepdims=True)
        shifted = logits - m
        exp = np.exp(shifted)
        sm = exp / exp.sum(axis=1, keepdims=True)         # softmax probs
        log_sm = shifted - np.log(exp.sum(axis=1, keepdims=True))
        N = logits.shape[0]
        targets_arr = np.asarray(targets, dtype=np.int64)
        loss_val = -log_sm[np.arange(N), targets_arr].mean()
        out = Tensor(np.float32(loss_val), (self,), "xent")

        def _backward():
            grad_logits = sm.copy()
            grad_logits[np.arange(N), targets_arr] -= 1.0
            grad_logits /= N
            self._accum(grad_logits * out.grad)
        out._backward = _backward
        return out

    # --- backward ----------------------------------------------------------

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        # Seed gradient at the output. Loss is a scalar, so this is just 1.
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
