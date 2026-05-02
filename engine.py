"""Scalar-valued reverse-mode autograd.

Each Value is a node in a DAG. Forward ops build the graph; backward() does a
reverse topological sweep, accumulating gradients via the chain rule.
"""

from __future__ import annotations
import math


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # --- ops ---------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only scalar powers"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # --- syntactic sugar ---------------------------------------------------

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else (1.0 / other))
    def __rtruediv__(self, other): return other * (self ** -1)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- backward ----------------------------------------------------------

    def backward(self):
        # Build reverse-topological order, then chain-rule from output to leaves.
        topo, visited = [], set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


if __name__ == "__main__":
    # Sanity check: f(x,y) = (x*y + x**2).tanh() against a numerical gradient.
    def f(x, y):
        return (x * y + x ** 2).tanh()

    x = Value(0.7); y = Value(-1.3)
    out = f(x, y); out.backward()

    def f_raw(a, b): return math.tanh(a * b + a ** 2)
    eps = 1e-6
    nx = (f_raw(0.7 + eps, -1.3) - f_raw(0.7 - eps, -1.3)) / (2 * eps)
    ny = (f_raw(0.7, -1.3 + eps) - f_raw(0.7, -1.3 - eps)) / (2 * eps)
    print(f"out = {out.data:.6f}")
    print(f"analytic dx={x.grad:+.6f}  numeric dx={nx:+.6f}")
    print(f"analytic dy={y.grad:+.6f}  numeric dy={ny:+.6f}")
