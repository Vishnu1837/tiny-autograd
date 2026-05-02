"""Tiny neural-net library on top of engine.Value.

Mirrors the PyTorch layout (Module / parameters / zero_grad) so the mental
model carries over.
"""

import random
from engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin="tanh"):
        # Small init keeps tanh away from saturation at start.
        self.w = [Value(random.uniform(-1, 1) * (n_in ** -0.5)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        return act  # linear

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in, n_out, nonlin="tanh"):
        self.neurons = [Neuron(n_in, nonlin=nonlin) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in, layer_sizes, hidden_nonlin="tanh"):
        sizes = [n_in] + list(layer_sizes)
        self.layers = []
        for i in range(len(layer_sizes)):
            is_last = (i == len(layer_sizes) - 1)
            self.layers.append(Layer(
                sizes[i], sizes[i + 1],
                nonlin="linear" if is_last else hidden_nonlin,
            ))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:
    def __init__(self, params, lr=0.05, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._v = [0.0] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            self._v[i] = self.momentum * self._v[i] - self.lr * p.grad
            p.data += self._v[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0
