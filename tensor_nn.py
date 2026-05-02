"""Tensor-based neural net layers + optimizers.

Modeled on torch.nn so the API transfers. Linear / Sequential / Adam are enough
to train an MLP on MNIST.
"""

import numpy as np
from tensor import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, n_in, n_out):
        # Kaiming-uniform-ish init: keeps ReLU pre-activations roughly unit-variance
        # at every layer, which is the difference between a network that trains and
        # one that stalls at chance.
        bound = (2.0 / n_in) ** 0.5
        self.W = Tensor(np.random.randn(n_in, n_out).astype(np.float32) * bound)
        self.b = Tensor(np.zeros(n_out, dtype=np.float32))

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# --- optimizers -----------------------------------------------------------


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._v[i] = self.momentum * self._v[i] - self.lr * p.grad
            p.data += self._v[i]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class Adam:
    """Standard Adam. State per parameter: first/second moments + step counter."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
