"""Train a tiny MLP on a 2D classification problem (interleaving spirals).

No external deps -- just stdlib. Demonstrates that the autograd engine in
engine.py actually learns a non-linear decision boundary.
"""

import math
import random
from engine import Value
from nn import MLP, SGD


def make_spirals(n_per_class=60, noise=0.1, twists=0.75, seed=0):
    """Two interleaving spirals -- classic 'needs nonlinearity' dataset.

    `twists` controls how many full rotations each arm sweeps. Tighter spirals
    (>=1.0) are great visualisations but punishing for a 337-param scalar net.
    """
    rng = random.Random(seed)
    xs, ys = [], []
    for cls in (0, 1):
        for i in range(n_per_class):
            r = 0.2 + 0.8 * i / n_per_class
            t = twists * 2 * math.pi * i / n_per_class + cls * math.pi
            x = r * math.sin(t) + rng.gauss(0, noise)
            y = r * math.cos(t) + rng.gauss(0, noise)
            xs.append([x, y])
            ys.append(1.0 if cls == 1 else -1.0)  # +/-1 for hinge / margin loss
    return xs, ys


def hinge_loss(model, xs, ys, l2=1e-4):
    """Mean SVM-style margin loss + L2. Per-sample: max(0, 1 - y*score)."""
    losses = []
    for xi, yi in zip(xs, ys):
        score = model([Value(xi[0]), Value(xi[1])])
        losses.append((1 + -yi * score).relu())
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = sum((p * p for p in model.parameters()), Value(0.0)) * l2
    total = data_loss + reg

    correct = sum(
        1 for xi, yi in zip(xs, ys)
        if (model([Value(xi[0]), Value(xi[1])]).data > 0) == (yi > 0)
    )
    return total, correct / len(xs)


def main():
    random.seed(42)
    xs, ys = make_spirals(n_per_class=50, noise=0.1)
    model = MLP(2, [16, 16, 1], hidden_nonlin="tanh")
    print(f"params: {len(model.parameters())}")

    base_lr = 0.1
    opt = SGD(model.parameters(), lr=base_lr, momentum=0.9)
    n_epochs = 200

    for epoch in range(n_epochs):
        loss, acc = hinge_loss(model, xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        opt.lr = base_lr * (1 - 0.9 * epoch / n_epochs)
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:3d}  loss={loss.data:.4f}  acc={acc*100:5.1f}%  lr={opt.lr:.4f}")

    # ASCII decision-boundary plot.
    print("\nDecision boundary (+/- = predicted class, o/x = data):")
    grid_n = 30
    pts_by_cell = {}
    for xi, yi in zip(xs, ys):
        gx = int((xi[0] + 1.5) / 3.0 * grid_n)
        gy = int((xi[1] + 1.5) / 3.0 * grid_n)
        pts_by_cell[(gx, gy)] = "o" if yi > 0 else "x"
    for gy in range(grid_n - 1, -1, -1):
        row = ""
        for gx in range(grid_n):
            if (gx, gy) in pts_by_cell:
                row += pts_by_cell[(gx, gy)]
            else:
                wx = gx / grid_n * 3.0 - 1.5
                wy = gy / grid_n * 3.0 - 1.5
                s = model([Value(wx), Value(wy)]).data
                row += "+" if s > 0 else "-" if s < 0 else " "
        print(row)


if __name__ == "__main__":
    main()
