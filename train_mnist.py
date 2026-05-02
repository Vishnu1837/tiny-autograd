"""Train an MLP on MNIST using our tensor autograd.

This is the proof: if forward/backward/optimizer are all correct, this hits
~97% test accuracy in a couple of epochs. Run it to train the model and
generate loss_curve.png / accuracy_curve.png for the README.
"""

import os
import time
import urllib.request
import numpy as np

from tensor import Tensor
from tensor_nn import Sequential, Linear, ReLU, Adam


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE = os.path.join(os.path.dirname(__file__), "mnist.npz")


def load_mnist():
    if not os.path.exists(CACHE):
        print(f"downloading mnist -> {CACHE}")
        urllib.request.urlretrieve(MNIST_URL, CACHE)
    with np.load(CACHE) as f:
        x_tr, y_tr = f["x_train"], f["y_train"]
        x_te, y_te = f["x_test"], f["y_test"]
    # Flatten 28x28 -> 784, scale to [0, 1].
    x_tr = x_tr.reshape(-1, 784).astype(np.float32) / 255.0
    x_te = x_te.reshape(-1, 784).astype(np.float32) / 255.0
    return (x_tr, y_tr.astype(np.int64)), (x_te, y_te.astype(np.int64))


def evaluate(model, x, y, batch_size=512):
    correct = 0
    for i in range(0, len(x), batch_size):
        xb = Tensor(x[i:i + batch_size])
        logits = model(xb).data
        correct += (logits.argmax(axis=1) == y[i:i + batch_size]).sum()
    return correct / len(x)


def plot_curves(batch_losses, epoch_train_accs, epoch_test_accs):
    """Generate publication-quality loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping curve plots.")
        return

    fig_dir = os.path.dirname(__file__)

    # --- smoothed loss curve ---
    plt.figure(figsize=(8, 4))
    plt.plot(batch_losses, alpha=0.15, color="#6366f1", linewidth=0.5)
    # running average
    window = max(1, len(batch_losses) // 50)
    if window > 1:
        smoothed = np.convolve(batch_losses, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, window - 1 + len(smoothed)), smoothed,
                 color="#6366f1", linewidth=2, label="smoothed")
    plt.xlabel("Batch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss — tiny-autograd MNIST")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print("saved loss_curve.png")

    # --- accuracy curve ---
    epochs = list(range(1, len(epoch_train_accs) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [a * 100 for a in epoch_train_accs],
             "o-", color="#6366f1", linewidth=2, label="train")
    plt.plot(epochs, [a * 100 for a in epoch_test_accs],
             "s--", color="#f43f5e", linewidth=2, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy — tiny-autograd MNIST")
    plt.ylim(80, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "accuracy_curve.png"), dpi=150)
    plt.close()
    print("saved accuracy_curve.png")


def main():
    np.random.seed(42)
    (x_tr, y_tr), (x_te, y_te) = load_mnist()
    print(f"train: {x_tr.shape}  test: {x_te.shape}")

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )
    n_params = sum(p.data.size for p in model.parameters())
    print(f"params: {n_params:,}")

    opt = Adam(model.parameters(), lr=1e-3)
    batch_size = 128
    n_epochs = 5
    n_train = len(x_tr)

    batch_losses = []
    epoch_train_accs = []
    epoch_test_accs = []

    for epoch in range(n_epochs):
        t0 = time.time()
        idx = np.random.permutation(n_train)
        running_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            sel = idx[start:start + batch_size]
            xb = Tensor(x_tr[sel])
            yb = y_tr[sel]

            logits = model(xb)
            loss = logits.softmax_cross_entropy(yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            lv = float(loss.data)
            running_loss += lv
            batch_losses.append(lv)
            n_batches += 1

        train_acc = evaluate(model, x_tr[:5000], y_tr[:5000])
        test_acc = evaluate(model, x_te, y_te)
        epoch_train_accs.append(train_acc)
        epoch_test_accs.append(test_acc)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{n_epochs}  "
              f"loss={running_loss/n_batches:.4f}  "
              f"train_acc={train_acc*100:.2f}%  "
              f"test_acc={test_acc*100:.2f}%  "
              f"({dt:.1f}s)")

    plot_curves(batch_losses, epoch_train_accs, epoch_test_accs)


if __name__ == "__main__":
    main()
