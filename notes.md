# Notes — Building tiny-autograd

Stream-of-consciousness notes from building this project. Not polished — these are the things I'd tell someone who's about to attempt the same thing.

---

## Starting Point: Scalar Autograd

I started with [Karpathy's micrograd](https://github.com/karpathy/micrograd) as inspiration but wrote the code from scratch. The scalar `Value` class is ~130 lines and genuinely beautiful in its simplicity:

- Each `Value` stores a float, a gradient, and a closure (`_backward`) that knows how to propagate gradients to its parents.
- `backward()` does a reverse topological sort and calls each closure once.
- That's it. The entire chain rule in ~15 lines of `backward()`.

The scalar version trained a 2-layer MLP on a spiral classification dataset. It worked but was absurdly slow — hundreds of times slower than the equivalent NumPy code — because every single weight, bias, activation, and intermediate result is its own Python object.

## The Tensor Rewrite

The tensor engine uses the same pattern (DAG + closures + reverse topo sort) but wraps NumPy arrays instead of floats. In theory, a simple lift. In practice, there were two hard parts:

### 1. Un-Broadcasting

This was the single hardest thing in the project. When NumPy computes `A + B` where `A` is `(32, 10)` and `B` is `(10,)`, it silently broadcasts `B` to `(32, 10)`. The forward pass just works. But in the backward pass:

- The gradient arriving is `(32, 10)` (same shape as the output).
- The gradient for `B` needs to be `(10,)`.
- So you sum along axis 0.

This seems simple until you hit edge cases:
- Scalar + tensor: sum everything.
- `(1, 10) + (32, 1)` → `(32, 10)`: sum axis 0 for the first operand, axis 1 for the second, both with `keepdims=True`.
- Higher-rank broadcasting with implicit leading 1s.

I wrote `_unbroadcast(grad, target_shape)` and every binary op routes through it. Took three attempts to get right. The gradient test in `test_tensor.py` caught every bug.

### 2. Reduction Backwards

`sum(axis=1, keepdims=False)` collapses a dimension. The incoming gradient has fewer dimensions than the input. You need to `expand_dims` the gradient back to the right shape before broadcasting. Same for `mean` and `max`.

`max` is additionally tricky because you need to route the gradient only to the argmax position(s), and handle ties by splitting evenly.

## Debugging Strategy: Numerical Gradient Checks

Early on I decided that every op would get a finite-difference check before I moved on. This saved me *hours* of debugging later. The pattern:

```python
# For each op: define f(x) -> scalar, compute analytic grad, compare to:
# (f(x + eps) - f(x - eps)) / (2 * eps)
```

Critical gotcha I hit: the random projection weights (used to turn a matrix output into a scalar for the gradient check) **must** be drawn outside the lambda. If you draw them inside, each finite-difference evaluation gets different weights, `f` is non-deterministic, and the numerical gradient is garbage. Six tests failed before I figured this out.

## MNIST: Where Everything Has to Be Right Simultaneously

Getting to 97% on MNIST required getting four things right at the same time:

1. **Kaiming init** — Without it, the network stalls at ~85%. With `W ~ N(0, sqrt(2/fan_in))`, pre-activations stay at unit variance through the network, and gradients are well-conditioned from the start.

2. **Fused softmax-cross-entropy** — Separate `softmax` → `log` → `nll` works on small inputs but produces NaN on real data because `exp(logits)` overflows float32. The fused version subtracts `max(logits)` first.

3. **Adam optimizer** — SGD works but needs careful learning rate tuning. Adam with default hyperparams (lr=1e-3, betas=(0.9, 0.999)) just works. The bias correction in the first few steps matters more than I expected.

4. **Correct un-broadcasting everywhere** — The `Linear` layer computes `x @ W + b` where `x` is `(batch, in)`, `W` is `(in, out)`, and `b` is `(out,)`. The `+ b` broadcasts, so the backward pass through `+` must un-broadcast the `b` gradient from `(batch, out)` to `(out,)`.

## What I'd Do Differently

- **Start with tensors.** The scalar engine was a great learning exercise but I should have moved to tensors sooner. The scalar version can't train anything useful.
- **Write gradient tests first.** I sometimes wrote the forward pass, tested it manually, then added the backward pass and gradient check. Should have written the gradient check first and let it guide the implementation.
- **Add `Conv2d`.** The MLP flattens images, losing spatial structure. A small conv net would get to 99%+ and demonstrate that the autograd handles more complex operations. Left as future work.

## Performance Profile

On my machine (CPU only):
- 5 epochs of MNIST: ~6 seconds total
- ~469 batches/epoch × 5 epochs = ~2,345 forward/backward passes
- Each pass: build graph → forward through 5 layers → cross-entropy → backward → Adam step
- Bottleneck is Python overhead in the backward pass (calling ~15 closures per batch). The actual NumPy matmuls are fast.

PyTorch on the same architecture does the same job in ~3 seconds. The 2× slowdown is the cost of doing everything in pure Python with no graph optimization, no kernel fusion, and no compiled backward ops.
