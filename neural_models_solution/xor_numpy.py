"""
XOR / sensor-disagreement neural model -- dependency-free NumPy version.
Neural Models Laboratory (Tasks 1-5).

PyTorch was not available in the environment used to run these experiments, so
this NumPy implementation reproduces EXACTLY the same model, loss, and tests as
`xor_torch.py` and is used to generate the recorded numbers.  The mathematics
(forward pass, BCE/softmax loss, backpropagation) is written out explicitly so
every step of Task 4 can be inspected.

Network:  2 inputs -> 2 hidden (nonlinear) -> output.
Binary task : 1 logit + sigmoid + binary cross-entropy.
3-class task : 3 logits + softmax + cross-entropy (Task 5).
"""

import numpy as np


# --- data ------------------------------------------------------------------
def xor_data():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])          # sensor-disagreement = XOR
    return X, y


# --- activations and their derivatives (w.r.t. pre-activation a) -----------
def act_fn(name, a):
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-a))
    if name == "tanh":
        return np.tanh(a)
    if name == "relu":
        return np.maximum(0.0, a)
    raise ValueError(name)


def act_grad(name, a, h):
    if name == "sigmoid":
        return h * (1.0 - h)
    if name == "tanh":
        return 1.0 - h * h
    if name == "relu":
        return (a > 0).astype(float)
    raise ValueError(name)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_with_logits(z, y):
    """Numerically-stable binary cross-entropy from logits (mean over examples)."""
    # max(z,0) - z*y + log(1 + exp(-|z|))
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))


# ---------------------------------------------------------------------------
# Binary 2-2-1 network with explicit backprop
# ---------------------------------------------------------------------------
def train_binary(activation="sigmoid", zero_init=False, steps=4000,
                 lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X, y = xor_data()
    N = X.shape[0]

    if zero_init:                                   # Task 4 Part C
        W1 = np.zeros((2, 2)); b1 = np.zeros((1, 2))
        W2 = np.zeros((2, 1)); b2 = np.zeros((1, 1))
    else:
        W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros((1, 2))
        W2 = rng.normal(0, 1, (2, 1)); b2 = np.zeros((1, 1))

    # initial loss
    a1 = X @ W1 + b1; h1 = act_fn(activation, a1); z = h1 @ W2 + b2
    initial_loss = bce_with_logits(z, y)
    early_grad_norm = None

    for step in range(steps):
        # ---- forward ----
        a1 = X @ W1 + b1
        h1 = act_fn(activation, a1)
        z = h1 @ W2 + b2                             # logit

        # ---- backward (reverse-mode / backpropagation) ----
        dz = (sigmoid(z) - y) / N                    # dL/dz for mean BCE = (p - y)/N
        dW2 = h1.T @ dz
        db2 = dz.sum(axis=0, keepdims=True)
        dh1 = dz @ W2.T
        da1 = dh1 * act_grad(activation, a1, h1)
        dW1 = X.T @ da1                              # dL/dW1
        db1 = da1.sum(axis=0, keepdims=True)

        if step == 5:
            early_grad_norm = np.linalg.norm(dW1)

        # ---- gradient-descent update ----
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    # final evaluation
    a1 = X @ W1 + b1; h1 = act_fn(activation, a1); z = h1 @ W2 + b2
    probs = sigmoid(z).ravel()
    preds = (probs >= 0.5).astype(int)
    targets = y.ravel().astype(int)
    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(bce_with_logits(z, y)),
        "probs": probs.tolist(),
        "preds": preds.tolist(),
        "targets": targets.tolist(),
        "correct": bool((preds == targets).all()),
        "early_grad_norm": float(early_grad_norm),
        "W1": W1.tolist(),
        "W1_grad": dW1.tolist(),
    }


# ---------------------------------------------------------------------------
# Task 5: three-class softmax extension (2 inputs -> 2 hidden -> 3 logits)
# ---------------------------------------------------------------------------
def three_class_data():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    # class 0: both inactive (0,0); class 1: disagree (0,1),(1,0); class 2: both active (1,1)
    cls = np.array([0, 1, 1, 2])
    Y = np.eye(3)[cls]                              # one-hot targets
    return X, Y, cls


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)           # stable: subtract row max
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_three_class(activation="tanh", steps=6000, lr=0.3, seed=1):
    rng = np.random.default_rng(seed)
    X, Y, cls = three_class_data()
    N = X.shape[0]
    W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros((1, 2))
    W2 = rng.normal(0, 1, (2, 3)); b2 = np.zeros((1, 3))   # 3 logits per example

    for _ in range(steps):
        a1 = X @ W1 + b1
        h1 = act_fn(activation, a1)
        z = h1 @ W2 + b2                            # (N, 3) logits
        p = softmax(z)

        dz = (p - Y) / N                            # logit gradient = p - y
        dW2 = h1.T @ dz; db2 = dz.sum(0, keepdims=True)
        dh1 = dz @ W2.T
        da1 = dh1 * act_grad(activation, a1, h1)
        dW1 = X.T @ da1; db1 = da1.sum(0, keepdims=True)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    a1 = X @ W1 + b1; h1 = act_fn(activation, a1); z = h1 @ W2 + b2
    p = softmax(z)
    preds = p.argmax(1)
    return {
        "probs": p, "preds": preds.tolist(), "targets": cls.tolist(),
        "W2_shape": W2.shape, "logits_example0": z[0].tolist(),
        "correct": bool((preds == cls).all()),
    }


# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Task 4 Part A/B -- basic learning and backpropagation check")
    print("=" * 64)
    r = train_binary("sigmoid")
    print(f"initial loss : {r['initial_loss']:.4f}")
    print(f"final loss   : {r['final_loss']:.4f}")
    print(f"probabilities: {[round(x, 3) for x in r['probs']]}")
    print(f"predictions  : {r['preds']}   target: {r['targets']}")
    print(f"all 4 correct: {r['correct']}")
    print(f"dL/dW1 (first-layer gradient, non-zero => learning signal):")
    for row in r["W1_grad"]:
        print("   ", [f"{v:+.4f}" for v in row])

    print("\n" + "=" * 64)
    print("Task 4 Part D -- activation experiment (random init, 3 runs)")
    print("=" * 64)
    print(f"{'activation':<12}{'final loss':<14}{'4/4 correct?':<14}{'early |grad W1|'}")
    for act in ["sigmoid", "tanh", "relu"]:
        r = train_binary(act)
        print(f"{act:<12}{r['final_loss']:<14.4f}{str(r['correct']):<14}"
              f"{r['early_grad_norm']:.4f}")

    print("\n" + "=" * 64)
    print("Task 4 Part C -- symmetry experiment (all weights = 0)")
    print("=" * 64)
    r = train_binary("sigmoid", zero_init=True)
    print(f"final loss   : {r['final_loss']:.4f}   all correct: {r['correct']}")
    print(f"hidden W1 rows after training:")
    for row in r["W1"]:
        print("   ", [f"{v:+.6f}" for v in row])
    rows_equal = np.allclose(r["W1"][0], r["W1"][1])
    print(f"the two hidden rows are identical: {rows_equal}")
    print("=> identical units compute the same thing & get the same gradient,")
    print("   so they never differentiate and XOR is not learned.")

    print("\n" + "=" * 64)
    print("Task 5 -- three-class softmax extension")
    print("=" * 64)
    r = train_three_class()
    print(f"output weight matrix W2 shape : {r['W2_shape']}  (2 hidden -> 3 logits)")
    print(f"logits per example            : 3")
    print(f"predictions                   : {r['preds']}   target: {r['targets']}")
    print(f"all correct                   : {r['correct']}")
    print("class probabilities per input (rows sum to 1):")
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for inp, row in zip(inputs, r["probs"]):
        print(f"   x={inp}: {[round(float(v), 3) for v in row]}  sum={float(row.sum()):.4f}")
    # Optional diagnostic: adding a constant to all logits leaves softmax unchanged
    z0 = np.array(r["logits_example0"])
    p_a = softmax(z0[None, :])[0]
    p_b = softmax((z0 + 100.0)[None, :])[0]
    print(f"\nsoftmax(logits)         = {[round(float(v),4) for v in p_a]}")
    print(f"softmax(logits + 100)   = {[round(float(v),4) for v in p_b]}")
    print(f"unchanged (invariant to additive constant): {np.allclose(p_a, p_b)}")
    print("=> this shift-invariance is why stable softmax subtracts max(logits).")


if __name__ == "__main__":
    main()
