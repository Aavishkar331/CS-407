"""
XOR / sensor-disagreement neural model -- PyTorch reference implementation.
Neural Models Laboratory (Tasks 2-4).

This is the implementation the lab asks you to submit (PyTorch, 2-2-1,
BCEWithLogitsLoss).  It mirrors `xor_numpy.py`, which is a dependency-free
version used to produce the recorded numbers when PyTorch is not installed.

Network:  2 inputs -> 2 hidden (nonlinear) -> 1 logit -> sigmoid.
Loss   :  BCEWithLogitsLoss (sigmoid + binary cross-entropy, numerically stable).
"""

import torch
import torch.nn as nn


def make_data():
    # The four XOR (sensor-disagreement) examples.
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return X, y


class XORNet(nn.Module):
    def __init__(self, activation="sigmoid", zero_init=False):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)     # first affine layer  a = W1 h0 + b1
        self.fc2 = nn.Linear(2, 1)     # output logit        z = W2 h1 + b2
        self.act = {"sigmoid": torch.sigmoid,
                    "tanh": torch.tanh,
                    "relu": torch.relu}[activation]
        if zero_init:                  # Task 4 Part C: symmetry experiment
            for p in self.parameters():
                nn.init.zeros_(p)

    def forward(self, x):
        h1 = self.act(self.fc1(x))     # nonlinear hidden layer
        return self.fc2(h1)            # raw logit (no sigmoid: loss adds it)


def train(activation="sigmoid", zero_init=False, steps=4000, lr=0.5, seed=0):
    torch.manual_seed(seed)            # reproducibility
    X, y = make_data()
    net = XORNet(activation, zero_init)
    loss_fn = nn.BCEWithLogitsLoss()   # sigmoid + BCE in one stable op
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    logits = net(X)
    initial_loss = loss_fn(logits, y).item()
    early_grad_norm = None

    for step in range(steps):
        opt.zero_grad()
        logits = net(X)
        loss = loss_fn(logits, y)
        loss.backward()                # reverse-mode AD == backpropagation
        if step == 5:                  # record an early first-layer grad norm
            early_grad_norm = net.fc1.weight.grad.norm().item()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(net(X)).squeeze()
        preds = (probs >= 0.5).int()
    return {
        "initial_loss": initial_loss,
        "final_loss": loss_fn(net(X), y).item(),
        "probs": probs.tolist(),
        "preds": preds.tolist(),
        "correct": (preds == y.squeeze().int()).all().item(),
        "early_grad_norm": early_grad_norm,
        "W1": net.fc1.weight.detach().tolist(),
        "W1_grad": net.fc1.weight.grad.detach().tolist(),
    }


if __name__ == "__main__":
    # Task 4 Part A/B -- basic learning + gradient
    r = train("sigmoid")
    print("== Binary XOR, sigmoid hidden, BCEWithLogitsLoss ==")
    print(f"initial loss : {r['initial_loss']:.4f}")
    print(f"final loss   : {r['final_loss']:.4f}")
    print(f"probabilities: {[round(p, 3) for p in r['probs']]}")
    print(f"predictions  : {r['preds']}  (target 0,1,1,0)")
    print(f"all correct  : {r['correct']}")
    print(f"dL/dW1 (first-layer gradient):\n{r['W1_grad']}")

    # Task 4 Part D -- activation experiment
    print("\n== Activation experiment ==")
    print(f"{'activation':<10}{'final loss':<14}{'4/4?':<8}{'early |grad W1|'}")
    for act in ["sigmoid", "tanh", "relu"]:
        r = train(act)
        print(f"{act:<10}{r['final_loss']:<14.4f}{str(bool(r['correct'])):<8}"
              f"{r['early_grad_norm']:.4f}")

    # Task 4 Part C -- symmetry experiment
    print("\n== Symmetry experiment (all weights initialised to zero) ==")
    r = train("sigmoid", zero_init=True)
    print(f"final loss   : {r['final_loss']:.4f}  all correct: {r['correct']}")
    print(f"hidden W1 rows after training: {r['W1']}")
    print("(identical rows => the two hidden units never differentiate)")
