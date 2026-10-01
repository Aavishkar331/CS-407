# Laboratory Report — Neural Models: Learning, Depth, Activations, Output Layers
### Artificial Intelligence · Neural Models Laboratory

**Files**

| File | Purpose |
|------|---------|
| `xor_torch.py` | PyTorch reference implementation (the deliverable the lab asks for) |
| `xor_numpy.py` | dependency-free NumPy version with explicit backprop (used to **run** the experiments) |
| `PROMPTS.md` | LLM prompt + corrections |

> **Note on environment.** PyTorch could not be installed in the machine used
> to run these experiments, so the recorded numbers come from `xor_numpy.py`,
> which implements the *identical* model, loss, and tests (2-2-1, sigmoid + BCE
> on logits, explicit backpropagation). `xor_torch.py` is the equivalent
> PyTorch code for submission. Run with `python3 xor_numpy.py`.

---

## Task 1 — Understand the problem before coding

- **Input space** `X = {0,1}²`; **output space** `Y = {0,1}`; the four labelled
  examples (sensor-disagreement = XOR):

  | x₁ | x₂ | y |
  |----|----|---|
  | 0 | 0 | 0 |
  | 0 | 1 | 1 |
  | 1 | 0 | 1 |
  | 1 | 1 | 0 |

- **The four points in the plane.** `(0,0)` and `(1,1)` are class 0; `(0,1)` and
  `(1,0)` are class 1. The two classes sit on *opposite diagonals*.

  ```
   x2
   1 |  (0,1)=1      (1,1)=0
   0 |  (0,0)=0      (1,0)=1
     +------------------- x1
          0             1
  ```

- **Why one straight line cannot separate them.** The positive points `(0,1),
  (1,0)` and the negative points `(0,0),(1,1)` alternate around the square; any
  single straight line leaves one point of each class on the wrong side. XOR is
  **not linearly separable**.

- **Prediction for a single affine + sigmoid model.** A single affine map +
  sigmoid is a linear classifier, so it *cannot* reach 0 loss on XOR; it will
  stall around 50% accuracy (loss ≈ `ln 2 ≈ 0.693`), misclassifying at least
  one point.

> **Think About It.** With only four points, XOR lets us test a *scientific
> claim about representation*: that a model needs a **nonlinear hidden
> representation**, not merely more parameters, to realise this behaviour.

---

## Task 2 — Design the intelligent agent

Baseline: **2 inputs → 2 hidden units → 1 output**, with a nonlinear hidden
activation, a sigmoid output, binary cross-entropy loss, and gradient descent.

1. **Why the hidden nonlinearity is scientifically necessary.** A stack of
   affine layers collapses into a *single* affine map (`W₂(W₁x+b₁)+b₂` is affine
   in `x`). Only a nonlinear hidden activation lets the network bend the input
   space so the two XOR diagonals become separable. Depth without nonlinearity
   adds parameters but not representational power.
2. **Why sigmoid + binary cross-entropy is a sensible output pairing.** The
   target is a single yes/no probability; sigmoid maps a logit to `(0,1)`, and
   binary cross-entropy is its matching loss, giving the clean logit gradient
   `p − y`. (In PyTorch, `BCEWithLogitsLoss` fuses sigmoid + BCE for numerical
   stability.)
3. **What counts as successful learning — at least three checks:**
   (i) final loss near 0 (≪ `ln 2`); (ii) **all four** thresholded predictions
   equal `0,1,1,0`; (iii) the first-layer gradient `dL/dW₁` is **non-zero**
   during training (a real learning signal); (iv) *(extra)* the result survives
   repeated runs / different seeds.

> **Think About It.** The hidden units are never given targets. If the network
> learns XOR, **backpropagation** is what decided what each hidden unit should
> compute — the output error is propagated back through `W₂` to assign each
> hidden unit a useful sub-feature.

---

## Task 3 — LLM-generated first implementation

Prompt and corrections are in `PROMPTS.md`. Before running, the code was
inspected to locate the four key stages:
- **forward pass** — `h1 = act(x·W1+b1)`, `z = h1·W2+b2`;
- **scalar loss** — `BCEWithLogitsLoss` / `bce_with_logits`;
- **reverse-mode AD (backprop)** — `loss.backward()` in PyTorch; the explicit
  `dz, dW2, dh1, da1, dW1` chain in NumPy;
- **optimiser update** — `opt.step()` / the `W -= lr·dW` lines.

> **Think About It.** Parts verifiable *from the code without running*: the
> architecture is 2-2-1, the loss matches the task, the output is a single
> logit. Parts needing *execution + measurement*: whether it actually reaches 0
> loss, whether all four labels are correct, gradient magnitudes, and the
> symmetry/activation behaviour.

---

## Task 4 — Execute, test, and diagnose

### Part A — basic learning check (sigmoid, seed 0)

| Quantity | Value |
|----------|-------|
| initial loss | 0.6959 |
| final loss | **0.0088** |
| probabilities | `[0.009, 0.992, 0.989, 0.007]` |
| predictions | `[0, 1, 1, 0]` (target `0,1,1,0`) |
| all four correct | **yes** |

The 2-2-1 network *does* learn XOR — confirming that the nonlinear hidden layer
(not depth alone) supplies the needed representation.

### Part B — backpropagation check

`parameter.grad` (i.e. `dL/dW⁽¹⁾`) is the gradient of the scalar loss w.r.t. the
first-layer weights, computed by reverse-mode AD. Observed first-layer gradient
(near convergence) ≈ `[[+0.0005, +0.0006], [−0.0005, −0.0006]]` — small because
the loss is nearly minimised, but **non-zero**, so a learning signal is flowing
back to the first layer. Because the loss is the **mean** over the four
examples, this gradient is the *average* of the four example-wise gradients
(the `/N` factor in `dz = (p − y)/N`).

### Part C — symmetry experiment (all weights initialised to 0)

| Quantity | Value |
|----------|-------|
| final loss | **0.6931** (= `ln 2`) |
| all correct | **no** |
| two rows of `W₁` after training | identical (`[0,0]` and `[0,0]`) |

With identical initial weights the two hidden units compute the **same** output
and therefore receive the **same** gradient at every step, so they stay
identical forever — the network effectively has one hidden unit and cannot
represent XOR. **Random initialisation is what breaks this symmetry.**

### Part D — activation experiment (random init, seed 0)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇_{W⁽¹⁾}L‖ |
|-------------------|-----------|--------------|-------------------|
| Sigmoid | 0.0088 | yes | 0.0033 |
| Tanh | 0.0016 | yes | 0.0241 |
| ReLU | 0.3467 | **no** | 0.0741 |

**Interpreting *this* experiment (not a universal claim).** On this single
seed, sigmoid and tanh solved XOR while ReLU got stuck at loss ≈ 0.35. Running
all three across 8 seeds shows the real picture — a 2-hidden-unit XOR net is
*minimal and brittle*:

| activation | successes over seeds 0–7 |
|------------|--------------------------|
| sigmoid | 7 / 8 |
| tanh | 3 / 8 |
| ReLU | 2 / 8 |

The failures are **local minima / dead units**, not evidence that an activation
is "bad". ReLU's early gradient norm is the *largest*, yet it fails most often:
a negative pre-activation gives a ReLU unit derivative exactly 0, so with only
two hidden units one "dead" unit can leave too little capacity to represent XOR.
Sigmoid/tanh saturate more gracefully here. The lesson is that with such a tiny
model the **initialisation** matters as much as the activation.

> **Think About It — distinguishing a saturated sigmoid from a dead ReLU.**
> Both give a near-zero derivative but by different mechanisms. Inspect the
> *pre-activations* `a⁽¹⁾`: a saturated sigmoid has `|a|` large (output near 0 or
> 1) while still being differentiable; a dead ReLU has `a < 0` so its output and
> derivative are both exactly 0. Looking at activations vs pre-activations tells
> the two apart.

---

## Task 5 — Three-class softmax extension

Same hidden size, but the output becomes **3 logits + softmax + cross-entropy**.
Classes: 0 = both inactive `(0,0)`; 1 = disagree `(0,1),(1,0)`; 2 = both active
`(1,1)`.

**Predictions before running:**
1. final weight matrix shape `W₂`: **(2, 3)** — 2 hidden units → 3 logits;
2. logits per example: **3**;
3. softmax probabilities sum to 1 because softmax normalises by the sum of
   exponentials (`e_k / Σ_j e_j`);
4. the logit gradient has the form `p − y` because that is the derivative of
   softmax cross-entropy w.r.t. the logits.

**Results (verified):** `W₂` shape `(2,3)`; predictions `[0,1,1,2]` = targets;
all correct. Class probabilities per input (each row sums to 1.0000):

| input | P(class0) | P(class1) | P(class2) | sum |
|-------|-----------|-----------|-----------|-----|
| (0,0) | 0.998 | 0.002 | 0.000 | 1.0000 |
| (0,1) | 0.001 | 0.997 | 0.001 | 1.0000 |
| (1,0) | 0.002 | 0.997 | 0.002 | 1.0000 |
| (1,1) | 0.000 | 0.002 | 0.998 | 1.0000 |

**Optional stability diagnostic.** Adding 100 to all three logits leaves the
softmax vector unchanged (`[0.9978, 0.0022, 0.0]` both times). Softmax is
invariant to an additive constant on the logits, which is exactly why stable
implementations subtract `max(logits)` before exponentiating — to avoid
`exp(large)` overflow without changing the result.

> **Think About It.** Next-token prediction is classification over a huge
> vocabulary. The *mathematics stays the same* (softmax + cross-entropy, logit
> gradient `p − y`, probabilities summing to 1); what changes dramatically is
> scale — the output matrix has tens of thousands of columns, and the
> surrounding architecture (embeddings, attention, tied weights, sampling) grows
> around that same classification core.

---

## Reflection Questions

1. **Depth vs nonlinearity.** XOR showed that *depth alone is not enough*:
   affine layers collapse to one affine map. The **nonlinear hidden
   activation** is what provided a representation able to separate the diagonals.
2. **Evidence of a useful learning signal (not merely a nonzero gradient).** The
   loss fell from 0.70 to 0.009, **all four** labels became correct, and the
   probabilities moved decisively toward `0,1,1,0` — behaviour, not just a
   nonzero number.
3. **Why zero/identical init prevents distinct features.** Identical units
   produce identical outputs and receive identical gradients, so they update
   identically and never diverge — symmetry is never broken.
4. **Effect of activation on the observed gradient — science vs engineering.**
   *Science:* saturated sigmoid units and negative (dead) ReLU units both yield
   small/zero derivatives, shaping how gradient flows to the first layer.
   *Engineering observation:* on this tiny net the activation changed whether
   training converged and how often, interacting strongly with initialisation
   (table above).
5. **Why output layer and loss are chosen together.** The output activation and
   loss must match the task's probabilistic form: sigmoid+BCE for a single
   yes/no; softmax+cross-entropy for K classes. Mismatching them breaks the
   clean `p − y` gradient and the probabilistic interpretation.
6. **LLM help vs human verification.** *Productivity:* the LLM wrote the
   training loop and tensor plumbing quickly. *Essential human verification:*
   confirming the architecture matched the intended experiment, that the loss
   matched the task, and that the symmetry/activation *results* were real —
   measured, not assumed.
7. **Which tests scale up, which don't.** The learning check (loss, accuracy),
   the symmetry check, and watching gradient norms remain cheap and useful at
   scale. Exhaustive per-weight finite-difference gradient checks and enumerating
   *all* inputs become infeasible for large models and large input spaces.

---

## Responsible Use of LLMs

The LLM was used as an **AI-engineering collaborator** — boilerplate, library
calls, and translating a mathematical design into code — not as a substitute for
the scientific reasoning that specifies *what behaviour the model should exhibit*
and *what evidence supports that claim*. Every result above was executed and
checked rather than taken on trust.
