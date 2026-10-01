# CS F407 — Artificial Intelligence · Laboratory Solutions

Solutions for four AI laboratory exercises. Each lab is a self-contained folder
with runnable code, systematic tests, and a full write-up (`REPORT.md`) that
answers every task and reflection question. The common theme across all four is
using an **LLM as an engineering assistant** while keeping responsibility for
specification, testing, and validation with the engineer:

> **Understand → Specify → Generate → Execute → Verify**

All code uses **Python 3 and the standard library only** (plus NumPy for the
neural lab). No specialised AI/ML library is required.

---

## Contents

| # | Folder | Topic | Run | Status |
|---|--------|-------|-----|--------|
| 1 | [`search_lab_solution/`](search_lab_solution/) | Search & A\* (warehouse robot) | `python3 search_agent.py` | path = 40 moves; tests **4/4** |
| 2 | [`agents_lab_solution/`](agents_lab_solution/) | Goal-based agent | `python3 goal_based_agent.py` | path = 19 moves |
| 3 | [`logic_planning_solution/`](logic_planning_solution/) | Logical reasoning for planning (STRIPS) | `python3 planner.py` | plan found; tests **4/4** |
| 4 | [`neural_models_solution/`](neural_models_solution/) | Neural models: XOR, depth, activations, output layers | `python3 xor_numpy.py` | XOR learned; all experiments run |

Each folder also contains `REPORT.md` (task answers + reflections) and, where an
LLM prompt is part of the submission, `PROMPTS.md`.

---

## 1. Search & A\* — `search_lab_solution/`

A\* and BFS for a warehouse grid (`S → G`), with four selectable heuristics.

- `search_agent.py` — the search problem + `a_star` and `bfs` solvers (CLI).
- `test_cases.py` — tests with known answers (original, trivial, no-solution,
  alternative-paths). **4/4 pass.**
- `run_experiments.py` — A\*-vs-BFS comparison and heuristic investigation,
  plus two demo maps (`demo_open_room.txt`, `demo_heuristic_trap.txt`) that make
  the heuristic's effect visible.

```bash
cd search_lab_solution
python3 search_agent.py                   # A* on the warehouse map
python3 search_agent.py --algo bfs        # blind BFS
python3 search_agent.py --heuristic zero  # try other heuristics
python3 test_cases.py                     # Task 3 tests
python3 run_experiments.py                # Tasks 5 & 6
```

**Result:** solution found, path length **40**, **64** states expanded. On the
corridor map A\* and BFS tie; the demo maps show A\* pruning and an inadmissible
heuristic losing optimality.

## 2. Goal-Based Agent — `agents_lab_solution/`

A goal-based agent (explicit goal + search) for warehouse navigation.

- `goal_based_agent.py` — A\*/BFS decision-making; prints the Up/Down/Left/Right
  moves and draws the path.

```bash
cd agents_lab_solution
python3 goal_based_agent.py               # A* (default)
python3 goal_based_agent.py --algo bfs
```

**Result:** path found, length **19 moves**, 22 states expanded.

## 3. Logical Planning — `logic_planning_solution/`

A STRIPS-style planner: state = set of logical propositions; actions carry
positive/negative preconditions and effects; BFS over states.
`Logic + Search = Planning`.

- `planner.py` — `Action`, applicability (`S |= Preconditions`), `Apply`, BFS.
- `test_planner.py` — tests (solvable, impossible, irrelevant-actions,
  already-satisfied). **4/4 pass.**
- `planner.pl` — optional Prolog extension (Tasks 6–8).

```bash
cd logic_planning_solution
python3 planner.py                        # deliver Package A -> C
python3 test_planner.py                   # Task 3 tests
# optional: swipl planner.pl              # Prolog verifier (Tasks 6-8)
```

**Result:** plan `PickUp(A) → Move(A,B) → Move(B,C) → Drop(C)`.

## 4. Neural Models — `neural_models_solution/`

A 2-2-1 network for the XOR / sensor-disagreement problem: learning check,
backpropagation check, weight-symmetry experiment, activation comparison, and a
three-class softmax extension.

- `xor_torch.py` — PyTorch reference implementation (the lab's requested form).
- `xor_numpy.py` — dependency-free version with explicit backpropagation, used
  to **run** the experiments.

```bash
cd neural_models_solution
python3 xor_numpy.py                       # runs all experiments (NumPy)
# python3 xor_torch.py                     # same experiment in PyTorch (needs torch)
```

**Result:** XOR learned (loss 0.70 → 0.009, all four correct); zero-init breaks
symmetry (units stay identical, XOR not learned); three-class softmax
probabilities sum to 1 and are shift-invariant.

> **Environment note:** PyTorch could not be installed on the machine used to
> run these experiments, so the recorded numbers come from the equivalent
> `xor_numpy.py`. `xor_torch.py` is the PyTorch code for submission.
> Likewise, SWI-Prolog was unavailable, so `planner.pl`'s expected query results
> are documented inline as comments.

---

## Running everything

```bash
# from the repository root
for d in search_lab_solution agents_lab_solution logic_planning_solution neural_models_solution; do
  echo "== $d =="; ( cd "$d" && ls *.py 2>/dev/null ); done
```

`__pycache__/` and other generated files are ignored via `.gitignore`.
