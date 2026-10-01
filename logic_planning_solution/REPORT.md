# Laboratory Report — Logical Reasoning for Planning
### Artificial Intelligence · Logical Planning Laboratory

**Central idea:** `Logic + Search = Planning`. Logic decides *what is possible*
(which actions are applicable and how they change the world); search decides
*what to try* (which sequence of applicable actions reaches the goal).

**Files**

| File | Purpose |
|------|---------|
| `planner.py` | STRIPS-style planner (Action, state, applicability, BFS) |
| `test_planner.py` | systematic tests A–D with known outcomes (Task 3) |
| `planner.pl` | optional Prolog extension (Tasks 6–8) |
| `PROMPTS.md` | LLM prompt + corrections |

**Run**

```bash
python3 planner.py          # solve the warehouse delivery problem
python3 test_planner.py     # Task 3 tests (4/4 pass)
# Optional: swipl planner.pl    (SWI-Prolog, for Tasks 6-8)
```

---

## Task 0 — Understand the Planning Problem

**(a) Initial state `I`:** `{ At(Robot, A), At(Package, A) }`.

**(b) Goal `G`:** `{ At(Package, C) }`.

**(c) Actions available.** `Move`, `PickUp`, `Drop`, instantiated over the
locations. With connections A–B and B–C the concrete actions are:
`Move(A,B)`, `Move(B,A)`, `Move(B,C)`, `Move(C,B)`,
`PickUp(Package,A/B/C)`, `Drop(Package,A/B/C)`.

**(d) Preconditions and effects.**

| Action | Positive preconditions | Positive effects | Negative effects |
|--------|------------------------|------------------|------------------|
| `Move(X,Y)` | `At(Robot,X)` | `At(Robot,Y)` | `At(Robot,X)` |
| `PickUp(Package,X)` | `At(Robot,X)`, `At(Package,X)` | `Holding(Package)` | `At(Package,X)` |
| `Drop(Package,X)` | `At(Robot,X)`, `Holding(Package)` | `At(Package,X)` | `Holding(Package)` |

**Which actions are initially applicable?** In
`I = { At(Robot,A), At(Package,A) }`:
- `Move(A,B)` — applicable (`At(Robot,A)` holds).
- `PickUp(Package,A)` — **applicable**: both `At(Robot,A)` and `At(Package,A)`
  hold, so `I |= Preconditions(PickUp(Package,A))`.
- `Drop(Package,C)` — **not applicable**: it needs `At(Robot,C)` and
  `Holding(Package)`, neither of which is true in `I`.

> **Think About It.** An action is applicable only if *all* its preconditions
> are satisfied in the current state, not merely because it is in the action
> list. Checking `S |= Preconditions(a)` is exactly where logical reasoning
> enters planning.

---

## Task 1 — Construct a Plan by Hand

A plan is a sequence `a₁,…,aₙ` with `I →a₁→ S₁ →…→ Sₙ` and `Sₙ |= G`.

| State | Facts |
|-------|-------|
| `S₀` | `At(Robot,A), At(Package,A)` |
| `S₁` = after `PickUp(Package,A)` | `At(Robot,A), Holding(Package)` |
| `S₂` = after `Move(A,B)` | `At(Robot,B), Holding(Package)` |
| `S₃` = after `Move(B,C)` | `At(Robot,C), Holding(Package)` |
| `S₄` = after `Drop(Package,C)` | `At(Robot,C), At(Package,C)` ✓ |

`S₄ |= G` since `At(Package,C)` holds. Plan length = 4.

---

## Task 2 — LLM-Implemented Planner

The design was specified *before* prompting (see `PROMPTS.md`). The program
(`planner.py`) represents a **state as a set of logical propositions**
(`frozenset`), and each **action** carries positive/negative preconditions and
positive/negative effects. Mapping the lab's four ideas into the code:

| Idea | Where it appears |
|------|------------------|
| **Preconditions → when is an action applicable?** | `Action.applicable`: `pre_pos ⊆ state and pre_neg ∩ state = ∅` (this is `S |= Preconditions(a)`) |
| **Effects → how does the state change?** | `Action.apply`: `(state − eff_neg) ∪ eff_pos` (delete then add) |
| **Goal → when does planning terminate?** | `PlanningProblem.is_goal`: `goal ⊆ state` |
| **BFS → how are alternative plans explored?** | `plan_bfs`: FIFO frontier over states, with a `visited` set and parent pointers for reconstruction |

**Result of running the planner**

```
Plan  : ['PickUp(Package,A)', 'Move(A,B)', 'Move(B,C)', 'Drop(Package,C)']
S0 : At(Package,A), At(Robot,A)
S1 : after PickUp(Package,A) -> At(Robot,A), Holding(Package)
S2 : after Move(A,B)         -> At(Robot,B), Holding(Package)
S3 : after Move(B,C)         -> At(Robot,C), Holding(Package)
S4 : after Drop(Package,C)   -> At(Package,C), At(Robot,C)
```

This matches the hand-built plan (BFS expanded 6 states to find it).

---

## Task 3 — Testing the Planner (4/4 pass)

| Test | Setup | Expected | Result |
|------|-------|----------|--------|
| **A. Solvable** | original problem | a valid plan ending with `At(Package,C)` | **PASS** — plan of length 4; re-executed independently and verified valid |
| **B. Impossible** | remove `PickUp` | report **"No plan found"**, not an invented action | **PASS** — robot can never hold the package, planner reports failure |
| **C. Irrelevant actions** | robot can `Move` to C alone | planner must **not** treat *robot at C* as *package at C* | **PASS** — goal unsatisfied when only the robot is at C; the real plan carries the package (`Holding` appears) |
| **D. Already satisfied** | goal true in `I` | empty plan | **PASS** |

Test C is the key logical check: the goal is `At(Package,C)`, **not**
`At(Robot,C)`. Because the goal test is `goal ⊆ state`, moving the robot to C
without the package leaves the goal false — the planner does not confuse the
two propositions.

Each returned plan is **independently re-executed** by the test harness,
re-checking every action's preconditions in the state where it is applied —
this is validation, not merely trusting the planner's output.

---

## Task 4 — Logic and Search

The planner uses **logical reasoning** and **search** together:

- **Logical reasoning** determines, for a state `S` and action `a`, whether
  `S |= Preconditions(a)` (applicability) and computes the successor
  `S' = Apply(S, a)` from the effects. This is the *what is possible* part.
- **Search** (BFS) decides *which sequence* of applicable actions to explore,
  expanding states until one satisfies the goal.

**Completed flow (the missing step is "Apply effects"):**

```
Current state
     ↓
Check action preconditions        (logic: S |= Preconditions(a)?)
     ↓
Apply the action's effects        (logic: S' = Apply(S, a))   ← the "?" step
     ↓
Generate successor state
     ↓
Search over alternatives          (BFS over successor states)
     ↓
Goal?                             (logic: G ⊆ S' ?)
```

*In words:* logic tells the agent which actions it is **allowed** to take in a
state and exactly what each one **does** to the world; search strings these
legal transitions together, trying alternatives until the goal holds. **Logic
determines what is possible; search determines what to try.**

---

## Task 5 — Can the LLM Verify Its Own Plan?

Asking the LLM to "show that every action's preconditions are satisfied" yields
a *plausible explanation*, but it is generated by the same system that produced
the plan. We should trust **(b) the independently executed state transitions**
more than **(a) the LLM's explanation**, because the Python program actually
recomputes each state from the effects and re-checks each precondition against
it — an objective check independent of the generator.

> **Important.** A generated explanation is not the same as an independent
> verification. The point is not that LLMs are unreliable, but the general
> engineering principle: *generation and verification should be separate.*

---

## Optional Extension — Prolog as a Logical Verifier (`planner.pl`)

### Task 6 — connectivity facts and `can_move`

```prolog
connected(a,b). connected(b,a). connected(b,c). connected(c,b).
can_move(X,Y) :- connected(X,Y).
```

- **(a) Why `can_move(a,b)` is `true`:** the fact `connected(a,b)` exists, so
  the body of the rule succeeds.
- **(b) Why `can_move(a,c)` is not established:** there is no fact
  `connected(a,c)`; a and c are connected only *through* b, and the rule does
  not chain transitively.
- **(c) Relationship to logic:** the rule `can_move(X,Y) :- connected(X,Y)` is
  the Horn-clause form of the implication `Connected(X,Y) → CanMove(X,Y)`; a
  query asks whether `CanMove(a,b)` logically follows from the facts and rules.

### Task 7 — checking a proposed plan

With `valid_move(X,Y) :- connected(X,Y)`: `valid_move(a,b)` and `valid_move(b,c)`
succeed, while `valid_move(a,c)` **fails**. So if the Python planner proposed
`Move(a,c)`, Prolog — used independently — rejects it as unsupported by the
warehouse knowledge base. This is the **Generate → Independent verification**
architecture: the Python side *generates* a candidate, Prolog *checks* it.

### Task 8 — connecting Prolog to logical reasoning

```prolog
wet_road.
slippery    :- wet_road.
reduce_speed :- slippery.
```

`?- reduce_speed.` succeeds by chaining:
`wet_road  ⇒  (wet_road → slippery)  ⇒  slippery  ⇒  (slippery → reduce_speed)  ⇒  reduce_speed`,
i.e. **Fact ⇒ Rule ⇒ Rule ⇒ Conclusion.**

*(SWI-Prolog was not installed in the environment used to run the Python parts;
the expected query results are documented as comments inside `planner.pl`.)*

### 7.2 Reflection (Prolog)

1. **Fact vs rule.** A *fact* asserts something unconditionally
   (`connected(a,b).`); a *rule* (`head :- body`) asserts the head *if* the
   body can be proved.
2. **Query = entailment.** Posing `?- G.` asks whether `G` follows from the
   facts and rules — Prolog searches for a proof by resolution/backtracking,
   mirroring "does `G` follow from the knowledge base?".
3. **Why verify a Python plan with Prolog?** Because the checker is a *separate*
   system with an independent encoding of the rules; it can catch a plan that
   is syntactically fine but logically inconsistent with the warehouse.
4. **Advantage of an independent verifier for LLM output.** The plan and its
   check come from different systems, so an error (or hallucinated action) in
   the generator is not automatically echoed by the verifier.

> Key idea: *An AI system can generate a candidate solution, while a separate
> logical system checks it.*

---

## Section 5 — Reflection Questions

1. **Why specify preconditions and effects before prompting?** Because they
   *define* the problem: without them "applicable action" and "successor state"
   are undefined, and neither the planner nor its tests can be correct. The
   specification is what the generated code must satisfy.
2. **An error from not checking preconditions.** The planner might apply
   `Drop(Package,C)` without `Holding(Package)`, "teleporting" the package to C
   — producing a plan that is invalid in the real world.
3. **Why a plan that "looks reasonable" may be invalid.** It might apply an
   action whose preconditions are unmet, or confuse two propositions (robot at
   C vs package at C). Only re-executing it against the preconditions/effects
   proves validity.
4. **What the LLM contributed.** It translated the STRIPS design into clean
   Python (action class, applicability test, BFS) quickly.
5. **What had to be verified independently.** That applicability uses *both*
   positive and negative preconditions, that effects delete-then-add correctly,
   that the goal test is subset-based, and that impossible problems report
   failure — all checked by `test_planner.py`.
6. **Where logical reasoning is used.** In applicability (`S |= Preconditions`),
   in computing successors (`Apply`), and in the goal test (`G ⊆ S`); and, in
   the extension, in Prolog's inference.
7. **How planning relates to search.** Planning *is* search over states, where
   the states are sets of logical facts, the successor function is defined by
   action effects, and the goal test is logical entailment — the same search
   machinery as the previous module, applied to a logically-described state
   space.

---

## Takeaway

```
Logical reasoning + Search → Planning
Understand → Specify → Generate → Execute → Verify
```

Logic says which actions are possible and how they change the world; search
explores sequences of them; an LLM can help *implement* the planner, but the
engineer stays responsible for testing and validating the result.
