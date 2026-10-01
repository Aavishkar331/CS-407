# Laboratory Report — Constructing a Goal-Based Agent with an LLM
### Agents Laboratory · Artificial Intelligence

**Files**

| File | Purpose |
|------|---------|
| `warehouse_map.txt` | the warehouse grid (S → G) |
| `goal_based_agent.py` | the goal-based agent (A\* + BFS decision-making) |
| `PROMPTS.md` | the LLM prompt used and corrections made |

**Run**

```bash
python3 goal_based_agent.py              # A* (default)
python3 goal_based_agent.py --algo bfs   # blind BFS, same map
```

---

## Task 1 — Understanding the Problem

1. **What is the environment?** A 2-D warehouse grid. Each cell is a shelf
   (`#`, impassable) or free space (`.`); `S` is the loading bay, `G` the
   dispatch point. It is *fully observable, deterministic, static, discrete,
   and single-agent*.
2. **What is the goal of the agent?** To move the vehicle from `S` to `G`
   along a collision-free path (never entering a `#` cell).
3. **What actions are available?** `Up, Down, Left, Right` — each moves the
   vehicle one cell, and is only valid if the target cell is free.
4. **What information must the agent maintain to choose its next action?** Its
   current cell, the map (to test which moves are collision-free), the goal
   cell, and — because it plans ahead — a frontier of cells to explore plus the
   parent pointers needed to reconstruct the path.
5. **Why is this a goal-based agent, not a simple reflex agent?** A simple
   reflex agent maps the *current percept* directly to an action via
   condition–action rules; it has no notion of a destination and could loop or
   wander. This agent has an **explicit goal** (`G`) and **searches** for a
   sequence of actions that provably reaches it, choosing moves by their
   expected progress toward the goal rather than by a fixed reflex.

**Think About It — if the warehouse doubled in size.** The same search
*strategy* (A\*/BFS) still applies and is still correct, because the problem
structure is unchanged. The difficulty is *scale*: the number of states grows
with the area, so BFS's memory and time grow accordingly. A\* with a good
heuristic scales better, but memory can still become the bottleneck; very large
maps may need more memory-efficient variants (e.g. IDA\*) or a coarser grid.

---

## Task 2 — Designing the Agent

| Component | Design |
|-----------|--------|
| **Environment** | list of equal-length strings; `grid[r][c]`; `#` blocks |
| **Current state** | `(row, col)` tuple of the vehicle |
| **Goal** | the cell equal to `G`; goal test is `state == goal` |
| **Actions** | `Up/Down/Left/Right`, kept only when the target cell is free |
| **Decision-making** | A\* search (`f = g + h`, Manhattan heuristic) plans a shortest path before the vehicle moves; BFS provided for comparison |

**Block diagram** (how the components interact):

```
   Environment (grid) ──percept──▶ Agent
                                    │
                      ┌─────────────┼──────────────┐
                      ▼             ▼              ▼
                 current state    goal        action model
                      └──────▶ SEARCH (A*) ◀──────┘
                                    │
                                 plan: S → … → G
                                    │
   Environment ◀──action (Up/Down/Left/Right)── Agent
```

The agent first **plans** the whole path with search, then **executes** the
moves. This separation of planning from acting is the essence of a goal-based
agent.

---

## Task 3 — Prompt Engineering, Testing, and Questions

The LLM prompt (see `PROMPTS.md`) gave a precise specification rather than
"write a program". The generated code was run and validated.

**Result on the supplied map**

- Path found: **yes**
- Path length: **19 moves**
- States expanded: A\* **22**, BFS **22** (both optimal; equal here because the
  map's open top corridor leaves little for the heuristic to prune)
- Path drawn:

```
####################
#S****#***********G#
#.##.***#########..#
#....##............#
#.#####.###.#.###..#
####################
```

(The `*` marks the planned route from S to G.)

**Questions**

1. **Did the LLM generate a working program on the first attempt?** The core
   A\*/BFS logic was correct; the parts needing care were the *goal test on
   expansion*, the *closed set* to avoid re-expanding cells, and *failure
   reporting* when the goal is unreachable.
2. **If not, how can you improve your prompt?** State the environment encoding
   exactly (symbols, that moves off-grid or into `#` are invalid), demand an
   explicit "no path found" message, and ask it to report path length and
   states expanded so the output is testable.
3. **What search algorithm did the LLM choose?** A\* (with BFS as a blind
   baseline). A\* is informed: it uses `f(n) = g(n) + h(n)` with the Manhattan
   heuristic.
4. **Why did it select this algorithm?** Because the task is *shortest-path on
   a uniform-cost grid with 4-connected moves*, which is the textbook setting
   for A\* with Manhattan distance: the heuristic is admissible (never
   over-estimates), so A\* returns an optimal, collision-free path while
   expanding fewer states than blind search in general.

**Validation beyond "it runs".** The program was also checked on a trivial
one-step map and an unreachable-goal map to confirm it finds the one-step
solution and reports failure instead of looping — *a program that produces a
plausible path is not the same as a validated one*.

---

## Reflection — LLM as an engineering assistant

- **What the LLM did well:** quickly produced idiomatic `heapq`/`deque` search
  code and a clear code structure from a precise specification.
- **What required human responsibility:** specifying the problem correctly,
  insisting on a closed set and explicit failure handling, and *testing* the
  output against cases with known answers. The engineer — not the LLM — remains
  responsible for the correctness of the agent.

**Central lesson:** `Specify → Generate → Execute → Validate`. An LLM is a fast
software-engineering assistant; a goal-based agent it writes is only trustworthy
once its plans have been tested against known outcomes.
