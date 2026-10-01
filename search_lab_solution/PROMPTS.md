# LLM Prompts Appendix (Task 2)

Per the lab's instruction, the prompts used with the LLM are recorded here so
another student can reproduce the work. The design in Task 1 was completed
*before* prompting.

## Prompt 1 — generate the A\* agent

> I am implementing a simple goal-based search agent in Python.
> The environment is a grid represented by an ASCII map. The agent starts at S
> and must reach G. The symbols `#` represent obstacles and `.` represents free
> cells. The agent can move up, down, left, or right, and every movement has
> cost 1. Implement A\* search. Use Manhattan distance as the heuristic:
> `h(n) = |x − x_G| + |y − y_G|`.
> The program should: represent grid positions as states; maintain an
> appropriate frontier; calculate g(n), h(n) and f(n); avoid repeatedly
> expanding the same state; reconstruct the path when the goal is reached;
> report the path and its length; report the number of states expanded.
> Keep the implementation simple and explain the main components of the code.

## Prompt 2 — BFS version for comparison (Task 5)

> Produce a breadth-first search (BFS) version of the same agent on the same
> ASCII-grid environment, reporting whether a solution was found, the path, the
> path length, and the number of states expanded, so I can compare it against
> the A\* version.

## Prompt 3 — heuristic explanation (Task 6)

> Explain why Manhattan distance is an appropriate (admissible) heuristic for a
> grid where the robot can move only horizontally and vertically, and what
> happens to A\*'s optimality and number of expanded states if the heuristic is
> replaced by 0, by Euclidean distance, or by 2 × Manhattan distance.

## What was accepted vs. changed

- **Accepted:** the `heapq`-based priority-queue frontier, the `came_from`
  parent-pointer reconstruction, the Manhattan heuristic.
- **Changed / hardened by hand:** an explicit tie-break counter in the heap
  entries, a `visited` (closed) set plus a `g_score` improvement check to avoid
  re-expansion, and explicit failure reporting (so an unreachable goal returns
  "no solution" rather than looping). All behaviour was then validated with
  `test_cases.py`.
