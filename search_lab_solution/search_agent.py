"""
Warehouse Robot Navigation -- Search Agent
Artificial Intelligence (CS F407), Laboratory Exercise: Search and A*

This module implements the warehouse navigation problem as a formal search
problem and provides two solvers:

    * a_star(...)  -- informed search using f(n) = g(n) + h(n)
    * bfs(...)     -- uninformed (blind) breadth-first search

It is deliberately written WITHOUT any AI / ML library.  The only imports are
the Python standard library (heapq, collections, math, argparse).

The design matches the formal model from the lab sheet:

        P = (S, A, T, s0, G, c)

    S   : the set of (row, col) grid cells that are not obstacles
    A   : {Up, Down, Left, Right}
    T   : moving one cell in a direction, if the target cell is free
    s0  : the cell marked 'S'
    G   : the (single) cell marked 'G'
    c   : every move has cost 1

See REPORT.md for the full written answers to Tasks 0-7.
"""

from __future__ import annotations

import argparse
import heapq
import math
from collections import deque


# ---------------------------------------------------------------------------
# Task 0 / Task 1 : the search problem itself
# ---------------------------------------------------------------------------
class WarehouseProblem:
    """A goal-based search problem built from an ASCII warehouse map.

    A *state* is simply a cell position (row, col).  That is all that is
    needed here: the robot has no orientation, no load, no battery, so the
    cell fully determines the situation (answer to Task 0(a)).
    """

    # The four moves the robot can make.  (name, d_row, d_col)
    ACTIONS = [
        ("Up",    -1,  0),
        ("Down",   1,  0),
        ("Left",   0, -1),
        ("Right",  0,  1),
    ]

    def __init__(self, grid):
        # grid is a list of equal-length strings (the ASCII rows).
        self.grid = grid
        self.rows = len(grid)
        self.cols = max(len(r) for r in grid) if grid else 0
        self.start = None
        self.goal = None
        for r, line in enumerate(grid):
            for c, ch in enumerate(line):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None:
            raise ValueError("Map has no start cell 'S'.")
        if self.goal is None:
            raise ValueError("Map has no goal cell 'G'.")

    # --- primitive queries ------------------------------------------------
    def is_free(self, r, c):
        """A cell is usable if it is inside the grid and not an obstacle."""
        if not (0 <= r < self.rows):
            return False
        line = self.grid[r]
        if not (0 <= c < len(line)):
            return False
        return line[c] != "#"

    def initial_state(self):
        return self.start

    def is_goal(self, state):            # Task 4: goal test
        return state == self.goal

    def successors(self, state):
        """Transition function T.

        Yields (action_name, next_state, step_cost) for every *valid* move.
        A move is invalid (Task 0(b)) when it leaves the grid or lands on an
        obstacle '#'; such moves are simply not generated.
        """
        r, c = state
        for name, dr, dc in self.ACTIONS:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                yield name, (nr, nc), 1   # Task 0: every move has cost 1


# ---------------------------------------------------------------------------
# Heuristics (Task 6)
# ---------------------------------------------------------------------------
def manhattan(state, goal):
    """h(n) = |x - xG| + |y - yG|.  Admissible for 4-connected grids."""
    (r, c), (gr, gc) = state, goal
    return abs(r - gr) + abs(c - gc)


def euclidean(state, goal):
    """Straight-line distance.  Admissible but weaker than Manhattan here."""
    (r, c), (gr, gc) = state, goal
    return math.hypot(r - gr, c - gc)


def zero(state, goal):
    """h(n) = 0.  Turns A* into uniform-cost search (blind)."""
    return 0


def manhattan_x2(state, goal):
    """2 * Manhattan.  Inadmissible -- may lose optimality."""
    return 2 * manhattan(state, goal)


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "zero": zero,
    "manhattan_x2": manhattan_x2,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class SearchResult:
    """What the agent reports when it terminates (Task 1)."""

    def __init__(self, found, path, states_expanded):
        self.found = found
        self.path = path                    # list of (r, c) from start to goal
        self.states_expanded = states_expanded

    @property
    def path_length(self):
        # Number of moves == number of cells - 1.  None if no path.
        return (len(self.path) - 1) if self.found else None

    def __str__(self):
        lines = [
            f"  solution found : {self.found}",
            f"  path length    : {self.path_length}",
            f"  states expanded: {self.states_expanded}",
        ]
        if self.found:
            lines.append(f"  path           : {self.path}")
        return "\n".join(lines)


def reconstruct_path(came_from, state):
    """Walk the parent pointers back to the start to rebuild the path."""
    path = [state]
    while came_from[state] is not None:
        state = came_from[state]
        path.append(state)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Task 2 : A* search
# ---------------------------------------------------------------------------
def a_star(problem, heuristic=manhattan):
    """A* search.  f(n) = g(n) + h(n).

    Frontier      : a binary min-heap (priority queue) keyed on f(n).
    g_score       : best known cost from the start to each state.
    visited       : states already expanded (closed set) -- this is what
                    prevents repeatedly expanding the same state (Task 4(e)).
    """
    start = problem.initial_state()
    goal = problem.goal

    # tie-breaker 'counter' keeps the heap from comparing states directly
    counter = 0
    g_score = {start: 0}
    came_from = {start: None}
    f_start = g_score[start] + heuristic(start, goal)
    frontier = [(f_start, 0, start)]
    visited = set()
    states_expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        # A state can sit in the heap more than once with different f values;
        # skip the stale copies.
        if current in visited:
            continue
        visited.add(current)
        states_expanded += 1

        if problem.is_goal(current):        # goal test on expansion
            return SearchResult(True, reconstruct_path(came_from, current),
                                states_expanded)

        g_current = g_score[current]
        for _action, nxt, step in problem.successors(current):
            tentative_g = g_current + step
            if nxt in visited:
                continue
            if tentative_g < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                f_next = tentative_g + heuristic(nxt, goal)
                counter += 1
                heapq.heappush(frontier, (f_next, counter, nxt))

    # Frontier emptied without reaching the goal -> no solution (Task 3).
    return SearchResult(False, None, states_expanded)


# ---------------------------------------------------------------------------
# Task 5 : Breadth-first (blind) search, for comparison
# ---------------------------------------------------------------------------
def bfs(problem):
    """Breadth-first search.  Frontier is a FIFO queue.

    Because every step has cost 1, BFS also returns a shortest path, but it
    uses no heuristic -- it expands outward uniformly in all directions.
    """
    start = problem.initial_state()
    frontier = deque([start])
    came_from = {start: None}
    visited = {start}
    states_expanded = 0

    while frontier:
        current = frontier.popleft()
        states_expanded += 1

        if problem.is_goal(current):
            return SearchResult(True, reconstruct_path(came_from, current),
                                states_expanded)

        for _action, nxt, _step in problem.successors(current):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                frontier.append(nxt)

    return SearchResult(False, None, states_expanded)


# ---------------------------------------------------------------------------
# Utilities: loading maps and pretty-printing a solved map
# ---------------------------------------------------------------------------
def load_map(path):
    """Read an ASCII map file into a list of rows (newline stripped)."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = [line.rstrip("\n") for line in fh]
    # drop any trailing blank line(s) at the end of the file
    while rows and rows[-1] == "":
        rows.pop()
    return rows


def parse_map(text):
    """Build a map (list of rows) from a multiline string."""
    rows = [line for line in text.splitlines()]
    while rows and rows[-1] == "":
        rows.pop()
    return rows


def render_solution(problem, result):
    """Return the map as a string with the path drawn using '*'."""
    grid = [list(row) for row in problem.grid]
    if result.found:
        for (r, c) in result.path:
            if grid[r][c] not in ("S", "G"):
                grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Warehouse robot navigation using A* / BFS search.")
    parser.add_argument("map", nargs="?", default="warehouse_map.txt",
                        help="path to an ASCII map file (default: warehouse_map.txt)")
    parser.add_argument("--algo", choices=["astar", "bfs"], default="astar",
                        help="search algorithm to use (default: astar)")
    parser.add_argument("--heuristic", choices=list(HEURISTICS), default="manhattan",
                        help="heuristic for A* (default: manhattan)")
    parser.add_argument("--no-draw", action="store_true",
                        help="do not draw the path on the map")
    args = parser.parse_args()

    problem = WarehouseProblem(load_map(args.map))

    print(f"Map            : {args.map}")
    print(f"Start -> Goal  : {problem.start} -> {problem.goal}")

    if args.algo == "astar":
        print(f"Algorithm      : A* (heuristic = {args.heuristic})")
        result = a_star(problem, HEURISTICS[args.heuristic])
    else:
        print("Algorithm      : BFS")
        result = bfs(problem)

    print(result)

    if not args.no_draw:
        print("\nMap with path ('*'):\n")
        print(render_solution(problem, result))


if __name__ == "__main__":
    main()
