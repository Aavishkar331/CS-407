"""
Goal-Based Agent -- Warehouse Navigation
Artificial Intelligence, Agents Laboratory Exercise.

This program implements a *goal-based agent* (not a simple reflex agent) for
the warehouse navigation problem.  A goal-based agent has an explicit goal and
chooses actions by searching for a sequence that reaches it.

Components of the goal-based architecture (Task 2):

    Environment     : the 2-D ASCII grid (walls '#', free '.', start 'S', goal 'G')
    State           : the vehicle's current cell (row, col)
    Goal            : reach the cell marked 'G'
    Actions         : Up, Down, Left, Right (one cell, if not blocked)
    Decision-making : a search algorithm (A* by default, BFS available) that
                      plans a collision-free path from S to G before acting.

    +-------------+     percept (map)     +------------------+
    | Environment | --------------------> |  Goal-Based      |
    |  (grid)     | <-------------------- |  Agent           |
    +-------------+     action (move)     |                  |
                                          |  state  --+      |
                                          |  goal     |      |
                                          |  search --+--> plan (path)
                                          +------------------+

No AI/ML library is used -- only the Python standard library.
"""

from __future__ import annotations

import argparse
import heapq
from collections import deque


class Warehouse:
    """The environment and the goal-based agent's model of it."""

    ACTIONS = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]

    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.start = self.goal = None
        for r, line in enumerate(grid):
            for c, ch in enumerate(line):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("Map must contain both 'S' and 'G'.")

    def free(self, r, c):
        """Is cell (r, c) inside the grid and not a shelf '#'?"""
        if not (0 <= r < self.rows):
            return False
        line = self.grid[r]
        return 0 <= c < len(line) and line[c] != "#"

    def neighbours(self, state):
        """Valid moves from a cell: the collision-free successors."""
        r, c = state
        for name, dr, dc in self.ACTIONS:
            nr, nc = r + dr, c + dc
            if self.free(nr, nc):
                yield name, (nr, nc)

    def is_goal(self, state):
        return state == self.goal


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct(came_from, state):
    path = [state]
    while came_from[state] is not None:
        state = came_from[state]
        path.append(state)
    path.reverse()
    return path


def a_star(env):
    """Decision-making component: A* search for a shortest collision-free path."""
    start, goal = env.start, env.goal
    frontier = [(manhattan(start, goal), 0, start)]
    g = {start: 0}
    came_from = {start: None}
    visited = set()
    expanded = 0
    counter = 0
    while frontier:
        _f, _, cur = heapq.heappop(frontier)
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1
        if env.is_goal(cur):
            return reconstruct(came_from, cur), expanded
        for _name, nxt in env.neighbours(cur):
            ng = g[cur] + 1
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came_from[nxt] = cur
                counter += 1
                heapq.heappush(frontier, (ng + manhattan(nxt, goal), counter, nxt))
    return None, expanded


def bfs(env):
    """Alternative blind strategy, for comparison."""
    start = env.start
    frontier = deque([start])
    came_from = {start: None}
    visited = {start}
    expanded = 0
    while frontier:
        cur = frontier.popleft()
        expanded += 1
        if env.is_goal(cur):
            return reconstruct(came_from, cur), expanded
        for _name, nxt in env.neighbours(cur):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = cur
                frontier.append(nxt)
    return None, expanded


def path_to_moves(path):
    """Translate a cell path into the Up/Down/Left/Right actions the agent takes."""
    moves = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = r2 - r1, c2 - c1
        moves.append({(-1, 0): "Up", (1, 0): "Down",
                      (0, -1): "Left", (0, 1): "Right"}[(dr, dc)])
    return moves


def draw(env, path):
    grid = [list(row) for row in env.grid]
    for (r, c) in path:
        if grid[r][c] not in ("S", "G"):
            grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


def load_map(path):
    with open(path, encoding="utf-8") as fh:
        rows = [line.rstrip("\n") for line in fh]
    while rows and rows[-1] == "":
        rows.pop()
    return rows


def main():
    ap = argparse.ArgumentParser(description="Goal-based warehouse navigation agent.")
    ap.add_argument("map", nargs="?", default="warehouse_map.txt")
    ap.add_argument("--algo", choices=["astar", "bfs"], default="astar")
    args = ap.parse_args()

    env = Warehouse(load_map(args.map))
    print(f"Environment    : {args.map}")
    print(f"Start -> Goal  : {env.start} -> {env.goal}")
    print(f"Strategy       : {'A* (Manhattan)' if args.algo == 'astar' else 'BFS'}")

    path, expanded = (a_star(env) if args.algo == "astar" else bfs(env))

    if path is None:
        print("Result         : NO PATH FOUND (goal is unreachable).")
        print(f"States expanded: {expanded}")
        return

    print(f"Result         : path found")
    print(f"Path length    : {len(path) - 1} moves")
    print(f"States expanded: {expanded}")
    print(f"Moves          : {path_to_moves(path)}")
    print(f"Cells          : {path}")
    print("\nMap with path ('*'):\n")
    print(draw(env, path))


if __name__ == "__main__":
    main()
