"""
Task 5 -- Compare A* with blind search (BFS).
Task 6 -- Investigate the heuristic.

Both experiments run on the SAME warehouse map so the comparison is fair.

Run with:   python3 run_experiments.py
"""

from search_agent import (
    WarehouseProblem, a_star, bfs, load_map, HEURISTICS,
)


def _row(label, res):
    found = "yes" if res.found else "no"
    length = res.path_length if res.found else "-"
    return f"{label:<16}{found:<10}{str(length):<14}{res.states_expanded}"


def task5_compare(problem):
    print("=" * 56)
    print("Task 5: A* vs BFS (same map)")
    print("=" * 56)
    a = a_star(problem, HEURISTICS["manhattan"])
    b = bfs(problem)

    print(f"{'Measure':<16}{'BFS':<16}{'A*':<16}")
    print("-" * 48)
    print(f"{'Solution found':<16}"
          f"{('yes' if b.found else 'no'):<16}"
          f"{('yes' if a.found else 'no'):<16}")
    print(f"{'Path length':<16}{str(b.path_length):<16}{str(a.path_length):<16}")
    print(f"{'States expanded':<16}{b.states_expanded:<16}{a.states_expanded:<16}")

    print("\nAnswers:")
    print(f"  (a) Both found a solution?  "
          f"{'yes' if (a.found and b.found) else 'no'}")
    print(f"  (b) Same path length?       "
          f"{'yes' if a.path_length == b.path_length else 'no'} "
          f"(BFS={b.path_length}, A*={a.path_length})")
    fewer = "A*" if a.states_expanded < b.states_expanded else \
            ("BFS" if b.states_expanded < a.states_expanded else "equal")
    print(f"  (c) Fewer states expanded:  {fewer} "
          f"(BFS={b.states_expanded}, A*={a.states_expanded})")
    print("  (d) Why can A* expand fewer? Because the Manhattan heuristic")
    print("      pulls the search toward the goal instead of expanding")
    print("      uniformly in all directions as BFS does.")


def task6_heuristics(problem):
    print("\n" + "=" * 56)
    print("Task 6: Heuristic investigation (A* on the same map)")
    print("=" * 56)
    print(f"{'Heuristic':<16}{'Found':<10}{'Path len':<14}{'Expanded'}")
    print("-" * 48)
    order = ["zero", "euclidean", "manhattan", "manhattan_x2"]
    base_len = None
    for name in order:
        res = a_star(problem, HEURISTICS[name])
        if name == "manhattan":
            base_len = res.path_length
        print(_row(name, res))

    print("\nInterpretation:")
    print("  zero         -> A* degenerates to uniform-cost/BFS: optimal but")
    print("                  expands the most states (no guidance).")
    print("  euclidean    -> admissible (<= true cost) so still optimal, but")
    print("                  weaker than Manhattan on a 4-connected grid, so")
    print("                  it usually expands a few more states.")
    print("  manhattan    -> admissible AND tight for 4-connected moves:")
    print("                  optimal path, fewest expansions. Best choice.")
    print("  manhattan_x2 -> INADMISSIBLE (over-estimates). Faster/greedier,")
    print("                  expands few states, but the path is NOT")
    print("                  guaranteed optimal -- compare its length above.")
    if base_len is not None:
        x2 = a_star(problem, HEURISTICS["manhattan_x2"])
        verdict = ("still optimal here" if x2.path_length == base_len
                   else "LONGER than optimal -- optimality lost")
        print(f"\n  manhattan_x2 path length vs optimal: "
              f"{x2.path_length} vs {base_len} ({verdict})")


def supplementary_demos():
    """Two extra maps that make the Task 5/6 lessons visible.

    The supplied warehouse map is essentially one long winding corridor, so
    BFS and every heuristic expand the same cells (there is nothing to
    prune).  These small maps show the behaviour the lab is pointing at.
    """
    print("\n" + "=" * 56)
    print("Supplementary demo A: open room (heuristic saves work)")
    print("=" * 56)
    room = WarehouseProblem(load_map("demo_open_room.txt"))
    b = bfs(room)
    print(f"{'Method':<16}{'Found':<8}{'Path len':<12}{'Expanded'}")
    print("-" * 44)
    print(_row("BFS", b))
    for name in ["zero", "euclidean", "manhattan", "manhattan_x2"]:
        print(_row(name, a_star(room, HEURISTICS[name])))
    print("Note: manhattan ties cover the whole S-G bounding box, so it still")
    print("      expands the plateau; the greedy manhattan_x2 expands far")
    print("      fewer -- and here it stays optimal because the room is open.")

    print("\n" + "=" * 56)
    print("Supplementary demo B: trap map (inadmissible h loses optimality)")
    print("=" * 56)
    trap = WarehouseProblem(load_map("demo_heuristic_trap.txt"))
    opt = a_star(trap, HEURISTICS["manhattan"])
    x2 = a_star(trap, HEURISTICS["manhattan_x2"])
    print("\n".join(trap.grid))
    print(f"\n  manhattan    : len={opt.path_length}, expanded={opt.states_expanded}  (admissible -> optimal)")
    print(f"  manhattan_x2 : len={x2.path_length}, expanded={x2.states_expanded}  "
          f"({'SUBOPTIMAL' if x2.path_length > opt.path_length else 'optimal'} -- over-estimate "
          f"made A* commit early and never reopen the better route)")
    print("\n  This is the admissibility lesson: h(n) <= h*(n) guarantees A* is")
    print("  optimal; an over-aggressive (inadmissible) heuristic can be")
    print("  faster but returns a longer path.")


if __name__ == "__main__":
    problem = WarehouseProblem(load_map("warehouse_map.txt"))
    print(f"Start -> Goal: {problem.start} -> {problem.goal}\n")
    task5_compare(problem)
    task6_heuristics(problem)
    supplementary_demos()
