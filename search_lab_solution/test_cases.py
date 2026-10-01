"""
Task 3 -- Systematic testing of the generated A* program.

Principle from the lab sheet:  Working output != validated algorithm.
Each test has a KNOWN expected outcome, so we can check correctness, not
just "did it produce some plausible path".

Run with:   python3 test_cases.py
"""

from search_agent import WarehouseProblem, a_star, bfs, parse_map, load_map


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return condition


# --- Test 1: Original warehouse -------------------------------------------
def test_original():
    problem = WarehouseProblem(load_map("warehouse_map.txt"))
    res = a_star(problem)
    bfs_res = bfs(problem)
    print("Test 1 (original warehouse):")
    print(res)
    ok = True
    ok &= check("  path found", res.found)
    # A* with an admissible heuristic must match BFS's optimal length.
    ok &= check("  path is optimal (== BFS length)",
                res.path_length == bfs_res.path_length,
                f"A*={res.path_length}, BFS={bfs_res.path_length}")
    # start and end of path are correct
    ok &= check("  path starts at S", res.path[0] == problem.start)
    ok &= check("  path ends at G", res.path[-1] == problem.goal)
    return ok


# --- Test 2: Trivial case (goal adjacent to start) ------------------------
def test_trivial():
    grid = parse_map(
        "#####\n"
        "#SG##\n"
        "#####\n"
    )
    problem = WarehouseProblem(grid)
    res = a_star(problem)
    print("\nTest 2 (trivial, one step):")
    print(res)
    ok = True
    ok &= check("  path found", res.found)
    ok &= check("  path length == 1", res.path_length == 1)
    return ok


# --- Test 3: No solution (goal walled off) --------------------------------
def test_no_solution():
    grid = parse_map(
        "#######\n"
        "#S....#\n"
        "###.###\n"
        "#...#G#\n"
        "#######\n"
    )
    problem = WarehouseProblem(grid)
    res = a_star(problem)
    print("\nTest 3 (no solution -- goal is walled off):")
    print(res)
    ok = True
    ok &= check("  reports failure (no infinite loop)", res.found is False)
    ok &= check("  path length is None", res.path_length is None)
    return ok


# --- Test 4: Alternative paths (must return a SHORTEST one) ----------------
def test_alternative_paths():
    # Open room: many S->G paths exist; shortest has length 6 (3 down + 3 right).
    grid = parse_map(
        "#####\n"
        "#S..#\n"
        "#...#\n"
        "#...#\n"
        "#..G#\n"
        "#####\n"
    )
    problem = WarehouseProblem(grid)
    res = a_star(problem)
    bfs_res = bfs(problem)
    print("\nTest 4 (alternative paths -- check shortest):")
    print(res)
    # Manhattan distance S(1,1)->G(4,3) = 3 + 2 = 5 moves, the true optimum.
    ok = True
    ok &= check("  path found", res.found)
    ok &= check("  path length == 5 (optimal)", res.path_length == 5)
    ok &= check("  matches BFS optimal length",
                res.path_length == bfs_res.path_length)
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("Task 3: Systematic tests")
    print("=" * 60)
    results = [
        test_original(),
        test_trivial(),
        test_no_solution(),
        test_alternative_paths(),
    ]
    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} test groups passed.")
    print("=" * 60)
