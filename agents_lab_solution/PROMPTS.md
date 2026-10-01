# LLM Prompt Appendix

## Prompt used (Task 3)

> Write a well-documented Python program implementing a goal-based agent for a
> warehouse navigation problem. The warehouse is a 2-D grid given as an ASCII
> map where `#` is an obstacle (shelf), `.` is free space, `S` is the start and
> `G` is the goal. The vehicle may move Up, Down, Left, or Right by one cell; a
> move is invalid if it leaves the grid or enters a `#`. The program should:
> represent the warehouse as a 2-D grid; determine a collision-free path from
> `S` to `G` using A\* search with the Manhattan-distance heuristic; print the
> path (and the Up/Down/Left/Right moves) or a clear "no path found" message if
> none exists; report the path length and the number of states expanded; and
> explain the chosen search algorithm and why it is appropriate. Use only the
> Python standard library.

## Corrections made before accepting the code

1. **Goal test on expansion + closed set.** Ensured a cell is marked visited
   when expanded and that stale priority-queue entries are skipped, so cells are
   not re-expanded.
2. **Explicit failure path.** Made the program return/print "NO PATH FOUND"
   when the frontier empties, instead of silently finishing or looping.
3. **Tie-breaking in the heap.** Added a monotonic counter to the heap tuples so
   two cells with equal `f` are never compared directly.

All three were then checked against a trivial map and an unreachable-goal map.
