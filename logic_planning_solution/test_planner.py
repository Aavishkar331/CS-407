"""
Task 3 -- Systematic tests for the planner.

Each test has a known expected outcome, so we validate the *algorithm*, not
just "did it return some plan".

Run:  python3 test_planner.py
"""

from planner import (
    PlanningProblem, Action, plan_bfs, warehouse_problem, show_plan,
)


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    return condition


def verify_plan_valid(problem, plan, states):
    """Independently re-execute the plan and confirm each precondition holds
    in the state where the action is applied, and that the goal is reached."""
    state = problem.initial
    for action in plan:
        if not action.applicable(state):
            return False
        state = action.apply(state)
    return problem.is_goal(state)


def test_A_solvable():
    print("Test A: solvable problem (deliver A -> C)")
    prob = warehouse_problem(include_pickup=True)
    plan, states = plan_bfs(prob)
    show_plan(prob, plan, states)
    ok = True
    ok &= check("  plan found", plan is not None)
    ok &= check("  plan is actually valid (re-executed independently)",
                verify_plan_valid(prob, plan, states))
    ok &= check("  goal reached: At(Package,C)",
                "At(Package,C)" in states[-1])
    return ok


def test_B_impossible():
    print("\nTest B: impossible problem (PickUp action removed)")
    prob = warehouse_problem(include_pickup=False)
    plan, states = plan_bfs(prob)
    show_plan(prob, plan, states)
    # Without PickUp the robot can never hold the package, so no plan exists.
    return check("  reports 'No plan found' (does not invent an action)",
                 plan is None)


def test_C_irrelevant_actions():
    print("\nTest C: irrelevant actions (robot at C != package at C)")
    # The robot can move to C on its own.  Check the planner does NOT accept
    # 'robot reached C' as satisfying the goal 'package at C'.
    prob = warehouse_problem(include_pickup=True)
    # A state where the robot is at C but the package is still at A.
    robot_only_at_C = frozenset(["At(Robot,C)", "At(Package,A)"])
    ok = True
    ok &= check("  goal NOT satisfied when only the robot is at C",
                not prob.is_goal(robot_only_at_C))
    # And the real plan must actually carry the package (Holding appears).
    plan, states = plan_bfs(prob)
    carried = any("Holding(Package)" in s for s in states)
    ok &= check("  valid plan actually carries the package (Holding occurs)",
                carried)
    ok &= check("  package ends at C, not merely the robot",
                "At(Package,C)" in states[-1])
    return ok


def test_D_already_satisfied():
    print("\nTest D: goal already true (edge case)")
    prob = PlanningProblem(initial=["At(Package,C)"], actions=[],
                           goal=["At(Package,C)"])
    plan, states = plan_bfs(prob)
    return check("  empty plan for an already-satisfied goal",
                 plan == [])


if __name__ == "__main__":
    print("=" * 60)
    print("Task 3: planner tests")
    print("=" * 60)
    results = [
        test_A_solvable(),
        test_B_impossible(),
        test_C_irrelevant_actions(),
        test_D_already_satisfied(),
    ]
    print("\n" + "=" * 60)
    print(f"RESULT: {sum(results)}/{len(results)} test groups passed.")
    print("=" * 60)
