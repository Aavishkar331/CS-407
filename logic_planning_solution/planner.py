"""
Logical Reasoning for Planning -- a tiny STRIPS-style planner.
Artificial Intelligence, Laboratory: Logical Planning.

A planning problem is (I, A, G):

    I : the initial state   -- a set of true logical propositions
    A : the available actions
    G : the goal            -- a set of propositions that must hold

Each action has:
    name, positive preconditions, negative preconditions,
    positive effects (added), negative effects (deleted).

Logic connects to search through applicability:

    S |= Preconditions(a)        action a is applicable in state S
    S' = Apply(S, a)             the successor state

The planner uses breadth-first search over states to find a sequence of
applicable actions a1, a2, ..., an with  I -a1-> S1 -a2-> ... -> Sn |= G.

No external library is used -- only the standard library.
"""

from __future__ import annotations

from collections import deque


class Action:
    """A STRIPS action: name + (pos/neg) preconditions and (pos/neg) effects."""

    def __init__(self, name, pre_pos=(), pre_neg=(), eff_pos=(), eff_neg=()):
        self.name = name
        self.pre_pos = frozenset(pre_pos)   # must be TRUE to apply
        self.pre_neg = frozenset(pre_neg)   # must be FALSE to apply
        self.eff_pos = frozenset(eff_pos)   # become TRUE after applying
        self.eff_neg = frozenset(eff_neg)   # become FALSE after applying

    def applicable(self, state):
        """S |= Preconditions(a):  all positive pres hold AND no negative pre holds."""
        return self.pre_pos <= state and self.pre_neg.isdisjoint(state)

    def apply(self, state):
        """S' = Apply(S, a):  delete negative effects, then add positive effects."""
        return (state - self.eff_neg) | self.eff_pos

    def __repr__(self):
        return self.name


class PlanningProblem:
    def __init__(self, initial, actions, goal):
        self.initial = frozenset(initial)
        self.actions = list(actions)
        self.goal = frozenset(goal)

    def is_goal(self, state):
        """Goal reached when every goal proposition is true in the state."""
        return self.goal <= state

    def applicable_actions(self, state):
        return [a for a in self.actions if a.applicable(state)]


def plan_bfs(problem, trace=False):
    """Breadth-first search for a plan.

    Returns (plan, states) where plan is a list of actions and states is the
    list of states S0, S1, ..., Sn.  Returns (None, None) if no plan exists.
    """
    start = problem.initial
    if problem.is_goal(start):
        return [], [start]

    frontier = deque([start])
    came_from = {start: (None, None)}       # state -> (prev_state, action taken)
    visited = {start}
    expanded = 0

    while frontier:
        state = frontier.popleft()
        expanded += 1
        for action in problem.applicable_actions(state):
            nxt = action.apply(state)
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = (state, action)
            if problem.is_goal(nxt):
                if trace:
                    print(f"  (BFS expanded {expanded} states)")
                return _reconstruct(came_from, nxt)
            frontier.append(nxt)

    if trace:
        print(f"  (BFS expanded {expanded} states, frontier exhausted)")
    return None, None


def _reconstruct(came_from, state):
    plan, states = [], [state]
    prev, action = came_from[state]
    while action is not None:
        plan.append(action)
        states.append(prev)
        prev, action = came_from[prev]
    plan.reverse()
    states.reverse()
    return plan, states


# ---------------------------------------------------------------------------
# The warehouse delivery problem (Section 3)
# ---------------------------------------------------------------------------
def warehouse_actions(include_pickup=True):
    """Build the robot's action set.

    Locations A, B, C with connections A-B and B-C (so A and C are NOT
    directly connected).  Set include_pickup=False for Test B (impossible).
    """
    moves = []
    for x, y in [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]:
        moves.append(Action(
            name=f"Move({x},{y})",
            pre_pos=[f"At(Robot,{x})"],
            eff_pos=[f"At(Robot,{y})"],
            eff_neg=[f"At(Robot,{x})"],
        ))

    pickups = []
    if include_pickup:
        for loc in ["A", "B", "C"]:
            pickups.append(Action(
                name=f"PickUp(Package,{loc})",
                pre_pos=[f"At(Robot,{loc})", f"At(Package,{loc})"],
                eff_pos=["Holding(Package)"],
                eff_neg=[f"At(Package,{loc})"],
            ))

    drops = []
    for loc in ["A", "B", "C"]:
        drops.append(Action(
            name=f"Drop(Package,{loc})",
            pre_pos=[f"At(Robot,{loc})", "Holding(Package)"],
            eff_pos=[f"At(Package,{loc})"],
            eff_neg=["Holding(Package)"],
        ))

    return moves + pickups + drops


def warehouse_problem(include_pickup=True):
    return PlanningProblem(
        initial=["At(Robot,A)", "At(Package,A)"],
        actions=warehouse_actions(include_pickup),
        goal=["At(Package,C)"],
    )


def show_plan(problem, plan, states):
    if plan is None:
        print("  RESULT: No plan found.")
        return
    print(f"  RESULT: plan of length {len(plan)} found.")
    print(f"  Plan  : {[a.name for a in plan]}")
    print("  State trace:")
    print(f"    S0 : {sorted(states[0])}")
    for i, (a, s) in enumerate(zip(plan, states[1:]), start=1):
        print(f"    S{i} : after {a.name:<20} -> {sorted(s)}")


if __name__ == "__main__":
    print("Warehouse delivery: move Package from A to C.\n")
    prob = warehouse_problem()
    plan, states = plan_bfs(prob, trace=True)
    show_plan(prob, plan, states)
