# LLM Prompt Appendix (Task 2)

The design (state = set of propositions; actions with positive/negative
preconditions and effects; BFS over states) was written **before** prompting.

## Prompt used

> I want to implement a simple planning agent in Python. Represent a state as a
> set of logical propositions. Each action should contain: a name; positive
> preconditions; negative preconditions; positive effects; negative effects. An
> action is applicable if all of its preconditions are satisfied by the current
> state. When an action is applied: remove its negative effects from the state,
> then add its positive effects. Use breadth-first search to find a sequence of
> actions that achieves a specified goal. The program should also: detect when
> no plan exists; print the resulting sequence of actions; and print the states
> reached after each action. Explain the implementation and identify any
> assumptions you make.

## Corrections / hardening before accepting the code

1. **Negative preconditions actually checked.** Ensured `applicable` tests both
   `pre_pos ⊆ state` *and* `pre_neg ∩ state = ∅` (a naive version checked only
   the positive ones).
2. **Effect order.** Applied negative effects (delete) *before* positive effects
   (add), matching STRIPS semantics.
3. **Failure detection + no cycling.** Added a `visited` set over states and an
   explicit "No plan found" return when the frontier empties, so impossible
   problems terminate instead of looping.
4. **Goal test is subset-based** (`goal ⊆ state`), so extra true facts don't
   block goal recognition and the robot-at-C vs package-at-C distinction holds.

All of these were validated with `test_planner.py` (Tests A–D).
