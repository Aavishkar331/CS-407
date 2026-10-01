% ===========================================================================
% Optional Extension -- Prolog as a Logical Verifier
% Artificial Intelligence, Laboratory: Logical Planning (Tasks 6, 7, 8)
%
% Load in SWI-Prolog:   ?- [planner].
% Then run the queries noted under each task.
% ===========================================================================

% ---------------------------------------------------------------------------
% Task 6 -- Warehouse connectivity (facts) and the can_move rule
% ---------------------------------------------------------------------------
% Facts: the robot can move between directly connected locations.
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: a move is possible exactly when the two locations are connected.
% This Prolog rule corresponds to the logical implication
%     Connected(X, Y) -> CanMove(X, Y).
can_move(X, Y) :-
    connected(X, Y).

% Expected results (Task 6):
%   ?- can_move(a, b).   ->  true    (connected(a,b) is a fact)
%   ?- can_move(a, c).   ->  false   (no fact connected(a,c); a and c are
%                                     only connected *through* b)

% ---------------------------------------------------------------------------
% Task 7 -- Using Prolog to check a proposed plan
% ---------------------------------------------------------------------------
% A single proposed move is valid iff the locations are connected.
valid_move(X, Y) :-
    connected(X, Y).

% Expected results (Task 7):
%   ?- valid_move(a, b).   ->  true
%   ?- valid_move(b, c).   ->  true
%   ?- valid_move(a, c).   ->  false   % the Python planner's Move(a,c)
%                                       % is NOT supported by the warehouse
%                                       % knowledge base.
%
% Challenge: the Python program *generated* the candidate Move(a, c); Prolog,
% used independently, *verifies* it against the warehouse facts and rejects it.
% This is the "Generate -> Independent verification" architecture.
%
% (Optional) check a whole path of connected moves:
valid_path([_]).
valid_path([X, Y | Rest]) :-
    valid_move(X, Y),
    valid_path([Y | Rest]).
%   ?- valid_path([a, b, c]).   ->  true
%   ?- valid_path([a, c]).      ->  false

% ---------------------------------------------------------------------------
% Task 8 -- Connecting Prolog to logical reasoning (chaining rules)
% ---------------------------------------------------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% Expected result (Task 8):
%   ?- reduce_speed.   ->  true
%
% Logical reasoning behind it (Fact => Rule => Rule => Conclusion):
%     wet_road                                   (fact, given)
%     wet_road -> slippery                       (rule)   => slippery
%     slippery -> reduce_speed                   (rule)   => reduce_speed
% so reduce_speed follows by forward chaining from the single fact wet_road.
