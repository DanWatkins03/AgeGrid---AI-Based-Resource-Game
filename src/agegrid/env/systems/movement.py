from __future__ import annotations
from collections import deque
from typing import Tuple

Position = Tuple[int,int]


# Move around the x and y axis
_DELTAS: dict[str, Position] = {
    "up" : (0,-1),
    "down" : (0,1),
    "left" : (-1,-0),
    "right" : (1,-0)
}

_NEIGHBOR_ORDER: tuple[tuple[str, Position], ...] = (
    ("right", (1, 0)),
    ("down", (0, 1)),
    ("left", (-1, 0)),
    ("up", (0, -1)),
)

# Moves the unit, verifies its a legal move
def move_unit(env, unit_id: int, direction: str) -> bool:
    unit = next((u for u in env.units if u.id == unit_id), None)
    
    # Check if unit exists
    if unit is None:
        return False
    
    # Check if direction is valid
    if direction not in _DELTAS:
        return False
    
    dx, dy = _DELTAS[direction]
    new_pos = (unit.position[0] + dx, unit.position[1] + dy)

    # More validation checks
    if not env._in_bounds(new_pos):
        return False
    if new_pos in env._occupied_positions():
        return False
    unit.position = new_pos
    return True


def can_move_unit(env, unit_id: int, direction: str) -> bool:
    unit = next((u for u in env.units if u.id == unit_id), None)

    if unit is None:
        return False
    if direction not in _DELTAS:
        return False

    dx, dy = _DELTAS[direction]
    new_pos = (unit.position[0] + dx, unit.position[1] + dy)
    if not env._in_bounds(new_pos):
        return False
    if new_pos in env._occupied_positions():
        return False
    return True


def can_move_towards(env, unit_id: int, target: Position) -> bool:
    unit = next((u for u in env.units if u.id == unit_id), None)
    if unit is None:
        return False
    probe_position = unit.position
    steps = max(1, getattr(unit, "move_steps", 1))

    for _ in range(steps):
        next_pos = _next_step_toward(env, unit_id, target, start=probe_position)
        if next_pos is None:
            return probe_position != unit.position
        probe_position = next_pos
        if probe_position == target:
            return True
    return probe_position != unit.position


def _next_step_toward(env, unit_id: int, target: Position, start: Position | None = None) -> Position | None:
    unit = next((u for u in env.units if u.id == unit_id), None)
    current_position = unit.position if unit is not None and start is None else start
    if unit is None or current_position is None or current_position == target:
        return None

    occupied = env._occupied_positions() - {unit.position}
    queue: deque[Position] = deque([current_position])
    came_from: dict[Position, Position | None] = {current_position: None}

    while queue:
        current = queue.popleft()
        if current == target:
            break

        for _, (dx, dy) in _NEIGHBOR_ORDER:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in came_from:
                continue
            if not env._in_bounds(nxt):
                continue
            if nxt in occupied and nxt != target:
                continue
            came_from[nxt] = current
            queue.append(nxt)

    if target not in came_from:
        return None

    step = target
    while came_from[step] != current_position:
        parent = came_from[step]
        if parent is None:
            return None
        step = parent
    return step


def move_towards(env, unit_id: int, target: Position) -> bool:
    unit = next((u for u in env.units if u.id == unit_id), None)
    if unit is None:
        return False

    moved = False
    for _ in range(max(1, getattr(unit, "move_steps", 1))):
        next_pos = _next_step_toward(env, unit_id, target)
        if next_pos is None:
            break
        unit.position = next_pos
        moved = True
        if unit.position == target:
            break
    return moved



