from __future__ import annotations
from typing import Tuple

Position = Tuple[int,int]


# Move around the x and y axis
_DELTAS: dict[str, Position] = {
    "up" : (0,-1),
    "down" : (0,1),
    "left" : (-1,-0),
    "right" : (1,-0)
}

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

def move_towards(env, unit_id: int, target: Position) -> bool:
    unit=next((u for u in env.units if u.id == unit_id), None)

    if unit is None:
        return False
    
    x, y = unit.position
    tx, ty = target

    if tx > x and move_unit(env, unit_id, "right"):
        return True
    if tx < x and move_unit(env, unit_id, "left"):
        return True
    if ty > y and move_unit(env, unit_id, "down"):
        return True
    if ty < y and move_unit(env, unit_id, "up"):
        return True
    
    return False



