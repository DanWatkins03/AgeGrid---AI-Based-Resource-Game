from __future__ import annotations

from src.agegrid.env import hexgrid

Position = tuple[int, int]


def _approach_positions(target: Position, env) -> list[Position]:
    return [pos for pos in hexgrid.neighbors(target) if env._in_bounds(pos)]


def _resolved_target(env, unit, target: Position) -> Position | None:
    occupied = env._occupied_positions() - {unit.position}
    if target not in occupied:
        return target

    candidates = [pos for pos in _approach_positions(target, env) if pos not in occupied]
    if not candidates:
        return None
    return min(candidates, key=lambda pos: (hexgrid.distance(unit.position, pos), hexgrid.distance(pos, target)))


def move_unit(env, unit_id: int, direction: str) -> bool:
    unit = env.get_unit(unit_id)
    if unit is None:
        return False

    deltas = hexgrid.direction_map(unit.position[1])
    if direction not in deltas:
        return False

    dx, dy = deltas[direction]
    new_pos = (unit.position[0] + dx, unit.position[1] + dy)
    if not env._in_bounds(new_pos):
        return False
    if new_pos in env._occupied_positions():
        return False
    unit.position = new_pos
    return True


def can_move_unit(env, unit_id: int, direction: str) -> bool:
    unit = env.get_unit(unit_id)
    if unit is None:
        return False

    deltas = hexgrid.direction_map(unit.position[1])
    if direction not in deltas:
        return False

    dx, dy = deltas[direction]
    new_pos = (unit.position[0] + dx, unit.position[1] + dy)
    if not env._in_bounds(new_pos):
        return False
    if new_pos in env._occupied_positions():
        return False
    return True


def can_move_towards(env, unit_id: int, target: Position) -> bool:
    unit = env.get_unit(unit_id)
    if unit is None:
        return False
    resolved_target = _resolved_target(env, unit, target)
    if resolved_target is None:
        return False

    probe_position = unit.position
    steps = max(1, getattr(unit, "move_steps", 1))
    blocked = env._occupied_positions() - {unit.position}
    for _ in range(steps):
        next_pos = hexgrid.connected_step_toward(
            probe_position,
            resolved_target,
            env.config.width,
            env.config.height,
            blocked,
        )
        if next_pos is None:
            return probe_position != unit.position
        probe_position = next_pos
        if probe_position == resolved_target:
            return True
    return probe_position != unit.position


def move_towards(env, unit_id: int, target: Position) -> bool:
    unit = env.get_unit(unit_id)
    if unit is None:
        return False
    resolved_target = _resolved_target(env, unit, target)
    if resolved_target is None:
        return False

    moved = False
    for _ in range(max(1, getattr(unit, "move_steps", 1))):
        next_pos = hexgrid.connected_step_toward(
            unit.position,
            resolved_target,
            env.config.width,
            env.config.height,
            env._occupied_positions() - {unit.position},
        )
        if next_pos is None:
            break
        unit.position = next_pos
        moved = True
        if unit.position == resolved_target:
            break
    return moved
