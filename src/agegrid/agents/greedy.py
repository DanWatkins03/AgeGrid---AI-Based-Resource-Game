from __future__ import annotations
from typing import Tuple

from src.agegrid.env.agegrid_env import AgeGridEnv

Position = Tuple[int, int]


def _nearest_resource_pos(env: AgeGridEnv, pos: Position) -> Position | None:
    nodes = [r for r in env.resources if r.remaining > 0]
    if not nodes:
        return None

    def dist(p: Position) -> int:
        return abs(p[0] - pos[0]) + abs(p[1] - pos[1])

    return min(nodes, key=lambda r: dist(r.position)).position


class GreedyAgent:
    """
    Simple baseline policy:
    - Spawn until N workers
    - If any worker is on a resource: gather
    - Else move one worker toward nearest resource
    """

    def __init__(self, desired_workers: int = 2):
        self.desired_workers = desired_workers

        # Tiny bit of state to avoid always picking the same worker
        self._last_seen_key: tuple[int, int] | None = None  # (turn, current_player)
        self._rr_index: int = 0

    def act(self, env: AgeGridEnv) -> tuple | None:
        faction = env.factions[env.current_player]

        # Reset round-robin index at the start of each faction-turn
        key = (env.turn, env.current_player)
        if key != self._last_seen_key:
            self._last_seen_key = key
            self._rr_index = 0

        workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
        if not workers:
            return None

        # Spawn up to desired_workers
        if len(workers) < self.desired_workers and env.bank[faction] >= env.config.worker_spawn_cost:
            return ("spawn_worker",)

        # Gather if possible (any worker on resource)
        for w in workers:
            if env._resource_at(w.position) is not None:
                return ("gather", w.id)

        # Otherwise move a worker (round-robin so we don't always pick workers[0])
        w = workers[self._rr_index % len(workers)]
        self._rr_index += 1

        target = _nearest_resource_pos(env, w.position)
        if target is None:
            return None

        return ("move_towards", w.id, target)