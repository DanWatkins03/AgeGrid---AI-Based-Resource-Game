from __future__ import annotations
from typing import Tuple

from src.agegrid.env.actions import Action
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
    Economy-first baseline policy:
    - attack when possible
    - spawn workers up to N
    - gather nearby resources
    - follow a simple tech/build/train ladder
    - otherwise move toward the nearest resource or enemy base
    """

    def __init__(self, desired_workers: int = 2):
        self.desired_workers = desired_workers

        # Tiny bit of state to avoid always picking the same worker
        self._last_seen_key: tuple[int, int] | None = None  # (turn, current_player)
        self._rr_index: int = 0

    def act(self, env: AgeGridEnv) -> Action | None:
        faction = env.factions[env.current_player]
        legal = env.legal_actions(faction)
        if not legal:
            return None

        # Reset round-robin index at the start of each faction-turn
        key = (env.turn, env.current_player)
        if key != self._last_seen_key:
            self._last_seen_key = key
            self._rr_index = 0

        for action_kind in ("attack", "attack_base"):
            candidate = next((action for action in legal if action[0] == action_kind), None)
            if candidate is not None:
                return candidate

        workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
        if not workers:
            return None

        # Spawn up to desired_workers
        if len(workers) < self.desired_workers and ("spawn_worker",) in legal:
            return ("spawn_worker",)

        # Gather if possible (any worker on resource)
        for w in workers:
            if env._resource_at(w.position) is not None:
                return ("gather", w.id)

        buildings = {b.building_type for b in env.buildings if b.faction == faction and b.hp > 0}
        state = env.faction_state(faction)

        if "mining" not in state.techs_unlocked and ("research", "mining") in legal:
            return ("research", "mining")
        if "storehouse" not in buildings:
            build_storehouse = next((a for a in legal if a[0] == "build" and a[2] == "storehouse"), None)
            if build_storehouse is not None:
                return build_storehouse
        if "bronze_working" not in state.techs_unlocked and ("research", "bronze_working") in legal:
            return ("research", "bronze_working")
        if "barracks" not in buildings:
            build_barracks = next((a for a in legal if a[0] == "build" and a[2] == "barracks"), None)
            if build_barracks is not None:
                return build_barracks
        if ("train", "soldier") in legal:
            return ("train", "soldier")
        if "fletching" not in state.techs_unlocked and ("research", "fletching") in legal:
            return ("research", "fletching")
        if ("train", "archer") in legal:
            return ("train", "archer")

        # Otherwise move a worker (round-robin so we don't always pick workers[0])
        w = workers[self._rr_index % len(workers)]
        self._rr_index += 1

        military = [u for u in env.units if u.faction == faction and u.attack_damage > 0]
        if military:
            soldier = military[self._rr_index % len(military)]
            enemy = next(name for name in env.factions if name != faction)
            move_action = ("move_towards", soldier.id, env.bases[enemy].position)
            if move_action in legal:
                return move_action

        target = _nearest_resource_pos(env, w.position)
        if target is None:
            return next((action for action in legal if action[0] == "move_towards"), None)

        return ("move_towards", w.id, target)
