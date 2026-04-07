from __future__ import annotations

from typing import Iterable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_enemy_base(env: AgeGridEnv, faction: str) -> tuple[int, int]:
    enemy = next(name for name in env.factions if name != faction)
    return env.bases[enemy].position


def _nearest_resource(env: AgeGridEnv, pos: tuple[int, int]) -> tuple[int, int] | None:
    nodes = [r for r in env.resources if r.remaining > 0]
    if not nodes:
        return None
    return min(nodes, key=lambda r: abs(r.position[0] - pos[0]) + abs(r.position[1] - pos[1])).position


def _buildings(env: AgeGridEnv, faction: str) -> set[str]:
    return {b.building_type for b in env.buildings if b.faction == faction and b.hp > 0}


def _unit_count(env: AgeGridEnv, faction: str, unit_type: str) -> int:
    return sum(1 for u in env.units if u.faction == faction and u.unit_type == unit_type)


def _actions_of_kind(actions: Iterable[Action], kind: str) -> list[Action]:
    return [action for action in actions if action[0] == kind]


def _nearest_build_site(env: AgeGridEnv, faction: str, worker_pos: tuple[int, int]) -> tuple[int, int] | None:
    base = env.bases[faction].position
    occupied = env._occupied_positions()
    candidates = [
        (base[0] + 2, base[1]),
        (base[0], base[1] + 2),
        (base[0] - 2, base[1]),
        (base[0], base[1] - 2),
        (base[0] + 1, base[1] + 1),
        (base[0] - 1, base[1] + 1),
        (base[0] + 1, base[1] - 1),
        (base[0] - 1, base[1] - 1),
    ]
    valid = [pos for pos in candidates if env._in_bounds(pos) and pos not in occupied]
    if not valid:
        return None
    return min(valid, key=lambda pos: _distance(worker_pos, pos))


class HeuristicAgent:
    """
    Progression-focused baseline:
    - gather and expand workers early
    - research along a fixed economic -> military ladder
    - build a barracks and train soldiers / archers
    - attack nearby targets, otherwise move military toward the enemy base
    """

    def __init__(self, desired_workers: int = 3):
        self.desired_workers = desired_workers
        self._military_targets: dict[int, tuple[int, int]] = {}

    def act(self, env: AgeGridEnv) -> Action | None:
        faction = env.factions[env.current_player]
        state = env.faction_state(faction)
        legal = env.legal_actions(faction)
        if not legal:
            return None

        attacks = _actions_of_kind(legal, "attack")
        if attacks:
            return attacks[0]

        base_attacks = _actions_of_kind(legal, "attack_base")
        if base_attacks:
            return base_attacks[0]

        workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
        buildings = _buildings(env, faction)

        if _unit_count(env, faction, "worker") < self.desired_workers:
            spawn_actions = _actions_of_kind(legal, "spawn_worker")
            if spawn_actions:
                return spawn_actions[0]

        if "storehouse" not in buildings:
            desired = ("research", "mining")
            if "mining" not in state.techs_unlocked and desired in legal:
                return desired
            build_actions = [a for a in legal if a[0] == "build" and a[2] == "storehouse"]
            if build_actions:
                return build_actions[0]

        if "barracks" not in buildings:
            desired = ("research", "bronze_working")
            if "bronze_working" not in state.techs_unlocked and desired in legal:
                return desired
            build_actions = [a for a in legal if a[0] == "build" and a[2] == "barracks"]
            if build_actions:
                return build_actions[0]

        research_priority = ["fletching", "masonry"]
        for tech_id in research_priority:
            if tech_id not in state.techs_unlocked:
                desired = ("research", tech_id)
                if desired in legal:
                    return desired

        if _unit_count(env, faction, "soldier") < 2:
            desired = ("train", "soldier")
            if desired in legal:
                return desired

        desired_archers = 1 if "fletching" in state.techs_unlocked else 0
        if _unit_count(env, faction, "archer") < desired_archers:
            desired = ("train", "archer")
            if desired in legal:
                return desired

        gathers = _actions_of_kind(legal, "gather")
        if gathers:
            return gathers[0]

        military = [u for u in env.units if u.faction == faction and u.attack_damage > 0]
        if military:
            enemy_units = [u for u in env.units if u.faction != faction]
            for unit in military:
                target = self._military_targets.get(unit.id)
                if target is None or all(enemy.position != target for enemy in enemy_units):
                    if enemy_units:
                        target = min(enemy_units, key=lambda enemy: _distance(unit.position, enemy.position)).position
                    else:
                        target = _nearest_enemy_base(env, faction)
                    self._military_targets[unit.id] = target

                action = ("move_towards", unit.id, target)
                if action in legal:
                    return action

        for worker in workers:
            target = _nearest_resource(env, worker.position)
            if target is not None:
                action = ("move_towards", worker.id, target)
                if action in legal:
                    return action

        build_site_needed = "storehouse" not in buildings or "barracks" not in buildings
        if build_site_needed:
            for worker in workers:
                target = _nearest_build_site(env, faction, worker.position)
                if target is None:
                    continue
                action = ("move_towards", worker.id, target)
                if action in legal:
                    return action

        return None
