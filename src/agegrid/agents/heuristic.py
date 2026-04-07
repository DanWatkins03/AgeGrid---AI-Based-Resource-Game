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
    faction = env.factions[env.current_player]
    nodes = env.visible_resources(faction)
    if not nodes:
        return None
    return min(nodes, key=lambda r: abs(r.position[0] - pos[0]) + abs(r.position[1] - pos[1])).position


def _buildings(env: AgeGridEnv, faction: str) -> set[str]:
    return {b.building_type for b in env.buildings if b.faction == faction and b.hp > 0}


def _unit_count(env: AgeGridEnv, faction: str, unit_type: str) -> int:
    return sum(1 for u in env.units if u.faction == faction and u.unit_type == unit_type)


def _actions_of_kind(actions: Iterable[Action], kind: str) -> list[Action]:
    return [action for action in actions if action[0] == kind]


def _threatened_positions(env: AgeGridEnv, faction: str) -> list[tuple[int, int]]:
    positions = [env.bases[faction].position]
    positions.extend(u.position for u in env.units if u.faction == faction and u.unit_type == "worker")
    positions.extend(b.position for b in env.buildings if b.faction == faction and b.hp > 0)
    return positions


def _threat_score(env: AgeGridEnv, faction: str, enemy_pos: tuple[int, int]) -> int:
    threatened = _threatened_positions(env, faction)
    if not threatened:
        return 99
    return min(_distance(enemy_pos, pos) for pos in threatened)


def _defensive_targets(env: AgeGridEnv, faction: str) -> list:
    return [
        enemy
        for enemy in env.units
        if enemy.faction != faction and enemy.attack_damage > 0 and _threat_score(env, faction, enemy.position) <= 4
    ]


def _emergency_targets(env: AgeGridEnv, faction: str) -> list:
    return [
        enemy
        for enemy in env.units
        if enemy.faction != faction and enemy.attack_damage > 0 and _threat_score(env, faction, enemy.position) <= 2
    ]


def _in_defense_mode(env: AgeGridEnv, faction: str) -> bool:
    return bool(_emergency_targets(env, faction))


def defense_mode_active(env: AgeGridEnv, faction: str) -> bool:
    return _in_defense_mode(env, faction)


def _total_force(env: AgeGridEnv, faction: str) -> int:
    return sum(max(1, unit.attack_damage) + unit.hp // 4 for unit in env.units if unit.faction == faction and unit.attack_damage > 0)


def _push_mode_active(env: AgeGridEnv, faction: str) -> bool:
    if _in_defense_mode(env, faction) or _defensive_targets(env, faction):
        return False
    military = [unit for unit in env.units if unit.faction == faction and unit.attack_damage > 0]
    if len(military) < 2:
        return False
    enemy = next(name for name in env.factions if name != faction)
    home_friendly, home_enemy = _home_force_balance(env, faction)
    if home_enemy > 0:
        return False
    return _total_force(env, faction) >= _total_force(env, enemy) + 2 or (len(military) >= 4 and home_friendly >= 2)


def push_mode_active(env: AgeGridEnv, faction: str) -> bool:
    return _push_mode_active(env, faction)


def army_plan(env: AgeGridEnv, faction: str) -> str:
    military = [unit for unit in env.units if unit.faction == faction and unit.attack_damage > 0]
    if _in_defense_mode(env, faction) or _defensive_targets(env, faction):
        return "Hold"
    if _push_mode_active(env, faction):
        return "Push"
    if _rally_anchor(env, faction, military) is not None:
        return "Rally"
    return "Hold"


def threat_level(env: AgeGridEnv, faction: str) -> str:
    if _emergency_targets(env, faction):
        return "Emergency"
    if _defensive_targets(env, faction):
        return "Guarded"
    return "Stable"


def army_strength_near_base(env: AgeGridEnv, faction: str, radius: int = 4) -> tuple[int, int]:
    base_pos = env.bases[faction].position
    enemy = next(name for name in env.factions if name != faction)
    return _local_force(env, base_pos, faction, radius=radius), _local_force(env, base_pos, enemy, radius=radius)


def unit_composition(env: AgeGridEnv, faction: str) -> dict[str, int]:
    return {
        unit_type: _unit_count(env, faction, unit_type)
        for unit_type in ("worker", "soldier", "archer", "horseman")
    }


def _home_anchor(env: AgeGridEnv, faction: str) -> tuple[int, int]:
    workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
    if workers:
        return min(workers, key=lambda worker: _distance(worker.position, env.bases[faction].position)).position
    return env.bases[faction].position


def _defender_slots(env: AgeGridEnv, faction: str, military: list) -> int:
    if _in_defense_mode(env, faction):
        return min(2, len(military))
    if _defensive_targets(env, faction):
        return min(1, len(military))
    if _push_mode_active(env, faction):
        return 0
    return 1 if len(military) >= 4 else 0


def _home_guard_target(env: AgeGridEnv, faction: str, unit) -> tuple[int, int] | None:
    anchor = _home_anchor(env, faction)
    candidates = [
        anchor,
        (anchor[0] + 1, anchor[1]),
        (anchor[0] - 1, anchor[1]),
        (anchor[0], anchor[1] + 1),
        (anchor[0], anchor[1] - 1),
        (anchor[0] + 1, anchor[1] + 1),
        (anchor[0] - 1, anchor[1] + 1),
        (anchor[0] + 1, anchor[1] - 1),
        (anchor[0] - 1, anchor[1] - 1),
    ]
    valid = [
        pos
        for pos in candidates
        if env._in_bounds(pos) and pos != env.bases[faction].position and pos != unit.position
    ]
    if not valid:
        return None
    return min(valid, key=lambda pos: _distance(unit.position, pos))


def _military_role(env: AgeGridEnv, faction: str, unit, military: list) -> str:
    if len(military) == 1:
        return "line"
    if unit.unit_type == "horseman":
        return "raider"
    if not military:
        return "line"
    defender_slots = _defender_slots(env, faction, military)
    if defender_slots <= 0:
        return "line"
    home_anchor = _home_anchor(env, faction)
    defenders = sorted(
        [member for member in military if member.unit_type != "horseman"],
        key=lambda member: (_distance(member.position, home_anchor), member.id),
    )[:defender_slots]
    if any(unit.id == defender.id for defender in defenders):
        return "defender"
    return "line"


def _enemy_cavalry_nearby(env: AgeGridEnv, faction: str, radius: int = 4) -> list:
    defended = _threatened_positions(env, faction)
    return [
        enemy
        for enemy in env.units
        if enemy.faction != faction
        and enemy.unit_type == "horseman"
        and any(_distance(enemy.position, pos) <= radius for pos in defended)
    ]


def _worker_retreat_target(env: AgeGridEnv, faction: str, worker) -> tuple[int, int] | None:
    base = env.bases[faction].position
    if _distance(worker.position, base) == 0:
        return None
    return base


def _attack_priority(env: AgeGridEnv, faction: str, action: Action) -> tuple[int, int, int, int]:
    target_id = action[2]
    target = next((u for u in env.units if u.id == target_id), None)
    if target is None:
        return (99, 99, 99, 99)
    emergency_rank = 0 if target.attack_damage > 0 and _threat_score(env, faction, target.position) <= 2 else 1
    threat_rank = 0 if target.attack_damage > 0 and _threat_score(env, faction, target.position) <= 4 else 1
    type_rank = {"horseman": 0, "archer": 1, "soldier": 2, "worker": 3}.get(target.unit_type, 4)
    return (emergency_rank, threat_rank, type_rank, target.hp)


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


def _visible_resource_positions(env: AgeGridEnv, faction: str, resource_type: str) -> set[tuple[int, int]]:
    return {
        resource.position
        for resource in env.visible_resources(faction)
        if resource.resource_type == resource_type
    }


def _build_actions_for(legal: list[Action], building_type: str) -> list[Action]:
    return [action for action in legal if action[0] == "build" and action[2] == building_type]


def _choose_closest_build_action(env: AgeGridEnv, workers: list, actions: list[Action]) -> Action | None:
    if not actions:
        return None
    return min(
        actions,
        key=lambda action: _distance(
            next(worker for worker in workers if worker.id == action[1]).position,
            action[3],
        ),
    )


def _local_force(env: AgeGridEnv, center: tuple[int, int], faction: str, radius: int = 3) -> int:
    return sum(
        max(1, unit.attack_damage) + unit.hp // 4
        for unit in env.units
        if unit.faction == faction and unit.attack_damage > 0 and _distance(unit.position, center) <= radius
    )


def _support_count(env: AgeGridEnv, center: tuple[int, int], faction: str, radius: int = 2) -> int:
    return sum(
        1
        for unit in env.units
        if unit.faction == faction and unit.attack_damage > 0 and _distance(unit.position, center) <= radius
    )


def _rally_anchor(env: AgeGridEnv, faction: str, military: list) -> tuple[int, int] | None:
    line_units = [unit for unit in military if unit.unit_type != "horseman"]
    if len(line_units) < 2:
        return None
    enemy_base = _nearest_enemy_base(env, faction)
    frontline = min(line_units, key=lambda unit: (_distance(unit.position, enemy_base), unit.id))
    return frontline.position


def _army_ready_to_push(env: AgeGridEnv, faction: str, target: tuple[int, int], minimum_units: int = 2) -> bool:
    enemy = next(name for name in env.factions if name != faction)
    friendly_support = _support_count(env, target, faction)
    enemy_support = _local_force(env, target, enemy)
    return friendly_support >= minimum_units and _local_force(env, target, faction) >= enemy_support + 1


def _home_force_balance(env: AgeGridEnv, faction: str, radius: int = 4) -> tuple[int, int]:
    base_pos = env.bases[faction].position
    enemy = next(name for name in env.factions if name != faction)
    return _local_force(env, base_pos, faction, radius), _local_force(env, base_pos, enemy, radius)


def _preferred_emergency_train(legal: list[Action], nearby_cavalry: list) -> Action | None:
    train_actions = [action for action in legal if action[0] == "train" and action[1] in {"soldier", "horseman", "archer"}]
    if not train_actions:
        return None
    if nearby_cavalry:
        preferred = {"archer": 0, "soldier": 1, "horseman": 2}
    else:
        preferred = {"soldier": 0, "archer": 1, "horseman": 2}
    return min(train_actions, key=lambda action: preferred.get(action[1], 9))


def _viable_unit_attack(env: AgeGridEnv, faction: str, action: Action) -> bool:
    target = next((u for u in env.units if u.id == action[2]), None)
    attacker = next((u for u in env.units if u.id == action[1]), None)
    if target is None or attacker is None:
        return False
    if target.attack_damage == 0:
        return True
    if _threat_score(env, faction, target.position) <= 2:
        return True
    friendly_force = _local_force(env, target.position, faction)
    enemy_force = _local_force(env, target.position, target.faction)
    return friendly_force >= enemy_force or target.hp <= attacker.attack_damage


def _viable_base_attack(env: AgeGridEnv, faction: str, action: Action) -> bool:
    attacker = next((u for u in env.units if u.id == action[1]), None)
    if attacker is None:
        return False
    enemy = action[2]
    base_pos = env.bases[enemy].position
    nearby_friendlies = [
        unit
        for unit in env.units
        if unit.faction == faction and unit.attack_damage > 0 and _distance(unit.position, base_pos) <= 3
    ]
    friendly_force = _local_force(env, base_pos, faction)
    enemy_force = _local_force(env, base_pos, enemy)
    base_threat = env.config.base_attack_damage
    if "masonry" in env.faction_state(enemy).techs_unlocked:
        base_threat += env.config.masonry_base_attack_bonus
    if env.bases[enemy].hp <= attacker.attack_damage:
        return True
    if len(nearby_friendlies) < 2:
        return False
    return friendly_force >= enemy_force + base_threat


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

    def _choose_research(self, env: AgeGridEnv, faction: str, legal: list[Action], buildings: set[str]) -> Action | None:
        state = env.faction_state(faction)
        if state.tech_in_progress is not None:
            return None
        if "mining" in state.techs_unlocked and "storehouse" not in buildings:
            return None
        if "bronze_working" in state.techs_unlocked and "barracks" not in buildings:
            return None

        visible_stone = bool(_visible_resource_positions(env, faction, "stone"))
        visible_horses = bool(_visible_resource_positions(env, faction, "horses"))
        research_order = [
            "mining",
            "bronze_working",
            "masonry" if visible_stone else "horsemanship",
            "horsemanship" if visible_stone else "masonry",
            "fletching",
        ]
        for tech_id in research_order:
            action = ("research", tech_id)
            if action in legal and tech_id not in state.techs_unlocked:
                return action
        return None

    def act(self, env: AgeGridEnv) -> Action | None:
        faction = env.factions[env.current_player]
        state = env.faction_state(faction)
        legal = env.legal_actions(faction)
        if not legal:
            return None
        defense_mode = _in_defense_mode(env, faction)
        workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
        buildings = _buildings(env, faction)
        military = [u for u in env.units if u.faction == faction and u.attack_damage > 0]
        emergency_targets = _emergency_targets(env, faction)
        nearby_cavalry = _enemy_cavalry_nearby(env, faction)
        rally_anchor = _rally_anchor(env, faction, military)
        spawn_actions = _actions_of_kind(legal, "spawn_worker")
        emergency_train = _preferred_emergency_train(legal, nearby_cavalry)
        home_friendly_force, home_enemy_force = _home_force_balance(env, faction)
        push_mode = _push_mode_active(env, faction)
        collapse_mode = not workers
        last_stand = defense_mode and home_enemy_force > home_friendly_force
        desired_home_force = 2 if defense_mode or home_enemy_force > 0 else 1

        if not defense_mode:
            research_action = self._choose_research(env, faction, legal, buildings)
            if research_action is not None:
                return research_action

        if collapse_mode:
            if last_stand and emergency_train is not None:
                return emergency_train
            if spawn_actions:
                return spawn_actions[0]
            if emergency_train is not None:
                return emergency_train

        attacks = _actions_of_kind(legal, "attack")
        if attacks:
            viable_attacks = [action for action in attacks if _viable_unit_attack(env, faction, action)]
            chosen_attacks = viable_attacks or attacks
            return min(chosen_attacks, key=lambda action: _attack_priority(env, faction, action))

        if defense_mode and military:
            role_order = {"defender": 0, "line": 1, "raider": 2}
            ordered_military = sorted(
                military,
                key=lambda unit: (role_order.get(_military_role(env, faction, unit, military), 3), unit.id),
            )
            for unit in ordered_military:
                if emergency_targets:
                    target = min(
                        emergency_targets,
                        key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp),
                    ).position
                    action = ("move_towards", unit.id, target)
                    if action in legal:
                        return action
            if (home_friendly_force < desired_home_force or last_stand) and emergency_train is not None:
                return emergency_train
            fallback_moves = _actions_of_kind(legal, "move_towards")
            defensive_positions = {
                env.bases[faction].position,
                *(u.position for u in workers),
                *(b.position for b in env.buildings if b.faction == faction and b.hp > 0),
            }
            guarded_moves = [action for action in fallback_moves if action[2] in defensive_positions]
            if guarded_moves:
                return guarded_moves[0]
            for worker in workers:
                retreat = _worker_retreat_target(env, faction, worker)
                if retreat is None:
                    continue
                action = ("move_towards", worker.id, retreat)
                if action in legal:
                    return action
            return None

        base_attacks = _actions_of_kind(legal, "attack_base")
        if base_attacks:
            viable_base_attacks = [action for action in base_attacks if _viable_base_attack(env, faction, action)]
            if viable_base_attacks:
                return viable_base_attacks[0]

        if nearby_cavalry:
            archer_action = ("train", "archer")
            if archer_action in legal:
                return archer_action

        if home_enemy_force > home_friendly_force and emergency_train is not None:
            return emergency_train

        if not defense_mode and _unit_count(env, faction, "worker") < self.desired_workers:
            if spawn_actions:
                return spawn_actions[0]

        if not defense_mode and "storehouse" not in buildings:
            build_action = _choose_closest_build_action(env, workers, _build_actions_for(legal, "storehouse"))
            if build_action is not None:
                return build_action

        if not defense_mode and "barracks" not in buildings:
            build_action = _choose_closest_build_action(env, workers, _build_actions_for(legal, "barracks"))
            if build_action is not None:
                return build_action

        if not defense_mode and "quarry" not in buildings and "masonry" in state.techs_unlocked:
            build_action = _choose_closest_build_action(env, workers, _build_actions_for(legal, "quarry"))
            if build_action is not None:
                return build_action

        if not defense_mode and "stable" not in buildings and "horsemanship" in state.techs_unlocked:
            build_action = _choose_closest_build_action(env, workers, _build_actions_for(legal, "stable"))
            if build_action is not None:
                return build_action

        if defense_mode and "turret" not in buildings and "masonry" in state.techs_unlocked:
            build_action = _choose_closest_build_action(env, workers, _build_actions_for(legal, "turret"))
            if build_action is not None:
                return build_action

        if _unit_count(env, faction, "soldier") < 2:
            desired = ("train", "soldier")
            if desired in legal:
                return desired

        desired_archers = 1 if "fletching" in state.techs_unlocked else 0
        if _unit_count(env, faction, "archer") < desired_archers:
            desired = ("train", "archer")
            if desired in legal:
                return desired

        desired_horsemen = 2 if "horsemanship" in state.techs_unlocked and "stable" in buildings else 0
        if _unit_count(env, faction, "horseman") < desired_horsemen:
            desired = ("train", "horseman")
            if desired in legal:
                return desired

        if "masonry" in state.techs_unlocked and "quarry" in buildings and "turret" not in buildings:
            build_action = _choose_closest_build_action(env, workers, _build_actions_for(legal, "turret"))
            if build_action is not None:
                return build_action

        if not defense_mode:
            gathers = _actions_of_kind(legal, "gather")
            if gathers:
                return gathers[0]

        if military:
            enemy_units = [u for u in env.units if u.faction != faction]
            defensive_targets = _defensive_targets(env, faction)
            role_order = {"defender": 0, "raider": 1, "line": 2}
            ordered_military = sorted(
                military,
                key=lambda unit: (role_order[_military_role(env, faction, unit, military)], unit.id),
            )
            for unit in ordered_military:
                role = _military_role(env, faction, unit, military)
                if push_mode and home_enemy_force == 0 and len(military) >= 4 and role == "defender":
                    role = "line"
                target = self._military_targets.get(unit.id)
                if (
                    not defense_mode
                    and not push_mode
                    and role != "raider"
                    and rally_anchor is not None
                    and unit.position != rally_anchor
                    and not _army_ready_to_push(env, faction, _nearest_enemy_base(env, faction))
                ):
                    rally_action = ("move_towards", unit.id, rally_anchor)
                    if rally_action in legal:
                        return rally_action

                if role == "defender" and defensive_targets:
                    target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                    self._military_targets[unit.id] = target
                elif role == "defender":
                    target = _home_guard_target(env, faction, unit)
                    if target is not None:
                        self._military_targets[unit.id] = target
                elif push_mode and role != "raider":
                    target = _nearest_enemy_base(env, faction)
                    self._military_targets[unit.id] = target
                elif target is None or all(enemy.position != target for enemy in enemy_units):
                    if role == "raider":
                        enemy_workers = [enemy for enemy in enemy_units if enemy.unit_type == "worker"]
                        if enemy_workers:
                            target = min(enemy_workers, key=lambda enemy: _distance(unit.position, enemy.position)).position
                        else:
                            target = _nearest_enemy_base(env, faction)
                    elif defensive_targets:
                        target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                    elif enemy_units:
                        target = min(enemy_units, key=lambda enemy: _distance(unit.position, enemy.position)).position
                    else:
                        target = _nearest_enemy_base(env, faction)
                    self._military_targets[unit.id] = target

                if role != "raider" and defensive_targets and target not in {enemy.position for enemy in defensive_targets}:
                    target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                    self._military_targets[unit.id] = target
                elif (
                    not defense_mode
                    and not push_mode
                    and role == "line"
                    and target == _nearest_enemy_base(env, faction)
                    and not _army_ready_to_push(env, faction, target)
                    and rally_anchor is not None
                ):
                    target = rally_anchor
                    self._military_targets[unit.id] = target

                if target is not None:
                    action = ("move_towards", unit.id, target)
                    if action in legal:
                        return action

        if not defense_mode:
            for worker in workers:
                target = _nearest_resource(env, worker.position)
                if target is not None:
                    action = ("move_towards", worker.id, target)
                    if action in legal:
                        return action

        if not defense_mode:
            build_site_needed = "storehouse" not in buildings or "barracks" not in buildings
            if build_site_needed:
                for worker in workers:
                    target = _nearest_build_site(env, faction, worker.position)
                    if target is None:
                        continue
                    action = ("move_towards", worker.id, target)
                    if action in legal:
                        return action

        if defense_mode:
            for worker in workers:
                retreat = _worker_retreat_target(env, faction, worker)
                if retreat is None:
                    continue
                action = ("move_towards", worker.id, retreat)
                if action in legal:
                    return action

        return None
