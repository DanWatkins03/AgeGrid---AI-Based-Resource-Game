from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env import hexgrid
from src.agegrid.env.systems import production, tech


@dataclass(frozen=True)
class HeuristicProfile:
    name: str
    desired_workers: int = 3
    research_order: tuple[str, ...] = (
        "mining",
        "bronze",
        "masonry",
        "animal_husbandry",
        "trade",
        "horseback_riding",
        "fletching",
        "construction",
        "iron",
        "stirrups",
        "engineering",
        "currency",
        "fortify",
        "steel",
        "markets",
        "walls",
        "infrastructure",
        "precision",
        "logistics",
        "stronghold",
        "heavy_cavalry",
        "advanced_siege",
        "war_economy",
        "agriculture",
    )
    desired_archers: int = 1
    desired_horsemen: int = 2
    defense_home_force: int = 2
    contested_home_force: int = 3
    cavalry_home_force: int = 3
    emergency_cavalry_prefers_archer: bool = True


HEURISTIC_PROFILES: dict[str, HeuristicProfile] = {
    "balanced": HeuristicProfile(name="balanced"),
    "greedy": HeuristicProfile(
        name="greedy",
        desired_workers=2,
        desired_archers=0,
        desired_horsemen=1,
        defense_home_force=1,
        contested_home_force=2,
        cavalry_home_force=2,
    ),
    "aggressive": HeuristicProfile(
        name="aggressive",
        desired_workers=2,
        desired_archers=0,
        desired_horsemen=3,
        defense_home_force=1,
        contested_home_force=2,
        cavalry_home_force=2,
    ),
    "defensive": HeuristicProfile(
        name="defensive",
        desired_workers=3,
        desired_archers=2,
        desired_horsemen=1,
        defense_home_force=3,
        contested_home_force=4,
        cavalry_home_force=4,
    ),
}


@dataclass(frozen=True)
class HeuristicDiagnostics:
    tech_count: int
    tech_deficit: int
    economy_score: int
    economy_gap: int
    military_score: int
    military_gap: int
    behind: bool
    recovery: bool


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return hexgrid.distance(a, b)


def _nearest_enemy_base(env: AgeGridEnv, faction: str) -> tuple[int, int]:
    enemy = next(name for name in env.factions if name != faction)
    return env.bases[enemy].position


def _enemy_broken(env: AgeGridEnv, faction: str) -> bool:
    enemy = next(name for name in env.factions if name != faction)
    enemy_workers = any(unit.faction == enemy and unit.unit_type == "worker" for unit in env.units)
    enemy_military = any(unit.faction == enemy and unit.attack_damage > 0 for unit in env.units)
    return not enemy_workers and not enemy_military


def _enemy_base_siege_target(
    env: AgeGridEnv,
    faction: str,
    unit,
    reserved: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    base = _nearest_enemy_base(env, faction)
    occupied = env._occupied_positions()
    occupied.discard(unit.position)
    candidates = hexgrid.neighbors(base)
    reserved = reserved or set()
    valid = [
        pos
        for pos in candidates
        if env._in_bounds(pos)
        and pos not in occupied
        and (pos == unit.position or pos not in reserved)
    ]
    if not valid:
        valid = [pos for pos in candidates if env._in_bounds(pos) and pos not in occupied]
    if not valid:
        return None
    return min(
        valid,
        key=lambda pos: (
            0 if pos == unit.position else 1,
            _distance(unit.position, pos),
            _distance(pos, base),
            pos[1],
            pos[0],
        ),
    )


def _nearest_resource(env: AgeGridEnv, pos: tuple[int, int]) -> tuple[int, int] | None:
    faction = env.factions[env.current_player]
    nodes = env.visible_resources(faction)
    if not nodes:
        return None
    return min(nodes, key=lambda r: _distance(r.position, pos)).position


def _economic_resource_positions(env: AgeGridEnv, faction: str, radius: int = 6) -> list[tuple[int, int]]:
    base = env.bases[faction].position
    nearby = [
        node.position
        for node in env.visible_resources(faction)
        if _distance(node.position, base) <= radius
    ]
    if nearby:
        return nearby
    return [node.position for node in env.visible_resources(faction)]


def _buildings(env: AgeGridEnv, faction: str) -> set[str]:
    return {b.building_type for b in env.buildings if b.faction == faction and b.hp > 0}


def _unit_count(env: AgeGridEnv, faction: str, unit_type: str) -> int:
    return sum(1 for u in env.units if u.faction == faction and u.unit_type == unit_type)


def _actions_of_kind(actions: Iterable[Action], kind: str) -> list[Action]:
    return [action for action in actions if action[0] == kind]


def _tech_count(env: AgeGridEnv, faction: str) -> int:
    return len(env.faction_state(faction).techs_unlocked)


def _economy_income(env: AgeGridEnv, faction: str) -> int:
    return sum(
        building.resource_income
        for structure in env.buildings
        if structure.faction == faction and structure.hp > 0
        if (building := production.building_stats(env, faction, structure.building_type)) is not None
    )


def _economy_score(env: AgeGridEnv, faction: str) -> int:
    workers = _unit_count(env, faction, "worker")
    buildings = _buildings(env, faction)
    return (
        env.bank[faction] // 20
        + workers * 3
        + _economy_income(env, faction) * 2
        + int("market" in buildings) * 2
        + int("storehouse" in buildings)
        + int("quarry" in buildings)
    )


def heuristic_diagnostics(env: AgeGridEnv, faction: str) -> HeuristicDiagnostics:
    enemy = next(name for name in env.factions if name != faction)
    tech_count = _tech_count(env, faction)
    enemy_tech_count = _tech_count(env, enemy)
    economy_score = _economy_score(env, faction)
    enemy_economy_score = _economy_score(env, enemy)
    military_score = _total_force(env, faction)
    enemy_military_score = _total_force(env, enemy)
    tech_deficit = max(0, enemy_tech_count - tech_count)
    economy_gap = max(0, enemy_economy_score - economy_score)
    military_gap = max(0, enemy_military_score - military_score)
    behind = (
        tech_deficit >= 3
        or economy_gap >= 5
        or military_gap >= 5
        or (
            tech_deficit >= 2
            and (economy_gap >= 3 or military_gap >= 3)
        )
    )
    recovery = behind and (economy_gap >= 4 or military_gap >= 4)
    return HeuristicDiagnostics(
        tech_count=tech_count,
        tech_deficit=tech_deficit,
        economy_score=economy_score,
        economy_gap=economy_gap,
        military_score=military_score,
        military_gap=military_gap,
        behind=behind,
        recovery=recovery,
    )


def heuristic_diagnostics_label(env: AgeGridEnv, faction: str) -> str:
    diagnostics = heuristic_diagnostics(env, faction)
    state = "Recovery" if diagnostics.recovery else "Behind" if diagnostics.behind else "Stable"
    return (
        f"{state} | tech gap {diagnostics.tech_deficit} "
        f"econ gap {diagnostics.economy_gap} mil gap {diagnostics.military_gap} "
        f"support {env.faction_state(faction).war_support}"
    )


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


def _base_camp_targets(env: AgeGridEnv, faction: str) -> list:
    base = env.bases[faction].position
    spawn_ring = set(_spawn_ring_positions(env, faction))
    return [
        enemy
        for enemy in env.units
        if enemy.faction != faction
        and enemy.attack_damage > 0
        and (enemy.position in spawn_ring or _distance(enemy.position, base) <= 1)
    ]


def _base_siege_targets(env: AgeGridEnv, faction: str) -> list:
    base = env.bases[faction].position
    return sorted(
        [
            enemy
            for enemy in env.units
            if enemy.faction != faction
            and enemy.attack_damage > 0
            and enemy.attack_range > 0
            and _distance(enemy.position, base) <= enemy.attack_range
        ],
        key=lambda enemy: (enemy.hp, _distance(enemy.position, base), enemy.id),
    )


def _defensive_targets(env: AgeGridEnv, faction: str) -> list:
    return [
        enemy
        for enemy in env.units
        if enemy.faction != faction and enemy.attack_damage > 0 and _threat_score(env, faction, enemy.position) <= 5
    ]


def _emergency_targets(env: AgeGridEnv, faction: str) -> list:
    return [
        enemy
        for enemy in env.units
        if enemy.faction != faction and enemy.attack_damage > 0 and _threat_score(env, faction, enemy.position) <= 3
    ]


def _in_defense_mode(env: AgeGridEnv, faction: str) -> bool:
    return bool(_emergency_targets(env, faction))


def defense_mode_active(env: AgeGridEnv, faction: str) -> bool:
    return _in_defense_mode(env, faction)


def _base_under_siege(env: AgeGridEnv, faction: str) -> bool:
    return bool(_base_siege_targets(env, faction))


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
    enemy = next(name for name in env.factions if name != faction)
    enemy_military = [unit for unit in env.units if unit.faction == enemy and unit.attack_damage > 0]
    if _in_defense_mode(env, faction) or _defensive_targets(env, faction):
        return "Hold"
    if len(military) == 1 and not enemy_military:
        return "Advance"
    if _enemy_broken(env, faction) and military:
        return "Siege"
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
    candidates = [anchor, *hexgrid.neighbors(anchor)]
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
        enemy = next(name for name in env.factions if name != faction)
        _, home_enemy_force = _home_force_balance(env, faction)
        safe_to_raid = (
            not _defensive_targets(env, faction)
            and home_enemy_force == 0
            and (
                _push_mode_active(env, faction)
                or (len(military) >= 3 and _total_force(env, faction) >= _total_force(env, enemy) + 2)
            )
        )
        if not safe_to_raid:
            return "line"
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


def _archer_disengage_target(env: AgeGridEnv, faction: str, unit) -> tuple[int, int] | None:
    if unit.unit_type != "archer" or unit.attack_range <= 1:
        return None
    nearby_cavalry = [
        enemy
        for enemy in env.units
        if enemy.faction != faction and enemy.unit_type == "horseman" and _distance(unit.position, enemy.position) <= 2
    ]
    if not nearby_cavalry:
        return None
    friendly_support = _local_force(env, unit.position, faction, radius=2)
    enemy_support = _local_force(env, unit.position, next(name for name in env.factions if name != faction), radius=2)
    if friendly_support >= enemy_support + 1:
        return None
    base = env.bases[faction].position
    if unit.position == base:
        return None
    return base


def _recover_target(env: AgeGridEnv, faction: str, unit) -> tuple[int, int] | None:
    if unit.attack_damage <= 0:
        return None
    max_hp = env.unit_max_hp(unit)
    if unit.hp > max_hp // 2:
        return None
    enemy = next(name for name in env.factions if name != faction)
    local_enemy_force = _local_force(env, unit.position, enemy, radius=2)
    local_friendly_force = _local_force(env, unit.position, faction, radius=2)
    if unit.position == env.bases[faction].position:
        return None
    if local_enemy_force > local_friendly_force or unit.hp <= max(2, max_hp // 3):
        return env.bases[faction].position
    return None


def _attack_priority(env: AgeGridEnv, faction: str, action: Action) -> tuple[float, float, float, float, float]:
    target_id = action[2]
    target = next((u for u in env.units if u.id == target_id), None)
    if target is None:
        return (math.inf, math.inf, math.inf, math.inf, math.inf)
    camp_rank = 0 if any(enemy.id == target.id for enemy in _base_camp_targets(env, faction)) else 1
    emergency_rank = 0 if target.attack_damage > 0 and _threat_score(env, faction, target.position) <= 2 else 1
    threat_rank = 0 if target.attack_damage > 0 and _threat_score(env, faction, target.position) <= 4 else 1
    type_rank = {"horseman": 0, "archer": 1, "soldier": 2, "worker": 3}.get(target.unit_type, 4)
    return (camp_rank, emergency_rank, threat_rank, type_rank, target.hp)


def _nearest_build_site(env: AgeGridEnv, faction: str, worker_pos: tuple[int, int]) -> tuple[int, int] | None:
    base = env.bases[faction].position
    occupied = env._occupied_positions()
    candidates = [pos for pos in hexgrid.positions_at_distance(base, 2, env.config.width, env.config.height) if pos != base]
    valid = [pos for pos in candidates if env._in_bounds(pos) and pos not in occupied]
    if not valid:
        return None
    return min(valid, key=lambda pos: _distance(worker_pos, pos))


def _spawn_ring_positions(env: AgeGridEnv, faction: str) -> list[tuple[int, int]]:
    return [pos for pos in hexgrid.neighbors(env.bases[faction].position) if env._in_bounds(pos)]


def _is_on_spawn_ring(env: AgeGridEnv, faction: str, pos: tuple[int, int]) -> bool:
    return pos in _spawn_ring_positions(env, faction)


def _spawn_ring_blocked(env: AgeGridEnv, faction: str) -> bool:
    occupied = env._occupied_positions()
    return all(env._in_bounds(pos) and pos in occupied for pos in _spawn_ring_positions(env, faction))


def _visible_resource_positions(env: AgeGridEnv, faction: str, resource_type: str) -> set[tuple[int, int]]:
    return {
        resource.position
        for resource in env.visible_resources(faction)
        if resource.resource_type == resource_type
    }


def _resource_positions(env: AgeGridEnv, resource_type: str) -> set[tuple[int, int]]:
    return {
        resource.position
        for resource in env.resources
        if resource.abundance > 0 and resource.resource_type == resource_type
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


def _worker_useful_slots(env: AgeGridEnv, faction: str, buildings: set[str], desired_workers: int) -> int:
    resource_slots = len({pos for pos in _economic_resource_positions(env, faction)})
    build_slots = int("storehouse" not in buildings) + int("barracks" not in buildings)
    return max(1, min(desired_workers, resource_slots + build_slots))


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


def _frontline_is_losing_locally(env: AgeGridEnv, faction: str, target: tuple[int, int]) -> bool:
    enemy = next(name for name in env.factions if name != faction)
    friendly_force = _local_force(env, target, faction)
    enemy_force = _local_force(env, target, enemy)
    friendly_support = _support_count(env, target, faction)
    enemy_support = _support_count(env, target, enemy)
    friendly_ranged = sum(
        1
        for unit in env.units
        if unit.faction == faction and unit.attack_damage > 0 and unit.attack_range > 1 and _distance(unit.position, target) <= 3
    )
    enemy_ranged = sum(
        1
        for unit in env.units
        if unit.faction == enemy and unit.attack_damage > 0 and unit.attack_range > 1 and _distance(unit.position, target) <= 3
    )
    return (
        enemy_force > friendly_force
        or enemy_support > friendly_support
        or enemy_ranged > friendly_ranged
    )


def _frontline_target(env: AgeGridEnv, faction: str, unit) -> tuple[int, int] | None:
    enemy_combat = [enemy for enemy in env.units if enemy.faction != faction and enemy.attack_damage > 0]
    if enemy_combat:
        return min(enemy_combat, key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp)).position
    enemy_workers = [enemy for enemy in env.units if enemy.faction != faction and enemy.unit_type == "worker"]
    if enemy_workers:
        return min(enemy_workers, key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp)).position
    return None


def _raid_target(env: AgeGridEnv, faction: str, unit, enemy_units: list) -> tuple[int, int] | None:
    enemy = next(name for name in env.factions if name != faction)
    enemy_workers = [candidate for candidate in enemy_units if candidate.unit_type == "worker"]
    if not enemy_workers:
        return None

    def raid_score(target) -> tuple[int, int, int, int, int]:
        local_enemy_force = _local_force(env, target.position, enemy, radius=2)
        local_friendly_force = _local_force(env, target.position, faction, radius=2)
        can_finish = 0 if target.hp <= unit.attack_damage else 1
        danger = max(0, local_enemy_force - local_friendly_force)
        ranged_pressure = sum(
            max(1, hostile.attack_damage)
            for hostile in enemy_units
            if hostile.attack_damage > 0 and _distance(hostile.position, target.position) <= max(2, hostile.attack_range)
        )
        return (can_finish, danger, ranged_pressure, _distance(unit.position, target.position), target.hp)

    viable = [
        target
        for target in enemy_workers
        if target.hp <= unit.attack_damage
        or _local_force(env, target.position, faction, radius=2) >= _local_force(env, target.position, enemy, radius=2)
    ]
    pool = viable or enemy_workers
    best = min(pool, key=raid_score)
    best_score = raid_score(best)
    if best_score[1] >= 3 and best_score[0] > 0:
        return None
    return best.position


def _staging_target(env: AgeGridEnv, faction: str, unit, rally_anchor: tuple[int, int] | None) -> tuple[int, int] | None:
    if rally_anchor is not None:
        if unit.position == rally_anchor or _distance(unit.position, rally_anchor) <= 1:
            return None
        return rally_anchor
    guard_target = _home_guard_target(env, faction, unit)
    if guard_target is not None and unit.position != guard_target:
        return guard_target
    home = _home_anchor(env, faction)
    if unit.position != home:
        return home
    return None


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
    if attacker.unit_type == "archer" and _archer_disengage_target(env, faction, attacker) is not None:
        return False
    if _recover_target(env, faction, attacker) is not None and target.hp > attacker.attack_damage:
        return False
    if target.attack_damage == 0:
        friendly_force = _local_force(env, target.position, faction)
        enemy_force = _local_force(env, target.position, target.faction)
        can_finish = target.hp <= attacker.attack_damage
        if enemy_force > friendly_force + max(1, attacker.hp // 4):
            return False
        return can_finish or friendly_force >= enemy_force
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
    base_threat += tech.passive_modifier_total(env, enemy, "base_attack_bonus")
    if env.bases[enemy].hp <= attacker.attack_damage:
        return True
    if len(nearby_friendlies) < 2:
        return False
    return friendly_force >= enemy_force + base_threat


def _enemy_attackers_in_base_range(env: AgeGridEnv, faction: str) -> list:
    base_pos = env.bases[faction].position
    return [
        unit
        for unit in env.units
        if unit.faction != faction
        and unit.attack_damage > 0
        and unit.attack_range > 0
        and _distance(unit.position, base_pos) <= unit.attack_range
    ]


def _enemy_can_finish_base_next_turn(env: AgeGridEnv, faction: str) -> bool:
    if env.bases[faction].hp <= 0:
        return True
    attackers = _enemy_attackers_in_base_range(env, faction)
    if not attackers:
        return False
    strongest_attack = max(unit.attack_damage for unit in attackers)
    return strongest_attack * env.config.actions_per_turn >= env.bases[faction].hp


def _pressure_fallback_action(ctx) -> Action | None:
    priorities = (
        "attack",
        "attack_base",
        "train",
        "build",
        "move_towards",
        "gather",
        "accept_peace",
    )
    for kind in priorities:
        for action in ctx.legal:
            if action[0] == kind:
                return action
    return None


class HeuristicAgent:
    """
    Progression-focused baseline:
    - gather and expand workers early
    - research along a fixed economic -> military ladder
    - build a barracks and train soldiers / archers
    - attack nearby targets, otherwise move military toward the enemy base
    """

    def __init__(self, desired_workers: int | None = None, profile: HeuristicProfile | str | None = None):
        if profile is None:
            resolved_profile = HEURISTIC_PROFILES["balanced"]
        elif isinstance(profile, str):
            resolved_profile = HEURISTIC_PROFILES[profile]
        else:
            resolved_profile = profile
        if desired_workers is not None:
            resolved_profile = replace(resolved_profile, desired_workers=desired_workers)
        self.profile = resolved_profile
        self.desired_workers = resolved_profile.desired_workers
        self._military_targets: dict[int, tuple[int, int]] = {}
        self._military_stall: dict[int, int] = {}
        self._military_last_positions: dict[int, tuple[int, int]] = {}
        self._military_last_distance: dict[int, int] = {}
        self._military_target_lock: dict[int, int] = {}
        self._last_worker_id: dict[str, int] = {}

    @dataclass
    class _Context:
        faction: str
        enemy_faction: str
        state: object
        legal: list[Action]
        diagnostics: HeuristicDiagnostics
        at_war: bool
        in_truce: bool
        defense_mode: bool
        base_under_siege: bool
        workers: list
        buildings: set[str]
        military: list
        emergency_targets: list
        nearby_cavalry: list
        rally_anchor: tuple[int, int] | None
        spawn_actions: list[Action]
        emergency_train: Action | None
        home_friendly_force: int
        home_enemy_force: int
        tech_deficit: int
        economy_gap: int
        military_gap: int
        behind_mode: bool
        recovery_mode: bool
        push_mode: bool
        siege_finish: bool
        collapse_mode: bool
        last_stand: bool
        desired_home_force: int
        useful_worker_slots: int
        rebuild_mode: bool
        declare_war_actions: list[Action]
        offer_peace_actions: list[Action]
        accept_peace_actions: list[Action]

    def _ordered_workers(self, faction: str, workers: list) -> list:
        if not workers:
            return []
        ordered = sorted(workers, key=lambda worker: worker.id)
        last_worker_id = self._last_worker_id.get(faction)
        if last_worker_id is None:
            return ordered
        for index, worker in enumerate(ordered):
            if worker.id == last_worker_id:
                return ordered[index + 1 :] + ordered[: index + 1]
        return ordered

    def _record_worker_assignment(self, action: Action | None, env: AgeGridEnv, faction: str) -> None:
        if action is None or len(action) < 2 or not isinstance(action[1], int):
            return
        unit = next((candidate for candidate in env.units if candidate.id == action[1] and candidate.faction == faction), None)
        if unit is None or unit.unit_type != "worker":
            return
        self._last_worker_id[faction] = unit.id

    def _clear_military_target(self, unit_id: int) -> None:
        self._military_targets.pop(unit_id, None)
        self._military_target_lock.pop(unit_id, None)

    def _set_military_target(self, unit_id: int, target: tuple[int, int], lock: int = 0) -> None:
        self._military_targets[unit_id] = target
        self._military_target_lock[unit_id] = max(0, lock)

    def _refresh_military_stall(self, env: AgeGridEnv, faction: str) -> None:
        active_ids = {unit.id for unit in env.units if unit.faction == faction and unit.attack_damage > 0}
        self._military_targets = {unit_id: target for unit_id, target in self._military_targets.items() if unit_id in active_ids}
        self._military_stall = {unit_id: turns for unit_id, turns in self._military_stall.items() if unit_id in active_ids}
        self._military_last_positions = {
            unit_id: pos for unit_id, pos in self._military_last_positions.items() if unit_id in active_ids
        }
        self._military_last_distance = {
            unit_id: distance for unit_id, distance in self._military_last_distance.items() if unit_id in active_ids
        }
        self._military_target_lock = {
            unit_id: max(0, turns - 1) for unit_id, turns in self._military_target_lock.items() if unit_id in active_ids
        }
        for unit in env.units:
            if unit.id not in active_ids:
                continue
            target = self._military_targets.get(unit.id)
            last_pos = self._military_last_positions.get(unit.id)
            previous_distance = self._military_last_distance.get(unit.id)
            if target is None:
                self._military_stall[unit.id] = 0
                self._military_last_distance[unit.id] = 0
            else:
                current_distance = _distance(unit.position, target)
                if last_pos is None or previous_distance is None:
                    self._military_stall[unit.id] = 0
                elif last_pos == unit.position or current_distance >= previous_distance:
                    self._military_stall[unit.id] = self._military_stall.get(unit.id, 0) + 1
                else:
                    self._military_stall[unit.id] = 0
                self._military_last_distance[unit.id] = current_distance
            self._military_last_positions[unit.id] = unit.position

    def _research_priority(self, env: AgeGridEnv, ctx: _Context) -> list[str]:
        enemy_state = env.faction_state(ctx.enemy_faction)
        friendly_archers = _unit_count(env, ctx.faction, "archer")
        enemy_archers = _unit_count(env, ctx.enemy_faction, "archer")
        priorities: list[str] = []

        if ctx.behind_mode:
            priorities.extend(("construction", "fortify", "walls"))
            if ctx.economy_gap >= 4:
                priorities.extend(("trade", "markets", "currency", "infrastructure"))
            if ctx.military_gap >= 4:
                priorities.extend(("iron", "steel", "precision"))
            if enemy_archers > friendly_archers:
                priorities.extend(("fletching", "precision", "engineering"))

        if "construction" in enemy_state.techs_unlocked or "engineering" in enemy_state.techs_unlocked:
            priorities.extend(("construction", "fortify", "walls"))
        if "horseback_riding" in enemy_state.techs_unlocked and "fletching" not in ctx.state.techs_unlocked:
            priorities.extend(("fletching", "construction"))
        if env.bases[ctx.faction].hp <= max(10, env.config.base_hp // 2):
            priorities.extend(("construction", "fortify", "walls", "stronghold"))

        ordered: list[str] = []
        seen: set[str] = set()
        for tech_id in priorities:
            if tech_id not in seen:
                ordered.append(tech_id)
                seen.add(tech_id)
        return ordered

    def _choose_research(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        faction = ctx.faction
        legal = ctx.legal
        buildings = ctx.buildings
        state = ctx.state
        if state.tech_in_progress is not None:
            return None
        if "mining" in state.techs_unlocked and "storehouse" not in buildings:
            return None
        if "bronze" in state.techs_unlocked and "barracks" not in buildings:
            return None

        visible_stone = bool(_resource_positions(env, "stone"))
        visible_horses = bool(_resource_positions(env, "horses"))
        research_order = list(self.profile.research_order)
        if "masonry" in research_order and "animal_husbandry" in research_order:
            masonry_index = research_order.index("masonry")
            horse_index = research_order.index("animal_husbandry")
            if visible_horses and not visible_stone and horse_index > masonry_index:
                research_order[masonry_index], research_order[horse_index] = research_order[horse_index], research_order[masonry_index]
            elif visible_stone and not visible_horses and masonry_index > horse_index:
                research_order[masonry_index], research_order[horse_index] = research_order[horse_index], research_order[masonry_index]
        if "fortify" in research_order and "stirrups" in research_order:
            fort_index = research_order.index("fortify")
            stirrup_index = research_order.index("stirrups")
            if visible_horses and not visible_stone and stirrup_index > fort_index:
                research_order[fort_index], research_order[stirrup_index] = research_order[stirrup_index], research_order[fort_index]
            elif visible_stone and not visible_horses and fort_index > stirrup_index:
                research_order[fort_index], research_order[stirrup_index] = research_order[stirrup_index], research_order[fort_index]
        research_order = self._research_priority(env, ctx) + research_order
        seen: set[str] = set()
        for tech_id in research_order:
            if tech_id in seen:
                continue
            seen.add(tech_id)
            action = ("research", tech_id)
            if action in legal and tech_id not in state.techs_unlocked:
                return action
        return None

    def _build_context(self, env: AgeGridEnv, faction: str, legal: list[Action]) -> _Context:
        enemy_faction = next(name for name in env.factions if name != faction)
        relation = env.relation_state(faction, enemy_faction)
        defense_mode = _in_defense_mode(env, faction)
        base_under_siege = _base_under_siege(env, faction)
        diagnostics = heuristic_diagnostics(env, faction)
        workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
        buildings = _buildings(env, faction)
        military = [u for u in env.units if u.faction == faction and u.attack_damage > 0]
        emergency_targets = _emergency_targets(env, faction)
        nearby_cavalry = _enemy_cavalry_nearby(env, faction)
        home_friendly_force, home_enemy_force = _home_force_balance(env, faction)
        return self._Context(
            faction=faction,
            enemy_faction=enemy_faction,
            state=env.faction_state(faction),
            legal=legal,
            diagnostics=diagnostics,
            at_war=relation.state == "war",
            in_truce=relation.state == "truce",
            defense_mode=defense_mode,
            base_under_siege=base_under_siege,
            workers=workers,
            buildings=buildings,
            military=military,
            emergency_targets=emergency_targets,
            nearby_cavalry=nearby_cavalry,
            rally_anchor=_rally_anchor(env, faction, military),
            spawn_actions=_actions_of_kind(legal, "spawn_worker"),
            emergency_train=_preferred_emergency_train(legal, nearby_cavalry),
            home_friendly_force=home_friendly_force,
            home_enemy_force=home_enemy_force,
            tech_deficit=diagnostics.tech_deficit,
            economy_gap=diagnostics.economy_gap,
            military_gap=diagnostics.military_gap,
            behind_mode=diagnostics.behind,
            recovery_mode=diagnostics.recovery,
            push_mode=_push_mode_active(env, faction),
            siege_finish=_enemy_broken(env, faction) and bool(military),
            collapse_mode=not workers,
            last_stand=defense_mode and home_enemy_force > home_friendly_force,
            desired_home_force=(
                self.profile.cavalry_home_force
                if nearby_cavalry
                else self.profile.contested_home_force
                if defense_mode or home_enemy_force > 0
                else self.profile.defense_home_force
            ),
            useful_worker_slots=_worker_useful_slots(env, faction, buildings, self.profile.desired_workers),
            rebuild_mode=(
                home_enemy_force > 0
                and home_friendly_force < max(
                    self.profile.cavalry_home_force if nearby_cavalry else self.profile.contested_home_force,
                    home_enemy_force,
                )
            ),
            declare_war_actions=_actions_of_kind(legal, "declare_war"),
            offer_peace_actions=_actions_of_kind(legal, "offer_peace"),
            accept_peace_actions=_actions_of_kind(legal, "accept_peace"),
        )

    def _choose_research_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if ctx.defense_mode or ctx.base_under_siege or ctx.siege_finish:
            return None
        if ctx.at_war and (ctx.home_enemy_force > 0 or ctx.military_gap > 0):
            return None
        if ctx.at_war and _total_force(env, ctx.enemy_faction) > _total_force(env, ctx.faction) and not ctx.behind_mode:
            return None
        return self._choose_research(env, ctx)

    def _choose_diplomacy_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        relation = env.relation_state(ctx.faction, ctx.enemy_faction)
        war_support = env.faction_state(ctx.faction).war_support
        enemy_support = env.faction_state(ctx.enemy_faction).war_support
        direct_siege = bool(_base_camp_targets(env, ctx.faction)) or ctx.base_under_siege
        imminent_base_loss = _enemy_can_finish_base_next_turn(env, ctx.faction)
        if ctx.accept_peace_actions:
            losing_war = (
                ctx.home_enemy_force > ctx.home_friendly_force
                or len(ctx.military) < max(1, len([u for u in env.units if u.faction == ctx.enemy_faction and u.attack_damage > 0]) // 2)
                or env.bank[ctx.faction] < env.bank[ctx.enemy_faction] // 2
                or war_support <= 35
            )
            if losing_war and not ctx.siege_finish and not direct_siege and not imminent_base_loss:
                return ctx.accept_peace_actions[0]
        if ctx.offer_peace_actions:
            war_turns = env.turn - relation.since_turn
            losing_score = env.war_score(ctx.faction, ctx.enemy_faction) <= env.war_score(ctx.enemy_faction, ctx.faction)
            if (
                war_turns >= env.config.min_war_duration
                and (ctx.last_stand or ctx.rebuild_mode or war_support <= 45 or losing_score)
                and not direct_siege
                and not imminent_base_loss
            ):
                return ctx.offer_peace_actions[0]
        if ctx.declare_war_actions:
            if ctx.in_truce:
                return None
            frontier_pressure = any(
                _distance(unit.position, env.bases[ctx.enemy_faction].position) <= 6
                for unit in ctx.military
            )
            enemy_incursion = any(
                _distance(unit.position, env.bases[ctx.faction].position) <= 6
                for unit in env.units
                if unit.faction == ctx.enemy_faction and unit.attack_damage > 0
            )
            support_floor = env.config.war_support_to_declare_min
            if not (frontier_pressure or enemy_incursion or ctx.push_mode or ctx.siege_finish):
                support_floor += 10
            if war_support < support_floor:
                return None
            if env.bank[ctx.faction] < env.config.war_declaration_cost + env.config.war_upkeep_per_turn * 3:
                return None
            ready_force = len(ctx.military) >= 2 and (
                _total_force(env, ctx.faction) >= _total_force(env, ctx.enemy_faction)
                or len(ctx.military) >= 4
            )
            defensive_declare = enemy_incursion and len(ctx.military) >= 2 and war_support >= env.config.war_support_to_declare_min
            if (ready_force and (ctx.push_mode or ctx.siege_finish or frontier_pressure)) or defensive_declare:
                return ctx.declare_war_actions[0]
        return None

    def _choose_collapse_recovery(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if not ctx.collapse_mode:
            return None
        if (ctx.last_stand or ctx.rebuild_mode) and ctx.emergency_train is not None:
            return ctx.emergency_train
        if ctx.rebuild_mode:
            return None
        if ctx.spawn_actions:
            return ctx.spawn_actions[0]
        return ctx.emergency_train

    def _choose_attack_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        attacks = _actions_of_kind(ctx.legal, "attack")
        if not attacks:
            return None
        filtered_attacks = []
        recovering_attacks = []
        for action in attacks:
            attacker = next((u for u in env.units if u.id == action[1]), None)
            target = next((u for u in env.units if u.id == action[2]), None)
            if attacker is None or target is None:
                continue
            if _recover_target(env, ctx.faction, attacker) is not None and target.hp > attacker.attack_damage:
                recovering_attacks.append(action)
                continue
            filtered_attacks.append(action)
        if filtered_attacks:
            attacks = filtered_attacks
        elif recovering_attacks:
            return None
        viable_attacks = [action for action in attacks if _viable_unit_attack(env, ctx.faction, action)]
        if not viable_attacks:
            pressured_archer_attacks = []
            for action in attacks:
                attacker = next((u for u in env.units if u.id == action[1]), None)
                if attacker is not None and attacker.unit_type == "archer" and _archer_disengage_target(env, ctx.faction, attacker) is not None:
                    pressured_archer_attacks.append(action)
            if len(pressured_archer_attacks) == len(attacks):
                return None
            return min(attacks, key=lambda action: _attack_priority(env, ctx.faction, action))
        return min(viable_attacks, key=lambda action: _attack_priority(env, ctx.faction, action))

    def _choose_defense_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if not ctx.defense_mode or not ctx.military:
            return None
        if ctx.rebuild_mode and ctx.emergency_train is not None:
            return ctx.emergency_train
        role_order = {"defender": 0, "line": 1, "raider": 2}
        ordered_military = sorted(
            ctx.military,
            key=lambda unit: (role_order.get(_military_role(env, ctx.faction, unit, ctx.military), 3), unit.id),
        )
        siege_targets = _base_siege_targets(env, ctx.faction)
        if siege_targets:
            for unit in ordered_military:
                target = min(
                    siege_targets,
                    key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp, enemy.id),
                ).position
                action = ("move_towards", unit.id, target)
                if action in ctx.legal:
                    return action
        camp_targets = _base_camp_targets(env, ctx.faction)
        for unit in ordered_military:
            if camp_targets:
                target = min(
                    camp_targets,
                    key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp),
                ).position
                action = ("move_towards", unit.id, target)
                if action in ctx.legal:
                    return action
            if ctx.emergency_targets:
                target = min(
                    ctx.emergency_targets,
                    key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp),
                ).position
                action = ("move_towards", unit.id, target)
                if action in ctx.legal:
                    return action
        if (ctx.home_friendly_force < ctx.desired_home_force or ctx.last_stand) and ctx.emergency_train is not None:
            return ctx.emergency_train
        fallback_moves = _actions_of_kind(ctx.legal, "move_towards")
        defensive_positions = {
            env.bases[ctx.faction].position,
            *(u.position for u in ctx.workers),
            *(b.position for b in env.buildings if b.faction == ctx.faction and b.hp > 0),
        }
        guarded_moves = [action for action in fallback_moves if action[2] in defensive_positions]
        if guarded_moves:
            return guarded_moves[0]
        return self._choose_worker_retreat(env, ctx)

    def _choose_base_attack_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        base_attacks = _actions_of_kind(ctx.legal, "attack_base")
        if ctx.siege_finish and base_attacks:
            return base_attacks[0]
        viable_base_attacks = [action for action in base_attacks if _viable_base_attack(env, ctx.faction, action)]
        if viable_base_attacks:
            return viable_base_attacks[0]
        return None

    def _choose_emergency_production(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if ctx.base_under_siege:
            return ctx.emergency_train
        if _base_camp_targets(env, ctx.faction):
            return ctx.emergency_train
        if ctx.nearby_cavalry and self.profile.emergency_cavalry_prefers_archer:
            archer_action = ("train", "archer")
            if archer_action in ctx.legal:
                return archer_action
        if ctx.home_enemy_force + len(ctx.nearby_cavalry) > ctx.home_friendly_force:
            return ctx.emergency_train
        return None

    def _choose_economy_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if ctx.defense_mode or ctx.base_under_siege or ctx.siege_finish or ctx.rebuild_mode:
            return None
        if ctx.military:
            military_move_available = any(
                action[0] == "move_towards"
                and any(unit.id == action[1] for unit in ctx.military)
                for action in ctx.legal
            )
            if military_move_available and (ctx.push_mode or env.bank[ctx.faction] >= 100):
                return None
        if _unit_count(env, ctx.faction, "worker") < ctx.useful_worker_slots and ctx.spawn_actions:
            return ctx.spawn_actions[0]
        if _spawn_ring_blocked(env, ctx.faction):
            return None
        gather_by_worker = {action[1]: action for action in _actions_of_kind(ctx.legal, "gather")}
        for worker in self._ordered_workers(ctx.faction, ctx.workers):
            gather = gather_by_worker.get(worker.id)
            if gather is not None:
                return gather
        return None

    def _choose_production_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if ctx.siege_finish:
            return None
        if ctx.rebuild_mode:
            if ctx.emergency_train is not None:
                return ctx.emergency_train
            return None
        if ctx.base_under_siege and ctx.emergency_train is not None:
            return ctx.emergency_train
        if not ctx.defense_mode and "storehouse" not in ctx.buildings:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "storehouse"))
            if build_action is not None:
                return build_action
        if not ctx.defense_mode and "barracks" not in ctx.buildings:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "barracks"))
            if build_action is not None:
                return build_action
        if not ctx.defense_mode and "quarry" not in ctx.buildings and "masonry" in ctx.state.techs_unlocked:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "quarry"))
            if build_action is not None:
                return build_action
        if (
            not ctx.defense_mode
            and "market" not in ctx.buildings
            and "markets" in ctx.state.techs_unlocked
            and not (ctx.behind_mode and ctx.military_gap > 0)
        ):
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "market"))
            if build_action is not None:
                return build_action
        if not ctx.defense_mode and "stable" not in ctx.buildings and "horseback_riding" in ctx.state.techs_unlocked:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "stable"))
            if build_action is not None:
                return build_action
        if (
            (ctx.defense_mode or ctx.behind_mode or ctx.home_enemy_force > 0)
            and "construction" in ctx.state.techs_unlocked
            and "archer_tower" not in ctx.buildings
        ):
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "archer_tower"))
            if build_action is not None:
                return build_action
        if (
            (ctx.defense_mode or ctx.behind_mode or env.bases[ctx.faction].hp < env.config.base_hp)
            and "walls" in ctx.state.techs_unlocked
            and "wall" not in ctx.buildings
        ):
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "wall"))
            if build_action is not None:
                return build_action
        if ctx.defense_mode and "archer_tower" not in ctx.buildings and "construction" in ctx.state.techs_unlocked:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "archer_tower"))
            if build_action is not None:
                return build_action
        if ctx.defense_mode and "wall" not in ctx.buildings and "walls" in ctx.state.techs_unlocked:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "wall"))
            if build_action is not None:
                return build_action
        if _unit_count(env, ctx.faction, "soldier") < 2:
            desired = ("train", "soldier")
            if desired in ctx.legal:
                return desired
        desired_archers = self.profile.desired_archers if "fletching" in ctx.state.techs_unlocked else 0
        if _unit_count(env, ctx.faction, "archer") < desired_archers:
            desired = ("train", "archer")
            if desired in ctx.legal:
                return desired
        enemy_force = _total_force(env, ctx.enemy_faction)
        friendly_force = _total_force(env, ctx.faction)
        enemy_archers = _unit_count(env, ctx.enemy_faction, "archer")
        friendly_archers = _unit_count(env, ctx.faction, "archer")
        if not ctx.push_mode and (enemy_force > friendly_force or ctx.behind_mode):
            if enemy_archers > friendly_archers:
                desired = ("train", "archer")
                if desired in ctx.legal:
                    return desired
            desired = ("train", "soldier")
            if desired in ctx.legal:
                return desired
        desired_horsemen = self.profile.desired_horsemen if "horseback_riding" in ctx.state.techs_unlocked and "stable" in ctx.buildings else 0
        if _unit_count(env, ctx.faction, "horseman") < desired_horsemen and not ctx.behind_mode:
            desired = ("train", "horseman")
            if desired in ctx.legal:
                return desired
        if "construction" in ctx.state.techs_unlocked and "quarry" in ctx.buildings and "archer_tower" not in ctx.buildings:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "archer_tower"))
            if build_action is not None:
                return build_action
        if "engineering" in ctx.state.techs_unlocked and "archer_tower" in ctx.buildings and "ballista_tower" not in ctx.buildings:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "ballista_tower"))
            if build_action is not None:
                return build_action
        if (
            "engineering" in ctx.state.techs_unlocked
            and "siege_workshop" not in ctx.buildings
            and (ctx.push_mode or not ctx.behind_mode or friendly_force >= enemy_force)
        ):
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "siege_workshop"))
            if build_action is not None:
                return build_action
        if "stronghold" in ctx.state.techs_unlocked and "stronghold" not in ctx.buildings:
            build_action = _choose_closest_build_action(env, ctx.workers, _build_actions_for(ctx.legal, "stronghold"))
            if build_action is not None:
                return build_action
        if "heavy_cavalry" in ctx.state.techs_unlocked and "stable" in ctx.buildings:
            desired = ("train", "heavy_cavalry")
            if desired in ctx.legal and _unit_count(env, ctx.faction, "heavy_cavalry") < 1:
                return desired
        if "advanced_siege" in ctx.state.techs_unlocked and "siege_workshop" in ctx.buildings:
            desired = ("train", "ballista")
            if desired in ctx.legal and _unit_count(env, ctx.faction, "ballista") < 1:
                return desired
        return None

    def _choose_military_movement(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        if not ctx.military:
            return None
        enemy_units = [u for u in env.units if u.faction != ctx.faction]
        defensive_targets = _defensive_targets(env, ctx.faction)
        siege_reserved = {
            target
            for unit_id, target in self._military_targets.items()
            if any(unit.id == unit_id for unit in ctx.military) and env._in_bounds(target)
        }
        role_order = {"defender": 0, "raider": 1, "line": 2}
        ordered_military = sorted(
            ctx.military,
            key=lambda unit: (role_order[_military_role(env, ctx.faction, unit, ctx.military)], unit.id),
        )
        for unit in ordered_military:
            role = _military_role(env, ctx.faction, unit, ctx.military)
            stalled = self._military_stall.get(unit.id, 0) >= 3
            disengage_target = _archer_disengage_target(env, ctx.faction, unit)
            if disengage_target is not None:
                disengage_action = ("move_towards", unit.id, disengage_target)
                if disengage_action in ctx.legal:
                    return disengage_action
            recover_target = _recover_target(env, ctx.faction, unit)
            if recover_target is not None:
                recover_action = ("move_towards", unit.id, recover_target)
                if recover_action in ctx.legal:
                    return recover_action
            if (
                ctx.behind_mode
                and role != "raider"
                and not ctx.push_mode
                and len(ctx.military) < 3
            ):
                staging_target = _staging_target(env, ctx.faction, unit, ctx.rally_anchor)
                if staging_target is not None:
                    staging_action = ("move_towards", unit.id, staging_target)
                    if staging_action in ctx.legal:
                        self._set_military_target(unit.id, staging_target, lock=2)
                        return staging_action
            if ctx.rebuild_mode:
                defensive_targets = _defensive_targets(env, ctx.faction)
                if defensive_targets:
                    target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                else:
                    target = _home_guard_target(env, ctx.faction, unit)
                if target is not None and target != unit.position:
                    action = ("move_towards", unit.id, target)
                    if action in ctx.legal:
                        self._military_targets[unit.id] = target
                        return action
            if ctx.siege_finish:
                siege_target = _enemy_base_siege_target(env, ctx.faction, unit, reserved=siege_reserved - {unit.position})
                if siege_target is not None and unit.position != siege_target:
                    action = ("move_towards", unit.id, siege_target)
                    if action in ctx.legal:
                        self._set_military_target(unit.id, siege_target, lock=2)
                        siege_reserved.add(siege_target)
                        return action
            if ctx.push_mode and ctx.home_enemy_force == 0 and len(ctx.military) >= 4 and role == "defender":
                role = "line"
            target = self._military_targets.get(unit.id)
            if stalled:
                self._clear_military_target(unit.id)
                target = None
            if target is not None and self._military_target_lock.get(unit.id, 0) > 0:
                locked_action = ("move_towards", unit.id, target)
                if locked_action in ctx.legal:
                    return locked_action
                self._clear_military_target(unit.id)
                target = None
            if (
                not ctx.defense_mode
                and not ctx.push_mode
                and role != "raider"
                and ctx.rally_anchor is not None
                and unit.position != ctx.rally_anchor
                and not _army_ready_to_push(env, ctx.faction, _nearest_enemy_base(env, ctx.faction))
            ):
                rally_action = ("move_towards", unit.id, ctx.rally_anchor)
                if rally_action in ctx.legal:
                    return rally_action
            if role == "defender" and defensive_targets:
                target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                self._set_military_target(unit.id, target, lock=2)
            elif role == "defender":
                target = _home_guard_target(env, ctx.faction, unit)
                if target is not None:
                    self._set_military_target(unit.id, target, lock=2)
            elif ctx.push_mode and role != "raider":
                if target is None:
                    frontline_target = _frontline_target(env, ctx.faction, unit)
                    if frontline_target is not None and _frontline_is_losing_locally(env, ctx.faction, frontline_target):
                        staging_target = _staging_target(env, ctx.faction, unit, ctx.rally_anchor)
                        if staging_target is not None:
                            staging_action = ("move_towards", unit.id, staging_target)
                            if staging_action in ctx.legal:
                                self._set_military_target(unit.id, staging_target, lock=2)
                                return staging_action
                        target = frontline_target
                    else:
                        target = frontline_target or _nearest_enemy_base(env, ctx.faction)
                    self._set_military_target(unit.id, target, lock=2)
            elif target is None or all(enemy.position != target for enemy in enemy_units):
                if role == "raider":
                    frontline_target = _frontline_target(env, ctx.faction, unit)
                    if frontline_target is not None and _frontline_is_losing_locally(env, ctx.faction, frontline_target):
                        staging_target = _staging_target(env, ctx.faction, unit, ctx.rally_anchor)
                        if staging_target is not None:
                            staging_action = ("move_towards", unit.id, staging_target)
                            if staging_action in ctx.legal:
                                self._set_military_target(unit.id, staging_target, lock=1)
                                return staging_action
                        target = frontline_target
                    else:
                        target = _raid_target(env, ctx.faction, unit, enemy_units)
                        if target is None:
                            target = frontline_target or _nearest_enemy_base(env, ctx.faction)
                elif defensive_targets:
                    target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                elif enemy_units:
                    target = min(enemy_units, key=lambda enemy: _distance(unit.position, enemy.position)).position
                else:
                    target = _nearest_enemy_base(env, ctx.faction)
                self._set_military_target(unit.id, target, lock=2 if role != "raider" else 1)
            if stalled and enemy_units:
                target = _frontline_target(env, ctx.faction, unit) or min(
                    enemy_units,
                    key=lambda enemy: (_distance(unit.position, enemy.position), enemy.hp),
                ).position
                self._set_military_target(unit.id, target, lock=2)
            if role != "raider" and defensive_targets and target not in {enemy.position for enemy in defensive_targets}:
                target = min(defensive_targets, key=lambda enemy: _distance(unit.position, enemy.position)).position
                self._set_military_target(unit.id, target, lock=2)
            elif (
                not ctx.defense_mode
                and not ctx.push_mode
                and role == "line"
                and target == _nearest_enemy_base(env, ctx.faction)
                and not _army_ready_to_push(env, ctx.faction, target)
                and ctx.rally_anchor is not None
            ):
                target = ctx.rally_anchor
                self._set_military_target(unit.id, target, lock=2)
            elif (
                not ctx.defense_mode
                and not ctx.siege_finish
                and role != "raider"
                and target is not None
                and any(enemy.attack_damage > 0 for enemy in enemy_units)
                and not _army_ready_to_push(env, ctx.faction, target, minimum_units=3 if ctx.behind_mode else 2)
            ):
                staging_target = _staging_target(env, ctx.faction, unit, ctx.rally_anchor)
                staging_action = None
                if staging_target is not None:
                    staging_action = ("move_towards", unit.id, staging_target)
                if staging_action in ctx.legal:
                    target = staging_target
                    self._set_military_target(unit.id, target, lock=2)
            if target is not None:
                action = ("move_towards", unit.id, target)
                if action in ctx.legal:
                    return action
        return None

    def _choose_worker_action(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        ordered_workers = self._ordered_workers(ctx.faction, ctx.workers)
        if not ctx.defense_mode and not ctx.rebuild_mode:
            if _spawn_ring_blocked(env, ctx.faction):
                for worker in ordered_workers:
                    if not _is_on_spawn_ring(env, ctx.faction, worker.position):
                        continue
                    target = _nearest_resource(env, worker.position)
                    if target is not None and target != worker.position:
                        action = ("move_towards", worker.id, target)
                        if action in ctx.legal:
                            return action
                    target = _nearest_build_site(env, ctx.faction, worker.position)
                    if target is not None:
                        action = ("move_towards", worker.id, target)
                        if action in ctx.legal:
                            return action
            for worker in ordered_workers:
                target = _nearest_resource(env, worker.position)
                if target is not None:
                    action = ("move_towards", worker.id, target)
                    if action in ctx.legal:
                        return action
            build_site_needed = "storehouse" not in ctx.buildings or "barracks" not in ctx.buildings
            if build_site_needed:
                for worker in ordered_workers:
                    target = _nearest_build_site(env, ctx.faction, worker.position)
                    if target is None:
                        continue
                    action = ("move_towards", worker.id, target)
                    if action in ctx.legal:
                        return action
            return None
        return self._choose_worker_retreat(env, ctx)

    def _choose_worker_retreat(self, env: AgeGridEnv, ctx: _Context) -> Action | None:
        for worker in self._ordered_workers(ctx.faction, ctx.workers):
            retreat = _worker_retreat_target(env, ctx.faction, worker)
            if retreat is None:
                continue
            action = ("move_towards", worker.id, retreat)
            if action in ctx.legal:
                return action
        return None

    def act(self, env: AgeGridEnv) -> Action | None:
        faction = env.factions[env.current_player]
        legal = env.legal_actions(faction)
        if not legal:
            return None
        self._refresh_military_stall(env, faction)
        ctx = self._build_context(env, faction, legal)
        for chooser in (
            self._choose_diplomacy_action,
            self._choose_research_action,
            self._choose_collapse_recovery,
            self._choose_attack_action,
            self._choose_defense_action,
            self._choose_base_attack_action,
            self._choose_emergency_production,
            self._choose_production_action,
            self._choose_economy_action,
            self._choose_military_movement,
            self._choose_worker_action,
        ):
            action = chooser(env, ctx)
            if action is not None:
                self._record_worker_assignment(action, env, faction)
                return action
        if ctx.defense_mode or ctx.last_stand or ctx.rebuild_mode:
            fallback = _pressure_fallback_action(ctx)
            if fallback is not None:
                self._record_worker_assignment(fallback, env, faction)
                return fallback
        return None
