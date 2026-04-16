from __future__ import annotations

from src.agegrid.env import hexgrid
from src.agegrid.env.systems import tech


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return hexgrid.distance(a, b)


def _structure_target_priority(structure_pos: tuple[int, int], unit) -> tuple[int, int, int]:
    threat_rank = 0 if unit.attack_damage > 0 else 1
    return (threat_rank, unit.hp, _distance(structure_pos, unit.position))


def _structure_attack_stats(env, building) -> tuple[int, int]:
    damage = building.attack_damage
    attack_range = building.attack_range
    damage += tech.passive_modifier_total(env, building.faction, "tower_damage_bonus")
    attack_range += tech.passive_modifier_total(env, building.faction, "tower_range_bonus")
    return damage, attack_range


def attack(env, faction: str, attacker_id: int, target_id: int) -> bool:
    attacker = env.get_unit(attacker_id)
    target = env.get_unit(target_id)

    if attacker is None or target is None:
        return False
    if attacker.faction != faction or target.faction == faction:
        return False
    if not env.at_war(faction, target.faction):
        return False
    if attacker.attack_damage <= 0 or attacker.attack_range <= 0:
        return False
    if target.unit_type == "worker" and env.turn < env.config.worker_peace_until_turn:
        return False

    distance = _distance(attacker.position, target.position)
    if distance > attacker.attack_range:
        return False

    target.hp -= attacker.attack_damage
    if target.hp <= 0:
        env._remove_unit(target.id)

    return True


def attack_base(env, faction: str, attacker_id: int, target_faction: str) -> bool:
    attacker = env.get_unit(attacker_id)
    target_base = env.bases.get(target_faction)

    if attacker is None or target_base is None:
        return False
    if attacker.faction != faction or target_faction == faction:
        return False
    if not env.at_war(faction, target_faction):
        return False
    if target_base.hp <= 0:
        return False
    if attacker.attack_damage <= 0 or attacker.attack_range <= 0:
        return False
    if env.turn < env.config.base_peace_until_turn:
        return False

    distance = _distance(attacker.position, target_base.position)
    if distance > attacker.attack_range:
        return False

    target_base.hp = max(0, target_base.hp - attacker.attack_damage)
    return True


def _base_attack_stats(env, faction: str) -> tuple[int, int]:
    damage = env.config.base_attack_damage
    attack_range = env.config.base_attack_range
    damage += tech.passive_modifier_total(env, faction, "base_attack_bonus")
    attack_range += tech.passive_modifier_total(env, faction, "base_attack_range_bonus")
    return damage, attack_range


def resolve_defensive_fire(env, faction: str) -> list[str]:
    events: list[str] = []
    enemy_units = env.get_enemy_units(faction)
    if not enemy_units:
        return events

    base = env.bases[faction]
    if base.hp > 0:
        base_damage, base_range = _base_attack_stats(env, faction)
        targets = [unit for unit in enemy_units if _distance(base.position, unit.position) <= base_range]
        if targets:
            target = min(targets, key=lambda unit: _structure_target_priority(base.position, unit))
            target.hp -= base_damage
            if target.hp <= 0:
                env._remove_unit(target.id)
                events.append(f"{faction} base shot down {target.faction} {target.unit_type}#{target.id}")
            else:
                events.append(f"{faction} base hit {target.faction} {target.unit_type}#{target.id} ({target.hp} hp left)")

    for building in env.buildings:
        if building.faction != faction or building.hp <= 0 or building.attack_damage <= 0 or building.attack_range <= 0:
            continue
        building_damage, building_range = _structure_attack_stats(env, building)
        current_targets = [
            unit for unit in env.units if unit.faction != faction and _distance(building.position, unit.position) <= building_range
        ]
        if not current_targets:
            continue
        target = min(current_targets, key=lambda unit: _structure_target_priority(building.position, unit))
        target.hp -= building_damage
        if target.hp <= 0:
            env._remove_unit(target.id)
            events.append(
                f"{faction} {building.building_type} shot down {target.faction} {target.unit_type}#{target.id}"
            )
        else:
            events.append(
                f"{faction} {building.building_type} hit {target.faction} {target.unit_type}#{target.id} ({target.hp} hp left)"
            )

    return events
