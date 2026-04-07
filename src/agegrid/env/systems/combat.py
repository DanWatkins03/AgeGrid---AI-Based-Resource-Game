from __future__ import annotations


def attack(env, faction: str, attacker_id: int, target_id: int) -> bool:
    attacker = next((u for u in env.units if u.id == attacker_id), None)
    target = next((u for u in env.units if u.id == target_id), None)

    if attacker is None or target is None:
        return False
    if attacker.faction != faction or target.faction == faction:
        return False
    if attacker.attack_damage <= 0 or attacker.attack_range <= 0:
        return False

    distance = abs(attacker.position[0] - target.position[0]) + abs(attacker.position[1] - target.position[1])
    if distance > attacker.attack_range:
        return False

    target.hp -= attacker.attack_damage
    if target.hp <= 0:
        env._remove_unit(target.id)

    return True


def attack_base(env, faction: str, attacker_id: int, target_faction: str) -> bool:
    attacker = next((u for u in env.units if u.id == attacker_id), None)
    target_base = env.bases.get(target_faction)

    if attacker is None or target_base is None:
        return False
    if attacker.faction != faction or target_faction == faction:
        return False
    if target_base.hp <= 0:
        return False
    if attacker.attack_damage <= 0 or attacker.attack_range <= 0:
        return False

    distance = abs(attacker.position[0] - target_base.position[0]) + abs(attacker.position[1] - target_base.position[1])
    if distance > attacker.attack_range:
        return False

    target_base.hp = max(0, target_base.hp - attacker.attack_damage)
    return True
