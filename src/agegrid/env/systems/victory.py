from __future__ import annotations


def _collapsed(env, faction: str) -> bool:
    workers = sum(1 for unit in env.units if unit.faction == faction and unit.unit_type == "worker")
    military = sum(1 for unit in env.units if unit.faction == faction and unit.attack_damage > 0)
    if workers > 0 or military > 0:
        return False
    return env.bank[faction] < env.config.worker_spawn_cost


def winner(env) -> str | None:
    if env.config.target_bank is not None:
        if env.bank["Red"] >= env.config.target_bank:
            return "Red"
        if env.bank["Blue"] >= env.config.target_bank:
            return "Blue"

    red_base_alive = env.bases["Red"].hp > 0
    blue_base_alive = env.bases["Blue"].hp > 0
    if red_base_alive and not blue_base_alive:
        return "Red"
    if blue_base_alive and not red_base_alive:
        return "Blue"
    if env.config.collapse_enabled:
        if _collapsed(env, "Red") and not _collapsed(env, "Blue"):
            return "Blue"
        if _collapsed(env, "Blue") and not _collapsed(env, "Red"):
            return "Red"

    return None
