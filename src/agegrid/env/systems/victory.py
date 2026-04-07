from __future__ import annotations


def winner(env) -> str | None:
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

    return None

