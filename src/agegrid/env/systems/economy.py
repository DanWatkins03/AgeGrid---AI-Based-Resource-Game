from __future__ import annotations
from typing import Tuple

from src.agegrid.env.systems import production, tech

Position = Tuple[int, int]

# Gather Resources

def gather(env, worker_id: int) -> bool:
    unit = env.get_unit(worker_id)
    if unit is None or unit.unit_type != "worker":
        return False

    node = env.resource_at_for_faction(unit.position, unit.faction)
    if node is None:
        return False

    amount = env.config.worker_gather_amount + tech.passive_modifier_total(env, unit.faction, "worker_gather_bonus")
    env.faction_state(unit.faction).resources += amount
    return True

# Spend resources to recruit or "spawn" a worker

def spawn_worker(env, faction: str) -> bool:
    return production.spawn_worker(env, faction)
