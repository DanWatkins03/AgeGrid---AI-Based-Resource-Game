from __future__ import annotations
from typing import Tuple

from src.agegrid.env.systems import production

Position = Tuple[int, int]

# Gather Resources

def gather(env, worker_id: int) -> bool:
    unit = next((u for u in env.units if u.id == worker_id), None)
    if unit is None or unit.unit_type != "worker":
        return False

    node = env._resource_at(unit.position)
    if node is None:
        return False

    amount = min(env.config.worker_gather_amount, node.remaining)
    node.remaining -= amount
    env.faction_state(unit.faction).resources += amount
    return True

# Spend resources to recruit or "spawn" a worker

def spawn_worker(env, faction: str) -> bool:
    return production.spawn_worker(env, faction)
