from __future__ import annotations
from typing import Tuple

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
    env.bank[unit.faction] += amount
    return True

# Spend resources to recruit or "spawn" a worker

def spawn_worker(env, faction: str) -> bool:
    # Check if workers exceed the max amount
    workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
    if len(workers) >= env.config.max_workers:
        return False
    
    # Check if they can afford to recruit a worker
    cost = env.config.worker_spawn_cost
    if env.bank[faction] < cost:
        return False
    
    base_pos = env.bases[faction].position
    candidates: list[Position] = [
        (base_pos[0] + 1, base_pos[1]),
        (base_pos[0] -1, base_pos[1]),
        (base_pos[0], base_pos[1] + 1),
        (base_pos[0], base_pos[1] - 1)
    ]

    occ = env._occupied_positions()
    for pos in candidates:
        if env._in_bounds(pos) and pos not in occ:
            env.bank[faction] -= cost
            env._spawn_worker(faction, pos)
            return True
        
    return False