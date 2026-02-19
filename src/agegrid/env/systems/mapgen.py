from __future__ import annotations
from typing import List, Tuple
from src.agegrid.env.entities import ResourceNode

Position = Tuple[int, int]


def place_symmetric_resources(env, n: int, remaining: int) -> List[ResourceNode]:
    # Ensure even number for symmetry
    if n % 2 == 1:
        n += 1

    forbidden = {env.bases["Red"].position, env.bases["Blue"].position}
    used: set[Position] = set(forbidden)

    resources: List[ResourceNode] = []
    rid = 1
    attempts = 0

    while len(resources) < n and attempts < 10_000:
        attempts += 1

        x = env.rng.randint(0, env.config.width // 2 - 1)
        y = env.rng.randint(0, env.config.height - 1)

        p1 = (x, y)
        p2 = env._mirror(p1)

        if p1 in used or p2 in used or p1 == p2:
            continue

        used.add(p1)
        used.add(p2)

        resources.append(ResourceNode(id=rid, position=p1, remaining=remaining))
        rid += 1
        resources.append(ResourceNode(id=rid, position=p2, remaining=remaining))
        rid += 1

    if len(resources) < n:
        raise RuntimeError("Failed to place symmetric resources. Try a different seed/config.")

    return resources