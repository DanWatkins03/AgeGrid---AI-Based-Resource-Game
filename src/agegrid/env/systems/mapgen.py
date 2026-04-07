from __future__ import annotations
from typing import List, Tuple
from src.agegrid.env.entities import ResourceNode

Position = Tuple[int, int]


def _place_symmetric_resource_group(
    env,
    n: int,
    remaining: int,
    rid_start: int,
    used: set[Position],
    resource_type: str,
    required_tech: str | None = None,
) -> tuple[list[ResourceNode], int]:
    if n % 2 == 1:
        n += 1

    resources: List[ResourceNode] = []
    rid = rid_start
    attempts = 0
    max_x = max(0, env.config.width // 2 - 1)
    forbidden = {env.bases["Red"].position, env.bases["Blue"].position}

    while len(resources) < n and attempts < 10_000:
        attempts += 1

        x = env.rng.randint(0, max_x)
        y = env.rng.randint(0, env.config.height - 1)

        p1 = (x, y)
        p2 = env._mirror(p1)

        if p1 in used or p2 in used or p1 == p2:
            continue

        used.add(p1)
        used.add(p2)

        resources.append(
            ResourceNode(
                id=rid,
                position=p1,
                remaining=remaining,
                resource_type=resource_type,
                required_tech=required_tech,
            )
        )
        rid += 1
        resources.append(
            ResourceNode(
                id=rid,
                position=p2,
                remaining=remaining,
                resource_type=resource_type,
                required_tech=required_tech,
            )
        )
        rid += 1

    if len(resources) < n:
        raise RuntimeError("Failed to place symmetric resources. Try a different seed/config.")

    return resources, rid


def place_symmetric_resources(env, n: int, remaining: int) -> List[ResourceNode]:
    forbidden = {env.bases["Red"].position, env.bases["Blue"].position}
    used: set[Position] = set(forbidden)
    resources, next_rid = _place_symmetric_resource_group(env, n, remaining, 1, used, "ore")

    stone_nodes = getattr(env.config, "stone_resource_nodes", 0)
    stone_remaining = getattr(env.config, "stone_resource_amount", remaining)
    if stone_nodes > 0:
        stone_resources, next_rid = _place_symmetric_resource_group(
            env,
            stone_nodes,
            stone_remaining,
            next_rid,
            used,
            "stone",
            required_tech="mining",
        )
        resources.extend(stone_resources)

    horse_nodes = getattr(env.config, "horse_resource_nodes", 0)
    horse_remaining = getattr(env.config, "horse_resource_amount", remaining)
    if horse_nodes > 0:
        horse_resources, next_rid = _place_symmetric_resource_group(
            env,
            horse_nodes,
            horse_remaining,
            next_rid,
            used,
            "horses",
            required_tech="horsemanship",
        )
        resources.extend(horse_resources)

    return resources
