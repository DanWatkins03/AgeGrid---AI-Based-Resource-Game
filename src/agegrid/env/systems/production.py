from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from src.agegrid.env.entities import Position


@dataclass(frozen=True)
class UnitDefinition:
    cost: int
    hp: int
    attack_damage: int = 0
    attack_range: int = 0
    move_steps: int = 1
    required_tech: str | None = None
    required_building: str | None = None


@dataclass(frozen=True)
class BuildingDefinition:
    cost: int
    hp: int
    attack_damage: int = 0
    attack_range: int = 0
    required_tech: str | None = None
    required_building: str | None = None
    resource_income: int = 0
    required_resource_adjacent: str | None = None


UNIT_DEFS: dict[str, UnitDefinition] = {
    "worker": UnitDefinition(cost=20, hp=5),
    "soldier": UnitDefinition(
        cost=30,
        hp=10,
        attack_damage=3,
        attack_range=1,
        required_tech="bronze_working",
        required_building="barracks",
    ),
    "archer": UnitDefinition(
        cost=36,
        hp=8,
        attack_damage=3,
        attack_range=3,
        required_tech="fletching",
        required_building="barracks",
    ),
    "horseman": UnitDefinition(
        cost=34,
        hp=12,
        attack_damage=4,
        attack_range=1,
        move_steps=3,
        required_tech="horsemanship",
        required_building="stable",
    ),
}

BUILDING_DEFS: dict[str, BuildingDefinition] = {
    "storehouse": BuildingDefinition(cost=36, hp=18, required_tech="mining", resource_income=3),
    "barracks": BuildingDefinition(cost=48, hp=30, required_tech="bronze_working"),
    "quarry": BuildingDefinition(
        cost=34,
        hp=20,
        required_tech="mining",
        required_resource_adjacent="stone",
        resource_income=4,
    ),
    "stable": BuildingDefinition(
        cost=36,
        hp=24,
        required_tech="horsemanship",
        required_resource_adjacent="horses",
    ),
    "archer_tower": BuildingDefinition(
        cost=42,
        hp=22,
        attack_damage=3,
        attack_range=3,
        required_tech="masonry",
        required_building="quarry",
    ),
    "ballista_tower": BuildingDefinition(
        cost=62,
        hp=28,
        attack_damage=5,
        attack_range=4,
        required_tech="engineering",
        required_building="archer_tower",
    ),
}


def unit_stats(env, faction: str, unit_type: str) -> UnitDefinition | None:
    spec = UNIT_DEFS.get(unit_type)
    if spec is None:
        return None
    techs = env.faction_state(faction).techs_unlocked
    if unit_type == "soldier" and "iron_working" in techs:
        return replace(spec, hp=spec.hp + 2, attack_damage=spec.attack_damage + 1)
    if unit_type == "archer" and "engineering" in techs:
        return replace(spec, hp=spec.hp + 1, attack_range=spec.attack_range + 1)
    if unit_type == "horseman" and "stirrups" in techs:
        return replace(spec, hp=spec.hp + 2, attack_damage=spec.attack_damage + 1, move_steps=spec.move_steps + 1)
    return spec


def building_stats(env, faction: str, building_type: str) -> BuildingDefinition | None:
    spec = BUILDING_DEFS.get(building_type)
    if spec is None:
        return None
    techs = env.faction_state(faction).techs_unlocked
    if building_type == "archer_tower" and "fortification" in techs:
        return replace(spec, hp=spec.hp + 4, attack_damage=spec.attack_damage + 1)
    if building_type == "ballista_tower" and "fortification" in techs:
        return replace(spec, hp=spec.hp + 4, attack_damage=spec.attack_damage + 1, attack_range=spec.attack_range + 1)
    return spec


def _can_afford(env, faction: str, cost: int) -> bool:
    return env.faction_state(faction).resources >= cost


def _has_required_tech(env, faction: str, required_tech: str | None) -> bool:
    return required_tech is None or required_tech in env.faction_state(faction).techs_unlocked


def _has_required_building(env, faction: str, required_building: str | None) -> bool:
    if required_building is None:
        return True
    return any(b.faction == faction and b.building_type == required_building for b in env.buildings)


def _adjacent_spawn_positions(env, faction: str) -> list[Position]:
    base_pos = env.bases[faction].position
    return [
        (base_pos[0] + 1, base_pos[1]),
        (base_pos[0] - 1, base_pos[1]),
        (base_pos[0], base_pos[1] + 1),
        (base_pos[0], base_pos[1] - 1),
    ]


def can_train_unit(env, faction: str, unit_type: str) -> bool:
    spec = unit_stats(env, faction, unit_type)
    if spec is None:
        return False
    cost = env.config.worker_spawn_cost if unit_type == "worker" else spec.cost
    if not _can_afford(env, faction, cost):
        return False
    if not _has_required_tech(env, faction, spec.required_tech):
        return False
    if not _has_required_building(env, faction, spec.required_building):
        return False

    workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
    if unit_type == "worker" and len(workers) >= env.config.max_workers:
        return False

    occ = env._occupied_positions()
    return any(env._in_bounds(pos) and pos not in occ for pos in _adjacent_spawn_positions(env, faction))


def train_unit(env, faction: str, unit_type: str) -> bool:
    spec = unit_stats(env, faction, unit_type)
    if spec is None or not can_train_unit(env, faction, unit_type):
        return False

    cost = env.config.worker_spawn_cost if unit_type == "worker" else spec.cost
    occ = env._occupied_positions()
    for pos in _adjacent_spawn_positions(env, faction):
        if env._in_bounds(pos) and pos not in occ:
            env.faction_state(faction).resources -= cost
            env._spawn_unit(
                faction=faction,
                unit_type=unit_type,
                hp=spec.hp,
                pos=pos,
                attack_damage=spec.attack_damage,
                attack_range=spec.attack_range,
                move_steps=spec.move_steps,
            )
            return True
    return False


def spawn_worker(env, faction: str) -> bool:
    return train_unit(env, faction, "worker")


def can_build(env, faction: str, worker_id: int, building_type: str, pos: Position) -> bool:
    spec = building_stats(env, faction, building_type)
    worker = next((u for u in env.units if u.id == worker_id), None)
    if spec is None or worker is None:
        return False
    if worker.faction != faction or worker.unit_type != "worker":
        return False
    if not _can_afford(env, faction, spec.cost):
        return False
    if not _has_required_tech(env, faction, spec.required_tech):
        return False
    if not _has_required_building(env, faction, spec.required_building):
        return False
    if abs(worker.position[0] - pos[0]) + abs(worker.position[1] - pos[1]) != 1:
        return False
    if not env._in_bounds(pos) or pos in env._occupied_positions() or pos in {b.position for b in env.buildings}:
        return False
    if spec.required_resource_adjacent is not None:
        adjacent_resources = [
            env.resource_at_for_faction((pos[0] + 1, pos[1]), faction),
            env.resource_at_for_faction((pos[0] - 1, pos[1]), faction),
            env.resource_at_for_faction((pos[0], pos[1] + 1), faction),
            env.resource_at_for_faction((pos[0], pos[1] - 1), faction),
        ]
        if not any(
            resource is not None and resource.resource_type == spec.required_resource_adjacent
            for resource in adjacent_resources
        ):
            return False
    return True


def build(env, faction: str, worker_id: int, building_type: str, pos: Position) -> bool:
    spec = building_stats(env, faction, building_type)
    if spec is None or not can_build(env, faction, worker_id, building_type, pos):
        return False

    env.faction_state(faction).resources -= spec.cost
    env._spawn_building(
        faction=faction,
        building_type=building_type,
        hp=spec.hp,
        pos=pos,
        attack_damage=spec.attack_damage,
        attack_range=spec.attack_range,
    )
    return True
