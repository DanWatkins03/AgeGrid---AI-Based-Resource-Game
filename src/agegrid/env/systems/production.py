from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from src.agegrid.env.entities import Position
from src.agegrid.env import hexgrid
from src.agegrid.env.systems import tech


@dataclass(frozen=True)
class UnitDefinition:
    cost: int
    hp: int
    attack_damage: int = 0
    attack_range: int = 0
    move_steps: int = 1
    required_tech: str | None = None
    required_building: str | None = None
    tags: tuple[str, ...] = ()


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
    tags: tuple[str, ...] = ()


UNIT_DEFS: dict[str, UnitDefinition] = {
    "worker": UnitDefinition(cost=20, hp=5),
    "soldier": UnitDefinition(
        cost=30,
        hp=10,
        attack_damage=3,
        attack_range=1,
        required_tech="bronze",
        required_building="barracks",
        tags=("military", "infantry"),
    ),
    "archer": UnitDefinition(
        cost=36,
        hp=8,
        attack_damage=3,
        attack_range=3,
        required_tech="fletching",
        required_building="barracks",
        tags=("military", "ranged"),
    ),
    "horseman": UnitDefinition(
        cost=34,
        hp=12,
        attack_damage=4,
        attack_range=1,
        move_steps=3,
        required_tech="horseback_riding",
        required_building="stable",
        tags=("military", "cavalry"),
    ),
    "heavy_cavalry": UnitDefinition(
        cost=48,
        hp=16,
        attack_damage=5,
        attack_range=1,
        move_steps=3,
        required_tech="heavy_cavalry",
        required_building="stable",
        tags=("military", "cavalry", "elite"),
    ),
    "ballista": UnitDefinition(
        cost=52,
        hp=10,
        attack_damage=5,
        attack_range=4,
        move_steps=1,
        required_tech="advanced_siege",
        required_building="siege_workshop",
        tags=("military", "siege", "ranged"),
    ),
}

BUILDING_DEFS: dict[str, BuildingDefinition] = {
    "storehouse": BuildingDefinition(cost=36, hp=18, required_tech="mining", resource_income=3, tags=("economy",)),
    "barracks": BuildingDefinition(cost=48, hp=30, required_tech="bronze", tags=("military",)),
    "quarry": BuildingDefinition(
        cost=34,
        hp=20,
        required_tech="masonry",
        required_resource_adjacent="stone",
        resource_income=4,
        tags=("economy",),
    ),
    "stable": BuildingDefinition(
        cost=36,
        hp=24,
        required_tech="horseback_riding",
        required_resource_adjacent="horses",
        tags=("military",),
    ),
    "archer_tower": BuildingDefinition(
        cost=42,
        hp=22,
        attack_damage=3,
        attack_range=3,
        required_tech="construction",
        required_building="quarry",
        tags=("defense",),
    ),
    "ballista_tower": BuildingDefinition(
        cost=62,
        hp=28,
        attack_damage=5,
        attack_range=4,
        required_tech="engineering",
        required_building="archer_tower",
        tags=("defense",),
    ),
    "market": BuildingDefinition(
        cost=54,
        hp=20,
        required_tech="markets",
        required_building="storehouse",
        resource_income=4,
        tags=("economy",),
    ),
    "wall": BuildingDefinition(
        cost=24,
        hp=34,
        required_tech="walls",
        required_building="quarry",
        tags=("defense",),
    ),
    "stronghold": BuildingDefinition(
        cost=78,
        hp=42,
        attack_damage=4,
        attack_range=3,
        required_tech="stronghold",
        required_building="wall",
        tags=("defense",),
    ),
    "siege_workshop": BuildingDefinition(
        cost=64,
        hp=24,
        required_tech="engineering",
        required_building="barracks",
        tags=("military",),
    ),
}

def _apply_discount(cost: int, discount_pct: int) -> int:
    if discount_pct <= 0:
        return cost
    return max(1, (cost * max(0, 100 - discount_pct)) // 100)


def unit_cost(env, faction: str, unit_type: str) -> int | None:
    spec = UNIT_DEFS.get(unit_type)
    if spec is None:
        return None
    if unit_type == "worker":
        base_cost = env.config.worker_spawn_cost
    else:
        base_cost = spec.cost
    discount = tech.passive_modifier_total(env, faction, "unit_cost_discount_pct")
    if "military" in spec.tags:
        discount += tech.passive_modifier_total(env, faction, "military_cost_discount_pct")
    return _apply_discount(base_cost, discount)


def building_cost(env, faction: str, building_type: str) -> int | None:
    spec = BUILDING_DEFS.get(building_type)
    if spec is None:
        return None
    discount = tech.passive_modifier_total(env, faction, "building_cost_discount_pct")
    return _apply_discount(spec.cost, discount)


def unit_stats(env, faction: str, unit_type: str) -> UnitDefinition | None:
    spec = UNIT_DEFS.get(unit_type)
    if spec is None:
        return None

    hp = spec.hp
    attack_damage = spec.attack_damage
    attack_range = spec.attack_range
    move_steps = spec.move_steps

    if unit_type == "soldier":
        hp += tech.passive_modifier_total(env, faction, "soldier_hp_bonus")
        attack_damage += tech.passive_modifier_total(env, faction, "soldier_attack_bonus")
    if unit_type == "archer":
        attack_damage += tech.passive_modifier_total(env, faction, "archer_attack_bonus")
        attack_range += tech.passive_modifier_total(env, faction, "archer_range_bonus")
    if "cavalry" in spec.tags:
        hp += tech.passive_modifier_total(env, faction, "cavalry_hp_bonus")
        attack_damage += tech.passive_modifier_total(env, faction, "cavalry_attack_bonus")
        move_steps += tech.passive_modifier_total(env, faction, "cavalry_move_bonus")

    return replace(spec, hp=hp, attack_damage=attack_damage, attack_range=attack_range, move_steps=move_steps)


def building_stats(env, faction: str, building_type: str) -> BuildingDefinition | None:
    spec = BUILDING_DEFS.get(building_type)
    if spec is None:
        return None

    hp = spec.hp + tech.passive_modifier_total(env, faction, "building_hp_bonus")
    attack_damage = spec.attack_damage
    attack_range = spec.attack_range
    resource_income = spec.resource_income

    if "defense" in spec.tags:
        attack_damage += tech.passive_modifier_total(env, faction, "tower_damage_bonus")
        attack_range += tech.passive_modifier_total(env, faction, "tower_range_bonus")
    if building_type == "storehouse":
        resource_income += tech.passive_modifier_total(env, faction, "storehouse_income_bonus")
    if building_type == "quarry":
        resource_income += tech.passive_modifier_total(env, faction, "quarry_income_bonus")
    if "economy" in spec.tags:
        resource_income += tech.passive_modifier_total(env, faction, "economy_income_bonus")
        multiplier = 100 + tech.passive_modifier_total(env, faction, "passive_income_multiplier_pct")
        resource_income = max(0, (resource_income * multiplier) // 100)

    return replace(
        spec,
        hp=hp,
        attack_damage=attack_damage,
        attack_range=attack_range,
        resource_income=resource_income,
    )


def _can_afford(env, faction: str, cost: int) -> bool:
    return env.faction_state(faction).resources >= cost


def _has_required_tech(env, faction: str, required_tech: str | None) -> bool:
    return required_tech is None or required_tech in env.faction_state(faction).techs_unlocked


def _has_required_building(env, faction: str, required_building: str | None) -> bool:
    if required_building is None:
        return True
    return any(building.building_type == required_building for building in env.get_buildings_for_faction(faction))


def _adjacent_spawn_positions(env, faction: str) -> list[Position]:
    base_pos = env.bases[faction].position
    return hexgrid.neighbors(base_pos)


def can_train_unit(env, faction: str, unit_type: str) -> bool:
    spec = unit_stats(env, faction, unit_type)
    cost = unit_cost(env, faction, unit_type)
    if spec is None or cost is None:
        return False
    if not _can_afford(env, faction, cost):
        return False
    if not _has_required_tech(env, faction, spec.required_tech):
        return False
    if not _has_required_building(env, faction, spec.required_building):
        return False

    workers = [unit for unit in env.get_units_for_faction(faction) if unit.unit_type == "worker"]
    if unit_type == "worker" and len(workers) >= env.config.max_workers:
        return False

    occ = env._unit_positions()
    return any(env._in_bounds(pos) and pos not in occ for pos in _adjacent_spawn_positions(env, faction))


def train_unit(env, faction: str, unit_type: str) -> bool:
    spec = unit_stats(env, faction, unit_type)
    cost = unit_cost(env, faction, unit_type)
    if spec is None or cost is None or not can_train_unit(env, faction, unit_type):
        return False

    occ = env._unit_positions()
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
    cost = building_cost(env, faction, building_type)
    worker = env.get_unit(worker_id)
    if spec is None or cost is None or worker is None:
        return False
    if worker.faction != faction or worker.unit_type != "worker":
        return False
    if not _can_afford(env, faction, cost):
        return False
    if not _has_required_tech(env, faction, spec.required_tech):
        return False
    if not _has_required_building(env, faction, spec.required_building):
        return False
    if hexgrid.distance(worker.position, pos) != 1:
        return False
    if not env._in_bounds(pos) or pos in env._construction_blocked_positions(faction):
        return False
    if env.resource_at(pos) is not None:
        return False
    if spec.required_resource_adjacent is not None:
        adjacent_resources = [env.resource_at_for_faction(neighbor, faction) for neighbor in hexgrid.neighbors(pos)]
        if not any(
            resource is not None and resource.resource_type == spec.required_resource_adjacent
            for resource in adjacent_resources
        ):
            return False
    return True


def build(env, faction: str, worker_id: int, building_type: str, pos: Position) -> bool:
    spec = building_stats(env, faction, building_type)
    cost = building_cost(env, faction, building_type)
    if spec is None or cost is None or not can_build(env, faction, worker_id, building_type, pos):
        return False

    env.faction_state(faction).resources -= cost
    env._spawn_building(
        faction=faction,
        building_type=building_type,
        hp=spec.hp,
        pos=pos,
        attack_damage=spec.attack_damage,
        attack_range=spec.attack_range,
    )
    return True
