from __future__ import annotations

from dataclasses import dataclass

from src.agegrid.env.entities import Position


@dataclass(frozen=True)
class UnitDefinition:
    cost: int
    hp: int
    attack_damage: int = 0
    attack_range: int = 0
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


UNIT_DEFS: dict[str, UnitDefinition] = {
    "worker": UnitDefinition(cost=20, hp=5),
    "soldier": UnitDefinition(
        cost=35,
        hp=10,
        attack_damage=3,
        attack_range=1,
        required_tech="bronze_working",
        required_building="barracks",
    ),
    "archer": UnitDefinition(
        cost=45,
        hp=8,
        attack_damage=2,
        attack_range=3,
        required_tech="fletching",
        required_building="barracks",
    ),
}

BUILDING_DEFS: dict[str, BuildingDefinition] = {
    "storehouse": BuildingDefinition(cost=45, hp=18, required_tech="mining", resource_income=2),
    "barracks": BuildingDefinition(cost=60, hp=30, required_tech="bronze_working"),
    "turret": BuildingDefinition(cost=50, hp=20, attack_damage=2, attack_range=2, required_tech="masonry"),
}


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
    spec = UNIT_DEFS.get(unit_type)
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
    spec = UNIT_DEFS.get(unit_type)
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
            )
            return True
    return False


def spawn_worker(env, faction: str) -> bool:
    return train_unit(env, faction, "worker")


def can_build(env, faction: str, worker_id: int, building_type: str, pos: Position) -> bool:
    spec = BUILDING_DEFS.get(building_type)
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
    return True


def build(env, faction: str, worker_id: int, building_type: str, pos: Position) -> bool:
    spec = BUILDING_DEFS.get(building_type)
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
