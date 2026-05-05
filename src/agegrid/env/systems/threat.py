from __future__ import annotations

from dataclasses import dataclass

from src.agegrid.env import hexgrid
from src.agegrid.env.entities import Position


@dataclass(frozen=True)
class ThreatMap:
    faction: str
    enemy_faction: str
    enemy_attack: dict[Position, int]
    friendly_cover: dict[Position, int]
    enemy_pressure: tuple[Position, ...]
    contested_resources: tuple[Position, ...]

    def enemy_threat_at(self, pos: Position) -> int:
        return self.enemy_attack.get(pos, 0)

    def friendly_cover_at(self, pos: Position) -> int:
        return self.friendly_cover.get(pos, 0)

    def danger_at(self, pos: Position) -> int:
        return max(0, self.enemy_threat_at(pos) - self.friendly_cover_at(pos))


def _positions_in_range(env, center: Position, attack_range: int) -> list[Position]:
    return [
        (x, y)
        for x in range(env.config.width)
        for y in range(env.config.height)
        if hexgrid.distance(center, (x, y)) <= attack_range
    ]


def _add_attack_zone(env, zones: dict[Position, int], center: Position, attack_range: int, attack_damage: int) -> None:
    if attack_damage <= 0 or attack_range < 0:
        return
    for pos in _positions_in_range(env, center, attack_range):
        zones[pos] = zones.get(pos, 0) + attack_damage


def build_threat_map(env, faction: str) -> ThreatMap:
    enemy = next(name for name in env.factions if name != faction)
    enemy_attack: dict[Position, int] = {}
    friendly_cover: dict[Position, int] = {}
    enemy_pressure: list[Position] = []

    for unit in env.units:
        if unit.attack_damage <= 0:
            continue
        if unit.faction == faction:
            _add_attack_zone(env, friendly_cover, unit.position, unit.attack_range, unit.attack_damage)
        elif env.at_war(faction, unit.faction):
            enemy_pressure.append(unit.position)
            _add_attack_zone(env, enemy_attack, unit.position, unit.attack_range, unit.attack_damage)

    for building in env.buildings:
        if building.hp <= 0 or building.attack_damage <= 0:
            continue
        if building.faction == faction:
            _add_attack_zone(env, friendly_cover, building.position, building.attack_range, building.attack_damage)
        elif env.at_war(faction, building.faction):
            enemy_pressure.append(building.position)
            _add_attack_zone(env, enemy_attack, building.position, building.attack_range, building.attack_damage)

    contested_resources = tuple(
        resource.position
        for resource in env.visible_resources(faction)
        if env.resource_is_contested(resource, faction)
    )
    enemy_pressure.extend(contested_resources)

    return ThreatMap(
        faction=faction,
        enemy_faction=enemy,
        enemy_attack=enemy_attack,
        friendly_cover=friendly_cover,
        enemy_pressure=tuple(sorted(set(enemy_pressure))),
        contested_resources=tuple(sorted(set(contested_resources))),
    )
