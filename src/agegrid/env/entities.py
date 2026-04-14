from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Position = Tuple[int, int]


@dataclass
class ResourceNode:
    id: int
    position: Position
    remaining: int
    resource_type: str = "ore"
    required_tech: str | None = None


@dataclass
class Unit:
    id: int
    faction: str
    unit_type: str  # "worker" or "soldier"
    hp: int
    position: Position
    attack_damage: int = 0
    attack_range: int = 0
    move_steps: int = 1


@dataclass
class Building:
    id: int
    faction: str
    building_type: str  # "archer_tower", "ballista_tower", etc.
    hp: int
    position: Position
    attack_damage: int
    attack_range: int


@dataclass
class Base:
    faction: str
    hp: int
    position: Position


