from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Position = Tuple[int, int]


@dataclass(init=False)
class ResourceNode:
    id: int
    position: Position
    abundance: int
    resource_type: str = "ore"
    required_tech: str | None = None

    def __init__(
        self,
        id: int,
        position: Position,
        abundance: int | None = None,
        resource_type: str = "ore",
        required_tech: str | None = None,
        *,
        remaining: int | None = None,
    ) -> None:
        self.id = id
        self.position = position
        self.abundance = abundance if abundance is not None else remaining if remaining is not None else 0
        self.resource_type = resource_type
        self.required_tech = required_tech

    @property
    def remaining(self) -> int:
        """Compatibility alias for older tests/saves; resources are infinite."""
        return self.abundance


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


