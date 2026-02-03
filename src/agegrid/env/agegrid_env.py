from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from src.agegrid.env.entities import Base, ResourceNode, Unit

Position = Tuple[int, int]


@dataclass
class GameConfig:
    width: int = 12
    height: int = 12
    max_turns: int = 200
    base_hp: int = 30
    starting_resources: int = 30
    num_resource_nodes: int = 8
    resource_per_node: int = 60
    worker_gather_amount: int = 5
    seed: int = 42


class AgeGridEnv:
    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random(self.config.seed)

        self.turn: int = 0
        self.current_player: int = 0
        self.factions: Tuple[str, str] = ("Red", "Blue")

        self.bases: Dict[str, Base] = {}
        self.resources: List[ResourceNode] = []
        self.units: List[Unit] = []
        self.bank: Dict[str, int] = {}
        self._next_unit_id: int = 1

        self.reset()

    # ---------- Setup ----------

    def reset(self) -> None:
        self.turn = 0
        self.current_player = 0
        self._next_unit_id = 1

        self.bases = {
            "Red": Base("Red", self.config.base_hp, (1, 1)),
            "Blue": Base("Blue", self.config.base_hp, (self.config.width - 2, self.config.height - 2)),
        }

        self.bank = {f: self.config.starting_resources for f in self.factions}
        self.resources = self._place_symmetric_resources(
            self.config.num_resource_nodes,
            self.config.resource_per_node,
        )

        self.units = []
        self._spawn_worker("Red", (2, 1))
        self._spawn_worker("Blue", (self.config.width - 3, self.config.height - 2))

    def _spawn_worker(self, faction: str, pos: Position) -> None:
        self.units.append(Unit(self._next_unit_id, faction, "worker", 5, pos))
        self._next_unit_id += 1

    # ---------- Helpers ----------

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _mirror(self, pos: Position) -> Position:
        x, y = pos
        return (self.config.width - 1 - x, self.config.height - 1 - y)
    
    def _place_symmetric_resources(self, n: int, remaining: int) -> List[ResourceNode]:
        # Ensure even number for symmetry
        if n % 2 == 1:
            n += 1

        forbidden = {self.bases["Red"].position, self.bases["Blue"].position}
        used: set[Position] = set(forbidden)

        resources: List[ResourceNode] = []
        rid = 1
        attempts = 0

        while len(resources) < n and attempts < 10_000:
            attempts += 1

            # Sample only from left half to avoid duplicates
            x = self.rng.randint(0, self.config.width // 2 - 1)
            y = self.rng.randint(0, self.config.height - 1)

            p1 = (x, y)
            p2 = self._mirror(p1)

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


    def _occupied_positions(self) -> set[Position]:
        occ = {b.position for b in self.bases.values()}
        occ.update(u.position for u in self.units)
        return occ

    def _resource_at(self, pos: Position) -> ResourceNode | None:
        for r in self.resources:
            if r.position == pos and r.remaining > 0:
                return r
        return None

    def _delta(self, direction: str) -> Position:
        return {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }[direction]

    # ---------- Actions ----------

    def move_unit(self, unit_id: int, direction: str) -> bool:
        unit = next((u for u in self.units if u.id == unit_id), None)
        if not unit:
            return False

        dx, dy = self._delta(direction)
        new_pos = (unit.position[0] + dx, unit.position[1] + dy)

        if not self._in_bounds(new_pos):
            return False
        if new_pos in self._occupied_positions():
            return False

        unit.position = new_pos
        return True

    def move_towards(self, unit_id: int, target: Position) -> bool:
        unit = next((u for u in self.units if u.id == unit_id), None)
        if not unit:
            return False

        x, y = unit.position
        tx, ty = target

        if tx > x and self.move_unit(unit_id, "right"):
            return True
        if tx < x and self.move_unit(unit_id, "left"):
            return True
        if ty > y and self.move_unit(unit_id, "down"):
            return True
        if ty < y and self.move_unit(unit_id, "up"):
            return True
        return False

    def gather(self, worker_id: int) -> bool:
        unit = next((u for u in self.units if u.id == worker_id), None)
        if not unit or unit.unit_type != "worker":
            return False

        node = self._resource_at(unit.position)
        if not node:
            return False

        amount = min(self.config.worker_gather_amount, node.remaining)
        node.remaining -= amount
        self.bank[unit.faction] += amount
        return True

    # ---------- Turn + Display ----------

    def step_end_turn(self) -> None:
        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            self.turn += 1

    def summary(self) -> str:
        lines = [
            f"Turn: {self.turn}/{self.config.max_turns} | Current: {self.factions[self.current_player]}",
            f"Red base @ {self.bases['Red'].position} HP={self.bases['Red'].hp} | Bank={self.bank['Red']}",
            f"Blue base @ {self.bases['Blue'].position} HP={self.bases['Blue'].hp} | Bank={self.bank['Blue']}",
            f"Resources: {len(self.resources)} nodes",
            "Units: " + ", ".join(f"{u.faction} worker#{u.id} @ {u.position}" for u in self.units),
        ]
        return "\n".join(lines)
