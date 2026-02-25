from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from src.agegrid.env.entities import Base, ResourceNode, Unit

from src.agegrid.env.systems import movement, economy, mapgen

Position = Tuple[int, int]


@dataclass
class GameConfig:
    width: int = 12
    height: int = 12
    # Game turn configs
    max_turns: int = 200
    actions_per_turn: int = 3
    # Designed to limit agents from randomly guessing
    max_attempts_per_turn: int = 10 

    base_hp: int = 30
    starting_resources: int = 30
    num_resource_nodes: int = 8
    resource_per_node: int = 60
    worker_gather_amount: int = 5
    seed: int = 42

    # Multiple workers
    worker_spawn_cost: int = 20
    max_workers: int = 10

    # Win Conditions
    # Eventually add more like money win, combat win etc.
    target_bank: int = 200 # Temp resource win


class AgeGridEnv:
    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random(self.config.seed)

        # Turn actions
        self.turn: int = 0
        self.actions_left: int = 0
        self.attempts_left: int = 0

        self.current_player: int = 0

        # Later can be facitons such as vikings, raiders etc.
        self.factions: Tuple[str, str] = ("Red", "Blue")

        self.bases: Dict[str, Base] = {}
        self.resources: List[ResourceNode] = []
        self.units: List[Unit] = []
        self.bank: Dict[str, int] = {}
        self._next_unit_id: int = 1

        self.reset()

    # Game setup

    def reset(self) -> None:
        self.turn = 0
        self.current_player = 0
        self._next_unit_id = 1

        self.bases = {
            "Red": Base("Red", self.config.base_hp, (1, 1)),
            "Blue": Base("Blue", self.config.base_hp, (self.config.width - 2, self.config.height - 2)),
        }

        self.bank = {f: self.config.starting_resources for f in self.factions}

        self.resources = mapgen.place_symmetric_resources(
            self,
            self.config.num_resource_nodes,
            self.config.resource_per_node,
        )

        self.units = []
        self._spawn_worker("Red", (2, 1))
        self._spawn_worker("Blue", (self.config.width - 3, self.config.height - 2))

        self.actions_left = self.config.actions_per_turn
        self.attempts_left = self.config.max_attempts_per_turn

    def _spawn_worker(self, faction: str, pos: Position) -> None:
        self.units.append(Unit(self._next_unit_id, faction, "worker", 5, pos))
        self._next_unit_id += 1

    # Game Helpers

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _mirror(self, pos: Position) -> Position:
        x, y = pos
        return (self.config.width - 1 - x, self.config.height - 1 - y)
    

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

    # Game actions
    def move_unit(self, unit_id: int, direction: str) -> bool:
        return movement.move_unit(self, unit_id, direction)

    def move_towards(self, unit_id: int, target: Position) -> bool:
        return movement.move_towards(self, unit_id, target)
    
    def gather(self, worker_id: int) -> bool:
        return economy.gather(self, worker_id)

    def resource_at(self, pos: Position) -> ResourceNode | None:
        return self._resource_at(pos)


    # Game turn + display

    def start_faction_turn(self) -> None:
        """Reset counters for the currently active faction."""
        self.actions_left = self.config.actions_per_turn
        self.attempts_left = self.config.max_attempts_per_turn

    def _current_faction(self) -> str:
        return self.factions[self.current_player]

    def apply_action(self, action: tuple) -> tuple[bool, str]:
        """
        Apply one action for the current faction.
        Valid action -> consumes 1 action point.
        Invalid action -> consumes 1 attempt (but not an action point).
        Returns (success, reason).
        """
        if self.attempts_left <= 0:
            return False, "no_attempts"
        if self.actions_left <= 0:
            return False, "no_actions"

        # every proposal costs an attempt
        self.attempts_left -= 1

        faction = self._current_faction()

        if not isinstance(action, tuple) or len(action) == 0:
            return False, "bad_action"

        kind = action[0]

        if kind == "gather":
            if len(action) != 2:
                return False, "bad_args"
            unit_id = action[1]
            unit = next((u for u in self.units if u.id == unit_id), None)
            if unit is None or unit.faction != faction:
                return False, "not_your_unit"

            ok = self.gather(unit_id)
            if ok:
                self.actions_left -= 1
                return True, "gather"
            return False, "gather_failed"
        
        if kind == "spawn_worker":
            if len(action) !=1:
                return False, "bad_args"
            
            ok = economy.spawn_worker(self, faction)
            if ok:
                self.actions_left -=1
                return True, "spawn_worker"
            return False, "spawn_failed"

        if kind == "move_towards":
            if len(action) != 3:
                return False, "bad_args"
            unit_id = action[1]
            target = action[2]
            unit = next((u for u in self.units if u.id == unit_id), None)
            if unit is None or unit.faction != faction:
                return False, "not_your_unit"

            ok = self.move_towards(unit_id, target)
            if ok:
                self.actions_left -= 1
                return True, "move"
            return False, "move_blocked"

        return False, "unknown_action"
    
        

    def step_faction(self, decide_action) -> list[str]:
        """
        Run the current faction until it spends all actions OR runs out of attempts.
        decide_action(env) -> action tuple OR None to stop early.
        Returns a log of reasons (useful for UI).
        """
        self.start_faction_turn()
        log: list[str] = []

        while self.actions_left > 0 and self.attempts_left > 0:
            action = decide_action(self)
            if action is None:
                log.append("stop")
                break

            ok, reason = self.apply_action(action)
            log.append(reason if ok else f"invalid:{reason}")

        if self.attempts_left == 0 and self.actions_left > 0:
            log.append("turn_end:no_attempts")

        return log


    def step_end_turn(self) -> None:
        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            self.turn += 1

    # Eventually add more win conditions other than resource
    def winner(self) -> str | None:
        if self.bank["Red"] >= self.config.target_bank:
            return "Red"
        if self.bank["Blue"] >= self.config.target_bank:
            return "Blue"

    def summary(self) -> str:
        lines = [
            f"Turn: {self.turn}/{self.config.max_turns} | Current: {self.factions[self.current_player]}",
            f"Red base @ {self.bases['Red'].position} HP={self.bases['Red'].hp} | Bank={self.bank['Red']}",
            f"Blue base @ {self.bases['Blue'].position} HP={self.bases['Blue'].hp} | Bank={self.bank['Blue']}",
            f"Resources: {len(self.resources)} nodes",
            "Units: " + ", ".join(f"{u.faction} worker#{u.id} @ {u.position}" for u in self.units),
        ]
        return "\n".join(lines)
