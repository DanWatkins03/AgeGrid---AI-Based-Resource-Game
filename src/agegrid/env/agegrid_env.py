from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from src.agegrid.env.actions import Action
from src.agegrid.env.entities import Base, Building, ResourceNode, Unit

from src.agegrid.env.state import BankView, FactionState
from src.agegrid.env.systems import combat, economy, mapgen, movement, production, tech, victory

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
        self.buildings: List[Building] = []
        self.resources: List[ResourceNode] = []
        self.units: List[Unit] = []
        self.faction_states: Dict[str, FactionState] = {}
        self.bank = BankView(self.faction_states)
        self._next_unit_id: int = 1
        self._next_building_id: int = 1

        self.reset()

    # Game setup

    def reset(self) -> None:
        self.turn = 0
        self.current_player = 0
        self._next_unit_id = 1
        self._next_building_id = 1

        self.bases = {
            "Red": Base("Red", self.config.base_hp, (1, 1)),
            "Blue": Base("Blue", self.config.base_hp, (self.config.width - 2, self.config.height - 2)),
        }

        self.faction_states = {
            faction: FactionState(name=faction, resources=self.config.starting_resources)
            for faction in self.factions
        }
        self.bank = BankView(self.faction_states)

        self.resources = mapgen.place_symmetric_resources(
            self,
            self.config.num_resource_nodes,
            self.config.resource_per_node,
        )

        self.buildings = []
        self.units = []
        self._spawn_unit("Red", "worker", 5, (2, 1))
        self._spawn_unit("Blue", "worker", 5, (self.config.width - 3, self.config.height - 2))

        self.actions_left = self.config.actions_per_turn
        self.attempts_left = self.config.max_attempts_per_turn

    def _spawn_unit(
        self,
        faction: str,
        unit_type: str,
        hp: int,
        pos: Position,
        attack_damage: int = 0,
        attack_range: int = 0,
    ) -> None:
        unit = Unit(self._next_unit_id, faction, unit_type, hp, pos, attack_damage, attack_range)
        self.units.append(unit)
        self.faction_state(faction).unit_ids.append(unit.id)
        self._next_unit_id += 1

    def _spawn_worker(self, faction: str, pos: Position) -> None:
        self._spawn_unit(faction, "worker", 5, pos)

    def _spawn_building(
        self,
        faction: str,
        building_type: str,
        hp: int,
        pos: Position,
        attack_damage: int = 0,
        attack_range: int = 0,
    ) -> None:
        building = Building(
            id=self._next_building_id,
            faction=faction,
            building_type=building_type,
            hp=hp,
            position=pos,
            attack_damage=attack_damage,
            attack_range=attack_range,
        )
        self.buildings.append(building)
        self.faction_state(faction).building_ids.append(building.id)
        self._next_building_id += 1

    def _remove_unit(self, unit_id: int) -> None:
        unit = next((u for u in self.units if u.id == unit_id), None)
        if unit is None:
            return
        self.units = [u for u in self.units if u.id != unit_id]
        state = self.faction_state(unit.faction)
        if unit_id in state.unit_ids:
            state.unit_ids.remove(unit_id)

    # Game Helpers

    def faction_state(self, faction: str) -> FactionState:
        return self.faction_states[faction]

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _mirror(self, pos: Position) -> Position:
        x, y = pos
        return (self.config.width - 1 - x, self.config.height - 1 - y)
    

    def _occupied_positions(self) -> set[Position]:
        occ = {b.position for b in self.bases.values()}
        occ.update(b.position for b in self.buildings)
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

    def legal_actions(self, faction: str | None = None) -> list[Action]:
        faction = faction or self._current_faction()
        action_set: set[Action] = set()

        units = [u for u in self.units if u.faction == faction]
        workers = [u for u in units if u.unit_type == "worker"]
        combat_units = [u for u in units if u.attack_damage > 0]
        enemy_units = [u for u in self.units if u.faction != faction]
        enemy_base_positions = [base.position for name, base in self.bases.items() if name != faction]

        for worker in workers:
            if self._resource_at(worker.position) is not None:
                action_set.add(("gather", worker.id))

            for resource in self.resources:
                if resource.remaining > 0 and worker.position != resource.position and movement.can_move_towards(self, worker.id, resource.position):
                    action_set.add(("move_towards", worker.id, resource.position))

            for pos in enemy_base_positions:
                if movement.can_move_towards(self, worker.id, pos):
                    action_set.add(("move_towards", worker.id, pos))

            for enemy in enemy_units:
                if movement.can_move_towards(self, worker.id, enemy.position):
                    action_set.add(("move_towards", worker.id, enemy.position))

        if production.can_train_unit(self, faction, "worker"):
            action_set.add(("spawn_worker",))

        for tech_id in tech.TECH_DEFS:
            if tech.can_research(self, faction, tech_id):
                action_set.add(("research", tech_id))

        for unit_type in production.UNIT_DEFS:
            if unit_type != "worker" and production.can_train_unit(self, faction, unit_type):
                action_set.add(("train", unit_type))

        for worker in workers:
            for building_type in production.BUILDING_DEFS:
                x, y = worker.position
                for pos in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if production.can_build(self, faction, worker.id, building_type, pos):
                        action_set.add(("build", worker.id, building_type, pos))

        for attacker in combat_units:
            for enemy in enemy_units:
                if movement.can_move_towards(self, attacker.id, enemy.position):
                    action_set.add(("move_towards", attacker.id, enemy.position))
            for pos in enemy_base_positions:
                if movement.can_move_towards(self, attacker.id, pos):
                    action_set.add(("move_towards", attacker.id, pos))

            for target in self.units:
                if target.faction == faction:
                    continue
                distance = abs(attacker.position[0] - target.position[0]) + abs(attacker.position[1] - target.position[1])
                if distance <= attacker.attack_range:
                    action_set.add(("attack", attacker.id, target.id))
            for target_faction, base in self.bases.items():
                if target_faction == faction:
                    continue
                distance = abs(attacker.position[0] - base.position[0]) + abs(attacker.position[1] - base.position[1])
                if distance <= attacker.attack_range:
                    action_set.add(("attack_base", attacker.id, target_faction))

        return sorted(action_set, key=str)


    # Game turn + display

    def start_faction_turn(self) -> None:
        """Reset counters for the currently active faction."""
        self.actions_left = self.config.actions_per_turn
        self.attempts_left = self.config.max_attempts_per_turn

    def _current_faction(self) -> str:
        return self.factions[self.current_player]

    def apply_action(self, action: Action | tuple) -> tuple[bool, str]:
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

            ok = production.spawn_worker(self, faction)
            if ok:
                self.actions_left -=1
                return True, "spawn_worker"
            return False, "spawn_failed"

        if kind == "train":
            if len(action) != 2:
                return False, "bad_args"

            ok = production.train_unit(self, faction, action[1])
            if ok:
                self.actions_left -= 1
                return True, "train"
            return False, "train_failed"

        if kind == "build":
            if len(action) != 4:
                return False, "bad_args"
            worker_id = action[1]
            building_type = action[2]
            pos = action[3]
            unit = next((u for u in self.units if u.id == worker_id), None)
            if unit is None or unit.faction != faction:
                return False, "not_your_unit"

            ok = production.build(self, faction, worker_id, building_type, pos)
            if ok:
                self.actions_left -= 1
                return True, "build"
            return False, "build_failed"

        if kind == "research":
            if len(action) != 2:
                return False, "bad_args"

            ok = tech.research(self, faction, action[1])
            if ok:
                self.actions_left -= 1
                return True, "research"
            return False, "research_failed"

        if kind == "attack":
            if len(action) != 3:
                return False, "bad_args"

            ok = combat.attack(self, faction, action[1], action[2])
            if ok:
                self.actions_left -= 1
                return True, "attack"
            return False, "attack_failed"

        if kind == "attack_base":
            if len(action) != 3:
                return False, "bad_args"

            ok = combat.attack_base(self, faction, action[1], action[2])
            if ok:
                self.actions_left -= 1
                return True, "attack_base"
            return False, "attack_base_failed"

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
            if self.winner() is not None:
                log.append("turn_end:winner")
                break

            action = decide_action(self)
            if action is None:
                log.append("stop")
                break

            ok, reason = self.apply_action(action)
            log.append(reason if ok else f"invalid:{reason}")

            if ok and self.winner() is not None:
                log.append("turn_end:winner")
                break

        if self.attempts_left == 0 and self.actions_left > 0:
            log.append("turn_end:no_attempts")

        return log


    def step_end_turn(self) -> None:
        faction = self._current_faction()
        income = sum(
            production.BUILDING_DEFS[b.building_type].resource_income
            for b in self.buildings
            if b.faction == faction and b.hp > 0 and b.building_type in production.BUILDING_DEFS
        )
        self.faction_state(faction).resources += income

        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            self.turn += 1

    # Eventually add more win conditions other than resource
    def winner(self) -> str | None:
        return victory.winner(self)

    def summary(self) -> str:
        unit_summary = ", ".join(f"{u.faction} {u.unit_type}#{u.id} @ {u.position}" for u in self.units)
        tech_summary = " | ".join(
            f"{f}: {', '.join(sorted(self.faction_state(f).techs_unlocked)) or '-'}"
            for f in self.factions
        )
        lines = [
            f"Turn: {self.turn}/{self.config.max_turns} | Current: {self.factions[self.current_player]}",
            f"Red base @ {self.bases['Red'].position} HP={self.bases['Red'].hp} | Bank={self.bank['Red']}",
            f"Blue base @ {self.bases['Blue'].position} HP={self.bases['Blue'].hp} | Bank={self.bank['Blue']}",
            f"Resources: {len(self.resources)} nodes",
            "Units: " + (unit_summary if unit_summary else "-"),
            "Techs: " + tech_summary,
        ]
        return "\n".join(lines)
