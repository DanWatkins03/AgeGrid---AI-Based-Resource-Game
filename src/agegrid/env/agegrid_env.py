from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from src.agegrid.env.actions import Action
from src.agegrid.env.entities import Base, Building, ResourceNode, Unit
from src.agegrid.env import hexgrid

from src.agegrid.env.state import BankView, FactionState, RelationState
from src.agegrid.env.systems import combat, economy, mapgen, movement, production, tech, victory

Position = Tuple[int, int]


@dataclass
class GameConfig:
    width: int = 14
    height: int = 14
    # Game turn configs
    max_turns: int = 200
    actions_per_turn: int = 4
    start_year: int = -3000
    years_per_turn: int = 25
    worker_peace_until_turn: int = 20
    base_peace_until_turn: int = 20
    # Designed to limit agents from randomly guessing
    max_attempts_per_turn: int = 10 

    base_hp: int = 30
    base_attack_damage: int = 2
    base_attack_range: int = 2
    masonry_base_attack_bonus: int = 1
    starting_resources: int = 30
    num_resource_nodes: int = 8
    resource_per_node: int = 60
    stone_resource_nodes: int = 2
    stone_resource_amount: int = 50
    horse_resource_nodes: int = 2
    horse_resource_amount: int = 50
    worker_gather_amount: int = 5
    unit_heal_per_turn: int = 1
    unit_heal_near_base_bonus: int = 1
    unit_heal_base_radius: int = 3
    seed: int = 42

    # Multiple workers
    worker_spawn_cost: int = 20
    max_workers: int = 10

    # Win Conditions
    # Eventually add more like money win, combat win etc.
    target_bank: int | None = None
    collapse_enabled: bool = True
    min_war_duration: int = 6
    peace_offer_min_turns: int = 4
    truce_turns: int = 8
    peace_indemnity_base: int = 24


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
        self.relations: Dict[frozenset[str], RelationState] = {}
        self._next_unit_id: int = 1
        self._next_building_id: int = 1
        self.current_events: list[str] = []
        self.recent_events: list[str] = []
        self.free_research_used: bool = False
        self.attacked_unit_ids: set[int] = set()

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
        self.relations = {
            frozenset(self.factions): RelationState(
                state="peace",
                since_turn=0,
                truce_until_turn=0,
            )
        }

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
        self.current_events = []
        self.recent_events = []
        self.free_research_used = False

    def _spawn_unit(
        self,
        faction: str,
        unit_type: str,
        hp: int,
        pos: Position,
        attack_damage: int = 0,
        attack_range: int = 0,
        move_steps: int = 1,
    ) -> None:
        unit = Unit(self._next_unit_id, faction, unit_type, hp, pos, attack_damage, attack_range, move_steps)
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

    def _record_event(self, message: str) -> None:
        self.current_events.append(message)
        self.recent_events.append(message)
        self.recent_events = self.recent_events[-12:]

    def faction_state(self, faction: str) -> FactionState:
        return self.faction_states[faction]

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _mirror(self, pos: Position) -> Position:
        x, y = pos
        return (self.config.width - 1 - x, self.config.height - 1 - y)

    def _relation_key(self, faction_a: str, faction_b: str) -> frozenset[str]:
        return frozenset((faction_a, faction_b))

    def relation_state(self, faction_a: str, faction_b: str) -> RelationState:
        key = self._relation_key(faction_a, faction_b)
        if key not in self.relations:
            self.relations[key] = RelationState()
        relation = self.relations[key]
        if relation.state == "truce" and self.turn >= relation.truce_until_turn:
            relation.state = "peace"
            relation.since_turn = self.turn
            relation.pending_peace_by = None
            relation.pending_indemnity = 0
        return relation

    def at_war(self, faction_a: str, faction_b: str) -> bool:
        return self.relation_state(faction_a, faction_b).state == "war"

    def can_declare_war(self, faction: str, target_faction: str) -> bool:
        relation = self.relation_state(faction, target_faction)
        return relation.state != "war" and self.turn >= relation.truce_until_turn

    def declare_war(self, faction: str, target_faction: str) -> bool:
        if not self.can_declare_war(faction, target_faction):
            return False
        relation = self.relation_state(faction, target_faction)
        relation.state = "war"
        relation.since_turn = self.turn
        relation.pending_peace_by = None
        relation.pending_indemnity = 0
        return True

    def can_offer_peace(self, faction: str, target_faction: str) -> bool:
        relation = self.relation_state(faction, target_faction)
        return (
            relation.state == "war"
            and self.turn - relation.since_turn >= self.config.peace_offer_min_turns
            and relation.pending_peace_by is None
        )

    def offer_peace(self, faction: str, target_faction: str, indemnity: int) -> bool:
        if not self.can_offer_peace(faction, target_faction):
            return False
        relation = self.relation_state(faction, target_faction)
        relation.pending_peace_by = faction
        relation.pending_indemnity = max(0, indemnity)
        return True

    def can_accept_peace(self, faction: str, target_faction: str) -> bool:
        relation = self.relation_state(faction, target_faction)
        return relation.state == "war" and relation.pending_peace_by == target_faction

    def accept_peace(self, faction: str, target_faction: str) -> int | None:
        if not self.can_accept_peace(faction, target_faction):
            return None
        relation = self.relation_state(faction, target_faction)
        payer = relation.pending_peace_by
        if payer is None:
            return None
        receiver = target_faction if payer == faction else faction
        indemnity = min(relation.pending_indemnity, self.bank[payer])
        self.bank[payer] -= indemnity
        self.bank[receiver] += indemnity
        relation.state = "truce"
        relation.since_turn = self.turn
        relation.truce_until_turn = self.turn + self.config.truce_turns
        relation.pending_peace_by = None
        relation.pending_indemnity = 0
        return indemnity
    

    def _occupied_positions(self) -> set[Position]:
        occ = {b.position for b in self.bases.values()}
        occ.update(b.position for b in self.buildings)
        occ.update(u.position for u in self.units)
        return occ

    def _resource_at(self, pos: Position, faction: str | None = None) -> ResourceNode | None:
        for r in self.resources:
            if faction is not None and r.required_tech is not None and r.required_tech not in self.faction_state(faction).techs_unlocked:
                continue
            if r.position == pos and r.remaining > 0:
                return r
        return None

    def _delta(self, direction: str) -> Position:
        return hexgrid.direction_map(0)[direction]

    # Game actions
    def move_unit(self, unit_id: int, direction: str) -> bool:
        return movement.move_unit(self, unit_id, direction)

    def move_towards(self, unit_id: int, target: Position) -> bool:
        return movement.move_towards(self, unit_id, target)
    
    def gather(self, worker_id: int) -> bool:
        return economy.gather(self, worker_id)

    def resource_at(self, pos: Position) -> ResourceNode | None:
        return self._resource_at(pos)

    def resource_at_for_faction(self, pos: Position, faction: str) -> ResourceNode | None:
        return self._resource_at(pos, faction)

    def visible_resources(self, faction: str) -> list[ResourceNode]:
        return [
            resource
            for resource in self.resources
            if resource.remaining > 0
            and (resource.required_tech is None or resource.required_tech in self.faction_state(faction).techs_unlocked)
        ]

    def legal_actions(self, faction: str | None = None) -> list[Action]:
        faction = faction or self._current_faction()
        action_set: set[Action] = set()
        enemies = [name for name in self.factions if name != faction]

        units = [u for u in self.units if u.faction == faction]
        workers = [u for u in units if u.unit_type == "worker"]
        combat_units = [u for u in units if u.attack_damage > 0]
        enemy_units = [u for u in self.units if u.faction != faction]
        enemy_base_positions = [base.position for name, base in self.bases.items() if name != faction]
        enemy_base_approach_positions: list[Position] = []
        for base_pos in enemy_base_positions:
            enemy_base_approach_positions.extend(pos for pos in hexgrid.neighbors(base_pos) if self._in_bounds(pos))
        friendly_guard_positions = [self.bases[faction].position]
        friendly_guard_positions.extend(worker.position for worker in workers)
        friendly_guard_positions.extend(unit.position for unit in combat_units)
        friendly_guard_positions.extend(
            building.position for building in self.buildings if building.faction == faction and building.hp > 0
        )

        for worker in workers:
            if self._resource_at(worker.position, faction) is not None:
                action_set.add(("gather", worker.id))

            for resource in self.visible_resources(faction):
                if worker.position != resource.position and movement.can_move_towards(self, worker.id, resource.position):
                    action_set.add(("move_towards", worker.id, resource.position))

            for pos in enemy_base_positions:
                if movement.can_move_towards(self, worker.id, pos):
                    action_set.add(("move_towards", worker.id, pos))

            for enemy in enemy_units:
                if movement.can_move_towards(self, worker.id, enemy.position):
                    action_set.add(("move_towards", worker.id, enemy.position))
            for pos in friendly_guard_positions:
                if pos != worker.position and movement.can_move_towards(self, worker.id, pos):
                    action_set.add(("move_towards", worker.id, pos))

        for enemy_faction in enemies:
            if self.can_declare_war(faction, enemy_faction):
                action_set.add(("declare_war", enemy_faction))
            if self.can_offer_peace(faction, enemy_faction):
                indemnity = min(self.config.peace_indemnity_base, self.bank[faction])
                action_set.add(("offer_peace", enemy_faction, indemnity))
            if self.can_accept_peace(faction, enemy_faction):
                action_set.add(("accept_peace", enemy_faction))

        if production.can_train_unit(self, faction, "worker"):
            action_set.add(("spawn_worker",))

        for tech_id in tech.TECH_DEFS:
            if not self.free_research_used and tech.can_research(self, faction, tech_id):
                action_set.add(("research", tech_id))

        for unit_type in production.UNIT_DEFS:
            if unit_type != "worker" and production.can_train_unit(self, faction, unit_type):
                action_set.add(("train", unit_type))

        for worker in workers:
            for building_type in production.BUILDING_DEFS:
                for pos in hexgrid.neighbors(worker.position):
                    if production.can_build(self, faction, worker.id, building_type, pos):
                        action_set.add(("build", worker.id, building_type, pos))

        for attacker in combat_units:
            for enemy in enemy_units:
                if movement.can_move_towards(self, attacker.id, enemy.position):
                    action_set.add(("move_towards", attacker.id, enemy.position))
            for pos in enemy_base_positions:
                if movement.can_move_towards(self, attacker.id, pos):
                    action_set.add(("move_towards", attacker.id, pos))
            for pos in enemy_base_approach_positions:
                if movement.can_move_towards(self, attacker.id, pos):
                    action_set.add(("move_towards", attacker.id, pos))
            for pos in friendly_guard_positions:
                if pos != attacker.position and movement.can_move_towards(self, attacker.id, pos):
                    action_set.add(("move_towards", attacker.id, pos))

            for target in self.units:
                if target.faction == faction:
                    continue
                distance = hexgrid.distance(attacker.position, target.position)
                if self.at_war(faction, target.faction) and distance <= attacker.attack_range and (
                    target.unit_type != "worker" or self.turn >= self.config.worker_peace_until_turn
                ):
                    action_set.add(("attack", attacker.id, target.id))
            for target_faction, base in self.bases.items():
                if target_faction == faction:
                    continue
                distance = hexgrid.distance(attacker.position, base.position)
                if self.at_war(faction, target_faction) and distance <= attacker.attack_range and self.turn >= self.config.base_peace_until_turn:
                    action_set.add(("attack_base", attacker.id, target_faction))

        return sorted(action_set, key=str)


    # Game turn + display

    def start_faction_turn(self) -> None:
        """Reset counters for the currently active faction."""
        self.actions_left = self.config.actions_per_turn
        self.attempts_left = self.config.max_attempts_per_turn
        self.current_events = []
        self.free_research_used = False
        self.attacked_unit_ids = set()

    def _current_faction(self) -> str:
        return self.factions[self.current_player]

    def unit_max_hp(self, unit: Unit) -> int:
        return production.unit_stats(self, unit.faction, unit.unit_type).hp

    def heal_amount_for(self, unit: Unit) -> int:
        amount = self.config.unit_heal_per_turn
        if hexgrid.distance(unit.position, self.bases[unit.faction].position) <= self.config.unit_heal_base_radius:
            amount += self.config.unit_heal_near_base_bonus
        return amount

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
                self._record_event(f"{faction} worker#{unit_id} gathered resources")
                return True, "gather"
            return False, "gather_failed"
        
        if kind == "spawn_worker":
            if len(action) !=1:
                return False, "bad_args"

            ok = production.spawn_worker(self, faction)
            if ok:
                self.actions_left -=1
                new_unit = max((u for u in self.units if u.faction == faction), key=lambda u: u.id)
                self._record_event(f"{faction} spawned worker#{new_unit.id}")
                return True, "spawn_worker"
            return False, "spawn_failed"

        if kind == "train":
            if len(action) != 2:
                return False, "bad_args"

            ok = production.train_unit(self, faction, action[1])
            if ok:
                self.actions_left -= 1
                new_unit = max((u for u in self.units if u.faction == faction), key=lambda u: u.id)
                self._record_event(f"{faction} trained {action[1]}#{new_unit.id}")
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
                self._record_event(f"{faction} built {building_type} at {pos}")
                return True, "build"
            return False, "build_failed"

        if kind == "research":
            if len(action) != 2:
                return False, "bad_args"
            if self.free_research_used:
                return False, "research_used"

            ok = tech.research(self, faction, action[1])
            if ok:
                self.free_research_used = True
                self._record_event(f"{faction} researched {action[1]}")
                return True, "research"
            return False, "research_failed"

        if kind == "declare_war":
            if len(action) != 2:
                return False, "bad_args"
            target_faction = action[1]
            ok = self.declare_war(faction, target_faction)
            if ok:
                self.actions_left -= 1
                self._record_event(f"{faction} declared war on {target_faction}")
                return True, "declare_war"
            return False, "declare_war_failed"

        if kind == "offer_peace":
            if len(action) != 3:
                return False, "bad_args"
            target_faction = action[1]
            indemnity = int(action[2])
            ok = self.offer_peace(faction, target_faction, indemnity)
            if ok:
                self.actions_left -= 1
                self._record_event(f"{faction} offered peace to {target_faction} for {indemnity}")
                return True, "offer_peace"
            return False, "offer_peace_failed"

        if kind == "accept_peace":
            if len(action) != 2:
                return False, "bad_args"
            target_faction = action[1]
            indemnity = self.accept_peace(faction, target_faction)
            if indemnity is not None:
                self.actions_left -= 1
                self._record_event(f"{faction} accepted peace with {target_faction} (indemnity {indemnity})")
                return True, "accept_peace"
            return False, "accept_peace_failed"

        if kind == "attack":
            if len(action) != 3:
                return False, "bad_args"
            attacker_id = action[1]
            target_id = action[2]
            target = next((u for u in self.units if u.id == target_id), None)
            ok = combat.attack(self, faction, action[1], action[2])
            if ok:
                self.attacked_unit_ids.add(action[1])
                self.actions_left -= 1
                if target is not None and target.hp <= 0:
                    self._record_event(f"{faction} unit#{attacker_id} defeated {target.faction} {target.unit_type}#{target_id}")
                elif target is not None:
                    self._record_event(f"{faction} unit#{attacker_id} hit {target.faction} {target.unit_type}#{target_id} ({target.hp} hp left)")
                return True, "attack"
            return False, "attack_failed"

        if kind == "attack_base":
            if len(action) != 3:
                return False, "bad_args"
            attacker_id = action[1]
            target_faction = action[2]
            ok = combat.attack_base(self, faction, attacker_id, target_faction)
            if ok:
                self.attacked_unit_ids.add(attacker_id)
                self.actions_left -= 1
                base_hp = self.bases[target_faction].hp
                if base_hp == 0:
                    self._record_event(f"{faction} unit#{attacker_id} destroyed the {target_faction} base")
                else:
                    self._record_event(f"{faction} unit#{attacker_id} hit the {target_faction} base ({base_hp} hp left)")
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

            start_pos = unit.position
            ok = self.move_towards(unit_id, target)
            if ok:
                self.actions_left -= 1
                self._record_event(
                    f"{faction} unit#{unit_id} moved from {start_pos} to {unit.position} toward {target}"
                )
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
        for event in combat.resolve_defensive_fire(self, faction):
            self._record_event(event)
        if self.winner() is not None:
            self.current_player = 1 - self.current_player
            if self.current_player == 0:
                self.turn += 1
            return

        completed_tech = tech.progress_research(self, faction)
        if completed_tech is not None:
            self._record_event(f"{faction} completed research: {completed_tech}")

        income = sum(
            production.BUILDING_DEFS[b.building_type].resource_income
            for b in self.buildings
            if b.faction == faction and b.hp > 0 and b.building_type in production.BUILDING_DEFS
        )
        self.faction_state(faction).resources += income
        if income > 0:
            self._record_event(f"{faction} gained {income} passive income")

        healed_units: list[str] = []
        enemy = next(name for name in self.factions if name != faction)
        for unit in self.units:
            if unit.faction != faction or unit.hp <= 0 or unit.id in self.attacked_unit_ids:
                continue
            if any(
                other.faction == enemy and other.attack_damage > 0 and hexgrid.distance(other.position, unit.position) <= 1
                for other in self.units
            ):
                continue
            max_hp = self.unit_max_hp(unit)
            if unit.hp >= max_hp:
                continue
            healed = min(self.heal_amount_for(unit), max_hp - unit.hp)
            if healed <= 0:
                continue
            unit.hp += healed
            healed_units.append(f"{faction} {unit.unit_type}#{unit.id} recovered {healed} hp")
        for event in healed_units:
            self._record_event(event)

        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            self.turn += 1

    # Eventually add more win conditions other than resource
    def winner(self) -> str | None:
        return victory.winner(self)

    def current_year(self) -> int:
        return self.config.start_year + self.turn * self.config.years_per_turn

    def current_era(self) -> str:
        unlocked = set().union(*(state.techs_unlocked for state in self.faction_states.values()))
        if "engineering" in unlocked:
            return "Engineering Age"
        if unlocked.intersection({"iron_working", "fortification", "stirrups", "fletching"}):
            return "Iron Age"
        if unlocked.intersection({"bronze_working", "masonry", "horsemanship"}):
            return "Bronze Age"
        if "mining" in unlocked:
            return "Stone Age"
        return "Founding Age"

    def formatted_year(self) -> str:
        year = self.current_year()
        if year < 0:
            return f"{abs(year)} BCE"
        return f"{year} CE"

    def summary(self) -> str:
        unit_summary = ", ".join(f"{u.faction} {u.unit_type}#{u.id} @ {u.position}" for u in self.units)
        tech_summary = " | ".join(
            f"{f}: {', '.join(sorted(self.faction_state(f).techs_unlocked)) or '-'}"
            for f in self.factions
        )
        lines = [
            f"Turn: {self.turn}/{self.config.max_turns} | Year: {self.formatted_year()} | Era: {self.current_era()} | Current: {self.factions[self.current_player]}",
            f"Red base @ {self.bases['Red'].position} HP={self.bases['Red'].hp} | Bank={self.bank['Red']}",
            f"Blue base @ {self.bases['Blue'].position} HP={self.bases['Blue'].hp} | Bank={self.bank['Blue']}",
            f"Relations: {self.relation_state('Red', 'Blue').state.title()}",
            f"Resources: {len(self.resources)} nodes",
            "Units: " + (unit_summary if unit_summary else "-"),
            "Techs: " + tech_summary,
        ]
        return "\n".join(lines)
