from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from src.agegrid.env import hexgrid
from src.agegrid.env.actions import Action
from src.agegrid.env.entities import Base, Building, ResourceNode, Unit
from src.agegrid.env.state import BankView, FactionState, RelationState
from src.agegrid.env.systems import combat, economy, mapgen, movement, production, tech, victory

Position = tuple[int, int]
ActionHandler = Callable[[str, Action | tuple], tuple[bool, str]]


@dataclass
class GameConfig:
    width: int = 14
    height: int = 14
    max_turns: int = 200
    actions_per_turn: int = 4
    start_year: int = -3000
    years_per_turn: int = 25
    worker_peace_until_turn: int = 20
    base_peace_until_turn: int = 20
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
    worker_spawn_cost: int = 20
    max_workers: int = 10
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

        self.turn: int = 0
        self.actions_left: int = 0
        self.attempts_left: int = 0
        self.current_player: int = 0
        self.factions: tuple[str, str] = ("Red", "Blue")

        self.bases: dict[str, Base] = {}
        self.buildings: list[Building] = []
        self.resources: list[ResourceNode] = []
        self.units: list[Unit] = []
        self.faction_states: dict[str, FactionState] = {}
        self.bank = BankView(self.faction_states)
        self.relations: dict[frozenset[str], RelationState] = {}
        self._unit_index: dict[int, Unit] = {}
        self._building_index: dict[int, Building] = {}
        self._next_unit_id: int = 1
        self._next_building_id: int = 1
        self.current_events: list[str] = []
        self.recent_events: list[str] = []
        self.free_research_used: bool = False
        self.attacked_unit_ids: set[int] = set()

        self._action_handlers: dict[str, ActionHandler] = {
            "gather": self._handle_gather,
            "spawn_worker": self._handle_spawn_worker,
            "train": self._handle_train,
            "build": self._handle_build,
            "research": self._handle_research,
            "declare_war": self._handle_declare_war,
            "offer_peace": self._handle_offer_peace,
            "accept_peace": self._handle_accept_peace,
            "attack": self._handle_attack,
            "attack_base": self._handle_attack_base,
            "move_towards": self._handle_move_towards,
        }

        self.reset()

    def reset(self) -> None:
        self.turn = 0
        self.current_player = 0
        self._next_unit_id = 1
        self._next_building_id = 1
        self._unit_index = {}
        self._building_index = {}

        self.bases = {
            "Red": Base("Red", self.config.base_hp, (1, 1)),
            "Blue": Base("Blue", self.config.base_hp, (self.config.width - 2, self.config.height - 2)),
        }
        self.buildings = []
        self.units = []
        self.faction_states = {
            faction: FactionState(name=faction, resources=self.config.starting_resources)
            for faction in self.factions
        }
        self.bank = BankView(self.faction_states)
        self.relations = {
            frozenset(self.factions): RelationState(state="peace", since_turn=0, truce_until_turn=0)
        }
        self.resources = mapgen.place_symmetric_resources(
            self,
            self.config.num_resource_nodes,
            self.config.resource_per_node,
        )

        self._spawn_unit("Red", "worker", 5, (2, 1))
        self._spawn_unit("Blue", "worker", 5, (self.config.width - 3, self.config.height - 2))
        self._reset_turn_state()
        self.recent_events = []

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
        self._unit_index[unit.id] = unit
        self.faction_state(faction).unit_ids.append(unit.id)
        self._next_unit_id += 1

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
        self._building_index[building.id] = building
        self.faction_state(faction).building_ids.append(building.id)
        self._next_building_id += 1

    def _remove_unit(self, unit_id: int) -> None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return
        self.units = [u for u in self.units if u.id != unit_id]
        self._unit_index.pop(unit_id, None)
        state = self.faction_state(unit.faction)
        if unit_id in state.unit_ids:
            state.unit_ids.remove(unit_id)

    def _record_event(self, message: str) -> None:
        self.current_events.append(message)
        self.recent_events.append(message)
        self.recent_events = self.recent_events[-12:]

    def _record_events(self, messages: list[str]) -> None:
        for message in messages:
            self._record_event(message)

    def _reset_turn_state(self) -> None:
        self.actions_left = self.config.actions_per_turn
        self.attempts_left = self.config.max_attempts_per_turn
        self.current_events = []
        self.free_research_used = False
        self.attacked_unit_ids = set()

    def faction_state(self, faction: str) -> FactionState:
        return self.faction_states[faction]

    def get_unit(self, unit_id: int) -> Unit | None:
        return self._unit_index.get(unit_id)

    def get_building(self, building_id: int) -> Building | None:
        return self._building_index.get(building_id)

    def get_units_for_faction(self, faction: str) -> list[Unit]:
        return [
            unit
            for unit_id in self.faction_state(faction).unit_ids
            if (unit := self.get_unit(unit_id)) is not None
        ]

    def get_buildings_for_faction(self, faction: str) -> list[Building]:
        return [
            building
            for building_id in self.faction_state(faction).building_ids
            if (building := self.get_building(building_id)) is not None
        ]

    def get_enemy_units(self, faction: str) -> list[Unit]:
        return [unit for unit in self.units if unit.faction != faction]

    def _get_owned_unit(self, unit_id: int, faction: str) -> Unit | None:
        unit = self.get_unit(unit_id)
        if unit is None or unit.faction != faction:
            return None
        return unit

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _mirror(self, pos: Position) -> Position:
        x, y = pos
        return (self.config.width - 1 - x, self.config.height - 1 - y)

    def _relation_key(self, faction_a: str, faction_b: str) -> frozenset[str]:
        return frozenset((faction_a, faction_b))

    def _refresh_relation_state(self, relation: RelationState) -> RelationState:
        if relation.state == "truce" and self.turn >= relation.truce_until_turn:
            relation.state = "peace"
            relation.since_turn = self.turn
            relation.pending_peace_by = None
            relation.pending_indemnity = 0
        return relation

    def _clear_pending_peace(self, relation: RelationState) -> None:
        relation.pending_peace_by = None
        relation.pending_indemnity = 0

    def relation_state(self, faction_a: str, faction_b: str) -> RelationState:
        key = self._relation_key(faction_a, faction_b)
        if key not in self.relations:
            self.relations[key] = RelationState()
        return self._refresh_relation_state(self.relations[key])

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
        self._clear_pending_peace(relation)
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
        self._clear_pending_peace(relation)
        return indemnity

    def _occupied_positions(self) -> set[Position]:
        occupied = {base.position for base in self.bases.values()}
        occupied.update(building.position for building in self.buildings)
        occupied.update(unit.position for unit in self.units)
        return occupied

    def _resource_at(self, pos: Position, faction: str | None = None) -> ResourceNode | None:
        for resource in self.resources:
            if (
                faction is not None
                and resource.required_tech is not None
                and resource.required_tech not in self.faction_state(faction).techs_unlocked
            ):
                continue
            if resource.position == pos and resource.remaining > 0:
                return resource
        return None

    def _current_faction(self) -> str:
        return self.factions[self.current_player]

    def _enemy_factions(self, faction: str) -> list[str]:
        return [name for name in self.factions if name != faction]

    def _enemy_base_positions(self, faction: str) -> list[Position]:
        return [base.position for name, base in self.bases.items() if name != faction]

    def _enemy_base_approach_positions(self, faction: str) -> list[Position]:
        positions: list[Position] = []
        for base_pos in self._enemy_base_positions(faction):
            positions.extend(pos for pos in hexgrid.neighbors(base_pos) if self._in_bounds(pos))
        return positions

    def _friendly_guard_positions(
        self,
        faction: str,
        workers: list[Unit],
        combat_units: list[Unit],
    ) -> list[Position]:
        positions = [self.bases[faction].position]
        positions.extend(worker.position for worker in workers)
        positions.extend(unit.position for unit in combat_units)
        positions.extend(
            building.position for building in self.get_buildings_for_faction(faction) if building.hp > 0
        )
        return positions

    def _add_move_actions(self, action_set: set[Action], unit: Unit, targets: list[Position]) -> None:
        for pos in targets:
            if movement.can_move_towards(self, unit.id, pos):
                action_set.add(("move_towards", unit.id, pos))

    def _units_by_role(self, faction: str) -> tuple[list[Unit], list[Unit], list[Unit]]:
        faction_units = self.get_units_for_faction(faction)
        workers = [unit for unit in faction_units if unit.unit_type == "worker"]
        combat_units = [unit for unit in faction_units if unit.attack_damage > 0]
        return faction_units, workers, combat_units

    def _worker_move_targets(
        self,
        worker: Unit,
        visible_resources: list[ResourceNode],
        enemy_units: list[Unit],
        enemy_base_positions: list[Position],
        friendly_guard_positions: list[Position],
    ) -> list[Position]:
        targets = [resource.position for resource in visible_resources if resource.position != worker.position]
        targets.extend(enemy_base_positions)
        targets.extend(enemy.position for enemy in enemy_units)
        targets.extend(pos for pos in friendly_guard_positions if pos != worker.position)
        return targets

    def _combat_move_targets(
        self,
        attacker: Unit,
        enemy_units: list[Unit],
        enemy_base_positions: list[Position],
        enemy_base_approach_positions: list[Position],
        friendly_guard_positions: list[Position],
    ) -> list[Position]:
        targets = [enemy.position for enemy in enemy_units]
        targets.extend(enemy_base_positions)
        targets.extend(enemy_base_approach_positions)
        targets.extend(pos for pos in friendly_guard_positions if pos != attacker.position)
        return targets

    def _can_attack_unit(self, faction: str, attacker: Unit, target: Unit) -> bool:
        distance = hexgrid.distance(attacker.position, target.position)
        return (
            self.at_war(faction, target.faction)
            and distance <= attacker.attack_range
            and (target.unit_type != "worker" or self.turn >= self.config.worker_peace_until_turn)
        )

    def _can_attack_base(self, faction: str, attacker: Unit, target_faction: str, base: Base) -> bool:
        distance = hexgrid.distance(attacker.position, base.position)
        return (
            self.at_war(faction, target_faction)
            and distance <= attacker.attack_range
            and self.turn >= self.config.base_peace_until_turn
        )

    def _last_faction_unit(self, faction: str) -> Unit:
        return self.get_units_for_faction(faction)[-1]

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
            and (
                resource.required_tech is None
                or resource.required_tech in self.faction_state(faction).techs_unlocked
            )
        ]

    def _legal_worker_actions(
        self,
        faction: str,
        workers: list[Unit],
        enemy_units: list[Unit],
        enemy_base_positions: list[Position],
        friendly_guard_positions: list[Position],
        action_set: set[Action],
    ) -> None:
        visible_resources = self.visible_resources(faction)

        for worker in workers:
            if self._resource_at(worker.position, faction) is not None:
                action_set.add(("gather", worker.id))

            self._add_move_actions(
                action_set,
                worker,
                self._worker_move_targets(
                    worker,
                    visible_resources,
                    enemy_units,
                    enemy_base_positions,
                    friendly_guard_positions,
                ),
            )

    def _legal_diplomacy_actions(self, faction: str, action_set: set[Action]) -> None:
        for enemy_faction in self._enemy_factions(faction):
            if self.can_declare_war(faction, enemy_faction):
                action_set.add(("declare_war", enemy_faction))
            if self.can_offer_peace(faction, enemy_faction):
                indemnity = min(self.config.peace_indemnity_base, self.bank[faction])
                action_set.add(("offer_peace", enemy_faction, indemnity))
            if self.can_accept_peace(faction, enemy_faction):
                action_set.add(("accept_peace", enemy_faction))

    def _legal_progression_actions(self, faction: str, workers: list[Unit], action_set: set[Action]) -> None:
        if production.can_train_unit(self, faction, "worker"):
            action_set.add(("spawn_worker",))

        if not self.free_research_used:
            for tech_id in tech.TECH_DEFS:
                if tech.can_research(self, faction, tech_id):
                    action_set.add(("research", tech_id))

        for unit_type in production.UNIT_DEFS:
            if unit_type != "worker" and production.can_train_unit(self, faction, unit_type):
                action_set.add(("train", unit_type))

        for worker in workers:
            for building_type in production.BUILDING_DEFS:
                for pos in hexgrid.neighbors(worker.position):
                    if production.can_build(self, faction, worker.id, building_type, pos):
                        action_set.add(("build", worker.id, building_type, pos))

    def _legal_combat_actions(
        self,
        faction: str,
        combat_units: list[Unit],
        enemy_units: list[Unit],
        enemy_base_positions: list[Position],
        enemy_base_approach_positions: list[Position],
        friendly_guard_positions: list[Position],
        action_set: set[Action],
    ) -> None:
        for attacker in combat_units:
            self._add_move_actions(
                action_set,
                attacker,
                self._combat_move_targets(
                    attacker,
                    enemy_units,
                    enemy_base_positions,
                    enemy_base_approach_positions,
                    friendly_guard_positions,
                ),
            )

            for target in enemy_units:
                if self._can_attack_unit(faction, attacker, target):
                    action_set.add(("attack", attacker.id, target.id))

            for target_faction, base in self.bases.items():
                if target_faction == faction:
                    continue
                if self._can_attack_base(faction, attacker, target_faction, base):
                    action_set.add(("attack_base", attacker.id, target_faction))

    def legal_actions(self, faction: str | None = None) -> list[Action]:
        faction = faction or self._current_faction()
        action_set: set[Action] = set()

        _faction_units, workers, combat_units = self._units_by_role(faction)
        enemy_units = self.get_enemy_units(faction)
        enemy_base_positions = self._enemy_base_positions(faction)
        enemy_base_approach_positions = self._enemy_base_approach_positions(faction)
        friendly_guard_positions = self._friendly_guard_positions(faction, workers, combat_units)

        self._legal_worker_actions(
            faction,
            workers,
            enemy_units,
            enemy_base_positions,
            friendly_guard_positions,
            action_set,
        )
        self._legal_diplomacy_actions(faction, action_set)
        self._legal_progression_actions(faction, workers, action_set)
        self._legal_combat_actions(
            faction,
            combat_units,
            enemy_units,
            enemy_base_positions,
            enemy_base_approach_positions,
            friendly_guard_positions,
            action_set,
        )

        return sorted(action_set, key=str)

    def _success(
        self,
        reason: str,
        *,
        event: str | None = None,
        spend_action: bool = True,
    ) -> tuple[bool, str]:
        if spend_action:
            self.actions_left -= 1
        if event is not None:
            self._record_event(event)
        return True, reason

    def _handle_gather(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 2:
            return False, "bad_args"
        unit_id = action[1]
        if self._get_owned_unit(unit_id, faction) is None:
            return False, "not_your_unit"
        if not self.gather(unit_id):
            return False, "gather_failed"
        return self._success("gather", event=f"{faction} worker#{unit_id} gathered resources")

    def _handle_spawn_worker(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 1:
            return False, "bad_args"
        if not production.spawn_worker(self, faction):
            return False, "spawn_failed"
        new_unit = self._last_faction_unit(faction)
        return self._success("spawn_worker", event=f"{faction} spawned worker#{new_unit.id}")

    def _handle_train(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 2:
            return False, "bad_args"
        unit_type = action[1]
        if not production.train_unit(self, faction, unit_type):
            return False, "train_failed"
        new_unit = self._last_faction_unit(faction)
        return self._success("train", event=f"{faction} trained {unit_type}#{new_unit.id}")

    def _handle_build(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 4:
            return False, "bad_args"
        worker_id = action[1]
        building_type = action[2]
        pos = action[3]
        if self._get_owned_unit(worker_id, faction) is None:
            return False, "not_your_unit"
        if not production.build(self, faction, worker_id, building_type, pos):
            return False, "build_failed"
        return self._success("build", event=f"{faction} built {building_type} at {pos}")

    def _handle_research(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 2:
            return False, "bad_args"
        if self.free_research_used:
            return False, "research_used"
        tech_id = action[1]
        if not tech.research(self, faction, tech_id):
            return False, "research_failed"
        self.free_research_used = True
        return self._success("research", event=f"{faction} researched {tech_id}", spend_action=False)

    def _handle_declare_war(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 2:
            return False, "bad_args"
        target_faction = action[1]
        if not self.declare_war(faction, target_faction):
            return False, "declare_war_failed"
        return self._success("declare_war", event=f"{faction} declared war on {target_faction}")

    def _handle_offer_peace(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 3:
            return False, "bad_args"
        target_faction = action[1]
        indemnity = int(action[2])
        if not self.offer_peace(faction, target_faction, indemnity):
            return False, "offer_peace_failed"
        return self._success(
            "offer_peace",
            event=f"{faction} offered peace to {target_faction} for {indemnity}",
        )

    def _handle_accept_peace(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 2:
            return False, "bad_args"
        target_faction = action[1]
        indemnity = self.accept_peace(faction, target_faction)
        if indemnity is None:
            return False, "accept_peace_failed"
        return self._success(
            "accept_peace",
            event=f"{faction} accepted peace with {target_faction} (indemnity {indemnity})",
        )

    def _handle_attack(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 3:
            return False, "bad_args"
        attacker_id = action[1]
        target_id = action[2]
        target = self.get_unit(target_id)
        if not combat.attack(self, faction, attacker_id, target_id):
            return False, "attack_failed"
        self.attacked_unit_ids.add(attacker_id)
        if target is not None and target.hp <= 0:
            event = f"{faction} unit#{attacker_id} defeated {target.faction} {target.unit_type}#{target_id}"
        elif target is not None:
            event = f"{faction} unit#{attacker_id} hit {target.faction} {target.unit_type}#{target_id} ({target.hp} hp left)"
        else:
            event = None
        return self._success("attack", event=event)

    def _handle_attack_base(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 3:
            return False, "bad_args"
        attacker_id = action[1]
        target_faction = action[2]
        if not combat.attack_base(self, faction, attacker_id, target_faction):
            return False, "attack_base_failed"
        self.attacked_unit_ids.add(attacker_id)
        base_hp = self.bases[target_faction].hp
        if base_hp == 0:
            event = f"{faction} unit#{attacker_id} destroyed the {target_faction} base"
        else:
            event = f"{faction} unit#{attacker_id} hit the {target_faction} base ({base_hp} hp left)"
        return self._success("attack_base", event=event)

    def _handle_move_towards(self, faction: str, action: Action | tuple) -> tuple[bool, str]:
        if len(action) != 3:
            return False, "bad_args"
        unit_id = action[1]
        target = action[2]
        unit = self._get_owned_unit(unit_id, faction)
        if unit is None:
            return False, "not_your_unit"
        start_pos = unit.position
        if not self.move_towards(unit_id, target):
            return False, "move_blocked"
        return self._success(
            "move",
            event=f"{faction} unit#{unit_id} moved from {start_pos} to {unit.position} toward {target}",
        )

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

        self.attempts_left -= 1
        faction = self._current_faction()

        if not isinstance(action, tuple) or len(action) == 0:
            return False, "bad_action"

        handler = self._action_handlers.get(action[0])
        if handler is None:
            return False, "unknown_action"
        return handler(faction, action)

    def start_faction_turn(self) -> None:
        """Reset counters for the currently active faction."""
        self._reset_turn_state()

    def unit_max_hp(self, unit: Unit) -> int:
        return production.unit_stats(self, unit.faction, unit.unit_type).hp

    def heal_amount_for(self, unit: Unit) -> int:
        amount = self.config.unit_heal_per_turn
        if hexgrid.distance(unit.position, self.bases[unit.faction].position) <= self.config.unit_heal_base_radius:
            amount += self.config.unit_heal_near_base_bonus
        return amount

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

    def _passive_income_for(self, faction: str) -> int:
        return sum(
            production.BUILDING_DEFS[building.building_type].resource_income
            for building in self.get_buildings_for_faction(faction)
            if building.hp > 0 and building.building_type in production.BUILDING_DEFS
        )

    def _should_block_healing(self, faction: str, unit: Unit) -> bool:
        enemy = next(name for name in self.factions if name != faction)
        return any(
            other.faction == enemy
            and other.attack_damage > 0
            and hexgrid.distance(other.position, unit.position) <= 1
            for other in self.units
        )

    def _healing_events_for(self, faction: str) -> list[str]:
        events: list[str] = []
        for unit in self.get_units_for_faction(faction):
            if unit.hp <= 0 or unit.id in self.attacked_unit_ids:
                continue
            if self._should_block_healing(faction, unit):
                continue
            max_hp = self.unit_max_hp(unit)
            if unit.hp >= max_hp:
                continue
            healed = min(self.heal_amount_for(unit), max_hp - unit.hp)
            if healed <= 0:
                continue
            unit.hp += healed
            events.append(f"{faction} {unit.unit_type}#{unit.id} recovered {healed} hp")
        return events

    def _advance_turn_pointer(self) -> None:
        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            self.turn += 1

    def step_end_turn(self) -> None:
        faction = self._current_faction()
        self._record_events(combat.resolve_defensive_fire(self, faction))
        if self.winner() is not None:
            self._advance_turn_pointer()
            return

        completed_tech = tech.progress_research(self, faction)
        if completed_tech is not None:
            self._record_event(f"{faction} completed research: {completed_tech}")

        income = self._passive_income_for(faction)
        self.faction_state(faction).resources += income
        if income > 0:
            self._record_event(f"{faction} gained {income} passive income")

        self._record_events(self._healing_events_for(faction))
        self._advance_turn_pointer()

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
