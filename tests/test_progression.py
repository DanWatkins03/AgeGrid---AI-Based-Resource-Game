from __future__ import annotations

import unittest

import pygame

from src.agegrid.agents.greedy import GreedyAgent
from src.agegrid.agents.heuristic_arbitration import candidate_tiebreak_priority
from src.agegrid.agents.heuristic import HEURISTIC_PROFILES, HeuristicAgent, army_plan, heuristic_diagnostics, unit_composition
from src.agegrid.agents.heuristic_scoring import ScoringHelpers, utility_modifier
from src.agegrid.agents.heuristic_strategy import StrategicIntent, UtilityCandidate
from src.agegrid.agents.registry import create_agent
from src.agegrid.env.agegrid_env import AgeGridEnv, GameConfig
from src.agegrid.env.entities import ResourceNode
from src.agegrid.env import hexgrid
from src.agegrid.env.systems import combat, movement, production, tech, threat
from src.agegrid.ui.pygame_viewer import _fit_sprite_to_box, _trim_sprite_alpha
from src.agegrid.ui.turn_trace import build_debug_snapshot, step_full_turn, turn_snapshot


def make_env(**config_overrides) -> AgeGridEnv:
    config_kwargs = {
        "width": 12,
        "height": 12,
        "max_turns": 50,
        "actions_per_turn": 3,
        "max_attempts_per_turn": 10,
        "starting_resources": 250,
        "num_resource_nodes": 0,
        "stone_resource_nodes": 0,
        "horse_resource_nodes": 0,
        "seed": 7,
    }
    config_kwargs.update(config_overrides)
    config = GameConfig(**config_kwargs)
    return AgeGridEnv(config)


class TechSystemTests(unittest.TestCase):
    def test_research_requires_prerequisites_and_spends_resources(self) -> None:
        env = make_env()

        self.assertTrue(tech.can_research(env, "Red", "mining"))
        self.assertFalse(tech.can_research(env, "Red", "bronze"))

        self.assertTrue(tech.research(env, "Red", "mining"))
        self.assertEqual(env.faction_state("Red").tech_in_progress, "mining")
        self.assertEqual(env.bank["Red"], 215)
        self.assertFalse(tech.can_research(env, "Red", "bronze"))

    def test_research_completes_after_required_turns(self) -> None:
        env = make_env()

        self.assertTrue(tech.research(env, "Red", "mining"))
        self.assertEqual(tech.research_turns_remaining(env, "Red"), 2)
        self.assertIsNone(tech.progress_research(env, "Red"))
        self.assertEqual(tech.research_turns_remaining(env, "Red"), 1)
        self.assertEqual(tech.progress_research(env, "Red"), "mining")
        self.assertIn("mining", env.faction_state("Red").techs_unlocked)

    def test_year_formatting_advances_with_turns(self) -> None:
        env = make_env()

        self.assertEqual(env.current_year(), -3000)
        self.assertEqual(env.formatted_year(), "3000 BCE")

        env.turn = 10
        self.assertEqual(env.current_year(), -2750)
        self.assertEqual(env.formatted_year(), "2750 BCE")

    def test_era_tracks_highest_researched_tech_milestone(self) -> None:
        env = make_env()
        self.assertEqual(env.current_era(), "Founding Age")

        env.faction_state("Red").techs_unlocked.add("mining")
        self.assertEqual(env.current_era(), "Stone Age")

        env.faction_state("Blue").techs_unlocked.add("bronze")
        self.assertEqual(env.current_era(), "Bronze Age")

        env.faction_state("Red").techs_unlocked.add("iron")
        self.assertEqual(env.current_era(), "Iron Age")

        env.faction_state("Blue").techs_unlocked.add("engineering")
        self.assertEqual(env.current_era(), "Engineering Age")
        self.assertIsNone(env.faction_state("Red").tech_in_progress)
        self.assertTrue(tech.can_research(env, "Red", "bronze"))

    def test_research_is_free_once_per_turn_and_preserves_action_points(self) -> None:
        env = make_env()
        env.start_faction_turn()

        ok, reason = env.apply_action(("research", "mining"))
        self.assertTrue(ok)
        self.assertEqual(reason, "research")
        self.assertEqual(env.actions_left, env.config.actions_per_turn)
        self.assertFalse(env.apply_action(("research", "bronze"))[0])

    def test_stronghold_requires_both_fortify_and_construction(self) -> None:
        env = make_env()
        state = env.faction_state("Red")
        state.techs_unlocked.update({"mining", "bronze", "masonry", "iron"})

        self.assertFalse(tech.can_research(env, "Red", "stronghold"))

        state.techs_unlocked.add("fortify")
        self.assertFalse(tech.can_research(env, "Red", "stronghold"))

        state.techs_unlocked.add("construction")
        self.assertTrue(tech.can_research(env, "Red", "stronghold"))

    def test_advanced_siege_unlocks_ballista_after_dual_prerequisites(self) -> None:
        env = make_env()
        state = env.faction_state("Red")
        state.techs_unlocked.update({"mining", "bronze", "fletching", "engineering"})

        self.assertFalse(tech.can_research(env, "Red", "advanced_siege"))

        state.techs_unlocked.add("iron")
        state.techs_unlocked.add("steel")
        self.assertTrue(tech.can_research(env, "Red", "advanced_siege"))
        self.assertTrue(tech.research(env, "Red", "advanced_siege"))
        while tech.progress_research(env, "Red") is None:
            pass

        self.assertIn("advanced_siege", state.techs_unlocked)
        self.assertIn("ballista", tech.unlocked_units(env, "Red"))

    def test_war_economy_applies_military_discount_and_income_bonus(self) -> None:
        env = make_env()
        state = env.faction_state("Red")
        state.techs_unlocked.update({"mining", "bronze", "masonry", "trade", "iron", "war_economy"})

        self.assertEqual(production.unit_cost(env, "Red", "soldier"), 25)
        self.assertEqual(tech.passive_modifier_total(env, "Red", "economy_income_bonus"), 1)

    def test_infrastructure_no_longer_adds_passive_income_bonus(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry", "construction", "infrastructure"})

        self.assertEqual(tech.passive_modifier_total(env, "Red", "building_cost_discount_pct"), 10)
        self.assertEqual(tech.passive_modifier_total(env, "Red", "economy_income_bonus"), 0)


class ProductionSystemTests(unittest.TestCase):
    def _stone_build_setup(self, env: AgeGridEnv) -> tuple[object, object, tuple[int, int]]:
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        stone = next(resource for resource in env.visible_resources("Red") if resource.resource_type == "stone")
        build_pos = next(
            pos
            for pos in (
                (stone.position[0] + 1, stone.position[1]),
                (stone.position[0] - 1, stone.position[1]),
                (stone.position[0], stone.position[1] + 1),
                (stone.position[0], stone.position[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos not in {b.position for b in env.buildings}
        )
        worker.position = next(
            pos
            for pos in (
                (build_pos[0] + 1, build_pos[1]),
                (build_pos[0] - 1, build_pos[1]),
                (build_pos[0], build_pos[1] + 1),
                (build_pos[0], build_pos[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos != stone.position
        )
        return worker, stone, build_pos

    def _horse_build_setup(self, env: AgeGridEnv) -> tuple[object, object, tuple[int, int]]:
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        horse = next(resource for resource in env.visible_resources("Red") if resource.resource_type == "horses")
        build_pos = next(
            pos
            for pos in (
                (horse.position[0] + 1, horse.position[1]),
                (horse.position[0] - 1, horse.position[1]),
                (horse.position[0], horse.position[1] + 1),
                (horse.position[0], horse.position[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos not in {b.position for b in env.buildings}
        )
        worker.position = next(
            pos
            for pos in (
                (build_pos[0] + 1, build_pos[1]),
                (build_pos[0] - 1, build_pos[1]),
                (build_pos[0], build_pos[1] + 1),
                (build_pos[0], build_pos[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos != horse.position
        )
        return worker, horse, build_pos

    def test_training_soldier_requires_barracks(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})

        self.assertFalse(production.can_train_unit(env, "Red", "soldier"))

        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "barracks", (3, 1)))
        self.assertTrue(production.can_train_unit(env, "Red", "soldier"))
        self.assertTrue(production.train_unit(env, "Red", "soldier"))

        soldiers = [u for u in env.units if u.faction == "Red" and u.unit_type == "soldier"]
        self.assertEqual(len(soldiers), 1)
        self.assertEqual(soldiers[0].attack_damage, 3)

    def test_training_can_spawn_on_friendly_building_tiles(self) -> None:
        env = make_env()
        env.units = []
        env._unit_index = {}
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        spawn_ring = [pos for pos in hexgrid.neighbors(env.bases["Red"].position) if env._in_bounds(pos)]
        for pos in spawn_ring:
            env._spawn_building("Red", "storehouse", 18, pos)

        self.assertTrue(production.can_train_unit(env, "Red", "worker"))
        self.assertTrue(production.train_unit(env, "Red", "worker"))
        worker = env.get_units_for_faction("Red")[0]
        self.assertIn(worker.position, {building.position for building in env.get_buildings_for_faction("Red")})

    def test_building_can_be_constructed_on_friendly_troop_tile(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.add("mining")
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        build_pos = (3, 1)
        env._spawn_unit("Red", "soldier", 10, build_pos, attack_damage=3, attack_range=1)

        self.assertTrue(production.can_build(env, "Red", worker.id, "storehouse", build_pos))
        self.assertTrue(production.build(env, "Red", worker.id, "storehouse", build_pos))
        self.assertTrue(any(building.position == build_pos for building in env.get_buildings_for_faction("Red")))
        self.assertTrue(any(unit.position == build_pos for unit in env.get_units_for_faction("Red")))

    def test_enemy_troops_block_construction(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.add("mining")
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        build_pos = (3, 1)
        env._spawn_unit("Blue", "soldier", 10, build_pos, attack_damage=3, attack_range=1)

        self.assertFalse(production.can_build(env, "Red", worker.id, "storehouse", build_pos))

    def test_storehouse_adds_income_on_owner_turn_end(self) -> None:
        env = make_env(target_bank=999)
        env.faction_state("Red").techs_unlocked.add("mining")

        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "storehouse", (3, 1)))

        red_before = env.bank["Red"]
        blue_before = env.bank["Blue"]
        env.step_end_turn()

        self.assertEqual(env.bank["Red"], red_before + 3)
        self.assertEqual(env.bank["Blue"], blue_before)

    def test_gathering_does_not_deplete_resource_node(self) -> None:
        env = make_env(num_resource_nodes=0)
        from src.agegrid.env.entities import ResourceNode

        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        worker.position = (3, 1)
        env.resources = [ResourceNode(id=1, position=(3, 1), abundance=60, resource_type="ore")]

        self.assertTrue(env.gather(worker.id))
        self.assertEqual(env.resources[0].abundance, 60)
        self.assertEqual(env.bank["Red"], 255)

    def test_enemy_military_contests_infinite_resource_access(self) -> None:
        env = make_env(num_resource_nodes=0, target_bank=999)
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        worker.position = (3, 1)
        env.resources = [ResourceNode(id=1, position=(3, 1), abundance=60, resource_type="ore")]
        env._spawn_unit("Blue", "soldier", 10, (4, 1), attack_damage=3, attack_range=1)
        env.declare_war("Blue", "Red")

        self.assertTrue(env.resource_is_contested(env.resources[0], "Red"))
        self.assertFalse(env.gather(worker.id))
        self.assertNotIn(("gather", worker.id), env.legal_actions("Red"))

    def test_building_cannot_be_placed_directly_on_resource_tile(self) -> None:
        env = make_env(num_resource_nodes=0)
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        env.faction_state("Red").techs_unlocked.add("mining")
        worker.position = (3, 2)
        env.resources = [ResourceNode(id=1, position=(3, 1), abundance=60, resource_type="ore")]

        self.assertFalse(production.can_build(env, "Red", worker.id, "storehouse", (3, 1)))
        self.assertFalse(production.build(env, "Red", worker.id, "storehouse", (3, 1)))

    def test_buildings_do_not_block_base_approach_lanes(self) -> None:
        env = make_env(num_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.add("mining")
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        red_base = env.bases["Red"].position
        approaches = [pos for pos in hexgrid.neighbors(red_base) if env._in_bounds(pos)]
        for pos in approaches[:-2]:
            env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=pos)
        blocked_pos = approaches[-2]
        worker.position = next(
            pos
            for pos in hexgrid.neighbors(blocked_pos)
            if env._in_bounds(pos) and pos not in env._occupied_positions() and env.resource_at(pos) is None
        )

        self.assertTrue(production.can_build(env, "Red", worker.id, "storehouse", blocked_pos))
        self.assertTrue(production.build(env, "Red", worker.id, "storehouse", blocked_pos))

    def test_quarry_requires_visible_stone_resource(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry"})
        worker, _, build_pos = self._stone_build_setup(env)

        self.assertTrue(production.can_build(env, "Red", worker.id, "quarry", build_pos))
        self.assertTrue(production.build(env, "Red", worker.id, "quarry", build_pos))

    def test_stable_requires_visible_horse_resource(self) -> None:
        env = make_env(num_resource_nodes=0, horse_resource_nodes=2, horse_resource_amount=20)
        env.faction_state("Red").techs_unlocked.update(
            {
                "mining",
                "bronze",
                "masonry",
                "animal_husbandry",
                "horseback_riding",
                "fletching",
                "iron",
                "construction",
                "fortify",
                "stirrups",
                "engineering",
            }
        )
        worker, _, build_pos = self._horse_build_setup(env)

        self.assertTrue(production.can_build(env, "Red", worker.id, "stable", build_pos))
        self.assertTrue(production.build(env, "Red", worker.id, "stable", build_pos))

    def test_horseman_requires_stable_and_has_extended_move(self) -> None:
        env = make_env(num_resource_nodes=0, horse_resource_nodes=2, horse_resource_amount=20)
        env.faction_state("Red").techs_unlocked.update({"mining", "animal_husbandry", "horseback_riding"})
        worker, _, build_pos = self._horse_build_setup(env)
        self.assertTrue(production.build(env, "Red", worker.id, "stable", build_pos))

        self.assertTrue(production.can_train_unit(env, "Red", "horseman"))
        self.assertTrue(production.train_unit(env, "Red", "horseman"))
        horseman = next(u for u in env.units if u.faction == "Red" and u.unit_type == "horseman")
        start = horseman.position
        self.assertTrue(movement.move_towards(env, horseman.id, env.bases["Blue"].position))
        self.assertGreaterEqual(
            abs(horseman.position[0] - start[0]) + abs(horseman.position[1] - start[1]),
            2,
        )

    def test_iron_upgrades_new_soldiers(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "iron"})
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "barracks", (3, 1)))
        self.assertTrue(production.train_unit(env, "Red", "soldier"))

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertEqual(soldier.hp, 12)
        self.assertEqual(soldier.attack_damage, 4)

    def test_engineering_unlocks_ballista_tower(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry", "construction", "engineering"})
        worker, _, quarry_pos = self._stone_build_setup(env)
        self.assertTrue(production.build(env, "Red", worker.id, "quarry", quarry_pos))
        tower_worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        tower_worker.position = (quarry_pos[0] + 1, quarry_pos[1])
        tower_pos = next(
            pos
            for pos in (
                (tower_worker.position[0] + 1, tower_worker.position[1]),
                (tower_worker.position[0] - 1, tower_worker.position[1]),
                (tower_worker.position[0], tower_worker.position[1] + 1),
                (tower_worker.position[0], tower_worker.position[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos not in {b.position for b in env.buildings}
        )
        self.assertTrue(production.build(env, "Red", tower_worker.id, "archer_tower", tower_pos))
        ballista_worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        ballista_worker.position = (tower_pos[0] + 1, tower_pos[1])
        ballista_pos = next(
            pos
            for pos in (
                (ballista_worker.position[0] + 1, ballista_worker.position[1]),
                (ballista_worker.position[0] - 1, ballista_worker.position[1]),
                (ballista_worker.position[0], ballista_worker.position[1] + 1),
                (ballista_worker.position[0], ballista_worker.position[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos not in {b.position for b in env.buildings}
        )

        self.assertTrue(production.can_build(env, "Red", ballista_worker.id, "ballista_tower", ballista_pos))

    def test_market_income_uses_tuned_base_value(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry", "trade", "markets"})
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "storehouse", (3, 1)))
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "market", (3, 2)))

        market = next(b for b in env.buildings if b.faction == "Red" and b.building_type == "market")
        self.assertEqual(production.building_stats(env, "Red", market.building_type).resource_income, 4)


class HeuristicAgentTests(unittest.TestCase):
    def test_agent_declares_war_when_push_ready(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (7, 7), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (8, 7), attack_damage=3, attack_range=1, move_steps=1)
        env.units = [u for u in env.units if not (u.faction == "Blue" and u.attack_damage > 0)]
        env.faction_states["Blue"].unit_ids = [u.id for u in env.units if u.faction == "Blue"]

        self.assertEqual(agent.act(env), ("declare_war", "Blue"))

    def test_agent_explains_no_legal_actions_during_collapse_recovery(self) -> None:
        env = make_env(collapse_enabled=False)
        agent = HeuristicAgent()
        env.declare_war("Red", "Blue")
        env.units = [unit for unit in env.units if unit.faction != "Red"]
        env.faction_states["Red"].unit_ids.clear()
        env.bank["Red"] = env.config.worker_spawn_cost - 1

        self.assertEqual(env.legal_actions("Red"), [])
        self.assertIsNone(agent.act(env))
        self.assertEqual(
            agent.explain_last_decision(),
            f"No legal actions: no units, bank {env.config.worker_spawn_cost - 1}/{env.config.worker_spawn_cost} for worker recovery",
        )

    def test_agent_redeclares_war_under_border_pressure_even_if_enemy_support_is_higher(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = 40
        env.resources = []
        env.bank["Red"] = 60
        env.bank["Blue"] = 400
        env.faction_state("Red").war_support = 56
        env.faction_state("Blue").war_support = 91
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (5, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (5, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (6, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (7, 2), attack_damage=3, attack_range=1, move_steps=1)

        self.assertEqual(env.relation_state("Red", "Blue").state, "peace")
        self.assertIn(("declare_war", "Blue"), env.legal_actions("Red"))
        self.assertEqual(agent.act(env), ("declare_war", "Blue"))

    def test_agent_prefers_progression_actions(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)

        self.assertEqual(agent.act(env), ("research", "mining"))

        env.faction_state("Red").techs_unlocked.add("mining")
        build_action = agent.act(env)
        self.assertIsNotNone(build_action)
        self.assertEqual(build_action[0], "build")
        self.assertEqual(build_action[2], "storehouse")

        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(3, 1))
        self.assertEqual(agent.act(env), ("research", "bronze"))

    def test_agent_pursues_animal_husbandry_when_horses_are_available(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=0, horse_resource_nodes=2, horse_resource_amount=20)
        agent = HeuristicAgent(desired_workers=1)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))

        self.assertEqual(agent.act(env), ("research", "animal_husbandry"))

    def test_agent_pursues_masonry_when_stone_is_available(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        agent = HeuristicAgent(desired_workers=1)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))

        self.assertEqual(agent.act(env), ("research", "masonry"))

    def test_heuristic_diagnostics_flags_recovery_when_far_behind(self) -> None:
        env = make_env()
        env.bank["Red"] = 20
        env.bank["Blue"] = 220
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update(
            {"mining", "bronze", "masonry", "construction", "trade", "markets", "currency", "engineering"}
        )
        env._spawn_building(faction="Blue", building_type="storehouse", hp=18, pos=(10, 10))
        env._spawn_building(faction="Blue", building_type="market", hp=20, pos=(10, 11))
        env._spawn_unit("Blue", "soldier", 10, (9, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (9, 10), attack_damage=3, attack_range=1, move_steps=1)

        diagnostics = heuristic_diagnostics(env, "Red")
        self.assertTrue(diagnostics.behind)
        self.assertTrue(diagnostics.recovery)
        self.assertGreaterEqual(diagnostics.tech_deficit, 4)
        self.assertGreaterEqual(diagnostics.economy_gap, 5)

    def test_agent_prioritizes_construction_research_when_behind(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "trade", "fletching"})
        env.faction_state("Blue").techs_unlocked.update(
            {"mining", "bronze", "masonry", "construction", "engineering", "iron", "fortify", "walls"}
        )
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))

        self.assertEqual(agent.act(env), ("research", "construction"))

    def test_agent_builds_archer_tower_before_siege_workshop_when_behind(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 200
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "construction", "engineering"})
        env.faction_state("Red").tech_in_progress = "engineering"
        env.faction_state("Blue").techs_unlocked.update(
            {"mining", "bronze", "masonry", "construction", "engineering", "fortify", "walls", "currency"}
        )
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="quarry", hp=20, pos=(3, 1))
        env._spawn_unit("Blue", "soldier", 10, (10, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (10, 9), attack_damage=3, attack_range=1, move_steps=1)

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "build")
        self.assertEqual(action[2], "archer_tower")

    def test_agent_builds_siege_workshop_for_late_siege_plan(self) -> None:
        env = make_env(width=14, height=14, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.bank["Red"] = 500
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze", "masonry", "fletching", "construction", "iron", "steel", "engineering", "advanced_siege"}
        )
        env.faction_state("Red").tech_in_progress = "advanced_siege"
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Red", "soldier", 14, (2, 0), attack_damage=4, attack_range=1)
        env._spawn_unit("Red", "soldier", 14, (2, 1), attack_damage=4, attack_range=1)
        env._spawn_unit("Blue", "soldier", 10, (10, 10), attack_damage=3, attack_range=1)

        action = agent.act(env)

        self.assertIsNotNone(action)
        self.assertEqual(action[0], "build")
        self.assertEqual(action[2], "siege_workshop")

    def test_agent_trains_multiple_ballistae_against_static_defense(self) -> None:
        env = make_env(width=14, height=14, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.bank["Red"] = 500
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze", "masonry", "fletching", "construction", "iron", "steel", "engineering", "advanced_siege"}
        )
        env.faction_state("Red").tech_in_progress = "advanced_siege"
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="quarry", hp=20, pos=(1, 2))
        env._spawn_building(faction="Red", building_type="archer_tower", hp=22, pos=(2, 1))
        env._spawn_building(faction="Red", building_type="ballista_tower", hp=28, pos=(3, 1))
        env._spawn_building(faction="Red", building_type="siege_workshop", hp=24, pos=(3, 2))
        env._spawn_unit("Red", "ballista", 10, (2, 2), attack_damage=5, attack_range=4)
        env._spawn_building(faction="Blue", building_type="stronghold", hp=42, pos=(10, 10))

        self.assertEqual(agent.act(env), ("train", "ballista"))

    def test_agent_transitions_to_heavy_cavalry_core(self) -> None:
        env = make_env(width=14, height=14, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.bank["Red"] = 500
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze", "animal_husbandry", "horseback_riding", "iron", "stirrups", "heavy_cavalry"}
        )
        env.faction_state("Red").tech_in_progress = "heavy_cavalry"
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "heavy_cavalry", 16, (2, 2), attack_damage=5, attack_range=1, move_steps=3)

        self.assertEqual(agent.act(env), ("train", "heavy_cavalry"))

    def test_agent_fortifies_recovery_with_stronghold(self) -> None:
        env = make_env(width=14, height=14, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.bank["Red"] = 500
        env.bank["Blue"] = 800
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze", "masonry", "construction", "iron", "fortify", "walls", "stronghold"}
        )
        env.faction_state("Blue").techs_unlocked.update(
            {"mining", "bronze", "masonry", "construction", "iron", "steel", "engineering", "currency"}
        )
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="quarry", hp=20, pos=(2, 0))
        env._spawn_building(faction="Red", building_type="archer_tower", hp=22, pos=(2, 1))
        env._spawn_building(faction="Red", building_type="ballista_tower", hp=28, pos=(3, 1))
        env._spawn_building(faction="Red", building_type="wall", hp=34, pos=(1, 2))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Blue", "soldier", 14, (10, 10), attack_damage=5, attack_range=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 11), attack_damage=5, attack_range=1)

        action = agent.act(env)

        self.assertIsNotNone(action)
        self.assertEqual(action[0], "build")
        self.assertEqual(action[2], "stronghold")

    def test_unit_composition_reports_late_game_units(self) -> None:
        env = make_env()
        env._spawn_unit("Red", "heavy_cavalry", 16, (2, 2), attack_damage=5, attack_range=1, move_steps=3)
        env._spawn_unit("Red", "ballista", 10, (2, 3), attack_damage=5, attack_range=4)

        composition = unit_composition(env, "Red")

        self.assertEqual(composition["heavy_cavalry"], 1)
        self.assertEqual(composition["ballista"], 1)

    def test_threat_map_scores_enemy_threat_and_friendly_cover(self) -> None:
        env = make_env(width=12, height=12, target_bank=999)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "archer", 8, (5, 5), attack_damage=3, attack_range=3)
        env.declare_war("Red", "Blue")

        threat_map = threat.build_threat_map(env, "Red")

        self.assertEqual(threat_map.enemy_threat_at((4, 4)), 3)
        self.assertEqual(threat_map.friendly_cover_at((2, 3)), 3)
        self.assertEqual(threat_map.danger_at((4, 4)), 3)
        self.assertEqual(threat_map.danger_at((2, 3)), 0)

    def test_agent_places_defense_building_near_contested_resource_pressure(self) -> None:
        env = make_env(width=12, height=12, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.resources = [ResourceNode(id=1, position=(7, 5), abundance=60)]
        env.bank["Red"] = 500
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "construction"})
        env.faction_state("Red").tech_in_progress = "construction"
        env.faction_state("Blue").techs_unlocked.update(
            {"mining", "bronze", "masonry", "construction", "engineering", "trade", "markets"}
        )
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="quarry", hp=20, pos=(2, 0))
        env._spawn_unit("Red", "worker", 5, (6, 5))
        env._spawn_unit("Blue", "soldier", 10, (8, 5), attack_damage=3, attack_range=1)
        env.declare_war("Red", "Blue")

        action = agent.act(env)

        self.assertIsNotNone(action)
        self.assertEqual(action[0], "build")
        self.assertEqual(action[2], "archer_tower")
        self.assertIn(action[3], {(7, 4), (7, 6)})

    def test_agent_builds_storehouse_before_endless_gathering(self) -> None:
        env = make_env(num_resource_nodes=2, resource_per_node=60)
        agent = HeuristicAgent(desired_workers=1)
        env.faction_state("Red").techs_unlocked.add("mining")
        env.bank["Red"] = 100

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "build")
        self.assertEqual(action[2], "storehouse")

    def test_greedy_profile_uses_shared_heuristic_progression(self) -> None:
        env = make_env(num_resource_nodes=2, resource_per_node=60)
        agent = GreedyAgent()
        env.faction_state("Red").techs_unlocked.add("mining")
        env.bank["Red"] = 100

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "build")
        self.assertEqual(action[2], "storehouse")

    def test_registry_personality_agents_are_profiled_heuristics(self) -> None:
        aggressive = create_agent("aggressive")
        defensive = create_agent("defensive")
        greedy = create_agent("greedy")

        self.assertIsInstance(aggressive, HeuristicAgent)
        self.assertIsInstance(defensive, HeuristicAgent)
        self.assertIsInstance(greedy, HeuristicAgent)
        self.assertEqual(aggressive.profile, HEURISTIC_PROFILES["aggressive"])
        self.assertEqual(defensive.profile, HEURISTIC_PROFILES["defensive"])
        self.assertEqual(greedy.profile, HEURISTIC_PROFILES["greedy"])

    def test_agent_clears_spawn_ring_before_gathering_again(self) -> None:
        env = make_env(num_resource_nodes=0)
        agent = HeuristicAgent(desired_workers=1)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env.bank["Red"] = 0
        env.resources = [ResourceNode(id=1, position=(0, 0), abundance=60)]
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(3, 3))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(4, 1))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(4, 2))
        env._spawn_unit("Red", "worker", 5, (0, 1))
        env._spawn_unit("Red", "worker", 5, (1, 2))
        env._spawn_unit("Red", "worker", 5, (1, 0))
        spawn_ring_ids = {
            unit.id
            for unit in env.units
            if unit.faction == "Red" and unit.position in {(2, 1), (0, 1), (1, 2), (1, 0)}
        }

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertIn(action[1], spawn_ring_ids)

    def test_worker_does_not_chase_enemy_without_useful_economic_task(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 17

        self.assertIsNone(agent.act(env))

    def test_agent_does_not_spawn_extra_worker_without_useful_jobs(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 100

        action = agent.act(env)
        self.assertNotEqual(action, ("spawn_worker",))

    def test_agent_rotates_worker_gather_actions_across_workers(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=2)
        env.bank["Red"] = 0
        env.resources = [
            ResourceNode(id=1, position=(3, 1), abundance=60),
            ResourceNode(id=2, position=(4, 2), abundance=60),
        ]
        env.faction_state("Red").techs_unlocked.update(
            {
                "mining",
                "bronze",
                "masonry",
                "animal_husbandry",
                "horseback_riding",
                "fletching",
                "iron",
                "fortify",
                "stirrups",
                "engineering",
            }
        )
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        red_workers = [u for u in env.units if u.faction == "Red" and u.unit_type == "worker"]
        red_workers[0].position = (3, 1)
        env._spawn_unit("Red", "worker", 5, (4, 2))

        first_action = agent.act(env)
        self.assertIsNotNone(first_action)
        self.assertEqual(first_action[0], "gather")
        self.assertTrue(env.apply_action(first_action)[0])

        second_action = agent.act(env)
        self.assertIsNotNone(second_action)
        self.assertEqual(second_action[0], "gather")
        self.assertNotEqual(second_action[1], first_action[1])

    def test_worker_moves_toward_uncontested_resource_before_contested_one(self) -> None:
        env = make_env(num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.resources = [
            ResourceNode(id=1, position=(5, 2), abundance=60),
            ResourceNode(id=2, position=(7, 3), abundance=60),
        ]
        env._spawn_unit("Red", "worker", 5, (1, 0))
        env._spawn_unit("Blue", "soldier", 10, (6, 2), attack_damage=3, attack_range=1)
        env.bank["Red"] = 12
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        env.faction_state("Red").tech_in_progress = "mining"
        red_worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")

        self.assertTrue(env.resource_is_contested(env.resources[0], "Red"))
        self.assertFalse(env.resource_is_contested(env.resources[1], "Red"))
        self.assertEqual(agent.act(env), ("move_towards", red_worker.id, (7, 3)))

    def test_agent_sends_military_to_clear_contested_resource(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(6, 6), abundance=60)]
        env.faction_state("Red").tech_in_progress = "mining"
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (7, 6), attack_damage=3, attack_range=1, move_steps=1)
        env.bank["Red"] = 12
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_soldier = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")

        self.assertTrue(env.resource_is_contested(env.resources[0], "Red"))
        self.assertEqual(agent.act(env), ("move_towards", red_soldier.id, blue_soldier.position))

    def test_agent_raids_enemy_worker_on_resource_when_pressure_is_safe(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.resources = [ResourceNode(id=1, position=(6, 5), abundance=60)]
        env._spawn_unit("Red", "worker", 5, (1, 0))
        env._spawn_unit("Red", "soldier", 10, (3, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (6, 5))
        env.turn = 120
        env.bank["Red"] = 12
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        env.faction_state("Red").tech_in_progress = "mining"
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        self.assertEqual(agent.act(env), ("move_towards", red_soldier.id, (6, 5)))

    def test_military_unit_moves_toward_enemy_when_no_attack_available(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 0
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(4, 4),
            attack_damage=3,
            attack_range=1,
            move_steps=1,
        )

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[1], next(u.id for u in env.units if u.faction == "Red" and u.unit_type == "soldier"))

    def test_agent_prefers_attacking_combat_unit_over_worker(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 12
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))
        env._spawn_unit("Blue", "soldier", 10, (4, 5), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_soldier = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, blue_soldier.id))

    def test_agent_moves_to_defend_worker_from_nearby_enemy(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        red_worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        red_worker.position = (3, 3)
        env._spawn_unit("Red", "soldier", 10, (1, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (4, 3), attack_damage=3, attack_range=1, move_steps=1)
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("move_towards", red_soldier.id, (4, 3)))

    def test_defense_mode_deprioritizes_economy_when_base_is_threatened(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 4, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, next(u.id for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")))

    def test_base_siege_worker_fallback_retreats_to_passable_base(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(0, 11), abundance=60)]
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "worker", 5, (2, 0))
        env._spawn_unit("Blue", "archer", 8, (3, 4), attack_damage=3, attack_range=3)
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 1))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(3, 0))
        env.bank["Red"] = 24
        env.bank["Blue"] = 12
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching"})
        env.declare_war("Blue", "Red")

        action = agent.act(env)

        self.assertEqual(action, ("move_towards", 1, env.bases["Red"].position))

    def test_agent_trains_defender_when_enemy_pressure_approaches_base(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=3)
        env.resources = [ResourceNode(id=1, position=(2, 0), abundance=60)]
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Red").tech_in_progress = "bronze"
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 1))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(3, 0))
        env.units = [unit for unit in env.units if unit.faction == "Red" and unit.unit_type == "worker"]
        env.faction_states["Red"].unit_ids = [unit.id for unit in env.units if unit.faction == "Red"]
        env.faction_states["Blue"].unit_ids.clear()
        env._spawn_unit("Blue", "soldier", 10, (6, 5), attack_damage=3, attack_range=1)
        env.bank["Red"] = 30
        env.bank["Blue"] = 12
        env.declare_war("Blue", "Red")

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "train")
        self.assertIsNotNone(agent.last_decision)
        self.assertEqual(agent.last_decision.source, "emergency_production")
        self.assertIn("home_pressure", agent.last_decision.reasons)
        self.assertIn("needs_home_force", agent.last_decision.reasons)

    def test_agent_holds_worker_spawn_money_for_defender_under_home_pressure(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=3)
        env.resources = [ResourceNode(id=1, position=(2, 0), abundance=60)]
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Red").tech_in_progress = "bronze"
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 1))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(3, 0))
        env.units = [unit for unit in env.units if unit.faction == "Red" and unit.unit_type == "worker"]
        env.faction_states["Red"].unit_ids = [unit.id for unit in env.units if unit.faction == "Red"]
        env.faction_states["Blue"].unit_ids.clear()
        red_worker = next(unit for unit in env.units if unit.faction == "Red")
        red_worker.position = (2, 0)
        env._spawn_unit("Blue", "soldier", 10, (6, 5), attack_damage=3, attack_range=1)
        env.bank["Red"] = 25
        env.bank["Blue"] = 12
        env.declare_war("Blue", "Red")

        self.assertIn(("spawn_worker",), env.legal_actions("Red"))
        self.assertEqual(agent.act(env), ("gather", red_worker.id))
        self.assertIsNotNone(agent.last_decision)
        self.assertEqual(agent.last_decision.source, "economy")
        self.assertIn("income_now", agent.last_decision.reasons)

    def test_heuristic_records_ranked_utility_candidates(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)

        action = agent.act(env)

        self.assertEqual(action, ("research", "mining"))
        self.assertIsNotNone(agent.last_decision)
        self.assertEqual(agent.last_decision.action, action)
        self.assertEqual(agent.last_candidates[0], agent.last_decision)
        self.assertGreaterEqual(agent.last_candidates[0].score, agent.last_candidates[-1].score)
        self.assertIn("research:mining", agent.explain_last_decision())
        self.assertIn("Intent develop", agent.explain_last_decision())

    def test_candidate_tiebreak_prefers_military_movement_over_equal_gather(self) -> None:
        gather = UtilityCandidate(
            action=("gather", 1),
            source="economy",
            score=540,
            reasons=("recover", "income_now"),
        )
        movement = UtilityCandidate(
            action=("move_towards", 2, (10, 10)),
            source="military_movement",
            score=540,
            reasons=("recover", "move:soldier", "military_positioning"),
        )

        ranked = sorted(
            [gather, movement],
            key=lambda candidate: (
                -candidate.score,
                candidate_tiebreak_priority(candidate),
                candidate.source,
                candidate.action,
            ),
        )

        self.assertEqual(ranked[0], movement)

    def test_recovery_intent_avoids_deep_resource_pressure_when_far_behind(self) -> None:
        env = make_env(width=14, height=14, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(11, 11), abundance=60)]
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "iron", "steel", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 10), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 11), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (11, 10), attack_damage=3, attack_range=4, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (11, 11))
        env.declare_war("Red", "Blue")

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))

        self.assertTrue(ctx.recovery_mode)
        self.assertIsNone(agent._choose_resource_pressure_action(env, ctx))

    def test_push_intent_overrides_moderate_recovery_gap_when_pressure_is_live(self) -> None:
        env = make_env(width=14, height=14, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(10, 10), abundance=60)]
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (1, 2))
        env._spawn_unit("Red", "soldier", 10, (8, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (9, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (10, 10))
        env._spawn_unit("Blue", "worker", 5, (11, 10))
        env._spawn_unit("Blue", "worker", 5, (10, 11))
        env.bank["Red"] = 12
        env.bank["Blue"] = 120
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        self.assertTrue(ctx.recovery_mode)
        self.assertTrue(ctx.push_mode)
        self.assertEqual(ctx.recovery_posture, "fragile")
        self.assertEqual(agent._strategic_intent(ctx).name, "push")

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertIn(action[0], {"attack", "move_towards"})
        self.assertIn("Intent push", agent.explain_last_decision())

    def test_recovery_scoring_prefers_war_pressure_over_gathering_when_ahead(self) -> None:
        env = make_env(width=14, height=14)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(1, 2), abundance=60)]
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (1, 2))
        env._spawn_unit("Red", "soldier", 10, (8, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (9, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (10, 10))
        env._spawn_unit("Blue", "worker", 5, (11, 10))
        env._spawn_unit("Blue", "worker", 5, (10, 11))
        env.bank["Red"] = 12
        env.bank["Blue"] = 120
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        gather_action = next(action for action in ctx.legal if action[0] == "gather")
        pressure_action = next(
            action
            for action in ctx.legal
            if action[0] == "move_towards"
            and any(unit.id == action[1] and unit.attack_damage > 0 for unit in ctx.military)
            and hexgrid.distance(action[2], env.bases["Blue"].position) <= 3
        )
        helpers = ScoringHelpers(
            distance=hexgrid.distance,
            enemy_pressure_near_base=lambda _env, _faction: False,
            enemy_can_finish_base_next_turn=lambda _env, _faction: False,
            hold_defender_reserve=lambda _env, _ctx, _spend: False,
            unit_count=lambda _env, faction, unit_type: sum(
                1 for unit in _env.units if unit.faction == faction and unit.unit_type == unit_type
            ),
        )
        recovery_intent = StrategicIntent("recover", "moderate economy gap", "stabilize", "medium")

        self.assertTrue(ctx.recovery_mode)
        self.assertTrue(ctx.at_war)
        self.assertTrue(ctx.push_mode)
        self.assertGreater(
            utility_modifier(env, ctx, pressure_action, recovery_intent, helpers),
            utility_modifier(env, ctx, gather_action, recovery_intent, helpers),
        )

    def test_nonterminal_gathering_loses_to_military_pressure_when_ahead(self) -> None:
        env = make_env(width=14, height=14)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(1, 2), abundance=60)]
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (1, 2))
        env._spawn_unit("Red", "soldier", 10, (8, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (9, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (10, 10))
        env._spawn_unit("Blue", "worker", 5, (11, 10))
        env.bank["Red"] = 12
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        gather_action = next(action for action in ctx.legal if action[0] == "gather")
        pressure_action = next(
            action
            for action in ctx.legal
            if action[0] == "move_towards"
            and any(unit.id == action[1] and unit.attack_damage > 0 for unit in ctx.military)
            and hexgrid.distance(action[2], env.bases["Blue"].position) <= 3
        )
        helpers = ScoringHelpers(
            distance=hexgrid.distance,
            enemy_pressure_near_base=lambda _env, _faction: False,
            enemy_can_finish_base_next_turn=lambda _env, _faction: False,
            hold_defender_reserve=lambda _env, _ctx, _spend: False,
            unit_count=lambda _env, faction, unit_type: sum(
                1 for unit in _env.units if unit.faction == faction and unit.unit_type == unit_type
            ),
        )
        push_intent = StrategicIntent("push", "safe army advantage", "pressure frontline", "medium")

        self.assertIsNone(env.config.target_bank)
        self.assertGreater(
            utility_modifier(env, ctx, pressure_action, push_intent, helpers),
            utility_modifier(env, ctx, gather_action, push_intent, helpers),
        )

    def test_nonterminal_gathering_stays_useful_for_immediate_war_purchase(self) -> None:
        env = make_env(width=14, height=14)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(1, 2), abundance=60)]
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze", "masonry", "fletching", "construction", "iron", "steel", "engineering", "advanced_siege"}
        )
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="siege_workshop", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "worker", 5, (1, 2))
        env._spawn_unit("Red", "soldier", 10, (8, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (9, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (10, 10))
        env.bank["Red"] = 12
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 47

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        gather_action = next(action for action in ctx.legal if action[0] == "gather")
        helpers = ScoringHelpers(
            distance=hexgrid.distance,
            enemy_pressure_near_base=lambda _env, _faction: False,
            enemy_can_finish_base_next_turn=lambda _env, _faction: False,
            hold_defender_reserve=lambda _env, _ctx, _spend: False,
            unit_count=lambda _env, faction, unit_type: sum(
                1 for unit in _env.units if unit.faction == faction and unit.unit_type == unit_type
            ),
        )
        push_intent = StrategicIntent("push", "safe army advantage", "pressure frontline", "medium")

        self.assertIsNone(env.config.target_bank)
        self.assertNotIn(("train", "ballista"), ctx.legal)
        self.assertGreaterEqual(utility_modifier(env, ctx, gather_action, push_intent, helpers), -30)

    def test_fragile_recovery_keeps_lone_defender_near_home(self) -> None:
        env = make_env(width=14, height=14)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "iron", "steel", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Red", "soldier", 10, (7, 6), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 10), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 11), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (11, 10), attack_damage=3, attack_range=4, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (11, 11))
        env.bank["Red"] = 0
        env.bank["Blue"] = 300
        env.declare_war("Red", "Blue")
        red_soldier = next(unit for unit in env.units if unit.faction == "Red" and unit.unit_type == "soldier")

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        self.assertEqual(ctx.recovery_posture, "critical")

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[1], red_soldier.id)
        self.assertLessEqual(hexgrid.distance(action[2], env.bases["Red"].position), 2)
        self.assertIn("recovery_posture:critical", agent.explain_last_decision())

    def test_recovery_fallback_acts_when_ranked_candidates_are_filtered_out(self) -> None:
        env = make_env(width=14, height=14)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "iron", "steel", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Red", "soldier", 10, (7, 6), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 10), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 11), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (11, 10), attack_damage=3, attack_range=4, move_steps=1)
        env.bank["Blue"] = 300
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0

        agent._rank_candidates = lambda _env, _ctx: []

        action = agent.act(env)

        self.assertIsNotNone(action)
        self.assertIn(action, env.legal_actions("Red"))
        self.assertIn("survival_fallback chose", agent.explain_last_decision())

    def test_peace_recovery_fallback_acts_when_ranked_candidates_are_filtered_out(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(2, 2), abundance=60)]
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "iron", "steel", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Blue", "soldier", 14, (10, 10), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 11), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (11, 10), attack_damage=3, attack_range=4, move_steps=1)
        env.bank["Red"] = 100
        env.bank["Blue"] = 300

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        self.assertTrue(ctx.recovery_mode)
        self.assertFalse(ctx.at_war)

        agent._rank_candidates = lambda _env, _ctx: []
        action = agent.act(env)

        self.assertEqual(action, ("train", "soldier"))
        self.assertIn("survival_fallback chose", agent.explain_last_decision())

    def test_critical_recovery_fallback_trains_before_gathering(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(2, 2), abundance=60)]
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "iron", "steel", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Blue", "soldier", 14, (10, 10), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (10, 11), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (11, 10), attack_damage=3, attack_range=4, move_steps=1)
        env.bank["Blue"] = 300
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 100

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        self.assertEqual(ctx.recovery_posture, "critical")
        self.assertIn(("gather", 1), env.legal_actions("Red"))
        self.assertIn(("train", "soldier"), env.legal_actions("Red"))

        agent._rank_candidates = lambda _env, _ctx: []
        action = agent.act(env)

        self.assertEqual(action, ("train", "soldier"))
        self.assertIn("survival_fallback chose", agent.explain_last_decision())

    def test_last_stand_gathers_only_when_it_immediately_buys_defender(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(1, 1), abundance=60)]
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(2, 0))
        env._spawn_unit("Red", "worker", 5, (1, 1))
        env._spawn_unit("Blue", "archer", 8, (3, 2), attack_damage=3, attack_range=3)
        env.bank["Blue"] = 300
        env.declare_war("Blue", "Red")
        env.bank["Red"] = 25

        action = agent.act(env)
        self.assertEqual(action, ("gather", 1))
        self.assertTrue(env.apply_action(action)[0])

        action = agent.act(env)
        self.assertEqual(action, ("train", "soldier"))

    def test_last_stand_worker_evades_instead_of_busywork_gathering(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = [ResourceNode(id=1, position=(1, 1), abundance=60)]
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(2, 0))
        env._spawn_unit("Red", "worker", 5, (1, 1))
        env._spawn_unit("Blue", "archer", 8, (3, 2), attack_damage=3, attack_range=3)
        env.bank["Blue"] = 300
        env.declare_war("Blue", "Red")
        env.bank["Red"] = 0

        action = agent.act(env)

        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertNotEqual(action[2], env.bases["Red"].position)
        self.assertEqual(threat.build_threat_map(env, "Red").danger_at(action[2]), 0)

    def test_debug_snapshot_includes_ai_decision_reasoning(self) -> None:
        env = make_env()
        red_agent = HeuristicAgent(desired_workers=3)
        blue_agent = HeuristicAgent(desired_workers=3)

        red_info, blue_info, _red_actions, _blue_actions = step_full_turn(env, red_agent, blue_agent)
        snapshot = turn_snapshot(env, red_info, blue_info)
        text = build_debug_snapshot(env, "Heuristic", "Heuristic", red_info, blue_info, [snapshot])

        self.assertIn("Red AI decision:", text)
        self.assertIn("Blue AI decision:", text)
        self.assertIn("research:mining", text)
        self.assertIn("Red AI:", text)

    def test_low_support_peace_rebuilds_instead_of_freezing_in_push_mode(self) -> None:
        env = make_env(width=14, height=14, num_resource_nodes=0, target_bank=999)
        agent = HeuristicAgent(desired_workers=3)
        env.turn = 120
        env.resources = []
        env.faction_state("Red").war_support = env.config.war_support_to_declare_min - 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching", "masonry"})
        env.faction_state("Red").tech_in_progress = "masonry"
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "fletching", "masonry", "iron", "steel"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (8, 5), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (9, 5), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "worker", 5, (1, 2))
        env._spawn_unit("Red", "worker", 5, (2, 2))
        env._spawn_unit("Red", "worker", 5, (2, 1))
        env._spawn_unit("Blue", "soldier", 14, (10, 6), attack_damage=5, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 14, (11, 6), attack_damage=5, attack_range=1, move_steps=1)
        env.bank["Red"] = 60
        env.bank["Blue"] = 200

        self.assertEqual(env.relation_state("Red", "Blue").state, "peace")
        self.assertNotIn(("declare_war", "Blue"), env.legal_actions("Red"))
        self.assertFalse(army_plan(env, "Red") == "Push")
        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "train")

    def test_defense_prioritizes_enemy_camping_spawn_ring(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 12
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 4, (4, 3), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_camper = next(u for u in env.units if u.faction == "Blue" and u.position == (2, 1))

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, blue_camper.id))

    def test_base_siege_priority_targets_enemy_already_in_base_attack_range(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (1, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (4, 4), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")

        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        base_sieger = next(u for u in env.units if u.faction == "Blue" and u.position == (1, 2))

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, base_sieger.id))

    def test_defense_mode_blocks_worker_spawn_during_emergency(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=5)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env._spawn_unit("Red", "soldier", 10, (3, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)

        action = agent.act(env)
        self.assertNotEqual(action, ("spawn_worker",))

    def test_collapse_mode_rebuilds_worker_when_pressure_is_not_active_siege(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.bank["Red"] = 20
        env.units = [u for u in env.units if u.faction != "Red" or u.unit_type != "worker"]
        env.faction_state("Red").unit_ids = {u.id for u in env.units if u.faction == "Red"}
        env._spawn_unit("Blue", "soldier", 10, (6, 6), attack_damage=3, attack_range=1, move_steps=1)

        action = agent.act(env)
        self.assertEqual(action, ("spawn_worker",))

    def test_collapse_mode_trains_defender_before_worker_in_last_stand(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env.units = [u for u in env.units if u.faction != "Red" or u.unit_type != "worker"]
        env.faction_state("Red").unit_ids = {u.id for u in env.units if u.faction == "Red"}
        env._spawn_unit("Blue", "horseman", 12, (1, 2), attack_damage=4, attack_range=1, move_steps=3)

        action = agent.act(env)
        self.assertEqual(action, ("train", "archer"))

    def test_collapse_mode_does_not_spawn_worker_into_active_siege(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining"})
        env.bank["Red"] = 20
        env.units = [u for u in env.units if u.faction != "Red" or u.unit_type != "worker"]
        env.faction_state("Red").unit_ids = {u.id for u in env.units if u.faction == "Red"}
        env._spawn_unit("Blue", "horseman", 12, (2, 1), attack_damage=4, attack_range=1, move_steps=3)

        action = agent.act(env)
        self.assertNotEqual(action, ("spawn_worker",))

    def test_rebuild_mode_trains_before_sending_lone_unit_out(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env._spawn_unit("Red", "soldier", 10, (5, 5), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (3, 1), attack_damage=3, attack_range=1, move_steps=1)

        action = agent.act(env)
        self.assertEqual(action, ("train", "soldier"))

    def test_rebuild_mode_keeps_lone_defender_near_home_when_training_unavailable(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 10, (5, 5), attack_damage=3, attack_range=1, move_steps=1)
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (3, 1), attack_damage=3, attack_range=1, move_steps=1)

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[1], red_soldier.id)
        self.assertNotEqual(action[2], env.bases["Blue"].position)

    def test_damaged_unit_prefers_recovering_over_low_value_attack(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 3, (5, 5), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (6, 5), attack_damage=3, attack_range=1, move_steps=1)
        damaged = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier" and u.position == (5, 5))

        action = agent.act(env)
        self.assertEqual(action, ("move_towards", damaged.id, env.bases["Red"].position))

    def test_worker_retreats_toward_base_in_defense_mode(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.bank["Red"] = 0
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        worker.position = (6, 6)
        env._spawn_unit("Blue", "horseman", 12, (5, 6), attack_damage=4, attack_range=1, move_steps=3)

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[1], worker.id)
        self.assertLess(
            abs(action[2][0] - env.bases["Red"].position[0]) + abs(action[2][1] - env.bases["Red"].position[1]),
            abs(worker.position[0] - env.bases["Red"].position[0]) + abs(worker.position[1] - env.bases["Red"].position[1]),
        )

    def test_defense_mode_prefers_archer_when_horseman_threatens_base(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env._spawn_unit("Blue", "horseman", 12, (2, 1), attack_damage=4, attack_range=1, move_steps=3)

        action = agent.act(env)
        self.assertEqual(action, ("train", "archer"))

    def test_agent_prefers_attacking_horseman_over_soldier(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching", "masonry", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 12
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (5, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "horseman", 12, (4, 5), attack_damage=4, attack_range=1, move_steps=3)
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_horseman = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "horseman")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, blue_horseman.id))

    def test_horseman_prefers_raiding_worker_targets(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "horseman", 12, (4, 4), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Blue", "worker", 5, (8, 4))
        env._spawn_unit("Blue", "soldier", 10, (11, 11), attack_damage=3, attack_range=1, move_steps=1)
        horseman = next(u for u in env.units if u.faction == "Red" and u.unit_type == "horseman")

        action = agent.act(env)
        self.assertEqual(action, ("move_towards", horseman.id, (8, 4)))

    def test_horseman_avoids_screened_worker_raid_when_enemy_support_overwhelms(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "horseman", 12, (4, 4), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Blue", "worker", 5, (5, 4))
        env._spawn_unit("Blue", "archer", 8, (6, 5), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (6, 4), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        horseman = next(u for u in env.units if u.faction == "Red" and u.unit_type == "horseman")
        worker = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "worker")

        action = agent.act(env)
        self.assertNotEqual(action, ("attack", horseman.id, worker.id))

    def test_horseman_defends_home_before_raiding_when_enemy_pressure_exists(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "horseman", 12, (4, 4), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Blue", "worker", 5, (8, 4))
        env._spawn_unit("Blue", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        horseman = next(u for u in env.units if u.faction == "Red" and u.unit_type == "horseman")

        action = agent.act(env)
        self.assertEqual(action, ("move_towards", horseman.id, (2, 2)))

    def test_archer_disengages_toward_base_when_cavalry_closes_in(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "archer", 8, (4, 4), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "horseman", 12, (5, 4), attack_damage=4, attack_range=1, move_steps=2)
        archer = next(u for u in env.units if u.faction == "Red" and u.unit_type == "archer")

        action = agent.act(env)
        self.assertEqual(action, ("move_towards", archer.id, env.bases["Red"].position))

    def test_agent_avoids_suicidal_solo_base_attack(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "masonry"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 4, (10, 11), attack_damage=3, attack_range=1, move_steps=1)
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertNotEqual(action, ("attack_base", red_soldier.id, "Blue"))

    def test_agent_allows_supported_siege_unit_to_chip_base(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = env.config.base_peace_until_turn
        env.resources = []
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "ballista", 10, (8, 9), attack_damage=5, attack_range=4, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (8, 10), attack_damage=3, attack_range=1, move_steps=1)
        env.bank["Red"] = 12
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        ballista = next(u for u in env.units if u.faction == "Red" and u.unit_type == "ballista")

        self.assertGreater(env.bases["Blue"].hp, ballista.attack_damage)
        self.assertEqual(agent.act(env), ("attack_base", ballista.id, "Blue"))

    def test_agent_trains_line_reinforcement_before_horseman_when_behind_ranged_enemy(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 100
        env.bank["Blue"] = 100
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze", "masonry", "animal_husbandry", "horseback_riding", "fletching"}
        )
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "fletching", "iron"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (1, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (2, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "archer", 8, (2, 4), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (8, 9), attack_damage=4, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (9, 9), attack_damage=4, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (9, 10), attack_damage=3, attack_range=4, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (10, 10), attack_damage=3, attack_range=4, move_steps=1)
        env.declare_war("Red", "Blue")

        action = agent.act(env)
        self.assertIn(action, {("train", "soldier"), ("train", "archer")})

    def test_agent_allows_base_attack_when_it_can_finish_base(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = env.config.base_peace_until_turn
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 12
        env.bases["Blue"].hp = 3
        env._spawn_unit("Red", "soldier", 10, (10, 11), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack_base", red_soldier.id, "Blue"))

    def test_agent_does_not_offer_peace_when_enemy_has_lethal_base_attack(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = 20
        env.resources = []
        env.bank["Red"] = 22
        env.bank["Blue"] = 0
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env.bases["Red"].hp = 8
        env.declare_war("Red", "Blue")
        env.relation_state("Red", "Blue").since_turn = 0

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertNotEqual(action[0], "offer_peace")

    def test_agent_does_not_offer_peace_when_enemy_is_adjacent_to_base(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = 20
        env.resources = []
        env.bank["Red"] = 22
        env.bank["Blue"] = 200
        env.faction_state("Red").war_support = 30
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (1, 2), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        env.relation_state("Red", "Blue").since_turn = 0

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertNotEqual(action[0], "offer_peace")

    def test_agent_does_not_accept_peace_when_enemy_is_adjacent_to_base(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = 20
        env.resources = []
        env.bank["Red"] = 12
        env.bank["Blue"] = 200
        env.faction_state("Red").war_support = 30
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (1, 2), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        env.bank["Red"] = 0
        env.relation_state("Red", "Blue").since_turn = 0
        self.assertTrue(env.offer_peace("Blue", "Red", 0))

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertNotEqual(action[0], "accept_peace")

    def test_agent_prefers_non_idle_action_under_pressure_when_available(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = 20
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (3, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")

        action = agent.act(env)
        self.assertIsNotNone(action)

    def test_agent_rallies_line_units_before_base_push(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (9, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (10, 11), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (11, 10), attack_damage=3, attack_range=1, move_steps=1)
        rear_soldier = next(
            u for u in env.units if u.faction == "Red" and u.unit_type == "soldier" and u.position == (2, 2)
        )
        env.declare_war("Red", "Blue")

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertNotEqual(action[2], env.bases["Blue"].position)

    def test_rallied_line_unit_does_not_immediately_retreat_home(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (5, 6), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (3, 7), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "horseman", 12, (5, 5), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Blue", "soldier", 10, (9, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 8, (10, 10), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "archer", 8, (10, 9), attack_damage=3, attack_range=3, move_steps=1)
        env.declare_war("Red", "Blue")
        rallied = next(
            unit for unit in env.units if unit.faction == "Red" and unit.unit_type == "soldier" and unit.position == (5, 6)
        )

        action = agent.act(env)
        self.assertNotEqual(action, ("move_towards", rallied.id, env.bases["Red"].position))
        self.assertNotEqual(action, ("move_towards", rallied.id, (2, 2)))

    def test_agent_stages_reinforcements_when_frontline_is_losing_locally(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (8, 8), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "horseman", 12, (7, 7), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Red", "horseman", 12, (6, 7), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Blue", "soldier", 10, (9, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 8, (10, 10), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "archer", 8, (10, 9), attack_damage=3, attack_range=3, move_steps=1)
        env.declare_war("Red", "Blue")
        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        acting_unit = next(u for u in env.units if u.id == action[1])
        if acting_unit.unit_type != "horseman":
            self.assertNotEqual(action[2], (9, 10))

    def test_staged_army_commits_when_global_push_window_is_favorable(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze", "fletching"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (6, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (6, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (7, 9), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (7, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 8, (9, 10), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "archer", 8, (10, 10), attack_damage=3, attack_range=3, move_steps=1)
        env.declare_war("Red", "Blue")

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        action = agent._choose_military_movement(env, ctx)

        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[2], (9, 10))

    def test_agent_breaks_out_of_stalled_military_target_loop(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (3, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (8, 8), attack_damage=3, attack_range=1, move_steps=1)
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        stale_target = (3, 1)
        agent._military_targets[red_soldier.id] = stale_target
        agent._military_last_positions[red_soldier.id] = red_soldier.position
        agent._military_stall[red_soldier.id] = 3

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[1], red_soldier.id)
        self.assertNotEqual(action[2], stale_target)

    def test_agent_keeps_committed_military_target_for_short_window(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (3, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (10, 12), attack_damage=3, attack_range=1, move_steps=1)

        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        target = (10, 12)
        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        agent._set_military_target(red_soldier.id, target, lock=2)

        action = agent._choose_military_movement(env, ctx)

        self.assertEqual(action, ("move_towards", red_soldier.id, target))

    def test_agent_enters_push_mode_with_large_safe_army_lead(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (3, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (4, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (5, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (6, 2), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")

        action = agent.act(env)
        self.assertEqual(action[0], "move_towards")
        self.assertLessEqual(abs(action[2][0] - env.bases["Blue"].position[0]) + abs(action[2][1] - env.bases["Blue"].position[1]), 2)

    def test_agent_pushes_with_two_soldier_safe_advantage(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (3, 2), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")

        action = agent.act(env)
        self.assertEqual(action[0], "move_towards")
        self.assertLessEqual(abs(action[2][0] - env.bases["Blue"].position[0]) + abs(action[2][1] - env.bases["Blue"].position[1]), 2)

    def test_push_mode_targets_enemy_frontline_before_enemy_base(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 0
        env.bank["Blue"] = 0
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (5, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (8, 7), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")

        action = agent.act(env)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[2], (8, 7))

    def test_push_mode_does_not_spend_turn_gathering_when_military_can_advance(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=3)
        env.bank["Red"] = 250
        env.bank["Blue"] = 0
        env.resources = [ResourceNode(id=1, position=(2, 1), abundance=60)]
        env.units = [u for u in env.units if u.faction == "Red" and u.unit_type == "worker"]
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = max((u.id for u in env.units), default=0) + 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze", "animal_husbandry", "horseback_riding"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (3, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (4, 3), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "horseman", 12, (5, 3), attack_damage=4, attack_range=1, move_steps=2)
        env._spawn_unit("Blue", "soldier", 10, (8, 7), attack_damage=3, attack_range=1, move_steps=1)

        ctx = agent._build_context(env, "Red", env.legal_actions("Red"))
        action = agent._choose_economy_action(env, ctx)

        self.assertIsNone(action)

    def test_army_plan_shows_advance_for_lone_uncontested_attacker(self) -> None:
        env = make_env(target_bank=999)
        env.resources = []
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (8, 8), attack_damage=3, attack_range=1, move_steps=1)

        self.assertEqual(army_plan(env, "Red"), "Advance")

    def test_army_plan_shows_siege_when_enemy_is_broken(self) -> None:
        env = make_env(target_bank=999, collapse_enabled=False)
        env.units = [u for u in env.units if u.faction == "Red" and u.unit_type == "worker"]
        env.faction_states["Blue"].unit_ids.clear()
        env._spawn_unit("Red", "soldier", 10, (8, 8), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (7, 8), attack_damage=3, attack_range=1, move_steps=1)

        self.assertEqual(army_plan(env, "Red"), "Siege")

    def test_agent_moves_to_siege_tile_when_enemy_is_broken(self) -> None:
        env = make_env(target_bank=999, collapse_enabled=False)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.units = [u for u in env.units if u.faction != "Blue"]
        env.faction_states["Blue"].unit_ids.clear()
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (8, 8), attack_damage=3, attack_range=1, move_steps=1)

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertNotEqual(action[2], env.bases["Blue"].position)
        self.assertLessEqual(abs(action[2][0] - env.bases["Blue"].position[0]) + abs(action[2][1] - env.bases["Blue"].position[1]), 2)

    def test_siege_units_claim_different_base_approach_tiles(self) -> None:
        env = make_env(target_bank=999, collapse_enabled=False)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.units = [u for u in env.units if u.faction != "Blue"]
        env.faction_states["Blue"].unit_ids.clear()
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_unit("Red", "soldier", 10, (8, 8), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (7, 8), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")

        first = agent.act(env)
        self.assertIsNotNone(first)
        self.assertEqual(first[0], "move_towards")
        self.assertTrue(env.apply_action(first)[0])

        second = agent.act(env)
        self.assertIsNotNone(second)
        self.assertEqual(second[0], "move_towards")
        self.assertNotEqual(first[2], second[2])


class CombatAndVictoryTests(unittest.TestCase):
    def test_diplomacy_starts_in_peace_and_exposes_declare_war(self) -> None:
        env = make_env(target_bank=999)
        self.assertEqual(env.relation_state("Red", "Blue").state, "peace")
        self.assertIn(("declare_war", "Blue"), env.legal_actions("Red"))

    def test_declare_war_spends_resources_and_war_support(self) -> None:
        env = make_env(target_bank=999)
        red_bank_before = env.bank["Red"]
        red_support_before = env.faction_state("Red").war_support

        self.assertTrue(env.declare_war("Red", "Blue"))

        self.assertEqual(env.bank["Red"], red_bank_before - env.config.war_declaration_cost)
        self.assertEqual(
            env.faction_state("Red").war_support,
            red_support_before - env.config.war_declaration_support_penalty,
        )
        self.assertEqual(env.relation_state("Red", "Blue").aggressor, "Red")

    def test_declare_war_requires_minimum_war_support(self) -> None:
        env = make_env(target_bank=999)
        env.faction_state("Red").war_support = env.config.war_support_to_declare_min - 1

        self.assertFalse(env.can_declare_war("Red", "Blue"))
        self.assertFalse(env.declare_war("Red", "Blue"))

    def test_declare_war_enables_attacks(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))

        attacker = next(u for u in env.units if u.faction == "Red")
        target = next(u for u in env.units if u.faction == "Blue")
        self.assertNotIn(("attack", attacker.id, target.id), env.legal_actions("Red"))
        self.assertTrue(env.declare_war("Red", "Blue"))
        self.assertIn(("attack", attacker.id, target.id), env.legal_actions("Red"))
        self.assertTrue(combat.attack(env, "Red", attacker.id, target.id))

    def test_accepting_peace_creates_truce_and_transfers_indemnity(self) -> None:
        env = make_env(target_bank=999)
        env.turn = 12
        self.assertTrue(env.declare_war("Red", "Blue"))
        env.turn = 18
        self.assertTrue(env.offer_peace("Red", "Blue", 20))
        red_before = env.bank["Red"]
        blue_before = env.bank["Blue"]

        indemnity = env.accept_peace("Blue", "Red")

        self.assertEqual(indemnity, 20)
        self.assertEqual(env.relation_state("Red", "Blue").state, "truce")
        self.assertEqual(env.bank["Red"], red_before - 20)
        self.assertEqual(env.bank["Blue"], blue_before + 20)

    def test_legal_peace_offer_scales_indemnity_with_war_outcome(self) -> None:
        env = make_env(target_bank=999)
        env.turn = 12
        self.assertTrue(env.declare_war("Red", "Blue"))
        relation = env.relation_state("Red", "Blue")
        env.turn = 36
        env.bank["Red"] = 300
        env.bases["Red"].hp = 12
        relation.war_score = {"Red": 2, "Blue": 20}

        offer = next(action for action in env.legal_actions("Red") if action[0] == "offer_peace")

        self.assertEqual(offer[1], "Blue")
        self.assertGreater(offer[2], env.config.peace_indemnity_base)
        self.assertLessEqual(offer[2], env.bank["Red"] * env.config.peace_indemnity_max_bank_pct // 100)

    def test_scaled_peace_indemnity_respects_bank_cap(self) -> None:
        env = make_env(target_bank=999)
        env.turn = 12
        self.assertTrue(env.declare_war("Red", "Blue"))
        relation = env.relation_state("Red", "Blue")
        env.turn = 48
        env.bank["Red"] = 50
        env.bases["Red"].hp = 1
        relation.war_score = {"Red": 0, "Blue": 80}

        offer = next(action for action in env.legal_actions("Red") if action[0] == "offer_peace")

        self.assertEqual(offer[2], env.bank["Red"] * env.config.peace_indemnity_max_bank_pct // 100)

    def test_losing_peace_applies_reparations_and_war_support_cap(self) -> None:
        env = make_env(target_bank=999)
        env.turn = 12
        self.assertTrue(env.declare_war("Red", "Blue"))
        relation = env.relation_state("Red", "Blue")
        env.turn = 18
        env.bank["Red"] = 100
        env.faction_state("Red").war_support = 90
        env.faction_state("Red").techs_unlocked.add("mining")
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env.bases["Red"].hp = 12
        relation.war_score = {"Red": 2, "Blue": 22}
        self.assertTrue(env.offer_peace("Red", "Blue", 30))

        self.assertEqual(env.accept_peace("Blue", "Red"), 30)

        concessions = env.relation_state("Red", "Blue").concessions
        self.assertIsNotNone(concessions)
        self.assertEqual(concessions.payer, "Red")
        self.assertEqual(concessions.receiver, "Blue")
        self.assertGreater(concessions.reparations_per_turn, 0)
        self.assertEqual(env.faction_state("Red").war_support, env.config.concession_war_support_cap)

        red_before = env.bank["Red"]
        blue_before = env.bank["Blue"]
        env.step_end_turn()

        self.assertEqual(env.bank["Red"], red_before - concessions.reparations_per_turn + env._passive_income_for("Red"))
        self.assertEqual(env.bank["Blue"], blue_before + concessions.reparations_per_turn)
        self.assertIn("paid", " | ".join(env.current_events + env.recent_events))

    def test_agent_accepts_surrender_when_offer_contains_concessions(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.turn = 12
        self.assertTrue(env.declare_war("Blue", "Red"))
        relation = env.relation_state("Red", "Blue")
        env.turn = 20
        env.bases["Blue"].hp = 12
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1)
        relation.war_score = {"Red": 24, "Blue": 4}
        self.assertTrue(env.offer_peace("Blue", "Red", 40))

        action = agent.act(env)

        self.assertEqual(action, ("accept_peace", "Blue"))

    def test_agent_refuses_peace_when_winning_and_base_is_reachable(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.turn = 24
        env.bank["Red"] = 120
        env.bank["Blue"] = 120
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._spawn_unit("Red", "soldier", 10, (10, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "archer", 8, (9, 10), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (10, 9), attack_damage=3, attack_range=1, move_steps=1)
        self.assertTrue(env.declare_war("Red", "Blue"))
        env.bank["Red"] = 0
        relation = env.relation_state("Red", "Blue")
        relation.since_turn = 0
        relation.war_score = {"Red": 24, "Blue": 2}
        env.bases["Blue"].hp = 14
        self.assertTrue(env.offer_peace("Blue", "Red", 50))

        action = agent.act(env)

        self.assertIsNotNone(action)
        self.assertNotEqual(action[0], "accept_peace")
        self.assertIn(action[0], {"attack_base", "move_towards"})

    def test_agent_can_accept_peace_when_winning_but_war_support_is_exhausted(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.turn = 24
        env.bank["Red"] = 120
        env.bank["Blue"] = 120
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._spawn_unit("Red", "soldier", 10, (10, 10), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "archer", 8, (9, 10), attack_damage=3, attack_range=3, move_steps=1)
        self.assertTrue(env.declare_war("Red", "Blue"))
        env.bank["Red"] = 0
        env.faction_state("Red").war_support = 10
        relation = env.relation_state("Red", "Blue")
        relation.since_turn = 0
        relation.war_score = {"Red": 24, "Blue": 2}
        env.bases["Blue"].hp = 14
        self.assertTrue(env.offer_peace("Blue", "Red", 50))

        action = agent.act(env)

        self.assertEqual(action, ("accept_peace", "Blue"))

    def test_failed_aggressor_peace_applies_support_penalty_and_relief(self) -> None:
        env = make_env(target_bank=999)
        env.turn = 12
        env.faction_state("Blue").war_support = 80
        self.assertTrue(env.declare_war("Red", "Blue"))
        env.turn = 18
        self.assertTrue(env.offer_peace("Red", "Blue", 20))

        indemnity = env.accept_peace("Blue", "Red")

        self.assertEqual(indemnity, 20)
        self.assertEqual(env.faction_state("Red").war_support, 63)
        self.assertEqual(env.faction_state("Blue").war_support, 89)
        self.assertEqual(env.relation_state("Red", "Blue").aggressor, None)
        self.assertEqual(env.relation_state("Red", "Blue").failed_aggressor, "Red")
        self.assertEqual(env.relation_state("Red", "Blue").truce_until_turn, env.turn + env.config.truce_turns)

    def test_failed_aggressor_does_not_recover_war_support_during_truce(self) -> None:
        env = make_env(target_bank=999)
        env.turn = 12
        env.faction_state("Blue").war_support = 80
        self.assertTrue(env.declare_war("Red", "Blue"))
        env.turn = 18
        self.assertTrue(env.offer_peace("Red", "Blue", 20))
        self.assertEqual(env.accept_peace("Blue", "Red"), 20)
        red_support = env.faction_state("Red").war_support
        blue_support = env.faction_state("Blue").war_support

        env.step_end_turn()
        env.step_end_turn()

        self.assertEqual(env.faction_state("Red").war_support, red_support)
        self.assertEqual(env.faction_state("Blue").war_support, blue_support + env.config.peace_support_recovery_per_turn)

    def test_war_upkeep_and_support_drain_apply_each_turn(self) -> None:
        env = make_env(target_bank=999)
        self.assertTrue(env.declare_war("Red", "Blue"))
        red_bank_before = env.bank["Red"]
        red_support_before = env.faction_state("Red").war_support

        env.step_end_turn()

        self.assertEqual(
            env.bank["Red"],
            red_bank_before - env.config.war_upkeep_per_turn - env.config.war_upkeep_aggressor_bonus,
        )
        self.assertEqual(env.faction_state("Red").war_support, red_support_before - env.config.war_support_drain_per_turn)
        self.assertIn("Red paid 3 war upkeep against Blue", " | ".join(env.current_events + env.recent_events))

    def test_war_support_recovers_during_peace(self) -> None:
        env = make_env(target_bank=999)
        env.faction_state("Red").war_support = 9

        env.step_end_turn()

        self.assertEqual(env.faction_state("Red").war_support, 9 + env.config.peace_support_recovery_per_turn)
        self.assertIn("Red war support recovered", " | ".join(env.current_events + env.recent_events))

    def test_worker_attacks_are_blocked_during_peace_window(self) -> None:
        env = make_env(target_bank=999)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))

        attacker = next(u for u in env.units if u.faction == "Red")
        target = next(u for u in env.units if u.faction == "Blue")
        self.assertFalse(combat.attack(env, "Red", attacker.id, target.id))
        self.assertNotIn(("attack", attacker.id, target.id), env.legal_actions("Red"))

    def test_base_attacks_are_blocked_during_peace_window(self) -> None:
        env = make_env(target_bank=999)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (10, 9), attack_damage=3, attack_range=1)

        attacker = next(u for u in env.units if u.faction == "Red")
        self.assertFalse(combat.attack_base(env, "Red", attacker.id, "Blue"))
        self.assertNotIn(("attack_base", attacker.id, "Blue"), env.legal_actions("Red"))

    def test_worker_attacks_open_up_after_peace_window(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))

        attacker = next(u for u in env.units if u.faction == "Red")
        target = next(u for u in env.units if u.faction == "Blue")
        self.assertTrue(env.declare_war("Red", "Blue"))
        self.assertTrue(combat.attack(env, "Red", attacker.id, target.id))

    def test_units_recover_hp_at_end_of_turn_when_idle(self) -> None:
        env = make_env(target_bank=999)
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 6, (2, 2), attack_damage=3, attack_range=1)
        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        env.start_faction_turn()
        env.step_end_turn()

        self.assertEqual(soldier.hp, 8)

    def test_units_do_not_recover_if_they_attacked_this_turn(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 6, (4, 4), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "soldier", 10, (5, 4), attack_damage=3, attack_range=1)
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_soldier = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")
        env.declare_war("Red", "Blue")

        env.start_faction_turn()
        self.assertTrue(env.apply_action(("attack", red_soldier.id, blue_soldier.id))[0])
        env.step_end_turn()

        self.assertEqual(red_soldier.hp, 6)

    def test_military_skirmishes_are_still_allowed_during_peace_window(self) -> None:
        env = make_env(target_bank=999)
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "soldier", 10, (5, 4), attack_damage=3, attack_range=1)

        attacker = next(u for u in env.units if u.faction == "Red")
        target = next(u for u in env.units if u.faction == "Blue")
        self.assertTrue(combat.attack(env, "Red", attacker.id, target.id))

    def test_base_attack_can_end_the_game(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(10, 9),
            attack_damage=15,
            attack_range=1,
            move_steps=1,
        )

        attacker = next(
            u for u in env.units if u.faction == "Red" and u.unit_type == "soldier" and u.position == (10, 9)
        )
        self.assertTrue(combat.attack_base(env, "Red", attacker.id, "Blue"))
        self.assertTrue(combat.attack_base(env, "Red", attacker.id, "Blue"))
        self.assertEqual(env.bases["Blue"].hp, 0)
        self.assertEqual(env.winner(), "Red")

    def test_base_attack_clamps_hp_and_stops_followup_attacks(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(10, 9),
            attack_damage=50,
            attack_range=1,
            move_steps=1,
        )

        attacker = next(
            u for u in env.units if u.faction == "Red" and u.unit_type == "soldier" and u.position == (10, 9)
        )
        self.assertTrue(combat.attack_base(env, "Red", attacker.id, "Blue"))
        self.assertEqual(env.bases["Blue"].hp, 0)
        self.assertFalse(combat.attack_base(env, "Red", attacker.id, "Blue"))

    def test_single_basic_unit_cannot_solo_healthy_base(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (10, 9), attack_damage=3, attack_range=1, move_steps=1)
        attacker = next(u for u in env.units if u.faction == "Red")

        while combat.attack_base(env, "Red", attacker.id, "Blue"):
            if env.get_unit(attacker.id) is None:
                break

        self.assertIsNone(env.get_unit(attacker.id))
        self.assertGreater(env.bases["Blue"].hp, 0)

    def test_single_archer_cannot_destroy_base_for_free(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "archer", 8, (7, 10), attack_damage=3, attack_range=3, move_steps=1)
        attacker = next(u for u in env.units if u.faction == "Red")

        self.assertEqual(hexgrid.distance(attacker.position, env.bases["Blue"].position), 3)
        while combat.attack_base(env, "Red", attacker.id, "Blue"):
            if env.get_unit(attacker.id) is None:
                break

        self.assertIsNone(env.get_unit(attacker.id))
        self.assertGreater(env.bases["Blue"].hp, 0)

    def test_siege_damages_base_better_than_archer_but_takes_attrition(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "archer", 8, (7, 10), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Red", "ballista", 10, (6, 10), attack_damage=5, attack_range=4, move_steps=1)
        archer = next(u for u in env.units if u.unit_type == "archer")
        ballista = next(u for u in env.units if u.unit_type == "ballista")

        self.assertTrue(combat.attack_base(env, "Red", archer.id, "Blue"))
        hp_after_archer = env.bases["Blue"].hp
        env.bases["Blue"].hp = env.config.base_hp
        self.assertTrue(combat.attack_base(env, "Red", ballista.id, "Blue"))

        self.assertLess(env.bases["Blue"].hp, hp_after_archer)
        self.assertEqual(ballista.hp, 10 - env.config.base_siege_attrition_damage)

    def test_step_faction_stops_when_winner_is_found(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (10, 9), attack_damage=15, attack_range=1)

        attacker = next(u for u in env.units if u.faction == "Red")

        def decide(_env):
            return ("attack_base", attacker.id, "Blue")

        log = env.step_faction(decide)
        self.assertEqual(env.bases["Blue"].hp, 0)
        self.assertEqual(env.winner(), "Red")
        self.assertEqual(log, ["attack_base", "attack_base", "turn_end:winner"])

    def test_step_end_turn_does_not_advance_after_winner(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (10, 9), attack_damage=50, attack_range=1)

        attacker = next(u for u in env.units if u.faction == "Red")
        env.step_faction(lambda _env: ("attack_base", attacker.id, "Blue"))
        self.assertEqual(env.winner(), "Red")

        env.step_end_turn()

        self.assertEqual(env.current_player, 0)
        self.assertEqual(env.factions[env.current_player], "Red")

    def test_turn_snapshot_uses_round_start_number_after_red_wins(self) -> None:
        env = make_env(target_bank=999)
        env.turn = env.config.base_peace_until_turn
        env.declare_war("Red", "Blue")
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (10, 9), attack_damage=50, attack_range=1)
        attacker = next(u for u in env.units if u.faction == "Red")

        class AttackAgent:
            def act(self, _env):
                return ("attack_base", attacker.id, "Blue")

        red_info, blue_info, _red_actions, _blue_actions = step_full_turn(env, AttackAgent(), AttackAgent())
        snapshot = turn_snapshot(env, red_info, blue_info)

        self.assertEqual(snapshot.turn_number, env.config.base_peace_until_turn + 1)
        self.assertEqual(blue_info.log, ["turn_skipped:winner"])

    def test_collapse_rule_ends_game_when_faction_cannot_recover(self) -> None:
        env = make_env(target_bank=999)
        env.units = [u for u in env.units if u.faction != "Blue"]
        env.faction_states["Blue"].unit_ids.clear()
        env.bank["Blue"] = 0

        self.assertEqual(env.winner(), "Red")

    def test_game_can_continue_without_collapse_rule(self) -> None:
        env = make_env(target_bank=999, collapse_enabled=False)
        env.units = [u for u in env.units if u.faction != "Blue"]
        env.faction_states["Blue"].unit_ids.clear()
        env.bank["Blue"] = 0

        self.assertIsNone(env.winner())

    def test_combat_records_kill_events(self) -> None:
        env = make_env()
        env.turn = env.config.worker_peace_until_turn
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=5, attack_range=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))
        env.declare_war("Red", "Blue")

        attacker = next(u for u in env.units if u.faction == "Red")
        ok, reason = env.apply_action(("attack", attacker.id, 2))
        self.assertTrue(ok)
        self.assertEqual(reason, "attack")
        self.assertIn("defeated Blue worker#2", " | ".join(env.current_events))

    def test_base_retaliation_hits_enemy_in_range(self) -> None:
        env = make_env()
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)

        env.step_end_turn()

        attacker = next(u for u in env.units if u.faction == "Blue")
        self.assertEqual(attacker.hp, 8)
        self.assertIn("Red base hit Blue soldier#1", " | ".join(env.recent_events))

    def test_archer_tower_retaliation_can_finish_enemy(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry", "construction"})
        stone_worker, _, quarry_pos = ProductionSystemTests()._stone_build_setup(env)
        self.assertTrue(production.build(env, "Red", stone_worker.id, "quarry", quarry_pos))
        tower_worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        tower_worker.position = (quarry_pos[0] + 1, quarry_pos[1])
        tower_pos = next(
            pos
            for pos in (
                (tower_worker.position[0] + 1, tower_worker.position[1]),
                (tower_worker.position[0] - 1, tower_worker.position[1]),
                (tower_worker.position[0], tower_worker.position[1] + 1),
                (tower_worker.position[0], tower_worker.position[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos not in {b.position for b in env.buildings}
        )
        self.assertTrue(production.build(env, "Red", tower_worker.id, "archer_tower", tower_pos))
        enemy_pos = (min(env.config.width - 1, tower_pos[0] + 1), tower_pos[1])
        if enemy_pos == tower_pos or enemy_pos in env._occupied_positions():
            enemy_pos = (max(0, tower_pos[0] - 1), tower_pos[1])
        env._spawn_unit("Blue", "soldier", 2, enemy_pos, attack_damage=3, attack_range=1, move_steps=1)

        env.step_end_turn()

        self.assertFalse(any(u.faction == "Blue" and u.unit_type == "soldier" for u in env.units))
        self.assertIn("Red archer_tower shot down Blue soldier#", " | ".join(env.recent_events))

    def test_legal_actions_include_military_movement_toward_enemy_base(self) -> None:
        env = make_env()
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(4, 4),
            attack_damage=3,
            attack_range=1,
            move_steps=1,
        )

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertIn(("move_towards", soldier.id, env.bases["Blue"].position), env.legal_actions("Red"))

    def test_legal_actions_include_military_rally_moves_toward_allies(self) -> None:
        env = make_env()
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(3, 3),
            attack_damage=3,
            attack_range=1,
            move_steps=1,
        )
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(8, 8),
            attack_damage=3,
            attack_range=1,
            move_steps=1,
        )

        soldier = next(
            u for u in env.units if u.faction == "Red" and u.unit_type == "soldier" and u.position == (3, 3)
        )
        self.assertIn(("move_towards", soldier.id, (8, 8)), env.legal_actions("Red"))

    def test_workers_keep_fallback_movement_when_no_resources_exist(self) -> None:
        env = make_env()
        env.resources = []

        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        legal = env.legal_actions("Red")

        self.assertIn(("move_towards", worker.id, env.bases["Blue"].position), legal)


class EnvironmentRefactorTests(unittest.TestCase):
    def test_unit_lookup_helpers_stay_in_sync_when_units_are_removed(self) -> None:
        env = make_env()
        red_worker = env.get_units_for_faction("Red")[0]
        blue_worker = env.get_units_for_faction("Blue")[0]

        self.assertIs(env.get_unit(red_worker.id), red_worker)
        self.assertEqual([unit.id for unit in env.get_enemy_units("Red")], [blue_worker.id])

        env._remove_unit(red_worker.id)

        self.assertIsNone(env.get_unit(red_worker.id))
        self.assertEqual(env.get_units_for_faction("Red"), [])

    def test_building_lookup_helpers_return_spawned_buildings(self) -> None:
        env = make_env()

        env._spawn_building("Red", "storehouse", 18, (3, 3))
        building = env.get_buildings_for_faction("Red")[0]

        self.assertIs(env.get_building(building.id), building)
        self.assertEqual(building.position, (3, 3))


class UiAssetScalingTests(unittest.TestCase):
    def test_fit_sprite_to_box_preserves_aspect_and_stays_inside_bounds(self) -> None:
        wide = pygame.Surface((400, 100), pygame.SRCALPHA)
        wide.fill((255, 255, 255, 255))

        fitted = _fit_sprite_to_box(wide, (40, 40))

        self.assertIsNotNone(fitted)
        self.assertLessEqual(fitted.get_width(), 40)
        self.assertLessEqual(fitted.get_height(), 40)
        self.assertEqual(fitted.get_width(), 40)
        self.assertEqual(fitted.get_height(), 10)

    def test_trim_sprite_alpha_uses_visible_pixels_for_consistent_asset_fit(self) -> None:
        padded = pygame.Surface((80, 80), pygame.SRCALPHA)
        padded.fill((0, 0, 0, 0))
        pygame.draw.rect(padded, (255, 255, 255, 255), pygame.Rect(30, 20, 12, 18))

        trimmed = _trim_sprite_alpha(padded)

        self.assertIsNotNone(trimmed)
        self.assertEqual(trimmed.get_size(), (12, 18))


class MovementSystemTests(unittest.TestCase):
    def test_units_can_move_onto_building_tiles(self) -> None:
        env = make_env()
        env.units = []
        env._unit_index = {}
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        env._spawn_building("Red", "storehouse", 18, (3, 1))
        env._spawn_unit("Red", "soldier", 10, (4, 1), attack_damage=3, attack_range=1)

        soldier = env.get_units_for_faction("Red")[0]
        self.assertTrue(movement.can_move_unit(env, soldier.id, "west"))
        self.assertTrue(movement.move_unit(env, soldier.id, "west"))
        self.assertEqual(soldier.position, (3, 1))

    def test_move_towards_paths_through_building_tiles(self) -> None:
        env = make_env()
        env.units = []
        env._unit_index = {}
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        env._spawn_building("Blue", "wall", 34, (2, 1))
        env._spawn_unit("Red", "soldier", 10, (1, 1), attack_damage=3, attack_range=1)

        soldier = env.get_units_for_faction("Red")[0]
        self.assertTrue(movement.can_move_towards(env, soldier.id, (4, 1)))
        self.assertTrue(movement.move_towards(env, soldier.id, (4, 1)))
        self.assertEqual(soldier.position, (2, 1))

    def test_move_towards_routes_around_blocker(self) -> None:
        env = make_env()
        env.units = []
        env.buildings = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        env._spawn_unit("Red", "soldier", 10, (1, 1), attack_damage=3, attack_range=1)
        env._spawn_unit("Red", "worker", 5, (2, 1))

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertTrue(movement.can_move_towards(env, soldier.id, (4, 1)))
        self.assertTrue(movement.move_towards(env, soldier.id, (4, 1)))
        self.assertNotEqual(soldier.position, (2, 1))
        self.assertLessEqual(soldier.position[0], 2)
        self.assertNotEqual(soldier.position, (1, 1))

    def test_move_towards_enemy_unit_stops_on_open_approach_tile(self) -> None:
        env = make_env()
        env.units = []
        env.buildings = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1)
        env._spawn_unit("Blue", "soldier", 10, (5, 4), attack_damage=3, attack_range=1)

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertFalse(movement.can_move_towards(env, soldier.id, (5, 4)))
        self.assertFalse(movement.move_towards(env, soldier.id, (5, 4)))
        self.assertEqual(soldier.position, (4, 4))

    def test_move_towards_base_uses_open_approach_tile_instead_of_base_tile(self) -> None:
        env = make_env()
        env.units = []
        env.buildings = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        env._spawn_unit("Red", "soldier", 10, (8, 10), attack_damage=3, attack_range=1)

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertTrue(movement.can_move_towards(env, soldier.id, env.bases["Blue"].position))
        self.assertTrue(movement.move_towards(env, soldier.id, env.bases["Blue"].position))
        self.assertNotEqual(soldier.position, env.bases["Blue"].position)
        self.assertEqual(soldier.position, (9, 10))

    def test_ranged_unit_moves_to_firing_tile_when_base_approaches_are_blocked(self) -> None:
        env = make_env(width=14, height=14)
        env.units = []
        env.buildings = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        blue_base = env.bases["Blue"].position
        for pos in hexgrid.neighbors(blue_base):
            if env._in_bounds(pos):
                env._spawn_building(faction="Blue", building_type="storehouse", hp=18, pos=pos)
        env._spawn_unit("Red", "ballista", 10, (7, 10), attack_damage=5, attack_range=4)

        ballista = next(u for u in env.units if u.faction == "Red" and u.unit_type == "ballista")
        self.assertTrue(movement.can_move_towards(env, ballista.id, blue_base))
        for _ in range(4):
            self.assertTrue(movement.move_towards(env, ballista.id, blue_base))
            if hexgrid.distance(ballista.position, blue_base) <= ballista.attack_range:
                break
        self.assertLessEqual(hexgrid.distance(ballista.position, blue_base), ballista.attack_range)
        self.assertNotIn(ballista.position, {building.position for building in env.buildings})

    def test_repeated_move_towards_frontline_makes_real_progress(self) -> None:
        env = make_env(width=14, height=14)
        env.units = []
        env.buildings = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1

        env._spawn_unit("Red", "soldier", 12, (3, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (10, 12), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 12, (12, 13), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "archer", 9, (11, 13), attack_damage=3, attack_range=3, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (11, 11))
        env._spawn_unit("Blue", "worker", 5, (12, 11))
        env._spawn_unit("Blue", "worker", 5, (13, 12))

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        target = (10, 12)
        previous_distance = min(
            hexgrid.distance(soldier.position, pos)
            for pos in [target, *hexgrid.neighbors(target)]
            if env._in_bounds(pos)
        )

        moved_positions = {soldier.position}
        for _ in range(6):
            before = soldier.position
            self.assertTrue(movement.can_move_towards(env, soldier.id, target))
            self.assertTrue(movement.move_towards(env, soldier.id, target))
            self.assertNotEqual(soldier.position, before)
            moved_positions.add(soldier.position)

            current_distance = min(
                hexgrid.distance(soldier.position, pos)
                for pos in [target, *hexgrid.neighbors(target)]
                if env._in_bounds(pos)
            )
            self.assertLessEqual(current_distance, previous_distance)
            previous_distance = current_distance

        self.assertGreater(len(moved_positions), 3)

    def test_move_towards_event_records_actual_destination(self) -> None:
        env = make_env(width=14, height=14)
        env.units = []
        env.buildings = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (3, 4), attack_damage=3, attack_range=1, move_steps=1)

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        env.start_faction_turn()

        ok, reason = env.apply_action(("move_towards", soldier.id, (10, 12)))

        self.assertTrue(ok)
        self.assertEqual(reason, "move")
        self.assertIn(f"to {soldier.position}", env.current_events[-1])
        self.assertIn("toward (10, 12)", env.current_events[-1])


if __name__ == "__main__":
    unittest.main()
