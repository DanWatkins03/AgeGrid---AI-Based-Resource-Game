from __future__ import annotations

import unittest

from src.agegrid.agents.greedy import GreedyAgent
from src.agegrid.agents.heuristic import HEURISTIC_PROFILES, HeuristicAgent, army_plan
from src.agegrid.agents.registry import create_agent
from src.agegrid.env.agegrid_env import AgeGridEnv, GameConfig
from src.agegrid.env.entities import ResourceNode
from src.agegrid.env import hexgrid
from src.agegrid.env.systems import combat, movement, production, tech


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
        self.assertFalse(tech.can_research(env, "Red", "bronze_working"))

        self.assertTrue(tech.research(env, "Red", "mining"))
        self.assertEqual(env.faction_state("Red").tech_in_progress, "mining")
        self.assertEqual(env.bank["Red"], 215)
        self.assertFalse(tech.can_research(env, "Red", "bronze_working"))

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

        env.faction_state("Blue").techs_unlocked.add("bronze_working")
        self.assertEqual(env.current_era(), "Bronze Age")

        env.faction_state("Red").techs_unlocked.add("iron_working")
        self.assertEqual(env.current_era(), "Iron Age")

        env.faction_state("Blue").techs_unlocked.add("engineering")
        self.assertEqual(env.current_era(), "Engineering Age")
        self.assertIsNone(env.faction_state("Red").tech_in_progress)
        self.assertTrue(tech.can_research(env, "Red", "bronze_working"))

    def test_research_is_free_once_per_turn_and_preserves_action_points(self) -> None:
        env = make_env()
        env.start_faction_turn()

        ok, reason = env.apply_action(("research", "mining"))
        self.assertTrue(ok)
        self.assertEqual(reason, "research")
        self.assertEqual(env.actions_left, env.config.actions_per_turn)
        self.assertFalse(env.apply_action(("research", "bronze_working"))[0])


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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})

        self.assertFalse(production.can_train_unit(env, "Red", "soldier"))

        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "barracks", (3, 1)))
        self.assertTrue(production.can_train_unit(env, "Red", "soldier"))
        self.assertTrue(production.train_unit(env, "Red", "soldier"))

        soldiers = [u for u in env.units if u.faction == "Red" and u.unit_type == "soldier"]
        self.assertEqual(len(soldiers), 1)
        self.assertEqual(soldiers[0].attack_damage, 3)

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
        env.resources = [ResourceNode(id=1, position=(3, 1), remaining=60, resource_type="ore")]

        self.assertTrue(env.gather(worker.id))
        self.assertEqual(env.resources[0].remaining, 60)
        self.assertEqual(env.bank["Red"], 255)

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
                "bronze_working",
                "masonry",
                "horsemanship",
                "fletching",
                "iron_working",
                "fortification",
                "stirrups",
                "engineering",
            }
        )
        worker, _, build_pos = self._horse_build_setup(env)

        self.assertTrue(production.can_build(env, "Red", worker.id, "stable", build_pos))
        self.assertTrue(production.build(env, "Red", worker.id, "stable", build_pos))

    def test_horseman_requires_stable_and_has_extended_move(self) -> None:
        env = make_env(num_resource_nodes=0, horse_resource_nodes=2, horse_resource_amount=20)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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

    def test_iron_working_upgrades_new_soldiers(self) -> None:
        env = make_env()
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "iron_working"})
        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "barracks", (3, 1)))
        self.assertTrue(production.train_unit(env, "Red", "soldier"))

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertEqual(soldier.hp, 12)
        self.assertEqual(soldier.attack_damage, 4)

    def test_engineering_unlocks_ballista_tower(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry", "engineering"})
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


class HeuristicAgentTests(unittest.TestCase):
    def test_agent_declares_war_when_push_ready(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "masonry", "horsemanship"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env._spawn_unit("Red", "soldier", 10, (7, 7), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Red", "soldier", 10, (8, 7), attack_damage=3, attack_range=1, move_steps=1)
        env.units = [u for u in env.units if not (u.faction == "Blue" and u.attack_damage > 0)]
        env.faction_states["Blue"].unit_ids = [u.id for u in env.units if u.faction == "Blue"]

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
        self.assertEqual(agent.act(env), ("research", "bronze_working"))

    def test_agent_pursues_horsemanship_when_horses_are_available(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=0, horse_resource_nodes=2, horse_resource_amount=20)
        agent = HeuristicAgent(desired_workers=1)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))

        self.assertEqual(agent.act(env), ("research", "horsemanship"))

    def test_agent_pursues_masonry_when_stone_is_available(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        agent = HeuristicAgent(desired_workers=1)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))

        self.assertEqual(agent.act(env), ("research", "masonry"))

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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
        env.bank["Red"] = 0
        env.resources = [ResourceNode(id=1, position=(0, 0), remaining=60)]
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "masonry", "horsemanship"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 17

        self.assertIsNone(agent.act(env))

    def test_agent_does_not_spawn_extra_worker_without_useful_jobs(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "masonry", "horsemanship"})
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
            ResourceNode(id=1, position=(3, 1), remaining=60),
            ResourceNode(id=2, position=(4, 2), remaining=60),
        ]
        env.faction_state("Red").techs_unlocked.update(
            {
                "mining",
                "bronze_working",
                "masonry",
                "horsemanship",
                "fletching",
                "iron_working",
                "fortification",
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

    def test_military_unit_moves_toward_enemy_when_no_attack_available(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching", "masonry", "horsemanship"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching", "masonry", "horsemanship"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env._spawn_building(faction="Red", building_type="stable", hp=24, pos=(2, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))
        env._spawn_unit("Blue", "soldier", 10, (4, 5), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_soldier = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, blue_soldier.id))

    def test_agent_moves_to_defend_worker_from_nearby_enemy(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 100
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 4, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, next(u.id for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")))

    def test_defense_prioritizes_enemy_camping_spawn_ring(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 10, (2, 2), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 4, (4, 3), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_camper = next(u for u in env.units if u.faction == "Blue" and u.position == (2, 1))

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, blue_camper.id))

    def test_defense_mode_blocks_worker_spawn_during_emergency(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=5)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching", "masonry", "horsemanship"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "soldier", 10, (5, 4), attack_damage=3, attack_range=1, move_steps=1)
        env._spawn_unit("Blue", "horseman", 12, (4, 5), attack_damage=4, attack_range=1, move_steps=3)
        env.declare_war("Red", "Blue")
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        blue_horseman = next(u for u in env.units if u.faction == "Blue" and u.unit_type == "horseman")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, blue_horseman.id))

    def test_horseman_prefers_raiding_worker_targets(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze_working", "masonry"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit("Red", "soldier", 4, (10, 11), attack_damage=3, attack_range=1, move_steps=1)
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertNotEqual(action, ("attack_base", red_soldier.id, "Blue"))

    def test_agent_trains_line_reinforcement_before_horseman_when_behind_ranged_enemy(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.bank["Red"] = 100
        env.bank["Blue"] = 100
        env.faction_state("Red").techs_unlocked.update(
            {"mining", "bronze_working", "masonry", "horsemanship", "fletching"}
        )
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze_working", "fletching", "iron_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env.bases["Blue"].hp = 3
        env._spawn_unit("Red", "soldier", 10, (10, 11), attack_damage=3, attack_range=1, move_steps=1)
        env.declare_war("Red", "Blue")
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack_base", red_soldier.id, "Blue"))

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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze_working", "fletching"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
        env.faction_state("Blue").techs_unlocked.update({"mining", "bronze_working", "fletching"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.resources = [ResourceNode(id=1, position=(2, 1), remaining=60)]
        env.units = [u for u in env.units if u.faction == "Red" and u.unit_type == "worker"]
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = max((u.id for u in env.units), default=0) + 1
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
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
        env.faction_state("Red").techs_unlocked.add("masonry")
        env.faction_state("Red").techs_unlocked.add("mining")
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


class MovementSystemTests(unittest.TestCase):
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
