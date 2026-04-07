from __future__ import annotations

import unittest

from src.agegrid.agents.heuristic import HeuristicAgent
from src.agegrid.env.agegrid_env import AgeGridEnv, GameConfig
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

    def test_quarry_requires_visible_stone_resource(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.update({"mining", "masonry"})
        worker, _, build_pos = self._stone_build_setup(env)

        self.assertTrue(production.can_build(env, "Red", worker.id, "quarry", build_pos))
        self.assertTrue(production.build(env, "Red", worker.id, "quarry", build_pos))

    def test_stable_requires_visible_horse_resource(self) -> None:
        env = make_env(num_resource_nodes=0, horse_resource_nodes=2, horse_resource_amount=20)
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "horsemanship"})
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


class HeuristicAgentTests(unittest.TestCase):
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
        red_soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")

        action = agent.act(env)
        self.assertEqual(action, ("attack", red_soldier.id, next(u.id for u in env.units if u.faction == "Blue" and u.unit_type == "soldier")))

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

    def test_collapse_mode_rebuilds_worker_even_in_defense_mode(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env.bank["Red"] = 20
        env.units = [u for u in env.units if u.faction != "Red" or u.unit_type != "worker"]
        env.faction_state("Red").unit_ids = {u.id for u in env.units if u.faction == "Red"}
        env._spawn_unit("Blue", "soldier", 10, (2, 1), attack_damage=3, attack_range=1, move_steps=1)

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

    def test_agent_allows_base_attack_when_it_can_finish_base(self) -> None:
        env = make_env(target_bank=999)
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env.bases["Blue"].hp = 3
        env._spawn_unit("Red", "soldier", 10, (10, 11), attack_damage=3, attack_range=1, move_steps=1)
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

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertNotEqual(action[2], env.bases["Blue"].position)

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

        action = agent.act(env)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[2], env.bases["Blue"].position)

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

        action = agent.act(env)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[2], env.bases["Blue"].position)


class CombatAndVictoryTests(unittest.TestCase):
    def test_base_attack_can_end_the_game(self) -> None:
        env = make_env(target_bank=999)
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

    def test_combat_records_kill_events(self) -> None:
        env = make_env()
        env.units = []
        env.faction_states["Red"].unit_ids.clear()
        env.faction_states["Blue"].unit_ids.clear()
        env._next_unit_id = 1
        env._spawn_unit("Red", "soldier", 10, (4, 4), attack_damage=5, attack_range=1)
        env._spawn_unit("Blue", "worker", 5, (5, 4))

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

    def test_turret_retaliation_can_finish_enemy(self) -> None:
        env = make_env(num_resource_nodes=0, stone_resource_nodes=2, stone_resource_amount=20, horse_resource_nodes=0)
        env.faction_state("Red").techs_unlocked.add("masonry")
        env.faction_state("Red").techs_unlocked.add("mining")
        stone_worker, _, quarry_pos = ProductionSystemTests()._stone_build_setup(env)
        self.assertTrue(production.build(env, "Red", stone_worker.id, "quarry", quarry_pos))
        turret_worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        turret_worker.position = (quarry_pos[0] + 1, quarry_pos[1])
        turret_pos = next(
            pos
            for pos in (
                (turret_worker.position[0] + 1, turret_worker.position[1]),
                (turret_worker.position[0] - 1, turret_worker.position[1]),
                (turret_worker.position[0], turret_worker.position[1] + 1),
                (turret_worker.position[0], turret_worker.position[1] - 1),
            )
            if env._in_bounds(pos) and pos not in env._occupied_positions() and pos not in {b.position for b in env.buildings}
        )
        self.assertTrue(production.build(env, "Red", turret_worker.id, "turret", turret_pos))
        enemy_pos = (min(env.config.width - 1, turret_pos[0] + 1), turret_pos[1])
        if enemy_pos == turret_pos or enemy_pos in env._occupied_positions():
            enemy_pos = (max(0, turret_pos[0] - 1), turret_pos[1])
        env._spawn_unit("Blue", "soldier", 2, enemy_pos, attack_damage=3, attack_range=1, move_steps=1)

        env.step_end_turn()

        self.assertFalse(any(u.faction == "Blue" and u.unit_type == "soldier" for u in env.units))
        self.assertIn("Red turret shot down Blue soldier#", " | ".join(env.recent_events))

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
        self.assertEqual(soldier.position, (1, 2))


if __name__ == "__main__":
    unittest.main()
