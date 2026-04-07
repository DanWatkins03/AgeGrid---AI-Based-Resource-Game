from __future__ import annotations

import unittest

from src.agegrid.agents.heuristic import HeuristicAgent
from src.agegrid.env.agegrid_env import AgeGridEnv, GameConfig
from src.agegrid.env.systems import combat, movement, production, tech


def make_env(**config_overrides) -> AgeGridEnv:
    config = GameConfig(
        width=12,
        height=12,
        max_turns=50,
        actions_per_turn=3,
        max_attempts_per_turn=10,
        starting_resources=250,
        num_resource_nodes=0,
        seed=7,
        **config_overrides,
    )
    return AgeGridEnv(config)


class TechSystemTests(unittest.TestCase):
    def test_research_requires_prerequisites_and_spends_resources(self) -> None:
        env = make_env()

        self.assertTrue(tech.can_research(env, "Red", "mining"))
        self.assertFalse(tech.can_research(env, "Red", "bronze_working"))

        self.assertTrue(tech.research(env, "Red", "mining"))
        self.assertIn("mining", env.faction_state("Red").techs_unlocked)
        self.assertEqual(env.bank["Red"], 210)
        self.assertTrue(tech.can_research(env, "Red", "bronze_working"))


class ProductionSystemTests(unittest.TestCase):
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
        env = make_env()
        env.faction_state("Red").techs_unlocked.add("mining")

        worker = next(u for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        self.assertTrue(production.build(env, "Red", worker.id, "storehouse", (3, 1)))

        red_before = env.bank["Red"]
        blue_before = env.bank["Blue"]
        env.step_end_turn()

        self.assertEqual(env.bank["Red"], red_before + 2)
        self.assertEqual(env.bank["Blue"], blue_before)


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

    def test_worker_does_not_chase_enemy_without_useful_economic_task(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=3)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "masonry"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 17

        self.assertIsNone(agent.act(env))

    def test_military_unit_moves_toward_enemy_when_no_attack_available(self) -> None:
        env = make_env()
        agent = HeuristicAgent(desired_workers=1)
        env.resources = []
        env.faction_state("Red").techs_unlocked.update({"mining", "bronze_working", "fletching", "masonry"})
        env._spawn_building(faction="Red", building_type="storehouse", hp=18, pos=(0, 2))
        env._spawn_building(faction="Red", building_type="barracks", hp=30, pos=(1, 0))
        env.bank["Red"] = 0
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(4, 4),
            attack_damage=3,
            attack_range=1,
        )

        action = agent.act(env)
        self.assertIsNotNone(action)
        self.assertEqual(action[0], "move_towards")
        self.assertEqual(action[1], next(u.id for u in env.units if u.faction == "Red" and u.unit_type == "soldier"))


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

    def test_legal_actions_include_military_movement_toward_enemy_base(self) -> None:
        env = make_env()
        env._spawn_unit(
            faction="Red",
            unit_type="soldier",
            hp=10,
            pos=(4, 4),
            attack_damage=3,
            attack_range=1,
        )

        soldier = next(u for u in env.units if u.faction == "Red" and u.unit_type == "soldier")
        self.assertIn(("move_towards", soldier.id, env.bases["Blue"].position), env.legal_actions("Red"))

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
