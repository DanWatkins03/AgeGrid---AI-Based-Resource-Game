from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.agents.heuristic_strategy import StrategicIntent


@dataclass(frozen=True)
class VetoHelpers:
    distance: Callable[[tuple[int, int], tuple[int, int]], int]
    enemy_pressure_near_base: Callable[[AgeGridEnv, str], bool]
    enemy_can_finish_base_next_turn: Callable[[AgeGridEnv, str], bool]
    base_camp_targets: Callable[[AgeGridEnv, str], list]


def veto_action(
    env: AgeGridEnv,
    ctx: object,
    action: Action,
    source: str,
    intent: StrategicIntent,
    helpers: VetoHelpers,
) -> str | None:
    kind = action[0]
    if source == "resource_pressure":
        if ctx.base_under_siege or ctx.rebuild_mode:
            return "home emergency blocks resource pressure"
        enemy_combat_exists = any(
            enemy.faction == ctx.enemy_faction and enemy.attack_damage > 0
            for enemy in env.units
        )
        if (
            enemy_combat_exists
            and ctx.recovery_posture in {"critical", "fragile"}
            and not ctx.resource_contesters
        ):
            return "fragile recovery blocks opportunistic pressure"
        if ctx.recovery_mode and ctx.military_gap >= 8 and ctx.home_enemy_force == 0:
            return "severe recovery blocks deep resource pressure"
    if kind == "research" and (ctx.base_under_siege or ctx.home_enemy_force > 0):
        return "urgent home pressure blocks research"
    if kind in {"offer_peace", "accept_peace"}:
        if helpers.enemy_can_finish_base_next_turn(env, ctx.faction) or helpers.base_camp_targets(env, ctx.faction):
            return "imminent base loss blocks peace"
    if kind == "declare_war":
        border_pressure = helpers.enemy_pressure_near_base(env, ctx.faction) or any(
            helpers.distance(unit.position, env.bases[ctx.enemy_faction].position) <= 6
            for unit in ctx.military
        ) or any(
            unit.faction == ctx.enemy_faction
            and unit.attack_damage > 0
            and any(helpers.distance(unit.position, friendly.position) <= 2 for friendly in ctx.military)
            for unit in env.units
        )
        if (
            intent.name in {"recover", "rebuild", "catch_up"}
            and ctx.home_enemy_force == 0
            and not ctx.siege_finish
            and not border_pressure
        ):
            return "strategic deficit blocks new war"
        if (
            "advanced_siege" in ctx.state.techs_unlocked
            and "siege_workshop" not in ctx.buildings
            and ctx.home_enemy_force == 0
            and not ctx.siege_finish
            and not border_pressure
        ):
            return "siege tech waits for siege workshop before war"
    if kind == "attack_base":
        attacker = next((unit for unit in env.units if unit.id == action[1]), None)
        target_base = env.bases[action[2]]
        lethal = attacker is not None and target_base.hp <= attacker.attack_damage
        if not lethal and ctx.recovery_mode and intent.name != "push" and not ctx.siege_finish:
            return "recovery blocks nonlethal base attack"
        if not lethal and ctx.home_enemy_force > ctx.home_friendly_force:
            return "home losing blocks nonlethal base attack"
    if kind == "move_towards":
        unit = next((candidate for candidate in env.units if candidate.id == action[1]), None)
        if unit is not None and unit.attack_damage > 0:
            danger = ctx.threat_map.danger_at(action[2])
            emergency_positions = {target.position for target in ctx.emergency_targets}
            pressure_positions = emergency_positions | {target.position for target in ctx.resource_contesters}
            if (
                intent.name in {"recover", "rebuild", "catch_up"}
                and danger >= 10
                and action[2] not in pressure_positions
            ):
                return "high danger movement during recovery"
            enemy_combat_exists = any(
                enemy.faction == ctx.enemy_faction and enemy.attack_damage > 0
                for enemy in env.units
            )
            if (
                enemy_combat_exists
                and (
                    ctx.recovery_posture == "critical"
                    or (ctx.recovery_posture == "fragile" and ctx.military_gap >= 5 and intent.name != "push")
                )
                and action[2] not in pressure_positions
            ):
                home = env.bases[ctx.faction].position
                if (
                    helpers.distance(action[2], home) > 3
                    and helpers.distance(action[2], home) > helpers.distance(unit.position, home)
                ):
                    return "fragile recovery blocks forward movement"
    return None
