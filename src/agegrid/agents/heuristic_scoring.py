from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env.systems import combat, production
from src.agegrid.agents.heuristic_strategy import StrategicIntent


@dataclass(frozen=True)
class ScoringHelpers:
    distance: Callable[[tuple[int, int], tuple[int, int]], int]
    enemy_pressure_near_base: Callable[[AgeGridEnv, str], bool]
    enemy_can_finish_base_next_turn: Callable[[AgeGridEnv, str], bool]
    hold_defender_reserve: Callable[[AgeGridEnv, object, int], bool]
    unit_count: Callable[[AgeGridEnv, str, str], int]


def _economy_direct_win_path(env: AgeGridEnv, ctx: object) -> bool:
    target_bank = env.config.target_bank
    if target_bank is None:
        return False
    gather_swing = env.config.worker_gather_amount * max(1, len(getattr(ctx, "workers", ())))
    return env.bank[ctx.faction] + gather_swing >= target_bank


def _military_action_pressures_base(env: AgeGridEnv, ctx: object, action: Action, helpers: ScoringHelpers) -> bool:
    if action[0] == "attack_base":
        return True
    if action[0] != "move_towards":
        return False
    unit = next((candidate for candidate in ctx.military if candidate.id == action[1]), None)
    if unit is None:
        return False
    enemy_base = env.bases[ctx.enemy_faction].position
    return helpers.distance(action[2], enemy_base) <= max(3, unit.attack_range + 1)


def _credible_military_opportunity(env: AgeGridEnv, ctx: object, helpers: ScoringHelpers) -> bool:
    if not ctx.at_war or not ctx.military or ctx.home_enemy_force > 0 or ctx.base_under_siege:
        return False
    if ctx.recovery_posture == "critical":
        return False
    if not (ctx.push_mode or ctx.siege_finish or ctx.military_gap <= 0):
        return False
    if _economy_direct_win_path(env, ctx):
        return False
    if any(action[0] in {"attack", "attack_base"} for action in ctx.legal):
        return True
    return any(
        _military_action_pressures_base(env, ctx, action, helpers)
        for action in ctx.legal
        if action[0] == "move_towards"
    )


def _winning_finish_window(env: AgeGridEnv, ctx: object, helpers: ScoringHelpers) -> bool:
    if not ctx.at_war or not ctx.military:
        return False
    our_score = env.war_score(ctx.faction, ctx.enemy_faction)
    enemy_score = env.war_score(ctx.enemy_faction, ctx.faction)
    if our_score - enemy_score < env.config.peace_winning_score_margin:
        return False
    if ctx.military_gap > 0 or ctx.home_enemy_force > ctx.home_friendly_force:
        return False
    enemy_base = env.bases[ctx.enemy_faction]
    base_damaged = enemy_base.hp < env.config.base_hp
    front_units = [
        unit for unit in ctx.military
        if helpers.distance(unit.position, enemy_base.position) <= env.config.peace_winning_front_radius
    ]
    reachable_pressure = bool(front_units) or any(action[0] == "attack_base" for action in ctx.legal)
    if not reachable_pressure:
        return False
    near_damage = sum(max(1, unit.attack_damage) for unit in front_units)
    plausibly_finishes = ctx.siege_finish or enemy_base.hp <= env.config.peace_winning_finish_hp + near_damage
    return base_damaged or plausibly_finishes or ctx.push_mode


def _peace_escape_needed(env: AgeGridEnv, ctx: object, helpers: ScoringHelpers) -> bool:
    if env.faction_state(ctx.faction).war_support <= env.config.peace_exhausted_war_support:
        return True
    if ctx.recovery_posture == "critical" or ctx.last_stand or ctx.rebuild_mode:
        return True
    if ctx.military_gap >= 5 or ctx.home_enemy_force > ctx.home_friendly_force:
        return True
    war_turns = env.turn - env.relation_state(ctx.faction, ctx.enemy_faction).since_turn
    enemy_base = env.bases[ctx.enemy_faction]
    front_units = [
        unit for unit in ctx.military
        if helpers.distance(unit.position, enemy_base.position) <= env.config.peace_winning_front_radius
    ]
    # Long wars with no base damage and no active front are treated as stalled;
    # in that case peace is a reset, not surrendering a live finish.
    return (
        war_turns >= env.config.peace_stalled_war_turns
        and enemy_base.hp >= env.config.base_hp
        and not front_units
    )


def _near_immediate_war_purchase(env: AgeGridEnv, ctx: object) -> bool:
    next_bank = env.bank[ctx.faction] + env.config.worker_gather_amount
    if ctx.home_friendly_force < ctx.desired_home_force or ctx.military_gap > 0:
        for unit_type in ("soldier", "archer", "horseman"):
            cost = production.unit_cost(env, ctx.faction, unit_type)
            if cost is not None and env.bank[ctx.faction] < cost <= next_bank:
                return True
    if ctx.push_mode or ctx.siege_finish:
        ballista_cost = production.unit_cost(env, ctx.faction, "ballista")
        if ballista_cost is not None and env.bank[ctx.faction] < ballista_cost <= next_bank:
            return True
        workshop_cost = production.building_cost(env, ctx.faction, "siege_workshop")
        if (
            workshop_cost is not None
            and "advanced_siege" in ctx.state.techs_unlocked
            and "siege_workshop" not in ctx.buildings
            and env.bank[ctx.faction] < workshop_cost <= next_bank
        ):
            return True
    return False


def utility_modifier(
    env: AgeGridEnv,
    ctx: object,
    action: Action,
    intent: StrategicIntent,
    helpers: ScoringHelpers,
) -> int:
    kind = action[0]
    modifier = 0
    military_opportunity = _credible_military_opportunity(env, ctx, helpers)
    winning_finish = _winning_finish_window(env, ctx, helpers)
    economy_terminal = env.config.target_bank is not None
    if ctx.base_under_siege:
        modifier += 120 if kind in {"attack", "train", "move_towards"} else -80
    elif ctx.defense_mode or ctx.home_enemy_force > 0:
        modifier += 70 if kind in {"attack", "train", "move_towards", "build"} else -35
    elif ctx.push_mode or ctx.siege_finish:
        modifier += 55 if kind in {"attack", "attack_base", "move_towards", "declare_war"} else 0
    elif ctx.behind_mode:
        modifier += 40 if kind in {"train", "build", "gather", "spawn_worker"} else 0

    if intent.name in {"recover", "rebuild"}:
        if kind in {"train", "build", "gather", "spawn_worker", "accept_peace", "offer_peace"}:
            modifier += 35
        if military_opportunity and kind in {"build", "gather", "spawn_worker", "accept_peace", "offer_peace"}:
            # Recovery normally pads passive stabilizers, but when the army is
            # already winning a reachable war front and bank victory is not
            # immediate, spending the turn on economy can throw away the finish.
            modifier -= 45
        if kind == "move_towards" and ctx.military_gap >= 8 and not _military_action_pressures_base(env, ctx, action, helpers):
            modifier -= 35
        if kind in {"declare_war", "attack_base"} and not helpers.enemy_can_finish_base_next_turn(env, ctx.enemy_faction):
            modifier -= 20 if military_opportunity and kind == "attack_base" else 80
    elif intent.name == "siege":
        if kind == "attack_base":
            modifier += 160
        elif kind == "move_towards":
            modifier += 75
        elif kind in {"research", "gather", "spawn_worker"}:
            modifier -= 120
    elif intent.name == "push":
        if kind in {"attack", "move_towards", "attack_base", "declare_war"}:
            modifier += 25
        if kind in {"gather", "spawn_worker"} and ctx.military:
            modifier -= 30

    if kind == "train":
        unit_type = action[1]
        if ctx.home_friendly_force < ctx.desired_home_force:
            modifier += 45
        if ctx.military_gap > 0:
            modifier += min(60, ctx.military_gap * 6)
        if unit_type == "archer" and ctx.nearby_cavalry:
            modifier += 30
    elif kind == "build":
        building_type = action[2]
        if building_type in {"archer_tower", "ballista_tower", "wall", "stronghold"}:
            modifier += 35 if ctx.defense_mode or ctx.behind_mode else 10
            if building_type == "stronghold" and (ctx.recovery_mode or ctx.behind_mode or ctx.base_under_siege):
                modifier += 100
            elif building_type in {"ballista_tower", "wall"} and (ctx.recovery_mode or ctx.behind_mode):
                modifier += 45
        if building_type in {"storehouse", "market", "quarry"} and ctx.economy_gap > 0:
            modifier += min(40, ctx.economy_gap * 5)
        if building_type == "siege_workshop":
            if ctx.push_mode or ctx.siege_finish:
                modifier += 120
            elif "advanced_siege" in ctx.state.techs_unlocked:
                modifier += 80
    elif kind == "research":
        modifier += min(35, ctx.tech_deficit * 7)
        if ctx.at_war:
            modifier -= 25
    elif kind == "attack":
        target = next((unit for unit in env.units if unit.id == action[2]), None)
        attacker = next((unit for unit in env.units if unit.id == action[1]), None)
        if target is not None:
            if target.attack_damage > 0:
                modifier += 35
            if attacker is not None and target.hp <= attacker.attack_damage:
                modifier += 25
            modifier -= min(15, target.hp)
    elif kind == "attack_base":
        enemy = action[2]
        attacker = next((unit for unit in env.units if unit.id == action[1]), None)
        if attacker is not None and env.bases[enemy].hp <= combat.base_assault_damage(env, attacker, enemy):
            modifier += 100
    elif kind == "declare_war":
        border_pressure = helpers.enemy_pressure_near_base(env, ctx.faction) or any(
            helpers.distance(unit.position, env.bases[ctx.enemy_faction].position) <= 6
            for unit in ctx.military
        ) or any(
            unit.faction == ctx.enemy_faction
            and unit.attack_damage > 0
            and any(helpers.distance(unit.position, friendly.position) <= 2 for friendly in ctx.military)
            for unit in env.units
        )
        if border_pressure:
            modifier += 90
    elif kind in {"offer_peace", "accept_peace"}:
        escape_needed = _peace_escape_needed(env, ctx, helpers)
        if escape_needed:
            modifier += 300
        if winning_finish and not escape_needed:
            # Peace while winning is closer to surrendering initiative than
            # stabilizing. If we are ahead on score, stronger in the field, and
            # already near a damaged or reachable base, make war-winning moves
            # beat safe but pointless treaty actions.
            modifier -= env.config.peace_winning_disfavor
    elif kind == "move_towards":
        unit = next((candidate for candidate in env.units if candidate.id == action[1]), None)
        if unit is not None and unit.attack_damage > 0:
            modifier += 20
            danger = ctx.threat_map.danger_at(action[2])
            pressure_move = _military_action_pressures_base(env, ctx, action, helpers)
            danger_cap = (
                40
                if intent.name in {"recover", "rebuild"} and pressure_move and military_opportunity
                else 80
                if intent.name in {"recover", "rebuild"}
                else 55
                if intent.name != "siege"
                else 35
            )
            modifier -= min(danger_cap, danger * 8)
            if danger >= 6 and not (ctx.siege_finish or (pressure_move and military_opportunity)):
                modifier -= 20
        elif ctx.defense_mode or ctx.rebuild_mode:
            modifier -= 40
    elif kind == "spawn_worker":
        worker_cost = production.unit_cost(env, ctx.faction, "worker") or env.config.worker_spawn_cost
        if helpers.hold_defender_reserve(env, ctx, worker_cost):
            modifier -= 120
        elif helpers.unit_count(env, ctx.faction, "worker") < ctx.useful_worker_slots:
            modifier += 35
    elif kind == "gather":
        modifier += min(25, ctx.economy_gap * 4)
        if ctx.base_under_siege:
            modifier -= 75
        if not economy_terminal:
            defensive_need = ctx.home_friendly_force < ctx.desired_home_force or ctx.recovery_posture == "critical"
            immediate_war_buy = _near_immediate_war_purchase(env, ctx)
            # With bank victory disabled, resources are instrumental rather than
            # terminal: they matter when they unlock defense or war production,
            # but they do not directly win.  If a useful pressure action exists,
            # stop letting "gather now" compete with base-kill/collapse paths.
            if military_opportunity and not defensive_need and not immediate_war_buy:
                modifier -= 65 if ctx.at_war and ctx.military_gap <= 0 else 45
            elif ctx.at_war and ctx.military_gap <= 0 and not immediate_war_buy:
                modifier -= 25
    return modifier
