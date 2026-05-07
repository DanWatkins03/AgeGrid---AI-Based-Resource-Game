from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env.systems import production


@dataclass(frozen=True)
class RecoveryHelpers:
    actions_of_kind: Callable[[list[Action], str], list[Action]]
    attack_priority: Callable[[AgeGridEnv, str, Action], tuple]
    base_camp_targets: Callable[[AgeGridEnv, str], list]
    distance: Callable[[tuple[int, int], tuple[int, int]], int]
    enemy_can_finish_base_next_turn: Callable[[AgeGridEnv, str], bool]
    legal_home_guard_action: Callable[[AgeGridEnv, object, object], Action | None]
    ordered_workers: Callable[[str, list], list]
    threat_score: Callable[[AgeGridEnv, str, tuple[int, int]], int]
    worker_retreat: Callable[[AgeGridEnv, object], Action | None]


def pressure_fallback_action(ctx: object) -> Action | None:
    military_ids = {unit.id for unit in ctx.military}
    priorities = (
        "attack",
        "attack_base",
        "train",
        "build",
        "move_towards",
        "gather",
        "accept_peace",
    )
    for kind in priorities:
        for action in ctx.legal:
            if action[0] == kind:
                if kind == "gather" and (ctx.last_stand or ctx.base_under_siege):
                    # Last-stand fallback should not rank generic economy as a
                    # pressure response. Survival fallback below can still
                    # gather deliberately when one gather buys a defender.
                    continue
                if kind == "move_towards" and action[1] not in military_ids:
                    continue
                return action
    return None


def _gather_to_emergency_train(env: AgeGridEnv, ctx: object, helpers: RecoveryHelpers) -> Action | None:
    if not ctx.workers or ctx.emergency_train is not None:
        return None
    affordable_costs = []
    for unit_type in ("soldier", "archer", "horseman"):
        spec = production.UNIT_DEFS.get(unit_type)
        if spec is None:
            continue
        if spec.required_tech is not None and spec.required_tech not in ctx.state.techs_unlocked:
            continue
        if spec.required_building is not None and spec.required_building not in ctx.buildings:
            continue
        cost = production.unit_cost(env, ctx.faction, unit_type)
        if cost is not None:
            affordable_costs.append(cost)
    if not affordable_costs:
        return None
    cheapest_defender = min(affordable_costs)
    if env.bank[ctx.faction] >= cheapest_defender:
        return None
    if env.bank[ctx.faction] + env.config.worker_gather_amount < cheapest_defender:
        return None
    gather_actions = {action[1]: action for action in helpers.actions_of_kind(ctx.legal, "gather")}
    for worker in helpers.ordered_workers(ctx.faction, ctx.workers):
        gather = gather_actions.get(worker.id)
        if gather is not None:
            return gather
    return None


def _worker_evasion_action(env: AgeGridEnv, ctx: object, helpers: RecoveryHelpers) -> Action | None:
    worker_ids = {worker.id for worker in ctx.workers}
    if not worker_ids:
        return None
    enemy_positions = [
        enemy.position
        for enemy in env.units
        if enemy.faction == ctx.enemy_faction and enemy.attack_damage > 0
    ]
    home = env.bases[ctx.faction].position
    moves = [
        action
        for action in helpers.actions_of_kind(ctx.legal, "move_towards")
        if action[1] in worker_ids
    ]
    if not moves:
        return None

    def priority(action: Action) -> tuple:
        destination = action[2]
        min_enemy_distance = min((helpers.distance(destination, pos) for pos in enemy_positions), default=99)
        on_home = destination == home
        on_spawn_ring = helpers.distance(destination, home) == 1
        base_is_falling = ctx.base_under_siege or ctx.last_stand
        return (
            # This must use actual destination danger, not _threat_score: the
            # latter asks "is this enemy near something important?" and will
            # mis-rank worker escape hexes during a base siege.
            ctx.threat_map.danger_at(destination),
            1 if (on_home or on_spawn_ring) and base_is_falling else 0,
            -min_enemy_distance,
            -helpers.distance(destination, home) if base_is_falling else helpers.distance(destination, home),
            action[1],
            destination,
        )

    return min(moves, key=priority)


def choose_survival_fallback(env: AgeGridEnv, ctx: object, helpers: RecoveryHelpers) -> Action | None:
    attack_actions = helpers.actions_of_kind(ctx.legal, "attack")
    home = env.bases[ctx.faction].position
    critical_recovery = (
        ctx.last_stand
        or ctx.base_under_siege
        or ctx.recovery_posture == "critical"
        or ctx.home_enemy_force > 0
    )
    threatened_workers = [
        worker
        for worker in ctx.workers
        if helpers.threat_score(env, ctx.faction, worker.position) <= 4
    ]
    military_ids = {unit.id for unit in ctx.military}
    military_moves = [
        action
        for action in helpers.actions_of_kind(ctx.legal, "move_towards")
        if action[1] in military_ids
    ]
    if attack_actions:
        lethal_attacks = [
            action
            for action in attack_actions
            if (
                (attacker := next((unit for unit in env.units if unit.id == action[1]), None)) is not None
                and (target := next((unit for unit in env.units if unit.id == action[2]), None)) is not None
                and target.hp <= attacker.attack_damage
            )
        ]
        if lethal_attacks:
            return min(lethal_attacks, key=lambda action: helpers.attack_priority(env, ctx.faction, action))
        threat_attacks = [
            action
            for action in attack_actions
            if (target := next((unit for unit in env.units if unit.id == action[2]), None)) is not None
            and target.attack_damage > 0
        ]
        if threat_attacks:
            return min(threat_attacks, key=lambda action: helpers.attack_priority(env, ctx.faction, action))

    for action in helpers.actions_of_kind(ctx.legal, "attack_base"):
        attacker = next((unit for unit in env.units if unit.id == action[1]), None)
        if attacker is not None and env.bases[action[2]].hp <= attacker.attack_damage:
            return action

    if ctx.emergency_train is not None:
        return ctx.emergency_train

    if critical_recovery:
        gather_for_defender = _gather_to_emergency_train(env, ctx, helpers)
        if gather_for_defender is not None:
            # In a last stand, gathering is only smart when it immediately buys
            # a defender this same turn. Generic economy under siege just burns
            # actions while the base dies.
            return gather_for_defender
        for unit in sorted(ctx.military, key=lambda unit: (helpers.distance(unit.position, home), unit.id)):
            guard_action = helpers.legal_home_guard_action(env, ctx, unit)
            if guard_action is not None:
                return guard_action
        if military_moves:
            return min(
                military_moves,
                key=lambda action: (
                    ctx.threat_map.danger_at(action[2]),
                    helpers.distance(action[2], home),
                    action[1],
                    action[2],
                ),
            )
        worker_evasion = _worker_evasion_action(env, ctx, helpers)
        if worker_evasion is not None:
            return worker_evasion

    worker_retreat = helpers.worker_retreat(env, ctx)
    if worker_retreat is not None and (critical_recovery or threatened_workers):
        return worker_retreat

    if ctx.base_under_siege or helpers.base_camp_targets(env, ctx.faction):
        return None

    if ctx.spawn_actions and (not ctx.workers or ctx.rebuild_mode):
        return ctx.spawn_actions[0]

    for kind in ("train", "build"):
        actions = helpers.actions_of_kind(ctx.legal, kind)
        if actions:
            return actions[0]

    gather_actions = {action[1]: action for action in helpers.actions_of_kind(ctx.legal, "gather")}
    if not critical_recovery:
        for worker in helpers.ordered_workers(ctx.faction, ctx.workers):
            gather = gather_actions.get(worker.id)
            if gather is not None:
                return gather

    for unit in sorted(ctx.military, key=lambda unit: (helpers.distance(unit.position, env.bases[ctx.faction].position), unit.id)):
        guard_action = helpers.legal_home_guard_action(env, ctx, unit)
        if guard_action is not None:
            return guard_action

    if ctx.spawn_actions:
        return ctx.spawn_actions[0]

    if military_moves:
        return min(
            military_moves,
            key=lambda action: (
                ctx.threat_map.danger_at(action[2]),
                helpers.distance(action[2], home),
                action[1],
                action[2],
            ),
        )

    if worker_retreat is not None:
        return worker_retreat

    for worker in helpers.ordered_workers(ctx.faction, ctx.workers):
        gather = gather_actions.get(worker.id)
        if gather is not None:
            return gather

    non_war_diplomacy = [
        action
        for action in ctx.accept_peace_actions + ctx.offer_peace_actions
        if not helpers.enemy_can_finish_base_next_turn(env, ctx.faction)
        and not helpers.base_camp_targets(env, ctx.faction)
    ]
    if non_war_diplomacy:
        return non_war_diplomacy[0]

    for action in ctx.legal:
        if action[0] != "declare_war":
            return action
    if not ctx.at_war:
        return None
    return ctx.legal[0] if ctx.legal else None
