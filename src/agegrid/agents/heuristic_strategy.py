from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv


@dataclass(frozen=True)
class UtilityCandidate:
    action: Action
    source: str
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StrategicIntent:
    name: str
    reason: str
    next_action: str
    risk: str


@dataclass(frozen=True)
class CandidateSource:
    name: str
    base_score: int
    chooser: Callable[[AgeGridEnv, object], Action | None]


def _military_pressure_action_available(ctx: object) -> bool:
    military_ids = {unit.id for unit in getattr(ctx, "military", ())}
    for action in getattr(ctx, "legal", ()):
        if action[0] in {"attack", "attack_base"} and action[1] in military_ids:
            return True
        if action[0] == "move_towards" and action[1] in military_ids:
            return True
    return False


def _push_overrides_recovery(ctx: object) -> bool:
    if not (ctx.push_mode and ctx.recovery_mode):
        return False
    if ctx.rebuild_mode or ctx.base_under_siege or ctx.last_stand or ctx.home_enemy_force > 0:
        return False
    if ctx.recovery_posture == "critical":
        return False

    severe_gap = (
        ctx.military_gap >= 8
        or ctx.economy_gap >= 8
        or (ctx.tech_deficit >= 3 and (ctx.military_gap >= 5 or ctx.economy_gap >= 5))
    )
    if severe_gap and not ctx.siege_finish:
        return False

    immediate_pressure = (
        ctx.siege_finish
        or bool(ctx.resource_contesters)
        or bool(ctx.enemy_resource_workers)
        or any(action[0] == "attack" for action in ctx.legal)
    )
    # Push mode already proves a safe local/global army advantage.  Recovery
    # only yields here when the gap is not severe and there is a reachable
    # pressure action, so mild economy/tech lag does not cancel a live attack.
    return immediate_pressure and _military_pressure_action_available(ctx)


def strategic_intent(ctx: object) -> StrategicIntent:
    if ctx.base_under_siege or ctx.last_stand:
        reason = "base under siege" if ctx.base_under_siege else "home force losing"
        return StrategicIntent("last_stand", reason, "remove immediate threat", "high")
    if ctx.rebuild_mode:
        return StrategicIntent(
            "rebuild",
            f"home force {ctx.home_friendly_force} vs {ctx.home_enemy_force}",
            "train or pull defenders home",
            "high",
        )
    if ctx.defense_mode or ctx.home_enemy_force > 0:
        reason = "enemy near home" if ctx.home_enemy_force > 0 else "worker/base threat"
        return StrategicIntent("defend", reason, "intercept threat", "medium")
    if ctx.siege_finish:
        return StrategicIntent("siege", "enemy army and workers broken", "attack base", "low")
    if _push_overrides_recovery(ctx):
        return StrategicIntent(
            "push",
            "safe army advantage despite recoverable gap",
            "pressure frontline",
            "medium",
        )
    if ctx.recovery_mode:
        reason_parts = []
        if ctx.economy_gap:
            reason_parts.append(f"economy gap {ctx.economy_gap}")
        if ctx.military_gap:
            reason_parts.append(f"military gap {ctx.military_gap}")
        if ctx.tech_deficit:
            reason_parts.append(f"tech gap {ctx.tech_deficit}")
        if ctx.recovery_posture == "critical":
            next_action = "protect workers and hold base"
            risk = "high"
        elif ctx.recovery_posture == "fragile":
            next_action = "regroup defenders before pressure"
            risk = "high" if ctx.military_gap >= 8 or ctx.economy_gap >= 8 else "medium"
        else:
            next_action = "stabilize economy and defenders"
            risk = "high" if ctx.military_gap >= 8 or ctx.economy_gap >= 8 else "medium"
        return StrategicIntent(
            "recover",
            ", ".join(reason_parts) or "behind",
            next_action,
            risk,
        )
    if ctx.push_mode:
        return StrategicIntent("push", "safe army advantage", "pressure frontline", "medium")
    if ctx.behind_mode:
        return StrategicIntent("catch_up", "strategic deficit", "close tech/economy gap", "medium")
    return StrategicIntent("develop", "no urgent threat", "grow economy and tech", "low")


def action_reasons(
    env: AgeGridEnv,
    ctx: object,
    action: Action,
    intent: StrategicIntent,
    home_pressure: bool,
) -> tuple[str, ...]:
    reasons: list[str] = [intent.name]
    kind = action[0]
    if kind == "train":
        reasons.append(f"train:{action[1]}")
        if ctx.base_under_siege or home_pressure:
            reasons.append("home_pressure")
        if ctx.home_friendly_force < ctx.desired_home_force:
            reasons.append("needs_home_force")
        if ctx.military_gap > 0:
            reasons.append(f"military_gap:{ctx.military_gap}")
    elif kind == "build":
        reasons.append(f"build:{action[2]}")
        if action[2] in {"archer_tower", "ballista_tower", "wall", "stronghold"}:
            reasons.append("territory_defense")
        elif action[2] in {"storehouse", "market", "quarry"}:
            reasons.append("economy_infrastructure")
    elif kind == "research":
        reasons.append(f"research:{action[1]}")
        if ctx.tech_deficit > 0:
            reasons.append(f"tech_gap:{ctx.tech_deficit}")
    elif kind == "attack":
        target = next((unit for unit in env.units if unit.id == action[2]), None)
        if target is not None:
            reasons.append(f"target:{target.unit_type}")
            if target.attack_damage > 0:
                reasons.append("remove_threat")
            if target.hp <= next((unit.attack_damage for unit in env.units if unit.id == action[1]), 0):
                reasons.append("lethal")
    elif kind == "attack_base":
        reasons.append("base_damage")
        if ctx.siege_finish:
            reasons.append("enemy_broken")
    elif kind == "move_towards":
        unit = next((candidate for candidate in env.units if candidate.id == action[1]), None)
        if unit is not None:
            reasons.append(f"move:{unit.unit_type}")
            if unit.attack_damage > 0:
                reasons.append("military_positioning")
                if getattr(ctx, "recovery_posture", "none") in {"critical", "fragile"}:
                    reasons.append(f"recovery_posture:{ctx.recovery_posture}")
                danger = ctx.threat_map.danger_at(action[2])
                if danger > 0:
                    reasons.append(f"destination_danger:{danger}")
            else:
                reasons.append("worker_positioning")
    elif kind == "spawn_worker":
        reasons.append("worker_count")
    elif kind == "gather":
        reasons.append("income_now")
    elif kind in {"declare_war", "offer_peace", "accept_peace"}:
        reasons.append(f"diplomacy:{kind}")
    return tuple(reasons)


def source_intent_bias(intent: StrategicIntent, ctx: object, source: str) -> int:
    if intent.name in {"recover", "rebuild"}:
        if getattr(ctx, "recovery_posture", "none") == "critical":
            if source in {"emergency_production", "production", "defense", "economy", "diplomacy"}:
                return 70
            if source == "resource_pressure":
                return -240
            if source == "military_movement":
                return -70
        if getattr(ctx, "recovery_posture", "none") == "fragile":
            if source in {"emergency_production", "production", "defense", "economy", "diplomacy"}:
                return 55
            if source == "resource_pressure" and not ctx.resource_contesters:
                return -160
        if source in {"emergency_production", "production", "defense", "economy", "diplomacy"}:
            return 45
        if source == "resource_pressure" and ctx.military_gap >= 8 and not ctx.resource_contesters:
            return -180
        return 0
    if intent.name == "catch_up":
        if source in {"research", "production", "economy"}:
            return 55
        if source in {"defense", "emergency_production"}:
            return 30
        if source in {"resource_pressure", "base_attack"}:
            return -85
        return 0
    if intent.name == "siege":
        if source == "base_attack":
            return 175
        if source == "military_movement":
            return 90
        if source in {"economy", "research", "production"}:
            return -140
        return 0
    if intent.name == "defend":
        if source in {"attack", "defense", "emergency_production"}:
            return 60
        return 0
    if intent.name == "push":
        if source in {"attack", "resource_pressure", "base_attack", "military_movement", "diplomacy"}:
            return 35
        if source in {"economy", "worker"}:
            return -45
        return 0
    return 0
