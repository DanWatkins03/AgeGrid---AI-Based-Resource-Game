from __future__ import annotations

from typing import Callable

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.agents.heuristic_strategy import CandidateSource, UtilityCandidate


ScoreAction = Callable[[AgeGridEnv, object, Action, str, int], UtilityCandidate]
VetoAction = Callable[[AgeGridEnv, object, Action, str], str | None]
TargetingSnapshot = tuple[dict[int, tuple[int, int]], dict[int, int]]


def candidate_tiebreak_priority(candidate: UtilityCandidate) -> int:
    kind = candidate.action[0]
    reasons = set(candidate.reasons)
    if "lethal" in reasons or "enemy_broken" in reasons:
        return 0
    if kind in {"attack_base", "attack"}:
        return 1
    if kind == "move_towards" and (
        "military_positioning" in reasons
        or candidate.source in {"military_movement", "resource_pressure", "base_attack"}
    ):
        return 2
    if candidate.source in {"defense", "emergency_production", "pressure_fallback"} or "home_pressure" in reasons:
        return 3
    if kind in {"train", "spawn_worker"}:
        return 4
    if kind == "build":
        return 5
    if kind == "gather":
        return 6
    return 7


def rank_candidates(
    env: AgeGridEnv,
    ctx: object,
    sources: tuple[CandidateSource, ...],
    score_action: ScoreAction,
    veto_action: VetoAction,
    pressure_fallback_action: Callable[[object], Action | None],
    targeting_snapshot: Callable[[], TargetingSnapshot],
    restore_targeting_snapshot: Callable[[TargetingSnapshot], None],
) -> tuple[list[UtilityCandidate], list[str]]:
    candidates: list[UtilityCandidate] = []
    vetoes: list[str] = []
    for source in sources:
        snapshot = targeting_snapshot()
        action = source.chooser(env, ctx)
        restore_targeting_snapshot(snapshot)
        if action is None:
            continue
        veto_reason = veto_action(env, ctx, action, source.name)
        if veto_reason is not None:
            vetoes.append(f"{source.name} {action}: {veto_reason}")
            continue
        candidates.append(score_action(env, ctx, action, source.name, source.base_score))

    if ctx.defense_mode or ctx.last_stand or ctx.rebuild_mode:
        fallback = pressure_fallback_action(ctx)
        if fallback is not None:
            veto_reason = veto_action(env, ctx, fallback, "pressure_fallback")
            if veto_reason is None:
                candidates.append(score_action(env, ctx, fallback, "pressure_fallback", 480))
            else:
                vetoes.append(f"pressure_fallback {fallback}: {veto_reason}")

    # Source names are useful for debugging, but they are not strategy.  At
    # equal utility, prefer concrete war-winning or emergency actions before
    # economy busywork, then use source/action for deterministic trace order.
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate_tiebreak_priority(candidate),
            candidate.source,
            candidate.action,
        ),
    ), vetoes[-6:]


def source_lookup(sources: tuple[CandidateSource, ...]) -> dict[str, CandidateSource]:
    return {source.name: source for source in sources}
