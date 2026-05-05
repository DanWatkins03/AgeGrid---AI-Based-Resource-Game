from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agegrid.agents.heuristic import (
    army_plan,
    army_strength_near_base,
    defense_mode_active,
    heuristic_diagnostics_label,
    push_mode_active,
    threat_level,
    unit_composition,
)
from src.agegrid.env.agegrid_env import AgeGridEnv


BUILDING_LABELS = {
    "storehouse": "Storehouse",
    "barracks": "Barracks",
    "quarry": "Quarry",
    "stable": "Stable",
    "archer_tower": "Archer Tower",
    "ballista_tower": "Ballista Tower",
    "market": "Market",
    "wall": "Wall",
    "stronghold": "Stronghold",
    "siege_workshop": "Siege Workshop",
}


@dataclass
class FactionTurnInfo:
    label: str
    log: list[str]
    last_action: str = "-"
    research: str = "-"
    attacks: str = "-"
    turn_number: int = 0
    decisions: list[str] | None = None


@dataclass
class TurnSnapshot:
    turn_number: int
    red: FactionTurnInfo
    blue: FactionTurnInfo


def _building_label(building_id: str) -> str:
    return BUILDING_LABELS.get(building_id, building_id.replace("_", " ").title())


def _format_action(action: tuple | None) -> str:
    if action is None:
        return "stop"
    kind = action[0]
    if kind == "gather":
        return f"gather worker#{action[1]}"
    if kind == "move_towards":
        return f"move unit#{action[1]} toward {action[2]}"
    if kind == "spawn_worker":
        return "spawn worker"
    if kind == "train":
        return f"train {action[1]}"
    if kind == "build":
        return f"build {action[2]} at {action[3]} with worker#{action[1]}"
    if kind == "research":
        return f"research {action[1]}"
    if kind == "attack":
        return f"attack unit#{action[2]} with unit#{action[1]}"
    if kind == "attack_base":
        return f"attack {action[2]} base with unit#{action[1]}"
    if kind == "declare_war":
        return f"declare war on {action[1]}"
    if kind == "offer_peace":
        return f"offer peace to {action[1]} for {action[2]}"
    if kind == "accept_peace":
        return f"accept peace with {action[1]}"
    return str(action)


def build_turn_info(
    label: str,
    actions: list[tuple | None],
    log: list[str],
    events: list[str],
    turn_number: int = 0,
    decisions: list[str] | None = None,
) -> FactionTurnInfo:
    research = next((_format_action(action) for action in actions if action and action[0] == "research"), "-")
    attacks = next(
        (_format_action(action) for action in actions if action and action[0] in {"attack", "attack_base"}),
        "-",
    )
    return FactionTurnInfo(
        label=label,
        log=log,
        last_action=events[-1] if events else (_format_action(actions[-1]) if actions else "stop"),
        research=research,
        attacks=attacks,
        turn_number=turn_number,
        decisions=(decisions or [])[-6:],
    )


def _agent_decision_lines(agent) -> list[str]:
    explain = getattr(agent, "explain_last_decision", None)
    if explain is None:
        return []
    summary = explain()
    if summary == "No decision":
        return [summary]
    lines = [summary]
    candidates = getattr(agent, "last_candidates", [])
    for candidate in candidates[1:3]:
        reasons = ", ".join(candidate.reasons)
        lines.append(f"Candidate {candidate.source}: {candidate.action} score={candidate.score} ({reasons})")
    return lines


def step_faction_with_trace(env: AgeGridEnv, agent) -> tuple[FactionTurnInfo, list[tuple | None]]:
    actions: list[tuple | None] = []
    decisions: list[str] = []

    def decide(current_env: AgeGridEnv):
        action = agent.act(current_env)
        actions.append(action)
        decisions.extend(_agent_decision_lines(agent))
        return action

    faction = env.factions[env.current_player]
    log = env.step_faction(decide)
    return build_turn_info(faction, actions, log, list(env.current_events), env.turn + 1, decisions), actions


def turn_snapshot(env: AgeGridEnv, red_info: FactionTurnInfo, blue_info: FactionTurnInfo) -> TurnSnapshot:
    return TurnSnapshot(red_info.turn_number or env.turn + 1, red_info, blue_info)


def step_full_turn(
    env: AgeGridEnv,
    red_agent,
    blue_agent,
) -> tuple[FactionTurnInfo, FactionTurnInfo, list[tuple | None], list[tuple | None]]:
    red_info, red_actions = step_faction_with_trace(env, red_agent)
    env.step_end_turn()

    if env.winner() is not None:
        return (
            red_info,
            FactionTurnInfo("Blue", ["turn_skipped:winner"], last_action="-", research="-", attacks="-"),
            red_actions,
            [],
        )

    blue_info, blue_actions = step_faction_with_trace(env, blue_agent)
    env.step_end_turn()

    return red_info, blue_info, red_actions, blue_actions


def turn_history_text(history: list[TurnSnapshot]) -> list[str]:
    lines: list[str] = []
    for snapshot in history:
        lines.append(
            f"Turn {snapshot.turn_number}: Red={snapshot.red.last_action} | Blue={snapshot.blue.last_action}"
        )
        lines.append(
            f"  Red log: {', '.join(snapshot.red.log) if snapshot.red.log else '-'}"
        )
        if snapshot.red.decisions:
            lines.append(f"  Red AI: {' | '.join(snapshot.red.decisions)}")
        lines.append(
            f"  Blue log: {', '.join(snapshot.blue.log) if snapshot.blue.log else '-'}"
        )
        if snapshot.blue.decisions:
            lines.append(f"  Blue AI: {' | '.join(snapshot.blue.decisions)}")
    return lines


def build_debug_snapshot(
    env: AgeGridEnv,
    red_agent_label: str,
    blue_agent_label: str,
    red_info: FactionTurnInfo,
    blue_info: FactionTurnInfo,
    history: list[TurnSnapshot],
) -> str:
    lines = [
        "=== AgeGrid Debug Snapshot ===",
        f"Turn: {env.turn}",
        f"Year/Era: {env.formatted_year()} / {env.current_era()}",
        f"Current player: {env.factions[env.current_player]}",
        f"Winner: {env.winner() or '-'}",
        f"Relations: {env.relation_state('Red', 'Blue').state.title()}",
        f"Collapse rule: {'On' if env.config.collapse_enabled else 'Off'}",
        f"Red agent: {red_agent_label}",
        f"Blue agent: {blue_agent_label}",
        f"Red bank/base HP: {env.bank['Red']} / {env.bases['Red'].hp}",
        f"Blue bank/base HP: {env.bank['Blue']} / {env.bases['Blue'].hp}",
        (
            f"Red threat/mode: {threat_level(env, 'Red')} / "
            f"{'Defense' if defense_mode_active(env, 'Red') else 'Push' if push_mode_active(env, 'Red') else 'Field'}"
        ),
        (
            f"Blue threat/mode: {threat_level(env, 'Blue')} / "
            f"{'Defense' if defense_mode_active(env, 'Blue') else 'Push' if push_mode_active(env, 'Blue') else 'Field'}"
        ),
        f"Red army plan: {army_plan(env, 'Red')}",
        f"Blue army plan: {army_plan(env, 'Blue')}",
        f"Red heuristic: {heuristic_diagnostics_label(env, 'Red')}",
        f"Blue heuristic: {heuristic_diagnostics_label(env, 'Blue')}",
        f"Red AI decision: {' | '.join(red_info.decisions or []) or '-'}",
        f"Blue AI decision: {' | '.join(blue_info.decisions or []) or '-'}",
        f"Red techs: {', '.join(sorted(env.faction_state('Red').techs_unlocked)) or '-'}",
        f"Blue techs: {', '.join(sorted(env.faction_state('Blue').techs_unlocked)) or '-'}",
        f"Red buildings: {', '.join(sorted(f'{_building_label(b.building_type)}@{b.position}' for b in env.buildings if b.faction == 'Red' and b.hp > 0)) or '-'}",
        f"Blue buildings: {', '.join(sorted(f'{_building_label(b.building_type)}@{b.position}' for b in env.buildings if b.faction == 'Blue' and b.hp > 0)) or '-'}",
        (
            "Red comp/base force: "
            f"{unit_composition(env, 'Red')} / {army_strength_near_base(env, 'Red')[0]} vs {army_strength_near_base(env, 'Red')[1]}"
        ),
        (
            "Blue comp/base force: "
            f"{unit_composition(env, 'Blue')} / {army_strength_near_base(env, 'Blue')[0]} vs {army_strength_near_base(env, 'Blue')[1]}"
        ),
        "Units:",
    ]
    lines.extend(
        f"  {u.faction} {u.unit_type}#{u.id} pos={u.position} hp={u.hp} atk={u.attack_damage} rng={u.attack_range}"
        for u in sorted(env.units, key=lambda unit: (unit.faction, unit.unit_type, unit.id))
    )
    lines.extend(
        [
            f"Last red action: {red_info.last_action}",
            f"Last blue action: {blue_info.last_action}",
            f"Red log: {', '.join(red_info.log) if red_info.log else '-'}",
            f"Blue log: {', '.join(blue_info.log) if blue_info.log else '-'}",
            f"Red AI trace: {' | '.join(red_info.decisions or []) or '-'}",
            f"Blue AI trace: {' | '.join(blue_info.decisions or []) or '-'}",
            "Full turn history:",
        ]
    )
    lines.extend(turn_history_text(history))
    return "\n".join(lines)


def write_debug_snapshot(snapshot_text: str) -> Path:
    output_path = Path.cwd() / "agegrid_debug_snapshot.txt"
    output_path.write_text(snapshot_text, encoding="utf-8")
    return output_path
