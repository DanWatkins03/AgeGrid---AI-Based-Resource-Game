from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pygame

from src.agegrid.agents.heuristic import (
    army_plan,
    army_strength_near_base,
    defense_mode_active,
    heuristic_diagnostics_label,
    push_mode_active,
    threat_level,
    unit_composition,
)
from src.agegrid.agents.registry import AGENT_SPECS, create_agent
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env import hexgrid
from src.agegrid.env.systems import movement, production, tech
from src.agegrid.ui.assets import BoardAssets


@dataclass
class FactionTurnInfo:
    label: str
    log: list[str]
    last_action: str = "-"
    research: str = "-"
    attacks: str = "-"


@dataclass
class TurnSnapshot:
    turn_number: int
    red: FactionTurnInfo
    blue: FactionTurnInfo


@dataclass
class VisualEffect:
    kind: str
    ttl: int
    max_ttl: int
    color: tuple[int, int, int]
    pos: tuple[int, int] | None = None
    label: str = ""


@dataclass(frozen=True)
class HumanActionOption:
    label: str
    payload: object
    active: bool = False
    enabled: bool = True
    reason: str | None = None


RED_PRIMARY = (196, 88, 80)
RED_ACCENT = (241, 202, 160)
BLUE_PRIMARY = (88, 126, 212)
BLUE_ACCENT = (176, 222, 255)
HUD_BG = (12, 17, 24)
PANEL_BG = (20, 27, 36)
PANEL_INSET = (26, 33, 43)
PANEL_BORDER = (70, 82, 98)
PANEL_SOFT = (40, 48, 59)
TEXT_PRIMARY = (236, 240, 245)
TEXT_SECONDARY = (170, 179, 190)
TEXT_MUTED = (127, 137, 149)
GOLD_ACCENT = (210, 176, 104)
GOLD_DIM = (148, 124, 72)

# HUD layout constants
HUD_H = 110           # height of the three top-bar panels
HUD_INNER_PAD = 9     # inner padding inside each HUD panel
HUD_FACTION_W = 210   # each faction bar width; may be capped to fit board width

# Text tones for drawing on top of parchment/asset panel backgrounds
PARCH_TITLE = (80, 62, 46)    # main title on parchment
PARCH_BODY = (92, 74, 52)     # body text on parchment
PARCH_MUTED = (118, 97, 74)   # muted/secondary on parchment
PARCH_SHADOW = (240, 224, 196)
PARCH_ACCENT = (150, 120, 84)
PARCH_LINE = (188, 168, 136)
GRID_BG = (24, 30, 38)
GRID_ALT = (28, 35, 43)
GRID_LINE = (61, 72, 84)
GRID_HOVER = (188, 214, 234)
RESOURCE_GLOW = (109, 192, 116)
HEX_HOVER_FILL = (176, 206, 230)
HEX_SELECT_FILL = (245, 216, 120)
HEX_SELECT_LINE = (250, 234, 167)
HEX_MOVE_FILL = (111, 201, 136)
HEX_MOVE_LINE = (194, 247, 205)
HEX_BUILD_FILL = (214, 164, 92)
HEX_BUILD_LINE = (248, 220, 168)
BOARD_SHADOW = (6, 9, 13)
BASE_HEX_SIZE = 31
HEX_SIZE = BASE_HEX_SIZE
HEX_WIDTH = round(math.sqrt(3) * HEX_SIZE)
HEX_HEIGHT = HEX_SIZE * 2
MIN_ZOOM = 0.75
MAX_ZOOM = 2.0
ZOOM_STEP = 0.12

TECH_ICON_STYLES = {
    "mining": {"label": "M", "bg": (79, 121, 82), "fg": (232, 245, 220)},
    "bronze": {"label": "B", "bg": (145, 103, 64), "fg": (247, 229, 202)},
    "masonry": {"label": "S", "bg": (102, 108, 120), "fg": (236, 240, 246)},
    "animal_husbandry": {"label": "A", "bg": (112, 94, 62), "fg": (246, 230, 205)},
    "iron": {"label": "I", "bg": (95, 103, 112), "fg": (236, 239, 242)},
    "fletching": {"label": "F", "bg": (72, 108, 134), "fg": (224, 238, 250)},
    "construction": {"label": "C", "bg": (122, 112, 92), "fg": (244, 236, 220)},
    "trade": {"label": "T", "bg": (94, 122, 88), "fg": (230, 243, 222)},
    "horseback_riding": {"label": "H", "bg": (126, 90, 58), "fg": (246, 230, 205)},
    "agriculture": {"label": "G", "bg": (102, 132, 74), "fg": (236, 244, 222)},
    "steel": {"label": "S", "bg": (86, 94, 104), "fg": (236, 239, 242)},
    "fortify": {"label": "F", "bg": (92, 98, 84), "fg": (238, 241, 228)},
    "stirrups": {"label": "R", "bg": (108, 82, 58), "fg": (245, 232, 214)},
    "engineering": {"label": "E", "bg": (84, 104, 128), "fg": (227, 238, 248)},
    "precision": {"label": "P", "bg": (86, 114, 140), "fg": (229, 240, 250)},
    "walls": {"label": "W", "bg": (118, 116, 106), "fg": (242, 240, 232)},
    "infrastructure": {"label": "I", "bg": (120, 110, 88), "fg": (244, 236, 220)},
    "markets": {"label": "M", "bg": (116, 96, 58), "fg": (248, 232, 204)},
    "currency": {"label": "$", "bg": (140, 118, 64), "fg": (252, 242, 210)},
    "logistics": {"label": "L", "bg": (96, 118, 136), "fg": (232, 242, 250)},
    "stronghold": {"label": "H", "bg": (100, 88, 70), "fg": (246, 236, 224)},
    "heavy_cavalry": {"label": "C", "bg": (118, 84, 68), "fg": (248, 232, 220)},
    "advanced_siege": {"label": "A", "bg": (88, 102, 118), "fg": (236, 242, 248)},
    "war_economy": {"label": "W", "bg": (122, 92, 70), "fg": (246, 234, 222)},
}

TECH_LABELS = {
    "mining": "Mining",
    "bronze": "Bronze",
    "masonry": "Masonry",
    "animal_husbandry": "Animal Husbandry",
    "iron": "Iron",
    "fletching": "Fletching",
    "construction": "Construction",
    "trade": "Trade",
    "horseback_riding": "Horseback Riding",
    "agriculture": "Agriculture",
    "steel": "Steel",
    "fortify": "Fortify",
    "engineering": "Engineering",
    "precision": "Precision",
    "walls": "Walls",
    "infrastructure": "Infrastructure",
    "markets": "Markets",
    "currency": "Currency",
    "stirrups": "Stirrups",
    "logistics": "Logistics",
    "stronghold": "Stronghold",
    "heavy_cavalry": "Heavy Cavalry",
    "advanced_siege": "Advanced Siege",
    "war_economy": "War Economy",
}

TECH_TREE_ORDER = list(tech.TECH_TREE_ORDER)

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

BUILDING_HELP = {
    "storehouse": "Passive income each turn",
    "barracks": "Trains soldiers and supports infantry tech",
    "quarry": "Unlocks stone economy scaling",
    "stable": "Trains horsemen",
    "archer_tower": "Defensive ranged tower",
    "ballista_tower": "Late defensive siege tower",
    "market": "Trade hub that improves economic scaling",
    "wall": "Cheap blocking fortification that hardens your frontier",
    "stronghold": "Heavy defensive fortification with stronger fire",
    "siege_workshop": "Produces late siege units",
}

UNIT_LABELS = {
    "worker": "Worker",
    "soldier": "Soldier",
    "archer": "Archer",
    "horseman": "Horseman",
    "heavy_cavalry": "Heavy Cavalry",
    "ballista": "Ballista",
}

UNIT_HELP = {
    "worker": "Economic unit that gathers resources and constructs buildings.",
    "soldier": "Frontline melee fighter used to hold territory and pressure bases.",
    "archer": "Ranged support unit that softens targets before they can answer back.",
    "horseman": "Fast cavalry raider that can flank, chase, and punish exposed units.",
    "heavy_cavalry": "Armored cavalry shock unit that hits harder and lasts longer.",
    "ballista": "Slow siege engine with long range and heavy anti-structure pressure.",
}

RESOURCE_LABELS = {
    "ore": "Ore Vein",
    "stone": "Stone Deposit",
    "horses": "Horse Herd",
}

RESOURCE_HELP = {
    "ore": "Basic gatherable resource node that feeds your economy.",
    "stone": "Strategic node used for quarry, walls, and stronger structures.",
    "horses": "Strategic node used for cavalry infrastructure and mounted units.",
}


def _set_hex_zoom(zoom: float) -> float:
    global HEX_SIZE, HEX_WIDTH, HEX_HEIGHT
    clamped = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
    HEX_SIZE = max(18, round(BASE_HEX_SIZE * clamped))
    HEX_WIDTH = round(math.sqrt(3) * HEX_SIZE)
    HEX_HEIGHT = HEX_SIZE * 2
    return HEX_SIZE / BASE_HEX_SIZE


def _tech_label(tech_id: str) -> str:
    return TECH_LABELS.get(tech_id, tech_id.replace("_", " ").title())


def _building_label(building_id: str) -> str:
    return BUILDING_LABELS.get(building_id, building_id.replace("_", " ").title())


def _tech_unlock_summary(tech_id: str) -> str:
    labels: list[str] = []
    for item in tech.unlock_items(tech_id):
        if item in TECH_LABELS:
            labels.append(_tech_label(item))
        elif item in BUILDING_LABELS:
            labels.append(_building_label(item))
        elif item in UNIT_LABELS:
            labels.append(UNIT_LABELS[item])
        else:
            labels.append(item.replace("_", " ").title())
    return ", ".join(labels) if labels else "-"


def _tech_status(env: AgeGridEnv, faction: str, tech_id: str) -> str:
    state = env.faction_state(faction)
    if tech_id in state.techs_unlocked:
        return "Done"
    if state.tech_in_progress == tech_id:
        return f"Active ({tech.research_turns_remaining(env, faction)}t)"
    if tech.can_research(env, faction, tech_id):
        return "Ready"
    return "-"


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


def _build_turn_info(label: str, actions: list[tuple | None], log: list[str], events: list[str]) -> FactionTurnInfo:
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
    )


def _step_faction_with_trace(env: AgeGridEnv, agent) -> tuple[FactionTurnInfo, list[tuple | None]]:
    actions: list[tuple | None] = []

    def decide(e: AgeGridEnv):
        action = agent.act(e)
        actions.append(action)
        return action

    faction = env.factions[env.current_player]
    log = env.step_faction(decide)
    return _build_turn_info(faction, actions, log, list(env.current_events)), actions


def _step_full_turn(
    env: AgeGridEnv,
    red_agent,
    blue_agent,
) -> tuple[FactionTurnInfo, FactionTurnInfo, list[tuple | None], list[tuple | None]]:
    red_info, red_actions = _step_faction_with_trace(env, red_agent)
    env.step_end_turn()

    if env.winner() is not None:
        return (
            red_info,
            FactionTurnInfo("Blue", ["turn_skipped:winner"], last_action="-", research="-", attacks="-"),
            red_actions,
            [],
        )

    blue_info, blue_actions = _step_faction_with_trace(env, blue_agent)
    env.step_end_turn()

    return red_info, blue_info, red_actions, blue_actions


def _is_human_agent_key(agent_key: str) -> bool:
    return agent_key == "human"


def _current_agent_key(env: AgeGridEnv, red_index: int, blue_index: int) -> str:
    return AGENT_SPECS[red_index].key if env.factions[env.current_player] == "Red" else AGENT_SPECS[blue_index].key


def _match_has_human_players(red_index: int, blue_index: int) -> bool:
    return _is_human_agent_key(AGENT_SPECS[red_index].key) or _is_human_agent_key(AGENT_SPECS[blue_index].key)


def _valid_human_move_targets(env: AgeGridEnv, unit, moved_units: set[int]) -> list[tuple[int, int]]:
    if unit.faction != env.factions[env.current_player]:
        return []
    if env.actions_left <= 0 or env.attempts_left <= 0:
        return []

    valid_targets: list[tuple[int, int]] = []
    for pos in hexgrid.neighbors(unit.position):
        if not env._in_bounds(pos):
            continue
        if movement.can_move_towards(env, unit.id, pos):
            valid_targets.append(pos)
    return valid_targets


def _can_human_gather(env: AgeGridEnv, unit, moved_units: set[int]) -> bool:
    if unit is None or unit.unit_type != "worker":
        return False
    if unit.faction != env.factions[env.current_player]:
        return False
    if env.actions_left <= 0 or env.attempts_left <= 0:
        return False
    return env.resource_at_for_faction(unit.position, unit.faction) is not None


def _human_build_targets(env: AgeGridEnv, worker, building_type: str) -> list[tuple[int, int]]:
    if worker is None or worker.unit_type != "worker":
        return []
    return [
        pos
        for pos in hexgrid.neighbors(worker.position)
        if production.can_build(env, worker.faction, worker.id, building_type, pos)
    ]


def _turn_budget_reason(env: AgeGridEnv) -> str | None:
    if env.actions_left <= 0:
        return "No actions left this turn."
    if env.attempts_left <= 0:
        return "No attempts left this turn."
    return None


def _train_option(env: AgeGridEnv, faction: str, unit_type: str) -> HumanActionOption:
    label = UNIT_LABELS.get(unit_type, unit_type.title()) if unit_type != "worker" else "Worker"
    payload = ("spawn_worker",) if unit_type == "worker" else ("train", unit_type)
    budget_reason = _turn_budget_reason(env)
    if budget_reason:
        return HumanActionOption(label=label, payload=payload, enabled=False, reason=budget_reason)

    spec = production.unit_stats(env, faction, unit_type)
    if spec is None:
        return HumanActionOption(label=label, payload=payload, enabled=False, reason="Unavailable.")
    cost = env.config.worker_spawn_cost if unit_type == "worker" else spec.cost
    state = env.faction_state(faction)
    if state.resources < cost:
        return HumanActionOption(label=label, payload=payload, enabled=False, reason=f"Costs {cost}.")
    if spec.required_tech and spec.required_tech not in state.techs_unlocked:
        return HumanActionOption(label=label, payload=payload, enabled=False, reason=f"Needs {_tech_label(spec.required_tech)}.")
    if spec.required_building and not any(
        building.building_type == spec.required_building for building in env.get_buildings_for_faction(faction)
    ):
        return HumanActionOption(label=label, payload=payload, enabled=False, reason=f"Needs {_building_label(spec.required_building)}.")
    if unit_type == "worker":
        workers = [unit for unit in env.get_units_for_faction(faction) if unit.unit_type == "worker"]
        if len(workers) >= env.config.max_workers:
            return HumanActionOption(label=label, payload=payload, enabled=False, reason="Worker cap reached.")
    occ = env._occupied_positions()
    if not any(env._in_bounds(pos) and pos not in occ for pos in hexgrid.neighbors(env.bases[faction].position)):
        return HumanActionOption(label=label, payload=payload, enabled=False, reason="Spawn tiles are blocked.")
    return HumanActionOption(label=label, payload=payload)


def _build_option(env: AgeGridEnv, faction: str, worker, building_type: str, pending_build_type: str | None) -> HumanActionOption:
    label = _building_label(building_type)
    payload = ("build_mode", building_type)
    budget_reason = _turn_budget_reason(env)
    if budget_reason:
        return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type, enabled=False, reason=budget_reason)

    spec = production.building_stats(env, faction, building_type)
    if spec is None:
        return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type, enabled=False, reason="Unavailable.")
    state = env.faction_state(faction)
    if state.resources < spec.cost:
        return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type, enabled=False, reason=f"Costs {spec.cost}.")
    if spec.required_tech and spec.required_tech not in state.techs_unlocked:
        return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type, enabled=False, reason=f"Needs {_tech_label(spec.required_tech)}.")
    if spec.required_building and not any(
        building.building_type == spec.required_building for building in env.get_buildings_for_faction(faction)
    ):
        return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type, enabled=False, reason=f"Needs {_building_label(spec.required_building)}.")
    if spec.required_resource_adjacent:
        has_resource_target = any(
            production.can_build(env, faction, worker.id, building_type, pos) for pos in hexgrid.neighbors(worker.position)
        )
        if not has_resource_target:
            resource_label = RESOURCE_LABELS.get(spec.required_resource_adjacent, spec.required_resource_adjacent.title())
            return HumanActionOption(
                label=label,
                payload=payload,
                active=pending_build_type == building_type,
                enabled=False,
                reason=f"Needs adjacent {resource_label}.",
            )
    targets = _human_build_targets(env, worker, building_type)
    if not targets:
        return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type, enabled=False, reason="No adjacent build tile.")
    return HumanActionOption(label=label, payload=payload, active=pending_build_type == building_type)


def _disabled_option_hint(options: list[HumanActionOption]) -> str | None:
    disabled = [f"{option.label}: {option.reason}" for option in options if not option.enabled and option.reason]
    if not disabled:
        return None
    preview = disabled[:3]
    return "Unavailable: " + "  ".join(preview)


def _human_action_options(env: AgeGridEnv, faction: str, selected_unit, selected_tile, pending_build_type: str | None) -> tuple[str | None, list[HumanActionOption], str | None]:
    options: list[HumanActionOption] = []
    hint: str | None = None

    if selected_unit is not None and selected_unit.faction == faction:
        if selected_unit.unit_type == "worker":
            gather_reason = _turn_budget_reason(env)
            if gather_reason is None and env.resource_at_for_faction(selected_unit.position, selected_unit.faction) is None:
                gather_reason = "Stand on a resource tile."
            options.append(
                HumanActionOption(
                    label="Gather",
                    payload=("gather", selected_unit.id),
                    enabled=gather_reason is None,
                    reason=gather_reason,
                )
            )
            for building_type in production.BUILDING_DEFS:
                options.append(_build_option(env, faction, selected_unit, building_type, pending_build_type))
            enabled_options = [option for option in options if option.enabled]
            if not enabled_options:
                hint = "No worker actions available on this tile."
            elif pending_build_type is not None:
                hint = f"Click a highlighted adjacent hex to place {_building_label(pending_build_type)}."
            disabled_hint = _disabled_option_hint(options)
            if disabled_hint:
                hint = f"{hint}  {disabled_hint}" if hint else disabled_hint
            return "Worker Actions", options, hint

        return "Unit Actions", options, "Select a worker or base to access build and recruit commands."

    if selected_tile is not None:
        selected_base = next(((base_faction, base) for base_faction, base in env.bases.items() if base.position == selected_tile), None)
        if selected_base is not None and selected_base[0] == faction:
            options.append(_train_option(env, faction, "worker"))
            for unit_type in production.UNIT_DEFS:
                if unit_type == "worker":
                    continue
                options.append(_train_option(env, faction, unit_type))
            if not any(option.enabled for option in options):
                hint = "No recruits available right now."
            disabled_hint = _disabled_option_hint(options)
            if disabled_hint:
                hint = f"{hint}  {disabled_hint}" if hint else disabled_hint
            return "Base Actions", options, hint

    return None, [], None


def _human_research_options(env: AgeGridEnv, faction: str) -> tuple[str, list[tuple[str, object, bool]], str | None]:
    state = env.faction_state(faction)
    if state.tech_in_progress is not None:
        turns_left = tech.research_turns_remaining(env, faction)
        hint = f"Researching {_tech_label(state.tech_in_progress)}. {turns_left} turn{'s' if turns_left != 1 else ''} left."
        return "Research", [], hint

    options: list[tuple[str, object, bool]] = []
    for tech_id, spec in tech.TECH_DEFS.items():
        if tech.can_research(env, faction, tech_id):
            options.append((f"{_tech_label(tech_id)} ({spec.cost})", ("research", tech_id), False))

    hint = None if options else "No research currently available."
    return "Research", options, hint


def _available_research_ids(env: AgeGridEnv, faction: str) -> list[str]:
    return [tech_id for tech_id in TECH_TREE_ORDER if tech.can_research(env, faction, tech_id)]


def _tech_detail_lines(env: AgeGridEnv, faction: str, tech_id: str) -> list[str]:
    definition = tech.TECH_DEFS[tech_id]
    state = env.faction_state(faction)
    if tech_id in state.techs_unlocked:
        status = "Unlocked"
    elif state.tech_in_progress == tech_id:
        status = f"In progress: {tech.research_turns_remaining(env, faction)} turns left"
    elif tech.can_research(env, faction, tech_id):
        status = "Available now"
    else:
        missing = [_tech_label(req) for req in definition.requires if req not in state.techs_unlocked]
        status = f"Requires: {', '.join(missing)}" if missing else "Locked"
    unlocks = _tech_unlock_summary(tech_id)
    return [
        definition.summary,
        f"Cost: {definition.cost}",
        f"Turns: {definition.turns}",
        status,
        f"Unlocks: {unlocks}",
    ]


def _draw_human_action_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    title: str,
    options: list[HumanActionOption],
    hint: str | None,
) -> list[tuple[pygame.Rect, object, bool]]:
    inner = _draw_parchment_panel_frame(surface, board_assets, rect)
    y = _draw_parchment_header(surface, title_font, body_font, inner, title)

    button_rects: list[tuple[pygame.Rect, object, bool]] = []
    columns = 2
    button_w = (inner.width - 18) // columns
    button_h = 30
    for idx, option in enumerate(options):
        col = idx % columns
        row = idx // columns
        button_rect = pygame.Rect(inner.x + 6 + col * (button_w + 6), y + row * (button_h + 8), button_w, button_h)
        _draw_parchment_button(surface, body_font, board_assets, button_rect, option.label, active=option.active, enabled=option.enabled)
        button_rects.append((button_rect, option.payload, option.enabled))

    if hint:
        hint_y = y + ((len(options) + 1) // columns) * (button_h + 8) + 4
        wrapped = _wrap_lines(hint, body_font, inner.width - 12)
        _draw_text_block(surface, body_font, wrapped, inner.x + 6, hint_y, PARCH_MUTED, 18)

    return button_rects


def _human_action_panel_height(body_font: pygame.font.Font, options: list[HumanActionOption], hint: str | None) -> int:
    columns = 2
    button_h = 30
    rows = (len(options) + columns - 1) // columns
    height = 74 + rows * (button_h + 8)
    if hint:
        height += len(_wrap_lines(hint, body_font, 252)) * 16 + 8
    return max(106, height)


def _advance_until_human_or_end(
    env: AgeGridEnv,
    red_agent,
    blue_agent,
    red_index: int,
    blue_index: int,
    red_info: FactionTurnInfo,
    blue_info: FactionTurnInfo,
) -> tuple[FactionTurnInfo, FactionTurnInfo, list[TurnSnapshot], list[VisualEffect]]:
    completed_rounds: list[TurnSnapshot] = []
    effects: list[VisualEffect] = []

    while env.winner() is None:
        current_faction = env.factions[env.current_player]
        if _is_human_agent_key(_current_agent_key(env, red_index, blue_index)):
            break

        agent = red_agent if current_faction == "Red" else blue_agent
        info, actions = _step_faction_with_trace(env, agent)
        effects.extend(_effects_from_actions(env, actions, current_faction))

        if current_faction == "Red":
            red_info = info
        else:
            blue_info = info

        env.step_end_turn()
        if env.current_player == 0:
            completed_rounds.append(TurnSnapshot(env.turn, red_info, blue_info))

    return red_info, blue_info, completed_rounds, effects


def _wrap_lines(text: str, font: pygame.font.Font, width: int) -> list[str]:
    if not text:
        return [""]

    words = text.split()
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_text(text: str, font: pygame.font.Font, width: int) -> str:
    if font.size(text)[0] <= width:
        return text
    trimmed = text
    while trimmed and font.size(f"{trimmed}...")[0] > width:
        trimmed = trimmed[:-1]
    return f"{trimmed}..." if trimmed else "..."


def _draw_text_block(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    x: int,
    y: int,
    color: tuple[int, int, int],
    line_height: int = 22,
) -> None:
    for idx, line in enumerate(lines):
        surface.blit(font.render(line, True, color), (x, y + idx * line_height))


def _draw_shadow_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    shadow: tuple[int, int, int] = (0, 0, 0),
    shadow_offset: int = 2,
) -> pygame.Rect:
    shadow_surface = font.render(text, True, shadow)
    text_surface = font.render(text, True, color)
    surface.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))
    surface.blit(text_surface, (x, y))
    return text_surface.get_rect(topleft=(x, y))


def _draw_labeled_block(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    title: str,
    body_lines: list[str],
    x: int,
    y: int,
    width: int,
    title_color: tuple[int, int, int],
    body_color: tuple[int, int, int],
) -> int:
    _draw_shadow_text(surface, title_font, title, x, y, title_color, shadow=(10, 12, 16))
    wrapped: list[str] = []
    for line in body_lines:
        wrapped.extend(_wrap_lines(line, body_font, width))
    _draw_text_block(surface, body_font, wrapped, x, y + 22, body_color, 18)
    return y + 22 + len(wrapped) * 18


def _draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fill: tuple[int, int, int] = PANEL_BG,
    border: tuple[int, int, int] = PANEL_BORDER,
    radius: int = 14,
) -> None:
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=radius)


def _draw_faction_bar(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    faction: str,
    agent_label: str,
    bank: int,
    workers: int,
    military: int,
    base_hp: int,
    max_base_hp: int,
    era: str,
    stance: str,
    accent: tuple[int, int, int],
) -> None:
    """Slim faction summary bar for the top HUD (Red left, Blue right)."""
    panel_key = "panel_blue" if faction == "Blue" else "panel_brown"
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(panel_key), rect):
        _draw_panel(surface, rect, fill=PANEL_BG, border=accent, radius=12)

    # Thin faction-accent bar at the top edge of the panel
    accent_bar = pygame.Rect(rect.x + 3, rect.y + 3, rect.width - 6, 4)
    pygame.draw.rect(surface, accent, accent_bar, border_radius=6)

    ix = rect.x + HUD_INNER_PAD
    iw = rect.width - HUD_INNER_PAD * 2
    y = rect.y + HUD_INNER_PAD + 4

    # Row 1: faction name + era (right-aligned)
    _draw_shadow_text(surface, body_font, faction, ix, y, TEXT_PRIMARY, shadow=(10, 12, 16), shadow_offset=1)
    era_w = small_font.size(era)[0]
    _draw_shadow_text(surface, small_font, era, rect.right - era_w - HUD_INNER_PAD, y + 3, TEXT_SECONDARY, shadow=(10, 12, 16), shadow_offset=1)
    y += body_font.get_height() + 5

    # Row 2: three stat chips  [$]  [W]  [M]
    stat_items = [
        (f"$ {bank}", PARCH_TITLE),
        (f"W {workers}  M {military}", PARCH_BODY),
        (stance, PARCH_MUTED),
    ]
    chip_gap = 4
    chip_w = (iw - chip_gap * 2) // 3
    chip_h = 26
    inset_sprite = board_assets.ui_sprite("panelInset_beigeLight") or board_assets.ui_sprite("panelInset_beige")
    for i, (text, text_color) in enumerate(stat_items):
        chip_rect = pygame.Rect(ix + i * (chip_w + chip_gap), y, chip_w, chip_h)
        if not _draw_scaled_sprite(surface, inset_sprite, chip_rect):
            pygame.draw.rect(surface, PANEL_INSET, chip_rect, border_radius=6)
            pygame.draw.rect(surface, PANEL_SOFT, chip_rect, width=1, border_radius=6)
        label = _fit_text(text, small_font, chip_w - 8)
        tx = chip_rect.centerx - small_font.size(label)[0] // 2
        _draw_shadow_text(surface, small_font, label, tx, chip_rect.y + 5, text_color, shadow=PARCH_SHADOW, shadow_offset=0)
    y += chip_h + 6

    # Row 3: base HP bar
    hp_text = f"Base {base_hp}/{max_base_hp}"
    _draw_shadow_text(surface, small_font, hp_text, ix, y + 2, TEXT_SECONDARY, shadow=(10, 12, 16), shadow_offset=1)
    bar_x = ix + small_font.size(hp_text)[0] + 6
    bar_w = rect.right - HUD_INNER_PAD - bar_x
    if bar_w > 16:
        bar_rect = pygame.Rect(bar_x, y + 4, bar_w, 14)
        color_family = "blue" if faction == "Blue" else "red"
        _draw_ui_meter(surface, board_assets, bar_rect, base_hp, max_base_hp, color_family)


def _draw_center_hud(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    btn_rect: pygame.Rect,
    env: AgeGridEnv,
    winner: str | None,
    human_turn_active: bool,
    btn_label: str,
    current_faction: str,
) -> None:
    """Central top-HUD zone: turn info, round flow, action budget, end-turn button."""
    panel_key = "panel_beigeLight"
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(panel_key), rect):
        _draw_panel(surface, rect, fill=(20, 27, 36), border=PANEL_SOFT, radius=12)

    ix = rect.x + HUD_INNER_PAD
    y = rect.y + HUD_INNER_PAD

    # Left: "TURN" label + big number  |  Right: button (drawn at btn_rect, pre-computed)
    label_turn = "TURN"
    lw = small_font.size(label_turn)[0]
    turn_str = str(env.turn)
    _draw_shadow_text(surface, small_font, label_turn, ix, y + 6, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
    _draw_shadow_text(surface, title_font, turn_str, ix + lw + 6, y, GOLD_ACCENT, shadow=PARCH_SHADOW, shadow_offset=0)
    y += title_font.get_height() + 4

    # Row 2: flow arrow (left-aligned) + action budget
    if winner is not None:
        result_text = f"{winner} wins!"
        _draw_shadow_text(surface, body_font, result_text, ix, y, GOLD_ACCENT, shadow=PARCH_SHADOW, shadow_offset=0)
    else:
        flow_color = (180, 90, 84) if current_faction == "Red" else (88, 126, 212)
        other = "Blue" if current_faction == "Red" else "Red"
        arrow = f"{current_faction}  \u2192  {other}"
        _draw_shadow_text(surface, body_font, arrow, ix, y, flow_color, shadow=PARCH_SHADOW, shadow_offset=0)
        if human_turn_active:
            y += body_font.get_height() + 2
            budget = f"Atk {env.actions_left}  Att {env.attempts_left}"
            _draw_shadow_text(surface, small_font, budget, ix, y, PARCH_BODY, shadow=PARCH_SHADOW, shadow_offset=0)

    # End-turn / Next-turn button (uses blue button asset)
    btn_fill = (52, 82, 124) if winner is None else (58, 62, 68)
    btn_border = (120, 172, 234) if winner is None else (90, 96, 108)
    btn_sprite_key = "button_blue" if winner is None else "button_grey"
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(btn_sprite_key), btn_rect):
        _draw_panel(surface, btn_rect, fill=btn_fill, border=btn_border, radius=10)
    display_label = btn_label if winner is None else "Game Over"
    _draw_shadow_text(
        surface,
        small_font,
        display_label,
        btn_rect.centerx - small_font.size(display_label)[0] // 2,
        btn_rect.y + (btn_rect.height - small_font.get_height()) // 2,
        PARCH_TITLE if winner is None else PARCH_MUTED,
        shadow=PARCH_SHADOW,
        shadow_offset=0,
    )


def _event_color(line: str) -> tuple[int, int, int]:
    if line.startswith("Red "):
        return (244, 184, 180)
    if line.startswith("Blue "):
        return (176, 214, 255)
    return (214, 220, 226)


def _draw_event_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    lines: list[str],
    rect: pygame.Rect,
    scroll_offset: int = 0,
) -> None:
    inner = _draw_parchment_panel_frame(surface, board_assets, rect)
    total_rows = len(lines if lines else ["No events yet"])
    note = f"{min(total_rows, max(0, scroll_offset) + 5)}/{total_rows}" if total_rows > 5 else None
    y = _draw_parchment_header(surface, title_font, body_font, inner, "Events", note=note)
    max_width = inner.width - 18
    line_height = 19
    event_lines = lines if lines else ["No events yet"]
    wrapped_entries: list[tuple[list[str], tuple[int, int, int]]] = []
    for entry in event_lines:
        wrapped = _wrap_lines(entry, body_font, max_width)
        color = _event_color(entry)
        wrapped_entries.append((wrapped, color))
    visible_height = inner.bottom - y - 8
    cursor = 0
    start = max(0, scroll_offset)
    for wrapped, color in wrapped_entries[start:]:
        block_h = len(wrapped) * line_height + 10
        if cursor + block_h > visible_height:
            break
        _draw_text_block(surface, body_font, wrapped, inner.x + 8, y + cursor, color, line_height)
        cursor += block_h


def _draw_research_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    rect: pygame.Rect,
) -> None:
    inner = _draw_parchment_panel_frame(surface, board_assets, rect)
    y = _draw_parchment_header(surface, title_font, body_font, inner, "Research")
    max_width = inner.width - 16
    for faction, color in (("Red", (244, 184, 180)), ("Blue", (176, 214, 255))):
        state = env.faction_state(faction)
        available = [
            tech_id
            for tech_id in tech.TECH_DEFS
            if tech.can_research(env, faction, tech_id)
        ]
        available_labels = [_tech_label(tech_id) for tech_id in available]
        if state.tech_in_progress:
            turns_left = tech.research_turns_remaining(env, faction)
            current_definition = tech.TECH_DEFS[state.tech_in_progress]
            total_turns = current_definition.turns
            progress_ratio = min(1.0, state.research_points / total_turns) if total_turns > 0 else 1.0
            status = _tech_label(state.tech_in_progress)
            eta = f"{turns_left} turn{'s' if turns_left != 1 else ''} left"
            summary = current_definition.summary
        else:
            progress_ratio = 0.0
            status = "None"
            eta = None
            summary = tech.TECH_DEFS[available[0]].summary if available else "No research currently available."
        icon_key = state.tech_in_progress or (available[0] if available else None)
        icon_style = TECH_ICON_STYLES.get(icon_key or "", {"label": "?", "bg": (70, 76, 84), "fg": (232, 236, 240)})
        subsection_h = _research_subsection_height(body_font, faction, status, eta, summary, available_labels, max_width)
        subsection_rect = pygame.Rect(inner.x + 8, y, inner.width - 16, subsection_h)
        inset_sprite = board_assets.ui_sprite("panelInset_beigeLight") or board_assets.ui_sprite("panelInset_beige")
        if not _draw_scaled_sprite(surface, inset_sprite, subsection_rect):
            pygame.draw.rect(surface, PANEL_INSET, subsection_rect, border_radius=8)
            pygame.draw.rect(surface, PANEL_SOFT, subsection_rect, width=1, border_radius=8)

        inner_x = subsection_rect.x + 10
        inner_y = subsection_rect.y + 10
        inner_width = subsection_rect.width - 20
        icon_rect = pygame.Rect(inner_x, inner_y, 24, 24)
        pygame.draw.rect(surface, icon_style["bg"], icon_rect, border_radius=6)
        pygame.draw.rect(surface, (236, 240, 244), icon_rect, width=1, border_radius=6)
        _draw_shadow_text(
            surface,
            body_font,
            icon_style["label"],
            icon_rect.x + 6,
            icon_rect.y + 2,
            icon_style["fg"],
            shadow=(18, 22, 28),
            shadow_offset=1,
        )
        header_lines = _wrap_lines(f"{faction}: {status}", body_font, inner_width - 30)
        _draw_text_block(surface, body_font, header_lines, inner_x + 32, inner_y, color, 19)
        content_y = inner_y + max(26, len(header_lines) * 19) + 8

        detail_lines = [summary, f"Next: {', '.join(available_labels[:3]) if available_labels else '-'}"]
        if eta:
            detail_lines.insert(0, eta)
        for line in detail_lines:
            wrapped = _wrap_lines(line, body_font, inner_width)
            _draw_text_block(surface, body_font, wrapped, inner_x, content_y, PARCH_BODY, 19)
            content_y += len(wrapped) * 19
            content_y += 5
        bar_rect = pygame.Rect(inner_x, content_y + 6, inner_width, 12)
        pygame.draw.rect(surface, (155, 129, 89), bar_rect, border_radius=5)
        if progress_ratio > 0:
            fill_width = max(8, int(bar_rect.width * progress_ratio))
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, min(fill_width, bar_rect.width), bar_rect.height)
            pygame.draw.rect(surface, color, fill_rect, border_radius=5)
        pygame.draw.rect(surface, (124, 98, 65), bar_rect, width=1, border_radius=5)
        y = subsection_rect.bottom + 12


def _draw_status_badge(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    fill: tuple[int, int, int],
    border: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    badge = pygame.Rect(x, y, max(108, font.size(text)[0] + 20), 24)
    pygame.draw.rect(surface, fill, badge, border_radius=10)
    pygame.draw.rect(surface, border, badge, width=1, border_radius=10)
    _draw_shadow_text(surface, font, text, badge.x + 10, badge.y + 3, text_color, shadow=(14, 16, 20), shadow_offset=1)


def _research_panel_height(
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    env: AgeGridEnv,
    width: int,
) -> int:
    # 18 = outer padding (inflate -18,-18 adds 9 each side); header = 22 + title_font height
    header_h = 22 + title_font.get_height()
    total = 18 + header_h
    subsection_width = width - 34  # inner.width - 16, where inner = outer.inflate(-18,-18)
    for faction in ("Red", "Blue"):
        state = env.faction_state(faction)
        available = [
            tech_id
            for tech_id in tech.TECH_DEFS
            if tech.can_research(env, faction, tech_id)
        ]
        available_labels = [_tech_label(tech_id) for tech_id in available]
        if state.tech_in_progress:
            turns_left = tech.research_turns_remaining(env, faction)
            current_def = tech.TECH_DEFS[state.tech_in_progress]
            status = _tech_label(state.tech_in_progress)
            eta = f"{turns_left} turn{'s' if turns_left != 1 else ''} left"
            summary = current_def.summary
        else:
            status = "None"
            eta = None
            summary = tech.TECH_DEFS[available[0]].summary if available else "No research currently available."
        total += _research_subsection_height(body_font, faction, status, eta, summary, available_labels, subsection_width)
        total += 12
    return total


def _research_subsection_height(
    body_font: pygame.font.Font,
    faction: str,
    status: str,
    eta: str | None,
    summary: str,
    available_labels: list[str],
    width: int,
) -> int:
    # Mirror _draw_research_panel exactly:
    # inner_width = subsection.width - 20
    inner_width = width - 20
    header_lines = _wrap_lines(f"{faction}: {status}", body_font, inner_width - 30)
    # 10 top-pad + header zone + 8 gap after header
    total = 10 + max(26, len(header_lines) * 19) + 8
    # detail blocks: [eta (opt), summary, "Next: ..."], each adds lines*19 + 5
    detail = [summary, f"Next: {', '.join(available_labels[:3]) if available_labels else '-'}"]
    if eta:
        detail.insert(0, eta)
    for line in detail:
        total += len(_wrap_lines(line, body_font, inner_width)) * 19 + 5
    # gap + progress bar + bottom padding
    total += 6 + 12 + 10
    return total


def _tactical_panel_lines(env: AgeGridEnv, faction: str) -> list[str]:
    composition = unit_composition(env, faction)
    mode = "Defense" if defense_mode_active(env, faction) else "Push" if push_mode_active(env, faction) else "Field"
    enemy = next(name for name in env.factions if name != faction)
    relation = env.relation_state(faction, enemy).state.title()
    return [
        f"Diplomacy {relation}",
        f"Threat {threat_level(env, faction)}",
        f"Plan {army_plan(env, faction)} | {mode}",
        f"Army W{composition['worker']} S{composition['soldier']} A{composition['archer']} H{composition['horseman']}",
    ]


def _tactical_panel_height(title_font: pygame.font.Font, body_font: pygame.font.Font, env: AgeGridEnv, width: int) -> int:
    # Mirror _draw_tactical_panel exactly:
    # 18 outer padding; header = 22 + title_font height (from _draw_parchment_header)
    header_h = 22 + title_font.get_height()
    total = 18 + header_h
    # block width = inner.width - 16, inner = outer.inflate(-18,-18)
    block_width = width - 34
    for faction in ("Red", "Blue"):
        lines = _tactical_panel_lines(env, faction)
        # block_rect.height = 14 + 20 + sum(lines*18) + 8, then gap of 10
        block_h = 42 + sum(len(_wrap_lines(line, body_font, block_width - 18)) * 18 for line in lines)
        total += block_h + 10
    return total


def _draw_tactical_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    rect: pygame.Rect,
) -> None:
    inner = _draw_parchment_panel_frame(surface, board_assets, rect)
    y = _draw_parchment_header(surface, title_font, body_font, inner, "Tactical")
    for faction, color in (("Red", (244, 184, 180)), ("Blue", (176, 214, 255))):
        block_rect = pygame.Rect(inner.x + 8, y, inner.width - 16, 68)
        lines = _tactical_panel_lines(env, faction)
        block_rect.height = 14 + 20 + sum(len(_wrap_lines(line, body_font, block_rect.width - 18)) * 18 for line in lines) + 8
        inset_sprite = board_assets.ui_sprite("panelInset_beigeLight") or board_assets.ui_sprite("panelInset_beige")
        if not _draw_scaled_sprite(surface, inset_sprite, block_rect):
            pygame.draw.rect(surface, PANEL_INSET, block_rect, border_radius=8)
            pygame.draw.rect(surface, PANEL_SOFT, block_rect, width=1, border_radius=8)
        _draw_shadow_text(surface, body_font, faction, block_rect.x + 10, block_rect.y + 8, color, shadow=PARCH_SHADOW, shadow_offset=0)
        line_y = block_rect.y + 28
        for line in _tactical_panel_lines(env, faction):
            wrapped = _wrap_lines(line, body_font, block_rect.width - 20)
            _draw_text_block(surface, body_font, wrapped, block_rect.x + 10, line_y, PARCH_BODY, 18)
            line_y += len(wrapped) * 18
        y = block_rect.bottom + 10


def _draw_small_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    active: bool = False,
    enabled: bool = True,
) -> None:
    if not enabled:
        fill = (34, 39, 47)
        border = (72, 79, 90)
        text_color = TEXT_MUTED
    else:
        fill = (58, 79, 108) if active else PANEL_INSET
        border = (132, 176, 232) if active else PANEL_SOFT
        text_color = TEXT_PRIMARY
    _draw_panel(surface, rect, fill=fill, border=border, radius=10)
    _draw_shadow_text(
        surface,
        font,
        label,
        rect.centerx - font.size(label)[0] // 2,
        rect.y + 6,
        text_color,
        shadow=(10, 12, 16),
        shadow_offset=1,
    )


def _draw_parchment_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    label: str,
    active: bool = False,
    enabled: bool = True,
) -> None:
    sprite = board_assets.ui_sprite("button")
    if sprite is not None:
        _draw_scaled_sprite(surface, sprite, rect)
    else:
        _draw_panel(surface, rect, fill=(159, 118, 76), border=(210, 174, 123), radius=10)

    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    if not enabled:
        overlay.fill((70, 74, 82, 120))
    elif active:
        overlay.fill((103, 142, 191, 60))
    surface.blit(overlay, rect.topleft)

    text_color = (74, 61, 47) if enabled else (120, 116, 110)
    _draw_shadow_text(
        surface,
        font,
        label,
        rect.centerx - font.size(label)[0] // 2,
        rect.y + 6,
        text_color,
        shadow=(240, 228, 206) if enabled else (200, 192, 180),
        shadow_offset=0,
    )


def _draw_parchment_close_button(
    surface: pygame.Surface,
    board_assets: BoardAssets,
    rect: pygame.Rect,
) -> pygame.Rect:
    sprite = board_assets.ui_sprite("button_grey")
    if not _draw_scaled_sprite(surface, sprite, rect):
        _draw_panel(surface, rect, fill=(186, 177, 160), border=(136, 122, 102), radius=8)

    close_sprite = board_assets.ui_sprite("close")
    close_inner = rect.inflate(-6, -6)
    if not _draw_scaled_sprite(surface, close_sprite, close_inner):
        pygame.draw.line(surface, PARCH_TITLE, (close_inner.x + 3, close_inner.y + 3), (close_inner.right - 3, close_inner.bottom - 3), 2)
        pygame.draw.line(surface, PARCH_TITLE, (close_inner.right - 3, close_inner.y + 3), (close_inner.x + 3, close_inner.bottom - 3), 2)
    return rect


def _draw_parchment_header(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    inner: pygame.Rect,
    title: str,
    subtitle: str | None = None,
    note: str | None = None,
) -> int:
    title_y = inner.y + 10
    _draw_shadow_text(surface, title_font, title, inner.x + 12, title_y, PARCH_TITLE, shadow=PARCH_SHADOW, shadow_offset=0)
    if note:
        note_w = body_font.size(note)[0]
        _draw_shadow_text(surface, body_font, note, inner.right - note_w - 10, title_y + 2, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
    y = title_y + title_font.get_height()
    if subtitle:
        _draw_shadow_text(surface, body_font, subtitle, inner.x + 12, y - 2, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
        y += body_font.get_height() + 4
    divider_y = y + 2
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 10, divider_y), (inner.right - 10, divider_y), 1)
    return divider_y + 10


def _draw_parchment_chip(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label_font: pygame.font.Font,
    value_font: pygame.font.Font,
    label: str,
    value: str,
    value_color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(surface, (166, 129, 88), rect, border_radius=10)
    pygame.draw.rect(surface, (209, 177, 134), rect, width=2, border_radius=10)
    _draw_shadow_text(surface, label_font, label, rect.x + 10, rect.y + 6, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
    _draw_shadow_text(surface, value_font, value, rect.x + 10, rect.y + 22, value_color, shadow=PARCH_SHADOW, shadow_offset=0)


def _draw_badge(
    surface: pygame.Surface,
    font: pygame.font.Font,
    center: tuple[int, int],
    label: str,
    fill: tuple[int, int, int] = (188, 74, 68),
    border: tuple[int, int, int] = (246, 212, 188),
) -> pygame.Rect:
    radius = max(11, font.size(label)[0] // 2 + 7)
    badge_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    pygame.draw.circle(surface, fill, badge_rect.center, radius)
    pygame.draw.circle(surface, border, badge_rect.center, radius, width=2)
    _draw_shadow_text(
        surface,
        font,
        label,
        badge_rect.centerx - font.size(label)[0] // 2,
        badge_rect.centery - font.get_height() // 2 + 1,
        TEXT_PRIMARY,
        shadow=(10, 12, 16),
        shadow_offset=1,
    )
    return badge_rect


def _draw_scaled_sprite(surface: pygame.Surface, sprite: pygame.Surface | None, rect: pygame.Rect) -> bool:
    if sprite is None:
        return False
    scaled = _safe_scale(sprite, rect.size)
    if scaled is None:
        return False
    surface.blit(scaled, rect.topleft)
    return True


def _draw_ui_meter(
    surface: pygame.Surface,
    assets: BoardAssets,
    rect: pygame.Rect,
    value: int,
    maximum: int,
    color_family: str = "red",
) -> None:
    maximum = max(1, maximum)
    fill_ratio = max(0.0, min(1.0, value / maximum))
    back_rect = rect.copy()
    if not (
        _draw_scaled_sprite(surface, assets.ui_sprite("bar_back_left"), pygame.Rect(back_rect.x, back_rect.y, 8, back_rect.height))
        and _draw_scaled_sprite(surface, assets.ui_sprite("bar_back_mid"), pygame.Rect(back_rect.x + 8, back_rect.y, max(1, back_rect.width - 16), back_rect.height))
        and _draw_scaled_sprite(surface, assets.ui_sprite("bar_back_right"), pygame.Rect(back_rect.right - 8, back_rect.y, 8, back_rect.height))
    ):
        pygame.draw.rect(surface, (155, 129, 89), back_rect, border_radius=9)
        pygame.draw.rect(surface, (124, 98, 65), back_rect, width=2, border_radius=9)

    fill_width = max(14, round((rect.width - 8) * fill_ratio))
    fill_rect = pygame.Rect(rect.x + 4, rect.y + 3, min(fill_width, rect.width - 8), rect.height - 6)
    left = assets.ui_sprite(f"bar_{color_family}_left")
    mid = assets.ui_sprite(f"bar_{color_family}_mid")
    right = assets.ui_sprite(f"bar_{color_family}_right")
    if not (
        _draw_scaled_sprite(surface, left, pygame.Rect(fill_rect.x, fill_rect.y, 8, fill_rect.height))
        and _draw_scaled_sprite(surface, mid, pygame.Rect(fill_rect.x + 8, fill_rect.y, max(1, fill_rect.width - 16), fill_rect.height))
        and _draw_scaled_sprite(surface, right, pygame.Rect(fill_rect.right - 8, fill_rect.y, 8, fill_rect.height))
    ):
        fill_color = (232, 100, 58) if color_family == "red" else (84, 175, 224) if color_family == "blue" else (224, 184, 76)
        pygame.draw.rect(surface, fill_color, fill_rect, border_radius=8)


def _selected_unit_lines(env: AgeGridEnv, unit) -> list[str]:
    spec = production.unit_stats(env, unit.faction, unit.unit_type)
    label = UNIT_LABELS.get(unit.unit_type, unit.unit_type.replace("_", " ").title())
    description = UNIT_HELP.get(unit.unit_type, "Military unit.")
    if spec is None:
        return [label, description]
    return [
        label,
        description,
        f"Position {unit.position[0]}, {unit.position[1]}",
    ]


def _selected_resource_lines(resource) -> list[str]:
    label = RESOURCE_LABELS.get(resource.resource_type, resource.resource_type.replace("_", " ").title())
    help_text = RESOURCE_HELP.get(resource.resource_type, "Strategic resource node.")
    extra = "Visible once its required tech is unlocked." if resource.required_tech else "Available to gather immediately."
    return [
        label,
        help_text,
        extra,
        f"Remaining {resource.remaining}",
    ]


def _selected_building_lines(env: AgeGridEnv, building) -> list[str]:
    label = _building_label(building.building_type)
    help_text = BUILDING_HELP.get(building.building_type, "Faction structure.")
    spec = production.building_stats(env, building.faction, building.building_type)
    extras: list[str] = [help_text]
    if spec is not None and spec.resource_income > 0:
        extras.append(f"Income +{spec.resource_income} each turn")
    if spec is not None and spec.attack_damage > 0:
        extras.append(f"Attack {spec.attack_damage}  Range {spec.attack_range}")
    extras.append(f"Position {building.position[0]}, {building.position[1]}")
    return [label, *extras]


def _selected_base_lines(base, faction: str) -> list[str]:
    return [
        f"{faction} Base",
        "Primary stronghold. Lose this and the match is over.",
        f"Position {base.position[0]}, {base.position[1]}",
    ]


def _draw_selected_unit_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    unit,
    rect: pygame.Rect,
) -> pygame.Rect:
    accent = RED_PRIMARY if unit.faction == "Red" else BLUE_PRIMARY
    inner = _draw_parchment_panel_frame(surface, board_assets, rect)

    close_rect = _draw_parchment_close_button(surface, board_assets, pygame.Rect(inner.right - 26, inner.y + 6, 22, 22))

    icon_center = (inner.x + 38, inner.y + 38)
    pygame.draw.circle(surface, (*accent, 48), icon_center, 24)
    if not _draw_unit_sprite(
        surface,
        board_assets,
        env,
        unit,
        icon_center,
        RED_ACCENT if unit.faction == "Red" else BLUE_ACCENT,
        accent,
        size=34,
    ):
        _draw_unit_icon(surface, unit, icon_center, RED_ACCENT if unit.faction == "Red" else BLUE_ACCENT, accent)

    unit_name = UNIT_LABELS.get(unit.unit_type, unit.unit_type.replace("_", " ").title())
    _draw_shadow_text(surface, title_font, unit_name, inner.x + 72, inner.y + 10, PARCH_TITLE, shadow=PARCH_SHADOW, shadow_offset=0)
    _draw_shadow_text(
        surface,
        body_font,
        f"{unit.faction} #{unit.id}",
        inner.x + 72,
        inner.y + 36,
        accent,
        shadow=PARCH_SHADOW,
        shadow_offset=0,
    )
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 12, inner.y + 64), (inner.right - 12, inner.y + 64), 1)

    spec = production.unit_stats(env, unit.faction, unit.unit_type)
    max_hp = spec.hp if spec is not None else unit.hp
    health_rect = pygame.Rect(inner.x + 14, inner.y + 84, inner.width - 28, 24)
    _draw_shadow_text(surface, body_font, f"Health {unit.hp}/{max_hp}", health_rect.x, health_rect.y - 18, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
    _draw_ui_meter(surface, board_assets, health_rect, unit.hp, max_hp, color_family="red")

    stat_y = health_rect.bottom + 20
    attack = spec.attack_damage if spec is not None else unit.attack_damage
    attack_range = spec.attack_range if spec is not None else unit.attack_range
    move_steps = spec.move_steps if spec is not None else unit.move_steps
    stats = [
        ("Attack", str(attack), (238, 163, 92)),
        ("Range", str(attack_range), (129, 190, 228)),
        ("Move", str(move_steps), (165, 213, 122)),
    ]
    stat_w = (inner.width - 28 - 10) // 3
    for idx, (label, value, color) in enumerate(stats):
        stat_rect = pygame.Rect(inner.x + 14 + idx * (stat_w + 5), stat_y, stat_w, 50)
        _draw_parchment_chip(surface, stat_rect, small_font, title_font, label, value, color)

    desc_y = stat_y + 64
    body_lines = _selected_unit_lines(env, unit)[1:]
    wrapped: list[str] = []
    for line in body_lines:
        wrapped.extend(_wrap_lines(line, body_font, inner.width - 28))
    _draw_text_block(surface, body_font, wrapped, inner.x + 14, desc_y, PARCH_BODY, 19)
    return close_rect


def _draw_selected_object_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    rect: pygame.Rect,
    *,
    title: str,
    subtitle: str,
    lines: list[str],
    hp_value: int | None = None,
    hp_max: int | None = None,
    icon_kind: str | None = None,
    icon_drawer=None,
    accent: tuple[int, int, int] = TEXT_PRIMARY,
) -> pygame.Rect:
    inner = _draw_parchment_panel_frame(surface, board_assets, rect)

    close_rect = _draw_parchment_close_button(surface, board_assets, pygame.Rect(inner.right - 26, inner.y + 6, 22, 22))

    icon_center = (inner.x + 38, inner.y + 38)
    pygame.draw.circle(surface, (*accent, 48), icon_center, 24)
    drew_icon = False
    if icon_kind is not None:
        sprite = board_assets.object_sprite(icon_kind)
        if sprite is not None:
            scaled = _safe_scale(sprite, (36, 36))
            if scaled is not None:
                screen_rect = scaled.get_rect(center=(icon_center[0], icon_center[1] + 2))
                surface.blit(scaled, screen_rect)
                drew_icon = True
    if not drew_icon and icon_drawer is not None:
        icon_drawer(icon_center)

    _draw_shadow_text(surface, title_font, title, inner.x + 72, inner.y + 10, PARCH_TITLE, shadow=PARCH_SHADOW, shadow_offset=0)
    _draw_shadow_text(surface, body_font, subtitle, inner.x + 72, inner.y + 36, accent, shadow=PARCH_SHADOW, shadow_offset=0)
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 12, inner.y + 64), (inner.right - 12, inner.y + 64), 1)

    body_y = inner.y + 84
    if hp_value is not None and hp_max is not None:
        health_rect = pygame.Rect(inner.x + 14, body_y, inner.width - 28, 24)
        _draw_shadow_text(surface, body_font, f"Health {hp_value}/{hp_max}", health_rect.x, health_rect.y - 18, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
        _draw_ui_meter(surface, board_assets, health_rect, hp_value, hp_max, color_family="red")
        body_y = health_rect.bottom + 20

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_lines(line, body_font, inner.width - 28))
    _draw_text_block(surface, body_font, wrapped, inner.x + 14, body_y, PARCH_BODY, 19)
    return close_rect

def _draw_parchment_panel_frame(
    surface: pygame.Surface,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    *,
    panel_key: str = "panel_beigeLight",
    inset_key: str = "panelInset_beigeLight",
    fallback_fill: tuple[int, int, int] = (135, 98, 61),
    fallback_border: tuple[int, int, int] = (193, 149, 98),
) -> pygame.Rect:
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(panel_key), rect):
        _draw_panel(surface, rect, fill=fallback_fill, border=fallback_border, radius=14)
    inner = rect.inflate(-18, -18)
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(inset_key), inner):
        pygame.draw.rect(surface, (216, 193, 144), inner, border_radius=12)
        pygame.draw.rect(surface, (180, 152, 111), inner, width=2, border_radius=12)
    return inner


def _draw_resource_icon(screen: pygame.Surface, resource, center: tuple[int, int], small_font: pygame.font.Font) -> None:
    cx, cy = center
    if resource.resource_type == "horses":
        pygame.draw.circle(screen, (91, 73, 49), (cx, cy), 14)
        pygame.draw.circle(screen, (212, 182, 117), (cx, cy), 10)
        pygame.draw.circle(screen, (91, 73, 49), (cx - 3, cy - 2), 3)
        pygame.draw.polygon(screen, (91, 73, 49), [(cx + 2, cy - 8), (cx + 8, cy - 14), (cx + 6, cy - 5)])
        pygame.draw.arc(screen, (244, 233, 205), (cx - 7, cy - 5, 14, 14), 0.2, 2.9, 2)
    elif resource.resource_type == "stone":
        rock = [(cx - 12, cy + 2), (cx - 6, cy - 10), (cx + 5, cy - 12), (cx + 13, cy - 2), (cx + 7, cy + 11), (cx - 7, cy + 10)]
        pygame.draw.polygon(screen, (124, 132, 144), rock)
        pygame.draw.polygon(screen, (205, 212, 222), rock, width=2)
        pygame.draw.line(screen, (88, 94, 104), (cx - 4, cy - 6), (cx + 3, cy + 6), 2)
        pygame.draw.line(screen, (88, 94, 104), (cx + 5, cy - 5), (cx + 8, cy + 4), 2)
    else:
        gem = [(cx, cy - 14), (cx + 12, cy - 3), (cx + 8, cy + 10), (cx, cy + 14), (cx - 8, cy + 10), (cx - 12, cy - 3)]
        pygame.draw.polygon(screen, (92, 170, 98), gem)
        pygame.draw.polygon(screen, (219, 247, 200), gem, width=2)
        pygame.draw.line(screen, (219, 247, 200), (cx, cy - 12), (cx, cy + 11), 2)
        pygame.draw.line(screen, (219, 247, 200), (cx - 7, cy - 1), (cx + 7, cy - 1), 2)


def _draw_building_icon(screen: pygame.Surface, building, rect: pygame.Rect, border: tuple[int, int, int]) -> None:
    body = rect.inflate(-2, -2)
    roof = [(body.x + 4, body.y + 7), (body.centerx, body.y - 5), (body.right - 4, body.y + 7)]
    pygame.draw.polygon(screen, (88, 59, 41), roof)
    if building.building_type == "storehouse":
        pygame.draw.rect(screen, (225, 176, 128), body, border_radius=6)
        pygame.draw.rect(screen, (244, 236, 220), body, width=2, border_radius=6)
        door = pygame.Rect(body.centerx - 5, body.bottom - 12, 10, 10)
        pygame.draw.rect(screen, (123, 79, 47), door, border_radius=2)
        pygame.draw.line(screen, (244, 236, 220), (body.x + 7, body.y + 14), (body.right - 7, body.y + 14), 2)
    elif building.building_type == "stable":
        pygame.draw.rect(screen, (198, 146, 103), body, border_radius=6)
        pygame.draw.rect(screen, (244, 236, 220), body, width=2, border_radius=6)
        pygame.draw.arc(screen, (244, 236, 220), (body.centerx - 7, body.y + 10, 14, 12), 0.1, 3.0, 2)
        pygame.draw.circle(screen, (244, 236, 220), (body.centerx + 7, body.y + 16), 2)
    elif building.building_type == "barracks":
        pygame.draw.rect(screen, (191, 130, 108), body, border_radius=4)
        pygame.draw.rect(screen, (244, 236, 220), body, width=2, border_radius=4)
        pygame.draw.rect(screen, (244, 236, 220), (body.x + 5, body.y + 7, 5, 8))
        pygame.draw.rect(screen, (244, 236, 220), (body.right - 10, body.y + 7, 5, 8))
        pygame.draw.line(screen, border, (body.centerx + 7, body.y + 2), (body.centerx + 7, body.y - 9), 2)
        pygame.draw.polygon(screen, border, [(body.centerx + 7, body.y - 9), (body.centerx + 14, body.y - 6), (body.centerx + 7, body.y - 3)])
    elif building.building_type == "quarry":
        pygame.draw.rect(screen, (150, 156, 168), body, border_radius=5)
        pygame.draw.rect(screen, (240, 242, 246), body, width=2, border_radius=5)
        pygame.draw.line(screen, (96, 102, 114), (body.x + 7, body.bottom - 6), (body.right - 7, body.y + 8), 3)
        pygame.draw.line(screen, (240, 242, 246), (body.x + 8, body.bottom - 7), (body.x + 14, body.bottom - 13), 2)
    elif building.building_type == "archer_tower":
        tower = pygame.Rect(body.x + 7, body.y + 2, body.width - 14, body.height - 4)
        pygame.draw.rect(screen, (184, 157, 120), tower, border_radius=4)
        pygame.draw.rect(screen, (244, 236, 220), tower, width=2, border_radius=4)
        pygame.draw.line(screen, (244, 236, 220), (tower.x + 5, tower.y + 8), (tower.right - 5, tower.y + 8), 2)
        pygame.draw.arc(screen, border, (tower.centerx - 8, tower.y + 10, 16, 14), 0.5, 2.6, 2)
    elif building.building_type == "ballista_tower":
        tower = pygame.Rect(body.x + 7, body.y + 2, body.width - 14, body.height - 4)
        pygame.draw.rect(screen, (170, 146, 122), tower, border_radius=4)
        pygame.draw.rect(screen, (244, 236, 220), tower, width=2, border_radius=4)
        pygame.draw.line(screen, border, (tower.centerx - 8, tower.y + 11), (tower.centerx + 8, tower.y + 11), 2)
        pygame.draw.line(screen, border, (tower.centerx, tower.y + 7), (tower.centerx, tower.y + 17), 2)
        pygame.draw.line(screen, border, (tower.centerx + 8, tower.y + 11), (tower.centerx + 14, tower.y + 6), 2)


def _unit_visual_tier(env: AgeGridEnv, unit) -> int:
    techs = env.faction_state(unit.faction).techs_unlocked
    if unit.unit_type == "worker":
        if "infrastructure" in techs or "currency" in techs:
            return 2
        if "mining" in techs or "masonry" in techs:
            return 1
        return 0
    if unit.unit_type == "soldier":
        if "steel" in techs or "fortify" in techs:
            return 2
        if "iron" in techs:
            return 1
        return 0
    if unit.unit_type == "archer":
        if "precision" in techs or "engineering" in techs:
            return 2
        if "fletching" in techs:
            return 1
        return 0
    if unit.unit_type == "horseman":
        if "heavy_cavalry" in techs or ("stirrups" in techs and "logistics" in techs):
            return 2
        if "stirrups" in techs:
            return 1
        return 0
    if unit.unit_type in {"heavy_cavalry", "ballista"}:
        return 2
    return 0


def _draw_unit_sprite(
    screen: pygame.Surface,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    unit,
    center: tuple[int, int],
    ring_fill: tuple[int, int, int],
    ring_border: tuple[int, int, int],
    size: int = 30,
) -> bool:
    sprite = board_assets.character_sprite(unit.unit_type, _unit_visual_tier(env, unit))
    if sprite is None:
        return False
    ring = pygame.Surface((size + 18, size + 18), pygame.SRCALPHA)
    mid = (ring.get_width() // 2, ring.get_height() // 2 + 1)
    pygame.draw.circle(ring, (*ring_fill, 58), mid, size // 2 + 6)
    pygame.draw.circle(ring, (*ring_border, 112), mid, size // 2 + 8, width=2)
    screen.blit(ring, (center[0] - ring.get_width() // 2, center[1] - ring.get_height() // 2 + 4))
    scaled = _safe_scale(sprite, (size, size))
    if scaled is None:
        return False
    sprite_rect = scaled.get_rect(center=(center[0], center[1] + 1))
    screen.blit(scaled, sprite_rect)
    return True


def _draw_unit_icon(screen: pygame.Surface, unit, center: tuple[int, int], fill: tuple[int, int, int], border: tuple[int, int, int]) -> None:
    cx, cy = center
    if unit.unit_type == "worker":
        pygame.draw.circle(screen, fill, (cx, cy - 6), 7)
        pygame.draw.circle(screen, border, (cx, cy - 6), 7, width=2)
        pygame.draw.line(screen, border, (cx, cy + 2), (cx, cy + 14), 3)
        pygame.draw.line(screen, border, (cx - 7, cy + 7), (cx + 7, cy + 3), 3)
        pygame.draw.line(screen, border, (cx + 4, cy - 1), (cx + 13, cy - 11), 3)
        pygame.draw.line(screen, (240, 240, 240), (cx + 11, cy - 13), (cx + 17, cy - 7), 2)
    elif unit.unit_type == "soldier":
        shield = [(cx, cy - 14), (cx + 11, cy - 6), (cx + 8, cy + 10), (cx, cy + 15), (cx - 8, cy + 10), (cx - 11, cy - 6)]
        pygame.draw.polygon(screen, fill, shield)
        pygame.draw.polygon(screen, border, shield, width=2)
        pygame.draw.line(screen, (244, 244, 244), (cx, cy - 10), (cx, cy + 10), 2)
        pygame.draw.line(screen, border, (cx + 9, cy - 12), (cx + 17, cy - 20), 3)
    elif unit.unit_type == "horseman":
        horse = [(cx - 13, cy + 5), (cx - 8, cy - 6), (cx + 4, cy - 10), (cx + 13, cy - 3), (cx + 10, cy + 8), (cx - 2, cy + 12)]
        pygame.draw.polygon(screen, fill, horse)
        pygame.draw.polygon(screen, border, horse, width=2)
        pygame.draw.line(screen, border, (cx - 6, cy + 10), (cx - 8, cy + 16), 2)
        pygame.draw.line(screen, border, (cx + 2, cy + 10), (cx + 1, cy + 17), 2)
        pygame.draw.line(screen, border, (cx + 5, cy - 6), (cx + 15, cy - 18), 3)
    elif unit.unit_type == "archer":
        pygame.draw.circle(screen, fill, (cx, cy - 6), 7)
        pygame.draw.circle(screen, border, (cx, cy - 6), 7, width=2)
        pygame.draw.line(screen, border, (cx, cy + 2), (cx, cy + 14), 3)
        pygame.draw.arc(screen, border, (cx + 2, cy - 8, 16, 22), 4.8, 1.4, 2)
        pygame.draw.line(screen, border, (cx + 10, cy - 6), (cx + 10, cy + 11), 2)
        pygame.draw.line(screen, (244, 244, 244), (cx - 4, cy + 1), (cx + 12, cy - 4), 2)
    else:
        pygame.draw.circle(screen, fill, (cx, cy), 13)
        pygame.draw.circle(screen, border, (cx, cy), 13, width=2)


def _hex_origin(col: int, row: int, board_origin: tuple[int, int]) -> tuple[int, int]:
    left, top = hexgrid.hex_to_pixel(col, row, HEX_SIZE)
    return round(board_origin[0] + left), round(board_origin[1] + top)


def _hex_center(col: int, row: int, board_origin: tuple[int, int]) -> tuple[int, int]:
    left, top = _hex_origin(col, row, board_origin)
    return left + HEX_WIDTH // 2, top


def _hex_points(col: int, row: int, board_origin: tuple[int, int]) -> list[tuple[int, int]]:
    left, top = _hex_origin(col, row, board_origin)
    return hexgrid.hex_polygon_points(left, top, HEX_SIZE)


def _hex_bounds(col: int, row: int, board_origin: tuple[int, int]) -> pygame.Rect:
    left, top = _hex_origin(col, row, board_origin)
    return pygame.Rect(left, top - HEX_SIZE, HEX_WIDTH, HEX_HEIGHT)


def _board_pixel_size(env: AgeGridEnv) -> tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for row in range(env.config.height):
        for col in range(env.config.width):
            left, top = hexgrid.hex_to_pixel(col, row, HEX_SIZE)
            max_x = max(max_x, left + HEX_WIDTH)
            max_y = max(max_y, top + HEX_SIZE)
    return math.ceil(max_x), math.ceil(max_y + HEX_SIZE)


def _safe_scale(surface: pygame.Surface | None, size: tuple[int, int]) -> pygame.Surface | None:
    if surface is None:
        return None
    return pygame.transform.smoothscale(surface, size)

def _trim_sprite_alpha(sprite: pygame.Surface | None) -> pygame.Surface | None:
    if sprite is None:
        return None
    bounds = sprite.get_bounding_rect(min_alpha=1)
    if bounds.width <= 0 or bounds.height <= 0:
        return sprite
    return sprite.subsurface(bounds).copy()


def _blit_centered(surface: pygame.Surface, sprite: pygame.Surface | None, center: tuple[int, int], offset_y: int = 0) -> None:
    if sprite is None:
        return
    rect = sprite.get_rect(center=(center[0], center[1] + offset_y))
    surface.blit(sprite, rect.topleft)


def _blit_bottom_centered(
    surface: pygame.Surface,
    sprite: pygame.Surface | None,
    anchor: tuple[int, int],
    offset_y: int = 0,
) -> pygame.Rect | None:
    if sprite is None:
        return None
    rect = sprite.get_rect(midbottom=(anchor[0], anchor[1] + offset_y))
    surface.blit(sprite, rect.topleft)
    return rect


def _draw_soft_shadow(
    surface: pygame.Surface,
    center: tuple[int, int],
    width: int,
    height: int,
    alpha: int = 70,
) -> None:
    shadow = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (*BOARD_SHADOW, alpha), shadow.get_rect())
    surface.blit(shadow, (center[0] - width // 2, center[1] - height // 2))


def _terrain_kind(env: AgeGridEnv, pos: tuple[int, int]) -> str:
    if any(base.position == pos for base in env.bases.values()):
        return "stone"
    if any(building.position == pos and building.hp > 0 for building in env.buildings):
        return "dirt"
    resource = next((r for r in env.resources if r.position == pos and r.remaining > 0), None)
    if resource is None:
        return "grass" if (pos[0] + pos[1]) % 4 else "dirt"
    if resource.resource_type == "stone":
        return "stone"
    if resource.resource_type == "horses":
        return "sand"
    return "grass"


def _draw_asset_marker(
    surface: pygame.Surface,
    sprite: pygame.Surface | None,
    center: tuple[int, int],
    fallback,
    tint: tuple[int, int, int] | None = None,
    scale: tuple[int, int] = (40, 40),
    anchor_mode: str = "bottom",
    offset_y: int = 0,
) -> None:
    trimmed = _trim_sprite_alpha(sprite)
    scaled = _safe_scale(trimmed, scale)
    if scaled is not None:
        if anchor_mode == "bottom":
            _blit_bottom_centered(surface, scaled, center, offset_y=offset_y)
        else:
            _blit_centered(surface, scaled, center, offset_y=offset_y)
        if tint is not None:
            glow = pygame.Surface((scale[0] + 14, scale[1] + 14), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (*tint, 42), glow.get_rect(), width=4)
            surface.blit(glow, (center[0] - glow.get_width() // 2, center[1] - glow.get_height() // 2))
        return
    fallback()


def _hover_tile_lines(env: AgeGridEnv, pos: tuple[int, int]) -> list[str]:
    unit = next((u for u in env.units if u.position == pos), None)
    if unit is not None:
        label = unit.unit_type.replace("_", " ").title()
        return [
            f"{unit.faction} {label}",
            f"HP {unit.hp}  ATK {unit.attack_damage}  RNG {unit.attack_range}",
        ]
    building = next((b for b in env.buildings if b.position == pos and b.hp > 0), None)
    if building is not None:
        label = _building_label(building.building_type)
        return [
            f"{building.faction} {label}",
            f"HP {building.hp}",
            BUILDING_HELP.get(building.building_type, "Structure"),
        ]
    base = next((base for faction, base in env.bases.items() if base.position == pos), None)
    if base is not None:
        faction = next(name for name, current_base in env.bases.items() if current_base is base)
        return [f"{faction} Base", f"HP {base.hp}"]
    resource = next((r for r in env.resources if r.position == pos and r.remaining > 0), None)
    if resource is not None:
        label = "Horse Herd" if resource.resource_type == "horses" else "Stone Deposit" if resource.resource_type == "stone" else "Ore Vein"
        detail = "Unlocks horseback riding and cavalry" if resource.resource_type == "horses" else "Unlocks quarry, walls, and stronger structures" if resource.resource_type == "stone" else "Gatherable resource"
        return [label, detail]
    return []


def _draw_hover_tile_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    lines: list[str],
) -> None:
    if not lines:
        return
    inner = _draw_parchment_panel_frame(surface, board_assets, rect, panel_key="panel_beigeLight", inset_key="panelInset_beigeLight")
    y = _draw_parchment_header(surface, title_font, body_font, inner, lines[0])
    wrapped: list[str] = []
    for line in lines[1:]:
        wrapped.extend(_wrap_lines(line, body_font, inner.width - 16))
    _draw_text_block(surface, body_font, wrapped, inner.x + 8, y, PARCH_BODY, 18)


def _board_origin_for_layout(
    env: AgeGridEnv,
    board_x: int,
    top_bar: int,
    pad: int,
    height_px: int,
    board_width: int,
    board_height: int,
) -> tuple[int, int]:
    content_top = pad + top_bar + 8
    content_bottom = height_px - pad - 18
    available_height = max(board_height, content_bottom - content_top)
    origin_y = content_top + max(0, (available_height - board_height) // 2) + HEX_SIZE
    inner_width = board_width + 28
    origin_x = board_x + max(0, (board_width - inner_width) // 2) + 8
    return origin_x, origin_y


def _board_origin_in_viewport(
    viewport: pygame.Rect,
    board_width: int,
    board_height: int,
    pan: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, int]:
    origin_x = viewport.x + (viewport.width - board_width) / 2 + pan[0]
    origin_y = viewport.y + (viewport.height - board_height) / 2 + HEX_SIZE + pan[1]
    return round(origin_x), round(origin_y)


def _clamp_camera_pan(
    viewport: pygame.Rect,
    board_width: int,
    board_height: int,
    pan: tuple[float, float],
) -> tuple[float, float]:
    bleed = 56
    if board_width <= viewport.width:
        max_x = bleed
    else:
        max_x = (board_width - viewport.width) / 2 + bleed
    if board_height <= viewport.height:
        max_y = bleed
    else:
        max_y = (board_height - viewport.height) / 2 + bleed
    return (
        max(-max_x, min(max_x, pan[0])),
        max(-max_y, min(max_y, pan[1])),
    )


def _hover_panel_rect(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    tile_rect: pygame.Rect,
    lines: list[str],
) -> pygame.Rect:
    line_height = 16
    width = 210
    height = 40 + max(1, len(lines) - 1) * line_height
    x = tile_rect.right + 10
    y = tile_rect.y - 4
    if x + width > surface.get_width() - 8:
        x = tile_rect.x - width - 10
    if x < 8:
        x = 8
    if y + height > surface.get_height() - 8:
        y = surface.get_height() - height - 8
    if y < 8:
        y = 8
    return pygame.Rect(x, y, width, height)


def _tech_tree_positions(rect: pygame.Rect) -> dict[str, tuple[int, int]]:
    max_column = max(definition.column for definition in tech.TECH_DEFS.values())
    max_row = max(definition.row for definition in tech.TECH_DEFS.values())
    detail_width = min(268, max(220, rect.width // 4))
    tree_width = max(360, rect.width - detail_width - 80)
    left = rect.x + 72
    top = rect.y + 96
    x_gap = max(116, tree_width // max(1, max_column - 1))
    available_height = max(420, rect.height - 180)
    y_gap = max(56, available_height // max(1, max_row + 1))
    return {
        tech_id: (left + (definition.column - 1) * x_gap, top + definition.row * y_gap)
        for tech_id, definition in tech.TECH_DEFS.items()
    }


def _draw_tech_tree_overlay(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    rect: pygame.Rect,
    focus_faction: str,
    human_turn_active: bool,
    mouse_pos: tuple[int, int],
) -> dict[str, pygame.Rect]:
    detail_width = min(268, max(220, rect.width // 4))
    node_width = 96 if rect.width < 1180 else 112
    node_height = 74 if rect.height < 860 else 68
    node_half_w = node_width // 2
    node_half_h = node_height // 2
    unlock_wrap_width = max(74, node_width - 16)

    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((8, 12, 18, 170))
    surface.blit(veil, (0, 0))
    pygame.draw.rect(surface, (20, 27, 35), rect, border_radius=14)
    pygame.draw.rect(surface, (86, 102, 119), rect, width=2, border_radius=14)
    _draw_shadow_text(
        surface,
        title_font,
        "Research tree",
        rect.x + 16,
        rect.y + 14,
        (235, 239, 244),
        shadow=(10, 12, 16),
    )
    note = "Press T to close"
    note_width = body_font.size(note)[0]
    _draw_shadow_text(
        surface,
        body_font,
        note,
        rect.right - note_width - 18,
        rect.y + 18,
        (186, 194, 202),
        shadow=(10, 12, 16),
        shadow_offset=1,
    )
    subtitle = "Unlock paths, status, and what each tech enables"
    _draw_shadow_text(surface, body_font, subtitle, rect.x + 16, rect.y + 46, TEXT_SECONDARY, shadow=(10, 12, 16), shadow_offset=1)

    positions = _tech_tree_positions(rect)
    node_rects: dict[str, pygame.Rect] = {}
    for tech_id, definition in tech.TECH_DEFS.items():
        start = positions[tech_id]
        for req in definition.requires:
            end = positions[req]
            points = [
                (end[0] + node_half_w, end[1]),
                ((start[0] + end[0]) // 2, end[1]),
                ((start[0] + end[0]) // 2, start[1]),
                (start[0] - node_half_w, start[1]),
            ]
            pygame.draw.lines(surface, (88, 102, 118), False, points, 3)

    legend = [("Done", (78, 122, 86)), ("Active", (95, 121, 166)), ("Ready", (129, 112, 70)), ("Locked", (70, 76, 84))]
    lx = rect.x + 16
    for label, color in legend:
        chip = pygame.Rect(lx, rect.bottom - 34, 70, 20)
        pygame.draw.rect(surface, color, chip, border_radius=10)
        _draw_shadow_text(surface, body_font, label, chip.x + 12, chip.y + 2, TEXT_PRIMARY, shadow=(10, 12, 16), shadow_offset=1)
        lx += 82

    for tech_id in TECH_TREE_ORDER:
        cx, cy = positions[tech_id]
        node_rect = pygame.Rect(cx - node_half_w, cy - node_half_h, node_width, node_height)
        node_rects[tech_id] = node_rect
        red_status = _tech_status(env, "Red", tech_id)
        blue_status = _tech_status(env, "Blue", tech_id)
        focus_status = _tech_status(env, focus_faction, tech_id)
        unlocked_any = red_status == "Done" or blue_status == "Done"
        ready_any = red_status == "Ready" or blue_status == "Ready"
        active_any = red_status.startswith("Active") or blue_status.startswith("Active")
        node_fill = (74, 80, 88)
        if unlocked_any:
            node_fill = (54, 75, 58)
        elif active_any:
            node_fill = (58, 76, 104)
        elif ready_any:
            node_fill = (92, 79, 54)
        pygame.draw.rect(surface, node_fill, node_rect, border_radius=14)
        border_color = (184, 194, 206)
        if human_turn_active and focus_status == "Ready":
            border_color = (241, 214, 154)
        if node_rect.collidepoint(mouse_pos):
            border_color = (224, 233, 242)
        pygame.draw.rect(surface, border_color, node_rect, width=2, border_radius=14)

        style = TECH_ICON_STYLES.get(tech_id, {"label": "?", "bg": (74, 80, 88), "fg": (235, 239, 242)})
        icon_rect = pygame.Rect(node_rect.x + 8, node_rect.y + 8, 24, 24)
        pygame.draw.rect(surface, style["bg"], icon_rect, border_radius=6)
        pygame.draw.rect(surface, (236, 240, 244), icon_rect, width=1, border_radius=6)
        _draw_shadow_text(surface, body_font, style["label"], icon_rect.x + 6, icon_rect.y + 2, style["fg"], shadow=(12, 16, 20), shadow_offset=1)
        label_lines = _wrap_lines(_tech_label(tech_id), body_font, max(48, node_width - 44))
        _draw_text_block(surface, body_font, label_lines[:2], node_rect.x + 38, node_rect.y + 8, TEXT_PRIMARY, 15)
        unlocks = tech.TECH_DEFS[tech_id].summary
        unlock_lines = _wrap_lines(unlocks, body_font, unlock_wrap_width)
        _draw_text_block(surface, body_font, unlock_lines[:2], node_rect.x + 8, node_rect.y + 38, TEXT_SECONDARY, 14)

        red_chip = pygame.Rect(node_rect.x + 8, node_rect.bottom + 6, 44, 18)
        blue_chip = pygame.Rect(node_rect.right - 52, node_rect.bottom + 6, 44, 18)
        pygame.draw.rect(surface, (82, 52, 52), red_chip, border_radius=9)
        pygame.draw.rect(surface, (52, 66, 92), blue_chip, border_radius=9)
        red_text = "Done" if red_status == "Done" else "Act" if red_status.startswith("Active") else "Go" if red_status == "Ready" else "-"
        blue_text = "Done" if blue_status == "Done" else "Act" if blue_status.startswith("Active") else "Go" if blue_status == "Ready" else "-"
        _draw_shadow_text(surface, body_font, f"R {red_text}", red_chip.x + 7, red_chip.y + 1, (244, 184, 180), shadow=(10, 12, 16), shadow_offset=1)
        _draw_shadow_text(surface, body_font, f"B {blue_text}", blue_chip.x + 7, blue_chip.y + 1, (176, 214, 255), shadow=(10, 12, 16), shadow_offset=1)

    hovered_tech_id = next((tech_id for tech_id, node_rect in node_rects.items() if node_rect.collidepoint(mouse_pos)), None)
    if hovered_tech_id is None:
        available_focus = _available_research_ids(env, focus_faction)
        hovered_tech_id = available_focus[0] if available_focus else TECH_TREE_ORDER[0]

    detail_rect = pygame.Rect(rect.right - detail_width - 24, rect.y + 84, detail_width, rect.height - 138)
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite("panel"), detail_rect):
        _draw_panel(surface, detail_rect, fill=(135, 98, 61), border=(193, 149, 98), radius=14)
    inner = detail_rect.inflate(-18, -22)
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite("panel_inset"), inner):
        pygame.draw.rect(surface, (216, 193, 144), inner, border_radius=12)
        pygame.draw.rect(surface, (180, 152, 111), inner, width=2, border_radius=12)

    style = TECH_ICON_STYLES.get(hovered_tech_id, {"label": "?", "bg": (74, 80, 88), "fg": (235, 239, 242)})
    icon_rect = pygame.Rect(inner.x + 14, inner.y + 14, 34, 34)
    pygame.draw.rect(surface, style["bg"], icon_rect, border_radius=8)
    pygame.draw.rect(surface, (236, 240, 244), icon_rect, width=1, border_radius=8)
    _draw_shadow_text(surface, body_font, style["label"], icon_rect.x + 10, icon_rect.y + 5, style["fg"], shadow=(12, 16, 20), shadow_offset=1)
    _draw_shadow_text(surface, title_font, _tech_label(hovered_tech_id), inner.x + 58, inner.y + 14, (76, 58, 40), shadow=(244, 226, 190), shadow_offset=0)

    detail_y = inner.y + 60
    for line in _tech_detail_lines(env, focus_faction, hovered_tech_id):
        wrapped = _wrap_lines(line, body_font, inner.width - 28)
        _draw_text_block(surface, body_font, wrapped, inner.x + 14, detail_y, (84, 72, 58), 18)
        detail_y += len(wrapped) * 18 + 6

    action_hint = (
        "Click this node to start research."
        if human_turn_active and tech.can_research(env, focus_faction, hovered_tech_id)
        else f"Viewing {focus_faction} research state."
        if not human_turn_active
        else "Hover another tech to inspect prerequisites."
    )
    hint_lines = _wrap_lines(action_hint, body_font, inner.width - 28)
    _draw_text_block(surface, body_font, hint_lines, inner.x + 14, inner.bottom - 42, (112, 93, 72), 18)
    return node_rects


def _tile_center(ox: int, oy: int, tile: int, pos: tuple[int, int]) -> tuple[int, int]:
    return _hex_center(pos[0], pos[1], (ox, oy))


def _unit_by_id(env: AgeGridEnv, unit_id: int):
    return next((u for u in env.units if u.id == unit_id), None)


def _effects_from_actions(
    env: AgeGridEnv,
    actions: list[tuple | None],
    faction: str,
) -> list[VisualEffect]:
    effects: list[VisualEffect] = []
    primary = RED_PRIMARY if faction == "Red" else BLUE_PRIMARY
    accent = RED_ACCENT if faction == "Red" else BLUE_ACCENT

    for action in actions:
        if action is None:
            continue
        kind = action[0]
        if kind == "gather":
            unit = _unit_by_id(env, action[1])
            if unit is not None:
                effects.append(VisualEffect("gather", 34, 34, RESOURCE_GLOW, pos=unit.position, label="+"))
        elif kind == "research":
            effects.append(VisualEffect("banner", 42, 42, accent, label=f"{faction}: {action[1]}"))
        elif kind == "build":
            effects.append(VisualEffect("build", 40, 40, accent, pos=action[3], label=action[2].title()))
        elif kind == "train":
            base_pos = env.bases[faction].position
            effects.append(VisualEffect("train", 34, 34, accent, pos=base_pos, label=action[1].title()))
        elif kind == "spawn_worker":
            base_pos = env.bases[faction].position
            effects.append(VisualEffect("train", 34, 34, accent, pos=base_pos, label="Worker"))
        elif kind == "attack":
            attacker = _unit_by_id(env, action[1])
            target = _unit_by_id(env, action[2])
            if attacker is not None:
                effects.append(VisualEffect("attack", 26, 26, primary, pos=attacker.position, label="!"))
            if target is not None:
                effects.append(VisualEffect("hit", 26, 26, accent, pos=target.position))
        elif kind == "attack_base":
            base_pos = env.bases[action[2]].position
            effects.append(VisualEffect("attack_base", 34, 34, primary, pos=base_pos, label="BASE"))
        elif kind == "move_towards":
            unit = _unit_by_id(env, action[1])
            if unit is not None:
                effects.append(VisualEffect("move", 18, 18, accent, pos=unit.position))
    return effects


def _update_effects(effects: list[VisualEffect]) -> list[VisualEffect]:
    updated: list[VisualEffect] = []
    for effect in effects:
        effect.ttl -= 1
        if effect.ttl > 0:
            updated.append(effect)
    return updated


def _draw_effects(
    screen: pygame.Surface,
    effects: list[VisualEffect],
    ox: int,
    oy: int,
    tile: int,
    font: pygame.font.Font,
) -> None:
    for effect in effects:
        progress = effect.ttl / effect.max_ttl
        alpha = max(40, min(220, int(220 * progress)))
        color = effect.color
        if effect.kind == "banner":
            banner = pygame.Surface((340, 42), pygame.SRCALPHA)
            pygame.draw.rect(banner, (15, 20, 27, min(230, alpha + 20)), banner.get_rect(), border_radius=12)
            pygame.draw.rect(banner, (*color, alpha), banner.get_rect(), width=2, border_radius=12)
            screen.blit(banner, (screen.get_width() // 2 - 150, 14))
            text = font.render(effect.label, True, (244, 246, 248))
            screen.blit(text, (screen.get_width() // 2 - text.get_width() // 2, 26))
            continue
        if effect.pos is None:
            continue

        cx, cy = _tile_center(ox, oy, tile, effect.pos)
        overlay = pygame.Surface((HEX_WIDTH * 2, HEX_HEIGHT * 2), pygame.SRCALPHA)
        local_center = (overlay.get_width() // 2, overlay.get_height() // 2)

        if effect.kind in {"gather", "build", "train"}:
            radius = int(10 + (1 - progress) * 18)
            pygame.draw.circle(overlay, (*color, alpha), local_center, radius, width=3)
        elif effect.kind in {"attack", "attack_base"}:
            radius = int(12 + (1 - progress) * 20)
            pygame.draw.circle(overlay, (*color, alpha), local_center, radius, width=4)
            pygame.draw.line(overlay, (*color, alpha), (local_center[0] - 12, local_center[1] - 12), (local_center[0] + 12, local_center[1] + 12), 3)
            pygame.draw.line(overlay, (*color, alpha), (local_center[0] + 12, local_center[1] - 12), (local_center[0] - 12, local_center[1] + 12), 3)
        elif effect.kind == "hit":
            pygame.draw.circle(overlay, (*color, alpha), local_center, 14, width=4)
        elif effect.kind == "move":
            pygame.draw.circle(overlay, (*color, alpha), local_center, 8, width=2)

        screen.blit(overlay, (cx - overlay.get_width() // 2, cy - overlay.get_height() // 2))

        if effect.label:
            label = font.render(effect.label, True, (245, 245, 245))
            label_bg = pygame.Surface((label.get_width() + 12, label.get_height() + 8), pygame.SRCALPHA)
            pygame.draw.rect(label_bg, (12, 16, 22, min(220, alpha + 40)), label_bg.get_rect(), border_radius=8)
            pygame.draw.rect(label_bg, (*color, alpha), label_bg.get_rect(), width=2, border_radius=8)
            screen.blit(label_bg, (cx - label_bg.get_width() // 2, cy - 34))
            screen.blit(label, (cx - label.get_width() // 2, cy - 30))


def _draw_setup_screen(
    screen: pygame.Surface,
    width_px: int,
    height_px: int,
    title_font: pygame.font.Font,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    red_index: int,
    blue_index: int,
    collapse_enabled: bool,
) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
    screen.fill((19, 24, 30))

    title = title_font.render("AgeGrid Viewer Setup", True, (238, 240, 242))
    subtitle_lines = _wrap_lines(
        "Choose an agent for each faction, then start the turn-by-turn simulation.",
        font,
        width_px - 80,
    )
    screen.blit(title, (40, 30))
    _draw_text_block(screen, font, subtitle_lines, 40, 70, (190, 198, 205), 24)

    gap = 24
    card_w = max(260, (width_px - 80 - gap) // 2)
    card_h = 190
    card_y = 120
    red_card = pygame.Rect(40, card_y, card_w, card_h)
    blue_card = pygame.Rect(width_px - 40 - card_w, card_y, card_w, card_h)
    tips = [
        "Controls after start: Human uses left click, G gathers, Space ends turn.",
        "R resets to setup. P saves a debug snapshot. Esc closes the viewer.",
    ]
    tips_height = len(tips) * 18
    rules_btn = pygame.Rect(width_px // 2 - 110, blue_card.bottom + 22, 220, 36)
    start_btn_y = rules_btn.bottom + 18
    start_btn = pygame.Rect(width_px // 2 - 90, start_btn_y, 180, 46)

    for rect, faction, idx, color in (
        (red_card, "Red", red_index, (140, 58, 58)),
        (blue_card, "Blue", blue_index, (58, 84, 148)),
    ):
        pygame.draw.rect(screen, (32, 38, 46), rect, border_radius=12)
        pygame.draw.rect(screen, color, rect, width=3, border_radius=12)
        spec = AGENT_SPECS[idx]
        screen.blit(title_font.render(f"{faction} Faction", True, (235, 235, 235)), (rect.x + 18, rect.y + 18))
        screen.blit(font.render(spec.label, True, (245, 223, 162)), (rect.x + 18, rect.y + 64))
        desc_lines = _wrap_lines(spec.description, small_font, rect.w - 36)
        _draw_text_block(screen, small_font, desc_lines, rect.x + 18, rect.y + 96, (205, 210, 214), 18)
        controls_text = "Use A/D" if faction == "Red" else "Use J/L"
        hint_lines = _wrap_lines(f"Click card to cycle or {controls_text} on keyboard.", small_font, rect.w - 36)
        hint_y = rect.y + rect.h - 18 - len(hint_lines) * 18
        _draw_text_block(screen, small_font, hint_lines, rect.x + 18, hint_y, (160, 168, 176), 18)

    rule_fill = (58, 79, 108) if collapse_enabled else PANEL_INSET
    rule_border = (132, 176, 232) if collapse_enabled else PANEL_SOFT
    _draw_panel(screen, rules_btn, fill=rule_fill, border=rule_border, radius=10)
    rule_label = "Collapse: On" if collapse_enabled else "Collapse: Off"
    _draw_shadow_text(
        screen,
        small_font,
        rule_label,
        rules_btn.centerx - small_font.size(rule_label)[0] // 2,
        rules_btn.y + 8,
        TEXT_PRIMARY,
        shadow=(10, 12, 16),
        shadow_offset=1,
    )

    pygame.draw.rect(screen, (74, 102, 72), start_btn, border_radius=10)
    pygame.draw.rect(screen, (136, 181, 131), start_btn, width=2, border_radius=10)
    screen.blit(font.render("Start Match", True, (244, 246, 244)), (start_btn.x + 42, start_btn.y + 11))

    tips_y = min(height_px - 24 - tips_height, start_btn.bottom + 22)
    _draw_text_block(screen, small_font, tips, 40, tips_y, (180, 186, 192), 18)
    return red_card, blue_card, rules_btn, start_btn


def _turn_history_text(history: list[TurnSnapshot]) -> list[str]:
    lines: list[str] = []
    for snapshot in history:
        lines.append(
            f"Turn {snapshot.turn_number}: Red={snapshot.red.last_action} | Blue={snapshot.blue.last_action}"
        )
        lines.append(
            f"  Red log: {', '.join(snapshot.red.log) if snapshot.red.log else '-'}"
        )
        lines.append(
            f"  Blue log: {', '.join(snapshot.blue.log) if snapshot.blue.log else '-'}"
        )
    return lines


def _build_debug_snapshot(
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
            "Full turn history:",
        ]
    )
    lines.extend(_turn_history_text(history))
    return "\n".join(lines)


def _write_debug_snapshot(snapshot_text: str) -> Path:
    output_path = Path.cwd() / "agegrid_debug_snapshot.txt"
    output_path.write_text(snapshot_text, encoding="utf-8")
    return output_path


def run_viewer() -> None:
    pygame.init()
    pygame.display.set_caption("AgeGrid Viewer")

    _set_hex_zoom(1.0)
    env = AgeGridEnv()
    pad = 20
    side_panel = 320
    default_board_width, default_board_height = _board_pixel_size(env)

    # HUD layout — three horizontal zones across the board width
    faction_bar_w = max(180, min(HUD_FACTION_W, (default_board_width - 20) // 3))
    center_zone_x = pad + faction_bar_w + 8
    center_zone_w = default_board_width - 2 * faction_bar_w - 16
    top_bar = HUD_H + pad + 10   # total vertical space reserved for the top HUD strip

    width_px = pad * 3 + default_board_width + side_panel + 36
    height_px = pad * 2 + top_bar + default_board_height + BASE_HEX_SIZE + 44

    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()
    board_assets = BoardAssets.load(Path.cwd())

    font = pygame.font.SysFont("segoeui", 24, bold=True)
    big = pygame.font.SysFont("segoeui", 34, bold=True)
    small = pygame.font.SysFont("segoeui", 18)
    tiny = pygame.font.SysFont("segoeui", 16)

    board_x = pad
    board_content_top = pad + top_bar + 4
    board_content_bottom = height_px - pad - 18
    board_rect = pygame.Rect(
        board_x,
        board_content_top,
        default_board_width,
        max(default_board_height, board_content_bottom - board_content_top),
    )
    # End-turn button on the right of the centre HUD zone, vertically centred
    btn_w, btn_h = 130, 32
    btn_rect = pygame.Rect(
        center_zone_x + center_zone_w - btn_w - HUD_INNER_PAD,
        pad + (HUD_H - btn_h) // 2,
        btn_w, btn_h,
    )

    red_index = 0
    blue_index = 0
    collapse_enabled = True
    in_setup = True
    red_agent = None
    blue_agent = None
    red_info = FactionTurnInfo("Red", [])
    blue_info = FactionTurnInfo("Blue", [])
    turn_history: list[TurnSnapshot] = []
    effects: list[VisualEffect] = []
    human_moved_units: set[int] = set()
    human_turn_actions: list[tuple | None] = []
    human_turn_log: list[str] = []
    active_human_faction: str | None = None
    show_tech_tree = False
    show_debug = False
    event_scroll = 0
    event_panel_rect = pygame.Rect(0, 0, 0, 0)
    tech_btn_rect = pygame.Rect(0, 0, 0, 0)
    tech_tree_node_rects: dict[str, pygame.Rect] = {}
    camera_btn_rect = pygame.Rect(0, 0, 0, 0)
    selected_unit_close_rect = pygame.Rect(0, 0, 0, 0)
    human_action_button_rects: list[tuple[pygame.Rect, object, bool]] = []
    selected_tile: tuple[int, int] | None = None
    selected_unit_id: int | None = None
    pending_build_type: str | None = None
    camera_zoom = 1.0
    camera_pan = [0.0, 0.0]
    panning = False
    pan_anchor_mouse = (0, 0)
    pan_anchor = (0.0, 0.0)
    pan_drag_distance = 0.0
    board_backdrop = board_rect.inflate(22, 28)
    board_origin = _board_origin_in_viewport(board_rect, default_board_width, default_board_height)

    running = True
    while running:
        clock.tick(60)

        if in_setup:
            red_card, blue_card, rules_btn, start_btn = _draw_setup_screen(
                screen,
                width_px,
                height_px,
                big,
                font,
                small,
                red_index,
                blue_index,
                collapse_enabled,
            )
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_a:
                        red_index = (red_index - 1) % len(AGENT_SPECS)
                    elif event.key == pygame.K_d:
                        red_index = (red_index + 1) % len(AGENT_SPECS)
                    elif event.key == pygame.K_j:
                        blue_index = (blue_index - 1) % len(AGENT_SPECS)
                    elif event.key == pygame.K_l:
                        blue_index = (blue_index + 1) % len(AGENT_SPECS)
                    elif event.key == pygame.K_c:
                        collapse_enabled = not collapse_enabled
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        env = AgeGridEnv(config=env.config.__class__(collapse_enabled=collapse_enabled))
                        red_agent = create_agent(AGENT_SPECS[red_index].key, seed=0)
                        blue_agent = create_agent(AGENT_SPECS[blue_index].key, seed=1)
                        red_info = FactionTurnInfo("Red", [])
                        blue_info = FactionTurnInfo("Blue", [])
                        turn_history = []
                        effects = []
                        human_moved_units = set()
                        human_turn_actions = []
                        human_turn_log = []
                        active_human_faction = None
                        selected_tile = None
                        selected_unit_id = None
                        pending_build_type = None
                        camera_zoom = _set_hex_zoom(1.0)
                        camera_pan = [0.0, 0.0]
                        if _is_human_agent_key(_current_agent_key(env, red_index, blue_index)):
                            env.start_faction_turn()
                            active_human_faction = env.factions[env.current_player]
                        in_setup = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if red_card.collidepoint(event.pos):
                        red_index = (red_index + 1) % len(AGENT_SPECS)
                    elif blue_card.collidepoint(event.pos):
                        blue_index = (blue_index + 1) % len(AGENT_SPECS)
                    elif rules_btn.collidepoint(event.pos):
                        collapse_enabled = not collapse_enabled
                    elif start_btn.collidepoint(event.pos):
                        env = AgeGridEnv(config=env.config.__class__(collapse_enabled=collapse_enabled))
                        red_agent = create_agent(AGENT_SPECS[red_index].key, seed=0)
                        blue_agent = create_agent(AGENT_SPECS[blue_index].key, seed=1)
                        red_info = FactionTurnInfo("Red", [])
                        blue_info = FactionTurnInfo("Blue", [])
                        turn_history = []
                        effects = []
                        human_moved_units = set()
                        human_turn_actions = []
                        human_turn_log = []
                        active_human_faction = None
                        selected_tile = None
                        selected_unit_id = None
                        pending_build_type = None
                        camera_zoom = _set_hex_zoom(1.0)
                        camera_pan = [0.0, 0.0]
                        if _is_human_agent_key(_current_agent_key(env, red_index, blue_index)):
                            env.start_faction_turn()
                            active_human_faction = env.factions[env.current_player]
                        in_setup = False
            continue

        current_faction = env.factions[env.current_player]
        current_agent_key = _current_agent_key(env, red_index, blue_index)
        human_turn_active = _is_human_agent_key(current_agent_key)
        has_human_players = _match_has_human_players(red_index, blue_index)
        if human_turn_active and active_human_faction != current_faction:
            env.start_faction_turn()
            active_human_faction = current_faction
            human_moved_units = set()
            human_turn_actions = []
            human_turn_log = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    in_setup = True
                    show_tech_tree = False
                    human_moved_units = set()
                    human_turn_actions = []
                    human_turn_log = []
                    active_human_faction = None
                    selected_tile = None
                    selected_unit_id = None
                    pending_build_type = None
                    camera_zoom = _set_hex_zoom(1.0)
                    camera_pan = [0.0, 0.0]
                elif event.key == pygame.K_t:
                    show_tech_tree = not show_tech_tree
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_p:
                    snapshot_text = _build_debug_snapshot(
                        env,
                        AGENT_SPECS[red_index].label,
                        AGENT_SPECS[blue_index].label,
                        red_info,
                        blue_info,
                        turn_history,
                    )
                    output_path = _write_debug_snapshot(snapshot_text)
                    print(snapshot_text)
                    print(f"\nSaved debug snapshot to: {output_path}\n")
                elif event.key == pygame.K_g and human_turn_active and env.winner() is None:
                    selected_unit = next((u for u in env.units if u.id == selected_unit_id), None)
                    if _can_human_gather(env, selected_unit, human_moved_units):
                        action = ("gather", selected_unit.id)
                        ok, reason = env.apply_action(action)
                        if ok:
                            human_turn_actions.append(action)
                            human_turn_log.append(reason)
                            effects.extend(_effects_from_actions(env, [action], current_faction))
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if env.winner() is None:
                        if not has_human_players:
                            red_info, blue_info, red_actions, blue_actions = _step_full_turn(env, red_agent, blue_agent)
                            turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))
                            effects.extend(_effects_from_actions(env, red_actions, "Red"))
                            effects.extend(_effects_from_actions(env, blue_actions, "Blue"))
                        else:
                            if human_turn_active:
                                completed_info = _build_turn_info(
                                    current_faction,
                                    human_turn_actions,
                                    human_turn_log or ["stop"],
                                    list(env.current_events),
                                )
                                if current_faction == "Red":
                                    red_info = completed_info
                                else:
                                    blue_info = completed_info

                                human_moved_units = set()
                                human_turn_actions = []
                                human_turn_log = []
                                active_human_faction = None
                                pending_build_type = None
                                env.step_end_turn()
                                if env.current_player == 0:
                                    turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))

                            red_info, blue_info, new_rounds, new_effects = _advance_until_human_or_end(
                                env,
                                red_agent,
                                blue_agent,
                                red_index,
                                blue_index,
                                red_info,
                                blue_info,
                            )
                            turn_history.extend(new_rounds)
                            effects.extend(new_effects)
                        event_scroll = 0

            if event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if event_panel_rect.width > 0 and event_panel_rect.collidepoint(mouse_pos):
                    event_scroll = max(0, event_scroll - event.y)
                elif board_backdrop.collidepoint(mouse_pos):
                    old_zoom = camera_zoom
                    old_board_width, old_board_height = _board_pixel_size(env)
                    old_origin = _board_origin_in_viewport(board_rect, old_board_width, old_board_height, tuple(camera_pan))
                    zoom_delta = ZOOM_STEP if event.y > 0 else -ZOOM_STEP
                    camera_zoom = _set_hex_zoom(camera_zoom + zoom_delta)
                    new_board_width, new_board_height = _board_pixel_size(env)
                    centered_origin = _board_origin_in_viewport(board_rect, new_board_width, new_board_height)
                    scale = camera_zoom / old_zoom if old_zoom else 1.0
                    camera_pan[0] = mouse_pos[0] - scale * (mouse_pos[0] - old_origin[0]) - centered_origin[0]
                    camera_pan[1] = mouse_pos[1] - scale * (mouse_pos[1] - old_origin[1]) - centered_origin[1]
                    camera_pan[0], camera_pan[1] = _clamp_camera_pan(
                        board_rect,
                        new_board_width,
                        new_board_height,
                        tuple(camera_pan),
                    )

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(event.pos) and env.winner() is None:
                    if not has_human_players:
                        red_info, blue_info, red_actions, blue_actions = _step_full_turn(env, red_agent, blue_agent)
                        turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))
                        effects.extend(_effects_from_actions(env, red_actions, "Red"))
                        effects.extend(_effects_from_actions(env, blue_actions, "Blue"))
                    else:
                        if human_turn_active:
                            completed_info = _build_turn_info(
                                current_faction,
                                human_turn_actions,
                                human_turn_log or ["stop"],
                                list(env.current_events),
                            )
                            if current_faction == "Red":
                                red_info = completed_info
                            else:
                                blue_info = completed_info

                            human_moved_units = set()
                            human_turn_actions = []
                            human_turn_log = []
                            active_human_faction = None
                            pending_build_type = None
                            env.step_end_turn()
                            if env.current_player == 0:
                                turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))

                        red_info, blue_info, new_rounds, new_effects = _advance_until_human_or_end(
                            env,
                            red_agent,
                            blue_agent,
                            red_index,
                            blue_index,
                            red_info,
                            blue_info,
                        )
                        turn_history.extend(new_rounds)
                        effects.extend(new_effects)
                    event_scroll = 0
                elif show_tech_tree and any(node_rect.collidepoint(event.pos) for node_rect in tech_tree_node_rects.values()):
                    if human_turn_active and env.winner() is None:
                        clicked_tech_id = next(
                            (tech_id for tech_id, node_rect in tech_tree_node_rects.items() if node_rect.collidepoint(event.pos)),
                            None,
                        )
                        if clicked_tech_id is not None and tech.can_research(env, current_faction, clicked_tech_id):
                            payload = ("research", clicked_tech_id)
                            ok, reason = env.apply_action(payload)
                            if ok:
                                human_turn_actions.append(payload)
                                human_turn_log.append(reason)
                                effects.extend(_effects_from_actions(env, [payload], current_faction))
                elif any(button_rect.collidepoint(event.pos) for button_rect, _, _ in human_action_button_rects):
                    for button_rect, payload, enabled in human_action_button_rects:
                        if not button_rect.collidepoint(event.pos):
                            continue
                        if not enabled:
                            break
                        if isinstance(payload, tuple) and payload and payload[0] == "build_mode":
                            pending_build_type = None if pending_build_type == payload[1] else payload[1]
                        else:
                            ok, reason = env.apply_action(payload)
                            if ok:
                                human_turn_actions.append(payload)
                                human_turn_log.append(reason)
                                effects.extend(_effects_from_actions(env, [payload], current_faction))
                                pending_build_type = None
                        break
                elif selected_unit_close_rect.collidepoint(event.pos):
                    selected_unit_id = None
                    selected_tile = None
                    pending_build_type = None
                elif tech_btn_rect.collidepoint(event.pos):
                    show_tech_tree = not show_tech_tree
                elif camera_btn_rect.collidepoint(event.pos):
                    camera_zoom = _set_hex_zoom(1.0)
                    camera_pan = [0.0, 0.0]
                elif board_rect.inflate(18, 18).collidepoint(event.pos):
                    clicked_tile = hexgrid.nearest_hex(
                        event.pos,
                        env.config.width,
                        env.config.height,
                        HEX_SIZE,
                        board_origin,
                    )
                    if clicked_tile is not None:
                        selected_unit = next((u for u in env.units if u.id == selected_unit_id), None)
                        valid_build_targets = (
                            _human_build_targets(env, selected_unit, pending_build_type)
                            if human_turn_active and pending_build_type is not None and selected_unit is not None
                            else []
                        )
                        valid_targets = (
                            _valid_human_move_targets(env, selected_unit, human_moved_units)
                            if human_turn_active and pending_build_type is None and selected_unit is not None
                            else []
                        )
                        if clicked_tile in valid_build_targets and selected_unit is not None and pending_build_type is not None:
                            action = ("build", selected_unit.id, pending_build_type, clicked_tile)
                            ok, reason = env.apply_action(action)
                            if ok:
                                human_turn_actions.append(action)
                                human_turn_log.append(reason)
                                effects.extend(_effects_from_actions(env, [action], current_faction))
                                pending_build_type = None
                                selected_tile = clicked_tile
                                selected_unit_id = None
                        elif (
                            human_turn_active
                            and selected_unit is not None
                            and clicked_tile == selected_unit.position
                            and _can_human_gather(env, selected_unit, human_moved_units)
                        ):
                            action = ("gather", selected_unit.id)
                            ok, reason = env.apply_action(action)
                            if ok:
                                human_turn_actions.append(action)
                                human_turn_log.append(reason)
                                effects.extend(_effects_from_actions(env, [action], current_faction))
                                selected_tile = selected_unit.position
                                pending_build_type = None
                        elif clicked_tile in valid_targets and selected_unit is not None:
                            action = ("move_towards", selected_unit.id, clicked_tile)
                            ok, reason = env.apply_action(action)
                            if ok:
                                human_turn_actions.append(action)
                                human_turn_log.append(reason)
                                effects.extend(_effects_from_actions(env, [action], current_faction))
                                selected_tile = selected_unit.position
                                pending_build_type = None
                            else:
                                selected_tile = clicked_tile
                        else:
                            selected_tile = clicked_tile

                        clicked_unit = next((u for u in env.units if u.position == selected_tile), None)
                        selected_unit_id = clicked_unit.id if clicked_unit is not None else None
                        if selected_unit_id is None or clicked_unit is None or clicked_unit.unit_type != "worker":
                            pending_build_type = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                if board_backdrop.collidepoint(event.pos):
                    panning = True
                    pan_anchor_mouse = event.pos
                    pan_anchor = (camera_pan[0], camera_pan[1])
                    pan_drag_distance = 0.0
            elif event.type == pygame.MOUSEMOTION and panning:
                dx = event.pos[0] - pan_anchor_mouse[0]
                dy = event.pos[1] - pan_anchor_mouse[1]
                board_width, board_height = _board_pixel_size(env)
                camera_pan[0], camera_pan[1] = _clamp_camera_pan(
                    board_rect,
                    board_width,
                    board_height,
                    (pan_anchor[0] + dx, pan_anchor[1] + dy),
                )
                pan_drag_distance = max(pan_drag_distance, math.hypot(dx, dy))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2 and panning:
                panning = False
                if pan_drag_distance < 8:
                    camera_zoom = _set_hex_zoom(1.0)
                    camera_pan = [0.0, 0.0]

        effects = _update_effects(effects)
        board_width, board_height = _board_pixel_size(env)
        camera_pan[0], camera_pan[1] = _clamp_camera_pan(board_rect, board_width, board_height, tuple(camera_pan))
        board_origin = _board_origin_in_viewport(board_rect, board_width, board_height, tuple(camera_pan))
        board_backdrop = board_rect.inflate(22, 28)
        screen.fill((9, 13, 18))
        hud_rect = pygame.Rect(0, 0, width_px, top_bar)
        pygame.draw.rect(screen, HUD_BG, hud_rect)
        pygame.draw.line(screen, PANEL_SOFT, (0, top_bar - 1), (width_px, top_bar - 1), 2)
        sidebar_x = board_x + default_board_width + pad
        sidebar_rect = pygame.Rect(sidebar_x, pad, side_panel, height_px - pad * 2)
        _draw_panel(screen, sidebar_rect, fill=(16, 22, 29), border=PANEL_SOFT, radius=16)

        red_workers = sum(1 for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        blue_workers = sum(1 for u in env.units if u.faction == "Blue" and u.unit_type == "worker")
        red_military = sum(1 for u in env.units if u.faction == "Red" and u.attack_damage > 0)
        blue_military = sum(1 for u in env.units if u.faction == "Blue" and u.attack_damage > 0)
        winner = env.winner()
        current_agent_key = _current_agent_key(env, red_index, blue_index)
        human_turn_active = _is_human_agent_key(current_agent_key)
        mouse_pos = pygame.mouse.get_pos()
        current_research_count = len(_available_research_ids(env, current_faction))
        turn_button_label = "End Turn" if human_turn_active and winner is None else "Next Turn"

        # ── Three-zone top HUD ────────────────────────────────────────────────
        red_bar_rect  = pygame.Rect(board_x, pad, faction_bar_w, HUD_H)
        blue_bar_rect = pygame.Rect(board_x + default_board_width - faction_bar_w, pad, faction_bar_w, HUD_H)
        center_rect   = pygame.Rect(center_zone_x, pad, center_zone_w, HUD_H)

        red_stance  = "Defense" if defense_mode_active(env, "Red")  else "Push" if push_mode_active(env, "Red")  else "Field"
        blue_stance = "Defense" if defense_mode_active(env, "Blue") else "Push" if push_mode_active(env, "Blue") else "Field"
        max_base_hp = env.base_max_hp(current_faction)

        _draw_faction_bar(screen, small, tiny, board_assets, red_bar_rect,
            "Red", AGENT_SPECS[red_index].label,
            env.bank["Red"], red_workers, red_military,
            env.bases["Red"].hp, env.base_max_hp("Red"),
            env.current_era(), red_stance, RED_PRIMARY)

        _draw_faction_bar(screen, small, tiny, board_assets, blue_bar_rect,
            "Blue", AGENT_SPECS[blue_index].label,
            env.bank["Blue"], blue_workers, blue_military,
            env.bases["Blue"].hp, env.base_max_hp("Blue"),
            env.current_era(), blue_stance, BLUE_PRIMARY)

        _draw_center_hud(screen, big, small, tiny, board_assets, center_rect, btn_rect,
            env, winner, human_turn_active, turn_button_label, current_faction)

        # ── Right sidebar ─────────────────────────────────────────────────────
        # Cap panel heights so research + tactical + event all fit inside the sidebar
        _min_event_h = 140
        _btn_area = 8 + 30 + 10  # gap-before + btn-height + gap-after
        _panel_budget = max(200, sidebar_rect.bottom - 8 - (pad + 12) - _min_event_h - _btn_area)
        research_panel_h = min(_research_panel_height(font, tiny, env, side_panel - 24), _panel_budget * 55 // 100)
        research_panel = pygame.Rect(sidebar_x + 12, pad + 12, side_panel - 24, research_panel_h)
        _draw_research_panel(screen, font, tiny, board_assets, env, research_panel)
        tech_btn_rect = pygame.Rect(research_panel.x, research_panel.bottom + 8, 132, 30)
        _draw_small_button(screen, tiny, tech_btn_rect, "Research Tree", active=show_tech_tree)
        if current_research_count > 0:
            _draw_badge(screen, tiny, (tech_btn_rect.right - 8, tech_btn_rect.y + 5), str(current_research_count))
        if tech_btn_rect.collidepoint(mouse_pos):
            hover_lines = [
                "Open the research tree",
                (
                    f"{current_research_count} tech{'s' if current_research_count != 1 else ''} available for {current_faction}."
                    if current_research_count > 0
                    else f"No techs currently available for {current_faction}."
                ),
                "Hover nodes for details. Click a ready node to start it.",
            ]
            hover_rect = pygame.Rect(tech_btn_rect.right + 10, tech_btn_rect.y - 8, 190, 78)
            _draw_hover_tile_panel(screen, small, tiny, board_assets, hover_rect, hover_lines)

        tactical_panel_h = min(_tactical_panel_height(font, tiny, env, side_panel - 24), _panel_budget - research_panel_h)
        tactical_panel = pygame.Rect(sidebar_x + 12, tech_btn_rect.bottom + 10, side_panel - 24, tactical_panel_h)
        _draw_tactical_panel(screen, font, tiny, board_assets, env, tactical_panel)

        event_panel_h = max(_min_event_h, sidebar_rect.bottom - 8 - (tactical_panel.bottom + 12))
        event_panel = pygame.Rect(sidebar_x + 12, tactical_panel.bottom + 12, side_panel - 24, event_panel_h)
        event_panel_rect = event_panel
        _draw_event_panel(screen, font, tiny, board_assets, env.recent_events[-14:], event_panel, event_scroll)

        hover_pos = None
        if board_rect.inflate(24, 24).collidepoint(mouse_pos):
            hover_pos = hexgrid.nearest_hex(mouse_pos, env.config.width, env.config.height, HEX_SIZE, board_origin)

        shadow_rect = board_backdrop.inflate(18, 22)
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (*BOARD_SHADOW, 105), shadow_surface.get_rect(), border_radius=28)
        screen.blit(shadow_surface, (shadow_rect.x + 6, shadow_rect.y + 10))
        _draw_panel(screen, board_backdrop, fill=(13, 19, 26), border=PANEL_SOFT, radius=20)
        inset = board_backdrop.inflate(-10, -10)
        pygame.draw.rect(screen, (17, 24, 32), inset, border_radius=18)
        zoom_label = f"{int(round(camera_zoom * 100))}%"
        zoom_badge = pygame.Rect(board_backdrop.right - 158, board_backdrop.y + 12, 64, 28)
        pygame.draw.rect(screen, PANEL_INSET, zoom_badge, border_radius=10)
        pygame.draw.rect(screen, PANEL_SOFT, zoom_badge, width=1, border_radius=10)
        _draw_shadow_text(
            screen,
            tiny,
            zoom_label,
            zoom_badge.centerx - tiny.size(zoom_label)[0] // 2,
            zoom_badge.y + 6,
            TEXT_PRIMARY,
            shadow=(10, 12, 16),
            shadow_offset=1,
        )
        camera_btn_rect = pygame.Rect(board_backdrop.right - 86, board_backdrop.y + 11, 74, 30)
        _draw_small_button(screen, tiny, camera_btn_rect, "Reset")

        selected_unit = next((u for u in env.units if u.id == selected_unit_id), None)
        valid_move_targets: set[tuple[int, int]] = set()
        valid_build_targets: set[tuple[int, int]] = set()
        if human_turn_active and selected_unit is not None:
            valid_move_targets = set(_valid_human_move_targets(env, selected_unit, human_moved_units))
            if pending_build_type is not None and selected_unit.unit_type == "worker":
                valid_build_targets = set(_human_build_targets(env, selected_unit, pending_build_type))
                valid_move_targets = set()

        previous_clip = screen.get_clip()
        screen.set_clip(inset)
        for row in range(env.config.height):
            for col in range(env.config.width):
                pos = (col, row)
                points = _hex_points(col, row, board_origin)
                bounds = _hex_bounds(col, row, board_origin)
                terrain_kind = _terrain_kind(env, pos)
                terrain_sprite = board_assets.terrain_tile(terrain_kind, col * 17 + row * 23)
                scaled_tile = _safe_scale(terrain_sprite, (HEX_WIDTH, HEX_HEIGHT))
                if scaled_tile is not None:
                    screen.blit(scaled_tile, bounds.topleft)
                else:
                    fill = GRID_BG if (col + row) % 2 == 0 else GRID_ALT
                    pygame.draw.polygon(screen, fill, points)
                pygame.draw.polygon(screen, GRID_LINE, points, width=1)

                if pos in valid_build_targets:
                    highlight = pygame.Surface((bounds.width, bounds.height), pygame.SRCALPHA)
                    pygame.draw.polygon(
                        highlight,
                        (*HEX_BUILD_FILL, 72),
                        [(x - bounds.x, y - bounds.y) for x, y in points],
                    )
                    screen.blit(highlight, bounds.topleft)
                    pygame.draw.polygon(screen, HEX_BUILD_LINE, points, width=3)
                elif pos in valid_move_targets:
                    highlight = pygame.Surface((bounds.width, bounds.height), pygame.SRCALPHA)
                    pygame.draw.polygon(
                        highlight,
                        (*HEX_MOVE_FILL, 68),
                        [(x - bounds.x, y - bounds.y) for x, y in points],
                    )
                    screen.blit(highlight, bounds.topleft)
                    pygame.draw.polygon(screen, HEX_MOVE_LINE, points, width=3)

                if selected_tile == pos:
                    highlight = pygame.Surface((bounds.width, bounds.height), pygame.SRCALPHA)
                    pygame.draw.polygon(
                        highlight,
                        (*HEX_SELECT_FILL, 72),
                        [(x - bounds.x, y - bounds.y) for x, y in points],
                    )
                    screen.blit(highlight, bounds.topleft)
                    pygame.draw.polygon(screen, HEX_SELECT_LINE, points, width=4)
                elif hover_pos == pos:
                    highlight = pygame.Surface((bounds.width, bounds.height), pygame.SRCALPHA)
                    pygame.draw.polygon(
                        highlight,
                        (*HEX_HOVER_FILL, 56),
                        [(x - bounds.x, y - bounds.y) for x, y in points],
                    )
                    screen.blit(highlight, bounds.topleft)
                    pygame.draw.polygon(screen, GRID_HOVER, points, width=3)

        visible_special_resources = {
            resource.id
            for resource in env.resources
            if resource.required_tech is None
            or any(resource.required_tech in env.faction_state(faction).techs_unlocked for faction in env.factions)
        }
        for r in env.resources:
            if r.id not in visible_special_resources:
                continue
            center = _hex_center(r.position[0], r.position[1], board_origin)

            resource_scale = max(20, int(HEX_SIZE * 1.15))
            resource_shadow_w = max(18, int(HEX_SIZE * 1.1))
            resource_shadow_h = max(8, int(HEX_SIZE * 0.45))
            resource_offset_y = int(HEX_SIZE * 0.58)

            _draw_soft_shadow(
                screen,
                (center[0], center[1] + resource_offset_y - 4),
                resource_shadow_w,
                resource_shadow_h,
                alpha=60,
            )

            sprite_key = "horses" if r.resource_type == "horses" else "stone" if r.resource_type == "stone" else "resource"
            _draw_asset_marker(
                screen,
                board_assets.object_sprite(sprite_key),
                center,
                lambda resource=r, resource_center=center: _draw_resource_icon(screen, resource, resource_center, tiny),
                scale=(resource_scale, resource_scale),
                offset_y=resource_offset_y,
            )

        for faction, base in env.bases.items():
            color = RED_PRIMARY if faction == "Red" else BLUE_PRIMARY
            center = _hex_center(base.position[0], base.position[1], board_origin)
            bounds = _hex_bounds(base.position[0], base.position[1], board_origin)

            base_scale = max(30, int(HEX_SIZE * 1.8))
            base_shadow_w = max(24, int(HEX_SIZE * 1.6))
            base_shadow_h = max(10, int(HEX_SIZE * 0.6))
            base_offset_y = int(HEX_SIZE * 0.7)
            base_ring_radius = max(14, int(HEX_SIZE * 0.85))
            base_circle_radius = max(12, int(HEX_SIZE * 0.65))

            _draw_soft_shadow(
                screen,
                (center[0], center[1] + base_offset_y - 2),
                base_shadow_w,
                base_shadow_h,
                alpha=78,
            )
            pygame.draw.circle(screen, (*color, 90), (center[0], center[1] + 2), base_ring_radius, width=4)
            _draw_asset_marker(
                screen,
                board_assets.object_sprite("base"),
                center,
                lambda base_center=center, faction_color=color: pygame.draw.circle(screen, faction_color, base_center, base_circle_radius),
                tint=color,
                scale=(base_scale, base_scale),
                offset_y=base_offset_y,
            )

            hp_chip = pygame.Rect(bounds.centerx - 18, bounds.bottom - 12, 36, 22)
            pygame.draw.rect(screen, (16, 21, 28), hp_chip, border_radius=10)
            pygame.draw.rect(screen, color, hp_chip, width=2, border_radius=10)
            _draw_shadow_text(screen, tiny, str(base.hp), hp_chip.x + 10, hp_chip.y + 2, TEXT_PRIMARY, shadow=(8, 10, 14), shadow_offset=1)

            for b in env.buildings:
                color = (213, 136, 104) if b.faction == "Red" else (123, 164, 230)
                center = _hex_center(b.position[0], b.position[1], board_origin)

                building_scale = max(24, int(HEX_SIZE * 1.4))
                building_rect_size = max(20, int(HEX_SIZE * 1.15))
                building_shadow_w = max(20, int(HEX_SIZE * 1.35))
                building_shadow_h = max(8, int(HEX_SIZE * 0.5))
                building_offset_y = int(HEX_SIZE * 0.64)

                rect = pygame.Rect(
                    center[0] - building_rect_size // 2,
                    center[1] - int(HEX_SIZE * 0.7),
                    building_rect_size,
                    building_rect_size,
                )

                _draw_soft_shadow(
                    screen,
                    (center[0], center[1] + building_offset_y - 2),
                    building_shadow_w,
                    building_shadow_h,
                    alpha=70,
                )
                _draw_asset_marker(
                    screen,
                    board_assets.object_sprite(b.building_type),
                    center,
                    lambda building=b, building_rect=rect, border=color: _draw_building_icon(screen, building, building_rect, border),
                    tint=color,
                    scale=(building_scale, building_scale),
                    offset_y=building_offset_y,
                )

        for u in env.units:
            cx, cy = _hex_center(u.position[0], u.position[1], board_origin)
            color = (242, 206, 142) if u.faction == "Red" else (189, 225, 255)
            border = RED_PRIMARY if u.faction == "Red" else BLUE_PRIMARY

            unit_size = max(18, int(HEX_SIZE * 0.97))
            unit_shadow_w = max(16, int(HEX_SIZE * 0.9))
            unit_shadow_h = max(7, int(HEX_SIZE * 0.38))
            unit_offset_y = int(HEX_SIZE * 0.03)

            select_radius = max(14, int(HEX_SIZE * 0.78))
            select_outer_radius = max(16, int(HEX_SIZE * 0.87))
            select_y = cy + int(HEX_SIZE * 0.06)

            _draw_soft_shadow(screen, (cx, cy + int(HEX_SIZE * 0.58)), unit_shadow_w, unit_shadow_h, alpha=64)

            if selected_unit_id == u.id:
                pygame.draw.circle(screen, (*HEX_SELECT_FILL, 90), (cx, select_y), select_radius, width=4)
                pygame.draw.circle(screen, (*border, 105), (cx, select_y), select_outer_radius, width=2)

            draw_center = (cx, cy + unit_offset_y)
            if not _draw_unit_sprite(screen, board_assets, env, u, draw_center, color, border, size=unit_size):
                _draw_unit_icon(screen, u, draw_center, color, border)

        _draw_effects(screen, effects, board_origin[0], board_origin[1], HEX_WIDTH, small)
        screen.set_clip(previous_clip)

        if hover_pos is not None:
            hover_lines = _hover_tile_lines(env, hover_pos)
            if hover_lines:
                tile_rect = _hex_bounds(hover_pos[0], hover_pos[1], board_origin)
                hover_rect = _hover_panel_rect(screen, tiny, tile_rect, hover_lines)
                _draw_hover_tile_panel(screen, small, tiny, board_assets, hover_rect, hover_lines)

        if selected_unit is None:
            selected_unit_id = None
            selected_unit_close_rect = pygame.Rect(0, 0, 0, 0)
        else:
            inspect_rect = pygame.Rect(board_backdrop.x + 18, board_backdrop.y + 18, 300, 268)
            selected_unit_close_rect = _draw_selected_unit_panel(
                screen,
                small,
                tiny,
                tiny,
                board_assets,
                env,
                selected_unit,
                inspect_rect,
            )
        if selected_unit_id is None and selected_tile is not None:
            inspect_rect = pygame.Rect(board_backdrop.x + 18, board_backdrop.y + 18, 300, 252)
            selected_building = next((b for b in env.buildings if b.position == selected_tile), None)
            selected_resource = next((r for r in env.resources if r.position == selected_tile and r.remaining > 0), None)
            selected_base_entry = next(((faction, base) for faction, base in env.bases.items() if base.position == selected_tile), None)
            if selected_building is not None:
                spec = production.building_stats(env, selected_building.faction, selected_building.building_type)
                max_hp = spec.hp if spec is not None else selected_building.hp
                selected_unit_close_rect = _draw_selected_object_panel(
                    screen,
                    small,
                    tiny,
                    tiny,
                    board_assets,
                    env,
                    inspect_rect,
                    title=_building_label(selected_building.building_type),
                    subtitle=f"{selected_building.faction} Building",
                    lines=_selected_building_lines(env, selected_building)[1:],
                    hp_value=selected_building.hp,
                    hp_max=max_hp,
                    icon_kind=selected_building.building_type,
                    accent=RED_PRIMARY if selected_building.faction == "Red" else BLUE_PRIMARY,
                )
            elif selected_base_entry is not None:
                faction, selected_base = selected_base_entry
                selected_unit_close_rect = _draw_selected_object_panel(
                    screen,
                    small,
                    tiny,
                    tiny,
                    board_assets,
                    env,
                    inspect_rect,
                    title=f"{faction} Base",
                    subtitle="Capital stronghold",
                    lines=_selected_base_lines(selected_base, faction)[1:],
                    hp_value=selected_base.hp,
                    hp_max=env.base_max_hp(base.faction),
                    icon_kind="base",
                    accent=RED_PRIMARY if faction == "Red" else BLUE_PRIMARY,
                )
            elif selected_resource is not None:
                resource_kind = "horses" if selected_resource.resource_type == "horses" else "stone" if selected_resource.resource_type == "stone" else "resource"
                selected_unit_close_rect = _draw_selected_object_panel(
                    screen,
                    small,
                    tiny,
                    tiny,
                    board_assets,
                    env,
                    inspect_rect,
                    title=RESOURCE_LABELS.get(selected_resource.resource_type, selected_resource.resource_type.title()),
                    subtitle="Map resource",
                    lines=_selected_resource_lines(selected_resource)[1:],
                    icon_kind=resource_kind,
                    accent=(136, 170, 102),
                )
            else:
                selected_unit_close_rect = pygame.Rect(0, 0, 0, 0)

        human_action_button_rects = []
        if human_turn_active:
            panel_title, panel_options, panel_hint = _human_action_options(
                env,
                current_faction,
                selected_unit,
                selected_tile,
                pending_build_type,
            )
            if panel_title is not None:
                action_panel_rect = pygame.Rect(
                    board_backdrop.x + 18,
                    board_backdrop.y + 292,
                    300,
                    _human_action_panel_height(tiny, panel_options, panel_hint),
                )
                human_action_button_rects = _draw_human_action_panel(
                    screen,
                    small,
                    tiny,
                    board_assets,
                    action_panel_rect,
                    panel_title,
                    panel_options,
                    panel_hint,
                )
        else:
            pending_build_type = None

        if show_debug:
            debug_rect = pygame.Rect(board_x, height_px - 44, 258, 30)
            pygame.draw.rect(screen, (16, 22, 30), debug_rect, border_radius=10)
            pygame.draw.rect(screen, PANEL_SOFT, debug_rect, width=1, border_radius=10)
            _draw_text_block(
                screen,
                tiny,
                ["Debug: D  Snapshot: P  Tech tree: T  Gather: G  Reset: R  Wheel zoom  Middle click reset"],
                debug_rect.x + 10,
                debug_rect.y + 7,
                TEXT_MUTED,
                16,
            )

        tech_tree_node_rects = {}
        if show_tech_tree:
            overlay_rect = pygame.Rect(pad * 2, pad * 2, width_px - pad * 4, height_px - pad * 4)
            tech_tree_node_rects = _draw_tech_tree_overlay(
                screen,
                font,
                tiny,
                board_assets,
                env,
                overlay_rect,
                current_faction,
                human_turn_active,
                mouse_pos,
            )

        pygame.display.flip()

    pygame.quit()
