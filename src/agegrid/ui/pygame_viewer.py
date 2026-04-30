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
from src.agegrid.ui.camera import (
    CameraState,
    apply_zoom as _apply_camera_zoom,
    begin_pan as _begin_camera_pan,
    board_origin_in_viewport as _board_origin_in_viewport,
    clamp_camera_pan as _clamp_camera_pan,
    end_pan as _end_camera_pan,
    reset_camera_state as _reset_camera_state,
    update_pan as _update_camera_pan,
)
from src.agegrid.ui.research_view import (
    ResearchDrawHelpers,
    available_research_ids as _available_research_ids,
    clamp_tech_tree_view as _clamp_tech_tree_view,
    draw_research_panel as _draw_research_panel,
    draw_tech_tree_overlay as _draw_tech_tree_overlay,
    research_panel_height as _research_panel_height,
    tech_tree_layout as _tech_tree_layout,
    tech_label as _tech_label,
)
from src.agegrid.ui.viewer_panels import (
    PanelColors,
    PanelDrawHelpers,
    PanelText,
    draw_hover_tile_panel as _draw_hover_tile_panel,
    draw_human_action_panel as _draw_human_action_panel,
    draw_selected_object_panel as _draw_selected_object_panel,
    draw_selected_unit_panel as _draw_selected_unit_panel,
    human_action_panel_height as _human_action_panel_height,
    selected_base_lines as _selected_base_lines,
    selected_building_lines as _selected_building_lines,
    selected_resource_lines as _selected_resource_lines,
)
from src.agegrid.ui.turn_trace import (
    FactionTurnInfo,
    TurnSnapshot,
    build_debug_snapshot,
    build_turn_info,
    step_faction_with_trace,
    step_full_turn,
    turn_snapshot,
    write_debug_snapshot,
)


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
HUD_H = 158           # height of the three top-bar panels
HUD_INNER_PAD = 9     # inner padding inside each HUD panel
HUD_FACTION_W = 280   # each faction bar width; may be capped to fit board width

# Text tones for drawing on top of parchment/asset panel backgrounds
PARCH_TITLE = (68, 50, 34)    # main title on parchment
PARCH_BODY = (82, 63, 42)     # body text on parchment
PARCH_MUTED = (106, 84, 60)   # muted/secondary on parchment
PARCH_SHADOW = (240, 224, 196)
PARCH_ACCENT = (150, 120, 84)
PARCH_LINE = (188, 168, 136)
PARCH_GOOD = (72, 120, 70)
PARCH_WARN = (170, 112, 48)
PARCH_DANGER = (164, 76, 60)
PARCH_INFO = (72, 106, 152)
GRID_BG = (24, 30, 38)
GRID_ALT = (28, 35, 43)
GRID_LINE = (61, 72, 84)
GRID_HOVER = (188, 214, 234)
RESOURCE_GLOW = (109, 192, 116)
RESOURCE_CONTESTED = (226, 86, 76)
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
DEFAULT_VIEWER_ZOOM = 1.2
BOARD_VIEWPORT_EXTRA_W = 180
BOARD_VIEWPORT_EXTRA_H = 96

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

def _building_label(building_id: str) -> str:
    return BUILDING_LABELS.get(building_id, building_id.replace("_", " ").title())


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
    resource = env.resource_at_for_faction(unit.position, unit.faction)
    return resource is not None and env.can_gather_resource(unit, resource)


def _human_gather_block_reason(env: AgeGridEnv, unit) -> str | None:
    if unit is None or unit.unit_type != "worker":
        return "Select a worker."
    resource = env.resource_at_for_faction(unit.position, unit.faction)
    if resource is None:
        return "Stand on a resource tile."
    if not env.can_gather_resource(unit, resource):
        return "Resource contested by enemy military."
    return None


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
            if gather_reason is None:
                gather_reason = _human_gather_block_reason(env, selected_unit)
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
        info, actions = step_faction_with_trace(env, agent)
        effects.extend(_effects_from_actions(env, actions, current_faction))

        if current_faction == "Red":
            red_info = info
        else:
            blue_info = info

        env.step_end_turn()
        if env.current_player == 0:
            completed_rounds.append(turn_snapshot(env, red_info, blue_info))

    return red_info, blue_info, completed_rounds, effects


def _step_ai_faction(
    env: AgeGridEnv,
    red_agent,
    blue_agent,
    red_info: FactionTurnInfo,
    blue_info: FactionTurnInfo,
) -> tuple[FactionTurnInfo, FactionTurnInfo, TurnSnapshot | None, list[VisualEffect]]:
    """Run exactly one AI faction's turn and end it. Returns updated infos, an optional
    completed-round snapshot (when the round boundary is crossed), and visual effects."""
    faction = env.factions[env.current_player]
    agent = red_agent if faction == "Red" else blue_agent
    info, actions = step_faction_with_trace(env, agent)
    effects = _effects_from_actions(env, actions, faction)
    if faction == "Red":
        red_info = info
    else:
        blue_info = info
    env.step_end_turn()
    snapshot = turn_snapshot(env, red_info, blue_info) if env.current_player == 0 else None
    return red_info, blue_info, snapshot, effects


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


def _relation_color(relation: str) -> tuple[int, int, int]:
    lowered = relation.lower()
    if lowered == "peace":
        return PARCH_GOOD
    if lowered == "war":
        return PARCH_DANGER
    if lowered == "truce":
        return PARCH_WARN
    return PARCH_MUTED


def _income_color(income: int) -> tuple[int, int, int]:
    if income > 0:
        return PARCH_GOOD
    if income < 0:
        return PARCH_DANGER
    return PARCH_MUTED


def _base_hp_color(base_hp: int, max_base_hp: int) -> tuple[int, int, int]:
    if max_base_hp <= 0:
        return (175, 50, 45)
    ratio = base_hp / max_base_hp
    if ratio <= 0.33:
        return (175, 50, 45)   # vivid red
    if ratio <= 0.66:
        return (190, 120, 30)  # vivid amber
    return (45, 145, 45)       # vivid green


def _era_color(era: str) -> tuple[int, int, int]:
    if "Engineering" in era:
        return GOLD_ACCENT             # golden
    if "Iron" in era:
        return (145, 155, 168)         # steel grey
    if "Bronze" in era:
        return (185, 118, 52)          # bronze
    if "Stone" in era:
        return (140, 125, 110)         # grey-brown
    return (180, 160, 130)             # founding tan


def _draw_faction_bar(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    faction: str,
    agent_label: str,
    bank: int,
    income: int,
    army_summary: str,
    base_hp: int,
    max_base_hp: int,
    era: str,
    relation: str,
    research_label: str,
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
    _draw_shadow_text(surface, body_font, faction, ix, y, accent, shadow=PARCH_SHADOW, shadow_offset=1)
    era_w = small_font.size(era)[0]
    _draw_shadow_text(surface, small_font, era, rect.right - era_w - HUD_INNER_PAD, y + 3, _era_color(era), shadow=PARCH_SHADOW, shadow_offset=1)
    y += body_font.get_height() + 6

    # Row 2: economy / diplomacy / army
    stat_items = [
        (f"$ {bank}", PARCH_TITLE),
        (f"{income:+}/turn", _income_color(income)),
        (relation, _relation_color(relation)),
    ]
    chip_gap = 4
    chip_w = (iw - chip_gap * 2) // 3
    chip_h = 28
    inset_sprite = board_assets.ui_sprite("panelInset_beigeLight") or board_assets.ui_sprite("panelInset_beige")
    for i, (text, text_color) in enumerate(stat_items):
        chip_rect = pygame.Rect(ix + i * (chip_w + chip_gap), y, chip_w, chip_h)
        if not _draw_scaled_sprite(surface, inset_sprite, chip_rect):
            pygame.draw.rect(surface, PANEL_INSET, chip_rect, border_radius=6)
            pygame.draw.rect(surface, PANEL_SOFT, chip_rect, width=1, border_radius=6)
        label = _fit_text(text, small_font, chip_w - 8)
        tx = chip_rect.centerx - small_font.size(label)[0] // 2
        _draw_shadow_text(surface, small_font, label, tx, chip_rect.y + 6, text_color, shadow=PARCH_SHADOW, shadow_offset=1)
    y += chip_h + 6

    # Row 3: research + army summary
    research_text = _fit_text(f"Research {research_label}", small_font, iw)
    _draw_shadow_text(surface, small_font, research_text, ix, y, PARCH_INFO, shadow=PARCH_SHADOW, shadow_offset=1)
    y += small_font.get_height() + 3
    army_text = _fit_text(f"Army {army_summary}", small_font, iw)
    _draw_shadow_text(surface, small_font, army_text, ix, y, (90, 138, 68), shadow=PARCH_SHADOW, shadow_offset=1)
    y += small_font.get_height() + 6

    # Row 4: base HP bar
    hp_text = f"Base {base_hp}/{max_base_hp}"
    hp_color = _base_hp_color(base_hp, max_base_hp)
    _draw_shadow_text(surface, small_font, hp_text, ix, y + 1, hp_color, shadow=PARCH_SHADOW, shadow_offset=1)
    bar_x = ix + small_font.size(hp_text)[0] + 6
    bar_w = rect.right - HUD_INNER_PAD - bar_x
    if bar_w > 16:
        bar_rect = pygame.Rect(bar_x, y + 3, bar_w, 15)
        color_family = "blue" if faction == "Blue" else "red"
        _draw_ui_meter(surface, board_assets, bar_rect, base_hp, max_base_hp, color_family)


def _faction_income(env: AgeGridEnv, faction: str) -> int:
    total = 0
    for structure in env.get_buildings_for_faction(faction):
        spec = production.building_stats(env, faction, structure.building_type)
        if spec is not None:
            total += spec.resource_income
    return total


def _faction_army_summary(env: AgeGridEnv, faction: str) -> str:
    composition = unit_composition(env, faction)
    total = composition["soldier"] + composition["archer"] + composition["horseman"]
    return f"{total}  S{composition['soldier']} A{composition['archer']} H{composition['horseman']}"


def _draw_center_hud_idle(
    surface: pygame.Surface,
    small_font: pygame.font.Font,
    env: AgeGridEnv,
    ix: int,
    y: int,
    width: int,
    human_turn_active: bool,
    current_faction: str,
) -> None:
    """Idle (nothing selected) middle section for the center HUD."""
    if human_turn_active:
        budget = f"Atk left: {env.actions_left}   Att left: {env.attempts_left}"
        _draw_shadow_text(surface, small_font, budget, ix, y, PARCH_BODY, shadow=PARCH_SHADOW, shadow_offset=1)
        y += small_font.get_height() + 3
    income = _faction_income(env, current_faction)
    army = _faction_army_summary(env, current_faction)
    summary = _fit_text(f"Income {income:+}/turn   Army: {army}", small_font, width)
    _draw_shadow_text(surface, small_font, summary, ix, y, _income_color(income), shadow=PARCH_SHADOW, shadow_offset=1)
    y += small_font.get_height() + 3
    year_text = _fit_text(f"Year {env.formatted_year()}   {env.current_era()}", small_font, width)
    _draw_shadow_text(surface, small_font, year_text, ix, y, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=1)


def _draw_center_hud(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    env: AgeGridEnv,
    winner: str | None,
    human_turn_active: bool,
    btn_label: str,
    current_faction: str,
    selected_unit=None,
    selected_tile: tuple[int, int] | None = None,
    ai_vs_ai: bool = False,
) -> tuple[pygame.Rect, pygame.Rect]:
    """Context-sensitive center HUD: turn header, selection details, recent event.

    Returns (next_turn_btn_rect, next_faction_btn_rect). next_faction_btn_rect is a
    zero-size rect when not in AI-vs-AI mode.
    """
    panel_key = "panel_beigeLight"
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(panel_key), rect):
        _draw_panel(surface, rect, fill=(20, 27, 36), border=PANEL_SOFT, radius=12)

    ix = rect.x + HUD_INNER_PAD
    iw = rect.width - HUD_INNER_PAD * 2
    y = rect.y + HUD_INNER_PAD

    # ── Row 1: turn label + faction + Next Turn button (top-right) ─────────
    btn_w, btn_h = 110, 30
    btn_rect = pygame.Rect(rect.right - HUD_INNER_PAD - btn_w, rect.y + HUD_INNER_PAD + 1, btn_w, btn_h)
    btn_fill = (52, 82, 124) if winner is None else (58, 62, 68)
    btn_border = (120, 172, 234) if winner is None else (90, 96, 108)
    btn_sprite_key = "button_blue" if winner is None else "button_grey"
    if not _draw_scaled_sprite(surface, board_assets.ui_sprite(btn_sprite_key), btn_rect):
        _draw_panel(surface, btn_rect, fill=btn_fill, border=btn_border, radius=10)
    btn_display = btn_label if winner is None else "Game Over"
    _draw_shadow_text(
        surface, small_font, btn_display,
        btn_rect.centerx - small_font.size(btn_display)[0] // 2,
        btn_rect.y + (btn_rect.height - small_font.get_height()) // 2,
        PARCH_TITLE if winner is None else PARCH_MUTED,
        shadow=PARCH_SHADOW, shadow_offset=1,
    )

    if winner is not None:
        _draw_shadow_text(surface, body_font, f"{winner} wins!", ix, y + 4, GOLD_ACCENT, shadow=PARCH_SHADOW, shadow_offset=1)
    else:
        faction_color = RED_PRIMARY if current_faction == "Red" else BLUE_PRIMARY
        turn_part = f"TURN {env.turn}"
        _draw_shadow_text(surface, body_font, turn_part, ix, y + 2, GOLD_ACCENT, shadow=PARCH_SHADOW, shadow_offset=1)
        sep_x = ix + body_font.size(turn_part)[0] + 8
        _draw_shadow_text(surface, body_font, "\u2014", sep_x, y + 2, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)
        faction_x = sep_x + body_font.size("\u2014 ")[0] + 2
        faction_text = _fit_text(f"{current_faction} TURN", body_font, btn_rect.left - 8 - faction_x)
        _draw_shadow_text(surface, body_font, faction_text, faction_x, y + 2, faction_color, shadow=PARCH_SHADOW, shadow_offset=1)

    y += body_font.get_height() + 6
    pygame.draw.line(surface, PARCH_LINE, (ix, y), (rect.right - HUD_INNER_PAD, y), 1)
    y += 7

    # ── Next Faction button (AI-vs-AI only, right column below divider) ─────
    nf_btn_rect = pygame.Rect(0, 0, 0, 0)
    if ai_vs_ai and winner is None:
        nf_btn_w, nf_btn_h = 106, 26
        nf_btn_rect = pygame.Rect(rect.right - HUD_INNER_PAD - nf_btn_w, y + 2, nf_btn_w, nf_btn_h)
        nf_sprite = board_assets.ui_sprite("button")
        if not _draw_scaled_sprite(surface, nf_sprite, nf_btn_rect):
            _draw_panel(surface, nf_btn_rect, fill=(68, 86, 58), border=(112, 148, 86), radius=10)
        nf_label = "Next Faction"
        _draw_shadow_text(
            surface, small_font, nf_label,
            nf_btn_rect.centerx - small_font.size(nf_label)[0] // 2,
            nf_btn_rect.y + (nf_btn_rect.height - small_font.get_height()) // 2,
            PARCH_TITLE,
            shadow=PARCH_SHADOW, shadow_offset=1,
        )
        # hint: Tab keybind
        hint_text = "Tab"
        _draw_shadow_text(surface, small_font, hint_text, nf_btn_rect.right + 4, nf_btn_rect.y + (nf_btn_rect.height - small_font.get_height()) // 2, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)

    # ── Middle: context-sensitive content ───────────────────────────────────
    mid_right = (nf_btn_rect.left - 8) if nf_btn_rect.width > 0 else (rect.right - HUD_INNER_PAD)
    mid_w = mid_right - ix

    if selected_unit is not None:
        spec = production.unit_stats(env, selected_unit.faction, selected_unit.unit_type)
        max_hp = spec.hp if spec is not None else selected_unit.hp
        acc = RED_PRIMARY if selected_unit.faction == "Red" else BLUE_PRIMARY
        unit_label = UNIT_LABELS.get(selected_unit.unit_type, selected_unit.unit_type.replace("_", " ").title())
        _draw_shadow_text(surface, body_font, _fit_text(f"{unit_label}  \u2014  {selected_unit.faction} #{selected_unit.id}", body_font, mid_w), ix, y, acc, shadow=PARCH_SHADOW, shadow_offset=1)
        y += body_font.get_height() + 3
        hp_text = f"HP {selected_unit.hp}/{max_hp}"
        _draw_shadow_text(surface, small_font, hp_text, ix, y + 1, _base_hp_color(selected_unit.hp, max_hp), shadow=PARCH_SHADOW, shadow_offset=1)
        bar_x = ix + small_font.size(hp_text)[0] + 7
        bar_w = min(110, mid_right - bar_x)
        if bar_w > 20:
            _draw_ui_meter(surface, board_assets, pygame.Rect(bar_x, y + 2, bar_w, 14), selected_unit.hp, max_hp, "blue" if selected_unit.faction == "Blue" else "red")
        y += small_font.get_height() + 4
        atk = spec.attack_damage if spec else selected_unit.attack_damage
        rng = spec.attack_range if spec else selected_unit.attack_range
        mv = spec.move_steps if spec else selected_unit.move_steps
        _draw_shadow_text(surface, small_font, f"ATK {atk}   RNG {rng}   MOVE {mv}", ix, y, PARCH_BODY, shadow=PARCH_SHADOW, shadow_offset=1)

    elif selected_tile is not None:
        sel_base_entry = next(((f, b) for f, b in env.bases.items() if b.position == selected_tile), None)
        sel_building = next((b for b in env.buildings if b.position == selected_tile), None)
        if sel_base_entry is not None:
            faction_name, sel_base = sel_base_entry
            acc = RED_PRIMARY if faction_name == "Red" else BLUE_PRIMARY
            max_hp = env.base_max_hp(faction_name)
            _draw_shadow_text(surface, body_font, f"{faction_name} Base", ix, y, acc, shadow=PARCH_SHADOW, shadow_offset=1)
            y += body_font.get_height() + 3
            hp_text = f"HP {sel_base.hp}/{max_hp}"
            _draw_shadow_text(surface, small_font, hp_text, ix, y + 1, _base_hp_color(sel_base.hp, max_hp), shadow=PARCH_SHADOW, shadow_offset=1)
            bar_x = ix + small_font.size(hp_text)[0] + 7
            bar_w = min(110, mid_right - bar_x)
            if bar_w > 20:
                _draw_ui_meter(surface, board_assets, pygame.Rect(bar_x, y + 2, bar_w, 14), sel_base.hp, max_hp, "blue" if faction_name == "Blue" else "red")
            y += small_font.get_height() + 4
            _draw_shadow_text(surface, small_font, "Primary stronghold. Protect at all costs.", ix, y, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=1)
        elif sel_building is not None:
            acc = RED_PRIMARY if sel_building.faction == "Red" else BLUE_PRIMARY
            bspec = production.building_stats(env, sel_building.faction, sel_building.building_type)
            max_hp = bspec.hp if bspec is not None else sel_building.hp
            blabel = _building_label(sel_building.building_type)
            _draw_shadow_text(surface, body_font, _fit_text(f"{blabel}  \u2014  {sel_building.faction}", body_font, mid_w), ix, y, acc, shadow=PARCH_SHADOW, shadow_offset=1)
            y += body_font.get_height() + 3
            hp_text = f"HP {sel_building.hp}/{max_hp}"
            _draw_shadow_text(surface, small_font, hp_text, ix, y + 1, _base_hp_color(sel_building.hp, max_hp), shadow=PARCH_SHADOW, shadow_offset=1)
            bar_x = ix + small_font.size(hp_text)[0] + 7
            bar_w = min(110, mid_right - bar_x)
            if bar_w > 20:
                _draw_ui_meter(surface, board_assets, pygame.Rect(bar_x, y + 2, bar_w, 14), sel_building.hp, max_hp, "blue" if sel_building.faction == "Blue" else "red")
            if bspec is not None and bspec.resource_income > 0:
                y += small_font.get_height() + 4
                _draw_shadow_text(surface, small_font, f"Income +{bspec.resource_income}/turn", ix, y, PARCH_GOOD, shadow=PARCH_SHADOW, shadow_offset=1)
        else:
            _draw_center_hud_idle(surface, small_font, env, ix, y, mid_w, human_turn_active, current_faction)
    else:
        _draw_center_hud_idle(surface, small_font, env, ix, y, mid_w, human_turn_active, current_faction)

    # ── Bottom: most recent event ───────────────────────────────────────────
    if env.recent_events:
        event_y = rect.bottom - small_font.get_height() - 7
        pygame.draw.line(surface, PARCH_LINE, (ix, event_y - 5), (rect.right - HUD_INNER_PAD, event_y - 5), 1)
        recent = env.recent_events[-1]
        _draw_shadow_text(surface, small_font, _fit_text(recent, small_font, iw), ix, event_y, _event_color(recent), shadow=PARCH_SHADOW, shadow_offset=1)

    return btn_rect, nf_btn_rect


def _event_color(line: str) -> tuple[int, int, int]:
    if line.startswith("Red "):
        return (198, 92, 78)
    if line.startswith("Blue "):
        return (82, 118, 188)
    return PARCH_BODY


def _faction_research_label(env: AgeGridEnv, faction: str) -> str:
    state = env.faction_state(faction)
    if state.tech_in_progress:
        turns_left = tech.research_turns_remaining(env, faction)
        return f"{_tech_label(state.tech_in_progress)} ({turns_left}t)"
    available = _available_research_ids(env, faction)
    if available:
        return f"Ready: {_tech_label(available[0])}"
    return "None"


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


def _contested_resource_count(env: AgeGridEnv, faction: str) -> int:
    return sum(1 for resource in env.visible_resources(faction) if env.resource_is_contested(resource, faction))


def _tactical_panel_lines(env: AgeGridEnv, faction: str) -> list[str]:
    composition = unit_composition(env, faction)
    mode = "Defense" if defense_mode_active(env, faction) else "Push" if push_mode_active(env, faction) else "Field"
    enemy = next(name for name in env.factions if name != faction)
    relation = env.relation_state(faction, enemy).state.title()
    contested = _contested_resource_count(env, faction)
    return [
        f"Diplomacy {relation}",
        f"Threat {threat_level(env, faction)}",
        f"Plan {army_plan(env, faction)} | {mode}",
        f"Resources {contested} contested",
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
            line_color = PARCH_BODY
            if line.startswith("Diplomacy "):
                line_color = _relation_color(line.removeprefix("Diplomacy ").strip())
            elif line.startswith("Threat "):
                threat_text = line.removeprefix("Threat ").strip().lower()
                line_color = PARCH_DANGER if threat_text == "emergency" else PARCH_WARN if threat_text == "guarded" else PARCH_BODY
            elif line.startswith("Plan "):
                line_color = PARCH_INFO
            elif line.startswith("Resources ") and not line.startswith("Resources 0 "):
                line_color = PARCH_DANGER
            elif line.startswith("Army "):
                line_color = PARCH_MUTED
            _draw_text_block(surface, body_font, wrapped, block_rect.x + 10, line_y, line_color, 18)
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


_RESEARCH_DRAW_HELPERS = ResearchDrawHelpers(
    draw_parchment_panel_frame=_draw_parchment_panel_frame,
    draw_parchment_header=_draw_parchment_header,
    draw_parchment_close_button=_draw_parchment_close_button,
    draw_scaled_sprite=_draw_scaled_sprite,
    draw_shadow_text=_draw_shadow_text,
    draw_text_block=_draw_text_block,
    wrap_lines=_wrap_lines,
    fit_text=_fit_text,
)


_PANEL_DRAW_HELPERS = PanelDrawHelpers(
    draw_parchment_panel_frame=_draw_parchment_panel_frame,
    draw_parchment_header=_draw_parchment_header,
    draw_parchment_close_button=_draw_parchment_close_button,
    draw_unit_sprite=_draw_unit_sprite,
    draw_unit_icon=_draw_unit_icon,
    draw_shadow_text=_draw_shadow_text,
    draw_text_block=_draw_text_block,
    draw_ui_meter=_draw_ui_meter,
    draw_parchment_chip=_draw_parchment_chip,
    draw_parchment_button=_draw_parchment_button,
    wrap_lines=_wrap_lines,
    building_label=_building_label,
)

_PANEL_TEXT = PanelText(
    unit_labels=UNIT_LABELS,
    unit_help=UNIT_HELP,
    building_help=BUILDING_HELP,
    resource_labels=RESOURCE_LABELS,
    resource_help=RESOURCE_HELP,
)

_PANEL_COLORS = PanelColors(
    red_primary=RED_PRIMARY,
    blue_primary=BLUE_PRIMARY,
    red_accent=RED_ACCENT,
    blue_accent=BLUE_ACCENT,
    parch_title=PARCH_TITLE,
    parch_body=PARCH_BODY,
    parch_muted=PARCH_MUTED,
    parch_shadow=PARCH_SHADOW,
    parch_line=PARCH_LINE,
    text_primary=TEXT_PRIMARY,
)


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
    resource = next((r for r in env.resources if r.position == pos and r.abundance > 0), None)
    if resource is None:
        return "grass" if (pos[0] + pos[1]) % 4 else "dirt"
    if resource.resource_type == "stone":
        return "stone"
    if resource.resource_type == "horses":
        return "sand"
    return "grass"


def _resource_contest_labels(env: AgeGridEnv, resource) -> list[str]:
    return [faction for faction in env.factions if env.resource_is_contested(resource, faction)]


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
    resource = next((r for r in env.resources if r.position == pos and r.abundance > 0), None)
    if resource is not None:
        label = "Horse Herd" if resource.resource_type == "horses" else "Stone Deposit" if resource.resource_type == "stone" else "Ore Vein"
        detail = "Unlocks horseback riding and cavalry" if resource.resource_type == "horses" else "Unlocks quarry, walls, and stronger structures" if resource.resource_type == "stone" else "Gatherable resource"
        contest_labels = _resource_contest_labels(env, resource)
        access = (
            f"Contested for {', '.join(contest_labels)}"
            if contest_labels
            else "Infinite source"
        )
        return [label, detail, access]
    return []


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


def _hover_panel_rect(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    tile_rect: pygame.Rect,
    lines: list[str],
) -> pygame.Rect:
    line_height = 18
    width = 224
    wrapped_lines = 0
    for line in lines[1:]:
        wrapped_lines += max(1, len(_wrap_lines(line, body_font, width - 34)))
    inner_height = 8 + title_font.get_height() + 8 + 10 + max(1, wrapped_lines) * line_height + 10
    height = max(88, inner_height + 36)
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
        elif kind == "declare_war":
            target = action[1]
            effects.append(
                VisualEffect(
                    "banner",
                    74,
                    74,
                    RED_PRIMARY,
                    label=(
                        f"{faction} declared war on {target}  "
                        f"(-${env.config.war_declaration_cost}, support {env.faction_state(faction).war_support})"
                    ),
                )
            )
        elif kind == "offer_peace":
            target = action[1]
            indemnity = int(action[2])
            effects.append(
                VisualEffect(
                    "banner",
                    64,
                    64,
                    PARCH_WARN,
                    label=f"{faction} offered peace to {target}  (${indemnity})",
                )
            )
        elif kind == "accept_peace":
            target = action[1]
            relation = env.relation_state(faction, target)
            detail = next(
                (
                    event
                    for event in reversed(env.current_events)
                    if event.startswith(f"{faction} accepted peace with {target}")
                ),
                None,
            )
            label = (
                detail
                if detail is not None
                else f"{faction} accepted peace with {target}  (truce {relation.truce_until_turn - env.turn} turns)"
            )
            effects.append(VisualEffect("banner", 90, 90, PARCH_GOOD, label=label))
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
            text = font.render(effect.label, True, (244, 246, 248))
            banner_width = max(340, min(screen.get_width() - 40, text.get_width() + 44))
            banner = pygame.Surface((banner_width, 42), pygame.SRCALPHA)
            pygame.draw.rect(banner, (15, 20, 27, min(230, alpha + 20)), banner.get_rect(), border_radius=12)
            pygame.draw.rect(banner, (*color, alpha), banner.get_rect(), width=2, border_radius=12)
            banner_x = screen.get_width() // 2 - banner_width // 2
            screen.blit(banner, (banner_x, 14))
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


def run_viewer() -> None:
    pygame.init()
    pygame.display.set_caption("AgeGrid Viewer")

    _set_hex_zoom(DEFAULT_VIEWER_ZOOM)
    env = AgeGridEnv()
    pad = 20
    side_panel = 320
    default_board_width, default_board_height = _board_pixel_size(env)
    board_viewport_width = default_board_width + BOARD_VIEWPORT_EXTRA_W
    board_viewport_height = default_board_height + BOARD_VIEWPORT_EXTRA_H

    # HUD layout — three horizontal zones across the board width
    faction_bar_w = max(220, min(HUD_FACTION_W, (board_viewport_width - 28) // 3))
    center_zone_x = pad + faction_bar_w + 8
    center_zone_w = board_viewport_width - 2 * faction_bar_w - 16
    top_bar = HUD_H + pad + 10   # total vertical space reserved for the top HUD strip

    width_px = pad * 3 + board_viewport_width + side_panel + 36
    height_px = pad * 2 + top_bar + board_viewport_height + BASE_HEX_SIZE + 44

    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()
    board_assets = BoardAssets.load(Path.cwd())

    font = pygame.font.SysFont("segoeui", 24, bold=True)
    big = pygame.font.SysFont("segoeui", 34, bold=True)
    small = pygame.font.SysFont("segoeui", 19)
    tiny = pygame.font.SysFont("segoeui", 17)

    board_x = pad
    board_content_top = pad + top_bar + 4
    board_content_bottom = height_px - pad - 18
    board_rect = pygame.Rect(
        board_x,
        board_content_top,
        board_viewport_width,
        max(board_viewport_height, board_content_bottom - board_content_top),
    )
    # btn_rect and next_faction_btn_rect are updated each frame from _draw_center_hud
    btn_rect = pygame.Rect(0, 0, 0, 0)
    next_faction_btn_rect = pygame.Rect(0, 0, 0, 0)

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
    tech_tree_scroll = 0
    tech_tree_pan_x = 0
    tech_tree_dragging = False
    tech_tree_last_mouse = (0, 0)
    tech_tree_drag_start_mouse = (0, 0)
    tech_tree_drag_moved = False
    tech_tree_close_rect = pygame.Rect(0, 0, 0, 0)
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
    camera_state = CameraState()
    _reset_camera_state(camera_state, DEFAULT_VIEWER_ZOOM)
    board_backdrop = board_rect.inflate(22, 28)
    board_origin = _board_origin_in_viewport(board_rect, default_board_width, default_board_height, HEX_SIZE)

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
                        _reset_camera_state(camera_state, _set_hex_zoom(DEFAULT_VIEWER_ZOOM))
                        if _is_human_agent_key(_current_agent_key(env, red_index, blue_index)):
                            env.start_faction_turn()
                            active_human_faction = env.factions[env.current_player]
                        in_setup = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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
                        _reset_camera_state(camera_state, _set_hex_zoom(DEFAULT_VIEWER_ZOOM))
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
                    tech_tree_dragging = False
                    tech_tree_drag_moved = False
                    human_moved_units = set()
                    human_turn_actions = []
                    human_turn_log = []
                    active_human_faction = None
                    selected_tile = None
                    selected_unit_id = None
                    pending_build_type = None
                    _reset_camera_state(camera_state, _set_hex_zoom(DEFAULT_VIEWER_ZOOM))
                elif event.key == pygame.K_t:
                    show_tech_tree = not show_tech_tree
                    if show_tech_tree:
                        tech_tree_scroll = 0
                        tech_tree_pan_x = 0
                        tech_tree_dragging = False
                        tech_tree_drag_moved = False
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_p:
                    snapshot_text = build_debug_snapshot(
                        env,
                        AGENT_SPECS[red_index].label,
                        AGENT_SPECS[blue_index].label,
                        red_info,
                        blue_info,
                        turn_history,
                    )
                    output_path = write_debug_snapshot(snapshot_text)
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
                elif event.key == pygame.K_TAB and not has_human_players and env.winner() is None:
                    red_info, blue_info, snap, new_effects = _step_ai_faction(env, red_agent, blue_agent, red_info, blue_info)
                    if snap is not None:
                        turn_history.append(snap)
                    effects.extend(new_effects)
                    event_scroll = 0
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if env.winner() is None:
                        if not has_human_players:
                            red_info, blue_info, red_actions, blue_actions = step_full_turn(env, red_agent, blue_agent)
                            turn_history.append(turn_snapshot(env, red_info, blue_info))
                            effects.extend(_effects_from_actions(env, red_actions, "Red"))
                            effects.extend(_effects_from_actions(env, blue_actions, "Blue"))
                        else:
                            if human_turn_active:
                                completed_info = build_turn_info(
                                    current_faction,
                                    human_turn_actions,
                                    human_turn_log or ["stop"],
                                    list(env.current_events),
                                    env.turn + 1,
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
                                    turn_history.append(turn_snapshot(env, red_info, blue_info))

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

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if show_tech_tree and tech_tree_dragging and not tech_tree_drag_moved:
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
                tech_tree_dragging = False
                tech_tree_drag_moved = False

            elif event.type == pygame.MOUSEMOTION:
                if show_tech_tree and tech_tree_dragging:
                    dx = event.pos[0] - tech_tree_last_mouse[0]
                    dy = event.pos[1] - tech_tree_last_mouse[1]
                    if abs(event.pos[0] - tech_tree_drag_start_mouse[0]) >= 6 or abs(event.pos[1] - tech_tree_drag_start_mouse[1]) >= 6:
                        tech_tree_drag_moved = True

                    overlay_rect = pygame.Rect(pad * 2, pad * 2, width_px - pad * 4, height_px - pad * 4)
                    tree_layout = _tech_tree_layout(font, tiny, overlay_rect)
                    tech_tree_scroll, tech_tree_pan_x = _clamp_tech_tree_view(
                        tree_layout.tree_rect,
                        tech_tree_scroll - dy,
                        tech_tree_pan_x + dx,
                    )
                    tech_tree_last_mouse = event.pos

            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()

                if show_tech_tree:
                    overlay_rect = pygame.Rect(pad * 2, pad * 2, width_px - pad * 4, height_px - pad * 4)
                    tree_layout = _tech_tree_layout(font, tiny, overlay_rect)
                    tech_tree_scroll, tech_tree_pan_x = _clamp_tech_tree_view(
                        tree_layout.tree_rect,
                        tech_tree_scroll - event.y * 36,
                        tech_tree_pan_x,
                    )

                elif event_panel_rect.width > 0 and event_panel_rect.collidepoint(mouse_pos):
                    event_scroll = max(0, event_scroll - event.y)

                elif board_backdrop.collidepoint(mouse_pos):
                    old_zoom = camera_state.zoom
                    old_board_width, old_board_height = _board_pixel_size(env)
                    zoom_delta = ZOOM_STEP if event.y > 0 else -ZOOM_STEP
                    new_zoom = _set_hex_zoom(camera_state.zoom + zoom_delta)
                    new_board_width, new_board_height = _board_pixel_size(env)
                    _apply_camera_zoom(
                        camera_state,
                        mouse_pos,
                        board_rect,
                        old_zoom,
                        (old_board_width, old_board_height),
                        new_zoom,
                        (new_board_width, new_board_height),
                        HEX_SIZE,
                    )

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if show_tech_tree:
                    if tech_tree_close_rect.collidepoint(event.pos):
                        show_tech_tree = False
                        tech_tree_dragging = False
                        tech_tree_drag_moved = False
                        continue
                    overlay_rect = pygame.Rect(pad * 2, pad * 2, width_px - pad * 4, height_px - pad * 4)
                    tree_layout = _tech_tree_layout(font, tiny, overlay_rect)
                    if tree_layout.tree_rect.collidepoint(event.pos):
                        tech_tree_dragging = True
                        tech_tree_last_mouse = event.pos
                        tech_tree_drag_start_mouse = event.pos
                        tech_tree_drag_moved = False
                    elif not overlay_rect.collidepoint(event.pos):
                        show_tech_tree = False
                        tech_tree_dragging = False
                        tech_tree_drag_moved = False
                    continue

                if btn_rect.collidepoint(event.pos) and env.winner() is None:
                    if not has_human_players:
                        red_info, blue_info, red_actions, blue_actions = step_full_turn(env, red_agent, blue_agent)
                        turn_history.append(turn_snapshot(env, red_info, blue_info))
                        effects.extend(_effects_from_actions(env, red_actions, "Red"))
                        effects.extend(_effects_from_actions(env, blue_actions, "Blue"))
                    else:
                        if human_turn_active:
                            completed_info = build_turn_info(
                                current_faction,
                                human_turn_actions,
                                human_turn_log or ["stop"],
                                list(env.current_events),
                                env.turn + 1,
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
                                turn_history.append(turn_snapshot(env, red_info, blue_info))

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
                elif next_faction_btn_rect.width > 0 and next_faction_btn_rect.collidepoint(event.pos) and env.winner() is None:
                    red_info, blue_info, snap, new_effects = _step_ai_faction(env, red_agent, blue_agent, red_info, blue_info)
                    if snap is not None:
                        turn_history.append(snap)
                    effects.extend(new_effects)
                    event_scroll = 0
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
                    tech_tree_dragging = False
                    tech_tree_drag_moved = False
                    if show_tech_tree:
                        tech_tree_scroll = 0
                        tech_tree_pan_x = 0
                elif camera_btn_rect.collidepoint(event.pos):
                    _reset_camera_state(camera_state, _set_hex_zoom(DEFAULT_VIEWER_ZOOM))
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
                    _begin_camera_pan(camera_state, event.pos)
            elif event.type == pygame.MOUSEMOTION and camera_state.panning:
                board_width, board_height = _board_pixel_size(env)
                _update_camera_pan(camera_state, event.pos, board_rect, board_width, board_height)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2 and camera_state.panning:
                if _end_camera_pan(camera_state):
                    _reset_camera_state(camera_state, _set_hex_zoom(DEFAULT_VIEWER_ZOOM))

        effects = _update_effects(effects)
        board_width, board_height = _board_pixel_size(env)
        camera_state.pan[0], camera_state.pan[1] = _clamp_camera_pan(board_rect, board_width, board_height, tuple(camera_state.pan))
        board_origin = _board_origin_in_viewport(board_rect, board_width, board_height, HEX_SIZE, tuple(camera_state.pan))
        board_backdrop = board_rect.inflate(22, 28)
        screen.fill((9, 13, 18))
        hud_rect = pygame.Rect(0, 0, width_px, top_bar)
        pygame.draw.rect(screen, HUD_BG, hud_rect)
        pygame.draw.line(screen, PANEL_SOFT, (0, top_bar - 1), (width_px, top_bar - 1), 2)
        sidebar_x = board_x + board_viewport_width + pad
        sidebar_rect = pygame.Rect(sidebar_x, pad, side_panel, height_px - pad * 2)
        _draw_panel(screen, sidebar_rect, fill=(16, 22, 29), border=PANEL_SOFT, radius=16)

        winner = env.winner()
        current_agent_key = _current_agent_key(env, red_index, blue_index)
        human_turn_active = _is_human_agent_key(current_agent_key)
        mouse_pos = pygame.mouse.get_pos()
        current_research_count = len(_available_research_ids(env, current_faction))
        turn_button_label = "End Turn" if human_turn_active and winner is None else "Next Turn"

        # ── Three-zone top HUD ────────────────────────────────────────────────
        red_bar_rect  = pygame.Rect(board_x, pad, faction_bar_w, HUD_H)
        blue_bar_rect = pygame.Rect(board_x + board_viewport_width - faction_bar_w, pad, faction_bar_w, HUD_H)
        center_rect   = pygame.Rect(center_zone_x, pad, center_zone_w, HUD_H)

        red_relation = env.relation_state("Red", "Blue").state.title()
        blue_relation = env.relation_state("Blue", "Red").state.title()

        _draw_faction_bar(screen, small, tiny, board_assets, red_bar_rect,
            "Red", AGENT_SPECS[red_index].label,
            env.bank["Red"], _faction_income(env, "Red"), _faction_army_summary(env, "Red"),
            env.bases["Red"].hp, env.base_max_hp("Red"),
            env.current_era(), red_relation, _faction_research_label(env, "Red"), RED_PRIMARY)

        _draw_faction_bar(screen, small, tiny, board_assets, blue_bar_rect,
            "Blue", AGENT_SPECS[blue_index].label,
            env.bank["Blue"], _faction_income(env, "Blue"), _faction_army_summary(env, "Blue"),
            env.bases["Blue"].hp, env.base_max_hp("Blue"),
            env.current_era(), blue_relation, _faction_research_label(env, "Blue"), BLUE_PRIMARY)

        # Resolve selection early so the center HUD can show context-sensitive info
        selected_unit = next((u for u in env.units if u.id == selected_unit_id), None)
        btn_rect, next_faction_btn_rect = _draw_center_hud(screen, big, small, tiny, board_assets, center_rect,
            env, winner, human_turn_active, turn_button_label, current_faction,
            selected_unit=selected_unit, selected_tile=selected_tile,
            ai_vs_ai=not has_human_players)

        # ── Right sidebar ─────────────────────────────────────────────────────
        # Cap panel heights so research + tactical + event all fit inside the sidebar
        _min_event_h = 140
        _btn_area = 8 + 30 + 10  # gap-before + btn-height + gap-after
        _panel_budget = max(200, sidebar_rect.bottom - 8 - (pad + 12) - _min_event_h - _btn_area)
        research_panel_h = min(
            _research_panel_height(_RESEARCH_DRAW_HELPERS, font, tiny, env, side_panel - 24),
            _panel_budget * 55 // 100,
        )
        research_panel = pygame.Rect(sidebar_x + 12, pad + 12, side_panel - 24, research_panel_h)
        _draw_research_panel(_RESEARCH_DRAW_HELPERS, screen, font, tiny, board_assets, env, research_panel)
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
            _draw_hover_tile_panel(_PANEL_DRAW_HELPERS, _PANEL_COLORS, screen, small, tiny, board_assets, hover_rect, hover_lines)

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
        display_zoom = 100 if DEFAULT_VIEWER_ZOOM <= 0 else int(round((camera_state.zoom / DEFAULT_VIEWER_ZOOM) * 100))
        zoom_label = f"{display_zoom}%"
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

            contest_labels = _resource_contest_labels(env, r)
            ring_center = (center[0], center[1] + max(4, resource_offset_y // 2))
            if contest_labels:
                pulse = (pygame.time.get_ticks() // 220) % 2
                ring_radius = max(15, int(HEX_SIZE * (0.62 + 0.08 * pulse)))
                pygame.draw.circle(screen, RESOURCE_CONTESTED, ring_center, ring_radius, width=4)
                pygame.draw.circle(screen, (255, 226, 170), ring_center, max(8, ring_radius - 7), width=2)
                badge_rect = pygame.Rect(ring_center[0] + ring_radius - 5, ring_center[1] - ring_radius - 4, 20, 20)
                pygame.draw.rect(screen, (68, 28, 26), badge_rect, border_radius=9)
                pygame.draw.rect(screen, RESOURCE_CONTESTED, badge_rect, width=2, border_radius=9)
                _draw_shadow_text(screen, tiny, "!", badge_rect.x + 7, badge_rect.y + 1, TEXT_PRIMARY, shadow=(0, 0, 0), shadow_offset=1)
            else:
                pygame.draw.circle(screen, (*RESOURCE_GLOW, 115), ring_center, max(12, int(HEX_SIZE * 0.48)), width=2)

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

            # Keep buildings grounded lower in the hex than units/resources so they
            # read as placed structures rather than floating markers.
            building_scale = max(22, int(HEX_SIZE * 1.22))
            building_rect_size = max(18, int(HEX_SIZE * 1.02))
            building_shadow_w = max(18, int(HEX_SIZE * 1.18))
            building_shadow_h = max(7, int(HEX_SIZE * 0.42))
            building_offset_y = int(HEX_SIZE * 0.56)

            rect = pygame.Rect(
                center[0] - building_rect_size // 2,
                center[1] - int(HEX_SIZE * 0.56),
                building_rect_size,
                building_rect_size,
            )

            _draw_soft_shadow(
                screen,
                (center[0], center[1] + building_offset_y + 1),
                building_shadow_w,
                building_shadow_h,
                alpha=58,
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
            if u.unit_type == "worker":
                resource = env.resource_at_for_faction(u.position, u.faction)
                if resource is not None and not env.can_gather_resource(u, resource):
                    badge = pygame.Rect(cx + unit_size // 3, cy - unit_size, 18, 18)
                    pygame.draw.rect(screen, (68, 28, 26), badge, border_radius=8)
                    pygame.draw.rect(screen, RESOURCE_CONTESTED, badge, width=2, border_radius=8)
                    _draw_shadow_text(screen, tiny, "!", badge.x + 6, badge.y, TEXT_PRIMARY, shadow=(0, 0, 0), shadow_offset=1)

        _draw_effects(screen, effects, board_origin[0], board_origin[1], HEX_WIDTH, small)
        screen.set_clip(previous_clip)

        if hover_pos is not None:
            hover_lines = _hover_tile_lines(env, hover_pos)
            if hover_lines:
                tile_rect = _hex_bounds(hover_pos[0], hover_pos[1], board_origin)
                hover_rect = _hover_panel_rect(screen, small, tiny, tile_rect, hover_lines)
                _draw_hover_tile_panel(_PANEL_DRAW_HELPERS, _PANEL_COLORS, screen, small, tiny, board_assets, hover_rect, hover_lines)

        if selected_unit is None:
            selected_unit_id = None
            selected_unit_close_rect = pygame.Rect(0, 0, 0, 0)
        else:
            inspect_rect = pygame.Rect(board_backdrop.x + 18, board_backdrop.y + 18, 300, 268)
            selected_unit_close_rect = _draw_selected_unit_panel(
                _PANEL_DRAW_HELPERS,
                _PANEL_TEXT,
                _PANEL_COLORS,
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
            selected_resource = next((r for r in env.resources if r.position == selected_tile and r.abundance > 0), None)
            selected_base_entry = next(((faction, base) for faction, base in env.bases.items() if base.position == selected_tile), None)
            if selected_building is not None:
                spec = production.building_stats(env, selected_building.faction, selected_building.building_type)
                max_hp = spec.hp if spec is not None else selected_building.hp
                selected_unit_close_rect = _draw_selected_object_panel(
                    _PANEL_DRAW_HELPERS,
                    _PANEL_COLORS,
                    screen,
                    small,
                    tiny,
                    tiny,
                    board_assets,
                    env,
                    inspect_rect,
                    title=_building_label(selected_building.building_type),
                    subtitle=f"{selected_building.faction} Building",
                    lines=_selected_building_lines(_PANEL_DRAW_HELPERS, _PANEL_TEXT, env, selected_building)[1:],
                    hp_value=selected_building.hp,
                    hp_max=max_hp,
                    icon_kind=selected_building.building_type,
                    accent=RED_PRIMARY if selected_building.faction == "Red" else BLUE_PRIMARY,
                )
            elif selected_base_entry is not None:
                faction, selected_base = selected_base_entry
                selected_unit_close_rect = _draw_selected_object_panel(
                    _PANEL_DRAW_HELPERS,
                    _PANEL_COLORS,
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
                    _PANEL_DRAW_HELPERS,
                    _PANEL_COLORS,
                    screen,
                    small,
                    tiny,
                    tiny,
                    board_assets,
                    env,
                    inspect_rect,
                    title=RESOURCE_LABELS.get(selected_resource.resource_type, selected_resource.resource_type.title()),
                    subtitle="Map resource",
                    lines=_selected_resource_lines(_PANEL_TEXT, selected_resource, env, current_faction)[1:],
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
                    _human_action_panel_height(_PANEL_DRAW_HELPERS, tiny, panel_options, panel_hint),
                )
                human_action_button_rects = _draw_human_action_panel(
                    _PANEL_DRAW_HELPERS,
                    _PANEL_COLORS,
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
                ["Debug: D  Snapshot: P  Tech tree: T  Gather: G  Reset: R  Space: next turn  Tab: next faction  Wheel zoom"],
                debug_rect.x + 10,
                debug_rect.y + 7,
                TEXT_MUTED,
                16,
            )

        tech_tree_node_rects = {}
        tech_tree_close_rect = pygame.Rect(0, 0, 0, 0)
        if show_tech_tree:
            overlay_rect = pygame.Rect(pad * 2, pad * 2, width_px - pad * 4, height_px - pad * 4)
            tech_tree_result = _draw_tech_tree_overlay(
                _RESEARCH_DRAW_HELPERS,
                screen,
                font,
                tiny,
                board_assets,
                env,
                overlay_rect,
                current_faction,
                human_turn_active,
                mouse_pos,
                scroll_y=tech_tree_scroll,
                pan_x=tech_tree_pan_x,
            )
            tech_tree_node_rects = tech_tree_result.node_rects
            tech_tree_close_rect = tech_tree_result.close_rect

        pygame.display.flip()

    pygame.quit()
