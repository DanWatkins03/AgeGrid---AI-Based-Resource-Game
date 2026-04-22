from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame

from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env.systems import tech
from src.agegrid.ui.assets import BoardAssets

PARCH_TITLE = (80, 62, 46)
PARCH_BODY = (92, 74, 52)
PARCH_MUTED = (118, 97, 74)
PARCH_SHADOW = (240, 224, 196)
PARCH_LINE = (188, 168, 136)
TEXT_PRIMARY = (236, 240, 245)
PANEL_INSET = (26, 33, 43)
PANEL_SOFT = (40, 48, 59)

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

UNIT_LABELS = {
    "worker": "Worker",
    "soldier": "Soldier",
    "archer": "Archer",
    "horseman": "Horseman",
    "heavy_cavalry": "Heavy Cavalry",
    "ballista": "Ballista",
}

TECH_TREE_ORDER = list(tech.TECH_TREE_ORDER)


@dataclass(frozen=True)
class ResearchDrawHelpers:
    draw_parchment_panel_frame: Callable[..., pygame.Rect]
    draw_parchment_header: Callable[..., int]
    draw_parchment_close_button: Callable[..., pygame.Rect]
    draw_scaled_sprite: Callable[..., bool]
    draw_shadow_text: Callable[..., None]
    draw_text_block: Callable[..., None]
    wrap_lines: Callable[[str, pygame.font.Font, int], list[str]]
    fit_text: Callable[[str, pygame.font.Font, int], str]


@dataclass(frozen=True)
class TechTreeLayout:
    outer: pygame.Rect
    tree_rect: pygame.Rect
    detail_rect: pygame.Rect


@dataclass(frozen=True)
class TechTreeDrawResult:
    node_rects: dict[str, pygame.Rect]
    close_rect: pygame.Rect


NODE_WIDTH = 146
NODE_HEIGHT = 82
NODE_HALF_W = NODE_WIDTH // 2
NODE_HALF_H = NODE_HEIGHT // 2
DETAIL_WIDTH = 270
DETAIL_GAP = 18
TREE_SIDE_PADDING = 48
TREE_VERTICAL_PADDING = 48


def tech_label(tech_id: str) -> str:
    return TECH_LABELS.get(tech_id, tech_id.replace("_", " ").title())


def tech_status(env: AgeGridEnv, faction: str, tech_id: str) -> str:
    state = env.faction_state(faction)
    if tech_id in state.techs_unlocked:
        return "Done"
    if state.tech_in_progress == tech_id:
        return f"Active ({tech.research_turns_remaining(env, faction)}t)"
    if tech.can_research(env, faction, tech_id):
        return "Ready"
    return "-"


def available_research_ids(env: AgeGridEnv, faction: str) -> list[str]:
    return [tech_id for tech_id in TECH_TREE_ORDER if tech.can_research(env, faction, tech_id)]


def _building_label(building_id: str) -> str:
    return BUILDING_LABELS.get(building_id, building_id.replace("_", " ").title())


def _tech_unlock_summary(tech_id: str) -> str:
    labels: list[str] = []
    for item in tech.unlock_items(tech_id):
        if item in TECH_LABELS:
            labels.append(tech_label(item))
        elif item in BUILDING_LABELS:
            labels.append(_building_label(item))
        elif item in UNIT_LABELS:
            labels.append(UNIT_LABELS[item])
        else:
            labels.append(item.replace("_", " ").title())
    return ", ".join(labels) if labels else "-"


def tech_detail_lines(env: AgeGridEnv, faction: str, tech_id: str) -> list[str]:
    state = env.faction_state(faction)
    definition = tech.TECH_DEFS[tech_id]
    unlocks = _tech_unlock_summary(tech_id)
    lines = [
        f"Cost {definition.cost} | {definition.turns} turn{'s' if definition.turns != 1 else ''}",
        f"Status: {tech_status(env, faction, tech_id)}",
    ]
    if definition.requires:
        missing = [tech_label(req) for req in definition.requires if req not in state.techs_unlocked]
        if missing:
            lines.append(f"Requires: {', '.join(tech_label(req) for req in definition.requires)}")
            lines.append(f"Missing: {', '.join(missing)}")
        else:
            lines.append(f"Prereqs: {', '.join(tech_label(req) for req in definition.requires)}")
    else:
        lines.append("Prereqs: None")
    lines.append(f"Unlocks: {unlocks}")
    if definition.summary:
        lines.append(f"Summary: {definition.summary}")
    return lines


def _research_subsection_height(
    helpers: ResearchDrawHelpers,
    body_font: pygame.font.Font,
    faction: str,
    status: str,
    eta: str | None,
    summary: str,
    available_labels: list[str],
    width: int,
) -> int:
    inner_width = width - 20
    header_lines = helpers.wrap_lines(f"{faction}: {status}", body_font, inner_width - 30)
    total = 10 + max(26, len(header_lines) * 19) + 8
    detail = [summary, f"Next: {', '.join(available_labels[:3]) if available_labels else '-'}"]
    if eta:
        detail.insert(0, eta)
    for line in detail:
        total += len(helpers.wrap_lines(line, body_font, inner_width)) * 19 + 5
    total += 6 + 12 + 10
    return total


def research_panel_height(
    helpers: ResearchDrawHelpers,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    env: AgeGridEnv,
    width: int,
) -> int:
    header_h = 22 + title_font.get_height()
    total = 18 + header_h
    subsection_width = width - 34
    for faction in ("Red", "Blue"):
        state = env.faction_state(faction)
        available = [tech_id for tech_id in tech.TECH_DEFS if tech.can_research(env, faction, tech_id)]
        available_labels = [tech_label(tech_id) for tech_id in available]
        if state.tech_in_progress:
            turns_left = tech.research_turns_remaining(env, faction)
            current_def = tech.TECH_DEFS[state.tech_in_progress]
            status = tech_label(state.tech_in_progress)
            eta = f"{turns_left} turn{'s' if turns_left != 1 else ''} left"
            summary = current_def.summary
        else:
            status = "None"
            eta = None
            summary = tech.TECH_DEFS[available[0]].summary if available else "No research currently available."
        total += _research_subsection_height(helpers, body_font, faction, status, eta, summary, available_labels, subsection_width)
        total += 12
    return total


def draw_research_panel(
    helpers: ResearchDrawHelpers,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    rect: pygame.Rect,
) -> None:
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)
    y = helpers.draw_parchment_header(surface, title_font, body_font, inner, "Research")
    max_width = inner.width - 16
    for faction, color in (("Red", (244, 184, 180)), ("Blue", (176, 214, 255))):
        state = env.faction_state(faction)
        available = [tech_id for tech_id in tech.TECH_DEFS if tech.can_research(env, faction, tech_id)]
        available_labels = [tech_label(tech_id) for tech_id in available]
        if state.tech_in_progress:
            turns_left = tech.research_turns_remaining(env, faction)
            current_definition = tech.TECH_DEFS[state.tech_in_progress]
            total_turns = current_definition.turns
            progress_ratio = min(1.0, state.research_points / total_turns) if total_turns > 0 else 1.0
            status = tech_label(state.tech_in_progress)
            eta = f"{turns_left} turn{'s' if turns_left != 1 else ''} left"
            summary = current_definition.summary
        else:
            progress_ratio = 0.0
            status = "None"
            eta = None
            summary = tech.TECH_DEFS[available[0]].summary if available else "No research currently available."
        icon_key = state.tech_in_progress or (available[0] if available else None)
        icon_style = TECH_ICON_STYLES.get(icon_key or "", {"label": "?", "bg": (70, 76, 84), "fg": (232, 236, 240)})
        subsection_h = _research_subsection_height(helpers, body_font, faction, status, eta, summary, available_labels, max_width)
        subsection_rect = pygame.Rect(inner.x + 8, y, inner.width - 16, subsection_h)
        inset_sprite = board_assets.ui_sprite("panelInset_beigeLight") or board_assets.ui_sprite("panelInset_beige")
        if not helpers.draw_scaled_sprite(surface, inset_sprite, subsection_rect):
            pygame.draw.rect(surface, PANEL_INSET, subsection_rect, border_radius=8)
            pygame.draw.rect(surface, PANEL_SOFT, subsection_rect, width=1, border_radius=8)

        inner_x = subsection_rect.x + 10
        inner_y = subsection_rect.y + 10
        inner_width = subsection_rect.width - 20
        icon_rect = pygame.Rect(inner_x, inner_y, 24, 24)
        pygame.draw.rect(surface, icon_style["bg"], icon_rect, border_radius=6)
        pygame.draw.rect(surface, (236, 240, 244), icon_rect, width=1, border_radius=6)
        helpers.draw_shadow_text(
            surface,
            body_font,
            icon_style["label"],
            icon_rect.x + 6,
            icon_rect.y + 2,
            icon_style["fg"],
            shadow=(18, 22, 28),
            shadow_offset=1,
        )
        header_lines = helpers.wrap_lines(f"{faction}: {status}", body_font, inner_width - 30)
        helpers.draw_text_block(surface, body_font, header_lines, inner_x + 32, inner_y, color, 19)
        content_y = inner_y + max(26, len(header_lines) * 19) + 8

        detail_lines = [summary, f"Next: {', '.join(available_labels[:3]) if available_labels else '-'}"]
        if eta:
            detail_lines.insert(0, eta)
        for line in detail_lines:
            wrapped = helpers.wrap_lines(line, body_font, inner_width)
            helpers.draw_text_block(surface, body_font, wrapped, inner_x, content_y, PARCH_BODY, 19)
            content_y += len(wrapped) * 19 + 5
        bar_rect = pygame.Rect(inner_x, content_y + 6, inner_width, 12)
        pygame.draw.rect(surface, (155, 129, 89), bar_rect, border_radius=5)
        if progress_ratio > 0:
            fill_width = max(8, int(bar_rect.width * progress_ratio))
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, min(fill_width, bar_rect.width), bar_rect.height)
            pygame.draw.rect(surface, color, fill_rect, border_radius=5)
        pygame.draw.rect(surface, (124, 98, 65), bar_rect, width=1, border_radius=5)
        y = subsection_rect.bottom + 12


def _tech_tree_positions(
    tree_rect: pygame.Rect,
    scroll_y: int = 0,
    pan_x: int = 0,
) -> dict[str, tuple[int, int]]:
    tiers: dict[int, list[str]] = {}
    for tech_id, definition in tech.TECH_DEFS.items():
        tiers.setdefault(definition.column, []).append(tech_id)
    for tier_techs in tiers.values():
        tier_techs.sort(key=lambda tech_id: tech.TECH_DEFS[tech_id].row)

    positions: dict[str, tuple[int, int]] = {}
    center_x = tree_rect.centerx + pan_x
    start_y = tree_rect.y + 90 - scroll_y
    tier_gap = 210
    branch_gap = 170

    for tier_index in sorted(tiers):
        tier_techs = tiers[tier_index]
        y = start_y + (tier_index - 1) * tier_gap
        count = len(tier_techs)
        if count == 1:
            offsets = [0]
        elif count == 2:
            offsets = [-branch_gap // 2, branch_gap // 2]
        elif count == 3:
            offsets = [-branch_gap, 0, branch_gap]
        elif count == 4:
            offsets = [-int(branch_gap * 1.5), -branch_gap // 2, branch_gap // 2, int(branch_gap * 1.5)]
        elif count == 5:
            offsets = [-2 * branch_gap, -branch_gap, 0, branch_gap, 2 * branch_gap]
        else:
            step = branch_gap
            start = -((count - 1) * step) / 2
            offsets = [int(start + i * step) for i in range(count)]
        for tech_id, offset in zip(tier_techs, offsets):
            positions[tech_id] = (int(center_x + offset), int(y))

    return positions


def tech_tree_layout(
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    rect: pygame.Rect,
) -> TechTreeLayout:
    outer = rect.inflate(-18, -18)
    header_y = outer.y + 10 + title_font.get_height()
    header_y += body_font.get_height() + 4
    header_y += 12
    tree_rect = pygame.Rect(
        outer.x + 8,
        header_y + 8,
        outer.width - DETAIL_WIDTH - DETAIL_GAP - 16,
        outer.bottom - header_y - 58,
    )
    detail_rect = pygame.Rect(tree_rect.right + DETAIL_GAP, tree_rect.y, DETAIL_WIDTH, tree_rect.height)
    return TechTreeLayout(outer=outer, tree_rect=tree_rect, detail_rect=detail_rect)


def clamp_tech_tree_view(
    tree_rect: pygame.Rect,
    scroll_y: int,
    pan_x: int,
) -> tuple[int, int]:
    positions = _tech_tree_positions(tree_rect, scroll_y=0, pan_x=0)
    if not positions:
        return max(0, scroll_y), pan_x

    content_min_x = min(x - NODE_HALF_W for x, _ in positions.values())
    content_max_x = max(x + NODE_HALF_W for x, _ in positions.values())
    content_max_y = max(y + NODE_HALF_H for _, y in positions.values())

    min_pan_x = int(tree_rect.right - TREE_SIDE_PADDING - content_max_x)
    max_pan_x = int(tree_rect.x + TREE_SIDE_PADDING - content_min_x)
    if min_pan_x > max_pan_x:
        center_pan = (min_pan_x + max_pan_x) // 2
        min_pan_x = center_pan
        max_pan_x = center_pan

    max_scroll_y = max(0, int(content_max_y - (tree_rect.bottom - TREE_VERTICAL_PADDING)))
    clamped_scroll_y = max(0, min(max_scroll_y, scroll_y))
    clamped_pan_x = max(min_pan_x, min(max_pan_x, pan_x))
    return clamped_scroll_y, clamped_pan_x


def draw_tech_tree_overlay(
    helpers: ResearchDrawHelpers,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    rect: pygame.Rect,
    focus_faction: str,
    human_turn_active: bool,
    mouse_pos: tuple[int, int],
    scroll_y: int = 0,
    pan_x: int = 0,
) -> TechTreeDrawResult:
    outer = helpers.draw_parchment_panel_frame(surface, board_assets, rect)
    close_rect = helpers.draw_parchment_close_button(
        surface,
        board_assets,
        pygame.Rect(outer.right - 30, outer.y + 6, 24, 24),
    )
    header_y = helpers.draw_parchment_header(
        surface,
        title_font,
        body_font,
        outer,
        "Research tree",
        subtitle="Scroll to browse unlock paths and what each tech enables",
    )

    layout = tech_tree_layout(title_font, body_font, rect)
    tree_rect = layout.tree_rect
    detail_rect = layout.detail_rect

    inset_sprite = board_assets.ui_sprite("panelInset_beigeLight") or board_assets.ui_sprite("panelInset_beige")
    if not helpers.draw_scaled_sprite(surface, inset_sprite, tree_rect):
        pygame.draw.rect(surface, (216, 193, 144), tree_rect, border_radius=12)
        pygame.draw.rect(surface, (180, 152, 111), tree_rect, width=2, border_radius=12)
    if not helpers.draw_scaled_sprite(surface, inset_sprite, detail_rect):
        pygame.draw.rect(surface, (216, 193, 144), detail_rect, border_radius=12)
        pygame.draw.rect(surface, (180, 152, 111), detail_rect, width=2, border_radius=12)

    clip_prev = surface.get_clip()
    surface.set_clip(tree_rect)

    unlock_wrap_width = NODE_WIDTH - 22

    positions = _tech_tree_positions(tree_rect, scroll_y=scroll_y, pan_x=pan_x)
    node_rects: dict[str, pygame.Rect] = {}
    visible_top = tree_rect.y - 120
    visible_bottom = tree_rect.bottom + 120

    for tech_id, definition in tech.TECH_DEFS.items():
        start = positions[tech_id]
        if start[1] < visible_top or start[1] > visible_bottom:
            continue
        for req in definition.requires:
            end = positions[req]
            points = [
                (end[0], end[1] + NODE_HALF_H),
                (end[0], (end[1] + start[1]) // 2),
                (start[0], (end[1] + start[1]) // 2),
                (start[0], start[1] - NODE_HALF_H),
            ]
            pygame.draw.lines(surface, (164, 146, 118), False, points, 3)

    hovered_tech_id: str | None = None
    for tech_id in TECH_TREE_ORDER:
        cx, cy = positions[tech_id]
        if cy < visible_top or cy > visible_bottom:
            continue

        node_rect = pygame.Rect(cx - NODE_HALF_W, cy - NODE_HALF_H, NODE_WIDTH, NODE_HEIGHT)
        node_rects[tech_id] = node_rect

        red_status = tech_status(env, "Red", tech_id)
        blue_status = tech_status(env, "Blue", tech_id)
        focus_status = tech_status(env, focus_faction, tech_id)
        unlocked_any = red_status == "Done" or blue_status == "Done"
        ready_any = red_status == "Ready" or blue_status == "Ready"
        active_any = red_status.startswith("Active") or blue_status.startswith("Active")

        node_fill = (122, 114, 102)
        if unlocked_any:
            node_fill = (106, 133, 92)
        elif active_any:
            node_fill = (102, 121, 150)
        elif ready_any:
            node_fill = (145, 121, 78)

        node_border = (188, 168, 136)
        if human_turn_active and focus_status == "Ready":
            node_border = (240, 208, 148)
        if node_rect.collidepoint(mouse_pos):
            node_border = (247, 236, 212)
            hovered_tech_id = tech_id

        pygame.draw.rect(surface, node_fill, node_rect, border_radius=14)
        pygame.draw.rect(surface, node_border, node_rect, width=2, border_radius=14)

        style = TECH_ICON_STYLES.get(tech_id, {"label": "?", "bg": (74, 80, 88), "fg": (235, 239, 242)})
        icon_rect = pygame.Rect(node_rect.x + 10, node_rect.y + 10, 26, 26)
        pygame.draw.rect(surface, style["bg"], icon_rect, border_radius=7)
        pygame.draw.rect(surface, (242, 236, 224), icon_rect, width=1, border_radius=7)
        helpers.draw_shadow_text(surface, body_font, style["label"], icon_rect.x + 7, icon_rect.y + 3, style["fg"], shadow=PARCH_SHADOW, shadow_offset=0)

        label = helpers.fit_text(tech_label(tech_id), body_font, NODE_WIDTH - 50)
        helpers.draw_shadow_text(surface, body_font, label, node_rect.x + 42, node_rect.y + 11, PARCH_TITLE, shadow=PARCH_SHADOW, shadow_offset=0)

        summary_lines = helpers.wrap_lines(tech.TECH_DEFS[tech_id].summary, body_font, unlock_wrap_width)[:2]
        helpers.draw_text_block(surface, body_font, summary_lines, node_rect.x + 10, node_rect.y + 42, PARCH_BODY, 17)

        red_chip = pygame.Rect(node_rect.x + 10, node_rect.bottom - 20, 52, 16)
        blue_chip = pygame.Rect(node_rect.x + 68, node_rect.bottom - 20, 52, 16)
        pygame.draw.rect(surface, (146, 86, 82), red_chip, border_radius=8)
        pygame.draw.rect(surface, (88, 126, 170), blue_chip, border_radius=8)
        red_short = "Done" if red_status == "Done" else "Act" if red_status.startswith("Active") else "Ready" if red_status == "Ready" else "-"
        blue_short = "Done" if blue_status == "Done" else "Act" if blue_status.startswith("Active") else "Ready" if blue_status == "Ready" else "-"
        helpers.draw_shadow_text(surface, body_font, f"R {red_short}", red_chip.x + 7, red_chip.y + 1, TEXT_PRIMARY, shadow=(0, 0, 0), shadow_offset=1)
        helpers.draw_shadow_text(surface, body_font, f"B {blue_short}", blue_chip.x + 7, blue_chip.y + 1, TEXT_PRIMARY, shadow=(0, 0, 0), shadow_offset=1)

    surface.set_clip(clip_prev)

    detail_tech_id = hovered_tech_id
    if detail_tech_id is None:
        available = available_research_ids(env, focus_faction)
        detail_tech_id = available[0] if available else TECH_TREE_ORDER[0]

    detail_inner = detail_rect.inflate(-14, -14)
    helpers.draw_shadow_text(surface, title_font, tech_label(detail_tech_id), detail_inner.x + 8, detail_inner.y + 6, PARCH_TITLE, shadow=PARCH_SHADOW, shadow_offset=0)

    detail_lines = tech_detail_lines(env, focus_faction, detail_tech_id)
    wrapped: list[str] = []
    for line in detail_lines:
        wrapped.extend(helpers.wrap_lines(line, body_font, detail_inner.width - 16))
        wrapped.append("")
    if wrapped and wrapped[-1] == "":
        wrapped.pop()

    helpers.draw_text_block(surface, body_font, wrapped, detail_inner.x + 8, detail_inner.y + 46, PARCH_BODY, 22)
    footer = f"Viewing {focus_faction} research state"
    helpers.draw_shadow_text(surface, body_font, footer, detail_inner.x + 8, detail_inner.bottom - 24, PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0)

    legend = [("Done", (106, 133, 92)), ("Active", (102, 121, 150)), ("Ready", (145, 121, 78)), ("Locked", (122, 114, 102))]
    lx = outer.x + 12
    ly = outer.bottom - 28
    for label, color in legend:
        chip = pygame.Rect(lx, ly, 78, 20)
        pygame.draw.rect(surface, color, chip, border_radius=10)
        pygame.draw.rect(surface, PARCH_LINE, chip, width=1, border_radius=10)
        helpers.draw_shadow_text(surface, body_font, label, chip.x + 12, chip.y + 2, TEXT_PRIMARY, shadow=(0, 0, 0), shadow_offset=1)
        lx += 88

    return TechTreeDrawResult(node_rects=node_rects, close_rect=close_rect)
