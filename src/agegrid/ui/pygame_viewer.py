from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from src.agegrid.agents.heuristic import (
    army_plan,
    army_strength_near_base,
    defense_mode_active,
    push_mode_active,
    threat_level,
    unit_composition,
)
from src.agegrid.agents.registry import AGENT_SPECS, create_agent
from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env.systems import tech


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


RED_PRIMARY = (196, 88, 80)
RED_ACCENT = (241, 202, 160)
BLUE_PRIMARY = (88, 126, 212)
BLUE_ACCENT = (176, 222, 255)
HUD_BG = (15, 20, 27)
GRID_BG = (29, 35, 43)
GRID_ALT = (33, 40, 48)
GRID_LINE = (58, 69, 80)
RESOURCE_GLOW = (109, 192, 116)

TECH_ICON_STYLES = {
    "mining": {"label": "M", "bg": (79, 121, 82), "fg": (232, 245, 220)},
    "bronze_working": {"label": "B", "bg": (145, 103, 64), "fg": (247, 229, 202)},
    "masonry": {"label": "S", "bg": (102, 108, 120), "fg": (236, 240, 246)},
    "horsemanship": {"label": "H", "bg": (126, 90, 58), "fg": (246, 230, 205)},
    "fletching": {"label": "F", "bg": (72, 108, 134), "fg": (224, 238, 250)},
}

TECH_LABELS = {
    "mining": "Mining",
    "bronze_working": "Bronze",
    "masonry": "Masonry",
    "horsemanship": "Horse",
    "fletching": "Fletching",
}


def _tech_label(tech_id: str) -> str:
    return TECH_LABELS.get(tech_id, tech_id.replace("_", " ").title())


def _format_action(action: tuple | None) -> str:
    if action is None:
        return "stop"
    kind = action[0]
    if kind == "gather":
        return f"gather worker#{action[1]}"
    if kind == "move_towards":
        return f"move worker/unit#{action[1]} -> {action[2]}"
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
    return str(action)


def _build_turn_info(label: str, actions: list[tuple | None], log: list[str]) -> FactionTurnInfo:
    research = next((_format_action(action) for action in actions if action and action[0] == "research"), "-")
    attacks = next(
        (_format_action(action) for action in actions if action and action[0] in {"attack", "attack_base"}),
        "-",
    )
    return FactionTurnInfo(
        label=label,
        log=log,
        last_action=_format_action(actions[-1]) if actions else "stop",
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
    return _build_turn_info(faction, actions, log), actions


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
    lines: list[str],
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, (24, 31, 40), rect, border_radius=10)
    pygame.draw.rect(surface, (70, 84, 99), rect, width=2, border_radius=10)
    _draw_shadow_text(
        surface,
        title_font,
        "Recent events",
        rect.x + 12,
        rect.y + 10,
        (231, 236, 241),
        shadow=(10, 12, 16),
    )

    y = rect.y + 34
    max_width = rect.width - 24
    event_lines = lines if lines else ["-"]
    for entry in event_lines:
        wrapped = _wrap_lines(entry, body_font, max_width)
        color = _event_color(entry)
        _draw_text_block(surface, body_font, wrapped, rect.x + 12, y, color, 18)
        y += len(wrapped) * 18 + 8


def _draw_research_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    env: AgeGridEnv,
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, (24, 31, 40), rect, border_radius=10)
    pygame.draw.rect(surface, (70, 84, 99), rect, width=2, border_radius=10)
    _draw_shadow_text(
        surface,
        title_font,
        "Research status",
        rect.x + 12,
        rect.y + 10,
        (231, 236, 241),
        shadow=(10, 12, 16),
    )

    y = rect.y + 36
    max_width = rect.width - 24
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
            total_turns = tech.TECH_DEFS[state.tech_in_progress].turns
            progress_ratio = min(1.0, state.research_points / total_turns) if total_turns > 0 else 1.0
            status = f"Researching {_tech_label(state.tech_in_progress)}"
            eta = f"ETA: {turns_left} turn{'s' if turns_left != 1 else ''} left"
        else:
            progress_ratio = 0.0
            status = "No active research"
            eta = "ETA: choose a tech to begin"
        icon_key = state.tech_in_progress or (available[0] if available else None)
        icon_style = TECH_ICON_STYLES.get(icon_key or "", {"label": "?", "bg": (70, 76, 84), "fg": (232, 236, 240)})
        subsection_h = _research_subsection_height(body_font, faction, status, eta, available_labels, max_width)
        subsection_rect = pygame.Rect(rect.x + 10, y, rect.width - 20, subsection_h)
        pygame.draw.rect(surface, (30, 37, 46), subsection_rect, border_radius=8)
        pygame.draw.rect(surface, (82, 94, 108), subsection_rect, width=1, border_radius=8)

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
        _draw_text_block(surface, body_font, header_lines, inner_x + 30, inner_y, color, 18)
        content_y = inner_y + max(26, len(header_lines) * 18) + 6

        detail_lines = [
            eta,
            f"Available: {', '.join(available_labels) if available_labels else '-'}",
        ]
        for line in detail_lines:
            wrapped = _wrap_lines(line, body_font, inner_width)
            _draw_text_block(surface, body_font, wrapped, inner_x, content_y, color, 18)
            content_y += len(wrapped) * 18
            content_y += 4
        bar_rect = pygame.Rect(inner_x, content_y + 4, inner_width, 10)
        pygame.draw.rect(surface, (44, 52, 62), bar_rect, border_radius=5)
        if progress_ratio > 0:
            fill_width = max(8, int(bar_rect.width * progress_ratio))
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, min(fill_width, bar_rect.width), bar_rect.height)
            pygame.draw.rect(surface, color, fill_rect, border_radius=5)
        pygame.draw.rect(surface, (96, 108, 120), bar_rect, width=1, border_radius=5)
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
    body_font: pygame.font.Font,
    env: AgeGridEnv,
    width: int,
) -> int:
    total = 36
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
            status = f"Researching {_tech_label(state.tech_in_progress)}"
            eta = f"ETA: {turns_left} turn{'s' if turns_left != 1 else ''} left"
        else:
            status = "No active research"
            eta = "ETA: choose a tech to begin"
        summary_lines = [
            f"{faction}: {status}",
            eta,
            f"Available: {', '.join(available_labels) if available_labels else '-'}",
        ]
        total += _research_subsection_height(body_font, faction, status, eta, available_labels, width - 24)
        total += 12
    return total + 12


def _research_subsection_height(
    body_font: pygame.font.Font,
    faction: str,
    status: str,
    eta: str,
    available_labels: list[str],
    width: int,
) -> int:
    header_lines = _wrap_lines(f"{faction}: {status}", body_font, width - 40)
    detail_one = _wrap_lines(eta, body_font, width)
    detail_two = _wrap_lines(f"Available: {', '.join(available_labels) if available_labels else '-'}", body_font, width)
    total = 10
    total += max(26, len(header_lines) * 18)
    total += 6
    total += len(detail_one) * 18 + 4
    total += len(detail_two) * 18 + 4
    total += 24
    total += 10
    return total


def _tactical_panel_lines(env: AgeGridEnv, faction: str) -> list[str]:
    composition = unit_composition(env, faction)
    friendly_strength, enemy_strength = army_strength_near_base(env, faction)
    army_delta = friendly_strength - enemy_strength
    delta_label = f"+{army_delta}" if army_delta > 0 else str(army_delta)
    mode = "Defense" if defense_mode_active(env, faction) else "Push" if push_mode_active(env, faction) else "Field"
    return [
        f"{faction} threat: {threat_level(env, faction)}",
        f"{faction} mode: {mode}",
        f"{faction} army plan: {army_plan(env, faction)}",
        (
            f"{faction} comp: W{composition['worker']} "
            f"S{composition['soldier']} A{composition['archer']} H{composition['horseman']}"
        ),
        f"{faction} base force: {friendly_strength} vs {enemy_strength} ({delta_label})",
    ]


def _tactical_panel_height(body_font: pygame.font.Font, env: AgeGridEnv, width: int) -> int:
    total = 36
    content_width = width - 24
    for faction in ("Red", "Blue"):
        for line in _tactical_panel_lines(env, faction):
            total += len(_wrap_lines(line, body_font, content_width)) * 18
        total += 10
    return total + 8


def _draw_tactical_panel(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    env: AgeGridEnv,
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, (24, 31, 40), rect, border_radius=10)
    pygame.draw.rect(surface, (70, 84, 99), rect, width=2, border_radius=10)
    _draw_shadow_text(
        surface,
        title_font,
        "Tactical status",
        rect.x + 12,
        rect.y + 10,
        (231, 236, 241),
        shadow=(10, 12, 16),
    )

    y = rect.y + 36
    for faction, color in (("Red", (244, 184, 180)), ("Blue", (176, 214, 255))):
        for line in _tactical_panel_lines(env, faction):
            wrapped = _wrap_lines(line, body_font, rect.width - 24)
            _draw_text_block(surface, body_font, wrapped, rect.x + 12, y, color, 18)
            y += len(wrapped) * 18
        y += 10


def _tile_center(ox: int, oy: int, tile: int, pos: tuple[int, int]) -> tuple[int, int]:
    return (ox + pos[0] * tile + tile // 2, oy + pos[1] * tile + tile // 2)


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
        overlay = pygame.Surface((tile * 2, tile * 2), pygame.SRCALPHA)
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
) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
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
        "Controls after start: Space / Enter or click Next Turn.",
        "R resets to setup. P saves a debug snapshot. Esc closes the viewer.",
    ]
    tips_height = len(tips) * 18
    start_btn_y = blue_card.bottom + 28
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

    pygame.draw.rect(screen, (74, 102, 72), start_btn, border_radius=10)
    pygame.draw.rect(screen, (136, 181, 131), start_btn, width=2, border_radius=10)
    screen.blit(font.render("Start Match", True, (244, 246, 244)), (start_btn.x + 42, start_btn.y + 11))

    tips_y = min(height_px - 24 - tips_height, start_btn.bottom + 22)
    _draw_text_block(screen, small_font, tips, 40, tips_y, (180, 186, 192), 18)
    return red_card, blue_card, start_btn


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
        f"Current player: {env.factions[env.current_player]}",
        f"Winner: {env.winner() or '-'}",
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
        f"Red techs: {', '.join(sorted(env.faction_state('Red').techs_unlocked)) or '-'}",
        f"Blue techs: {', '.join(sorted(env.faction_state('Blue').techs_unlocked)) or '-'}",
        f"Red buildings: {', '.join(sorted(f'{b.building_type}@{b.position}' for b in env.buildings if b.faction == 'Red' and b.hp > 0)) or '-'}",
        f"Blue buildings: {', '.join(sorted(f'{b.building_type}@{b.position}' for b in env.buildings if b.faction == 'Blue' and b.hp > 0)) or '-'}",
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

    env = AgeGridEnv()

    tile = 48
    pad = 16
    top_bar = 252
    side_panel = 300
    board_width = env.config.width * tile

    width_px = pad * 3 + board_width + side_panel
    height_px = pad * 2 + top_bar + env.config.height * tile

    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    big = pygame.font.SysFont(None, 30)
    small = pygame.font.SysFont(None, 19)
    tiny = pygame.font.SysFont(None, 17)

    btn_w, btn_h = 140, 36
    btn_rect = pygame.Rect(width_px - pad - btn_w, pad, btn_w, btn_h)

    red_index = 0
    blue_index = 1 if len(AGENT_SPECS) > 1 else 0
    in_setup = True
    red_agent = None
    blue_agent = None
    red_info = FactionTurnInfo("Red", [])
    blue_info = FactionTurnInfo("Blue", [])
    turn_history: list[TurnSnapshot] = []
    effects: list[VisualEffect] = []

    running = True
    while running:
        clock.tick(60)

        if in_setup:
            red_card, blue_card, start_btn = _draw_setup_screen(
                screen,
                width_px,
                height_px,
                big,
                font,
                small,
                red_index,
                blue_index,
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
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        env = AgeGridEnv()
                        red_agent = create_agent(AGENT_SPECS[red_index].key, seed=0)
                        blue_agent = create_agent(AGENT_SPECS[blue_index].key, seed=1)
                        red_info = FactionTurnInfo("Red", [])
                        blue_info = FactionTurnInfo("Blue", [])
                        turn_history = []
                        effects = []
                        in_setup = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if red_card.collidepoint(event.pos):
                        red_index = (red_index + 1) % len(AGENT_SPECS)
                    elif blue_card.collidepoint(event.pos):
                        blue_index = (blue_index + 1) % len(AGENT_SPECS)
                    elif start_btn.collidepoint(event.pos):
                        env = AgeGridEnv()
                        red_agent = create_agent(AGENT_SPECS[red_index].key, seed=0)
                        blue_agent = create_agent(AGENT_SPECS[blue_index].key, seed=1)
                        red_info = FactionTurnInfo("Red", [])
                        blue_info = FactionTurnInfo("Blue", [])
                        turn_history = []
                        effects = []
                        in_setup = False
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    in_setup = True
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
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if env.winner() is None:
                        red_info, blue_info, red_actions, blue_actions = _step_full_turn(env, red_agent, blue_agent)
                        turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))
                        effects.extend(_effects_from_actions(env, red_actions, "Red"))
                        effects.extend(_effects_from_actions(env, blue_actions, "Blue"))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(event.pos) and env.winner() is None:
                    red_info, blue_info, red_actions, blue_actions = _step_full_turn(env, red_agent, blue_agent)
                    turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))
                    effects.extend(_effects_from_actions(env, red_actions, "Red"))
                    effects.extend(_effects_from_actions(env, blue_actions, "Blue"))

        effects = _update_effects(effects)
        screen.fill((10, 14, 20))
        hud_rect = pygame.Rect(0, 0, width_px, top_bar)
        pygame.draw.rect(screen, HUD_BG, hud_rect)
        pygame.draw.line(screen, (48, 58, 72), (0, top_bar - 1), (width_px, top_bar - 1), 2)
        board_x = pad
        sidebar_x = board_x + board_width + pad
        sidebar_rect = pygame.Rect(sidebar_x, pad, side_panel, height_px - pad * 2)
        pygame.draw.rect(screen, (18, 24, 32), sidebar_rect, border_radius=14)
        pygame.draw.rect(screen, (54, 66, 82), sidebar_rect, width=2, border_radius=14)

        red_workers = sum(1 for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        blue_workers = sum(1 for u in env.units if u.faction == "Blue" and u.unit_type == "worker")
        red_military = sum(1 for u in env.units if u.faction == "Red" and u.attack_damage > 0)
        blue_military = sum(1 for u in env.units if u.faction == "Blue" and u.attack_damage > 0)
        red_defense_mode = defense_mode_active(env, "Red")
        blue_defense_mode = defense_mode_active(env, "Blue")
        red_push_mode = push_mode_active(env, "Red")
        blue_push_mode = push_mode_active(env, "Blue")
        red_techs = ", ".join(_tech_label(tech_id) for tech_id in sorted(env.faction_state("Red").techs_unlocked)) or "-"
        blue_techs = ", ".join(_tech_label(tech_id) for tech_id in sorted(env.faction_state("Blue").techs_unlocked)) or "-"
        red_buildings = ", ".join(sorted(b.building_type for b in env.buildings if b.faction == "Red" and b.hp > 0)) or "-"
        blue_buildings = ", ".join(sorted(b.building_type for b in env.buildings if b.faction == "Blue" and b.hp > 0)) or "-"
        winner = env.winner()

        _draw_shadow_text(
            screen,
            big,
            f"Turn {env.turn} | Current: {env.factions[env.current_player]} | Winner: {winner or '-'}",
            board_x,
            pad,
            (240, 240, 240),
            shadow=(5, 8, 12),
        )

        red_panel = pygame.Rect(board_x, pad + 40, board_width // 2 - 8, 192)
        blue_panel = pygame.Rect(board_x + board_width // 2 + 8, pad + 40, board_width // 2 - 8, 192)
        for rect, color in ((red_panel, RED_PRIMARY), (blue_panel, BLUE_PRIMARY)):
            pygame.draw.rect(screen, (23, 30, 38), rect, border_radius=12)
            pygame.draw.rect(screen, color, rect, width=2, border_radius=12)

        if red_defense_mode:
            _draw_status_badge(screen, tiny, "Defense Mode", red_panel.right - 126, red_panel.y + 12, (90, 46, 46), (212, 116, 116), (249, 224, 224))
        elif red_push_mode:
            _draw_status_badge(screen, tiny, "Push Mode", red_panel.right - 110, red_panel.y + 12, (58, 78, 44), (156, 202, 118), (235, 247, 226))
        if blue_defense_mode:
            _draw_status_badge(screen, tiny, "Defense Mode", blue_panel.right - 126, blue_panel.y + 12, (46, 66, 96), (116, 164, 224), (224, 238, 252))
        elif blue_push_mode:
            _draw_status_badge(screen, tiny, "Push Mode", blue_panel.right - 110, blue_panel.y + 12, (58, 78, 44), (156, 202, 118), (235, 247, 226))

        _draw_labeled_block(
            screen,
            font,
            small,
            f"Red {AGENT_SPECS[red_index].label}",
            [
                f"Bank {env.bank['Red']} | Workers {red_workers} | Military {red_military} | Base HP {env.bases['Red'].hp}",
                f"Techs: {red_techs}",
                f"Buildings: {red_buildings}",
                f"Last: {red_info.last_action}",
                f"Research: {red_info.research}",
                f"Attack: {red_info.attacks}",
                f"Log: {', '.join(red_info.log) if red_info.log else '-'}",
            ],
            red_panel.x + 12,
            red_panel.y + 10,
            red_panel.width - 24,
            (245, 226, 221),
            (212, 214, 218),
        )
        _draw_labeled_block(
            screen,
            font,
            small,
            f"Blue {AGENT_SPECS[blue_index].label}",
            [
                f"Bank {env.bank['Blue']} | Workers {blue_workers} | Military {blue_military} | Base HP {env.bases['Blue'].hp}",
                f"Techs: {blue_techs}",
                f"Buildings: {blue_buildings}",
                f"Last: {blue_info.last_action}",
                f"Research: {blue_info.research}",
                f"Attack: {blue_info.attacks}",
                f"Log: {', '.join(blue_info.log) if blue_info.log else '-'}",
            ],
            blue_panel.x + 12,
            blue_panel.y + 10,
            blue_panel.width - 24,
            (221, 232, 247),
            (212, 214, 218),
        )

        research_panel_h = _research_panel_height(tiny, env, side_panel - 24)
        research_panel = pygame.Rect(sidebar_x + 12, btn_rect.bottom + 78, side_panel - 24, research_panel_h)
        _draw_research_panel(screen, font, tiny, env, research_panel)

        tactical_panel_h = _tactical_panel_height(tiny, env, side_panel - 24)
        tactical_panel = pygame.Rect(sidebar_x + 12, research_panel.bottom + 12, side_panel - 24, tactical_panel_h)
        _draw_tactical_panel(screen, font, tiny, env, tactical_panel)

        event_panel_h = max(160, sidebar_rect.bottom - pad - (tactical_panel.bottom + 12))
        event_panel = pygame.Rect(sidebar_x + 12, tactical_panel.bottom + 12, side_panel - 24, event_panel_h)
        _draw_event_panel(
            screen,
            font,
            tiny,
            env.recent_events[-6:],
            event_panel,
        )

        pygame.draw.rect(screen, (46, 54, 66), btn_rect, border_radius=10)
        border_color = (120, 120, 120) if winner is None else (90, 90, 90)
        text_color = (245, 245, 245) if winner is None else (170, 170, 170)
        pygame.draw.rect(screen, border_color, btn_rect, width=2, border_radius=10)
        btn_label = "Next Turn" if winner is None else "Game Over"
        _draw_shadow_text(screen, font, btn_label, btn_rect.x + 24, btn_rect.y + 9, text_color, shadow=(12, 14, 18))
        reset_message = "Press R to return to setup"
        if winner is not None:
            reset_message = f"{winner} wins - Press R to reset"
        sidebar_notes = _wrap_lines(reset_message, tiny, side_panel - 32)
        sidebar_notes.extend(_wrap_lines("Press P for debug dump", tiny, side_panel - 32))
        _draw_text_block(screen, tiny, sidebar_notes, sidebar_x + 16, btn_rect.y + 52, (190, 196, 202), 18)

        ox = board_x
        oy = pad + top_bar

        for y in range(env.config.height):
            for x in range(env.config.width):
                rect = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
                base_color = GRID_BG if (x + y) % 2 == 0 else GRID_ALT
                pygame.draw.rect(screen, base_color, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, width=1)

        visible_special_resources = {
            resource.id
            for resource in env.resources
            if resource.required_tech is None
            or any(resource.required_tech in env.faction_state(faction).techs_unlocked for faction in env.factions)
        }
        for r in env.resources:
            if r.remaining <= 0:
                continue
            if r.id not in visible_special_resources:
                continue
            x, y = r.position
            cx = ox + x * tile + tile // 2
            cy = oy + y * tile + tile // 2
            if r.resource_type == "horses":
                pygame.draw.circle(screen, (89, 74, 42), (cx, cy), 15)
                pygame.draw.circle(screen, (212, 181, 104), (cx, cy), 11)
                glyph = "H"
                amount_color = (246, 228, 180)
            elif r.resource_type == "stone":
                pygame.draw.circle(screen, (86, 92, 102), (cx, cy), 15)
                pygame.draw.circle(screen, (168, 176, 190), (cx, cy), 11)
                glyph = "S"
                amount_color = (232, 236, 244)
            else:
                pygame.draw.circle(screen, (42, 92, 58), (cx, cy), 15)
                pygame.draw.circle(screen, RESOURCE_GLOW, (cx, cy), 11)
                pygame.draw.circle(screen, (220, 245, 196), (cx - 3, cy - 3), 4)
                glyph = None
                amount_color = (222, 241, 214)
            if glyph is not None:
                _draw_shadow_text(screen, tiny, glyph, cx - 5, cy - 8, (28, 22, 14), shadow=(240, 220, 176), shadow_offset=1)
            _draw_shadow_text(screen, tiny, str(r.remaining), cx - 8, cy + 10, amount_color, shadow=(16, 28, 18), shadow_offset=1)

        for faction, base in env.bases.items():
            x, y = base.position
            rect = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
            color = RED_PRIMARY if faction == "Red" else BLUE_PRIMARY
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (242, 232, 220), rect, width=2)
            inner = rect.inflate(-14, -14)
            pygame.draw.rect(screen, (243, 229, 195), inner)
            _draw_shadow_text(screen, small, str(base.hp), rect.x + 14, rect.y + 14, (245, 245, 245), shadow=(44, 44, 44), shadow_offset=1)

        for b in env.buildings:
            x, y = b.position
            rect = pygame.Rect(ox + x * tile + 8, oy + y * tile + 8, tile - 16, tile - 16)
            color = (213, 136, 104) if b.faction == "Red" else (123, 164, 230)
            pygame.draw.rect(screen, color, rect, border_radius=6)
            pygame.draw.rect(screen, (242, 233, 220), rect, width=2, border_radius=6)
            roof = [(rect.x + 4, rect.y + 6), (rect.centerx, rect.y - 6), (rect.right - 4, rect.y + 6)]
            pygame.draw.polygon(screen, (86, 58, 42), roof)
            _draw_shadow_text(screen, small, b.building_type[0].upper(), rect.x + 10, rect.y + 8, (24, 24, 24), shadow=(230, 220, 210), shadow_offset=1)

        for u in env.units:
            x, y = u.position
            cx = ox + x * tile + tile // 2
            cy = oy + y * tile + tile // 2
            color = (242, 206, 142) if u.faction == "Red" else (189, 225, 255)
            border = RED_PRIMARY if u.faction == "Red" else BLUE_PRIMARY
            if u.unit_type == "worker":
                points = [(cx, cy - 13), (cx + 12, cy), (cx, cy + 13), (cx - 12, cy)]
                pygame.draw.polygon(screen, color, points)
                pygame.draw.polygon(screen, border, points, width=2)
            elif u.unit_type == "soldier":
                points = [(cx, cy - 14), (cx + 13, cy - 2), (cx + 8, cy + 14), (cx - 8, cy + 14), (cx - 13, cy - 2)]
                pygame.draw.polygon(screen, color, points)
                pygame.draw.polygon(screen, border, points, width=2)
            elif u.unit_type == "horseman":
                points = [(cx, cy - 14), (cx + 14, cy - 6), (cx + 10, cy + 12), (cx - 2, cy + 15), (cx - 14, cy + 4), (cx - 10, cy - 10)]
                pygame.draw.polygon(screen, color, points)
                pygame.draw.polygon(screen, border, points, width=2)
            else:
                pygame.draw.circle(screen, color, (cx, cy), 13)
                pygame.draw.circle(screen, border, (cx, cy), 13, width=2)
            label = "W" if u.unit_type == "worker" else ("H" if u.unit_type == "horseman" else u.unit_type[0].upper())
            _draw_shadow_text(screen, small, label, cx - 5, cy - 8, (18, 18, 18), shadow=(236, 236, 236), shadow_offset=1)
            if u.attack_damage > 0:
                pygame.draw.line(screen, border, (cx + 10, cy - 10), (cx + 18, cy - 18), 3)
                pygame.draw.line(screen, border, (cx + 13, cy - 20), (cx + 18, cy - 18), 2)
                pygame.draw.line(screen, border, (cx + 18, cy - 18), (cx + 16, cy - 13), 2)

        _draw_effects(screen, effects, ox, oy, tile, small)

        pygame.display.flip()

    pygame.quit()
