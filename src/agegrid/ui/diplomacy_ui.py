"""Diplomacy and war-resolution UI panels for AgeGrid.

Provides war declaration confirmation, peace negotiation, and war status HUD
components. All drawing functions are decoupled from simulation logic and work
through the PanelDrawHelpers abstraction defined in viewer_panels.py.
"""
from __future__ import annotations

import pygame

from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env.systems.diplomacy import (
    preview_peace_concessions,
    recommended_peace_indemnity,
)
from src.agegrid.env.systems import production
from src.agegrid.ui.assets import BoardAssets
from src.agegrid.ui.viewer_panels import PanelColors, PanelDrawHelpers

# ── Colour palette (mirrors pygame_viewer.py) ─────────────────────────────────
PARCH_TITLE  = (68,  50,  34)
PARCH_BODY   = (82,  63,  42)
PARCH_MUTED  = (106, 84,  60)
PARCH_SHADOW = (240, 224, 196)
PARCH_LINE   = (188, 168, 136)
PARCH_GOOD   = (72,  120, 70)
PARCH_WARN   = (170, 112, 48)
PARCH_DANGER = (164, 76,  60)
PARCH_INFO   = (72,  106, 152)

RED_PRIMARY  = (196, 88,  80)
BLUE_PRIMARY = (88,  126, 212)
TEXT_PRIMARY = (236, 240, 245)
TEXT_MUTED   = (127, 137, 149)

# Dark-panel colours for the war-status bar (not parchment)
_BAR_BG     = (15, 20, 28)
_BAR_BORDER = (40, 48, 59)

# War outcome bands: (min_score_diff, label, colour)
_OUTCOME_BANDS: list[tuple[int, str, tuple[int, int, int]]] = [
    (12,    "Dominating",       (72,  180, 80)),
    (5,     "Winning",          (94,  160, 78)),
    (2,     "Slightly Winning", (138, 176, 82)),
    (-1,    "Even War",         (170, 148, 72)),
    (-5,    "Slightly Losing",  (186, 128, 66)),
    (-10,   "Losing",           (180, 88,  66)),
    (-9999, "Collapsing",       (196, 68,  60)),
]


# ── Public data helpers ────────────────────────────────────────────────────────

def war_outcome_label(
    my_score: int, enemy_score: int,
) -> tuple[str, tuple[int, int, int]]:
    """Return (label, colour) describing the current war standing."""
    diff = my_score - enemy_score
    for threshold, label, color in _OUTCOME_BANDS:
        if diff >= threshold:
            return label, color
    return "Collapsing", (196, 68, 60)


def relation_chip_text(env: AgeGridEnv, faction: str, enemy: str) -> str:
    """Short text for the faction-bar relation chip (fits ~8 chars)."""
    state = env.relation_state(faction, enemy)
    if state.state == "war":
        mine  = state.war_score.get(faction, 0)
        theirs = state.war_score.get(enemy, 0)
        return f"War {mine}:{theirs}"
    if state.state == "truce":
        turns_left = max(0, state.truce_until_turn - env.turn)
        return f"Truce {turns_left}t"
    return "Peace"


def ai_flavor_text(
    env: AgeGridEnv, player_faction: str, enemy_faction: str,
) -> str:
    """Context-sensitive flavour text about the current diplomatic situation."""
    relation = env.relation_state(player_faction, enemy_faction)
    p_state  = env.faction_state(player_faction)
    e_state  = env.faction_state(enemy_faction)

    if relation.state == "war":
        p_score = relation.war_score.get(player_faction, 0)
        e_score = relation.war_score.get(enemy_faction, 0)
        turns   = max(0, env.turn - relation.since_turn)

        if relation.pending_peace_by == enemy_faction:
            if e_score < p_score:
                return f"{enemy_faction} seeks peace from a position of weakness."
            if e_score > p_score + 5:
                return f"{enemy_faction} offers peace, knowing they hold the advantage."
            return f"{enemy_faction} proposes an end to hostilities."

        if e_state.war_support <= 25:
            return f"{enemy_faction}'s war effort is crumbling. Their people demand an end."
        if p_state.war_support <= 25:
            return "Your people are exhausted. The cost of war is too great."
        if p_score >= e_score + 10:
            return f"{enemy_faction} is struggling to hold the line. Press the assault."
        if e_score >= p_score + 10:
            return "You are losing ground. Seek terms before the situation worsens."
        if turns >= 18:
            return "This war has ground on for too long. Both sides pay a heavy toll."
        if p_state.war_support <= 45:
            return "War fatigue is spreading. Your people grow weary of the conflict."
        return "The conflict continues. Each side fights for the upper hand."

    if relation.state == "truce":
        turns_left = max(0, relation.truce_until_turn - env.turn)
        if relation.failed_aggressor == player_faction:
            return f"Your failed war left your position weakened. Truce ends in {turns_left} turns."
        if turns_left <= 4:
            return "The truce is nearly over. Prepare for what comes next."
        return f"A fragile peace holds. The truce expires in {turns_left} turns."

    # Peace — pre-declaration context
    p_mil = [u for u in env.units if u.faction == player_faction and u.unit_type != "worker"]
    e_mil = [u for u in env.units if u.faction == enemy_faction  and u.unit_type != "worker"]
    if p_state.war_support <= 30:
        return "Your people have little appetite for war. Seek a better moment."
    if len(p_mil) > len(e_mil) + 2:
        return f"{enemy_faction}'s frontier looks vulnerable. A decisive strike could work."
    if len(e_mil) > len(p_mil) + 2:
        return f"{enemy_faction} fields a larger force. War now would be a gamble."
    return "Forces appear balanced. The outcome of war is uncertain."


# ── Drawing: War Declaration Modal ────────────────────────────────────────────

def draw_declare_war_modal(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    player_faction: str,
    enemy_faction: str,
    rect: pygame.Rect,
) -> tuple[pygame.Rect, pygame.Rect]:
    """Confirmation modal before declaring war.

    Returns (confirm_rect, cancel_rect) for click handling.
    """
    _dim(surface)
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)
    y = helpers.draw_parchment_header(
        surface, title_font, body_font, inner, f"Declare War on {enemy_faction}"
    )

    relation = env.relation_state(player_faction, enemy_faction)

    # Diplomatic status line
    if relation.state == "truce":
        turns_left = max(0, relation.truce_until_turn - env.turn)
        status_text  = f"Breaking truce early — {turns_left} turns remaining"
        status_color = PARCH_DANGER
    else:
        status_text  = "Diplomatic status: Peace"
        status_color = PARCH_MUTED
    _shadow(surface, small_font, status_text, inner.x + 12, y, status_color)
    y += small_font.get_height() + 8

    # ── Military comparison ─────────────────────────────────────────────────
    _section(surface, body_font, inner.x + 10, y, "Military Strength")
    y += body_font.get_height() + 6

    p_mil = [u for u in env.units if u.faction == player_faction and u.unit_type != "worker"]
    e_mil = [u for u in env.units if u.faction == enemy_faction  and u.unit_type != "worker"]
    p_col = RED_PRIMARY  if player_faction == "Red" else BLUE_PRIMARY
    e_col = RED_PRIMARY  if enemy_faction  == "Red" else BLUE_PRIMARY
    half_w = (inner.width - 20) // 2

    _army_chip(surface, body_font, small_font,
               pygame.Rect(inner.x + 8, y, half_w - 4, 46),
               "YOUR ARMY", len(p_mil), p_col)
    _army_chip(surface, body_font, small_font,
               pygame.Rect(inner.x + 8 + half_w + 4, y, half_w - 4, 46),
               f"{enemy_faction.upper()} ARMY", len(e_mil), e_col)
    y += 54

    # ── War costs ──────────────────────────────────────────────────────────
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 8, y), (inner.right - 8, y), 1)
    y += 8
    _section(surface, body_font, inner.x + 10, y, "War Costs")
    y += body_font.get_height() + 4

    aggressor_upkeep = env.config.war_upkeep_per_turn + env.config.war_upkeep_aggressor_bonus
    for line in [
        f"  Declaration:  -{env.config.war_declaration_cost} gold (paid now)",
        f"  Upkeep:       -{aggressor_upkeep}/turn as aggressor",
        f"  War support:  -{env.config.war_declaration_support_penalty} immediately",
    ]:
        _shadow(surface, small_font, line, inner.x + 10, y, PARCH_WARN)
        y += small_font.get_height() + 3
    y += 4

    # ── Assessment ─────────────────────────────────────────────────────────
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 8, y), (inner.right - 8, y), 1)
    y += 8
    _section(surface, body_font, inner.x + 10, y, "Assessment")
    y += body_font.get_height() + 4

    for text, positive in _war_assessment(env, player_faction, enemy_faction)[:4]:
        if positive is True:
            prefix, c = "+", PARCH_GOOD
        elif positive is False:
            prefix, c = "!", PARCH_DANGER
        else:
            prefix, c = "~", PARCH_WARN
        for w_line in helpers.wrap_lines(f"  {prefix} {text}", small_font, inner.width - 22)[:2]:
            _shadow(surface, small_font, w_line, inner.x + 10, y, c)
            y += small_font.get_height() + 2
    y += 4

    # ── Flavour text ───────────────────────────────────────────────────────
    for line in helpers.wrap_lines(
        f'"{ai_flavor_text(env, player_faction, enemy_faction)}"',
        small_font, inner.width - 22,
    )[:2]:
        _shadow(surface, small_font, line, inner.x + 12, y, PARCH_INFO)
        y += small_font.get_height() + 2

    # ── Buttons ────────────────────────────────────────────────────────────
    btn_h = 34
    btn_w = (inner.width - 18) // 2
    btn_y = inner.bottom - btn_h - 10
    confirm = pygame.Rect(inner.x + 6,          btn_y, btn_w, btn_h)
    cancel  = pygame.Rect(inner.x + btn_w + 12, btn_y, btn_w, btn_h)

    pygame.draw.rect(surface, (102, 40, 36), confirm, border_radius=8)
    pygame.draw.rect(surface, RED_PRIMARY,   confirm, width=2, border_radius=8)
    _btn_label(surface, body_font, confirm, "Declare War", (240, 200, 190))
    helpers.draw_parchment_button(surface, body_font, board_assets, cancel,
                                  "Cancel", active=False, enabled=True)
    return confirm, cancel


# ── Drawing: Peace Negotiation Panel ──────────────────────────────────────────

def draw_peace_panel(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    player_faction: str,
    enemy_faction: str,
    mode: str,          # "accept" | "offer"
    rect: pygame.Rect,
) -> tuple[pygame.Rect, pygame.Rect]:
    """Peace negotiation overlay with war status, terms, and consequence preview.

    mode="accept"  — enemy has offered peace; player decides whether to accept.
    mode="offer"   — player initiates a peace offer.

    Returns (confirm_rect, cancel_rect).
    """
    _dim(surface)
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)
    y = helpers.draw_parchment_header(
        surface, title_font, body_font, inner,
        "Accept Peace" if mode == "accept" else "Offer Peace",
    )

    relation = env.relation_state(player_faction, enemy_faction)
    p_state  = env.faction_state(player_faction)

    p_score      = relation.war_score.get(player_faction, 0)
    e_score      = relation.war_score.get(enemy_faction, 0)
    turns_at_war = max(0, env.turn - relation.since_turn)
    label, color = war_outcome_label(p_score, e_score)

    # ── War status banner ──────────────────────────────────────────────────
    banner = pygame.Rect(inner.x + 8, y, inner.width - 16, 38)
    pygame.draw.rect(surface, (18, 24, 32), banner, border_radius=8)
    pygame.draw.rect(surface, color, banner, width=2, border_radius=8)
    _shadow(surface, body_font, label, banner.x + 12, banner.y + 8, color)
    turns_text = f"Turn {turns_at_war} of war"
    _shadow(surface, small_font, turns_text,
            banner.right - small_font.size(turns_text)[0] - 12,
            banner.y + 11, PARCH_MUTED)
    y += 46

    # ── Stat chips (2 × 2 grid) ─────────────────────────────────────────────
    p_col  = RED_PRIMARY  if player_faction == "Red" else BLUE_PRIMARY
    e_col  = RED_PRIMARY  if enemy_faction  == "Red" else BLUE_PRIMARY
    chip_w = (inner.width - 20 - 4) // 2
    chip_h = 42
    chips: list[tuple[str, str, tuple[int, int, int]]] = [
        ("Your Score",  str(p_score),  p_col),
        ("Enemy Score", str(e_score),  e_col),
        ("Your Support",
         f"{p_state.war_support}%",
         _support_color(p_state.war_support)),
        ("Your Base HP",
         f"{env.bases[player_faction].hp}/{env.base_max_hp(player_faction)}",
         _hp_color(env.bases[player_faction].hp, env.base_max_hp(player_faction))),
    ]
    for i, (chip_label, chip_value, chip_color) in enumerate(chips):
        col = i % 2
        row = i // 2
        chip_rect = pygame.Rect(
            inner.x + 8 + col * (chip_w + 4),
            y + row * (chip_h + 4),
            chip_w, chip_h,
        )
        helpers.draw_parchment_chip(
            surface, chip_rect, small_font, body_font,
            chip_label, chip_value, chip_color,
        )
    y += 2 * (chip_h + 4) + 4

    pygame.draw.line(surface, PARCH_LINE, (inner.x + 8, y), (inner.right - 8, y), 1)
    y += 8

    # ── Peace terms ────────────────────────────────────────────────────────
    _section(surface, body_font, inner.x + 10, y, "Peace Terms")
    y += body_font.get_height() + 4

    if mode == "accept":
        indemnity   = relation.pending_indemnity
        term_color  = PARCH_GOOD if indemnity > 0 else PARCH_MUTED
        term_text   = (f"  You receive: +{indemnity} gold"
                       if indemnity > 0 else "  No indemnity offered")
        conc_payer  = player_faction
        conc_recv   = enemy_faction
    else:
        indemnity   = recommended_peace_indemnity(env, player_faction, enemy_faction)
        term_color  = PARCH_WARN if indemnity > 0 else PARCH_MUTED
        term_text   = (f"  You offer: -{indemnity} gold"
                       if indemnity > 0 else "  No indemnity required")
        conc_payer  = player_faction
        conc_recv   = enemy_faction

    _shadow(surface, small_font, term_text, inner.x + 10, y, term_color)
    y += small_font.get_height() + 8
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 8, y), (inner.right - 8, y), 1)
    y += 8

    # ── Consequence preview ────────────────────────────────────────────────
    _section(surface, body_font, inner.x + 10, y, "If Peace is Made:")
    y += body_font.get_height() + 4

    concession = preview_peace_concessions(env, conc_payer, conc_recv)
    positives, negatives = _peace_consequences(
        env, player_faction, enemy_faction, mode, indemnity, concession
    )
    for text in positives[:3]:
        _shadow(surface, small_font, text, inner.x + 12, y, PARCH_GOOD)
        y += small_font.get_height() + 2
    for text in negatives[:3]:
        _shadow(surface, small_font, text, inner.x + 12, y, PARCH_DANGER)
        y += small_font.get_height() + 2
    y += 4

    # ── Flavour text ───────────────────────────────────────────────────────
    for line in helpers.wrap_lines(
        f'"{ai_flavor_text(env, player_faction, enemy_faction)}"',
        small_font, inner.width - 22,
    )[:2]:
        _shadow(surface, small_font, line, inner.x + 12, y, PARCH_INFO)
        y += small_font.get_height() + 2

    # ── Buttons ────────────────────────────────────────────────────────────
    btn_h = 34
    btn_w = (inner.width - 18) // 2
    btn_y = inner.bottom - btn_h - 10
    confirm = pygame.Rect(inner.x + 6,          btn_y, btn_w, btn_h)
    cancel  = pygame.Rect(inner.x + btn_w + 12, btn_y, btn_w, btn_h)

    pygame.draw.rect(surface, (36, 84, 44), confirm, border_radius=8)
    pygame.draw.rect(surface, PARCH_GOOD,   confirm, width=2, border_radius=8)
    confirm_label = "Accept Peace" if mode == "accept" else "Offer Peace"
    _btn_label(surface, body_font, confirm, confirm_label, (190, 240, 196))

    cancel_label = "Refuse" if mode == "accept" else "Cancel"
    helpers.draw_parchment_button(surface, body_font, board_assets, cancel,
                                  cancel_label, active=False, enabled=True)
    return confirm, cancel


# ── Drawing: Compact War Status HUD strip ─────────────────────────────────────

def draw_war_status_hud(
    surface: pygame.Surface,
    small_font: pygame.font.Font,
    env: AgeGridEnv,
    player_faction: str,
    enemy_faction: str,
    x: int,
    y: int,
    _width: int,
) -> int:
    """Draw 1-2 compact war/truce status lines starting at (x, y).

    Returns the new y position after drawing (unchanged if no war/truce).
    Uses a subtle drop-shadow that matches the parchment HUD palette.
    """
    relation = env.relation_state(player_faction, enemy_faction)

    if relation.state == "truce":
        turns_left = max(0, relation.truce_until_turn - env.turn)
        text = f"Truce  |  {turns_left} turn{'s' if turns_left != 1 else ''} remaining"
        _shadow(surface, small_font, text, x, y, PARCH_WARN)
        return y + small_font.get_height() + 3

    if relation.state == "war":
        p_score = relation.war_score.get(player_faction, 0)
        e_score = relation.war_score.get(enemy_faction, 0)
        turns   = max(0, env.turn - relation.since_turn)
        support = env.faction_state(player_faction).war_support
        lbl, col = war_outcome_label(p_score, e_score)

        line1 = f"WAR  {player_faction} {p_score} : {e_score} {enemy_faction}  ({lbl})"
        _shadow(surface, small_font, line1, x, y, col)
        y += small_font.get_height() + 2

        line2 = f"Turn {turns}  |  Support {support}%"
        if relation.pending_peace_by is not None:
            line2 += f"  |  {relation.pending_peace_by} offered peace"
        _shadow(surface, small_font, line2, x, y, _support_color(support))
        y += small_font.get_height() + 3

    return y


# ── Private helpers ────────────────────────────────────────────────────────────

def _war_assessment(
    env: AgeGridEnv,
    player_faction: str,
    enemy_faction: str,
) -> list[tuple[str, bool | None]]:
    """Assessment items: (text, True=positive / False=negative / None=neutral)."""
    items: list[tuple[str, bool | None]] = []
    p_state = env.faction_state(player_faction)
    relation = env.relation_state(player_faction, enemy_faction)
    p_mil = [u for u in env.units if u.faction == player_faction and u.unit_type != "worker"]
    e_mil = [u for u in env.units if u.faction == enemy_faction  and u.unit_type != "worker"]

    if len(p_mil) > len(e_mil) + 1:
        items.append((f"Your army is stronger than {enemy_faction}'s.", True))
    elif len(e_mil) > len(p_mil) + 1:
        items.append((f"{enemy_faction}'s army outnumbers yours.", False))
    else:
        items.append(("Forces are roughly matched — victory is uncertain.", None))

    if p_state.war_support < 40:
        items.append(("Your war support is low. Failure could be devastating.", False))
    elif p_state.war_support >= 75:
        items.append(("High war support — your people back this war.", True))

    p_income = sum(
        s.resource_income
        for b in env.get_buildings_for_faction(player_faction)
        for s in [production.building_stats(env, player_faction, b.building_type)]
        if s is not None
    )
    if env.bank[player_faction] < env.config.war_declaration_cost + 6:
        items.append(("You are low on gold. War upkeep may drain you.", False))
    elif p_income >= 4:
        items.append((f"Your economy ({p_income:+}/turn) can sustain a long war.", True))

    p_base = env.bases[player_faction].position
    near_home = any(_hdist(u.position, p_base) <= 6 for u in e_mil)
    if near_home:
        items.append(("Enemy units are near your base — declaring war is risky.", False))

    e_base = env.bases[enemy_faction].position
    near_enemy = any(_hdist(u.position, e_base) <= 7 for u in p_mil)
    if near_enemy:
        items.append(("Your forces are close to enemy territory. A fast strike is possible.", True))

    if relation.state == "truce" and env.turn < relation.truce_until_turn:
        items.append(("Breaking a truce will significantly reduce war support.", False))

    p_hp  = env.bases[player_faction].hp
    p_max = env.base_max_hp(player_faction)
    if p_max > 0 and p_hp / p_max < 0.55:
        items.append(("Your base is damaged. A prolonged war may prove fatal.", False))

    return items[:5]


def _peace_consequences(
    env: AgeGridEnv,
    player_faction: str,
    enemy_faction: str,
    mode: str,
    indemnity: int,
    concession_preview,
) -> tuple[list[str], list[str]]:
    """Return (positives, negatives) string lists for the consequence preview."""
    pos: list[str] = []
    neg: list[str] = []

    upkeep = env.config.war_upkeep_per_turn + env.config.war_upkeep_aggressor_bonus

    if mode == "accept":
        if indemnity > 0:
            pos.append(f"+ Receive {indemnity} gold indemnity")
        else:
            pos.append("+ No payment required from you")
    else:
        if indemnity > 0:
            neg.append(f"- Pay {indemnity} gold indemnity")
        else:
            pos.append("+ No indemnity required")

    pos.append(f"+ End war upkeep (save {upkeep}/turn)")
    pos.append("+ War support recovery resumes")
    neg.append(f"- {env.config.truce_turns}-turn truce begins (no redeclaration)")

    if concession_preview.has_concessions:
        if concession_preview.reparations_per_turn > 0:
            neg.append(
                f"- Reparations: {concession_preview.reparations_per_turn}/turn"
                f" for {concession_preview.reparations_turns} turns"
            )
        if concession_preview.war_support_cap < 100:
            neg.append(
                f"- War support capped at {concession_preview.war_support_cap}"
                f" for {concession_preview.war_support_cap_turns} turns"
            )

    return pos, neg


def _hdist(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def _support_color(support: int) -> tuple[int, int, int]:
    if support >= 70:
        return (94, 160, 78)
    if support >= 45:
        return (170, 148, 72)
    if support >= 25:
        return (186, 128, 66)
    return (180, 88, 66)


def _hp_color(hp: int, max_hp: int) -> tuple[int, int, int]:
    ratio = hp / max_hp if max_hp > 0 else 0
    if ratio >= 0.7:
        return (94, 160, 78)
    if ratio >= 0.4:
        return (170, 148, 72)
    return (180, 88, 66)


def _dim(surface: pygame.Surface) -> None:
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.rect(overlay, (0, 0, 0, 158), overlay.get_rect())
    surface.blit(overlay, (0, 0))


def _shadow(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    surface.blit(font.render(text, True, PARCH_SHADOW), (x + 1, y + 1))
    surface.blit(font.render(text, True, color),        (x,     y))


def _section(
    surface: pygame.Surface,
    font: pygame.font.Font,
    x: int,
    y: int,
    text: str,
) -> None:
    _shadow(surface, font, text, x, y, PARCH_TITLE)


def _army_chip(
    surface: pygame.Surface,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    count: int,
    color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(surface, (26, 33, 42), rect, border_radius=8)
    pygame.draw.rect(surface, color,        rect, width=2, border_radius=8)
    _shadow(surface, small_font, label, rect.x + 10, rect.y + 5, color)
    count_text = f"{count} unit{'s' if count != 1 else ''}"
    _shadow(surface, body_font, count_text, rect.x + 10, rect.y + 22, PARCH_TITLE)


def _btn_label(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    text: str,
    color: tuple[int, int, int],
) -> None:
    tx = rect.centerx - font.size(text)[0] // 2
    ty = rect.y + (rect.height - font.get_height()) // 2
    _shadow(surface, font, text, tx, ty, color)


# ── Bottom war-status bar ──────────────────────────────────────────────────────

def draw_war_status_bar(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
    surface: pygame.Surface,
    small_font: pygame.font.Font,
    tiny_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    player_faction: str,
    enemy_faction: str,
    rect: pygame.Rect,
) -> None:
    """Informational-only bottom HUD strip showing current diplomatic state.

    No action buttons — all interaction is via the faction banner.
    Spans the full board width and stays visible at all times.
    """
    pygame.draw.rect(surface, _BAR_BG,     rect, border_radius=10)
    pygame.draw.rect(surface, _BAR_BORDER, rect, width=1, border_radius=10)

    relation = env.relation_state(player_faction, enemy_faction)
    f_state  = env.faction_state(player_faction)
    cfg      = env.config
    cy       = rect.y + (rect.height - small_font.get_height()) // 2
    ty       = rect.y + (rect.height - tiny_font.get_height()) // 2

    # ── State badge ──────────────────────────────────────────────────────────
    if relation.state == "war":
        badge_color, badge_text = PARCH_DANGER, "AT WAR"
    elif relation.state == "truce":
        badge_color, badge_text = PARCH_WARN, "TRUCE"
    else:
        badge_color, badge_text = PARCH_GOOD, "PEACE"

    b_surf = small_font.render(badge_text, True, badge_color)
    badge  = pygame.Rect(rect.x + 10, rect.y + 7, b_surf.get_width() + 16, rect.height - 14)
    pygame.draw.rect(surface, (22, 28, 38), badge, border_radius=6)
    pygame.draw.rect(surface, badge_color,  badge, width=2, border_radius=6)
    surface.blit(b_surf, (badge.x + 8, badge.y + (badge.height - b_surf.get_height()) // 2))

    x = badge.right + 10
    pygame.draw.line(surface, _BAR_BORDER, (x, rect.y + 7), (x, rect.bottom - 7), 1)
    x += 10

    # ── Content segments ─────────────────────────────────────────────────────
    segs: list[tuple[str, tuple[int, int, int]]] = []

    if relation.state == "peace":
        segs = [(f"Diplomatic relations with {enemy_faction} are peaceful.", TEXT_MUTED)]

    elif relation.state == "truce":
        truce_left = max(0, relation.truce_until_turn - env.turn)
        recovery   = cfg.peace_support_recovery_per_turn
        if relation.failed_aggressor == player_faction:
            recovery = cfg.failed_aggressor_truce_support_recovery_per_turn
        segs = [
            (f"With {enemy_faction}", TEXT_PRIMARY),
            (f"  |  Ends in {truce_left} turn{'s' if truce_left != 1 else ''}",
             PARCH_WARN if truce_left <= 5 else TEXT_MUTED),
            (f"  |  Support: {f_state.war_support}%"
             f"  ({'+' if recovery >= 0 else ''}{recovery}/t)",
             _support_color(f_state.war_support)),
        ]
        conc = relation.concessions
        if conc and conc.payer == player_faction and env.turn < conc.reparations_until_turn:
            rep_left = max(0, conc.reparations_until_turn - env.turn)
            segs.append((
                f"  |  Reparations: -{conc.reparations_per_turn}/t  ({rep_left}t left)",
                PARCH_DANGER,
            ))

    else:  # war
        p_score      = relation.war_score.get(player_faction, 0)
        e_score      = relation.war_score.get(enemy_faction, 0)
        turns_at_war = max(0, env.turn - relation.since_turn)
        upkeep       = cfg.war_upkeep_per_turn
        if relation.aggressor == player_faction:
            upkeep += cfg.war_upkeep_aggressor_bonus
        lbl, col = war_outcome_label(p_score, e_score)

        segs = [
            (f"With {enemy_faction}", TEXT_PRIMARY),
            (f"  |  Score  {p_score} : {e_score}  ({lbl})", col),
            (f"  |  Turn {turns_at_war}", TEXT_MUTED),
            (f"  |  Support: {f_state.war_support}%", _support_color(f_state.war_support)),
            (f"  |  Upkeep: -{upkeep}/t", PARCH_WARN),
        ]

        # Support cap indicator
        if f_state.war_support_cap < 100 and f_state.war_support_cap_until_turn > env.turn:
            cap_left = f_state.war_support_cap_until_turn - env.turn
            segs.append((f"  |  Cap: {f_state.war_support_cap}% ({cap_left}t)", PARCH_WARN))

        if relation.pending_peace_by == enemy_faction:
            segs.append((f"  |  {enemy_faction} offered peace!", PARCH_GOOD))
        elif relation.pending_peace_by == player_faction:
            segs.append((f"  |  You offered peace  ({relation.pending_indemnity}g)", PARCH_WARN))

    # Draw each segment, stopping before the right-side hint
    right_margin = rect.right - tiny_font.size(f"Click {enemy_faction} bar")[0] - 30
    for text, color in segs:
        surf = small_font.render(text, True, color)
        shad = small_font.render(text, True, PARCH_SHADOW)
        if x + surf.get_width() > right_margin:
            break
        surface.blit(shad, (x + 1, cy + 1))
        surface.blit(surf, (x, cy))
        x += surf.get_width()

    # ── Right-side hint ───────────────────────────────────────────────────────
    hint = f"Click {enemy_faction} bar to interact"
    h_s  = tiny_font.render(hint, True, TEXT_MUTED)
    surface.blit(h_s, (rect.right - h_s.get_width() - 12, ty))


# ── Faction diplomacy banner ──────────────────────────────────────────────────

def draw_faction_diplomacy_banner(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    player_faction: str,
    target_faction: str,
    rect: pygame.Rect,
) -> tuple[list[tuple[pygame.Rect, str, bool]], pygame.Rect]:
    """Faction interaction card opened by clicking the enemy's faction bar.

    Shows the target faction's current state and available diplomatic actions.
    Returns (button_rects, close_rect) where button_rects is
    a list of (rect, action_key, enabled).

    action_key is one of: "declare_war" | "offer_peace" | "accept_peace"
    Clicking an enabled button should open the corresponding confirmation modal.
    """
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)

    relation  = env.relation_state(player_faction, target_faction)
    t_state   = env.faction_state(target_faction)
    f_state   = env.faction_state(player_faction)
    t_color   = RED_PRIMARY if target_faction == "Red" else BLUE_PRIMARY

    # ── Header ───────────────────────────────────────────────────────────────
    close_rect = helpers.draw_parchment_close_button(
        surface, board_assets,
        pygame.Rect(inner.right - 26, inner.y + 6, 22, 22),
    )

    helpers.draw_shadow_text(
        surface, title_font, target_faction,
        inner.x + 12, inner.y + 8,
        t_color, shadow=PARCH_SHADOW, shadow_offset=0,
    )

    # Relation state chip (top-right, next to close button)
    if relation.state == "war":
        rel_text, rel_color = "AT WAR", PARCH_DANGER
    elif relation.state == "truce":
        tl = max(0, relation.truce_until_turn - env.turn)
        rel_text, rel_color = f"TRUCE  {tl}t", PARCH_WARN
    else:
        rel_text, rel_color = "PEACE", PARCH_GOOD

    rel_surf = small_font.render(rel_text, True, rel_color)
    rel_bg   = pygame.Rect(
        inner.right - 34 - rel_surf.get_width() - 14,
        inner.y + 10,
        rel_surf.get_width() + 14, 20,
    )
    pygame.draw.rect(surface, (20, 26, 34), rel_bg, border_radius=5)
    pygame.draw.rect(surface, rel_color, rel_bg, width=1, border_radius=5)
    surface.blit(rel_surf, (rel_bg.x + 7, rel_bg.y + 2))

    y = inner.y + title_font.get_height() + 14
    pygame.draw.line(surface, PARCH_LINE, (inner.x + 8, y), (inner.right - 8, y), 1)
    y += 8

    # ── Comparison section ────────────────────────────────────────────────────
    # Military count
    p_mil = sum(1 for u in env.units if u.faction == player_faction and u.unit_type != "worker")
    t_mil = sum(1 for u in env.units if u.faction == target_faction and u.unit_type != "worker")
    mil_color = PARCH_GOOD if p_mil > t_mil else PARCH_DANGER if p_mil < t_mil else PARCH_MUTED
    mil_text  = f"Military:  {player_faction} {p_mil}  vs  {target_faction} {t_mil}"
    helpers.draw_shadow_text(surface, small_font, mil_text, inner.x + 12, y,
                             mil_color, shadow=PARCH_SHADOW, shadow_offset=0)
    y += small_font.get_height() + 4

    # Target economy
    t_income = sum(
        spec.resource_income
        for b in env.get_buildings_for_faction(target_faction)
        for spec in [production.building_stats(env, target_faction, b.building_type)]
        if spec is not None
    )
    econ_text = f"Economy:  ${env.bank[target_faction]}  ({t_income:+}/turn)"
    helpers.draw_shadow_text(surface, small_font, econ_text, inner.x + 12, y,
                             PARCH_BODY, shadow=PARCH_SHADOW, shadow_offset=0)
    y += small_font.get_height() + 4

    # Target base HP
    t_hp  = env.bases[target_faction].hp
    t_max = env.base_max_hp(target_faction)
    hp_text  = f"Base HP:  {t_hp}/{t_max}"
    hp_color = _hp_color(t_hp, t_max)
    helpers.draw_shadow_text(surface, small_font, hp_text, inner.x + 12, y,
                             hp_color, shadow=PARCH_SHADOW, shadow_offset=0)
    y += small_font.get_height() + 4

    # War score (shown during war)
    if relation.state == "war":
        p_score = relation.war_score.get(player_faction, 0)
        t_score = relation.war_score.get(target_faction, 0)
        turns   = max(0, env.turn - relation.since_turn)
        lbl, col = war_outcome_label(p_score, t_score)
        score_text = (
            f"War Score:  {player_faction} {p_score} — {target_faction} {t_score}"
            f"  ({lbl}, T{turns})"
        )
        helpers.draw_shadow_text(
            surface, small_font,
            score_text[:48] if small_font.size(score_text)[0] > inner.width - 24 else score_text,
            inner.x + 12, y, col, shadow=PARCH_SHADOW, shadow_offset=0,
        )
        y += small_font.get_height() + 4

    pygame.draw.line(surface, PARCH_LINE, (inner.x + 8, y + 2), (inner.right - 8, y + 2), 1)
    y += 10

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_defs: list[tuple[str, str, bool, str]] = []  # label, key, enabled, reason

    if relation.state == "war":
        # Accept peace (only if enemy offered)
        if env.can_accept_peace(player_faction, target_faction):
            ind = relation.pending_indemnity
            lbl = f"Accept (+{ind}g)" if ind > 0 else "Accept Peace"
            btn_defs.append((lbl, "accept_peace", True, ""))

        # Offer peace
        if env.can_offer_peace(player_faction, target_faction):
            btn_defs.append(("Offer Peace", "offer_peace", True, ""))
        elif not env.can_accept_peace(player_faction, target_faction):
            tso = max(0, env.turn - relation.since_turn)
            tl  = max(0, env.config.peace_offer_min_turns - tso)
            reason = f"Wait {tl}t" if tl > 0 else "Offer pending"
            btn_defs.append(("Offer Peace", "offer_peace", False, reason))

    else:  # peace or truce
        if env.can_declare_war(player_faction, target_faction):
            btn_defs.append(("Declare War", "declare_war", True, ""))
        else:
            if relation.state == "truce":
                tl = max(0, relation.truce_until_turn - env.turn)
                reason = f"Truce: {tl}t"
            elif f_state.war_support < env.config.war_support_to_declare_min:
                reason = f"Support: {f_state.war_support}%"
            elif env.bank[player_faction] < env.config.war_declaration_cost:
                reason = f"Need {env.config.war_declaration_cost}g"
            else:
                reason = "Unavailable"
            btn_defs.append(("Declare War", "declare_war", False, reason))

    button_rects: list[tuple[pygame.Rect, str, bool]] = []
    if btn_defs:
        n      = len(btn_defs)
        gap    = 6
        btn_h  = 30
        btn_w  = (inner.width - 16 - gap * (n - 1)) // n
        hint_lines: list[str] = []

        for i, (lbl, key, enabled, reason) in enumerate(btn_defs):
            bx   = inner.x + 8 + i * (btn_w + gap)
            br   = pygame.Rect(bx, y, btn_w, btn_h)
            helpers.draw_parchment_button(
                surface, small_font, board_assets, br, lbl,
                active=False, enabled=enabled,
            )
            button_rects.append((br, key, enabled))
            if not enabled and reason:
                hint_lines.append(f"{lbl}: {reason}")

        if hint_lines:
            hint_y = y + btn_h + 4
            helpers.draw_shadow_text(
                surface, small_font,
                ", ".join(hint_lines)[:52],
                inner.x + 8, hint_y,
                PARCH_MUTED, shadow=PARCH_SHADOW, shadow_offset=0,
            )

    return button_rects, close_rect
