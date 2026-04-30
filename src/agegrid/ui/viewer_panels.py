from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame

from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.env.systems import production
from src.agegrid.ui.assets import BoardAssets


@dataclass(frozen=True)
class PanelDrawHelpers:
    draw_parchment_panel_frame: Callable[..., pygame.Rect]
    draw_parchment_header: Callable[..., int]
    draw_parchment_close_button: Callable[..., pygame.Rect]
    draw_unit_sprite: Callable[..., bool]
    draw_unit_icon: Callable[..., None]
    draw_shadow_text: Callable[..., pygame.Rect]
    draw_text_block: Callable[..., None]
    draw_ui_meter: Callable[..., None]
    draw_parchment_chip: Callable[..., None]
    draw_parchment_button: Callable[..., None]
    wrap_lines: Callable[[str, pygame.font.Font, int], list[str]]
    building_label: Callable[[str], str]


@dataclass(frozen=True)
class PanelText:
    unit_labels: dict[str, str]
    unit_help: dict[str, str]
    building_help: dict[str, str]
    resource_labels: dict[str, str]
    resource_help: dict[str, str]


@dataclass(frozen=True)
class PanelColors:
    red_primary: tuple[int, int, int]
    blue_primary: tuple[int, int, int]
    red_accent: tuple[int, int, int]
    blue_accent: tuple[int, int, int]
    parch_title: tuple[int, int, int]
    parch_body: tuple[int, int, int]
    parch_muted: tuple[int, int, int]
    parch_shadow: tuple[int, int, int]
    parch_line: tuple[int, int, int]
    text_primary: tuple[int, int, int]


def selected_unit_lines(text: PanelText, env: AgeGridEnv, unit) -> list[str]:
    label = text.unit_labels.get(unit.unit_type, unit.unit_type.replace("_", " ").title())
    help_text = text.unit_help.get(unit.unit_type, "Unit")
    spec = production.unit_stats(env, unit.faction, unit.unit_type)
    move_steps = spec.move_steps if spec is not None else unit.move_steps
    extras = [
        help_text,
        f"ATK {unit.attack_damage}  RNG {unit.attack_range}  MOVE {move_steps}",
        f"Position {unit.position[0]}, {unit.position[1]}",
    ]
    if unit.unit_type == "worker":
        resource = env.resource_at_for_faction(unit.position, unit.faction)
        if resource is None:
            extras.append("Gather: stand on a resource tile")
        elif env.can_gather_resource(unit, resource):
            extras.append("Gather: available from infinite source")
        else:
            extras.append("Gather blocked: resource contested")
    return [label, *extras]


def selected_resource_lines(text: PanelText, resource, env: AgeGridEnv | None = None, faction: str | None = None) -> list[str]:
    label = text.resource_labels.get(resource.resource_type, resource.resource_type.title())
    help_text = text.resource_help.get(resource.resource_type, "Strategic resource node.")
    extra = "Visible once its required tech is unlocked." if resource.required_tech else "Available to gather immediately."
    access = "Contested by enemy military" if env is not None and faction is not None and env.resource_is_contested(resource, faction) else "Infinite source"
    return [
        label,
        help_text,
        extra,
        access,
        f"Abundance {resource.abundance}",
    ]


def selected_building_lines(
    helpers: PanelDrawHelpers,
    text: PanelText,
    env: AgeGridEnv,
    building,
) -> list[str]:
    label = helpers.building_label(building.building_type)
    help_text = text.building_help.get(building.building_type, "Faction structure.")
    spec = production.building_stats(env, building.faction, building.building_type)
    extras: list[str] = [help_text]
    if spec is not None and spec.resource_income > 0:
        extras.append(f"Income +{spec.resource_income} each turn")
    if spec is not None and spec.attack_damage > 0:
        extras.append(f"Attack {spec.attack_damage}  Range {spec.attack_range}")
    extras.append(f"Position {building.position[0]}, {building.position[1]}")
    return [label, *extras]


def selected_base_lines(base, faction: str) -> list[str]:
    return [
        f"{faction} Base",
        "Primary stronghold. Lose this and the match is over.",
        f"Position {base.position[0]}, {base.position[1]}",
    ]


def draw_selected_unit_panel(
    helpers: PanelDrawHelpers,
    text: PanelText,
    colors: PanelColors,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    board_assets: BoardAssets,
    env: AgeGridEnv,
    unit,
    rect: pygame.Rect,
) -> pygame.Rect:
    accent = colors.red_primary if unit.faction == "Red" else colors.blue_primary
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)

    close_rect = helpers.draw_parchment_close_button(surface, board_assets, pygame.Rect(inner.right - 26, inner.y + 6, 22, 22))

    icon_center = (inner.x + 38, inner.y + 44)
    pygame.draw.circle(surface, (*accent, 48), icon_center, 24)
    if not helpers.draw_unit_sprite(
        surface,
        board_assets,
        env,
        unit,
        icon_center,
        colors.red_accent if unit.faction == "Red" else colors.blue_accent,
        accent,
        size=34,
    ):
        helpers.draw_unit_icon(
            surface,
            unit,
            icon_center,
            colors.red_accent if unit.faction == "Red" else colors.blue_accent,
            accent,
        )

    unit_name = text.unit_labels.get(unit.unit_type, unit.unit_type.replace("_", " ").title())
    title_x = inner.x + 78
    helpers.draw_shadow_text(surface, title_font, unit_name, title_x, inner.y + 16, colors.parch_title, shadow=colors.parch_shadow, shadow_offset=0)
    helpers.draw_shadow_text(
        surface,
        body_font,
        f"{unit.faction} #{unit.id}",
        title_x,
        inner.y + 42,
        accent,
        shadow=colors.parch_shadow,
        shadow_offset=0,
    )
    pygame.draw.line(surface, colors.parch_line, (inner.x + 12, inner.y + 72), (inner.right - 12, inner.y + 72), 1)

    spec = production.unit_stats(env, unit.faction, unit.unit_type)
    max_hp = spec.hp if spec is not None else unit.hp
    health_rect = pygame.Rect(inner.x + 14, inner.y + 92, inner.width - 28, 24)
    helpers.draw_shadow_text(surface, body_font, f"Health {unit.hp}/{max_hp}", health_rect.x, health_rect.y - 18, colors.parch_muted, shadow=colors.parch_shadow, shadow_offset=0)
    helpers.draw_ui_meter(surface, board_assets, health_rect, unit.hp, max_hp, color_family="red")

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
        helpers.draw_parchment_chip(surface, stat_rect, small_font, title_font, label, value, color)

    desc_y = stat_y + 64
    body_lines = selected_unit_lines(text, env, unit)[1:]
    wrapped: list[str] = []
    for line in body_lines:
        wrapped.extend(helpers.wrap_lines(line, body_font, inner.width - 28))
    helpers.draw_text_block(surface, body_font, wrapped, inner.x + 14, desc_y, colors.parch_body, 19)
    return close_rect


def draw_selected_object_panel(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
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
    accent: tuple[int, int, int] = (236, 240, 245),
) -> pygame.Rect:
    del env, small_font
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)

    close_rect = helpers.draw_parchment_close_button(surface, board_assets, pygame.Rect(inner.right - 26, inner.y + 6, 22, 22))

    icon_center = (inner.x + 38, inner.y + 44)
    pygame.draw.circle(surface, (*accent, 48), icon_center, 24)
    drew_icon = False
    if icon_kind is not None:
        sprite = board_assets.object_sprite(icon_kind)
        if sprite is not None:
            scaled = pygame.transform.smoothscale(sprite, (34, 34))
            screen_rect = scaled.get_rect(center=(icon_center[0], icon_center[1] + 2))
            surface.blit(scaled, screen_rect)
            drew_icon = True
    if not drew_icon and icon_drawer is not None:
        icon_drawer(icon_center)

    title_x = inner.x + 78
    helpers.draw_shadow_text(surface, title_font, title, title_x, inner.y + 16, colors.parch_title, shadow=colors.parch_shadow, shadow_offset=0)
    helpers.draw_shadow_text(surface, body_font, subtitle, title_x, inner.y + 42, accent, shadow=colors.parch_shadow, shadow_offset=0)
    pygame.draw.line(surface, colors.parch_line, (inner.x + 12, inner.y + 72), (inner.right - 12, inner.y + 72), 1)

    body_y = inner.y + 92
    if hp_value is not None and hp_max is not None:
        health_rect = pygame.Rect(inner.x + 14, body_y, inner.width - 28, 24)
        helpers.draw_shadow_text(surface, body_font, f"Health {hp_value}/{hp_max}", health_rect.x, health_rect.y - 18, colors.parch_muted, shadow=colors.parch_shadow, shadow_offset=0)
        helpers.draw_ui_meter(surface, board_assets, health_rect, hp_value, hp_max, color_family="red")
        body_y = health_rect.bottom + 20

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(helpers.wrap_lines(line, body_font, inner.width - 28))
    helpers.draw_text_block(surface, body_font, wrapped, inner.x + 14, body_y, colors.parch_body, 19)
    return close_rect


def draw_hover_tile_panel(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    lines: list[str],
) -> None:
    if not lines:
        return
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect, panel_key="panel_beigeLight", inset_key="panelInset_beigeLight")
    title = lines[0]
    title_y = inner.y + 8
    helpers.draw_shadow_text(surface, title_font, title, inner.x + 12, title_y, colors.parch_title, shadow=colors.parch_shadow, shadow_offset=0)
    divider_y = title_y + title_font.get_height() + 8
    pygame.draw.line(surface, colors.parch_line, (inner.x + 10, divider_y), (inner.right - 10, divider_y), 1)
    y = divider_y + 10
    wrapped: list[str] = []
    for line in lines[1:]:
        wrapped.extend(helpers.wrap_lines(line, body_font, inner.width - 16))
    helpers.draw_text_block(surface, body_font, wrapped, inner.x + 8, y, colors.parch_body, 18)


def draw_human_action_panel(
    helpers: PanelDrawHelpers,
    colors: PanelColors,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    board_assets: BoardAssets,
    rect: pygame.Rect,
    title: str,
    options: list,
    hint: str | None,
) -> list[tuple[pygame.Rect, object, bool]]:
    inner = helpers.draw_parchment_panel_frame(surface, board_assets, rect)
    y = helpers.draw_parchment_header(surface, title_font, body_font, inner, title)

    button_rects: list[tuple[pygame.Rect, object, bool]] = []
    columns = 2
    button_w = (inner.width - 18) // columns
    button_h = 30
    for idx, option in enumerate(options):
        col = idx % columns
        row = idx // columns
        button_rect = pygame.Rect(inner.x + 6 + col * (button_w + 6), y + row * (button_h + 8), button_w, button_h)
        helpers.draw_parchment_button(surface, body_font, board_assets, button_rect, option.label, active=option.active, enabled=option.enabled)
        button_rects.append((button_rect, option.payload, option.enabled))

    if hint:
        hint_y = y + ((len(options) + 1) // columns) * (button_h + 8) + 4
        wrapped = helpers.wrap_lines(hint, body_font, inner.width - 12)
        helpers.draw_text_block(surface, body_font, wrapped, inner.x + 6, hint_y, colors.parch_muted, 18)

    return button_rects


def human_action_panel_height(
    helpers: PanelDrawHelpers,
    body_font: pygame.font.Font,
    options: list,
    hint: str | None,
) -> int:
    columns = 2
    button_h = 30
    rows = (len(options) + columns - 1) // columns
    height = 74 + rows * (button_h + 8)
    if hint:
        height += len(helpers.wrap_lines(hint, body_font, 252)) * 16 + 8
    return max(106, height)
