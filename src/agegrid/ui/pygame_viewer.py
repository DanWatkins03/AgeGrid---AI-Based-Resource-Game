from __future__ import annotations

import pygame
from typing import Tuple

from src.agegrid.env.agegrid_env import AgeGridEnv

Position = Tuple[int, int]


def _nearest_resource_pos(env: AgeGridEnv, pos: Position) -> Position | None:
    # only consider non-empty nodes
    nodes = [r for r in env.resources if r.remaining > 0]
    if not nodes:
        return None

    def dist(p: Position) -> int:
        return abs(p[0] - pos[0]) + abs(p[1] - pos[1])

    return min(nodes, key=lambda r: dist(r.position)).position


def _decide_action(env: AgeGridEnv) -> tuple | None:
    """
    Decide ONE action for the env's current faction.
    Returns an action tuple or None to stop early.
    """
    faction = env.factions[env.current_player]

    workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
    if not workers:
        return None
    w = workers[0]

    # If standing on a resource, gather
    if env._resource_at(w.position) is not None:
        return ("gather", w.id)

    # Otherwise move towards nearest resource
    target = _nearest_resource_pos(env, w.position)
    if target is None:
        return None

    return ("move_towards", w.id, target)



def _step_full_turn(env: AgeGridEnv) -> tuple[list[str], list[str]]:
    # Red acts (current_player should be 0 / Red)
    red_log = env.step_faction(_decide_action)
    env.step_end_turn()  # switch to Blue

    # Blue acts
    blue_log = env.step_faction(_decide_action)
    env.step_end_turn()  # switch back to Red, increments env.turn

    return red_log, blue_log



def run_viewer() -> None:
    env = AgeGridEnv()

    pygame.init()
    pygame.display.set_caption("AgeGrid Viewer (v1)")

    tile = 48
    pad = 16
    top_bar = 90

    width_px = pad * 2 + env.config.width * tile
    height_px = pad * 2 + top_bar + env.config.height * tile

    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    big = pygame.font.SysFont(None, 28)

    # Button
    btn_w, btn_h = 140, 36
    btn_rect = pygame.Rect(width_px - pad - btn_w, pad, btn_w, btn_h)

    last_red: list[str] = []
    last_blue: list[str] = []

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    last_red, last_blue = _step_full_turn(env)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(event.pos):
                    last_red, last_blue = _step_full_turn(env)

        # ---------- Draw ----------
        screen.fill((22, 22, 22))

        # Top info bar
        title = big.render(
        f"Turn {env.turn} | Current: {env.factions[env.current_player]} | "
        f"Red bank: {env.bank['Red']} | Blue bank: {env.bank['Blue']} | "
        f"Actions left: {env.actions_left} | Attempts left: {env.attempts_left}",
        True,
        (240, 240, 240),
        )

        screen.blit(title, (pad, pad))

        # Last actions
        lr = font.render(f"Red actions: {', '.join(last_red) if last_red else '-'}", True, (210, 210, 210))
        lb = font.render(f"Blue actions: {', '.join(last_blue) if last_blue else '-'}", True, (210, 210, 210))
        screen.blit(lr, (pad, pad + 34))
        screen.blit(lb, (pad, pad + 58))

        # Button
        pygame.draw.rect(screen, (60, 60, 60), btn_rect, border_radius=8)
        pygame.draw.rect(screen, (120, 120, 120), btn_rect, width=2, border_radius=8)
        btn_text = font.render("Next Turn", True, (245, 245, 245))
        screen.blit(btn_text, (btn_rect.x + 24, btn_rect.y + 9))

        # Grid origin
        ox = pad
        oy = pad + top_bar

        # Draw tiles
        for y in range(env.config.height):
            for x in range(env.config.width):
                rect = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
                pygame.draw.rect(screen, (35, 35, 35), rect)
                pygame.draw.rect(screen, (55, 55, 55), rect, width=1)

        # Draw resources
        for r in env.resources:
            if r.remaining <= 0:
                continue
            x, y = r.position
            cx = ox + x * tile + tile // 2
            cy = oy + y * tile + tile // 2
            pygame.draw.circle(screen, (60, 160, 90), (cx, cy), 10)

        # Draw bases
        for faction, base in env.bases.items():
            x, y = base.position
            rect = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
            color = (180, 60, 60) if faction == "Red" else (70, 90, 190)
            pygame.draw.rect(screen, color, rect)

        # Draw units (workers)
        for u in env.units:
            x, y = u.position
            cx = ox + x * tile + tile // 2
            cy = oy + y * tile + tile // 2
            color = (240, 210, 120) if u.faction == "Red" else (180, 220, 255)
            pygame.draw.circle(screen, color, (cx, cy), 12)

        pygame.display.flip()

    pygame.quit()
