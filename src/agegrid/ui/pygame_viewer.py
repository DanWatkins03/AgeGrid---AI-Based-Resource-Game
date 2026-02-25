from __future__ import annotations
import pygame

from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.agents.greedy import GreedyAgent


def _step_full_turn(env: AgeGridEnv, red_agent, blue_agent) -> tuple[list[str], list[str]]:
    red_log = env.step_faction(lambda e: red_agent.act(e))
    env.step_end_turn()

    blue_log = env.step_faction(lambda e: blue_agent.act(e))
    env.step_end_turn()

    return red_log, blue_log


def run_viewer() -> None:
    env = AgeGridEnv()

    red_agent = GreedyAgent(desired_workers=2)
    blue_agent = GreedyAgent(desired_workers=2)

    pygame.init()
    pygame.display.set_caption("AgeGrid Viewer (v1)")

    tile = 48
    pad = 16
    top_bar = 125

    width_px = pad * 2 + env.config.width * tile
    height_px = pad * 2 + top_bar + env.config.height * tile

    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()


    font = pygame.font.SysFont(None, 24)
    big = pygame.font.SysFont(None, 28)

    # Button
    btn_w, btn_h = 140, 36
    btn_rect = pygame.Rect(width_px - pad - btn_w, pad, btn_w, btn_h)

    last_red, last_blue = _step_full_turn(env, red_agent, blue_agent)

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    last_red, last_blue = _step_full_turn(env, red_agent, blue_agent)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(event.pos):
                    last_red, last_blue = _step_full_turn(env, red_agent, blue_agent)   
        # Draw 
        screen.fill((22, 22, 22))

        # Top info bar 

        red_workers = sum(1 for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        blue_workers = sum(1 for u in env.units if u.faction == "Blue" and u.unit_type == "worker")
        spawn_cost = env.config.worker_spawn_cost

        # Line 1 – turn info
        line1 = big.render(
            f"Turn {env.turn} | Current: {env.factions[env.current_player]}",
            True,
            (240, 240, 240),
        )
        screen.blit(line1, (pad, pad))

        # Line 2 – banks + workers
        line2 = font.render(
            f"Red: {env.bank['Red']} ({red_workers} workers)   |   "
            f"Blue: {env.bank['Blue']} ({blue_workers} workers)",
            True,
            (210, 210, 210),
        )
        screen.blit(line2, (pad, pad + 32))

        # Line 3 – turn mechanics
        line3 = font.render(
            f"Spawn cost: {spawn_cost}   |   "
            f"Actions left: {env.actions_left}   Attempts left: {env.attempts_left}",
            True,
            (210, 210, 210),
        )
        screen.blit(line3, (pad, pad + 58))

        # Last actions 
        lr = font.render(f"Red actions: {', '.join(last_red) if last_red else '-'}", True, (200, 200, 200))
        lb = font.render(f"Blue actions: {', '.join(last_blue) if last_blue else '-'}", True, (200, 200, 200))
        screen.blit(lr, (pad, pad + 84))
        screen.blit(lb, (pad, pad + 106))

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
