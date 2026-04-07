from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from src.agegrid.agents.registry import AGENT_SPECS, create_agent
from src.agegrid.env.agegrid_env import AgeGridEnv


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


def _step_full_turn(env: AgeGridEnv, red_agent, blue_agent) -> tuple[FactionTurnInfo, FactionTurnInfo]:
    red_info, _ = _step_faction_with_trace(env, red_agent)
    env.step_end_turn()

    if env.winner() is not None:
        return red_info, FactionTurnInfo("Blue", ["turn_skipped:winner"], last_action="-", research="-", attacks="-")

    blue_info, _ = _step_faction_with_trace(env, blue_agent)
    env.step_end_turn()

    return red_info, blue_info


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
    card_h = 210
    card_y = 130
    red_card = pygame.Rect(40, card_y, card_w, card_h)
    blue_card = pygame.Rect(width_px - 40 - card_w, card_y, card_w, card_h)
    start_btn = pygame.Rect(width_px // 2 - 90, height_px - 90, 180, 46)

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

    tips = [
        "Controls after start: Space / Enter or click Next Turn.",
        "R resets to setup. P saves a debug snapshot. Esc closes the viewer.",
    ]
    _draw_text_block(screen, small_font, tips, 40, height_px - 88, (180, 186, 192), 18)
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
        f"Red techs: {', '.join(sorted(env.faction_state('Red').techs_unlocked)) or '-'}",
        f"Blue techs: {', '.join(sorted(env.faction_state('Blue').techs_unlocked)) or '-'}",
        f"Red buildings: {', '.join(sorted(f'{b.building_type}@{b.position}' for b in env.buildings if b.faction == 'Red' and b.hp > 0)) or '-'}",
        f"Blue buildings: {', '.join(sorted(f'{b.building_type}@{b.position}' for b in env.buildings if b.faction == 'Blue' and b.hp > 0)) or '-'}",
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
    top_bar = 220

    width_px = pad * 2 + env.config.width * tile
    height_px = pad * 2 + top_bar + env.config.height * tile

    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    big = pygame.font.SysFont(None, 30)
    small = pygame.font.SysFont(None, 19)

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
                    red_info, blue_info = _step_full_turn(env, red_agent, blue_agent)
                    turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(event.pos):
                    red_info, blue_info = _step_full_turn(env, red_agent, blue_agent)
                    turn_history.append(TurnSnapshot(env.turn, red_info, blue_info))

        screen.fill((22, 22, 22))

        red_workers = sum(1 for u in env.units if u.faction == "Red" and u.unit_type == "worker")
        blue_workers = sum(1 for u in env.units if u.faction == "Blue" and u.unit_type == "worker")
        red_military = sum(1 for u in env.units if u.faction == "Red" and u.attack_damage > 0)
        blue_military = sum(1 for u in env.units if u.faction == "Blue" and u.attack_damage > 0)
        red_techs = ", ".join(sorted(env.faction_state("Red").techs_unlocked)) or "-"
        blue_techs = ", ".join(sorted(env.faction_state("Blue").techs_unlocked)) or "-"
        red_buildings = ", ".join(sorted(b.building_type for b in env.buildings if b.faction == "Red" and b.hp > 0)) or "-"
        blue_buildings = ", ".join(sorted(b.building_type for b in env.buildings if b.faction == "Blue" and b.hp > 0)) or "-"
        winner = env.winner()

        line1 = big.render(
            f"Turn {env.turn} | Current: {env.factions[env.current_player]} | Winner: {winner or '-'}",
            True,
            (240, 240, 240),
        )
        screen.blit(line1, (pad, pad))

        line2 = font.render(
            f"Red {AGENT_SPECS[red_index].label}: bank {env.bank['Red']} | workers {red_workers} | military {red_military} | base HP {env.bases['Red'].hp}",
            True,
            (225, 205, 205),
        )
        line3 = font.render(
            f"Blue {AGENT_SPECS[blue_index].label}: bank {env.bank['Blue']} | workers {blue_workers} | military {blue_military} | base HP {env.bases['Blue'].hp}",
            True,
            (198, 214, 238),
        )
        screen.blit(line2, (pad, pad + 30))
        screen.blit(line3, (pad, pad + 54))

        _draw_text_block(screen, small, [f"Red techs: {red_techs}", f"Red buildings: {red_buildings}"], pad, pad + 84, (210, 210, 210), 18)
        _draw_text_block(
            screen,
            small,
            [f"Blue techs: {blue_techs}", f"Blue buildings: {blue_buildings}"],
            width_px // 2,
            pad + 84,
            (210, 210, 210),
            18,
        )

        red_lines = [
            f"Red last: {red_info.last_action}",
            f"Red research: {red_info.research}",
            f"Red attack: {red_info.attacks}",
            f"Red log: {', '.join(red_info.log) if red_info.log else '-'}",
        ]
        blue_lines = [
            f"Blue last: {blue_info.last_action}",
            f"Blue research: {blue_info.research}",
            f"Blue attack: {blue_info.attacks}",
            f"Blue log: {', '.join(blue_info.log) if blue_info.log else '-'}",
        ]
        _draw_text_block(screen, small, red_lines, pad, pad + 126, (205, 205, 205), 18)
        _draw_text_block(screen, small, blue_lines, width_px // 2, pad + 126, (205, 205, 205), 18)

        pygame.draw.rect(screen, (60, 60, 60), btn_rect, border_radius=8)
        pygame.draw.rect(screen, (120, 120, 120), btn_rect, width=2, border_radius=8)
        btn_text = font.render("Next Turn", True, (245, 245, 245))
        screen.blit(btn_text, (btn_rect.x + 24, btn_rect.y + 9))
        reset_text = small.render("Press R for setup", True, (180, 180, 180))
        screen.blit(reset_text, (btn_rect.x - 6, btn_rect.y + 42))
        snapshot_text = small.render("Press P for debug dump", True, (180, 180, 180))
        screen.blit(snapshot_text, (btn_rect.x - 18, btn_rect.y + 60))

        ox = pad
        oy = pad + top_bar

        for y in range(env.config.height):
            for x in range(env.config.width):
                rect = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
                pygame.draw.rect(screen, (35, 35, 35), rect)
                pygame.draw.rect(screen, (55, 55, 55), rect, width=1)

        for r in env.resources:
            if r.remaining <= 0:
                continue
            x, y = r.position
            cx = ox + x * tile + tile // 2
            cy = oy + y * tile + tile // 2
            pygame.draw.circle(screen, (60, 160, 90), (cx, cy), 10)
            amount = small.render(str(r.remaining), True, (200, 230, 200))
            screen.blit(amount, (cx - 10, cy + 10))

        for faction, base in env.bases.items():
            x, y = base.position
            rect = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
            color = (180, 60, 60) if faction == "Red" else (70, 90, 190)
            pygame.draw.rect(screen, color, rect)
            hp = small.render(str(base.hp), True, (245, 245, 245))
            screen.blit(hp, (rect.x + 14, rect.y + 14))

        for b in env.buildings:
            x, y = b.position
            rect = pygame.Rect(ox + x * tile + 8, oy + y * tile + 8, tile - 16, tile - 16)
            color = (205, 120, 90) if b.faction == "Red" else (110, 150, 220)
            pygame.draw.rect(screen, color, rect, border_radius=6)
            letter = small.render(b.building_type[0].upper(), True, (24, 24, 24))
            screen.blit(letter, (rect.x + 10, rect.y + 8))

        for u in env.units:
            x, y = u.position
            cx = ox + x * tile + tile // 2
            cy = oy + y * tile + tile // 2
            color = (240, 210, 120) if u.faction == "Red" else (180, 220, 255)
            radius = 12 if u.unit_type == "worker" else 14
            pygame.draw.circle(screen, color, (cx, cy), radius)
            label = "W" if u.unit_type == "worker" else u.unit_type[0].upper()
            unit_text = small.render(label, True, (18, 18, 18))
            screen.blit(unit_text, (cx - 5, cy - 8))

        pygame.display.flip()

    pygame.quit()
