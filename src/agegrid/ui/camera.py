from __future__ import annotations

from dataclasses import dataclass, field
import math

import pygame


@dataclass
class CameraState:
    zoom: float = 1.0
    pan: list[float] = field(default_factory=lambda: [0.0, 0.0])
    panning: bool = False
    pan_anchor_mouse: tuple[int, int] = (0, 0)
    pan_anchor: tuple[float, float] = (0.0, 0.0)
    pan_drag_distance: float = 0.0


def board_origin_in_viewport(
    viewport: pygame.Rect,
    board_width: int,
    board_height: int,
    hex_size: int,
    pan: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, int]:
    origin_x = viewport.x + (viewport.width - board_width) / 2 + pan[0]
    origin_y = viewport.y + (viewport.height - board_height) / 2 + hex_size + pan[1]
    return round(origin_x), round(origin_y)


def clamp_camera_pan(
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


def reset_camera_state(state: CameraState, zoom: float) -> None:
    state.zoom = zoom
    state.pan = [0.0, 0.0]
    state.panning = False
    state.pan_anchor_mouse = (0, 0)
    state.pan_anchor = (0.0, 0.0)
    state.pan_drag_distance = 0.0


def begin_pan(state: CameraState, mouse_pos: tuple[int, int]) -> None:
    state.panning = True
    state.pan_anchor_mouse = mouse_pos
    state.pan_anchor = (state.pan[0], state.pan[1])
    state.pan_drag_distance = 0.0


def update_pan(
    state: CameraState,
    mouse_pos: tuple[int, int],
    viewport: pygame.Rect,
    board_width: int,
    board_height: int,
) -> None:
    dx = mouse_pos[0] - state.pan_anchor_mouse[0]
    dy = mouse_pos[1] - state.pan_anchor_mouse[1]
    state.pan[0], state.pan[1] = clamp_camera_pan(
        viewport,
        board_width,
        board_height,
        (state.pan_anchor[0] + dx, state.pan_anchor[1] + dy),
    )
    state.pan_drag_distance = max(state.pan_drag_distance, math.hypot(dx, dy))


def end_pan(state: CameraState) -> bool:
    state.panning = False
    return state.pan_drag_distance < 8


def apply_zoom(
    state: CameraState,
    mouse_pos: tuple[int, int],
    viewport: pygame.Rect,
    old_zoom: float,
    old_board_size: tuple[int, int],
    new_zoom: float,
    new_board_size: tuple[int, int],
    hex_size: int,
) -> None:
    old_origin = board_origin_in_viewport(
        viewport,
        old_board_size[0],
        old_board_size[1],
        hex_size,
        tuple(state.pan),
    )
    centered_origin = board_origin_in_viewport(
        viewport,
        new_board_size[0],
        new_board_size[1],
        hex_size,
    )
    scale = new_zoom / old_zoom if old_zoom else 1.0
    state.pan[0] = mouse_pos[0] - scale * (mouse_pos[0] - old_origin[0]) - centered_origin[0]
    state.pan[1] = mouse_pos[1] - scale * (mouse_pos[1] - old_origin[1]) - centered_origin[1]
    state.pan[0], state.pan[1] = clamp_camera_pan(
        viewport,
        new_board_size[0],
        new_board_size[1],
        tuple(state.pan),
    )
    state.zoom = new_zoom
