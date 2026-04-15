from __future__ import annotations

from collections import deque
from math import sqrt

Position = tuple[int, int]


_EVEN_ROW_DELTAS: tuple[Position, ...] = (
    (1, 0),
    (-1, 0),
    (0, -1),
    (-1, -1),
    (0, 1),
    (-1, 1),
)

_ODD_ROW_DELTAS: tuple[Position, ...] = (
    (1, 0),
    (-1, 0),
    (1, -1),
    (0, -1),
    (1, 1),
    (0, 1),
)

HEX_DIRECTION_LABELS = ("east", "west", "northeast", "northwest", "southeast", "southwest")


def neighbor_deltas(row: int) -> tuple[Position, ...]:
    return _ODD_ROW_DELTAS if row % 2 else _EVEN_ROW_DELTAS


def direction_map(row: int) -> dict[str, Position]:
    deltas = neighbor_deltas(row)
    mapping = dict(zip(HEX_DIRECTION_LABELS, deltas, strict=True))
    mapping["right"] = mapping["east"]
    mapping["left"] = mapping["west"]
    mapping["up_right"] = mapping["northeast"]
    mapping["up_left"] = mapping["northwest"]
    mapping["down_right"] = mapping["southeast"]
    mapping["down_left"] = mapping["southwest"]
    return mapping


def neighbors(pos: Position) -> list[Position]:
    col, row = pos
    return [(col + dc, row + dr) for dc, dr in neighbor_deltas(row)]


def oddr_to_cube(pos: Position) -> tuple[int, int, int]:
    col, row = pos
    x = col - (row - (row & 1)) // 2
    z = row
    y = -x - z
    return x, y, z


def distance(a: Position, b: Position) -> int:
    ax, ay, az = oddr_to_cube(a)
    bx, by, bz = oddr_to_cube(b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def positions_at_distance(center: Position, radius: int, width: int, height: int) -> list[Position]:
    if radius <= 0:
        return [center]
    result: list[Position] = []
    for row in range(height):
        for col in range(width):
            pos = (col, row)
            if distance(center, pos) == radius:
                result.append(pos)
    return result


def positions_within(center: Position, radius: int, width: int, height: int) -> list[Position]:
    result: list[Position] = []
    for row in range(height):
        for col in range(width):
            pos = (col, row)
            if distance(center, pos) <= radius:
                result.append(pos)
    return result


def hex_to_pixel(col: int, row: int, size: float) -> tuple[float, float]:
    x = size * sqrt(3) * (col + 0.5 * (row & 1))
    y = size * 1.5 * row
    return x, y


def hex_polygon_points(x: float, y: float, size: float) -> list[tuple[int, int]]:
    half_w = sqrt(3) * size / 2
    quarter_h = size / 2
    return [
        (round(x), round(y + quarter_h)),
        (round(x), round(y - quarter_h)),
        (round(x + half_w), round(y - size)),
        (round(x + 2 * half_w), round(y - quarter_h)),
        (round(x + 2 * half_w), round(y + quarter_h)),
        (round(x + half_w), round(y + size)),
    ]


def nearest_hex(point: tuple[int, int], width: int, height: int, size: float, origin: tuple[float, float]) -> Position | None:
    px, py = point
    ox, oy = origin
    best: Position | None = None
    best_distance = float("inf")
    for row in range(height):
        for col in range(width):
            left, top = hex_to_pixel(col, row, size)
            center_x = ox + left + sqrt(3) * size / 2
            center_y = oy + top
            score = (center_x - px) ** 2 + (center_y - py) ** 2
            if score < best_distance:
                best = (col, row)
                best_distance = score
    return best


def connected_step_toward(
    start: Position,
    target: Position,
    width: int,
    height: int,
    blocked: set[Position],
) -> Position | None:
    if start == target:
        return None

    queue: deque[Position] = deque([start])
    came_from: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == target:
            break
        for nxt in neighbors(current):
            if nxt in came_from:
                continue
            col, row = nxt
            if not (0 <= col < width and 0 <= row < height):
                continue
            if nxt in blocked and nxt != target:
                continue
            came_from[nxt] = current
            queue.append(nxt)

    if target not in came_from:
        return None

    step = target
    while came_from[step] != start:
        parent = came_from[step]
        if parent is None:
            return None
        step = parent
    return step
