from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame


def _load_png(path: Path) -> pygame.Surface | None:
    try:
        return pygame.image.load(path.as_posix()).convert_alpha()
    except (pygame.error, FileNotFoundError):
        return None


def _load_first(paths: list[Path]) -> pygame.Surface | None:
    for path in paths:
        surface = _load_png(path)
        if surface is not None:
            return surface
    return None


def _load_variants(folder: Path) -> list[pygame.Surface]:
    surfaces: list[pygame.Surface] = []
    if not folder.exists():
        return surfaces
    for path in sorted(folder.glob("*.png")):
        surface = _load_png(path)
        if surface is not None:
            surfaces.append(surface)
    return surfaces


def _crop_sheet(sheet: pygame.Surface | None, col: int, row: int, tile: int = 16, margin: int = 1) -> pygame.Surface | None:
    if sheet is None:
        return None
    x = col * (tile + margin)
    y = row * (tile + margin)
    if x + tile > sheet.get_width() or y + tile > sheet.get_height():
        return None
    return sheet.subsurface(pygame.Rect(x, y, tile, tile)).copy()


@dataclass
class BoardAssets:
    terrain: dict[str, list[pygame.Surface]]
    objects: dict[str, pygame.Surface | None]
    ui: dict[str, pygame.Surface | None]
    characters: dict[str, list[pygame.Surface | None]]

    @classmethod
    def load(cls, root: Path) -> "BoardAssets":
        tile_root = root / "assets" / "tiles"
        object_root = root / "assets" / "objects"
        ui_root = root / "assets" / "ui"
        character_root = root / "assets" / "Characters"
        terrain = {
            "grass": _load_variants(tile_root / "Terrain" / "Grass"),
            "dirt": _load_variants(tile_root / "Terrain" / "Dirt"),
            "sand": _load_variants(tile_root / "Terrain" / "Sand"),
            "stone": _load_variants(tile_root / "Terrain" / "Stone"),
        }
        objects = {
            "base": _load_first(
                [
                    object_root / "castle_small.png",
                    object_root / "castle_open.png",
                    object_root / "castle_large.png",
                ]
            ),
            "storehouse": _load_first([object_root / "store.png", object_root / "mill_house.png", object_root / "cargo.png"]),
            "barracks": _load_first([object_root / "militaryTent.png", object_root / "militaryOutlook.png"]),
            "stable": _load_first([object_root / "horseTrough.png", object_root / "hay.png"]),
            "quarry": _load_first([object_root / "mine.png", object_root / "rockGrey_large.png"]),
            "archer_tower": _load_first([object_root / "tower.png", object_root / "turret_small.png"]),
            "ballista_tower": _load_first([object_root / "turret_large.png", object_root / "tower.png"]),
            "market": _load_first([object_root / "Market.png"]),
            "stone": _load_first([object_root / "rockGrey_large.png", object_root / "rockGrey_medium2.png"]),
            "horses": _load_first([object_root / "hay.png", object_root / "horseTrough.png"]),
            "resource": _load_first([object_root / "crystals1.png", object_root / "mine.png"]),
        }
        ui = {
            # generic fallback panel / inset / button (keep existing callers working)
            "panel": _load_first([ui_root / "panel_brown.png", ui_root / "panel_beige.png"]),
            "panel_inset": _load_first([ui_root / "panelInset_beigeLight.png", ui_root / "panelInset_beige.png"]),
            "button": _load_first([ui_root / "buttonLong_brown.png", ui_root / "buttonLong_beige.png"]),
            "close": _load_first([ui_root / "iconCross_grey.png", ui_root / "iconCross_brown.png"]),
            # faction-specific and themed panels
            "panel_brown": _load_first([ui_root / "panel_brown.png"]),
            "panel_beige": _load_first([ui_root / "panel_beige.png"]),
            "panel_beigeLight": _load_first([ui_root / "panel_beigeLight.png"]),
            "panel_blue": _load_first([ui_root / "panel_blue.png"]),
            "panelInset_beige": _load_first([ui_root / "panelInset_beige.png"]),
            "panelInset_beigeLight": _load_first([ui_root / "panelInset_beigeLight.png"]),
            "panelInset_blue": _load_first([ui_root / "panelInset_blue.png"]),
            "button_blue": _load_first([ui_root / "buttonLong_blue.png"]),
            "button_grey": _load_first([ui_root / "buttonLong_grey.png"]),
            "bar_back_left": _load_first([ui_root / "barBack_horizontalLeft.png"]),
            "bar_back_mid": _load_first([ui_root / "barBack_horizontalMid.png"]),
            "bar_back_right": _load_first([ui_root / "barBack_horizontalRight.png"]),
            "bar_red_left": _load_first([ui_root / "barRed_horizontalLeft.png"]),
            "bar_red_mid": _load_first([ui_root / "barRed_horizontalMid.png"]),
            "bar_red_right": _load_first([ui_root / "barRed_horizontalRight.png"]),
            "bar_yellow_left": _load_first([ui_root / "barYellow_horizontalLeft.png"]),
            "bar_yellow_mid": _load_first([ui_root / "barYellow_horizontalMid.png"]),
            "bar_yellow_right": _load_first([ui_root / "barYellow_horizontalRight.png"]),
            "bar_blue_left": _load_first([ui_root / "barBlue_horizontalLeft.png"]),
            "bar_blue_mid": _load_first([ui_root / "barBlue_horizontalMid.png"]),
            "bar_blue_right": _load_first([ui_root / "barBlue_horizontalRight.png"]),
        }
        char_sheet = _load_first(
            [
                character_root / "roguelikeChar_transparent.png",
                character_root / "roguelikeChar_magenta.png",
            ]
        )
        characters = {
            "worker": [
                _crop_sheet(char_sheet, 0, 6),
                _crop_sheet(char_sheet, 1, 6),
                _crop_sheet(char_sheet, 0, 10),
            ],
            "soldier": [
                _crop_sheet(char_sheet, 0, 7),
                _crop_sheet(char_sheet, 1, 9),
                _crop_sheet(char_sheet, 0, 11),
            ],
            "archer": [
                _crop_sheet(char_sheet, 1, 7),
                _crop_sheet(char_sheet, 0, 9),
                _crop_sheet(char_sheet, 1, 11),
            ],
            "horseman": [
                _crop_sheet(char_sheet, 1, 8),
                _crop_sheet(char_sheet, 0, 8),
                _crop_sheet(char_sheet, 1, 10),
            ],
        }
        return cls(terrain=terrain, objects=objects, ui=ui, characters=characters)

    def terrain_tile(self, kind: str, index: int) -> pygame.Surface | None:
        variants = self.terrain.get(kind) or self.terrain.get("grass") or []
        if not variants:
            return None
        return variants[index % len(variants)]

    def object_sprite(self, kind: str) -> pygame.Surface | None:
        return self.objects.get(kind) or self.objects.get("resource")

    def ui_sprite(self, kind: str) -> pygame.Surface | None:
        return self.ui.get(kind)

    def character_sprite(self, kind: str, tier: int = 0) -> pygame.Surface | None:
        variants = self.characters.get(kind) or []
        if not variants:
            return None
        index = max(0, min(tier, len(variants) - 1))
        sprite = variants[index]
        if sprite is not None:
            return sprite
        for fallback in reversed(variants):
            if fallback is not None:
                return fallback
        return None
