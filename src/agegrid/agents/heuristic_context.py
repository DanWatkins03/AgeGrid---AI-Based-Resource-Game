from __future__ import annotations

from dataclasses import dataclass

from src.agegrid.env.actions import Action
from src.agegrid.env.systems import threat


@dataclass
class HeuristicContext:
    faction: str
    enemy_faction: str
    state: object
    legal: list[Action]
    diagnostics: object
    threat_map: threat.ThreatMap
    at_war: bool
    in_truce: bool
    defense_mode: bool
    base_under_siege: bool
    workers: list
    buildings: set[str]
    military: list
    emergency_targets: list
    contested_resources: list
    resource_contesters: list
    enemy_resource_workers: list
    nearby_cavalry: list
    rally_anchor: tuple[int, int] | None
    spawn_actions: list[Action]
    emergency_train: Action | None
    home_friendly_force: int
    home_enemy_force: int
    tech_deficit: int
    economy_gap: int
    military_gap: int
    behind_mode: bool
    recovery_mode: bool
    recovery_posture: str
    push_mode: bool
    siege_finish: bool
    collapse_mode: bool
    last_stand: bool
    desired_home_force: int
    useful_worker_slots: int
    rebuild_mode: bool
    declare_war_actions: list[Action]
    offer_peace_actions: list[Action]
    accept_peace_actions: list[Action]
