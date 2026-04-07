from __future__ import annotations

from typing import Literal, TypeAlias

from src.agegrid.env.entities import Position

GatherAction: TypeAlias = tuple[Literal["gather"], int]
MoveTowardsAction: TypeAlias = tuple[Literal["move_towards"], int, Position]
SpawnWorkerAction: TypeAlias = tuple[Literal["spawn_worker"]]
TrainAction: TypeAlias = tuple[Literal["train"], str]
BuildAction: TypeAlias = tuple[Literal["build"], int, str, Position]
ResearchAction: TypeAlias = tuple[Literal["research"], str]
AttackAction: TypeAlias = tuple[Literal["attack"], int, int]
AttackBaseAction: TypeAlias = tuple[Literal["attack_base"], int, str]

Action: TypeAlias = (
    GatherAction
    | MoveTowardsAction
    | SpawnWorkerAction
    | TrainAction
    | BuildAction
    | ResearchAction
    | AttackAction
    | AttackBaseAction
)
