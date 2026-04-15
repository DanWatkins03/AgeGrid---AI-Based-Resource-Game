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
DeclareWarAction: TypeAlias = tuple[Literal["declare_war"], str]
OfferPeaceAction: TypeAlias = tuple[Literal["offer_peace"], str, int]
AcceptPeaceAction: TypeAlias = tuple[Literal["accept_peace"], str]

Action: TypeAlias = (
    GatherAction
    | MoveTowardsAction
    | SpawnWorkerAction
    | TrainAction
    | BuildAction
    | ResearchAction
    | AttackAction
    | AttackBaseAction
    | DeclareWarAction
    | OfferPeaceAction
    | AcceptPeaceAction
)
