from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, MutableMapping


@dataclass
class FactionState:
    name: str
    resources: int = 0
    war_support: int = 100
    war_support_cap: int = 100
    war_support_cap_until_turn: int = 0
    techs_unlocked: set[str] = field(default_factory=set)
    tech_in_progress: str | None = None
    research_points: int = 0
    unit_ids: list[int] = field(default_factory=list)
    building_ids: list[int] = field(default_factory=list)


@dataclass
class PeaceConcessionState:
    payer: str
    receiver: str
    reparations_per_turn: int = 0
    reparations_until_turn: int = 0
    war_support_cap: int = 100
    war_support_cap_until_turn: int = 0


@dataclass
class RelationState:
    state: str = "peace"  # peace, war, truce
    since_turn: int = 0
    truce_until_turn: int = 0
    aggressor: str | None = None
    failed_aggressor: str | None = None
    pending_peace_by: str | None = None
    pending_indemnity: int = 0
    war_score: dict[str, int] = field(default_factory=dict)
    concessions: PeaceConcessionState | None = None


class BankView(MutableMapping[str, int]):
    """Backwards-compatible dict-like access to faction resources."""

    def __init__(self, faction_states: dict[str, FactionState]):
        self._faction_states = faction_states

    def __getitem__(self, key: str) -> int:
        return self._faction_states[key].resources

    def __setitem__(self, key: str, value: int) -> None:
        self._faction_states[key].resources = value

    def __delitem__(self, key: str) -> None:
        raise TypeError("Faction resources cannot be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter(self._faction_states)

    def __len__(self) -> int:
        return len(self._faction_states)
