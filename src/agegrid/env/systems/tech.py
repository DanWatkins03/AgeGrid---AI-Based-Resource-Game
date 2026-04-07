from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechDefinition:
    cost: int
    requires: tuple[str, ...] = ()
    unlocks: tuple[str, ...] = ()


TECH_DEFS: dict[str, TechDefinition] = {
    "mining": TechDefinition(cost=40, unlocks=("storehouse",)),
    "bronze_working": TechDefinition(cost=70, requires=("mining",), unlocks=("barracks", "soldier")),
    "masonry": TechDefinition(cost=60, requires=("mining",), unlocks=("turret",)),
    "fletching": TechDefinition(cost=85, requires=("bronze_working",), unlocks=("archer",)),
}


def can_research(env, faction: str, tech_id: str) -> bool:
    tech = TECH_DEFS.get(tech_id)
    if tech is None:
        return False

    state = env.faction_state(faction)
    if tech_id in state.techs_unlocked:
        return False
    if state.resources < tech.cost:
        return False

    return all(req in state.techs_unlocked for req in tech.requires)


def research(env, faction: str, tech_id: str) -> bool:
    if not can_research(env, faction, tech_id):
        return False

    state = env.faction_state(faction)
    state.resources -= TECH_DEFS[tech_id].cost
    state.techs_unlocked.add(tech_id)
    state.tech_in_progress = None
    state.research_points = 0
    return True
