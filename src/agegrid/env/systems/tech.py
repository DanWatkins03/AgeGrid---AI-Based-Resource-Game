from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechDefinition:
    cost: int
    turns: int
    requires: tuple[str, ...] = ()
    unlocks: tuple[str, ...] = ()


TECH_DEFS: dict[str, TechDefinition] = {
    "mining": TechDefinition(cost=35, turns=2, unlocks=("storehouse",)),
    "bronze_working": TechDefinition(cost=55, turns=3, requires=("mining",), unlocks=("barracks", "soldier")),
    "masonry": TechDefinition(cost=45, turns=2, requires=("mining",), unlocks=("quarry", "turret")),
    "horsemanship": TechDefinition(cost=45, turns=2, requires=("bronze_working",), unlocks=("stable", "horseman")),
    "fletching": TechDefinition(cost=55, turns=3, requires=("bronze_working",), unlocks=("archer",)),
}


def can_research(env, faction: str, tech_id: str) -> bool:
    tech = TECH_DEFS.get(tech_id)
    if tech is None:
        return False

    state = env.faction_state(faction)
    if tech_id in state.techs_unlocked:
        return False
    if state.tech_in_progress is not None:
        return False
    if state.resources < tech.cost:
        return False

    return all(req in state.techs_unlocked for req in tech.requires)


def research(env, faction: str, tech_id: str) -> bool:
    if not can_research(env, faction, tech_id):
        return False

    state = env.faction_state(faction)
    state.resources -= TECH_DEFS[tech_id].cost
    state.tech_in_progress = tech_id
    state.research_points = 0
    return True


def research_turns_remaining(env, faction: str) -> int | None:
    state = env.faction_state(faction)
    if state.tech_in_progress is None:
        return None
    definition = TECH_DEFS[state.tech_in_progress]
    return max(0, definition.turns - state.research_points)


def progress_research(env, faction: str) -> str | None:
    state = env.faction_state(faction)
    if state.tech_in_progress is None:
        return None

    tech_id = state.tech_in_progress
    state.research_points += 1
    definition = TECH_DEFS[tech_id]
    if state.research_points < definition.turns:
        return None

    state.techs_unlocked.add(tech_id)
    state.tech_in_progress = None
    state.research_points = 0
    return tech_id
