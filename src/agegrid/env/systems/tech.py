from __future__ import annotations

from dataclasses import dataclass


def _mods(**kwargs: int) -> tuple[tuple[str, int], ...]:
    return tuple(kwargs.items())


@dataclass(frozen=True)
class TechDefinition:
    cost: int
    turns: int
    requires: tuple[str, ...] = ()
    unlock_buildings: tuple[str, ...] = ()
    unlock_units: tuple[str, ...] = ()
    passive_modifiers: tuple[tuple[str, int], ...] = ()
    summary: str = ""
    column: int = 0
    row: int = 0

    def modifier(self, key: str) -> int:
        return dict(self.passive_modifiers).get(key, 0)


TECH_DEFS: dict[str, TechDefinition] = {
    "mining": TechDefinition(
        cost=35,
        turns=2,
        unlock_buildings=("storehouse",),
        summary="Storehouse, stone prospecting",
        column=1,
        row=0,
    ),
    "bronze": TechDefinition(
        cost=55,
        turns=3,
        requires=("mining",),
        unlock_buildings=("barracks",),
        unlock_units=("soldier",),
        summary="Barracks, Soldier",
        column=2,
        row=0,
    ),
    "masonry": TechDefinition(
        cost=50,
        turns=3,
        requires=("mining",),
        unlock_buildings=("quarry",),
        passive_modifiers=_mods(base_attack_bonus=1),
        summary="Quarry, base attack+",
        column=2,
        row=1,
    ),
    "animal_husbandry": TechDefinition(
        cost=45,
        turns=2,
        requires=("mining",),
        summary="Reveals horses, animal economy",
        column=2,
        row=2,
    ),
    "iron": TechDefinition(
        cost=65,
        turns=3,
        requires=("bronze",),
        passive_modifiers=_mods(soldier_hp_bonus=2, soldier_attack_bonus=1),
        summary="Soldier+",
        column=3,
        row=0,
    ),
    "fletching": TechDefinition(
        cost=55,
        turns=3,
        requires=("bronze",),
        unlock_units=("archer",),
        summary="Archer",
        column=3,
        row=1,
    ),
    "construction": TechDefinition(
        cost=60,
        turns=3,
        requires=("masonry",),
        unlock_buildings=("archer_tower",),
        passive_modifiers=_mods(base_hp_bonus=4, building_hp_bonus=2),
        summary="Archer Tower, structures+",
        column=3,
        row=2,
    ),
    "trade": TechDefinition(
        cost=55,
        turns=3,
        requires=("masonry",),
        passive_modifiers=_mods(storehouse_income_bonus=1, quarry_income_bonus=1),
        summary="Income+, economy branch",
        column=3,
        row=3,
    ),
    "horseback_riding": TechDefinition(
        cost=55,
        turns=3,
        requires=("animal_husbandry",),
        unlock_buildings=("stable",),
        unlock_units=("horseman",),
        summary="Stable, Horseman",
        column=3,
        row=4,
    ),
    "agriculture": TechDefinition(
        cost=50,
        turns=2,
        requires=("animal_husbandry",),
        passive_modifiers=_mods(worker_gather_bonus=1),
        summary="Gather+",
        column=3,
        row=5,
    ),
    "steel": TechDefinition(
        cost=75,
        turns=4,
        requires=("iron",),
        passive_modifiers=_mods(soldier_hp_bonus=2, soldier_attack_bonus=1),
        summary="Soldier++",
        column=4,
        row=0,
    ),
    "fortify": TechDefinition(
        cost=70,
        turns=4,
        requires=("iron",),
        passive_modifiers=_mods(base_attack_bonus=1, tower_damage_bonus=1, building_hp_bonus=4),
        summary="Base+, Towers+",
        column=4,
        row=1,
    ),
    "engineering": TechDefinition(
        cost=75,
        turns=4,
        requires=("fletching",),
        unlock_buildings=("ballista_tower", "siege_workshop"),
        passive_modifiers=_mods(archer_range_bonus=1),
        summary="Ballista Tower, Siege Workshop",
        column=4,
        row=2,
    ),
    "precision": TechDefinition(
        cost=65,
        turns=3,
        requires=("fletching",),
        passive_modifiers=_mods(archer_attack_bonus=1),
        summary="Archer damage+",
        column=4,
        row=3,
    ),
    "walls": TechDefinition(
        cost=65,
        turns=3,
        requires=("construction",),
        unlock_buildings=("wall",),
        passive_modifiers=_mods(base_hp_bonus=6),
        summary="Wall, Base HP+",
        column=4,
        row=4,
    ),
    "infrastructure": TechDefinition(
        cost=65,
        turns=3,
        requires=("construction",),
        passive_modifiers=_mods(building_cost_discount_pct=10),
        summary="Buildings cheaper",
        column=4,
        row=5,
    ),
    "markets": TechDefinition(
        cost=60,
        turns=3,
        requires=("trade",),
        unlock_buildings=("market",),
        summary="Market",
        column=4,
        row=6,
    ),
    "currency": TechDefinition(
        cost=70,
        turns=4,
        requires=("trade",),
        passive_modifiers=_mods(passive_income_multiplier_pct=20),
        summary="Passive income x1.20",
        column=4,
        row=7,
    ),
    "stirrups": TechDefinition(
        cost=65,
        turns=3,
        requires=("horseback_riding",),
        passive_modifiers=_mods(cavalry_hp_bonus=2, cavalry_attack_bonus=1),
        summary="Horseman+",
        column=4,
        row=8,
    ),
    "logistics": TechDefinition(
        cost=60,
        turns=3,
        requires=("horseback_riding",),
        passive_modifiers=_mods(cavalry_move_bonus=1),
        summary="Cavalry move+",
        column=4,
        row=9,
    ),
    "stronghold": TechDefinition(
        cost=85,
        turns=4,
        requires=("fortify", "construction"),
        unlock_buildings=("stronghold",),
        passive_modifiers=_mods(base_hp_bonus=8),
        summary="Stronghold, Base HP++",
        column=5,
        row=1,
    ),
    "heavy_cavalry": TechDefinition(
        cost=85,
        turns=4,
        requires=("iron", "stirrups"),
        unlock_units=("heavy_cavalry",),
        summary="Heavy Cavalry",
        column=5,
        row=3,
    ),
    "advanced_siege": TechDefinition(
        cost=90,
        turns=4,
        requires=("engineering", "steel"),
        unlock_units=("ballista",),
        passive_modifiers=_mods(tower_range_bonus=1),
        summary="Ballista, Siege range+",
        column=5,
        row=5,
    ),
    "war_economy": TechDefinition(
        cost=80,
        turns=4,
        requires=("trade", "iron"),
        passive_modifiers=_mods(military_cost_discount_pct=15, economy_income_bonus=1),
        summary="Military cheaper, income+",
        column=5,
        row=7,
    ),
}


TECH_TREE_ORDER: tuple[str, ...] = tuple(
    tech_id
    for tech_id, _definition in sorted(
        TECH_DEFS.items(),
        key=lambda item: (item[1].column, item[1].row, item[0]),
    )
)


def can_research(env, faction: str, tech_id: str) -> bool:
    definition = TECH_DEFS.get(tech_id)
    if definition is None:
        return False

    state = env.faction_state(faction)
    if tech_id in state.techs_unlocked:
        return False
    if state.tech_in_progress is not None:
        return False
    if state.resources < definition.cost:
        return False
    return all(req in state.techs_unlocked for req in definition.requires)


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
    apply_completion_effects(env, faction, tech_id)
    return tech_id


def apply_completion_effects(env, faction: str, tech_id: str) -> None:
    bonus = TECH_DEFS[tech_id].modifier("base_hp_bonus")
    if bonus <= 0:
        return
    env.bases[faction].hp = min(env.base_max_hp(faction), env.bases[faction].hp + bonus)


def passive_modifier_total(env, faction: str, key: str) -> int:
    total = 0
    for tech_id in env.faction_state(faction).techs_unlocked:
        definition = TECH_DEFS.get(tech_id)
        if definition is None:
            continue
        total += definition.modifier(key)
    return total


def unlocked_buildings(env, faction: str) -> set[str]:
    return {
        building_type
        for tech_id in env.faction_state(faction).techs_unlocked
        for building_type in TECH_DEFS[tech_id].unlock_buildings
    }


def unlocked_units(env, faction: str) -> set[str]:
    return {
        unit_type
        for tech_id in env.faction_state(faction).techs_unlocked
        for unit_type in TECH_DEFS[tech_id].unlock_units
    }


def unlock_items(tech_id: str) -> tuple[str, ...]:
    definition = TECH_DEFS[tech_id]
    modifier_labels = []
    for key, value in definition.passive_modifiers:
        modifier_labels.append(modifier_label(key, value))
    return (*definition.unlock_buildings, *definition.unlock_units, *modifier_labels)


def modifier_label(key: str, value: int) -> str:
    labels = {
        "worker_gather_bonus": f"Gather+{value}",
        "soldier_hp_bonus": f"Soldier HP+{value}",
        "soldier_attack_bonus": f"Soldier ATK+{value}",
        "archer_attack_bonus": f"Archer ATK+{value}",
        "archer_range_bonus": f"Archer RNG+{value}",
        "cavalry_hp_bonus": f"Cavalry HP+{value}",
        "cavalry_attack_bonus": f"Cavalry ATK+{value}",
        "cavalry_move_bonus": f"Cavalry MOVE+{value}",
        "base_attack_bonus": f"Base ATK+{value}",
        "tower_damage_bonus": f"Towers ATK+{value}",
        "tower_range_bonus": f"Towers RNG+{value}",
        "base_hp_bonus": f"Base HP+{value}",
        "building_hp_bonus": f"Building HP+{value}",
        "storehouse_income_bonus": f"Storehouse +{value}",
        "quarry_income_bonus": f"Quarry +{value}",
        "economy_income_bonus": f"Economy +{value}",
        "passive_income_multiplier_pct": f"Income x{100 + value}%",
        "building_cost_discount_pct": f"Build -{value}%",
        "military_cost_discount_pct": f"Military -{value}%",
    }
    return labels.get(key, f"{key}+{value}")
