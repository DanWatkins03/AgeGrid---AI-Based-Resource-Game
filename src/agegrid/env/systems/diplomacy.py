from __future__ import annotations

from src.agegrid.env.state import RelationState


def refresh_relation_state(env, relation: RelationState) -> RelationState:
    if relation.state == "truce" and env.turn >= relation.truce_until_turn:
        relation.state = "peace"
        relation.since_turn = env.turn
        relation.aggressor = None
        relation.pending_peace_by = None
        relation.pending_indemnity = 0
        relation.war_score.clear()
    return relation


def clear_pending_peace(relation: RelationState) -> None:
    relation.pending_peace_by = None
    relation.pending_indemnity = 0


def ensure_war_score(env, relation: RelationState) -> None:
    for faction in env.factions:
        relation.war_score.setdefault(faction, 0)


def change_war_support(env, faction: str, delta: int) -> int:
    state = env.faction_state(faction)
    before = state.war_support
    state.war_support = max(0, min(100, state.war_support + delta))
    return state.war_support - before


def war_score(env, faction: str, enemy_faction: str) -> int:
    relation = env.relation_state(faction, enemy_faction)
    ensure_war_score(env, relation)
    return relation.war_score[faction]


def award_war_score(env, faction: str, enemy_faction: str, amount: int) -> None:
    if amount <= 0:
        return
    relation = env.relation_state(faction, enemy_faction)
    ensure_war_score(env, relation)
    relation.war_score[faction] += amount


def record_unit_casualty(env, attacker_faction: str, defender_faction: str, unit_type: str) -> None:
    if not env.at_war(attacker_faction, defender_faction):
        return
    amount = env.config.war_score_worker_kill if unit_type == "worker" else env.config.war_score_unit_kill
    award_war_score(env, attacker_faction, defender_faction, amount)


def record_base_damage(env, attacker_faction: str, defender_faction: str, damage: int) -> None:
    if not env.at_war(attacker_faction, defender_faction):
        return
    award_war_score(env, attacker_faction, defender_faction, damage * env.config.war_score_base_damage)


def war_upkeep(env, faction: str, relation: RelationState) -> int:
    upkeep = env.config.war_upkeep_per_turn
    if relation.aggressor == faction:
        upkeep += env.config.war_upkeep_aggressor_bonus
    return upkeep


def can_declare_war(env, faction: str, target_faction: str) -> bool:
    relation = env.relation_state(faction, target_faction)
    return (
        relation.state != "war"
        and env.turn >= relation.truce_until_turn
        and env.bank[faction] >= env.config.war_declaration_cost
        and env.faction_state(faction).war_support >= env.config.war_support_to_declare_min
    )


def declare_war(env, faction: str, target_faction: str) -> bool:
    if not can_declare_war(env, faction, target_faction):
        return False
    relation = env.relation_state(faction, target_faction)
    env.bank[faction] -= env.config.war_declaration_cost
    change_war_support(env, faction, -env.config.war_declaration_support_penalty)
    relation.state = "war"
    relation.since_turn = env.turn
    relation.aggressor = faction
    relation.war_score = {name: 0 for name in env.factions}
    clear_pending_peace(relation)
    return True


def can_offer_peace(env, faction: str, target_faction: str) -> bool:
    relation = env.relation_state(faction, target_faction)
    return (
        relation.state == "war"
        and env.turn - relation.since_turn >= env.config.peace_offer_min_turns
        and relation.pending_peace_by is None
    )


def offer_peace(env, faction: str, target_faction: str, indemnity: int) -> bool:
    if not can_offer_peace(env, faction, target_faction):
        return False
    relation = env.relation_state(faction, target_faction)
    relation.pending_peace_by = faction
    relation.pending_indemnity = max(0, indemnity)
    return True


def can_accept_peace(env, faction: str, target_faction: str) -> bool:
    relation = env.relation_state(faction, target_faction)
    return relation.state == "war" and relation.pending_peace_by == target_faction


def accept_peace(env, faction: str, target_faction: str) -> int | None:
    if not can_accept_peace(env, faction, target_faction):
        return None
    relation = env.relation_state(faction, target_faction)
    payer = relation.pending_peace_by
    if payer is None:
        return None
    receiver = target_faction if payer == faction else faction
    indemnity = min(relation.pending_indemnity, env.bank[payer])
    env.bank[payer] -= indemnity
    env.bank[receiver] += indemnity
    ensure_war_score(env, relation)
    aggressor = relation.aggressor
    if aggressor is not None:
        defender = receiver if aggressor == payer else payer
        aggressor_score = relation.war_score.get(aggressor, 0)
        defender_score = relation.war_score.get(defender, 0)
        if aggressor_score <= defender_score:
            change_war_support(env, aggressor, -env.config.failed_war_support_penalty)
            change_war_support(env, defender, env.config.successful_war_support_bonus)
        else:
            change_war_support(env, aggressor, env.config.successful_war_support_bonus)
    change_war_support(env, faction, env.config.peace_relief_support_bonus)
    change_war_support(env, target_faction, env.config.peace_relief_support_bonus)
    relation.state = "truce"
    relation.since_turn = env.turn
    relation.truce_until_turn = env.turn + env.config.truce_turns
    relation.aggressor = None
    clear_pending_peace(relation)
    relation.war_score.clear()
    return indemnity


def apply_turn_costs(env, faction: str) -> None:
    for enemy in env._enemy_factions(faction):
        relation = env.relation_state(faction, enemy)
        if relation.state != "war":
            support_delta = change_war_support(env, faction, env.config.peace_support_recovery_per_turn)
            if support_delta > 0 and env.faction_state(faction).war_support <= env.config.war_support_to_declare_min:
                env._record_event(f"{faction} war support recovered to {env.faction_state(faction).war_support}")
            continue
        upkeep = min(war_upkeep(env, faction, relation), env.bank[faction])
        if upkeep > 0:
            env.bank[faction] -= upkeep
            env._record_event(f"{faction} paid {upkeep} war upkeep against {enemy}")
        support_delta = change_war_support(env, faction, -env.config.war_support_drain_per_turn)
        if support_delta < 0:
            env._record_event(f"{faction} war support fell to {env.faction_state(faction).war_support}")
