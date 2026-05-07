from __future__ import annotations

from dataclasses import dataclass

from src.agegrid.env.state import PeaceConcessionState, RelationState


@dataclass(frozen=True)
class PeaceConcessionPreview:
    payer: str
    receiver: str
    reparations_per_turn: int = 0
    reparations_turns: int = 0
    war_support_cap: int = 100
    war_support_cap_turns: int = 0
    reason: str = "balanced_peace"

    @property
    def has_concessions(self) -> bool:
        return self.reparations_per_turn > 0 or self.war_support_cap < 100


def refresh_relation_state(env, relation: RelationState) -> RelationState:
    if relation.state == "truce" and env.turn >= relation.truce_until_turn:
        relation.state = "peace"
        relation.since_turn = env.turn
        relation.aggressor = None
        relation.failed_aggressor = None
        relation.pending_peace_by = None
        relation.pending_indemnity = 0
        relation.war_score.clear()
        relation.concessions = None
    return relation


def clear_pending_peace(relation: RelationState) -> None:
    relation.pending_peace_by = None
    relation.pending_indemnity = 0


def ensure_war_score(env, relation: RelationState) -> None:
    for faction in env.factions:
        relation.war_score.setdefault(faction, 0)


def change_war_support(env, faction: str, delta: int) -> int:
    state = env.faction_state(faction)
    if state.war_support_cap_until_turn <= env.turn:
        state.war_support_cap = 100
    before = state.war_support
    state.war_support = max(0, min(state.war_support_cap, state.war_support + delta))
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
    relation.failed_aggressor = None
    relation.concessions = None
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


def recommended_peace_indemnity(env, payer: str, receiver: str) -> int:
    bank = env.bank[payer]
    if bank <= 0:
        return 0

    relation = env.relation_state(payer, receiver)
    ensure_war_score(env, relation)
    payer_score = relation.war_score.get(payer, 0)
    receiver_score = relation.war_score.get(receiver, 0)
    score_deficit = max(0, receiver_score - payer_score)
    war_turns = max(0, env.turn - relation.since_turn)
    game_turn_pressure = env.turn // max(1, env.config.peace_indemnity_game_turn_divisor)

    payer_income = env._passive_income_for(payer)
    receiver_income = env._passive_income_for(receiver)
    income_gap = max(0, receiver_income - payer_income)

    # Base damage is a direct "you nearly lost the game" signal. War score
    # also includes base damage, but keeping this explicit makes late sieges
    # demand a noticeably stronger concession than a few skirmish trades.
    payer_base_damage = max(0, env.config.base_hp - env.bases[payer].hp)

    # The base value keeps early peace legible. Outcome, economic position,
    # war duration, and current turn then scale the ask so late-game surrender
    # is materially different from a short border incident.
    indemnity = env.config.peace_indemnity_base
    indemnity += score_deficit * env.config.peace_indemnity_score_multiplier
    indemnity += payer_base_damage * env.config.peace_indemnity_base_damage_multiplier
    indemnity += income_gap * env.config.peace_indemnity_income_gap_turns
    indemnity += war_turns // max(1, env.config.peace_indemnity_war_turn_divisor)
    indemnity += game_turn_pressure

    max_indemnity = max(1, (bank * env.config.peace_indemnity_max_bank_pct) // 100)
    return max(0, min(indemnity, max_indemnity))


def preview_peace_concessions(env, payer: str, receiver: str) -> PeaceConcessionPreview:
    relation = env.relation_state(payer, receiver)
    ensure_war_score(env, relation)
    score_deficit = max(0, relation.war_score.get(receiver, 0) - relation.war_score.get(payer, 0))
    base_damage = max(0, env.config.base_hp - env.bases[payer].hp)
    if (
        score_deficit < env.config.concession_score_deficit_threshold
        and base_damage < env.config.concession_base_damage_threshold
    ):
        return PeaceConcessionPreview(payer=payer, receiver=receiver)

    payer_income = env._passive_income_for(payer)
    reparations_per_turn = 0
    if payer_income > 0:
        reparations_per_turn = max(
            env.config.concession_min_reparations_per_turn,
            (payer_income * env.config.concession_reparations_income_pct) // 100,
        )

    # Concessions are deliberately modest: reparations skim post-war income,
    # while the support cap prevents the loser from immediately selling the
    # next war to their population. Both are short-lived truce consequences.
    reason_parts = []
    if score_deficit:
        reason_parts.append(f"score deficit {score_deficit}")
    if base_damage:
        reason_parts.append(f"base damage {base_damage}")
    return PeaceConcessionPreview(
        payer=payer,
        receiver=receiver,
        reparations_per_turn=reparations_per_turn,
        reparations_turns=env.config.concession_reparations_turns if reparations_per_turn > 0 else 0,
        war_support_cap=env.config.concession_war_support_cap,
        war_support_cap_turns=env.config.concession_war_support_cap_turns,
        reason=", ".join(reason_parts) or "lost_war",
    )


def _apply_peace_concessions(env, relation: RelationState, preview: PeaceConcessionPreview) -> None:
    if not preview.has_concessions:
        relation.concessions = None
        return
    relation.concessions = PeaceConcessionState(
        payer=preview.payer,
        receiver=preview.receiver,
        reparations_per_turn=preview.reparations_per_turn,
        reparations_until_turn=env.turn + preview.reparations_turns,
        war_support_cap=preview.war_support_cap,
        war_support_cap_until_turn=env.turn + preview.war_support_cap_turns,
    )
    payer_state = env.faction_state(preview.payer)
    payer_state.war_support_cap = min(payer_state.war_support_cap, preview.war_support_cap)
    payer_state.war_support_cap_until_turn = max(
        payer_state.war_support_cap_until_turn,
        env.turn + preview.war_support_cap_turns,
    )
    if payer_state.war_support > payer_state.war_support_cap:
        payer_state.war_support = payer_state.war_support_cap
    env._record_event(
        f"{preview.payer} accepted concessions to {preview.receiver}: "
        f"reparations {preview.reparations_per_turn}/turn for {preview.reparations_turns} turns, "
        f"war support cap {preview.war_support_cap} for {preview.war_support_cap_turns} turns "
        f"({preview.reason})"
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
    concessions = preview_peace_concessions(env, payer, receiver)
    aggressor = relation.aggressor
    failed_aggressor = None
    if aggressor is not None:
        defender = receiver if aggressor == payer else payer
        aggressor_score = relation.war_score.get(aggressor, 0)
        defender_score = relation.war_score.get(defender, 0)
        if aggressor_score <= defender_score:
            # A war that fails to outscore the defender should meaningfully
            # remove repeat aggression as an immediate option for the initiator.
            failed_aggressor = aggressor
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
    relation.failed_aggressor = failed_aggressor
    _apply_peace_concessions(env, relation, concessions)
    clear_pending_peace(relation)
    relation.war_score.clear()
    return indemnity


def apply_turn_costs(env, faction: str) -> None:
    for enemy in env._enemy_factions(faction):
        relation = env.relation_state(faction, enemy)
        if relation.state != "war":
            concessions = relation.concessions
            if (
                relation.state == "truce"
                and concessions is not None
                and concessions.payer == faction
                and env.turn < concessions.reparations_until_turn
                and concessions.reparations_per_turn > 0
            ):
                payment = min(concessions.reparations_per_turn, env.bank[faction])
                if payment > 0:
                    env.bank[faction] -= payment
                    env.bank[concessions.receiver] += payment
                    env._record_event(f"{faction} paid {payment} reparations to {concessions.receiver}")
            recovery_per_turn = env.config.peace_support_recovery_per_turn
            if relation.state == "truce" and relation.failed_aggressor == faction:
                # During truce, the failed aggressor's public appetite for war
                # should stay suppressed; the defender may still recover normally.
                recovery_per_turn = env.config.failed_aggressor_truce_support_recovery_per_turn
            support_delta = change_war_support(env, faction, recovery_per_turn)
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
