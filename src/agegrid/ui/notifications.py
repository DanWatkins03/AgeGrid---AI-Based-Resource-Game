"""Player-facing notification generation for AgeGrid.

Returns state-derived notifications — no history tracking required.
These update naturally each frame and disappear when conditions resolve.
"""
from __future__ import annotations

from src.agegrid.env.agegrid_env import AgeGridEnv

# Colour palette shared with pygame_viewer.py
_CRITICAL = (196, 68,  60)   # red    — imminent threat / collapse
_WARN     = (170, 112, 48)   # amber  — caution / active penalty
_INFO     = (72,  106, 152)  # blue   — neutral information
_GOOD     = (72,  120, 70)   # green  — opportunity / positive state

_Notification = tuple[str, tuple[int, int, int]]


def player_notifications(
    env: AgeGridEnv,
    faction: str,
) -> list[_Notification]:
    """Return priority-ordered (message, colour) notifications for *faction*.

    Only current-state conditions are checked — no per-turn bookkeeping
    needed.  Capped at 4 entries to avoid overflowing the event panel.
    """
    enemy = next((f for f in env.factions if f != faction), None)
    if enemy is None:
        return []

    notes: list[_Notification] = []
    relation = env.relation_state(faction, enemy)
    f_state  = env.faction_state(faction)
    cfg      = env.config

    # ── War-state notifications ────────────────────────────────────────────
    if relation.state == "war":

        # War support critically low
        if f_state.war_support <= 10:
            notes.append((
                f"! War support critical ({f_state.war_support}%) — a lost war hits very hard",
                _CRITICAL,
            ))
        elif f_state.war_support <= 30:
            notes.append((
                f"! War support low ({f_state.war_support}%) — consider seeking peace",
                _WARN,
            ))

        # Support cap from previous concessions
        if (
            f_state.war_support_cap < 100
            and f_state.war_support_cap_until_turn > env.turn
        ):
            tl = f_state.war_support_cap_until_turn - env.turn
            notes.append((
                f"War support capped at {f_state.war_support_cap}% "
                f"({tl} turn{'s' if tl != 1 else ''} left)",
                _WARN,
            ))

        # Enemy offered peace
        if relation.pending_peace_by == enemy:
            ind = relation.pending_indemnity
            if ind > 0:
                notes.append((
                    f"{enemy} offered peace — you would receive {ind} gold",
                    _INFO,
                ))
            else:
                notes.append((f"{enemy} has offered peace terms", _INFO))

        # War upkeep reminder (first few turns only, to avoid permanent noise)
        turns_at_war = max(0, env.turn - relation.since_turn)
        upkeep = cfg.war_upkeep_per_turn
        if relation.aggressor == faction:
            upkeep += cfg.war_upkeep_aggressor_bonus
        if turns_at_war <= 3 and upkeep > 0:
            role = "aggressor" if relation.aggressor == faction else "defender"
            notes.append((
                f"War upkeep: -{upkeep} gold/turn ({role})",
                _INFO,
            ))

    # ── Truce-state notifications ──────────────────────────────────────────
    elif relation.state == "truce":
        truce_left = max(0, relation.truce_until_turn - env.turn)

        if truce_left <= 3:
            notes.append((
                f"! Truce with {enemy} expires in "
                f"{truce_left} turn{'s' if truce_left != 1 else ''}",
                _WARN,
            ))

        # Concession reparations still running
        conc = relation.concessions
        if (
            conc is not None
            and conc.payer == faction
            and env.turn < conc.reparations_until_turn
        ):
            rep_left = max(0, conc.reparations_until_turn - env.turn)
            notes.append((
                f"Paying reparations: -{conc.reparations_per_turn}/turn "
                f"({rep_left} turn{'s' if rep_left != 1 else ''} left)",
                _WARN,
            ))

        # Support recovery blocked for failed aggressor
        if (
            relation.failed_aggressor == faction
            and cfg.failed_aggressor_truce_support_recovery_per_turn == 0
        ):
            notes.append((
                f"Failed war: support recovery blocked for {truce_left} more turn"
                f"{'s' if truce_left != 1 else ''}",
                _WARN,
            ))

        # Support cap still active
        if (
            f_state.war_support_cap < 100
            and f_state.war_support_cap_until_turn > env.turn
        ):
            cap_left = f_state.war_support_cap_until_turn - env.turn
            notes.append((
                f"War support capped at {f_state.war_support_cap}% "
                f"({cap_left} turn{'s' if cap_left != 1 else ''} left)",
                _WARN,
            ))

    # ── Enemy collapse indicator ───────────────────────────────────────────
    if cfg.collapse_enabled:
        e_has_mil = any(
            u for u in env.units
            if u.faction == enemy and u.unit_type != "worker"
        )
        e_has_wkr = any(
            u for u in env.units
            if u.faction == enemy and u.unit_type == "worker"
        )
        e_bank = env.bank[enemy]
        if not e_has_mil and not e_has_wkr and e_bank < cfg.worker_spawn_cost:
            notes.append((
                f"! {enemy} is on the verge of collapse — finish them!",
                _GOOD,
            ))
        elif not e_has_mil and e_bank < cfg.worker_spawn_cost + 15:
            notes.append((
                f"{enemy} has no military and limited gold — consider pressing now",
                _INFO,
            ))

    return notes[:4]
