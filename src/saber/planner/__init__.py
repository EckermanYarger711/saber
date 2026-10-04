"""Budgeted information-gain acquisition.

Ref: Sec. 2.5 (the decision currency and the receding-horizon objective),
Eq. (4) and Eq. (7), and Algorithm 2 (BIG).
"""

from saber.planner.big import (
    ActionSet,
    BigPlanner,
    BigStep,
    BigTrace,
    candidates_for,
    plan_session,
    reference_schedule,
)
from saber.planner.currency import (
    currency,
    currency_per_unit_damage,
    currency_vector,
    rank_by_currency,
)
from saber.planner.info_gain import (
    EncodingTable,
    design_vector,
    encode_actions,
    observation_noise,
)
from saber.planner.lagrangian import DualState, charge, dual_update, lagrangian_score
from saber.planner.policies import (
    POLICY_NAMES,
    Candidates,
    DecisionContext,
    Policy,
    PolicyDecision,
    build_policy,
)
from saber.planner.rollout import (
    SessionOutcome,
    action_sequence_indices,
    logged_observer,
    run_policy_session,
)

__all__ = [
    "POLICY_NAMES",
    "ActionSet",
    "BigPlanner",
    "BigStep",
    "BigTrace",
    "Candidates",
    "DecisionContext",
    "DualState",
    "EncodingTable",
    "Policy",
    "PolicyDecision",
    "SessionOutcome",
    "action_sequence_indices",
    "build_policy",
    "candidates_for",
    "charge",
    "currency",
    "currency_per_unit_damage",
    "currency_vector",
    "design_vector",
    "dual_update",
    "encode_actions",
    "lagrangian_score",
    "logged_observer",
    "observation_noise",
    "plan_session",
    "rank_by_currency",
    "reference_schedule",
    "run_policy_session",
]
