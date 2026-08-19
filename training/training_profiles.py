"""Named, reproducible training profiles for the MARL experiments.

The historical ``baseline`` profile reproduces the published Phase 25--33
experiments.  The ``improved`` profile is a validation-motivated optimisation
of the learner, not a change to the environment or reward.  Keeping the two
profiles separate prevents a tuning run from silently overwriting the
published models.  ``improved_v2`` changes exactly one representation choice
from ``improved``: it restores the one-hot agent identity so a shared policy
can learn stable asymmetric roles under the coordinated channel constraint.
``refined`` defines a corrected future low-learning-rate warm-start from
``extended``. The historical 2026-08-20 artifacts with that label are explicitly
invalidated because their restored optimizer remained at the old rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    recommended_t_max: int
    learning_rate: float
    gamma: float
    batch_size: int
    buffer_size: int
    hidden_dim: int
    epsilon_finish: float
    epsilon_anneal_fraction: float
    target_update_interval_or_tau: float
    updates_per_episode: int
    checkpoint_count: int
    include_agent_id: bool
    mask_neighbor_signal: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PROFILES: Dict[str, TrainingProfile] = {
    "baseline": TrainingProfile(
        name="baseline",
        description="Exact historical Phase 25--33 training settings.",
        recommended_t_max=60_000,
        learning_rate=5e-4,
        gamma=0.99,
        batch_size=16,
        buffer_size=5_000,
        hidden_dim=64,
        epsilon_finish=0.05,
        epsilon_anneal_fraction=0.60,
        target_update_interval_or_tau=200,
        updates_per_episode=1,
        checkpoint_count=4,
        include_agent_id=True,
        mask_neighbor_signal=False,
    ),
    "improved": TrainingProfile(
        name="improved",
        description=(
            "Validation-driven homogeneous shared policy with the noisy lagged "
            "neighbor feature masked, more replay updates, soft target updates, "
            "and denser checkpoint selection."
        ),
        recommended_t_max=180_000,
        learning_rate=3e-4,
        gamma=0.95,
        batch_size=32,
        buffer_size=1_000,
        hidden_dim=64,
        epsilon_finish=0.02,
        epsilon_anneal_fraction=0.70,
        target_update_interval_or_tau=0.01,
        updates_per_episode=2,
        checkpoint_count=10,
        include_agent_id=False,
        mask_neighbor_signal=True,
    ),
    "improved_v2": TrainingProfile(
        name="improved_v2",
        description=(
            "One-factor coordinated-policy candidate based on improved: restore "
            "agent identity for role separation while retaining the masked lagged "
            "neighbor feature and all learner hyperparameters."
        ),
        recommended_t_max=180_000,
        learning_rate=3e-4,
        gamma=0.95,
        batch_size=32,
        buffer_size=1_000,
        hidden_dim=64,
        epsilon_finish=0.02,
        epsilon_anneal_fraction=0.70,
        target_update_interval_or_tau=0.01,
        updates_per_episode=2,
        checkpoint_count=10,
        include_agent_id=True,
        mask_neighbor_signal=True,
    ),
    "extended": TrainingProfile(
        name="extended",
        description=(
            "One-factor continuation candidate based on improved: retain the "
            "same architecture and optimiser settings, but extend the training "
            "budget from 180k to 360k environment steps."
        ),
        recommended_t_max=360_000,
        learning_rate=3e-4,
        gamma=0.95,
        batch_size=32,
        buffer_size=1_000,
        hidden_dim=64,
        epsilon_finish=0.02,
        # Preserve improved's absolute 126k-step annealing horizon when a run
        # is resumed at 180k rather than increasing exploration again.
        epsilon_anneal_fraction=0.35,
        target_update_interval_or_tau=0.01,
        updates_per_episode=2,
        # Keep the same approximately 18k-step checkpoint density.
        checkpoint_count=20,
        include_agent_id=False,
        mask_neighbor_signal=True,
    ),
    "refined": TrainingProfile(
        name="refined",
        description=(
            "Corrected low-learning-rate warm-start protocol for Extended QMIX "
            "from its validation-selected checkpoint to a 540k-step horizon; "
            "historical v4 artifacts using this name are invalidated."
        ),
        recommended_t_max=540_000,
        learning_rate=1e-4,
        gamma=0.95,
        batch_size=32,
        buffer_size=1_000,
        hidden_dim=64,
        epsilon_finish=0.02,
        # Keep the original absolute 126k exploration-annealing horizon.
        epsilon_anneal_fraction=125_999 / 540_000,
        target_update_interval_or_tau=0.01,
        updates_per_episode=2,
        checkpoint_count=30,
        include_agent_id=False,
        mask_neighbor_signal=True,
    ),
}


def get_training_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown training profile {name!r}; choose from {sorted(PROFILES)}") from exc
