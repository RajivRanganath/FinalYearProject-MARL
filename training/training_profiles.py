"""Named, reproducible training profiles for the MARL experiments.

The historical ``baseline`` profile reproduces the published Phase 25--33
experiments.  The ``improved`` profile is a validation-motivated optimisation
of the learner, not a change to the environment or reward.  Keeping the two
profiles separate prevents a tuning run from silently overwriting the
published models.
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
}


def get_training_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown training profile {name!r}; choose from {sorted(PROFILES)}") from exc
