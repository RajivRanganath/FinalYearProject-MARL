"""Predeclared split protocol for the weight-updating QMIX refinement."""

from __future__ import annotations


REFINED_SELECTION_SEEDS = tuple(range(251, 271))
REFINED_FINAL_SEEDS = tuple(range(5001, 5031))

PROMOTION_RULE = (
    "Promote only if refined mean paired reward has a 95% CI above zero, all "
    "three matched training replicas improve mean reward, mean recall advantage "
    "is at least -0.002, mean energy advantage is at least -0.10, and mean "
    "redundancy advantage is at least -1.0 on seeds 251--270."
)

