"""Isolated, offline-testable post-baseline iteration workflow.

This package deliberately does not import or mutate the production C workflow.
It is a small harness for testing the logic after a baseline item set:
diagnosis, item repair, local re-measurement, replacement/replenishment,
checkpointing, and resume.
"""

from .core import LabConfig, PostBaselineIterationLab, demo_snapshot, load_snapshot

__all__ = [
    "LabConfig",
    "PostBaselineIterationLab",
    "demo_snapshot",
    "load_snapshot",
]
