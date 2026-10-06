"""Persistent policy helpers for same-slot item replacement.

Replacement quota is an operator policy, not a content-quality threshold.  New
and resumed runs use an explicit ``unlimited`` policy so a rejected/deferred
candidate can never be admitted merely because a finite replacement counter was
reached.  The legacy integer field remains in the state for compatibility and
audit readability, but ``None`` is the serialized representation of unlimited.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


UNLIMITED_REPLACEMENT_POLICY = "unlimited"


def replacement_quota_unlimited(state: Mapping[str, Any]) -> bool:
    """Return whether same-slot replacement attempts are unbounded."""

    # The quota was permanently raised by operator policy.  Keep the state
    # argument for a stable call contract, but never let a legacy integer
    # checkpoint reintroduce a finite cap.
    return True


def replacement_quota_limit(state: Mapping[str, Any]) -> int | None:
    """Return the legacy finite quota, or ``None`` for unlimited policy."""

    if replacement_quota_unlimited(state):
        return None
    value = state.get("max_item_replacement_attempts")
    return int(value) if value is not None else None


def replacement_capacity_available(
    state: Mapping[str, Any], item_id: str,
) -> bool:
    """Check the same-slot quota without admitting a failed candidate."""

    if replacement_quota_unlimited(state):
        return True
    lineage = state.get("item_lineage") or {}
    root_id = str(
        (lineage.get(str(item_id)) or {}).get("root_item_id") or item_id
    )
    used = max(
        (
            int(value.get("replacement_number") or 0)
            for value in lineage.values()
            if isinstance(value, Mapping)
            and str(value.get("root_item_id") or "") == root_id
        ),
        default=0,
    )
    limit = replacement_quota_limit(state)
    return limit is None or used < limit


def normalize_replacement_policy(state: dict[str, Any]) -> dict[str, Any]:
    """Apply the permanent unlimited policy to a mutable state mapping."""

    state["replacement_quota_policy"] = UNLIMITED_REPLACEMENT_POLICY
    state["max_item_replacement_attempts"] = None
    return state

