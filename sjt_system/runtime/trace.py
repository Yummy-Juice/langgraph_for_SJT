"""Workflow trace timestamps, errors, and state diffs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


_IGNORED_DIFF_FIELDS = {
    "errors",
    "execution_history",
    "route",
    "step_count",
}
_MAX_STRING_LENGTH = 300
_MAX_MAPPING_ITEMS = 20
_MAX_COLLECTION_ITEMS = 10
_MAX_DEPTH = 3
_MAX_ERROR_LENGTH = 2000
_BULK_DIFF_FIELDS = {
    "execution_history",
    "item_history",
    "item_pool",
    "rejected_items",
    "removed_items",
    "frozen_item_bank",
    "item_specifications",
    "item_pattern_profiles",
    "psychometric_repair_history",
    "psychometric_selection_history",
    "psychometric_iteration_history",
    "virtual_respondents",
}
_IDENTITY_FIELDS = (
    "item_id",
    "specification_id",
    "event_id",
    "cell_id",
    "respondent_id",
    "task_id",
)


def utc_timestamp() -> str:
    """返回适合写入轨迹事件的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat()


def summarize_error_message(error: object) -> str:
    """Bound persisted exception text while retaining diagnostic context."""

    message = str(error)
    if len(message) <= _MAX_ERROR_LENGTH:
        return message
    return message[:_MAX_ERROR_LENGTH] + "…"


def summarize_value(value: Any, *, depth: int = 0) -> Any:
    """压缩大型状态值，同时保留足够的调试上下文。"""

    if isinstance(value, str):
        if len(value) <= _MAX_STRING_LENGTH:
            return value
        return value[:_MAX_STRING_LENGTH] + "…"

    if depth >= _MAX_DEPTH:
        if isinstance(value, Mapping):
            return {"type": "dict", "field_count": len(value)}
        if isinstance(value, (list, tuple, set)):
            return {"type": type(value).__name__, "item_count": len(value)}
        return value

    if isinstance(value, Mapping):
        items = list(value.items())
        summary = {
            str(key): summarize_value(item, depth=depth + 1)
            for key, item in items[:_MAX_MAPPING_ITEMS]
        }
        if len(items) > _MAX_MAPPING_ITEMS:
            summary["__omitted_fields__"] = len(items) - _MAX_MAPPING_ITEMS
        return summary

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        preview = [
            summarize_value(item, depth=depth + 1)
            for item in items[:_MAX_COLLECTION_ITEMS]
        ]
        if len(items) <= _MAX_COLLECTION_ITEMS:
            return preview
        return {
            "type": type(value).__name__,
            "item_count": len(items),
            "preview": preview,
        }

    return value


def summarize_bulk_value(value: Any) -> Any:
    """Summarize large authority collections without embedding their bodies."""

    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "field_count": len(value),
            "keys": [str(key) for key in list(value)[:_MAX_MAPPING_ITEMS]],
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        identifiers = []
        for item in items[:_MAX_COLLECTION_ITEMS]:
            if not isinstance(item, Mapping):
                identifiers.append(summarize_value(item, depth=_MAX_DEPTH))
                continue
            identity = next(
                (
                    item.get(field)
                    for field in _IDENTITY_FIELDS
                    if item.get(field) is not None
                ),
                None,
            )
            identifiers.append(identity or {"field_count": len(item)})
        return {
            "type": type(value).__name__,
            "item_count": len(items),
            "identifiers": identifiers,
        }
    return summarize_value(value)


def build_state_diff(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """仅比较 Agent 本次尝试更新的业务字段。"""

    changes: dict[str, Any] = {}
    for field, new_value in update.items():
        if field in _IGNORED_DIFF_FIELDS:
            continue

        old_value = state.get(field)
        if old_value == new_value:
            continue

        summarizer = (
            summarize_bulk_value
            if field in _BULK_DIFF_FIELDS
            else summarize_value
        )
        changes[field] = {
            "before": summarizer(old_value),
            "after": summarizer(new_value),
        }
    return changes
