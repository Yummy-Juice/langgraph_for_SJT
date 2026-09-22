"""Translate workflow state and updates into stable user-facing view models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any


ACTION_LABELS = {
    "clarify_requirements": "需求澄清",
    "build_blueprint": "构念—题目细目表",
    "generate_item": "生成题目",
    "review_item": "审查题目",
    "revise_item": "修改题目",
    "regenerate_item": "重写题目",
    "simulate_responses": "虚拟被试作答",
    "analyze_psychometrics": "心理测量分析",
    "select_items": "筛选题目",
    "psychometric_repair_batch": "并发心理测量返修",
    "assemble_test": "测验组卷",
    "review_test": "整体审核",
    "rescore_test": "重新计分",
    "generate_reports": "生成报告",
    "accept_item": "题目入库",
    "abandon_item": "候选处理",
    "manual": "人工逐题模式",
    "automatic": "自动开发模式",
    "finish": "完成",
}

PHASE_LABELS = {
    "requirements": "需求确认",
    "construct_blueprint": "构念—题目细目表",
    "item_development": "题目开发",
    "virtual_simulation": "虚拟作答",
    "psychometric_analysis": "测量分析",
    "item_selection": "题目筛选",
    "test_assembly": "测验组装",
    "reporting": "成果交付",
}

CONTENT_FIELDS_BY_ACTION = {
    "clarify_requirements": ("test_specification",),
    "build_blueprint": ("blueprint",),
    "generate_item": ("current_item",),
    "review_item": ("current_item_review", "current_item"),
    "revise_item": ("current_item",),
    "regenerate_item": ("current_item",),
    "simulate_responses": ("virtual_response_summary",),
    "analyze_psychometrics": (
        "psychometric_round_result",
    ),
    "select_items": ("selection_results", "selected_items"),
    "psychometric_repair_batch": ("item_pool", "psychometric_repair_history"),
    "assemble_test": ("assembled_test",),
    "review_test": ("test_review_result",),
    "rescore_test": ("test_review_result", "test_statistics"),
    "generate_reports": (
        "final_test",
        "technical_report",
        "virtual_respondent_report",
    ),
}


def action_label(action: object) -> str:
    value = str(action or "unknown")
    return ACTION_LABELS.get(value, value)


def phase_label(phase: object) -> str:
    value = str(phase or "requirements")
    return PHASE_LABELS.get(value, value)


def extract_action_content(
    action: str,
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only business output relevant to one completed action."""

    pending = update.get("pending_state_update")
    source = pending if isinstance(pending, Mapping) else update
    content = {}
    for field in CONTENT_FIELDS_BY_ACTION.get(action, ()):
        value = source.get(field)
        if value is not None and value != [] and value != {}:
            content[field] = deepcopy(value)
    return content


def build_timeline_entries(
    updates: Iterable[Mapping[str, Any]],
    seen_event_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Build deduplicated timeline cards from graph update chunks."""

    seen = set(seen_event_ids or ())
    entries = []
    for record in updates:
        node = str(record.get("node") or "unknown")
        update = record.get("update")
        if not isinstance(update, Mapping):
            continue
        history = update.get("execution_history") or []
        for event in history:
            if not isinstance(event, Mapping):
                continue
            event_id = event.get("event_id")
            if isinstance(event_id, str) and event_id in seen:
                continue
            if isinstance(event_id, str):
                seen.add(event_id)
            action = str(event.get("action") or "unknown")
            entries.append(
                {
                    "node": node,
                    "event": deepcopy(dict(event)),
                    "content": (
                        extract_action_content(action, update)
                        if event.get("event_type") == "completed"
                        else {}
                    ),
                }
            )
    return entries, seen


def progress_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small set of run metrics useful in the page header."""

    item_pool = state.get("item_pool") or []
    rejected = state.get("rejected_items") or []
    selected = state.get("selected_items") or []
    return {
        "phase": phase_label(state.get("current_phase")),
        "status": state.get("status", "running"),
        "steps": state.get("step_count", 0),
        "candidate_items": len(item_pool),
        "rejected_items": len(rejected),
        "selected_items": len(selected),
    }
