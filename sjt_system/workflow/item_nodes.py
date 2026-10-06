"""Deterministic item review, revision, acceptance, and rejection nodes."""

from __future__ import annotations

from typing import Any

from sjt_system.authoring.items import (
    build_accept_item_update,
    build_item_agent_route,
    build_review_transition_update,
    derive_item_review_decision,
    derive_item_repair_action,
    record_committed_item_output,
)
from sjt_system.runtime.trace import build_state_diff, utc_timestamp
from sjt_system.state import PSJTState

def _item_transition_result(
    state: PSJTState,
    *,
    node: str,
    action: str,
    reason: str,
    update: dict[str, Any],
) -> dict[str, Any]:
    """为确定性题目迁移补充统一轨迹。"""

    return {
        **update,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    f"{node}:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": node,
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": reason,
                "state_changes": build_state_diff(state, update),
            },
        ],
    }


def prepare_item_review_node(state: PSJTState) -> dict:
    """记录已提交的题目版本，并确定性进入审题。"""

    route = state.get("route")
    if route is None:
        raise ValueError("题目审查准备节点缺少 Route")
    committed_action = route["next_action"]
    from sjt_system.evaluation.virtual_content_review import is_enabled
    if is_enabled(state) and committed_action == "generate_item" and not state.get("active_psychometric_repair"):
        if int(state.get("psychometric_analysis_round") or 0) != 0:
            raise ValueError("虚拟内容复审协议不允许在已施测后自动追加初始候选题")
        update = record_committed_item_output(state, committed_action)
        update.update(initial_candidate_admission=True, current_item_review=None,
                      review_process_status="not_started", item_content_status="not_evaluated")
        return _item_transition_result(
            state, node="prepare_item_review", action="admit_initial_candidate",
            reason="初始候选仅经结构校验入库，先虚拟施测再开展内容调查", update=update,
        )
    repaired_from_blocking_review = (
        committed_action in {"revise_item", "regenerate_item"}
        and bool(state.get("current_item_repair_attempted"))
        and (
            isinstance(state.get("active_psychometric_repair"), dict)
            or any(
                isinstance(finding, dict)
                and finding.get("severity") == "blocking"
                for finding in (
                    (state.get("current_item_review") or {}).get("findings") or []
                )
            )
        )
    )
    update = record_committed_item_output(state, committed_action)
    if repaired_from_blocking_review:
        update["current_item_review"] = {
            "findings": [],
            "repair_tasks": [],
            "summary": "审题意见指定的位置已经修改，按定向修复规则直接通过。",
        }
    else:
        update.update(
            build_item_agent_route(
                state,
                "review_item",
                f"{committed_action} 已确认，进入题目审查",
            )
        )
    return _item_transition_result(
        state,
        node="prepare_item_review",
        action="review_item",
        reason="题目版本已记录，下一步执行审题",
        update=update,
    )


def prepare_item_revision_node(state: PSJTState) -> dict:
    """Record the diagnosis and choose rewrite or option-only revision."""

    review = state.get("current_item_review")
    if not isinstance(review, dict):
        raise ValueError("题目修订准备节点缺少统一审题结果")
    active_psychometric_repair = state.get("active_psychometric_repair")
    atomic_advice = (
        active_psychometric_repair.get("atomic_repair_advice")
        if isinstance(active_psychometric_repair, dict)
        else None
    )
    is_psychometric_atomic_repair = isinstance(atomic_advice, dict) and bool(
        atomic_advice.get("repair_tasks")
        or atomic_advice.get("atomic_edit")
    )
    if is_psychometric_atomic_repair:
        # A failed psychometric patch must retry the same scoped patch.  Do
        # not reinterpret the stale/fallback review as permission to launch a
        # full item regeneration.
        decision = "REVISE"
        repair_action = "revise_item"
    else:
        decision = derive_item_review_decision(
            review,
            repair_attempted=bool(state.get("current_item_repair_attempted")),
            repair_attempt_count=int(
                state.get("current_item_revision_count") or 0
            ),
            max_repair_attempts=int(
                state.get("max_item_revision_attempts") or 3
            ),
            rewrite_count=int(state.get("current_item_rewrite_count") or 0),
            max_rewrite_rounds=int(state.get("max_item_rewrite_rounds") or 3),
        )
        repair_action = (
            "regenerate_item"
            if decision in {"REWRITE", "REJECT"}
            else derive_item_repair_action(review)
        )
    is_rewrite = repair_action == "regenerate_item"
    skeleton_blocking = (
        not is_psychometric_atomic_repair
        and any(
            finding.get("severity") == "blocking"
            and finding.get("locus") == "skeleton"
            for finding in review.get("findings") or []
            if isinstance(finding, dict)
        )
    )
    next_revision = int(state.get("current_item_revision_count") or 0) + 1
    next_rewrite = int(state.get("current_item_rewrite_count") or 0) + 1
    update = build_review_transition_update(
        state,
        repair_action,
        (
            "情境存在 blocking 问题，保持固定骨架并修复成题实现"
            if is_rewrite
            else (
                "问题仅位于选项或计分映射，执行第 "
                f"{next_revision} 次定向修改"
            )
        ),
    )
    if decision == "REJECT":
        # The original fixed slot is not discarded. Start a fresh realization
        # cycle under the same item_id and blueprint cell.
        update["current_item_revision_count"] = 0
        update["current_item_rewrite_count"] = 0
        update["current_item_replacement_count"] = int(
            state.get("current_item_replacement_count") or 0
        ) + 1
    if skeleton_blocking:
        specification = state.get("current_item_specification") or {}
        specification_id = str(specification.get("specification_id") or "")
        update.update(
            {
                "current_skeleton_repair_required": True,
                "item_skeletons": {
                    key: value
                    for key, value in (state.get("item_skeletons") or {}).items()
                    if key != specification_id
                },
                "skeleton_reviews": {
                    key: value
                    for key, value in (state.get("skeleton_reviews") or {}).items()
                    if key != specification_id
                },
                "item_specifications": [
                    row
                    for row in state.get("item_specifications") or []
                    if row.get("specification_id") != specification_id
                ],
                "current_item_specification": {
                    "specification_id": specification_id,
                    "blueprint_cell_id": specification.get(
                        "blueprint_cell_id"
                    ),
                    "target_dimension_id": specification.get(
                        "target_dimension_id"
                    ),
                },
            }
        )
    return _item_transition_result(
        state,
        node="prepare_item_revision",
        action=repair_action,
        reason=(
            "题目忠实实现了有阻断问题的骨架；下一步在原题号下重建骨架并修复成题"
            if skeleton_blocking
            else f"已记录诊断，下一步执行同一题号的第 {next_rewrite} 次完整重写"
            if is_rewrite
            else "已记录诊断，下一步只修改被指向的选项或计分映射"
        ),
        update=update,
    )


def accept_item_node(state: PSJTState) -> dict:
    """将通过的当前题目移入候选题库。"""

    update = build_accept_item_update(state)
    return _item_transition_result(
        state,
        node="accept_item",
        action="accept_item",
        reason=("结构合法的初始候选进入 item_pool，内容尚未审查"
                if state.get("initial_candidate_admission") else "题目通过审查并进入 item_pool"),
        update=update,
    )


def abandon_item_node(state: PSJTState) -> dict:
    """After all bounded retries, keep the latest structurally valid item."""

    message = (
        "题目已达到全部内容修订与重写上限；保留同一题号下的最新结构合法版本，"
        "放入题库并继续流程。"
    )
    acceptance_state = {
        **state,
        "current_item_review": {
            "findings": [],
            "repair_tasks": [],
            "summary": message,
        },
    }
    update = build_accept_item_update(acceptance_state)
    return _item_transition_result(
        state,
        node="abandon_item",
        action="accept_latest_item",
        reason=message,
        update=update,
    )
