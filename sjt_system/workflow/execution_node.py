"""Execute one routed action and prepare its pending update."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any

from sjt_system.authoring.bank import validate_item_bank_owned_fields
from sjt_system.authoring.blueprint import (
    format_blueprint_errors_for_user,
    validate_blueprint_agent_update,
    validate_integrated_blueprint,
)
from sjt_system.authoring.items import validate_item_agent_update
from sjt_system.authoring.construct_registry import (
    ConstructResolutionError,
    construct_selection_label,
)
from sjt_system.authoring.requirements import (
    build_requirement_interaction,
    build_confirmed_requirement_fields_update,
    canonicalize_requirement_agent_update,
)
from sjt_system.runtime.trace import (
    build_state_diff,
    summarize_error_message,
    utc_timestamp,
)
from sjt_system.state import PSJTState
from sjt_system.workflow.constants import PHASE_BY_ACTION
from sjt_system.workflow.executor import (
    ItemReviewProcessError,
    SkeletonDevelopmentFailure,
    execute_agent,
)

async def execute_node(state: PSJTState) -> dict:
    """调用 Router 指定的专业 Agent。"""
    started_at = perf_counter()
    route = state["route"]
    if route is None:
        message = "Execute 未收到 Router 决策"
        return {
            "status": "failed",
            "errors": [
                *state["errors"],
                {"action": None, "message": message},
            ],
            "execution_history": [
                *state["execution_history"],
                {
                    "event_id": f'{state["run_id"]}:{state["step_count"]}:execute:failed',
                    "run_id": state["run_id"],
                    "step": state["step_count"],
                    "node": "execute",
                    "action": "unknown",
                    "event_type": "failed",
                    "recorded_at": utc_timestamp(),
                    "duration_ms": round((perf_counter() - started_at) * 1000),
                    "error": message,
                },
            ],
        }
    if state["step_count"] >= state["max_steps"]:
        message = "测试达到最大执行步数，流程已停止"
        return {
            "status": "failed",
            "errors": [
                *state["errors"],
                {"action": route["next_action"], "message": message},
            ],
            "execution_history": [
                *state["execution_history"],
                {
                    "event_id": (
                        f'{state["run_id"]}:{state["step_count"]}:'
                        "execute:failed"
                    ),
                    "run_id": state["run_id"],
                    "step": state["step_count"],
                    "node": "execute",
                    "action": route["next_action"],
                    "event_type": "failed",
                    "recorded_at": utc_timestamp(),
                    "duration_ms": round((perf_counter() - started_at) * 1000),
                    "error": message,
                },
            ],
        }
    try:
        result = await execute_agent(route, state)
        effective_action = route["next_action"]
        if effective_action not in PHASE_BY_ACTION:
            raise ValueError(
                f"Agent 返回了无效的 effective_action：{effective_action!r}"
            )
        proposed_update = result.get("state_update")
        if not isinstance(proposed_update, dict):
            raise ValueError("Agent 输出缺少有效的 state_update")
        if effective_action == "clarify_requirements":
            proposed_update = canonicalize_requirement_agent_update(
                proposed_update,
                previous_candidate=state.get("pending_state_update"),
                user_feedback=state.get("user_feedback"),
                user_request=state.get("user_request"),
            )
            result = {**result, "state_update": proposed_update}
        validate_item_bank_owned_fields(
            effective_action,
            proposed_update,
        )
        if effective_action == "build_blueprint":
            validate_blueprint_agent_update(proposed_update)
            blueprint_errors = validate_integrated_blueprint(
                proposed_update.get("blueprint"),
                state.get("test_specification"),
            )
            if blueprint_errors:
                raise ValueError(
                    format_blueprint_errors_for_user(blueprint_errors)
                )
        if effective_action != "psychometric_repair_batch":
            validate_item_agent_update(
                effective_action,
                proposed_update,
                target_item_id=route.get("target_item_id"),
                target_blueprint_cell_id=route.get("target_blueprint_cell_id"),
                specification=state.get("test_specification"),
                blueprint_cell=state.get("current_blueprint_cell"),
                item_specification=(
                    proposed_update.get("current_item_specification")
                    or state.get("current_item_specification")
                ),
                previous_item=state.get("current_item"),
            )
        is_requirement_action = effective_action == "clarify_requirements"
        pending_interaction = (
            build_requirement_interaction(result)
            if is_requirement_action
            else None
        )
        requirement_status_update: dict[str, Any] = {}
        if is_requirement_action and pending_interaction is not None:
            requirement_status_update = build_confirmed_requirement_fields_update(
                result,
                state["confirmed_requirement_fields"],
            )
            questions = pending_interaction["questions"]
            if questions:
                requirement_status_update["requirement_conversation"] = [
                    *state["requirement_conversation"],
                    {
                        "role": "assistant",
                        "content": "\n".join(
                            str(question["text"]) for question in questions
                        ),
                    },
                ]
        state_changes = build_state_diff(
            state,
            {**proposed_update, **requirement_status_update},
        )
        repair_attempt_count = result.get("repair_attempt_count", 0)
        semantic_retry_count = result.get("semantic_retry_count", 0)
        if is_requirement_action:
            specification = proposed_update["test_specification"]
            readiness = (
                "需求字段完整，等待用户确认"
                if pending_interaction and not pending_interaction["questions"]
                else "需求仍需用户补充或确认"
            )
            construct_label = construct_selection_label(
                specification["construct_selection"]
            )
            summary = (
                "当前候选规格："
                f"测量构念（{construct_label}）、"
                f"目标群体（{specification['target_population']}）、"
                f"题目数量（{specification['final_item_count']}）。"
                f"{readiness}。"
            )
        else:
            summary = result.get("summary")
        if repair_attempt_count:
            repair_note = (
                f"结构输出自动修复 {repair_attempt_count} 次后通过校验"
            )
            summary = f"{summary}（{repair_note}）" if summary else repair_note
        if semantic_retry_count:
            retry_note = (
                f"蓝图语义校验自动重试 {semantic_retry_count} 次后通过校验"
            )
            summary = f"{summary}（{retry_note}）" if summary else retry_note
        return {
            "step_count": state["step_count"] + 1,
            "pending_action": effective_action,
            "pending_state_update": proposed_update,
            "pending_summary": summary,
            "pending_state_changes": state_changes,
            "pending_interaction": pending_interaction,
            "user_decision": None,
            "requirements_confirmed": (
                False
                if is_requirement_action
                else state["requirements_confirmed"]
            ),
            **requirement_status_update,
            "execution_history": [
                *state["execution_history"],
                {
                    "event_id": f'{state["run_id"]}:{state["step_count"] + 1}:execute:completed',
                    "run_id": state["run_id"],
                    "step": state["step_count"] + 1,
                    "node": "execute",
                    "action": effective_action,
                    "event_type": "completed",
                    "recorded_at": utc_timestamp(),
                    "duration_ms": round((perf_counter() - started_at) * 1000),
                    "reason": (
                        "；".join(
                            note for note in (
                                (
                                    f"结构输出自动修复 {repair_attempt_count} 次"
                                    if repair_attempt_count
                                    else ""
                                ),
                                (
                                    f"蓝图语义校验自动重试 {semantic_retry_count} 次"
                                    if semantic_retry_count
                                    else ""
                                ),
                            )
                            if note
                        )
                        or "Agent 输出首次通过结构校验"
                    ),
                    "state_changes": state_changes,
                },
            ],
        }
    except Exception as exc:
        message = summarize_error_message(exc)
        if isinstance(exc, ItemReviewProcessError):
            # Invalid reviewer JSON is a process failure, not an item-content
            # judgment. After bounded retries, keep the latest structurally
            # valid item and let the workflow accept it without stopping.
            return {
                "status": "running",
                "step_count": state["step_count"] + 1,
                "review_process_status": "exhausted",
                "item_content_status": "not_evaluated",
                "current_review_request_id": exc.review_request_id,
                "current_review_item_id": exc.item_id,
                "current_review_item_version": exc.item_version,
                "current_review_retry_count": exc.attempt_count,
                "current_item_review": {
                    "findings": [],
                    "repair_tasks": [],
                    "summary": (
                        "审题输出在限定次数内未形成合法结构；"
                        "保留当前最新结构合法题目并继续流程。"
                    ),
                },
                "current_item_repair_failure": None,
                "pending_action": None,
                "pending_state_update": None,
                "pending_summary": None,
                "pending_state_changes": None,
                "pending_interaction": None,
                "errors": [
                    *state["errors"],
                    {
                        "action": "review_item",
                        "message": message,
                        "recoverable": True,
                        "review_request_id": exc.review_request_id,
                        "item_id": exc.item_id,
                        "item_version": exc.item_version,
                    },
                ],
                "execution_history": [
                    *state["execution_history"],
                    {
                        "event_id": (
                            f'{state["run_id"]}:{state["step_count"] + 1}:'
                            "execute:review_process_failed"
                        ),
                        "run_id": state["run_id"],
                        "step": state["step_count"] + 1,
                        "node": "execute",
                        "action": "review_item",
                        "event_type": "review_process_failed",
                        "recorded_at": utc_timestamp(),
                        "duration_ms": round(
                            (perf_counter() - started_at) * 1000
                        ),
                        "error": message,
                        "review_request_id": exc.review_request_id,
                        "item_id": exc.item_id,
                        "item_version": exc.item_version,
                        "review_process_status": "exhausted",
                        "item_content_status": "not_evaluated",
                    },
                ],
            }
        if isinstance(exc, SkeletonDevelopmentFailure):
            specification = state.get("current_item_specification") or {}
            cell_id = str(specification.get("blueprint_cell_id") or "")
            progress = {
                key: dict(value)
                for key, value in (state.get("blueprint_progress") or {}).items()
            }
            cell_progress = progress.setdefault(
                cell_id,
                {"generated": 0, "passed": 0, "rejected": 0, "missing": 0},
            )
            if not isinstance(state.get("current_item"), Mapping):
                cell_progress["generated"] = int(
                    cell_progress.get("generated", 0)
                ) + 1
                cell_progress["missing"] = max(
                    0, int(cell_progress.get("missing", 0)) - 1
                )
            cell_progress["rejected"] = int(
                cell_progress.get("rejected", 0)
            ) + 1
            failures = {
                **(state.get("skeleton_failures") or {}),
                exc.specification_id: {
                    "status": "skeleton_failed",
                    "reason": message,
                    "final_review": deepcopy(
                        (exc.history[-1] if exc.history else {}).get("review")
                    ),
                    "attempt_count": len(exc.history),
                },
            }
            histories = {
                **(state.get("skeleton_review_history") or {}),
                exc.specification_id: deepcopy(exc.history),
            }
            rejected_items = [
                deepcopy(item) for item in state.get("rejected_items") or []
            ]
            if isinstance(state.get("current_item"), Mapping):
                rejected_items.append(deepcopy(state["current_item"]))
            return {
                "status": "running",
                "step_count": state["step_count"] + 1,
                "skeleton_review_history": histories,
                "skeleton_failures": failures,
                "skeleton_slot_failure_pending": True,
                "blueprint_progress": progress,
                "rejected_items": rejected_items,
                "current_item": None,
                "current_item_specification": None,
                "current_blueprint_cell": None,
                "current_item_review": None,
                "current_item_repair_attempted": False,
                "current_item_repair_failure": None,
                "current_item_revision_count": 0,
                "current_item_rewrite_count": 0,
                "current_skeleton_repair_required": False,
                "pending_action": None,
                "pending_state_update": None,
                "pending_summary": None,
                "pending_state_changes": None,
                "pending_interaction": None,
                "errors": [
                    *state["errors"],
                    {
                        "action": "skeleton_failed",
                        "message": message,
                        "recoverable": True,
                        "specification_id": exc.specification_id,
                    },
                ],
                "execution_history": [
                    *state["execution_history"],
                    {
                        "event_id": (
                            f'{state["run_id"]}:{state["step_count"] + 1}:'
                            "execute:skeleton_rejected"
                        ),
                        "run_id": state["run_id"],
                        "step": state["step_count"] + 1,
                        "node": "execute",
                        "action": "skeleton_failed",
                        "event_type": "skeleton_rejected",
                        "recorded_at": utc_timestamp(),
                        "duration_ms": round(
                            (perf_counter() - started_at) * 1000
                        ),
                        "error": message,
                        "reason": "当前固定槽位已记录为失败，继续下一个槽位",
                    },
                ],
            }
        if isinstance(exc, ConstructResolutionError):
            return {
                "status": "running",
                "step_count": state["step_count"] + 1,
                "requirements_confirmed": False,
                "confirmed_requirement_fields": [
                    field
                    for field in state.get("confirmed_requirement_fields") or []
                    if field != "construct_selection"
                ],
                "user_feedback": message,
                "pending_action": None,
                "pending_state_update": None,
                "pending_summary": None,
                "pending_state_changes": None,
                "pending_interaction": None,
                "errors": [
                    *state["errors"],
                    {
                        "action": route["next_action"],
                        "message": message,
                        "recoverable": True,
                    },
                ],
                "execution_history": [
                    *state["execution_history"],
                    {
                        "event_id": (
                            f'{state["run_id"]}:{state["step_count"] + 1}:'
                            "execute:requirements_reopened"
                        ),
                        "run_id": state["run_id"],
                        "step": state["step_count"] + 1,
                        "node": "execute",
                        "action": route["next_action"],
                        "event_type": "failed",
                        "recorded_at": utc_timestamp(),
                        "duration_ms": round(
                            (perf_counter() - started_at) * 1000
                        ),
                        "error": message,
                        "reason": (
                            "目标构念无法唯一解析，已返回需求澄清阶段"
                        ),
                    },
                ],
            }
        if (
            (
                route["next_action"] in {"revise_item"}
                or (
                    route["next_action"] == "regenerate_item"
                    and (
                        bool(state.get("current_item_repair_attempted"))
                        or isinstance(
                            state.get("active_psychometric_repair"), Mapping
                        )
                    )
                )
            )
            and isinstance(state.get("current_item"), Mapping)
        ):
            failed_review = state.get("current_item_review")
            if not isinstance(failed_review, Mapping):
                failed_review = {
                    "findings": [
                        {
                            "criterion": "ecological_plausibility",
                            "severity": "blocking",
                            "locus": "skeleton",
                            "affected_option_ids": [],
                            "evidence": "审题模型未返回可校验的结构化诊断。",
                            "problem": "无法确认当前候选题是否满足最低审题要求。",
                            "repair_instruction": "淘汰当前候选并补充新候选。",
                        }
                    ],
                    "repair_tasks": [
                        {
                            "task_id": "deterministic-review-output-invalid",
                            "source": "deterministic",
                            "targets": [
                                {
                                    "field": "scenario",
                                    "option_ids": [],
                                }
                            ],
                            "problem": "审题模型两次输出均未通过结构校验",
                            "instruction": "题目直接淘汰，不启动自动修题",
                        }
                    ],
                    "summary": f"审题输出无效：{message}",
                }
            action = route["next_action"]
            return {
                "status": "running",
                "step_count": state["step_count"] + 1,
                "current_item_review": dict(failed_review),
                "current_item_repair_attempted": True,
                "current_item_repair_failure": message,
                "execution_history": [
                    *state["execution_history"],
                    {
                        "event_id": (
                            f'{state["run_id"]}:{state["step_count"] + 1}:'
                            "execute:repair_failed"
                        ),
                        "run_id": state["run_id"],
                        "step": state["step_count"] + 1,
                        "node": "execute",
                        "action": action,
                        "event_type": "failed",
                        "recorded_at": utc_timestamp(),
                        "duration_ms": round(
                            (perf_counter() - started_at) * 1000
                        ),
                        "error": message,
                        "reason": (
                            "审题两次输出均无效，题目进入淘汰流程"
                            if action == "review_item"
                            else "定向修改两次输出均无效，题目进入淘汰流程"
                        ),
                    },
                ],
            }
        return {
            "status": "failed",
            "step_count": state["step_count"] + 1,
            "errors": [
                *state["errors"],
                {
                    "action": route["next_action"],
                    "message": message,
                },
            ],
            "execution_history": [
                *state["execution_history"],
                {
                    "event_id": f'{state["run_id"]}:{state["step_count"] + 1}:execute:failed',
                    "run_id": state["run_id"],
                    "step": state["step_count"] + 1,
                    "node": "execute",
                    "action": route["next_action"],
                    "event_type": "failed",
                    "recorded_at": utc_timestamp(),
                    "duration_ms": round((perf_counter() - started_at) * 1000),
                    "error": message,
                },
            ],
        }
