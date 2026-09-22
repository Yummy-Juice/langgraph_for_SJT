"""User interaction, approval, commit, regeneration, and stop nodes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langgraph.types import interrupt

from sjt_system.authoring.blueprint import (
    initialize_blueprint_progress,
    validate_blueprint_agent_update,
    validate_integrated_blueprint,
)
from sjt_system.authoring.items import validate_item_agent_update
from sjt_system.authoring.requirements import validate_requirement_confirmation
from sjt_system.authoring.construct_registry import construct_selection_catalog
from sjt_system.evaluation.respondents import (
    DEFAULT_SPREAD_CAP,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SELECTION_SEED,
    MAX_VIRTUAL_SAMPLE_SIZE,
    MAX_ALLOWED_CONCURRENCY,
    MAXIMUM_SCORE_TIER_COUNT,
    PERSONA_MODE_SCORE_PROFILE,
    build_score_dimension_catalog,
    build_matched_condition_sample_config,
    build_virtual_sample_recommendations,
    generate_matched_condition_respondent_refs,
    normalize_matched_conditions,
    MATCHED_CONDITION_IDS,
    matched_condition_sample_is_current,
)
from sjt_system.evaluation.round_results import (
    build_psychometric_round_result,
)
from sjt_system.evaluation.diagnosis import (
    build_psychometric_agent_input,
    repair_tasks_from_advice,
    validate_atomic_repair_advice,
)
from sjt_system.runtime.trace import build_state_diff, utc_timestamp
from sjt_system.state import PSJTState
from sjt_system.workflow.approval import (
    build_committed_update,
    normalize_user_decision,
)
from sjt_system.workflow.constants import (
    DETERMINISTIC_AUTO_APPROVAL_ACTIONS,
    ITEM_DEVELOPMENT_ACTIONS,
    PHASE_BY_ACTION,
)

def approval_node(state: PSJTState) -> dict:
    """暂停工作流，等待用户处理本次 Agent 的候选结果。"""

    pending_update = state.get("pending_state_update")
    if pending_update is None:
        raise ValueError("Approval 节点没有收到待确认的 Agent 结果")

    is_requirement_action = (
        state.get("pending_action") == "clarify_requirements"
    )
    requirement_errors = (
        validate_requirement_confirmation(
            pending_update.get("test_specification"),
            state.get("pending_interaction"),
            state["confirmed_requirement_fields"],
            pending_update.get("specification_sources"),
        )
        if is_requirement_action
        else {}
    )
    inferred_fields = {
        field
        for field, source in (
            pending_update.get("specification_sources") or {}
        ).items()
        if source == "inferred"
    }
    requirement_decisions = (
        ["confirm", "revise", "stop"]
        if not requirement_errors
        else [
            "answer",
            *(
                ["accept_suggestions"]
                if (
                    (state.get("pending_interaction") or {}).get("suggestions")
                    or inferred_fields
                )
                else []
            ),
            "stop",
        ]
    )
    payload: dict[str, Any] = {
        "type": (
            "requirement_confirmation"
            if is_requirement_action
            else "agent_result_approval"
        ),
        "action": state.get("pending_action"),
        "summary": state.get("pending_summary"),
        "proposed_update": pending_update,
        "state_changes": state.get("pending_state_changes") or {},
        "available_decisions": (
            requirement_decisions
            if is_requirement_action
            else ["approve", "edit", "regenerate", "stop"]
        ),
    }
    if is_requirement_action:
        payload.update(state.get("pending_interaction") or {})
        if requirement_errors:
            payload["field_errors"] = requirement_errors
    while True:
        raw_decision = interrupt(payload)
        try:
            decision = normalize_user_decision(raw_decision, pending_update)
            if is_requirement_action:
                if decision["decision"] not in {
                    "answer",
                    "accept_suggestions",
                    "confirm",
                    "revise",
                    "stop",
                }:
                    raise ValueError("需求阶段不支持该决策")
                if (
                    decision["decision"] == "accept_suggestions"
                    and not payload.get("suggestions")
                    and not inferred_fields
                ):
                    raise ValueError(
                        "当前没有可接受的系统建议或默认推断值"
                    )
                if decision["decision"] == "confirm":
                    errors = validate_requirement_confirmation(
                        pending_update.get("test_specification"),
                        state.get("pending_interaction"),
                        state["confirmed_requirement_fields"],
                        pending_update.get("specification_sources"),
                    )
                    if errors:
                        payload = {
                            **payload,
                            "validation_error": "候选需求尚不能最终确认",
                            "field_errors": errors,
                        }
                        continue
            elif decision["decision"] not in {
                "approve",
                "edit",
                "regenerate",
                "stop",
            }:
                raise ValueError("普通 Agent 审批不支持该决策")
            break
        except ValueError as exc:
            payload = {**payload, "validation_error": str(exc)}

    sourced_decision = {
        **decision,
        "approval_source": "user",
    }
    return {
        "user_decision": sourced_decision,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "approval:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "approval",
                "action": decision["decision"],
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": decision.get("feedback") or "用户已提交决策",
                "approval_source": "user",
            },
        ],
    }


def item_development_mode_selection_node(state: PSJTState) -> dict:
    """在首次执行题目开发动作前暂停并选择本次运行的开发模式。"""

    if state.get("item_development_mode") in {"manual", "automatic"}:
        return {}

    payload = {
        "type": "item_development_mode_selection",
        "modes": [
            {
                "mode": "manual",
                "label": "人工逐题",
                "description": "每次出题、审题、修改和重写后都由人工确认。",
            },
            {
                "mode": "automatic",
                "label": "Agent 自动完成",
                "description": (
                    "自动完成出题、审题、修改、重写、题库冻结、虚拟验证、"
                    "筛选和组卷；终端持续展示题目与进度。"
                ),
            },
        ],
    }
    while True:
        raw_selection = interrupt(payload)
        if isinstance(raw_selection, Mapping):
            mode = raw_selection.get("mode")
            if mode in {"manual", "automatic"}:
                break
        payload = {
            **payload,
            "validation_error": (
                "题目开发模式必须是 manual 或 automatic"
            ),
        }

    return {
        "item_development_mode": mode,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "item_development_mode:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "item_development_mode_selection",
                "action": mode,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": (
                    "用户选择人工逐题模式"
                    if mode == "manual"
                    else "用户选择 Agent 自动完成模式"
                ),
                "approval_source": "user",
            },
        ],
    }


def virtual_sample_selection_node(state: PSJTState) -> dict:
    """Configure deterministic domain/facet score-conditioned respondents."""

    current_config = state.get("virtual_sample_config")
    current_respondents = state.get("virtual_respondents") or []
    if matched_condition_sample_is_current(current_config, current_respondents):
        return {}

    source_items = state.get("frozen_item_bank") or state.get("item_pool") or []
    target_dimension_ids = list(
        dict.fromkeys(
            str(item.get("target_dimension_id"))
            for item in source_items
            if isinstance(item, Mapping) and item.get("target_dimension_id")
        )
    )
    if not target_dimension_ids:
        raise ValueError("配置分数型虚拟被试前缺少题目目标维度")
    dimension_catalog = build_score_dimension_catalog(
        construct_selection_catalog()
    )
    recommendations = build_virtual_sample_recommendations(
        MAX_VIRTUAL_SAMPLE_SIZE
    )
    target_set = set(target_dimension_ids)
    display_catalog = [
        {**row, "required_target": row["dimension_id"] in target_set}
        for row in dimension_catalog
        if row.get("level") == "facet"
    ]
    payload = {
        "type": "virtual_sample_selection",
        "pool": {
            "available_count": MAX_VIRTUAL_SAMPLE_SIZE,
            "source": "deterministic_score_profile_generation",
        },
        "recommendations": recommendations,
        "recommended_sample_size": next(
            option["sample_size"]
            for option in recommendations
            if option["recommended"]
        ),
        "default_seed": DEFAULT_SELECTION_SEED,
        "default_max_concurrency": DEFAULT_MAX_CONCURRENCY,
        "max_allowed_concurrency": MAX_ALLOWED_CONCURRENCY,
        "default_max_retries": DEFAULT_MAX_RETRIES,
        "sampling_design": "matched_facet_conditions",
        "default_score_mean": 50.0,
        "default_score_sd": 15.0,
        "score_scale": [0.0, 100.0],
        "target_dimension_ids": target_dimension_ids,
        "dimension_catalog": display_catalog,
        "method_note": (
            (
                str(state.get("virtual_sample_reconfiguration_reason")) + " "
                if state.get("virtual_sample_reconfiguration_reason")
                else ""
            )
            + "固定三个顶层臂：target、same_domain、cross_domain；每个非目标臂可配置多个facet group。"
            "每个facet group独立生成一组匹配条件并共享同一正态分数向量，只在提示中提供当前group facet。"
            "每组人数相同，主施测中每名被试对每题只回答一次；target组额外完成一次整卷重测以估计虚拟作答稳定性。"
            "VTS在同域/跨域臂内取最大带符号rho；target被试同步完成目标facet对应的10题IPIP-NEO参照问卷。"
        ),
    }

    while True:
        raw_selection = interrupt(payload)
        try:
            if not isinstance(raw_selection, Mapping):
                raise ValueError("虚拟样本选择必须是对象")
            sample_size = raw_selection.get("sample_size_per_condition")
            seed = raw_selection.get("seed", DEFAULT_SELECTION_SEED)
            score_distribution = raw_selection.get("score_distribution") or {}
            mean_score = score_distribution.get("mean", 50.0)
            standard_deviation = score_distribution.get("sd", 15.0)
            max_concurrency = raw_selection.get(
                "max_concurrency",
                DEFAULT_MAX_CONCURRENCY,
            )
            max_retries = raw_selection.get(
                "max_retries",
                DEFAULT_MAX_RETRIES,
            )
            raw_conditions = raw_selection.get("conditions")
            if isinstance(sample_size, str) and sample_size.isdigit():
                sample_size = int(sample_size)
            if isinstance(seed, str) and seed.lstrip("-").isdigit():
                seed = int(seed)
            if (
                isinstance(max_concurrency, str)
                and max_concurrency.isdigit()
            ):
                max_concurrency = int(max_concurrency)
            if isinstance(max_retries, str) and max_retries.isdigit():
                max_retries = int(max_retries)
            if not isinstance(target_dimension_ids, list) or len(target_dimension_ids) != 1:
                raise ValueError("当前三臂协议要求本轮题库只包含一个目标 facet")
            conditions = normalize_matched_conditions(
                raw_conditions,
                dimension_catalog=dimension_catalog,
                target_dimension_id=target_dimension_ids[0],
            )
            selected, generation_diagnostics = generate_matched_condition_respondent_refs(
                sample_size,
                conditions,
                mean_score=float(mean_score),
                standard_deviation=float(standard_deviation),
                seed=seed,
            )
            config = build_matched_condition_sample_config(
                sample_size,
                conditions=conditions,
                generation_diagnostics=generation_diagnostics,
                mean_score=float(mean_score),
                standard_deviation=float(standard_deviation),
                seed=seed,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
            )
            break
        except (TypeError, ValueError) as exc:
            payload = {**payload, "validation_error": str(exc)}

    return {
        "virtual_sample_config": config,
        "virtual_sample_reconfiguration_reason": None,
        "virtual_respondents": selected,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "virtual_sample_selection:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "virtual_sample_selection",
                "action": "configure_virtual_sample",
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": (
                    f"用户配置固定三臂、{config.get('group_count')} 个匹配 facet group，每组 {sample_size} 名被试；"
                    f"均值 {float(mean_score):g}、SD {float(standard_deviation):g}、随机种子 {seed}"
                    f"；最大并发 {max_concurrency}"
                ),
                "approval_source": "user",
            },
        ],
    }


def post_virtual_response_decision_node(state: PSJTState) -> dict:
    """Show the round result before allowing psychometric diagnosis to start."""

    summary = state.get("virtual_response_summary") or {}
    round_result = (
        deepcopy(state.get("psychometric_round_result"))
        if isinstance(state.get("psychometric_round_result"), Mapping)
        else build_psychometric_round_result(state)
    )
    statistics = state.get("item_statistics") or {}
    failing_items = [
        {
            **deepcopy(row),
            "quality_evaluation": deepcopy(
                dict(
                    (statistics.get(str(row.get("item_id"))) or {}).get(
                        "quality_evaluation"
                    )
                    or {}
                )
            ),
        }
        for row in round_result["pending_items"]
    ]
    monitoring_warnings = [
        {
            **deepcopy(row),
            "warning": "正式题本轮监测指标未通过；资格保持锁定，不进入返修。",
            "message": "正式题本轮监测指标未通过；资格保持锁定，不进入返修。",
        }
        for row in round_result["monitoring_warnings"]
    ]
    payload = {
        "type": "post_virtual_response_decision",
        "summary": (
            "本轮虚拟作答、条件VTS分析与临时组卷基线已完成。"
            "请先核对整卷指标、未通过题目和正式题监测警告。"
        ),
        "virtual_response_summary": summary,
        "round_result": round_result,
        "failing_items": failing_items,
        "monitoring_warnings": monitoring_warnings,
        "psychometric_iteration_history": deepcopy(
            state.get("psychometric_iteration_history") or []
        ),
        "condition_score_diagnostics": deepcopy(
            round_result["condition_score_diagnostics"]
        ),
        "available_decisions": ["start", "stop"],
    }
    while True:
        raw_decision = interrupt(payload)
        decision = raw_decision.get("decision") if isinstance(raw_decision, Mapping) else None
        if decision in {"start", "stop"}:
            break
        payload = {**payload, "validation_error": "请选择开始处理或暂停并保存"}
    if decision == "stop":
        return {
            "status": "stopped",
            "psychometric_round_result": round_result,
        }
    update = {
        "psychometric_repair_user_decision": decision,
        "psychometric_round_result": round_result,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": f'{state["run_id"]}:{state["step_count"]}:post_virtual_response_decision:completed',
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "post_virtual_response_decision",
                "action": decision,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": "用户选择是否开始心理测量修题",
                "approval_source": "user",
            },
        ],
    }
    return update


def _manual_psychometric_item(
    current_item: Mapping[str, Any],
    raw_item: object,
) -> dict[str, Any]:
    if not isinstance(raw_item, Mapping):
        raise ValueError("人工修改必须提交情境与四个选项文本")
    extra_item_fields = set(raw_item) - {"scenario", "response_options"}
    if extra_item_fields:
        raise ValueError(
            "人工修改只能提交 scenario 和 response_options 文本；"
            f"锁定字段不可修改：{sorted(extra_item_fields)}"
        )
    scenario = raw_item.get("scenario")
    raw_options = raw_item.get("response_options")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("人工修改后的情境不能为空")
    if not isinstance(raw_options, list):
        raise ValueError("人工修改必须提交四个选项")
    if any(
        not isinstance(row, Mapping)
        or set(row) - {"option_id", "text"}
        for row in raw_options
    ):
        raise ValueError("人工修改的选项只能包含 option_id 和 text")
    current_options = [
        deepcopy(dict(row))
        for row in current_item.get("response_options") or []
        if isinstance(row, Mapping)
    ]
    submitted = {
        str(row.get("option_id")): row.get("text")
        for row in raw_options
        if isinstance(row, Mapping)
    }
    expected_ids = [str(row.get("option_id")) for row in current_options]
    if set(submitted) != set(expected_ids) or len(raw_options) != len(expected_ids):
        raise ValueError("人工修改必须保留原有四个 option_id")
    if any(not isinstance(submitted[option_id], str) or not submitted[option_id].strip() for option_id in expected_ids):
        raise ValueError("人工修改后的选项文本不能为空")
    edited = deepcopy(dict(current_item))
    edited["scenario"] = scenario.strip()
    edited["response_options"] = [
        {**row, "text": str(submitted[str(row["option_id"])]).strip()}
        for row in current_options
    ]
    if (
        edited["scenario"] == current_item.get("scenario")
        and edited["response_options"] == current_options
    ):
        raise ValueError("人工修改没有改变情境或选项文本")
    edited["version"] = int(current_item.get("version") or 0) + 1
    return edited


def _dequeue_psychometric_item(state: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    return {
        "items_to_revise": [
            deepcopy(entry)
            for entry in state.get("items_to_revise") or []
            if isinstance(entry, Mapping) and str(entry.get("item_id")) != item_id
        ],
        "items_to_regenerate": [
            deepcopy(entry)
            for entry in state.get("items_to_regenerate") or []
            if isinstance(entry, Mapping) and str(entry.get("item_id")) != item_id
        ],
    }


def psychometric_repair_confirmation_node(state: PSJTState) -> dict:
    """Apply a psychometric repair or automatically replenish an exhausted item.

    A defer diagnosis is not permission to invent a text patch.  It is handled
    as an automatic same-cell replenishment transaction: discard the current
    candidate, generate a new candidate under the same blueprint cell, and
    send it through the normal review and measurement loop.
    """

    pending = state.get("psychometric_repair_confirmation")
    if not isinstance(pending, Mapping) or pending.get("status") != "pending":
        raise ValueError("当前没有等待确认的单题心理测量返修建议")
    advice = pending.get("atomic_repair_advice") or {}
    evidence = pending.get("diagnosis_evidence") or {}
    item = evidence.get("current_item") or {}
    if advice.get("decision") == "repair":
        # Repair diagnoses are already user-visible in the round result. The
        # default action is now to apply the complete validated task set; do
        # not pause between diagnosis and the atomic repair transaction.
        validate_atomic_repair_advice(advice, evidence)
        tasks = repair_tasks_from_advice(advice)
        if not tasks:
            raise ValueError("repair 诊断缺少可执行的原子任务")
        history = [
            *state.get("execution_history", []),
            {
                "event_id": (
                    f'{state.get("run_id", "unknown")}:{state.get("step_count", 0)}:'
                    "psychometric_repair_confirmation:completed"
                ),
                "run_id": state.get("run_id"),
                "step": state.get("step_count", 0),
                "node": "psychometric_repair_confirmation",
                "action": "approve",
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": f"系统默认确认全部 {len(tasks)} 条原子返修任务，随后统一重测",
                "approval_source": "system_default",
            },
        ]
        return {
            "psychometric_repair_confirmation": {
                **dict(pending),
                "status": "approved",
                "decision": "approve",
                "approval_source": "system_default",
                "confirmed_task_count": len(tasks),
            },
            "execution_history": history,
        }
    payload = {
        "type": "psychometric_repair_confirmation",
        "item_id": pending.get("item_id"),
        "revision_round": pending.get("revision_round"),
        "queue_status": pending.get("queue_status"),
        "diagnosis_status": pending.get("diagnosis_status"),
        "completed_repair_rounds": pending.get("completed_repair_rounds"),
        "defer_after_rounds": pending.get("defer_after_rounds"),
        "pending_item_queue": [
            {
                "item_id": entry.get("item_id"),
                "revision_round": entry.get("revision_round"),
                "queue_status": entry.get("queue_status", "pending_diagnosis"),
            }
            for entry in state.get("items_to_revise") or []
            if isinstance(entry, Mapping) and entry.get("item_id")
        ],
        "item": item,
        "diagnosis": {
            "summary": advice.get("summary"),
            "decision": advice.get("decision"),
            "observed_discrepancies": advice.get("observed_discrepancies") or [],
            "candidate_diagnoses": advice.get("candidate_diagnoses") or [],
            "repair_tasks": advice.get("repair_tasks") or [],
            "selected_diagnosis_id": advice.get("selected_diagnosis_id"),
            "atomic_edit": advice.get("atomic_edit"),
        },
        "observations": evidence.get("observations") or [],
        "non_target_construct_constraints": [
            deepcopy(dict(row))
            for row in evidence.get("normal_constraints") or []
            if isinstance(row, Mapping)
            and str(row.get("constraint_id") or "").startswith("NON_TARGET_")
        ],
        "target_construct_constraints": [
            deepcopy(dict(row))
            for row in evidence.get("target_construct_constraints")
            or evidence.get("normal_constraints")
            or []
            if isinstance(row, Mapping)
            and row.get("component") in {"facet", "behavior_evidence"}
        ],
        "option_evidence": [
            {
                key: row.get(key)
                for key in ("option_id", "text", "behavioral_level", "score")
            }
            for row in evidence.get("option_evidence") or []
            if isinstance(row, Mapping)
        ],
        "option_score_comparisons": build_psychometric_agent_input(evidence).get(
            "option_score_comparisons"
        ) or [],
        "forced_vts_gradient_repairs": build_psychometric_agent_input(evidence).get(
            "forced_vts_gradient_repairs"
        ) or [],
        "default_defer_decision": "eliminate_replenish",
        "available_decisions": (
            ["approve", "stop"]
            if advice.get("decision") == "repair"
            else [
                "manual_edit",
                "pending_sme",
                "eliminate_replenish",
                "stop",
            ]
        ),
        "instruction": (
            "repair：确认后按原子任务自动返修；defer：人工修改、保留待SME审核、"
            "淘汰补题或暂停保存。每道 defer 题必须单独处置。"
        ),
    }

    automatic_replenish = advice.get("decision") == "defer"

    def _has_replacement_capacity() -> bool:
        lineage = state.get("item_lineage") or {}
        root_id = str(
            (lineage.get(str(item.get("item_id"))) or {}).get("root_item_id")
            or item.get("item_id")
        )
        replacement_count = sum(
            1
            for value in lineage.values()
            if isinstance(value, Mapping)
            and value.get("root_item_id") == root_id
            and isinstance(value.get("replacement_number"), int)
        )
        return replacement_count < int(
            state.get("max_item_replacement_attempts") or 2
        )

    edited_item = None
    if automatic_replenish:
        decision = "eliminate_replenish"
        raw = {"decision": decision, "approval_source": "automatic"}
        if not _has_replacement_capacity():
            item_id = str(pending.get("item_id") or item.get("item_id") or "")
            message = (
                f"题目 {item_id} 的 defer 诊断已自动转为补题，"
                f"已完成 {pending.get('completed_repair_rounds', 0)} 轮返修，"
                "且同一蓝图槽位已达到自动补题上限；"
                "系统无法在不降低质量门槛的前提下完成该槽位。"
            )
            return {
                "status": "failed",
                "errors": [
                    *(state.get("errors") or []),
                    {
                        "action": "automatic_replenishment_exhausted",
                        "item_id": item_id,
                        "message": message,
                    },
                ],
                "execution_history": [
                    *state.get("execution_history", []),
                    {
                        "event_id": (
                            f'{state.get("run_id", "unknown")}:'
                            f'{state.get("step_count", 0)}:'
                            "psychometric_repair_confirmation:failed"
                        ),
                        "run_id": state.get("run_id"),
                        "step": state.get("step_count", 0),
                        "node": "psychometric_repair_confirmation",
                        "action": "automatic_replenishment_exhausted",
                        "event_type": "failed",
                        "recorded_at": utc_timestamp(),
                        "reason": message,
                        "approval_source": "automatic_after_defer",
                    },
                ],
            }
    else:
        while True:
            raw = interrupt(payload)
            decision = raw.get("decision") if isinstance(raw, Mapping) else None
            available = set(payload["available_decisions"])
            if decision not in available:
                payload = {
                    **payload,
                    "validation_error": "请选择当前诊断允许的处置方式",
                }
                continue
            if decision == "manual_edit":
                try:
                    edited_item = _manual_psychometric_item(item, raw.get("manual_item"))
                except ValueError as exc:
                    payload = {**payload, "validation_error": str(exc)}
                    continue
            if decision == "eliminate_replenish" and not _has_replacement_capacity():
                payload = {
                    **payload,
                    "validation_error": (
                        "该蓝图槽位已达到补题次数上限，请选择人工修改或待SME审核"
                    ),
                }
                continue
            break
    if decision == "stop":
        return {"status": "stopped"}

    history = [
        *state.get("execution_history", []),
        {
            "event_id": (
                f'{state.get("run_id", "unknown")}:{state.get("step_count", 0)}:'
                "psychometric_repair_confirmation:completed"
            ),
            "run_id": state.get("run_id"),
            "step": state.get("step_count", 0),
            "node": "psychometric_repair_confirmation",
            "action": decision,
            "event_type": "completed",
            "recorded_at": utc_timestamp(),
            "reason": (
                "defer 诊断不执行不充分证据下的原题修改，系统自动启动同槽位补题"
                if automatic_replenish
                else "用户确认当前单题心理测量返修建议"
            ),
            "approval_source": (
                "automatic_after_defer" if automatic_replenish else "user"
            ),
        },
    ]
    if decision == "approve":
        return {
            "psychometric_repair_confirmation": {
                **dict(pending),
                "status": "approved",
                "decision": "approve",
            },
            "execution_history": history,
        }

    item_id = str(pending.get("item_id") or "")
    dispositions = deepcopy(state.get("item_final_dispositions") or {})
    item_version = (item or {}).get("version")
    queue_update = _dequeue_psychometric_item(state, item_id)
    common_history = [
        *deepcopy(state.get("psychometric_repair_history") or []),
        {
            "event": f"psychometric_defer_{decision}",
            "recorded_at": utc_timestamp(),
            "item_id": item_id,
            "revision_round": pending.get("revision_round"),
            "diagnosis_fingerprint": pending.get("diagnosis_fingerprint"),
            "approval_source": (
                "automatic_after_defer" if automatic_replenish else "user"
            ),
        },
    ]
    if decision == "manual_edit":
        cell_id = str(item.get("blueprint_cell_id") or "")
        blueprint_cell = next(
            (deepcopy(dict(row)) for row in (state.get("blueprint") or {}).get("cells") or [] if isinstance(row, Mapping) and row.get("cell_id") == cell_id),
            None,
        )
        item_specification = next(
            (deepcopy(dict(row)) for row in state.get("item_specifications") or [] if isinstance(row, Mapping) and row.get("specification_id") == item_id),
            None,
        )
        if blueprint_cell is None or item_specification is None:
            raise ValueError("人工修改找不到原蓝图槽位或题目规格")
        validate_item_agent_update(
            "revise_item",
            {"current_item": edited_item},
            target_item_id=item_id,
            target_blueprint_cell_id=cell_id,
            blueprint_cell=blueprint_cell,
            item_specification=item_specification,
            previous_item=dict(item),
        )
        return {
            "psychometric_repair_confirmation": None,
            "current_item": edited_item,
            "current_blueprint_cell": blueprint_cell,
            "current_item_specification": item_specification,
            "current_item_review": None,
            "review_process_status": "not_started",
            "current_item_repair_attempted": False,
            "current_item_repair_failure": None,
            "active_psychometric_repair": {
                **deepcopy(dict(pending)),
                "action": "manual_edit",
                "manual_edit_pending_review": True,
                "baseline_item": deepcopy(dict(item)),
                "baseline_profile": deepcopy((state.get("item_pattern_profiles") or {}).get(item_id)),
            },
            **queue_update,
            "psychometric_repair_history": common_history,
            "execution_history": history,
        }
    if decision == "pending_sme":
        dispositions[item_id] = {
            "status": "pending_sme_review",
            "item_version": item_version,
            "diagnosis": deepcopy(dict(advice)),
        }
        return {
            "psychometric_repair_confirmation": None,
            "active_psychometric_repair": None,
            **queue_update,
            "items_deferred_for_revision": [
                *deepcopy(state.get("items_deferred_for_revision") or []),
                {"item_id": item_id, "item": deepcopy(dict(item)), "reason": "pending_sme_review"},
            ],
            "item_final_dispositions": dispositions,
            "selection_results": None,
            "psychometric_repair_history": common_history,
            "execution_history": history,
        }

    lineage = deepcopy(state.get("item_lineage") or {})
    root_id = str((lineage.get(item_id) or {}).get("root_item_id") or item_id)
    lineage_number = 1 + sum(
        1 for value in lineage.values()
        if isinstance(value, Mapping)
        and value.get("root_item_id") == root_id
        and isinstance(value.get("replacement_number"), int)
    )
    existing_ids = {
        str(row.get("item_id")) for row in state.get("item_pool") or [] if isinstance(row, Mapping)
    } | {str(row.get("specification_id")) for row in state.get("item_specifications") or [] if isinstance(row, Mapping)}
    # 补题号同时考虑已存在的 R-n 补题 ID（checkpoint 恢复后 item_pool/规格
    # 可能已含上次生成的补题而 lineage 未同步），取更大值 +1，避免重复。
    prefix = f"{root_id}-R"
    existing_number = max(
        (
            int(repl_id[len(prefix):])
            for repl_id in existing_ids
            if repl_id.startswith(prefix)
            and repl_id[len(prefix):].isdigit()
        ),
        default=0,
    )
    replacement_number = max(lineage_number, existing_number + 1)
    replacement_id = f"{root_id}-R{replacement_number}"
    if replacement_id in existing_ids:
        raise ValueError(f"补题ID已存在：{replacement_id}")
    cell_id = str(item.get("blueprint_cell_id") or "")
    blueprint = deepcopy(state.get("blueprint") or {})
    replacement_cell = None
    for cell in blueprint.get("cells") or []:
        if isinstance(cell, dict) and cell.get("cell_id") == cell_id:
            cell["planned_generation_count"] = int(cell.get("planned_generation_count") or 0) + 1
            replacement_cell = deepcopy(cell)
            break
    else:
        raise ValueError("淘汰补题找不到原蓝图单元")
    source_spec = next(
        (deepcopy(dict(row)) for row in state.get("item_specifications") or [] if isinstance(row, Mapping) and row.get("specification_id") == item_id),
        None,
    )
    if source_spec is None:
        raise ValueError("淘汰补题找不到原题目规格")
    blueprint.setdefault("slots", []).append(
        {
            "specification_id": replacement_id,
            "blueprint_cell_id": cell_id,
            "candidate_reference": {
                "mechanism_id": source_spec.get("mechanism_id")
                or replacement_cell.get("mechanism_id"),
                "situation_id": source_spec.get("situation_id")
                or replacement_cell.get("situation_id"),
            },
        }
    )
    source_skeleton = (state.get("item_skeletons") or {}).get(item_id)
    source_skeleton_review = (state.get("skeleton_reviews") or {}).get(item_id)
    if not isinstance(source_skeleton, Mapping):
        raise ValueError("淘汰补题找不到原题心理骨架")
    replacement_spec = {**source_spec, "specification_id": replacement_id}
    progress = deepcopy(state.get("blueprint_progress") or {})
    cell_progress = progress.setdefault(cell_id, {"generated": 0, "passed": 0, "rejected": 0, "missing": 0})
    cell_progress["passed"] = max(0, int(cell_progress.get("passed") or 0) - 1)
    cell_progress["rejected"] = int(cell_progress.get("rejected") or 0) + 1
    cell_progress["missing"] = int(cell_progress.get("missing") or 0) + 1
    lineage[item_id] = {"root_item_id": root_id, "status": "eliminated", "replaced_by_item_id": replacement_id}
    lineage[replacement_id] = {"root_item_id": root_id, "replaces_item_id": item_id, "replacement_number": replacement_number}
    dispositions[item_id] = {"status": "eliminated", "item_version": item_version, "replacement_item_id": replacement_id}
    remaining_revise = queue_update.get("items_to_revise") or []
    remaining_regenerate = queue_update.get("items_to_regenerate") or []
    batch_has_remaining = bool(remaining_revise or remaining_regenerate)
    update = {
        "psychometric_repair_confirmation": None,
        "active_psychometric_repair": None,
        **queue_update,
        "current_blueprint_cell": replacement_cell,
        "current_item_specification": None,
        "item_pool": [deepcopy(row) for row in state.get("item_pool") or [] if isinstance(row, Mapping) and str(row.get("item_id")) != item_id],
        "removed_items": [*deepcopy(state.get("removed_items") or []), deepcopy(dict(item))],
        "rejected_items": [
            *deepcopy(state.get("rejected_items") or []),
            {
                "item": deepcopy(dict(item)),
                "reason": (
                    "automatic_replenishment_after_defer"
                    if automatic_replenish
                    else "user_eliminated_for_replenishment"
                ),
            },
        ],
        "item_final_dispositions": dispositions,
        "item_lineage": lineage,
        "blueprint": blueprint,
        "blueprint_progress": progress,
        "item_specifications": [*deepcopy(state.get("item_specifications") or []), replacement_spec],
        "item_skeletons": {**deepcopy(state.get("item_skeletons") or {}), replacement_id: deepcopy(dict(source_skeleton))},
        "skeleton_reviews": {**deepcopy(state.get("skeleton_reviews") or {}), replacement_id: deepcopy(source_skeleton_review)},
        "selection_results": None,
        "selected_items": [],
        "psychometric_repair_history": common_history,
        "execution_history": history,
    }
    if not batch_has_remaining:
        previous_response_ref = state.get("virtual_response_data_ref")
        update.update(
            {
                "virtual_response_data_ref": None,
                "virtual_response_summary": None,
                "virtual_response_item_bank_id": None,
                "virtual_response_item_bank_version": None,
                "item_statistics": {},
                "psychometric_round_result": None,
                "test_statistics": None,
            }
        )
        if isinstance(previous_response_ref, str) and previous_response_ref:
            update["previous_virtual_response_data_ref"] = previous_response_ref
    return update


def plateau_gap_decision_node(state: PSJTState) -> dict:
    """Plateau close-out reached a blueprint gap: ask the user how to fill it.

    Eligible candidates are content-reviewed items that are not pending SME or
    eliminated. The user may pick a candidate as-is (provisional fill) or
    manually rewrite one, then the next selection pass consumes the fill and
    proceeds to assembly as a developmental version.
    """

    pending = state.get("plateau_gap_decision")
    if not isinstance(pending, Mapping) or pending.get("status") != "pending":
        raise ValueError("当前没有待处置的平台期蓝图缺口清单")
    gap_cells = [
        dict(cell)
        for cell in pending.get("gap_cells") or []
        if isinstance(cell, Mapping)
    ]
    dispositions = state.get("item_final_dispositions") or {}

    def _eligible(item_id: str) -> bool:
        dis = dispositions.get(item_id)
        return not (
            isinstance(dis, Mapping)
            and dis.get("status") in {"pending_sme_review", "eliminated"}
        )

    frozen_index = {
        str(item.get("item_id")): item
        for item in state.get("frozen_item_bank") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    payload: dict[str, Any] = {
        "type": "plateau_gap_decision",
        "summary": (
            "整卷质量已到平台期，但仍有蓝图单元没有可正式入卷的题目。"
            "请对每个缺口单元处理：直接点选一道候选（开发版补位），"
            "或手动修改一道候选后再收卷。待SME/已淘汰的候选不可选。"
        ),
        "gap_cells": gap_cells,
        "available_modes": ["pick", "manual", "stop"],
    }
    while True:
        raw = interrupt(payload)
        if not isinstance(raw, Mapping) or raw.get("decision") not in {
            "resolve",
            "stop",
        }:
            payload = {
                **payload,
                "validation_error": "请提交 resolve 或 stop",
            }
            continue
        if raw.get("decision") == "stop":
            break
        resolutions = raw.get("resolutions")
        if not isinstance(resolutions, list) or not resolutions:
            payload = {
                **payload,
                "validation_error": "请为每个缺口单元选择候选，或选择停止",
            }
            continue
        fills: dict[str, Any] = {}
        errors: list[str] = []
        for res in resolutions:
            if not isinstance(res, Mapping):
                errors.append("决议格式无效")
                break
            cell_id = str(res.get("cell_id") or "")
            cell = next(
                (
                    row
                    for row in gap_cells
                    if str(row.get("blueprint_cell_id")) == cell_id
                ),
                None,
            )
            if cell is None:
                errors.append(f"未知缺口单元 {cell_id}")
                break
            if cell_id in fills:
                errors.append(f"单元 {cell_id} 重复处置")
                break
            item_id = str(res.get("item_id") or "")
            sme_override = bool(res.get("sme_override"))
            candidate = next(
                (
                    row
                    for row in cell.get("candidates") or []
                    if str(row.get("item_id")) == item_id
                ),
                None,
            )
            allowed = bool(
                candidate is not None
                and (
                    candidate.get("eligible")
                    or (candidate.get("force_allowed") and sme_override)
                )
            )
            if not allowed:
                errors.append(f"单元 {cell_id} 的候选 {item_id} 不可选")
                break
            base_item = frozen_index.get(item_id)
            if base_item is None:
                errors.append(f"找不到候选题目 {item_id}")
                break
            mode = str(res.get("mode") or "pick")
            if mode == "manual":
                try:
                    edited_item = _manual_psychometric_item(
                        base_item,
                        res.get("manual_item"),
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                    break
                fills[cell_id] = {
                    "item_id": item_id,
                    "mode": "manual",
                    "sme_override": sme_override,
                    "edited_item": edited_item,
                }
            elif mode == "pick":
                fills[cell_id] = {
                    "item_id": item_id,
                    "mode": "pick",
                    "sme_override": sme_override,
                }
            else:
                errors.append(f"单元 {cell_id} 的模式无效")
                break
        if not errors:
            unresolved = [
                str(cell.get("blueprint_cell_id"))
                for cell in gap_cells
                if str(cell.get("blueprint_cell_id")) not in fills
            ]
            if unresolved:
                errors.append("以下单元尚未处置：" + "、".join(unresolved))
        if errors:
            payload = {
                **payload,
                "validation_error": "；".join(errors),
            }
            continue
        break

    history = [
        *state.get("execution_history", []),
        {
            "event_id": (
                f'{state.get("run_id", "unknown")}:'
                f'{state.get("step_count", 0)}:plateau_gap_decision:completed'
            ),
            "run_id": state.get("run_id"),
            "step": state.get("step_count", 0),
            "node": "plateau_gap_decision",
            "action": (
                "stop" if raw.get("decision") == "stop" else "resolve_gap"
            ),
            "event_type": "completed",
            "recorded_at": utc_timestamp(),
            "reason": (
                "用户选择暂停保存"
                if raw.get("decision") == "stop"
                else "用户已处置全部平台期缺口单元"
            ),
            "approval_source": "user",
        },
    ]
    if raw.get("decision") == "stop":
        return {"status": "stopped", "execution_history": history}
    return {
        "plateau_gap_fills": fills,
        "plateau_gap_decision": None,
        "selection_results": None,
        "execution_history": history,
    }


def automatic_approval_node(state: PSJTState) -> dict:
    """为自动模式下已通过执行校验的题目动作生成系统批准决策。"""

    action = state.get("pending_action")
    is_automatic_item_action = (
        state.get("item_development_mode") == "automatic"
        and action in ITEM_DEVELOPMENT_ACTIONS
    )
    is_automatic_deterministic_action = (
        state.get("item_development_mode") == "automatic"
        and action in DETERMINISTIC_AUTO_APPROVAL_ACTIONS
    )
    if (
        not is_automatic_item_action
        and not is_automatic_deterministic_action
    ):
        raise ValueError(
            "自动审批仅适用于自动模式下的闭环动作"
        )
    if state.get("pending_state_update") is None:
        raise ValueError("自动审批节点没有收到待提交的 Agent 结果")

    decision = {
        "decision": "approve",
        "feedback": None,
        "state_patch": None,
        "approval_source": "system",
    }
    return {
        "user_decision": decision,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "approval:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "approval",
                "action": "approve",
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": f"自动模式已批准题目动作 {action}",
                "approval_source": "system",
            },
        ],
    }


def commit_node(state: PSJTState) -> dict:
    """把用户确认后的候选更新写入正式业务 State。"""

    pending_update = state.get("pending_state_update")
    decision = state.get("user_decision")
    if pending_update is None or decision is None:
        raise ValueError("Commit 节点缺少候选结果或用户决策")

    committed_update = build_committed_update(pending_update, decision)
    action = state.get("pending_action") or "unknown"
    requirement_update: dict[str, Any] = {}
    if action == "clarify_requirements":
        if decision["decision"] != "confirm":
            raise ValueError("只有 confirm 决策可以提交需求规格")
        errors = validate_requirement_confirmation(
            committed_update.get("test_specification"),
            state.get("pending_interaction"),
            state["confirmed_requirement_fields"],
            committed_update.get("specification_sources"),
        )
        if errors:
            raise ValueError(f"需求确认校验失败：{errors}")
        requirement_update["requirements_confirmed"] = True
    blueprint_update: dict[str, Any] = {}
    if action == "build_blueprint":
        validate_blueprint_agent_update(committed_update)
        blueprint = committed_update.get("blueprint")
        errors = validate_integrated_blueprint(
            blueprint,
            state.get("test_specification"),
        )
        if errors:
            raise ValueError(f"蓝图提交校验失败：{errors}")
        blueprint_update["blueprint_progress"] = (
            initialize_blueprint_progress(blueprint)
        )
        if "construct_profile_snapshot" in blueprint:
            profile = blueprint["construct_profile_snapshot"]
            blueprint_update["construct_profile"] = profile
            blueprint_update["item_skeletons"] = {}
            blueprint_update["skeleton_reviews"] = {}
            blueprint_update["item_specifications"] = []
        else:
            raise ValueError("当前工作流不再接受旧版动态构念蓝图")
    full_committed_update = {
        **committed_update,
        **requirement_update,
        **blueprint_update,
        "current_phase": PHASE_BY_ACTION.get(
            action,
            state.get("current_phase"),
        ),
    }
    state_changes = build_state_diff(state, full_committed_update)
    approval_source = decision.get("approval_source", "user")

    return {
        **full_committed_update,
        "pending_action": None,
        "pending_state_update": None,
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": None,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:commit:completed'
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "commit",
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": (
                    "系统已自动批准，候选结果写入正式 State"
                    if approval_source == "system"
                    else "用户已确认，候选结果写入正式 State"
                ),
                "state_changes": state_changes,
                "approval_source": approval_source,
            },
        ],
    }


def prepare_regeneration_node(state: PSJTState) -> dict:
    """丢弃候选结果，并保留用户意见供同一 Agent 重新生成。"""

    decision = state.get("user_decision") or {}
    action = state.get("pending_action") or "unknown"
    is_requirement_action = action == "clarify_requirements"
    feedback = decision.get("feedback")
    confirmed_fields = state["confirmed_requirement_fields"]
    if is_requirement_action and decision.get("decision") == "accept_suggestions":
        suggestion_fields = {
            item.get("field")
            for item in (
                (state.get("pending_interaction") or {}).get("suggestions")
                or []
            )
            if isinstance(item, dict) and isinstance(item.get("field"), str)
        }
        pending_sources = (
            (state.get("pending_state_update") or {}).get(
                "specification_sources",
                {},
            )
        )
        inferred_fields = {
            field
            for field, source in pending_sources.items()
            if source == "inferred"
        }
        accepted_fields = suggestion_fields | inferred_fields
        confirmed_fields = sorted({*confirmed_fields, *accepted_fields})
        feedback = (
            "我接受本轮全部系统建议：" + "、".join(sorted(accepted_fields))
        )
    conversation = state["requirement_conversation"]
    if is_requirement_action and feedback:
        conversation = [
            *conversation,
            {"role": "user", "content": feedback},
        ]
    return {
        "pending_action": None,
        # Requirement Agent 需要看到上一版候选，才能只修改用户指出的字段。
        "pending_state_update": (
            state.get("pending_state_update")
            if is_requirement_action
            else None
        ),
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": feedback,
        "confirmed_requirement_fields": confirmed_fields,
        "requirement_conversation": conversation,
        "requirements_confirmed": (
            False
            if is_requirement_action
            else state["requirements_confirmed"]
        ),
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "regenerate:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "prepare_regeneration",
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": feedback or "用户要求继续需求澄清",
            },
        ],
    }


def stop_node(state: PSJTState) -> dict:
    """记录用户主动停止并结束工作流。"""

    decision = state.get("user_decision") or {}
    action = state.get("pending_action") or "unknown"
    return {
        "status": "stopped",
        "pending_action": None,
        "pending_state_update": None,
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": decision.get("feedback"),
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": f'{state["run_id"]}:{state["step_count"]}:stop:completed',
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "stop",
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": decision.get("feedback") or "用户主动停止工作流",
            },
        ],
    }
