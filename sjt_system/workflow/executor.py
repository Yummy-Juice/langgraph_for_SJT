"""Dispatch LLM-backed and deterministic workflow actions."""

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any

from langchain_core.runnables import Runnable

from sjt_system.agent.agent_factory import (
    PSYCHOMETRIC_REASONING_ROLE_MANIFEST,
    compact_skeleton_agent,
    item_regeneration_agent,
    item_review_agent,
    item_writer_agent,
    psychometric_item_repair_agent,
    psychometric_repair_diagnosis_agent,
    requirement_agent,
    revision_agent,
)
from sjt_system.agent.client import normalize_model_output_shape
from sjt_system.authoring.blueprint import (
    format_blueprint_errors_for_user,
)
from sjt_system.authoring.construct_registry import resolve_specification_profile
from sjt_system.authoring.generation_plan import (
    build_generation_blueprint,
    classify_compact_skeletons,
    GENERATION_BLUEPRINT_VERSION,
    materialize_item_specifications,
    planned_generation_count,
    required_expansion_situation_total,
    validate_generation_blueprint,
    required_generation_total,
    resolve_blueprint_design,
)
from sjt_system.state import ItemRepairResult, PSJTRouteDecision, PSJTState
from sjt_system.authoring.context import (
    build_item_generation_context,
    build_item_pattern_profile,
    build_item_model_state,
    build_psychometric_repair_generation_context,
    build_psychometric_repair_model_state,
    build_requirement_model_state,
    build_unified_review_context,
)
from sjt_system.authoring.items import (
    canonicalize_item_agent_update,
    validate_item_agent_update,
    validate_item_review,
    validate_item_review_diagnosis,
)
from sjt_system.authoring.bank import (
    audit_candidate_item_bank,
    build_item_bank_freeze_update,
)
from sjt_system.agent.retry import ainvoke_model_with_schema_repair
from sjt_system.runtime.progress import emit_progress
from sjt_system.runtime.telemetry import (
    aggregate_iteration_calls,
    iteration_context,
    read_ledger,
)
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.evaluation.simulation import (
    run_single_item_virtual_retest,
    run_virtual_response_simulation,
)
from sjt_system.evaluation.psychometrics import (
    evaluate_single_item_candidate,
    run_psychometric_analysis,
)
from sjt_system.evaluation.form_metrics import (
    PLATEAU_DEFAULT_MIN_DELTA,
    PLATEAU_DEFAULT_PATIENCE,
    assess_form_plateau,
    build_provisional_form_metrics,
    form_quality_summary,
)
from sjt_system.evaluation.selection import (
    build_psychometric_repair_evidence,
    _psychometric_repair_entry,
    psychometric_diagnosis_to_review,
    run_item_selection,
    validate_psychometric_repair_diagnosis,
)
from sjt_system.evaluation.form_optimizer import optimize_test_form_with_agent
from sjt_system.evaluation.diagnosis import (
    build_construct_diagnosis_evidence,
    build_deterministic_forced_vts_repair_advice,
    build_deterministic_defer_advice,
    build_deterministic_target_gradient_repair_advice,
    build_psychometric_agent_input,
    diagnosis_fingerprint,
    item_requires_psychometric_diagnosis,
    normalize_atomic_option_patch_scope,
    normalize_target_gradient_repair_advice,
    repair_tasks_from_advice,
    validate_atomic_item_patch,
    validate_atomic_repair_advice,
)
from sjt_system.knowledge.behavior_evidence import (
    attach_behavior_evidence,
    load_ipip_corpus,
)
from sjt_system.knowledge.behavior_evidence_agents import ensure_behavior_evidence
from sjt_system.authoring.situation_space import (
    BLUEPRINT_SEMANTIC_RETRY_ATTEMPTS,
    INCREMENTAL_CANDIDATES_PER_CELL,
    ensure_facet_expansion,
    propose_blueprint_rows,
)
from sjt_system.delivery.assembly import run_test_assembly
from sjt_system.delivery.lifecycle import run_test_rescore, run_test_review
from sjt_system.delivery.reporting import run_report_generation
from sjt_system.workflow.constants import PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS


MAX_ITEM_OUTPUT_CANDIDATES = 2
MAX_ITEM_SKELETON_ATTEMPTS = 1


def _development_iteration_for_action(
    action: str,
    state: Mapping[str, Any],
) -> int | None:
    """Return the development iteration that should own model-call usage."""

    current_round = int(state.get("psychometric_analysis_round") or 0)
    if action == "psychometric_repair_batch":
        return max(1, current_round)
    if action in {
        "generate_item",
        "regenerate_item",
        "review_item",
        "simulate_responses",
        "analyze_psychometrics",
    }:
        return current_round + 1
    if action == "revise_item":
        return max(1, current_round) if state.get("active_psychometric_repair") else current_round + 1
    if action == "select_items":
        return max(1, current_round)
    return None


def _iteration_token_usage(
    state: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    """Read the current session's usage for one development iteration."""

    records = read_ledger()
    run_id = state.get("run_id")
    usage = aggregate_iteration_calls(
        records,
        iteration=iteration,
        run_id=str(run_id) if isinstance(run_id, str) and run_id else None,
    )
    if not usage.get("data_available"):
        # Direct simulation/agent invocations may not establish run_context.
        # Keep the curve usable, but mark that the session-wide fallback was
        # used so the report does not imply perfect run attribution.
        usage = aggregate_iteration_calls(
            records,
            iteration=iteration,
            run_id=None,
        )
        if usage.get("data_available"):
            usage["scope_fallback"] = "session"
    return usage


def _partial_provisional_form_ids(
    state: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> list[str]:
    """Build a transparent partial form when a complete form is infeasible."""

    by_cell: dict[str, list[str]] = {}
    item_ids = {
        str(item.get("item_id"))
        for item in candidates
        if item.get("item_id")
    }
    for item in candidates:
        item_id = str(item.get("item_id") or "")
        cell_id = str(item.get("blueprint_cell_id") or "")
        if item_id and cell_id and item_id in item_ids:
            by_cell.setdefault(cell_id, []).append(item_id)
    selected: list[str] = []
    for cell in (state.get("blueprint") or {}).get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("cell_id") or "")
        planned = int(cell.get("planned_retention_count") or 0)
        selected.extend(by_cell.get(cell_id, [])[: max(0, planned)])
    return selected


async def _build_provisional_iteration_record(
    state: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble and score a provisional form before item repair starts."""

    iteration = int(state.get("psychometric_analysis_round") or 0)
    if iteration < 1:
        raise ValueError("临时组卷需要先完成至少一轮心理测量分析")
    item_statistics = state.get("item_statistics") or {}
    test_statistics = state.get("test_statistics")
    optimizer_result: dict[str, Any] | None = None
    selection_error: str | None = None
    try:
        optimizer_result = await optimize_test_form_with_agent(
            state,
            candidates,
            item_statistics,
            test_statistics if isinstance(test_statistics, Mapping) else None,
        )
        selected_ids = [
            str(item_id) for item_id in optimizer_result.get("selected_item_ids") or []
        ]
    except Exception as exc:
        selection_error = str(exc)
        selected_ids = _partial_provisional_form_ids(state, candidates)

    final_item_count = sum(
        int(cell.get("planned_retention_count") or 0)
        for cell in (state.get("blueprint") or {}).get("cells") or []
        if isinstance(cell, Mapping)
    )
    form_metrics = build_provisional_form_metrics(state, selected_ids)
    qualified_count = sum(
        1
        for item_id in candidates
        if (
            (item_statistics.get(str(item_id.get("item_id"))) or {})
            .get("quality_evaluation", {})
            .get("recommendation")
            == "retain"
        )
        and isinstance(item_id, Mapping)
    )
    return {
        "analysis_round": iteration,
        "recorded_at": utc_timestamp(),
        "candidate_count": len(candidates),
        "qualified_item_count": qualified_count,
        "requested_item_count": final_item_count,
        "item_count": len(selected_ids),
        "form_status": "complete" if len(selected_ids) == final_item_count else "incomplete",
        "form_item_ids": selected_ids,
        "form_metrics": form_metrics,
        "form_optimizer": deepcopy(optimizer_result),
        "form_selection_error": selection_error,
        "token_usage": _iteration_token_usage(state, iteration),
    }


def _upsert_iteration_record(
    history: list[dict[str, Any]],
    record: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Persist one round once, then refresh its cumulative usage."""

    iteration = int(record.get("analysis_round") or 0)
    output = [deepcopy(dict(row)) for row in history if isinstance(row, Mapping)]
    refreshed = deepcopy(dict(record))
    refreshed["token_usage"] = _iteration_token_usage(state, iteration)
    for index, existing in enumerate(output):
        if int(existing.get("analysis_round") or 0) == iteration:
            output[index] = refreshed
            break
    else:
        output.append(refreshed)
    output.sort(key=lambda row: int(row.get("analysis_round") or 0))
    return output


def _annotate_iteration_quality(
    history: list[dict[str, Any]],
    plateau_status: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Persist candidate and retained-best quality on every iteration row."""

    trajectory = {
        int(row.get("analysis_round") or 0): row
        for row in plateau_status.get("trajectory") or []
        if isinstance(row, Mapping)
    }
    current_round = int(plateau_status.get("current_round") or 0)
    annotated: list[dict[str, Any]] = []
    for entry in history:
        row = deepcopy(dict(entry))
        round_number = int(row.get("analysis_round") or 0)
        quality_row = trajectory.get(round_number) or {}
        summary = form_quality_summary(row.get("form_metrics") or {})
        row["candidate_form_quality"] = quality_row.get(
            "candidate_form_quality",
            summary.get("candidate_form_quality"),
        )
        row["best_so_far_form_quality"] = quality_row.get(
            "best_so_far_form_quality"
        )
        row["accepted_as_best"] = bool(
            quality_row.get("accepted_as_best", False)
        )
        row["eligible_for_best_so_far"] = bool(
            quality_row.get(
                "eligible_for_best_so_far",
                summary.get("eligible_for_best_so_far", False),
            )
        )
        if round_number == current_round:
            row["plateau_status"] = deepcopy(dict(plateau_status))
        annotated.append(row)
    return annotated


async def execute_psychometric_analysis_with_provisional_form(
    state: PSJTState,
) -> dict[str, Any]:
    """Analyze one round, then assemble its provisional form immediately.

    The provisional form is a round-level baseline. It is deliberately built
    before the repair decision so the user can see whole-test quality before
    deciding whether to enter the single-item repair queue.
    """

    result = await asyncio.to_thread(run_psychometric_analysis, state)
    state_update = result.get("state_update")
    if not isinstance(state_update, dict):
        raise ValueError("心理测量分析缺少有效的 state_update")

    analysis_state: PSJTState = {
        **state,
        **state_update,
    }
    candidates = [
        deepcopy(dict(item))
        for item in analysis_state.get("frozen_item_bank") or []
        if isinstance(item, Mapping)
    ]
    if not candidates:
        raise ValueError("心理测量分析后缺少冻结题库，无法临时组卷")

    provisional = await _build_provisional_iteration_record(
        analysis_state,
        candidates,
    )
    iteration = int(provisional.get("analysis_round") or 0)
    prior_history = [
        deepcopy(dict(row))
        for row in state.get("psychometric_iteration_history") or []
        if isinstance(row, Mapping)
        and int(row.get("analysis_round") or 0) != iteration
    ]
    plateau_status = assess_form_plateau(
        [*prior_history, provisional],
        patience=int(
            state.get("psychometric_plateau_patience")
            or PLATEAU_DEFAULT_PATIENCE
        ),
        min_delta=float(
            state.get("psychometric_plateau_min_delta")
            if state.get("psychometric_plateau_min_delta") is not None
            else PLATEAU_DEFAULT_MIN_DELTA
        ),
    )
    iteration_history = _upsert_iteration_record(
        [
            deepcopy(dict(row))
            for row in state.get("psychometric_iteration_history") or []
            if isinstance(row, Mapping)
        ],
        provisional,
        state=analysis_state,
    )
    iteration_history = _annotate_iteration_quality(
        iteration_history,
        plateau_status,
    )
    state_update = {
        **state_update,
        "psychometric_iteration_history": iteration_history,
        "psychometric_plateau_status": deepcopy(plateau_status),
    }
    return {
        **result,
        "state_update": state_update,
        "summary": (
            f"{result.get('summary') or '心理测量分析完成'}"
            f" 已完成第 {iteration} 轮临时组卷，"
            f"整卷指标状态={provisional.get('form_status', 'unknown')}。"
        ),
    }


def distribute_situation_quotas(
    final_item_count: int, facet_count: int
) -> list[int]:
    """Distribute the requested unique situations evenly across facets."""

    final_item_count = int(final_item_count)
    facet_count = int(facet_count)
    if final_item_count < 1 or facet_count < 1:
        raise ValueError("题数和 facet 数必须为正整数")
    if final_item_count < facet_count:
        raise ValueError("最终题数必须不少于所选 facet 数")
    base, remainder = divmod(final_item_count, facet_count)
    return [base + (1 if index < remainder else 0) for index in range(facet_count)]


class SkeletonDevelopmentFailure(ValueError):
    """One fixed slot exhausted its bounded skeleton-development budget."""

    def __init__(
        self,
        specification_id: str,
        history: list[dict[str, Any]],
        reason: str | None = None,
    ) -> None:
        self.specification_id = specification_id
        self.history = deepcopy(history)
        super().__init__(
            "当前固定槽位的心理骨架未通过程序确定性校验"
            + (f"：{reason}" if reason else "")
        )


class ItemReviewProcessError(ValueError):
    """The reviewer exhausted retries without producing a valid review."""

    def __init__(
        self,
        message: str,
        *,
        item_id: str,
        item_version: int,
        review_request_id: str,
        attempt_count: int,
    ) -> None:
        super().__init__(message)
        self.item_id = item_id
        self.item_version = item_version
        self.review_request_id = review_request_id
        self.attempt_count = attempt_count


# 这里只注册 LLM 任务；统计、筛选、组卷和重计分后续注册确定性函数。
AGENT_MAP: dict[str, Runnable] = {
    "clarify_requirements": requirement_agent,
    "generate_item": item_writer_agent,
    "revise_item": revision_agent,
    "regenerate_item": item_regeneration_agent,
}


async def _ainvoke_model(
    agent: Runnable,
    input_data: dict[str, Any],
    *,
    job_label: str,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
) -> Any:
    """Invoke one model request with a bounded end-to-end timeout."""

    return await ainvoke_model_with_schema_repair(
        agent,
        input_data,
        job_label=job_label,
        max_schema_repair_attempts=1,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )


class PsychometricDiagnosisUnavailable(RuntimeError):
    """A repair diagnosis failed before it could support a safe item change."""


PSYCHOMETRIC_REPAIR_TIMEOUT_SECONDS = 600.0
PSYCHOMETRIC_REPAIR_SUBAGENT_CONCURRENCY = 4


def _output_error_kind(exc: ValueError) -> str:
    """Classify output failures without conflating JSON and business errors."""

    value = getattr(exc, "error_kind", None)
    return value if isinstance(value, str) else "business_validation"


def _invalid_candidate(
    result: Any,
    exc: ValueError,
    *,
    state_update_only: bool = False,
) -> Any:
    """Preserve parser/schema candidates that failed before result assignment."""

    candidate = getattr(exc, "candidate", None)
    if candidate is not None:
        return candidate
    if state_update_only and isinstance(result, dict):
        return result.get("state_update")
    return result


def _normalize_item_repair_result(result: Any) -> Any:
    """Use the shared structural adapter for direct or monkeypatched calls."""

    return normalize_model_output_shape(result, ItemRepairResult)


async def execute_item_review(state: PSJTState) -> dict[str, Any]:
    """Run one diagnosis call, then derive all repair tasks in code."""

    current_item = state.get("current_item")
    if not isinstance(current_item, Mapping):
        raise ValueError("review_item requires current_item")
    item_id = current_item.get("item_id")
    item_version = current_item.get("version")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("review_item current_item is missing item_id")
    if not isinstance(item_version, int) or isinstance(item_version, bool):
        raise ValueError("review_item current_item is missing item version")
    review_request_id = (
        f"{state.get('run_id')}:{item_id}:v{item_version}:"
        f"review:{int(state.get('step_count') or 0)}"
    )
    context = build_unified_review_context(state)
    input_data = context
    last_error: ValueError | None = None
    max_attempts = 4
    for attempt in range(max_attempts):
        result: Any = None
        try:
            result = await _ainvoke_model(
                item_review_agent,
                {"input_data": input_data},
                job_label="统一题目审查",
            )
            validate_item_review_diagnosis(
                result,
                current_item=state.get("current_item"),
            )
            findings = deepcopy(result["findings"])
            review = {
                "findings": findings,
                "repair_tasks": [],
                "summary": result["summary"],
            }
            validate_item_review(
                review,
                current_item=state.get("current_item"),
            )
            return {
                "state_update": {
                    "current_item_review": review,
                    "review_process_status": "valid",
                    "item_content_status": (
                        "needs_repair"
                        if any(
                            finding.get("severity") == "blocking"
                            for finding in findings
                            if isinstance(finding, Mapping)
                        )
                        else "pass"
                    ),
                    "current_review_request_id": review_request_id,
                    "current_review_item_id": item_id,
                    "current_review_item_version": item_version,
                    "current_review_retry_count": attempt,
                },
                "summary": review["summary"],
                "repair_attempt_count": attempt,
            }
        except ValueError as exc:
            last_error = exc
            if attempt >= max_attempts - 1:
                break
            emit_progress(
                {
                    "type": "output_repair",
                    "retry_kind": _output_error_kind(exc),
                    "job_label": "统一题目审查",
                    "attempt": attempt + 2,
                    "max_attempts": max_attempts,
                    "reason": str(exc),
                }
            )
            input_data = {
                **context,
                "validation_feedback": str(exc),
                "previous_invalid_candidate": _invalid_candidate(result, exc),
            }
    raise ItemReviewProcessError(
        "review_item failed to produce a valid structured review after "
        f"{max_attempts} attempts: {last_error}",
        item_id=item_id,
        item_version=item_version,
        review_request_id=review_request_id,
        attempt_count=max_attempts,
    )


async def execute_virtual_simulation(state: PSJTState) -> dict[str, Any]:
    """Freeze the live candidate pool, then simulate against that exact version."""

    candidate_bank_audit = audit_candidate_item_bank(state)
    freeze_update = build_item_bank_freeze_update(state)
    simulation_state = {**state, **freeze_update}
    result = await run_virtual_response_simulation(simulation_state)
    simulation_update = result.get("state_update")
    if not isinstance(simulation_update, Mapping):
        raise ValueError("虚拟作答没有返回有效的 state_update")
    return {
        **result,
        "state_update": {
            **freeze_update,
            "candidate_bank_audit": candidate_bank_audit,
            **dict(simulation_update),
        },
        "summary": (
            f"候选题库已冻结为版本 {freeze_update['item_bank_version']}；"
            + str(result.get("summary") or "虚拟作答完成")
        ),
    }


def _frozen_item_index(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the current frozen bank by item_id."""

    return {
        str(item.get("item_id")): item
        for item in (state.get("frozen_item_bank") or [])
        if isinstance(item, Mapping) and item.get("item_id")
    }


def _is_plateau_gap_eligible(disposition: Mapping[str, Any]) -> bool:
    """A candidate may be offered for A/B fill only after content review.

    Items still awaiting SME review or already eliminated stay outside the
    two-choice resolution and keep requiring a separate human decision.
    """

    return (
        not isinstance(disposition, Mapping)
        or disposition.get("status") not in {"pending_sme_review", "eliminated"}
    )


def build_plateau_gap_decision_payload(
    state: Mapping[str, Any],
    blueprint_coverage: Mapping[str, Any],
) -> dict[str, Any] | None:
    """List, for every uncovered blueprint cell, its resolvable candidates."""

    gap_cells = [
        row
        for row in blueprint_coverage.get("cells") or []
        if isinstance(row, Mapping) and not row.get("passed")
    ]
    if not gap_cells:
        return None
    items = _frozen_item_index(state)
    statistics = state.get("item_statistics") or {}
    dispositions = state.get("item_final_dispositions") or {}
    output_cells: list[dict[str, Any]] = []
    for cell in gap_cells:
        cell_id = str(cell.get("blueprint_cell_id") or "")
        candidates: list[dict[str, Any]] = []
        for item in items.values():
            if str(item.get("blueprint_cell_id") or "") != cell_id:
                continue
            item_id = str(item.get("item_id") or "")
            dis = dispositions.get(item_id) or {}
            quality = (statistics.get(item_id) or {}).get("quality_evaluation") or {}
            candidates.append(
                {
                    "item_id": item_id,
                    "version": item.get("version"),
                    "disposition_status": str(dis.get("status") or ""),
                    "eligible": _is_plateau_gap_eligible(dis),
                    # pending-SME candidates may still be force-resolved by an
                    # explicit user override (developmental fill), but they are
                    # never offered as ordinary A/B choices.
                    "force_allowed": (
                        str(dis.get("status") or "") == "pending_sme_review"
                    ),
                    "failed_gates": quality.get("failed_gates") or [],
                    "recommendation": quality.get("recommendation"),
                    "facet_citc": quality.get("facet_citc"),
                    "difficulty": (statistics.get(item_id) or {}).get(
                        "difficulty"
                    ),
                    "scenario": item.get("scenario"),
                    "response_options": [
                        {
                            "option_id": option.get("option_id"),
                            "text": option.get("text"),
                        }
                        for option in (item.get("response_options") or [])
                        if isinstance(option, Mapping)
                        and option.get("option_id")
                    ],
                }
            )
        output_cells.append(
            {
                "blueprint_cell_id": cell_id,
                "planned_retention_count": cell.get("planned_retention_count"),
                "missing_count": cell.get("missing_count"),
                "candidates": candidates,
            }
        )
    return {"status": "pending", "gap_cells": output_cells}


def apply_plateau_gap_fills(
    state: Mapping[str, Any],
    fills: Mapping[str, Any],
    retained: list[dict[str, Any]],
    selected_items: list[dict[str, Any]],
    blueprint_coverage: Mapping[str, Any],
    dispositions: Mapping[str, Any],
    reasons: Mapping[str, Any],
    provisional_flags: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Promote user-selected plateau-gap candidates as provisional fills.

    Only content-reviewed candidates (not SME/eliminated) are accepted. A
    manually edited payload replaces the frozen text; a plain pick reuses the
    frozen candidate. Every promoted item is flagged provisional so the final
    report is produced as a developmental version.
    """

    retained_out = [deepcopy(dict(item)) for item in retained]
    selected_out = [deepcopy(dict(item)) for item in selected_items]
    selected_ids = {
        str(item.get("item_id")) for item in selected_out if item.get("item_id")
    }
    retained_ids = {
        str(item.get("item_id")) for item in retained_out if item.get("item_id")
    }
    items = _frozen_item_index(state)
    dispositions_out = {
        str(key): deepcopy(dict(value))
        for key, value in dispositions.items()
        if isinstance(value, Mapping)
    }
    reasons_out = {
        str(key): deepcopy(value) for key, value in reasons.items()
    }
    flags_out = {
        str(key): deepcopy(dict(value))
        for key, value in provisional_flags.items()
        if isinstance(value, Mapping)
    }
    coverage_cells = [
        deepcopy(dict(cell))
        for cell in blueprint_coverage.get("cells") or []
        if isinstance(cell, Mapping)
    ]
    by_cell = {str(cell["blueprint_cell_id"]): cell for cell in coverage_cells}
    for cell_id, fill in fills.items():
        if not isinstance(fill, Mapping):
            continue
        cell = by_cell.get(str(cell_id))
        if cell is None or cell.get("passed"):
            continue
        item_id = str(fill.get("item_id") or "")
        if item_id not in items or item_id in selected_ids:
            continue
        dis = dispositions_out.get(item_id) or {}
        dis_status = str(dis.get("status") or "")
        sme_override = bool(fill.get("sme_override"))
        if dis_status == "eliminated":
            continue
        if dis_status == "pending_sme_review" and not sme_override:
            continue
        mode = str(fill.get("mode") or "pick")
        if mode == "manual" and isinstance(fill.get("edited_item"), Mapping):
            chosen = deepcopy(dict(fill["edited_item"]))
        else:
            chosen = deepcopy(dict(items[item_id]))
        selected_out.append(chosen)
        selected_ids.add(item_id)
        if item_id not in retained_ids:
            retained_out.append(deepcopy(dict(chosen)))
            retained_ids.add(item_id)
        if dis_status == "pending_sme_review" and sme_override:
            disposition_reason = "user_forced_sme_plateau_fill"
            reason_text = "平台期补位：该题仍在待SME，用户以开发版证据强制补位收卷。"
        else:
            disposition_reason = "user_resolved_plateau_gap"
            reason_text = (
                "平台期补位：用户选定该题并以开发版证据进入正式卷。"
                if mode == "pick"
                else "平台期补位：用户手动修改后以开发版证据进入正式卷。"
            )
        dispositions_out[item_id] = {
            "status": "provisional_plateau_fill",
            "item_version": chosen.get("version"),
            "mode": mode,
            "reason": disposition_reason,
        }
        reasons_out[item_id] = reason_text
        flags_out[item_id] = {
            "reason": "plateau_gap_provisional_fill",
            "mode": mode,
            "item_version": chosen.get("version"),
            "sme_override": sme_override,
        }
    for cell in coverage_cells:
        cell_id = str(cell["blueprint_cell_id"])
        cell["selected_count"] = sum(
            1
            for item in selected_out
            if str(item.get("blueprint_cell_id") or "") == cell_id
        )
        planned = int(cell.get("planned_retention_count") or 0)
        cell["passed"] = bool(
            cell["selected_count"] >= planned
        )
    coverage_out = {
        **dict(blueprint_coverage),
        "cells": coverage_cells,
        "passed": all(bool(cell.get("passed")) for cell in coverage_cells),
        "selected_total": len(selected_out),
        "available_total": len(retained_out),
    }
    return (
        retained_out,
        selected_out,
        coverage_out,
        dispositions_out,
        reasons_out,
        flags_out,
    )


async def execute_item_selection_with_diagnosis(
    state: PSJTState,
) -> dict[str, Any]:
    """Diagnose flagged items and queue all confirmed edits for one item."""
    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("心理测量诊断前缺少冻结题库")
    statistics = state.get("item_statistics") or {}
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    defer_after_rounds = PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS
    prior_fingerprints = {
        str(event.get("diagnosis_fingerprint"))
        for event in state.get("psychometric_repair_history") or []
        if isinstance(event, Mapping) and event.get("diagnosis_fingerprint")
    }
    item_by_id: dict[str, dict[str, Any]] = {}
    for raw_item in frozen:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = deepcopy(dict(raw_item))
        item_id = str(item.get("item_id") or "")
        if not item_id or item_id in item_by_id:
            raise ValueError("冻结题库 item_id 缺失或重复")
        item_by_id[item_id] = item

    iteration_history = [
        deepcopy(dict(row))
        for row in state.get("psychometric_iteration_history") or []
        if isinstance(row, Mapping)
    ]
    current_iteration = int(state.get("psychometric_analysis_round") or 0)
    provisional_iteration: dict[str, Any] | None = next(
        (
            row
            for row in iteration_history
            if int(row.get("analysis_round") or 0) == current_iteration
        ),
        None,
    )
    if current_iteration > 0 and provisional_iteration is None:
        provisional_iteration = await _build_provisional_iteration_record(
            state,
            list(item_by_id.values()),
        )
    plateau_history = [
        row
        for row in iteration_history
        if int(row.get("analysis_round") or 0) != current_iteration
    ]
    if provisional_iteration is not None:
        plateau_history.append(provisional_iteration)
    plateau_status = assess_form_plateau(
        plateau_history,
        patience=int(
            state.get("psychometric_plateau_patience")
            or PLATEAU_DEFAULT_PATIENCE
        ),
        min_delta=float(
            state.get("psychometric_plateau_min_delta")
            if state.get("psychometric_plateau_min_delta") is not None
            else PLATEAU_DEFAULT_MIN_DELTA
        ),
    )
    plateau_reached = bool(plateau_status.get("reached"))
    if provisional_iteration is not None:
        provisional_iteration = {
            **deepcopy(dict(provisional_iteration)),
            "plateau_status": deepcopy(plateau_status),
        }
    provisional_form_optimizer = (
        provisional_iteration.get("form_optimizer")
        if isinstance(provisional_iteration, Mapping)
        else None
    )
    existing_queue_entries = {
        str(entry.get("item_id")): deepcopy(dict(entry))
        for entry in [
            *(state.get("items_to_revise") or []),
            *(state.get("items_to_regenerate") or []),
        ]
        if isinstance(entry, Mapping) and entry.get("item_id") is not None
    }
    queued_item_ids = set(existing_queue_entries)
    continuing_existing_batch = bool(queued_item_ids)

    retained: list[dict[str, Any]] = []
    # Outer queue: keep every statistically abnormal item here.  The first
    # pass diagnoses the full queue so the next action can dispatch one
    # isolated subagent per repairable item.
    repair_queue: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    dispositions: dict[str, dict[str, Any]] = deepcopy(
        state.get("item_final_dispositions") or {}
    )
    locked_versions: dict[str, int] = {
        str(item_id): int(version)
        for item_id, version in (state.get("locked_retained_item_versions") or {}).items()
        if isinstance(version, int) and not isinstance(version, bool)
    }
    monitoring_warnings: list[dict[str, Any]] = []
    reasons: dict[str, str] = deepcopy(state.get("selection_reasons") or {})
    diagnoses: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    diagnosis_call_count = 0
    diagnosis_events: list[dict[str, Any]] = []
    diagnosis_jobs: list[dict[str, Any]] = []
    existing_confirmation = state.get("psychometric_repair_confirmation")
    if (
        isinstance(existing_confirmation, Mapping)
        and existing_confirmation.get("status") == "pending"
    ):
        pending_item_id = str(existing_confirmation.get("item_id") or "")
        pending_entry = next(
            (
                deepcopy(entry)
                for entry in state.get("items_to_revise") or []
                if isinstance(entry, Mapping)
                and str(entry.get("item_id")) == pending_item_id
            ),
            None,
        )
        if pending_entry is not None:
            return {
                "state_update": {
                    "selection_results": None,
                    "psychometric_repair_confirmation": deepcopy(
                        dict(existing_confirmation)
                    ),
                    "items_to_revise": [
                        deepcopy(entry)
                        for entry in state.get("items_to_revise") or []
                        if isinstance(entry, Mapping)
                    ],
                    "items_to_regenerate": [
                        deepcopy(entry)
                        for entry in state.get("items_to_regenerate") or []
                        if isinstance(entry, Mapping)
                    ],
                    "selected_items": [],
                    "reserve_items": [],
                },
                "summary": f"等待用户确认单题返修：{pending_item_id}",
            }

    def accept(
        item_id: str,
        item: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        retained.append(deepcopy(dict(item)))
        item_version = int(item.get("version") or 0)
        locked_versions[item_id] = item_version
        dispositions[item_id] = {
            "status": "qualified_locked",
            "warning_reason": None,
            "item_version": item_version,
            "qualified_at_repair_round": int(rounds.get(item_id, 0)),
            "qualification_snapshot": deepcopy(statistics.get(item_id) or {}),
        }
        reasons[item_id] = reason

    for item_id, item in item_by_id.items():
        if plateau_reached:
            existing_disposition = dispositions.get(item_id)
            if isinstance(existing_disposition, Mapping) and existing_disposition.get(
                "status"
            ) in {"pending_sme_review", "eliminated"}:
                continue
            retained.append(deepcopy(item))
            item_version = int(item.get("version") or 0)
            locked_versions[item_id] = item_version
            dispositions[item_id] = {
                "status": "qualified_locked",
                "warning_reason": "整卷指标达到平台期，停止继续自动返修。",
                "item_version": item_version,
                "qualified_at_repair_round": int(rounds.get(item_id, 0)),
                "qualification_snapshot": deepcopy(statistics.get(item_id) or {}),
                "monitoring_pass": False,
                "monitoring_metrics": deepcopy(statistics.get(item_id) or {}),
            }
            reasons[item_id] = "整卷指标达到平台期，保留当前最佳组卷候选。"
            continue
        if continuing_existing_batch and item_id not in queued_item_ids:
            # A repaired/eliminated item has already been handled within this
            # analysis batch. Continue diagnosing only the remaining baseline
            # queue; all changed items are re-measured together after it drains.
            continue
        existing_queue_entry = existing_queue_entries.get(item_id)
        if (
            continuing_existing_batch
            and isinstance(existing_queue_entry, Mapping)
            and isinstance(
                existing_queue_entry.get("atomic_repair_advice"), Mapping
            )
        ):
            # This item has already been diagnosed in the current outer
            # batch. Preserve that repair/defer decision exactly. Rebuilding
            # it from the same item version and statistics would hit the
            # duplicate-fingerprint guard and incorrectly turn a valid repair
            # into a defer decision.
            preserved_entry = deepcopy(dict(existing_queue_entry))
            repair_queue.append(preserved_entry)
            repairs.append(preserved_entry)
            continue
        existing_disposition = dispositions.get(item_id)
        if (
            isinstance(existing_disposition, Mapping)
            and existing_disposition.get("status")
            in {"pending_sme_review", "eliminated"}
        ):
            continue
        item_statistics = statistics.get(item_id) or {}
        completed_rounds = int(rounds.get(item_id, 0))
        if locked_versions.get(item_id) == int(item.get("version") or 0):
            retained.append(deepcopy(item))
            monitored_pass = not item_requires_psychometric_diagnosis(item_statistics)
            dispositions[item_id] = {
                **deepcopy(dict(existing_disposition or {})),
                "status": "qualified_locked",
                "item_version": item.get("version"),
                "monitoring_pass": monitored_pass,
                "monitoring_metrics": deepcopy(item_statistics),
            }
            if not monitored_pass:
                monitoring_warnings.append(
                    {
                        "item_id": item_id,
                        "item_version": item.get("version"),
                        "statistics": deepcopy(item_statistics),
                        "message": "锁定正式题的最新监测指标未通过；资格不撤销且不返修。",
                    }
                )
            continue
        if not item_requires_psychometric_diagnosis(item_statistics):
            accept(
                item_id,
                item,
                reason=(
                    "合并分数档后的CITC、条件目标相关、同域条件VTS与跨域条件VTS"
                    "均达到虚拟迭代阈值。"
                ),
            )
            continue
        if completed_rounds >= defer_after_rounds:
            queue_entry = _psychometric_repair_entry(
                item=item,
                statistics=item_statistics,
                revision_round=completed_rounds + 1,
            )
            queue_entry.update(
                {
                    "action": "defer",
                    "queue_status": "deferred_decision",
                    "atomic_repair_advice": {
                        "decision": "defer",
                        "summary": (
                            f"已完成 {completed_rounds} 轮返修仍未达标，"
                            "自动进入同一蓝图槽位补题队列。"
                        ),
                        "observed_discrepancies": [],
                        "candidate_diagnoses": [],
                        "repair_tasks": [],
                    },
                    "diagnosis_evidence": build_construct_diagnosis_evidence(
                        state, item_id, revision_round=completed_rounds + 1
                    ),
                    "completed_repair_rounds": completed_rounds,
                    "defer_after_rounds": defer_after_rounds,
                    "diagnosis_status": "repair_rounds_exhausted",
                }
            )
            repair_queue.append(queue_entry)
            repairs.append(queue_entry)
            continue
        queue_entry = _psychometric_repair_entry(
            item=item,
            statistics=item_statistics,
            revision_round=completed_rounds + 1,
        )
        queue_entry["queue_status"] = "pending_diagnosis"
        repair_queue.append(queue_entry)
        evidence = build_construct_diagnosis_evidence(
            state,
            item_id,
            revision_round=completed_rounds + 1,
        )
        fingerprint = diagnosis_fingerprint(evidence)
        evidence["diagnosis_fingerprint"] = fingerprint
        fingerprints[item_id] = fingerprint
        if fingerprint in prior_fingerprints:
            duplicate_advice = {
                "decision": "defer",
                "summary": "相同题目版本和统计指纹已经诊断，自动转为同槽位补题。",
                "observed_discrepancies": [],
                "candidate_diagnoses": [],
                "repair_tasks": [],
            }
            repair_queue[-1].update(
                {
                    "action": "defer",
                    "queue_status": "deferred_decision",
                    "atomic_repair_advice": duplicate_advice,
                    "diagnosis_evidence": evidence,
                    "diagnosis_fingerprint": fingerprint,
                    "diagnosis_status": "duplicate_fingerprint",
                }
            )
            repairs.append(repair_queue[-1])
            continue
        diagnosis_jobs.append(
            {
                "item_id": item_id,
                "item": deepcopy(item),
                "completed_rounds": completed_rounds,
                "queue_index": len(repair_queue) - 1,
                "evidence": deepcopy(evidence),
                "fingerprint": fingerprint,
            }
        )

    if diagnosis_jobs:
        diagnosis_concurrency = max(
            1,
            min(
                8,
                int(
                    state.get("psychometric_diagnosis_concurrency")
                    or PSYCHOMETRIC_REPAIR_SUBAGENT_CONCURRENCY
                ),
            ),
        )
        diagnosis_batch_id = (
            f"psychometric-diagnosis-batch/"
            f"{current_iteration}/{len(diagnosis_events) + 1}"
        )
        emit_progress(
            {
                "type": "psychometric_subagent_progress",
                "status": "batch_started",
                "batch_id": diagnosis_batch_id,
                "batch_total": len(diagnosis_jobs),
                "concurrency": diagnosis_concurrency,
                "message": (
                    f"启动 {len(diagnosis_jobs)} 个心理测量诊断任务，"
                    f"最大并发 {diagnosis_concurrency}；全部完成后统一形成返修队列"
                ),
            }
        )
        diagnosis_semaphore = asyncio.Semaphore(diagnosis_concurrency)

        async def diagnose_one(job: Mapping[str, Any]) -> dict[str, Any]:
            item_id = str(job["item_id"])
            evidence = job["evidence"]
            queue_position = int(job["queue_index"]) + 1
            emit_progress(
                {
                    "type": "psychometric_subagent_progress",
                    "status": "started",
                    "batch_id": diagnosis_batch_id,
                    "item_id": item_id,
                    "queue_position": queue_position,
                    "queue_total": len(diagnosis_jobs),
                    "message": "开始生成心理测量返修诊断",
                }
            )
            async with diagnosis_semaphore:
                try:
                    diagnosis = await _ainvoke_model(
                        psychometric_repair_diagnosis_agent,
                        {
                            "input_data": build_psychometric_agent_input(
                                evidence
                            )
                        },
                        job_label=f"psychometric_repair_diagnosis / {item_id}",
                        timeout_seconds=PSYCHOMETRIC_REPAIR_TIMEOUT_SECONDS,
                        max_attempts=1,
                    )
                    if isinstance(diagnosis, Mapping) and not diagnosis.get(
                        "item_id"
                    ):
                        diagnosis = {**dict(diagnosis), "item_id": item_id}
                    diagnosis_status = "completed"
                    diagnosis_validation_error = None
                    try:
                        validate_atomic_repair_advice(diagnosis, evidence)
                    except ValueError as exc:
                        fallback = build_deterministic_forced_vts_repair_advice(
                            evidence,
                            validation_error=str(exc),
                        )
                        fallback_status = "deterministic_forced_vts_fallback"
                        if fallback is None:
                            fallback = build_deterministic_target_gradient_repair_advice(
                                evidence,
                                validation_error=str(exc),
                            )
                            fallback_status = "deterministic_target_gradient_fallback"
                        if fallback is None:
                            # Ordinary VTS repairs require a literal quote from
                            # the current item and a matching NON_TARGET
                            # constraint.  If the model does not provide that
                            # evidence, do not guess a patch and do not stop
                            # the whole concurrent batch; defer only this item.
                            fallback = build_deterministic_defer_advice(
                                evidence,
                                validation_error=str(exc),
                            )
                            fallback_status = "validation_fallback_defer"
                        if fallback is None:
                            raise
                        diagnosis = fallback
                        validate_atomic_repair_advice(diagnosis, evidence)
                        diagnosis_status = fallback_status
                        diagnosis_validation_error = str(exc)
                    if diagnosis.get("decision") == "repair":
                        try:
                            diagnosis = normalize_target_gradient_repair_advice(
                                diagnosis,
                                evidence,
                            )
                            validate_atomic_repair_advice(
                                diagnosis,
                                evidence,
                                require_target_gradient_task=True,
                            )
                        except ValueError as exc:
                            # Normalization adds the mandatory target-gradient
                            # preflight.  If that second validation reveals an
                            # invalid ordinary repair link, keep the same safe
                            # per-item defer behavior instead of aborting the
                            # entire diagnosis batch.
                            fallback = build_deterministic_defer_advice(
                                evidence,
                                validation_error=str(exc),
                            )
                            if fallback is None:
                                raise
                            diagnosis = fallback
                            validate_atomic_repair_advice(diagnosis, evidence)
                            diagnosis_status = "validation_fallback_defer"
                            diagnosis_validation_error = str(exc)
                    emit_progress(
                        {
                            "type": "psychometric_subagent_progress",
                            "status": "completed",
                            "batch_id": diagnosis_batch_id,
                            "item_id": item_id,
                            "queue_position": queue_position,
                            "queue_total": len(diagnosis_jobs),
                            "message": (
                                "心理测量诊断完成；决策="
                                f"{diagnosis.get('decision')}"
                            ),
                        }
                    )
                    return {
                        "item_id": item_id,
                        "diagnosis": deepcopy(dict(diagnosis)),
                        "diagnosis_status": diagnosis_status,
                        "diagnosis_validation_error": diagnosis_validation_error,
                        "error": None,
                    }
                except TimeoutError as exc:
                    # A timeout means that no diagnosis was received.  It is
                    # therefore unsafe to invent a repair, but it is also not
                    # necessary to abort all other independent items.  Keep
                    # the failed item for manual review and let the batch
                    # barrier continue with the remaining results.
                    fallback = build_deterministic_defer_advice(
                        evidence,
                        validation_error=str(exc),
                    )
                    if fallback is None:
                        emit_progress(
                            {
                                "type": "psychometric_subagent_progress",
                                "status": "failed",
                                "batch_id": diagnosis_batch_id,
                                "item_id": item_id,
                                "queue_position": queue_position,
                                "queue_total": len(diagnosis_jobs),
                                "message": f"心理测量诊断超时且无法安全降级：{exc}",
                            }
                        )
                        return {
                            "item_id": item_id,
                            "diagnosis": None,
                            "diagnosis_status": "failed",
                            "diagnosis_validation_error": None,
                            "error": str(exc),
                        }
                    validate_atomic_repair_advice(fallback, evidence)
                    emit_progress(
                        {
                            "type": "psychometric_subagent_progress",
                            "status": "completed",
                            "batch_id": diagnosis_batch_id,
                            "item_id": item_id,
                            "queue_position": queue_position,
                            "queue_total": len(diagnosis_jobs),
                            "message": "心理测量诊断请求超时，本题已自动转为同槽位补题",
                        }
                    )
                    return {
                        "item_id": item_id,
                        "diagnosis": deepcopy(dict(fallback)),
                        "diagnosis_status": "timeout_fallback_defer",
                        "diagnosis_validation_error": str(exc),
                        "error": None,
                    }
                except Exception as exc:
                    # 单题诊断异常：优先安全转 defer（后续自动同槽位补题），
                    # 不让单题故障停整批；仅当连 defer 兜底都不可用时才标记该题失败。
                    try:
                        fallback = build_deterministic_defer_advice(
                            evidence,
                            validation_error=str(exc),
                        )
                    except Exception:
                        fallback = None
                    if fallback is not None:
                        emit_progress(
                            {
                                "type": "psychometric_subagent_progress",
                                "status": "completed",
                                "batch_id": diagnosis_batch_id,
                                "item_id": item_id,
                                "queue_position": queue_position,
                                "queue_total": len(diagnosis_jobs),
                                "message": (
                                    "心理测量诊断异常，本题已安全转为 defer"
                                ),
                            }
                        )
                        return {
                            "item_id": item_id,
                            "diagnosis": deepcopy(dict(fallback)),
                            "diagnosis_status": "exception_fallback_defer",
                            "diagnosis_validation_error": str(exc),
                            "error": None,
                        }
                    emit_progress(
                        {
                            "type": "psychometric_subagent_progress",
                            "status": "failed",
                            "batch_id": diagnosis_batch_id,
                            "item_id": item_id,
                            "queue_position": queue_position,
                            "queue_total": len(diagnosis_jobs),
                            "message": f"心理测量诊断失败：{exc}",
                        }
                    )
                    return {
                        "item_id": item_id,
                        "diagnosis": None,
                        "diagnosis_status": "failed",
                        "diagnosis_validation_error": None,
                        "error": str(exc),
                    }

        diagnosis_results = await asyncio.gather(
            *(diagnose_one(job) for job in diagnosis_jobs),
            return_exceptions=False,
        )
        diagnosis_failures = [
            result
            for result in diagnosis_results
            if result.get("error")
        ]
        successful_count = len(diagnosis_results) - len(diagnosis_failures)
        if diagnosis_failures and successful_count == 0:
            # 全部失败 = 服务级故障（模型端点/配置问题），保留 checkpoint 停止，
            # 避免把系统性故障伪装成逐题 defer 空转。
            first_failure = diagnosis_failures[0]
            raise PsychometricDiagnosisUnavailable(
                "心理测量返修诊断不可用，已停止自动返修队列；"
                f"题目 {first_failure.get('item_id')} 的诊断未完成："
                f"{first_failure.get('error')}"
            )
        # 部分失败：失败题保留在待诊断队列（下一轮再试），
        # 其余题目照常进入返修队列，不再因单题故障停整批。

        diagnosis_call_count = len(diagnosis_results)
        for job, result in zip(diagnosis_jobs, diagnosis_results, strict=True):
            item_id = str(job["item_id"])
            item = job["item"]
            completed_rounds = int(job["completed_rounds"])
            evidence = job["evidence"]
            fingerprint = str(job["fingerprint"])
            diagnosis = result["diagnosis"]
            if diagnosis is None:
                # 单题诊断失败：保留待诊断状态，下一轮再试，不阻塞其他题目。
                continue
            diagnosis_status = str(result["diagnosis_status"])
            diagnosis_validation_error = result.get(
                "diagnosis_validation_error"
            )
            diagnoses[item_id] = deepcopy(dict(diagnosis))
            diagnosis_event = {
                "event": "psychometric_item_diagnosed",
                "item_id": item_id,
                "item_version": item.get("version"),
                "revision_round": completed_rounds + 1,
                "diagnosis_fingerprint": fingerprint,
                "decision": diagnosis.get("decision"),
                "summary": diagnosis.get("summary"),
                "repair_task_count": len(repair_tasks_from_advice(diagnosis)),
                "diagnosis_status": diagnosis_status,
            }
            if diagnosis_validation_error is not None:
                diagnosis_event["validation_error"] = diagnosis_validation_error
            diagnosis_events.append(diagnosis_event)
            if diagnosis["decision"] == "defer":
                diagnosed_entry = {
                    "item_id": item_id,
                    "blueprint_cell_id": item.get("blueprint_cell_id"),
                    "target_dimension_id": item.get("target_dimension_id"),
                    "action": "defer",
                    "revision_round": completed_rounds + 1,
                    "atomic_repair_advice": deepcopy(dict(diagnosis)),
                    "diagnosis_evidence": deepcopy(evidence),
                    "diagnosis_fingerprint": fingerprint,
                    "diagnosis_status": diagnosis_status,
                    "queue_status": "deferred_decision",
                }
                if diagnosis_validation_error is not None:
                    diagnosed_entry["diagnosis_validation_error"] = diagnosis_validation_error
                repair_queue[int(job["queue_index"])] = diagnosed_entry
                repairs.append(diagnosed_entry)
                reasons[item_id] = str(
                    diagnosis.get("summary") or "证据不足，自动转为同槽位补题。"
                )
                continue
            diagnosed_entry = {
                "item_id": item_id,
                "blueprint_cell_id": item.get("blueprint_cell_id"),
                "target_dimension_id": item.get("target_dimension_id"),
                "action": "revise_item",
                "revision_round": completed_rounds + 1,
                "atomic_repair_advice": deepcopy(dict(diagnosis)),
                "diagnosis_evidence": deepcopy(evidence),
                "diagnosis_fingerprint": fingerprint,
                "diagnosis_status": diagnosis_status,
                "queue_status": "diagnosed",
            }
            if diagnosis_validation_error is not None:
                diagnosed_entry["diagnosis_validation_error"] = diagnosis_validation_error
            repair_queue[int(job["queue_index"])] = diagnosed_entry
            repairs.append(diagnosed_entry)
            reasons[item_id] = str(diagnosis.get("summary") or "进入原子返修。")

    pending_sme = [
        item_id for item_id, disposition in dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "pending_sme_review"
    ]
    # 平台期收卷：即使存在待 SME 题或蓝图缺口，也进入 ready，
    # 用当前合格题由组卷Agent重新选最优组合收卷（历史 best 轮的旧组合
    # 可能已被后续返修淘汰，不能直接沿用）。
    plateau_finalized = plateau_reached
    status = (
        "repair_confirmation_required"
        if repairs
        else "diagnosis_pending"
        if repair_queue
        else "awaiting_sme_review"
        if (pending_sme and not plateau_finalized)
        else "ready_for_assembly"
    )
    coverage_cells: list[dict[str, Any]] = []
    coverage_passed = True
    item_counts: dict[str, int] = {}
    for item in retained:
        cell_id = item.get("blueprint_cell_id")
        if isinstance(cell_id, str):
            item_counts[cell_id] = item_counts.get(cell_id, 0) + 1
    for cell in (state.get("blueprint") or {}).get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("cell_id") or "")
        planned = int(cell.get("planned_retention_count") or 0)
        actual = item_counts.get(cell_id, 0)
        # In incremental development, retained is the candidate pool. A cell
        # passes coverage when it has at least the planned final count; the
        # form optimizer will later select the exact retained count.
        passed = actual >= planned
        coverage_passed = coverage_passed and passed
        coverage_cells.append(
            {
                "blueprint_cell_id": cell_id,
                "planned_retention_count": planned,
                "available_count": actual,
                "selected_count": min(actual, planned),
                "missing_count": max(0, planned - actual),
                "passed": passed,
            }
        )
    blueprint_coverage = {
        "passed": coverage_passed,
        "cells": coverage_cells,
        "expected_total": sum(
            int(cell.get("planned_retention_count") or 0)
            for cell in coverage_cells
        ),
        "selected_total": sum(
            int(cell.get("selected_count") or 0)
            for cell in coverage_cells
        ),
        "available_total": len(retained),
    }
    form_optimizer: dict[str, Any] | None = None
    selected_items: list[dict[str, Any]] = []
    reserve_items: list[dict[str, Any]] = []
    if plateau_finalized:
        # 平台期收卷：用当前合格题重新让组卷Agent选最优组合（绕过 SME/缺口卡点）
        if (
            isinstance(provisional_form_optimizer, Mapping)
            and provisional_form_optimizer.get("status") == "validated"
        ):
            form_optimizer = deepcopy(dict(provisional_form_optimizer))
        else:
            try:
                form_optimizer = await optimize_test_form_with_agent(
                    state,
                    retained,
                    statistics,
                    state.get("test_statistics")
                    if isinstance(state.get("test_statistics"), Mapping)
                    else None,
                )
            except Exception:
                form_optimizer = None
        if form_optimizer is not None and form_optimizer.get("selected_item_ids"):
            selected_set = set(form_optimizer["selected_item_ids"])
            form_optimizer = {
                **dict(form_optimizer),
                "mode": "plateau_finalized",
                "rationale": (
                    "平台期收卷：用当前合格题由组卷Agent重新选出的最优正式组合。"
                ),
            }
            selected_items = [
                deepcopy(item)
                for item in retained
                if str(item.get("item_id")) in selected_set
            ]
            reserve_items = [
                deepcopy(item)
                for item in retained
                if str(item.get("item_id")) not in selected_set
            ]
        else:
            # 兜底：每个蓝图 cell 取合格题；缺口 cell 直接从冻结题库候选补齐
            # （plateau 收卷接受 revise 题，不标开发版标记）。
            by_cell_retained: dict[str, list[dict[str, Any]]] = {}
            for item in retained:
                by_cell_retained.setdefault(
                    str(item.get("blueprint_cell_id") or ""), []
                ).append(item)
            by_cell_all: dict[str, list[dict[str, Any]]] = {}
            for item in state.get("frozen_item_bank") or []:
                if not isinstance(item, Mapping):
                    continue
                by_cell_all.setdefault(
                    str(item.get("blueprint_cell_id") or ""), []
                ).append(item)
            selected_items = []
            for cell in (state.get("blueprint") or {}).get("cells") or []:
                if not isinstance(cell, Mapping):
                    continue
                cell_id = str(cell.get("cell_id") or "")
                planned = int(cell.get("planned_retention_count") or 0)
                chosen = list(by_cell_retained.get(cell_id, []))
                if len(chosen) < planned:
                    chosen_ids = {str(i.get("item_id")) for i in chosen}
                    for candidate in by_cell_all.get(cell_id, []):
                        if len(chosen) >= planned:
                            break
                        if str(candidate.get("item_id")) not in chosen_ids:
                            chosen.append(candidate)
                            chosen_ids.add(str(candidate.get("item_id")))
                selected_items.extend(deepcopy(chosen[: max(0, planned)]))
            selected_item_ids = [
                str(i.get("item_id")) for i in selected_items
            ]
            reserve_items = [
                deepcopy(item)
                for item in retained
                if str(item.get("item_id")) not in set(selected_item_ids)
            ]
            form_optimizer = {
                "status": "validated",
                "mode": "plateau_finalized_partial_fallback",
                "selected_item_ids": selected_item_ids,
                "rationale": "平台期收卷（确定性补齐）：组卷Agent不可用或缺口cell无合格题，"
                "按蓝图单元从冻结题库候选补齐。",
                "theory_coverage_summary": "",
            }
        for coverage_cell in blueprint_coverage["cells"]:
            cell_id = str(coverage_cell["blueprint_cell_id"])
            coverage_cell["selected_count"] = sum(
                1
                for item in selected_items
                if item.get("blueprint_cell_id") == cell_id
            )
            coverage_cell["passed"] = (
                coverage_cell["selected_count"]
                == int(coverage_cell["planned_retention_count"])
            )
        blueprint_coverage["passed"] = all(
            bool(cell.get("passed"))
            for cell in blueprint_coverage["cells"]
        )
        blueprint_coverage["selected_total"] = len(selected_items)
        blueprint_coverage["available_total"] = len(retained)
    elif not repair_queue and not pending_sme and coverage_passed:
        if (
            isinstance(provisional_form_optimizer, Mapping)
            and provisional_form_optimizer.get("status") == "validated"
        ):
            form_optimizer = deepcopy(dict(provisional_form_optimizer))
        else:
            form_optimizer = await optimize_test_form_with_agent(
                state,
                retained,
                statistics,
                state.get("test_statistics")
                if isinstance(state.get("test_statistics"), Mapping)
                else None,
            )
        selected_ids = set(form_optimizer["selected_item_ids"])
        selected_items = [
            deepcopy(item)
            for item in retained
            if str(item.get("item_id")) in selected_ids
        ]
        reserve_items = [
            deepcopy(item)
            for item in retained
            if str(item.get("item_id")) not in selected_ids
        ]
        for coverage_cell in blueprint_coverage["cells"]:
            cell_id = str(coverage_cell["blueprint_cell_id"])
            coverage_cell["selected_count"] = sum(
                1
                for item in selected_items
                if item.get("blueprint_cell_id") == cell_id
            )
            coverage_cell["passed"] = (
                coverage_cell["selected_count"]
                == int(coverage_cell["planned_retention_count"])
            )
        blueprint_coverage["passed"] = all(
            bool(cell.get("passed"))
            for cell in blueprint_coverage["cells"]
        )
        blueprint_coverage["selected_total"] = len(selected_items)
        blueprint_coverage["available_total"] = len(retained)
    # Plateau auto-close must not silently proceed to assembly when a
    # blueprint cell still has no usable formal item (e.g. every candidate is
    # pending SME or eliminated).  When the user already provided resolutions
    # (plateau_gap_fills), consume them as provisional developmental fills;
    # otherwise pause and carry the resolvable candidate list so 2b can offer
    # the manual-edit / pick interaction instead of run_test_assembly raising.
    plateau_flags_update: dict[str, Any] | None = None
    plateau_gap_decision: dict[str, Any] | None = None
    if plateau_finalized and not blueprint_coverage.get("passed"):
        fills = state.get("plateau_gap_fills") or {}
        if fills:
            (
                retained,
                selected_items,
                blueprint_coverage,
                dispositions,
                reasons,
                plateau_flags_update,
            ) = apply_plateau_gap_fills(
                state,
                fills,
                retained,
                selected_items,
                blueprint_coverage,
                dispositions,
                reasons,
                state.get("provisional_item_flags") or {},
            )
        if not blueprint_coverage.get("passed"):
            status = "awaiting_plateau_gap"
            plateau_gap_decision = build_plateau_gap_decision_payload(
                state,
                blueprint_coverage,
            )
    if provisional_iteration is not None:
        iteration_history = _upsert_iteration_record(
            iteration_history,
            provisional_iteration,
            state=state,
        )
        iteration_history = _annotate_iteration_quality(
            iteration_history,
            plateau_status,
        )
    return {
        "state_update": {
            "psychometric_repair_defer_after_rounds": defer_after_rounds,
            "psychometric_plateau_status": plateau_status,
            "selected_items": selected_items,
            "reserve_items": reserve_items,
            "items_to_revise": repair_queue,
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "selection_results": None if repair_queue else {
                "status": status,
                "plateau_finalized": bool(plateau_finalized),
                "retained_count": len(retained),
                "repair_count": len(repair_queue),
                "selected_count": len(selected_items),
                "reserve_count": len(reserve_items),
                "psychometric_repair_diagnoses": diagnoses,
                "diagnosis_evidence_fingerprints": fingerprints,
                "diagnosis_call_count": diagnosis_call_count,
                "final_dispositions": dispositions,
                "model_manifest": deepcopy(PSYCHOMETRIC_REASONING_ROLE_MANIFEST),
                "next_effect": {
                    "repair_items": bool(repair_queue),
                    "reanalyze_after_bank_change": bool(repair_queue),
                },
                "form_optimizer": form_optimizer,
            },
            "blueprint_coverage": blueprint_coverage,
            "plateau_gap_decision": plateau_gap_decision,
            "plateau_gap_fills": None,
            "provisional_item_flags": (
                plateau_flags_update
                if plateau_flags_update is not None
                else state.get("provisional_item_flags") or {}
            ),
            "selection_reasons": reasons,
            "item_final_dispositions": dispositions,
            "locked_retained_item_versions": locked_versions,
            "psychometric_repair_confirmation": (
                {
                    "status": "pending",
                    "item_id": repair_queue[0]["item_id"],
                    "revision_round": repair_queue[0]["revision_round"],
                    "queue_status": repair_queue[0].get("queue_status"),
                    "diagnosis_status": repair_queue[0].get("diagnosis_status"),
                    "completed_repair_rounds": repair_queue[0].get(
                        "completed_repair_rounds"
                    ),
                    "defer_after_rounds": repair_queue[0].get("defer_after_rounds"),
                    "diagnosis_fingerprint": repair_queue[0].get("diagnosis_fingerprint"),
                    "atomic_repair_advice": deepcopy(
                        repair_queue[0].get("atomic_repair_advice")
                    ),
                    "diagnosis_evidence": deepcopy(
                        repair_queue[0].get("diagnosis_evidence")
                    ),
                }
                if repairs and isinstance(repair_queue[0].get("atomic_repair_advice"), Mapping)
                else None
            ),
            "psychometric_repair_history": [
                *deepcopy(state.get("psychometric_repair_history") or []),
                *diagnosis_events,
            ],
            "item_pool": (
                [
                    deepcopy(dict(item))
                    for item in state.get("item_pool") or []
                    if isinstance(item, Mapping)
                ]
                if continuing_existing_batch
                else [deepcopy(item) for item in item_by_id.values()]
            ),
            "psychometric_monitoring_warnings": monitoring_warnings,
            "psychometric_iteration_history": iteration_history,
        },
        "summary": (
            f"构念约束诊断完成：当前保留 {len(retained)} 题，"
            f"外层待处理队列 {len(repair_queue)} 题。"
            + (
                f"历史最优整卷质量连续 {plateau_status.get('non_improving_rounds', 0)} 轮未达到改善幅度，"
                "已自动进入平台期并停止继续返修。"
                if plateau_reached
                else ""
            )
            + (
                f"测验组合优化后选入 {len(selected_items)} 题，"
                f"保留 {len(reserve_items)} 题作为备用。"
                if form_optimizer is not None
                else ""
            )
        ),
    }


def _construct_domain_summary(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Project stable domain identity without repeating all facet content."""

    fields = (
        "inventory_id",
        "inventory_name",
        "inventory_version",
        "review_status",
        "selection_level",
        "domain_id",
        "domain_name",
        "domain_name_en",
        "profile_hash",
    )
    return {
        field: deepcopy(profile.get(field))
        for field in fields
        if profile.get(field) is not None
    }


async def _fill_compact_slots(
    state: PSJTState,
    blueprint: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    valid_seed: Mapping[str, Mapping[str, Any]] | None = None,
    target_ids: set[str] | None = None,
    attempts_used: int = 0,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Fill fixed slots one at a time without exposing their IDs to the model.

    Slot identity is workflow state, not generated content.  Each model call
    therefore returns one anonymous skeleton payload; this function attaches
    it to the active program-owned specification ID and performs all duplicate
    and schema validation locally.
    """

    all_cell_ids = {
        slot["specification_id"]
        for slot in blueprint.get("slots") or []
        if isinstance(slot, Mapping)
        and slot.get("blueprint_cell_id") == cell.get("cell_id")
    }
    targets = set(target_ids or all_cell_ids) & all_cell_ids
    valid = {
        key: deepcopy(dict(value))
        for key, value in (valid_seed or {}).items()
        if isinstance(value, Mapping)
    }
    profile = blueprint["construct_profile_snapshot"]
    slot_by_id = {
        str(slot["specification_id"]): slot
        for slot in blueprint.get("slots") or []
        if isinstance(slot, Mapping) and slot.get("specification_id")
    }
    slot_order = [
        str(slot["specification_id"])
        for slot in blueprint.get("slots") or []
        if isinstance(slot, Mapping)
        and slot.get("specification_id") in targets
    ]
    max_attempt_depth = max(1, attempts_used)
    for ordinal, specification_id in enumerate(slot_order, start=1):
        if specification_id in valid:
            continue
        slot = slot_by_id.get(specification_id)
        if not isinstance(slot, Mapping):
            raise ValueError(f"心理骨架槽位不存在：{specification_id}")
        candidate_reference = slot.get("candidate_reference")
        if not isinstance(candidate_reference, Mapping):
            candidate_reference = {
                "mechanism_id": cell["mechanism_id"],
                "situation_id": cell["situation_id"],
            }
        design = resolve_blueprint_design(
            blueprint,
            cell,
            candidate_reference=candidate_reference,
        )
        facet = deepcopy(design["facet"])
        facet.pop("behavior_evidence", None)
        behavior = deepcopy(design["behavior_evidence"])
        behavior.pop("source_item_ids", None)
        last_error: ValueError | None = None
        last_problems: list[str] = []
        for attempt in range(1, MAX_ITEM_SKELETON_ATTEMPTS + 1):
            max_attempt_depth = max(max_attempt_depth, attempt)
            input_data: dict[str, Any] = {
                "test_specification": state.get("test_specification"),
                "domain_summary": _construct_domain_summary(profile),
                "current_facet": facet,
                "behavior_evidence": behavior,
                "activation_mechanism": design["activation_mechanism"],
                "situation": design["situation"],
            }
            if last_error is not None or last_problems:
                input_data["validation_feedback"] = {
                    "problems": last_problems or [str(last_error)],
                    "attempt": attempt,
                    "max_attempts": MAX_ITEM_SKELETON_ATTEMPTS,
                    "instruction": (
                        "只修正当前匿名骨架；不要返回任何题号或映射键"
                    ),
                }
            try:
                result = await _ainvoke_model(
                    compact_skeleton_agent,
                    {"input_data": input_data},
                    job_label=(
                        f"心理骨架-{facet['facet_name']}-{ordinal}"
                    ),
                )
                update = result.get("state_update")
                if not isinstance(update, Mapping):
                    raise ValueError("骨架输出缺少 state_update")
                candidate = update.get("item_skeleton")
                if not isinstance(candidate, Mapping):
                    raise ValueError("骨架输出必须包含一个 item_skeleton 对象")
            except ValueError as exc:
                last_error = exc
                last_problems = []
            else:
                check = classify_compact_skeletons(
                    blueprint,
                    {specification_id: candidate},
                )
                if specification_id in check["valid"]:
                    valid[specification_id] = deepcopy(
                        check["valid"][specification_id]
                    )
                    last_error = None
                    last_problems = []
                    break
                last_error = None
                last_problems = list(
                    check["invalid"].get(
                        specification_id,
                        ["当前骨架未通过程序校验"],
                    )
                )
            if attempt < MAX_ITEM_SKELETON_ATTEMPTS:
                emit_progress(
                    {
                        "type": "output_repair",
                        "retry_kind": (
                            _output_error_kind(last_error)
                            if last_error is not None
                            else "business_validation"
                        ),
                        "job_label": f"心理骨架-{facet['facet_name']}",
                        "attempt": attempt + 1,
                        "max_attempts": MAX_ITEM_SKELETON_ATTEMPTS,
                        "reason": (
                            str(last_error)
                            if last_error is not None
                            else "；".join(last_problems)
                        ),
                    }
                )
        else:
            details = last_problems or [str(last_error or "未知错误")]
            raise ValueError(
                "当前固定槽位经过 "
                f"{MAX_ITEM_SKELETON_ATTEMPTS} 次尝试仍未形成有效骨架："
                + "；".join(details)
            )
    return valid, max_attempt_depth


async def execute_fixed_blueprint(state: PSJTState) -> dict:
    """Prepare evidence, expansion, and the immutable two-way table."""

    specification = state.get("test_specification")
    if not isinstance(specification, Mapping):
        raise ValueError("建立题目计划前缺少 TestSpecification")
    profile = resolve_specification_profile(specification)
    corpus = load_ipip_corpus()
    bundles = {}
    for facet in profile["facets"]:
        facet_id = str(facet["facet_id"])
        bundles[facet_id] = await ensure_behavior_evidence(facet_id, corpus)
    profile = attach_behavior_evidence(profile, bundles)
    retention_total = int(specification["final_item_count"])
    generation_total = required_generation_total(retention_total)
    expansion_situation_total = required_expansion_situation_total(
        retention_total
    )
    situation_quotas = distribute_situation_quotas(
        expansion_situation_total, len(profile["facets"])
    )
    expansions = []
    for facet, required_situation_count in zip(
        profile["facets"], situation_quotas, strict=True
    ):
        expansions.append(
            await ensure_facet_expansion(
                run_id=state["run_id"],
                facet=facet,
                behavior_evidence=facet["behavior_evidence"],
                target_population=str(specification["target_population"]),
                output_language=str(specification["output_language"]),
                required_situation_count=required_situation_count,
            )
        )
    blueprint: dict[str, Any] | None = None
    errors: dict[str, str] = {}
    retry_feedback = ""
    for attempt in range(BLUEPRINT_SEMANTIC_RETRY_ATTEMPTS + 1):
        if attempt:
            emit_progress(
                {
                    "type": "output_repair",
                    "retry_kind": "blueprint_semantic",
                    "job_label": "双向细目表设计",
                    "attempt": attempt + 1,
                    "max_attempts": BLUEPRINT_SEMANTIC_RETRY_ATTEMPTS + 1,
                    "reason": retry_feedback,
                }
            )
        proposal = await propose_blueprint_rows(
            profile=profile,
            expansions=expansions,
            generation_total=generation_total,
            retention_total=retention_total,
            retry_feedback=retry_feedback,
        )
        try:
            candidate_blueprint = build_generation_blueprint(
                specification,
                profile,
                state["run_id"],
                expansions=expansions,
                proposal=proposal,
            )
            errors = validate_generation_blueprint(
                candidate_blueprint,
                specification,
            )
        except ValueError as exc:
            candidate_blueprint = None
            errors = {"blueprint": str(exc)}
        if not errors:
            blueprint = candidate_blueprint
            break
        retry_feedback = (
            "上一版蓝图候选未通过程序语义校验。请保留正确的构念、行为证据和"
            "候选情境范围，只修正以下问题；特别是不得复用已经在其他测量单元"
            "出现过的 mechanism_id/situation_id：\n"
            + format_blueprint_errors_for_user(errors)
        )
    if blueprint is None:
        raise ValueError(format_blueprint_errors_for_user(errors))
    return {
        "state_update": {
            "construct_profile": profile,
            "blueprint": blueprint,
        },
        "summary": (
            f"题目计划引用 {profile['inventory_name']} "
            f"{profile['domain_name']}，包含 {len(profile['facets'])} 个 "
            f"facet；情境扩展池固定 {expansion_situation_total} 个，"
            f"蓝图筛选 {planned_generation_count(blueprint)} 个候选槽位（每个测量单元 "
            f"{INCREMENTAL_CANDIDATES_PER_CELL} 个候选），"
            f"计划最终保留 {specification['final_item_count']} 题。"
        ),
        "repair_attempt_count": 0,
        "semantic_retry_count": attempt,
    }


async def _prepare_current_slot_skeleton(
    state: PSJTState,
) -> dict[str, Any]:
    """Generate one skeleton and apply program-owned deterministic checks."""

    blueprint = state.get("blueprint")
    item_specification = state.get("current_item_specification")
    if not isinstance(blueprint, Mapping) or not isinstance(
        item_specification, Mapping
    ):
        raise ValueError("当前题目槽缺少固定蓝图或槽位信息")
    specification_id = str(item_specification["specification_id"])
    existing = {
        str(key): deepcopy(dict(value))
        for key, value in (state.get("item_skeletons") or {}).items()
        if isinstance(value, Mapping)
    }
    skeleton = existing.get(specification_id)
    history = [
        deepcopy(dict(entry))
        for entry in (state.get("skeleton_review_history") or {}).get(
            specification_id, []
        )
        if isinstance(entry, Mapping)
    ]

    async def generate() -> dict[str, Any]:
        seed = {
            key: value for key, value in existing.items()
            if key != specification_id
        }
        cell = next(
            candidate
            for candidate in blueprint["cells"]
            if candidate["cell_id"]
            == item_specification["blueprint_cell_id"]
        )
        generated, _ = await _fill_compact_slots(
            state,
            blueprint,
            cell,
            valid_seed=seed,
            target_ids={specification_id},
        )
        return generated[specification_id]

    if skeleton is None:
        try:
            skeleton = await generate()
        except ValueError as exc:
            history.append(
                {
                    "round": len(history) + 1,
                    "mode": "program_validation_failed",
                    "skeleton": None,
                    "validation_error": str(exc),
                }
            )
            emit_progress(
                {
                    "type": "skeleton_slot_failed",
                    "specification_id": specification_id,
                    "history": deepcopy(history),
                    "final_reason": str(exc),
                }
            )
            raise SkeletonDevelopmentFailure(
                specification_id,
                history,
                str(exc),
            ) from exc

    history.append(
        {
            "round": len(history) + 1,
            "mode": "program_validation_passed",
            "skeleton": deepcopy(dict(skeleton)),
            "validation_summary": (
                "Schema、固定槽位映射及基础重复规则通过；"
                "未执行独立 LLM 骨架审核"
            ),
        }
    )

    committed_skeletons = {**existing, specification_id: skeleton}
    rows = materialize_item_specifications(
        blueprint,
        {specification_id: skeleton},
    )
    if len(rows) != 1:
        raise ValueError("当前心理骨架无法映射为唯一题目规格")
    current_row = rows[0]
    all_rows = [
        deepcopy(dict(row))
        for row in state.get("item_specifications") or []
        if isinstance(row, Mapping)
        and row.get("specification_id") != specification_id
    ]
    all_rows.append(current_row)
    return {
        "item_skeletons": committed_skeletons,
        "skeleton_reviews": {
            key: value
            for key, value in (state.get("skeleton_reviews") or {}).items()
            if key != specification_id
        },
        "skeleton_review_history": {
            **(state.get("skeleton_review_history") or {}),
            specification_id: history,
        },
        "item_specifications": all_rows,
        "current_item_specification": current_row,
    }


def _build_option_evidence_for_repair(
    state: Mapping[str, Any],
    active_psychometric_repair: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Extract per-option psychometric evidence for the repair LLM."""
    if not isinstance(active_psychometric_repair, Mapping):
        return None
    statistics = (state.get("item_statistics") or {}).get(
        active_psychometric_repair.get("item_id")
    ) or {}
    option_stats = statistics.get("option_statistics") or {}
    if not option_stats:
        return None
    result: list[dict[str, Any]] = []
    for option_id in sorted(option_stats):
        opt_stat = option_stats.get(option_id) or {}
        result.append({
            "option_id": option_id,
            "score": opt_stat.get("score"),
        })
    return result


async def _run_psychometric_local_retest(
    *,
    state: PSJTState,
    candidate_item: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one candidate-only administration and return the four item gates."""

    simulation = await run_single_item_virtual_retest(
        state,
        candidate_item,
    )
    metrics = evaluate_single_item_candidate(
        state,
        candidate_item,
        simulation.get("records") or [],
    )
    return {
        "simulation": {
            key: value
            for key, value in simulation.items()
            if key != "records"
        },
        "metrics": metrics,
    }


async def _execute_psychometric_repair_item(
    *,
    action: str,
    route: PSJTRouteDecision,
    state: PSJTState,
    item_specification: Mapping[str, Any] | None,
    active_psychometric_repair: Mapping[str, Any],
    atomic_advice: Mapping[str, Any],
    diagnosis_evidence: Mapping[str, Any],
    program_update: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Apply every confirmed task for one item before any re-simulation.

    Each task still gets its own narrowly scoped repair-model call.  The
    intermediate item is kept local until all tasks succeed, so a failed task
    cannot leave a partially repaired item in the workflow.  The caller then
    commits one new item version and clears statistics once, which triggers one
    subsequent virtual re-test for the completed batch.
    """

    effective_advice = deepcopy(dict(atomic_advice))
    if (
        isinstance(active_psychometric_repair.get("atomic_repair_advice"), Mapping)
        and diagnosis_evidence
    ):
        effective_advice = normalize_target_gradient_repair_advice(
            effective_advice,
            diagnosis_evidence,
        )
        validate_atomic_repair_advice(
            effective_advice,
            diagnosis_evidence,
            require_target_gradient_task=True,
        )
    tasks = repair_tasks_from_advice(effective_advice)
    if not tasks:
        return None
    working_item = deepcopy(state.get("current_item") or {})
    if not working_item:
        raise ValueError("psychometric repair batch requires current_item")
    base_version = int(working_item.get("version") or 0)
    agent_packet = build_psychometric_agent_input(diagnosis_evidence)
    normal_constraints = agent_packet.get("normal_constraints")
    target_construct_constraints = agent_packet.get("target_construct_constraints")
    item_content = agent_packet.get("item_content")
    option_evidence = agent_packet.get("option_evidence")
    option_score_comparisons = agent_packet.get("option_score_comparisons")
    target_gradient_plan = agent_packet.get("target_gradient_plan")
    total_attempts = 0
    local_retest_history: list[dict[str, Any]] = []
    local_feedback: Mapping[str, Any] | None = None
    best_item = deepcopy(working_item)
    best_metrics: dict[str, Any] | None = None
    best_pass_count = -1
    local_round_limit = max(
        1,
        min(
            5,
            int(state.get("max_item_revision_attempts") or 3),
        ),
    )
    item_id = str(working_item.get("item_id") or route.get("target_item_id") or "item")
    queued_item_ids = [
        str(entry.get("item_id"))
        for entry in state.get("items_to_revise") or []
        if isinstance(entry, Mapping) and entry.get("item_id") is not None
    ]
    queue_position = (
        queued_item_ids.index(item_id) + 1
        if item_id in queued_item_ids
        else None
    )
    queue_total = len(queued_item_ids) or None
    subagent_id = f"psychometric-repair/{item_id}"
    subagent_started_at = perf_counter()
    emit_progress(
        {
            "type": "psychometric_subagent_progress",
            "subagent_id": subagent_id,
            "item_id": item_id,
            "status": "started",
            "queue_position": queue_position,
            "queue_total": queue_total,
            "round": 0,
            "max_rounds": local_round_limit,
            "elapsed_ms": 0,
            "message": "开始处理该题的修改—单题复测闭环",
        }
    )

    # A psychometric repair is now an item-local loop.  All model calls in this
    # block belong to one candidate; no candidate is committed to item_pool and
    # no whole-form measurement is triggered until the outer repair queue drains.
    for local_round in range(1, local_round_limit + 1):
        for task_index, task in enumerate(tasks, start=1):
            diagnosis_id = str(task.get("diagnosis_id") or f"D{task_index}")
            task_advice = deepcopy(dict(effective_advice))
            task_advice["selected_diagnosis_id"] = diagnosis_id
            task_advice["atomic_edit"] = deepcopy(task.get("atomic_edit"))
            if "repair_tasks" in effective_advice:
                task_advice["repair_tasks"] = [deepcopy(dict(task))]
            emit_progress(
                {
                    "type": "psychometric_subagent_progress",
                    "subagent_id": subagent_id,
                    "item_id": item_id,
                    "status": "editing",
                    "queue_position": queue_position,
                    "queue_total": queue_total,
                    "round": local_round,
                    "max_rounds": local_round_limit,
                    "task_index": task_index,
                    "task_total": len(tasks),
                    "diagnosis_id": diagnosis_id,
                    "elapsed_ms": round(
                        (perf_counter() - subagent_started_at) * 1000
                    ),
                    "message": "返修 Agent 正在生成当前题的候选修改",
                }
            )
            input_data: dict[str, Any] = {
                "action": action,
                "state": build_psychometric_repair_model_state(
                    {**state, "current_item": working_item}
                ),
                "generation_context": build_psychometric_repair_generation_context(
                    {**state, "current_item": working_item}
                ),
                "blocking_findings": [],
                "repair_source": "psychometric_diagnosis",
                "atomic_repair_advice": task_advice,
                "normal_constraints": normal_constraints,
                "target_construct_constraints": target_construct_constraints,
                "item_content": item_content,
                "option_evidence": option_evidence,
                "option_score_comparisons": option_score_comparisons,
                "target_gradient_plan": target_gradient_plan,
                "local_retest_feedback": deepcopy(local_feedback),
                "local_retest_round": local_round,
                "required_context_category": (
                    item_specification.get("context_category")
                    if isinstance(item_specification, Mapping)
                    else None
                ),
                "validation_feedback": None,
                "previous_invalid_candidate": None,
            }
            task_error: ValueError | None = None
            task_succeeded = False
            for repair_attempt in range(MAX_ITEM_OUTPUT_CANDIDATES):
                total_attempts += 1
                result: Any = None
                try:
                    result = await _ainvoke_model(
                        psychometric_item_repair_agent,
                        {"input_data": input_data},
                        job_label=(
                            f"{action} / {route.get('target_item_id') or 'item'}"
                            f" / {diagnosis_id} / local-{local_round}"
                        ),
                        timeout_seconds=PSYCHOMETRIC_REPAIR_TIMEOUT_SECONDS,
                    )
                    result = _normalize_item_repair_result(result)
                    if not isinstance(result, Mapping):
                        raise ValueError("Agent output must be an object")
                    proposed_update = result.get("state_update")
                    if not isinstance(proposed_update, Mapping):
                        raise ValueError("Agent output missing valid state_update")
                    proposed_update = deepcopy(dict(proposed_update))
                    proposed_update = normalize_atomic_option_patch_scope(
                        proposed_update,
                        task_advice,
                    )
                    validate_atomic_item_patch(
                        proposed_update,
                        working_item,
                        task_advice,
                    )
                    proposed_update = canonicalize_item_agent_update(
                        action,
                        proposed_update,
                        specification=state.get("test_specification"),
                        blueprint_cell=state.get("current_blueprint_cell"),
                        item_specification=item_specification,
                        previous_item=working_item,
                    )
                    validate_item_agent_update(
                        action,
                        proposed_update,
                        target_item_id=route.get("target_item_id"),
                        target_blueprint_cell_id=route.get("target_blueprint_cell_id"),
                        specification=state.get("test_specification"),
                        blueprint_cell=state.get("current_blueprint_cell"),
                        item_specification=item_specification,
                        previous_item=working_item,
                    )
                    working_item = deepcopy(proposed_update["current_item"])
                    task_succeeded = True
                    break
                except ValueError as exc:
                    task_error = exc
                    if repair_attempt >= MAX_ITEM_OUTPUT_CANDIDATES - 1:
                        break
                    emit_progress(
                        {
                            "type": "output_repair",
                            "retry_kind": _output_error_kind(exc),
                            "job_label": f"{action} / {diagnosis_id}",
                            "attempt": repair_attempt + 2,
                            "max_attempts": MAX_ITEM_OUTPUT_CANDIDATES,
                            "reason": str(exc),
                        }
                    )
                    input_data = {
                        **input_data,
                        "validation_feedback": str(exc),
                        "previous_invalid_candidate": _invalid_candidate(
                            result,
                            exc,
                            state_update_only=True,
                        ),
                    }
            if not task_succeeded:
                raise ValueError(
                    f"{action} task {diagnosis_id} did not produce a valid patch: "
                    f"{task_error}"
                )

        candidate_for_retest = deepcopy(working_item)
        candidate_for_retest["version"] = base_version + 1
        emit_progress(
            {
                "type": "psychometric_subagent_progress",
                "subagent_id": subagent_id,
                "item_id": item_id,
                "status": "retesting",
                "queue_position": queue_position,
                "queue_total": queue_total,
                "round": local_round,
                "max_rounds": local_round_limit,
                "elapsed_ms": round(
                    (perf_counter() - subagent_started_at) * 1000
                ),
                "message": "候选题已生成，开始单题局部复测",
            }
        )
        local_result = await _run_psychometric_local_retest(
            state=state,
            candidate_item=candidate_for_retest,
        )
        metrics = deepcopy(local_result["metrics"])
        qualification = metrics.get("qualification") or {}
        pass_count = sum(
            bool(qualification.get(key))
            for key in (
                "citc_pass",
                "target_rho_pass",
                "same_domain_vts_pass",
                "cross_domain_vts_pass",
            )
        )
        local_event = {
            "round": local_round,
            "candidate_version": candidate_for_retest.get("version"),
            "candidate_item": deepcopy(candidate_for_retest),
            "metrics": metrics,
            "simulation": deepcopy(local_result.get("simulation") or {}),
            "pass_count": pass_count,
        }
        local_retest_history.append(local_event)
        failed_gates = [
            key
            for key in (
                "citc_pass",
                "target_rho_pass",
                "same_domain_vts_pass",
                "cross_domain_vts_pass",
            )
            if not bool(qualification.get(key))
        ]
        emit_progress(
            {
                "type": "psychometric_subagent_progress",
                "subagent_id": subagent_id,
                "item_id": item_id,
                "status": "round_completed",
                "queue_position": queue_position,
                "queue_total": queue_total,
                "round": local_round,
                "max_rounds": local_round_limit,
                "passed_gate_count": pass_count,
                "gate_total": 4,
                "qualified": bool(qualification.get("qualified")),
                "failed_gates": failed_gates,
                "elapsed_ms": round(
                    (perf_counter() - subagent_started_at) * 1000
                ),
                "message": (
                    "单题局部复测完成"
                    if qualification.get("qualified")
                    else "单题局部复测未通过"
                ),
            }
        )
        if pass_count > best_pass_count:
            best_pass_count = pass_count
            best_item = deepcopy(candidate_for_retest)
            best_metrics = metrics
        if bool(qualification.get("qualified")):
            best_item = deepcopy(candidate_for_retest)
            best_metrics = metrics
            break
        local_feedback = metrics
        if local_round < local_round_limit:
            emit_progress(
                {
                    "type": "psychometric_local_retest",
                    "status": "failed",
                    "item_id": candidate_for_retest.get("item_id"),
                    "round": local_round,
                    "max_rounds": local_round_limit,
                    "passed_gate_count": pass_count,
                    "message": "单题局部复测未通过，继续让返修 Agent 修改当前候选",
                }
            )

    working_item = deepcopy(best_item)
    working_item["version"] = base_version + 1
    local_status = (
        "passed"
        if best_metrics
        and bool((best_metrics.get("qualification") or {}).get("qualified"))
        else "bounded_not_passed"
    )
    local_retest = {
        "status": local_status,
        "max_rounds": local_round_limit,
        "rounds_completed": len(local_retest_history),
        "best_passed_gate_count": max(0, best_pass_count),
        "best_metrics": deepcopy(best_metrics),
        "history": local_retest_history,
    }
    emit_progress(
        {
            "type": "psychometric_subagent_progress",
            "subagent_id": subagent_id,
            "item_id": item_id,
            "status": "completed",
            "queue_position": queue_position,
            "queue_total": queue_total,
            "round": len(local_retest_history),
            "max_rounds": local_round_limit,
            "passed_gate_count": max(0, best_pass_count),
            "gate_total": 4,
            "qualified": local_status == "passed",
            "local_status": local_status,
            "elapsed_ms": round(
                (perf_counter() - subagent_started_at) * 1000
            ),
            "message": "该题返修闭环完成，等待主流程汇总",
        }
    )

    # The individual task results are deliberately kept in memory.  They are
    # one psychometric repair transaction, so the persisted item receives one
    # version bump, not one bump per option/task.  The local retest history is
    # retained on the active repair for audit; the main process still performs
    # one unified full-bank administration after all item candidates are ready.

    return {
        "state_update": {
            "current_item": working_item,
            "active_psychometric_repair": {
                **deepcopy(dict(active_psychometric_repair)),
                "local_retest": local_retest,
            },
            **dict(program_update),
        },
        "repair_attempt_count": total_attempts,
        "summary": (
            f"同一题完成 {len(local_retest_history)} 轮‘修改—单题复测’；"
            f"局部状态={local_status}，随后由主流程统一整卷施测"
        ),
    }


def _psychometric_batch_item_specification(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the fixed slot metadata needed by one isolated subagent."""

    item_id = str(item.get("item_id") or "")
    specification = next(
        (
            deepcopy(dict(row))
            for row in state.get("item_specifications") or []
            if isinstance(row, Mapping)
            and str(row.get("specification_id")) == item_id
        ),
        None,
    )
    if specification is not None:
        return specification
    return {
        "specification_id": item_id,
        "blueprint_cell_id": item.get("blueprint_cell_id"),
        "target_dimension_id": item.get("target_dimension_id"),
        "context_category": item.get("context_category"),
        "context_seed": item.get("scenario"),
        "avoid_scenario_patterns": [],
        "avoid_response_patterns": [],
    }


def _psychometric_batch_blueprint_cell(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any] | None:
    cell_id = item.get("blueprint_cell_id")
    return next(
        (
            deepcopy(dict(cell))
            for cell in (state.get("blueprint") or {}).get("cells") or []
            if isinstance(cell, Mapping) and cell.get("cell_id") == cell_id
        ),
        None,
    )


async def _execute_psychometric_repair_batch(
    *,
    action: str,
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict[str, Any]:
    """Run all diagnosed item-local repair loops concurrently, then merge once.

    Subagents receive independent shallow state copies and can only return an
    item candidate plus its local retest history.  They never mutate the
    shared item pool and never run the whole-form administration.  The main
    workflow merges successful candidates after the batch barrier, invalidates
    the old formal response snapshot once, and lets the next workflow pass
    perform one unified administration for the updated bank.
    """

    del action, route
    queue_entries = [
        deepcopy(dict(entry))
        for entry in [
            *(state.get("items_to_regenerate") or []),
            *(state.get("items_to_revise") or []),
        ]
        if isinstance(entry, Mapping)
        and entry.get("item_id")
        and isinstance(entry.get("atomic_repair_advice"), Mapping)
        and entry["atomic_repair_advice"].get("decision") == "repair"
        and entry["atomic_repair_advice"].get("repair_tasks")
    ]
    unique_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in queue_entries:
        item_id = str(entry["item_id"])
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_entries.append(entry)
    if not unique_entries:
        raise ValueError("并发心理测量返修没有可执行的诊断任务")

    item_by_id = {
        str(item.get("item_id")): deepcopy(dict(item))
        for item in state.get("item_pool") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    if not item_by_id:
        item_by_id = {
            str(item.get("item_id")): deepcopy(dict(item))
            for item in state.get("frozen_item_bank") or []
            if isinstance(item, Mapping) and item.get("item_id")
        }
    missing_ids = [
        str(entry["item_id"])
        for entry in unique_entries
        if str(entry["item_id"]) not in item_by_id
    ]
    if missing_ids:
        raise ValueError("并发返修找不到题目：" + "、".join(missing_ids))

    batch_id = (
        f"psychometric-repair-batch/"
        f"{int(state.get('psychometric_analysis_round') or 0)}/"
        f"{len(state.get('psychometric_repair_history') or []) + 1}"
    )
    concurrency = max(
        1,
        min(
            8,
            int(
                state.get("psychometric_subagent_max_concurrency")
                or PSYCHOMETRIC_REPAIR_SUBAGENT_CONCURRENCY
            ),
        ),
    )
    emit_progress(
        {
            "type": "psychometric_subagent_progress",
            "status": "batch_started",
            "batch_id": batch_id,
            "batch_total": len(unique_entries),
            "concurrency": concurrency,
            "message": (
                f"启动 {len(unique_entries)} 个单题返修 subagent，"
                f"最大并发 {concurrency}；全部完成后统一合并"
            ),
        }
    )
    semaphore = asyncio.Semaphore(concurrency)
    batch_response_ref = (
        state.get("virtual_response_data_ref")
        or state.get("previous_virtual_response_data_ref")
    )

    async def run_one(entry: Mapping[str, Any]) -> dict[str, Any]:
        item_id = str(entry["item_id"])
        item = item_by_id[item_id]
        blueprint_cell = _psychometric_batch_blueprint_cell(state, item)
        active_repair = {
            **deepcopy(dict(entry)),
            "baseline_item": deepcopy(item),
            "baseline_profile": deepcopy(
                (state.get("item_pattern_profiles") or {}).get(item_id)
            ),
            # A batch has one shared formal baseline.  The full snapshot is
            # intentionally not copied into every task; the next admission
            # pass can safely fall back to incremental remeasurement.
            "baseline_analysis_snapshot": None,
        }
        local_state = dict(state)
        local_state.update(
            {
                "current_item": deepcopy(item),
                "current_blueprint_cell": blueprint_cell,
                "current_item_specification": _psychometric_batch_item_specification(
                    state,
                    item,
                ),
                "current_item_review": None,
                "active_psychometric_repair": active_repair,
                "_psychometric_batch_id": batch_id,
            }
        )
        # Pass the shared formal baseline explicitly into every isolated
        # worker. After a bank change the current reference may be cleared
        # while the previous baseline is retained for reuse/local retesting.
        if batch_response_ref:
            local_state["virtual_response_data_ref"] = batch_response_ref
        local_route: PSJTRouteDecision = {
            "next_action": "revise_item",
            "reason": "批量返修 subagent 的单题局部任务",
            "target_item_id": item_id,
            "target_blueprint_cell_id": item.get("blueprint_cell_id"),
        }
        async with semaphore:
            try:
                result = await _execute_psychometric_repair_item(
                    action="revise_item",
                    route=local_route,
                    state=local_state,  # type: ignore[arg-type]
                    item_specification=local_state["current_item_specification"],
                    active_psychometric_repair=active_repair,
                    atomic_advice=entry["atomic_repair_advice"],
                    diagnosis_evidence=entry.get("diagnosis_evidence") or {},
                    program_update={},
                )
                return {
                    "item_id": item_id,
                    "result": result,
                    "error": None,
                }
            except Exception as exc:
                emit_progress(
                    {
                        "type": "psychometric_subagent_progress",
                        "status": "failed",
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "error": str(exc),
                        "message": "该题 subagent 失败，主流程保留原版本",
                    }
                )
                return {
                    "item_id": item_id,
                    "result": None,
                    "error": str(exc),
                }

    outcomes = await asyncio.gather(
        *(run_one(entry) for entry in unique_entries),
        return_exceptions=False,
    )
    successful: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for outcome in outcomes:
        item_id = str(outcome["item_id"])
        result = outcome.get("result")
        if not isinstance(result, Mapping):
            failures[item_id] = str(outcome.get("error") or "subagent 没有返回结果")
            continue
        candidate = (result.get("state_update") or {}).get("current_item")
        if not isinstance(candidate, Mapping) or str(candidate.get("item_id")) != item_id:
            failures[item_id] = "subagent 返回的题目 ID 不匹配"
            continue
        if int(candidate.get("version") or 0) <= int(item_by_id[item_id].get("version") or 0):
            failures[item_id] = "subagent 没有生成新题目版本"
            continue
        successful[item_id] = {
            "candidate": deepcopy(dict(candidate)),
            "active_repair": deepcopy(
                (result.get("state_update") or {}).get(
                    "active_psychometric_repair"
                )
                or {}
            ),
            "summary": result.get("summary"),
        }

    if not successful:
        # A batch-level model/output failure must not discard the current bank
        # or invalidate an otherwise usable response snapshot. Keep every item
        # in its queue with a retryable status; the next route pass can
        # diagnose it again instead of terminating the whole run.
        failed_revise = [
            {
                **deepcopy(dict(entry)),
                "queue_status": "batch_failed",
                "batch_error": failures.get(str(entry.get("item_id")))
                or "subagent 没有返回有效候选",
            }
            for entry in state.get("items_to_revise") or []
            if isinstance(entry, Mapping) and entry.get("item_id")
        ]
        failed_regenerate = [
            {
                **deepcopy(dict(entry)),
                "queue_status": "batch_failed",
                "batch_error": failures.get(str(entry.get("item_id")))
                or "subagent 没有返回有效候选",
            }
            for entry in state.get("items_to_regenerate") or []
            if isinstance(entry, Mapping) and entry.get("item_id")
        ]
        detail = "；".join(
            f"{item_id}：{reason}" for item_id, reason in failures.items()
        )
        batch_summary = {
            "batch_id": batch_id,
            "batch_total": len(unique_entries),
            "completed_count": 0,
            "failed_count": len(failures),
            "remaining_count": len(failed_revise) + len(failed_regenerate),
            "concurrency": concurrency,
            "status": "completed_with_failures",
            "error_detail": detail,
        }
        emit_progress(
            {
                "type": "psychometric_subagent_progress",
                "status": "batch_completed",
                "batch_id": batch_id,
                "batch_total": len(unique_entries),
                "completed_count": 0,
                "failed_count": len(failures),
                "remaining_count": len(failed_revise) + len(failed_regenerate),
                "message": (
                    f"并发返修本批没有成功候选，已保留 {len(failures)} 道题，"
                    "下一轮重新诊断；当前施测数据保留"
                ),
            }
        )
        return {
            "state_update": {
                "current_item": None,
                "current_item_specification": None,
                "current_blueprint_cell": None,
                "current_item_review": None,
                "current_item_repair_attempted": False,
                "current_item_repair_failure": None,
                "active_psychometric_repair": None,
                "psychometric_repair_confirmation": None,
                "items_to_revise": failed_revise,
                "items_to_regenerate": failed_regenerate,
                "psychometric_repair_batch_summary": batch_summary,
            },
            "summary": (
                f"并发返修本批 {len(failures)} 道题均未返回有效候选；"
                "已保留原题和当前施测数据，下一轮重新诊断"
            ),
            "repair_attempt_count": 0,
        }

    merged_pool = []
    base_pool = state.get("item_pool") or state.get("frozen_item_bank") or []
    for item in base_pool:
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("item_id") or "")
        merged_pool.append(
            deepcopy(successful[item_id]["candidate"])
            if item_id in successful
            else deepcopy(dict(item))
        )
    profiles = dict(state.get("item_pattern_profiles") or {})
    for item_id, payload in successful.items():
        profiles[item_id] = build_item_pattern_profile(
            payload["candidate"],
            _psychometric_batch_item_specification(state, payload["candidate"]),
        )

    successful_ids = set(successful)
    remaining_revise = [
        {
            **deepcopy(dict(entry)),
            **(
                {"queue_status": "batch_failed", "batch_error": failures[item_id]}
                if (item_id := str(entry.get("item_id"))) in failures
                else {}
            ),
        }
        for entry in state.get("items_to_revise") or []
        if isinstance(entry, Mapping) and str(entry.get("item_id")) not in successful_ids
    ]
    remaining_regenerate = [
        {
            **deepcopy(dict(entry)),
            **(
                {"queue_status": "batch_failed", "batch_error": failures[item_id]}
                if (item_id := str(entry.get("item_id"))) in failures
                else {}
            ),
        }
        for entry in state.get("items_to_regenerate") or []
        if isinstance(entry, Mapping) and str(entry.get("item_id")) not in successful_ids
    ]
    repair_history = deepcopy(state.get("psychometric_repair_history") or [])
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    for entry in unique_entries:
        item_id = str(entry["item_id"])
        if item_id not in successful:
            continue
        active_repair = successful[item_id]["active_repair"]
        round_number = int(entry.get("revision_round") or 1)
        rounds[item_id] = round_number
        repair_history.append(
            {
                "event": "psychometric_item_repaired",
                "recorded_at": utc_timestamp(),
                "item_id": item_id,
                "revision_round": round_number,
                "action": entry.get("action") or "revise_item",
                "baseline_metrics": deepcopy(entry.get("baseline_metrics") or {}),
                "baseline_item": deepcopy(item_by_id[item_id]),
                "baseline_profile": deepcopy(
                    (state.get("item_pattern_profiles") or {}).get(item_id)
                ),
                "baseline_analysis_snapshot": None,
                "new_item_version": successful[item_id]["candidate"].get("version"),
                "diagnosis_fingerprint": entry.get("diagnosis_fingerprint"),
                "atomic_repair_advice": deepcopy(
                    entry.get("atomic_repair_advice")
                ),
                "local_retest": deepcopy(active_repair.get("local_retest")),
                "subagent_id": f"psychometric-repair/{item_id}",
                "batch_id": batch_id,
            }
        )

    previous_response_ref = state.get("virtual_response_data_ref")
    invalidated_statistics = {
        str(item_id): deepcopy(dict(statistics))
        for item_id, statistics in (state.get("item_statistics") or {}).items()
        if str(item_id) not in successful_ids and isinstance(statistics, Mapping)
    }
    reset_update: dict[str, Any] = {
        "current_item": None,
        "current_item_specification": None,
        "current_blueprint_cell": None,
        "current_item_review": None,
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "active_psychometric_repair": None,
        "psychometric_repair_confirmation": None,
        "items_to_revise": remaining_revise,
        "items_to_regenerate": remaining_regenerate,
        "selected_items": [],
        "reserve_items": [],
        "selection_results": None,
        "selection_reasons": {},
        "item_final_dispositions": {
            str(item_id): deepcopy(dict(disposition))
            for item_id, disposition in (state.get("item_final_dispositions") or {}).items()
            if str(item_id) not in successful_ids and isinstance(disposition, Mapping)
        },
        "item_pool": merged_pool,
        "item_pattern_profiles": profiles,
        "candidate_bank_audit": None,
        "psychometric_repair_rounds": rounds,
        "psychometric_repair_history": repair_history,
        "psychometric_repair_batch_summary": {
            "batch_id": batch_id,
            "batch_total": len(unique_entries),
            "completed_count": len(successful),
            "failed_count": len(failures),
            "remaining_count": len(remaining_revise) + len(remaining_regenerate),
            "concurrency": concurrency,
            "status": "completed" if not failures else "completed_with_failures",
        },
        "blueprint_coverage": None,
        "assembled_test": None,
        "test_review_result": None,
        "final_test": None,
        "item_database_ref": None,
        "technical_report": None,
        "virtual_respondent_report": None,
        "virtual_response_data_ref": None,
        "virtual_response_summary": None,
        "virtual_response_item_bank_id": None,
        "virtual_response_item_bank_version": None,
        "item_statistics": invalidated_statistics,
        "psychometric_round_result": None,
        "test_statistics": None,
        "factor_results": None,
        "irt_results": None,
        "dif_results": None,
        "best_assembly_candidate": None,
    }
    if isinstance(previous_response_ref, str) and previous_response_ref:
        reset_update["previous_virtual_response_data_ref"] = previous_response_ref
    emit_progress(
        {
            "type": "psychometric_subagent_progress",
            "status": "batch_completed",
            "batch_id": batch_id,
            "batch_total": len(unique_entries),
            "completed_count": len(successful),
            "failed_count": len(failures),
            "remaining_count": len(remaining_revise) + len(remaining_regenerate),
            "message": (
                f"并发返修完成：成功 {len(successful)} 题，"
                f"失败 {len(failures)} 题；主流程已统一合并，等待整批施测"
            ),
        }
    )
    return {
        "state_update": reset_update,
        "summary": (
            f"并发完成 {len(successful)} 道题的修改—单题复测闭环；"
            f"失败 {len(failures)} 道，之后统一对更新后的题库施测"
        ),
        "repair_attempt_count": len(successful),
    }


async def execute_item_action_with_repair(
    action: str,
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict[str, Any]:
    """对题目 Agent 的无效结构输出进行最多3次有界修复。"""

    active_psychometric_repair = state.get("active_psychometric_repair")
    agent = (
        psychometric_item_repair_agent
        if action in {"revise_item", "regenerate_item"}
        and isinstance(active_psychometric_repair, Mapping)
        else AGENT_MAP[action]
    )
    program_update: dict[str, Any] = {}
    item_specification = state.get("current_item_specification")
    if (
        (
            action == "generate_item"
            or (
                action == "regenerate_item"
                and bool(state.get("current_skeleton_repair_required"))
            )
        )
        and isinstance(state.get("blueprint"), Mapping)
        and state["blueprint"].get("version") == GENERATION_BLUEPRINT_VERSION
        and (
        not isinstance(item_specification, Mapping)
        or "activation_mechanism" not in item_specification
        )
    ):
        program_update = await _prepare_current_slot_skeleton(state)
        state = {**state, **program_update}
        item_specification = state["current_item_specification"]
    current_review = state.get("current_item_review")
    blocking_findings = [
        deepcopy(finding)
        for finding in (current_review or {}).get("findings") or []
        if isinstance(finding, Mapping)
        and finding.get("severity") == "blocking"
    ]
    # Resume older checkpoints safely after behavioral levels became immutable.
    # Legacy level-edit requests are realized as option-text repairs.
    for finding in blocking_findings:
        if finding.get("locus") == "behavioral_level":
            finding["locus"] = "response_options"
            finding["repair_instruction"] = (
                str(finding.get("repair_instruction") or "")
                + " Keep behavioral_level and scoring_key unchanged; rewrite "
                "the named option text to realize its fixed level."
            )
        for edit in finding.get("required_edits") or []:
            if isinstance(edit, dict) and edit.get("field") == "behavioral_level":
                edit["field"] = "response_options"
    atomic_advice = (
        deepcopy(active_psychometric_repair.get("atomic_repair_advice"))
        if isinstance(active_psychometric_repair, Mapping)
        and isinstance(
            active_psychometric_repair.get("atomic_repair_advice"), Mapping
        )
        else None
    )
    diagnosis_evidence = (
        deepcopy(active_psychometric_repair.get("diagnosis_evidence"))
        if isinstance(active_psychometric_repair, Mapping)
        and isinstance(active_psychometric_repair.get("diagnosis_evidence"), Mapping)
        else None
    )
    if (
        action in {"revise_item", "regenerate_item"}
        and not blocking_findings
        and atomic_advice is None
    ):
        raise ValueError(f"{action} 缺少 blocking 审题意见")
    if (
        action in {"revise_item", "regenerate_item"}
        and isinstance(active_psychometric_repair, Mapping)
        and isinstance(atomic_advice, Mapping)
        and repair_tasks_from_advice(atomic_advice)
    ):
        return await _execute_psychometric_repair_item(
            action=action,
            route=route,
            state=state,
            item_specification=(
                item_specification if isinstance(item_specification, Mapping) else None
            ),
            active_psychometric_repair=active_psychometric_repair,
            atomic_advice=atomic_advice,
            diagnosis_evidence=(diagnosis_evidence or {}),
            program_update=program_update,
        )
    model_state = (
        build_psychometric_repair_model_state(state)
        if diagnosis_evidence is not None
        else build_item_model_state(state)
    )
    if blocking_findings:
        model_state = {
            **model_state,
            "current_item_review": None,
        }
    agent_packet = (
        build_psychometric_agent_input(diagnosis_evidence)
        if diagnosis_evidence is not None
        else None
    )
    input_data: dict[str, Any] = {
        "action": action,
        "state": model_state,
        "generation_context": (
            build_psychometric_repair_generation_context(state)
            if diagnosis_evidence is not None
            else build_item_generation_context(state)
        ),
        "blocking_findings": blocking_findings,
        "repair_source": (
            "psychometric_diagnosis"
            if atomic_advice is not None
            else "content_review"
        ),
        "atomic_repair_advice": atomic_advice,
        "normal_constraints": (
            agent_packet.get("normal_constraints")
            if agent_packet is not None
            else None
        ),
        "option_evidence": (
            agent_packet.get("option_evidence")
            if agent_packet is not None
            else _build_option_evidence_for_repair(state, active_psychometric_repair)
        ),
        "option_score_comparisons": (
            agent_packet.get("option_score_comparisons")
            if agent_packet is not None
            else None
        ),
        "required_context_category": (
            item_specification.get("context_category")
            if isinstance(item_specification, dict)
            else None
        ),
        "validation_feedback": None,
        "previous_invalid_candidate": None,
    }
    last_error: ValueError | None = None
    max_candidates = MAX_ITEM_OUTPUT_CANDIDATES
    accumulated_patch: dict[str, Any] | None = None
    for repair_attempt in range(max_candidates):
        result: Any = None
        try:
            result = await _ainvoke_model(
                agent,
                {"input_data": input_data},
                job_label=(
                    f"{action} / {route.get('target_item_id') or '新题'}"
                ),
            )
            result = _normalize_item_repair_result(result)
            if not isinstance(result, dict):
                raise ValueError("Agent 输出必须是对象")
            proposed_update = result.get("state_update")
            if not isinstance(proposed_update, Mapping):
                raise ValueError("Agent 输出缺少有效的 state_update")
            proposed_update = deepcopy(dict(proposed_update))
            if action in {"revise_item", "regenerate_item"}:
                scenario_update = proposed_update.get("scenario_update")
                option_updates = proposed_update.get("option_updates")
                if atomic_advice is not None:
                    validate_atomic_item_patch(
                        proposed_update,
                        state.get("current_item") or {},
                        atomic_advice,
                    )
                elif isinstance(accumulated_patch, dict):
                    merged_options = {
                        str(patch.get("option_id")): deepcopy(patch)
                        for patch in accumulated_patch.get("option_updates") or []
                        if isinstance(patch, Mapping) and patch.get("option_id")
                    }
                    if isinstance(option_updates, list):
                        for patch in option_updates:
                            if isinstance(patch, Mapping) and patch.get("option_id"):
                                merged_options[str(patch["option_id"])] = deepcopy(
                                    dict(patch)
                                )
                    proposed_update = {
                        "scenario_update": (
                            scenario_update
                            if scenario_update is not None
                            else accumulated_patch.get("scenario_update")
                        ),
                        "option_updates": list(merged_options.values()),
                    }
            proposed_update = canonicalize_item_agent_update(
                action,
                proposed_update,
                specification=state.get("test_specification"),
                blueprint_cell=state.get("current_blueprint_cell"),
                item_specification=item_specification,
                previous_item=state.get("current_item"),
            )
            if action in {"revise_item", "regenerate_item"}:
                accumulated_patch = {
                    "scenario_update": (
                        proposed_update["current_item"].get("scenario")
                        if proposed_update["current_item"].get("scenario")
                        != (state.get("current_item") or {}).get("scenario")
                        else None
                    ),
                    "option_updates": [
                        {
                            "option_id": option.get("option_id"),
                            "text": option.get("text"),
                        }
                        for option in proposed_update["current_item"].get(
                            "response_options"
                        )
                        or []
                        if isinstance(option, Mapping)
                        and next(
                            (
                                prior.get("text")
                                for prior in (
                                    (state.get("current_item") or {}).get(
                                        "response_options"
                                    )
                                    or []
                                )
                                if isinstance(prior, Mapping)
                                and prior.get("option_id")
                                == option.get("option_id")
                            ),
                            None,
                        )
                        != option.get("text")
                    ],
                }
            result = {
                **result,
                "state_update": proposed_update,
            }
            validate_item_agent_update(
                action,
                proposed_update,
                target_item_id=route.get("target_item_id"),
                target_blueprint_cell_id=route.get(
                    "target_blueprint_cell_id"
                ),
                specification=state.get("test_specification"),
                blueprint_cell=state.get("current_blueprint_cell"),
                item_specification=item_specification,
                previous_item=state.get("current_item"),
            )
            return {
                **result,
                "state_update": {
                    **proposed_update,
                    **program_update,
                },
                "repair_attempt_count": repair_attempt,
            }
        except ValueError as exc:
            last_error = exc
            if repair_attempt >= max_candidates - 1:
                break
            emit_progress(
                {
                    "type": "output_repair",
                    "retry_kind": _output_error_kind(exc),
                    "job_label": action,
                    "attempt": repair_attempt + 2,
                    "max_attempts": max_candidates,
                    "reason": str(exc),
                }
            )
            input_data = {
                **input_data,
                "validation_feedback": str(exc),
                "previous_invalid_candidate": _invalid_candidate(
                    result,
                    exc,
                    state_update_only=True,
                ),
            }
            if accumulated_patch is not None:
                input_data["previous_invalid_candidate"] = deepcopy(
                    accumulated_patch
                )

    raise ValueError(
        f"{action} 经过 {max_candidates} 次候选输出后"
        f"仍未通过结构校验：{last_error}"
    )


async def _execute_agent(
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict:
    action = route["next_action"]
    if action == "build_blueprint":
        return await execute_fixed_blueprint(state)
    if action == "review_item":
        return await execute_item_review(state)
    if action in {"generate_item", "regenerate_item", "revise_item"}:
        return await execute_item_action_with_repair(action, route, state)
    if action == "simulate_responses":
        return await execute_virtual_simulation(state)
    if action == "analyze_psychometrics":
        return await execute_psychometric_analysis_with_provisional_form(state)
    if action == "select_items":
        return await execute_item_selection_with_diagnosis(state)
    if action == "psychometric_repair_batch":
        return await _execute_psychometric_repair_batch(
            action=action,
            route=route,
            state=state,
        )
    if action == "assemble_test":
        return await asyncio.to_thread(run_test_assembly, state)
    if action == "review_test":
        return await asyncio.to_thread(run_test_review, state)
    if action == "rescore_test":
        return await asyncio.to_thread(run_test_rescore, state)
    if action == "generate_reports":
        return await asyncio.to_thread(run_report_generation, state)
    agent = AGENT_MAP.get(action)
    if agent is None:
        raise ValueError(f"没有为任务 {action!r} 注册对应的 Agent")
    input_data = {
        "state": build_requirement_model_state(state),
        "target_item_id": route.get("target_item_id"),
        "target_blueprint_cell_id": route.get("target_blueprint_cell_id"),
    }
    result = await _ainvoke_model(
        agent,
        {"input_data": input_data},
        job_label=action,
    )
    return result


async def execute_agent(
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict:
    """Execute one action while attributing model usage to its iteration."""

    action = route["next_action"]
    iteration = _development_iteration_for_action(action, state)
    with iteration_context(iteration):
        return await _execute_agent(route, state)
