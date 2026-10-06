"""Dispatch LLM-backed and deterministic workflow actions."""

import asyncio
import os
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
    repair_blueprint_proposal,
    required_expansion_situation_total,
    validate_generation_blueprint,
    required_generation_total,
    resolve_facet_item_counts,
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
from sjt_system.runtime.concurrency import UnlimitedConcurrency, gather_all
from sjt_system.runtime.telemetry import (
    aggregate_iteration_calls,
    iteration_context,
    read_ledger,
)
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.runtime.iteration_metrics import (
    persist_iteration_metrics_snapshot,
    validate_same_measurement,
)
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
from sjt_system.evaluation.round_results import (
    build_iteration_metrics_snapshot,
    build_facet_iteration_metric,
    build_item_iteration_metric,
)
from sjt_system.evaluation.selection import (
    build_psychometric_repair_evidence,
    _psychometric_repair_entry,
    psychometric_diagnosis_to_review,
    run_item_selection,
    validate_psychometric_repair_diagnosis,
)
from sjt_system.evaluation.form_optimizer import optimize_test_form_with_agent
from sjt_system.evaluation.virtual_content_review import (
    MAX_MEASUREMENTS,
    PROTOCOL as VIRTUAL_CONTENT_REVIEW_PROTOCOL,
    build_virtual_content_repair_entry,
    iteration_gates_pass,
    is_enabled as virtual_content_review_enabled,
    round_label as virtual_content_round_label,
)
from sjt_system.workflow.replacement_policy import replacement_capacity_available
from sjt_system.evaluation.diagnosis import (
    build_scenario_repair_entry,
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
from sjt_system.delivery.reporting import (
    development_round_batch_label,
    run_report_generation,
)
from sjt_system.workflow.constants import PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS
from sjt_system.evaluation.facet_iteration import (
    is_enabled as fixed_facet_iteration_enabled,
    ensure_measurement_allowed,
    facet_repair_candidates,
    record_measurement as record_facet_measurement,
)


MAX_ITEM_OUTPUT_CANDIDATES = 2
MAX_ITEM_SKELETON_ATTEMPTS = 1


def _development_iteration_for_action(
    action: str,
    state: Mapping[str, Any],
) -> int | None:
    """Return the measurement batch that owns model-call usage."""

    current_round = int(state.get("psychometric_analysis_round") or 0)
    if action == "psychometric_repair_batch":
        return max(1, current_round)
    if action in {
        "generate_item",
        "generate_items_batch",
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
    optimizer_candidates = candidates
    if fixed_facet_iteration_enabled(state):
        accepted = state["facet_iteration_state"].get("accepted_facets") or {}
        optimizer_candidates = [
            item for item in candidates
            if str(item.get("target_dimension_id")) not in accepted
            or str(item.get("item_id")) in accepted[str(item.get("target_dimension_id"))]["selected_item_ids"]
        ]
    try:
        optimizer_result = await optimize_test_form_with_agent(
            state,
            optimizer_candidates,
            item_statistics,
            test_statistics if isinstance(test_statistics, Mapping) else None,
        )
        selected_ids = [
            str(item_id) for item_id in optimizer_result.get("selected_item_ids") or []
        ]
    except Exception as exc:
        selection_error = str(exc)
        selected_ids = _partial_provisional_form_ids(state, optimizer_candidates)

    final_item_count = sum(
        int(cell.get("planned_retention_count") or 0)
        for cell in (state.get("blueprint") or {}).get("cells") or []
        if isinstance(cell, Mapping)
    )
    form_metrics = build_provisional_form_metrics(state, selected_ids)
    candidate_by_id = {
        str(item.get("item_id")): item
        for item in candidates
        if isinstance(item, Mapping) and item.get("item_id") is not None
    }
    candidate_item_metrics = {
        str(item_id): build_item_iteration_metric(
            str(item_id),
            item_statistics.get(str(item_id)) or {},
            candidate_by_id.get(str(item_id)),
        )
        for item_id in candidate_by_id
    }
    for item_id, metric in candidate_item_metrics.items():
        metric["selected_for_form"] = item_id in selected_ids
    item_metrics = {
        item_id: metric
        for item_id, metric in candidate_item_metrics.items()
        if item_id in selected_ids
    }
    facet_metrics = {
        str(row.get("sjt_facet_id")): build_facet_iteration_metric(row)
        for row in (form_metrics.get("facet_metrics") or [])
        if isinstance(row, Mapping) and row.get("sjt_facet_id") is not None
    }
    qualified_count = sum(
        1 for item in candidates if isinstance(item, Mapping)
        and (iteration_gates_pass(item_statistics.get(str(item.get("item_id"))) or {})
             if virtual_content_review_enabled(state)
             else (item_statistics.get(str(item.get("item_id"))) or {}).get("quality_evaluation", {}).get("recommendation") == "retain")
    )
    record = {
        "analysis_round": iteration,
        "recorded_at": utc_timestamp(),
        "candidate_count": len(candidates),
        "qualified_item_count": qualified_count,
        "requested_item_count": final_item_count,
        "item_count": len(selected_ids),
        "form_status": "complete" if len(selected_ids) == final_item_count else "incomplete",
        "form_item_ids": selected_ids,
        "item_metrics": item_metrics,
        "candidate_item_metrics": candidate_item_metrics,
        "facet_metrics": facet_metrics,
        "form_metrics": form_metrics,
        "form_optimizer": deepcopy(optimizer_result),
        "form_selection_error": selection_error,
        "token_usage": _iteration_token_usage(state, iteration),
    }
    if virtual_content_review_enabled(state):
        record.update(
            virtual_content_review_protocol=VIRTUAL_CONTENT_REVIEW_PROTOCOL,
            round_label=virtual_content_round_label(iteration),
            workflow_stage="initial_measurement" if iteration == 1 else "virtual_content_review",
            item_snapshots=deepcopy(candidate_by_id),
            item_statistics_snapshot=deepcopy(item_statistics),
            item_content_evidence=deepcopy(state.get("item_content_evidence") or {}),
            item_lineage=deepcopy(state.get("item_lineage") or {}),
            response_data_ref=state.get("virtual_response_data_ref"),
            item_bank_id=state.get("item_bank_id"),
            item_bank_version=state.get("item_bank_version"),
            item_dispositions=deepcopy(state.get("item_final_dispositions") or {}),
            virtual_sample_config=deepcopy(state.get("virtual_sample_config")),
        )
    if fixed_facet_iteration_enabled(state):
        record.update(
            iteration_policy_version=state["facet_iteration_state"]["policy_version"],
            development_round=state["facet_iteration_state"]["development_round"],
            measurement_batch=iteration,
            development_round_batch=(
                int(state["facet_iteration_state"].get("development_round_batch") or 0) + 1
            ),
            round_label=f"第{state['facet_iteration_state']['development_round']}轮",
        )
    artifact_path = persist_iteration_metrics_snapshot(state, record)
    if artifact_path:
        record["iteration_metrics_artifact"] = artifact_path
    return record


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
            # Older in-memory/checkpoint records may only contain the legacy
            # ``candidate_item_metrics`` field. Do not compare those records
            # with the newer canonical snapshot shape: the durable artifact
            # writer below is the authority for immutable measured values.
            # Once both sides carry canonical ``item_metrics``, keep the
            # overwrite guard active for resumed/current runs.
            if (
                virtual_content_review_enabled(state)
                and isinstance(existing.get("item_metrics"), Mapping)
                and isinstance(refreshed.get("item_metrics"), Mapping)
            ):
                validate_same_measurement(
                    build_iteration_metrics_snapshot(existing, run_id=state.get("run_id")),
                    build_iteration_metrics_snapshot(refreshed, run_id=state.get("run_id")),
                )
            # A later selection/annotation pass can carry less metric detail
            # than the analysis pass. Preserve the already-persisted snapshot
            # instead of replacing it with an empty or unavailable value.
            for field in (
                "item_metrics",
                "candidate_item_metrics",
                "facet_metrics",
                "iteration_metrics_artifact",
            ):
                if not refreshed.get(field) and existing.get(field):
                    refreshed[field] = deepcopy(existing[field])
            existing_form = existing.get("form_metrics")
            refreshed_form = refreshed.get("form_metrics")
            existing_facets = (
                existing_form.get("facet_metrics")
                if isinstance(existing_form, Mapping)
                else None
            )
            refreshed_facets = (
                refreshed_form.get("facet_metrics")
                if isinstance(refreshed_form, Mapping)
                else None
            )
            if existing_facets and not refreshed_facets:
                refreshed["form_metrics"] = deepcopy(existing_form)
            output[index] = refreshed
            break
    else:
        output.append(refreshed)
    output.sort(key=lambda row: int(row.get("analysis_round") or 0))
    if virtual_content_review_enabled(state):
        artifact = persist_iteration_metrics_snapshot(state, refreshed)
        if artifact:
            refreshed["iteration_metrics_artifact"] = artifact
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
        if (
            plateau_status.get("objective_source") == "ipip_facet_gates_v4"
            and (row.get("form_metrics") or {}).get("metric_framework")
            != "virtual_form_response_transmission_v4"
        ):
            annotated.append(row)
            continue
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

    ensure_measurement_allowed(state)
    if (
        virtual_content_review_enabled(state)
        and not fixed_facet_iteration_enabled(state)
        and int(state.get("psychometric_analysis_round") or 0) >= MAX_MEASUREMENTS
        and not state.get("deferred_replacement_measurement_pending")
    ):
        raise ValueError("首测及三轮虚拟内容复审已完成，且当前没有待测同槽位替代题")
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
    if virtual_content_review_enabled(state):
        content_evidence = deepcopy(state.get("item_content_evidence") or {})
        for item in candidates:
            item_id = str(item["item_id"])
            evidence = content_evidence.setdefault(item_id, {"status": "not_evaluated"})
            evidence["last_measured_version"] = item["version"]
            evidence["last_measurement_round"] = analysis_state["psychometric_analysis_round"]
            if evidence.get("status") == "pending_remeasurement":
                evidence["status"] = "virtually_remeasured"
            # A statistical retest does not invent an unperformed interview.
            evidence["evidence_scope"] = "exploratory_virtual_development_evidence"
        state_update["item_content_evidence"] = content_evidence
        analysis_state["item_content_evidence"] = content_evidence

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
    if fixed_facet_iteration_enabled(state):
        controller = record_facet_measurement(analysis_state, provisional)
        provisional.update(
            facet_iteration_state=deepcopy(controller),
            round_completed=controller["completed_rounds"] > state["facet_iteration_state"]["completed_rounds"],
        )
        history = _upsert_iteration_record(prior_history, provisional, state=analysis_state)
        return {
            **result,
            "state_update": {
                **state_update, "facet_iteration_state": controller,
                "psychometric_iteration_history": history, "psychometric_plateau_status": None,
                "deferred_replacement_measurement_pending": False,
            },
        }
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
        # The extra measurement was consumed; a later deferred replacement
        # may open the flag again when its same-slot transaction commits.
        "deferred_replacement_measurement_pending": False,
    }
    return {
        **result,
        "state_update": state_update,
        "summary": (
            f"{result.get('summary') or '心理测量分析完成'}"
            f" 已完成{development_round_batch_label(iteration_history, provisional)}临时组卷，"
            f"整卷指标状态={provisional.get('form_status', 'unknown')}。"
        ),
    }


def distribute_situation_quotas(
    total_situations: int,
    facet_item_counts: Mapping[str, int],
    *,
    minimum_situations: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Allocate expansion situations in proportion to explicit facet quotas.

    Each retained item needs the configured number of candidate references.
    The base allocation therefore reserves that many situations per requested
    item and puts the remaining expansion buffer back across facets without
    changing the retained-item quotas.
    """

    total_situations = int(total_situations)
    if total_situations < 1 or not facet_item_counts:
        raise ValueError("情境池总数和 facet 配额必须有效")
    quotas = {
        str(facet_id): int(quota)
        for facet_id, quota in facet_item_counts.items()
    }
    if any(quota < 1 for quota in quotas.values()):
        raise ValueError("facet 配额必须是正整数")
    minimum = {
        facet_id: INCREMENTAL_CANDIDATES_PER_CELL * quota
        for facet_id, quota in quotas.items()
    }
    if minimum_situations:
        for facet_id, required in minimum_situations.items():
            if facet_id in minimum:
                minimum[facet_id] = max(minimum[facet_id], int(required))
    minimum_total = sum(minimum.values())
    if total_situations < minimum_total:
        raise ValueError(
            "情境扩展池不足以为每个保留题提供所需候选情境引用"
        )
    result = dict(minimum)
    remaining = total_situations - minimum_total
    facet_ids = list(result)
    index = 0
    while remaining:
        result[facet_ids[index % len(facet_ids)]] += 1
        remaining -= 1
        index += 1
    return result


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

    ensure_measurement_allowed(state)
    if (
        virtual_content_review_enabled(state)
        and not fixed_facet_iteration_enabled(state)
        and int(state.get("psychometric_analysis_round") or 0) >= MAX_MEASUREMENTS
        and not state.get("deferred_replacement_measurement_pending")
    ):
        raise ValueError("首测及三轮虚拟内容复审已完成，且当前没有待测同槽位替代题")

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


def _select_fixed_facet_iteration(state: PSJTState) -> dict[str, Any]:
    """Freeze qualifying versions, then investigate only unfinished facets."""
    controller = state["facet_iteration_state"]
    items = {str(item["item_id"]): deepcopy(dict(item)) for item in state.get("frozen_item_bank") or []}
    statistics = state.get("item_statistics") or {}
    batch = int(state.get("psychometric_analysis_round") or 0)
    record = next((row for row in state.get("psychometric_iteration_history") or []
                   if int(row.get("analysis_round") or 0) == batch), None)
    if not items or not isinstance(record, Mapping):
        raise ValueError("Fixed-facet selection requires the committed measurement ledger")
    dispositions = deepcopy(state.get("item_final_dispositions") or {})
    locks = dict(state.get("locked_retained_item_versions") or {})
    for item_id, item in items.items():
        if iteration_gates_pass(statistics.get(item_id) or {}):
            if locks.get(item_id) != item["version"] or not (dispositions.get(item_id) or {}).get("qualification_snapshot"):
                dispositions[item_id] = {
                    "status": "qualified_locked", "item_version": item["version"],
                    "retention_basis": "four_iteration_gates_passed",
                    "qualification_analysis_round": batch,
                    "qualification_response_data_ref": record.get("response_data_ref"),
                    "qualification_snapshot": deepcopy(statistics[item_id]),
                }
            locks[item_id] = item["version"]
    base_update = {
        "locked_retained_item_versions": locks, "item_final_dispositions": dispositions,
        "psychometric_plateau_status": None, "items_to_regenerate": [],
        "psychometric_repair_confirmation": None, "active_psychometric_repair": None,
    }
    if controller["status"] == "paused":
        return {"state_update": {
            **base_update, "status": "stopped", "items_to_revise": [],
            "selection_results": {"status": "facet_iteration_paused", "reason": controller["paused_reason"]},
            "virtual_content_review_stop_reason": controller["paused_reason"],
        }}
    if controller["status"] == "complete":
        selected_ids = [item_id for snapshot in controller["baseline_facets"].values()
                        for item_id in snapshot["selected_item_ids"]]
        if len(set(selected_ids)) != len(selected_ids) or any(item_id not in items for item_id in selected_ids):
            raise ValueError("Completed facet forms cannot be resolved to the current item bank")
        coverage_cells = []
        for cell in (state.get("blueprint") or {}).get("cells") or []:
            cell_ids = [item_id for item_id in selected_ids
                        if items[item_id].get("blueprint_cell_id") == cell["cell_id"]]
            planned = int(cell.get("planned_retention_count") or 0)
            coverage_cells.append({"blueprint_cell_id": cell["cell_id"], "planned_retention_count": planned,
                                   "selected_item_ids": cell_ids, "selected_count": len(cell_ids),
                                   "passed": len(cell_ids) == planned})
        if not coverage_cells or not all(cell["passed"] for cell in coverage_cells):
            raise ValueError("Completed facet forms no longer satisfy the fixed blueprint")
        for item_id in selected_ids:
            if not iteration_gates_pass(statistics.get(item_id) or {}):
                dispositions[item_id] = {
                    "status": "facet_form_retained", "item_version": items[item_id]["version"],
                    "retention_basis": "three_successful_facet_rounds", "monitoring_pass": False,
                    "source_analysis_round": batch,
                }
        return {"state_update": {
            **base_update, "items_to_revise": [],
            "blueprint_coverage": {"passed": True, "cells": coverage_cells,
                                   "selected_total": len(selected_ids), "available_total": len(items)},
            "selection_reasons": {item_id: dispositions[item_id]["retention_basis"] for item_id in selected_ids},
            "selected_items": [items[item_id] for item_id in selected_ids],
            "reserve_items": [item for item_id, item in items.items()
                              if item_id not in selected_ids and iteration_gates_pass(statistics.get(item_id) or {})],
            "selection_results": {"status": "ready_for_assembly", "iteration_policy_version": controller["policy_version"],
                                  "completed_rounds": 3, "selected_count": len(selected_ids)},
            "virtual_content_review_stop_reason": "three_successful_facet_repair_rounds_completed",
        }}
    if controller["status"] != "awaiting_repair":
        raise ValueError("Cannot plan a second repair before the committed changes are measured")
    selected_ids = set(record.get("form_item_ids") or [])
    comparisons = {row["sjt_facet_id"]: row for row in controller.get("comparisons") or []}
    queue = []
    for facet_id in controller["failed_facet_ids"]:
        members = [item for item in items.values() if str(item.get("target_dimension_id")) == facet_id]
        candidates, thaw_selection = facet_repair_candidates(members, statistics)
        if not candidates and thaw_selection is None:
            candidates = [item for item in members if str(item["item_id"]) in selected_ids]
        if not candidates:
            raise ValueError(f"Failed facet has no version-bound repair candidates: {facet_id}")
        for item in candidates:
            item_id = str(item["item_id"])
            entry = build_virtual_content_repair_entry(
                state, item, revision_round=int((state.get("psychometric_repair_rounds") or {}).get(item_id) or 0) + 1,
            )
            failure = {
                "facet_id": facet_id, "development_round": controller["development_round"],
                "comparison": deepcopy(comparisons.get(facet_id) or {}),
                "baseline_metrics": deepcopy(controller["baseline_facets"][facet_id]["metrics"]),
                "required_change": (
                    "All three facet validity values must strictly increase and alpha/ICC must reach .70; "
                    "qualified items in the bottom quartile of any active item-validity metric receive a full repair."
                ),
            }
            if thaw_selection is not None:
                failure["qualified_item_thaw"] = True
                failure["thaw_selection"] = deepcopy(thaw_selection)
            entry.update(facet_level_failure=True, facet_id=facet_id,
                         development_round=controller["development_round"], facet_form_failure=failure)
            entry["diagnosis_evidence"]["facet_form_failure"] = failure
            if thaw_selection is not None:
                entry["qualified_item_thaw"] = True
                entry["thaw_selection"] = deepcopy(thaw_selection)
            queue.append(entry)
    return {"state_update": {
        **base_update, "items_to_revise": queue, "selection_results": None,
        "selected_items": [], "reserve_items": [],
        "scenario_repair_progress": {
            str(entry["item_id"]): deepcopy((state.get("scenario_repair_progress") or {}).get(str(entry["item_id"])))
                                           or {"status": "planned", "stage": "planning", "rewrite_count": 0,
                                               "archive_ref": entry.get("content_review_archive_ref")}
            for entry in queue
        },
    }}


async def execute_item_selection_with_diagnosis(
    state: PSJTState,
) -> dict[str, Any]:
    """Diagnose flagged items and queue all confirmed edits for one item."""
    if fixed_facet_iteration_enabled(state) and virtual_content_review_enabled(state):
        return _select_fixed_facet_iteration(state)
    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("心理测量诊断前缺少冻结题库")
    statistics = state.get("item_statistics") or {}
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    virtual_review = virtual_content_review_enabled(state)
    defer_after_rounds = PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS
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
    continuing_existing_batch = bool(queued_item_ids) and not virtual_review

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
    existing_confirmation = state.get("psychometric_repair_confirmation")
    if (
        isinstance(existing_confirmation, Mapping)
        and existing_confirmation.get("status") == "pending"
        and (existing_confirmation.get("atomic_repair_advice") or {}).get("protocol") == "scenario_detection_first_v1"
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
            "retention_basis": "four_iteration_gates_passed",
            "warning_reason": None,
            "item_version": item_version,
            "qualification_analysis_round": current_iteration,
            "qualified_at_repair_round": int(rounds.get(item_id, 0)),
            "qualification_snapshot": deepcopy(statistics.get(item_id) or {}),
        }
        reasons[item_id] = reason

    for item_id, item in item_by_id.items():
        item_statistics = statistics.get(item_id) or {}
        requires_diagnosis = (not iteration_gates_pass(item_statistics)
                              if virtual_review else item_requires_psychometric_diagnosis(item_statistics))
        if plateau_reached:
            existing_disposition = dispositions.get(item_id)
            if isinstance(existing_disposition, Mapping) and existing_disposition.get(
                "status"
            ) in {"pending_sme_review", "eliminated"}:
                continue
            existing_basis = (
                existing_disposition.get("retention_basis")
                if isinstance(existing_disposition, Mapping)
                else None
            )
            previously_gate_qualified = (
                existing_basis == "four_iteration_gates_passed"
            )
            retention_basis = (
                existing_basis
                if previously_gate_qualified
                else (
                    "four_iteration_gates_passed"
                    if not requires_diagnosis
                    else "plateau_retained"
                )
            )
            qualification_snapshot = (
                deepcopy(existing_disposition.get("qualification_snapshot"))
                if previously_gate_qualified
                and isinstance(existing_disposition.get("qualification_snapshot"), Mapping)
                else deepcopy(statistics.get(item_id) or {})
            )
            retained.append(deepcopy(item))
            item_version = int(item.get("version") or 0)
            locked_versions[item_id] = item_version
            dispositions[item_id] = {
                "status": "qualified_locked",
                "retention_basis": retention_basis,
                "warning_reason": "整卷指标达到平台期，停止继续自动返修。",
                "item_version": item_version,
                "qualification_analysis_round": (
                    current_iteration
                    if retention_basis == "four_iteration_gates_passed"
                    else existing_disposition.get("qualification_analysis_round")
                    if isinstance(existing_disposition, Mapping)
                    else None
                ),
                "qualified_at_repair_round": int(rounds.get(item_id, 0)),
                "qualification_snapshot": qualification_snapshot,
                "monitoring_pass": not requires_diagnosis if virtual_review else False,
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
            and existing_queue_entry["atomic_repair_advice"].get("protocol") == (
                VIRTUAL_CONTENT_REVIEW_PROTOCOL if virtual_review else "scenario_detection_first_v1"
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
        root_id = (state.get("item_lineage") or {}).get(item_id, {}).get("root_item_id", item_id)
        completed_rounds = max(int(rounds.get(item_id, 0)), int(rounds.get(root_id, 0)))
        if locked_versions.get(item_id) == int(item.get("version") or 0):
            retained.append(deepcopy(item))
            monitored_pass = not requires_diagnosis
            dispositions[item_id] = {
                **deepcopy(dict(existing_disposition or {})),
                "status": "qualified_locked",
                "retention_basis": (existing_disposition or {}).get("retention_basis", "previously_qualified_locked"),
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
        if not requires_diagnosis:
            accept(
                item_id,
                item,
                reason=(
                    "合并分数档后的CITC及三项单题IPIP指标均达到四项虚拟迭代阈值；"
                    "条件目标相关与两项VTS仅作诊断。"
                ),
            )
            continue
        if virtual_review:
            if current_iteration >= MAX_MEASUREMENTS or completed_rounds >= defer_after_rounds:
                # A deferred item is a discarded candidate, never a temporary
                # form fill.  Queue a replacement-only transaction so the
                # same blueprint slot is occupied again and measured next.
                if replacement_capacity_available(state, str(item_id)):
                    queue_entry = build_virtual_content_repair_entry(
                        state, item, revision_round=completed_rounds + 1,
                    )
                    queue_entry.update(
                        {
                            "action": "replace",
                            "queue_status": "deferred_replenishment",
                            "deferred_replacement_only": True,
                            "completed_repair_rounds": completed_rounds,
                            "defer_after_rounds": defer_after_rounds,
                            "diagnosis_status": "repair_rounds_exhausted",
                            "atomic_repair_advice": {
                                "protocol": VIRTUAL_CONTENT_REVIEW_PROTOCOL,
                                "decision": "defer",
                                "summary": (
                                    f"已完成 {completed_rounds} 轮返修仍未达标，"
                                    "删除原题并在同一蓝图槽位生成替代题。"
                                ),
                                "repair_tasks": [],
                            },
                        }
                    )
                    repair_queue.append(queue_entry)
                    repairs.append(queue_entry)
                    reasons[item_id] = (
                        "返修额度耗尽，删除原题并在同一蓝图槽位补题；"
                        "替代题将在下一轮虚拟测量。"
                    )
                    continue
                dispositions[item_id] = {
                    "status": "deferred_decision", "item_version": item["version"],
                    "source_analysis_round": current_iteration,
                    "diagnosis_status": "repair_rounds_exhausted",
                    "completed_repair_rounds": completed_rounds,
                    "reason": (
                        "返修额度和同槽位补题额度均已耗尽；"
                        "不将未通过题目纳入组卷。"
                    ),
                }
                reasons[item_id] = dispositions[item_id]["reason"]
                continue
            if (isinstance(existing_disposition, Mapping)
                    and existing_disposition.get("status") == "deferred_decision"
                    and existing_disposition.get("item_version") == item["version"]
                    and existing_disposition.get("source_analysis_round") == current_iteration):
                continue
            if (isinstance(existing_queue_entry, Mapping)
                    and existing_queue_entry.get("repair_protocol") == VIRTUAL_CONTENT_REVIEW_PROTOCOL
                    and existing_queue_entry.get("source_analysis_round") == current_iteration
                    and (existing_queue_entry.get("diagnosis_evidence") or {}).get("current_item") == item):
                repair_queue.append(deepcopy(dict(existing_queue_entry)))
                repairs.append(deepcopy(dict(existing_queue_entry)))
                continue
            queue_entry = build_virtual_content_repair_entry(
                state, item, revision_round=completed_rounds + 1,
            )
            repair_queue.append(queue_entry)
            repairs.append(queue_entry)
            reasons[item_id] = "当轮四项单题门槛未通过，进入材料核查、虚拟专家审题、认知访谈和证据诊断。"
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
        queue_entry = build_scenario_repair_entry(
            state, item, revision_round=completed_rounds + 1,
        )
        repair_queue.append(queue_entry)
        repairs.append(queue_entry)
        reasons[item_id] = "按正式指标确定检测组，先检测情境再修改选项。"

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
            and set(provisional_form_optimizer.get("selected_item_ids") or []) <= {
                str(item["item_id"]) for item in retained
            }
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
    if plateau_finalized and not blueprint_coverage.get("passed") and not virtual_review:
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
    virtual_stop_reason = None
    if virtual_review and not repair_queue:
        if plateau_finalized:
            virtual_stop_reason = "plateau_retained"
        elif current_iteration > MAX_MEASUREMENTS:
            virtual_stop_reason = "deferred_replacement_measurement_completed"
        elif current_iteration >= MAX_MEASUREMENTS:
            virtual_stop_reason = "three_content_review_rounds_completed_without_deferred_items"
        elif any(row.get("status") == "deferred_decision" for row in dispositions.values()):
            virtual_stop_reason = "no_evidence_supported_edit"
        else:
            virtual_stop_reason = "all_items_qualified_or_locked"
        if not blueprint_coverage.get("passed"):
            status = "virtual_review_complete"
    if provisional_iteration is not None:
        if virtual_review:
            provisional_iteration["item_dispositions"] = deepcopy(dispositions)
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
            "scenario_repair_progress": {
                str(entry["item_id"]): deepcopy(
                    (state.get("scenario_repair_progress") or {}).get(str(entry["item_id"]))
                    or {"status": "planned", "stage": "planning", "rewrite_count": 0,
                        "archive_ref": entry.get("content_review_archive_ref") or entry.get("scenario_archive_ref")}
                ) for entry in repair_queue
                if entry.get("repair_protocol") in {"scenario_detection_first_v1", VIRTUAL_CONTENT_REVIEW_PROTOCOL}
            },
            "items_deferred_for_revision": [],
            "selection_results": None if repair_queue else {
                "status": status,
                "plateau_finalized": bool(plateau_finalized),
                "retained_count": len(retained),
                "repair_count": len(repair_queue),
                "selected_count": len(selected_items),
                "reserve_count": len(reserve_items),
                "mode": (form_optimizer or {}).get("mode") if isinstance(form_optimizer, Mapping) else None,
                "temporary_unqualified_item_ids": deepcopy(
                    (form_optimizer or {}).get("temporary_unqualified_item_ids")
                    if isinstance(form_optimizer, Mapping) else []
                ),
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
                if repairs and not virtual_review and isinstance(repair_queue[0].get("atomic_repair_advice"), Mapping)
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
            **({"virtual_content_review_stop_reason": virtual_stop_reason} if virtual_review else {}),
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
        "domain_ids",
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
    """Fill independent fixed slots together without exposing IDs to the model.

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
    async def fill_one(ordinal: int, specification_id: str) -> None:
        nonlocal max_attempt_depth
        if specification_id in valid:
            return
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
    await gather_all(*(fill_one(index, key) for index, key in enumerate(slot_order, start=1)))
    return valid, max_attempt_depth


async def execute_fixed_blueprint(state: PSJTState) -> dict:
    """Prepare evidence, expansion, and the immutable two-way table."""

    specification = state.get("test_specification")
    if not isinstance(specification, Mapping):
        raise ValueError("建立题目计划前缺少 TestSpecification")
    profile = resolve_specification_profile(specification)
    corpus = load_ipip_corpus()
    evidence_bundles = await gather_all(*(
        ensure_behavior_evidence(
            str(facet["facet_id"]),
            corpus,
            # A cross-domain/multi-facet request must be executable even when
            # only part of the registry has curated evidence. The fallback is
            # explicitly marked on each facet and remains development-only;
            # single-facet runs retain the curated-evidence gate.
            allow_legacy_fallback=(
                len(profile["facets"]) > 1
                or profile.get("selection_level") == "inventory"
            ),
        ) for facet in profile["facets"]
    ))
    bundles = {
        str(facet["facet_id"]): bundle
        for facet, bundle in zip(profile["facets"], evidence_bundles, strict=True)
    }
    profile = attach_behavior_evidence(profile, bundles)
    retention_total = int(specification["final_item_count"])
    facet_quotas = resolve_facet_item_counts(specification, profile)
    generation_total = required_generation_total(retention_total)
    behavior_counts = {
        str(facet["facet_id"]): len(facet.get("behavior_evidence") or [])
        for facet in profile["facets"]
    }
    base_expansion_situation_total = required_expansion_situation_total(
        retention_total
    )
    pairability_minimums = {
        facet_id: INCREMENTAL_CANDIDATES_PER_CELL * int(facet_quotas[facet_id]) + max(0, behavior_count - 1)
        for facet_id, behavior_count in behavior_counts.items()
    }
    expansion_situation_total = max(
        base_expansion_situation_total,
        sum(pairability_minimums.values()),
    )
    situation_quotas = distribute_situation_quotas(
        expansion_situation_total,
        facet_quotas,
        minimum_situations=pairability_minimums,
    )
    expansions = await gather_all(*(
            ensure_facet_expansion(
                run_id=state["run_id"],
                facet=facet,
                behavior_evidence=facet["behavior_evidence"],
                target_population=str(specification["target_population"]),
                output_language=str(specification["output_language"]),
                required_situation_count=situation_quotas[str(facet["facet_id"])],
            )
            for facet in profile["facets"]
    ))
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
            facet_retention_quotas=facet_quotas,
            retry_feedback=retry_feedback,
        )
        try:
            proposal = repair_blueprint_proposal(
                proposal,
                profile,
                expansions,
                facet_quotas,
            )
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
    fallback_count = sum(
        1
        for bundle in bundles.values()
        if str(getattr(bundle, "generated_at", "")).startswith(
            "legacy_registry_fallback:"
        )
    )
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
            f"计划最终保留 {specification['final_item_count']} 题，facet 配额为 "
            f"{facet_quotas}。"
            + (
                f"其中 {fallback_count} 个 facet 使用未完成 SME 审查的注册表证据兜底，"
                "仅用于开发期生成。"
                if fallback_count
                else ""
            )
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
    """Legacy candidate-only administration, outside fixed-cohort rounds."""
    if fixed_facet_iteration_enabled(state):
        raise ValueError("Fixed-cohort candidates must be committed and measured in a scheduled batch")

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
    progress_reporter: Any | None = None,
) -> dict[str, Any]:
    """Stage one repair-only candidate; never run a local formal retest."""
    from sjt_system.evaluation.scenario_detection import (
        archive_path, make_model_invoker, run_scenario_repair,
    )
    from pathlib import Path

    item = deepcopy(state.get("current_item") or {})
    if not item:
        raise ValueError("scenario repair requires current_item")
    invoke, model_info = make_model_invoker(state)
    from sjt_system.evaluation.repair_reconstruction import make_rebuilder
    progress = await run_scenario_repair(
        item=item,
        evidence=diagnosis_evidence,
        path=(Path(active_psychometric_repair["scenario_archive_ref"])
              if active_psychometric_repair.get("scenario_archive_ref")
              else archive_path(state, item, int(active_psychometric_repair.get("revision_round") or 1))),
        invoke=invoke,
        model_info=model_info,
        knowledge_snapshot=state.get("_repair_knowledge_snapshot"),
        source_context={"run_id": state.get("run_id"),
                        "psychometric_analysis_round": state.get("psychometric_analysis_round", 0)},
        rebuild=make_rebuilder(state, item, state.get("_repair_knowledge_snapshot")),
        progress=progress_reporter,
    )
    if progress_reporter is None:
        emit_progress({
            "type": "psychometric_subagent_progress",
            "item_id": item["item_id"],
            "status": "completed" if progress["status"] == "ready" else "failed",
            "stage": progress["stage"],
            "rewrite_count": progress["rewrite_count"],
            "archive_ref": progress["archive_ref"],
            "message": "情境检测及选项修改已暂存" if progress["status"] == "ready" else "情境检测暂停，整批不提交",
        })
    return {
        "state_update": {
            **dict(program_update),
            "current_item": progress.get("candidate"),
            "active_psychometric_repair": {
                **deepcopy(dict(active_psychometric_repair)),
                "scenario_detection": progress,
            },
        },
        "repair_attempt_count": 0,
        "summary": "候选已暂存，等待整批提交" if progress["status"] == "ready" else str(progress.get("pause_reason")),
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


async def prepare_repair_knowledge_review(state: PSJTState) -> dict[str, Any]:
    """Learn from the complete formal review queue before repair/defer decisions."""
    from sjt_system.evaluation.repair_knowledge import KnowledgeSession
    session = None
    try:
        entries = [*(state.get("items_to_revise") or []), *(state.get("items_to_regenerate") or [])]
        session = KnowledgeSession(state, entries)
        await session.prepare()
        return {"repair_knowledge_state": session.summary()}
    except Exception as exc:
        if session is not None:
            session.pause("review", exc)
        return {"repair_knowledge_state": session.summary() if session else {
                    "stage": "review", "status": "paused", "error": str(exc)},
                "scenario_repair_pause": {"recovery_version": 2, "failures": {"knowledge": str(exc)},
                    "knowledge_pending": True, "limit_reached": False,
                    "message": "审题知识归纳暂停，返修和外层确认尚未推进。"}}


async def _execute_psychometric_repair_batch(
    *, action: str, route: PSJTRouteDecision, state: PSJTState,
) -> dict[str, Any]:
    """Stage all repairs, stop dispatch on failure, and commit only all-ready batches."""
    if virtual_content_review_enabled(state):
        from sjt_system.workflow.virtual_content_review import execute_virtual_content_review_batch
        return await execute_virtual_content_review_batch(state)
    from sjt_system.evaluation.scenario_detection import (
        DESIGN_LIMIT_REASON, DETECTION_PROTOCOL, PROTOCOL, is_current_ready,
    )
    from sjt_system.evaluation.repair_reconstruction import (
        is_budget_replenishment, replenish_exhausted_design, validate_replacement,
    )

    pending_entries = [*(state.get("items_to_regenerate") or []),
                       *(state.get("items_to_revise") or [])]
    exhausted = next((entry for entry in pending_entries if isinstance(entry, Mapping)
                      and entry.get("diagnosis_status") == "repair_rounds_exhausted"), None)
    if exhausted is not None:
        # Outer-round decisions are not new repair candidates or legacy advice to replan.
        return {
            "state_update": {
                "psychometric_repair_confirmation": {
                    **deepcopy(dict(exhausted)), "status": "pending", "decision": None,
                },
            },
            "summary": "先处理原有外层返修次数耗尽的确认项，再启动整批情境检测。",
            "repair_attempt_count": 0,
        }

    item_by_id = {
        str(item["item_id"]): deepcopy(dict(item))
        for item in state.get("item_pool") or state.get("frozen_item_bank") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    unique_entries = []
    seen_ids = set()
    for raw in pending_entries:
        if not isinstance(raw, Mapping) or not raw.get("item_id"):
            continue
        item_id = str(raw["item_id"])
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        entry = deepcopy(dict(raw))
        if entry.get("repair_protocol") != PROTOCOL and item_id in item_by_id:
            entry = build_scenario_repair_entry(
                state, item_by_id[item_id], revision_round=int(entry.get("revision_round") or 1),
            )
        unique_entries.append(entry)
    if not unique_entries:
        raise ValueError("情境检测返修没有待处理题目")
    from sjt_system.runtime.checkpoint import DEFAULT_CHECKPOINT_ROOT, save_run_checkpoint
    from sjt_system.runtime.output_paths import scoped_output
    # A long-running batch must have a durable queue before its first model call.
    save_run_checkpoint(state, checkpoint_root=scoped_output("run_checkpoints", DEFAULT_CHECKPOINT_ROOT))
    from sjt_system.evaluation.repair_knowledge import KnowledgeSession
    knowledge = None
    try:
        knowledge = KnowledgeSession(state, unique_entries)
        await knowledge.prepare()
    except Exception as exc:
        if knowledge is not None:
            knowledge.pause("review", exc)
        summary = knowledge.summary() if knowledge is not None else {"status": "paused", "stage": "review", "error": str(exc)}
        return {"state_update": {
            "repair_knowledge_state": summary,
            "scenario_repair_pause": {"recovery_version": 2, "detection_protocol": DETECTION_PROTOCOL,
                "failures": {"knowledge": str(exc)}, "knowledge_pending": True,
                "limit_reached": False, "message": "审题知识归纳暂停；候选题库尚未修改。"},
            "items_to_revise": unique_entries, "items_to_regenerate": [],
        }, "summary": "知识归纳失败，保存后等待恢复", "repair_attempt_count": 0}
    batch_id = (
        f"scenario-repair-batch/{state.get('run_id')}/"
        f"{int(state.get('psychometric_analysis_round') or 0)}"
    )
    concurrency = len(unique_entries)
    semaphore = UnlimitedConcurrency()
    progress_lock = asyncio.Lock()
    progress_counts = {
        "started_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "active_count": 0,
    }
    progress_by_id = {key: deepcopy(value) for key, value in
                      (state.get("scenario_repair_progress") or {}).items() if key in seen_ids}
    batch_total = len(unique_entries)

    def progress_event(*, status: str, item_id: str = "batch", stage: str | None = None,
                       message: str | None = None, **extra: Any) -> None:
        event = {
            "type": "psychometric_subagent_progress",
            "status": status,
            "item_id": item_id,
            "batch_id": batch_id,
            "batch_total": batch_total,
            "concurrency": concurrency,
            "stage": stage,
            "message": message,
            **progress_counts,
            **extra,
        }
        emit_progress({key: value for key, value in event.items() if value is not None})

    async def update_progress_counts(*, item_id: str, started: bool = False,
                                     completed: bool = False, failed: bool = False) -> None:
        async with progress_lock:
            if started:
                progress_counts["started_count"] += 1
                progress_counts["active_count"] += 1
            if completed or failed:
                progress_counts["active_count"] = max(0, progress_counts["active_count"] - 1)
                progress_counts["completed_count"] += int(completed)
                progress_counts["failed_count"] += int(failed)

    async def snapshot_progress_counts() -> dict[str, int]:
        async with progress_lock:
            return dict(progress_counts)

    progress_event(
        status="batch_started",
        message="开始情境检测优先返修；全部就绪才提交，不执行逐题复测",
    )

    async def batch_heartbeat() -> None:
        try:
            interval = max(
                5.0,
                float(os.getenv("SJT_PSYCHOMETRIC_REPAIR_PROGRESS_INTERVAL_SECONDS", "15")),
            )
        except (TypeError, ValueError):
            interval = 15.0
        while True:
            await asyncio.sleep(interval)
            counts = await snapshot_progress_counts()
            progress_event(
                status="running",
                message="批次仍在运行，等待并发返修任务完成",
                **counts,
            )

    heartbeat_task = asyncio.create_task(batch_heartbeat())

    async def run_one(entry: Mapping[str, Any]) -> dict[str, Any]:
        item_id = str(entry["item_id"])
        async with semaphore:
            await update_progress_counts(item_id=item_id, started=True)
            counts = await snapshot_progress_counts()
            progress_event(
                status="started",
                item_id=item_id,
                stage="planning",
                message="子任务已开始，等待情境检测模型调用",
                **counts,
            )

            def report_stage(event: Mapping[str, Any]) -> None:
                progress_event(
                    status=str(event.get("status") or "editing"),
                    item_id=item_id,
                    stage=str(event.get("stage") or "unknown"),
                    message="情境检测模型调用中",
                    rewrite_count=event.get("rewrite_count", 0),
                    model_call_count=event.get("model_call_count", 0),
                    **progress_counts,
                )

            try:
                item = item_by_id[item_id]
                active = {**deepcopy(dict(entry)), "baseline_item": deepcopy(item),
                          "scenario_archive_ref": entry.get("scenario_archive_ref"),
                          "baseline_profile": deepcopy((state.get("item_pattern_profiles") or {}).get(item_id)),
                          "baseline_analysis_snapshot": None}
                local_state = {**state, "current_item": deepcopy(item),
                               "_repair_knowledge_snapshot": knowledge.for_item(entry.get("diagnosis_evidence") or {}),
                               "current_blueprint_cell": _psychometric_batch_blueprint_cell(state, item),
                               "current_item_specification": _psychometric_batch_item_specification(state, item),
                               "current_item_review": None, "active_psychometric_repair": active,
                               "_psychometric_batch_id": batch_id}
                result = await _execute_psychometric_repair_item(
                    action="revise_item", route={"next_action": "revise_item", "reason": "情境检测返修",
                    "target_item_id": item_id, "target_blueprint_cell_id": item.get("blueprint_cell_id")},
                    state=local_state, item_specification=local_state["current_item_specification"],
                    active_psychometric_repair=active, atomic_advice=entry.get("atomic_repair_advice") or {},
                    diagnosis_evidence=entry.get("diagnosis_evidence") or {}, program_update={},
                    progress_reporter=report_stage,
                )
                progress = (result["state_update"].get("active_psychometric_repair") or {}).get("scenario_detection") or {}
                if progress.get("status") == "paused" and progress.get("pause_reason") == DESIGN_LIMIT_REASON:
                    progress_event(
                        status="replenishing", item_id=item_id, stage="blueprint",
                        message="设计预算耗尽，正在同一蓝图单元重新组装候选题",
                        **(await snapshot_progress_counts()),
                    )
                    progress = await replenish_exhausted_design(
                        local_state, item, progress, local_state["_repair_knowledge_snapshot"],
                    )
                    result["state_update"]["current_item"] = progress["candidate"]
                    result["state_update"]["active_psychometric_repair"]["scenario_detection"] = progress
                if progress.get("status") == "ready" and not is_current_ready(progress):
                    raise ValueError("候选尚未通过当前差异检测和选项修改协议，必须重新检测")
                accepted = is_current_ready(progress) or is_budget_replenishment(progress)
                await update_progress_counts(
                    item_id=item_id,
                    completed=accepted,
                    failed=not accepted,
                )
                counts = await snapshot_progress_counts()
                progress_event(
                    status="completed" if accepted else "failed",
                    item_id=item_id,
                    stage=progress.get("stage"),
                    message=("同槽位补题已暂存，等待整批提交" if is_budget_replenishment(progress)
                             else "情境检测及选项修改已暂存" if accepted else "情境检测暂停，整批不提交"),
                    rewrite_count=progress.get("rewrite_count", 0),
                    archive_ref=progress.get("archive_ref"),
                    **counts,
                )
                return {"item_id": item_id, "result": result, "error": progress.get("pause_reason")}
            except Exception as exc:
                await update_progress_counts(item_id=item_id, failed=True)
                counts = await snapshot_progress_counts()
                progress_event(
                    status="failed",
                    item_id=item_id,
                    stage="error",
                    message=str(exc),
                    **counts,
                )
                return {"item_id": item_id, "result": None, "error": str(exc)}

    # Wait for in-flight jobs to finish journaling; never cancel their partial records.
    try:
        outcomes = await asyncio.gather(*(run_one(entry) for entry in unique_entries))
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
    successful, failures = {}, {}
    candidate_ids = set(item_by_id)
    for outcome in outcomes:
        item_id = outcome["item_id"]
        update = ((outcome.get("result") or {}).get("state_update") or {})
        active = update.get("active_psychometric_repair") or {}
        progress = active.get("scenario_detection") or {}
        if progress:
            progress_by_id[item_id] = deepcopy(progress)
        elif item_id not in progress_by_id:
            progress_by_id[item_id] = {
                "status": "paused", "stage": "not_started", "pause_reason": outcome.get("error"),
            }
        candidate = update.get("current_item")
        try:
            if not (is_current_ready(progress) or is_budget_replenishment(progress)) or not isinstance(candidate, Mapping):
                raise ValueError(outcome.get("error") or "情境检测未完成")
            validate_replacement(item_by_id[item_id], candidate, progress)
            candidate_id = str(candidate["item_id"])
            if candidate_id != item_id and candidate_id in candidate_ids:
                raise ValueError(f"replacement ID already exists: {candidate_id}")
        except (KeyError, TypeError, ValueError) as exc:
            failures[item_id] = str(exc)
            continue
        candidate_ids.add(candidate_id)
        successful[item_id] = {"candidate": deepcopy(dict(candidate)), "active_repair": deepcopy(active)}

    knowledge_error = None
    try:
        await knowledge.finish(progress_by_id)
    except Exception as exc:
        knowledge_error = str(exc)
        knowledge.pause("post_repair", exc)
    if failures or knowledge_error:
        staged = {key: deepcopy(value) for key, value in
                  (state.get("scenario_repair_staged") or {}).items() if key in seen_ids}
        staged.update({item_id: deepcopy(payload["candidate"]) for item_id, payload in successful.items()})
        pause = {"recovery_version": 2, "detection_protocol": DETECTION_PROTOCOL,
                 "batch_id": batch_id, "failures": {**failures, **({"knowledge": knowledge_error} if knowledge_error else {})},
                 "knowledge_pending": bool(knowledge_error),
                 "limit_reached": not knowledge_error and any(row.get("rewrite_count", 0) >= row.get("max_rewrites", 5) for row in progress_by_id.values()),
                 "message": "自动恢复或设计升级额度已用尽；原题库、正式版本和正式作答均未更新，已完成候选已暂存。"}
        return {
            "state_update": {
                "repair_knowledge_state": knowledge.summary(),
                "scenario_repair_pause": pause, "scenario_repair_progress": progress_by_id,
                "scenario_repair_staged": staged,
                "items_to_revise": [{
                    **entry,
                    "scenario_archive_ref": (
                        (progress_by_id.get(str(entry["item_id"])) or {}).get("archive_ref")
                        or entry.get("scenario_archive_ref")
                    ),
                    "queue_status": "staged" if str(entry["item_id"]) in successful else "paused",
                }
                                    for entry in unique_entries],
                "items_to_regenerate": [],
                "psychometric_repair_batch_summary": {"batch_id": batch_id, "batch_total": len(unique_entries),
                    "completed_count": len(successful), "failed_count": len(failures), "status": "paused",
                    "remaining_count": len(failures), "concurrency": concurrency},
            },
            "summary": pause["message"], "repair_attempt_count": 0,
        }

    budget_replenished_ids = {item_id for item_id, payload in successful.items()
                              if is_budget_replenishment(payload["active_repair"]["scenario_detection"])}
    # Version advancement is part of the all-ready commit, not any internal rewrite.
    for item_id, payload in successful.items():
        replacement = (payload["active_repair"].get("scenario_detection") or {}).get("staged_design")
        payload["candidate"]["version"] = 1 if replacement else int(item_by_id[item_id].get("version") or 0) + 1

    staged_blueprint = deepcopy(state.get("blueprint") or {})
    staged_specs = deepcopy(state.get("item_specifications") or [])
    staged_skeletons = deepcopy(state.get("item_skeletons") or {})
    staged_reviews = deepcopy(state.get("skeleton_review_history") or {})
    lineage = deepcopy(state.get("item_lineage") or {})
    replacement_dispositions = {}
    for old_id, payload in successful.items():
        bundle = (payload["active_repair"].get("scenario_detection") or {}).get("staged_design")
        if not bundle:
            continue
        new_id = payload["candidate"]["item_id"]
        if any(s["specification_id"] == new_id for s in staged_blueprint.get("slots", [])):
            raise ValueError(f"replacement ID already exists: {new_id}")
        staged_blueprint.setdefault("slots", []).append(deepcopy(bundle["slot"]))
        cell = next(c for c in staged_blueprint["cells"] if c["cell_id"] == payload["candidate"]["blueprint_cell_id"])
        cell["planned_generation_count"] += 1
        staged_specs.append(deepcopy(bundle["specification"]))
        staged_skeletons[new_id] = deepcopy(bundle["skeleton"])
        staged_reviews[new_id] = [{"mode": "repair_reconstruction", "skeleton": deepcopy(bundle["skeleton"]),
                                   "content_review": deepcopy(bundle["review"])}]
        lineage[old_id] = {**lineage.get(old_id, {}), "root_item_id": bundle["root_item_id"],
                           "status": "eliminated" if old_id in budget_replenished_ids else "replaced",
                           "replaced_by_item_id": new_id}
        lineage[new_id] = {"root_item_id": bundle["root_item_id"], "replaces_item_id": old_id,
                           "replacement_number": bundle["replacement_number"]}
        replacement_dispositions[old_id] = {"status": "eliminated", "item_version": item_by_id[old_id]["version"],
                                             "replacement_item_id": new_id}

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
        profiles[payload["candidate"]["item_id"]] = build_item_pattern_profile(
            payload["candidate"],
            _psychometric_batch_item_specification({**state, "item_specifications": staged_specs}, payload["candidate"]),
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
    item_history = deepcopy(state.get("item_history") or {})
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    for entry in unique_entries:
        item_id = str(entry["item_id"])
        if item_id not in successful:
            continue
        active_repair = successful[item_id]["active_repair"]
        round_number = int(entry.get("revision_round") or 1)
        rounds[item_id] = round_number
        candidate_id = successful[item_id]["candidate"]["item_id"]
        if candidate_id != item_id:
            rounds[candidate_id] = 0
            item_history.setdefault(item_id, []).append({"event": "eliminated" if item_id in budget_replenished_ids else "replaced",
                "item": deepcopy(item_by_id[item_id]),
                "replacement_item_id": candidate_id, "recorded_at": utc_timestamp()})
        item_history.setdefault(candidate_id, []).append({
            "event": ("replenished" if item_id in budget_replenished_ids else
                      "reconstructed" if candidate_id != item_id else "revised"),
            "source": PROTOCOL, "recorded_at": utc_timestamp(),
            "item": deepcopy(successful[item_id]["candidate"]),
            "previous_version": item_by_id[item_id].get("version"),
            "scenario_archive_ref": (active_repair.get("scenario_detection") or {}).get("archive_ref"),
        })
        repair_history.append(
            {
                "event": "psychometric_item_repaired",
                "recorded_at": utc_timestamp(),
                "item_id": item_id,
                "revision_round": round_number,
                "action": "eliminate_replenish" if item_id in budget_replenished_ids else entry.get("action") or "revise_item",
                "resolution": "design_budget_replenishment" if item_id in budget_replenished_ids else "repaired",
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
                "repair_protocol": PROTOCOL,
                "knowledge_snapshot_id": knowledge.data["snapshot"]["snapshot_id"],
                "scenario_detection": deepcopy(active_repair.get("scenario_detection")),
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
        "blueprint": staged_blueprint,
        "item_specifications": staged_specs,
        "item_skeletons": staged_skeletons,
        "skeleton_review_history": staged_reviews,
        "item_lineage": lineage,
        "repair_knowledge_state": knowledge.summary(),
        "scenario_repair_pause": None,
        "scenario_repair_progress": progress_by_id,
        "scenario_repair_staged": {},
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
            **replacement_dispositions,
            **{
            str(item_id): deepcopy(dict(disposition))
            for item_id, disposition in (state.get("item_final_dispositions") or {}).items()
            if str(item_id) not in successful_ids and isinstance(disposition, Mapping)
            },
        },
        "item_pool": merged_pool,
        "item_pattern_profiles": profiles,
        "candidate_bank_audit": None,
        "psychometric_repair_rounds": rounds,
        "psychometric_repair_history": repair_history,
        "item_history": item_history,
        "psychometric_repair_batch_summary": {
            "batch_id": batch_id,
            "batch_total": len(unique_entries),
            "completed_count": len(successful),
            "replenished_count": len(budget_replenished_ids),
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
    if budget_replenished_ids:
        reset_update["removed_items"] = [*deepcopy(state.get("removed_items") or []),
            *(deepcopy(item_by_id[item_id]) for item_id in budget_replenished_ids)]
        reset_update["rejected_items"] = [*deepcopy(state.get("rejected_items") or []),
            *({"item": deepcopy(item_by_id[item_id]), "reason": "scenario_design_budget_exhausted"}
              for item_id in budget_replenished_ids)]
    emit_progress(
        {
            "type": "psychometric_subagent_progress",
            "status": "batch_completed",
            "batch_id": batch_id,
            "batch_total": len(unique_entries),
            "completed_count": len(successful),
            "replenished_count": len(budget_replenished_ids),
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
            f"完成 {len(successful)} 道题的情境检测与文本修改；"
            f"失败 {len(failures)} 道，之后统一对更新后的题库施测"
        ),
        "repair_attempt_count": 0,
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
    ):
        raise ValueError("心理测量返修必须通过整批情境检测提交，不能继续旧逐题任务；请从检查点恢复。")
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
    if action == "generate_items_batch":
        from sjt_system.workflow.item_batch import execute_item_generation_batch

        return await execute_item_generation_batch(state)
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
