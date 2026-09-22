"""Versioned workflow checkpoint persistence and resume support."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from time import sleep
from typing import Any
from uuid import uuid4

from sjt_system.authoring.blueprint import (
    BLUEPRINT_REMOVED_FIELDS,
    CELL_REMOVED_FIELDS,
)
from sjt_system.authoring.construct_registry import (
    construct_selection_from_profile,
    resolve_construct_profile,
)
from sjt_system.config import DEFAULT_OUTPUT_LANGUAGE
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.workflow.constants import PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS


CHECKPOINT_SCHEMA_VERSION = 20
CHECKPOINT_REPLACE_ATTEMPTS = 5
CHECKPOINT_REPLACE_BACKOFF_SECONDS = 0.05
_SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    CHECKPOINT_SCHEMA_VERSION,
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "run_checkpoints"
TERMINAL_STATUSES = {"completed", "stopped"}
_REMOVED_STATE_FIELDS = {
    "psychometric_defer_batch_eliminate",
    "current_expert_review_results",
    "current_review_results",
    "current_review_decision",
    "item_revision_counts",
    "max_item_revision_count",
    "current_item_regeneration_count",
    "max_item_regeneration_count",
    "psychometric_revision_queue",
    "psychometric_revision_round",
    "max_psychometric_revision_rounds",
    "psychometric_revision_history",
    "active_psychometric_diagnostic",
    "quality_constraints",
    "reproducibility_config",
    "generation_strategy_history",
    "requirements_ready_for_confirmation",
    "unconfirmed_requirement_fields",
    "ambiguous_requirement_fields",
    "requirement_questions",
    "requirement_suggestions",
    "requirements_round",
    "construct_model",
    "construct_profile_ref",
    "theory_search_queries",
    "theory_evidence",
    "theory_search_completed",
    "blueprint_review",
    "blueprint_revision_count",
    "blueprint_review_status",
    "blueprint_candidate_attempted_ids",
    "supplement_request",
    "supplement_history",
    "supplement_round",
    "max_supplement_rounds",
    "candidate_reserve_round",
    "max_candidate_reserve_rounds",
    "initial_candidate_reserve_exhausted",
    "max_candidate_replacement_rounds",
}
_RESUME_RESET_FIELDS = (
    "route",
    "pending_action",
    "pending_state_update",
    "pending_summary",
    "pending_state_changes",
    "pending_interaction",
    "user_decision",
    "user_feedback",
    "current_item_review",
    "current_item_repair_failure",
)


def _has_revised_event(state: Mapping[str, Any]) -> bool:
    current_item = state.get("current_item")
    item_id = (
        current_item.get("item_id")
        if isinstance(current_item, Mapping)
        else None
    )
    history = state.get("item_history")
    records = history.get(item_id) if isinstance(history, Mapping) else None
    return bool(
        isinstance(records, list)
        and any(
            isinstance(record, Mapping)
            and record.get("event") == "revised"
            for record in records
        )
    )


def _strip_removed_state_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(dict(state))
    if (
        migrated.get("pending_action") == "clarify_requirements"
        and not isinstance(migrated.get("pending_interaction"), Mapping)
    ):
        legacy_interaction = {
            "suggestions": deepcopy(
                migrated.get("requirement_suggestions") or []
            ),
            "questions": list(migrated.get("requirement_questions") or []),
            "unconfirmed_fields": list(
                migrated.get("unconfirmed_requirement_fields") or []
            ),
            "ambiguous_fields": list(
                migrated.get("ambiguous_requirement_fields") or []
            ),
            "ready_for_confirmation": bool(
                migrated.get("requirements_ready_for_confirmation")
            ),
        }
        migrated["pending_interaction"] = legacy_interaction
    for field in _REMOVED_STATE_FIELDS:
        migrated.pop(field, None)
    return migrated


def _migrate_v1_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    state = deepcopy(dict(payload["state"]))
    psychometric_revision_in_progress = bool(
        state.get("active_psychometric_diagnostic")
        or state.get("psychometric_revision_queue")
        or (
            isinstance(state.get("selection_results"), Mapping)
            and state["selection_results"].get("status")
            == "revision_required"
        )
    )

    if psychometric_revision_in_progress:
        # This working candidate was never accepted; item_pool is authoritative.
        state["current_item"] = None
        state["current_blueprint_cell"] = None
        state["current_item_specification"] = None
        state["selection_results"] = None
        state["route"] = None
        if (
            isinstance(state.get("frozen_item_bank"), list)
            and state.get("frozen_item_bank")
            and isinstance(state.get("item_statistics"), Mapping)
            and isinstance(state.get("blueprint"), Mapping)
        ):
            from sjt_system.evaluation.selection import run_item_selection

            state.update(run_item_selection(state)["state_update"])

    state["current_item_review"] = None
    state["review_process_status"] = "not_started"
    state["item_content_status"] = "not_evaluated"
    state["current_review_request_id"] = None
    state["current_review_item_id"] = None
    state["current_review_item_version"] = None
    state["current_review_retry_count"] = 0
    state["current_item_repair_attempted"] = (
        False
        if state.get("current_item") is None
        else _has_revised_event(state)
    )
    state["current_item_repair_failure"] = None
    state["current_item_revision_count"] = (
        0
        if state.get("current_item") is None
        else max(
            int(state.get("current_item_revision_count") or 0),
            int(state["current_item_repair_attempted"]),
        )
    )
    state.setdefault("current_item_rewrite_count", 0)
    state.setdefault("current_item_replacement_count", 0)
    state.setdefault("current_skeleton_repair_required", False)
    state.setdefault("max_item_replacement_attempts", 2)
    state.setdefault("max_item_revision_attempts", 3)
    state.setdefault("max_item_rewrite_rounds", 3)
    state = _strip_removed_state_fields(state)
    return {
        **deepcopy(dict(payload)),
        "schema_version": 2,
        "state": state,
    }


def _migrate_v2_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the approved situation-domain boundary to legacy specifications."""

    state = deepcopy(dict(payload["state"]))
    specification = state.get("test_specification")
    if isinstance(specification, dict) and not specification.get(
        "application_context"
    ):
        specification["application_context"] = (
            "与目标人群和测验用途匹配的综合情境"
        )
        sources = state.get("specification_sources")
        if not isinstance(sources, dict):
            sources = {}
            state["specification_sources"] = sources
        if state.get("requirements_confirmed"):
            sources["application_context"] = "system_default"
        else:
            sources["application_context"] = "inferred"
            state["requirements_ready_for_confirmation"] = False
            unconfirmed = set(
                state.get("unconfirmed_requirement_fields") or []
            )
            unconfirmed.add("application_context")
            state["unconfirmed_requirement_fields"] = sorted(unconfirmed)
            state["confirmed_requirement_fields"] = [
                field
                for field in state.get("confirmed_requirement_fields") or []
                if field != "application_context"
            ]
    return {
        **deepcopy(dict(payload)),
        "schema_version": 3,
        "state": state,
    }


def _migrate_v3_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the short-lived application-context requirement field."""

    state = deepcopy(dict(payload["state"]))
    specification = state.get("test_specification")
    if isinstance(specification, dict):
        specification.pop("application_context", None)
    sources = state.get("specification_sources")
    if isinstance(sources, dict):
        sources.pop("application_context", None)
    for field in (
        "confirmed_requirement_fields",
        "unconfirmed_requirement_fields",
        "ambiguous_requirement_fields",
    ):
        values = state.get(field)
        if isinstance(values, list):
            state[field] = [
                value for value in values if value != "application_context"
            ]
    suggestions = state.get("requirement_suggestions")
    if isinstance(suggestions, list):
        state["requirement_suggestions"] = [
            suggestion
            for suggestion in suggestions
            if not (
                isinstance(suggestion, Mapping)
                and suggestion.get("field") == "application_context"
            )
        ]
    return {
        **deepcopy(dict(payload)),
        "schema_version": 4,
        "state": state,
    }


def _migrate_v4_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove response type and apply the fixed 1–4 behavior-level scoring."""

    state = deepcopy(dict(payload["state"]))
    specification = state.get("test_specification")
    if isinstance(specification, dict):
        specification.pop("response_instruction_type", None)
        specification["scoring_method"] = "1-4行为高低计分"
    sources = state.get("specification_sources")
    if isinstance(sources, dict):
        sources.pop("response_instruction_type", None)
        sources["scoring_method"] = "system_locked"
    blueprint = state.get("blueprint")
    if isinstance(blueprint, dict):
        blueprint.pop("response_instruction_type", None)
        blueprint["scoring_method"] = "1-4行为高低计分"
    return {
        **deepcopy(dict(payload)),
        "schema_version": 5,
        "state": state,
    }


def _migrate_v5_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove retired requirement and response-format fields."""

    state = deepcopy(dict(payload["state"]))

    def clean_requirement_container(container: Any) -> None:
        if not isinstance(container, dict):
            return
        specification = container.get("test_specification")
        if isinstance(specification, dict):
            for field in (
                "test_purpose",
                "response_format",
                "explicit_constraints",
                "assumptions",
            ):
                specification.pop(field, None)
        sources = container.get("specification_sources")
        if isinstance(sources, dict):
            for field in (
                "test_purpose",
                "response_format",
                "explicit_constraints",
                "assumptions",
            ):
                sources.pop(field, None)

    clean_requirement_container(state)
    clean_requirement_container(state.get("pending_state_update"))

    blueprint = state.get("blueprint")
    if isinstance(blueprint, dict):
        blueprint.pop("response_format", None)

    assembled_test = state.get("assembled_test")
    respondent_form = (
        assembled_test.get("respondent_form")
        if isinstance(assembled_test, dict)
        else None
    )
    administration = (
        respondent_form.get("administration")
        if isinstance(respondent_form, dict)
        else None
    )
    if isinstance(administration, dict):
        administration.pop("response_format", None)

    return {
        **deepcopy(dict(payload)),
        "schema_version": 6,
        "state": state,
    }


def _migrate_v6_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Embed legacy construct and item slots in the unified blueprint."""

    state = deepcopy(dict(payload["state"]))
    blueprint = state.get("blueprint")
    construct_model = state.get("construct_model")
    item_specifications = state.get("item_specifications")
    if isinstance(blueprint, dict):
        if isinstance(construct_model, Mapping):
            blueprint.setdefault(
                "construct_model",
                deepcopy(dict(construct_model)),
            )
        if isinstance(item_specifications, list):
            blueprint.setdefault(
                "item_specifications",
                deepcopy(item_specifications),
            )

    if state.get("pending_action") in {
        "build_construct_model",
        "build_blueprint",
    }:
        for field in (
            "pending_action",
            "pending_state_update",
            "pending_summary",
            "pending_state_changes",
            "pending_interaction",
            "user_decision",
        ):
            state[field] = None
    route = state.get("route")
    if isinstance(route, dict) and route.get("next_action") == "build_construct_model":
        route["next_action"] = "build_blueprint"
        route["reason"] = "旧构念建模任务迁移为统一构念—题目细目表"
    if state.get("current_phase") in {"construct_modeling", "blueprint"}:
        state["current_phase"] = "construct_blueprint"

    return {
        **deepcopy(dict(payload)),
        "schema_version": 7,
        "state": state,
    }


def _stable_indicator_mapping(
    dimension_id: Any,
    indicators: Any,
) -> dict[str, str]:
    if isinstance(indicators, Mapping):
        return {
            str(key): str(value)
            for key, value in indicators.items()
            if isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
        }
    if not isinstance(indicators, list):
        return {}
    prefix = str(dimension_id or "dimension")
    return {
        f"{prefix}-ind-{index:02d}": value.strip()
        for index, value in enumerate(indicators, start=1)
        if isinstance(value, str) and value.strip()
    }


def _legacy_anchor_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        normalized = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        if normalized:
            return "；".join(normalized)
    return fallback


def _migrate_construct_model_v8(model: Any) -> Any:
    if not isinstance(model, dict):
        return model
    migrated = deepcopy(model)
    framework = migrated.get("situation_activation_framework")
    shared_contexts = (
        framework.get("recommended_contexts")
        if isinstance(framework, Mapping)
        else []
    )
    if not isinstance(shared_contexts, list):
        shared_contexts = []
    cultural_notes = (
        framework.get("cultural_adaptation_notes")
        if isinstance(framework, Mapping)
        else []
    )
    reading_notes = (
        framework.get("reading_level_notes")
        if isinstance(framework, Mapping)
        else []
    )
    dimensions = migrated.get("dimensions")
    if isinstance(dimensions, list):
        for dimension in dimensions:
            if not isinstance(dimension, dict):
                continue
            dimension_id = dimension.get("dimension_id")
            dimension["behavioral_indicators"] = _stable_indicator_mapping(
                dimension_id,
                dimension.get("behavioral_indicators"),
            )
            anchors = dimension.get("behavioral_anchors")
            if not isinstance(anchors, Mapping):
                low = _legacy_anchor_text(
                    dimension.get("low_anchor"),
                    "较少表现该维度行为",
                )
                medium = _legacy_anchor_text(
                    dimension.get("medium_anchor"),
                    "在部分条件下表现该维度行为",
                )
                high = _legacy_anchor_text(
                    dimension.get("high_anchor"),
                    "稳定表现该维度行为",
                )
                dimension["behavioral_anchors"] = {
                    "low": low,
                    "medium_low": f"{medium}（偏低兼容推导）",
                    "medium_high": f"{medium}（偏高兼容推导）",
                    "high": high,
                }
            dimension["recommended_contexts"] = list(
                dict.fromkeys(
                    [
                        context.strip()
                        for context in (
                            dimension.get("recommended_contexts")
                            or shared_contexts
                        )
                        if isinstance(context, str) and context.strip()
                    ]
                )
            )
            for field in (
                "high_anchor",
                "medium_anchor",
                "low_anchor",
                "boundary_notes",
                "social_desirability_risks",
            ):
                dimension.pop(field, None)
    overall = migrated.get("overall_behavioral_anchors")
    if isinstance(overall, Mapping):
        migrated["overall_behavioral_anchors"] = {
            level: _legacy_anchor_text(
                overall.get(level),
                f"总体{level}行为水平",
            )
            for level in ("low", "medium_low", "medium_high", "high")
        }
    for field in (
        "construct_level",
        "framework_rationale",
        "assumptions",
        "situation_activation_framework",
    ):
        migrated.pop(field, None)
    migrated["_v8_migration_notes"] = {
        "anchors": "旧三档锚点已拆分为四档兼容推导值，需独立审核确认",
        "cell_scenario_constraints": [
            *(
                cultural_notes
                if isinstance(cultural_notes, list)
                else []
            ),
            *(reading_notes if isinstance(reading_notes, list) else []),
        ],
    }
    return migrated


def _state_has_generated_items(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("current_item")
        or state.get("item_pool")
        or state.get("rejected_items")
        or state.get("frozen_item_bank")
        or state.get("selected_items")
    )


def _migrate_v7_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade pre-review runs to the staged v8 blueprint contract."""

    state = deepcopy(dict(payload["state"]))
    if state.get("pending_action") == "build_blueprint":
        for field in (
            "pending_action",
            "pending_state_update",
            "pending_summary",
            "pending_state_changes",
            "pending_interaction",
            "user_decision",
        ):
            state[field] = None
    if _state_has_generated_items(state):
        # Rewriting slots after items exist would invalidate traceability.
        state["blueprint_review"] = None
        state["blueprint_revision_count"] = 0
        state["blueprint_review_status"] = "legacy_unreviewed"
        return {
            **deepcopy(dict(payload)),
            "schema_version": 8,
            "state": state,
        }

    model = _migrate_construct_model_v8(state.get("construct_model"))
    blueprint = state.get("blueprint")
    if isinstance(blueprint, dict):
        blueprint_model = _migrate_construct_model_v8(
            blueprint.get("construct_model") or model
        )
        blueprint["construct_model"] = blueprint_model
        model = blueprint_model
        migration_notes = blueprint_model.pop("_v8_migration_notes", {})
        dimensions = {
            dimension.get("dimension_id"): dimension
            for dimension in blueprint_model.get("dimensions") or []
            if isinstance(dimension, Mapping)
        }
        extra_constraints = migration_notes.get(
            "cell_scenario_constraints",
            [],
        )
        cells_by_id: dict[str, Mapping[str, Any]] = {}
        for cell in blueprint.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            dimension = dimensions.get(cell.get("dimension_id"), {})
            cell["scenario_constraints"] = list(
                dict.fromkeys(
                    [
                        *(
                            cell.get("scenario_constraints")
                            if isinstance(
                                cell.get("scenario_constraints"),
                                list,
                            )
                            else []
                        ),
                        *(
                            extra_constraints
                            if isinstance(extra_constraints, list)
                            else []
                        ),
                    ]
                )
            )
            for field in CELL_REMOVED_FIELDS:
                cell.pop(field, None)
            if isinstance(cell.get("cell_id"), str):
                cells_by_id[cell["cell_id"]] = cell
        for row in blueprint.get("item_specifications") or []:
            if not isinstance(row, dict):
                continue
            cell = cells_by_id.get(row.get("blueprint_cell_id"), {})
            dimension = dimensions.get(cell.get("dimension_id"), {})
            indicators = dimension.get("behavioral_indicators") or {}
            old_indicator = row.pop("behavioral_indicator", None)
            if row.get("behavioral_indicator_id") not in indicators:
                matching = [
                    indicator_id
                    for indicator_id, text in indicators.items()
                    if text == old_indicator
                ]
                row["behavioral_indicator_id"] = (
                    matching[0] if matching else next(iter(indicators), "")
                )
            if not isinstance(row.get("behavioral_anchors"), Mapping):
                row["behavioral_anchors"] = {
                    "low": _legacy_anchor_text(
                        row.pop("low_anchor", None),
                        "较少表现目标行为",
                    ),
                    "medium_low": _legacy_anchor_text(
                        row.pop("medium_low_anchor", None),
                        "有限表现目标行为",
                    ),
                    "medium_high": _legacy_anchor_text(
                        row.pop("medium_high_anchor", None),
                        "较充分表现目标行为",
                    ),
                    "high": _legacy_anchor_text(
                        row.pop("high_anchor", None),
                        "稳定充分表现目标行为",
                    ),
                }
            for field in (
                "low_anchor",
                "medium_low_anchor",
                "medium_high_anchor",
                "high_anchor",
            ):
                row.pop(field, None)
        for field in BLUEPRINT_REMOVED_FIELDS:
            blueprint.pop(field, None)
        blueprint["version"] = 2
    if isinstance(model, dict):
        model.pop("_v8_migration_notes", None)
    state["construct_model"] = model
    if isinstance(blueprint, dict):
        state["item_specifications"] = deepcopy(
            blueprint.get("item_specifications") or []
        )
    state["blueprint_review"] = None
    state["blueprint_revision_count"] = 0
    state["blueprint_review_status"] = "pending"
    return {
        **deepcopy(dict(payload)),
        "schema_version": 8,
        "state": state,
    }


def _migrate_v8_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Move unfinished runs to the registry-driven v9 planning boundary.

    Existing generated items keep their historical blueprint unchanged so item
    traceability is not broken. Runs that have not generated items discard the
    old per-run construct/blueprint draft and rebuild it from the registry.
    """

    state = deepcopy(dict(payload["state"]))
    state.setdefault("construct_profile", None)
    state.setdefault("construct_profile_ref", None)
    if _state_has_generated_items(state):
        state["blueprint_review_status"] = "legacy_unreviewed"
        state["blueprint_review"] = None
        state["blueprint_revision_count"] = 0
    else:
        state["construct_model"] = None
        state["construct_profile"] = None
        state["construct_profile_ref"] = None
        state["blueprint"] = None
        state["item_specifications"] = []
        state["blueprint_review"] = None
        state["blueprint_revision_count"] = 0
        state["blueprint_review_status"] = "pending"
        state["blueprint_progress"] = {}
        state["current_blueprint_cell"] = None
        state["current_item_specification"] = None
        if state.get("current_phase") != "requirements":
            state["current_phase"] = "construct_blueprint"
        if state.get("requirements_confirmed"):
            state["theory_search_completed"] = True
    if state.get("pending_action") == "build_blueprint":
        for field in (
            "pending_action",
            "pending_state_update",
            "pending_summary",
            "pending_state_changes",
            "pending_interaction",
            "user_decision",
        ):
            state[field] = None
    return {
        **deepcopy(dict(payload)),
        "schema_version": 9,
        "state": state,
    }


def _migrate_v9_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add bounded initial-candidate reserve state without rewriting v4 plans."""

    state = deepcopy(dict(payload["state"]))
    state.setdefault("candidate_reserve_round", 0)
    state.setdefault("max_candidate_reserve_rounds", 3)
    state.setdefault("initial_candidate_reserve_exhausted", False)
    state.setdefault("current_item_revision_count", 0)
    state.setdefault("current_item_rewrite_count", 0)
    state.setdefault("current_skeleton_repair_required", False)
    state.setdefault("max_item_revision_attempts", 3)
    state.setdefault("max_item_rewrite_rounds", 3)
    return {
        **deepcopy(dict(payload)),
        "schema_version": 10,
        "state": state,
    }


def _migrate_v10_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy requirements, then rebuild planning in v12."""

    state = deepcopy(dict(payload["state"]))
    specification = state.get("test_specification")
    if isinstance(specification, Mapping):
        specification = dict(specification)
        if not isinstance(specification.get("construct_selection"), Mapping):
            target = specification.get("target_construct")
            if isinstance(target, str) and target.strip():
                try:
                    specification["construct_selection"] = (
                        construct_selection_from_profile(
                            resolve_construct_profile(target)
                        )
                    )
                except ValueError:
                    specification["construct_selection"] = None
        state["test_specification"] = {
            "construct_selection": specification.get("construct_selection"),
            "target_population": specification.get("target_population"),
            "final_item_count": specification.get("final_item_count"),
            "output_language": specification.get("output_language")
            if isinstance(specification.get("output_language"), str)
            else DEFAULT_OUTPUT_LANGUAGE,
        }
        sources = state.get("specification_sources")
        if isinstance(sources, Mapping):
            state["specification_sources"] = {
                field: deepcopy(sources[field])
                for field in (
                    "construct_selection",
                    "target_population",
                    "final_item_count",
                    "output_language",
                )
                if field in sources
            }
    return {
        **deepcopy(dict(payload)),
        "schema_version": 11,
        "state": state,
    }


def _migrate_v11_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Discard pre-v12 planning artifacts instead of mapping old semantics."""

    state = deepcopy(dict(payload["state"]))
    if isinstance(state.get("blueprint"), Mapping):
        reset = {
            "construct_profile": None,
            "blueprint": None,
            "current_blueprint_cell": None,
            "blueprint_progress": {},
            "blueprint_coverage": None,
            "item_skeletons": {},
            "skeleton_reviews": {},
            "skeleton_review_history": {},
            "skeleton_failures": {},
            "skeleton_slot_failure_pending": False,
            "item_specifications": [],
            "current_item_specification": None,
            "current_item": None,
            "item_pool": [],
            "item_pattern_profiles": {},
            "context_usage": {},
            "rejected_items": [],
            "current_item_review": None,
            "item_history": {},
            "item_bank_id": None,
            "item_bank_version": 0,
            "item_bank_fingerprint": None,
            "item_bank_frozen_at": None,
            "frozen_item_bank": [],
            "provisional_item_flags": {},
            "virtual_response_data_ref": None,
            "previous_virtual_response_data_ref": None,
            "virtual_response_summary": None,
            "virtual_response_item_bank_id": None,
            "virtual_response_item_bank_version": None,
            "item_statistics": {},
            "psychometric_round_result": None,
            "psychometric_iteration_history": [],
            "test_statistics": None,
            "factor_results": None,
            "irt_results": None,
            "dif_results": None,
            "selected_items": [],
            "reserve_items": [],
            "items_to_revise": [],
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "removed_items": [],
            "selection_reasons": {},
            "selection_results": None,
            "psychometric_selection_history": [],
            "locked_retained_item_versions": {},
            "best_assembly_candidate": None,
            "active_psychometric_repair": None,
            "psychometric_repair_rounds": {},
            "psychometric_repair_history": [],
            "assembled_test": None,
            "test_review_result": None,
            "test_revision_history": [],
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
            "completion_checks": {},
            "unmet_completion_conditions": [],
            "route": None,
            "pending_action": None,
            "pending_state_update": None,
            "pending_summary": None,
            "pending_state_changes": None,
        }
        state.update(reset)
        state["current_phase"] = "construct_blueprint"
    return {
        **deepcopy(dict(payload)),
        "schema_version": 14,
        "state": _strip_removed_state_fields(state),
    }


def _migrate_v12_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Restart unfinished legacy repair routing from current statistics."""

    state = deepcopy(dict(payload["state"]))
    selection = state.get("selection_results")
    unfinished_repair = bool(
        state.get("active_psychometric_repair")
        or state.get("items_to_revise")
        or state.get("items_to_regenerate")
        or (
            isinstance(selection, Mapping)
            and selection.get("status") in {"repair_required", "revision_required"}
        )
    )
    if unfinished_repair:
        state.update(
            {
                "route": None,
                "pending_action": None,
                "pending_state_update": None,
                "pending_summary": None,
                "pending_state_changes": None,
                "current_item": None,
                "current_blueprint_cell": None,
                "current_item_specification": None,
                "current_item_review": None,
                "active_psychometric_repair": None,
                "psychometric_repair_confirmation": None,
                "items_to_revise": [],
                "items_to_regenerate": [],
                "items_deferred_for_revision": [],
                "selected_items": [],
                "reserve_items": [],
                "selection_reasons": {},
                "selection_results": None,
                "item_final_dispositions": {},
                "blueprint_coverage": None,
                "assembled_test": None,
                "test_review_result": None,
                "final_test": None,
                "item_database_ref": None,
                "technical_report": None,
                "virtual_respondent_report": None,
            }
        )
    state.setdefault("item_final_dispositions", {})
    state.setdefault("psychometric_monitoring_warnings", [])
    state.setdefault("item_lineage", {})
    state.setdefault("virtual_sample_migration_events", [])
    state.setdefault("virtual_analysis_reconfiguration_reason", None)
    state.setdefault("psychometric_repair_confirmation", None)
    return {
        **deepcopy(dict(payload)),
        "schema_version": 14,
        "state": _strip_removed_state_fields(state),
    }


def _migrate_v13_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the per-item user-confirmation field introduced in v14."""

    state = deepcopy(dict(payload["state"]))
    state.setdefault("psychometric_repair_confirmation", None)
    return {
        **deepcopy(dict(payload)),
        "schema_version": 14,
        "state": _strip_removed_state_fields(state),
    }


def _virtual_sample_protocol_is_current(state: Mapping[str, Any]) -> bool:
    """Return whether resumable virtual evidence uses the score protocol."""

    from sjt_system.evaluation.respondents import matched_condition_sample_is_current
    from sjt_system.evaluation.simulation import VIRTUAL_RESPONSE_PROMPT_VERSION

    config = state.get("virtual_sample_config")
    respondents = state.get("virtual_respondents")
    if not matched_condition_sample_is_current(config, respondents):
        return False
    response_ref = state.get("virtual_response_data_ref")
    if response_ref is None:
        return True
    if not isinstance(response_ref, str) or not Path(response_ref).is_file():
        return False
    try:
        response_manifest = json.loads(
            Path(response_ref).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return False
    from sjt_system.evaluation.respondents import MATCHED_CONDITION_SCHEMA_VERSION
    if response_manifest.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION:
        return False
    if config.get("migrated_from_equal_score_tiers") is True:
        return bool(
            response_manifest.get("persona_modes") == ["score_profile"]
            and response_manifest.get("sample_size") == len(respondents)
        )
    return bool(
        response_manifest.get("prompt_version") == VIRTUAL_RESPONSE_PROMPT_VERSION
        and response_manifest.get("score_prompt_version")
        == config.get("prompt_version")
        and response_manifest.get("generator_version")
        == config.get("generator_version")
    )


def _migrate_equal_legacy_score_sample(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Legacy tier artifacts are retained but require explicit reconfiguration."""

    # The matched-condition protocol deliberately does not silently reinterpret
    # old tier respondents; frozen items and lineage remain in the checkpoint.
    return None

    from sjt_system.evaluation.respondents import (
        SCORE_PROFILE_GENERATOR_VERSION,
        SCORE_PROFILE_PROMPT_VERSION,
    )

    config = state.get("virtual_sample_config")
    respondents = state.get("virtual_respondents")
    if (
        not isinstance(config, Mapping)
        or config.get("schema_version") != 3
        or not isinstance(respondents, list)
        or not respondents
    ):
        return None
    specs = config.get("score_specs")
    tiers = config.get("score_tiers")
    if not isinstance(specs, list) or not specs or not isinstance(tiers, list):
        return None
    facet_specs = [
        deepcopy(dict(row))
        for row in specs
        if isinstance(row, Mapping) and row.get("level") == "facet"
    ]
    dimension_ids = [
        str(row.get("dimension_id") or "")
        for row in facet_specs
    ]
    if (
        not facet_specs
        or len(dimension_ids) != len(facet_specs)
        or not all(dimension_ids)
        or len(set(dimension_ids)) != len(dimension_ids)
    ):
        return None
    migrated_tiers: list[dict[str, Any]] = []
    for raw_tier in tiers:
        scores = raw_tier.get("score_means") if isinstance(raw_tier, Mapping) else None
        if not isinstance(scores, Mapping) or not set(dimension_ids).issubset(
            set(map(str, scores))
        ):
            return None
        values = [scores[dimension_id] for dimension_id in dimension_ids]
        if (
            not values
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
            or len({round(float(value), 12) for value in values}) != 1
        ):
            return None
        migrated_tiers.append(
            {
                "tier_id": str(raw_tier.get("tier_id") or ""),
                "facet_score": float(values[0]),
            }
        )
    if not migrated_tiers or any(not row["tier_id"] for row in migrated_tiers):
        return None
    target_ids = {
        str(item.get("target_dimension_id"))
        for item in (state.get("frozen_item_bank") or state.get("item_pool") or [])
        if isinstance(item, Mapping) and item.get("target_dimension_id")
    }
    non_target_ids = [value for value in dimension_ids if value not in target_ids]
    if not target_ids or not non_target_ids:
        return None
    diagnostics = deepcopy(config.get("generation_diagnostics") or {})
    diagnostics["input_correlation_filtering_authority"] = False
    diagnostics["note"] = "输入维度相关仅作诊断，不参与运行拒绝或题目过滤。"
    migrated_config = {
        **deepcopy(dict(config)),
        "schema_version": 4,
        "score_specs": facet_specs,
        "score_tiers": migrated_tiers,
        "non_target_facet_ids": non_target_ids,
        "prompt_version": SCORE_PROFILE_PROMPT_VERSION,
        "generator_version": SCORE_PROFILE_GENERATOR_VERSION,
        "generation_diagnostics": diagnostics,
        "migrated_from_equal_score_tiers": True,
        "migration_source_schema_version": 3,
    }
    migrated = deepcopy(dict(state))
    migrated["virtual_sample_config"] = migrated_config
    migrated["virtual_respondents"] = [
        {
            **deepcopy(dict(row)),
            "score_values": {
                dimension_id: min(
                    100.0,
                    max(0.0, float(row["score_values"][dimension_id])),
                )
                for dimension_id in dimension_ids
            },
        }
        for row in respondents
        if isinstance(row, Mapping)
        and isinstance(row.get("score_values"), Mapping)
        and set(dimension_ids).issubset(row["score_values"])
    ]
    if len(migrated["virtual_respondents"]) != len(respondents):
        return None
    migrated["virtual_sample_reconfiguration_reason"] = None
    migrated["virtual_sample_migration_events"] = [
        *deepcopy(state.get("virtual_sample_migration_events") or []),
        {
            "event": "equal_score_tiers_migrated",
            "recorded_at": utc_timestamp(),
            "source_schema_version": 3,
            "target_schema_version": 4,
            "raw_responses_preserved": bool(state.get("virtual_response_data_ref")),
        },
    ]
    return migrated


def _clear_legacy_analysis_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Invalidate old formula results and qualifications while preserving responses."""

    migrated = deepcopy(dict(state))
    migrated.update(
        {
            "item_statistics": {},
            "psychometric_round_result": None,
            "test_statistics": None,
            "factor_results": None,
            "irt_results": None,
            "dif_results": None,
            "selected_items": [],
            "reserve_items": [],
            "items_to_revise": [],
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "selection_reasons": {},
            "selection_results": None,
            "psychometric_selection_history": [],
            "locked_retained_item_versions": {},
            "psychometric_monitoring_warnings": [],
            "best_assembly_candidate": None,
            "active_psychometric_repair": None,
            "psychometric_repair_confirmation": None,
            "psychometric_repair_rounds": {},
            "psychometric_repair_history": [],
            "item_final_dispositions": {},
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
            "completion_checks": {},
            "unmet_completion_conditions": [],
            "virtual_analysis_reconfiguration_reason": (
                "旧心理测量公式和旧锁定资格已失效；保留兼容原始作答并按条件VTS重新分析。"
            ),
        }
    )
    return migrated


def _migrate_matched_form_retest_protocol(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Upgrade matched v6 samples while preserving item-development history."""

    from sjt_system.evaluation.respondents import (
        DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT,
        MATCHED_CONDITION_GENERATOR_VERSION,
        MATCHED_CONDITION_PROMPT_VERSION,
        MATCHED_CONDITION_SCHEMA_VERSION,
        matched_condition_sample_is_current,
    )

    config = state.get("virtual_sample_config")
    respondents = state.get("virtual_respondents")
    if (
        not isinstance(config, Mapping)
        or config.get("schema_version") != 6
        or config.get("sampling_design") != "matched_facet_conditions"
        or not isinstance(respondents, list)
        or not respondents
    ):
        return None
    upgraded_config = {
        **deepcopy(dict(config)),
        "schema_version": MATCHED_CONDITION_SCHEMA_VERSION,
        "generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
        "prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        "response_count_per_respondent_item": 1,
        "target_form_administration_count": (
            DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT
        ),
        "generation_diagnostics": {
            **deepcopy(dict(config.get("generation_diagnostics") or {})),
            "generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
            "target_form_administration_count": (
                DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT
            ),
        },
    }
    if not matched_condition_sample_is_current(upgraded_config, respondents):
        return None
    migrated = deepcopy(dict(state))
    migrated.update(
        {
            "virtual_sample_config": upgraded_config,
            "virtual_response_data_ref": None,
            "previous_virtual_response_data_ref": None,
            "virtual_response_summary": None,
            "virtual_response_item_bank_id": None,
            "virtual_response_item_bank_version": None,
            "item_statistics": {},
            "test_statistics": None,
            "psychometric_round_result": None,
            "psychometric_analysis_round": 0,
            "psychometric_iteration_history": [],
            "psychometric_plateau_status": None,
            "factor_results": None,
            "irt_results": None,
            "dif_results": None,
            "selected_items": [],
            "reserve_items": [],
            "items_to_revise": [],
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "selection_reasons": {},
            "selection_results": None,
            "best_assembly_candidate": None,
            "active_psychometric_repair": None,
            "psychometric_repair_confirmation": None,
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
            "completion_checks": {},
            "unmet_completion_conditions": [],
            "virtual_sample_reconfiguration_reason": None,
            "virtual_analysis_reconfiguration_reason": (
                "虚拟整卷指标已升级为Cronbach α、ICC、目标IPIP相关、"
                "Δmin和目标Hedges_g；旧R²、S与I_g仅保留为诊断字段；"
                "保留题目、正式题锁定及返修历史，重新执行虚拟施测。"
            ),
            "virtual_sample_migration_events": [
                *deepcopy(state.get("virtual_sample_migration_events") or []),
                {
                    "event": "matched_form_retest_protocol_upgrade",
                    "recorded_at": utc_timestamp(),
                    "source_schema_version": 6,
                    "target_schema_version": MATCHED_CONDITION_SCHEMA_VERSION,
                    "item_repair_history_preserved": True,
                    "locked_qualifications_preserved": True,
                    "old_iteration_curve_invalidated": True,
                },
            ],
        }
    )
    return migrated


def _prepare_diagnostic_output_reanalysis(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Refresh v8 diagnostic artifacts without revoking unchanged qualifications."""

    migrated = deepcopy(dict(state))
    prior_round = max(
        int(migrated.get("psychometric_analysis_round") or 0),
        len(migrated.get("psychometric_selection_history") or []),
        1,
    )
    migrated.update(
        {
            "psychometric_analysis_round": prior_round,
            "psychometric_round_result": None,
            "item_statistics": {},
            "test_statistics": None,
            "factor_results": None,
            "irt_results": None,
            "dif_results": None,
            "selected_items": [],
            "reserve_items": [],
            "items_to_revise": [],
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "selection_reasons": {},
            "selection_results": None,
            "psychometric_monitoring_warnings": [],
            "best_assembly_candidate": None,
            "active_psychometric_repair": None,
            "psychometric_repair_confirmation": None,
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
            "completion_checks": {},
            "unmet_completion_conditions": [],
            "virtual_analysis_reconfiguration_reason": (
                "评估展示与VTS选项频率证据已升级；保留原始作答、题目文本、"
                "正式题锁定资格及返修历史，并使用原始作答重新分析。"
            ),
            "virtual_sample_migration_events": [
                *deepcopy(migrated.get("virtual_sample_migration_events") or []),
                {
                    "event": "diagnostic_output_upgrade_pending",
                    "recorded_at": utc_timestamp(),
                    "source_evaluation_version": "sjt-evaluation-v8",
                    "target_evaluation_version": "sjt-evaluation-v9",
                    "raw_responses_preserved": bool(
                        migrated.get("virtual_response_data_ref")
                    ),
                    "locked_qualifications_preserved": bool(
                        migrated.get("locked_retained_item_versions")
                    ),
                },
            ],
        }
    )
    return migrated


def _invalidate_legacy_virtual_screening(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Force old persona/checkpoint evidence back through sample setup."""

    migrated = deepcopy(dict(state))
    form_retest_migration = _migrate_matched_form_retest_protocol(migrated)
    if form_retest_migration is not None:
        migrated = form_retest_migration
    legacy_migration = _migrate_equal_legacy_score_sample(migrated)
    if legacy_migration is not None:
        migrated = legacy_migration
    has_virtual_state = bool(
        migrated.get("virtual_sample_config")
        or migrated.get("virtual_respondents")
        or migrated.get("virtual_response_data_ref")
        or migrated.get("item_statistics")
        or migrated.get("test_statistics")
    )
    if not has_virtual_state:
        migrated.setdefault("virtual_sample_reconfiguration_reason", None)
        return migrated
    if _virtual_sample_protocol_is_current(migrated):
        from sjt_system.evaluation.psychometrics import (
            MEASUREMENT_EVALUATION_VERSION,
            PSYCHOMETRIC_FORMULA_VERSION,
        )

        statistics = migrated.get("test_statistics")
        diagnostic_upgrade_pending = any(
            isinstance(event, Mapping)
            and event.get("event") == "diagnostic_output_upgrade_pending"
            for event in migrated.get("virtual_sample_migration_events") or []
        ) and not statistics
        if (
            isinstance(statistics, Mapping)
            and statistics.get("formula_version") == PSYCHOMETRIC_FORMULA_VERSION
            and statistics.get("evaluation_version") == "sjt-evaluation-v8"
            and MEASUREMENT_EVALUATION_VERSION == "sjt-evaluation-v9"
        ):
            return _prepare_diagnostic_output_reanalysis(migrated)
        analysis_is_current = (
            not statistics
            or (
                isinstance(statistics, Mapping)
                and statistics.get("formula_version") == PSYCHOMETRIC_FORMULA_VERSION
                and statistics.get("evaluation_version") == MEASUREMENT_EVALUATION_VERSION
            )
        )
        has_legacy_qualification = bool(
            migrated.get("locked_retained_item_versions")
            or migrated.get("item_final_dispositions")
        ) and not statistics
        migrated.setdefault("virtual_sample_reconfiguration_reason", None)
        if analysis_is_current and (
            not has_legacy_qualification or diagnostic_upgrade_pending
        ):
            migrated.setdefault("virtual_analysis_reconfiguration_reason", None)
            return migrated
        return _clear_legacy_analysis_state(migrated)

    frozen = migrated.get("frozen_item_bank")
    migrated.update(
        {
            "virtual_sample_config": None,
            "virtual_respondents": [],
            "virtual_response_data_ref": None,
            "previous_virtual_response_data_ref": None,
            "virtual_response_summary": None,
            "virtual_response_item_bank_id": None,
            "virtual_response_item_bank_version": None,
            "virtual_sample_reconfiguration_reason": (
                "此检查点使用旧 tier/重复作答协议；必须重新选择目标、同域和跨域 facet，"
                "并配置共享正态分布的均值与SD。旧文件保留但不会静默复用。"
            ),
            "item_statistics": {},
            "psychometric_round_result": None,
            "test_statistics": None,
            "factor_results": None,
            "irt_results": None,
            "dif_results": None,
            "selected_items": [],
            "reserve_items": [],
            "items_to_revise": [],
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "removed_items": [],
            "selection_reasons": {},
            "selection_results": None,
            "psychometric_selection_history": [],
            "locked_retained_item_versions": {},
            "best_assembly_candidate": None,
            "active_psychometric_repair": None,
            "psychometric_repair_confirmation": None,
            "psychometric_repair_rounds": {},
            "psychometric_repair_history": [],
            "item_final_dispositions": {},
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
            "completion_checks": {},
            "unmet_completion_conditions": [],
            "rescore_pending_revalidation": False,
            "route": None,
            "pending_action": None,
            "pending_state_update": None,
            "pending_summary": None,
            "pending_state_changes": None,
        }
    )
    if isinstance(frozen, list) and frozen:
        migrated["item_pool"] = deepcopy(frozen)
    return migrated


def _migrate_v14_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Invalidate virtual evidence created before the score-only protocol."""

    return {
        **deepcopy(dict(payload)),
        "schema_version": 15,
        "state": _strip_removed_state_fields(
            _invalidate_legacy_virtual_screening(payload["state"])
        ),
    }


def _migrate_v15_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Adopt conditional VTS, migrating only compatible equal-score tiers."""

    return {
        **deepcopy(dict(payload)),
        "schema_version": 16,
        "state": _strip_removed_state_fields(
            _invalidate_legacy_virtual_screening(payload["state"])
        ),
    }


def _migrate_v16_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add round diagnostics and preserve v8 locked qualifications for reanalysis."""

    state = deepcopy(dict(payload["state"]))
    state.setdefault("psychometric_analysis_round", 0)
    state.setdefault("psychometric_round_result", None)
    state.setdefault("psychometric_iteration_history", [])
    return {
        **deepcopy(dict(payload)),
        "schema_version": 17,
        "state": _strip_removed_state_fields(
            _invalidate_legacy_virtual_screening(state)
        ),
    }


def _migrate_v17_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retire the former batch defer-elimination disposition mode."""

    state = deepcopy(dict(payload["state"]))
    state.pop("psychometric_defer_batch_eliminate", None)
    return {
        **deepcopy(dict(payload)),
        "schema_version": 18,
        "state": _strip_removed_state_fields(state),
    }


def _migrate_v18_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the explicit three-round-to-defer policy field.

    ``max_psychometric_repair_rounds`` is retained as a legacy state key so
    older callers/checkpoints can still be inspected, but it no longer
    controls admission, removal, or replacement decisions.
    """

    state = deepcopy(dict(payload["state"]))
    state["psychometric_repair_defer_after_rounds"] = (
        PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS
    )
    state.setdefault(
        "max_psychometric_repair_rounds",
        PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS,
    )
    return {
        **deepcopy(dict(payload)),
        "schema_version": 19,
        "state": _strip_removed_state_fields(state),
    }


def _migrate_v19_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add persistent whole-test iteration curve history."""

    state = deepcopy(dict(payload["state"]))
    state.setdefault("psychometric_iteration_history", [])
    state.setdefault("psychometric_plateau_status", None)
    state.setdefault("psychometric_plateau_patience", 2)
    state.setdefault("psychometric_plateau_min_delta", 0.01)
    return {
        **deepcopy(dict(payload)),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state": _strip_removed_state_fields(state),
    }


def save_run_checkpoint(
    state: Mapping[str, Any],
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> Path:
    """Atomically save one complete application-level workflow state."""

    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("运行状态缺少有效 run_id")
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{run_id}.json"
    temporary = root / f".{run_id}.{uuid4().hex}.json.tmp"
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "saved_at": utc_timestamp(),
        "is_terminal": state.get("status") in TERMINAL_STATUSES,
        "state": _strip_removed_state_fields(state),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            temporary.replace(target)
            return target
        except PermissionError as exc:
            if attempt + 1 >= CHECKPOINT_REPLACE_ATTEMPTS:
                raise PermissionError(
                    f"checkpoint 被其他进程占用；最新状态保留在 {temporary}"
                ) from exc
            sleep(CHECKPOINT_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    return target


def load_run_checkpoint(path: Path) -> dict[str, Any]:
    """Load and validate a run-checkpoint envelope."""

    checkpoint_path = Path(path)
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"无法读取运行检查点：{checkpoint_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("运行检查点顶层必须是对象")
    schema_version = payload.get("schema_version")
    if schema_version not in _SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(
            f"不支持的运行检查点版本：{payload.get('schema_version')!r}"
        )
    run_id = payload.get("run_id")
    state = payload.get("state")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("运行检查点缺少有效 run_id")
    if not isinstance(state, dict):
        raise ValueError("运行检查点缺少有效 state")
    if state.get("run_id") != run_id:
        raise ValueError("运行检查点与状态的 run_id 不一致")
    if not isinstance(payload.get("saved_at"), str):
        raise ValueError("运行检查点缺少有效 saved_at")
    if not isinstance(payload.get("is_terminal"), bool):
        raise ValueError("运行检查点缺少有效 is_terminal")
    expected_terminal = state.get("status") in TERMINAL_STATUSES
    if payload["is_terminal"] != expected_terminal:
        raise ValueError("运行检查点终态标记与状态不一致")
    if schema_version == 1:
        payload = _migrate_v1_payload(payload)
    if payload["schema_version"] == 2:
        payload = _migrate_v2_payload(payload)
    if payload["schema_version"] == 3:
        payload = _migrate_v3_payload(payload)
    if payload["schema_version"] == 4:
        payload = _migrate_v4_payload(payload)
    if payload["schema_version"] == 5:
        payload = _migrate_v5_payload(payload)
    if payload["schema_version"] == 6:
        payload = _migrate_v6_payload(payload)
    if payload["schema_version"] == 7:
        payload = _migrate_v7_payload(payload)
    if payload["schema_version"] == 8:
        payload = _migrate_v8_payload(payload)
    if payload["schema_version"] == 9:
        payload = _migrate_v9_payload(payload)
    if payload["schema_version"] == 10:
        payload = _migrate_v10_payload(payload)
    if payload["schema_version"] == 11:
        payload = _migrate_v11_payload(payload)
    if payload["schema_version"] == 12:
        payload = _migrate_v12_payload(payload)
    if payload["schema_version"] == 13:
        payload = _migrate_v13_payload(payload)
    if payload["schema_version"] == 14:
        payload = _migrate_v14_payload(payload)
    if payload["schema_version"] == 15:
        payload = _migrate_v15_payload(payload)
    if payload["schema_version"] == 16:
        payload = _migrate_v16_payload(payload)
    if payload["schema_version"] == 17:
        payload = _migrate_v17_payload(payload)
    if payload["schema_version"] == 18:
        payload = _migrate_v18_payload(payload)
    if payload["schema_version"] == 19:
        payload = _migrate_v19_payload(payload)
    canonical_payload = deepcopy(dict(payload))
    canonical_state = deepcopy(dict(payload["state"]))
    canonical_state.setdefault("psychometric_plateau_status", None)
    canonical_state.setdefault("psychometric_plateau_patience", 2)
    canonical_state.setdefault("psychometric_plateau_min_delta", 0.01)
    canonical_payload["state"] = _strip_removed_state_fields(canonical_state)
    return canonical_payload


def find_latest_resumable_checkpoint(
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any] | None:
    """Return the newest validated nonterminal run checkpoint."""

    root = Path(checkpoint_root)
    if not root.exists():
        return None
    invalid_paths: list[Path] = []
    valid_checkpoint_count = 0
    paths = sorted(
        root.glob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in paths:
        try:
            checkpoint = load_run_checkpoint(path)
        except (ValueError, KeyError, TypeError):
            invalid_paths.append(path)
            continue
        valid_checkpoint_count += 1
        if not checkpoint["is_terminal"]:
            return checkpoint
    if invalid_paths and valid_checkpoint_count == 0:
        names = ", ".join(path.name for path in invalid_paths[:3])
        raise ValueError(
            "没有可读取的运行检查点；损坏或不兼容的文件："
            f"{names}"
        )
    return None


def prepare_resumed_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Reset transient control fields while preserving committed work."""

    resumed = _invalidate_legacy_virtual_screening(state)
    resumed.pop("psychometric_defer_batch_eliminate", None)
    run_id = resumed.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("恢复状态缺少有效 run_id")
    resumed["status"] = "running"
    resumed.setdefault(
        "psychometric_repair_defer_after_rounds",
        PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS,
    )
    # Keep the legacy field readable for old UI/checkpoint consumers, but do
    # not use it as a repair admission/removal budget.
    resumed.setdefault(
        "max_psychometric_repair_rounds",
        PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS,
    )
    for field in _RESUME_RESET_FIELDS:
        resumed[field] = None
    if isinstance(resumed.get("active_psychometric_repair"), Mapping):
        # item_pool is the last committed authority; the router will stage the
        # queued repair again instead of resuming an uncommitted working copy.
        resumed["current_item"] = None
        resumed["current_blueprint_cell"] = None
        resumed["current_item_specification"] = None
        resumed["active_psychometric_repair"] = None
        resumed["current_item_repair_attempted"] = False
        resumed["current_item_revision_count"] = 0
        resumed["current_item_rewrite_count"] = 0
        resumed["current_skeleton_repair_required"] = False
    return resumed


def prepare_retry_state(
    current_state: Mapping[str, Any],
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Resume a run from its newest saved state, falling back to memory."""

    run_id = current_state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("重试状态缺少有效 run_id")
    saved_path = Path(checkpoint_root) / f"{run_id}.json"
    source_state: Mapping[str, Any] = current_state
    if saved_path.exists():
        checkpoint = load_run_checkpoint(saved_path)
        source_state = checkpoint["state"]
    return prepare_resumed_state(source_state)
