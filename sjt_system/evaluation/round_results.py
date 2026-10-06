"""Normalized presentation model for one virtual psychometric round."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from sjt_system.evaluation.psychometrics import (
    CITC_REVISION_THRESHOLD,
    ITEM_DISCRIMINANT_DELTA_MIN_THRESHOLD,
    ITEM_TARGET_HEDGES_G_THRESHOLD,
    ITEM_TARGET_IPIP_SPEARMAN_RHO_THRESHOLD,
)


GATE_ORDER = (
    "citc_pass",
    "target_hedges_g_pass",
    "target_ipip_spearman_rho_pass",
    "discriminant_delta_min_pass",
)

_GATE_LABELS = {
    "citc_pass": "CITC",
    "target_hedges_g_pass": "单题目标Hedges'g",
    "target_ipip_spearman_rho_pass": "单题目标IPIP rho_s",
    "discriminant_delta_min_pass": "单题Δmin",
}

PROFILE_DIAGNOSTIC_ORDER = (
    "target_rho_diagnostic",
    "same_domain_vts_diagnostic",
    "cross_domain_vts_diagnostic",
)

_PROFILE_DIAGNOSTIC_LABELS = {
    "target_rho_diagnostic": "Profile目标rho_s",
    "same_domain_vts_diagnostic": "Profile同域VTS",
    "cross_domain_vts_diagnostic": "Profile跨域VTS",
}

ITEM_ITERATION_METRIC_FIELDS = (
    "citc",
    "target_rho",
    "same_domain_vts",
    "cross_domain_vts",
    "target_hedges_g",
    "target_ipip_spearman_rho",
    "discriminant_delta_min",
)

ITEM_ITERATION_METRIC_STATUS_FIELDS = (
    "iteration_metric_status",
    "measurement_skipped",
    "frozen_from_analysis_round",
)

FACET_FORM_METRIC_FIELDS = (
    "cronbach_alpha",
    "virtual_test_retest_icc",
    "target_hedges_g",
    "target_spearman_rho",
    "discriminant_delta_min",
)


def metric_scalar(value: Any, *preferred_keys: str) -> float | None:
    """Extract a finite numeric metric from scalar or structured evidence."""

    if isinstance(value, Mapping):
        keys = preferred_keys or ("value", "r", "rho")
        for key in keys:
            if key in value:
                return metric_scalar(value.get(key), *preferred_keys)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _gate_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = statistics.get("quality_evaluation") or {}
    qualification = statistics.get("qualification") or {}
    citc = quality.get("facet_citc") or {}
    item_ipip = (
        quality.get("single_item_ipip_metrics")
        or (statistics.get("virtual_screening_metrics") or {}).get("item_ipip_metrics")
        or {}
    )
    hedges = item_ipip.get("target_hedges_g") or {}
    ipip_rho = item_ipip.get("target_ipip_spearman_rho") or {}
    delta_min = item_ipip.get("discriminant_delta_min") or {}
    values = {
        "citc_pass": citc.get("r"),
        "target_hedges_g_pass": hedges.get("standardized_effect"),
        "target_ipip_spearman_rho_pass": ipip_rho.get("rho"),
        "discriminant_delta_min_pass": delta_min.get("delta_min"),
    }
    thresholds = {
        "citc_pass": CITC_REVISION_THRESHOLD,
        "target_hedges_g_pass": ITEM_TARGET_HEDGES_G_THRESHOLD,
        "target_ipip_spearman_rho_pass": ITEM_TARGET_IPIP_SPEARMAN_RHO_THRESHOLD,
        "discriminant_delta_min_pass": ITEM_DISCRIMINANT_DELTA_MIN_THRESHOLD,
    }
    computed_passes = {
        gate_id: metric_scalar(values[gate_id]) is not None
        and metric_scalar(values[gate_id]) >= thresholds[gate_id]
        for gate_id in GATE_ORDER
    }
    return [
        {
            "gate_id": gate_id,
            "label": _GATE_LABELS[gate_id],
            "value": values[gate_id],
            "threshold": thresholds[gate_id],
            "operator": ">=",
            "passes": (
                qualification.get(gate_id)
                if isinstance(qualification.get(gate_id), bool)
                else computed_passes[gate_id]
            ),
            "estimable": metric_scalar(values[gate_id]) is not None,
            "filtering_authority": True,
        }
        for gate_id in GATE_ORDER
    ]


def _profile_diagnostic_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = statistics.get("quality_evaluation") or {}
    raw_diagnostics = quality.get("profile_diagnostics") or {}
    specificity = quality.get("virtual_target_specificity") or {}
    target = specificity.get("rho_target") or specificity.get("target_spearman") or {}
    same_domain = specificity.get("same_domain_non_target") or {}
    cross_domain = specificity.get("cross_domain_non_target") or {}
    fallbacks = {
        "target_rho_diagnostic": target.get("rho"),
        "same_domain_vts_diagnostic": same_domain.get("specificity_margin"),
        "cross_domain_vts_diagnostic": cross_domain.get("specificity_margin"),
    }
    source_aliases = {
        "target_rho_diagnostic": ("target_rho", "target_rho_diagnostic"),
        "same_domain_vts_diagnostic": ("same_domain_vts", "same_domain_vts_diagnostic"),
        "cross_domain_vts_diagnostic": ("cross_domain_vts", "cross_domain_vts_diagnostic"),
    }
    rows = []
    for diagnostic_id in PROFILE_DIAGNOSTIC_ORDER:
        raw_value = None
        for alias in source_aliases[diagnostic_id]:
            if alias in raw_diagnostics:
                raw_value = raw_diagnostics.get(alias)
                break
        value = metric_scalar(raw_value, "value", "rho", "specificity_margin")
        if value is None:
            value = metric_scalar(fallbacks[diagnostic_id], "rho", "specificity_margin")
        rows.append(
            {
                "diagnostic_id": diagnostic_id,
                "label": _PROFILE_DIAGNOSTIC_LABELS[diagnostic_id],
                "value": value,
                "threshold": None,
                "passes": None,
                "estimable": value is not None,
                "filtering_authority": False,
            }
        )
    return rows


def _contaminant(group: Mapping[str, Any]) -> dict[str, Any]:
    facet = group.get("selected_non_target_facet") or group.get("largest_non_target_facet") or group
    selected_rho = group.get("max_non_target_rho")
    if selected_rho is None:
        selected_rho = group.get("largest_non_target_rho")
    return {
        "dimension_id": group.get("largest_non_target_dimension_id") or group.get("dimension_id"),
        "facet_name": group.get("largest_non_target_facet_name") or group.get("facet_name"),
        "facet_name_en": facet.get("facet_name_en"),
        "domain_id": group.get("largest_non_target_domain_id"),
        "domain_name": facet.get("domain_name"),
        "definition": facet.get("definition"),
        "high_behavior": facet.get("high_behavior"),
        "low_behavior": facet.get("low_behavior"),
        "group_id": group.get("selected_non_target_group_id") or facet.get("group_id"),
        "condition_id": group.get("selected_non_target_condition_id") or facet.get("condition_id"),
        "rho": selected_rho,
        "signed_rho": selected_rho,
        "non_target_spearman": group.get("non_target_spearman") or [],
        "vts": group.get("specificity_margin"),
        "threshold": None,
    }


def _condition_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = statistics.get("quality_evaluation") or {}
    rows: list[dict[str, Any]] = []
    for condition_id, metric in (quality.get("per_condition_metrics") or {}).items():
        if not isinstance(metric, Mapping):
            continue
        # Matched-condition analysis stores the structured value under
        # ``citc``; older/other producers may use ``facet_citc``. Prefer the
        # first shape that actually contains a numeric correlation.
        citc_value = metric_scalar(metric.get("facet_citc"), "r", "value")
        if citc_value is None:
            citc_value = metric_scalar(metric.get("citc"), "r", "value")
        rows.append(
            {
                "condition_id": str(condition_id),
                "arm_id": metric.get("arm_id") or ("target" if condition_id == "target" else str(condition_id).split("__", 1)[0]),
                "group_id": metric.get("group_id") or ("target" if condition_id == "target" else str(condition_id).split("__", 1)[-1]),
                "filtering_authority": False,
                "citc": citc_value,
                "rho": metric_scalar(metric.get("rho"), "rho", "value"),
            }
        )
    return rows


def build_item_iteration_metric(
    item_id: str,
    statistics: Mapping[str, Any],
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the compact, round-stable metric snapshot for one item.

    A psychometric analysis replaces ``state.item_statistics`` on every round.
    Keeping the four active gate values and three diagnostic-only profile
    values here lets the iteration history report
    historical item values without duplicating the large response diagnostics.
    Qualified locked items carry their original values with an explicit frozen
    status; the whole-form metrics are recorded separately from this snapshot.
    """

    gates = _gate_rows(statistics)
    values = {str(row["gate_id"]): row.get("value") for row in gates}
    passes = {str(row["gate_id"]): row.get("passes") for row in gates}
    profile_diagnostics = _profile_diagnostic_rows(statistics)
    profile_values = {
        str(row["diagnostic_id"]): row.get("value")
        for row in profile_diagnostics
    }
    quality = statistics.get("quality_evaluation") or {}
    virtual_metrics = statistics.get("virtual_screening_metrics") or {}
    raw_item = item or {}
    return {
        "item_id": str(item_id),
        "item_version": raw_item.get("version"),
        "facet_id": raw_item.get("target_dimension_id")
        or raw_item.get("dimension_id")
        or statistics.get("dimension_id"),
        "citc": values.get("citc_pass"),
        "target_rho": profile_values.get("target_rho_diagnostic"),
        "same_domain_vts": profile_values.get("same_domain_vts_diagnostic"),
        "cross_domain_vts": profile_values.get("cross_domain_vts_diagnostic"),
        "target_hedges_g": values.get("target_hedges_g_pass"),
        "target_ipip_spearman_rho": values.get("target_ipip_spearman_rho_pass"),
        "discriminant_delta_min": values.get("discriminant_delta_min_pass"),
        "citc_pass": passes.get("citc_pass"),
        "target_rho_pass": None,
        "same_domain_vts_pass": None,
        "cross_domain_vts_pass": None,
        "target_hedges_g_pass": passes.get("target_hedges_g_pass"),
        "target_ipip_spearman_rho_pass": passes.get("target_ipip_spearman_rho_pass"),
        "discriminant_delta_min_pass": passes.get("discriminant_delta_min_pass"),
        "qualified": all(row.get("passes") is True for row in gates),
        "profile_diagnostics": profile_diagnostics,
        "recommendation": quality.get("recommendation"),
        "iteration_metric_freeze_policy": statistics.get(
            "iteration_metric_freeze_policy"
        ) or virtual_metrics.get("iteration_metric_freeze_policy"),
        "iteration_metric_status": statistics.get(
            "iteration_metric_status"
        ) or virtual_metrics.get("iteration_metric_status"),
        "measurement_skipped": bool(statistics.get("measurement_skipped"))
        or bool(virtual_metrics.get("measurement_skipped")),
        "frozen_from_analysis_round": statistics.get(
            "frozen_from_analysis_round"
        ) or virtual_metrics.get("frozen_from_analysis_round"),
    }


def build_facet_iteration_metric(
    facet: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the five stable whole-form metrics for one facet."""

    facet_id = facet.get("sjt_facet_id") or facet.get("facet_id")
    return {
        "facet_id": str(facet_id) if facet_id is not None else None,
        "item_count": facet.get("item_count"),
        "selected_item_ids": list(facet.get("selected_item_ids") or []),
        **{
            field: facet.get(field)
            for field in FACET_FORM_METRIC_FIELDS
        },
        "status": facet.get("status"),
    }


def _iteration_controller_metadata(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    controller = record.get("facet_iteration_state")
    controller = controller if isinstance(controller, Mapping) else {}
    analysis_round = record.get("analysis_round")
    if analysis_round is None:
        analysis_round = record.get("psychometric_analysis_round")
    completed_history = controller.get("completed_history") or []
    completed_record = next(
        (
            row
            for row in reversed(completed_history)
            if isinstance(row, Mapping)
            and str(row.get("analysis_round")) == str(analysis_round)
        ),
        None,
    )
    development_round = record.get("development_round")
    if development_round is None and isinstance(completed_record, Mapping):
        development_round = completed_record.get("development_round")
    if development_round is None:
        development_round = controller.get("development_round")
    measurement_batch = record.get("measurement_batch")
    if measurement_batch is None:
        measurement_batch = controller.get("measurement_batch")
    if measurement_batch is None:
        measurement_batch = analysis_round
    round_completed = record.get("round_completed")
    if round_completed is None:
        round_completed = isinstance(completed_record, Mapping)
    return {
        "iteration_policy_version": record.get("iteration_policy_version")
        or controller.get("policy_version"),
        "development_round": development_round,
        "development_round_batch": record.get("development_round_batch")
        or controller.get("development_round_batch"),
        "measurement_batch": measurement_batch,
        "round_completed": round_completed,
        "facet_iteration_state": deepcopy(dict(controller)),
    }


def build_iteration_metrics_snapshot(
    record: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a compact, JSON-safe snapshot for one completed iteration.

    The snapshot intentionally contains only four active item gates, three
    diagnostic-only profile metrics, and five
    facet-level form metrics. Full response diagnostics remain in the official
    psychometrics artifacts; this file is the durable iteration ledger.
    """

    raw_items = record.get("candidate_item_metrics") or record.get("item_metrics") or {}
    selected_ids = {
        str(item_id) for item_id in record.get("form_item_ids") or []
    }
    dispositions = record.get("item_dispositions") or {}

    def _annotate_item_metric(item_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        measurement_qualified = row.get("qualified")
        disposition = dispositions.get(item_id) if isinstance(dispositions, Mapping) else None
        disposition = disposition if isinstance(disposition, Mapping) else {}
        retention_basis = disposition.get("retention_basis")
        iteration_metric_status = row.get("iteration_metric_status")
        facet_form_retained = (
            disposition.get("status") == "facet_form_retained"
            or retention_basis == "three_successful_facet_rounds"
        )
        locked_for_current_version = (
            disposition.get("status") == "qualified_locked"
            and row.get("item_version") is not None
            and disposition.get("item_version") == row.get("item_version")
            and retention_basis == "four_iteration_gates_passed"
        )
        annotated = {
            **deepcopy(dict(row)),
            # Keep the raw gate result separate from the display-level
            # ``qualified`` value, which may be suppressed after a later
            # retention disposition is committed.
            "measurement_qualified": measurement_qualified,
            "qualified": False if facet_form_retained else row.get("qualified"),
            "selected_for_form": item_id in selected_ids,
            # ``qualified`` is the committed value carried by this row. A
            # gate-qualified item is permanently accepted separately via the
            # historical lock marker; its four active gates and three profile
            # diagnostics are frozen
            # on later rounds, while the form metrics remain current. The
            # current-round count below therefore excludes frozen rows.
            "frozen_qualified": (
                not facet_form_retained
                and locked_for_current_version
                and iteration_metric_status == "frozen"
            ),
            "retention_basis": retention_basis,
        }
        return annotated

    if isinstance(raw_items, Mapping):
        item_metrics = {
            str(item_id): _annotate_item_metric(str(item_id), row)
            for item_id, row in raw_items.items()
            if isinstance(row, Mapping)
        }
    elif isinstance(raw_items, list):
        item_metrics = {
            str(row.get("item_id")): _annotate_item_metric(str(row.get("item_id")), row)
            for row in raw_items
            if isinstance(row, Mapping) and row.get("item_id") is not None
        }
    else:
        item_metrics = {}

    raw_facets = record.get("facet_metrics")
    if not isinstance(raw_facets, Mapping) or not raw_facets:
        form_metrics = record.get("form_metrics") or {}
        raw_facets = {
            str(row.get("sjt_facet_id")): row
            for row in form_metrics.get("facet_metrics") or []
            if isinstance(row, Mapping) and row.get("sjt_facet_id") is not None
        }
    facet_metrics = {
        str(facet_id): build_facet_iteration_metric(row)
        for facet_id, row in raw_facets.items()
        if isinstance(row, Mapping)
    }
    snapshot = {
        "schema_version": 1,
        "run_id": run_id,
        "analysis_round": record.get("analysis_round"),
        "recorded_at": record.get("recorded_at"),
        "item_metric_fields": list(ITEM_ITERATION_METRIC_FIELDS),
        "item_gate_count": len(GATE_ORDER),
        "item_metric_status_fields": list(ITEM_ITERATION_METRIC_STATUS_FIELDS),
        "facet_metric_fields": list(FACET_FORM_METRIC_FIELDS),
        "item_metrics": item_metrics,
        "candidate_count": record.get("candidate_count") or len(item_metrics),
        "selected_item_ids": list(record.get("form_item_ids") or []),
        "facet_metrics": facet_metrics,
        "item_count": record.get("item_count"),
        "form_status": record.get("form_status"),
        "current_round_qualified_count": sum(
            row.get("qualified") is True
            and row.get("frozen_qualified") is not True
            and row.get("iteration_metric_status") != "frozen"
            for row in item_metrics.values()
        ),
        "frozen_qualified_count": sum(
            row.get("frozen_qualified") is True for row in item_metrics.values()
        ),
        "frozen_metric_count": sum(
            row.get("iteration_metric_status") == "frozen"
            for row in item_metrics.values()
        ),
        "measured_metric_count": sum(
            row.get("iteration_metric_status") == "measured"
            for row in item_metrics.values()
        ),
        "evidence_scope": "exploratory_virtual_screening_evidence",
        **_iteration_controller_metadata(record),
    }
    if record.get("virtual_content_review_protocol") == "mte_cosmin_virtual_content_review_v1":
        snapshot["schema_version"] = 2
        for field in (
            "virtual_content_review_protocol", "round_label", "workflow_stage",
            "item_snapshots", "item_statistics_snapshot", "item_content_evidence",
            "item_lineage", "item_dispositions", "form_metrics", "response_data_ref",
            "item_bank_id", "item_bank_version", "virtual_sample_config",
        ):
            snapshot[field] = deepcopy(record.get(field))
        snapshot["evidence_scope"] = "exploratory_virtual_development_evidence"
    return snapshot


def _option_facet_mean_rows(
    item_id: str,
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in diagnostics.get("by_condition") or []:
        if not isinstance(condition, Mapping):
            continue
        for option in condition.get("options") or []:
            if not isinstance(option, Mapping):
                continue
            rows.append(
                {
                    "item_id": item_id,
                    "condition_id": condition.get("condition_id"),
                    "construct_role": condition.get("construct_role"),
                    "facet_id": condition.get("facet_id"),
                    "facet_name": condition.get("facet_name"),
                    "option_id": option.get("option_id"),
                    "option_score": option.get("score"),
                    "n": option.get("selection_count"),
                    "facet_mean": option.get("facet_mean"),
                    "facet_standard_error": option.get("facet_standard_error"),
                    "filtering_authority": False,
                }
            )
    return rows


def _option_score_comparison_rows(
    item_id: str,
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return option-aligned target and selected non-target facet means."""

    return [
        {
            "item_id": item_id,
            "option_id": row.get("option_id"),
            "option_score": row.get("score"),
            "target_n": row.get("target_n"),
            "target_mean_score": row.get("target_mean_score"),
            "target_standard_error": row.get("target_standard_error"),
            "same_domain_n": row.get("same_domain_n"),
            "same_domain_mean_score": row.get("same_domain_mean_score"),
            "same_domain_standard_error": row.get("same_domain_standard_error"),
            "same_domain_dimension_id": row.get("same_domain_dimension_id"),
            "same_domain_facet_name": row.get("same_domain_facet_name"),
            "same_domain_group_id": row.get("same_domain_group_id"),
            "cross_domain_n": row.get("cross_domain_n"),
            "cross_domain_mean_score": row.get("cross_domain_mean_score"),
            "cross_domain_standard_error": row.get("cross_domain_standard_error"),
            "cross_domain_dimension_id": row.get("cross_domain_dimension_id"),
            "cross_domain_facet_name": row.get("cross_domain_facet_name"),
            "cross_domain_group_id": row.get("cross_domain_group_id"),
            "same_domain_groups": row.get("same_domain_groups") or [],
            "cross_domain_groups": row.get("cross_domain_groups") or [],
            "mean_source": row.get("mean_source"),
            "filtering_authority": False,
        }
        for row in diagnostics.get("option_score_comparisons") or []
        if isinstance(row, Mapping)
    ]


def build_psychometric_round_result(state: Mapping[str, Any]) -> dict[str, Any]:
    """Create one canonical result used by all user-facing surfaces."""

    statistics_by_item = state.get("item_statistics") or {}
    locked_versions = state.get("locked_retained_item_versions") or {}
    dispositions = state.get("item_final_dispositions") or {}
    item_by_id: dict[str, Mapping[str, Any]] = {}
    for collection in (state.get("item_pool") or [], state.get("frozen_item_bank") or []):
        for item in collection:
            if isinstance(item, Mapping) and item.get("item_id") is not None:
                item_by_id[str(item.get("item_id"))] = item

    items: list[dict[str, Any]] = []
    gate_summary = {
        gate_id: {
            "gate_id": gate_id,
            "label": _GATE_LABELS[gate_id],
            "threshold": threshold,
            "operator": ">=",
            "pass_count": 0,
            "item_count": 0,
            "unestimable_count": 0,
        }
        for gate_id, threshold in (
            ("citc_pass", CITC_REVISION_THRESHOLD),
            ("target_hedges_g_pass", ITEM_TARGET_HEDGES_G_THRESHOLD),
            ("target_ipip_spearman_rho_pass", ITEM_TARGET_IPIP_SPEARMAN_RHO_THRESHOLD),
            ("discriminant_delta_min_pass", ITEM_DISCRIMINANT_DELTA_MIN_THRESHOLD),
        )
    }
    unestimable_option_panels = 0
    all_option_facet_means: list[dict[str, Any]] = []
    all_option_score_comparisons: list[dict[str, Any]] = []
    for item_id, raw_statistics in statistics_by_item.items():
        if not isinstance(raw_statistics, Mapping):
            continue
        item_id = str(item_id)
        item = item_by_id.get(item_id) or {}
        item_version = item.get("version")
        gates = _gate_rows(raw_statistics)
        failed_thresholds = [
            gate["gate_id"] for gate in gates if gate.get("passes") is not True
        ]
        for gate in gates:
            summary = gate_summary[gate["gate_id"]]
            summary["item_count"] += 1
            if gate.get("passes") is True:
                summary["pass_count"] += 1
            if gate.get("estimable") is not True:
                summary["unestimable_count"] += 1

        disposition = dispositions.get(item_id) or {}
        qualified = (
            all(gate.get("passes") is True for gate in gates)
            and disposition.get("status") != "facet_form_retained"
            and disposition.get("retention_basis") != "three_successful_facet_rounds"
        )
        locked = (
            item_version is not None
            and item_id in locked_versions
            and locked_versions[item_id] == item_version
        )
        retention_basis = disposition.get("retention_basis")
        virtual_metrics = raw_statistics.get("virtual_screening_metrics") or {}
        iteration_metric_status = raw_statistics.get(
            "iteration_metric_status"
        ) or virtual_metrics.get("iteration_metric_status")
        frozen_qualified = (
            disposition.get("status") != "facet_form_retained"
            and retention_basis != "three_successful_facet_rounds"
            and locked
            and (
                iteration_metric_status == "frozen"
                or (
                    disposition.get("status") == "qualified_locked"
                    and retention_basis == "four_iteration_gates_passed"
                )
            )
        )
        if disposition.get("status") == "facet_form_retained":
            status = "facet_form_retained"
        elif disposition.get("status") == "pending_sme_review":
            status = "pending_sme_review"
        elif disposition.get("status") == "eliminated":
            status = "eliminated"
        elif locked and failed_thresholds:
            status = "qualified_locked_warning"
        elif locked:
            status = "qualified_locked"
        elif qualified:
            status = "newly_qualified"
        else:
            status = "pending_treatment"

        quality = raw_statistics.get("quality_evaluation") or {}
        specificity = quality.get("virtual_target_specificity") or {}
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        profile_diagnostics = _profile_diagnostic_rows(raw_statistics)
        option_diagnostics = raw_statistics.get("option_choice_diagnostics") or {}
        option_facet_means = _option_facet_mean_rows(
            item_id,
            option_diagnostics,
        )
        all_option_facet_means.extend(option_facet_means)
        option_score_comparisons = _option_score_comparison_rows(
            item_id,
            option_diagnostics,
        )
        all_option_score_comparisons.extend(option_score_comparisons)
        gradient = raw_statistics.get("target_option_gradient") or {}
        unestimable_option_panels += int(gradient.get("estimable") is not True)
        items.append(
            {
                "item_id": item_id,
                "item_version": item_version,
                "status": status,
                "qualified": qualified,
                "frozen_qualified": frozen_qualified,
                "retention_basis": retention_basis,
                "monitoring_pass": disposition.get("monitoring_pass"),
                "iteration_metric_freeze_policy": raw_statistics.get(
                    "iteration_metric_freeze_policy"
                ) or virtual_metrics.get(
                    "iteration_metric_freeze_policy"
                ),
                "iteration_metric_status": iteration_metric_status,
                "measurement_skipped": bool(
                    raw_statistics.get("measurement_skipped")
                ) or bool(
                    virtual_metrics.get("measurement_skipped")
                ),
                "frozen_from_analysis_round": raw_statistics.get(
                    "frozen_from_analysis_round"
                ) or virtual_metrics.get(
                    "frozen_from_analysis_round"
                ),
                "failed_thresholds": failed_thresholds,
                "gates": gates,
                "profile_diagnostics": profile_diagnostics,
                "max_contaminants": {
                    "same_domain": _contaminant(same_domain),
                    "cross_domain": _contaminant(cross_domain),
                },
                "per_condition_metrics": _condition_rows(raw_statistics),
                "target_option_gradient": deepcopy(raw_statistics.get("target_option_gradient") or {}),
                "option_choice_diagnostics": deepcopy(dict(option_diagnostics)),
                "option_facet_means": option_facet_means,
                "option_score_comparisons": option_score_comparisons,
                "arm_difference_diagnostics": deepcopy(
                    dict(
                        option_diagnostics.get("arm_difference_diagnostics")
                        or {}
                    )
                ),
                "item": deepcopy(dict(item)),
                "diagnostic_flags": deepcopy(quality.get("diagnostic_flags") or []),
            }
        )

    items.sort(key=lambda row: row["item_id"])
    pending = [row for row in items if row["status"] == "pending_treatment"]
    newly_qualified = [row for row in items if row["status"] == "newly_qualified"]
    locked = [
        row
        for row in items
        if row["status"] in {"qualified_locked", "qualified_locked_warning"}
    ]
    monitoring = [
        row for row in items if row["status"] == "qualified_locked_warning"
    ]
    facet_form_retained = [
        row for row in items if row["status"] == "facet_form_retained"
    ]
    unestimable_metric_count = sum(
        row["unestimable_count"] for row in gate_summary.values()
    )
    return {
        "schema_version": 2,
        "analysis_round": int(state.get("psychometric_analysis_round") or 0),
        **_iteration_controller_metadata(state),
        "conditioning": {
            "variable": "condition_id",
            "method": "matched_facet_arms",
            "rank_method": None,
        },
        "summary": {
            "item_count": len(items),
            "newly_qualified_count": len(newly_qualified),
            "pending_treatment_count": len(pending),
            "qualified_locked_count": len(locked),
            "facet_form_retained_count": len(facet_form_retained),
            "frozen_qualified_count": sum(
                row["frozen_qualified"] is True for row in items
            ),
            "current_round_qualified_count": sum(
                row["qualified"] is True
                and row["frozen_qualified"] is not True
                and row.get("iteration_metric_status") != "frozen"
                for row in items
            ),
            "frozen_metric_count": sum(
                row.get("iteration_metric_status") == "frozen"
                for row in items
            ),
            "measured_metric_count": sum(
                row.get("iteration_metric_status") == "measured"
                for row in items
            ),
            "monitoring_warning_count": len(monitoring),
            "unestimable_metric_count": unestimable_metric_count,
            "unestimable_option_panel_count": unestimable_option_panels,
        },
        "gate_summary": [gate_summary[gate_id] for gate_id in GATE_ORDER],
        "profile_diagnostic_order": list(PROFILE_DIAGNOSTIC_ORDER),
        "items": items,
        "pending_items": pending,
        "newly_qualified_items": newly_qualified,
        "locked_items": locked,
        "facet_form_retained_items": facet_form_retained,
        "monitoring_warnings": monitoring,
        "option_facet_means": all_option_facet_means,
        "option_score_comparisons": all_option_score_comparisons,
        "condition_score_diagnostics": deepcopy(
            (state.get("virtual_sample_config") or {}).get(
                "generation_diagnostics"
            )
            or {}
        ),
        "evidence_scope": "exploratory_virtual_screening_evidence",
    }
