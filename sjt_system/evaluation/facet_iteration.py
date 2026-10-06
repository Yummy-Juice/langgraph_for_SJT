"""Fixed-cohort development rounds, distinct from measurement batches."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

POLICY_VERSION = "fixed_cohort_facet_iteration_v2"
# A qualified item may now be thawed for a failed facet form when it is in the
# bottom quartile of any active item-validity metric.  Keep this as an explicit
# policy marker so new and resumed runs expose the rule in their audit trail.
QUALIFIED_ITEM_THAW_POLICY_VERSION = "qualified_item_bottom_quartile_any_validity_metric_v1"
QUALIFIED_ITEM_THAW_QUANTILE = 0.25
QUALIFIED_ITEM_THAW_METRICS = (
    "target_hedges_g",
    "target_ipip_spearman_rho",
    "discriminant_delta_min",
)
MAX_COMPLETED_ROUNDS = 3
# Local facet repairs are intentionally unbounded within each of the two
# post-baseline development rounds.
LOCAL_REPAIR_BUDGET_POLICY = "unlimited"
MAX_LOCAL_ATTEMPTS: int | None = None
LEGACY_LOCAL_REPAIR_EXHAUSTION_REASON = "Five local repairs exhausted without all facet gates passing"
ITERATION_GATES = (
    "citc_pass",
    "target_hedges_g_pass",
    "target_ipip_spearman_rho_pass",
    "discriminant_delta_min_pass",
)


def initial_iteration_state() -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION, "status": "awaiting_baseline",
        "qualified_item_thaw_policy_version": QUALIFIED_ITEM_THAW_POLICY_VERSION,
        "qualified_item_thaw_quantile": QUALIFIED_ITEM_THAW_QUANTILE,
        "qualified_item_thaw_metrics": list(QUALIFIED_ITEM_THAW_METRICS),
        "development_round": 1, "completed_rounds": 0, "measurement_batch": 0,
        "development_round_batch": 0,
        "local_repair_budget_policy": LOCAL_REPAIR_BUDGET_POLICY,
        "max_local_attempts": None,
        "local_attempts": {}, "accepted_facets": {}, "baseline_facets": {},
        "failed_facet_ids": [], "pending_facet_ids": [], "comparisons": [],
        "completed_history": [], "cohort_source_ref": None,
        "first_round_reference_ref": None, "thawed_item_versions": {},
        "thaw_history": [], "paused_reason": None,
    }


def local_repair_capacity_available(
    iteration: Mapping[str, Any], facet_id: str,
) -> bool:
    """Return whether a failed facet may receive another local repair.

    ``None`` is the serialized unlimited value.  The default is also
    unlimited so older checkpoints cannot silently restore the former cap.
    """

    policy = str(
        iteration.get("local_repair_budget_policy")
        or LOCAL_REPAIR_BUDGET_POLICY
    )
    limit = iteration.get("max_local_attempts")
    if policy == LOCAL_REPAIR_BUDGET_POLICY or limit is None:
        return True
    return int(iteration.get("local_attempts", {}).get(facet_id, 0)) < int(limit)


def is_enabled(state: Mapping[str, Any]) -> bool:
    value = state.get("facet_iteration_state")
    return isinstance(value, Mapping) and value.get("policy_version") == POLICY_VERSION


def _item_gates_pass(statistics: Mapping[str, Any]) -> bool:
    qualification = statistics.get("qualification") or {}
    return isinstance(qualification, Mapping) and all(
        qualification.get(gate) is True for gate in ITERATION_GATES
    )


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _item_validity_metrics(statistics: Mapping[str, Any]) -> dict[str, float | None]:
    quality = statistics.get("quality_evaluation") or {}
    item_ipip = (
        quality.get("single_item_ipip_metrics")
        or (statistics.get("virtual_screening_metrics") or {}).get("item_ipip_metrics")
        or {}
    )
    if not isinstance(item_ipip, Mapping):
        item_ipip = {}
    hedges = item_ipip.get("target_hedges_g") or {}
    rho = item_ipip.get("target_ipip_spearman_rho") or {}
    delta = item_ipip.get("discriminant_delta_min") or {}
    return {
        "target_hedges_g": _finite_metric(
            hedges.get("standardized_effect") if isinstance(hedges, Mapping) else None
        ),
        "target_ipip_spearman_rho": _finite_metric(
            rho.get("rho") if isinstance(rho, Mapping) else None
        ),
        "discriminant_delta_min": _finite_metric(
            delta.get("delta_min") if isinstance(delta, Mapping) else None
        ),
    }


def _linear_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a thaw quantile without finite metric values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def qualified_item_thaw_selection(
    members: list[Mapping[str, Any]],
    statistics: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Select qualified facet items below the 25th percentile on any metric.

    The three metrics are higher-is-better.  The cutoff is strict, so an item
    exactly on the percentile is not thawed; ties are retained unless their
    value is strictly below the cutoff.
    """

    rows = []
    for item in members:
        item_id = str(item.get("item_id") or "")
        if not item_id:
            raise ValueError("Qualified facet thaw requires item IDs")
        row = _item_validity_metrics(statistics.get(item_id) or {})
        missing = [metric for metric, value in row.items() if value is None]
        if missing:
            raise ValueError(
                f"Qualified facet thaw requires finite item metrics: {item_id} {missing}"
            )
        rows.append((item_id, row))
    if not rows:
        raise ValueError("Qualified facet thaw requires at least one facet item")

    selected: set[str] = set()
    thresholds: dict[str, float] = {}
    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric in QUALIFIED_ITEM_THAW_METRICS:
        values = [float(row[metric]) for _, row in rows]
        threshold = _linear_quantile(values, QUALIFIED_ITEM_THAW_QUANTILE)
        thresholds[metric] = threshold
        ordered = sorted(
            ((item_id, float(row[metric])) for item_id, row in rows),
            key=lambda pair: (pair[1], pair[0]),
        )
        rankings[metric] = [
            {"item_id": item_id, "value": value, "rank": index + 1,
             "below_cutoff": value < threshold}
            for index, (item_id, value) in enumerate(ordered)
        ]
        selected.update(item_id for item_id, value in ordered if value < threshold)

    return sorted(selected), {
        "policy_version": QUALIFIED_ITEM_THAW_POLICY_VERSION,
        "quantile": QUALIFIED_ITEM_THAW_QUANTILE,
        "direction": "lower_values_first",
        "metrics": list(QUALIFIED_ITEM_THAW_METRICS),
        "thresholds": thresholds,
        "selected_item_ids": sorted(selected),
        "rankings": rankings,
    }


def facet_repair_candidates(
    members: list[Mapping[str, Any]],
    statistics: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any] | None]:
    """Return ordinary failing items or the qualified-item bottom-quartile set."""

    failing = [
        item for item in members
        if not _item_gates_pass(statistics.get(str(item.get("item_id"))) or {})
    ]
    if failing:
        return failing, None
    selected_ids, selection = qualified_item_thaw_selection(members, statistics)
    selected = [item for item in members if str(item.get("item_id")) in set(selected_ids)]
    return selected, selection


def refresh_repair_queue_for_thaw_policy(state: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the qualified-item thaw rule to a queued/resumed repair batch.

    A stopped checkpoint may have been queued before the policy was relaxed.
    Re-filter that uncommitted queue before resuming so completed-but-uncommitted
    drafts from the old all-items thaw are never silently published.
    """

    resumed = deepcopy(dict(state))
    if not is_enabled(resumed):
        return resumed
    controller = resumed.get("facet_iteration_state") or {}
    if controller.get("status") != "awaiting_repair":
        return resumed
    pending = [
        entry for entry in resumed.get("items_to_revise") or []
        if isinstance(entry, Mapping)
    ]
    if not pending:
        return resumed
    items = {
        str(item.get("item_id")): item
        for item in (resumed.get("item_pool") or resumed.get("frozen_item_bank") or [])
        if isinstance(item, Mapping) and item.get("item_id")
    }
    statistics = resumed.get("item_statistics") or {}
    allowed_by_facet: dict[str, set[str]] = {}
    selection_by_facet: dict[str, dict[str, Any]] = {}
    for facet_id in controller.get("failed_facet_ids") or []:
        members = [
            item for item in items.values()
            if str(item.get("target_dimension_id")) == str(facet_id)
        ]
        candidates, selection = facet_repair_candidates(members, statistics)
        allowed_by_facet[str(facet_id)] = {
            str(item.get("item_id")) for item in candidates
        }
        if selection is not None:
            selection_by_facet[str(facet_id)] = selection

    queue: list[dict[str, Any]] = []
    for raw_entry in pending:
        entry = deepcopy(dict(raw_entry))
        item_id = str(entry.get("item_id") or "")
        facet_id = str(entry.get("facet_id") or (
            items.get(item_id) or {}
        ).get("target_dimension_id") or "")
        allowed = allowed_by_facet.get(facet_id)
        if entry.get("facet_level_failure") is True and allowed is not None:
            if item_id not in allowed:
                continue
            selection = selection_by_facet.get(facet_id)
            if selection is not None:
                # Existing archives bind the full source evidence immutably.
                # Keep the new thaw authorization beside that evidence on
                # resume; newly planned entries receive it in their evidence
                # before an archive is created.
                entry["qualified_item_thaw"] = True
                entry["thaw_selection"] = deepcopy(selection)
        queue.append(entry)

    policy = {
        "policy_version": QUALIFIED_ITEM_THAW_POLICY_VERSION,
        "quantile": QUALIFIED_ITEM_THAW_QUANTILE,
        "metrics": list(QUALIFIED_ITEM_THAW_METRICS),
        "facets": deepcopy(selection_by_facet),
    }
    iteration = deepcopy(controller)
    iteration.setdefault("qualified_item_thaw_policy_version", QUALIFIED_ITEM_THAW_POLICY_VERSION)
    iteration.setdefault("qualified_item_thaw_quantile", QUALIFIED_ITEM_THAW_QUANTILE)
    iteration.setdefault("qualified_item_thaw_metrics", list(QUALIFIED_ITEM_THAW_METRICS))
    history = iteration.setdefault("qualified_item_thaw_history", [])
    marker = {
        "development_round": iteration.get("development_round"),
        "measurement_batch": iteration.get("measurement_batch"),
        "selection": policy,
    }
    if marker not in history:
        history.append(marker)
    resumed["facet_iteration_state"] = iteration
    if len(queue) != len(pending) or selection_by_facet:
        resumed["items_to_revise"] = queue
        resumed["scenario_repair_staged"] = {}
        progress = resumed.get("scenario_repair_progress") or {}
        resumed["scenario_repair_progress"] = {
            item_id: progress[item_id]
            for item_id in (str(entry.get("item_id")) for entry in queue)
            if item_id in progress
        }
        resumed["scenario_repair_pause"] = None
    return resumed


def ensure_measurement_allowed(state: Mapping[str, Any]) -> None:
    if not is_enabled(state):
        return
    iteration = state["facet_iteration_state"]
    if iteration.get("status") in {"paused", "complete"}:
        raise ValueError(iteration.get("paused_reason") or "All three development rounds are complete")
    expected_status = "awaiting_measurement" if iteration.get("completed_rounds") else "awaiting_baseline"
    if iteration.get("status") != expected_status:
        raise ValueError("A measurement is not scheduled for this development round")
    if iteration.get("completed_rounds") and not iteration.get("pending_facet_ids"):
        raise ValueError("A later measurement requires a committed repair or replacement")


def _snapshot(row: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    ids = [str(value) for value in row.get("selected_item_ids") or []]
    items = record.get("item_snapshots") or {}
    if not ids or any(item_id not in items for item_id in ids):
        raise ValueError("Facet form is missing its measured item snapshots")
    return {
        "metrics": deepcopy(dict(row)), "selected_item_ids": ids,
        "item_versions": {item_id: items[item_id]["version"] for item_id in ids},
        "item_snapshots": {item_id: deepcopy(items[item_id]) for item_id in ids},
        "analysis_round": record["analysis_round"],
        "response_data_ref": record.get("response_data_ref"),
    }


def record_measurement(state: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Advance only after a complete, immutable measurement has been committed."""
    from sjt_system.evaluation.form_metrics import facet_form_gate_comparison, form_quality_summary

    iteration = deepcopy(state["facet_iteration_state"])
    batch = int(record["analysis_round"])
    if batch <= int(iteration.get("measurement_batch") or 0):
        return iteration
    ensure_measurement_allowed(state)
    metrics = record.get("form_metrics") or {}
    quality = form_quality_summary(metrics)
    rows = {str(row["sjt_facet_id"]): row for row in metrics.get("facet_metrics") or []}
    facet_ids = [str(value) for value in metrics.get("selected_facet_ids") or []]
    iteration["measurement_batch"] = batch
    round_batch = int(
        record.get("development_round_batch")
        or int(iteration.get("development_round_batch") or 0) + 1
    )
    iteration["development_round_batch"] = round_batch
    pending = set(iteration.get("pending_facet_ids") or [])
    for facet_id in pending:
        if (facet_id not in iteration["baseline_facets"]
                or facet_id in iteration["accepted_facets"]
                or not local_repair_capacity_available(iteration, facet_id)):
            raise ValueError("Committed local repair does not have remaining facet authority")
        iteration["local_attempts"][facet_id] += 1
    if (record.get("form_status") != "complete" or quality.get("status") != "complete"
            or not facet_ids or len(facet_ids) != len(set(facet_ids)) or set(facet_ids) != set(rows)):
        iteration.update(status="paused", paused_reason="Incomplete facet form or required metrics")
        return iteration
    snapshots = {facet_id: _snapshot(rows[facet_id], record) for facet_id in facet_ids}
    if not iteration["completed_rounds"]:
        iteration.update(
            baseline_facets=snapshots, completed_rounds=1, development_round=2,
            status="awaiting_repair", failed_facet_ids=facet_ids,
            development_round_batch=0,
            local_attempts={facet_id: 0 for facet_id in facet_ids},
            cohort_source_ref=record.get("response_data_ref"),
            first_round_reference_ref=state.get("frozen_reference_questionnaire_ref"),
        )
        iteration["completed_history"].append({
            "development_round": 1, "baseline_only": True,
            "analysis_round": batch, "development_round_batch": round_batch,
            "facet_forms": deepcopy(snapshots),
        })
        return iteration
    baseline = iteration["baseline_facets"]
    if set(baseline) != set(rows):
        iteration.update(status="paused", paused_reason="Selected facets changed after the initial baseline")
        return iteration
    for change in iteration["thawed_item_versions"].values():
        if (change.get("development_round") != iteration["development_round"]
                or change.get("source_measurement_batch") != int(state["facet_iteration_state"]["measurement_batch"])):
            continue
        item = (record.get("item_snapshots") or {}).get(change["new_item_id"]) or {}
        if item.get("version") != change["new_version"]:
            iteration.update(status="paused", paused_reason="Committed new item version was not measured")
            return iteration
    comparisons = facet_form_gate_comparison(
        metrics, {"facet_metrics": [baseline[facet_id]["metrics"] for facet_id in facet_ids]},
    )
    iteration["comparisons"] = comparisons
    accepted = iteration["accepted_facets"]
    for comparison in comparisons:
        facet_id = comparison["sjt_facet_id"]
        if facet_id in accepted:
            if snapshots[facet_id]["item_snapshots"] != accepted[facet_id]["item_snapshots"]:
                iteration.update(status="paused", paused_reason="An accepted facet was modified within the round")
                return iteration
            if not comparison["passed"]:
                iteration.update(status="paused", paused_reason="An accepted facet failed combined-form verification")
                return iteration
        elif comparison["passed"]:
            accepted[facet_id] = snapshots[facet_id]
    failed = [facet_id for facet_id in facet_ids if facet_id not in accepted]
    iteration.update(failed_facet_ids=failed, pending_facet_ids=[])
    if not failed and len(comparisons) == len(facet_ids):
        completed = int(iteration["completed_rounds"]) + 1
        iteration["completed_history"].append({
            "development_round": completed, "baseline_only": False,
            "analysis_round": batch, "development_round_batch": round_batch,
            "facet_forms": deepcopy(snapshots),
            "local_attempts": deepcopy(iteration["local_attempts"]),
            "comparisons": deepcopy(comparisons),
        })
        iteration.update(
            completed_rounds=completed, development_round=min(completed + 1, MAX_COMPLETED_ROUNDS),
            status="complete" if completed == MAX_COMPLETED_ROUNDS else "awaiting_repair",
            baseline_facets=snapshots, accepted_facets={},
            development_round_batch=0,
            failed_facet_ids=[] if completed == MAX_COMPLETED_ROUNDS else facet_ids,
            local_attempts={facet_id: 0 for facet_id in facet_ids},
        )
    else:
        iteration["status"] = "awaiting_repair"
    return iteration


def mark_committed_repairs(state: Mapping[str, Any], changes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    iteration = deepcopy(state["facet_iteration_state"])
    if iteration.get("status") != "awaiting_repair":
        raise ValueError("Local edits require an unfinished development round")
    if not changes:
        raise ValueError("At least one committed new item version is required")
    pending = set(iteration.get("pending_facet_ids") or [])
    for item_id, change in changes.items():
        facet_id = str(change["facet_id"])
        if (facet_id not in iteration.get("failed_facet_ids", [])
                or facet_id in iteration["accepted_facets"]
                or not local_repair_capacity_available(iteration, facet_id)):
            raise ValueError("Repair cannot alter an accepted or exhausted facet")
        if change.get("old_version") == change.get("new_version") and item_id == change.get("new_item_id"):
            raise ValueError("A measurement cannot be scheduled without a new item version")
        pending.add(facet_id)
        thaw = {
            **deepcopy(dict(change)), "development_round": iteration["development_round"],
            "previous_item_id": item_id, "source_measurement_batch": iteration["measurement_batch"],
        }
        iteration["thawed_item_versions"][item_id] = thaw
        iteration.setdefault("thaw_history", []).append(deepcopy(thaw))
    iteration.update(pending_facet_ids=sorted(pending), status="awaiting_measurement")
    return iteration


def prepare_resume(state: Mapping[str, Any]) -> dict[str, Any]:
    """Never relabel legacy measurements as completed fixed-cohort rounds."""
    if is_enabled(state):
        iteration = deepcopy(state["facet_iteration_state"])
        iteration.setdefault(
            "qualified_item_thaw_policy_version",
            QUALIFIED_ITEM_THAW_POLICY_VERSION,
        )
        iteration.setdefault(
            "qualified_item_thaw_quantile", QUALIFIED_ITEM_THAW_QUANTILE
        )
        iteration.setdefault(
            "qualified_item_thaw_metrics", list(QUALIFIED_ITEM_THAW_METRICS)
        )
        # The local-repair budget is a permanent policy.  Normalize old
        # checkpoints and reopen only the former finite-budget pause; other
        # safety pauses remain pauses and still require explicit resolution.
        iteration["local_repair_budget_policy"] = LOCAL_REPAIR_BUDGET_POLICY
        iteration["max_local_attempts"] = None
        if (
            iteration.get("status") == "paused"
            and iteration.get("paused_reason")
            == LEGACY_LOCAL_REPAIR_EXHAUSTION_REASON
            and int(iteration.get("completed_rounds") or 0) < MAX_COMPLETED_ROUNDS
        ):
            iteration.update(
                status="awaiting_repair",
                paused_reason=None,
                pending_facet_ids=[],
            )
        return iteration
    iteration = initial_iteration_state()
    if int(state.get("psychometric_analysis_round") or 0) or state.get("psychometric_iteration_history"):
        iteration.update(status="paused", paused_reason="Legacy iteration history requires a new explicitly configured run")
    return iteration
