"""Concurrent fixed-slot development using the existing single-item nodes."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sjt_system.runtime.io import write_json_atomic
from sjt_system.runtime.output_paths import scoped_output
from sjt_system.runtime.progress import emit_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_MAPS = (
    "item_skeletons", "skeleton_reviews", "skeleton_review_history",
    "skeleton_failures", "item_history", "item_pattern_profiles", "item_lineage",
    "item_content_evidence",
)


def pending_slots(state: Mapping[str, Any]) -> list[tuple[dict, dict]]:
    blueprint = state.get("blueprint") or {}
    cells = {str(row["cell_id"]): row for row in blueprint.get("cells") or []}
    specifications = {
        str(row["specification_id"]): row for row in state.get("item_specifications") or []
    }
    occupied = {str(row["item_id"]) for row in state.get("item_pool") or []}
    consumed = set(state.get("item_generation_completed_ids") or [])
    explicit_slots = "item_generation_completed_ids" in state
    counts: dict[str, int] = {}
    result = []
    seen = set()
    for slot in blueprint.get("slots") or []:
        item_id = str(slot["specification_id"])
        if item_id in seen:
            raise ValueError(f"Duplicate fixed slot: {item_id}")
        seen.add(item_id)
        cell_id = str(slot["blueprint_cell_id"])
        ordinal = counts.get(cell_id, 0)
        counts[cell_id] = ordinal + 1
        generated = int((state.get("blueprint_progress") or {}).get(cell_id, {}).get("generated") or 0)
        if item_id in occupied or item_id in consumed or (not explicit_slots and ordinal < generated):
            continue
        cell = deepcopy(cells[cell_id])
        specification = deepcopy(specifications.get(item_id) or {
            "specification_id": item_id,
            "blueprint_cell_id": cell_id,
            "target_dimension_id": cell["facet_id"],
        })
        result.append((cell, specification))
    return result


async def _develop_slot(
    state: Mapping[str, Any], cell: dict, specification: dict,
    *, checkpoint_path: Path,
) -> dict[str, Any]:
    # Imports stay local because execute_node dispatches back through executor.
    from sjt_system.workflow.execution_node import execute_node
    from sjt_system.workflow.interaction_nodes import automatic_approval_node, commit_node
    from sjt_system.workflow.item_nodes import (
        abandon_item_node, accept_item_node, prepare_item_review_node,
        prepare_item_revision_node,
    )
    from sjt_system.workflow.routes import (
        route_after_commit, route_after_execute, route_after_prepare_item_review,
    )

    item_id = str(specification["specification_id"])
    source = {
        "protocol": "fixed_slot_concurrent_v1",
        "blueprint": state.get("blueprint"),
        "test_specification": state.get("test_specification"),
        "specification_id": item_id,
        "model_settings": {name: os.getenv(name) for name in (
            "MODEL_ID", "BASE_URL", "STRUCTURED_OUTPUT_METHOD",
            "SKELETON_GENERATION_MODEL_ID", "SKELETON_GENERATION_TEMPERATURE",
        )},
    }
    from sjt_system.evaluation.virtual_content_review import is_enabled
    if is_enabled(state):
        source["virtual_content_review_protocol"] = state["virtual_content_review_protocol"]
    local = {
        **deepcopy(dict(state)),
        "item_development_mode": "automatic",
        "current_blueprint_cell": cell,
        "current_item_specification": specification,
        "current_item": None,
        "current_item_review": None,
        "active_psychometric_repair": None,
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "current_item_revision_count": 0,
        "current_item_rewrite_count": 0,
        "current_item_replacement_count": 0,
        "review_process_status": "not_started",
        "skeleton_slot_failure_pending": False,
        "current_skeleton_repair_required": False,
        "pending_action": None, "pending_state_update": None,
        "pending_interaction": None, "user_decision": None,
        "execution_history": [],
        "errors": [],
        "route": {
            "next_action": "generate_item", "reason": "Concurrent fixed-slot development",
            "target_item_id": item_id, "target_blueprint_cell_id": cell["cell_id"],
        },
    }
    node = "execute"
    baseline_progress = deepcopy((state.get("blueprint_progress") or {}).get(str(cell["cell_id"])) or {})
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("source") != source:
            raise ValueError(f"Fixed-slot checkpoint inputs changed: {item_id}")
        local = saved["state"]
        node = saved["next_node"]
        baseline_progress = saved["baseline_progress"]
        if local.get("status") == "failed" and node == "execute":
            local["status"] = "running"
    local.pop("max_steps", None)
    initial_steps = int(local.get("step_count") or 0)
    consumed_steps = 0

    def save() -> None:
        write_json_atomic(checkpoint_path, {
            "source": source, "next_node": node, "state": local,
            "baseline_progress": baseline_progress,
        })

    emit_progress({"type": "item_batch_progress", "status": "started", "item_id": item_id})
    try:
        while node != "done":
            if node == "execute":
                consumed_steps += 1
                local.update(await execute_node(local))
                if local.get("status") == "failed":
                    save()
                    break
                if local.get("skeleton_slot_failure_pending"):
                    node = "done"
                else:
                    node = route_after_execute(local)
            elif node == "automatic_approval":
                local.update(automatic_approval_node(local))
                node = "commit"
            elif node == "commit":
                local.update(commit_node(local))
                node = route_after_commit(local)
            elif node == "review":
                local.update(prepare_item_review_node(local))
                node = route_after_prepare_item_review(local)
            elif node in {"revise", "retry_item"}:
                local.update(prepare_item_revision_node(local))
                node = "execute"
            elif node in {"accept", "accept_latest"}:
                local.update(accept_item_node(local))
                node = "done"
            elif node == "abandon":
                local.update(abandon_item_node(local))
                node = "done"
            else:
                raise RuntimeError(f"Unexpected fixed-slot transition: {node}")
            save()
    except Exception as exc:
        local["status"] = "failed"
        local["errors"].append({"action": "generate_items_batch", "item_id": item_id, "message": str(exc)})
        save()
    failed = local.get("status") == "failed"
    emit_progress({"type": "item_batch_progress", "status": "failed" if failed else "completed", "item_id": item_id})
    return {
        "item_id": item_id, "cell_id": str(cell["cell_id"]), "state": local,
        "consumed_steps": max(consumed_steps, int(local.get("step_count") or 0) - initial_steps),
        "failed": failed,
        "progress_delta": {
            key: int(count) - int(baseline_progress.get(key) or 0)
            for key, count in (local.get("blueprint_progress") or {}).get(str(cell["cell_id"]), {}).items()
        },
    }


async def execute_item_generation_batch(state: Mapping[str, Any]) -> dict[str, Any]:
    slots = pending_slots(state)
    pending_ids = {str(specification["specification_id"]) for _, specification in slots}
    completed_ids = set(state.get("item_generation_completed_ids") or [])
    if "item_generation_completed_ids" not in state:
        completed_ids.update(str(row["specification_id"]) for row in (state.get("blueprint") or {}).get("slots") or []
                             if str(row["specification_id"]) not in pending_ids)
    checkpoint_dir = scoped_output("item_generation", PROJECT_ROOT / "outputs" / "item_generation") / str(state["run_id"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outcomes = await asyncio.gather(*(
        _develop_slot(state, cell, specification,
            checkpoint_path=checkpoint_dir / (quote(str(specification["specification_id"]), safe="") + ".json"))
        for cell, specification in slots
    ), return_exceptions=True)
    update: dict[str, Any] = {
        key: deepcopy(state.get(key) or {}) for key in MERGED_MAPS
    }
    pool = {str(row["item_id"]): deepcopy(row) for row in state.get("item_pool") or []}
    specifications = {str(row["specification_id"]): deepcopy(row) for row in state.get("item_specifications") or []}
    progress = deepcopy(state.get("blueprint_progress") or {})
    context_usage = deepcopy(state.get("context_usage") or {})
    errors = deepcopy(state.get("errors") or [])
    rejected = deepcopy(state.get("rejected_items") or [])
    history = deepcopy(state.get("execution_history") or [])
    failures = 0
    steps = int(state["step_count"])
    for (cell, specification), outcome in zip(slots, outcomes, strict=True):
        item_id = str(specification["specification_id"])
        if isinstance(outcome, BaseException):
            failures += 1
            errors.append({"action": "generate_items_batch", "item_id": item_id, "message": str(outcome)})
            continue
        local = outcome["state"]
        steps += outcome["consumed_steps"]
        failures += int(outcome["failed"])
        if not outcome["failed"]:
            completed_ids.add(item_id)
        for key in MERGED_MAPS:
            if item_id in (local.get(key) or {}):
                update[key][item_id] = deepcopy(local[key][item_id])
            elif key == "skeleton_reviews":
                update[key].pop(item_id, None)
        for row in local.get("item_specifications") or []:
            if str(row.get("specification_id")) == item_id:
                specifications[item_id] = deepcopy(row)
        for row in local.get("item_pool") or []:
            if str(row.get("item_id")) == item_id and item_id not in pool:
                pool[item_id] = deepcopy(row)
                category = (local.get("item_pattern_profiles") or {}).get(item_id, {}).get("context_category")
                if category:
                    context_usage[category] = context_usage.get(category, 0) + 1
        rejected.extend(deepcopy(row) for row in local.get("rejected_items") or []
                        if str(row.get("item_id")) == item_id)
        cell_id = str(cell["cell_id"])
        if not outcome["failed"]:
            counters = progress.setdefault(cell_id, {})
            for key, delta in outcome["progress_delta"].items():
                counters[key] = counters.get(key, 0) + delta
        errors.extend(deepcopy(local.get("errors") or []))
        for event in local.get("execution_history") or []:
            event = deepcopy(event)
            event["event_id"] = f"{event.get('event_id')}:{item_id}"
            event["batch_item_id"] = item_id
            history.append(event)
    update.update({
        "item_pool": list(pool.values()), "item_specifications": list(specifications.values()),
        "blueprint_progress": progress, "context_usage": context_usage,
        "item_generation_completed_ids": sorted(completed_ids),
        "errors": errors, "rejected_items": rejected, "execution_history": history,
        "step_count": steps + 1, "status": "failed" if failures else "running",
        "current_item": None, "current_item_specification": None,
        "current_blueprint_cell": None, "current_item_review": None,
        "current_item_repair_failure": None, "current_item_repair_attempted": False,
        "skeleton_slot_failure_pending": False, "review_process_status": "not_started",
    })
    return {"state_update": update, "summary": f"Concurrent fixed-slot batch: {len(slots)} slots, {failures} failures; progress saved per slot."}
