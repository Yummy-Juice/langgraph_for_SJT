"""Run a fresh configuration replica, or resume its own saved checkpoint.

This launcher only supplies operator decisions. Authoring, review, repair,
measurement budgets, and stopping rules remain owned by the current workflow.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.authoring.requirements import (
    SPECIFICATION_FIELDS,
    validate_test_specification,
)
from sjt_system.runtime.checkpoint import (
    DEFAULT_CHECKPOINT_ROOT,
    load_run_checkpoint,
    prepare_resumed_state,
    resolve_run_checkpoint_path,
    save_run_checkpoint,
)
from sjt_system.runtime.io import write_json_atomic
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.state import create_initial_state
from sjt_system.delivery.reporting import (
    development_round_batch_label,
    strict_development_round_entries,
)


EXPECTED_FACETS = {
    "extraversion_warmth",
    "extraversion_gregariousness",
    "agreeableness_altruism",
}
TASK_ROOT = PROJECT_ROOT / "outputs" / "configured_tasks"
REPLICA_REQUEST = (
    "Create a new NEO-PI-R personality situational judgment test for university "
    "students, measuring warmth, gregariousness, and altruism with 10 final "
        "items per facet and 30 final items in total. Generate one candidate per "
        "final item. If a candidate can no longer be repaired and is deferred, "
        "delete it and regenerate one same-cell candidate for the next measurement. "
        "Same-slot replacement and per-facet local repair submissions are unlimited. "
    "Complete exactly three development rounds at most: one baseline round and "
    "up to two successful repair rounds. Number measurement batches from one "
    "within each development round, retain batch numbers internally, and display "
    "development rounds only. "
        "Never admit deferred items "
    "to a temporary form. Use "
    "50 complete-profile virtual respondents per measurement, "
    "with a shared normal facet score distribution of mean 50 and SD 15. "
    "Generate all test content in zh-CN."
)


def new_state_from_source(source_id: str, run_id: str) -> tuple[dict, dict]:
    source_path = DEFAULT_CHECKPOINT_ROOT / f"{source_id}.json"
    # Read the source without migration or writeback. Only configuration is reused.
    source = json.loads(source_path.read_text(encoding="utf-8"))["state"]
    old_spec = source["test_specification"]
    spec = {key: deepcopy(old_spec[key]) for key in SPECIFICATION_FIELDS}
    errors = validate_test_specification(spec)
    if errors:
        raise ValueError(f"Source specification is incompatible: {errors}")
    if set(spec["facet_item_counts"]) != EXPECTED_FACETS:
        raise ValueError("Source does not contain the requested three facets")
    if set(spec["facet_item_counts"].values()) != {10} or spec["final_item_count"] != 30:
        raise ValueError("Source must request 10 items per facet and 30 in total")
    old_sample = source["virtual_sample_config"]
    distribution = old_sample.get("shared_facet_score_distribution") or old_sample["score_distribution"]
    if old_sample.get("sample_size") != 50:
        raise ValueError("Source must contain exactly 50 complete-profile respondents")
    if float(distribution["mean"]) != 50 or float(distribution["sd"]) != 15:
        raise ValueError("Source must use M=50 and SD=15")
    knowledge = source.get("repair_knowledge_config") or {}
    knowledge_mode = knowledge.get("mode") or "run_only"
    state = create_initial_state(
        REPLICA_REQUEST,
        repair_knowledge_mode=knowledge_mode,
        repair_knowledge_namespace=knowledge.get("namespace") or "development",
    )
    state.update(
        run_id=run_id,
        test_specification=spec,
        failure_call_replenishment=True,
        specification_sources={key: "user" for key in SPECIFICATION_FIELDS},
        requirements_confirmed=True,
        confirmed_requirement_fields=sorted(SPECIFICATION_FIELDS),
    )
    sample_selection = {
        "sample_size_per_condition": 50,
        "seed": int(old_sample.get("seed", 7)),
        "max_concurrency": 0,
        "max_retries": int(old_sample.get("max_retries", 2)),
        "score_distribution": {"family": "normal", "mean": 50.0, "sd": 15.0},
    }
    manifest = {
        "run_id": run_id,
        "source_run_id": source_id,
        "created_at": utc_timestamp(),
        "project_root": str(PROJECT_ROOT),
        "interpreter": sys.executable,
        "user_request": state["user_request"],
        "test_specification": spec,
        "sample_selection": sample_selection,
        "repair_knowledge_config": deepcopy(state["repair_knowledge_config"]),
        "virtual_content_review_protocol": state["virtual_content_review_protocol"],
        "facet_iteration_policy": {
            "total_development_rounds": 3,
            "baseline_rounds": 1,
            "repair_rounds": 2,
            "batch_counter": "per_development_round_start_at_1",
            "batch_display": "internal_only",
        },
        "authorization": {
            "fresh_items_and_blueprint": True,
            "reuse_old_responses": False,
            "continue_from_checkpoint_on_interruption": True,
            "change_review_or_repair_logic": "candidate_count_equals_final_with_call_budget_replenishment",
            "failed_call_limit": 3,
            "replenish_same_cell_after_failed_call_limit": True,
            "replenish_same_cell_after_deferred": True,
            "deferred_items_enter_next_measurement": False,
            "same_cell_replacement_enters_next_measurement": True,
            "same_cell_replacement_quota": "unlimited",
            "max_item_replacement_attempts": None,
            "temporary_form_with_deferred_items": False,
            "qualified_item_direct_evidence_pause": False,
            "qualified_item_thaw_policy": "bottom_quartile_any_validity_metric",
            "qualified_item_thaw_quantile": 0.25,
            "qualified_item_thaw_metrics": [
                "target_hedges_g",
                "target_ipip_spearman_rho",
                "discriminant_delta_min",
            ],
            "hash_verification": False,
            "visual_qa": False,
        },
        "checkpoint": str(DEFAULT_CHECKPOINT_ROOT / f"{run_id}.json"),
    }
    state["execution_history"].append({
        "event_id": f"{run_id}:0:configuration_replica:completed",
        "run_id": run_id,
        "step": 0,
        "node": "configured_launcher",
        "action": "confirm_configuration_replica",
        "event_type": "completed",
        "recorded_at": utc_timestamp(),
        "reason": f"User authorized a fresh task with the configuration of {source_id}; no item or response reuse.",
        "approval_source": "user",
    })
    return state, manifest


def operator_decision(payload: dict, manifest: dict, task_dir: Path) -> dict:
    kind = payload.get("type")
    if payload.get("validation_error"):
        write_json_atomic(task_dir / "operator_interrupt.json", payload)
        raise RuntimeError(f"Operator decision failed validation: {payload['validation_error']}")
    if kind == "repair_knowledge_selection":
        decision = {"mode": manifest["repair_knowledge_config"]["mode"]}
    elif kind == "item_development_mode_selection":
        decision = {"mode": "automatic"}
    elif kind == "virtual_sample_selection":
        decision = deepcopy(manifest["sample_selection"])
    elif kind == "post_virtual_response_decision":
        decision = {"decision": "start"}
    elif kind == "agent_result_approval" and "approve" in payload.get("available_decisions", []):
        decision = {"decision": "approve", "feedback": None}
    else:
        write_json_atomic(task_dir / "operator_interrupt.json", payload)
        raise RuntimeError(f"Unconfigured operator decision: {kind!r}, action={payload.get('action')!r}")
    print(f"[operator] {kind}: {json.dumps(decision, ensure_ascii=False)}", flush=True)
    with (task_dir / "operator_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "recorded_at": utc_timestamp(), "type": kind,
            "action": payload.get("action"), "decision": decision,
            "authorization": "user_requested_full_run_and_continue_option_1",
        }, ensure_ascii=False) + "\n")
    return decision


def outcome(state: dict, *, error: str | None = None) -> dict:
    history = state.get("psychometric_iteration_history") or []
    display_entries = strict_development_round_entries(history, max_rounds=3)
    development_rounds = []
    for row in history:
        if not isinstance(row, dict):
            continue
        controller = row.get("facet_iteration_state") or {}
        development_round = row.get("development_round") or (
            controller.get("development_round")
            if isinstance(controller, dict)
            else None
        )
        try:
            development_round = int(development_round)
        except (TypeError, ValueError):
            continue
        if development_round not in development_rounds and 1 <= development_round <= 3:
            development_rounds.append(development_round)
    checkpoint_candidates = [
        path for path in DEFAULT_CHECKPOINT_ROOT.glob(f"{state['run_id']}*.json")
        if path.is_file()
    ]
    checkpoint_path = max(
        checkpoint_candidates,
        key=lambda path: path.stat().st_mtime_ns,
        default=DEFAULT_CHECKPOINT_ROOT / f"{state['run_id']}.json",
    )
    return {
        "run_id": state["run_id"], "recorded_at": utc_timestamp(),
        "status": state.get("status"), "current_phase": state.get("current_phase"),
        "committed_rounds": sorted(development_rounds),
        "round_labels": [
            development_round_batch_label(history, row)
            for row in display_entries
        ],
        "candidate_count": len(state.get("item_pool") or []),
        "selected_count": len(state.get("selected_items") or []),
        "stop_reason": state.get("virtual_content_review_stop_reason"),
        "scenario_repair_pause": state.get("scenario_repair_pause"),
        "latest_errors": (state.get("errors") or [])[-3:],
        "completion_checks": state.get("completion_checks"),
        "unmet_completion_conditions": state.get("unmet_completion_conditions"),
        "checkpoint": str(checkpoint_path),
        "error": error,
    }


async def execute(state: dict, manifest: dict, task_dir: Path) -> int:
    import cli_app
    from sjt_system.workflow.graph import build_sjt_graph

    cli_app.app = build_sjt_graph()
    cli_app.prompt_user_decision = lambda payload: operator_decision(payload, manifest, task_dir)
    try:
        result = await cli_app.run_with_trace(state, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
    except Exception as exc:
        traceback.print_exc()
        checkpoint = resolve_run_checkpoint_path(state["run_id"], DEFAULT_CHECKPOINT_ROOT)
        result = load_run_checkpoint(checkpoint)["state"] if checkpoint.is_file() else state
        result["status"] = "failed"
        result["errors"] = [*result.get("errors", []), {
            "action": "configured_launcher", "message": str(exc),
            "error_type": type(exc).__name__, "recorded_at": utc_timestamp(),
        }]
        save_run_checkpoint(result, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
        write_json_atomic(task_dir / "outcome.json", outcome(result, error=str(exc)))
        return 2
    save_run_checkpoint(result, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
    write_json_atomic(task_dir / "outcome.json", outcome(result))
    cli_app.print_final_result(result)
    print("[outcome] " + json.dumps(outcome(result), ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "completed" else 3


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    UUID(args.run_id)
    task_dir = TASK_ROOT / args.run_id
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / "launcher.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Task has an existing launcher lock; inspect its PID before resuming: {lock_path}")
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    try:
        os.chdir(PROJECT_ROOT)
        checkpoint = resolve_run_checkpoint_path(args.run_id, DEFAULT_CHECKPOINT_ROOT)
        manifest_path = task_dir / "task_manifest.json"
        if args.source_run:
            UUID(args.source_run)
            if checkpoint.exists() or manifest_path.exists():
                raise ValueError("Fresh task already exists; omit --source-run to resume it")
            state, manifest = new_state_from_source(args.source_run, args.run_id)
            write_json_atomic(manifest_path, manifest)
            save_run_checkpoint(state, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = load_run_checkpoint(checkpoint)["state"]
            if state.get("status") == "completed":
                print("[outcome] Task is already completed; no model calls launched.", flush=True)
                return 0
            state = prepare_resumed_state(state, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
        # Preserve the configured replacement policy across checkpoint resumes,
        # including checkpoints written before this state field was persisted.
        state["failure_call_replenishment"] = True
        print(f"[task] run_id={args.run_id} checkpoint={checkpoint}", flush=True)
        print(f"[task] protocol={state['virtual_content_review_protocol']} python={sys.executable}", flush=True)
        if args.prepare_only:
            print("[task] Configuration saved; no model calls launched.", flush=True)
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.environ["TELEMETRY_SESSION"] = f"configured_{args.run_id}_{stamp}_{os.getpid()}"
        write_json_atomic(task_dir / "active_launch.json", {
            "run_id": args.run_id, "pid": os.getpid(), "started_at": utc_timestamp(),
            "interpreter": sys.executable, "checkpoint": str(checkpoint),
            "telemetry_session": os.environ["TELEMETRY_SESSION"],
        })
        return asyncio.run(execute(state, manifest, task_dir))
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
