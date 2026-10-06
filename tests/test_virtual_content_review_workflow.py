from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from test_virtual_content_review import FakeInvoke, evidence_fixture, item_fixture
from sjt_system.authoring import bank, items as authoring_items
from sjt_system.authoring.generation_plan import GENERATION_BLUEPRINT_VERSION
from sjt_system.config import PSJT_RESPONSE_INSTRUCTION
from sjt_system.evaluation import virtual_content_review as review
from sjt_system.evaluation.psychometrics import _qualified_item_metric_snapshots
from sjt_system.evaluation.round_results import build_iteration_metrics_snapshot
from sjt_system.runtime.iteration_metrics import persist_iteration_metrics_snapshot
from sjt_system.runtime.output_paths import output_scope
from sjt_system.state import create_initial_state
from sjt_system.workflow import executor, item_nodes, router, routes
from sjt_system.workflow import virtual_content_review as batch


def measured_state(*, measurement=1, count=2):
    state = create_initial_state("virtual workflow", repair_knowledge_mode="run_only")
    # Historical workflow coverage is separate from the fixed-cohort controller tests.
    state["facet_iteration_state"] = None
    candidates = []
    for index in range(count):
        item = item_fixture()
        item.update(item_id=f"item-{index + 1}", scenario=f"Original scenario {index + 1}")
        candidates.append(item)
    state.update(
        run_id="virtual-workflow-test", requirements_confirmed=True,
        test_specification={"target_population": "virtual university students"},
        blueprint={"version": GENERATION_BLUEPRINT_VERSION,
                   "cells": [{"cell_id": "cell-1", "facet_id": "target",
                              "planned_retention_count": 1, "planned_generation_count": count}],
                   "slots": [{"specification_id": item["item_id"], "blueprint_cell_id": "cell-1"}
                             for item in candidates]},
        blueprint_progress={"cell-1": {"generated": count, "passed": count}},
        item_specifications=[{"specification_id": item["item_id"], "blueprint_cell_id": "cell-1"}
                             for item in candidates],
        item_pool=candidates, frozen_item_bank=deepcopy(candidates),
        psychometric_analysis_round=measurement,
        item_statistics={item["item_id"]: stat(False) for item in candidates},
        virtual_response_data_ref="formal/manifest.json",
        virtual_response_summary={"total": 24},
        item_generation_completed_ids=[item["item_id"] for item in candidates],
    )
    state["psychometric_iteration_history"] = [measurement_record(state)]
    state["items_to_revise"] = [entry(state, item) for item in candidates]
    return state


def stat(passed):
    return {"quality_evaluation": {"facet_citc": {}, "recommendation": "retain" if passed else "revise"},
            "qualification": {"qualified": passed,
                              **{name: passed for name in review.ITERATION_GATES}}}


def measurement_record(state):
    measurement = state["psychometric_analysis_round"]
    candidates = state["frozen_item_bank"]
    ids = [row["item_id"] for row in candidates]
    return {"analysis_round": measurement, "virtual_content_review_protocol": review.PROTOCOL,
            "round_label": review.round_label(measurement), "workflow_stage": "virtual_content_review",
            "form_item_ids": ids[:1], "item_count": 1, "candidate_count": len(ids),
            "form_status": "complete", "form_metrics": {}, "candidate_item_metrics": {
                row["item_id"]: {"item_id": row["item_id"], "item_version": row["version"],
                                 "qualified": (state["item_statistics"].get(row["item_id"]) or {}).get("qualification", {}).get("qualified", False)}
                for row in candidates},
            "item_snapshots": {row["item_id"]: deepcopy(row) for row in candidates},
            "item_statistics_snapshot": deepcopy(state["item_statistics"]),
            "form_optimizer": {"status": "validated", "selected_item_ids": ids[:1]}}


def entry(state, item, *, revision_round=1):
    evidence = evidence_fixture()
    evidence["current_item"] = deepcopy(item)
    return {"item_id": item["item_id"], "blueprint_cell_id": item["blueprint_cell_id"],
            "revision_round": revision_round, "repair_protocol": review.PROTOCOL,
            "queue_status": "pending_virtual_review", "diagnosis_evidence": evidence,
            "source_analysis_round": state["psychometric_analysis_round"],
            "baseline_metrics": deepcopy(state["item_statistics"].get(item["item_id"], {})),
            "atomic_repair_advice": {"protocol": review.PROTOCOL, "decision": "investigate", "repair_tasks": []}}


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path):
    with output_scope(tmp_path):
        yield


def install_models(monkeypatch, models):
    def invoker(local):
        # Workers receive a shared read-only baseline. Bind the mock by the
        # first expert payload instead of giving it hidden item metadata.
        class RoutedInvoke:
            def __init__(self):
                self.model = None

            async def __call__(self, kind, payload):
                if self.model is None:
                    scenario = payload["item"]["scenario"]
                    index = int(scenario.rsplit(" ", 1)[-1])
                    self.model = models[f"item-{index}"]
                return await self.model(kind, payload)
        return RoutedInvoke(), {"expert_review": {"model_id": "offline-mock"}}
    monkeypatch.setattr(review, "make_model_invoker", invoker)


def select(state, monkeypatch, *, plateau=False):
    monkeypatch.setattr(executor, "assess_form_plateau", lambda *_args, **_kwargs: {
        "reached": plateau, "trajectory": [], "current_round": state["psychometric_analysis_round"]})
    monkeypatch.setattr(executor, "build_virtual_content_repair_entry", entry_adapter)
    return asyncio.run(executor.execute_item_selection_with_diagnosis(state))["state_update"]


def entry_adapter(state, item, *, revision_round):
    return entry(state, item, revision_round=revision_round)


def test_new_runs_enable_protocol_but_legacy_states_are_not_relabelled():
    assert review.is_enabled(create_initial_state("new"))
    assert not review.is_enabled({"item_pool": []})
    assert not review.is_enabled(create_initial_state("legacy", virtual_content_review_protocol="scenario_detection_first_v1"))


def test_initial_admission_has_no_expert_call_or_fabricated_content_pass():
    state = measured_state(count=1)
    item = deepcopy(state["item_pool"][0])
    item.update(response_instruction=PSJT_RESPONSE_INSTRUCTION, construct_rationale="frozen target rationale",
                contamination_risks=[], context_signature="original signature")
    state.update(psychometric_analysis_round=0, item_pool=[], current_item=item,
                 current_blueprint_cell=state["blueprint"]["cells"][0],
                 current_item_specification={"specification_id": "item-1", "context_category": "team"},
                 route={"next_action": "generate_item"})
    state.update(item_nodes.prepare_item_review_node(state))
    assert routes.route_after_prepare_item_review(state) == "accept"
    assert state["current_item_review"] is None
    update = authoring_items.build_accept_item_update(state)
    assert len(update["item_pool"]) == 1
    assert update["item_content_evidence"]["item-1"]["status"] == "not_evaluated"
    assert update["item_content_evidence"]["item-1"]["last_reviewed_version"] is None
    assert [row["event"] for row in update["item_history"]["item-1"]] == ["generated", "initial_candidate_admitted"]


def test_initial_admission_cannot_be_used_after_measurement():
    state = measured_state(count=1)
    state.update(current_item=state["item_pool"][0], route={"next_action": "generate_item"})
    with pytest.raises(ValueError, match="施测后"):
        item_nodes.prepare_item_review_node(state)


def test_only_failing_unlocked_items_enter_queue(monkeypatch):
    state = measured_state(count=3)
    state["items_to_revise"] = []
    state["item_statistics"]["item-1"] = stat(True)
    state["locked_retained_item_versions"] = {"item-2": 2}
    update = select(state, monkeypatch)
    assert [row["item_id"] for row in update["items_to_revise"]] == ["item-3"]
    assert update["psychometric_repair_confirmation"] is None
    assert update["locked_retained_item_versions"] == {"item-1": 2, "item-2": 2}
    assert update["item_final_dispositions"]["item-2"]["monitoring_pass"] is False
    assert len(update["psychometric_monitoring_warnings"]) == 1
    routed = asyncio.run(router.router_node({**state, **update}))
    assert routed["route"]["next_action"] == "psychometric_repair_batch"


def test_plateau_retains_failing_items_without_changing_raw_gate_results(monkeypatch):
    state = measured_state()
    state["items_to_revise"] = []
    update = select(state, monkeypatch, plateau=True)
    assert update["items_to_revise"] == []
    assert update["selection_results"]["plateau_finalized"] is True
    assert len(update["locked_retained_item_versions"]) == 2
    assert all(row["status"] == "qualified_locked" for row in update["item_final_dispositions"].values())
    record = update["psychometric_iteration_history"][-1]
    assert all(row["qualified"] is False for row in record["candidate_item_metrics"].values())
    assert record["item_dispositions"]["item-1"]["warning_reason"]
    assert update["virtual_content_review_stop_reason"] == "plateau_retained"


def test_new_protocol_requires_all_four_gates_not_descriptive_or_recommendation_flags(monkeypatch):
    state = measured_state(count=1)
    state["items_to_revise"] = []
    state["item_statistics"]["item-1"] = stat(True)
    state["item_statistics"]["item-1"]["qualification"].pop("target_ipip_spearman_rho_pass")
    state["item_statistics"]["item-1"].update(difficulty=.5, effective_options=4)
    update = select(state, monkeypatch)
    assert [row["item_id"] for row in update["items_to_revise"]] == ["item-1"]
    assert not update["locked_retained_item_versions"]
    assert review.iteration_gates_pass(stat(True))
    assert not review.iteration_gates_pass(state["item_statistics"]["item-1"])


def test_plateau_keeps_actual_monitoring_pass_separate_from_retention_basis(monkeypatch):
    state = measured_state(count=2)
    state["items_to_revise"] = []
    state["item_statistics"]["item-1"] = stat(True)
    update = select(state, monkeypatch, plateau=True)
    passed, failed = (update["item_final_dispositions"][identifier] for identifier in ("item-1", "item-2"))
    assert passed["monitoring_pass"] is True and passed["retention_basis"] == "four_iteration_gates_passed"
    assert failed["monitoring_pass"] is False and failed["retention_basis"] == "plateau_retained"


def test_plateau_preserves_prior_gate_lock_when_monitoring_drifts(monkeypatch):
    state = measured_state(count=1)
    state["items_to_revise"] = []
    item = state["item_pool"][0]
    item_id = item["item_id"]
    state["locked_retained_item_versions"] = {item_id: item["version"]}
    state["item_final_dispositions"] = {
        item_id: {
            "status": "qualified_locked",
            "retention_basis": "four_iteration_gates_passed",
            "item_version": item["version"],
            "qualification_snapshot": deepcopy(stat(True)),
        }
    }
    state["item_statistics"][item_id] = stat(False)

    update = select(state, monkeypatch, plateau=True)
    disposition = update["item_final_dispositions"][item_id]
    assert disposition["retention_basis"] == "four_iteration_gates_passed"
    assert disposition["monitoring_pass"] is False
    assert disposition["qualification_snapshot"]["qualification"]["qualified"] is True


def test_fourth_measurement_queues_same_slot_replacements_for_next_measurement(monkeypatch):
    state = measured_state(measurement=4)
    state["items_to_revise"] = []
    update = select(state, monkeypatch)
    assert len(update["items_to_revise"]) == len(state["item_pool"])
    assert all(row["deferred_replacement_only"] for row in update["items_to_revise"])
    assert all(row["queue_status"] == "deferred_replenishment" for row in update["items_to_revise"])
    assert all(row["diagnosis_status"] == "repair_rounds_exhausted" for row in update["items_to_revise"])
    assert update["selection_results"] is None
    routed = asyncio.run(router.router_node({**state, **update}))
    assert routed["status"] == "running" and routed["route"]["next_action"] == "psychometric_repair_batch"
    assert router._prepare_automatic_blueprint_gap_replenishment(state, [{"blueprint_cell_id": "cell-1"}]) is None


def test_three_review_round_stop_still_routes_ready_form_to_assembly(monkeypatch):
    state = measured_state(measurement=4, count=1)
    state["items_to_revise"] = []
    state["item_statistics"]["item-1"] = stat(True)

    async def optimize(*_args, **_kwargs):
        return {"status": "validated", "selected_item_ids": ["item-1"]}

    monkeypatch.setattr(executor, "optimize_test_form_with_agent", optimize)
    update = select(state, monkeypatch)

    assert update["virtual_content_review_stop_reason"] == (
        "three_content_review_rounds_completed_without_deferred_items"
    )
    assert update["selection_results"]["status"] == "ready_for_assembly"
    routed = asyncio.run(router.router_node({**state, **update}))
    assert routed["route"]["next_action"] == "assemble_test"


@pytest.mark.parametrize("decision", ["RESCORE", "SUPPLEMENT"])
def test_delivery_checks_do_not_bypass_frozen_key_or_fixed_slot_rules(decision):
    state = measured_state(count=1)
    state.update(items_to_revise=[], test_review_result={"decision": decision}, selection_results={"status": "ready_for_assembly"})
    result = asyncio.run(router.router_node(state))
    assert result["status"] == "stopped" and result["route"]["next_action"] == "finish"
    assert result["virtual_content_review_stop_reason"] == "frozen_design_or_coverage_error"
    from sjt_system.delivery.lifecycle import run_test_rescore
    with pytest.raises(ValueError, match="冻结评分键"):
        run_test_rescore(state)


def test_batch_requires_persisted_round_before_any_model_call(monkeypatch):
    state = measured_state()
    state["psychometric_iteration_history"] = []
    monkeypatch.setattr(review, "make_model_invoker", lambda _state: pytest.fail("must not call a model"))
    with pytest.raises(ValueError, match="先保存"):
        asyncio.run(batch.execute_virtual_content_review_batch(state))


def test_batch_rejects_stale_round_or_changed_measurement_inputs(monkeypatch):
    state = measured_state(count=1)
    monkeypatch.setattr(review, "make_model_invoker", lambda _state: pytest.fail("must not call a model"))
    state["items_to_revise"][0]["source_analysis_round"] = 0
    with pytest.raises(ValueError, match="轮次"):
        asyncio.run(batch.execute_virtual_content_review_batch(state))
    state["items_to_revise"][0]["source_analysis_round"] = 1
    state["item_statistics"]["item-1"]["difficulty"] = .9
    with pytest.raises(ValueError, match="账本"):
        asyncio.run(batch.execute_virtual_content_review_batch(state))


def test_missing_materials_are_archived_as_pause_without_a_model_call(monkeypatch):
    state = measured_state(count=1)
    state["items_to_revise"][0]["diagnosis_evidence"]["planning_error"] = "Missing frozen construct material"
    models = {"item-1": FakeInvoke()}
    install_models(monkeypatch, models)
    update = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    result = update["scenario_repair_progress"]["item-1"]
    from pathlib import Path
    archive = json.loads(Path(result["archive_ref"]).read_text(encoding="utf-8"))
    assert archive["status"] == "paused" and archive["stage"] == "materials"
    assert archive["errors"][0]["message"] == "Missing frozen construct material"
    assert models["item-1"].calls == []
    assert "item_pool" not in update and "item_statistics" not in update


def test_deferred_result_without_version_bound_interviews_pauses_batch(monkeypatch):
    state = measured_state(count=1)

    async def invalid(**kwargs):
        return {"protocol": review.PROTOCOL, "status": "deferred", "stage": "diagnosed",
                "source_analysis_round": 1, "reviewed_item_id": "item-1", "reviewed_item_version": 2,
                "expert_reviews": [{}, {}], "cognitive_interviews": [], "archive_ref": "mock.json",
                "diagnosis": {"decision": "defer", "replacement_scope": "none", "repair_tasks": [], "summary": "Uncertain"}}

    monkeypatch.setattr(review, "make_model_invoker", lambda _state: (None, {}))
    monkeypatch.setattr(review, "run_virtual_content_repair", invalid)
    update = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert update["scenario_repair_pause"]["failures"]["item-1"]
    assert "item_final_dispositions" not in update and "item_pool" not in update


def test_ready_batch_commits_once_then_invalidates_all_measurement_refs(monkeypatch):
    state = measured_state()
    original = deepcopy(state)
    models = {identifier: FakeInvoke() for identifier in ("item-1", "item-2")}
    install_models(monkeypatch, models)
    result = asyncio.run(executor._execute_psychometric_repair_batch(
        action="psychometric_repair_batch", route={"next_action": "psychometric_repair_batch"}, state=state))
    update = result["state_update"]
    assert state == original
    assert [row["version"] for row in update["item_pool"]] == [3, 3]
    assert update["psychometric_repair_rounds"] == {"item-1": 1, "item-2": 1}
    assert update["virtual_response_data_ref"] is None and update["item_statistics"] == {}
    assert update["previous_virtual_response_data_ref"] == "formal/manifest.json"
    assert all(row["last_reviewed_version"] == 2 and row["item_version"] == 3
               for row in update["item_content_evidence"].values())
    assert all(row["post_edit_comprehensibility"] == "not_reassessed"
               for row in update["item_content_evidence"].values())
    assert all(row["committed"] and row["new_item_version"] == 3 for row in update["virtual_content_review_history"])
    assert all(len(model.calls) == 16 for model in models.values())
    assert all(not kind.startswith("post_") for model in models.values() for kind, _ in model.calls)


def test_one_transient_failure_retries_with_five_call_budget_and_commits_batch(monkeypatch):
    state = measured_state()
    original = deepcopy(state)
    models = {"item-1": FakeInvoke(), "item-2": FakeInvoke(fault_kind="interview_probe", fault_mode="timeout")}
    install_models(monkeypatch, models)
    committed = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert state == original
    assert committed["scenario_repair_pause"] is None
    assert committed["virtual_response_data_ref"] is None
    assert committed["item_statistics"] == {}
    assert review.MAX_CALLS == 5
    assert len(models["item-1"].calls) == 16
    assert len(models["item-2"].calls) == 17
    assert [row["version"] for row in committed["item_pool"]] == [3, 3]


def test_ready_archive_is_revalidated_before_reuse(monkeypatch):
    state = measured_state(count=1)
    models = {"item-1": FakeInvoke()}
    install_models(monkeypatch, models)
    ready = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    from pathlib import Path
    from sjt_system.runtime.io import write_json_atomic
    path = Path(ready["scenario_repair_progress"]["item-1"]["archive_ref"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["candidate"]["scenario"] = "Unauthorized scenario modification"
    write_json_atomic(path, saved)
    model_calls = len(models["item-1"].calls)
    rejected = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert rejected["scenario_repair_pause"] is not None
    assert len(models["item-1"].calls) == model_calls
    assert "item_pool" not in rejected


def test_deferred_result_is_deleted_and_replenished_in_same_slot(monkeypatch):
    state = measured_state(count=1)

    async def deferred(**kwargs):
        item = kwargs["item"]
        return {
            "protocol": review.PROTOCOL,
            "status": "deferred",
            "stage": "diagnosed",
            "source_analysis_round": 1,
            "reviewed_item_id": item["item_id"],
            "reviewed_item_version": item["version"],
            "expert_reviews": [{}, {}],
            "cognitive_interviews": [{} for _ in range(6)],
            "archive_ref": str(review.archive_path(state, item)),
            "diagnosis": {"decision": "defer", "replacement_scope": "none",
                          "repair_tasks": [], "summary": "Insufficient evidence"},
            "candidate": None,
            "staged_design": None,
        }

    async def rebuild(_archive, _stage):
        original = state["item_pool"][0]
        candidate = deepcopy(original)
        candidate.update(item_id="item-1-R1", version=1, scenario="Replacement scenario")
        return {
            "item": deepcopy(candidate),
            "skeleton": {"replacement": True},
            "specification": {"specification_id": "item-1-R1"},
            "slot": {"specification_id": "item-1-R1", "blueprint_cell_id": "cell-1"},
            "root_item_id": "item-1",
            "replacement_number": 1,
            "replaces_item_id": "item-1",
        }

    monkeypatch.setattr(review, "run_virtual_content_repair", deferred)
    monkeypatch.setattr(
        "sjt_system.evaluation.repair_reconstruction.make_rebuilder",
        lambda *_args, **_kwargs: rebuild,
    )
    replenished = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert [row["item_id"] for row in replenished["item_pool"]] == ["item-1-R1"]
    assert [row["specification_id"] for row in replenished["blueprint"]["slots"]] == ["item-1-R1"]
    assert replenished["item_final_dispositions"]["item-1"]["status"] == "replaced"
    assert replenished["virtual_response_data_ref"] is None
    assert replenished["deferred_replacement_measurement_pending"] is True


def test_deferred_item_is_replaced_in_same_slot_for_next_measurement(monkeypatch):
    state = measured_state(count=1)
    install_models(monkeypatch, {"item-1": FakeInvoke(decision="defer")})

    async def deferred(**kwargs):
        item = kwargs["item"]
        return {
            "protocol": review.PROTOCOL,
            "status": "deferred",
            "stage": "diagnosed",
            "source_analysis_round": 1,
            "reviewed_item_id": item["item_id"],
            "reviewed_item_version": item["version"],
            "expert_reviews": [{}, {}],
            "cognitive_interviews": [{} for _ in range(6)],
            "archive_ref": str(review.archive_path(state, item)),
            "diagnosis": {
                "decision": "defer",
                "replacement_scope": "none",
                "repair_tasks": [],
                "summary": "Insufficient evidence",
            },
            "candidate": None,
            "staged_design": None,
        }

    async def rebuild(_archive, _stage):
        original = state["item_pool"][0]
        candidate = deepcopy(original)
        candidate.update(item_id="item-1-R1", version=1, scenario="Replacement scenario")
        return {
            "item": deepcopy(candidate),
            "skeleton": {"replacement": True},
            "specification": {"specification_id": "item-1-R1"},
            "slot": {"specification_id": "item-1-R1", "blueprint_cell_id": "cell-1"},
            "root_item_id": "item-1",
            "replacement_number": 1,
            "replaces_item_id": "item-1",
        }

    monkeypatch.setattr(review, "run_virtual_content_repair", deferred)
    monkeypatch.setattr(
        "sjt_system.evaluation.repair_reconstruction.make_rebuilder",
        lambda *_args, **_kwargs: rebuild,
    )
    update = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert [row["item_id"] for row in update["item_pool"]] == ["item-1-R1"]
    assert [row["specification_id"] for row in update["blueprint"]["slots"]] == ["item-1-R1"]
    assert update["item_final_dispositions"]["item-1"]["status"] == "replaced"
    assert update["item_pool"][0]["version"] == 1
    assert update["virtual_response_data_ref"] is None
    assert update["deferred_replacement_measurement_pending"] is True


def test_replacement_keeps_slot_count_and_root_budget(monkeypatch):
    state = measured_state(count=1)
    state["psychometric_repair_rounds"] = {"item-1": 2}

    async def staged(**kwargs):
        original = kwargs["item"]
        candidate = deepcopy(original)
        candidate.update(item_id="item-1-R1", version=1, scenario="Replacement scenario")
        return {"protocol": review.PROTOCOL, "status": "ready", "stage": "staged",
                "source_analysis_round": 1, "reviewed_item_id": "item-1", "reviewed_item_version": 2,
                "expert_reviews": [{}, {}], "cognitive_interviews": [{} for _ in range(6)],
                "requires_remeasurement": True, "candidate": candidate, "archive_ref": "mock-archive.json",
                "diagnosis": {"decision": "replace"},
                "staged_design": {"item": deepcopy(candidate), "root_item_id": "item-1", "replaces_item_id": "item-1",
                                  "replacement_number": 1, "skeleton": {"new": True},
                                  "specification": {"specification_id": "item-1-R1"},
                                  "slot": {"specification_id": "item-1-R1", "blueprint_cell_id": "cell-1"}}}
    monkeypatch.setattr(review, "make_model_invoker", lambda _state: (None, {}))
    monkeypatch.setattr(review, "run_virtual_content_repair", staged)
    update = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert len(update["blueprint"]["slots"]) == len(state["blueprint"]["slots"])
    assert update["blueprint"]["cells"] == state["blueprint"]["cells"]
    assert update["item_lineage"]["item-1-R1"]["root_item_id"] == "item-1"
    assert update["psychometric_repair_rounds"]["item-1-R1"] == 3

    async def invalid_root(**kwargs):
        result = await staged(**kwargs)
        result["staged_design"]["root_item_id"] = "new-budget-root"
        return result

    monkeypatch.setattr(review, "run_virtual_content_repair", invalid_root)
    rejected = asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"]
    assert rejected["scenario_repair_pause"]["failures"]["item-1"]
    assert "item_pool" not in rejected and "blueprint" not in rejected and "psychometric_repair_rounds" not in rejected
    next_state = {**state, **update, "psychometric_analysis_round": 2,
                  "frozen_item_bank": deepcopy(update["item_pool"]),
                  "item_statistics": {"item-1-R1": stat(False)}}
    next_state["psychometric_iteration_history"].append(measurement_record(next_state))
    next_state["items_to_revise"] = [entry(next_state, next_state["item_pool"][0])]
    with pytest.raises(ValueError, match="根项"):
        asyncio.run(batch.execute_virtual_content_review_batch(next_state))


def test_design_replacement_carries_cited_problems_and_respects_authorized_scope():
    from test_virtual_content_review import diagnosis, expert_report
    from sjt_system.evaluation.repair_reconstruction import _content_replacement_context
    from sjt_system.evaluation.repair_recovery import RepairStopped
    original = item_fixture()
    authorization = diagnosis(decision="replace")
    authorization["replacement_scope"] = "skeleton"
    data = {"diagnosis": authorization, "expert_reviews": [expert_report(), expert_report()],
            "cognitive_interviews": [], "source": {"analysis_round": 2},
            "plan": {"constraints": evidence_fixture()["normal_constraints"]}}
    context = _content_replacement_context(data, original, 1)
    assert context["reviewed_item"] == original
    assert context["authorized_diagnosis"] == authorization
    assert list(context["cited_content_evidence"]) == ["EXPERT:1:F1"]
    assert context["cited_content_evidence"]["EXPERT:1:F1"]["quote"] == "old D"
    assert context["source_analysis_round"] == 2
    context["reviewed_item"]["scenario"] = "Changed local copy"
    assert original["scenario"] == "Original scenario"
    with pytest.raises(RepairStopped, match="scope"):
        _content_replacement_context(data, original, 2)


def test_iteration_snapshot_records_all_candidates_and_actual_selected_form(monkeypatch, tmp_path):
    state = measured_state(count=3)
    state["item_statistics"]["item-1"].update(
        iteration_metric_status="frozen", measurement_skipped=True,
        frozen_from_analysis_round=1,
    )
    manifest = tmp_path / "responses" / "bank-v1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    # Atomic JSON utility is used to prepare the offline fixture.
    from sjt_system.runtime.io import write_json_atomic
    write_json_atomic(manifest, {})
    state["virtual_response_data_ref"] = str(manifest)

    async def optimize(*_args, **_kwargs):
        return {"status": "validated", "selected_item_ids": ["item-2"]}
    monkeypatch.setattr(executor, "optimize_test_form_with_agent", optimize)
    monkeypatch.setattr(executor, "build_provisional_form_metrics", lambda _state, ids: {
        "selected_item_ids": ids, "facet_metrics": []})
    record = asyncio.run(executor._build_provisional_iteration_record(state, state["item_pool"]))
    saved = json.loads(open(record["iteration_metrics_artifact"], encoding="utf-8").read())
    assert len(saved["item_metrics"]) == 3
    assert saved["selected_item_ids"] == ["item-2"]
    assert saved["item_metrics"]["item-1"]["selected_for_form"] is False
    assert saved["item_metrics"]["item-2"]["selected_for_form"] is True
    assert saved["item_snapshots"]["item-1"]["version"] == 2
    assert saved["item_statistics_snapshot"] == state["item_statistics"]
    assert saved["form_metrics"]["selected_item_ids"] == ["item-2"]
    annotated = {**record, "item_dispositions": {"item-1": {
        "status": "qualified_locked",
        "item_version": 2,
        "retention_basis": "four_iteration_gates_passed",
    }}}
    persist_iteration_metrics_snapshot(state, annotated)
    refreshed = json.loads(open(record["iteration_metrics_artifact"], encoding="utf-8").read())
    assert refreshed["item_metrics"]["item-1"]["frozen_qualified"] is True
    altered = deepcopy(record)
    altered["candidate_item_metrics"]["item-1"]["qualified"] = True
    with pytest.raises(ValueError, match="不可覆盖"):
        persist_iteration_metrics_snapshot(state, altered)
    original = json.loads(open(record["iteration_metrics_artifact"], encoding="utf-8").read())
    assert original["item_metrics"]["item-1"]["qualified"] is False


def test_three_retests_record_four_measurements_and_versions(monkeypatch):
    state = measured_state(count=1)
    for measurement in range(1, 4):
        fake = FakeInvoke()
        async def invoke(kind, payload, fake=fake):
            if kind == "expert_review":
                from test_virtual_content_review import expert_report
                finding = expert_report()
                finding["findings"][0]["quote"] = state["item_pool"][0]["response_options"][0]["text"]
                return finding
            if kind == "content_diagnosis":
                from test_virtual_content_review import diagnosis
                result = diagnosis()
                result["repair_tasks"][0]["quote"] = state["item_pool"][0]["response_options"][0]["text"]
                return result
            if kind == "content_edit":
                return {"reason": "mock evidence-authorized edit", "response_options": [
                    {"option_id": "D", "text": f"revised D round {measurement}"}]}
            return await fake(kind, payload)
        monkeypatch.setattr(review, "make_model_invoker", lambda _state: (invoke, {}))
        state["items_to_revise"] = [entry(state, state["item_pool"][0], revision_round=measurement)]
        state.update(asyncio.run(batch.execute_virtual_content_review_batch(state))["state_update"])
        assert state["item_pool"][0]["version"] == 2 + measurement
        state["frozen_item_bank"] = deepcopy(state["item_pool"])
        state["item_statistics"] = {"item-1": stat(False)}
        state["psychometric_analysis_round"] = measurement + 1
        state["psychometric_iteration_history"].append(measurement_record(state))
    update = select(state, monkeypatch)
    assert len(update["items_to_revise"]) == 1
    assert update["items_to_revise"][0]["item_id"] == "item-1"
    assert update["items_to_revise"][0]["deferred_replacement_only"] is True
    assert update["items_to_revise"][0]["queue_status"] == "deferred_replenishment"
    assert len(update["psychometric_iteration_history"]) == 4
    assert [row["item_snapshots"]["item-1"]["version"] for row in update["psychometric_iteration_history"]] == [2, 3, 4, 5]
    assert state["psychometric_repair_rounds"]["item-1"] == 3
    assert [row["round_label"] for row in update["psychometric_iteration_history"]] == [
        review.round_label(number) for number in range(1, 5)]


def test_version_guard_rejects_same_version_changed_text():
    state = measured_state(count=1)
    state["item_pool"][0]["scenario"] = "Changed without new version"
    with pytest.raises(ValueError, match="必须升级版本"):
        bank.build_item_bank_freeze_update(state)


def test_snapshot_does_not_turn_retention_into_qualification():
    state = measured_state(count=1)
    record = measurement_record(state)
    record["item_dispositions"] = {"item-1": {"status": "qualified_locked"}}
    snapshot = build_iteration_metrics_snapshot(record, run_id=state["run_id"])
    assert snapshot["item_metrics"]["item-1"]["qualified"] is False
    assert snapshot["item_dispositions"]["item-1"]["status"] == "qualified_locked"


def test_snapshot_separates_frozen_qualification_from_current_monitoring():
    state = measured_state(count=1)
    record = measurement_record(state)
    record["candidate_item_metrics"]["item-1"].update(
        iteration_metric_status="frozen", measurement_skipped=True,
        frozen_from_analysis_round=1,
    )
    record["item_dispositions"] = {
        "item-1": {
            "status": "qualified_locked",
            "item_version": 2,
            "retention_basis": "four_iteration_gates_passed",
        }
    }
    snapshot = build_iteration_metrics_snapshot(record, run_id=state["run_id"])
    row = snapshot["item_metrics"]["item-1"]
    assert row["qualified"] is False
    assert row["frozen_qualified"] is True
    assert snapshot["current_round_qualified_count"] == 0
    assert snapshot["frozen_qualified_count"] == 1


def test_only_four_gate_qualified_locked_items_freeze_item_metrics():
    qualification = {
        "qualified": True,
        "citc_pass": True,
        "target_rho_pass": None,
        "same_domain_vts_pass": None,
        "cross_domain_vts_pass": None,
        "target_hedges_g_pass": True,
        "target_ipip_spearman_rho_pass": True,
        "discriminant_delta_min_pass": True,
    }
    qualified = {
        "qualification": qualification,
        "virtual_screening_metrics": {
            "marker": "first-pass-values",
            "facet_citc": {"r": 0.51},
        },
    }
    plateau_retained = {
        "qualification": qualification,
        "virtual_screening_metrics": {
            "marker": "must-be-measured-again",
            "facet_citc": {"r": 0.12},
        },
    }
    state = {
        "psychometric_analysis_round": 2,
        "item_statistics": {
            "qualified": deepcopy(qualified),
            "plateau": deepcopy(plateau_retained),
        },
        "locked_retained_item_versions": {"qualified": 1, "plateau": 1},
        "item_final_dispositions": {
            "qualified": {
                "status": "qualified_locked",
                "retention_basis": "four_iteration_gates_passed",
                "qualification_analysis_round": 1,
                "qualification_snapshot": deepcopy(qualified),
            },
            "plateau": {
                "status": "qualified_locked",
                "retention_basis": "plateau_retained",
                "qualification_snapshot": deepcopy(plateau_retained),
            },
        },
    }
    items = {
        "qualified": {"item_id": "qualified", "version": 1},
        "plateau": {"item_id": "plateau", "version": 1},
    }

    snapshots = _qualified_item_metric_snapshots(
        state,
        item_order=["qualified", "plateau"],
        items=items,
    )

    assert list(snapshots) == ["qualified"]
    assert snapshots["qualified"]["virtual_metrics"]["marker"] == "first-pass-values"
    assert snapshots["qualified"]["source_analysis_round"] == 1


def test_frozen_item_still_enters_provisional_form_scoring(monkeypatch):
    state = measured_state(count=2)
    state["item_final_dispositions"] = {
        "item-1": {
            "status": "qualified_locked",
            "retention_basis": "four_iteration_gates_passed",
        }
    }
    state["item_statistics"]["item-1"]["iteration_metric_status"] = "frozen"
    state["item_statistics"]["item-1"]["measurement_skipped"] = True
    state["item_statistics"]["item-2"]["iteration_metric_status"] = "measured"
    selected_for_form = ["item-1", "item-2"]
    captured: dict[str, list[str]] = {}

    async def optimize(*_args, **_kwargs):
        return {"status": "validated", "selected_item_ids": selected_for_form}

    def form_metrics(_state, item_ids):
        captured["selected_item_ids"] = list(item_ids)
        return {"selected_item_ids": list(item_ids), "facet_metrics": []}

    monkeypatch.setattr(executor, "optimize_test_form_with_agent", optimize)
    monkeypatch.setattr(executor, "build_provisional_form_metrics", form_metrics)

    record = asyncio.run(
        executor._build_provisional_iteration_record(state, state["frozen_item_bank"])
    )

    assert captured["selected_item_ids"] == selected_for_form
    assert record["form_item_ids"] == selected_for_form
    assert record["candidate_item_metrics"]["item-1"]["measurement_skipped"] is True
    assert record["candidate_item_metrics"]["item-2"]["iteration_metric_status"] == "measured"


def test_frozen_item_is_retained_in_round_form_metrics_snapshot():
    state = measured_state(count=2)
    record = measurement_record(state)
    record["candidate_item_metrics"]["item-1"].update(
        qualified=True,
        iteration_metric_status="frozen",
        measurement_skipped=True,
        frozen_from_analysis_round=1,
    )
    record["form_item_ids"] = ["item-1"]
    record["form_metrics"] = {
        "selected_item_ids": ["item-1"],
        "facet_metrics": [],
    }

    snapshot = build_iteration_metrics_snapshot(record, run_id=state["run_id"])

    assert snapshot["selected_item_ids"] == ["item-1"]
    assert snapshot["form_metrics"]["selected_item_ids"] == ["item-1"]
    assert snapshot["item_metrics"]["item-1"]["iteration_metric_status"] == "frozen"
    assert snapshot["item_metrics"]["item-1"]["measurement_skipped"] is True
    assert snapshot["frozen_metric_count"] == 1
    assert snapshot["current_round_qualified_count"] == 0
