"""Offline acceptance contracts for the isolated A/B/C experiment."""
import asyncio
from copy import deepcopy
import json

import pytest


def item(number=1, cell=None):
    return {
        "item_id": f"item-{number}", "version": 1,
        "blueprint_cell_id": cell or f"cell-{number}",
        "target_dimension_id": "extraversion_gregariousness",
        "scenario": f"情境{number}", "response_instruction": "你会怎么做？",
        "response_options": [{"option_id": k, "text": f"行为{k}"} for k in "ABCD"],
        "scoring_key": dict(zip("ABCD", range(1, 5))),
    }


def a_output(items):
    return {"items": items, "design_rationales": [
        {"item_number": i, "scenario_rationale": "设计依据：群体规模的自由选择。",
         "option_rationales": {o: f"设计依据：{o}反映相应程度的群体偏好。" for o in "ABCD"}}
        for i in range(1, len(items) + 1)]}


def test_participants_are_disjoint_but_keep_matched_contract():
    from experiments.system_comparison.config import ExperimentConfig, make_participants
    from sjt_system.evaluation.respondents import matched_condition_sample_is_current
    config = ExperimentConfig(sample_size=30)
    dev = make_participants(config, "development")
    ev = make_participants(config, "evaluation")
    assert matched_condition_sample_is_current(dev["config"], dev["respondents"])
    assert matched_condition_sample_is_current(ev["config"], ev["respondents"])
    assert not ({r["respondent_id"] for r in dev["respondents"]} &
                {r["respondent_id"] for r in ev["respondents"]})
    assert dev["respondents"][0]["score_values"] != ev["respondents"][0]["score_values"]


def test_experiment_models_resolve_authoring_virtual_and_evaluation_roles(monkeypatch):
    from experiments.system_comparison.config import ExperimentConfig

    monkeypatch.delenv("VIRTUAL_RESPONDENT_MODEL_ID", raising=False)
    config = ExperimentConfig(
        sample_size=30,
        model_id="author-model",
        virtual_respondent_model_id="virtual-model",
    ).resolved()

    assert config.model_id == "author-model"
    assert config.virtual_respondent_model_id == "virtual-model"
    assert config.evaluation_model_id == "virtual-model"


def test_store_freezes_versions_and_rejects_overwrite(tmp_path):
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.config import ExperimentConfig
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    items = [item()]
    store.save_form("B/round_01", items)
    items[0]["scenario"] = "修改后"
    store.save_form("C/round_02", items)
    assert store.read("B/round_01/form.json")["items"][0]["scenario"] == "情境1"
    with pytest.raises(ValueError, match="冻结"):
        store.save_form("B/round_01", items)
    reopened = ExperimentStore(store.root)
    assert reopened.config.sample_size == 30
    assert (store.root / "participants/evaluation.json").exists()


def test_store_can_reopen_config_created_before_virtual_model_field(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig, fingerprint
    from experiments.system_comparison.storage import ExperimentStore

    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    old_config = store.read("config.json")
    old_config.pop("virtual_respondent_model_id")
    store.write("config.json", old_config)
    progress = store.read("progress.json")
    progress["config_hash"] = fingerprint(old_config)
    store.write("progress.json", progress)

    reopened = ExperimentStore(store.root)
    assert reopened.config.virtual_respondent_model_id is None


def test_a_validates_format_and_b_cannot_see_metrics():
    from experiments.system_comparison.generation import theory_payload, validate_a_form, validate_b_selection
    state = {"blueprint": {"cells": [{"cell_id": "cell-1", "planned_retention_count": 1}]},
             "construct_profile": {"facets": []}, "item_statistics": {"SECRET": 10},
             "test_statistics": {"SECRET": 20}, "virtual_respondents": ["SECRET"],
             "frozen_item_bank": [item(), item(2, "cell-1")]}
    payload = theory_payload(state)
    assert "SECRET" not in json.dumps(payload)
    assert len(validate_b_selection({"selected_item_ids": ["item-1"]}, state)) == 1
    with pytest.raises(ValueError):
        validate_b_selection({"selected_item_ids": ["invented"]}, state)
    assert len(validate_a_form({"items": [item()]}, count=1, facet="extraversion_gregariousness")) == 1
    bad = item()
    bad["scoring_key"]["A"] = True
    with pytest.raises(ValueError):
        validate_a_form({"items": [bad]}, count=1, facet="extraversion_gregariousness")


def test_persistent_graph_resumes_interrupt_without_repeating_model(tmp_path):
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt, Command
    from experiments.system_comparison.checkpoints import DiskSaver
    class State(TypedDict):
        text: str
    calls = []
    def generate(state):
        calls.append(1)
        return {"text": "generated"}
    def approve(state):
        answer = interrupt({"text": state["text"]})
        return {"text": answer}
    def build():
        g = StateGraph(State)
        g.add_node("generate", generate)
        g.add_node("approve", approve)
        g.add_edge(START, "generate")
        g.add_edge("generate", "approve")
        g.add_edge("approve", END)
        return g
    cfg = {"configurable": {"thread_id": "test"}}
    with DiskSaver(tmp_path / "graph.sqlite") as saver:
        build().compile(checkpointer=saver).invoke({"text": ""}, cfg)
    with DiskSaver(tmp_path / "graph.sqlite") as saver:
        result = build().compile(checkpointer=saver).invoke(Command(resume="accepted"), cfg)
    assert result["text"] == "accepted"
    assert len(calls) == 1


def test_evaluation_state_has_only_current_form_and_no_development_sources(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.evaluation import evaluation_state
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    store.save_form("A/round_01", [item()])
    state = evaluation_state(store, "A/round_01")
    assert len(state["frozen_item_bank"]) == 1
    assert not state.get("previous_virtual_response_data_ref")
    assert not state.get("locked_retained_item_versions")
    assert not state.get("psychometric_repair_history")
    assert "evaluation" in state["virtual_respondents"][0]["respondent_id"]


def test_missing_gates_are_not_zero_or_pass(tmp_path):
    from experiments.system_comparison.evaluation import flatten_items
    state = {"frozen_item_bank": [item()], "item_statistics": {"item-1": {
        "quality_evaluation": {}, "qualification": {}}}}
    rows = flatten_items(state, "A/round_01")
    assert len(rows) == 1
    assert rows[0]["citc"] is None
    assert rows[0]["qualified"] is False
    assert rows[0]["citc_estimable"] is False


def test_runtime_output_scope_does_not_change_default_paths(tmp_path):
    from sjt_system.runtime.output_paths import output_scope, scoped_output
    default = tmp_path / "old"
    assert scoped_output("responses", default) == default
    with output_scope(tmp_path / "experiment"):
        assert scoped_output("responses", default) == tmp_path / "experiment/responses"
    assert scoped_output("responses", default) == default


def test_report_can_be_rebuilt_without_models(tmp_path, monkeypatch):
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.reporting import write_report
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    store.save_form("A/round_01", [item()])
    store.write("A/round_01/evaluation/metrics.json", {
        "status": "complete", "item_count": 1, "qualified_item_count": 0,
        "quality": {"target_recovery_raw": -0.1, "construct_selectivity": None,
                    "candidate_form_quality": None}})
    write_report(store)
    assert (store.root / "summary/item_comparison.csv").exists()
    report = (store.root / "summary/report.html").read_text(encoding="utf-8")
    assert "Cronbach α" in report
    assert "Q" not in report
    import os
    import subprocess
    import sys
    environment = dict(os.environ)
    environment["API_KEY"] = ""  # get_model() would fail; report must not call it.
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "experiments.system_comparison",
                             "report", "--experiment", str(store.root)],
                            env=environment, capture_output=True, text=True,
                            encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_report_charts_abc_aggregate_item_metric_changes_using_final_c_only(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.reporting import write_report
    from experiments.system_comparison.storage import ExperimentStore

    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    forms = {
        "A/round_01": (.10, False),
        "B/round_01": (.20, True),
        "C/round_01": (.99, True),
        "C/round_02": (.40, True),
    }
    for key, (base, qualified) in forms.items():
        store.save_form(key, [item()])
        store.write(f"{key}/evaluation/metrics.json", {
            "status": "complete", "item_count": 1,
            "qualified_item_count": int(qualified),
            "qualification_rate": float(qualified),
            "quality": {
                "candidate_form_quality": base + .01,
                "alpha_gate": {"observed": .90, "passed": True},
                "stability_gate": {"observed": .90, "passed": True},
                "ipip_target_facet_spearman_rho": base,
                "ipip_discriminant_delta_min": base - .05,
                "ipip_target_known_groups_hedges_g": base + .01,
            },
        })
        method, round_name = key.split("/")
        store.write(f"{key}/evaluation/items.json", [{
            "method": method, "round": round_name, "item_id": "item-1",
            "citc": base, "target_rho": base + .1,
            "same_domain_vts": base + .2, "cross_domain_vts": base + .3,
            "qualified": qualified,
        }])
    store.write("C/final.json", {"source_round": "C/round_02"})
    store.write("reference/neo_ffi/metrics.json", {
        "status": "complete",
        "cross_model_exploration": True,
        "neo_ffi_model_id": "new-model",
        "sjt_model_ids": {"A": "old-model", "B": "old-model", "C": "old-model"},
        "correlations": [
            {"method": "A", "neo_domain": "E", "evidence_type": "convergent", "spearman_rho": .3, "n": 100},
            {"method": "A", "neo_domain": "N", "evidence_type": "discriminant", "spearman_rho": .1, "n": 100},
        ],
    })

    write_report(store)
    html = store.path("summary/report.html").read_text(encoding="utf-8")

    assert "A→B→C单题指标总体变化" in html
    assert "方法层面的汇总比较，不是同一道题的纵向追踪" in html
    assert "C 均值: 0.400" in html
    assert "C 中位数: 0.400" in html
    assert "C 均值: 0.990" not in html
    assert "四门槛通过率" in html
    assert "A→B→C整卷指标" in html
    assert "目标Hedges’ g" in html
    assert "Δmin区分效度" in html
    assert "Cronbach α" in html
    assert "整卷总体质量Q" not in html
    assert "I_g" not in html
    assert "NEO-FFI汇聚与区分相关" in html
    assert "跨模型探索" in html
    assert "new-model" in html


def test_neo_reference_correlations_are_limited_to_convergent_and_discriminant():
    from experiments.system_comparison.neo_reference import compute_neo_correlations

    subject_ids = [f"s{i}" for i in range(1, 6)]
    neo_scores = [
        {
            "matched_subject_id": subject_id,
            "N": 60 - index * 10,
            "E": index * 10,
            "O": 20,
            "A": 10 + index,
            "C": 30 - index,
        }
        for index, subject_id in enumerate(subject_ids, 1)
    ]
    form_scores = {
        "A": dict(zip(subject_ids, [1, 2, 3, 4, 5])),
        "B": dict(zip(subject_ids, [5, 4, 3, 2, 1])),
        "C": dict(zip(subject_ids, [1, 3, 2, 5, 4])),
    }

    rows = compute_neo_correlations(neo_scores, form_scores)

    assert len(rows) == 15
    assert {row["evidence_type"] for row in rows} == {"convergent", "discriminant"}
    assert next(row for row in rows if row["method"] == "A" and row["neo_domain"] == "E")["spearman_rho"] == pytest.approx(1.0)
    assert next(row for row in rows if row["method"] == "A" and row["neo_domain"] == "N")["spearman_rho"] == pytest.approx(-1.0)
    unavailable = next(row for row in rows if row["method"] == "A" and row["neo_domain"] == "O")
    assert unavailable["spearman_rho"] is None
    assert unavailable["unavailable_reason"] == "NEO-FFI维度得分无变异"
    assert all(row["n"] == 5 for row in rows)


class RespondentModel:
    """No network: deterministic option selection from the displayed persona."""
    model_name = "offline-model"

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda
        import re
        async def respond(messages):
            assert schema.__name__ == "SJTSelectionOutput", "实验不应调用参照问卷"
            self.calls += 1
            system, user = messages[0][1], messages[1][1]
            score = float(re.search(r"\| ([0-9.]+)\n\[/PERSONALITY", system).group(1))
            original = "ABCD"[min(3, max(0, int((score - 30) // 15)))]
            display = re.search(r"([ABCD])\. 行为" + original, user).group(1)
            return {"selected_option_id": display}
        return RunnableLambda(respond)


def test_real_evaluator_exports_four_gates_and_reuses_only_raw_evaluation(tmp_path, monkeypatch):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.evaluation import evaluate_form
    monkeypatch.setenv("STRUCTURED_OUTPUT_METHOD", "json_schema")
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=2, sample_size=30))
    model = RespondentModel()
    store.save_form("B/round_01", [item(1), item(2)])
    store.save_form("C/baseline", [item(1), item(2)])
    first = asyncio.run(evaluate_form(store, "B/round_01", model))
    assert first["item_count"] == 2
    assert model.calls == 30 * 2 * 4  # 3 arms + target retest; no references
    import pandas as pd
    matrix = pd.read_csv(store.path("B/round_01/evaluation/score_matrices/target.csv"))
    assert matrix.shape == (30, 3)  # matched_subject_id plus two scored items
    initial_calls = model.calls
    asyncio.run(evaluate_form(store, "B/round_01", model))
    alias = asyncio.run(evaluate_form(store, "C/baseline", model))
    assert alias["reused_from"] == "B/round_01"
    assert model.calls == initial_calls
    for row in store.read("C/baseline/evaluation/items.json"):
        assert row["method"] == "C"
        for gate in ("citc", "target_rho", "same_domain_vts", "cross_domain_vts"):
            assert gate + "_threshold" in row
            assert gate + "_passes" in row
    # Same item, different form => reuse its responses but recompute form-scoped CITC.
    store.save_form("C/round_01", [item(1)])
    one = asyncio.run(evaluate_form(store, "C/round_01", model))
    assert model.calls == initial_calls
    assert store.read("C/round_01/evaluation/items.json")[0]["citc"] is None
    assert one["qualified_item_count"] == 0
    changed = item(1)
    changed["version"] = 2
    changed["scenario"] = "new scenario"
    store.save_form("C/round_02", [changed])
    asyncio.run(evaluate_form(store, "C/round_02", model))
    assert model.calls == initial_calls + 30 * 4


def test_single_experiment_pause_resume_and_full_report(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.runner import ExperimentRunner
    from experiments.system_comparison.workflow import ExperimentPaused, capture_rounds
    from sjt_system.authoring.bank import build_item_bank_freeze_update
    monkeypatch.setenv("STRUCTURED_OUTPUT_METHOD", "json_schema")
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=2, sample_size=30))
    class FormModel:
        def __init__(self, payload):
            self.payload, self.calls = payload, 0
        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))
    a = FormModel(a_output([item(1), item(2)]))
    b = FormModel({"selected_item_ids": ["item-1", "item-2"]})
    calls = []
    async def driver(store, phase, state):
        calls.append(phase)
        if phase == "shared":
            state["blueprint"] = {"cells": [{"cell_id": f"cell-{i}", "planned_retention_count": 1} for i in (1, 2)]}
            state["item_pool"] = [item(1), item(2), item(3, "cell-1"), item(4, "cell-2")]
            return state
        if calls.count("C") == 1:
            raise ExperimentPaused("待人工确认")
        state["psychometric_analysis_round"] = 1
        state["psychometric_iteration_history"] = [{"analysis_round": 1,
            "form_item_ids": ["item-1", "item-2"], "form_status": "complete",
            "form_metrics": {"reliability": {"virtual_test_retest_icc": .9},
                             "validity": {"target_recovery": {"cross_validated_r2": .8},
                                          "construct_selectivity": {"value": .6}}}}]
        capture_rounds(store, state)
        return state
    runner = ExperimentRunner(store, a_model=a, b_model=b, evaluation_model=RespondentModel(), workflow_driver=driver)
    asyncio.run(runner.run())
    assert store.read("progress.json")["status"] == "paused"
    resumed = ExperimentRunner(ExperimentStore(store.root), a_model=a, b_model=b,
                               evaluation_model=RespondentModel(), workflow_driver=driver)
    asyncio.run(resumed.run())
    assert store.read("progress.json")["status"] == "completed"
    assert a.calls == b.calls == 1
    assert calls.count("shared") == 1
    assert len(store.read("C/round_01/evaluation/items.json")) == 2
    assert (store.root / "summary/report.html").exists()


def test_real_workflow_driver_resumes_static_and_human_interrupts(tmp_path):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from sjt_system.state import PSJTState
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import drive_workflow, ExperimentPaused
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    calls = []
    def factory(checkpointer, interrupt_before):
        graph = StateGraph(PSJTState)
        def execute(state):
            calls.append("generated")
            return {"status": "running"}
        def approve(state):
            answer = interrupt({"type": "approve", "content": "already generated"})
            assert answer["decision"] == "approve"
            return {"status": "completed"}
        graph.add_node("execute", execute)
        graph.add_node("approve", approve)
        graph.add_edge(START, "execute")
        graph.add_edge("execute", "approve")
        graph.add_edge("approve", END)
        return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    state = {"run_id": "driver-test", "status": "running"}
    with pytest.raises(ExperimentPaused):
        asyncio.run(drive_workflow(store, "C", state, graph_factory=factory,
                                 decision_provider=lambda p: {"decision": "stop"}))
    result = asyncio.run(drive_workflow(ExperimentStore(store.root), "C", state,
                         graph_factory=factory, decision_provider=lambda p: {"decision": "approve"}))
    assert result["status"] == "completed"
    assert calls == ["generated"]
    assert len(store.read("C/human_decisions.json")) == 2


def test_round_capture_recovers_partial_save_and_freezes_cost_cutoff(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import capture_rounds
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    record = {"analysis_round": 1, "form_item_ids": ["item-1"], "form_status": "complete"}
    state = {"psychometric_analysis_round": 1, "psychometric_iteration_history": [record],
             "frozen_item_bank": [item()]}
    store.save_form("C/round_01", [item()], development_record=record)
    ledger = store.path("C/telemetry/calls.jsonl")
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"total_tokens": 10, "iteration": 1}) + "\n", encoding="utf-8")
    capture_rounds(store, state)
    assert store.read("C/round_01/development/snapshot_complete.json")["complete"]
    assert store.read("C/round_01/cost.json")["cumulative_c_cost"]["total_tokens"] == 10
    ledger.write_text(ledger.read_text() + json.dumps({"total_tokens": 50, "iteration": 1}) + "\n", encoding="utf-8")
    capture_rounds(store, state)
    assert store.read("C/round_01/cost.json")["cumulative_c_cost"]["total_tokens"] == 10


def test_a_structure_retries_are_bounded_and_saved(tmp_path):
    from langchain_core.messages import AIMessage
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.generation import generate_a
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    class BadModel:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content='{"items": []}')
    model = BadModel()
    for _ in range(2):
        with pytest.raises(ValueError, match="3次"):
            asyncio.run(generate_a(store, model))
    assert model.calls == 3
    assert store.read("A/round_01/attempt_03.json")["error"]


def test_config_edits_are_rejected_on_resume(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    changed = store.read("config.json")
    changed["seed"] += 1
    store.write("config.json", changed)
    with pytest.raises(ValueError, match="配置已改变"):
        ExperimentStore(store.root)


def test_alias_does_not_publish_complete_marker_before_reanalysis(tmp_path, monkeypatch):
    import experiments.system_comparison.evaluation as evaluation
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    store.save_form("B/round_01", [item()])
    store.save_form("C/baseline", [item()])
    model = RespondentModel()
    asyncio.run(evaluation.evaluate_form(store, "B/round_01", model))
    analyze = evaluation.run_psychometric_analysis
    def fail(state):
        raise RuntimeError("interrupted reanalysis")
    monkeypatch.setattr(evaluation, "run_psychometric_analysis", fail)
    with pytest.raises(RuntimeError):
        asyncio.run(evaluation.evaluate_form(store, "C/baseline", model))
    assert store.read("C/baseline/evaluation/metrics.json", {}).get("status") != "complete"
    monkeypatch.setattr(evaluation, "run_psychometric_analysis", analyze)
    asyncio.run(evaluation.evaluate_form(store, "C/baseline", model))
    assert model.calls == 120
    assert store.read("C/baseline/evaluation/items.json")[0]["method"] == "C"


def test_edited_participants_are_not_silently_accepted(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    sample = store.read("participants/development.json")
    sample["respondents"][0]["respondent_id"] = "edited"
    store.write("participants/development.json", sample)
    with pytest.raises(ValueError, match="被试"):
        ExperimentStore(store.root)


def test_scoped_telemetry_records_tokens_and_missing_ledger_is_unknown(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.reporting import cost_summary
    from sjt_system.runtime.output_paths import output_scope
    from sjt_system.runtime.telemetry import TelemetryHandler, run_context, job_context
    config = ExperimentConfig(sample_size=30, input_price_per_million=1,
                               cached_input_price_per_million=.1, output_price_per_million=3)
    store = ExperimentStore.create(tmp_path, config)
    with output_scope(store.path("A/round_01")), run_context("test"), job_context("generate", iteration=1):
        TelemetryHandler()._finish("call-1", status="success", error_kind=None, llm_output={
            "model_name": store.config.model_id, "token_usage": {
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 40}}})
    cost = cost_summary(store, "A/round_01")
    assert cost["total_tokens"] == 120
    assert cost["fee"] == pytest.approx(.000124)
    assert cost_summary(store, "shared")["total_tokens"] is None


def test_interrupted_evaluation_resumes_only_missing_responses(tmp_path, monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.evaluation import evaluate_form
    monkeypatch.setenv("STRUCTURED_OUTPUT_METHOD", "json_schema")
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    store.save_form("A/round_01", [item()])
    class InterruptedModel(RespondentModel):
        enabled = True
        fail_prompt = None
        def with_structured_output(self, schema, **kwargs):
            inner = super().with_structured_output(schema, **kwargs)
            async def respond(messages):
                if self.fail_prompt is None:
                    self.fail_prompt = messages[0][1]
                if self.enabled and messages[0][1] == self.fail_prompt:
                    raise RuntimeError("offline simulated request failure")
                return await inner.ainvoke(messages)
            return RunnableLambda(respond)
    model = InterruptedModel()
    with pytest.raises(Exception):
        asyncio.run(evaluate_form(store, "A/round_01", model))
    assert 0 < model.calls < 120
    assert store.read("A/round_01/evaluation/metrics.json", {}).get("status") != "complete"
    model.enabled = False
    asyncio.run(evaluate_form(ExperimentStore(store.root), "A/round_01", model))
    assert model.calls == 120  # each successful response performed exactly once


def test_final_uses_original_historical_content_not_latest_id_version(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import select_final
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    def record(r2):
        return {"form_metrics": {"reliability": {"virtual_test_retest_icc": .9},
                "validity": {"target_recovery": {"cross_validated_r2": r2},
                             "construct_selectivity": {"value": .6}}}}
    store.save_form("C/round_01", [item()], development_record=record(.9))
    changed = item()
    changed.update(version=2, scenario="worse revision")
    store.save_form("C/round_02", [changed], development_record=record(.4))
    # Much higher independent evaluation cannot change the development selection.
    store.write("C/round_02/evaluation/metrics.json", {"quality": {"candidate_form_quality": 1}})
    assert select_final(store) == "C/round_01"
    assert store.read("C/final.json")["items"][0]["version"] == 1


def test_stopping_boundaries_and_incomplete_ledger(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import stopping_reason
    from experiments.system_comparison.reporting import write_report, cost_summary
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    assert stopping_reason(store, "shared", {"route": {"next_action": "simulate_responses"}}, ("execute",)) == "shared_bank_ready"
    assert stopping_reason(store, "C", {"psychometric_analysis_round": 3}, ("execute",)) is None
    assert stopping_reason(store, "C", {"psychometric_analysis_round": 4}, ("execute",)) == "round_limit"
    assert stopping_reason(store, "C", {"psychometric_plateau_status": {"reached": True}}, ()) == "plateau"
    path = store.path("C/telemetry/broken.jsonl")
    path.parent.mkdir(parents=True)
    path.write_text('{"total_tokens": 100}\n{"partial":', encoding="utf-8")
    assert cost_summary(store, "C")["total_tokens"] is None
    write_report(store)
    assert store.read("C/cost.json")["ledger_errors"] == 1


def test_workflow_model_scope_rebinds_eager_agents_and_restores(tmp_path, monkeypatch):
    import sjt_system.agent.agent_factory as factory
    import sjt_system.workflow.executor as executor
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.models import workflow_model_scope
    store = ExperimentStore.create(
        tmp_path,
        ExperimentConfig(
            sample_size=30,
            model_id="experiment-model",
            virtual_respondent_model_id="virtual-model",
        ),
    )
    original = executor.item_writer_agent
    monkeypatch.setattr(factory, "create_agent", lambda *a, **kw: kw)
    with workflow_model_scope(store):
        assert executor.item_writer_agent["model_id"] == "experiment-model"
        assert executor.AGENT_MAP["generate_item"] is executor.item_writer_agent
        assert executor.psychometric_item_repair_agent["model_id"] == "experiment-model"
        assert store.read("model_roles.json")["roles"]["development_virtual_respondent"]["model_id"] == "virtual-model"
    assert executor.item_writer_agent is original


def test_c_development_state_uses_virtual_respondent_model(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import initial_workflow_state

    store = ExperimentStore.create(
        tmp_path,
        ExperimentConfig(
            sample_size=30,
            model_id="author-model",
            virtual_respondent_model_id="virtual-model",
            evaluation_model_id="evaluation-model",
        ),
    )

    state = initial_workflow_state(store)
    assert state["virtual_sample_config"]["model_id"] == "virtual-model"
    assert state["virtual_sample_config"]["experiment_protocol"]["sample"] == "development"


def test_independent_evaluation_state_uses_evaluation_model(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.evaluation import evaluation_state
    from experiments.system_comparison.storage import ExperimentStore

    store = ExperimentStore.create(
        tmp_path,
        ExperimentConfig(
            sample_size=30,
            model_id="author-model",
            virtual_respondent_model_id="virtual-model",
            evaluation_model_id="evaluation-model",
        ),
    )
    store.save_form("A/round_01", [item()])

    state = evaluation_state(store, "A/round_01")
    assert state["virtual_sample_config"]["model_id"] == "evaluation-model"


def test_resume_retries_only_failed_execute_node(tmp_path):
    from langgraph.graph import StateGraph, START, END
    from sjt_system.state import PSJTState
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import drive_workflow
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    calls = []
    def factory(checkpointer, interrupt_before):
        graph = StateGraph(PSJTState)
        def router(state):
            calls.append("router")
            return {"route": {"next_action": "simulate_responses"}}
        def execute(state):
            calls.append("execute")
            if calls.count("execute") == 1:
                return {"status": "failed", "errors": [{"action": "simulate_responses", "message": "temporary failure"}],
                        "execution_history": [{"node": "execute", "event_type": "failed"}]}
            return {"status": "completed"}
        graph.add_node("router", router)
        graph.add_node("execute", execute)
        graph.add_edge(START, "router")
        graph.add_edge("router", "execute")
        graph.add_edge("execute", END)
        return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    state = {"run_id": "retry-test", "status": "running"}
    with pytest.raises(RuntimeError):
        asyncio.run(drive_workflow(store, "C", state, graph_factory=factory))
    result = asyncio.run(drive_workflow(ExperimentStore(store.root), "C", state, graph_factory=factory))
    assert result["status"] == "completed"
    assert calls == ["router", "execute", "execute"]
    assert store.read("C/recovery_events.json")[0]["action"] == "simulate_responses"


def test_recovery_with_real_execution_node(tmp_path, monkeypatch):
    from langgraph.graph import StateGraph, START, END
    from sjt_system.state import PSJTState, create_initial_state
    import sjt_system.workflow.execution_node as execution
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.workflow import drive_workflow
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    attempts = []
    async def request(route, state):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary service failure")
        return {"state_update": {}, "summary": "offline completed"}
    monkeypatch.setattr(execution, "execute_agent", request)
    def factory(checkpointer, interrupt_before):
        graph = StateGraph(PSJTState)
        graph.add_node("router", lambda s: {"route": {"next_action": "simulate_responses"}})
        graph.add_node("execute", execution.execute_node)
        graph.add_node("finish", lambda s: {"status": "completed"})
        graph.add_edge(START, "router")
        graph.add_edge("router", "execute")
        graph.add_conditional_edges("execute", lambda s: END if s["status"] == "failed" else "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    state = create_initial_state("offline recovery")
    with pytest.raises(RuntimeError):
        asyncio.run(drive_workflow(store, "C", state, graph_factory=factory))
    result = asyncio.run(drive_workflow(ExperimentStore(store.root), "C", state, graph_factory=factory))
    assert result["status"] == "completed"
    assert len(attempts) == 2


def test_a_prompt_fills_construct_without_examples():
    from experiments.system_comparison.generation import build_a_prompt
    from experiments.system_comparison.config import ExperimentConfig, profile_for
    config = ExperimentConfig(final_item_count=7, target_population="职场新人")
    profile = profile_for(config)
    prompt = build_a_prompt(config, profile)
    assert "7 道" in prompt and "职场新人" in prompt
    assert profile["facets"][0]["definition"] in prompt
    assert profile["facets"][0]["high_behavior"] in prompt
    assert profile["facets"][0]["low_behavior"] in prompt
    assert profile["facets"][0]["common_confounds"][0] in prompt
    assert "示例一" not in prompt and "情境九" not in prompt
    assert "${" not in prompt and "design_rationales" in prompt
    assert "不表示行为越正确" in prompt and "1、2、3、4分各使用一次" in prompt
    alternative = deepcopy(profile)
    alternative["facets"][0].update(facet_name="不同构念", definition="不同定义", high_behavior="高表现", low_behavior="低表现")
    other = build_a_prompt(config, alternative)
    assert "不同构念" in other and "不同定义" in other
    assert profile["facets"][0]["definition"] not in other


@pytest.mark.parametrize("problem", ["missing", "too_few", "wrong_number", "bool_number", "missing_option", "blank_option", "blank_scenario"])
def test_a_design_rationales_reject_incomplete_output(problem):
    from experiments.system_comparison.generation import validate_a_design_rationales
    payload = a_output([item()])
    row = payload["design_rationales"][0]
    if problem == "missing":
        payload.pop("design_rationales")
    elif problem == "too_few":
        payload["design_rationales"] = []
    elif problem == "wrong_number":
        row["item_number"] = 2
    elif problem == "bool_number":
        row["item_number"] = True
    elif problem == "missing_option":
        row["option_rationales"].pop("D")
    elif problem == "blank_option":
        row["option_rationales"]["A"] = "  "
    else:
        row["scenario_rationale"] = ""
    with pytest.raises(ValueError, match="设计依据"):
        validate_a_design_rationales(payload, [item()])


def test_a_saves_rationales_separately_and_resumes_without_model_calls(tmp_path):
    from langchain_core.messages import AIMessage
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.generation import generate_a
    from experiments.system_comparison.evaluation import evaluation_state
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    class Model:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content=json.dumps(a_output([item()]), ensure_ascii=False))
    model = Model()
    asyncio.run(generate_a(store, model))
    asyncio.run(generate_a(ExperimentStore(store.root), model))
    assert model.calls == 1
    details = store.read("A/round_01/design_rationales.json")
    assert details["items"][0]["item_id"] == "A-001"
    assert details["form_fingerprint"] == store.read("A/round_01/form.json")["fingerprint"]
    assert "设计依据" not in json.dumps(evaluation_state(store, "A/round_01")["frozen_item_bank"], ensure_ascii=False)
    assert "design_rationales" not in store.path("A/round_01/form.html").read_text(encoding="utf-8")
    assert store.read("A/round_01/request.json")["payload"]["prompt_version"]


def test_a_old_saved_request_is_not_silently_upgraded(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.generation import generate_a
    store = ExperimentStore.create(tmp_path, ExperimentConfig(sample_size=30))
    store.write("A/round_01/request.json", {"system": "旧提示词", "payload": {}, "model": store.config.model_id})
    class ForbiddenModel:
        async def ainvoke(self, messages):
            raise AssertionError("must reject before calling model")
    with pytest.raises(ValueError, match="新建实验"):
        asyncio.run(generate_a(store, ForbiddenModel()))


def test_a_form_drops_extra_explanation_fields():
    from experiments.system_comparison.generation import validate_a_form
    raw = item()
    raw["response_options"][0]["rationale"] = "PRIVATE EXPLANATION"
    raw["scenario_rationale"] = "PRIVATE EXPLANATION"
    result = validate_a_form({"items": [raw]}, count=1, facet="extraversion_gregariousness")
    assert "PRIVATE EXPLANATION" not in json.dumps(result)


def test_a_missing_rationale_uses_only_bounded_structure_retry(tmp_path):
    from langchain_core.messages import AIMessage
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.storage import ExperimentStore
    from experiments.system_comparison.generation import generate_a
    store = ExperimentStore.create(tmp_path, ExperimentConfig(final_item_count=1, sample_size=30))
    class Model:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=json.dumps({"items": [item()]}))
            assert not store.read("A/round_01/form.json")
            assert "设计依据" in messages[-1].content
            return AIMessage(content=json.dumps(a_output([item()])))
    model = Model()
    asyncio.run(generate_a(store, model))
    assert model.calls == 2
    assert "设计依据" in store.read("A/round_01/attempt_01.json")["error"]
    assert store.read("A/round_01/design_rationales.json")["items"][0]["item_id"] == "A-001"
