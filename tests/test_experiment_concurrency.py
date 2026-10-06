import asyncio
import json

import pytest


def _audit_response():
    return {
        "option_assessments": [
            {
                "option_id": option_id,
                "time_energy_cost": 20,
                "emotional_discomfort": 10,
                "relationship_reputation_risk": 15,
                "subjective_benefit": 60,
                "willingness": 50,
                "first_person_reaction": "我会考虑这样做。",
            }
            for option_id in "ABCD"
        ],
        "choice_probabilities": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
    }


def test_experiment_configs_default_unbounded_and_accept_legacy_limits(tmp_path):
    from experiments.system_comparison.config import ExperimentConfig
    from experiments.system_comparison.current_system_item_defect_pilot import (
        CurrentSystemDefectConfig,
    )
    from experiments.system_comparison.legacy_persona_summary import LegacySummaryConfig

    assert ExperimentConfig.model_fields["max_concurrency"].default == 0
    assert LegacySummaryConfig().max_concurrency == 0

    current = CurrentSystemDefectConfig(experiment=tmp_path)
    current.validate()
    CurrentSystemDefectConfig(experiment=tmp_path, max_concurrency=25).validate()
    with pytest.raises(ValueError, match="max_concurrency"):
        CurrentSystemDefectConfig(experiment=tmp_path, max_concurrency=-1).validate()


def test_embodied_prompt_audit_dispatches_all_items_before_results(monkeypatch, tmp_path):
    from langchain_core.runnables import RunnableLambda

    from experiments.system_comparison import embodied_prompt_audit as audit

    forms_dir = tmp_path / "legacy_summary_embodied_probability_abc"
    forms_dir.mkdir()
    items = [
        {
            "item_id": f"A-{index}",
            "scenario": f"scenario {index}",
            "response_instruction": "What would you do?",
            "response_options": [
                {"option_id": option_id, "text": f"action {option_id}"}
                for option_id in "ABCD"
            ],
        }
        for index in (1, 2)
    ]
    (forms_dir / "forms.json").write_text(
        json.dumps({"A": {"items": items}}), encoding="utf-8"
    )

    both_started = asyncio.Event()
    calls = []

    async def fake_model(_messages):
        calls.append(True)
        if len(calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return json.dumps(_audit_response())

    monkeypatch.setattr(audit, "_load_summary", lambda *_args: "profile summary")
    monkeypatch.setattr(audit, "build_legacy_persona_prompt", lambda *_args, **_kwargs: "persona")
    monkeypatch.setattr(audit, "get_model", lambda _model_id: RunnableLambda(fake_model))
    monkeypatch.setattr(audit, "get_model_request_timeout_seconds", lambda: 5)

    output, result = asyncio.run(
        audit.run_audit(
            experiment=tmp_path,
            summary_pool=tmp_path,
            respondent_id="VR-0001",
            item_ids=["A-1", "A-2"],
            model_id="offline-test-model",
            output=tmp_path / "audit-output",
        )
    )

    assert len(calls) == 2
    assert result["call_count"] == 2
    assert [row["item_id"] for row in result["records"]] == ["A-1", "A-2"]
    assert (output / "A-1.json").is_file()
    assert (output / "A-2.json").is_file()


def test_post_baseline_slots_fan_out_without_reusing_shared_reserves(tmp_path):
    from experiments.post_baseline_iteration_lab.core import (
        LabConfig,
        PostBaselineIterationLab,
        demo_snapshot,
    )

    lab = PostBaselineIterationLab(
        demo_snapshot(5),
        tmp_path / "post-baseline",
        LabConfig(max_replacements_per_slot=1),
        {"items": {"baseline-05": {"disable_reserve": True}}},
    )
    for item_id in ("baseline-02", "baseline-03", "reserve-02"):
        lab.items[item_id]["blueprint_cell_id"] = "shared-cell"
    lab.items["baseline-04"]["blueprint_cell_id"] = "cell-without-reserve"
    lab.reserve_ids = ["reserve-02"]

    async def exhausted_repair(item_id):
        return {
            "item_id": item_id,
            "root_item_id": item_id,
            "outcome": "repair_exhausted",
            "selected_item_id": None,
            "attempts": [],
        }

    lab._repair_one = exhausted_repair
    both_independent_started = asyncio.Event()
    reserve_test_finished = asyncio.Event()
    reserve_tests = []
    independent_started = []
    shared_followup_started_early = []

    async def fake_local_retest(item, attempt, action):
        item_id = item["item_id"]
        root_id = item.get("root_item_id")
        if item_id == "reserve-02":
            reserve_tests.append((item_id, root_id))
            await asyncio.wait_for(both_independent_started.wait(), timeout=3)
            reserve_test_finished.set()
            return {"status": "ok", "metrics": {"passed": True}}
        if item_id in {
            "replacement-baseline-04-01",
            "replacement-baseline-05-01",
        }:
            independent_started.append(item_id)
            if len(independent_started) == 2:
                both_independent_started.set()
            await asyncio.wait_for(both_independent_started.wait(), timeout=3)
        if item_id == "replacement-baseline-03-01" and not reserve_test_finished.is_set():
            shared_followup_started_early.append(item_id)
        return {"status": "ok", "metrics": {"passed": False}}

    lab.engine.local_retest = fake_local_retest
    round_record = asyncio.run(lab._run_repair_round(0))

    assert both_independent_started.is_set()
    assert set(independent_started) == {
        "replacement-baseline-04-01",
        "replacement-baseline-05-01",
    }
    assert reserve_tests == [("reserve-02", "baseline-02")]
    assert lab.state["used_reserve_ids"] == ["reserve-02"]
    assert not shared_followup_started_early
    assert [row["item_id"] for row in round_record["repair_results"]] == [
        "baseline-02",
        "baseline-03",
        "baseline-04",
        "baseline-05",
    ]


def test_fresh_classifier_batch_fans_out_and_resumes_only_missing_jobs(monkeypatch, tmp_path):
    from experiments.system_comparison import defect_type_classifier as classifier

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    stimuli = tmp_path / "stimuli.json"
    stimuli.write_text("{}", encoding="utf-8")
    output = tmp_path / "classifier-output"
    items = [
        {
            "blind_id": blind_id,
            "internal_id": blind_id,
            "source_item_id": f"source-{index}",
            "true_label": label,
            "target_dimension_id": "facet-1",
            "scenario": "scenario",
            "response_instruction": "instruction",
            "options": {},
            "scoring_key": {},
        }
        for index, (blind_id, label) in enumerate(
            (("B-01", "intact"), ("B-02", "construct_shift")),
            1,
        )
    ]
    monkeypatch.setattr(classifier, "build_blind_items", lambda *_args, **_kwargs: items)
    monkeypatch.setattr(
        classifier,
        "_facet_specs",
        lambda: {"facet-1": {"display_label": "facet"}},
    )
    monkeypatch.setattr(classifier, "get_model", lambda *_args: object())
    monkeypatch.setattr(
        classifier,
        "with_compatible_structured_output",
        lambda model, _schema: (model, "mock"),
    )
    monkeypatch.setattr(classifier, "_model_id", lambda *_args: "mock-model")
    monkeypatch.setattr(
        classifier,
        "build_classification_messages",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        classifier,
        "_telemetry_summary",
        lambda *_args: {
            "calls": 2,
            "error_calls": 0,
            "total_tokens": 0,
            "duration_ms": 0,
        },
    )
    config = classifier.DefectClassifierConfig(
        experiment=experiment,
        stimuli_path=stimuli,
        repeats=1,
        max_retries=3,
        output=output,
    )
    both_started = asyncio.Event()
    first_batch_calls = []

    def prediction():
        return {
            "classification": "intact",
            "confidence": 0.9,
            "target_construct_relevance": 4,
            "option_gradient_clarity": 4,
            "social_desirability_pressure": 2,
            "non_target_contamination": 2,
            "likely_non_target_construct": None,
            "rationale": "mock response",
        }

    async def fail_first_after_batch_starts(_runnable, _messages, **kwargs):
        first_batch_calls.append(kwargs)
        if len(first_batch_calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=3)
        if "B-01" in kwargs["job_label"]:
            raise RuntimeError("first call failed")
        return prediction()

    monkeypatch.setattr(classifier, "_invoke_with_retry", fail_first_after_batch_starts)
    with pytest.raises(RuntimeError, match="盲态分类有1次失败"):
        asyncio.run(classifier.run_defect_classifier(config))

    assert both_started.is_set()
    assert {
        "B-01": next(row["max_retries"] for row in first_batch_calls if "B-01" in row["job_label"]),
        "B-02": next(row["max_retries"] for row in first_batch_calls if "B-02" in row["job_label"]),
    } == {"B-01": 0, "B-02": 3}
    saved = classifier._load_jsonl(output / "predictions.jsonl")
    assert [row["blind_id"] for row in saved] == ["B-02"]
    errors = classifier._read_json(output / "errors.json")
    assert errors["phase"] == "bulk_classification"
    assert errors["bulk_started"] is True
    assert errors["completed_calls"] == 1
    assert errors["errors"] == [
        {"blind_id": "B-01", "repeat": 1, "error": "first call failed"}
    ]
    failed_manifest = classifier._read_json(output / "manifest.json")
    assert failed_manifest["failure_phase"] == "bulk_classification"
    assert failed_manifest["completed_calls"] == 1
    assert failed_manifest["total_calls"] == 2

    resume_calls = []

    async def resume_missing(_runnable, _messages, **kwargs):
        resume_calls.append(kwargs)
        return prediction()

    monkeypatch.setattr(classifier, "_invoke_with_retry", resume_missing)
    asyncio.run(classifier.run_defect_classifier(config))

    assert len(resume_calls) == 1
    assert "B-01" in resume_calls[0]["job_label"]
    assert resume_calls[0]["max_retries"] == 3
    assert len(classifier._load_jsonl(output / "predictions.jsonl")) == 2
    complete_manifest = classifier._read_json(output / "manifest.json")
    assert complete_manifest["status"] == "complete"
    assert complete_manifest["completed_calls"] == complete_manifest["total_calls"] == 2
