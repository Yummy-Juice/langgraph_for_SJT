from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from sjt_system.evaluation import virtual_content_review as review


def item_fixture():
    return {
        "item_id": "item-1",
        "version": 2,
        "target_dimension_id": "target",
        "blueprint_cell_id": "cell-1",
        "context_category": "team",
        "scenario": "Original scenario",
        "response_instruction": "What would you do?",
        "skeleton_id": "fixed-skeleton",
        "activation_mechanism": "fixed mechanism",
        "scoring_key": {"D": 1, "B": 2, "A": 3, "C": 4},
        "response_options": [
            {"option_id": option_id, "text": f"old {option_id}", "behavioral_level": level}
            for option_id, level in zip(
                ("D", "B", "A", "C"),
                ("low", "medium_low", "medium_high", "high"),
            )
        ],
    }


def evidence_fixture():
    return {
        "normal_constraints": [
            {
                "constraint_id": "FACET_DEFINITION:target",
                "component": "facet",
                "statement": "Target facet definition",
            },
            {
                "constraint_id": "BE_HIGH:target",
                "component": "behavior_evidence",
                "statement": "Target high-behavior boundary",
            },
            {
                "constraint_id": "BE_LOW:target",
                "component": "behavior_evidence",
                "statement": "Target low-behavior boundary",
            },
        ],
        "target_construct_constraints": [
            {
                "constraint_id": "FACET_DEFINITION:target",
                "component": "facet",
                "statement": "Target facet definition",
            },
            {
                "constraint_id": "BE_HIGH:target",
                "component": "behavior_evidence",
                "statement": "Target high-behavior boundary",
            },
            {
                "constraint_id": "BE_LOW:target",
                "component": "behavior_evidence",
                "statement": "Target low-behavior boundary",
            },
        ],
        "fixed_skeleton": {"behavioral_tension": "fixed tension"},
        "construct_version": "construct-v1",
        "observations": [
            {"observation_id": "OBS:CITC", "value": 0.1, "threshold": 0.3,
             "diagnostic_only": "STATISTICS_MUST_NOT_REACH_EXPERT"},
            {"observation_id": "OBS:TARGET_RHO", "value": 0.2, "threshold": 0.4},
        ],
        "prior_atomic_repairs": [
            {"prior_item_text": "HISTORY_MUST_NOT_REACH_EXPERT"},
        ],
    }


def state_fixture(*, analysis_round=1):
    item = item_fixture()
    return {
        "run_id": "virtual-review-test",
        "psychometric_analysis_round": analysis_round,
        "target_population": "virtual university-student respondents",
        "test_specification": {"target_population": "virtual university-student respondents"},
        "frozen_item_bank": [item],
        "item_pool": [item],
        "blueprint": {"cells": [{"blueprint_cell_id": "cell-1"}]},
    }


def expert_report(*, marker="review-one"):
    return {
        "criteria": {
            "relevance": "supported",
            "comprehensiveness": "supported",
            "comprehensibility": "problem",
        },
        "findings": [{
            "criterion": "option_gradient",
            "field": "response_options",
            "quote": "old D",
            "constraint_id": "FACET_DEFINITION:target",
            "problem": marker,
            "required_change": "Clarify the target behavior in this option.",
        }],
        "needs_scenario_probe": False,
        "summary": marker,
    }


def diagnosis(*, decision="repair", evidence_ids=None, constraint_id="FACET_DEFINITION:target"):
    tasks = []
    if decision != "defer":
        tasks = [{
            "task_id": "edit-option-D",
            "field": "response_options",
            "option_ids": ["D"],
            "quote": "old D",
            "constraint_id": constraint_id,
            "evidence_ids": ["EXPERT:1:F1"] if evidence_ids is None else evidence_ids,
            "required_change": "Clarify the target behavior in option D.",
        }]
    return {
        "decision": decision,
        "summary": "The virtual evidence identifies a localized wording issue.",
        "replacement_scope": "none",
        "repair_tasks": tasks,
    }


def answer_payload(payload):
    first_option = payload["item"]["response_options"][0]
    return {
        "selected_option_id": first_option["option_id"],
        "paraphrase": "I would coordinate the next step.",
        "retrieval": "I recalled a similar team situation.",
        "judgment": "I considered the impact on the team.",
        "selection_reason": "This option best fits my intended action.",
        "unclear_quotes": [],
    }


def probe_payload(payload):
    return {
        "interpretation": "The item asks for a likely action in a team situation.",
        "option_interpretations": [
            {"option_id": row["option_id"], "meaning": f"Meaning of {row['option_id']}"}
            for row in payload["item"]["response_options"]
        ],
        "issues": [],
        "summary": "The options were interpreted as distinct actions.",
    }


def patch_payload():
    return {
        "reason": "The authorized option wording is clarified.",
        "response_options": [{"option_id": "D", "text": "Clarified low-level action"}],
    }


class FakeInvoke:
    def __init__(self, *, decision="repair", disagreement=False, fault_kind=None, fault_mode=None):
        self.decision = decision
        self.disagreement = disagreement
        self.fault_kind = fault_kind
        self.fault_mode = fault_mode
        self.fault_used = False
        self.calls = []

    async def __call__(self, kind, payload):
        self.calls.append((kind, deepcopy(payload)))
        if kind == self.fault_kind and not self.fault_used:
            self.fault_used = True
            if self.fault_mode == "timeout":
                raise TimeoutError("simulated timeout")
            if self.fault_mode == "structure":
                return {}
        if kind == "expert_review":
            report = expert_report()
            if self.disagreement and payload["expertise"] == "sjt_and_measurement":
                report["criteria"]["relevance"] = "problem"
            return report
        if kind == "expert_arbitration":
            return expert_report(marker="arbitrated")
        if kind == "interview_answer":
            return answer_payload(payload)
        if kind == "interview_probe":
            return probe_payload(payload)
        if kind == "content_diagnosis":
            return diagnosis(decision=self.decision)
        if kind == "content_edit":
            return patch_payload()
        raise AssertionError(f"Unexpected virtual review call: {kind}")


def run_review(tmp_path, invoke, *, state=None, item=None, evidence=None, path=None):
    return asyncio.run(review.run_virtual_content_repair(
        state=state or state_fixture(),
        item=item or item_fixture(),
        evidence=evidence or evidence_fixture(),
        path=path or (tmp_path / "review.json"),
        invoke=invoke,
        model_info={"expert_review": {"model_id": "fake"}},
    ))


def test_review_sequence_is_two_experts_six_answer_probe_pairs_then_diagnosis_and_edit(tmp_path):
    fake = FakeInvoke()
    result = run_review(tmp_path, fake)

    expected = ["expert_review", "expert_review"]
    for _ in range(6):
        expected.extend(["interview_answer", "interview_probe"])
    expected.extend(["content_diagnosis", "content_edit"])
    assert [kind for kind, _ in fake.calls] == expected
    assert result["status"] == "ready"
    assert len(result["expert_reviews"]) == 2
    assert len(result["cognitive_interviews"]) == 6
    assert result["cosmin_gate"]["post_edit_comprehensibility"] == "not_reassessed"


def test_single_pair_difference_restores_only_the_omitted_pair_prefix():
    pair = {"pair_id": "p30684eff", "left": {}, "right": {}}
    raw = {"records": [{"pair_id": "30684eff", "difference": "A shows more warmth than B."}]}

    result, correction = review._validate_single_pair_difference(raw, pair)

    assert result == {"p30684eff": "A shows more warmth than B."}
    assert correction == {
        "provided_id": "30684eff",
        "canonical_id": "p30684eff",
        "method": "single_pair_missing_prefix",
    }
    assert raw["records"][0]["pair_id"] == "30684eff"

    with pytest.raises(ValueError, match="unknown or duplicate anonymous pair ID"):
        review._validate_single_pair_difference(
            {"records": [{"pair_id": "other", "difference": "A shows more warmth than B."}]}, pair
        )


def test_experts_are_blind_to_statistics_and_history_and_interviews_to_construct_scoring_and_levels(tmp_path):
    fake = FakeInvoke()
    run_review(tmp_path, fake)

    expert_payloads = [payload for kind, payload in fake.calls if kind == "expert_review"]
    assert len(expert_payloads) == 2
    for payload in expert_payloads:
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "STATISTICS_MUST_NOT_REACH_EXPERT" not in encoded
        assert "HISTORY_MUST_NOT_REACH_EXPERT" not in encoded
        assert "statistical_symptoms" not in payload
        assert "prior_repairs" not in payload

    interview_payloads = [payload for kind, payload in fake.calls
                          if kind in {"interview_answer", "interview_probe"}]
    assert len(interview_payloads) == 12
    for payload in interview_payloads:
        assert set(payload) in ({"role", "item"}, {"role", "item", "previous_answer"})
        assert set(payload["role"]) == {
            "role_id", "target_population", "reading_style", "experience", "decision_style", "synthetic"
        }
        assert payload["role"]["synthetic"] is True
        assert set(payload["item"]) == {"scenario", "response_instruction", "response_options"}
        assert all(set(option) == {"option_id", "text"} for option in payload["item"]["response_options"])
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "target_construct_constraints" not in encoded
        assert "scoring_key" not in encoded
        assert "behavioral_level" not in encoded
        assert '"target"' not in encoded


@pytest.mark.parametrize("invalid", ["no_evidence", "external_citation", "unknown_constraint", "overlap", "protected_field"])
def test_diagnosis_rejects_unverifiable_or_unsafe_repair_tasks(invalid):
    item = item_fixture()
    evidence = evidence_fixture()
    materials = review.material_packet(state_fixture(), item, evidence)
    finding = expert_report()["findings"][0]
    registry = {"EXPERT:1:F1": finding}
    proposed = diagnosis()

    if invalid == "no_evidence":
        proposed["repair_tasks"][0]["evidence_ids"] = []
    elif invalid == "external_citation":
        proposed["repair_tasks"][0]["evidence_ids"] = ["COSMIN:page-13"]
    elif invalid == "unknown_constraint":
        proposed["repair_tasks"][0]["constraint_id"] = "UNKNOWN:constraint"
    elif invalid == "overlap":
        proposed["repair_tasks"].append({
            **deepcopy(proposed["repair_tasks"][0]), "task_id": "edit-option-D-again",
        })
    else:
        proposed["repair_tasks"][0]["field"] = "scoring_key"

    with pytest.raises(ValueError):
        review.validate_diagnosis(proposed, item, materials, registry)


def test_legal_patch_changes_only_authorized_option_without_incrementing_version(tmp_path):
    original = item_fixture()
    fake = FakeInvoke()
    result = run_review(tmp_path, fake, item=original)

    candidate = result["candidate"]
    assert candidate["version"] == original["version"]
    assert candidate["response_options"][0]["text"] == "Clarified low-level action"
    assert candidate["scenario"] == original["scenario"]
    assert candidate["response_instruction"] == original["response_instruction"]
    assert candidate["scoring_key"] == original["scoring_key"]
    assert [row["behavioral_level"] for row in candidate["response_options"]] == [
        row["behavioral_level"] for row in original["response_options"]
    ]
    assert candidate["response_options"][1:] == original["response_options"][1:]


def test_defer_preserves_original_and_returns_no_candidate(tmp_path):
    original = item_fixture()
    fake = FakeInvoke(decision="defer")
    result = run_review(tmp_path, fake, item=original)

    assert result["status"] == "deferred"
    assert result["candidate"] is None
    assert result["diagnosis"]["decision"] == "defer"
    assert not any(kind == "content_edit" for kind, _ in fake.calls)
    assert original == item_fixture()


@pytest.mark.parametrize("fault_mode", ["timeout", "structure"])
def test_failure_pauses_and_resume_reuses_completed_calls_without_resetting_attempts(tmp_path, fault_mode):
    path = tmp_path / f"resume-{fault_mode}.json"
    fake = FakeInvoke(fault_kind="content_diagnosis", fault_mode=fault_mode)
    first = run_review(tmp_path, fake, path=path)

    assert first["status"] == "paused"
    assert first["diagnosis"] is None
    assert not any(kind == "content_edit" for kind, _ in fake.calls)

    second = run_review(tmp_path, fake, path=path)
    assert second["status"] == "ready"
    assert second["diagnosis"]["decision"] == "repair"
    assert [kind for kind, _ in fake.calls].count("expert_review") == 2
    assert [kind for kind, _ in fake.calls].count("interview_answer") == 6
    assert [kind for kind, _ in fake.calls].count("interview_probe") == 6
    assert [kind for kind, _ in fake.calls].count("content_diagnosis") == 2
    assert [kind for kind, _ in fake.calls].count("content_edit") == 1

    archive = json.loads(path.read_text(encoding="utf-8"))
    diagnosis_attempts = [row["attempt"] for row in archive["calls"] if row["key"] == "diagnosis"]
    assert diagnosis_attempts == [1, 2]
    assert all(row["attempt"] <= 2 for row in archive["calls"])


def test_fourth_measurement_round_rejects_new_content_repair(tmp_path):
    path = tmp_path / "must-not-start.json"
    with pytest.raises(ValueError, match="completed measurement"):
        run_review(tmp_path, FakeInvoke(), state=state_fixture(analysis_round=4), path=path)
    assert not path.exists()


@pytest.mark.parametrize("changed", ["item_version", "source_evidence"])
def test_archive_rejects_reentry_when_source_version_or_evidence_changed(tmp_path, changed):
    path = tmp_path / "bound-source.json"
    first = run_review(tmp_path, FakeInvoke(), path=path)
    assert first["status"] == "ready"

    item = item_fixture()
    evidence = evidence_fixture()
    if changed == "item_version":
        item["version"] += 1
    else:
        evidence["normal_constraints"].append({
            "constraint_id": "NEW:source-evidence", "statement": "Changed frozen source evidence",
        })
    with pytest.raises(ValueError, match="inputs changed"):
        run_review(tmp_path, FakeInvoke(), item=item, evidence=evidence, path=path)


def test_disagreement_triggers_at_most_one_virtual_arbitration(tmp_path):
    fake = FakeInvoke(disagreement=True)
    result = run_review(tmp_path, fake)

    assert result["status"] == "ready"
    assert [kind for kind, _ in fake.calls].count("expert_arbitration") == 1
    assert len(result["expert_reviews"]) == 3

