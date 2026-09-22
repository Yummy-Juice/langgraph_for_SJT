from __future__ import annotations

import json
from pathlib import Path

from experiments.post_baseline_iteration_lab.core import LabConfig, PostBaselineIterationLab, demo_snapshot


def _run(tmp_path: Path, scenario: dict) -> tuple[dict, PostBaselineIterationLab]:
    lab = PostBaselineIterationLab(
        demo_snapshot(4),
        tmp_path / "lab",
        LabConfig(
            max_item_repair_attempts=3,
            max_model_retries=1,
            max_replacements_per_slot=2,
            max_repair_rounds=3,
        ),
        scenario,
    )
    return lab.run(), lab


def test_post_baseline_flow_only_repairs_items_and_replaces_failed_slots(tmp_path: Path) -> None:
    state, lab = _run(
        tmp_path,
        {
            "items": {
                "baseline-02": {"pass_after": 2},
                "baseline-03": {"pass_after": None, "replacement_pass_after": 1},
                "baseline-04": {"pass_after": None, "disable_reserve": True, "replacement_pass_after": 1},
            }
        },
    )

    assert state["status"] == "complete"
    assert "reserve-03" in state["current_item_ids"]
    assert any(item_id.startswith("replacement-baseline-04") for item_id in state["current_item_ids"])
    assert lab.engine.calls["diagnosis"] >= 3
    assert "form_evaluation" not in lab.engine.calls
    assert (tmp_path / "lab" / "rounds" / "round_02.json").exists()
    assert not (tmp_path / "lab" / "final" / "form.json").exists()
    assert (tmp_path / "lab" / "summary" / "current_items.json").exists()


def test_three_failed_repairs_are_auto_replenished_or_marked_unresolved(tmp_path: Path) -> None:
    state, _ = _run(
        tmp_path,
        {
            "items": {
                "baseline-02": {"pass_after": None},
                "baseline-03": {"pass_after": None},
                "baseline-04": {"pass_after": None},
            }
        },
    )
    assert state["status"] == "infeasible"
    assert state["unresolved_slots"]
    assert any(entry["status"] == "unresolved" for entry in state["unresolved_slots"])
    assert not any(
        item.get("status") == "qualified_locked"
        for item in json.loads(
            (tmp_path / "lab" / "checkpoint.json").read_text(encoding="utf-8")
        )["items"].values()
    )


def test_diagnosis_failure_is_technical_failure_not_item_elimination(tmp_path: Path) -> None:
    state, _ = _run(tmp_path, {"items": {"baseline-02": {"diagnosis_failures": 99}}})
    assert state["status"] == "infeasible"
    technical = [entry for entry in state["unresolved_slots"] if entry["item_id"] == "baseline-02"]
    assert technical and technical[0]["status"] == "technical_failure"


def test_resume_after_baseline_restores_item_repair_state_without_form_state(tmp_path: Path) -> None:
    output = tmp_path / "lab"
    lab = PostBaselineIterationLab(
        demo_snapshot(4),
        output,
        LabConfig(max_repair_rounds=3),
        {"items": {"baseline-02": {"pass_after": 1}, "baseline-03": {"pass_after": 1}, "baseline-04": {"pass_after": 1}}},
    )
    paused = lab.run(stop_after="baseline")
    assert paused["status"] == "paused"
    before = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert before["state"]["repair_rounds"] == []
    resumed = PostBaselineIterationLab.resume(output).run()
    after = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert resumed["status"] == "complete"
    assert len(after["state"]["repair_rounds"]) == 1
    assert "form_evaluation" not in after["engine_calls"]
    assert not (output / "summary" / "metrics.json").exists()


def test_report_contains_item_repair_summary_not_form_metrics(tmp_path: Path) -> None:
    state, _ = _run(
        tmp_path,
        {
            "items": {
                "baseline-02": {"pass_after": 1},
                "baseline-03": {"pass_after": 1},
                "baseline-04": {"pass_after": 1},
            }
        },
    )
    report = (tmp_path / "lab" / "summary" / "report.html").read_text(encoding="utf-8")
    assert state["status"] == "complete"
    assert "单题返修轮次" in report
    assert "整卷历史" not in report
    assert "质量主指标" not in report
