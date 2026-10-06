import asyncio
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _expansion():
    from sjt_system.authoring.situation_space import (
        BehaviorExpansionRecord,
        FacetExpansion,
        MechanismRecord,
        SituationRecord,
    )

    behaviors = []
    for behavior_index, count in enumerate((2, 1)):
        mechanisms = []
        for mechanism_index in range(count):
            key = f"b{behavior_index}m{mechanism_index}"
            mechanisms.append(
                MechanismRecord(
                    mechanism_id=key,
                    activation_mechanism=key,
                    situations=[
                        SituationRecord(
                            situation_id=f"{key}-s0",
                            domain="work",
                            actor_relation="colleague",
                            event_class="handoff",
                        )
                    ],
                )
            )
        behaviors.append(
            BehaviorExpansionRecord(
                behavior_id=f"b{behavior_index}", mechanisms=mechanisms
            )
        )
    return FacetExpansion(
        facet_id="target_facet",
        target_population="adults",
        behavior_expansions=behaviors,
    )


def test_mechanism_validation_fans_out_and_keeps_failure_order(monkeypatch):
    import sjt_system.authoring.situation_space as situation_space

    expansion = _expansion()
    all_started = asyncio.Event()
    started = []
    feedback = []

    async def validate(text, facet, neighbors):
        started.append(text)
        if len(started) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        if len(started) <= 3:
            return {"target_is_first": text not in {"b0m1", "b1m0"}, "reason": f"reason-{text}"}
        return {"target_is_first": True, "reason": ""}

    async def regenerate(*args, **kwargs):
        feedback.append(kwargs["extra_context"])
        return expansion

    monkeypatch.setattr(situation_space, "_validate_one_mechanism", validate)
    monkeypatch.setattr(situation_space, "_generate_expansion", regenerate)

    result = asyncio.run(
        situation_space._validate_and_repair_mechanisms(
            expansion,
            runnable=None,
            facet={"facet_id": "target_facet"},
            behavior_evidence=[],
            target_population="adults",
            output_language="zh",
            required_situation_count=1,
        )
    )

    assert result is expansion
    assert started[:3] == ["b0m0", "b0m1", "b1m0"]
    assert len(feedback) == 1
    assert feedback[0].index("失败的 mechanism: b0m1") < feedback[0].index(
        "失败的 mechanism: b1m0"
    )


def test_mechanism_validation_drains_round_before_raising(monkeypatch):
    import sjt_system.authoring.situation_space as situation_space

    expansion = _expansion()
    all_started = asyncio.Event()
    started = []

    async def validate(text, facet, neighbors):
        started.append(text)
        if len(started) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        if text == "b0m1":
            raise RuntimeError("first indexed failure")
        if text == "b1m0":
            raise RuntimeError("later indexed failure")
        return {"target_is_first": True, "reason": ""}

    monkeypatch.setattr(situation_space, "_validate_one_mechanism", validate)

    with pytest.raises(RuntimeError, match="first indexed failure"):
        asyncio.run(
            situation_space._validate_and_repair_mechanisms(
                expansion,
                runnable=None,
                facet={"facet_id": "target_facet"},
                behavior_evidence=[],
                target_population="adults",
                output_language="zh",
                required_situation_count=1,
            )
        )
    assert started == ["b0m0", "b0m1", "b1m0"]


def _repair_item():
    return {
        "item_id": "probe-item",
        "version": 1,
        "blueprint_cell_id": "cell-1",
        "target_dimension_id": "extraversion_gregariousness",
        "scenario": "同事邀请你参加团队活动。",
        "response_instruction": "你会怎么做？",
        "response_options": [
            {"option_id": code, "text": f"原行为{code}"} for code in "ABCD"
        ],
        "scoring_key": {"A": 1, "B": 2, "C": 3, "D": 4},
    }


def _repair_evidence(item):
    return {
        "current_item": item,
        "observations": [
            {"observation_id": "OBS:CITC", "value": 0.1, "threshold": 0.3},
            {"observation_id": "OBS:TARGET_RHO", "value": 0.5, "threshold": 0.4},
            {"observation_id": "OBS:SAME_DOMAIN_VTS", "value": 0.3, "threshold": 0.2},
            {"observation_id": "OBS:CROSS_DOMAIN_VTS", "value": 0.4, "threshold": 0.3},
        ],
        "normal_constraints": [
            {"constraint_id": "FACET_DEFINITION:target", "statement": "目标定义"},
            {"constraint_id": "BE_HIGH:target", "statement": "高水平行为锚点"},
            {"constraint_id": "BE_LOW:target", "statement": "低水平行为锚点"},
        ],
        "option_choice_diagnostics": {
            "target_option_gradient": {
                "passes": False,
                "failed_adjacent_pairs": [
                    {"lower_option_id": "A", "higher_option_id": "B"}
                ],
            }
        },
    }


def _action(level):
    return {
        "initial_ideas": f"想法-{level}",
        "practical_action": f"行动-{level}",
        "when_facilitated": f"促进-{level}",
        "when_blocked": f"阻碍-{level}",
    }


def test_scenario_probes_fan_out_cache_holes_and_resume(tmp_path):
    from sjt_system.evaluation.scenario_detection import run_scenario_repair

    item = _repair_item()
    evidence = _repair_evidence(item)
    path = tmp_path / "probe.json"
    all_started = asyncio.Event()
    first_round_started = []
    resumed_calls = []
    first_round = True

    async def invoke(kind, payload):
        nonlocal first_round
        if kind == "open_action":
            level = payload["persona"]["level"]
            if first_round:
                first_round_started.append(level)
                if len(first_round_started) == 4:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=2)
                if level == "medium_low":
                    raise asyncio.CancelledError()
            else:
                resumed_calls.append(level)
            return _action(level)
        if kind == "compare_adjacent":
            return {
                "records": [
                    {"pair_id": row["pair_id"], "difference": "目标行为差异"}
                    for row in payload["pairs"]
                ]
            }
        if kind == "rewrite_options":
            return {
                "response_options": [
                    {"option_id": code, "text": f"新行为{code}"} for code in ("A", "B")
                ],
                "reason": "依据目标行为差异调整相邻选项",
            }
        raise AssertionError(f"unexpected model call: {kind}")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_scenario_repair(
                item=item,
                evidence=evidence,
                path=path,
                invoke=invoke,
            )
        )

    paused = json.loads(path.read_text(encoding="utf-8"))
    actions = paused["rounds"][0]["actions"]
    assert first_round_started == ["low", "medium_low", "medium_high", "high"]
    assert len(actions) == 4
    assert actions[1] is None
    assert [actions[index]["action"]["practical_action"] for index in (0, 2, 3)] == [
        "行动-low",
        "行动-medium_high",
        "行动-high",
    ]

    first_round = False
    result = asyncio.run(
        run_scenario_repair(
            item=item,
            evidence=evidence,
            path=path,
            invoke=invoke,
        )
    )

    assert resumed_calls == ["medium_low"]
    assert result["status"] == "ready"
    final = json.loads(path.read_text(encoding="utf-8"))
    final_actions = final["rounds"][0]["actions"]
    assert [row["action"]["practical_action"] for row in final_actions] == [
        "行动-low",
        "行动-medium_low",
        "行动-medium_high",
        "行动-high",
    ]


def test_behavior_evidence_cli_mines_unique_facets_concurrently(monkeypatch, tmp_path, capsys):
    import sjt_system.knowledge.behavior_evidence_cli as cli

    all_started = asyncio.Event()
    started = []
    saved = []

    monkeypatch.setattr(cli, "load_ipip_corpus", lambda _path: object())

    async def mine(code, corpus):
        started.append(code)
        if len(started) == 2:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        return SimpleNamespace(facet_code=code, evidence=[object()])

    def save(bundle, root):
        saved.append(bundle.facet_code)
        return Path(root) / f"{bundle.facet_code}.json"

    monkeypatch.setattr(cli, "mine_behavior_evidence", mine)
    monkeypatch.setattr(cli, "save_behavior_evidence_bundle", save)

    status = asyncio.run(
        cli._build(
            Namespace(
                facet="A4, E2, A4",
                corpus=tmp_path / "corpus.json",
                output_root=tmp_path / "out",
            )
        )
    )

    assert status == 0
    assert started == ["A4", "E2"]
    assert len(saved) == 2
    assert set(saved) == {"A4", "E2"}
    assert set(capsys.readouterr().out.splitlines()) == {
        f"A4: evidence=1 -> {tmp_path / 'out' / 'A4.json'}",
        f"E2: evidence=1 -> {tmp_path / 'out' / 'E2.json'}",
    }
