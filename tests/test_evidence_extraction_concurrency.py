import asyncio
import json
from types import SimpleNamespace

import pytest


def _items(count):
    return [
        {
            "item_id": f"openness_to_ideas_{index:02d}",
            "facet": "openness_to_ideas",
            "situation": f"Situation {index}",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        }
        for index in range(1, count + 1)
    ]


def _unit(module, item):
    return module.ItemEvidenceUnit.model_validate(
        {
            "item_id": item["item_id"],
            "facet": item["facet"],
            "situation_surface": item["situation"],
            "trait_relevant_cue": "An optional intellectual challenge",
            "evidence_proposition": "Whether the person explores the unfamiliar idea",
            "high_behaviors": ["explore one", "explore two"],
            "low_behaviors": ["avoid one", "avoid two"],
            "behavioral_mechanism": "Approach versus avoidance of intellectual exploration",
            "alternative_explanation": "No obvious alternative explanation",
            "competing_facets": [],
            "situation_cost": "Time and effort",
            "high_option_ids": ["A", "B"],
            "low_option_ids": ["C", "D"],
        }
    )


def _library():
    return SimpleNamespace(
        evidence_library=[],
        model_dump=lambda **_kwargs: {"facet": "openness_to_ideas", "evidence_library": []},
    )


def _behavior_bank():
    return SimpleNamespace(
        manifestations=["one"],
        model_dump=lambda **_kwargs: {"facet": "openness_to_ideas", "manifestations": []},
    )


def _cue_bank():
    return SimpleNamespace(
        cues=["one"],
        model_dump=lambda **_kwargs: {"facet": "openness_to_ideas", "cues": []},
    )


def _stub_downstream(monkeypatch, module, *, stage2=None, stage3=None, stage4=None):
    async def build_stage2(facet_key, evidence_units, agent):
        if stage2 is not None:
            return await stage2(facet_key, evidence_units, agent)
        return _library()

    async def build_stage3(facet_key, library, evidence_units, agent):
        if stage3 is not None:
            return await stage3(facet_key, library, evidence_units, agent)
        return _behavior_bank()

    async def build_stage4(facet_key, library, evidence_units, agent):
        if stage4 is not None:
            return await stage4(facet_key, library, evidence_units, agent)
        return _cue_bank()

    monkeypatch.setattr(module, "run_stage2_family_builder", build_stage2)
    monkeypatch.setattr(module, "run_stage3_behavior_bank", build_stage3)
    monkeypatch.setattr(module, "run_stage4_cue_bank", build_stage4)


def test_process_facet_fans_out_stage1_in_order_and_stages_3_4_together(
    monkeypatch, tmp_path
):
    import mussel_evidence_extraction as module

    items = _items(4)
    monkeypatch.setattr(module, "load_mussel_items", lambda _facet: items)
    all_items_started = asyncio.Event()
    all_libraries_started = asyncio.Event()
    stage1_started = []
    stage1_finished = []
    downstream_started = []
    stage2_order = []

    async def extract(facet_key, item, agent):
        stage1_started.append(item["item_id"])
        if len(stage1_started) == len(items):
            all_items_started.set()
        await asyncio.wait_for(all_items_started.wait(), timeout=2)
        stage1_finished.append(item["item_id"])
        return _unit(module, item)

    async def build_stage2(facet_key, units, agent):
        assert len(stage1_finished) == len(items)
        assert set(stage1_finished) == {item["item_id"] for item in items}
        stage2_order.extend(unit.item_id for unit in units)
        return _library()

    async def build_stage3(facet_key, library, units, agent):
        downstream_started.append("stage3")
        if len(downstream_started) == 2:
            all_libraries_started.set()
        await asyncio.wait_for(all_libraries_started.wait(), timeout=2)
        return _behavior_bank()

    async def build_stage4(facet_key, library, units, agent):
        downstream_started.append("stage4")
        if len(downstream_started) == 2:
            all_libraries_started.set()
        await asyncio.wait_for(all_libraries_started.wait(), timeout=2)
        return _cue_bank()

    monkeypatch.setattr(module, "run_stage1_single_item", extract)
    _stub_downstream(
        monkeypatch, module, stage2=build_stage2, stage3=build_stage3, stage4=build_stage4
    )

    asyncio.run(
        module.process_facet(
            "开放性", tmp_path, object(), object(), object(), object()
        )
    )

    expected_ids = [item["item_id"] for item in items]
    output_dir = tmp_path / "openness_to_ideas"
    aggregate_path = output_dir / "stage1_per_item_evidence.json"
    assert stage1_started == expected_ids
    assert stage2_order == expected_ids
    assert json.loads(aggregate_path.read_text(encoding="utf-8"))[0]["item_id"] == expected_ids[0]
    assert [
        json.loads((output_dir / "stage1_items" / f"{item_id}.json").read_text(encoding="utf-8"))["item_id"]
        for item_id in expected_ids
    ] == expected_ids
    assert set(downstream_started) == {"stage3", "stage4"}
    assert (output_dir / "stage3_behavior_bank.json").exists()
    assert (output_dir / "stage4_cue_bank.json").exists()


def test_stage1_partial_failure_saves_successes_and_resume_calls_only_missing(
    monkeypatch, tmp_path
):
    import mussel_evidence_extraction as module

    items = _items(3)
    monkeypatch.setattr(module, "load_mussel_items", lambda _facet: items)
    all_started = asyncio.Event()
    calls = []
    fail_first_batch = True

    async def extract(facet_key, item, agent):
        calls.append(item["item_id"])
        if fail_first_batch:
            if len(calls) == len(items):
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=2)
            if item["item_id"] == items[0]["item_id"]:
                raise RuntimeError("stage1 item failed")
        return _unit(module, item)

    monkeypatch.setattr(module, "run_stage1_single_item", extract)
    _stub_downstream(monkeypatch, module)
    output_dir = tmp_path / "openness_to_ideas"
    item_cache_dir = output_dir / "stage1_items"

    with pytest.raises(RuntimeError, match="stage1 item failed"):
        asyncio.run(
            module.process_facet(
                "开放性", tmp_path, object(), object(), object(), object()
            )
        )

    expected_ids = [item["item_id"] for item in items]
    assert calls == expected_ids
    assert not (output_dir / "stage1_per_item_evidence.json").exists()
    assert not (item_cache_dir / f"{expected_ids[0]}.json").exists()
    assert all((item_cache_dir / f"{item_id}.json").exists() for item_id in expected_ids[1:])

    fail_first_batch = False
    asyncio.run(
        module.process_facet(
            "开放性", tmp_path, object(), object(), object(), object()
        )
    )

    assert calls == [*expected_ids, expected_ids[0]]
    aggregate = json.loads(
        (output_dir / "stage1_per_item_evidence.json").read_text(encoding="utf-8")
    )
    assert [row["item_id"] for row in aggregate] == expected_ids


def test_cached_stage1_unit_must_match_current_source_item(monkeypatch, tmp_path):
    import mussel_evidence_extraction as module

    items = _items(2)
    monkeypatch.setattr(module, "load_mussel_items", lambda _facet: items)
    item_cache_dir = tmp_path / "openness_to_ideas" / "stage1_items"
    item_cache_dir.mkdir(parents=True)
    cache_path = item_cache_dir / f"{items[0]['item_id']}.json"
    cached_text = json.dumps(_unit(module, items[1]).model_dump(mode="json"))
    cache_path.write_text(cached_text, encoding="utf-8")

    async def should_not_run(*_args):
        pytest.fail("a cache with the wrong source item_id must not be used")

    monkeypatch.setattr(module, "run_stage1_single_item", should_not_run)
    _stub_downstream(monkeypatch, module)

    with pytest.raises(ValueError, match="does not match source item"):
        asyncio.run(
            module.process_facet(
                "开放性", tmp_path, object(), object(), object(), object()
            )
        )
    assert cache_path.read_text(encoding="utf-8") == cached_text


def test_complete_legacy_stage1_aggregate_is_validated_but_not_rewritten(
    monkeypatch, tmp_path
):
    import mussel_evidence_extraction as module

    items = _items(3)
    monkeypatch.setattr(module, "load_mussel_items", lambda _facet: items)
    output_dir = tmp_path / "openness_to_ideas"
    output_dir.mkdir(parents=True)
    aggregate_path = output_dir / "stage1_per_item_evidence.json"
    cached_units = [_unit(module, item) for item in reversed(items)]
    aggregate_text = json.dumps(
        [unit.model_dump(mode="json") for unit in cached_units],
        ensure_ascii=False,
    )
    aggregate_path.write_text(aggregate_text, encoding="utf-8")
    stage2_order = []

    async def should_not_run(*_args):
        pytest.fail("a valid legacy aggregate should avoid Stage 1 calls")

    async def build_stage2(facet_key, units, agent):
        stage2_order.extend(unit.item_id for unit in units)
        return _library()

    monkeypatch.setattr(module, "run_stage1_single_item", should_not_run)
    _stub_downstream(monkeypatch, module, stage2=build_stage2)

    asyncio.run(
        module.process_facet(
            "开放性", tmp_path, object(), object(), object(), object()
        )
    )

    assert stage2_order == [item["item_id"] for item in items]
    assert aggregate_path.read_text(encoding="utf-8") == aggregate_text


def test_main_starts_all_five_facets_together(monkeypatch):
    import mussel_evidence_extraction as module
    import sjt_system.agent.agent_factory as agent_factory

    monkeypatch.setattr(agent_factory, "create_agent", lambda *_args, **_kwargs: object())
    all_started = asyncio.Event()
    started = []

    async def process(facet_key, output_root, stage1, stage2, stage3, stage4):
        started.append(facet_key)
        if len(started) == 5:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)

    monkeypatch.setattr(module, "process_facet", process)
    asyncio.run(module.main())

    assert started == ["开放性", "责任心", "外倾性", "宜人性", "神经质"]
