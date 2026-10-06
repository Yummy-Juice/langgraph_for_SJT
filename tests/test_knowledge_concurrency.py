from __future__ import annotations

import asyncio
from copy import deepcopy
import json

from sjt_system.evaluation import repair_knowledge as knowledge
from sjt_system.evaluation import repair_recovery
from test_repair_knowledge import RuleModel, entry, option_evidence, state_fixture


def two_scope_sources(tmp_path):
    state = state_fixture(tmp_path)
    first = option_evidence(same=False, cross=False)
    second = deepcopy(first)
    second["construct_version"] = "definition-v2"
    second["current_item"]["item_id"] = "item-two"
    sources = knowledge.review_sources(state, [entry(first), entry(second)])
    return state, sources


def install_single_attempt_journal(monkeypatch):
    async def journaled_call(*, invoke, kind, payload, validate, **_kwargs):
        return validate(await invoke(kind, payload))

    monkeypatch.setattr(repair_recovery, "journaled_call", journaled_call)


async def learn(store, sources, invoke, tmp_path):
    await store.learn(
        sources,
        stage_id="scope-concurrency-test",
        invoke=invoke,
        model_info={"model_id": "mock"},
        log_root=tmp_path / "calls",
    )


def make_store(tmp_path, state):
    return knowledge.KnowledgeStore(knowledge.config_for_mode(state, "run_only"))


def test_learn_starts_all_scope_invocations_before_any_can_finish(tmp_path, monkeypatch):
    state, sources = two_scope_sources(tmp_path)
    store = make_store(tmp_path, state)
    expected_ids = {source["source_id"] for source in sources}
    assert len(expected_ids) == 2
    assert len({store.scope(source["construct"]) for source in sources}) == 2
    install_single_attempt_journal(monkeypatch)

    all_started = asyncio.Event()
    started_ids = []
    active = 0
    peak = 0
    rule_model = RuleModel()

    async def invoke(payload):
        nonlocal active, peak
        source_id = payload["sources"][0]["source_id"]
        started_ids.append(source_id)
        active += 1
        peak = max(peak, active)
        if len(started_ids) == len(expected_ids):
            all_started.set()
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2)
            return await rule_model(payload)
        finally:
            active -= 1

    asyncio.run(learn(store, sources, invoke, tmp_path))

    assert set(started_ids) == expected_ids
    assert len(started_ids) == len(expected_ids)
    assert peak == len(expected_ids)


def test_scope_failure_drains_siblings_and_resume_skips_processed_scope(tmp_path, monkeypatch):
    state, sources = two_scope_sources(tmp_path)
    store = make_store(tmp_path, state)
    source_ids = {source["source_id"] for source in sources}
    assert len(source_ids) == 2
    install_single_attempt_journal(monkeypatch)

    failed_id = sources[0]["source_id"]
    succeeded_id = next(source_id for source_id in source_ids if source_id != failed_id)
    all_started = asyncio.Event()
    started_ids = []
    first_model = RuleModel()

    async def first_invoke(payload):
        source_id = payload["sources"][0]["source_id"]
        started_ids.append(source_id)
        if len(started_ids) == len(source_ids):
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        if source_id == failed_id:
            raise RuntimeError("mock scope failure")
        return await first_model(payload)

    try:
        asyncio.run(learn(store, sources, first_invoke, tmp_path))
    except RuntimeError as exc:
        assert str(exc) == "mock scope failure"
    else:
        raise AssertionError("learn must propagate the failed scope after draining siblings")

    assert set(started_ids) == source_ids
    with store.connect() as db:
        processed = {row[0] for row in db.execute(
            "SELECT id FROM sources WHERE processed=1"
        )}
        jobs = db.execute("SELECT status,archive FROM jobs").fetchall()
        revision_count = db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    assert processed == {store.source_key(succeeded_id)}
    assert revision_count == 1
    assert len(jobs) == 2
    assert {status for status, _archive in jobs} == {"completed", "error"}
    assert all(_archive for _status, _archive in jobs)

    archives = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "calls").glob("*.json")
    ]
    assert len(archives) == 2
    archive_by_source = {
        record["input"]["sources"][0]["source_id"]: record for record in archives
    }
    assert set(archive_by_source) == source_ids
    assert archive_by_source[succeeded_id]["status"] == "completed"
    assert archive_by_source[failed_id]["status"] == "error"

    resumed_store = make_store(tmp_path, state)
    resumed_ids = []
    resumed_model = RuleModel()

    async def resumed_invoke(payload):
        source_id = payload["sources"][0]["source_id"]
        resumed_ids.append(source_id)
        return await resumed_model(payload)

    asyncio.run(learn(resumed_store, sources, resumed_invoke, tmp_path))

    assert resumed_ids == [failed_id]
    with resumed_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sources WHERE processed=1").fetchone()[0] == 2
        assert {row[0] for row in db.execute("SELECT status FROM jobs")} == {"completed"}
    archives = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "calls").glob("*.json")
    ]
    assert len(archives) == 2
    assert all(record["status"] == "completed" for record in archives)
