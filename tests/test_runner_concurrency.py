from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


class _MemoryStore:
    def __init__(self, root: Path, *, completed=(), forms=()) -> None:
        self.root = SimpleNamespace(name="concurrency-test")
        self.config = SimpleNamespace(model_id="experiment-model")
        self._root_path = root
        self._files: dict[str, object] = {}
        self._form_order = list(forms)
        self.progress_state = {"completed_stages": list(completed)}
        self.progress_events: list[dict] = []
        self.report_count = 0

    def read(self, relative: str):
        if relative == "progress.json":
            return deepcopy(self.progress_state)
        return deepcopy(self._files.get(relative))

    def write(self, relative: str, value) -> None:
        self._files[relative] = deepcopy(value)

    def save_form(self, key: str, items, **metadata) -> None:
        self.write(f"{key}/form.json", {"items": deepcopy(items), **metadata})

    def path(self, relative: str) -> Path:
        return self._root_path / relative

    def timer(self, key: str):
        return nullcontext()

    def progress(self, **updates) -> None:
        self.progress_events.append(deepcopy(updates))
        self.progress_state.update(deepcopy(updates))

    def forms(self) -> list[str]:
        return list(self._form_order)


def _item(item_id: str) -> dict:
    return {"item_id": item_id, "text": f"item {item_id}"}


def _install_memory_runner(monkeypatch, store: _MemoryStore, *, driver):
    from experiments.system_comparison import runner

    monkeypatch.setattr(runner, "initial_workflow_state", lambda _: {"run_id": "workflow-run"})
    monkeypatch.setattr(runner, "drive_workflow", driver)
    monkeypatch.setattr(runner, "workflow_model_scope", lambda _: nullcontext())

    def freeze_shared(target_store, state):
        target_store.write("shared/frozen_state.json", state)

    monkeypatch.setattr(runner, "freeze_shared", freeze_shared)
    monkeypatch.setattr(runner, "write_report", lambda target_store: setattr(
        target_store, "report_count", target_store.report_count + 1
    ))
    return runner


def test_develop_starts_a_with_shared_then_b_and_c_and_keeps_c_work_independent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from experiments.system_comparison import runner

    store = _MemoryStore(tmp_path)
    initial_started: list[str] = []
    downstream_started: list[str] = []
    initial_barrier = asyncio.Event()
    downstream_barrier = asyncio.Event()
    release_shared = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    release_c = asyncio.Event()
    a_finished = asyncio.Event()
    shared_frozen = asyncio.Event()
    b_baseline_saved = asyncio.Event()
    c_model_finished = asyncio.Event()
    selected = asyncio.Event()
    calls: list[str] = []

    async def generate_a(target_store, model):
        calls.append("A-model")
        initial_started.append("A")
        if len(initial_started) == 2:
            initial_barrier.set()
        await initial_barrier.wait()
        await release_a.wait()
        target_store.save_form("A/round_01", [_item("a")])
        a_finished.set()

    async def generate_b(target_store, frozen_state, model):
        calls.append("B-model")
        assert shared_frozen.is_set()
        downstream_started.append("B")
        if len(downstream_started) == 2:
            downstream_barrier.set()
        await downstream_barrier.wait()
        await release_b.wait()
        target_store.save_form("B/round_01", [_item("b")])

    async def drive(target_store, phase, state):
        if phase == "shared":
            calls.append("shared-model")
            initial_started.append("shared")
            if len(initial_started) == 2:
                initial_barrier.set()
            await initial_barrier.wait()
            await release_shared.wait()
            return {"run_id": state["run_id"], "shared_ready": True}
        assert phase == "C"
        calls.append("C-model")
        assert shared_frozen.is_set()
        downstream_started.append("C")
        if len(downstream_started) == 2:
            downstream_barrier.set()
        await downstream_barrier.wait()
        await release_c.wait()
        c_model_finished.set()
        return state

    def freeze_shared(target_store, state):
        target_store.write("shared/frozen_state.json", state)
        shared_frozen.set()

    def select_final(target_store):
        assert b_baseline_saved.is_set()
        assert target_store.read("C/baseline/form.json") is not None
        selected.set()

    from experiments.system_comparison import runner as runner_module

    monkeypatch.setattr(runner_module, "initial_workflow_state", lambda _: {"run_id": "workflow-run"})
    monkeypatch.setattr(runner_module, "drive_workflow", drive)
    monkeypatch.setattr(runner_module, "workflow_model_scope", lambda _: nullcontext())
    monkeypatch.setattr(runner_module, "freeze_shared", freeze_shared)
    monkeypatch.setattr(runner_module, "generate_a", generate_a)
    monkeypatch.setattr(runner_module, "generate_b", generate_b)
    monkeypatch.setattr(runner_module, "select_final", select_final)
    monkeypatch.setattr(runner_module, "write_report", lambda _: None)

    original_save_form = store.save_form

    def save_form(key, items, **metadata):
        original_save_form(key, items, **metadata)
        if key == "C/baseline":
            b_baseline_saved.set()

    store.save_form = save_form

    async def exercise() -> None:
        task = asyncio.create_task(
            runner.ExperimentRunner(store, workflow_driver=drive).develop()
        )
        await asyncio.wait_for(initial_barrier.wait(), timeout=1)
        assert set(initial_started) == {"A", "shared"}
        assert not a_finished.is_set()
        assert not shared_frozen.is_set()

        release_shared.set()
        await asyncio.wait_for(downstream_barrier.wait(), timeout=1)
        assert set(downstream_started) == {"B", "C"}
        assert not b_baseline_saved.is_set()
        assert not selected.is_set()

        release_c.set()
        await asyncio.wait_for(c_model_finished.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not b_baseline_saved.is_set()
        assert not selected.is_set()

        release_a.set()
        await asyncio.wait_for(a_finished.wait(), timeout=1)
        release_b.set()
        assert await asyncio.wait_for(task, timeout=2) is True

    asyncio.run(exercise())

    assert calls == ["A-model", "shared-model", "B-model", "C-model"]
    assert b_baseline_saved.is_set()
    assert selected.is_set()
    running_stages = [
        event["stage"]
        for event in store.progress_events
        if event.get("status") == "running"
    ]
    assert set(running_stages[:2]) == {"A", "shared"}
    assert set(running_stages[2:]) == {"B", "C"}


def test_develop_drains_and_records_other_stages_after_failure_then_resumes_only_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from experiments.system_comparison import runner

    store = _MemoryStore(tmp_path)
    a_started = asyncio.Event()
    shared_started = asyncio.Event()
    release_shared = asyncio.Event()
    calls: list[str] = []

    async def generate_a(target_store, model):
        calls.append("A")
        a_started.set()
        if calls.count("A") == 1:
            raise RuntimeError("A stage failed")
        target_store.save_form("A/round_01", [_item("a")])

    async def generate_b(target_store, frozen_state, model):
        calls.append("B")
        target_store.save_form("B/round_01", [_item("b")])

    async def drive(target_store, phase, state):
        calls.append(phase)
        if phase == "shared":
            shared_started.set()
            await release_shared.wait()
            return {"run_id": state["run_id"], "shared_ready": True}
        assert phase == "C"
        return state

    def select_final(target_store):
        assert target_store.read("C/baseline/form.json") is not None

    _install_memory_runner(monkeypatch, store, driver=drive)
    monkeypatch.setattr(runner, "generate_a", generate_a)
    monkeypatch.setattr(runner, "generate_b", generate_b)
    monkeypatch.setattr(runner, "select_final", select_final)

    async def first_attempt() -> None:
        task = asyncio.create_task(
            runner.ExperimentRunner(store, workflow_driver=drive).develop()
        )
        await asyncio.wait_for(a_started.wait(), timeout=1)
        await asyncio.wait_for(shared_started.wait(), timeout=1)
        assert not task.done()
        release_shared.set()
        with pytest.raises(RuntimeError, match="A stage failed"):
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(first_attempt())

    assert store.progress_state["status"] == "failed"
    assert store.progress_state["completed_stages"] == ["shared", "B", "C"]
    assert store.report_count == 1
    assert calls == ["A", "shared", "B", "C"]

    assert asyncio.run(
        runner.ExperimentRunner(store, workflow_driver=drive).develop()
    ) is True

    assert calls == ["A", "shared", "B", "C", "A"]
    assert store.progress_state["completed_stages"] == ["A", "shared", "B", "C"]
    assert store.progress_state["status"] == "development_complete"
    assert store.read("A/round_01/form.json") is not None
    assert store.report_count == 1


def test_evaluate_runs_disjoint_forms_together_and_waits_for_alias_and_overlap_producers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from experiments.system_comparison import runner

    keys = [
        "A/round_01",
        "A/round_02",
        "B/round_01",
        "B/round_02",
        "C/round_01",
    ]
    store = _MemoryStore(tmp_path, completed=("C",), forms=keys)
    forms = {
        "A/round_01": [_item("one"), _item("two")],
        "A/round_02": [_item("one"), _item("two")],
        "B/round_01": [_item("three")],
        "B/round_02": [_item("three")],
        "C/round_01": [_item("one"), _item("three")],
    }
    for key, items in forms.items():
        store.write(f"{key}/form.json", {"items": items})

    _install_memory_runner(monkeypatch, store, driver=lambda *args: None)
    starts: list[str] = []
    completed_producers: set[str] = set()
    both_producers_started = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    alias_a_started = asyncio.Event()
    alias_b_started = asyncio.Event()

    async def evaluate_form(target_store, key, model):
        starts.append(key)
        if key == "A/round_01":
            if {"A/round_01", "B/round_01"}.issubset(starts):
                both_producers_started.set()
            await release_a.wait()
            completed_producers.add(key)
            return
        if key == "B/round_01":
            if {"A/round_01", "B/round_01"}.issubset(starts):
                both_producers_started.set()
            await release_b.wait()
            completed_producers.add(key)
            return
        if key == "A/round_02":
            assert "A/round_01" in completed_producers
            alias_a_started.set()
            return
        if key == "B/round_02":
            assert "B/round_01" in completed_producers
            alias_b_started.set()
            return
        assert key == "C/round_01"
        assert completed_producers == {"A/round_01", "B/round_01"}
        assert alias_a_started.is_set()
        assert alias_b_started.is_set()

    monkeypatch.setattr(runner, "evaluate_form", evaluate_form)

    async def exercise() -> None:
        task = asyncio.create_task(runner.ExperimentRunner(store).evaluate())
        await asyncio.wait_for(both_producers_started.wait(), timeout=1)
        assert starts[:2] == ["A/round_01", "B/round_01"]
        assert not alias_a_started.is_set()
        assert not alias_b_started.is_set()
        assert "C/round_01" not in starts

        release_a.set()
        await asyncio.wait_for(alias_a_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert "C/round_01" not in starts
        assert not alias_b_started.is_set()

        release_b.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(exercise())

    assert starts == [
        "A/round_01",
        "B/round_01",
        "A/round_02",
        "B/round_02",
        "C/round_01",
    ]
    assert store.progress_state["status"] == "completed"
    assert store.report_count == 1


def test_develop_scopes_model_environment_once_and_restores_caller_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import os

    from experiments.system_comparison import runner

    monkeypatch.setenv("MODEL_ID", "caller-model")
    store = _MemoryStore(tmp_path)
    observed_models: list[tuple[str, str | None]] = []
    scope_enters: list[str] = []
    scope_exits: list[str] = []

    def record(stage: str) -> None:
        observed_models.append((stage, os.environ.get("MODEL_ID")))

    async def generate_a(target_store, model):
        record("A")
        await asyncio.sleep(0)
        target_store.save_form("A/round_01", [_item("a")])

    async def generate_b(target_store, frozen_state, model):
        record("B")
        await asyncio.sleep(0)
        target_store.save_form("B/round_01", [_item("b")])

    async def drive(target_store, phase, state):
        record(phase)
        await asyncio.sleep(0)
        return state

    real_model_environment = runner.model_environment

    @contextmanager
    def tracked_model_environment(model_id):
        scope_enters.append(model_id)
        with real_model_environment(model_id):
            yield
        scope_exits.append(model_id)

    _install_memory_runner(monkeypatch, store, driver=drive)
    monkeypatch.setattr(runner, "model_environment", tracked_model_environment)
    monkeypatch.setattr(runner, "generate_a", generate_a)
    monkeypatch.setattr(runner, "generate_b", generate_b)
    monkeypatch.setattr(runner, "select_final", lambda _: None)

    assert asyncio.run(
        runner.ExperimentRunner(store, workflow_driver=drive).develop()
    ) is True

    assert scope_enters == ["experiment-model"]
    assert scope_exits == ["experiment-model"]
    assert {stage for stage, _ in observed_models} == {"A", "shared", "B", "C"}
    assert all(model_id == "experiment-model" for _, model_id in observed_models)
    assert os.environ["MODEL_ID"] == "caller-model"
