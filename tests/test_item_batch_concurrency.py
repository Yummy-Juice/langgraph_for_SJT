from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest


def _batch_state(item_ids: list[str]) -> dict:
    from sjt_system.workflow.item_batch import MERGED_MAPS

    cell_id = "cell-a"
    state = {
        "run_id": "batch-test",
        "blueprint": {
            "cells": [{"cell_id": cell_id, "facet_id": "facet-a"}],
            "slots": [
                {"specification_id": item_id, "blueprint_cell_id": cell_id}
                for item_id in item_ids
            ],
        },
        "test_specification": {"target": "facet-a"},
        "item_specifications": [
            {
                "specification_id": item_id,
                "blueprint_cell_id": cell_id,
                "target_dimension_id": "facet-a",
            }
            for item_id in item_ids
        ],
        "item_pool": [],
        "item_generation_completed_ids": [],
        "blueprint_progress": {cell_id: {"generated": 0, "accepted": 0}},
        "context_usage": {},
        "execution_history": [],
        "errors": [],
        "rejected_items": [],
        "step_count": 0,
        "max_steps": 50,
    }
    state.update({key: {} for key in MERGED_MAPS})
    return state


def _slot_ids(slots: list[tuple[dict, dict]]) -> list[str]:
    return [str(specification["specification_id"]) for _, specification in slots]


def test_pending_slots_preserves_order_and_uses_explicit_completion_ids() -> None:
    from sjt_system.workflow.item_batch import pending_slots

    state = _batch_state(["slot-1", "slot-2", "slot-3", "slot-4"])
    state.pop("item_generation_completed_ids")
    state["blueprint_progress"]["cell-a"]["generated"] = 2
    assert _slot_ids(pending_slots(state)) == ["slot-3", "slot-4"]

    explicit = deepcopy(state)
    explicit["item_generation_completed_ids"] = ["slot-2"]
    explicit["item_pool"] = [{"item_id": "slot-1"}]
    explicit["blueprint_progress"]["cell-a"]["generated"] = 99
    assert _slot_ids(pending_slots(explicit)) == ["slot-3", "slot-4"]


def test_item_batch_fans_out_all_slots_and_merges_partial_successes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from sjt_system.workflow import item_batch

    item_ids = ["slot-good", "slot-failed", "slot-raises"]
    state = _batch_state(item_ids)
    state["item_generation_completed_ids"] = ["already-complete"]
    state["step_count"] = 3
    state["max_steps"] = 4
    state["item_pool"] = [{"item_id": "existing", "version": 1}]
    state["blueprint_progress"]["cell-a"] = {"generated": 7, "accepted": 2}
    state["context_usage"] = {"routine": 5}
    state["execution_history"] = [{"event_id": "prior-event", "node": "prior"}]
    state["errors"] = [{"item_id": "prior-error"}]
    original = deepcopy(state)
    monkeypatch.setattr(
        item_batch,
        "scoped_output",
        lambda category, default: tmp_path / category,
    )

    async def exercise() -> dict:
        started: list[str] = []
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def develop(
            local_state,
            cell,
            specification,
            *,
            checkpoint_path,
        ):
            item_id = str(specification["specification_id"])
            started.append(item_id)
            if len(started) == len(item_ids):
                all_started.set()
            await release.wait()
            if item_id == "slot-raises":
                raise RuntimeError("child slot failed before producing state")

            local = deepcopy(dict(local_state))
            local["execution_history"] = [
                {
                    "event_id": "shared-child-event",
                    "node": "execute",
                    "item_version": 1,
                }
            ]
            local["errors"] = (
                [{"item_id": item_id, "message": "slot failed"}]
                if item_id == "slot-failed"
                else []
            )
            if item_id == "slot-good":
                local["item_specifications"] = [
                    {**specification, "generation_status": "complete"}
                ]
                local["item_pool"] = [
                    *local_state["item_pool"],
                    {"item_id": item_id, "version": 1},
                ]
                local["item_pattern_profiles"] = {
                    **local_state["item_pattern_profiles"],
                    item_id: {"context_category": "routine"},
                }
                local["item_skeletons"] = {
                    **local_state["item_skeletons"],
                    item_id: {"item_id": item_id, "version": 1},
                }
                local["item_history"] = {
                    **local_state["item_history"],
                    item_id: [{"item_id": item_id, "version": 1}],
                }
                local["item_lineage"] = {
                    **local_state["item_lineage"],
                    item_id: [{"item_id": item_id, "version": 1}],
                }
            else:
                local["status"] = "failed"
            return {
                "item_id": item_id,
                "cell_id": cell["cell_id"],
                "state": local,
                "consumed_steps": 2 if item_id == "slot-good" else 3,
                "failed": item_id == "slot-failed",
                "progress_delta": (
                    {"generated": 1, "accepted": 1}
                    if item_id == "slot-good"
                    else {"generated": 99, "accepted": 99}
                ),
            }

        monkeypatch.setattr(item_batch, "_develop_slot", develop)
        task = asyncio.create_task(item_batch.execute_item_generation_batch(state))
        all_fanned_out = False
        try:
            await asyncio.wait_for(all_started.wait(), timeout=1)
            all_fanned_out = True
        except TimeoutError:
            pass
        finally:
            release.set()
        result = await asyncio.wait_for(task, timeout=2)
        assert all_fanned_out
        assert started == item_ids
        return result

    result = asyncio.run(exercise())
    update = result["state_update"]

    assert state == original
    assert result["summary"].startswith("Concurrent fixed-slot batch: 3 slots, 2 failures")
    assert update["status"] == "failed"
    assert update["item_generation_completed_ids"] == [
        "already-complete",
        "slot-good",
    ]
    assert [row["item_id"] for row in update["item_pool"]] == [
        "existing",
        "slot-good",
    ]
    assert update["item_specifications"][
        0
    ]["generation_status"] == "complete"
    assert update["blueprint_progress"]["cell-a"] == {
        "generated": 8,
        "accepted": 3,
    }
    assert update["context_usage"] == {"routine": 6}
    assert update["step_count"] == 9
    assert update["item_history"]["slot-good"] == [
        {"item_id": "slot-good", "version": 1}
    ]
    assert update["item_lineage"]["slot-good"][0]["version"] == 1
    assert update["item_skeletons"]["slot-good"]["version"] == 1

    child_events = [
        event for event in update["execution_history"] if event.get("batch_item_id")
    ]
    assert {event["batch_item_id"] for event in child_events} == {
        "slot-good",
        "slot-failed",
    }
    assert len({event["event_id"] for event in child_events}) == len(child_events)
    assert any(error.get("item_id") == "slot-failed" for error in update["errors"])
    assert any(
        error.get("item_id") == "slot-raises"
        and "child slot failed" in error.get("message", "")
        for error in update["errors"]
    )

    resumable_state = {**original, **update}
    assert _slot_ids(item_batch.pending_slots(resumable_state)) == [
        "slot-failed",
        "slot-raises",
    ]


def test_item_batch_completes_all_slots_past_legacy_step_limit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from sjt_system.workflow import item_batch

    item_ids = ["slot-a", "slot-b", "slot-c"]
    state = _batch_state(item_ids)
    state["step_count"] = 12
    state["max_steps"] = 1
    started: list[str] = []
    monkeypatch.setattr(
        item_batch,
        "scoped_output",
        lambda category, _default: tmp_path / category,
    )

    async def develop(local_state, cell, specification, *, checkpoint_path):
        item_id = str(specification["specification_id"])
        started.append(item_id)
        local = deepcopy(dict(local_state))
        local["step_count"] = int(local.get("step_count") or 0) + 2
        local["item_specifications"] = [
            {**specification, "generation_status": "complete"}
        ]
        local["item_pool"] = [
            *local_state["item_pool"], {"item_id": item_id, "version": 1}
        ]
        local["item_pattern_profiles"] = {
            **local_state["item_pattern_profiles"],
            item_id: {"context_category": "routine"},
        }
        local["execution_history"] = [
            {
                "event_id": f"{item_id}-completed",
                "node": "execute",
                "event_type": "completed",
            }
        ]
        return {
            "item_id": item_id,
            "cell_id": cell["cell_id"],
            "state": local,
            "consumed_steps": 2,
            "failed": False,
            "progress_delta": {"generated": 1, "accepted": 1},
        }

    monkeypatch.setattr(item_batch, "_develop_slot", develop)
    result = asyncio.run(item_batch.execute_item_generation_batch(state))
    update = result["state_update"]

    assert set(started) == set(item_ids)
    assert update["item_generation_completed_ids"] == item_ids
    assert {row["item_id"] for row in update["item_pool"]} == set(item_ids)
    assert update["blueprint_progress"]["cell-a"] == {
        "generated": 3,
        "accepted": 3,
    }
    assert update["step_count"] == 19
    assert update["status"] == "running"
    assert state["step_count"] == 12


def test_develop_slot_runs_the_isolated_item_microflow_in_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from sjt_system.workflow import execution_node, interaction_nodes, item_batch, item_nodes

    state = _batch_state(["slot-a"])
    state.update({"item_development_mode": "automatic", "status": "running"})
    cell, specification = item_batch.pending_slots(state)[0]
    calls: list[str] = []

    async def execute(local_state):
        calls.append("execute")
        return {
            "pending_action": "generate_item",
            "current_item": {
                "item_id": specification["specification_id"],
                "version": 1,
            },
        }

    def automatic_approval(local_state):
        calls.append("automatic_approval")
        return {"automatic_approved": True}

    def commit(local_state):
        calls.append("commit")
        return {"committed": True}

    def prepare_review(local_state):
        calls.append("review")
        return {"current_item_review": {"findings": []}}

    def accept(local_state):
        calls.append("accept")
        return {
            "item_pool": [{"item_id": specification["specification_id"], "version": 1}],
            "status": "running",
        }

    monkeypatch.setattr(execution_node, "execute_node", execute)
    monkeypatch.setattr(interaction_nodes, "automatic_approval_node", automatic_approval)
    monkeypatch.setattr(interaction_nodes, "commit_node", commit)
    monkeypatch.setattr(item_nodes, "prepare_item_review_node", prepare_review)
    monkeypatch.setattr(item_nodes, "accept_item_node", accept)
    monkeypatch.setattr(item_batch, "emit_progress", lambda event: None)

    outcome = asyncio.run(
        item_batch._develop_slot(
            state,
            cell,
            specification,
            checkpoint_path=tmp_path / "slot-a.json",
        )
    )

    assert calls == ["execute", "automatic_approval", "commit", "review", "accept"]
    assert not outcome["failed"]
    assert outcome["state"]["item_pool"][0]["item_id"] == "slot-a"


def test_develop_slot_resumes_a_durable_midflight_result_without_regeneration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from sjt_system.workflow import execution_node, interaction_nodes, item_batch, item_nodes

    state = _batch_state(["slot-resume"])
    state.update({"item_development_mode": "automatic", "status": "running"})
    state["max_steps"] = 1
    cell, specification = item_batch.pending_slots(state)[0]
    checkpoint_path = tmp_path / "slot-resume.json"
    calls: list[str] = []
    real_write = item_batch.write_json_atomic

    async def execute(local_state):
        calls.append("execute")
        return {
            "pending_action": "generate_item",
            "current_item": {"item_id": "slot-resume", "version": 1},
        }

    def automatic_approval(local_state):
        calls.append("automatic_approval")
        return {"automatic_approved": True}

    def commit(local_state):
        calls.append("commit")
        return {"committed": True}

    def prepare_review(local_state):
        calls.append("review")
        return {"current_item_review": {"findings": []}}

    def accept(local_state):
        calls.append("accept")
        return {
            "item_pool": [{"item_id": "slot-resume", "version": 1}],
            "status": "running",
        }

    class _CheckpointPersisted(BaseException):
        pass

    def persist_then_pause(path, payload):
        real_write(path, payload)
        if payload["next_node"] == "review":
            raise _CheckpointPersisted

    monkeypatch.setattr(execution_node, "execute_node", execute)
    monkeypatch.setattr(interaction_nodes, "automatic_approval_node", automatic_approval)
    monkeypatch.setattr(interaction_nodes, "commit_node", commit)
    monkeypatch.setattr(item_nodes, "prepare_item_review_node", prepare_review)
    monkeypatch.setattr(item_nodes, "accept_item_node", accept)
    monkeypatch.setattr(item_batch, "emit_progress", lambda event: None)
    monkeypatch.setattr(item_batch, "write_json_atomic", persist_then_pause)

    with pytest.raises(_CheckpointPersisted):
        asyncio.run(
            item_batch._develop_slot(
                state,
                cell,
                specification,
                checkpoint_path=checkpoint_path,
            )
        )

    saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert saved["next_node"] == "review"
    assert saved["state"]["current_item"] == {
        "item_id": "slot-resume",
        "version": 1,
    }
    assert saved["source"]["protocol"] == "fixed_slot_concurrent_v1"

    monkeypatch.setattr(item_batch, "write_json_atomic", real_write)
    outcome = asyncio.run(
        item_batch._develop_slot(
            state,
            cell,
            specification,
            checkpoint_path=checkpoint_path,
        )
    )

    assert calls == ["execute", "automatic_approval", "commit", "review", "accept"]
    assert not outcome["failed"]
    assert outcome["state"]["item_pool"] == [
        {"item_id": "slot-resume", "version": 1}
    ]
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["next_node"] == "done"
