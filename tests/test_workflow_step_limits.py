from __future__ import annotations

import asyncio
import json

import pytest


@pytest.mark.parametrize("legacy_limit", [None, 0, 1, 100])
def test_initial_state_ignores_legacy_max_steps_keyword(legacy_limit) -> None:
    from sjt_system.state import create_initial_state

    state = create_initial_state("unbounded workflow", max_steps=legacy_limit)

    assert "max_steps" not in state
    assert state["step_count"] == 0


def test_router_ignores_legacy_max_steps_after_counter_passes_it() -> None:
    from sjt_system.state import create_initial_state
    from sjt_system.workflow.router import router_node

    state = create_initial_state("router without a step ceiling", max_steps=1)
    state["step_count"] = 2
    state["max_steps"] = 1

    result = asyncio.run(router_node(state))

    assert result["route"]["next_action"] == "clarify_requirements"
    assert result.get("status", state["status"]) != "failed"


def test_execute_ignores_legacy_max_steps_and_increments_counter(monkeypatch) -> None:
    from sjt_system.state import create_initial_state
    from sjt_system.workflow import execution_node

    state = create_initial_state("execute without a step ceiling", max_steps=1)
    state["step_count"] = 2
    state["max_steps"] = 1
    state["route"] = {"next_action": "simulate_responses"}
    calls: list[str] = []

    async def execute_agent(route, _state):
        calls.append(route["next_action"])
        return {"state_update": {}, "summary": "stubbed execution"}

    monkeypatch.setattr(execution_node, "execute_agent", execute_agent)
    monkeypatch.setattr(
        execution_node,
        "validate_item_bank_owned_fields",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        execution_node,
        "validate_item_agent_update",
        lambda *_args, **_kwargs: None,
    )

    result = asyncio.run(execution_node.execute_node(state))

    assert calls == ["simulate_responses"]
    assert result["step_count"] == 3
    assert result["pending_action"] == "simulate_responses"
    assert result["execution_history"][-1]["event_type"] == "completed"


def test_legacy_checkpoint_drops_max_steps_and_preserves_committed_state(
    tmp_path,
) -> None:
    from sjt_system.runtime.checkpoint import (
        CHECKPOINT_SCHEMA_VERSION,
        load_run_checkpoint,
        prepare_retry_state,
        save_run_checkpoint,
    )
    from sjt_system.state import create_initial_state

    state = create_initial_state("legacy checkpoint", max_steps=1)
    state["max_steps"] = 1
    state["step_count"] = 17
    state["item_pool"] = [{"item_id": "committed-item", "version": 2}]
    state["item_history"] = {
        "committed-item": [{"event": "accepted", "version": 2}]
    }

    path = save_run_checkpoint(state, checkpoint_root=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "max_steps" not in saved["state"]

    legacy_envelope = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": state["run_id"],
        "saved_at": "2026-10-02T00:00:00Z",
        "is_terminal": False,
        "state": state,
    }
    path.write_text(
        json.dumps(legacy_envelope, ensure_ascii=False), encoding="utf-8"
    )

    loaded = load_run_checkpoint(path)
    resumed = prepare_retry_state(
        {"run_id": state["run_id"]}, checkpoint_root=tmp_path
    )

    assert "max_steps" not in loaded["state"]
    assert "max_steps" not in resumed
    assert resumed["step_count"] == 17
    assert resumed["item_pool"] == [
        {"item_id": "committed-item", "version": 2}
    ]
    assert resumed["item_history"] == {
        "committed-item": [{"event": "accepted", "version": 2}]
    }


def test_application_graph_uses_unbounded_recursion_and_langgraph_finishes_loop() -> None:
    from math import isinf

    from langgraph.graph import END, START, StateGraph

    from sjt_system.workflow.graph import build_sjt_graph

    assert isinf(build_sjt_graph().config["recursion_limit"])

    graph_builder = StateGraph(dict)
    graph_builder.add_node(
        "count",
        lambda state: {"count": state["count"] + 1},
    )
    graph_builder.add_edge(START, "count")
    graph_builder.add_conditional_edges(
        "count",
        lambda state: "done" if state["count"] >= 40 else "count",
        {"count": "count", "done": END},
    )
    graph = graph_builder.compile().with_config(
        {"recursion_limit": float("inf")}
    )

    result = graph.invoke({"count": 0})

    assert result["count"] == 40

