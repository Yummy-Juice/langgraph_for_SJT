"""Framework-neutral driver for advancing the workflow to its next pause."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.types import Command

from sjt_system.runtime.checkpoint import save_run_checkpoint
from sjt_system.runtime.progress import progress_callback
from sjt_system.runtime.telemetry import run_context as telemetry_run_context


UpdateCallback = Callable[[str, dict[str, Any], dict[str, Any]], None]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class WorkflowTurn:
    """Result of running from one user interaction to the next."""

    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    updates: list[dict[str, Any]] = field(default_factory=list)
    runtime_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.interrupt is None


class WorkflowRunError(RuntimeError):
    """A stream failure that retains every state update already received."""

    def __init__(
        self,
        error: Exception,
        *,
        state: Mapping[str, Any],
        updates: list[dict[str, Any]],
        runtime_events: list[dict[str, Any]],
    ) -> None:
        super().__init__(str(error))
        self.original_error = error
        self.state = dict(state)
        self.updates = list(updates)
        self.runtime_events = list(runtime_events)


def extract_interrupt_payload(interrupt_update: object) -> dict[str, Any]:
    """Extract the first user-facing payload from a LangGraph interrupt."""

    interrupts = (
        list(interrupt_update)
        if isinstance(interrupt_update, (list, tuple))
        else [interrupt_update]
    )
    if not interrupts:
        raise ValueError("LangGraph 返回了空的 interrupt 更新")
    payload = getattr(interrupts[0], "value", interrupts[0])
    if not isinstance(payload, Mapping):
        raise ValueError("LangGraph interrupt 负载必须是对象")
    return dict(payload)


async def run_until_pause(
    workflow: Any,
    graph_input: dict[str, Any] | Command,
    current_state: Mapping[str, Any],
    *,
    checkpoint_root: Path | None = None,
    on_update: UpdateCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> WorkflowTurn:
    """Advance a compiled graph until it completes or requests user input."""

    state = dict(current_state)
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("工作流状态缺少有效 run_id")

    with telemetry_run_context(run_id):
        return await _run_until_pause_impl(
            workflow=workflow,
            graph_input=graph_input,
            state=state,
            checkpoint_root=checkpoint_root,
            on_update=on_update,
            on_progress=on_progress,
        )


async def _run_until_pause_impl(
    workflow: Any,
    graph_input: dict[str, Any] | Command,
    state: Mapping[str, Any],
    *,
    checkpoint_root: Path | None,
    on_update: UpdateCallback | None,
    on_progress: ProgressCallback | None,
) -> WorkflowTurn:
    """Advance a compiled graph until it completes or requests user input."""
    updates: list[dict[str, Any]] = []
    runtime_events: list[dict[str, Any]] = []

    def record_progress(event: dict[str, Any]) -> None:
        runtime_event = dict(event)
        runtime_events.append(runtime_event)
        if on_progress is not None:
            on_progress(runtime_event)

    config = {"configurable": {"thread_id": run_id}}
    try:
        with progress_callback(record_progress):
            async for chunk in workflow.astream(
                graph_input,
                config=config,
                stream_mode="updates",
            ):
                if "__interrupt__" in chunk:
                    return WorkflowTurn(
                        state=state,
                        interrupt=extract_interrupt_payload(
                            chunk["__interrupt__"]
                        ),
                        updates=updates,
                        runtime_events=runtime_events,
                    )

                for node, raw_update in chunk.items():
                    if not isinstance(raw_update, Mapping):
                        continue
                    update = dict(raw_update)
                    state.update(update)
                    record = {"node": str(node), "update": update}
                    updates.append(record)
                    if checkpoint_root is not None:
                        save_run_checkpoint(
                            state,
                            checkpoint_root=Path(checkpoint_root),
                        )
                    if on_update is not None:
                        on_update(str(node), update, dict(state))
    except Exception as exc:
        raise WorkflowRunError(
            exc,
            state=state,
            updates=updates,
            runtime_events=runtime_events,
        ) from exc

    return WorkflowTurn(
        state=state,
        updates=updates,
        runtime_events=runtime_events,
    )
