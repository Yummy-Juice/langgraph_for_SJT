"""Interactive adapter around the real graph; static stage boundaries are opt-in."""
import asyncio
from copy import deepcopy
import inspect
from time import monotonic

from langgraph.types import Command

from sjt_system.authoring.bank import build_item_bank_freeze_update
from sjt_system.authoring.construct_registry import construct_selection_from_profile
from sjt_system.runtime.progress import progress_callback
from sjt_system.state import create_initial_state
from sjt_system.workflow.graph import build_sjt_graph

from .checkpoints import DiskSaver
from .evaluation import _copy_tree_rebased


class ExperimentPaused(Exception):
    """A saved human decision, not a failed scientific outcome."""


def initial_workflow_state(store):
    config = store.config
    state = create_initial_state("独立A/B/C实验：开发共享候选题库", max_steps=10000)
    profile = store.read("shared/construct_profile.json")
    specification = {"construct_selection": construct_selection_from_profile(profile),
                     "target_population": config.target_population,
                     "final_item_count": config.final_item_count, "output_language": "zh-CN"}
    sample = store.read("participants/development.json")
    virtual_sample_config = deepcopy(sample["config"])
    virtual_sample_config["model_id"] = (
        config.virtual_respondent_model_id or config.model_id
    )
    virtual_sample_config["experiment_protocol"] = {
        "model": virtual_sample_config["model_id"],
        "sample": "development",
        "role": "virtual_respondent",
    }
    state.update({"run_id": store.root.name + "-workflow", "requirements_confirmed": True,
                  "confirmed_requirement_fields": list(specification), "test_specification": specification,
                  "construct_profile": profile,
                  "virtual_sample_config": virtual_sample_config,
                  "virtual_respondents": deepcopy(sample["respondents"]),
                  "psychometric_plateau_patience": config.plateau_patience,
                  "psychometric_plateau_min_delta": config.plateau_min_delta})
    return state


def capture_rounds(store, state):
    """Called after every graph update, before the next round can mutate items."""
    current = {i["item_id"]: i for i in state.get("frozen_item_bank", [])}
    history = state.get("psychometric_iteration_history") or []
    for record in history:
        index = int(record.get("analysis_round") or 0)
        key = f"C/round_{index:02d}"
        if index < 1 or store.read(f"{key}/development/snapshot_complete.json"):
            continue
        ids = record.get("form_item_ids") or []
        if not ids or any(i not in current for i in ids):
            continue
        # Do not reconstruct an old unsaved round from the current bank.
        if index != state.get("psychometric_analysis_round"):
            raise ValueError("历史轮次缺少完整题目快照，不能用当前版本冒充")
        snapshot = store.read(f"{key}/checkpoint.json") or deepcopy(state)
        bank = {i["item_id"]: i for i in snapshot["frozen_item_bank"]}
        store.write(f"{key}/checkpoint.json", snapshot)
        store.save_form(key, [bank[i] for i in ids],
                        phase="first_metric_assembly" if index == 1 else "post_repair_assembly",
                        form_status=record.get("form_status"),
                        development_record=record)
        store.write(f"{key}/development/item_bank.json", list(bank.values()))
        store.write(f"{key}/development/item_statistics.json", snapshot.get("item_statistics"))
        store.write(f"{key}/development/repair_history.json", snapshot.get("psychometric_repair_history", []))
        store.write(f"{key}/development/human_decisions.json", store.read("C/human_decisions.json", []))
        store.write(f"{key}/development/iteration_record.json", record)
        if not store.read(f"{key}/cost.json"):
            from .reporting import cost_summary
            cumulative = cost_summary(store, "C")
            cumulative["wall_seconds"] = store.elapsed("C")
            previous = store.read(f"C/round_{index - 1:02d}/cost.json", {}).get("cumulative_c_cost", {})
            delta = {}
            for field in ("calls", "total_tokens", "prompt_tokens", "completion_tokens", "cached_input_tokens",
                          "wall_seconds", "model_duration_ms", "fee"):
                value, before = cumulative.get(field), previous.get(field, 0)
                delta[field] = value - before if value is not None and before is not None else None
            store.write(f"{key}/cost.json", {**delta, "cumulative_c_cost": cumulative,
                        "cost_scope": "C增量，截至本轮组卷快照；不含共享开发及独立评估"})
        retained = best_development_form(store)
        store.write(f"{key}/retained_form.json", retained or {"status": "no_eligible_form"})
        manifest = snapshot.get("virtual_response_data_ref")
        if manifest:
            from pathlib import Path
            _copy_tree_rebased(Path(manifest).parent, store.path(f"{key}/development/responses"))
        store.write(f"{key}/development/snapshot_complete.json", {"complete": True})


def stopping_reason(store, phase, state, next_nodes):
    action = (state.get("route") or {}).get("next_action")
    if phase == "shared" and "execute" in next_nodes and action == "simulate_responses":
        return "shared_bank_ready"
    if phase != "C":
        return None
    # Bound development before another expensive diagnostic/repair stage.
    if (state.get("psychometric_plateau_status") or {}).get("reached"):
        return "plateau"
    if int(state.get("psychometric_analysis_round") or 0) >= store.config.max_repair_rounds + 1:
        return "round_limit"
    if "execute" in next_nodes and action in {"assemble_test", "review_test", "rescore_test", "generate_reports"}:
        return "development_finished"
    return None


async def drive_workflow(store, phase, state, *, decision_provider=None, graph_factory=None):
    from cli_app import prompt_user_decision, print_runtime_progress, print_heartbeat, print_user_progress_event
    provider = decision_provider or prompt_user_decision
    checkpoint_key = "shared" if phase == "shared" else "C"
    with DiskSaver(store.path(f"{checkpoint_key}/graph.sqlite")) as saver:
        graph = (graph_factory or build_sjt_graph)(checkpointer=saver, interrupt_before=["execute"])
        cfg = {"configurable": {"thread_id": state["run_id"]}, "recursion_limit": 10000}
        snapshot = await graph.aget_state(cfg)
        if snapshot.values and snapshot.values.get("status") == "failed":
            # Only a NEW resume invocation retries. Never loop on a failure
            # within this invocation, or reset already completed upstream work.
            failed_action = (snapshot.values.get("route") or {}).get("next_action")
            events = snapshot.values.get("execution_history") or []
            if not events or events[-1].get("node") != "execute":
                raise RuntimeError("不是可重试的执行节点失败，请检查已保存的错误；不会重置整个实验")
            predecessor = None
            async for candidate in graph.aget_state_history(cfg):
                if (candidate.next == ("execute",) and candidate.values.get("status") != "failed"
                    and (candidate.values.get("route") or {}).get("next_action") == failed_action):
                    predecessor = candidate
                    break
            if predecessor is None:
                raise RuntimeError("缺少失败步骤之前的可靠检查点；拒绝从头重跑")
            # A new checkpoint branch avoids replaying pending writes containing
            # the failed node's output. Router edges schedule the same action.
            await graph.aupdate_state(predecessor.config, dict(predecessor.values), as_node="router")
            recoveries = store.read(f"{checkpoint_key}/recovery_events.json", [])
            recoveries.append({"action": failed_action, "source_checkpoint": predecessor.config,
                               "errors": snapshot.values.get("errors"), "mode": "retry_failed_execute_only"})
            store.write(f"{checkpoint_key}/recovery_events.json", recoveries)
            snapshot = await graph.aget_state(cfg)
        graph_input = None if snapshot.values else state
        result = dict(snapshot.values or state)
        displayed = set()
        while True:
            snapshot = await graph.aget_state(cfg)
            if snapshot.values:
                result = dict(snapshot.values)
                if phase == "C":
                    capture_rounds(store, result)
                store.write(f"{checkpoint_key}/checkpoint.json", result)
                reason = stopping_reason(store, phase, result, snapshot.next)
                if reason:
                    store.write(f"{checkpoint_key}/stop.json", {"reason": reason})
                    return result
                if result.get("status") == "failed":
                    raise RuntimeError(str((result.get("errors") or [])[-1:]))
                if not snapshot.next:
                    if result.get("status") == "stopped":
                        raise ExperimentPaused("工作流已停止，原始人工处置保留在检查点")
                    return result
                interrupts = [i for task in snapshot.tasks for i in task.interrupts]
                if interrupts:
                    payload = interrupts[0].value
                    store.write(f"{checkpoint_key}/pending_interaction.json", payload)
                    answer = provider(payload)
                    if inspect.isawaitable(answer):
                        answer = await answer
                    decisions = store.read(f"{checkpoint_key}/human_decisions.json", [])
                    decisions.append({"type": payload.get("type"), "payload": payload, "answer": answer})
                    store.write(f"{checkpoint_key}/human_decisions.json", decisions)
                    # Do not submit stop to the graph: retain the exact interrupt for resume.
                    if isinstance(answer, dict) and answer.get("decision") == "stop":
                        raise ExperimentPaused("用户暂停；恢复后继续当前确认，不重复已完成生成")
                    graph_input = Command(resume=answer)
            previous = dict(result)
            with progress_callback(print_runtime_progress):
                stream = graph.astream(graph_input, cfg, stream_mode="updates")
                task = None
                started = monotonic()
                try:
                    iterator = stream.__aiter__()
                    while True:
                        if task is None:
                            task = asyncio.create_task(anext(iterator))
                        done, _ = await asyncio.wait({task}, timeout=20)
                        if not done:
                            print_heartbeat((result.get("route") or {}).get("next_action"), monotonic() - started)
                            continue
                        try:
                            update_chunk = task.result()
                        except StopAsyncIteration:
                            break
                        finally:
                            task = None
                        for node, update in update_chunk.items():
                            if node == "__interrupt__" or not isinstance(update, dict):
                                continue
                            previous = dict(result)
                            result.update(update)
                            store.write(f"{checkpoint_key}/checkpoint.json", result)
                            if phase == "C":
                                capture_rounds(store, result)
                            for event in update.get("execution_history", []):
                                event_id = event.get("event_id")
                                if event_id and event_id not in displayed:
                                    print_user_progress_event(event, update, result, previous)
                                    displayed.add(event_id)
                finally:
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    await stream.aclose()
            graph_input = None


def freeze_shared(store, state):
    state = deepcopy(state)
    state.update(build_item_bank_freeze_update(state))
    if len(state["frozen_item_bank"]) != store.config.final_item_count * 2:
        raise ValueError("共享候选题库不足2倍题量；保存断点后人工处理")
    store.write("shared/blueprint.json", state["blueprint"])
    store.write("shared/initial_bank.json", state["frozen_item_bank"])
    store.write("shared/frozen_state.json", state)
    return state


def best_development_form(store):
    eligible = []
    from sjt_system.evaluation.form_metrics import (
        form_quality_summary,
        whole_form_objective_improves,
    )
    for key in store.forms():
        if not key.startswith("C/round_"):
            continue
        form = store.read(f"{key}/form.json")
        record = form.get("development_record") or {}
        metrics = record.get("form_metrics") or {}
        quality = form_quality_summary(metrics)
        if len(form["items"]) == store.config.final_item_count and quality.get("eligible_for_best_so_far"):
            eligible.append((key, form, metrics, quality))
    if not eligible:
        return None
    incumbent = None
    for candidate in eligible:
        if incumbent is None or whole_form_objective_improves(
            candidate[2], incumbent[2], min_delta=store.config.plateau_min_delta
        ):
            incumbent = candidate
    if incumbent is None:
        incumbent = eligible[0]
    key, form, metrics, quality = incumbent
    return {"source_round": key, "fingerprint": form["fingerprint"],
                                 "items": form["items"], "development_quality": quality,
                                 "selection_source": "development_only", "is_provisional": True}


def select_final(store):
    final = best_development_form(store)
    if final is None:
        raise ValueError("C没有题量完整且达到稳定性门槛的测验；保留所有阶段结果")
    store.write("C/final.json", final)
    return final["source_round"]
