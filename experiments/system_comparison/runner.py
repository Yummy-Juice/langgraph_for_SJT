"""Single experiment orchestration; the reusable unit for a future batch runner."""
from contextlib import contextmanager
from copy import deepcopy
import asyncio
import json
import os

from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context
from sjt_system.runtime.concurrency import gather_all

from .generation import generate_a, generate_b
from .evaluation import evaluate_form
from .reporting import write_report
from .models import workflow_model_scope
from .workflow import initial_workflow_state, drive_workflow, freeze_shared, select_final, ExperimentPaused


@contextmanager
def model_environment(model_id):
    previous = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = model_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous


class ExperimentRunner:
    def __init__(self, store, *, a_model=None, b_model=None, evaluation_model=None,
                 workflow_driver=drive_workflow):
        self.store = store
        self.a_model, self.b_model = a_model, b_model
        self.evaluation_model = evaluation_model
        self.workflow_driver = workflow_driver

    async def develop(self):
        store = self.store
        phase_order = ("A", "shared", "B", "C")
        completed = set(store.read("progress.json")["completed_stages"])
        tasks = {}

        async def develop_phase(phase):
            if phase in {"B", "C"} and "shared" in tasks:
                await tasks["shared"]
            key = {"A": "A/round_01", "B": "B/round_01"}.get(phase, phase)
            store.progress(stage=phase, status="running", error=None)
            print(f"\n===== 实验 {store.root.name} · {phase} =====", flush=True)
            with store.timer(key), \
                 output_scope(store.path("runtime"), telemetry=store.path(f"{key}/telemetry")), \
                 run_context(store.root.name + "-" + phase):
                if phase == "A":
                    if not store.read("A/round_01/form.json"):
                        await generate_a(store, self.a_model)
                elif phase == "shared":
                    state = initial_workflow_state(store)
                    with run_context(state["run_id"]), workflow_model_scope(store):
                        state = await self.workflow_driver(store, phase, state)
                    freeze_shared(store, state)
                elif phase == "B":
                    if not store.read("B/round_01/form.json"):
                        await generate_b(store, store.read("shared/frozen_state.json"), self.b_model)
                    form = store.read("B/round_01/form.json")
                    store.save_form("C/baseline", form["items"], source="B/round_01", phase="theory_baseline")
                    store.write("C/baseline/checkpoint.json", {"source": "B/round_01/checkpoint.json", "status": "frozen"})
                    store.write("C/baseline/cost.json", {"reused_from": "B/round_01", "wall_seconds": 0})
                else:
                    state = deepcopy(store.read("shared/frozen_state.json"))
                    # Only the workflow mutates scoped agent roles. Independent
                    # prompt baselines use explicit models in the same environment.
                    with run_context(state["run_id"]), workflow_model_scope(store):
                        await self.workflow_driver(store, phase, state)
                    if "B" in tasks:
                        await tasks["B"]
                    select_final(store)
            completed.add(phase)
            store.progress(completed_stages=[p for p in phase_order if p in completed])

        try:
            # Set the shared process environment once; overlapping context
            # managers must not restore MODEL_ID while another call is active.
            with model_environment(store.config.model_id):
                for phase in phase_order:
                    if phase not in completed:
                        tasks[phase] = asyncio.create_task(develop_phase(phase))
                await gather_all(*tasks.values())
        except ExperimentPaused as exc:
            store.progress(status="paused", error=str(exc))
            write_report(store)
            return False
        except (KeyboardInterrupt, EOFError):
            store.progress(status="paused", error="用户中断；检查点和已完成作答已保留")
            write_report(store)
            return False
        except Exception as exc:
            store.progress(status="failed", error=str(exc))
            write_report(store)
            raise
        store.progress(stage="evaluation", status="development_complete")
        return True

    async def evaluate(self):
        store = self.store
        if "C" not in store.read("progress.json")["completed_stages"]:
            raise ValueError("请先完成或处置C开发；独立评估不能提前反馈给仍在开发的任务")
        store.progress(stage="evaluation", status="evaluating", error=None)
        try:
            groups = []
            by_snapshot = {}
            for key in store.forms():
                form = store.read(f"{key}/form.json")
                snapshot = json.dumps(form["items"], ensure_ascii=False, sort_keys=True)
                if snapshot in by_snapshot:
                    groups[by_snapshot[snapshot]]["keys"].append(key)
                    continue
                item_snapshots = {
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    for item in form["items"]
                }
                # Unchanged items must reuse earlier independent evaluation
                # records. Aliases and lineage consumers wait on their producer;
                # disjoint forms have no reason to queue behind one another.
                dependencies = [
                    index for index, group in enumerate(groups)
                    if item_snapshots.intersection(group["items"])
                ]
                by_snapshot[snapshot] = len(groups)
                groups.append({"keys": [key], "items": item_snapshots, "dependencies": dependencies})
            tasks = []

            async def evaluate_group(group):
                await gather_all(*(tasks[index] for index in group["dependencies"]))
                first, *aliases = group["keys"]
                print(f"\n===== 独立评估 · {first} =====", flush=True)
                await evaluate_form(store, first, self.evaluation_model)
                await gather_all(*(evaluate_form(store, key, self.evaluation_model) for key in aliases))

            for group in groups:
                tasks.append(asyncio.create_task(evaluate_group(group)))
            await gather_all(*tasks)
            store.progress(stage="finished", status="completed")
        except (KeyboardInterrupt, EOFError):
            store.progress(status="paused", error="独立评估暂停，恢复时补齐缺失作答")
        except Exception as exc:
            store.progress(status="evaluation_failed", error=str(exc))
            raise
        finally:
            write_report(store)

    async def run(self):
        if "C" in self.store.read("progress.json")["completed_stages"] or await self.develop():
            await self.evaluate()
