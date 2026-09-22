"""Single experiment orchestration; the reusable unit for a future batch runner."""
from contextlib import contextmanager
from copy import deepcopy
import os

from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

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
        completed = store.read("progress.json")["completed_stages"]
        for phase in ("A", "shared", "B", "C"):
            if phase in completed:
                continue
            key = {"A": "A/round_01", "B": "B/round_01"}.get(phase, phase)
            store.progress(stage=phase, status="running", error=None)
            print(f"\n===== 实验 {store.root.name} · {phase} =====", flush=True)
            try:
                with store.timer(key), model_environment(store.config.model_id), \
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
                        # New graph/checkpoint lineage; shared state and files remain immutable.
                        with run_context(state["run_id"]), workflow_model_scope(store):
                            await self.workflow_driver(store, phase, state)
                        select_final(store)
                completed.append(phase)
                store.progress(completed_stages=completed)
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
            for key in store.forms():
                print(f"\n===== 独立评估 · {key} =====", flush=True)
                await evaluate_form(store, key, self.evaluation_model)
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
