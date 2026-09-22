"""Real-model adapter for the isolated post-baseline lab.

The adapter reuses the production agents and production single-item
retest/evaluation functions, but builds a private state projection and writes
all local-retest response artifacts below the lab output directory. It does
not enter the production LangGraph or mutate a production checkpoint.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _item_fingerprint(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        result = model_dump()
        if isinstance(result, dict):
            return deepcopy(result)
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        result = dict_method()
        if isinstance(result, dict):
            return deepcopy(result)
    raise ValueError("LLM 返回结果不是对象")


class LiveLLMRepairEngine:
    """Use the configured production diagnosis/repair agents in isolation."""

    def __init__(self, *, snapshot: dict[str, Any], output_dir: Path, config: Any):
        self.snapshot = deepcopy(snapshot)
        self.output_dir = Path(output_dir)
        self.config = config
        self.calls = {
            "diagnosis": 0,
            "revision": 0,
            "local_retest": 0,
        }
        self._diagnoses: dict[str, dict[str, Any]] = {}

        # The isolated CLI's --model is an experiment-level override. Apply it
        # before importing the cached production agents so diagnosis, repair,
        # and local virtual response use one declared model.
        configured_model = getattr(self.config, "model_id", None)
        if configured_model:
            for role in (
                "PSYCHOMETRIC_DIAGNOSIS_MODEL_ID",
                "PSYCHOMETRIC_ITEM_REPAIR_MODEL_ID",
            ):
                os.environ[role] = str(configured_model)

        response_ref = self.snapshot.get("virtual_response_data_ref")
        if not isinstance(response_ref, str) or not response_ref:
            raise ValueError("隔离正式 LLM 流程缺少已有开发作答 manifest")
        self.source_manifest_path = Path(response_ref).resolve()
        if not self.source_manifest_path.is_file():
            raise ValueError(f"已有开发作答 manifest 不存在：{self.source_manifest_path}")

        # Import lazily so offline regression tests never instantiate a model.
        from sjt_system.agent.agent_factory import (
            psychometric_item_repair_agent,
            psychometric_repair_diagnosis_agent,
        )

        self.diagnosis_agent = psychometric_repair_diagnosis_agent
        self.repair_agent = psychometric_item_repair_agent

    def restore_history(self, item_history: list[dict[str, Any]] | None) -> None:
        """Keep the method for checkpoint compatibility; form data is not restored."""
        return None

    def manifest(self) -> dict[str, Any]:
        try:
            from sjt_system.agent.agent_factory import PSYCHOMETRIC_REASONING_ROLE_MANIFEST

            return deepcopy(PSYCHOMETRIC_REASONING_ROLE_MANIFEST)
        except Exception as exc:  # noqa: BLE001 - manifest must not block a run
            return {"manifest_error": str(exc)}

    def _find_cell(self, item: dict[str, Any]) -> dict[str, Any]:
        blueprint = self.snapshot.get("blueprint_detail") or self.snapshot.get("blueprint") or {}
        for cell in blueprint.get("cells") or []:
            if isinstance(cell, dict) and str(cell.get("cell_id")) == str(item.get("blueprint_cell_id")):
                return deepcopy(cell)
        return {
            "cell_id": item.get("blueprint_cell_id"),
            "facet_id": item.get("target_dimension_id"),
            "behavior_id": item.get("behavior_evidence_id"),
            "mechanism_id": item.get("mechanism_id"),
            "situation_id": item.get("situation_id"),
            "domain": item.get("context_category"),
            "activation_mechanism": item.get("activation_mechanism") or item.get("construct_rationale"),
        }

    def _find_specification(self, item: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item.get("item_id") or "")
        cell_id = str(cell.get("cell_id") or item.get("blueprint_cell_id") or "")
        for spec in self.snapshot.get("item_specifications") or []:
            if not isinstance(spec, dict):
                continue
            if str(spec.get("specification_id") or spec.get("item_id") or "") == item_id:
                return deepcopy(spec)
        for spec in self.snapshot.get("item_specifications") or []:
            if isinstance(spec, dict) and str(spec.get("blueprint_cell_id") or "") == cell_id:
                return deepcopy(spec)
        return {
            "specification_id": item_id,
            "blueprint_cell_id": cell_id,
            "target_dimension_id": item.get("target_dimension_id") or cell.get("facet_id"),
            "context_category": item.get("context_category") or cell.get("domain"),
            "behavior_evidence_id": cell.get("behavior_id") or item.get("behavior_evidence_id"),
            "mechanism_id": cell.get("mechanism_id") or item.get("mechanism_id"),
            "situation_id": cell.get("situation_id") or item.get("situation_id"),
            "activation_mechanism": cell.get("activation_mechanism") or item.get("construct_rationale"),
            "core_tension": item.get("construct_rationale") or "",
            "behavioral_anchors": {},
            "avoid_scenario_patterns": [],
            "avoid_response_patterns": [],
        }

    def _find_skeleton(self, item: dict[str, Any], specification: dict[str, Any]) -> dict[str, Any]:
        skeletons = self.snapshot.get("item_skeletons") or {}
        specification_id = str(specification.get("specification_id") or item.get("item_id") or "")
        candidate = skeletons.get(specification_id) if isinstance(skeletons, dict) else None
        if isinstance(candidate, dict):
            return deepcopy(candidate)
        return {
            "behavioral_tension": specification.get("core_tension") or item.get("construct_rationale") or "",
            "option_structure": [
                {"behavioral_level": option.get("behavioral_level"), "behavioral_tendency": option.get("text")}
                for option in item.get("response_options") or []
                if isinstance(option, dict)
            ],
        }

    def _build_state(
        self,
        current_item: dict[str, Any],
        *,
        frozen_items: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        item_bank_id: str | None = None,
        item_bank_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        item_id = str(current_item.get("item_id") or "")
        items = [deepcopy(item) for item in self.snapshot.get("items") or [] if isinstance(item, dict)]
        replaced = False
        for index, item in enumerate(items):
            if str(item.get("item_id")) == item_id:
                items[index] = deepcopy(current_item)
                replaced = True
                break
        if not replaced:
            items.append(deepcopy(current_item))
        cell = self._find_cell(current_item)
        specification = self._find_specification(current_item, cell)
        skeleton_id = str(specification.get("specification_id") or item_id)
        skeletons = deepcopy(self.snapshot.get("item_skeletons") or {})
        skeletons.setdefault(skeleton_id, self._find_skeleton(current_item, specification))
        statistics = {
            str(item.get("item_id")): deepcopy(item.get("item_statistics") or item.get("metrics") or {})
            for item in items
            if item.get("item_id")
        }
        blueprint = deepcopy(self.snapshot.get("blueprint_detail") or self.snapshot.get("blueprint") or {})
        if not isinstance(blueprint, dict):
            blueprint = {}
        blueprint.setdefault("cells", [cell])
        response_ref = self.snapshot.get("virtual_response_data_ref")
        if not response_ref:
            raise ValueError("正式 LLM 局部复测缺少 virtual_response_data_ref")
        sample_config = deepcopy(self.snapshot.get("virtual_sample_config") or {})
        configured_model = getattr(self.config, "model_id", None)
        if configured_model:
            sample_config["model_id"] = str(configured_model)
        frozen = deepcopy(frozen_items if frozen_items is not None else items)
        resolved_run_id = run_id or "isolated-post-baseline-lab"
        resolved_item_bank_id = item_bank_id or "isolated-post-baseline-lab"
        resolved_fingerprint = item_bank_fingerprint or _item_fingerprint(frozen)
        return {
            "run_id": resolved_run_id,
            "frozen_item_bank": frozen,
            "item_pool": frozen,
            "selected_items": frozen,
            "item_statistics": statistics,
            "blueprint": blueprint,
            "construct_profile": deepcopy(self.snapshot.get("construct_profile") or blueprint.get("construct_profile_snapshot") or {}),
            "item_specifications": deepcopy(self.snapshot.get("item_specifications") or [specification]),
            "item_skeletons": skeletons,
            "test_specification": deepcopy(self.snapshot.get("test_specification") or {}),
            "current_item": deepcopy(current_item),
            "current_blueprint_cell": cell,
            "current_item_specification": specification,
            "current_item_review": None,
            "item_reviews": {},
            "item_history": {},
            "psychometric_repair_history": [],
            "virtual_response_data_ref": response_ref,
            "previous_virtual_response_data_ref": response_ref,
            "virtual_sample_config": sample_config,
            "virtual_respondents": deepcopy(self.snapshot.get("virtual_respondents") or []),
            "item_bank_id": resolved_item_bank_id,
            "item_bank_version": 1,
            "item_bank_fingerprint": resolved_fingerprint,
        }

    async def _invoke(
        self,
        agent: Any,
        input_data: dict[str, Any],
        *,
        job_label: str,
    ) -> dict[str, Any]:
        from sjt_system.workflow.executor import _ainvoke_model

        timeout = float(getattr(self.config, "request_timeout_seconds", 300.0))
        result = await _ainvoke_model(
            agent,
            {"input_data": input_data},
            job_label=job_label,
            timeout_seconds=timeout,
            max_attempts=1,
        )
        return _as_dict(result)

    async def diagnose(self, item: dict[str, Any]) -> dict[str, Any]:
        self.calls["diagnosis"] += 1
        from sjt_system.evaluation.diagnosis import (
            build_construct_diagnosis_evidence,
            build_deterministic_defer_advice,
            build_deterministic_forced_vts_repair_advice,
            build_deterministic_target_gradient_repair_advice,
            build_psychometric_agent_input,
            normalize_target_gradient_repair_advice,
            validate_atomic_repair_advice,
        )

        state = self._build_state(item)
        evidence = build_construct_diagnosis_evidence(
            state,
            str(item["item_id"]),
            revision_round=int(item.get("revision_round") or 1),
        )
        packet = build_psychometric_agent_input(evidence)
        advice = await self._invoke(
            self.diagnosis_agent,
            packet,
            job_label=f"psychometric_repair_diagnosis / {item['item_id']}",
        )
        advice.setdefault("item_id", str(item["item_id"]))
        try:
            validate_atomic_repair_advice(advice, evidence)
        except ValueError as exc:
            fallback = build_deterministic_forced_vts_repair_advice(
                evidence,
                validation_error=str(exc),
            )
            fallback_status = "deterministic_forced_vts_fallback"
            if fallback is None:
                fallback = build_deterministic_target_gradient_repair_advice(
                    evidence,
                    validation_error=str(exc),
                )
                fallback_status = "deterministic_target_gradient_fallback"
            if fallback is None:
                fallback = build_deterministic_defer_advice(
                    evidence,
                    validation_error=str(exc),
                )
                fallback_status = "validation_fallback_defer"
            if fallback is None:
                raise
            advice = fallback
            advice["item_id"] = str(item["item_id"])
            advice["diagnosis_status"] = fallback_status
            advice["diagnosis_validation_error"] = str(exc)
            validate_atomic_repair_advice(advice, evidence)
        if advice.get("decision") == "repair":
            try:
                advice = normalize_target_gradient_repair_advice(advice, evidence)
                validate_atomic_repair_advice(advice, evidence, require_target_gradient_task=True)
            except ValueError as exc:
                fallback = build_deterministic_defer_advice(
                    evidence,
                    validation_error=str(exc),
                )
                if fallback is None:
                    raise
                advice = fallback
                advice["item_id"] = str(item["item_id"])
                advice["diagnosis_status"] = "validation_fallback_defer"
                advice["diagnosis_validation_error"] = str(exc)
                validate_atomic_repair_advice(advice, evidence)
        root_id = str(item.get("root_item_id", item["item_id"]))
        self._diagnoses[root_id] = {"advice": deepcopy(advice), "evidence": deepcopy(evidence)}
        return {
            "status": "ok",
            "decision": advice.get("decision"),
            "advice": advice,
            "evidence": evidence,
        }

    async def revise(
        self,
        item: dict[str, Any],
        attempt: int,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls["revision"] += 1
        from sjt_system.authoring.context import (
            build_psychometric_repair_generation_context,
            build_psychometric_repair_model_state,
        )
        from sjt_system.authoring.items import (
            canonicalize_item_agent_update,
            validate_item_agent_update,
        )
        from sjt_system.evaluation.diagnosis import (
            normalize_atomic_option_patch_scope,
            repair_tasks_from_advice,
            validate_atomic_item_patch,
        )
        from sjt_system.workflow.executor import MAX_ITEM_OUTPUT_CANDIDATES, _normalize_item_repair_result

        root_id = str(item.get("root_item_id", item["item_id"]))
        record = self._diagnoses.get(root_id)
        if not record:
            raise ValueError(f"缺少题目 {root_id} 的正式诊断结果")
        evidence = record["evidence"]
        advice = deepcopy(record["advice"])
        packet = build_psychometric_agent_input(evidence)
        tasks = repair_tasks_from_advice(advice)
        if not tasks:
            raise ValueError("正式诊断没有可执行的 atomic repair task")

        working_item = deepcopy(item)
        for task_index, task in enumerate(tasks, start=1):
            task_advice = deepcopy(advice)
            task_advice["selected_diagnosis_id"] = str(
                task.get("diagnosis_id") or f"D{task_index}"
            )
            task_advice["atomic_edit"] = deepcopy(task.get("atomic_edit") or {})
            task_advice["repair_tasks"] = [deepcopy(task)]
            state = self._build_state(working_item)
            input_data = {
                "action": "revise_item",
                "state": build_psychometric_repair_model_state(state),
                "generation_context": build_psychometric_repair_generation_context(state),
                "blocking_findings": [],
                "repair_source": "psychometric_diagnosis",
                "atomic_repair_advice": task_advice,
                "normal_constraints": packet.get("normal_constraints"),
                "target_construct_constraints": packet.get("target_construct_constraints"),
                "item_content": packet.get("item_content"),
                "option_evidence": packet.get("option_evidence"),
                "option_score_comparisons": packet.get("option_score_comparisons"),
                "target_gradient_plan": packet.get("target_gradient_plan"),
                "local_retest_feedback": deepcopy(feedback),
                "local_retest_round": attempt,
                "required_context_category": state["current_item_specification"].get("context_category"),
                "validation_feedback": None,
                "previous_invalid_candidate": None,
            }
            last_error: ValueError | None = None
            task_succeeded = False
            for output_attempt in range(MAX_ITEM_OUTPUT_CANDIDATES):
                result: Any = None
                try:
                    result = _normalize_item_repair_result(
                        await self._invoke(
                            self.repair_agent,
                            input_data,
                            job_label=(
                                f"revise_item / {item.get('item_id')} / "
                                f"{task_advice['selected_diagnosis_id']} / local-{attempt}"
                            ),
                        )
                    )
                    update = result.get("state_update") if isinstance(result, dict) else None
                    if not isinstance(update, dict):
                        raise ValueError("正式返修 Agent 输出缺少有效 state_update")
                    update = normalize_atomic_option_patch_scope(update, task_advice)
                    validate_atomic_item_patch(update, working_item, task_advice)
                    canonical = canonicalize_item_agent_update(
                        "revise_item",
                        deepcopy(update),
                        specification=state.get("test_specification"),
                        blueprint_cell=state.get("current_blueprint_cell"),
                        item_specification=state.get("current_item_specification"),
                        previous_item=working_item,
                    )
                    validate_item_agent_update(
                        "revise_item",
                        canonical,
                        target_item_id=item.get("item_id"),
                        target_blueprint_cell_id=item.get("blueprint_cell_id"),
                        specification=state.get("test_specification"),
                        blueprint_cell=state.get("current_blueprint_cell"),
                        item_specification=state.get("current_item_specification"),
                        previous_item=working_item,
                    )
                    working_item = deepcopy(canonical["current_item"])
                    task_succeeded = True
                    last_error = None
                    break
                except ValueError as exc:
                    last_error = exc
                    if output_attempt >= MAX_ITEM_OUTPUT_CANDIDATES - 1:
                        break
                    input_data = {
                        **input_data,
                        "validation_feedback": str(exc),
                        "previous_invalid_candidate": (
                            result.get("state_update")
                            if isinstance(result, dict)
                            else result
                        ),
                    }
            else:  # pragma: no cover - the range always executes at least once
                last_error = ValueError("返修输出重试没有执行")
            if not task_succeeded:
                raise ValueError(
                    f"正式返修任务 {task_advice['selected_diagnosis_id']} 未生成有效候选：{last_error}"
                ) from last_error
        return deepcopy(working_item)

    async def local_retest(self, item: dict[str, Any], attempt: int, action: str) -> dict[str, Any]:
        self.calls["local_retest"] += 1
        from sjt_system.evaluation.psychometrics import evaluate_single_item_candidate
        from sjt_system.evaluation.simulation import run_virtual_response_simulation

        state = self._build_state(item)
        item_id = str(item["item_id"])
        version = int(item["version"])
        run_fragment = "".join(character if character.isalnum() else "-" for character in item_id)[-48:]
        local_run_id = f"{self.snapshot.get('source') or 'source'}-isolated-{run_fragment}-v{version}"
        local_bank_id = f"isolated-local-{run_fragment}-v{version}"
        local_fingerprint = _item_fingerprint([item])
        local_state = self._build_state(
            item,
            frozen_items=[deepcopy(item)],
            run_id=local_run_id,
            item_bank_id=local_bank_id,
            item_bank_fingerprint=local_fingerprint,
        )
        simulation = await run_virtual_response_simulation(
            local_state,
            output_root=self.output_dir / "live_virtual_responses",
            request_timeout_seconds=float(getattr(self.config, "request_timeout_seconds", 300.0)),
        )
        state_update = simulation.get("state_update") or {}
        manifest_path_value = state_update.get("virtual_response_data_ref")
        if not isinstance(manifest_path_value, str):
            raise ValueError("正式单题局部复测没有返回 response manifest")
        manifest_path = Path(manifest_path_value).resolve()
        local_manifest = _read_json(manifest_path)
        records = _read_jsonl(manifest_path.parent / "sjt_responses.jsonl")
        target_meta = local_manifest.get("target_form_retest") or {}
        target_path = target_meta.get("path") if isinstance(target_meta, dict) else None
        target_records = _read_jsonl(Path(target_path).resolve()) if isinstance(target_path, str) else []
        if not records:
            raise ValueError("正式单题局部复测没有返回虚拟作答记录")
        metrics = evaluate_single_item_candidate(self._build_state(item), item, records)
        qualification = metrics.get("qualification") or {}
        metrics["passed"] = bool(qualification.get("qualified") is True)
        return {
            "status": "ok",
            "metrics": metrics,
            "simulation": {
                "manifest_path": str(manifest_path),
                "sjt_response_count": len(records),
                "target_retest_response_count": len(target_records),
                "new_calls": simulation.get("summary", {}).get("scheduled_sjt_api_calls"),
                "target_retest_new_calls": simulation.get("summary", {}).get("scheduled_target_form_retest_api_calls"),
            },
        }
