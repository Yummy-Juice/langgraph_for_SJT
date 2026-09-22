from __future__ import annotations

import asyncio
import copy
import html
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"complete", "infeasible", "technical_failure"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"不是有效 JSON：{path}：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _read_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"不是有效 JSON：{path}：{exc}") from exc


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed", "qualified", "合格", "通过"}:
            return True
        if lowered in {"false", "no", "fail", "failed", "unqualified", "不合格", "未通过"}:
            return False
    return None


@dataclass(frozen=True)
class LabConfig:
    """Controls only the isolated lab; production defaults are not imported."""

    max_item_repair_attempts: int = 3
    max_model_retries: int = 2
    max_replacements_per_slot: int = 2
    max_repair_rounds: int = 3
    max_concurrency: int = 8
    engine_mode: str = "offline"
    request_timeout_seconds: float = 300.0
    model_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LabConfig":
        raw = raw or {}
        values = {}
        for name, default in asdict(cls()).items():
            value = raw.get(name, default)
            if name in {"max_item_repair_attempts", "max_model_retries", "max_replacements_per_slot", "max_repair_rounds", "max_concurrency"}:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = default
                value = max(1, value)
            elif name == "min_delta":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = default
                value = max(0.0, value)
            elif name == "request_timeout_seconds":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = default
                value = max(1.0, value)
            elif name == "engine_mode":
                value = str(value).strip().lower()
                if value not in {"offline", "llm"}:
                    value = default
            elif name == "model_id":
                value = str(value).strip() or None if value is not None else None
            values[name] = value
        return cls(**values)


class SnapshotError(ValueError):
    pass


def _find_snapshot_file(source: Path) -> Path:
    if source.is_file():
        return source
    if not source.is_dir():
        raise SnapshotError(f"找不到快照路径：{source}")
    candidates = [
        source / "post_baseline_snapshot.json",
        source / "baseline_snapshot.json",
        source / "checkpoint.json",
        source / "C" / "baseline" / "checkpoint.json",
        source / "C" / "round_01" / "checkpoint.json",
        source / "C" / "checkpoint.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SnapshotError(
        f"目录中没有可识别的快照：{source}。可显式提供 post_baseline_snapshot.json，"
        "或先用 export 命令生成标准化快照。"
    )


def _unwrap_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    for key in ("snapshot", "state", "checkpoint"):
        value = raw.get(key)
        if isinstance(value, dict) and any(k in value for k in ("items", "frozen_item_bank", "baseline_form", "selected_items")):
            return value
    return raw


def _extract_item_id(item: dict[str, Any]) -> str | None:
    for key in ("item_id", "id", "question_id", "题目编号"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _extract_cell_id(item: dict[str, Any]) -> str:
    for key in ("blueprint_cell_id", "cell_id", "measurement_unit_id", "blueprint_slot_id", "slot_id", "facet_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("blueprint_cell_id", "cell_id", "measurement_unit_id", "blueprint_slot_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return "cell-unassigned"


def _extract_passed(item: dict[str, Any]) -> bool | None:
    for key in ("passed", "qualified", "is_qualified", "item_passed"):
        result = _as_bool(item.get(key))
        if result is not None:
            return result
    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        for key in ("passed", "qualified", "all_gates_passed", "item_passed"):
            result = _as_bool(metrics.get(key))
            if result is not None:
                return result
    for key in ("item_statistics", "item_statistic", "quality_evaluation"):
        nested = item.get(key)
        if isinstance(nested, dict):
            result = _extract_passed(nested)
            if result is not None:
                return result
    return None


def _normalize_item(raw_item: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    item = _deepcopy(raw_item)
    item_id = _extract_item_id(item) or fallback_id
    item["item_id"] = item_id
    item["blueprint_cell_id"] = _extract_cell_id(item)
    item.setdefault("version", 1)
    item.setdefault("scenario", item.get("situation", item.get("context", "")))
    options = item.get("response_options", item.get("options", []))
    item["response_options"] = _deepcopy(options) if isinstance(options, list) else []
    item.setdefault("metrics", {})
    passed = _extract_passed(item)
    if passed is not None:
        item["metrics"]["passed"] = passed
    item.setdefault("status", "baseline")
    return item


def _extract_item_list(raw: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "frozen_item_bank", "item_bank", "candidate_items", "questions"):
        value = raw.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            result = []
            for key_id, value_item in value.items():
                if isinstance(value_item, dict):
                    value_item = _deepcopy(value_item)
                    value_item.setdefault("item_id", key_id)
                    result.append(value_item)
            if result:
                return result
    return []


def _extract_ids(raw: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    item_id = _extract_item_id(item)
                    if item_id:
                        result.append(item_id)
                elif item is not None:
                    result.append(str(item))
            if result:
                return result
        if isinstance(value, dict):
            result = [str(k) for k in value]
            if result:
                return result
    return []


def _normalize_blueprint(raw: dict[str, Any], items: list[dict[str, Any]], final_count: int | None) -> dict[str, Any]:
    blueprint = raw.get("blueprint") or raw.get("blueprint_spec") or raw.get("test_blueprint") or {}
    if not isinstance(blueprint, dict):
        blueprint = {}
    raw_cells = blueprint.get("cells") or raw.get("blueprint_cells") or []
    cells: list[dict[str, Any]] = []
    if isinstance(raw_cells, dict):
        raw_cells = list(raw_cells.values())
    if isinstance(raw_cells, list):
        for index, cell in enumerate(raw_cells, start=1):
            if isinstance(cell, str):
                cells.append({"cell_id": cell, "required_count": 1, "planned_retention_count": 1})
                continue
            if not isinstance(cell, dict):
                continue
            cell_id = str(cell.get("cell_id") or cell.get("id") or cell.get("slot_id") or f"cell-{index}")
            required = cell.get("required_count", cell.get("item_count", cell.get("count", 1)))
            try:
                required = max(1, int(required))
            except (TypeError, ValueError):
                required = 1
            cells.append({"cell_id": cell_id, "required_count": required, "planned_retention_count": required, **{k: _deepcopy(v) for k, v in cell.items() if k not in {"cell_id", "id", "slot_id", "required_count", "item_count", "count", "planned_retention_count"}}})
    if not cells:
        by_cell: dict[str, int] = {}
        for item in items:
            cell_id = item["blueprint_cell_id"]
            by_cell[cell_id] = by_cell.get(cell_id, 0) + 1
        for cell_id in sorted(by_cell):
            cells.append({"cell_id": cell_id, "required_count": 1, "planned_retention_count": 1})
    if final_count is None:
        final_count = sum(int(cell["required_count"]) for cell in cells)
    if final_count <= 0:
        final_count = len(cells)
    return {"cells": cells, "final_item_count": int(final_count)}


def normalize_snapshot(raw: dict[str, Any], source: str = "") -> dict[str, Any]:
    raw = _unwrap_snapshot(raw)
    source_items = _extract_item_list(raw)
    items = [_normalize_item(item, f"item-{index:03d}") for index, item in enumerate(source_items, start=1)]
    if not items:
        raise SnapshotError("快照中没有题目列表，无法从第一次组卷开始测试。")
    item_by_id = {item["item_id"]: item for item in items}
    baseline = raw.get("baseline_form") or raw.get("provisional_form") or raw.get("test_form") or {}
    if not isinstance(baseline, dict):
        baseline = {}
    baseline_ids = _extract_ids(baseline, "item_ids", "selected_item_ids", "items", "questions")
    if not baseline_ids:
        baseline_ids = _extract_ids(raw, "baseline_item_ids", "selected_item_ids", "current_form_item_ids")
    if not baseline_ids:
        raise SnapshotError("快照中没有第一次临时组卷的 item_ids，拒绝猜测基线问卷。")
    missing = [item_id for item_id in baseline_ids if item_id not in item_by_id]
    if missing:
        raise SnapshotError(f"第一次组卷引用了快照中不存在的题目：{missing[:5]}")
    final_count = baseline.get("final_item_count") or raw.get("final_item_count") or len(baseline_ids)
    blueprint = _normalize_blueprint(raw, items, int(final_count))
    reserve_ids = _extract_ids(raw, "reserve_item_ids", "reserve_items", "replenishment_items", "backup_items")
    reserve_ids = [item_id for item_id in reserve_ids if item_id in item_by_id and item_id not in baseline_ids]
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "created_at": raw.get("created_at", _now()),
        "blueprint": blueprint,
        "blueprint_detail": _deepcopy(raw.get("blueprint_detail") or raw.get("blueprint") or {}),
        "construct_profile": _deepcopy(raw.get("construct_profile") or {}),
        "item_specifications": _deepcopy(raw.get("item_specifications") or []),
        "item_skeletons": _deepcopy(raw.get("item_skeletons") or {}),
        "test_specification": _deepcopy(raw.get("test_specification") or {}),
        "virtual_response_data_ref": raw.get("virtual_response_data_ref"),
        "baseline_virtual_response_data_ref": raw.get("baseline_virtual_response_data_ref"),
        "virtual_respondents": _deepcopy(raw.get("virtual_respondents") or []),
        "response_paths": _deepcopy(raw.get("response_paths") or {}),
        "virtual_sample_config": _deepcopy(raw.get("virtual_sample_config") or {}),
        "baseline_form": {
            "item_ids": baseline_ids,
            "final_item_count": int(final_count),
        },
        "items": items,
        "reserve_item_ids": reserve_ids,
        "raw_metadata": {
            "source_schema_version": raw.get("schema_version"),
            "source_keys": sorted(str(key) for key in raw.keys()),
        },
    }
    return normalized


def load_snapshot(source: str | Path) -> dict[str, Any]:
    source_path = Path(source)
    # The production experiment stores a form and its evaluation as separate
    # files rather than as one portable checkpoint. Adapt that layout here,
    # without importing production workflow state or writing back to it.
    if source_path.is_dir() and (source_path / "form.json").exists():
        return _load_form_directory_snapshot(source_path)
    path = _find_snapshot_file(Path(source))
    return normalize_snapshot(_read_json(path), source=str(path))


def _load_form_directory_snapshot(form_dir: Path) -> dict[str, Any]:
    form = _read_json(form_dir / "form.json")
    form_items = form.get("items", [])
    if not isinstance(form_items, list) or not form_items:
        raise SnapshotError(f"form.json 中没有题目：{form_dir / 'form.json'}")
    baseline_ids = [
        str(item_id)
        for item_id in (_extract_item_id(raw_item) for raw_item in form_items if isinstance(raw_item, dict))
        if item_id
    ]

    # Prefer the 32-item development bank when the supplied directory is
    # C/baseline. This gives the isolated lab a real same-slot reserve pool.
    candidate_bank_path = form_dir.parent / "round_01" / "development" / "item_bank.json"
    development_stats_path = form_dir.parent / "round_01" / "development" / "item_statistics.json"
    if not candidate_bank_path.exists():
        candidate_bank_path = form_dir / "evaluation" / "items.json"
    candidate_items = _read_json_any(candidate_bank_path) if candidate_bank_path.suffix == ".json" else {}
    if isinstance(candidate_items, dict):
        candidate_items = candidate_items.get("items", [])
    if not isinstance(candidate_items, list) or not candidate_items:
        candidate_items = form_items
    stats: dict[str, Any] = {}
    baseline_stats: dict[str, Any] = {}
    stats_path = development_stats_path if development_stats_path.exists() else form_dir / "evaluation" / "item_statistics.json"
    if stats_path.exists():
        stats_payload = _read_json(stats_path)
        if isinstance(stats_payload, dict):
            stats = stats_payload
    baseline_stats_path = form_dir / "evaluation" / "item_statistics.json"
    if baseline_stats_path.exists():
        baseline_stats_payload = _read_json(baseline_stats_path)
        if isinstance(baseline_stats_payload, dict):
            baseline_stats = baseline_stats_payload
    enriched_items = []
    for index, raw_item in enumerate(candidate_items, start=1):
        if not isinstance(raw_item, dict):
            continue
        item = _deepcopy(raw_item)
        item_id = _extract_item_id(item) or f"item-{index:03d}"
        stat = (baseline_stats.get(item_id, {}) if item_id in baseline_ids else stats.get(item_id, {})) if isinstance(stats, dict) else {}
        if isinstance(stat, dict):
            item["item_statistics"] = _deepcopy(stat)
            qualification = stat.get("qualification", {})
            if isinstance(qualification, dict) and "qualified" in qualification:
                item.setdefault("metrics", {})["passed"] = bool(qualification["qualified"])
            elif "quality_evaluation" in stat:
                recommendation = stat.get("quality_evaluation", {}).get("recommendation")
                item.setdefault("metrics", {})["passed"] = recommendation in {"pass", "qualified", "retain"}
        if item_id in baseline_ids:
            item.setdefault("metrics", {})["passed"] = _extract_passed(item)
        enriched_items.append(item)

    blueprint_payload: dict[str, Any] = {}
    workflow_context: dict[str, Any] = {}
    workflow_checkpoint = form_dir.parent / "checkpoint.json"
    if workflow_checkpoint.exists():
        checkpoint = _read_json(workflow_checkpoint)
        workflow_context = checkpoint
        if isinstance(checkpoint.get("blueprint"), dict):
            blueprint_payload = checkpoint["blueprint"]
    analysis_manifest_path = form_dir / "evaluation" / "analysis_manifest.json"
    analysis_manifest: dict[str, Any] = {}
    if analysis_manifest_path.exists():
        analysis_value = _read_json(analysis_manifest_path)
        if isinstance(analysis_value, dict):
            analysis_manifest = analysis_value
    baseline_response_ref = (
        ((analysis_manifest.get("input_files") or {}).get("response_manifest") or {}).get("path")
        if isinstance(analysis_manifest, dict)
        else None
    )
    development_response_path = form_dir.parent / "round_01" / "development" / "responses" / "manifest.json"
    response_ref = (
        str(development_response_path.resolve())
        if development_response_path.is_file()
        else baseline_response_ref
    )
    virtual_sample_config: dict[str, Any] = {}
    virtual_respondents: list[dict[str, Any]] = []
    response_paths: dict[str, str] = {}
    if isinstance(response_ref, str) and Path(response_ref).is_file():
        response_manifest = _read_json(Path(response_ref))
        if isinstance(response_manifest, dict):
            virtual_sample_config = _deepcopy(response_manifest.get("virtual_sample_config") or {})
            profiles_path = response_manifest.get("score_profiles_path")
            if isinstance(profiles_path, str) and Path(profiles_path).is_file():
                profiles_payload = _read_json(Path(profiles_path))
                if isinstance(profiles_payload.get("profiles"), list):
                    virtual_respondents = _deepcopy(profiles_payload["profiles"])
            for key in (
                "score_profiles_path",
                "scoring_snapshot_path",
                "option_order_path",
                "source_manifest_path",
            ):
                value = response_manifest.get(key)
                if isinstance(value, str):
                    response_paths[key] = value
            for key in ("sample_size_per_condition", "conditions", "max_concurrency", "max_retries", "model_id"):
                if key in response_manifest and key not in virtual_sample_config:
                    virtual_sample_config[key] = _deepcopy(response_manifest[key])
    raw = {
        "schema_version": SCHEMA_VERSION,
        "blueprint": blueprint_payload,
        "blueprint_detail": blueprint_payload,
        "construct_profile": blueprint_payload.get("construct_profile_snapshot") if isinstance(blueprint_payload, dict) else {},
        "item_specifications": workflow_context.get("item_specifications") or [],
        "item_skeletons": workflow_context.get("item_skeletons") or {},
        "test_specification": workflow_context.get("test_specification") or {},
        "virtual_response_data_ref": response_ref,
        "baseline_virtual_response_data_ref": baseline_response_ref,
        "virtual_respondents": virtual_respondents,
        "response_paths": response_paths,
        "virtual_sample_config": virtual_sample_config,
        "items": enriched_items,
        "baseline_form": {"item_ids": baseline_ids, "final_item_count": len(baseline_ids)},
        "reserve_item_ids": [str(_extract_item_id(item)) for item in enriched_items if _extract_item_id(item) and str(_extract_item_id(item)) not in baseline_ids],
    }
    return normalize_snapshot(raw, source=str(form_dir))


def demo_snapshot(final_item_count: int = 4) -> dict[str, Any]:
    """Create a tiny deterministic fixture for offline workflow testing."""
    final_item_count = max(2, int(final_item_count))
    items: list[dict[str, Any]] = []
    baseline_ids: list[str] = []
    reserve_ids: list[str] = []
    for index in range(final_item_count):
        cell_id = f"facet-{index + 1:02d}"
        item_id = f"baseline-{index + 1:02d}"
        passed = index == 0
        baseline_ids.append(item_id)
        items.append({
            "item_id": item_id,
            "blueprint_cell_id": cell_id,
            "version": 1,
            "scenario": f"基线情境 {index + 1}",
            "response_options": ["A", "B", "C", "D"],
            "metrics": {"passed": passed, "target_citc": 0.8 if passed else 0.1},
            "status": "baseline",
        })
        reserve_id = f"reserve-{index + 1:02d}"
        reserve_ids.append(reserve_id)
        items.append({
            "item_id": reserve_id,
            "blueprint_cell_id": cell_id,
            "version": 1,
            "scenario": f"备用情境 {index + 1}",
            "response_options": ["A", "B", "C", "D"],
            "metrics": {"passed": False},
            "status": "reserve",
        })
    cells = [{"cell_id": f"facet-{index + 1:02d}", "required_count": 1} for index in range(final_item_count)]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "built-in-demo",
        "blueprint": {"cells": cells, "final_item_count": final_item_count},
        "baseline_form": {
            "item_ids": baseline_ids,
            "final_item_count": final_item_count,
        },
        "items": items,
        "reserve_item_ids": reserve_ids,
    }


def _item_passed(item: dict[str, Any]) -> bool:
    return bool(item.get("metrics", {}).get("passed") is True)


class OfflineRepairEngine:
    """Deterministic model substitute; it makes workflow branches testable."""

    def __init__(self, scenario: dict[str, Any] | None = None):
        self.scenario = scenario or {}
        self.calls = {"diagnosis": 0, "revision": 0, "local_retest": 0}

    def _plan(self, root_id: str) -> dict[str, Any]:
        plans = self.scenario.get("items", {})
        value = plans.get(root_id, {}) if isinstance(plans, dict) else {}
        return value if isinstance(value, dict) else {}

    async def diagnose(self, item: dict[str, Any]) -> dict[str, Any]:
        self.calls["diagnosis"] += 1
        plan = self._plan(str(item.get("root_item_id", item["item_id"])))
        failures = int(plan.get("diagnosis_failures", 0) or 0)
        if int(item.get("diagnosis_attempt", 1)) <= failures:
            return {"status": "technical_failure", "reason": "offline diagnosis stub failure"}
        return {"status": str(plan.get("diagnosis", "repair")), "reason": "offline deterministic diagnosis"}

    async def revise(
        self,
        item: dict[str, Any],
        attempt: int,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls["revision"] += 1
        root_id = str(item.get("root_item_id", item["item_id"]))
        plan = self._plan(root_id)
        if plan.get("revision_error"):
            raise RuntimeError("offline revision stub failure")
        candidate = _deepcopy(item)
        candidate["version"] = int(item.get("version", 1)) + int(attempt)
        candidate["revision_attempt"] = attempt
        candidate["scenario"] = f"{item.get('scenario', '')}｜修订候选 v{candidate['version']}"
        candidate["status"] = "revised_candidate"
        return candidate

    async def local_retest(self, item: dict[str, Any], attempt: int, action: str) -> dict[str, Any]:
        self.calls["local_retest"] += 1
        root_id = str(item.get("root_item_id", item["item_id"]))
        plan = self._plan(root_id)
        technical_attempts = plan.get("local_retest_failures", [])
        if isinstance(technical_attempts, int):
            technical_attempts = list(range(1, technical_attempts + 1))
        if attempt in technical_attempts:
            return {"status": "technical_failure", "reason": "offline local-retest stub failure"}
        if action == "repair":
            pass_after = plan.get("pass_after")
        else:
            pass_after = plan.get("replacement_pass_after")
        passed = pass_after is not None and int(attempt) >= int(pass_after)
        metrics = {"passed": bool(passed), "target_citc": 0.82 if passed else 0.14, "target_rho": 0.65 if passed else 0.18}
        return {"status": "ok", "metrics": metrics}

class PostBaselineIterationLab:
    """Full post-baseline workflow in an isolated directory.

    The lab deliberately treats technical/model failures separately from item
    failures. It never turns a diagnosis failure into an item
    pass, and it never overwrites earlier item versions.
    """

    def __init__(self, snapshot: dict[str, Any], output_dir: str | Path, config: LabConfig | None = None, scenario: dict[str, Any] | None = None):
        self.snapshot = normalize_snapshot(snapshot, source=str(snapshot.get("source", "inline"))) if snapshot.get("schema_version") != SCHEMA_VERSION or "items" not in snapshot else _deepcopy(snapshot)
        self.output_dir = Path(output_dir)
        self.config = config or LabConfig()
        self.scenario = _deepcopy(scenario or {})
        if self.config.engine_mode == "llm":
            from .live_engine import LiveLLMRepairEngine

            self.engine = LiveLLMRepairEngine(
                snapshot=self.snapshot,
                output_dir=self.output_dir,
                config=self.config,
            )
        else:
            self.engine = OfflineRepairEngine(self.scenario)
        self.items: dict[str, dict[str, Any]] = {item["item_id"]: _deepcopy(item) for item in self.snapshot["items"]}
        self.reserve_ids = [item_id for item_id in self.snapshot.get("reserve_item_ids", []) if item_id in self.items]
        self.state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "phase": "baseline",
            "started_at": _now(),
            "updated_at": _now(),
            "baseline_item_ids": list(self.snapshot["baseline_form"]["item_ids"]),
            "current_item_ids": list(self.snapshot["baseline_form"]["item_ids"]),
            "unresolved_slots": [],
            "repair_rounds": [],
            "item_history": [],
            "events": [],
            "used_reserve_ids": [],
            "replacement_counts": {},
            "stopping_reason": None,
            "engine_calls": self.engine.calls,
        }

    @classmethod
    def resume(cls, output_dir: str | Path) -> "PostBaselineIterationLab":
        output_dir = Path(output_dir)
        checkpoint = _read_json(output_dir / "checkpoint.json")
        lab = cls(checkpoint["snapshot"], output_dir, LabConfig.from_dict(checkpoint.get("config")), checkpoint.get("scenario", {}))
        lab.items = {item["item_id"]: item for item in checkpoint["items"].values()}
        lab.reserve_ids = list(checkpoint.get("reserve_ids", []))
        lab.state = checkpoint["state"]
        lab.engine.calls = checkpoint.get("engine_calls", lab.engine.calls)
        restore_history = getattr(lab.engine, "restore_history", None)
        if callable(restore_history):
            restore_history(lab.state.get("item_history") or [])
        return lab

    def _write(self, relative: str, payload: Any) -> None:
        _atomic_write_json(self.output_dir / relative, payload)

    def _checkpoint(self) -> None:
        self.state["updated_at"] = _now()
        self.state["engine_calls"] = dict(self.engine.calls)
        engine_manifest = {}
        manifest_method = getattr(self.engine, "manifest", None)
        if callable(manifest_method):
            try:
                engine_manifest = manifest_method()
            except Exception as exc:  # noqa: BLE001 - metadata must not block checkpointing
                engine_manifest = {"manifest_error": str(exc)}
        self._write("checkpoint.json", {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(self.config),
            "scenario": self.scenario,
            "snapshot": self.snapshot,
            "items": self.items,
            "reserve_ids": self.reserve_ids,
            "engine_calls": self.engine.calls,
            "engine_manifest": engine_manifest,
            "state": self.state,
        })

    def _event(self, event: str, **details: Any) -> None:
        self.state["events"].append({"at": _now(), "event": event, **details})

    async def _diagnose_with_retry(self, item: dict[str, Any]) -> dict[str, Any]:
        for diagnosis_attempt in range(1, self.config.max_model_retries + 2):
            candidate = _deepcopy(item)
            candidate["diagnosis_attempt"] = diagnosis_attempt
            try:
                result = await self.engine.diagnose(candidate)
            except Exception as exc:  # noqa: BLE001 - model failure stays item-local
                result = {"status": "technical_failure", "reason": str(exc)}
            if result.get("status") != "technical_failure":
                return {**result, "attempts": diagnosis_attempt}
        return {"status": "technical_failure", "reason": "diagnosis retry budget exhausted", "attempts": self.config.max_model_retries + 1}

    async def _repair_one(self, item_id: str) -> dict[str, Any]:
        original = _deepcopy(self.items[item_id])
        root_id = str(original.get("root_item_id", item_id))
        original["root_item_id"] = root_id
        diagnosis = await self._diagnose_with_retry(original)
        result = {
            "item_id": item_id,
            "root_item_id": root_id,
            "initial_version": original.get("version", 1),
            "diagnosis": diagnosis,
            "attempts": [],
            "outcome": None,
            "selected_item_id": None,
        }
        if diagnosis.get("status") == "technical_failure":
            result["outcome"] = "technical_failure"
            result["reason"] = diagnosis.get("reason")
            return result
        if diagnosis.get("decision") == "defer":
            # The isolated runner has no human CLI. A defer therefore enters
            # the same automatic replacement path as an exhausted repair
            # budget; it is never treated as a pass.
            result["outcome"] = "repair_exhausted"
            result["reason"] = "diagnosis deferred; automatic runner routes to reserve/replacement"
            return result
        technical_failures = 0
        valid_retests = 0
        working_item = _deepcopy(original)
        local_retest_feedback: dict[str, Any] | None = None
        for attempt in range(1, self.config.max_item_repair_attempts + 1):
            try:
                candidate = await self.engine.revise(
                    working_item,
                    attempt,
                    feedback=local_retest_feedback,
                )
            except Exception as exc:  # noqa: BLE001 - branch is explicitly tested
                result["attempts"].append({"attempt": attempt, "status": "technical_failure", "reason": str(exc)})
                technical_failures += 1
                continue
            candidate["root_item_id"] = root_id
            candidate["item_id"] = item_id
            current_version = int(working_item.get("version") or 1)
            try:
                candidate_version = int(candidate.get("version") or 0)
            except (TypeError, ValueError):
                candidate_version = 0
            if candidate_version <= current_version:
                candidate["version"] = current_version + 1
            try:
                retest = await self.engine.local_retest(candidate, attempt, "repair")
            except Exception as exc:  # noqa: BLE001 - model/simulation failure is a branch under test
                retest = {"status": "technical_failure", "reason": str(exc)}
            attempt_record = {"attempt": attempt, "candidate_version": candidate["version"], **retest}
            result["attempts"].append(attempt_record)
            self.state["item_history"].append({"root_item_id": root_id, "item_id": item_id, "action": "repair", "attempt": attempt, "candidate": _deepcopy(candidate), "retest": _deepcopy(retest)})
            if retest.get("status") != "ok":
                technical_failures += 1
                continue
            valid_retests += 1
            if retest.get("status") == "ok":
                candidate["metrics"] = retest["metrics"]
                if _item_passed(candidate):
                    candidate["status"] = "qualified_revised"
                    self.items[item_id] = candidate
                    result["outcome"] = "repaired"
                    result["selected_item_id"] = item_id
                    return result
                working_item = _deepcopy(candidate)
                local_retest_feedback = _deepcopy(retest.get("metrics") or {})
            # Technical failures do not count as item failure, but this
            # deterministic lab records them and moves to the next attempt.
        if valid_retests == 0 and technical_failures == self.config.max_item_repair_attempts:
            result["outcome"] = "technical_failure"
            result["reason"] = "repair/local-retest technical failures exhausted the local budget"
            return result
        result["outcome"] = "repair_exhausted"
        return result

    async def _test_reserve(self, reserve_id: str, root_id: str, replacement_number: int) -> dict[str, Any]:
        reserve = _deepcopy(self.items[reserve_id])
        reserve["root_item_id"] = root_id
        reserve["replacement_number"] = replacement_number
        try:
            retest = await self.engine.local_retest(reserve, 1, "replacement")
        except Exception as exc:  # noqa: BLE001 - preserve slot-level failure
            retest = {"status": "technical_failure", "reason": str(exc)}
        self.state["item_history"].append({"root_item_id": root_id, "item_id": reserve_id, "action": "reserve", "attempt": 1, "candidate": _deepcopy(reserve), "retest": _deepcopy(retest)})
        if retest.get("status") == "ok":
            reserve["metrics"] = retest["metrics"]
            if _item_passed(reserve):
                reserve["status"] = "qualified_reserve"
                self.items[reserve_id] = reserve
                return {"status": "ok", "item": reserve, "source": "reserve"}
        return {"status": "failed", "item": reserve, "source": "reserve", "reason": retest.get("reason", "reserve not qualified")}

    async def _generate_replacement(self, root_id: str, cell_id: str) -> dict[str, Any]:
        used = int(self.state["replacement_counts"].get(root_id, 0))
        for replacement_number in range(used + 1, self.config.max_replacements_per_slot + 1):
            replacement_id = f"replacement-{root_id}-{replacement_number:02d}"
            if replacement_id not in self.items:
                source = _deepcopy(self.items[root_id])
                source.update({
                    "item_id": replacement_id,
                    "root_item_id": root_id,
                    "blueprint_cell_id": cell_id,
                    "version": 1,
                    "replacement_number": replacement_number,
                    "status": "generated_replacement",
                    "scenario": f"{source.get('scenario', '')}｜同槽位补题候选 {replacement_number}",
                })
                self.items[replacement_id] = source
            try:
                retest = await self.engine.local_retest(self.items[replacement_id], 1, "replacement")
            except Exception as exc:  # noqa: BLE001 - preserve slot-level failure
                retest = {"status": "technical_failure", "reason": str(exc)}
            self.state["item_history"].append({"root_item_id": root_id, "item_id": replacement_id, "action": "generated_replacement", "attempt": 1, "candidate": _deepcopy(self.items[replacement_id]), "retest": _deepcopy(retest)})
            self.state["replacement_counts"][root_id] = replacement_number
            if retest.get("status") == "ok":
                self.items[replacement_id]["metrics"] = retest["metrics"]
                if _item_passed(self.items[replacement_id]):
                    self.items[replacement_id]["status"] = "qualified_replacement"
                    return {"status": "ok", "item": _deepcopy(self.items[replacement_id]), "source": "generated_replacement"}
        return {"status": "failed", "source": "generated_replacement", "reason": "replacement budget exhausted"}

    async def _resolve_failed_slot(self, root_id: str, cell_id: str) -> dict[str, Any]:
        root_plan = self.engine._plan(root_id)
        reserve_allowed = not bool(root_plan.get("disable_reserve", False))
        for reserve_id in self.reserve_ids if reserve_allowed else []:
            if reserve_id in self.state["used_reserve_ids"]:
                continue
            if str(self.items[reserve_id].get("blueprint_cell_id")) != str(cell_id):
                continue
            self.state["used_reserve_ids"].append(reserve_id)
            result = await self._test_reserve(reserve_id, root_id, len(self.state["used_reserve_ids"]))
            if result.get("status") == "ok":
                return result
        return await self._generate_replacement(root_id, cell_id)

    async def _run_repair_round(self, round_number: int) -> dict[str, Any]:
        current = list(self.state["current_item_ids"])
        pending = [item_id for item_id in current if not _item_passed(self.items[item_id])]
        self._event("repair_round_started", round=round_number, pending_item_ids=pending)
        repair_results: list[dict[str, Any]] = []
        if pending:
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            async def run_limited(item_id: str) -> dict[str, Any]:
                async with semaphore:
                    return await self._repair_one(item_id)

            repair_results = await asyncio.gather(*(run_limited(item_id) for item_id in pending))
        changed_ids: list[str] = []
        unresolved: list[dict[str, Any]] = []
        for result in repair_results:
            item_id = result["item_id"]
            if result.get("selected_item_id"):
                changed_ids.append(result["selected_item_id"])
                continue
            if result.get("outcome") == "technical_failure":
                unresolved.append({"item_id": item_id, "cell_id": self.items[item_id]["blueprint_cell_id"], "status": "technical_failure", "reason": result.get("reason")})
                continue
            cell_id = self.items[item_id]["blueprint_cell_id"]
            replacement = await self._resolve_failed_slot(item_id, cell_id)
            if replacement.get("status") == "ok":
                replacement_item = replacement["item"]
                changed_ids.append(replacement_item["item_id"])
                current[current.index(item_id)] = replacement_item["item_id"]
                result["replacement"] = {"source": replacement.get("source"), "item_id": replacement_item["item_id"]}
            else:
                unresolved.append({"item_id": item_id, "cell_id": cell_id, "status": "unresolved", "reason": replacement.get("reason")})
        self.state["current_item_ids"] = current
        self.state["unresolved_slots"] = unresolved
        round_record = {
            "round": round_number + 1,
            "repair_batch": round_number,
            "repair_results": repair_results,
            "unresolved_slots": _deepcopy(unresolved),
            "current_item_ids": list(current),
            "changed_item_ids": list(changed_ids),
            "model_calls": dict(self.engine.calls),
        }
        self._event("repair_round_finished", round=round_number, unresolved_count=len(unresolved))
        self._write(f"rounds/round_{round_number + 1:02d}.json", round_record)
        return round_record

    def _write_report(self) -> None:
        self._write("summary/state.json", self.state)
        self._write("summary/item_history.json", self.state.get("item_history", []))
        self._write("summary/current_items.json", {
            "item_ids": list(self.state.get("current_item_ids") or []),
            "items": [
                _deepcopy(self.items[item_id])
                for item_id in self.state.get("current_item_ids") or []
                if item_id in self.items
            ],
            "unresolved_slots": _deepcopy(self.state.get("unresolved_slots") or []),
        })
        round_rows = []
        for round_record in self.state.get("repair_rounds", []):
            round_rows.append(
                "<tr>"
                f"<td>{html.escape(str(round_record.get('round')))}</td>"
                f"<td>{html.escape(str(len(round_record.get('current_item_ids') or [])))}</td>"
                f"<td>{html.escape(str(len(round_record.get('changed_item_ids') or [])))}</td>"
                f"<td>{html.escape(str(len(round_record.get('unresolved_slots') or [])))}</td>"
                "</tr>"
            )
        body = f"""<!doctype html>
<html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>后半程迭代隔离测试</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#202124}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d9dce1;padding:8px;text-align:center}}th{{background:#f3f5f7}}.ok{{color:#137333}}.bad{{color:#b3261e}}code{{background:#f1f3f4;padding:2px 4px}}</style>
<h1>从基线到单题返修：隔离测试</h1>
<p>状态：<strong>{html.escape(str(self.state.get('status')))}</strong>；停止原因：{html.escape(str(self.state.get('stopping_reason')))}</p>
<p>当前题目数：{len(self.state.get('current_item_ids') or [])}；未解决槽位：{len(self.state.get('unresolved_slots') or [])}</p>
<h2>单题返修轮次</h2>
<table><thead><tr><th>轮次</th><th>当前题数</th><th>变更题数</th><th>未解决槽位</th></tr></thead><tbody>{''.join(round_rows)}</tbody></table>
<h2>流程检查</h2><ul><li>修订最大局部轮次：{self.config.max_item_repair_attempts}</li><li>同槽位备用/补题预算：{self.config.max_replacements_per_slot}</li><li>引擎调用：{html.escape(json.dumps(self.engine.calls, ensure_ascii=False))}</li></ul>
<p class=\"ok\">诊断失败、返修失败和槽位不足均单独记录，不会被转换成“题目通过”。</p>
</html>"""
        self.output_dir.joinpath("summary").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary" / "report.html").write_text(body, encoding="utf-8")

    async def run_async(self, stop_after: str | None = None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write("config.json", asdict(self.config))
        self._write("input_snapshot.json", self.snapshot)
        self._write("scenario.json", self.scenario)
        if self.state["phase"] == "baseline":
            baseline = {
                "round": 1,
                "kind": "baseline",
                "item_ids": list(self.state["baseline_item_ids"]),
                "item_count": len(self.state["baseline_item_ids"]),
                "failed_item_ids": [
                    item_id
                    for item_id in self.state["baseline_item_ids"]
                    if item_id in self.items and not _item_passed(self.items[item_id])
                ],
                "model_calls": dict(self.engine.calls),
            }
            self._write("rounds/round_01.json", baseline)
            self.state["phase"] = "iteration"
            self._checkpoint()
            if stop_after == "baseline":
                self.state["status"] = "paused"
                self.state["stopping_reason"] = "用户/测试请求在基线后暂停"
                self._checkpoint()
                self._write_report()
                return self.state
            if not any(
                item_id in self.items and not _item_passed(self.items[item_id])
                for item_id in self.state.get("current_item_ids") or []
            ):
                self.state["status"] = "complete"
                self.state["stopping_reason"] = "基线题目均已通过，无需单题返修"
                self.state["phase"] = "finished"
                self.state["finished_at"] = _now()
                self._checkpoint()
                self._write_report()
                return self.state
        if self.state["status"] in TERMINAL_STATUSES:
            self._write_report()
            return self.state
        start_batch = len(self.state.get("repair_rounds") or [])
        for batch_index in range(start_batch + 1, self.config.max_repair_rounds + 1):
            round_record = await self._run_repair_round(batch_index)
            self.state["repair_rounds"].append(_deepcopy(round_record))
            self._checkpoint()
            if stop_after == f"round_{batch_index + 1:02d}":
                self.state["status"] = "paused"
                self.state["stopping_reason"] = f"用户/测试请求在 round_{batch_index + 1:02d} 后暂停"
                self._checkpoint()
                self._write_report()
                return self.state
            pending_item_ids = [
                item_id
                for item_id in self.state.get("current_item_ids") or []
                if item_id in self.items and not _item_passed(self.items[item_id])
            ]
            if not pending_item_ids and not self.state.get("unresolved_slots"):
                self.state["status"] = "complete"
                self.state["stopping_reason"] = "当前题目集合中的单题均已通过"
                break
            if self.state["unresolved_slots"] and batch_index >= self.config.max_repair_rounds:
                self.state["status"] = "infeasible"
                self.state["stopping_reason"] = "同槽位备用/补题预算耗尽，未强行锁定未通过题目"
                break
        else:
            self.state["status"] = "complete" if not self.state.get("unresolved_slots") else "infeasible"
            self.state["stopping_reason"] = "达到最大自动返修轮次"
        if self.state["status"] == "running":
            self.state["status"] = "complete" if not self.state.get("unresolved_slots") else "infeasible"
            self.state["stopping_reason"] = self.state.get("stopping_reason") or "自动流程结束"
        self.state["phase"] = "finished"
        self.state["finished_at"] = _now()
        self._checkpoint()
        self._write_report()
        return self.state

    def run(self, stop_after: str | None = None) -> dict[str, Any]:
        return asyncio.run(self.run_async(stop_after=stop_after))


def export_snapshot(source: str | Path, output: str | Path) -> Path:
    normalized = load_snapshot(source)
    output_path = Path(output)
    _atomic_write_json(output_path, normalized)
    return output_path


def load_scenario(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return _read_json(Path(path))


def render_report(output_dir: str | Path) -> Path:
    checkpoint = _read_json(Path(output_dir) / "checkpoint.json")
    lab = PostBaselineIterationLab.resume(output_dir)
    lab.state = checkpoint["state"]
    lab._write_report()
    return Path(output_dir) / "summary" / "report.html"
