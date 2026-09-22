"""Use the current matched-facet virtual respondents on the controlled defect pilot.

This is intentionally separate from the legacy-summary experiment.  It uses
the frozen current-system evaluation sample, the current score-profile prompt,
balanced option display, and the current four item gates:
target-arm CITC, target-arm rho, same-domain VTS, and cross-domain VTS.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import math
from pathlib import Path
from time import perf_counter
from typing import Any
import json

import numpy as np

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.evaluation.respondents import (
    MATCHED_CONDITION_PROMPT_VERSION,
    PERSONA_MODE_SCORE_PROFILE,
    flatten_matched_condition_groups,
    resolve_virtual_respondent_profiles,
)
from sjt_system.evaluation.simulation import (
    SJTSelectionOutput,
    VIRTUAL_RESPONSE_PROMPT_VERSION,
    _invoke_with_retry,
    _load_jsonl_keys,
    balanced_option_order,
    build_persona_prompt,
    build_sjt_messages,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import (
    _json_value,
    _model_id,
    _pearson,
    _spearman,
    _telemetry_summary,
)
from .mussel_item_defect_pilot import (
    DEFAULT_STIMULI,
    load_pilot_items,
)
from .storage import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSSEL = PROJECT_ROOT / "docs" / "mussel_zh.json"
FORMULA_VERSION = "current-system-matched-item-defect-v1"
PROMPT_VERSION = "current-system-matched-item-defect-v1"
OPTION_IDS = ("A", "B", "C", "D")
TARGET_CITC_MINIMUM = 0.20
TARGET_RHO_MINIMUM = 0.30
SAME_DOMAIN_VTS_MINIMUM = 0.10
CROSS_DOMAIN_VTS_MINIMUM = 0.20


@dataclass(frozen=True)
class CurrentSystemDefectConfig:
    experiment: Path
    stimuli_path: Path = DEFAULT_STIMULI
    mussel_path: Path = DEFAULT_MUSSEL
    model_id: str | None = None
    max_concurrency: int = 30
    max_retries: int = 2
    timeout_seconds: float | None = None
    sampling_seed: int = 20260915
    output: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")
        if isinstance(self.sampling_seed, bool) or not isinstance(self.sampling_seed, int):
            raise ValueError("sampling_seed必须是整数")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON文件必须是对象：{path}")
    return value


def _load_current_sample(experiment: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = experiment / "participants" / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少当前系统evaluation被试：{path}")
    sample = _read_json(path)
    refs = sample.get("respondents")
    config = sample.get("config")
    if sample.get("role") != "evaluation" or not isinstance(refs, list) or not isinstance(config, Mapping):
        raise ValueError("当前系统evaluation被试结构无效")
    if len(refs) != 300:
        raise ValueError(f"方案1要求当前系统冻结300名匹配被试，实际为{len(refs)}")
    if len({str(row.get("respondent_id")) for row in refs}) != len(refs):
        raise ValueError("当前系统evaluation被试ID重复")
    if config.get("persona_modes") != [PERSONA_MODE_SCORE_PROFILE]:
        raise ValueError("当前系统evaluation不是score_profile虚拟被试条件")
    return sample, [dict(row) for row in refs]


def _condition_specs(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    conditions = config.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("当前系统evaluation缺少conditions")
    groups = flatten_matched_condition_groups(conditions)
    if {str(row.get("condition_id")) for row in groups} != {
        "target",
        "same_domain__extraversion_warmth",
        "cross_domain__neuroticism_anxiety",
    }:
        raise ValueError("方案1要求当前冻结的target/same-domain/cross-domain三臂")
    rows = {str(row["condition_id"]): dict(row) for row in groups}
    specs: dict[str, list[dict[str, Any]]] = {}
    for condition_id, row in rows.items():
        specs[condition_id] = [{
            "dimension_id": str(row.get("dimension_id") or ""),
            "level": "facet",
            "domain_id": row.get("domain_id"),
            "domain_name_en": row.get("domain_name_en") or row.get("domain_id"),
            "domain_name": row.get("domain_name") or row.get("domain_id"),
            "facet_name_en": row.get("facet_name_en") or row.get("dimension_id"),
            "facet_name": row.get("facet_name") or row.get("dimension_id"),
        }]
    return rows, specs


def _target_items(stimuli: Path, mussel: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original, modified, metadata = load_pilot_items(stimuli, mussel)
    all_items = original + modified
    selected = [
        item for item in all_items
        if item["target_dimension_id"] == "extraversion_gregariousness"
    ]
    if len(selected) != 4:
        raise ValueError("当前冻结目标facet应恰好对应2组原题—缺陷题")
    return selected, metadata


def _validate_selection(value: Mapping[str, Any], allowed: set[str]) -> str:
    selected = str(value.get("selected_option_id") or "").strip().upper()
    if selected not in allowed:
        raise ValueError(f"模型选择了无效展示选项：{selected!r}")
    return selected


async def _run_current_sjt(
    *,
    root: Path,
    run_id: str,
    items: Sequence[Mapping[str, Any]],
    refs: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    condition_rows: Mapping[str, Mapping[str, Any]],
    condition_specs: Mapping[str, Sequence[Mapping[str, Any]]],
    runnable: Any,
    model_id: str,
    config: CurrentSystemDefectConfig,
    item_order_by_respondent: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    response_path_override: Path | None = None,
) -> list[dict[str, Any]]:
    response_path = response_path_override or root / "responses.jsonl"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_jsonl_keys(response_path, ("respondent_id", "item_id"))
    profile_by_id = {str(profile["respondent_id"]): profile for profile in profiles}
    ref_by_id = {str(ref["respondent_id"]): ref for ref in refs}
    index_by_matched = {
        str(ref["matched_subject_id"]): index
        for index, ref in enumerate(refs)
        if ref.get("condition_id") == "target"
    }
    if len(index_by_matched) != 100:
        raise ValueError("当前系统target臂必须包含100名唯一matched_subject_id")
    prompts = {
        respondent_id: build_persona_prompt(
            profile,
            persona_mode=PERSONA_MODE_SCORE_PROFILE,
            score_specs=condition_specs[str(ref_by_id[respondent_id]["condition_id"])],
        )
        for respondent_id, profile in profile_by_id.items()
    }
    jobs = []
    for ref in refs:
        respondent_id = str(ref["respondent_id"])
        respondent_items = list(
            item_order_by_respondent.get(respondent_id, items)
            if item_order_by_respondent is not None
            else items
        )
        for item in respondent_items:
            if (respondent_id, str(item["item_id"])) not in existing:
                jobs.append((respondent_id, item))
    semaphore = asyncio.Semaphore(config.max_concurrency)
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    lock = asyncio.Lock()
    completed = len(existing)
    total = len(refs) * len(items)

    async def one(respondent_id: str, item: Mapping[str, Any]) -> None:
        nonlocal completed
        item_id = str(item["item_id"])
        ref = ref_by_id[respondent_id]
        condition_id = str(ref["condition_id"])
        matched_id = str(ref["matched_subject_id"])
        option_ids = [str(option["option_id"]) for option in item["response_options"]]
        ordered_ids = balanced_option_order(
            option_ids,
            respondent_index=index_by_matched.get(matched_id, 0),
            seed=config.sampling_seed,
            item_id=item_id,
        )
        display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
        display_to_original = dict(zip(display_ids, ordered_ids))
        display_choice = await _invoke_with_retry(
            runnable,
            build_sjt_messages(prompts[respondent_id], item, display_option_order=ordered_ids),
            semaphore=semaphore,
            validator=lambda value: _validate_selection(value, set(display_ids)),
            max_retries=config.max_retries,
            retry_delay_seconds=1.0,
            request_timeout_seconds=timeout,
            job_label=f"current system item defect {condition_id}/{matched_id}/{item_id}",
        )
        selected = display_to_original[display_choice]
        presentation_items = list(
            item_order_by_respondent.get(respondent_id, items)
            if item_order_by_respondent is not None
            else items
        )
        presentation_index = next(
            index
            for index, value in enumerate(presentation_items, start=1)
            if str(value["item_id"]) == item_id
        )
        record = {
            "record_type": "current_system_mussel_item_defect_response",
            "run_id": run_id,
            "respondent_id": respondent_id,
            "condition_id": condition_id,
            "arm_id": ref.get("arm_id") or condition_rows[condition_id].get("arm_id"),
            "group_id": ref.get("group_id") or condition_rows[condition_id].get("group_id"),
            "matched_subject_id": matched_id,
            "active_dimension_id": ref.get("active_dimension_id"),
            "active_score": float(next(iter((ref.get("score_values") or {}).values()))),
            "persona_mode": PERSONA_MODE_SCORE_PROFILE,
            "item_id": item_id,
            "presentation_index": presentation_index,
            "item_version": item.get("version"),
            "display_option_order": [
                {"display_option_id": display_id, "option_id": original_id}
                for display_id, original_id in zip(display_ids, ordered_ids)
            ],
            "raw_display_option_id": display_choice,
            "selected_option_id": selected,
            "score": float(item["scoring_key"][selected]),
            "scoring_version": "mussel-ab-high-cd-low-binary-v1",
            "response_mode": "current_system_score_profile_balanced_order",
            "model_id": model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        }
        async with lock:
            with response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed == total or completed % max(1, total // 20) == 0:
                print(
                    f"[current system] SJT {completed}/{total} ({completed / total:.0%})",
                    flush=True,
                )

    if jobs:
        print(
            f"[current system] SJT开始：待调用={len(jobs)}；已缓存={len(existing)}",
            flush=True,
        )
        results = await asyncio.gather(*(one(*job) for job in jobs), return_exceptions=True)
        errors = [value for value in results if isinstance(value, Exception)]
        if errors:
            write_json(root / "errors.json", {"errors": [str(error) for error in errors]})
            raise RuntimeError(f"当前系统SJT有{len(errors)}个调用失败；首个错误：{errors[0]}")
    records = [
        json.loads(line)
        for line in response_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {(str(ref["respondent_id"]), str(item["item_id"])) for ref in refs for item in items}
    actual = {(str(row.get("respondent_id")), str(row.get("item_id"))) for row in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError(f"当前系统作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    return records


def _metric_row(
    *,
    item: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    refs: Sequence[Mapping[str, Any]],
    item_ids: Sequence[str],
    condition_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {(str(row["respondent_id"]), str(row["item_id"])): row for row in records}
    refs_by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for ref in refs:
        refs_by_condition.setdefault(str(ref["condition_id"]), []).append(ref)
    condition_for_arm: dict[str, str] = {}
    for condition_id, row in condition_rows.items():
        condition_for_arm[str(row.get("arm_id"))] = condition_id
    current_item_id = str(item["item_id"])
    target_id = "target"
    same_id = condition_for_arm.get("same_domain")
    cross_id = condition_for_arm.get("cross_domain")
    if not same_id or not cross_id:
        raise ValueError("当前系统缺少唯一同领域或跨领域条件")

    def arrays(condition_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        condition_refs = sorted(refs_by_condition[condition_id], key=lambda row: str(row["matched_subject_id"]))
        scores = np.asarray([
            float(by_key[(str(ref["respondent_id"]), current_item_id)]["score"])
            for ref in condition_refs
        ])
        criteria = np.asarray([
            float(next(iter((ref.get("score_values") or {}).values())))
            for ref in condition_refs
        ])
        return scores, criteria, [str(ref["matched_subject_id"]) for ref in condition_refs]

    target_scores, target_criteria, target_matched = arrays(target_id)
    same_scores, same_criteria, same_matched = arrays(same_id)
    cross_scores, cross_criteria, cross_matched = arrays(cross_id)
    if target_matched != same_matched or target_matched != cross_matched:
        raise ValueError(f"{current_item_id}三臂matched_subject_id无法对齐")
    other_item_ids = [value for value in item_ids if value != current_item_id]
    if len(other_item_ids) != 1:
        raise ValueError("当前方案1每个facet必须有且只有另一道配对题作为CITC锚点")
    other_scores = np.asarray([
        float(by_key[(str(ref["respondent_id"]), other_item_ids[0])]["score"])
        for ref in sorted(refs_by_condition[target_id], key=lambda row: str(row["matched_subject_id"]))
    ])
    target_rho = _spearman(target_scores.tolist(), target_criteria.tolist())
    same_rho = _spearman(same_scores.tolist(), same_criteria.tolist())
    cross_rho = _spearman(cross_scores.tolist(), cross_criteria.tolist())
    same_vts = float(target_rho) - float(same_rho) if target_rho is not None and same_rho is not None else None
    cross_vts = float(target_rho) - float(cross_rho) if target_rho is not None and cross_rho is not None else None
    citc = _pearson(target_scores.tolist(), other_scores.tolist())
    gates = {
        "target_facet_citc": citc is not None and citc >= TARGET_CITC_MINIMUM,
        "target_rho": target_rho is not None and target_rho >= TARGET_RHO_MINIMUM,
        "same_domain_vts": same_vts is not None and same_vts >= SAME_DOMAIN_VTS_MINIMUM,
        "cross_domain_vts": cross_vts is not None and cross_vts >= CROSS_DOMAIN_VTS_MINIMUM,
    }
    target_selections = [
        str(by_key[(str(ref["respondent_id"]), current_item_id)]["selected_option_id"])
        for ref in sorted(refs_by_condition[target_id], key=lambda row: str(row["matched_subject_id"]))
    ]
    row: dict[str, Any] = {
        "condition": str(item["condition"]),
        "item_id": current_item_id,
        "source_item_id": str(item["source_item_id"]),
        "facet_key": str(item["facet_key"]),
        "target_dimension_id": str(item["target_dimension_id"]),
        "respondent_count_per_condition": len(target_scores),
        "citc_anchor_item_id": other_item_ids[0],
        "citc_anchor_item_count": 1,
        "target_facet_citc": citc,
        "target_rho": target_rho,
        "same_domain_rho": same_rho,
        "cross_domain_rho": cross_rho,
        "same_domain_vts": same_vts,
        "cross_domain_vts": cross_vts,
        "same_domain_condition_id": same_id,
        "cross_domain_condition_id": cross_id,
        "target_facet_citc_threshold": TARGET_CITC_MINIMUM,
        "target_rho_threshold": TARGET_RHO_MINIMUM,
        "same_domain_vts_threshold": SAME_DOMAIN_VTS_MINIMUM,
        "cross_domain_vts_threshold": CROSS_DOMAIN_VTS_MINIMUM,
        "gates": gates,
        "qualified": all(gates.values()),
        "formula_version": FORMULA_VERSION,
    }
    for option_id in OPTION_IDS:
        count = target_selections.count(option_id)
        row[f"target_option_{option_id}_n"] = count
        row[f"target_option_{option_id}_rate"] = count / len(target_selections)
    return _json_value(row)


def analyze_current_system_item_defect(
    *,
    items: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    refs: Sequence[Mapping[str, Any]],
    condition_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    item_ids = [str(item["item_id"]) for item in items]
    if len(items) != 4 or len(set(item_ids)) != 4:
        raise ValueError("当前系统方案1必须分析4个题目版本")
    expected = {(str(ref["respondent_id"]), item_id) for ref in refs for item_id in item_ids}
    actual = {(str(row.get("respondent_id")), str(row.get("item_id"))) for row in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError(f"当前系统作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    by_source = {str(item["source_item_id"]): item for item in items}
    if len(by_source) != 2:
        raise ValueError("当前系统方案1必须有2组原题—缺陷题")
    item_rows = [
        _metric_row(
            item=item,
            records=records,
            refs=refs,
            item_ids=[
                str(value["item_id"])
                for value in items
                if str(value["condition"]) == str(item["condition"])
            ],
            condition_rows=condition_rows,
        )
        for item in items
    ]
    keyed = {(row["source_item_id"], row["condition"]): row for row in item_rows}
    metrics = ("target_facet_citc", "target_rho", "same_domain_vts", "cross_domain_vts")
    pair_rows = []
    for source_id in sorted(by_source):
        original = keyed[(source_id, "original")]
        modified = keyed[(source_id, "high_option_absence")]
        pair = {
            "source_item_id": source_id,
            "facet_key": original["facet_key"],
            "original_item_id": original["item_id"],
            "modified_item_id": modified["item_id"],
        }
        detected = 0
        for metric in metrics:
            original_value = original.get(metric)
            modified_value = modified.get(metric)
            delta = float(original_value) - float(modified_value) if original_value is not None and modified_value is not None else None
            pair[f"original_{metric}"] = original_value
            pair[f"modified_{metric}"] = modified_value
            pair[f"delta_original_minus_modified_{metric}"] = delta
            pair[f"detected_{metric}"] = delta is not None and delta > 0
            detected += int(pair[f"detected_{metric}"])
        pair["estimable_metric_count"] = sum(pair.get(f"delta_original_minus_modified_{metric}") is not None for metric in metrics)
        pair["detected_metric_count"] = detected
        pair["majority_detected"] = pair["estimable_metric_count"] == 4 and detected >= 3
        pair_rows.append(_json_value(pair))
    metric_summary = []
    for metric in metrics:
        deltas = [float(row[f"delta_original_minus_modified_{metric}"]) for row in pair_rows if row[f"delta_original_minus_modified_{metric}"] is not None]
        metric_summary.append({
            "metric": metric,
            "estimable_pairs": len(deltas),
            "detected_pairs": sum(value > 0 for value in deltas),
            "detection_rate": sum(value > 0 for value in deltas) / len(deltas) if deltas else None,
            "mean_delta_original_minus_modified": float(np.mean(deltas)) if deltas else None,
        })
    majority = sum(bool(row["majority_detected"]) for row in pair_rows)
    summary = {
        "status": "complete",
        "respondent_count_total": len(refs),
        "respondent_count_per_condition": len(refs) // 3,
        "pair_count": len(pair_rows),
        "primary_rule": "current four gates all estimable and at least three have original > high-option-absence",
        "majority_detected_pairs": majority,
        "majority_detection_rate": majority / len(pair_rows),
        "metric_summary": metric_summary,
        "metric_definitions": {
            "target_facet_citc": "Pearson(target-arm item score, target-arm other item score); only one other item is available in this two-pair pilot",
            "target_rho": "Spearman(target-arm item score, target-arm active facet score)",
            "same_domain_vts": "target rho minus same-domain arm rho, using the current signed-rho rule",
            "cross_domain_vts": "target rho minus cross-domain arm rho, using the current signed-rho rule",
        },
        "interpretation_boundary": "This is a two-item, known severe option-defect sensitivity pilot for the current score-profile virtual respondent; it is not human validation or a general psychometric quality claim.",
        "formula_version": FORMULA_VERSION,
    }
    return item_rows, pair_rows, _json_value(summary)


def _render_report(root: Path, summary: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]], model_id: str, telemetry: Mapping[str, Any]) -> Path:
    def f(value: Any) -> str:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "不可估计"
        return f"{float(value):.3f}"

    labels = {
        "target_facet_citc": "目标facet CITC",
        "target_rho": "目标rho",
        "same_domain_vts": "同领域VTS",
        "cross_domain_vts": "跨领域VTS",
    }
    pair_html = []
    for row in pairs:
        pair_html.append(
            "<tr>"
            f"<td>{escape(str(row['source_item_id']))}</td>"
            f"<td>{escape(str(row['facet_key']))}</td>"
            + "".join(f"<td>{f(row.get(f'delta_original_minus_modified_{metric}'))}</td>" for metric in labels)
            + f"<td>{row['detected_metric_count']}/4</td><td>{'是' if row['majority_detected'] else '否'}</td></tr>"
        )
    summary_html = "".join(
        "<tr>"
        f"<td>{escape(labels[row['metric']])}</td>"
        f"<td>{row['detected_pairs']}/{row['estimable_pairs']}</td>"
        f"<td>{f(row.get('detection_rate'))}</td>"
        f"<td>{f(row.get('mean_delta_original_minus_modified'))}</td></tr>"
        for row in summary["metric_summary"]
    )
    report = root / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>当前系统虚拟被试缺陷敏感性</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd2d8;padding:7px;text-align:left}}th{{background:#eef2f6}}.note{{background:#fff6d8;padding:12px;border-left:4px solid #d99b00}}.result{{font-size:1.3em}}</style>
<h1>当前系统虚拟被试：高水平选项缺失实验</h1>
<p>当前系统score_profile匹配样本；总被试={summary['respondent_count_total']}（每臂={summary['respondent_count_per_condition']}）；配对题={summary['pair_count']}；模型=<code>{escape(model_id)}</code>。</p>
<p class='result'>按当前四门槛，检出 <strong>{summary['majority_detected_pairs']}/{summary['pair_count']}</strong> 组缺陷题。</p>
<div class='note'>本实验只覆盖当前系统已经冻结的“外倾性·乐群性”目标facet，因此只分析两组题。原题与缺陷题均使用同一批300名匹配被试、当前score_profile提示词和选项平衡显示。正差表示原题指标高于缺陷题。</div>
<h2>配对差值</h2><table><thead><tr><th>源题</th><th>facet</th><th>ΔCITC</th><th>Δ目标rho</th><th>Δ同领域VTS</th><th>Δ跨领域VTS</th><th>下降项</th><th>多数检出</th></tr></thead><tbody>{''.join(pair_html)}</tbody></table>
<h2>指标总体检出</h2><table><thead><tr><th>指标</th><th>检出/可估计</th><th>检出率</th><th>平均原题−缺陷题</th></tr></thead><tbody>{summary_html}</tbody></table>
<h2>解释</h2><ul><li>CITC：目标臂中该题与另一道乐群性题的相关；本试验只有2题，因此只是小样本锚点。</li><li>目标rho：乐群性分数与题目得分的Spearman相关。</li><li>同领域/跨领域VTS：目标rho减去相应非目标臂rho，沿用当前系统的带符号规则。</li></ul>
<p>调用数={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；Token={telemetry.get('total_tokens')}。结论边界：这只能说明当前score_profile虚拟被试对这一类严重选项缺陷的敏感性，不能说明它已经等同真人。</p></html>""",
        encoding="utf-8",
    )
    return report


async def run_current_system_item_defect_pilot(config: CurrentSystemDefectConfig) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    if not (experiment / "progress.json").is_file():
        raise FileNotFoundError(f"不是完整当前系统实验目录：{experiment}")
    sample, refs = _load_current_sample(experiment)
    condition_rows, condition_specs = _condition_specs(sample["config"])
    items, stimulus_metadata = _target_items(config.stimuli_path.resolve(), config.mussel_path.resolve())
    profiles: list[dict[str, Any]] = []
    for condition_id in condition_rows:
        condition_refs = [
            ref for ref in refs if str(ref.get("condition_id")) == condition_id
        ]
        profiles.extend(
            resolve_virtual_respondent_profiles(
                condition_refs,
                score_specs=condition_specs[condition_id],
            )
        )
    if {str(profile["respondent_id"]) for profile in profiles} != {str(ref["respondent_id"]) for ref in refs}:
        raise ValueError("当前系统被试画像与evaluation引用无法对齐")
    model = get_model(config.model_id)
    runnable, structured_output_method = with_compatible_structured_output(model, SJTSelectionOutput)
    model_id = _model_id(model, config.model_id)
    root = config.output.resolve() if config.output else experiment / "mussel_item_defect_pilot" / "current_system_score_profile"
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"current-system-item-defect-{root.name}"
    sample_path = experiment / "participants" / "evaluation.json"
    binding = fingerprint({
        "sample_sha256": _sha256_file(sample_path),
        "stimuli_sha256": _sha256_file(config.stimuli_path.resolve()),
        "mussel_sha256": _sha256_file(config.mussel_path.resolve()),
        "model_id": model_id,
        "persona_mode": PERSONA_MODE_SCORE_PROFILE,
        "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        "sampling_seed": config.sampling_seed,
        "formula_version": FORMULA_VERSION,
    })
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        if previous.get("binding") != binding:
            raise ValueError("当前系统缺陷实验输出目录已绑定不同输入，拒绝混合")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "experiment": str(experiment),
        "respondent_count_total": len(refs),
        "respondent_count_per_condition": 100,
        "conditions": list(condition_rows),
        "items_per_condition": 4,
        "model_id": model_id,
        "structured_output_method": structured_output_method,
        "persona_mode": PERSONA_MODE_SCORE_PROFILE,
        "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        "sampling_seed": config.sampling_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(root / "config.json", _json_value(asdict(config)))
    write_json(root / "participants.json", {
        "source": str(sample_path),
        "respondent_ids": [str(ref["respondent_id"]) for ref in refs],
        "matched_subject_ids": sorted({str(ref["matched_subject_id"]) for ref in refs}),
        "count": len(refs),
        "three_arms": True,
    })
    write_json(root / "stimuli.json", {"metadata": stimulus_metadata, "items": items})
    started = perf_counter()
    try:
        with output_scope(root / "runtime", telemetry=root / "telemetry"), run_context(run_id):
            records = await _run_current_sjt(
                root=root,
                run_id=run_id,
                items=items,
                refs=refs,
                profiles=profiles,
                condition_rows=condition_rows,
                condition_specs=condition_specs,
                runnable=runnable,
                model_id=model_id,
                config=config,
            )
        item_rows, pair_rows, summary = analyze_current_system_item_defect(
            items=items,
            records=records,
            refs=refs,
            condition_rows=condition_rows,
        )
        analysis = root / "analysis"
        write_csv(analysis / "item_metrics.csv", item_rows)
        write_json(analysis / "item_metrics.json", item_rows)
        write_csv(analysis / "pair_comparison.csv", pair_rows)
        write_json(analysis / "pair_comparison.json", pair_rows)
        write_csv(analysis / "metric_summary.csv", summary["metric_summary"])
        write_json(analysis / "summary.json", summary)
        telemetry = _telemetry_summary(root, run_id)
        write_json(root / "cost.json", _json_value({
            **telemetry,
            "wall_time_seconds_this_invocation": round(perf_counter() - started, 3),
            "price": None,
            "note": "未提供适用单价，因此不推测费用。",
        }))
        report = _render_report(root, summary, pair_rows, model_id, telemetry)
        finished = {
            **manifest,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "report": str(report),
            "summary": summary,
        }
        write_json(manifest_path, _json_value(finished))
        return root, {"report": report, "summary": summary}
    except Exception as exc:
        write_json(manifest_path, {**manifest, "status": "failed", "error": str(exc)})
        raise


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
