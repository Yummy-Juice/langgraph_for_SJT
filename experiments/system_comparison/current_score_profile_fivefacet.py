"""Five-facet defect-sensitivity experiment for the current score-profile prompt.

The experiment deliberately keeps the manipulation severe and known: ten
Mussel items are paired with versions in which only the high-scoring A/B
options are rewritten as low-trait behaviour.  One hundred score-profile
respondents per facet answer both versions.  The target facet score is the
criterion, but it is never included in the SJT question itself.

This is a defect-detection sensitivity experiment, not a claim that the
virtual respondents are human-equivalent.  The primary comparison is whether
the original item has a larger target-score Spearman rho and high-vs-low
score-group difference than its deliberately damaged counterpart.  CITC is
reported as a secondary, two-item within-facet anchor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sjt_system.agent.client import get_model, with_compatible_structured_output
from sjt_system.authoring.construct_registry import construct_selection_catalog
from sjt_system.evaluation.respondents import (
    MATCHED_CONDITION_PROMPT_VERSION,
    PERSONA_MODE_SCORE_PROFILE,
    build_score_dimension_catalog,
    resolve_virtual_respondent_profiles,
)
from sjt_system.evaluation.simulation import (
    VIRTUAL_RESPONSE_PROMPT_VERSION,
    _load_jsonl_keys,
    balanced_option_order,
    SJTSelectionOutput,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .current_system_item_defect_pilot import (
    CurrentSystemDefectConfig,
    _load_current_sample,
    _run_current_sjt,
)
from .legacy_pool_abc import (
    _json_value,
    _model_id,
    _pearson,
    _sha256_file,
    _spearman,
    _telemetry_summary,
)
from .mussel_item_defect_pilot import DEFAULT_STIMULI, load_pilot_items
from .storage import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSSEL = PROJECT_ROOT / "docs" / "mussel_zh.json"
DEFAULT_FACETS = (
    "openness_ideas",
    "conscientiousness_self_discipline",
    "extraversion_gregariousness",
    "agreeableness_compliance",
    "neuroticism_self_consciousness",
)
FORMULA_VERSION = "current-score-profile-fivefacet-defect-v1"
PRIMARY_METRICS = ("target_rho", "high_low_difference")
SECONDARY_METRICS = ("target_facet_citc",)
OPTION_IDS = ("A", "B", "C", "D")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON文件必须是对象：{path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} 第{line_number}行不是对象")
        rows.append(value)
    return rows


def _dimension_specs() -> dict[str, dict[str, Any]]:
    rows = build_score_dimension_catalog(construct_selection_catalog())
    result = {
        str(row["dimension_id"]): dict(row)
        for row in rows
        if row.get("level") == "facet" and row.get("dimension_id")
    }
    missing = [facet for facet in DEFAULT_FACETS if facet not in result]
    if missing:
        raise ValueError("构念注册表缺少五facet：" + "、".join(missing))
    return result


def _select_target_base_refs(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = sample.get("respondents")
    if not isinstance(refs, list):
        raise ValueError("evaluation被试缺少respondents")
    target = [dict(ref) for ref in refs if ref.get("condition_id") == "target"]
    target.sort(key=lambda row: str(row.get("matched_subject_id")))
    if len(target) != 100:
        raise ValueError(f"五facet实验需要100名target被试，实际为{len(target)}")
    values = []
    for ref in target:
        scores = ref.get("score_values")
        if not isinstance(scores, Mapping) or len(scores) != 1:
            raise ValueError("当前evaluation target分数结构无效")
        score = float(next(iter(scores.values())))
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("当前evaluation target分数超出0至100")
        values.append(score)
    return target


def _build_facet_refs(
    base_refs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    specs_by_dimension = _dimension_specs()
    all_refs: list[dict[str, Any]] = []
    refs_by_facet: dict[str, list[dict[str, Any]]] = {}
    profiles_by_facet: dict[str, dict[str, Any]] = {}
    for facet in DEFAULT_FACETS:
        refs: list[dict[str, Any]] = []
        spec = specs_by_dimension[facet]
        for index, base in enumerate(base_refs, 1):
            base_id = str(base["respondent_id"])
            respondent_id = (
                base_id
                if facet == "extraversion_gregariousness"
                else f"fivefacet-{facet}-{index:04d}"
            )
            source_score = float(next(iter(base["score_values"].values())))
            ref = {
                "respondent_id": respondent_id,
                "condition_id": "target",
                "arm_id": "target",
                "group_id": facet,
                "matched_subject_id": str(base["matched_subject_id"]),
                "active_dimension_id": facet,
                "score_values": {facet: source_score},
                "source_respondent_id": base_id,
                "source_sample": "current_evaluation_target_score_sequence",
            }
            refs.append(ref)
            all_refs.append(ref)
        profiles = resolve_virtual_respondent_profiles(
            refs,
            score_specs=[spec],
        )
        profiles_by_facet[facet] = {
            str(profile["respondent_id"]): profile for profile in profiles
        }
        refs_by_facet[facet] = refs
    if len({str(ref["respondent_id"]) for ref in all_refs}) != len(all_refs):
        raise ValueError("五facet派生被试ID重复")
    return all_refs, refs_by_facet, profiles_by_facet


def _items_by_facet(items: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {facet: [] for facet in DEFAULT_FACETS}
    for item in items:
        facet = str(item["target_dimension_id"])
        if facet not in result:
            raise ValueError(f"刺激包含方案外facet：{facet}")
        result[facet].append(dict(item))
    if any(len(values) != 4 for values in result.values()):
        raise ValueError("五facet刺激必须每个facet包含原题/缺陷题各2道")
    return result


def _ordered_items(
    facet: str,
    refs: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create reproducible per-respondent order; prompts contain no arm label."""

    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_source.setdefault(str(item["source_item_id"]), []).append(dict(item))
    if any(len(values) != 2 for values in by_source.values()):
        raise ValueError(f"{facet}没有形成完整的原题—缺陷题对")
    ordered: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        key = str(ref["respondent_id"])
        rng_seed = int(fingerprint({"seed": seed, "facet": facet, "respondent": key})[:16], 16)
        import random

        rng = random.Random(rng_seed)
        blocks = list(by_source.values())
        for block in blocks:
            rng.shuffle(block)
        rng.shuffle(blocks)
        sequence = [item for block in blocks for item in block]
        ordered[key] = sequence
    return ordered


def _seed_cached_gregariousness(
    *,
    target_path: Path,
    source_path: Path,
    allowed_item_ids: set[str],
    model_id: str,
    run_id: str,
) -> int:
    """Copy only compatible target-arm records from the previous two-pair run."""

    if target_path.exists() and target_path.stat().st_size > 0:
        return 0
    source_rows = [
        row for row in _load_jsonl(source_path)
        if row.get("condition_id") == "target"
        and str(row.get("item_id")) in allowed_item_ids
        and str(row.get("model_id")) == model_id
    ]
    keys = {(str(row.get("respondent_id")), str(row.get("item_id"))) for row in source_rows}
    if len(source_rows) != 400 or len(keys) != 400:
        return 0
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for source in source_rows:
            row = deepcopy(source)
            row["run_id"] = run_id
            row["record_type"] = "current_score_profile_fivefacet_response"
            row["reused_from"] = str(source_path)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(source_rows)


def _high_low_difference(scores: np.ndarray, criteria: np.ndarray) -> float | None:
    if len(scores) < 4 or len(scores) != len(criteria):
        return None
    count = max(1, int(round(len(scores) * 0.27)))
    order = np.argsort(criteria, kind="stable")
    low = scores[order[:count]]
    high = scores[order[-count:]]
    if len(low) == 0 or len(high) == 0:
        return None
    return float(np.mean(high) - np.mean(low))


def _metric_row(
    *,
    item: Mapping[str, Any],
    counterpart: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {(str(row["respondent_id"]), str(row["item_id"])): row for row in records}
    ordered_refs = sorted(refs, key=lambda row: str(row["matched_subject_id"]))
    item_id = str(item["item_id"])
    counterpart_id = str(counterpart["item_id"])
    scores = np.asarray([
        float(by_key[(str(ref["respondent_id"]), item_id)]["score"])
        for ref in ordered_refs
    ])
    anchor = np.asarray([
        float(by_key[(str(ref["respondent_id"]), counterpart_id)]["score"])
        for ref in ordered_refs
    ])
    criteria = np.asarray([
        float(next(iter(ref["score_values"].values()))) for ref in ordered_refs
    ])
    selections = [
        str(by_key[(str(ref["respondent_id"]), item_id)]["selected_option_id"])
        for ref in ordered_refs
    ]
    row: dict[str, Any] = {
        "condition": str(item["condition"]),
        "source_item_id": str(item["source_item_id"]),
        "item_id": item_id,
        "facet_key": str(item["facet_key"]),
        "target_dimension_id": str(item["target_dimension_id"]),
        "respondent_count": len(scores),
        "target_rho": _spearman(scores.tolist(), criteria.tolist()),
        "high_low_difference": _high_low_difference(scores, criteria),
        "target_facet_citc": _pearson(scores.tolist(), anchor.tolist()),
        "citc_anchor_item_id": counterpart_id,
        "scoring_version": "mussel-ab-high-cd-low-binary-v1",
        "formula_version": FORMULA_VERSION,
    }
    for option_id in OPTION_IDS:
        count = selections.count(option_id)
        row[f"target_option_{option_id}_n"] = count
        row[f"target_option_{option_id}_rate"] = count / len(selections) if selections else None
    return _json_value(row)


def analyze_fivefacet_defects(
    *,
    items: Sequence[Mapping[str, Any]],
    refs_by_facet: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    item_list = [dict(item) for item in items]
    expected = {
        (str(ref["respondent_id"]), str(item["item_id"]))
        for facet_refs in refs_by_facet.values()
        for ref in facet_refs
        for item in item_list
        if str(item["target_dimension_id"]) == str(ref["active_dimension_id"])
    }
    actual = {(str(row.get("respondent_id")), str(row.get("item_id"))) for row in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError(f"五facet作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for facet in DEFAULT_FACETS:
        facet_items = [item for item in item_list if str(item["target_dimension_id"]) == facet]
        original = {str(item["source_item_id"]): item for item in facet_items if item["condition"] == "original"}
        modified = {str(item["source_item_id"]): item for item in facet_items if item["condition"] == "high_option_absence"}
        if set(original) != set(modified) or len(original) != 2:
            raise ValueError(f"{facet}必须包含2组完整原题—缺陷题")
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for source_id, original_item in original.items():
            modified_item = modified[source_id]
            counterpart_source_id = next(
                key for key in original if key != source_id
            )
            original_row = _metric_row(
                item=original_item,
                counterpart=original[counterpart_source_id],
                records=records,
                refs=refs_by_facet[facet],
            )
            modified_row = _metric_row(
                item=modified_item,
                counterpart=modified[counterpart_source_id],
                records=records,
                refs=refs_by_facet[facet],
            )
            item_rows.extend([original_row, modified_row])
            keyed[(source_id, "original")] = original_row
            keyed[(source_id, "high_option_absence")] = modified_row
        for source_id in sorted(original):
            old = keyed[(source_id, "original")]
            new = keyed[(source_id, "high_option_absence")]
            pair: dict[str, Any] = {
                "facet": facet,
                "source_item_id": source_id,
                "original_item_id": old["item_id"],
                "modified_item_id": new["item_id"],
            }
            for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
                old_value = old.get(metric)
                new_value = new.get(metric)
                delta = (
                    float(old_value) - float(new_value)
                    if old_value is not None and new_value is not None
                    else None
                )
                pair[f"original_{metric}"] = old_value
                pair[f"modified_{metric}"] = new_value
                pair[f"delta_original_minus_modified_{metric}"] = delta
                pair[f"detected_{metric}"] = delta is not None and delta > 0
            pair["primary_estimable"] = all(
                pair[f"delta_original_minus_modified_{metric}"] is not None
                for metric in PRIMARY_METRICS
            )
            pair["primary_detected"] = pair["primary_estimable"] and all(
                bool(pair[f"detected_{metric}"]) for metric in PRIMARY_METRICS
            )
            pair_rows.append(_json_value(pair))
    metric_summary = []
    for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
        deltas = [
            float(row[f"delta_original_minus_modified_{metric}"])
            for row in pair_rows
            if row[f"delta_original_minus_modified_{metric}"] is not None
        ]
        metric_summary.append({
            "metric": metric,
            "estimable_pairs": len(deltas),
            "detected_pairs": sum(value > 0 for value in deltas),
            "detection_rate": sum(value > 0 for value in deltas) / len(deltas) if deltas else None,
            "mean_delta_original_minus_modified": float(np.mean(deltas)) if deltas else None,
        })
    primary_estimable = sum(bool(row["primary_estimable"]) for row in pair_rows)
    primary_detected = sum(bool(row["primary_detected"]) for row in pair_rows)
    summary = {
        "status": "complete",
        "facet_count": len(DEFAULT_FACETS),
        "pair_count": len(pair_rows),
        "respondent_count_per_facet": len(next(iter(refs_by_facet.values()))),
        "primary_rule": "原题相对缺陷题同时具有更高的目标rho和高低27%组得分差；两项均可估计才算检出",
        "primary_estimable_pairs": primary_estimable,
        "primary_detected_pairs": primary_detected,
        "primary_detection_rate": primary_detected / primary_estimable if primary_estimable else None,
        "metric_summary": metric_summary,
        "metric_definitions": {
            "target_rho": "目标facet设定分数与目标臂单题0/1得分的Spearman相关",
            "high_low_difference": "目标facet分数最高27%组与最低27%组的单题平均得分之差",
            "target_facet_citc": "同facet两道题得分的Pearson相关；每个facet仅2题，作为次要指标",
        },
        "design_note": "五个facet复用同一组目标分数秩序以控制抽样差异；这是受控缺陷敏感性实验，不是独立人格样本验证",
        "interpretation_boundary": "检出率只能说明当前score_profile虚拟被试对本实验预设的严重选项缺陷是否敏感，不能证明虚拟被试等同真人",
        "formula_version": FORMULA_VERSION,
    }
    return item_rows, pair_rows, _json_value(summary)


def _fmt(value: Any) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "不可估计"
    return f"{float(value):.3f}"


def _render_report(root: Path, summary: Mapping[str, Any], item_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]], model_id: str, telemetry: Mapping[str, Any], reused_calls: int) -> Path:
    pair_html = []
    for row in pair_rows:
        pair_html.append(
            "<tr>"
            f"<td>{escape(str(row['facet']))}</td>"
            f"<td>{escape(str(row['source_item_id']))}</td>"
            f"<td>{_fmt(row.get('delta_original_minus_modified_target_rho'))}</td>"
            f"<td>{_fmt(row.get('delta_original_minus_modified_high_low_difference'))}</td>"
            f"<td>{_fmt(row.get('delta_original_minus_modified_target_facet_citc'))}</td>"
            f"<td>{'是' if row.get('primary_detected') else '否'}</td></tr>"
        )
    item_html = []
    for row in item_rows:
        item_html.append(
            "<tr>"
            f"<td>{escape(str(row['facet_key']))}</td><td>{escape(str(row['condition']))}</td>"
            f"<td>{escape(str(row['source_item_id']))}</td><td>{_fmt(row.get('target_rho'))}</td>"
            f"<td>{_fmt(row.get('high_low_difference'))}</td><td>{_fmt(row.get('target_facet_citc'))}</td>"
            f"<td>{_fmt(row.get('target_option_A_rate'))}</td><td>{_fmt(row.get('target_option_B_rate'))}</td>"
            f"<td>{_fmt(row.get('target_option_C_rate'))}</td><td>{_fmt(row.get('target_option_D_rate'))}</td></tr>"
        )
    report = root / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>当前分数提示词五facet缺陷敏感性</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1400px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd2d8;padding:7px;text-align:left}}th{{background:#eef2f6}}.note{{background:#fff6d8;padding:12px;border-left:4px solid #d99b00}}.result{{font-size:1.3em}}</style>
<h1>当前score_profile虚拟被试：五facet坏题检出实验</h1>
<p>模型=<code>{escape(model_id)}</code>；每个facet=100名目标分数型虚拟被试；10组原题—缺陷题配对。</p>
<p class='result'>主要指标同时检出：<strong>{summary['primary_detected_pairs']}/{summary['primary_estimable_pairs']}</strong>。</p>
<div class='note'>缺陷操作只改写原题A/B高分选项为低水平行为，情境、C/D和A/B=1、C/D=0计分保持不变。每名被试的原题/缺陷题呈现顺序经过可复现随机化；SJT问题不显示目标facet名称、分数或计分方向。正差表示原题指标高于缺陷题。</div>
<h2>配对差值（原题−缺陷题）</h2><table><thead><tr><th>facet</th><th>源题</th><th>Δ目标rho</th><th>Δ高低组差</th><th>ΔCITC</th><th>主要指标同时检出</th></tr></thead><tbody>{''.join(pair_html)}</tbody></table>
<h2>单题结果</h2><table><thead><tr><th>facet</th><th>版本</th><th>源题</th><th>目标rho</th><th>高低组差</th><th>CITC</th><th>A比例</th><th>B比例</th><th>C比例</th><th>D比例</th></tr></thead><tbody>{''.join(item_html)}</tbody></table>
<h2>指标解释</h2><ul><li>目标rho：题目得分是否随设定目标facet分数有序变化。</li><li>高低组差：高目标分与低目标分虚拟被试的题目得分区分度。</li><li>CITC：两题facet内部相关；每个facet只有2题，因此只作次要参考。</li></ul>
<p>本次新调用={telemetry.get('calls')}；复用旧缓存={reused_calls}；Token={telemetry.get('total_tokens')}；模型耗时={telemetry.get('duration_ms')} ms。</p>
<p>结论边界：该实验检验的是当前分数提示词对预设严重选项缺陷的敏感性，不是虚拟被试与真人等价性证明。</p></html>""",
        encoding="utf-8",
    )
    return report


async def run_current_score_profile_fivefacet(config: CurrentSystemDefectConfig) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    sample, base_refs = _load_current_sample(experiment)
    all_refs, refs_by_facet, profiles_by_facet = _build_facet_refs(_select_target_base_refs(sample))
    original, modified, stimulus_metadata = load_pilot_items(
        (config.stimuli_path or DEFAULT_STIMULI).resolve(),
        (config.mussel_path or DEFAULT_MUSSEL).resolve(),
    )
    all_items = original + modified
    items_by_facet = _items_by_facet(all_items)
    specs_by_dimension = _dimension_specs()
    model = get_model(config.model_id)
    runnable, structured_output_method = with_compatible_structured_output(model, SJTSelectionOutput)
    model_id = _model_id(model, config.model_id)
    root = config.output.resolve() if config.output else experiment / "mussel_item_defect_pilot" / "current_score_profile_fivefacet"
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"current-score-profile-fivefacet-{root.name}"
    sample_path = experiment / "participants" / "evaluation.json"
    stimuli_path = (config.stimuli_path or DEFAULT_STIMULI).resolve()
    mussel_path = (config.mussel_path or DEFAULT_MUSSEL).resolve()
    source_cache = experiment / "mussel_item_defect_pilot" / "current_system_score_profile" / "responses.jsonl"
    binding = fingerprint({
        "sample_sha256": _sha256_file(sample_path),
        "stimuli_sha256": _sha256_file(stimuli_path),
        "mussel_sha256": _sha256_file(mussel_path),
        "model_id": model_id,
        "facets": DEFAULT_FACETS,
        "respondent_count_per_facet": 100,
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
            raise ValueError("五facet输出目录已绑定不同输入，拒绝混合")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "experiment": str(experiment),
        "facets": list(DEFAULT_FACETS),
        "respondent_count_per_facet": 100,
        "respondent_count_total": len(all_refs),
        "item_count_original": len(original),
        "item_count_modified": len(modified),
        "model_id": model_id,
        "structured_output_method": structured_output_method,
        "persona_mode": PERSONA_MODE_SCORE_PROFILE,
        "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(root / "config.json", _json_value({**{k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}, "facets": list(DEFAULT_FACETS)}))
    write_json(root / "participants.json", {
        "source": str(sample_path),
        "design": "same target score sequence reused across five facet-specific score_profile conditions",
        "respondents": all_refs,
    })
    write_json(root / "stimuli.json", {"metadata": stimulus_metadata, "items": all_items})
    presentation_orders: dict[str, list[dict[str, Any]]] = {}
    item_order_by_respondent: dict[str, list[dict[str, Any]]] = {}
    for facet in DEFAULT_FACETS:
        local = _ordered_items(facet, refs_by_facet[facet], items_by_facet[facet], config.sampling_seed)
        item_order_by_respondent.update(local)
        presentation_orders[facet] = [
            {"respondent_id": respondent_id, "item_ids": [str(item["item_id"]) for item in ordered]}
            for respondent_id, ordered in local.items()
        ]
    write_json(root / "presentation_orders.json", presentation_orders)
    condition_rows = {"target": {"arm_id": "target", "group_id": "target"}}
    condition_specs = {
        facet: [{
            "dimension_id": facet,
            "level": "facet",
            "domain_id": specs_by_dimension[facet]["domain_id"],
            "domain_name_en": specs_by_dimension[facet]["domain_name_en"],
            "domain_name": specs_by_dimension[facet]["domain_name"],
            "facet_name_en": specs_by_dimension[facet]["facet_name_en"],
            "facet_name": specs_by_dimension[facet]["facet_name"],
        }]
        for facet in DEFAULT_FACETS
    }
    started = perf_counter()
    reused_calls = 0
    profiles: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for facet in DEFAULT_FACETS:
        refs.extend(refs_by_facet[facet])
        profiles.extend(profiles_by_facet[facet].values())
    # The current runner accepts the same target-only shape; group calls by facet
    # because each facet has a different score-profile prompt.
    all_records: list[dict[str, Any]] = []
    try:
        with output_scope(root / "runtime", telemetry=root / "telemetry"), run_context(run_id):
            for facet in DEFAULT_FACETS:
                facet_refs = refs_by_facet[facet]
                facet_profiles = list(profiles_by_facet[facet].values())
                facet_response_path = root / "response_cache" / facet / "responses.jsonl"
                if facet == "extraversion_gregariousness":
                    reusable_ids = {
                        str(item["item_id"])
                        for item in items_by_facet[facet]
                    }
                    reused_calls += _seed_cached_gregariousness(
                        target_path=facet_response_path,
                        source_path=source_cache,
                        allowed_item_ids=reusable_ids,
                        model_id=model_id,
                        run_id=run_id,
                    )
                records = await _run_current_sjt(
                    root=root,
                    run_id=run_id,
                    items=items_by_facet[facet],
                    refs=facet_refs,
                    profiles=facet_profiles,
                    condition_rows=condition_rows,
                    condition_specs={"target": condition_specs[facet]},
                    runnable=runnable,
                    model_id=model_id,
                    config=config,
                    item_order_by_respondent=item_order_by_respondent,
                    response_path_override=facet_response_path,
                )
                all_records.extend(records)
                print(f"[五facet] {facet} 完成：{len(records)}条作答记录", flush=True)
        combined_path = root / "responses.jsonl"
        with combined_path.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        item_rows, pair_rows, summary = analyze_fivefacet_defects(
            items=all_items,
            refs_by_facet=refs_by_facet,
            records=all_records,
        )
        analysis = root / "analysis"
        write_csv(analysis / "item_metrics.csv", item_rows)
        write_json(analysis / "item_metrics.json", item_rows)
        write_csv(analysis / "pair_comparison.csv", pair_rows)
        write_json(analysis / "pair_comparison.json", pair_rows)
        write_csv(analysis / "metric_summary.csv", summary["metric_summary"])
        write_json(analysis / "summary.json", summary)
        telemetry = _telemetry_summary(root, run_id)
        cost = {
            **telemetry,
            "new_model_calls": telemetry.get("calls"),
            "reused_cached_calls": reused_calls,
            "wall_time_seconds_this_invocation": round(perf_counter() - started, 3),
            "price": None,
            "note": "复用旧缓存不重复计入本次新调用；未提供适用单价，不推测费用。",
        }
        write_json(root / "cost.json", _json_value(cost))
        report = _render_report(root, summary, item_rows, pair_rows, model_id, telemetry, reused_calls)
        finished = {
            **manifest,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "report": str(report),
            "summary": summary,
            "cost": cost,
        }
        write_json(manifest_path, _json_value(finished))
        return root, {"report": report, "summary": summary, "cost": cost}
    except Exception as exc:
        write_json(manifest_path, {**manifest, "status": "failed", "error": str(exc)})
        raise
