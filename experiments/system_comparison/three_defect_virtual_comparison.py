"""Compare two virtual-respondent methods on three controlled item defects.

Twenty blinded item versions are used: five intact Mussel source items and,
for each source, construct-shift, low-discrimination, and social-desirability
versions.  The embodied-probability and current score-profile methods never
receive condition labels, source IDs, defect rationales, or scoring keys.

This is a computer-experiment manipulation check.  It evaluates whether the
response patterns flag deliberately damaged items; it does not establish that
either virtual respondent is human-equivalent.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.evaluation.reference_questionnaires import load_mussel_items
from sjt_system.evaluation.respondents import (
    MATCHED_CONDITION_PROMPT_VERSION,
    PERSONA_MODE_SCORE_PROFILE,
)
from sjt_system.evaluation.simulation import (
    SJTSelectionOutput,
    VIRTUAL_RESPONSE_PROMPT_VERSION,
    _invoke_with_retry,
    balanced_option_order,
    build_persona_prompt,
    build_sjt_messages,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .current_score_profile_fivefacet import (
    DEFAULT_FACETS,
    _build_facet_refs,
    _dimension_specs,
    _high_low_difference,
    _select_target_base_refs,
)
from .current_system_item_defect_pilot import (
    CurrentSystemDefectConfig,
    _load_current_sample,
    _run_current_sjt,
    _validate_selection,
)
from .defect_type_classifier import (
    DEFAULT_STIMULI,
    LABELS,
    build_blind_items,
)
from .legacy_pool_abc import (
    PERSONA_PROMPT_VERSIONS,
    ChoiceProbabilityOutput,
    LegacyABCConfig,
    _json_value,
    _model_id,
    _pearson,
    _run_sjt_stage,
    _sha256_file,
    _spearman,
    _telemetry_summary,
    _validate_choice_probabilities,
    build_embodied_probability_sjt_messages,
    build_legacy_persona_prompt,
    load_comparison_summaries,
    load_legacy_pool,
)
from .mussel_item_defect_pilot import (
    MusselItemDefectConfig,
    _group_difference,
    _high_probability,
    _load_reference,
    _resolve_paths,
    _scores_by_key,
)
from .mussel_method_comparison import DIMENSION_TO_NEO, _load_neo_scores
from .storage import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSSEL = PROJECT_ROOT / "docs" / "mussel_zh.json"
METHODS = ("embodied_probability", "score_profile")
DEFECT_TYPES = ("construct_shift", "low_discrimination", "social_desirability")
FORMULA_VERSION = "three-defect-two-virtual-methods-v1"
PROMPT_VERSION = "three-defect-blind-response-comparison-v1"
OPTION_IDS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class ThreeDefectVirtualComparisonConfig:
    experiment: Path
    legacy_project: Path = Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT")
    summary_pool: Path = PROJECT_ROOT / "experiment_data" / "legacy_persona_summary_pool"
    stimuli_path: Path = DEFAULT_STIMULI
    mussel_path: Path = DEFAULT_MUSSEL
    reference_mussel_run: Path | None = None
    neo_scores_path: Path | None = None
    model_id: str | None = None
    respondents: int = 100
    max_concurrency: int = 30
    max_retries: int = 2
    timeout_seconds: float | None = None
    sampling_seed: int = 20260915
    output: Path | None = None

    def validate(self) -> None:
        if self.respondents != 100:
            raise ValueError("当前冻结设计要求respondents恰好为100")
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")


def load_three_defect_response_items(
    stimuli_path: Path = DEFAULT_STIMULI,
    mussel_path: Path = DEFAULT_MUSSEL,
    *,
    seed: int = 20260915,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 20 blinded response items and verify them against frozen Mussel."""

    blind = build_blind_items(stimuli_path, seed=seed)
    source_items, source_metadata = load_mussel_items(mussel_path)
    source_by_id = {str(item["item_id"]): item for item in source_items}
    response_items: list[dict[str, Any]] = []
    for row in blind:
        source_id = str(row["source_item_id"])
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"三类缺陷刺激引用未知Mussel题目：{source_id}")
        if str(row["target_dimension_id"]) != str(source["target_dimension_id"]):
            raise ValueError(f"{source_id}目标facet与冻结Mussel不一致")
        if str(row["scenario"]) != str(source["scenario"]):
            raise ValueError(f"{source_id}情境与冻结Mussel不一致")
        if str(row["response_instruction"]) != str(source["response_instruction"]):
            raise ValueError(f"{source_id}作答要求与冻结Mussel不一致")
        if dict(row["scoring_key"]) != {"A": 1, "B": 1, "C": 0, "D": 0}:
            raise ValueError(f"{source_id}计分键不是固定的A/B=1、C/D=0")
        options = {str(key): str(value) for key, value in row["options"].items()}
        if set(options) != set(OPTION_IDS):
            raise ValueError(f"{source_id}选项必须完整包含A-D")
        response_items.append({
            "item_id": str(row["blind_id"]),
            "blind_id": str(row["blind_id"]),
            "internal_id": str(row["internal_id"]),
            "source_item_id": source_id,
            "condition": str(row["true_label"]),
            "true_label": str(row["true_label"]),
            "facet_key": str(source["facet_key"]),
            "target_dimension_id": str(source["target_dimension_id"]),
            "scenario": str(row["scenario"]),
            "response_instruction": str(row["response_instruction"]),
            "response_options": [
                {"option_id": option_id, "text": options[option_id]}
                for option_id in OPTION_IDS
            ],
            "scoring_key": dict(row["scoring_key"]),
            "version": f"blind-three-defect-{row['blind_id']}-v1",
        })

    balance = Counter(str(item["condition"]) for item in response_items)
    if balance != Counter({label: 5 for label in LABELS}):
        raise ValueError(f"三类缺陷响应实验必须四类各5题，实际={dict(balance)}")
    per_facet: dict[str, Counter[str]] = defaultdict(Counter)
    for item in response_items:
        per_facet[str(item["target_dimension_id"])][str(item["condition"])] += 1
    if set(per_facet) != set(DEFAULT_FACETS) or any(
        counts != Counter({label: 1 for label in LABELS})
        for counts in per_facet.values()
    ):
        raise ValueError("每个facet必须恰好包含1道原题和3类缺陷题")
    return response_items, {
        "stimuli_path": str(stimuli_path.resolve()),
        "source_questionnaire": str(mussel_path.resolve()),
        "class_balance": dict(balance),
        "facet_balance": {key: dict(value) for key, value in per_facet.items()},
        "source_metadata": source_metadata,
        "prompt_blinding": (
            "模型只看到行为画像、情境、作答要求和选项；不显示盲号、源题号、"
            "条件标签、缺陷说明或计分键。"
        ),
    }


def _randomized_item_orders(
    refs: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    facet: str,
) -> dict[str, list[dict[str, Any]]]:
    import random

    result: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        respondent_id = str(ref["respondent_id"])
        local = [dict(item) for item in items]
        local_seed = int(fingerprint({
            "seed": seed,
            "facet": facet,
            "respondent_id": respondent_id,
            "purpose": "three-defect-blind-order",
        })[:16], 16)
        random.Random(local_seed).shuffle(local)
        result[respondent_id] = local
    return result


def _embodied_item_metrics(
    *,
    items: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    source_items: Sequence[Mapping[str, Any]],
    anchor_records: Sequence[Mapping[str, Any]],
    neo_scores: pd.DataFrame,
    subject_ids: Sequence[str],
) -> list[dict[str, Any]]:
    ids = list(map(str, subject_ids))
    by_key = {
        (str(row.get("respondent_id")), str(row.get("item_id"))): row
        for row in records
    }
    expected = {(rid, str(item["item_id"])) for rid in ids for item in items}
    if len(records) != len(expected) or set(by_key) != expected:
        raise ValueError(f"具身概率作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    neo = neo_scores.reindex(ids)
    if neo.isna().any().any():
        raise ValueError("具身概率被试无法与NEO分数完整对齐")
    anchor_by_key = _scores_by_key(anchor_records)
    source_by_dimension: dict[str, list[str]] = defaultdict(list)
    for source in source_items:
        source_by_dimension[str(source["target_dimension_id"])].append(str(source["item_id"]))

    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item["item_id"])
        source_id = str(item["source_item_id"])
        dimension = str(item["target_dimension_id"])
        target_domain = DIMENSION_TO_NEO[dimension]
        local = [by_key[(rid, item_id)] for rid in ids]
        high_probability = np.asarray([_high_probability(row) for row in local], dtype=float)
        sampled = np.asarray([float(row["score"]) for row in local], dtype=float)
        anchor_ids = [
            value for value in source_by_dimension[dimension] if value != source_id
        ]
        if len(anchor_ids) != 21:
            raise ValueError(f"{item_id}同facet锚点应为21题，实际为{len(anchor_ids)}")
        anchor = np.asarray([
            sum(anchor_by_key[(rid, anchor_id)] for anchor_id in anchor_ids)
            for rid in ids
        ], dtype=float)
        correlations = {
            domain: _spearman(high_probability.tolist(), neo[domain].astype(float).tolist())
            for domain in ("E", "N", "O", "A", "C")
        }
        non_target = [
            abs(float(value))
            for domain, value in correlations.items()
            if domain != target_domain and value is not None
        ]
        target_rho = correlations[target_domain]
        max_non_target = max(non_target) if non_target else None
        selections = [str(row["selected_option_id"]) for row in local]
        result: dict[str, Any] = {
            "method": "embodied_probability",
            "condition": str(item["condition"]),
            "true_label": str(item["true_label"]),
            "blind_id": str(item["blind_id"]),
            "source_item_id": source_id,
            "target_dimension_id": dimension,
            "target_criterion": f"independent_NEO_FFI_{target_domain}",
            "respondent_count": len(ids),
            "target_rho": target_rho,
            "high_low_difference": _group_difference(
                high_probability,
                neo[target_domain].astype(float).to_numpy(),
            ),
            "mean_high_score": float(np.mean(high_probability)),
            "score_sd": float(np.std(high_probability, ddof=1)),
            "target_facet_citc": _pearson(high_probability.tolist(), anchor.tolist()),
            "max_abs_non_target_rho": max_non_target,
            "construct_specificity": (
                float(target_rho) - max_non_target
                if target_rho is not None and max_non_target is not None
                else None
            ),
            "sampled_target_rho": _spearman(
                sampled.tolist(), neo[target_domain].astype(float).tolist()
            ),
            "sampled_high_low_difference": _group_difference(
                sampled, neo[target_domain].astype(float).to_numpy()
            ),
            "formula_version": FORMULA_VERSION,
        }
        for option_id in OPTION_IDS:
            result[f"option_{option_id}_rate"] = selections.count(option_id) / len(selections)
        rows.append(_json_value(result))
    return rows


def _score_profile_item_metrics(
    *,
    items: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    refs_by_facet: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("respondent_id")), str(row.get("item_id"))): row
        for row in records
    }
    expected = {
        (str(ref["respondent_id"]), str(item["item_id"]))
        for facet, refs in refs_by_facet.items()
        for ref in refs
        for item in items
        if str(item["target_dimension_id"]) == facet
    }
    if len(records) != len(expected) or set(by_key) != expected:
        raise ValueError(f"当前分数提示词作答不完整：应为{len(expected)}条，实际为{len(records)}条")

    rows: list[dict[str, Any]] = []
    for item in items:
        facet = str(item["target_dimension_id"])
        refs = sorted(refs_by_facet[facet], key=lambda row: str(row["matched_subject_id"]))
        local = [by_key[(str(ref["respondent_id"]), str(item["item_id"]))] for ref in refs]
        scores = np.asarray([float(row["score"]) for row in local], dtype=float)
        criteria = np.asarray([
            float(next(iter(ref["score_values"].values()))) for ref in refs
        ], dtype=float)
        selections = [str(row["selected_option_id"]) for row in local]
        result: dict[str, Any] = {
            "method": "score_profile",
            "condition": str(item["condition"]),
            "true_label": str(item["true_label"]),
            "blind_id": str(item["blind_id"]),
            "source_item_id": str(item["source_item_id"]),
            "target_dimension_id": facet,
            "target_criterion": "prompted_target_facet_score",
            "respondent_count": len(refs),
            "target_rho": _spearman(scores.tolist(), criteria.tolist()),
            "high_low_difference": _high_low_difference(scores, criteria),
            "mean_high_score": float(np.mean(scores)),
            "score_sd": float(np.std(scores, ddof=1)),
            "target_facet_citc": None,
            "target_facet_citc_missing_reason": (
                "每个facet只有一个独立源题，不能用同题的三个缺陷版本充当CITC锚点"
            ),
            "max_abs_non_target_rho": None,
            "construct_specificity": None,
            "formula_version": FORMULA_VERSION,
        }
        for option_id in OPTION_IDS:
            result[f"option_{option_id}_rate"] = selections.count(option_id) / len(selections)
        rows.append(_json_value(result))
    return rows


def summarize_defect_detection(
    item_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply defect-specific, directional rules fixed before model execution."""

    keyed = {
        (str(row["method"]), str(row["source_item_id"]), str(row["condition"])): row
        for row in item_rows
    }
    pairs: list[dict[str, Any]] = []
    sources = sorted({str(row["source_item_id"]) for row in item_rows})
    for method in METHODS:
        for source_id in sources:
            original = keyed.get((method, source_id, "intact"))
            if original is None:
                raise ValueError(f"{method}/{source_id}缺少正常原题")
            for defect_type in DEFECT_TYPES:
                defect = keyed.get((method, source_id, defect_type))
                if defect is None:
                    raise ValueError(f"{method}/{source_id}缺少{defect_type}版本")
                delta_rho = (
                    float(original["target_rho"]) - float(defect["target_rho"])
                    if original.get("target_rho") is not None and defect.get("target_rho") is not None
                    else None
                )
                delta_high_low = (
                    float(original["high_low_difference"])
                    - float(defect["high_low_difference"])
                    if original.get("high_low_difference") is not None
                    and defect.get("high_low_difference") is not None
                    else None
                )
                defect_minus_original_mean = (
                    float(defect["mean_high_score"]) - float(original["mean_high_score"])
                    if original.get("mean_high_score") is not None
                    and defect.get("mean_high_score") is not None
                    else None
                )
                if defect_type in {"construct_shift", "low_discrimination"}:
                    estimable = delta_rho is not None and delta_high_low is not None
                    detected = estimable and delta_rho > 0 and delta_high_low > 0
                    rule = "原题的目标rho和高低组差均高于缺陷题"
                else:
                    estimable = (
                        delta_high_low is not None
                        and defect_minus_original_mean is not None
                    )
                    detected = (
                        estimable
                        and delta_high_low > 0
                        and defect_minus_original_mean > 0
                    )
                    rule = "缺陷题高分均值升高且目标高低组差低于原题"
                pairs.append(_json_value({
                    "method": method,
                    "source_item_id": source_id,
                    "target_dimension_id": original["target_dimension_id"],
                    "defect_type": defect_type,
                    "original_blind_id": original["blind_id"],
                    "defect_blind_id": defect["blind_id"],
                    "delta_original_minus_defect_target_rho": delta_rho,
                    "delta_original_minus_defect_high_low_difference": delta_high_low,
                    "delta_defect_minus_original_mean_high_score": defect_minus_original_mean,
                    "estimable": estimable,
                    "detected": detected,
                    "detection_rule": rule,
                }))

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for defect_type in (*DEFECT_TYPES, "all"):
            selected = [
                row for row in pairs
                if row["method"] == method
                and (defect_type == "all" or row["defect_type"] == defect_type)
            ]
            estimable = [row for row in selected if row["estimable"]]
            detected = [row for row in estimable if row["detected"]]
            summary_rows.append({
                "method": method,
                "defect_type": defect_type,
                "pair_count": len(selected),
                "estimable_pairs": len(estimable),
                "detected_pairs": len(detected),
                "detection_rate": len(detected) / len(estimable) if estimable else None,
            })

    paired = []
    for source_id in sources:
        for defect_type in DEFECT_TYPES:
            values = {
                str(row["method"]): bool(row["detected"])
                for row in pairs
                if row["source_item_id"] == source_id
                and row["defect_type"] == defect_type
                and row["estimable"]
            }
            if set(values) == set(METHODS):
                paired.append(values)
    cross_method = {
        "comparable_defect_pairs": len(paired),
        "both_detected": sum(all(row.values()) for row in paired),
        "embodied_only": sum(
            row["embodied_probability"] and not row["score_profile"] for row in paired
        ),
        "score_profile_only": sum(
            row["score_profile"] and not row["embodied_probability"] for row in paired
        ),
        "neither_detected": sum(not any(row.values()) for row in paired),
    }
    summary = {
        "status": "complete",
        "method_count": 2,
        "source_item_count": len(sources),
        "defect_item_count": len(sources) * len(DEFECT_TYPES),
        "detection_summary": summary_rows,
        "cross_method_paired_detection": cross_method,
        "primary_rules": {
            "construct_shift": "原题的目标rho和高低组差均高于构念偏移题",
            "low_discrimination": "原题的目标rho和高低组差均高于低区分度题",
            "social_desirability": "社会赞许题高分均值升高，同时目标高低组差低于原题",
        },
        "interpretation_boundary": (
            "两种方法使用不同的人格输入和作答输出，因此这是方法级检出能力比较，"
            "不是只改变单个提示词的严格消融；每类只有5道题，只报告描述性结果。"
        ),
        "formula_version": FORMULA_VERSION,
    }
    return pairs, summary_rows, _json_value(summary)


def _fmt(value: Any) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    number = float(value)
    return f"{number:.3f}" if math.isfinite(number) else "—"


def _render_report(
    root: Path,
    *,
    summary: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    model_id: str,
    telemetry: Mapping[str, Any],
) -> Path:
    summary_html = "".join(
        "<tr>"
        f"<td>{escape(str(row['method']))}</td>"
        f"<td>{escape(str(row['defect_type']))}</td>"
        f"<td>{row['detected_pairs']}/{row['estimable_pairs']}</td>"
        f"<td>{_fmt(row.get('detection_rate'))}</td></tr>"
        for row in summary_rows
    )
    pair_html = "".join(
        "<tr>"
        f"<td>{escape(str(row['method']))}</td>"
        f"<td>{escape(str(row['target_dimension_id']))}</td>"
        f"<td>{escape(str(row['defect_type']))}</td>"
        f"<td>{_fmt(row.get('delta_original_minus_defect_target_rho'))}</td>"
        f"<td>{_fmt(row.get('delta_original_minus_defect_high_low_difference'))}</td>"
        f"<td>{_fmt(row.get('delta_defect_minus_original_mean_high_score'))}</td>"
        f"<td>{'是' if row['detected'] else '否'}</td></tr>"
        for row in pair_rows
    )
    cross = summary["cross_method_paired_detection"]
    report = root / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>三类坏题的双虚拟被试检出实验</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1400px;margin:30px auto;padding:0 20px;color:#20252b}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd2d8;padding:7px;text-align:left}}th{{background:#eef2f6}}.note{{background:#fff6d8;padding:12px;border-left:4px solid #d99b00}}.result{{font-size:1.2em}}</style>
<h1>三类受控坏题：两种虚拟被试方法的检出能力</h1>
<p>模型=<code>{escape(model_id)}</code>；5道Mussel源题；每题1个正常版本和3类缺陷版本；每种方法每个目标条件100名虚拟被试。</p>
<div class='note'>所有题目以盲号运行，作答模型看不到正常/缺陷标签、源题号、缺陷说明和计分键。本实验比较的是方法级检出能力；它不能证明虚拟被试等同真人。</div>
<h2>主要结果</h2>
<p class='result'>可配对缺陷={cross['comparable_defect_pairs']}；两种都检出={cross['both_detected']}；仅具身概率检出={cross['embodied_only']}；仅当前分数提示词检出={cross['score_profile_only']}；两种都未检出={cross['neither_detected']}。</p>
<table><thead><tr><th>方法</th><th>缺陷类型</th><th>检出/可估计</th><th>检出率</th></tr></thead><tbody>{summary_html}</tbody></table>
<h2>逐题配对差值</h2>
<table><thead><tr><th>方法</th><th>facet</th><th>缺陷类型</th><th>Δ目标rho</th><th>Δ高低组差</th><th>缺陷−原题高分均值</th><th>检出</th></tr></thead><tbody>{pair_html}</tbody></table>
<h2>判定规则</h2><ul><li>构念偏移、低区分度：原题目标rho和高低组差都高于缺陷题。</li><li>社会赞许：缺陷题高分均值高于原题，同时目标高低组差低于原题。</li></ul>
<p>调用={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；Token={telemetry.get('total_tokens')}；模型耗时={telemetry.get('duration_ms')} ms。</p>
<p>{escape(str(summary['interpretation_boundary']))}</p></html>""",
        encoding="utf-8",
    )
    return report


async def _run_preflights(
    *,
    probability_runnable: Any,
    selection_runnable: Any,
    embodied_persona: str,
    score_profile: Mapping[str, Any],
    score_spec: Mapping[str, Any],
    item: Mapping[str, Any],
    timeout: float,
    seed: int,
) -> None:
    semaphore = asyncio.Semaphore(1)
    await _invoke_with_retry(
        probability_runnable,
        build_embodied_probability_sjt_messages(embodied_persona, item),
        semaphore=semaphore,
        validator=_validate_choice_probabilities,
        max_retries=0,
        retry_delay_seconds=0,
        request_timeout_seconds=timeout,
        job_label="three defect embodied JSON preflight",
    )
    score_prompt = build_persona_prompt(
        score_profile,
        persona_mode=PERSONA_MODE_SCORE_PROFILE,
        score_specs=[score_spec],
    )
    option_ids = [str(option["option_id"]) for option in item["response_options"]]
    ordered_ids = balanced_option_order(
        option_ids,
        respondent_index=0,
        seed=seed,
        item_id=str(item["item_id"]),
    )
    display_ids = {chr(ord("A") + index) for index in range(len(ordered_ids))}
    await _invoke_with_retry(
        selection_runnable,
        build_sjt_messages(score_prompt, item, display_option_order=ordered_ids),
        semaphore=semaphore,
        validator=lambda value: _validate_selection(value, display_ids),
        max_retries=0,
        retry_delay_seconds=0,
        request_timeout_seconds=timeout,
        job_label="three defect score profile JSON preflight",
    )


async def run_three_defect_virtual_comparison(
    config: ThreeDefectVirtualComparisonConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"实验目录不存在：{experiment}")
    items, stimulus_metadata = load_three_defect_response_items(
        config.stimuli_path.resolve(),
        config.mussel_path.resolve(),
        seed=config.sampling_seed,
    )

    legacy_paths = _resolve_paths(MusselItemDefectConfig(
        experiment=experiment,
        legacy_project=config.legacy_project,
        summary_pool=config.summary_pool,
        stimuli_path=config.stimuli_path,
        mussel_path=config.mussel_path,
        reference_mussel_run=config.reference_mussel_run,
        neo_scores_path=config.neo_scores_path,
        model_id=config.model_id,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_seconds=config.timeout_seconds,
        sampling_seed=config.sampling_seed,
    ))
    source_items, _ = load_mussel_items(legacy_paths["mussel"])
    subject_ids, anchor_records, reference_manifest = _load_reference(
        legacy_paths["reference"], legacy_paths["anchor_responses"], source_items
    )
    if len(subject_ids) != config.respondents:
        raise ValueError(f"具身概率冻结被试应为{config.respondents}名，实际为{len(subject_ids)}")
    neo_scores = _load_neo_scores(legacy_paths["neo_scores"], subject_ids)
    _, legacy_profiles = load_legacy_pool(legacy_paths["legacy_project"])
    summaries, summary_path, _ = load_comparison_summaries(
        legacy_paths["legacy_project"], legacy_paths["summary_pool"]
    )
    missing = [rid for rid in subject_ids if rid not in legacy_profiles or rid not in summaries]
    if missing:
        raise ValueError(f"具身概率冻结被试缺少画像或总结：{missing[:3]}")

    current_sample, _ = _load_current_sample(experiment)
    all_refs, refs_by_facet, profiles_by_facet = _build_facet_refs(
        _select_target_base_refs(current_sample)
    )
    if any(len(refs) != config.respondents for refs in refs_by_facet.values()):
        raise ValueError("当前分数提示词每个facet必须恰好100名被试")
    specs_by_facet = _dimension_specs()

    model = get_model(config.model_id)
    probability_runnable, probability_method = with_compatible_structured_output(
        model, ChoiceProbabilityOutput
    )
    selection_runnable, selection_method = with_compatible_structured_output(
        model, SJTSelectionOutput
    )
    model_id = _model_id(model, config.model_id)
    reference_model = str(reference_manifest.get("embodied_condition_model") or "")
    if model_id != reference_model:
        raise ValueError(
            f"当前模型{model_id}与具身概率参照数据模型{reference_model}不同，拒绝混合"
        )
    expected_persona_version = PERSONA_PROMPT_VERSIONS["summary_embodied_probability"]
    if reference_manifest.get("persona_prompt_version") != expected_persona_version:
        raise ValueError("具身概率参照数据的人格提示词版本不一致")

    root = (
        config.output.resolve()
        if config.output
        else experiment / "mussel_item_defect_pilot" / "virtual_method_comparison_v1"
    )
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"three-defect-virtual-comparison-{root.name}"
    binding = fingerprint({
        "evaluation_sample": _sha256_file(experiment / "participants" / "evaluation.json"),
        "stimuli": _sha256_file(config.stimuli_path.resolve()),
        "mussel": _sha256_file(config.mussel_path.resolve()),
        "anchor": _sha256_file(legacy_paths["anchor_responses"]),
        "neo": _sha256_file(legacy_paths["neo_scores"]),
        "summary": _sha256_file(summary_path),
        "model_id": model_id,
        "respondents": config.respondents,
        "sampling_seed": config.sampling_seed,
        "prompt_version": PROMPT_VERSION,
        "formula_version": FORMULA_VERSION,
    })
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("binding") != binding:
            raise ValueError("输出目录已绑定不同实验输入，拒绝混合")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "experiment": str(experiment),
        "model_id": model_id,
        "methods": list(METHODS),
        "respondents_per_target_condition": config.respondents,
        "item_count": len(items),
        "expected_formal_calls": 4000,
        "preflight_calls": 2,
        "probability_structured_output_method": probability_method,
        "selection_structured_output_method": selection_method,
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(root / "config.json", _json_value(asdict(config)))
    write_json(root / "stimuli.json", {
        "metadata": stimulus_metadata,
        "warning": "items包含实验真值；这些字段不会发送给作答模型。",
        "items": items,
    })
    write_json(root / "participants.json", {
        "embodied_probability": {
            "respondent_ids": subject_ids,
            "count": len(subject_ids),
            "criterion": "independently completed NEO-FFI domain scores",
        },
        "score_profile": {
            "respondent_count_total": len(all_refs),
            "respondent_count_per_facet": config.respondents,
            "facets": list(DEFAULT_FACETS),
            "criterion": "prompted target facet score",
        },
    })

    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    legacy_run_config = LegacyABCConfig(
        experiment=experiment,
        legacy_project=config.legacy_project,
        model_id=model_id,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_seconds=config.timeout_seconds,
        persona_mode="summary_embodied_probability",
        summary_pool=config.summary_pool,
        respondent_source="legacy_100",
        sampling_seed=config.sampling_seed,
    )
    current_run_config = CurrentSystemDefectConfig(
        experiment=experiment,
        stimuli_path=config.stimuli_path,
        mussel_path=config.mussel_path,
        model_id=model_id,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_seconds=config.timeout_seconds,
        sampling_seed=config.sampling_seed,
    )
    started = perf_counter()
    failure_phase = "preflight"
    try:
        with output_scope(root / "runtime", telemetry=root / "telemetry"), run_context(run_id):
            preflight_path = root / "preflight.json"
            if not preflight_path.is_file():
                first_item = items[0]
                first_facet = str(first_item["target_dimension_id"])
                first_subject = subject_ids[0]
                first_ref = refs_by_facet[first_facet][0]
                print("[三类缺陷实验] 输出格式预检：具身概率 1/2", flush=True)
                await _run_preflights(
                    probability_runnable=probability_runnable,
                    selection_runnable=selection_runnable,
                    embodied_persona=build_legacy_persona_prompt(
                        legacy_profiles[first_subject],
                        summaries[first_subject],
                        persona_mode="summary_embodied_probability",
                    ),
                    score_profile=profiles_by_facet[first_facet][str(first_ref["respondent_id"])],
                    score_spec=specs_by_facet[first_facet],
                    item=first_item,
                    timeout=timeout,
                    seed=config.sampling_seed,
                )
                write_json(preflight_path, {
                    "status": "complete",
                    "calls": 2,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                print("[三类缺陷实验] 输出格式预检通过 2/2", flush=True)

            failure_phase = "embodied_probability_responses"
            print("[三类缺陷实验] 具身概率方法开始：100人 × 20题", flush=True)
            embodied_records = await _run_sjt_stage(
                "THREE_DEFECT_EMBODIED",
                {"items": items},
                subject_ids,
                legacy_profiles,
                summaries,
                probability_runnable,
                model_id,
                root / "embodied_probability" / "responses",
                legacy_run_config,
            )

            failure_phase = "score_profile_responses"
            score_records: list[dict[str, Any]] = []
            for index, facet in enumerate(DEFAULT_FACETS, 1):
                facet_items = [
                    item for item in items if item["target_dimension_id"] == facet
                ]
                facet_refs = refs_by_facet[facet]
                facet_profiles = list(profiles_by_facet[facet].values())
                orders = _randomized_item_orders(
                    facet_refs,
                    facet_items,
                    seed=config.sampling_seed,
                    facet=facet,
                )
                print(
                    f"[三类缺陷实验] 当前分数提示词 {index}/5：{facet}，100人 × 4题",
                    flush=True,
                )
                local = await _run_current_sjt(
                    root=root / "score_profile",
                    run_id=run_id,
                    items=facet_items,
                    refs=facet_refs,
                    profiles=facet_profiles,
                    condition_rows={"target": {"arm_id": "target", "group_id": facet}},
                    condition_specs={"target": [specs_by_facet[facet]]},
                    runnable=selection_runnable,
                    model_id=model_id,
                    config=current_run_config,
                    item_order_by_respondent=orders,
                    response_path_override=(
                        root / "score_profile" / "response_cache" / facet / "responses.jsonl"
                    ),
                )
                score_records.extend(local)

        combined_path = root / "score_profile" / "responses.jsonl"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", encoding="utf-8") as handle:
            for record in score_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        failure_phase = "analysis_and_report"
        embodied_rows = _embodied_item_metrics(
            items=items,
            records=embodied_records,
            source_items=source_items,
            anchor_records=anchor_records,
            neo_scores=neo_scores,
            subject_ids=subject_ids,
        )
        score_rows = _score_profile_item_metrics(
            items=items,
            records=score_records,
            refs_by_facet=refs_by_facet,
        )
        item_rows = embodied_rows + score_rows
        pair_rows, summary_rows, summary = summarize_defect_detection(item_rows)
        analysis = root / "analysis"
        write_csv(analysis / "item_metrics.csv", item_rows)
        write_json(analysis / "item_metrics.json", item_rows)
        write_csv(analysis / "pair_detection.csv", pair_rows)
        write_json(analysis / "pair_detection.json", pair_rows)
        write_csv(analysis / "detection_summary.csv", summary_rows)
        write_json(analysis / "summary.json", summary)
        telemetry = _telemetry_summary(root, run_id)
        cost = {
            **telemetry,
            "wall_time_seconds_this_invocation": round(perf_counter() - started, 3),
            "expected_formal_calls": 4000,
            "preflight_calls": 2,
            "price": None,
            "note": "未提供适用单价，不推测费用。",
        }
        write_json(root / "cost.json", _json_value(cost))
        report = _render_report(
            root,
            summary=summary,
            summary_rows=summary_rows,
            pair_rows=pair_rows,
            model_id=model_id,
            telemetry=telemetry,
        )
        finished = {
            **manifest,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "report": str(report),
            "cost": cost,
        }
        write_json(manifest_path, _json_value(finished))
        return root, {"summary": summary, "report": report, "cost": cost}
    except Exception as exc:
        write_json(manifest_path, {
            **manifest,
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "failure_phase": failure_phase,
            "error": str(exc),
        })
        raise
