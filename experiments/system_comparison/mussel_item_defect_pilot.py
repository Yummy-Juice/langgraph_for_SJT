"""Controlled pilot: can embodied virtual respondents detect defective items?

The manipulation keeps each Mussel scenario and the A/B=1, C/D=0 scoring key
fixed, but rewrites A/B so that all four options express low-trait behaviour.
The same 100 frozen respondents answer both the source and manipulated versions
in independent calls.  Existing full-Mussel responses are used only as a fixed
21-item same-facet anchor; they are never substituted for either pilot arm.
"""

from __future__ import annotations

import asyncio
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

from sjt_system.agent.client import get_model, with_compatible_structured_output
from sjt_system.evaluation.reference_questionnaires import load_mussel_items
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import (
    DEFAULT_IMPORTED_SUMMARY_POOL,
    DEFAULT_LEGACY_PROJECT,
    NEO_DOMAINS,
    OPTION_IDS,
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
    load_comparison_summaries,
    load_legacy_pool,
    probability_diagnostics,
)
from .mussel_method_comparison import DIMENSION_TO_NEO, _jsonl, _load_neo_scores
from .storage import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STIMULI = Path(__file__).resolve().parent / "stimuli" / "mussel_high_option_absence_10.json"
FORMULA_VERSION = "mussel-item-defect-sensitivity-v1"
PROMPT_VERSION = "mussel-high-option-absence-pilot-v1"
CONDITIONS = ("original", "high_option_absence")
PRIMARY_SCORE = "probability"


@dataclass(frozen=True)
class MusselItemDefectConfig:
    experiment: Path
    legacy_project: Path = DEFAULT_LEGACY_PROJECT
    summary_pool: Path = DEFAULT_IMPORTED_SUMMARY_POOL
    stimuli_path: Path = DEFAULT_STIMULI
    mussel_path: Path | None = None
    reference_mussel_run: Path | None = None
    neo_scores_path: Path | None = None
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


def _resolve_paths(config: MusselItemDefectConfig) -> dict[str, Path]:
    experiment = config.experiment.resolve()
    reference = (
        config.reference_mussel_run.resolve()
        if config.reference_mussel_run
        else experiment / "mussel_legacy_vs_embodied_100"
    )
    paths = {
        "experiment": experiment,
        "legacy_project": config.legacy_project.resolve(),
        "summary_pool": config.summary_pool.resolve(),
        "stimuli": config.stimuli_path.resolve(),
        "mussel": (
            config.mussel_path.resolve()
            if config.mussel_path
            else PROJECT_ROOT / "docs" / "mussel_zh.json"
        ),
        "reference": reference,
        "anchor_responses": reference / "embodied_probability" / "responses.jsonl",
        "neo_scores": (
            config.neo_scores_path.resolve()
            if config.neo_scores_path
            else experiment
            / "legacy_summary_embodied_probability_abc"
            / "neo_ffi"
            / "scores.csv"
        ),
    }
    required = [
        paths["legacy_project"] / "sjt_system" / "data" / "virtual_respondents.json",
        paths["summary_pool"] / "persona_summaries.jsonl",
        paths["stimuli"],
        paths["mussel"],
        paths["reference"] / "manifest.json",
        paths["anchor_responses"],
        paths["neo_scores"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("题目缺陷实验缺少输入：" + "；".join(missing))
    return paths


def _option_map(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(option["option_id"]): str(option["text"])
        for option in item.get("response_options") or []
        if isinstance(option, Mapping)
    }


def load_pilot_items(
    stimuli_path: Path,
    mussel_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate the manipulation against the frozen source and build two arms."""

    stimulus = json.loads(stimuli_path.read_text(encoding="utf-8"))
    specs = stimulus.get("items") if isinstance(stimulus, Mapping) else None
    if not isinstance(specs, list) or len(specs) != 10:
        raise ValueError("缺陷刺激必须恰好包含10道题")
    source_items, source_metadata = load_mussel_items(mussel_path)
    source_by_id = {str(item["item_id"]): item for item in source_items}
    original: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    facet_counts: dict[str, int] = {}
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ValueError("缺陷刺激包含非对象题目")
        source_id = str(spec.get("source_item_id") or "")
        if source_id not in source_by_id or source_id in source_ids:
            raise ValueError(f"缺陷刺激source_item_id无效或重复：{source_id}")
        source_ids.add(source_id)
        source = source_by_id[source_id]
        if str(spec.get("scenario")) != str(source["scenario"]):
            raise ValueError(f"{source_id}修改了情境")
        if str(spec.get("target_dimension_id")) != str(source["target_dimension_id"]):
            raise ValueError(f"{source_id}目标维度不匹配")
        if str(spec.get("facet_key")) != str(source["facet_key"]):
            raise ValueError(f"{source_id} facet不匹配")
        source_options = _option_map(source)
        original_options = {str(k): str(v) for k, v in (spec.get("original_options") or {}).items()}
        changed_options = {str(k): str(v) for k, v in (spec.get("modified_options") or {}).items()}
        if source_options != original_options:
            raise ValueError(f"{source_id}原选项与冻结Mussel题库不一致")
        if set(changed_options) != set(OPTION_IDS):
            raise ValueError(f"{source_id}修改版必须包含A-D")
        if changed_options["C"] != source_options["C"] or changed_options["D"] != source_options["D"]:
            raise ValueError(f"{source_id}只能修改A/B，C/D必须保持不变")
        if changed_options["A"] == source_options["A"] or changed_options["B"] == source_options["B"]:
            raise ValueError(f"{source_id}必须同时修改A/B")
        if dict(source["scoring_key"]) != {"A": 1, "B": 1, "C": 0, "D": 0}:
            raise ValueError(f"{source_id}源计分键不是A/B=1、C/D=0")
        common = {
            "source_item_id": source_id,
            "source_item_number": str(spec.get("source_item_number") or ""),
            "facet_key": str(source["facet_key"]),
            "target_dimension_id": str(source["target_dimension_id"]),
            "scenario": str(source["scenario"]),
            "response_instruction": str(source["response_instruction"]),
            "scoring_key": dict(source["scoring_key"]),
            "defect_rationale": str(spec.get("defect_rationale") or ""),
        }
        original.append({
            **common,
            "item_id": f"{source_id}-PILOT-ORIGINAL",
            "condition": "original",
            "version": "original-v1",
            "response_options": [
                {"option_id": option_id, "text": source_options[option_id]}
                for option_id in OPTION_IDS
            ],
        })
        modified.append({
            **common,
            "item_id": str(spec.get("item_id") or f"{source_id}-HOA"),
            "condition": "high_option_absence",
            "version": "high-option-absence-v1",
            "response_options": [
                {"option_id": option_id, "text": changed_options[option_id]}
                for option_id in OPTION_IDS
            ],
        })
        facet = str(source["facet_key"])
        facet_counts[facet] = facet_counts.get(facet, 0) + 1
    if set(facet_counts.values()) != {2} or len(facet_counts) != 5:
        raise ValueError(f"缺陷刺激必须是5个facet各2题，实际为{facet_counts}")
    metadata = {
        "stimulus_schema_version": stimulus.get("schema_version"),
        "manipulation_id": stimulus.get("manipulation_id"),
        "manipulation_description": stimulus.get("manipulation_description"),
        "interpretation_boundary": stimulus.get("interpretation_boundary"),
        "selection_rule": stimulus.get("selection_rule"),
        "facet_counts": facet_counts,
        "source_metadata": source_metadata,
    }
    return original, modified, metadata


def _load_reference(
    reference_root: Path,
    anchor_path: Path,
    source_items: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((reference_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("作为锚点的完整Mussel实验尚未完成")
    subject_ids = [str(value) for value in manifest.get("respondent_ids") or []]
    if len(subject_ids) != 100 or len(set(subject_ids)) != 100:
        raise ValueError("锚点Mussel实验必须绑定唯一的100名被试")
    records = _jsonl(anchor_path)
    expected_items = {str(item["item_id"]) for item in source_items}
    expected = {(rid, item_id) for rid in subject_ids for item_id in expected_items}
    actual = {
        (str(row.get("respondent_id") or ""), str(row.get("item_id") or ""))
        for row in records
    }
    if len(records) != len(expected) or actual != expected:
        raise ValueError(
            f"完整Mussel锚点作答不完整：应为{len(expected)}条，实际为{len(records)}条"
        )
    return subject_ids, records, manifest


def _scores_by_key(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in records:
        key = (str(row.get("respondent_id") or ""), str(row.get("item_id") or ""))
        score = row.get("score")
        if key in result or isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"作答记录重复或缺少数值分数：{key}")
        result[key] = float(score)
    return result


def _high_probability(row: Mapping[str, Any]) -> float:
    probabilities = row.get("choice_probabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError("具身概率作答缺少choice_probabilities")
    values = {str(key): value for key, value in probabilities.items()}
    if set(values) != set(OPTION_IDS):
        raise ValueError("choice_probabilities必须完整包含A-D")
    numeric = [float(values[option_id]) for option_id in OPTION_IDS]
    if any(not math.isfinite(value) or value < 0 for value in numeric) or sum(numeric) <= 0:
        raise ValueError("choice_probabilities包含无效概率")
    denominator = sum(numeric)
    return (numeric[0] + numeric[1]) / denominator


def _group_difference(scores: np.ndarray, target: np.ndarray) -> float | None:
    if len(scores) < 4 or np.all(target == target[0]):
        return None
    k = max(1, int(math.floor(0.27 * len(scores))))
    if 2 * k > len(scores):
        return None
    order = np.argsort(target, kind="stable")
    return float(np.mean(scores[order[-k:]]) - np.mean(scores[order[:k]]))


def _metric_block(
    scores: np.ndarray,
    target_domain: str,
    neo: pd.DataFrame,
    anchor: np.ndarray,
) -> dict[str, Any]:
    correlations = {
        domain: _spearman(scores.tolist(), neo[domain].astype(float).tolist())
        for domain in NEO_DOMAINS
    }
    target_rho = correlations[target_domain]
    non_target = [
        abs(float(value))
        for domain, value in correlations.items()
        if domain != target_domain and value is not None
    ]
    maximum = max(non_target) if non_target else None
    specificity = (
        float(target_rho) - maximum
        if target_rho is not None and maximum is not None
        else None
    )
    return {
        "target_neo_spearman": target_rho,
        "max_abs_non_target_neo_spearman": maximum,
        "specificity": specificity,
        "anchor_citc": _pearson(scores.tolist(), anchor.tolist()),
        "high_minus_low_group_score": _group_difference(
            scores, neo[target_domain].astype(float).to_numpy()
        ),
        **{f"rho_neo_{domain}": correlations[domain] for domain in NEO_DOMAINS},
    }


def analyze_item_defect_pilot(
    *,
    pilot_items: Sequence[Mapping[str, Any]],
    pilot_records: Sequence[Mapping[str, Any]],
    source_items: Sequence[Mapping[str, Any]],
    anchor_records: Sequence[Mapping[str, Any]],
    neo_scores: pd.DataFrame,
    subject_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pure analysis function used by the real runner and offline tests."""

    ids = list(map(str, subject_ids))
    if len(ids) < 10 or len(set(ids)) != len(ids):
        raise ValueError("分析被试必须唯一且至少10名")
    neo = neo_scores.reindex(ids)
    if neo[list(NEO_DOMAINS)].isna().any().any():
        raise ValueError("NEO得分不能与分析被试完整对齐")
    pilot_by_key = {
        (str(row.get("respondent_id") or ""), str(row.get("item_id") or "")): row
        for row in pilot_records
    }
    expected = {
        (rid, str(item["item_id"])) for rid in ids for item in pilot_items
    }
    if len(pilot_records) != len(expected) or set(pilot_by_key) != expected:
        raise ValueError(
            f"试验作答不完整或重复：应为{len(expected)}条，实际为{len(pilot_records)}条"
        )
    source_by_dimension: dict[str, list[str]] = {}
    for source in source_items:
        source_by_dimension.setdefault(str(source["target_dimension_id"]), []).append(
            str(source["item_id"])
        )
    anchor_by_key = _scores_by_key(anchor_records)
    anchor_record_by_key = {
        (str(row.get("respondent_id") or ""), str(row.get("item_id") or "")): row
        for row in anchor_records
    }
    probability_overall, probability_rows = probability_diagnostics(pilot_records)
    probability_by_item = {str(row["item_id"]): row for row in probability_rows}
    item_rows: list[dict[str, Any]] = []
    for item in pilot_items:
        item_id = str(item["item_id"])
        source_id = str(item["source_item_id"])
        dimension = str(item["target_dimension_id"])
        target_domain = DIMENSION_TO_NEO[dimension]
        dimension_items = source_by_dimension.get(dimension) or []
        anchor_ids = [value for value in dimension_items if value != source_id]
        if len(anchor_ids) != 21:
            raise ValueError(f"{item_id}同facet固定锚点应为21题，实际为{len(anchor_ids)}")
        rows = [pilot_by_key[(rid, item_id)] for rid in ids]
        sampled = np.asarray([float(row["score"]) for row in rows], dtype=float)
        expected_probability = np.asarray([_high_probability(row) for row in rows], dtype=float)
        anchor = np.asarray(
            [sum(anchor_by_key[(rid, anchor_id)] for anchor_id in anchor_ids) for rid in ids],
            dtype=float,
        )
        sampled_metrics = _metric_block(sampled, target_domain, neo, anchor)
        probability_metrics = _metric_block(expected_probability, target_domain, neo, anchor)
        row: dict[str, Any] = {
            "condition": str(item["condition"]),
            "item_id": item_id,
            "source_item_id": source_id,
            "facet_key": str(item["facet_key"]),
            "target_dimension_id": dimension,
            "target_neo_domain": target_domain,
            "n": len(ids),
            "sampled_score_mean": float(np.mean(sampled)),
            "sampled_score_sd": float(np.std(sampled, ddof=1)),
            "probability_high_mean": float(np.mean(expected_probability)),
            "probability_high_sd": float(np.std(expected_probability, ddof=1)),
            "anchor_item_count": len(anchor_ids),
            "anchor_source": "other 21 original Mussel items in the same facet",
            "formula_version": FORMULA_VERSION,
        }
        if item["condition"] == "original":
            historical = [anchor_record_by_key[(rid, source_id)] for rid in ids]
            if all(isinstance(value.get("choice_probabilities"), Mapping) for value in historical):
                historical_probability = np.asarray(
                    [_high_probability(value) for value in historical], dtype=float
                )
                row["historical_vs_retest_probability_spearman"] = _spearman(
                    expected_probability.tolist(), historical_probability.tolist()
                )
            else:
                row["historical_vs_retest_probability_spearman"] = None
            historical_sampled = np.asarray(
                [float(value["score"]) for value in historical], dtype=float
            )
            row["historical_vs_retest_sampled_spearman"] = _spearman(
                sampled.tolist(), historical_sampled.tolist()
            )
            row["historical_vs_retest_exact_option_agreement"] = float(np.mean([
                str(current.get("selected_option_id") or "")
                == str(previous.get("selected_option_id") or "")
                for current, previous in zip(rows, historical)
            ]))
        for prefix, metrics in (("sampled", sampled_metrics), ("probability", probability_metrics)):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        selections = [str(value.get("selected_option_id") or "") for value in rows]
        for option_id in OPTION_IDS:
            count = selections.count(option_id)
            row[f"option_{option_id}_n"] = count
            row[f"option_{option_id}_rate"] = count / len(ids)
        row.update({
            key: value
            for key, value in (probability_by_item.get(item_id) or {}).items()
            if key not in {"item_id", "n"}
        })
        item_rows.append(_json_value(row))

    keyed = {(row["source_item_id"], row["condition"]): row for row in item_rows}
    source_ids = sorted({str(item["source_item_id"]) for item in pilot_items})
    metric_names = (
        "target_neo_spearman",
        "specificity",
        "anchor_citc",
        "high_minus_low_group_score",
    )
    pair_rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        original = keyed.get((source_id, "original"))
        modified = keyed.get((source_id, "high_option_absence"))
        if original is None or modified is None:
            raise ValueError(f"{source_id}缺少原题或缺陷题配对")
        pair: dict[str, Any] = {
            "source_item_id": source_id,
            "facet_key": original["facet_key"],
            "original_item_id": original["item_id"],
            "modified_item_id": modified["item_id"],
            "primary_score": PRIMARY_SCORE,
        }
        detected_count = 0
        estimable_count = 0
        for score_type in ("sampled", "probability"):
            for metric in metric_names:
                original_value = original.get(f"{score_type}_{metric}")
                modified_value = modified.get(f"{score_type}_{metric}")
                delta = (
                    float(original_value) - float(modified_value)
                    if original_value is not None and modified_value is not None
                    else None
                )
                pair[f"{score_type}_original_{metric}"] = original_value
                pair[f"{score_type}_modified_{metric}"] = modified_value
                pair[f"{score_type}_delta_original_minus_modified_{metric}"] = delta
                pair[f"{score_type}_detected_{metric}"] = delta is not None and delta > 0
                if score_type == PRIMARY_SCORE and delta is not None:
                    estimable_count += 1
                    detected_count += int(delta > 0)
        pair["primary_estimable_metric_count"] = estimable_count
        pair["primary_detected_metric_count"] = detected_count
        pair["primary_majority_detected"] = estimable_count == 4 and detected_count >= 3
        pair_rows.append(_json_value(pair))

    metric_summary: list[dict[str, Any]] = []
    for score_type in ("sampled", "probability"):
        for metric in metric_names:
            delta_key = f"{score_type}_delta_original_minus_modified_{metric}"
            deltas = [float(row[delta_key]) for row in pair_rows if row.get(delta_key) is not None]
            metric_summary.append({
                "score_type": score_type,
                "metric": metric,
                "estimable_pairs": len(deltas),
                "detected_pairs": sum(value > 0 for value in deltas),
                "detection_rate": (
                    sum(value > 0 for value in deltas) / len(deltas) if deltas else None
                ),
                "mean_delta_original_minus_modified": float(np.mean(deltas)) if deltas else None,
                "median_delta_original_minus_modified": float(np.median(deltas)) if deltas else None,
            })
    majority_count = sum(bool(row["primary_majority_detected"]) for row in pair_rows)
    original_rows = [row for row in item_rows if row["condition"] == "original"]
    probability_retest = [
        float(row["historical_vs_retest_probability_spearman"])
        for row in original_rows
        if row.get("historical_vs_retest_probability_spearman") is not None
    ]
    sampled_retest = [
        float(row["historical_vs_retest_sampled_spearman"])
        for row in original_rows
        if row.get("historical_vs_retest_sampled_spearman") is not None
    ]
    exact_agreement = [
        float(row["historical_vs_retest_exact_option_agreement"])
        for row in original_rows
        if row.get("historical_vs_retest_exact_option_agreement") is not None
    ]
    summary = {
        "status": "complete",
        "respondent_count": len(ids),
        "pair_count": len(pair_rows),
        "primary_score": PRIMARY_SCORE,
        "primary_majority_rule": "all four metrics estimable and at least three have original > modified",
        "primary_majority_detected_pairs": majority_count,
        "primary_majority_detection_rate": majority_count / len(pair_rows),
        "metric_summary": metric_summary,
        "probability_diagnostics": probability_overall,
        "original_item_retest_diagnostics": {
            "estimable_items": len(probability_retest),
            "mean_historical_vs_retest_probability_spearman": (
                float(np.mean(probability_retest)) if probability_retest else None
            ),
            "mean_historical_vs_retest_sampled_spearman": (
                float(np.mean(sampled_retest)) if sampled_retest else None
            ),
            "mean_exact_option_agreement": (
                float(np.mean(exact_agreement)) if exact_agreement else None
            ),
            "interpretation": (
                "Probability retest correlation checks whether the original-item signal "
                "is reproducible across runs; exact option agreement is expected to be "
                "lower because each run makes an independent categorical draw."
            ),
        },
        "metric_definitions": {
            "target_neo_spearman": "Spearman(item score, independently completed corresponding NEO-FFI domain score)",
            "specificity": "target NEO Spearman minus maximum absolute correlation with the other four NEO domains",
            "anchor_citc": "Pearson(item score, sum of the other 21 original same-facet Mussel items)",
            "high_minus_low_group_score": "mean item score in top 27% target-NEO group minus bottom 27% group",
            "sampled_score": "one reproducibly sampled A-D choice scored A/B=1 and C/D=0",
            "probability_score": "model probability(A)+probability(B); primary because it removes categorical sampling noise",
        },
        "interpretation_boundary": (
            "This controlled computer experiment tests sensitivity to one severe, known option defect. "
            "It does not establish correspondence with human item statistics or general item quality."
        ),
        "formula_version": FORMULA_VERSION,
    }
    return item_rows, pair_rows, _json_value(summary)


def _render_report(
    root: Path,
    summary: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    model_id: str,
    telemetry: Mapping[str, Any],
) -> Path:
    def f(value: Any) -> str:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return "不可估计"
        number = float(value)
        return f"{number:.3f}" if math.isfinite(number) else "不可估计"

    metric_labels = {
        "target_neo_spearman": "目标NEO相关",
        "specificity": "构念特异度",
        "anchor_citc": "21题锚点CITC",
        "high_minus_low_group_score": "高低组区分",
    }
    pair_html = []
    for row in pairs:
        cells = [
            f"<td>{escape(str(row['source_item_id']))}</td>",
            f"<td>{escape(str(row['facet_key']))}</td>",
        ]
        for metric in metric_labels:
            cells.append(f"<td>{f(row.get(f'probability_delta_original_minus_modified_{metric}'))}</td>")
        cells.extend([
            f"<td>{int(row['primary_detected_metric_count'])}/4</td>",
            f"<td>{'是' if row['primary_majority_detected'] else '否'}</td>",
        ])
        pair_html.append("<tr>" + "".join(cells) + "</tr>")
    summary_rows = []
    for row in summary["metric_summary"]:
        summary_rows.append(
            "<tr>"
            f"<td>{'概率分' if row['score_type'] == 'probability' else '抽样0/1分'}</td>"
            f"<td>{escape(metric_labels[str(row['metric'])])}</td>"
            f"<td>{row['detected_pairs']}/{row['estimable_pairs']}</td>"
            f"<td>{f(row.get('detection_rate'))}</td>"
            f"<td>{f(row.get('mean_delta_original_minus_modified'))}</td>"
            "</tr>"
        )
    report = root / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>Mussel选项缺陷敏感性实验</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:#20252b}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd2d8;padding:7px;text-align:left}}th{{background:#eef2f6}}.note{{background:#fff6d8;padding:12px;border-left:4px solid #d99b00}}.good{{font-size:1.3em}}</style>
<h1>Mussel“高水平选项缺失”敏感性实验</h1>
<p>同一批虚拟被试={summary['respondent_count']}；配对题目={summary['pair_count']}；模型=<code>{escape(model_id)}</code>。</p>
<p class='good'>主分析检出 <strong>{summary['primary_majority_detected_pairs']}/{summary['pair_count']}</strong> 道缺陷题（{f(summary['primary_majority_detection_rate'])}）。</p>
<div class='note'>主分析使用模型给出的A/B总概率，避免单次分类抽样噪声。正差表示原题指标高于缺陷题；“检出”是预先规定的4项中至少3项下降，并不是通行的心理测量合格标准。</div>
<h2>逐题配对结果（原题－缺陷题）</h2>
<table><thead><tr><th>源题</th><th>facet</th><th>Δ目标NEO相关</th><th>Δ特异度</th><th>Δ锚点CITC</th><th>Δ高低组区分</th><th>下降项</th><th>多数检出</th></tr></thead><tbody>{''.join(pair_html)}</tbody></table>
<h2>总体敏感性</h2>
<table><thead><tr><th>计分口径</th><th>指标</th><th>检出/可估计</th><th>检出率</th><th>平均差值</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<h2>原题跨批次复现</h2>
<p>10道原题与历史同模型作答的平均概率排序相关={f(summary['original_item_retest_diagnostics'].get('mean_historical_vs_retest_probability_spearman'))}；抽样0/1得分相关={f(summary['original_item_retest_diagnostics'].get('mean_historical_vs_retest_sampled_spearman'))}；完全相同选项比例={f(summary['original_item_retest_diagnostics'].get('mean_exact_option_agreement'))}。概率相关用于确认模型对原题的相对倾向可复现；后两项还包含两次独立概率抽样噪声。</p>
<h2>指标含义</h2>
<ul><li>目标NEO相关：单题能否随独立会话完成的对应NEO-FFI维度变化。</li><li>构念特异度：目标相关减去四个非目标维度绝对相关的最大值。</li><li>21题锚点CITC：单题与同facet其余21道原始Mussel题总分的相关。</li><li>高低组区分：目标NEO最高27%与最低27%的平均题分差。</li></ul>
<h2>审计边界</h2>
<p>这是一个已知严重缺陷的计算机内部敏感性实验。它可以回答当前虚拟被试是否能区分这类选项退化，但不能证明其指标等同真人，也不能推广到情境污染、措辞歧义或社会赞许等其他缺陷。</p>
<p>模型调用={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；Token={telemetry.get('total_tokens')}。</p>
</html>""",
        encoding="utf-8",
    )
    return report


async def run_mussel_item_defect_pilot(
    config: MusselItemDefectConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    paths = _resolve_paths(config)
    source_items, source_metadata = load_mussel_items(paths["mussel"])
    original, modified, stimulus_metadata = load_pilot_items(paths["stimuli"], paths["mussel"])
    pilot_items = original + modified
    subject_ids, anchor_records, reference_manifest = _load_reference(
        paths["reference"], paths["anchor_responses"], source_items
    )
    neo_scores = _load_neo_scores(paths["neo_scores"], subject_ids)
    pool, profiles = load_legacy_pool(paths["legacy_project"])
    summaries, summary_path, summary_manifest = load_comparison_summaries(
        paths["legacy_project"], paths["summary_pool"]
    )
    missing = [rid for rid in subject_ids if rid not in profiles or rid not in summaries]
    if missing:
        raise ValueError(f"冻结100名被试缺少画像或总结：{missing[:3]}")

    model = get_model(config.model_id)
    runnable, structured_output_method = with_compatible_structured_output(
        model, ChoiceProbabilityOutput
    )
    model_id = _model_id(model, config.model_id)
    reference_model = str(reference_manifest.get("embodied_condition_model") or "")
    if model_id != reference_model:
        raise ValueError(
            f"当前模型{model_id}与固定Mussel锚点模型{reference_model}不同；"
            "为避免把模型差异混入题目缺陷效应，必须使用同一模型"
        )
    expected_persona_version = PERSONA_PROMPT_VERSIONS["summary_embodied_probability"]
    if reference_manifest.get("persona_prompt_version") != expected_persona_version:
        raise ValueError("固定Mussel锚点与当前具身概率人格提示词版本不同")

    root = (
        config.output.resolve()
        if config.output
        else paths["experiment"]
        / "mussel_item_defect_pilot"
        / "high_option_absence"
    )
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"mussel-item-defect-{root.name}"
    binding = fingerprint({
        "subject_ids": subject_ids,
        "stimuli_sha256": _sha256_file(paths["stimuli"]),
        "mussel_sha256": _sha256_file(paths["mussel"]),
        "anchor_sha256": _sha256_file(paths["anchor_responses"]),
        "neo_sha256": _sha256_file(paths["neo_scores"]),
        "summary_sha256": _sha256_file(summary_path),
        "model_id": model_id,
        "persona_prompt_version": expected_persona_version,
        "sampling_seed": config.sampling_seed,
        "formula_version": FORMULA_VERSION,
    })
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("binding") != binding:
            raise ValueError("指定输出目录已绑定不同输入，拒绝混合或覆盖")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "experiment": str(paths["experiment"]),
        "respondent_count": len(subject_ids),
        "respondent_ids": subject_ids,
        "conditions": list(CONDITIONS),
        "items_per_condition": 10,
        "model_id": model_id,
        "structured_output_method": structured_output_method,
        "persona_mode": "summary_embodied_probability",
        "persona_prompt_version": expected_persona_version,
        "sampling_seed": config.sampling_seed,
        "reference_mussel_run": str(paths["reference"]),
        "neo_scores_source": str(paths["neo_scores"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(root / "config.json", _json_value(asdict(config)))
    write_json(root / "participants.json", {
        "respondent_ids": subject_ids,
        "count": len(subject_ids),
        "same_people_across_conditions": True,
        "independent_model_call_per_item_version": True,
    })
    write_json(root / "stimuli.json", {
        "metadata": stimulus_metadata,
        "source_metadata": source_metadata,
        "items": pilot_items,
    })

    run_config = LegacyABCConfig(
        experiment=paths["experiment"],
        legacy_project=paths["legacy_project"],
        model_id=model_id,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_seconds=config.timeout_seconds,
        persona_mode="summary_embodied_probability",
        summary_pool=paths["summary_pool"],
        respondent_source="legacy_100",
        sampling_seed=config.sampling_seed,
    )
    started = perf_counter()
    try:
        with output_scope(root / "runtime", telemetry=root / "telemetry"), run_context(run_id):
            pilot_records = await _run_sjt_stage(
                "MUSSEL_ITEM_DEFECT",
                {"items": pilot_items},
                subject_ids,
                profiles,
                summaries,
                runnable,
                model_id,
                root / "responses",
                run_config,
            )
        item_rows, pair_rows, summary = analyze_item_defect_pilot(
            pilot_items=pilot_items,
            pilot_records=pilot_records,
            source_items=source_items,
            anchor_records=anchor_records,
            neo_scores=neo_scores,
            subject_ids=subject_ids,
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
            "wall_time_seconds_this_invocation": round(perf_counter() - started, 3),
            "price": None,
            "note": "未提供适用的输入/输出/缓存单价，因此不推测费用。",
        }
        write_json(root / "cost.json", _json_value(cost))
        report = _render_report(root, summary, pair_rows, model_id, telemetry)
        finished = {
            **manifest,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "report": str(report),
            "summary": summary,
            "summary_source": str(summary_path),
            "summary_condition_id": (
                summary_manifest.get("condition_id") if summary_manifest else None
            ),
            "pool_id": pool.get("pool_id"),
        }
        write_json(manifest_path, _json_value(finished))
        return root, {"report": report, "summary": summary}
    except Exception as exc:
        write_json(manifest_path, {
            **manifest,
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        raise
