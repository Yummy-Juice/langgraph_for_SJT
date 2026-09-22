"""Blind four-way classification of controlled Mussel item defects.

The classifier sees the target facet definition, scenario, response options,
and scoring key.  It never sees source IDs, version names, true labels, or
manipulation rationales.  Five intact source items and fifteen controlled
defect versions form a balanced four-class set.

This is a manipulation check, not an independent psychometric gold standard.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sjt_system.agent.client import (
    build_json_output_instruction,
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.authoring.construct_registry import construct_selection_catalog
from sjt_system.evaluation.respondents import build_score_dimension_catalog
from sjt_system.evaluation.simulation import _invoke_with_retry
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import _json_value, _model_id, _sha256_file, _telemetry_summary
from .storage import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STIMULI = (
    Path(__file__).resolve().parent
    / "stimuli"
    / "mussel_three_defect_types_15.json"
)
LABELS = (
    "intact",
    "construct_shift",
    "low_discrimination",
    "social_desirability",
)
PROMPT_VERSION = "blind-mussel-defect-classification-v3"
FORMULA_VERSION = "balanced-four-class-majority-vote-v1"


class DefectClassificationOutput(BaseModel):
    # Some OpenAI-compatible providers append harmless explanatory fields such
    # as ``*_note`` even when the JSON schema does not request them.  Keep the
    # scientific payload strict (required fields, enums, and ranges below), but
    # ignore unrelated transport extras instead of discarding an otherwise
    # valid classification.
    model_config = ConfigDict(extra="ignore")

    classification: Literal[
        "intact",
        "construct_shift",
        "low_discrimination",
        "social_desirability",
    ]
    confidence: float = Field(ge=0, le=1)
    target_construct_relevance: int = Field(ge=1, le=5)
    option_gradient_clarity: int = Field(ge=1, le=5)
    social_desirability_pressure: int = Field(ge=1, le=5)
    non_target_contamination: int = Field(ge=1, le=5)
    likely_non_target_construct: str | None
    rationale: str


@dataclass(frozen=True)
class DefectClassifierConfig:
    experiment: Path
    stimuli_path: Path = DEFAULT_STIMULI
    model_id: str | None = None
    repeats: int = 3
    max_concurrency: int = 10
    max_retries: int = 2
    timeout_seconds: float | None = None
    seed: int = 20260915
    output: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.repeats <= 10:
            raise ValueError("repeats必须在1至10之间")
        if not 1 <= self.max_concurrency <= 30:
            raise ValueError("max_concurrency必须在1至30之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed必须是整数")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON必须是对象：{path}")
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


def _facet_specs() -> dict[str, dict[str, Any]]:
    return {
        str(row["dimension_id"]): dict(row)
        for row in build_score_dimension_catalog(construct_selection_catalog())
        if row.get("level") == "facet" and row.get("dimension_id")
    }


def build_blind_items(stimuli_path: Path = DEFAULT_STIMULI, *, seed: int = 20260915) -> list[dict[str, Any]]:
    stimulus = _read_json(stimuli_path)
    defects = stimulus.get("items")
    if not isinstance(defects, list) or len(defects) != 15:
        raise ValueError("盲态分类要求15道受控缺陷题")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in defects:
        if not isinstance(row, Mapping):
            raise ValueError("缺陷刺激包含非对象")
        by_source[str(row.get("source_item_id") or "")].append(dict(row))
    if len(by_source) != 5 or any(len(rows) != 3 for rows in by_source.values()):
        raise ValueError("盲态分类要求5道源题各3种缺陷")

    items: list[dict[str, Any]] = []
    for source_id, rows in sorted(by_source.items()):
        first = rows[0]
        items.append({
            "internal_id": f"{source_id}-INTACT-CONTROL",
            "source_item_id": source_id,
            "true_label": "intact",
            "target_dimension_id": first["target_dimension_id"],
            "scenario": first["scenario"],
            "response_instruction": first["response_instruction"],
            "options": dict(first["original_options"]),
            "scoring_key": dict(stimulus["scoring_key"]),
        })
        for row in rows:
            items.append({
                "internal_id": str(row["item_id"]),
                "source_item_id": source_id,
                "true_label": str(row["defect_type"]),
                "target_dimension_id": str(row["target_dimension_id"]),
                "scenario": str(row["scenario"]),
                "response_instruction": str(row["response_instruction"]),
                "options": dict(row["modified_options"]),
                "scoring_key": dict(stimulus["scoring_key"]),
            })
    counts = Counter(str(row["true_label"]) for row in items)
    if counts != Counter({label: 5 for label in LABELS}):
        raise ValueError(f"四分类必须完全平衡，实际={dict(counts)}")

    import random

    rng = random.Random(seed)
    rng.shuffle(items)
    for index, item in enumerate(items, 1):
        item["blind_id"] = f"BLIND-{index:03d}"
    return items


def _display_options(item: Mapping[str, Any], repeat: int, seed: int) -> list[dict[str, Any]]:
    option_ids = list(item["options"])
    import random

    rng_seed = int(fingerprint({"seed": seed, "blind_id": item["blind_id"], "repeat": repeat})[:16], 16)
    random.Random(rng_seed).shuffle(option_ids)
    return [
        {
            "display_id": chr(ord("A") + index),
            "text": str(item["options"][option_id]),
            "score": int(item["scoring_key"][option_id]),
        }
        for index, option_id in enumerate(option_ids)
    ]


def build_classification_messages(
    item: Mapping[str, Any],
    *,
    facet_spec: Mapping[str, Any],
    repeat: int,
    seed: int,
) -> list[tuple[str, str]]:
    options = _display_options(item, repeat, seed)
    option_lines = "\n".join(
        f"{row['display_id']}. {row['text']}（计分={row['score']}）"
        for row in options
    )
    system = (
        "你是一名独立的心理测量题目审查专家。请判断一道人格型情境判断题的主要状态。\n\n"
        "四个互斥类别：\n"
        "intact：题目与目标构念相关，选项形成可解释的高低梯度，且没有更突出的构念污染或社会赞许线索。\n"
        "construct_shift：选项主要测量另一个非目标构念，目标构念的高低含义被替换或显著污染。\n"
        "low_discrimination：四个选项在目标构念水平上过于相近或混合，计分边界缺乏清晰行为依据。\n"
        "social_desirability：高分与低分选项主要由明显的道德、礼貌、诚实、负责或违规线索区分，社会规范压过目标人格差异。\n\n"
        "请选择最主要的一个类别。不要推测题目来自哪个实验，不要根据措辞风格判断版本。\n\n"
        "输出必须是单个JSON对象，并且只能包含以下8个字段，不得增加、删除或改名，不得增加任何*_note字段：\n"
        "classification, confidence, target_construct_relevance, "
        "option_gradient_clarity, social_desirability_pressure, "
        "non_target_contamination, likely_non_target_construct, rationale。\n"
        "classification只能是上述四个英文类别之一；confidence为0至1；四项评分均为1至5整数；"
        "没有可识别的非目标构念时，likely_non_target_construct必须为null。\n\n"
        + build_json_output_instruction(DefectClassificationOutput)
    )
    user = (
        f"目标facet：{facet_spec.get('display_label')}\n"
        f"定义：{facet_spec.get('definition')}\n"
        f"高水平行为：{facet_spec.get('high_behavior')}\n"
        f"低水平行为：{facet_spec.get('low_behavior')}\n\n"
        f"情境：{item['scenario']}\n"
        f"作答要求：{item['response_instruction']}\n"
        f"选项：\n{option_lines}\n\n"
        "请同时给出四项1—5评分：目标构念相关性、选项梯度清晰度、社会赞许压力、非目标构念污染。"
    )
    return [("system", system), ("human", user)]


def _validate_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    return DefectClassificationOutput.model_validate(value).model_dump()


def _majority_prediction(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(row["prediction"]) for row in rows)
    best_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return candidates[0]
    mean_confidence = {
        label: float(np.mean([
            float(row["confidence"]) for row in rows if row["prediction"] == label
        ]))
        for label in candidates
    }
    return sorted(candidates, key=lambda label: (-mean_confidence[label], LABELS.index(label)))[0]


def _classification_metrics(true: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    if len(true) != len(predicted) or not true:
        raise ValueError("分类指标需要等长且非空的真值和预测")
    confusion = {
        actual: {label: 0 for label in LABELS}
        for actual in LABELS
    }
    for actual, guess in zip(true, predicted):
        if actual not in LABELS or guess not in LABELS:
            raise ValueError("分类标签超出预设四类")
        confusion[actual][guess] += 1
    per_class = []
    for label in LABELS:
        tp = confusion[label][label]
        fn = sum(confusion[label].values()) - tp
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        per_class.append({
            "label": label,
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    valid_f1 = [float(row["f1"]) for row in per_class if row["f1"] is not None]
    return _json_value({
        "accuracy": sum(a == b for a, b in zip(true, predicted)) / len(true),
        "macro_f1": float(np.mean(valid_f1)) if valid_f1 else None,
        "per_class": per_class,
        "confusion": confusion,
    })


def analyze_predictions(
    *,
    items: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    expected = {
        (str(item["blind_id"]), repeat)
        for item in items
        for repeat in range(1, repeats + 1)
    }
    actual = {(str(row.get("blind_id")), int(row.get("repeat", 0))) for row in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError(f"分类作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    item_by_id = {str(item["blind_id"]): item for item in items}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["blind_id"])].append(row)
    item_rows = []
    for blind_id in sorted(grouped):
        rows = sorted(grouped[blind_id], key=lambda row: int(row["repeat"]))
        item = item_by_id[blind_id]
        majority = _majority_prediction(rows)
        agreement = max(Counter(str(row["prediction"]) for row in rows).values()) / len(rows)
        item_rows.append({
            "blind_id": blind_id,
            "source_item_id": item["source_item_id"],
            "target_dimension_id": item["target_dimension_id"],
            "true_label": item["true_label"],
            "majority_prediction": majority,
            "correct": majority == item["true_label"],
            "agreement_rate": agreement,
            "mean_confidence": float(np.mean([float(row["confidence"]) for row in rows])),
            "mean_target_construct_relevance": float(np.mean([float(row["target_construct_relevance"]) for row in rows])),
            "mean_option_gradient_clarity": float(np.mean([float(row["option_gradient_clarity"]) for row in rows])),
            "mean_social_desirability_pressure": float(np.mean([float(row["social_desirability_pressure"]) for row in rows])),
            "mean_non_target_contamination": float(np.mean([float(row["non_target_contamination"]) for row in rows])),
            "predictions": [str(row["prediction"]) for row in rows],
        })
    item_metrics = _classification_metrics(
        [str(row["true_label"]) for row in item_rows],
        [str(row["majority_prediction"]) for row in item_rows],
    )
    call_metrics = _classification_metrics(
        [str(row["true_label"]) for row in records],
        [str(row["prediction"]) for row in records],
    )
    confusion_rows = [
        {
            "true_label": actual_label,
            **{f"predicted_{label}": item_metrics["confusion"][actual_label][label] for label in LABELS},
        }
        for actual_label in LABELS
    ]
    summary = {
        "status": "complete",
        "item_count": len(items),
        "repeats": repeats,
        "call_count": len(records),
        "class_balance": dict(Counter(str(item["true_label"]) for item in items)),
        "item_level_majority": item_metrics,
        "call_level": call_metrics,
        "mean_item_agreement": float(np.mean([float(row["agreement_rate"]) for row in item_rows])),
        "formula_version": FORMULA_VERSION,
        "interpretation_boundary": "这是单一模型对人为标签的盲态操纵检验，不是独立人类专家金标准，也不能证明题目在真人样本中一定存在同类缺陷。",
    }
    return _json_value(item_rows), _json_value(summary), _json_value(confusion_rows)


def _fmt(value: Any) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.3f}"


def _render_report(
    root: Path,
    *,
    summary: Mapping[str, Any],
    item_rows: Sequence[Mapping[str, Any]],
    confusion_rows: Sequence[Mapping[str, Any]],
    model_id: str,
    telemetry: Mapping[str, Any],
) -> Path:
    item_html = "".join(
        "<tr>"
        f"<td>{escape(str(row['blind_id']))}</td><td>{escape(str(row['target_dimension_id']))}</td>"
        f"<td>{escape(str(row['true_label']))}</td><td>{escape(str(row['majority_prediction']))}</td>"
        f"<td>{'是' if row['correct'] else '否'}</td><td>{_fmt(row['agreement_rate'])}</td>"
        f"<td>{_fmt(row['mean_confidence'])}</td></tr>"
        for row in item_rows
    )
    confusion_html = "".join(
        "<tr>" + f"<td>{escape(str(row['true_label']))}</td>" + "".join(
            f"<td>{row[f'predicted_{label}']}</td>" for label in LABELS
        ) + "</tr>"
        for row in confusion_rows
    )
    class_html = "".join(
        "<tr>"
        f"<td>{escape(str(row['label']))}</td><td>{row['support']}</td>"
        f"<td>{_fmt(row['precision'])}</td><td>{_fmt(row['recall'])}</td><td>{_fmt(row['f1'])}</td></tr>"
        for row in summary["item_level_majority"]["per_class"]
    )
    report = root / "report.html"
    report.write_text(
        f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Mussel缺陷盲态分类</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1400px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd2d8;padding:7px;text-align:left}}th{{background:#eef2f6}}.note{{background:#fff6d8;padding:12px;border-left:4px solid #d99b00}}.result{{font-size:1.3em}}</style>
<h1>Mussel题目缺陷类型盲态分类</h1>
<p>模型=<code>{escape(model_id)}</code>；20道题；四类各5题；每题重复{summary['repeats']}次。</p>
<p class='result'>题目多数投票准确率=<strong>{_fmt(summary['item_level_majority']['accuracy'])}</strong>；宏平均F1=<strong>{_fmt(summary['item_level_majority']['macro_f1'])}</strong>；逐次准确率={_fmt(summary['call_level']['accuracy'])}。</p>
<div class='note'>分类模型只看到目标facet定义、题目、选项和计分，不看到真实标签、版本名、源题号或缺陷说明。本结果只是模型操纵检验，不是人类专家金标准。</div>
<h2>逐题分类</h2><table><thead><tr><th>盲号</th><th>目标facet</th><th>真实类</th><th>多数预测</th><th>正确</th><th>一致率</th><th>平均置信度</th></tr></thead><tbody>{item_html}</tbody></table>
<h2>混淆矩阵</h2><table><thead><tr><th>真实\\预测</th>{''.join(f'<th>{escape(label)}</th>' for label in LABELS)}</tr></thead><tbody>{confusion_html}</tbody></table>
<h2>分类别指标</h2><table><thead><tr><th>类别</th><th>数量</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{class_html}</tbody></table>
<p>调用={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；Token={telemetry.get('total_tokens')}；模型耗时={telemetry.get('duration_ms')} ms。</p></html>""",
        encoding="utf-8",
    )
    return report


async def run_defect_classifier(config: DefectClassifierConfig) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"实验目录不存在：{experiment}")
    stimuli_path = config.stimuli_path.resolve()
    items = build_blind_items(stimuli_path, seed=config.seed)
    specs = _facet_specs()
    missing = sorted({str(item["target_dimension_id"]) for item in items} - set(specs))
    if missing:
        raise ValueError("构念注册表缺少：" + "、".join(missing))
    model = get_model(config.model_id)
    runnable, structured_output_method = with_compatible_structured_output(model, DefectClassificationOutput)
    model_id = _model_id(model, config.model_id)
    root = (
        config.output.resolve()
        if config.output
        else experiment / "mussel_item_defect_pilot" / "defect_type_classification_v3"
    )
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"mussel-defect-classifier-{root.name}"
    binding = fingerprint({
        "stimuli_sha256": _sha256_file(stimuli_path),
        "model_id": model_id,
        "repeats": config.repeats,
        "seed": config.seed,
        "prompt_version": PROMPT_VERSION,
        "formula_version": FORMULA_VERSION,
    })
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        if previous.get("binding") != binding:
            raise ValueError("分类输出目录已绑定不同输入，拒绝混合")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "model_id": model_id,
        "structured_output_method": structured_output_method,
        "item_count": len(items),
        "repeats": config.repeats,
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(root / "config.json", _json_value({
        "experiment": str(experiment),
        "stimuli_path": str(stimuli_path),
        "model_id": model_id,
        "repeats": config.repeats,
        "max_concurrency": config.max_concurrency,
        "max_retries": config.max_retries,
        "timeout_seconds": config.timeout_seconds,
        "seed": config.seed,
    }))
    write_json(root / "blind_map.json", {
        "warning": "此文件含实验真值；没有发送给分类模型。",
        "items": [{
            "blind_id": item["blind_id"],
            "internal_id": item["internal_id"],
            "source_item_id": item["source_item_id"],
            "true_label": item["true_label"],
            "target_dimension_id": item["target_dimension_id"],
        } for item in items],
    })
    response_path = root / "predictions.jsonl"
    existing = {
        (str(row.get("blind_id")), int(row.get("repeat", 0)))
        for row in _load_jsonl(response_path)
    }
    jobs = [
        (item, repeat)
        for item in items
        for repeat in range(1, config.repeats + 1)
        if (str(item["blind_id"]), repeat) not in existing
    ]
    semaphore = asyncio.Semaphore(config.max_concurrency)
    lock = asyncio.Lock()
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    completed = len(existing)
    total = len(items) * config.repeats
    started = perf_counter()

    async def one(
        item: Mapping[str, Any],
        repeat: int,
        *,
        max_retries_override: int | None = None,
    ) -> None:
        nonlocal completed
        prediction = await _invoke_with_retry(
            runnable,
            build_classification_messages(
                item,
                facet_spec=specs[str(item["target_dimension_id"])],
                repeat=repeat,
                seed=config.seed,
            ),
            semaphore=semaphore,
            validator=_validate_prediction,
            max_retries=(
                config.max_retries
                if max_retries_override is None
                else max_retries_override
            ),
            retry_delay_seconds=1.0,
            request_timeout_seconds=timeout,
            job_label=f"blind defect classification {item['blind_id']} repeat {repeat}",
        )
        record = {
            "record_type": "blind_mussel_defect_classification",
            "run_id": run_id,
            "blind_id": item["blind_id"],
            "repeat": repeat,
            "prediction": prediction.pop("classification"),
            **prediction,
            "true_label": item["true_label"],
            "target_dimension_id": item["target_dimension_id"],
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
        }
        async with lock:
            with response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed == total or completed % max(1, total // 10) == 0:
                print(f"[盲态分类] {completed}/{total} ({completed / total:.0%})", flush=True)

    failure_phase = "single_item_output_preflight"
    bulk_started = False
    try:
        with output_scope(root / "runtime", telemetry=root / "telemetry"), run_context(run_id):
            remaining_jobs = list(jobs)
            # A fresh run first proves that the selected provider/model obeys
            # the JSON contract.  This prevents one formatting incompatibility
            # from consuming the entire balanced classification batch.
            if not existing and remaining_jobs:
                preflight_item, preflight_repeat = remaining_jobs.pop(0)
                print("[盲态分类] JSON输出格式预检 1/1", flush=True)
                try:
                    await one(
                        preflight_item,
                        preflight_repeat,
                        max_retries_override=0,
                    )
                except Exception as exc:
                    write_json(root / "errors.json", {
                        "phase": failure_phase,
                        "bulk_started": False,
                        "errors": [str(exc)],
                    })
                    raise RuntimeError(
                        "盲态分类JSON输出预检失败，批量任务未启动："
                        f"{exc}"
                    ) from exc
                print("[盲态分类] JSON输出格式预检通过，开始批量分类", flush=True)

            failure_phase = "bulk_classification"
            bulk_started = bool(remaining_jobs)
            results = await asyncio.gather(
                *(one(item, repeat) for item, repeat in remaining_jobs),
                return_exceptions=True,
            )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            write_json(root / "errors.json", {
                "phase": failure_phase,
                "bulk_started": bulk_started,
                "errors": [str(error) for error in errors],
            })
            raise RuntimeError(f"盲态分类有{len(errors)}次失败；首个错误：{errors[0]}")
        failure_phase = "analysis_and_report"
        records = _load_jsonl(response_path)
        item_rows, summary, confusion_rows = analyze_predictions(
            items=items,
            records=records,
            repeats=config.repeats,
        )
        write_csv(root / "item_results.csv", item_rows)
        write_json(root / "item_results.json", item_rows)
        write_csv(root / "confusion_matrix.csv", confusion_rows)
        write_json(root / "summary.json", summary)
        telemetry = _telemetry_summary(root, run_id)
        cost = {
            **telemetry,
            "wall_time_seconds_this_invocation": round(perf_counter() - started, 3),
            "price": None,
            "note": "未提供适用单价，不推测费用。",
        }
        write_json(root / "cost.json", _json_value(cost))
        report = _render_report(
            root,
            summary=summary,
            item_rows=item_rows,
            confusion_rows=confusion_rows,
            model_id=model_id,
            telemetry=telemetry,
        )
        finished = {
            **manifest,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "report": str(report),
        }
        write_json(manifest_path, _json_value(finished))
        return root, {"summary": summary, "report": report, "cost": cost}
    except Exception as exc:
        write_json(manifest_path, {
            **manifest,
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "failure_phase": failure_phase,
            "bulk_started": bulk_started,
            "error": str(exc),
        })
        raise
