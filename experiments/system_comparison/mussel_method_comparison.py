"""Compare historical and embodied-probability virtual respondents on Mussel.

The historical condition reuses the frozen 100-person Mussel responses from
the legacy project.  The embodied condition administers the same 110 items to
the same respondents with the summary-only, subjective-consequence,
probability-sampling protocol.  Both conditions are rescored with the same
explicit A/B=1, C/D=0 key.  Existing NEO-FFI scores are retained only as a
broader parent-domain reference; strict facet-matched validity is provided by
the append-only ``mussel-ipip`` experiment.

This is a system-level comparison.  Because the historical Mussel responses
were generated with the historical runtime/model, condition differences must
not be attributed solely to the embodied prompt.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sjt_system.agent.client import get_model, with_compatible_structured_output
from sjt_system.evaluation.reference_questionnaires import (
    MUSSEL_FACET_TO_DIMENSION,
    MUSSEL_SCORING_VERSION,
    load_mussel_items,
    score_mussel_records,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import (
    DEFAULT_IMPORTED_SUMMARY_POOL,
    DEFAULT_LEGACY_PROJECT,
    NEO_DOMAINS,
    PERSONA_PROMPT_VERSIONS,
    ChoiceProbabilityOutput,
    LegacyABCConfig,
    _alpha,
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
from .storage import write_csv, write_json


CONDITIONS = ("legacy_historical", "embodied_probability")
DIMENSION_TO_NEO = {
    "openness_ideas": "O",
    "conscientiousness_self_discipline": "C",
    "extraversion_gregariousness": "E",
    "agreeableness_compliance": "A",
    "neuroticism_self_consciousness": "N",
}
PROMPT_VERSION = "mussel-legacy-vs-embodied-probability-v1"
FORMULA_VERSION = "mussel-method-comparison-neo-v1"


@dataclass(frozen=True)
class MusselComparisonConfig:
    experiment: Path
    legacy_project: Path = DEFAULT_LEGACY_PROJECT
    summary_pool: Path = DEFAULT_IMPORTED_SUMMARY_POOL
    old_abc_result: Path | None = None
    new_abc_result: Path | None = None
    mussel_path: Path | None = None
    model_id: str | None = None
    max_concurrency: int = 30
    max_retries: int = 2
    timeout_seconds: float | None = None
    sampling_seed: int = 20260913
    output: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed, int
        ):
            raise ValueError("sampling_seed必须是整数")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}第{line_number}行不是有效JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是对象")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_value(dict(record)), ensure_ascii=False))
            handle.write("\n")


def _resolve_inputs(config: MusselComparisonConfig) -> dict[str, Path]:
    experiment = config.experiment.resolve()
    paths = {
        "experiment": experiment,
        "legacy_project": config.legacy_project.resolve(),
        "summary_pool": config.summary_pool.resolve(),
        "old_abc": (
            config.old_abc_result.resolve()
            if config.old_abc_result
            else experiment / "legacy_pool_abc"
        ),
        "new_abc": (
            config.new_abc_result.resolve()
            if config.new_abc_result
            else experiment / "legacy_summary_embodied_probability_abc"
        ),
        "mussel": (
            config.mussel_path.resolve()
            if config.mussel_path
            else Path(__file__).resolve().parents[2] / "docs" / "mussel_zh.json"
        ),
    }
    required = [
        paths["legacy_project"] / "sjt_system" / "data" / "virtual_respondents.json",
        paths["summary_pool"] / "persona_summaries.jsonl",
        paths["old_abc"] / "manifest.json",
        paths["old_abc"] / "neo_ffi" / "scores.csv",
        paths["new_abc"] / "manifest.json",
        paths["new_abc"] / "neo_ffi" / "scores.csv",
        paths["mussel"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Mussel方法比较缺少输入：" + "；".join(missing))
    return paths


def _load_subject_ids(old_abc: Path, new_abc: Path) -> list[str]:
    old_manifest = json.loads((old_abc / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((new_abc / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("status") != "complete" or new_manifest.get("status") != "complete":
        raise ValueError("新旧A/B/C参照实验必须均已完成")
    subject_ids = [str(value) for value in old_manifest.get("respondent_ids") or []]
    if len(subject_ids) != 100 or len(set(subject_ids)) != 100:
        raise ValueError("历史条件必须绑定冻结且唯一的100名被试")
    new_ids = {str(value) for value in new_manifest.get("respondent_ids") or []}
    if not set(subject_ids) <= new_ids:
        raise ValueError("新条件NEO-FFI没有覆盖全部历史100名被试")
    return subject_ids


def _legacy_response_paths(legacy_project: Path) -> dict[str, Path]:
    root = legacy_project / "experiments" / "mussel_validation" / "out"
    paths = {
        facet_key: root / f"mussel_{facet_key}_responses.jsonl"
        for facet_key in MUSSEL_FACET_TO_DIMENSION
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("旧Mussel作答文件不完整：" + "；".join(missing))
    return paths


def load_historical_mussel_records(
    legacy_project: Path,
    subject_ids: Sequence[str],
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Import and rescore the frozen historical responses with one fixed key."""

    paths = _legacy_response_paths(legacy_project)
    subject_set = set(subject_ids)
    current_items = {
        (str(item["facet_key"]), str(item["source_item_id"])): dict(item)
        for item in items
    }
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for facet_key, path in paths.items():
        for row in _jsonl(path):
            respondent_id = str(row.get("respondent_id") or "")
            if respondent_id not in subject_set:
                continue
            old_item_id = str(row.get("item_id") or "")
            try:
                item_number = str(int(old_item_id.rsplit("_", 1)[-1]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"旧Mussel题号无法解析：{old_item_id}") from exc
            item = current_items.get((facet_key, item_number))
            if item is None:
                raise ValueError(f"旧Mussel题目无法映射到当前题本：{old_item_id}")
            selected = str(row.get("selected_option_id") or "").strip().upper()
            if selected not in item["scoring_key"]:
                raise ValueError(f"旧Mussel作答选项非法：{respondent_id}/{old_item_id}")
            item_id = str(item["item_id"])
            key = (respondent_id, item_id)
            if key in seen:
                raise ValueError(f"旧Mussel作答重复：{key}")
            seen.add(key)
            records.append({
                "record_type": "mussel_method_comparison_response",
                "condition": "legacy_historical",
                "respondent_id": respondent_id,
                "item_id": item_id,
                "facet_key": facet_key,
                "target_dimension_id": item["target_dimension_id"],
                "selected_option_id": selected,
                "score": int(item["scoring_key"][selected]),
                "scoring_version": MUSSEL_SCORING_VERSION,
                "source_item_id": old_item_id,
                "source_path": str(path),
            })
    expected = {
        (respondent_id, str(item["item_id"]))
        for respondent_id in subject_ids
        for item in items
    }
    if seen != expected or len(records) != len(expected):
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"旧Mussel作答应为{len(expected)}条，实际为{len(records)}条；"
            f"缺失={missing[:3]}；额外={extra[:3]}"
        )
    return records, {facet: _sha256_file(path) for facet, path in paths.items()}


def _load_neo_scores(path: Path, subject_ids: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "respondent_id" not in frame or not set(NEO_DOMAINS) <= set(frame.columns):
        raise ValueError(f"NEO-FFI得分文件列不完整：{path}")
    frame["respondent_id"] = frame["respondent_id"].astype(str)
    if frame["respondent_id"].duplicated().any():
        raise ValueError(f"NEO-FFI得分被试重复：{path}")
    frame = frame.set_index("respondent_id").reindex(list(subject_ids))
    if frame[list(NEO_DOMAINS)].isna().any().any():
        raise ValueError(f"NEO-FFI不能与冻结100名被试完整对齐：{path}")
    frame.index.name = "respondent_id"
    return frame[list(NEO_DOMAINS)].astype(float)


def _condition_metrics(
    *,
    condition: str,
    records: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    neo_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    long_frame, facet_scores = score_mussel_records(
        records,
        expected_respondent_ids=subject_ids,
        items=items,
    )
    facet_scores = facet_scores.reindex(index=list(subject_ids))
    item_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    item_map = {str(item["item_id"]): dict(item) for item in items}
    for dimension_id, target_domain in DIMENSION_TO_NEO.items():
        dimension_items = [
            str(item["item_id"])
            for item in items
            if str(item["target_dimension_id"]) == dimension_id
        ]
        subset = long_frame[long_frame["target_dimension_id"] == dimension_id]
        matrix = subset.pivot(
            index="respondent_id", columns="item_id", values="score"
        ).reindex(index=list(subject_ids), columns=dimension_items)
        if matrix.isna().any().any() or matrix.shape != (len(subject_ids), 22):
            raise ValueError(f"{condition}/{dimension_id}必须形成100×22完整矩阵")
        total = matrix.sum(axis=1)
        citcs: list[float] = []
        for item_id in dimension_items:
            citc = _pearson(
                matrix[item_id].astype(float).tolist(),
                (total - matrix[item_id]).astype(float).tolist(),
            )
            selections = subset[subset["item_id"] == item_id][
                "selected_option_id"
            ].astype(str)
            row: dict[str, Any] = {
                "condition": condition,
                "facet_key": item_map[item_id]["facet_key"],
                "target_dimension_id": dimension_id,
                "item_id": item_id,
                "citc": citc,
                "mean_score": float(matrix[item_id].mean()),
                "score_sd": float(matrix[item_id].std(ddof=1)),
                "n": len(subject_ids),
            }
            for option_id in ("A", "B", "C", "D"):
                count = int((selections == option_id).sum())
                row[f"option_{option_id}_n"] = count
                row[f"option_{option_id}_rate"] = count / len(subject_ids)
            item_rows.append(row)
            if citc is not None:
                citcs.append(float(citc))
        correlations = {
            domain: _spearman(total.tolist(), neo_scores[domain].tolist())
            for domain in NEO_DOMAINS
        }
        target_rho = correlations[target_domain]
        non_target_values = [
            abs(float(value))
            for domain, value in correlations.items()
            if domain != target_domain and value is not None
        ]
        max_non_target = max(non_target_values) if non_target_values else None
        gap = (
            abs(float(target_rho)) - max_non_target
            if target_rho is not None and max_non_target is not None
            else None
        )
        facet_key = item_map[dimension_items[0]]["facet_key"]
        metric_rows.append({
            "condition": condition,
            "facet_key": facet_key,
            "target_dimension_id": dimension_id,
            "target_neo_domain": target_domain,
            "respondent_count": len(subject_ids),
            "item_count": len(dimension_items),
            "cronbach_alpha": _alpha(matrix),
            "citc_mean": float(np.mean(citcs)) if citcs else None,
            "citc_min": float(np.min(citcs)) if citcs else None,
            "facet_score_mean": float(facet_scores[dimension_id].mean()),
            "facet_score_sd": float(facet_scores[dimension_id].std(ddof=1)),
            "target_neo_spearman": target_rho,
            "max_abs_non_target_neo_spearman": max_non_target,
            "discriminant_gap": gap,
            **{f"rho_neo_{domain}": value for domain, value in correlations.items()},
            "formula_version": FORMULA_VERSION,
        })
    long_frame = long_frame.copy()
    long_frame.insert(0, "condition", condition)
    facet_scores = facet_scores.copy()
    facet_scores.insert(0, "condition", condition)
    facet_scores.index.name = "respondent_id"
    return long_frame, facet_scores, metric_rows, item_rows


def _comparison_rows(
    old_scores: pd.DataFrame,
    new_scores: pd.DataFrame,
    metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metric_map = {
        (str(row["condition"]), str(row["target_dimension_id"])): row
        for row in metrics
    }
    rows: list[dict[str, Any]] = []
    for dimension_id in DIMENSION_TO_NEO:
        old = metric_map[("legacy_historical", dimension_id)]
        new = metric_map[("embodied_probability", dimension_id)]
        old_values = old_scores[dimension_id].astype(float)
        new_values = new_scores[dimension_id].astype(float)
        rows.append({
            "facet_key": old["facet_key"],
            "target_dimension_id": dimension_id,
            "old_new_score_spearman": _spearman(
                old_values.tolist(), new_values.tolist()
            ),
            "old_new_score_pearson": _pearson(
                old_values.tolist(), new_values.tolist()
            ),
            "mean_absolute_score_difference": float(
                np.mean(np.abs(old_values.to_numpy() - new_values.to_numpy()))
            ),
            "delta_alpha_new_minus_old": (
                float(new["cronbach_alpha"]) - float(old["cronbach_alpha"])
                if new.get("cronbach_alpha") is not None
                and old.get("cronbach_alpha") is not None
                else None
            ),
            "delta_target_rho_new_minus_old": (
                float(new["target_neo_spearman"])
                - float(old["target_neo_spearman"])
                if new.get("target_neo_spearman") is not None
                and old.get("target_neo_spearman") is not None
                else None
            ),
            "delta_discriminant_gap_new_minus_old": (
                float(new["discriminant_gap"])
                - float(old["discriminant_gap"])
                if new.get("discriminant_gap") is not None
                and old.get("discriminant_gap") is not None
                else None
            ),
        })
    return rows


def _render_report(
    root: Path,
    *,
    model_id: str,
    subject_count: int,
    metrics: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    probability_summary: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
) -> Path:
    def f(value: Any) -> str:
        if value is None or not isinstance(value, (int, float)):
            return "不可估计"
        number = float(value)
        return f"{number:.3f}" if math.isfinite(number) else "不可估计"

    metric_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']))}</td>"
        f"<td>{escape(str(row['facet_key']))}</td>"
        f"<td>{f(row.get('cronbach_alpha'))}</td>"
        f"<td>{f(row.get('citc_mean'))}</td>"
        f"<td>{f(row.get('facet_score_mean'))}</td>"
        f"<td>{f(row.get('facet_score_sd'))}</td>"
        f"<td>{f(row.get('target_neo_spearman'))}</td>"
        f"<td>{f(row.get('max_abs_non_target_neo_spearman'))}</td>"
        f"<td>{f(row.get('discriminant_gap'))}</td>"
        "</tr>"
        for row in metrics
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['facet_key']))}</td>"
        f"<td>{f(row.get('old_new_score_spearman'))}</td>"
        f"<td>{f(row.get('mean_absolute_score_difference'))}</td>"
        f"<td>{f(row.get('delta_alpha_new_minus_old'))}</td>"
        f"<td>{f(row.get('delta_target_rho_new_minus_old'))}</td>"
        f"<td>{f(row.get('delta_discriminant_gap_new_minus_old'))}</td>"
        "</tr>"
        for row in comparisons
    )
    probability = probability_summary or {}
    html = f"""<!doctype html>
<html lang='zh-CN'><meta charset='utf-8'><title>Mussel新旧虚拟被试方法比较</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1250px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f1f4f8}}code{{background:#f3f3f3;padding:2px 4px}}.note{{background:#fff8df;padding:12px}}</style>
<h1>Mussel新旧虚拟被试方法比较</h1>
<p>同一批被试={subject_count}；Mussel=5个facet×22题；统一计分=A/B记1、C/D记0；新条件模型=<code>{escape(model_id)}</code>。</p>
<div class='note'>旧条件复用历史Mussel作答，新条件使用行为总结、主观行动后果和概率抽样。旧作答来自历史运行环境，因此这是系统级比较，不是只改变提示词的纯净消融实验。NEO-FFI只提供上位domain相关，不是严格的facet汇聚效度；严格同粒度结果请运行mussel-ipip。</div>
<h2>五个facet的测量结果</h2>
<table><thead><tr><th>条件</th><th>facet</th><th>α</th><th>CITC均值</th><th>均分</th><th>SD</th><th>NEO上位domain相关（辅助）</th><th>最大非目标domain相关</th><th>domain区分差值</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h2>新方法相对旧方法</h2>
<table><thead><tr><th>facet</th><th>两方法得分相关</th><th>平均绝对得分差</th><th>Δα</th><th>Δ目标相关</th><th>Δ区分差值</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<h2>新方法概率诊断</h2>
<p>平均最大选择概率={f(probability.get('mean_max_probability'))}；标准化概率熵={f(probability.get('mean_normalized_entropy'))}；近确定性比例={f(probability.get('near_deterministic_rate'))}。这些只描述选择不确定性，不是信效度。</p>
<h2>运行审计</h2>
<p>新模型调用={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；Token={telemetry.get('total_tokens')}。旧条件没有重新调用模型。</p>
<p>本实验是计算机内部比较，不能单凭数值断言新方法“更像真人”。</p>
</html>"""
    report = root / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(html, encoding="utf-8")
    return report


async def run_mussel_method_comparison(
    config: MusselComparisonConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    paths = _resolve_inputs(config)
    subject_ids = _load_subject_ids(paths["old_abc"], paths["new_abc"])
    pool, profiles = load_legacy_pool(paths["legacy_project"])
    summaries, summary_path, summary_manifest = load_comparison_summaries(
        paths["legacy_project"], paths["summary_pool"]
    )
    missing_profiles = [rid for rid in subject_ids if rid not in profiles]
    missing_summaries = [rid for rid in subject_ids if rid not in summaries]
    if missing_profiles or missing_summaries:
        raise ValueError(
            f"冻结被试输入不完整：profile={missing_profiles[:3]}；"
            f"summary={missing_summaries[:3]}"
        )
    items, mussel_metadata = load_mussel_items(paths["mussel"])
    if len(items) != 110:
        raise ValueError(f"Mussel必须为110题，实际为{len(items)}")
    old_records, old_response_hashes = load_historical_mussel_records(
        paths["legacy_project"], subject_ids, items
    )
    old_neo_path = paths["old_abc"] / "neo_ffi" / "scores.csv"
    new_neo_path = paths["new_abc"] / "neo_ffi" / "scores.csv"
    old_neo = _load_neo_scores(old_neo_path, subject_ids)
    new_neo = _load_neo_scores(new_neo_path, subject_ids)

    model = get_model(config.model_id)
    runnable, structured_output_method = with_compatible_structured_output(
        model, ChoiceProbabilityOutput
    )
    model_id = _model_id(model, config.model_id)
    run_root = (
        config.output.resolve()
        if config.output
        else paths["experiment"] / "mussel_legacy_vs_embodied_100"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    run_id = f"mussel-comparison-{run_root.name}"
    binding = fingerprint({
        "subject_ids": subject_ids,
        "pool_id": pool.get("pool_id"),
        "summary_sha256": _sha256_file(summary_path),
        "mussel_sha256": _sha256_file(paths["mussel"]),
        "old_response_hashes": old_response_hashes,
        "old_neo_sha256": _sha256_file(old_neo_path),
        "new_neo_sha256": _sha256_file(new_neo_path),
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "sampling_seed": config.sampling_seed,
    })
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("binding") != binding:
            raise ValueError("指定Mussel比较目录已绑定到不同输入，拒绝混合")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "experiment": str(paths["experiment"]),
        "respondent_count": len(subject_ids),
        "respondent_ids": subject_ids,
        "conditions": list(CONDITIONS),
        "legacy_condition_source": "frozen historical Mussel responses",
        "embodied_condition_model": model_id,
        "summary_source": str(summary_path),
        "summary_condition_id": (
            summary_manifest.get("condition_id") if summary_manifest else None
        ),
        "mussel_source": str(paths["mussel"]),
        "mussel_metadata": mussel_metadata,
        "old_neo_source": str(old_neo_path),
        "new_neo_source": str(new_neo_path),
        "prompt_version": PROMPT_VERSION,
        "persona_prompt_version": PERSONA_PROMPT_VERSIONS[
            "summary_embodied_probability"
        ],
        "sampling_seed": config.sampling_seed,
        "structured_output_method": structured_output_method,
        "system_level_comparison_warning": (
            "Historical Mussel responses and embodied responses were produced "
            "under different runtime/model conditions."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(run_root / "config.json", _json_value(asdict(config)))
    write_json(run_root / "participants.json", {
        "respondent_ids": subject_ids,
        "count": len(subject_ids),
        "same_people_across_conditions": True,
    })
    write_json(run_root / "mussel_definition.json", {
        "source": str(paths["mussel"]),
        "sha256": _sha256_file(paths["mussel"]),
        "metadata": mussel_metadata,
        "items": items,
    })
    _write_jsonl(run_root / "legacy_historical" / "responses.jsonl", old_records)

    embodied_config = LegacyABCConfig(
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
    try:
        with output_scope(
            run_root / "runtime", telemetry=run_root / "telemetry"
        ), run_context(run_id):
            new_records = await _run_sjt_stage(
                "MUSSEL_NEW",
                {"items": items},
                subject_ids,
                profiles,
                summaries,
                runnable,
                model_id,
                run_root / "embodied_probability",
                embodied_config,
            )
    except Exception as exc:
        write_json(manifest_path, {**manifest, "status": "failed", "error": str(exc)})
        raise

    condition_inputs = {
        "legacy_historical": (old_records, old_neo),
        "embodied_probability": (new_records, new_neo),
    }
    all_metrics: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []
    score_frames: dict[str, pd.DataFrame] = {}
    for condition, (records, neo) in condition_inputs.items():
        long_frame, facet_scores, metrics, item_rows = _condition_metrics(
            condition=condition,
            records=records,
            items=items,
            subject_ids=subject_ids,
            neo_scores=neo,
        )
        condition_dir = run_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        long_frame.to_csv(
            condition_dir / "scored_responses.csv", index=False, encoding="utf-8-sig"
        )
        facet_scores.reset_index().to_csv(
            condition_dir / "facet_scores.csv", index=False, encoding="utf-8-sig"
        )
        write_csv(condition_dir / "facet_metrics.csv", metrics)
        write_json(condition_dir / "facet_metrics.json", _json_value(metrics))
        write_csv(condition_dir / "item_metrics.csv", item_rows)
        neo.reset_index().to_csv(
            condition_dir / "neo_ffi_scores.csv", index=False, encoding="utf-8-sig"
        )
        all_metrics.extend(metrics)
        all_items.extend(item_rows)
        score_frames[condition] = facet_scores.drop(columns=["condition"])

    probability_summary, probability_rows = probability_diagnostics(new_records)
    if probability_rows:
        write_csv(
            run_root / "embodied_probability" / "probability_diagnostics_by_item.csv",
            probability_rows,
        )
    write_json(
        run_root / "embodied_probability" / "probability_diagnostics.json",
        _json_value(probability_summary),
    )
    comparisons = _comparison_rows(
        score_frames["legacy_historical"],
        score_frames["embodied_probability"],
        all_metrics,
    )
    write_csv(run_root / "summary" / "facet_metrics.csv", all_metrics)
    write_csv(run_root / "summary" / "method_comparison.csv", comparisons)
    write_csv(run_root / "summary" / "item_metrics.csv", all_items)
    telemetry = _telemetry_summary(run_root, run_id)
    cost = {
        "model_id": model_id,
        "historical_calls_reused": len(old_records),
        "historical_new_model_calls": 0,
        "embodied_expected_calls": len(subject_ids) * len(items),
        "telemetry": telemetry,
        "fee": None,
        "fee_note": "未提供当前模型的完整适用单价，费用保持未知。",
    }
    write_json(run_root / "cost.json", _json_value(cost))
    result = {
        "status": "complete",
        "verification_status": "ANALYZED",
        "respondent_count": len(subject_ids),
        "item_count": len(items),
        "conditions": list(CONDITIONS),
        "facet_metrics": all_metrics,
        "comparisons": comparisons,
        "probability_diagnostics": probability_summary,
        "cost": cost,
        "interpretation_boundary": (
            "This system-level virtual comparison does not establish human likeness."
        ),
    }
    write_json(run_root / "summary.json", _json_value(result))
    report = _render_report(
        run_root,
        model_id=model_id,
        subject_count=len(subject_ids),
        metrics=all_metrics,
        comparisons=comparisons,
        probability_summary=probability_summary,
        telemetry=telemetry,
    )
    write_json(
        manifest_path,
        {
            **manifest,
            "status": "complete",
            "report": str(report),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return run_root, {**result, "report": str(report)}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="比较新旧虚拟被试的Mussel作答")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--legacy-project", type=Path, default=DEFAULT_LEGACY_PROJECT)
    parser.add_argument("--summary-pool", type=Path, default=DEFAULT_IMPORTED_SUMMARY_POOL)
    parser.add_argument("--old-abc-result", type=Path, default=None)
    parser.add_argument("--new-abc-result", type=Path, default=None)
    parser.add_argument("--mussel", dest="mussel_path", type=Path, default=None)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--max-concurrency", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    root, result = asyncio.run(run_mussel_method_comparison(MusselComparisonConfig(
        experiment=args.experiment,
        legacy_project=args.legacy_project,
        summary_pool=args.summary_pool,
        old_abc_result=args.old_abc_result,
        new_abc_result=args.new_abc_result,
        mussel_path=args.mussel_path,
        model_id=args.model_id,
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
        sampling_seed=args.sampling_seed,
        output=args.output,
    )))
    print(f"Mussel比较输出：{root}", flush=True)
    print(f"报告：{result['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
