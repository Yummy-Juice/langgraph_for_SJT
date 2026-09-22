"""Evaluate frozen A/B/C forms with embodied argmax responses and saved IPIP.

The model is called only for the three 16-item SJT forms.  The five-facet IPIP
criterion is read from a completed, frozen ``fresh-mussel-ipip`` run.  Raw
model probabilities are preserved; all reported SJT scores use the option with
the largest normalized probability (argmax), never a categorical draw.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from sjt_system.agent.client import get_model, with_compatible_structured_output
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import (
    ChoiceProbabilityOutput,
    DEFAULT_IMPORTED_SUMMARY_POOL,
    LegacyABCConfig,
    OPTION_IDS,
    _alpha,
    _json_value,
    _model_id,
    _spearman,
    _telemetry_summary,
    _run_sjt_stage,
    _validate_choice_probabilities,
    add_option_statistics,
    item_metrics,
    load_comparison_summaries,
    load_frozen_forms,
    probability_diagnostics,
    score_sjt_records,
)
from .storage import write_csv, write_json


FACET_CODES = ("N4", "E2", "O5", "A4", "C5")
TARGET_FACET = "E2"
METRIC_VERSION = "embodied-argmax-abc-ipip-v1"
CHOICE_RULE = "argmax_normalized_model_probability"


@dataclass(frozen=True)
class EmbodiedABCIPIPConfig:
    experiment: Path
    ipip_result: Path
    model_id: str = "glm-5.3-flash"
    summary_pool: Path = DEFAULT_IMPORTED_SUMMARY_POOL
    max_concurrency: int = 10
    max_retries: int = 2
    timeout_seconds: float | None = None
    bootstrap_replications: int = 5000
    bootstrap_seed: int = 20260921
    target_facet: str = TARGET_FACET
    output: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")
        if self.bootstrap_replications < 1:
            raise ValueError("bootstrap_replications必须至少为1")
        if self.target_facet not in FACET_CODES:
            raise ValueError(f"target_facet必须属于{FACET_CODES}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点必须是对象：{path}")
    return value


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_value(dict(record)), ensure_ascii=False) + "\n")


def _argmax_choice(
    probabilities: Mapping[str, float],
    *,
    respondent_id: str,
    item_id: str,
) -> tuple[str, int]:
    """Choose the highest probability; break exact ties reproducibly."""

    maximum = max(float(probabilities[option_id]) for option_id in OPTION_IDS)
    tied = [
        option_id
        for option_id in OPTION_IDS
        if math.isclose(float(probabilities[option_id]), maximum, abs_tol=1e-12)
    ]
    if len(tied) == 1:
        return tied[0], 1
    tie_key = fingerprint({
        "choice_rule": CHOICE_RULE,
        "respondent_id": respondent_id,
        "item_id": item_id,
        "tied_options": tied,
    })
    return tied[int(tie_key[:16], 16) % len(tied)], len(tied)


def argmax_records(
    records: Sequence[Mapping[str, Any]],
    form: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = {str(item["item_id"]): item for item in form["items"]}
    converted: list[dict[str, Any]] = []
    for source in records:
        item_id = str(source.get("item_id"))
        respondent_id = str(source.get("respondent_id"))
        if item_id not in items:
            raise ValueError(f"概率作答包含问卷外题目：{item_id}")
        validated = _validate_choice_probabilities({
            "choice_probabilities": source.get("choice_probabilities")
        })
        probabilities = validated["normalized"]
        selected, tie_count = _argmax_choice(
            probabilities,
            respondent_id=respondent_id,
            item_id=item_id,
        )
        converted.append({
            **dict(source),
            "source_sampled_option_id": source.get("selected_option_id"),
            "selected_option_id": selected,
            "score": float(items[item_id]["scoring_key"][selected]),
            "choice_generation": CHOICE_RULE,
            "argmax_probability": float(probabilities[selected]),
            "argmax_tie_count": tie_count,
        })
    return converted


def _hedges_g_extreme_groups(
    form_scores: Sequence[float],
    criterion: Sequence[float],
) -> tuple[float | None, int]:
    form = np.asarray(form_scores, dtype=float)
    grouping = np.asarray(criterion, dtype=float)
    if form.shape != grouping.shape or form.ndim != 1 or form.size < 6:
        return None, 0
    if not np.isfinite(form).all() or not np.isfinite(grouping).all():
        return None, 0
    group_size = form.size // 3
    row_order = np.arange(form.size)
    ascending = np.lexsort((row_order, grouping))
    low = form[ascending[:group_size]]
    high = form[ascending[-group_size:]]
    degrees_of_freedom = low.size + high.size - 2
    if degrees_of_freedom <= 0:
        return None, group_size
    pooled_variance = (
        (low.size - 1) * low.var(ddof=1)
        + (high.size - 1) * high.var(ddof=1)
    ) / degrees_of_freedom
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return None, group_size
    correction = 1.0 - 3.0 / (4.0 * (low.size + high.size) - 9.0)
    return float(correction * (high.mean() - low.mean()) / np.sqrt(pooled_variance)), group_size


def _citc_mean(matrix: np.ndarray) -> float | None:
    values = np.asarray(matrix, dtype=float)
    total = values.sum(axis=1)
    coefficients: list[float] = []
    for column in range(values.shape[1]):
        item = values[:, column]
        rest = total - item
        if np.ptp(item) <= 0 or np.ptp(rest) <= 0:
            continue
        rho = np.corrcoef(item, rest)[0, 1]
        if np.isfinite(rho):
            coefficients.append(float(rho))
    return float(np.mean(coefficients)) if coefficients else None


def _core_metrics(matrix: np.ndarray, ipip: pd.DataFrame) -> dict[str, Any]:
    totals = np.asarray(matrix, dtype=float).sum(axis=1)
    correlations = {
        code: _spearman(totals.tolist(), ipip[code].astype(float).tolist())
        for code in FACET_CODES
    }
    non_target = [
        abs(float(value))
        for code, value in correlations.items()
        if code != TARGET_FACET and value is not None
    ]
    maximum_non_target = max(non_target) if non_target else None
    target = correlations.get(TARGET_FACET)
    target_g, group_size = _hedges_g_extreme_groups(totals, ipip[TARGET_FACET])
    non_target_g = {
        code: _hedges_g_extreme_groups(totals, ipip[code])[0]
        for code in FACET_CODES
        if code != TARGET_FACET
    }
    return {
        "cronbach_alpha": _alpha(pd.DataFrame(matrix)),
        "citc_mean": _citc_mean(matrix),
        "target_spearman": target,
        "max_abs_non_target_spearman": maximum_non_target,
        "discriminant_correlation_gap": (
            float(target) - float(maximum_non_target)
            if target is not None and maximum_non_target is not None
            else None
        ),
        "target_hedges_g": target_g,
        "extreme_group_size_each": group_size,
        "correlations": correlations,
        "non_target_hedges_g": non_target_g,
    }


BOOTSTRAP_METRICS = (
    "cronbach_alpha",
    "citc_mean",
    "target_spearman",
    "max_abs_non_target_spearman",
    "discriminant_correlation_gap",
    "target_hedges_g",
)


def paired_bootstrap(
    matrices: Mapping[str, np.ndarray],
    ipip: pd.DataFrame,
    *,
    replications: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = tuple(matrices)
    respondent_count = len(ipip)
    rng = np.random.default_rng(seed)
    draws = {
        method: {metric: np.full(replications, np.nan) for metric in BOOTSTRAP_METRICS}
        for method in methods
    }
    for replication in range(replications):
        indexes = rng.integers(0, respondent_count, size=respondent_count)
        sampled_ipip = ipip.iloc[indexes].reset_index(drop=True)
        for method in methods:
            metrics = _core_metrics(matrices[method][indexes], sampled_ipip)
            for metric in BOOTSTRAP_METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    draws[method][metric][replication] = float(value)
        if (replication + 1) % 1000 == 0 or replication + 1 == replications:
            print(f"[配对bootstrap] {replication + 1}/{replications}", flush=True)

    def interval(values: np.ndarray) -> tuple[float | None, float | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None, None
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))

    method_rows: list[dict[str, Any]] = []
    point = {method: _core_metrics(matrices[method], ipip) for method in methods}
    for method in methods:
        for metric in BOOTSTRAP_METRICS:
            lower, upper = interval(draws[method][metric])
            method_rows.append({
                "method": method,
                "metric": metric,
                "estimate": point[method].get(metric),
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "valid_replications": int(np.isfinite(draws[method][metric]).sum()),
            })

    difference_rows: list[dict[str, Any]] = []
    for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
        for metric in BOOTSTRAP_METRICS:
            differences = draws[second][metric] - draws[first][metric]
            lower, upper = interval(differences)
            first_value = point[first].get(metric)
            second_value = point[second].get(metric)
            estimate = (
                float(second_value) - float(first_value)
                if isinstance(first_value, (int, float))
                and isinstance(second_value, (int, float))
                else None
            )
            difference_rows.append({
                "contrast": f"{second}-{first}",
                "metric": metric,
                "estimate_difference": estimate,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "interval_excludes_zero": (
                    lower is not None and upper is not None and (lower > 0 or upper < 0)
                ),
            })
    return method_rows, difference_rows


def _render_report(
    output: Path,
    method_metrics: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
    *,
    respondent_count: int,
    model_id: str,
    ipip_result: Path,
) -> Path:
    def number(value: Any) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    method_rows = []
    for row in method_metrics:
        correlations = row["correlations"]
        method_rows.append(
            "<tr>"
            f"<td>{escape(str(row['method']))}</td>"
            f"<td>{number(row['cronbach_alpha'])}</td>"
            f"<td>{number(row['citc_mean'])}</td>"
            f"<td>{number(row['target_spearman'])}</td>"
            f"<td>{number(correlations['N4'])}</td>"
            f"<td>{number(correlations['O5'])}</td>"
            f"<td>{number(correlations['A4'])}</td>"
            f"<td>{number(correlations['C5'])}</td>"
            f"<td>{number(row['max_abs_non_target_spearman'])}</td>"
            f"<td>{number(row['discriminant_correlation_gap'])}</td>"
            f"<td>{number(row['target_hedges_g'])}</td>"
            "</tr>"
        )
    difference_rows = []
    for row in differences:
        difference_rows.append(
            "<tr>"
            f"<td>{escape(str(row['contrast']))}</td>"
            f"<td>{escape(str(row['metric']))}</td>"
            f"<td>{number(row['estimate_difference'])}</td>"
            f"<td>[{number(row['ci_lower_95'])}, {number(row['ci_upper_95'])}]</td>"
            f"<td>{'是' if row['interval_excludes_zero'] else '否'}</td>"
            "</tr>"
        )
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>具身argmax A/B/C × IPIP评估</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1400px;margin:36px auto;padding:0 24px;color:#20242a}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d9dde3;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f3f5f7}}.note{{background:#fff7e6;border-left:4px solid #d98c00;padding:12px;margin:16px 0}}code{{background:#f2f4f7;padding:2px 4px}}</style>
<h1>具身虚拟被试：A/B/C × IPIP信效度</h1>
<p>被试={respondent_count}；SJT模型=<code>{escape(model_id)}</code>；每题按模型四选项概率的最大值直接选项；不计算ICC。</p>
<div class='note'>IPIP复用自已完成的独立黑盒作答：<code>{escape(str(ipip_result))}</code>。本实验比较同一虚拟模拟器下的A/B/C，不证明真人效度。</div>
<h2>点估计</h2>
<table><thead><tr><th>方法</th><th>α</th><th>CITC均值</th><th>目标E2 ρ</th><th>N4 ρ</th><th>O5 ρ</th><th>A4 ρ</th><th>C5 ρ</th><th>最大非目标|ρ|</th><th>相关差值</th><th>E2极端组g</th></tr></thead><tbody>{''.join(method_rows)}</tbody></table>
<h2>配对bootstrap差值</h2>
<table><thead><tr><th>比较</th><th>指标</th><th>差值</th><th>95% CI</th><th>不含0</th></tr></thead><tbody>{''.join(difference_rows)}</tbody></table>
<p>目标汇聚效度为SJT总分与IPIP E2乐群性总分的Spearman相关；非目标相关为N4、O5、A4、C5。极端组效应按E2最低与最高各三分之一计算Hedges’ g。</p>
</html>"""
    report = output / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(html, encoding="utf-8")
    return report


async def run_embodied_abc_ipip(
    config: EmbodiedABCIPIPConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    ipip_result = config.ipip_result.resolve()
    ipip_manifest = _read_json(ipip_result / "manifest.json")
    if ipip_manifest.get("status") != "complete":
        raise ValueError("指定IPIP实验尚未完整完成")
    if ipip_manifest.get("model_id") != config.model_id:
        raise ValueError(
            f"IPIP模型与本次SJT模型不一致：{ipip_manifest.get('model_id')}/{config.model_id}"
        )
    if "embodied_probability" not in (ipip_manifest.get("conditions") or []):
        raise ValueError("指定结果不包含embodied_probability条件")

    ipip_path = ipip_result / "embodied_probability" / "ipip" / "facet_scores.csv"
    ipip = pd.read_csv(ipip_path, encoding="utf-8-sig")
    required = {"respondent_id", *FACET_CODES}
    if not required.issubset(ipip.columns) or ipip["respondent_id"].duplicated().any():
        raise ValueError("IPIP facet_scores结构无效或被试重复")
    subject_ids = [str(value) for value in ipip_manifest.get("respondent_ids") or []]
    if not subject_ids:
        raise ValueError("IPIP manifest缺少respondent_ids")
    ipip["respondent_id"] = ipip["respondent_id"].astype(str)
    ipip = ipip.set_index("respondent_id").reindex(subject_ids)
    if ipip[list(FACET_CODES)].isna().any().any():
        raise ValueError("IPIP与冻结被试ID无法完整对齐")
    ipip = ipip[list(FACET_CODES)].astype(float)

    forms = load_frozen_forms(experiment)
    summaries, summary_path, summary_manifest = load_comparison_summaries(
        experiment,
        config.summary_pool.resolve(),
    )
    missing_summaries = [rid for rid in subject_ids if rid not in summaries]
    if missing_summaries:
        raise ValueError(f"具身persona缺少{len(missing_summaries)}名被试")

    model = get_model(config.model_id)
    runnable, structured_method = with_compatible_structured_output(
        model,
        ChoiceProbabilityOutput,
    )
    model_id = _model_id(model, config.model_id)
    output = (
        config.output.resolve()
        if config.output
        else experiment
        / "embodied_argmax_abc_ipip"
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"embodied-argmax-abc-ipip-{output.name}"
    binding = fingerprint({
        "forms": {method: form["source_sha256"] for method, form in forms.items()},
        "respondent_ids": subject_ids,
        "summary_path": str(summary_path),
        "ipip_path": str(ipip_path),
        "ipip_manifest_run_id": ipip_manifest.get("run_id"),
        "model_id": model_id,
        "choice_rule": CHOICE_RULE,
        "metric_version": METRIC_VERSION,
    })
    previous = _read_json(output / "manifest.json") if (output / "manifest.json").is_file() else None
    if previous and previous.get("binding") != binding:
        raise ValueError("输出目录已经绑定到不同问卷、被试、模型或IPIP结果")

    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "binding": binding,
        "experiment": str(experiment),
        "respondent_count": len(subject_ids),
        "respondent_ids": subject_ids,
        "model_id": model_id,
        "structured_output_method": structured_method,
        "persona_mode": "summary_embodied_probability",
        "choice_rule": CHOICE_RULE,
        "ipip_reused": True,
        "ipip_result": str(ipip_result),
        "ipip_new_model_calls": 0,
        "icc_computed": False,
        "metric_version": METRIC_VERSION,
        "created_at": previous.get("created_at") if previous else datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", _json_value(asdict(config)))
    write_json(output / "forms.json", forms)
    write_json(output / "participants.json", {
        "respondent_ids": subject_ids,
        "count": len(subject_ids),
        "summary_path": str(summary_path),
        "summary_manifest": summary_manifest,
        "ipip_result": str(ipip_result),
    })
    ipip.reset_index().to_csv(output / "reference_ipip_scores.csv", index=False, encoding="utf-8-sig")

    run_config = LegacyABCConfig(
        experiment=experiment,
        model_id=config.model_id,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_seconds=config.timeout_seconds,
        persona_mode="summary_embodied_probability",
        summary_pool=config.summary_pool.resolve(),
        respondent_source="all_285",
        sampling_seed=0,
    )
    profiles = {rid: {} for rid in subject_ids}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    try:
        with output_scope(output / "runtime", telemetry=output / "telemetry"), run_context(run_id):
            # One real job is cached as part of A before the full fan-out.  This
            # catches model/schema outages without spending thousands of calls.
            a_raw = output / "A" / "raw"
            if not (a_raw / "responses.jsonl").is_file():
                preflight = {**forms["A"], "items": [forms["A"]["items"][0]]}
                print("[预检] 具身概率JSON与模型渠道 1/1", flush=True)
                await _run_sjt_stage(
                    "A",
                    preflight,
                    subject_ids[:1],
                    profiles,
                    summaries,
                    runnable,
                    model_id,
                    a_raw,
                    run_config,
                )
                print("[预检] 通过，开始A/B/C正式作答", flush=True)
            for method in ("A", "B", "C"):
                raw_records[method] = await _run_sjt_stage(
                    method,
                    forms[method],
                    subject_ids,
                    profiles,
                    summaries,
                    runnable,
                    model_id,
                    output / method / "raw",
                    run_config,
                )
    except Exception as exc:
        write_json(output / "manifest.json", {**manifest, "status": "failed", "error": str(exc)})
        raise

    matrices: dict[str, np.ndarray] = {}
    method_metrics: list[dict[str, Any]] = []
    for method in ("A", "B", "C"):
        evaluation = output / method / "evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        converted = argmax_records(raw_records[method], forms[method])
        _write_jsonl(evaluation / "argmax_responses.jsonl", converted)
        matrix = score_sjt_records(converted, forms[method], subject_ids)
        matrix.reset_index().to_csv(evaluation / "score_matrix.csv", index=False, encoding="utf-8-sig")
        rows = add_option_statistics(item_metrics(matrix, forms[method]), converted)
        write_csv(evaluation / "item_metrics.csv", rows)
        diagnostics, diagnostic_rows = probability_diagnostics(converted)
        if diagnostic_rows:
            write_csv(evaluation / "probability_diagnostics_by_item.csv", diagnostic_rows)
        metrics = {
            "method": method,
            "respondent_count": len(subject_ids),
            "item_count": len(forms[method]["items"]),
            "choice_rule": CHOICE_RULE,
            **_core_metrics(matrix.to_numpy(dtype=float), ipip.reset_index(drop=True)),
            "probability_diagnostics": diagnostics,
            "metric_version": METRIC_VERSION,
            "icc": None,
            "icc_unavailable_reason": "按预注册方案未进行SJT重测",
        }
        write_json(evaluation / "metrics.json", _json_value(metrics))
        write_json(evaluation / "form.json", forms[method])
        matrices[method] = matrix.to_numpy(dtype=float)
        method_metrics.append(metrics)

    bootstrap_metrics, differences = paired_bootstrap(
        matrices,
        ipip.reset_index(drop=True),
        replications=config.bootstrap_replications,
        seed=config.bootstrap_seed,
    )
    write_csv(output / "summary" / "method_metrics.csv", method_metrics)
    write_csv(output / "summary" / "bootstrap_method_metrics.csv", bootstrap_metrics)
    write_csv(output / "summary" / "bootstrap_method_differences.csv", differences)
    write_json(output / "summary" / "summary.json", _json_value({
        "status": "complete",
        "respondent_count": len(subject_ids),
        "methods": method_metrics,
        "bootstrap_replications": config.bootstrap_replications,
        "bootstrap_seed": config.bootstrap_seed,
        "differences": differences,
    }))
    report = _render_report(
        output,
        method_metrics,
        differences,
        respondent_count=len(subject_ids),
        model_id=model_id,
        ipip_result=ipip_result,
    )
    telemetry = _telemetry_summary(output, run_id)
    cost = {
        "expected_new_sjt_calls": len(subject_ids) * 16 * 3,
        "new_ipip_calls": 0,
        "model_id": model_id,
        "telemetry": telemetry,
    }
    write_json(output / "cost.json", _json_value(cost))
    final_manifest = {
        **manifest,
        "status": "complete",
        "report": str(report),
        "actual_telemetry_calls": telemetry.get("calls"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", final_manifest)
    return output, {"status": "complete", "report": str(report), "metrics": method_metrics, "cost": cost}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="具身argmax虚拟被试复测A/B/C并复用五facet IPIP")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--ipip-result", type=Path, required=True)
    parser.add_argument("--model", dest="model_id", default="glm-5.3-flash")
    parser.add_argument("--summary-pool", type=Path, default=DEFAULT_IMPORTED_SUMMARY_POOL)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    parser.add_argument("--bootstrap-replications", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, result = asyncio.run(run_embodied_abc_ipip(EmbodiedABCIPIPConfig(**vars(args))))
    print(f"\n实验输出：{output}")
    print(f"报告：{result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
