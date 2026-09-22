"""Batch audit for visible compliance with the first-person embodied prompt."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
import numpy as np
import pandas as pd

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
)
from sjt_system.agent.json_parsing import parse_model_json_response
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .embodied_prompt_audit import (
    DEFAULT_SUMMARY_POOL,
    EmbodiedPromptAuditOutput,
    _audit_messages,
    _validate_output,
)
from .legacy_pool_abc import (
    OPTION_IDS,
    _spearman,
    build_legacy_persona_prompt,
)
from .source_e2_recovery import score_source_e2
from .storage import write_csv, write_json


DEFAULT_SOURCE_POOL = DEFAULT_SUMMARY_POOL / "source_virtual_respondents.json"
PROMPT_VERSION = "embodied-first-person-visible-audit-v2"


@dataclass(frozen=True)
class EmbodiedBatchAuditConfig:
    experiment: Path
    respondents: int = 30
    items: int = 4
    max_concurrency: int = 10
    model_id: str = "glm-5.3-flash"
    summary_pool: Path = DEFAULT_SUMMARY_POOL
    source_pool: Path = DEFAULT_SOURCE_POOL
    seed: int = 20260915
    output: Path | None = None

    def validate(self) -> None:
        if isinstance(self.respondents, bool) or self.respondents < 3:
            raise ValueError("respondents至少为3")
        if isinstance(self.items, bool) or self.items < 1:
            raise ValueError("items至少为1")
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not self.model_id.strip():
            raise ValueError("model_id不能为空")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed必须是整数")


def _read_summaries(path: Path) -> dict[str, str]:
    summaries: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            respondent_id = str(value.get("respondent_id") or "")
            summary = str(value.get("summary") or "").strip()
            if not respondent_id or not summary:
                raise ValueError(f"画像文件第{line_number}行不完整")
            if respondent_id in summaries:
                raise ValueError(f"画像文件被试重复：{respondent_id}")
            summaries[respondent_id] = summary
    return summaries


def select_stratified_participants(
    e2_scores: pd.DataFrame,
    summaries: Mapping[str, str],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    available = e2_scores.loc[e2_scores.index.intersection(list(summaries))].copy()
    if count > len(available):
        raise ValueError(f"要求{count}名被试，但只有{len(available)}名可完整对齐")
    ordered = available.sort_values(["E2_percent_0_100"], kind="stable")
    groups = np.array_split(ordered.index.to_numpy(), 3)
    labels = ("low", "mid", "high")
    base, remainder = divmod(count, 3)
    requested = {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for label, group_ids in zip(labels, groups):
        ids = list(map(str, group_ids))
        take = requested[label]
        if take > len(ids):
            raise ValueError(f"{label}分层不足：需要{take}，实际{len(ids)}")
        chosen = rng.sample(ids, take)
        chosen.sort()
        for respondent_id in chosen:
            rows.append({
                "respondent_id": respondent_id,
                "e2_stratum": label,
                "e2_percent_0_100": float(
                    available.loc[respondent_id, "E2_percent_0_100"]
                ),
                "summary": summaries[respondent_id],
            })
    return rows


def select_evenly_spaced_items(form: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    items = form.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("A问卷缺少题目")
    if count > len(items):
        raise ValueError(f"要求{count}道题，但A问卷只有{len(items)}道")
    if count == 1:
        indices = [0]
    else:
        indices = [round(value) for value in np.linspace(0, len(items) - 1, count)]
    if len(set(indices)) != count:
        raise ValueError("题目等距选择产生重复索引")
    return [dict(items[index]) for index in indices]


def _top_match(assessments: Sequence[Mapping[str, Any]], probabilities: Mapping[str, float]) -> bool:
    willingness = {
        str(row["option_id"]): float(row["willingness"]) for row in assessments
    }
    max_willingness = max(willingness.values())
    max_probability = max(float(value) for value in probabilities.values())
    willing_options = {
        option_id for option_id, value in willingness.items() if value == max_willingness
    }
    probability_options = {
        option_id
        for option_id, value in probabilities.items()
        if float(value) == max_probability
    }
    return bool(willing_options.intersection(probability_options))


def response_metric_rows(success_records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    for record in success_records:
        assessments = record["option_assessments"]
        probabilities = record["choice_probabilities"]
        willingness = [float(row["willingness"]) for row in assessments]
        benefits = [float(row["subjective_benefit"]) for row in assessments]
        burdens = [
            float(np.mean([
                row["time_energy_cost"],
                row["emotional_discomfort"],
                row["relationship_reputation_risk"],
            ]))
            for row in assessments
        ]
        probability_values = [
            float(probabilities[str(row["option_id"])]) for row in assessments
        ]
        compliant_count = sum(bool(row["first_person_compliant"]) for row in assessments)
        response_rows.append({
            "respondent_id": record["respondent_id"],
            "e2_stratum": record["e2_stratum"],
            "item_id": record["item_id"],
            "all_options_first_person_compliant": compliant_count == 4,
            "first_person_compliant_count": compliant_count,
            "first_person_compliant_rate": compliant_count / 4,
            "forbidden_term_option_count": sum(
                bool(row["forbidden_observer_terms"]) for row in assessments
            ),
            "top_willingness_matches_top_probability": _top_match(
                assessments, probabilities
            ),
            "willingness_probability_spearman": _spearman(
                willingness, probability_values
            ),
            "benefit_probability_spearman": _spearman(
                benefits, probability_values
            ),
            "burden_probability_spearman": _spearman(
                burdens, probability_values
            ),
            "duration_seconds": record.get("duration_seconds"),
        })
        for assessment in assessments:
            option_id = str(assessment["option_id"])
            option_rows.append({
                "respondent_id": record["respondent_id"],
                "e2_stratum": record["e2_stratum"],
                "item_id": record["item_id"],
                "option_id": option_id,
                "first_person_reaction": assessment["first_person_reaction"],
                "first_person_compliant": assessment["first_person_compliant"],
                "uses_explicit_first_person": assessment["uses_explicit_first_person"],
                "forbidden_observer_terms": "|".join(
                    assessment["forbidden_observer_terms"]
                ),
                "time_energy_cost": assessment["time_energy_cost"],
                "emotional_discomfort": assessment["emotional_discomfort"],
                "relationship_reputation_risk": assessment[
                    "relationship_reputation_risk"
                ],
                "subjective_benefit": assessment["subjective_benefit"],
                "willingness": assessment["willingness"],
                "choice_probability": probabilities[option_id],
            })
    return response_rows, option_rows


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(numeric)) if numeric else None


def summarize_batch(
    total_jobs: int,
    success_records: Sequence[Mapping[str, Any]],
    failure_records: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    option_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    success_count = len(success_records)
    return {
        "status": "complete",
        "total_jobs": total_jobs,
        "successful_outputs": success_count,
        "failed_outputs": len(failure_records),
        "structured_output_success_rate": success_count / total_jobs if total_jobs else None,
        "whole_response_first_person_compliance_rate": _mean([
            row["all_options_first_person_compliant"] for row in response_rows
        ]),
        "option_first_person_compliance_rate": _mean([
            row["first_person_compliant"] for row in option_rows
        ]),
        "forbidden_observer_term_option_rate": _mean([
            bool(row["forbidden_observer_terms"]) for row in option_rows
        ]),
        "top_willingness_probability_match_rate": _mean([
            row["top_willingness_matches_top_probability"] for row in response_rows
        ]),
        "mean_willingness_probability_spearman": _mean([
            row["willingness_probability_spearman"] for row in response_rows
        ]),
        "mean_benefit_probability_spearman": _mean([
            row["benefit_probability_spearman"] for row in response_rows
        ]),
        "mean_burden_probability_spearman": _mean([
            row["burden_probability_spearman"] for row in response_rows
        ]),
        "interpretation_boundary": (
            "Visible instruction-following audit only. Noncompliant and malformed "
            "outputs are counted without semantic regeneration. This does not prove "
            "human likeness or superiority over a control prompt."
        ),
    }


def _group_summary(response_rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(response_rows)
    if frame.empty:
        return []
    output: list[dict[str, Any]] = []
    for group, local in frame.groupby(field, sort=True):
        output.append({
            field: group,
            "n": int(local.shape[0]),
            "whole_response_first_person_compliance_rate": float(
                local["all_options_first_person_compliant"].astype(float).mean()
            ),
            "option_first_person_compliance_rate": float(
                local["first_person_compliant_rate"].mean()
            ),
            "top_willingness_probability_match_rate": float(
                local["top_willingness_matches_top_probability"].astype(float).mean()
            ),
            "mean_willingness_probability_spearman": float(
                pd.to_numeric(
                    local["willingness_probability_spearman"], errors="coerce"
                ).mean()
            ),
        })
    return output


def _fmt(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.3f}"


def _render_report(output: Path, config: EmbodiedBatchAuditConfig, summary: Mapping[str, Any], by_stratum: Sequence[Mapping[str, Any]], by_item: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> Path:
    key_rows = "".join(
        f"<tr><th>{escape(label)}</th><td>{_fmt(summary.get(key))}</td></tr>"
        for key, label in (
            ("structured_output_success_rate", "结构化输出成功率"),
            ("whole_response_first_person_compliance_rate", "整题第一人称合规率"),
            ("option_first_person_compliance_rate", "选项第一人称合规率"),
            ("forbidden_observer_term_option_rate", "观察者术语选项率"),
            ("top_willingness_probability_match_rate", "最高愿意度—最高概率一致率"),
            ("mean_willingness_probability_spearman", "愿意度—概率平均Spearman"),
            ("mean_benefit_probability_spearman", "收益—概率平均Spearman"),
            ("mean_burden_probability_spearman", "负担—概率平均Spearman"),
        )
    )
    def grouped_rows(rows: Sequence[Mapping[str, Any]], key: str) -> str:
        return "".join(
            "<tr>"
            f"<td>{escape(str(row[key]))}</td><td>{row['n']}</td>"
            f"<td>{_fmt(row['whole_response_first_person_compliance_rate'])}</td>"
            f"<td>{_fmt(row['option_first_person_compliance_rate'])}</td>"
            f"<td>{_fmt(row['top_willingness_probability_match_rate'])}</td>"
            f"<td>{_fmt(row['mean_willingness_probability_spearman'])}</td>"
            "</tr>"
            for row in rows
        )
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第一人称具身提示词批量审计</title>
<style>body{{font:15px/1.65 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #cbd5e1;padding:7px;text-align:left}}th{{background:#f1f5f9}}.note{{background:#fff7d7;border-left:4px solid #d5a400;padding:12px}}</style>
<h1>第一人称具身提示词批量遵循审计</h1>
<p>模型：<code>{escape(config.model_id)}</code>；被试={config.respondents}；题目={config.items}；计划调用={summary['total_jobs']}；成功={summary['successful_outputs']}；失败={summary['failed_outputs']}。</p>
<div class="note">本报告只检查可见输出是否遵循第一人称具身指令。语义不合规不会触发重试；失败也计入分母。结果不能证明模型更像真人，也不能证明相对普通提示词有增量价值。</div>
<h2>总体结果</h2><table>{key_rows}</table>
<h2>按E2分层</h2><table><thead><tr><th>分层</th><th>N</th><th>整题合规率</th><th>选项合规率</th><th>最高值一致率</th><th>愿意度—概率ρ</th></tr></thead><tbody>{grouped_rows(by_stratum, 'e2_stratum')}</tbody></table>
<h2>按题目</h2><table><thead><tr><th>题目</th><th>N</th><th>整题合规率</th><th>选项合规率</th><th>最高值一致率</th><th>愿意度—概率ρ</th></tr></thead><tbody>{grouped_rows(by_item, 'item_id')}</tbody></table>
<h2>失败记录</h2><p>{len(failures)} 条；详见 <code>failures.csv</code>。</p>
<h2>数据文件</h2><p><code>responses.jsonl</code>、<code>response_metrics.csv</code>、<code>option_metrics.csv</code>、<code>summary.json</code>。</p></html>"""
    path = output / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


async def run_embodied_batch_audit(config: EmbodiedBatchAuditConfig) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    forms_path = experiment / "legacy_summary_embodied_probability_abc" / "forms.json"
    forms = json.loads(forms_path.read_text(encoding="utf-8-sig"))
    selected_items = select_evenly_spaced_items(forms["A"], config.items)
    summaries = _read_summaries(config.summary_pool.resolve() / "persona_summaries.jsonl")
    e2_scores, _ = score_source_e2(config.source_pool.resolve())
    participants = select_stratified_participants(
        e2_scores, summaries, config.respondents, config.seed
    )
    if config.output is None:
        output = (
            experiment
            / "embodied_prompt_batch_audit"
            / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
    else:
        output = config.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {
        **asdict(config),
        "experiment": str(experiment),
        "summary_pool": str(config.summary_pool.resolve()),
        "source_pool": str(config.source_pool.resolve()),
        "output": str(output),
        "prompt_version": PROMPT_VERSION,
    })
    write_json(output / "participants.json", participants)
    write_json(output / "items.json", selected_items)

    model = get_model(config.model_id)
    runnable = (
        model
        | RunnableLambda(parse_model_json_response)
        | RunnableLambda(EmbodiedPromptAuditOutput.model_validate)
    )
    semaphore = asyncio.Semaphore(config.max_concurrency)
    timeout = get_model_request_timeout_seconds()
    jobs = [
        (participant, item)
        for participant in participants
        for item in selected_items
    ]
    total = len(jobs)
    completed = 0
    progress_lock = asyncio.Lock()

    async def one(participant: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal completed
        start = perf_counter()
        respondent_id = str(participant["respondent_id"])
        item_id = str(item["item_id"])
        try:
            persona = build_legacy_persona_prompt(
                {}, str(participant["summary"]), persona_mode="summary_embodied_probability"
            )
            messages = _audit_messages(persona, dict(item))
            async with semaphore:
                raw = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout)
            validated = _validate_output(raw)
            result = {
                "status": "success",
                "respondent_id": respondent_id,
                "e2_stratum": participant["e2_stratum"],
                "item_id": item_id,
                "duration_seconds": round(perf_counter() - start, 3),
                **validated,
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "respondent_id": respondent_id,
                "e2_stratum": participant["e2_stratum"],
                "item_id": item_id,
                "duration_seconds": round(perf_counter() - start, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        async with progress_lock:
            completed += 1
            if completed == total or completed % max(1, total // 10) == 0:
                print(f"[具身提示词批量审计] {completed}/{total} ({completed / total:.0%})", flush=True)
        return result

    run_id = f"embodied-prompt-batch-audit-{uuid4().hex}"
    with output_scope(output / "runtime", telemetry=output / "telemetry"), run_context(run_id):
        records = await asyncio.gather(*(one(participant, item) for participant, item in jobs))
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] == "failed"]
    with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for record in successes:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output / "failures.csv", failures)
    response_rows, option_rows = response_metric_rows(successes)
    write_csv(output / "response_metrics.csv", response_rows)
    write_csv(output / "option_metrics.csv", option_rows)
    by_stratum = _group_summary(response_rows, "e2_stratum")
    by_item = _group_summary(response_rows, "item_id")
    write_csv(output / "summary_by_stratum.csv", by_stratum)
    write_csv(output / "summary_by_item.csv", by_item)
    summary = summarize_batch(total, successes, failures, response_rows, option_rows)
    summary.update({
        "model_id": config.model_id,
        "respondent_count": len(participants),
        "item_count": len(selected_items),
        "participant_strata": {
            label: sum(row["e2_stratum"] == label for row in participants)
            for label in ("low", "mid", "high")
        },
        "selected_item_ids": [item["item_id"] for item in selected_items],
        "prompt_version": PROMPT_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    write_json(output / "summary.json", summary)
    report = _render_report(output, config, summary, by_stratum, by_item, failures)
    summary["report"] = str(report)
    return output, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一人称具身提示词批量遵循审计")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--respondents", type=int, default=30)
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--model", dest="model_id", default="glm-5.3-flash")
    parser.add_argument("--summary-pool", type=Path, default=DEFAULT_SUMMARY_POOL)
    parser.add_argument("--source-pool", type=Path, default=DEFAULT_SOURCE_POOL)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output, summary = asyncio.run(run_embodied_batch_audit(EmbodiedBatchAuditConfig(**vars(args))))
    print(f"批量审计输出：{output}")
    print(
        f"成功={summary['successful_outputs']}/{summary['total_jobs']}；"
        f"整题第一人称合规率={_fmt(summary['whole_response_first_person_compliance_rate'])}；"
        f"选项合规率={_fmt(summary['option_first_person_compliance_rate'])}"
    )
    print(f"报告：{summary['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
