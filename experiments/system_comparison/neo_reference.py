"""Append-only NEO-FFI evaluation for a completed A/B/C experiment."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from sjt_system.agent.client import get_model
from sjt_system.evaluation.respondents import flatten_matched_condition_groups
from sjt_system.evaluation.simulation import (
    VIRTUAL_RESPONSE_PROMPT_VERSION,
    VirtualResponseRunner,
    _invoke_with_retry,
    _load_jsonl_keys,
    _load_jsonl_records,
    build_neo_ffi_messages,
    build_persona_prompt,
    load_neo_ffi,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .storage import write_csv


NEO_DOMAINS = ("E", "N", "O", "A", "C")


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    x_rank = pd.Series(x, dtype="float64").rank(method="average")
    y_rank = pd.Series(y, dtype="float64").rank(method="average")
    if x_rank.nunique() < 2 or y_rank.nunique() < 2:
        return None
    value = x_rank.corr(y_rank)
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def compute_neo_correlations(
    neo_scores: list[Mapping[str, Any]],
    form_scores: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Return only the requested convergent and discriminant correlations."""

    neo_by_subject = {
        str(row["matched_subject_id"]): row
        for row in neo_scores
        if row.get("matched_subject_id")
    }
    rows: list[dict[str, Any]] = []
    for method in ("A", "B", "C"):
        totals = form_scores.get(method, {})
        subject_ids = sorted(set(totals).intersection(neo_by_subject))
        for domain in NEO_DOMAINS:
            usable = [
                subject_id
                for subject_id in subject_ids
                if isinstance(neo_by_subject[subject_id].get(domain), (int, float))
                and not isinstance(neo_by_subject[subject_id].get(domain), bool)
                and math.isfinite(float(neo_by_subject[subject_id][domain]))
                and math.isfinite(float(totals[subject_id]))
            ]
            form_values = [float(totals[subject_id]) for subject_id in usable]
            neo_values = [float(neo_by_subject[subject_id][domain]) for subject_id in usable]
            rho = _spearman(
                form_values,
                neo_values,
            )
            unavailable_reason = None
            if len(usable) < 3:
                unavailable_reason = "有效配对人数少于3"
            elif len(set(neo_values)) < 2:
                unavailable_reason = "NEO-FFI维度得分无变异"
            elif len(set(form_values)) < 2:
                unavailable_reason = "SJT问卷总分无变异"
            elif rho is None:
                unavailable_reason = "相关系数不可估计"
            rows.append(
                {
                    "method": method,
                    "neo_domain": domain,
                    "evidence_type": "convergent" if domain == "E" else "discriminant",
                    "spearman_rho": rho,
                    "n": len(usable),
                    "estimable": rho is not None,
                    "unavailable_reason": unavailable_reason,
                }
            )
    return rows


def _target_spec(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = flatten_matched_condition_groups(sample["config"]["conditions"])
    target = next((dict(row) for row in groups if row.get("condition_id") == "target"), None)
    if target is None:
        raise ValueError("评估样本缺少 target 条件的人格定义")
    return [
        {
            "dimension_id": target["dimension_id"],
            "level": "facet",
            "domain_id": target.get("domain_id"),
            "domain_name_en": target.get("domain_name_en"),
            "domain_name": target.get("domain_name"),
            "facet_name_en": target.get("facet_name_en"),
            "facet_name": target.get("facet_name"),
        }
    ]


def _score_neo_records(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    required = {"matched_subject_id", "dimension_code", "score", "item_id"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("NEO-FFI作答记录不完整")
    counts = frame.groupby(["matched_subject_id", "dimension_code"])["item_id"].nunique()
    if not (counts == 12).all():
        raise ValueError("每名虚拟被试的每个NEO-FFI维度必须恰好包含12题")
    totals = (
        frame.groupby(["matched_subject_id", "dimension_code"])["score"]
        .sum()
        .unstack("dimension_code")
    )
    if set(totals.columns) != set(NEO_DOMAINS):
        raise ValueError("NEO-FFI五个维度的得分没有全部生成")
    totals = totals.reindex(columns=NEO_DOMAINS).reset_index()
    return totals.to_dict("records")


def _form_sources(store) -> dict[str, str]:
    final = store.read("C/final.json", {})
    source = final.get("source_round")
    if not source:
        raise ValueError("C方法尚未冻结最终问卷")
    return {"A": "A/round_01", "B": "B/round_01", "C": str(source)}


def _load_form_totals(store, sources: Mapping[str, str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method, source in sources.items():
        path = store.path(f"{source}/evaluation/scored_responses.csv")
        if not path.exists():
            raise ValueError(f"{source}缺少已冻结的独立评估作答")
        frame = pd.read_csv(path)
        required = {"condition_id", "matched_subject_id", "item_id", "score"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{source}的SJT作答列不完整")
        target = frame.loc[frame["condition_id"] == "target"].copy()
        expected_items = len(store.read(f"{source}/form.json", {}).get("items", []))
        counts = target.groupby("matched_subject_id")["item_id"].nunique()
        if expected_items < 1 or not (counts == expected_items).all():
            raise ValueError(f"{source}的target组SJT作答不完整")
        output[method] = {
            str(subject_id): float(total)
            for subject_id, total in target.groupby("matched_subject_id")["score"].sum().items()
        }
    return output


def _sjt_model_id(store, source: str) -> str | None:
    checkpoint = store.read(f"{source}/evaluation/checkpoint.json", {})
    manifest_path = checkpoint.get("virtual_response_data_ref")
    if not manifest_path:
        return store.config.evaluation_model_id
    path = Path(manifest_path)
    if not path.exists():
        return store.config.evaluation_model_id
    return json.loads(path.read_text(encoding="utf-8")).get("model_id")


async def run_neo_ffi_addon(store, *, base_model=None) -> dict[str, Any]:
    """Run NEO-FFI once for target evaluation personas and reuse frozen SJT scores."""

    stage = "reference/neo_ffi"
    output_dir = store.path(stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    sample = store.read("participants/evaluation.json")
    target_references = [
        dict(row) for row in sample["respondents"] if row.get("condition_id") == "target"
    ]
    if len(target_references) != store.config.sample_size:
        raise ValueError("NEO-FFI追加评估必须使用完整的evaluation target组")
    if len({row["matched_subject_id"] for row in target_references}) != len(target_references):
        raise ValueError("NEO-FFI追加评估的被试ID不唯一")

    dimensions = load_neo_ffi()
    score_specs = _target_spec(sample)
    model = base_model or get_model(os.getenv("MODEL_ID") or None)
    runner = VirtualResponseRunner(
        base_model=model,
        max_concurrency=store.config.max_concurrency,
        max_retries=int(sample["config"].get("max_retries", 2)),
    )
    binding = fingerprint(
        {
            "sample": target_references,
            "neo_ffi": dimensions,
            "model_id": runner.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        }
    )
    previous_manifest = store.read(f"{stage}/manifest.json", {})
    if previous_manifest and previous_manifest.get("binding") != binding:
        raise ValueError("NEO-FFI目录中已有不同被试、模型或题本的作答，拒绝混合")

    existing_keys = _load_jsonl_keys(
        responses_path, ("respondent_id", "dimension_code", "item_id")
    )
    expected_records = len(target_references) * 60
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "binding": binding,
        "sample_role": "evaluation_target",
        "respondent_count": len(target_references),
        "neo_ffi_item_count": 60,
        "expected_response_records": expected_records,
        "completed_response_records": len(existing_keys),
        "model_id": runner.model_id,
        "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        "cross_model_exploration": True,
    }
    store.write(f"{stage}/manifest.json", manifest)

    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for reference in target_references:
        respondent_id = str(reference["respondent_id"])
        for dimension in dimensions:
            dimension_code = str(dimension["dimension_code"])
            keys = {
                (respondent_id, dimension_code, str(item["item_id"]))
                for item in dimension["items"]
            }
            present = keys.intersection(existing_keys)
            if present and present != keys:
                raise ValueError(f"{respondent_id}/{dimension_code}存在不完整NEO-FFI批次")
            if not present:
                jobs.append((reference, dimension))

    total_jobs = len(target_references) * len(dimensions)
    resumed_jobs = total_jobs - len(jobs)
    completed_jobs = resumed_jobs
    progress_step = max(1, total_jobs // 20)
    progress_lock = asyncio.Lock()

    def validate(result: Mapping[str, Any]) -> list[int]:
        ratings = result.get("ratings")
        if (
            not isinstance(ratings, list)
            or len(ratings) != 12
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 5
                for value in ratings
            )
        ):
            raise ValueError("NEO-FFI批次必须返回12个1至5的整数")
        return ratings

    async def run_job(reference: dict[str, Any], dimension: dict[str, Any]) -> None:
        nonlocal completed_jobs
        respondent_id = str(reference["respondent_id"])
        dimension_code = str(dimension["dimension_code"])
        persona_prompt = build_persona_prompt(reference, score_specs=score_specs)
        ratings = await _invoke_with_retry(
            runner.neo_ffi_model,
            build_neo_ffi_messages(persona_prompt, dimension["items"]),
            semaphore=runner.semaphore,
            validator=validate,
            max_retries=runner.max_retries,
            retry_delay_seconds=runner.retry_delay_seconds,
            request_timeout_seconds=runner.request_timeout_seconds,
            job_label=f"NEO-FFI addon {respondent_id}/{dimension_code}",
        )
        records = []
        for item, rating in zip(dimension["items"], ratings):
            direction = str(item["scoring_direction"])
            records.append(
                {
                    "record_type": "neo_ffi_response",
                    "respondent_id": respondent_id,
                    "matched_subject_id": str(reference["matched_subject_id"]),
                    "condition_id": "target",
                    "dimension_code": dimension_code,
                    "item_id": str(item["item_id"]),
                    "raw_response": int(rating),
                    "scoring_direction": direction,
                    "score": int(rating) if direction == "+" else 6 - int(rating),
                    "model_id": runner.model_id,
                    "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
                }
            )
        await runner._append_records(responses_path, records)
        async with progress_lock:
            completed_jobs += 1
            if completed_jobs == total_jobs or completed_jobs % progress_step == 0:
                print(
                    f"[NEO-FFI] {completed_jobs}/{total_jobs} "
                    f"({completed_jobs / total_jobs:.0%})",
                    flush=True,
                )

    if jobs:
        print(
            f"[NEO-FFI] 开始追加作答：待调用={len(jobs)}；已恢复={resumed_jobs}",
            flush=True,
        )
        with store.timer(stage), output_scope(
            store.path(f"{stage}/runtime"), telemetry=store.path(f"{stage}/telemetry")
        ), run_context(store.root.name + "-neo-ffi-addon"):
            results = await asyncio.gather(
                *(run_job(reference, dimension) for reference, dimension in jobs),
                return_exceptions=True,
            )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            manifest.update(
                status="failed",
                completed_response_records=len(
                    _load_jsonl_keys(responses_path, ("respondent_id", "dimension_code", "item_id"))
                ),
                errors=[str(error) for error in errors[:20]],
            )
            store.write(f"{stage}/manifest.json", manifest)
            raise RuntimeError(f"NEO-FFI有{len(errors)}个批次失败；首个错误：{errors[0]}")

    records = _load_jsonl_records(responses_path)
    if len(records) != expected_records:
        raise ValueError(f"NEO-FFI记录应为{expected_records}条，实际为{len(records)}条")
    neo_scores = _score_neo_records(records)
    sources = _form_sources(store)
    form_scores = _load_form_totals(store, sources)
    neo_subjects = {str(row["matched_subject_id"]) for row in neo_scores}
    for method, totals in form_scores.items():
        if set(totals) != neo_subjects:
            raise ValueError(f"{method}与NEO-FFI不是同一批完整虚拟被试")
    correlations = compute_neo_correlations(neo_scores, form_scores)
    sjt_models = {method: _sjt_model_id(store, source) for method, source in sources.items()}
    metrics = {
        "schema_version": 1,
        "status": "complete",
        "scope": ["convergent_spearman", "discriminant_spearman"],
        "sample_role": "evaluation_target",
        "respondent_count": len(neo_scores),
        "form_sources": sources,
        "sjt_model_ids": sjt_models,
        "neo_ffi_model_id": runner.model_id,
        "cross_model_exploration": len(set(filter(None, sjt_models.values()))) != 1
        or next(iter(sjt_models.values()), None) != runner.model_id,
        "correlations": correlations,
        "interpretation": (
            "同一批虚拟人格的跨模型探索性问卷关联；汇聚为SJT总分与NEO-FFI E的"
            "Spearman相关，区分为SJT总分与N/O/A/C的Spearman相关。不是正式真人效度。"
        ),
    }
    write_csv(store.path(f"{stage}/scores.csv"), neo_scores)
    write_csv(store.path(f"{stage}/correlations.csv"), correlations)
    store.write(f"{stage}/metrics.json", metrics)
    manifest.update(status="completed", completed_response_records=len(records))
    store.write(f"{stage}/manifest.json", manifest)
    return metrics
