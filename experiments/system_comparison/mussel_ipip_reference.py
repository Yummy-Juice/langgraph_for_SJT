"""Add a facet-matched IPIP criterion to a completed Mussel comparison.

The existing 110-item Mussel responses are reused.  Only the five selected
IPIP facets are administered: N4, E2, O5, A4, and C5.  NEO-FFI parent-domain
scores remain available as an auxiliary, broader-grained reference.
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
from pydantic import BaseModel, ConfigDict

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.evaluation.simulation import _invoke_with_retry
from sjt_system.knowledge.behavior_evidence import (
    DEFAULT_CORPUS_PATH,
    load_ipip_corpus,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import (
    DEFAULT_IMPORTED_SUMMARY_POOL,
    DEFAULT_LEGACY_PROJECT,
    _alpha,
    _json_value,
    _model_id,
    _sha256_file,
    _spearman,
    _telemetry_summary,
    build_legacy_persona_prompt,
    load_comparison_summaries,
    load_legacy_pool,
)
from .mussel_method_comparison import DIMENSION_TO_NEO
from .storage import write_csv, write_json


FACET_CODES = ("N4", "E2", "O5", "A4", "C5")
DIMENSION_TO_IPIP = {
    "neuroticism_self_consciousness": "N4",
    "extraversion_gregariousness": "E2",
    "openness_ideas": "O5",
    "agreeableness_compliance": "A4",
    "conscientiousness_self_discipline": "C5",
}
CONDITION_PERSONA_MODES = {
    "legacy_historical": "items_plus_summary",
    "embodied_probability": "summary_only",
}
PROMPT_VERSION = "mussel-ipip-facet-black-box-v1"
FORMULA_VERSION = "mussel-ipip-facet-validity-v1"


class IPIPRatingsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratings: list[int]


@dataclass(frozen=True)
class MusselIPIPConfig:
    experiment: Path
    comparison_result: Path | None = None
    legacy_project: Path = DEFAULT_LEGACY_PROJECT
    summary_pool: Path = DEFAULT_IMPORTED_SUMMARY_POOL
    ipip_path: Path = DEFAULT_CORPUS_PATH
    model_id: str | None = None
    max_concurrency: int = 30
    max_retries: int = 2
    timeout_seconds: float | None = None
    output: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            records.append(value)
    return records


def _ordered_items(scale: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use one deterministic mixed order without exposing polarity."""

    items = [dict(item) for item in scale.get("items") or []]
    return sorted(
        items,
        key=lambda item: fingerprint(
            {"prompt_version": PROMPT_VERSION, "item_id": item.get("item_id")}
        ),
    )


def build_ipip_messages(
    persona_prompt: str,
    scale: Mapping[str, Any],
) -> list[tuple[str, str]]:
    items = _ordered_items(scale)
    statements = "\n".join(
        f"{index}. {item['text']}" for index, item in enumerate(items, 1)
    )
    human = (
        "请按照这个人通常的真实情况评价下面每句话。"
        "1=非常不同意，2=比较不同意，3=不确定，4=比较同意，5=非常同意。\n"
        "不要猜测量表名称、人格维度、计分方向或研究者期待；每题独立判断。\n\n"
        f"{statements}\n\n"
        f"必须按原顺序返回恰好{len(items)}个1至5的整数。"
        '只返回JSON对象，例如：{"ratings":[1,2,3,4,5,1,2,3,4,5]}。'
    )
    return [("system", persona_prompt), ("human", human)]


def _validate_ratings(value: Any, expected_count: int) -> list[int]:
    if isinstance(value, Mapping):
        raw = value.get("ratings")
    else:
        dump = getattr(value, "model_dump", None)
        raw = dump().get("ratings") if callable(dump) else None
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise ValueError(f"IPIP批次必须返回{expected_count}个评分")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 5
        for item in raw
    ):
        raise ValueError("IPIP评分必须全部为1至5的整数")
    return [int(item) for item in raw]


def _load_selected_scales(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus = load_ipip_corpus(path)
    by_code = {
        scale.facet_code: scale.model_dump(mode="json") for scale in corpus.scales
    }
    missing = [code for code in FACET_CODES if code not in by_code]
    if missing:
        raise ValueError(f"IPIP题库缺少Mussel对应facet：{missing}")
    selected = [by_code[code] for code in FACET_CODES]
    for scale in selected:
        if len(scale["items"]) != 10:
            raise ValueError(f"IPIP {scale['facet_code']}必须恰好包含10题")
    metadata = {
        "schema_version": corpus.schema_version,
        "source_file": corpus.source_file,
        "source_sha256": corpus.source_sha256,
        "corpus_hash": corpus.corpus_hash,
        "usage_status": corpus.usage_status,
        "selected_facets": FACET_CODES,
        "item_language": "English",
    }
    return selected, metadata


async def _run_condition_ipip(
    *,
    condition: str,
    subject_ids: Sequence[str],
    profiles: Mapping[str, Any],
    summaries: Mapping[str, str],
    scales: Sequence[Mapping[str, Any]],
    runnable: Any,
    model_id: str,
    output_dir: Path,
    config: MusselIPIPConfig,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    existing_records = _load_jsonl(responses_path)
    existing = {
        (
            str(row.get("respondent_id")),
            str(row.get("facet_code")),
            str(row.get("item_id")),
        )
        for row in existing_records
    }
    expected = {
        (rid, str(scale["facet_code"]), str(item["item_id"]))
        for rid in subject_ids
        for scale in scales
        for item in scale["items"]
    }
    if not existing <= expected or len(existing) != len(existing_records):
        raise ValueError(f"{condition}现有IPIP缓存包含重复或非预期记录")

    jobs: list[tuple[str, Mapping[str, Any]]] = []
    for rid in subject_ids:
        for scale in scales:
            batch = {
                (rid, str(scale["facet_code"]), str(item["item_id"]))
                for item in scale["items"]
            }
            present = batch & existing
            if present and present != batch:
                raise ValueError(f"{condition}/{rid}/{scale['facet_code']}存在不完整IPIP缓存")
            if not present:
                jobs.append((rid, scale))

    persona_mode = CONDITION_PERSONA_MODES[condition]
    persona_prompts = {
        rid: build_legacy_persona_prompt(
            profiles[rid], summaries[rid], persona_mode=persona_mode
        )
        for rid in subject_ids
    }
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    semaphore = asyncio.Semaphore(config.max_concurrency)
    lock = asyncio.Lock()
    completed = len(existing) // 10
    total = len(subject_ids) * len(scales)
    started = perf_counter()

    async def one(rid: str, scale: Mapping[str, Any]) -> None:
        nonlocal completed
        ordered = _ordered_items(scale)
        ratings = await _invoke_with_retry(
            runnable,
            build_ipip_messages(persona_prompts[rid], scale),
            semaphore=semaphore,
            validator=lambda value: _validate_ratings(value, len(ordered)),
            max_retries=config.max_retries,
            retry_delay_seconds=1.0,
            request_timeout_seconds=timeout,
            job_label=f"Mussel IPIP {condition} {rid}/{scale['facet_code']}",
        )
        records = []
        for position, (item, raw) in enumerate(zip(ordered, ratings), 1):
            polarity = str(item["polarity"])
            records.append(
                {
                    "record_type": "mussel_ipip_facet_response",
                    "condition": condition,
                    "respondent_id": rid,
                    "facet_code": str(scale["facet_code"]),
                    "item_id": str(item["item_id"]),
                    "prompt_position": position,
                    "raw_response": int(raw),
                    "polarity": polarity,
                    "score": int(raw) if polarity == "positive" else 6 - int(raw),
                    "model_id": model_id,
                    "prompt_version": PROMPT_VERSION,
                    "persona_mode": persona_mode,
                }
            )
        async with lock:
            with responses_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed == total or completed % max(1, total // 20) == 0:
                print(
                    f"[Mussel IPIP] {condition} {completed}/{total} "
                    f"({completed / total:.0%})",
                    flush=True,
                )

    if jobs:
        print(
            f"[Mussel IPIP] {condition}开始：待调用={len(jobs)}；"
            f"已缓存批次={completed}",
            flush=True,
        )
        results = await asyncio.gather(
            *(one(rid, scale) for rid, scale in jobs), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            write_json(
                output_dir / "errors.json",
                {"condition": condition, "errors": [str(error) for error in errors]},
            )
            raise RuntimeError(
                f"{condition} IPIP有{len(errors)}个批次失败；首个错误：{errors[0]}"
            )

    records = _load_jsonl(responses_path)
    actual = {
        (
            str(row.get("respondent_id")),
            str(row.get("facet_code")),
            str(row.get("item_id")),
        )
        for row in records
    }
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"{condition} IPIP作答不完整：应为{len(expected)}条，实际为{len(records)}条"
        )
    write_json(
        output_dir / "run.json",
        {
            "condition": condition,
            "record_count": len(records),
            "batch_count": total,
            "new_batch_calls": len(jobs),
            "seconds": round(perf_counter() - started, 3),
            "model_id": model_id,
        },
    )
    return records


def score_ipip_records(
    records: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    scales: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = pd.DataFrame(records)
    required = {"respondent_id", "facet_code", "item_id", "score"}
    if frame.empty or not required <= set(frame.columns):
        raise ValueError("IPIP作答记录列不完整")
    item_counts = frame.groupby(["respondent_id", "facet_code"])["item_id"].nunique()
    if len(item_counts) != len(subject_ids) * len(scales) or not (item_counts == 10).all():
        raise ValueError("每名被试的每个IPIP facet必须恰好包含10题")
    totals = (
        frame.groupby(["respondent_id", "facet_code"])["score"]
        .sum()
        .unstack("facet_code")
        .reindex(index=list(subject_ids), columns=list(FACET_CODES))
    )
    if totals.isna().any().any():
        raise ValueError("IPIP五个facet得分不能与冻结被试完整对齐")
    totals.index.name = "respondent_id"
    scale_map = {str(scale["facet_code"]): scale for scale in scales}
    reliability: list[dict[str, Any]] = []
    for facet_code in FACET_CODES:
        item_ids = [str(item["item_id"]) for item in scale_map[facet_code]["items"]]
        matrix = (
            frame[frame["facet_code"] == facet_code]
            .pivot(index="respondent_id", columns="item_id", values="score")
            .reindex(index=list(subject_ids), columns=item_ids)
        )
        reliability.append(
            {
                "facet_code": facet_code,
                "item_count": len(item_ids),
                "respondent_count": len(subject_ids),
                "virtual_sample_alpha": _alpha(matrix),
                "source_reported_human_alpha": float(scale_map[facet_code]["alpha"]),
            }
        )
    return totals.astype(float), reliability


def calculate_facet_validity(
    *,
    condition: str,
    sjt_scores: pd.DataFrame,
    ipip_scores: pd.DataFrame,
    neo_scores: pd.DataFrame,
    sjt_alphas: Mapping[str, float | None],
    ipip_alphas: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension_id, target_ipip in DIMENSION_TO_IPIP.items():
        sjt = sjt_scores[dimension_id].astype(float)
        ipip_correlations = {
            facet_code: _spearman(sjt.tolist(), ipip_scores[facet_code].tolist())
            for facet_code in FACET_CODES
        }
        target_rho = ipip_correlations[target_ipip]
        non_target = [
            abs(float(value))
            for facet_code, value in ipip_correlations.items()
            if facet_code != target_ipip and value is not None
        ]
        maximum = max(non_target) if non_target else None
        gap = (
            float(target_rho) - maximum
            if target_rho is not None and maximum is not None
            else None
        )
        parent_domain = DIMENSION_TO_NEO[dimension_id]
        parent_rho = _spearman(sjt.tolist(), neo_scores[parent_domain].tolist())
        rows.append(
            {
                "condition": condition,
                "target_dimension_id": dimension_id,
                "target_ipip_facet": target_ipip,
                "respondent_count": len(sjt),
                "sjt_alpha": sjt_alphas.get(dimension_id),
                "ipip_virtual_alpha": ipip_alphas.get(target_ipip),
                "target_ipip_spearman": target_rho,
                "max_abs_non_target_ipip_spearman": maximum,
                "ipip_discriminant_gap": gap,
                "neo_ffi_parent_domain": parent_domain,
                "neo_ffi_parent_domain_spearman_auxiliary": parent_rho,
                **{
                    f"rho_ipip_{facet_code}": value
                    for facet_code, value in ipip_correlations.items()
                },
                "formula_version": FORMULA_VERSION,
            }
        )
    return rows


def _load_condition_inputs(
    comparison: Path,
    condition: str,
    subject_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | None]]:
    condition_dir = comparison / condition
    sjt = pd.read_csv(condition_dir / "facet_scores.csv", encoding="utf-8-sig")
    neo = pd.read_csv(condition_dir / "neo_ffi_scores.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(condition_dir / "facet_metrics.csv", encoding="utf-8-sig")
    for frame, name in ((sjt, "Mussel"), (neo, "NEO-FFI")):
        if "respondent_id" not in frame:
            raise ValueError(f"{condition}的{name}得分缺少respondent_id")
        frame["respondent_id"] = frame["respondent_id"].astype(str)
        if frame["respondent_id"].duplicated().any():
            raise ValueError(f"{condition}的{name}被试ID重复")
        frame.set_index("respondent_id", inplace=True)
        frame = frame.reindex(list(subject_ids))
    sjt = sjt.reindex(list(subject_ids))
    neo = neo.reindex(list(subject_ids))
    required_sjt = set(DIMENSION_TO_IPIP)
    required_neo = set(DIMENSION_TO_NEO.values())
    if not required_sjt <= set(sjt.columns) or sjt[list(required_sjt)].isna().any().any():
        raise ValueError(f"{condition}的Mussel facet得分不完整")
    if not required_neo <= set(neo.columns) or neo[list(required_neo)].isna().any().any():
        raise ValueError(f"{condition}的NEO-FFI得分不完整")
    alphas = {
        str(row["target_dimension_id"]): (
            float(row["cronbach_alpha"])
            if pd.notna(row.get("cronbach_alpha"))
            else None
        )
        for _, row in metrics.iterrows()
    }
    return sjt, neo, alphas


def _comparison_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["condition"]), str(row["target_dimension_id"])): row
        for row in metrics
    }
    rows: list[dict[str, Any]] = []
    for dimension_id, facet_code in DIMENSION_TO_IPIP.items():
        old = lookup[("legacy_historical", dimension_id)]
        new = lookup[("embodied_probability", dimension_id)]
        rows.append(
            {
                "target_dimension_id": dimension_id,
                "target_ipip_facet": facet_code,
                "delta_target_ipip_rho_new_minus_old": (
                    float(new["target_ipip_spearman"])
                    - float(old["target_ipip_spearman"])
                    if new.get("target_ipip_spearman") is not None
                    and old.get("target_ipip_spearman") is not None
                    else None
                ),
                "delta_ipip_discriminant_gap_new_minus_old": (
                    float(new["ipip_discriminant_gap"])
                    - float(old["ipip_discriminant_gap"])
                    if new.get("ipip_discriminant_gap") is not None
                    and old.get("ipip_discriminant_gap") is not None
                    else None
                ),
            }
        )
    return rows


def _render_report(
    output: Path,
    *,
    model_id: str,
    subject_count: int,
    metrics: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
) -> Path:
    def f(value: Any) -> str:
        if value is None or not isinstance(value, (int, float)):
            return "—"
        number = float(value)
        return f"{number:.3f}" if math.isfinite(number) else "—"

    metric_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']))}</td>"
        f"<td>{escape(str(row['target_ipip_facet']))}</td>"
        f"<td>{f(row.get('sjt_alpha'))}</td>"
        f"<td>{f(row.get('ipip_virtual_alpha'))}</td>"
        f"<td>{f(row.get('target_ipip_spearman'))}</td>"
        f"<td>{f(row.get('max_abs_non_target_ipip_spearman'))}</td>"
        f"<td>{f(row.get('ipip_discriminant_gap'))}</td>"
        f"<td>{escape(str(row['neo_ffi_parent_domain']))}</td>"
        f"<td>{f(row.get('neo_ffi_parent_domain_spearman_auxiliary'))}</td>"
        "</tr>"
        for row in metrics
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['target_ipip_facet']))}</td>"
        f"<td>{f(row.get('delta_target_ipip_rho_new_minus_old'))}</td>"
        f"<td>{f(row.get('delta_ipip_discriminant_gap_new_minus_old'))}</td>"
        "</tr>"
        for row in comparisons
    )
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>Mussel × IPIP facet级效标</title><style>body{{font:15px/1.65 sans-serif;max-width:1250px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f1f4f8}}.note{{background:#fff8df;padding:12px}}code{{background:#f3f3f3;padding:2px 4px}}</style></head><body>
<h1>Mussel × IPIP facet级效标</h1>
<p>同一批被试={subject_count}；效标模型=<code>{escape(model_id)}</code>；Mussel既有作答全部复用，本次只新增IPIP作答。</p>
<div class='note'>主要汇聚效度为Mussel facet与对应IPIP facet的Spearman相关。NEO-FFI只保留为上位domain相关，不再称为严格的facet汇聚效度。仓库中的IPIP题目为英文，正式论文前应确认语言版本和使用权。</div>
<h2>facet级信效度</h2><table><thead><tr><th>条件</th><th>目标IPIP facet</th><th>Mussel α</th><th>IPIP虚拟样本α</th><th>汇聚ρ</th><th>最大非目标|ρ|</th><th>区分差值</th><th>NEO上位域</th><th>上位域ρ（辅助）</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h2>新方法相对旧方法</h2><table><thead><tr><th>facet</th><th>Δ汇聚ρ</th><th>Δ区分差值</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<p>区分差值=对应IPIP facet的带符号相关−其他四个facet最大绝对相关。越大表示目标facet越突出。</p>
<p>新增模型调用={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；Token={telemetry.get('total_tokens')}。</p>
<p>这是虚拟被试内部的计算机实验，不构成真人效度证据。</p></body></html>"""
    report = output / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


async def run_mussel_ipip_reference(
    config: MusselIPIPConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    comparison = (
        config.comparison_result.resolve()
        if config.comparison_result is not None
        else experiment / "mussel_legacy_vs_embodied_100"
    )
    output = (
        config.output.resolve()
        if config.output is not None
        else comparison / "ipip_facet_reference"
    )
    base_manifest_path = comparison / "manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(f"找不到已完成Mussel比较：{base_manifest_path}")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8-sig"))
    if base_manifest.get("status") != "complete":
        raise ValueError("Mussel比较尚未完成，不能追加IPIP facet效标")
    subject_ids = [str(value) for value in base_manifest.get("respondent_ids") or []]
    if len(subject_ids) != 100 or len(set(subject_ids)) != 100:
        raise ValueError("Mussel IPIP效标必须绑定冻结且唯一的100名被试")

    required = [
        comparison / condition / "facet_scores.csv"
        for condition in CONDITION_PERSONA_MODES
    ] + [
        comparison / condition / "facet_metrics.csv"
        for condition in CONDITION_PERSONA_MODES
    ] + [
        comparison / condition / "neo_ffi_scores.csv"
        for condition in CONDITION_PERSONA_MODES
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Mussel IPIP效标缺少既有结果：" + "；".join(missing))

    pool, profiles = load_legacy_pool(config.legacy_project.resolve())
    summaries, summary_path, summary_manifest = load_comparison_summaries(
        config.legacy_project.resolve(), config.summary_pool.resolve()
    )
    if any(rid not in profiles or rid not in summaries for rid in subject_ids):
        raise ValueError("Mussel冻结被试的人格输入不完整")
    scales, ipip_metadata = _load_selected_scales(config.ipip_path.resolve())

    model = get_model(config.model_id)
    runnable, structured_output_method = with_compatible_structured_output(
        model, IPIPRatingsOutput
    )
    model_id = _model_id(model, config.model_id)
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"mussel-ipip-{output.name}"
    binding = fingerprint(
        {
            "base_manifest_binding": base_manifest.get("binding"),
            "subject_ids": subject_ids,
            "pool_id": pool.get("pool_id"),
            "summary_sha256": _sha256_file(summary_path),
            "ipip_corpus_hash": ipip_metadata["corpus_hash"],
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "condition_persona_modes": CONDITION_PERSONA_MODES,
        }
    )
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if previous.get("binding") != binding:
            raise ValueError("现有Mussel IPIP目录绑定到不同输入，拒绝混合作答")
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "binding": binding,
        "run_id": run_id,
        "comparison_result": str(comparison),
        "respondent_ids": subject_ids,
        "respondent_count": len(subject_ids),
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "structured_output_method": structured_output_method,
        "ipip_metadata": ipip_metadata,
        "summary_source": str(summary_path),
        "summary_condition_id": (
            summary_manifest.get("condition_id") if summary_manifest else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(output / "config.json", _json_value(asdict(config)))
    write_json(
        output / "ipip_definition.json",
        {"metadata": ipip_metadata, "scales": scales},
    )

    try:
        records_by_condition: dict[str, list[dict[str, Any]]] = {}
        with output_scope(output / "runtime", telemetry=output / "telemetry"), run_context(run_id):
            for condition in CONDITION_PERSONA_MODES:
                records_by_condition[condition] = await _run_condition_ipip(
                    condition=condition,
                    subject_ids=subject_ids,
                    profiles=profiles,
                    summaries=summaries,
                    scales=scales,
                    runnable=runnable,
                    model_id=model_id,
                    output_dir=output / condition,
                    config=config,
                )
    except Exception as exc:
        write_json(manifest_path, {**manifest, "status": "failed", "error": str(exc)})
        raise

    all_metrics: list[dict[str, Any]] = []
    all_reliability: list[dict[str, Any]] = []
    for condition in CONDITION_PERSONA_MODES:
        ipip_scores, reliability = score_ipip_records(
            records_by_condition[condition], subject_ids, scales
        )
        sjt, neo, sjt_alphas = _load_condition_inputs(
            comparison, condition, subject_ids
        )
        ipip_alpha_map = {
            str(row["facet_code"]): row["virtual_sample_alpha"]
            for row in reliability
        }
        metrics = calculate_facet_validity(
            condition=condition,
            sjt_scores=sjt,
            ipip_scores=ipip_scores,
            neo_scores=neo,
            sjt_alphas=sjt_alphas,
            ipip_alphas=ipip_alpha_map,
        )
        condition_dir = output / condition
        ipip_scores.reset_index().to_csv(
            condition_dir / "ipip_facet_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_csv(condition_dir / "ipip_reliability.csv", reliability)
        write_csv(condition_dir / "facet_validity.csv", metrics)
        all_metrics.extend(metrics)
        all_reliability.extend(
            [{"condition": condition, **row} for row in reliability]
        )

    comparisons = _comparison_rows(all_metrics)
    write_csv(output / "facet_validity.csv", all_metrics)
    write_csv(output / "ipip_reliability.csv", all_reliability)
    write_csv(output / "method_comparison.csv", comparisons)
    telemetry = _telemetry_summary(output, run_id)
    result = {
        "status": "complete",
        "verification_status": "ANALYZED",
        "respondent_count": len(subject_ids),
        "conditions": list(CONDITION_PERSONA_MODES),
        "primary_criterion": "IPIP facet matched to each Mussel facet",
        "auxiliary_criterion": "NEO-FFI parent domain",
        "facet_metrics": all_metrics,
        "comparisons": comparisons,
        "ipip_reliability": all_reliability,
        "telemetry": telemetry,
        "language_warning": "Repository IPIP items are English.",
        "interpretation_boundary": (
            "Virtual-respondent internal facet validity; not human validity evidence."
        ),
    }
    write_json(output / "result.json", _json_value(result))
    report = _render_report(
        output,
        model_id=model_id,
        subject_count=len(subject_ids),
        metrics=all_metrics,
        comparisons=comparisons,
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
    return output, {**result, "report": str(report)}

