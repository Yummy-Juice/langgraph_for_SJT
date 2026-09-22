"""CIBOL 内部传导自检（第 1 层）— 独立实验。

与 Mussel 实验同框架：对 CIBOL 的 5 个 domain（开放性/神经质/外倾性/
责任心/宜人性）各构造一组 target 虚拟被试（默认每组 100 人，已知
persona 分数 0-100 正态分布），作答全部 CIBOL 90 题（0/1 计分：
C/D=1、A/B=0，因 CIBOL 选项按"A=最低特质行为 → D=最高特质行为"排列）。

验证：
  ① 构念效度：已知 domain 分数 → CIBOL 对应 domain 0-1 均分的 Spearman；
  ② 判别效度：非对应 domain 的相关应明显更弱；
  ③ 计分方向自检：逐题高分箱/低分箱的 C/D 选择率是否同向。

输出到 outputs/cibol_conductance/<session>/：
  cibol_responses.jsonl / score_profiles.json / conductance_report.md /
  conductance_matrix.csv / option_rate_by_score_bin.csv / item_level_warnings.json

用法：
  python tools/run_cibol_conductance_check.py [--sample-size 100]
      [--concurrency 20] [--max-retries 2] [--seed 7]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.authoring.construct_registry import resolve_construct_profile
from sjt_system.evaluation.respondents import generate_score_respondent_refs
from sjt_system.evaluation.simulation import (
    SJTSelectionOutput,
    _invoke_with_retry,
    _structured_result_dict,
    balanced_option_order,
    build_persona_prompt,
    build_sjt_messages,
)
from sjt_system.runtime.telemetry import run_context as telemetry_run_context

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cibol_conductance"
DEFAULT_CIBOL_PATH = PROJECT_ROOT / "docs" / "CIBOL_ZH.json"
CIBOL_SCHEMA_VERSION = "cibol-zh-100-v1"
CIBOL_SCORING_VERSION = "cibol-cd-high-ab-low-binary-v1"
CIBOL_HIGH_OPTION_IDS = frozenset({"C", "D"})
CIBOL_LOW_OPTION_IDS = frozenset({"A", "B"})

# CIBOL 的 5 个中文维度 key → NEO domain id
CIBOL_FACET_KEY_TO_DOMAIN = {
    "开放性": "openness",
    "神经质": "neuroticism",
    "外倾性": "extraversion",
    "责任心": "conscientiousness",
    "宜人性": "agreeableness",
}
CIBOL_DOMAIN_IDS = list(CIBOL_FACET_KEY_TO_DOMAIN.values())

DOMAIN_LABELS = {
    "openness": "开放性",
    "neuroticism": "神经质",
    "extraversion": "外倾性",
    "conscientiousness": "责任心",
    "agreeableness": "宜人性",
}


def load_cibol_items(
    path: str | Path = DEFAULT_CIBOL_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the 90 CIBOL items with an explicit 0/1 key (C/D=1, A/B=0)."""

    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("CIBOL 题库必须是对象")
    if set(raw) != set(CIBOL_FACET_KEY_TO_DOMAIN):
        raise ValueError(
            "CIBOL 题库维度不完整或包含未知维度："
            + ", ".join(str(key) for key in raw)
        )

    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for facet_key, domain_id in CIBOL_FACET_KEY_TO_DOMAIN.items():
        facet_rows = raw.get(facet_key)
        if not isinstance(facet_rows, Mapping) or not facet_rows:
            raise ValueError(f"CIBOL 维度 {facet_key} 缺少题目")
        count = 0
        for raw_item_id, raw_item in facet_rows.items():
            if not isinstance(raw_item, Mapping):
                raise ValueError(f"CIBOL 题目 {facet_key}/{raw_item_id} 结构无效")
            situation = raw_item.get("situation")
            options = raw_item.get("options")
            if not isinstance(situation, str) or not situation.strip():
                raise ValueError(f"CIBOL 题目 {facet_key}/{raw_item_id} 缺少情境")
            if not isinstance(options, Mapping):
                raise ValueError(f"CIBOL 题目 {facet_key}/{raw_item_id} 缺少选项")
            option_ids = [str(option_id) for option_id in options]
            if option_ids != ["A", "B", "C", "D"]:
                raise ValueError(
                    "CIBOL 计分要求每题选项按 A、B、C、D 排列；"
                    f"当前为 {option_ids!r}"
                )
            scoring_key = {
                option_id: (
                    1
                    if option_id in CIBOL_HIGH_OPTION_IDS
                    else 0
                )
                for option_id in option_ids
            }
            item_number = int(str(raw_item_id))
            item_id = f"CIBOL-{domain_id}-{item_number:02d}"
            items.append(
                {
                    "item_id": item_id,
                    "source_item_id": str(raw_item_id),
                    "facet_key": facet_key,
                    "target_dimension_id": domain_id,
                    "scenario": situation,
                    "response_instruction": "请选择你最可能采取的行为。",
                    "response_options": [
                        {"option_id": option_id, "text": str(options[option_id])}
                        for option_id in option_ids
                    ],
                    "scoring_key": scoring_key,
                }
            )
            count += 1
        counts[facet_key] = count

    return items, {
        "schema_version": CIBOL_SCHEMA_VERSION,
        "scoring_version": CIBOL_SCORING_VERSION,
        "item_count": len(items),
        "items_per_facet": counts,
        "high_trait_option_ids": sorted(CIBOL_HIGH_OPTION_IDS),
        "low_trait_option_ids": sorted(CIBOL_LOW_OPTION_IDS),
        "scoring_description": "每题C/D为高特质行为记1，A/B为低特质行为记0。",
        "source_path": str(resolved.resolve()),
    }


def _score_specs_for_domain(domain_id: str) -> list[dict[str, Any]]:
    profile = resolve_construct_profile(domain_id)
    return [
        {
            "dimension_id": domain_id,
            "level": "domain",
            "domain_name_en": profile["domain_name_en"],
            "domain_name": profile["domain_name"],
        }
    ]


def _build_persona_profiles(
    refs: list[dict[str, Any]],
    domain_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "respondent_id": str(ref["respondent_id"]),
            "score_values": dict(ref["score_values"]),
            "active_dimension_id": domain_id,
            "condition_id": "target",
        }
        for ref in refs
    ]


async def _answer_cibol_group(
    *,
    domain_id: str,
    refs: list[dict[str, Any]],
    items: list[dict[str, Any]],
    sjt_model: Any,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    max_retries: int,
    retry_delay_seconds: float,
    request_timeout_seconds: float,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    profiles = _build_persona_profiles(refs, domain_id)
    score_specs = _score_specs_for_domain(domain_id)
    persona_prompts = {
        profile["respondent_id"]: build_persona_prompt(
            profile,
            persona_mode="score_profile",
            score_specs=score_specs,
        )
        for profile in profiles
    }
    existing_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        existing_keys.add(
                            (str(record.get("respondent_id")), str(record.get("item_id")))
                        )

    completed = 0
    errors: list[str] = []

    async def one_job(reference: dict[str, Any], item: dict[str, Any]) -> None:
        nonlocal completed
        respondent_id = str(reference["respondent_id"])
        item_id = str(item["item_id"])
        key = (respondent_id, item_id)
        if key in existing_keys:
            return
        option_ids = [str(option["option_id"]) for option in item["response_options"]]
        ordered_ids = balanced_option_order(
            option_ids,
            respondent_index=int(reference.get("respondent_index") or 0),
            seed=seed,
            item_id=item_id,
        )
        display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
        display_to_original = dict(zip(display_ids, ordered_ids))

        def validate(result: Any) -> str:
            selected = _structured_result_dict(result).get("selected_option_id")
            if selected not in set(display_ids):
                raise ValueError(f"模型选择了无效展示选项 {selected!r}")
            return str(selected)

        try:
            raw_display = await _invoke_with_retry(
                sjt_model,
                build_sjt_messages(
                    persona_prompts[respondent_id],
                    item,
                    display_option_order=ordered_ids,
                ),
                semaphore=semaphore,
                validator=validate,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                request_timeout_seconds=request_timeout_seconds,
                job_label=f"cibol-conductance/{domain_id}/{respondent_id}/{item_id}",
            )
        except Exception as exc:
            errors.append(f"{domain_id}/{respondent_id}/{item_id}: {exc}")
            return
        selected_option_id = display_to_original[raw_display]
        scoring_key = dict(item["scoring_key"])
        score = int(scoring_key[selected_option_id])
        record = {
            "record_type": "cibol_response",
            "run_id": run_id,
            "domain_group": domain_id,
            "respondent_id": respondent_id,
            "condition_id": "target",
            "item_id": item_id,
            "source_item_id": item.get("source_item_id"),
            "facet_key": item.get("facet_key"),
            "target_dimension_id": item.get("target_dimension_id"),
            "display_option_order": [
                {"display_option_id": did, "option_id": oid}
                for did, oid in zip(display_ids, ordered_ids)
            ],
            "raw_display_option_id": raw_display,
            "selected_option_id": selected_option_id,
            "score": score,
            "scoring_version": CIBOL_SCORING_VERSION,
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing_keys.add(key)
        completed += 1

    await asyncio.gather(
        *(one_job(reference, item) for reference in refs for item in items),
        return_exceptions=True,
    )
    return {
        "domain_id": domain_id,
        "respondent_count": len(refs),
        "completed_calls": completed,
        "error_count": len(errors),
        "sample_errors": errors[:20],
    }


# ============================================================================
# 报告
# ============================================================================


def _spearman(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    if len(left) < 3 or len(right) < 3:
        return None, None
    try:
        result = scipy_stats.spearmanr(left, right, nan_policy="omit")
    except Exception:
        return None, None
    rho = float(getattr(result, "statistic", np.nan))
    p = float(getattr(result, "pvalue", np.nan))
    if not np.isfinite(rho):
        return None, None
    return rho, p


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _build_report(
    output_dir: Path,
    *,
    run_id: str,
    domain_ids: list[str],
    items: list[dict[str, Any]],
    profiles_by_group: dict[str, list[dict[str, Any]]],
    responses: list[dict[str, Any]],
) -> None:
    import pandas as pd

    expected_respondent_ids = sorted(
        {str(r.get("respondent_id")) for r in responses}
    )
    known = {}
    for domain_id, refs in profiles_by_group.items():
        for ref in refs:
            values = ref["score_values"]
            dimension_id, value = next(iter(values.items()))
            known[str(ref["respondent_id"])] = {
                "domain_id": domain_id,
                "score": float(value),
            }
    respondents = sorted(known)
    # facet scores per domain from 0/1 records
    rows = []
    for rid in respondents:
        row = {
            "respondent_id": rid,
            "domain_id": known[rid]["domain_id"],
            "known_score": known[rid]["score"],
        }
        for domain_id in domain_ids:
            hits = [
                r.get("score")
                for r in responses
                if r.get("respondent_id") == rid
                and r.get("target_dimension_id") == domain_id
            ]
            row[f"cibol_{domain_id}"] = (
                float(np.mean(hits)) if hits else float("nan")
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = [f"cibol_{d}" for d in domain_ids]

    # ---- 传导矩阵 ----
    matrix_rows = []
    for domain_id in domain_ids:
        group = frame[frame["domain_id"] == domain_id]
        row = {"domain_id": domain_id, "n": int(len(group))}
        for column in columns:
            rho, p = _spearman(
                list(group["known_score"]),
                list(group[column].astype(float)),
            )
            dim = column[len("cibol_"):]
            row[f"rho_{dim}"] = rho
            row[f"p_{dim}"] = p
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)

    # ---- 分箱单调性 ----
    bin_rows = []
    for domain_id in domain_ids:
        group = frame[frame["domain_id"] == domain_id].sort_values("known_score")
        group = group.reset_index(drop=True)
        bins = np.array_split(group.index, 5)
        for bin_index, indexes in enumerate(bins, start=1):
            chunk = group.loc[indexes]
            bin_rows.append(
                {
                    "domain_id": domain_id,
                    "bin": bin_index,
                    "n": int(len(chunk)),
                    "known_score_min": float(chunk["known_score"].min()),
                    "known_score_max": float(chunk["known_score"].max()),
                    "cibol_target_mean": float(chunk[f"cibol_{domain_id}"].mean()),
                }
            )
    bin_frame = pd.DataFrame(bin_rows)

    # ---- 逐题 C/D 选择率 ----
    warnings = []
    option_rows = []
    for domain_id in domain_ids:
        group = frame[frame["domain_id"] == domain_id].sort_values("known_score")
        n = len(group)
        high_ids = set(group.tail(int(n * 0.4))["respondent_id"])
        low_ids = set(group.head(int(n * 0.4))["respondent_id"])
        for item in items:
            if item["target_dimension_id"] != domain_id:
                continue
            item_id = str(item["item_id"])
            item_records = [r for r in responses if r.get("item_id") == item_id]
            high_hits = sum(
                1
                for r in item_records
                if r.get("respondent_id") in high_ids and r.get("score") == 1
            )
            low_hits = sum(
                1
                for r in item_records
                if r.get("respondent_id") in low_ids and r.get("score") == 1
            )
            high_rate = high_hits / len(high_ids) if high_ids else float("nan")
            low_rate = low_hits / len(low_ids) if low_ids else float("nan")
            delta = high_rate - low_rate
            option_rows.append(
                {
                    "domain_id": domain_id,
                    "item_id": item_id,
                    "source_item_id": item.get("source_item_id"),
                    "high_bin_CD_rate": round(float(high_rate), 3),
                    "low_bin_CD_rate": round(float(low_rate), 3),
                    "delta": round(float(delta), 3),
                }
            )
            if delta < -0.10:
                warnings.append(
                    {
                        "item_id": item_id,
                        "source_item_id": item.get("source_item_id"),
                        "domain_id": domain_id,
                        "high_bin_CD_rate": round(float(high_rate), 3),
                        "low_bin_CD_rate": round(float(low_rate), 3),
                        "delta": round(float(delta), 3),
                        "note": "高分箱 C/D 选择率低于低分箱：计分方向或人格传导可疑",
                    }
                )
    option_frame = pd.DataFrame(option_rows)

    # ---- 均分分布 ----
    distribution_rows = []
    for domain_id in domain_ids:
        column = f"cibol_{domain_id}"
        values = frame[frame["domain_id"] == domain_id][column].dropna()
        distribution_rows.append(
            {
                "domain_id": domain_id,
                "mean": round(float(values.mean()), 3) if len(values) else None,
                "sd": round(float(values.std(ddof=1)), 3) if len(values) > 1 else None,
                "min": round(float(values.min()), 3) if len(values) else None,
                "max": round(float(values.max()), 3) if len(values) else None,
                "pct_below_0_15": round(float((values < 0.15).mean()), 3)
                if len(values)
                else None,
                "pct_above_0_85": round(float((values > 0.85).mean()), 3)
                if len(values)
                else None,
            }
        )
    distribution = pd.DataFrame(distribution_rows)

    matrix.to_csv(output_dir / "conductance_matrix.csv", index=False, encoding="utf-8-sig")
    option_frame.to_csv(
        output_dir / "option_rate_by_score_bin.csv", index=False, encoding="utf-8-sig"
    )
    bin_frame.to_csv(output_dir / "bin_monotonicity.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "item_level_warnings.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(warnings), handle, ensure_ascii=False, indent=2)

    header_cells = " | ".join(f"rho→{DOMAIN_LABELS.get(d, d)}" for d in domain_ids)
    lines = [
        "# CIBOL 内部传导自检报告（第 1 层）",
        "",
        f"- run_id: `{run_id}`",
        f"- 被试：{len(respondents)} 人（{len(domain_ids)} 组 × 每组 {len(respondents) // len(domain_ids)} 人）",
        f"- 计分：CIBOL 90 题，0/1（C/D=1、A/B=0），`{CIBOL_SCORING_VERSION}`",
        "",
        "## ① 构念效度：已知 domain 分数 → CIBOL 对应维度 0-1 均分（Spearman）",
        "",
        "期望：对角线（已知 domain 对应维度）显著为正。",
        "",
        f"| 已知 domain | n | {header_cells} |",
        "|---|--:|" + "|".join(["---:"] * len(domain_ids)) + "|",
    ]
    for _, row in matrix.iterrows():
        cells = []
        for domain_id in domain_ids:
            rho = row.get(f"rho_{domain_id}")
            p = row.get(f"p_{domain_id}")
            if rho is None or not np.isfinite(rho):
                cells.append("—")
            else:
                star = "**" if p is not None and p < 0.05 else ""
                cells.append(f"{star}{rho:.2f}{star}")
        lines.append(
            f"| {DOMAIN_LABELS.get(row['domain_id'], row['domain_id'])} | "
            f"{int(row['n'])} | " + " | ".join(cells) + " |"
        )

    lines += [
        "",
        "## ② 判别效度",
        "",
        "每一行中，对角线（对应维度）应明显大于非对角线。",
        "",
        "## ③ 计分方向自检（逐题 C/D 选择率）",
        "",
        f"反向可疑题数：{len(warnings)}",
        "",
    ]
    if warnings:
        lines.append("| item_id | 来源题号 | domain | 高分箱C/D率 | 低分箱C/D率 | Δ |")
        lines.append("|---|---|---|--:|--:|--:|")
        for warning in warnings:
            lines.append(
                f"| {warning['item_id']} | {warning['source_item_id']} | "
                f"{DOMAIN_LABELS.get(warning['domain_id'], warning['domain_id'])} | "
                f"{warning['high_bin_CD_rate']} | {warning['low_bin_CD_rate']} | "
                f"{warning['delta']} |"
            )
    else:
        lines.append("无反向可疑题。")

    lines += [
        "",
        "## 均分分布（0/1 计分方差可信度）",
        "",
        "| domain | mean | sd | min | max | <0.15比例 | >0.85比例 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in distribution.iterrows():
        lines.append(
            f"| {DOMAIN_LABELS.get(row['domain_id'], row['domain_id'])} | "
            f"{row['mean']} | {row['sd']} | {row['min']} | {row['max']} | "
            f"{row['pct_below_0_15']} | {row['pct_above_0_85']} |"
        )

    (output_dir / "conductance_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[report] 已写入 {output_dir}")


# ============================================================================
# 主流程
# ============================================================================


async def _main(args: argparse.Namespace) -> None:
    run_id = f"cibol-conductance-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "cibol_responses.jsonl"

    items, metadata = load_cibol_items()
    model = get_model()
    sjt_model, _method = with_compatible_structured_output(model, SJTSelectionOutput)
    semaphore = asyncio.Semaphore(args.concurrency)
    request_timeout_seconds = get_model_request_timeout_seconds()
    domain_ids = list(args.domains) if args.domains else CIBOL_DOMAIN_IDS
    if not domain_ids or set(domain_ids) - set(CIBOL_DOMAIN_IDS):
        raise ValueError(f"--domains 必须是 {CIBOL_DOMAIN_IDS} 的子集")

    print(
        f"[run] {run_id} | {len(domain_ids)} 组 × {args.sample_size} 人 × "
        f"{len(items)} 题 | 并发 {args.concurrency} | 模型 {getattr(model, 'model_name', '?')}"
    )
    profiles_by_group: dict[str, list[dict[str, Any]]] = {}
    group_summaries = []
    total_calls = 0
    total_errors = 0
    for group_index, domain_id in enumerate(domain_ids):
        refs, _diagnostics = generate_score_respondent_refs(
            args.sample_size,
            [{"dimension_id": domain_id, "mean_score": 50.0}],
            seed=args.seed + group_index,
        )
        for index, ref in enumerate(refs):
            ref["respondent_id"] = f"{domain_id}-{ref['respondent_id']}"
            ref["respondent_index"] = index
        profiles_by_group[domain_id] = refs
        started = time.perf_counter()
        summary = await _answer_cibol_group(
            domain_id=domain_id,
            refs=refs,
            items=items,
            sjt_model=sjt_model,
            semaphore=semaphore,
            output_path=response_path,
            max_retries=args.max_retries,
            retry_delay_seconds=1.0,
            request_timeout_seconds=request_timeout_seconds,
            seed=args.seed,
            run_id=run_id,
        )
        elapsed = time.perf_counter() - started
        group_summaries.append({**summary, "elapsed_seconds": round(elapsed, 1)})
        total_calls += summary["completed_calls"]
        total_errors += summary["error_count"]
        print(
            f"[group] {domain_id}: 新增 {summary['completed_calls']} 条作答，"
            f"失败 {summary['error_count']}，耗时 {elapsed:.0f}s"
        )
        with (output_dir / "group_summaries.json").open("w", encoding="utf-8") as handle:
            json.dump(group_summaries, handle, ensure_ascii=False, indent=2)

    if total_errors:
        print(
            f"[warn] 共 {total_errors} 条失败；重跑本脚本可断点续跑跳过已完成记录。"
        )
    with (output_dir / "score_profiles.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(profiles_by_group), handle, ensure_ascii=False, indent=2)
    (output_dir / "cibol_metadata.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    responses = []
    if response_path.exists():
        with response_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        responses.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    expected = args.sample_size * len(domain_ids) * len(items)
    if len(responses) < expected:
        print(
            f"[warn] 作答不完整：{len(responses)}/{expected}；"
            "报告基于现有数据，重跑可补全。"
        )
    if responses:
        _build_report(
            output_dir,
            run_id=run_id,
            domain_ids=domain_ids,
            items=items,
            profiles_by_group=profiles_by_group,
            responses=responses,
        )
    print(f"[done] 输出目录：{output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIBOL 内部传导自检（独立实验）")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--domains", nargs="*", help="domain 子集（默认全部 5 个）")
    args = parser.parse_args()
    with telemetry_run_context(f"cibol-conductance"):
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
