"""Mussel 内部传导自检（第 1 层）— 独立实验。

对 5 个 Mussel 维度（开放性/责任心/外倾性/宜人性/神经质）各构造一组
target 虚拟被试（默认每组 100 人，已知 persona 分数 0-100 正态分布），
让每组被试作答全部 Mussel 110 题（0/1 计分：A/B=1、C/D=0），然后验证：

  ① 构念效度：已知分数 → 对应 Mussel 维度 0-1 均分的 Spearman 是否显著为正；
  ② 判别效度：已知分数 → 非对应 Mussel 维度的相关是否明显更弱；
  ③ 计分方向自检：逐题高分箱/低分箱的 A/B 选择率是否同向，
     反向题单独标出（预期暴露 self_consciousness 的 05/13/22 计分问题）。

输出到 outputs/mussel_conductance/<session>/：
  mussel_responses.jsonl   逐条 0/1 作答（断点续跑，重跑跳过已有）
  score_profiles.json      每名被试的已知 facet 与分数
  conductance_report.md    5×5 传导矩阵 + 三验证点结论
  conductance_matrix.csv   行=已知facet，列=Mussel维度，值=Spearman rho
  option_rate_by_score_bin.csv  每题 A/B 选择率（高分箱 vs 低分箱）
  item_level_warnings.json 方向可疑的题目

用法：
  python tools/run_mussel_conductance_check.py [--sample-size 100]
      [--concurrency 10] [--max-retries 2] [--seed 7] [--groups ...]
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
from sjt_system.evaluation.reference_questionnaires import (
    MUSSEL_FACET_TO_DIMENSION,
    MUSSEL_SCORING_VERSION,
    load_mussel_items,
    score_mussel_records,
)
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

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mussel_conductance"
MUSSEL_FACET_IDS = list(MUSSEL_FACET_TO_DIMENSION.values())


# ============================================================================
# 作答
# ============================================================================


def _build_persona_profiles(
    refs: list[dict[str, Any]],
    facet_id: str,
) -> list[dict[str, Any]]:
    """Minimal persona profiles consumed by build_persona_prompt."""
    return [
        {
            "respondent_id": str(ref["respondent_id"]),
            "score_values": dict(ref["score_values"]),
            "active_dimension_id": facet_id,
            "condition_id": "target",
        }
        for ref in refs
    ]


def _score_specs_for_facet(facet_id: str) -> list[dict[str, Any]]:
    profile = resolve_construct_profile(facet_id)
    facet = profile["facets"][0]
    return [
        {
            "dimension_id": facet_id,
            "level": "facet",
            "domain_id": profile["domain_id"],
            "domain_name_en": profile["domain_name_en"],
            "domain_name": profile["domain_name"],
            "facet_name_en": facet.get("facet_name_en") or facet_id,
            "facet_name": facet.get("facet_name") or facet_id,
        }
    ]


async def _answer_mussel_group(
    *,
    facet_id: str,
    refs: list[dict[str, Any]],
    mussel_items: list[dict[str, Any]],
    sjt_model: Any,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    max_retries: int,
    retry_delay_seconds: float,
    request_timeout_seconds: float,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    """Answer all 110 Mussel items for one persona group (resumable)."""

    profiles = _build_persona_profiles(refs, facet_id)
    score_specs = _score_specs_for_facet(facet_id)
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
                if not line.strip():
                    continue
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
                job_label=f"mussel-conductance/{facet_id}/{respondent_id}/{item_id}",
            )
        except Exception as exc:
            errors.append(f"{facet_id}/{respondent_id}/{item_id}: {exc}")
            return
        selected_option_id = display_to_original[raw_display]
        scoring_key = dict(item["scoring_key"])
        score = int(scoring_key[selected_option_id])
        record = {
            "record_type": "mussel_response",
            "run_id": run_id,
            "facet_group": facet_id,
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
            "scoring_version": MUSSEL_SCORING_VERSION,
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing_keys.add(key)
        completed += 1

    await asyncio.gather(
        *(
            one_job(reference, item)
            for reference in refs
            for item in mussel_items
        ),
        return_exceptions=True,
    )
    return {
        "facet_id": facet_id,
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
    facet_ids: list[str],
    mussel_items: list[dict[str, Any]],
    profiles_by_group: dict[str, list[dict[str, Any]]],
    responses: list[dict[str, Any]],
) -> None:
    """Compute the 5×5 conductance matrix and per-item checks."""

    expected_respondent_ids = sorted(
        {str(r.get("respondent_id")) for r in responses}
    )
    long_frame, facet_scores = score_mussel_records(
        responses,
        expected_respondent_ids=expected_respondent_ids,
        items=mussel_items,
    )
    # known facet score per respondent
    known = {}
    for facet_id, refs in profiles_by_group.items():
        for ref in refs:
            values = ref["score_values"]
            dimension_id, value = next(iter(values.items()))
            known[str(ref["respondent_id"])] = {
                "facet_id": facet_id,
                "score": float(value),
            }
    respondents = sorted(known)
    rows = []
    for rid in respondents:
        rows.append(
            {
                "respondent_id": rid,
                "facet_id": known[rid]["facet_id"],
                "known_score": known[rid]["score"],
                **{
                    f"mussel_{col}": float(facet_scores.loc[rid, col])
                    if col in facet_scores.columns and rid in facet_scores.index
                    else float("nan")
                    for col in facet_scores.columns
                },
            }
        )
    import pandas as pd

    frame = pd.DataFrame(rows)
    mussel_columns = [col for col in facet_scores.columns]

    # ---- ①/② 传导矩阵：每组 known score vs 5 个 Mussel 维度 ----
    matrix_rows = []
    for facet_id in facet_ids:
        group = frame[frame["facet_id"] == facet_id]
        row = {"facet_id": facet_id, "n": int(len(group))}
        for column in mussel_columns:
            rho, p = _spearman(
                list(group["known_score"]),
                list(group[f"mussel_{column}"].astype(float)),
            )
            row[f"rho_{column}"] = rho
            row[f"p_{column}"] = p
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)

    # ---- 分箱单调性：每组按 known score 5 分位箱，对应维度均分 ----
    bin_rows = []
    for facet_id in facet_ids:
        group = frame[frame["facet_id"] == facet_id].sort_values("known_score")
        group = group.reset_index(drop=True)
        bins = np.array_split(group.index, 5)
        for bin_index, indexes in enumerate(bins, start=1):
            chunk = group.loc[indexes]
            bin_rows.append(
                {
                    "facet_id": facet_id,
                    "bin": bin_index,
                    "n": int(len(chunk)),
                    "known_score_min": float(chunk["known_score"].min()),
                    "known_score_max": float(chunk["known_score"].max()),
                    "mussel_target_mean": float(chunk[f"mussel_{facet_id}"].mean()),
                }
            )
    bin_frame = pd.DataFrame(bin_rows)

    # ---- ③ 逐题 A/B 选择率：每组高分 40% vs 低分 40% ----
    warnings = []
    option_rows = []
    for facet_id in facet_ids:
        group = frame[frame["facet_id"] == facet_id].sort_values("known_score")
        n = len(group)
        high_ids = set(group.tail(int(n * 0.4))["respondent_id"])
        low_ids = set(group.head(int(n * 0.4))["respondent_id"])
        for item in mussel_items:
            item_id = str(item["item_id"])
            facet_key = item["facet_key"]
            if item["target_dimension_id"] != facet_id:
                continue
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
                    "facet_id": facet_id,
                    "item_id": item_id,
                    "source_item_id": item.get("source_item_id"),
                    "high_bin_AB_rate": round(float(high_rate), 3),
                    "low_bin_AB_rate": round(float(low_rate), 3),
                    "delta": round(float(delta), 3),
                }
            )
            if delta < -0.10:
                warnings.append(
                    {
                        "item_id": item_id,
                        "source_item_id": item.get("source_item_id"),
                        "facet_id": facet_id,
                        "high_bin_AB_rate": round(float(high_rate), 3),
                        "low_bin_AB_rate": round(float(low_rate), 3),
                        "delta": round(float(delta), 3),
                        "note": (
                            "高分箱 A/B 选择率低于低分箱：计分方向或人格传导可疑"
                        ),
                    }
                )
    option_frame = pd.DataFrame(option_rows)

    # ---- 均分分布（0/1 计分方差可信度诊断）----
    distribution_rows = []
    for facet_id in facet_ids:
        column = f"mussel_{facet_id}"
        values = frame[frame["facet_id"] == facet_id][column].dropna()
        distribution_rows.append(
            {
                "facet_id": facet_id,
                "mean": round(float(values.mean()), 3) if len(values) else None,
                "sd": round(float(values.std(ddof=1)), 3) if len(values) > 1 else None,
                "min": round(float(values.min()), 3) if len(values) else None,
                "max": round(float(values.max()), 3) if len(values) else None,
                "pct_below_0_15": round(
                    float((values < 0.15).mean()), 3
                )
                if len(values)
                else None,
                "pct_above_0_85": round(
                    float((values > 0.85).mean()), 3
                )
                if len(values)
                else None,
            }
        )
    distribution = pd.DataFrame(distribution_rows)

    # ---- 写文件 ----
    matrix.to_csv(output_dir / "conductance_matrix.csv", index=False, encoding="utf-8-sig")
    option_frame.to_csv(
        output_dir / "option_rate_by_score_bin.csv", index=False, encoding="utf-8-sig"
    )
    bin_frame.to_csv(output_dir / "bin_monotonicity.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "item_level_warnings.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(warnings), handle, ensure_ascii=False, indent=2)

    # ---- Markdown 报告 ----
    lines = [
        "# Mussel 内部传导自检报告（第 1 层）",
        "",
        f"- run_id: `{run_id}`",
        f"- 被试：{len(respondents)} 人（{len(facet_ids)} 组 × 每组 {len(respondents) // len(facet_ids)} 人）",
        f"- 计分：Mussel 110 题，0/1（A/B=1、C/D=0），`{MUSSEL_SCORING_VERSION}`",
        "",
        "## ① 构念效度：已知分数 → Mussel 对应维度 0-1 均分（Spearman）",
        "",
        "期望：对角线（已知 facet 对应维度）显著为正。",
        "",
        "| 已知 facet | n | " + " | ".join(f"rho→{c}" for c in mussel_columns) + " |",
        "|---|--:|" + "|".join(["---:"] * len(mussel_columns)) + "|",
    ]
    for _, row in matrix.iterrows():
        cells = []
        for column in mussel_columns:
            rho = row.get(f"rho_{column}")
            p = row.get(f"p_{column}")
            if rho is None or not np.isfinite(rho):
                cells.append("—")
            else:
                star = "**" if p is not None and p < 0.05 else ""
                cells.append(f"{star}{rho:.2f}{star}")
        lines.append(f"| {row['facet_id']} | {int(row['n'])} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## ② 判别效度",
        "",
        "每一行中，对角线（对应维度）应明显大于非对角线。",
        "",
        "## ③ 计分方向自检（逐题 A/B 选择率）",
        "",
        f"反向可疑题数：{len(warnings)}",
        "",
    ]
    if warnings:
        lines.append("| item_id | 来源题号 | facet | 高分箱A/B率 | 低分箱A/B率 | Δ |")
        lines.append("|---|---|---|--:|--:|--:|")
        for warning in warnings:
            lines.append(
                f"| {warning['item_id']} | {warning['source_item_id']} | "
                f"{warning['facet_id']} | {warning['high_bin_AB_rate']} | "
                f"{warning['low_bin_AB_rate']} | {warning['delta']} |"
            )
    else:
        lines.append("无反向可疑题。")

    lines += [
        "",
        "## 均分分布（0/1 计分方差可信度）",
        "",
        "若某维度均分大量挤在 0.15 以下或 0.85 以上，说明 persona 传导过强或选项区分不足，"
        "相关会虚高，需谨慎解释。",
        "",
        "| facet | mean | sd | min | max | <0.15比例 | >0.85比例 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in distribution.iterrows():
        lines.append(
            f"| {row['facet_id']} | {row['mean']} | {row['sd']} | {row['min']} | "
            f"{row['max']} | {row['pct_below_0_15']} | {row['pct_above_0_85']} |"
        )

    (output_dir / "conductance_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[report] 已写入 {output_dir}")


# ============================================================================
# 主流程
# ============================================================================


async def _main(args: argparse.Namespace) -> None:
    run_id = f"mussel-conductance-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "mussel_responses.jsonl"

    mussel_items, mussel_meta = load_mussel_items()
    model = get_model()
    sjt_model, _method = with_compatible_structured_output(
        model,
        SJTSelectionOutput,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    request_timeout_seconds = get_model_request_timeout_seconds()
    facet_ids = list(args.groups) if args.groups else MUSSEL_FACET_IDS
    if not facet_ids or set(facet_ids) - set(MUSSEL_FACET_IDS):
        raise ValueError(f"--groups 必须是 {MUSSEL_FACET_IDS} 的子集")

    print(
        f"[run] {run_id} | {len(facet_ids)} 组 × {args.sample_size} 人 × "
        f"{len(mussel_items)} 题 | 并发 {args.concurrency} | 模型 {getattr(model, 'model_name', '?')}"
    )
    profiles_by_group: dict[str, list[dict[str, Any]]] = {}
    group_summaries = []
    total_calls = 0
    total_errors = 0
    for group_index, facet_id in enumerate(facet_ids):
        specs = [{"dimension_id": facet_id, "mean_score": 50.0}]
        refs, _diagnostics = generate_score_respondent_refs(
            args.sample_size,
            specs,
            seed=args.seed + group_index,
        )
        for index, ref in enumerate(refs):
            # generate_score_respondent_refs 生成的 ID 是固定格式（score-0001），
            # 不同 facet 组之间会冲突；加 facet 前缀保证全局唯一。
            ref["respondent_id"] = f"{facet_id}-{ref['respondent_id']}"
            ref["respondent_index"] = index
        profiles_by_group[facet_id] = refs
        started = time.perf_counter()
        summary = await _answer_mussel_group(
            facet_id=facet_id,
            refs=refs,
            mussel_items=mussel_items,
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
            f"[group] {facet_id}: 新增 {summary['completed_calls']} 条作答，"
            f"失败 {summary['error_count']}，耗时 {elapsed:.0f}s"
        )
        with (output_dir / "group_summaries.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(group_summaries, handle, ensure_ascii=False, indent=2)

    if total_errors:
        print(
            f"[warn] 共 {total_errors} 条失败；重跑本脚本可断点续跑跳过已完成记录。"
        )

    with (output_dir / "score_profiles.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(profiles_by_group),
            handle,
            ensure_ascii=False,
            indent=2,
        )
    mussel_meta_path = output_dir / "mussel_metadata.json"
    mussel_meta_path.write_text(
        json.dumps(_json_safe(mussel_meta), ensure_ascii=False, indent=2),
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

    expected = args.sample_size * len(facet_ids) * len(mussel_items)
    if len(responses) < expected:
        print(
            f"[warn] 作答不完整：{len(responses)}/{expected}；"
            "报告基于现有数据，重跑可补全。"
        )
    if responses:
        _build_report(
            output_dir,
            run_id=run_id,
            facet_ids=facet_ids,
            mussel_items=mussel_items,
            profiles_by_group=profiles_by_group,
            responses=responses,
        )
    print(f"[done] 输出目录：{output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mussel 内部传导自检（独立实验）")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--groups",
        nargs="*",
        help="facet_id 子集（默认全部 5 个）",
    )
    args = parser.parse_args()
    with telemetry_run_context(f"mussel-conductance"):
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
