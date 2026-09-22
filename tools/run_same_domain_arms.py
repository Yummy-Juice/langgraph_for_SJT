"""为传导实验补跑 same_domain 臂（同域其他 facet 的被试组）。

现有传导实验每组只有 target 臂（Mussel 的 5 个 facet 各属不同 domain，
没有同域组），VTS 的 same_domain 臂因此缺失。本脚本对每个目标维度
补跑一个同域 facet 组（100 人，facet 级 persona，答全部题），写入
<run_dir>/same_domain_responses.jsonl。cross_domain 臂直接复用现有
其他组作答，无需重跑。

映射（目标 → 同域 facet）：
  Mussel:  openness_ideas → openness_aesthetics(O2)
           conscientiousness_self_discipline → conscientiousness_order(C2)
           extraversion_gregariousness → extraversion_warmth(E1)
           agreeableness_compliance → agreeableness_altruism(A3)
           neuroticism_self_consciousness → neuroticism_anxiety(N1)
  CIBOL:   openness → openness_ideas；conscientiousness → conscientiousness_order
           extraversion → extraversion_warmth；agreeableness → agreeableness_altruism
           neuroticism → neuroticism_anxiety

用法：
  python tools/run_same_domain_arms.py --bank mussel --dir outputs/mussel_conductance/<run>
      [--sample-size 100] [--concurrency 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.evaluation.respondents import generate_score_respondent_refs
from sjt_system.evaluation.simulation import SJTSelectionOutput
from tools.run_mussel_conductance_check import _answer_mussel_group
from tools.run_cibol_conductance_check import load_cibol_items
from sjt_system.runtime.telemetry import run_context as telemetry_run_context

MUSSEL_SAME_DOMAIN = {
    "openness_ideas": "openness_aesthetics",
    "conscientiousness_self_discipline": "conscientiousness_order",
    "extraversion_gregariousness": "extraversion_warmth",
    "agreeableness_compliance": "agreeableness_altruism",
    "neuroticism_self_consciousness": "neuroticism_anxiety",
}
CIBOL_SAME_DOMAIN = {
    "openness": "openness_ideas",
    "conscientiousness": "conscientiousness_order",
    "extraversion": "extraversion_warmth",
    "agreeableness": "agreeableness_altruism",
    "neuroticism": "neuroticism_anxiety",
}


async def _main(args: argparse.Namespace) -> None:
    run_dir = Path(args.dir)
    mapping = (
        MUSSEL_SAME_DOMAIN if args.bank == "mussel" else CIBOL_SAME_DOMAIN
    )
    if args.bank == "mussel":
        from sjt_system.evaluation.reference_questionnaires import load_mussel_items

        items, _meta = load_mussel_items()
        suffix = "mussel"
    else:
        items, _meta = load_cibol_items()
        suffix = "cibol"
    output_path = run_dir / f"same_domain_{suffix}_responses.jsonl"

    model = get_model()
    sjt_model, _method = with_compatible_structured_output(model, SJTSelectionOutput)
    semaphore = asyncio.Semaphore(args.concurrency)
    request_timeout_seconds = get_model_request_timeout_seconds()

    print(
        f"[same-domain] {args.bank} | {len(mapping)} 组 × {args.sample_size} 人 × "
        f"{len(items)} 题 | 并发 {args.concurrency}"
    )
    summary_rows = []
    for group_index, (target, same_facet) in enumerate(mapping.items()):
        refs, _diag = generate_score_respondent_refs(
            args.sample_size,
            [{"dimension_id": same_facet, "mean_score": 50.0}],
            seed=args.seed + 100 + group_index,
        )
        for index, ref in enumerate(refs):
            ref["respondent_id"] = f"{same_facet}-{ref['respondent_id']}"
            ref["respondent_index"] = index
        started = time.perf_counter()
        summary = await _answer_mussel_group(
            facet_id=same_facet,
            refs=refs,
            mussel_items=items,
            sjt_model=sjt_model,
            semaphore=semaphore,
            output_path=output_path,
            max_retries=args.max_retries,
            retry_delay_seconds=1.0,
            request_timeout_seconds=request_timeout_seconds,
            seed=args.seed,
            run_id=f"{suffix}-same-domain",
        )
        summary_rows.append(
            {
                "target": target,
                "same_domain_facet": same_facet,
                **summary,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(
            f"[group] {target} <- {same_facet}: 新增 {summary['completed_calls']} 条，"
            f"失败 {summary['error_count']}"
        )
        (run_dir / "same_domain_summaries.json").write_text(
            json.dumps(summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] same_domain 作答已写入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", choices=["mussel", "cibol"], required=True)
    parser.add_argument("--dir", required=True, help="传导实验输出目录")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    with telemetry_run_context(f"{args.bank}-same-domain"):
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
