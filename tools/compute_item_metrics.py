"""为传导实验的每题计算"系统同款"单题迭代指标。

复用 sjt_system.evaluation.psychometrics._item_statistics 的口径
（标准化难度、分面内校正题总相关 CITC、选项功能、qualification 门禁），
并补算 rho_target（题分与已知 persona 分数的 Spearman）。

注意：
- 每题只用对应 persona 组（target 臂）的作答，与系统"target 臂资格"一致；
- same/cross-domain VTS 需要三臂设计（本实验每组只有 target 臂），故未计算，
  在输出中标注为 None；
- Mussel/CIBOL 是 0/1 二分计分，'effective_option_count' 门禁的解释要打折。

用法：
  python tools/compute_item_metrics.py --bank mussel --dir outputs/mussel_conductance/<run>
  python tools/compute_item_metrics.py --bank cibol  --dir outputs/cibol_conductance/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.evaluation.psychometrics import _item_statistics


def _load_bank(bank: str) -> list[dict[str, Any]]:
    if bank == "mussel":
        from sjt_system.evaluation.reference_questionnaires import load_mussel_items

        items, _meta = load_mussel_items()
        return items
    if bank == "cibol":
        from tools.run_cibol_conductance_check import load_cibol_items

        items, _meta = load_cibol_items()
        return items
    raise ValueError("bank 必须是 mussel 或 cibol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", choices=["mussel", "cibol"], required=True)
    parser.add_argument("--dir", required=True, help="传导实验输出目录")
    args = parser.parse_args()

    run_dir = Path(args.dir)
    response_path = next(run_dir.glob("*_responses.jsonl"))
    profile_path = run_dir / "score_profiles.json"
    records = [
        json.loads(line)
        for line in response_path.open(encoding="utf-8")
        if line.strip()
    ]
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    items = _load_bank(args.bank)
    items_by_id = {str(item["item_id"]): item for item in items}

    # 每组已知分数（respondent_id -> (group, score)）
    known = {}
    for group, refs in profiles.items():
        for ref in refs:
            rid = str(ref["respondent_id"])
            known[rid] = (group, float(ref["score_values"][group]))

    # 按 persona 组分组作答
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        rid = str(record["respondent_id"])
        group = known[rid][0]
        by_group.setdefault(group, []).append(record)

    rows: list[dict[str, Any]] = []
    for group, group_records in by_group.items():
        long_df = pd.DataFrame(
            [
                {
                    "respondent_id": r["respondent_id"],
                    "item_id": r["item_id"],
                    "selected_option_id": r["selected_option_id"],
                    "score": float(r["score"]),
                }
                for r in group_records
            ]
        )
        wide = long_df.pivot(index="respondent_id", columns="item_id", values="score")
        item_order = [str(item["item_id"]) for item in items]
        group_items = {
            str(item["item_id"]): item
            for item in items
            if item["target_dimension_id"] == group
        }
        group_order = [item_id for item_id in item_order if item_id in group_items]

        stats, _flat, _option = _item_statistics(
            long_df,
            wide,
            item_order=group_order,
            items=group_items,
        )
        for item_id in group_order:
            stat = stats[item_id]
            item = group_items[item_id]
            # rho_target：组内已知分数 vs 题分
            group_scores = wide[item_id]
            paired = [
                (known[rid][1], float(score))
                for rid, score in group_scores.items()
                if rid in known and known[rid][0] == group
            ]
            rho, p = scipy_stats.spearmanr(
                [pair[0] for pair in paired],
                [pair[1] for pair in paired],
            ) if len(paired) > 2 else (None, None)
            qual = stat["qualification"]
            rows.append(
                {
                    "item_id": item_id,
                    "source_item_id": item.get("source_item_id"),
                    "dimension_id": group,
                    "n": stat["n"],
                    "difficulty": stat["difficulty"],
                    "citc_r": stat["facet_corrected_item_total_correlation"].get("r"),
                    "rho_target": float(rho) if rho is not None else None,
                    "rho_target_p": float(p) if p is not None else None,
                    "mean": stat["mean"],
                    "sd": stat["standard_deviation"],
                    "min_option_rate": stat["minimum_option_rate"],
                    "effective_option_count": stat["effective_option_count"],
                    "difficulty_pass": qual["difficulty_pass"],
                    "citc_pass": qual["citc_pass"],
                    "option_rate_pass": qual["option_rate_pass"],
                    "qualified": qual["qualified"],
                    # 三臂指标：本实验未收集 same/cross 臂，无法计算
                    "same_domain_vts": None,
                    "cross_domain_vts": None,
                }
            )

    out = pd.DataFrame(rows).sort_values(["dimension_id", "item_id"])
    csv_path = run_dir / "item_iteration_metrics.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[item metrics] 已写入 {csv_path}（{len(out)} 题）")

    # 摘要
    print("\n=== 按维度汇总 ===")
    summary = (
        out.groupby("dimension_id")
        .agg(
            n_items=("item_id", "count"),
            qualified=("qualified", "sum"),
            mean_difficulty=("difficulty", "mean"),
            mean_citc=("citc_r", "mean"),
            mean_rho_target=("rho_target", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())
    print("\n=== 未达标题（difficulty/citc/option 任一门禁不过）===")
    failed = out[~out["qualified"]]
    if failed.empty:
        print("无")
    else:
        print(
            failed[
                [
                    "item_id",
                    "dimension_id",
                    "difficulty",
                    "citc_r",
                    "rho_target",
                    "difficulty_pass",
                    "citc_pass",
                    "option_rate_pass",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
