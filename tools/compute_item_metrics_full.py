"""用完整三臂数据计算每题的系统同款单题指标（含 VTS）。

数据来源（不重跑）：
- target 臂：<run_dir>/<bank>_responses.jsonl（原有 5 组）
- same_domain 臂：<run_dir>/same_domain_<bank>_responses.jsonl（补跑）
- cross_domain 臂：复用其他 target 组

对齐 sjt_system 的 matched 口径：
  facet_citc        = Pearson(题分, 同维剩余分)（target 臂内）
  rho_target        = Spearman(target 臂已知分数, 题分)
  rho_same_domain   = Spearman(same 臂已知分数, 题分)
  rho_cross_domain  = max(Spearman(各 cross 组已知分数, 题分))（带符号最大）
  same_domain_vts   = rho_target − rho_same_domain
  cross_domain_vts  = rho_target − rho_cross_domain

用法：
  python tools/compute_item_metrics_full.py --bank mussel --dir outputs/mussel_conductance/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.evaluation.psychometrics import _item_statistics

SAME_DOMAIN_MAP = {
    "openness_ideas": "openness_aesthetics",
    "conscientiousness_self_discipline": "conscientiousness_order",
    "extraversion_gregariousness": "extraversion_warmth",
    "agreeableness_compliance": "agreeableness_altruism",
    "neuroticism_self_consciousness": "neuroticism_anxiety",
    "openness": "openness_ideas",
    "conscientiousness": "conscientiousness_order",
    "extraversion": "extraversion_warmth",
    "agreeableness": "agreeableness_altruism",
    "neuroticism": "neuroticism_anxiety",
}


def _spearman(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    if len(left) < 3:
        return None, None
    try:
        result = scipy_stats.spearmanr(left, right)
    except Exception:
        return None, None
    rho = float(getattr(result, "statistic", np.nan))
    p = float(getattr(result, "pvalue", np.nan))
    if not np.isfinite(rho):
        return None, None
    return rho, p


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", choices=["mussel", "cibol"], required=True)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.dir)
    if args.bank == "mussel":
        from sjt_system.evaluation.reference_questionnaires import load_mussel_items

        items, _ = load_mussel_items()
    else:
        from tools.run_cibol_conductance_check import load_cibol_items

        items, _ = load_cibol_items()
    items_by_id = {str(item["item_id"]): item for item in items}

    target_records = _load_records(run_dir / f"{args.bank}_responses.jsonl")
    same_records = _load_records(run_dir / f"same_domain_{args.bank}_responses.jsonl")
    profiles = json.loads(
        (run_dir / "score_profiles.json").read_text(encoding="utf-8")
    )
    print(f"target 记录 {len(target_records)} | same_domain 记录 {len(same_records)}")

    # 已知分数映射
    known = {}
    for group, refs in profiles.items():
        for ref in refs:
            known[str(ref["respondent_id"])] = (group, float(ref["score_values"][group]))
    same_known = {}
    for record in same_records:
        rid = str(record["respondent_id"])
        if rid not in same_known:
            facet = rid.split("-")[0]
            # 从同域作答记录的 facet_group 字段或前缀推断已知分数：
            # same_domain 记录里 respondent_id 前缀 = same facet id
            same_known[rid] = None  # 分数在 score_profiles 里没有，需从 profiles 补
    # same_domain 组的分数：重新从 profiles 读（profiles 只有 target 组）
    # 补跑时 same 组分数由 generate_score_respondent_refs 生成，未存 profiles；
    # 从作答记录取 active score？same 记录里没有 active_score。
    # 方案：从 score_profiles.json 没有，则提示用户；或从同域组 refs 重新生成（种子可复现）。

    # ---- 用可复现生成重建 same 组分数（与补跑脚本 seed 一致）----
    from sjt_system.evaluation.respondents import generate_score_respondent_refs

    bank_mapping = (
        SAME_DOMAIN_MAP if args.bank == "mussel" else SAME_DOMAIN_MAP
    )
    bank_keys = (
        [
            "openness_ideas",
            "conscientiousness_self_discipline",
            "extraversion_gregariousness",
            "agreeableness_compliance",
            "neuroticism_self_consciousness",
        ]
        if args.bank == "mussel"
        else ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    )
    rebuilt = {}
    for group_index, target in enumerate(bank_keys):
        same_facet = SAME_DOMAIN_MAP[target]
        same_group_records = [
            r for r in same_records
            if r.get("facet_group") == same_facet
            or str(r.get("respondent_id", "")).startswith(same_facet + "-")
        ]
        sample_size = len(
            {str(r["respondent_id"]) for r in same_group_records}
        )
        if sample_size < 1:
            continue
        refs, _ = generate_score_respondent_refs(
            sample_size,
            [{"dimension_id": same_facet, "mean_score": 50.0}],
            seed=7 + 100 + group_index,
        )
        for ref in refs:
            rid = f"{same_facet}-{ref['respondent_id']}"
            rebuilt[rid] = (target, same_facet, float(ref["score_values"][same_facet]))
    print(f"重建 same_domain 组 {len(rebuilt)} 人")

    # 组装每题指标
    rows = []

    # 逐维度处理：按“被试组”（respondent_id 前缀/known 映射）过滤，而非题目维度
    def _by_known_group(records: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
        return [r for r in records if known.get(r.get("respondent_id"), (None,))[0] == group]

    for target, same_facet in SAME_DOMAIN_MAP.items():
        if target not in bank_keys:
            continue
        target_items = [i for i in items if i["target_dimension_id"] == target]
        if not target_items:
            continue
        # target 臂：被试组 == target 的作答
        target_group = _by_known_group(target_records, target)
        # same 臂（该同域 facet 组的作答）
        same_group = [
            r for r in same_records
            if r.get("facet_group") == same_facet
            or r.get("domain_group") == same_facet
            or str(r.get("respondent_id", "")).startswith(same_facet + "-")
        ]
        # cross 臂：其他被试组的作答
        cross_groups = [
            (g, _by_known_group(target_records, g))
            for g in bank_keys
            if g != target
        ]
        if not target_group or not same_group:
            print(f"[skip] {target}: target={len(target_group)} same={len(same_group)}")
            continue

        def _frame(records):
            df = pd.DataFrame(
                [
                    {
                        "respondent_id": r["respondent_id"],
                        "item_id": r["item_id"],
                        "selected_option_id": r["selected_option_id"],
                        "score": float(r["score"]),
                    }
                    for r in records
                ]
            )
            return df

        t_long, t_wide = _frame(target_group), _frame(target_group).pivot(
            index="respondent_id", columns="item_id", values="score"
        )
        s_long = _frame(same_group)
        item_map = {str(i["item_id"]): i for i in target_items}
        order = [str(i["item_id"]) for i in target_items]
        stats, _, _ = _item_statistics(t_long, t_wide, item_order=order, items=item_map)

        # target 臂已知分数
        t_known = {rid: known[rid][1] for rid in t_wide.index if rid in known}
        # same 臂已知分数
        s_known = {rid: rebuilt[rid][2] for rid in s_long["respondent_id"].unique() if rid in rebuilt}

        for item_id in order:
            item_scores_t = t_wide[item_id]
            rho_t, p_t = _spearman(
                [t_known[rid] for rid in item_scores_t.index if rid in t_known],
                [float(item_scores_t[rid]) for rid in item_scores_t.index if rid in t_known],
            )
            same_respondents = s_long[s_long["item_id"] == item_id]
            rho_s, p_s = _spearman(
                [s_known[rid] for rid in same_respondents["respondent_id"] if rid in s_known],
                [float(x) for x in same_respondents["score"]],
            ) if len(same_respondents) else (None, None)
            cross_rhos = []
            for g, g_records in cross_groups:
                g_records = [r for r in g_records if r.get("item_id") == item_id]
                if not g_records:
                    continue
                g_df = pd.DataFrame(
                    [
                        {"respondent_id": r["respondent_id"], "score": float(r["score"])}
                        for r in g_records
                    ]
                )
                g_known = {
                    rid: known[rid][1]
                    for rid in g_df["respondent_id"]
                    if rid in known
                }
                if len(g_known) >= 3:
                    rho_c, _ = _spearman(
                        [g_known[rid] for rid in g_df["respondent_id"] if rid in g_known],
                        [g_df.loc[g_df["respondent_id"] == rid, "score"].iloc[0] for rid in g_df["respondent_id"] if rid in g_known],
                    )
                    if rho_c is not None:
                        cross_rhos.append((g, rho_c))
            max_cross = max(cross_rhos, key=lambda pair: pair[1]) if cross_rhos else (None, None)
            stat = stats[item_id]
            qual = stat["qualification"]
            rows.append(
                {
                    "item_id": item_id,
                    "dimension_id": target,
                    "difficulty": stat["difficulty"],
                    "citc_r": stat["facet_corrected_item_total_correlation"].get("r"),
                    "rho_target": rho_t,
                    "rho_same_domain": rho_s,
                    "rho_cross_domain": max_cross[1],
                    "largest_cross_group": max_cross[0],
                    "same_domain_vts": (rho_t - rho_s) if rho_t is not None and rho_s is not None else None,
                    "cross_domain_vts": (rho_t - max_cross[1]) if rho_t is not None and max_cross[1] is not None else None,
                    "difficulty_pass": qual["difficulty_pass"],
                    "citc_pass": qual["citc_pass"],
                    "qualified": qual["qualified"],
                }
            )

    out = pd.DataFrame(rows).sort_values(["dimension_id", "item_id"])
    csv_path = run_dir / "item_iteration_metrics_full.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[full metrics] 已写入 {csv_path}（{len(out)} 题）")
    if out.empty:
        return
    summary = (
        out.groupby("dimension_id")
        .agg(
            n=("item_id", "count"),
            mean_rho_target=("rho_target", "mean"),
            mean_same_vts=("same_domain_vts", "mean"),
            mean_cross_vts=("cross_domain_vts", "mean"),
            mean_citc=("citc_r", "mean"),
            qualified=("qualified", "sum"),
        )
        .round(3)
    )
    print("\n=== 按维度汇总（含 VTS）===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
