"""Generate a single HTML report aggregating all conductance experiments.

Reads the Mussel / CIBOL experiment outputs under outputs/ and renders one
standalone page (with inline data + iframes to the existing heatmaps).

Usage:
  python tools/render_experiment_report.py --out outputs/experiment_report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MUSSEL_RUN = OUTPUT_ROOT / "mussel_conductance" / "mussel-conductance-20260826-233331"
CIBOL_RUN = OUTPUT_ROOT / "cibol_conductance" / "cibol-conductance-20260827-004906"

LABELS = {
    "openness_ideas": "开放性",
    "conscientiousness_self_discipline": "责任心",
    "extraversion_gregariousness": "外倾性",
    "agreeableness_compliance": "宜人性",
    "neuroticism_self_consciousness": "神经质",
    "openness": "开放性",
    "conscientiousness": "责任心",
    "extraversion": "外倾性",
    "agreeableness": "宜人性",
    "neuroticism": "神经质",
}


def _label(value: str) -> str:
    return LABELS.get(value, value)


def _matrix_table(matrix: pd.DataFrame) -> str:
    rho_cols = [c for c in matrix.columns if c.startswith("rho_")]
    dims = [c[4:] for c in rho_cols]
    header = "".join(
        f"<th>{_label(d)}</th>" for d in dims
    )
    rows = []
    for _, row in matrix.iterrows():
        row_id = str(row.get("facet_id") or row.get("domain_id") or "")
        cells = "".join(
            f"<td>{row.get(f'rho_{d}') if pd.notna(row.get(f'rho_{d}')) else '—'}</td>"
            for d in dims
        )
        rows.append(f"<tr><td><b>{_label(row_id)}</b></td>{cells}</tr>")
    return f"<table><tr><th></th>{header}</tr>{''.join(rows)}</table>"


def _summary_table(csv_name: str, run_dir: Path, columns: list[str]) -> str:
    path = run_dir / csv_name
    if not path.exists():
        return "<p>（无数据）</p>"
    frame = pd.read_csv(path)
    dim_col = "dimension_id" if "dimension_id" in frame.columns else "domain_id"
    summary = (
        frame.groupby(dim_col)
        .agg({col: "mean" for col in columns})
        .round(3)
    )
    header = "".join(f"<th>{c}</th>" for c in [dim_col] + columns)
    rows = "".join(
        f"<tr><td><b>{_label(idx)}</b></td>"
        + "".join(f"<td>{val}</td>" for val in row)
        + "</tr>"
        for idx, row in summary.iterrows()
    )
    return f"<table><tr>{header}</tr>{rows}</table>"


def _warnings_table(run_dir: Path) -> str:
    path = run_dir / "item_level_warnings.json"
    if not path.exists():
        return "<p>无</p>"
    warnings = json.loads(path.read_text(encoding="utf-8"))
    if not warnings:
        return "<p>无反向可疑题</p>"
    rows = "".join(
        f"<tr><td>{w['item_id']}</td><td>{w.get('source_item_id', '')}</td>"
        f"<td>{_label(w.get('facet_id') or w.get('domain_id', ''))}</td>"
        f"<td>{w.get('high_bin_CD_rate') or w.get('high_bin_AB_rate')}</td>"
        f"<td>{w.get('low_bin_CD_rate') or w.get('low_bin_AB_rate')}</td>"
        f"<td>{w.get('delta')}</td></tr>"
        for w in warnings
    )
    return (
        "<table><tr><th>题号</th><th>来源题号</th><th>维度</th>"
        "<th>高分箱选高率</th><th>低分箱选高率</th><th>Δ</th></tr>"
        f"{rows}</table>"
    )


def _section(bank_label: str, run_dir: Path) -> str:
    matrix_path = run_dir / "conductance_matrix.csv"
    matrix_html = ""
    if matrix_path.exists():
        matrix_html = _matrix_table(pd.read_csv(matrix_path))
    vts_cols = [
        "rho_target",
        "same_domain_vts",
        "cross_domain_vts",
        "citc_r",
        "qualified",
    ]
    return f"""
    <h2>{bank_label}</h2>
    <h3>① 传导矩阵（Spearman ρ，行=已知分数，列=维度得分）</h3>
    {matrix_html}
    <h3>② 完整单题指标（含 VTS，均值）</h3>
    {_summary_table('item_iteration_metrics_full.csv', run_dir, vts_cols)}
    <h3>③ 计分方向警告题</h3>
    {_warnings_table(run_dir)}
    <h3>④ 热图</h3>
    <iframe src="{run_dir.name}/conductance_heatmap.html" style="width:100%;height:420px;border:1px solid #ddd;"></iframe>
    <iframe src="{run_dir.name}/conductance_matrix_diagonal.html" style="width:100%;height:300px;border:1px solid #ddd;"></iframe>
    """


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(OUTPUT_ROOT / "experiment_report.html"))
    args = parser.parse_args()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>虚拟被试验证实验汇总</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1100px; }}
table {{ border-collapse: collapse; margin: 8px 0 16px; }}
th, td {{ border: 1px solid #ccc; padding: 5px 9px; font-size: 13px; text-align: center; }}
th {{ background: #f0f0f0; }}
h2 {{ border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 36px; }}
h3 {{ color: #444; }}
.verdict {{ background: #fff8e1; border: 1px solid #f0d070; padding: 12px 16px; border-radius: 6px; }}
.bad {{ color: #c62828; }} .good {{ color: #2e7d32; }}
</style></head><body>
<h1>虚拟被试验证实验汇总（Mussel + CIBOL）</h1>
<p style="color:#666;">全部为开发期虚拟证据：AI 扮演者（persona 分数驱动）作答真人验证过的题目，检验虚拟被试的作答行为是否可信。不是人类效度数据。</p>

<div class="verdict">
<b>核心结论：</b>
① 方向（构念效度）<span class="good">✅ 基本成立</span>——4/5 维度对角线相关 0.89~0.98，AI 会按给定分数答对应方向；
② 特异性（判别效度/VTS）<span class="bad">❌ 不成立</span>——一个分数拉着所有维度跑，same_domain VTS 几乎全线不达标；
③ 宜人性 <span class="bad">❌ 全面失灵</span>（两套题库都最弱）；
④ 与真人对比（第 3 层）<span>未验证</span>——需 CIBOL 真人逐题数据。
</div>

{_section('Mussel（110 题，500 人 + 500 同域臂）', MUSSEL_RUN)}
{_section('CIBOL（100 题，500 人 + 500 同域臂）', CIBOL_RUN)}

<h2>第 2 层：Mussel 原文 vs 虚拟（问卷级锚定）</h2>
<p>来源：Mussel, Gatzka &amp; Hewig (2018, EJPA) Table 1，255 名真人样本。
原文 SJT 的 5 个 facet 与我们的映射一致（N=自我意识、E=乐群、O=观念开放、A=顺从、C=自律）。</p>

<h3>① Cronbach α 对比</h3>
<table>
<tr><th>维度</th><th>原文 α</th><th>虚拟 α</th><th>判断</th></tr>
<tr><td>N 自我意识</td><td>.73</td><td>.878</td><td class="bad">虚拟偏高</td></tr>
<tr><td>E 乐群</td><td>.75</td><td>.927</td><td class="bad">虚拟偏高</td></tr>
<tr><td>O 观念开放</td><td>.70</td><td>.898</td><td class="bad">虚拟偏高</td></tr>
<tr><td>A 顺从</td><td>.56</td><td>.451</td><td>虚拟更低</td></tr>
<tr><td>C 自律</td><td>.55</td><td>.756</td><td class="bad">虚拟偏高</td></tr>
</table>

<h3>② 收敛效度对比（原文 SJT↔NEO 自陈 / 虚拟 persona↔维度分）</h3>
<table>
<tr><th>维度</th><th>原文</th><th>虚拟</th><th>判断</th></tr>
<tr><td>N</td><td>.60</td><td>.89</td><td>方向一致，虚拟高</td></tr>
<tr><td>E</td><td>.66</td><td>.93</td><td>方向一致</td></tr>
<tr><td>O</td><td>.70</td><td>.96</td><td>方向一致</td></tr>
<tr><td>A</td><td>.41</td><td>.53</td><td><b>两边都最低</b></td></tr>
<tr><td>C</td><td>.52</td><td>.96</td><td>方向一致</td></tr>
</table>

<h3>③ SJT 维度间相关方向（原文 / 虚拟）</h3>
<table>
<tr><th>维度对</th><th>原文</th><th>虚拟</th><th>判断</th></tr>
<tr><td>N–E</td><td>-.35</td><td>-.58</td><td class="good">方向一致</td></tr>
<tr><td>N–O</td><td>-.29</td><td>-.57</td><td class="good">方向一致</td></tr>
<tr><td>N–A</td><td>+.17</td><td>+.17</td><td class="good">方向一致</td></tr>
<tr><td>N–C</td><td>+.11</td><td>-.43</td><td class="bad">方向相反</td></tr>
<tr><td>E–O</td><td>+.19</td><td>+.54</td><td>方向一致，量级放大</td></tr>
<tr><td>E–C</td><td>-.04</td><td>+.27</td><td class="bad">方向相反</td></tr>
<tr><td>O–C</td><td>+.19</td><td>+.63</td><td>方向一致，量级放大</td></tr>
</table>
<p><b>第 2 层结论：</b>α 量级、N 负相关、收敛方向基本复现（部分通过）；但量级系统性放大
（全局传导 + 作答太一致），个别维度相关方向相反。
<b>重要发现：A（顺从）在原文里就是最弱维度</b>（α .56、收敛 .41 均最低）——
宜人性弱有题目层面（compliance 本来就难测）的真实原因，不全是虚拟机制的问题。</p>

<h2>第 3 层：CIBOL 真人 vs 虚拟（整卷对比）</h2>
<p>真人数据：大五文字 PSJT 真人施测（见数平台招募 242 人，有效 205 人，表 3-1~3-4）。
虚拟数据：500 名虚拟被试作答 CIBOL_ZH.json（100 题，0/1 计分）。</p>

<h3>① Cronbach α 对比</h3>
<table>
<tr><th>维度</th><th>真人</th><th>虚拟</th><th>判断</th></tr>
<tr><td>宜人性</td><td>.815</td><td>.660</td><td class="bad">虚拟偏低</td></tr>
<tr><td>尽责性</td><td>.85</td><td>.872</td><td>接近</td></tr>
<tr><td>外向性</td><td>.91</td><td>.844</td><td>接近</td></tr>
<tr><td>神经质</td><td>.827</td><td>.859</td><td>接近</td></tr>
<tr><td>开放性</td><td>.907</td><td>.891</td><td>接近</td></tr>
</table>

<h3>② 汇聚效度对比（真人 PSJT↔大五简版 / 虚拟 persona↔维度分）</h3>
<table>
<tr><th>维度</th><th>真人</th><th>虚拟</th><th>判断</th></tr>
<tr><td>宜人性</td><td>.544</td><td>.104</td><td class="bad">虚拟远低</td></tr>
<tr><td>尽责性</td><td>.618</td><td>.651</td><td>接近</td></tr>
<tr><td>外向性</td><td>.834</td><td>.591</td><td class="bad">虚拟偏低</td></tr>
<tr><td>神经质</td><td>.644</td><td>.531</td><td>虚拟偏低</td></tr>
<tr><td>开放性</td><td>.757</td><td>.708</td><td>虚拟偏低</td></tr>
</table>

<h3>③ 维度间相关量级（真人 / 虚拟）</h3>
<table>
<tr><th>维度对</th><th>真人</th><th>虚拟</th><th>判断</th></tr>
<tr><td>宜人–尽责</td><td>-.136</td><td>-.611</td><td class="bad">放大</td></tr>
<tr><td>宜人–外向</td><td>-.163</td><td>-.576</td><td class="bad">放大</td></tr>
<tr><td>尽责–外向</td><td>.492</td><td>.709</td><td>放大</td></tr>
<tr><td>外向–神经质</td><td>-.532</td><td>-.582</td><td>方向一致</td></tr>
<tr><td>尽责–开放</td><td>.507</td><td>.811</td><td class="bad">放大</td></tr>
</table>
<p><b>第 3 层结论（整卷层面）：</b>
① α 量级与真人接近（除宜人性）；
② 汇聚效度方向全一致，但<b>宜人性虚拟远低于真人</b>（.104 vs .544）、外向性也偏低——
<b>真人版宜人性并不弱</b>（α .815、汇聚 .544），说明宜人性弱是虚拟被试机制问题，不是题目问题；
③ 维度间相关方向大体一致，但虚拟量级为真人 2~3 倍（全局传导）。
<br>注意事项：真人表为 1-4 计分、每维 16 题（实测后删除不佳题），虚拟为 0/1 计分、18~22 题/维；
计分与题量差异会压低虚拟 α，但不影响"宜人性差距最大"的结论。逐题对比因题目对应关系不一致暂未做。</p>

<h2>系统整卷迭代质量（plateau 检测）</h2>
<p>系统开发迭代时用 <code>psychometric_iteration_history</code> 记录每轮整卷质量，
<code>assess_form_plateau</code> 连续 2 轮没有提高历史最优整卷质量（min_delta=0.01）即停止返修。</p>
<table>
<tr><th>指标</th><th>含义</th><th>期望</th></tr>
<tr><td>target_recovery_cross_validated_r2</td><td>留出交叉验证下，从整套题目作答模式恢复预设目标人格的 R²</td><td>高（如 &gt;0.3）</td></tr>
<tr><td>construct_selectivity</td><td>目标效应占目标效应与最大非目标泄漏总量的比例</td><td>接近 1</td></tr>
<tr><td>candidate_form_quality</td><td>目标恢复R²与构念选择性的几何平均</td><td>高且超过历史最优</td></tr>
<tr><td>virtual_test_retest_icc</td><td>同一 target 虚拟人格重复施测整套测验的 ICC(A,1) 绝对一致性</td><td>只作为稳定性门槛</td></tr>
</table>
<p style="color:#666;">注：target_recovery 和 virtual_test_retest_icc 需要"同人格重复施测"数据，本次传导实验未采集，
construct_selectivity 可用现有三臂数据估算。</p>

<h2>成本</h2>
<table>
<tr><th>实验</th><th>调用次数</th><th>token</th><th>费用（其余时段）</th></tr>
<tr><td>Mussel 传导 + 同域臂</td><td>110,000</td><td>≈ 2,240 万</td><td>≈ ¥60</td></tr>
<tr><td>CIBOL 传导 + 同域臂</td><td>100,000</td><td>≈ 1,900 万</td><td>≈ ¥50</td></tr>
</table>

</body></html>"""

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] 已写入 {out_path}")


if __name__ == "__main__":
    main()
