"""虚拟整卷传导指标迭代轨迹报告生成器（离线、确定性）。

读取运行检查点中的 psychometric_iteration_history，把每轮临时组卷的
候选质量、历史最优质量、目标恢复R²和构念选择性画成迭代曲线。

用法：
    python -X utf8 tools/iteration_trajectory.py
输出：
    outputs/iteration_trajectory.html
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sjt_system.evaluation.form_metrics import (  # noqa: E402
    assess_form_plateau,
    form_quality_summary,
)

OUT = ROOT / "outputs" / "iteration_trajectory.html"
VR = ROOT / "outputs" / "virtual_responses"


def version_number(name: str) -> int:
    return int(name.split("-v")[1].split("-")[0])


def load_run_rows(run_dir: Path) -> list[dict]:
    checkpoint = ROOT / "outputs" / "run_checkpoints" / f"{run_dir.name}.json"
    if not checkpoint.is_file():
        return []
    state = json.loads(checkpoint.read_text(encoding="utf-8")).get("state") or {}
    history = state.get("psychometric_iteration_history") or []
    plateau = assess_form_plateau(history)
    trajectory = {
        int(row.get("analysis_round") or 0): row
        for row in plateau.get("trajectory") or []
        if isinstance(row, dict)
    }
    rows = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("form_metrics") or {}
        reliability = metrics.get("reliability") or {}
        validity = metrics.get("validity") or {}
        recovery = validity.get("target_recovery") or {}
        quality = form_quality_summary(metrics)
        quality_row = trajectory.get(int(entry.get("analysis_round") or 0)) or {}
        rows.append({
            "version": int(entry.get("analysis_round") or 0),
            "n": metrics.get("sample_size"),
            "stability": reliability.get("virtual_test_retest_icc"),
            "stability_passed": (quality.get("stability_gate") or {}).get("passed"),
            "recovery": recovery.get("cross_validated_r2"),
            "selectivity": quality.get("construct_selectivity"),
            "candidate_quality": quality.get("candidate_form_quality"),
            "best_quality": quality_row.get("best_so_far_form_quality"),
            "tokens": int((entry.get("token_usage") or {}).get("total_tokens") or 0),
            "duration_ms": int((entry.get("token_usage") or {}).get("duration_ms") or 0),
        })
    return sorted(rows, key=lambda row: row["version"])


def repair_events(run_id: str) -> int | None:
    cp = ROOT / "outputs" / "run_checkpoints" / f"{run_id}.json"
    if not cp.exists():
        return None
    state = json.loads(cp.read_text(encoding="utf-8")).get("state") or {}
    history = state.get("psychometric_repair_history") or []
    return len(history)


def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def num(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN 检查


# ---------------- SVG 折线图 ----------------

def line_chart(
    series: list[dict], key: str, title: str, y_label: str,
    lo: float, hi: float, thresholds: list[tuple[float, str]] = (),
    color: str = "#2563eb", width: int = 560, height: int = 240,
) -> str:
    points = []
    for i, row in enumerate(series):
        y = num(row.get(key))
        if y is None:
            continue
        points.append((i, y))
    x_values = list(range(len(series)))
    pad = 0.10 * (hi - lo)
    y_lo = max(0.0, lo - pad) if lo >= 0.0 else lo - pad
    y_hi = hi + pad
    margin_l, margin_r, margin_t, margin_b = 56, 20, 28, 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def px(i):
        return margin_l + plot_w * (i / max(1, len(series) - 1))

    def py(y):
        return margin_t + plot_h * (1 - (y - y_lo) / (y_hi - y_lo))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" aria-label="{esc(title)}">']
    # 网格与 y 轴刻度
    ticks = 5
    for t in range(ticks + 1):
        yv = y_lo + (y_hi - y_lo) * t / ticks
        yy = py(yv)
        parts.append(f'<line x1="{margin_l}" y1="{yy:.1f}" x2="{width - margin_r}" '
                     f'y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 6}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#64748b">{yv:.2f}</text>')
    # x 轴标签 = 版本号
    for i in x_values:
        parts.append(f'<text x="{px(i):.1f}" y="{height - 12}" text-anchor="middle" '
                     f'font-size="10" fill="#475569">v{series[i]["version"]}</text>')
    # 阈值线
    for tv, label in thresholds:
        if y_lo <= tv <= y_hi:
            yy = py(tv)
            parts.append(f'<line x1="{margin_l}" y1="{yy:.1f}" x2="{width - margin_r}" '
                         f'y2="{yy:.1f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{width - margin_r - 4}" y="{yy - 4:.1f}" text-anchor="end" '
                         f'font-size="9" fill="#94a3b8">{esc(label)}</text>')
    # 折线
    if points:
        coords = " ".join(f"{px(i):.1f},{py(y):.1f}" for i, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for i, y in points:
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(y):.1f}" r="3.6" fill="{color}"/>')
    # 标题
    parts.append(f'<text x="{margin_l}" y="{margin_t - 10}" font-size="12.5" font-weight="700" '
                 f'fill="#22303f">{esc(title)}</text>')
    parts.append(f'<text x="{margin_l}" y="{height - 16}" font-size="10" fill="#64748b">'
                  f'x = 心理测量分析轮次　y = {esc(y_label)}　点旁数字 = 样本量 n</text>')
    # 样本量标注
    for i, row in enumerate(series):
        y = num(row.get(key))
        if y is None or row.get("n") is None:
            continue
        parts.append(f'<text x="{px(i):.1f}" y="{py(y) + 16:.1f}" text-anchor="middle" '
                     f'font-size="8.5" fill="#94a3b8">n={row["n"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def run_card(run_id: str, rows: list[dict], events: int | None) -> str:
    recoveries = [num(r.get("recovery")) for r in rows if num(r.get("recovery")) is not None]
    selectivities = [num(r.get("selectivity")) for r in rows if num(r.get("selectivity")) is not None]
    candidates = [num(r.get("candidate_quality")) for r in rows if num(r.get("candidate_quality")) is not None]
    bests = [num(r.get("best_quality")) for r in rows if num(r.get("best_quality")) is not None]
    if not recoveries and not selectivities and not candidates and not bests:
        return ""
    all_vals = recoveries + selectivities + candidates + bests
    lo, hi = min(all_vals), max(all_vals)
    lo = min(lo, 0.0)
    hi = max(hi, 1.0)
    cumulative_tokens = 0
    cumulative_duration = 0
    for row in rows:
        cumulative_tokens += int(row.get("tokens") or 0)
        cumulative_duration += int(row.get("duration_ms") or 0)
        row["cumulative_tokens"] = cumulative_tokens
        row["cumulative_duration_hours"] = cumulative_duration / 3_600_000

    events_note = ""
    if events is not None:
        events_note = f' · 心理测量修题事件 {events} 次'

    def fmt(value) -> str:
        numeric = num(value)
        return "—" if numeric is None else f"{numeric:.3f}"

    table_rows = "".join(
        f'<tr><td class="num">v{r["version"]}</td><td class="num">{r["n"] if r["n"] else "—"}</td>'
        f'<td class="num">{fmt(r["recovery"])}</td><td class="num">{fmt(r["selectivity"])}</td>'
        f'<td class="num">{fmt(r["candidate_quality"])}</td><td class="num">{fmt(r["best_quality"])}</td>'
        f'<td class="num">{fmt(r["stability"])}（{"通过" if r["stability_passed"] else "未通过"}）</td>'
        f'<td class="num">{r["cumulative_tokens"]}</td>'
        f'<td class="num">{r["cumulative_duration_hours"]:.2f}</td></tr>'
        for r in rows
    )
    return f"""
<div class="card">
  <h3>run {esc(run_id[:8])} · {len(rows)} 个版本{events_note}</h3>
  <div class="charts">
    {line_chart(rows, "best_quality", "历史最优整卷质量（只升或持平）", "BFQ", lo, hi,
                [(0.0, "0")], color="#6a1b9a")}
    {line_chart(rows, "candidate_quality", "本轮候选整卷质量（允许波动）", "U", lo, hi,
                [(0.0, "0")], color="#ef6c00")}
    {line_chart(rows, "recovery", "原始诊断：目标恢复 R²", "R²", lo, hi,
                [(0.0, "0")], color="#059669")}
    {line_chart(rows, "selectivity", "原始诊断：构念选择性", "C", lo, hi,
                [(0.0, "0")], color="#2563eb")}
    {line_chart(rows, "cumulative_tokens", "累计 Token", "Token", 0.0,
                max(1.0, float(cumulative_tokens)), [], color="#64748b")}
    {line_chart(rows, "cumulative_duration_hours", "累计模型耗时", "小时", 0.0,
                max(1.0, cumulative_duration / 3_600_000), [], color="#475569")}
  </div>
  <table>
    <tr><th>轮次</th><th class="num">n</th><th class="num">目标恢复R²</th><th class="num">构念选择性</th>
        <th class="num">本轮候选质量</th><th class="num">历史最优质量</th><th class="num">ICC门槛</th>
        <th class="num">累计Token</th><th class="num">累计耗时(h)</th></tr>
    {table_rows}
  </table>
</div>"""


def main() -> None:
    runs = []
    for run_dir in sorted(glob.glob(str(VR / "*"))):
        run_id = os.path.basename(run_dir)
        rows = load_run_rows(Path(run_dir))
        if len(rows) >= 2:
            runs.append((run_id, rows, repair_events(run_id)))

    # 汇总：首尾版本变化
    deltas = {"best_quality": [], "candidate_quality": [], "recovery": [], "selectivity": []}
    for _, rows, _ in runs:
        first, last = rows[0], rows[-1]
        for key in deltas:
            if num(first.get(key)) is not None and num(last.get(key)) is not None:
                deltas[key].append(num(last[key]) - num(first[key]))

    def stat(values):
        if not values:
            return "—"
        mean = sum(values) / len(values)
        pos = sum(1 for v in values if v > 0.001)
        return (f"均值 {mean:+.3f} · 上升 {pos}/{len(values)} 个run")

    cards = "".join(run_card(run_id, rows, events) for run_id, rows, events in runs)

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8">
<title>虚拟整卷传导指标迭代轨迹报告</title>
<style>
body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f6f8fb;color:#22303f;
  line-height:1.7;margin:0}}
.wrap{{max-width:1160px;margin:0 auto;padding:26px 18px 80px}}
h1{{font-size:24px;margin:0 0 6px}}
.sub{{color:#64748b;font-size:13.5px;margin-bottom:20px}}
h2{{font-size:18px;border-left:4px solid #2563eb;padding-left:10px;margin:34px 0 12px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;
  margin:14px 0;box-shadow:0 1px 3px rgba(15,40,80,.05)}}
.card h3{{margin:0 0 12px;font-size:15.5px}}
.charts{{display:flex;gap:14px;flex-wrap:wrap}}
.charts svg{{flex:1;min-width:420px;max-width:560px;border:1px solid #e2e8f0;border-radius:10px;
  background:#fff}}
table{{border-collapse:collapse;margin-top:12px;font-size:12.5px}}
th,td{{border:1px solid #e2e8f0;padding:5px 10px;text-align:left}}
th{{background:#f1f5f9;color:#475569}}
td.num{{text-align:center}}
.summary td{{padding:8px 12px}}
.summary th{{width:180px}}
.caveat{{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;
  font-size:13.5px;margin:10px 0}}
.caveat b{{color:#b45309}}
footer{{margin-top:50px;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;padding-top:12px}}
</style></head>
<body><div class="wrap">
<h1>虚拟整卷传导指标随迭代的轨迹</h1>
<div class="sub">数据源：outputs/run_checkpoints/* 中的整卷迭代历史（全部为虚拟被试开发期证据）·
生成器：tools/iteration_trajectory.py</div>

<h2>一、跨 run 汇总：从首版本到末版本的变化</h2>
<div class="card">
<table class="summary">
<tr><th>指标</th><th>首→末变化</th><th>含义</th></tr>
<tr><td>历史最优整卷质量</td><td class="num">{stat(deltas["best_quality"])}</td><td>系统截至当前保留的最佳完整卷质量，只升或持平</td></tr>
<tr><td>本轮候选整卷质量</td><td class="num">{stat(deltas["candidate_quality"])}</td><td>本轮新组合的质量，允许上下波动</td></tr>
<tr><td>目标恢复R²</td><td class="num">{stat(deltas["recovery"])}</td><td>整套作答模式能否在留出样本中恢复目标分数</td></tr>
<tr><td>构念选择性</td><td class="num">{stat(deltas["selectivity"])}</td><td>目标信号占目标信号与最大非目标泄漏总量的比例</td></tr>
</table>
<p style="font-size:12.5px;color:#64748b">注意：不同 run 的样本量不同（30/50/100）、目标构念不同（compliance / gregariousness / self-discipline / openness_to_ideas 等），汇总只是趋势参考，不作统计推断。</p>
</div>

<h2>二、每个 run 的迭代轨迹（{len(runs)} 个有 ≥2 个版本的 run）</h2>
{cards}

<h2>三、结论与使用边界</h2>
<div class="caveat"><b>读图须知：</b>
① x 轴是<b>题库版本号</b>，不是严格的"修题轮次"——版本也会因重计分（评分键变更）而递增；与心理测量修题事件数对照使用。<br>
② 点旁标注样本量 n：同一 run 内 n 变化（如 30→100）时，前后不可直接比。<br>
③ 虚拟被试证据只能用于<b>开发期决策</b>，这些指标描述的是模型—提示词—题目系统的内部传导，不是真人信效度。<br>
④ 虚拟重测ICC仅作为稳定性门槛，不作为递增优化目标。</div>
<footer>生成：tools/iteration_trajectory.py · 无 LLM 参与 · 可直接重新运行以包含最新 run</footer>
</div></body></html>"""

    OUT.write_text(html_doc, encoding="utf-8")
    print(f"[trajectory] {len(runs)} 个多版本 run 已写入 {OUT}")
    print(f"[trajectory] best quality: {stat(deltas['best_quality'])}")
    print(f"[trajectory] candidate:    {stat(deltas['candidate_quality'])}")
    print(f"[trajectory] recovery:     {stat(deltas['recovery'])}")
    print(f"[trajectory] selectivity:  {stat(deltas['selectivity'])}")


if __name__ == "__main__":
    main()
