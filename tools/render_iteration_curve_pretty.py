"""Pretty whole-test iteration curve renderer (standalone SVG/HTML).

Reads psychometric_iteration_history from a checkpoint and renders a polished
HTML page: candidate/best-so-far quality, raw diagnostic lines, cumulative
token/time costs, best-round marker, and plateau status.

Usage:
  python tools/render_iteration_curve_pretty.py <checkpoint.json> [--out out.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.evaluation.form_metrics import (  # noqa: E402
    assess_form_plateau,
    form_quality_summary,
)

METRIC_STYLE = {
    "best_quality": {"label": "历史最优质量", "color": "#6a1b9a", "icon": "●"},
    "candidate_quality": {"label": "本轮候选质量", "color": "#ef6c00", "icon": "◆"},
    "r2": {"label": "目标恢复 R²", "color": "#e91e63", "icon": "◉"},
    "selectivity": {"label": "构念选择性", "color": "#2196f3", "icon": "▲"},
}

W, H = 980, 620
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 40, 60
CHART_W = W - PAD_L - PAD_R
CHART_H = 300  # 主图高度
COST_H = 130   # 成本图高度
GAP = 30


def _smooth_path(points: list[tuple[float, float]]) -> str:
    """Catmull-Rom -> cubic Bezier for a smooth curve through all points."""
    if len(points) < 2:
        return ""
    d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        d += f" C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}"
    return d


def _load_rows(checkpoint_path: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state = raw.get("state") if isinstance(raw, dict) and "state" in raw else raw
    history = state.get("psychometric_iteration_history") or []
    plateau = assess_form_plateau(history)
    trajectory = {
        int(row.get("analysis_round") or 0): row
        for row in plateau.get("trajectory") or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for h in history:
        fm = h.get("form_metrics") or {}
        if fm.get("status") != "complete":
            continue
        rel = fm.get("reliability") or {}
        val = fm.get("validity") or {}
        rec = val.get("target_recovery") or {}
        summary = form_quality_summary(fm)
        quality_row = trajectory.get(int(h.get("analysis_round") or 0)) or {}
        icc = rel.get("virtual_test_retest_icc")
        r2 = rec.get("cross_validated_r2")
        selectivity = summary.get("construct_selectivity")
        candidate_quality = summary.get("candidate_form_quality")
        best_quality = quality_row.get("best_so_far_form_quality")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in (icc, r2, selectivity, candidate_quality, best_quality)
        ):
            continue
        rows.append({
            "round": int(h.get("analysis_round") or 0),
            "icc": float(icc),
            "icc_passed": bool((summary.get("stability_gate") or {}).get("passed")),
            "r2": float(r2),
            "selectivity": float(selectivity),
            "candidate_quality": float(candidate_quality),
            "best_quality": float(best_quality),
            "tokens": int((h.get("token_usage") or {}).get("total_tokens") or 0),
            "duration_ms": int((h.get("token_usage") or {}).get("duration_ms") or 0),
        })
    return rows, str(state.get("run_id") or checkpoint_path.stem), plateau


def render(checkpoint_path: Path, out_path: Path) -> None:
    rows, run_id, plateau = _load_rows(checkpoint_path)
    if not rows:
        print("[warn] 没有完整整卷指标轮次")
        return
    n = len(rows)
    xs = [PAD_L + i * CHART_W / max(1, n - 1) for i in range(n)]
    best_round = int(plateau.get("best_round") or rows[0]["round"])

    def _y(value: float, top: float, height: float) -> float:
        return top + height - max(0.0, min(1.0, value)) * height

    # ---- 主图折线 ----
    lines = ""
    dots = ""
    labels = ""
    for key in ("best_quality", "candidate_quality", "r2", "selectivity"):
        style = METRIC_STYLE[key]
        color = style["color"]
        pts = [(xs[i], _y(r[key], PAD_T, CHART_H)) for i, r in enumerate(rows)]
        # 面积渐变
        area = (
            f'<polygon points="{pts[0][0]:.1f},{PAD_T + CHART_H} '
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            + f" {pts[-1][0]:.1f},{PAD_T + CHART_H}"
            + f'" fill="{color}" opacity="0.08"/>'
        )
        lines += area
        lines += (
            f'<path d="{_smooth_path(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, r in enumerate(rows):
            x, y = pts[i]
            halo = (
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="#fff" stroke="{color}" '
                f'stroke-width="2.5"><title>R{r["round"]} {style["label"]}: {r[key]:.4f}</title></circle>'
            )
            val = (
                f'<text x="{x:.1f}" y="{y - 12 - (i % 2) * 14:.1f}" text-anchor="middle" '
                f'font-size="12" fill="{color}" font-weight="600">{r[key]:.3f}</text>'
            )
            dots += halo
            labels += val

    # ---- 成本柱（累计）----
    cost_rows = []
    cum = 0.0
    cum_time = 0.0
    for r in rows:
        cum += r["tokens"]
        cum_time += r["duration_ms"]
        cost_rows.append((r["tokens"], cum, cum_time))
    max_cum = max(cum for _, cum, _ in cost_rows) or 1.0
    max_time = max(duration for _, _, duration in cost_rows) or 1.0
    cost_top = PAD_T + CHART_H + GAP
    bars = ""
    cost_labels = ""
    time_points = []
    for i, (per_round, cum, cumulative_time) in enumerate(cost_rows):
        x = xs[i]
        h_bar = cum / max_cum * COST_H
        time_points.append(
            (
                x,
                cost_top + COST_H - cumulative_time / max_time * COST_H,
            )
        )
        bars += (
            f'<rect x="{x - 16}" y="{cost_top + COST_H - h_bar:.1f}" width="32" '
            f'height="{h_bar:.1f}" rx="4" fill="#b0bec5" opacity="0.85">'
            f'<title>R{i+1}: 本轮 {per_round/1e4:.0f} 万 / 累计 {cum/1e4:.0f} 万</title></rect>'
        )
        cost_labels += (
            f'<text x="{x:.0f}" y="{cost_top + COST_H - h_bar - 6:.0f}" text-anchor="middle" '
            f'font-size="11" fill="#546e7a">{cum/1e4:.0f}万</text>'
        )
    time_line = (
        f'<path d="{_smooth_path(time_points)}" fill="none" stroke="#455a64" '
        f'stroke-width="2" stroke-dasharray="5 3"/>'
        if len(time_points) >= 2
        else ""
    )
    cost_axis = (
        f'<text x="{PAD_L - 8}" y="{cost_top + 14}" text-anchor="end" font-size="11" fill="#888">0</text>'
        f'<text x="{PAD_L - 8}" y="{cost_top + COST_H - 6:.0f}" text-anchor="end" font-size="11" fill="#888">累计token</text>'
    )

    # ---- 网格 + 轴 ----
    grid = ""
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _y(v, PAD_T, CHART_H)
        grid += (
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
            f'stroke="#e3e6ea" stroke-width="1"/>'
            f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#9aa0a6">{v:.1f}</text>'
        )
    x_axis = "".join(
        f'<text x="{x:.0f}" y="{PAD_T + CHART_H + 22}" text-anchor="middle" font-size="13" font-weight="600" fill="#37474f">'
        f'第 {r["round"]} 轮</text>'
        for x, r in zip(xs, rows)
    )
    # best 标记
    best_marker = ""
    best_x = xs[rows.index(next(r for r in rows if r["round"] == best_round))]
    best_marker = (
        f'<rect x="{best_x - 24}" y="{PAD_T + CHART_H - 14}" width="48" height="18" rx="9" '
        f'fill="#ff9800" opacity="0.15"/>'
        f'<text x="{best_x:.0f}" y="{PAD_T + CHART_H - 1:.0f}" text-anchor="middle" font-size="10" '
        f'fill="#e65100" font-weight="700">BEST</text>'
    )

    # ---- 图例 ----
    legend = "".join(
        f'<span style="margin-right:18px;color:{METRIC_STYLE[k]["color"]};font-weight:600;">'
        f'{METRIC_STYLE[k]["icon"]} {METRIC_STYLE[k]["label"]}</span>'
        for k in ("best_quality", "candidate_quality", "r2", "selectivity")
    ) + '<span style="color:#78909c;">▮ 累计token　┄ 累计模型耗时</span>'

    plateau_txt = {
        "reached": "已到达平台期（停止返修，用最佳卷收卷）",
        "monitoring": "监测中（尚未触发平台期）",
        "insufficient_data": "数据不足",
    }.get(plateau.get("status"), plateau.get("status") or "未知")
    plateau_color = "#c62828" if plateau.get("reached") else "#2e7d32"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>整卷迭代曲线 · {run_id[:8]}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f6f8; margin: 0; padding: 28px; }}
  .card {{ background: #fff; border-radius: 14px; box-shadow: 0 2px 14px rgba(0,0,0,.06); padding: 24px 28px; max-width: 1020px; margin: 0 auto; }}
  h2 {{ margin: 0 0 6px; color: #263238; font-weight: 650; }}
  .meta {{ color: #78909c; font-size: 13px; margin-bottom: 14px; }}
  .legend {{ font-size: 13px; margin-bottom: 10px; }}
  .plateau {{ display:inline-block; padding: 4px 12px; border-radius: 12px; background: {plateau_color}18; color: {plateau_color}; font-size: 12.5px; font-weight: 600; }}
  .table {{ width:100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }}
  .table th, .table td {{ padding: 8px 10px; text-align: center; border-bottom: 1px solid #eceff1; }}
  .table th {{ color:#78909c; font-weight: 600; background:#fafbfc; }}
  .up {{ color:#2e7d32; }} .down {{ color:#c62828; }}
</style></head><body><div class="card">
<h2>整卷迭代曲线 <span class="plateau">{plateau_txt}</span></h2>
<div class="meta">run {run_id} · {n} 轮完整整卷指标 · BEST 标记 = 当前历史最佳轮次</div>
<div class="legend">{legend}</div>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="width:100%;height:auto;">
  {grid}{lines}{dots}{labels}
  {best_marker}
  <line x1="{PAD_L}" y1="{PAD_T + CHART_H}" x2="{W - PAD_R}" y2="{PAD_T + CHART_H}" stroke="#cfd8dc" stroke-width="1.5"/>
  <line x1="{PAD_L}" y1="{cost_top}" x2="{W - PAD_R}" y2="{cost_top}" stroke="#cfd8dc" stroke-width="1.5" stroke-dasharray="4"/>
  {x_axis}{bars}{time_line}{cost_labels}{cost_axis}
</svg>
<table class="table">
<tr><th>轮次</th><th>目标恢复 R²</th><th>构念选择性</th><th>本轮候选质量</th><th>历史最优质量</th><th>ICC门槛</th><th>累计 token</th><th>累计耗时</th></tr>
{"".join(
    f'<tr><td><b>第 {r["round"]} 轮</b>{(" 🏅" if r["round"] == best_round else "")}</td>'
    f'<td>{r["r2"]:.3f}</td><td>{r["selectivity"]:.3f}</td><td>{r["candidate_quality"]:.3f}</td>'
    f'<td>{r["best_quality"]:.3f}</td><td>{r["icc"]:.3f}（{"通过" if r["icc_passed"] else "未通过"}）</td>'
    f'<td>{sum(x["tokens"] for x in rows[:i+1])/1e4:.0f} 万</td>'
    f'<td>{sum(x["duration_ms"] for x in rows[:i+1])/3600000:.2f} 小时</td></tr>'
    for i, r in enumerate(rows)
)}
</table>
</div></body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[pretty curve] 已写入 {out_path}（{n} 轮）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    cp = Path(args.checkpoint)
    out = Path(args.out) if args.out else cp.with_name(cp.stem + "_iteration_curve_pretty.html")
    render(cp, out)


if __name__ == "__main__":
    main()
