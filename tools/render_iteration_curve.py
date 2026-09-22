"""Render the whole-test iteration curve from a run checkpoint.

Reads psychometric_iteration_history from a checkpoint and renders an HTML
page with candidate quality, retained best-so-far quality, their two diagnostic
components, and cumulative token cost. Pure inline SVG, no external deps.

Usage:
  python tools/render_iteration_curve.py <checkpoint.json> [--out out.html]
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


def _load_history(checkpoint_path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state = raw.get("state") if isinstance(raw, dict) and "state" in raw else raw
    history = state.get("psychometric_iteration_history") or []
    run_id = str(state.get("run_id") or raw.get("run_id") or checkpoint_path.stem)
    return list(history), run_id


def _row(
    h: dict[str, Any],
    trajectory_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    fm = h.get("form_metrics") or {}
    if fm.get("status") != "complete":
        return None
    rel = fm.get("reliability") or {}
    val = fm.get("validity") or {}
    rec = val.get("target_recovery") or {}
    summary = form_quality_summary(fm)
    usage = h.get("token_usage") or {}
    icc = rel.get("virtual_test_retest_icc")
    r2 = rec.get("cross_validated_r2")
    selectivity = summary.get("construct_selectivity")
    candidate_quality = summary.get("candidate_form_quality")
    best_quality = (trajectory_row or {}).get("best_so_far_form_quality")
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in (icc, r2, selectivity, candidate_quality, best_quality)
    ):
        return None
    return {
        "round": int(h.get("analysis_round") or 0),
        "icc": float(icc),
        "icc_passed": bool((summary.get("stability_gate") or {}).get("passed")),
        "r2": float(r2),
        "selectivity": float(selectivity),
        "candidate_quality": float(candidate_quality),
        "best_quality": float(best_quality) if isinstance(best_quality, (int, float)) else None,
        "tokens": int(usage.get("total_tokens") or 0),
        "duration_ms": int(usage.get("duration_ms") or 0),
    }


def render(checkpoint_path: Path, out_path: Path) -> None:
    history, run_id = _load_history(checkpoint_path)
    plateau = assess_form_plateau(history)
    trajectory = {
        int(row.get("analysis_round") or 0): dict(row)
        for row in plateau.get("trajectory") or []
        if isinstance(row, dict)
    }
    rows = [
        _row(h, trajectory.get(int(h.get("analysis_round") or 0)))
        for h in history
    ]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("[warn] 没有完整的整卷指标轮次")
        return

    width, height = 900, 420
    pad_left, pad_right, pad_top, pad_bottom = 60, 20, 24, 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def _points(values: list[float]) -> str:
        n = len(values)
        return " ".join(
            f"{pad_left + i * plot_w / max(1, n - 1):.1f},"
            f"{pad_top + plot_h - max(0.0, min(1.0, v)) * plot_h:.1f}"
            for i, v in enumerate(values)
        )

    r2_line = _points([r["r2"] for r in rows])
    selectivity_line = _points([r["selectivity"] for r in rows])
    candidate_line = _points([r["candidate_quality"] for r in rows])
    best_line = _points([r["best_quality"] for r in rows])

    x_labels = "".join(
        f'<text x="{pad_left + i * plot_w / max(1, len(rows) - 1):.0f}" '
        f'y="{height - 12}" text-anchor="middle" font-size="12">R{i + 1}</text>'
        for i in range(len(rows))
    )
    # y 轴刻度（0~1）
    y_ticks = "".join(
        f'<text x="{pad_left - 8}" y="{pad_top + plot_h - v * plot_h:.0f}" '
        f'text-anchor="end" font-size="11" fill="#666">{v:.1f}</text>'
        f'<line x1="{pad_left}" y1="{pad_top + plot_h - v * plot_h:.0f}" '
        f'x2="{width - pad_right}" y2="{pad_top + plot_h - v * plot_h:.0f}" '
        f'stroke="#eee" stroke-width="1"/>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    # 各点数值标注
    def _labels(rows_key: str, offset: int, color: str) -> str:
        parts = []
        for i, r in enumerate(rows):
            v = r[rows_key]
            parts.append(
                f'<text x="{pad_left + i * plot_w / max(1, len(rows) - 1):.0f}" '
                f'y="{pad_top + plot_h * (1 - v) - 6 - offset}" text-anchor="middle" '
                f'font-size="11" fill="{color}">{v:.3f}</text>'
            )
        return "".join(parts)

    # token 成本条（底部，用累计 token 归一化）
    cumulative = 0.0
    cumulative_time = 0.0
    token_bars = []
    for r in rows:
        cumulative += r["tokens"]
        cumulative_time += r["duration_ms"]
    max_cum = cumulative or 1.0
    max_time = cumulative_time or 1.0
    cum = 0.0
    cum_time = 0.0
    for index, r in enumerate(rows):
        cum += r["tokens"]
        cum_time += r["duration_ms"]
        bar_h = cum / max_cum * 40
        time_h = cum_time / max_time * 40
        x = pad_left + index * plot_w / max(1, len(rows) - 1)
        token_bars.append(
            f'<rect x="{x - 16:.0f}" y="{height - 52 - bar_h:.1f}" width="14" height="{bar_h:.1f}" '
            f'fill="#d0d0d0" rx="2"/>'
            f'<rect x="{x + 2:.0f}" y="{height - 52 - time_h:.1f}" width="14" height="{time_h:.1f}" '
            f'fill="#78909c" rx="2"/>'
            f'<text x="{x:.0f}" '
            f'y="{height - 56 - bar_h:.0f}" text-anchor="middle" font-size="10" '
            f'fill="#888">{cum / 10000:.0f}万</text>'
        )

    legend = (
        '<span style="color:#6a1b9a;">— 历史最优质量</span> '
        '<span style="color:#ef6c00;">— 本轮候选质量</span> '
        '<span style="color:#d32f2f;">— 目标恢复R²</span> '
        '<span style="color:#1976d2;">— 构念选择性</span> '
        '<span style="color:#999;">▮ 累计token</span> '
        '<span style="color:#78909c;">▮ 累计模型耗时</span>'
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>整卷迭代曲线 {run_id[:8]}</title></head>
<body style="font-family:system-ui,sans-serif;margin:24px;">
<h2>整卷迭代曲线（run {run_id[:8]}）</h2>
<p>{legend}</p>
<svg width="{width}" height="{height}" style="border:1px solid #eee;">
  {y_ticks}
  <polyline points="{best_line}" fill="none" stroke="#6a1b9a" stroke-width="3"/>
  <polyline points="{candidate_line}" fill="none" stroke="#ef6c00" stroke-width="2"/>
  <polyline points="{r2_line}" fill="none" stroke="#d32f2f" stroke-width="1.5" stroke-dasharray="5 3"/>
  <polyline points="{selectivity_line}" fill="none" stroke="#1976d2" stroke-width="1.5" stroke-dasharray="5 3"/>
  {_labels('best_quality', 0, '#6a1b9a')}
  {_labels('candidate_quality', 14, '#ef6c00')}
  {_labels('r2', 0, '#d32f2f')}
  {_labels('selectivity', 28, '#1976d2')}
  {token_bars}
  {x_labels}
</svg>
<table border="1" cellpadding="6" style="border-collapse:collapse;margin-top:12px;font-size:13px;">
<tr><th>轮次</th><th>目标恢复R²</th><th>构念选择性</th><th>本轮候选质量</th><th>历史最优质量</th><th>ICC门槛</th><th>累计token</th><th>累计耗时</th></tr>
{"".join(
    f'<tr><td>R{r["round"]}</td><td>{r["r2"]:.3f}</td><td>{r["selectivity"]:.3f}</td>'
    f'<td>{r["candidate_quality"]:.3f}</td><td>{r["best_quality"]:.3f}</td>'
    f'<td>{r["icc"]:.3f}（{"通过" if r["icc_passed"] else "未通过"}）</td>'
    f'<td>{sum(x["tokens"] for x in rows[:i+1])/10000:.0f}万</td>'
    f'<td>{sum(x["duration_ms"] for x in rows[:i+1])/3600000:.2f}小时</td></tr>'
    for i, r in enumerate(rows)
)}
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[curve] 已写入 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    cp = Path(args.checkpoint)
    out = Path(args.out) if args.out else cp.with_name(cp.stem + "_iteration_curve.html")
    render(cp, out)


if __name__ == "__main__":
    main()
