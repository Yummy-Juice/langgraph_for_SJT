"""Dose-response sensitivity experiment for saved virtual responses.

Starting from an independently permuted response matrix, this experiment
restores a nested proportion of intact items.  It tests whether reliability,
convergent association and discriminant separation respond gradually to known
changes in person-level signal.  It does not call a model or alter source data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats

from .negative_controls import METHODS, _number, _sha256, calculate_metrics, load_source
from .storage import write_csv, write_json


DEFAULT_LEVELS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
METRICS = (
    "cronbach_alpha",
    "rho_E",
    "max_abs_rho_non_target",
    "discriminant_gap",
)
SCHEMA_VERSION = 1
FORMULA_VERSION = "legacy-abc-signal-restoration-v1"


def validate_levels(levels: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in levels)
    if len(values) < 4 or any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("质量水平至少需要4个有限值，且必须位于0至1之间")
    if tuple(sorted(set(values))) != values or values[0] != 0 or values[-1] != 1:
        raise ValueError("质量水平必须严格递增、不得重复，并同时包含0和1")
    return values


def _item_count(quality: float, total: int) -> int:
    if quality <= 0:
        return 0
    if quality >= 1:
        return total
    return int(round(quality * total))


def _derived_metrics(matrix: np.ndarray, neo: np.ndarray) -> dict[str, Any]:
    result = calculate_metrics(matrix, neo)
    target = _number(result.get("rho_E"))
    non_target = _number(result.get("max_abs_rho_non_target"))
    result["discriminant_gap"] = (
        target - non_target if target is not None and non_target is not None else None
    )
    return result


def generate_gradient_rows(
    source: Mapping[str, Any],
    *,
    replications: int,
    seed: int,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> list[dict[str, Any]]:
    levels = validate_levels(levels)
    if replications < 1:
        raise ValueError("replications必须为正整数")
    neo = np.asarray(source["neo"], dtype=float)
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        data = source["methods"][method]
        original = np.asarray(data["matrix"], dtype=float)
        item_ids = list(data["item_ids"])
        item_total = original.shape[1]
        baseline = _derived_metrics(original, neo)
        child_seeds = np.random.SeedSequence([seed, method_index, 731]).spawn(replications)
        for replicate, child_seed in enumerate(child_seeds, 1):
            rng = np.random.default_rng(child_seed)
            restoration_order = rng.permutation(item_total)
            disrupted = np.column_stack([
                original[rng.permutation(original.shape[0]), column]
                for column in range(item_total)
            ])
            for quality in levels:
                intact_count = _item_count(quality, item_total)
                current = disrupted.copy()
                intact_indices = restoration_order[:intact_count]
                if intact_count:
                    current[:, intact_indices] = original[:, intact_indices]
                metrics = baseline if intact_count == item_total else _derived_metrics(current, neo)
                rows.append({
                    "method": method,
                    "replicate": replicate,
                    "seed": seed,
                    "quality_level": quality,
                    "intact_item_count": intact_count,
                    "corrupted_item_count": item_total - intact_count,
                    "intact_item_ids": [item_ids[index] for index in sorted(intact_indices.tolist())],
                    **metrics,
                })
    return rows


def _quantile(values: Sequence[float], probability: float) -> float | None:
    clean = [float(value) for value in values if _number(value) is not None]
    return float(np.quantile(clean, probability)) if clean else None


def summarize_gradient(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for (method, quality), group in frame.groupby(["method", "quality_level"], sort=True):
        row: dict[str, Any] = {
            "method": method,
            "quality_level": float(quality),
            "intact_item_count": int(group["intact_item_count"].iloc[0]),
            "corrupted_item_count": int(group["corrupted_item_count"].iloc[0]),
            "replications": int(group.shape[0]),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_median"] = float(np.median(values)) if values else None
            row[f"{metric}_q025"] = _quantile(values, 0.025)
            row[f"{metric}_q975"] = _quantile(values, 0.975)
            row[f"{metric}_estimable"] = len(values)
        output.append(row)
    return output


def trend_checks(
    summary: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary)
    replication_frame = pd.DataFrame(rows) if rows is not None else pd.DataFrame()
    checks: list[dict[str, Any]] = []
    for method in METHODS:
        subset = frame[frame["method"] == method].sort_values("quality_level")
        quality = subset["quality_level"].to_numpy(dtype=float)
        for metric in ("cronbach_alpha", "rho_E", "discriminant_gap"):
            values = pd.to_numeric(subset[f"{metric}_mean"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            trend = (
                _number(stats.spearmanr(quality[finite], values[finite]).statistic)
                if finite.sum() >= 3 else None
            )
            differences = np.diff(values)
            estimable_differences = differences[np.isfinite(differences)]
            adjacent_fraction = (
                float(np.mean(estimable_differences > 0)) if estimable_differences.size else None
            )
            endpoint_change = (
                float(values[-1] - values[0]) if finite[0] and finite[-1] else None
            )
            upper_mean = upper_q025 = upper_q975 = None
            upper_resolved = False
            if not replication_frame.empty:
                metric_rows = replication_frame[replication_frame["method"] == method]
                pivot = metric_rows.pivot(index="replicate", columns="quality_level", values=metric)
                upper_level = float(quality[-1])
                penultimate_level = float(quality[-2])
                if upper_level in pivot and penultimate_level in pivot:
                    upper_differences = pd.to_numeric(
                        pivot[upper_level] - pivot[penultimate_level], errors="coerce"
                    ).dropna().to_numpy(dtype=float)
                    if upper_differences.size:
                        upper_mean = float(np.mean(upper_differences))
                        upper_q025 = float(np.quantile(upper_differences, 0.025))
                        upper_q975 = float(np.quantile(upper_differences, 0.975))
                        upper_resolved = upper_q025 > 0
            if metric in {"cronbach_alpha", "rho_E"}:
                sensitive = (
                    trend is not None and trend >= 0.90
                    and adjacent_fraction is not None and adjacent_fraction >= 0.80
                    and endpoint_change is not None and endpoint_change >= 0.50
                )
            else:
                sensitive = (
                    trend is not None and trend >= 0.80
                    and adjacent_fraction is not None and adjacent_fraction >= 0.60
                    and endpoint_change is not None and endpoint_change >= 0.05
                )
            checks.append({
                "method": method,
                "metric": metric,
                "trend_spearman_rho": trend,
                "adjacent_increase_fraction": adjacent_fraction,
                "endpoint_change": endpoint_change,
                "sensitive": sensitive,
                "upper_step_from": float(quality[-2]),
                "upper_step_to": float(quality[-1]),
                "upper_step_change_mean": upper_mean,
                "upper_step_change_q025": upper_q025,
                "upper_step_change_q975": upper_q975,
                "upper_range_sensitive": upper_resolved,
                "criterion": (
                    "rho>=.90、相邻上升比例>=.80、端点差>=.50"
                    if metric in {"cronbach_alpha", "rho_E"}
                    else "rho>=.80、相邻上升比例>=.60、端点差>=.05"
                ),
            })
    return checks


def diagnose(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_metric = {
        metric: all(bool(row["sensitive"]) for row in checks if row["metric"] == metric)
        for metric in ("cronbach_alpha", "rho_E", "discriminant_gap")
    }
    upper_by_metric = {
        metric: all(bool(row.get("upper_range_sensitive")) for row in checks if row["metric"] == metric)
        for metric in ("cronbach_alpha", "rho_E", "discriminant_gap")
    }
    if all(by_metric.values()) and all(upper_by_metric.values()):
        verdict = "graded_sensitivity_confirmed"
        interpretation = (
            "三类指标均随完整人格信号恢复而稳定上升，并能分辨最高两个质量水平。"
        )
    elif all(by_metric.values()) and not upper_by_metric["discriminant_gap"]:
        verdict = "graded_sensitivity_with_specificity_ceiling"
        interpretation = (
            "三类指标的整体均值随信号恢复而上升，但高质量区间的区分效度差值不能稳定分开；"
            "评价器存在构念特异性天花板。"
        )
    elif by_metric["rho_E"] and not by_metric["discriminant_gap"]:
        verdict = "general_signal_only"
        interpretation = (
            "评价器能识别一般人格信号恢复，但不能稳定识别目标构念相对非目标构念的特异性改善。"
        )
    else:
        verdict = "fine_grained_sensitivity_not_confirmed"
        interpretation = (
            "至少一个关键指标没有随完整信号恢复而稳定上升；当前评价器不宜直接作为逐轮返修目标。"
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "metric_sensitivity": by_metric,
        "upper_range_sensitivity": upper_by_metric,
        "threshold_note": "判定阈值是本次计算机敏感性诊断规则，不是通用心理测量合格线。",
    }


def _fmt(value: Any) -> str:
    numeric = _number(value)
    return f"{numeric:.3f}" if numeric is not None else "不可估计"


def _line_chart(summary: Sequence[Mapping[str, Any]], metric: str, title: str) -> str:
    frame = pd.DataFrame(summary)
    colors = {"A": "#2563eb", "B": "#059669", "C": "#dc2626"}
    width, height = 760, 360
    left, right, top, bottom = 62, 25, 35, 55
    values = pd.to_numeric(frame[f"{metric}_mean"], errors="coerce")
    low = pd.to_numeric(frame[f"{metric}_q025"], errors="coerce")
    high = pd.to_numeric(frame[f"{metric}_q975"], errors="coerce")
    finite = pd.concat([values, low, high]).dropna().to_numpy(dtype=float)
    y_min = float(np.min(finite)) if finite.size else 0.0
    y_max = float(np.max(finite)) if finite.size else 1.0
    margin = max((y_max - y_min) * 0.12, 0.04)
    y_min, y_max = y_min - margin, y_max + margin
    if y_max <= y_min:
        y_max = y_min + 1

    def x(value: float) -> float:
        return left + value * (width - left - right)

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    for tick in np.linspace(y_min, y_max, 5):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
    for tick in (0, 0.125, 0.25, 0.5, 0.75, 1):
        xx = x(tick)
        parts.append(f'<text x="{xx:.1f}" y="{height-25}" text-anchor="middle" font-size="11">{tick*100:.1f}%</text>')
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#374151"/>')
    for method in METHODS:
        group = frame[frame["method"] == method].sort_values("quality_level")
        points = []
        for _, row in group.iterrows():
            value = _number(row[f"{metric}_mean"])
            if value is not None:
                points.append((x(float(row["quality_level"])), y(value)))
        if points:
            encoded = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
            parts.append(f'<polyline points="{encoded}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
            for xx, yy in points:
                parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.5" fill="{colors[method]}"/>')
    parts.append(f'<text x="{width/2}" y="{height-5}" text-anchor="middle" font-size="12">完整题目比例（人工信号恢复水平）</text>')
    for index, method in enumerate(METHODS):
        xx = width - 190 + index * 58
        parts.append(f'<line x1="{xx}" x2="{xx+18}" y1="18" y2="18" stroke="{colors[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{xx+23}" y="22" font-size="12">{method}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_report(
    path: Path,
    *,
    source: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
    replications: int,
    seed: int,
) -> None:
    table_rows = []
    for row in summary:
        table_rows.append(
            "<tr>"
            f"<td>{row['method']}</td><td>{float(row['quality_level'])*100:.1f}%</td>"
            f"<td>{row['intact_item_count']}</td>"
            f"<td>{_fmt(row['cronbach_alpha_mean'])}</td>"
            f"<td>{_fmt(row['rho_E_mean'])}</td>"
            f"<td>{_fmt(row['max_abs_rho_non_target_mean'])}</td>"
            f"<td>{_fmt(row['discriminant_gap_mean'])}</td>"
            "</tr>"
        )
    check_rows = "".join(
        "<tr>"
        f"<td>{row['method']}</td><td>{escape(str(row['metric']))}</td>"
        f"<td>{_fmt(row['trend_spearman_rho'])}</td>"
        f"<td>{_fmt(row['adjacent_increase_fraction'])}</td>"
        f"<td>{_fmt(row['endpoint_change'])}</td>"
        f"<td>{'通过' if row['sensitive'] else '未通过'}</td>"
        f"<td>{_fmt(row.get('upper_step_change_mean'))}</td>"
        f"<td>[{_fmt(row.get('upper_step_change_q025'))}, {_fmt(row.get('upper_step_change_q975'))}]</td>"
        f"<td>{'通过' if row.get('upper_range_sensitive') else '未通过'}</td>"
        "</tr>"
        for row in checks
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>虚拟被试质量梯度敏感性实验</title>
<style>body{{max-width:1180px;margin:32px auto;padding:0 20px;font:15px/1.65 sans-serif;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #d1d5db;padding:7px;text-align:center}}th{{background:#e8eef8}}code{{background:#f3f4f6;padding:2px 4px}}.lead{{font-size:17px}}.note{{background:#fff7ed;border-left:4px solid #f97316;padding:12px}}svg{{width:100%;height:auto;border:1px solid #e5e7eb;margin:8px 0 24px}}</style>
<h1>虚拟被试质量梯度敏感性实验</h1>
<p class="lead">结论：{escape(str(diagnosis['interpretation']))}</p>
<p>来源：<code>{escape(str(source['root']))}</code>；被试={len(source['subject_ids'])}；每个水平重复={replications}；seed={seed}；模型调用=0。</p>
<h2>信度α</h2>{_line_chart(summary, 'cronbach_alpha', '信度alpha质量梯度')}
<h2>汇聚效度：SJT与NEO-E相关</h2>{_line_chart(summary, 'rho_E', '汇聚效度质量梯度')}
<h2>区分效度差值</h2><p><code>Δ = rho_E - max(|rho_N|, |rho_O|, |rho_A|, |rho_C|)</code>，越大表示目标相关相对非目标相关更突出。</p>{_line_chart(summary, 'discriminant_gap', '区分效度差值质量梯度')}
<h2>各质量水平汇总</h2>
<table><thead><tr><th>方法</th><th>完整比例</th><th>完整题数</th><th>α</th><th>rho E</th><th>最大非目标|rho|</th><th>区分差值Δ</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>趋势检查</h2>
<table><thead><tr><th>方法</th><th>指标</th><th>趋势rho</th><th>相邻上升比例</th><th>端点变化</th><th>整体梯度</th><th>75%→100%均值变化</th><th>配对变化95%区间</th><th>高区间可分</th></tr></thead><tbody>{check_rows}</tbody></table>
<p class="note"><b>解释边界：</b>本实验恢复的是被试—题目对应关系中的人工人格信号，不是真实的题目返修。因此，曲线递增只能证明当前指标能识别已知信号强度，不能证明虚拟被试等同真人，也不能把曲线直接当作C方法的实际迭代结果。</p>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_sensitivity_curve(
    evaluation_root: str | Path,
    *,
    replications: int = 500,
    seed: int = 20260913,
    levels: Sequence[float] = DEFAULT_LEVELS,
    output_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    levels = validate_levels(levels)
    if replications < 100 or replications > 10000:
        raise ValueError("replications必须在100至10000之间")
    source = load_source(evaluation_root)
    if output_root is None:
        output = source["root"] / "diagnostics" / (
            "sensitivity_curve_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        )
    else:
        output = Path(output_root).resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("指定输出目录必须是空目录，避免覆盖既有结果")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    write_json(output / "config.json", {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "source": str(source["root"]),
        "replications": replications,
        "seed": seed,
        "quality_levels": list(levels),
        "degradation": "independent within-item participant permutation with nested item restoration",
        "model_calls": 0,
        "source_hashes": source["source_hashes"],
        "started_at": started.isoformat(),
    })
    rows = generate_gradient_rows(source, replications=replications, seed=seed, levels=levels)
    summary = summarize_gradient(rows)
    checks = trend_checks(summary, rows)
    diagnosis = diagnose(checks)
    changed_sources = [
        relative for relative, expected in source["source_hashes"].items()
        if _sha256(source["root"] / relative) != expected
    ]
    if changed_sources:
        raise RuntimeError("实验过程中来源文件发生变化：" + "；".join(changed_sources))
    write_csv(output / "gradient_replications.csv", rows)
    write_csv(output / "gradient_summary.csv", summary)
    write_csv(output / "trend_checks.csv", checks)
    result = {
        "status": "complete",
        "verification_status": "ANALYZED",
        "source": str(source["root"]),
        "respondent_count": len(source["subject_ids"]),
        "replications_per_level": replications,
        "quality_levels": list(levels),
        "model_calls": 0,
        "source_integrity_verified": True,
        **diagnosis,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "result.json", result)
    render_report(
        output / "report.html", source=source, summary=summary, checks=checks,
        diagnosis=diagnosis, replications=replications, seed=seed,
    )
    return output, result


def _parse_levels(value: str) -> tuple[float, ...]:
    try:
        return validate_levels(tuple(float(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检验信效度指标能否识别渐进恢复的人格作答信号")
    parser.add_argument("--evaluation", type=Path, required=True, help="legacy_pool_abc结果目录")
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--levels", type=_parse_levels, default=DEFAULT_LEVELS)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, result = run_sensitivity_curve(
        args.evaluation, replications=args.replications, seed=args.seed,
        levels=args.levels, output_root=args.output,
    )
    print(f"敏感性实验输出：{output}")
    print(f"结论：{result['interpretation']}")
    print(f"报告：{output / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
