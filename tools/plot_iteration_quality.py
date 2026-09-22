"""Create a publication-ready whole-form iteration figure from a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.evaluation.form_metrics import (  # noqa: E402
    assess_form_plateau,
    form_quality_summary,
)


COLORS = {
    "best": "#5E3C99",
    "candidate": "#E66101",
    "recovery": "#C51B7D",
    "selectivity": "#2B8CBE",
    "tokens": "#59636E",
    "time": "#16817A",
    "grid": "#D9DEE5",
    "text": "#263238",
}


def _load_rows(checkpoint_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state = payload.get("state") if isinstance(payload, dict) else payload
    history = state.get("psychometric_iteration_history") or []
    plateau = assess_form_plateau(history)
    trajectory = {
        int(row.get("analysis_round") or 0): row
        for row in plateau.get("trajectory") or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    cumulative_tokens = 0
    cumulative_duration_ms = 0
    for entry in history:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("form_metrics") or {}
        quality = form_quality_summary(metrics)
        round_number = int(entry.get("analysis_round") or 0)
        usage = entry.get("token_usage") or {}
        cumulative_tokens += int(usage.get("total_tokens") or 0)
        cumulative_duration_ms += int(usage.get("duration_ms") or 0)
        quality_row = trajectory.get(round_number) or {}
        values = {
            "round": round_number,
            "recovery": quality.get("target_recovery_raw"),
            "selectivity": quality.get("construct_selectivity"),
            "candidate": quality.get("candidate_form_quality"),
            "best": quality_row.get("best_so_far_form_quality"),
            "icc": (quality.get("stability_gate") or {}).get("observed"),
            "icc_passed": (quality.get("stability_gate") or {}).get("passed"),
            "cumulative_tokens_m": cumulative_tokens / 1_000_000,
            "cumulative_hours": cumulative_duration_ms / 3_600_000,
        }
        if all(
            isinstance(values[key], (int, float)) and not isinstance(values[key], bool)
            for key in ("recovery", "selectivity", "candidate", "best", "icc")
        ):
            rows.append(values)
    if not rows:
        raise ValueError("检查点中没有可绘制的完整整卷迭代指标")
    return rows, plateau


def _label_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, dy: float) -> None:
    for x_value, y_value in zip(x, y):
        ax.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            color=color,
            fontsize=7.5,
            fontweight="semibold",
        )


def render(checkpoint_path: Path, output_prefix: Path) -> list[Path]:
    rows, plateau = _load_rows(checkpoint_path)
    rounds = np.asarray([row["round"] for row in rows], dtype=float)
    recovery = np.asarray([row["recovery"] for row in rows], dtype=float)
    selectivity = np.asarray([row["selectivity"] for row in rows], dtype=float)
    candidate = np.asarray([row["candidate"] for row in rows], dtype=float)
    best = np.asarray([row["best"] for row in rows], dtype=float)
    cumulative_tokens = np.asarray(
        [row["cumulative_tokens_m"] for row in rows], dtype=float
    )
    cumulative_hours = np.asarray(
        [row["cumulative_hours"] for row in rows], dtype=float
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.35, 1.0, 1.0],
        hspace=0.60,
        left=0.11,
        right=0.89,
        top=0.84,
        bottom=0.11,
    )
    ax_quality = fig.add_subplot(grid[0, 0])
    ax_diagnostics = fig.add_subplot(grid[1, 0], sharex=ax_quality)
    ax_cost = fig.add_subplot(grid[2, 0], sharex=ax_quality)

    # a. Hero panel: candidate versus retained best-so-far quality.
    ax_quality.plot(
        rounds,
        best,
        color=COLORS["best"],
        linewidth=2.7,
        marker="o",
        markersize=6,
        label="历史最优整卷质量（BFQ）",
        zorder=3,
    )
    ax_quality.plot(
        rounds,
        candidate,
        color=COLORS["candidate"],
        linewidth=2.0,
        linestyle="--",
        marker="D",
        markersize=5,
        label="本轮候选整卷质量（U）",
        zorder=2,
    )
    _label_points(ax_quality, rounds, best, COLORS["best"], 8)
    _label_points(ax_quality, rounds, candidate, COLORS["candidate"], -12)
    best_round = int(plateau.get("best_round") or rounds[np.argmax(best)])
    ax_quality.axvspan(
        best_round - 0.18,
        best_round + 0.18,
        color=COLORS["best"],
        alpha=0.08,
        linewidth=0,
    )
    ax_quality.annotate(
        f"第{best_round}轮成为当前最佳",
        (best_round, best[list(rounds).index(float(best_round))]),
        xytext=(28, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["best"], "lw": 0.9},
        color=COLORS["best"],
        fontsize=8,
    )
    ax_quality.set_ylabel("整卷质量")
    ax_quality.set_ylim(0.66, 0.745)
    ax_quality.legend(loc="lower left", ncol=2, columnspacing=1.4, handlelength=2.5)
    ax_quality.set_title("a  整卷优化结果", loc="left", fontweight="bold", pad=8)

    # b. Raw diagnostics are intentionally allowed to fluctuate.
    ax_diagnostics.plot(
        rounds,
        recovery,
        color=COLORS["recovery"],
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="目标恢复度 R²",
    )
    ax_diagnostics.plot(
        rounds,
        selectivity,
        color=COLORS["selectivity"],
        linewidth=2.0,
        marker="^",
        markersize=5.5,
        label="构念选择性 C",
    )
    _label_points(ax_diagnostics, rounds, recovery, COLORS["recovery"], 7)
    _label_points(ax_diagnostics, rounds, selectivity, COLORS["selectivity"], 7)
    ax_diagnostics.set_ylabel("原始指标")
    ax_diagnostics.set_ylim(0.50, 1.00)
    ax_diagnostics.legend(loc="center right", ncol=2, columnspacing=1.5)
    ax_diagnostics.set_title(
        "b  原始诊断指标（允许波动；ICC门槛3/3通过）",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    # c. Cumulative development cost.
    token_line = ax_cost.plot(
        rounds,
        cumulative_tokens,
        color=COLORS["tokens"],
        linewidth=2.1,
        marker="s",
        markersize=5,
        label="累计Token",
    )[0]
    ax_time = ax_cost.twinx()
    time_line = ax_time.plot(
        rounds,
        cumulative_hours,
        color=COLORS["time"],
        linewidth=2.1,
        linestyle="--",
        marker="o",
        markersize=5,
        label="累计模型耗时",
    )[0]
    for x_value, token_value, hour_value in zip(
        rounds, cumulative_tokens, cumulative_hours
    ):
        ax_cost.annotate(
            f"{token_value:.2f}M",
            (x_value, token_value),
            xytext=(-16, 9),
            textcoords="offset points",
            ha="center",
            color=COLORS["tokens"],
            fontsize=7.5,
        )
        ax_time.annotate(
            f"{hour_value:.2f}h",
            (x_value, hour_value),
            xytext=(16, 9),
            textcoords="offset points",
            ha="center",
            color=COLORS["time"],
            fontsize=7.5,
        )
    ax_cost.set_ylabel("累计Token（百万）", color=COLORS["tokens"])
    ax_time.set_ylabel("累计模型耗时（小时）", color=COLORS["time"])
    ax_time.spines["right"].set_visible(True)
    ax_time.spines["right"].set_color(COLORS["time"])
    ax_cost.legend(
        [token_line, time_line],
        ["累计Token", "累计模型耗时"],
        loc="upper left",
        ncol=2,
    )
    ax_cost.set_title("c  累计开发成本", loc="left", fontweight="bold", pad=8)
    ax_cost.set_xlabel("心理测量分析轮次")

    for axis in (ax_quality, ax_diagnostics, ax_cost):
        axis.set_xticks(rounds)
        axis.set_xticklabels([f"第{int(value)}轮" for value in rounds])
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        axis.tick_params(colors=COLORS["text"])

    fig.suptitle(
        "虚拟整卷质量迭代轨迹",
        x=0.11,
        y=0.965,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.11,
        0.925,
        "第二轮取得最佳质量；第三轮候选回落，历史最优保持不变，而开发成本继续增加",
        ha="left",
        fontsize=8.5,
        color="#5F6B73",
    )
    fig.text(
        0.11,
        0.035,
        "注：U = √(R×C)；BFQ为达到最小有效增量后保留的历史最佳值。指标仅用于虚拟开发期，不代表真人信效度。",
        ha="left",
        fontsize=7.2,
        color="#5F6B73",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(".png"),
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.checkpoint, args.out):
        print(path.resolve())


if __name__ == "__main__":
    main()
