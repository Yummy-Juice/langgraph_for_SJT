"""Plot the primary A/B/C paired-bootstrap construct-isolation result."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC = "construct_isolation_hedges_g"
METHODS = ("A/round_01", "B/round_01", "C/round_02")
METHOD_LABELS = {
    "A/round_01": "A  Prompt only",
    "B/round_01": "B  Theory-based",
    "C/round_02": "C  Closed-loop",
}
METHOD_COLORS = {
    "A/round_01": "#777777",
    "B/round_01": "#4C78A8",
    "C/round_02": "#D86832",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def _as_float(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def _load(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates = pd.read_csv(input_dir / "method_metrics_ci.csv", encoding="utf-8-sig")
    differences = pd.read_csv(input_dir / "method_difference_ci.csv", encoding="utf-8-sig")

    estimates = estimates[
        (estimates["metric"] == METRIC) & estimates["method"].isin(METHODS)
    ].copy()
    estimates = _as_float(estimates, ("estimate", "ci_lower_95", "ci_upper_95"))
    estimates["method"] = pd.Categorical(estimates["method"], METHODS, ordered=True)
    estimates = estimates.sort_values("method")
    if estimates["method"].nunique() != len(METHODS):
        raise ValueError("A、B、C/final 的Hedges' g隔离度结果不完整")

    wanted_pairs = {
        ("A/round_01", "C/round_02"),
        ("B/round_01", "C/round_02"),
    }
    differences = differences[
        (differences["metric"] == METRIC)
        & differences.apply(
            lambda row: (row["method_a"], row["method_b"]) in wanted_pairs,
            axis=1,
        )
    ].copy()
    differences = _as_float(
        differences,
        ("difference_b_minus_a", "ci_lower_95", "ci_upper_95"),
    )
    differences["comparison"] = differences.apply(
        lambda row: "C − A" if row["method_a"] == "A/round_01" else "C − B",
        axis=1,
    )
    differences["comparison"] = pd.Categorical(
        differences["comparison"], ["C − A", "C − B"], ordered=True
    )
    differences = differences.sort_values("comparison")
    if len(differences) != 2:
        raise ValueError("缺少C−A或C−B的配对bootstrap差值")
    return estimates, differences


def _draw_interval(
    ax: plt.Axes,
    y: float,
    estimate: float,
    lower: float,
    upper: float,
    color: str,
) -> None:
    ax.errorbar(
        estimate,
        y,
        xerr=np.array([[estimate - lower], [upper - estimate]]),
        fmt="o",
        markersize=6,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        ecolor=color,
        elinewidth=2.0,
        capsize=3.5,
        capthick=1.2,
        zorder=3,
    )


def plot(input_dir: Path, output_stem: Path) -> None:
    _configure_style()
    estimates, differences = _load(input_dir)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.48},
    )

    ax = axes[0]
    y_positions = np.arange(len(estimates))[::-1]
    for y, (_, row) in zip(y_positions, estimates.iterrows()):
        method = str(row["method"])
        _draw_interval(
            ax,
            y,
            float(row["estimate"]),
            float(row["ci_lower_95"]),
            float(row["ci_upper_95"]),
            METHOD_COLORS[method],
        )
        ax.text(
            float(row["ci_upper_95"]) + 0.15,
            y,
            f'{float(row["estimate"]):.2f}',
            va="center",
            ha="left",
            fontsize=7.5,
            color=METHOD_COLORS[method],
        )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_LABELS[str(value)] for value in estimates["method"]])
    ax.set_xlabel("Construct isolation (Hedges’ g)")
    ax.set_title("Absolute construct isolation", loc="left", fontweight="bold")
    ax.set_xlim(-2.8, 6.2)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.16, 1.08, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    y_positions = np.arange(len(differences))[::-1]
    for y, (_, row) in zip(y_positions, differences.iterrows()):
        _draw_interval(
            ax,
            y,
            float(row["difference_b_minus_a"]),
            float(row["ci_lower_95"]),
            float(row["ci_upper_95"]),
            METHOD_COLORS["C/round_02"],
        )
        ax.text(
            float(row["ci_upper_95"]) + 0.12,
            y,
            f'{float(row["difference_b_minus_a"]):.2f}',
            va="center",
            ha="left",
            fontsize=7.5,
            color=METHOD_COLORS["C/round_02"],
        )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(differences["comparison"])
    ax.set_xlabel("Paired difference in Hedges’ g isolation")
    ax.set_title("Closed-loop advantage", loc="left", fontweight="bold")
    ax.set_xlim(-0.35, 7.1)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.16, 1.08, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.text(
        0.5,
        -0.02,
        "Points are estimates; error bars are 95% percentile CIs from 5,000 paired respondent bootstrap resamples (n = 100).",
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="绘制A/B/C的Hedges' g构念隔离度bootstrap结果")
    parser.add_argument("--input", type=Path, required=True, help="bootstrap CSV所在目录")
    parser.add_argument("--output", type=Path, required=True, help="输出文件路径（不含扩展名）")
    args = parser.parse_args(argv)
    plot(args.input.resolve(), args.output.resolve())
    print(f"图已输出：{args.output.resolve()}.png/.svg/.pdf/.tiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
