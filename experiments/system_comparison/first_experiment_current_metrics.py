"""Recompute the current whole-form metrics for the first A/B/C experiment.

This module is offline: it reads frozen response matrices and never calls a
language model.  The first experiment used NEO-FFI domain scores, so its
validity estimates must not be relabelled as later IPIP-NEO facet evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy import stats


EXTREME_GROUP_FRACTION = 1.0 / 3.0
EXTREME_GROUP_SIZE = 33
METHOD_PATHS = {
    "A": "A/round_01",
    "B": "B/round_01",
    "C": "C/round_02",
}
REPORTED_METRICS = (
    "cronbach_alpha",
    "icc_a1",
    "target_spearman",
    "delta_min",
    "target_hedges_g",
)
METRIC_LABELS = {
    "cronbach_alpha": "Cronbach’s α",
    "icc_a1": "ICC(A,1)",
    "target_spearman": "Target Spearman ρ",
    "delta_min": "Discriminant Δmin",
    "target_hedges_g": "Target Hedges’ g",
}
METHOD_COLORS = {"A": "#8A8F98", "B": "#4C78A8", "C": "#D86B3C"}


def _cronbach_alpha(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return float("nan")
    if not np.isfinite(values).all():
        return float("nan")
    total_variance = values.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_variance) or total_variance <= 0:
        return float("nan")
    item_variance = values.var(axis=0, ddof=1).sum()
    item_count = values.shape[1]
    return float(
        item_count / (item_count - 1.0)
        * (1.0 - item_variance / total_variance)
    )


def _icc_a1(first: np.ndarray, retest: np.ndarray) -> float:
    scores = np.column_stack(
        [
            np.asarray(first, dtype=float).mean(axis=1),
            np.asarray(retest, dtype=float).mean(axis=1),
        ]
    )
    if scores.shape[0] < 3 or not np.isfinite(scores).all():
        return float("nan")
    n_subjects, n_administrations = scores.shape
    grand_mean = float(scores.mean())
    subject_means = scores.mean(axis=1)
    administration_means = scores.mean(axis=0)
    ms_subject = n_administrations * float(
        np.square(subject_means - grand_mean).sum()
    ) / float(n_subjects - 1)
    ms_administration = n_subjects * float(
        np.square(administration_means - grand_mean).sum()
    ) / float(n_administrations - 1)
    residual = (
        scores
        - subject_means[:, None]
        - administration_means[None, :]
        + grand_mean
    )
    ms_error = float(np.square(residual).sum()) / float(
        (n_subjects - 1) * (n_administrations - 1)
    )
    denominator = (
        ms_subject
        + (n_administrations - 1) * ms_error
        + n_administrations
        * (ms_administration - ms_error)
        / n_subjects
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float((ms_subject - ms_error) / denominator)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.size < 3 or x.shape != y.shape:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _extreme_group_masks(
    grouping: np.ndarray,
    *,
    group_size: int = EXTREME_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Select fixed-size lower/upper groups with deterministic tie handling.

    The original quantile rule included every respondent tied at a cut point,
    which produced unequal groups for discrete virtual-reference scores.  This
    rule keeps the lowest/highest ``group_size`` rows.  When the boundary is
    tied, the frozen row order is used as the deterministic tie-breaker.  The
    same masks must be reused for every method in a comparison.
    """

    values = np.asarray(grouping, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("极端组效标必须是一维有限数值")
    if group_size < 2 or 2 * group_size > values.size:
        raise ValueError("极端组人数不足以形成不重叠的固定高低组")
    row_order = np.arange(values.size)
    ascending = np.lexsort((row_order, values))
    descending = np.lexsort((row_order, -values))
    low = np.zeros(values.size, dtype=bool)
    high = np.zeros(values.size, dtype=bool)
    low[ascending[:group_size]] = True
    high[descending[:group_size]] = True
    if np.any(low & high):
        raise ValueError("固定极端组发生重叠")
    return low, high


def _target_hedges_g(form_scores: np.ndarray, criterion: np.ndarray) -> float:
    form = np.asarray(form_scores, dtype=float)
    grouping = np.asarray(criterion, dtype=float)
    if form.shape != grouping.shape or form.size < 6:
        return float("nan")
    if not np.isfinite(form).all() or not np.isfinite(grouping).all():
        return float("nan")
    if np.ptp(grouping) <= 0:
        return float("nan")
    group_size = min(EXTREME_GROUP_SIZE, form.size // 3)
    if group_size < 2:
        return float("nan")
    try:
        low_mask, high_mask = _extreme_group_masks(
            grouping,
            group_size=group_size,
        )
    except ValueError:
        return float("nan")
    low = form[low_mask]
    high = form[high_mask]
    degrees_of_freedom = low.size + high.size - 2
    pooled_variance = (
        (low.size - 1) * low.var(ddof=1)
        + (high.size - 1) * high.var(ddof=1)
    ) / degrees_of_freedom
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return float("nan")
    cohen_d = (high.mean() - low.mean()) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (low.size + high.size) - 9.0)
    return float(correction * cohen_d)


def compute_form_metrics(
    first_scores: np.ndarray,
    retest_scores: np.ndarray,
    references: pd.DataFrame,
    *,
    target_column: str = "E",
) -> dict[str, Any]:
    """Compute five current metrics and retain non-target correlations."""

    first = np.asarray(first_scores, dtype=float)
    retest = np.asarray(retest_scores, dtype=float)
    if first.ndim != 2 or retest.ndim != 2 or first.shape != retest.shape:
        raise ValueError("初测与重测矩阵必须是形状相同的二维数组")
    if len(references) != first.shape[0]:
        raise ValueError("效标得分人数必须与作答矩阵一致")
    if target_column not in references:
        raise ValueError(f"效标数据缺少目标列：{target_column}")

    form_scores = first.mean(axis=1)
    target_spearman = _spearman(
        form_scores, references[target_column].to_numpy(dtype=float)
    )
    non_target = {
        str(column): _spearman(
            form_scores, references[column].to_numpy(dtype=float)
        )
        for column in references.columns
        if column != target_column
    }
    finite_non_target = [abs(value) for value in non_target.values() if np.isfinite(value)]
    delta_min = (
        float(target_spearman - max(finite_non_target))
        if np.isfinite(target_spearman) and finite_non_target
        else float("nan")
    )
    return {
        "cronbach_alpha": _cronbach_alpha(first),
        "icc_a1": _icc_a1(first, retest),
        "target_spearman": target_spearman,
        "delta_min": delta_min,
        "target_hedges_g": _target_hedges_g(
            form_scores, references[target_column].to_numpy(dtype=float)
        ),
        "non_target_spearman": non_target,
    }


def _read_score_matrix(path: Path, subject_ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "matched_subject_id" not in frame:
        raise ValueError(f"得分矩阵缺少 matched_subject_id：{path}")
    frame = frame.set_index("matched_subject_id")
    missing = [subject_id for subject_id in subject_ids if subject_id not in frame.index]
    if missing:
        raise ValueError(f"得分矩阵缺少 {len(missing)} 名效标被试：{path}")
    numeric = frame.reindex(subject_ids).apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"得分矩阵包含缺失值或非数值：{path}")
    return numeric.to_numpy(dtype=float)


def load_experiment(
    experiment: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, list[str]]:
    """Load and align the frozen final A/B/C forms to NEO-FFI respondents."""

    root = Path(experiment)
    reference_path = root / "reference" / "neo_ffi" / "scores.csv"
    references = pd.read_csv(reference_path, encoding="utf-8-sig")
    if "matched_subject_id" not in references:
        raise ValueError("NEO-FFI效标缺少 matched_subject_id")
    references = references.set_index("matched_subject_id")
    required_columns = ["E", "N", "O", "A", "C"]
    missing_columns = [column for column in required_columns if column not in references]
    if missing_columns:
        raise ValueError(f"NEO-FFI效标缺少列：{missing_columns}")
    references = references[required_columns].apply(pd.to_numeric, errors="coerce")
    if references.isna().any().any():
        raise ValueError("NEO-FFI效标包含缺失值或非数值")
    subject_ids = sorted(str(value) for value in references.index)
    references = references.reindex(subject_ids)

    methods: dict[str, dict[str, Any]] = {}
    for method, relative_path in METHOD_PATHS.items():
        score_dir = root / relative_path / "evaluation" / "score_matrices"
        first = _read_score_matrix(score_dir / "target.csv", subject_ids)
        retest = _read_score_matrix(score_dir / "target_retest_2.csv", subject_ids)
        if first.shape != retest.shape:
            raise ValueError(f"{method}初测与重测矩阵形状不一致")
        methods[method] = {
            "source_path": relative_path,
            "first": first,
            "retest": retest,
        }
    return methods, references.reset_index(drop=True), subject_ids


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(valid, 2.5)),
        float(np.percentile(valid, 97.5)),
    )


def paired_bootstrap(
    methods: dict[str, dict[str, Any]],
    references: pd.DataFrame,
    *,
    replications: int = 5000,
    seed: int = 20260920,
) -> pd.DataFrame:
    """Estimate five metrics with one shared respondent draw per replication."""

    if replications < 1:
        raise ValueError("bootstrap重复次数必须至少为1")
    respondent_count = len(references)
    if respondent_count < 6:
        raise ValueError("至少需要6名完整被试")
    rng = np.random.default_rng(seed)
    draws = {
        method: {
            metric: np.full(replications, np.nan, dtype=float)
            for metric in REPORTED_METRICS
        }
        for method in methods
    }
    estimates: dict[str, dict[str, Any]] = {}
    for method, data in methods.items():
        estimates[method] = compute_form_metrics(
            data["first"], data["retest"], references, target_column="E"
        )

    for replication in range(replications):
        index = rng.integers(0, respondent_count, respondent_count)
        sampled_references = references.iloc[index].reset_index(drop=True)
        for method, data in methods.items():
            values = compute_form_metrics(
                data["first"][index],
                data["retest"][index],
                sampled_references,
                target_column="E",
            )
            for metric in REPORTED_METRICS:
                draws[method][metric][replication] = float(values[metric])

    rows: list[dict[str, Any]] = []
    for method, data in methods.items():
        for metric in REPORTED_METRICS:
            lower, upper = _percentile_interval(draws[method][metric])
            rows.append(
                {
                    "method": method,
                    "source_path": data["source_path"],
                    "metric": metric,
                    "estimate": float(estimates[method][metric]),
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "respondent_count": respondent_count,
                    "criterion_scope": "NEO-FFI domain-level (E target; N/O/A/C non-target)",
                }
            )
    return pd.DataFrame(rows)


def _configure_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
        }
    )


def plot_metrics(results: pd.DataFrame, output_stem: Path) -> None:
    """Plot five native-scale panels using descriptive point estimates only."""

    required = {"method", "metric", "estimate"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"绘图数据缺少字段：{sorted(missing)}")
    expected_pairs = {
        (method, metric)
        for method in METHOD_PATHS
        for metric in REPORTED_METRICS
    }
    actual_pairs = set(zip(results["method"], results["metric"]))
    if not expected_pairs.issubset(actual_pairs):
        raise ValueError("绘图数据没有覆盖A/B/C的全部五项指标")

    _configure_figure_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8))
    axes_flat = axes.ravel()
    method_order = list(METHOD_PATHS)
    x_positions = np.arange(len(method_order))
    for panel_index, metric in enumerate(REPORTED_METRICS):
        ax = axes_flat[panel_index]
        panel = (
            results[results["metric"] == metric]
            .set_index("method")
            .reindex(method_order)
        )
        estimates = panel["estimate"].to_numpy(dtype=float)
        ax.plot(
            x_positions,
            estimates,
            color="#7A858A",
            linewidth=1.2,
            alpha=0.85,
            zorder=1,
        )
        for index, method in enumerate(method_order):
            color = METHOD_COLORS[method]
            ax.scatter(
                x_positions[index],
                estimates[index],
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                f"{estimates[index]:.3f}",
                (x_positions[index], estimates[index]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
            )
        if metric in {"delta_min", "target_hedges_g"}:
            ax.axhline(0, color="#5C6570", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_xticks(x_positions, method_order)
        ax.set_xlim(-0.35, len(method_order) - 0.65)
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E7EAED", linewidth=0.7, zorder=0)
        ax.tick_params(axis="x", length=0)
        y_ranges = {
            "cronbach_alpha": (0.90, 1.05, 0.02),
            "icc_a1": (0.90, 1.05, 0.02),
            "target_spearman": (0.80, 1.00, 0.05),
            "delta_min": (0.40, 0.80, 0.10),
            "target_hedges_g": (3.5, 5.5, 0.5),
        }
        y_min, y_max, y_step = y_ranges[metric]
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.text(
            -0.12,
            1.08,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
        )

    axes_flat[-1].axis("off")
    axes_flat[-1].text(
        0.02,
        0.85,
        "Methods",
        fontsize=9,
        fontweight="bold",
        transform=axes_flat[-1].transAxes,
    )
    method_descriptions = {
        "A": "Prompt-only generation",
        "B": "Theory-based assembly",
        "C": "Closed-loop final form (round 02)",
    }
    for index, method in enumerate(method_order):
        y = 0.66 - index * 0.18
        axes_flat[-1].scatter(
            [0.05], [y], s=42, color=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.8, transform=axes_flat[-1].transAxes,
        )
        axes_flat[-1].text(
            0.12, y, f"{method}  {method_descriptions[method]}",
            va="center", fontsize=8, transform=axes_flat[-1].transAxes,
        )
    axes_flat[-1].text(
        0.02,
        0.07,
        "Descriptive point estimates\nFrozen evaluation sample: n = 100\nSampling uncertainty not displayed",
        fontsize=7.2,
        color="#4E5963",
        linespacing=1.45,
        transform=axes_flat[-1].transAxes,
    )
    fig.suptitle(
        "First A/B/C experiment: current whole-form metrics",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.012,
        "Frozen NEO-FFI domain criterion (E target; N/O/A/C non-target); this is not the later IPIP-NEO facet protocol.",
        ha="left",
        fontsize=7.2,
        color="#4E5963",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.11, wspace=0.34, hspace=0.42)
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    experiment: Path,
    output_dir: Path,
    *,
    replications: int = 5000,
    seed: int = 20260920,
) -> pd.DataFrame:
    """Run the complete offline recomputation and export traceable outputs."""

    methods, references, subject_ids = load_experiment(Path(experiment))
    results = paired_bootstrap(
        methods,
        references,
        replications=replications,
        seed=seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "first_experiment_current_metrics"
    results.to_csv(
        stem.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )
    plot_metrics(results, stem)
    manifest = {
        "experiment": str(Path(experiment).resolve()),
        "methods": METHOD_PATHS,
        "metrics": list(REPORTED_METRICS),
        "respondent_count": len(subject_ids),
        "bootstrap_replications": int(replications),
        "seed": int(seed),
        "paired": True,
        "criterion_scope": "NEO-FFI domain-level",
        "criterion_mapping": {
            "target": "E",
            "non_target": ["N", "O", "A", "C"],
        },
        "formulas": {
            "cronbach_alpha": "k/(k-1) * (1 - sum(item variances)/total-score variance)",
            "icc_a1": "two-way absolute-agreement single-measure ICC(A,1)",
            "target_spearman": "Spearman(SJT total, NEO-FFI E)",
            "delta_min": "target Spearman - max(abs(non-target Spearman N/O/A/C))",
            "target_hedges_g": "J * (fixed 33-person high-group SJT mean - fixed 33-person low-group SJT mean) / pooled within-group SD",
        },
        "extreme_group_rule": {
            "group_size": EXTREME_GROUP_SIZE,
            "tie_break": "stable frozen respondent row order",
            "low_group": "lowest 33 criterion scores",
            "high_group": "highest 33 criterion scores",
            "middle_group": "remaining respondents excluded from Hedges g",
        },
        "limitation": (
            "This recomputes the first experiment with its frozen NEO-FFI "
            "domain criterion. It is not the later five-facet IPIP-NEO protocol."
        ),
    }
    (output / "first_experiment_current_metrics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="用当前五项整卷指标离线重算第一次A/B/C实验"
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    results = run_analysis(
        args.experiment.resolve(),
        args.output.resolve(),
        replications=args.replications,
        seed=args.seed,
    )
    print(results.to_string(index=False))
    print(f"\n输出目录：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
