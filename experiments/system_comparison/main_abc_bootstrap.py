"""Paired-respondent bootstrap for the main A/B/C experiment.

Reuses the frozen evaluation responses; no model calls. Resamples the 100
matched virtual respondents with replacement, recomputes the form metrics
for every method on the SAME resample index (paired), and reports percentile
95% CIs for each metric and for every pairwise method difference.

Metrics (traditional psychometrics where possible):
  - cronbach_alpha           reliability
  - citc_mean                reliability (item-total)
  - convergent_spearman      validity (SJT total vs NEO-FFI target domain)
  - qualification_count      items passing all four gates
  - same_domain_vts_mean     system diagnostic (mean target_rho - same_domain_rho)
  - construct_isolation_hedges_g
                              external-criterion extreme-groups construct
                              specificity (target g - max absolute non-target g)

Run:
  python -X utf8 -m experiments.system_comparison.main_abc_bootstrap \
      --experiment experiment_data/exp_20260911_155055_92946884
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Gate thresholds, matching the main system's analysis manifest.
CITC_THRESHOLD = 0.2
TARGET_RHO_THRESHOLD = 0.3
SAME_DOMAIN_VTS_THRESHOLD = 0.1
CROSS_DOMAIN_VTS_THRESHOLD = 0.2
EXTREME_GROUP_FRACTION = 1.0 / 3.0

DEFAULT_METHODS = ("A/round_01", "B/round_01", "C/round_02", "C/round_03", "C/round_04")
TARGET_DOMAIN = "E"  # criterion NEO-FFI domain for extraversion_gregariousness


def _read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame.set_index("matched_subject_id")


def _spearman_cols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of every column of X with the vector y."""
    rx = stats.rankdata(X, axis=0).astype(float)
    ry = stats.rankdata(y).astype(float)
    rx -= rx.mean(axis=0)
    ry -= ry.mean()
    numerator = rx.T @ ry
    denominator = np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum())
    return np.divide(numerator, denominator, out=np.full(X.shape[1], np.nan), where=denominator > 0)


def _alpha(matrix: np.ndarray) -> float:
    k = matrix.shape[1]
    item_variance = matrix.var(axis=0, ddof=1).sum()
    total_variance = matrix.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return np.nan
    return k / (k - 1) * (1 - item_variance / total_variance)


def _citc(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum(axis=1)
    rest = total[:, None] - matrix
    centered = matrix - matrix.mean(axis=0)
    rest_centered = rest - rest.mean(axis=0)
    numerator = (centered * rest_centered).sum(axis=0)
    denominator = np.sqrt((centered ** 2).sum(axis=0) * (rest_centered ** 2).sum(axis=0))
    return np.divide(numerator, denominator, out=np.full(matrix.shape[1], np.nan), where=denominator > 0)


def _hedges_g_extreme_effect(form: np.ndarray, preset: np.ndarray) -> float:
    """Return Hedges' g for externally defined upper/lower thirds.

    The grouping variable is the externally supplied construct score.  The
    effect denominator is the pooled within-group sample SD, so the same
    estimator can be applied to virtual respondents and human respondents.
    """
    low_threshold = np.quantile(preset, EXTREME_GROUP_FRACTION)
    high_threshold = np.quantile(preset, 1.0 - EXTREME_GROUP_FRACTION)
    if not np.isfinite(low_threshold) or not np.isfinite(high_threshold):
        return np.nan
    if low_threshold >= high_threshold:
        return np.nan

    low_scores = form[preset <= low_threshold]
    high_scores = form[preset >= high_threshold]
    low_n = low_scores.size
    high_n = high_scores.size
    if low_n < 2 or high_n < 2:
        return np.nan

    degrees_of_freedom = low_n + high_n - 2
    pooled_variance = (
        (low_n - 1) * low_scores.var(ddof=1)
        + (high_n - 1) * high_scores.var(ddof=1)
    ) / degrees_of_freedom
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    cohen_d = (high_scores.mean() - low_scores.mean()) / np.sqrt(pooled_variance)
    # Standard small-sample approximation to Hedges' correction factor J.
    correction = 1.0 - 3.0 / (4.0 * (low_n + high_n) - 9.0)
    return float(correction * cohen_d)


def load_method(experiment: Path, key: str) -> dict:
    base = experiment / key / "evaluation" / "score_matrices"
    if not base.is_dir():
        raise FileNotFoundError(f"缺少得分矩阵：{base}")
    target = _read_matrix(base / "target.csv")
    same = _read_matrix(next(p for p in base.glob("same_domain__*.csv")))
    cross = _read_matrix(next(p for p in base.glob("cross_domain__*.csv")))
    sample = json.loads((experiment / "participants" / "evaluation.json").read_text(encoding="utf-8"))
    by_condition: dict[str, dict[str, float]] = {}
    for row in sample["respondents"]:
        value = next(iter(row["score_values"].values()))
        by_condition.setdefault(row["condition_id"], {})[row["matched_subject_id"]] = float(value)
    return {"target": target, "same": same, "cross": cross,
            "preset": by_condition}


def align(method: dict, order: list[str], neo: pd.Series) -> dict:
    target = method["target"].reindex(order)
    same = method["same"].reindex(order)
    cross = method["cross"].reindex(order)
    if target.isna().any().any() or same.isna().any().any() or cross.isna().any().any():
        raise ValueError("得分矩阵与被试顺序无法对齐")
    preset_target = np.array([method["preset"]["target"][m] for m in order])
    same_cond = next(c for c in method["preset"] if c.startswith("same_domain"))
    cross_cond = next(c for c in method["preset"] if c.startswith("cross_domain"))
    preset_same = np.array([method["preset"][same_cond][m] for m in order])
    preset_cross = np.array([method["preset"][cross_cond][m] for m in order])
    return {"T": target.to_numpy(float), "S": same.to_numpy(float), "C": cross.to_numpy(float),
            "pt": preset_target, "ps": preset_same, "pc": preset_cross,
            "neo": neo.reindex(order).to_numpy(float)}


def metrics_for_draw(d: dict, idx: np.ndarray) -> dict:
    T, S, C = d["T"][idx], d["S"][idx], d["C"][idx]
    pt, ps, pc, neo = d["pt"][idx], d["ps"][idx], d["pc"][idx], d["neo"][idx]

    alpha = _alpha(T)
    citc = _citc(T)
    sjt_total = T.mean(axis=1)
    convergent = _spearman_cols(sjt_total[:, None], neo)[0] if np.std(neo) > 0 else np.nan

    target_rho = _spearman_cols(T, pt)
    same_rho = _spearman_cols(S, ps)
    cross_rho = _spearman_cols(C, pc)
    same_vts = target_rho - np.nanmax(np.vstack([same_rho]), axis=0)
    cross_vts = target_rho - np.nanmax(np.vstack([cross_rho]), axis=0)

    passes = (
        (citc >= CITC_THRESHOLD)
        & (target_rho >= TARGET_RHO_THRESHOLD)
        & (same_vts >= SAME_DOMAIN_VTS_THRESHOLD)
        & (cross_vts >= CROSS_DOMAIN_VTS_THRESHOLD)
    )
    qualification = int(np.nansum(passes))

    form_target = T.mean(axis=1)
    t_effect = _hedges_g_extreme_effect(form_target, pt)
    l_same = _hedges_g_extreme_effect(S.mean(axis=1), ps)
    l_cross = _hedges_g_extreme_effect(C.mean(axis=1), pc)
    leakage = np.nanmax([abs(l_same), abs(l_cross)]) if np.isfinite([l_same, l_cross]).any() else np.nan
    isolation = t_effect - leakage if np.isfinite(leakage) else np.nan

    return {
        "cronbach_alpha": alpha,
        "citc_mean": float(np.nanmean(citc)),
        "convergent_spearman": convergent,
        "qualification_count": qualification,
        "same_domain_vts_mean": float(np.nanmean(same_vts)),
        "construct_isolation_hedges_g": isolation,
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


def run(experiment: Path, methods: tuple[str, ...], replications: int, seed: int, output: Path) -> dict:
    raw = {key: load_method(experiment, key) for key in methods}
    neo = _read_matrix(experiment / "reference" / "neo_ffi" / "scores.csv")[TARGET_DOMAIN]
    order = sorted(neo.index)
    aligned = {key: align(raw[key], order, neo) for key in methods}
    n = len(order)
    rng = np.random.default_rng(seed)

    metric_names = list(next(iter(aligned.values())) and metrics_for_draw(next(iter(aligned.values())), np.arange(n)).keys())
    draws = {key: {m: np.full(replications, np.nan) for m in metric_names} for key in methods}
    for rep in range(replications):
        idx = rng.integers(0, n, n)
        for key in methods:
            values = metrics_for_draw(aligned[key], idx)
            for m in metric_names:
                draws[key][m][rep] = values[m]
        if (rep + 1) % 1000 == 0:
            print(f"[bootstrap] {rep + 1}/{replications}", flush=True)

    estimate = {key: {m: metrics_for_draw(aligned[key], np.arange(n))[m] for m in metric_names} for key in methods}
    rows = []
    for key in methods:
        for m in metric_names:
            lo, hi = percentile_ci(draws[key][m])
            rows.append({"method": key, "metric": m, "estimate": estimate[key][m],
                         "ci_lower_95": lo, "ci_upper_95": hi})
    result = pd.DataFrame(rows)

    diff_rows = []
    for i, first in enumerate(methods):
        for second in methods[i + 1:]:
            for m in metric_names:
                diff = draws[second][m] - draws[first][m]
                lo, hi = percentile_ci(diff)
                diff_rows.append({"method_a": first, "method_b": second, "metric": m,
                                  "estimate_a": estimate[first][m], "estimate_b": estimate[second][m],
                                  "difference_b_minus_a": estimate[second][m] - estimate[first][m],
                                  "ci_lower_95": lo, "ci_upper_95": hi,
                                  "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))})
    differences = pd.DataFrame(diff_rows)

    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "method_metrics_ci.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(output / "method_difference_ci.csv", index=False, encoding="utf-8-sig")
    (output / "manifest.json").write_text(json.dumps({
        "methods": list(methods), "replications": replications, "seed": seed,
        "respondent_count": n, "target_domain": TARGET_DOMAIN,
        "metric_names": metric_names, "paired": True,
        "paired_interpretation": "同一次有放回抽样索引用于全部方法",
        "metric_definitions": {
            "construct_isolation_hedges_g": {
                "name": "基于Hedges' g的外部效标极端组构念隔离度",
                "grouping": "分别按目标、同域非目标、跨域非目标的外部构念分数取上三分之一和下三分之一",
                "effect": "Hedges' g = J × (高组均值 - 低组均值) / 高低组合并样本标准差",
                "small_sample_correction": "J = 1 - 3 / (4 × (n_high + n_low) - 9)",
                "isolation": "g_target - max(abs(g_same_domain), abs(g_cross_domain))",
                "human_data_analogue": "真人数据可用独立facet效标分数按相同规则分组并计算",
            },
            "same_domain_vts_mean": {
                "name": "同域VTS均值",
                "classification": "系统诊断指标，不作为传统心理测量效度指标",
            },
            "qualification_count": {
                "name": "四门槛通过题数",
                "classification": "系统开发指标，不作为传统心理测量效度指标",
            },
        },
        "limitation": "只反映当前100名冻结虚拟被试的抽样不确定性，不含重跑模型或更换模型seed的波动。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"result": result, "differences": differences}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="主实验A/B/C的配对bootstrap置信区间（离线，不调用模型）")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    output = args.output or experiment / "summary" / "abc_bootstrap_ci"
    run(experiment, tuple(args.methods), args.replications, args.seed, output)
    print(f"\n输出：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
