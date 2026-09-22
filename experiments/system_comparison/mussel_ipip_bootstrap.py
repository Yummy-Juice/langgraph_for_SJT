"""Paired respondent bootstrap for the completed Mussel × IPIP experiment.

This module is deliberately offline: it resamples the frozen respondents and
never calls an LLM.  The same respondent draw is applied to both virtual
respondent conditions so confidence intervals for method differences preserve
the paired design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .mussel_ipip_reference import DIMENSION_TO_IPIP, FACET_CODES
from .storage import write_csv, write_json


CONDITIONS = ("legacy_historical", "embodied_probability")
DIMENSIONS = tuple(DIMENSION_TO_IPIP)
CONDITION_LABELS = {
    "legacy_historical": "分数侧历史方法",
    "embodied_probability": "具身概率方法",
}
FACET_LABELS = {
    "N4": "N4 自我意识",
    "E2": "E2 乐群性",
    "O5": "O5 观念开放",
    "A4": "A4 顺从",
    "C5": "C5 自律",
}
MUSSEL_HUMAN_ALPHA = {"N4": 0.73, "E2": 0.75, "O5": 0.70, "A4": 0.56, "C5": 0.55}
MUSSEL_HUMAN_CONVERGENT = {"N4": 0.60, "E2": 0.66, "O5": 0.70, "A4": 0.41, "C5": 0.52}
FORMULA_VERSION = "mussel-ipip-paired-participant-percentile-bootstrap-v1"


@dataclass(frozen=True)
class MusselIPIPBootstrapConfig:
    experiment: Path
    comparison_result: Path | None = None
    ipip_result: Path | None = None
    replications: int = 5000
    seed: int = 20260915
    output: Path | None = None

    def validate(self) -> None:
        if self.replications < 1000:
            raise ValueError("正式bootstrap至少需要1000次重抽样")


def _alpha(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        return math.nan
    item_variance = np.var(values, axis=0, ddof=1).sum()
    total_variance = np.var(values.sum(axis=1), ddof=1)
    if not np.isfinite(total_variance) or total_variance <= 0:
        return math.nan
    result = values.shape[1] / (values.shape[1] - 1) * (
        1 - item_variance / total_variance
    )
    return float(result) if np.isfinite(result) else math.nan


def _spearman_matrix(values: np.ndarray) -> np.ndarray:
    """Return a column-wise Spearman matrix, retaining NaN for constants."""

    ranked = stats.rankdata(values, axis=0, method="average")
    centered = ranked - ranked.mean(axis=0, keepdims=True)
    sums = np.sqrt(np.square(centered).sum(axis=0))
    denominator = np.outer(sums, sums)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = centered.T @ centered / denominator
    result[~np.isfinite(result)] = np.nan
    return result


def _interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None, None, 0
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper), int(len(finite))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}第{line_number}行不是有效JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            records.append(value)
    return records


def _pivot_complete(
    frame: pd.DataFrame,
    *,
    subject_ids: Sequence[str],
    item_count: int,
    label: str,
) -> np.ndarray:
    if frame.duplicated(["respondent_id", "item_id"]).any():
        raise ValueError(f"{label}存在重复的被试—题目记录")
    item_ids = sorted(frame["item_id"].astype(str).unique())
    if len(item_ids) != item_count:
        raise ValueError(f"{label}应有{item_count}题，实际为{len(item_ids)}题")
    matrix = (
        frame.assign(
            respondent_id=frame["respondent_id"].astype(str),
            item_id=frame["item_id"].astype(str),
        )
        .pivot(index="respondent_id", columns="item_id", values="score")
        .reindex(index=list(subject_ids), columns=item_ids)
    )
    if matrix.isna().any().any():
        raise ValueError(f"{label}不能与冻结被试完整对齐")
    return matrix.to_numpy(dtype=float)


def _load_condition(
    comparison: Path,
    ipip_result: Path,
    condition: str,
    subject_ids: Sequence[str],
) -> dict[str, Any]:
    condition_dir = comparison / condition
    scored = pd.read_csv(condition_dir / "scored_responses.csv", encoding="utf-8-sig")
    required_sjt = {"respondent_id", "item_id", "target_dimension_id", "score"}
    if not required_sjt <= set(scored.columns):
        raise ValueError(f"{condition}的Mussel逐题计分列不完整")

    sjt_items: dict[str, np.ndarray] = {}
    for dimension in DIMENSIONS:
        subset = scored[scored["target_dimension_id"] == dimension]
        sjt_items[dimension] = _pivot_complete(
            subset,
            subject_ids=subject_ids,
            item_count=22,
            label=f"{condition}/{dimension}/Mussel",
        )

    ipip_records = pd.DataFrame(
        _read_jsonl(ipip_result / condition / "responses.jsonl")
    )
    required_ipip = {"respondent_id", "item_id", "facet_code", "score"}
    if ipip_records.empty or not required_ipip <= set(ipip_records.columns):
        raise ValueError(f"{condition}的IPIP逐题计分列不完整")
    ipip_items: dict[str, np.ndarray] = {}
    for facet in FACET_CODES:
        subset = ipip_records[ipip_records["facet_code"] == facet]
        ipip_items[facet] = _pivot_complete(
            subset,
            subject_ids=subject_ids,
            item_count=10,
            label=f"{condition}/{facet}/IPIP",
        )

    sjt_totals = np.column_stack(
        [sjt_items[dimension].sum(axis=1) for dimension in DIMENSIONS]
    )
    ipip_totals = np.column_stack(
        [ipip_items[facet].sum(axis=1) for facet in FACET_CODES]
    )
    return {
        "sjt_items": sjt_items,
        "ipip_items": ipip_items,
        "sjt_totals": sjt_totals,
        "ipip_totals": ipip_totals,
    }


def _statistics_for_draw(
    data: Mapping[str, Any], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sjt_alpha = np.asarray(
        [_alpha(data["sjt_items"][dimension][indices]) for dimension in DIMENSIONS]
    )
    ipip_alpha = np.asarray(
        [_alpha(data["ipip_items"][facet][indices]) for facet in FACET_CODES]
    )
    joint = np.column_stack(
        (data["sjt_totals"][indices], data["ipip_totals"][indices])
    )
    matrix = _spearman_matrix(joint)
    correlations = matrix[: len(DIMENSIONS), len(DIMENSIONS) :]
    target = np.asarray(
        [
            correlations[index, FACET_CODES.index(DIMENSION_TO_IPIP[dimension])]
            for index, dimension in enumerate(DIMENSIONS)
        ]
    )
    maximum = np.asarray(
        [
            np.nanmax(
                np.abs(
                    np.delete(
                        correlations[index],
                        FACET_CODES.index(DIMENSION_TO_IPIP[dimension]),
                    )
                )
            )
            for index, dimension in enumerate(DIMENSIONS)
        ]
    )
    gap = target - maximum
    return sjt_alpha, ipip_alpha, correlations, maximum, gap


def _ci_row(
    *,
    estimate: float,
    replicates: np.ndarray,
    replications: int,
    **metadata: Any,
) -> dict[str, Any]:
    lower, upper, valid = _interval(replicates)
    return {
        **metadata,
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "valid_bootstraps": valid,
        "requested_bootstraps": replications,
        "formula_version": FORMULA_VERSION,
    }


def _render_report(
    output: Path,
    *,
    respondent_count: int,
    replications: int,
    seed: int,
    reliability_rows: Sequence[Mapping[str, Any]],
    construct_rows: Sequence[Mapping[str, Any]],
    difference_rows: Sequence[Mapping[str, Any]],
) -> Path:
    def f(value: Any) -> str:
        if value is None:
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return escape(str(value))
        return f"{number:.3f}" if math.isfinite(number) else "—"

    def ci(row: Mapping[str, Any]) -> str:
        return f"{f(row.get('estimate'))} [{f(row.get('ci_lower_95'))}, {f(row.get('ci_upper_95'))}]"

    alpha_rows = "".join(
        "<tr>"
        f"<td>{escape(CONDITION_LABELS[str(row['condition'])])}</td>"
        f"<td>{escape(str(row['instrument']))}</td>"
        f"<td>{escape(FACET_LABELS[str(row['facet_code'])])}</td>"
        f"<td>{ci(row)}</td>"
        f"<td>{f(row.get('mussel_human_reference'))}</td>"
        "</tr>"
        for row in reliability_rows
    )
    construct_html = "".join(
        "<tr>"
        f"<td>{escape(CONDITION_LABELS[str(row['condition'])])}</td>"
        f"<td>{escape(FACET_LABELS[str(row['facet_code'])])}</td>"
        f"<td>{ci(row)}</td>"
        f"<td>{f(row.get('mussel_human_convergent'))}</td>"
        f"<td>{f(row.get('max_non_target_estimate'))} "
        f"[{f(row.get('max_non_target_ci_lower_95'))}, {f(row.get('max_non_target_ci_upper_95'))}]</td>"
        f"<td>{f(row.get('gap_estimate'))} "
        f"[{f(row.get('gap_ci_lower_95'))}, {f(row.get('gap_ci_upper_95'))}]</td>"
        "</tr>"
        for row in construct_rows
    )
    differences = "".join(
        "<tr>"
        f"<td>{escape(FACET_LABELS[str(row['facet_code'])])}</td>"
        f"<td>{escape(str(row['metric']))}</td>"
        f"<td>{ci(row)}</td>"
        f"<td>{'是' if row.get('ci_excludes_zero') else '否'}</td>"
        "</tr>"
        for row in difference_rows
    )
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>Mussel × IPIP配对bootstrap</title><style>body{{font:15px/1.65 sans-serif;max-width:1280px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f1f4f8}}.note{{background:#fff8df;padding:12px}}code{{background:#f3f3f3;padding:2px 4px}}</style></head><body>
<h1>Mussel × IPIP配对bootstrap 95%置信区间</h1>
<p>冻结被试={respondent_count}；重抽样={replications}次；随机种子=<code>{seed}</code>；区间方法=百分位法。</p>
<div class='note'>每次按被试整行有放回重抽，并将同一抽样索引同时用于两种方法。区间只反映当前100名冻结被试的抽样不确定性，不包含重新调用模型、改变模型seed或更换persona样本造成的波动。Mussel真人值仅作点估计参照；IPIP与NEO-PI-R不是同一量表，区间重叠也不等于统计等效。</div>
<h2>信度</h2><table><thead><tr><th>方法</th><th>量表</th><th>facet</th><th>估计值 [95% CI]</th><th>Mussel真人参照</th></tr></thead><tbody>{alpha_rows}</tbody></table>
<h2>facet级汇聚与区分</h2><table><thead><tr><th>方法</th><th>facet</th><th>汇聚ρ [95% CI]</th><th>Mussel真人汇聚r</th><th>最大非目标|ρ| [95% CI]</th><th>区分差值 [95% CI]</th></tr></thead><tbody>{construct_html}</tbody></table>
<h2>具身减分数侧的配对差异</h2><table><thead><tr><th>facet</th><th>指标</th><th>Δ [95% CI]</th><th>CI排除0</th></tr></thead><tbody>{differences}</tbody></table>
<p>“CI排除0”只表示该差异在当前冻结数据的被试重抽样中稳定，不代表跨模型运行稳定，也不构成虚拟结果与真人等效的证据。</p>
</body></html>"""
    report = output / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


def run_mussel_ipip_bootstrap(
    config: MusselIPIPBootstrapConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    comparison = (
        config.comparison_result.resolve()
        if config.comparison_result is not None
        else experiment / "mussel_legacy_vs_embodied_100"
    )
    ipip_result = (
        config.ipip_result.resolve()
        if config.ipip_result is not None
        else comparison / "ipip_facet_reference"
    )
    output = (
        config.output.resolve()
        if config.output is not None
        else ipip_result / "bootstrap_ci"
    )
    manifest_path = comparison / "manifest.json"
    ipip_manifest_path = ipip_result / "manifest.json"
    if not manifest_path.is_file() or not ipip_manifest_path.is_file():
        raise FileNotFoundError("找不到已完成的Mussel或IPIP实验manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    ipip_manifest = json.loads(ipip_manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "complete" or ipip_manifest.get("status") != "complete":
        raise ValueError("Mussel与IPIP实验都必须完成后才能bootstrap")
    subject_ids = [str(value) for value in manifest.get("respondent_ids") or []]
    if len(subject_ids) != 100 or len(set(subject_ids)) != 100:
        raise ValueError("bootstrap必须绑定冻结且唯一的100名被试")
    if [str(value) for value in ipip_manifest.get("respondent_ids") or []] != subject_ids:
        raise ValueError("IPIP与Mussel冻结被试的身份或顺序不一致")

    datasets = {
        condition: _load_condition(comparison, ipip_result, condition, subject_ids)
        for condition in CONDITIONS
    }
    condition_count = len(CONDITIONS)
    facet_count = len(FACET_CODES)
    correlation_draws = np.full(
        (config.replications, condition_count, facet_count, facet_count), np.nan
    )
    sjt_alpha_draws = np.full(
        (config.replications, condition_count, facet_count), np.nan
    )
    ipip_alpha_draws = np.full_like(sjt_alpha_draws, np.nan)
    max_non_target_draws = np.full_like(sjt_alpha_draws, np.nan)
    gap_draws = np.full_like(sjt_alpha_draws, np.nan)

    full_indices = np.arange(len(subject_ids))
    points: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        condition: _statistics_for_draw(datasets[condition], full_indices)
        for condition in CONDITIONS
    }
    rng = np.random.default_rng(config.seed)
    progress_step = max(1, config.replications // 10)
    for replication in range(config.replications):
        indices = rng.integers(0, len(subject_ids), size=len(subject_ids))
        for condition_index, condition in enumerate(CONDITIONS):
            sjt_alpha, ipip_alpha, correlations, maximum, gap = _statistics_for_draw(
                datasets[condition], indices
            )
            sjt_alpha_draws[replication, condition_index] = sjt_alpha
            ipip_alpha_draws[replication, condition_index] = ipip_alpha
            correlation_draws[replication, condition_index] = correlations
            max_non_target_draws[replication, condition_index] = maximum
            gap_draws[replication, condition_index] = gap
        completed = replication + 1
        if completed == config.replications or completed % progress_step == 0:
            print(
                f"[bootstrap] {completed}/{config.replications} "
                f"({completed / config.replications:.0%})",
                flush=True,
            )

    reliability_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    construct_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        point_sjt_alpha, point_ipip_alpha, point_corr, point_maximum, point_gap = points[
            condition
        ]
        for facet_index, (dimension, facet) in enumerate(
            zip(DIMENSIONS, FACET_CODES)
        ):
            reliability_rows.append(
                _ci_row(
                    condition=condition,
                    instrument="Mussel SJT",
                    target_dimension_id=dimension,
                    facet_code=facet,
                    estimate=point_sjt_alpha[facet_index],
                    replicates=sjt_alpha_draws[:, condition_index, facet_index],
                    replications=config.replications,
                    mussel_human_reference=MUSSEL_HUMAN_ALPHA[facet],
                )
            )
            reliability_rows.append(
                _ci_row(
                    condition=condition,
                    instrument="IPIP facet",
                    target_dimension_id=dimension,
                    facet_code=facet,
                    estimate=point_ipip_alpha[facet_index],
                    replicates=ipip_alpha_draws[:, condition_index, facet_index],
                    replications=config.replications,
                    mussel_human_reference=None,
                )
            )
            for criterion_index, criterion_facet in enumerate(FACET_CODES):
                correlation_rows.append(
                    _ci_row(
                        condition=condition,
                        target_dimension_id=dimension,
                        target_ipip_facet=facet,
                        criterion_ipip_facet=criterion_facet,
                        is_target=criterion_facet == facet,
                        estimate=point_corr[facet_index, criterion_index],
                        replicates=correlation_draws[
                            :, condition_index, facet_index, criterion_index
                        ],
                        replications=config.replications,
                    )
                )
            max_lower, max_upper, max_valid = _interval(
                max_non_target_draws[:, condition_index, facet_index]
            )
            gap_lower, gap_upper, gap_valid = _interval(
                gap_draws[:, condition_index, facet_index]
            )
            target_index = FACET_CODES.index(facet)
            target_row = _ci_row(
                condition=condition,
                target_dimension_id=dimension,
                facet_code=facet,
                estimate=point_corr[facet_index, target_index],
                replicates=correlation_draws[
                    :, condition_index, facet_index, target_index
                ],
                replications=config.replications,
                mussel_human_convergent=MUSSEL_HUMAN_CONVERGENT[facet],
            )
            construct_rows.append(
                {
                    **target_row,
                    "max_non_target_estimate": float(point_maximum[facet_index]),
                    "max_non_target_ci_lower_95": max_lower,
                    "max_non_target_ci_upper_95": max_upper,
                    "max_non_target_valid_bootstraps": max_valid,
                    "gap_estimate": float(point_gap[facet_index]),
                    "gap_ci_lower_95": gap_lower,
                    "gap_ci_upper_95": gap_upper,
                    "gap_valid_bootstraps": gap_valid,
                }
            )

    old_index = CONDITIONS.index("legacy_historical")
    new_index = CONDITIONS.index("embodied_probability")
    for facet_index, facet in enumerate(FACET_CODES):
        dimension = DIMENSIONS[facet_index]
        target_index = FACET_CODES.index(facet)
        definitions = (
            (
                "SJT alpha",
                points["legacy_historical"][0][facet_index],
                points["embodied_probability"][0][facet_index],
                sjt_alpha_draws[:, new_index, facet_index]
                - sjt_alpha_draws[:, old_index, facet_index],
            ),
            (
                "target IPIP Spearman rho",
                points["legacy_historical"][2][facet_index, target_index],
                points["embodied_probability"][2][facet_index, target_index],
                correlation_draws[:, new_index, facet_index, target_index]
                - correlation_draws[:, old_index, facet_index, target_index],
            ),
            (
                "discriminant gap",
                points["legacy_historical"][4][facet_index],
                points["embodied_probability"][4][facet_index],
                gap_draws[:, new_index, facet_index]
                - gap_draws[:, old_index, facet_index],
            ),
        )
        for metric, old_estimate, new_estimate, draws in definitions:
            row = _ci_row(
                facet_code=facet,
                target_dimension_id=dimension,
                metric=metric,
                old_estimate=float(old_estimate),
                new_estimate=float(new_estimate),
                estimate=float(new_estimate - old_estimate),
                replicates=draws,
                replications=config.replications,
            )
            lower = row.get("ci_lower_95")
            upper = row.get("ci_upper_95")
            row["ci_excludes_zero"] = bool(
                lower is not None and upper is not None and (upper < 0 or lower > 0)
            )
            difference_rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {**asdict(config), "experiment": str(experiment), "comparison_result": str(comparison), "ipip_result": str(ipip_result), "output": str(output)})
    write_csv(output / "reliability_ci.csv", reliability_rows)
    write_csv(output / "correlation_ci.csv", correlation_rows)
    write_csv(output / "construct_ci.csv", construct_rows)
    write_csv(output / "method_difference_ci.csv", difference_rows)
    np.savez_compressed(
        output / "bootstrap_distributions.npz",
        conditions=np.asarray(CONDITIONS),
        dimensions=np.asarray(DIMENSIONS),
        facets=np.asarray(FACET_CODES),
        sjt_alpha=sjt_alpha_draws,
        ipip_alpha=ipip_alpha_draws,
        correlations=correlation_draws,
        max_non_target=max_non_target_draws,
        discriminant_gap=gap_draws,
    )
    warnings = []
    all_rows = [*reliability_rows, *correlation_rows, *construct_rows, *difference_rows]
    minimum_valid = min(int(row.get("valid_bootstraps") or 0) for row in all_rows)
    if minimum_valid < int(config.replications * 0.95):
        warnings.append(
            f"至少一个指标只有{minimum_valid}/{config.replications}次有效bootstrap"
        )
    result = {
        "status": "complete",
        "verification_status": "ANALYZED",
        "respondent_count": len(subject_ids),
        "replications": config.replications,
        "seed": config.seed,
        "paired_conditions": list(CONDITIONS),
        "minimum_valid_bootstraps": minimum_valid,
        "warnings": warnings,
        "interpretation_boundary": (
            "Percentile confidence intervals condition on the frozen model outputs; "
            "they do not measure model-run or persona-generation variability."
        ),
        "formula_version": FORMULA_VERSION,
    }
    report = _render_report(
        output,
        respondent_count=len(subject_ids),
        replications=config.replications,
        seed=config.seed,
        reliability_rows=reliability_rows,
        construct_rows=construct_rows,
        difference_rows=difference_rows,
    )
    result["report"] = str(report)
    write_json(output / "result.json", result)
    return output, result

