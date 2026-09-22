"""Offline scoring-sensitivity analysis for saved embodied probabilities.

The analysis never invokes a language model.  It holds the model-reported
choice probabilities and criterion scores fixed, then compares:

* the originally stored categorical draw;
* repeated categorical draws under a deterministic seed schedule;
* probability-weighted expected scores; and
* scores from the maximum-probability option.

This isolates uncertainty introduced by converting a four-option probability
vector into one observed option.  It does not test whether the probabilities
are calibrated to human behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .legacy_pool_abc import (
    NEO_DOMAINS,
    OPTION_IDS,
    _alpha,
    _sample_probability_choice,
    _spearman,
    _validate_choice_probabilities,
)
from .storage import write_json


METRIC_VERSION = "embodied-probability-scoring-sensitivity-v1"
METRIC_COLUMNS = (
    "cronbach_alpha",
    "convergent_rho_E",
    "discriminant_rho_N",
    "discriminant_rho_O",
    "discriminant_rho_A",
    "discriminant_rho_C",
    "max_abs_discriminant_rho",
    "convergent_minus_max_abs_discriminant",
)


@dataclass(frozen=True)
class ProbabilityResamplingConfig:
    experiment: Path
    evaluation_dirname: str = "legacy_summary_embodied_probability_abc"
    replications: int = 100
    seed: int = 20260913
    output: Path | None = None

    def validate(self) -> None:
        if not self.evaluation_dirname.strip():
            raise ValueError("evaluation_dirname不能为空")
        if isinstance(self.replications, bool) or self.replications < 2:
            raise ValueError("replications至少为2")
        if self.replications > 100_000:
            raise ValueError("replications不能超过100000")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed必须是整数")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            rows.append(value)
    return rows


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric_row(
    method: str,
    scoring_method: str,
    matrix: pd.DataFrame,
    criterion: pd.DataFrame,
    *,
    replication: int | None = None,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    totals = matrix.sum(axis=1).astype(float)
    correlations = {
        domain: _spearman(totals.tolist(), criterion[domain].astype(float).tolist())
        for domain in NEO_DOMAINS
    }
    available_discriminants = [
        abs(value)
        for domain, value in correlations.items()
        if domain != "E" and value is not None
    ]
    max_abs_discriminant = (
        max(available_discriminants)
        if len(available_discriminants) == len(NEO_DOMAINS) - 1
        else None
    )
    convergent = correlations["E"]
    validity_gap = (
        convergent - max_abs_discriminant
        if convergent is not None and max_abs_discriminant is not None
        else None
    )
    return {
        "method": method,
        "scoring_method": scoring_method,
        "replication": replication,
        "sampling_seed": sampling_seed,
        "respondent_count": int(matrix.shape[0]),
        "item_count": int(matrix.shape[1]),
        "cronbach_alpha": _alpha(matrix),
        "convergent_rho_E": convergent,
        "discriminant_rho_N": correlations["N"],
        "discriminant_rho_O": correlations["O"],
        "discriminant_rho_A": correlations["A"],
        "discriminant_rho_C": correlations["C"],
        "max_abs_discriminant_rho": max_abs_discriminant,
        "convergent_minus_max_abs_discriminant": validity_gap,
    }


def _validate_and_index_records(
    records: Sequence[Mapping[str, Any]],
    form: Mapping[str, Any],
    subject_ids: Sequence[str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Mapping[str, Any]]]:
    items = form.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("冻结问卷缺少items")
    item_by_id = {str(item["item_id"]): item for item in items}
    if len(item_by_id) != len(items):
        raise ValueError("冻结问卷题号重复")
    expected = {
        (str(respondent_id), item_id)
        for respondent_id in subject_ids
        for item_id in item_by_id
    }
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for source in records:
        row = dict(source)
        key = (str(row.get("respondent_id")), str(row.get("item_id")))
        if key not in expected:
            raise ValueError(f"概率作答包含未预期记录：{key}")
        if key in indexed:
            raise ValueError(f"概率作答重复：{key}")
        probabilities = row.get("choice_probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError(f"概率作答缺少choice_probabilities：{key}")
        normalized = _validate_choice_probabilities(
            {"choice_probabilities": probabilities}
        )["normalized"]
        selected = str(row.get("selected_option_id"))
        if selected not in OPTION_IDS:
            raise ValueError(f"原始作答选项无效：{key}/{selected}")
        scoring_key = item_by_id[key[1]].get("scoring_key")
        if not isinstance(scoring_key, Mapping) or set(scoring_key) != set(OPTION_IDS):
            raise ValueError(f"题目计分键不完整：{key[1]}")
        expected_stored_score = float(scoring_key[selected])
        if not math.isclose(
            float(row.get("score")), expected_stored_score, abs_tol=1e-12
        ):
            raise ValueError(f"原始选项与原始得分不一致：{key}")
        row["choice_probabilities"] = normalized
        indexed[key] = row
    missing = expected.difference(indexed)
    if missing or len(indexed) != len(expected):
        example = next(iter(missing), None)
        raise ValueError(
            f"概率作答不完整：应有{len(expected)}条，实际{len(indexed)}条；"
            f"示例缺失={example}"
        )
    return indexed, item_by_id


def _matrix_for_scoring(
    indexed: Mapping[tuple[str, str], Mapping[str, Any]],
    item_by_id: Mapping[str, Mapping[str, Any]],
    subject_ids: Sequence[str],
    scoring_method: str,
    *,
    sampling_seed: int | None = None,
) -> pd.DataFrame:
    item_ids = list(item_by_id)
    values = np.empty((len(subject_ids), len(item_ids)), dtype=float)
    for respondent_index, respondent_id in enumerate(subject_ids):
        for item_index, item_id in enumerate(item_ids):
            row = indexed[(str(respondent_id), item_id)]
            item = item_by_id[item_id]
            probabilities = row["choice_probabilities"]
            scoring_key = item["scoring_key"]
            if scoring_method == "current_sample":
                score = float(row["score"])
            elif scoring_method == "expected_score":
                score = sum(
                    float(probabilities[option_id]) * float(scoring_key[option_id])
                    for option_id in OPTION_IDS
                )
            elif scoring_method == "argmax":
                # Stable A-B-C-D tie breaking is intentional and documented.
                selected = max(
                    OPTION_IDS,
                    key=lambda option_id: float(probabilities[option_id]),
                )
                score = float(scoring_key[selected])
            elif scoring_method == "categorical_resample":
                if sampling_seed is None:
                    raise ValueError("categorical_resample缺少sampling_seed")
                selected, _, _ = _sample_probability_choice(
                    probabilities,
                    sampling_seed=sampling_seed,
                    respondent_id=str(respondent_id),
                    item=item,
                )
                score = float(scoring_key[selected])
            else:
                raise ValueError(f"未知计分方式：{scoring_method}")
            values[respondent_index, item_index] = score
    matrix = pd.DataFrame(values, index=list(subject_ids), columns=item_ids)
    matrix.index.name = "respondent_id"
    return matrix


def _distribution_summary(
    resampling_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("A", "B", "C"):
        method_rows = resampling_metrics.loc[resampling_metrics["method"] == method]
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(method_rows[metric], errors="coerce").dropna()
            rows.append({
                "method": method,
                "metric": metric,
                "n_estimable": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else None,
                "sd": float(values.std(ddof=1)) if values.shape[0] > 1 else None,
                "min": float(values.min()) if not values.empty else None,
                "p2_5": float(values.quantile(0.025)) if not values.empty else None,
                "median": float(values.median()) if not values.empty else None,
                "p97_5": float(values.quantile(0.975)) if not values.empty else None,
                "max": float(values.max()) if not values.empty else None,
            })
    return rows


def _comparison_rows(
    fixed_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixed = {
        (str(row["method"]), str(row["scoring_method"])): row
        for row in fixed_rows
    }
    distributions = {
        (str(row["method"]), str(row["metric"])): row
        for row in distribution_rows
    }
    output: list[dict[str, Any]] = []
    for method in ("A", "B", "C"):
        row: dict[str, Any] = {"method": method}
        for metric in METRIC_COLUMNS:
            row[f"current__{metric}"] = fixed[(method, "current_sample")].get(metric)
            row[f"expected__{metric}"] = fixed[(method, "expected_score")].get(metric)
            row[f"argmax__{metric}"] = fixed[(method, "argmax")].get(metric)
            distribution = distributions[(method, metric)]
            for statistic in ("mean", "sd", "p2_5", "p97_5", "min", "max"):
                row[f"resample_{statistic}__{metric}"] = distribution.get(statistic)
        output.append(row)
    return output


def _score_uncertainty(
    method: str,
    current: pd.DataFrame,
    expected: pd.DataFrame,
    argmax: pd.DataFrame,
    sampled_totals: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_totals = current.sum(axis=1).to_numpy(dtype=float)
    expected_totals = expected.sum(axis=1).to_numpy(dtype=float)
    argmax_totals = argmax.sum(axis=1).to_numpy(dtype=float)
    sampled_means = sampled_totals.mean(axis=1)
    sampled_variances = sampled_totals.var(axis=1, ddof=1)
    sampled_sds = np.sqrt(sampled_variances)
    respondent_rows = [
        {
            "method": method,
            "respondent_id": str(respondent_id),
            "current_total": float(current_totals[index]),
            "expected_total": float(expected_totals[index]),
            "argmax_total": float(argmax_totals[index]),
            "resampled_total_mean": float(sampled_means[index]),
            "resampled_total_sd": float(sampled_sds[index]),
            "resampled_total_variance": float(sampled_variances[index]),
        }
        for index, respondent_id in enumerate(current.index)
    ]
    between_expected_variance = float(np.var(expected_totals, ddof=1))
    mean_within_sampling_variance = float(np.mean(sampled_variances))
    denominator = between_expected_variance + mean_within_sampling_variance
    noise_ratio = mean_within_sampling_variance / denominator if denominator > 0 else None
    return respondent_rows, {
        "method": method,
        "between_person_expected_total_variance": between_expected_variance,
        "mean_within_person_sampling_variance": mean_within_sampling_variance,
        "conditional_sampling_noise_ratio": noise_ratio,
        "conditional_signal_ratio": 1.0 - noise_ratio if noise_ratio is not None else None,
        "mean_absolute_current_minus_expected_total": float(
            np.mean(np.abs(current_totals - expected_totals))
        ),
        "mean_resampled_total_sd": float(np.mean(sampled_sds)),
        "noise_ratio_formula": "mean within-person resampling variance / (variance of expected totals between persons + mean within-person resampling variance)",
    }


def _fmt(value: Any) -> str:
    number = _finite(value)
    return "不可估计" if number is None else f"{number:.3f}"


def _render_report(
    output: Path,
    *,
    source: Path,
    subject_count: int,
    replications: int,
    seed: int,
    fixed_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    noise_rows: Sequence[Mapping[str, Any]],
) -> Path:
    fixed = {
        (str(row["method"]), str(row["scoring_method"])): row
        for row in fixed_rows
    }
    distributions = {
        (str(row["method"]), str(row["metric"])): row
        for row in distribution_rows
    }
    noise = {str(row["method"]): row for row in noise_rows}
    scoring_labels = {
        "current_sample": "原始一次抽样",
        "categorical_resample": f"{replications}次重抽均值",
        "expected_score": "概率期望分",
        "argmax": "最大概率选项",
    }
    main_rows: list[str] = []
    for method in ("A", "B", "C"):
        for scoring_method in (
            "current_sample",
            "categorical_resample",
            "expected_score",
            "argmax",
        ):
            if scoring_method == "categorical_resample":
                values = {
                    metric: distributions[(method, metric)]["mean"]
                    for metric in METRIC_COLUMNS
                }
            else:
                values = fixed[(method, scoring_method)]
            main_rows.append(
                "<tr>"
                f"<td>{method}</td><td>{scoring_labels[scoring_method]}</td>"
                f"<td>{_fmt(values.get('cronbach_alpha'))}</td>"
                f"<td>{_fmt(values.get('convergent_rho_E'))}</td>"
                f"<td>{_fmt(values.get('max_abs_discriminant_rho'))}</td>"
                f"<td>{_fmt(values.get('convergent_minus_max_abs_discriminant'))}</td>"
                "</tr>"
            )
    interval_rows: list[str] = []
    for method in ("A", "B", "C"):
        for metric, label in (
            ("cronbach_alpha", "Cronbach α"),
            ("convergent_rho_E", "NEO-E汇聚相关"),
            ("convergent_minus_max_abs_discriminant", "汇聚−最大绝对区分相关"),
        ):
            row = distributions[(method, metric)]
            current = fixed[(method, "current_sample")].get(metric)
            inside = (
                _finite(current) is not None
                and _finite(row.get("p2_5")) is not None
                and _finite(row.get("p97_5")) is not None
                and float(row["p2_5"]) <= float(current) <= float(row["p97_5"])
            )
            interval_rows.append(
                "<tr>"
                f"<td>{method}</td><td>{label}</td><td>{_fmt(current)}</td>"
                f"<td>{_fmt(row.get('mean'))}</td><td>{_fmt(row.get('sd'))}</td>"
                f"<td>[{_fmt(row.get('p2_5'))}, {_fmt(row.get('p97_5'))}]</td>"
                f"<td>{'是' if inside else '否'}</td>"
                "</tr>"
            )
    noise_table = "".join(
        "<tr>"
        f"<td>{method}</td>"
        f"<td>{_fmt(noise[method].get('between_person_expected_total_variance'))}</td>"
        f"<td>{_fmt(noise[method].get('mean_within_person_sampling_variance'))}</td>"
        f"<td>{_fmt(noise[method].get('conditional_sampling_noise_ratio'))}</td>"
        f"<td>{_fmt(noise[method].get('mean_resampled_total_sd'))}</td>"
        "</tr>"
        for method in ("A", "B", "C")
    )
    source_text = escape(str(source))
    html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<title>具身概率离线重计分实验</title>
<style>
body{{font:15px/1.65 system-ui,sans-serif;max-width:1250px;margin:28px auto;padding:0 20px;color:#1d2733}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd3db;padding:7px;text-align:left}}
th{{background:#eef3f8}}code{{background:#f2f4f7;padding:2px 4px}}.note{{background:#fff7d7;border-left:4px solid #d6a600;padding:12px}}
.ok{{background:#eaf7ef;border-left:4px solid #228b4f;padding:12px}}
</style>
<h1>具身概率离线重计分实验</h1>
<div class="ok">模型调用=0，新增Token=0。固定模型输出概率与NEO-FFI效标，只改变概率转分数的规则。</div>
<p>来源：<code>{source_text}</code>；被试={subject_count}；重抽={replications}次；seed序列={seed}至{seed + replications - 1}。</p>
<h2>三种计分方式比较</h2>
<table><thead><tr><th>方法</th><th>计分方式</th><th>α</th><th>NEO-E汇聚ρ</th><th>最大绝对区分ρ</th><th>相关差值</th></tr></thead><tbody>{''.join(main_rows)}</tbody></table>
<p>相关差值 = NEO-E汇聚相关 − max(|NEO-N|, |NEO-O|, |NEO-A|, |NEO-C|)。它是描述性诊断量，不是已建立的传统效度系数。</p>
<h2>概率重抽的不确定性</h2>
<table><thead><tr><th>方法</th><th>指标</th><th>原始抽样</th><th>重抽均值</th><th>SD</th><th>95%经验区间</th><th>原始值在区间内</th></tr></thead><tbody>{''.join(interval_rows)}</tbody></table>
<h2>总分层面的条件抽样噪声</h2>
<table><thead><tr><th>方法</th><th>人际期望总分方差</th><th>人内重抽方差</th><th>抽样噪声占比</th><th>每人重抽总分平均SD</th></tr></thead><tbody>{noise_table}</tbody></table>
<div class="note">“抽样噪声占比”仅在模型给定概率固定的条件下成立。期望分与argmax是两种不同计分估计量；它们与抽样版的差异不能直接解释为真人测量误差，也不能证明概率已经校准成人类选择概率。</div>
<h2>输出文件</h2>
<ul><li><code>resampling_metrics.csv</code>：每个seed的完整指标</li><li><code>resampling_summary.csv</code>：均值、SD及经验区间</li><li><code>scoring_method_comparison.csv</code>：当前抽样、重抽、期望分与argmax比较</li><li><code>respondent_score_variance.csv</code>：每名被试的重抽总分波动</li><li><code>summary.json</code>：机器可读完整结果</li></ul>
</html>"""
    report = output / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


def run_probability_resampling(
    config: ProbabilityResamplingConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    source = experiment / config.evaluation_dirname
    if not source.is_dir() and (experiment / "forms.json").is_file():
        source = experiment
    forms_path = source / "forms.json"
    criterion_path = source / "neo_ffi" / "scores.csv"
    if not forms_path.is_file():
        raise FileNotFoundError(f"缺少冻结问卷：{forms_path}")
    if not criterion_path.is_file():
        raise FileNotFoundError(f"缺少冻结NEO-FFI效标得分：{criterion_path}")

    forms = _read_json(forms_path)
    if not isinstance(forms, Mapping) or set(forms) != {"A", "B", "C"}:
        raise ValueError("forms.json必须且只能包含A、B、C三套问卷")
    criterion = pd.read_csv(criterion_path, encoding="utf-8-sig")
    if "respondent_id" not in criterion.columns or not set(NEO_DOMAINS).issubset(
        criterion.columns
    ):
        raise ValueError("NEO-FFI效标得分必须包含respondent_id及E/N/O/A/C")
    if criterion["respondent_id"].duplicated().any():
        raise ValueError("NEO-FFI效标被试编号重复")
    criterion["respondent_id"] = criterion["respondent_id"].astype(str)
    criterion = criterion.set_index("respondent_id")
    subject_ids = criterion.index.tolist()

    if config.output is None:
        run_name = (
            f"resampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid4().hex[:8]}"
        )
        output = experiment / "probability_resampling" / run_name
    else:
        output = config.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixed_rows: list[dict[str, Any]] = []
    resampling_rows: list[dict[str, Any]] = []
    respondent_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    seed_values = [config.seed + index for index in range(config.replications)]

    for method in ("A", "B", "C"):
        responses_path = source / method / "evaluation" / "responses.jsonl"
        if not responses_path.is_file():
            raise FileNotFoundError(f"缺少{method}概率作答：{responses_path}")
        records = _read_jsonl(responses_path)
        indexed, item_by_id = _validate_and_index_records(
            records, forms[method], subject_ids
        )
        current = _matrix_for_scoring(
            indexed, item_by_id, subject_ids, "current_sample"
        )
        expected = _matrix_for_scoring(
            indexed, item_by_id, subject_ids, "expected_score"
        )
        argmax = _matrix_for_scoring(indexed, item_by_id, subject_ids, "argmax")
        fixed_rows.extend([
            _metric_row(method, "current_sample", current, criterion),
            _metric_row(method, "expected_score", expected, criterion),
            _metric_row(method, "argmax", argmax, criterion),
        ])
        sampled_totals = np.empty(
            (len(subject_ids), config.replications), dtype=float
        )
        for replication, sampling_seed in enumerate(seed_values, 1):
            sampled = _matrix_for_scoring(
                indexed,
                item_by_id,
                subject_ids,
                "categorical_resample",
                sampling_seed=sampling_seed,
            )
            sampled_totals[:, replication - 1] = sampled.sum(axis=1).to_numpy(
                dtype=float
            )
            resampling_rows.append(
                _metric_row(
                    method,
                    "categorical_resample",
                    sampled,
                    criterion,
                    replication=replication,
                    sampling_seed=sampling_seed,
                )
            )
        method_respondents, method_noise = _score_uncertainty(
            method, current, expected, argmax, sampled_totals
        )
        respondent_rows.extend(method_respondents)
        noise_rows.append(method_noise)

    # Because the first resampling seed equals the frozen source seed in the
    # intended experiment, this audit catches drift in the offline sampler.
    source_manifest_path = source / "manifest.json"
    source_manifest = (
        _read_json(source_manifest_path) if source_manifest_path.is_file() else {}
    )
    source_seed = source_manifest.get("sampling_seed")
    current_reproduction: dict[str, Any] = {
        "source_sampling_seed": source_seed,
        "first_resampling_seed": seed_values[0],
        "checked": source_seed == seed_values[0],
        "exact_metric_match": None,
    }
    if current_reproduction["checked"]:
        first_by_method = {
            row["method"]: row
            for row in resampling_rows
            if row["replication"] == 1
        }
        current_by_method = {
            row["method"]: row
            for row in fixed_rows
            if row["scoring_method"] == "current_sample"
        }
        matches = []
        for method in ("A", "B", "C"):
            for metric in METRIC_COLUMNS:
                left = _finite(first_by_method[method].get(metric))
                right = _finite(current_by_method[method].get(metric))
                matches.append(
                    (left is None and right is None)
                    or (
                        left is not None
                        and right is not None
                        and math.isclose(left, right, abs_tol=1e-12)
                    )
                )
        current_reproduction["exact_metric_match"] = all(matches)
        if not all(matches):
            raise ValueError("冻结seed未能精确复现原始计分指标")

    resampling_frame = pd.DataFrame(resampling_rows)
    distribution_rows = _distribution_summary(resampling_frame)
    comparison_rows = _comparison_rows(fixed_rows, distribution_rows)
    resampling_frame.to_csv(
        output / "resampling_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(distribution_rows).to_csv(
        output / "resampling_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(comparison_rows).to_csv(
        output / "scoring_method_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(respondent_rows).to_csv(
        output / "respondent_score_variance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(noise_rows).to_csv(
        output / "sampling_noise_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(output / "config.json", {
        **asdict(config),
        "experiment": str(experiment),
        "output": str(output),
        "source": str(source),
        "metric_version": METRIC_VERSION,
    })
    write_json(output / "cost.json", {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "note": "纯离线重计分；复用冻结概率与NEO-FFI效标。",
    })
    result = {
        "status": "complete",
        "metric_version": METRIC_VERSION,
        "source": str(source),
        "criterion": "frozen NEO-FFI domain scores",
        "respondent_count": len(subject_ids),
        "replications": config.replications,
        "seed_start": seed_values[0],
        "seed_end": seed_values[-1],
        "current_reproduction": current_reproduction,
        "fixed_scoring_metrics": fixed_rows,
        "resampling_summary": distribution_rows,
        "sampling_noise_summary": noise_rows,
        "metric_formulas": {
            "expected_item_score": "sum_option p(option) * scoring_key(option)",
            "argmax": "highest p(option), ties resolved A-B-C-D",
            "alpha": "Cronbach alpha across the 16 scored SJT items",
            "convergent": "Spearman(SJT total, frozen NEO-FFI E total)",
            "discriminant": "Spearman(SJT total, each frozen NEO-FFI N/O/A/C total)",
            "gap": "convergent rho E - max(abs(discriminant rho N/O/A/C))",
        },
        "scope_note": (
            "This conditions on fixed model probabilities and criterion scores. "
            "It quantifies scoring sensitivity, not human calibration or external validity."
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "summary.json", result)
    report = _render_report(
        output,
        source=source,
        subject_count=len(subject_ids),
        replications=config.replications,
        seed=config.seed,
        fixed_rows=fixed_rows,
        distribution_rows=distribution_rows,
        noise_rows=noise_rows,
    )
    result["report"] = str(report)
    return output, result

