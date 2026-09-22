"""Whole-test metrics for provisional iteration forms.

The item-level screening gates remain the authority for repairing individual
items.  This module evaluates the provisional form assembled at each
development round so the workflow can show whether the complete test is
improving, even while some items are still under treatment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PLATEAU_DEFAULT_PATIENCE = 2
PLATEAU_DEFAULT_MIN_DELTA = 0.01
TARGET_RECOVERY_DEFAULT_FOLDS = 5
TARGET_RECOVERY_RIDGE_PENALTY = 1.0
FORM_EFFECT_EXTREME_FRACTION = 1.0 / 3.0
VIRTUAL_FORM_ICC_DEFAULT_MINIMUM = 0.80
FORM_ALPHA_DEFAULT_MINIMUM = 0.80
FORM_CONVERGENT_NONINFERIORITY_TOLERANCE = 0.02
FORM_QUALITY_EPSILON = 1e-12
CURRENT_FORM_METRIC_FRAMEWORK = "virtual_form_response_transmission_v3"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _spearman(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) < 3 or len(right) < 3:
        return None
    try:
        result = stats.spearmanr(left, right, nan_policy="omit")
    except Exception:
        return None
    return _number(getattr(result, "statistic", None))


def _cronbach_alpha(frame: pd.DataFrame) -> float | None:
    if frame.shape[1] < 2 or frame.shape[0] < 3:
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        return None
    total_variance = float(numeric.sum(axis=1).var(ddof=1))
    if not np.isfinite(total_variance) or total_variance <= 0:
        return None
    item_variance = float(numeric.var(axis=0, ddof=1).sum())
    alpha = numeric.shape[1] / (numeric.shape[1] - 1) * (
        1.0 - item_variance / total_variance
    )
    return _number(alpha)


def _read_scored_responses(test_statistics: Mapping[str, Any]) -> pd.DataFrame | None:
    output_files = test_statistics.get("output_files") or {}
    if not isinstance(output_files, Mapping):
        return None
    path_value = output_files.get("scored_matched_condition_sjt_responses")
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return None
    try:
        frame = pd.read_csv(path_value)
    except Exception:
        return None
    required = {
        "condition_id",
        "arm_id",
        "group_id",
        "matched_subject_id",
        "item_id",
        "score",
        "active_score",
    }
    if not required <= set(frame.columns):
        return None
    frame["item_id"] = frame["item_id"].astype(str)
    frame["condition_id"] = frame["condition_id"].astype(str)
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["matched_subject_id"] = frame["matched_subject_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["active_score"] = pd.to_numeric(frame["active_score"], errors="coerce")
    return frame


def _read_scored_target_retests(
    test_statistics: Mapping[str, Any],
) -> pd.DataFrame | None:
    output_files = test_statistics.get("output_files") or {}
    if not isinstance(output_files, Mapping):
        return None
    path_value = output_files.get("scored_target_form_retest_sjt_responses")
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return None
    try:
        frame = pd.read_csv(path_value)
    except Exception:
        return None
    required = {
        "administration_id",
        "condition_id",
        "matched_subject_id",
        "item_id",
        "score",
    }
    if not required <= set(frame.columns):
        return None
    frame = frame[frame["condition_id"].astype(str) == "target"].copy()
    frame["administration_id"] = pd.to_numeric(
        frame["administration_id"], errors="coerce"
    )
    frame["matched_subject_id"] = frame["matched_subject_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(
        subset=["administration_id", "matched_subject_id", "item_id", "score"]
    )
    return frame if not frame.empty else None


def _read_jsonl(path_value: Any) -> list[dict[str, Any]]:
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with Path(path_value).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
    except Exception:
        return []
    return records


def _neo_dimension_code(domain_id: Any) -> str | None:
    normalized = str(domain_id or "").strip().lower()
    for prefix, code in {
        "openness": "O",
        "conscientiousness": "C",
        "extraversion": "E",
        "agreeableness": "A",
        "neuroticism": "N",
    }.items():
        if normalized == prefix or normalized.startswith(prefix + "_"):
            return code
    return None


def _read_virtual_reference_scores(
    test_statistics: Mapping[str, Any],
    *,
    target_subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Read matched target reference questionnaires for development-only validity."""

    output_files = test_statistics.get("output_files") or {}
    reference_meta = test_statistics.get("reference_questionnaires") or {}
    if not isinstance(output_files, Mapping) or not isinstance(reference_meta, Mapping):
        return {"status": "unavailable", "reason": "整卷统计缺少参照问卷元数据"}
    target_ids = {str(value) for value in target_subject_ids}
    if not target_ids:
        return {"status": "unavailable", "reason": "没有可对齐的 target 被试"}
    scores = pd.DataFrame(index=sorted(target_ids))
    details: dict[str, Any] = {}

    ipip_records = [
        record
        for record in _read_jsonl(output_files.get("ipip_neo_responses"))
        if str(record.get("condition_id")) == "target"
        and str(record.get("matched_subject_id")) in target_ids
    ]
    ipip_meta = reference_meta.get("ipip_neo")
    incomplete_reference = False
    if isinstance(ipip_meta, Mapping) and ipip_records:
        raw_facets = ipip_meta.get("facets")
        if isinstance(raw_facets, list) and raw_facets:
            facet_specs = [
                dict(row)
                for row in raw_facets
                if isinstance(row, Mapping) and row.get("facet_code")
            ]
        else:
            # Compatibility with the first IPIP implementation and its
            # already-written single-facet manifests.
            legacy_code = str(ipip_meta.get("target_facet_code") or "")
            facet_specs = (
                [
                    {
                        "facet_code": legacy_code,
                        "facet_id": ipip_meta.get("target_dimension_id"),
                        "item_count": ipip_meta.get("item_count"),
                    }
                ]
                if legacy_code
                else []
            )
        facet_details: list[dict[str, Any]] = []
        target_code = str(ipip_meta.get("target_facet_code") or "")
        for facet_spec in facet_specs:
            code = str(facet_spec.get("facet_code") or "")
            expected_items = int(facet_spec.get("item_count") or 0)
            selected = [
                record
                for record in ipip_records
                if str(record.get("facet_code")) == code
            ]
            grouped_scores: dict[str, dict[str, float]] = {}
            for record in selected:
                subject_id = str(record.get("matched_subject_id") or "")
                item_id = str(record.get("item_id") or "")
                raw_score = record.get("score")
                if not subject_id or not item_id or isinstance(raw_score, bool):
                    continue
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(score):
                    grouped_scores.setdefault(subject_id, {})[item_id] = score
            complete_scores = {
                subject_id: float(np.mean(list(item_scores.values())))
                for subject_id, item_scores in grouped_scores.items()
                if item_scores
                and (not expected_items or len(item_scores) == expected_items)
            }
            column = f"ipip_{code}_score"
            facet_series = pd.Series(complete_scores, dtype="float64", name=column)
            if code and not facet_series.empty:
                scores[column] = facet_series
            if len(facet_series) != len(target_ids):
                incomplete_reference = True
            facet_details.append(
                {
                    "facet_code": code,
                    "facet_id": facet_spec.get("facet_id"),
                    "score_column": column,
                    "sample_size": int(facet_series.notna().sum()),
                    "item_response_count": int(len(selected)),
                    "items_per_respondent": expected_items,
                    "complete": len(facet_series) == len(target_ids),
                }
            )
        if facet_details:
            target_code = target_code or str(facet_details[0]["facet_code"])
            target_column = f"ipip_{target_code}_score"
            if target_column in scores:
                # Keep the old public column name so existing reports/tests
                # remain readable while new code can use every facet column.
                scores["ipip_target_facet_score"] = scores[target_column]
            details["ipip_neo"] = {
                "target_facet_code": target_code,
                "target_dimension_id": ipip_meta.get("target_dimension_id"),
                "facet_count": len(facet_details),
                "facets": facet_details,
                "sample_size": int(
                    scores[target_column].notna().sum()
                    if target_column in scores
                    else 0
                ),
            }

    # Backward-compatible reader for completed historical runs.
    neo_records = [
        record
        for record in _read_jsonl(output_files.get("neo_ffi_responses"))
        if str(record.get("condition_id")) == "target"
        and str(record.get("matched_subject_id")) in target_ids
    ]
    neo_meta = reference_meta.get("neo_ffi")
    if isinstance(neo_meta, Mapping) and neo_records:
        code = _neo_dimension_code(neo_meta.get("target_domain_id"))
        if code is None:
            code = str(neo_meta.get("target_dimension_code") or "") or None
        selected = [
            record
            for record in neo_records
            if str(record.get("dimension_code")) == code
        ]
        if code and selected:
            # Do not construct a DataFrame from the complete model records.
            # Older response files may contain non-tabular metadata fields
            # (lists/dicts), which makes pandas' string dtype conversion fail.
            grouped_scores: dict[str, list[float]] = {}
            for record in selected:
                subject_id = str(record.get("matched_subject_id") or "")
                raw_score = record.get("score")
                if not subject_id or isinstance(raw_score, bool):
                    continue
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(score):
                    grouped_scores.setdefault(subject_id, []).append(score)
            neo_scores = pd.Series(
                {
                    subject_id: float(np.mean(values))
                    for subject_id, values in grouped_scores.items()
                    if values
                },
                dtype="float64",
                name="neo_ffi_target_score",
            )
            scores["neo_ffi_target_score"] = neo_scores
            details["neo_ffi"] = {
                "dimension_code": code,
                "sample_size": int(neo_scores.notna().sum()),
                "item_response_count": int(len(selected)),
            }

    available = [column for column in scores.columns if scores[column].notna().any()]
    if not available:
        return {
            "status": "unavailable",
            "reason": "当前轮次没有完整的IPIP-NEO或旧版NEO-FFI参照分数",
            "details": details,
        }
    complete = not incomplete_reference and all(
        int(scores[column].notna().sum()) == len(target_ids)
        for column in available
    )
    return {
        "status": "complete" if complete else "partial",
        "scores": scores[available],
        "details": details,
    }


def prepare_provisional_form_metric_context(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Load round-level form data once for repeated candidate-form scoring."""

    test_statistics = state.get("test_statistics") or {}
    if not isinstance(test_statistics, Mapping):
        return {"responses": None, "reference_result": {"status": "unavailable"}}
    responses = _read_scored_responses(test_statistics)
    if responses is None:
        return {"responses": None, "reference_result": {"status": "unavailable"}}
    condition_item_scores: dict[str, pd.DataFrame] = {}
    condition_active_scores: dict[str, pd.Series] = {}
    condition_metadata: dict[str, dict[str, str]] = {}
    for condition_id, condition in responses.groupby("condition_id", sort=True):
        condition_key = str(condition_id)
        condition_item_scores[condition_key] = condition.pivot_table(
            index="matched_subject_id",
            columns="item_id",
            values="score",
            aggfunc="first",
        ).sort_index()
        condition_active_scores[condition_key] = (
            condition[["matched_subject_id", "active_score"]]
            .drop_duplicates("matched_subject_id")
            .set_index("matched_subject_id")["active_score"]
            .sort_index()
        )
        first = condition.iloc[0]
        condition_metadata[condition_key] = {
            "arm_id": str(first.get("arm_id") or ""),
            "group_id": str(first.get("group_id") or ""),
        }
    target_ids = (
        responses.loc[
            responses["condition_id"] == "target", "matched_subject_id"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    target_scores = condition_item_scores.get("target", pd.DataFrame())
    target_active = condition_active_scores.get(
        "target", pd.Series(dtype="float64")
    )
    group_active: dict[str, dict[str, pd.Series]] = {}
    for condition_id, active in condition_active_scores.items():
        if condition_id == "target":
            continue
        metadata = condition_metadata.get(condition_id) or {}
        group_active.setdefault(str(metadata.get("arm_id") or ""), {})[
            str(metadata.get("group_id") or condition_id)
        ] = active
    target_retest_scores: dict[int, pd.DataFrame] = {}
    target_retests = _read_scored_target_retests(test_statistics)
    if target_retests is not None:
        for administration_id, administration in target_retests.groupby(
            "administration_id", sort=True
        ):
            target_retest_scores[int(administration_id)] = (
                administration.pivot_table(
                    index="matched_subject_id",
                    columns="item_id",
                    values="score",
                    aggfunc="first",
                ).sort_index()
            )
    item_facet_ids = {
        str(item.get("item_id")): str(item.get("target_dimension_id"))
        for item in [
            *(state.get("frozen_item_bank") or []),
            *(state.get("item_pool") or []),
        ]
        if isinstance(item, Mapping)
        and item.get("item_id")
        and item.get("target_dimension_id")
    }
    return {
        "responses": responses,
        "target_scores": target_scores,
        "target_active": target_active,
        "group_active": group_active,
        "condition_item_scores": condition_item_scores,
        "condition_active_scores": condition_active_scores,
        "condition_metadata": condition_metadata,
        "target_retest_scores": target_retest_scores,
        "item_facet_ids": item_facet_ids,
        "reference_result": _read_virtual_reference_scores(
            test_statistics,
            target_subject_ids=target_ids,
        ),
    }


def _icc_absolute_agreement_single(scores: pd.DataFrame) -> float | None:
    """Two-way absolute-agreement single-measure ICC, ICC(A,1)."""

    numeric = scores.apply(pd.to_numeric, errors="coerce").dropna(
        axis=0, how="any"
    )
    n_subjects, n_administrations = numeric.shape
    if n_subjects < 3 or n_administrations < 2:
        return None
    values = numeric.to_numpy(dtype=float)
    grand_mean = float(values.mean())
    subject_means = values.mean(axis=1)
    administration_means = values.mean(axis=0)
    ms_subject = n_administrations * float(
        np.square(subject_means - grand_mean).sum()
    ) / float(n_subjects - 1)
    ms_administration = n_subjects * float(
        np.square(administration_means - grand_mean).sum()
    ) / float(n_administrations - 1)
    residual = (
        values
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
        return None
    return _number((ms_subject - ms_error) / denominator)


def _stable_fold_ids(index: Sequence[Any], fold_count: int) -> np.ndarray:
    """Assign deterministic folds without depending on row order."""

    hashed = np.asarray(
        [
            int.from_bytes(
                sha256(str(value).encode("utf-8")).digest()[:8], "big"
            )
            for value in index
        ],
        dtype=np.uint64,
    )
    order = np.argsort(hashed, kind="stable")
    folds = np.empty(len(order), dtype=int)
    folds[order] = np.arange(len(order), dtype=int) % fold_count
    return folds


def _cross_validated_target_recovery(
    item_scores: pd.DataFrame,
    target_scores: pd.Series,
    *,
    fold_count: int = TARGET_RECOVERY_DEFAULT_FOLDS,
    ridge_penalty: float = TARGET_RECOVERY_RIDGE_PENALTY,
) -> dict[str, Any]:
    """Recover assigned target scores from the complete item-response pattern."""

    aligned = item_scores.join(target_scores.rename("target_score"), how="inner")
    aligned = aligned.apply(pd.to_numeric, errors="coerce").dropna(
        axis=0, how="any"
    )
    n = len(aligned)
    resolved_folds = min(int(fold_count), n)
    if n < 6 or resolved_folds < 2:
        return {
            "status": "unavailable",
            "reason": "交叉验证目标恢复至少需要6名完整虚拟被试",
            "sample_size": n,
        }
    x = aligned[item_scores.columns].to_numpy(dtype=float)
    y = aligned["target_score"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "unavailable", "reason": "目标恢复数据包含非有限值"}
    if float(np.var(y, ddof=1)) <= 0:
        return {"status": "unavailable", "reason": "目标构念分数没有变异"}
    folds = _stable_fold_ids(aligned.index.tolist(), resolved_folds)
    predictions = np.full(n, np.nan, dtype=float)
    for fold in range(resolved_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        if int(test_mask.sum()) == 0 or int(train_mask.sum()) < 3:
            continue
        train_x = x[train_mask]
        test_x = x[test_mask]
        train_y = y[train_mask]
        x_mean = train_x.mean(axis=0)
        x_sd = train_x.std(axis=0, ddof=1)
        x_sd = np.where(
            np.isfinite(x_sd) & (x_sd > 1e-12), x_sd, 1.0
        )
        train_z = (train_x - x_mean) / x_sd
        test_z = (test_x - x_mean) / x_sd
        y_mean = float(train_y.mean())
        centered_y = train_y - y_mean
        gram = train_z.T @ train_z
        penalty = float(ridge_penalty) * np.eye(gram.shape[0], dtype=float)
        try:
            coefficients = np.linalg.solve(
                gram + penalty, train_z.T @ centered_y
            )
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(gram + penalty) @ (
                train_z.T @ centered_y
            )
        predictions[test_mask] = y_mean + test_z @ coefficients
    if not np.isfinite(predictions).all():
        return {"status": "unavailable", "reason": "交叉验证预测不完整"}
    denominator = float(np.square(y - y.mean()).sum())
    if denominator <= 0:
        return {"status": "unavailable", "reason": "目标构念分数没有变异"}
    r_squared = 1.0 - float(np.square(y - predictions).sum()) / denominator
    prediction_rho = _spearman(predictions.tolist(), y.tolist())
    return {
        "status": "complete",
        "cross_validated_r2": _number(r_squared),
        "prediction_spearman": prediction_rho,
        "sample_size": n,
        "item_count": int(item_scores.shape[1]),
        "fold_count": resolved_folds,
        "model": "ridge_regression_on_complete_item_pattern",
        "ridge_penalty": float(ridge_penalty),
    }


def _extreme_group_effect(
    form_scores: pd.Series,
    active_scores: pd.Series,
    *,
    denominator_sd: float,
) -> dict[str, Any]:
    aligned = pd.concat(
        [form_scores.rename("form_score"), active_scores.rename("active_score")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 6 or not np.isfinite(denominator_sd) or denominator_sd <= 0:
        return {"status": "unavailable", "sample_size": int(len(aligned))}
    low_threshold = float(
        aligned["active_score"].quantile(FORM_EFFECT_EXTREME_FRACTION)
    )
    high_threshold = float(
        aligned["active_score"].quantile(1.0 - FORM_EFFECT_EXTREME_FRACTION)
    )
    low = aligned[aligned["active_score"] <= low_threshold]["form_score"]
    high = aligned[aligned["active_score"] >= high_threshold]["form_score"]
    if len(low) < 2 or len(high) < 2:
        return {"status": "unavailable", "sample_size": int(len(aligned))}
    raw_difference = float(high.mean() - low.mean())
    return {
        "status": "complete",
        "standardized_effect": _number(raw_difference / denominator_sd),
        "raw_score_difference": _number(raw_difference),
        "low_mean": _number(float(low.mean())),
        "high_mean": _number(float(high.mean())),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "sample_size": int(len(aligned)),
    }


def _hedges_extreme_group_effect(
    outcome_scores: pd.Series,
    grouping_scores: pd.Series,
) -> dict[str, Any]:
    """Compute a human-style upper-vs-lower-third Hedges' g.

    The grouping variable is an external IPIP facet score.  The outcome is a
    facet (or whole-form) SJT score.  This is deliberately separate from the
    virtual matched-condition effect used by the item-screening workflow.
    """

    aligned = pd.concat(
        [
            pd.to_numeric(outcome_scores, errors="coerce").rename("outcome"),
            pd.to_numeric(grouping_scores, errors="coerce").rename("group"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 6:
        return {"status": "unavailable", "sample_size": int(len(aligned))}
    low_threshold = float(aligned["group"].quantile(FORM_EFFECT_EXTREME_FRACTION))
    high_threshold = float(
        aligned["group"].quantile(1.0 - FORM_EFFECT_EXTREME_FRACTION)
    )
    low = aligned.loc[aligned["group"] <= low_threshold, "outcome"]
    high = aligned.loc[aligned["group"] >= high_threshold, "outcome"]
    if len(low) < 2 or len(high) < 2:
        return {
            "status": "unavailable",
            "reason": "高低组有效样本不足",
            "sample_size": int(len(aligned)),
        }
    low_sd = float(low.std(ddof=1))
    high_sd = float(high.std(ddof=1))
    degrees_of_freedom = len(low) + len(high) - 2
    pooled_variance = (
        (len(low) - 1) * low_sd**2 + (len(high) - 1) * high_sd**2
    ) / degrees_of_freedom
    pooled_sd = float(np.sqrt(pooled_variance))
    if not np.isfinite(pooled_sd) or pooled_sd <= 0:
        return {
            "status": "unavailable",
            "reason": "高低组SJT得分没有可用的组内方差",
            "sample_size": int(len(aligned)),
        }
    raw_difference = float(high.mean() - low.mean())
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
    return {
        "status": "complete",
        "standardized_effect": _number(correction * raw_difference / pooled_sd),
        "raw_score_difference": _number(raw_difference),
        "pooled_sd": _number(pooled_sd),
        "hedges_correction": _number(correction),
        "low_mean": _number(float(low.mean())),
        "high_mean": _number(float(high.mean())),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "sample_size": int(len(aligned)),
        "grouping_source": "IPIP-NEO facet upper/lower third",
    }


def _construct_selectivity_value(
    target_sensitivity: Any,
    maximum_leakage: Any,
) -> float | None:
    """Return the bounded target-signal share used by whole-form optimization."""

    target = _number(target_sensitivity)
    leakage = _number(maximum_leakage)
    if target is None or leakage is None:
        return None
    # A reversed target effect is not useful target transmission.  Preserve
    # the raw signed effect elsewhere, but give it zero optimization credit.
    target_signal = max(0.0, target)
    leakage_signal = abs(leakage)
    denominator = target_signal + leakage_signal
    if denominator <= FORM_QUALITY_EPSILON:
        # No target transmission receives zero quality credit rather than
        # becoming an unevaluable candidate.
        return 0.0
    return _number(target_signal / denominator)


def whole_form_objective_improves(
    current_metrics: Mapping[str, Any] | None,
    incumbent_metrics: Mapping[str, Any] | None,
    *,
    min_delta: float = PLATEAU_DEFAULT_MIN_DELTA,
    convergent_tolerance: float = FORM_CONVERGENT_NONINFERIORITY_TOLERANCE,
) -> bool:
    """Return whether a current v3 form can replace an eligible incumbent.

    Target known-groups Hedges' g is the only improving objective.  The
    convergent target correlation and the minimum discriminant gap are
    non-inferiority protections, not additional scalarized objectives.
    """

    current = form_quality_summary(current_metrics or {})
    incumbent = form_quality_summary(incumbent_metrics or {})
    if not current.get("eligible_for_best_so_far"):
        return False
    if not incumbent.get("eligible_for_best_so_far"):
        return True
    current_g = _number(current.get("objective_primary"))
    incumbent_g = _number(incumbent.get("objective_primary"))
    current_delta = _number(current.get("objective_secondary"))
    incumbent_delta = _number(incumbent.get("objective_secondary"))
    current_rho = _number(current.get("objective_tertiary"))
    incumbent_rho = _number(incumbent.get("objective_tertiary"))
    if any(
        value is None
        for value in (
            current_g,
            incumbent_g,
            current_delta,
            incumbent_delta,
            current_rho,
            incumbent_rho,
        )
    ):
        return False
    if current_g <= incumbent_g + float(min_delta):
        return False
    if current_delta < incumbent_delta - FORM_QUALITY_EPSILON:
        return False
    return current_rho >= incumbent_rho - float(convergent_tolerance)


def form_quality_summary(
    form_metrics: Mapping[str, Any],
    *,
    stability_minimum: float = VIRTUAL_FORM_ICC_DEFAULT_MINIMUM,
) -> dict[str, Any]:
    """Extract the current whole-form objective and its ICC gate.

    Current v3 runs use target-facet known-groups Hedges' g as the primary
    objective, with the minimum discriminant correlation gap and target-facet
    Spearman rho as non-inferiority protections. Cronbach alpha and virtual
    test-retest ICC are gates. Historical metric frameworks remain readable but
    are not mixed into a v3 optimization trajectory.
    """

    if (
        isinstance(stability_minimum, bool)
        or not isinstance(stability_minimum, (int, float))
        or not np.isfinite(float(stability_minimum))
        or not 0.0 <= float(stability_minimum) <= 1.0
    ):
        raise ValueError("虚拟整卷 ICC 门槛必须是 0 到 1 之间的数值")

    reliability = form_metrics.get("reliability") or {}
    validity = form_metrics.get("validity") or {}
    recovery = validity.get("target_recovery") or {}
    selectivity = validity.get("construct_selectivity") or {}
    isolation = validity.get("construct_isolation") or {}
    convergent = validity.get("convergent_validity") or {}
    ipip_isolation = validity.get("ipip_known_groups_isolation") or {}

    recovery_raw = _number(recovery.get("cross_validated_r2"))
    selectivity_value = _number(selectivity.get("value"))
    if selectivity_value is None:
        target_effect = isolation.get("target_sensitivity") or {}
        selectivity_value = _construct_selectivity_value(
            target_effect.get("standardized_effect"),
            isolation.get("maximum_absolute_non_target_leakage"),
    )
    icc = _number(reliability.get("virtual_test_retest_icc"))

    ipip_rho = _number(convergent.get("spearman_rho"))
    ipip_ig = _number(ipip_isolation.get("value"))
    current_framework = (
        form_metrics.get("metric_framework")
        == "virtual_form_response_transmission_v2"
    )
    current_framework_v3 = (
        form_metrics.get("metric_framework")
        == CURRENT_FORM_METRIC_FRAMEWORK
    )

    if current_framework_v3:
        discriminant = validity.get("discriminant_validity") or {}
        known_groups = validity.get("known_groups_validity") or {}
        target_delta = _number(discriminant.get("delta_min"))
        target_g = _number(known_groups.get("target_hedges_g"))
        alpha = _number(reliability.get("cronbach_alpha"))
        complete = all(
            value is not None
            for value in (alpha, icc, ipip_rho, target_delta, target_g)
        )
        alpha_passed = alpha is not None and alpha >= FORM_ALPHA_DEFAULT_MINIMUM
        stability_passed = icc is not None and icc >= float(stability_minimum)
        return {
            "status": "complete" if complete else "unavailable",
            "candidate_form_quality": target_g,
            "objective_primary": target_g,
            "objective_secondary": target_delta,
            "objective_tertiary": ipip_rho,
            "objective_primary_name": "IPIP_target_known_groups_hedges_g",
            "objective_secondary_name": "IPIP_discriminant_delta_min",
            "objective_tertiary_name": "IPIP_target_facet_spearman_rho",
            "objective_source": "ipip_human_style_v3",
            "target_recovery_raw": recovery_raw,
            "construct_selectivity": selectivity_value,
            "ipip_target_facet_spearman_rho": ipip_rho,
            "ipip_discriminant_delta_min": target_delta,
            "ipip_target_known_groups_hedges_g": target_g,
            "aggregation": "primary_target_hedges_g_secondary_delta_min_tertiary_target_rho",
            "formula": (
                "先最大化目标IPIP facet高低组Hedges_g；"
                "Δmin不得下降；目标IPIP Spearman rho最多下降0.02"
            ),
            "alpha_gate": {
                "metric": "cronbach_alpha",
                "minimum": FORM_ALPHA_DEFAULT_MINIMUM,
                "observed": alpha,
                "passed": alpha_passed,
            },
            "stability_gate": {
                "metric": "virtual_test_retest_icc",
                "minimum": float(stability_minimum),
                "observed": icc,
                "passed": stability_passed,
            },
            "eligible_for_best_so_far": bool(
                complete and alpha_passed and stability_passed
            ),
            "interpretation": (
                "当前整卷以目标IPIP facet已知组Hedges_g为主要优化指标，"
                "以Δmin和目标facet Spearman rho作效度保护条件；"
                "不是人类样本正式效度。"
            ),
        }

    # The new objective is usable only when both external-reference quantities
    # are available.  Do not silently fall back to the old virtual-transmission
    # Q for a current run with missing IPIP data.
    if (
        current_framework
        and ipip_rho is not None
        and ipip_ig is not None
    ):
        complete = icc is not None
        stability_passed = icc is not None and icc >= float(stability_minimum)
        return {
            "status": "complete" if complete else "unavailable",
            "candidate_form_quality": ipip_ig,
            "objective_primary": ipip_ig,
            "objective_secondary": ipip_rho,
            "objective_primary_name": "IPIP_known_groups_isolation_I_g",
            "objective_secondary_name": "IPIP_target_facet_spearman_rho",
            "objective_source": "ipip_human_style",
            "target_recovery_raw": recovery_raw,
            "target_recovery_component": (
                min(1.0, max(0.0, recovery_raw))
                if recovery_raw is not None
                else None
            ),
            "construct_selectivity": selectivity_value,
            "ipip_target_facet_spearman_rho": ipip_rho,
            "ipip_known_groups_isolation": ipip_ig,
            "aggregation": "lexicographic_primary_I_g_secondary_target_rho",
            "formula": (
                "先最大化 I_g = g_target - MAX(ABS(g_non_target))；"
                "I_g差异不超过min_delta时最大化目标IPIP facet Spearman rho"
            ),
            "stability_gate": {
                "metric": "virtual_test_retest_icc",
                "minimum": float(stability_minimum),
                "observed": icc,
                "passed": stability_passed,
            },
            "eligible_for_best_so_far": bool(complete and stability_passed),
            "interpretation": (
                "当前整卷迭代直接使用IPIP外部参照的已知组构念隔离度I_g，"
                "以目标facet汇聚Spearman rho作次级目标；不是人类样本正式效度。"
            ),
        }

    if current_framework:
        # A current-framework form without complete IPIP data is not eligible
        # for whole-form optimization.  In particular, missingness is not
        # turned into zero and the old Q is not used as a hidden substitute.
        stability_passed = icc is not None and icc >= float(stability_minimum)
        return {
            "status": "unavailable",
            "candidate_form_quality": None,
            "objective_primary": None,
            "objective_secondary": None,
            "objective_primary_name": "IPIP_known_groups_isolation_I_g",
            "objective_secondary_name": "IPIP_target_facet_spearman_rho",
            "objective_source": "ipip_human_style",
            "target_recovery_raw": recovery_raw,
            "target_recovery_component": (
                min(1.0, max(0.0, recovery_raw))
                if recovery_raw is not None
                else None
            ),
            "construct_selectivity": selectivity_value,
            "ipip_target_facet_spearman_rho": ipip_rho,
            "ipip_known_groups_isolation": ipip_ig,
            "aggregation": "lexicographic_primary_I_g_secondary_target_rho",
            "formula": (
                "需要完整的IPIP_known_groups_isolation_I_g和"
                "目标IPIP facet Spearman rho"
            ),
            "stability_gate": {
                "metric": "virtual_test_retest_icc",
                "minimum": float(stability_minimum),
                "observed": icc,
                "passed": stability_passed,
            },
            "eligible_for_best_so_far": False,
            "interpretation": "当前整卷缺少新的IPIP外部参照指标，不能进入整卷迭代比较。",
        }

    recovery_component = (
        min(1.0, max(0.0, recovery_raw))
        if recovery_raw is not None
        else None
    )
    selectivity_component = (
        min(1.0, max(0.0, selectivity_value))
        if selectivity_value is not None
        else None
    )
    candidate_quality = (
        _number(float(np.sqrt(recovery_component * selectivity_component)))
        if recovery_component is not None and selectivity_component is not None
        else None
    )
    stability_passed = (
        icc is not None and icc >= float(stability_minimum)
    )
    complete = candidate_quality is not None and icc is not None
    return {
        "status": "complete" if complete else "unavailable",
        "candidate_form_quality": candidate_quality,
        "objective_primary": candidate_quality,
        "objective_secondary": selectivity_component,
        "objective_primary_name": "legacy_virtual_Q",
        "objective_secondary_name": "legacy_construct_selectivity",
        "objective_source": "legacy_virtual_transmission",
        "target_recovery_raw": recovery_raw,
        "target_recovery_component": recovery_component,
        "construct_selectivity": selectivity_component,
        "aggregation": "geometric_mean",
        "formula": "sqrt(clipped_target_recovery_r2 * construct_selectivity)",
        "stability_gate": {
            "metric": "virtual_test_retest_icc",
            "minimum": float(stability_minimum),
            "observed": icc,
            "passed": stability_passed,
        },
        "eligible_for_best_so_far": bool(complete and stability_passed),
        "interpretation": (
            "虚拟开发期候选整卷质量；不是人类样本信度或效度。"
        ),
    }


def assess_form_plateau(
    history: Sequence[Mapping[str, Any]],
    *,
    patience: int = PLATEAU_DEFAULT_PATIENCE,
    min_delta: float = PLATEAU_DEFAULT_MIN_DELTA,
) -> dict[str, Any]:
    """Detect a plateau in the retained IPIP-based whole-form objective.

    A complete, blueprint-valid round enters the comparison only after its
    alpha and ICC gates pass.  Current v3 rounds require target Hedges' g to
    improve, while delta_min must not fall and target rho may fall by at most
    0.02. Legacy rounds are supported for reading old checkpoints, but are
    never mixed with current v3 rounds.
    """

    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("平台期 patience 必须是正整数")
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or min_delta < 0:
        raise ValueError("平台期 min_delta 必须是非负数")
    trajectory: list[dict[str, Any]] = []
    best_round: int | None = None
    best_quality: float | None = None
    best_secondary: float | None = None
    best_tertiary: float | None = None
    best_summary: dict[str, Any] | None = None
    best_form_metrics: Mapping[str, Any] | None = None
    non_improving = 0
    reached_round: int | None = None
    usable_rounds = 0
    current_round: int | None = None
    current_summary: dict[str, Any] | None = None

    summaries = [
        form_quality_summary(entry.get("form_metrics") or {})
        for entry in history
        if isinstance(entry, Mapping)
    ]
    preferred_source = (
        "ipip_human_style_v3"
        if any(
            summary.get("objective_source") == "ipip_human_style_v3"
            and summary.get("status") == "complete"
            for summary in summaries
        )
        else (
            "ipip_human_style"
            if any(
                summary.get("objective_source") == "ipip_human_style"
                and summary.get("status") == "complete"
                for summary in summaries
            )
            else None
        )
    )
    if preferred_source is None:
        preferred_source = next(
            (
                str(summary.get("objective_source"))
                for summary in summaries
                if summary.get("objective_source")
                and summary.get("status") == "complete"
            ),
            "ipip_human_style",
        )

    ordered_history = sorted(
        (entry for entry in history if isinstance(entry, Mapping)),
        key=lambda entry: int(entry.get("analysis_round") or 0),
    )
    for entry in ordered_history:
        round_number = int(entry.get("analysis_round") or 0)
        current_round = round_number
        summary = form_quality_summary(entry.get("form_metrics") or {})
        current_summary = summary
        complete_form = entry.get("form_status") in (None, "complete")
        objective_source = summary.get("objective_source")
        quality = _number(summary.get("objective_primary"))
        secondary = _number(summary.get("objective_secondary"))
        tertiary = _number(summary.get("objective_tertiary"))
        eligible = bool(
            complete_form
            and objective_source == preferred_source
            and summary.get("eligible_for_best_so_far")
            and quality is not None
        )
        accepted = False
        if eligible:
            usable_rounds += 1
            if preferred_source == "ipip_human_style_v3":
                accepted = best_form_metrics is None or whole_form_objective_improves(
                    entry.get("form_metrics") or {},
                    best_form_metrics,
                    min_delta=min_delta,
                )
            else:
                primary_improved = best_quality is None or (
                    quality > best_quality + float(min_delta)
                )
                secondary_improved = bool(
                    best_quality is not None
                    and best_secondary is not None
                    and secondary is not None
                    and abs(quality - best_quality) <= float(min_delta)
                    and secondary > best_secondary + float(min_delta)
                )
                accepted = primary_improved or secondary_improved
            if accepted:
                best_quality = quality
                best_secondary = secondary
                best_tertiary = tertiary
                best_round = round_number
                best_summary = summary
                best_form_metrics = entry.get("form_metrics") or {}
                non_improving = 0
            else:
                non_improving += 1
                if non_improving >= patience and reached_round is None:
                    reached_round = round_number
        trajectory.append(
            {
                "analysis_round": round_number,
                "candidate_form_quality": quality,
                "objective_primary": quality,
                "objective_secondary": secondary,
                "objective_source": objective_source,
                "best_so_far_form_quality": best_quality,
                "best_objective_secondary": best_secondary,
                "objective_tertiary": tertiary,
                "best_objective_tertiary": best_tertiary,
                "accepted_as_best": accepted,
                "eligible_for_best_so_far": eligible,
                "stability_gate": dict(summary.get("stability_gate") or {}),
                "non_improving_rounds": non_improving,
            }
        )
    if preferred_source == "ipip_human_style_v3":
        metric_names = [
            "IPIP_target_known_groups_hedges_g",
            "IPIP_discriminant_delta_min",
            "IPIP_target_facet_spearman_rho",
        ]
    elif preferred_source == "ipip_human_style":
        metric_names = [
            "IPIP_known_groups_isolation_I_g",
            "IPIP_target_facet_spearman_rho",
        ]
    else:
        metric_names = [
            "legacy_virtual_Q",
            "legacy_construct_selectivity",
        ]
    base = {
        "status": "insufficient_data",
        "reached": False,
        "patience": patience,
        "min_delta": float(min_delta),
        "usable_rounds": usable_rounds,
        "non_improving_rounds": non_improving,
        "best_round": best_round,
        "best_objective_secondary": best_secondary,
        "current_round": current_round,
        "best_form_quality": best_quality,
        "current_candidate_form_quality": (
            current_summary.get("objective_primary")
            if current_summary is not None
            else None
        ),
        "current_objective_secondary": (
            current_summary.get("objective_secondary")
            if current_summary is not None
            else None
        ),
        "objective_source": preferred_source,
        "best_metrics": best_summary,
        "current_metrics": current_summary,
        "metric_names": metric_names,
        "stability_gate_metric": "virtual_test_retest_icc",
        "trajectory": trajectory,
    }
    if not usable_rounds:
        base["reason"] = "尚无通过 ICC 稳定性门槛的完整整卷轮次"
        return base

    base.update(
        {
            "status": "reached" if reached_round is not None else "monitoring",
            "reached": reached_round is not None,
            "non_improving_rounds": non_improving,
            "best_round": best_round,
            "plateau_round": reached_round,
            "reason": (
                f"连续 {non_improving} 轮候选整卷未使历史最优目标Hedges_g提高至少 {float(min_delta):.3f}"
                if reached_round is not None
                else "继续观察后续整卷轮次"
            ),
        }
    )
    return base


def build_provisional_form_metrics(
    state: Mapping[str, Any],
    selected_item_ids: Sequence[str],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one complete test as a virtual response-transmission system."""

    item_ids = list(dict.fromkeys(str(item_id) for item_id in selected_item_ids))
    if not item_ids:
        return {
            "status": "unavailable",
            "reason": "临时测验没有可计算的题目",
            "item_count": 0,
            "selected_item_ids": [],
        }
    metric_context = (
        context
        if isinstance(context, Mapping)
        else prepare_provisional_form_metric_context(state)
    )
    responses = metric_context.get("responses")
    condition_matrices = metric_context.get("condition_item_scores")
    condition_active = metric_context.get("condition_active_scores")
    condition_metadata = metric_context.get("condition_metadata")
    if (
        not isinstance(responses, pd.DataFrame)
        or not isinstance(condition_matrices, Mapping)
        or not isinstance(condition_active, Mapping)
        or "target" not in condition_matrices
        or "target" not in condition_active
    ):
        return {
            "status": "unavailable",
            "reason": "当前轮次缺少完整的匹配条件逐题作答数据",
            "item_count": len(item_ids),
            "selected_item_ids": item_ids,
        }
    target_scores = condition_matrices["target"]
    target_active = condition_active["target"]
    if not isinstance(target_scores, pd.DataFrame) or not isinstance(
        target_active, pd.Series
    ):
        return {
            "status": "unavailable",
            "reason": "target 条件数据结构无效",
            "item_count": len(item_ids),
            "selected_item_ids": item_ids,
        }
    missing = [
        item_id for item_id in item_ids if item_id not in target_scores.columns
    ]
    if missing:
        return {
            "status": "unavailable",
            "reason": "逐题作答数据缺少临时测验题目：" + ", ".join(missing),
            "item_count": len(item_ids),
            "selected_item_ids": item_ids,
        }
    target_matrix = target_scores[item_ids].dropna(axis=0, how="any")
    form_score = target_matrix.mean(axis=1)
    item_by_id = {
        str(item.get("item_id")): item
        for item in (state.get("frozen_item_bank") or [])
        if isinstance(item, Mapping) and item.get("item_id")
    }
    facet_item_ids: dict[str, list[str]] = {}
    for item_id in item_ids:
        facet_id = str(item_by_id.get(item_id, {}).get("target_dimension_id") or "")
        if facet_id:
            facet_item_ids.setdefault(facet_id, []).append(item_id)

    # 1) Virtual whole-form reliability: repeat the same target personas and
    # calculate absolute-agreement ICC on complete-form scores.
    administration_scores = {"1": form_score}
    target_retests = metric_context.get("target_retest_scores")
    if isinstance(target_retests, Mapping):
        for administration_id, matrix in target_retests.items():
            if not isinstance(matrix, pd.DataFrame) or any(
                item_id not in matrix.columns for item_id in item_ids
            ):
                continue
            administration_scores[str(administration_id)] = matrix[
                item_ids
            ].dropna(axis=0, how="any").mean(axis=1)
    stability_frame = pd.concat(administration_scores, axis=1, join="inner")
    stability_icc = _icc_absolute_agreement_single(stability_frame)
    stability = {
        "status": "complete" if stability_icc is not None else "unavailable",
        "virtual_test_retest_icc": stability_icc,
        "method": "ICC(A,1)_absolute_agreement_single_measure",
        "sample_size": int(len(stability_frame)),
        "administration_count": int(stability_frame.shape[1]),
        "administration_ids": list(stability_frame.columns),
        "interpretation": (
            "同一 target 虚拟人格重复完成整套测验时的总分绝对一致性；"
            "不是人类样本信度。"
        ),
    }
    if stability_icc is None:
        stability["reason"] = "缺少至少两次完整 target 整卷施测或总分无变异"

    # 2) Target recovery: held-out prediction from the complete response
    # pattern, not an average of item-level target correlations.
    recovery = _cross_validated_target_recovery(target_matrix, target_active)
    recovery["interpretation"] = (
        "使用整套题目的完整作答模式，在留出虚拟被试上恢复预设目标构念；"
        "不是人类汇聚效度或校标效度。"
    )

    # 3) Construct isolation: compare target sensitivity with the largest
    # high-low score effect produced by any non-target condition's own form.
    target_sd = _number(float(form_score.std(ddof=1)))
    target_effect = _extreme_group_effect(
        form_score,
        target_active,
        denominator_sd=target_sd or 0.0,
    )
    leakage_groups: list[dict[str, Any]] = []
    for condition_id, matrix in condition_matrices.items():
        condition_id = str(condition_id)
        if condition_id == "target" or not isinstance(matrix, pd.DataFrame):
            continue
        if any(item_id not in matrix.columns for item_id in item_ids):
            continue
        active = condition_active.get(condition_id)
        if not isinstance(active, pd.Series):
            continue
        effect = _extreme_group_effect(
            matrix[item_ids].dropna(axis=0, how="any").mean(axis=1),
            active,
            denominator_sd=target_sd or 0.0,
        )
        metadata = (
            condition_metadata.get(condition_id, {})
            if isinstance(condition_metadata, Mapping)
            else {}
        )
        leakage_groups.append(
            {
                "condition_id": condition_id,
                "arm_id": metadata.get("arm_id"),
                "group_id": metadata.get("group_id"),
                **effect,
            }
        )
    estimable_leakage = [
        row
        for row in leakage_groups
        if isinstance(row.get("standardized_effect"), (int, float))
        and not isinstance(row.get("standardized_effect"), bool)
    ]
    maximum_leakage = (
        max(abs(float(row["standardized_effect"])) for row in estimable_leakage)
        if estimable_leakage
        else None
    )
    target_sensitivity = target_effect.get("standardized_effect")
    isolation_value = (
        float(target_sensitivity) - float(maximum_leakage)
        if isinstance(target_sensitivity, (int, float))
        and maximum_leakage is not None
        else None
    )
    selectivity_value = _construct_selectivity_value(
        target_sensitivity,
        maximum_leakage,
    )
    construct_isolation = {
        "status": "complete" if isolation_value is not None else "unavailable",
        "value": _number(isolation_value),
        "target_sensitivity": target_effect,
        "maximum_absolute_non_target_leakage": _number(maximum_leakage),
        "non_target_groups": leakage_groups,
        "effect_definition": (
            "upper-third minus lower-third form-score mean, divided by the "
            "target-arm form-score SD"
        ),
        "interpretation": (
            "目标构念高低变化引起的整卷效应减去最大非目标构念泄漏；"
            "不是人类区分效度。"
        ),
    }
    construct_selectivity = {
        "status": "complete" if selectivity_value is not None else "unavailable",
        "value": _number(selectivity_value),
        "target_sensitivity": _number(target_sensitivity),
        "maximum_absolute_non_target_leakage": _number(maximum_leakage),
        "target_direction_passed": bool(
            isinstance(target_sensitivity, (int, float))
            and not isinstance(target_sensitivity, bool)
            and float(target_sensitivity) > 0.0
        ),
        "formula": "max(0,T) / (max(0,T) + abs(L))",
        "interpretation": (
            "目标构念信号占目标信号与最大非目标泄漏总量的比例；"
            "仅用于虚拟开发期构念特异性评价。"
        ),
    }

    # Human-style quantities remain descriptive diagnostics only.  They are
    # retained for backward-compatible reports but are not iteration targets.
    alpha = _cronbach_alpha(target_matrix)
    target_aligned = pd.concat(
        [form_score.rename("form_score"), target_active.rename("active_score")],
        axis=1,
        join="inner",
    ).dropna()
    target_rho = _spearman(
        target_aligned["form_score"].tolist(),
        target_aligned["active_score"].tolist(),
    )
    reference_result = metric_context.get("reference_result")
    if not isinstance(reference_result, Mapping):
        reference_result = {"status": "unavailable"}
    reference_scores = reference_result.get("scores")
    ipip_rho = None
    ipip_sample_size = 0
    ipip_facet_rhos: list[dict[str, Any]] = []
    ipip_spearman_matrix: list[dict[str, Any]] = []
    ipip_isolation_results: list[dict[str, Any]] = []
    neo_rho = None
    sjt_facet_scores = {
        facet_id: target_matrix[facet_ids].mean(axis=1)
        for facet_id, facet_ids in facet_item_ids.items()
        if facet_ids
    }
    if isinstance(reference_scores, pd.DataFrame):
        reference_details = reference_result.get("details") or {}
        ipip_details = (
            reference_details.get("ipip_neo")
            if isinstance(reference_details, Mapping)
            else None
        )
        ipip_facets = (
            ipip_details.get("facets")
            if isinstance(ipip_details, Mapping)
            else None
        )
        if not isinstance(ipip_facets, list):
            ipip_facets = []
        ipip_columns: list[dict[str, Any]] = []
        for raw_facet in ipip_facets:
            if not isinstance(raw_facet, Mapping):
                continue
            code = str(raw_facet.get("facet_code") or "")
            column = str(
                raw_facet.get("score_column") or f"ipip_{code}_score"
            )
            if code and column in reference_scores:
                ipip_columns.append(
                    {
                        "facet_code": code,
                        "facet_id": raw_facet.get("facet_id"),
                        "score_column": column,
                    }
                )
        if not ipip_columns and "ipip_target_facet_score" in reference_scores:
            target_code = str(
                (ipip_details or {}).get("target_facet_code") or "target"
            )
            ipip_columns = [
                {
                    "facet_code": target_code,
                    "facet_id": (ipip_details or {}).get("target_dimension_id"),
                    "score_column": "ipip_target_facet_score",
                }
            ]

        for sjt_facet_id, sjt_scores in sjt_facet_scores.items():
            matching = next(
                (
                    spec
                    for spec in ipip_columns
                    if str(spec.get("facet_id") or "") == sjt_facet_id
                ),
                None,
            )
            for ipip_spec in ipip_columns:
                aligned = pd.concat(
                    [
                        sjt_scores.rename("sjt_score"),
                        reference_scores[ipip_spec["score_column"]].rename(
                            "ipip_score"
                        ),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                rho = _spearman(
                    aligned["sjt_score"].tolist(),
                    aligned["ipip_score"].tolist(),
                )
                ipip_spearman_matrix.append(
                    {
                        "sjt_facet_id": sjt_facet_id,
                        "ipip_facet_id": ipip_spec.get("facet_id"),
                        "ipip_facet_code": ipip_spec["facet_code"],
                        "spearman_rho": rho,
                        "sample_size": int(len(aligned)),
                        "diagonal": bool(
                            str(ipip_spec.get("facet_id") or "") == sjt_facet_id
                        ),
                    }
                )
            if matching is not None:
                aligned = pd.concat(
                    [
                        sjt_scores.rename("sjt_score"),
                        reference_scores[matching["score_column"]].rename(
                            "ipip_score"
                        ),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                facet_rho = _spearman(
                    aligned["sjt_score"].tolist(),
                    aligned["ipip_score"].tolist(),
                )
                ipip_facet_rhos.append(
                    {
                        "sjt_facet_id": sjt_facet_id,
                        "ipip_facet_id": matching.get("facet_id"),
                        "ipip_facet_code": matching["facet_code"],
                        "spearman_rho": facet_rho,
                        "sample_size": int(len(aligned)),
                    }
                )
                for non_target in ipip_columns:
                    if non_target is matching:
                        continue
                    non_target_result = _hedges_extreme_group_effect(
                        sjt_scores,
                        reference_scores[non_target["score_column"]],
                    )
                    ipip_isolation_results.append(
                        {
                            "sjt_facet_id": sjt_facet_id,
                            "target_ipip_facet_code": matching["facet_code"],
                            "ipip_facet_code": non_target["facet_code"],
                            "ipip_facet_id": non_target.get("facet_id"),
                            **non_target_result,
                        }
                    )
        diagonal_rhos = [
            row["spearman_rho"]
            for row in ipip_facet_rhos
            if isinstance(row.get("spearman_rho"), (int, float))
            and not isinstance(row.get("spearman_rho"), bool)
        ]
        ipip_rho = (
            _number(float(np.mean(diagonal_rhos))) if diagonal_rhos else None
        )
        ipip_sample_size = (
            min(int(row["sample_size"]) for row in ipip_facet_rhos)
            if ipip_facet_rhos
            else 0
        )
        aligned_reference = pd.concat(
            [form_score.rename("form_score"), reference_scores],
            axis=1,
            join="inner",
        )
        if "neo_ffi_target_score" in aligned_reference:
            neo = aligned_reference[
                ["form_score", "neo_ffi_target_score"]
            ].dropna()
            neo_rho = _spearman(
                neo["form_score"].tolist(), neo["neo_ffi_target_score"].tolist()
            )
    reference_rho = ipip_rho if ipip_rho is not None else neo_rho

    ipip_discriminant_by_sjt: list[dict[str, Any]] = []
    for facet_id in facet_item_ids:
        target_row = next(
            (
                row
                for row in ipip_facet_rhos
                if row.get("sjt_facet_id") == facet_id
            ),
            None,
        )
        target_rho = (
            target_row.get("spearman_rho")
            if isinstance(target_row, Mapping)
            else None
        )
        non_target_rows = [
            row
            for row in ipip_spearman_matrix
            if row.get("sjt_facet_id") == facet_id
            and not row.get("diagonal")
            and isinstance(row.get("spearman_rho"), (int, float))
            and not isinstance(row.get("spearman_rho"), bool)
        ]
        maximum = (
            max(abs(float(row["spearman_rho"])) for row in non_target_rows)
            if non_target_rows
            else None
        )
        delta = (
            float(target_rho) - float(maximum)
            if isinstance(target_rho, (int, float)) and maximum is not None
            else None
        )
        ipip_discriminant_by_sjt.append(
            {
                "sjt_facet_id": facet_id,
                "target_spearman_rho": _number(target_rho),
                "maximum_absolute_non_target_spearman_rho": _number(maximum),
                "delta_min": _number(delta),
                "non_target_correlations": non_target_rows,
                "status": "complete" if delta is not None else "unavailable",
                "formula": "target_rho - MAX(ABS(non_target_rho))",
            }
        )

    ipip_isolation_by_sjt: list[dict[str, Any]] = []
    for facet_id in facet_item_ids:
        rows = [
            row
            for row in ipip_isolation_results
            if row.get("sjt_facet_id") == facet_id
            and row.get("status") == "complete"
            and isinstance(row.get("standardized_effect"), (int, float))
            and not isinstance(row.get("standardized_effect"), bool)
        ]
        target_row = next(
            (
                row
                for row in ipip_facet_rhos
                if row.get("sjt_facet_id") == facet_id
            ),
            None,
        )
        target_code = target_row.get("ipip_facet_code") if target_row else None
        # The target effect uses the same upper/lower IPIP grouping rule as the
        # non-target effects.  Rho is reported separately and is not substituted
        # into Hedges' g.
        target_spec = next(
            (
                spec
                for spec in ipip_columns
                if spec.get("facet_code") == target_code
            ),
            None,
        ) if isinstance(reference_scores, pd.DataFrame) else None
        target_effect = None
        if target_spec is not None:
            target_effect = _hedges_extreme_group_effect(
                sjt_facet_scores.get(facet_id, pd.Series(dtype="float64")),
                reference_scores[target_spec["score_column"]],
            )
        estimable = [
            row
            for row in rows
            if isinstance(row.get("standardized_effect"), (int, float))
        ]
        maximum = (
            max(abs(float(row["standardized_effect"])) for row in estimable)
            if estimable
            else None
        )
        target_g = (
            target_effect.get("standardized_effect")
            if isinstance(target_effect, Mapping)
            else None
        )
        isolation = (
            float(target_g) - float(maximum)
            if isinstance(target_g, (int, float)) and maximum is not None
            else None
        )
        ipip_isolation_by_sjt.append(
            {
                "sjt_facet_id": facet_id,
                "target_ipip_facet_code": target_code,
                "target_effect": target_effect,
                "maximum_absolute_non_target_effect": _number(maximum),
                "non_target_effects": rows,
                "isolation": _number(isolation),
                "status": "complete" if isolation is not None else "unavailable",
                "formula": "Hedges_g_target - MAX(ABS(Hedges_g_non_target))",
            }
        )
    estimable_isolations = [
        float(row["isolation"])
        for row in ipip_isolation_by_sjt
        if isinstance(row.get("isolation"), (int, float))
        and not isinstance(row.get("isolation"), bool)
    ]
    ipip_isolation_macro = (
        _number(float(np.mean(estimable_isolations)))
        if estimable_isolations
        else None
    )
    target_hedges_values = [
        float(row["target_effect"]["standardized_effect"])
        for row in ipip_isolation_by_sjt
        if isinstance(row.get("target_effect"), Mapping)
        and isinstance(row["target_effect"].get("standardized_effect"), (int, float))
        and not isinstance(row["target_effect"].get("standardized_effect"), bool)
    ]
    target_hedges_g = (
        _number(float(np.mean(target_hedges_values)))
        if target_hedges_values
        else None
    )
    estimable_deltas = [
        float(row["delta_min"])
        for row in ipip_discriminant_by_sjt
        if isinstance(row.get("delta_min"), (int, float))
        and not isinstance(row.get("delta_min"), bool)
    ]
    discriminant_delta_min = (
        _number(float(min(estimable_deltas)))
        if estimable_deltas
        else None
    )

    ipip_reference_complete = bool(
        isinstance(reference_result, Mapping)
        and reference_result.get("status") == "complete"
        and len(ipip_columns) == 5
        and len(
            {
                str(spec.get("facet_code") or "")
                for spec in ipip_columns
            }
        )
        == 5
        and len(ipip_facet_rhos) == len(facet_item_ids)
        and all(
            row.get("spearman_rho") is not None
            for row in ipip_facet_rhos
        )
        and len(ipip_discriminant_by_sjt) == len(facet_item_ids)
        and all(
            row.get("status") == "complete"
            for row in ipip_discriminant_by_sjt
        )
        and ipip_rho is not None
        and target_hedges_g is not None
        and discriminant_delta_min is not None
    )
    # R² and matched-condition selectivity remain diagnostic quantities.  The
    # current whole-form objective is based on the complete IPIP external
    # reference instead of those virtual-transmission measures.
    complete = all(
        value is not None
        for value in (
            stability_icc,
            ipip_rho if ipip_reference_complete else None,
            target_hedges_g if ipip_reference_complete else None,
            discriminant_delta_min if ipip_reference_complete else None,
        )
    )
    result = {
        "status": "complete" if complete else "partial",
        "selected_item_ids": item_ids,
        "item_count": len(item_ids),
        "sample_size": int(len(target_matrix)),
        "metric_framework": CURRENT_FORM_METRIC_FRAMEWORK,
        "reliability": {
            **stability,
            "cronbach_alpha": alpha,
            "cronbach_alpha_role": "whole_form_reliability_gate",
        },
        "validity": {
            "target_recovery": recovery,
            "construct_selectivity": construct_selectivity,
            "construct_isolation": construct_isolation,
            "convergent_validity": {
                "status": (
                    "complete"
                    if ipip_reference_complete
                    and facet_item_ids
                    and len(ipip_facet_rhos) == len(facet_item_ids)
                    and all(
                        row.get("spearman_rho") is not None
                        for row in ipip_facet_rhos
                    )
                    else "unavailable"
                ),
                "reference_questionnaire": (
                    "IPIP-NEO selected facets"
                    if ipip_rho is not None
                    else (
                        "NEO-FFI parent domain (legacy)"
                        if neo_rho is not None
                        else None
                    )
                ),
                "spearman_rho": reference_rho,
                "facet_results": ipip_facet_rhos,
                "cross_facet_spearman_matrix": ipip_spearman_matrix,
                "sample_size": (
                    ipip_sample_size if ipip_rho is not None else None
                ),
                "aggregation": "mean_of_target_facet_rhos",
                "role": "whole_form_iteration_secondary_objective",
            },
            "ipip_known_groups_isolation": {
                "status": (
                    "complete"
                    if ipip_reference_complete
                    else "unavailable"
                ),
                "value": ipip_isolation_macro,
                "facet_results": ipip_isolation_by_sjt,
                "formula": (
                    "I_g = Hedges_g_target - "
                    "MAX(ABS(Hedges_g_non_target))"
                ),
                "grouping": "each IPIP facet's upper and lower third",
                "role": "whole_form_iteration_primary_objective",
                "interpretation": (
                    "目标SJT facet在对应IPIP facet高低组上的效应，"
                    "减去其在其他IPIP facet高低组上的最大绝对效应；"
                    "不是独立真人校标。"
                ),
            },
            "discriminant_validity": {
                "status": (
                    "complete"
                    if discriminant_delta_min is not None
                    else "unavailable"
                ),
                "delta_min": discriminant_delta_min,
                "facet_results": ipip_discriminant_by_sjt,
                "formula": "target_rho - MAX(ABS(non_target_rho))",
                "role": "whole_form_noninferiority_constraint",
            },
            "known_groups_validity": {
                "status": (
                    "complete" if target_hedges_g is not None else "unavailable"
                ),
                "target_hedges_g": target_hedges_g,
                "facet_results": [
                    {
                        "sjt_facet_id": row.get("sjt_facet_id"),
                        "target_ipip_facet_code": row.get(
                            "target_ipip_facet_code"
                        ),
                        "hedges_g": (
                            row.get("target_effect", {}) or {}
                        ).get("standardized_effect"),
                    }
                    for row in ipip_isolation_by_sjt
                ],
                "formula": "target IPIP facet upper/lower third Hedges_g",
                "role": "whole_form_primary_iteration_objective",
            },
            "legacy_virtual_diagnostics": {
                "target_total_score_spearman": target_rho,
                "ipip_neo_target_facet_rho": ipip_rho,
                "ipip_neo_facet_rhos": ipip_facet_rhos,
                "ipip_neo_spearman_matrix": ipip_spearman_matrix,
                "ipip_known_groups_isolation": ipip_isolation_by_sjt,
                "neo_ffi_rho": neo_rho,
                "combined_reference_rho": reference_rho,
                "reference_details": reference_result.get("details") or {},
                "filtering_authority": False,
            },
            "criterion_validity": {
                "status": "not_available",
                "value": None,
                "reason": (
                    "active_score与IPIP-NEO参照作答由同一虚拟人格生成过程产生，"
                    "不能作为独立外部校标。"
                ),
            },
        },
        "iteration_objectives": [
            "validity.known_groups_validity.target_hedges_g",
        ],
        "iteration_components": [
            "validity.known_groups_validity.target_hedges_g",
            "validity.discriminant_validity.delta_min",
            "validity.convergent_validity.spearman_rho",
        ],
        "iteration_constraints": [
            "reliability.cronbach_alpha >= 0.80",
            "optimization.stability_gate",
            "validity.discriminant_validity.delta_min_non_decrease",
            "validity.convergent_validity.spearman_rho_noninferiority",
        ],
        "interpretation": (
            "目标IPIP facet已知组Hedges_g为整卷主目标，Δmin和目标facet "
            "Spearman rho为保护条件，Cronbach alpha和虚拟重测ICC为门槛；"
            "不能替代真人信效度验证。"
        ),
    }
    result["optimization"] = form_quality_summary(result)
    return result


def batch_provisional_form_quality(
    context: Mapping[str, Any] | None,
    item_combinations: Sequence[Sequence[str]],
) -> dict[str, np.ndarray] | None:
    """Fast first-stage ranking proxies for many complete forms.

    The final shortlisted forms are always rescored with the full ICC and
    external-reference metrics. This vectorized pass only narrows a
    potentially large search. The current objective proxies are target-facet
    Hedges' g, the minimum discriminant correlation gap, and target-facet
    Spearman rho; older virtual transmission quantities remain diagnostics.
    """

    if not isinstance(context, Mapping) or not item_combinations:
        return None
    target_scores = context.get("target_scores")
    target_active = context.get("target_active")
    condition_matrices = context.get("condition_item_scores")
    condition_active = context.get("condition_active_scores")
    target_retests = context.get("target_retest_scores")
    if (
        not isinstance(target_scores, pd.DataFrame)
        or not isinstance(target_active, pd.Series)
        or not isinstance(condition_matrices, Mapping)
        or not isinstance(condition_active, Mapping)
        or not isinstance(target_retests, Mapping)
        or not target_retests
    ):
        return None
    combinations = [list(map(str, row)) for row in item_combinations]
    item_count = len(combinations[0])
    if item_count < 2 or any(
        len(row) != item_count or len(set(row)) != item_count
        for row in combinations
    ):
        return None
    candidate_ids = list(
        dict.fromkeys(item_id for row in combinations for item_id in row)
    )
    if any(item_id not in target_scores.columns for item_id in candidate_ids):
        return None
    item_scores = target_scores[candidate_ids].to_numpy(dtype=float)
    if item_scores.ndim != 2 or not np.isfinite(item_scores).all():
        return None
    id_to_index = {
        item_id: index for index, item_id in enumerate(candidate_ids)
    }
    selection = np.zeros(
        (len(combinations), len(candidate_ids)), dtype=float
    )
    for row_index, row in enumerate(combinations):
        selection[
            row_index, [id_to_index[item_id] for item_id in row]
        ] = 1.0
    form_scores = selection @ item_scores.T / float(item_count)

    def rowwise_spearman(
        values: np.ndarray, criterion: pd.Series
    ) -> np.ndarray | None:
        aligned = criterion.reindex(target_scores.index).to_numpy(dtype=float)
        if aligned.ndim != 1 or not np.isfinite(aligned).all():
            return None
        ranked_values = stats.rankdata(values, axis=1, method="average")
        ranked_criterion = stats.rankdata(aligned, method="average")
        centered_values = ranked_values - ranked_values.mean(
            axis=1, keepdims=True
        )
        centered_criterion = ranked_criterion - ranked_criterion.mean()
        numerator = centered_values @ centered_criterion
        denominator = np.sqrt(
            (centered_values * centered_values).sum(axis=1)
            * float((centered_criterion * centered_criterion).sum())
        )
        return np.divide(
            numerator,
            denominator,
            out=np.full(len(values), np.nan, dtype=float),
            where=denominator > 0,
        )

    target_rho = rowwise_spearman(form_scores, target_active)
    if target_rho is None:
        return None
    # Signed squared correlation is only a screening proxy.  Full candidates
    # use held-out ridge prediction from every selected item response.
    recovery_proxy = target_rho * np.abs(target_rho)

    # New whole-form objective: evaluate each candidate form against the
    # complete IPIP five-facet reference using the same facet-level formulas as
    # the full evaluator.  This keeps the exhaustive search fast without
    # reverting to the old virtual-transmission Q as a ranking proxy.
    ipip_target_rho: np.ndarray | None = None
    ipip_target_hedges_g: np.ndarray | None = None
    ipip_discriminant_delta_min: np.ndarray | None = None
    ipip_known_groups_isolation: np.ndarray | None = None
    reference_result = context.get("reference_result")
    item_facet_ids = context.get("item_facet_ids")
    if (
        not isinstance(reference_result, Mapping)
        or reference_result.get("status") != "complete"
        or not isinstance(reference_result.get("scores"), pd.DataFrame)
        or not isinstance(item_facet_ids, Mapping)
    ):
        return None
    reference_scores = reference_result["scores"].reindex(target_scores.index)
    reference_details = reference_result.get("details") or {}
    ipip_details = (
        reference_details.get("ipip_neo")
        if isinstance(reference_details, Mapping)
        else None
    )
    raw_facets = ipip_details.get("facets") if isinstance(ipip_details, Mapping) else None
    if not isinstance(raw_facets, list) or not raw_facets:
        return None
    ipip_specs = [
        {
            "facet_id": str(row.get("facet_id") or ""),
            "facet_code": str(row.get("facet_code") or ""),
            "score_column": str(
                row.get("score_column")
                or f"ipip_{row.get('facet_code')}_score"
            ),
        }
        for row in raw_facets
        if isinstance(row, Mapping)
        and row.get("facet_id")
        and row.get("facet_code")
    ]
    if (
        len(ipip_specs) != 5
        or len({spec["facet_code"] for spec in ipip_specs}) != 5
        or len({spec["facet_id"] for spec in ipip_specs}) != 5
        or any(
        spec["score_column"] not in reference_scores.columns
        for spec in ipip_specs
        )
    ):
        return None
    if not np.isfinite(
        reference_scores[[spec["score_column"] for spec in ipip_specs]]
        .to_numpy(dtype=float)
    ).all():
        return None

    def rowwise_hedges_effect(
        values: np.ndarray,
        grouping: np.ndarray,
    ) -> np.ndarray:
        if values.ndim != 2 or grouping.ndim != 1 or values.shape[1] != len(grouping):
            return np.full(len(values), np.nan, dtype=float)
        low_threshold = float(
            np.quantile(grouping, FORM_EFFECT_EXTREME_FRACTION)
        )
        high_threshold = float(
            np.quantile(grouping, 1.0 - FORM_EFFECT_EXTREME_FRACTION)
        )
        low_mask = grouping <= low_threshold
        high_mask = grouping >= high_threshold
        low_n = int(low_mask.sum())
        high_n = int(high_mask.sum())
        if low_n < 2 or high_n < 2:
            return np.full(len(values), np.nan, dtype=float)
        low_values = values[:, low_mask]
        high_values = values[:, high_mask]
        low_mean = low_values.mean(axis=1)
        high_mean = high_values.mean(axis=1)
        low_variance = np.square(low_values - low_mean[:, None]).sum(axis=1) / (low_n - 1)
        high_variance = np.square(high_values - high_mean[:, None]).sum(axis=1) / (high_n - 1)
        degrees_of_freedom = low_n + high_n - 2
        pooled_variance = (
            (low_n - 1) * low_variance
            + (high_n - 1) * high_variance
        ) / degrees_of_freedom
        pooled_sd = np.sqrt(pooled_variance)
        correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
        return np.divide(
            correction * (high_mean - low_mean),
            pooled_sd,
            out=np.full(len(values), np.nan, dtype=float),
            where=np.isfinite(pooled_sd) & (pooled_sd > 0),
        )

    candidate_facet_ids = sorted(
        {
            str(item_facet_ids.get(item_id) or "")
            for item_id in candidate_ids
            if item_facet_ids.get(item_id)
        }
    )
    if not candidate_facet_ids or any(
        item_id not in item_facet_ids for item_id in candidate_ids
    ):
        return None
    target_rho_rows: list[np.ndarray] = []
    target_hedges_rows: list[np.ndarray] = []
    discriminant_delta_rows: list[np.ndarray] = []
    isolation_rows: list[np.ndarray] = []
    for facet_id in candidate_facet_ids:
        facet_indices = [
            id_to_index[item_id]
            for item_id in candidate_ids
            if str(item_facet_ids.get(item_id)) == facet_id
        ]
        if not facet_indices:
            return None
        facet_scores = (
            selection[:, facet_indices] @ item_scores[:, facet_indices].T
            / float(len(facet_indices))
        )
        matching = next(
            (spec for spec in ipip_specs if spec["facet_id"] == facet_id),
            None,
        )
        if matching is None:
            return None
        grouping_columns = {
            spec["facet_id"]: reference_scores[spec["score_column"]]
            .to_numpy(dtype=float)
            for spec in ipip_specs
        }
        target_group = grouping_columns[matching["facet_id"]]
        target_rho_row = rowwise_spearman(
            facet_scores,
            pd.Series(target_group, index=target_scores.index),
        )
        if target_rho_row is None:
            return None
        target_rho_rows.append(target_rho_row)
        target_effect = rowwise_hedges_effect(facet_scores, target_group)
        target_hedges_rows.append(target_effect)
        non_target_rhos = [
            rowwise_spearman(
                facet_scores,
                pd.Series(grouping, index=target_scores.index),
            )
            for reference_id, grouping in grouping_columns.items()
            if reference_id != facet_id
        ]
        if not non_target_rhos or any(row is None for row in non_target_rhos):
            return None
        maximum_non_target_rho = np.max(
            np.vstack([np.abs(row) for row in non_target_rhos]), axis=0
        )
        discriminant_delta_rows.append(
            target_rho_row - maximum_non_target_rho
        )
        non_target_effects = [
            np.abs(rowwise_hedges_effect(facet_scores, grouping))
            for reference_id, grouping in grouping_columns.items()
            if reference_id != facet_id
        ]
        if not non_target_effects:
            return None
        maximum_leakage = np.max(np.vstack(non_target_effects), axis=0)
        isolation_rows.append(target_effect - maximum_leakage)
    ipip_target_rho = np.mean(np.vstack(target_rho_rows), axis=0)
    ipip_target_hedges_g = np.mean(np.vstack(target_hedges_rows), axis=0)
    ipip_discriminant_delta_min = np.min(
        np.vstack(discriminant_delta_rows), axis=0
    )
    ipip_known_groups_isolation = np.mean(np.vstack(isolation_rows), axis=0)
    if any(
        not np.isfinite(array).all()
        for array in (
            ipip_target_rho,
            ipip_target_hedges_g,
            ipip_discriminant_delta_min,
            ipip_known_groups_isolation,
        )
    ):
        return None

    target_sd = form_scores.std(axis=1, ddof=1)
    target_sd = np.where(target_sd > 0, target_sd, np.nan)

    def contrast(
        values: np.ndarray, active: pd.Series
    ) -> np.ndarray | None:
        aligned = active.reindex(target_scores.index).to_numpy(dtype=float)
        if not np.isfinite(aligned).all():
            return None
        low_cut = float(np.quantile(aligned, FORM_EFFECT_EXTREME_FRACTION))
        high_cut = float(
            np.quantile(aligned, 1.0 - FORM_EFFECT_EXTREME_FRACTION)
        )
        low = aligned <= low_cut
        high = aligned >= high_cut
        if int(low.sum()) < 2 or int(high.sum()) < 2:
            return None
        return (
            values[:, high].mean(axis=1)
            - values[:, low].mean(axis=1)
        ) / target_sd

    target_effect = contrast(form_scores, target_active)
    if target_effect is None:
        return None
    leakage_effects: list[np.ndarray] = []
    for condition_id, matrix in condition_matrices.items():
        if str(condition_id) == "target" or not isinstance(matrix, pd.DataFrame):
            continue
        if any(item_id not in matrix.columns for item_id in candidate_ids):
            return None
        active = condition_active.get(condition_id)
        if not isinstance(active, pd.Series):
            return None
        condition_items = matrix.reindex(target_scores.index)[
            candidate_ids
        ].to_numpy(dtype=float)
        if not np.isfinite(condition_items).all():
            return None
        condition_forms = selection @ condition_items.T / float(item_count)
        effect = contrast(condition_forms, active)
        if effect is None:
            return None
        leakage_effects.append(np.abs(effect))
    if not leakage_effects:
        return None
    maximum_leakage = np.max(np.vstack(leakage_effects), axis=0)
    construct_isolation = target_effect - maximum_leakage
    positive_target = np.maximum(target_effect, 0.0)
    selectivity_denominator = positive_target + maximum_leakage
    construct_selectivity = np.divide(
        positive_target,
        selectivity_denominator,
        out=np.zeros(len(positive_target), dtype=float),
        where=selectivity_denominator > FORM_QUALITY_EPSILON,
    )
    candidate_form_quality_proxy = np.sqrt(
        np.clip(recovery_proxy, 0.0, 1.0)
        * np.clip(construct_selectivity, 0.0, 1.0)
    )

    def concordance(
        primary: np.ndarray, repeated: np.ndarray
    ) -> np.ndarray:
        primary_mean = primary.mean(axis=1)
        repeated_mean = repeated.mean(axis=1)
        centered_primary = primary - primary_mean[:, None]
        centered_repeated = repeated - repeated_mean[:, None]
        covariance = (
            centered_primary * centered_repeated
        ).mean(axis=1)
        denominator = (
            np.square(centered_primary).mean(axis=1)
            + np.square(centered_repeated).mean(axis=1)
            + np.square(primary_mean - repeated_mean)
        )
        return np.divide(
            2.0 * covariance,
            denominator,
            out=np.full(len(primary), np.nan, dtype=float),
            where=denominator > 0,
        )

    stability_proxies: list[np.ndarray] = []
    for matrix in target_retests.values():
        if not isinstance(matrix, pd.DataFrame) or any(
            item_id not in matrix.columns for item_id in candidate_ids
        ):
            return None
        repeated_items = matrix.reindex(target_scores.index)[
            candidate_ids
        ].to_numpy(dtype=float)
        if not np.isfinite(repeated_items).all():
            return None
        repeated_forms = selection @ repeated_items.T / float(item_count)
        stability_proxies.append(concordance(form_scores, repeated_forms))
    stability_proxy = np.min(np.vstack(stability_proxies), axis=0)
    values = {
        "ipip_target_facet_spearman_rho": ipip_target_rho,
        "ipip_target_known_groups_hedges_g": ipip_target_hedges_g,
        "ipip_discriminant_delta_min": ipip_discriminant_delta_min,
        "ipip_known_groups_isolation": ipip_known_groups_isolation,
        "target_recovery_proxy": recovery_proxy,
        "construct_selectivity": construct_selectivity,
        "candidate_form_quality_proxy": candidate_form_quality_proxy,
        "construct_isolation": construct_isolation,
        "stability_proxy": stability_proxy,
    }
    if any(not np.isfinite(array).all() for array in values.values()):
        return None
    return values
