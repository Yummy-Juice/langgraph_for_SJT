"""Deterministic scoring and psychometric analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sjt_system.evaluation.respondents import (
    PERSONA_MODE_SCORE_PROFILE,
    score_spec_is_target_related,
    MATCHED_CONDITION_IDS,
    MATCHED_CONDITION_SCHEMA_VERSION,
    flatten_matched_condition_groups,
)
from sjt_system.state import PSJTState, create_initial_state
from sjt_system.runtime.trace import utc_timestamp


PSYCHOMETRIC_FORMULA_VERSION = "sjt-matched-facet-virtual-screening-v11"
MEASUREMENT_EVALUATION_VERSION = "sjt-evaluation-v12"
OPTION_CHOICE_DIAGNOSTICS_VERSION = "matched-condition-option-choice-v3"
MINIMUM_OPTION_DIAGNOSTIC_GROUP_N = 10
DIFFICULTY_LOWER_BOUND = 0.20
DIFFICULTY_UPPER_BOUND = 0.80
MINIMUM_OPTION_RATE = 0.05
CITC_MINIMUM_ACCEPTABLE = 0.30
CITC_REVISION_THRESHOLD = 0.20
CITC_STRONG = 0.50
TARGET_RHO_THRESHOLD = 0.30
SAME_DOMAIN_VTS_THRESHOLD = 0.10
CROSS_DOMAIN_VTS_THRESHOLD = 0.20


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须包含 JSON 对象")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是有效 JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} 第 {line_number} 行必须是 JSON 对象"
                )
            records.append(record)
    if not records:
        raise ValueError(f"{path.name} 没有作答记录")
    return records


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)


def standardized_item_difficulty(
    scores: Sequence[float],
    theoretical_minimum: float,
    theoretical_maximum: float,
) -> float:
    """Return the polytomous item mean normalized to its theoretical range."""

    if theoretical_maximum <= theoretical_minimum:
        raise ValueError("理论最高分必须大于理论最低分")
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("难度计算需要非空且有限的题目得分")
    return float(
        (values.mean() - theoretical_minimum)
        / (theoretical_maximum - theoretical_minimum)
    )


def cronbach_alpha(matrix: np.ndarray) -> float | None:
    """Compute raw Cronbach alpha with sample variances (ddof=1)."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Cronbach alpha 输入必须是二维矩阵")
    sample_count, item_count = values.shape
    if sample_count < 2 or item_count < 2:
        return None
    if not np.isfinite(values).all():
        raise ValueError("Cronbach alpha 输入不能包含缺失值或无穷值")
    item_variances = values.var(axis=0, ddof=1)
    total_variance = values.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return None
    return float(
        item_count
        / (item_count - 1)
        * (1 - item_variances.sum() / total_variance)
    )


def _pearson(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"r": None, "p_value": None, "n": int(x.size)}
    result = stats.pearsonr(x, y)
    return {
        "r": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(x.size),
    }


def _spearman(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": None, "p_value": None, "n": int(x.size)}
    result = stats.spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(x.size),
    }


def _validate_analysis_identity(
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "completed":
        raise ValueError("只有 completed 的虚拟作答数据可以进入分析")
    expected = {
        "run_id": state.get("run_id"),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(
                f"虚拟作答 {field} 与当前冻结题库不一致："
                f"{manifest.get(field)!r} != {expected_value!r}"
            )
    if state.get("virtual_response_item_bank_id") != state.get(
        "item_bank_id"
    ):
        raise ValueError("State 中虚拟作答绑定的 item_bank_id 已失效")
    if state.get("virtual_response_item_bank_version") != state.get(
        "item_bank_version"
    ):
        raise ValueError("State 中虚拟作答绑定的题库版本已失效")


def _resolve_response_paths(
    state: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    reference = state.get("virtual_response_data_ref")
    if not isinstance(reference, str) or not reference:
        raise ValueError("心理测量分析前缺少 virtual_response_data_ref")
    manifest_path = Path(reference).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"虚拟作答 manifest 不存在：{manifest_path}")
    manifest = _read_json(manifest_path)
    _validate_analysis_identity(state, manifest)
    sjt_path = manifest_path.parent / "sjt_responses.jsonl"
    ipip_path = manifest_path.parent / "ipip_neo_responses.jsonl"
    legacy_neo_path = manifest_path.parent / "neo_ffi_responses.jsonl"
    if not sjt_path.is_file():
        raise ValueError("虚拟作答目录缺少 SJT JSONL 文件")
    if manifest.get("schema_version") not in {2, 3, MATCHED_CONDITION_SCHEMA_VERSION} and not legacy_neo_path.is_file():
        raise ValueError("旧版虚拟作答目录缺少 Neo-FFI JSONL 文件")
    reference_path = ipip_path if ipip_path.is_file() else legacy_neo_path
    return manifest_path, sjt_path, reference_path, manifest


def _prepare_item_contract(
    state: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("心理测量分析前缺少冻结题库")
    item_order: list[str] = []
    items: dict[str, dict[str, Any]] = {}
    for raw_item in frozen:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = dict(raw_item)
        item_id = item.get("item_id")
        scoring_key = item.get("scoring_key")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("冻结题库包含缺少 item_id 的题目")
        if item_id in items:
            raise ValueError(f"冻结题库包含重复题目：{item_id}")
        if not isinstance(scoring_key, Mapping) or len(scoring_key) < 2:
            raise ValueError(f"题目 {item_id} 缺少有效 scoring_key")
        if not isinstance(item.get("target_dimension_id"), str) or not item[
            "target_dimension_id"
        ].strip():
            raise ValueError(
                f"题目 {item_id} 缺少有效 target_dimension_id"
            )
        scores = list(scoring_key.values())
        if any(
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            for score in scores
        ):
            raise ValueError(f"题目 {item_id} 的 scoring_key 包含无效分数")
        item_order.append(item_id)
        items[item_id] = item
    return item_order, items


def _item_statistics(
    sjt_long: pd.DataFrame,
    sjt_wide: pd.DataFrame,
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    facet_columns: dict[str, list[str]] = {}
    for item_id in item_order:
        dimension_id = str(items[item_id]["target_dimension_id"])
        facet_columns.setdefault(dimension_id, []).append(item_id)
    statistics_by_item: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []

    for item_id in item_order:
        item = items[item_id]
        dimension_id = str(item["target_dimension_id"])
        scores = sjt_wide[item_id]
        scoring_key = item["scoring_key"]
        theoretical_minimum = float(min(scoring_key.values()))
        theoretical_maximum = float(max(scoring_key.values()))
        difficulty = standardized_item_difficulty(
            scores,
            theoretical_minimum,
            theoretical_maximum,
        )
        same_facet_items = facet_columns[dimension_id]
        facet_rest_score = sjt_wide[same_facet_items].sum(axis=1) - scores
        corrected = _pearson(scores, facet_rest_score)

        item_responses = sjt_long[sjt_long["item_id"] == item_id]
        option_counts = item_responses[
            "selected_option_id"
        ].value_counts()
        option_statistics: dict[str, dict[str, Any]] = {}
        for option_id in scoring_key:
            count = int(option_counts.get(option_id, 0))
            rate = count / len(item_responses)
            option_result = {
                "count": count,
                "rate": float(rate),
                "score": float(scoring_key[option_id]),
                "passes_minimum_rate": rate >= MINIMUM_OPTION_RATE,
            }
            option_statistics[str(option_id)] = option_result
            option_rows.append(
                {
                    "item_id": item_id,
                    "option_id": option_id,
                    **option_result,
                }
            )

        minimum_option_rate = min(
            option["rate"] for option in option_statistics.values()
        )
        effective_option_count = sum(
            option["rate"] >= MINIMUM_OPTION_RATE
            for option in option_statistics.values()
        )
        difficulty_pass = (
            DIFFICULTY_LOWER_BOUND
            <= difficulty
            <= DIFFICULTY_UPPER_BOUND
        )
        citc_pass = (
            corrected["r"] is not None
            and corrected["r"] >= CITC_REVISION_THRESHOLD
        )
        option_rate_pass = effective_option_count >= 3
        qualified = difficulty_pass and citc_pass and option_rate_pass

        result = {
            "item_id": item_id,
            "dimension_id": dimension_id,
            "context_category": item.get("context_category"),
            "n": int(scores.count()),
            "mean": float(scores.mean()),
            "standard_deviation": float(scores.std(ddof=1)),
            "observed_minimum": float(scores.min()),
            "observed_maximum": float(scores.max()),
            "theoretical_minimum": theoretical_minimum,
            "theoretical_maximum": theoretical_maximum,
            "difficulty": difficulty,
            "facet_corrected_item_total_correlation": corrected,
            "option_statistics": option_statistics,
            "minimum_option_rate": minimum_option_rate,
            "effective_option_count": effective_option_count,
            "qualification": {
                "difficulty_pass": difficulty_pass,
                "citc_pass": citc_pass,
                "option_rate_pass": option_rate_pass,
                "qualified": qualified,
            },
        }
        statistics_by_item[item_id] = _json_safe(result)
        flat_rows.append(
            {
                key: value
                for key, value in {
                    "item_id": item_id,
                    "dimension_id": dimension_id,
                    "context_category": item.get("context_category"),
                    "n": result["n"],
                    "mean": result["mean"],
                    "standard_deviation": result["standard_deviation"],
                    "observed_minimum": result["observed_minimum"],
                    "observed_maximum": result["observed_maximum"],
                    "theoretical_minimum": theoretical_minimum,
                    "theoretical_maximum": theoretical_maximum,
                    "difficulty": difficulty,
                    "citc_r": corrected["r"],
                    "citc_p": corrected["p_value"],
                    "minimum_option_rate": minimum_option_rate,
                    "effective_option_count": effective_option_count,
                    "difficulty_pass": difficulty_pass,
                    "citc_pass": citc_pass,
                    "option_rate_pass": option_rate_pass,
                    "qualified": qualified,
                }.items()
            }
        )
    return (
        statistics_by_item,
        pd.DataFrame(flat_rows),
        pd.DataFrame(option_rows),
    )


def _score_matched_sjt(
    records: Sequence[Mapping[str, Any]],
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondent_count: int,
    condition_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Score one response per item and validate exact matching across facet groups."""

    expected_ids = tuple(str(value) for value in (condition_ids or MATCHED_CONDITION_IDS))
    if "target" not in expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("matched condition_ids 必须包含唯一的 target group")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    condition_subjects: dict[str, set[str]] = {condition_id: set() for condition_id in expected_ids}
    for record in records:
        respondent_id = str(record.get("respondent_id") or "")
        condition_id = str(record.get("condition_id") or "")
        matched_subject_id = str(record.get("matched_subject_id") or "")
        item_id = str(record.get("item_id") or "")
        option_id = record.get("selected_option_id")
        if condition_id not in expected_ids or not respondent_id or not matched_subject_id:
            raise ValueError("SJT 作答缺少有效 condition_id 或 matched_subject_id")
        if item_id not in items:
            raise ValueError(f"SJT 作答引用未知题目：{item_id!r}")
        key = (condition_id, matched_subject_id, respondent_id, item_id)
        if key in seen:
            raise ValueError(f"SJT 作答包含重复记录：{key}")
        seen.add(key)
        item = items[item_id]
        if record.get("item_version") != item.get("version"):
            raise ValueError(f"题目 {item_id} 的作答版本与冻结题库不一致")
        scoring_key = item["scoring_key"]
        if option_id not in scoring_key:
            raise ValueError(f"题目 {item_id} 选择了 scoring_key 中不存在的选项")
        condition_subjects[condition_id].add(matched_subject_id)
        rows.append({
            "respondent_id": respondent_id,
            "condition_id": condition_id,
            "arm_id": record.get("arm_id") or ("target" if condition_id == "target" else condition_id.split("__", 1)[0]),
            "group_id": record.get("group_id") or ("target" if condition_id == "target" else condition_id.split("__", 1)[-1]),
            "matched_subject_id": matched_subject_id,
            "item_id": item_id,
            "item_version": item.get("version"),
            "dimension_id": item.get("target_dimension_id"),
            "context_category": item.get("context_category"),
            "raw_display_option_id": record.get("raw_display_option_id"),
            "selected_option_id": str(option_id),
            "score": float(scoring_key[option_id]),
            "active_score": float(record.get("active_score")) if record.get("active_score") is not None else np.nan,
        })
    long_frame = pd.DataFrame(rows)
    if long_frame.empty:
        raise ValueError("匹配 facet SJT 作答为空")
    for condition_id, subjects in condition_subjects.items():
        if len(subjects) != expected_respondent_count:
            raise ValueError(f"条件 {condition_id} 的被试数与 manifest 不一致")
    target_subjects = set(condition_subjects["target"])
    if any(set(subjects) != target_subjects for subjects in condition_subjects.values()):
        raise ValueError("所有 facet group 的 matched_subject_id 集合必须完全一致")
    expected_records = expected_respondent_count * len(item_order) * len(expected_ids)
    if len(long_frame) != expected_records:
        raise ValueError(f"匹配 facet SJT 作答不完整：应有 {expected_records} 条，实际 {len(long_frame)} 条")
    condition_wide: dict[str, pd.DataFrame] = {}
    condition_scores: dict[str, pd.Series] = {}
    for condition_id in expected_ids:
        frame = long_frame[long_frame["condition_id"] == condition_id]
        wide = frame.pivot(index="matched_subject_id", columns="item_id", values="score").reindex(columns=list(item_order))
        if wide.isna().any().any() or len(wide) != expected_respondent_count:
            raise ValueError(f"条件 {condition_id} 的被试×题目矩阵存在缺失作答")
        condition_wide[condition_id] = wide.sort_index()
        score_series = frame.drop_duplicates("matched_subject_id").set_index("matched_subject_id")["active_score"].reindex(wide.index)
        if score_series.isna().any():
            raise ValueError(f"条件 {condition_id} 缺少匹配 facet 分数")
        condition_scores[condition_id] = score_series.astype(float)
    return long_frame, condition_wide, condition_scores


def _score_target_form_retests(
    records: Sequence[Mapping[str, Any]],
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondent_count: int,
    administration_ids: Sequence[int],
) -> pd.DataFrame:
    """Score target-only repeated administrations without changing item gates."""

    expected_ids = tuple(int(value) for value in administration_ids)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("target 重测 administration_ids 必须非空且唯一")
    rows: list[pd.DataFrame] = []
    for administration_id in expected_ids:
        administration_records = [
            record
            for record in records
            if record.get("administration_id") == administration_id
        ]
        scored, _, _ = _score_matched_sjt(
            administration_records,
            item_order=item_order,
            items=items,
            expected_respondent_count=expected_respondent_count,
            condition_ids=("target",),
        )
        scored.insert(0, "administration_id", administration_id)
        rows.append(scored)
    unexpected = {
        record.get("administration_id")
        for record in records
        if record.get("administration_id") not in set(expected_ids)
    }
    if unexpected:
        raise ValueError(f"target 重测包含未声明施测轮次：{sorted(unexpected)}")
    return pd.concat(rows, ignore_index=True)


def _matched_item_metrics(
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    condition_wide: Mapping[str, pd.DataFrame],
    condition_scores: Mapping[str, pd.Series],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate target-arm CITC and signed MAX VTS across fixed arms."""

    group_rows = flatten_matched_condition_groups(conditions)
    condition_rows = {str(row.get("condition_id")): dict(row) for row in group_rows}
    target_row = condition_rows.get("target") or {}
    arm_group_ids: dict[str, list[str]] = {arm_id: [] for arm_id in MATCHED_CONDITION_IDS}
    for row in group_rows:
        arm_group_ids.setdefault(str(row.get("arm_id")), []).append(str(row.get("condition_id")))
    target_wide = condition_wide["target"]
    facet_columns: dict[str, list[str]] = {}
    for item_id in item_order:
        facet_columns.setdefault(str(items[item_id]["target_dimension_id"]), []).append(item_id)
    metrics: dict[str, dict[str, Any]] = {}
    for item_id in item_order:
        item = items[item_id]
        target_id = str(item["target_dimension_id"])
        same_items = facet_columns[target_id]
        citc = _pearson(target_wide[item_id], target_wide[same_items].sum(axis=1) - target_wide[item_id])
        rhos = {
            condition_id: _spearman(
                condition_scores[condition_id],
                condition_wide[condition_id][item_id],
            )
            for condition_id in condition_rows
        }
        target_rho = rhos["target"].get("rho")

        def vts_group(arm_id: str, threshold: float) -> dict[str, Any]:
            rows = []
            for condition_id in arm_group_ids.get(arm_id, []):
                metadata = condition_rows.get(condition_id, {})
                rows.append({
                    **metadata,
                    "condition_id": condition_id,
                    "arm_id": arm_id,
                    "rho": rhos[condition_id].get("rho"),
                })
            estimable = [row for row in rows if row.get("rho") is not None]
            largest = max(estimable, key=lambda row: float(row["rho"])) if estimable else None
            max_rho = float(largest["rho"]) if largest else None
            margin = float(target_rho) - max_rho if target_rho is not None and max_rho is not None else None
            return {
                "non_target_spearman": rows,
                "max_non_target_rho": max_rho,
                "largest_non_target_dimension_id": largest.get("dimension_id") if largest else None,
                "largest_non_target_facet_name": largest.get("facet_name") if largest else None,
                "largest_non_target_domain_id": largest.get("domain_id") if largest else None,
                "largest_non_target_rho": max_rho,
                "selected_non_target_condition_id": largest.get("condition_id") if largest else None,
                "selected_non_target_group_id": largest.get("group_id") if largest else None,
                "selected_non_target_facet": deepcopy(largest) if largest else None,
                "specificity_margin": margin,
                "margin_threshold": threshold,
                "passes": margin is not None and margin >= threshold,
                "largest_non_target_facet": deepcopy(largest) if largest else None,
            }

        same_domain = vts_group("same_domain", SAME_DOMAIN_VTS_THRESHOLD)
        cross_domain = vts_group("cross_domain", CROSS_DOMAIN_VTS_THRESHOLD)
        same_rho = same_domain.get("max_non_target_rho")
        cross_rho = cross_domain.get("max_non_target_rho")
        same_vts = same_domain.get("specificity_margin")
        cross_vts = cross_domain.get("specificity_margin")
        metrics[item_id] = {
            "facet_citc": {**citc, "threshold": CITC_REVISION_THRESHOLD, "passes": citc.get("r") is not None and citc["r"] >= CITC_REVISION_THRESHOLD, "condition_id": "target", "filtering_authority": True},
            "virtual_target_specificity": {
                "target_dimension_id": target_id,
                "target_spearman": rhos["target"],
                "rho_target": rhos["target"],
                "rho_same_domain": {"rho": same_rho, "selected_group_id": same_domain.get("selected_non_target_group_id"), "selected_condition_id": same_domain.get("selected_non_target_condition_id"), "groups": same_domain.get("non_target_spearman", [])},
                "rho_cross_domain": {"rho": cross_rho, "selected_group_id": cross_domain.get("selected_non_target_group_id"), "selected_condition_id": cross_domain.get("selected_non_target_condition_id"), "groups": cross_domain.get("non_target_spearman", [])},
                "correlation_scope": "matched_facet_condition",
                "conditioning_variable": "condition_id",
                "target_rho_threshold": TARGET_RHO_THRESHOLD,
                "target_rho_pass": target_rho is not None and target_rho >= TARGET_RHO_THRESHOLD,
                "same_domain_non_target": {
                    **same_domain,
                    "arm_id": "same_domain",
                    "filtering_authority": True,
                },
                "cross_domain_non_target": {
                    **cross_domain,
                    "arm_id": "cross_domain",
                    "filtering_authority": True,
                },
            },
            "per_condition_metrics": {
                condition_id: {
                    "condition_id": condition_id,
                    "arm_id": condition_rows[condition_id].get("arm_id"),
                    "group_id": condition_rows[condition_id].get("group_id"),
                    "rho": rhos[condition_id],
                    "citc": _pearson(
                        condition_wide[condition_id][item_id],
                        condition_wide[condition_id][same_items].sum(axis=1) - condition_wide[condition_id][item_id],
                    ),
                    "filtering_authority": condition_id == "target",
                }
                for condition_id in condition_rows
            },
            "citc_pass": citc.get("r") is not None and citc["r"] >= CITC_REVISION_THRESHOLD,
            "target_rho_pass": target_rho is not None and target_rho >= TARGET_RHO_THRESHOLD,
            "same_domain_vts_pass": same_vts is not None and same_vts >= SAME_DOMAIN_VTS_THRESHOLD,
            "cross_domain_vts_pass": cross_vts is not None and cross_vts >= CROSS_DOMAIN_VTS_THRESHOLD,
        }
        metric = metrics[item_id]
        metric["passes"] = bool(metric["citc_pass"] and metric["target_rho_pass"] and metric["same_domain_vts_pass"] and metric["cross_domain_vts_pass"])
        metric["qualified"] = metric["passes"]
    return metrics


def _target_option_gradient(
    *,
    target_long: pd.DataFrame,
    item: Mapping[str, Any],
    target_scores: pd.Series,
) -> dict[str, Any]:
    """Summarize target-arm facet means by scored option and adjacent failures."""

    item_id = str(item["item_id"])
    frame = target_long[target_long["item_id"] == item_id].copy()
    scoring_key = {str(key): float(value) for key, value in (item.get("scoring_key") or {}).items()}
    result_options: list[dict[str, Any]] = []
    means: dict[str, float | None] = {}
    ordered_scoring = sorted(scoring_key.items(), key=lambda pair: (pair[1], pair[0]))
    for option_id, option_score in ordered_scoring:
        ids = frame.loc[frame["selected_option_id"].astype(str) == option_id, "matched_subject_id"]
        values = target_scores.reindex(ids).dropna()
        mean = float(values.mean()) if len(values) else None
        means[option_id] = mean
        result_options.append({"option_id": option_id, "score": option_score, "n": int(len(values)), "target_facet_mean": mean, "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else None})
    failures: list[dict[str, Any]] = []
    option_ids = [option_id for option_id, _ in ordered_scoring]
    estimable = True
    for option_id in option_ids:
        if means[option_id] is None:
            estimable = False
    for lower, higher in zip(option_ids, option_ids[1:]):
        low_mean, high_mean = means[lower], means[higher]
        passed = low_mean is not None and high_mean is not None and low_mean < high_mean
        if not passed:
            failures.append({"lower_option_id": lower, "higher_option_id": higher, "lower_mean": low_mean, "higher_mean": high_mean, "direction": "加强高等级选项目标线索，或削弱低等级选项非目标/过强目标线索"})
    return {"estimable": bool(estimable), "options": result_options, "adjacent_comparisons": [{"lower_option_id": lower, "higher_option_id": higher, "lower_mean": means[lower], "higher_mean": means[higher], "passes": means[lower] is not None and means[higher] is not None and means[lower] < means[higher]} for lower, higher in zip(option_ids, option_ids[1:])], "failed_adjacent_pairs": failures, "passes": bool(estimable and not failures), "filtering_authority": False}


def _matched_option_diagnostics(
    *,
    sjt_long: pd.DataFrame,
    items: Mapping[str, Mapping[str, Any]],
    item_order: Sequence[str],
    condition_scores: Mapping[str, pd.Series],
    gradients: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    condition_groups = flatten_matched_condition_groups(conditions)
    condition_metadata = {
        str(row.get("condition_id")): dict(row)
        for row in condition_groups
        if isinstance(row, Mapping)
    }
    condition_ids = tuple(condition_metadata)

    def mean_and_se(values: pd.Series) -> tuple[float | None, float | None]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None, None
        mean = float(numeric.mean())
        standard_error = (
            float(numeric.std(ddof=1) / np.sqrt(len(numeric)))
            if len(numeric) >= 2
            else None
        )
        return mean, standard_error

    def arm_difference(
        item_id: str,
        item: Mapping[str, Any],
        comparator_id: str,
    ) -> dict[str, Any]:
        target = sjt_long[
            (sjt_long["item_id"] == item_id)
            & (sjt_long["condition_id"] == "target")
        ][
            [
                "matched_subject_id",
                "selected_option_id",
                "score",
                "active_score",
            ]
        ].rename(
            columns={
                "selected_option_id": "target_option_id",
                "score": "target_item_score",
                "active_score": "target_facet_score",
            }
        )
        comparator = sjt_long[
            (sjt_long["item_id"] == item_id)
            & (sjt_long["condition_id"] == comparator_id)
        ][
            [
                "matched_subject_id",
                "selected_option_id",
                "score",
                "active_score",
            ]
        ].rename(
            columns={
                "selected_option_id": "comparator_option_id",
                "score": "comparator_item_score",
                "active_score": "comparator_facet_score",
            }
        )
        paired = target.merge(
            comparator,
            on="matched_subject_id",
            how="inner",
            validate="one_to_one",
        ).sort_values("matched_subject_id")
        score_sequences_matched = bool(
            len(paired) == len(target) == len(comparator)
            and np.allclose(
                paired["target_facet_score"].to_numpy(dtype=float),
                paired["comparator_facet_score"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
        )
        estimable = bool(
            score_sequences_matched
            and len(paired) >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N
            and paired["target_facet_score"].nunique(dropna=True) > 1
        )
        ordered_options = sorted(
            (
                (str(option_id), float(score))
                for option_id, score in (item.get("scoring_key") or {}).items()
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        high_option_ids = {
            option_id for option_id, _ in ordered_options[-2:]
        }

        def paired_summary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
            group_n = len(frame)
            target_mean = (
                float(frame["target_item_score"].mean()) if group_n else None
            )
            comparator_mean = (
                float(frame["comparator_item_score"].mean()) if group_n else None
            )
            option_rows: list[dict[str, Any]] = []
            for option_id, option_score in ordered_options:
                target_count = int(
                    (frame["target_option_id"].astype(str) == option_id).sum()
                )
                comparator_count = int(
                    (frame["comparator_option_id"].astype(str) == option_id).sum()
                )
                target_rate = float(target_count / group_n) if group_n else None
                comparator_rate = (
                    float(comparator_count / group_n) if group_n else None
                )
                option_rows.append(
                    {
                        "option_id": option_id,
                        "score": option_score,
                        "target_count": target_count,
                        "target_rate": target_rate,
                        "comparator_count": comparator_count,
                        "comparator_rate": comparator_rate,
                        "target_minus_comparator_rate": (
                            target_rate - comparator_rate
                            if target_rate is not None and comparator_rate is not None
                            else None
                        ),
                    }
                )
            target_high_rate = (
                float(frame["target_option_id"].isin(high_option_ids).mean())
                if group_n
                else None
            )
            comparator_high_rate = (
                float(frame["comparator_option_id"].isin(high_option_ids).mean())
                if group_n
                else None
            )
            return {
                "score_band": label,
                "group_n": group_n,
                "target_mean_item_score": target_mean,
                "comparator_mean_item_score": comparator_mean,
                "target_minus_comparator_mean_item_score": (
                    target_mean - comparator_mean
                    if target_mean is not None and comparator_mean is not None
                    else None
                ),
                "target_high_option_rate": target_high_rate,
                "comparator_high_option_rate": comparator_high_rate,
                "target_minus_comparator_high_option_rate": (
                    target_high_rate - comparator_high_rate
                    if target_high_rate is not None
                    and comparator_high_rate is not None
                    else None
                ),
                "options": option_rows,
            }

        score_bands: list[dict[str, Any]] = []
        if estimable:
            try:
                paired = paired.copy()
                paired["score_band"] = pd.qcut(
                    paired["target_facet_score"],
                    q=3,
                    labels=["low", "middle", "high"],
                    duplicates="drop",
                )
                labels = {
                    str(value)
                    for value in paired["score_band"].dropna().astype(str).unique()
                }
                if labels == {"low", "middle", "high"}:
                    score_bands = [
                        paired_summary(
                            paired[paired["score_band"].astype(str) == label],
                            label,
                        )
                        for label in ("low", "middle", "high")
                    ]
                else:
                    estimable = False
            except (ValueError, TypeError):
                estimable = False

        overall = paired_summary(paired, "all")
        overall.update(
            {
                "same_option_count": int(
                    (
                        paired["target_option_id"].astype(str)
                        == paired["comparator_option_id"].astype(str)
                    ).sum()
                ),
                "same_option_rate": (
                    float(
                        (
                            paired["target_option_id"].astype(str)
                            == paired["comparator_option_id"].astype(str)
                        ).mean()
                    )
                    if len(paired)
                    else None
                ),
            }
        )
        metadata = condition_metadata.get(comparator_id) or {}
        return {
            "comparison_id": f"target_vs_{comparator_id}",
            "target_condition_id": "target",
            "comparator_condition_id": comparator_id,
            "comparator_role": metadata.get("role"),
            "comparator_arm_id": metadata.get("arm_id"),
            "comparator_group_id": metadata.get("group_id"),
            "comparator_dimension_id": metadata.get("dimension_id"),
            "comparator_facet_name": metadata.get("facet_name"),
            "matched_subject_count": len(paired),
            "score_sequences_matched": score_sequences_matched,
            "estimable": estimable,
            "reason": (
                None
                if estimable
                else "匹配分数序列、样本量或分数方差不足；差异仅展示，不得用于返修推断"
            ),
            "overall": overall,
            "by_score_band": score_bands,
            "filtering_authority": False,
            "diagnostic_use": "localization_only",
        }

    diagnostics: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item_id in item_order:
        item = items[item_id]
        item_diag: dict[str, Any] = {
            "version": OPTION_CHOICE_DIAGNOSTICS_VERSION,
            "filtering_authority": False,
            "all": None,
            "by_condition": [],
            "target_option_gradient": gradients[item_id],
            "arm_difference_diagnostics": {
                "filtering_authority": False,
                "diagnostic_use": "localization_only",
                "comparisons": [
                    arm_difference(item_id, item, condition_id)
                    for condition_id in condition_ids
                    if condition_id != "target"
                ],
            },
        }
        all_group = sjt_long[sjt_long["item_id"] == item_id]
        condition_option_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for scope, condition_id, group in [("all", None, all_group), *[("condition", cid, sjt_long[(sjt_long["item_id"] == item_id) & (sjt_long["condition_id"] == cid)]) for cid in condition_ids]]:
            counts = group["selected_option_id"].astype(str).value_counts()
            group_n = len(group)
            options = []
            metadata = condition_metadata.get(str(condition_id)) or {}
            for option_id, score in sorted(
                (item.get("scoring_key") or {}).items(),
                key=lambda pair: (float(pair[1]), str(pair[0])),
            ):
                option_id = str(option_id)
                count = int(counts.get(option_id, 0))
                selected_scores = group.loc[
                    group["selected_option_id"].astype(str) == option_id,
                    "active_score",
                ]
                facet_mean, facet_standard_error = mean_and_se(selected_scores)
                row = {
                    "option_id": option_id,
                    "score": float(score),
                    "selection_count": count,
                    "selection_rate": float(count / group_n) if group_n else None,
                    "facet_mean": facet_mean,
                    "facet_standard_error": facet_standard_error,
                }
                options.append(row)
                rows.append({"item_id": item_id, "grouping_scope": scope, "condition_id": condition_id, "arm_id": metadata.get("arm_id") if condition_id is not None else None, "group_id": metadata.get("group_id") if condition_id is not None else None, "matched_group": "all", "construct_role": metadata.get("role") if condition_id is not None else "pooled_matched_active_facet", "facet_id": metadata.get("dimension_id") if condition_id is not None else None, "group_n": group_n, "option_id": option_id, "option_score": float(score), "selection_count": count, "selection_rate": row["selection_rate"], "facet_mean": facet_mean, "facet_standard_error": facet_standard_error, "estimable": group_n >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N, "filtering_authority": False})
                if condition_id is not None:
                    condition_option_rows.setdefault(str(condition_id), {})[option_id] = {
                        "n": count,
                        "mean_score": facet_mean,
                        "standard_error": facet_standard_error,
                    }
            group_result = {
                "condition_id": condition_id,
                "arm_id": metadata.get("arm_id") if condition_id is not None else None,
                "group_id": metadata.get("group_id") if condition_id is not None else None,
                "construct_role": metadata.get("role") if condition_id is not None else "pooled_matched_active_facet",
                "facet_id": metadata.get("dimension_id") if condition_id is not None else None,
                "facet_name": metadata.get("facet_name") if condition_id is not None else None,
                "group_n": group_n,
                "options": options,
                "estimable": group_n >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N,
                "reason": (
                    None
                    if group_n >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N
                    else "该条件组少于10人；频率仅展示，不得用于返修推断"
                ),
            }
            if scope == "all":
                item_diag["all"] = group_result
            else:
                item_diag["by_condition"].append(group_result)
        paired_rows: list[dict[str, Any]] = []
        for option_id, score in sorted(
            ((str(key), float(value)) for key, value in (item.get("scoring_key") or {}).items()),
            key=lambda pair: (pair[1], pair[0]),
        ):
            target = condition_option_rows.get("target", {}).get(option_id, {})
            same_groups = [
                condition_option_rows.get(condition_id, {}).get(option_id, {})
                for condition_id in condition_ids
                if condition_metadata.get(condition_id, {}).get("arm_id") == "same_domain"
            ]
            cross_groups = [
                condition_option_rows.get(condition_id, {}).get(option_id, {})
                for condition_id in condition_ids
                if condition_metadata.get(condition_id, {}).get("arm_id") == "cross_domain"
            ]
            same = next((row for row in same_groups if row), {})
            cross = next((row for row in cross_groups if row), {})
            paired_rows.append(
                {
                    "option_id": option_id,
                    "score": score,
                    "target_n": target.get("n"),
                    "target_mean_score": target.get("mean_score"),
                    "target_standard_error": target.get("standard_error"),
                    "same_domain_n": same.get("n"),
                    "same_domain_mean_score": same.get("mean_score"),
                    "same_domain_standard_error": same.get("standard_error"),
                    "cross_domain_n": cross.get("n"),
                    "cross_domain_mean_score": cross.get("mean_score"),
                    "cross_domain_standard_error": cross.get("standard_error"),
                    "same_domain_groups": same_groups,
                    "cross_domain_groups": cross_groups,
                    "filtering_authority": False,
                }
            )
        item_diag["option_score_comparisons"] = paired_rows
        diagnostics[item_id] = item_diag
    return diagnostics, pd.DataFrame(rows)


def _apply_matched_quality(
    item_statistics: dict[str, dict[str, Any]],
    virtual_metrics: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, Mapping[str, Any]],
    *,
    item_order: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    rows = []
    for item_id in item_order:
        item = item_statistics[item_id]
        metric = virtual_metrics[item_id]
        specificity = metric["virtual_target_specificity"]
        same = specificity["same_domain_non_target"]
        cross = specificity["cross_domain_non_target"]
        failed = []
        if not metric["citc_pass"]: failed.append("CITC")
        if not metric["target_rho_pass"]: failed.append("目标rho")
        if not metric["same_domain_vts_pass"]: failed.append("同域VTS")
        if not metric["cross_domain_vts_pass"]: failed.append("跨域VTS")
        qualified = bool(metric["passes"])
        item["target_option_gradient"] = _json_safe(gradients[item_id])
        item["virtual_screening_metrics"] = _json_safe(metric)
        item["qualification"] = {"citc_pass": metric["citc_pass"], "target_rho_pass": metric["target_rho_pass"], "same_domain_vts_pass": metric["same_domain_vts_pass"], "cross_domain_vts_pass": metric["cross_domain_vts_pass"], "qualified": qualified}
        item["quality_evaluation"] = {"version": MEASUREMENT_EVALUATION_VERSION, "quality_grade": "acceptable" if qualified else "needs_revision", "recommendation": "retain" if qualified else "revise", "facet_citc": metric["facet_citc"], "virtual_target_specificity": specificity, "per_condition_metrics": metric.get("per_condition_metrics") or {}, "diagnostic_flags": failed, "failed_gates": failed, "target_option_gradient_failed": not gradients[item_id].get("passes", False), "decision_rule": "资格仅使用目标臂CITC、三臂rho和两类VTS；每个非目标facet group单独估计并取臂内最大带符号rho；目标组选项梯度只触发返修。"}
        rows.append({"item_id": item_id, "metric_scope": "matched_conditions", "filtering_authority": True, "correlation_method": "ordinary_spearman", "conditioning_variable": "condition_id", "quality_grade": item["quality_evaluation"]["quality_grade"], "recommendation": item["quality_evaluation"]["recommendation"], "citc_r": metric["facet_citc"].get("r"), "rho_target": specificity["rho_target"].get("rho"), "rho_same_domain": specificity["rho_same_domain"].get("rho"), "rho_cross_domain": specificity["rho_cross_domain"].get("rho"), "same_domain_max_non_target_rho": same.get("max_non_target_rho"), "cross_domain_max_non_target_rho": cross.get("max_non_target_rho"), "same_domain_group_count": len(same.get("non_target_spearman") or []), "cross_domain_group_count": len(cross.get("non_target_spearman") or []), "same_domain_vts": same.get("specificity_margin"), "cross_domain_vts": cross.get("specificity_margin"), "failed_gates": "；".join(failed), "target_option_gradient_pass": gradients[item_id].get("passes"), "target_option_gradient_estimable": gradients[item_id].get("estimable")})
    return item_statistics, pd.DataFrame(rows)


def _run_matched_condition_analysis(
    *,
    state: PSJTState,
    manifest_path: Path,
    sjt_path: Path,
    response_manifest: Mapping[str, Any],
    item_order: list[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondents: int,
) -> dict[str, Any]:
    """Persist the active fixed-three-arm, nested-facet-group screening analysis."""

    if response_manifest.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION:
        raise ValueError(f"匹配 facet 分析需要 schema_version={MATCHED_CONDITION_SCHEMA_VERSION}")
    conditions = response_manifest.get("conditions")
    if not isinstance(conditions, list) or {str(row.get("condition_id")) for row in conditions if isinstance(row, Mapping)} != set(MATCHED_CONDITION_IDS):
        raise ValueError("作答 manifest 缺少固定三臂匹配条件")
    condition_groups = flatten_matched_condition_groups(conditions)
    condition_ids = tuple(str(row.get("condition_id")) for row in condition_groups)
    if "target" not in condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("作答 manifest 的 facet groups 缺少唯一 target 或存在重复 condition_id")
    expected_per_condition = response_manifest.get("sample_size_per_condition")
    if not isinstance(expected_per_condition, int) or expected_respondents != expected_per_condition * len(condition_ids):
        raise ValueError("facet group 作答人数与 sample_size_per_condition 不一致")
    sjt_records = _read_jsonl(sjt_path)
    sjt_long, condition_wide, condition_scores = _score_matched_sjt(
        sjt_records,
        item_order=item_order,
        items=items,
        expected_respondent_count=expected_per_condition,
        condition_ids=condition_ids,
    )
    target_retest = response_manifest.get("target_form_retest")
    if not isinstance(target_retest, Mapping):
        raise ValueError("作答 manifest 缺少 target 整卷重测元数据")
    target_retest_path_value = target_retest.get("path")
    if (
        not isinstance(target_retest_path_value, str)
        or not Path(target_retest_path_value).is_file()
    ):
        raise ValueError("作答 manifest 缺少有效 target 整卷重测文件")
    raw_administration_ids = target_retest.get("administration_ids")
    if (
        not isinstance(raw_administration_ids, list)
        or not raw_administration_ids
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_administration_ids
        )
    ):
        raise ValueError("target 整卷重测缺少有效 administration_ids")
    target_retest_path = Path(target_retest_path_value)
    target_retest_long = _score_target_form_retests(
        _read_jsonl(target_retest_path),
        item_order=item_order,
        items=items,
        expected_respondent_count=expected_per_condition,
        administration_ids=raw_administration_ids,
    )
    virtual_metrics = _matched_item_metrics(
        item_order=item_order,
        items=items,
        condition_wide=condition_wide,
        condition_scores=condition_scores,
        conditions=conditions,
    )
    item_statistics, item_frame, _ = _item_statistics(
        sjt_long,
        pd.concat([condition_wide[c].assign(condition_id=c) for c in condition_ids]).drop(columns=["condition_id"]),
        item_order=item_order,
        items=items,
    )
    item_statistics, _ = _enrich_item_quality(item_statistics, item_order=item_order)
    target_long = sjt_long[sjt_long["condition_id"] == "target"]
    gradients = {
        item_id: _target_option_gradient(target_long=target_long, item=items[item_id], target_scores=condition_scores["target"])
        for item_id in item_order
    }
    item_statistics, item_quality_frame = _apply_matched_quality(
        item_statistics,
        virtual_metrics,
        gradients,
        item_order=item_order,
    )
    option_diagnostics, option_frame = _matched_option_diagnostics(
        sjt_long=sjt_long,
        items=items,
        item_order=item_order,
        condition_scores=condition_scores,
        gradients=gradients,
        conditions=conditions,
    )
    for item_id in item_order:
        item_statistics[item_id]["option_choice_diagnostics"] = option_diagnostics[item_id]
    target_wide = condition_wide["target"]
    scale_statistics = _scale_statistics(target_wide, items)
    analysis_round = int(state.get("psychometric_analysis_round") or 0) + 1
    respondent_rows = []
    for condition_id in condition_ids:
        frame = condition_wide[condition_id]
        scores = condition_scores[condition_id]
        for matched_subject_id in frame.index:
            group = next((entry for entry in condition_groups if entry.get("condition_id") == condition_id), {})
            row = {"respondent_id": f"{condition_id}-{matched_subject_id}", "condition_id": condition_id, "arm_id": group.get("arm_id"), "group_id": group.get("group_id"), "matched_subject_id": matched_subject_id, "active_score": float(scores.loc[matched_subject_id]), "sjt_total_score": float(frame.loc[matched_subject_id].mean())}
            respondent_rows.append(row)
    respondent_scores = pd.DataFrame(respondent_rows)
    target_rhos = [virtual_metrics[item_id]["virtual_target_specificity"]["rho_target"].get("rho") for item_id in item_order]
    citcs = [virtual_metrics[item_id]["facet_citc"].get("r") for item_id in item_order]
    same_vts = [virtual_metrics[item_id]["virtual_target_specificity"]["same_domain_non_target"].get("specificity_margin") for item_id in item_order]
    cross_vts = [virtual_metrics[item_id]["virtual_target_specificity"]["cross_domain_non_target"].get("specificity_margin") for item_id in item_order]
    validity_statistics = _json_safe({
        "evidence_scope": "exploratory_virtual_matched_facet_screening",
        "psychometric_analysis_round": analysis_round,
        "sampling_design": "matched_facet_conditions",
        "condition_ids": list(condition_ids),
        "arm_ids": list(MATCHED_CONDITION_IDS),
        "group_count": len(condition_ids),
        "sample_size_per_condition": expected_per_condition,
        "item_metrics": virtual_metrics,
        "summary": {"median_target_rho": float(np.nanmedian(target_rhos)) if any(value is not None for value in target_rhos) else None, "median_citc": float(np.nanmedian(citcs)) if any(value is not None for value in citcs) else None, "minimum_same_domain_vts": min((value for value in same_vts if value is not None), default=None), "minimum_cross_domain_vts": min((value for value in cross_vts if value is not None), default=None)},
        "interpretation": "固定三个顶层臂复用完全匹配的分数序列；每个facet group单独计算普通Spearman，臂级VTS取非目标组最大带符号rho，CITC过滤权仅属于target臂。",
    })
    recommendation_counts = {"retain": 0, "revise": 0, "remove": 0}
    for item in item_statistics.values():
        recommendation_counts[item["quality_evaluation"]["recommendation"]] += 1
    qualified_count = recommendation_counts["retain"]
    measurement_evaluation = _json_safe({
        "version": MEASUREMENT_EVALUATION_VERSION,
        "psychometric_analysis_round": analysis_round,
        "evidence_scope": "exploratory_virtual_screening_evidence",
        "overall_status": "development_ready" if qualified_count == len(item_order) else "development_ready_with_revision",
        "item_recommendation_counts": recommendation_counts,
        "reliability": {"overall_grade": "exploratory_virtual_screening", "cronbach_alpha": scale_statistics.get("cronbach_alpha"), "median_target_citc": validity_statistics["summary"]["median_citc"]},
        "validity": {"overall_grade": "exploratory_virtual_screening", "virtual_target_specificity": validity_statistics["summary"], "conditioning": {"variable": "condition_id", "method": "matched_facet_arms", "filtering_authority": True}},
        "interpretation": "四项资格门槛使用目标臂CITC、三臂rho和两类VTS；非目标facet group独立估计并取臂内最大带符号rho，目标组选项梯度仅为返修触发器。",
    })
    output_dir = manifest_path.parent / "psychometrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_matched_condition_sjt_responses.csv"
    scored_target_retest_path = (
        output_dir / "scored_target_form_retest_sjt_responses.csv"
    )
    respondent_path = output_dir / "respondent_scores.csv"
    item_path = output_dir / "item_statistics.csv"
    quality_path = output_dir / "item_quality.csv"
    option_path = output_dir / "option_statistics.csv"
    option_json_path = output_dir / "option_choice_diagnostics.json"
    scale_path = output_dir / "scale_statistics.json"
    validity_path = output_dir / "virtual_screening_metrics.json"
    measurement_path = output_dir / "measurement_evaluation.json"
    analysis_path = output_dir / "analysis_manifest.json"
    _write_csv_atomic(scored_path, sjt_long)
    _write_csv_atomic(scored_target_retest_path, target_retest_long)
    _write_csv_atomic(respondent_path, respondent_scores)
    _write_csv_atomic(item_path, item_frame)
    _write_csv_atomic(quality_path, item_quality_frame)
    _write_csv_atomic(option_path, option_frame)
    _write_json_atomic(option_json_path, {"schema_version": 3, "version": OPTION_CHOICE_DIAGNOSTICS_VERSION, "filtering_authority": False, "items": option_diagnostics})
    _write_json_atomic(scale_path, scale_statistics)
    _write_json_atomic(validity_path, validity_statistics)
    _write_json_atomic(measurement_path, measurement_evaluation)
    option_order_path = response_manifest.get("option_order_path")
    if not isinstance(option_order_path, str) or not Path(option_order_path).is_file():
        raise ValueError("作答 manifest 缺少有效 option_order_path")
    output_files = {"scored_matched_condition_sjt_responses": str(scored_path.resolve()), "scored_target_form_retest_sjt_responses": str(scored_target_retest_path.resolve()), "respondent_scores": str(respondent_path.resolve()), "item_statistics": str(item_path.resolve()), "item_quality": str(quality_path.resolve()), "option_statistics": str(option_path.resolve()), "option_choice_diagnostics": str(option_json_path.resolve()), "scale_statistics": str(scale_path.resolve()), "virtual_screening_metrics": str(validity_path.resolve()), "measurement_evaluation": str(measurement_path.resolve()), "analysis_manifest": str(analysis_path.resolve())}
    reference_questionnaires = response_manifest.get("reference_questionnaires") or {}
    if isinstance(reference_questionnaires, Mapping):
        ipip_reference = reference_questionnaires.get("ipip_neo")
        if isinstance(ipip_reference, Mapping) and isinstance(ipip_reference.get("path"), str) and Path(ipip_reference["path"]).is_file():
            output_files["ipip_neo_responses"] = str(Path(ipip_reference["path"]).resolve())
        neo_reference = reference_questionnaires.get("neo_ffi")
        if isinstance(neo_reference, Mapping) and isinstance(neo_reference.get("path"), str) and Path(neo_reference["path"]).is_file():
            output_files["neo_ffi_responses"] = str(Path(neo_reference["path"]).resolve())
    analysis_manifest = {"schema_version": MATCHED_CONDITION_SCHEMA_VERSION, "status": "completed", "formula_version": PSYCHOMETRIC_FORMULA_VERSION, "evaluation_version": MEASUREMENT_EVALUATION_VERSION, "psychometric_analysis_round": analysis_round, "run_id": state["run_id"], "item_bank_id": state["item_bank_id"], "item_bank_version": state["item_bank_version"], "item_bank_fingerprint": state.get("item_bank_fingerprint"), "sample_size": expected_respondents, "sample_size_per_condition": expected_per_condition, "condition_count": 3, "group_count": len(condition_ids), "condition_ids": list(condition_ids), "arm_ids": list(MATCHED_CONDITION_IDS), "sampling_design": "matched_facet_conditions", "conditions": deepcopy(conditions), "persona_mode": PERSONA_MODE_SCORE_PROFILE, "prompt_version": response_manifest.get("prompt_version"), "score_prompt_version": response_manifest.get("score_prompt_version"), "generator_version": response_manifest.get("generator_version"), "virtual_sample_config": deepcopy(response_manifest.get("virtual_sample_config") or {}), "item_count": len(item_order), "qualified_item_count": qualified_count, "qualification_rate": qualified_count / len(item_order), "criteria": {"iteration_gates": {"facet_citc_minimum": CITC_REVISION_THRESHOLD, "target_rho_minimum": TARGET_RHO_THRESHOLD, "same_domain_vts_minimum": SAME_DOMAIN_VTS_THRESHOLD, "cross_domain_vts_minimum": CROSS_DOMAIN_VTS_THRESHOLD, "condition_method": "fixed_three_arms_nested_facet_groups", "filtering_authority": True}, "target_option_gradient": {"filtering_authority": False, "repair_trigger": True}}, "formulas": {"facet_citc": "Pearson(target arm item, target arm same-facet remaining score sum)", "rho_target": "Spearman(Y_target, z_target)", "rho_group": "Spearman(Y_group, z_group), one group per non-target facet", "same_domain_vts": "rho_target - MAX(signed rho of same-domain facet groups)", "cross_domain_vts": "rho_target - MAX(signed rho of cross-domain facet groups)", "target_form_retest": "same target persona and item set under a second balanced option order; used only by whole-form stability"}, "input_files": {"response_manifest": {"path": str(manifest_path.resolve()), "sha256": _file_sha256(manifest_path)}, "sjt_responses": {"path": str(sjt_path.resolve()), "sha256": _file_sha256(sjt_path)}, "target_form_retest_responses": {"path": str(target_retest_path.resolve()), "sha256": _file_sha256(target_retest_path)}, "score_profiles": {"path": str(Path(response_manifest["score_profiles_path"]).resolve()), "sha256": _file_sha256(Path(response_manifest["score_profiles_path"]))}, "option_orders": {"path": str(Path(option_order_path).resolve()), "sha256": _file_sha256(Path(option_order_path))}}, "output_files": output_files, "completed_at": utc_timestamp()}
    _write_json_atomic(analysis_path, analysis_manifest)
    test_statistics = {**scale_statistics, "sampling_design": "matched_facet_conditions", "condition_count": 3, "group_count": len(condition_ids), "condition_ids": list(condition_ids), "arm_ids": list(MATCHED_CONDITION_IDS), "sample_size_per_condition": expected_per_condition, "qualified_item_count": qualified_count, "qualification_rate": qualified_count / len(item_order), "qualification_criteria": analysis_manifest["criteria"], "virtual_screening_metrics": validity_statistics, "measurement_evaluation": measurement_evaluation, "reference_questionnaires": _json_safe(reference_questionnaires), "output_files": output_files, "formula_version": PSYCHOMETRIC_FORMULA_VERSION, "evaluation_version": MEASUREMENT_EVALUATION_VERSION, "psychometric_analysis_round": analysis_round, "option_choice_diagnostics_version": OPTION_CHOICE_DIAGNOSTICS_VERSION}
    state_update = {"item_statistics": item_statistics, "test_statistics": _json_safe(test_statistics), "psychometric_analysis_round": analysis_round, "virtual_analysis_reconfiguration_reason": None}
    from sjt_system.evaluation.round_results import build_psychometric_round_result
    state_update["psychometric_round_result"] = build_psychometric_round_result({**state, **state_update})
    return {"state_update": state_update, "summary": f"已完成固定三臂、多 facet group 匹配分析：每组{expected_per_condition}人、共{len(condition_ids)}组、{len(item_order)}道题，保留{qualified_count}题，待返修{recommendation_counts['revise']}题。"}


def _option_frequency_group(
    item_responses: pd.DataFrame,
    *,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    scoring_key = item.get("scoring_key") or {}
    option_metadata = {
        str(row.get("option_id")): row
        for row in item.get("response_options") or []
        if isinstance(row, Mapping) and row.get("option_id") is not None
    }
    counts = item_responses["selected_option_id"].astype(str).value_counts()
    group_n = int(len(item_responses))
    options: list[dict[str, Any]] = []
    for raw_option_id, raw_score in scoring_key.items():
        option_id = str(raw_option_id)
        count = int(counts.get(option_id, 0))
        metadata = option_metadata.get(option_id) or {}
        options.append(
            {
                "option_id": option_id,
                "behavioral_level": metadata.get("behavioral_level"),
                "score": float(raw_score),
                "selection_count": count,
                "selection_rate": (
                    float(count / group_n) if group_n > 0 else None
                ),
            }
        )
    return {"group_n": group_n, "options": options}


def _scale_statistics(
    sjt_wide: pd.DataFrame,
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    respondent_mean = sjt_wide.mean(axis=1)
    overall_alpha = cronbach_alpha(sjt_wide.to_numpy())
    dimensions: dict[str, Any] = {}
    dimension_ids = list(
        dict.fromkeys(
            str(item.get("target_dimension_id"))
            for item in items.values()
            if item.get("target_dimension_id") is not None
        )
    )
    for dimension_id in dimension_ids:
        columns = [
            item_id
            for item_id, item in items.items()
            if str(item.get("target_dimension_id")) == dimension_id
        ]
        frame = sjt_wide[columns]
        scores = frame.mean(axis=1)
        dimensions[dimension_id] = {
            "item_count": len(columns),
            "item_ids": columns,
            "mean": float(scores.mean()),
            "standard_deviation": float(scores.std(ddof=1)),
            "cronbach_alpha": cronbach_alpha(frame.to_numpy()),
        }
    return _json_safe(
        {
            "sample_size": len(sjt_wide),
            "item_count": len(sjt_wide.columns),
            "score_type": "mean_item_score",
            "score_minimum": float(respondent_mean.min()),
            "score_maximum": float(respondent_mean.max()),
            "score_mean": float(respondent_mean.mean()),
            "score_standard_deviation": float(
                respondent_mean.std(ddof=1)
            ),
            "cronbach_alpha": overall_alpha,
            "dimensions": dimensions,
        }
    )


def _benjamini_hochberg(
    values: Sequence[float | None],
) -> list[float | None]:
    """Adjust a family of p-values while preserving missing positions."""

    values = list(values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda pair: pair[1])
    running_minimum = 1.0
    count = len(ordered)
    for rank_from_end in range(count, 0, -1):
        index, p_value = ordered[rank_from_end - 1]
        candidate = min(1.0, p_value * count / rank_from_end)
        running_minimum = min(running_minimum, candidate)
        adjusted[index] = float(running_minimum)
    return adjusted


def _reliability_grade(value: float | None) -> str:
    if value is None:
        return "insufficient_evidence"
    if value >= 0.80:
        return "strong"
    if value >= 0.70:
        return "acceptable"
    if value >= 0.60:
        return "weak"
    return "inadequate"


def _enrich_item_quality(
    item_statistics: dict[str, dict[str, Any]],
    *,
    item_order: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Add facet-level discrimination and item-function decisions."""

    citc_adjusted = _benjamini_hochberg(
        item_statistics[item_id][
            "facet_corrected_item_total_correlation"
        ]["p_value"]
        for item_id in item_order
    )

    quality_rows: list[dict[str, Any]] = []
    for index, item_id in enumerate(item_order):
        item = item_statistics[item_id]
        item["facet_corrected_item_total_correlation"]["p_value_bh"] = (
            citc_adjusted[index]
        )

        citc = item["facet_corrected_item_total_correlation"]["r"]
        difficulty = item["difficulty"]
        option_statistics = item["option_statistics"]
        observed_option_count = sum(
            option["count"] > 0 for option in option_statistics.values()
        )
        effective_option_count = sum(
            option["rate"] >= MINIMUM_OPTION_RATE
            for option in option_statistics.values()
        )
        if citc is None:
            discrimination_rating = "insufficient_evidence"
        elif citc < 0:
            discrimination_rating = "poor"
        elif citc < CITC_REVISION_THRESHOLD:
            discrimination_rating = "weak"
        elif citc < CITC_MINIMUM_ACCEPTABLE:
            discrimination_rating = "acceptable_with_warning"
        elif citc >= CITC_STRONG:
            discrimination_rating = "strong"
        else:
            discrimination_rating = "acceptable"

        if 0.30 <= difficulty <= 0.75:
            distribution_rating = "strong"
        elif DIFFICULTY_LOWER_BOUND <= difficulty <= DIFFICULTY_UPPER_BOUND:
            distribution_rating = "acceptable"
        else:
            distribution_rating = "weak"

        if effective_option_count == len(option_statistics):
            option_rating = "strong"
        elif effective_option_count >= 3:
            option_rating = "acceptable_with_warning"
        else:
            option_rating = "weak"

        flags: list[str] = []
        if discrimination_rating == "poor":
            flags.append("区分方向错误或区分度过低")
        elif discrimination_rating == "weak":
            flags.append("区分度低于开发期建议值")
        elif discrimination_rating == "acceptable_with_warning":
            flags.append("分面内CITC为较弱支持性证据，不单独触发返修")
        if distribution_rating == "weak":
            flags.append("题目得分分布过于偏高或偏低")
        if option_rating == "weak":
            flags.append("实际仅有两个或更少选项发挥区分作用")
        elif option_rating == "acceptable_with_warning":
            flags.append("存在低频选项，需检查措辞而非自动删题")
        revision_needed = (
            discrimination_rating
            in {"poor", "weak", "insufficient_evidence"}
            or distribution_rating == "weak"
            or option_rating == "weak"
        )
        if revision_needed:
            recommendation = "revise"
            quality_grade = (
                "poor"
                if discrimination_rating == "poor"
                else "needs_revision"
            )
        else:
            recommendation = "retain"
            quality_grade = (
                "strong"
                if (
                    discrimination_rating == "strong"
                    and distribution_rating == "strong"
                    and option_rating == "strong"
                )
                else "acceptable"
            )

        evaluation = {
            "version": MEASUREMENT_EVALUATION_VERSION,
            "quality_grade": quality_grade,
            "recommendation": recommendation,
            "discrimination_rating": discrimination_rating,
            "distribution_rating": distribution_rating,
            "option_function_rating": option_rating,
            "observed_option_count": observed_option_count,
            "effective_option_count": effective_option_count,
            "diagnostic_flags": flags,
            "decision_rule": (
                "题项决策仅使用分面内CITC、标准化平均得分和"
                "有效选项数；单题与IPIP-NEO目标facet的相关不参与返修。"
            ),
        }
        item["quality_evaluation"] = _json_safe(evaluation)
        quality_rows.append(
            {
                "item_id": item_id,
                "dimension_id": item.get("dimension_id"),
                "quality_grade": quality_grade,
                "recommendation": recommendation,
                "citc_r": citc,
                "citc_p_bh": citc_adjusted[index],
                "difficulty": difficulty,
                "observed_option_count": observed_option_count,
                "effective_option_count": effective_option_count,
                "minimum_option_rate": item["minimum_option_rate"],
                "diagnostic_flags": "；".join(flags),
            }
        )

    return item_statistics, pd.DataFrame(quality_rows)


def evaluate_single_item_candidate(
    state: PSJTState,
    candidate_item: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """用正式基线的其余题目，评估一个候选题的四项单题指标。

    局部复测只替换候选题的作答记录；同一批 matched facet 被试、同一
    condition 分布和其余题目的作答保持不变。因此该结果可用于题目级
    “修改—复测”反馈，但不冒充整卷信效度报告。
    """

    manifest_path: Path | None = None
    manifest: Mapping[str, Any] | None = None
    attempted_refs: list[str] = []
    for field in (
        "virtual_response_data_ref",
        "previous_virtual_response_data_ref",
    ):
        reference = state.get(field)
        if not isinstance(reference, str) or not reference:
            continue
        attempted_refs.append(f"{field}={reference}")
        candidate_manifest_path = Path(reference).resolve()
        if not candidate_manifest_path.is_file():
            continue
        candidate_manifest = _read_json(candidate_manifest_path)
        if candidate_manifest.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION:
            continue
        manifest_path = candidate_manifest_path
        manifest = candidate_manifest
        break
    if manifest_path is None or manifest is None:
        if not attempted_refs:
            raise ValueError(
                "单题局部指标计算前缺少 virtual_response_data_ref；"
                "当前和上一版基线引用均不存在"
            )
        raise ValueError(
            "单题局部指标找不到可用的 matched-condition 作答 manifest："
            + "；".join(attempted_refs)
        )
    item_id = str(candidate_item.get("item_id") or "")
    if not item_id:
        raise ValueError("单题候选缺少 item_id")
    if not isinstance(candidate_item.get("version"), int) or isinstance(
        candidate_item.get("version"), bool
    ):
        raise ValueError("单题候选缺少有效 version")
    item_order, items = _prepare_item_contract(state)
    if item_id not in items:
        raise ValueError(f"单题候选不在当前冻结题库中：{item_id}")
    items[item_id] = deepcopy(dict(candidate_item))
    baseline_records = _read_jsonl(manifest_path.parent / "sjt_responses.jsonl")
    replacement_records = [
        dict(record)
        for record in candidate_records
        if isinstance(record, Mapping)
        and str(record.get("item_id") or "") == item_id
    ]
    if not replacement_records:
        raise ValueError("单题局部复测没有返回候选题作答")
    combined_records = [
        record
        for record in baseline_records
        if str(record.get("item_id") or "") != item_id
    ] + replacement_records
    conditions = manifest.get("conditions")
    condition_groups = flatten_matched_condition_groups(conditions or [])
    condition_ids = tuple(str(row.get("condition_id")) for row in condition_groups)
    expected_per_condition = manifest.get("sample_size_per_condition")
    if not isinstance(expected_per_condition, int) or expected_per_condition < 1:
        raise ValueError("作答 manifest 缺少有效 sample_size_per_condition")
    sjt_long, condition_wide, condition_scores = _score_matched_sjt(
        combined_records,
        item_order=item_order,
        items=items,
        expected_respondent_count=expected_per_condition,
        condition_ids=condition_ids,
    )
    virtual_metrics = _matched_item_metrics(
        item_order=item_order,
        items=items,
        condition_wide=condition_wide,
        condition_scores=condition_scores,
        conditions=conditions,
    )
    item_statistics, _, _ = _item_statistics(
        sjt_long,
        pd.concat(
            [condition_wide[condition_id].assign(condition_id=condition_id)
             for condition_id in condition_ids]
        ).drop(columns=["condition_id"]),
        item_order=item_order,
        items=items,
    )
    item_statistics, _ = _enrich_item_quality(
        item_statistics,
        item_order=item_order,
    )
    target_long = sjt_long[sjt_long["condition_id"] == "target"]
    gradients = {
        current_item_id: _target_option_gradient(
            target_long=target_long,
            item=items[current_item_id],
            target_scores=condition_scores["target"],
        )
        for current_item_id in item_order
    }
    item_statistics, _ = _apply_matched_quality(
        item_statistics,
        virtual_metrics,
        gradients,
        item_order=item_order,
    )
    result = deepcopy(item_statistics[item_id])
    result["local_retest"] = {
        "status": "passed" if result.get("qualification", {}).get("qualified") else "failed",
        "filtering_authority": False,
        "source_manifest": str(manifest_path),
        "replaced_item_id": item_id,
        "replaced_item_version": candidate_item.get("version"),
        "candidate_response_count": len(replacement_records),
        "metric_scope": "single_item_candidate_against_current_bank",
    }
    return _json_safe(result)


def run_psychometric_analysis(
    state: PSJTState,
) -> dict[str, Any]:
    """Score saved responses, compute paper-aligned CTT metrics, and persist them."""

    manifest_path, sjt_path, neo_path, response_manifest = (
        _resolve_response_paths(state)
    )
    item_order, items = _prepare_item_contract(state)
    expected_respondents = response_manifest.get("sample_size")
    if (
        not isinstance(expected_respondents, int)
        or isinstance(expected_respondents, bool)
        or expected_respondents < 1
    ):
        raise ValueError("虚拟作答 manifest 缺少有效 sample_size")
    if response_manifest.get("schema_version") == MATCHED_CONDITION_SCHEMA_VERSION:
        return _run_matched_condition_analysis(
            state=state,
            manifest_path=manifest_path,
            sjt_path=sjt_path,
            response_manifest=response_manifest,
            item_order=item_order,
            items=items,
            expected_respondents=expected_respondents,
        )
    raise ValueError(
        "旧 tier/重复作答/条件残差化虚拟作答结果已失效；"
        f"请使用 schema_version={MATCHED_CONDITION_SCHEMA_VERSION} 的固定三臂、多 facet group 匹配协议重新作答"
    )

def run_saved_psychometric_analysis(
    manifest_path: str | Path,
    *,
    scoring_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a completed saved run without requiring an in-memory checkpoint."""

    resolved_manifest = Path(manifest_path).resolve()
    manifest = _read_json(resolved_manifest)
    resolved_snapshot = Path(
        scoring_snapshot_path
        or manifest.get("scoring_snapshot_path")
        or resolved_manifest.parent / "scoring_snapshot.json"
    ).resolve()
    if not resolved_snapshot.is_file():
        raise ValueError(
            "保存的虚拟作答缺少 scoring_snapshot.json，"
            "无法在脱离工作流 State 后恢复计分键"
        )
    snapshot = _read_json(resolved_snapshot)
    for field in (
        "run_id",
        "item_bank_id",
        "item_bank_version",
        "item_bank_fingerprint",
    ):
        if snapshot.get(field) != manifest.get(field):
            raise ValueError(
                f"scoring snapshot 的 {field} 与作答 manifest 不一致"
            )
    items = snapshot.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("scoring snapshot 没有冻结题目")

    state = create_initial_state("分析已保存的虚拟作答")
    state.update(
        {
            "run_id": manifest["run_id"],
            "item_bank_id": manifest["item_bank_id"],
            "item_bank_version": manifest["item_bank_version"],
            "item_bank_fingerprint": manifest["item_bank_fingerprint"],
            "frozen_item_bank": items,
            "virtual_response_data_ref": str(resolved_manifest),
            "virtual_response_item_bank_id": manifest["item_bank_id"],
            "virtual_response_item_bank_version": manifest[
                "item_bank_version"
            ],
        }
    )
    return run_psychometric_analysis(state)
