"""Select retained and reserve items from validated evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations, product
from math import prod
from pathlib import Path
from typing import Any

import pandas as pd

from sjt_system.authoring.context import text_similarity
from sjt_system.authoring.items import build_repair_tasks_from_findings
from sjt_system.authoring.budget import developed_candidate_ids
from sjt_system.evaluation.respondents import (
    MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE,
)
from sjt_system.evaluation.round_results import metric_scalar
from sjt_system.state import PSJTState
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.workflow.constants import PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS


_DISCRIMINATION_RANK = {
    "strong": 3,
    "acceptable": 2,
    "acceptable_with_warning": 1,
    "weak": 0,
    "marginal": 0,
    "insufficient_evidence": 0,
    "poor": -1,
}
MAX_COMBINATION_CANDIDATES = 50_000
_METRIC_TOLERANCE = 1e-9


def _uses_exploratory_virtual_evidence(state: Mapping[str, Any]) -> bool:
    config = state.get("virtual_sample_config")
    if not isinstance(config, Mapping):
        return False
    sample_size = config.get("sample_size")
    minimum_size = config.get(
        "automatic_selection_minimum_sample_size",
        MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE,
    )
    return (
        isinstance(sample_size, int)
        and not isinstance(sample_size, bool)
        and isinstance(minimum_size, int)
        and not isinstance(minimum_size, bool)
        and sample_size < minimum_size
    )


def _items_by_id(state: Mapping[str, Any]) -> tuple[list[str], dict[str, dict]]:
    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("题目筛选前缺少冻结题库")
    order: list[str] = []
    items: dict[str, dict] = {}
    for raw_item in frozen:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = deepcopy(dict(raw_item))
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("冻结题库包含缺少 item_id 的题目")
        if item_id in items:
            raise ValueError(f"冻结题库包含重复题目：{item_id}")
        order.append(item_id)
        items[item_id] = item
    return order, items


def _cell_id_for_item(
    item: Mapping[str, Any],
    blueprint: Mapping[str, Any],
) -> str | None:
    cell_id = item.get("blueprint_cell_id")
    if isinstance(cell_id, str) and cell_id:
        return cell_id
    dimension_id = item.get("target_dimension_id")
    matches = [
        str(cell.get("cell_id"))
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping)
        and cell.get("facet_id") == dimension_id
        and isinstance(cell.get("cell_id"), str)
    ]
    return matches[0] if len(matches) == 1 else None


def _quality_sort_key(
    item_id: str,
    statistics: Mapping[str, Any],
    order_index: Mapping[str, int],
) -> tuple[Any, ...]:
    quality = statistics["quality_evaluation"]
    citc = quality.get("facet_citc") or {}
    specificity = quality.get("virtual_target_specificity") or {}
    if citc or specificity:
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        target_rho = (specificity.get("target_spearman") or {}).get("rho")
        return (
            -float(citc.get("r"))
            if isinstance(citc.get("r"), (int, float))
            else float("inf"),
            -float(target_rho)
            if isinstance(target_rho, (int, float))
            else float("inf"),
            -float(same_domain.get("specificity_margin"))
            if isinstance(same_domain.get("specificity_margin"), (int, float))
            else float("inf"),
            -float(cross_domain.get("specificity_margin"))
            if isinstance(cross_domain.get("specificity_margin"), (int, float))
            else float("inf"),
            order_index[item_id],
        )
    difficulty = statistics.get("difficulty")
    corrected = (
        statistics.get("facet_corrected_item_total_correlation")
        or statistics.get("corrected_item_total_correlation")
        or {}
    )
    return (
        -_DISCRIMINATION_RANK.get(
            str(quality.get("discrimination_rating")), -2
        ),
        -int(quality.get("effective_option_count") or 0),
        abs(float(difficulty) - 0.50)
        if isinstance(difficulty, (int, float))
        else float("inf"),
        -float(corrected.get("r"))
        if isinstance(corrected.get("r"), (int, float))
        else float("inf"),
        order_index[item_id],
    )


def _cronbach_alpha(frame: pd.DataFrame) -> float | None:
    if frame.shape[1] < 2 or frame.shape[0] < 2:
        return None
    item_variances = frame.var(axis=0, ddof=1)
    total_variance = frame.sum(axis=1).var(ddof=1)
    if not isinstance(total_variance, (int, float)) or total_variance <= 0:
        return None
    count = frame.shape[1]
    return float(
        count / (count - 1)
        * (1 - float(item_variances.sum()) / float(total_variance))
    )


def _load_optimizer_dataset(
    statistics: Mapping[str, Any],
) -> dict[str, Any] | None:
    files = statistics.get("output_files")
    criterion = statistics.get("criterion")
    if not isinstance(files, Mapping) or not isinstance(criterion, Mapping):
        return None
    sjt_path = files.get("scored_sjt_responses")
    respondent_path = files.get("respondent_scores")
    target_code = criterion.get("neo_ffi_dimension")
    if (
        not isinstance(sjt_path, str)
        or not isinstance(respondent_path, str)
        or not isinstance(target_code, str)
        or not Path(sjt_path).is_file()
        or not Path(respondent_path).is_file()
    ):
        return None
    sjt = pd.read_csv(sjt_path)
    respondents = pd.read_csv(respondent_path)
    required_sjt = {"respondent_id", "item_id", "score"}
    if not required_sjt <= set(sjt.columns) or "respondent_id" not in respondents:
        return None
    target_column = f"neo_{target_code}"
    if target_column not in respondents:
        return None
    item_scores = sjt.pivot(
        index="respondent_id",
        columns="item_id",
        values="score",
    )
    neo_columns = [
        column for column in respondents.columns if str(column).startswith("neo_")
    ]
    criteria = respondents.set_index("respondent_id")[neo_columns]
    joined_ids = item_scores.index.intersection(criteria.index)
    return {
        "item_scores": item_scores.loc[joined_ids],
        "criteria": criteria.loc[joined_ids],
        "target_column": target_column,
        "label": statistics.get("persona_mode") or "primary",
    }


def _optimizer_datasets(
    test_statistics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(test_statistics, Mapping):
        return []
    mode_results = test_statistics.get("persona_mode_results")
    datasets: list[dict[str, Any]] = []
    if isinstance(mode_results, Mapping):
        for mode, result in mode_results.items():
            if not isinstance(result, Mapping):
                continue
            mode_statistics = result.get("test_statistics")
            if not isinstance(mode_statistics, Mapping):
                continue
            dataset = _load_optimizer_dataset(mode_statistics)
            if dataset is not None:
                dataset["label"] = str(mode)
                datasets.append(dataset)
    if not datasets:
        dataset = _load_optimizer_dataset(test_statistics)
        if dataset is not None:
            datasets.append(dataset)
    return datasets


def _semantic_redundancy(
    item_ids: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
) -> float:
    values = [
        text_similarity(
            items[left].get("scenario"),
            items[right].get("scenario"),
        )
        for left, right in combinations(item_ids, 2)
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _combination_metrics(
    item_ids: Sequence[str],
    *,
    datasets: Sequence[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    mode_metrics: list[dict[str, Any]] = []
    for dataset in datasets:
        item_scores = dataset["item_scores"]
        if not set(item_ids) <= set(item_scores.columns):
            return None
        frame = item_scores[list(item_ids)].dropna()
        criteria = dataset["criteria"].loc[frame.index]
        if frame.empty:
            return None
        total = frame.mean(axis=1)
        target_column = str(dataset["target_column"])
        target_rho = total.corr(
            criteria[target_column],
            method="spearman",
        )
        non_target_values = [
            total.corr(criteria[column], method="spearman")
            for column in criteria.columns
            if column != target_column
        ]
        non_target_values = [
            float(value)
            for value in non_target_values
            if pd.notna(value)
        ]
        max_non_target = max(
            (abs(value) for value in non_target_values),
            default=0.0,
        )
        target_value = float(target_rho) if pd.notna(target_rho) else -1.0
        inter_item = frame.corr(method="spearman").abs()
        inter_values = [
            float(inter_item.iloc[left, right])
            for left in range(len(item_ids))
            for right in range(left + 1, len(item_ids))
            if pd.notna(inter_item.iloc[left, right])
        ]
        mode_metrics.append(
            {
                "mode": dataset.get("label"),
                "cronbach_alpha": _cronbach_alpha(frame),
                "target_rho": target_value,
                "max_non_target_rho": max_non_target,
                "discriminant_margin": target_value - max_non_target,
                "mean_absolute_inter_item_r": (
                    sum(inter_values) / len(inter_values)
                    if inter_values
                    else 0.0
                ),
            }
        )
    if not mode_metrics:
        return None
    worst_alpha = min(
        (
            metric["cronbach_alpha"]
            if isinstance(metric["cronbach_alpha"], (int, float))
            else -1.0
        )
        for metric in mode_metrics
    )
    worst_target = min(metric["target_rho"] for metric in mode_metrics)
    worst_margin = min(
        metric["discriminant_margin"] for metric in mode_metrics
    )
    worst_redundancy = max(
        metric["mean_absolute_inter_item_r"] for metric in mode_metrics
    )
    semantic_redundancy = _semantic_redundancy(item_ids, items)
    objective = (
        2.0 * worst_target
        + 1.0 * worst_margin
        + 0.75 * worst_alpha
        - 0.20 * worst_redundancy
        - 0.20 * semantic_redundancy
    )
    return {
        "objective": objective,
        "worst_case_target_rho": worst_target,
        "worst_case_discriminant_margin": worst_margin,
        "worst_case_cronbach_alpha": worst_alpha,
        "worst_case_inter_item_redundancy": worst_redundancy,
        "semantic_redundancy": semantic_redundancy,
        "persona_mode_metrics": mode_metrics,
    }


def _combination_admission_key(
    metrics: Mapping[str, Any],
) -> tuple[float, ...]:
    """Prefer construct separation, then target validity and reliability."""

    def number(field: str, fallback: float) -> float:
        value = metrics.get(field)
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else fallback
        )

    return (
        number("worst_case_discriminant_margin", -1.0),
        number("worst_case_target_rho", -1.0),
        number("worst_case_cronbach_alpha", -1.0),
        -number("worst_case_inter_item_redundancy", 1.0),
        -number("semantic_redundancy", 1.0),
    )


def _challenger_dominates(
    challenger: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> bool:
    """Admit only a combination that does not reduce any primary metric."""

    primary_fields = (
        "worst_case_discriminant_margin",
        "worst_case_target_rho",
        "worst_case_cronbach_alpha",
    )
    challenger_values = [
        float(challenger.get(field, -1.0)) for field in primary_fields
    ]
    incumbent_values = [
        float(incumbent.get(field, -1.0)) for field in primary_fields
    ]
    if any(
        candidate < baseline - _METRIC_TOLERANCE
        for candidate, baseline in zip(
            challenger_values,
            incumbent_values,
        )
    ):
        return False
    if any(
        candidate > baseline + _METRIC_TOLERANCE
        for candidate, baseline in zip(
            challenger_values,
            incumbent_values,
        )
    ):
        return True
    return _combination_admission_key(
        challenger
    ) > _combination_admission_key(incumbent)


def _apply_selected_ids_to_coverage(
    selected_ids: Sequence[str],
    *,
    retained_ids: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    blueprint: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    selected = list(selected_ids)
    selected_set = set(selected)
    reserve = [
        item_id for item_id in retained_ids if item_id not in selected_set
    ]
    updated = deepcopy(dict(coverage))
    for cell in updated.get("cells") or []:
        cell_id = cell.get("blueprint_cell_id")
        candidates = [
            item_id
            for item_id in retained_ids
            if _cell_id_for_item(items[item_id], blueprint) == cell_id
        ]
        cell["selected_item_ids"] = [
            item_id for item_id in selected if item_id in candidates
        ]
        cell["reserve_item_ids"] = [
            item_id for item_id in reserve if item_id in candidates
        ]
        cell["selected_count"] = len(cell["selected_item_ids"])
        cell["available_count"] = len(candidates)
        cell["missing_count"] = max(
            0,
            int(cell.get("planned_retention_count") or 0)
            - cell["selected_count"],
        )
        cell["passed"] = cell["missing_count"] == 0
    updated["selected_count"] = len(selected)
    updated["reserve_count"] = len(reserve)
    updated["missing_count"] = sum(
        int(cell.get("missing_count") or 0)
        for cell in updated.get("cells") or []
    )
    updated["passed"] = bool(updated.get("cells")) and all(
        bool(cell.get("passed")) for cell in updated.get("cells") or []
    )
    return reserve, updated


def _valid_incumbent_ids(
    snapshot: Any,
    *,
    retained_ids: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    blueprint: Mapping[str, Any],
) -> list[str] | None:
    if not isinstance(snapshot, Mapping):
        return None
    item_ids = snapshot.get("selected_item_ids")
    versions = snapshot.get("item_versions")
    if (
        not isinstance(item_ids, list)
        or not item_ids
        or not isinstance(versions, Mapping)
        or len(item_ids) != len(set(item_ids))
        or not set(item_ids) <= set(retained_ids)
    ):
        return None
    for item_id in item_ids:
        if (
            item_id not in items
            or versions.get(item_id) != items[item_id].get("version")
        ):
            return None
    required_by_cell = {
        str(cell["cell_id"]): int(cell["planned_retention_count"])
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping)
        and isinstance(cell.get("cell_id"), str)
        and isinstance(cell.get("planned_retention_count"), int)
    }
    actual_by_cell = {cell_id: 0 for cell_id in required_by_cell}
    for item_id in item_ids:
        cell_id = _cell_id_for_item(items[item_id], blueprint)
        if cell_id not in actual_by_cell:
            return None
        actual_by_cell[cell_id] += 1
    return list(item_ids) if actual_by_cell == required_by_cell else None


def _admit_monotonic_combination(
    state: Mapping[str, Any],
    *,
    selected_ids: Sequence[str],
    reserve_ids: Sequence[str],
    retained_ids: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    blueprint: Mapping[str, Any],
    coverage: Mapping[str, Any],
    test_statistics: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    datasets = _optimizer_datasets(test_statistics)
    challenger_metrics = (
        _combination_metrics(
            selected_ids,
            datasets=datasets,
            items=items,
        )
        if datasets
        else None
    )
    incumbent_snapshot = state.get("best_assembly_candidate")
    incumbent_ids = _valid_incumbent_ids(
        incumbent_snapshot,
        retained_ids=retained_ids,
        items=items,
        blueprint=blueprint,
    )
    incumbent_metrics = (
        _combination_metrics(
            incumbent_ids,
            datasets=datasets,
            items=items,
        )
        if datasets and incumbent_ids is not None
        else None
    )

    admitted_ids = list(selected_ids)
    decision = "initial_candidate"
    reason = "尚无历史正式组合，保存首个满足蓝图的组合"
    if incumbent_ids is not None:
        if list(selected_ids) == incumbent_ids:
            decision = "incumbent_unchanged"
            reason = "最优组合未发生变化"
        elif (
            challenger_metrics is not None
            and incumbent_metrics is not None
            and _challenger_dominates(
                challenger_metrics,
                incumbent_metrics,
            )
        ):
            decision = "challenger_accepted"
            reason = "挑战者的区分差值、目标相关和信度均未下降且至少一项提高"
        else:
            admitted_ids = incumbent_ids
            decision = "incumbent_retained"
            reason = (
                "挑战者未同时满足主要指标不下降要求，保留历史最佳组合"
                if challenger_metrics is not None
                and incumbent_metrics is not None
                else "缺少可比较的组合级证据，保留历史最佳组合"
            )

    admitted_metrics = (
        challenger_metrics
        if admitted_ids == list(selected_ids)
        else incumbent_metrics
    )
    reserve, updated_coverage = _apply_selected_ids_to_coverage(
        admitted_ids,
        retained_ids=retained_ids,
        items=items,
        blueprint=blueprint,
        coverage=coverage,
    )
    admission = {
        "decision": decision,
        "reason": reason,
        "challenger_item_ids": list(selected_ids),
        "admitted_item_ids": admitted_ids,
        "challenger_metrics": deepcopy(challenger_metrics),
        "incumbent_metrics": deepcopy(incumbent_metrics),
        "admitted_metrics": deepcopy(admitted_metrics),
        "primary_rule": (
            "discriminant_margin、target_rho、cronbach_alpha 均不得下降；"
            "至少一项提高才替换"
        ),
    }
    optimizer = updated_coverage.setdefault("optimizer", {})
    optimizer["pre_admission_challenger_metrics"] = deepcopy(
        challenger_metrics
    )
    optimizer["selected_metrics"] = deepcopy(admitted_metrics)
    optimizer["monotonic_admission"] = deepcopy(admission)
    snapshot = {
        "selected_item_ids": admitted_ids,
        "item_versions": {
            item_id: items[item_id].get("version")
            for item_id in admitted_ids
        },
        "metrics": deepcopy(admitted_metrics),
        "source_item_bank_id": state.get("item_bank_id"),
        "source_item_bank_version": state.get("item_bank_version"),
        "recorded_at": utc_timestamp(),
        "admission_decision": decision,
    }
    return admitted_ids, reserve, updated_coverage, snapshot, admission


def _select_by_blueprint(
    *,
    retained_ids: Sequence[str],
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    item_statistics: Mapping[str, Mapping[str, Any]],
    blueprint: Mapping[str, Any],
    test_statistics: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    order_index = {item_id: index for index, item_id in enumerate(item_order)}
    selected: list[str] = []
    reserve: list[str] = []
    cells: list[dict[str, Any]] = []
    assigned: set[str] = set()
    combination_groups: list[list[tuple[str, ...]]] = []

    for raw_cell in blueprint.get("cells") or []:
        if not isinstance(raw_cell, Mapping):
            continue
        cell_id = raw_cell.get("cell_id")
        planned = raw_cell.get("planned_retention_count")
        if (
            not isinstance(cell_id, str)
            or not isinstance(planned, int)
            or isinstance(planned, bool)
        ):
            continue
        candidates = [
            item_id
            for item_id in retained_ids
            if _cell_id_for_item(items[item_id], blueprint) == cell_id
        ]
        candidates.sort(
            key=lambda item_id: _quality_sort_key(
                item_id,
                item_statistics[item_id],
                order_index,
            )
        )
        chosen = candidates[:planned]
        extras = candidates[planned:]
        combination_groups.append(
            list(combinations(candidates, planned))
            if len(candidates) >= planned
            else []
        )
        selected.extend(chosen)
        reserve.extend(extras)
        assigned.update(candidates)
        cells.append(
            {
                "blueprint_cell_id": cell_id,
                "target_dimension_id": raw_cell.get("facet_id"),
                "planned_retention_count": planned,
                "available_count": len(candidates),
                "selected_count": len(chosen),
                "missing_count": max(0, planned - len(chosen)),
                "selected_item_ids": chosen,
                "reserve_item_ids": extras,
                "passed": len(chosen) >= planned,
            }
        )

    unassigned = [
        item_id for item_id in retained_ids if item_id not in assigned
    ]
    reserve.extend(unassigned)
    passed = bool(cells) and all(cell["passed"] for cell in cells)
    optimizer: dict[str, Any] = {
        "method": "deterministic_item_ranking",
        "combination_count": 0,
        "optimized": False,
        "reason": "缺少可读取的组合级作答数据",
    }
    datasets = _optimizer_datasets(test_statistics)
    combination_count = (
        prod(len(group) for group in combination_groups)
        if combination_groups
        and all(combination_groups)
        else 0
    )
    if (
        passed
        and datasets
        and 0 < combination_count <= MAX_COMBINATION_CANDIDATES
    ):
        best_ids: list[str] | None = None
        best_metrics: dict[str, Any] | None = None
        for grouped_choice in product(*combination_groups):
            candidate_ids = [
                item_id
                for group in grouped_choice
                for item_id in group
            ]
            metrics = _combination_metrics(
                candidate_ids,
                datasets=datasets,
                items=items,
            )
            if metrics is None:
                continue
            if (
                best_metrics is None
                or _combination_admission_key(metrics)
                > _combination_admission_key(best_metrics)
            ):
                best_ids = candidate_ids
                best_metrics = metrics
        if best_ids is not None and best_metrics is not None:
            selected = best_ids
            selected_set = set(selected)
            reserve = [
                item_id
                for item_id in retained_ids
                if item_id not in selected_set
            ]
            for cell in cells:
                cell_candidates = [
                    item_id
                    for item_id in retained_ids
                    if _cell_id_for_item(items[item_id], blueprint)
                    == cell["blueprint_cell_id"]
                ]
                cell["selected_item_ids"] = [
                    item_id
                    for item_id in selected
                    if item_id in cell_candidates
                ]
                cell["reserve_item_ids"] = [
                    item_id
                    for item_id in reserve
                    if item_id in cell_candidates
                ]
            optimizer = {
                "method": "exhaustive_blueprint_constrained_combination",
                "combination_count": combination_count,
                "optimized": True,
                "objective_definition": (
                    "按 worst_discriminant_margin、worst_target_rho、"
                    "worst_alpha、低冗余依次进行确定性比较"
                ),
                "selected_metrics": best_metrics,
                "persona_modes": [
                    dataset.get("label") for dataset in datasets
                ],
            }
    elif combination_count > MAX_COMBINATION_CANDIDATES:
        optimizer = {
            "method": "deterministic_item_ranking",
            "combination_count": combination_count,
            "optimized": False,
            "reason": (
                f"候选组合超过上限 {MAX_COMBINATION_CANDIDATES}，"
                "回退到稳定逐题排序"
            ),
        }
    return (
        selected,
        reserve,
        {
            "passed": passed,
            "cells": cells,
            "selected_count": len(selected),
            "reserve_count": len(reserve),
            "missing_count": sum(cell["missing_count"] for cell in cells),
            "unassigned_item_ids": unassigned,
            "optimizer": optimizer,
        },
    )


def _deduplicate_items(items: Sequence[Mapping[str, Any]]) -> list[dict]:
    output: list[dict] = []
    seen: set[str] = set()
    for raw_item in items:
        item_id = raw_item.get("item_id")
        if not isinstance(item_id, str) or item_id in seen:
            continue
        seen.add(item_id)
        output.append(deepcopy(dict(raw_item)))
    return output


def _metric_text(value: Any, *, digits: int = 3) -> str:
    numeric = metric_scalar(value)
    if numeric is None:
        return "证据不足"
    return f"{numeric:.{digits}f}"


def _selection_reason(
    recommendation: str,
    statistics: Mapping[str, Any],
) -> str:
    quality = statistics.get("quality_evaluation") or {}
    citc = quality.get("facet_citc") or {}
    specificity = quality.get("virtual_target_specificity") or {}
    if citc or specificity:
        target = specificity.get("target_spearman") or {}
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        metrics = (
            f"建议={recommendation}；"
            f"分面内CITC={_metric_text(citc.get('r'))}；"
            f"目标ρs={_metric_text(target.get('rho'))}；"
            "同域VTS="
            f"{_metric_text(same_domain.get('specificity_margin'))}；"
            "跨域VTS="
            f"{_metric_text(cross_domain.get('specificity_margin'))}"
        )
        flags = [
            str(flag)
            for flag in quality.get("diagnostic_flags") or []
            if str(flag).strip()
        ]
        return "；".join([metrics, *flags])
    corrected = (
        statistics.get("facet_corrected_item_total_correlation")
        or statistics.get("corrected_item_total_correlation")
        or {}
    )
    flags = [
        str(flag)
        for flag in quality.get("diagnostic_flags") or []
        if str(flag).strip()
    ]
    metrics = (
        f"建议={recommendation}；"
        f"分面内CITC={_metric_text(corrected.get('r'))}；"
        f"难度={_metric_text(statistics.get('difficulty'))}；"
        f"有效选项={quality.get('effective_option_count', '证据不足')}"
    )
    return "；".join([metrics, *flags])


def _mode_recommendations(
    state: Mapping[str, Any],
    item_id: str,
) -> dict[str, str]:
    mode_results = (
        (state.get("test_statistics") or {}).get("persona_mode_results") or {}
    )
    output: dict[str, str] = {}
    for mode, result in mode_results.items():
        if not isinstance(result, Mapping):
            continue
        statistics = (result.get("item_statistics") or {}).get(item_id)
        quality = (
            statistics.get("quality_evaluation")
            if isinstance(statistics, Mapping)
            else None
        )
        recommendation = (
            quality.get("recommendation")
            if isinstance(quality, Mapping)
            else None
        )
        if recommendation in {"retain", "revise", "remove"}:
            output[str(mode)] = str(recommendation)
    return output


def _effective_recommendation(
    state: Mapping[str, Any],
    item_id: str,
    item_version: Any,
    primary: str,
) -> tuple[str, str | None]:
    """Lock accepted versions and require agreement before hard removal."""

    locked_versions = state.get("locked_retained_item_versions")
    if (
        isinstance(locked_versions, Mapping)
        and locked_versions.get(item_id) == item_version
    ):
        return (
            "retain",
            (
                "该题目版本已锁定通过资格；本轮统计仅用于监测和组卷，"
                "不再触发自动返修或淘汰"
            ),
        )
    mode_recommendations = _mode_recommendations(state, item_id)
    if (
        primary == "remove"
        and mode_recommendations
        and any(value != "remove" for value in mode_recommendations.values())
    ):
        detail = "、".join(
            f"{mode}={value}"
            for mode, value in sorted(mode_recommendations.items())
        )
        return (
            "revise",
            "不同人格提示模式对删除结论不一致，降级为返修：" + detail,
        )
    return primary, None


def _restored_locked_versions(
    state: Mapping[str, Any],
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Restore locks for checkpoints created before explicit lock state."""

    locked = {
        str(item_id): int(version)
        for item_id, version in (
            state.get("locked_retained_item_versions") or {}
        ).items()
        if isinstance(item_id, str)
        and isinstance(version, int)
        and not isinstance(version, bool)
        and item_id in items
        and items[item_id].get("version") == version
    }
    seen_latest: set[str] = set()
    for entry in reversed(state.get("psychometric_selection_history") or []):
        if not isinstance(entry, Mapping):
            continue
        recommendations = entry.get("effective_recommendations")
        if not isinstance(recommendations, Mapping):
            continue
        for item_id, recommendation in recommendations.items():
            if (
                not isinstance(item_id, str)
                or item_id in seen_latest
                or item_id not in items
            ):
                continue
            seen_latest.add(item_id)
            version = items[item_id].get("version")
            if (
                recommendation == "retain"
                and isinstance(version, int)
                and not isinstance(version, bool)
            ):
                locked.setdefault(item_id, version)
    return locked


def _psychometric_repair_entry(
    *,
    item: Mapping[str, Any],
    statistics: Mapping[str, Any],
    revision_round: int,
) -> dict[str, Any]:
    item_id = str(item["item_id"])
    quality = statistics.get("quality_evaluation") or {}
    if (
        "facet_citc" in quality
        or "virtual_target_specificity" in quality
    ):
        # The queue only marks a symptom for diagnosis. It deliberately does
        # not manufacture an editable cause from the virtual metrics.
        return {
            "item_id": item_id,
            "blueprint_cell_id": item.get("blueprint_cell_id"),
            "target_dimension_id": item.get("target_dimension_id"),
            "action": "revise_item",
            "revision_round": revision_round,
            "baseline_metrics": {
                "quality_evaluation": deepcopy(quality),
            },
            "review": {
                "findings": [],
                "repair_tasks": [],
                "summary": (
                    f"题目 {item_id} 未通过虚拟迭代四门槛筛查；等待基于题面与"
                    "构念约束的可定位诊断，若无题面证据则 defer。"
                ),
            },
        }
    option_ids = [
        str(option.get("option_id"))
        for option in item.get("response_options") or []
        if isinstance(option, Mapping) and option.get("option_id")
    ]
    discrimination = quality.get("discrimination_rating")
    distribution = quality.get("distribution_rating")
    option_function = quality.get("option_function_rating")
    corrected = (
        statistics.get("facet_corrected_item_total_correlation")
        or statistics.get("corrected_item_total_correlation")
        or {}
    )
    findings: list[dict[str, Any]] = []

    rewrite_required = discrimination == "poor"
    if rewrite_required:
        findings.append(
            {
                "criterion": "construct_purity",
                "severity": "blocking",
                "locus": "scenario",
                "affected_option_ids": [],
                "evidence": (
                    "分面内CITC="
                    f"{_metric_text(corrected.get('r'))}"
                ),
                "problem": (
                    "当前情境—反应结构未稳定区分目标特质水平，"
                    "继续保留会削弱构念解释。"
                ),
                "repair_instruction": (
                    "保持目标 facet 和已审核骨架，实质重写具体事件与四个"
                    "行为选项；增强目标特质相关选择张力，减少非目标能力、"
                    "资源、服从和社会赞许性线索。"
                ),
            }
        )
    else:
        if distribution == "weak":
            findings.append(
                {
                    "criterion": "option_anti_faking",
                    "severity": "blocking",
                    "locus": "response_options",
                    "affected_option_ids": option_ids,
                    "evidence": (
                        "标准化题目均值="
                        f"{_metric_text(statistics.get('difficulty'))}"
                    ),
                    "problem": (
                        "得分分布过度集中于量尺一端，四级行为距离或表面"
                        "吸引力不平衡。"
                    ),
                    "repair_instruction": (
                        "保持 behavioral_level 不变，重新平衡四个选项的"
                        "现实收益、代价、长度和行为距离，避免明显最佳或"
                        "稻草人选项。"
                    ),
                }
            )
        if option_function == "weak":
            option_statistics = statistics.get("option_statistics") or {}
            affected = [
                option_id
                for option_id in option_ids
                if (
                    isinstance(option_statistics.get(option_id), Mapping)
                    and float(
                        option_statistics[option_id].get("rate") or 0.0
                    )
                    < 0.05
                )
            ] or option_ids
            findings.append(
                {
                    "criterion": "ecological_plausibility",
                    "severity": "blocking",
                    "locus": "response_options",
                    "affected_option_ids": affected,
                    "evidence": (
                        f"有效选项数={quality.get('effective_option_count')}；"
                        "低频选项=" + "、".join(affected)
                    ),
                    "problem": (
                        "部分反应在虚拟样本中几乎无人选择，可能缺乏现实"
                        "可接受性或与相邻等级距离失衡。"
                    ),
                    "repair_instruction": (
                        "只改写指定低频选项，使其对目标人群真实可选，同时"
                        "保持对应行为等级和单一构念方向。"
                    ),
                }
            )

    if not findings:
        findings.append(
            {
                "criterion": "construct_purity",
                "severity": "blocking",
                "locus": "response_options",
                "affected_option_ids": option_ids,
                "evidence": _selection_reason("revise", statistics),
                "problem": "题目未达到当前开发期保留标准。",
                "repair_instruction": (
                    "保持情境与行为等级，改写四个选项以提高目标构念"
                    "区分度并降低表面答案透明度。"
                ),
            }
        )

    review = {
        "findings": findings,
        "repair_tasks": build_repair_tasks_from_findings(
            findings,
            current_item=dict(item),
        ),
        "summary": (
            f"心理测量第 {revision_round} 轮诊断要求定向返修题目 "
            f"{item_id}。"
        ),
    }
    action = (
        "regenerate_item"
        if any(
            target.get("field") == "scenario"
            for task in review["repair_tasks"]
            for target in task.get("targets") or []
        )
        else "revise_item"
    )
    return {
        "item_id": item_id,
        "blueprint_cell_id": item.get("blueprint_cell_id"),
        "target_dimension_id": item.get("target_dimension_id"),
        "action": action,
        "revision_round": revision_round,
        "baseline_metrics": {
            "difficulty": statistics.get("difficulty"),
            "facet_corrected_item_total_correlation": deepcopy(corrected),
            "quality_evaluation": deepcopy(quality),
        },
        "review": review,
    }


def _psychometric_defer_entry(
    *,
    item: Mapping[str, Any],
    statistics: Mapping[str, Any],
    completed_rounds: int,
) -> dict[str, Any]:
    """Create an automatic same-cell replenishment entry after the repair limit."""

    entry = _psychometric_repair_entry(
        item=item,
        statistics=statistics,
        revision_round=completed_rounds + 1,
    )
    entry.update(
        {
            "action": "defer",
            "queue_status": "deferred_decision",
            "completed_repair_rounds": completed_rounds,
            "defer_after_rounds": PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS,
            "diagnosis_status": "repair_rounds_exhausted",
            "atomic_repair_advice": {
                "decision": "defer",
                "summary": (
                    f"已完成 {completed_rounds} 轮返修仍未达标，"
                    "自动进入同一蓝图槽位补题队列。"
                ),
                "observed_discrepancies": [],
                "candidate_diagnoses": [],
                "repair_tasks": [],
            },
        }
    )
    return entry


def build_psychometric_repair_evidence(
    state: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded text-plus-statistics packet for repair diagnosis."""

    item_id = str(entry.get("item_id") or "")
    item = next(
        (
            deepcopy(dict(candidate))
            for candidate in [
                *(state.get("frozen_item_bank") or []),
                *(state.get("item_pool") or []),
            ]
            if isinstance(candidate, Mapping)
            and candidate.get("item_id") == item_id
        ),
        {},
    )
    specification = next(
        (
            deepcopy(dict(candidate))
            for candidate in state.get("item_specifications") or []
            if isinstance(candidate, Mapping)
            and candidate.get("specification_id") == item_id
        ),
        {},
    )
    latest_review: dict[str, Any] | None = None
    for record in reversed(
        (state.get("item_history") or {}).get(item_id, [])
    ):
        if isinstance(record, Mapping) and isinstance(
            record.get("review"), Mapping
        ):
            latest_review = deepcopy(dict(record["review"]))
            break
    statistics = deepcopy(
        dict((state.get("item_statistics") or {}).get(item_id) or {})
    )
    option_statistics = statistics.get("option_statistics") or {}
    option_evidence = []
    for option in item.get("response_options") or []:
        if not isinstance(option, Mapping):
            continue
        option_id = str(option.get("option_id") or "")
        option_evidence.append(
            {
                "option_id": option_id,
                "text": option.get("text"),
                "behavioral_level": option.get("behavioral_level"),
                "score": (item.get("scoring_key") or {}).get(option_id),
                **deepcopy(dict(option_statistics.get(option_id) or {})),
            }
        )
    prior_repairs = [
        {
            key: deepcopy(event.get(key))
            for key in (
                "event",
                "revision_round",
                "action",
                "candidate_admitted",
                "admission_evidence",
                "reason",
            )
            if key in event
        }
        for event in state.get("psychometric_repair_history") or []
        if isinstance(event, Mapping) and event.get("item_id") == item_id
    ][-4:]
    return {
        "current_item": item,
        "reviewed_item_specification": specification,
        "latest_content_review": latest_review,
        "psychometric_statistics": {
            key: deepcopy(statistics.get(key))
            for key in (
                "difficulty",
                "facet_corrected_item_total_correlation",
                "quality_evaluation",
                "minimum_option_rate",
            )
        },
        "option_evidence": option_evidence,
        "persona_mode_recommendations": _mode_recommendations(
            state,
            item_id,
        ),
        "baseline_or_prior_repairs": prior_repairs,
        "blueprint_need": {
            "blueprint_cell_id": entry.get("blueprint_cell_id"),
            "revision_round": entry.get("revision_round"),
            "candidate_is_needed_for_gap": True,
        },
    }


def validate_psychometric_repair_diagnosis(
    diagnosis: Any,
    evidence: Mapping[str, Any],
) -> None:
    """Reject unsupported or structurally ambiguous repair diagnoses."""

    expected_top = {"decision", "primary_hypothesis", "acceptance_criteria", "summary"}
    if not isinstance(diagnosis, dict) or set(diagnosis) != expected_top:
        raise ValueError(
            "心理测量返修诊断只能包含 decision、primary_hypothesis、"
            "acceptance_criteria、summary"
        )
    decision = diagnosis.get("decision")
    if decision not in {
        "retain",
        "revise_item",
        "revise_options",
        "regenerate_realization",
        "replace_candidate",
        "defer",
    }:
        raise ValueError("心理测量返修诊断 decision 无效")

    hypothesis = diagnosis.get("primary_hypothesis")
    if not isinstance(hypothesis, dict):
        raise ValueError("心理测量返修诊断缺少 primary_hypothesis")

    expected_hypothesis_fields = {
        "failure_mode",
        "locus",
        "affected_option_ids",
        "observed_pattern",
        "textual_evidence",
        "alternative_explanations",
        "minimal_edit_operator",
        "repair_instruction",
        "predicted_change",
        "confidence",
    }
    if set(hypothesis) != expected_hypothesis_fields:
        raise ValueError(
            "primary_hypothesis 字段不完整，必须包含："
            + "、".join(sorted(expected_hypothesis_fields))
        )

    valid_ids = {
        str(option.get("option_id"))
        for option in (evidence.get("current_item") or {}).get(
            "response_options"
        )
        or []
        if isinstance(option, Mapping) and option.get("option_id")
    }

    locus = hypothesis.get("locus")
    option_ids = hypothesis.get("affected_option_ids")
    if locus not in {
        "scenario",
        "response_options",
        "behavioral_level",
        "skeleton",
        "construct",
        "uncertain",
    }:
        raise ValueError("primary_hypothesis locus 无效")
    if not isinstance(option_ids, list) or not all(
        isinstance(option_id, str) for option_id in option_ids
    ):
        raise ValueError("primary_hypothesis affected_option_ids 无效")
    if locus == "response_options":
        if not option_ids or not set(option_ids).issubset(valid_ids):
            raise ValueError("primary_hypothesis 指向了无效选项")
    elif option_ids:
        raise ValueError("非选项 locus 不得包含 affected_option_ids")

    if hypothesis.get("confidence") not in {"low", "medium", "high"}:
        raise ValueError("primary_hypothesis confidence 无效")

    for field in (
        "observed_pattern",
        "textual_evidence",
        "minimal_edit_operator",
        "repair_instruction",
        "predicted_change",
    ):
        if not str(hypothesis.get(field) or "").strip():
            raise ValueError(f"primary_hypothesis 缺少 {field}")

    alt_explanations = hypothesis.get("alternative_explanations")
    if not isinstance(alt_explanations, list) or not alt_explanations:
        raise ValueError(
            "primary_hypothesis.alternative_explanations 必须至少包含一条"
        )
    if not all(
        isinstance(exp, str) and exp.strip() for exp in alt_explanations
    ):
        raise ValueError(
            "primary_hypothesis.alternative_explanations 每一条必须是非空字符串"
        )

    acceptance_criteria = diagnosis.get("acceptance_criteria")
    if not isinstance(acceptance_criteria, list) or not acceptance_criteria:
        raise ValueError("acceptance_criteria 必须至少包含一条")
    if not all(
        isinstance(crit, str) and crit.strip() for crit in acceptance_criteria
    ):
        raise ValueError("acceptance_criteria 每一条必须是非空字符串")

    # Decision / locus consistency checks.
    actionable_locus = locus not in {"uncertain", "construct"}
    if decision == "revise_item" and not actionable_locus:
        raise ValueError(
            "revise_item 的主假说 locus 必须可执行 "
            "(scenario 或 response_options)"
        )
    if decision == "revise_options" and locus != "response_options":
        raise ValueError("revise_options 的主假说 locus 必须是 response_options")
    if decision == "regenerate_realization" and locus != "scenario":
        raise ValueError("regenerate_realization 的主假说 locus 必须是 scenario")
    if decision == "replace_candidate" and not (
        locus == "skeleton" and hypothesis.get("confidence") == "high"
    ):
        raise ValueError("replace_candidate 必须有 locus=skeleton 且 confidence=high")

    if not str(diagnosis.get("summary") or "").strip():
        raise ValueError("心理测量返修诊断缺少 summary")


def psychometric_diagnosis_to_review(
    diagnosis: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a single primary_hypothesis into the unified review contract."""

    decision = diagnosis.get("decision")
    hypothesis = diagnosis.get("primary_hypothesis")
    if not isinstance(hypothesis, dict):
        raise ValueError("psychometric repair diagnosis missing primary_hypothesis")

    hyp_locus = str(hypothesis.get("locus") or "uncertain")
    hyp_option_ids = list(hypothesis.get("affected_option_ids") or [])
    hyp_failure_mode = str(hypothesis.get("failure_mode") or "")

    # Map locus + failure_mode to the content-review criterion.
    if hyp_locus == "scenario":
        criterion = "trait_activation"
    elif hyp_failure_mode in {
        "low_frequency_options",
        "score_concentration",
        "transparent_scoring",
    }:
        criterion = "option_anti_faking"
    else:
        criterion = "construct_purity"

    if decision == "retain":
        return {
            "findings": [
                {
                    "criterion": criterion,
                    "severity": "warning",
                    "locus": hyp_locus,
                    "affected_option_ids": hyp_option_ids,
                    "evidence": (
                        str(hypothesis.get("observed_pattern", ""))
                        + "; "
                        + str(hypothesis.get("textual_evidence", ""))
                    ),
                    "problem": (
                        str(hypothesis.get("textual_evidence", ""))
                        or "统计症状未指向可操作的文本缺陷"
                    ),
                    "repair_instruction": str(
                        hypothesis.get("repair_instruction")
                        or "保留当前版本，监测下一轮表现"
                    ),
                }
            ],
            "repair_tasks": [],
            "summary": str(
                diagnosis.get("summary")
                or "项目达到开发期保留标准，无需返修"
            ),
        }

    # For all non-retain decisions, produce a single blocking finding.
    review_finding = {
        "criterion": criterion,
        "severity": "blocking",
        "locus": hyp_locus,
        "affected_option_ids": hyp_option_ids,
        "evidence": (
            str(hypothesis.get("observed_pattern", ""))
            + "; "
            + str(hypothesis.get("textual_evidence", ""))
        ),
        "problem": str(
            hypothesis.get("textual_evidence", "")
            or "LLM 诊断未返回可执行的文本 level 发现"
        ),
        "repair_instruction": str(
            hypothesis.get("repair_instruction")
            or "根据心理测量统计指标和构念定义修改题目"
        ),
    }
    return {
        "findings": [review_finding],
        "repair_tasks": build_repair_tasks_from_findings(
            [review_finding],
            current_item=dict(item),
        ),
        "summary": str(diagnosis.get("summary") or "心理测量返修诊断完成"),
    }


def _metric_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _repair_candidate_admission(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Require a repaired version to improve without degrading key evidence."""

    baseline_quality = baseline.get("quality_evaluation") or {}
    current_quality = current.get("quality_evaluation") or {}
    recommendation_rank = {"remove": 0, "revise": 1, "retain": 2}
    baseline_recommendation = str(
        baseline_quality.get("recommendation") or "revise"
    )
    current_recommendation = str(
        current_quality.get("recommendation") or "remove"
    )
    baseline_citc_metric = baseline_quality.get("facet_citc") or {}
    current_citc_metric = current_quality.get("facet_citc") or {}
    baseline_specificity = baseline_quality.get(
        "virtual_target_specificity"
    ) or {}
    current_specificity = current_quality.get(
        "virtual_target_specificity"
    ) or {}
    if baseline_citc_metric or baseline_specificity:
        baseline_same = baseline_specificity.get("same_domain_non_target") or {}
        current_same = current_specificity.get("same_domain_non_target") or {}
        baseline_cross = baseline_specificity.get("cross_domain_non_target") or {}
        current_cross = current_specificity.get("cross_domain_non_target") or {}
        baseline_values = {
            "facet_citc": _metric_number(baseline_citc_metric.get("r")),
            "target_rho": _metric_number(
                (baseline_specificity.get("target_spearman") or {}).get("rho")
            ),
            "same_domain_vts": _metric_number(
                baseline_same.get("specificity_margin")
            ),
            "cross_domain_vts": _metric_number(
                baseline_cross.get("specificity_margin")
            ),
        }
        current_values = {
            "facet_citc": _metric_number(current_citc_metric.get("r")),
            "target_rho": _metric_number(
                (current_specificity.get("target_spearman") or {}).get("rho")
            ),
            "same_domain_vts": _metric_number(
                current_same.get("specificity_margin")
            ),
            "cross_domain_vts": _metric_number(
                current_cross.get("specificity_margin")
            ),
        }
        rank_not_worse = recommendation_rank.get(
            current_recommendation, -1
        ) >= recommendation_rank.get(baseline_recommendation, -1)
        estimable = all(value is not None for value in current_values.values())
        not_worse = all(
            baseline_values[key] is None
            or current_values[key] is None
            or current_values[key] >= baseline_values[key] - 0.03
            for key in baseline_values
        )
        improved = (
            recommendation_rank.get(current_recommendation, -1)
            > recommendation_rank.get(baseline_recommendation, -1)
            or any(
                baseline_values[key] is not None
                and current_values[key] is not None
                and current_values[key] >= baseline_values[key] + 0.02
                for key in baseline_values
            )
        )
        evidence = {
            "baseline_recommendation": baseline_recommendation,
            "current_recommendation": current_recommendation,
            "baseline_virtual_metrics": baseline_values,
            "current_virtual_metrics": current_values,
        }
        return rank_not_worse and estimable and not_worse and improved, evidence

    baseline_citc = _metric_number(
        (
            baseline.get("facet_corrected_item_total_correlation")
            or baseline.get("corrected_item_total_correlation")
            or {}
        ).get("r")
    )
    current_citc = _metric_number(
        (
            current.get("facet_corrected_item_total_correlation")
            or current.get("corrected_item_total_correlation")
            or {}
        ).get("r")
    )
    baseline_options = int(baseline_quality.get("effective_option_count") or 0)
    current_options = int(current_quality.get("effective_option_count") or 0)

    rank_not_worse = recommendation_rank.get(current_recommendation, -1) >= (
        recommendation_rank.get(baseline_recommendation, -1)
    )
    citc_not_worse = (
        baseline_citc is None
        or current_citc is None
        or current_citc >= baseline_citc - 0.03
    )
    sign_not_reversed = not (
        baseline_citc is not None
        and current_citc is not None
        and baseline_citc >= 0
        and current_citc < 0
    )
    options_not_worse = current_options >= baseline_options
    improved = (
        recommendation_rank.get(current_recommendation, -1)
        > recommendation_rank.get(baseline_recommendation, -1)
        or current_options > baseline_options
        or (
            baseline_citc is not None
            and current_citc is not None
            and current_citc >= baseline_citc + 0.02
        )
    )
    evidence = {
        "baseline_recommendation": baseline_recommendation,
        "current_recommendation": current_recommendation,
        "baseline_citc": baseline_citc,
        "current_citc": current_citc,
        "baseline_effective_option_count": baseline_options,
        "current_effective_option_count": current_options,
    }
    return (
        rank_not_worse
        and citc_not_worse
        and sign_not_reversed
        and options_not_worse
        and improved,
        evidence,
    )


def _evaluate_latest_repair_candidate(
    state: Mapping[str, Any],
    items: Mapping[str, Mapping[str, Any]],
    item_statistics: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Evaluate one repair batch and roll every rejected candidate back once."""

    history = deepcopy(state.get("psychometric_repair_history") or [])
    rejected: list[dict[str, Any]] = []
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if (
            not isinstance(event, dict)
            or event.get("event") != "psychometric_item_repaired"
            or event.get("candidate_evaluated") is True
        ):
            continue
        item_id = event.get("item_id")
        baseline_item = event.get("baseline_item")
        current_item = items.get(item_id)
        current_statistics = item_statistics.get(item_id)
        if (
            not isinstance(item_id, str)
            or not isinstance(baseline_item, Mapping)
            or not isinstance(current_item, Mapping)
            or current_item.get("version") != event.get("new_item_version")
            or not isinstance(current_statistics, Mapping)
        ):
            event["candidate_evaluated"] = True
            event["candidate_admitted"] = False
            event["evaluation_reason"] = "repair baseline no longer matches current bank"
            continue
        admitted, evidence = _repair_candidate_admission(
            event.get("baseline_metrics") or {},
            current_statistics,
        )
        event["candidate_evaluated"] = True
        event["candidate_admitted"] = admitted
        event["admission_evidence"] = evidence
        if admitted:
            continue
        rejected.append(
            {
                "item_id": item_id,
                "baseline_item": deepcopy(dict(baseline_item)),
                "baseline_profile": deepcopy(event.get("baseline_profile")),
                "baseline_analysis_snapshot": deepcopy(
                    event.get("baseline_analysis_snapshot")
                ),
                "restored_version": baseline_item.get("version"),
                "rejected_version": current_item.get("version"),
                "admission_evidence": evidence,
            }
        )

    if rejected:
        rejected_by_id = {
            str(entry["item_id"]): entry for entry in rejected
        }
        restored_pool = [
            deepcopy(rejected_by_id[str(item.get("item_id"))]["baseline_item"])
            if str(item.get("item_id")) in rejected_by_id
            else deepcopy(dict(item))
            for item in state.get("item_pool") or []
            if isinstance(item, Mapping)
        ]
        profiles = dict(state.get("item_pattern_profiles") or {})
        for entry in rejected:
            baseline_profile = entry.get("baseline_profile")
            if isinstance(baseline_profile, Mapping):
                profiles[str(entry["item_id"])] = deepcopy(
                    dict(baseline_profile)
                )

        # If the whole restored pool matches a saved pre-repair snapshot, its
        # responses and statistics are still valid and can be restored without
        # another model call. Partial rollbacks fall back to one incremental
        # simulation for the whole rejected set, never one cycle per event.
        reusable_snapshot = next(
            (
                entry.get("baseline_analysis_snapshot")
                for entry in reversed(rejected)
                if isinstance(entry.get("baseline_analysis_snapshot"), Mapping)
                and entry["baseline_analysis_snapshot"].get("frozen_item_bank")
                == restored_pool
                and entry["baseline_analysis_snapshot"].get(
                    "virtual_response_data_ref"
                )
                and entry["baseline_analysis_snapshot"].get("item_statistics")
                and entry["baseline_analysis_snapshot"].get("test_statistics")
            ),
            None,
        )
        rollback = {
            "item_pool": restored_pool,
            "item_pattern_profiles": profiles,
            "psychometric_repair_history": history,
            # Force one clean selection pass over the restored bank. Keeping a
            # terminal-looking rollback status here previously trapped Router
            # in a simulate/analyze/rollback loop.
            "selection_results": None,
            "selected_items": [],
            "reserve_items": [],
            "selection_reasons": {},
            "blueprint_coverage": None,
            "factor_results": None,
            "irt_results": None,
            "dif_results": None,
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
        }
        if isinstance(reusable_snapshot, Mapping):
            rollback.update(
                {
                    key: deepcopy(reusable_snapshot.get(key))
                    for key in (
                        "frozen_item_bank",
                        "item_bank_id",
                        "item_bank_version",
                        "item_bank_fingerprint",
                        "item_bank_frozen_at",
                        "virtual_response_data_ref",
                        "virtual_response_summary",
                        "virtual_response_item_bank_id",
                        "virtual_response_item_bank_version",
                        "item_statistics",
                        "test_statistics",
                    )
                }
            )
        else:
            current_response_ref = state.get("virtual_response_data_ref")
            rollback.update(
                {
                    "virtual_response_data_ref": None,
                    "virtual_response_summary": None,
                    "virtual_response_item_bank_id": None,
                    "virtual_response_item_bank_version": None,
                    "item_statistics": {},
                    "psychometric_round_result": None,
                    "test_statistics": None,
                }
            )
            if isinstance(current_response_ref, str) and current_response_ref:
                rollback["previous_virtual_response_data_ref"] = (
                    current_response_ref
                )
        return rollback, history
    return None, history


def run_item_selection(state: PSJTState) -> dict[str, Any]:
    """Classify items, return repairable items, then optimize passed items."""

    item_order, items = _items_by_id(state)
    item_statistics = state.get("item_statistics")
    if not isinstance(item_statistics, Mapping) or not item_statistics:
        raise ValueError("题目筛选前缺少 item_statistics")
    rollback, repair_history = _evaluate_latest_repair_candidate(
        state,
        items,
        item_statistics,
    )
    if rollback is not None:
        return {
            "state_update": rollback,
            "summary": (
                "返修候选未优于基线版本；已批量恢复上一版，并复用可用的基线作答与统计。"
            ),
        }

    restored_locked_versions = _restored_locked_versions(state, items)
    recommendation_state = {
        **state,
        "locked_retained_item_versions": restored_locked_versions,
    }
    source_recommendations: dict[str, str] = {}
    effective_recommendations: dict[str, str] = {}
    source_counts = {"retain": 0, "revise": 0, "remove": 0}
    effective_counts = {"retain": 0, "revise": 0, "remove": 0}
    reasons: dict[str, str] = {}
    for item_id in item_order:
        statistics = item_statistics.get(item_id)
        if not isinstance(statistics, Mapping):
            raise ValueError(f"题目 {item_id} 缺少统计结果")
        quality = statistics.get("quality_evaluation")
        recommendation = (
            quality.get("recommendation")
            if isinstance(quality, Mapping)
            else None
        )
        if recommendation not in source_counts:
            raise ValueError(f"题目 {item_id} 缺少有效质量评价")
        source_recommendations[item_id] = str(recommendation)
        source_counts[str(recommendation)] += 1
        effective, stability_notice = _effective_recommendation(
            recommendation_state,
            item_id,
            items[item_id].get("version"),
            str(recommendation),
        )
        effective_recommendations[item_id] = effective
        effective_counts[effective] += 1
        reasons[item_id] = _selection_reason(effective, statistics)
        if stability_notice:
            reasons[item_id] += "；" + stability_notice

    blueprint = state.get("blueprint")
    if not isinstance(blueprint, Mapping):
        raise ValueError("题目筛选前缺少有效蓝图")

    exploratory_evidence = _uses_exploratory_virtual_evidence(state)
    locked_retained_versions = dict(restored_locked_versions)
    if not exploratory_evidence:
        for item_id, recommendation in effective_recommendations.items():
            version = items[item_id].get("version")
            if (
                recommendation == "retain"
                and isinstance(version, int)
                and not isinstance(version, bool)
            ):
                locked_retained_versions[item_id] = version
    repair_rounds = dict(state.get("psychometric_repair_rounds") or {})
    revision_candidates: list[dict[str, Any]] = []
    defer_candidates: list[dict[str, Any]] = []
    removed_ids: list[str] = []
    if exploratory_evidence:
        retained_ids = list(item_order)
        for item_id in item_order:
            if source_recommendations[item_id] != "retain":
                notice = (
                    "虚拟样本量低于自动筛选阈值，原始统计建议仅作探索性报告，"
                    "未据此自动淘汰或修改题目"
                )
                reasons[item_id] = "；".join(
                    part for part in (reasons[item_id], notice) if part
                )
    else:
        retained_ids = [
            item_id
            for item_id in item_order
            if effective_recommendations[item_id] == "retain"
        ]
        for item_id in item_order:
            recommendation = effective_recommendations[item_id]
            if recommendation == "retain":
                continue
            if recommendation == "remove":
                recommendation = "revise"
            completed_rounds = int(repair_rounds.get(item_id, 0))
            next_round = completed_rounds + 1
            if completed_rounds >= PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS:
                defer_candidates.append(
                    _psychometric_defer_entry(
                        item=items[item_id],
                        statistics=item_statistics[item_id],
                        completed_rounds=completed_rounds,
                    )
                )
                reasons[item_id] += (
                    "；已完成三轮心理测量返修仍未达标，自动进入同一蓝图槽位补题队列"
                )
                continue
            entry = _psychometric_repair_entry(
                item=items[item_id],
                statistics=item_statistics[item_id],
                revision_round=next_round,
            )
            revision_candidates.append(entry)

    selected_ids, reserve_ids, coverage = _select_by_blueprint(
        retained_ids=retained_ids,
        item_order=item_order,
        items=items,
        item_statistics=item_statistics,
        blueprint=blueprint,
        test_statistics=(
            state.get("test_statistics")
            if isinstance(state.get("test_statistics"), Mapping)
            else None
        ),
    )
    provisional_flags = {
        str(item_id): deepcopy(dict(flag))
        for item_id, flag in (state.get("provisional_item_flags") or {}).items()
        if isinstance(item_id, str) and isinstance(flag, Mapping)
    }
    selected_provisional_ids = sorted(
        item_id for item_id in selected_ids if item_id in provisional_flags
    )
    developmental_override = bool(selected_provisional_ids)
    best_assembly_candidate = state.get("best_assembly_candidate")
    monotonic_admission: dict[str, Any] | None = None
    if coverage["passed"] and not developmental_override:
        (
            selected_ids,
            reserve_ids,
            coverage,
            best_assembly_candidate,
            monotonic_admission,
        ) = _admit_monotonic_combination(
            state,
            selected_ids=selected_ids,
            reserve_ids=reserve_ids,
            retained_ids=retained_ids,
            items=items,
            blueprint=blueprint,
            coverage=coverage,
            test_statistics=(
                state.get("test_statistics")
                if isinstance(state.get("test_statistics"), Mapping)
                else None
            ),
        )
    revision_entries: list[dict[str, Any]] = [*defer_candidates]
    deferred_revision_entries: list[dict[str, Any]] = []
    if not exploratory_evidence:
        missing_by_cell = {
            str(cell["blueprint_cell_id"]): int(cell["missing_count"])
            for cell in coverage["cells"]
            if int(cell.get("missing_count") or 0) > 0
        }
        if coverage["passed"]:
            deferred_revision_entries = revision_candidates
        else:
            for entry in revision_candidates:
                cell_id = str(entry.get("blueprint_cell_id") or "")
                if missing_by_cell.get(cell_id, 0) > 0:
                    revision_entries.append(entry)
                    missing_by_cell[cell_id] -= 1
                else:
                    deferred_revision_entries.append(entry)

        for entry in revision_entries:
            reasons[entry["item_id"]] += (
                (
                    "；已完成三轮返修仍未达标，自动进入同一蓝图槽位补题队列"
                    if entry.get("action") == "defer"
                    else (
                        f"；该蓝图单元存在保留题缺口，进入第 "
                        f"{entry['revision_round']} 轮心理测量定向返修，"
                        f"动作={entry['action']}"
                    )
                )
            )
        for entry in deferred_revision_entries:
            reasons[entry["item_id"]] += (
                (
                    "；现有 retain 题已覆盖该蓝图单元，本轮暂不返修，"
                    if coverage["passed"]
                    else "；该蓝图单元已安排足够的缺口返修候选，本轮暂不返修，"
                )
                + "不阻塞正式组卷"
            )
        for item_id in removed_ids:
            reasons[item_id] += (
                "；现有 retain 题已满足蓝图，不触发补题"
                if coverage["passed"]
                else "；retain 题尚未满足蓝图，后续按实际缺口补题"
            )

    if exploratory_evidence:
        status = (
            "ready_for_assembly"
            if coverage["passed"]
            else "fixed_blueprint_gap"
        )
    elif any(entry.get("action") == "defer" for entry in revision_entries):
        status = "repair_confirmation_required"
    elif coverage["passed"]:
        status = "ready_for_assembly"
    elif revision_entries:
        status = "revision_required"
    elif removed_ids:
        status = "fixed_blueprint_gap"
    else:
        status = (
            "ready_for_assembly"
            if coverage["passed"]
            else "fixed_blueprint_gap"
        )
    active_delivery_ids = (
        set(selected_ids) | set(reserve_ids)
        if coverage["passed"]
        else set()
    )
    previous_removed = _deduplicate_items(
        item
        for item in state.get("removed_items") or []
        if isinstance(item, Mapping)
        and item.get("item_id") not in active_delivery_ids
    )
    removed = _deduplicate_items(
        [
            *previous_removed,
            *(
                items[item_id]
                for item_id in removed_ids
                if item_id not in active_delivery_ids
            ),
        ]
    )
    removed_set = set(removed_ids) if not coverage["passed"] else set()
    remaining_pool = [
        deepcopy(dict(item))
        for item in state.get("item_pool") or []
        if isinstance(item, Mapping) and item.get("item_id") not in removed_set
    ]
    revise_entries = [
        entry
        for entry in revision_entries
        if entry["action"] in {"defer", "revise_item"}
    ]
    regenerate_entries = [
        entry
        for entry in revision_entries
        if entry["action"] == "regenerate_item"
    ]

    sample_config = state.get("virtual_sample_config")
    sample_size = (
        sample_config.get("sample_size")
        if isinstance(sample_config, Mapping)
        else None
    )
    recommended_size = (
        sample_config.get("recommended_sample_size")
        if isinstance(sample_config, Mapping)
        else None
    )
    automatic_selection_minimum = (
        sample_config.get(
            "automatic_selection_minimum_sample_size",
            MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE,
        )
        if isinstance(sample_config, Mapping)
        else MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE
    )
    history = deepcopy(state.get("psychometric_selection_history") or [])
    candidate_used = len(developed_candidate_ids(state))
    history.append(
        {
            "event": "psychometric_selection",
            "recorded_at": utc_timestamp(),
            "item_bank_id": state.get("item_bank_id"),
            "item_bank_version": state.get("item_bank_version"),
            "status": status,
            "evidence_level": (
                "exploratory" if exploratory_evidence else "developmental"
            ),
            "sample_size": sample_size,
            "recommended_sample_size": recommended_size,
            "automatic_selection_minimum_sample_size": (
                automatic_selection_minimum
            ),
            "source_recommendations": deepcopy(source_recommendations),
            "effective_recommendations": deepcopy(
                effective_recommendations
            ),
            "locked_retained_item_versions": deepcopy(
                locked_retained_versions
            ),
            "monotonic_admission": deepcopy(monotonic_admission),
            "selected_item_ids": (
                selected_ids if status == "ready_for_assembly" else []
            ),
            "reserve_item_ids": (
                reserve_ids if status == "ready_for_assembly" else []
            ),
            "revision_item_ids": [
                entry["item_id"] for entry in revision_entries
            ],
            "deferred_revision_item_ids": [
                entry["item_id"] for entry in deferred_revision_entries
            ],
            "excluded_item_ids": removed_ids,
            "reasons": deepcopy(reasons),
            "automatic_removal_suppressed": exploratory_evidence,
            "developmental_override": developmental_override,
            "provisional_item_ids": selected_provisional_ids,
            "developed_candidate_count": candidate_used,
        }
    )

    final_selection_ready = status == "ready_for_assembly"
    update = {
        "psychometric_repair_defer_after_rounds": PSYCHOMETRIC_REPAIR_DEFER_AFTER_ROUNDS,
        "selected_items": (
            [deepcopy(items[item_id]) for item_id in selected_ids]
            if final_selection_ready
            else []
        ),
        "reserve_items": (
            [deepcopy(items[item_id]) for item_id in reserve_ids]
            if final_selection_ready
            else []
        ),
        "items_to_revise": revise_entries,
        "items_to_regenerate": regenerate_entries,
        "items_deferred_for_revision": deferred_revision_entries,
        "removed_items": removed,
        "rejected_items": [
            deepcopy(dict(item))
            for item in state.get("rejected_items") or []
            if isinstance(item, Mapping)
            and item.get("item_id") not in active_delivery_ids
        ],
        "selection_reasons": reasons,
        "selection_results": {
            "status": status,
            "item_bank_id": state.get("item_bank_id"),
            "item_bank_version": state.get("item_bank_version"),
            "selected_count": len(selected_ids) if final_selection_ready else 0,
            "reserve_count": len(reserve_ids) if final_selection_ready else 0,
            "removed_count": len(removed),
            "revision_count": len(revision_entries),
            "defer_count": sum(
                1 for entry in revision_entries if entry.get("action") == "defer"
            ),
            "deferred_revision_count": len(deferred_revision_entries),
            "evidence_level": (
                "exploratory" if exploratory_evidence else "developmental"
            ),
            "automatic_removal_suppressed": exploratory_evidence,
            "developmental_override": developmental_override,
            "provisional_item_ids": selected_provisional_ids,
            "delivery_warnings": [
                deepcopy(provisional_flags[item_id])
                for item_id in selected_provisional_ids
            ],
            "sample_size": sample_size,
            "recommended_sample_size": recommended_size,
            "automatic_selection_minimum_sample_size": (
                automatic_selection_minimum
            ),
            "source_recommendation_counts": source_counts,
            "source_recommendations": source_recommendations,
            "effective_recommendation_counts": effective_counts,
            "effective_recommendations": effective_recommendations,
            "locked_retained_item_versions": deepcopy(
                locked_retained_versions
            ),
            "locked_retained_count": len(locked_retained_versions),
            "monotonic_admission": deepcopy(monotonic_admission),
            "developed_candidate_count": candidate_used,
            "revision_item_ids": [
                entry["item_id"] for entry in revision_entries
            ],
            "deferred_revision_item_ids": [
                entry["item_id"] for entry in deferred_revision_entries
            ],
            "removed_item_ids": removed_ids,
            "next_effect": {
                "repair_items": bool(revision_entries),
                "update_item_bank": bool(
                    removed_ids and not coverage["passed"]
                ),
                "fixed_blueprint_gap": (
                    not revision_entries
                    and not coverage["passed"]
                ),
                "reanalyze_after_bank_change": not coverage["passed"],
                "reassemble": False,
            },
        },
        "blueprint_coverage": coverage,
        "item_pool": remaining_pool,
        "psychometric_selection_history": history,
        "psychometric_repair_history": repair_history,
        "locked_retained_item_versions": locked_retained_versions,
        "best_assembly_candidate": deepcopy(best_assembly_candidate),
        "provisional_item_flags": provisional_flags,
    }
    if not coverage["passed"]:
        update.update(
            {
                "assembled_test": None,
                "test_review_result": None,
                "final_test": None,
                "item_database_ref": None,
                "technical_report": None,
                "virtual_respondent_report": None,
            }
        )
    return {
        "state_update": update,
        "summary": (
            f"题目筛选完成：正式候选 "
            f"{len(selected_ids) if final_selection_ready else 0} 题，"
            f"备用 {len(reserve_ids) if final_selection_ready else 0} 题，"
            f"返修 {sum(1 for entry in revision_entries if entry.get('action') != 'defer')} 题，"
            f"自动 defer {sum(1 for entry in revision_entries if entry.get('action') == 'defer')} 题，"
            f"暂缓返修 {len(deferred_revision_entries)} 题，"
            f"淘汰 {len(removed_ids)} 题；"
            f"状态 {status}。"
        ),
    }
