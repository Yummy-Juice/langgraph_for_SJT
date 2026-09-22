"""Theory-guided whole-test candidate-form optimization.

The LLM in this module is an orchestrator. Numerical evaluation and the
blueprint constraints remain program-owned so a model cannot invent item IDs or
psychometric results.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from sjt_system.agent.client import get_model
from sjt_system.agent.json_parsing import parse_model_json_response
from sjt_system.agent.retry import ainvoke_model_with_retry
from sjt_system.authoring.generation_plan import planned_retention_count
from sjt_system.knowledge.behavior_evidence import StrictModel
from sjt_system.prompt.form_optimizer_prompt import FORM_OPTIMIZER_PROMPT
from sjt_system.evaluation.selection import (
    _combination_admission_key,
    _combination_metrics,
    _optimizer_datasets,
)
from sjt_system.evaluation.form_metrics import (
    PLATEAU_DEFAULT_MIN_DELTA,
    VIRTUAL_FORM_ICC_DEFAULT_MINIMUM,
    batch_provisional_form_quality,
    build_provisional_form_metrics,
    form_quality_summary,
    prepare_provisional_form_metric_context,
)


MAX_FORM_SEARCH_COMBINATIONS = 200_000
MAX_FORM_SEARCH_RESULTS = 8
MAX_FORM_AGENT_TOOL_ROUNDS = 6


class FormOptimizationDecision(StrictModel):
    selected_item_ids: list[str] = Field(min_length=0)
    rationale: str = Field(min_length=1, max_length=2000)
    theory_coverage_summary: str = Field(min_length=1, max_length=1000)
    evaluation_status: str = Field(pattern=r"^(validated|infeasible)$")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _item_specifications(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["specification_id"]): deepcopy(dict(row))
        for row in state.get("item_specifications") or []
        if isinstance(row, Mapping) and row.get("specification_id")
    }


def _facet_theory_context(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile = state.get("construct_profile") or {}
    result = []
    for facet in profile.get("facets") or []:
        if not isinstance(facet, Mapping):
            continue
        result.append(
            {
                "facet_id": facet.get("facet_id"),
                "facet_name": facet.get("facet_name"),
                "definition": facet.get("definition"),
                "high_behavior": facet.get("high_behavior"),
                "low_behavior": facet.get("low_behavior"),
                "common_confounds": facet.get("common_confounds") or [],
                "inappropriate_conditions": facet.get(
                    "inappropriate_conditions"
                )
                or [],
                "behavior_evidence": [
                    {
                        "behavior_id": evidence.get("behavior_id"),
                        "behavior_dimension": evidence.get(
                            "behavior_dimension"
                        ),
                        "high_expression": evidence.get("high_expression"),
                        "low_expression": evidence.get("low_expression"),
                        "boundary_condition": evidence.get(
                            "boundary_condition"
                        ),
                    }
                    for evidence in facet.get("behavior_evidence") or []
                    if isinstance(evidence, Mapping)
                ],
            }
        )
    return result


def _candidate_descriptor(
    item: Mapping[str, Any],
    specification: Mapping[str, Any] | None,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    quality = statistics.get("quality_evaluation") or {}
    facet_citc = quality.get("facet_citc") or {}
    specificity = quality.get("virtual_target_specificity") or {}
    target = specificity.get("target_spearman") or {}
    same_domain = specificity.get("same_domain_non_target") or {}
    cross_domain = specificity.get("cross_domain_non_target") or {}
    specification = specification or {}
    return {
        "item_id": item.get("item_id"),
        "blueprint_cell_id": item.get("blueprint_cell_id"),
        "facet_id": item.get("target_dimension_id"),
        "behavior_id": specification.get("behavior_evidence_id"),
        "mechanism_id": specification.get("mechanism_id"),
        "situation_id": specification.get("situation_id"),
        "activation_mechanism": specification.get("activation_mechanism"),
        "context_seed": specification.get("context_seed"),
        "core_tension": specification.get("core_tension"),
        "scenario": item.get("scenario"),
        "facet_citc": facet_citc.get("r"),
        "target_rho": target.get("rho"),
        "same_domain_margin": same_domain.get("specificity_margin"),
        "cross_domain_margin": cross_domain.get("specificity_margin"),
        "difficulty": statistics.get("difficulty"),
    }


def _candidate_groups(
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specifications = _item_specifications(state)
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for item in candidates:
        cell_id = item.get("blueprint_cell_id")
        if isinstance(cell_id, str) and cell_id:
            by_cell.setdefault(cell_id, []).append(item)
    groups = []
    for cell in (state.get("blueprint") or {}).get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("cell_id") or "")
        groups.append(
            {
                "cell_id": cell_id,
                "facet_id": cell.get("facet_id"),
                "behavior_id": cell.get("behavior_id"),
                "planned_retention_count": int(
                    cell.get("planned_retention_count") or 0
                ),
                "candidate_count": len(by_cell.get(cell_id) or []),
                "candidates": [
                    _candidate_descriptor(
                        item,
                        specifications.get(str(item.get("item_id"))),
                        statistics.get(str(item.get("item_id"))) or {},
                    )
                    for item in by_cell.get(cell_id) or []
                ],
            }
        )
    return groups


def _theory_profile(
    item_ids: Sequence[str],
    item_map: Mapping[str, Mapping[str, Any]],
    specifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    facets = set()
    behaviors = set()
    mechanisms = set()
    situations = set()
    for item_id in item_ids:
        item = item_map.get(str(item_id)) or {}
        specification = specifications.get(str(item_id)) or {}
        if item.get("target_dimension_id"):
            facets.add(str(item["target_dimension_id"]))
        if specification.get("behavior_evidence_id"):
            behaviors.add(str(specification["behavior_evidence_id"]))
        if specification.get("mechanism_id"):
            mechanisms.add(str(specification["mechanism_id"]))
        if specification.get("situation_id"):
            situations.add(str(specification["situation_id"]))
    return {
        "facet_count": len(facets),
        "behavior_evidence_count": len(behaviors),
        "mechanism_count": len(mechanisms),
        "situation_count": len(situations),
        "facet_ids": sorted(facets),
        "behavior_ids": sorted(behaviors),
        "mechanism_ids": sorted(mechanisms),
        "situation_ids": sorted(situations),
    }


def _evaluate_form(
    item_ids: Sequence[str],
    *,
    item_map: Mapping[str, Mapping[str, Any]],
    specifications: Mapping[str, Mapping[str, Any]],
    item_statistics: Mapping[str, Mapping[str, Any]],
    test_statistics: Mapping[str, Any] | None,
    blueprint: Mapping[str, Any],
    form_metric_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_ids = [str(item_id) for item_id in item_ids]
    errors: list[str] = []
    if len(selected_ids) != len(set(selected_ids)):
        errors.append("selected_item_ids 不得重复")
    unknown = [item_id for item_id in selected_ids if item_id not in item_map]
    if unknown:
        errors.append("包含候选题库之外的题目")
    required_by_cell = {
        str(cell.get("cell_id")): int(cell.get("planned_retention_count") or 0)
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping) and cell.get("cell_id")
    }
    selected_by_cell: dict[str, list[str]] = {}
    for item_id in selected_ids:
        cell_id = str((item_map.get(item_id) or {}).get("blueprint_cell_id") or "")
        selected_by_cell.setdefault(cell_id, []).append(item_id)
    for cell_id, required in required_by_cell.items():
        actual = len(selected_by_cell.get(cell_id) or [])
        if actual != required:
            errors.append(
                f"蓝图单元 {cell_id} 需要 {required} 题，实际 {actual} 题"
            )
    unknown_cells = set(selected_by_cell) - set(required_by_cell)
    if unknown_cells:
        errors.append("选入题目包含未知蓝图单元")

    references = []
    for item_id in selected_ids:
        specification = specifications.get(item_id) or {}
        references.append(
            (
                str(specification.get("mechanism_id") or ""),
                str(specification.get("situation_id") or ""),
            )
        )
    if len(references) != len(set(references)):
        errors.append("测验内重复使用了相同机制—情境引用")

    metrics: dict[str, Any] = {}
    datasets = _optimizer_datasets(test_statistics)
    if not errors and datasets:
        combination_metrics = _combination_metrics(
            selected_ids,
            datasets=datasets,
            items=item_map,
        )
        if combination_metrics is not None:
            metrics["combination"] = combination_metrics
        else:
            metrics["combination"] = None
    else:
        metrics["combination"] = None
    item_quality = []
    for item_id in selected_ids:
        quality = (item_statistics.get(item_id) or {}).get(
            "quality_evaluation"
        ) or {}
        item_quality.append(
            {
                "item_id": item_id,
                "facet_citc": (quality.get("facet_citc") or {}).get("r"),
                "target_rho": (
                    (quality.get("virtual_target_specificity") or {})
                    .get("target_spearman")
                    or {}
                ).get("rho"),
                "same_domain_margin": (
                    (quality.get("virtual_target_specificity") or {})
                    .get("same_domain_non_target")
                    or {}
                ).get("specificity_margin"),
                "cross_domain_margin": (
                    (quality.get("virtual_target_specificity") or {})
                    .get("cross_domain_non_target")
                    or {}
                ).get("specificity_margin"),
            }
        )
    theory = _theory_profile(selected_ids, item_map, specifications)
    if not errors:
        metrics["whole_test"] = build_provisional_form_metrics(
            {
                "test_statistics": test_statistics or {},
            },
            selected_ids,
            context=form_metric_context,
        )
    else:
        metrics["whole_test"] = None
    return _json_safe(
        {
            "valid": not errors,
            "errors": errors,
            "selected_item_ids": selected_ids,
            "metrics": metrics,
            "item_quality": item_quality,
            "theory_profile": theory,
            "metrics_available": bool(datasets),
        }
    )


def _fallback_metrics_key(
    item_ids: Sequence[str],
    item_statistics: Mapping[str, Mapping[str, Any]],
) -> tuple[float, ...]:
    values = []
    for item_id in item_ids:
        quality = (item_statistics.get(item_id) or {}).get(
            "quality_evaluation"
        ) or {}
        citc = (quality.get("facet_citc") or {}).get("r")
        target = (
            (quality.get("virtual_target_specificity") or {})
            .get("target_spearman")
            or {}
        ).get("rho")
        same = (
            (quality.get("virtual_target_specificity") or {})
            .get("same_domain_non_target")
            or {}
        ).get("specificity_margin")
        cross = (
            (quality.get("virtual_target_specificity") or {})
            .get("cross_domain_non_target")
            or {}
        ).get("specificity_margin")
        values.append(
            (
                float(citc) if isinstance(citc, (int, float)) else -1.0,
                float(target) if isinstance(target, (int, float)) else -1.0,
                float(same) if isinstance(same, (int, float)) else -1.0,
                float(cross) if isinstance(cross, (int, float)) else -1.0,
            )
        )
    if not values:
        return (-1.0, -1.0, -1.0, -1.0)
    return tuple(min(row[index] for row in values) for index in range(4))


def _whole_test_metrics_key(metrics: Mapping[str, Any] | None) -> tuple[float, ...] | None:
    """Rank complete forms by ICC eligibility and the current objective.

    Current v3 forms are ranked by target known-groups Hedges' g, then the
    minimum discriminant correlation gap, then target-facet Spearman rho.  The
    legacy tuple is retained only so historical experiment fixtures and old
    checkpoints remain readable.
    """
    if not isinstance(metrics, Mapping) or metrics.get("status") != "complete":
        return None
    summary = form_quality_summary(metrics)
    primary = summary.get("objective_primary")
    secondary = summary.get("objective_secondary")
    tertiary = summary.get("objective_tertiary")
    recovery = summary.get("target_recovery_component")
    selectivity = summary.get("construct_selectivity")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in (primary, secondary)
    ):
        return None
    if summary.get("objective_source") == "ipip_human_style_v3":
        if not isinstance(tertiary, (int, float)) or isinstance(tertiary, bool):
            return None
        return (
            1.0 if summary.get("eligible_for_best_so_far") else 0.0,
            float(primary),
            float(secondary),
            float(tertiary),
        )
    if summary.get("objective_source") == "ipip_human_style":
        return (
            1.0 if summary.get("eligible_for_best_so_far") else 0.0,
            float(primary),
            float(secondary),
            0.0,
        )
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in (recovery, selectivity)
    ):
        return None
    return (
        1.0 if summary.get("eligible_for_best_so_far") else 0.0,
        float(primary),
        float(selectivity),
        float(recovery),
    )


def _whole_test_objective_improves(
    current_key: tuple[float, ...] | None,
    incumbent_key: tuple[float, ...] | None,
    *,
    min_delta: float,
) -> bool:
    """Apply the same meaningful-improvement rule as the plateau detector."""

    if current_key is None:
        return False
    if incumbent_key is None:
        return current_key[0] > 0.0
    if current_key[0] < incumbent_key[0]:
        return False
    if current_key[0] > incumbent_key[0]:
        return True
    current_primary, incumbent_primary = current_key[1], incumbent_key[1]
    if current_primary <= incumbent_primary + float(min_delta):
        return False
    # For v3 keys the second component is delta_min and the third is target
    # rho.  Do not accept a larger target g if construct discrimination falls
    # or convergent validity drops by more than the configured tolerance.
    if current_key[2] < incumbent_key[2] - 1e-12:
        return False
    return current_key[3] >= incumbent_key[3] - 0.02


def _historical_best_form(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the highest eligible historical form under the active metric.

    A checkpoint may contain pre-IPIP rounds.  Once the current state carries
    the five-facet IPIP reference, those old Q values are not comparable and
    must not become the incumbent for the new objective.
    """
    test_statistics = state.get("test_statistics") or {}
    reference_questionnaires = (
        test_statistics.get("reference_questionnaires")
        if isinstance(test_statistics, Mapping)
        else None
    )
    ipip_meta = (
        reference_questionnaires.get("ipip_neo")
        if isinstance(reference_questionnaires, Mapping)
        else None
    )
    active_ipip_objective = bool(
        isinstance(ipip_meta, Mapping)
        and isinstance(ipip_meta.get("facets"), list)
        and ipip_meta.get("facets")
    )
    best: dict[str, Any] | None = None
    for entry in state.get("psychometric_iteration_history") or []:
        if not isinstance(entry, Mapping):
            continue
        fm = entry.get("form_metrics") or {}
        summary = form_quality_summary(fm)
        if active_ipip_objective and summary.get("objective_source") not in {
            "ipip_human_style",
            "ipip_human_style_v3",
        }:
            continue
        key = _whole_test_metrics_key(fm)
        item_ids = entry.get("form_item_ids") or []
        if not (
            key is not None
            and key[0] == 1.0
            and isinstance(item_ids, list)
            and item_ids
        ):
            continue
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "item_ids": [str(item_id) for item_id in item_ids],
                "round": int(entry.get("analysis_round") or 0),
            }
    return best


def _search_best_test_forms(
    *,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    item_statistics: Mapping[str, Mapping[str, Any]],
    test_statistics: Mapping[str, Any] | None,
    max_results: int,
    form_metric_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item_map = {
        str(item.get("item_id")): dict(item)
        for item in candidates
        if item.get("item_id")
    }
    specifications = _item_specifications(state)
    groups = _candidate_groups(state, candidates, item_statistics)
    choices: list[list[tuple[str, ...]]] = []
    for group in groups:
        planned = int(group.get("planned_retention_count") or 0)
        ids = tuple(
            str(candidate.get("item_id"))
            for candidate in group.get("candidates") or []
            if candidate.get("item_id")
        )
        if planned < 0 or len(ids) < planned:
            return {
                "status": "infeasible",
                "reason": (
                    f"蓝图单元 {group.get('cell_id')} 候选不足："
                    f"需要 {planned}，实际 {len(ids)}"
                ),
                "forms": [],
            }
        choices.append(list(itertools.combinations(ids, planned)))
    combination_count = 1
    for group_choices in choices:
        combination_count *= max(1, len(group_choices))
    if combination_count > MAX_FORM_SEARCH_COMBINATIONS:
        return {
            "status": "infeasible",
            "reason": (
                f"候选组合数 {combination_count} 超过上限 "
                f"{MAX_FORM_SEARCH_COMBINATIONS}"
            ),
            "forms": [],
        }

    if form_metric_context is not None and not _optimizer_datasets(test_statistics):
        combinations = [
            [item_id for choice in grouped for item_id in choice]
            for grouped in itertools.product(*choices)
        ]
        valid_combinations = []
        for item_ids in combinations:
            references = [
                (
                    str((specifications.get(item_id) or {}).get("mechanism_id") or ""),
                    str((specifications.get(item_id) or {}).get("situation_id") or ""),
                )
                for item_id in item_ids
            ]
            if len(references) == len(set(references)):
                valid_combinations.append(item_ids)
        batch_quality = batch_provisional_form_quality(
            form_metric_context,
            valid_combinations,
        )
        if batch_quality is not None:
            ranked: list[dict[str, Any]] = []
            for index, item_ids in enumerate(valid_combinations):
                theory = _theory_profile(item_ids, item_map, specifications)
                fallback_key = _fallback_metrics_key(item_ids, item_statistics)
                ranking_key = (
                    1.0
                    if float(batch_quality["stability_proxy"][index])
                    >= VIRTUAL_FORM_ICC_DEFAULT_MINIMUM
                    else 0.0,
                    float(batch_quality["ipip_target_known_groups_hedges_g"][index]),
                    float(batch_quality["ipip_discriminant_delta_min"][index]),
                    float(batch_quality["ipip_target_facet_spearman_rho"][index]),
                    *fallback_key,
                    int(theory.get("mechanism_count") or 0),
                    int(theory.get("situation_count") or 0),
                )
                ranked.append(
                    {
                        "selected_item_ids": item_ids,
                        "ranking_key": list(ranking_key),
                        "theory_profile": theory,
                    }
                )
            ranked.sort(
                key=lambda value: tuple(value["ranking_key"]),
                reverse=True,
            )
            evaluated: list[dict[str, Any]] = []
            for candidate in ranked[:MAX_FORM_SEARCH_RESULTS]:
                evaluation = _evaluate_form(
                    candidate["selected_item_ids"],
                    item_map=item_map,
                    specifications=specifications,
                    item_statistics=item_statistics,
                    test_statistics=test_statistics,
                    blueprint=state.get("blueprint") or {},
                    form_metric_context=form_metric_context,
                )
                evaluated.append(
                    {
                        "selected_item_ids": candidate["selected_item_ids"],
                        "metrics": evaluation.get("metrics"),
                        "theory_profile": candidate["theory_profile"],
                        "ranking_key": candidate["ranking_key"],
                    }
                )
            evaluated.sort(
                key=lambda candidate: (
                    _whole_test_metrics_key(
                        (candidate.get("metrics") or {}).get("whole_test")
                    )
                    or (-1.0, -1.0, -1.0, -1.0),
                    tuple(candidate.get("ranking_key") or []),
                ),
                reverse=True,
            )
            return {
                "status": "complete" if evaluated else "infeasible",
                "combination_count": combination_count,
                "forms": evaluated[
                    : max(1, min(max_results, MAX_FORM_SEARCH_RESULTS))
                ],
            }

    evaluated: list[dict[str, Any]] = []
    for grouped in itertools.product(*choices):
        item_ids = [item_id for choice in grouped for item_id in choice]
        evaluation = _evaluate_form(
            item_ids,
            item_map=item_map,
            specifications=specifications,
            item_statistics=item_statistics,
            test_statistics=test_statistics,
            blueprint=state.get("blueprint") or {},
            form_metric_context=form_metric_context,
        )
        if not evaluation.get("valid"):
            continue
        combination = (evaluation.get("metrics") or {}).get("combination")
        whole_test = (evaluation.get("metrics") or {}).get("whole_test")
        whole_key = _whole_test_metrics_key(whole_test)
        if whole_key is not None:
            if isinstance(combination, Mapping):
                combination_key = _combination_admission_key(combination)
            else:
                combination_key = _fallback_metrics_key(item_ids, item_statistics)
            key = (*whole_key, *combination_key)
        elif isinstance(combination, Mapping):
            key = _combination_admission_key(combination)
        else:
            key = _fallback_metrics_key(item_ids, item_statistics)
        theory = evaluation.get("theory_profile") or {}
        tie_break = (
            int(theory.get("mechanism_count") or 0),
            int(theory.get("situation_count") or 0),
        )
        evaluated.append(
            {
                "selected_item_ids": item_ids,
                "metrics": evaluation.get("metrics"),
                "theory_profile": theory,
                "ranking_key": [*key, *tie_break],
            }
        )
    evaluated.sort(key=lambda value: tuple(value["ranking_key"]), reverse=True)
    return {
        "status": "complete" if evaluated else "infeasible",
        "combination_count": combination_count,
        "forms": evaluated[: max(1, min(max_results, MAX_FORM_SEARCH_RESULTS))],
    }


def _build_tools(
    *,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    item_statistics: Mapping[str, Mapping[str, Any]],
    test_statistics: Mapping[str, Any] | None,
    form_metric_context: Mapping[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    item_map = {
        str(item.get("item_id")): dict(item)
        for item in candidates
        if item.get("item_id")
    }
    specifications = _item_specifications(state)

    @tool("get_candidate_groups")
    def get_candidate_groups() -> dict[str, Any]:
        """Return theory metadata and candidates grouped by blueprint cell."""

        return {
            "final_item_count": planned_retention_count(
                state.get("blueprint") or {}
            ),
            "theory": _facet_theory_context(state),
            "groups": _candidate_groups(state, candidates, item_statistics),
        }

    @tool("evaluate_test_form")
    def evaluate_test_form(item_ids: list[str]) -> dict[str, Any]:
        """Evaluate one complete form using program-owned psychometric code."""

        return _evaluate_form(
            item_ids,
            item_map=item_map,
            specifications=specifications,
            item_statistics=item_statistics,
            test_statistics=test_statistics,
            blueprint=state.get("blueprint") or {},
            form_metric_context=form_metric_context,
        )

    @tool("search_best_test_forms")
    def search_best_test_forms(max_results: int = 5) -> dict[str, Any]:
        """Search feasible complete forms and return the best scored forms."""

        return _search_best_test_forms(
            state=state,
            candidates=candidates,
            item_statistics=item_statistics,
            test_statistics=test_statistics,
            max_results=max_results,
            form_metric_context=form_metric_context,
        )

    return [get_candidate_groups, evaluate_test_form, search_best_test_forms], {
        tool.name: tool for tool in (get_candidate_groups, evaluate_test_form, search_best_test_forms)
    }


async def _invoke_form_optimizer_agent(
    *,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    item_statistics: Mapping[str, Mapping[str, Any]],
    test_statistics: Mapping[str, Any] | None,
    form_metric_context: Mapping[str, Any] | None = None,
) -> tuple[FormOptimizationDecision, dict[str, Any], int]:
    tools, tool_map = _build_tools(
        state=state,
        candidates=candidates,
        item_statistics=item_statistics,
        test_statistics=test_statistics,
        form_metric_context=form_metric_context,
    )
    model_id = os.getenv("FORM_OPTIMIZER_MODEL_ID") or None
    model = get_model(model_id, temperature=0.1).bind_tools(tools)
    input_data = {
        "test_specification": {
            key: state.get("test_specification", {}).get(key)
            for key in (
                "target_construct",
                "target_population",
                "final_item_count",
                "output_language",
            )
            if isinstance(state.get("test_specification"), Mapping)
        },
        "blueprint_summary": {
            "cell_count": len((state.get("blueprint") or {}).get("cells") or []),
            "final_item_count": planned_retention_count(
                state.get("blueprint") or {}
            ),
        },
        "instruction": "先调用工具，再返回一套完整的最终测验题目ID。",
    }
    messages: list[Any] = [
        SystemMessage(content=FORM_OPTIMIZER_PROMPT),
        HumanMessage(
            content=json.dumps(_json_safe(input_data), ensure_ascii=False)
        ),
    ]
    for round_index in range(1, MAX_FORM_AGENT_TOOL_ROUNDS + 1):
        response = await ainvoke_model_with_retry(
            model,
            messages,
            job_label="测验组合优化Agent",
            max_attempts=1,
        )
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                name = str(call.get("name") or "")
                call_id = str(call.get("id") or f"form-tool-{round_index}")
                runnable = tool_map.get(name)
                if runnable is None:
                    result = {"error": f"未知工具：{name}"}
                else:
                    try:
                        result = runnable.invoke(call.get("args") or {})
                    except Exception as exc:
                        result = {"error": f"工具执行失败：{exc}"}
                messages.append(
                    ToolMessage(
                        content=json.dumps(_json_safe(result), ensure_ascii=False),
                        tool_call_id=call_id,
                    )
                )
            continue
        decision = FormOptimizationDecision.model_validate(
            parse_model_json_response(response)
        )
        evaluation = _evaluate_form(
            decision.selected_item_ids,
            item_map={
                str(item.get("item_id")): dict(item)
                for item in candidates
                if item.get("item_id")
            },
            specifications=_item_specifications(state),
            item_statistics=item_statistics,
            test_statistics=test_statistics,
            blueprint=state.get("blueprint") or {},
            form_metric_context=form_metric_context,
        )
        if decision.evaluation_status == "validated":
            if not evaluation.get("valid"):
                raise ValueError(
                    "测验组合Agent返回的组合未通过程序校验："
                    + "；".join(evaluation.get("errors") or [])
                )
            expected = planned_retention_count(state.get("blueprint") or {})
            if len(decision.selected_item_ids) != expected:
                raise ValueError(
                    f"测验组合Agent返回 {len(decision.selected_item_ids)} 题，"
                    f"但蓝图要求 {expected} 题"
                )
        return decision, evaluation, round_index
    raise ValueError("测验组合Agent在工具调用轮次上限内没有返回最终组合")


async def optimize_test_form_with_agent(
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    item_statistics: Mapping[str, Mapping[str, Any]],
    test_statistics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Select one theory-valid candidate per blueprint cell with an LLM tool loop."""

    form_metric_context = prepare_provisional_form_metric_context(state)
    deterministic = _search_best_test_forms(
        state=state,
        candidates=candidates,
        item_statistics=item_statistics,
        test_statistics=test_statistics,
        max_results=1,
        form_metric_context=form_metric_context,
    )
    if deterministic.get("status") != "complete":
        raise ValueError(str(deterministic.get("reason") or "没有可行测验组合"))
    try:
        decision, evaluation, tool_rounds = await _invoke_form_optimizer_agent(
            state=state,
            candidates=candidates,
            item_statistics=item_statistics,
            test_statistics=test_statistics,
            form_metric_context=form_metric_context,
        )
        selected_ids = list(decision.selected_item_ids)
        mode = "llm_tool_guided"
        fallback_reason = None
    except Exception as exc:
        best = (deterministic.get("forms") or [])[0]
        selected_ids = list(best.get("selected_item_ids") or [])
        evaluation = _evaluate_form(
            selected_ids,
            item_map={
                str(item.get("item_id")): dict(item)
                for item in candidates
                if item.get("item_id")
            },
            specifications=_item_specifications(state),
            item_statistics=item_statistics,
            test_statistics=test_statistics,
            blueprint=state.get("blueprint") or {},
            form_metric_context=form_metric_context,
        )
        mode = "deterministic_fallback"
        fallback_reason = str(exc)
        tool_rounds = 0
        decision = FormOptimizationDecision(
            selected_item_ids=selected_ids,
            rationale="测验组合Agent不可用，采用程序确定性最优组合。",
            theory_coverage_summary="程序按蓝图单元、机制—情境引用和可用测量指标完成约束筛选。",
            evaluation_status="validated",
        )
    if not evaluation.get("valid"):
        raise ValueError(
            "最终测验组合未通过程序校验："
            + "；".join(evaluation.get("errors") or [])
        )
    # Historical incumbent protection: if the previous best form is still
    # available, replace it only after the current two-dimensional objective
    # exceeds the incumbent by the configured meaningful increment.
    historical = _historical_best_form(state)
    if historical is not None:
        item_map_here = {
            str(item.get("item_id")): dict(item)
            for item in candidates
            if item.get("item_id")
        }
        if all(iid in item_map_here for iid in historical["item_ids"]):
            historical_evaluation = _evaluate_form(
                historical["item_ids"],
                item_map=item_map_here,
                specifications=_item_specifications(state),
                item_statistics=item_statistics,
                test_statistics=test_statistics,
                blueprint=state.get("blueprint") or {},
                form_metric_context=form_metric_context,
            )
            if historical_evaluation.get("valid"):
                historical_key = _whole_test_metrics_key(
                    (historical_evaluation.get("metrics") or {}).get("whole_test")
                )
                current_key = _whole_test_metrics_key(
                    (evaluation.get("metrics") or {}).get("whole_test")
                )
                min_delta = float(
                    state.get("psychometric_plateau_min_delta")
                    if state.get("psychometric_plateau_min_delta") is not None
                    else PLATEAU_DEFAULT_MIN_DELTA
                )
                if not _whole_test_objective_improves(
                    current_key,
                    historical_key,
                    min_delta=min_delta,
                ):
                    selected_ids = list(historical["item_ids"])
                    evaluation = historical_evaluation
                    mode = f"historical_best_hold (round {historical['round']})"
                    fallback_reason = (
                        "历史最佳组合保底：本轮未优于历史 best，沿用上一轮最优组卷"
                    )
    selected_set = set(selected_ids)
    reserve_ids = [
        str(item.get("item_id"))
        for item in candidates
        if item.get("item_id") and str(item.get("item_id")) not in selected_set
    ]
    return {
        "status": "validated",
        "mode": mode,
        "selected_item_ids": selected_ids,
        "reserve_item_ids": reserve_ids,
        "metrics": evaluation.get("metrics") or {},
        "theory_profile": evaluation.get("theory_profile") or {},
        "rationale": decision.rationale,
        "theory_coverage_summary": decision.theory_coverage_summary,
        "tool_rounds": tool_rounds,
        "fallback_reason": fallback_reason,
    }
