"""Assemble respondent-facing tests and isolated scoring artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from sjt_system.state import PSJTState
from sjt_system.runtime.io import (
    write_json_atomic as _write_json_atomic,
    write_text_atomic as _write_text_atomic,
)
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.config import PSJT_RESPONSE_INSTRUCTION, PSJT_SCORING_METHOD


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSEMBLY_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "assembled_tests"
DEVELOPMENT_EVIDENCE_NOTICE = (
    "开发期虚拟证据，不代表已经建立真实被试效度"
)


EXPLORATORY_ASSEMBLY_NOTICE = (
    "虚拟样本低于建议规模，统计质量建议未作为入卷淘汰依据；"
    "本测验仅用于流程验证和探索性检查"
)

DEVELOPMENTAL_OVERRIDE_NOTICE = (
    "已达到定向补题轮数上限；为满足蓝图，测验包含带明确质量标记的开发版题目。"
    "这些题目可用于继续开发与人工复核，不应视为已通过正式测量质量门槛。"
)


def _resolve_items(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = state.get("selected_items")
    frozen = state.get("frozen_item_bank")
    if not isinstance(selected, list) or not selected:
        raise ValueError("组卷前 selected_items 不能为空")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("组卷前缺少冻结题库")

    frozen_by_id: dict[str, dict[str, Any]] = {}
    frozen_order: dict[str, int] = {}
    for index, raw_item in enumerate(frozen):
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = deepcopy(dict(raw_item))
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("冻结题库题目缺少 item_id")
        if item_id in frozen_by_id:
            raise ValueError(f"冻结题库包含重复题目 {item_id}")
        frozen_by_id[item_id] = item
        frozen_order[item_id] = index

    resolved: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for raw_item in selected:
        if not isinstance(raw_item, Mapping):
            raise ValueError("selected_items 包含无效题目")
        item_id = raw_item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("selected_items 题目缺少 item_id")
        if item_id in selected_ids:
            raise ValueError(f"selected_items 包含重复题目 {item_id}")
        selected_ids.add(item_id)
        frozen_item = frozen_by_id.get(item_id)
        if frozen_item is None:
            raise ValueError(f"入卷题目 {item_id} 不属于当前冻结题库")
        if raw_item.get("version") != frozen_item.get("version"):
            raise ValueError(f"入卷题目 {item_id} 的版本不是当前冻结版本")
        resolved.append(frozen_item)

    reserve = state.get("reserve_items") or []
    reserve_ids = {
        item.get("item_id")
        for item in reserve
        if isinstance(item, Mapping)
    }
    overlap = selected_ids & reserve_ids
    if overlap:
        raise ValueError(
            "正式题与备选题不能重叠：" + "、".join(sorted(overlap))
        )
    return resolved, [
        deepcopy(dict(item))
        for item in reserve
        if isinstance(item, Mapping)
    ]


def _validate_blueprint_coverage(
    state: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blueprint = state.get("blueprint")
    coverage = state.get("blueprint_coverage")
    if not isinstance(blueprint, Mapping):
        raise ValueError("组卷前缺少蓝图")
    if not isinstance(coverage, Mapping) or not coverage.get("passed"):
        raise ValueError("组卷前蓝图覆盖检查必须通过")

    item_ids_by_cell: dict[str, list[str]] = {}
    for item in items:
        cell_id = item.get("blueprint_cell_id")
        item_id = item.get("item_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"题目 {item_id} 缺少 blueprint_cell_id")
        item_ids_by_cell.setdefault(cell_id, []).append(str(item_id))

    cell_results: list[dict[str, Any]] = []
    expected_total = 0
    blueprint_cell_ids: set[str] = set()
    for raw_cell in blueprint.get("cells") or []:
        if not isinstance(raw_cell, Mapping):
            continue
        cell_id = raw_cell.get("cell_id")
        planned = raw_cell.get("planned_retention_count")
        if (
            not isinstance(cell_id, str)
            or not isinstance(planned, int)
            or isinstance(planned, bool)
            or planned < 0
        ):
            raise ValueError("蓝图包含无效的保留题量")
        blueprint_cell_ids.add(cell_id)
        expected_total += planned
        item_ids = item_ids_by_cell.get(cell_id, [])
        if len(item_ids) != planned:
            raise ValueError(
                f"蓝图单元 {cell_id} 计划 {planned} 题，实际入卷 "
                f"{len(item_ids)} 题"
            )
        cell_results.append(
            {
                "blueprint_cell_id": cell_id,
                "target_dimension_id": raw_cell.get("facet_id"),
                "planned_count": planned,
                "assembled_count": len(item_ids),
                "item_ids": item_ids,
                "passed": True,
            }
        )
    unknown_cells = set(item_ids_by_cell) - blueprint_cell_ids
    if unknown_cells:
        raise ValueError(
            "入卷题目引用未知蓝图单元："
            + "、".join(sorted(unknown_cells))
        )
    if len(items) != expected_total:
        raise ValueError(
            f"蓝图计划入卷 {expected_total} 题，实际 {len(items)} 题"
        )
    return cell_results


def _validate_quality_legacy(
    state: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statistics = state.get("item_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("组卷前缺少 item_statistics")
    selection = state.get("selection_results")
    exploratory_override = bool(
        isinstance(selection, Mapping)
        and selection.get("evidence_level") == "exploratory"
        and selection.get("automatic_removal_suppressed") is True
    )
    developmental_override = bool(
        isinstance(selection, Mapping)
        and selection.get("developmental_override") is True
    )
    # 平台期收卷：接受 revise 题直接入卷（plateau = 停止返修、按现状收卷），
    # 不标开发版标记，蓝图覆盖按需求凑满。
    plateau_finalized = bool(
        isinstance(selection, Mapping)
        and selection.get("plateau_finalized") is True
    )
    recommendations: dict[str, str] = {}
    source_recommendations: dict[str, str] = {}
    effective_recommendations = (
        selection.get("effective_recommendations")
        if isinstance(selection, Mapping)
        and isinstance(selection.get("effective_recommendations"), Mapping)
        else {}
    )
    non_retained_item_ids: list[str] = []
    for item in items:
        item_id = item["item_id"]
        source_recommendation = (
            (statistics.get(item_id) or {})
            .get("quality_evaluation", {})
            .get("recommendation")
        )
        recommendation = effective_recommendations.get(
            item_id,
            source_recommendation,
        )
        if recommendation not in {"retain", "revise", "remove"}:
            raise ValueError(f"题目 {item_id} 缺少有效的最终质量建议")
        source_recommendations[item_id] = str(source_recommendation)
        recommendations[item_id] = recommendation
        if recommendation != "retain":
            non_retained_item_ids.append(item_id)
        if recommendation != "retain" and not (
            exploratory_override
            or developmental_override
            or plateau_finalized
        ):
            raise ValueError(
                f"题目 {item_id} 的最终建议不是 retain，不能入卷"
            )
    return {
        "mode": (
            "exploratory_override"
            if exploratory_override
            else "developmental_override"
            if developmental_override
            else "strict"
        ),
        "provisional": exploratory_override or developmental_override,
        "non_retained_item_ids": non_retained_item_ids,
        "provisional_item_flags": {
            item_id: deepcopy(flag)
            for item_id, flag in (
                state.get("provisional_item_flags") or {}
            ).items()
            if item_id in {item.get("item_id") for item in items}
            and isinstance(flag, Mapping)
        },
        "recommendations": recommendations,
        "source_recommendations": source_recommendations,
    }


def _validate_quality(
    state: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit items by one final disposition; keep old statistics as evidence."""

    statistics = state.get("item_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("组卷前缺少 item_statistics")
    selection = state.get("selection_results")
    final_dispositions = state.get("item_final_dispositions")
    if not isinstance(final_dispositions, Mapping) or not final_dispositions:
        final_dispositions = (
            selection.get("final_dispositions")
            if isinstance(selection, Mapping)
            and isinstance(selection.get("final_dispositions"), Mapping)
            else None
        )
    if final_dispositions is None:
        return _validate_quality_legacy(state, items)
    source_recommendations: dict[str, str] = {}
    effective: dict[str, str] = {}
    warning_item_ids: list[str] = []
    provisional_fill_item_ids: list[str] = []
    for item in items:
        item_id = str(item.get("item_id") or "")
        disposition = final_dispositions.get(item_id)
        disposition_status = (
            disposition.get("status")
            if isinstance(disposition, Mapping)
            else None
        )
        if disposition_status not in {
            "qualified_locked",
            "provisional_plateau_fill",
        }:
            raise ValueError(f"题目 {item_id} 缺少可入卷的最终状态")
        if disposition_status == "provisional_plateau_fill":
            provisional_fill_item_ids.append(item_id)
        status = str(disposition_status)
        disposition_version = disposition.get("item_version")
        if (
            disposition_version is not None
            and disposition_version != item.get("version")
        ):
            raise ValueError(f"题目 {item_id} 的最终状态绑定了旧版本")
        effective[item_id] = status
        source_recommendations[item_id] = str(
            ((statistics.get(item_id) or {}).get("quality_evaluation") or {}).get(
                "recommendation"
            )
        )
        if disposition.get("monitoring_pass") is False:
            warning_item_ids.append(item_id)
    state_flags = state.get("provisional_item_flags") or {}
    fill_flags = {
        item_id: state_flags.get(item_id)
        for item_id in provisional_fill_item_ids
        if isinstance(state_flags.get(item_id), Mapping)
    }
    return {
        "mode": (
            "developmental_override"
            if provisional_fill_item_ids
            else "final_disposition"
        ),
        "provisional": bool(
            provisional_fill_item_ids or warning_item_ids
        ),
        "non_retained_item_ids": [],
        "warning_item_ids": warning_item_ids,
        "provisional_item_flags": fill_flags,
        "recommendations": effective,
        "source_recommendations": source_recommendations,
        "final_dispositions": deepcopy(dict(final_dispositions)),
    }


def _sequence_items(
    items: Sequence[Mapping[str, Any]],
    *,
    variant: int,
) -> list[dict[str, Any]]:
    """Greedily reduce adjacent facet and context clustering."""

    remaining = [deepcopy(dict(item)) for item in items]
    ordered: list[dict[str, Any]] = []
    while remaining:
        previous = ordered[-1] if ordered else {}
        candidates = list(enumerate(remaining))
        index, chosen = min(
            candidates,
            key=lambda pair: (
                int(
                    pair[1].get("blueprint_cell_id")
                    == previous.get("blueprint_cell_id")
                ),
                int(
                    pair[1].get("context_category")
                    == previous.get("context_category")
                ),
                (
                    pair[0] + variant
                )
                % max(1, len(remaining)),
            ),
        )
        ordered.append(chosen)
        remaining.pop(index)
    return ordered


def _respondent_item(
    item: Mapping[str, Any],
    item_number: int,
) -> dict[str, Any]:
    options = item.get("response_options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"题目 {item.get('item_id')} 缺少作答选项")
    public_options: list[dict[str, str]] = []
    for option in options:
        if not isinstance(option, Mapping):
            raise ValueError("题目包含无效作答选项")
        option_id = option.get("option_id")
        text = option.get("text")
        if not isinstance(option_id, str) or not isinstance(text, str):
            raise ValueError("作答选项缺少 option_id 或 text")
        public_options.append({"option_id": option_id, "text": text})
    return {
        "item_number": item_number,
        "public_item_id": f"Q{item_number:03d}",
        "scenario": item.get("scenario"),
        "response_instruction": item.get("response_instruction"),
        "response_options": public_options,
    }


def _render_markdown(respondent_form: Mapping[str, Any]) -> str:
    lines = [
        f"# {respondent_form['title']}",
        "",
        str(respondent_form["evidence_notice"]),
        "",
        "## 作答说明",
        "",
        str(respondent_form["administration"]["response_instruction"]),
        "",
    ]
    for item in respondent_form["items"]:
        lines.extend(
            [
                f"## 第 {item['item_number']} 题",
                "",
                str(item["scenario"]),
                "",
                str(item["response_instruction"]),
                "",
            ]
        )
        lines.extend(
            f"- {option['option_id']}. {option['text']}"
            for option in item["response_options"]
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_test_assembly(
    state: PSJTState,
    *,
    output_root: str | Path = DEFAULT_ASSEMBLY_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Create respondent-safe and scoring artifacts from selected items."""

    selection = state.get("selection_results")
    if (
        not isinstance(selection, Mapping)
        or selection.get("status") != "ready_for_assembly"
    ):
        raise ValueError("组卷前题目筛选必须达到 ready_for_assembly")
    if (
        not state.get("item_bank_id")
        or not state.get("item_bank_version")
        or not state.get("item_bank_fingerprint")
    ):
        raise ValueError("组卷前缺少有效的冻结题库身份")

    items, reserve = _resolve_items(state)
    cell_results = _validate_blueprint_coverage(state, items)
    quality_gate = _validate_quality(state, items)
    provisional = bool(quality_gate["provisional"])
    previous_review = state.get("test_review_result")
    is_reassembly = (
        isinstance(previous_review, Mapping)
        and previous_review.get("decision") == "REASSEMBLE"
    )
    reassembly_round = int(state.get("reassembly_round") or 0)
    if is_reassembly:
        if reassembly_round >= int(state.get("max_reassembly_rounds") or 0):
            raise ValueError("已达到最大重新组卷轮数")
        reassembly_round += 1
    items = _sequence_items(items, variant=reassembly_round)

    specification = state.get("test_specification") or {}
    construct_profile = state.get("construct_profile") or {}
    assembled_at = utc_timestamp()
    test_id = (
        f"{state['run_id']}-bank-v{state['item_bank_version']}-test"
    )
    quality_mode = quality_gate["mode"]
    title = (
        "情境判断测验（探索性版本）"
        if quality_mode == "exploratory_override"
        else "情境判断测验（开发版本）"
        if quality_mode == "developmental_override"
        else "情境判断测验"
    )
    evidence_notice = (
        EXPLORATORY_ASSEMBLY_NOTICE
        if quality_mode == "exploratory_override"
        else DEVELOPMENTAL_OVERRIDE_NOTICE
        if quality_mode == "developmental_override"
        else DEVELOPMENT_EVIDENCE_NOTICE
    )
    response_instruction = PSJT_RESPONSE_INSTRUCTION
    respondent_items = [
        _respondent_item(item, index)
        for index, item in enumerate(items, 1)
    ]
    respondent_form = {
        "schema_version": 1,
        "test_id": test_id,
        "title": title,
        "target_population": specification.get("target_population"),
        "item_count": len(respondent_items),
        "administration": {
            "response_instruction": response_instruction,
            "estimated_time_minutes": None,
        },
        "items": respondent_items,
        "evidence_notice": evidence_notice,
    }
    scoring_items = [
        {
            "item_number": index,
            "public_item_id": f"Q{index:03d}",
            "item_id": item["item_id"],
            "item_version": item.get("version"),
            "blueprint_cell_id": item.get("blueprint_cell_id"),
            "target_dimension_id": item.get("target_dimension_id"),
            "context_category": item.get("context_category"),
            "scoring_key": deepcopy(item.get("scoring_key") or {}),
        }
        for index, item in enumerate(items, 1)
    ]
    scoring_key = {
        "schema_version": 1,
        "test_id": test_id,
        "scoring_method": PSJT_SCORING_METHOD,
        "items": scoring_items,
        "security_notice": "评分键不得与被试卷一同发放。",
    }
    blueprint_summary = {
        "blueprint_id": (state.get("blueprint") or {}).get("blueprint_id"),
        "construct_selection": deepcopy(
            specification.get("construct_selection")
        ),
        "construct_profile_hash": construct_profile.get("profile_hash"),
        "cells": cell_results,
        "combination_optimizer": deepcopy(
            (state.get("blueprint_coverage") or {}).get("optimizer")
        ),
        "passed": True,
    }

    fingerprint = str(state["item_bank_fingerprint"])
    output_dir = (
        Path(output_root)
        / str(state["run_id"])
        / f"bank-v{state['item_bank_version']}-{fingerprint[:12]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    respondent_json_path = output_dir / "respondent_test.json"
    respondent_markdown_path = output_dir / "respondent_test.md"
    scoring_key_path = output_dir / "scoring_key.json"
    assembly_manifest_path = output_dir / "assembly_manifest.json"

    _write_json_atomic(respondent_json_path, respondent_form)
    _write_text_atomic(
        respondent_markdown_path,
        _render_markdown(respondent_form),
    )
    _write_json_atomic(scoring_key_path, scoring_key)
    manifest = {
        "schema_version": 1,
        "test_id": test_id,
        "run_id": state["run_id"],
        "item_bank_id": state["item_bank_id"],
        "item_bank_version": state["item_bank_version"],
        "item_bank_fingerprint": fingerprint,
        "assembled_at": assembled_at,
        "item_count": len(items),
        "selected_item_ids": [item["item_id"] for item in items],
        "reserve_item_ids": [
            item.get("item_id") for item in reserve
        ],
        "provisional": provisional,
        "quality_gate": quality_gate,
        "blueprint_summary": blueprint_summary,
        "evidence_notice": evidence_notice,
        "files": {
            "respondent_test_json": str(respondent_json_path.resolve()),
            "respondent_test_markdown": str(
                respondent_markdown_path.resolve()
            ),
            "scoring_key": str(scoring_key_path.resolve()),
        },
    }
    _write_json_atomic(assembly_manifest_path, manifest)

    assembled = {
        **manifest,
        "files": {
            **manifest["files"],
            "assembly_manifest": str(assembly_manifest_path.resolve()),
        },
        "items": deepcopy(items),
        "respondent_form": respondent_form,
        "scoring_key": scoring_key,
        "blueprint_coverage": deepcopy(state["blueprint_coverage"]),
    }
    return {
        "state_update": {
            "assembled_test": assembled,
            "test_review_result": None,
            "reassembly_round": reassembly_round,
        },
        "summary": (
            (
                f"已按蓝图生成探索性测验，共 {len(items)} 道题；"
                f"{len(reserve)} 道题保留为备选；"
                "统计质量门槛因小样本暂未执行。"
            )
            if provisional
            else (
                f"已按蓝图组卷，共 {len(items)} 道正式题；"
                f"{len(reserve)} 道合格题保留为备选。"
            )
        ),
    }
