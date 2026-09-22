"""Item contracts, review transitions, revision checks, and history."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

from sjt_system.state import PSJTState, RouteType
from sjt_system.authoring.context import (
    build_item_pattern_profile,
    text_similarity,
)
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.config import PSJT_OPTION_COUNT, PSJT_RESPONSE_INSTRUCTION


ITEM_AGENT_OUTPUT_FIELDS: dict[str, set[str]] = {
    "generate_item": {
        "current_item",
        "current_item_specification",
        "item_specifications",
        "item_skeletons",
        "skeleton_reviews",
        "skeleton_review_history",
    },
    "regenerate_item": {
        "current_item",
        "current_item_specification",
        "item_specifications",
        "item_skeletons",
        "skeleton_reviews",
        "skeleton_review_history",
    },
    "revise_item": {"current_item"},
    "review_item": {
        "current_item_review",
        "review_process_status",
        "item_content_status",
        "current_review_request_id",
        "current_review_item_id",
        "current_review_item_version",
        "current_review_retry_count",
    },
}
_ITEM_CONTENT_FIELDS = (
    "context_signature",
    "scenario",
    "response_instruction",
    "response_options",
    "scoring_key",
    "construct_rationale",
    "contamination_risks",
)
_BEHAVIORAL_LEVEL_RANK = {
    "low": 0,
    "medium_low": 1,
    "medium_high": 2,
    "high": 3,
}


def derive_scoring_key_from_behavioral_levels(
    item: dict[str, Any],
) -> dict[str, int]:
    """Derive the only valid score mapping from program-owned level labels."""

    options = item.get("response_options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"题目 {item.get('item_id')} 缺少有效选项")
    ranked: list[tuple[int, str]] = []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("题目包含无效选项")
        option_id = option.get("option_id")
        level = option.get("behavioral_level")
        if (
            not isinstance(option_id, str)
            or not option_id.strip()
            or level not in _BEHAVIORAL_LEVEL_RANK
        ):
            raise ValueError(
                f"题目 {item.get('item_id')} 无法根据 behavioral_level 计分"
            )
        ranked.append((_BEHAVIORAL_LEVEL_RANK[str(level)], option_id))
    if len({option_id for _, option_id in ranked}) != len(ranked):
        raise ValueError(f"题目 {item.get('item_id')} 的 option_id 必须唯一")
    if len({rank for rank, _ in ranked}) != len(ranked):
        raise ValueError(
            f"题目 {item.get('item_id')} 的 behavioral_level 必须形成唯一等级"
        )
    ranked.sort()
    return {
        option_id: score
        for score, (_, option_id) in enumerate(ranked, 1)
    }


def validate_item_review(
    review: Any,
    *,
    current_item: dict[str, Any] | None = None,
) -> None:
    """Validate the four-dimension diagnosis and program-derived tasks."""

    if not isinstance(review, dict):
        raise ValueError("统一审题结果必须是对象")
    if set(review) - {"findings", "repair_tasks", "summary"}:
        raise ValueError(
            "统一审题结果只能包含 findings、repair_tasks 和 summary"
        )
    findings = review.get("findings")
    if not isinstance(findings, list):
        raise ValueError("findings 必须是列表")
    valid_option_ids = {
        str(option.get("option_id"))
        for option in (current_item or {}).get("response_options") or []
        if isinstance(option, dict) and option.get("option_id")
    }
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"findings[{index}] 必须是对象")
        required_fields = {
            "criterion",
            "severity",
            "locus",
            "affected_option_ids",
            "evidence",
            "problem",
            "repair_instruction",
        }
        if not required_fields.issubset(finding) or set(finding) - (
            required_fields | {"required_edits"}
        ):
            raise ValueError(
                f"findings[{index}] 字段必须与审题 Finding Schema 完全一致"
            )
        if finding.get("criterion") not in {
            "trait_activation",
            "ecological_plausibility",
            "option_anti_faking",
            "construct_purity",
        }:
            raise ValueError(f"findings[{index}].criterion 无效")
        if finding.get("severity") not in {"warning", "blocking"}:
            raise ValueError(f"findings[{index}].severity 无效")
        locus = finding.get("locus")
        if locus not in {
            "scenario",
            "response_options",
            "scoring_key",
            "skeleton",
        }:
            raise ValueError(f"findings[{index}].locus 无效")
        option_ids = finding.get("affected_option_ids")
        if not isinstance(option_ids, list) or not all(
            isinstance(option_id, str) and option_id.strip()
            for option_id in option_ids
        ):
            raise ValueError(
                f"findings[{index}].affected_option_ids 必须是字符串列表"
            )
        if locus != "response_options" and option_ids:
            raise ValueError(
                f"findings[{index}] 的 {locus} 问题不得指定 affected_option_ids"
            )
        if (
            current_item is not None
            and locus == "response_options"
            and not set(option_ids).issubset(valid_option_ids)
        ):
            raise ValueError(f"findings[{index}] 指向不存在的选项")
        for field in ("evidence", "problem", "repair_instruction"):
            if not str(finding.get(field) or "").strip():
                raise ValueError(f"findings[{index}] 缺少 {field}")
        required_edits = finding.get("required_edits")
        if required_edits is not None:
            if not isinstance(required_edits, list):
                raise ValueError(f"findings[{index}].required_edits 必须是列表")
            for edit_index, edit in enumerate(required_edits):
                if not isinstance(edit, dict) or set(edit) != {
                    "field",
                    "option_ids",
                    "instruction",
                }:
                    raise ValueError(
                        f"findings[{index}].required_edits[{edit_index}] 字段无效"
                    )
                edit_field = edit.get("field")
                edit_option_ids = edit.get("option_ids")
                if edit_field not in {
                    "scenario",
                    "response_options",
                }:
                    raise ValueError(
                        f"findings[{index}].required_edits[{edit_index}].field 无效"
                    )
                if not isinstance(edit_option_ids, list) or not all(
                    isinstance(option_id, str) and option_id.strip()
                    for option_id in edit_option_ids
                ):
                    raise ValueError(
                        f"findings[{index}].required_edits[{edit_index}].option_ids 无效"
                    )
                if edit_field == "response_options" and not edit_option_ids:
                    raise ValueError("选项修改任务必须指定至少一个 option_id")
                if edit_field == "scenario" and edit_option_ids:
                    raise ValueError("情境修改任务不得包含 option_id")
                if current_item is not None and not set(
                    edit_option_ids
                ).issubset(valid_option_ids):
                    raise ValueError("required_edits 指向不存在的选项")
                if not str(edit.get("instruction") or "").strip():
                    raise ValueError("required_edits 缺少具体修改指令")

    if not str(review.get("summary") or "").strip():
        raise ValueError("统一审题结果缺少 summary")


def validate_item_review_diagnosis(
    diagnosis: Any,
    *,
    current_item: dict[str, Any] | None = None,
) -> None:
    """Validate the reviewer-only output before program enrichment."""

    if not isinstance(diagnosis, dict):
        raise ValueError("审题诊断必须是对象")
    if set(diagnosis) != {"findings", "summary"}:
        raise ValueError("审题诊断只能包含 findings 和 summary")
    validate_item_review(
        {
            "findings": diagnosis.get("findings"),
            "repair_tasks": build_repair_tasks_from_findings(
                diagnosis.get("findings"),
                current_item=current_item,
            )
            if isinstance(diagnosis.get("findings"), list)
            else None,
            "summary": diagnosis.get("summary"),
        },
        current_item=current_item,
    )
    for index, finding in enumerate(diagnosis.get("findings") or []):
        if finding.get("severity") != "blocking":
            continue
        required_edits = finding.get("required_edits")
        if finding.get("locus") == "skeleton":
            if required_edits:
                raise ValueError(
                    f"findings[{index}] 的骨架问题不得伪装成题面修改任务"
                )
            continue
        if not isinstance(required_edits, list) or not required_edits:
            raise ValueError(
                f"findings[{index}] 是 blocking，但缺少审题模型明确给出的 required_edits"
            )


def build_repair_tasks_from_findings(
    findings: list[dict[str, Any]],
    *,
    current_item: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert blocking diagnoses into deterministic, field-scoped tasks."""

    if not isinstance(findings, list):
        raise ValueError("findings 必须是列表")
    option_ids = [
        str(option.get("option_id"))
        for option in (current_item or {}).get("response_options") or []
        if isinstance(option, dict) and option.get("option_id")
    ]
    tasks: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"findings[{index}] 必须是对象")
        if finding.get("severity") != "blocking":
            continue
        required_edits = finding.get("required_edits")
        if isinstance(required_edits, list) and required_edits:
            targets = [
                {
                    "field": str(edit.get("field")),
                    "option_ids": list(edit.get("option_ids") or []),
                }
                for edit in required_edits
                if isinstance(edit, Mapping)
            ]
            instructions = [
                str(edit.get("instruction") or "").strip()
                for edit in required_edits
                if isinstance(edit, Mapping)
                and str(edit.get("instruction") or "").strip()
            ]
            source = (
                "construct"
                if finding.get("criterion")
                in {"trait_activation", "construct_purity"}
                else "content"
            )
            tasks.append(
                {
                    "task_id": f"review-{index + 1:02d}",
                    "source": source,
                    "targets": targets,
                    "problem": str(finding.get("problem") or ""),
                    "instruction": "；".join(instructions),
                }
            )
            continue
        locus = finding.get("locus")
        if locus == "skeleton":
            continue
        if locus == "scenario":
            targets = [{"field": "scenario", "option_ids": []}]
            instruction = (
                "在保持固定骨架与题目身份不变的前提下修复具体情境；"
                + str(finding.get("repair_instruction") or "")
            )
        elif locus == "response_options":
            targets = [
                {
                    "field": "response_options",
                    "option_ids": list(finding.get("affected_option_ids") or []),
                }
            ]
            instruction = str(finding.get("repair_instruction") or "")
        elif locus == "behavioral_level":
            # Compatibility for older checkpoints: keep the expert-defined
            # ordering and repair the option realization instead.
            targets = [
                {
                    "field": "response_options",
                    "option_ids": list(finding.get("affected_option_ids") or []),
                }
            ]
            instruction = (
                str(finding.get("repair_instruction") or "")
                + " Keep behavioral_level and scoring_key unchanged; rewrite "
                "the named option text to realize its fixed expert-defined level."
            )
        elif locus == "scoring_key":
            # Scores are program-owned. A semantic ordering complaint is an
            # option-realization problem and must not become an editable
            # scoring task.
            targets = [
                {
                    "field": "response_options",
                    "option_ids": option_ids,
                }
            ]
            instruction = (
                "scoring_key 由程序根据 behavioral_level 自动生成，不得直接改分；"
                "保持既定 behavioral_level，改写语义不匹配的选项文本，使四个选项"
                "分别忠实实现 low、medium_low、medium_high、high。"
            )
        else:
            continue
        source = (
            "construct"
            if finding.get("criterion")
            in {"trait_activation", "construct_purity"}
            else "content"
        )
        tasks.append(
            {
                "task_id": f"review-{index + 1:02d}",
                "source": source,
                "targets": targets,
                "problem": str(finding.get("problem") or ""),
                "instruction": instruction,
            }
        )
    return tasks


def derive_item_review_decision(
    review: dict[str, Any],
    *,
    repair_attempted: bool,
    repair_attempt_count: int | None = None,
    max_repair_attempts: int = 3,
    rewrite_count: int = 0,
    max_rewrite_rounds: int = 3,
) -> str:
    """Derive fixed-ID item control from review content and program counters."""

    validate_item_review(review)
    blocking_findings = [
        finding
        for finding in review["findings"]
        if finding.get("severity") == "blocking"
    ]
    if not blocking_findings:
        return "PASS"
    requires_rewrite = any(
        finding.get("locus") in {"scenario", "skeleton"}
        for finding in blocking_findings
    )
    if requires_rewrite:
        return "REJECT" if rewrite_count >= max_rewrite_rounds else "REWRITE"
    attempts = (
        int(repair_attempt_count)
        if repair_attempt_count is not None
        else int(bool(repair_attempted))
    )
    if attempts < max_repair_attempts:
        return "REVISE"
    return "REJECT" if rewrite_count >= max_rewrite_rounds else "REWRITE"


def derive_item_repair_action(review: dict[str, Any]) -> RouteType:
    """Choose full realization rewrite versus option-scoped revision."""

    validate_item_review(review)
    if any(
        finding.get("severity") == "blocking"
        and finding.get("locus") in {"scenario", "skeleton"}
        for finding in review["findings"]
    ):
        return "regenerate_item"
    return "revise_item"


def _item_content(item: dict[str, Any]) -> dict[str, Any]:
    """Return only fields whose change constitutes an actual item revision."""

    return {
        field: deepcopy(item.get(field))
        for field in _ITEM_CONTENT_FIELDS
    }


def canonicalize_item_agent_update(
    action: str,
    update: dict[str, Any],
    *,
    specification: dict[str, Any] | None = None,
    blueprint_cell: dict[str, Any] | None,
    item_specification: dict[str, Any] | None,
    previous_item: dict[str, Any] | None,
) -> dict[str, Any]:
    """Restore item fields that are owned by deterministic workflow state."""

    canonical = deepcopy(update)
    if action == "generate_item" and isinstance(
        canonical.get("strategies"), list
    ):
        if not isinstance(item_specification, dict):
            return canonical
        if set(canonical) != {"scenario", "strategies"}:
            raise ValueError(
                "generate_item 语言实现只能返回 scenario 和 strategies"
            )
        scenario = canonical.get("scenario")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError("generate_item 的 scenario 必须是非空文本")
        strategies_by_level: dict[str, str] = {}
        for strategy in canonical["strategies"]:
            if not isinstance(strategy, dict) or set(strategy) != {
                "behavioral_level",
                "text",
            }:
                raise ValueError(
                    "generate_item 每个策略只能包含 behavioral_level 和 text"
                )
            level = strategy.get("behavioral_level")
            strategy_text = strategy.get("text")
            if level not in _BEHAVIORAL_LEVEL_RANK:
                raise ValueError(f"generate_item 返回无效行为等级：{level!r}")
            if level in strategies_by_level:
                raise ValueError(f"generate_item 重复返回行为等级：{level!r}")
            if not isinstance(strategy_text, str) or not strategy_text.strip():
                raise ValueError(f"generate_item 的 {level} 缺少有效策略文本")
            strategies_by_level[str(level)] = strategy_text.strip()
        if set(strategies_by_level) != set(_BEHAVIORAL_LEVEL_RANK):
            raise ValueError(
                "generate_item 必须唯一覆盖 low、medium_low、medium_high、high"
            )
        specification_id = str(item_specification.get("specification_id") or "")
        ordered_levels = sorted(
            _BEHAVIORAL_LEVEL_RANK,
            key=lambda level: sha256(
                f"{specification_id}|{level}".encode("utf-8")
            ).digest(),
        )
        response_options = [
            {
                "option_id": chr(ord("A") + index),
                "text": strategies_by_level[level],
                "behavioral_level": level,
            }
            for index, level in enumerate(ordered_levels)
        ]
        situation_type = str(
            item_specification.get("situation_type")
            or item_specification.get("context_seed")
            or ""
        ).strip()
        social_context = str(
            item_specification.get("social_context") or ""
        ).strip()
        core_tension = str(
            item_specification.get("core_tension") or ""
        ).strip()
        context_signature = " | ".join(
            value for value in (situation_type, social_context, core_tension)
            if value
        )
        dimension_id = str(
            (blueprint_cell or {}).get("facet_id")
            or (blueprint_cell or {}).get("dimension_id")
            or item_specification.get("target_dimension_id")
            or ""
        )
        activation_mechanism = str(
            item_specification.get("activation_mechanism") or ""
        ).strip()
        item = {
            "item_id": specification_id,
            "blueprint_cell_id": item_specification.get("blueprint_cell_id"),
            "target_dimension_id": dimension_id,
            "context_category": item_specification.get("context_category"),
            "context_signature": context_signature,
            "scenario": scenario.strip(),
            "response_instruction": PSJT_RESPONSE_INSTRUCTION,
            "response_options": response_options,
            "scoring_key": {},
            "construct_rationale": (
                f"情境通过“{activation_mechanism}”激活 {dimension_id}；"
                "四个选项按固定的单一构念行为等级实现。"
            ),
            "contamination_risks": [],
            "version": 1,
        }
        canonical = {"current_item": item}
    if action in {"revise_item", "regenerate_item"} and set(canonical) == {
        "scenario_update",
        "option_updates",
    }:
        if not isinstance(previous_item, dict):
            return canonical
        item = deepcopy(previous_item)
        options = item.get("response_options")
        if not isinstance(options, list):
            return canonical
        option_by_id = {
            option.get("option_id"): option
            for option in options
            if isinstance(option, dict)
            and isinstance(option.get("option_id"), str)
        }
        scenario_update = canonical.get("scenario_update")
        if scenario_update is not None:
            if not isinstance(scenario_update, str) or not scenario_update.strip():
                raise ValueError(f"{action} 的 scenario_update 必须为非空文本或 null")
            item["scenario"] = scenario_update.strip()
        patches = canonical.get("option_updates")
        if not isinstance(patches, list):
            raise ValueError(f"{action} 的 option_updates 必须是列表")
        if scenario_update is None and not patches:
            raise ValueError(f"{action} 返回了空修复补丁")
        seen: set[str] = set()
        for patch in patches:
            if not isinstance(patch, dict):
                raise ValueError(f"{action} 的 option_updates 必须是对象列表")
            unexpected = set(patch) - {"option_id", "text"}
            if unexpected:
                raise ValueError(
                    f"{action} 的选项补丁包含多余字段："
                    + ", ".join(sorted(unexpected))
                )
            option_id = patch.get("option_id")
            text = patch.get("text")
            if option_id not in option_by_id:
                raise ValueError(
                    f"{action} 返回了未知选项：{option_id!r}"
                )
            if option_id in seen:
                raise ValueError(
                    f"{action} 重复返回选项：{option_id!r}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{action} 的 {option_id} 缺少有效的新文本"
                )
            seen.add(option_id)
            option_by_id[option_id]["text"] = text.strip()
        canonical = {"current_item": item}
    item = canonical.get("current_item")
    if not isinstance(item, dict):
        return canonical
    item["response_instruction"] = PSJT_RESPONSE_INSTRUCTION
    if blueprint_cell:
        item["blueprint_cell_id"] = blueprint_cell.get("cell_id")
        item["target_dimension_id"] = (
            blueprint_cell.get("facet_id")
            or blueprint_cell.get("dimension_id")
        )
    if item_specification:
        item["context_category"] = item_specification.get(
            "context_category"
        )
        if action == "generate_item":
            specification_id = item_specification.get("specification_id")
            if isinstance(specification_id, str) and specification_id.strip():
                item["item_id"] = specification_id
    if action in {"revise_item", "regenerate_item"} and previous_item:
        item["item_id"] = previous_item.get("item_id")
        item["version"] = previous_item.get("version", 0) + 1
    if isinstance(item.get("response_options"), list):
        item["scoring_key"] = derive_scoring_key_from_behavioral_levels(item)
    return canonical


def validate_item_agent_update(
    action: str,
    update: dict[str, Any],
    *,
    target_item_id: str | None = None,
    target_blueprint_cell_id: str | None = None,
    specification: dict[str, Any] | None = None,
    blueprint_cell: dict[str, Any] | None = None,
    item_specification: dict[str, Any] | None = None,
    previous_item: dict[str, Any] | None = None,
) -> None:
    """校验题目、评分键、稳定身份和结构化审题结果。"""

    allowed_fields = ITEM_AGENT_OUTPUT_FIELDS.get(action)
    if allowed_fields is None:
        return

    unexpected_fields = set(update) - allowed_fields
    if unexpected_fields:
        raise ValueError(
            f"{action} 不允许修改字段："
            + ", ".join(sorted(unexpected_fields))
        )

    if action in {"generate_item", "regenerate_item", "revise_item"}:
        item = update.get("current_item")
        if not isinstance(item, dict):
            raise ValueError(f"{action} 必须返回对象类型的 current_item")
        item_id = item.get("item_id")
        cell_id = item.get("blueprint_cell_id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{action} 返回的 current_item 缺少有效 item_id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise ValueError(
                f"{action} 返回的 current_item 缺少有效 blueprint_cell_id"
            )
        if (
            action in {"revise_item", "regenerate_item"}
            and target_item_id
            and item_id != target_item_id
        ):
            raise ValueError(f"{action} 不允许修改 item_id")
        if target_blueprint_cell_id and cell_id != target_blueprint_cell_id:
            raise ValueError(
                f"{action} 返回的 blueprint_cell_id 与目标单元不一致"
            )
        dimension_id = item.get("target_dimension_id")
        if not isinstance(dimension_id, str) or not dimension_id.strip():
            raise ValueError(f"{action} 缺少有效 target_dimension_id")
        expected_dimension_id = (
            blueprint_cell.get("facet_id")
            if blueprint_cell
            else None
        )
        if expected_dimension_id is None and blueprint_cell:
            expected_dimension_id = blueprint_cell.get("dimension_id")
        if blueprint_cell and dimension_id != expected_dimension_id:
            raise ValueError(f"{action} 的目标维度与蓝图单元不一致")
        if item_specification:
            expected_category = item_specification.get("context_category")
            if item.get("context_category") != expected_category:
                raise ValueError(
                    f"{action} 的情境类别必须与题目生成槽一致："
                    f"期望={expected_category!r}，"
                    f"实际={item.get('context_category')!r}"
                )
            if not isinstance(item.get("context_signature"), str) or not item[
                "context_signature"
            ].strip():
                raise ValueError(f"{action} 的 context_signature 必须是非空文本")
            # Content-diversity constraints are program owned.  They do not
            # authorize the model to create or replace any slot identity.
            if (
                item_specification.get("avoid_scenario_patterns")
                or item_specification.get("avoid_response_patterns")
            ):
                scenario_matches = [
                    str(pattern)
                    for pattern in item_specification.get(
                        "avoid_scenario_patterns"
                    )
                    or []
                    if isinstance(pattern, str)
                    and pattern.strip()
                    and text_similarity(item.get("scenario"), pattern) >= 0.90
                ]
                option_matches = [
                    str(pattern)
                    for option in item.get("response_options") or []
                    if isinstance(option, Mapping)
                    for pattern in item_specification.get(
                        "avoid_response_patterns"
                    )
                    or []
                    if isinstance(pattern, str)
                    and pattern.strip()
                    and text_similarity(option.get("text"), pattern) >= 0.90
                ]
                if scenario_matches or len(option_matches) >= 2:
                    raise ValueError(
                        "replacement 复用了已淘汰题目的内容："
                        f"情境命中={len(scenario_matches)}，"
                        f"选项命中={len(option_matches)}；"
                        "必须实质更换情境实现和反应策略表述"
                    )

        for field in (
            "scenario",
            "response_instruction",
            "construct_rationale",
        ):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"{action} 的 {field} 必须是非空文本")
        if item.get("response_instruction") != PSJT_RESPONSE_INSTRUCTION:
            raise ValueError(
                f"{action} 的指导语必须固定为 {PSJT_RESPONSE_INSTRUCTION!r}"
            )
        risks = item.get("contamination_risks")
        if not isinstance(risks, list) or not all(
            isinstance(risk, str) for risk in risks
        ):
            raise ValueError(f"{action} 的 contamination_risks 必须是字符串列表")

        options = item.get("response_options")
        if not isinstance(options, list) or len(options) != PSJT_OPTION_COUNT:
            raise ValueError(f"{action} 的选项数量与测验规格不一致")
        option_ids: list[str] = []
        for option in options:
            if not isinstance(option, dict):
                raise ValueError(f"{action} 的每个选项必须是对象")
            option_id = option.get("option_id")
            if not isinstance(option_id, str) or not option_id.strip():
                raise ValueError(f"{action} 的选项缺少有效 option_id")
            if not isinstance(option.get("text"), str) or not option["text"].strip():
                raise ValueError(f"{action} 的选项文本不能为空")
            if option.get("behavioral_level") not in {
                "low",
                "medium_low",
                "medium_high",
                "high",
            }:
                raise ValueError(f"{action} 的 behavioral_level 无效")
            option_ids.append(option_id)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError(f"{action} 的 option_id 必须唯一")

        scoring_key = item.get("scoring_key")
        if not isinstance(scoring_key, dict) or set(scoring_key) != set(option_ids):
            raise ValueError(f"{action} 的评分键必须精确覆盖全部选项")
        scores = list(scoring_key.values())
        if any(
            isinstance(score, bool) or not isinstance(score, (int, float))
            for score in scores
        ):
            raise ValueError(f"{action} 的评分必须是数值")
        if sorted(scores) != list(range(1, len(options) + 1)):
            raise ValueError(
                f"{action} 的行为水平评分必须唯一覆盖 1..option_count"
            )
        expected_scoring_key = derive_scoring_key_from_behavioral_levels(item)
        if scoring_key != expected_scoring_key:
            raise ValueError(
                f"{action} 的 scoring_key 必须由 behavioral_level 自动派生"
            )

        version = item.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(f"{action} 的 version 必须是正整数")
        if action in {"revise_item", "regenerate_item"} and previous_item:
            if dimension_id != previous_item.get("target_dimension_id"):
                raise ValueError(f"{action} 不允许修改目标维度")
            if cell_id != previous_item.get("blueprint_cell_id"):
                raise ValueError(f"{action} 不允许修改蓝图单元")
            if version != previous_item.get("version", 0) + 1:
                raise ValueError(f"{action} 的 version 必须递增 1")
            previous_levels = {
                str(option.get("option_id")): option.get("behavioral_level")
                for option in previous_item.get("response_options") or []
                if isinstance(option, Mapping)
            }
            current_levels = {
                str(option.get("option_id")): option.get("behavioral_level")
                for option in item.get("response_options") or []
                if isinstance(option, Mapping)
            }
            if current_levels != previous_levels:
                raise ValueError(
                    f"{action} must preserve expert-defined behavioral_level; "
                    "rewrite option text when a level is not realized"
                )
            if item.get("scoring_key") != previous_item.get("scoring_key"):
                raise ValueError(
                    f"{action} must preserve the expert-defined scoring_key"
                )
            if _item_content(item) == _item_content(previous_item):
                raise ValueError(
                    f"{action} 未修改任何题目内容；"
                    "不能只递增 version，必须回应审题或心理测量诊断"
                )
        return

    unified_review = update.get("current_item_review")
    if unified_review is None:
        raise ValueError("review_item 必须返回 current_item_review")
    validate_item_review(unified_review)
    if update.get("review_process_status") != "valid":
        raise ValueError("review_item must mark review_process_status as valid")
    if update.get("item_content_status") not in {"pass", "needs_repair"}:
        raise ValueError("review_item returned invalid item_content_status")
    if target_item_id and update.get("current_review_item_id") != target_item_id:
        raise ValueError("review_item item_id does not match the routed item")
    if previous_item is not None:
        expected_version = previous_item.get("version")
        if update.get("current_review_item_version") != expected_version:
            raise ValueError("review_item version does not match current_item")


def get_current_item_identity(state: PSJTState) -> tuple[str, str]:
    """取得当前题目和蓝图单元标识；缺失时明确失败。"""

    item = state.get("current_item")
    if not isinstance(item, dict):
        raise ValueError("当前没有可处理的 current_item")

    item_id = item.get("item_id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("current_item 缺少有效的 item_id")

    route = state.get("route") or {}
    blueprint_cell = state.get("current_blueprint_cell") or {}
    cell_id = (
        item.get("blueprint_cell_id")
        or route.get("target_blueprint_cell_id")
        or blueprint_cell.get("cell_id")
    )
    if not isinstance(cell_id, str) or not cell_id.strip():
        raise ValueError("current_item 缺少有效的 blueprint_cell_id")

    return item_id, cell_id


def _append_history(
    state: PSJTState,
    item_id: str,
    entry: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    history = {
        key: [deepcopy(record) for record in records]
        for key, records in state["item_history"].items()
    }
    history.setdefault(item_id, []).append(deepcopy(entry))
    return history


def _increment_progress(
    state: PSJTState,
    cell_id: str,
    field: str,
) -> dict[str, dict[str, int]]:
    progress = {
        key: dict(value)
        for key, value in state["blueprint_progress"].items()
    }
    cell_progress = progress.setdefault(
        cell_id,
        {"generated": 0, "passed": 0, "rejected": 0, "missing": 0},
    )
    cell_progress[field] = cell_progress.get(field, 0) + 1
    if field == "generated":
        cell_progress["missing"] = max(
            0,
            cell_progress.get("missing", 0) - 1,
        )
    return progress


def record_committed_item_output(
    state: PSJTState,
    action: str,
) -> dict[str, Any]:
    """记录已由用户确认的生成、修改或重新生成结果。"""

    if action not in {"generate_item", "revise_item", "regenerate_item"}:
        raise ValueError(f"不支持记录题目动作 {action!r}")

    item_id, cell_id = get_current_item_identity(state)
    item = deepcopy(state["current_item"])
    item["blueprint_cell_id"] = cell_id

    progress = state["blueprint_progress"]

    if action == "generate_item":
        event = "generated"
        progress = _increment_progress(state, cell_id, "generated")
        repair_attempted = False
    elif action == "revise_item":
        event = "revised"
        repair_attempted = True
    else:
        event = "regenerated"
        repair_attempted = bool(
            state.get("current_item_repair_attempted")
        )
        # 重写仍属于原题目生成槽，不能再次消耗蓝图计划生成量。

    history = _append_history(
        state,
        item_id,
        {"event": event, "item": item},
    )

    return {
        "current_item": item,
        "current_item_review": None,
        "review_process_status": "not_started",
        "item_content_status": "not_evaluated",
        "current_review_request_id": None,
        "current_review_item_id": None,
        "current_review_item_version": None,
        "current_review_retry_count": 0,
        "current_item_repair_attempted": repair_attempted,
        "current_item_repair_failure": None,
        "current_skeleton_repair_required": False,
        "current_item_revision_count": (
            0
            if action == "generate_item"
            else int(state.get("current_item_revision_count") or 0)
        ),
        "current_item_rewrite_count": (
            0
            if action == "generate_item"
            else int(state.get("current_item_rewrite_count") or 0)
        ),
        "current_item_replacement_count": (
            0
            if action == "generate_item"
            else int(state.get("current_item_replacement_count") or 0)
        ),
        "item_history": history,
        "blueprint_progress": progress,
    }


def next_step_after_review(state: PSJTState) -> str:
    """根据审题结论和次数上限选择确定性的下一步。"""

    unified_review = state.get("current_item_review")
    if not isinstance(unified_review, dict):
        raise ValueError("当前题目缺少统一审题结果")

    # An invalid psychometric patch is an output/repair failure, not a
    # skeleton-review failure.  The fallback review created by execute_node
    # uses locus="skeleton" for compatibility, which otherwise routes through
    # the rewrite budget and can bypass the directed-revision limit forever.
    # Respect the item-level revision budget before deciding whether to retry.
    if (
        state.get("current_item_repair_failure")
        and isinstance(state.get("active_psychometric_repair"), Mapping)
    ):
        repair_attempts = int(state.get("current_item_revision_count") or 0)
        max_repair_attempts = int(
            state.get("max_item_revision_attempts") or 3
        )
        if repair_attempts >= max_repair_attempts:
            return "abandon"
        return "revise"

    decision = derive_item_review_decision(
        unified_review,
        repair_attempted=bool(state.get("current_item_repair_attempted")),
        repair_attempt_count=int(
            state.get("current_item_revision_count") or 0
        ),
        max_repair_attempts=int(
            state.get("max_item_revision_attempts") or 3
        ),
        rewrite_count=int(state.get("current_item_rewrite_count") or 0),
        max_rewrite_rounds=int(state.get("max_item_rewrite_rounds") or 3),
    )
    if decision == "REJECT":
        if isinstance(state.get("active_psychometric_repair"), Mapping):
            return "abandon"
        if int(state.get("current_item_replacement_count") or 0) < int(
            state.get("max_item_replacement_attempts") or 2
        ):
            return "revise"
        return "abandon"
    return {
        "PASS": "accept",
        "REVISE": "revise",
        "REWRITE": "revise",
    }[decision]


def build_item_agent_route(
    state: PSJTState,
    action: RouteType,
    reason: str,
) -> dict[str, Any]:
    """为单题内部动作建立确定性 Route，不调用 LLM Router。"""

    item_id, cell_id = get_current_item_identity(state)
    return {
        "route": {
            "next_action": action,
            "reason": reason,
            "target_item_id": item_id,
            "target_blueprint_cell_id": cell_id,
        },
    }


def _review_history(state: PSJTState) -> dict[str, Any]:
    review = state.get("current_item_review")
    if not isinstance(review, dict):
        raise ValueError("当前题目缺少统一审题结果")
    review = deepcopy(review)
    return {
        "event": "reviewed",
        "decision": derive_item_review_decision(
            review,
            repair_attempted=bool(state.get("current_item_repair_attempted")),
            repair_attempt_count=int(
                state.get("current_item_revision_count") or 0
            ),
            max_repair_attempts=int(
                state.get("max_item_revision_attempts") or 3
            ),
            rewrite_count=int(
                state.get("current_item_rewrite_count") or 0
            ),
            max_rewrite_rounds=int(
                state.get("max_item_rewrite_rounds") or 3
            ),
        ),
        "review": review,
        "repair_tasks": deepcopy(review.get("repair_tasks") or []),
    }


def build_review_transition_update(
    state: PSJTState,
    action: RouteType,
    reason: str,
) -> dict[str, Any]:
    """记录本轮审题并准备修改或重新生成。"""

    item_id, _ = get_current_item_identity(state)
    update = build_item_agent_route(state, action, reason)
    update["current_item_repair_attempted"] = True
    if action == "regenerate_item":
        update["current_item_revision_count"] = 0
        update["current_item_rewrite_count"] = (
            int(state.get("current_item_rewrite_count") or 0) + 1
        )
    else:
        update["current_item_revision_count"] = (
            int(state.get("current_item_revision_count") or 0) + 1
        )
        update["current_item_rewrite_count"] = int(
            state.get("current_item_rewrite_count") or 0
        )
    update["item_history"] = _append_history(
        state,
        item_id,
        _review_history(state),
    )
    return update


def build_accept_item_update(state: PSJTState) -> dict[str, Any]:
    """把通过审查的当前题目移入候选题库。"""

    unified_review = state.get("current_item_review")
    if not isinstance(unified_review, dict) or derive_item_review_decision(
        unified_review,
        repair_attempted=bool(state.get("current_item_repair_attempted")),
        repair_attempt_count=int(
            state.get("current_item_revision_count") or 0
        ),
        max_repair_attempts=int(
            state.get("max_item_revision_attempts") or 3
        ),
        rewrite_count=int(state.get("current_item_rewrite_count") or 0),
        max_rewrite_rounds=int(state.get("max_item_rewrite_rounds") or 3),
    ) != "PASS":
        raise ValueError("只有 PASS 题目可以进入 item_pool")

    item_id, cell_id = get_current_item_identity(state)
    active_repair = state.get("active_psychometric_repair")
    is_psychometric_repair = (
        isinstance(active_repair, dict)
        and active_repair.get("item_id") == item_id
    )
    if is_psychometric_repair and isinstance(
        active_repair.get("baseline_item"), Mapping
    ):
        baseline_item = active_repair["baseline_item"]
        baseline_version = int(baseline_item.get("version") or 0)
        current_version = int(state["current_item"].get("version") or 0)
        if current_version <= baseline_version:
            raise ValueError(
                "心理测量返修未提交新题目版本，不能记录为修复成功"
            )
        if _item_content(state["current_item"]) == _item_content(
            baseline_item
        ):
            raise ValueError(
                "心理测量返修未改变指定题目内容，不能记录为修复成功"
            )
    existing_index = next(
        (
            index
            for index, item in enumerate(state["item_pool"])
            if item.get("item_id") == item_id
        ),
        None,
    )
    if existing_index is not None and not is_psychometric_repair:
        raise ValueError(f"item_pool 已存在题目 {item_id!r}")

    profile = build_item_pattern_profile(
        state["current_item"],
        state.get("current_item_specification"),
    )
    profiles = dict(state.get("item_pattern_profiles", {}))
    profiles[item_id] = profile
    context_usage = dict(state.get("context_usage", {}))
    category = profile["context_category"]
    if not is_psychometric_repair:
        context_usage[category] = context_usage.get(category, 0) + 1
    item_pool = [deepcopy(item) for item in state["item_pool"]]
    if is_psychometric_repair and existing_index is not None:
        item_pool[existing_index] = deepcopy(state["current_item"])
    else:
        item_pool.append(deepcopy(state["current_item"]))

    update = {
        "current_item": None,
        "current_item_specification": None,
        "current_blueprint_cell": None,
        "current_item_review": None,
        "review_process_status": "not_started",
        "item_content_status": "not_evaluated",
        "current_review_request_id": None,
        "current_review_item_id": None,
        "current_review_item_version": None,
        "current_review_retry_count": 0,
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "current_item_revision_count": 0,
        "current_item_rewrite_count": 0,
        "current_item_replacement_count": 0,
        "current_skeleton_repair_required": False,
        "item_pool": item_pool,
        "item_pattern_profiles": profiles,
        "context_usage": context_usage,
        "item_history": _append_history(
            state,
            item_id,
            _review_history(state),
        ),
        "blueprint_progress": (
            {
                key: dict(value)
                for key, value in state["blueprint_progress"].items()
            }
            if is_psychometric_repair
            else _increment_progress(
                state,
                cell_id,
                "passed",
            )
        ),
    }
    if not is_psychometric_repair:
        return update

    remaining_revise = [
        deepcopy(entry)
        for entry in state.get("items_to_revise") or []
        if isinstance(entry, dict) and entry.get("item_id") != item_id
    ]
    remaining_regenerate = [
        deepcopy(entry)
        for entry in state.get("items_to_regenerate") or []
        if isinstance(entry, dict) and entry.get("item_id") != item_id
    ]
    batch_has_remaining = bool(remaining_revise or remaining_regenerate)
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    rounds[item_id] = int(active_repair.get("revision_round") or 1)
    repair_history = deepcopy(
        state.get("psychometric_repair_history") or []
    )
    repair_history.append(
        {
            "event": "psychometric_item_repaired",
            "recorded_at": utc_timestamp(),
            "item_id": item_id,
            "revision_round": rounds[item_id],
            "action": active_repair.get("action"),
            "baseline_metrics": deepcopy(
                active_repair.get("baseline_metrics") or {}
            ),
            "baseline_item": deepcopy(active_repair.get("baseline_item")),
            "baseline_profile": deepcopy(
                active_repair.get("baseline_profile")
            ),
            "baseline_analysis_snapshot": deepcopy(
                active_repair.get("baseline_analysis_snapshot")
            ),
            "new_item_version": state["current_item"].get("version"),
            "diagnosis_fingerprint": active_repair.get(
                "diagnosis_fingerprint"
            ),
            "atomic_repair_advice": deepcopy(
                active_repair.get("atomic_repair_advice")
            ),
            "local_retest": deepcopy(active_repair.get("local_retest")),
        }
    )
    update.update(
        {
            "active_psychometric_repair": None,
            "psychometric_repair_confirmation": None,
            "items_to_revise": remaining_revise,
            "items_to_regenerate": remaining_regenerate,
            "items_deferred_for_revision": [],
            "psychometric_repair_rounds": rounds,
            "psychometric_repair_history": repair_history,
            "selected_items": [],
            "reserve_items": [],
            "selection_reasons": {},
            "selection_results": None,
            "item_final_dispositions": {
                key: deepcopy(value)
                for key, value in (
                    state.get("item_final_dispositions") or {}
                ).items()
                if key != item_id
            },
            "blueprint_coverage": None,
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
        }
    )
    if not batch_has_remaining:
        previous_response_ref = state.get("virtual_response_data_ref")
        update.update(
            {
                "virtual_response_data_ref": None,
                "virtual_response_summary": None,
                "virtual_response_item_bank_id": None,
                "virtual_response_item_bank_version": None,
                "item_statistics": {},
                "psychometric_round_result": None,
                "test_statistics": None,
                "factor_results": None,
                "irt_results": None,
                "dif_results": None,
            }
        )
        if isinstance(previous_response_ref, str) and previous_response_ref:
            update["previous_virtual_response_data_ref"] = previous_response_ref
    return update


def build_abandon_item_update(state: PSJTState) -> dict[str, Any]:
    """达到重新生成上限后放弃当前题目。"""

    item_id, cell_id = get_current_item_identity(state)
    active_repair = state.get("active_psychometric_repair")
    is_psychometric_repair = (
        isinstance(active_repair, dict)
        and active_repair.get("item_id") == item_id
    )
    if is_psychometric_repair:
        remaining_revise = [
            deepcopy(entry)
            for entry in state.get("items_to_revise") or []
            if isinstance(entry, dict) and entry.get("item_id") != item_id
        ]
        remaining_regenerate = [
            deepcopy(entry)
            for entry in state.get("items_to_regenerate") or []
            if isinstance(entry, dict) and entry.get("item_id") != item_id
        ]
        rounds = dict(state.get("psychometric_repair_rounds") or {})
        repair_history = deepcopy(
            state.get("psychometric_repair_history") or []
        )
        repair_history.append(
            {
                "event": "psychometric_item_repair_candidate_rejected",
                "recorded_at": utc_timestamp(),
                "item_id": item_id,
                "revision_round": int(active_repair.get("revision_round") or 1),
                "action": active_repair.get("action"),
                "baseline_metrics": deepcopy(
                    active_repair.get("baseline_metrics") or {}
                ),
                "rejected_candidate": deepcopy(state.get("current_item")),
                "reason": (
                    state.get("current_item_repair_failure")
                    or (state.get("current_item_review") or {}).get("summary")
                    or "返修候选未通过内容审查"
                ),
            }
        )
        # A failed repair candidate was never committed to item_pool. Keep the
        # last accepted version and its responses/statistics intact, then let
        # selection decide whether another bounded repair round is warranted.
        return {
            "current_item": None,
            "current_item_specification": None,
            "current_blueprint_cell": None,
            "current_item_review": None,
            "review_process_status": "not_started",
            "item_content_status": "not_evaluated",
            "current_review_request_id": None,
            "current_review_item_id": None,
            "current_review_item_version": None,
            "current_review_retry_count": 0,
            "current_item_repair_attempted": False,
            "current_item_repair_failure": None,
            "current_item_revision_count": 0,
            "current_item_rewrite_count": 0,
            "current_item_replacement_count": 0,
            "current_skeleton_repair_required": False,
            "active_psychometric_repair": None,
            "psychometric_repair_confirmation": None,
            "items_to_revise": remaining_revise,
            "items_to_regenerate": remaining_regenerate,
            "items_deferred_for_revision": [],
            "psychometric_repair_rounds": rounds,
            "psychometric_repair_history": repair_history,
            "item_history": _append_history(
                state,
                item_id,
                _review_history(state),
            ),
            "selected_items": [],
            "reserve_items": [],
            "selection_reasons": {},
            "selection_results": None,
            "blueprint_coverage": None,
            "assembled_test": None,
            "test_review_result": None,
            "final_test": None,
            "item_database_ref": None,
            "technical_report": None,
            "virtual_respondent_report": None,
        }
    rejected_items = [
        deepcopy(item) for item in state["rejected_items"]
    ]
    duplicate_index = next(
        (
            index
            for index, item in enumerate(rejected_items)
            if item.get("item_id") == item_id
        ),
        None,
    )
    if duplicate_index is None:
        rejected_items.append(deepcopy(state["current_item"]))
    else:
        # Older checkpoints could contain a model-supplied duplicate item_id.
        # Keep the latest rejected candidate under that identity; full version
        # history remains available in item_history.
        rejected_items[duplicate_index] = deepcopy(state["current_item"])

    progress = _increment_progress(
        state,
        cell_id,
        "rejected",
    )
    update = {
        "current_item": None,
        "current_item_specification": None,
        "current_blueprint_cell": None,
        "current_item_review": None,
        "review_process_status": "not_started",
        "item_content_status": "not_evaluated",
        "current_review_request_id": None,
        "current_review_item_id": None,
        "current_review_item_version": None,
        "current_review_retry_count": 0,
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "current_item_revision_count": 0,
        "current_item_rewrite_count": 0,
        "current_item_replacement_count": 0,
        "current_skeleton_repair_required": False,
        "rejected_items": rejected_items,
        "item_history": _append_history(
            state,
            item_id,
            _review_history(state),
        ),
        "blueprint_progress": progress,
    }
    return update
