"""Candidate-bank audit contracts, freezing, and version identity."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

from sjt_system.runtime.trace import utc_timestamp


_DOWNSTREAM_INVALIDATION = {
    "candidate_bank_audit": None,
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
    "selected_items": [],
    "reserve_items": [],
    "items_to_revise": [],
    "items_to_regenerate": [],
    "items_deferred_for_revision": [],
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

_FREEZE_FIELDS = {
    "item_bank_id",
    "item_bank_version",
    "item_bank_fingerprint",
    "item_bank_frozen_at",
    "frozen_item_bank",
}

_VIRTUAL_RESPONSE_FIELDS = {
    "virtual_sample_config",
    "virtual_respondents",
    "virtual_response_data_ref",
    "virtual_response_summary",
    "virtual_response_item_bank_id",
    "virtual_response_item_bank_version",
}


def _fingerprint(items: list[dict[str, Any]]) -> str:
    serialized = json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def build_item_bank_freeze_update(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an immutable, versioned snapshot of the approved item pool."""

    raw_items = state.get("item_pool")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("冻结题库前必须至少有一道已通过审查的题目")
    if not all(isinstance(item, Mapping) for item in raw_items):
        raise ValueError("冻结题库时发现无效题目记录")

    items = deepcopy([dict(item) for item in raw_items])
    for item in items:
        if not isinstance(item.get("item_id"), str) or not item["item_id"].strip():
            raise ValueError("冻结题库时发现缺少 item_id 的题目")
        version = item.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(
                f"冻结题库时发现无效题目版本：{item['item_id']!r}"
            )

    fingerprint = _fingerprint(items)
    previous_fingerprint = state.get("item_bank_fingerprint")
    if (
        fingerprint == previous_fingerprint
        and state.get("item_bank_id")
        and state.get("frozen_item_bank")
    ):
        return {
            "item_bank_id": state["item_bank_id"],
            "item_bank_version": state["item_bank_version"],
            "item_bank_fingerprint": previous_fingerprint,
            "item_bank_frozen_at": state["item_bank_frozen_at"],
            "frozen_item_bank": deepcopy(state["frozen_item_bank"]),
        }

    previous_version = state.get("item_bank_version")
    version = (
        previous_version + 1
        if isinstance(previous_version, int)
        and not isinstance(previous_version, bool)
        and previous_version >= 1
        else 1
    )
    run_fragment = str(state.get("run_id") or "unknown")[:12]
    update = {
        "item_bank_id": (
            f"bank-{run_fragment}-v{version}-{fingerprint[:12]}"
        ),
        "item_bank_version": version,
        "item_bank_fingerprint": fingerprint,
        "item_bank_frozen_at": utc_timestamp(),
        "frozen_item_bank": items,
    }
    if previous_fingerprint and previous_fingerprint != fingerprint:
        previous_response_ref = state.get("virtual_response_data_ref")
        update.update(deepcopy(_DOWNSTREAM_INVALIDATION))
        if isinstance(previous_response_ref, str) and previous_response_ref:
            update["previous_virtual_response_data_ref"] = (
                previous_response_ref
            )
    return update


def audit_candidate_item_bank(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """统一检查局部返修候选汇总后的硬约束。

    语义构念纯度仍以逐题审题为准；这里检查审题留下的阻断风险、固定
    蓝图映射，以及可由程序确定的重复情境/重复题目。候选在通过本审计
    前不会进入下一次正式虚拟施测。
    """

    raw_items = state.get("item_pool")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("候选题库审计前必须存在 item_pool")
    blueprint = state.get("blueprint")
    blueprint_mapping = blueprint if isinstance(blueprint, Mapping) else {}
    cells = {
        str(cell.get("cell_id")): cell
        for cell in blueprint_mapping.get("cells") or []
        if isinstance(cell, Mapping) and cell.get("cell_id")
    }
    seen_ids: set[str] = set()
    seen_contexts: dict[str, str] = {}
    seen_contents: dict[str, str] = {}
    warnings: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            errors.append("item_pool 包含无效题目记录")
            continue
        item = dict(raw_item)
        item_id = str(item.get("item_id") or "")
        if not item_id:
            errors.append("候选题缺少 item_id")
            continue
        if item_id in seen_ids:
            errors.append(f"候选题库存在重复 item_id：{item_id}")
        seen_ids.add(item_id)
        cell_id = str(item.get("blueprint_cell_id") or "")
        cell = cells.get(cell_id)
        if cell is None:
            errors.append(f"题目 {item_id} 未映射到有效蓝图槽位：{cell_id}")
        else:
            expected_dimension = str(cell.get("facet_id") or cell.get("dimension_id") or "")
            if expected_dimension and str(item.get("target_dimension_id") or "") != expected_dimension:
                errors.append(
                    f"题目 {item_id} 的 target_dimension_id 与蓝图槽位不一致"
                )
        scenario = " ".join(str(item.get("scenario") or "").split())
        context_signature = " ".join(
            str(item.get("context_signature") or scenario).split()
        )
        if context_signature:
            prior_item_id = seen_contexts.get(context_signature)
            if prior_item_id and prior_item_id != item_id:
                errors.append(
                    f"题目 {item_id} 与 {prior_item_id} 重复情境签名"
                )
            seen_contexts[context_signature] = item_id
        content_key = json.dumps(
            {
                "scenario": scenario,
                "response_options": [
                    {
                        "option_id": option.get("option_id"),
                        "text": " ".join(str(option.get("text") or "").split()),
                    }
                    for option in item.get("response_options") or []
                    if isinstance(option, Mapping)
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prior_item_id = seen_contents.get(content_key)
        if prior_item_id and prior_item_id != item_id:
            errors.append(f"题目 {item_id} 与 {prior_item_id} 内容完全重复")
        seen_contents[content_key] = item_id
        risks = item.get("contamination_risks") or []
        blocking_risks = [
            risk
            for risk in risks
            if isinstance(risk, Mapping)
            and str(risk.get("severity") or "").lower() == "blocking"
        ]
        if blocking_risks:
            errors.append(f"题目 {item_id} 仍保留审题阻断构念污染风险")
        elif risks:
            warnings.append(
                {
                    "item_id": item_id,
                    "message": "题目保留非阻断污染风险，已由逐题审题放行；不作为程序过滤条件。",
                }
            )

    if errors:
        raise ValueError("候选题库统一审计失败：" + "；".join(errors))
    return {
        "status": "passed",
        "item_count": len(raw_items),
        "duplicate_check": {
            "status": "passed",
            "context_signature_count": len(seen_contexts),
            "content_signature_count": len(seen_contents),
        },
        "blueprint_check": {
            "status": "passed",
            "mapped_cell_count": len({
                str(item.get("blueprint_cell_id"))
                for item in raw_items
                if isinstance(item, Mapping)
            }),
        },
        "construct_purity_check": {
            "status": "passed_by_item_review",
            "authority": "逐题审题；程序仅阻断明确标记的 blocking 污染风险",
        },
        "warnings": warnings,
    }


def item_bank_snapshot_is_current(state: Mapping[str, Any]) -> bool:
    """Return whether the frozen snapshot exactly matches the live item pool."""

    raw_items = state.get("item_pool")
    frozen_items = state.get("frozen_item_bank")
    fingerprint = state.get("item_bank_fingerprint")
    if (
        not isinstance(raw_items, list)
        or not raw_items
        or not all(isinstance(item, Mapping) for item in raw_items)
        or not isinstance(frozen_items, list)
        or not frozen_items
        or not all(isinstance(item, Mapping) for item in frozen_items)
        or not isinstance(fingerprint, str)
        or not fingerprint
    ):
        return False
    live_fingerprint = _fingerprint([dict(item) for item in raw_items])
    frozen_fingerprint = _fingerprint([dict(item) for item in frozen_items])
    return live_fingerprint == fingerprint == frozen_fingerprint


def build_virtual_response_context(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a frozen-bank-bound context without scoring or construct leakage."""

    bank_id = state.get("item_bank_id")
    bank_version = state.get("item_bank_version")
    frozen_items = state.get("frozen_item_bank")
    if (
        not isinstance(bank_id, str)
        or not bank_id
        or not isinstance(bank_version, int)
        or not isinstance(frozen_items, list)
        or not frozen_items
    ):
        raise ValueError("虚拟作答前必须先冻结已批准的题库")

    respondent_items = []
    for raw_item in frozen_items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目记录")
        options = raw_item.get("response_options")
        if not isinstance(options, list):
            raise ValueError("冻结题库中的题目缺少有效选项")
        respondent_items.append(
            {
                "item_id": raw_item.get("item_id"),
                "item_version": raw_item.get("version"),
                "context_category": raw_item.get("context_category"),
                "scenario": raw_item.get("scenario"),
                "response_instruction": raw_item.get("response_instruction"),
                "response_options": [
                    {
                        "option_id": option.get("option_id"),
                        "text": option.get("text"),
                    }
                    for option in options
                    if isinstance(option, Mapping)
                ],
            }
        )

    return {
        "item_bank_id": bank_id,
        "item_bank_version": bank_version,
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "virtual_sample_config": deepcopy(
            state.get("virtual_sample_config")
        ),
        "virtual_respondents": deepcopy(
            state.get("virtual_respondents") or []
        ),
        "items": respondent_items,
    }


def validate_item_bank_owned_fields(
    action: str,
    update: Mapping[str, Any],
) -> None:
    """Keep deterministic freeze fields out of all Agent-owned updates."""

    if action == "simulate_responses":
        allowed = {
            *_FREEZE_FIELDS,
            *_VIRTUAL_RESPONSE_FIELDS,
            *_DOWNSTREAM_INVALIDATION,
            "previous_virtual_response_data_ref",
        }
        unexpected = set(update) - allowed
        if unexpected:
            raise ValueError(
                "simulate_responses 返回了不属于虚拟作答阶段的字段："
                + "、".join(sorted(unexpected))
            )
        return
    overwritten = set(update) & _FREEZE_FIELDS
    if overwritten:
        raise ValueError(
            f"{action} 不允许修改题库冻结字段："
            + "、".join(sorted(overwritten))
        )


def build_blueprint_retention_gaps(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compute hard retention deficits from accepted items, not LLM prose."""

    blueprint = state.get("blueprint")
    if not isinstance(blueprint, Mapping):
        return []
    accepted_by_cell: dict[str, int] = {}
    for item in state.get("item_pool") or []:
        if not isinstance(item, Mapping):
            continue
        cell_id = item.get("blueprint_cell_id")
        if isinstance(cell_id, str) and cell_id:
            accepted_by_cell[cell_id] = accepted_by_cell.get(cell_id, 0) + 1

    gaps: list[dict[str, Any]] = []
    for cell in blueprint.get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        cell_id = cell.get("cell_id")
        planned = cell.get("planned_retention_count")
        if (
            not isinstance(cell_id, str)
            or not isinstance(planned, int)
            or isinstance(planned, bool)
        ):
            continue
        accepted = accepted_by_cell.get(cell_id, 0)
        missing = max(0, planned - accepted)
        if missing:
            gaps.append(
                {
                    "blueprint_cell_id": cell_id,
                    "target_dimension_id": cell.get("facet_id"),
                    "dimension_name": cell.get("facet_id"),
                    "planned_retention_count": planned,
                    "accepted_count": accepted,
                    "missing_count": missing,
                }
            )
    return gaps
