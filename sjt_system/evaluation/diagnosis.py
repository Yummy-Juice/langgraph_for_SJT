"""Construct-constrained diagnosis support for post-simulation item repair.

The functions in this module deliberately separate three responsibilities:

* deterministic statistics decide whether an item needs diagnosis;
* the evidence packet exposes the normal construct model and observations;
* validators restrict an LLM diagnosis and patch to one evidence-backed edit.

They do not infer a textual defect from a statistical threshold.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Any, Mapping


CITC_DIAGNOSIS_THRESHOLD = 0.20
DIFFICULTY_LOWER_BOUND = 0.20
DIFFICULTY_UPPER_BOUND = 0.80
MINIMUM_EFFECTIVE_OPTION_COUNT = 3

_EDITABLE_COMPONENTS = {"scenario", "response_options"}
_UPSTREAM_COMPONENTS = {
    "skeleton",
    "activation_mechanism",
    "behavior_evidence",
    "construct",
}
_NON_ACTIONABLE_COMPONENTS = {
    "simulation",
    "simulation_or_insufficient_evidence",
    "insufficient_evidence",
}
_ALL_COMPONENTS = (
    _EDITABLE_COMPONENTS | _UPSTREAM_COMPONENTS | _NON_ACTIONABLE_COMPONENTS
)
_LEVEL_RANK = {"low": 0, "medium_low": 1, "medium_high": 2, "high": 3}
_VTS_OBSERVATION_IDS = {"OBS:SAME_DOMAIN_VTS", "OBS:CROSS_DOMAIN_VTS"}
_FORCED_VTS_GRADIENT_OBSERVATION_IDS = {
    "OBS:SAME_DOMAIN_VTS_OPTION_MEAN_GRADIENT",
    "OBS:CROSS_DOMAIN_VTS_OPTION_MEAN_GRADIENT",
}
_VTS_CATEGORIES = ("same_domain", "cross_domain")
_TARGET_GRADIENT_REQUIRED_OBSERVATION_ID = (
    "OBS:TARGET_FACET_GRADIENT_REQUIRED"
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _failed_vts_categories(observations: Any) -> set[str]:
    failed: set[str] = set()
    observation_to_category = {
        "OBS:SAME_DOMAIN_VTS": "same_domain",
        "OBS:CROSS_DOMAIN_VTS": "cross_domain",
    }
    for row in observations or []:
        if not isinstance(row, Mapping):
            continue
        category = observation_to_category.get(str(row.get("observation_id") or ""))
        if category is None:
            continue
        value = _number(row.get("value"))
        threshold = _number(row.get("threshold"))
        if (
            row.get("passes") is False
            or value is None
            or (threshold is not None and value < threshold)
        ):
            failed.add(category)
    return failed


def derive_forced_vts_gradient_repairs(
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Derive deterministic endpoint-repair triggers from failed VTS means.

    The trigger is intentionally strict: all four scoring levels must be
    present exactly once, all corresponding non-target means must be finite,
    and the means must increase strictly from score 1 through score 4.
    """

    existing = evidence.get("forced_vts_gradient_repairs")
    if isinstance(existing, list):
        return [
            deepcopy(dict(row))
            for row in existing
            if isinstance(row, Mapping)
        ]
    raw_diagnostics = evidence.get("option_choice_diagnostics") or {}
    paired = (
        raw_diagnostics.get("option_score_comparisons")
        if isinstance(raw_diagnostics, Mapping)
        else None
    ) or evidence.get("option_score_comparisons") or []
    failed_categories = _failed_vts_categories(evidence.get("observations"))
    repairs: list[dict[str, Any]] = []
    for category in _VTS_CATEGORIES:
        if category not in failed_categories:
            continue
        mean_key = f"{category}_mean_score"
        rows_by_score: dict[int, dict[str, Any]] = {}
        duplicate_score = False
        for row in paired:
            if not isinstance(row, Mapping):
                continue
            score = _number(row.get("score"))
            option_id = _text(row.get("option_id"))
            mean = _number(row.get(mean_key))
            if (
                score is None
                or score not in {1.0, 2.0, 3.0, 4.0}
                or not option_id
                or mean is None
            ):
                continue
            score_int = int(score)
            if score_int in rows_by_score:
                duplicate_score = True
                break
            rows_by_score[score_int] = {
                "option_id": option_id,
                "target_mean_score": _number(row.get("target_mean_score")),
                "non_target_mean_score": mean,
            }
        if duplicate_score or set(rows_by_score) != {1, 2, 3, 4}:
            continue
        non_target_means = [
            rows_by_score[score]["non_target_mean_score"]
            for score in range(1, 5)
        ]
        if not all(
            current > previous
            for previous, current in zip(non_target_means, non_target_means[1:])
        ):
            continue
        target_means = [
            rows_by_score[score]["target_mean_score"]
            for score in range(1, 5)
        ]
        observation_id = (
            "OBS:SAME_DOMAIN_VTS_OPTION_MEAN_GRADIENT"
            if category == "same_domain"
            else "OBS:CROSS_DOMAIN_VTS_OPTION_MEAN_GRADIENT"
        )
        repairs.append(
            {
                "vts_category": category,
                "observation_id": observation_id,
                "vts_observation_id": (
                    "OBS:SAME_DOMAIN_VTS"
                    if category == "same_domain"
                    else "OBS:CROSS_DOMAIN_VTS"
                ),
                "option_ids_by_score": {
                    str(score): rows_by_score[score]["option_id"]
                    for score in range(1, 5)
                },
                "endpoint_option_ids": [
                    rows_by_score[1]["option_id"],
                    rows_by_score[4]["option_id"],
                ],
                "target_means_by_score": {
                    str(score): rows_by_score[score]["target_mean_score"]
                    for score in range(1, 5)
                },
                "non_target_means_by_score": {
                    str(score): rows_by_score[score]["non_target_mean_score"]
                    for score in range(1, 5)
                },
            }
        )
    return repairs


def _forced_vts_gradient_repair_ids(
    evidence: Mapping[str, Any],
) -> set[str]:
    return {
        str(row.get("observation_id"))
        for row in derive_forced_vts_gradient_repairs(evidence)
        if row.get("observation_id")
    }


def _has_direct_item_quote(value: Any, evidence: Mapping[str, Any]) -> bool:
    text = _text(value)
    if not text:
        return False
    item = evidence.get("current_item") or {}
    source_texts = [
        _text(item.get("scenario")),
        *[
            _text(row.get("text"))
            for row in item.get("response_options") or []
            if isinstance(row, Mapping)
        ],
    ]
    candidates = [text.strip(" \t\r\n\"'“”‘’")]
    candidates.extend(
        match.strip()
        for match in re.findall(r'["“‘]([^"”’]+)["”’]', text)
    )
    return any(
        len(candidate) >= 2 and candidate in source
        for candidate in candidates
        for source in source_texts
        if source
    )


def _references_vts(candidate: Mapping[str, Any]) -> bool:
    return bool(
        _VTS_OBSERVATION_IDS
        & {str(value) for value in candidate.get("observation_refs") or []}
    )


def _has_direct_contaminant_expression(
    candidate: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> bool:
    """Accept quoted wording that directly expresses a supplied contaminant.

    Construct contamination can co-measure a non-target facet without
    contradicting the target facet.  For a VTS repair we therefore require a
    current-item quote plus an explicit supplied NON_TARGET constraint, rather
    than a target-construct contradiction.
    """

    observation_refs = {
        str(value) for value in candidate.get("observation_refs") or []
    }
    constraint_refs = {
        str(value) for value in candidate.get("constraint_refs") or []
    }
    has_matching_constraint = True
    if "OBS:SAME_DOMAIN_VTS" in observation_refs:
        has_matching_constraint = has_matching_constraint and any(
            value.startswith("NON_TARGET_SAME_DOMAIN:")
            for value in constraint_refs
        )
    if "OBS:CROSS_DOMAIN_VTS" in observation_refs:
        has_matching_constraint = has_matching_constraint and any(
            value.startswith("NON_TARGET_CROSS_DOMAIN:")
            for value in constraint_refs
        )
    return bool(
        _references_vts(candidate)
        and has_matching_constraint
        and _has_direct_item_quote(candidate.get("textual_evidence"), evidence)
    )


def _repair_has_allowed_text_evidence(
    candidate: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    gradient_failed: bool,
    forced_vts_gradient: bool = False,
) -> bool:
    if forced_vts_gradient:
        forced_ids = _forced_vts_gradient_repair_ids(evidence)
        candidate_ids = {
            str(value) for value in candidate.get("observation_refs") or []
        }
        candidate_constraint_ids = {
            str(value) for value in candidate.get("constraint_refs") or []
        }
        required_constraint_prefixes = {
            (
                "NON_TARGET_SAME_DOMAIN:"
                if row.get("vts_category") == "same_domain"
                else "NON_TARGET_CROSS_DOMAIN:"
            )
            for row in derive_forced_vts_gradient_repairs(evidence)
        }
        endpoint_ids = {
            str(option_id)
            for repair in derive_forced_vts_gradient_repairs(evidence)
            for option_id in repair.get("endpoint_option_ids") or []
        }
        return bool(
            forced_ids.issubset(candidate_ids)
            and all(
                any(value.startswith(prefix) for value in candidate_constraint_ids)
                for prefix in required_constraint_prefixes
            )
            and candidate.get("suspect_components") == ["response_options"]
            and endpoint_ids.issubset(
                {str(value) for value in candidate.get("affected_option_ids") or []}
            )
        )
    if (
        gradient_failed
        and "OBS:TARGET_OPTION_GRADIENT"
        in (candidate.get("observation_refs") or [])
        and "response_options" in (candidate.get("suspect_components") or [])
    ):
        return True
    if (
        candidate.get("suspect_components") == ["response_options"]
        and _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID
        in {str(value) for value in candidate.get("observation_refs") or []}
    ):
        return True
    if _references_vts(candidate):
        return _has_direct_contaminant_expression(candidate, evidence)
    return _has_direct_item_quote(candidate.get("textual_evidence"), evidence)


def _textual_evidence_may_be_empty(
    candidate: Mapping[str, Any],
    *,
    gradient_failed: bool,
    forced_vts_gradient: bool,
) -> bool:
    """Keep the prompt's deterministic gradient exception aligned with validation."""

    if forced_vts_gradient:
        return True
    if (
        candidate.get("suspect_components") == ["response_options"]
        and _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID
        in {str(value) for value in candidate.get("observation_refs") or []}
    ):
        return True
    return bool(
        gradient_failed
        and candidate.get("suspect_components") == ["response_options"]
        and "OBS:TARGET_OPTION_GRADIENT"
        in {str(value) for value in candidate.get("observation_refs") or []}
    )


def build_deterministic_defer_advice(
    evidence: Mapping[str, Any],
    *,
    validation_error: str,
) -> dict[str, Any] | None:
    """Create a safe per-item defer result after an invalid diagnosis.

    An ordinary VTS repair is not safe to execute without a grounded quote and
    a matching NON_TARGET constraint.  The model response must therefore not
    be weakened or guessed at when validation fails.  This fallback preserves
    the failed observation and moves only this item to manual review so that a
    malformed diagnosis cannot stop the whole diagnosis batch.

    Deterministic gradient triggers are intentionally excluded: callers must
    try their dedicated repair fallbacks first, and validation still rejects a
    defer when one of those triggers is present.
    """

    # 确定性梯度触发不再强制排除 defer：当专用 repair fallback 因数据/约束
    # 不完整而不可用时，defer（人工复核）是安全出口，避免整个诊断批停止。
    observations = [
        row
        for row in evidence.get("observations") or []
        if isinstance(row, Mapping)
    ]
    failed_vts_categories = _failed_vts_categories(observations)
    failed_observation = next(
        (
            row
            for row in observations
            if str(row.get("observation_id") or "") in _VTS_OBSERVATION_IDS
            and (
                row.get("passes") is False
                or _number(row.get("value")) is None
                or (
                    _number(row.get("threshold")) is not None
                    and _number(row.get("value")) < _number(row.get("threshold"))
                )
            )
        ),
        None,
    )
    if failed_observation is None:
        failed_observation = next(
            (
                row
                for row in observations
                if row.get("passes") is False
                or (
                    _number(row.get("value")) is not None
                    and _number(row.get("threshold")) is not None
                    and _number(row.get("value")) < _number(row.get("threshold"))
                )
            ),
            None,
        )
    if not isinstance(failed_observation, Mapping):
        return None

    observation_id = _text(failed_observation.get("observation_id"))
    if not observation_id:
        return None

    constraint_ids = [
        _text(row.get("constraint_id"))
        for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping) and _text(row.get("constraint_id"))
    ]
    if failed_vts_categories:
        prefixes = {
            (
                "NON_TARGET_SAME_DOMAIN:"
                if category == "same_domain"
                else "NON_TARGET_CROSS_DOMAIN:"
            )
            for category in failed_vts_categories
        }
        matching_constraint_ids = [
            value
            for value in constraint_ids
            if any(value.startswith(prefix) for prefix in prefixes)
        ]
    else:
        matching_constraint_ids = []
    candidate_constraint_ids = matching_constraint_ids or constraint_ids[:1]
    if not candidate_constraint_ids:
        return None

    validation_summary = _text(validation_error) or "返修诊断未通过证据校验"
    return {
        "item_id": _text(evidence.get("item_id")),
        "decision": "defer",
        "observed_discrepancies": [
            {
                "observation_refs": [observation_id],
                "constraint_refs": matching_constraint_ids,
                "description": (
                    "模型返修诊断未形成可执行的题面证据链，"
                    f"暂缓自动修改：{validation_summary}"
                ),
            }
        ],
        "candidate_diagnoses": [
            {
                "diagnosis_id": "deterministic-insufficient-repair-evidence",
                "suspect_components": ["insufficient_evidence"],
                "affected_option_ids": [],
                "observation_refs": [observation_id],
                "constraint_refs": candidate_constraint_ids,
                "textual_evidence": "",
                "explanation": (
                    "当前统计观察不足以授权自动题面修改；模型诊断未满足"
                    "当前题目原文引用和构念约束链接要求，转人工审核。"
                ),
                "confidence": "low",
            }
        ],
        "summary": "返修诊断证据链不完整，本题自动转为同槽位补题，其他题继续处理。",
        "repair_tasks": [],
    }


def build_deterministic_target_gradient_repair_advice(
    evidence: Mapping[str, Any],
    *,
    validation_error: str,
) -> dict[str, Any] | None:
    """Recover only an explicit target-gradient repair after an invalid VTS link.

    The model is still required to diagnose ordinary VTS contamination with a
    quoted item span and a matching ``NON_TARGET`` constraint.  A failed target
    option gradient is different: it is an explicit, deterministic repair
    trigger already authorized by the diagnosis contract.  If the model mixes
    that trigger with an invalid ordinary-VTS candidate, retain only the
    deterministic gradient scope so a bad optional branch cannot discard a
    valid repair opportunity or silently become a contamination edit.
    """

    if not (
        validation_error.startswith(
            "repair task must quote current item wording; VTS repair must"
        )
        or validation_error.startswith(
            "selected repair diagnosis must quote current item wording; VTS repair may"
        )
        or validation_error == "constraint refs cannot be empty"
        or validation_error
        == "目标组选项梯度失败时必须 repair，不能 defer"
    ):
        return None
    # 注意：不因 derive_forced_vts_gradient_repairs 非空而放弃。
    # forced_vts 端点修复有更严格的契约（NON_TARGET constraint + 端点数据），
    # 在其不可用（返回 None）时，本 fallback 仍是“目标组选项梯度失败”的
    # 确定性修复出口；本函数自身通过 OBS:TARGET_OPTION_GRADIENT 检查保证
    # 只在目标组梯度确实失败时生成，不会误用为 forced VTS 修复。

    gradient_observation = next(
        (
            row
            for row in evidence.get("observations") or []
            if isinstance(row, Mapping)
            and row.get("observation_id") == "OBS:TARGET_OPTION_GRADIENT"
        ),
        None,
    )
    if not isinstance(gradient_observation, Mapping) or gradient_observation.get(
        "passes"
    ) is True:
        return None

    valid_option_ids = {
        str(row.get("option_id"))
        for row in evidence.get("option_evidence") or []
        if isinstance(row, Mapping) and row.get("option_id")
    }
    failed_pairs: list[tuple[str, str]] = []
    for row in gradient_observation.get("failed_adjacent_pairs") or []:
        if not isinstance(row, Mapping):
            continue
        lower = str(row.get("lower_option_id") or "")
        higher = str(row.get("higher_option_id") or "")
        if (
            lower
            and higher
            and lower != higher
            and {lower, higher}.issubset(valid_option_ids)
            and (lower, higher) not in failed_pairs
        ):
            failed_pairs.append((lower, higher))
    if not failed_pairs:
        return None

    # Keep tasks non-overlapping.  For a chain such as D-C and C-B, rewriting
    # the shared middle option is the smallest scope that addresses both
    # failures.  A singleton task also avoids introducing a new level-order
    # assumption when a legacy item has incomplete behavioral-level metadata.
    task_option_ids: list[str] = []
    covered_pairs: set[int] = set()
    for index, pair in enumerate(failed_pairs):
        if index in covered_pairs:
            continue
        shared_with_next = (
            set(pair) & set(failed_pairs[index + 1])
            if index + 1 < len(failed_pairs)
            else set()
        )
        chosen = next(iter(shared_with_next), pair[1])
        if chosen in task_option_ids:
            continue
        task_option_ids.append(chosen)
        for pair_index, candidate_pair in enumerate(failed_pairs):
            if chosen in candidate_pair:
                covered_pairs.add(pair_index)

    if not task_option_ids or len(task_option_ids) > 4:
        return None
    valid_constraints = [
        str(row.get("constraint_id"))
        for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping) and row.get("constraint_id")
    ]
    target_constraint = next(
        (
            constraint_id
            for constraint_id in valid_constraints
            if constraint_id.startswith(("FACET_DEFINITION:", "BE_DEFINITION:", "BE_HIGH:", "BE_LOW:"))
        ),
        None,
    )
    if target_constraint is None:
        return None

    diagnosis_id = "deterministic-target-option-gradient"
    pair_label = ", ".join(f"{lower}<{higher}" for lower, higher in failed_pairs)
    affected_option_ids: list[str] = []
    for pair in failed_pairs:
        for option_id in pair:
            if option_id not in affected_option_ids:
                affected_option_ids.append(option_id)
    tasks = [
        {
            "diagnosis_id": diagnosis_id,
            "phase": "target_facet_gradient" if index == 0 else "other",
            "atomic_edit": {
                "target_field": "response_options",
                "option_ids": [option_id],
                "problem": f"目标组选项梯度失败，涉及相邻对 {pair_label}",
                "instruction": (
                    f"仅重写选项 {option_id} 的题面，保留行为等级、计分键和心理骨架，"
                    "补足目标facet的相邻梯度线索。"
                ),
            },
        }
        for index, option_id in enumerate(task_option_ids)
    ]
    return {
        "item_id": str(evidence.get("item_id") or ""),
        "decision": "repair",
        "observed_discrepancies": [
            {
                "observation_refs": ["OBS:TARGET_OPTION_GRADIENT"],
                "constraint_refs": [target_constraint],
                "description": f"目标组选项梯度存在失败相邻对：{pair_label}",
            }
        ],
        "candidate_diagnoses": [
            {
                "diagnosis_id": diagnosis_id,
                "suspect_components": ["response_options"],
                "affected_option_ids": affected_option_ids,
                "observation_refs": ["OBS:TARGET_OPTION_GRADIENT"],
                "constraint_refs": [target_constraint],
                "textual_evidence": "",
                "explanation": (
                    "目标组选项梯度是明确的确定性返修触发器；模型未能提供普通VTS"
                    "返修所需的题面引句和NON_TARGET约束链接，因此不自动修改VTS污染。"
                ),
                "confidence": "high",
            }
        ],
        "repair_tasks": tasks,
        "summary": (
            "模型普通VTS证据链无效；仅保留明确的目标组选项梯度失败，"
            "生成最小response_options返修，普通VTS污染不自动修改。"
        ),
    }


def build_deterministic_forced_vts_repair_advice(
    evidence: Mapping[str, Any],
    *,
    validation_error: str,
) -> dict[str, Any] | None:
    """Recover a malformed model response for a forced VTS trigger.

    A strict non-target option-mean gradient is already a deterministic repair
    authorization.  The model still supplies useful prose in the normal path,
    but missing trigger references or endpoints must not turn this deterministic
    evidence into an endpoint outage.  This fallback requires both the trigger
    evidence and matching constraint IDs to be present in the local packet.
    """

    repairs = derive_forced_vts_gradient_repairs(evidence)
    if not repairs:
        return None

    valid_constraints = [
        str(row.get("constraint_id"))
        for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping) and row.get("constraint_id")
    ]
    constraint_refs: list[str] = []
    for category in ("same_domain", "cross_domain"):
        if not any(row.get("vts_category") == category for row in repairs):
            continue
        prefix = (
            "NON_TARGET_SAME_DOMAIN:"
            if category == "same_domain"
            else "NON_TARGET_CROSS_DOMAIN:"
        )
        matching = [value for value in valid_constraints if value.startswith(prefix)]
        if not matching:
            return None
        # Keep the definition and the two behavior poles when available; the
        # complete matching set is retained as a deterministic fallback for
        # legacy constraint IDs that do not use those suffixes.
        preferred = [
            value
            for suffix in (":DEFINITION", ":HIGH", ":LOW")
            for value in matching
            if value.endswith(suffix)
        ]
        for value in preferred or matching:
            if value not in constraint_refs:
                constraint_refs.append(value)

    endpoint_ids: list[str] = []
    for repair in repairs:
        for option_id in repair.get("endpoint_option_ids") or []:
            option_id = str(option_id)
            if option_id not in endpoint_ids:
                endpoint_ids.append(option_id)
    valid_option_ids = {
        str(row.get("option_id"))
        for row in evidence.get("option_evidence") or []
        if isinstance(row, Mapping) and row.get("option_id")
    }
    if len(endpoint_ids) != 2 or not set(endpoint_ids).issubset(valid_option_ids):
        return None

    # The workflow requires a target-gradient first task. Prefer an interior
    # adjacent pair so the first atomic edit still covers a gradient while
    # remaining non-overlapping with the forced score-1/score-4 endpoint task.
    ordered = _ordered_target_option_rows(evidence)
    endpoint_id_set = set(endpoint_ids)
    preflight_option_ids: list[str] = []
    for left, right in zip(ordered, ordered[1:]):
        pair_ids = [str(left["option_id"]), str(right["option_id"])]
        if not endpoint_id_set.intersection(pair_ids):
            preflight_option_ids = pair_ids
            break
    interior_ids = [
        str(row["option_id"])
        for row in ordered
        if str(row["option_id"]) not in endpoint_id_set
    ]
    if not interior_ids:
        # Legacy/minimal fixtures may omit option scores from option_evidence;
        # the deterministic trigger itself still carries the numeric order.
        by_score: dict[int, str] = {}
        for repair in repairs:
            for raw_score, option_id in (repair.get("option_ids_by_score") or {}).items():
                try:
                    score = int(float(raw_score))
                except (TypeError, ValueError):
                    continue
                option_id = str(option_id)
                if option_id and option_id not in endpoint_id_set:
                    by_score.setdefault(score, option_id)
        interior_ids = [by_score[score] for score in sorted(by_score)]
    if not interior_ids:
        return None
    if not preflight_option_ids:
        # If the fixture omits scores/levels, validation cannot establish that
        # a two-option edit is adjacent; keep a singleton safe interior scope.
        preflight_option_ids = interior_ids[:1]
    diagnosis_id = "forced-vts-endpoints"
    trigger_ids = [
        str(row["observation_id"])
        for row in repairs
        if row.get("observation_id")
    ]
    affected_option_ids = [*endpoint_ids, *preflight_option_ids]
    return {
        "item_id": str(evidence.get("item_id") or ""),
        "decision": "repair",
        "observed_discrepancies": [
            {
                "observation_refs": [
                    str(row["vts_observation_id"]),
                    str(row["observation_id"]),
                ],
                "constraint_refs": [
                    value
                    for value in constraint_refs
                    if value.startswith(
                        "NON_TARGET_SAME_DOMAIN:"
                        if row.get("vts_category") == "same_domain"
                        else "NON_TARGET_CROSS_DOMAIN:"
                    )
                ],
                "description": (
                    f"{row['observation_id']} 表明非目标组均值在计分值1至4间严格上升，"
                    "按强制 VTS 端点规则返修。"
                ),
            }
            for row in repairs
        ],
        "candidate_diagnoses": [
            {
                "diagnosis_id": diagnosis_id,
                "suspect_components": ["response_options"],
                "affected_option_ids": affected_option_ids,
                "observation_refs": [
                    *trigger_ids,
                    _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID,
                ],
                "constraint_refs": constraint_refs,
                "textual_evidence": "",
                "explanation": (
                    "模型诊断缺少强制 VTS 契约字段；本地已确认同域/跨域非目标组均值"
                    "均按分数1至4严格上升，因此仅按证据生成目标梯度首任务和分数端点任务。"
                ),
                "confidence": "high",
            }
        ],
        "repair_tasks": [
            {
                "diagnosis_id": diagnosis_id,
                "phase": "target_facet_gradient",
                "atomic_edit": {
                    "target_field": "response_options",
                    "option_ids": preflight_option_ids,
                    "problem": "目标facet梯度返修首任务需要先行执行。",
                    "instruction": (
                        f"仅优化选项 {', '.join(preflight_option_ids)} 的目标facet行为线索，"
                        "保留behavioral_level、scoring_key、心理骨架和目标facet。"
                    ),
                },
            },
            {
                "diagnosis_id": diagnosis_id,
                "phase": "other",
                "atomic_edit": {
                    "target_field": "response_options",
                    "option_ids": endpoint_ids,
                    "problem": "强制 VTS 触发器要求清理分数1和分数4端点的非目标构念梯度。",
                    "instruction": (
                        "将分数1选项向对应污染facet的HIGH行为靠拢，将分数4选项向"
                        "对应污染facet的LOW行为靠拢；保留目标facet梯度、行为等级、计分键、"
                        "心理骨架、激活机制和构念。"
                    ),
                },
            },
        ],
        "summary": (
            "模型强制VTS诊断契约无效；已按本地确定性均值梯度证据生成最小安全返修。"
        ),
    }


def _ordered_target_option_rows(
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return option rows in fixed numeric scoring order."""

    raw_gradient = evidence.get("target_option_gradient") or (
        evidence.get("option_choice_diagnostics") or {}
    ).get("target_option_gradient") or {}
    gradient_means = {
        str(row.get("option_id")): _number(row.get("target_facet_mean"))
        for row in (raw_gradient.get("options") or [])
        if isinstance(row, Mapping) and row.get("option_id")
    } if isinstance(raw_gradient, Mapping) else {}
    scoring_key = {
        str(key): _number(value)
        for key, value in ((evidence.get("current_item") or {}).get("scoring_key") or {}).items()
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in evidence.get("option_evidence") or []:
        if not isinstance(raw, Mapping):
            continue
        option_id = _text(raw.get("option_id"))
        score = _number(raw.get("score"))
        if score is None:
            score = scoring_key.get(option_id)
        if not option_id or score is None or option_id in seen:
            continue
        seen.add(option_id)
        row = {
            "option_id": option_id,
            "score": score,
            "behavioral_level": raw.get("behavioral_level"),
        }
        for key in ("target_facet_mean", "facet_mean", "target_mean_score"):
            value = _number(raw.get(key))
            if value is not None:
                row["target_facet_mean"] = value
                break
        if row.get("target_facet_mean") is None and gradient_means.get(option_id) is not None:
            row["target_facet_mean"] = gradient_means[option_id]
        rows.append(row)
    rows.sort(key=lambda row: (float(row["score"]), str(row["option_id"])))
    return rows


def _target_gradient_pairs(
    evidence: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = _ordered_target_option_rows(evidence)
    if len(rows) < 2:
        return []
    return list(zip(rows, rows[1:]))


def _target_gradient_pair_for_repair(
    evidence: Mapping[str, Any],
    occupied_option_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Choose the smallest evidence-backed adjacent target-gradient scope."""

    pairs = _target_gradient_pairs(evidence)
    if not pairs:
        return None
    occupied = occupied_option_ids or set()
    gradient = evidence.get("target_option_gradient") or (
        evidence.get("option_choice_diagnostics") or {}
    ).get("target_option_gradient") or {}
    if not isinstance(gradient, Mapping):
        gradient = {}
    failed_pairs = gradient.get("failed_adjacent_pairs") or []
    failed_ids: list[tuple[str, str]] = []
    for raw in failed_pairs:
        if not isinstance(raw, Mapping):
            continue
        pair = (_text(raw.get("lower_option_id")), _text(raw.get("higher_option_id")))
        if pair[0] and pair[1]:
            failed_ids.append(pair)
    for left, right in pairs:
        ids = {left["option_id"], right["option_id"]}
        if ids & occupied:
            continue
        if any(set(pair) == ids for pair in failed_ids):
            return left, right

    # When no failed pair is available, use the smallest finite target-facet
    # mean gap. This is still a local adjacent edit, never a global rewrite.
    finite = [
        (abs(float(right["target_facet_mean"]) - float(left["target_facet_mean"])), left, right)
        for left, right in pairs
        if left.get("target_facet_mean") is not None
        and right.get("target_facet_mean") is not None
        and not ({left["option_id"], right["option_id"]} & occupied)
    ]
    if finite:
        finite.sort(key=lambda row: (row[0], float(row[1]["score"]), str(row[1]["option_id"])))
        return finite[0][1], finite[0][2]

    # With no estimable means, use the middle adjacent pair when possible.
    available = [pair for pair in pairs if not ({pair[0]["option_id"], pair[1]["option_id"]} & occupied)]
    if not available:
        available = pairs
    return available[(len(available) - 1) // 2]


def _target_constraint_id(evidence: Mapping[str, Any]) -> str | None:
    rows = [
        row for row in evidence.get("target_construct_constraints") or []
        if isinstance(row, Mapping) and row.get("constraint_id")
    ]
    rows += [
        row for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping) and row.get("constraint_id")
    ]
    for row in rows:
        constraint_id = str(row["constraint_id"])
        if constraint_id.startswith(("FACET_DEFINITION:", "BE_DEFINITION:", "BE_HIGH:", "BE_LOW:")):
            return constraint_id
    return str(rows[0]["constraint_id"]) if rows else None


def normalize_target_gradient_repair_advice(
    advice: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Ensure every executable repair starts with a target-gradient task.

    The normalizer is deliberately deterministic and only runs after the
    diagnosis agent has selected ``repair``. It may add or reorder an atomic
    response-option task, but never changes scoring, behavioral levels,
    skeleton, activation mechanism, or target facet fields.
    """

    normalized = deepcopy(dict(advice))
    if normalized.get("decision") != "repair":
        return normalized
    tasks = repair_tasks_from_advice(normalized)
    if not tasks:
        tasks = []
    existing_target_index = next(
        (
            index
            for index, task in enumerate(tasks)
            if task.get("phase") == "target_facet_gradient"
            or str(task.get("diagnosis_id")) == "deterministic-target-option-gradient"
        ),
        None,
    )
    existing_option_ids = {
        str(option_id)
        for task in tasks
        if isinstance(task, Mapping)
        for option_id in ((task.get("atomic_edit") or {}).get("option_ids") or [])
    }
    pair = (
        None
        if existing_target_index is not None
        else _target_gradient_pair_for_repair(evidence, existing_option_ids)
    )
    # If every adjacent pair is already occupied, reuse a response-options
    # task as the mandatory phase instead of creating an overlapping scope.
    target_index: int | None = None
    if pair is None and existing_target_index is None:
        pair = _target_gradient_pair_for_repair(evidence)
    if pair is None and existing_target_index is None:
        normalized["decision"] = "defer"
        normalized["repair_tasks"] = []
        normalized["summary"] = (
            str(normalized.get("summary") or "").rstrip("。")
            + "；题面缺少至少两个可排序选项，无法安全执行目标facet梯度首任务，自动 defer。"
        )
        return normalized
    pair_ids = (
        {pair[0]["option_id"], pair[1]["option_id"]}
        if pair is not None
        else {
            str(value)
            for value in (tasks[existing_target_index].get("atomic_edit") or {}).get("option_ids") or []
        }
    )
    for index, task in enumerate(tasks):
        edit = task.get("atomic_edit") or {}
        if edit.get("target_field") != "response_options":
            continue
        option_ids = {str(value) for value in edit.get("option_ids") or []}
        if option_ids and option_ids.issubset(pair_ids):
            target_index = index
            break

    candidate_rows = [
        row for row in normalized.get("candidate_diagnoses") or []
        if isinstance(row, Mapping)
    ]
    candidate_by_id = {
        str(row.get("diagnosis_id")): row for row in candidate_rows
        if row.get("diagnosis_id")
    }
    target_constraint = _target_constraint_id(evidence)
    if target_index is None and existing_target_index is not None:
        target_index = existing_target_index
    if target_index is None:
        diagnosis_id = "target-facet-gradient-preflight"
        if diagnosis_id not in candidate_by_id:
            if target_constraint is None:
                raise ValueError("无法构造目标facet梯度首任务：缺少目标构念约束")
            candidate = {
                "diagnosis_id": diagnosis_id,
                "suspect_components": ["response_options"],
                "affected_option_ids": [pair[0]["option_id"], pair[1]["option_id"]],
                "observation_refs": [_TARGET_GRADIENT_REQUIRED_OBSERVATION_ID],
                "constraint_refs": [target_constraint],
                "textual_evidence": "",
                "explanation": "所有进入repair的题目先执行目标facet相邻选项梯度优化。",
                "confidence": "high",
            }
            candidate_rows.insert(0, candidate)
            candidate_by_id[diagnosis_id] = candidate
        target_task = {
            "diagnosis_id": diagnosis_id,
            "phase": "target_facet_gradient",
            "atomic_edit": {
                "target_field": "response_options",
                "option_ids": [pair[0]["option_id"], pair[1]["option_id"]],
                "problem": "目标facet选项梯度需要先行优化",
                "instruction": (
                    f"先仅优化分数{pair[0]['score']}与{pair[1]['score']}的相邻选项，"
                    "强化目标facet构念梯度；保留behavioral_level、scoring_key、心理骨架和目标facet。"
                ),
            },
        }
        # Keep the bounded four-task contract. Existing tasks that overlap the
        # mandatory pair cannot remain executable because scopes must not overlap.
        tasks = [
            target_task,
            *[
                task for task in tasks
                if not pair_ids.intersection(
                    {str(value) for value in ((task.get("atomic_edit") or {}).get("option_ids") or [])}
                )
            ],
        ][:4]
    else:
        target_task = deepcopy(dict(tasks.pop(target_index)))
        target_task["phase"] = "target_facet_gradient"
        target_task["atomic_edit"] = deepcopy(dict(target_task.get("atomic_edit") or {}))
        tasks.insert(0, target_task)

    target_option_ids = {
        str(value)
        for value in (target_task.get("atomic_edit") or {}).get("option_ids") or []
    }
    tasks = [
        target_task,
        *[
            task
            for task in tasks[1:]
            if not target_option_ids.intersection(
                {
                    str(value)
                    for value in ((task.get("atomic_edit") or {}).get("option_ids") or [])
                }
            )
        ],
    ][:4]

    target_diagnosis_id = str(target_task.get("diagnosis_id") or "")
    for row in candidate_rows:
        if str(row.get("diagnosis_id")) != target_diagnosis_id:
            continue
        observation_refs = [str(value) for value in row.get("observation_refs") or []]
        if (
            target_diagnosis_id != "deterministic-target-option-gradient"
            and _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID not in observation_refs
        ):
            observation_refs.append(_TARGET_GRADIENT_REQUIRED_OBSERVATION_ID)
        row["observation_refs"] = observation_refs
        if target_constraint and target_constraint not in (row.get("constraint_refs") or []):
            row["constraint_refs"] = [*(row.get("constraint_refs") or []), target_constraint]
        row["suspect_components"] = ["response_options"]
        affected = [str(value) for value in row.get("affected_option_ids") or []]
        for option_id in target_task["atomic_edit"].get("option_ids") or []:
            if option_id not in affected:
                affected.append(option_id)
        row["affected_option_ids"] = affected
        break

    # The synthetic candidate can be used by validation even if the model
    # returned only unrelated candidates; retain candidates referenced by tasks.
    referenced = {str(task.get("diagnosis_id")) for task in tasks}
    candidate_rows = [
        row for row in candidate_rows
        if str(row.get("diagnosis_id")) in referenced or row.get("diagnosis_id") == "target-facet-gradient-preflight"
    ][:3]
    normalized["candidate_diagnoses"] = candidate_rows
    normalized["repair_tasks"] = tasks
    normalized["summary"] = (
        str(normalized.get("summary") or "").rstrip("。")
        + "；首任务已规范化为目标facet梯度选项优化。"
    )
    return normalized


def _semantic_constraint(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the constraint reference and wording needed for diagnosis."""

    result: dict[str, Any] = {}
    for key in ("constraint_id", "component", "statement", "text", "level"):
        if row.get(key) is not None:
            result[key] = deepcopy(row[key])
    return result


def _semantic_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strip routing/group metadata while preserving diagnostic evidence."""

    result: dict[str, Any] = {}
    for key in (
        "observation_id", "metric", "value", "threshold", "passes", "estimable",
        "role", "filtering_authority", "failed_adjacent_pairs", "vts_category",
        "vts_observation_id", "endpoint_option_ids", "option_ids_by_score",
        "target_means_by_score", "non_target_means_by_score",
    ):
        if row.get(key) is not None:
            result[key] = deepcopy(row[key])
    return result


def _semantic_target_gradient(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("estimable", "passes", "filtering_authority"):
        if raw.get(key) is not None:
            result[key] = deepcopy(raw[key])
    options = []
    for row in raw.get("options") or []:
        if not isinstance(row, Mapping):
            continue
        option = {
            key: deepcopy(row[key])
                for key in ("option_id", "score", "target_facet_mean", "standard_error")
            if row.get(key) is not None
        }
        if option.get("option_id"):
            options.append(option)
    if options:
        result["options"] = options
    for key in ("failed_adjacent_pairs", "adjacent_comparisons"):
        rows = []
        for row in raw.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            rows.append({
                field: deepcopy(row[field])
                for field in (
                    "lower_option_id", "higher_option_id", "lower_mean",
                    "higher_mean", "passes", "direction",
                )
                if row.get(field) is not None
            })
        if rows:
            result[key] = rows
    return result


def _forced_vts_contract_gaps(
    candidate: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Describe missing deterministic references for a forced VTS repair."""

    repairs = derive_forced_vts_gradient_repairs(evidence)
    forced_ids = {
        str(row.get("observation_id"))
        for row in repairs
        if row.get("observation_id")
    }
    candidate_ids = {str(value) for value in candidate.get("observation_refs") or []}
    gaps: list[str] = []
    if not forced_ids.issubset(candidate_ids):
        gaps.append(
            "observation_refs must include " + ", ".join(sorted(forced_ids))
        )

    constraint_refs = {
        str(value) for value in candidate.get("constraint_refs") or []
    }
    required_prefixes = {
        (
            "NON_TARGET_SAME_DOMAIN:"
            if row.get("vts_category") == "same_domain"
            else "NON_TARGET_CROSS_DOMAIN:"
        )
        for row in repairs
    }
    missing_prefixes = sorted(
        prefix
        for prefix in required_prefixes
        if not any(value.startswith(prefix) for value in constraint_refs)
    )
    if missing_prefixes:
        gaps.append(
            "constraint_refs must include " + ", ".join(missing_prefixes)
        )

    if candidate.get("suspect_components") != ["response_options"]:
        gaps.append('suspect_components must equal ["response_options"]')

    endpoint_ids = {
        str(option_id)
        for row in repairs
        for option_id in row.get("endpoint_option_ids") or []
    }
    affected = {str(value) for value in candidate.get("affected_option_ids") or []}
    if not endpoint_ids.issubset(affected):
        gaps.append(
            "affected_option_ids must include score endpoints "
            + ", ".join(sorted(endpoint_ids))
        )
    return gaps


def _citc_value(statistics: Mapping[str, Any]) -> float | None:
    payload = (
        statistics.get("facet_corrected_item_total_correlation")
        or statistics.get("corrected_item_total_correlation")
        or {}
    )
    return _number(payload.get("r")) if isinstance(payload, Mapping) else None


def _effective_option_count(statistics: Mapping[str, Any]) -> int | None:
    quality = statistics.get("quality_evaluation") or {}
    value = quality.get("effective_option_count") if isinstance(quality, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def item_requires_psychometric_diagnosis(statistics: Mapping[str, Any]) -> bool:
    """Return whether current deterministic evidence triggers LLM diagnosis."""

    quality = statistics.get("quality_evaluation") or {}
    if isinstance(quality, Mapping) and (
        "facet_citc" in quality or "virtual_target_specificity" in quality
    ):
        # Only the four matched-arm qualification gates place an item in the queue.
        return str(quality.get("recommendation") or "revise") != "retain"

    citc = _citc_value(statistics)
    difficulty = _number(statistics.get("difficulty"))
    effective_options = _effective_option_count(statistics)
    return (
        citc is None
        or citc < CITC_DIAGNOSIS_THRESHOLD
        or difficulty is None
        or difficulty < DIFFICULTY_LOWER_BOUND
        or difficulty > DIFFICULTY_UPPER_BOUND
        or effective_options is None
        or effective_options < MINIMUM_EFFECTIVE_OPTION_COUNT
    )


def _find_item(state: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    pools = [
        state.get("frozen_item_bank") or [],
        state.get("item_pool") or [],
        state.get("selected_items") or [],
    ]
    for pool in pools:
        for candidate in pool:
            if isinstance(candidate, Mapping) and str(candidate.get("item_id")) == item_id:
                return deepcopy(dict(candidate))
    raise ValueError(f"item_id cannot be resolved: {item_id}")


def _find_specification(state: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    cell_id = str(item.get("blueprint_cell_id") or "")
    for candidate in state.get("item_specifications") or []:
        if not isinstance(candidate, Mapping):
            continue
        if (
            str(candidate.get("specification_id") or "") == item_id
            or str(candidate.get("item_id") or "") == item_id
        ):
            return deepcopy(dict(candidate))
    blueprint = state.get("blueprint") or {}
    for slot in blueprint.get("slots") or []:
        if not isinstance(slot, Mapping):
            continue
        if str(slot.get("specification_id") or "") != item_id:
            continue
        for candidate in state.get("item_specifications") or []:
            if (
                isinstance(candidate, Mapping)
                and str(candidate.get("specification_id") or "")
                == str(slot.get("specification_id") or "")
            ):
                return deepcopy(dict(candidate))
    if cell_id:
        for candidate in state.get("item_specifications") or []:
            if (
                isinstance(candidate, Mapping)
                and str(candidate.get("blueprint_cell_id") or "") == cell_id
            ):
                return deepcopy(dict(candidate))
    raise ValueError(f"item specification cannot be resolved: {item_id}")


def _find_cell(state: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    cell_id = str(item.get("blueprint_cell_id") or "")
    for candidate in (state.get("blueprint") or {}).get("cells") or []:
        if isinstance(candidate, Mapping) and str(candidate.get("cell_id")) == cell_id:
            return deepcopy(dict(candidate))
    raise ValueError(f"blueprint cell cannot be resolved: {cell_id}")


def _find_facet_and_behavior(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
    cell: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    facet_id = _text(
        cell.get("facet_id")
        or cell.get("dimension_id")
        or item.get("target_dimension_id")
    )
    behavior_id = _text(
        cell.get("behavior_id")
        or cell.get("behavior_evidence_id")
        or specification.get("behavior_evidence_id")
    )
    profile = (state.get("blueprint") or {}).get("construct_profile_snapshot") or {}
    facet = next(
        (
            deepcopy(dict(candidate))
            for candidate in profile.get("facets") or []
            if isinstance(candidate, Mapping)
            and str(candidate.get("facet_id") or "") == facet_id
        ),
        None,
    )
    if facet is None:
        raise ValueError(f"facet cannot be resolved: {facet_id}")
    behavior = next(
        (
            deepcopy(dict(candidate))
            for candidate in facet.get("behavior_evidence") or []
            if isinstance(candidate, Mapping)
            and str(candidate.get("behavior_id") or "") == behavior_id
        ),
        None,
    )
    if behavior is None:
        raise ValueError(f"behavior evidence cannot be resolved: {behavior_id}")
    return facet, behavior


def _find_skeleton(state: Mapping[str, Any], specification: Mapping[str, Any]) -> dict[str, Any]:
    specification_id = _text(specification.get("specification_id"))
    candidates = state.get("item_skeletons") or {}
    if isinstance(candidates, Mapping):
        value = candidates.get(specification_id)
        if isinstance(value, Mapping):
            return deepcopy(dict(value))
    raise ValueError(f"item skeleton cannot be resolved: {specification_id}")


def _append_constraint(
    rows: list[dict[str, Any]],
    constraint_id: str,
    component: str,
    statement: Any,
) -> None:
    text = _text(statement)
    if text:
        rows.append(
            {
                "constraint_id": constraint_id,
                "component": component,
                "statement": text,
            }
        )


def _option_rates(statistics: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, float]:
    direct = statistics.get("option_selection_rates")
    if isinstance(direct, Mapping):
        return {
            str(option_id): float(rate)
            for option_id, rate in direct.items()
            if _number(rate) is not None
        }
    rows = statistics.get("option_statistics") or {}
    if isinstance(rows, Mapping):
        return {
            str(option_id): float(
                row.get("selection_rate", row.get("rate"))
            )
            for option_id, row in rows.items()
            if isinstance(row, Mapping)
            and _number(row.get("selection_rate", row.get("rate"))) is not None
        }
    return {
        str(option.get("option_id")): 0.0
        for option in item.get("response_options") or []
        if isinstance(option, Mapping) and option.get("option_id")
    }


def _latest_content_review(state: Mapping[str, Any], item_id: str) -> dict[str, Any] | None:
    direct = (state.get("item_reviews") or {}).get(item_id)
    if isinstance(direct, Mapping):
        return deepcopy(dict(direct))
    if isinstance(direct, list) and direct:
        for candidate in reversed(direct):
            if isinstance(candidate, Mapping):
                return deepcopy(dict(candidate))
    history = (state.get("item_history") or {}).get(item_id) or []
    for candidate in reversed(history):
        if isinstance(candidate, Mapping) and isinstance(candidate.get("review"), Mapping):
            return deepcopy(dict(candidate["review"]))
    return None


def build_construct_diagnosis_evidence(
    state: Mapping[str, Any],
    item_id: str,
    *,
    revision_round: int,
) -> dict[str, Any]:
    """Resolve a diagnosis packet from stable workflow references."""

    item = _find_item(state, item_id)
    specification = _find_specification(state, item)
    cell = _find_cell(state, item)
    facet, behavior = _find_facet_and_behavior(state, item, cell, specification)
    skeleton = _find_skeleton(state, specification)

    behavior_id = _text(behavior.get("behavior_id"))
    mechanism_id = _text(cell.get("mechanism_id") or specification.get("mechanism_id"))
    situation_id = _text(cell.get("situation_id") or specification.get("situation_id"))
    constraints: list[dict[str, Any]] = []
    _append_constraint(
        constraints,
        f"FACET_DEFINITION:{facet.get('facet_id')}",
        "facet",
        facet.get("definition"),
    )
    _append_constraint(constraints, f"BE_DEFINITION:{behavior_id}", "behavior_evidence", behavior.get("observable_behavior"))
    _append_constraint(constraints, f"BE_HIGH:{behavior_id}", "behavior_evidence", behavior.get("high_expression"))
    _append_constraint(constraints, f"BE_LOW:{behavior_id}", "behavior_evidence", behavior.get("low_expression"))
    _append_constraint(constraints, f"BE_BOUNDARY:{behavior_id}", "behavior_evidence", behavior.get("boundary_condition"))
    for source_field, label in (
        ("common_confounds", "FACET_CONFOUND"),
        ("inappropriate_contexts", "FACET_INAPPROPRIATE"),
        ("inappropriate_conditions", "FACET_INAPPROPRIATE"),
        ("forbidden_patterns", "FACET_FORBIDDEN"),
        ("option_design_rules", "FACET_OPTION_RULE"),
    ):
        for index, statement in enumerate(facet.get(source_field) or [], start=1):
            _append_constraint(constraints, f"{label}:{index}", "facet", statement)
    _append_constraint(
        constraints,
        f"MECHANISM:{mechanism_id}",
        "activation_mechanism",
        cell.get("activation_mechanism") or specification.get("activation_mechanism"),
    )
    situation_statement = " | ".join(
        filter(
            None,
            [
                _text(cell.get("domain") or specification.get("context_category")),
                _text(cell.get("actor_relation") or specification.get("social_context")),
                _text(cell.get("event_class") or specification.get("context_seed")),
            ],
        )
    )
    _append_constraint(constraints, f"SITUATION:{situation_id}", "situation", situation_statement)
    _append_constraint(
        constraints,
        f"SKELETON:TENSION:{specification.get('specification_id')}",
        "skeleton",
        skeleton.get("behavioral_tension") or specification.get("core_tension") or specification.get("core_behavioral_tension"),
    )
    option_structure = skeleton.get("option_structure") or specification.get("behavioral_anchors") or {}
    if isinstance(option_structure, Mapping):
        option_rows = [
            {"behavioral_level": level, "behavioral_tendency": statement}
            for level, statement in option_structure.items()
        ]
    else:
        option_rows = option_structure
    for row in option_rows or []:
        if not isinstance(row, Mapping):
            continue
        level = _text(row.get("behavioral_level"))
        statement = row.get("behavioral_tendency") or row.get("text")
        _append_constraint(constraints, f"SKELETON:OPTION:{level}", "skeleton", statement)

    statistics = deepcopy(dict((state.get("item_statistics") or {}).get(item_id) or {}))
    rates = _option_rates(statistics, item)
    quality = statistics.get("quality_evaluation") or {}
    gate_citc = (
        quality.get("facet_citc") or {}
        if isinstance(quality, Mapping)
        else {}
    )
    specificity = (
        quality.get("virtual_target_specificity") or {}
        if isinstance(quality, Mapping)
        else {}
    )
    observations = []
    if gate_citc or specificity:
        target = specificity.get("rho_target") or specificity.get("target_spearman") or {}
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        observations.extend(
            [
                {
                    "observation_id": "OBS:CITC",
                    "metric": "facet_citc",
                    "value": _number(gate_citc.get("r")),
                    "threshold": gate_citc.get("threshold"),
                    "role": "iteration_gate",
                },
                {
                    "observation_id": "OBS:TARGET_RHO",
                    "metric": "target_condition_score_spearman_rho",
                    "value": _number(target.get("rho")),
                    "threshold": specificity.get("target_rho_threshold"),
                    "role": "iteration_gate",
                    "correlation_method": "ordinary_spearman",
                    "conditioning_variable": "condition_id",
                },
                {
                    "observation_id": "OBS:SAME_DOMAIN_MAX_NON_TARGET_RHO",
                    "metric": "same_domain_non_target_facet_max_signed_rho",
                    "dimension_id": same_domain.get("dimension_id") or same_domain.get("largest_non_target_dimension_id"),
                    "facet_name": same_domain.get("facet_name") or same_domain.get("largest_non_target_facet_name"),
                    "domain_id": same_domain.get("domain_id") or same_domain.get("largest_non_target_domain_id"),
                    "group_id": same_domain.get("selected_non_target_group_id"),
                    "condition_id": same_domain.get("selected_non_target_condition_id"),
                    "value": _number(same_domain.get("max_non_target_rho") if same_domain.get("max_non_target_rho") is not None else same_domain.get("largest_non_target_rho")),
                    "signed_rho": _number(same_domain.get("max_non_target_rho") if same_domain.get("max_non_target_rho") is not None else same_domain.get("largest_non_target_rho")),
                    "non_target_spearman": deepcopy(same_domain.get("non_target_spearman") or []),
                    "role": "iteration_gate_component",
                    "correlation_method": "ordinary_spearman",
                    "conditioning_variable": "condition_id",
                },
                {
                    "observation_id": "OBS:SAME_DOMAIN_VTS",
                    "metric": "target_minus_max_signed_same_domain_rho",
                    "value": _number(same_domain.get("specificity_margin")),
                    "threshold": same_domain.get("margin_threshold"),
                    "role": "iteration_gate",
                },
                {
                    "observation_id": "OBS:CROSS_DOMAIN_MAX_NON_TARGET_RHO",
                    "metric": "cross_domain_non_target_facet_max_signed_rho",
                    "dimension_id": cross_domain.get("dimension_id") or cross_domain.get("largest_non_target_dimension_id"),
                    "facet_name": cross_domain.get("facet_name") or cross_domain.get("largest_non_target_facet_name"),
                    "domain_id": cross_domain.get("domain_id") or cross_domain.get("largest_non_target_domain_id"),
                    "group_id": cross_domain.get("selected_non_target_group_id"),
                    "condition_id": cross_domain.get("selected_non_target_condition_id"),
                    "value": _number(cross_domain.get("max_non_target_rho") if cross_domain.get("max_non_target_rho") is not None else cross_domain.get("largest_non_target_rho")),
                    "signed_rho": _number(cross_domain.get("max_non_target_rho") if cross_domain.get("max_non_target_rho") is not None else cross_domain.get("largest_non_target_rho")),
                    "non_target_spearman": deepcopy(cross_domain.get("non_target_spearman") or []),
                    "role": "iteration_gate_component",
                    "correlation_method": "ordinary_spearman",
                    "conditioning_variable": "condition_id",
                },
                {
                    "observation_id": "OBS:CROSS_DOMAIN_VTS",
                    "metric": "target_minus_max_signed_cross_domain_rho",
                    "value": _number(cross_domain.get("specificity_margin")),
                    "threshold": cross_domain.get("margin_threshold"),
                    "role": "iteration_gate",
                },
            ]
        )
        for group_name, group in (
            ("SAME_DOMAIN", same_domain),
            ("CROSS_DOMAIN", cross_domain),
        ):
            if group.get("passes") is not False:
                continue
            competitor = group.get("largest_non_target_facet") or group
            if not isinstance(competitor, Mapping):
                continue
            competitor_id = _text(competitor.get("dimension_id"))
            for suffix, field in (
                ("DEFINITION", "definition"),
                ("HIGH", "high_behavior"),
                ("LOW", "low_behavior"),
            ):
                _append_constraint(
                    constraints,
                    f"NON_TARGET_{group_name}:{competitor_id}:{suffix}",
                    "construct",
                    competitor.get(field),
                )
    observations.extend(
        [
            *([] if gate_citc or specificity else [{
                "observation_id": "OBS:CITC",
                "metric": "facet_citc",
                "value": _citc_value(statistics),
                "role": "legacy_gate",
            }]),
            {
                "observation_id": "OBS:DIFFICULTY",
                "metric": "standardized_mean_score",
                "value": _number(statistics.get("difficulty")),
                "role": "descriptive_only" if gate_citc or specificity else "legacy_gate",
            },
            {
                "observation_id": "OBS:EFFECTIVE_OPTION_COUNT",
                "metric": "effective_option_count",
                "value": _effective_option_count(statistics),
                "role": "descriptive_only" if gate_citc or specificity else "legacy_gate",
            },
        ]
    )
    observations.extend(
        {
            "observation_id": f"OBS:OPTION_RATE:{option_id}",
            "metric": "option_selection_rate",
            "option_id": option_id,
            "value": rate,
        }
        for option_id, rate in sorted(rates.items())
    )
    gradient = statistics.get("target_option_gradient") or {}
    if isinstance(gradient, Mapping):
        observations.append({
            "observation_id": "OBS:TARGET_OPTION_GRADIENT",
            "metric": "target_option_mean_gradient",
            "estimable": gradient.get("estimable"),
            "passes": gradient.get("passes"),
            "failed_adjacent_pairs": deepcopy(gradient.get("failed_adjacent_pairs") or []),
            "role": "repair_trigger",
            "filtering_authority": False,
        })
    observations.append(
        {
            "observation_id": _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID,
            "metric": "target_facet_construct_gradient_preflight",
            "role": "mandatory_repair_preflight",
            "passes": (
                gradient.get("passes")
                if isinstance(gradient, Mapping)
                else None
            ),
            "filtering_authority": False,
        }
    )
    option_evidence = []
    aggregate_option_statistics = (
        (statistics.get("option_choice_diagnostics") or {}).get("aggregate")
        or (statistics.get("option_choice_diagnostics") or {}).get("all")
        or {}
    )
    aggregate_by_option = {
        str(row.get("option_id")): row
        for row in aggregate_option_statistics.get("options") or []
        if isinstance(row, Mapping)
    }
    for option in item.get("response_options") or []:
        if not isinstance(option, Mapping):
            continue
        option_id = _text(option.get("option_id"))
        option_evidence.append(
            {
                "option_id": option_id,
                "text": option.get("text"),
                "behavioral_level": option.get("behavioral_level"),
                "score": (item.get("scoring_key") or {}).get(option_id),
                "selection_count": (
                    aggregate_by_option.get(option_id) or {}
                ).get("selection_count"),
                "selection_rate": rates.get(option_id),
                "facet_mean": (
                    aggregate_by_option.get(option_id) or {}
                ).get("facet_mean"),
                "facet_standard_error": (
                    aggregate_by_option.get(option_id) or {}
                ).get("facet_standard_error"),
            }
        )
    raw_choice_diagnostics = statistics.get("option_choice_diagnostics") or {}
    qualification = statistics.get("qualification") or {}
    allowed_condition_ids = {"target"}
    failed_vts_arms = set()
    if qualification.get("same_domain_vts_pass") is not True:
        failed_vts_arms.add("same_domain")
    if qualification.get("cross_domain_vts_pass") is not True:
        failed_vts_arms.add("cross_domain")
    vts_option_choice_diagnostics = {
        "version": raw_choice_diagnostics.get("version"),
        "filtering_authority": False,
        "diagnostic_use": "localization_only",
        "interpretation": (
            "选择频率和匹配实验臂差异只用于定位需要核查的选项；VTS返修要求"
            "引用题面或选项原文，并说明该措辞直接表达所给污染facet的定义、"
            "高低行为边界，或使高分行为依赖该污染构念。污染不必与目标构念矛盾。"
            "若出现严格非目标组均值梯度观察，则按强制端点返修规则处理。"
        ),
        "aggregate": deepcopy(raw_choice_diagnostics.get("aggregate") or raw_choice_diagnostics.get("all") or {}),
        "by_condition": [
            deepcopy(dict(row))
            for row in raw_choice_diagnostics.get("by_condition") or []
            if isinstance(row, Mapping)
            and (
                row.get("condition_id") in allowed_condition_ids
                or row.get("arm_id") in failed_vts_arms
                or str(row.get("condition_id") or "").split("__", 1)[0] in failed_vts_arms
            )
        ],
        "target_option_gradient": deepcopy(gradient),
        "arm_difference_diagnostics": {
            "filtering_authority": False,
            "diagnostic_use": "localization_only",
            "comparisons": [
                deepcopy(dict(row))
                for row in (
                    raw_choice_diagnostics.get("arm_difference_diagnostics")
                    or {}
                ).get("comparisons") or []
                if isinstance(row, Mapping)
                and (
                    row.get("comparator_condition_id") in allowed_condition_ids
                    or row.get("comparator_arm_id") in failed_vts_arms
                    or str(row.get("comparator_condition_id") or "").split("__", 1)[0] in failed_vts_arms
                )
            ],
        },
        "option_score_comparisons": deepcopy(
            raw_choice_diagnostics.get("option_score_comparisons") or []
        ),
    }
    evidence_for_forced_gradient = {
        "observations": observations,
        "option_choice_diagnostics": vts_option_choice_diagnostics,
    }
    forced_vts_gradient_repairs = derive_forced_vts_gradient_repairs(
        evidence_for_forced_gradient
    )
    for repair in forced_vts_gradient_repairs:
        observations.append(
            {
                "observation_id": repair["observation_id"],
                "metric": f"{repair['vts_category']}_non_target_option_mean_gradient",
                "role": "forced_repair_trigger",
                "filtering_authority": False,
                "vts_category": repair["vts_category"],
                "vts_observation_id": repair["vts_observation_id"],
                "option_ids_by_score": deepcopy(repair["option_ids_by_score"]),
                "endpoint_option_ids": deepcopy(repair["endpoint_option_ids"]),
                "target_means_by_score": deepcopy(repair["target_means_by_score"]),
                "non_target_means_by_score": deepcopy(
                    repair["non_target_means_by_score"]
                ),
            }
        )
    prior_repairs = [
        deepcopy(dict(row))
        for row in state.get("psychometric_repair_history") or []
        if (
            isinstance(row, Mapping)
            and str(row.get("item_id")) == item_id
            and row.get("event") == "psychometric_item_repaired"
        )
    ]
    return {
        "item_id": item_id,
        "item_version": item.get("version"),
        "revision_round": revision_round,
        "blueprint_refs": {
            "blueprint_cell_id": item.get("blueprint_cell_id"),
            "facet_id": facet.get("facet_id"),
            "behavior_evidence_id": behavior_id,
            "mechanism_id": mechanism_id,
            "situation_id": situation_id,
            "specification_id": specification.get("specification_id"),
        },
        "normal_constraints": constraints,
        "target_construct_constraints": [
            deepcopy(dict(row))
            for row in constraints
            if isinstance(row, Mapping)
            and row.get("component") in {"facet", "behavior_evidence"}
        ],
        "observations": observations,
        "current_item": item,
        "option_evidence": option_evidence,
        "option_choice_diagnostics": vts_option_choice_diagnostics,
        "forced_vts_gradient_repairs": forced_vts_gradient_repairs,
        "latest_content_review": _latest_content_review(state, item_id),
        "prior_atomic_repairs": prior_repairs,
    }


def build_psychometric_agent_input(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact evidence packet shown to psychometric agents.

    Workflow identifiers and raw selection counts are routing/audit metadata,
    not diagnosis evidence.  Keep the item wording and construct constraints,
    while exposing only the failed VTS-aligned option mean comparisons.
    """

    observations = []
    failed_vts = set()
    for row in evidence.get("observations") or []:
        if not isinstance(row, Mapping):
            continue
        observation_id = str(row.get("observation_id") or "")
        if observation_id in _VTS_OBSERVATION_IDS:
            threshold = _number(row.get("threshold"))
            value = _number(row.get("value"))
            # An unestimable required VTS is a failed gate as well.  Keep the
            # paired comparison context available so the diagnosis agent can
            # see the same evidence in both numeric and missing-value cases.
            if value is None or (threshold is not None and value < threshold):
                failed_vts.add(
                    "same_domain" if observation_id.startswith("OBS:SAME") else "cross_domain"
                )
        if observation_id in {
            "OBS:CITC",
            "OBS:TARGET_RHO",
            "OBS:TARGET_OPTION_GRADIENT",
            _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID,
            *_FORCED_VTS_GRADIENT_OBSERVATION_IDS,
        }:
            observations.append(_semantic_observation(row))
        elif observation_id in {
            "OBS:SAME_DOMAIN_MAX_NON_TARGET_RHO",
            "OBS:SAME_DOMAIN_VTS",
            "OBS:CROSS_DOMAIN_MAX_NON_TARGET_RHO",
            "OBS:CROSS_DOMAIN_VTS",
        }:
            observations.append(_semantic_observation(row))

    raw_diagnostics = evidence.get("option_choice_diagnostics") or {}
    paired = raw_diagnostics.get("option_score_comparisons") or []
    comparison_rows: list[dict[str, Any]] = []
    for category in ("same_domain", "cross_domain"):
        if category not in failed_vts:
            continue
        for row in paired:
            if not isinstance(row, Mapping):
                continue
            comparison_rows.append(
                {
                    "vts_category": category,
                    "option_id": row.get("option_id"),
                    "score": row.get("score"),
                    "target_mean_score": row.get("target_mean_score"),
                    f"{category}_mean_score": row.get(f"{category}_mean_score"),
                }
            )

    current_item = evidence.get("current_item") or {}
    compact_options = [
        {
            "option_id": row.get("option_id"),
            "text": row.get("text"),
            "behavioral_level": row.get("behavioral_level"),
            "score": (current_item.get("scoring_key") or {}).get(row.get("option_id")),
        }
        for row in current_item.get("response_options") or []
        if isinstance(row, Mapping)
    ]
    item_content = {
        "scenario": current_item.get("scenario"),
        "response_instruction": current_item.get("response_instruction"),
        "response_options": [
            {
                "option_id": row.get("option_id"),
                "text": row.get("text"),
                "behavioral_level": row.get("behavioral_level"),
                "score": row.get("score"),
            }
            for row in compact_options
        ],
        "scoring_key": deepcopy(current_item.get("scoring_key") or {}),
    }
    core_constraint_prefixes = (
        "FACET_DEFINITION:",
        "BE_DEFINITION:",
        "BE_HIGH:",
        "BE_LOW:",
        "BE_BOUNDARY:",
        "MECHANISM:",
        "SITUATION:",
        "SKELETON:TENSION:",
        "SKELETON:OPTION:",
    )
    allowed_constraint_prefixes = set(core_constraint_prefixes)
    if "same_domain" in failed_vts:
        allowed_constraint_prefixes.add("NON_TARGET_SAME_DOMAIN:")
    if "cross_domain" in failed_vts:
        allowed_constraint_prefixes.add("NON_TARGET_CROSS_DOMAIN:")
    compact_constraints = [
        _semantic_constraint(row)
        for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping)
        and any(
            str(row.get("constraint_id") or "").startswith(prefix)
            for prefix in allowed_constraint_prefixes
        )
    ]
    forced_vts_gradient_repairs = [
        {
            key: deepcopy(row[key])
            for key in (
                "vts_category", "observation_id", "vts_observation_id",
                "option_ids_by_score", "endpoint_option_ids",
                "target_means_by_score", "non_target_means_by_score",
            )
            if row.get(key) is not None
        }
        for row in derive_forced_vts_gradient_repairs(evidence)
        if isinstance(row, Mapping)
    ]
    target_construct_constraints = [
        _semantic_constraint(row)
        for row in (
            evidence.get("target_construct_constraints")
            or evidence.get("normal_constraints")
            or []
        )
        if isinstance(row, Mapping)
        and row.get("component") in {"facet", "behavior_evidence"}
    ]
    target_option_gradient = _semantic_target_gradient(
        raw_diagnostics.get("target_option_gradient")
        or evidence.get("target_option_gradient")
        or {}
    )
    ordered_rows = _ordered_target_option_rows({**evidence, "option_evidence": compact_options})
    target_gradient_plan = {
        "mandatory_first_task": True,
        "option_order": [
            {
                "option_id": row["option_id"],
                "score": row["score"],
                "behavioral_level": row.get("behavioral_level"),
            }
            for row in ordered_rows
        ],
        "failed_adjacent_pairs": deepcopy(target_option_gradient.get("failed_adjacent_pairs") or []),
    }
    return {
        "normal_constraints": compact_constraints,
        "target_construct_constraints": target_construct_constraints,
        "observations": observations,
        "item_content": item_content,
        "option_evidence": compact_options,
        "option_score_comparisons": comparison_rows,
        "forced_vts_gradient_repairs": forced_vts_gradient_repairs,
        "target_option_gradient": target_option_gradient,
        "target_gradient_plan": target_gradient_plan,
        "filtering_authority": False,
    }


def diagnosis_fingerprint(evidence: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(evidence))
    payload.pop("diagnosis_fingerprint", None)
    payload.pop("revision_round", None)
    payload.pop("prior_atomic_repairs", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _validate_refs(
    refs: Any,
    valid: set[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(refs, list) or not all(isinstance(value, str) for value in refs):
        raise ValueError(f"{label} refs must be a list of strings")
    if not refs and not allow_empty:
        raise ValueError(f"{label} refs cannot be empty")
    unknown = set(refs) - valid
    if unknown:
        raise ValueError(f"unknown {label} reference: {sorted(unknown)}")
    return refs


def _validate_atomic_edit(
    atomic_edit: Any,
    *,
    selected: Mapping[str, Any],
    evidence: Mapping[str, Any],
    allow_forced_vts_endpoints: bool = False,
) -> None:
    """Validate one edit while keeping the expert level and scoring immutable."""

    if not isinstance(atomic_edit, Mapping) or set(atomic_edit) != {
        "target_field", "option_ids", "problem", "instruction"
    }:
        raise ValueError("repair task requires one atomic_edit")
    components = selected.get("suspect_components") or []
    if len(components) != 1 or components[0] not in _EDITABLE_COMPONENTS:
        raise ValueError("repair task must point to one editable component")
    target_field = atomic_edit.get("target_field")
    if target_field != components[0]:
        raise ValueError("atomic edit target must match selected diagnosis")
    option_ids = atomic_edit.get("option_ids")
    if not isinstance(option_ids, list) or not all(
        isinstance(value, str) for value in option_ids
    ):
        raise ValueError("atomic edit option_ids are invalid")
    valid_option_ids = {
        str(row.get("option_id"))
        for row in evidence.get("option_evidence") or []
        if isinstance(row, Mapping)
    }
    if target_field == "scenario" and option_ids:
        raise ValueError("scenario edit cannot include option_ids")
    if target_field == "scenario" and selected.get("affected_option_ids"):
        raise ValueError("scenario diagnosis cannot include affected_option_ids")
    if target_field == "response_options":
        if not 1 <= len(option_ids) <= 2 or not set(option_ids).issubset(valid_option_ids):
            raise ValueError("response_options edit must name 1-2 valid options")
        diagnosed_option_ids = set(selected.get("affected_option_ids") or [])
        # One diagnosis may cover several failed adjacent pairs, while each
        # executable edit must stay minimal.  The task therefore narrows the
        # diagnosed scope; it must never introduce an undiagnosed option.
        if not set(option_ids).issubset(diagnosed_option_ids):
            raise ValueError("atomic edit options must stay within selected diagnosis")
        if len(option_ids) == 2 and not allow_forced_vts_endpoints:
            scores = {
                str(row.get("option_id")): _number(row.get("score"))
                for row in evidence.get("option_evidence") or []
                if isinstance(row, Mapping)
            }
            score_ranks = [scores.get(option_id) for option_id in option_ids]
            if all(value is not None for value in score_ranks):
                if abs(float(score_ranks[0]) - float(score_ranks[1])) != 1:
                    raise ValueError("two option edits must target adjacent behavioral levels")
            else:
                levels = {
                    str(row.get("option_id")): str(row.get("behavioral_level"))
                    for row in evidence.get("option_evidence") or []
                    if isinstance(row, Mapping)
                }
                ranks = [_LEVEL_RANK.get(levels.get(option_id, "")) for option_id in option_ids]
                if None in ranks or abs(ranks[0] - ranks[1]) != 1:
                    raise ValueError("two option edits must target adjacent behavioral levels")
        if allow_forced_vts_endpoints:
            forced_endpoint_ids = {
                str(option_id)
                for repair in derive_forced_vts_gradient_repairs(evidence)
                for option_id in repair.get("endpoint_option_ids") or []
            }
            if set(option_ids) != forced_endpoint_ids:
                raise ValueError(
                    "forced VTS gradient edit must target exactly the score-1 and score-4 options"
                )
    if not _text(atomic_edit.get("problem")) or not _text(atomic_edit.get("instruction")):
        raise ValueError("atomic edit problem and instruction cannot be empty")


def validate_atomic_repair_advice(
    advice: Any,
    evidence: Mapping[str, Any],
    *,
    require_target_gradient_task: bool = False,
) -> None:
    """Validate all confirmed, evidence-backed edits for one item.

    ``repair_tasks`` is the current contract.  The former
    ``selected_diagnosis_id``/``atomic_edit`` pair remains accepted so older
    checkpoints and test fixtures can be resumed safely.
    """

    expected = {
        "item_id",
        "decision",
        "observed_discrepancies",
        "candidate_diagnoses",
        "summary",
    }
    legacy_fields = {"selected_diagnosis_id", "atomic_edit"}
    current_fields = {"repair_tasks"}
    if not isinstance(advice, dict):
        raise ValueError("AtomicRepairAdvice fields are invalid")
    fields = set(advice)
    if "repair_tasks" in fields:
        # ``repair_tasks`` is authoritative for current output. Older schemas
        # exposed the two legacy fields as independent optional properties, so
        # a model could legally echo only one of them. Accept either legacy
        # property as redundant input instead of rejecting an otherwise valid
        # current diagnosis.
        required_fields = expected | current_fields
        allowed_fields = required_fields | legacy_fields
        fields_are_valid = required_fields.issubset(fields) and fields.issubset(
            allowed_fields
        )
    else:
        # Persisted legacy checkpoints remain valid only when the old pair is
        # complete; a partial legacy edit has no safe executable meaning.
        fields_are_valid = fields == expected | legacy_fields
    if not fields_are_valid:
        raise ValueError("AtomicRepairAdvice fields are invalid")
    if str(advice.get("item_id")) != str(evidence.get("item_id")):
        raise ValueError("AtomicRepairAdvice item_id does not match evidence")
    decision = advice.get("decision")
    if decision not in {"repair", "defer"}:
        raise ValueError("AtomicRepairAdvice decision must be repair or defer")
    valid_observations = {
        str(row.get("observation_id"))
        for row in evidence.get("observations") or []
        if isinstance(row, Mapping)
    }
    valid_observations.update(_forced_vts_gradient_repair_ids(evidence))
    valid_observations.add(_TARGET_GRADIENT_REQUIRED_OBSERVATION_ID)
    valid_constraints = {
        str(row.get("constraint_id"))
        for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping)
    }
    valid_option_ids = {
        str(row.get("option_id"))
        for row in evidence.get("option_evidence") or []
        if isinstance(row, Mapping)
    }
    gradient_failed = any(
        isinstance(row, Mapping)
        and row.get("observation_id") == "OBS:TARGET_OPTION_GRADIENT"
        and row.get("passes") is not True
        for row in evidence.get("observations") or []
    )
    forced_vts_repairs = derive_forced_vts_gradient_repairs(evidence)
    forced_vts_gradient = bool(forced_vts_repairs)
    forced_vts_observation_ids = {
        str(row.get("observation_id"))
        for row in forced_vts_repairs
        if row.get("observation_id")
    }
    forced_vts_endpoint_ids = {
        str(option_id)
        for row in forced_vts_repairs
        for option_id in row.get("endpoint_option_ids") or []
    }
    discrepancies = advice.get("observed_discrepancies")
    if not isinstance(discrepancies, list) or not discrepancies:
        raise ValueError("observed_discrepancies cannot be empty")
    for row in discrepancies:
        if not isinstance(row, Mapping) or set(row) != {
            "observation_refs", "constraint_refs", "description"
        }:
            raise ValueError("observed discrepancy fields are invalid")
        _validate_refs(row.get("observation_refs"), valid_observations, label="observation")
        _validate_refs(row.get("constraint_refs"), valid_constraints, label="constraint", allow_empty=True)
        if not _text(row.get("description")):
            raise ValueError("observed discrepancy description cannot be empty")
    candidates = advice.get("candidate_diagnoses")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("candidate_diagnoses must contain 1-3 diagnoses")
    candidate_ids: set[str] = set()
    for row in candidates:
        if not isinstance(row, Mapping) or set(row) != {
            "diagnosis_id", "suspect_components", "affected_option_ids",
            "observation_refs", "constraint_refs", "textual_evidence",
            "explanation", "confidence",
        }:
            raise ValueError("candidate diagnosis fields are invalid")
        diagnosis_id = _text(row.get("diagnosis_id"))
        if not diagnosis_id or diagnosis_id in candidate_ids:
            raise ValueError("diagnosis_id must be non-empty and unique")
        candidate_ids.add(diagnosis_id)
        components = row.get("suspect_components")
        if (
            not isinstance(components, list)
            or not components
            or not all(component in _ALL_COMPONENTS for component in components)
        ):
            raise ValueError("suspect_components contain an invalid component")
        option_ids = row.get("affected_option_ids")
        if not isinstance(option_ids, list) or not all(isinstance(value, str) for value in option_ids):
            raise ValueError("affected_option_ids must be a list of strings")
        if not set(option_ids).issubset(valid_option_ids):
            raise ValueError("candidate diagnosis points to an unknown option")
        if "response_options" not in components and option_ids:
            raise ValueError("non-option diagnosis cannot include affected_option_ids")
        _validate_refs(row.get("observation_refs"), valid_observations, label="observation")
        _validate_refs(row.get("constraint_refs"), valid_constraints, label="constraint")
        if row.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError("candidate confidence is invalid")
        if not _text(row.get("explanation")):
            raise ValueError("candidate explanation cannot be empty")
    repair_tasks = advice.get("repair_tasks")
    if repair_tasks is None:
        selected_id = advice.get("selected_diagnosis_id")
        atomic_edit = advice.get("atomic_edit")
        if decision == "defer":
            if forced_vts_gradient:
                raise ValueError(
                    "failed VTS option-mean gradient requires repair"
                )
            if selected_id is not None or atomic_edit is not None:
                raise ValueError("defer cannot select a diagnosis or atomic edit")
            return
        if not isinstance(selected_id, str) or selected_id not in candidate_ids:
            raise ValueError("repair must select a candidate diagnosis")
        selected = next(row for row in candidates if row.get("diagnosis_id") == selected_id)
        if selected.get("confidence") not in {"medium", "high"}:
            raise ValueError("selected diagnosis confidence must be medium or high")
        if not _text(selected.get("textual_evidence")) and not _textual_evidence_may_be_empty(
            selected,
            gradient_failed=gradient_failed,
            forced_vts_gradient=forced_vts_gradient,
        ):
            raise ValueError("selected repair diagnosis requires concrete textual evidence")
        if not _repair_has_allowed_text_evidence(
            selected,
            evidence,
            gradient_failed=gradient_failed,
            forced_vts_gradient=forced_vts_gradient,
        ):
            if forced_vts_gradient:
                gaps = _forced_vts_contract_gaps(selected, evidence)
                raise ValueError(
                    "forced VTS repair contract invalid: "
                    + ("; ".join(gaps) or "missing deterministic evidence links")
                )
            raise ValueError(
                "selected repair diagnosis must quote current item wording; "
                "VTS repair may instead link that quote directly to a supplied "
                "NON_TARGET contaminant constraint"
            )
        _validate_atomic_edit(
            atomic_edit,
            selected=selected,
            evidence=evidence,
            allow_forced_vts_endpoints=forced_vts_gradient,
        )
        if forced_vts_gradient and set(atomic_edit.get("option_ids") or []) != forced_vts_endpoint_ids:
            raise ValueError(
                "forced VTS gradient edit must include both score endpoints"
            )
        return

    if not isinstance(repair_tasks, list) or len(repair_tasks) > 4:
        raise ValueError("repair_tasks must contain 0-4 tasks")
    if decision == "defer" and repair_tasks:
        raise ValueError("defer cannot contain repair_tasks")
    # 放宽：梯度类问题（目标组/forced VTS 选项均分）允许模型 defer，
    # 进入人工处置队列，避免确定性兜底死锁导致整个诊断批停止。
    if decision == "repair" and not repair_tasks:
        raise ValueError("repair requires at least one repair_task")
    citc_observation = next(
        (
            row
            for row in evidence.get("observations") or []
            if isinstance(row, Mapping) and row.get("observation_id") == "OBS:CITC"
        ),
        {},
    )
    citc_value = _number(citc_observation.get("value"))
    if (
        citc_value is not None
        and 0.0 <= citc_value < CITC_DIAGNOSIS_THRESHOLD
        and any(
            isinstance(task, Mapping)
            and isinstance(task.get("atomic_edit"), Mapping)
            and task["atomic_edit"].get("target_field") == "scenario"
            for task in repair_tasks
        )
    ):
        raise ValueError("0<=CITC<.20 must preserve the scenario")
    seen_scopes: set[tuple[str, tuple[str, ...]]] = set()
    seen_option_ids: set[str] = set()
    forced_endpoint_task_found = False
    for task_index, task in enumerate(repair_tasks):
        if not isinstance(task, Mapping) or not set(task).issubset(
            {"diagnosis_id", "atomic_edit", "phase"}
        ) or not {"diagnosis_id", "atomic_edit"}.issubset(set(task)):
            raise ValueError("repair task fields are invalid")
        phase = task.get("phase", "other")
        if phase not in {"target_facet_gradient", "other"}:
            raise ValueError("repair task phase is invalid")
        diagnosis_id = _text(task.get("diagnosis_id"))
        if diagnosis_id not in candidate_ids:
            raise ValueError("repair task references unknown diagnosis")
        selected = next(row for row in candidates if row.get("diagnosis_id") == diagnosis_id)
        if selected.get("confidence") not in {"medium", "high"}:
            raise ValueError("repair task confidence must be medium or high")
        if not _text(selected.get("textual_evidence")) and not _textual_evidence_may_be_empty(
            selected,
            gradient_failed=gradient_failed,
            forced_vts_gradient=forced_vts_gradient,
        ):
            raise ValueError("repair task requires concrete textual evidence")
        if not _repair_has_allowed_text_evidence(
            selected,
            evidence,
            gradient_failed=gradient_failed,
            forced_vts_gradient=forced_vts_gradient,
        ):
            if forced_vts_gradient:
                gaps = _forced_vts_contract_gaps(selected, evidence)
                raise ValueError(
                    "forced VTS repair contract invalid: "
                    + ("; ".join(gaps) or "missing deterministic evidence links")
                )
            raise ValueError(
                "repair task must quote current item wording; VTS repair must "
                "link the quote to a supplied NON_TARGET contaminant constraint"
            )
        atomic_edit = task.get("atomic_edit")
        if phase == "target_facet_gradient":
            if task_index != 0:
                raise ValueError("目标facet梯度任务必须首先执行")
            if atomic_edit.get("target_field") != "response_options":
                raise ValueError("目标facet梯度任务只能修改response_options")
            if (
                require_target_gradient_task
                and _TARGET_GRADIENT_REQUIRED_OBSERVATION_ID
                not in {str(value) for value in selected.get("observation_refs") or []}
                and "OBS:TARGET_OPTION_GRADIENT"
                not in {str(value) for value in selected.get("observation_refs") or []}
            ):
                raise ValueError("目标facet梯度任务必须引用目标梯度观察")
        elif require_target_gradient_task and task_index == 0:
            raise ValueError("repair首任务必须是目标facet梯度任务")
        atomic_option_ids = {
            str(option_id) for option_id in atomic_edit.get("option_ids") or []
        }
        is_forced_endpoint_task = (
            forced_vts_gradient
            and atomic_edit.get("target_field") == "response_options"
            and atomic_option_ids == forced_vts_endpoint_ids
            and forced_vts_observation_ids
            & {
                str(value) for value in selected.get("observation_refs") or []
            }
        )
        _validate_atomic_edit(
            atomic_edit,
            selected=selected,
            evidence=evidence,
            allow_forced_vts_endpoints=bool(is_forced_endpoint_task),
        )
        if forced_vts_gradient and is_forced_endpoint_task:
            forced_endpoint_task_found = True
        scope = (
            str(atomic_edit.get("target_field")),
            tuple(sorted(str(option_id) for option_id in atomic_edit.get("option_ids") or [])),
        )
        if scope in seen_scopes:
            raise ValueError("repair_tasks contain duplicate edit scopes")
        seen_scopes.add(scope)
        if atomic_edit.get("target_field") == "response_options":
            option_ids = set(atomic_edit.get("option_ids") or [])
            if seen_option_ids & option_ids:
                raise ValueError("repair_tasks contain overlapping option edit scopes")
            seen_option_ids.update(option_ids)
    if forced_vts_gradient and not forced_endpoint_task_found:
        raise ValueError(
            "forced VTS gradient repair must include one task for score-1 and score-4 options"
        )
    if require_target_gradient_task and decision == "repair":
        if repair_tasks[0].get("phase") != "target_facet_gradient":
            raise ValueError("repair首任务必须是目标facet梯度任务")
        if sum(task.get("phase") == "target_facet_gradient" for task in repair_tasks) != 1:
            raise ValueError("repair只能包含一个目标facet梯度首任务")


def repair_tasks_from_advice(advice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the confirmed edit tasks, including legacy single-edit advice."""

    tasks = advice.get("repair_tasks")
    if isinstance(tasks, list):
        ordered = [deepcopy(dict(task)) for task in tasks if isinstance(task, Mapping)]
        return [row for _, row in sorted(
            enumerate(ordered),
            key=lambda pair: (0 if pair[1].get("phase") == "target_facet_gradient" else 1, pair[0]),
        )]
    selected_id = advice.get("selected_diagnosis_id")
    atomic_edit = advice.get("atomic_edit")
    if isinstance(selected_id, str) and isinstance(atomic_edit, Mapping):
        return [
            {
                "diagnosis_id": selected_id,
                "atomic_edit": deepcopy(dict(atomic_edit)),
            }
        ]
    return []


def validate_atomic_item_patch(
    patch: Any,
    current_item: Mapping[str, Any],
    advice: Mapping[str, Any],
) -> None:
    """Reject a patch that changes anything outside the selected atomic scope."""

    required = {"scenario_update", "option_updates"}
    allowed = {*required, "change_summary"}
    if (
        not isinstance(patch, Mapping)
        or not required.issubset(patch)
        or not set(patch).issubset(allowed)
    ):
        raise ValueError("item patch fields are invalid")
    atomic_edit = advice.get("atomic_edit") or {}
    target_field = atomic_edit.get("target_field")
    scenario_update = patch.get("scenario_update")
    option_updates = patch.get("option_updates")
    if not isinstance(option_updates, list):
        raise ValueError("option_updates must be a list")
    if scenario_update is not None and option_updates:
        raise ValueError("an atomic patch may change only scenario or options, not both")
    if target_field == "scenario":
        if not isinstance(scenario_update, str) or not scenario_update.strip():
            raise ValueError("scenario patch must change the scenario")
        if scenario_update.strip() == _text(current_item.get("scenario")):
            raise ValueError("scenario patch did not change the selected field")
        if option_updates:
            raise ValueError("scenario patch may change only the scenario")
        return
    if target_field != "response_options":
        raise ValueError("atomic patch target is not editable")
    if scenario_update is not None:
        raise ValueError("option patch may change only selected options")
    expected_ids = set(atomic_edit.get("option_ids") or [])
    update_ids = {
        str(row.get("option_id"))
        for row in option_updates
        if isinstance(row, Mapping)
    }
    if update_ids != expected_ids or len(option_updates) != len(expected_ids):
        raise ValueError("option patch must change only all selected options")
    current = {
        str(row.get("option_id")): _text(row.get("text"))
        for row in current_item.get("response_options") or []
        if isinstance(row, Mapping)
    }
    for row in option_updates:
        if not isinstance(row, Mapping) or set(row) != {"option_id", "text"}:
            raise ValueError("option update fields are invalid")
        option_id = str(row.get("option_id"))
        new_text = _text(row.get("text"))
        if not new_text or new_text == current.get(option_id):
            raise ValueError("option patch did not change every selected option")


def normalize_atomic_option_patch_scope(
    patch: Any,
    advice: Mapping[str, Any],
) -> Any:
    """Trim extra option edits while preserving the diagnosed atomic scope.

    Repair models occasionally return a complete option set even when the
    diagnosis selected one option or one adjacent pair.  Extra edits are not
    allowed to reach the item, but they can be safely discarded before strict
    validation because the selected option texts are still usable.  Missing
    selected options, duplicate selected options, and scenario edits remain
    invalid and are left for the normal validator to reject.
    """

    if not isinstance(patch, Mapping):
        return patch
    atomic_edit = advice.get("atomic_edit")
    if not isinstance(atomic_edit, Mapping):
        return patch
    if atomic_edit.get("target_field") != "response_options":
        return patch
    expected_ids = [
        str(option_id)
        for option_id in atomic_edit.get("option_ids") or []
        if option_id is not None
    ]
    option_updates = patch.get("option_updates")
    if not expected_ids or not isinstance(option_updates, list):
        return patch

    selected_updates: dict[str, dict[str, Any]] = {}
    for row in option_updates:
        if not isinstance(row, Mapping) or row.get("option_id") is None:
            continue
        option_id = str(row.get("option_id"))
        if option_id not in expected_ids:
            continue
        if option_id in selected_updates:
            # Keep duplicate outputs invalid rather than silently selecting one.
            return patch
        selected_updates[option_id] = {
            "option_id": option_id,
            "text": row.get("text"),
        }

    if set(selected_updates) != set(expected_ids):
        return patch

    # A scenario edit is outside this task's scope and must still fail.  Only
    # extra option rows are safely discarded here.
    if patch.get("scenario_update") is not None:
        return patch

    normalized = dict(patch)
    normalized["option_updates"] = [selected_updates[option_id] for option_id in expected_ids]
    return normalized
