"""Test-level review, rescoring, and completion gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from sjt_system.authoring.bank import item_bank_snapshot_is_current
from sjt_system.authoring.context import build_item_pattern_profile
from sjt_system.authoring.items import (
    derive_scoring_key_from_behavioral_levels,
)
from sjt_system.evaluation.psychometrics import (
    MEASUREMENT_EVALUATION_VERSION,
    PSYCHOMETRIC_FORMULA_VERSION,
)
from sjt_system.state import PSJTState
from sjt_system.runtime.trace import utc_timestamp


_PUBLIC_FORBIDDEN_KEYS = {
    "scoring_key",
    "target_dimension_id",
    "blueprint_cell_id",
    "behavioral_level",
    "construct_rationale",
    "quality_evaluation",
}


def psychometric_results_are_current(state: Mapping[str, Any]) -> bool:
    """Return whether saved statistics use the active metric contract."""

    statistics = state.get("test_statistics")
    return (
        bool(state.get("item_statistics"))
        and isinstance(statistics, Mapping)
        and statistics.get("formula_version")
        == PSYCHOMETRIC_FORMULA_VERSION
        and statistics.get("evaluation_version")
        == MEASUREMENT_EVALUATION_VERSION
    )


def _scoring_issues(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for item in items:
        item_id = item.get("item_id")
        options = item.get("response_options")
        scoring_key = item.get("scoring_key")
        if not isinstance(options, list) or not options:
            issues.append(
                {"item_id": item_id, "reason": "缺少有效作答选项"}
            )
            continue
        option_ids = [
            option.get("option_id")
            for option in options
            if isinstance(option, Mapping)
        ]
        if (
            len(option_ids) != len(options)
            or len(option_ids) != len(set(option_ids))
            or not isinstance(scoring_key, Mapping)
            or set(scoring_key) != set(option_ids)
        ):
            issues.append(
                {"item_id": item_id, "reason": "评分键未准确覆盖选项"}
            )
            continue
        scores = list(scoring_key.values())
        if (
            any(
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                for score in scores
            )
            or sorted(scores) != list(range(1, len(options) + 1))
        ):
            issues.append(
                {
                    "item_id": item_id,
                    "reason": "行为倾向题评分必须唯一覆盖1至选项数",
                }
            )
            continue
        try:
            expected = derive_scoring_key_from_behavioral_levels(dict(item))
        except ValueError as exc:
            issues.append({"item_id": item_id, "reason": str(exc)})
            continue
        if dict(scoring_key) != expected:
            issues.append(
                {
                    "item_id": item_id,
                    "reason": "scoring_key 与 behavioral_level 不一致",
                }
            )
    return issues


def _adjacency_findings(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for field, label in (
        ("blueprint_cell_id", "同一facet"),
        ("context_category", "同一情境类别"),
    ):
        values = [item.get(field) for item in items]
        adjacent = sum(
            left == right
            for left, right in zip(values, values[1:])
            if left is not None
        )
        counts = Counter(values)
        unavoidable = (
            max(counts.values(), default=0) > (len(values) + 1) // 2
        )
        if adjacent:
            findings.append(
                {
                    "criterion": f"item_order_{field}",
                    "severity": "warning" if unavoidable else "info",
                    "evidence": {
                        "adjacent_pair_count": adjacent,
                        "unavoidable_due_to_composition": unavoidable,
                    },
                    "finding": f"测验顺序中存在相邻{label}题目",
                    "recommendation": (
                        "题目构成决定无法完全分散，施测时保留当前顺序"
                        if unavoidable
                        else "重新组卷时进一步分散相邻题目"
                    ),
                }
            )
    return findings


def run_test_review(state: PSJTState) -> dict[str, Any]:
    """Deterministically review the complete assembled test."""

    assembled = state.get("assembled_test")
    if not isinstance(assembled, Mapping):
        raise ValueError("测验审核前缺少 assembled_test")
    items = assembled.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("assembled_test 缺少正式题目")

    findings: list[dict[str, Any]] = []
    decision = "PASS"
    if (
        assembled.get("item_bank_id") != state.get("item_bank_id")
        or assembled.get("item_bank_version")
        != state.get("item_bank_version")
        or assembled.get("item_bank_fingerprint")
        != state.get("item_bank_fingerprint")
    ):
        decision = "REASSEMBLE"
        findings.append(
            {
                "criterion": "item_bank_identity",
                "severity": "blocking",
                "evidence": {
                    "assembled_item_bank_id": assembled.get("item_bank_id"),
                    "current_item_bank_id": state.get("item_bank_id"),
                },
                "finding": "组卷结果未绑定当前冻结题库",
                "recommendation": "使用当前冻结题库重新组卷",
            }
        )

    coverage = assembled.get("blueprint_summary") or {}
    if not coverage.get("passed"):
        decision = "SUPPLEMENT"
        findings.append(
            {
                "criterion": "blueprint_coverage",
                "severity": "blocking",
                "evidence": deepcopy(coverage),
                "finding": "正式测验未满足蓝图覆盖要求",
                "recommendation": "按蓝图缺口补题后重新验证",
            }
        )

    public_form = assembled.get("respondent_form")
    if not isinstance(public_form, Mapping):
        decision = "REASSEMBLE"
        findings.append(
            {
                "criterion": "respondent_form",
                "severity": "blocking",
                "evidence": "respondent_form missing",
                "finding": "缺少被试卷",
                "recommendation": "重新组卷并生成被试安全版本",
            }
        )
    else:
        public_text = json.dumps(public_form, ensure_ascii=False)
        leaked_keys = sorted(
            key for key in _PUBLIC_FORBIDDEN_KEYS if key in public_text
        )
        internal_ids = {
            str(item.get("item_id"))
            for item in items
            if item.get("item_id")
        }
        leaked_ids = sorted(
            item_id for item_id in internal_ids if item_id in public_text
        )
        if leaked_keys or leaked_ids:
            decision = "REASSEMBLE"
            findings.append(
                {
                    "criterion": "respondent_form_security",
                    "severity": "blocking",
                    "evidence": {
                        "leaked_keys": leaked_keys,
                        "leaked_internal_item_ids": leaked_ids,
                    },
                    "finding": "被试卷泄露内部测量或评分信息",
                    "recommendation": "重新生成被试安全版本",
                }
            )

    scoring_issues = _scoring_issues(items)
    if scoring_issues:
        decision = "RESCORE"
        findings.append(
            {
                "criterion": "scoring_consistency",
                "severity": "blocking",
                "evidence": scoring_issues,
                "finding": "部分题目的评分键结构不一致",
                "recommendation": "按行为水平修复评分键并重新进行心理测量验证",
            }
        )

    findings.extend(_adjacency_findings(items))
    if not findings:
        findings.append(
            {
                "criterion": "complete_test_review",
                "severity": "info",
                "evidence": {
                    "item_count": len(items),
                    "item_bank_version": state.get("item_bank_version"),
                },
                "finding": "蓝图、评分、版本和被试卷安全检查均通过",
                "recommendation": "进入最终报告生成",
            }
        )
    report = {
        "decision": decision,
        "findings": findings,
        "summary": (
            "测验级审核通过"
            if decision == "PASS"
            else f"测验级审核要求执行 {decision}"
        ),
        "reviewed_at": utc_timestamp(),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
    }
    return {
        "state_update": {
            "test_review_result": report,
            "rescore_pending_revalidation": False
            if decision == "PASS"
            else state.get("rescore_pending_revalidation", False),
        },
        "summary": report["summary"],
    }


def run_test_rescore(state: PSJTState) -> dict[str, Any]:
    """Repair scoring keys and force a new bank-level validation cycle."""

    from sjt_system.evaluation.virtual_content_review import is_enabled
    if is_enabled(state):
        raise ValueError("虚拟内容复审协议冻结评分键，不能自动重计分")
    review = state.get("test_review_result")
    if not isinstance(review, Mapping) or review.get("decision") != "RESCORE":
        raise ValueError("只有 RESCORE 审核结论可以触发重计分")
    current_round = int(state.get("rescore_round") or 0)
    max_rounds = int(state.get("max_rescore_rounds") or 0)
    if current_round >= max_rounds:
        raise ValueError("已达到最大重计分轮数")

    issue_ids = {
        str(issue.get("item_id"))
        for finding in review.get("findings") or []
        if isinstance(finding, Mapping)
        and finding.get("criterion") == "scoring_consistency"
        for issue in finding.get("evidence") or []
        if isinstance(issue, Mapping) and issue.get("item_id")
    }
    if not issue_ids:
        raise ValueError("RESCORE 结论没有提供可修复题目")

    changed_ids: list[str] = []
    pool: list[dict[str, Any]] = []
    profiles = dict(state.get("item_pattern_profiles") or {})
    item_history = deepcopy(state.get("item_history") or {})
    for raw_item in state.get("item_pool") or []:
        if not isinstance(raw_item, Mapping):
            continue
        item = deepcopy(dict(raw_item))
        item_id = str(item.get("item_id"))
        if item_id in issue_ids:
            item["scoring_key"] = derive_scoring_key_from_behavioral_levels(
                item
            )
            item["version"] = int(item.get("version") or 0) + 1
            changed_ids.append(item_id)
            profiles[item_id] = build_item_pattern_profile(item, None)
            item_history.setdefault(item_id, []).append(
                {
                    "event": "rescored",
                    "recorded_at": utc_timestamp(),
                    "item": deepcopy(item),
                }
            )
        pool.append(item)
    missing = issue_ids - set(changed_ids)
    if missing:
        raise ValueError(
            "重计分找不到题目：" + "、".join(sorted(missing))
        )

    revision_history = deepcopy(state.get("test_revision_history") or [])
    revision_history.append(
        {
            "event": "rescored",
            "recorded_at": utc_timestamp(),
            "rescore_round": current_round + 1,
            "item_ids": changed_ids,
        }
    )
    previous_response_ref = state.get("virtual_response_data_ref")
    update: dict[str, Any] = {
        "item_pool": pool,
        "item_pattern_profiles": profiles,
        "item_history": item_history,
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
        "selection_reasons": {},
        "selection_results": None,
        "blueprint_coverage": None,
        "assembled_test": None,
        "test_review_result": None,
        "final_test": None,
        "item_database_ref": None,
        "technical_report": None,
        "virtual_respondent_report": None,
        "completion_checks": {},
        "unmet_completion_conditions": [],
        "rescore_round": current_round + 1,
        "rescore_pending_revalidation": True,
        "test_revision_history": revision_history,
    }
    if isinstance(previous_response_ref, str) and previous_response_ref:
        update["previous_virtual_response_data_ref"] = (
            previous_response_ref
        )
    return {
        "state_update": update,
        "summary": (
            f"已重计分 {len(changed_ids)} 道题；必须重新冻结、施测并"
            "执行心理测量验证。"
        ),
    }


def evaluate_completion(
    state: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    """Evaluate every hard completion requirement in code."""

    selection = state.get("selection_results")
    review = state.get("test_review_result")
    assembled = state.get("assembled_test")
    checks = {
        "requirements_confirmed": bool(state.get("requirements_confirmed")),
        "construct_profile_exists": isinstance(
            state.get("construct_profile"), Mapping
        ),
        "blueprint_exists": isinstance(state.get("blueprint"), Mapping),
        "item_bank_current": item_bank_snapshot_is_current(state),
        "responses_bound_to_current_bank": (
            bool(state.get("virtual_response_data_ref"))
            and state.get("virtual_response_item_bank_id")
            == state.get("item_bank_id")
            and state.get("virtual_response_item_bank_version")
            == state.get("item_bank_version")
        ),
        "psychometrics_complete": psychometric_results_are_current(state),
        "selection_ready": (
            isinstance(selection, Mapping)
            and selection.get("status") == "ready_for_assembly"
        ),
        "blueprint_coverage_passed": bool(
            (state.get("blueprint_coverage") or {}).get("passed")
        ),
        "assembly_matches_current_bank": (
            isinstance(assembled, Mapping)
            and assembled.get("item_bank_id") == state.get("item_bank_id")
            and assembled.get("item_bank_version")
            == state.get("item_bank_version")
        ),
        "test_review_passed": (
            isinstance(review, Mapping) and review.get("decision") == "PASS"
        ),
        "final_test_exists": isinstance(state.get("final_test"), Mapping),
        "item_database_exists": bool(state.get("item_database_ref")),
        "technical_report_exists": isinstance(
            state.get("technical_report"), Mapping
        ),
        "virtual_respondent_report_exists": isinstance(
            state.get("virtual_respondent_report"), Mapping
        ),
        "no_pending_rescore_revalidation": not bool(
            state.get("rescore_pending_revalidation")
        ),
    }
    unmet = [name for name, passed in checks.items() if not passed]
    return checks, unmet
