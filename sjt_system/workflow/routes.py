"""Conditional-edge routing functions for the compiled workflow."""

from __future__ import annotations

from collections.abc import Mapping

from sjt_system.authoring.items import next_step_after_review
from sjt_system.evaluation.respondents import matched_condition_sample_is_current
from sjt_system.state import PSJTState
from sjt_system.workflow.constants import (
    DETERMINISTIC_AUTO_APPROVAL_ACTIONS,
    ITEM_DEVELOPMENT_ACTIONS,
)

def route_after_router(state: PSJTState) -> str:
    if state["status"] == "failed":
        return "end"
    if state["route"] and state["route"]["next_action"] == "finish":
        return "end"
    if (
        state.get("route")
        and state["route"]["next_action"] == "plateau_gap_decision"
    ):
        return "plateau_gap_decision"
    if (
        state.get("route")
        and state["route"]["next_action"] == "confirm_psychometric_repair"
    ):
        return "confirm_psychometric_repair"
    if (
        state["route"]
        and state["route"]["next_action"] == "simulate_responses"
        and (
            not matched_condition_sample_is_current(
                state.get("virtual_sample_config"),
                state.get("virtual_respondents"),
            )
        )
    ):
        return "select_virtual_sample"
    if (
        state["route"]
        and state["route"]["next_action"] in ITEM_DEVELOPMENT_ACTIONS
        and state.get("item_development_mode") not in {"manual", "automatic"}
    ):
        return "select_item_development_mode"
    return "execute"


def route_after_execute(state: PSJTState) -> str:
    if state.get("skeleton_slot_failure_pending"):
        return "router"
    if state.get("pending_action") == "psychometric_repair_batch":
        return "automatic_approval"
    if state.get("review_process_status") == "exhausted":
        return "accept_latest"
    if state.get("current_item_repair_failure"):
        # Invalid model output is a content-attempt failure, not authority to
        # discard the program-owned slot or invent a replacement ID.
        return (
            "abandon"
            if next_step_after_review(state) == "abandon"
            else "retry_item"
        )
    if (
        state.get("route")
        and state["route"]["next_action"] == "build_blueprint"
        and not state.get("requirements_confirmed")
        and state.get("pending_action") is None
    ):
        return "router"
    if state["status"] == "failed":
        return "end"
    if (
        state.get("item_development_mode") == "automatic"
        and state.get("pending_action")
        in (
            ITEM_DEVELOPMENT_ACTIONS
            | DETERMINISTIC_AUTO_APPROVAL_ACTIONS
        )
    ):
        return "automatic_approval"
    return "approval"


def route_after_approval(state: PSJTState) -> str:
    decision = state.get("user_decision")
    if decision is None:
        raise ValueError("Approval 路由缺少用户决策")
    if decision["decision"] in {"approve", "edit", "confirm"}:
        return "commit"
    if decision["decision"] in {
        "regenerate",
        "answer",
        "accept_suggestions",
        "revise",
    }:
        return "regenerate"
    return "stop"


def route_after_commit(state: PSJTState) -> str:
    """题目微流程确定性分流；其他任务仍返回顶层 Router。"""

    route = state.get("route")
    if route is None:
        raise ValueError("Commit 后路由缺少上一项动作")
    action = route["next_action"]

    if action == "simulate_responses":
        # Psychometrics must run before the user sees the repair decision.
        return "router"
    if action == "analyze_psychometrics":
        # Every analysis round must show its failures and locked-item monitoring
        # before the next diagnosis is allowed to start.
        return "post_simulation_review"
    if action == "psychometric_repair_batch":
        return "router"

    if action in {"generate_item", "revise_item", "regenerate_item"}:
        return "review"
    if action == "review_item":
        return next_step_after_review(state)
    if action == "select_items":
        return "router"
    return "router"


def route_after_post_simulation_review(state: PSJTState) -> str:
    return "end" if state.get("status") == "stopped" else "router"


def route_after_psychometric_repair_confirmation(state: PSJTState) -> str:
    return "end" if state.get("status") == "stopped" else "router"


def route_after_plateau_gap_decision(state: PSJTState) -> str:
    return "end" if state.get("status") == "stopped" else "router"


def route_after_prepare_item_review(state: PSJTState) -> str:
    """Skip a second quality review after a requested text change landed."""

    review = state.get("current_item_review")
    if isinstance(review, Mapping) and not any(
        isinstance(finding, Mapping)
        and finding.get("severity") == "blocking"
        for finding in review.get("findings") or []
    ):
        return "accept"
    return "execute"


def route_after_item_resolution(state: PSJTState) -> str:
    """Return to the router after accepting or abandoning one item."""

    return "router"
