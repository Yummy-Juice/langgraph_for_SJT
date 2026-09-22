"""Assemble and compile the SJT LangGraph workflow."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sjt_system.state import PSJTState
from sjt_system.workflow.execution_node import execute_node
from sjt_system.workflow.interaction_nodes import (
    approval_node,
    automatic_approval_node,
    commit_node,
    item_development_mode_selection_node,
    plateau_gap_decision_node,
    prepare_regeneration_node,
    stop_node,
    post_virtual_response_decision_node,
    psychometric_repair_confirmation_node,
    virtual_sample_selection_node,
)
from sjt_system.workflow.item_nodes import (
    abandon_item_node,
    accept_item_node,
    prepare_item_review_node,
    prepare_item_revision_node,
)
from sjt_system.workflow.router import router_node
from sjt_system.workflow.routes import (
    route_after_approval,
    route_after_commit,
    route_after_execute,
    route_after_item_resolution,
    route_after_plateau_gap_decision,
    route_after_prepare_item_review,
    route_after_router,
    route_after_post_simulation_review,
    route_after_psychometric_repair_confirmation,
)


def build_sjt_graph(checkpointer=None, *, interrupt_before=None):
    """创建带逐步用户确认的 Router–Execute 工作流。"""

    builder = StateGraph(PSJTState)
    builder.add_node("router", router_node)
    builder.add_node("execute", execute_node)
    builder.add_node(
        "item_development_mode_selection",
        item_development_mode_selection_node,
    )
    builder.add_node(
        "virtual_sample_selection",
        virtual_sample_selection_node,
    )
    builder.add_node(
        "post_virtual_response_decision",
        post_virtual_response_decision_node,
    )
    builder.add_node(
        "psychometric_repair_confirmation",
        psychometric_repair_confirmation_node,
    )
    builder.add_node(
        "plateau_gap_decision",
        plateau_gap_decision_node,
    )
    builder.add_node("approval", approval_node)
    builder.add_node("automatic_approval", automatic_approval_node)
    builder.add_node("commit", commit_node)
    builder.add_node("prepare_regeneration", prepare_regeneration_node)
    builder.add_node("stop", stop_node)
    builder.add_node("prepare_item_review", prepare_item_review_node)
    builder.add_node("prepare_item_revision", prepare_item_revision_node)
    builder.add_node("accept_item", accept_item_node)
    builder.add_node("abandon_item", abandon_item_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "select_item_development_mode": (
                "item_development_mode_selection"
            ),
            "select_virtual_sample": "virtual_sample_selection",
            "confirm_psychometric_repair": "psychometric_repair_confirmation",
            "plateau_gap_decision": "plateau_gap_decision",
            "execute": "execute",
            "end": END,
        },
    )
    builder.add_edge("item_development_mode_selection", "execute")
    builder.add_edge("virtual_sample_selection", "execute")
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "router": "router",
            "approval": "approval",
            "automatic_approval": "automatic_approval",
            "retry_item": "prepare_item_revision",
            "abandon": "abandon_item",
            "accept_latest": "accept_item",
            "end": END,
        },
    )
    builder.add_edge("automatic_approval", "commit")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "commit": "commit",
            "regenerate": "prepare_regeneration",
            "stop": "stop",
        },
    )
    builder.add_conditional_edges(
        "commit",
        route_after_commit,
        {
            "router": "router",
            "review": "prepare_item_review",
            "accept": "accept_item",
            "revise": "prepare_item_revision",
            "abandon": "abandon_item",
            "post_simulation_review": "post_virtual_response_decision",
        },
    )
    builder.add_conditional_edges(
        "post_virtual_response_decision",
        route_after_post_simulation_review,
        {"router": "router", "end": END},
    )
    builder.add_conditional_edges(
        "psychometric_repair_confirmation",
        route_after_psychometric_repair_confirmation,
        {"router": "router", "end": END},
    )
    builder.add_conditional_edges(
        "plateau_gap_decision",
        route_after_plateau_gap_decision,
        {"router": "router", "end": END},
    )
    builder.add_conditional_edges(
        "prepare_item_review",
        route_after_prepare_item_review,
        {"execute": "execute", "accept": "accept_item"},
    )
    builder.add_edge("prepare_item_revision", "execute")
    builder.add_conditional_edges(
        "accept_item",
        route_after_item_resolution,
        {"router": "router"},
    )
    builder.add_conditional_edges(
        "abandon_item",
        route_after_item_resolution,
        {"router": "router"},
    )
    builder.add_edge("prepare_regeneration", "execute")
    builder.add_edge("stop", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=interrupt_before)


graph = build_sjt_graph()
