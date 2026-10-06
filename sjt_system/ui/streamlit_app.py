"""Streamlit workspace for developing an SJT through the full workflow."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st
from langgraph.types import Command

from sjt_system.authoring.construct_registry import (
    resolve_specification_profile,
)
from sjt_system.config import (
    DEFAULT_REQUESTED_ITEM_COUNT,
    DEFAULT_TARGET_CONSTRUCT,
    DEFAULT_TARGET_POPULATION,
    DEFAULT_USER_REQUEST,
)
from sjt_system.authoring.generation_plan import (
    planned_generation_count,
    planned_retention_count,
)
from sjt_system.runtime.checkpoint import (
    DEFAULT_CHECKPOINT_ROOT,
    find_latest_resumable_checkpoint,
    prepare_retry_state,
    prepare_resumed_state,
)
from sjt_system.state import create_initial_state
from sjt_system.ui.presenters import (
    action_label,
    build_timeline_entries,
    extract_action_content,
    phase_label,
    progress_summary,
)
from sjt_system.ui.workflow_runner import WorkflowRunError, run_until_pause
from sjt_system.workflow.graph import build_sjt_graph
from sjt_system.evaluation.form_metrics import form_quality_summary
from sjt_system.evaluation.round_results import GATE_ORDER, metric_scalar
from sjt_system.delivery.reporting import (
    development_round_batch_label,
    strict_development_round_entries,
)
from sjt_system.evaluation.demographics import (
    DEFAULT_RESPONSE_TEMPERATURE,
    DEMOGRAPHICS_VERSION,
    demographics_to_columns,
)


SESSION_DEFAULTS = {
    "sjt_state": None,
    "sjt_interrupt": None,
    "sjt_diagnostics_reconnect": False,
    "sjt_timeline": [],
    "sjt_seen_event_ids": set(),
    "sjt_runtime_events": [],
    "sjt_error": None,
    "sjt_latest_checkpoint": None,
    "sjt_workflow": None,
}

SPECIFICATION_LABELS = {
    "construct_selection": "测量构念",
    "target_population": "目标人群",
    "facet_item_counts": "各 facet 题数",
    "final_item_count": "最终题量",
    "output_language": "输出语言",
}

CONTENT_LABELS = {
    "virtual_response_summary": "虚拟作答摘要",
    "test_statistics": "测验统计",
    "item_statistics": "题目统计",
    "psychometric_round_result": "本轮虚拟筛查结果",
    "psychometric_iteration_history": "整卷迭代曲线",
    "factor_results": "因素分析",
    "irt_results": "IRT 分析",
    "dif_results": "DIF 分析",
    "selection_results": "筛选结果",
    "selected_items": "入选题目",
    "assembled_test": "组卷结果",
    "final_test": "正式测验",
    "technical_report": "技术报告",
    "virtual_respondent_report": "虚拟被试报告",
}

USER_VISIBLE_TIMELINE_NODES = {
    "execute",
    "accept_item",
    "abandon_item",
    "item_development_mode_selection",
    "virtual_sample_selection",
    "stop",
}


def _workflow():
    workflow = st.session_state.sjt_workflow
    if workflow is None:
        workflow = build_sjt_graph()
        st.session_state.sjt_workflow = workflow
    return workflow


def _initialize_session() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                set(default) if isinstance(default, set)
                else list(default) if isinstance(default, list)
                else default
            )


def _reset_session() -> None:
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = (
            set(default) if isinstance(default, set)
            else list(default) if isinstance(default, list)
            else default
        )


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 85% 0%, #eef6f3 0, transparent 30rem),
                #f7f8f6;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }
        h1, h2, h3 { letter-spacing: -0.025em; }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid #e2e7e3;
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid #dde5df;
            border-radius: 18px;
            padding: 1.2rem 1.35rem 0.5rem;
            box-shadow: 0 12px 36px rgba(27, 54, 43, 0.06);
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.82);
            border-color: #e0e5e1;
            border-radius: 14px;
        }
        .sjt-kicker {
            color: #28705a;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .sjt-subtitle {
            color: #5c6862;
            font-size: 1.05rem;
            max-width: 760px;
            margin-top: -0.4rem;
            margin-bottom: 1.8rem;
        }
        .sjt-note {
            color: #617068;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _runtime_message(event: Mapping[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "simulation_progress":
        return (
            f"虚拟作答：{event.get('completed', 0)}/"
            f"{event.get('total', '?')}"
        )
    if event_type == "simulation_stage":
        return (
            f"虚拟作答 · {event.get('stage', '处理中')} · "
            f"{event.get('status', 'running')}"
        )
    if event_type in {"request_retry", "output_repair"}:
        return (
            f"{event.get('job_label', '模型请求')}正在重试："
            f"{event.get('reason', '输出未通过校验')}"
        )
    if event_type == "action_fallback":
        return (
            f"{action_label(event.get('from_action'))}未通过校验，"
            f"转为{action_label(event.get('to_action'))}"
        )
    return str(event.get("message") or event_type or "正在处理")


def _latest_update_message(update: Mapping[str, Any]) -> str:
    history = update.get("execution_history") or []
    if history and isinstance(history[-1], Mapping):
        event = history[-1]
        return (
            f"{action_label(event.get('action'))} · "
            f"{event.get('event_type', 'completed')}"
        )
    return "流程状态已更新"


def _advance(graph_input: dict[str, Any] | Command) -> None:
    state = st.session_state.sjt_state
    if not isinstance(state, Mapping):
        raise ValueError("页面中没有可运行的工作流状态")

    with st.status("正在推进测验开发流程", expanded=True) as status:
        try:
            turn = asyncio.run(
                run_until_pause(
                    _workflow(),
                    graph_input,
                    state,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                    on_update=lambda _node, update, _state: status.write(
                        _latest_update_message(update)
                    ),
                    on_progress=lambda event: status.write(
                        _runtime_message(event)
                    ),
                )
            )
            _store_turn_result(
                turn.state,
                turn.interrupt,
                turn.updates,
                turn.runtime_events,
            )
            if turn.state.get("status") == "failed":
                st.session_state.sjt_error = _latest_state_error(turn.state)
                status.update(
                    label="流程已暂停，可以从检查点重试",
                    state="error",
                    expanded=True,
                )
            else:
                st.session_state.sjt_error = None
                status.update(
                    label=(
                        "流程等待你的确认"
                        if turn.interrupt
                        else "本轮流程已经完成"
                    ),
                    state="complete",
                    expanded=False,
                )
        except WorkflowRunError as exc:
            _store_turn_result(
                exc.state,
                None,
                exc.updates,
                exc.runtime_events,
            )
            st.session_state.sjt_error = str(exc)
            status.update(
                label="流程运行失败",
                state="error",
                expanded=True,
            )
            st.exception(exc.original_error)
        except Exception as exc:
            st.session_state.sjt_error = str(exc)
            status.update(
                label="流程运行失败",
                state="error",
                expanded=True,
            )
            st.exception(exc)


def _store_turn_result(
    state: Mapping[str, Any],
    interrupt: Mapping[str, Any] | None,
    updates: list[dict[str, Any]],
    runtime_events: list[dict[str, Any]],
) -> None:
    entries, seen = build_timeline_entries(
        updates,
        st.session_state.sjt_seen_event_ids,
    )
    st.session_state.sjt_state = dict(state)
    st.session_state.sjt_interrupt = (
        dict(interrupt) if interrupt is not None else None
    )
    st.session_state.sjt_timeline.extend(entries)
    st.session_state.sjt_seen_event_ids = seen
    st.session_state.sjt_runtime_events.extend(runtime_events)


def _latest_state_error(state: Mapping[str, Any]) -> str:
    errors = state.get("errors") or []
    if not errors:
        return "流程未完成，请从最近检查点重试。"
    latest = errors[-1]
    if isinstance(latest, Mapping):
        return str(latest.get("message") or "流程未完成")
    return str(latest)


def _retry_current_run(state: Mapping[str, Any]) -> None:
    retry_state = prepare_retry_state(
        state,
        checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
    )
    st.session_state.sjt_state = retry_state
    st.session_state.sjt_interrupt = None
    st.session_state.sjt_error = None
    # Drop the in-memory graph checkpoint before replaying the durable state.
    st.session_state.sjt_workflow = build_sjt_graph()
    _advance(retry_state)


def _start_new_run(user_request: str) -> None:
    _reset_session()
    default_kwargs = {}
    if user_request.strip() == DEFAULT_USER_REQUEST:
        default_kwargs = {
            "target_population": DEFAULT_TARGET_POPULATION,
            "target_construct": DEFAULT_TARGET_CONSTRUCT,
            "requested_item_count": DEFAULT_REQUESTED_ITEM_COUNT,
        }
    state = create_initial_state(user_request, **default_kwargs)
    st.session_state.sjt_state = state
    _advance(state)


def _resume_latest_run() -> None:
    latest = find_latest_resumable_checkpoint(DEFAULT_CHECKPOINT_ROOT)
    if latest is None:
        st.warning("没有找到可继续的运行。")
        return
    _reset_session()
    state = prepare_resumed_state(
        latest["state"],
        checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
    )
    st.session_state.sjt_state = state
    _advance(state)


def _render_start_page() -> None:
    left, right = st.columns([1.55, 0.7], gap="large")
    with left:
        st.markdown(
            '<div class="sjt-kicker">SJT DEVELOPMENT WORKSPACE</div>',
            unsafe_allow_html=True,
        )
        st.title("把测验需求变成可审查的完整成果")
        st.markdown(
            '<div class="sjt-subtitle">'
            "描述你要测量什么、面向谁以及希望得到多少道题。"
            "系统会逐步完成需求澄清、构念档案解析、生成计划、题目开发、"
            "虚拟检验和报告生成。"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.form("new_sjt_request"):
            request = st.text_area(
                "你想开发什么测验？",
                height=170,
                value=DEFAULT_USER_REQUEST,
                placeholder=(
                    "默认按 NEO-PI-R 全部 facet、每个 facet 5 道题处理；"
                    "也可以输入其他构念、facet 或题数……"
                ),
                label_visibility="visible",
            )
            st.markdown(
                '<div class="sjt-note">'
                "暂时不确定的内容可以留给系统在下一步与你确认。"
                "</div>",
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button(
                "开始开发",
                type="primary",
                use_container_width=True,
            )
        if submitted:
            if not request.strip():
                st.error("请先描述你的测验需求。")
            else:
                _start_new_run(request.strip())
                st.rerun()

    with right:
        st.subheader("你会看到什么")
        st.markdown(
            "1. **当前进度**：流程进行到了哪一步\n\n"
            "2. **阶段成果**：模型、蓝图、题目和分析结果\n\n"
            "3. **待确认事项**：需要你决定时才出现\n\n"
            "4. **最终交付**：正式测验及技术报告"
        )
        latest = st.session_state.sjt_latest_checkpoint
        if latest is None and st.button(
            "查找未完成任务",
            use_container_width=True,
        ):
            with st.spinner("正在检查历史任务……"):
                try:
                    latest = find_latest_resumable_checkpoint(
                        DEFAULT_CHECKPOINT_ROOT
                    )
                    st.session_state.sjt_latest_checkpoint = latest or False
                except ValueError as exc:
                    st.session_state.sjt_latest_checkpoint = False
                    st.caption(f"已有检查点不可用：{exc}")
        if latest is not None:
            if latest is False:
                st.caption("没有找到可继续的任务。")
                return
            st.divider()
            state = latest["state"]
            st.caption("检测到一项未完成任务")
            st.write(
                f"**{phase_label(state.get('current_phase'))}** · "
                f"{len(state.get('item_pool') or [])} 道候选题"
            )
            if st.button(
                "继续上次任务",
                use_container_width=True,
            ):
                _resume_latest_run()
                st.rerun()


def _render_specification(specification: object) -> None:
    if not isinstance(specification, Mapping):
        st.info("尚未形成测验规格。")
        return
    rows = []
    for field, label in SPECIFICATION_LABELS.items():
        value = specification.get(field)
        if value is not None and value != []:
            rendered = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
            rows.append({"项目": label, "内容": rendered})
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_item(item: object) -> None:
    if not isinstance(item, Mapping):
        st.info("暂无题目内容。")
        return
    item_id = item.get("item_id")
    if item_id:
        st.caption(f"题目编号：{item_id}")
    st.markdown(f"**情境**\n\n{item.get('scenario', '未提供')}")
    stem = item.get("stem")
    if stem:
        st.markdown(f"**问题**\n\n{stem}")
    options = item.get("options") or []
    if options:
        st.markdown("**选项**")
        for index, option in enumerate(options, 1):
            if isinstance(option, Mapping):
                label = option.get("label") or chr(64 + index)
                text = (
                    option.get("text")
                    or option.get("content")
                    or str(dict(option))
                )
                st.write(f"{label}. {text}")
            else:
                st.write(f"{index}. {option}")
    with st.expander("查看题目结构与计分信息"):
        st.json(dict(item), expanded=False)


def _render_blueprint(blueprint: object) -> None:
    if not isinstance(blueprint, Mapping):
        st.info("暂无蓝图内容。")
        return
    for field in ("title", "summary", "rationale"):
        if blueprint.get(field):
            st.write(blueprint[field])
    is_fixed_blueprint = blueprint.get("version") == 6
    profile = blueprint.get("construct_profile_snapshot")
    if is_fixed_blueprint and isinstance(profile, Mapping):
        st.subheader("构念档案")
        st.write(
            f"{profile.get('inventory_name', '')} · "
            f"{profile.get('domain_name', '')} · "
            f"{profile.get('selection_level', '')}"
        )
        facets = [
            {
                "facet": facet.get("facet_name"),
                "definition": facet.get("definition"),
            }
            for facet in profile.get("facets") or []
            if isinstance(facet, Mapping)
        ]
        if facets:
            st.dataframe(facets, hide_index=True, width="stretch")
        st.caption(
            f"程序汇总：计划生成 {planned_generation_count(blueprint)} 题，"
            f"最终保留 {planned_retention_count(blueprint)} 题。"
        )
    cells = blueprint.get("cells") or []
    if cells:
        st.subheader("维度分配")
        st.dataframe(cells, hide_index=True, width="stretch")
    slots = blueprint.get("slots") or []
    if slots:
        st.subheader("固定题号")
        st.dataframe(slots, hide_index=True, width="stretch")
    with st.expander("查看完整细目表数据"):
        st.json(dict(blueprint), expanded=False)


def _render_review(review: object) -> None:
    if not isinstance(review, Mapping):
        st.info("暂无审查内容。")
        return
    decision = (
        review.get("decision")
        or review.get("overall_decision")
        or review.get("status")
    )
    if decision:
        st.write(f"**审查结论：** {decision}")
    if review.get("summary"):
        st.write(review["summary"])
    issues = list(review.get("issues") or [])
    issues.extend(review.get("findings") or [])
    for section_name in (
        "construct_review",
        "item_skeleton_review",
        "content_review",
    ):
        section = review.get(section_name)
        if isinstance(section, Mapping):
            issues.extend(section.get("issues") or [])
    for issue in issues:
        if isinstance(issue, Mapping):
            message = (
                issue.get("description")
                or issue.get("issue")
                or issue.get("problem")
                or str(dict(issue))
            )
            if issue.get("severity") == "blocking":
                st.error(message)
            else:
                st.warning(message)
        else:
            st.warning(str(issue))
    tasks = review.get("repair_tasks") or []
    if tasks:
        with st.expander("查看程序派生修复任务"):
            st.json(tasks, expanded=False)
    with st.expander("查看完整审查记录"):
        st.json(dict(review), expanded=False)


def _render_generic(value: object) -> None:
    if isinstance(value, list) and value and all(
        isinstance(item, Mapping) for item in value
    ):
        st.dataframe(value, hide_index=True, use_container_width=True)
    elif isinstance(value, Mapping):
        st.json(dict(value), expanded=False)
    else:
        st.write(value)


def _metric_text(value: object) -> str:
    numeric = metric_scalar(value)
    if numeric is None:
        return "证据不足"
    return f"{numeric:.3f}"


def _option_mean_text(value: object, count: object) -> str:
    return "无人选择" if count == 0 else _metric_text(value)


_ROUND_STATUS_LABELS = {
    "pending_treatment": "待处理",
    "newly_qualified": "本轮新合格",
    "qualified_locked": "正式题已锁定",
    "qualified_locked_warning": "正式题监测警告",
    "pending_sme_review": "待SME审核",
    "eliminated": "已淘汰",
    "facet_form_retained": "Facet整卷达标保留（单题门槛未全过）",
}


def _gate_display(gate: Mapping[str, Any]) -> str:
    if gate.get("threshold") is None or gate.get("passes") is None:
        return f"{_metric_text(gate.get('value'))} / 仅诊断"
    status = "通过" if gate.get("passes") is True else (
        "未通过" if gate.get("estimable") is True else "不可估计"
    )


def _profile_diagnostic_value(entry: Mapping[str, Any], diagnostic_id: str) -> Any:
    for row in entry.get("profile_diagnostics") or []:
        if isinstance(row, Mapping) and row.get("diagnostic_id") == diagnostic_id:
            return row.get("value")
    legacy_gate_id = {
        "target_rho_diagnostic": "target_rho_pass",
        "same_domain_vts_diagnostic": "same_domain_vts_pass",
        "cross_domain_vts_diagnostic": "cross_domain_vts_pass",
    }.get(diagnostic_id)
    value_alias = {
        "target_rho_diagnostic": "target_rho",
        "same_domain_vts_diagnostic": "same_domain_vts",
        "cross_domain_vts_diagnostic": "cross_domain_vts",
    }.get(diagnostic_id)
    if value_alias in entry:
        return entry.get(value_alias)
    for row in entry.get("gates") or []:
        if isinstance(row, Mapping) and row.get("gate_id") == legacy_gate_id:
            return row.get("value")
    return None


def _comparison_gate(
    comparison: Mapping[str, Any],
    current_key: str,
    legacy_key: str | None = None,
) -> Mapping[str, Any]:
    current = comparison.get(current_key)
    if isinstance(current, Mapping):
        return current
    legacy = comparison.get(legacy_key) if legacy_key else None
    return legacy if isinstance(legacy, Mapping) else {}


def _improvement_value(gate: Mapping[str, Any]) -> float | None:
    observed = metric_scalar(gate.get("observed"))
    incumbent = metric_scalar(gate.get("incumbent"))
    return None if observed is None or incumbent is None else observed - incumbent
    return (
        f"{_metric_text(gate.get('value'))} / "
        f"≥{_metric_text(gate.get('threshold'))} / {status}"
    )


def _frequency_display(option: Mapping[str, Any], group_n: object) -> str:
    rate = option.get("selection_rate")
    count = option.get("selection_count")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return f"- ({count or 0}/{group_n or 0})"
    return f"{float(rate) * 100:.1f}% ({count}/{group_n})"


def _option_frequency_rows(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate = diagnostics.get("aggregate") or diagnostics.get("all") or {}
    option_ids = [
        str(row.get("option_id"))
        for row in aggregate.get("options") or []
        if isinstance(row, Mapping)
    ]
    rows: list[dict[str, Any]] = []

    def add_group(label: str, group: Mapping[str, Any], estimable: object) -> None:
        by_option = {
            str(row.get("option_id")): row
            for row in group.get("options") or []
            if isinstance(row, Mapping)
        }
        result = {
            "分组": label,
            "N": group.get("group_n"),
            "可用于定位": "是" if estimable is True else "否",
        }
        for option_id in option_ids:
            result[option_id] = _frequency_display(
                by_option.get(option_id) or {}, group.get("group_n")
            )
        rows.append(result)

    add_group("全样本", aggregate, True)
    for condition in diagnostics.get("by_condition") or []:
        if isinstance(condition, Mapping):
            add_group(
                f"条件 {condition.get('condition_id')}",
                condition,
                True,
            )
    return rows


def _option_facet_mean_display_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "题目": row.get("item_id"),
            "实验臂": row.get("condition_id"),
            "facet": row.get("facet_name") or row.get("facet_id"),
            "选项": row.get("option_id"),
            "计分": _metric_text(row.get("option_score", row.get("score"))),
            "选择人数": row.get("n"),
            "facet均值": _metric_text(row.get("facet_mean")),
            "标准误": _metric_text(row.get("facet_standard_error")),
            "参与过滤": "否",
        }
        for row in rows
        if isinstance(row, Mapping)
    ]


def _option_score_comparison_display_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "题目": row.get("item_id"),
            "选项": row.get("option_id"),
            "计分": _metric_text(row.get("option_score")),
            "目标N": row.get("target_n"),
            "目标facet平均设定分数": _option_mean_text(row.get("target_mean_score"), row.get("target_n")),
            "同域最大rho facet": row.get("same_domain_facet_name") or row.get("same_domain_dimension_id"),
            "同域N": row.get("same_domain_n"),
            "同域非目标facet平均设定分数": _option_mean_text(
                row.get("same_domain_mean_score"), row.get("same_domain_n")
            ),
            "跨域最大rho facet": row.get("cross_domain_facet_name") or row.get("cross_domain_dimension_id"),
            "跨域N": row.get("cross_domain_n"),
            "跨域非目标facet平均设定分数": _option_mean_text(
                row.get("cross_domain_mean_score"), row.get("cross_domain_n")
            ),
            "参与过滤": "否",
        }
        for row in rows
        if isinstance(row, Mapping)
    ]


def _agent_option_score_comparison_display_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "VTS类别": row.get("vts_category"),
            "最大rho非目标facet": row.get("non_target_facet_name") or row.get("non_target_dimension_id"),
            "选项": row.get("option_id"),
            "计分": _metric_text(row.get("score")),
            "目标facet平均设定分数": _option_mean_text(row.get("target_mean_score"), row.get("target_n")),
            "对应非目标facet平均设定分数": _option_mean_text(
                row.get(f"{row.get('vts_category')}_mean_score"), row.get("non_target_n")
            ),
        }
        for row in rows
        if isinstance(row, Mapping)
    ]


def _arm_difference_display_rows(
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in diagnostics.get("comparisons") or []:
        if not isinstance(comparison, Mapping):
            continue
        overall = comparison.get("overall") or {}
        high_band = next(
            (
                row
                for row in comparison.get("by_score_band") or []
                if isinstance(row, Mapping) and row.get("score_band") == "high"
            ),
            {},
        )
        rows.append(
            {
                "比较": comparison.get("comparison_id"),
                "污染facet": comparison.get("comparator_facet_name")
                or comparison.get("comparator_dimension_id"),
                "配对N": comparison.get("matched_subject_count"),
                "同选项率": _metric_text(overall.get("same_option_rate")),
                "总体目标-污染臂题分": _metric_text(
                    overall.get("target_minus_comparator_mean_item_score")
                ),
                "高分组三四级选择率差": _metric_text(
                    high_band.get("target_minus_comparator_high_option_rate")
                ),
                "可估计": "是" if comparison.get("estimable") is True else "否",
                "参与过滤": "否",
            }
        )
    return rows


def _round_overview_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        str(row.get("gate_id")): row
        for row in entry.get("gates") or []
        if isinstance(row, Mapping) and row.get("gate_id") in GATE_ORDER
    }
    contaminants = entry.get("max_contaminants") or {}
    same = contaminants.get("same_domain") or {}
    cross = contaminants.get("cross_domain") or {}
    return {
        "题目": entry.get("item_id"),
        "状态": _ROUND_STATUS_LABELS.get(
            str(entry.get("status")), entry.get("status")
        ),
        "监测通过": entry.get("monitoring_pass"),
        "保留依据": entry.get("retention_basis"),
        "题目级指标": (
            "冻结"
            if entry.get("iteration_metric_status") == "frozen"
            else "本轮重算"
            if entry.get("iteration_metric_status") == "measured"
            else "未标记"
        ),
        "CITC（值/门槛/状态）": _gate_display(gates.get("citc_pass") or {}),
        "Profile目标rho_s（诊断）": _metric_text(
            _profile_diagnostic_value(entry, "target_rho_diagnostic")
        ),
        "Profile同域VTS（诊断）": _metric_text(
            _profile_diagnostic_value(entry, "same_domain_vts_diagnostic")
        ),
        "同域最大污染facet": same.get("facet_name") or same.get("dimension_id"),
        "Profile跨域VTS（诊断）": _metric_text(
            _profile_diagnostic_value(entry, "cross_domain_vts_diagnostic")
        ),
        "跨域最大污染facet": cross.get("facet_name") or cross.get("dimension_id"),
        "单题Hedges'g（值/门槛/状态）": _gate_display(
            gates.get("target_hedges_g_pass") or {}
        ),
        "单题IPIP rho_s（值/门槛/状态）": _gate_display(
            gates.get("target_ipip_spearman_rho_pass") or {}
        ),
        "单题Δmin（值/门槛/状态）": _gate_display(
            gates.get("discriminant_delta_min_pass") or {}
        ),
    }


def _render_round_item_details(entry: Mapping[str, Any]) -> None:
    st.dataframe(
        [
            {
                "指标": gate.get("label"),
                "当前值": _metric_text(gate.get("value")),
                "门槛": "≥" + _metric_text(gate.get("threshold")),
                "状态": (
                    "通过" if gate.get("passes") is True else (
                        "未通过" if gate.get("estimable") is True else "不可估计"
                    )
                ),
                "参与过滤": "是" if gate.get("filtering_authority") else "否",
            }
            for gate in entry.get("gates") or []
            if isinstance(gate, Mapping) and gate.get("gate_id") in GATE_ORDER
        ],
        hide_index=True,
        use_container_width=True,
    )
    profile_rows = [
        {
            "指标": row.get("label"),
            "当前值": _metric_text(row.get("value")),
            "阈值": "无",
            "状态": "仅诊断",
            "参与过滤": "否",
        }
        for row in entry.get("profile_diagnostics") or []
        if isinstance(row, Mapping)
    ]
    if not profile_rows:
        profile_rows = [
            {"指标": label, "当前值": _metric_text(_profile_diagnostic_value(entry, key)),
             "阈值": "无", "状态": "仅诊断", "参与过滤": "否"}
            for key, label in (
                ("target_rho_diagnostic", "Profile目标rho_s"),
                ("same_domain_vts_diagnostic", "Profile同域VTS"),
                ("cross_domain_vts_diagnostic", "Profile跨域VTS"),
            )
        ]
    st.markdown("**Profile诊断指标（不参与资格）**")
    st.dataframe(profile_rows, hide_index=True, use_container_width=True)
    contaminants = entry.get("max_contaminants") or {}
    contaminant_rows = []
    for category, label in (("same_domain", "同域"), ("cross_domain", "跨域")):
        row = contaminants.get(category) or {}
        contaminant_rows.append(
            {
                "类别": label,
                "最大污染facet": row.get("facet_name") or row.get("dimension_id"),
                "选中group": row.get("group_id") or row.get("condition_id"),
                "domain": row.get("domain_id"),
                "条件rho_s": _metric_text(row.get("signed_rho")),
                "VTS": _metric_text(row.get("vts")),
                "定义": row.get("definition"),
            }
        )
    st.markdown("**最大污染 facet**")
    st.dataframe(contaminant_rows, hide_index=True, use_container_width=True)
    condition_rows = []
    for condition in entry.get("per_condition_metrics") or []:
        if isinstance(condition, Mapping):
            condition_rows.append(
                {
                    "条件": condition.get("condition_id"),
                    "CITC": _metric_text(condition.get("citc")),
                    "rho": _metric_text(condition.get("rho")),
                    "参与过滤": "是" if condition.get("filtering_authority") else "否",
                }
            )
    if condition_rows:
        st.markdown("**各条件臂诊断指标**")
        st.dataframe(condition_rows, hide_index=True, use_container_width=True)
    gradient = entry.get("target_option_gradient") or {}
    if gradient:
        st.markdown("**目标组选项梯度（返修触发器）**")
        st.dataframe(gradient.get("options") or [], hide_index=True, use_container_width=True)
        if gradient.get("failed_adjacent_pairs"):
            st.warning("失败相邻对：" + "、".join(f"{row.get('lower_option_id')} < {row.get('higher_option_id')}" for row in gradient.get("failed_adjacent_pairs") or []))
    item = entry.get("item") or {}
    if isinstance(item, Mapping) and item:
        st.markdown("**题面与选项**")
        _render_item(item)
    score_rows = _option_score_comparison_display_rows(
        entry.get("option_score_comparisons") or []
    )
    if score_rows:
        st.markdown("**按 option_id 对齐的设定分数均值**")
        st.dataframe(score_rows, hide_index=True, use_container_width=True)


def _render_psychometric_round_result(round_result: Mapping[str, Any]) -> None:
    summary = round_result.get("summary") or {}
    development_round = round_result.get("development_round") or (
        (round_result.get("facet_iteration_state") or {}).get("development_round")
        if isinstance(round_result.get("facet_iteration_state"), Mapping)
        else None
    )
    round_label = f"第 {development_round or '?'} 开发轮虚拟筛查"
    st.markdown(
        f"**{round_label}**"
    )
    summary_values = (
        ("分析题目", summary.get("item_count")),
        ("本轮新合格", summary.get("newly_qualified_count")),
        ("本轮四项主动门槛均通过", summary.get("current_round_qualified_count")),
        ("累计冻结合格", summary.get("frozen_qualified_count")),
        ("冻结题目级指标", summary.get("frozen_metric_count")),
        ("重算题目级指标", summary.get("measured_metric_count")),
        ("待处理", summary.get("pending_treatment_count")),
        ("正式题已锁定", summary.get("qualified_locked_count")),
        ("整卷达标保留但单题未全过", summary.get("facet_form_retained_count")),
        ("监测警告", summary.get("monitoring_warning_count")),
        ("不可估计门槛", summary.get("unestimable_metric_count")),
    )
    for offset in range(0, len(summary_values), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, summary_values[offset:offset + 3]):
            column.metric(label, value if value is not None else 0)
    st.markdown("**四项主动门槛通过概况**")
    gates = [
        gate for gate in round_result.get("gate_summary") or []
        if isinstance(gate, Mapping) and gate.get("gate_id") in GATE_ORDER
    ]
    for offset in range(0, len(gates), 4):
        gate_columns = st.columns(4)
        for column, gate in zip(gate_columns, gates[offset:offset + 4]):
            column.metric(
                str(gate.get("label")),
                f"{gate.get('pass_count', 0)}/{gate.get('item_count', 0)}",
            )
            column.caption(f"门槛 ≥{_metric_text(gate.get('threshold'))}")

    candidates = [
        row
        for row in round_result.get("items") or []
        if isinstance(row, Mapping)
        and row.get("status") in {"pending_treatment", "newly_qualified"}
    ]
    st.markdown("**本轮候选题总览**")
    if candidates:
        st.dataframe(
            [_round_overview_row(row) for row in candidates],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("本轮没有待处理或新合格的候选题。")
    locked = [
        row for row in round_result.get("locked_items") or []
        if isinstance(row, Mapping)
    ]
    st.markdown("**已锁定正式题监测**")
    if locked:
        st.dataframe(
            [_round_overview_row(row) for row in locked],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("当前没有已锁定正式题。")
    retained = [
        row for row in round_result.get("facet_form_retained_items") or []
        if isinstance(row, Mapping)
    ]
    st.markdown("**Facet整卷达标保留（不代表单题门槛通过）**")
    if retained:
        st.dataframe(
            [_round_overview_row(row) for row in retained],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "仅在三个开发轮次全部完成后纳入交付组卷；该处置不改写单题资格，"
            "仍属于探索性虚拟开发证据。"
        )
    else:
        st.info("当前没有按Facet整卷指标保留但未通过全部单题门槛的题目。")
    st.markdown("**所有题目的按 option_id 对齐设定分数均值**")
    all_score_rows = _option_score_comparison_display_rows(
        round_result.get("option_score_comparisons") or []
    )
    if all_score_rows:
        st.dataframe(
            all_score_rows,
            hide_index=True,
            use_container_width=True,
        )
        st.caption("完整profile按选择同一选项的被试的各facet分数计算；同域和跨域分别取最大带符号rho的非目标facet。独立条件作答沿用各臂均值；仅作诊断。")
    else:
        st.info("本轮没有可展示的选项 facet 均值。")
    pending = [
        row for row in round_result.get("pending_items") or []
        if isinstance(row, Mapping)
    ]
    st.markdown(f"**待处理题目明细：{len(pending)} 题**")
    for entry in pending:
        failed = "、".join(entry.get("failed_thresholds") or [])
        with st.expander(
            f"{entry.get('item_id')} · 失败门槛：{failed or '无'}",
            expanded=False,
        ):
            _render_round_item_details(entry)


def _render_virtual_content_review_history(state: Mapping[str, Any]) -> None:
    history = [
        row
        for row in state.get("virtual_content_review_history") or []
        if isinstance(row, Mapping)
    ]
    if not history:
        return
    st.markdown("**虚拟内容复审历史**")
    st.caption(
        f"协议：{state.get('virtual_content_review_protocol') or '未记录'}。"
        "以下为虚拟开发证据，不是真人SME评审，也不代表真人内容效度。"
    )
    rows: list[dict[str, Any]] = []
    for entry in history:
        experts = entry.get("expert_reviews")
        expert_count = (
            sum(isinstance(value, Mapping) for value in experts)
            if isinstance(experts, list)
            else "未记录"
        )
        interviews = entry.get("cognitive_interviews")
        if isinstance(interviews, list):
            role_ids = [
                str((value.get("role") or {}).get("role_id"))
                for value in interviews
                if isinstance(value, Mapping)
                and isinstance(value.get("role"), Mapping)
                and value.get("role", {}).get("role_id")
            ]
            interview_count: int | str = (
                len(set(role_ids)) if len(role_ids) == len(interviews) else "未记录"
            )
        else:
            interview_count = "未记录"
        diagnosis = entry.get("diagnosis")
        decision = diagnosis.get("decision") if isinstance(diagnosis, Mapping) else None
        candidate_id = entry.get("candidate_item_id")
        candidate = entry.get("candidate")
        candidate_version = entry.get("new_item_version")
        if candidate_version is None and isinstance(candidate, Mapping):
            candidate_version = candidate.get("version")
        status = entry.get("status") or "未记录"
        candidate_id = candidate_id or (
            "候选ID未记录" if candidate_version is not None else
            "待提交" if status == "ready" else "未生成"
        )
        rows.append(
            {
                "题目": entry.get("reviewed_item_id") or "未记录",
                "所审版本": entry.get("reviewed_item_version") or "未记录",
                "虚拟专家数": expert_count,
                "访谈角色数": interview_count,
                "decision / status": f"{decision or '未记录'} / {status}",
                "候选版本": (
                    f"{candidate_id} v{candidate_version}"
                    if candidate_version is not None
                    else str(candidate_id)
                ),
                "归档路径": entry.get("archive_ref") or "未记录",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_facet_iteration_state(state: Mapping[str, Any]) -> None:
    iteration = state.get("facet_iteration_state")
    if not isinstance(iteration, Mapping):
        return
    with st.expander("Facet开发轮次控制器", expanded=True):
        round_result = state.get("psychometric_round_result") or {}
        st.write(
            f"策略：{iteration.get('policy_version') or '未记录'}；"
            f"状态：{iteration.get('status') or '未记录'}；"
            f"开发轮：{round_result.get('development_round') or iteration.get('development_round') or '未记录'}；"
            f"已完成开发轮：{iteration.get('completed_rounds', 0)}；"
            f"本测量是否完成开发轮：{round_result.get('round_completed', '未记录')}"
        )
        st.caption(
            "主动单题门槛：CITC≥.30、IPIP目标Hedges'g≥.50、目标IPIP rho_s≥.40、Δmin≥.30。"
            "Profile目标rho_s/同域VTS/跨域VTS只作诊断，threshold=None、passes=None。"
            "整卷五项：α、ICC、目标g、目标rho、Δmin；基线首测可直接建立，即使α/ICC未过。"
            "后续成功轮要求α和ICC≥.70，且g/rho/Δmin均较上一完成轮严格上升>1e-12。"
        )
        attempts = iteration.get("local_attempts") or {}
        st.dataframe(
            [
                {
                    "facet": facet_id,
                    "本开发轮局部提交并复测次数": count,
                    "本开发轮上限": "无限",
                    "已接受本轮快照": facet_id in (iteration.get("accepted_facets") or {}),
                    "当前失败": facet_id in (iteration.get("failed_facet_ids") or []),
                }
                for facet_id, count in sorted(attempts.items())
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.write(
            "失败 facet：" + ("、".join(iteration.get("failed_facet_ids") or []) or "无")
            + "；暂停原因：" + str(iteration.get("paused_reason") or "无")
        )
        st.caption(
            f"首轮 cohort 响应引用：{iteration.get('cohort_source_ref') or '未记录'}；"
            f"首轮 IPIP 校标引用：{iteration.get('first_round_reference_ref') or '未记录'}。"
            "首轮后固定被试ID、完整人格分数、人口学变量与模拟设置；IPIP缺失或身份错配时暂停且不得重施。"
            "后续只测改过的题目新版本；未改且未过门槛复用最新完整作答，已过门槛复用资格来源的主测和每个retest作答。"
            "逐题按subject_id/item_id/item_version/administration合并；不拼接整卷总分、不跨版本平均、不复用旧版答案，主测不作重测输入。"
            "每次逻辑失败调用最多5次尝试（含首次，底层重试计入）；每facet每开发轮局部提交并复测次数不设上限。"
        )
        for label, key in (("首轮基线五项整卷指标", "baseline_facets"), ("本轮已接受 facet 快照", "accepted_facets")):
            snapshots = iteration.get(key) or {}
            if not snapshots:
                st.write(f"{label}：尚无")
                continue
            rows = []
            for facet_id, snapshot in sorted(snapshots.items()):
                metrics = snapshot.get("metrics") or {}
                rows.append(
                    {
                        "facet": facet_id,
                        "题数": metrics.get("item_count"),
                        "α": metrics.get("cronbach_alpha"),
                        "α≥.70": (metrics.get("alpha_gate") or {}).get("passed"),
                        "ICC": metrics.get("virtual_test_retest_icc"),
                        "ICC≥.70": (metrics.get("stability_gate") or {}).get("passed"),
                        "Hedges' g": metrics.get("target_hedges_g"),
                        "目标rho": metrics.get("target_spearman_rho"),
                        "Δmin": metrics.get("discriminant_delta_min"),
                        "题目版本": snapshot.get("item_versions"),
                    }
                )
            st.markdown(f"**{label}**")
            st.dataframe(rows, hide_index=True, use_container_width=True)
        completed_rows = [
            {
                "开发轮": row.get("development_round"),
                "基线轮": row.get("baseline_only", False),
                "facet数": len(row.get("facet_forms") or {}),
                "局部次数": row.get("local_attempts"),
            }
            for row in iteration.get("completed_history") or []
            if isinstance(row, Mapping)
        ]
        if completed_rows:
            st.markdown("**已完成轮次记录**")
            st.dataframe(completed_rows, hide_index=True, use_container_width=True)
        if iteration.get("status") == "paused":
            st.warning(iteration.get("paused_reason") or "Facet开发轮次已暂停。")


def _render_virtual_test_statistics(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    screening = value.get("virtual_screening_metrics") or {}
    summary = screening.get("summary") if isinstance(screening, Mapping) else None
    if not isinstance(summary, Mapping):
        return False
    st.markdown("**探索性虚拟筛查：四项单题主动门槛 + 三项 profile 诊断**")
    summary_values = (
        ("CITC中位数", summary.get("median_citc")),
        ("Profile目标ρs（诊断）", summary.get("median_target_rho")),
        ("Profile同域VTS（诊断）", summary.get("minimum_same_domain_vts")),
        ("Profile跨域VTS（诊断）", summary.get("minimum_cross_domain_vts")),
        ("单题Hedges'g中位数", summary.get("median_item_target_hedges_g")),
        ("单题IPIP rho_s中位数", summary.get("median_item_target_ipip_spearman_rho")),
        ("单题Δmin最小值", summary.get("minimum_item_discriminant_delta_min")),
        ("描述性Cronbach α", value.get("cronbach_alpha")),
    )
    for offset in range(0, len(summary_values), 4):
        columns = st.columns(4)
        for column, (label, metric_value) in zip(columns, summary_values[offset:offset + 4]):
            column.metric(label, _metric_text(metric_value))
    st.caption(
        "主动门槛：CITC≥.30、单题IPIP目标Hedges'g≥.50、目标IPIP rho_s≥.40、Δmin≥.30。"
        "Profile目标ρs、同域VTS与跨域VTS仅作诊断，无阈值、无资格判定权。"
        "整卷五项指标为alpha、ICC、目标g、目标rho与Δmin；alpha/ICC门槛均为.70。"
        "难度、选项使用率与Cronbach alpha仅作描述；结果不是正式单题信效度。"
    )
    with st.expander("查看完整分析配置与路径"):
        st.json(dict(value), expanded=False)
    return True


def _render_iteration_history(value: object) -> bool:
    """Render whole-test quality and cost curves across development rounds."""

    if not isinstance(value, list) or not value:
        return False
    rows: list[dict[str, Any]] = []
    facet_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for entry in strict_development_round_entries(value):
        if not isinstance(entry, Mapping):
            continue
        analysis_round = int(entry.get("analysis_round") or 0)
        development_round = entry.get("development_round") or (
            (entry.get("facet_iteration_state") or {}).get("development_round")
            if isinstance(entry.get("facet_iteration_state"), Mapping)
            else None
        )
        round_label = development_round_batch_label(value, entry)
        form_metrics = entry.get("form_metrics") or {}
        reliability = form_metrics.get("reliability") or {}
        validity = form_metrics.get("validity") or {}
        recovery = validity.get("target_recovery") or {}
        selectivity = validity.get("construct_selectivity") or {}
        convergent = validity.get("convergent_validity") or {}
        discriminant = validity.get("discriminant_validity") or {}
        known_groups = validity.get("known_groups_validity") or {}
        quality = form_quality_summary(form_metrics)
        uses_ipip_objective = quality.get("objective_source") in {"ipip_human_style_v3", "ipip_facet_gates_v4"}
        iteration_state = entry.get("facet_iteration_state") or {}
        plateau = entry.get("plateau_status") or {}
        trajectory = plateau.get("trajectory") or []
        controller_comparisons = iteration_state.get("comparisons") or []
        if controller_comparisons:
            raw_comparisons = controller_comparisons
        elif entry.get("facet_gate_comparison"):
            raw_comparisons = entry.get("facet_gate_comparison")
        else:
            raw_comparisons = trajectory[-1].get("facet_gate_comparison") or [] if trajectory else []
        comparisons = {
            str(row.get("sjt_facet_id")): row
            for row in raw_comparisons
            if isinstance(row, Mapping)
        }
        for facet in quality.get("facet_metrics") or []:
            facet_id = str(facet.get("sjt_facet_id"))
            comparison = comparisons.get(facet_id) or {}
            accepted_facets = iteration_state.get("accepted_facets")
            accepted = (
                facet_id in accepted_facets
                if isinstance(accepted_facets, Mapping)
                else comparison.get("passed")
            )
            facet_rows.append({
                "轮次": round_label,
                "开发轮次": development_round,
                "facet": facet_id,
                "题数": facet.get("item_count"),
                "α": facet.get("cronbach_alpha"),
                "ICC": facet.get("virtual_test_retest_icc"),
                "Hedges' g": facet.get("target_hedges_g"),
                "目标rho": facet.get("target_spearman_rho"),
                "Δmin": facet.get("discriminant_delta_min"),
                "α≥.70": (facet.get("alpha_gate") or {}).get("passed"),
                "ICC≥.70": (facet.get("stability_gate") or {}).get("passed"),
                "g升幅": _improvement_value(_comparison_gate(comparison, "hedges_g_improvement_gate")),
                "g升幅通过": (_comparison_gate(comparison, "hedges_g_improvement_gate")).get("passed"),
                "rho升幅": _improvement_value(_comparison_gate(comparison, "target_rho_improvement_gate", "target_rho_noninferiority_gate")),
                "rho升幅通过": _comparison_gate(comparison, "target_rho_improvement_gate", "target_rho_noninferiority_gate").get("passed"),
                "Δmin升幅": _improvement_value(_comparison_gate(comparison, "discriminant_delta_improvement_gate", "discriminant_delta_non_decrease_gate")),
                "Δmin升幅通过": _comparison_gate(comparison, "discriminant_delta_improvement_gate", "discriminant_delta_non_decrease_gate").get("passed"),
                "该facet本轮已接受": accepted,
            })
        raw_item_metrics = (
            entry.get("candidate_item_metrics")
            or entry.get("item_metrics")
            or {}
        )
        if isinstance(raw_item_metrics, Mapping):
            round_item_metrics = [row for row in raw_item_metrics.values() if isinstance(row, Mapping)]
        elif isinstance(raw_item_metrics, list):
            round_item_metrics = [row for row in raw_item_metrics if isinstance(row, Mapping)]
        else:
            round_item_metrics = []
        saved_facet_metrics = entry.get("facet_metrics") or {}
        if isinstance(saved_facet_metrics, Mapping) and saved_facet_metrics:
            facet_by_id = {
                str(facet_id): row
                for facet_id, row in saved_facet_metrics.items()
                if isinstance(row, Mapping)
            }
        else:
            facet_by_id = {
                str(row.get("sjt_facet_id")): row
                for row in quality.get("facet_metrics") or []
                if isinstance(row, Mapping) and row.get("sjt_facet_id") is not None
            }
        for item_metric in round_item_metrics:
            facet_id = str(item_metric.get("facet_id") or "未标注 facet")
            facet = facet_by_id.get(facet_id) or {}
            item_rows.append(
                {
                "轮次": round_label,
                "开发轮次": development_round,
                    "facet": facet_id,
                    "题目": item_metric.get("item_id"),
                    "入选整卷": item_metric.get("selected_for_form"),
                    "CITC": item_metric.get("citc"),
                    "Profile目标rho_s（诊断）": _profile_diagnostic_value(item_metric, "target_rho_diagnostic"),
                    "Profile同域VTS（诊断）": _profile_diagnostic_value(item_metric, "same_domain_vts_diagnostic"),
                    "Profile跨域VTS（诊断）": _profile_diagnostic_value(item_metric, "cross_domain_vts_diagnostic"),
                    "单题Hedges'g": item_metric.get("target_hedges_g"),
                    "单题IPIP rho_s": item_metric.get("target_ipip_spearman_rho"),
                    "单题Δmin": item_metric.get("discriminant_delta_min"),
                    "题目通过": item_metric.get("qualified"),
                    "整卷α": facet.get("cronbach_alpha"),
                    "整卷ICC": facet.get("virtual_test_retest_icc"),
                    "整卷Hedges' g": facet.get("target_hedges_g"),
                    "整卷目标rho": facet.get("target_spearman_rho"),
                    "整卷Δmin": facet.get("discriminant_delta_min"),
                }
            )
        usage = entry.get("token_usage") or {}
        rows.append(
            {
                "轮次": round_label,
                "_analysis_round": analysis_round,
                "开发轮次": development_round,
                "题目数": entry.get("item_count"),
                "候选题数": entry.get("candidate_count"),
                "虚拟重测ICC": reliability.get(
                    "virtual_test_retest_icc"
                ),
                "ICC门槛通过": (quality.get("stability_gate") or {}).get(
                    "passed"
                ),
                "目标恢复R²": recovery.get("cross_validated_r2"),
                "构念选择性": (
                    selectivity.get("value")
                    if selectivity.get("value") is not None
                    else quality.get("construct_selectivity")
                ),
                "IPIP目标facet Spearman rho": (
                    convergent.get("spearman_rho")
                    if uses_ipip_objective
                    else None
                ),
                "目标IPIP Hedges’ g": (
                    known_groups.get("target_hedges_g")
                    if uses_ipip_objective
                    else None
                ),
                "本轮主目标 Hedges’ g": (
                    entry.get("candidate_form_quality")
                    if uses_ipip_objective
                    and entry.get("candidate_form_quality") is not None
                    else quality.get("objective_primary")
                    if uses_ipip_objective
                    else None
                ),
                "Δmin区分效度": (
                    discriminant.get("delta_min")
                    if uses_ipip_objective
                    else None
                ),
                "Cronbach α": (quality.get("alpha_gate") or {}).get("observed"),
                "本轮Token": usage.get("total_tokens"),
                "本轮模型耗时(ms)": usage.get("duration_ms"),
                "模型调用次数": usage.get("calls"),
                "整卷状态": entry.get("form_status"),
                "开发轮状态": iteration_state.get("status") or entry.get("status") or "未记录",
            }
        )
    if not rows:
        return False
    frame = (
        pd.DataFrame(rows)
        .sort_values("_analysis_round")
        .set_index("轮次")
        .drop(columns=["_analysis_round"])
    )
    frame["累计Token"] = pd.to_numeric(
        frame["本轮Token"], errors="coerce"
    ).fillna(0).cumsum()
    frame["累计模型耗时(ms)"] = pd.to_numeric(
        frame["本轮模型耗时(ms)"], errors="coerce"
    ).fillna(0).cumsum()
    st.markdown("**整卷虚拟开发指标迭代曲线**")
    st.caption(
        "每轮先用候选题组成临时测验，再计算整卷指标；这些是虚拟开发期筛查结果，"
        "不能替代真人样本的正式信效度。每facet报告五项整卷指标；首测建立基线，"
        "基线后最多完成两轮成功返修；每轮各facet须满足α/ICC≥.70，且g、rho、Δmin原始值较上一完成轮分别严格上升>1e-12；不使用平台期早停。"
    )
    if facet_rows:
        st.markdown("**各 facet 五项整卷指标及相对上一完成轮的升幅**")
        st.dataframe(pd.DataFrame(facet_rows), hide_index=True, use_container_width=True)
    if item_rows:
        st.markdown("**每开发轮、每 facet、每题：四项主动门槛 + 三项 profile 诊断 + 对应整卷指标**")
        st.dataframe(pd.DataFrame(item_rows), hide_index=True, use_container_width=True)
    else:
        st.info("当前历史记录尚未保存逐题轮次快照；不会用当前轮指标回填历史轮次。")
    primary_columns = [
        column
        for column in ("本轮主目标 Hedges’ g",)
        if column in frame.columns and frame[column].notna().any()
    ]
    if primary_columns:
        st.line_chart(frame[primary_columns], use_container_width=True)
    else:
        st.info("当前轮次尚无可计算的IPIP诊断汇总指标。")
    diagnostic_columns = [
        column
        for column in (
            "目标恢复R²",
            "构念选择性",
            "IPIP目标facet Spearman rho",
            "Δmin区分效度",
        )
        if column in frame.columns and frame[column].notna().any()
    ]
    if diagnostic_columns:
        st.markdown("**原始诊断指标（允许波动）**")
        st.line_chart(frame[diagnostic_columns], use_container_width=True)
    st.markdown("**迭代成本曲线**")
    cost_columns = [
        column
        for column in ("累计Token", "累计模型耗时(ms)")
        if column in frame.columns and frame[column].notna().any()
    ]
    if cost_columns:
        st.line_chart(frame[cost_columns], use_container_width=True)
    else:
        st.info("当前尚无可归属到迭代轮次的 Token 或耗时记录。")
    st.dataframe(frame.reset_index(), hide_index=True, use_container_width=True)
    return True


def _render_virtual_item_statistics(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    rows: list[dict[str, Any]] = []
    detected = False
    for item_id, item in value.items():
        if not isinstance(item, Mapping):
            continue
        quality = item.get("quality_evaluation") or {}
        citc = quality.get("facet_citc") or {}
        specificity = quality.get("virtual_target_specificity") or {}
        if not citc and not specificity:
            continue
        detected = True
        target = specificity.get("rho_target") or specificity.get("target_spearman") or {}
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        rows.append(
            {
                "题目": item_id,
                "CITC": citc.get("r"),
                "目标ρs": target.get("rho"),
                "同域最大污染facet": same_domain.get("facet_name") or same_domain.get("largest_non_target_facet_name"),
                "同域最大带符号相关": same_domain.get("max_non_target_rho") if same_domain.get("max_non_target_rho") is not None else same_domain.get("largest_non_target_conditional_rho"),
                "同域VTS": same_domain.get("specificity_margin"),
                "跨域最大污染facet": cross_domain.get("facet_name") or cross_domain.get("largest_non_target_facet_name"),
                "跨域最大带符号相关": cross_domain.get("max_non_target_rho") if cross_domain.get("max_non_target_rho") is not None else cross_domain.get("largest_non_target_conditional_rho"),
                "跨域VTS": cross_domain.get("specificity_margin"),
                "建议": quality.get("recommendation"),
                "难度(描述)": item.get("difficulty"),
                "有效选项(描述)": quality.get("effective_option_count"),
            }
        )
        for condition_id, condition_metric in (quality.get("per_condition_metrics") or {}).items():
            rows.append(
                {
                    "题目": f"{item_id} / {condition_id}（诊断）",
                    "CITC": (condition_metric or {}).get("citc", {}).get("r"),
                    "rho": (condition_metric or {}).get("rho", {}).get("rho"),
                    "参与过滤": "是" if condition_id == "target" else "否",
                }
            )
    if not detected:
        return False
    st.markdown("**单题总指标与各条件臂诊断**")
    st.dataframe(rows, hide_index=True, use_container_width=True)
    return True


def _render_content(content: Mapping[str, Any]) -> None:
    for field, value in content.items():
        if field == "test_specification":
            _render_specification(value)
        elif field == "current_item":
            _render_item(value)
        elif field == "blueprint":
            _render_blueprint(value)
        elif field == "test_statistics" and _render_virtual_test_statistics(value):
            continue
        elif field == "item_statistics" and _render_virtual_item_statistics(value):
            continue
        elif field == "psychometric_round_result" and isinstance(value, Mapping):
            _render_psychometric_round_result(value)
            continue
        elif field == "psychometric_iteration_history" and _render_iteration_history(value):
            continue
        elif field in {
            "blueprint_review",
            "current_item_review",
            "test_review_result",
        }:
            _render_review(value)
        else:
            label = CONTENT_LABELS.get(field, field.replace("_", " "))
            st.markdown(f"**{label}**")
            _render_generic(value)


def _render_timeline() -> None:
    entries = st.session_state.sjt_timeline
    st.subheader("开发过程")
    visible = [
        entry
        for entry in entries
        if (
            (entry.get("event") or {}).get("event_type") == "failed"
            or (
                entry.get("node") in USER_VISIBLE_TIMELINE_NODES
                and (entry.get("event") or {}).get("event_type")
                in {"completed", "waiting"}
            )
        )
    ]
    if not visible:
        st.info("流程启动后，各阶段的输出会依次出现在这里。")
        return

    for index, entry in enumerate(visible):
        event = entry["event"]
        event_type = event.get("event_type")
        icon = "✅" if event_type == "completed" else "⚠️"
        title = f"{icon} {action_label(event.get('action'))}"
        duration = event.get("duration_ms")
        if isinstance(duration, (int, float)):
            title += f" · {duration / 1000:.1f}s"
        expanded = index >= max(0, len(visible) - 2)
        with st.expander(title, expanded=expanded):
            if event.get("reason"):
                st.caption(event["reason"])
            if event.get("error"):
                st.error(event["error"])
            content = entry.get("content") or {}
            if content:
                _render_content(content)
            else:
                st.write("该步骤已完成，没有新增需要展示的业务内容。")

    if st.session_state.sjt_runtime_events:
        with st.expander("运行细节与重试记录"):
            for event in st.session_state.sjt_runtime_events:
                st.write(f"• {_runtime_message(event)}")


def _submit_decision(decision: dict[str, Any]) -> None:
    _advance(Command(resume=decision))
    st.rerun()


def _render_requirement_decision(payload: Mapping[str, Any]) -> None:
    proposed = payload.get("proposed_update") or {}
    if payload.get("summary"):
        st.write(payload["summary"])
    if payload.get("validation_error"):
        st.error(payload["validation_error"])
    st.markdown("**当前测验规格**")
    candidate_specification = proposed.get("test_specification")
    _render_specification(candidate_specification)
    if isinstance(candidate_specification, Mapping):
        try:
            resolved_profile = resolve_specification_profile(
                candidate_specification
            )
        except ValueError:
            resolved_profile = None
        if resolved_profile is not None:
            st.info(
                "构念库解析："
                f"{resolved_profile['inventory_name']} / "
                f"{resolved_profile['domain_name']} / "
                f"{resolved_profile['selection_level']}，"
                f"{len(resolved_profile['facets'])} 个 facet"
            )
    questions = payload.get("questions") or []
    if questions:
        st.markdown("**还需要你确认**")
        for question in questions:
            text = question.get("text") if isinstance(question, Mapping) else question
            st.write(f"• {text}")
    suggestions = payload.get("suggestions") or []
    if suggestions:
        with st.expander("查看系统建议", expanded=True):
            for suggestion in suggestions:
                if isinstance(suggestion, Mapping):
                    st.write(f"**{suggestion.get('field', '字段')}**")
                    if suggestion.get("reason"):
                        st.caption(suggestion["reason"])

    available = set(payload.get("available_decisions") or [])
    choices = {}
    if "confirm" in available:
        choices["确认规格并继续"] = "confirm"
    if "accept_suggestions" in available:
        choices["接受系统建议"] = "accept_suggestions"
    choices["补充或修改需求"] = (
        "revise" if "confirm" in available else "answer"
    )
    choices["停止任务"] = "stop"

    with st.form("requirement_decision"):
        label = st.radio("下一步", list(choices), horizontal=True)
        feedback = st.text_area(
            "补充说明",
            placeholder="选择补充或修改时，请在这里说明。",
        )
        submitted = st.form_submit_button(
            "提交并继续",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        decision = choices[label]
        if decision in {"answer", "revise"} and not feedback.strip():
            st.error("请填写需要补充或修改的内容。")
            return
        _submit_decision(
            {
                "decision": decision,
                "feedback": feedback.strip() or None,
            }
        )


def _render_mode_selection(payload: Mapping[str, Any]) -> None:
    modes = payload.get("modes") or []
    labels = {
        str(mode.get("label") or mode.get("mode")): mode.get("mode")
        for mode in modes
        if isinstance(mode, Mapping)
    }
    if not labels:
        labels = {"逐题人工确认": "manual", "自动开发并集中查看": "automatic"}
    with st.form("mode_selection"):
        selected = st.radio("题目开发方式", list(labels))
        for mode in modes:
            if isinstance(mode, Mapping) and mode.get("description"):
                st.caption(
                    f"{mode.get('label', mode.get('mode'))}："
                    f"{mode['description']}"
                )
        submitted = st.form_submit_button(
            "采用此模式",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        _submit_decision({"mode": labels[selected]})


def _render_sample_selection(payload: Mapping[str, Any]) -> None:
    recommendations = payload.get("recommendations") or []
    catalog = [
        dict(row)
        for row in payload.get("dimension_catalog") or []
        if isinstance(row, Mapping) and row.get("dimension_id")
    ]
    targets = [row for row in catalog if row.get("required_target")]
    with st.form("sample_selection"):
        st.caption(payload.get("method_note") or "")
        response_temperature = payload.get(
            "response_temperature", DEFAULT_RESPONSE_TEMPERATURE
        )
        demographics_version = payload.get(
            "demographics_version", DEMOGRAPHICS_VERSION
        )
        st.caption(
            f"拟用作答温度：{response_temperature}；人口学库版本：{demographics_version}。"
            "确认并生成样本后冻结；后续不更换cohort或重施IPIP，仅测改过的题目新版本。"
        )
        minimum_sample_size = int(payload.get("minimum_sample_size") or 30)
        custom_size = st.number_input(
            f"首轮虚拟被试人数（后续沿用冻结cohort，必须至少{minimum_sample_size}）",
            min_value=minimum_sample_size,
            max_value=int((payload.get("pool") or {}).get("available_count", 1000)),
            value=max(
                minimum_sample_size,
                int(payload.get("recommended_sample_size") or 100),
            ),
        )
        st.markdown("**本轮目标 facet（由题目自动确定）**")
        for row in targets:
            st.write(
                f"{row['dimension_id']} = "
                f"{row.get('display_label') or row['dimension_id']}"
            )
        st.caption(
            "系统会对每个目标 facet 分别建立三臂：同域另外5个 facet、跨域另外24个 facet；"
            "所有 facet 使用同一组分布参数。"
        )
        score_mean = float(
            st.number_input(
                "所有 facet 共享的正态分布均值",
                min_value=0.1,
                max_value=99.9,
                value=50.0,
                key="shared_facet_mean",
            )
        )
        score_sd = float(
            st.number_input(
                "所有 facet 共享的正态分布 SD",
                min_value=0.1,
                max_value=50.0,
                value=15.0,
                key="shared_facet_sd",
            )
        )
        st.caption("模型请求将全量并发启动。")
        with st.expander("复现设置"):
            seed = st.number_input(
                "随机种子",
                min_value=0,
                value=int(payload.get("default_seed", 7)),
            )
            retries = st.number_input(
                "失败重试次数",
                min_value=0,
                max_value=4,
                value=max(0, min(4, int(payload.get("default_max_retries", 4)))),
            )
        submitted = st.form_submit_button(
            "开始虚拟作答",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        sample_size = int(custom_size)
        errors: list[str] = []
        if not targets:
            errors.append("题库缺少目标 facet")
        if errors:
            st.error("；".join(errors))
            return
        _submit_decision(
            {
                "sample_size_per_condition": sample_size,
                "score_distribution": {
                    "family": "normal",
                    "mean": score_mean,
                    "sd": score_sd,
                },
                "seed": int(seed),
                "max_concurrency": 0,
                "max_retries": int(retries),
            }
        )


def _render_post_virtual_response_decision(payload: Mapping[str, Any]) -> None:
    st.write(payload.get("summary") or "")
    round_result = payload.get("round_result") or {}
    if isinstance(round_result, Mapping) and round_result:
        _render_psychometric_round_result(round_result)
    else:
        st.warning("本轮缺少统一结果结构，请重新运行心理测量分析。")
    iteration_history = payload.get("psychometric_iteration_history") or []
    if iteration_history:
        st.markdown("**本轮临时组卷（单题返修前基线）**")
        _render_iteration_history(iteration_history)
    st.caption("正式题资格一经锁定不撤销，监测警告不会触发返修。")
    diagnostics = payload.get("condition_score_diagnostics") or {}
    correlation_rows = []
    for condition in diagnostics.get("conditions") or []:
        if not isinstance(condition, Mapping):
            continue
        correlation_rows.append({"范围": str(condition.get("condition_id")), "均值": condition.get("actual_mean"), "SD": condition.get("actual_sample_sd")})
    with st.expander("补充：完整 facet 分数分布（不参与过滤）"):
        st.caption("每名虚拟被试携带完整 facet 分数向量；所有 facet 共享均值/SD，但每个 facet 使用独立分数向量；分布核查不具备 filtering authority。")
        if correlation_rows:
            st.dataframe(correlation_rows, hide_index=True, use_container_width=True)
        else:
            st.info("没有可报告的维度对。")
    with st.form("post_virtual_response_decision"):
        selected = st.radio("下一步", ["开始处理未通过题目", "暂停并保存"])
        submitted = st.form_submit_button(
            "提交决策", type="primary", use_container_width=True
        )
    if submitted:
        _submit_decision(
            {"decision": "start" if selected == "开始处理未通过题目" else "stop"}
        )


def _render_psychometric_repair_confirmation(payload: Mapping[str, Any]) -> None:
    """Render one complete diagnosis and its allowed user disposition."""

    diagnosis = payload.get("diagnosis") or {}
    item = payload.get("item")
    queue = payload.get("pending_item_queue") or []
    st.markdown(
        f"**题目 {payload.get('item_id', '?')} · 第 {payload.get('revision_round', '?')} 轮 · "
        f"队列 1/{max(1, len(queue))}**"
    )
    if payload.get("queue_status") == "deferred_decision":
        st.warning(
            "当前 defer 诊断将自动生成同槽位补题。"
            "不再请求人工确认；新题仍需经过完整审题、施测和单题指标筛选。"
        )
        return
    observations = [
        row
        for row in payload.get("observations") or []
        if isinstance(row, Mapping) and row.get("role") != "descriptive_only"
    ]
    if observations:
        st.markdown("**四项主动门槛、三项Profile诊断与最大污染 facet**")
        st.dataframe(
            [
                {
                    "指标": row.get("metric"),
                    "值": row.get("value"),
                    "阈值": row.get("threshold"),
                    "污染facet": row.get("facet_name") or row.get("dimension_id"),
                    "条件rho": row.get("signed_rho"),
                }
                for row in observations
            ],
            hide_index=True,
            use_container_width=True,
        )
    target_constraints = payload.get("target_construct_constraints") or []
    if target_constraints:
        st.markdown("**目标 facet 的构念约束**")
        for constraint in target_constraints:
            if isinstance(constraint, Mapping):
                st.write(
                    f"{constraint.get('constraint_id', '?')}："
                    f"{constraint.get('statement', constraint.get('text', ''))}"
                )
    constraints = payload.get("non_target_construct_constraints") or []
    if constraints:
        st.markdown("**最大污染 facet 的定义与高低行为边界**")
        for constraint in constraints:
            if isinstance(constraint, Mapping):
                st.write(
                    f"{constraint.get('constraint_id', '?')}："
                    f"{constraint.get('statement', '')}"
                )
    if isinstance(item, Mapping):
        st.markdown("**题面与选项**")
        _render_item(item)
    agent_comparisons = payload.get("option_score_comparisons") or []
    if isinstance(agent_comparisons, list) and agent_comparisons:
        st.markdown("**VTS 按 option_id 对齐的均值定位证据**")
        st.caption("只用于定位污染候选，不参与资格过滤。")
        st.dataframe(
            _agent_option_score_comparison_display_rows(agent_comparisons),
            hide_index=True,
            use_container_width=True,
        )
    if isinstance(diagnosis, Mapping) and diagnosis.get("summary"):
        st.markdown("**诊断摘要**")
        st.write(diagnosis["summary"])
    candidates = diagnosis.get("candidate_diagnoses") if isinstance(diagnosis, Mapping) else []
    if candidates:
        with st.expander("文本证据与诊断候选", expanded=True):
            st.json(candidates, expanded=False)
    tasks = diagnosis.get("repair_tasks") if isinstance(diagnosis, Mapping) else []
    if tasks:
        st.markdown("**原子修改任务**")
        st.json(tasks, expanded=False)

    is_repair = isinstance(diagnosis, Mapping) and diagnosis.get("decision") == "repair"
    form_key = f"psychometric_repair_confirmation_{payload.get('item_id')}_{payload.get('revision_round')}"
    with st.form(form_key):
        available = set(payload.get("available_decisions") or [])
        if is_repair:
            choices = {
                label: decision
                for label, decision in (
                    ("确认并自动执行原子返修", "approve"),
                    ("暂停并保存", "stop"),
                )
                if not available or decision in available
            }
        else:
            choices = {
                label: decision
                for label, decision in (
                    ("人工修改", "manual_edit"),
                    ("保留待 SME 审核", "pending_sme"),
                    ("淘汰补题", "eliminate_replenish"),
                    ("暂停并保存", "stop"),
                )
                if not available or decision in available
            }
        selected = st.radio("处置方式", list(choices))
        scenario = None
        option_texts: dict[str, str] = {}
        if not is_repair and isinstance(item, Mapping):
            st.caption("选择人工修改时，仅情境与选项文本会被提交；其余字段保持锁定。")
            scenario = st.text_area("情境", value=str(item.get("scenario") or ""))
            for option in item.get("response_options") or []:
                if isinstance(option, Mapping):
                    option_id = str(option.get("option_id") or "")
                    option_texts[option_id] = st.text_area(
                        f"选项 {option_id}", value=str(option.get("text") or "")
                    )
        submitted = st.form_submit_button(
            "提交决策", type="primary", use_container_width=True
        )
    if submitted:
        decision = choices[selected]
        response: dict[str, Any] = {"decision": decision}
        if decision == "manual_edit":
            response["manual_item"] = {
                "scenario": scenario,
                "response_options": [
                    {"option_id": option_id, "text": text}
                    for option_id, text in option_texts.items()
                ],
            }
        _submit_decision(response)


def _render_generic_approval(payload: Mapping[str, Any]) -> None:
    if payload.get("summary"):
        st.write(payload["summary"])
    if payload.get("validation_error"):
        st.error(payload["validation_error"])
    proposed = payload.get("proposed_update") or {}
    action = str(payload.get("action") or "unknown")
    content = {}
    if isinstance(proposed, Mapping):
        content = extract_action_content(action, proposed)
        if not content:
            content = dict(proposed)
    if content:
        _render_content(content)

    choices = {
        "通过并继续": "approve",
        "提出修改意见": "regenerate",
        "停止任务": "stop",
    }
    with st.form("generic_approval"):
        selected = st.radio("你的决定", list(choices), horizontal=True)
        feedback = st.text_area(
            "修改意见",
            placeholder="需要重新生成时，请说明具体要改什么。",
        )
        submitted = st.form_submit_button(
            "提交决定",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        decision = choices[selected]
        if decision == "regenerate" and not feedback.strip():
            st.error("请填写具体修改意见。")
            return
        _submit_decision(
            {
                "decision": decision,
                "feedback": feedback.strip() or None,
                "state_patch": None,
            }
        )


def _render_plateau_gap_decision(payload: Mapping[str, Any]) -> None:
    gap_cells = [
        dict(row)
        for row in payload.get("gap_cells") or []
        if isinstance(row, Mapping)
    ]
    st.markdown(str(payload.get("summary") or ""))
    any_sme_offered = False
    unresolvable = False
    for cell in gap_cells:
        candidates = [
            dict(row)
            for row in cell.get("candidates") or []
            if isinstance(row, Mapping)
        ]
        eligible = [row for row in candidates if row.get("eligible")]
        sme = [row for row in candidates if row.get("force_allowed")]
        any_sme_offered = any_sme_offered or bool(sme)
        if not eligible and not sme:
            unresolvable = True
    if unresolvable:
        st.error(
            "存在没有可处置候选的缺口单元（候选均已淘汰）："
            "请先人工处理，或暂停保存。"
        )
        if st.button(
            "暂停保存",
            type="secondary",
            use_container_width=True,
        ):
            _submit_decision({"decision": "stop"})
        return
    if any_sme_offered:
        st.warning(
            "部分候选仍在待 SME。选择它们将按『开发版强制补位』收卷："
            "报告会标注该题未经专家审、以开发版证据进入正式卷。"
        )
    with st.form("plateau_gap_decision"):
        choices: list[dict] = []
        for index, cell in enumerate(gap_cells):
            cell_id = str(cell.get("blueprint_cell_id") or f"cell{index}")
            candidates = [
                dict(row)
                for row in cell.get("candidates") or []
                if isinstance(row, Mapping)
            ]
            eligible = [row for row in candidates if row.get("eligible")]
            sme = [row for row in candidates if row.get("force_allowed")]
            st.markdown(
                f"**缺口单元 {cell_id}**"
                f"（需保留 {cell.get('planned_retention_count')} 题）"
            )
            labels = {}
            for row in eligible:
                gates = "、".join(row.get("failed_gates") or []) or "无"
                labels[
                    f"{row.get('item_id')} v{row.get('version')}"
                    f"（未过：{gates}）"
                ] = (row, False)
            for row in sme:
                if row.get("item_id") in {
                    r.get("item_id") for r, _ in labels.values()
                }:
                    continue
                labels[
                    f"{row.get('item_id')} v{row.get('version')}"
                    f"（待SME · 强制补位）"
                ] = (row, True)
            if not labels:
                st.warning("无可用候选")
                continue
            default_label = next(iter(labels))
            chosen = st.selectbox(
                f"{cell_id} 候选",
                list(labels),
                key=f"pg_cand_{index}",
            )
            row, is_sme = labels[chosen]
            mode = st.radio(
                f"{cell_id} 处理方式",
                ["直接补位", "手动修改"],
                key=f"pg_mode_{index}",
                horizontal=True,
            )
            manual_item = None
            if mode == "手动修改":
                scenario = st.text_area(
                    f"{cell_id} 新情境",
                    value=str(row.get("scenario") or ""),
                    key=f"pg_scenario_{index}",
                )
                option_texts = {}
                for opt in row.get("response_options") or []:
                    if not isinstance(opt, Mapping):
                        continue
                    option_id = str(opt.get("option_id") or "")
                    option_texts[option_id] = st.text_area(
                        f"选项 {option_id}",
                        value=str(opt.get("text") or ""),
                        key=f"pg_opt_{index}_{option_id}",
                    )
                manual_item = {
                    "scenario": scenario,
                    "response_options": [
                        {
                            "option_id": option_id,
                            "text": text,
                        }
                        for option_id, text in option_texts.items()
                    ],
                }
            choices.append(
                {
                    "cell_id": cell_id,
                    "item_id": str(row["item_id"]),
                    "mode": "pick" if mode == "直接补位" else "manual",
                    "manual_item": manual_item,
                    "sme_override": is_sme,
                }
            )
        submitted = st.form_submit_button(
            "确认处置并收卷",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        resolutions = []
        for choice in choices:
            resolutions.append(
                {
                    "cell_id": choice["cell_id"],
                    "item_id": choice["item_id"],
                    "mode": choice["mode"],
                    "sme_override": choice["sme_override"],
                    **({"manual_item": choice["manual_item"]}
                      if choice["manual_item"] is not None
                      else {}),
                }
            )
        _submit_decision({"decision": "resolve", "resolutions": resolutions})


def _render_repair_knowledge_selection(payload: Mapping[str, Any]) -> None:
    labels = {"shared": "共享历史知识并继续积累", "run_only": "仅本任务积累"}
    mode = st.radio("返修知识范围", options=list(labels), format_func=labels.get,
                    index=None, key="repair_knowledge_mode")
    if st.button("确认", key="repair_knowledge_confirm", disabled=mode is None):
        _submit_decision({"mode": mode})


def _render_scenario_repair_pause(payload: Mapping[str, Any]) -> None:
    st.warning(payload.get("message") or "情境检测返修已暂停，整批尚未提交。")
    knowledge = payload.get("knowledge") or {}
    if knowledge.get("status") == "paused":
        st.error(f"知识归纳：{knowledge.get('stage')}；{knowledge.get('error')}")
        st.caption(str(knowledge.get("journal_ref") or ""))
    rows = []
    for item_id, progress in (payload.get("progress") or {}).items():
        rows.append({"题目": item_id, "阶段": progress.get("stage"),
                     "情境改写次数": progress.get("rewrite_count", 0),
                     "原因": progress.get("pause_reason"), "存档": progress.get("archive_ref")})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("自动恢复完成后继续；额度耗尽则自动保存退出，不重复询问补做。")
    _submit_decision({"decision": "stop" if payload.get("recovery_version") == 2 else "retry"})


def _render_interrupt() -> None:
    payload = st.session_state.sjt_interrupt
    if not isinstance(payload, Mapping):
        return
    st.subheader("需要你的确认")
    with st.container(border=True):
        interaction_type = payload.get("type")
        if interaction_type == "repair_knowledge_selection":
            _render_repair_knowledge_selection(payload)
        elif interaction_type == "requirement_confirmation":
            _render_requirement_decision(payload)
        elif interaction_type == "item_development_mode_selection":
            _render_mode_selection(payload)
        elif interaction_type == "virtual_sample_selection":
            _render_sample_selection(payload)
        elif interaction_type == "post_virtual_response_decision":
            _render_post_virtual_response_decision(payload)
        elif interaction_type == "psychometric_repair_confirmation":
            _render_psychometric_repair_confirmation(payload)
        elif interaction_type == "scenario_repair_pause":
            _render_scenario_repair_pause(payload)
        elif interaction_type == "plateau_gap_decision":
            _render_plateau_gap_decision(payload)
        else:
            _render_generic_approval(payload)


def _download_json(label: str, value: object, filename: str) -> None:
    st.download_button(
        label,
        data=json.dumps(value, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def _render_deliverables(state: Mapping[str, Any]) -> None:
    final_test = state.get("final_test")
    technical = state.get("technical_report")
    virtual = state.get("virtual_respondent_report")
    if not any((final_test, technical, virtual)):
        return
    st.subheader("最终交付")
    tabs = st.tabs(["正式测验", "技术报告", "虚拟被试报告"])
    with tabs[0]:
        if final_test:
            _render_generic(final_test)
            _download_json("下载正式测验", final_test, "final_test.json")
        else:
            st.info("尚未生成。")
    with tabs[1]:
        if technical:
            _render_generic(technical)
            _download_json(
                "下载技术报告",
                technical,
                "technical_report.json",
            )
        else:
            st.info("尚未生成。")
    with tabs[2]:
        if virtual:
            _render_generic(virtual)
            _download_json(
                "下载虚拟被试报告",
                virtual,
                "virtual_respondent_report.json",
            )
        else:
            st.info("尚未生成。")


def _render_active_page(state: Mapping[str, Any]) -> None:
    summary = progress_summary(state)
    top_left, top_right = st.columns([1, 0.25])
    with top_left:
        st.markdown(
            '<div class="sjt-kicker">ACTIVE DEVELOPMENT</div>',
            unsafe_allow_html=True,
        )
        st.title("测验开发工作台")
        st.caption(state.get("user_request", ""))
    with top_right:
        if st.button("新建任务", use_container_width=True):
            _reset_session()
            st.rerun()

    columns = st.columns(4)
    columns[0].metric("当前阶段", summary["phase"])
    columns[1].metric("已执行步骤", summary["steps"])
    columns[2].metric("候选题", summary["candidate_items"])
    columns[3].metric(
        "最终入选",
        summary["selected_items"] or "—",
    )

    if st.session_state.sjt_error:
        st.error(st.session_state.sjt_error)
        if st.button("从当前状态重试", type="primary"):
            _retry_current_run(state)
            st.rerun()

    _render_interrupt()
    _render_facet_iteration_state(state)
    _render_virtual_content_review_history(state)
    _render_virtual_respondent_summary(state)
    _render_deliverables(state)
    st.divider()
    _render_timeline()

    with st.sidebar:
        st.header("任务概览")
        st.write(f"**阶段**：{summary['phase']}")
        st.write(f"**状态**：{summary['status']}")
        st.write(f"**运行编号**：`{state.get('run_id', '—')}`")
        completion = state.get("completion_checks") or {}
        if completion:
            passed = sum(bool(value) for value in completion.values())
            st.progress(
                passed / len(completion),
                text=f"完成条件 {passed}/{len(completion)}",
            )
        unmet = state.get("unmet_completion_conditions") or []
        if unmet:
            with st.expander("尚未满足的完成条件"):
                for condition in unmet:
                    st.write(f"• {condition}")
        with st.expander("当前完整状态（调试）"):
            st.json(dict(state), expanded=False)


def _render_virtual_respondent_summary(state: Mapping[str, Any]) -> None:
    config = state.get("virtual_sample_config") or {}
    respondents = [
        row for row in state.get("virtual_respondents") or []
        if isinstance(row, Mapping)
    ]
    if not config and not respondents:
        return

    with st.expander("虚拟被试人口学资料", expanded=False):
        temperature = config.get("response_temperature")
        version = config.get("demographics_version")
        st.write(
            f"作答温度：{temperature if temperature is not None else '旧协议/未冻结'}；"
            f"人口学库版本：{version or '旧协议/未冻结'}；"
            f"被试数：{len(respondents)}"
        )
        if not respondents:
            if config:
                st.warning("当前配置未包含被试资料；请重新配置虚拟样本。")
            return

        snapshot = config.get("demographics_snapshot")
        if not isinstance(snapshot, Mapping) or any(
            not isinstance(row.get("demographics"), Mapping)
            for row in respondents
        ):
            st.warning(
                "旧协议检查点缺少冻结人口学数据库或完整被试资料，需重新配置；"
                "未推断或补造人口学信息。"
            )
            return

        try:
            records = [
                {
                    "被试ID": row.get("respondent_id"),
                    **demographics_to_columns(row["demographics"], snapshot),
                }
                for row in respondents
            ]
        except ValueError as exc:
            st.warning(f"冻结人口学资料无法校验，需重新配置：{exc}")
            return
        frame = pd.DataFrame(records).rename(
            columns={
                "age": "年龄",
                "gender": "性别",
                "nationality": "国籍",
                "education": "学历",
                "occupation": "职业",
                "monthly_income_cny": "税前月收入（CNY，合成）",
            }
        )
        st.caption(
            "收入由冻结的合成规则生成，不代表观察到的真实收入或人口收入分布。"
        )
        st.dataframe(frame, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="SJT 测验开发工作台",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_page_style()
    _initialize_session()
    state = st.session_state.sjt_state
    if isinstance(state, Mapping):
        from sjt_system.evaluation.psychometrics import refresh_saved_option_diagnostics

        try:
            refreshed = refresh_saved_option_diagnostics(state)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if refreshed.get("item_statistics") is not state.get("item_statistics"):
            st.session_state.sjt_state = refreshed
            st.session_state.sjt_interrupt = None
            st.session_state.sjt_workflow = None
            st.session_state.sjt_diagnostics_reconnect = True
            state = refreshed
        if st.session_state.sjt_diagnostics_reconnect:
            st.warning("旧选项诊断已更新。继续处理前需从当前运行状态重建流程。")
            if st.button("继续运行", type="primary", key="reconnect_refreshed_diagnostics"):
                resumed = prepare_resumed_state(
                    state,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                )
                st.session_state.sjt_state = resumed
                st.session_state.sjt_workflow = build_sjt_graph()
                st.session_state.sjt_diagnostics_reconnect = False
                _advance(resumed)
                st.rerun()
        _render_active_page(state)
    else:
        _render_start_page()
