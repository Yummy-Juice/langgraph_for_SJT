"""Legacy command-line interface for the SJT workflow."""

import asyncio
import math
import os
from pathlib import Path
from pprint import pprint
from time import monotonic
from typing import Callable

from langgraph.types import Command

from sjt_system.workflow.graph import build_sjt_graph
from sjt_system.runtime.progress import progress_callback
from sjt_system.runtime.checkpoint import (
    DEFAULT_CHECKPOINT_ROOT,
    find_latest_resumable_checkpoint,
    prepare_retry_state,
    prepare_resumed_state,
    save_run_checkpoint,
)
from sjt_system.runtime.telemetry import run_context as telemetry_run_context
from sjt_system.state import TraceEvent, create_initial_state
from sjt_system.authoring.items import derive_item_review_decision
from sjt_system.authoring.construct_registry import (
    resolve_specification_profile,
)
from sjt_system.authoring.generation_plan import (
    planned_generation_count,
    planned_retention_count,
)
from sjt_system.evaluation.round_results import (
    GATE_ORDER,
    build_psychometric_round_result,
    metric_scalar,
)
from sjt_system.evaluation.form_metrics import form_quality_summary
from sjt_system.delivery.reporting import (
    development_round_batch_label,
    strict_development_round_entries,
)
from sjt_system.evaluation.demographics import (
    DEFAULT_RESPONSE_TEMPERATURE,
    DEMOGRAPHICS_VERSION,
    demographics_to_columns,
)
from sjt_system.config import (
    DEFAULT_REQUESTED_ITEM_COUNT,
    DEFAULT_TARGET_CONSTRUCT,
    DEFAULT_TARGET_POPULATION,
    DEFAULT_USER_REQUEST,
)

app = build_sjt_graph()


SPECIFICATION_LABELS = {
    "construct_selection": "测量构念",
    "target_population": "目标人群",
    "facet_item_counts": "各 facet 题数",
    "final_item_count": "最终题量",
    "output_language": "输出语言",
}

ACTION_LABELS = {
    "clarify_requirements": "需求澄清",
    "build_blueprint": "构念—题目细目表",
    "generate_items_batch": "并发生成全部题目",
    "generate_item": "生成题目",
    "review_item": "审查题目",
    "revise_item": "定向修改题目",
    "regenerate_item": "重写题目",
    "simulate_responses": "虚拟被试作答",
    "analyze_psychometrics": "心理测量分析",
    "select_items": "心理测量返修诊断",
    "psychometric_repair_batch": "并发心理测量返修",
    "assemble_test": "测验组卷",
    "review_test": "测验整体审核",
    "rescore_test": "重新计分",
    "generate_reports": "生成测验与报告",
    "finish": "完成工作流",
}

ITEM_OUTPUT_ACTIONS = {
    "generate_item": "生成",
    "revise_item": "定向修改",
    "regenerate_item": "重写",
}

ACTION_START_NODES = {
    "router",
    "prepare_item_review",
    "prepare_item_revision",
}


def action_label(action: object) -> str:
    value = str(action or "unknown")
    return ACTION_LABELS.get(value, value)


def item_stage_label(action: str, state: dict) -> str:
    if action == "review_item":
        return (
            "复审题目"
            if state.get("current_item_repair_attempted")
            else "首次审题"
        )
    return action_label(action)


def print_runtime_progress(event: dict) -> None:
    """Render transient model and simulation progress events."""

    event_type = event.get("type")
    if event_type == "simulation_stage":
        status = {
            "started": "开始",
            "completed": "完成",
            "failed": "失败",
        }.get(event.get("status"), event.get("status"))
        print(
            f"\n[虚拟作答] {event.get('stage')} {status}"
            f"（本轮调用 {event.get('total', '?')} 次）",
            flush=True,
        )
    elif event_type == "simulation_progress":
        completed = event.get("completed", 0)
        total = event.get("total", 0)
        percent = 100 if total == 0 else round(completed / total * 100)
        print(
            f"[虚拟作答] {event.get('stage')}："
            f"{completed}/{total}（{percent}%）",
            flush=True,
        )
    elif event_type == "psychometric_subagent_progress":
        status = {
            "batch_started": "批次启动",
            "batch_completed": "批次完成",
            "started": "启动",
            "editing": "修改中",
            "retesting": "局部复测中",
            "round_completed": "本轮完成",
            "completed": "完成",
            "failed": "失败",
            "running": "运行中",
        }.get(event.get("status"), event.get("status"))
        position = ""
        if event.get("queue_position") and event.get("queue_total"):
            position = (
                f" [{event.get('queue_position')}/{event.get('queue_total')}]"
            )
        round_text = ""
        if event.get("round"):
            round_text = (
                f"；模型调用轮次 {event.get('round')}/"
                f"{event.get('max_rounds', '?')}"
            )
        gate_text = ""
        if event.get("passed_gate_count") is not None:
            gate_text = (
                f"；通过门槛 {event.get('passed_gate_count')}/"
                f"{event.get('gate_total', 4)}"
            )
        elapsed_text = ""
        if event.get("elapsed_ms") is not None:
            elapsed_text = f"；耗时 {event.get('elapsed_ms')} ms"
        details = str(event.get("message") or "")
        if event.get("diagnosis_id"):
            details += f"；诊断={event.get('diagnosis_id')}"
        if event.get("failed_gates"):
            details += "；未通过=" + ",".join(
                str(value) for value in event.get("failed_gates") or []
            )
        if event.get("local_status"):
            details += f"；局部状态={event.get('local_status')}"
        if event.get("batch_total") is not None:
            details += (
                f"；批次进度={event.get('completed_count', 0)}/"
                f"{event.get('batch_total')}"
            )
        if event.get("active_count") is not None:
            details += f"；进行中={event.get('active_count')}"
        if event.get("stage"):
            details += f"；阶段={event.get('stage')}"
        if event.get("concurrency") is not None or event.get("concurrency_policy") == "all_at_once":
            details += "；并发策略=全量并发"
        print(
            f"[心理测量 subagent]{position} "
            f"{event.get('item_id', 'unknown')}：{status}"
            f"{round_text}{gate_text}{elapsed_text}"
            + (f"；{details}" if details else ""),
            flush=True,
        )
    elif event_type in {"request_retry", "output_repair"}:
        print(
            f"\n[重试] {event.get('job_label', '模型请求')}："
            f"第 {event.get('attempt', '?')}/"
            f"{event.get('max_attempts', '?')} 次；"
            f"原因：{event.get('reason', '未知')}",
            flush=True,
        )
    elif event_type == "action_fallback":
        print(
            f"\n[升级处理] {action_label(event.get('from_action'))}"
            f"连续 {event.get('failed_attempts', '?')} 次未通过校验，"
            f"已自动转为{action_label(event.get('to_action'))}；"
            f"原因：{event.get('reason', '未知')}",
            flush=True,
        )
    elif event_type == "request_timeout":
        print(
            f"\n[超时] {event.get('job_label', '模型请求')} 超过 "
            f"{event.get('timeout_seconds', '?')} 秒，已取消。",
            flush=True,
        )
    elif event_type == "skeleton_slot_failed":
        print(
            "\n[骨架槽位失败] 当前骨架未通过程序确定性校验，当前槽位将被拒绝，"
            "工作流继续处理下一固定槽位。",
            flush=True,
        )
        if event.get("final_reason"):
            print(f"最终原因：{event['final_reason']}", flush=True)


def print_automatic_item_result(
    action: str,
    update: dict,
    state: dict,
    event: dict | None = None,
) -> None:
    """Show automatic-mode item outputs without adding approval pauses."""

    if state.get("item_development_mode") != "automatic":
        return
    proposed_update = update.get("pending_state_update") or {}
    if action in ITEM_OUTPUT_ACTIONS:
        print_skeleton_candidate(proposed_update, state)
        print(f"\n===== 题目已{ITEM_OUTPUT_ACTIONS[action]} =====")
        print_item_candidate(proposed_update.get("current_item"))
    elif action == "review_item":
        # Some stream updates expose the committed state directly while the
        # nested pending proposal is omitted.  The review event has already
        # recorded the review in its state diff; use that value for display
        # instead of reporting a spurious missing current_item_review.
        if not isinstance(proposed_update.get("current_item_review"), dict):
            trace_review = (
                ((event or {}).get("state_changes") or {})
                .get("current_item_review", {})
                .get("after")
            )
            review_from_trace = isinstance(trace_review, dict)
            state_review = trace_review
            if not review_from_trace:
                state_review = state.get("current_item_review")
            if isinstance(state_review, dict):
                proposed_update = {
                    **proposed_update,
                    "current_item_review": state_review,
                    "current_item": state.get("current_item"),
                    "_review_from_trace_summary": review_from_trace,
                }
        print("\n===== 题目审查结果 =====")
        print_review_candidate(proposed_update)


def _format_metric(value: object, *, digits: int = 3) -> str:
    numeric = metric_scalar(value)
    if numeric is None:
        return "证据不足"
    return f"{numeric:.{digits}f}"


def _format_option_mean(value: object, count: object) -> str:
    return "无人选择" if count == 0 else _format_metric(value)


def _cli_gate_display(gate: dict) -> str:
    if gate.get("threshold") is None or gate.get("passes") is None:
        return f"{_format_metric(gate.get('value'))} / 仅诊断"
    status = "通过" if gate.get("passes") is True else (
        "未通过" if gate.get("estimable") is True else "不可估计"
    )
    return (
        f"{_format_metric(gate.get('value'))}/"
        f"≥{_format_metric(gate.get('threshold'))}/{status}"
    )


def _profile_diagnostic_value(entry: dict, diagnostic_id: str) -> object:
    for row in entry.get("profile_diagnostics") or []:
        if isinstance(row, dict) and row.get("diagnostic_id") == diagnostic_id:
            return row.get("value")
    legacy_gate_id = {
        "target_rho_diagnostic": "target_rho_pass",
        "same_domain_vts_diagnostic": "same_domain_vts_pass",
        "cross_domain_vts_diagnostic": "cross_domain_vts_pass",
    }.get(diagnostic_id)
    if legacy_gate_id:
        for row in entry.get("gates") or []:
            if isinstance(row, dict) and row.get("gate_id") == legacy_gate_id:
                return row.get("value")
    return None


def _cli_frequency_display(option: dict, group_n: object) -> str:
    rate = option.get("selection_rate")
    count = option.get("selection_count")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return f"-({count or 0}/{group_n or 0})"
    return f"{float(rate) * 100:.1f}%({count}/{group_n})"


def _print_option_choice_diagnostics(diagnostics: dict) -> None:
    aggregate = diagnostics.get("aggregate") or diagnostics.get("all") or {}
    option_ids = [
        str(row.get("option_id"))
        for row in aggregate.get("options") or []
        if isinstance(row, dict)
    ]
    if not option_ids:
        print("选项选择频率：无可用数据")
        return
    print("选项选择频率（仅用于定位文本核查点，不参与过滤）：")
    print("| 分组 | N | 可用于定位 | " + " | ".join(option_ids) + " |")
    print("|---|---:|---|" + "---:|" * len(option_ids))

    def print_group(label: str, group: dict, estimable: bool) -> None:
        options = {
            str(row.get("option_id")): row
            for row in group.get("options") or []
            if isinstance(row, dict)
        }
        values = [
            _cli_frequency_display(options.get(option_id) or {}, group.get("group_n"))
            for option_id in option_ids
        ]
        print(
            f"| {label} | {group.get('group_n', 0)} | "
            f"{'是' if estimable else '否'} | " + " | ".join(values) + " |"
        )

    print_group("全样本", aggregate, True)
    for condition in diagnostics.get("by_condition") or []:
        if isinstance(condition, dict):
            print_group(
                f"条件 {condition.get('condition_id')}",
                condition,
                True,
            )


def _print_option_score_comparisons(rows: list[dict]) -> None:
    if not rows:
        print("按选项对齐的设定分数均值：无可用数据")
        return
    print("按 option_id 对齐的设定分数均值（仅诊断，不参与过滤）：")
    print("| 题目 | 选项 | 计分 | 目标N | 目标facet均值 | 同域最大rho facet | 同域N | 同域facet均值 | 跨域最大rho facet | 跨域N | 跨域facet均值 |")
    print("|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|")
    for row in rows:
        if not isinstance(row, dict):
            continue
        print(
            f"| {row.get('item_id')} | {row.get('option_id')} | "
            f"{_format_metric(row.get('option_score', row.get('score')))} | "
            f"{row.get('target_n', 0)} | {_format_option_mean(row.get('target_mean_score'), row.get('target_n'))} | "
            f"{row.get('same_domain_facet_name') or row.get('same_domain_dimension_id') or '-'} | "
            f"{row.get('same_domain_n', 0)} | {_format_option_mean(row.get('same_domain_mean_score'), row.get('same_domain_n'))} | "
            f"{row.get('cross_domain_facet_name') or row.get('cross_domain_dimension_id') or '-'} | "
            f"{row.get('cross_domain_n', 0)} | {_format_option_mean(row.get('cross_domain_mean_score'), row.get('cross_domain_n'))} |"
        )


def _print_diagnosis_option_score_comparisons(rows: list[dict]) -> None:
    if not rows:
        return
    print("VTS 按 option_id 对齐的均值定位证据：")
    print("| VTS类别 | 最大rho非目标facet | 选项 | 计分 | 目标均值 | 对应非目标均值 |")
    print("|---|---|---|---:|---:|---:|")
    for row in rows:
        if not isinstance(row, dict):
            continue
        category = str(row.get("vts_category") or "")
        print(
            f"| {category or '-'} | {row.get('non_target_facet_name') or row.get('non_target_dimension_id') or '-'} | {row.get('option_id')} | "
            f"{_format_metric(row.get('score'))} | "
            f"{_format_option_mean(row.get('target_mean_score'), row.get('target_n'))} | "
            f"{_format_option_mean(row.get(f'{category}_mean_score'), row.get('non_target_n'))} |"
        )


def print_psychometric_round_result(
    round_result: dict,
    *,
    history: list[dict] | None = None,
) -> None:
    summary = round_result.get("summary") or {}
    iteration = round_result.get("facet_iteration_state") or {}
    development_round = round_result.get("development_round") or (
        iteration.get("development_round") if isinstance(iteration, dict) else None
    )
    # Internal measurement batches are intentionally silent. When history is
    # available, the selector returns at most the three protocol rounds.
    if isinstance(history, list) and history:
        displayed = {
            str(entry.get("analysis_round"))
            for entry in strict_development_round_entries(history, max_rounds=3)
            if isinstance(entry, dict)
        }
        if str(round_result.get("analysis_round")) not in displayed:
            return
    elif development_round != 1 and round_result.get("round_completed") is not True:
        return
    public_label = (
        development_round_batch_label(history, round_result)
        if isinstance(history, list) and history
        else f"第{development_round or '?'}轮"
    )
    heading = f"{public_label}开发轮虚拟筛查"
    print(f"\n===== {heading} =====")
    print(
        f"分析题目={summary.get('item_count', 0)}；"
        f"本轮新合格={summary.get('newly_qualified_count', 0)}；"
        f"本轮四项主动门槛均通过={summary.get('current_round_qualified_count', 0)}；"
        f"累计冻结合格={summary.get('frozen_qualified_count', 0)}；"
        f"题目级指标冻结={summary.get('frozen_metric_count', 0)}；"
        f"题目级指标本轮重算={summary.get('measured_metric_count', 0)}；"
        f"待处理={summary.get('pending_treatment_count', 0)}；"
        f"正式题已锁定={summary.get('qualified_locked_count', 0)}；"
        f"整卷达标保留（单题未全过）={summary.get('facet_form_retained_count', 0)}；"
        f"监测警告={summary.get('monitoring_warning_count', 0)}；"
        f"不可估计门槛={summary.get('unestimable_metric_count', 0)}"
    )
    print("\n| 门槛 | 通过题数 | 阈值 | 不可估计 |")
    print("|---|---:|---:|---:|")
    for gate in round_result.get("gate_summary") or []:
        if isinstance(gate, dict):
            if gate.get("gate_id") not in GATE_ORDER:
                continue
            print(
                f"| {gate.get('label')} | {gate.get('pass_count', 0)}/"
                f"{gate.get('item_count', 0)} | ≥{_format_metric(gate.get('threshold'))} | "
                f"{gate.get('unestimable_count', 0)} |"
            )

    def print_overview(title: str, rows: list[dict]) -> None:
        print(f"\n{title}：{len(rows)} 题")
        if not rows:
            print("  无")
            return
        print("| 题目 | 状态 | CITC | Profile目标rho_s（诊断） | Profile同域VTS/最大rho facet | Profile跨域VTS/最大rho facet | 单题Hedges'g | 单题IPIP rho_s | 单题Δmin |")
        print("|---|---|---|---|---|---|---:|---:|---:|")
        for entry in rows:
            gates = {
                row.get("gate_id"): row
                for row in entry.get("gates") or []
                if isinstance(row, dict) and row.get("gate_id") in GATE_ORDER
            }
            contaminants = entry.get("max_contaminants") or {}
            same = contaminants.get("same_domain") or {}
            cross = contaminants.get("cross_domain") or {}
            print(
                f"| {entry.get('item_id')} | {entry.get('status')} | "
                f"{_cli_gate_display(gates.get('citc_pass') or {})} | "
                f"{_format_metric(_profile_diagnostic_value(entry, 'target_rho_diagnostic'))} | "
                f"{_format_metric(_profile_diagnostic_value(entry, 'same_domain_vts_diagnostic'))} / "
                f"{same.get('facet_name') or same.get('dimension_id') or '-'} | "
                f"{_format_metric(_profile_diagnostic_value(entry, 'cross_domain_vts_diagnostic'))} / "
                f"{cross.get('facet_name') or cross.get('dimension_id') or '-'} | "
                f"{_cli_gate_display(gates.get('target_hedges_g_pass') or {})} | "
                f"{_cli_gate_display(gates.get('target_ipip_spearman_rho_pass') or {})} | "
                f"{_cli_gate_display(gates.get('discriminant_delta_min_pass') or {})} |"
            )

    candidate_rows = [
        row
        for row in round_result.get("items") or []
        if isinstance(row, dict)
        and row.get("status") in {"pending_treatment", "newly_qualified"}
    ]
    print_overview("本轮候选题总览", candidate_rows)
    print_overview(
        "已锁定正式题监测",
        [row for row in round_result.get("locked_items") or [] if isinstance(row, dict)],
    )
    print_overview(
        "Facet整卷达标保留（不计qualified/frozen_qualified）",
        [row for row in round_result.get("facet_form_retained_items") or [] if isinstance(row, dict)],
    )
    option_score_rows = [
        row
        for row in round_result.get("option_score_comparisons") or []
        if isinstance(row, dict)
    ]
    print(f"\n所有题目的按选项对齐设定分数均值：{len(option_score_rows)} 行（仅诊断）")
    _print_option_score_comparisons(option_score_rows)
    for entry in round_result.get("pending_items") or []:
        if not isinstance(entry, dict):
            continue
        print(
            f"\n--- 待处理题 {entry.get('item_id')}；失败门槛="
            f"{','.join(entry.get('failed_thresholds') or [])} ---"
        )
        print("| 主动门槛 | 当前值 | 门槛 | 状态 | 参与过滤 |")
        print("|---|---:|---:|---|---|")
        for gate in entry.get("gates") or []:
            if isinstance(gate, dict):
                if gate.get("gate_id") not in GATE_ORDER:
                    continue
                status = "通过" if gate.get("passes") is True else (
                    "未通过" if gate.get("estimable") is True else "不可估计"
                )
                print(
                    f"| {gate.get('label')} | {_format_metric(gate.get('value'))} | "
                    f"≥{_format_metric(gate.get('threshold'))} | {status} | 是 |"
                )
        for diagnostic_id, label in (
            ("target_rho_diagnostic", "Profile目标rho_s"),
            ("same_domain_vts_diagnostic", "Profile同域VTS"),
            ("cross_domain_vts_diagnostic", "Profile跨域VTS"),
        ):
            print(
                f"| {label} | {_format_metric(_profile_diagnostic_value(entry, diagnostic_id))} | "
                "无 | 仅诊断 | 否 |"
            )
        for condition in entry.get("per_condition_metrics") or []:
            if isinstance(condition, dict):
                print(
                    f"  - {condition.get('condition_id')}（过滤权={'是' if condition.get('filtering_authority') else '否'}）："
                    f"CITC={_format_metric(condition.get('citc'))}；"
                    f"rho={_format_metric(condition.get('rho'))}"
                )
        gradient = entry.get("target_option_gradient") or {}
        if gradient:
            print(f"  目标组选项梯度：{'通过' if gradient.get('passes') else '失败'}；失败相邻对=" + ",".join(f"{row.get('lower_option_id')}<{row.get('higher_option_id')}" for row in gradient.get('failed_adjacent_pairs') or []))
        arm_diagnostics = entry.get("arm_difference_diagnostics") or {}
        for comparison in arm_diagnostics.get("comparisons") or []:
            if not isinstance(comparison, dict):
                continue
            overall = comparison.get("overall") or {}
            high_band = next(
                (
                    row
                    for row in comparison.get("by_score_band") or []
                    if isinstance(row, dict) and row.get("score_band") == "high"
                ),
                {},
            )
            print(
                f"  实验臂差异 {comparison.get('comparison_id')}（仅定位）："
                f"同选项率={_format_metric(overall.get('same_option_rate'))}；"
                f"总体题分差={_format_metric(overall.get('target_minus_comparator_mean_item_score'))}；"
                f"高分组三四级选择率差={_format_metric(high_band.get('target_minus_comparator_high_option_rate'))}"
            )
        item = entry.get("item") or {}
        if isinstance(item, dict) and item:
            print_item_candidate(item)
        comparisons = entry.get("option_score_comparisons") or []
        if isinstance(comparisons, list) and comparisons:
            _print_option_score_comparisons(
                [row for row in comparisons if isinstance(row, dict)]
            )

    if isinstance(iteration, dict) and iteration:
        print("\n===== Facet 开发轮次控制器 =====")
        print(
            f"policy_version={round_result.get('iteration_policy_version') or iteration.get('policy_version', '未记录')}；"
            f"status={iteration.get('status', '未记录')}；"
            f"开发轮={public_label}；"
            f"completed_rounds={iteration.get('completed_rounds', '未记录')}；"
            f"round_completed={round_result.get('round_completed', '未记录')}"
        )
        attempts = iteration.get("local_attempts") or {}
        print(
            "各facet本轮局部提交/复测次数："
            + (", ".join(f"{facet}={count}" for facet, count in sorted(attempts.items())) or "无")
        )
        print("已接受facet：" + (", ".join(sorted(iteration.get("accepted_facets") or {})) or "无"))
        print("失败facet：" + (", ".join(iteration.get("failed_facet_ids") or []) or "无"))
        print("基线facet：" + (", ".join(sorted(iteration.get("baseline_facets") or {})) or "无"))
        print(f"首轮作答 cohort 引用：{iteration.get('cohort_source_ref') or '未记录'}")
        print(f"首轮校标问卷引用：{iteration.get('first_round_reference_ref') or '未记录'}")
        print(f"暂停原因：{iteration.get('paused_reason') or '无'}")
        for completed in iteration.get("completed_history") or []:
            if isinstance(completed, dict):
                print(
                    f"已完成开发轮 {completed.get('development_round', '?')}；"
                    f"baseline_only={completed.get('baseline_only', False)}；"
                    f"完成facet数={len(completed.get('facet_forms') or {})}"
                )
        for label, key in (("首轮基线", "baseline_facets"), ("本轮已接受", "accepted_facets")):
            snapshots = iteration.get(key) or {}
            if not snapshots:
                continue
            print(f"{label}快照（五项整卷指标）：")
            print("| facet | 题数 | α | ICC | Hedges' g | 目标rho | Δmin |")
            print("|---|---:|---:|---:|---:|---:|---:|")
            for facet_id, snapshot in sorted(snapshots.items()):
                metrics = snapshot.get("metrics") or {}
                print(
                    f"| {facet_id} | {metrics.get('item_count', 0)} | "
                    f"{_format_metric(metrics.get('cronbach_alpha'))} | "
                    f"{_format_metric(metrics.get('virtual_test_retest_icc'))} | "
                    f"{_format_metric(metrics.get('target_hedges_g'))} | "
                    f"{_format_metric(metrics.get('target_spearman_rho'))} | "
                    f"{_format_metric(metrics.get('discriminant_delta_min'))} |"
                )


def print_psychometric_summary(
    proposed_update: dict,
    state: dict,
) -> None:
    statistics = proposed_update.get("test_statistics") or {}
    virtual_summary = (
        (statistics.get("virtual_screening_metrics") or {}).get("summary")
        or {}
    )
    config = state.get("virtual_sample_config") or {}
    print("\n===== 探索性虚拟开发：4项主动门槛、3项profile诊断、5项整卷指标 =====")
    print(
        f"总样本量：{config.get('sample_size', '未知')}；"
        f"固定顶层臂：{config.get('condition_count', statistics.get('condition_count', 3))}；"
        f"facet group：{config.get('group_count', statistics.get('group_count', '?'))}；"
        f"每组人数：{config.get('sample_size_per_condition', '未知')}；"
        "主施测每人每题一次，target组另做一次整卷重测"
    )
    print(
        "分面内题项一致性：CITC中位数="
        f"{_format_metric(virtual_summary.get('median_citc'))}；单题门槛CITC≥.30"
    )
    print(
        "profile诊断（无阈值、不参与资格）：目标ρs中位数="
        f"{_format_metric(virtual_summary.get('median_target_rho'))}；"
        "最小同域VTS="
        f"{_format_metric(virtual_summary.get('minimum_same_domain_vts'))}；"
        "最小跨域VTS="
        f"{_format_metric(virtual_summary.get('minimum_cross_domain_vts'))}；"
        "目标ρs、同域VTS、跨域VTS仅供诊断"
    )
    print(
        "单题主动门槛：CITC≥.30、IPIP目标Hedges'g≥.50、"
        "目标IPIP rho_s≥.40、Δmin≥.30；当前摘要：Hedges'g中位数="
        f"{_format_metric(virtual_summary.get('median_item_target_hedges_g'))}；"
        "目标IPIP rho_s中位数="
        f"{_format_metric(virtual_summary.get('median_item_target_ipip_spearman_rho'))}；"
        "Δmin最小值="
        f"{_format_metric(virtual_summary.get('minimum_item_discriminant_delta_min'))}；"
        "；整卷alpha≥.70、ICC≥.70，后续成功轮每facet的g/rho/Δmin均须较上一完成轮严格上升>1e-12"
    )
    combined_state = {**state, **proposed_update}
    round_result = proposed_update.get("psychometric_round_result") or (
        build_psychometric_round_result(combined_state)
    )
    print_psychometric_round_result(
        round_result,
        history=[
            row for row in combined_state.get("psychometric_iteration_history") or []
            if isinstance(row, dict)
        ],
    )
    print_iteration_item_metrics(combined_state)
    print("\n描述性诊断（不参与返修）：")
    print(
        "Cronbach α="
        f"{_format_metric(statistics.get('cronbach_alpha'))}"
    )
    print("| 题目 | 难度 | 有效选项数 |")
    print("|---|---:|---:|")
    for item_id, item in (proposed_update.get("item_statistics") or {}).items():
        quality = item.get("quality_evaluation") or {}
        print(
            f"| {item_id} | {_format_metric(item.get('difficulty'))} | "
            f"{quality.get('effective_option_count', '?')} |"
        )
    print("注：上述结果仅是探索性虚拟筛查证据，不是正式单题信效度。")


def _print_item_id_list(
    label: str,
    items: list[dict],
    reasons: dict[str, str],
) -> None:
    print(f"{label}（{len(items)}题）：")
    if not items:
        print("  - 无")
        return
    for item in items:
        item_id = str(item.get("item_id") or "?")
        reason = reasons.get(item_id) or "未记录原因"
        print(f"  - {item_id}：{reason}")


def print_provisional_iteration_summary(proposed_update: dict) -> None:
    """Print the whole-test baseline assembled before single-item repair."""

    history = [
        row
        for row in proposed_update.get("psychometric_iteration_history") or []
        if isinstance(row, dict)
    ]
    if not history:
        return
    provisional = max(
        history,
        key=lambda row: int(row.get("analysis_round") or 0),
    )
    form_metrics = provisional.get("form_metrics") or {}
    reliability = form_metrics.get("reliability") or {}
    validity = form_metrics.get("validity") or {}
    recovery = validity.get("target_recovery") or {}
    selectivity = validity.get("construct_selectivity") or {}
    optimization = form_metrics.get("optimization") or {}
    derived_quality = form_quality_summary(form_metrics)
    stability_gate = (
        optimization.get("stability_gate")
        or derived_quality.get("stability_gate")
        or {}
    )
    selectivity_value = selectivity.get("value")
    if selectivity_value is None:
        selectivity_value = derived_quality.get("construct_selectivity")
    candidate_quality = provisional.get("candidate_form_quality")
    if candidate_quality is None:
        candidate_quality = derived_quality.get("candidate_form_quality")
    plateau = provisional.get("plateau_status") or {}
    best_quality = provisional.get("best_so_far_form_quality")
    if best_quality is None:
        best_quality = plateau.get("best_form_quality")
    iteration = provisional.get("facet_iteration_state") or {}
    public_label = development_round_batch_label(history, provisional)
    print("\n===== 本轮临时组卷（单题返修前基线） =====")
    print(
        f"开发轮={public_label}；"
        f"候选题={provisional.get('candidate_count', 0)}；"
        f"单题通过={provisional.get('qualified_item_count', 0)}；"
        f"临时测验={provisional.get('item_count', 0)}；"
        f"状态={provisional.get('form_status', '未记录')}"
    )
    print(
        "整卷指标："
        "目标恢复R²="
        f"{_format_metric(recovery.get('cross_validated_r2'))}；"
        "构念选择性="
        f"{_format_metric(selectivity_value)}；"
        "本轮候选质量="
        f"{_format_metric(candidate_quality)}；"
        "历史最优质量="
        f"{_format_metric(best_quality)}"
    )
    if not derived_quality.get("facet_metrics"):
        print(
            "稳定性门槛："
            "虚拟重测ICC="
            f"{_format_metric(reliability.get('virtual_test_retest_icc'))}；"
            f"最低={_format_metric(stability_gate.get('minimum'))}；"
            f"通过={'是' if stability_gate.get('passed') else '否'}"
        )
    for facet in derived_quality.get("facet_metrics") or []:
        print(
            f"facet={facet.get('sjt_facet_id')}；题数={facet.get('item_count')}；"
            f"α={_format_metric(facet.get('cronbach_alpha'))}"
            f"({'通过' if (facet.get('alpha_gate') or {}).get('passed') else '未通过'})；"
            f"ICC={_format_metric(facet.get('virtual_test_retest_icc'))}"
            f"({'通过' if (facet.get('stability_gate') or {}).get('passed') else '未通过'})；"
            f"g={_format_metric(facet.get('target_hedges_g'))}；"
            f"rho={_format_metric(facet.get('target_spearman_rho'))}；"
            f"Δ={_format_metric(facet.get('discriminant_delta_min'))}；"
            f"首轮基线允许alpha/ICC低于.70；alpha门槛=.70；ICC门槛=.70"
        )
    iteration = proposed_update.get("facet_iteration_state") or {}
    comparisons = iteration.get("comparisons") or []
    if not comparisons:
        trajectory = plateau.get("trajectory") or []
        comparisons = trajectory[-1].get("facet_gate_comparison") or [] if trajectory else []
    if comparisons:
        accepted_facets = iteration.get("accepted_facets") or {}
        for comparison in comparisons:
            def gate_label(gate: dict) -> str:
                return "基线" if gate.get("passed") is None else "通过" if gate["passed"] else "未通过"

            def improvement(gate: dict) -> str:
                observed = metric_scalar(gate.get("observed"))
                incumbent = metric_scalar(gate.get("incumbent"))
                return "未记录" if observed is None or incumbent is None else f"{observed - incumbent:+.6g}"

            rho_gate = comparison.get("target_rho_improvement_gate") or comparison.get("target_rho_noninferiority_gate") or {}
            delta_gate = comparison.get("discriminant_delta_improvement_gate") or comparison.get("discriminant_delta_non_decrease_gate") or {}
            g_gate = comparison.get("hedges_g_improvement_gate") or {}
            facet_id = str(comparison.get("sjt_facet_id"))

            print(
                f"facet={facet_id} 相对上一完成轮："
                f"g升幅={improvement(g_gate)}（{gate_label(g_gate)}）；"
                f"rho升幅={improvement(rho_gate)}（{gate_label(rho_gate)}）；"
                f"Δmin升幅={improvement(delta_gate)}（{gate_label(delta_gate)}）；"
                f"该facet本轮已接受={'是' if facet_id in accepted_facets else '否'}"
            )
    usage = provisional.get("token_usage") or {}
    print(
        "本轮成本："
        f"Token={usage.get('total_tokens', 0)}；"
        f"模型耗时={usage.get('duration_ms', 0)} ms"
    )
    if plateau.get("reached") and not iteration.get("policy_version"):
        print(
            "旧协议历史平台期字段（不参与当前控制器判定）："
            f"连续未改善轮数={plateau.get('non_improving_rounds', '?')}"
        )
    if provisional.get("form_selection_error"):
        print(f"临时组卷备注：{provisional['form_selection_error']}")


def print_iteration_item_metrics(proposed_update: dict) -> None:
    """Show round-specific item gates and the owning facet's form metrics."""

    history = [
        row
        for row in proposed_update.get("psychometric_iteration_history") or []
        if isinstance(row, dict)
    ]
    rendered = False
    round_entries = strict_development_round_entries(history)
    for entry in round_entries:
        raw_item_metrics = (
            entry.get("candidate_item_metrics")
            or entry.get("item_metrics")
            or {}
        )
        if isinstance(raw_item_metrics, dict):
            item_rows = [row for row in raw_item_metrics.values() if isinstance(row, dict)]
        elif isinstance(raw_item_metrics, list):
            item_rows = [row for row in raw_item_metrics if isinstance(row, dict)]
        else:
            item_rows = []
        if not item_rows:
            continue
        rendered = True
        saved_facet_metrics = entry.get("facet_metrics") or {}
        if isinstance(saved_facet_metrics, dict) and saved_facet_metrics:
            facet_metrics = {
                str(facet_id): row
                for facet_id, row in saved_facet_metrics.items()
                if isinstance(row, dict)
            }
        else:
            facet_metrics = {
                str(row.get("sjt_facet_id")): row
                for row in (entry.get("form_metrics") or {}).get("facet_metrics") or []
                if isinstance(row, dict) and row.get("sjt_facet_id") is not None
            }
        development_round = entry.get("development_round") or (
            (entry.get("facet_iteration_state") or {}).get("development_round")
            if isinstance(entry.get("facet_iteration_state"), dict)
            else None
        )
        round_label = development_round_batch_label(history, entry)
        print(f"\n===== {round_label}：每 facet / 每题指标 =====")
        print(
            "整卷指标使用本轮入选题目列表计算；已冻结题目仍进入临时组卷，"
            "每题保存四项主动门槛与三项profile诊断；未改题复用最新完整作答。"
        )
        print("| facet | 题数 | α | ICC | Hedges' g | 整卷目标rho | 整卷Δmin |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for facet_id, facet in facet_metrics.items():
            print(
                f"| {facet_id} | {facet.get('item_count', 0)} | "
                f"{_format_metric(facet.get('cronbach_alpha'))} | "
                f"{_format_metric(facet.get('virtual_test_retest_icc'))} | "
                f"{_format_metric(facet.get('target_hedges_g'))} | "
                f"{_format_metric(facet.get('target_spearman_rho'))} | "
                f"{_format_metric(facet.get('discriminant_delta_min'))} |"
            )
        print("| facet | 题目 | 入选整卷 | 题目级指标 | CITC | 目标rho_s | 同域VTS | 跨域VTS | 单题Hedges'g | 单题IPIP rho_s | 单题Δmin | 题目通过 | α | ICC | 整卷Hedges' g | 整卷目标rho | 整卷Δmin |")
        print("|---|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|")
        for row in sorted(item_rows, key=lambda value: str(value.get("item_id") or "")):
            facet_id = str(row.get("facet_id") or "未标注 facet")
            facet = facet_metrics.get(facet_id) or {}
            metric_status = {
                "frozen": "冻结",
                "measured": "本轮重算",
            }.get(str(row.get("iteration_metric_status") or ""), "未标记")
            print(
                f"| {facet_id} | {row.get('item_id', '未记录')} | "
                f"{'是' if row.get('selected_for_form') is True else '否'} | "
                f"{metric_status} | "
                f"{_format_metric(row.get('citc'))} | "
                f"{_format_metric(row.get('target_rho'))} | "
                f"{_format_metric(row.get('same_domain_vts'))} | "
                f"{_format_metric(row.get('cross_domain_vts'))} | "
                f"{_format_metric(row.get('target_hedges_g'))} | "
                f"{_format_metric(row.get('target_ipip_spearman_rho'))} | "
                f"{_format_metric(row.get('discriminant_delta_min'))} | "
                f"{'通过' if row.get('qualified') is True else '未通过'} | "
                f"{_format_metric(facet.get('cronbach_alpha'))} | "
                f"{_format_metric(facet.get('virtual_test_retest_icc'))} | "
                f"{_format_metric(facet.get('target_hedges_g'))} | "
                f"{_format_metric(facet.get('target_spearman_rho'))} | "
                f"{_format_metric(facet.get('discriminant_delta_min'))} |"
            )
    if history and not rendered:
        print("\n逐题轮次快照：当前历史记录尚未保存；不会用当前轮指标回填历史轮次。")


def print_virtual_content_review_history(state: dict) -> None:
    history = [
        row
        for row in state.get("virtual_content_review_history") or []
        if isinstance(row, dict)
    ]
    if not history:
        return
    protocol = state.get("virtual_content_review_protocol") or "未记录"
    print("\n===== 虚拟内容复审历史（虚拟开发证据；非真人SME评审） =====")
    print(f"协议：{protocol}；不代表真人内容效度或真人心理测量结果。")
    print("| 题目 | 所审版本 | 虚拟专家数 | 访谈角色数 | decision / status | 候选版本 | 归档路径 |")
    print("|---|---:|---:|---:|---|---|---|")
    for row in history:
        experts = row.get("expert_reviews")
        expert_count = (
            str(sum(isinstance(value, dict) for value in experts))
            if isinstance(experts, list)
            else "未记录"
        )
        interviews = row.get("cognitive_interviews")
        if isinstance(interviews, list):
            role_ids = [
                str((value.get("role") or {}).get("role_id"))
                for value in interviews
                if isinstance(value, dict)
                and isinstance(value.get("role"), dict)
                and value.get("role", {}).get("role_id")
            ]
            interview_count = (
                str(len(set(role_ids)))
                if len(role_ids) == len(interviews)
                else "未记录"
            )
        else:
            interview_count = "未记录"
        diagnosis = row.get("diagnosis")
        decision = diagnosis.get("decision") if isinstance(diagnosis, dict) else None
        status = row.get("status") or "未记录"
        candidate_id = row.get("candidate_item_id")
        candidate = row.get("candidate")
        candidate_version = row.get("new_item_version")
        if candidate_version is None and isinstance(candidate, dict):
            candidate_version = candidate.get("version")
        candidate_id = candidate_id or (
            "候选ID未记录" if candidate_version is not None else
            "待提交" if status == "ready" else "未生成"
        )
        candidate_text = (
            f"{candidate_id} v{candidate_version}"
            if candidate_version is not None
            else str(candidate_id)
        )
        cells = [
            row.get("reviewed_item_id") or "未记录",
            row.get("reviewed_item_version") or "未记录",
            expert_count,
            interview_count,
            f"{decision or '未记录'} / {status}",
            candidate_text,
            row.get("archive_ref") or "未记录",
        ]
        print(
            "| "
            + " | ".join(str(value).replace("|", "｜").replace("\n", " ") for value in cells)
            + " |"
        )


def print_selection_summary(proposed_update: dict) -> None:
    raw_selection = proposed_update.get("selection_results")
    selection = raw_selection if isinstance(raw_selection, dict) else {}
    reasons = proposed_update.get("selection_reasons") or {}
    selected = proposed_update.get("selected_items") or []
    repair_by_id = {
        str(entry.get("item_id")): entry
        for entry in [
            *(proposed_update.get("items_to_revise") or []),
            *(proposed_update.get("items_to_regenerate") or []),
        ]
        if isinstance(entry, dict) and entry.get("item_id")
    }
    repair_order = list(repair_by_id)
    revise = [
        repair_by_id[str(item_id)]
        for item_id in repair_order
        if str(item_id) in repair_by_id
    ]
    dispositions = proposed_update.get("item_final_dispositions") or selection.get("final_dispositions") or {}
    pending_sme = [
        {"item_id": item_id}
        for item_id, disposition in dispositions.items()
        if isinstance(disposition, dict)
        and disposition.get("status") == "pending_sme_review"
    ]
    eliminated = [
        {"item_id": item_id}
        for item_id, disposition in dispositions.items()
        if isinstance(disposition, dict) and disposition.get("status") == "eliminated"
    ]

    print("\n===== 心理测量返修诊断结果 =====")
    status = selection.get("status")
    if not status and revise:
        status = "逐题诊断进行中"
    print(f"状态：{status or '等待下一步'}")
    print("筛选权：仅使用三臂匹配条件的总指标；各条件臂诊断与输入相关不参与过滤。")

    print_provisional_iteration_summary(proposed_update)
    print_iteration_item_metrics(proposed_update)

    _print_item_id_list("正式题", selected, reasons)
    _print_item_id_list("待诊断题" if not raw_selection else "待处理题", revise, reasons)
    _print_item_id_list("待 SME 题（不进入正式题）", pending_sme, reasons)
    _print_item_id_list("淘汰题", eliminated, reasons)
    if revise:
        print("逐题诊断队列：")
        for entry in revise:
            advice = entry.get("atomic_repair_advice") or {}
            if (
                entry.get("queue_status") == "deferred_decision"
                and entry.get("diagnosis_status") == "repair_rounds_exhausted"
            ):
                print(
                    f"  - {entry.get('item_id', '?')}：已完成三轮返修仍未达标，"
                    "自动 defer，等待 SME/人工修改/淘汰处置"
                )
            else:
                print(
                    f"  - {entry.get('item_id', '?')} / "
                    f"第 {entry.get('revision_round', '?')} 轮 / "
                    f"{entry.get('queue_status', 'pending_diagnosis')}："
                    f"{advice.get('summary') or '等待诊断'}"
                )
    monitoring = proposed_update.get("psychometric_monitoring_warnings") or []
    if monitoring:
        print("正式题监测警告（资格不撤销且不返修）：")
        for entry in monitoring:
            if isinstance(entry, dict):
                print(f"  - {entry.get('item_id', '?')}：{entry.get('message', '')}")
    lineage = proposed_update.get("item_lineage") or {}
    replacement_rows = [
        (item_id, row)
        for item_id, row in lineage.items()
        if isinstance(row, dict) and row.get("replaces_item_id")
    ]
    if replacement_rows:
        print("补题 lineage：")
        for item_id, row in replacement_rows:
            print(f"  - {item_id} 替代 {row.get('replaces_item_id')}（槽位根题 {row.get('root_item_id')}）")


def print_workflow_effect(action: str, proposed_update: dict) -> None:
    if action == "simulate_responses":
        summary = proposed_update.get("virtual_response_summary") or {}
        print(
            "\n[虚拟作答] "
            f"复用未变题作答 {summary.get('reused_sjt_records', 0)} 条；"
            f"新增主施测SJT调用 {summary.get('scheduled_sjt_api_calls', 0)} 次；"
            "新增target重测调用 "
            f"{summary.get('scheduled_target_form_retest_api_calls', 0)} 次；"
            f"匹配条件组 {summary.get('condition_count', '?')} 个。"
        )
    elif action == "assemble_test":
        assembled = proposed_update.get("assembled_test") or {}
        round_number = proposed_update.get("reassembly_round", 0)
        print(
            "\n[流程触发] 已完成"
            + ("再次组卷" if round_number else "首次组卷")
            + f"：正式题 {assembled.get('item_count', '?')} 道。"
        )


def print_user_progress_event(
    event: TraceEvent,
    update: dict,
    state: dict,
    previous_state: dict | None = None,
) -> None:
    """Render stable, user-facing stage and item lifecycle progress."""

    node = event.get("node")
    action = str(event.get("action") or "unknown")
    event_type = event.get("event_type")
    if node in ACTION_START_NODES and event_type == "completed":
        if action == "finish":
            print("\n===== 工作流全部完成 =====", flush=True)
        else:
            print(
                f"\n===== 开始：{item_stage_label(action, state)} =====",
                flush=True,
            )
        return

    if node == "execute":
        if event_type == "completed":
            duration_ms = event.get("duration_ms")
            duration = (
                f"，耗时 {duration_ms / 1000:.1f} 秒"
                if isinstance(duration_ms, (int, float))
                else ""
            )
            print(
                f"\n===== 完成：{item_stage_label(action, state)}{duration} =====",
                flush=True,
            )
            print_automatic_item_result(action, update, state, event)
            proposed_update = update.get("pending_state_update") or {}
            if action == "analyze_psychometrics":
                print_psychometric_summary(proposed_update, state)
            elif action == "select_items":
                print_selection_summary(proposed_update)
            elif action == "simulate_responses":
                print_virtual_respondent_summary(
                    {**state, **proposed_update}, preview_limit=10
                )
            print_workflow_effect(action, proposed_update)
            if (
                action == "psychometric_repair_batch"
                and isinstance(proposed_update.get("virtual_content_review_history"), list)
            ):
                print_virtual_content_review_history({**state, **proposed_update})
        elif event_type == "failed":
            print(
                f"\n===== 失败：{item_stage_label(action, state)} =====\n"
                f"{event.get('error', '未知错误')}",
                flush=True,
            )
        return

    if (
        node == "accept_item"
        and event_type == "completed"
        and state.get("item_development_mode") == "automatic"
    ):
        print("\n===== 题目通过并进入候选题库 =====")
        print_item_candidate(
            (previous_state or {}).get("current_item")
            or state.get("current_item")
        )
    elif node == "abandon_item" and event_type == "completed":
        was_psychometric_candidate = isinstance(
            (previous_state or {}).get("active_psychometric_repair"), dict
        )
        current_item = (
            (previous_state or {}).get("current_item")
            or state.get("current_item")
            or {}
        )
        heading = (
            "返修候选未通过，已保留上一有效版本"
            if was_psychometric_candidate
            else "题目候选淘汰"
        )
        print(
            f"\n===== {heading} =====\n"
            f"题目编号：{current_item.get('item_id', '未知')}",
            flush=True,
        )


def print_heartbeat(action: object, elapsed_seconds: float) -> None:
    print(
        f"\n[运行中] {action_label(action)}仍在执行，"
        f"已等待 {round(elapsed_seconds)} 秒……",
        flush=True,
    )


def print_trace_event(event: TraceEvent) -> None:
    """以便于扫描的格式输出一个新的轨迹事件。"""

    step = event.get("step", "?")
    node = event.get("node", "unknown")
    action = event.get("action", "unknown")
    event_type = event.get("event_type", "unknown")
    duration_ms = event.get("duration_ms")
    duration = f" ({duration_ms} ms)" if duration_ms is not None else ""

    print(f"\n[{step}] {node} → {action}: {event_type}{duration}")
    if event.get("reason"):
        print(f"    原因：{event['reason']}")
    if event.get("error"):
        print(f"    错误：{event['error']}")

    state_changes = event.get("state_changes", {})
    if state_changes:
        print("    State changes:")
        for field, change in state_changes.items():
            print(f"      - {field}")
            print("        before:")
            pprint(change.get("before"), indent=10, width=100)
            print("        after:")
            pprint(change.get("after"), indent=10, width=100)


def print_test_specification(specification: object) -> None:
    """以用户可读形式展示测验规格，不暴露 State 结构。"""

    if not isinstance(specification, dict):
        print("暂未形成测验规格。")
        return
    for field, label in SPECIFICATION_LABELS.items():
        value = specification.get(field)
        if isinstance(value, list):
            value = "；".join(value) if value else "无"
        print(f"  {label}：{value}")


def print_blueprint_candidate(blueprint: object) -> None:
    if not isinstance(blueprint, dict) or blueprint.get("version") != 7:
        return
    retained = planned_retention_count(blueprint)
    generated = planned_generation_count(blueprint)
    print(
        "构念—题目细目表："
        f"最终保留 {retained} 题，"
        f"计划生成 {generated} 题"
    )
    profile = blueprint.get("construct_profile_snapshot") or {}
    print(
        f"构念档案：{profile.get('inventory_name', '?')} / "
        f"{profile.get('domain_name', '?')} / "
        f"{profile.get('selection_level', '?')}"
    )
    for facet in profile.get("facets") or []:
        if isinstance(facet, dict):
            print(
                f"  - {facet.get('facet_name', facet.get('facet_id', '?'))}："
                f"{facet.get('definition', '')}"
            )
    print("维度单元：")
    for cell in blueprint.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        print(
            f"  - {cell.get('facet_id', '?')} / "
            f"{cell.get('behavior_id', '?')} / "
            f"{cell.get('mechanism_id', '?')} / "
            f"{cell.get('situation_id', '?')} / "
            + f"生成 {cell.get('planned_generation_count', '?')}，"
            f"保留 {cell.get('planned_retention_count', '?')}"
        )
    print("固定题目槽位：")
    for slot in blueprint.get("slots") or []:
        if isinstance(slot, dict):
            print(
                f"  - {slot.get('specification_id', '?')} / "
                f"{slot.get('blueprint_cell_id', '?')}"
            )


def print_skeleton_candidate(update: dict, state: dict) -> None:
    """Render the committed abstract skeleton before its concrete item."""

    current = update.get("current_item_specification") or {}
    specification_id = current.get("specification_id")
    skeletons = update.get("item_skeletons") or {}
    skeleton = skeletons.get(specification_id)
    if not isinstance(skeleton, dict):
        return
    blueprint = state.get("blueprint") or {}
    profile = blueprint.get("construct_profile_snapshot") or {}
    facet = next(
        (
            row for row in profile.get("facets") or []
            if isinstance(row, dict)
            and row.get("facet_id") == current.get("target_dimension_id")
        ),
        {},
    )
    print("\n===== 当前心理骨架 =====")
    print(f"固定槽位：{specification_id or '未知'}")
    print(f"目标 facet：{facet.get('facet_name', current.get('target_dimension_id', '?'))}")
    print(f"行为证据：{current.get('behavior_evidence_id', '?')}")
    print(f"激活机制：{current.get('activation_mechanism', '?')}")
    print(f"情境引用：{current.get('situation_id', '?')}")
    print(f"情境类型：{skeleton.get('situation_type', '')}")
    print(f"风险水平：{skeleton.get('stakes_level', '')}")
    print(f"社会情境：{skeleton.get('social_context', '')}")
    print(f"核心冲突：{skeleton.get('behavioral_tension', '')}")
    print("四级行为结构：")
    for row in skeleton.get("option_structure") or []:
        if not isinstance(row, dict):
            continue
        print(
            f"  - {row.get('behavioral_level', '?')}："
            f"{row.get('behavioral_tendency', '')}；"
            f"心理功能：{row.get('psychological_function', '')}"
        )
    print("骨架校验：程序确定性校验通过；未执行独立 LLM 骨架审核")


def print_item_candidate(item: object) -> None:
    if not isinstance(item, dict):
        return
    print(f"题目编号：{item.get('item_id', '未编号')}")
    print(f"情境：{item.get('scenario', '')}")
    print(f"问题：{item.get('response_instruction', '')}")
    scoring_key = item.get("scoring_key") or {}
    print("选项：")
    for option in item.get("response_options") or []:
        if not isinstance(option, dict):
            continue
        option_id = option.get("option_id", "?")
        score = scoring_key.get(option_id, "?")
        print(f"  {option_id}. {option.get('text', '')}（计分：{score}）")


def print_review_candidate(update: dict) -> None:
    review = update.get("current_item_review")
    if isinstance(review, dict):
        if update.get("_review_from_trace_summary"):
            print("审题结论：已记录（轨迹摘要未展开）")
            summary = review.get("summary")
            if summary:
                print(f"审题摘要：{summary}")
            return
        tasks = review.get("repair_tasks") or []
        try:
            decision = derive_item_review_decision(
                review,
                repair_attempted=bool(
                    update.get("current_item_repair_attempted")
                ),
            )
        except (TypeError, ValueError):
            # A trace state diff is intentionally lossy. It may be useful for
            # display but must never terminate the workflow's CLI.
            print("审题结论：已记录（详细字段未展开）")
            summary = review.get("summary")
            if summary:
                print(f"审题摘要：{summary}")
            return
        print(f"审题结论：{decision}")
        summary = review.get("summary")
        if summary:
            print(f"审题摘要：{summary}")
        findings = review.get("findings") or []
        if findings:
            print("四维诊断：")
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            option_ids = finding.get("affected_option_ids") or []
            locus = str(finding.get("locus") or "?")
            location = (
                f"{locus}[{','.join(option_ids)}]"
                if option_ids
                else locus
            )
            print(
                f"  - [{finding.get('severity', '?')}] "
                f"{finding.get('criterion', '?')} / {location}："
                f"{finding.get('problem', '')}"
            )
            evidence = finding.get("evidence")
            if evidence:
                print(f"    依据：{evidence}")
        if tasks:
            print("程序派生修复任务：")
        for task in tasks:
            if not isinstance(task, dict):
                continue
            target_labels = []
            for target in task.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                field = target.get("field", "?")
                option_ids = target.get("option_ids") or []
                target_labels.append(
                    f"{field}[{','.join(option_ids)}]"
                    if option_ids
                    else str(field)
                )
            print(
                f"  - 修复任务 {task.get('task_id', '?')}："
                f"{', '.join(target_labels)}"
            )
            print(f"    问题：{task.get('problem', '')}")
            print(f"    修改要求：{task.get('instruction', '')}")
        return

    print("审题结果无效：缺少 current_item_review")


def print_candidate(payload: dict) -> None:
    """按业务动作展示用户需要审核的内容。"""

    action = payload.get("action")
    update = payload.get("proposed_update") or {}
    if action == "build_blueprint":
        print_blueprint_candidate(update.get("blueprint"))
    elif action in {"generate_item", "revise_item", "regenerate_item"}:
        print_skeleton_candidate(update, payload)
        print_item_candidate(update.get("current_item"))
    elif action == "review_item":
        print_review_candidate(update)


def prompt_requirement_decision(payload: dict) -> dict:
    """展示需求缺口、问题和建议，并收集一轮自然语言答复。"""

    proposed_update = payload.get("proposed_update") or {}
    print("\n===== 测验需求澄清 =====")
    if payload.get("summary"):
        print(f"摘要：{payload['summary']}")
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")
    if payload.get("field_errors"):
        print("字段问题：")
        for field, error in payload["field_errors"].items():
            label = SPECIFICATION_LABELS.get(field, field)
            print(f"  - {label}: {error}")

    print("\n当前候选测验规格：")
    candidate_specification = proposed_update.get("test_specification")
    print_test_specification(candidate_specification)
    if isinstance(candidate_specification, dict):
        try:
            resolved_profile = resolve_specification_profile(
                candidate_specification
            )
        except ValueError:
            resolved_profile = None
        if resolved_profile is not None:
            print(
                "  构念库解析："
                f"{resolved_profile['inventory_name']} / "
                f"{resolved_profile['domain_name']} / "
                f"{resolved_profile['selection_level']}，"
                f"{len(resolved_profile['facets'])} 个 facet"
            )
    questions = payload.get("questions") or []
    suggestions = payload.get("suggestions") or []
    if questions:
        print("\n请补充以下信息：")
        for index, question in enumerate(questions, 1):
            text = question.get("text") if isinstance(question, dict) else question
            print(f"  {index}. {text}")
    if suggestions:
        print("\n系统建议：")
        for suggestion in suggestions:
            print(f"  - {suggestion.get('field')}")
            print(f"    理由：{suggestion.get('reason')}")

    decisions = payload.get("available_decisions") or []
    if "confirm" in decisions:
        print("\n需求字段已经完整，请选择：")
        print("  1. 确认规格并进入生成计划")
        print("  2. 用自然语言提出修改")
        print("  3. 停止工作流")
        while True:
            choice = input("你的选择 [1-3]：").strip()
            if choice == "1":
                return {"decision": "confirm", "feedback": None}
            if choice == "2":
                feedback = input("请说明需要修改的内容：").strip()
                if feedback:
                    return {"decision": "revise", "feedback": feedback}
                print("修改需求时必须提供具体内容")
                continue
            if choice == "3":
                feedback = input("停止原因（可留空）：").strip() or None
                return {"decision": "stop", "feedback": feedback}
            print("请输入 1、2 或 3")

    print("\n请选择：")
    print("  1. 回答上述问题或补充需求")
    if "accept_suggestions" in decisions:
        print("  2. 接受本轮全部系统建议和默认推断值")
        print("  3. 停止工作流")
    else:
        print("  2. 停止工作流")
    max_choice = 3 if "accept_suggestions" in decisions else 2
    while True:
        choice = input(f"你的选择 [1-{max_choice}]：").strip()
        if choice == "1":
            feedback = input("请用自然语言回答：").strip()
            if feedback:
                return {"decision": "answer", "feedback": feedback}
            print("回答不能为空")
            continue
        if choice == "2" and "accept_suggestions" in decisions:
            return {"decision": "accept_suggestions", "feedback": None}
        if choice == ("3" if "accept_suggestions" in decisions else "2"):
            feedback = input("停止原因（可留空）：").strip() or None
            return {"decision": "stop", "feedback": feedback}
        print(f"请输入 1-{max_choice}")


def prompt_virtual_sample_selection(payload: dict) -> dict:
    """Collect shared score settings; target facet groups come from the item bank."""

    pool = payload.get("pool") or {}
    recommendations = payload.get("recommendations") or []
    recommended_size = payload.get("recommended_sample_size")
    default_seed = payload.get("default_seed", 7)
    default_max_retries = max(0, min(4, int(payload.get("default_max_retries", 4))))

    print("\n===== 配置三臂匹配 facet 虚拟被试 =====")
    print(
        f"最多生成：{pool.get('available_count', '?')} 名；"
        "主施测中每名虚拟被试对每题回答1次；target组额外完成一次整卷重测。"
    )
    if payload.get("method_note"):
        print(f"说明：{payload['method_note']}")
    print(
        f"模型请求策略：全量并发启动；每次逻辑失败调用最多尝试 {default_max_retries + 1} 次（含首次），底层重试计入该预算。"
    )
    response_temperature = payload.get(
        "response_temperature", DEFAULT_RESPONSE_TEMPERATURE
    )
    demographics_version = payload.get("demographics_version", DEMOGRAPHICS_VERSION)
    print(
        f"拟用作答温度={response_temperature}；人口学库版本={demographics_version}"
        "（确认生成后才冻结到本轮样本）"
    )
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")

    print("\nCohort、完整人格分数、人口学变量和模拟设置仅首轮生成并冻结；后续仅测改过的题目新版本，未改题复用对应完整作答，不重复施测IPIP校标。")
    max_sample_size = int(pool.get("available_count") or 1000)
    minimum_sample_size = int(payload.get("minimum_sample_size") or 30)
    sample_size = None
    while sample_size is None:
        raw_size = input(
            f"请输入本轮虚拟被试人数（必须是 >={minimum_sample_size} 的整数）："
        ).strip()
        if not raw_size.isdigit():
            print("人数必须是整数")
            continue
        candidate_size = int(raw_size)
        if candidate_size < minimum_sample_size:
            print(f"本轮人数必须大于等于 {minimum_sample_size}")
            continue
        if candidate_size > max_sample_size:
            print(f"本轮人数不能超过 {max_sample_size}")
            continue
        sample_size = candidate_size

    catalog = [
        row for row in payload.get("dimension_catalog") or []
        if isinstance(row, dict) and row.get("dimension_id")
    ]
    targets = [
        row for row in catalog if row.get("required_target") is True
    ]
    def read_score(label: str, default: float = 50.0) -> float:
        while True:
            raw_value = input(
                f"{label} 的样本平均分 [>0 且 <100]（默认 {default:g}）："
            ).strip()
            if not raw_value:
                return default
            try:
                value = float(raw_value)
            except ValueError:
                print("请输入0–100范围内的有限数值。")
                continue
            if math.isfinite(value) and 0.0 < value < 100.0:
                return value
            print("自动迭代需要非零方差，因此平均分须大于0且小于100。")

    if not targets:
        raise ValueError("题库没有目标 facet")
    print("\n本轮题库目标 facet 由题目自动确定：")
    for row in targets:
        print(f"  {row['dimension_id']} = {row.get('display_label', row['dimension_id'])}")
    print(
        "系统会对每个目标 facet 分别计算：同域另外5个 facet，"
        "以及跨域另外24个 facet；无需手动输入 facet ID。"
    )
    def read_numeric(prompt: str, default: float, *, positive: bool = False) -> float:
        while True:
            raw = input(f"{prompt}（默认 {default:g}）：").strip()
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError:
                print("请输入有限数值")
                continue
            if not math.isfinite(value) or (positive and value <= 0):
                print("请输入正的有限数值")
                continue
            return value
    mean_score = read_score("所有 facet")
    standard_deviation = read_numeric(
        "请输入所有 facet 共享的正态分布 SD", 15.0, positive=True
    )

    return {
        "sample_size_per_condition": sample_size,
        "score_distribution": {
            "family": "normal",
            "mean": mean_score,
            "sd": standard_deviation,
        },
        "seed": default_seed,
        "max_concurrency": 0,
        "max_retries": default_max_retries,
    }


def print_virtual_respondent_summary(
    state: dict,
    *,
    preview_limit: int = 10,
) -> None:
    """Display frozen demographics without expanding the CLI output unboundedly."""

    config = state.get("virtual_sample_config") or {}
    respondents = [
        row for row in state.get("virtual_respondents") or []
        if isinstance(row, dict)
    ]
    if not config and not respondents:
        return

    version = config.get("demographics_version")
    temperature = config.get("response_temperature")
    print(
        "虚拟被试设置："
        f"作答温度={temperature if temperature is not None else '旧协议/未冻结'}；"
        f"人口学库版本={version or '旧协议/未冻结'}"
    )
    if not respondents:
        return

    snapshot = config.get("demographics_snapshot")
    if not isinstance(snapshot, dict) or any(
        not isinstance(row.get("demographics"), dict) for row in respondents
    ):
        print("旧协议检查点缺少冻结人口学资料，需重新配置；未推断或补造数据。")
        return

    try:
        rows = [
            {
                "被试ID": respondent.get("respondent_id"),
                **demographics_to_columns(respondent["demographics"], snapshot),
            }
            for respondent in respondents[: max(0, preview_limit)]
        ]
    except ValueError as exc:
        print(f"冻结人口学资料无法校验，需重新配置：{exc}")
        return
    headers = [
        ("被试ID", "被试ID"),
        ("年龄", "age"),
        ("性别", "gender"),
        ("国籍", "nationality"),
        ("学历", "education"),
        ("职业", "occupation"),
        ("税前月收入（CNY，合成）", "monthly_income_cny"),
    ]
    print(
        f"人口学资料预览（{len(rows)}/{len(respondents)}人；收入为合成税前月收入，CNY）："
    )
    print(" | ".join(label for label, _ in headers))
    for row in rows:
        print(" | ".join(str(row.get(key, "")) for _, key in headers))


def prompt_user_decision(payload: dict) -> dict:
    """按任务类型收集一次有效的用户决策。"""
    if payload.get("type") == "repair_knowledge_selection":
        while True:
            choice = input("返修知识：1. 共享历史知识并继续积累  2. 仅本任务积累 [1-2]：").strip()
            if choice in {"1", "2"}:
                return {"mode": "shared" if choice == "1" else "run_only"}
    if payload.get("type") == "scenario_repair_pause":
        print(payload.get("message") or "情境检测返修已暂停，整批尚未提交。")
        knowledge = payload.get("knowledge") or {}
        if knowledge.get("status") == "paused":
            print(f"  知识归纳：{knowledge.get('stage')}; {knowledge.get('error')}; 存档={knowledge.get('journal_ref')}")
        for item_id, reason in (payload.get("failures") or {}).items():
            progress = (payload.get("progress") or {}).get(item_id) or {}
            print(f"  {item_id}: {reason}; 设计阶段={progress.get('design_stage', 0)}; 改写={progress.get('rewrite_count', 0)}/5; 存档={progress.get('archive_ref')}")
        return {"decision": "stop" if payload.get("recovery_version") == 2 else "retry"}

    if payload.get("type") == "virtual_sample_selection":
        return prompt_virtual_sample_selection(payload)

    if payload.get("type") == "post_virtual_response_decision":
        print(payload.get("summary") or "")
        round_result = payload.get("round_result") or {}
        if isinstance(round_result, dict) and round_result:
            print_psychometric_round_result(
                round_result,
                history=[
                    row for row in payload.get("psychometric_iteration_history") or []
                    if isinstance(row, dict)
                ],
            )
        else:
            print("本轮缺少统一结果结构，请重新运行心理测量分析。")
        print_provisional_iteration_summary(payload)
        print_iteration_item_metrics(payload)
        diagnostics = payload.get("condition_score_diagnostics") or {}
        print("\n补充：匹配条件组分数分布（不参与过滤）")
        for condition in diagnostics.get("conditions") or []:
            if isinstance(condition, dict):
                print(
                    f"  - {condition.get('condition_id')}："
                    f"均值={_format_metric(condition.get('actual_mean'))}；"
                    f"SD={_format_metric(condition.get('actual_sample_sd'))}"
                )
        print("正式题资格一经锁定不撤销，监测警告不会触发返修。")
        print("  1. 开始处理未通过题目")
        print("  2. 暂停并保存本次运行")
        while True:
            choice = input("你的选择 [1-2]：").strip()
            if choice == "1":
                return {"decision": "start"}
            if choice == "2":
                return {"decision": "stop"}
            print("请输入 1 或 2")


    if payload.get("type") == "plateau_gap_decision":
        print("\n===== 平台期收卷：蓝图缺口处置 =====")
        print(payload.get("summary") or "")
        gap_cells = payload.get("gap_cells") or []
        resolutions: list[dict] = []
        stopped = False
        for index, cell in enumerate(gap_cells, start=1):
            cell_id = cell.get("blueprint_cell_id")
            candidates = cell.get("candidates") or []
            eligible = [row for row in candidates if row.get("eligible")]
            sme_rows = [
                row for row in candidates if row.get("force_allowed")
            ]
            print(
                f"\n[{index}/{len(gap_cells)}] 缺口单元 {cell_id}"
                f"（需保留 {cell.get('planned_retention_count')} 题）"
            )
            for row in candidates:
                if row.get("eligible"):
                    tag = "（可直接补位/手动改）"
                elif row.get("force_allowed"):
                    tag = "（待SME：可强制补位）"
                else:
                    tag = "（不可选：已淘汰）"
                gates = "、".join(row.get("failed_gates") or []) or "无"
                print(
                    f"  - {row.get('item_id')} v{row.get('version')} "
                    f"状态={row.get('disposition_status') or 'none'}{tag} "
                    f"失败门槛={gates}"
                )
            if not eligible and not sme_rows:
                print("  该单元没有可选的候选（已淘汰）。")
                print("  请选择停止，先人工处置后再恢复。")
                stopped = True
                break
            while True:
                cmd = input(
                    f"  单元 {cell_id} 处理：输入候选 ID 直接补位；"
                    "改:<ID> 手动修改；强制:<ID> 待SME题开发版补位；"
                    "强改:<ID> 手动改待SME题；stop 停止："
                ).strip()
                if cmd.lower() == "stop":
                    stopped = True
                    break
                manual = False
                sme_override = False
                item_id = cmd
                for prefix in ("强改:", "强改："):
                    if cmd.startswith(prefix):
                        manual = True
                        sme_override = True
                        item_id = cmd[len(prefix):].strip()
                        break
                if not manual:
                    for prefix in ("强制:", "强制："):
                        if cmd.startswith(prefix):
                            sme_override = True
                            item_id = cmd[len(prefix):].strip()
                            break
                if not manual:
                    for prefix in ("改:", "改："):
                        if cmd.startswith(prefix):
                            manual = True
                            item_id = cmd[len(prefix):].strip()
                            break
                if not manual:
                    for prefix in ("pick:", "pick："):
                        if cmd.startswith(prefix):
                            item_id = cmd[len(prefix):].strip()
                            break
                candidate = next(
                    (row for row in eligible if str(row.get("item_id")) == item_id),
                    None,
                )
                if candidate is None and sme_override:
                    candidate = next(
                        (
                            row
                            for row in sme_rows
                            if str(row.get("item_id")) == item_id
                        ),
                        None,
                    )
                if candidate is None:
                    print(
                        "  请输入上面列出的候选 ID；待SME题需加 强制:/强改: 前缀"
                    )
                    continue
                if sme_override and not candidate.get("force_allowed"):
                    print("  该题不是待SME题，不需要 强制 前缀")
                    continue
                if manual:
                    options = candidate.get("response_options") or []
                    print(
                        "  当前情境："
                        + str(candidate.get("scenario") or "")
                    )
                    scenario = input("  新的情境文本：").strip()
                    option_texts = {}
                    for option in options:
                        print(
                            f"  当前选项 {option.get('option_id')}："
                            f"{option.get('text')}"
                        )
                        option_texts[str(option.get("option_id"))] = input(
                            f"  选项 {option.get('option_id')} 新文本："
                        ).strip()
                    resolutions.append(
                        {
                            "cell_id": cell_id,
                            "item_id": item_id,
                            "mode": "manual",
                            "sme_override": sme_override,
                            "manual_item": {
                                "scenario": scenario,
                                "response_options": [
                                    {
                                        "option_id": option.get("option_id"),
                                        "text": option_texts.get(
                                            str(option.get("option_id")), ""
                                        ),
                                    }
                                    for option in options
                                ],
                            },
                        }
                    )
                else:
                    resolutions.append(
                        {
                            "cell_id": cell_id,
                            "item_id": item_id,
                            "mode": "pick",
                            "sme_override": sme_override,
                        }
                    )
                break
            if stopped:
                break
        if stopped:
            return {"decision": "stop"}
        return {"decision": "resolve", "resolutions": resolutions}

    if payload.get("type") == "psychometric_repair_confirmation":
        print("\n===== 心理测量返修确认 =====")
        print(f"题目编号：{payload.get('item_id', '?')}")
        item = payload.get("item")
        if isinstance(item, dict):
            print_item_candidate(item)
        diagnosis = payload.get("diagnosis") or {}
        print(
            f"返修轮次：{payload.get('revision_round', '?')}；"
            f"队列位置：1/{max(1, len(payload.get('pending_item_queue') or []))}"
        )
        observations = payload.get("observations") or []
        if observations:
            print("四项主动门槛、三项Profile诊断与最大污染facet：")
            for observation in observations:
                if not isinstance(observation, dict) or observation.get("role") == "descriptive_only":
                    continue
                facet = observation.get("facet_name") or observation.get("dimension_id")
                suffix = f"；facet={facet}" if facet else ""
                signed = observation.get("signed_rho")
                if signed is not None:
                    suffix += f"；rho={_format_metric(signed)}"
                print(
                    f"  - {observation.get('metric', '?')}="
                    f"{_format_metric(observation.get('value'))}；"
                    f"阈值={_format_metric(observation.get('threshold'))}{suffix}"
                )
        constraints = payload.get("non_target_construct_constraints") or []
        target_constraints = payload.get("target_construct_constraints") or []
        if target_constraints:
            print("目标 facet 的构念约束：")
            for constraint in target_constraints:
                if isinstance(constraint, dict):
                    print(
                        f"  - {constraint.get('constraint_id', '?')}："
                        f"{constraint.get('statement', constraint.get('text', ''))}"
                    )
        if constraints:
            print("最大污染 facet 的构念定义与高低行为边界：")
            for constraint in constraints:
                if isinstance(constraint, dict):
                    print(
                        f"  - {constraint.get('constraint_id', '?')}："
                        f"{constraint.get('statement', '')}"
                    )
        option_score_comparisons = payload.get("option_score_comparisons") or []
        if isinstance(option_score_comparisons, list) and option_score_comparisons:
            _print_diagnosis_option_score_comparisons(
                [row for row in option_score_comparisons if isinstance(row, dict)]
            )
        if diagnosis.get("summary"):
            print(f"诊断摘要：{diagnosis['summary']}")
        candidates = diagnosis.get("candidate_diagnoses") or []
        if candidates:
            print("候选问题：")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                options = ",".join(candidate.get("affected_option_ids") or [])
                location = candidate.get("suspect_components") or []
                print(
                    f"  - {candidate.get('diagnosis_id', '?')} / "
                    f"{','.join(location)}[{options}] / "
                    f"置信度={candidate.get('confidence', '?')}"
                )
                print(f"    证据：{candidate.get('textual_evidence', '')}")
                print(f"    说明：{candidate.get('explanation', '')}")
        tasks = diagnosis.get("repair_tasks") or []
        if tasks:
            print("已确认的修改任务：")
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                edit = task.get("atomic_edit") or {}
                option_ids = ",".join(edit.get("option_ids") or [])
                print(
                    f"  - {task.get('diagnosis_id', '?')} / "
                    f"{edit.get('target_field', '?')}[{option_ids}]："
                    f"{edit.get('problem', '')}"
                )
                print(f"    修改要求：{edit.get('instruction', '')}")
        if diagnosis.get("decision") == "repair":
            print("  1. 确认全部任务，自动原子返修后统一重测")
            print("  2. 暂停并保存")
            while True:
                choice = input("你的选择 [1-2]：").strip()
                if choice == "1":
                    return {"decision": "approve"}
                if choice == "2":
                    return {"decision": "stop"}
                print("请输入 1 或 2")

        print("诊断结论为 defer，请选择处置：")
        print("  1. 人工修改情境与四个选项")
        print("  2. 保留待 SME 审核")
        print("  3. 淘汰并在同一蓝图槽位补题")
        print("  4. 暂停并保存")
        while True:
            choice = input("你的选择 [1-4]：").strip()
            if choice == "1":
                if not isinstance(item, dict):
                    print("当前题目不可用，不能人工修改")
                    continue
                original_scenario = str(item.get("scenario") or "")
                scenario = input(f"情境（回车保留原文）\n[{original_scenario}]\n> ").strip() or original_scenario
                options = []
                for option in item.get("response_options") or []:
                    if not isinstance(option, dict):
                        continue
                    option_id = str(option.get("option_id") or "")
                    original_text = str(option.get("text") or "")
                    revised_text = input(
                        f"选项 {option_id}（回车保留原文）\n[{original_text}]\n> "
                    ).strip() or original_text
                    options.append({"option_id": option_id, "text": revised_text})
                return {
                    "decision": "manual_edit",
                    "manual_item": {
                        "scenario": scenario,
                        "response_options": options,
                    },
                }
            if choice == "2":
                return {"decision": "pending_sme"}
            if choice == "3":
                return {"decision": "eliminate_replenish"}
            if choice == "4":
                return {"decision": "stop"}
            print("请输入 1、2、3 或 4")

    if payload.get("type") == "item_development_mode_selection":
        print("\n===== 选择题目开发模式 =====")
        for index, mode in enumerate(payload.get("modes") or [], 1):
            print(f"  {index}. {mode.get('label')}")
            print(f"     {mode.get('description')}")
        while True:
            choice = input("你的选择 [1-2]：").strip()
            if choice == "1":
                return {"mode": "manual"}
            if choice == "2":
                return {"mode": "automatic"}
            print("请输入 1 或 2")

    if payload.get("type") == "requirement_confirmation":
        return prompt_requirement_decision(payload)

    print("\n===== 请审核候选结果 =====")
    if payload.get("summary"):
        print(f"摘要：{payload['summary']}")
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")

    print_candidate(payload)
    print("\n请选择：")
    print("  1. 通过并继续")
    print("  2. 提供意见并重新生成")
    print("  3. 停止工作流")

    while True:
        choice = input("你的选择 [1-3]：").strip()
        if choice == "1":
            return {
                "decision": "approve",
                "feedback": None,
                "state_patch": None,
            }
        if choice == "2":
            feedback = input("请说明需要如何调整：").strip()
            if not feedback:
                print("重新生成时必须提供调整意见")
                continue
            return {
                "decision": "regenerate",
                "feedback": feedback,
                "state_patch": None,
            }
        if choice == "3":
            feedback = input("停止原因（可留空）：").strip() or None
            return {
                "decision": "stop",
                "feedback": feedback,
                "state_patch": None,
            }
        print("请输入 1、2 或 3")


def get_interrupt_payload(interrupt_update: object) -> dict:
    """从 LangGraph 的中断更新中提取用户可见负载。"""

    interrupts = (
        list(interrupt_update)
        if isinstance(interrupt_update, (list, tuple))
        else [interrupt_update]
    )
    if not interrupts:
        raise ValueError("LangGraph 返回了空的 interrupt 更新")

    payload = getattr(interrupts[0], "value", interrupts[0])
    if not isinstance(payload, dict):
        raise ValueError("LangGraph interrupt 负载必须是对象")
    return payload


async def run_with_trace(
    initial_state: dict,
    *,
    debug: bool = False,
    heartbeat_interval_seconds: float | None = None,
    checkpoint_root: Path | None = None,
) -> dict:
    """流式执行图，并在每个 Agent 结果后暂停等待用户确认。"""

    with telemetry_run_context(initial_state["run_id"]):
        return await _run_with_trace_impl(
            initial_state,
            debug=debug,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            checkpoint_root=checkpoint_root,
        )


async def _run_with_trace_impl(
    initial_state: dict,
    *,
    debug: bool = False,
    heartbeat_interval_seconds: float | None = None,
    checkpoint_root: Path | None = None,
) -> dict:
    """流式执行图，并在每个 Agent 结果后暂停等待用户确认。"""

    if heartbeat_interval_seconds is None:
        heartbeat_interval_seconds = float(
            os.getenv("SJT_HEARTBEAT_INTERVAL_SECONDS", "20")
        )
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds 必须是正数")

    result = dict(initial_state)
    displayed_event_ids: set[str] = set()
    config = {"configurable": {"thread_id": initial_state["run_id"]}}
    graph_input: dict | Command = initial_state

    with progress_callback(print_runtime_progress):
        while True:
            resume_command: Command | None = None
            stream = app.astream(
                graph_input,
                config=config,
                stream_mode="updates",
            )
            iterator = stream.__aiter__()
            next_chunk_task: asyncio.Task | None = None
            wait_started_at = monotonic()
            try:
                while True:
                    if next_chunk_task is None:
                        next_chunk_task = asyncio.create_task(anext(iterator))
                    done, _ = await asyncio.wait(
                        {next_chunk_task},
                        timeout=heartbeat_interval_seconds,
                    )
                    if not done:
                        active_action = (
                            result.get("pending_action")
                            or (result.get("route") or {}).get("next_action")
                            or "初始化工作流"
                        )
                        print_heartbeat(
                            active_action,
                            monotonic() - wait_started_at,
                        )
                        continue

                    try:
                        chunk = next_chunk_task.result()
                    except StopAsyncIteration:
                        break
                    finally:
                        next_chunk_task = None
                    wait_started_at = monotonic()

                    if "__interrupt__" in chunk:
                        payload = get_interrupt_payload(
                            chunk["__interrupt__"]
                        )
                        resume_command = Command(
                            resume=prompt_user_decision(payload)
                        )
                        break

                    for update in chunk.values():
                        if not isinstance(update, dict):
                            continue
                        previous_result = dict(result)
                        result.update(update)
                        if (
                            update.get("virtual_sample_config")
                            and update.get("virtual_sample_config") != previous_result.get("virtual_sample_config")
                        ):
                            print_virtual_respondent_summary(result, preview_limit=10)
                        if checkpoint_root is not None:
                            save_run_checkpoint(
                                result,
                                checkpoint_root=Path(checkpoint_root),
                            )
                        for event in update.get("execution_history", []):
                            event_id = event.get("event_id")
                            if (
                                event_id
                                and event_id in displayed_event_ids
                            ):
                                continue
                            print_user_progress_event(
                                event,
                                update,
                                result,
                                previous_result,
                            )
                            if debug:
                                print_trace_event(event)
                            if event_id:
                                displayed_event_ids.add(event_id)
            finally:
                if next_chunk_task is not None:
                    next_chunk_task.cancel()
                    await asyncio.gather(
                        next_chunk_task,
                        return_exceptions=True,
                    )
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    await aclose()

            if resume_command is None:
                break
            graph_input = resume_command

    return result


def print_final_result(result: dict) -> None:
    """只展示流程状态和可交付结果，不打印内部 State。"""

    status = result.get("status", "unknown")
    print("\n===== 本次运行结束 =====")
    if status == "failed":
        errors = result.get("errors") or []
        message = errors[-1].get("message") if errors else "未知错误"
        print(f"运行失败：{message}")
        return
    if status == "stopped":
        print("工作流已停止。")
        pause = result.get("scenario_repair_pause") or {}
        if pause:
            if pause.get("limit_reached") is True:
                print("自动恢复或设计升级额度已用尽，已保存；原题库未提交，不再询问重复补做。")
            else:
                print(
                    "审题返修已暂停，检查点和已完成步骤已保存；原题库未提交。"
                    "是否可继续取决于暂停原因和既定调用额度。"
                )
            for item_id, reason in (pause.get("failures") or {}).items():
                print(f"  {item_id}: {reason}")
        return
    selection = result.get("selection_results") or {}
    provisional_count = len(selection.get("provisional_item_ids") or [])
    selected_count = len(result.get("selected_items") or [])
    reserve_count = len(result.get("reserve_items") or [])
    if result.get("final_test"):
        print(
            "开发版测验已经生成。"
            if selection.get("developmental_override")
            else "候选测验已经生成。"
        )
    else:
        print(f"运行状态：{status}")
    print(f"开发题库：{len(result.get('item_pool') or [])} 题")
    if selected_count or reserve_count:
        print(
            f"入卷：{selected_count} 题；备用：{reserve_count} 题；"
            f"开发版标记：{provisional_count} 题"
        )
    if result.get("item_bank_id"):
        print(
            "冻结题库："
            f"{result['item_bank_id']}（v{result.get('item_bank_version')}）"
        )
    deliverables = [
        (
            "正式测验",
            (result.get("final_test") or {}).get("file_path"),
        ),
        (
            "技术报告",
            (result.get("technical_report") or {}).get("markdown_path"),
        ),
        ("题库文件", result.get("item_database_ref")),
        (
            "虚拟被试报告",
            (result.get("virtual_respondent_report") or {}).get(
                "file_path"
            ),
        ),
    ]
    available = [
        (label, path)
        for label, path in deliverables
        if isinstance(path, str) and path
    ]
    if available:
        print("交付文件：")
        for label, path in available:
            print(f"  - {label}：{path}")


def select_start_state(
    new_state_factory: Callable[[], dict],
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict:
    """Select a fresh run or resume the newest nonterminal checkpoint."""

    latest = find_latest_resumable_checkpoint(checkpoint_root, include_stopped_repair=True)
    if latest is None:
        return new_state_factory()
    state = latest["state"]
    errors = state.get("errors") or []
    latest_error = (
        errors[-1].get("message")
        if isinstance(errors[-1], dict)
        else str(errors[-1])
    ) if errors else "无"
    print(
        "\n===== 检测到未完成运行 =====\n"
        f"运行编号：{latest['run_id']}\n"
        f"保存时间：{latest['saved_at']}\n"
        f"当前阶段：{state.get('current_phase', 'unknown')}\n"
        f"候选题目：{len(state.get('item_pool') or [])}\n"
        f"最近错误：{latest_error}\n"
        "  1. 继续上次运行\n"
        "  2. 放弃上次运行并开始新运行",
        flush=True,
    )
    while True:
        choice = input("你的选择 [1-2]：").strip()
        if choice == "1":
            return prepare_resumed_state(
                state,
                checkpoint_root=Path(checkpoint_root),
            )
        if choice == "2":
            abandoned = dict(state)
            abandoned["status"] = "stopped"
            save_run_checkpoint(
                abandoned,
                checkpoint_root=Path(checkpoint_root),
            )
            return new_state_factory()
        print("请输入 1 或 2")


async def main() -> None:
    global app
    state = select_start_state(
        lambda: create_initial_state(
            DEFAULT_USER_REQUEST,
            target_population=DEFAULT_TARGET_POPULATION,
            target_construct=DEFAULT_TARGET_CONSTRUCT,
            requested_item_count=DEFAULT_REQUESTED_ITEM_COUNT,
        )
    )
    debug = os.getenv("SJT_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    while True:
        result = await run_with_trace(
            state,
            debug=debug,
            checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
        )
        print_final_result(result)
        if result.get("status") != "failed":
            return
        while True:
            choice = input(
                "运行已暂停。输入 1 从最近检查点重试，"
                "输入 2 停止本次运行 [1-2]："
            ).strip()
            if choice == "1":
                state = prepare_retry_state(
                    result,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                )
                app = build_sjt_graph()
                break
            if choice == "2":
                stopped = dict(result)
                stopped["status"] = "stopped"
                save_run_checkpoint(
                    stopped,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                )
                return
            print("请输入 1 或 2")

if __name__ == "__main__":
    asyncio.run(main())
