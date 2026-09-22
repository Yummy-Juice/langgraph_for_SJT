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
    build_psychometric_round_result,
    metric_scalar,
)
from sjt_system.evaluation.form_metrics import form_quality_summary

app = build_sjt_graph()


SPECIFICATION_LABELS = {
    "construct_selection": "测量构念",
    "target_population": "目标人群",
    "final_item_count": "最终题量",
    "output_language": "输出语言",
}

ACTION_LABELS = {
    "clarify_requirements": "需求澄清",
    "build_blueprint": "构念—题目细目表",
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
        }.get(event.get("status"), event.get("status"))
        position = ""
        if event.get("queue_position") and event.get("queue_total"):
            position = (
                f" [{event.get('queue_position')}/{event.get('queue_total')}]"
            )
        round_text = ""
        if event.get("round"):
            round_text = (
                f"；局部轮次 {event.get('round')}/"
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
        if event.get("concurrency") is not None:
            details += f"；最大并发={event.get('concurrency')}"
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
        print("\n===== 题目审查结果 =====")
        print_review_candidate(proposed_update)


def _format_metric(value: object, *, digits: int = 3) -> str:
    numeric = metric_scalar(value)
    if numeric is None:
        return "证据不足"
    return f"{numeric:.{digits}f}"


def _cli_gate_display(gate: dict) -> str:
    status = "通过" if gate.get("passes") is True else (
        "未通过" if gate.get("estimable") is True else "不可估计"
    )
    return (
        f"{_format_metric(gate.get('value'))}/"
        f"≥{_format_metric(gate.get('threshold'))}/{status}"
    )


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
    print("| 题目 | 选项 | 计分 | 目标N | 目标组均值 | 同域N | 同域组均值 | 跨域N | 跨域组均值 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if not isinstance(row, dict):
            continue
        print(
            f"| {row.get('item_id')} | {row.get('option_id')} | "
            f"{_format_metric(row.get('option_score', row.get('score')))} | "
            f"{row.get('target_n', 0)} | {_format_metric(row.get('target_mean_score'))} | "
            f"{row.get('same_domain_n', 0)} | {_format_metric(row.get('same_domain_mean_score'))} | "
            f"{row.get('cross_domain_n', 0)} | {_format_metric(row.get('cross_domain_mean_score'))} |"
        )


def _print_diagnosis_option_score_comparisons(rows: list[dict]) -> None:
    if not rows:
        return
    print("VTS 按 option_id 对齐的均值定位证据：")
    print("| VTS类别 | 选项 | 计分 | 目标组均值 | 对应非目标组均值 |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        if not isinstance(row, dict):
            continue
        category = str(row.get("vts_category") or "")
        print(
            f"| {category or '-'} | {row.get('option_id')} | "
            f"{_format_metric(row.get('score'))} | "
            f"{_format_metric(row.get('target_mean_score'))} | "
            f"{_format_metric(row.get(f'{category}_mean_score'))} |"
        )


def print_psychometric_round_result(round_result: dict) -> None:
    summary = round_result.get("summary") or {}
    print(f"\n===== 第 {round_result.get('analysis_round', '?')} 轮虚拟筛查 =====")
    print(
        f"分析题目={summary.get('item_count', 0)}；"
        f"本轮新合格={summary.get('newly_qualified_count', 0)}；"
        f"待处理={summary.get('pending_treatment_count', 0)}；"
        f"正式题已锁定={summary.get('qualified_locked_count', 0)}；"
        f"监测警告={summary.get('monitoring_warning_count', 0)}；"
        f"不可估计门槛={summary.get('unestimable_metric_count', 0)}"
    )
    print("\n| 门槛 | 通过题数 | 阈值 | 不可估计 |")
    print("|---|---:|---:|---:|")
    for gate in round_result.get("gate_summary") or []:
        if isinstance(gate, dict):
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
        print("| 题目 | 状态 | CITC | 目标rho_s | 同域VTS/污染facet | 跨域VTS/污染facet |")
        print("|---|---|---|---|---|---|")
        for entry in rows:
            gates = {
                row.get("gate_id"): row
                for row in entry.get("gates") or []
                if isinstance(row, dict)
            }
            contaminants = entry.get("max_contaminants") or {}
            same = contaminants.get("same_domain") or {}
            cross = contaminants.get("cross_domain") or {}
            print(
                f"| {entry.get('item_id')} | {entry.get('status')} | "
                f"{_cli_gate_display(gates.get('citc_pass') or {})} | "
                f"{_cli_gate_display(gates.get('target_rho_pass') or {})} | "
                f"{_cli_gate_display(gates.get('same_domain_vts_pass') or {})} / "
                f"{same.get('facet_name') or same.get('dimension_id') or '-'} | "
                f"{_cli_gate_display(gates.get('cross_domain_vts_pass') or {})} / "
                f"{cross.get('facet_name') or cross.get('dimension_id') or '-'} |"
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
        print("| 指标 | 当前值 | 门槛 | 状态 | 参与过滤 |")
        print("|---|---:|---:|---|---|")
        for gate in entry.get("gates") or []:
            if isinstance(gate, dict):
                status = "通过" if gate.get("passes") is True else (
                    "未通过" if gate.get("estimable") is True else "不可估计"
                )
                print(
                    f"| {gate.get('label')} | {_format_metric(gate.get('value'))} | "
                    f"≥{_format_metric(gate.get('threshold'))} | {status} | 是 |"
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
    print("\n===== 探索性虚拟迭代四门槛（三臂匹配 facet） =====")
    print(
        f"总样本量：{config.get('sample_size', '未知')}；"
        f"固定顶层臂：{config.get('condition_count', statistics.get('condition_count', 3))}；"
        f"facet group：{config.get('group_count', statistics.get('group_count', '?'))}；"
        f"每组人数：{config.get('sample_size_per_condition', '未知')}；"
        "主施测每人每题一次，target组另做一次整卷重测"
    )
    print(
        "分面内题项一致性：CITC中位数="
        f"{_format_metric(virtual_summary.get('median_citc'))}；单题门槛CITC≥.20"
    )
    print(
        "三臂匹配 facet 相关：目标ρs中位数="
        f"{_format_metric(virtual_summary.get('median_target_rho'))}；"
        "最小同域VTS="
        f"{_format_metric(virtual_summary.get('minimum_same_domain_vts'))}；"
        "最小跨域VTS="
        f"{_format_metric(virtual_summary.get('minimum_cross_domain_vts'))}；"
        "单题门槛目标ρs≥.30、同域VTS≥.10、跨域VTS≥.20"
    )
    combined_state = {**state, **proposed_update}
    round_result = proposed_update.get("psychometric_round_result") or (
        build_psychometric_round_result(combined_state)
    )
    print_psychometric_round_result(round_result)
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
    print("\n===== 本轮临时组卷（单题返修前基线） =====")
    print(
        f"轮次={provisional.get('analysis_round', '?')}；"
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
    print(
        "稳定性门槛："
        "虚拟重测ICC="
        f"{_format_metric(reliability.get('virtual_test_retest_icc'))}；"
        f"最低={_format_metric(stability_gate.get('minimum'))}；"
        f"通过={'是' if stability_gate.get('passed') else '否'}"
    )
    usage = provisional.get("token_usage") or {}
    print(
        "本轮成本："
        f"Token={usage.get('total_tokens', 0)}；"
        f"模型耗时={usage.get('duration_ms', 0)} ms"
    )
    if plateau.get("reached"):
        print(
            "平台期：已达到，"
            f"连续未改善轮数={plateau.get('non_improving_rounds', '?')}"
        )
    if provisional.get("form_selection_error"):
        print(f"临时组卷备注：{provisional['form_selection_error']}")


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
            print_automatic_item_result(action, update, state)
            proposed_update = update.get("pending_state_update") or {}
            if action == "analyze_psychometrics":
                print_psychometric_summary(proposed_update, state)
            elif action == "select_items":
                print_selection_summary(proposed_update)
            print_workflow_effect(action, proposed_update)
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
        tasks = review.get("repair_tasks") or []
        decision = derive_item_review_decision(
            review,
            repair_attempted=bool(
                update.get("current_item_repair_attempted")
            ),
        )
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
    while True:
        choice = input("你的选择 [1-3]：").strip()
        if choice == "1":
            feedback = input("请用自然语言回答：").strip()
            if feedback:
                return {"decision": "answer", "feedback": feedback}
            print("回答不能为空")
            continue
        if choice == "2" and "accept_suggestions" in decisions:
            return {"decision": "accept_suggestions", "feedback": None}
        if choice == "3":
            feedback = input("停止原因（可留空）：").strip() or None
            return {"decision": "stop", "feedback": feedback}
        print("请输入有效选项")


def prompt_virtual_sample_selection(payload: dict) -> dict:
    """Collect three fixed arms with independently matched facet groups."""

    pool = payload.get("pool") or {}
    recommendations = payload.get("recommendations") or []
    recommended_size = payload.get("recommended_sample_size")
    default_seed = payload.get("default_seed", 7)
    default_max_concurrency = payload.get("default_max_concurrency", 5)
    default_max_retries = payload.get("default_max_retries", 2)

    print("\n===== 配置三臂匹配 facet 虚拟被试 =====")
    print(
        f"最多生成：{pool.get('available_count', '?')} 名；"
        "主施测中每名虚拟被试对每题回答1次；target组额外完成一次整卷重测。"
    )
    if payload.get("method_note"):
        print(f"说明：{payload['method_note']}")
    print(
        f"模型请求控制：最大并发 {default_max_concurrency}，"
        f"失败后最多重试 {default_max_retries} 次。"
    )
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")

    recommended_index = None
    print("\n推荐档位：")
    for index, option in enumerate(recommendations, 1):
        marker = "（推荐）" if option.get("recommended") else ""
        print(
            f"  {index}. {option.get('label')}："
            f"{option.get('sample_size')} 名{marker}"
        )
        print(f"     {option.get('description')}")
        if option.get("recommended"):
            recommended_index = index
    custom_index = len(recommendations) + 1
    print(f"  {custom_index}. 自定义人数")

    prompt_suffix = (
        f"，直接回车使用推荐值 {recommended_size}"
        if recommended_size is not None
        else ""
    )
    sample_size = None
    while sample_size is None:
        choice = input(
            f"你的选择 [1-{custom_index}{prompt_suffix}]："
        ).strip()
        if not choice and recommended_size is not None:
            sample_size = int(recommended_size)
            break
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(recommendations):
                sample_size = int(
                    recommendations[selected_index - 1]["sample_size"]
                )
                break
            if selected_index == custom_index:
                custom_value = input(
                    "请输入希望使用的虚拟被试人数："
                ).strip()
                if custom_value.isdigit():
                    sample_size = int(custom_value)
                    break
                print("人数必须是正整数")
                continue
        default_hint = (
            f"；推荐档位是 {recommended_index}"
            if recommended_index is not None
            else ""
        )
        print(f"请输入 1 到 {custom_index}{default_hint}")

    catalog = [
        row for row in payload.get("dimension_catalog") or []
        if isinstance(row, dict) and row.get("dimension_id")
    ]
    catalog_by_id = {str(row["dimension_id"]): row for row in catalog}
    targets = [
        row for row in catalog if row.get("required_target") is True
    ]
    def read_score(label: str) -> float:
        while True:
            raw_value = input(f"{label} 的样本平均分 [>0 且 <100]：").strip()
            try:
                value = float(raw_value)
            except ValueError:
                print("请输入0–100范围内的有限数值。")
                continue
            if math.isfinite(value) and 0.0 < value < 100.0:
                return value
            print("自动迭代需要非零方差，因此平均分须大于0且小于100。")

    if not targets:
        raise ValueError("题库没有可选择的目标 facet")
    print("\n请选择目标 facet：")
    for row in targets:
        print(f"  {row['dimension_id']} = {row.get('display_label', row['dimension_id'])}")
    print("\n请选择固定三臂中的 facet groups；同域/跨域臂可分别包含多个 group：")
    optional_facets = [
        row for row in catalog
        if not row.get("required_target") and row.get("level") == "facet"
    ]
    current_domain = None
    for row in optional_facets:
        domain = row.get("domain_name_en") or row.get("domain_name") or row.get("domain_id")
        if domain != current_domain:
            current_domain = domain
            print(f"  [{domain}]")
        print(f"    {row['dimension_id']} = {row.get('facet_name_en') or row.get('facet_name')}")
    def choose_facet(label: str, candidates: list[dict]) -> str:
        allowed = {str(row["dimension_id"]): row for row in candidates}
        while True:
            value = input(f"请输入{label} facet ID：").strip()
            if value in allowed:
                return value
            print("请选择当前列表中的 facet：" + "、".join(allowed))
    target_id = (
        str(targets[0]["dimension_id"])
        if len(targets) == 1
        else choose_facet("目标", targets)
    )
    target_row = catalog_by_id[target_id]
    target_domain = target_row.get("domain_id") if target_row else None
    same_candidates = [row for row in optional_facets if row.get("domain_id") == target_domain]
    cross_candidates = [row for row in optional_facets if row.get("domain_id") != target_domain]
    def choose_many(label: str, candidates: list[dict]) -> list[str]:
        if not candidates:
            raise ValueError(f"没有可用于{label}的 facet")
        while True:
            raw_count = input(f"请输入{label} group 数量（默认1）：").strip()
            if not raw_count:
                count = 1
            elif raw_count.isdigit():
                count = int(raw_count)
            else:
                print("group 数量必须是正整数")
                continue
            if 1 <= count <= len(candidates):
                break
            print(f"group 数量必须在1到{len(candidates)}之间")
        selected: list[str] = []
        remaining = list(candidates)
        for index in range(count):
            selected_id = choose_facet(f"{label}第{index + 1}", remaining)
            selected.append(selected_id)
            remaining = [row for row in remaining if str(row.get("dimension_id")) != selected_id]
        return selected

    same_ids = choose_many("同域非目标", same_candidates)
    cross_ids = choose_many("跨域非目标", cross_candidates)
    def read_numeric(prompt: str, default: float) -> float:
        while True:
            raw = input(f"{prompt}（默认 {default:g}）：").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                print("请输入有限数值")
    mean_score = read_numeric("请输入共享正态分布均值", 50.0)
    standard_deviation = read_numeric("请输入共享正态分布 SD", 15.0)
    def condition_row(condition_id: str, role: str, dimension_id: str) -> dict:
        return {
            **dict(catalog_by_id[dimension_id]),
            "condition_id": condition_id,
            "role": role,
            "dimension_id": dimension_id,
        }
    conditions = [
        {
            "condition_id": "target",
            "role": "target",
            "groups": [condition_row("target", "target", str(target_row["dimension_id"]))],
        },
        {
            "condition_id": "same_domain",
            "role": "same_domain_non_target",
            "groups": [condition_row(f"same_domain_group_{index + 1}", "same_domain_non_target", dimension_id) for index, dimension_id in enumerate(same_ids)],
        },
        {
            "condition_id": "cross_domain",
            "role": "cross_domain_non_target",
            "groups": [condition_row(f"cross_domain_group_{index + 1}", "cross_domain_non_target", dimension_id) for index, dimension_id in enumerate(cross_ids)],
        },
    ]

    return {
        "sample_size_per_condition": sample_size,
        "score_distribution": {"family": "normal", "mean": mean_score, "sd": standard_deviation},
        "conditions": conditions,
        "seed": default_seed,
        "max_concurrency": default_max_concurrency,
        "max_retries": default_max_retries,
    }


def prompt_user_decision(payload: dict) -> dict:
    """按任务类型收集一次有效的用户决策。"""

    if payload.get("type") == "virtual_sample_selection":
        return prompt_virtual_sample_selection(payload)

    if payload.get("type") == "post_virtual_response_decision":
        print(payload.get("summary") or "")
        round_result = payload.get("round_result") or {}
        if isinstance(round_result, dict) and round_result:
            print_psychometric_round_result(round_result)
        else:
            print("本轮缺少统一结果结构，请重新运行心理测量分析。")
        print_provisional_iteration_summary(payload)
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
            print("四门槛与最大污染facet：")
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

    latest = find_latest_resumable_checkpoint(checkpoint_root)
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
            return prepare_resumed_state(state)
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
            "请帮我出一个测量gregariousness的人格情境判断测验，测量大学生的，用于人格测评，一共16道题",
            target_population=None,
            target_construct=None,
            requested_item_count=None,
            # 为单题生成、审查和可能的修改循环保留足够执行步数。
            max_steps=1000,
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
