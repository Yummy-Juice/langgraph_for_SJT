"""Generate final test materials and technical reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from sjt_system.state import PSJTState
from sjt_system.delivery.lifecycle import evaluate_completion
from sjt_system.evaluation.round_results import (
    build_psychometric_round_result,
    metric_scalar,
)
from sjt_system.evaluation.respondents import MATCHED_CONDITION_SCHEMA_VERSION
from sjt_system.evaluation.form_metrics import form_quality_summary
from sjt_system.runtime.io import (
    write_json_atomic as _write_json_atomic,
    write_text_atomic as _write_text_atomic,
)
from sjt_system.runtime.trace import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_reports"
EVIDENCE_NOTICE = (
    "探索性虚拟筛查证据，不代表正式单题信效度或真实被试心理测量证据"
)


def _artifact_reference(path_value: object, *, label: str) -> dict[str, str]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"最终报告缺少{label}路径")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"最终报告引用的{label}不存在：{path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _score_protocol_artifacts(
    state: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Resolve the complete matched-condition evidence chain for final reports."""

    response_ref = _artifact_reference(
        state.get("virtual_response_data_ref"),
        label="虚拟作答manifest",
    )
    response_manifest = json.loads(
        Path(response_ref["path"]).read_text(encoding="utf-8")
    )
    if (
        response_manifest.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION
        or response_manifest.get("status") != "completed"
    ):
        raise ValueError("最终报告只接受已完成的匹配三臂虚拟作答manifest")
    output_files = (state.get("test_statistics") or {}).get("output_files") or {}
    if not isinstance(output_files, Mapping):
        raise ValueError("最终报告缺少虚拟迭代指标输出清单")
    response_path = Path(response_ref["path"])
    return {
        "response_manifest": response_ref,
        "score_profiles": _artifact_reference(
            response_manifest.get("score_profiles_path"),
            label="匹配条件个体得分档案",
        ),
        "sjt_responses": _artifact_reference(
            str(response_path.parent / "sjt_responses.jsonl"),
            label="匹配条件组单次SJT作答",
        ),
        "target_form_retest_responses": _artifact_reference(
            (
                (response_manifest.get("target_form_retest") or {}).get(
                    "path"
                )
            ),
            label="target整卷重测作答",
        ),
        "option_orders": _artifact_reference(
            response_manifest.get("option_order_path"),
            label="选项排列记录",
        ),
        "analysis_manifest": _artifact_reference(
            output_files.get("analysis_manifest"),
            label="心理测量分析manifest",
        ),
        "scored_target_form_retest_responses": _artifact_reference(
            output_files.get("scored_target_form_retest_sjt_responses"),
            label="计分后的target整卷重测作答",
        ),
        "virtual_screening_metrics": _artifact_reference(
            output_files.get("virtual_screening_metrics"),
            label="虚拟迭代指标表",
        ),
        "item_quality": _artifact_reference(
            output_files.get("item_quality"),
            label="单题质量表",
        ),
        "option_statistics": _artifact_reference(
            output_files.get("option_statistics"),
            label="选项诊断长表",
        ),
        "option_choice_diagnostics": _artifact_reference(
            output_files.get("option_choice_diagnostics"),
            label="选项诊断JSON",
        ),
    }


def _item_statuses(state: Mapping[str, Any]) -> dict[str, str]:
    selected = {
        item.get("item_id")
        for item in state.get("selected_items") or []
        if isinstance(item, Mapping)
    }
    reserve = {
        item.get("item_id")
        for item in state.get("reserve_items") or []
        if isinstance(item, Mapping)
    }
    deferred = {
        item_id
        for item_id in (
            (state.get("selection_results") or {}).get(
                "deferred_revision_item_ids"
            )
            or []
        )
        if item_id
    }
    removed = {
        item.get("item_id")
        for item in state.get("removed_items") or []
        if isinstance(item, Mapping)
    }
    statuses: dict[str, str] = {}
    for item_id, disposition in (state.get("item_final_dispositions") or {}).items():
        if isinstance(disposition, Mapping) and disposition.get("status"):
            statuses[str(item_id)] = str(disposition["status"])
    for item_id in removed:
        if item_id:
            statuses.setdefault(str(item_id), "removed")
    for item_id in deferred:
        statuses[str(item_id)] = "deferred_revision"
    for item_id in reserve:
        if item_id:
            statuses[str(item_id)] = "reserve"
    for item_id in selected:
        if item_id:
            statuses[str(item_id)] = (
                "formal_qualified_locked"
                if statuses.get(str(item_id)) == "qualified_locked"
                else "selected"
            )
    return statuses


def _display_value(value: Any, *, fallback: str = "证据不足") -> str:
    if isinstance(value, Mapping):
        numeric = metric_scalar(value)
        return f"{numeric:.3f}" if numeric is not None else fallback
    if value is None:
        return fallback
    if isinstance(value, float):
        numeric = metric_scalar(value)
        return f"{numeric:.3f}" if numeric is not None else fallback
    return str(value)


def _round_result_markdown(round_result: Mapping[str, Any]) -> list[str]:
    summary = round_result.get("summary") or {}
    lines = [
        "## 最近一轮虚拟筛查总览",
        "",
        f"- 分析轮次：{round_result.get('analysis_round', '未记录')}",
        f"- 分析题目：{summary.get('item_count', 0)}",
        f"- 本轮新合格：{summary.get('newly_qualified_count', 0)}",
        f"- 尚待处理：{summary.get('pending_treatment_count', 0)}",
        f"- 已锁定正式题：{summary.get('qualified_locked_count', 0)}",
        f"- 正式题监测警告：{summary.get('monitoring_warning_count', 0)}",
        f"- 不可估计门槛：{summary.get('unestimable_metric_count', 0)}",
        "",
        "| 门槛 | 通过题数 | 阈值 | 不可估计 |",
        "|---|---:|---:|---:|",
    ]
    for gate in round_result.get("gate_summary") or []:
        if isinstance(gate, Mapping):
            lines.append(
                f"| {gate.get('label')} | {gate.get('pass_count', 0)}/"
                f"{gate.get('item_count', 0)} | ≥{_display_value(gate.get('threshold'))} | "
                f"{gate.get('unestimable_count', 0)} |"
            )
    lines.extend(
        [
            "",
            "| 题目 | 状态 | CITC | 目标rho_s | 同域VTS | 同域最大带符号rho facet | 跨域VTS | 跨域最大带符号rho facet |",
            "|---|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for entry in round_result.get("items") or []:
        if not isinstance(entry, Mapping):
            continue
        gates = {
            row.get("gate_id"): row
            for row in entry.get("gates") or []
            if isinstance(row, Mapping)
        }
        contaminants = entry.get("max_contaminants") or {}
        same = contaminants.get("same_domain") or {}
        cross = contaminants.get("cross_domain") or {}
        lines.append(
            f"| {entry.get('item_id')} | {entry.get('status')} | "
            f"{_display_value((gates.get('citc_pass') or {}).get('value'))} | "
            f"{_display_value((gates.get('target_rho_pass') or {}).get('value'))} | "
            f"{_display_value((gates.get('same_domain_vts_pass') or {}).get('value'))} | "
            f"{same.get('facet_name') or same.get('dimension_id') or '-'}（rho={_display_value(same.get('signed_rho'))}） | "
            f"{_display_value((gates.get('cross_domain_vts_pass') or {}).get('value'))} | "
            f"{cross.get('facet_name') or cross.get('dimension_id') or '-'}（rho={_display_value(cross.get('signed_rho'))}） |"
        )
    lines.extend(
        [
            "",
            "### 所有题目的按选项对齐设定分数均值",
            "",
            "均值按各实验臂内选择同一选项的虚拟被试计算，filtering_authority=false。",
            "",
            "| 题目 | 选项 | 计分 | 目标N | 目标组均值 | 同域N | 同域组均值 | 跨域N | 跨域组均值 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in round_result.get("option_score_comparisons") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('item_id')} | {row.get('option_id')} | "
            f"{_display_value(row.get('option_score'))} | "
            f"{row.get('target_n', 0)} | {_display_value(row.get('target_mean_score'))} | "
            f"{row.get('same_domain_n', 0)} | {_display_value(row.get('same_domain_mean_score'))} | "
            f"{row.get('cross_domain_n', 0)} | {_display_value(row.get('cross_domain_mean_score'))} |"
        )
    lines.append("")
    for entry in round_result.get("pending_items") or []:
        if not isinstance(entry, Mapping):
            continue
        lines.extend(
            [
                f"### 待处理题 {entry.get('item_id')}",
                "",
                "失败门槛：" + "、".join(entry.get("failed_thresholds") or []),
                "",
                "| 指标 | 当前值 | 阈值 | 状态 | 筛选权 |",
                "|---|---:|---:|---|---|",
            ]
        )
        for gate in entry.get("gates") or []:
            if isinstance(gate, Mapping):
                status = "通过" if gate.get("passes") is True else (
                    "未通过" if gate.get("estimable") is True else "不可估计"
                )
                lines.append(
                    f"| {gate.get('label')} | {_display_value(gate.get('value'))} | "
                    f"≥{_display_value(gate.get('threshold'))} | {status} | true |"
                )
        score_comparisons = [
            row for row in entry.get("option_score_comparisons") or []
            if isinstance(row, Mapping)
        ]
        if score_comparisons:
            lines.extend(
                [
                    "",
                    "按 option_id 对齐的设定分数均值仅用于定位，filtering_authority=false。",
                    "",
                    "| 选项 | 计分 | 目标N | 目标组均值 | 同域N | 同域组均值 | 跨域N | 跨域组均值 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for comparison in score_comparisons:
                lines.append(
                    f"| {comparison.get('option_id')} | "
                    f"{_display_value(comparison.get('option_score'))} | "
                    f"{comparison.get('target_n', 0)} | {_display_value(comparison.get('target_mean_score'))} | "
                    f"{comparison.get('same_domain_n', 0)} | {_display_value(comparison.get('same_domain_mean_score'))} | "
                    f"{comparison.get('cross_domain_n', 0)} | {_display_value(comparison.get('cross_domain_mean_score'))} |"
                )
        lines.append("")
    return lines


def _repair_evidence(event: Mapping[str, Any]) -> str:
    baseline = event.get("baseline_metrics") or {}
    quality = baseline.get("quality_evaluation") or {}
    citc = quality.get("facet_citc") or {}
    specificity = quality.get("virtual_target_specificity") or {}
    if citc or specificity:
        target = specificity.get("rho_target") or specificity.get("target_spearman") or {}
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        return (
            "分面内CITC="
            + _display_value(citc.get("r"))
            + "；目标ρs="
            + _display_value(target.get("rho"))
            + "；同域VTS="
            + _display_value(same_domain.get("specificity_margin"))
            + "；跨域VTS="
            + _display_value(cross_domain.get("specificity_margin"))
        )
    corrected = (
        baseline.get("facet_corrected_item_total_correlation")
        or baseline.get("corrected_item_total_correlation")
        or {}
    )
    return f"分面内CITC={_display_value(corrected.get('r'))}"


def _iteration_history_markdown(
    history: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines = [
        "## 整卷迭代曲线数据",
        "",
        "每轮先组成临时测验，再计算整卷虚拟开发期指标；Token和耗时为模型调用记录。",
        "",
        "| 轮次 | 临时题数 | 候选题数 | Cronbach α | 目标Hedges’ g | 目标IPIP rho | Δmin区分效度 | ICC门槛 | 平台期 | 累计Token | 累计模型耗时(ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|",
    ]
    cumulative_tokens = 0
    cumulative_duration = 0
    data_rows = 0
    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        data_rows += 1
        metrics = entry.get("form_metrics") or {}
        validity = metrics.get("validity") or {}
        convergent = validity.get("convergent_validity") or {}
        discriminant = validity.get("discriminant_validity") or {}
        known_groups = validity.get("known_groups_validity") or {}
        quality = form_quality_summary(metrics)
        uses_ipip_objective = quality.get("objective_source") == "ipip_human_style_v3"
        usage = entry.get("token_usage") or {}
        cumulative_tokens += int(usage.get("total_tokens") or 0)
        cumulative_duration += int(usage.get("duration_ms") or 0)
        plateau = entry.get("plateau_status") or {}
        best_quality = entry.get("best_so_far_form_quality")
        if best_quality is None:
            best_quality = plateau.get("best_form_quality")
        objective_primary = entry.get("objective_primary")
        if objective_primary is None:
            objective_primary = quality.get("objective_primary")
        if not uses_ipip_objective:
            best_quality = None
            objective_primary = None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.get("analysis_round") or "未记录"),
                    str(entry.get("item_count") or 0),
                    str(entry.get("candidate_count") or 0),
                    _display_value((quality.get("alpha_gate") or {}).get("observed")),
                    _display_value(known_groups.get("target_hedges_g")),
                    _display_value(
                        convergent.get("spearman_rho")
                        if uses_ipip_objective
                        else None
                    ),
                    _display_value(
                        discriminant.get("delta_min")
                        if uses_ipip_objective
                        else None
                    ),
                    "通过" if (quality.get("stability_gate") or {}).get("passed") else "未通过",
                    str(plateau.get("status", "未开始")),
                    str(cumulative_tokens),
                    str(cumulative_duration),
                ]
            )
            + " |"
        )
    if data_rows == 0:
        lines.append("| 无 | - | - | - | - | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "> 当前整卷迭代以目标IPIP facet高低组Hedges’ g为主要优化指标；Δmin不得下降，目标facet Spearman rho最多下降0.02。Cronbach α和ICC是整卷门槛。",
            "> 连续2轮候选卷未使历史最优目标Hedges’ g提高至少0.01后，系统自动进入平台期并停止继续返修。",
            "> 这些是虚拟作答系统内部的开发期传导指标，不能替代真人样本的正式信效度验证。",
            "",
        ]
    )
    return lines


def _technical_report_markdown(
    technical_report: Mapping[str, Any],
) -> str:
    test_quality = technical_report.get("test_quality") or {}
    reliability = test_quality.get("reliability") or {}
    validity = test_quality.get("validity") or {}
    convergent = validity.get("convergent") or {}
    discriminant = validity.get("discriminant") or {}
    selection = technical_report.get("selection_results") or {}
    optimizer = (
        technical_report.get("blueprint_coverage") or {}
    ).get("optimizer") or {}
    selected_metrics = optimizer.get("selected_metrics") or {}
    developmental_status = technical_report.get("development_status")
    provisional_item_ids = technical_report.get("provisional_item_ids") or []
    repair_history = technical_report.get("psychometric_repair_history") or []
    selection_history = (
        technical_report.get("psychometric_selection_history") or []
    )
    test_statistics = technical_report.get("test_statistics") or {}
    screening = test_statistics.get("virtual_screening_metrics") or {}
    screening_summary = screening.get("summary") or {}
    facet_statistics = test_statistics.get("dimensions") or {}
    item_statistics = technical_report.get("item_statistics") or {}
    round_result = technical_report.get("psychometric_round_result") or {}
    iteration_history = technical_report.get("psychometric_iteration_history") or []
    item_pools = technical_report.get("item_pools") or {}
    item_lineage = technical_report.get("item_lineage") or {}
    condition_diagnostics = technical_report.get("condition_score_diagnostics") or {}
    condition_diagnostic_lines = [
        "| 条件 | Facet | 均值 | SD | 筛选权 |",
        "|---|---|---:|---:|---|",
    ]
    for condition in condition_diagnostics.get("conditions") or []:
        if isinstance(condition, Mapping):
            condition_diagnostic_lines.append(
                f"| {condition.get('condition_id')} | {condition.get('dimension_id')} | "
                f"{_display_value(condition.get('actual_mean'))} | "
                f"{_display_value(condition.get('actual_sample_sd'))} | false |"
            )
    if len(condition_diagnostic_lines) == 2:
        condition_diagnostic_lines.append("| 无 | - | - | - | false |")
    lineage_lines = [
        "| 题目 | 根题号 | 替代题号 | 被替代题号 | 状态 |",
        "|---|---|---|---|---|",
        *[
            "| "
            + " | ".join(
                str(value or "-").replace("|", "｜")
                for value in (
                    item_id,
                    row.get("root_item_id"),
                    row.get("replaced_by_item_id"),
                    row.get("replaces_item_id"),
                    row.get("status"),
                )
            )
            + " |"
            for item_id, row in item_lineage.items()
            if isinstance(row, Mapping)
        ],
    ]
    if len(lineage_lines) == 2:
        lineage_lines.append("| 无 | - | - | - | - |")
    facet_lines: list[str] = ["## 分面测量结果", ""]
    for facet_id, facet in facet_statistics.items():
        facet_lines.extend(
            [
                f"### {facet_id}",
                "",
                f"- 题数：{facet.get('item_count', 0)}",
                "- Cronbach α："
                + _display_value(facet.get("cronbach_alpha")),
                "",
                (
                    "| 题目 | 目标组CITC | 目标ρs | 同域最大带符号rho facet | 同域最大带符号ρs | 同域VTS | "
                    "跨域最大带符号rho facet | 跨域最大带符号ρs | 跨域VTS | 结果 | 难度(描述) | 有效选项(描述) |"
                ),
                "|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|",
            ]
        )
        tier_diagnostic_lines: list[str] = []
        for item_id in facet.get("item_ids") or []:
            item = item_statistics.get(item_id) or {}
            quality = item.get("quality_evaluation") or {}
            citc = quality.get("facet_citc") or {}
            specificity = quality.get("virtual_target_specificity") or {}
            target = specificity.get("rho_target") or specificity.get("target_spearman") or {}
            same_domain = specificity.get("same_domain_non_target") or {}
            cross_domain = specificity.get("cross_domain_non_target") or {}
            facet_lines.append(
                f"| {item_id} | "
                f"{_display_value(citc.get('r'))} | "
                f"{_display_value(target.get('rho'))} | "
                f"{_display_value(same_domain.get('facet_name') or same_domain.get('largest_non_target_facet_name'))} | "
                f"{_display_value(same_domain.get('max_non_target_rho') if same_domain.get('max_non_target_rho') is not None else same_domain.get('largest_non_target_conditional_rho'))} | "
                f"{_display_value(same_domain.get('specificity_margin'))} | "
                f"{_display_value(cross_domain.get('facet_name') or cross_domain.get('largest_non_target_facet_name'))} | "
                f"{_display_value(cross_domain.get('max_non_target_rho') if cross_domain.get('max_non_target_rho') is not None else cross_domain.get('largest_non_target_conditional_rho'))} | "
                f"{_display_value(cross_domain.get('specificity_margin'))} | "
                f"{_display_value(quality.get('recommendation'))} | "
                f"{_display_value(item.get('difficulty'))} | "
                f"{_display_value(quality.get('effective_option_count'))} |"
            )
            per_condition = quality.get("per_condition_metrics") or {}
            if per_condition:
                tier_diagnostic_lines.append(
                    f"- {item_id} 各条件臂诊断："
                )
                for condition_id, condition_metric in per_condition.items():
                    tier_diagnostic_lines.append(
                        "  - "
                        + str(condition_id)
                        + ": CITC="
                        + _display_value(((condition_metric or {}).get("citc") or {}).get("r"))
                        + "，ρs="
                        + _display_value(((condition_metric or {}).get("rho") or {}).get("rho"))
                    )
        facet_lines.append("")
        if tier_diagnostic_lines:
            facet_lines.extend(
                ["各条件臂指标：", "", *tier_diagnostic_lines, ""]
            )
    lines = [
        "# PSJT开发技术报告",
        "",
        f"> {EVIDENCE_NOTICE}",
        "",
        "## 开发概况",
        "",
        f"- 题库版本：{technical_report.get('item_bank_version')}",
        f"- 正式题数：{technical_report.get('selected_item_count')}",
        f"- 备选题数：{technical_report.get('reserve_item_count')}",
        (
            "- 交付等级：开发版（含正式题监测警告，既有资格未撤销）"
            if developmental_status == "developmental"
            else "- 交付等级：标准候选版"
        ),
        f"- 暂定题数：{len(provisional_item_ids)}",
        (
            "- 待后续返修题数："
            f"{technical_report.get('deferred_revision_count', 0)}"
        ),
        (
            "- 三轮返修仍未达标并自动 defer："
            f"{technical_report.get('deferred_decision_count', 0)}"
        ),
        "- 心理骨架审核：按固定槽位逐题独立审核",
        "- 心理测量处理：结构化诊断、定向返修、重新模拟与重新分析",
        "",
        "## 候选题库探索性虚拟筛查结果",
        "",
        f"- 开发状态：{_display_value(test_quality.get('overall_status'))}",
        (
            "- 使用状态："
            + _display_value(test_quality.get("operational_use_status"))
        ),
        (
            "- Cronbach α（描述性）："
            + _display_value(reliability.get("cronbach_alpha"))
        ),
        (
            "- 分面内CITC中位数："
            + _display_value(screening_summary.get("median_citc"))
        ),
        (
            "- 目标条件组ρs中位数："
            + _display_value(screening_summary.get("median_target_rho"))
        ),
        (
            "- 最小同domain非目标facet VTS："
            + _display_value(screening_summary.get("minimum_same_domain_vts"))
        ),
        (
            "- 最小不同domain非目标facet VTS："
            + _display_value(screening_summary.get("minimum_cross_domain_vts"))
        ),
        "- 自动返修门槛：目标组CITC≥.20、目标ρs≥.30、同域VTS≥.10且跨域VTS≥.20",
        "- 固定三个顶层臂；每个非目标facet group独立使用同一匹配分数序列，VTS取臂内最大带符号rho；CITC过滤权仅属于目标臂",
        "- 条件组分数分布仅作设计核查，filtering_authority=false，不参与题目筛选",
        "- 难度、选项使用率与Cronbach alpha仅为描述性诊断",
        (
            "- 结论："
            + _display_value(test_quality.get("interpretation"))
        ),
        "",
        *_round_result_markdown(round_result),
        *_iteration_history_markdown(iteration_history),
        "## 补充：匹配条件组分数分布",
        "",
        *condition_diagnostic_lines,
        "",
        *facet_lines,
        "## 最终正式组合的开发期指标",
        "",
        (
            "- 最差人格模式目标相关："
            + _display_value(selected_metrics.get("worst_case_target_rho"))
        ),
        (
            "- 最差人格模式区分差值："
            + _display_value(
                selected_metrics.get("worst_case_discriminant_margin")
            )
        ),
        (
            "- 最差人格模式 Cronbach α："
            + _display_value(
                selected_metrics.get("worst_case_cronbach_alpha")
            )
        ),
        (
            "- 题间冗余："
            + _display_value(
                selected_metrics.get(
                    "worst_case_inter_item_redundancy"
                )
            )
            + "；语义冗余："
            + _display_value(selected_metrics.get("semantic_redundancy"))
        ),
        "",
        "## 题目筛选与组卷",
        "",
        (
            "- 筛选状态："
            + _display_value(selection.get("status"), fallback="未执行")
        ),
        (
            "- 自动淘汰是否抑制："
            + ("是" if selection.get("automatic_removal_suppressed") else "否")
        ),
        (
            "- 组合优化："
            + _display_value(optimizer.get("method"), fallback="未执行")
            + f"；候选组合={optimizer.get('combination_count', 0)}"
        ),
        (
            f"- 心理测量分析轮数：{len(selection_history)}；"
            f"完成返修事件：{len(repair_history)}；"
            f"重新组卷轮数：{technical_report.get('reassembly_round', 0)}"
        ),
        "",
        "## 题目去向",
        "",
        "- 已锁定正式题：" + ("、".join(item_pools.get("formal_items") or []) or "无"),
        "- 尚待处理题：" + ("、".join(item_pools.get("pending_items") or []) or "无"),
        "- 待SME审核题：" + ("、".join(item_pools.get("pending_sme_review") or []) or "无"),
        "- 已淘汰题：" + ("、".join(item_pools.get("eliminated_items") or []) or "无"),
        "",
        "| 题目 | 最终状态 | 统计建议 | 原因 |",
        "|---|---|---|---|",
        *[
            "| "
            + " | ".join(
                str(row.get(key) or "未记录").replace("|", "｜")
                for key in (
                    "item_id",
                    "final_status",
                    "recommendation",
                    "reason",
                )
            )
            + " |"
            for row in technical_report.get("item_decisions") or []
        ],
        "",
        "## 补题 Lineage",
        "",
        *lineage_lines,
        "",
        "## 心理测量返修记录",
        "",
        "| 题目 | 轮次 | 动作 | 结果 | 返修前证据 |",
        "|---|---:|---|---|---|",
        *[
            "| "
            + " | ".join(
                [
                    str(event.get("item_id") or "未记录").replace("|", "｜"),
                    str(event.get("revision_round") or "未记录"),
                    str(event.get("action") or "未记录").replace("|", "｜"),
                    (
                        "通过"
                        if event.get("event") == "psychometric_item_repaired"
                        else "未通过并退出"
                    ),
                    _repair_evidence(event).replace("|", "｜"),
                ]
            )
            + " |"
            for event in repair_history
            if isinstance(event, Mapping)
        ],
        *(
            ["| 无 | - | - | - | - |"]
            if not repair_history
            else []
        ),
        "",
        "## 证据边界",
        "",
        *(
            [technical_report["context_source_notice"], ""]
            if technical_report.get("context_source_notice")
            else []
        ),
        *(
            [technical_report["construct_registry_notice"], ""]
            if technical_report.get("construct_registry_notice")
            else []
        ),
        EVIDENCE_NOTICE + "。",
        (
            "显式人格分数与SJT作答由同一模型和提示流程连接，相关结果不能"
            "替代独立SME判断、真人构念效度或真实外部效标。"
        ),
        "",
    ]
    return "\n".join(lines)


def run_report_generation(
    state: PSJTState,
    *,
    output_root: str | Path = DEFAULT_REPORT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Persist final materials using only evidence already in State."""

    review = state.get("test_review_result")
    assembled = state.get("assembled_test")
    if not isinstance(review, Mapping) or review.get("decision") != "PASS":
        raise ValueError("只有测验级审核 PASS 后才能生成最终报告")
    if not isinstance(assembled, Mapping):
        raise ValueError("生成最终报告前缺少 assembled_test")
    if (
        assembled.get("item_bank_id") != state.get("item_bank_id")
        or assembled.get("item_bank_version")
        != state.get("item_bank_version")
    ):
        raise ValueError("assembled_test 与当前题库版本不一致")

    fingerprint = str(state.get("item_bank_fingerprint") or "unknown")
    output_dir = (
        Path(output_root)
        / str(state["run_id"])
        / f"bank-v{state['item_bank_version']}-{fingerprint[:12]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_test_path = output_dir / "final_test.json"
    item_database_path = output_dir / "item_database.json"
    technical_report_path = output_dir / "technical_report.json"
    technical_markdown_path = output_dir / "technical_report.md"
    virtual_report_path = output_dir / "virtual_respondent_report.json"
    report_manifest_path = output_dir / "report_manifest.json"
    generated_at = utc_timestamp()
    score_protocol_artifacts = _score_protocol_artifacts(state)

    final_test = {
        "schema_version": 1,
        "test_id": assembled.get("test_id"),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "respondent_form": deepcopy(assembled.get("respondent_form")),
        "scoring_key": deepcopy(assembled.get("scoring_key")),
        "blueprint_summary": deepcopy(assembled.get("blueprint_summary")),
        "development_status": (
            "developmental" if assembled.get("provisional") else "standard"
        ),
        "quality_gate": deepcopy(assembled.get("quality_gate")),
        "delivery_warnings": deepcopy(
            (state.get("selection_results") or {}).get("delivery_warnings")
            or []
        ),
        "assembly_files": deepcopy(assembled.get("files")),
        "evidence_notice": EVIDENCE_NOTICE,
        "finalized_at": generated_at,
    }
    statuses = _item_statuses(state)
    item_statistics = state.get("item_statistics") or {}
    database_items = []
    known_ids: set[str] = set()
    for raw_item in state.get("frozen_item_bank") or []:
        if not isinstance(raw_item, Mapping):
            continue
        item = deepcopy(dict(raw_item))
        item_id = str(item.get("item_id"))
        known_ids.add(item_id)
        database_items.append(
            {
                **item,
                "final_status": statuses.get(item_id, "unclassified"),
                "psychometric_statistics": deepcopy(
                    item_statistics.get(item_id)
                ),
                "provisional_quality_flag": deepcopy(
                    (state.get("provisional_item_flags") or {}).get(item_id)
                ),
            }
        )
    for raw_item in state.get("removed_items") or []:
        if not isinstance(raw_item, Mapping):
            continue
        item_id = str(raw_item.get("item_id"))
        if item_id in known_ids:
            continue
        database_items.append(
            {
                **deepcopy(dict(raw_item)),
                "final_status": "removed",
                "psychometric_statistics": None,
            }
        )
    item_database = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "items": database_items,
        "generated_at": generated_at,
    }

    measurement_evaluation = (
        (state.get("test_statistics") or {}).get(
            "measurement_evaluation"
        )
        or {}
    )
    decision_index: dict[str, dict[str, Any]] = {}
    for history_entry in state.get("psychometric_selection_history") or []:
        if not isinstance(history_entry, Mapping):
            continue
        recommendations = (
            history_entry.get("effective_recommendations")
            or history_entry.get("source_recommendations")
            or {}
        )
        reasons = history_entry.get("reasons") or {}
        for item_id, recommendation in recommendations.items():
            decision_index[str(item_id)] = {
                "recommendation": recommendation,
                "reason": reasons.get(item_id) or "未记录详细原因",
            }
    current_recommendations = (
        (state.get("selection_results") or {}).get(
            "effective_recommendations"
        )
        or (state.get("selection_results") or {}).get(
            "source_recommendations"
        )
        or {}
    )
    for item_id, recommendation in current_recommendations.items():
        decision_index[str(item_id)] = {
            "recommendation": recommendation,
            "reason": (state.get("selection_reasons") or {}).get(item_id)
            or decision_index.get(str(item_id), {}).get("reason")
            or "未记录详细原因",
        }
    final_dispositions = state.get("item_final_dispositions") or {}
    for item_id, disposition in final_dispositions.items():
        if isinstance(disposition, Mapping):
            decision_index[str(item_id)] = {
                "recommendation": disposition.get("status"),
                "reason": (
                    disposition.get("warning_reason")
                    or (state.get("selection_reasons") or {}).get(item_id)
                    or "accepted"
                ),
            }
    all_decision_ids = list(
        dict.fromkeys(
            [
                *decision_index,
                *statuses,
                *[
                    str(item.get("item_id"))
                    for item in state.get("removed_items") or []
                    if isinstance(item, Mapping) and item.get("item_id")
                ],
            ]
        )
    )
    item_decisions = [
        {
            "item_id": item_id,
            "final_status": (
                (final_dispositions.get(item_id) or {}).get("status")
                or statuses.get(item_id, "unclassified")
            ),
            "recommendation": decision_index.get(item_id, {}).get(
                "recommendation"
            ),
            "reason": decision_index.get(item_id, {}).get("reason"),
        }
        for item_id in all_decision_ids
    ]
    current_item_ids = {
        str(item.get("item_id"))
        for item in state.get("item_pool") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    locked_ids = {
        str(item_id)
        for item_id, disposition in final_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "qualified_locked"
    }
    pending_sme_ids = {
        str(item_id)
        for item_id, disposition in final_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "pending_sme_review"
    }
    eliminated_ids = {
        str(item_id)
        for item_id, disposition in final_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "eliminated"
    }
    item_pools = {
        "formal_items": sorted(locked_ids),
        "pending_items": sorted(
            current_item_ids - locked_ids - pending_sme_ids - eliminated_ids
        ),
        "pending_sme_review": sorted(pending_sme_ids),
        "eliminated_items": sorted(eliminated_ids),
    }
    round_result = build_psychometric_round_result(state)
    deferred_decision_entries = [
        entry
        for entry in [
            *(state.get("items_to_revise") or []),
            *(state.get("items_to_regenerate") or []),
        ]
        if isinstance(entry, Mapping)
        and entry.get("queue_status") == "deferred_decision"
        and entry.get("diagnosis_status") == "repair_rounds_exhausted"
    ]
    technical_report = {
        "schema_version": 3,
        "run_id": state["run_id"],
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "test_specification": deepcopy(state.get("test_specification")),
        "construct_profile_ref": deepcopy(
            (state.get("blueprint") or {}).get("construct_profile_ref")
        ),
        "construct_profile": deepcopy(state.get("construct_profile")),
        "blueprint": deepcopy(state.get("blueprint")),
        "skeleton_reviews": deepcopy(state.get("skeleton_reviews") or {}),
        "context_source_notice": (
            "情境类别由模型根据目标人群与使用需求合成，仅作为内容设计候选；"
            "不代表访谈资料、实证频率或效标证据。"
            if state.get("construct_profile") is not None
            else None
        ),
        "construct_registry_notice": (
            "本次构念语义来自版本化种子注册表；其来源边界和本地化表述仍应"
            "在正式使用前由心理测量领域专家审定。"
            if isinstance(state.get("construct_profile"), Mapping)
            and state["construct_profile"].get("review_status")
            == "versioned_seed"
            else None
        ),
        "selected_item_count": len(state.get("selected_items") or []),
        "reserve_item_count": len(state.get("reserve_items") or []),
        "deferred_revision_count": len(
            state.get("items_deferred_for_revision") or []
        ),
        "deferred_decision_count": len(deferred_decision_entries),
        "removed_item_count": len(state.get("removed_items") or []),
        "development_status": (
            "developmental" if assembled.get("provisional") else "standard"
        ),
        "provisional_item_ids": deepcopy(
            (state.get("selection_results") or {}).get("provisional_item_ids")
            or []
        ),
        "provisional_item_flags": deepcopy(
            state.get("provisional_item_flags") or {}
        ),
        "test_statistics": deepcopy(state.get("test_statistics")),
        "virtual_sample_config": deepcopy(state.get("virtual_sample_config")),
        "condition_score_diagnostics": deepcopy(
            (state.get("virtual_sample_config") or {}).get(
                "generation_diagnostics"
            )
            or {}
        ),
        "item_statistics": deepcopy(state.get("item_statistics") or {}),
        "psychometric_analysis_round": state.get("psychometric_analysis_round", 0),
        "psychometric_round_result": round_result,
        "psychometric_iteration_history": deepcopy(
            state.get("psychometric_iteration_history") or []
        ),
        "test_quality": deepcopy(measurement_evaluation),
        "selection_results": deepcopy(state.get("selection_results")),
        "item_final_dispositions": deepcopy(
            state.get("item_final_dispositions") or {}
        ),
        "item_pools": item_pools,
        "item_lineage": deepcopy(state.get("item_lineage") or {}),
        "psychometric_monitoring_warnings": deepcopy(
            state.get("psychometric_monitoring_warnings") or []
        ),
        "locked_retained_item_versions": deepcopy(
            state.get("locked_retained_item_versions") or {}
        ),
        "best_assembly_candidate": deepcopy(
            state.get("best_assembly_candidate")
        ),
        "blueprint_coverage": deepcopy(state.get("blueprint_coverage")),
        "test_review_result": deepcopy(state.get("test_review_result")),
        "psychometric_selection_history": deepcopy(
            state.get("psychometric_selection_history") or []
        ),
        "psychometric_repair_history": deepcopy(
            state.get("psychometric_repair_history") or []
        ),
        "item_decisions": item_decisions,
        "reassembly_round": state.get("reassembly_round", 0),
        "test_revision_history": deepcopy(
            state.get("test_revision_history") or []
        ),
        "evidence_notice": EVIDENCE_NOTICE,
        "generated_at": generated_at,
    }
    virtual_report = {
        "schema_version": 4,
        "run_id": state["run_id"],
        "sample_config": deepcopy(state.get("virtual_sample_config")),
        "respondent_count": len(state.get("virtual_respondents") or []),
        "response_summary": deepcopy(
            state.get("virtual_response_summary")
        ),
        "virtual_screening_metrics": deepcopy(
            (state.get("test_statistics") or {}).get(
                "virtual_screening_metrics"
            )
        ),
        "psychometric_round_result": round_result,
        "psychometric_iteration_history": deepcopy(
            state.get("psychometric_iteration_history") or []
        ),
        "artifacts": deepcopy(score_protocol_artifacts),
        "response_manifest": state.get("virtual_response_data_ref"),
        "item_bank_id": state.get("virtual_response_item_bank_id"),
        "item_bank_version": state.get(
            "virtual_response_item_bank_version"
        ),
        "interpretation_limitations": [
            EVIDENCE_NOTICE,
            (
                "人格分数操纵与SJT作答来自同一模型提示流程，相关性可能"
                "受到共同方法、提示响应和语义重叠影响。"
            ),
            "主迭代未调用人格总结；target组独立完成目标facet对应的IPIP-NEO参照量表，固定三个顶层臂下每个facet group分别匹配，每名被试对每题各作答一次。",
            "本报告不包含真实被试信度、真人效度或人口学DIF证据；虚拟重测ICC只描述模型作答稳定性。",
        ],
        "generated_at": generated_at,
    }

    _write_json_atomic(final_test_path, final_test)
    _write_json_atomic(item_database_path, item_database)
    _write_json_atomic(technical_report_path, technical_report)
    _write_text_atomic(
        technical_markdown_path,
        _technical_report_markdown(technical_report),
    )
    _write_json_atomic(virtual_report_path, virtual_report)

    # 生成被试视角的正式测验表单（HTML，可打印为 PDF）；失败不阻塞主流程
    try:
        from sjt_system.delivery.test_pdf import build_test_form_html

        build_test_form_html(
            {
                **state,
                "final_test": final_test,
                "item_database": item_database,
            },
            output_dir / "test_form.html",
        )
    except Exception:
        pass

    proposed_state = {
        **state,
        "final_test": final_test,
        "item_database_ref": str(item_database_path.resolve()),
        "technical_report": technical_report,
        "virtual_respondent_report": virtual_report,
    }
    checks, unmet = evaluate_completion(proposed_state)
    manifest = {
        "schema_version": 2,
        "run_id": state["run_id"],
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "generated_at": generated_at,
        "evidence_notice": EVIDENCE_NOTICE,
        "evidence_scope": "exploratory_virtual_screening_evidence",
        "virtual_sample_config": deepcopy(
            state.get("virtual_sample_config")
        ),
        "screening_criteria": deepcopy(
            (state.get("test_statistics") or {}).get(
                "qualification_criteria"
            )
        ),
        "completion_checks": checks,
        "unmet_completion_conditions": unmet,
        "evidence_files": deepcopy(score_protocol_artifacts),
        "files": {
            "final_test": str(final_test_path.resolve()),
            "item_database": str(item_database_path.resolve()),
            "technical_report": str(technical_report_path.resolve()),
            "technical_report_markdown": str(
                technical_markdown_path.resolve()
            ),
            "virtual_respondent_report": str(
                virtual_report_path.resolve()
            ),
        },
    }
    _write_json_atomic(report_manifest_path, manifest)
    return {
        "state_update": {
            "final_test": {
                **final_test,
                "file_path": str(final_test_path.resolve()),
                "report_manifest_path": str(
                    report_manifest_path.resolve()
                ),
            },
            "item_database_ref": str(item_database_path.resolve()),
            "technical_report": {
                **technical_report,
                "file_path": str(technical_report_path.resolve()),
                "markdown_path": str(
                    technical_markdown_path.resolve()
                ),
            },
            "virtual_respondent_report": {
                **virtual_report,
                "file_path": str(virtual_report_path.resolve()),
            },
            "completion_checks": checks,
            "unmet_completion_conditions": unmet,
        },
        "summary": (
            "最终测验、题库、技术报告和虚拟被试报告已生成；"
            f"未满足完成条件 {len(unmet)} 项。"
        ),
    }
