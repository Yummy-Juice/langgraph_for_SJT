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
    GATE_ORDER,
    build_psychometric_round_result,
    metric_scalar,
)
from sjt_system.evaluation.respondents import MATCHED_CONDITION_SCHEMA_VERSION
from sjt_system.evaluation.demographics import demographics_to_columns
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
            label="完整 facet profile 单次SJT作答",
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
            current = statuses.get(str(item_id))
            statuses[str(item_id)] = (
                "facet_form_retained"
                if current == "facet_form_retained"
                else "formal_qualified_locked"
                if current == "qualified_locked"
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


def _option_mean_value(value: Any, count: Any) -> str:
    return "无人选择" if count == 0 else _display_value(value)


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
    comparison: Mapping[str, Any], current_key: str, legacy_key: str | None = None
) -> Mapping[str, Any]:
    current = comparison.get(current_key)
    if isinstance(current, Mapping):
        return current
    legacy = comparison.get(legacy_key) if legacy_key else None
    return legacy if isinstance(legacy, Mapping) else {}


def _gate_status(gate: Mapping[str, Any]) -> str:
    if gate.get("passed") is None:
        return "基线/未比较"
    return "通过" if gate.get("passed") is True else "未通过"


def _improvement_text(gate: Mapping[str, Any]) -> str:
    observed = metric_scalar(gate.get("observed"))
    incumbent = metric_scalar(gate.get("incumbent"))
    return "未记录" if observed is None or incumbent is None else f"{observed - incumbent:+.6g}"


def _markdown_cell(value: Any, fallback: str = "未记录") -> str:
    if value is None or value == "":
        return fallback
    return str(value).replace("|", "｜").replace("\r", " ").replace("\n", " ")


def _virtual_review_count(entry: Mapping[str, Any], key: str, *, roles: bool = False) -> str:
    values = entry.get(key)
    if not isinstance(values, list):
        return "未记录"
    if not roles:
        return str(sum(isinstance(value, Mapping) for value in values))
    role_ids = [
        str((value.get("role") or {}).get("role_id"))
        for value in values
        if isinstance(value, Mapping)
        and isinstance(value.get("role"), Mapping)
        and value.get("role", {}).get("role_id")
    ]
    return str(len(set(role_ids))) if len(role_ids) == len(values) else "未记录"


def _virtual_content_review_markdown(
    protocol: Any,
    history: Sequence[Mapping[str, Any]],
    item_content_evidence: Mapping[str, Any],
) -> list[str]:
    records = [row for row in history if isinstance(row, Mapping)]
    if not protocol and not records:
        return []
    lines = [
        "## 虚拟内容复审记录",
        "",
        f"- 协议：{_markdown_cell(protocol)}",
        "- 证据边界：本节是虚拟开发过程证据，不是真人SME评审或真人内容效度证据。",
        "- 解释边界：facet整卷达标保留（facet_form_retained）与单题主动门槛资格分开记录；该处置不代表单题qualified或正式资格，不计入frozen_qualified。",
        "",
        "| 来源测量轮次 | 题目 | 所审版本 | 虚拟专家数 | 访谈角色数 | decision / status | 候选版本 | 归档路径 |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    if not records:
        lines.append("| 无已记录复审 | - | - | - | - | - | - | - |")
    for entry in records:
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
        candidate_text = (
            f"{candidate_id} v{candidate_version}"
            if candidate_version is not None
            else str(candidate_id)
        )
        values = [
            entry.get("source_analysis_round") or "未记录",
            entry.get("reviewed_item_id") or "未记录",
            entry.get("reviewed_item_version") or "未记录",
            _virtual_review_count(entry, "expert_reviews"),
            _virtual_review_count(entry, "cognitive_interviews", roles=True),
            f"{decision or '未记录'} / {status}",
            candidate_text,
            entry.get("archive_ref") or "未记录",
        ]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    evidence_rows = [
        (item_id, value)
        for item_id, value in item_content_evidence.items()
        if isinstance(value, Mapping)
    ]
    if evidence_rows:
        lines.extend(
            [
                "",
                "### 题目内容证据状态",
                "",
                "| 题目 | 状态 | 最近审查版本 | 来源测量轮次 | 证据范围 | 归档路径 |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for item_id, value in evidence_rows:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(cell)
                    for cell in (
                        item_id,
                        value.get("status"),
                        value.get("last_reviewed_version"),
                        value.get("source_analysis_round"),
                        value.get("evidence_scope"),
                        value.get("archive_ref"),
                    )
                )
                + " |"
            )
    lines.append("")
    return lines


def _facet_iteration_state_markdown(
    iteration: Any, round_result: Mapping[str, Any]
) -> list[str]:
    if not isinstance(iteration, Mapping) or not iteration:
        return []
    lines = [
        "## Facet 开发轮次控制器",
        "",
        f"- 策略版本：{_markdown_cell(iteration.get('policy_version'))}",
        f"- 状态：{_markdown_cell(iteration.get('status'))}",
        f"- 开发轮：{_markdown_cell(round_result.get('development_round') or iteration.get('development_round'))}",
        f"- 已完成开发轮：{_markdown_cell(iteration.get('completed_rounds'), '0')}/3",
        f"- 本测量是否完成开发轮：{_markdown_cell(round_result.get('round_completed'))}",
        f"- 暂停原因：{_markdown_cell(iteration.get('paused_reason'), '无')}",
        "- 门槛规则：单题四项主动门槛分别为 CITC≥.30、IPIP目标 Hedges' g≥.50、目标 IPIP rho_s≥.40、Δmin≥.30；Profile目标rho/同域VTS/跨域VTS仅诊断，threshold=None、passes=None。整卷五项指标为α、ICC、目标g、目标rho和Δmin。",
        "- 首轮基线允许α或ICC低于.70；随后最多两轮成功返修须各 facet α/ICC≥.70，且 g、rho、Δmin 均相对上一完成轮严格上升（原始差值>1e-12）。每 facet 每开发轮局部提交并复测次数不设上限；每次逻辑失败调用最多5次尝试（含首次），底层重试计入且重启不重置。",
        f"- 首轮 cohort 作答引用：{_markdown_cell(iteration.get('cohort_source_ref'), '未记录')}",
        f"- 首轮 IPIP 校标引用：{_markdown_cell(iteration.get('first_round_reference_ref'), '未记录')}",
        "- 首轮固定被试ID、完整人格分数、人口学变量与模拟设置；IPIP校标或身份匹配缺失时暂停且不得重施。后续只测改过的题目新版本；未改且未过单题门槛的题复用最近完整作答，已过门槛的题复用形成资格来源的主测及各retest作答。按 subject_id/item_id/item_version/administration 逐题合并，不拼接整卷总分、不跨版本平均、不复用旧版答案，主测答案不作重测输入。",
        "",
        "### 当前 facet 局部状态",
        "",
        "| facet | 本轮局部提交并复测次数 | 已接受快照 | 当前失败 | 有基线 |",
        "|---|---:|:---:|:---:|:---:|",
    ]
    attempts = iteration.get("local_attempts") or {}
    accepted = iteration.get("accepted_facets") or {}
    baseline = iteration.get("baseline_facets") or {}
    failed = set(str(value) for value in iteration.get("failed_facet_ids") or [])
    facet_ids = sorted(
        set(str(value) for value in attempts)
        | set(str(value) for value in accepted)
        | set(str(value) for value in baseline)
        | failed
    )
    if facet_ids:
        for facet_id in facet_ids:
            lines.append(
                f"| {_markdown_cell(facet_id)} | {_markdown_cell(attempts.get(facet_id), '0')}/无限 | "
                f"{'是' if facet_id in accepted else '否'} | "
                f"{'是' if facet_id in failed else '否'} | "
                f"{'是' if facet_id in baseline else '否'} |"
            )
    else:
        lines.append("| 未记录 | - | - | - | - |")
    completed_history = [
        row for row in iteration.get("completed_history") or []
        if isinstance(row, Mapping)
    ]
    if completed_history:
        lines.extend(
            [
                "",
                "### 已完成轮次历史",
                "",
                "| 开发轮 | 基线记录 | facet数 | 局部次数 |",
                "|---:|:---:|---:|---|",
            ]
        )
        for row in completed_history:
            lines.append(
                f"| {_markdown_cell(row.get('development_round'))} | "
                f"{'是' if row.get('baseline_only') else '否'} | "
                f"{len(row.get('facet_forms') or {})} | "
                f"{_markdown_cell(row.get('local_attempts'), '-')} |"
            )
    snapshots = [
        ("首轮基线", facet_id, value)
        for facet_id, value in baseline.items()
        if isinstance(value, Mapping)
    ] + [
        ("本轮已接受", facet_id, value)
        for facet_id, value in accepted.items()
        if isinstance(value, Mapping)
    ]
    if snapshots:
        lines.extend(
            [
                "",
                "### 基线与接受快照（五项整卷指标）",
                "",
                "| 快照 | facet | 题数 | α | ICC | 目标g | 目标rho | Δmin | 题目版本 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for label, facet_id, snapshot in snapshots:
            metrics = snapshot.get("metrics") or {}
            lines.append(
                "| " + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        label,
                        facet_id,
                        metrics.get("item_count"),
                        _display_value(metrics.get("cronbach_alpha")),
                        _display_value(metrics.get("virtual_test_retest_icc")),
                        _display_value(metrics.get("target_hedges_g")),
                        _display_value(metrics.get("target_spearman_rho")),
                        _display_value(metrics.get("discriminant_delta_min")),
                        snapshot.get("item_versions"),
                    )
                ) + " |"
            )
    lines.append("")
    return lines


def _round_result_markdown(round_result: Mapping[str, Any]) -> list[str]:
    summary = round_result.get("summary") or {}
    lines = [
        "## 最近完成开发轮虚拟筛查总览",
        "",
        f"- 开发轮：{_markdown_cell(round_result.get('development_round'), '未记录')}",
        f"- 分析题目：{summary.get('item_count', 0)}",
        f"- 本轮新合格：{summary.get('newly_qualified_count', 0)}",
        f"- 本轮四项主动门槛均通过：{summary.get('current_round_qualified_count', 0)}",
        f"- 累计冻结合格：{summary.get('frozen_qualified_count', 0)}",
        f"- 本轮冻结题目级指标：{summary.get('frozen_metric_count', 0)}",
        f"- 本轮重算题目级指标：{summary.get('measured_metric_count', 0)}",
        f"- 尚待处理：{summary.get('pending_treatment_count', 0)}",
        f"- 已锁定正式题：{summary.get('qualified_locked_count', 0)}",
        f"- Facet整卷达标保留但未过全部单题门槛：{summary.get('facet_form_retained_count', 0)}",
        f"- 正式题监测警告：{summary.get('monitoring_warning_count', 0)}",
        f"- 不可估计门槛：{summary.get('unestimable_metric_count', 0)}",
        "",
        "| 门槛 | 通过题数 | 阈值 | 不可估计 |",
        "|---|---:|---:|---:|",
    ]
    for gate in round_result.get("gate_summary") or []:
        if isinstance(gate, Mapping) and gate.get("gate_id") in GATE_ORDER:
            lines.append(
                f"| {gate.get('label')} | {gate.get('pass_count', 0)}/"
                f"{gate.get('item_count', 0)} | ≥{_display_value(gate.get('threshold'))} | "
                f"{gate.get('unestimable_count', 0)} |"
            )
    lines.extend(
        [
            "",
            "| 题目 | 状态 | 单题四项qualified | monitoring_pass | 保留依据 | 指标状态 | CITC | 单题Hedges'g | 单题IPIP rho_s | 单题Δmin | Profile目标rho（诊断） | 同域VTS（诊断）/最大rho facet | 跨域VTS（诊断）/最大rho facet |",
            "|---|---|:---:|:---:|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for entry in round_result.get("items") or []:
        if not isinstance(entry, Mapping):
            continue
        gates = {
            row.get("gate_id"): row
            for row in entry.get("gates") or []
            if isinstance(row, Mapping) and row.get("gate_id") in GATE_ORDER
        }
        contaminants = entry.get("max_contaminants") or {}
        same = contaminants.get("same_domain") or {}
        cross = contaminants.get("cross_domain") or {}
        lines.append(
            f"| {entry.get('item_id')} | {entry.get('status')} | "
            f"{'是' if entry.get('qualified') is True and entry.get('status') != 'facet_form_retained' else '否'} | "
            f"{_markdown_cell(entry.get('monitoring_pass'), '-')} | "
            f"{entry.get('retention_basis') or '-'} | "
            f"{'冻结' if entry.get('iteration_metric_status') == 'frozen' else '本轮重算' if entry.get('iteration_metric_status') == 'measured' else '未标记'} | "
            f"{_display_value((gates.get('citc_pass') or {}).get('value'))} | "
            f"{_display_value((gates.get('target_hedges_g_pass') or {}).get('value'))} | "
            f"{_display_value((gates.get('target_ipip_spearman_rho_pass') or {}).get('value'))} | "
            f"{_display_value((gates.get('discriminant_delta_min_pass') or {}).get('value'))} | "
            f"{_display_value(_profile_diagnostic_value(entry, 'target_rho_diagnostic'))} | "
            f"{_display_value(_profile_diagnostic_value(entry, 'same_domain_vts_diagnostic'))} / "
            f"{same.get('facet_name') or same.get('dimension_id') or '-'}（rho={_display_value(same.get('signed_rho'))}） | "
            f"{_display_value(_profile_diagnostic_value(entry, 'cross_domain_vts_diagnostic'))} / "
            f"{cross.get('facet_name') or cross.get('dimension_id') or '-'}（rho={_display_value(cross.get('signed_rho'))}） |"
        )
    lines.extend(
        [
            "",
            "### 所有题目的按选项对齐设定分数均值",
            "",
            "完整profile按选择同一选项的同批被试的facet分数计算，同域和跨域分别取最大带符号rho的非目标facet；独立条件作答沿用各臂均值。filtering_authority=false。",
            "",
            "| 题目 | 选项 | 计分 | 目标N | 目标facet均值 | 同域最大rho facet | 同域N | 同域facet均值 | 跨域最大rho facet | 跨域N | 跨域facet均值 |",
            "|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in round_result.get("option_score_comparisons") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('item_id')} | {row.get('option_id')} | "
            f"{_display_value(row.get('option_score'))} | "
            f"{row.get('target_n', 0)} | {_option_mean_value(row.get('target_mean_score'), row.get('target_n'))} | "
            f"{row.get('same_domain_facet_name') or row.get('same_domain_dimension_id') or '-'} | "
            f"{row.get('same_domain_n', 0)} | {_option_mean_value(row.get('same_domain_mean_score'), row.get('same_domain_n'))} | "
            f"{row.get('cross_domain_facet_name') or row.get('cross_domain_dimension_id') or '-'} | "
            f"{row.get('cross_domain_n', 0)} | {_option_mean_value(row.get('cross_domain_mean_score'), row.get('cross_domain_n'))} |"
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
            if isinstance(gate, Mapping) and gate.get("gate_id") in GATE_ORDER:
                status = "通过" if gate.get("passes") is True else (
                    "未通过" if gate.get("estimable") is True else "不可估计"
                )
                lines.append(
                    f"| {gate.get('label')} | {_display_value(gate.get('value'))} | "
                    f"≥{_display_value(gate.get('threshold'))} | {status} | true |"
                )
        for diagnostic in entry.get("profile_diagnostics") or []:
            if isinstance(diagnostic, Mapping):
                lines.append(
                    f"| {diagnostic.get('label')} | {_display_value(diagnostic.get('value'))} | "
                    "无 | 仅诊断 | false |"
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
                    "| 选项 | 计分 | 目标N | 目标facet均值 | 同域最大rho facet | 同域N | 同域facet均值 | 跨域最大rho facet | 跨域N | 跨域facet均值 |",
                    "|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
                ]
            )
            for comparison in score_comparisons:
                lines.append(
                    f"| {comparison.get('option_id')} | "
                    f"{_display_value(comparison.get('option_score'))} | "
                    f"{comparison.get('target_n', 0)} | {_option_mean_value(comparison.get('target_mean_score'), comparison.get('target_n'))} | "
                    f"{comparison.get('same_domain_facet_name') or comparison.get('same_domain_dimension_id') or '-'} | "
                    f"{comparison.get('same_domain_n', 0)} | {_option_mean_value(comparison.get('same_domain_mean_score'), comparison.get('same_domain_n'))} | "
                    f"{comparison.get('cross_domain_facet_name') or comparison.get('cross_domain_dimension_id') or '-'} | "
                    f"{comparison.get('cross_domain_n', 0)} | {_option_mean_value(comparison.get('cross_domain_mean_score'), comparison.get('cross_domain_n'))} |"
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
        item_ipip = quality.get("single_item_ipip_metrics") or {}
        item_hedges = item_ipip.get("target_hedges_g") or {}
        item_rho = item_ipip.get("target_ipip_spearman_rho") or {}
        item_delta = item_ipip.get("discriminant_delta_min") or {}
        return (
            "分面内CITC="
            + _display_value(citc.get("r"))
            + "；目标ρs="
            + _display_value(target.get("rho"))
            + "；同域VTS="
            + _display_value(same_domain.get("specificity_margin"))
            + "；跨域VTS="
            + _display_value(cross_domain.get("specificity_margin"))
            + "；单题目标Hedges'g="
            + _display_value(item_hedges.get("standardized_effect"))
            + "；单题目标IPIP rho_s="
            + _display_value(item_rho.get("rho"))
            + "；单题Δmin="
            + _display_value(item_delta.get("delta_min"))
        )
    corrected = (
        baseline.get("facet_corrected_item_total_correlation")
        or baseline.get("corrected_item_total_correlation")
        or {}
    )
    return f"分面内CITC={_display_value(corrected.get('r'))}"


def strict_development_round_entries(
    history: Sequence[Mapping[str, Any]],
    *,
    max_rounds: int = 3,
) -> list[Mapping[str, Any]]:
    """Collapse internal measurement batches to one snapshot per dev round."""

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for raw in history:
        if not isinstance(raw, Mapping):
            continue
        controller = raw.get("facet_iteration_state") or {}
        value = raw.get("development_round") or (
            controller.get("development_round")
            if isinstance(controller, Mapping)
            else None
        )
        try:
            development_round = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= development_round <= max_rounds:
            grouped.setdefault(development_round, []).append(raw)

    def analysis_key(row: Mapping[str, Any]) -> int:
        try:
            return int(row.get("analysis_round") or 0)
        except (TypeError, ValueError):
            return 0

    def is_completed(row: Mapping[str, Any]) -> bool:
        if row.get("round_completed") is True:
            return True
        controller = row.get("facet_iteration_state") or {}
        batch = row.get("measurement_batch") or row.get("analysis_round")
        return any(
            isinstance(history_row, Mapping)
            and str(history_row.get("analysis_round")) == str(batch)
            for history_row in (
                controller.get("completed_history") or []
                if isinstance(controller, Mapping)
                else []
            )
        )

    selected: list[Mapping[str, Any]] = []
    for development_round in sorted(grouped):
        rows = sorted(grouped[development_round], key=analysis_key)
        completed_rows = [row for row in rows if is_completed(row)]
        selected.append(
            rows[0]
            if development_round == 1
            else (completed_rows[-1] if completed_rows else rows[-1])
        )
    return selected[:max_rounds]


def development_round_batch_label(
    history: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> str:
    """Return a public development-round label without exposing batch counts."""

    controller = entry.get("facet_iteration_state") or {}
    value = entry.get("development_round") or (
        controller.get("development_round") if isinstance(controller, Mapping) else None
    )
    try:
        development_round = int(value)
    except (TypeError, ValueError):
        development_round = None

    round_text = "?" if development_round is None else str(development_round)
    return f"第{round_text}轮"


def _iteration_history_markdown(
    history: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines = [
        "## 整卷迭代曲线数据",
        "",
        "严格展示最多三个开发轮：第1轮为基线，随后最多完成两轮返修。内部记录每个开发轮从第1批重新计数；批次号仅保留在运行账本，不在对外报告中展示。",
        "",
        "| 开发轮 | 本轮完成 | 临时题数 | 候选题数 | Cronbach α | 目标IPIP Hedges’ g | 目标IPIP rho | Δmin | α/ICC≥.70 | 控制器状态 | 累计Token | 累计模型耗时(ms) |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---|---:|---:|",
    ]
    cumulative_tokens = 0
    cumulative_duration = 0
    data_rows = 0
    for entry in strict_development_round_entries(history):
        if not isinstance(entry, Mapping):
            continue
        data_rows += 1
        metrics = entry.get("form_metrics") or {}
        validity = metrics.get("validity") or {}
        convergent = validity.get("convergent_validity") or {}
        discriminant = validity.get("discriminant_validity") or {}
        known_groups = validity.get("known_groups_validity") or {}
        quality = form_quality_summary(metrics)
        uses_ipip_objective = quality.get("objective_source") in {"ipip_human_style_v3", "ipip_facet_gates_v4"}
        usage = entry.get("token_usage") or {}
        cumulative_tokens += int(usage.get("total_tokens") or 0)
        cumulative_duration += int(usage.get("duration_ms") or 0)
        iteration_state = entry.get("facet_iteration_state") or {}
        development_round = entry.get("development_round") or iteration_state.get("development_round")
        measurement_batch = entry.get("measurement_batch") or entry.get("analysis_round")
        completed_history = iteration_state.get("completed_history") or []
        round_completed = entry.get("round_completed")
        if round_completed is None:
            round_completed = any(
                isinstance(row, Mapping)
                and str(row.get("analysis_round")) == str(measurement_batch)
                for row in completed_history
            )
        facet_metrics = quality.get("facet_metrics") or []
        is_baseline = any(
            isinstance(row, Mapping)
            and row.get("baseline_only") is True
            and str(row.get("analysis_round")) == str(measurement_batch)
            for row in completed_history
        ) or development_round == 1
        reliability_gates_pass = bool(facet_metrics) and all(
            (facet.get("alpha_gate") or {}).get("passed") is True
            and (facet.get("stability_gate") or {}).get("passed") is True
            for facet in facet_metrics
            if isinstance(facet, Mapping)
        )
        reliability_gate_label = (
            "基线，不作通过要求"
            if is_baseline
            else "各facet通过"
            if reliability_gates_pass
            else "未通过/未估计"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    development_round_batch_label(history, entry),
                    "是" if round_completed is True else "否" if round_completed is False else "未记录",
                    str(entry.get("item_count") or 0),
                    str(entry.get("candidate_count") or 0),
                    _display_value((quality.get("alpha_gate") or {}).get("observed")) if not quality.get("facet_metrics") else "见下表",
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
                    reliability_gate_label,
                    _markdown_cell(iteration_state.get("status") or entry.get("status")),
                    str(cumulative_tokens),
                    str(cumulative_duration),
                ]
            )
            + " |"
        )
    if data_rows == 0:
        lines.append("| 无 | - | - | - | - | - | - | - | - | - | - | - | - |")
    lines.extend(["", "### 每个 facet 的五项指标与严格升幅", "", "| 开发轮 | facet | 题数 | α | ICC | Hedges' g | 目标rho | Δmin | α≥.70 | ICC≥.70 | g升幅 | g门槛 | rho升幅 | rho门槛 | Δmin升幅 | Δmin门槛 |", "|---:|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|"])
    facet_rows = 0
    for entry in strict_development_round_entries(history):
        if not isinstance(entry, Mapping):
            continue
        quality = form_quality_summary(entry.get("form_metrics") or {})
        iteration_state = entry.get("facet_iteration_state") or {}
        plateau = entry.get("plateau_status") or {}
        trajectory = plateau.get("trajectory") or []
        raw_comparisons = iteration_state.get("comparisons") or entry.get("facet_gate_comparison") or []
        if not raw_comparisons and trajectory:
            raw_comparisons = trajectory[-1].get("facet_gate_comparison") or []
        comparisons = {
            str(row.get("sjt_facet_id")): row
            for row in raw_comparisons
            if isinstance(row, Mapping)
        }
        for facet in quality.get("facet_metrics") or []:
            facet_rows += 1
            comparison = comparisons.get(str(facet.get("sjt_facet_id"))) or {}
            g_gate = _comparison_gate(comparison, "hedges_g_improvement_gate")
            rho_gate = _comparison_gate(comparison, "target_rho_improvement_gate", "target_rho_noninferiority_gate")
            delta_gate = _comparison_gate(comparison, "discriminant_delta_improvement_gate", "discriminant_delta_non_decrease_gate")
            lines.append("| " + " | ".join([
                development_round_batch_label(history, entry),
                str(facet.get("sjt_facet_id")),
                str(facet.get("item_count") or 0), _display_value(facet.get("cronbach_alpha")),
                _display_value(facet.get("virtual_test_retest_icc")), _display_value(facet.get("target_hedges_g")),
                _display_value(facet.get("target_spearman_rho")), _display_value(facet.get("discriminant_delta_min")),
                "通过" if (facet.get("alpha_gate") or {}).get("passed") else "未通过",
                "通过" if (facet.get("stability_gate") or {}).get("passed") else "未通过",
                _improvement_text(g_gate), _gate_status(g_gate),
                _improvement_text(rho_gate), _gate_status(rho_gate),
                _improvement_text(delta_gate), _gate_status(delta_gate),
            ]) + " |")
    if not facet_rows:
        lines.append("| - | - | 旧版/暂无 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "> 首轮完整测量直接建立基线，即使α/ICC<.70；后续成功轮要求每个facet α/ICC≥.70，且目标g、目标rho、Δmin相对上一完成轮分别严格上升（每项差值>1e-12）。",
            "> 每 facet 每开发轮局部提交并复测次数不设上限；每次逻辑失败调用最多5次尝试（含首次），底层重试计入、恢复不重置；不使用平台期或连续未改善早停。",
            "> 这些是虚拟作答系统内部的开发期传导指标，不能替代真人样本的正式信效度验证。",
            "",
        ]
    )
    return lines


def _iteration_item_metrics_markdown(
    history: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Render each round's item gates beside that facet's form metrics."""

    lines = [
        "## 分轮次、分 facet、分题指标",
        "",
        "下表包含该轮全部候选题；‘入选整卷’标记实际临时组卷。单题仅四项主动门槛参与资格判定，Profile目标rho/同域VTS/跨域VTS是无阈值、无过滤权的诊断；整卷记录五项指标。已过门槛题目的资格来源快照冻结。",
        "",
    ]
    rendered_rounds = 0
    for entry in strict_development_round_entries(history):
        if not isinstance(entry, Mapping):
            continue
        item_metrics = (
            entry.get("candidate_item_metrics")
            or entry.get("item_metrics")
            or {}
        )
        if isinstance(item_metrics, Mapping):
            item_rows = [row for row in item_metrics.values() if isinstance(row, Mapping)]
        elif isinstance(item_metrics, Sequence) and not isinstance(item_metrics, (str, bytes)):
            item_rows = [row for row in item_metrics if isinstance(row, Mapping)]
        else:
            item_rows = []
        if not item_rows:
            continue
        rendered_rounds += 1
        development_round = entry.get("development_round") or (
            (entry.get("facet_iteration_state") or {}).get("development_round")
            if isinstance(entry.get("facet_iteration_state"), Mapping)
            else None
        )
        form_metrics = entry.get("form_metrics") or {}
        saved_facet_metrics = entry.get("facet_metrics") or {}
        if isinstance(saved_facet_metrics, Mapping) and saved_facet_metrics:
            facet_metrics = {
                str(facet_id): row
                for facet_id, row in saved_facet_metrics.items()
                if isinstance(row, Mapping)
            }
        else:
            facet_metrics = {
                str(row.get("sjt_facet_id")): row
                for row in form_metrics.get("facet_metrics") or []
                if isinstance(row, Mapping) and row.get("sjt_facet_id") is not None
            }
        lines.extend(
            [
                f"### {development_round_batch_label(history, entry)}",
                "",
                "| facet | 题数 | α | ICC | Hedges' g | 目标rho | Δmin |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for facet_id, facet in facet_metrics.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        facet_id,
                        str(facet.get("item_count") or 0),
                        _display_value(facet.get("cronbach_alpha")),
                        _display_value(facet.get("virtual_test_retest_icc")),
                        _display_value(facet.get("target_hedges_g")),
                        _display_value(facet.get("target_spearman_rho")),
                        _display_value(facet.get("discriminant_delta_min")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "| facet | 题目 | 入选整卷 | 题目级指标 | CITC | 目标rho_s | 同域VTS | 跨域VTS | 单题Hedges'g | 单题IPIP rho_s | 单题Δmin | 题目通过 | α | ICC | 整卷Hedges' g | 整卷目标rho | 整卷Δmin |",
                "|---|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(item_rows, key=lambda value: str(value.get("item_id") or "")):
            facet_id = str(row.get("facet_id") or "未标注 facet")
            facet = facet_metrics.get(facet_id) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        facet_id,
                        str(row.get("item_id") or "未记录"),
                        "是" if row.get("selected_for_form") is True else "否",
                        "冻结" if row.get("iteration_metric_status") == "frozen" else "本轮重算" if row.get("iteration_metric_status") == "measured" else "未标记",
                        _display_value(row.get("citc")),
                        _display_value(row.get("target_rho")),
                        _display_value(row.get("same_domain_vts")),
                        _display_value(row.get("cross_domain_vts")),
                        _display_value(row.get("target_hedges_g")),
                        _display_value(row.get("target_ipip_spearman_rho")),
                        _display_value(row.get("discriminant_delta_min")),
                        "通过" if row.get("qualified") is True else "未通过",
                        _display_value(facet.get("cronbach_alpha")),
                        _display_value(facet.get("virtual_test_retest_icc")),
                        _display_value(facet.get("target_hedges_g")),
                        _display_value(facet.get("target_spearman_rho")),
                        _display_value(facet.get("discriminant_delta_min")),
                    ]
                )
                + " |"
            )
        lines.append("")
    if not rendered_rounds:
        lines.extend(
            [
                "> 当前历史记录尚未保存逐题轮次快照；不会用当前轮指标回填历史轮次。",
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
            "- 等待既有 deferred 决策的题数："
            f"{technical_report.get('deferred_decision_count', 0)}"
        ),
        ("- 初始候选入库：固定槽位与结构校验，不虚构首轮内容审题通过"
         if technical_report.get("virtual_content_review_protocol") == "mte_cosmin_virtual_content_review_v1"
         else "- 心理骨架审核：按固定槽位逐题独立审核"),
        ("- 返修流程：首测与证据诊断、未达标题的材料核查/虚拟专家审题/认知访谈、至多三轮虚拟内容复审"
         if technical_report.get("virtual_content_review_protocol") == "mte_cosmin_virtual_content_review_v1"
         else "- 心理测量处理：结构化诊断、定向返修、重新模拟与重新分析"),
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
        "- 四项主动单题门槛：CITC≥.30、目标IPIP Hedges'g≥.50、目标IPIP rho_s≥.40、Δmin≥.30；Profile目标rho、同域VTS、跨域VTS仅诊断，threshold=None且passes=None，不参与过滤",
        "- 首轮固定 cohort、完整人格分数、人口学与模拟设置并一次完成基线施测；后续只测修改题目新版本，未改题按资格来源或最新完整作答逐题合并，IPIP不重施",
        "- 条件组分数分布仅作设计核查，filtering_authority=false，不参与题目筛选",
        "- 难度、选项使用率与Cronbach alpha仅为描述性诊断",
        (
            "- 结论："
            + _display_value(test_quality.get("interpretation"))
        ),
        "",
        *_round_result_markdown(round_result),
        *_facet_iteration_state_markdown(
            technical_report.get("facet_iteration_state") or {}, round_result
        ),
        *_iteration_history_markdown(iteration_history),
        *_iteration_item_metrics_markdown(iteration_history),
        "## 补充：完整 facet 分数分布",
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
                        "淘汰并同槽补题（待重新施测）"
                        if event.get("resolution") == "design_budget_replenishment"
                        else "通过" if event.get("event") == "psychometric_item_repaired"
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
        *_virtual_content_review_markdown(
            technical_report.get("virtual_content_review_protocol"),
            technical_report.get("virtual_content_review_history") or [],
            technical_report.get("item_content_evidence") or {},
        ),
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
    current_item_versions = {
        str(item.get("item_id")): item.get("version")
        for item in state.get("item_pool") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    current_item_ids = set(current_item_versions)
    locked_versions = {
        str(item_id): version
        for item_id, version in (state.get("locked_retained_item_versions") or {}).items()
    }
    locked_ids = {
        str(item_id)
        for item_id, disposition in final_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "qualified_locked"
        and current_item_versions.get(str(item_id)) is not None
        and str(item_id) in locked_versions
        and locked_versions.get(str(item_id)) == current_item_versions.get(str(item_id))
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
    facet_form_retained_ids = {
        str(item_id)
        for item_id, disposition in final_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "facet_form_retained"
        and disposition.get("item_version") == current_item_versions.get(str(item_id))
    }
    item_pools = {
        "formal_items": sorted(locked_ids),
        "facet_form_retained_items": sorted(facet_form_retained_ids),
        "pending_items": sorted(
            current_item_ids - locked_ids - facet_form_retained_ids - pending_sme_ids - eliminated_ids
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
        "facet_iteration_state": deepcopy(state.get("facet_iteration_state") or {}),
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
        "virtual_content_review_protocol": state.get(
            "virtual_content_review_protocol"
        ),
        "virtual_content_review_history": deepcopy(
            state.get("virtual_content_review_history") or []
        ),
        "item_content_evidence": deepcopy(
            state.get("item_content_evidence") or {}
        ),
        "item_decisions": item_decisions,
        "reassembly_round": state.get("reassembly_round", 0),
        "test_revision_history": deepcopy(
            state.get("test_revision_history") or []
        ),
        "evidence_notice": EVIDENCE_NOTICE,
        "generated_at": generated_at,
    }
    virtual_sample_config = state.get("virtual_sample_config") or {}
    respondent_refs = state.get("virtual_respondents") or []
    virtual_respondents = [
        respondent
        for respondent in respondent_refs
        if isinstance(respondent, Mapping)
    ]
    demographics_snapshot = virtual_sample_config.get("demographics_snapshot")
    demographics_profiles_available = (
        isinstance(demographics_snapshot, Mapping)
        and len(virtual_respondents) == len(respondent_refs)
        and bool(virtual_respondents)
        and all(isinstance(row.get("demographics"), Mapping) for row in virtual_respondents)
    )
    demographics_ready = (
        demographics_profiles_available
        and bool(virtual_sample_config.get("demographics_version"))
        and virtual_sample_config.get("response_temperature") is not None
    )
    demographics_profiles = (
        [
            {
                "respondent_id": respondent.get("respondent_id"),
                "condition_id": respondent.get("condition_id"),
                "matched_subject_id": respondent.get("matched_subject_id"),
                **demographics_to_columns(
                    respondent["demographics"], demographics_snapshot
                ),
            }
            for respondent in virtual_respondents
        ]
        if demographics_profiles_available
        else []
    )
    virtual_report = {
        "schema_version": 5,
        "run_id": state["run_id"],
        "sample_config": deepcopy(virtual_sample_config),
        "respondent_count": len(respondent_refs),
        "demographics_version": virtual_sample_config.get("demographics_version"),
        "response_temperature": virtual_sample_config.get("response_temperature"),
        "demographics_profiles": demographics_profiles,
        "demographics_export_status": (
            "complete" if demographics_ready else "reconfiguration_required"
        ),
        "demographics_income_interpretation": (
            "月收入是按冻结的版本化合成数据库，根据职业收入范围、年龄与学历系数"
            "及100元步长生成的税前人民币等值；不是观察到的实际收入，也不代表任何"
            "国家或地区的真实收入分布。"
        ),
        "response_summary": deepcopy(
            state.get("virtual_response_summary")
        ),
        "virtual_screening_metrics": deepcopy(
            (state.get("test_statistics") or {}).get(
                "virtual_screening_metrics"
            )
        ),
        "psychometric_round_result": round_result,
        "facet_iteration_state": deepcopy(state.get("facet_iteration_state") or {}),
        "psychometric_iteration_history": deepcopy(
            state.get("psychometric_iteration_history") or []
        ),
        "virtual_content_review_protocol": state.get(
            "virtual_content_review_protocol"
        ),
        "virtual_content_review_history": deepcopy(
            state.get("virtual_content_review_history") or []
        ),
        "item_content_evidence": deepcopy(
            state.get("item_content_evidence") or {}
        ),
        "artifacts": deepcopy(score_protocol_artifacts),
        "response_manifest": state.get("virtual_response_data_ref"),
        "frozen_reference_questionnaire": state.get(
            "frozen_reference_questionnaire_ref"
        ),
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
            "主迭代未调用人格总结；每名虚拟被试同时携带完整facet分数画像，三个顶层臂共享匹配被试ID，target组首轮按出题阶段选定的facet调用IPIP-NEO（每个facet 10题），每名被试一次调用完成全部所选题目；后续轮次沿用首轮冻结结果，不重复调用。",
            "本报告不包含真实被试信度、真人效度或人口学DIF证据；虚拟重测ICC只描述模型作答稳定性。",
            "虚拟被试人口学属性是按冻结的合成数据库和随机种子生成的背景变量，不是现实人口抽样或真实样本描述。",
            "response_temperature 表示模型作答采样随机性，未校准为真人个体噪声。",
            "虚拟内容复审记录属于开发期虚拟证据，不是真人SME评审；qualified/qualified_locked资格与facet_form_retained整卷达标但单题门槛未全过的处置分别报告。单题资格按四项主动门槛记录，Profile三项仅诊断。",
        ] + ([] if demographics_ready else [
                "旧协议样本缺少完整的冻结人口学配置或被试资料；人口学明细未导出，"
                "需要重新配置虚拟样本。"
            ]),
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
