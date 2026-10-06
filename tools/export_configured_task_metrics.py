"""Export committed task measurements without recalculating or changing them."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sjt_system.delivery.reporting import (
    development_round_batch_label,
    strict_development_round_entries,
)

ITEM_FIELDS = (
    "citc",
    "target_rho",
    "same_domain_vts",
    "cross_domain_vts",
    "target_hedges_g",
    "target_ipip_spearman_rho",
    "discriminant_delta_min",
)
GATE_FIELDS = tuple(f"{field}_pass" for field in ITEM_FIELDS)
FORM_FIELDS = (
    "cronbach_alpha", "virtual_test_retest_icc", "target_hedges_g",
    "target_spearman_rho", "discriminant_delta_min",
)
FACET_LABELS = {
    "extraversion_warmth": "温暖性",
    "extraversion_gregariousness": "乐群性",
    "agreeableness_altruism": "利他性",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.3f}" if math.isfinite(value) else "不可估计"
    return "不可估计" if value is None else str(value).replace("|", "\\|")


def frozen_qualified(
    row: Mapping,
    dispositions: Mapping | None = None,
) -> bool:
    """Return historical gate qualification, separate from current monitoring."""

    if row.get("frozen_qualified") is True:
        return True
    # Older snapshots did not carry the explicit boolean.  Their disposition
    # basis is still sufficient to recover the non-revocable qualification.
    if row.get("retention_basis") == "four_iteration_gates_passed":
        return True
    if isinstance(dispositions, Mapping):
        disposition = dispositions.get(str(row.get("item_id"))) or dispositions.get(
            row.get("item_id")
        )
        return (
            isinstance(disposition, Mapping)
            and disposition.get("status") == "qualified_locked"
            and disposition.get("retention_basis") == "four_iteration_gates_passed"
        )
    return False


def resource_usage(run_id: str) -> dict:
    paths = sorted((PROJECT_ROOT / "outputs" / "run_telemetry").glob(f"calls_configured_{run_id}_*.jsonl"))
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("run_id") == run_id:
                    records.append(record)
    return {
        "ledger_paths": [str(path) for path in paths],
        "recorded_model_calls": len(records),
        "status_counts": dict(Counter(row.get("status", "unknown") for row in records)),
        "model_ids": sorted({str(row["model_id"]) for row in records if row.get("model_id")}),
        "recorded_prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in records),
        "recorded_completion_tokens": sum(row.get("completion_tokens") or 0 for row in records),
        "recorded_total_tokens": sum(row.get("total_tokens") or 0 for row in records),
        "calls_without_token_usage": sum(row.get("total_tokens") is None for row in records),
        "token_usage_note": "Only recorded usage is summed; missing usage is not an estimate of zero cost.",
    }


def validate_round(snapshot: dict, manifest: dict) -> None:
    if snapshot["run_id"] != manifest["run_id"]:
        raise ValueError("Round artifact belongs to another task")
    items = snapshot["item_metrics"]
    selected = snapshot["selected_item_ids"]
    if len(items) != snapshot["candidate_count"]:
        raise ValueError("Candidate metric coverage is incomplete")
    if len(set(selected)) != snapshot["item_count"] or not set(selected).issubset(items):
        raise ValueError("Selected-item metric coverage is incomplete")
    form_metrics = snapshot.get("form_metrics")
    if isinstance(form_metrics, Mapping) and "selected_item_ids" in form_metrics:
        form_selected = [str(item_id) for item_id in form_metrics.get("selected_item_ids") or []]
        if (
            len(form_selected) != len(set(form_selected))
            or set(form_selected) != {str(item_id) for item_id in selected}
        ):
            raise ValueError("Whole-form metrics do not cover the committed selected items")
    facets = snapshot["facet_metrics"]
    expected = manifest["test_specification"]["facet_item_counts"]
    if set(facets) != set(expected):
        raise ValueError("Facet metric coverage differs from the frozen specification")
    if snapshot["form_status"] == "complete":
        if len(selected) != manifest["test_specification"]["final_item_count"]:
            raise ValueError("Complete form does not match the frozen item count")
        for facet_id, count in expected.items():
            if facets[facet_id]["item_count"] != count:
                raise ValueError(f"Complete form quota differs for {facet_id}")
    for item_id, row in items.items():
        if row.get("selected_for_form") != (item_id in selected):
            raise ValueError(f"Selection flag differs from the measured form: {item_id}")
        if any(field not in row for field in (*ITEM_FIELDS, *GATE_FIELDS, "qualified")):
            raise ValueError(f"Missing measured gate fields: {item_id}")


def final_form_summary(
    state: dict,
    latest_snapshot: dict | None,
    *,
    frozen_item_ids: set[str] | None = None,
) -> dict | None:
    selected_items = state.get("selected_items") or []
    if not selected_items:
        return None
    selected_ids = [str(item["item_id"]) for item in selected_items]
    optimizer = (state.get("selection_results") or {}).get("form_optimizer") or {}
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != set(optimizer.get("selected_item_ids") or []):
        raise ValueError("Final selected items and optimizer result disagree")
    metrics = (optimizer.get("metrics") or {}).get("whole_test") or {}
    result = {
        "selected_item_ids": selected_ids,
        "selected_count": len(selected_ids),
        "source_analysis_round": state.get("psychometric_analysis_round"),
        "uses_existing_committed_responses": True,
        "optimizer_mode": optimizer.get("mode"),
        "deterministic_search": optimizer.get("deterministic_search"),
        "metric_status": metrics.get("status", "unavailable"),
        "facet_metrics": {str(row["sjt_facet_id"]): row
                          for row in metrics.get("facet_metrics") or []},
    }
    # A final form may reuse the latest committed measurement rather than run a
    # second measurement.  Its form statistics remain the exact evidence used
    # for the selected item versions.
    if not result["facet_metrics"] and latest_snapshot is not None:
        snapshot_facets = latest_snapshot.get("facet_metrics") or {}
        if isinstance(snapshot_facets, Mapping):
            result["facet_metrics"] = {
                str(facet_id): deepcopy(dict(row))
                for facet_id, row in snapshot_facets.items()
                if isinstance(row, Mapping)
            }
        elif isinstance(snapshot_facets, list):
            result["facet_metrics"] = {
                str(row["sjt_facet_id"]): deepcopy(dict(row))
                for row in snapshot_facets
                if isinstance(row, Mapping) and row.get("sjt_facet_id")
            }
        if result["facet_metrics"]:
            result["metric_status"] = "latest_committed_measurement"
            result["metrics_source_analysis_round"] = latest_snapshot.get("analysis_round")
    if latest_snapshot is not None:
        rows = latest_snapshot["item_metrics"]
        for item in selected_items:
            item_id = str(item["item_id"])
            if item_id not in rows or rows[item_id]["item_version"] != int(item.get("version") or 0):
                raise ValueError("Final selected item lacks matching committed measurement")
        result["gate_pass_counts"] = {key: sum(rows[item_id][key] is True for item_id in selected_ids)
                                      for key in GATE_FIELDS}
        result["numerically_qualified_count"] = sum(rows[item_id]["qualified"] is True for item_id in selected_ids)
        dispositions = (
            latest_snapshot.get("item_dispositions")
            if isinstance(latest_snapshot, Mapping)
            else None
        ) or state.get("item_final_dispositions") or {}
        if frozen_item_ids is None:
            frozen_item_ids = {
                item_id
                for item_id, row in rows.items()
                if frozen_qualified(row, dispositions)
            }
        result["frozen_qualified_count"] = sum(
            item_id in frozen_item_ids for item_id in selected_ids
        )
        result["current_round_qualified_count"] = sum(
            rows[item_id]["qualified"] is True
            and rows[item_id].get("frozen_qualified") is not True
            for item_id in selected_ids
        )
    return result


def export(run_id: str) -> dict:
    task_dir = PROJECT_ROOT / "outputs" / "configured_tasks" / run_id
    manifest = read_json(task_dir / "task_manifest.json")
    state = read_json(Path(manifest["checkpoint"]))["state"]
    workflow_completed = state.get("status") == "completed"
    deliverables_in_checkpoint = {
        "final_test": bool(state.get("final_test")),
        "item_database_ref": bool(state.get("item_database_ref")),
        "technical_report": bool(state.get("technical_report")),
        "virtual_respondent_report": bool(state.get("virtual_respondent_report")),
    }
    completion_checks = state.get("completion_checks") or {}
    checkpoint_rounds = {int(row["analysis_round"]): row for row in state.get("psychometric_iteration_history") or []}
    metric_dir = PROJECT_ROOT / "outputs" / "virtual_responses" / run_id / "iteration_metrics"
    paths = sorted(metric_dir.glob("iteration_metrics_round_*.json"))
    snapshots = [read_json(path) for path in paths]
    numbers = [int(row["analysis_round"]) for row in snapshots]
    if len(numbers) != len(set(numbers)) or set(numbers) != set(checkpoint_rounds):
        raise ValueError("Committed checkpoint rounds and round artifacts disagree")
    for snapshot in snapshots:
        validate_round(snapshot, manifest)
    path_by_analysis_round = {
        int(snapshot["analysis_round"]): path
        for snapshot, path in zip(snapshots, paths)
    }
    display_snapshots = strict_development_round_entries(snapshots, max_rounds=3)
    frozen_item_ids_seen: set[str] = set()
    frozen_item_ids_by_analysis: dict[int, set[str]] = {}
    for snapshot in snapshots:
        rows = snapshot["item_metrics"]
        dispositions = snapshot.get("item_dispositions") or {}
        frozen_item_ids_seen.update(
            str(item_id)
            for item_id, row in rows.items()
            if frozen_qualified(row, dispositions)
        )
        frozen_item_ids_by_analysis[int(snapshot["analysis_round"])] = set(
            frozen_item_ids_seen
        )
    all_frozen_item_ids = set(frozen_item_ids_seen)
    summaries = []
    lines = [
        "# 新任务逐轮迭代指标与整卷指标", "",
        f"- 运行编号：`{run_id}`",
        f"- 来源配置：`{manifest['source_run_id']}`；新蓝图、新题库、新作答。",
        "- 规格：温暖性、乐群性、利他性各10题，每轮组卷共30题；每轮50名完整facet档案虚拟被试，M=50、SD=15。",
        f"- 当前状态：`{state.get('status')}`；阶段：`{state.get('current_phase')}`；公开开发轮节点：{len(display_snapshots)}（最多3轮）。",
        f"- 当前协议：`{state.get('virtual_content_review_protocol')}`；停止原因：`{state.get('virtual_content_review_stop_reason') or 'none'}`。",
        f"- 原生工作流完成：{fmt(workflow_completed)}；检查点内最终入卷：{len(state.get('selected_items') or [])}题；交付字段存在性：`{json.dumps(deliverables_in_checkpoint, ensure_ascii=False)}`。",
        "- 本轮原始数值直接读取已提交轮次快照，不重新计分；历史锁定合格与本轮监测结果分开报告。",
        "- 本报告属于探索性虚拟开发证据，不是真人信度或效度验证。未做哈希验证或视觉QA。", "",
        "## 各轮指标概览", "",
        "七个迭代门槛依次为CITC >= .30、目标rho >= .40、同域VTS >= .30、跨域VTS >= .40、单题目标IPIP Hedges'g >= .50、单题目标IPIP rho >= .40、单题Delta_min >= .30。",
        "下表通过数的分母为当轮全部候选；整卷指标只对应当轮选中的30题，不对应60题候选库。", "",
        "现行复审仅处理本轮未达标且未锁定的题项；已锁定题的七项题目级指标沿用首次过关快照，不再重算，但仍可进入临时组卷并参与本轮整卷指标。‘本轮七项门槛均通过’仅适用于本轮重算题；‘累计冻结合格’不会因后续监测失败而下降。", "",
        "| 开发轮 | 候选数 | 入卷数 | CITC通过 | 目标rho通过 | 同域VTS通过 | 跨域VTS通过 | 单题Hedges'g通过 | 单题IPIP rho通过 | 单题Δmin通过 | 本轮七项门槛均通过 | 累计冻结合格 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    frozen_item_ids_by_round: dict[int, set[str]] = {}
    for snapshot in display_snapshots:
        number = int(snapshot["analysis_round"])
        path = path_by_analysis_round[number]
        rows = snapshot["item_metrics"]
        frozen_item_ids_seen = set(frozen_item_ids_by_analysis.get(number, set()))
        frozen_item_ids_by_round[number] = set(frozen_item_ids_seen)
        gate_counts = {key: sum(row[key] is True for row in rows.values()) for key in GATE_FIELDS}
        reviews = [row for row in state.get("virtual_content_review_history") or [] if row.get("source_analysis_round") == number]
        repairs = [row for row in state.get("psychometric_repair_history") or []
                   if row.get("source_analysis_round") == number and row.get("event") == "psychometric_item_repaired"]
        development_round = snapshot.get("development_round") or (
            (snapshot.get("facet_iteration_state") or {}).get("development_round")
        )
        repair_counts = dict(Counter(row.get("resolution", "unknown") for row in repairs))
        summary = {
            "analysis_round": number,
            "development_round": development_round,
            "measurement_batch": snapshot.get("measurement_batch") or number,
            "public_label": development_round_batch_label(snapshots, snapshot),
            "round_label": snapshot.get("round_label"),
            "candidate_count": snapshot["candidate_count"],
            "selected_count": snapshot["item_count"],
            "form_status": snapshot["form_status"],
            "gate_pass_counts": gate_counts,
            "numerically_qualified_count": sum(row["qualified"] is True for row in rows.values()),
            "current_round_qualified_count": sum(
                row["qualified"] is True
                and row.get("frozen_qualified") is not True
                for row in rows.values()
            ),
            "frozen_qualified_count": len(frozen_item_ids_seen),
            "frozen_metric_count": sum(
                row.get("iteration_metric_status") == "frozen"
                for row in rows.values()
            ),
            "measured_metric_count": sum(
                row.get("iteration_metric_status") == "measured"
                for row in rows.values()
            ),
            "selected_numerically_qualified_count": sum(rows[item_id]["qualified"] is True for item_id in snapshot["selected_item_ids"]),
            "selected_frozen_qualified_count": sum(
                str(item_id) in frozen_item_ids_seen
                for item_id in snapshot["selected_item_ids"]
            ),
            "facet_metrics": snapshot["facet_metrics"],
            "post_measurement_review_counts": dict(Counter(row.get("status", "unknown") for row in reviews)),
            "post_measurement_committed_repair_counts": repair_counts,
            "round_artifact": str(path),
            "response_data_ref": snapshot.get("response_data_ref"),
        }
        summaries.append(summary)
        values = [gate_counts[key] for key in GATE_FIELDS]
        lines.append(f"| {summary['public_label']} | {len(rows)} | {snapshot['item_count']} | "
                     + " | ".join(str(value) for value in (*values, summary["current_round_qualified_count"], summary["frozen_qualified_count"])) + " |")
    lines += ["", "## 各轮组卷的五个整卷指标", "",
              "按照现行统计实现，五个指标分别报告三个10题facet子卷。多facet总分的诊断统计保留在原始form_metrics中，不替代各facet指标。", "",
              "| 轮次 | Facet | 题数 | alpha | 虚拟重测ICC | 目标IPIP Hedges g | 目标IPIP rho | Delta_min |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for summary in summaries:
        for facet_id, row in summary["facet_metrics"].items():
            lines.append(f"| {summary['public_label']} | {FACET_LABELS.get(facet_id, facet_id)} | {row['item_count']} | "
                         + " | ".join(fmt(row.get(key)) for key in FORM_FIELDS) + " |")
    final_form = final_form_summary(
        state,
        snapshots[-1] if snapshots else None,
        frozen_item_ids=all_frozen_item_ids,
    )
    if final_form is not None:
        lines += ["", "## 原生组卷结果", "",
                  f"最终组卷{final_form['selected_count']}题，模式`{final_form['optimizer_mode']}`；使用{development_round_batch_label(snapshots, snapshots[-1])}已提交作答，不是新增一轮测量。",
                  f"最终入卷题的记录门槛通过数（冻结题沿用首次快照）：{json.dumps(final_form.get('gate_pass_counts'), ensure_ascii=False)}；本轮重算题七项门槛均通过{final_form.get('current_round_qualified_count', final_form.get('numerically_qualified_count'))}题，累计冻结合格{final_form.get('frozen_qualified_count', 0)}题。",
                  f"选题搜索记录：`{json.dumps(final_form.get('deterministic_search'), ensure_ascii=False)}`。",
                  "| Facet | 题数 | alpha | 虚拟重测ICC | 目标IPIP Hedges g | 目标IPIP rho | Delta_min |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for facet_id, row in final_form["facet_metrics"].items():
            lines.append(f"| {FACET_LABELS.get(facet_id, facet_id)} | {row['item_count']} | "
                         + " | ".join(fmt(row.get(key)) for key in FORM_FIELDS) + " |")
        if not final_form["facet_metrics"]:
            lines.append(f"最终组卷数值指标不可用：`{final_form['metric_status']}`。")
    for snapshot, summary in zip(display_snapshots, summaries):
        number = summary["analysis_round"]
        frozen_ids_for_round = frozen_item_ids_by_round.get(number, set())
        lines += ["", f"## {summary['public_label']}：测量结果", "",
                  "测量快照已保存在本次运行账本。",
                  f"本轮之后审题状态：`{json.dumps(summary['post_measurement_review_counts'], ensure_ascii=False)}`；已提交返修：`{json.dumps(summary['post_measurement_committed_repair_counts'], ensure_ascii=False)}`。",
                  "审题与返修记录不等于一次新测量；修改后只有下一轮实际提交的指标能检验新版本。冻结题仍保留在临时整卷的入选题目列表中。", "",
                  "| Facet | 题目ID | 版本 | 入卷 | 题目级指标 | CITC | 目标rho | 同域VTS | 跨域VTS | 单题Hedges'g | 单题IPIP rho | 单题Δmin | CITC通过 | 目标通过 | 同域通过 | 跨域通过 | Hedges'g通过 | IPIP rho通过 | Δmin通过 | 本轮七项门槛均通过 | 累计冻结合格 |",
                  "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for item_id, row in sorted(snapshot["item_metrics"].items(), key=lambda pair: (pair[1].get("facet_id") or "", pair[0])):
            metric_status = {"frozen": "冻结", "measured": "本轮重算"}.get(
                str(row.get("iteration_metric_status") or ""), "未标记"
            )
            values = [row.get(key) for key in ("item_version", "selected_for_form", *ITEM_FIELDS, *GATE_FIELDS, "qualified")]
            lines.append(f"| {FACET_LABELS.get(row.get('facet_id'), row.get('facet_id'))} | `{item_id}` | "
                         + " | ".join(
                             fmt(value)
                             for value in (
                                 values[0], values[1], metric_status, *values[2:],
                                 str(item_id) in frozen_ids_for_round,
                             )
                         ) + " |")
        reviews = [row for row in state.get("virtual_content_review_history") or []
                   if row.get("source_analysis_round") == number]
        if reviews:
            lines += ["", "本轮之后的审题/返修记录（不是下一轮测量结果）：", "",
                      "| 原题目ID | 审查版本 | 诊断 | 状态 | 已提交修改 | 提交后题目ID | 新版本 | 专家份数 | 访谈角色数 |",
                      "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |"]
            for row in sorted(reviews, key=lambda value: value.get("reviewed_item_id") or ""):
                diagnosis = row.get("diagnosis") or {}
                lines.append(f"| `{row.get('reviewed_item_id')}` | {row.get('reviewed_item_version')} | "
                             f"{diagnosis.get('decision') or '-'} | {row.get('status') or '-'} | "
                             f"{fmt(row.get('committed') is True)} | `{row.get('candidate_item_id') or '-'}` | "
                             f"{row.get('new_item_version') or '-'} | {len(row.get('expert_reviews') or [])} | "
                             f"{len(row.get('cognitive_interviews') or [])} |")
    usage = resource_usage(run_id)
    lines += ["", "## 资源与状态", "",
              f"- 模型接口调用账本：{usage['recorded_model_calls']}次；接口状态计数：`{json.dumps(usage['status_counts'], ensure_ascii=False)}`。接口成功不等于本地校验通过或返修已发布。",
              f"- 已记录token：输入{usage['recorded_prompt_tokens']}，输出{usage['recorded_completion_tokens']}，合计{usage['recorded_total_tokens']}；缺失用量记录{usage['calls_without_token_usage']}条（未估算费用）。",
              f"- 完成检查：`{json.dumps(completion_checks, ensure_ascii=False)}`。" if completion_checks else "- 完成门禁尚未评估，不能将空记录视为通过。",
              f"- 未满足条件：`{json.dumps(state.get('unmet_completion_conditions') or [], ensure_ascii=False)}`。" if completion_checks else "- 未满足条件列表尚未形成；空列表不代表所有完成条件已满足。",
              f"- 检查点：`{manifest['checkpoint']}`。",
              f"- 各次启动与恢复的stdout/stderr保存在：`{task_dir}`，文件为`launch_XX.stdout.log`和`launch_XX.stderr.log`。", ""]
    result = {
        "run_id": run_id, "source_run_id": manifest["source_run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": state.get("status"), "stop_reason": state.get("virtual_content_review_stop_reason"),
        "current_phase": state.get("current_phase"),
        "native_workflow_completed": workflow_completed,
        "deliverables_in_checkpoint": deliverables_in_checkpoint,
        "native_selected_count": len(state.get("selected_items") or []),
        "final_form": final_form,
        "rounds": summaries, "resource_usage": usage,
        "completion_checks": state.get("completion_checks"),
        "unmet_completion_conditions": state.get("unmet_completion_conditions"),
        "evidence_scope": "exploratory_virtual_development_evidence",
    }
    (task_dir / "all_round_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    (task_dir / "metrics_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    UUID(args.run_id)
    result = export(args.run_id)
    print(json.dumps({"run_id": args.run_id, "status": result["status"],
                      "committed_rounds": [row["development_round"] for row in result["rounds"]],
                      "round_labels": [row["public_label"] for row in result["rounds"]]}, ensure_ascii=False))
