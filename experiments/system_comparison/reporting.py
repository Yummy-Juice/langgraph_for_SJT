"""Offline descriptive report. One batch never produces inferential statistics."""
from html import escape
import json
import math
from statistics import median

from sjt_system.evaluation.form_metrics import form_quality_summary

from .storage import write_csv


def cost_summary(store, key):
    records = []
    for file in sorted(store.path(f"{key}/telemetry").glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record if isinstance(record, dict) else {"ledger_error": "invalid record"})
                except ValueError:
                    # An interrupted telemetry append must not destroy recovery
                    # or turn unobserved usage into a zero-cost claim.
                    records.append({"ledger_error": "incomplete telemetry record"})
    existing = store.read(f"{key}/cost.json", {})
    alias = existing.get("reused_from")
    if alias:
        return {**existing, "total_tokens": 0, "calls": 0, "fee": 0,
                "data_available": True, "cost_scope": "additional_only"}
    def total(name):
        values = [r.get(name) for r in records]
        return sum(values) if values and all(isinstance(v, (int, float)) for v in values) else None
    prompt, completion, cached = (total(k) for k in ("prompt_tokens", "completion_tokens", "cached_input_tokens"))
    prices = (store.config.input_price_per_million, store.config.cached_input_price_per_million,
              store.config.output_price_per_million)
    fee = None
    model_matches = all(r.get("model_id") in (None, store.config.model_id) for r in records)
    if key.endswith("evaluation") and store.config.evaluation_model_id != store.config.model_id:
        model_matches = False
    if model_matches and all(v is not None for v in (*prices, prompt, completion, cached)) and 0 <= cached <= prompt:
        fee = ((prompt - cached) * prices[0] + cached * prices[1] + completion * prices[2]) / 1e6
    summary = {**existing, "calls": len(records), "data_available": bool(records),
               "prompt_tokens": prompt, "completion_tokens": completion, "cached_input_tokens": cached,
               "total_tokens": total("total_tokens"), "model_duration_ms": total("duration_ms"), "fee": fee,
               "ledger_errors": sum(bool(r.get("ledger_error")) for r in records),
               "fee_note": "缺少用量、缓存拆分或适用单价时费用未知"}
    return summary


def _number(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else "不可估计"


def _table(rows, columns):
    head = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, bool):
                text = "是" if value else "否"
            elif value is None:
                text = "不可估计"
            elif isinstance(value, float):
                text = _number(value)
            else:
                text = str(value)
            cells.append("<td>" + escape(text) + "</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _chart(rows, column, label, bounded=False):
    values = [r.get(column) for r in rows]
    valid = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not valid:
        return f"<p>{escape(label)}：尚无可绘制数据。</p>"
    low, high = (0, 1) if bounded else (min(0, min(valid)), max(valid))
    if high <= low:
        high = low + 1
    content = []
    previous = None
    for i, (row, value) in enumerate(zip(rows, values)):
        x = 65 + i * 640 / max(1, len(rows) - 1)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            previous = None
            continue
        y = 225 - (value - low) / (high - low) * 175
        if previous:
            content.append(f'<line x1="{previous[0]}" y1="{previous[1]}" x2="{x}" y2="{y}" stroke="#266451" stroke-width="2"/>')
        content.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#266451"><title>{escape(row["round"])}: {value:.3f}</title></circle>')
        content.append(f'<text x="{x}" y="250" text-anchor="middle" font-size="12">{escape(row["round"])}</text>')
        previous = (x, y)
    return (f'<svg viewBox="0 0 770 275" role="img" aria-label="{escape(label)}">'
            f'<text x="20" y="22">{escape(label)}</text><path d="M65 45 V225 H720" fill="none" stroke="#aaa"/>'
            f'<text x="5" y="55" font-size="11">{high:.2f}</text><text x="5" y="225" font-size="11">{low:.2f}</text>'
            + "".join(content) + '</svg>')


def _finite_summary(rows, column):
    values = [
        row.get(column)
        for row in rows
        if isinstance(row.get(column), (int, float))
        and not isinstance(row.get(column), bool)
        and math.isfinite(row[column])
    ]
    if not values:
        return None, None
    return sum(values) / len(values), median(values)


def _abc_item_summaries(rows, items_by_form, c_final):
    paths = {
        "A": "A/round_01",
        "B": "B/round_01",
        "C": c_final.get("source_round"),
    }
    form_rows = {row["path"]: row for row in rows}
    summaries = []
    for method in ("A", "B", "C"):
        path = paths.get(method)
        form_row = form_rows.get(path)
        item_rows = items_by_form.get(path, [])
        summary = {
            "method": method,
            "path": path,
            "available": bool(
                form_row
                and form_row.get("evaluation_status") == "complete"
                and item_rows
            ),
        }
        if summary["available"]:
            for column in (
                "citc",
                "target_rho",
                "same_domain_vts",
                "cross_domain_vts",
            ):
                mean, middle = _finite_summary(item_rows, column)
                summary[f"{column}_mean"] = mean
                summary[f"{column}_median"] = middle
            qualified = sum(row.get("qualified") is True for row in item_rows)
            summary["qualification_rate"] = qualified / len(item_rows)
        summaries.append(summary)
    return summaries


def _abc_form_summaries(rows, c_final):
    paths = {
        "A": "A/round_01",
        "B": "B/round_01",
        "C": c_final.get("source_round"),
    }
    form_rows = {row["path"]: row for row in rows}
    summaries = []
    for method in ("A", "B", "C"):
        source = form_rows.get(paths.get(method))
        summary = {"method": method, "round": method, "available": False}
        if source and source.get("evaluation_status") == "complete":
            quality = source.get("quality") or {}
            summary.update(
                available=True,
                alpha=source.get("alpha"),
                icc=source.get("icc"),
                target_rho=source.get("target_rho"),
                delta_min=source.get("delta_min"),
                target_g=source.get("target_g"),
                source_round=source.get("path"),
            )
        summaries.append(summary)
    return summaries


def _abc_q_bar_chart(rows):
    valid_values = [
        row.get("q")
        for row in rows
        if isinstance(row.get("q"), (int, float))
        and not isinstance(row.get("q"), bool)
        and math.isfinite(row["q"])
    ]
    if not valid_values:
        return "<p>整卷总体质量Q：尚无可绘制数据。</p>"

    span = max(valid_values) - min(valid_values)
    padding = max(0.01, span * 0.25)
    low = max(0.0, min(valid_values) - padding)
    high = min(1.0, max(valid_values) + padding)
    if high <= low:
        low = max(0.0, min(valid_values) - 0.05)
        high = min(1.0, max(valid_values) + 0.05)

    x_positions = {"A": 160.0, "B": 385.0, "C": 610.0}
    colors = {"A": "#8a9a91", "B": "#c7924b", "C": "#266451"}
    chart_top, chart_bottom = 55.0, 225.0
    baseline = next(
        (
            row.get("q")
            for row in rows
            if row.get("method") == "A"
            and isinstance(row.get("q"), (int, float))
            and not isinstance(row.get("q"), bool)
            and math.isfinite(row["q"])
        ),
        None,
    )
    content = []
    for row in rows:
        method = row["method"]
        x = x_positions[method]
        value = row.get("q")
        content.append(
            f'<text x="{x:.1f}" y="252" text-anchor="middle" font-size="13">{method}</text>'
        )
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            content.append(
                f'<text x="{x:.1f}" y="145" text-anchor="middle" font-size="12" fill="#777">待评估</text>'
            )
            continue
        y = chart_bottom - (value - low) / (high - low) * (chart_bottom - chart_top)
        height = chart_bottom - y
        if method == "A":
            accessible_label = f"A 基线: {value:.3f}"
            delta_label = "基线"
        elif baseline is None:
            accessible_label = f"{method}: {value:.3f}"
            delta_label = "缺少A基线"
        else:
            delta = value - baseline
            accessible_label = f"{method}: {value:.3f}（较A {delta:+.3f}）"
            delta_label = f"较A {delta:+.3f}"
        content.append(
            f'<rect x="{x - 50:.1f}" y="{y:.1f}" width="100" height="{height:.1f}" '
            f'fill="{colors[method]}" opacity="0.9"><title>{accessible_label}</title></rect>'
        )
        content.append(
            f'<text x="{x:.1f}" y="{max(chart_top - 5, y - 8):.1f}" text-anchor="middle" '
            f'font-size="13" font-weight="bold">{value:.3f}</text>'
        )
        content.append(
            f'<text x="{x:.1f}" y="273" text-anchor="middle" font-size="12" '
            f'fill="#4f5e57">{delta_label}</text>'
        )

    warning = "纵轴截断：柱高从所示下限起算" if low > 0 else "纵轴从0开始"
    return (
        '<svg viewBox="0 0 770 300" role="img" aria-label="整卷总体质量Q放大柱状图">'
        '<text x="20" y="24">整卷总体质量Q（放大尺度）</text>'
        f'<text x="750" y="24" text-anchor="end" font-size="12" fill="#a24b32">{warning}</text>'
        f'<path d="M65 {chart_top:.0f} V{chart_bottom:.0f} H720" fill="none" stroke="#aaa"/>'
        f'<text x="8" y="{chart_top + 5:.0f}" font-size="11">{high:.3f}</text>'
        f'<text x="8" y="{chart_bottom:.0f}" font-size="11">{low:.3f}</text>'
        + "".join(content)
        + "</svg>"
    )


def _abc_chart(rows, mean_column, label, median_column=None, bounded=False):
    series = [(mean_column, "均值", "#266451", "")]
    if median_column:
        series.append((median_column, "中位数", "#c46a32", ' stroke-dasharray="7 5"'))
    values = [
        row.get(column)
        for row in rows
        for column, _, _, _ in series
        if isinstance(row.get(column), (int, float))
        and not isinstance(row.get(column), bool)
        and math.isfinite(row[column])
    ]
    if not values:
        return f"<p>{escape(label)}：尚无可绘制数据。</p>"
    low, high = (0.0, 1.0) if bounded else (min(0.0, min(values)), max(1.0, max(values)))
    if high <= low:
        high = low + 1.0
    x_positions = {"A": 160.0, "B": 385.0, "C": 610.0}
    content = []
    for column, series_label, color, dash in series:
        previous = None
        for row in rows:
            method = row["method"]
            x = x_positions[method]
            value = row.get(column)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                previous = None
                continue
            y = 225 - (value - low) / (high - low) * 175
            if previous:
                content.append(
                    f'<line x1="{previous[0]:.1f}" y1="{previous[1]:.1f}" '
                    f'x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2.5"{dash}/>'
                )
            content.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}">'
                f'<title>{method} {series_label}: {value:.3f}</title></circle>'
            )
            previous = (x, y)
    for row in rows:
        method = row["method"]
        x = x_positions[method]
        content.append(f'<text x="{x:.1f}" y="250" text-anchor="middle" font-size="13">{method}</text>')
        if not row.get("available"):
            content.append(f'<text x="{x:.1f}" y="140" text-anchor="middle" font-size="12" fill="#777">待评估</text>')
    legend = (
        '<line x1="505" y1="22" x2="535" y2="22" stroke="#266451" stroke-width="2.5"/>'
        '<text x="541" y="26" font-size="12">均值</text>'
    )
    if median_column:
        legend += (
            '<line x1="605" y1="22" x2="635" y2="22" stroke="#c46a32" '
            'stroke-width="2.5" stroke-dasharray="7 5"/>'
            '<text x="641" y="26" font-size="12">中位数</text>'
        )
    return (
        f'<svg viewBox="0 0 770 275" role="img" aria-label="{escape(label)}">'
        f'<text x="20" y="22">{escape(label)}</text>{legend}'
        '<path d="M65 45 V225 H720" fill="none" stroke="#aaa"/>'
        f'<text x="5" y="55" font-size="11">{high:.2f}</text>'
        f'<text x="5" y="225" font-size="11">{low:.2f}</text>'
        + "".join(content)
        + '</svg>'
    )


def write_report(store):
    rows, item_rows, items_by_form = [], [], {}
    c_final = store.read("C/final.json", {})
    neo_reference = store.read("reference/neo_ffi/metrics.json", {})
    for key in store.forms():
        form = store.read(f"{key}/form.json")
        metrics = store.read(f"{key}/evaluation/metrics.json", {})
        quality = metrics.get("quality") or {}
        row = {"method": form["method"], "round": form["round"], "path": key,
               "final": key == c_final.get("source_round") if form["method"] == "C" else True,
               "evaluation_status": metrics.get("status", "pending"), "item_count": len(form["items"]),
               "qualified_count": metrics.get("qualified_item_count"), "qualification_rate": metrics.get("qualification_rate"),
               "alpha": (quality.get("alpha_gate") or {}).get("observed"),
               "icc": (quality.get("stability_gate") or {}).get("observed"),
               "target_rho": quality.get("ipip_target_facet_spearman_rho"),
               "delta_min": quality.get("ipip_discriminant_delta_min"),
               "target_g": quality.get("ipip_target_known_groups_hedges_g"),
               "quality": quality}
        rows.append(row)
        current_items = store.read(f"{key}/evaluation/items.json", [])
        items_by_form[key] = current_items
        item_rows.extend(current_items)
    costs = [{"stage": key, **cost_summary(store, key)} for key in
             ["A/round_01", "shared", "B/round_01", "C", *[r["path"] + "/evaluation" for r in rows]]]
    for cost in costs:
        store.write(f"{cost['stage']}/cost.json", {k: v for k, v in cost.items() if k != "stage"})
    stage_cost = {c["stage"]: c for c in costs}
    method_costs = []
    for method, components in (("A", ["A/round_01"]), ("B", ["shared", "B/round_01"]),
                               ("C", ["shared", "B/round_01", "C"])):
        row = {"method": method, "components": components}
        for field in ("total_tokens", "fee", "wall_seconds"):
            vals = [stage_cost[k].get(field) for k in components]
            row[field] = sum(vals) if all(v is not None for v in vals) else None
        method_costs.append(row)
    iteration_rows = [r for r in rows if r["method"] == "C"]
    # Freeze costs at the assembly cutoff, not by action iteration labels:
    # repairs AFTER a form may still carry that form's iteration label.
    for row in iteration_rows:
        record = store.read(row["path"] + "/development/iteration_record.json", {})
        development_quality = record.get("form_metrics") or {}
        development_summary = form_quality_summary(development_quality)
        row["development_target_g"] = development_summary.get("objective_primary")
        row["development_delta_min"] = development_summary.get("objective_secondary")
        row["development_target_rho"] = development_summary.get("objective_tertiary")
        cost = store.read(row["path"] + "/cost.json", {})
        row["cumulative_tokens"] = 0 if row["round"] == "baseline" else cost.get("cumulative_c_cost", {}).get("total_tokens")
        row["cumulative_wall_seconds"] = 0 if row["round"] == "baseline" else cost.get("cumulative_c_cost", {}).get("wall_seconds")
        retained = store.read(row["path"] + "/retained_form.json", {})
        row["retained_source_round"] = retained.get("source_round")
        row["retained_development_q"] = retained.get("development_quality")
    write_csv(store.path("summary/method_comparison.csv"), rows)
    write_csv(store.path("summary/item_comparison.csv"), item_rows)
    write_csv(store.path("summary/iteration_metrics.csv"), iteration_rows)
    write_csv(store.path("summary/stage_costs.csv"), costs)
    write_csv(store.path("summary/method_costs.csv"), method_costs)
    abc_form_rows = _abc_form_summaries(rows, c_final)
    abc_item_rows = _abc_item_summaries(rows, items_by_form, c_final)
    html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A/B/C实验报告</title>',
            '<style>body{font:15px/1.7 "Microsoft YaHei",sans-serif;max-width:1150px;margin:35px auto;padding:20px;color:#243b32;background:#f6f8f5}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border-bottom:1px solid #d8e3dd;text-align:left;white-space:nowrap}.scroll{overflow:auto}svg{max-width:760px;background:white;margin:12px 0}h2{margin-top:32px}a{color:#266451}</style>',
            f'<h1>A/B/C独立单批次实验</h1><p>{escape(store.root.name)} · 状态：{escape(store.read("progress.json")["status"])}</p>',
            '<p>单批次描述性结果，无跨批次显著性检验。虚拟指标不等同于真人信效度。不可估计值不按0处理。</p>',
            '<h2>各方法与各轮问卷</h2>',
            _table(rows, [("method", "方法"), ("round", "轮次"), ("final", "最终选定"), ("evaluation_status", "评估"),
                          ("item_count", "题数"), ("qualified_count", "通过题数"), ("qualification_rate", "通过比例"),
                          ("target_g", "目标Hedges’ g"), ("target_rho", "目标IPIP相关"),
                          ("delta_min", "Δmin区分效度"), ("alpha", "Cronbach α"), ("icc", "ICC")]),
            '<h2>A→B→C整卷指标</h2>',
            '<p>α和ICC是整卷信度与稳定性门槛；目标Hedges’ g是主要优化指标；目标IPIP相关是聚合效度，Δmin是区分效度保护指标。C使用流程最终选定轮次，不按独立评估结果事后挑选。</p>',
            _table(abc_form_rows, [("method", "方法"), ("target_g", "目标Hedges’ g"),
                                   ("target_rho", "目标IPIP相关"), ("delta_min", "Δmin区分效度"),
                                   ("alpha", "Cronbach α"), ("icc", "ICC")]),
            _abc_chart(abc_form_rows, "target_g", "目标Hedges’ g", None, False),
            _abc_chart(abc_form_rows, "target_rho", "目标IPIP Spearman相关", None, False),
            _abc_chart(abc_form_rows, "delta_min", "Δmin区分效度", None, False)]
    if neo_reference.get("status") == "complete":
        neo_rows = [
            {
                **row,
                "evidence_label": "汇聚" if row.get("evidence_type") == "convergent" else "区分",
            }
            for row in neo_reference.get("correlations", [])
        ]
        model_note = (
            f'NEO-FFI模型：{escape(str(neo_reference.get("neo_ffi_model_id") or "未知"))}；'
            f'SJT模型：{escape(str(neo_reference.get("sjt_model_ids") or "未知"))}。'
        )
        limitation = (
            "这是跨模型探索结果：同一批虚拟人格由不同模型完成SJT和NEO-FFI，"
            "不能解释为严格同模型效度或真人效度。"
            if neo_reference.get("cross_model_exploration")
            else "SJT与NEO-FFI由同一模型完成；结果仍只代表虚拟被试内部关联。"
        )
        html += [
            '<h2>NEO-FFI汇聚与区分相关</h2>',
            f'<p>{model_note}{limitation}</p>',
            _table(
                neo_rows,
                [
                    ("method", "方法"),
                    ("neo_domain", "NEO-FFI维度"),
                    ("evidence_label", "类型"),
                    ("spearman_rho", "Spearman ρ"),
                    ("n", "配对人数"),
                    ("estimable", "可估计"),
                    ("unavailable_reason", "不可估计原因"),
                ],
            ),
        ]
    html += [
        '<h2>A→B→C单题指标总体变化</h2>',
        '<p>这是方法层面的汇总比较，不是同一道题的纵向追踪。A为单提示词卷，B为理论组卷，C仅取最终选定轮次；未完成独立评估的方法显示为待评估。</p>'
    ]
    for mean_column, median_column, label, bounded in (
        ("citc_mean", "citc_median", "CITC", False),
        ("target_rho_mean", "target_rho_median", "目标ρ", False),
        ("same_domain_vts_mean", "same_domain_vts_median", "同域VTS", False),
        ("cross_domain_vts_mean", "cross_domain_vts_median", "跨域VTS", False),
        ("qualification_rate", None, "四门槛通过率", True),
    ):
        html.append(_abc_chart(abc_item_rows, mean_column, label, median_column, bounded))
    html += [
            '<h2>C迭代轨迹</h2><p>独立评估质量可以下降。baseline为B理论卷，round_01为首次指标组卷；后续为返修后组卷。</p>']
    for column, label, bounded in (("target_g", "独立评估目标Hedges’ g", False),
                                    ("delta_min", "独立评估Δmin区分效度", False),
                                    ("target_rho", "独立评估目标IPIP相关", False),
                                    ("alpha", "独立评估Cronbach α", True),
                                    ("icc", "独立评估ICC", True),
                                    ("cumulative_tokens", "C新增累计Token（不含共享开发与独立评估）", False),
                                    ("cumulative_wall_seconds", "C新增累计运行秒数（含人工等待）", False)):
        html.append(_chart(iteration_rows, column, label, bounded))
    html += ['<h2>每道题的四项指标</h2>',
             _table(item_rows, [("method", "方法"), ("round", "轮次"), ("item_id", "题号"), ("item_version", "版本"),
                                ("citc", "CITC"), ("target_rho", "目标ρ"), ("same_domain_vts", "同域VTS"),
                                ("cross_domain_vts", "跨域VTS"), ("qualified", "四门槛通过"), ("failed_gates", "未通过门槛")]),
             '<p>各轮evaluation/item_metrics.csv包含阈值、可估计状态与非目标相关；option_statistics.csv记录选项人数及比例。</p>',
             '<details><summary>指标口径与含义</summary><p>CITC：目标条件中单题分数与同facet其余题总分的Pearson相关；单题目标ρ和VTS仍按原单题规则计算。</p>',
             '<p>整卷Cronbach α用于内部一致性门槛；ICC衡量同一虚拟人格重复施测时的整卷总分一致性。目标IPIP相关是聚合效度；Δmin=目标IPIP相关−四个非目标IPIP相关绝对值中的最大值，用于区分效度保护；目标Hedges’ g是IPIP目标facet高低组的已知组效度效应量。</p>',
             '<p>新流程要求α≥0.80、ICC≥0.80；组卷主要提高目标Hedges’ g，且目标IPIP相关最多下降0.02、Δmin不得下降。旧版虚拟综合指标仅保留在历史数据中，不参与新流程选卷。</p></details>',
             '<h2>方法端到端开发成本</h2><p>B/C分别计入共享题库及理论组卷成本；真实账单应求和下方阶段表，不能将B/C端到端费用相加。</p>',
             _table(method_costs, [("method", "方法"), ("total_tokens", "Token"), ("wall_seconds", "墙钟秒数（含人工等待）"), ("fee", "费用")]),
             '<h2>实际阶段成本（共享成本只列一次）</h2>',
             _table(costs, [("stage", "阶段"), ("calls", "调用"), ("total_tokens", "Token"), ("wall_seconds", "墙钟秒数"),
                            ("model_duration_ms", "模型累计毫秒"), ("fee", "费用")]),
             '<p>缺少用量、缓存明细或价格时费用未知。A/B无虚拟反馈；C/B差异包含额外计算投入的总体收益。</p>',
             '<h2>问卷文件</h2><ul>']
    for row in rows:
        html.append(f'<li><a href="../{escape(row["path"], quote=True)}/form.html">{escape(row["path"])}</a></li>')
    html += ['</ul></html>']
    store.path("summary/report.html").write_text("\n".join(html), encoding="utf-8")
    return store.path("summary/report.html")
