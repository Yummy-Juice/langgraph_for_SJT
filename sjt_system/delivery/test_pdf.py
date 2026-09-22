"""Build a respondent-facing Chinese test form as a printable HTML document.

The output is a self-contained HTML file (inline CSS) that can be printed to
PDF from any browser. Structure:

  1. Cover / administration instructions
  2. Items, organized by target dimension
  3. Scoring and score interpretation
  4. Psychometric evidence report (Cronbach alpha, convergent validity vs
     facet-matched IPIP-NEO, discriminant structure) — clearly labelled as development-stage
     virtual evidence, not human criterion data.

Usage (programmatic):
    from sjt_system.delivery.test_pdf import build_test_form_html
    build_test_form_html(state, output_path)
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from sjt_system.config import (
    PSJT_OPTION_COUNT,
    PSJT_RESPONSE_INSTRUCTION,
    PSJT_SCORING_METHOD,
)
from sjt_system.runtime.trace import utc_timestamp

DEVELOPMENT_EVIDENCE_NOTICE = (
    "本测验的全部统计证据来自开发期虚拟被试作答，属于模拟筛选结果，"
    "尚未经过真实被试验证，不构成正式信度、效度或常模证据。"
)

_OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def _esc(value: Any) -> str:
    return _html.escape(str(value if value is not None else ""))


def _dimension_label(construct_profile: Mapping[str, Any], dimension_id: str) -> str:
    for facet in construct_profile.get("facets") or []:
        if not isinstance(facet, Mapping):
            continue
        if str(facet.get("facet_id")) == dimension_id:
            return str(
                facet.get("facet_name")
                or facet.get("facet_name_en")
                or dimension_id
            )
    return dimension_id


def _items_by_dimension(
    respondent_items: Sequence[Mapping[str, Any]],
    database_items: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group the official (assembled) items by target dimension.

    The respondent form carries Q-numbered items without dimension ids;
    dimension is resolved by scenario text against the item database.
    """
    dim_by_scenario: dict[str, str] = {}
    for item in database_items:
        if not isinstance(item, Mapping):
            continue
        scenario = str(item.get("scenario") or "")
        if scenario:
            dim_by_scenario[scenario] = str(
                item.get("target_dimension_id") or "未分类"
            )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for respondent_item in respondent_items:
        if not isinstance(respondent_item, Mapping):
            continue
        scenario = str(respondent_item.get("scenario") or "")
        dimension_id = dim_by_scenario.get(scenario, "未分类")
        grouped.setdefault(dimension_id, []).append(dict(respondent_item))
    order = sorted(grouped)
    return [(dimension_id, grouped[dimension_id]) for dimension_id in order]


def build_test_form_html(
    state: Mapping[str, Any],
    output_path: str | Path,
) -> str:
    final_test = state.get("final_test") or {}
    item_database = state.get("item_database") or {}
    test_statistics = state.get("test_statistics") or {}
    construct_profile = state.get("construct_profile") or {}
    specification = state.get("test_specification") or {}

    respondent_form = final_test.get("respondent_form") or {}
    scoring_key = final_test.get("scoring_key") or {}
    title = str(respondent_form.get("title") or final_test.get("test_id") or "情境判断测验")
    population = str(specification.get("target_population") or respondent_form.get("target_population") or "—")

    database_items = (
        item_database.get("items")
        if isinstance(item_database, Mapping) and isinstance(item_database.get("items"), list)
        else item_database if isinstance(item_database, list) else []
    )
    # 正式题以组卷后的被试卷为准（final_test.respondent_form.items）
    respondent_items = (respondent_form.get("items") or [])
    dimensions = _items_by_dimension(respondent_items, database_items)
    # 未拿到组卷被试卷时（如仅 item_database），按数据库题展示（用于演示/调试）
    if not dimensions:
        dimensions = [
            (str(item.get("target_dimension_id") or "未分类"), [dict(item)])
            for item in database_items
            if isinstance(item, Mapping)
        ]

    # ---- 信效度证据 ----
    alpha = test_statistics.get("cronbach_alpha")
    dimension_alphas = {}
    for dimension_id, dimension_stats in (test_statistics.get("dimensions") or {}).items():
        if isinstance(dimension_stats, Mapping) and dimension_stats.get("cronbach_alpha") is not None:
            dimension_alphas[str(dimension_id)] = dimension_stats["cronbach_alpha"]
    # 汇聚效度：优先读取facet匹配的IPIP-NEO，兼容历史NEO-FFI结果。
    reference_rho = None
    reference_label = "IPIP-NEO目标facet"
    iteration_history = state.get("psychometric_iteration_history") or []
    for entry in reversed(iteration_history):
        if not isinstance(entry, Mapping):
            continue
        fm = entry.get("form_metrics") or {}
        validity = fm.get("validity") or {}
        convergent = validity.get("convergent_validity") or {}
        candidate = convergent.get("spearman_rho")
        if isinstance(candidate, (int, float)):
            reference_rho = float(candidate)
            reference_label = str(
                convergent.get("reference_questionnaire")
                or reference_label
            )
            break
        legacy = validity.get("legacy_virtual_diagnostics") or {}
        candidate = next(
            (
                legacy.get(key)
                for key in (
                    "ipip_neo_target_facet_rho",
                    "neo_ffi_rho",
                    "combined_reference_rho",
                )
                if isinstance(legacy.get(key), (int, float))
            ),
            None,
        )
        if isinstance(candidate, (int, float)):
            reference_rho = float(candidate)
            if legacy.get("ipip_neo_target_facet_rho") is None:
                reference_label = "NEO-FFI目标domain（历史）"
            break
    item_metrics = (
        (test_statistics.get("virtual_screening_metrics") or {}).get("item_metrics") or {}
    )
    citc_values = []
    target_rho_values = []
    for _item_id, metrics in item_metrics.items():
        if not isinstance(metrics, Mapping):
            continue
        citc = (metrics.get("facet_citc") or {}).get("r")
        rho = ((metrics.get("virtual_target_specificity") or {}).get("rho_target") or {}).get("rho")
        if isinstance(citc, (int, float)):
            citc_values.append(float(citc))
        if isinstance(rho, (int, float)):
            target_rho_values.append(float(rho))
    median_citc = sorted(citc_values)[len(citc_values) // 2] if citc_values else None
    median_target_rho = sorted(target_rho_values)[len(target_rho_values) // 2] if target_rho_values else None

    # ---- ① 说明 ----
    item_count = sum(len(items) for _, items in dimensions)
    sections = [
        "<div class='cover'>",
        f"<h1>{_esc(title)}</h1>",
        "<p class='subtitle'>作答与计分说明</p>",
        "<table class='info'>",
        f"<tr><td>适用人群</td><td>{_esc(population)}</td></tr>",
        f"<tr><td>题量</td><td>{item_count} 道情境判断题</td></tr>",
        f"<tr><td>作答方式</td><td>每题阅读情境后，从 {PSJT_OPTION_COUNT} 个选项中选出你<strong>最可能采取</strong>的行为（{_esc(PSJT_RESPONSE_INSTRUCTION)}）</td></tr>",
        f"<tr><td>计分方式</td><td>{_esc(PSJT_SCORING_METHOD)}</td></tr>",
        "</table>",
        "<div class='notice'>作答提示：选项没有绝对对错，请选择最符合你实际行为倾向的一项。</div>",
        "</div>",
    ]

    # ---- ② 题目 ----
    sections.append("<div class='items'>")
    sections.append("<h2>测验题目</h2>")
    running = 0
    for dimension_id, items in dimensions:
        sections.append(f"<h3>{_esc(_dimension_label(construct_profile, dimension_id))}</h3>")
        for item in items:
            running += 1
            scenario = str(item.get("scenario") or "")
            options = item.get("response_options") or []
            sections.append(
                f"<div class='item'><p class='q'><strong>{running}.</strong> {_esc(scenario)}</p>"
            )
            sections.append(
                f"<p class='qi'>{_esc(item.get('response_instruction') or PSJT_RESPONSE_INSTRUCTION)}</p>"
            )
            for index, option in enumerate(options):
                if not isinstance(option, Mapping):
                    continue
                label = _OPTION_LABELS[index] if index < len(_OPTION_LABELS) else str(index + 1)
                sections.append(
                    f"<p class='opt'><span class='opt-label'>{label}.</span> {_esc(option.get('text') or '')}</p>"
                )
            sections.append("</div>")
    sections.append("</div>")

    # ---- ③ 分数解释 ----
    sections.append("<div class='scoring'>")
    sections.append("<h2>计分与分数解释</h2>")
    sections.append(
        f"<p>每题选项按行为高低记为 1–4 分（{_esc(PSJT_SCORING_METHOD)}）。"
        "每个维度由若干题组成，维度得分 = 该维度题目得分之和（或均值，见具体版本说明）。"
        "分数越高表示在该人格特质上的倾向越明显。</p>"
    )
    if construct_profile.get("facets"):
        sections.append("<table class='info'><tr><th>维度</th><th>定义</th></tr>")
        for facet in construct_profile.get("facets") or []:
            if not isinstance(facet, Mapping):
                continue
            fid = str(facet.get("facet_id") or "")
            label = str(facet.get("facet_name") or facet.get("facet_name_en") or fid)
            definition = str(facet.get("definition") or "")
            sections.append(
                f"<tr><td>{_esc(label)}</td><td>{_esc(definition)}</td></tr>"
            )
        sections.append("</table>")
    sections.append(
        "<div class='notice'>注意：本测验当前无真实被试常模，分数解释仅基于开发期虚拟作答的分布范围，"
        "正式使用前应完成真实样本的常模与效度研究。</div>"
    )
    sections.append("</div>")

    # ---- ④ 信效度报告 ----
    sections.append("<div class='evidence'>")
    sections.append("<h2>信效度验证报告（开发期）</h2>")
    sections.append(
        f"<div class='notice'>{_esc(DEVELOPMENT_EVIDENCE_NOTICE)}</div>"
    )
    sections.append("<table class='info'>")
    sections.append(
        f"<tr><td>内部一致性</td><td>Cronbach α = {_esc(round(alpha, 3) if alpha is not None else '—')}</td></tr>"
    )
    sections.append(
        f"<tr><td>汇聚效度参照</td><td>与 {_esc(reference_label)} 相关 rho = {_esc(round(reference_rho, 3) if reference_rho is not None else '—')}</td></tr>"
    )
    sections.append(
        f"<tr><td>题目区分度（中位）</td><td>CITC = {_esc(round(median_citc, 3) if median_citc is not None else '—')}；"
        f"目标相关 rho = {_esc(round(median_target_rho, 3) if median_target_rho is not None else '—')}</td></tr>"
    )
    sections.append(
        f"<tr><td>区分效度</td><td>按匹配条件三臂设计（target / 同域 / 跨域）计算，"
        "目标臂相关显著高于非目标臂为合格。</td></tr>"
    )
    sections.append("</table>")
    sections.append("</div>")

    css = """
      body { font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif; font-size: 12pt; line-height: 1.7; color: #222; margin: 40px 60px; }
      h1 { font-size: 22pt; text-align: center; margin-top: 80px; }
      h2 { font-size: 16pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 36px; }
      h3 { font-size: 14pt; color: #444; margin-top: 22px; }
      .subtitle { text-align: center; color: #666; margin-bottom: 40px; }
      .cover { page-break-after: always; }
      .info { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 11pt; }
      .info td, .info th { border: 1px solid #ccc; padding: 6px 10px; vertical-align: top; text-align: left; }
      .notice { background: #fff8e1; border: 1px solid #f0d070; padding: 10px 14px; border-radius: 6px; margin: 12px 0; }
      .item { margin: 16px 0; page-break-inside: avoid; }
      .q { font-weight: 600; margin: 4px 0; }
      .qi { color: #555; margin: 2px 0 6px; }
      .opt { margin: 2px 0 2px 18px; }
      .opt-label { font-weight: 600; }
      @media print { .cover { page-break-after: always; } }
    """

    html_doc = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{css}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    Path(output_path).write_text(html_doc, encoding="utf-8")
    return str(Path(output_path).resolve())


def generate_test_form_pdf_html(
    state: Mapping[str, Any],
    output_root: str | Path,
) -> str:
    """Write the printable test form under outputs/final_reports/<run>/."""
    run_id = str(state.get("run_id") or "unknown")
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_form.html"
    return build_test_form_html(state, path)
