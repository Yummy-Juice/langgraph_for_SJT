"""Render a Spearman conductance matrix as a standalone HTML heatmap.

Usage:
  python tools/render_heatmap.py <conductance_matrix.csv> [--out out.html]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


LABEL_MAP = {
    "openness_ideas": "开放性",
    "conscientiousness_self_discipline": "责任心",
    "extraversion_gregariousness": "外倾性",
    "agreeableness_compliance": "宜人性",
    "neuroticism_self_consciousness": "神经质",
    "openness": "开放性",
    "neuroticism": "神经质",
    "extraversion": "外倾性",
    "conscientiousness": "责任心",
    "agreeableness": "宜人性",
}

# 规范维度族顺序（开放性 → 责任心 → 外倾性 → 宜人性 → 神经质），
# 行列都按此重排，保证对角线在视觉上成一条斜线。
FAMILY_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def _family(dimension_id: str) -> str:
    for prefix in FAMILY_ORDER:
        if dimension_id.startswith(prefix):
            return prefix
    return dimension_id


def _sort_key(dimension_id: str) -> int:
    family = _family(dimension_id)
    return FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)


def _color(rho: float) -> str:
    """Red-blue diverging scale for rho in [-1, 1]."""
    if rho is None or pd.isna(rho):
        return "#ffffff"
    t = max(-1.0, min(1.0, float(rho)))
    if t >= 0:
        intensity = t ** 0.7  # gamma for readability
        red = 255
        green = int(255 * (1 - intensity))
        blue = int(255 * (1 - intensity))
    else:
        intensity = (-t) ** 0.7
        red = int(255 * (1 - intensity))
        green = int(255 * (1 - intensity))
        blue = 255
    return f"rgb({red},{green},{blue})"


def render(matrix: pd.DataFrame, out_path: Path) -> None:
    rho_cols = [col for col in matrix.columns if col.startswith("rho_")]
    dims = sorted(
        (col[len("rho_"):] for col in rho_cols),
        key=_sort_key,
    )

    header = "".join(
        f'<th style="padding:8px;background:#f0f0f0;font-size:13px;">'
        f'{LABEL_MAP.get(dim, dim)}</th>'
        for dim in dims
    )
    body_rows = []
    for _, row in matrix.sort_values(
        "facet_id"
        if "facet_id" in matrix.columns
        else "domain_id",
        key=lambda series: series.map(_sort_key),
    ).iterrows():
        facet_id = str(row.get("facet_id") or row.get("domain_id") or "")
        cells = []
        for dim in dims:
            rho = row.get(f"rho_{dim}")
            p = row.get(f"p_{dim}")
            if rho is None or pd.isna(rho):
                cells.append('<td style="padding:12px;background:#ffffff;"></td>')
                continue
            color = _color(float(rho))
            text = f"{float(rho):+.2f}"
            star = "*" if p is not None and not pd.isna(p) and float(p) < 0.05 else ""
            bold = "font-weight:bold;" if _family(dim) == _family(facet_id) else ""
            cells.append(
                f'<td style="padding:12px;text-align:center;background:{color};'
                f'{bold}font-size:14px;border:1px solid #ddd;">{text}{star}</td>'
            )
        body_rows.append(
            f'<tr><td style="padding:8px;font-weight:bold;font-size:13px;">'
            f'{LABEL_MAP.get(facet_id, facet_id)}</td>' + "".join(cells) + "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Mussel 传导矩阵热图</title></head>
<body style="font-family:system-ui,sans-serif;margin:24px;">
<h2>Mussel 传导矩阵（Spearman ρ）</h2>
<p>行 = 已知 persona 分数维度；列 = Mussel 维度 0-1 均分。<br>
红色 = 正相关，蓝色 = 负相关，<b>加粗</b> = 对角线（对应维度），* = p &lt; 0.05。</p>
<table style="border-collapse:collapse;">{header}{''.join(body_rows)}</table>
<p style="color:#666;font-size:12px;margin-top:16px;">
解释：对角线应显著为正（构念效度）；非对角线应明显更弱（判别效度）。</p>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[heatmap] 已写入 {out_path}")


def render_diagonal(matrix: pd.DataFrame, out_path: Path, *, title: str) -> None:
    """Render only the diagonal (each dimension against itself) as bars."""

    rows_html = []
    for _, row in matrix.iterrows():
        facet_id = str(row.get("facet_id") or row.get("domain_id") or "")
        rho = row.get(f"rho_{facet_id}")
        p = row.get(f"p_{facet_id}")
        if rho is None or pd.isna(rho):
            rows_html.append(
                f'<tr><td>{LABEL_MAP.get(facet_id, facet_id)}</td>'
                f'<td style="text-align:right;">—</td><td></td></tr>'
            )
            continue
        value = float(rho)
        width = max(2, int(abs(value) * 240))
        color = _color(value)
        star = "*" if p is not None and not pd.isna(p) and float(p) < 0.05 else ""
        bar = (
            f'<div style="background:{color};width:{width}px;height:22px;'
            f'border-radius:3px;"></div>'
        )
        rows_html.append(
            f'<tr>'
            f'<td style="padding:8px;font-weight:bold;font-size:14px;">'
            f'{LABEL_MAP.get(facet_id, facet_id)}</td>'
            f'<td style="padding:8px;text-align:right;font-size:14px;">'
            f'{value:+.2f}{star}</td>'
            f'<td style="padding:8px;">{bar}</td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title} 对角线传导</title></head>
<body style="font-family:system-ui,sans-serif;margin:24px;">
<h2>{title}：已知分数 → 对应维度 0-1 均分（Spearman ρ，对角线）</h2>
<p>每组被试只给一个维度的分数；数值 = 该分数与同维度作答得分的相关，
* = p &lt; 0.05。非对角线（判别效度）数据见 conductance_matrix.csv。</p>
<table>{''.join(rows_html)}</table>
<p style="color:#666;font-size:12px;margin-top:16px;">
参考：&gt;0.7 强传导；0.3~0.7 中等；&lt;0.3 弱（宜人性偏低需单独排查）。</p>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[diagonal] 已写入 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_csv", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--diagonal",
        action="store_true",
        help="只渲染对角线（每个维度自己对自己）",
    )
    parser.add_argument("--title", type=str, default="传导矩阵")
    args = parser.parse_args()
    matrix = pd.read_csv(args.matrix_csv)
    if args.diagonal:
        out = (
            Path(args.out)
            if args.out
            else Path(args.matrix_csv).with_name(
                Path(args.matrix_csv).stem + "_diagonal.html"
            )
        )
        render_diagonal(matrix, out, title=args.title)
        return
    out = Path(args.out) if args.out else Path(args.matrix_csv).with_suffix(".html")
    render(matrix, out)


if __name__ == "__main__":
    main()
