"""Recover source E2 gregariousness scores from saved embodied responses.

This is an offline experiment.  It never calls an LLM: the source personality
responses, the embodied A/B/C SJT responses, and the embodied NEO-FFI scores
must already exist on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from sjt_system.evaluation.form_metrics import (
    _cross_validated_target_recovery,
    _spearman,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_POOL = PROJECT_ROOT / "sjt_system" / "data" / "virtual_respondents.json"
DEFAULT_EVALUATION_DIRNAME = "legacy_summary_embodied_probability_abc"

# Direction is defined against high E2 gregariousness.  The imported source
# file contains wording and answers but not an official scoring-key field.
# These directions therefore remain explicit and auditable rather than hidden
# in a heuristic.  Replace them if the source questionnaire manual differs.
E2_ITEM_DIRECTIONS: dict[str, int] = {
    "Q9_2": -1,  # avoids crowds
    "Q9_3": 1,   # likes many people nearby
    "Q9_4": -1,  # prefers doing things alone
    "Q9_5": 1,   # needs other people after being alone
    "Q9_6": -1,  # prefers solitary work
    "Q9_7": 1,   # prefers a crowded beach
    "Q9_8": -1,  # finds social gatherings boring
    "Q9_9": 1,   # likes gatherings with many people
}


@dataclass(frozen=True)
class E2RecoveryConfig:
    experiment: Path
    source_pool: Path = DEFAULT_SOURCE_POOL
    evaluation_dirname: str = DEFAULT_EVALUATION_DIRNAME
    output: Path | None = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cronbach_alpha(matrix: pd.DataFrame) -> float | None:
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    total_variance = float(matrix.sum(axis=1).var(ddof=1))
    if not np.isfinite(total_variance) or total_variance <= 0:
        return None
    item_variance = float(matrix.var(axis=0, ddof=1).sum())
    value = matrix.shape[1] / (matrix.shape[1] - 1) * (
        1 - item_variance / total_variance
    )
    return float(value) if np.isfinite(value) else None


def score_source_e2(source_pool: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return keyed E2 item scores and source wording for every respondent."""

    payload = _read_json(source_pool)
    items = payload.get("items")
    respondents = payload.get("respondents")
    if not isinstance(items, list) or not isinstance(respondents, list):
        raise ValueError("原始虚拟被试池缺少items或respondents")

    item_index: dict[str, int] = {}
    wording: dict[str, str] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("item_id") or "")
        if item_id in E2_ITEM_DIRECTIONS:
            item_index[item_id] = index
            wording[item_id] = str(item.get("statement") or "")
    missing = sorted(set(E2_ITEM_DIRECTIONS) - set(item_index))
    if missing:
        raise ValueError(f"原始虚拟被试池缺少E2题目：{missing}")

    rows: list[dict[str, Any]] = []
    for respondent in respondents:
        if not isinstance(respondent, Mapping):
            continue
        respondent_id = str(respondent.get("respondent_id") or "")
        values = respondent.get("response_values")
        if not respondent_id or not isinstance(values, list):
            raise ValueError("原始虚拟被试记录缺少respondent_id或response_values")
        row: dict[str, Any] = {"respondent_id": respondent_id}
        for item_id, direction in E2_ITEM_DIRECTIONS.items():
            raw = values[item_index[item_id]]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{respondent_id}/{item_id}不是有效数值作答")
            raw_value = float(raw)
            if raw_value < 1 or raw_value > 5:
                raise ValueError(f"{respondent_id}/{item_id}超出1–5计分范围")
            row[f"{item_id}_raw"] = raw_value
            row[f"{item_id}_scored"] = raw_value if direction == 1 else 6 - raw_value
        scored_columns = [f"{item_id}_scored" for item_id in E2_ITEM_DIRECTIONS]
        total = float(sum(row[column] for column in scored_columns))
        row["E2_sum"] = total
        row["E2_mean"] = total / len(scored_columns)
        row["E2_percent_0_100"] = (total - 8.0) / 32.0 * 100.0
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("respondent_id")
    if frame.index.has_duplicates:
        raise ValueError("原始虚拟被试池存在重复respondent_id")
    return frame, wording


def _load_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "respondent_id" not in frame.columns:
        raise ValueError(f"{path}缺少respondent_id")
    frame["respondent_id"] = frame["respondent_id"].astype(str)
    frame = frame.set_index("respondent_id")
    if frame.index.has_duplicates or frame.empty:
        raise ValueError(f"{path}为空或被试ID重复")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"{path}包含缺失或非数值作答")
    return numeric


def _metric_row(
    name: str,
    predictor_type: str,
    predictors: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    result = _cross_validated_target_recovery(predictors, target)
    if predictor_type == "total_score":
        direct_rho = _spearman(predictors.iloc[:, 0].tolist(), target.tolist())
    else:
        direct_rho = None
    return {
        "predictor": name,
        "predictor_type": predictor_type,
        "feature_count": int(predictors.shape[1]),
        "n": int(len(predictors.join(target.rename("target"), how="inner").dropna())),
        "direct_spearman": direct_rho,
        "cv_r2": result.get("cross_validated_r2"),
        "cv_prediction_spearman": result.get("prediction_spearman"),
        "cv_status": result.get("status"),
        "cv_reason": result.get("reason"),
        "fold_count": result.get("fold_count"),
        "ridge_penalty": result.get("ridge_penalty"),
    }


def _render_report(
    output: Path,
    summary: Mapping[str, Any],
    metrics: pd.DataFrame,
    wording: Mapping[str, str],
) -> Path:
    def number(value: Any) -> str:
        return "—" if pd.isna(value) else f"{float(value):.3f}"

    metric_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.predictor))}</td>"
        f"<td>{escape(str(row.predictor_type))}</td>"
        f"<td>{int(row.feature_count)}</td>"
        f"<td>{int(row.n)}</td>"
        f"<td>{number(row.direct_spearman)}</td>"
        f"<td>{number(row.cv_r2)}</td>"
        f"<td>{number(row.cv_prediction_spearman)}</td>"
        "</tr>"
        for row in metrics.itertuples(index=False)
    )
    item_rows = "".join(
        "<tr>"
        f"<td>{escape(item_id)}</td>"
        f"<td>{'正向' if direction == 1 else '反向（6−原分）'}</td>"
        f"<td>{escape(wording.get(item_id, ''))}</td>"
        "</tr>"
        for item_id, direction in E2_ITEM_DIRECTIONS.items()
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>乐群性E2恢复实验</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1100px;margin:32px auto;line-height:1.55;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:14px 0 28px}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:left}}th{{background:#f3f4f6}}.note{{background:#fff7ed;border-left:4px solid #f97316;padding:12px}}code{{background:#f3f4f6;padding:2px 5px}}</style></head>
<body><h1>具身虚拟被试：原始乐群性E2恢复实验</h1>
<p>样本数：{summary['respondent_count']}；原始E2均值：{summary['source_e2_mean']:.3f}；标准差：{summary['source_e2_sd']:.3f}；范围：{summary['source_e2_min']:.0f}–{summary['source_e2_max']:.0f}；原始8题α：{number(summary['source_e2_alpha'])}。</p>
<div class="note">本实验检查保存的人格信息能否从后续SJT或NEO-FFI作答中恢复。它是虚拟被试内部的人格传导检验，不等同于真人效度。当前原始数据未保存正式计分键，E2方向按题目含义显式指定，正式论文使用前仍须与原问卷手册核对。</div>
<h2>恢复结果</h2><table><thead><tr><th>预测来源</th><th>输入</th><th>变量数</th><th>N</th><th>总分与E2相关</th><th>五折预测R²</th><th>预测值与E2相关</th></tr></thead><tbody>{metric_rows}</tbody></table>
<p><strong>读法：</strong>五折预测R²越高，表示没有参与训练的虚拟被试，其原始E2分数越容易从该问卷作答中猜回来。A/B/C的16题模型使用完整逐题答案；“总分”行更接近传统测验总分相关。</p>
<h2>E2计分方向</h2><table><thead><tr><th>题号</th><th>方向</th><th>题目</th></tr></thead><tbody>{item_rows}</tbody></table>
</body></html>"""
    path = output / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_source_e2_recovery(config: E2RecoveryConfig) -> tuple[Path, dict[str, Any]]:
    experiment = config.experiment.resolve()
    evaluation = experiment / config.evaluation_dirname
    output = (
        config.output.resolve()
        if config.output is not None
        else evaluation / "source_e2_recovery"
    )
    output.mkdir(parents=True, exist_ok=True)

    source, wording = score_source_e2(config.source_pool.resolve())
    target = source["E2_sum"].rename("source_E2_sum")
    matrices = {
        method: _load_matrix(evaluation / method / "evaluation" / "score_matrix.csv")
        for method in ("A", "B", "C")
    }
    neo = _load_matrix(evaluation / "neo_ffi" / "scores.csv")

    expected_ids = set(target.index)
    for name, matrix in {**matrices, "NEO": neo}.items():
        if set(matrix.index) != expected_ids:
            missing = sorted(expected_ids - set(matrix.index))
            extra = sorted(set(matrix.index) - expected_ids)
            raise ValueError(
                f"{name}与原始人格被试无法一一对齐：缺少{missing[:3]}，多出{extra[:3]}"
            )
        matrix.sort_index(inplace=True)
    source.sort_index(inplace=True)
    target = source["E2_sum"].rename("source_E2_sum")

    rows: list[dict[str, Any]] = []
    for method, matrix in matrices.items():
        rows.append(
            _metric_row(
                f"SJT-{method}-总分",
                "total_score",
                matrix.sum(axis=1).to_frame(f"{method}_total"),
                target,
            )
        )
        rows.append(
            _metric_row(f"SJT-{method}-16题", "item_pattern", matrix, target)
        )
    rows.append(
        _metric_row("NEO-FFI-E总分", "total_score", neo[["E"]], target)
    )
    rows.append(
        _metric_row("NEO-FFI五维", "domain_pattern", neo[["E", "N", "O", "A", "C"]], target)
    )
    for method, matrix in matrices.items():
        combined = matrix.join(
            neo[["E", "N", "O", "A", "C"]].add_prefix("NEO_"),
            how="inner",
        )
        rows.append(
            _metric_row(
                f"SJT-{method}-16题+NEO五维",
                "combined_pattern",
                combined,
                target,
            )
        )

    metrics = pd.DataFrame(rows)
    scored_columns = [f"{item_id}_scored" for item_id in E2_ITEM_DIRECTIONS]
    source_alpha = _cronbach_alpha(source[scored_columns])
    summary: dict[str, Any] = {
        "status": "complete",
        "respondent_count": int(len(source)),
        "source_e2_item_count": len(E2_ITEM_DIRECTIONS),
        "source_e2_mean": float(target.mean()),
        "source_e2_sd": float(target.std(ddof=1)),
        "source_e2_min": float(target.min()),
        "source_e2_max": float(target.max()),
        "source_e2_alpha": source_alpha,
        "source_score_range": [8, 40],
        "scoring_key_status": "explicit_semantic_direction_requires_manual_verification",
        "primary_comparison": "SJT A/B/C item-pattern five-fold CV R2 against source E2 sum",
        "interpretation_boundary": (
            "Internal source-trait transmission/recovery in embodied virtual respondents; "
            "not evidence of validity in human respondents."
        ),
    }

    source.reset_index().to_csv(
        output / "source_e2_scores.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(output / "recovery_metrics.csv", index=False, encoding="utf-8-sig")
    aligned = pd.DataFrame(index=target.index)
    aligned["source_E2_sum"] = target
    aligned["source_E2_mean"] = source["E2_mean"]
    aligned["NEO_E"] = neo["E"]
    for method, matrix in matrices.items():
        aligned[f"SJT_{method}_total"] = matrix.sum(axis=1)
    aligned.reset_index().to_csv(
        output / "aligned_totals.csv", index=False, encoding="utf-8-sig"
    )
    _write_json(
        output / "config.json",
        {
            "experiment": str(experiment),
            "source_pool": str(config.source_pool.resolve()),
            "evaluation": str(evaluation.resolve()),
            "output": str(output.resolve()),
            "e2_item_directions": E2_ITEM_DIRECTIONS,
            "model_calls": 0,
        },
    )
    _write_json(output / "result.json", {"summary": summary, "metrics": rows})
    report = _render_report(output, summary, metrics, wording)
    summary["report"] = str(report.resolve())
    return output, summary

