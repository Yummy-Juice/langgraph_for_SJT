"""Offline negative-control diagnostics for the legacy A/B/C evaluation.

The diagnostics answer a narrow question: are the unusually high alpha/CITC
and SJT--NEO correlations caused by a broken calculation/alignment pipeline,
or are they properties of the virtual-response data?  The source responses are
never modified and no model is invoked.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats

from .storage import write_csv, write_json


METHODS = ("A", "B", "C")
NEO_DOMAINS = ("E", "N", "O", "A", "C")
OPTION_IDS = ("A", "B", "C", "D")
CONTROL_TYPES = (
    "participant_id_permutation",
    "within_item_permutation",
    "random_miskey",
    "half_reverse_miskey",
)
SCHEMA_VERSION = 1
FORMULA_VERSION = "legacy-abc-negative-controls-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}第{line_number}行不是有效JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是对象")
            records.append(value)
    return records


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    return _number(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    return _number(stats.spearmanr(left, right, nan_policy="omit").statistic)


def _alpha(matrix: np.ndarray) -> float | None:
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    total_variance = float(np.var(matrix.sum(axis=1), ddof=1))
    if not math.isfinite(total_variance) or total_variance <= 0:
        return None
    item_variance = float(np.var(matrix, axis=0, ddof=1).sum())
    value = matrix.shape[1] / (matrix.shape[1] - 1) * (
        1.0 - item_variance / total_variance
    )
    return _number(value)


def calculate_metrics(matrix: np.ndarray, neo: np.ndarray) -> dict[str, Any]:
    """Calculate only the metrics needed to check negative-control sensitivity."""

    matrix = np.asarray(matrix, dtype=float)
    neo = np.asarray(neo, dtype=float)
    if matrix.ndim != 2 or neo.ndim != 2 or neo.shape[1] != len(NEO_DOMAINS):
        raise ValueError("计分矩阵或NEO矩阵维度无效")
    if matrix.shape[0] != neo.shape[0] or not np.isfinite(matrix).all() or not np.isfinite(neo).all():
        raise ValueError("SJT与NEO必须是同一批完整有限数值")
    total = matrix.sum(axis=1)
    citcs = []
    for column in range(matrix.shape[1]):
        value = _pearson(matrix[:, column], total - matrix[:, column])
        if value is not None:
            citcs.append(value)
    result: dict[str, Any] = {
        "cronbach_alpha": _alpha(matrix),
        "citc_mean": float(np.mean(citcs)) if citcs else None,
        "citc_min": float(np.min(citcs)) if citcs else None,
        "estimable_citc_count": len(citcs),
        "respondent_count": int(matrix.shape[0]),
        "item_count": int(matrix.shape[1]),
    }
    for index, domain in enumerate(NEO_DOMAINS):
        result[f"rho_{domain}"] = _spearman(total, neo[:, index])
    result["abs_rho_E"] = abs(result["rho_E"]) if result["rho_E"] is not None else None
    non_target = [
        abs(result[f"rho_{domain}"])
        for domain in NEO_DOMAINS
        if domain != "E" and result[f"rho_{domain}"] is not None
    ]
    all_domains = [
        abs(result[f"rho_{domain}"])
        for domain in NEO_DOMAINS
        if result[f"rho_{domain}"] is not None
    ]
    result["max_abs_rho_non_target"] = max(non_target) if non_target else None
    result["max_abs_rho_neo"] = max(all_domains) if all_domains else None
    return result


def _validate_form(form: Mapping[str, Any], item_ids: Sequence[str]) -> list[dict[str, Any]]:
    items = form.get("items")
    if not isinstance(items, list) or [str(item.get("item_id")) for item in items] != list(item_ids):
        raise ValueError("form.json题目顺序与score_matrix.csv不一致")
    output = []
    for item in items:
        key = item.get("scoring_key")
        if not isinstance(key, Mapping) or set(map(str, key)) != set(OPTION_IDS):
            raise ValueError(f"题目{item.get('item_id')}缺少A-D计分键")
        scores = {option: _number(key.get(option)) for option in OPTION_IDS}
        if any(value is None for value in scores.values()) or len(set(scores.values())) != 4:
            raise ValueError(f"题目{item.get('item_id')}必须有四个不同且有效的分值")
        output.append({**dict(item), "scoring_key": scores})
    return output


def load_source(evaluation_root: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root).resolve()
    participants_path = root / "participants.json"
    neo_path = root / "neo_ffi" / "scores.csv"
    summary_path = root / "summary.json"
    required = [participants_path, neo_path, summary_path]
    for method in METHODS:
        required.extend([
            root / method / "form.json",
            root / method / "evaluation" / "score_matrix.csv",
            root / method / "evaluation" / "responses.jsonl",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("负对照缺少输入文件：" + "；".join(missing))

    participants = json.loads(participants_path.read_text(encoding="utf-8"))
    subject_ids = [str(value) for value in participants.get("respondent_ids", [])]
    if len(subject_ids) < 3 or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("participants.json中的被试ID不足或重复")

    neo_frame = pd.read_csv(neo_path)
    if "respondent_id" not in neo_frame or not set(NEO_DOMAINS) <= set(neo_frame.columns):
        raise ValueError("NEO得分文件缺少respondent_id或五个维度")
    neo_frame["respondent_id"] = neo_frame["respondent_id"].astype(str)
    neo_frame = neo_frame.set_index("respondent_id").reindex(subject_ids)
    if neo_frame[list(NEO_DOMAINS)].isna().any().any() or set(neo_frame.index) != set(subject_ids):
        raise ValueError("NEO得分不能与完整被试名单对齐")

    methods: dict[str, Any] = {}
    hashes = {str(path.relative_to(root)): _sha256(path) for path in required}
    for method in METHODS:
        matrix_path = root / method / "evaluation" / "score_matrix.csv"
        response_path = root / method / "evaluation" / "responses.jsonl"
        form_path = root / method / "form.json"
        frame = pd.read_csv(matrix_path)
        if "respondent_id" not in frame:
            raise ValueError(f"{method}计分矩阵缺少respondent_id")
        frame["respondent_id"] = frame["respondent_id"].astype(str)
        frame = frame.set_index("respondent_id").reindex(subject_ids)
        if frame.isna().any().any():
            raise ValueError(f"{method}计分矩阵存在缺失或被试不能对齐")
        item_ids = [str(column) for column in frame.columns]
        form = json.loads(form_path.read_text(encoding="utf-8"))
        items = _validate_form(form, item_ids)
        records = _jsonl(response_path)
        response_frame = pd.DataFrame(records)
        required_columns = {"respondent_id", "item_id", "selected_option_id"}
        if response_frame.empty or not required_columns <= set(response_frame.columns):
            raise ValueError(f"{method}原始选项作答不完整")
        response_frame["respondent_id"] = response_frame["respondent_id"].astype(str)
        response_frame["item_id"] = response_frame["item_id"].astype(str)
        response_frame["selected_option_id"] = response_frame["selected_option_id"].astype(str)
        choices = response_frame.pivot(
            index="respondent_id", columns="item_id", values="selected_option_id"
        ).reindex(index=subject_ids, columns=item_ids)
        if choices.isna().any().any() or not choices.isin(OPTION_IDS).all().all():
            raise ValueError(f"{method}选项作答存在缺失、重复或非法选项")
        methods[method] = {
            "matrix": frame.to_numpy(dtype=float),
            "choices": choices.to_numpy(dtype=str),
            "item_ids": item_ids,
            "items": items,
        }
    return {
        "root": root,
        "subject_ids": subject_ids,
        "neo": neo_frame[list(NEO_DOMAINS)].to_numpy(dtype=float),
        "methods": methods,
        "source_hashes": hashes,
    }


def _rescore_random_keys(
    choices: np.ndarray,
    items: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.empty(choices.shape, dtype=float)
    for column, item in enumerate(items):
        original = np.array([float(item["scoring_key"][option]) for option in OPTION_IDS])
        replacement = rng.permutation(original)
        mapping = dict(zip(OPTION_IDS, replacement))
        matrix[:, column] = np.fromiter(
            (mapping[value] for value in choices[:, column]),
            dtype=float,
            count=choices.shape[0],
        )
    return matrix


def _half_reverse(
    matrix: np.ndarray,
    items: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    output = matrix.copy()
    selected = sorted(rng.choice(matrix.shape[1], matrix.shape[1] // 2, replace=False).tolist())
    for column in selected:
        scores = [float(value) for value in items[column]["scoring_key"].values()]
        output[:, column] = min(scores) + max(scores) - output[:, column]
    return output, selected


def generate_control_rows(
    source: Mapping[str, Any],
    *,
    replications: int,
    seed: int,
) -> list[dict[str, Any]]:
    if replications < 100:
        raise ValueError("replications至少为100，避免负对照分布过于不稳定")
    rows: list[dict[str, Any]] = []
    neo = np.asarray(source["neo"], dtype=float)
    for method_index, method in enumerate(METHODS):
        data = source["methods"][method]
        matrix = np.asarray(data["matrix"], dtype=float)
        choices = np.asarray(data["choices"], dtype=str)
        items = data["items"]
        baseline = calculate_metrics(matrix, neo)
        rows.append({
            "method": method,
            "control": "baseline",
            "replicate": 0,
            "seed": seed,
            "source_item_id": None,
            "changed_item_indices": None,
            **baseline,
        })
        method_seed = np.random.SeedSequence([seed, method_index])
        child_seeds = method_seed.spawn(replications)
        for replicate, child_seed in enumerate(child_seeds, 1):
            rng = np.random.default_rng(child_seed)

            neo_permuted = neo[rng.permutation(neo.shape[0]), :]
            rows.append({
                "method": method,
                "control": "participant_id_permutation",
                "replicate": replicate,
                "seed": seed,
                "source_item_id": None,
                "changed_item_indices": None,
                **calculate_metrics(matrix, neo_permuted),
            })

            shuffled = np.column_stack([
                matrix[rng.permutation(matrix.shape[0]), column]
                for column in range(matrix.shape[1])
            ])
            rows.append({
                "method": method,
                "control": "within_item_permutation",
                "replicate": replicate,
                "seed": seed,
                "source_item_id": None,
                "changed_item_indices": list(range(matrix.shape[1])),
                **calculate_metrics(shuffled, neo),
            })

            miskeyed = _rescore_random_keys(choices, items, rng)
            rows.append({
                "method": method,
                "control": "random_miskey",
                "replicate": replicate,
                "seed": seed,
                "source_item_id": None,
                "changed_item_indices": list(range(matrix.shape[1])),
                **calculate_metrics(miskeyed, neo),
            })

            half_reversed, reversed_indices = _half_reverse(matrix, items, rng)
            rows.append({
                "method": method,
                "control": "half_reverse_miskey",
                "replicate": replicate,
                "seed": seed,
                "source_item_id": None,
                "changed_item_indices": reversed_indices,
                **calculate_metrics(half_reversed, neo),
            })

        # Repeat each single item sixteen times. This deliberately destroys
        # content coverage while demonstrating that alpha/CITC can reach 1.
        for column, item_id in enumerate(data["item_ids"], 1):
            duplicated = np.repeat(matrix[:, [column - 1]], matrix.shape[1], axis=1)
            rows.append({
                "method": method,
                "control": "single_item_duplicated_16x",
                "replicate": column,
                "seed": seed,
                "source_item_id": item_id,
                "changed_item_indices": list(range(matrix.shape[1])),
                **calculate_metrics(duplicated, neo),
            })
    return rows


def _quantile(values: Sequence[float], probability: float) -> float | None:
    clean = [float(value) for value in values if _number(value) is not None]
    return float(np.quantile(clean, probability)) if clean else None


def summarize_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metrics = (
        "cronbach_alpha", "citc_mean", "citc_min", "rho_E", "abs_rho_E",
        "rho_N", "rho_O", "rho_A", "rho_C", "max_abs_rho_non_target",
        "max_abs_rho_neo",
    )
    output: list[dict[str, Any]] = []
    for (method, control), group in frame.groupby(["method", "control"], sort=False):
        row: dict[str, Any] = {
            "method": method,
            "control": control,
            "replications": int(group.shape[0]),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_median"] = float(np.median(values)) if values else None
            row[f"{metric}_q025"] = _quantile(values, 0.025)
            row[f"{metric}_q975"] = _quantile(values, 0.975)
        output.append(row)
    return output


def _summary_index(summary: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["method"]), str(row["control"])): row for row in summary}


def _at_most(value: Any, maximum: float) -> bool:
    numeric = _number(value)
    return numeric is not None and numeric <= maximum


def _at_least(value: Any, minimum: float) -> bool:
    numeric = _number(value)
    return numeric is not None and numeric >= minimum


def diagnostic_checks(
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = _summary_index(summary)
    frame = pd.DataFrame(rows)
    checks = []
    for method in METHODS:
        baseline = index[(method, "baseline")]
        id_null = index[(method, "participant_id_permutation")]
        item_null = index[(method, "within_item_permutation")]
        miskey = index[(method, "random_miskey")]
        duplicate = index[(method, "single_item_duplicated_16x")]
        baseline_rho = _number(baseline["abs_rho_E_mean"])
        method_checks = [
            {
                "method": method,
                "check": "respondent_alignment_sensitivity",
                "passes": (
                    _at_most(id_null["abs_rho_E_mean"], 0.15)
                    and _at_most(id_null["max_abs_rho_neo_mean"], 0.25)
                ),
                "observed": {
                    "mean_abs_rho_E": id_null["abs_rho_E_mean"],
                    "mean_max_abs_rho_all_NEO": id_null["max_abs_rho_neo_mean"],
                },
                "criterion": "被试ID随机错配后，平均|rho_E| <= 0.15且五维最大|rho|均值 <= 0.25",
            },
            {
                "method": method,
                "check": "person_structure_sensitivity",
                "passes": (
                    _at_most(item_null["cronbach_alpha_mean"], 0.30)
                    and _at_most(item_null["abs_rho_E_mean"], 0.15)
                ),
                "observed": {
                    "alpha": item_null["cronbach_alpha_mean"],
                    "abs_rho_E": item_null["abs_rho_E_mean"],
                },
                "criterion": "题内独立打乱后，平均alpha <= 0.30且平均|rho_E| <= 0.15",
            },
            {
                "method": method,
                "check": "scoring_key_sensitivity",
                "passes": (
                    baseline_rho is not None
                    and _at_most(miskey["abs_rho_E_mean"], max(0.15, baseline_rho - 0.30))
                ),
                "observed": {
                    "baseline_abs_rho_E": baseline_rho,
                    "random_miskey_abs_rho_E": miskey["abs_rho_E_mean"],
                },
                "criterion": "随机错配各题计分键后，平均|rho_E|至少下降0.30，或降至0.15以内",
            },
            {
                "method": method,
                "check": "alpha_redundancy_demonstration",
                "passes": _at_least(duplicate["cronbach_alpha_mean"], 0.99),
                "observed": duplicate["cronbach_alpha_mean"],
                "criterion": "单题复制16次时alpha >= 0.99，证明alpha可被内容重复抬高",
            },
        ]
        checks.extend(method_checks)

        id_rows = frame[(frame["method"] == method) & (frame["control"] == "participant_id_permutation")]
        if baseline_rho is None:
            empirical_p = None
        else:
            exceedances = int((id_rows["abs_rho_E"] >= baseline_rho).sum())
            empirical_p = (exceedances + 1) / (len(id_rows) + 1)
        checks.append({
            "method": method,
            "check": "permutation_association_test",
            "passes": empirical_p is not None,
            "observed": empirical_p,
            "criterion": "被试错配置换检验的经验双侧p；仅用于确认原关联不是随机配对产生",
        })

    core = [
        check for check in checks
        if check["check"] in {
            "respondent_alignment_sensitivity",
            "person_structure_sensitivity",
            "scoring_key_sensitivity",
        }
    ]
    if all(check["passes"] for check in core):
        verdict = "calculation_sensitive_virtual_data_ceiling"
        interpretation = (
            "负对照按预期破坏指标，未发现明显的计算或被试对齐故障；"
            "原始高值主要来自虚拟作答数据的强一致结构。"
        )
    else:
        verdict = "pipeline_investigation_required"
        failed = [f"{check['method']}/{check['check']}" for check in core if not check["passes"]]
        interpretation = "至少一项负对照未按预期下降，应先检查计分、被试对齐或数据泄漏：" + "、".join(failed)
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "checks": checks,
        "threshold_note": "这些阈值是诊断启发式规则，不是通用心理测量合格线。",
    }


def _fmt(value: Any) -> str:
    numeric = _number(value)
    return f"{numeric:.3f}" if numeric is not None else "不可估计"


def render_report(
    path: Path,
    *,
    source: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
    replications: int,
    seed: int,
) -> None:
    index = _summary_index(summary)
    labels = {
        "baseline": "原始数据",
        "participant_id_permutation": "错配被试ID",
        "within_item_permutation": "题内独立打乱",
        "random_miskey": "随机错配计分键",
        "half_reverse_miskey": "随机反向一半题目",
        "single_item_duplicated_16x": "单题复制16次",
    }
    rows = []
    for method in METHODS:
        for control in ("baseline", *CONTROL_TYPES, "single_item_duplicated_16x"):
            item = index[(method, control)]
            rows.append(
                "<tr>"
                f"<td>{method}</td><td>{escape(labels[control])}</td>"
                f"<td>{_fmt(item['cronbach_alpha_mean'])}</td>"
                f"<td>{_fmt(item['citc_mean_mean'])}</td>"
                f"<td>{_fmt(item['rho_E_mean'])}</td>"
                f"<td>{_fmt(item['abs_rho_E_mean'])}</td>"
                f"<td>{_fmt(item['max_abs_rho_non_target_mean'])}</td>"
                f"<td>[{_fmt(item['rho_E_q025'])}, {_fmt(item['rho_E_q975'])}]</td>"
                "</tr>"
            )
    check_rows = []
    for check in diagnosis["checks"]:
        observed = check["observed"]
        observed_text = json.dumps(observed, ensure_ascii=False) if isinstance(observed, Mapping) else _fmt(observed)
        check_rows.append(
            f"<tr><td>{check['method']}</td><td>{escape(check['check'])}</td>"
            f"<td>{'通过' if check['passes'] else '未通过'}</td>"
            f"<td>{escape(observed_text)}</td><td>{escape(check['criterion'])}</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>虚拟被试信效度负对照诊断</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 20px;font:15px/1.65 sans-serif;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #d1d5db;padding:7px;vertical-align:top}}th{{background:#e8eef8}}code{{background:#f3f4f6;padding:2px 4px}}.lead{{font-size:17px}}.warn{{color:#8a4b08}}</style>
<h1>虚拟被试信效度负对照诊断</h1>
<p class="lead">结论：{escape(str(diagnosis['interpretation']))}</p>
<p>来源：<code>{escape(str(source['root']))}</code>；被试={len(source['subject_ids'])}；每类随机负对照重复={replications}；seed={seed}。全程复用已保存作答，不调用模型。</p>
<h2>负对照结果</h2>
<table><thead><tr><th>方法</th><th>条件</th><th>alpha均值</th><th>CITC均值</th><th>rho E均值</th><th>|rho E|均值</th><th>最大非目标|rho|均值</th><th>rho E 95%模拟区间</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="warn">整体反转所有题目的计分方向不会降低相关绝对值，只会改变正负号，因此本实验采用随机错配计分键和随机反向一半题目作为有效的评分键破坏条件。</p>
<h2>诊断检查</h2>
<table><thead><tr><th>方法</th><th>检查</th><th>状态</th><th>观察值</th><th>诊断规则</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table>
<h2>解释边界</h2>
<p>通过负对照只能说明计算实现能够识别被试错配、作答结构破坏和计分键破坏。它不能证明虚拟信效度能够代表真人信效度。单题复制条件用于展示alpha和CITC可能被内容重复人为抬高。</p>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_negative_controls(
    evaluation_root: str | Path,
    *,
    replications: int = 500,
    seed: int = 20260913,
    output_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    source = load_source(evaluation_root)
    if replications < 100 or replications > 10000:
        raise ValueError("replications必须在100至10000之间")
    if output_root is None:
        output = source["root"] / "diagnostics" / (
            "negative_control_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        )
    else:
        output = Path(output_root).resolve()
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise ValueError("指定输出目录必须是空目录，避免覆盖既有诊断结果")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    config = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "source": str(source["root"]),
        "replications": replications,
        "seed": seed,
        "controls": [*CONTROL_TYPES, "single_item_duplicated_16x"],
        "model_calls": 0,
        "source_hashes": source["source_hashes"],
        "started_at": started.isoformat(),
    }
    write_json(output / "config.json", config)
    rows = generate_control_rows(source, replications=replications, seed=seed)
    summary = summarize_controls(rows)
    diagnosis = diagnostic_checks(rows, summary)
    changed_sources = [
        relative
        for relative, expected in source["source_hashes"].items()
        if _sha256(source["root"] / relative) != expected
    ]
    if changed_sources:
        raise RuntimeError("诊断过程中来源文件发生变化：" + "；".join(changed_sources))
    write_csv(output / "control_replications.csv", rows)
    write_csv(output / "control_summary.csv", summary)
    write_csv(output / "diagnostic_checks.csv", diagnosis["checks"])
    result = {
        "status": "complete",
        "verification_status": "ANALYZED",
        "source": str(source["root"]),
        "respondent_count": len(source["subject_ids"]),
        "replications": replications,
        "seed": seed,
        "model_calls": 0,
        "source_integrity_verified": True,
        "verdict": diagnosis["verdict"],
        "interpretation": diagnosis["interpretation"],
        "threshold_note": diagnosis["threshold_note"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "result.json", result)
    render_report(
        output / "report.html",
        source=source,
        summary=summary,
        diagnosis=diagnosis,
        replications=replications,
        seed=seed,
    )
    return output, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线检验虚拟信效度指标对破坏性负对照是否敏感")
    parser.add_argument("--evaluation", type=Path, required=True, help="legacy_pool_abc结果目录")
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, result = run_negative_controls(
        args.evaluation,
        replications=args.replications,
        seed=args.seed,
        output_root=args.output,
    )
    print(f"诊断输出：{output}")
    print(f"结论：{result['interpretation']}")
    print(f"报告：{output / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
