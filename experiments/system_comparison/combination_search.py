"""Exhaustive, offline selection within a frozen blueprint (no LLM calls)."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from itertools import combinations, islice, product
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sjt_system.evaluation.form_metrics import (
    FORM_EFFECT_EXTREME_FRACTION, TARGET_RECOVERY_DEFAULT_FOLDS,
    TARGET_RECOVERY_RIDGE_PENALTY, _stable_fold_ids,
)
from .combination_data import content_hash, load_search_data, read_json
from .storage import write_csv, write_json


@dataclass(frozen=True)
class SearchConfig:
    sample_role: str = "evaluation"
    objective: str = "neo"
    bank_path: str = "shared/initial_bank.json"
    allow_partial_bank: bool = False
    alpha_min: float = 0.80
    icc_min: float = 0.80
    max_non_target_rho: float | None = None
    target_domain: str = "E"
    batch_size: int = 512
    max_combinations: int = 1_000_000
    top_k: int = 20

    def validate(self):
        if self.objective not in ("neo", "q") or self.target_domain not in tuple("ENOAC"):
            raise ValueError("不支持的排序指标/目标维度")
        if not 0 <= self.alpha_min <= 1 or not 0 <= self.icc_min <= 1:
            raise ValueError("信度门槛必须在[0,1]内")
        if self.max_non_target_rho is not None and not 0 <= self.max_non_target_rho <= 1:
            raise ValueError("非目标相关上限必须在[0,1]内")
        if min(self.batch_size, self.max_combinations, self.top_k) < 1:
            raise ValueError("批量大小、组合上限和top-k必须为正整数")


def blueprint_groups(data, *, available_only):
    available = set(data.audit["item_sources"])
    known_cells = {cell["cell_id"] for cell in data.blueprint["cells"]}
    if len(known_cells) != len(data.blueprint["cells"]):
        raise ValueError("蓝图单元ID重复")
    if any(item.get("blueprint_cell_id") not in known_cells for item in data.bank):
        raise ValueError("候选题引用了未知蓝图单元")
    choices, counts = [], []
    for cell in data.blueprint["cells"]:
        keep = cell["planned_retention_count"]
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("蓝图保留量必须为正整数")
        indices = [j for j, item in enumerate(data.bank)
                   if item["blueprint_cell_id"] == cell["cell_id"]
                   and (not available_only or item["item_id"] in available)]
        if len(indices) < keep:
            raise ValueError(f"蓝图单元候选不足：{cell['cell_id']}")
        counts.append({"cell_id": cell["cell_id"], "available": len(indices), "retain": keep})
        choices.append(list(combinations(indices, keep)))
    return choices, counts


def combination_indices(groups):
    for selected in product(*groups):
        yield tuple(j for group in selected for j in group)


def normalized_scenario(item):
    return "".join(str(item.get("scenario", "")).split()).casefold()


def duplicate_pairs(bank):
    # A deterministic content constraint, not a substitute for semantic/SME review.
    return [(i, j) for i, j in combinations(range(len(bank)), 2)
            if normalized_scenario(bank[i]) == normalized_scenario(bank[j])]


def row_correlations(totals, criterion):
    """Spearman with average tied ranks; undefined values remain NaN."""
    result = np.full(len(totals), np.nan)
    if len(criterion) < 3 or not np.isfinite(criterion).all() or np.ptp(criterion) == 0:
        return result
    valid = np.isfinite(totals).all(axis=1)
    if not valid.any():
        return result
    x = rankdata(totals[valid], axis=1, method="average")
    y = rankdata(criterion, method="average")
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean()
    denominator = np.sqrt(np.square(x).sum(axis=1) * np.square(y).sum())
    result[valid] = np.divide(x @ y, denominator, out=np.full(len(x), np.nan), where=denominator > 0)
    return result


def batch_icc(first, second):
    """Vectorized existing ICC(A,1), retaining negative values."""
    values = np.stack([first, second], axis=2)
    n = first.shape[1]
    grand = values.mean(axis=(1, 2))
    subjects = values.mean(axis=2)
    administrations = values.mean(axis=1)
    ms_subject = 2 * np.square(subjects - grand[:, None]).sum(axis=1) / (n - 1)
    ms_admin = n * np.square(administrations - grand[:, None]).sum(axis=1)
    residual = values - subjects[:, :, None] - administrations[:, None, :] + grand[:, None, None]
    ms_error = np.square(residual).sum(axis=(1, 2)) / (n - 1)
    denominator = ms_subject + ms_error + 2 * (ms_admin - ms_error) / n
    return np.divide(ms_subject - ms_error, denominator, out=np.full(len(first), np.nan),
                     where=np.isfinite(denominator) & (denominator > 0))


class BatchMetrics:
    """Exact existing Q, reusing full-bank fold Gram matrices for speed."""
    def __init__(self, data):
        self.data = data
        self.x = data.matrices["target"]
        self.item_variances = np.var(self.x, axis=0, ddof=1)
        self.y = data.active["target"]
        self.y_ss = np.square(self.y - self.y.mean()).sum()
        self.folds = []
        count = min(TARGET_RECOVERY_DEFAULT_FOLDS, len(data.subjects))
        fold_ids = _stable_fold_ids(data.subjects, count)
        # Missing columns remain in the bank for provenance but never enter a candidate.
        x = np.nan_to_num(self.x, nan=0.0)
        for fold in range(count):
            test = fold_ids == fold
            train = ~test
            mean = x[train].mean(axis=0)
            sd = x[train].std(axis=0, ddof=1)
            sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
            z = (x - mean) / sd
            y_mean = self.y[train].mean()
            self.folds.append((test, z[test], z[train].T @ z[train],
                               z[train].T @ (self.y[train] - y_mean), y_mean))
        self.extreme_masks = {}
        for condition, values in data.active.items():
            self.extreme_masks[condition] = (
                values <= np.quantile(values, FORM_EFFECT_EXTREME_FRACTION),
                values >= np.quantile(values, 1 - FORM_EFFECT_EXTREME_FRACTION))

    def calculate(self, indices):
        indices = np.asarray(indices, dtype=int)
        size, k = indices.shape
        totals = self.x.T[indices].sum(axis=1)
        variance = np.var(totals, axis=1, ddof=1)
        alpha = np.full(size, np.nan)
        if k > 1:
            alpha = np.divide(self.item_variances[indices].sum(axis=1), variance,
                              out=np.full(size, np.nan), where=variance > 0)
            alpha = k / (k - 1) * (1 - alpha)
        repeated = self.data.retest.T[indices].sum(axis=1)
        icc = batch_icc(totals, repeated)
        r2 = np.full(size, np.nan)
        if self.y_ss > 0:
            residual_ss = np.zeros(size)
            for mask, test_z, gram, rhs, y_mean in self.folds:
                matrices = gram[indices[:, :, None], indices[:, None, :]].copy()
                matrices += TARGET_RECOVERY_RIDGE_PENALTY * np.eye(k)[None, :, :]
                coefficients = np.linalg.solve(matrices, rhs[indices][..., None])[..., 0]
                predicted = y_mean + np.einsum("bkn,bk->bn", test_z.T[indices], coefficients)
                residual_ss += np.square(predicted - self.y[mask]).sum(axis=1)
            r2 = 1 - residual_ss / self.y_ss
        effects = {}
        for condition, matrix in self.data.matrices.items():
            low, high = self.extreme_masks[condition]
            form = matrix.T[indices].sum(axis=1)
            if min(low.sum(), high.sum()) >= 2:
                effects[condition] = form[:, high].mean(axis=1) - form[:, low].mean(axis=1)
        selectivity = np.full(size, np.nan)
        if "target" in effects and len(effects) == len(self.data.matrices) and len(effects) > 1:
            signal = np.maximum(0, effects["target"])
            leakage = np.max(np.abs([v for c, v in effects.items() if c != "target"]), axis=0)
            denominator = signal + leakage
            selectivity = np.divide(signal, denominator, out=np.zeros(size), where=denominator > 1e-12)
            selectivity[variance <= 0] = np.nan
        results = {"alpha": alpha, "icc": icc, "target_recovery_r2": r2,
                   "construct_selectivity": selectivity,
                   "q": np.sqrt(np.clip(r2, 0, 1) * np.clip(selectivity, 0, 1))}
        for domain in "ENOAC":
            results[f"neo_{domain}_rho"] = (row_correlations(totals, self.data.reference[domain].to_numpy())
                                           if domain in self.data.reference else np.full(size, np.nan))
        return results


def ranked(frame, objective, config):
    column = f"neo_{config.target_domain}_rho" if objective == "neo" else "q"
    valid = frame[frame["eligible"] & frame[column].notna()]
    # Do not let partial discriminant evidence silently influence tie-breaking.
    return valid.sort_values([column, "combination_id"], ascending=[False, True], kind="stable")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_form(folder, items, metrics):
    write_json(folder / "form.json", {"items": items, "metrics": clean(metrics),
                                     "item_hashes": {i["item_id"]: content_hash(i) for i in items},
                                     "status": "exploratory_candidate_not_formally_qualified"})
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>组合筛选问卷</title>',
             '<style>body{max-width:900px;margin:40px auto;font:17px/1.8 sans-serif;padding:20px}section{margin:30px 0}</style>',
             '<h1>情境问卷</h1><p>请选择最符合您实际行为的一项。</p>']
    for index, item in enumerate(items, 1):
        parts.append(f"<section><b>{index}. {escape(item['scenario'])}</b><p>{escape(item['response_instruction'])}</p>")
        parts.extend(f"<div>{escape(str(o['option_id']))}. {escape(o['text'])}</div>" for o in item["response_options"])
        parts.append("</section>")
    (folder / "form.html").write_text("\n".join(parts) + "</html>", encoding="utf-8")


def write_search_report(folder, summary, leaders, baselines):
    def number(value):
        return "不可估计" if value is None or pd.isna(value) else f"{value:.6f}"

    fields = ["alpha", "icc", "neo_E_rho", "neo_N_rho", "neo_O_rho", "neo_A_rho", "neo_C_rho", "q"]
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>受约束组合筛选</title>',
             '<style>body{font:16px/1.7 system-ui;margin:35px auto;max-width:1200px;padding:20px;color:#18283b}',
             'table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ccd5df;text-align:right}',
             'th:first-child,td:first-child{text-align:left}.note{background:#fff5df;padding:15px}a{color:#165bad}</style>',
             '<h1>冻结题库 · 受约束组合筛选</h1>',
             f"<p>样本：{escape(summary['config']['sample_role'])}；每条件 {summary['subject_count']} 人；"
             f"可用候选 {summary['available_count']}/{summary['candidate_count']}；"
             f"已枚举 {summary['evaluated_combinations']:,}/{summary['theoretical_combinations']:,} 种理论组合。</p>",
             '<div class="note">这是同一批数据上的探索性搜索结果，不是独立验证。Q仍是开发评分，不是信效度。'
             '缺失相关不填0；NEO其他维度不齐全时，不能宣称区分效度合格。'
             '候选沿用冻结题库的既有审题记录，本程序只额外排除完全重复情境，不替代语义/构念审查。</div>',
             f"<p>排序：{escape(summary['ranking_rule'])}</p>",
             '<h2>搜索结果与同数据基线</h2><table><tr><th>问卷</th>' + ''.join(f'<th>{f}</th>' for f in fields) + '</tr>']
    for label, row in [*leaders.items(), *baselines.items()]:
        link = f'<a href="{label}/form.html">{escape(label)}</a>' if label in leaders else escape(label)
        parts.append('<tr><td>' + link + '</td>' + ''.join(f'<td>{number(row.get(f))}</td>' for f in fields) + '</tr>')
    parts.append('</table><h2>解释与限制</h2><ul>')
    parts.extend(f'<li>{escape(w)}</li>' for w in summary["warnings"])
    parts.extend(['</ul><p><a href="combinations.csv">全部组合CSV</a> · <a href="summary.json">完整结果JSON</a> · '
                  '<a href="data_audit.json">数据覆盖与来源审计</a></p>',
                  f"<p>新增模型调用：0；新增Token：0；本地耗时：{summary['wall_seconds']:.2f} 秒。</p>", '</html>'])
    (folder / "report.html").write_text("\n".join(parts), encoding="utf-8")


def run_combination_search(root, config=None, *, progress=print):
    config = config or SearchConfig()
    config.validate()
    root = Path(root).resolve()
    folder = root / "combination_search" / (
        datetime.now().strftime("search_%Y%m%d_%H%M%S_") + config.sample_role + "_" + config.objective + "_" + uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    write_json(folder / "config.json", asdict(config))
    write_json(folder / "progress.json", {"status": "loading", "completed_combinations": 0})
    progress(f"组合筛选目录：{folder}")
    try:
        data = load_search_data(root, config.sample_role, config.bank_path)
        write_json(folder / "data_audit.json", data.audit)
        if data.audit["missing_item_ids"] and not config.allow_partial_bank:
            raise ValueError(f"缺少{len(data.audit['missing_item_ids'])}道原版本作答；见data_audit.json。"
                             "如只探索已覆盖子题库，请显式使用--allow-partial-bank")
        if config.objective == "neo" and config.target_domain not in data.reference:
            raise ValueError("所选样本没有匹配的NEO作答，不能按汇聚相关排序；不会退回Q冒充效度")
        if config.objective == "neo" and data.reference[config.target_domain].nunique() < 2:
            raise ValueError("目标NEO维度没有变异，不能执行汇聚排序")
        all_groups, _ = blueprint_groups(data, available_only=False)
        groups, cell_counts = blueprint_groups(data, available_only=True)
        theoretical = math.prod(len(g) for g in all_groups)
        total = math.prod(len(g) for g in groups)
        if total > config.max_combinations:
            raise ValueError(f"组合数{total:,}超过安全上限{config.max_combinations:,}，请明确增加--max-combinations")
        engine = BatchMetrics(data)
        duplicates = duplicate_pairs(data.bank)
        iterator = combination_indices(groups)
        tables, selected_rows = [], []
        completed = 0
        while batch := list(islice(iterator, config.batch_size)):
            indices = np.asarray(batch)
            table = pd.DataFrame(engine.calculate(indices))
            table.insert(0, "combination_id", np.arange(completed, completed + len(batch)))
            table["content_pass"] = [not any(i in row and j in row for i, j in duplicates) for row in batch]
            table["reliability_pass"] = (table.alpha >= config.alpha_min) & (table.icc >= config.icc_min)
            non_target = [f"neo_{d}_rho" for d in "ENOAC" if d != config.target_domain]
            table["discriminant_complete"] = table[non_target].notna().all(axis=1)
            table["max_abs_non_target_rho"] = table[non_target].abs().max(axis=1, skipna=False)
            table["discriminant_gate"] = pd.Series([None] * len(table), dtype=object)
            table["eligible"] = table.content_pass & table.reliability_pass
            if config.max_non_target_rho is not None:
                table["discriminant_gate"] = (table.discriminant_complete
                                               & (table.max_abs_non_target_rho <= config.max_non_target_rho))
                table["eligible"] &= table.discriminant_gate.astype(bool)
            tables.append(table)
            selected_rows.extend(batch)
            completed += len(batch)
            write_json(folder / "progress.json", {"status": "running", "completed_combinations": completed,
                                                   "total_combinations": total, "wall_seconds": perf_counter() - start})
            if completed == len(batch) or completed == total or completed % (config.batch_size * 8) == 0:
                progress(f"[组合筛选] {completed:,}/{total:,}（{completed / total:.0%}）；仅本地计算")
        frame = pd.concat(tables, ignore_index=True)
        write_csv(folder / "item_index.csv", [{"index": j, "item_id": i["item_id"], "version": i["version"],
                                                "blueprint_cell_id": i["blueprint_cell_id"], "content_hash": content_hash(i)}
                                               for j, i in enumerate(data.bank)])
        frame["selected_indices"] = [",".join(map(str, row)) for row in selected_rows]
        frame.to_csv(folder / "combinations.csv", index=False, encoding="utf-8-sig")
        leaders = {}
        for objective in ("neo", "q"):
            ordered = ranked(frame, objective, config)
            ordered.head(config.top_k).to_csv(folder / f"top_{objective}.csv", index=False, encoding="utf-8-sig")
            if not ordered.empty:
                row = clean(ordered.iloc[0].to_dict())
                leaders[f"best_{objective}"] = row
                items = [data.bank[j] for j in selected_rows[int(row["combination_id"])]]
                save_form(folder / f"best_{objective}", items, row)
        baselines = {}
        index_by_hash = {content_hash(item): j for j, item in enumerate(data.bank)}
        for key in ("B/round_01", "C/round_01"):
            form_file = root / key / "form.json"
            if form_file.exists():
                items = read_json(form_file)["items"]
                if all(content_hash(i) in index_by_hash for i in items):
                    idx = [index_by_hash[content_hash(i)] for i in items]
                    if all(i["item_id"] in data.audit["item_sources"] for i in items):
                        baselines[key] = {k: clean(v[0]) for k, v in engine.calculate([idx]).items()}
        warnings = ["搜索和排名使用同一批样本，仅是探索性可行性摸底，不提供显著性结论。",
                    "信度门槛是本次搜索配置，不是适用于所有人格测验的统一行业标准。",
                    "Q为完整5折岭回归R²与构念选择性的几何平均；不是批量近似Q。",
                    "重测ICC使用首次与第二次逐题作答合成；不得解释为真人重测信度。",
                    "不依据搜索结果改计分键，也不改变原实验的正式资格/最终选卷。"]
        if data.audit["missing_item_ids"]:
            warnings.append(f"缺少{len(data.audit['missing_item_ids'])}道原版本评估，结果只在已覆盖子题库内穷举；不是完整32题库的最优。")
        if not frame.discriminant_complete.all():
            warnings.append("部分或全部非目标NEO相关不可估计；未设置非目标上限时，neo排名仅优化汇聚，不能称为综合信效度最优。")
        if data.audit["reference"].get("cross_model_exploration"):
            warnings.append("SJT与NEO由不同模型作答，本次延用原追加分析，属于混合模型探索。")
        if not data.audit["reference"]["available"]:
            warnings.append("所选样本没有实际NEO作答；本次Q筛选不构成汇聚或区分效度筛选。")
        rule = (f"alpha>={config.alpha_min}且ICC>={config.icc_min}，排除完全重复情境；"
                + (f"最大化NEO-{config.target_domain}的Spearman相关" if config.objective == "neo" else "最大化原始定义Q")
                + (f"；要求四个非目标相关齐全且最大绝对值<={config.max_non_target_rho}"
                   if config.max_non_target_rho is not None else "；不对缺失区分相关进行奖励或补零"))
        summary = {"schema_version": 1, "status": "completed" if f"best_{config.objective}" in leaders else "no_eligible_form",
                   "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
                   "candidate_count": len(data.bank), "available_count": data.audit["available_count"],
                   "subject_count": len(data.subjects), "theoretical_combinations": theoretical,
                   "evaluated_combinations": total, "content_valid_combinations": int(frame.content_pass.sum()),
                   "eligible_combinations": int(frame.eligible.sum()), "cell_counts": cell_counts,
                   "ranking_rule": rule, "leaders": leaders, "baselines": baselines, "warnings": warnings,
                   "ranges": {c: {"min": clean(frame[c].min()),
                                  "median": clean(frame[c].median()) if frame[c].notna().any() else None,
                                  "max": clean(frame[c].max())} for c in ("alpha", "icc", "q", f"neo_{config.target_domain}_rho")},
                   "wall_seconds": perf_counter() - start, "additional_model_calls": 0, "additional_tokens": 0}
        write_json(folder / "summary.json", clean(summary))
        write_json(folder / "cost.json", {"wall_seconds": summary["wall_seconds"], "model_calls": 0, "tokens": 0})
        write_json(folder / "progress.json", {"status": summary["status"], "completed_combinations": total})
        write_search_report(folder, summary, leaders, baselines)
        progress(f"筛选完成：{summary['status']}；报告：{folder / 'report.html'}")
        return folder, summary
    except BaseException as exc:
        write_json(folder / "progress.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "blocked",
                                               "error": str(exc), "wall_seconds": perf_counter() - start})
        raise
