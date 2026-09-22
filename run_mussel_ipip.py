"""One-command launcher for the fresh Mussel × IPIP virtual-respondent run.

The launcher intentionally has no command-line arguments. It selects an
experiment directory interactively, then the experiment itself displays the
available virtual-respondent prompt variants.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import sys

from experiments.system_comparison.virtual_prompt_registry import (
    DEFAULT_PROMPT_ROOT,
    format_prompt_list,
    load_prompt_specs,
)
from sjt_system.authoring.construct_registry import construct_selection_catalog
from sjt_system.evaluation.respondents import build_score_dimension_catalog, normalize_matched_conditions


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DATA_ROOT = PROJECT_ROOT / "experiment_data"


def _candidate_experiments() -> list[Path]:
    if not EXPERIMENT_DATA_ROOT.is_dir():
        return []
    candidates = [
        path
        for path in EXPERIMENT_DATA_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("exp_")
    ]
    if not candidates:
        candidates = [path for path in EXPERIMENT_DATA_ROOT.iterdir() if path.is_dir()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _choose_experiment() -> Path:
    candidates = _candidate_experiments()
    if not candidates:
        raise FileNotFoundError(
            f"没有找到实验目录：{EXPERIMENT_DATA_ROOT}\n"
            "请先运行一次主实验，或手动创建 experiment_data/exp_... 目录。"
        )
    selected = candidates[0]
    # The source experiment is an implementation detail.  Do not expose its
    # directory name or make the user choose it before selecting the method
    # and questionnaire.  The newest source is kept for the current launcher;
    # the legacy CLI remains available when an old run must be selected.
    return selected


def _show_questionnaire_scope() -> None:
    print("\n当前可用的问卷与题本：", flush=True)
    print("1. Mussel：110道情境判断题", flush=True)
    print("2. IPIP-NEO：5个对应facet，作为外部人格参照", flush=True)
    print("3. A/B/C：三套已冻结的16道SJT问卷", flush=True)
    print(
        "当前这个启动器先执行 Mussel + IPIP；A/B/C仍使用独立入口，接入统一选择后再开放。",
        flush=True,
    )


def _choose_respondent_count() -> int:
    print("\n请选择虚拟被试数量：", flush=True)
    print("1. 30 人（快速检查）", flush=True)
    print("2. 100 人（小规模实验）", flush=True)
    print("3. 285 人（完整实验）", flush=True)
    print("4. 自定义人数", flush=True)
    while True:
        raw = input("你的选择，直接回车使用285人：").strip()
        if not raw:
            return 285
        if raw in {"1", "2", "3"}:
            return {"1": 30, "2": 100, "3": 285}[raw]
        if raw == "4":
            custom = input("请输入虚拟被试人数：").strip()
            try:
                count = int(custom)
            except ValueError:
                count = 0
            if count >= 3:
                return count
        else:
            try:
                count = int(raw)
            except ValueError:
                count = 0
            if count >= 3:
                return count
        print("输入无效，请选择1-4，或直接输入不少于3的整数。", flush=True)


def _read_float(text: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    while True:
        raw = input(f"{text}（默认{default:g}）：").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            value = float("nan")
        if math.isfinite(value) and (minimum is None or value > minimum) and (maximum is None or value < maximum):
            return value
        print("请输入有效数值。", flush=True)


def _choose_score_settings(prompt_id: str) -> tuple[float, float, int]:
    spec = load_prompt_specs(DEFAULT_PROMPT_ROOT)[prompt_id]
    if spec.persona_input not in {"score_values", "five_mussel_facets"}:
        return 50.0, 15.0, 7
    print("\n===== 设置人格分数 =====", flush=True)
    print(
        "本次设置沿用主流程的三臂匹配规则：target、同域非目标、跨域非目标。\n"
        "每名被试只接收所属匹配臂的一个facet分数；IPIP仍然测量五个参照facet。",
        flush=True,
    )
    mean_score = _read_float("请输入共享正态分布均值", 50.0, minimum=0.0, maximum=100.0)
    score_sd = _read_float("请输入共享正态分布 SD", 15.0, minimum=0.0)
    # The main CLI does not ask the user for a seed; it uses the fixed
    # protocol default.  Keep this launcher identical.
    return mean_score, score_sd, 7


def _facet_options() -> list[dict[str, object]]:
    return [
        dict(row)
        for row in build_score_dimension_catalog(construct_selection_catalog())
        if row.get("level") == "facet"
    ]


def _choose_facet(label: str, *, allowed: list[dict[str, object]], default: str) -> str:
    print(f"\n请选择{label}：", flush=True)
    for index, row in enumerate(allowed, 1):
        print(
            f"{index}. {row.get('dimension_id')} · {row.get('facet_name')}"
            f"（{row.get('domain_name')}）",
            flush=True,
        )
    while True:
        raw = input(f"你的选择，直接回车使用{default}：").strip()
        if not raw:
            return default
        try:
            index = int(raw)
        except ValueError:
            index = 0
        if 1 <= index <= len(allowed):
            return str(allowed[index - 1]["dimension_id"])
        print(f"输入无效，请输入1-{len(allowed)}。", flush=True)


def _choose_matched_facets() -> tuple[str, str, str]:
    options = _facet_options()
    by_id = {str(row["dimension_id"]): row for row in options}
    target = _choose_facet(
        "目标facet",
        allowed=options,
        default="extraversion_gregariousness",
    )
    target_domain = by_id[target].get("domain_id")
    same_options = [
        row for row in options
        if row.get("dimension_id") != target
        and row.get("domain_id") == target_domain
    ]
    same_default = next(
        (str(row["dimension_id"]) for row in same_options if row["dimension_id"] == "extraversion_warmth"),
        str(same_options[0]["dimension_id"]),
    )
    same = _choose_facet("同域非目标facet", allowed=same_options, default=same_default)
    cross_options = [
        row for row in options
        if row.get("domain_id") != target_domain
        and row.get("dimension_id") not in {target, same}
    ]
    cross_default = next(
        (str(row["dimension_id"]) for row in cross_options if row["dimension_id"] == "neuroticism_anxiety"),
        str(cross_options[0]["dimension_id"]),
    )
    cross = _choose_facet("跨域非目标facet", allowed=cross_options, default=cross_default)
    normalize_matched_conditions(
        [
            {"condition_id": "target", "role": "target", "dimension_id": target},
            {"condition_id": "same_domain", "role": "same_domain_non_target", "dimension_id": same},
            {"condition_id": "cross_domain", "role": "cross_domain_non_target", "dimension_id": cross},
        ],
        dimension_catalog=options,
        target_dimension_id=target,
    )
    return target, same, cross


def _available_prompt_specs():
    specs = load_prompt_specs(DEFAULT_PROMPT_ROOT)
    # The old five-facet-joint input path is intentionally not exposed by this
    # launcher.  Five IPIP facets are still administered; only the persona
    # score-generation method is removed.
    return {
        prompt_id: spec
        for prompt_id, spec in specs.items()
        if spec.persona_input != "five_mussel_facets"
    }


def _choose_prompt_id() -> str:
    specs = _available_prompt_specs()
    print("\n当前可用的虚拟被试方法：", flush=True)
    print(format_prompt_list(specs), flush=True)
    print("N. 新建一种虚拟被试方法", flush=True)
    print("本次运行只选择一种方法；不同方法的结果后续再统一比较。", flush=True)
    while True:
        raw = input("请选择方法编号：").strip()
        if raw.lower() == "n":
            from experiments.system_comparison.create_virtual_prompt import (
                create_prompt_interactively,
            )

            create_prompt_interactively(DEFAULT_PROMPT_ROOT)
            specs = _available_prompt_specs()
            ordered = list(specs.values())
            print("\n新方法已加入，请选择它的编号：", flush=True)
            print(format_prompt_list(specs), flush=True)
            continue
        ordered = list(specs.values())
        try:
            index = int(raw)
        except ValueError:
            index = 0
        if 1 <= index <= len(ordered):
            selected = ordered[index - 1]
            print(f"已选择：{selected.label}", flush=True)
            return selected.prompt_id
        print(f"输入无效，请输入1-{len(ordered)}。", flush=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    experiment = _choose_experiment()
    prompt_id = _choose_prompt_id()
    _show_questionnaire_scope()
    respondents = _choose_respondent_count()
    score_mean, score_sd, score_seed = _choose_score_settings(prompt_id)
    target_facet, same_domain_facet, cross_domain_facet = (
        _choose_matched_facets()
        if load_prompt_specs(DEFAULT_PROMPT_ROOT)[prompt_id].persona_input in {"score_values", "five_mussel_facets"}
        else ("extraversion_gregariousness", "extraversion_warmth", "neuroticism_anxiety")
    )
    from experiments.system_comparison.fresh_mussel_ipip_comparison import (
        FreshComparisonConfig,
        run_fresh_comparison,
    )

    print("\n即将运行全量 Mussel × IPIP 实验。", flush=True)
    print(f"被试数：{respondents}；提示词将在下一步交互选择。", flush=True)
    output, result = asyncio.run(
        run_fresh_comparison(
            FreshComparisonConfig(
                experiment=experiment,
                respondents=respondents,
                prompt_ids=prompt_id,
                score_mean=score_mean,
                score_sd=score_sd,
                score_seed=score_seed,
                target_facet=target_facet,
                same_domain_facet=same_domain_facet,
                cross_domain_facet=cross_domain_facet,
            )
        )
    )
    print(f"\n实验完成：{output}", flush=True)
    print(
        f"被试={result['respondent_count']}；"
        f"预计模型调用={result['expected_model_calls']}",
        flush=True,
    )
    print(f"报告：{result['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
