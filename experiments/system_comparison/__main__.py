"""Run with: python -m experiments.system_comparison --help."""
import argparse
import asyncio
import json
from pathlib import Path

from .config import ExperimentConfig
from .storage import ExperimentStore
from .reporting import write_report
from .virtual_prompt_registry import (
    DEFAULT_PROMPT_ROOT,
    format_prompt_list,
    get_prompt_spec,
    load_prompt_specs,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立单批次A/B/C实验；C保留人工确认")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="创建独立实验并运行（会调用真实模型）")
    run.add_argument("--config", type=Path, help="非敏感配置JSON；省略则使用默认参数")
    run.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2] / "experiment_data")
    for name in ("resume", "evaluate", "report", "neo-ffi"):
        command = sub.add_parser(name)
        command.add_argument("--experiment", type=Path, required=True)
    combine = sub.add_parser("combine", help="仅用已保存作答穷举蓝图组合，不调用模型")
    combine.add_argument("--experiment", type=Path, required=True)
    combine.add_argument("--sample-role", choices=("development", "evaluation"), default="evaluation")
    combine.add_argument("--objective", choices=("neo", "q"), default="neo")
    combine.add_argument("--bank", default="shared/initial_bank.json", help="实验目录内的冻结题库路径")
    combine.add_argument("--allow-partial-bank", action="store_true", help="显式允许只搜索已有完整作答的子题库")
    combine.add_argument("--alpha-min", type=float, default=0.80)
    combine.add_argument("--icc-min", type=float, default=0.80)
    combine.add_argument("--max-non-target-rho", type=float, help="若设定，四个非目标相关必须齐全且满足上限")
    combine.add_argument("--target-domain", choices=tuple("ENOAC"), default="E")
    combine.add_argument("--batch-size", type=int, default=512)
    combine.add_argument("--max-combinations", type=int, default=1_000_000)
    combine.add_argument("--top-k", type=int, default=20)
    prompts = sub.add_parser("virtual-prompts", help="查看和校验可选虚拟被试提示词")
    prompts.add_argument("action", choices=("list", "show", "validate"))
    prompts.add_argument("--prompt-id", default=None)
    prompts.add_argument("--root", type=Path, default=DEFAULT_PROMPT_ROOT)
    legacy = sub.add_parser("legacy-abc", help="用旧项目同一批虚拟被试复测A/B/C并计算CITC、alpha和NEO效度")
    legacy.add_argument("--experiment", type=Path, required=True, help="已冻结A/B/C问卷所在的独立实验目录")
    legacy.add_argument("--legacy-project", type=Path, default=Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT"))
    legacy.add_argument("--model", dest="model_id", default=None, help="省略则使用当前.env的MODEL_ID")
    legacy.add_argument("--neo-ffi", dest="neo_ffi_path", type=Path, default=None)
    legacy.add_argument("--max-concurrency", type=int, default=5)
    legacy.add_argument("--max-retries", type=int, default=2)
    legacy.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    legacy.add_argument("--respondent-limit", type=int, default=None, help="联调时只用来源记录前N名；正式运行省略")
    legacy.add_argument("--output", type=Path, default=None, help="可选；指定同一目录可续跑已落盘作答")
    legacy.add_argument("--persona-mode", choices=("items_plus_summary", "summary_only", "summary_embodied_probability"), default="items_plus_summary")
    legacy.add_argument("--summary-pool", type=Path, default=None, help="summary-only条件目录；默认使用本项目全量总结池")
    legacy.add_argument("--respondent-source", choices=("legacy_100", "all_285"), default="legacy_100")
    legacy.add_argument("--sampling-seed", type=int, default=20260913, help="概率转实际选项的可复现抽样种子")
    summarize = sub.add_parser("summarize-legacy-pool", help="把旧版285名虚拟被试生成统一的summary-only对比条件")
    summarize.add_argument("--legacy-project", type=Path, default=Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT"))
    summarize.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "experiment_data" / "legacy_persona_summary_pool")
    summarize.add_argument("--model", dest="model_id", default=None, help="省略则使用当前.env的MODEL_ID")
    summarize.add_argument("--max-concurrency", type=int, default=5)
    summarize.add_argument("--max-retries", type=int, default=2)
    summarize.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    controls = sub.add_parser("negative-controls", help="用已保存作答做离线负对照，不调用模型")
    controls.add_argument("--evaluation", type=Path, required=True, help="legacy_pool_abc结果目录")
    controls.add_argument("--replications", type=int, default=500)
    controls.add_argument("--seed", type=int, default=20260913)
    controls.add_argument("--output", type=Path, default=None)
    sensitivity = sub.add_parser("sensitivity-curve", help="用已保存作答做质量梯度敏感性实验，不调用模型")
    sensitivity.add_argument("--evaluation", type=Path, required=True, help="legacy_pool_abc结果目录")
    sensitivity.add_argument("--replications", type=int, default=500)
    sensitivity.add_argument("--seed", type=int, default=20260913)
    sensitivity.add_argument("--levels", default="0,0.125,0.25,0.5,0.75,1")
    sensitivity.add_argument("--output", type=Path, default=None)
    mussel = sub.add_parser(
        "mussel-compare",
        help="复用历史Mussel作答，并让具身概率虚拟被试完成同题本后比较",
    )
    mussel.add_argument("--experiment", type=Path, required=True)
    mussel.add_argument(
        "--legacy-project",
        type=Path,
        default=Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT"),
    )
    mussel.add_argument(
        "--summary-pool",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "experiment_data"
            / "legacy_persona_summary_pool"
        ),
    )
    mussel.add_argument("--old-abc-result", type=Path, default=None)
    mussel.add_argument("--new-abc-result", type=Path, default=None)
    mussel.add_argument("--mussel", dest="mussel_path", type=Path, default=None)
    mussel.add_argument("--model", dest="model_id", default=None)
    mussel.add_argument("--max-concurrency", type=int, default=30)
    mussel.add_argument("--max-retries", type=int, default=2)
    mussel.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    mussel.add_argument("--sampling-seed", type=int, default=20260913)
    mussel.add_argument("--output", type=Path, default=None)
    mussel_ipip = sub.add_parser(
        "mussel-ipip",
        help="复用已完成Mussel作答，追加N4/E2/O5/A4/C5的IPIP facet级效标",
    )
    mussel_ipip.add_argument("--experiment", type=Path, required=True)
    mussel_ipip.add_argument("--comparison-result", type=Path, default=None)
    mussel_ipip.add_argument(
        "--legacy-project",
        type=Path,
        default=Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT"),
    )
    mussel_ipip.add_argument(
        "--summary-pool",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "experiment_data"
            / "legacy_persona_summary_pool"
        ),
    )
    mussel_ipip.add_argument(
        "--ipip",
        dest="ipip_path",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "knowledge_base"
            / "items"
            / "ipip_neo_items.json"
        ),
    )
    mussel_ipip.add_argument("--model", dest="model_id", default=None)
    mussel_ipip.add_argument("--max-concurrency", type=int, default=30)
    mussel_ipip.add_argument("--max-retries", type=int, default=2)
    mussel_ipip.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    mussel_ipip.add_argument("--output", type=Path, default=None)
    mussel_ipip_bootstrap = sub.add_parser(
        "mussel-ipip-bootstrap",
        help="对已完成的Mussel×IPIP两方法结果做被试配对bootstrap，不调用模型",
    )
    mussel_ipip_bootstrap.add_argument("--experiment", type=Path, required=True)
    mussel_ipip_bootstrap.add_argument("--comparison-result", type=Path, default=None)
    mussel_ipip_bootstrap.add_argument("--ipip-result", type=Path, default=None)
    mussel_ipip_bootstrap.add_argument("--replications", type=int, default=5000)
    mussel_ipip_bootstrap.add_argument("--seed", type=int, default=20260915)
    mussel_ipip_bootstrap.add_argument("--output", type=Path, default=None)
    defect = sub.add_parser(
        "mussel-item-defect",
        help="同一批100名具身概率虚拟被试比较10道Mussel原题与高水平选项缺失题",
    )
    defect.add_argument("--experiment", type=Path, required=True)
    defect.add_argument(
        "--legacy-project",
        type=Path,
        default=Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT"),
    )
    defect.add_argument(
        "--summary-pool",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "experiment_data"
            / "legacy_persona_summary_pool"
        ),
    )
    defect.add_argument("--stimuli", dest="stimuli_path", type=Path, default=None)
    defect.add_argument("--mussel", dest="mussel_path", type=Path, default=None)
    defect.add_argument("--reference-mussel-run", type=Path, default=None)
    defect.add_argument("--neo-scores", type=Path, default=None)
    defect.add_argument("--model", dest="model_id", default=None)
    defect.add_argument("--max-concurrency", type=int, default=30)
    defect.add_argument("--max-retries", type=int, default=2)
    defect.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    defect.add_argument("--sampling-seed", type=int, default=20260915)
    defect.add_argument("--output", type=Path, default=None)
    current_defect = sub.add_parser(
        "mussel-item-defect-current",
        help="使用当前系统冻结的300名matched score_profile虚拟被试检验2组乐群性缺陷题",
    )
    current_defect.add_argument("--experiment", type=Path, required=True)
    current_defect.add_argument("--stimuli", dest="stimuli_path", type=Path, default=None)
    current_defect.add_argument("--mussel", dest="mussel_path", type=Path, default=None)
    current_defect.add_argument("--model", dest="model_id", default=None)
    current_defect.add_argument("--max-concurrency", type=int, default=30)
    current_defect.add_argument("--max-retries", type=int, default=2)
    current_defect.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    current_defect.add_argument("--sampling-seed", type=int, default=20260915)
    current_defect.add_argument("--output", type=Path, default=None)
    current_fivefacet = sub.add_parser(
        "mussel-item-defect-current-fivefacet",
        help="使用当前score_profile提示词，在五个facet上检验10组原题—缺陷题",
    )
    current_fivefacet.add_argument("--experiment", type=Path, required=True)
    current_fivefacet.add_argument("--stimuli", dest="stimuli_path", type=Path, default=None)
    current_fivefacet.add_argument("--mussel", dest="mussel_path", type=Path, default=None)
    current_fivefacet.add_argument("--model", dest="model_id", default=None)
    current_fivefacet.add_argument("--max-concurrency", type=int, default=30)
    current_fivefacet.add_argument("--max-retries", type=int, default=2)
    current_fivefacet.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    current_fivefacet.add_argument("--sampling-seed", type=int, default=20260915)
    current_fivefacet.add_argument("--output", type=Path, default=None)
    classify_defects = sub.add_parser(
        "classify-mussel-defects",
        help="盲态四分类：正常、构念失效、区分度不足、社会赞许偏差",
    )
    classify_defects.add_argument("--experiment", type=Path, required=True)
    classify_defects.add_argument("--stimuli", dest="stimuli_path", type=Path, default=None)
    classify_defects.add_argument("--model", dest="model_id", default=None)
    classify_defects.add_argument("--repeats", type=int, default=3)
    classify_defects.add_argument("--max-concurrency", type=int, default=10)
    classify_defects.add_argument("--max-retries", type=int, default=2)
    classify_defects.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    classify_defects.add_argument("--seed", type=int, default=20260915)
    classify_defects.add_argument("--output", type=Path, default=None)
    compare_defect_detection = sub.add_parser(
        "compare-mussel-defect-detection",
        help="比较具身概率与当前分数提示词对三类受控坏题的检出能力",
    )
    compare_defect_detection.add_argument("--experiment", type=Path, required=True)
    compare_defect_detection.add_argument("--respondents", type=int, default=100)
    compare_defect_detection.add_argument("--model", dest="model_id", default=None)
    compare_defect_detection.add_argument("--max-concurrency", type=int, default=30)
    compare_defect_detection.add_argument("--max-retries", type=int, default=2)
    compare_defect_detection.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    compare_defect_detection.add_argument("--sampling-seed", type=int, default=20260915)
    compare_defect_detection.add_argument("--output", type=Path, default=None)
    e2_recovery = sub.add_parser(
        "source-e2-recovery",
        help="离线检验具身A/B/C与NEO作答能否恢复原始乐群性E2分数",
    )
    e2_recovery.add_argument("--experiment", type=Path, required=True)
    e2_recovery.add_argument(
        "--source-pool",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "sjt_system"
        / "data"
        / "virtual_respondents.json",
    )
    e2_recovery.add_argument(
        "--evaluation-dirname",
        default="legacy_summary_embodied_probability_abc",
    )
    e2_recovery.add_argument("--output", type=Path, default=None)
    probability_resampling = sub.add_parser(
        "probability-resampling",
        help="复用具身A/B/C冻结概率，离线比较重抽、期望分与argmax",
    )
    probability_resampling.add_argument("--experiment", type=Path, required=True)
    probability_resampling.add_argument(
        "--evaluation-dirname",
        default="legacy_summary_embodied_probability_abc",
    )
    probability_resampling.add_argument("--replications", type=int, default=100)
    probability_resampling.add_argument("--seed", type=int, default=20260913)
    probability_resampling.add_argument("--output", type=Path, default=None)
    embodied_batch = sub.add_parser(
        "embodied-prompt-batch-audit",
        help="批量检查第一人称具身提示词的可见遵循情况",
    )
    embodied_batch.add_argument("--experiment", type=Path, required=True)
    embodied_batch.add_argument("--respondents", type=int, default=30)
    embodied_batch.add_argument("--items", type=int, default=4)
    embodied_batch.add_argument("--max-concurrency", type=int, default=10)
    embodied_batch.add_argument("--model", dest="model_id", default="glm-5.3-flash")
    embodied_batch.add_argument("--summary-pool", type=Path, default=None)
    embodied_batch.add_argument("--source-pool", type=Path, default=None)
    embodied_batch.add_argument("--seed", type=int, default=20260915)
    embodied_batch.add_argument("--output", type=Path, default=None)
    fresh_mussel_ipip = sub.add_parser(
        "fresh-mussel-ipip",
        help="不复用旧作答，重新运行具身与显式分数条件的全量Mussel×IPIP实验",
    )
    fresh_mussel_ipip.add_argument("--experiment", type=Path, required=True)
    fresh_mussel_ipip.add_argument("--respondents", type=int, default=285)
    fresh_mussel_ipip.add_argument("--summary-pool", type=Path, default=None)
    fresh_mussel_ipip.add_argument("--mussel", dest="mussel_path", type=Path, default=None)
    fresh_mussel_ipip.add_argument("--ipip", dest="ipip_path", type=Path, default=None)
    fresh_mussel_ipip.add_argument("--model", dest="model_id", default="glm-5.3-flash")
    fresh_mussel_ipip.add_argument("--max-concurrency", type=int, default=30)
    fresh_mussel_ipip.add_argument("--max-retries", type=int, default=2)
    fresh_mussel_ipip.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    fresh_mussel_ipip.add_argument("--sampling-seed", type=int, default=20260916)
    fresh_mussel_ipip.add_argument("--score-seed", type=int, default=20260916)
    fresh_mussel_ipip.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "virtual-prompts":
            specs = load_prompt_specs(args.root)
            if args.action == "list":
                print(format_prompt_list(specs))
            elif args.action == "show":
                if not args.prompt_id:
                    raise ValueError("show必须提供 --prompt-id")
                spec = get_prompt_spec(args.prompt_id, args.root)
                print(json.dumps({
                    "prompt_id": spec.prompt_id,
                    "label": spec.label,
                    "description": spec.description,
                    "persona_information": spec.persona_information,
                    "response_behavior": spec.response_behavior,
                    "extra_parameters": list(spec.extra_parameters),
                    "parameter_values_file": spec.parameter_values_file,
                    "persona_input": spec.persona_input,
                    "sjt_response_mode": spec.sjt_response_mode,
                    "ipip_response_mode": spec.ipip_response_mode,
                    "version": spec.version,
                    "source": spec.source,
                    "template": spec.template,
                }, ensure_ascii=False, indent=2))
            else:
                print(f"提示词校验通过：{len(specs)} 个；目录={Path(args.root).resolve()}")
            return 0
        if args.command == "combine":
            from .combination_search import SearchConfig, run_combination_search

            _, summary = run_combination_search(args.experiment, SearchConfig(
                sample_role=args.sample_role, objective=args.objective, bank_path=args.bank,
                allow_partial_bank=args.allow_partial_bank, alpha_min=args.alpha_min, icc_min=args.icc_min,
                max_non_target_rho=args.max_non_target_rho, target_domain=args.target_domain,
                batch_size=args.batch_size, max_combinations=args.max_combinations, top_k=args.top_k))
            return 0 if summary["status"] == "completed" else 2
        if args.command == "legacy-abc":
            from .legacy_pool_abc import LegacyABCConfig, run_legacy_abc

            root, result = asyncio.run(run_legacy_abc(LegacyABCConfig(
                experiment=args.experiment,
                legacy_project=args.legacy_project,
                model_id=args.model_id,
                neo_ffi_path=args.neo_ffi_path,
                max_concurrency=args.max_concurrency,
                max_retries=args.max_retries,
                timeout_seconds=args.timeout_seconds,
                respondent_limit=args.respondent_limit,
                output=args.output,
                persona_mode=args.persona_mode,
                summary_pool=args.summary_pool,
                respondent_source=args.respondent_source,
                sampling_seed=args.sampling_seed,
            )))
            print(f"实验输出：{root}")
            print(f"报告：{result['report']}")
            return 0
        if args.command == "summarize-legacy-pool":
            from .legacy_persona_summary import (
                LegacySummaryConfig,
                run_legacy_summary_pool,
            )

            root, result = asyncio.run(run_legacy_summary_pool(
                LegacySummaryConfig(
                    legacy_project=args.legacy_project,
                    output=args.output,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                )
            ))
            print(f"总结条件输出：{root}")
            print(f"被试数：{result['respondent_count']}")
            print(f"报告：{result['report']}")
            return 0
        if args.command == "negative-controls":
            from .negative_controls import run_negative_controls

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
        if args.command == "sensitivity-curve":
            from .sensitivity_curve import _parse_levels, run_sensitivity_curve

            output, result = run_sensitivity_curve(
                args.evaluation,
                replications=args.replications,
                seed=args.seed,
                levels=_parse_levels(args.levels),
                output_root=args.output,
            )
            print(f"敏感性实验输出：{output}")
            print(f"结论：{result['interpretation']}")
            print(f"报告：{output / 'report.html'}")
            return 0
        if args.command == "mussel-compare":
            from .mussel_method_comparison import (
                MusselComparisonConfig,
                run_mussel_method_comparison,
            )

            output, result = asyncio.run(run_mussel_method_comparison(
                MusselComparisonConfig(
                    experiment=args.experiment,
                    legacy_project=args.legacy_project,
                    summary_pool=args.summary_pool,
                    old_abc_result=args.old_abc_result,
                    new_abc_result=args.new_abc_result,
                    mussel_path=args.mussel_path,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    sampling_seed=args.sampling_seed,
                    output=args.output,
                )
            ))
            print(f"Mussel比较输出：{output}")
            print(f"报告：{result['report']}")
            return 0
        if args.command == "mussel-ipip":
            from .mussel_ipip_reference import (
                MusselIPIPConfig,
                run_mussel_ipip_reference,
            )

            output, result = asyncio.run(
                run_mussel_ipip_reference(
                    MusselIPIPConfig(
                        experiment=args.experiment,
                        comparison_result=args.comparison_result,
                        legacy_project=args.legacy_project,
                        summary_pool=args.summary_pool,
                        ipip_path=args.ipip_path,
                        model_id=args.model_id,
                        max_concurrency=args.max_concurrency,
                        max_retries=args.max_retries,
                        timeout_seconds=args.timeout_seconds,
                        output=args.output,
                    )
                )
            )
            print(f"Mussel IPIP facet效标输出：{output}")
            print(f"被试={result['respondent_count']}；新增效标=5个facet×10题")
            print(f"报告：{result['report']}")
            return 0
        if args.command == "mussel-ipip-bootstrap":
            from .mussel_ipip_bootstrap import (
                MusselIPIPBootstrapConfig,
                run_mussel_ipip_bootstrap,
            )

            output, result = run_mussel_ipip_bootstrap(
                MusselIPIPBootstrapConfig(
                    experiment=args.experiment,
                    comparison_result=args.comparison_result,
                    ipip_result=args.ipip_result,
                    replications=args.replications,
                    seed=args.seed,
                    output=args.output,
                )
            )
            print(f"Mussel IPIP bootstrap输出：{output}")
            print(
                f"被试={result['respondent_count']}；"
                f"配对重抽样={result['replications']}次"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "mussel-item-defect":
            from .mussel_item_defect_pilot import (
                DEFAULT_STIMULI,
                MusselItemDefectConfig,
                run_mussel_item_defect_pilot,
            )

            output, result = asyncio.run(run_mussel_item_defect_pilot(
                MusselItemDefectConfig(
                    experiment=args.experiment,
                    legacy_project=args.legacy_project,
                    summary_pool=args.summary_pool,
                    stimuli_path=args.stimuli_path or DEFAULT_STIMULI,
                    mussel_path=args.mussel_path,
                    reference_mussel_run=args.reference_mussel_run,
                    neo_scores_path=args.neo_scores,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    sampling_seed=args.sampling_seed,
                    output=args.output,
                )
            ))
            print(f"题目缺陷实验输出：{output}")
            print(
                "主分析检出："
                f"{result['summary']['primary_majority_detected_pairs']}/"
                f"{result['summary']['pair_count']}"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "mussel-item-defect-current":
            from .current_system_item_defect_pilot import (
                DEFAULT_MUSSEL,
                DEFAULT_STIMULI,
                CurrentSystemDefectConfig,
                run_current_system_item_defect_pilot,
            )

            output, result = asyncio.run(run_current_system_item_defect_pilot(
                CurrentSystemDefectConfig(
                    experiment=args.experiment,
                    stimuli_path=args.stimuli_path or DEFAULT_STIMULI,
                    mussel_path=args.mussel_path or DEFAULT_MUSSEL,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    sampling_seed=args.sampling_seed,
                    output=args.output,
                )
            ))
            print(f"当前系统缺陷实验输出：{output}")
            print(
                "当前四门槛多数检出："
                f"{result['summary']['majority_detected_pairs']}/"
                f"{result['summary']['pair_count']}"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "mussel-item-defect-current-fivefacet":
            from .current_score_profile_fivefacet import (
                DEFAULT_MUSSEL,
                DEFAULT_STIMULI,
                run_current_score_profile_fivefacet,
            )
            from .current_system_item_defect_pilot import CurrentSystemDefectConfig

            output, result = asyncio.run(run_current_score_profile_fivefacet(
                CurrentSystemDefectConfig(
                    experiment=args.experiment,
                    stimuli_path=args.stimuli_path or DEFAULT_STIMULI,
                    mussel_path=args.mussel_path or DEFAULT_MUSSEL,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    sampling_seed=args.sampling_seed,
                    output=args.output,
                )
            ))
            print(f"当前分数提示词五facet实验输出：{output}")
            print(
                "主要指标同时检出："
                f"{result['summary']['primary_detected_pairs']}/"
                f"{result['summary']['primary_estimable_pairs']}"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "classify-mussel-defects":
            from .defect_type_classifier import (
                DEFAULT_STIMULI,
                DefectClassifierConfig,
                run_defect_classifier,
            )

            output, result = asyncio.run(run_defect_classifier(
                DefectClassifierConfig(
                    experiment=args.experiment,
                    stimuli_path=args.stimuli_path or DEFAULT_STIMULI,
                    model_id=args.model_id,
                    repeats=args.repeats,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    seed=args.seed,
                    output=args.output,
                )
            ))
            metrics = result["summary"]["item_level_majority"]
            print(f"盲态分类输出：{output}")
            print(
                f"题目多数投票准确率={metrics['accuracy']:.3f}；"
                f"宏平均F1={metrics['macro_f1']:.3f}"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "compare-mussel-defect-detection":
            from .three_defect_virtual_comparison import (
                ThreeDefectVirtualComparisonConfig,
                run_three_defect_virtual_comparison,
            )

            output, result = asyncio.run(run_three_defect_virtual_comparison(
                ThreeDefectVirtualComparisonConfig(
                    experiment=args.experiment,
                    model_id=args.model_id,
                    respondents=args.respondents,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    sampling_seed=args.sampling_seed,
                    output=args.output,
                )
            ))
            print(f"三类坏题双虚拟被试实验输出：{output}")
            for row in result["summary"]["detection_summary"]:
                if row["defect_type"] == "all":
                    print(
                        f"{row['method']}：{row['detected_pairs']}/"
                        f"{row['estimable_pairs']}，检出率={row['detection_rate']:.3f}"
                    )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "source-e2-recovery":
            from .source_e2_recovery import E2RecoveryConfig, run_source_e2_recovery

            output, result = run_source_e2_recovery(
                E2RecoveryConfig(
                    experiment=args.experiment,
                    source_pool=args.source_pool,
                    evaluation_dirname=args.evaluation_dirname,
                    output=args.output,
                )
            )
            print(f"乐群性E2恢复实验输出：{output}")
            print(
                f"被试={result['respondent_count']}；"
                f"原始E2均值={result['source_e2_mean']:.3f}；"
                f"SD={result['source_e2_sd']:.3f}"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "probability-resampling":
            from .probability_resampling import (
                ProbabilityResamplingConfig,
                run_probability_resampling,
            )

            output, result = run_probability_resampling(
                ProbabilityResamplingConfig(
                    experiment=args.experiment,
                    evaluation_dirname=args.evaluation_dirname,
                    replications=args.replications,
                    seed=args.seed,
                    output=args.output,
                )
            )
            print(f"概率离线重计分输出：{output}")
            print(
                f"被试={result['respondent_count']}；"
                f"重抽={result['replications']}次；模型调用=0；Token=0"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "embodied-prompt-batch-audit":
            from .embodied_prompt_audit import DEFAULT_SUMMARY_POOL
            from .embodied_prompt_batch_audit import (
                DEFAULT_SOURCE_POOL,
                EmbodiedBatchAuditConfig,
                run_embodied_batch_audit,
            )

            output, result = asyncio.run(run_embodied_batch_audit(
                EmbodiedBatchAuditConfig(
                    experiment=args.experiment,
                    respondents=args.respondents,
                    items=args.items,
                    max_concurrency=args.max_concurrency,
                    model_id=args.model_id,
                    summary_pool=args.summary_pool or DEFAULT_SUMMARY_POOL,
                    source_pool=args.source_pool or DEFAULT_SOURCE_POOL,
                    seed=args.seed,
                    output=args.output,
                )
            ))
            print(f"批量审计输出：{output}")
            print(
                f"成功={result['successful_outputs']}/{result['total_jobs']}；"
                f"整题第一人称合规率={result['whole_response_first_person_compliance_rate']:.3f}；"
                f"选项合规率={result['option_first_person_compliance_rate']:.3f}"
            )
            print(f"报告：{result['report']}")
            return 0
        if args.command == "fresh-mussel-ipip":
            from .fresh_mussel_ipip_comparison import (
                DEFAULT_IPIP,
                DEFAULT_MUSSEL,
                DEFAULT_IMPORTED_SUMMARY_POOL,
                FreshComparisonConfig,
                run_fresh_comparison,
            )

            output, result = asyncio.run(run_fresh_comparison(
                FreshComparisonConfig(
                    experiment=args.experiment,
                    respondents=args.respondents,
                    summary_pool=args.summary_pool or DEFAULT_IMPORTED_SUMMARY_POOL,
                    mussel_path=args.mussel_path or DEFAULT_MUSSEL,
                    ipip_path=args.ipip_path or DEFAULT_IPIP,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    sampling_seed=args.sampling_seed,
                    score_seed=args.score_seed,
                    output=args.output,
                )
            ))
            print(f"全量Mussel×IPIP双方法实验输出：{output}")
            print(f"被试={result['respondent_count']}；预计模型调用={result['expected_model_calls']}")
            print(f"报告：{result['report']}")
            return 0
        if args.command == "run":
            cfg = ExperimentConfig.model_validate(json.loads(args.config.read_text(encoding="utf-8-sig"))) if args.config else ExperimentConfig()
            store = ExperimentStore.create(args.output_root, cfg)
            print(f"实验目录：{store.root}", flush=True)
        else:
            store = ExperimentStore(args.experiment)
        if args.command == "report":
            print(write_report(store))
            return 0
        if args.command == "neo-ffi":
            from .neo_reference import run_neo_ffi_addon

            metrics = asyncio.run(run_neo_ffi_addon(store))
            report = write_report(store)
            print(
                f"\nNEO-FFI追加评估完成：被试={metrics['respondent_count']}；"
                f"模型={metrics['neo_ffi_model_id']}\n报告：{report}"
            )
            return 0
        from .runner import ExperimentRunner
        runner = ExperimentRunner(store)
        asyncio.run(runner.evaluate() if args.command == "evaluate" else runner.run())
        progress = store.read("progress.json")
        print(f"\n状态：{progress['status']}\n报告：{store.path('summary/report.html')}")
        if progress.get("error"):
            print(progress["error"])
        return 0 if progress["status"] == "completed" else 2
    except KeyboardInterrupt:
        if args.command == "combine":
            print("组合筛选已中断；原实验未修改。重新执行相同combine命令可从头计算，不调用模型。")
        else:
            print("已中断；请使用resume和上面显示的实验目录恢复。")
        return 130
    except Exception as exc:
        print(f"实验未完成：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
