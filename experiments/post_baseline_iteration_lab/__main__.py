from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .core import LabConfig, PostBaselineIterationLab, demo_snapshot, export_snapshot, load_scenario, load_snapshot, render_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从基线到单题返修的隔离流程测试器")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="从标准化基线快照开始运行")
    run.add_argument("--snapshot", required=True, help="快照 JSON 或包含快照的目录")
    run.add_argument("--output", required=True, help="隔离输出目录")
    run.add_argument("--scenario", help="离线模型替身场景 JSON")
    run.add_argument("--config", help="隔离测试配置 JSON")
    run.add_argument("--engine", choices=["offline", "llm"], default=None, help="offline 为离线回归；llm 调用正式诊断/返修/局部复测模型")
    run.add_argument("--model", help="在隔离进程中统一覆盖诊断、返修和局部施测模型，例如 glm-5.3-flash")
    run.add_argument("--stop-after", help="测试恢复：baseline 或 round_02 等")

    resume = sub.add_parser("resume", help="从 checkpoint.json 恢复")
    resume.add_argument("--lab", required=True, help="隔离输出目录")

    export = sub.add_parser("export", help="把现有实验快照标准化为隔离测试输入")
    export.add_argument("--source", required=True, help="现有实验目录或 JSON")
    export.add_argument("--output", required=True, help="标准化快照 JSON")

    report = sub.add_parser("report", help="只根据 checkpoint 重建报告")
    report.add_argument("--lab", required=True, help="隔离输出目录")

    demo = sub.add_parser("demo", help="运行内置离线演示")
    demo.add_argument("--output", default="experiment_data/post_baseline_lab_demo")
    demo.add_argument("--final-items", type=int, default=4)

    self_test = sub.add_parser("self-test", help="运行内置流程测试场景")
    self_test.add_argument("--output", default="experiment_data/post_baseline_lab_self_test")
    return parser


def _load_config(path: str | None) -> LabConfig:
    if not path:
        return LabConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LabConfig.from_dict(payload)


def _run_self_test(root: Path) -> int:
    scenarios = {
        "repair_then_complete": {
            "items": {
                "baseline-02": {"pass_after": 2},
                "baseline-03": {"pass_after": None, "replacement_pass_after": 1},
                "baseline-04": {"pass_after": None, "disable_reserve": True, "replacement_pass_after": 1},
            },
        },
        "technical_failure": {
            "items": {"baseline-02": {"diagnosis_failures": 10}},
        },
    }
    for name, scenario in scenarios.items():
        output = root / name
        lab = PostBaselineIterationLab(demo_snapshot(4), output, LabConfig(max_model_retries=1, max_repair_rounds=3), scenario)
        state = lab.run()
        print(f"{name}: {state['status']}；报告={output / 'summary' / 'report.html'}")
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "export":
        output = export_snapshot(args.source, args.output)
        print(f"标准化快照：{output}")
        return 0
    if args.command == "report":
        output = render_report(args.lab)
        print(f"报告：{output}")
        return 0
    if args.command == "resume":
        state = PostBaselineIterationLab.resume(args.lab).run()
        print(f"恢复完成：状态={state['status']}；报告={Path(args.lab) / 'summary' / 'report.html'}")
        return 0
    if args.command == "self-test":
        return _run_self_test(Path(args.output))
    if args.command == "demo":
        output = Path(args.output)
        scenario = {
            "items": {
                "baseline-02": {"pass_after": 2},
                "baseline-03": {"pass_after": None, "replacement_pass_after": 1},
                "baseline-04": {"pass_after": None, "disable_reserve": True, "replacement_pass_after": 1},
            },
        }
        state = PostBaselineIterationLab(demo_snapshot(args.final_items), output, scenario=scenario).run()
        print(f"演示完成：状态={state['status']}；报告={output / 'summary' / 'report.html'}")
        return 0
    if args.command == "run":
        snapshot = load_snapshot(args.snapshot)
        scenario = load_scenario(args.scenario)
        config = _load_config(args.config)
        if args.engine:
            config = LabConfig.from_dict({**asdict(config), "engine_mode": args.engine})
        if args.model:
            config = LabConfig.from_dict({**asdict(config), "model_id": args.model})
        state = PostBaselineIterationLab(snapshot, args.output, config, scenario).run(stop_after=args.stop_after)
        print(f"隔离测试完成：状态={state['status']}；报告={Path(args.output) / 'summary' / 'report.html'}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
