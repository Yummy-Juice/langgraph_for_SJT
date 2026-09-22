# 后半程单题返修隔离测试器

这个模块用于单独验证“基线之后单题返修”的逻辑，不修改生产环境的 C 流程，也不覆盖原实验目录。

## 覆盖范围

```text
基线题目集合
  -> 找出未通过题
  -> 并发诊断
  -> 每题最多三次返修 + 单题局部复测
  -> 三次仍未通过：同槽位备用题 -> 自动补题
  -> 当前题目集合直接进入下一轮
  -> 保存每轮当前题目集合和题目版本
  -> 达到最大返修轮次或全部题目通过后停止
```

技术失败、诊断失败和槽位不足是独立状态，不会被转成“题目合格”，也不会触发人工 CLI。

## 先运行离线自测

```powershell
python -X utf8 -m experiments.post_baseline_iteration_lab self-test
python -X utf8 -m experiments.post_baseline_iteration_lab demo
```

演示报告在 `experiment_data/post_baseline_lab_demo/summary/report.html`。

## 从真实基线快照开始

先把已有快照标准化：

```powershell
python -X utf8 -m experiments.post_baseline_iteration_lab export `
  --source "E:\DR_projects\langgraph_for_SJT\experiment_data\exp_xxx" `
  --output "E:\DR_projects\langgraph_for_SJT\experiment_data\post_baseline_snapshot.json"
```

然后运行：

```powershell
python -X utf8 -m experiments.post_baseline_iteration_lab run `
  --snapshot "E:\DR_projects\langgraph_for_SJT\experiment_data\post_baseline_snapshot.json" `
  --output "E:\DR_projects\langgraph_for_SJT\experiment_data\post_baseline_lab\run_01"
```

真实模型接入前，使用 `--scenario` 以离线替身验证所有分支。当前模块只验证单题返修、局部复测、备用题/补题和保存/恢复，不执行候选题重新组卷。

## 接入正式 LLM

正式模型模式复用主系统中已经配置好的：

- `psychometric_repair_diagnosis_agent`；
- `psychometric_item_repair_agent`；
- `run_virtual_response_simulation` 的单题私有作答适配；
- `evaluate_single_item_candidate`。

它们运行在隔离测试目录的私有状态投影上。局部复测每次只把当前候选题放入正式的 matched-condition 施测器，
同时得到三个条件组的单题作答和 target 重测记录。局部复测产生的虚拟作答和模型调用记录不会写回原 C 任务。
```powershell
python -X utf8 -m experiments.post_baseline_iteration_lab run `
  --snapshot "E:\DR_projects\langgraph_for_SJT\experiment_data\post_baseline_snapshot.json" `
  --output "E:\DR_projects\langgraph_for_SJT\experiment_data\post_baseline_lab\llm_run_01" `
  --engine llm `
  --model glm-5.3-flash
```

也可以直接把正式实验的 `C\\baseline` 目录作为 `--snapshot`，无需先导出 JSON：

```powershell
python -X utf8 -m experiments.post_baseline_iteration_lab run `
  --snapshot "E:\\DR_projects\\langgraph_for_SJT\\experiment_data\\exp_xxx\\C\\baseline" `
  --output "E:\\DR_projects\\langgraph_for_SJT\\experiment_data\\post_baseline_lab\\llm_run_01" `
  --engine llm `
  --model glm-5.3-flash
```

在隔离进程中，`--model` 会同时覆盖心理测量诊断、题目返修和局部虚拟施测；不会改写主系统的 `.env` 文件。
`--engine llm` 会真实消耗模型调用和虚拟局部复测成本；不要对内置 demo 快照使用该模式，因为 demo 没有正式 matched-condition 作答 manifest。
