# 独立单批次 A/B/C 实验

一次启动完成一组方法对照，不是自动跑10批。C保留现有CLI的人工作答、确认、defer处置与暂停。程序在阶段结束即保存，不需要等实验全部完成。

## 运行

在项目根目录、已安装项目依赖的Python环境中运行；模型凭据仍从现有`.env`读取，不写入实验目录。

```powershell
python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/example_config.json
```

此命令会调用真实模型并产生费用。正式默认是16题、32题候选库、每组100名虚拟被试；建议先复制配置，用较少题量、每组30人做联调，再建立新的正式实验。离线自动测试不调用真实模型。

示例配置把角色严格分离：`model_id=deepseek-flash-guan`负责A/B/C的出题、审题、诊断、返修与组卷；`virtual_respondent_model_id=glm-5.3-flash`负责C开发施测和单题局部复测；`evaluation_model_id=glm-5.3-flash`负责A/B/C独立评估及后续NEO-FFI追加作答。三者均冻结进实验目录，恢复时不得静默更换。旧配置未填写新增字段时，虚拟被试仍回退到`model_id`，保持原有行为。

启动时输出独立目录，例如`experiment_data/exp_20260911_100000_ab12cd34/`。每次`run`创建新目录。恢复已有实验使用：

```powershell
python -X utf8 -m experiments.system_comparison resume --experiment experiment_data/exp_20260911_100000_ab12cd34
```

C需要暂停时使用原菜单的“暂停”选项，或中断程序。检查点会保留已完成的节点结果；正常在确认处恢复不会重新出题。若在模型请求已送达但结果尚未落盘时进程被强制关闭，该未确认请求无法保证不重复计费。系统异常后仍保留阶段、错误原因和已落盘作答；恢复不能修复错误配置或损坏文件。

执行节点暂时失败后，重新运行`resume`会从失败步骤前的检查点创建重试分支，保留已完成的上游阶段，并写入`recovery_events.json`。每次恢复最多重新尝试该步骤一次，不会在同一进程内无限重试。未落盘的节点内部子任务仍可能需要重做。

全部开发冻结后，可以补做独立评估或只重建报告：

```powershell
python -X utf8 -m experiments.system_comparison evaluate --experiment experiment_data/exp_20260911_100000_ab12cd34
python -X utf8 -m experiments.system_comparison report --experiment experiment_data/exp_20260911_100000_ab12cd34
```

`report`只读取已有数据，不请求模型。C尚未完成时不开放独立评估，避免评估结果反馈到开发中。

## 方法及轮次

- A：无示例单次生成基线。提示词自动填入固定构念定义、高低表现、区分提醒、目标人群与题量；四个选项按目标特质高低分别计1—4分，各一次。每题和每个选项的简短设计依据单独保存；只做结构与完整性校验，不使用虚拟指标改题。
- shared：调用现有系统开发并审查2倍题库。共享详细蓝图和初始题库在此冻结。A使用预先确定的内容要求，不获得后续生成的题目、证据扩展或虚拟结果。
- B：专用理论组卷提示词只能看到蓝图、构念和候选题文字，按每个蓝图单元保留量选题，不调用指标工具。
- C/baseline：B的理论组卷。
- C/round_01：首次开发虚拟施测后的组卷。
- C/round_02起：返修后组卷。`max_repair_rounds=3`表示最多初次施测后再完成3个整批返修轮次；单题内部返修仍由原系统控制。
- C/final.json：按开发质量和稳定性门槛选择历史保存版本；不会用独立评估挑最高分卷。结果注明临时卷性质，未通过单题门槛的题不会被自动标为合格。

## 保存内容

```text
experiment_data/<实验编号>/
  config.json                  # 固定配置（修改后不能直接恢复原实验）
  model_roles.json             # 各角色实际模型、温度、推理设置及非敏感环境设置
  provenance.json              # 代码版本/源码摘要
  progress.json                # 当前阶段、完成阶段、错误或暂停信息
  participants/               # development与evaluation分开保存
  shared/                     # 内容要求、构念、详细蓝图、32题库与检查点
  A/round_01/                 # form.json/html、调用记录、cost.json
  B/round_01/                 # 同上，含理论选择理由
  C/baseline/                 # B的基线快照
  C/round_01/                 # 每轮冻结问卷、development开发证据
  C/round_02/...
  C/graph.sqlite              # 真正的工作流状态/未提交节点写入/人工中断
  C/checkpoint.json           # 可读的业务状态
  C/human_decisions.json      # 人工处置记录
  runtime/                    # 本实验独占的情境扩展、开发作答、局部复测缓存
  summary/report.html         # 浏览器直接打开
```

每个问卷目录下的`evaluation/`保存独立作答、计分矩阵、整卷指标，以及：

- `item_metrics.csv`、`items.json`：四项单题指标、阈值、是否可估计、通过状态、失败门槛、非目标facet及相关原值。
- `item_statistics.json`：完整单题计算结果和解释。
- `option_statistics.csv`、`option_choice_diagnostics.json`：选项人数、比例、均值及定位证据。
- `analysis_manifest.json`：公式版本、阈值和原始数据引用。
- `score_matrices/`：每个匹配条件的“被试×题目”计分矩阵；target重测另表保存。

问卷HTML隐藏计分键、方法标签；计分键保存在JSON。需要正式盲法分配时还应另行编号分发。

A额外输出`A/round_01/design_rationales.json`，记录逐题情境与A/B/C/D选项的简短说明，绑定题号、版本、计分键与问卷指纹。这些说明不写入正式问卷，不进入独立评估或虚拟被试输入；它们是模型的设计理由，不是已验证的信效度证据。缺少或错配说明时沿用最多2次结构重试。

A提示词版本为`a-zero-shot-four-level-v2`，完整渲染内容保存在`request.json`。已冻结的旧A卷继续保留，不会因升级重出；如果旧实验A尚未完成且已保存旧版请求，系统拒绝静默替换提示词，需新建实验使用新版。该无示例版本参考用户提供的提示词框架改编，原论文来源待补充，不宣称严格复现原研究。

C各轮还保存`retained_form.json`（当时历史保留卷的完整题目版本）与`cost.json`（截至该次组卷的C累计及阶段成本）。组卷后的返修不会被回填进上一轮成本。`development/snapshot_complete.json`标记快照已完整落盘；普通恢复会补齐未完成的保存。

## 指标与数据隔离

直接复用当前四门槛：target组同facet CITC、target Spearman相关、同领域VTS、跨领域VTS；VTS使用当前实现的最大**带符号**非目标相关。选项梯度保留为诊断信息，不新增资格门槛。

A/B/C评估时都在当前问卷内重新计算CITC；C开发32题库CITC存于development，不能替代最终16题的CITC。一个facet仅有一道题、常量分数或其他缺失情况记为不可估计，不填0。整卷新规则使用Cronbach α、ICC、目标IPIP相关、Δmin和目标Hedges’ g；旧R²、S和I_g只作为历史或诊断字段。

开发和评估使用不同人格样本及独立目录。各方法/轮次共用评估被试。缓存复用必须由现有签名校验通过：配置、模型、题目文字及计分一致；首次与重测分别存储。不复用开发数据作为独立评估。不额外调用Neo-FFI或Mussel。相同B/C基线评价可复用，但分别导出并注明来源。

## 成本与解释

共享开发和理论组卷成本计入B与C各自的端到端成本；真实实验账单的阶段表中只计一次。C新增开发成本、独立评估成本单列。墙钟时间含人工等待；模型毫秒为并发调用累计值，不是实际运行时长。

可以在配置中填写`input_price_per_million`、`cached_input_price_per_million`、`output_price_per_million`。只有用量、缓存拆分和适用单价齐全时才计算费用，未知不显示为0。不同评估模型未提供对应价格时其费用保持未知。

`model_id`用于全部作者角色，包括A/B、情境扩展、蓝图、出题、审题、心理测量诊断、返修和组卷；`virtual_respondent_model_id`只用于C开发作答与单题局部复测；`evaluation_model_id`只用于独立评估和NEO-FFI追加作答。实际设置冻结在`model_roles.json`。混用其他模型时，不把主模型单价冒用到这些调用。实验适配器会重新绑定已在导入时创建的Agent并在退出时还原，因此同一个进程一次只运行一个实验，不支持并发启动多个ExperimentRunner。

单批次报告为描述性结果，不提供跨批次显著性或置信区间。目标Hedges’ g只有在α、ICC通过且Δmin不下降、目标IPIP相关最多下降0.02时才可替换历史最佳卷。C优于B也不能排除额外计算投入的作用。模型调用的随机性不保证仅凭本地seed逐字复现。

## 冻结题库的离线组合筛选

`combine`不出题、不改题、不重新施测，也不修改主实验的检查点、报告或最终选卷。它从冻结题库中按蓝图保留量枚举选法。16个单元、每单元2选1时共有65,536种组合。

在当前实验的完整开发作答上，检查原定义Q的优化空间：

```powershell
python -X utf8 -m experiments.system_comparison combine --experiment experiment_data/exp_20260911_155055_92946884 --sample-role development --objective q
```

在追加了NEO问卷的评估样本上，直接按SJT总分与NEO-E实际得分的Spearman相关排序：

```powershell
python -X utf8 -m experiments.system_comparison combine --experiment experiment_data/exp_20260911_155055_92946884 --sample-role evaluation --objective neo --allow-partial-bank
```

这次评估数据仅覆盖初始32题中的23道原版本，9道没有原版本作答，满足蓝图的可枚举组合只有128种。`--allow-partial-bank`明确表示只搜索这一子题库；默认不允许缺题。不跨开发/评估样本补分，不把同题号的新版本混入旧版本。

- 默认要求alpha≥0.80、虚拟重测ICC≥0.80，并排除完全相同的情境文字；门槛可配置，不代表统一心理测量标准。不重新做语义审题，不自动获得单题资格。
- `--objective q`沿用现有5折岭回归R²与构念选择性S，Q=√(clip(R²,0,1)×S)，批量化计算与逐卷公式一致。原始R²仍保存负值。Q不是常规信效度。
- `--objective neo`只最大化实际目标NEO相关，并列时按固定组合编号排序，不根据缺失的区分指标挑卷。也同时导出同样约束下的Q最优卷供比较。
- 若希望限制非目标相关，可设置`--max-non-target-rho`。此时必须四个非目标NEO维度都可估计；任一缺失就不通过，绝不填0。目前N/O/C在这100人中无变异，不能声称完整区分效度合格。
- 搜索使用评估样本后，该样本不再是这次搜索的独立验证集；结果仅是探索性搜索内最优。当前SJT与追加NEO使用不同模型，报告明确标记混合模型探索。

每次结果独立保存在`<实验目录>/combination_search/search_<时间>_<样本>_<目标>_<编号>/`，包含全部组合的`combinations.csv`、题号/版本索引、前20名、最佳卷JSON/HTML、同数据重算的B/C首轮基线、数据覆盖与来源摘要、配置、成本和`report.html`。缺失值保留为空并解释原因。`--max-combinations`默认上限100万；中断后可重新执行相同命令从头计算，不产生模型费用，不支持用主实验`resume`恢复组合搜索。

## 离线验证

## 旧虚拟被试 A/B/C 黑盒复测

### 生成285人summary-only对比条件

先把旧项目冻结池中的285名虚拟被试全部重新总结，并将来源快照、总结、模型与提示词版本保存到当前项目：

```powershell
python -X utf8 -m experiments.system_comparison summarize-legacy-pool
```

默认输出到`experiment_data/legacy_persona_summary_pool/`。同一命令可断点续跑；目录会绑定旧池哈希、模型和提示词版本，任一项改变都必须换新目录，避免混合条件。旧项目已有的100条摘要不会混入本次全量重生成结果。

该目录定义`legacy_summary_only`条件：后续SJT和NEO-FFI作答只接收一段行为画像，不再接收39道人格原题、人格分数或构念名称。这个操作减少直接输入信息，但摘要仍来源于人格题，不能被解释为完全消除语义线索。

若已有冻结的A、B、C问卷，需要用旧项目中同一批100名虚拟被试重新作答，可以使用独立入口。它不会进入主流程，也不会计算目标恢复R²、VTS或构念选择性S：

```powershell
python -X utf8 -m experiments.system_comparison legacy-abc `
  --experiment experiment_data/exp_20260911_155055_92946884
```

默认从当前`.env`读取模型，从旧项目的`sjt_system/data/virtual_respondents.json`、`experiments/mussel_validation/out/persona_summaries.jsonl`和旧宜人性作答记录中读取同一批被试。NEO-FFI由这批被试重新作答，反向题按题本中的`scoring`方向计分；不会使用预设人格分数作为NEO答案。

输出在`<实验目录>/legacy_pool_abc/summary/report.html`。A、B、C各自保存单题CITC、选项分布、难度、整卷Cronbach α，以及与NEO-FFI E的汇聚Spearman相关和与N/O/A/C的区分Spearman相关。正式运行省略`--respondent-limit`；联调可先加`--respondent-limit 5`。若中断，使用同一个`--output`目录恢复已落盘的调用。

使用同一批旧实验100人运行summary-only对比条件：

```powershell
python -X utf8 -m experiments.system_comparison legacy-abc `
  --experiment experiment_data/exp_20260911_155055_92946884 `
  --persona-mode summary_only `
  --summary-pool experiment_data/legacy_persona_summary_pool `
  --respondent-source legacy_100
```

默认另存到`<实验目录>/legacy_summary_only_abc/`，不会覆盖原来的`items_plus_summary`结果。两种输入条件比较时应保持被试ID、A/B/C问卷、作答模型和计分完全相同。`--respondent-source all_285`可以让全部285人作答，但它不再与原100人结果构成严格配对比较，且模型调用成本会显著增加。

### 主观后果—概率抽样条件

`summary_embodied_probability`只把统一行为总结交给SJT作答模型，要求模型考虑该被试亲自采取每个行为时的主观时间、精力、情绪、人际和机会成本，并输出A—D概率。程序不会取最大概率或期望分，而是使用冻结种子进行一次分类抽样，再按原问卷计分键计分。原始概率、归一化概率、随机数、抽样键和最终选项全部保存。NEO-FFI在独立调用中只接收summary-only画像，不接收SJT后果模拟指令或作答历史。

使用全部285人运行A/B/C及共享NEO-FFI：

```powershell
python -X utf8 -m experiments.system_comparison legacy-abc `
  --experiment experiment_data/exp_20260911_155055_92946884 `
  --persona-mode summary_embodied_probability `
  --summary-pool experiment_data/legacy_persona_summary_pool `
  --respondent-source all_285 `
  --sampling-seed 20260913
```

默认输出到`<实验目录>/legacy_summary_embodied_probability_abc/`，支持按被试—题目断点续跑。除原有α、CITC、选项分布和NEO汇聚/区分相关外，还输出平均最大选择概率、标准化概率熵和近确定性概率比例；这些概率统计只描述模型自报的不确定性，不属于心理测量信效度。

### 虚拟信效度负对照

完成旧虚拟被试复测后，可直接复用已保存的A/B/C与NEO-FFI作答，检查异常高指标来自计算故障还是虚拟作答结构。此命令不调用模型，也不修改原始结果：

```powershell
python -X utf8 -m experiments.system_comparison negative-controls `
  --evaluation experiment_data/exp_20260911_155055_92946884/legacy_pool_abc `
  --replications 500
```

负对照包括被试ID错配、每题独立打乱作答、随机错配计分键、随机反向一半题目，以及把单题复制16次。整体反转全部题目只会反转相关符号而不会降低绝对值，因此不把它误当作有效的负对照。结果保存到`legacy_pool_abc/diagnostics/negative_control_<时间>_<编号>/report.html`，并同时导出每次模拟、汇总分布和诊断检查CSV。

### 质量梯度敏感性实验

负对照通过后，可以从全部题目均被试内错配的状态开始，按嵌套顺序逐步恢复原始题目，检查alpha、SJT与NEO-E的汇聚相关及区分效度差值是否随完整题目比例稳定上升：

```powershell
python -X utf8 -m experiments.system_comparison sensitivity-curve `
  --evaluation experiment_data/exp_20260911_155055_92946884/legacy_pool_abc `
  --replications 500
```

默认完整题目比例为0%、12.5%、25%、50%、75%和100%，每条随机恢复路径使用相同的嵌套题目顺序。结果保存到`legacy_pool_abc/diagnostics/sensitivity_curve_<时间>_<编号>/`。该实验不调用模型、不修改原始作答；曲线只能证明指标对人工信号强度的敏感性，不能代替真人效度证据，也不能冒充C方法真实返修曲线。

测试额外需要`pytest`；正式运行入口不依赖它。 

```powershell
python -X utf8 -m pytest tests/test_system_comparison.py -q
python -X utf8 -m pytest tests/test_combination_search.py -q
```

测试用替身代替收费模型，实际运行施测保存、当前问卷计分、四门槛计算、缓存复用、检查点恢复及HTML汇总。运行数据目录被Git忽略；只对本实验源码、示例配置、README和专用测试设置追踪例外。

## Mussel已知选项缺陷敏感性实验

该实验检验当前具身概率虚拟被试能否发现人为植入的严重题目缺陷。刺激固定为Mussel五个facet各2题：原题不变；缺陷版保持情境、C/D及A/B=1、C/D=0计分键不变，只把A/B改成低特质行为，使四个选项都缺少真正的高水平端点。

同一批100名冻结虚拟被试分别完成10道原题和10道缺陷题，每个题目版本均为独立模型调用。NEO-FFI复用这批被试已在独立会话完成的得分；同facet其余21道原始Mussel作答只作为固定CITC锚点。正式运行约调用模型2,000次：

```powershell
python -X utf8 -m experiments.system_comparison mussel-item-defect `
  --experiment experiment_data/exp_20260911_155055_92946884 `
  --model deepseek-flash-guan
```

默认输出到`<实验目录>/mussel_item_defect_pilot/high_option_absence/`，相同命令支持按被试—题目断点续跑。报告包含逐题目标NEO相关、构念特异度、固定21题锚点CITC、高低27%组区分以及选项/概率分布。主分析使用模型报告的`P(A)+P(B)`以减少一次概率抽样的噪声，实际抽样0/1结果同时保留。预设探索性检出规则为4项均可估计且至少3项满足“原题高于缺陷题”。这只是已知严重缺陷的计算机内部敏感性，不是通用题目质量标准，也不能证明指标与真人一致。

```powershell
python -X utf8 -m pytest tests/test_mussel_item_defect_pilot.py -q
```

现有全库测试中，`test_defer_fallback_does_not_override_forced_vts_trigger`与已提交的Forced VTS降级defer规则存在冲突；这不是本实验新增的规则。本实现不擅自更改该业务决策，也不把它报告为全库测试通过。
