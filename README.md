# LangGraph SJT

基于 LangGraph 和大语言模型的构念驱动人格情境判断测验（Personality SJT）开发平台。

当前版本面向研究与教学用途，首版主链支持已经注册的 NEO-PI-R facet。系统把人格构念、行为证据、目标人群情境、题目设计、虚拟施测和确定性统计连接起来，用于快速构建和诊断一套 PSJT。虚拟被试结果属于开发期模拟证据，不能替代真实被试研究。

## 当前流程

```text
需求规格
  ↓
构念预检
  ↓
Curated Behavior Evidence
  ↓
Behavior Expansion（按目标人群缓存）
  ↓
Blueprint / 双向细目表（每个测量单元先生成2个候选）
  ↓
Skeleton + Item Writer
  ↓
首轮内容审查
  ↓
固定虚拟被试施测
  ↓
确定性题项与测验统计
  ↓
异常诊断与逐题原子返修（可选）
  ↓
冻结候选题库、组卷与报告
```

正式运行由程序负责状态迁移、JSON 校验、ID/引用、题量、版本和统计计算；模型负责构念内容的生成、题目语言实现、内容审查和异常诊断。当前版本不包含 UI 重构和真实被试数据导入分析。

## 环境要求

- Windows 10/11，或 Ubuntu 22.04 LTS 等主流 Linux 发行版
- Python 3.11（推荐使用与项目开发环境相同的版本）
- 一个支持 OpenAI 兼容 Chat Completions 接口的模型服务
- 可用的模型 API Key

## 安装

在 PowerShell 中执行：

```powershell
git clone <你的 GitHub 仓库地址>
cd langgraph_for_SJT

python -m venv env
env\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

如果 PowerShell 不允许激活脚本，可以只对当前窗口放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
env\Scripts\activate
```

也可以不激活环境，直接使用虚拟环境中的 Python：

```powershell
env\Scripts\python.exe -m pip install -r requirements.txt
```

在 Ubuntu/Linux 服务器中执行：

```bash
git clone <你的 GitHub 仓库地址>
cd langgraph_for_SJT

python3 -m venv env
source env/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

`requirements.txt` 同时包含主流程、实验绘图和 Word 报告所需的运行依赖；
本地测试工具单独列在 `requirements-dev.txt`。

## 配置模型服务

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后编辑项目根目录下的 `.env`。至少填写：

```env
API_KEY=你的模型服务密钥
MODEL_ID=你的模型名称
```

如果服务不是默认的 OpenAI 接口，需要填写兼容接口地址：

```env
BASE_URL=https://你的服务地址/v1
```

常用的可选配置：

```env
MODEL_REQUEST_TIMEOUT_SECONDS=120
MODEL_REQUEST_MAX_ATTEMPTS=2
STRUCTURED_OUTPUT_METHOD=plain_json

# 虚拟被试后的高推理诊断和修题模型
PSYCHOMETRIC_DIAGNOSIS_MODEL_ID=deepseek-v4-pro-guan
PSYCHOMETRIC_DIAGNOSIS_THINKING=enabled
PSYCHOMETRIC_DIAGNOSIS_REASONING_EFFORT=high
PSYCHOMETRIC_ITEM_REPAIR_MODEL_ID=deepseek-v4-pro-guan
PSYCHOMETRIC_ITEM_REPAIR_THINKING=enabled
PSYCHOMETRIC_ITEM_REPAIR_REASONING_EFFORT=high
```

不要把真实密钥提交到 Git。`.env` 已被 `.gitignore` 忽略，仓库中只保留 `.env.example`。

## 启动方式

### Streamlit 工作台（推荐）

```powershell
python -m streamlit run app.py
```

浏览器打开 Streamlit 显示的地址，根据页面提示填写需求并运行流程。运行过程中生成的 checkpoint 和报告保存在 `outputs/` 下。

### 命令行调试入口

```powershell
python cli_app.py
```

`cli_app.py` 是开发和故障排查入口，适合检查 checkpoint 恢复、模型输出和流程节点；正式使用优先使用 Streamlit 工作台。命令行入口中的示例需求可能是固定的，若要运行自定义需求，请使用工作台或修改对应的调试配置。

### 独立 A/B/C 系统比较实验

以下命令新建并运行一次独立实验：

```powershell
python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/example_config.json
```

每次运行都会在 `experiment_data/exp_<时间>_<编号>/` 下建立独立目录，分别保存
A简单提示词问卷、B理论组卷、C闭环迭代问卷、虚拟作答、整卷指标和报告。
中断后使用终端输出的实验目录恢复：

```powershell
python -X utf8 -m experiments.system_comparison resume --experiment "experiment_data/exp_<时间>_<编号>"
```

只根据已有结果重建报告：

```powershell
python -X utf8 -m experiments.system_comparison report --experiment "experiment_data/exp_<时间>_<编号>"
```

当前入口一次运行一个完整 A/B/C 实验，不是无人值守的多实验批量调度器。
选择 Agent 自动开发模式后，题目生成、审查、虚拟施测、返修、同槽位补题和组卷可以自动推进，
但蓝图审核、施测后继续处理以及自动补题耗尽等情况仍可能等待人工输入。

### 服务器并行运行五个 Mussel facet

可为下面五个目标 facet 分别复制一份 `example_config.json`：

| 代码 | `target_facet` | 建议同域参照 | 建议跨域参照 |
|---|---|---|---|
| N4 自我意识 | `neuroticism_self_consciousness` | `neuroticism_anxiety` | `extraversion_gregariousness` |
| E2 乐群性 | `extraversion_gregariousness` | `extraversion_warmth` | `neuroticism_anxiety` |
| O5 观念开放 | `openness_ideas` | `openness_actions` | `conscientiousness_self_discipline` |
| A4 顺从 | `agreeableness_compliance` | `agreeableness_trust` | `neuroticism_angry_hostility` |
| C5 自律 | `conscientiousness_self_discipline` | `conscientiousness_deliberation` | `neuroticism_impulsiveness` |

五个完整实验可以在五个独立 `tmux` 会话中运行。每份配置建议设定不同 `seed`，并将
`max_concurrency` 设为 `2`，使五进程总模型并发约为10；不要把每进程并发设为5后直接启动
五个进程，否则容易触发模型服务限流或503错误。示例：

```bash
tmux new-session -d -s sjt_n4 "source env/bin/activate && python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/five_facet_configs/N4_self_consciousness.json 2>&1 | tee logs/N4_self_consciousness.log"
tmux new-session -d -s sjt_e2 "source env/bin/activate && python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/five_facet_configs/E2_gregariousness.json 2>&1 | tee logs/E2_gregariousness.log"
tmux new-session -d -s sjt_o5 "source env/bin/activate && python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/five_facet_configs/O5_ideas.json 2>&1 | tee logs/O5_ideas.log"
tmux new-session -d -s sjt_a4 "source env/bin/activate && python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/five_facet_configs/A4_compliance.json 2>&1 | tee logs/A4_compliance.log"
tmux new-session -d -s sjt_c5 "source env/bin/activate && python -X utf8 -m experiments.system_comparison run --config experiments/system_comparison/five_facet_configs/C5_self_discipline.json 2>&1 | tee logs/C5_self_discipline.log"
```

这些命令假定五份配置已放在 `experiments/system_comparison/five_facet_configs/`。
使用 `tmux attach -t sjt_n4` 等命令进入对应会话处理可能出现的交互提示。

## Behavior Evidence 数据

正式运行优先读取已经整理好的证据库：

```text
knowledge_base/evidence_library/<facet_id>/stage2_evidence_library_curated.json
```

这些文件提供可观察行为维度、高低表现、边界条件和来源条目。`docs/ipip.md` 与 `knowledge_base/items/ipip_neo_items.json` 是 IPIP/NEO 条目资料，不会在每道题的热路径中重新抽取行为证据。运行时缺少已支持 facet 的正式证据时，应先补齐证据库；系统不会在正式题目生成过程中临时捏造上游构念内容。

离线准备或检查行为证据时，可使用：

```powershell
# 将 docs/ipip.md 解析为结构化 IPIP 题库
python -m sjt_system.knowledge.behavior_evidence_cli parse-ipip

# 为一个 facet 离线生成候选行为证据
python -m sjt_system.knowledge.behavior_evidence_cli build --facet A4

# 为全部已注册 facet 离线生成候选证据
python -m sjt_system.knowledge.behavior_evidence_cli build --facet ALL

# 查看一个证据文件
python -m sjt_system.knowledge.behavior_evidence_cli show <证据文件路径>
```

离线 `build` 生成的是候选文件，正式运行使用前应放入对应的 curated 证据目录，并确保字段和来源完整。

## 项目结构

```text
langgraph_for_SJT/
├─ app.py                         # Streamlit 启动入口
├─ cli_app.py                     # 命令行调试入口
├─ requirements.txt               # 运行时依赖
├─ requirements-dev.txt           # 运行时依赖 + 本地测试工具
├─ .env.example                   # 环境变量模板（不含密钥）
│
├─ sjt_system/
│  ├─ authoring/                  # 需求、构念、行为证据、Expansion、蓝图、题目
│  ├─ evaluation/                 # 虚拟被试、施测、题项统计、诊断与返修
│  ├─ delivery/                   # 冻结题库、组卷、交付与报告
│  ├─ agent/                      # 模型客户端、JSON 解析、重试和 Agent 工厂
│  ├─ workflow/                   # LangGraph 图、执行器、节点与路由
│  ├─ runtime/                    # checkpoint、运行清单、日志与进度
│  ├─ prompt/                     # 结构化生成、审查、诊断和修题提示词
│  ├─ knowledge/                  # 构念/行为证据加载与离线构建工具
│  └─ ui/                         # Streamlit 页面
│
├─ knowledge_base/
│  ├─ evidence_library/           # 正式使用的 curated Behavior Evidence
│  └─ items/                      # IPIP/NEO 条目结构化资料
│
├─ docs/                          # 研究资料、构念资料和说明文档
├─ experiments/system_comparison/ # 独立A/B/C比较实验源码
├─ tools/                         # 本地分析和报告工具
└─ outputs/                       # 运行时生成，不提交到 Git
```

## 运行产物

运行产物默认写入 `outputs/`，主要包括：

```text
outputs/
├─ run_checkpoints/               # 可恢复运行的 checkpoint
├─ virtual_responses/             # 虚拟被试逐题作答记录
├─ assembled_tests/               # 组卷中间结果
├─ final_reports/                 # 正式测验、题库和技术报告
├─ run_telemetry/                 # 每轮模型 Token、调用次数和耗时记录
└─ behavior_evidence_candidates/  # 离线行为证据候选
```

最终报告目录通常包含：

- `final_test.json`：冻结后的正式测验；
- `item_database.json`：带来源链、版本和评分信息的题库；
- `technical_report.md`：流程、版本和统计结果；
- `virtual_respondent_report.json`：开发期虚拟施测报告。

当前整卷迭代采用与 A/B/C 独立评估一致的指标口径：Cronbach's α 和虚拟重测
ICC 均为最低 `.80` 的门槛；目标 IPIP facet 高低组的 Hedges' g 是主要优化目标；
目标相关减最大绝对非目标相关得到的 `Δmin` 不得下降；SJT 与目标 IPIP facet 的
Spearman 相关最多允许相对历史最佳下降 `.02`。旧的目标恢复 R²、构念选择性和 Q
只保留为历史诊断字段，不再决定新运行中的最优卷或平台期。

这些统计反映同一虚拟被试框架下的开发期表现，不等同于真实被试信效度。
技术报告还会记录题目数量、累计 Token 和累计模型耗时。
稳定性指标需要 target 组额外完成一次整卷重测，因此每轮会增加约
`target组人数 × 候选题数` 次模型调用；虚拟重测ICC只作为稳定性门槛，
这些调用计入同一轮Token与耗时。

## 恢复中断运行

如果运行中断，优先从同一运行的 checkpoint 恢复，不要直接删除 `outputs/run_checkpoints/`。恢复时应继续使用相同的 `.env`、模型配置、随机种子和输入需求，否则可能无法复用原有作答或版本指纹。

## 本地检查

如果本地保留测试文件，可以运行：

```powershell
python -m pytest -q
```

当前仓库策略会忽略本地测试文件、运行输出、checkpoint、环境文件、`wiki/` 和研究导出文件。
其中 `.env`、`outputs/`、`experiment_data/` 与 `blind_classification/` 均只保留在本地，
不会随常规 Git 提交上传。`experiments/system_comparison/` 的程序源码属于仓库内容，
但其生成的实验数据仍写入被忽略的 `experiment_data/`。

## 常见问题

### `Missing API_KEY environment variable`

确认根目录存在 `.env`，并且其中包含非空的 `API_KEY`。修改 `.env` 后重新启动 Streamlit 或命令行进程。

### 模型返回 JSON 校验失败

确认 `BASE_URL`、`MODEL_ID` 和 `STRUCTURED_OUTPUT_METHOD` 与模型服务兼容。DeepSeek 兼容接口通常先使用：

```env
STRUCTURED_OUTPUT_METHOD=plain_json
```

### PowerShell 无法激活虚拟环境

使用上文的 `Set-ExecutionPolicy -Scope Process Bypass`，或者直接调用 `env\Scripts\python.exe`。

### 找不到行为证据或构念不支持

检查 `knowledge_base/evidence_library/` 中是否存在对应 facet 的 curated 文件。正式运行只接受已注册且有合法证据来源的构念，不会把未知构念自动映射到相近构念。

## 安全与提交前检查

提交代码前确认：

```powershell
git status --short
git diff -- . ':!outputs'
```

不要提交 `.env`、API 密钥、`outputs/`、`experiment_data/`、`blind_classification/`、
真实或敏感被试数据，以及其他本地研究导出文件。
