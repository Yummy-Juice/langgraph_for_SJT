"""Generation-only adapters. B receives an explicit theory-only allowlist."""
from copy import deepcopy
import json
from string import Template

from langchain_core.messages import HumanMessage, SystemMessage
from sjt_system.agent.client import get_model
from sjt_system.agent.json_parsing import parse_model_json_response
from sjt_system.runtime.telemetry import job_context

from .config import fingerprint

A_PROMPT_VERSION = "a-zero-shot-four-level-v2"
A_PROMPT = """请你作为一名心理测量专家，根据以下构念定义和要求，编制一套包含 ${item_count} 道全新题目的人格情境判断测验，测量 ${population} 的【${construct_name}】水平。

一、构念定义
【${construct_name}】属于【${domain_name}】。
${definition}

高水平表现：
${high_behavior}

低水平表现：
${low_behavior}

需要区分的相近构念及边界：
${confounds}

现有构念资料中的情境与选项约束：
${constraints}

二、编制要求
1. 情境设计
（1）使用适合目标人群的日常生活、学习或常见工作情境。
（2）情境应具体、自然，提供理解行为选择所必需的信息。
（3）每道题主要反映目标构念，避免让其他人格特质、专业知识或能力成为作答的主要依据。
（4）不同题目应具有不同的事件、行为选择或决策条件，不要仅通过更换人物、地点和措辞生成重复题。
（5）每道题统一提问：“你会怎么做？”

2. 选项设计与计分
（1）每道题提供A、B、C、D四个选项，都应现实、可理解，并回应同一情境中的行为选择。
（2）四个选项分别体现目标特质的四个可区分水平：1分为低水平，2分为较低水平，3分为较高水平，4分为高水平。
（3）每题的1、2、3、4分各使用一次，不允许两个选项得分相同。
（4）分数越高，只表示目标特质越高，不表示行为越正确、越健康或越受社会欢迎。
（5）通过具体行为差异体现特质程度，不要仅添加“非常”“比较”等程度词。
（6）避免某个选项明显更礼貌、更负责、更有能力或更符合道德规范，使选择主要取决于社会赞许。
（7）低特质选项不应被写成明显不合理或恶劣的行为，高特质选项也不应被写成理想化的标准答案。
（8）选项表述清晰、长度大致均衡，不直接标注高低水平或透露分值。
（9）A、B、C、D与分数的对应顺序可以变化，以scoring_key为准。

3. 语言要求
使用流畅、自然且语法正确的中文，避免歧义、双重否定、过度专业化表达和不必要的人口特征限定。

三、简短设计依据
每道题提供一句情境设计依据，每个选项提供一句计分依据，将具体行为与目标构念水平相联系。
这些依据只用于记录设计意图，不是实际信度、效度或题目质量的证据。
不要编造施测结果、心理测量指标或质量评分，不需要输出逐步思考过程。
本任务为无示例单次生成，不使用虚拟被试结果或后续组卷结果修改题目。

四、输出要求
仅输出一个合法JSON对象，不要添加Markdown代码块或JSON之外的文字。
输出结构如下（仅说明字段，不是参考题目）：
{
  "items": [
    {
      "scenario": "情境描述，不包含选项和计分说明",
      "response_instruction": "你会怎么做？",
      "response_options": [
        {"option_id": "A", "text": "选项A的具体行为"},
        {"option_id": "B", "text": "选项B的具体行为"},
        {"option_id": "C", "text": "选项C的具体行为"},
        {"option_id": "D", "text": "选项D的具体行为"}
      ],
      "scoring_key": {"A": 1, "B": 2, "C": 3, "D": 4}
    }
  ],
  "design_rationales": [
    {
      "item_number": 1,
      "scenario_rationale": "简短的情境设计依据",
      "option_rationales": {"A": "计分依据", "B": "计分依据", "C": "计分依据", "D": "计分依据"}
    }
  ]
}
items和design_rationales都必须恰好包含 ${item_count} 条，按相同顺序一一对应，item_number从1开始连续编号。
设计依据与正式题目分开，不能写入情境或选项文字。scoring_key必须与实际选项内容一致。
"""


def build_a_prompt(config, profile):
    """Render only the configured construct; no exemplar or virtual data input."""
    facets = [f for f in profile.get("facets", []) if f.get("facet_id") == config.target_facet]
    if len(facets) != 1:
        raise ValueError("A提示词需要唯一且匹配配置的目标构念")
    facet = facets[0]
    for name in ("definition", "high_behavior", "low_behavior"):
        if not isinstance(facet.get(name), str) or not facet[name].strip():
            raise ValueError(f"A提示词缺少目标构念资料：{name}")
    constraints = [str(rule) for name in ("inappropriate_conditions", "forbidden_patterns", "option_design_rules", "hard_constraints")
                   for rule in facet.get(name, [])]
    return Template(A_PROMPT).substitute(
        item_count=config.final_item_count, population=config.target_population,
        construct_name=facet.get("facet_name") or config.target_facet,
        domain_name=profile.get("domain_name") or profile["domain_id"],
        definition=facet["definition"], high_behavior=facet["high_behavior"], low_behavior=facet["low_behavior"],
        confounds="\n".join(facet.get("common_confounds") or []) or "现有资料未列出额外区分提醒。",
        constraints="\n".join(constraints) or "现有资料未列出额外约束。",
    )

B_PROMPT = """你是理论指导的测验组卷专家。根据构念定义、行为证据、内容审查结果和题目文字，
从候选题中选出符合蓝图保留量的测验。兼顾构念纯度、覆盖、情境多样性，避免重复与社会赞许线索。
没有虚拟作答或心理测量指标，不得推测、编造这些指标。
只返回JSON：{"selected_item_ids":["实际题号"],"rationale":"理论与内容选择理由"}。
每个蓝图单元的选题数必须等于planned_retention_count。"""


def _validate_items(items, count):
    if not isinstance(items, list) or len(items) != count:
        raise ValueError(f"需要恰好{count}题")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("题目必须是对象")
        for key in ("scenario", "response_instruction"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"题目缺少{key}")
        options = item.get("response_options")
        if (not isinstance(options, list) or len(options) != 4 or
            any(not isinstance(o, dict) for o in options) or
            {o.get("option_id") for o in options} != set("ABCD") or
            any(not isinstance(o.get("text"), str) or not o["text"].strip() for o in options)):
            raise ValueError("每题必须包含A/B/C/D四个非空选项")
        scores = item.get("scoring_key")
        if (not isinstance(scores, dict) or set(scores) != set("ABCD") or
            any(type(v) is not int for v in scores.values()) or set(scores.values()) != {1, 2, 3, 4}):
            raise ValueError("计分必须是1/2/3/4各一次")
        identity = fingerprint({"scenario": item["scenario"], "options": options})
        if identity in seen:
            raise ValueError("存在完全重复题目")
        seen.add(identity)


def validate_a_form(payload, *, count, facet):
    items = payload.get("items") if isinstance(payload, dict) else None
    _validate_items(items, count)
    result = []
    for index, raw in enumerate(items, 1):
        result.append({"item_id": f"A-{index:03d}", "version": 1,
                       "target_dimension_id": facet, "blueprint_cell_id": f"A-cell-{index:03d}",
                       **{key: deepcopy(raw[key]) for key in
                          ("scenario", "response_instruction", "scoring_key")},
                       "response_options": [{"option_id": o["option_id"], "text": o["text"]}
                                            for o in raw["response_options"]]})
    return result


def validate_a_design_rationales(payload, items):
    """Validate explanation structure, not its scientific or semantic truth."""
    rows = payload.get("design_rationales") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError("设计依据必须与题目数量一致，并逐题对应")
    result = []
    for index, (row, item) in enumerate(zip(rows, items), 1):
        if not isinstance(row, dict) or type(row.get("item_number")) is not int or row["item_number"] != index:
            raise ValueError("设计依据的item_number必须从1开始按题目顺序连续编号")
        scenario = row.get("scenario_rationale")
        options = row.get("option_rationales")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError(f"第{index}题缺少非空情境设计依据")
        if (not isinstance(options, dict) or set(options) != set("ABCD") or
            any(not isinstance(v, str) or not v.strip() for v in options.values())):
            raise ValueError(f"第{index}题的设计依据必须包含A/B/C/D四个非空选项说明")
        result.append({"item_number": index, "item_id": item["item_id"], "item_version": item["version"],
                       "scoring_key": deepcopy(item["scoring_key"]), "scenario_rationale": scenario.strip(),
                       "option_rationales": {k: v.strip() for k, v in options.items()}})
    return {"schema_version": 1, "prompt_version": A_PROMPT_VERSION,
            "form_fingerprint": fingerprint(items), "items": result}


def theory_payload(state):
    profile = state.get("construct_profile") or {}
    return {
        "construct": {k: deepcopy(profile.get(k)) for k in ("domain_id", "domain_name", "facets")},
        "blueprint": {"cells": [
            {k: deepcopy(cell.get(k)) for k in
             ("cell_id", "facet_id", "behavior_id", "activation_mechanism", "domain", "actor_relation", "event_class", "planned_retention_count")}
            for cell in (state.get("blueprint") or {}).get("cells", [])]},
        "candidates": [
            {k: deepcopy(item.get(k)) for k in
             ("item_id", "blueprint_cell_id", "target_dimension_id", "scenario", "response_instruction", "response_options", "scoring_key")}
            for item in state.get("frozen_item_bank", [])],
    }


def validate_b_selection(payload, state):
    ids = payload.get("selected_item_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or any(not isinstance(v, str) for v in ids) or len(ids) != len(set(ids)):
        raise ValueError("组卷题号必须唯一")
    bank = {i["item_id"]: i for i in state["frozen_item_bank"]}
    if any(i not in bank for i in ids):
        raise ValueError("组卷包含不存在的题号")
    cells = state["blueprint"]["cells"]
    if len(ids) != sum(c["planned_retention_count"] for c in cells):
        raise ValueError("组卷题量不符合蓝图")
    for cell in cells:
        if sum(bank[i]["blueprint_cell_id"] == cell["cell_id"] for i in ids) != cell["planned_retention_count"]:
            raise ValueError(f"蓝图单元保留量不符：{cell['cell_id']}")
    selected = [deepcopy(bank[i]) for i in ids]
    _validate_items(selected, len(ids))
    return selected


async def _generate(store, key, system, payload, validate, model=None, *, details_validator=None):
    request = {"system": system, "payload": payload, "model": store.config.model_id}
    previous_request = store.read(f"{key}/request.json")
    if previous_request is not None and fingerprint(previous_request) != fingerprint(request):
        raise ValueError("已保存的生成提示词或输入发生变化，请新建实验")
    store.write(f"{key}/request.json", request)
    model = model or get_model(store.config.model_id)
    # All persisted responses, including a valid one preceding an interruption,
    # are reparsed locally before making another paid request.
    for attempt in range(1, 4):
        path = f"{key}/attempt_{attempt:02d}.json"
        saved = store.read(path)
        if saved is None:
            try:
                messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
                if attempt > 1:
                    previous = store.read(f"{key}/attempt_{attempt - 1:02d}.json", {})
                    messages.append(HumanMessage(content="仅修正结构错误并重新输出完整JSON：" + previous.get("error", "格式不正确")))
                with job_context(f"experiment_{key.split('/')[0]}", attempt=attempt):
                    response = await model.ainvoke(messages)
                saved = {"content": response.content}
            except Exception as exc:
                saved = {"error": str(exc), "transport_error": True}
            store.write(path, saved)
        try:
            if saved.get("transport_error"):
                raise ValueError(saved["error"])
            payload_out = parse_model_json_response(saved["content"])
            items = validate(payload_out)
            details = details_validator(payload_out, items) if details_validator else None
            store.write(f"{key}/decision.json", payload_out)
            # Persist explanations before the form's completion marker. A normal
            # resume reparses the saved response; it does not pay for new text.
            if details is not None:
                store.write(f"{key}/design_rationales.json", details)
            store.save_form(key, items)
            store.write(f"{key}/checkpoint.json", {"status": "frozen", "request_hash": fingerprint(request)})
            return items
        except (ValueError, TypeError, KeyError) as exc:
            saved["error"] = str(exc)
            store.write(path, saved)
    raise ValueError(f"{key}结构校验/请求失败，已保存3次尝试；不会按虚拟指标重新生成")


async def generate_a(store, model=None):
    profile = store.read("shared/construct_profile.json")
    system = build_a_prompt(store.config, profile)
    payload = {"prompt_version": A_PROMPT_VERSION,
               "content_blueprint": store.read("shared/content_blueprint.json")}
    return await _generate(store, "A/round_01", system, payload,
                           lambda p: validate_a_form(p, count=store.config.final_item_count,
                                                     facet=store.config.target_facet), model,
                           details_validator=validate_a_design_rationales)


async def generate_b(store, shared_state, model=None):
    return await _generate(store, "B/round_01", B_PROMPT, theory_payload(shared_state),
                           lambda p: validate_b_selection(p, shared_state), model)
