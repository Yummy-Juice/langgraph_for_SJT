"""Interactive creator for a new virtual-respondent prompt method."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .virtual_prompt_registry import DEFAULT_PROMPT_ROOT, load_prompt_specs


def _ask(text: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{text}{suffix}：").strip()
        if value:
            return value
        if default is not None:
            return default
        print("此项不能为空。", flush=True)


def _ask_choice(text: str, choices: dict[str, str], *, default: str) -> str:
    print(text, flush=True)
    for key, label in choices.items():
        print(f"{key}. {label}", flush=True)
    while True:
        value = input(f"请选择 [{default}]：").strip() or default
        if value in choices:
            return value
        print("输入无效，请重新选择。", flush=True)


def _make_template(persona_input: str, extra_parameter: bool, instruction: str) -> str:
    if persona_input in {"score_values", "five_mussel_facets"}:
        persona_block = "[PERSONALITY SCORES]\n{score_lines}\n[/PERSONALITY SCORES]"
    else:
        persona_block = "[BEHAVIORAL PROFILE]\n{persona_summary}\n[/BEHAVIORAL PROFILE]"
    parameter_block = (
        "\n\n[ADDITIONAL PERSONAL PARAMETERS]\n"
        "{extra_parameters}\n[/ADDITIONAL PERSONAL PARAMETERS]"
        if extra_parameter else ""
    )
    return (
        "从现在起，你就是下述虚拟被试。请保持人格信息在不同题目之间的一致性，"
        "但不要机械地每次选择相同答案。\n\n"
        f"{persona_block}{parameter_block}\n\n"
        "请从第一人称立场理解每道题，不要分析研究者意图、构念名称、计分方向或正确答案。\n"
        f"{instruction}"
    )


def _next_prompt_id(existing_ids: set[str]) -> str:
    numbers = []
    for prompt_id in existing_ids:
        match = re.fullmatch(r"custom_prompt_v(\d+)", prompt_id)
        if match:
            numbers.append(int(match.group(1)))
    next_number = max(numbers, default=0) + 1
    return f"custom_prompt_v{next_number}"


def create_prompt_interactively(root: Path = DEFAULT_PROMPT_ROOT) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = load_prompt_specs(root)

    print("\n===== 新建虚拟被试方法 =====", flush=True)
    label = _ask("菜单显示名称（可以填写中文，例如：觉醒水平人格法）")
    prompt_id = _next_prompt_id(set(existing))
    print(f"系统已自动生成方法ID：{prompt_id}", flush=True)
    description = _ask("方法简介")
    persona_information = _ask("提供给虚拟被试的人格信息")
    response_behavior = _ask("虚拟被试的作答方式")
    persona_input = {
        "1": "score_values",
        "2": "five_mussel_facets",
        "3": "persona_summary",
    }[_ask_choice(
        "这套方法使用哪类人格输入？",
        {
            "1": "显式人格分数",
            "2": "五个Mussel facet分数",
            "3": "行为化人格总结",
        },
        default="1",
    )]
    sjt_mode = {
        "1": "single_selection",
        "2": "choice_probability",
    }[_ask_choice(
        "SJT如何输出？",
        {"1": "直接选择A/B/C/D", "2": "输出四个选项的概率"},
        default="1",
    )]

    add_parameter = _ask_choice(
        "是否增加额外人格参数？",
        {"1": "是（可选，例如个人觉醒水平）", "2": "否"},
        default="2",
    ) == "1"
    extra_parameters: list[dict[str, str]] = []
    parameter_values_file: str | None = None
    if add_parameter:
        key = _ask("参数内部名称，例如 awakening_level")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("参数内部名称只能包含英文、数字、下划线或连字符")
        extra_parameters.append({
            "key": key,
            "label": _ask("参数显示名称，例如个人觉醒水平"),
            "description": _ask("参数含义"),
            "default": _ask("默认值，例如50"),
        })
        per_person = _ask_choice(
            "这个参数是否需要每个被试单独设置？",
            {"1": "是，使用被试参数值文件", "2": "否，所有被试使用默认值"},
            default="2",
        ) == "1"
        if per_person:
            parameter_values_file = _ask(
                "参数值文件名",
                default=f"{prompt_id}_parameters.json",
            )
            values_path = root / parameter_values_file
            if values_path.exists():
                raise FileExistsError(f"参数值文件已存在：{values_path}")
            values_path.write_text("{}\n", encoding="utf-8")
            print(
                f"已创建参数值文件：{values_path.name}；"
                "后续按 respondent_id 填写每名被试的参数。",
                flush=True,
            )

    instruction = _ask(
        "补充到提示词末尾的作答要求",
        default="请根据上述人格信息和额外参数，完成接下来每道情境题。",
    )
    payload = {
        "prompt_id": prompt_id,
        "label": label,
        "description": description,
        "persona_information": persona_information,
        "response_behavior": response_behavior,
        "persona_input": persona_input,
        "sjt_response_mode": sjt_mode,
        "ipip_response_mode": "likert_1_5",
        "version": f"{prompt_id}-v1",
        "extra_parameters": extra_parameters,
        "parameter_values_file": parameter_values_file,
        "template": _make_template(persona_input, add_parameter, instruction),
    }
    output = root / f"{prompt_id}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"新方法已保存：{output.name}", flush=True)
    return output


if __name__ == "__main__":
    create_prompt_interactively()
