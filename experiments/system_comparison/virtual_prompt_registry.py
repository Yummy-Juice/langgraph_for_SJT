"""Registry for selectable virtual-respondent prompt variants.

The registry is deliberately separate from the existing score-profile logic.
Adding a prompt variant does not alter the default prompt or any existing
experiment checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parent / "virtual_prompt_specs"


@dataclass(frozen=True)
class VirtualPromptSpec:
    prompt_id: str
    label: str
    description: str
    persona_information: str
    response_behavior: str
    persona_input: str
    sjt_response_mode: str
    ipip_response_mode: str
    template: str
    source: str
    version: str
    extra_parameters: tuple[dict[str, Any], ...] = ()
    parameter_values_file: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], *, source: Path) -> "VirtualPromptSpec":
        required = (
            "prompt_id",
            "label",
            "description",
            "persona_information",
            "response_behavior",
            "persona_input",
            "sjt_response_mode",
            "ipip_response_mode",
            "template",
            "version",
        )
        missing = [
            key
            for key in required
            if key != "template" and not str(value.get(key, "")).strip()
        ]
        if "template" not in value:
            missing.append("template")
        if missing:
            raise ValueError(f"提示词配置缺少字段：{', '.join(missing)}；文件={source}")
        prompt_id = str(value["prompt_id"]).strip()
        if any(char.isspace() for char in prompt_id):
            raise ValueError(f"prompt_id不能包含空白字符：{prompt_id!r}")
        return cls(
            prompt_id=prompt_id,
            label=str(value["label"]).strip(),
            description=str(value["description"]).strip(),
            persona_information=str(value["persona_information"]).strip(),
            response_behavior=str(value["response_behavior"]).strip(),
            persona_input=str(value["persona_input"]).strip(),
            sjt_response_mode=str(value["sjt_response_mode"]).strip(),
            ipip_response_mode=str(value["ipip_response_mode"]).strip(),
            template=str(value["template"]),
            source=str(source),
            version=str(value["version"]).strip(),
            extra_parameters=tuple(
                item for item in (value.get("extra_parameters") or [])
                if isinstance(item, dict)
            ),
            parameter_values_file=(
                str(value["parameter_values_file"]).strip()
                if value.get("parameter_values_file") else None
            ),
        )

    def render(
        self,
        *,
        persona_summary: str = "",
        score_lines: str = "",
        extra_parameters: str = "",
    ) -> str:
        """Render only the persona layer; task instructions are added later."""

        return self.template.format(
            persona_summary=persona_summary.strip(),
            score_lines=score_lines.strip(),
            extra_parameters=extra_parameters.strip(),
        ).strip()


def format_extra_parameters(
    spec: VirtualPromptSpec,
    respondent_id: str,
    *,
    root: Path = DEFAULT_PROMPT_ROOT,
) -> str:
    """Format custom persona fields for one respondent."""

    values: dict[str, Any] = {}
    if spec.parameter_values_file:
        value_path = Path(spec.parameter_values_file)
        if not value_path.is_absolute():
            value_path = Path(root) / value_path
        if value_path.is_file():
            payload = json.loads(value_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"参数值文件必须是对象：{value_path}")
            respondent_values = payload.get(str(respondent_id), {})
            if respondent_values is None:
                respondent_values = {}
            if not isinstance(respondent_values, dict):
                raise ValueError(
                    f"参数值文件中被试记录必须是对象：{value_path}/{respondent_id}"
                )
            values = respondent_values

    lines: list[str] = []
    for parameter in spec.extra_parameters:
        key = str(parameter.get("key") or "").strip()
        label = str(parameter.get("label") or key).strip()
        if not key:
            continue
        value = values.get(key, parameter.get("default", "未指定"))
        description = str(parameter.get("description") or "").strip()
        suffix = f"（{description}）" if description else ""
        lines.append(f"{label}：{value}{suffix}")
    return "\n".join(lines)


def load_prompt_specs(root: Path = DEFAULT_PROMPT_ROOT) -> dict[str, VirtualPromptSpec]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"提示词目录不存在：{root}")
    specs: dict[str, VirtualPromptSpec] = {}
    for path in sorted(root.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"提示词配置根节点必须是对象：{path}")
        spec = VirtualPromptSpec.from_mapping(value, source=path)
        if spec.prompt_id in specs:
            raise ValueError(f"提示词ID重复：{spec.prompt_id}")
        specs[spec.prompt_id] = spec
    if not specs:
        raise ValueError(f"提示词目录为空：{root}")
    return specs


def get_prompt_spec(prompt_id: str, root: Path = DEFAULT_PROMPT_ROOT) -> VirtualPromptSpec:
    specs = load_prompt_specs(root)
    try:
        return specs[str(prompt_id)]
    except KeyError as exc:
        available = ", ".join(sorted(specs))
        raise ValueError(f"未知虚拟被试提示词：{prompt_id}；可选：{available}") from exc


def format_prompt_list(specs: dict[str, VirtualPromptSpec]) -> str:
    lines = ["当前可用虚拟被试提示词："]
    for index, spec in enumerate(specs.values(), 1):
        lines.extend([
            f"{index}. {spec.prompt_id} · {spec.label}",
            f"   {spec.description}",
            f"   提供的人格信息：{spec.persona_information}",
            f"   作答方式：{spec.response_behavior}",
            f"   额外参数：{len(spec.extra_parameters)}个",
            f"   SJT：{spec.sjt_response_mode}；IPIP：{spec.ipip_response_mode}",
        ])
    return "\n".join(lines)


def select_prompt_ids_interactively(
    specs: dict[str, VirtualPromptSpec],
    *,
    default_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Show the prompt menu and return exactly one selected prompt ID."""

    ordered = list(specs.values())
    print(format_prompt_list(specs), flush=True)
    print("\n请选择本次要使用的一种虚拟被试方法。", flush=True)
    default_id = next((item for item in default_ids if item in specs), None)
    if default_id:
        default_index = next(
            index + 1 for index, spec in enumerate(ordered) if spec.prompt_id == default_id
        )
        print(f"直接回车使用默认方法：{default_index}", flush=True)
    while True:
        raw = input("你的选择：").strip().lower()
        if not raw and default_id:
            return (default_id,)
        try:
            index = int(raw)
        except ValueError:
            index = 0
        if 1 <= index <= len(ordered):
            return (ordered[index - 1].prompt_id,)
        print(f"输入无效，请输入1-{len(ordered)}中的一个编号。", flush=True)
