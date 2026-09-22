from functools import lru_cache
from hashlib import sha256
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from sjt_system.prompt import (
    COMPACT_SKELETON_BATCH_PROMPT,
    ITEM_REPAIR_PROMPT,
    ITEM_WRITER_PROMPT,
    MECHANISM_VALIDATION_PROMPT,
    PSYCHOMETRIC_REPAIR_DIAGNOSIS_PROMPT,
    REQUIREMENT_PROMPT,
    UNIFIED_ITEM_REVIEW_PROMPT,
)
from sjt_system.state import (
    CompactSkeletonResult,
    ItemReviewDiagnosis,
    ItemRepairResult,
    ItemRealizationResult,
    MechanismValidationResult,
    RequirementResult,
    AtomicRepairAdvice,
)
from .client import (
    build_json_output_instruction,
    get_model,
    with_compatible_structured_output,
)

def _task_model_parameters(
    prefix: str,
    *,
    default_temperature: float,
) -> tuple[str | None, float]:
    model_id = os.getenv(f"{prefix}_MODEL_ID") or None
    raw_temperature = os.getenv(
        f"{prefix}_TEMPERATURE",
        str(default_temperature),
    )
    try:
        temperature = float(raw_temperature)
    except ValueError as exc:
        raise ValueError(
            f"{prefix}_TEMPERATURE must be a number"
        ) from exc
    if temperature < 0 or temperature > 2:
        raise ValueError(
            f"{prefix}_TEMPERATURE must be between 0 and 2"
        )
    return model_id, temperature


def _reasoning_role_parameters(
    prefix: str,
    *,
    default_model_id: str,
) -> tuple[str, float | None, str | None, str | None]:
    """Read the configured reasoning-role defaults and optional overrides."""

    model_id = os.getenv(f"{prefix}_MODEL_ID", default_model_id).strip()
    if not model_id:
        raise ValueError(f"{prefix}_MODEL_ID cannot be empty")
    raw_temperature = os.getenv(f"{prefix}_TEMPERATURE", "").strip()
    temperature: float | None = None
    if raw_temperature:
        try:
            temperature = float(raw_temperature)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}_TEMPERATURE must be a number"
            ) from exc
        if temperature < 0 or temperature > 2:
            raise ValueError(
                f"{prefix}_TEMPERATURE must be between 0 and 2"
            )
    thinking_type = os.getenv(f"{prefix}_THINKING", "enabled").strip().lower()
    if thinking_type not in {"enabled", "disabled"}:
        raise ValueError(f"{prefix}_THINKING must be enabled or disabled")
    reasoning_effort = os.getenv(
        f"{prefix}_REASONING_EFFORT",
        "high",
    ).strip().lower()
    if reasoning_effort not in {"low", "medium", "high", "max"}:
        raise ValueError(
            f"{prefix}_REASONING_EFFORT must be low, medium, high, or max"
        )
    if thinking_type == "disabled":
        # 允许关闭推理模式：禁用时不再携带 reasoning_effort
        # （client.get_model 拒绝 reasoning_effort + thinking=disabled）
        reasoning_effort = None
    return model_id, temperature, thinking_type, reasoning_effort


@lru_cache(maxsize=None)
def create_agent(
    system_prompt: str,
    output_type: type,
    *,
    model_id: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    thinking_type: str | None = None,
    include_json_schema: bool = True,
) -> Runnable:
    """使用统一的 Prompt、模型和输出类型创建专业 Agent。

    ``include_json_schema`` controls whether the machine-generated JSON Schema
    is appended to the hand-written prompt for json_mode/plain_json.  It stays
    True for agents whose prompt does not describe the output object itself.
    An agent may disable it only when its prompt carries an explicit, current
    output contract with field checklists and minimal examples; the local
    Pydantic validation and the bounded schema-repair retry stay active either
    way.
    """

    model = get_model(
        model_id,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        thinking_type=thinking_type,
    )
    structured_model, method = with_compatible_structured_output(
        model,
        output_type,
    )
    if (
        method in {"json_mode", "plain_json"}
        and include_json_schema
    ):
        json_instruction = build_json_output_instruction(output_type)
        json_instruction = json_instruction.replace("{", "{{").replace(
            "}",
            "}}",
        )
        system_prompt = (
            f"{system_prompt}\n\n"
            f"{json_instruction}"
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Task input：\n{input_data}"),
    ])
    return prompt | structured_model


# 需求分析：形成 test_specification 等需求字段
requirement_agent = create_agent(
    system_prompt=REQUIREMENT_PROMPT,
    output_type=RequirementResult,
    # The prompt carries its own explicit output contract with a complete
    # example object, so the verbose machine-generated JSON Schema is not
    # appended here.
    include_json_schema=False,
)


# v5 abstract psychological-skeleton generation.
_skeleton_generation_model_id, _skeleton_generation_temperature = (
    _task_model_parameters(
        "SKELETON_GENERATION",
        default_temperature=0.7,
    )
)
compact_skeleton_agent = create_agent(
    system_prompt=COMPACT_SKELETON_BATCH_PROMPT,
    output_type=CompactSkeletonResult,
    model_id=_skeleton_generation_model_id,
    temperature=_skeleton_generation_temperature,
)

# 题目生成：形成 current_item
item_writer_agent = create_agent(
    system_prompt=ITEM_WRITER_PROMPT,
    output_type=ItemRealizationResult,
)


# 一次调用完成四维题目诊断；程序根据 severity 与 locus 派生修复任务和路由。
item_review_agent = create_agent(
    system_prompt=UNIFIED_ITEM_REVIEW_PROMPT,
    output_type=ItemReviewDiagnosis,
)


# 题目修改：根据审题意见定向更新 current_item
item_repair_agent = create_agent(
    system_prompt=ITEM_REPAIR_PROMPT,
    output_type=ItemRepairResult,
)

# Compatibility aliases: the workflow still uses the two action names to keep
# separate revision/rewrite counters, but both actions call one repair agent.
revision_agent = item_repair_agent
item_regeneration_agent = item_repair_agent


(
    _psychometric_diagnosis_model_id,
    _psychometric_diagnosis_temperature,
    _psychometric_diagnosis_thinking,
    _psychometric_diagnosis_reasoning_effort,
) = _reasoning_role_parameters(
    "PSYCHOMETRIC_DIAGNOSIS",
    default_model_id="glm-5.3-flash",
)
psychometric_repair_diagnosis_agent = create_agent(
    system_prompt=PSYCHOMETRIC_REPAIR_DIAGNOSIS_PROMPT,
    output_type=AtomicRepairAdvice,
    model_id=_psychometric_diagnosis_model_id,
    temperature=_psychometric_diagnosis_temperature,
    reasoning_effort=_psychometric_diagnosis_reasoning_effort,
    thinking_type=_psychometric_diagnosis_thinking,
    # The prompt carries its own explicit output contract with field
    # checklists and minimal defer/repair examples, so the verbose
    # machine-generated JSON Schema is not appended here.
    include_json_schema=False,
)


(
    _psychometric_item_repair_model_id,
    _psychometric_item_repair_temperature,
    _psychometric_item_repair_thinking,
    _psychometric_item_repair_reasoning_effort,
) = _reasoning_role_parameters(
    "PSYCHOMETRIC_ITEM_REPAIR",
    default_model_id="glm-5.3-flash",
)
psychometric_item_repair_agent = create_agent(
    system_prompt=ITEM_REPAIR_PROMPT,
    output_type=ItemRepairResult,
    model_id=_psychometric_item_repair_model_id,
    temperature=_psychometric_item_repair_temperature,
    reasoning_effort=_psychometric_item_repair_reasoning_effort,
    thinking_type=_psychometric_item_repair_thinking,
)


PSYCHOMETRIC_REASONING_ROLE_MANIFEST = {
    "psychometric_diagnosis": {
        "model_id": _psychometric_diagnosis_model_id,
        "temperature": _psychometric_diagnosis_temperature,
        "thinking": _psychometric_diagnosis_thinking,
        "reasoning_effort": _psychometric_diagnosis_reasoning_effort,
        "prompt_sha256": sha256(
            PSYCHOMETRIC_REPAIR_DIAGNOSIS_PROMPT.encode("utf-8")
        ).hexdigest(),
    },
    "psychometric_item_repair": {
        "model_id": _psychometric_item_repair_model_id,
        "temperature": _psychometric_item_repair_temperature,
        "thinking": _psychometric_item_repair_thinking,
        "reasoning_effort": _psychometric_item_repair_reasoning_effort,
        "prompt_sha256": sha256(ITEM_REPAIR_PROMPT.encode("utf-8")).hexdigest(),
    },
}


mechanism_validation_agent = create_agent(
    system_prompt=MECHANISM_VALIDATION_PROMPT,
    output_type=MechanismValidationResult,
    temperature=0.1,
)
