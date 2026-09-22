"""Small, explicitly non-production audit of the embodied consequence prompt.

This command asks the model for concise, observable consequence judgements.
It never requests hidden chain-of-thought and does not alter formal response
records used by the A/B/C experiment.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.runnables import RunnableLambda

from sjt_system.agent.client import (
    build_json_output_instruction,
    get_model,
    get_model_request_timeout_seconds,
)
from sjt_system.agent.json_parsing import parse_model_json_response
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .legacy_pool_abc import (
    OPTION_IDS,
    _result_dict,
    build_legacy_persona_prompt,
)
from .storage import write_json


DEFAULT_EXPERIMENT = Path(
    r"E:\DR_projects\langgraph_for_SJT\experiment_data\exp_20260911_155055_92946884"
)
DEFAULT_SUMMARY_POOL = Path(__file__).resolve().parents[2] / "experiment_data" / "legacy_persona_summary_pool"


class OptionConsequenceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str
    time_energy_cost: int = Field(ge=0, le=100)
    emotional_discomfort: int = Field(ge=0, le=100)
    relationship_reputation_risk: int = Field(ge=0, le=100)
    subjective_benefit: int = Field(ge=0, le=100)
    willingness: int = Field(ge=0, le=100)
    first_person_reaction: str


class EmbodiedPromptAuditOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_assessments: list[OptionConsequenceAudit]
    choice_probabilities: dict[str, float]


def _load_summary(summary_pool: Path, respondent_id: str) -> str:
    path = summary_pool / "persona_summaries.jsonl"
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("respondent_id")) == respondent_id:
                summary = str(row.get("summary") or "").strip()
                if summary:
                    return summary
    raise ValueError(f"找不到被试画像：{respondent_id}")


def _audit_messages(persona_prompt: str, item: dict[str, Any]) -> list[tuple[str, str]]:
    options = {str(value["option_id"]): value for value in item["response_options"]}
    option_lines = [
        f"{option_id}. {options[option_id]['text']}" for option_id in OPTION_IDS
    ]
    system = persona_prompt + """

[VISIBLE PROMPT-COMPLIANCE AUDIT]
这是一次小规模提示词遵循审计，不是正式施测。
不要输出隐藏思维链或逐步推理。情境正在发生在你自己身上，请只给出每个选项的第一人称主观体验结论：
- time_energy_cost：本人主观时间与精力负担，0—100；
- emotional_discomfort：本人主观情绪不适或尴尬，0—100；
- relationship_reputation_risk：本人主观关系、冲突或声誉风险，0—100；
- subjective_benefit：本人主观安心、社交、目标或自我保护收益，0—100；
- willingness：本人实际采取该行为的愿意程度，0—100；
- first_person_reaction：不超过50个汉字，必须以“我”的第一人称表达主观反应，只写结论，不写思维过程。
不要使用“这个人”“该人物”“符合画像”“人格特质”“目标构念”或“计分”等观察者和测量术语。
这些数值必须来自你已经内化的个人经历与当前情境；不要把积极、礼貌的选项自动评得更高。
[/VISIBLE PROMPT-COMPLIANCE AUDIT]"""
    human = (
        f"情境：\n{item['scenario']}\n\n"
        f"问题：\n{item.get('response_instruction', '你会怎么做？')}\n\n"
        "选项：\n"
        + "\n".join(option_lines)
        + "\n\n请返回A、B、C、D各一条option_assessments，并给出四个选择概率。"
        "概率必须介于0和1且总和为1。只返回结构化结果。"
        "\n\n"
        + build_json_output_instruction(EmbodiedPromptAuditOutput)
    )
    return [("system", system), ("human", human)]


def _validate_output(value: Any) -> dict[str, Any]:
    result = _result_dict(value)
    assessments = result.get("option_assessments")
    probabilities = result.get("choice_probabilities")
    if not isinstance(assessments, list) or len(assessments) != 4:
        raise ValueError("option_assessments必须恰好包含4条")
    normalized_assessments: list[dict[str, Any]] = []
    for assessment in assessments:
        if hasattr(assessment, "model_dump"):
            assessment = assessment.model_dump()
        if not isinstance(assessment, dict):
            raise ValueError("option_assessments元素必须是对象")
        normalized_assessments.append(dict(assessment))
    by_option = {
        str(row.get("option_id") or "").strip().upper(): row
        for row in normalized_assessments
    }
    if set(by_option) != set(OPTION_IDS) or len(by_option) != 4:
        raise ValueError("option_assessments必须且只能覆盖A-D")
    if not isinstance(probabilities, dict):
        raise ValueError("choice_probabilities必须是对象")
    normalized_probabilities = {
        str(key).strip().upper(): float(number)
        for key, number in probabilities.items()
    }
    if set(normalized_probabilities) != set(OPTION_IDS):
        raise ValueError("choice_probabilities必须且只能覆盖A-D")
    if any(
        not math.isfinite(number) or number < 0 or number > 1
        for number in normalized_probabilities.values()
    ):
        raise ValueError("choice_probabilities包含无效概率")
    probability_sum = sum(normalized_probabilities.values())
    if not math.isclose(probability_sum, 1.0, abs_tol=0.02):
        raise ValueError(f"choice_probabilities之和不是1：{probability_sum}")
    normalized_probabilities = {
        option_id: normalized_probabilities[option_id] / probability_sum
        for option_id in OPTION_IDS
    }
    forbidden_terms = ("这个人", "该人物", "符合画像", "人格特质", "目标构念", "计分")
    ordered_assessments: list[dict[str, Any]] = []
    for option_id in OPTION_IDS:
        assessment = dict(by_option[option_id])
        reaction = str(assessment.get("first_person_reaction") or "").strip()
        found = [term for term in forbidden_terms if term in reaction]
        assessment["uses_explicit_first_person"] = "我" in reaction
        assessment["forbidden_observer_terms"] = found
        assessment["first_person_compliant"] = "我" in reaction and not found
        ordered_assessments.append(assessment)
    return {
        "option_assessments": ordered_assessments,
        "choice_probabilities": normalized_probabilities,
        "raw_probability_sum": probability_sum,
        "prompt_compliance": {
            "first_person_compliant_count": sum(
                row["first_person_compliant"] for row in ordered_assessments
            ),
            "all_options_first_person_compliant": all(
                row["first_person_compliant"] for row in ordered_assessments
            ),
            "audit_rule": (
                "Record noncompliance without rejecting or regenerating the model output."
            ),
        },
    }


async def run_audit(
    *,
    experiment: Path,
    summary_pool: Path,
    respondent_id: str,
    item_ids: list[str],
    model_id: str,
    output: Path | None,
) -> tuple[Path, dict[str, Any]]:
    experiment = experiment.resolve()
    forms = json.loads(
        (experiment / "legacy_summary_embodied_probability_abc" / "forms.json").read_text(
            encoding="utf-8-sig"
        )
    )
    item_by_id = {str(item["item_id"]): item for item in forms["A"]["items"]}
    missing = [item_id for item_id in item_ids if item_id not in item_by_id]
    if missing:
        raise ValueError(f"找不到审计题目：{missing}")
    summary = _load_summary(summary_pool.resolve(), respondent_id)
    persona_prompt = build_legacy_persona_prompt(
        {}, summary, persona_mode="summary_embodied_probability"
    )
    if output is None:
        output = (
            experiment
            / "embodied_prompt_audit"
            / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = get_model(model_id)
    runnable = (
        model
        | RunnableLambda(parse_model_json_response)
        | RunnableLambda(EmbodiedPromptAuditOutput.model_validate)
    )
    structured_method = "plain_json_with_local_safe_repair"
    timeout = get_model_request_timeout_seconds()
    run_id = f"embodied-prompt-audit-{uuid4().hex}"
    records: list[dict[str, Any]] = []
    with output_scope(output / "runtime", telemetry=output / "telemetry"), run_context(run_id):
        for item_id in item_ids:
            item = item_by_id[item_id]
            messages = _audit_messages(persona_prompt, item)
            raw = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout)
            validated = _validate_output(raw)
            record = {
                "respondent_id": respondent_id,
                "model_id": model_id,
                "item_id": item_id,
                "scenario": item["scenario"],
                "response_options": item["response_options"],
                "persona_summary": summary,
                "structured_output_method": structured_method,
                **validated,
            }
            records.append(record)
            write_json(output / f"{item_id}.json", record)
    result = {
        "status": "complete",
        "model_id": model_id,
        "respondent_id": respondent_id,
        "item_ids": item_ids,
        "call_count": len(records),
        "records": records,
        "interpretation_boundary": (
            "The output audits visible prompt compliance only; it does not expose or "
            "verify hidden chain-of-thought and does not establish human likeness."
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "summary.json", result)
    return output, result


def main() -> int:
    parser = argparse.ArgumentParser(description="两题具身提示词遵循审计")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--summary-pool", type=Path, default=DEFAULT_SUMMARY_POOL)
    parser.add_argument("--respondent-id", default="VR-0001")
    parser.add_argument("--items", default="A-001,A-016")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    item_ids = [value.strip() for value in args.items.split(",") if value.strip()]
    output, result = asyncio.run(
        run_audit(
            experiment=args.experiment,
            summary_pool=args.summary_pool,
            respondent_id=args.respondent_id,
            item_ids=item_ids,
            model_id=args.model,
            output=args.output,
        )
    )
    print(f"审计输出：{output}")
    print(f"模型={result['model_id']}；调用={result['call_count']}次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
