"""Concurrent virtual-response generation and persistence."""

from __future__ import annotations

import asyncio
import csv
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from io import StringIO
import os
from pathlib import Path
import random
from typing import Any

from pydantic import BaseModel, ConfigDict

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.authoring.bank import build_virtual_response_context
from sjt_system.knowledge.behavior_evidence import (
    DEFAULT_CORPUS_PATH as DEFAULT_IPIP_NEO_PATH,
    load_ipip_corpus,
)
from sjt_system.runtime.progress import emit_progress
from sjt_system.runtime.concurrency import (
    UnlimitedConcurrency,
    validate_max_concurrency,
)
from sjt_system.runtime.telemetry import job_context as telemetry_job_context
from sjt_system.runtime.io import (
    write_json_atomic as _write_json_atomic,
    write_text_atomic as _write_text_atomic,
)
from sjt_system.evaluation.respondents import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    PERSONA_MODE_SCORE_PROFILE,
    SCORE_PROFILE_GENERATOR_VERSION,
    SCORE_PROFILE_PROMPT_VERSION,
    MATCHED_CONDITION_GENERATOR_VERSION,
    MATCHED_CONDITION_PROMPT_VERSION,
    MATCHED_CONDITION_SCHEMA_VERSION,
    DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT,
    MATCHED_CONDITION_IDS,
    flatten_matched_condition_groups,
    SUPPORTED_PERSONA_MODES,
    resolve_virtual_respondent_profiles,
    validate_score_definitions,
    matched_condition_sample_is_current,
)
from sjt_system.evaluation.demographics import (
    DEFAULT_RESPONSE_TEMPERATURE,
    DEMOGRAPHICS_VERSION,
    demographics_to_columns,
    load_demographics_snapshot,
    resolve_response_temperature,
    validate_demographics_config,
)
from sjt_system.state import PSJTState
from sjt_system.runtime.trace import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEO_FFI_PATH = PROJECT_ROOT / "docs" / "Neo-FFI.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "virtual_responses"
VIRTUAL_RESPONSE_PROMPT_VERSION = "defined-facet-demographics-fixed-score-sjt-v4"
PERSONA_SUMMARY_PROMPT_VERSION = "liu-item-response-summary-v2"
IPIP_NEO_REFERENCE_PROMPT_VERSION = "defined-facet-demographics-ipip-selected-batch-v6"


class SJTSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_option_id: str


class SJTBatchSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    selected_option_id: str


class SJTBatchSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    responses: list[SJTBatchSelection]


class NeoFFIBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratings: list[int]


class IPIPNEOBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratings: list[int]


class PersonaSummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


def load_neo_ffi(
    path: str | Path = DEFAULT_NEO_FFI_PATH,
) -> list[dict[str, Any]]:
    """读取五个维度、每个维度12题的 Neo-FFI 目标量表。"""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or list(raw) != ["N", "E", "O", "A", "C"]:
        raise ValueError("Neo-FFI 必须按 N、E、O、A、C 五个维度组织")

    dimensions = []
    for dimension_code, dimension_data in raw.items():
        if not isinstance(dimension_data, dict):
            raise ValueError(f"Neo-FFI 维度 {dimension_code} 结构无效")
        raw_items = dimension_data.get("items")
        if not isinstance(raw_items, dict) or len(raw_items) != 12:
            raise ValueError(
                f"Neo-FFI 维度 {dimension_code} 必须包含12道题"
            )
        items = []
        for item_id, item_data in raw_items.items():
            if not isinstance(item_data, dict):
                raise ValueError(f"Neo-FFI 题目 {item_id} 结构无效")
            text = item_data.get("item")
            scoring = item_data.get("scoring")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Neo-FFI 题目 {item_id} 缺少题干")
            if scoring not in {"+", "-"}:
                raise ValueError(f"Neo-FFI 题目 {item_id} 计分方向无效")
            items.append(
                {
                    "item_id": item_id,
                    "text": text,
                    "scoring_direction": scoring,
                }
            )
        dimensions.append(
            {
                "dimension_code": dimension_code,
                "domain": dimension_data.get("domain"),
                "items": items,
            }
        )
    return dimensions


def _prepare_ipip_neo_scale(
    scale_data: Mapping[str, Any],
    *,
    corpus: Any,
) -> dict[str, Any]:
    """Normalize one IPIP facet without exposing its construct to the model."""

    scale = dict(scale_data)
    items = list(scale.get("items") or [])
    if len(items) != 10:
        raise ValueError(
            f"IPIP-NEO {scale.get('facet_code')} 必须恰好包含10题"
        )
    # The source corpus groups positive and negative items.  A deterministic
    # mixed order prevents polarity blocks without introducing run-to-run noise.
    scale["items"] = sorted(
        items,
        key=lambda item: sha256(
            (
                IPIP_NEO_REFERENCE_PROMPT_VERSION
                + ":"
                + str(item.get("item_id") or "")
            ).encode("utf-8")
        ).hexdigest(),
    )
    scale["corpus_hash"] = corpus.corpus_hash
    scale["source_file"] = corpus.source_file
    scale["source_sha256"] = corpus.source_sha256
    return scale


def load_ipip_neo_facet_scales(
    facet_ids: Sequence[str],
    path: str | Path = DEFAULT_IPIP_NEO_PATH,
) -> list[dict[str, Any]]:
    """Load the ten-item IPIP-NEO scale for every selected SJT facet.

    The facet list is the measurement scope chosen during authoring.  It is
    intentionally not replaced with a hard-coded target facet: the same target
    respondents answer every selected scale.
    """

    normalized_ids = list(
        dict.fromkeys(str(value or "").strip() for value in facet_ids)
    )
    normalized_ids = [value for value in normalized_ids if value]
    if not normalized_ids:
        raise ValueError("IPIP-NEO参照问卷缺少出题阶段选定的facet")
    corpus = load_ipip_corpus(Path(path))
    by_facet_id = {
        str(scale.facet_id): scale
        for scale in corpus.scales
    }
    missing = [value for value in normalized_ids if value not in by_facet_id]
    if missing:
        raise ValueError(
            "IPIP-NEO中无法匹配出题阶段选定的facet：" + "、".join(missing)
        )
    return [
        _prepare_ipip_neo_scale(
            by_facet_id[facet_id].model_dump(mode="json"),
            corpus=corpus,
        )
        for facet_id in normalized_ids
    ]


def load_ipip_neo_target_scale(
    target_dimension_id: str,
    path: str | Path = DEFAULT_IPIP_NEO_PATH,
) -> dict[str, Any]:
    """Backward-compatible loader for one IPIP-NEO facet."""

    scales = load_ipip_neo_facet_scales([target_dimension_id], path)
    return scales[0]


def resolve_ipip_neo_reference_facet_ids(
    state: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Resolve the facet scope declared by authoring, with item fallback.

    Explicit construct selection is authoritative.  The item-bank facet list
    is only a compatibility fallback for older checkpoints that predate the
    persisted construct-selection snapshot.
    """

    candidates: list[Any] = []
    specification = state.get("test_specification")
    if isinstance(specification, Mapping):
        selection = specification.get("construct_selection")
        if isinstance(selection, Mapping):
            candidates.append(selection.get("facet_ids"))
    profile = state.get("construct_profile")
    if isinstance(profile, Mapping):
        candidates.append(profile.get("selected_facet_ids"))
        facets = profile.get("facets")
        if isinstance(facets, list):
            candidates.append(
                [
                    facet.get("facet_id")
                    for facet in facets
                    if isinstance(facet, Mapping)
                ]
            )
    blueprint = state.get("blueprint")
    if isinstance(blueprint, Mapping):
        candidates.append(blueprint.get("selected_facet_ids"))
        snapshot = blueprint.get("construct_profile_snapshot")
        if isinstance(snapshot, Mapping):
            candidates.append(snapshot.get("selected_facet_ids"))
    candidates.append(
        [
            item.get("target_dimension_id")
            for item in items
            if isinstance(item, Mapping)
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, list):
            resolved = list(
                dict.fromkeys(
                    str(value).strip()
                    for value in candidate
                    if isinstance(value, str) and value.strip()
                )
            )
            if resolved:
                return resolved
    raise ValueError("无法确定出题阶段选定的IPIP-NEO facet集合")


def build_persona_prompt(
    profile: Mapping[str, Any],
    *,
    persona_mode: str = PERSONA_MODE_SCORE_PROFILE,
    score_specs: Sequence[Mapping[str, Any]] | None = None,
    demographics_snapshot: Mapping[str, Any] | None = None,
) -> str:
    """Build the active explicit-score prompt."""

    if persona_mode not in SUPPORTED_PERSONA_MODES:
        raise ValueError(f"不支持的人格提示模式：{persona_mode}")
    if persona_mode == PERSONA_MODE_SCORE_PROFILE:
        values = profile.get("score_values")
        specs = list(score_specs or profile.get("score_specs") or [])
        if not isinstance(values, Mapping):
            raise ValueError("score_profile 模式缺少分数或 score_specs")
        validate_score_definitions(specs)
        if {spec.get("dimension_id") for spec in specs} != set(values):
            raise ValueError("人格分数与冻结的 facet 构念定义不一致")
        snapshot = demographics_snapshot if demographics_snapshot is not None else load_demographics_snapshot()
        demographics = demographics_to_columns(profile.get("demographics"), snapshot)
        demographic_lines = [
            f"年龄：{demographics['age']}岁",
            f"性别：{demographics['gender']}",
            f"国籍：{demographics['nationality']}",
            f"最高已完成学历：{demographics['education']}",
            f"职业或就业状态：{demographics['occupation']}",
            f"个人税前月收入：{demographics['monthly_income_cny']}元（人民币等值）",
        ]
        score_lines = []
        for spec in specs:
            if not isinstance(spec, Mapping):
                raise ValueError("score_specs 包含无效记录")
            dimension_id = spec.get("dimension_id")
            if dimension_id not in values:
                continue
            value = float(values[dimension_id])
            if spec.get("level") == "domain":
                label = (
                    f"Domain | {spec.get('domain_name_en')}"
                    f"（{spec.get('domain_name')}）"
                )
            else:
                label = (
                    f"Facet | {spec.get('domain_name_en')} > "
                    f"{spec.get('facet_name_en')}"
                    f"（{spec.get('facet_name')}）"
                )
            score_lines.append(f"{label} | {value:.1f}\n构念定义：{spec['definition'].strip()}")
        return (
            "请想象你正在扮演一个特定的人。\n"
            "下面的人口学资料和人格分数共同描述同一个稳定的人。\n"
            "人口学资料提供生活背景，不改变已经明确设定的 facet 分数；"
            "不要仅凭性别或国籍推断、覆盖其人格。\n\n"
            "[DEMOGRAPHICS]\n"
            + "\n".join(demographic_lines)
            + "\n[/DEMOGRAPHICS]\n\n"
            "下面给出这个人已经明确设定的人格 domain / facet 分数。\n"
            "所有分数均在 0–100 范围内：0 表示极低，50 表示中等，\n"
            "100 表示极高。分数表示跨情境的稳定倾向，而不是必然行为。\n\n"
            "[PERSONALITY SCORES]\n"
            + "\n".join(score_lines)
            + "\n[/PERSONALITY SCORES]\n\n"
            "只依据明确列出的分数塑造此人。未列出的 domain/facet 均为\n"
            "未设定，不要假定为50分、补齐、推断或提及。\n\n"
            "同一 domain 与 facet 同时出现时，直接涉及 facet 的行为以\n"
            "facet 分数为主要依据，domain 只表示更广泛倾向；允许二者不一致。\n"
            "不要补充未提供的背景、经历、能力、资源、诊断或动机。\n\n"
            "在后续任务中始终扮演这个人，不复述人格名称或分数。\n"
        )

def build_sjt_messages(
    persona_prompt: str,
    item: Mapping[str, Any],
    *,
    display_option_order: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """构造一次只回答一道 SJT 题、且不含评分信息的消息。"""

    options = item.get("response_options")
    if not isinstance(options, list) or not options:
        raise ValueError("SJT 题目缺少作答选项")
    options_by_id = {
        str(option.get("option_id")): option
        for option in options
        if isinstance(option, Mapping) and option.get("option_id")
    }
    ordered_ids = list(display_option_order or options_by_id)
    if set(ordered_ids) != set(options_by_id) or len(ordered_ids) != len(
        options_by_id
    ):
        raise ValueError("display_option_order 必须是全部选项ID的一个排列")
    option_lines = []
    for index, original_id in enumerate(ordered_ids):
        display_id = chr(ord("A") + index)
        text = options_by_id[original_id].get("text")
        if not isinstance(text, str):
            raise ValueError("SJT 选项缺少文本")
        option_lines.append(f"{display_id}. {text}")

    system_message = (
        persona_prompt
        + "\n\n现在请以这个人的真实行为倾向回答一道情境判断题。"
        "请选择此人最可能采取的行为，而不是理论上最好、最正确或"
        "社会赞许程度最高的行为。每次作答相互独立。"
        "不要在输出中复述人格名称或分数。"
        "只返回一个JSON对象，不要解释，格式为："
        '{"selected_option_id":"选项编号"}。'
    )
    user_message = (
        f"情境：\n{item.get('scenario', '')}\n\n"
        f"作答要求：\n{item.get('response_instruction', '')}\n\n"
        "选项：\n"
        + "\n".join(option_lines)
    )
    return [("system", system_message), ("human", user_message)]


def build_sjt_batch_messages(
    item_prompts: Sequence[tuple[Mapping[str, Any], str, Sequence[str]]],
) -> list[tuple[str, str]]:
    """Build one request containing a fully shuffled batch of SJT items.

    ``item_prompts`` contains ``(item, fixed_persona_prompt, option_order)``.
    Every item must use the same persona; its complete context is shown once.
    """

    if not item_prompts:
        raise ValueError("SJT批次不能为空")
    common_persona_prompt = item_prompts[0][1]
    if any(persona_prompt != common_persona_prompt for _, persona_prompt, _ in item_prompts):
        raise ValueError("同一被试的批量题目必须共享固定的人口学资料、构念定义和分数")
    blocks: list[str] = []
    for index, (item, persona_prompt, ordered_ids) in enumerate(item_prompts, 1):
        options = item.get("response_options") or []
        options_by_id = {
            str(option.get("option_id")): option
            for option in options
            if isinstance(option, Mapping) and option.get("option_id")
        }
        display_ids = [chr(ord("A") + offset) for offset in range(len(ordered_ids))]
        option_lines = [
            f"{display_id}. {options_by_id[original_id].get('text')}"
            for display_id, original_id in zip(display_ids, ordered_ids)
        ]
        blocks.append(
            f"题目 {index}（item_id={item.get('item_id')}）：\n"
            f"情境：\n{item.get('scenario', '')}\n\n"
            f"作答要求：\n{item.get('response_instruction', '')}\n\n"
            "选项：\n" + "\n".join(option_lines)
        )
    system_message = (
        common_persona_prompt
        + "\n\n请始终依据以上同一个人的固定资料和人格分数，分别判断其最可能采取的行为。"
        "每道题独立作答，不选择理论上最好或最受赞许的行为。"
        "严格按照题目出现顺序返回全部结果，不能遗漏或增加结果。"
        "必须原样返回每道题给出的 item_id，不得改写、合并或重复 item_id。"
        "selected_option_id 必须是该题显示的 A、B、C…字母。只返回JSON对象："
        '{"responses":[{"item_id":"题目ID","selected_option_id":"A"}]}'
    )
    return [("system", system_message), ("human", "\n\n".join(blocks))]


class SJTBatchValidationError(ValueError):
    """A batch response with enough structure to identify failed items."""

    def __init__(
        self,
        message: str,
        *,
        failed_item_ids: Sequence[str],
        valid_responses: Mapping[str, str],
    ) -> None:
        super().__init__(message)
        self.failed_item_ids = tuple(dict.fromkeys(str(item_id) for item_id in failed_item_ids))
        self.valid_responses = dict(valid_responses)


def _validate_sjt_batch_response(
    result: Mapping[str, Any],
    item_rows: Sequence[tuple[Mapping[str, Any], str, Sequence[str]]],
    *,
    label: str = "SJT批次",
) -> dict[str, str]:
    """Validate an ordered response list and bind it to the local item order.

    Keep the item ID as the binding key, but expose valid partial results when
    one response is malformed so target-form retest can retry only failed items.
    """

    expected_by_id = {
        str(item.get("item_id") or ""): (item, ordered_ids)
        for item, _persona_prompt, ordered_ids in item_rows
    }
    expected_ids = {item_id for item_id in expected_by_id if item_id}
    rows = result.get("responses")
    if not isinstance(rows, list):
        raise SJTBatchValidationError(
            f"{label}返回结果不是列表",
            failed_item_ids=expected_ids,
            valid_responses={},
        )
    parsed: dict[str, str] = {}
    invalid_item_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item_id = str(row.get("item_id") or "")
        selected = row.get("selected_option_id")
        item_and_order = expected_by_id.get(item_id)
        if item_and_order is None:
            continue
        item, ordered_ids = item_and_order
        allowed = {chr(ord("A") + offset) for offset in range(len(ordered_ids))}
        if selected not in allowed or item_id in parsed:
            invalid_item_ids.append(item_id)
            # A duplicate item_id invalidates that item's batch result even if
            # the first occurrence looked valid; force it through item-level
            # retest instead of silently retaining the first occurrence.
            parsed.pop(item_id, None)
            continue
        parsed[item_id] = str(selected)
    failed_item_ids = (expected_ids - set(parsed)) | set(invalid_item_ids)
    if len(rows) != len(expected_ids) or failed_item_ids:
        if not failed_item_ids:
            failed_item_ids = expected_ids
        raise SJTBatchValidationError(
            f"{label}返回了无效或不完整作答；失败题目：{', '.join(sorted(failed_item_ids))}",
            failed_item_ids=failed_item_ids,
            valid_responses=parsed,
        )
    return parsed


def build_neo_ffi_messages(
    persona_prompt: str,
    items: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """构造一个不暴露维度名称和正反向计分的12题批次。"""

    description_lines = [
        f"描述 {index}：{item['text']}"
        for index, item in enumerate(items, 1)
    ]
    system_message = (
        persona_prompt
        + "\n\n请判断你扮演的这个人对下面每项描述的同意程度："
        "1=非常不同意，2=比较不同意，3=不确定，4=比较同意，"
        "5=非常同意。按题目顺序返回12个整数。"
        "只返回一个JSON对象，不要解释，格式为："
        '{"ratings":[1,2,3,4,5,1,2,3,4,5,1,2]}。'
    )
    return [
        ("system", system_message),
        ("human", "\n".join(description_lines)),
    ]


def build_ipip_neo_messages(
    persona_prompt: str,
    items_or_scales: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Build one facet-blind IPIP-NEO request for all selected facets.

    The legacy one-facet ``items`` form remains accepted for old callers. New
    callers pass scale mappings; their ten-item groups are flattened in the
    selected-facet order and scored back into their original groups.
    """

    if not items_or_scales:
        raise ValueError("IPIP-NEO参照问卷缺少题目")
    grouped_items: list[list[Mapping[str, Any]]] = []
    if any(isinstance(row, Mapping) and "items" in row for row in items_or_scales):
        for scale in items_or_scales:
            if not isinstance(scale, Mapping):
                raise ValueError("IPIP-NEO参照问卷包含无效facet")
            items = list(scale.get("items") or [])
            if len(items) != 10:
                raise ValueError(
                    f"IPIP-NEO {scale.get('facet_code')} 必须恰好包含10题"
                )
            grouped_items.append(items)
    else:
        grouped_items.append(list(items_or_scales))
    items = [item for group in grouped_items for item in group]
    description_lines = [
        f"描述 {index}：{item['text']}"
        for index, item in enumerate(items, 1)
    ]
    count = len(description_lines)
    system_message = (
        persona_prompt
        + "\n\n请判断你扮演的这个人对下面每项描述的同意程度："
        "1=非常不同意，2=比较不同意，3=不确定，4=比较同意，"
        "5=非常同意。不要猜测量表名称、人格维度、计分方向或研究者期待；"
        f"每题独立判断；本次一次完成全部{count}题，并按题目顺序返回{count}个整数。"
        "只返回一个JSON对象，不要解释，格式为："
        '{"ratings":[1,2,3,4,5,1,2,3,4,5]}。'
    )
    return [
        ("system", system_message),
        ("human", "\n".join(description_lines)),
    ]


def score_ipip_neo_rating(rating: int, polarity: str) -> int:
    """Score one 1-5 IPIP response using the corpus polarity."""

    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError("IPIP-NEO评分必须是1到5之间的整数")
    if polarity == "positive":
        return rating
    if polarity == "negative":
        return 6 - rating
    raise ValueError(f"IPIP-NEO计分方向无效：{polarity!r}")


def _structured_result_dict(result: object) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    legacy_dict = getattr(result, "dict", None)
    if callable(legacy_dict):
        dumped = legacy_dict()
        if isinstance(dumped, dict):
            return dumped
    raise ValueError("模型没有返回有效的结构化对象")


class ResponseTemperatureRejected(RuntimeError):
    """A provider rejected a required protocol parameter; do not retry."""


async def _invoke_with_retry(
    runnable: Any,
    messages: list[tuple[str, str]],
    *,
    semaphore: Any,
    validator: Any,
    max_retries: int,
    retry_delay_seconds: float,
    request_timeout_seconds: float,
    job_label: str,
    attempt_ledger: MutableMapping[str, Any] | None = None,
    attempt_key: str | None = None,
    persist_attempt_ledger: Any | None = None,
) -> Any:
    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 0
        or max_retries > 4
    ):
        raise ValueError("max_retries 必须是 0 到 4 之间的整数（最多5次尝试）")
    if (attempt_ledger is None) != (attempt_key is None):
        raise ValueError("持久化请求预算必须同时提供 attempt_ledger 和 attempt_key")
    if attempt_ledger is not None and not callable(persist_attempt_ledger):
        raise ValueError("持久化请求预算必须提供写盘回调")

    max_attempts = max_retries + 1
    attempt_entry: MutableMapping[str, Any] | None = None
    history: list[dict[str, Any]] | None = None
    attempts_used = 0
    if attempt_ledger is not None and attempt_key is not None:
        raw_entry = attempt_ledger.get(attempt_key)
        if raw_entry is None:
            attempt_entry = {"attempts_used": 0, "max_attempts": max_attempts, "history": []}
            attempt_ledger[attempt_key] = attempt_entry
        elif isinstance(raw_entry, MutableMapping):
            attempt_entry = raw_entry
        else:
            raise ValueError(f"请求尝试账本条目格式无效：{job_label}")
        used_value = attempt_entry.get("attempts_used", 0)
        configured_limit = attempt_entry.get("max_attempts", max_attempts)
        if (
            not isinstance(used_value, int)
            or isinstance(used_value, bool)
            or used_value < 0
            or used_value > max_attempts
            or configured_limit != max_attempts
        ):
            raise ValueError(f"请求尝试账本预算与当前配置不一致：{job_label}")
        attempts_used = used_value
        raw_history = attempt_entry.get("history", [])
        if not isinstance(raw_history, list) or any(
            not isinstance(row, dict) for row in raw_history
        ):
            raise ValueError(f"请求尝试账本历史格式无效：{job_label}")
        history = raw_history
        if attempts_used >= max_attempts:
            raise RuntimeError(
                f"{job_label} 已累计耗尽 {max_attempts} 次持久化请求预算；拒绝再次调用模型"
            )

    last_error: Exception | None = None
    retry_messages = list(messages)
    for attempt in range(attempts_used, max_attempts):
        attempt_number = attempt + 1
        history_row: dict[str, Any] | None = None
        if attempt_entry is not None and history is not None:
            history_row = {
                "attempt": attempt_number,
                "status": "running",
                "started_at": utc_timestamp(),
            }
            history.append(history_row)
            attempt_entry.update(
                attempts_used=attempt_number,
                max_attempts=max_attempts,
                history=history,
                updated_at=utc_timestamp(),
            )
            persist_attempt_ledger()
        try:
            async with semaphore:
                with telemetry_job_context(job_label, attempt=attempt_number):
                    raw_result = await asyncio.wait_for(
                        runnable.ainvoke(retry_messages),
                        timeout=request_timeout_seconds,
                    )
            validated = validator(_structured_result_dict(raw_result))
            if history_row is not None and attempt_entry is not None:
                history_row.update(status="completed", completed_at=utc_timestamp())
                attempt_entry["updated_at"] = utc_timestamp()
                persist_attempt_ledger()
            return validated
        except Exception as exc:
            if isinstance(exc, ResponseTemperatureRejected):
                if history_row is not None and attempt_entry is not None:
                    history_row.update(
                        status="error", error=str(exc), completed_at=utc_timestamp()
                    )
                    attempt_entry["updated_at"] = utc_timestamp()
                    persist_attempt_ledger()
                raise
            if getattr(exc, "status_code", None) in {400, 422} and "temperature" in str(exc).lower():
                if history_row is not None and attempt_entry is not None:
                    history_row.update(
                        status="error", error=str(exc), completed_at=utc_timestamp()
                    )
                    attempt_entry["updated_at"] = utc_timestamp()
                    persist_attempt_ledger()
                raise ResponseTemperatureRejected("模型端点拒绝冻结的 response_temperature；停止施测，不回退温度") from exc
            if isinstance(exc, TimeoutError):
                last_error = TimeoutError(
                    f"单次请求超过 {request_timeout_seconds:g} 秒"
                )
            else:
                last_error = exc
            if history_row is not None and attempt_entry is not None:
                history_row.update(
                    status="error",
                    error=str(last_error),
                    error_type=type(last_error).__name__,
                    completed_at=utc_timestamp(),
                )
                attempt_entry["updated_at"] = utc_timestamp()
                persist_attempt_ledger()
            if attempt_number >= max_attempts:
                break
            if isinstance(exc, TimeoutError):
                retry_messages = list(messages)
            else:
                retry_messages = [
                    *messages,
                    (
                        "human",
                        "上一版JSON未通过本地校验："
                        f"{last_error}。请重新完整输出符合要求的JSON，"
                        "不要解释、不要省略任何结果。",
                    ),
                ]
            emit_progress(
                {
                    "type": "request_retry",
                    "retry_kind": (
                        "network_timeout"
                        if isinstance(exc, TimeoutError)
                        else "output_repair"
                    ),
                    "job_label": job_label,
                    "attempt": attempt_number + 1,
                    "max_attempts": max_attempts,
                    "reason": str(last_error),
                }
            )
            await asyncio.sleep(retry_delay_seconds * (2**attempt))
    raise RuntimeError(
        f"{job_label} 在累计 {max_attempts} 次尝试后仍失败：{last_error}"
    ) from last_error


def _simulation_attempt_key(
    stage: str,
    respondent_ref: Mapping[str, Any],
    item_versions: Sequence[tuple[str, Any]],
    *,
    persona_mode: str = PERSONA_MODE_SCORE_PROFILE,
    administration_id: int | None = None,
) -> str:
    """Build a stable per-respondent request key independent of missing rows."""

    payload = {
        "stage": stage,
        "respondent_id": str(respondent_ref.get("respondent_id") or ""),
        "persona_mode": persona_mode,
        "condition_id": str(respondent_ref.get("condition_id") or ""),
        "matched_subject_id": str(respondent_ref.get("matched_subject_id") or ""),
        "administration_id": administration_id,
        "item_versions": sorted(
            [[str(item_id), version] for item_id, version in item_versions],
            key=lambda row: row[0],
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _append_jsonl_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            )
        )


def _load_jsonl_keys(
    path: Path,
    key_fields: Sequence[str],
) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是有效 JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是对象"
                )
            keys.add(tuple(record.get(field) for field in key_fields))
    return keys


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是有效 JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是对象"
                )
            records.append(record)
    return records


def _normalize_response_jsonl(
    path: Path,
    *,
    key_fields: Sequence[str],
    item_versions: Mapping[str, Any],
) -> int:
    """Canonicalize item versions and remove duplicate response records.

    Older matched-condition runs projected frozen items with ``item_version``
    but the runner looked up ``version``.  Those records were written with a
    null version and were then treated as distinct from the reused records on
    resume.  Normalize the persisted file before calculating missing work so a
    failed run can be resumed in place.
    """

    if not path.is_file():
        return 0
    records = _load_jsonl_records(path)
    canonical: dict[tuple[Any, ...], dict[str, Any]] = {}
    changed = False

    def source_priority(record: Mapping[str, Any]) -> int:
        return {
            "reused_local_item_retest": 3,
            "reused_unchanged_item": 2,
        }.get(str(record.get("response_source") or ""), 1)

    for raw_record in records:
        record = dict(raw_record)
        item_id = str(record.get("item_id") or "")
        expected_version = item_versions.get(item_id)
        if item_id in item_versions:
            actual_version = record.get("item_version")
            if actual_version is None and expected_version is not None:
                record["item_version"] = expected_version
                changed = True
            elif actual_version != expected_version:
                raise ValueError(
                    f"{path.name} 包含与当前冻结题库不一致的题目版本："
                    f"{item_id}（记录={actual_version!r}，当前={expected_version!r}）"
                )
        key = tuple(record.get(field) for field in key_fields)
        previous = canonical.get(key)
        if previous is None:
            canonical[key] = record
        elif source_priority(record) > source_priority(previous):
            canonical[key] = record
            changed = True
        else:
            changed = True

    normalized = list(canonical.values())
    if changed or len(normalized) != len(records):
        _write_text_atomic(
            path,
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in normalized
            ),
        )
    return len(normalized)


def _item_version_value(item: Mapping[str, Any]) -> Any:
    """Return the canonical item version across legacy item schemas."""

    item_version = item.get("item_version")
    return item_version if item_version is not None else item.get("version")


def _canonicalize_item_response_keys(
    keys: set[tuple[Any, ...]],
    *,
    item_versions: Mapping[str, Any],
    item_id_index: int = 2,
    version_index: int = 3,
) -> set[tuple[Any, ...]]:
    """Add canonical aliases for records written before item_version was used."""

    canonical = set(keys)
    for key in keys:
        if len(key) <= max(item_id_index, version_index):
            continue
        item_id = str(key[item_id_index] or "")
        if key[version_index] is not None or item_id not in item_versions:
            continue
        normalized = list(key)
        normalized[version_index] = item_versions[item_id]
        canonical.add(tuple(normalized))
    return canonical


def _resolve_persona_modes(config: Mapping[str, Any]) -> list[str]:
    """Accept only the active score-profile protocol."""

    raw_modes = config.get("persona_modes")
    if raw_modes is None:
        raise ValueError("旧版虚拟样本配置缺少 persona_modes，请重新配置")
    if (
        not isinstance(raw_modes, Sequence)
        or isinstance(raw_modes, (str, bytes))
        or not raw_modes
    ):
        raise ValueError("virtual_sample_config.persona_modes 必须是非空列表")
    modes = list(raw_modes)
    if (
        len(set(modes)) != len(modes)
        or any(mode not in SUPPORTED_PERSONA_MODES for mode in modes)
    ):
        raise ValueError(
            "persona_modes 只能包含 score_profile；旧配置必须重新生成"
        )
    if modes != [PERSONA_MODE_SCORE_PROFILE]:
        raise ValueError("当前主迭代只支持 score_profile")
    return modes


def _simulation_signature(
    *,
    state: PSJTState,
    respondent_refs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    persona_modes: Sequence[str],
    criterion: Mapping[str, str],
    model_id: str,
) -> str:
    payload = {
        "run_id": state["run_id"],
        "item_bank_id": state["item_bank_id"],
        "item_bank_version": state["item_bank_version"],
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "respondents": list(respondent_refs),
        "pool_id": config.get("pool_id"),
        "source_sha256": config.get("source_sha256"),
        "persona_modes": list(persona_modes),
        "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        "ipip_neo_prompt_version": IPIP_NEO_REFERENCE_PROMPT_VERSION,
        "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        "score_generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
        "persona_summary_prompt_version": PERSONA_SUMMARY_PROMPT_VERSION,
        "criterion": dict(criterion),
        "model_id": model_id,
        "response_protocol": {
            key: deepcopy(config.get(key)) for key in (
                "response_temperature", "demographics_version", "demographics_seed",
                "demographics_snapshot", "score_specs",
            )
        },
    }
    # Preserve old CLI signatures; an experimental protocol must also bind its
    # full settings when resuming partially written response files.
    if "experiment_sample_role" in config:
        payload["experiment_config"] = dict(config)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_concurrency_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore legacy request caps when matching resumable response data."""

    normalized = dict(config)
    normalized["max_concurrency"] = 0
    normalized["concurrency_policy"] = "all_at_once"
    return normalized


def _normalize_matched_source_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize fields that identify a generation round, not its cohort."""

    normalized = _normalize_concurrency_config(config)
    for field in (
        "generation_round",
        "pool_id",
        "source_sha256",
        "generation_diagnostics",
        "reference_questionnaire_freeze_policy",
        "frozen_reference_generation_round",
        "frozen_reference_questionnaire_ref",
    ):
        normalized.pop(field, None)
    return normalized


def _manifest_generation_round(manifest: Mapping[str, Any]) -> int:
    config = manifest.get("virtual_sample_config") or {}
    value = config.get("generation_round") if isinstance(config, Mapping) else None
    return int(value or 1)


def _source_profiles_match(
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    *,
    respondent_refs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> bool:
    """Fail closed unless the source froze the exact respondent profiles."""

    profiles_ref = source_manifest.get("score_profiles_path")
    if not isinstance(profiles_ref, str) or not profiles_ref:
        return False
    profiles_path = Path(profiles_ref)
    if not profiles_path.is_absolute():
        profiles_path = source_manifest_path.parent / profiles_path
    try:
        snapshot = json.loads(profiles_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    if not isinstance(snapshot, Mapping):
        return False
    for field in (
        "demographics_snapshot",
        "demographics_version",
        "response_temperature",
        "score_specs",
    ):
        if snapshot.get(field) != config.get(field):
            return False
    source_profiles = snapshot.get("profiles")
    if not isinstance(source_profiles, list):
        return False

    def by_respondent_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]] | None:
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                return None
            respondent_id = str(row.get("respondent_id") or "")
            if not respondent_id or respondent_id in indexed:
                return None
            indexed[respondent_id] = dict(row)
        return indexed

    expected = by_respondent_id(respondent_refs)
    actual = by_respondent_id(source_profiles)
    return expected is not None and actual == expected


def _matched_manifest_core_is_compatible(
    existing_manifest: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    persona_modes: Sequence[str],
    model_id: str,
) -> bool:
    """Resume only current-protocol manifests with identical frozen settings."""

    if existing_manifest.get("status") not in {"in_progress", "completed", "failed"}:
        return False
    for field, expected in (
        ("schema_version", MATCHED_CONDITION_SCHEMA_VERSION),
        ("run_id", state.get("run_id")),
        ("sample_size_per_condition", config.get("sample_size_per_condition")),
        ("pool_id", config.get("pool_id")),
        ("source_sha256", config.get("source_sha256")),
        ("persona_modes", list(persona_modes)),
        ("model_id", model_id),
        ("prompt_version", VIRTUAL_RESPONSE_PROMPT_VERSION),
        ("score_noise_method", "fixed_scores_temperature_sampling"),
        ("response_temperature", config.get("response_temperature")),
        ("demographics_version", DEMOGRAPHICS_VERSION),
        ("score_prompt_version", MATCHED_CONDITION_PROMPT_VERSION),
        ("generator_version", MATCHED_CONDITION_GENERATOR_VERSION),
        ("virtual_sample_config", _normalize_concurrency_config(config)),
        ("conditions", list(conditions)),
    ):
        actual = existing_manifest.get(field)
        if field == "virtual_sample_config" and isinstance(actual, Mapping):
            actual = _normalize_concurrency_config(actual)
        if actual != expected:
            return False
    for field in ("item_bank_id", "item_bank_version", "item_bank_fingerprint"):
        expected = state.get(field)
        if expected not in (None, "", 0) and existing_manifest.get(field) != expected:
            return False
    return True


def _jsonl_record_count(path: Path) -> int:
    """Count non-empty JSONL records without changing the source file."""

    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _matched_output_has_complete_sjt_retest(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> bool:
    """Treat already-complete SJT/retest files as resumable evidence.

    This guard is deliberately independent of the historical simulation
    signature.  The response protocol changed after these two files had
    completed, so a signature-only guard must not force another SJT run.
    """

    expected_sjt = int(manifest.get("expected_sjt_records") or 0)
    expected_retest = int(
        manifest.get("expected_target_form_retest_records") or 0
    )
    if expected_sjt <= 0 or expected_retest <= 0:
        return False
    return (
        _jsonl_record_count(output_dir / "sjt_responses.jsonl")
        >= expected_sjt
        and _jsonl_record_count(
            output_dir / "target_form_retest_responses.jsonl"
        )
        >= expected_retest
    )


def _item_content_signature(item: Mapping[str, Any]) -> str:
    """Ignore version and bank identity when matching a local candidate."""

    payload = {
        "item_id": item.get("item_id"),
        "scenario": item.get("scenario"),
        "response_instruction": item.get("response_instruction"),
        "response_options": item.get("response_options") or [],
        "scoring_key": item.get("scoring_key") or {},
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _rewrite_response_identity(
    record: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    response_source: str,
    source_manifest_path: Path | None = None,
    source_generation_round: int | None = None,
    source_analysis_round: int | None = None,
) -> dict[str, Any]:
    """Bind a reusable raw response to the new frozen bank metadata."""

    rewritten = {
        **dict(record),
        "run_id": state.get("run_id"),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "response_source": response_source,
    }
    if source_manifest_path is not None:
        rewritten["reuse_source_manifest_path"] = str(source_manifest_path)
    if source_generation_round is not None:
        rewritten["reuse_source_generation_round"] = source_generation_round
    if source_analysis_round is not None:
        rewritten["reuse_source_analysis_round"] = source_analysis_round
    return rewritten


def _matched_source_is_compatible(
    source_manifest: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    model_id: str,
    respondent_refs: Sequence[Mapping[str, Any]] | None = None,
    source_manifest_path: Path | None = None,
) -> bool:
    """Check whether a source can safely seed the SJT/retest records.

    The IPIP reference protocol is versioned independently from the SJT and
    target-retest protocol.  A change from the old five-facet IPIP batches to
    the selected-facet batch must therefore invalidate only the IPIP records,
    not already-completed SJT and retest records.
    """

    compatible = True
    for field, expected in (
            ("schema_version", MATCHED_CONDITION_SCHEMA_VERSION),
            ("run_id", state.get("run_id")),
            ("sample_size_per_condition", config.get("sample_size_per_condition")),
            ("conditions", list(conditions)),
            ("persona_modes", [PERSONA_MODE_SCORE_PROFILE]),
            ("model_id", model_id),
            ("prompt_version", VIRTUAL_RESPONSE_PROMPT_VERSION),
            ("score_noise_method", "fixed_scores_temperature_sampling"),
            ("response_temperature", config.get("response_temperature")),
            ("demographics_version", DEMOGRAPHICS_VERSION),
            ("score_prompt_version", MATCHED_CONDITION_PROMPT_VERSION),
            ("generator_version", MATCHED_CONDITION_GENERATOR_VERSION),
        ("virtual_sample_config", _normalize_matched_source_config(config)),
        ):
        actual = source_manifest.get(field)
        if field == "virtual_sample_config" and isinstance(actual, Mapping):
            actual = _normalize_matched_source_config(actual)
        if actual != expected:
            compatible = False
            break
    if source_manifest.get("status") not in {"completed", "in_progress"}:
        return False
    if not compatible:
        return False
    if respondent_refs is not None:
        if source_manifest_path is None or not _source_profiles_match(
            source_manifest,
            source_manifest_path,
            respondent_refs=respondent_refs,
            config=config,
        ):
            return False
    return True


def _matched_source_ipip_is_compatible(
    source_manifest: Mapping[str, Any],
) -> bool:
    """Whether an old manifest's IPIP records use the current protocol."""

    reference_meta = source_manifest.get("reference_questionnaires") or {}
    ipip_meta = reference_meta.get("ipip_neo") if isinstance(reference_meta, Mapping) else None
    return isinstance(ipip_meta, Mapping) and (
        ipip_meta.get("prompt_version") == IPIP_NEO_REFERENCE_PROMPT_VERSION
        and ipip_meta.get("response_mode")
        == "selected_facets_single_batch_per_respondent"
    )


def _copy_frozen_reference_questionnaire(
    *,
    source_manifest_ref: str | Path,
    destination_path: Path,
    state: Mapping[str, Any],
    target_references: Sequence[Mapping[str, Any]],
    respondent_refs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    scales: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy the completed first-round IPIP responses into a later-round file."""

    manifest_path = Path(source_manifest_ref).resolve()
    if not manifest_path.is_file():
        raise ValueError(
            "后续轮次缺少首轮校标组卷 manifest；拒绝重新施测校标组卷"
        )
    try:
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("首轮校标组卷 manifest 无法读取") from exc
    if not isinstance(source_manifest, Mapping):
        raise ValueError("首轮校标组卷 manifest 格式无效")
    if not _source_profiles_match(
        source_manifest,
        manifest_path,
        respondent_refs=respondent_refs,
        config=config,
    ):
        raise ValueError("首轮 manifest 的冻结被试资料与当前 cohort 不一致；拒绝复用")
    if (
        source_manifest.get("status") != "completed"
        or source_manifest.get("run_id") != state.get("run_id")
        or int(
            (source_manifest.get("virtual_sample_config") or {}).get(
                "generation_round"
            )
            or 1
        ) != 1
    ):
        raise ValueError(
            "冻结校标组卷必须来自本运行中已完成的第一轮；拒绝重复施测"
        )
    reference_meta = source_manifest.get("reference_questionnaires") or {}
    ipip_meta = (
        reference_meta.get("ipip_neo")
        if isinstance(reference_meta, Mapping)
        else None
    )
    if (
        not isinstance(ipip_meta, Mapping)
        or not ipip_meta.get("enabled")
        or ipip_meta.get("prompt_version")
        != IPIP_NEO_REFERENCE_PROMPT_VERSION
        or ipip_meta.get("response_mode")
        != "selected_facets_single_batch_per_respondent"
    ):
        raise ValueError("首轮校标组卷不是当前可复用的 IPIP 施测版本")
    expected_facet_ids = [str(scale.get("facet_id")) for scale in scales]
    expected_facet_codes = [str(scale.get("facet_code")) for scale in scales]
    if (
        ipip_meta.get("facet_ids") != expected_facet_ids
        or ipip_meta.get("facet_codes") != expected_facet_codes
    ):
        raise ValueError("首轮校标组卷 facet 与当前题库不一致，不能复用")
    source_ipip_ref = ipip_meta.get("path")
    if not isinstance(source_ipip_ref, str) or not source_ipip_ref:
        raise ValueError("首轮 manifest 缺少校标组卷响应路径")
    source_ipip_path = Path(source_ipip_ref)
    if not source_ipip_path.is_absolute():
        source_ipip_path = manifest_path.parent / source_ipip_path
    source_ipip_path = source_ipip_path.resolve()
    if not source_ipip_path.is_file():
        raise ValueError("首轮校标组卷响应文件缺失；拒绝重新施测")

    expected_records: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for reference in target_references:
        respondent_id = str(reference.get("respondent_id") or "")
        for scale in scales:
            facet_code = str(scale.get("facet_code") or "")
            facet_id = str(scale.get("facet_id") or "")
            for item in scale.get("items") or []:
                key = (respondent_id, facet_code, str(item.get("item_id") or ""))
                if not all(key) or key in expected_records:
                    raise ValueError("当前校标组卷存在空值或重复题目 ID")
                expected_records[key] = {
                    "matched_subject_id": str(
                        reference.get("matched_subject_id") or ""
                    ),
                    "facet_id": facet_id,
                    "score_values": dict(reference.get("score_values") or {}),
                    "demographics": deepcopy(reference.get("demographics")),
                }

    source_records = _load_jsonl_records(source_ipip_path)
    source_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in source_records:
        if (
            record.get("record_type") != "ipip_neo_response"
            or record.get("condition_id") != "target"
            or record.get("prompt_version") != IPIP_NEO_REFERENCE_PROMPT_VERSION
        ):
            raise ValueError("首轮校标组卷响应记录不是当前 IPIP 协议")
        key = (
            str(record.get("respondent_id") or ""),
            str(record.get("facet_code") or ""),
            str(record.get("item_id") or ""),
        )
        if key in source_by_key:
            raise ValueError("首轮校标组卷响应含重复记录，拒绝复用")
        source_by_key[key] = record
    if set(source_by_key) != set(expected_records):
        raise ValueError("首轮校标组卷记录不完整或与当前 target cohort 不匹配")
    for key, expected in expected_records.items():
        record = source_by_key[key]
        if (
            str(record.get("matched_subject_id") or "")
            != expected["matched_subject_id"]
            or str(record.get("facet_id") or "") != expected["facet_id"]
            or record.get("score_values") != expected["score_values"]
            or record.get("demographics") != expected["demographics"]
        ):
            raise ValueError("首轮校标记录与固定 target cohort 不一致")

    existing_records = _load_jsonl_records(destination_path)
    for record in existing_records:
        if (
            record.get("response_source")
            != "reused_frozen_first_round_reference"
            or record.get("frozen_reference_manifest")
            != str(manifest_path)
        ):
            raise ValueError(
                "后续轮次输出目录已含非冻结校标记录；拒绝覆盖或混合"
            )
    existing_keys = {
        (
            str(record.get("respondent_id") or ""),
            str(record.get("facet_code") or ""),
            str(record.get("item_id") or ""),
        )
        for record in existing_records
    }
    if not existing_keys.issubset(expected_records):
        raise ValueError("后续轮次已有校标记录超出冻结结果范围")
    missing = [
        {
            **source_by_key[key],
            "response_source": "reused_frozen_first_round_reference",
            "frozen_reference_manifest": str(manifest_path),
            "reuse_source_manifest_path": str(manifest_path),
            "reuse_source_generation_round": 1,
        }
        for key in expected_records
        if key not in existing_keys
    ]
    if missing:
        _append_jsonl_records(destination_path, missing)
    return {
        "source_manifest_path": str(manifest_path),
        "source_ipip_path": str(source_ipip_path),
        "expected_records": len(expected_records),
        "copied_records": len(missing),
        "reused_existing_records": len(existing_records),
        "source_generation_round": 1,
    }


def _local_retest_cache_records(
    state: Mapping[str, Any],
    *,
    current_items: Mapping[str, Mapping[str, Any]],
    expected_count: int,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Find the best local-retest cache for each final repaired item."""

    found: dict[str, list[dict[str, Any]]] = {}
    history = state.get("psychometric_repair_history") or []
    for event in reversed(history):
        if not isinstance(event, Mapping) or event.get("event") != "psychometric_item_repaired":
            continue
        local_retest = event.get("local_retest")
        if not isinstance(local_retest, Mapping):
            continue
        for round_result in reversed(local_retest.get("history") or []):
            if not isinstance(round_result, Mapping):
                continue
            candidate = round_result.get("candidate_item")
            if not isinstance(candidate, Mapping):
                continue
            item_id = str(candidate.get("item_id") or "")
            final_item = current_items.get(item_id)
            if final_item is None or item_id in found:
                continue
            if _item_version_value(candidate) != _item_version_value(final_item):
                continue
            if _item_content_signature(candidate) != _item_content_signature(final_item):
                continue
            simulation = round_result.get("simulation") or {}
            cache_path = simulation.get("cache_path")
            if not isinstance(cache_path, str) or not Path(cache_path).is_file():
                continue
            try:
                cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if not isinstance(cached, Mapping) or (
                cached.get("virtual_sample_config") != _normalize_concurrency_config(config)
                or cached.get("prompt_version") != VIRTUAL_RESPONSE_PROMPT_VERSION
            ):
                continue
            records = cached.get("records") if isinstance(cached, Mapping) else None
            if not isinstance(records, list) or len(records) != expected_count:
                continue
            normalized = [
                _rewrite_response_identity(
                    {**dict(record), "record_type": "sjt_response"},
                    state,
                    response_source="reused_local_item_retest",
                )
                for record in records
                if isinstance(record, Mapping)
                and str(record.get("item_id") or "") == item_id
                and record.get("item_version") == _item_version_value(final_item)
            ]
            if len(normalized) == expected_count:
                found[item_id] = normalized
    return found


def _facet_iteration_response_refs(
    state: Mapping[str, Any],
    *,
    item_versions: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Index facet response sources only for the exact measured item versions."""

    iteration = state.get("facet_iteration_state") or {}
    if not isinstance(iteration, Mapping):
        return {}
    sources: dict[str, str] = {}
    for collection_name in ("accepted_facets", "baseline_facets"):
        collection = iteration.get(collection_name) or {}
        if not isinstance(collection, Mapping):
            continue
        for facet in collection.values():
            if not isinstance(facet, Mapping):
                continue
            response_ref = facet.get("response_data_ref")
            if not isinstance(response_ref, str) or not response_ref:
                continue
            source_versions = facet.get("item_versions") or {}
            if not isinstance(source_versions, Mapping):
                continue
            for item_id, source_version in source_versions.items():
                item_id = str(item_id)
                if item_versions is not None and (
                    isinstance(source_version, bool)
                    or item_versions.get(item_id) != source_version
                ):
                    continue
                sources.setdefault(item_id, response_ref)
    return sources


def _qualified_item_source_round(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
) -> tuple[int | None, str | None]:
    """Return the locked qualification source for an unchanged item version."""

    item_id = str(item.get("item_id") or "")
    disposition = (state.get("item_final_dispositions") or {}).get(item_id) or {}
    if not isinstance(disposition, Mapping):
        return None, None
    if disposition.get("status") != "qualified_locked":
        return None, None
    disposition_version = disposition.get("item_version")
    if disposition_version is not None and disposition_version != _item_version_value(item):
        return None, None
    raw_round = disposition.get("qualification_analysis_round")
    qualification_round = (
        raw_round
        if isinstance(raw_round, int) and not isinstance(raw_round, bool)
        else None
    )
    response_ref = disposition.get("qualification_response_data_ref")
    if not isinstance(response_ref, str) or not response_ref:
        history = state.get("psychometric_iteration_history") or []
        source_record = next(
            (
                row
                for row in reversed(history)
                if isinstance(row, Mapping)
                and row.get("analysis_round") == qualification_round
                and isinstance(row.get("response_data_ref"), str)
                and isinstance(row.get("item_snapshots"), Mapping)
                and isinstance(row["item_snapshots"].get(item_id), Mapping)
                and row["item_snapshots"][item_id].get("version")
                == _item_version_value(item)
            ),
            None,
        )
        response_ref = source_record.get("response_data_ref") if source_record else None
    if not isinstance(response_ref, str) or not response_ref:
        response_ref = _facet_iteration_response_refs(
            state,
            item_versions={item_id: _item_version_value(item)},
        ).get(item_id)
    return qualification_round, response_ref


def _seed_matched_response_files(
    *,
    state: Mapping[str, Any],
    output_dir: Path,
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    model_id: str,
    respondent_refs: Sequence[Mapping[str, Any]],
    sjt_items: Sequence[Mapping[str, Any]],
    sjt_path: Path,
    target_retest_path: Path,
    ipip_neo_path: Path,
    option_order_path: Path,
) -> dict[str, Any]:
    """Seed a new formal bank with reusable raw records.

    The bank identity changes after repair, so the old response manifest cannot
    be reused as-is. Its records can nevertheless be copied after identity
    rewriting when the item version and protocol still match. Local candidate
    caches take precedence for repaired items; aggregate statistics are never
    merged.
    """

    generation_round = int(config.get("generation_round") or 1)
    current_versions = {
        str(item.get("item_id")): _item_version_value(item)
        for item in state.get("frozen_item_bank") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    source_ref = state.get("previous_virtual_response_data_ref")
    candidate_paths: list[Path] = []
    if isinstance(source_ref, str) and source_ref:
        candidate_paths.append(Path(source_ref).resolve())
    for ref_field in (
        "virtual_response_data_ref",
        "first_round_reference_ref",
        "cohort_source_ref",
        "frozen_reference_questionnaire_ref",
    ):
        value = state.get(ref_field)
        if isinstance(value, str) and value:
            candidate_paths.append(Path(value).resolve())
    for item in state.get("frozen_item_bank") or []:
        if not isinstance(item, Mapping):
            continue
        _qualification_round, qualification_ref = _qualified_item_source_round(
            state, item
        )
        if qualification_ref:
            candidate_paths.append(Path(qualification_ref).resolve())
    candidate_paths.extend(
        Path(value).resolve()
        for value in _facet_iteration_response_refs(
            state, item_versions=current_versions
        ).values()
    )
    candidate_paths.extend(
        path.resolve()
        for path in output_dir.parent.glob("bank-*/manifest.json")
        if path.resolve() != (output_dir / "manifest.json").resolve()
    )
    run_output_dir = output_dir.parent.parent
    candidate_paths.extend(
        path.resolve()
        for path in run_output_dir.glob("matched-v*/bank-*/manifest.json")
        if path.resolve() != (output_dir / "manifest.json").resolve()
    )
    seen_paths: set[Path] = set()
    source_manifests: list[tuple[Path, Mapping[str, Any]]] = []
    ordered_paths = sorted(
        candidate_paths,
        key=lambda path: path.stat().st_mtime if path.is_file() else 0,
        reverse=True,
    )
    for candidate in ordered_paths:
        if candidate in seen_paths or not candidate.is_file():
            continue
        seen_paths.add(candidate)
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(loaded, Mapping):
            continue
        if _matched_source_is_compatible(
            loaded,
            state=state,
            config=config,
            conditions=conditions,
            model_id=model_id,
            respondent_refs=respondent_refs,
            source_manifest_path=candidate,
        ):
            source_manifests.append((candidate, loaded))
            continue
        if (
            generation_round > 1
            and isinstance(source_ref, str)
            and source_ref
            and candidate == Path(source_ref).resolve()
            and loaded.get("status") in {"completed", "in_progress"}
        ):
            raise ValueError(
                "前序响应 manifest 的 score_profiles 或冻结设定与当前固定 cohort 不一致；拒绝跨轮混用"
            )
    source_manifests.sort(
        key=lambda pair: (
            _manifest_generation_round(pair[1]),
            pair[0].stat().st_mtime if pair[0].is_file() else 0,
        ),
        reverse=True,
    )
    if not source_manifests:
        if generation_round > 1:
            raise ValueError(
                "后续轮次缺少可验证的同 cohort 源 manifest；拒绝重新采样或静默重新施测"
            )
        return {
            "source_manifest_path": None,
            "source_compatible": False,
            "reused_sjt_records": 0,
            "reused_local_item_count": 0,
            "reused_target_retest_records": 0,
            "reused_ipip_neo_records": 0,
        }
    source_manifest_path, source_manifest = source_manifests[0]

    current_versions = {
        str(item.get("item_id")): _item_version_value(item)
        for item in sjt_items
        if isinstance(item, Mapping) and item.get("item_id")
    }
    current_items = {
        str(item.get("item_id")): item
        for item in state.get("frozen_item_bank") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    item_source_paths: dict[str, set[Path]] = {}
    locked_source_rounds: dict[str, int] = {}
    locked_item_ids: set[str] = set()
    for item_id, item in current_items.items():
        if item_id not in current_versions:
            continue
        qualification_round, explicit_ref = _qualified_item_source_round(
            state,
            item,
        )
        if qualification_round is None and explicit_ref is None:
            continue
        selected_source: tuple[Path, Mapping[str, Any]] | None = None
        if explicit_ref:
            explicit_path = Path(explicit_ref).resolve()
            selected_source = next(
                (
                    pair
                    for pair in source_manifests
                    if pair[0] == explicit_path
                ),
                None,
            )
            if selected_source is None:
                raise ValueError(
                    f"已锁定题目 {item_id} 的资格来源 manifest 缺失或不兼容；拒绝重新施测"
                )
        elif qualification_round is not None:
            selected_source = next(
                (
                    pair
                    for pair in source_manifests
                    if _manifest_generation_round(pair[1]) == qualification_round
                ),
                None,
            )
            if selected_source is None:
                raise ValueError(
                    f"已锁定题目 {item_id} 的资格分析轮次 {qualification_round} 缺少来源 manifest；拒绝重新施测"
                )
        if selected_source is not None:
            source_path, source_data = selected_source
            if source_data.get("status") != "completed":
                raise ValueError(
                    f"已锁定题目 {item_id} 的资格来源尚未完整完成；拒绝重新施测"
                )
            item_source_paths[item_id] = {source_path}
            locked_item_ids.add(item_id)
            if qualification_round is not None:
                locked_source_rounds[item_id] = qualification_round
    default_source_paths = {path for path, _manifest in source_manifests}
    for item_id in current_versions:
        item_source_paths.setdefault(item_id, set(default_source_paths))

    reference_by_identity: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for reference in respondent_refs:
        identity = (
            str(reference.get("respondent_id") or ""),
            str(reference.get("condition_id") or ""),
            str(reference.get("matched_subject_id") or ""),
        )
        if not all(identity) or identity in reference_by_identity:
            raise ValueError("固定 cohort 中存在空或重复的 respondent/condition/matched_subject 键")
        reference_by_identity[identity] = reference

    def record_matches_reference(record: Mapping[str, Any]) -> bool:
        identity = (
            str(record.get("respondent_id") or ""),
            str(record.get("condition_id") or ""),
            str(record.get("matched_subject_id") or ""),
        )
        reference = reference_by_identity.get(identity)
        return bool(
            reference is not None
            and record.get("persona_mode") == PERSONA_MODE_SCORE_PROFILE
            and record.get("score_values") == reference.get("score_values")
            and record.get("demographics") == reference.get("demographics")
        )

    expected_per_condition = int(config.get("sample_size_per_condition") or 0)
    expected_local_records = expected_per_condition
    local_records_by_item = _local_retest_cache_records(
        state,
        current_items=current_items,
        expected_count=expected_local_records,
        config=config,
    )
    local_item_ids = set(local_records_by_item) - locked_item_ids
    reusable_sjt: list[dict[str, Any]] = []
    seen_sjt: set[tuple[Any, ...]] = set()
    existing_sjt = _canonicalize_item_response_keys(
        _load_jsonl_keys(
            sjt_path,
            (
                "respondent_id",
                "persona_mode",
                "item_id",
                "item_version",
                "condition_id",
                "matched_subject_id",
            ),
        ),
        item_versions=current_versions,
    )
    for candidate_path, candidate_manifest in source_manifests:
        source_item_bank_matches = all(
            not state.get(field)
            or candidate_manifest.get(field) == state.get(field)
            for field in (
                "item_bank_id",
                "item_bank_version",
                "item_bank_fingerprint",
            )
        )
        for record in _load_jsonl_records(
            candidate_path.parent / "sjt_responses.jsonl"
        ):
            item_id = str(record.get("item_id") or "")
            if (
                item_id not in current_versions
                or candidate_path not in item_source_paths.get(item_id, set())
                or item_id in local_item_ids
                or record.get("record_type") != "sjt_response"
                or not record.get("selected_option_id")
                or not record_matches_reference(record)
            ):
                continue
            source_version = record.get("item_version")
            canonical_source_version = (
                current_versions.get(item_id)
                if source_version is None and source_item_bank_matches
                else source_version
            )
            key = (
                record.get("respondent_id"),
                record.get("persona_mode"),
                item_id,
                canonical_source_version,
                record.get("condition_id"),
                record.get("matched_subject_id"),
            )
            version_matches = source_version == current_versions.get(item_id) or (
                source_version is None
                and source_item_bank_matches
                and item_id in current_versions
            )
            if not version_matches or key in existing_sjt or key in seen_sjt:
                continue
            seen_sjt.add(key)
            rewritten = _rewrite_response_identity(
                record,
                state,
                response_source="reused_unchanged_item",
                source_manifest_path=candidate_path,
                source_generation_round=_manifest_generation_round(
                    candidate_manifest
                ),
                source_analysis_round=locked_source_rounds.get(item_id),
            )
            rewritten["item_version"] = current_versions[item_id]
            reusable_sjt.append(rewritten)
    for item_id, records in local_records_by_item.items():
        for record in records:
            key = tuple(
                record.get(field)
                for field in (
                    "respondent_id",
                    "persona_mode",
                    "item_id",
                    "item_version",
                    "condition_id",
                    "matched_subject_id",
                )
            )
            if key in existing_sjt or key in seen_sjt:
                continue
            reusable_sjt.append(record)
            seen_sjt.add(key)
    if reusable_sjt:
        _append_jsonl_records(sjt_path, reusable_sjt)

    option_orders: list[dict[str, Any]] = []
    option_order_key_fields = (
        "respondent_id",
        "condition_id",
        "matched_subject_id",
        "item_id",
        "item_version",
    )
    existing_option_orders = _canonicalize_item_response_keys(
        _load_jsonl_keys(
            option_order_path,
            option_order_key_fields,
        ),
        item_versions=current_versions,
        item_id_index=3,
        version_index=4,
    )
    seen_option_orders: set[tuple[Any, ...]] = set()
    reused_sjt_order_keys = {
        (
            record.get("respondent_id"),
            record.get("condition_id"),
            record.get("matched_subject_id"),
            record.get("item_id"),
            record.get("item_version"),
        )
        for record in reusable_sjt
    }
    for candidate_path, candidate_manifest in source_manifests:
        source_item_bank_matches = all(
            not state.get(field)
            or candidate_manifest.get(field) == state.get(field)
            for field in (
                "item_bank_id",
                "item_bank_version",
                "item_bank_fingerprint",
            )
        )
        source_option_ref = candidate_manifest.get("option_order_path")
        if not isinstance(source_option_ref, str) or not source_option_ref:
            continue
        source_option_path = Path(source_option_ref)
        if not source_option_path.is_absolute():
            source_option_path = candidate_path.parent / source_option_path
        if not source_option_path.is_file():
            continue
        for record in _load_jsonl_records(source_option_path):
            item_id = str(record.get("item_id") or "")
            if (
                item_id not in current_versions
                or candidate_path not in item_source_paths.get(item_id, set())
            ):
                continue
            source_version = record.get("item_version")
            version_matches = source_version == current_versions.get(item_id) or (
                source_version is None
                and source_item_bank_matches
                and item_id in current_versions
            )
            if version_matches:
                key_values = [record.get(field) for field in option_order_key_fields]
                if key_values[-1] is None and source_item_bank_matches:
                    key_values[-1] = current_versions[item_id]
                key = tuple(key_values)
                matching_sjt_key = (
                    record.get("respondent_id"),
                    record.get("condition_id"),
                    record.get("matched_subject_id"),
                    item_id,
                    current_versions[item_id],
                )
                if (
                    matching_sjt_key in reused_sjt_order_keys
                    and key not in existing_option_orders
                    and key not in seen_option_orders
                ):
                    rewritten = _rewrite_response_identity(
                        record,
                        state,
                        response_source="reused_unchanged_item",
                        source_manifest_path=candidate_path,
                        source_generation_round=_manifest_generation_round(
                            candidate_manifest
                        ),
                        source_analysis_round=locked_source_rounds.get(item_id),
                    )
                    rewritten["item_version"] = current_versions[item_id]
                    option_orders.append(rewritten)
                    seen_option_orders.add(key)
    for records in local_records_by_item.values():
        for record in records:
            key = tuple(record.get(field) for field in option_order_key_fields)
            if key in existing_option_orders or key in seen_option_orders:
                continue
            option_orders.append({
                key: record.get(key)
                for key in (
                    "run_id", "respondent_id", "condition_id", "matched_subject_id",
                    "item_id", "item_version", "display_option_order",
                    "raw_display_option_id", "selected_option_id",
                )
            })
            seen_option_orders.add(key)
    if option_orders:
        _append_jsonl_records(option_order_path, option_orders)

    target_retest_records: list[dict[str, Any]] = []
    existing_target_retest = _canonicalize_item_response_keys(
        _load_jsonl_keys(
            target_retest_path,
            (
                "respondent_id",
                "persona_mode",
                "item_id",
                "item_version",
                "condition_id",
                "matched_subject_id",
                "administration_id",
            ),
        ),
        item_versions=current_versions,
    )
    seen_target_retest: set[tuple[Any, ...]] = set()
    administration_count = int(
        config.get(
            "target_form_administration_count",
            DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT,
        )
    )
    current_administrations = set(range(2, administration_count + 1))
    for candidate_path, candidate_manifest in source_manifests:
        source_item_bank_matches = all(
            not state.get(field)
            or candidate_manifest.get(field) == state.get(field)
            for field in (
                "item_bank_id",
                "item_bank_version",
                "item_bank_fingerprint",
            )
        )
        source_target_meta = candidate_manifest.get("target_form_retest") or {}
        source_target_ref = (
            source_target_meta.get("path")
            if isinstance(source_target_meta, Mapping)
            else None
        )
        allowed_administrations = {
            int(value)
            for value in source_target_meta.get("administration_ids") or []
            if isinstance(value, int) and not isinstance(value, bool)
        } & current_administrations if isinstance(source_target_meta, Mapping) else set()
        if not isinstance(source_target_ref, str) or not source_target_ref:
            continue
        source_target_path = Path(source_target_ref)
        if not source_target_path.is_absolute():
            source_target_path = candidate_path.parent / source_target_path
        if not source_target_path.is_file():
            continue
        for record in _load_jsonl_records(source_target_path):
            item_id = str(record.get("item_id") or "")
            if (
                item_id not in current_versions
                or candidate_path not in item_source_paths.get(item_id, set())
                or record.get("record_type") != "sjt_target_form_retest_response"
                or not record.get("selected_option_id")
                or not record_matches_reference(record)
            ):
                continue
            source_version = record.get("item_version")
            canonical_source_version = (
                current_versions.get(item_id)
                if source_version is None and source_item_bank_matches
                else source_version
            )
            version_matches = source_version == current_versions.get(item_id) or (
                source_version is None
                and source_item_bank_matches
                and item_id in current_versions
            )
            key = (
                record.get("respondent_id"),
                record.get("persona_mode"),
                item_id,
                canonical_source_version,
                record.get("condition_id"),
                record.get("matched_subject_id"),
                record.get("administration_id"),
            )
            if (
                version_matches
                and record.get("administration_id") in allowed_administrations
                and key not in existing_target_retest
                and key not in seen_target_retest
            ):
                rewritten = _rewrite_response_identity(
                    record,
                    state,
                    response_source="reused_unchanged_item",
                    source_manifest_path=candidate_path,
                    source_generation_round=_manifest_generation_round(
                        candidate_manifest
                    ),
                    source_analysis_round=locked_source_rounds.get(item_id),
                )
                rewritten["item_version"] = current_versions[item_id]
                target_retest_records.append(rewritten)
                seen_target_retest.add(key)
    if target_retest_records:
        _append_jsonl_records(target_retest_path, target_retest_records)

    if locked_item_ids:
        available_sjt_keys = existing_sjt | seen_sjt
        available_retest_keys = existing_target_retest | seen_target_retest
        target_references = [
            reference
            for reference in respondent_refs
            if str(reference.get("condition_id")) == "target"
        ]
        for item_id in sorted(locked_item_ids):
            required_sjt = {
                (
                    str(reference.get("respondent_id")),
                    PERSONA_MODE_SCORE_PROFILE,
                    item_id,
                    current_versions[item_id],
                    str(reference.get("condition_id")),
                    str(reference.get("matched_subject_id")),
                )
                for reference in respondent_refs
            }
            if not required_sjt.issubset(available_sjt_keys):
                raise ValueError(
                    f"已锁定题目 {item_id} 的 qualification source 缺少完整主测记录；拒绝重新施测"
                )
            required_retest = {
                (
                    str(reference.get("respondent_id")),
                    PERSONA_MODE_SCORE_PROFILE,
                    item_id,
                    current_versions[item_id],
                    "target",
                    str(reference.get("matched_subject_id")),
                    administration_id,
                )
                for reference in target_references
                for administration_id in current_administrations
            }
            if not required_retest.issubset(available_retest_keys):
                raise ValueError(
                    f"已锁定题目 {item_id} 的 qualification source 缺少完整整卷重测记录；拒绝重新施测"
                )

    reused_reference_counts: dict[str, int] = {}
    references = source_manifest.get("reference_questionnaires") or {}
    for name, target_path in (
        (("ipip_neo", ipip_neo_path),) if generation_round == 1 else ()
    ):
        metadata = references.get(name) if isinstance(references, Mapping) else None
        source_path = metadata.get("path") if isinstance(metadata, Mapping) else None
        records: list[dict[str, Any]] = []
        if (
            _matched_source_ipip_is_compatible(source_manifest)
            and isinstance(source_path, str)
        ):
            resolved_source_path = Path(source_path)
            if not resolved_source_path.is_absolute():
                resolved_source_path = source_manifest_path.parent / resolved_source_path
            candidate_records = (
                _load_jsonl_records(resolved_source_path)
                if resolved_source_path.is_file()
                else []
            )
            expected_records = int(
                source_manifest.get("expected_ipip_neo_records") or -1
            )
            if len(candidate_records) == expected_records:
                records = candidate_records
        if records:
            existing_reference_keys = _load_jsonl_keys(
                target_path,
                ("respondent_id", "facet_code", "item_id"),
            )
            reusable_records: list[dict[str, Any]] = []
            seen_reference_keys: set[tuple[Any, ...]] = set()
            for record in records:
                key = tuple(
                    record.get(field)
                    for field in ("respondent_id", "facet_code", "item_id")
                )
                if key in existing_reference_keys or key in seen_reference_keys:
                    continue
                seen_reference_keys.add(key)
                reusable_records.append(
                    _rewrite_response_identity(
                        record,
                        state,
                        response_source=f"reused_{name}",
                        source_manifest_path=source_manifest_path,
                        source_generation_round=_manifest_generation_round(
                            source_manifest
                        ),
                    )
                )
            if reusable_records:
                _append_jsonl_records(target_path, reusable_records)
            reused_reference_counts[name] = len(reusable_records)
        else:
            reused_reference_counts[name] = 0

    return {
        "source_manifest_path": str(source_manifest_path),
        "source_compatible": True,
        "reused_sjt_records": len(reusable_sjt),
        "reused_local_item_count": len(local_records_by_item),
        "reused_target_retest_records": len(target_retest_records),
        "reused_ipip_neo_records": reused_reference_counts.get("ipip_neo", 0),
    }


def balanced_option_order(
    option_ids: Sequence[str],
    *,
    respondent_index: int,
    seed: int,
    item_id: str,
) -> list[str]:
    """Return one deterministic row of a balanced Latin-square ordering."""

    ids = list(option_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("选项顺序平衡要求至少两个不重复的选项ID")
    base_seed = int.from_bytes(
        sha256(f"{seed}:{item_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    shuffled = list(ids)
    import random

    random.Random(base_seed).shuffle(shuffled)
    size = len(shuffled)
    first_row: list[int] = []
    left, right = 0, size - 1
    while left <= right:
        first_row.append(left)
        left += 1
        if left <= right:
            first_row.append(right)
            right -= 1
    row_index = respondent_index % size
    return [shuffled[(index + row_index) % size] for index in first_row]


class VirtualResponseRunner:
    """全量并发执行SJT、IPIP-NEO参照问卷与整卷重测。"""

    def __init__(
        self,
        *,
        base_model: Any | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = 1.0,
        request_timeout_seconds: float | None = None,
        response_temperature: float = DEFAULT_RESPONSE_TEMPERATURE,
    ) -> None:
        validate_max_concurrency(max_concurrency)
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
            or max_retries > 4
        ):
            raise ValueError("max_retries 必须是 0 到 4 之间的整数（最多5次尝试）")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds 不能为负数")
        if request_timeout_seconds is None:
            request_timeout_seconds = get_model_request_timeout_seconds()
        if (
            not isinstance(request_timeout_seconds, (int, float))
            or isinstance(request_timeout_seconds, bool)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds 必须是正数")

        self.response_temperature = resolve_response_temperature(response_temperature)
        self.base_model = base_model or get_model(temperature=self.response_temperature)
        if hasattr(self.base_model, "temperature") and getattr(self.base_model, "temperature") != self.response_temperature:
            copy_model = getattr(self.base_model, "model_copy", None)
            if not callable(copy_model):
                raise ValueError("传入的虚拟作答模型温度与冻结配置不一致，且不能独立复制设置")
            self.base_model = copy_model(update={"temperature": self.response_temperature})
        self.sjt_model, self.structured_output_method = (
            with_compatible_structured_output(
                self.base_model,
                SJTSelectionOutput,
            )
        )
        self.sjt_batch_model, batch_method = with_compatible_structured_output(
            self.base_model,
            SJTBatchSelectionOutput,
        )
        self.ipip_neo_model, reference_method = (
            with_compatible_structured_output(
                self.base_model,
                IPIPNEOBatchOutput,
            )
        )
        # Historical experiment add-ons still administer NEO-FFI explicitly.
        # The output contract is the same ratings-only schema, so keep the old
        # attribute as a compatibility alias without using it in the main flow.
        self.neo_ffi_model = self.ipip_neo_model
        self.persona_summary_model, summary_method = (
            with_compatible_structured_output(
                self.base_model,
                PersonaSummaryOutput,
            )
        )
        if (
            reference_method != self.structured_output_method
            or summary_method != self.structured_output_method
            or batch_method != self.structured_output_method
        ):
            raise ValueError("同一模型的结构化输出方式不一致")
        self.max_concurrency = 0
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.semaphore = UnlimitedConcurrency()
        self.write_lock = asyncio.Lock()
        self.model_id = (
            getattr(self.base_model, "model_name", None)
            or getattr(self.base_model, "model", None)
            or os.getenv("MODEL_ID")
            or "unknown"
        )

    async def _append_record(
        self,
        path: Path,
        record: Mapping[str, Any],
    ) -> None:
        async with self.write_lock:
            _append_jsonl_records(path, [record])

    async def _append_records(
        self,
        path: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        async with self.write_lock:
            _append_jsonl_records(path, records)

    async def _run_matched_condition_profile(
        self,
        *,
        state: PSJTState,
        output_dir: Path,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
        respondent_refs: Sequence[Mapping[str, Any]],
        criterion: Mapping[str, str],
        ipip_neo_path: str | Path,
    ) -> dict[str, Any]:
        """Run the matched-facet protocol plus a target-only form retest."""

        if not matched_condition_sample_is_current(config, list(respondent_refs)):
            raise ValueError("旧协议或不完整的虚拟被试资料不能用于施测；请重新配置，保留已有结果")
        if config["response_temperature"] != self.response_temperature:
            raise ValueError("虚拟作答模型温度与冻结样本配置不一致")
        if config.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION:
            raise ValueError(f"匹配 facet 虚拟作答需要 schema_version={MATCHED_CONDITION_SCHEMA_VERSION}；旧配置必须重新配置")
        conditions = config.get("conditions")
        if not isinstance(conditions, list) or {
            str(row.get("condition_id")) for row in conditions if isinstance(row, Mapping)
        } != set(MATCHED_CONDITION_IDS):
            raise ValueError("虚拟样本配置必须包含固定的三个匹配臂")
        condition_groups = flatten_matched_condition_groups(conditions)
        condition_ids = tuple(str(row.get("condition_id")) for row in condition_groups)
        if "target" not in condition_ids or len(set(condition_ids)) != len(condition_ids):
            raise ValueError("虚拟样本配置的 facet groups 缺少唯一 target 或存在重复 condition_id")
        profiles = resolve_virtual_respondent_profiles(
            respondent_refs, demographics_snapshot=config["demographics_snapshot"],
        )
        profile_by_id = {profile["respondent_id"]: profile for profile in profiles}
        persona_modes = _resolve_persona_modes(config)
        if persona_modes != [PERSONA_MODE_SCORE_PROFILE]:
            raise ValueError("匹配 facet 协议只支持 score_profile")
        persona_mode = persona_modes[0]
        sjt_items = list(context["items"])
        condition_rows = {
            str(row["condition_id"]): dict(row) for row in condition_groups
        }
        target_dimension_id = str(
            condition_rows["target"].get("dimension_id") or ""
        )
        # The reference scope must be identical to the construct scope used
        # during authoring. This also supports a form containing only one or a
        # small subset of facets instead of silently administering a fixed
        # reference set.
        selected_ipip_facet_ids = resolve_ipip_neo_reference_facet_ids(
            state,
            sjt_items,
        )
        # Experiment-only opt-out.
        ipip_scales = (
            None
            if config.get("reference_questionnaires_enabled") is False
            or config.get("ipip_neo_reference_enabled") is False
            else load_ipip_neo_facet_scales(
                selected_ipip_facet_ids,
                ipip_neo_path,
            )
        )
        target_ipip_scale = next(
            (
                scale
                for scale in (ipip_scales or [])
                if str(scale.get("facet_id") or "") == target_dimension_id
            ),
            None,
        )
        seed = int(config.get("seed", 0))
        all_score_specs = deepcopy(config["score_specs"])

        output_dir.mkdir(parents=True, exist_ok=True)
        sjt_path = output_dir / "sjt_responses.jsonl"
        target_retest_path = output_dir / "target_form_retest_responses.jsonl"
        ipip_path = output_dir / "ipip_neo_responses.jsonl"
        option_order_path = output_dir / "option_orders.jsonl"
        manifest_path = output_dir / "manifest.json"
        scoring_snapshot_path = output_dir / "scoring_snapshot.json"
        score_profiles_path = output_dir / "score_profiles.json"
        score_profiles_csv_path = output_dir / "score_profiles.csv"
        target_references = [
            reference
            for reference in respondent_refs
            if str(reference.get("condition_id")) == "target"
        ]
        target_reference_ids = {
            str(reference.get("respondent_id"))
            for reference in target_references
        }
        if len(target_reference_ids) != len(target_references):
            raise ValueError("匹配 facet 目标条件的参照问卷被试ID重复")
        if not target_references:
            raise ValueError("匹配 facet 协议至少需要一个 target 被试用于参照问卷")
        signature = _simulation_signature(
            state=state,
            respondent_refs=respondent_refs,
            config=config,
            persona_modes=persona_modes,
            criterion=criterion,
            model_id=self.model_id,
        )
        signature_migrated_from: str | None = None
        existing_request_attempts: dict[str, Any] = {}
        existing_manifest_without_attempt_ledger = False
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not _matched_manifest_core_is_compatible(
                existing_manifest, state=state, config=config, conditions=conditions,
                persona_modes=persona_modes, model_id=self.model_id,
            ):
                raise ValueError("现有虚拟作答使用旧协议或不同冻结配置，拒绝混合数据；请重新配置并使用新输出目录")
            if existing_manifest.get("simulation_signature") != signature:
                raise ValueError("现有作答的人物资料、分数或题目输入已改变；拒绝复用，请重新配置")
            raw_attempts = existing_manifest.get("request_attempts")
            existing_manifest_without_attempt_ledger = not isinstance(raw_attempts, Mapping)
            if isinstance(raw_attempts, Mapping):
                existing_request_attempts = deepcopy(dict(raw_attempts))
                for request_key, attempt_entry in existing_request_attempts.items():
                    if not isinstance(request_key, str) or not isinstance(attempt_entry, Mapping):
                        raise ValueError("现有 manifest 的请求尝试账本格式无效；拒绝恢复")
                    attempts_used = attempt_entry.get("attempts_used", 0)
                    history = attempt_entry.get("history", [])
                    if (
                        not isinstance(attempts_used, int)
                        or isinstance(attempts_used, bool)
                        or not 0 <= attempts_used <= self.max_retries + 1
                        or attempt_entry.get("max_attempts", self.max_retries + 1)
                        != self.max_retries + 1
                        or not isinstance(history, list)
                        or len(history) != attempts_used
                    ):
                        raise ValueError("现有 manifest 的请求尝试账本预算不一致；拒绝恢复")
                    if any(not isinstance(row, Mapping) for row in history):
                        raise ValueError("现有 manifest 的请求尝试历史格式无效；拒绝恢复")
            elif existing_manifest.get("status") in {"in_progress", "failed"}:
                raise ValueError(
                    "现有虚拟作答未记录持久化请求预算，无法证明恢复后总尝试不超过5次；拒绝重新施测"
                )
            existing_references = (
                existing_manifest.get("reference_questionnaires") or {}
            )
            existing_ipip = (
                existing_references.get("ipip_neo")
                if isinstance(existing_references, Mapping)
                else None
            )
            existing_facet_ids = (
                existing_ipip.get("facet_ids")
                if isinstance(existing_ipip, Mapping)
                else None
            )
            if not isinstance(existing_facet_ids, list) and isinstance(
                existing_ipip, Mapping
            ):
                # Manifests written by the first single-facet IPIP version did
                # not have facet_ids yet.
                legacy_facet_id = existing_ipip.get("target_dimension_id")
                existing_facet_ids = (
                    [str(legacy_facet_id)]
                    if isinstance(legacy_facet_id, str) and legacy_facet_id
                    else None
                )
            expected_facet_ids = [
                str(scale.get("facet_id"))
                for scale in (ipip_scales or [])
            ]
            existing_corpus_hash = (
                existing_ipip.get("corpus_hash")
                if isinstance(existing_ipip, Mapping)
                else None
            )
            expected_corpus_hash = (
                (ipip_scales[0] or {}).get("corpus_hash")
                if ipip_scales
                else None
            )
            if (
                isinstance(existing_ipip, Mapping)
                and ipip_scales is not None
                and (
                    existing_facet_ids != expected_facet_ids
                    or existing_corpus_hash != expected_corpus_hash
                )
            ):
                raise ValueError("现有IPIP-NEO作答使用了不同题库版本，拒绝混合数据")
        elif sjt_path.exists() or option_order_path.exists() or ipip_path.exists():
            raise ValueError("输出目录存在作答文件但缺少 manifest.json")

        reuse_summary = _seed_matched_response_files(
            state=state,
            output_dir=output_dir,
            config=config,
            conditions=conditions,
            model_id=self.model_id,
            respondent_refs=respondent_refs,
            sjt_items=sjt_items,
            sjt_path=sjt_path,
            target_retest_path=target_retest_path,
            ipip_neo_path=ipip_path,
            option_order_path=option_order_path,
        )
        generation_round = int(config.get("generation_round") or 1)
        frozen_reference_reuse: dict[str, Any] = {
            "enabled": False,
            "source_manifest_path": None,
            "expected_records": 0,
            "copied_records": 0,
            "reused_existing_records": 0,
        }
        if generation_round > 1 and ipip_scales is not None:
            frozen_reference_ref = (
                state.get("frozen_reference_questionnaire_ref")
                or state.get("previous_virtual_response_data_ref")
                or state.get("virtual_response_data_ref")
            )
            if not isinstance(frozen_reference_ref, str) or not frozen_reference_ref:
                raise ValueError(
                    "后续轮次必须沿用第一轮已完成的校标组卷结果；"
                    "缺少首轮冻结 manifest，拒绝重新施测"
                )
            frozen_reference_reuse = {
                "enabled": True,
                **_copy_frozen_reference_questionnaire(
                    source_manifest_ref=frozen_reference_ref,
                    destination_path=ipip_path,
                    state=state,
                    target_references=target_references,
                    respondent_refs=respondent_refs,
                    config=config,
                    scales=ipip_scales,
                ),
            }

        item_versions = {
            str(item.get("item_id")): _item_version_value(item)
            for item in sjt_items
            if isinstance(item, Mapping) and item.get("item_id")
        }
        _normalize_response_jsonl(
            sjt_path,
            key_fields=(
                "respondent_id",
                "persona_mode",
                "item_id",
                "item_version",
                "condition_id",
                "matched_subject_id",
            ),
            item_versions=item_versions,
        )
        _normalize_response_jsonl(
            target_retest_path,
            key_fields=(
                "respondent_id",
                "persona_mode",
                "item_id",
                "item_version",
                "condition_id",
                "matched_subject_id",
                "administration_id",
            ),
            item_versions=item_versions,
        )
        _normalize_response_jsonl(
            option_order_path,
            key_fields=(
                "respondent_id",
                "condition_id",
                "matched_subject_id",
                "item_id",
                "item_version",
            ),
            item_versions=item_versions,
        )
        sjt_keys = _canonicalize_item_response_keys(
            _load_jsonl_keys(
                sjt_path,
                ("respondent_id", "persona_mode", "item_id", "item_version", "condition_id", "matched_subject_id"),
            ),
            item_versions=item_versions,
        )
        initial_sjt_count = len(sjt_keys)
        expected_sjt_records = len(respondent_refs) * len(sjt_items)
        sjt_file_complete = (
            _jsonl_record_count(sjt_path) >= expected_sjt_records
        )
        scheduled_sjt_calls = sum(
            any(
                (str(reference.get("respondent_id")), PERSONA_MODE_SCORE_PROFILE, item.get("item_id"), _item_version_value(item), str(reference.get("condition_id")), str(reference.get("matched_subject_id"))) not in sjt_keys
                for item in sjt_items
            )
            for reference in respondent_refs
        )
        if sjt_file_complete:
            scheduled_sjt_calls = 0
        if scheduled_sjt_calls < 0:
            raise ValueError("现有SJT记录数超过当前配置预期")
        target_administration_count = int(
            config.get(
                "target_form_administration_count",
                DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT,
            )
        )
        if target_administration_count < 2:
            raise ValueError("整卷稳定性至少需要两次 target 施测")
        target_retest_keys = _canonicalize_item_response_keys(
            _load_jsonl_keys(
                target_retest_path,
                (
                    "respondent_id",
                    "persona_mode",
                    "item_id",
                    "item_version",
                    "condition_id",
                    "matched_subject_id",
                    "administration_id",
                ),
            ),
            item_versions=item_versions,
        )
        initial_target_retest_count = len(target_retest_keys)
        expected_target_retest_records = (
            len(target_references)
            * len(sjt_items)
            * (target_administration_count - 1)
        )
        target_retest_file_complete = (
            _jsonl_record_count(target_retest_path)
            >= expected_target_retest_records
        )
        scheduled_target_retest_calls = (
            sum(
                any(
                    (str(reference.get("respondent_id")), PERSONA_MODE_SCORE_PROFILE, item.get("item_id"), _item_version_value(item), "target", str(reference.get("matched_subject_id")), administration_id) not in target_retest_keys
                    for item in sjt_items
                )
                for administration_id in range(2, target_administration_count + 1)
                for reference in target_references
            )
        )
        if target_retest_file_complete:
            scheduled_target_retest_calls = 0
        if scheduled_target_retest_calls < 0:
            raise ValueError("现有 target 重测记录数超过当前配置预期")
        ipip_keys = _load_jsonl_keys(
            ipip_path,
            ("respondent_id", "facet_code", "item_id"),
        )
        initial_ipip_count = len(ipip_keys)
        expected_ipip_records = len(target_references) * sum(
            len(scale.get("items") or [])
            for scale in (ipip_scales or [])
            if isinstance(scale, Mapping)
        )
        if len(ipip_keys) > expected_ipip_records:
            raise ValueError("现有IPIP-NEO记录数超过当前配置预期")
        if existing_manifest_without_attempt_ledger and (
            scheduled_sjt_calls
            or scheduled_target_retest_calls
            or (
                expected_ipip_records > initial_ipip_count
                and not frozen_reference_reuse.get("enabled")
            )
        ):
            raise ValueError(
                "现有虚拟作答缺少持久化请求预算且仍有未完成请求；拒绝恢复以免重置尝试次数"
            )
        _write_json_atomic(score_profiles_path, {
            "schema_version": 2,
            "sampling_design": config.get("sampling_design"),
            "generator_version": config.get("generator_version"),
            "prompt_version": config.get("prompt_version"),
            "score_distribution": config.get("score_distribution"),
            "conditions": deepcopy(conditions),
            "profiles": list(respondent_refs),
            "demographics_snapshot": deepcopy(config["demographics_snapshot"]),
            "demographics_version": config["demographics_version"],
            "demographics_seed": config["demographics_seed"],
            "response_temperature": self.response_temperature,
            "score_specs": deepcopy(all_score_specs),
        })
        profile_rows = [{
            "respondent_id": reference["respondent_id"],
            "matched_subject_id": reference["matched_subject_id"],
            **demographics_to_columns(reference["demographics"], config["demographics_snapshot"]),
            "response_temperature": self.response_temperature,
            **{f"facet_score__{dimension_id}": value for dimension_id, value in reference["score_values"].items()},
        } for reference in respondent_refs]
        csv_buffer = StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
        _write_text_atomic(score_profiles_csv_path, "\ufeff" + csv_buffer.getvalue())
        _write_json_atomic(scoring_snapshot_path, {
            "schema_version": 4,
            "run_id": state["run_id"],
            "item_bank_id": state["item_bank_id"],
            "item_bank_version": state["item_bank_version"],
            "item_bank_fingerprint": state.get("item_bank_fingerprint"),
            "criterion_domain_id": criterion["domain_id"],
            "criterion_domain_name": criterion["domain_name_en"],
            "sampling_design": config.get("sampling_design"),
            "conditions": deepcopy(conditions),
            "score_distribution": deepcopy(config.get("score_distribution")),
            "items": [
                {
                    "item_id": item.get("item_id"),
                    "version": _item_version_value(item),
                    "target_dimension_id": item.get("target_dimension_id"),
                    "context_category": item.get("context_category"),
                    "option_ids": [option.get("option_id") for option in item.get("response_options") or [] if isinstance(option, Mapping)],
                    "scoring_key": dict(item.get("scoring_key") or {}),
                }
                for item in state["frozen_item_bank"]
            ],
        })
        iteration_controller = state.get("facet_iteration_state")
        facet_iteration_manifest: dict[str, Any] | None = None
        if (
            isinstance(iteration_controller, Mapping)
            and iteration_controller.get("policy_version")
            == "fixed_cohort_facet_iteration_v2"
        ):
            next_measurement_batch = int(
                state.get("psychometric_analysis_round") or 0
            ) + 1
            facet_iteration_manifest = deepcopy(dict(iteration_controller))
            facet_iteration_manifest.update(
                completed_rounds_before_measurement=int(
                    iteration_controller.get("completed_rounds") or 0
                ),
                last_completed_measurement_batch=iteration_controller.get(
                    "measurement_batch"
                ),
                measurement_batch=next_measurement_batch,
                development_round_batch=int(
                    iteration_controller.get("development_round_batch") or 0
                ) + 1,
            )
        manifest = {
            "schema_version": MATCHED_CONDITION_SCHEMA_VERSION,
            "status": "in_progress",
            "simulation_signature": signature,
            "run_id": state["run_id"],
            "item_bank_id": state["item_bank_id"],
            "item_bank_version": state["item_bank_version"],
            "item_bank_fingerprint": state.get("item_bank_fingerprint"),
            "pool_id": config.get("pool_id"),
            "source_sha256": config.get("source_sha256"),
            "criterion_domain_id": criterion["domain_id"],
            "criterion_domain_name": criterion["domain_name_en"],
            "sample_size": len(respondent_refs),
            "sample_size_per_condition": config.get("sample_size_per_condition"),
            "condition_count": 3,
            "group_count": len(condition_ids),
            "condition_ids": list(condition_ids),
            "arm_ids": list(MATCHED_CONDITION_IDS),
            "sampling_design": "matched_facet_conditions",
            "persona_modes": persona_modes,
            "persona_mode_count": 1,
            "conditions": deepcopy(conditions),
            "sjt_item_count": len(sjt_items),
            "sjt_response_mode": "complete_profile_batch_shuffled_items",
            "response_batch_design": "one_batch_per_matched_subject",
            "item_order_policy": "fully_shuffled_per_respondent_batch",
            "score_noise_method": "fixed_scores_temperature_sampling",
            "response_temperature": self.response_temperature,
            "demographics_version": config["demographics_version"],
            "target_form_administration_count": target_administration_count,
            "reference_questionnaire_freeze_policy": (
                "freeze_after_first_measurement"
            ),
            "target_form_retest": {
                "path": str(target_retest_path.resolve()),
                "administration_ids": list(
                    range(2, target_administration_count + 1)
                ),
                "response_mode": "target_form_retest_batch_shuffled_items",
            },
            "reference_questionnaires": {
                "ipip_neo": {
                    "enabled": ipip_scales is not None,
                    "path": str(ipip_path.resolve()),
                    "facet_ids": [
                        str(scale.get("facet_id"))
                        for scale in (ipip_scales or [])
                    ],
                    "facet_codes": [
                        str(scale.get("facet_code"))
                        for scale in (ipip_scales or [])
                    ],
                    "facets": [
                        {
                            "facet_id": scale.get("facet_id"),
                            "facet_code": scale.get("facet_code"),
                            "item_count": len(scale.get("items") or []),
                            "source_reported_alpha": scale.get("alpha"),
                        }
                        for scale in (ipip_scales or [])
                    ],
                    "item_count": sum(
                        len(scale.get("items") or [])
                        for scale in (ipip_scales or [])
                    ),
                    "items_per_facet": 10,
                    "response_mode": "selected_facets_single_batch_per_respondent",
                    "scope": "authoring_selected_facets",
                    "freeze_policy": "freeze_after_first_measurement",
                    "frozen_from_generation_round": 1,
                    "frozen_reference_manifest_path": (
                        frozen_reference_reuse.get("source_manifest_path")
                        or str(manifest_path.resolve())
                        if ipip_scales is not None
                        else None
                    ),
                    "reused_from_frozen_reference": bool(
                        frozen_reference_reuse.get("enabled")
                    ),
                    "score_scale": "1-5",
                    "prompt_version": IPIP_NEO_REFERENCE_PROMPT_VERSION,
                    "corpus_hash": (
                        ipip_scales[0].get("corpus_hash")
                        if ipip_scales
                        else None
                    ),
                    "source_file": (
                        ipip_scales[0].get("source_file")
                        if ipip_scales
                        else None
                    ),
                    "source_reported_alpha": (
                        target_ipip_scale.get("alpha")
                        if isinstance(target_ipip_scale, Mapping)
                        else None
                    ),
                    # Deprecated one-facet aliases remain for readers of old
                    # reports; new reports must use facets/facet_ids.
                    "target_dimension_id": (
                        target_dimension_id or None
                    ),
                    "target_facet_code": (
                        target_ipip_scale.get("facet_code")
                        if isinstance(target_ipip_scale, Mapping)
                        else None
                    ),
                },
            },
            "model_id": self.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
            "generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
            "virtual_sample_config": deepcopy(dict(config)),
            "request_attempts": existing_request_attempts,
            "request_attempts_updated_at": utc_timestamp(),
            "expected_sjt_records": expected_sjt_records,
            "expected_target_form_retest_records": (
                expected_target_retest_records
            ),
            "expected_ipip_neo_records": expected_ipip_records,
            "frozen_reference_questionnaire_ref": (
                frozen_reference_reuse.get("source_manifest_path")
                or (
                    str(manifest_path.resolve())
                    if ipip_scales is not None
                    else None
                )
            ),
            "frozen_reference_reuse": deepcopy(frozen_reference_reuse),
            "source_manifest_path": reuse_summary.get("source_manifest_path"),
            "source_response_reuse": deepcopy(reuse_summary),
            **(
                {
                    "iteration_policy_version": iteration_controller.get(
                        "policy_version"
                    ),
                    "development_round": iteration_controller.get(
                        "development_round"
                    ),
                    "completed_rounds": iteration_controller.get(
                        "completed_rounds"
                    ),
                    "measurement_batch": facet_iteration_manifest[
                        "measurement_batch"
                    ],
                    "facet_iteration_state": deepcopy(
                        facet_iteration_manifest
                    ),
                }
                if facet_iteration_manifest is not None
                and isinstance(iteration_controller, Mapping)
                else {}
            ),
            "signature_migrated_from": signature_migrated_from,
            "concurrency_policy": "all_at_once",
            "max_concurrency": 0,
            "score_profiles_path": str(score_profiles_path.resolve()),
            "score_profiles_csv_path": str(score_profiles_csv_path.resolve()),
            "scoring_snapshot_path": str(scoring_snapshot_path.resolve()),
            "option_order_path": str(option_order_path.resolve()),
            "started_at": utc_timestamp(),
            "interpretation_limitations": [
                "These are exploratory virtual screening results, not formal human psychometric evidence.",
                "Each matched virtual respondent carries a complete multi-facet profile; each facet is generated independently and all three arms share one mean/SD setting.",
                "The target-form retest measures model/prompt response stability under a second balanced option order; it is not human test-retest reliability.",
                "The authoring-selected IPIP-NEO facet scales are answered in one batch by the same target respondents; this is development-stage convergent/discriminant evidence, not human criterion data.",
                "The criterion questionnaire is administered only in the first virtual measurement round; later rounds reuse the frozen first-round records without new IPIP model calls.",
            ],
        }
        _write_json_atomic(manifest_path, manifest)

        def persist_attempt_ledger() -> None:
            manifest["request_attempts"] = existing_request_attempts
            manifest["request_attempts_updated_at"] = utc_timestamp()
            _write_json_atomic(manifest_path, manifest)

        full_sjt_item_versions = [
            (str(item.get("item_id")), _item_version_value(item))
            for item in sjt_items
        ]
        persona_prompts = {
            respondent_id: build_persona_prompt(
                profile,
                persona_mode=persona_mode,
                score_specs=all_score_specs,
                demographics_snapshot=config["demographics_snapshot"],
            )
            for respondent_id, profile in profile_by_id.items()
        }
        target_index_by_matched = {
            str(reference.get("matched_subject_id")): index
            for index, reference in enumerate(target_references)
        }
        index_by_matched = {
            str(reference.get("matched_subject_id")): index
            for index, reference in enumerate(respondent_refs)
            if reference.get("condition_id") == "target"
        }
        progress_lock = asyncio.Lock()
        completed_calls = 0

        async def report_completed() -> None:
            nonlocal completed_calls
            async with progress_lock:
                completed_calls += 1
                interval = max(1, scheduled_sjt_calls // 20)
                if completed_calls in {1, scheduled_sjt_calls} or completed_calls % interval == 0:
                    emit_progress({"type": "simulation_progress", "stage": "SJT matched facet response", "completed": completed_calls, "total": scheduled_sjt_calls})

        def validate_sjt(result: Mapping[str, Any], *, allowed_display_ids: set[str]) -> str:
            selected = result.get("selected_option_id")
            if selected not in allowed_display_ids:
                raise ValueError(f"模型选择了无效展示选项 {selected!r}")
            return str(selected)

        async def run_sjt_job(respondent_ref: Mapping[str, Any]) -> None:
            if sjt_file_complete:
                return
            condition_id = str(respondent_ref["condition_id"])
            matched_subject_id = str(respondent_ref["matched_subject_id"])
            respondent_id = str(respondent_ref["respondent_id"])
            missing_items = [
                item for item in sjt_items
                if (respondent_id, persona_mode, item.get("item_id"), _item_version_value(item), condition_id, matched_subject_id) not in sjt_keys
            ]
            if not missing_items:
                return
            item_rows: list[tuple[Mapping[str, Any], str, Sequence[str]]] = []
            option_maps: dict[str, tuple[list[str], dict[str, str]]] = {}
            for item in missing_items:
                option_ids = [str(option["option_id"]) for option in item["response_options"]]
                shuffled_item_seed = f"{seed}:{condition_id}:{matched_subject_id}:batch"
                ordered_ids = balanced_option_order(
                    option_ids,
                    respondent_index=index_by_matched[matched_subject_id],
                    seed=int.from_bytes(sha256(shuffled_item_seed.encode("utf-8")).digest()[:4], "big"),
                    item_id=str(item.get("item_id")),
                )
                display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
                option_maps[str(item.get("item_id"))] = (ordered_ids, dict(zip(display_ids, ordered_ids)))
                item_rows.append((item, persona_prompts[respondent_id], ordered_ids))
            rng = random.Random(f"{seed}:{respondent_id}:{condition_id}:item-order")
            rng.shuffle(item_rows)
            request_key = _simulation_attempt_key(
                "sjt_main",
                respondent_ref,
                full_sjt_item_versions,
                persona_mode=persona_mode,
            )
            selected_displays = await _invoke_with_retry(
                self.sjt_batch_model,
                build_sjt_batch_messages(item_rows),
                semaphore=self.semaphore,
                validator=lambda result: _validate_sjt_batch_response(
                    result,
                    item_rows,
                    label="SJT批次",
                ),
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=f"SJT batch {condition_id}/{matched_subject_id}",
                attempt_ledger=existing_request_attempts,
                attempt_key=request_key,
                persist_attempt_ledger=persist_attempt_ledger,
            )
            records: list[dict[str, Any]] = []
            for item in missing_items:
                item_id = str(item.get("item_id"))
                ordered_ids, display_to_original = option_maps[item_id]
                display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
                target_dimension_id = str(item.get("target_dimension_id") or "")
                score_dimension_id = (
                    target_dimension_id
                    if condition_id == "target"
                    else str(
                        condition_rows.get(condition_id, {}).get(
                            "dimension_id"
                        )
                        or respondent_ref.get("active_dimension_id")
                        or target_dimension_id
                    )
                )
                score_values = dict(respondent_ref.get("score_values") or {})
                active_score = score_values.get(score_dimension_id)
                if active_score is None:
                    active_score = score_values.get(str(respondent_ref.get("active_dimension_id")))
                key = (respondent_id, persona_mode, item_id, _item_version_value(item), condition_id, matched_subject_id)
                record = {
                    "record_type": "sjt_response",
                    "run_id": state["run_id"],
                    "respondent_id": respondent_id,
                    "condition_id": condition_id,
                    "arm_id": respondent_ref.get("arm_id") or condition_rows.get(condition_id, {}).get("arm_id"),
                    "group_id": respondent_ref.get("group_id") or condition_rows.get(condition_id, {}).get("group_id"),
                    "matched_subject_id": matched_subject_id,
                    "active_dimension_id": score_dimension_id,
                    "active_score": float(active_score) if active_score is not None else None,
                    "score_values": score_values,
                    "persona_mode": persona_mode,
                    "item_bank_id": state["item_bank_id"],
                    "item_bank_version": state["item_bank_version"],
                    "item_bank_fingerprint": state.get("item_bank_fingerprint"),
                    "item_id": item_id,
                    "item_version": _item_version_value(item),
                    "display_option_order": [{"display_option_id": display_id, "option_id": original_id} for display_id, original_id in zip(display_ids, ordered_ids)],
                    "raw_display_option_id": selected_displays[item_id],
                    "selected_option_id": display_to_original[selected_displays[item_id]],
                    "response_mode": "complete_profile_batch_shuffled_items",
                    "item_order_shuffled": True,
                    "demographics": deepcopy(respondent_ref["demographics"]),
                    **demographics_to_columns(respondent_ref["demographics"], config["demographics_snapshot"]),
                    "response_temperature": self.response_temperature,
                    "model_id": self.model_id,
                    "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
                }
                records.append(record)
                sjt_keys.add(key)
            await self._append_records(sjt_path, records)
            await self._append_records(option_order_path, [{key_name: record.get(key_name) for key_name in ("run_id", "respondent_id", "condition_id", "matched_subject_id", "item_id", "item_version", "display_option_order", "raw_display_option_id", "selected_option_id")} for record in records])
            await report_completed()

        async def run_target_retest_job(
            respondent_ref: Mapping[str, Any],
            administration_id: int,
        ) -> None:
            if target_retest_file_complete:
                return
            matched_subject_id = str(respondent_ref["matched_subject_id"])
            respondent_id = str(respondent_ref["respondent_id"])
            missing_items = [
                item for item in sjt_items
                if (respondent_id, persona_mode, item.get("item_id"), _item_version_value(item), "target", matched_subject_id, administration_id) not in target_retest_keys
            ]
            if not missing_items:
                return
            item_rows: list[tuple[Mapping[str, Any], str, Sequence[str]]] = []
            option_maps: dict[str, tuple[list[str], dict[str, str]]] = {}
            for item in missing_items:
                option_ids = [str(option["option_id"]) for option in item["response_options"]]
                ordered_ids = balanced_option_order(
                    option_ids,
                    respondent_index=index_by_matched[matched_subject_id],
                    seed=seed + 100_003 * administration_id,
                    item_id=str(item.get("item_id")),
                )
                display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
                option_maps[str(item.get("item_id"))] = (ordered_ids, dict(zip(display_ids, ordered_ids)))
                item_rows.append((
                    item,
                    persona_prompts[respondent_id],
                    ordered_ids,
                ))
            random.Random(f"{seed}:{respondent_id}:target-retest:{administration_id}").shuffle(item_rows)
            expected_item_ids = {str(item.get("item_id")) for item in missing_items}
            score_values = dict(respondent_ref.get("score_values") or {})

            async def invoke_retest_batch(
                rows: Sequence[tuple[Mapping[str, Any], str, Sequence[str]]],
                *,
                job_label: str,
            ) -> dict[str, str]:
                request_key = _simulation_attempt_key(
                    "sjt_target_retest",
                    respondent_ref,
                    full_sjt_item_versions,
                    persona_mode=persona_mode,
                    administration_id=administration_id,
                )
                return await _invoke_with_retry(
                    self.sjt_batch_model,
                    build_sjt_batch_messages(rows),
                    semaphore=self.semaphore,
                    validator=lambda result: _validate_sjt_batch_response(
                        result,
                        rows,
                        label="整卷重测",
                    ),
                    max_retries=self.max_retries,
                    retry_delay_seconds=self.retry_delay_seconds,
                    request_timeout_seconds=self.request_timeout_seconds,
                    job_label=job_label,
                    attempt_ledger=existing_request_attempts,
                    attempt_key=request_key,
                    persist_attempt_ledger=persist_attempt_ledger,
                )

            async def persist_retest_items(
                items: Sequence[Mapping[str, Any]],
            ) -> None:
                records: list[dict[str, Any]] = []
                for item in items:
                    item_id = str(item.get("item_id"))
                    if item_id not in selected_displays:
                        continue
                    ordered_ids, display_to_original = option_maps[item_id]
                    display_ids = [
                        chr(ord("A") + index)
                        for index in range(len(ordered_ids))
                    ]
                    target_dimension_id = str(
                        item.get("target_dimension_id")
                        or respondent_ref.get("active_dimension_id")
                        or ""
                    )
                    active_score = score_values.get(target_dimension_id)
                    key = (
                        respondent_id,
                        persona_mode,
                        item_id,
                        _item_version_value(item),
                        "target",
                        matched_subject_id,
                        administration_id,
                    )
                    if key in target_retest_keys:
                        continue
                    records.append(
                        {
                            "record_type": "sjt_target_form_retest_response",
                            "run_id": state["run_id"],
                            "respondent_id": respondent_id,
                            "condition_id": "target",
                            "arm_id": "target",
                            "group_id": respondent_ref.get("group_id")
                            or condition_rows["target"].get("group_id"),
                            "matched_subject_id": matched_subject_id,
                            "active_dimension_id": target_dimension_id,
                            "active_score": float(active_score)
                            if active_score is not None
                            else None,
                            "score_values": score_values,
                            "persona_mode": persona_mode,
                            "administration_id": administration_id,
                            "item_bank_id": state["item_bank_id"],
                            "item_bank_version": state["item_bank_version"],
                            "item_bank_fingerprint": state.get(
                                "item_bank_fingerprint"
                            ),
                            "item_id": item_id,
                            "item_version": _item_version_value(item),
                            "display_option_order": [
                                {
                                    "display_option_id": display_id,
                                    "option_id": original_id,
                                }
                                for display_id, original_id in zip(
                                    display_ids, ordered_ids
                                )
                            ],
                            "raw_display_option_id": selected_displays[item_id],
                            "selected_option_id": display_to_original[
                                selected_displays[item_id]
                            ],
                            "response_mode": "target_form_retest_batch_shuffled_items",
                            "item_order_shuffled": True,
                            "demographics": deepcopy(respondent_ref["demographics"]),
                            **demographics_to_columns(respondent_ref["demographics"], config["demographics_snapshot"]),
                            "response_temperature": self.response_temperature,
                            "model_id": self.model_id,
                            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
                        }
                    )
                    target_retest_keys.add(key)
                if records:
                    await self._append_records(target_retest_path, records)

            try:
                selected_displays = await invoke_retest_batch(
                    item_rows,
                    job_label=(
                        f"SJT target form batch retest "
                        f"{administration_id}/{matched_subject_id}"
                    ),
                )
            except ResponseTemperatureRejected:
                raise
            except RuntimeError as exc:
                cause = exc.__cause__
                if isinstance(cause, SJTBatchValidationError):
                    selected_displays = dict(cause.valid_responses)
                    failed_item_ids = sorted(
                        {
                            item_id
                            for item_id in cause.failed_item_ids
                            if item_id in expected_item_ids
                        }
                    )
                else:
                    # A transport/schema failure gives us no trustworthy
                    # partial response; every unanswered item is failed.
                    selected_displays = {}
                    failed_item_ids = list(expected_item_ids)
                failed_item_ids = list(
                    dict.fromkeys(
                        item_id
                        for item_id in failed_item_ids
                        if item_id not in selected_displays
                    )
                )
                if not failed_item_ids:
                    failed_item_ids = sorted(expected_item_ids - set(selected_displays))
                rows_by_item_id = {
                    str(row[0].get("item_id")): row
                    for row in item_rows
                }
                await persist_retest_items(
                    [
                        item
                        for item in missing_items
                        if str(item.get("item_id")) in selected_displays
                    ]
                )
                retry_rows = [
                    (item_id, rows_by_item_id[item_id])
                    for item_id in failed_item_ids
                    if item_id in rows_by_item_id
                ]
                retry_results = await asyncio.gather(
                    *(
                        invoke_retest_batch(
                            [row],
                            job_label=(
                                f"SJT target form item retest "
                                f"{administration_id}/{matched_subject_id}/{item_id}"
                            ),
                        )
                        for item_id, row in retry_rows
                    ),
                    return_exceptions=True,
                )
                retry_errors: list[BaseException] = []
                for (_, row), result in zip(retry_rows, retry_results):
                    if isinstance(result, BaseException):
                        retry_errors.append(result)
                        continue
                    selected_displays.update(result)
                    await persist_retest_items([row[0]])
                if set(selected_displays) != expected_item_ids:
                    unresolved = sorted(expected_item_ids - set(selected_displays))
                    raise RuntimeError(
                        "整卷重测的失败题目单题重测后仍缺失："
                        + ", ".join(unresolved)
                        + (
                            "；单题重测错误："
                            + "；".join(str(error) for error in retry_errors)
                            if retry_errors
                            else ""
                        )
                    ) from (retry_errors[0] if retry_errors else exc)
            await persist_retest_items(missing_items)

        reference_errors: list[Exception] = []
        scheduled_ipip_calls = 0
        ipip_stage_enabled = (
            ipip_scales is not None and not frozen_reference_reuse["enabled"]
        )

        def validate_ipip_reference(
            result: Mapping[str, Any],
            *,
            expected_count: int,
        ) -> list[int]:
            ratings = result.get("ratings")
            if not isinstance(ratings, list) or len(ratings) != expected_count:
                raise ValueError(
                    f"IPIP-NEO批次必须返回{expected_count}个评分"
                )
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > 5
                for value in ratings
            ):
                raise ValueError("IPIP-NEO评分必须是1到5之间的整数")
            return ratings

        async def run_ipip_reference_job(
            respondent_ref: Mapping[str, Any],
        ) -> None:
            nonlocal scheduled_ipip_calls
            respondent_id = str(respondent_ref["respondent_id"])
            scale_items = [
                (scale, item)
                for scale in (ipip_scales or [])
                for item in list(scale.get("items") or [])
            ]
            keys = {
                (
                    respondent_id,
                    str(scale["facet_code"]),
                    str(item["item_id"]),
                )
                for scale, item in scale_items
            }
            present = keys.intersection(ipip_keys)
            if present:
                if len(present) != len(keys):
                    raise ValueError(
                        f"IPIP-NEO {respondent_id} 存在不完整批次，拒绝混合恢复"
                    )
                return
            scheduled_ipip_calls += 1
            ipip_item_versions = [
                (
                    f"{scale.get('facet_code')}/{item.get('item_id')}",
                    item.get("version") or scale.get("corpus_hash"),
                )
                for scale, item in scale_items
            ]
            request_key = _simulation_attempt_key(
                "ipip_neo_reference",
                respondent_ref,
                ipip_item_versions,
                persona_mode=persona_mode,
            )
            ratings = await _invoke_with_retry(
                self.ipip_neo_model,
                build_ipip_neo_messages(
                    persona_prompts[respondent_id],
                    ipip_scales or [],
                ),
                semaphore=self.semaphore,
                validator=lambda result: validate_ipip_reference(
                    result,
                    expected_count=len(scale_items),
                ),
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=f"IPIP-NEO selected facets reference {respondent_id}",
                attempt_ledger=existing_request_attempts,
                attempt_key=request_key,
                persist_attempt_ledger=persist_attempt_ledger,
            )
            records = []
            for (scale, item), rating in zip(scale_items, ratings):
                facet_code = str(scale["facet_code"])
                facet_id = str(scale["facet_id"])
                polarity = str(item.get("polarity") or "")
                scored = score_ipip_neo_rating(int(rating), polarity)
                records.append(
                    {
                        "record_type": "ipip_neo_response",
                        "run_id": state["run_id"],
                        "respondent_id": respondent_id,
                        "matched_subject_id": str(respondent_ref["matched_subject_id"]),
                        "condition_id": "target",
                        "facet_code": facet_code,
                        "facet_id": facet_id,
                        "item_id": item["item_id"],
                        "raw_response": int(rating),
                        "polarity": polarity,
                        "score": scored,
                        "response_mode": "selected_reference_facets_single_batch",
                        "score_values": dict(respondent_ref["score_values"]),
                        "demographics": deepcopy(respondent_ref["demographics"]),
                        **demographics_to_columns(respondent_ref["demographics"], config["demographics_snapshot"]),
                        "response_temperature": self.response_temperature,
                        "model_id": self.model_id,
                        "prompt_version": IPIP_NEO_REFERENCE_PROMPT_VERSION,
                    }
                )
            await self._append_records(ipip_path, records)
            ipip_keys.update(keys)

        async def drain_stage(
            stage: str,
            jobs: Sequence[Any],
            *,
            total: int,
            complete: Any,
        ) -> list[Exception]:
            results = await asyncio.gather(*jobs, return_exceptions=True)
            errors = [result for result in results if isinstance(result, Exception)]
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": stage,
                    "status": "completed" if not errors and complete() else "failed",
                    "total": total,
                }
            )
            return errors

        sjt_jobs = [run_sjt_job(reference) for reference in respondent_refs]
        target_retest_jobs = [
            run_target_retest_job(reference, administration_id)
            for administration_id in range(2, target_administration_count + 1)
            for reference in target_references
        ]
        ipip_job_count = len(target_references) if ipip_stage_enabled else 0
        ipip_jobs = (
            [run_ipip_reference_job(reference) for reference in target_references]
            if ipip_stage_enabled
            else []
        )

        emit_progress({"type": "simulation_stage", "stage": "SJT matched facet response", "status": "started", "total": scheduled_sjt_calls})
        emit_progress({"type": "simulation_stage", "stage": "SJT target form retest", "status": "started", "total": scheduled_target_retest_calls})
        if ipip_scales is not None:
            emit_progress({
                "type": "simulation_stage",
                "stage": "IPIP-NEO selected facet response",
                "status": (
                    "started" if ipip_stage_enabled else "reused_frozen"
                ),
                "total": len(target_references),
            })

        stage_results = await asyncio.gather(
            drain_stage(
                "SJT matched facet response",
                sjt_jobs,
                total=scheduled_sjt_calls,
                complete=lambda: len(sjt_keys) == expected_sjt_records,
            ),
            drain_stage(
                "SJT target form retest",
                target_retest_jobs,
                total=scheduled_target_retest_calls,
                complete=lambda: len(target_retest_keys)
                == expected_target_retest_records,
            ),
            *(
                [
                    drain_stage(
                        "IPIP-NEO selected facet response",
                        ipip_jobs,
                        total=ipip_job_count,
                        complete=lambda: len(ipip_keys) == expected_ipip_records,
                    )
                ]
                if ipip_stage_enabled
                else []
            ),
            return_exceptions=True,
        )

        def collect_stage_errors(result: Any) -> list[Exception]:
            if isinstance(result, Exception):
                return [result]
            if isinstance(result, list):
                return [error for error in result if isinstance(error, Exception)]
            return []

        sjt_errors = collect_stage_errors(stage_results[0])
        target_retest_errors = collect_stage_errors(stage_results[1])
        reference_errors = (
            collect_stage_errors(stage_results[2])
            if ipip_stage_enabled
            else []
        )

        errors = [*sjt_errors, *target_retest_errors, *reference_errors]
        completed = (
            not errors
            and len(sjt_keys) == expected_sjt_records
            and len(target_retest_keys) == expected_target_retest_records
            and len(ipip_keys) == expected_ipip_records
        )
        manifest.update({
            "status": "completed" if completed else "failed",
            "completed_sjt_records": len(sjt_keys),
            "completed_target_form_retest_records": len(target_retest_keys),
            "completed_ipip_neo_records": len(ipip_keys),
            "frozen_reference_reuse": deepcopy(frozen_reference_reuse),
            "resumed_sjt_records": initial_sjt_count,
            "resumed_target_form_retest_records": (
                initial_target_retest_count
            ),
            "finished_at": utc_timestamp(),
            "errors": [str(error) for error in errors[:20]],
        })
        _write_json_atomic(manifest_path, manifest)
        emit_progress({"type": "simulation_stage", "stage": "matched facet virtual response", "status": "completed" if completed else "failed", "total": expected_sjt_records + expected_target_retest_records + expected_ipip_records})
        if not completed:
            if errors:
                raise RuntimeError(f"虚拟作答有 {len(errors)} 个任务失败；首个错误：{errors[0]}")
            raise RuntimeError("虚拟作答记录数与预期不一致")
        return {
            "manifest_path": str(manifest_path.resolve()),
            "scoring_snapshot_path": str(scoring_snapshot_path.resolve()),
            "score_profiles_path": str(score_profiles_path.resolve()),
            "option_order_path": str(option_order_path.resolve()),
            "persona_summary_path": None,
            "score_profiles_csv_path": str(score_profiles_csv_path.resolve()),
            "sjt_path": str(sjt_path.resolve()),
            "target_form_retest_path": str(target_retest_path.resolve()),
            "respondent_count": len(respondent_refs),
            "persona_modes": [PERSONA_MODE_SCORE_PROFILE],
            "persona_mode_count": 1,
            "condition_count": 3,
            "group_count": len(condition_ids),
            "condition_ids": list(condition_ids),
            "arm_ids": list(MATCHED_CONDITION_IDS),
            "sampling_design": "matched_facet_conditions",
            "response_batch_design": "one_batch_per_matched_subject",
            "persona_summary_count": 0,
            "sjt_item_count": len(sjt_items),
            "sjt_response_count": len(sjt_keys),
            "target_form_retest_response_count": len(target_retest_keys),
            "ipip_neo_path": str(ipip_path.resolve()),
            "ipip_neo_response_count": len(ipip_keys),
            "ipip_neo_facet_count": len(ipip_scales or []),
            "ipip_neo_facet_ids": [
                str(scale.get("facet_id"))
                for scale in (ipip_scales or [])
                if isinstance(scale, Mapping)
            ],
            "ipip_neo_items_per_respondent": sum(
                len(scale.get("items") or [])
                for scale in (ipip_scales or [])
                if isinstance(scale, Mapping)
            ),
            "scheduled_persona_summary_api_calls": 0,
            "scheduled_sjt_api_calls": scheduled_sjt_calls,
            "scheduled_target_form_retest_api_calls": (
                scheduled_target_retest_calls
            ),
            "scheduled_ipip_neo_api_calls": scheduled_ipip_calls,
            "concurrency_policy": "all_at_once",
            "max_concurrency": self.max_concurrency,
            "max_retries": self.max_retries,
            "request_timeout_seconds": self.request_timeout_seconds,
            "resumed_sjt_records": initial_sjt_count,
            "resumed_target_form_retest_records": (
                initial_target_retest_count
            ),
            "reused_persona_summary_records": 0,
            "reused_sjt_records": initial_sjt_count,
            "reused_target_retest_records": initial_target_retest_count,
            "reused_ipip_neo_records": initial_ipip_count,
            "frozen_reference_reused": bool(frozen_reference_reuse.get("enabled")),
            "frozen_reference_reused_records": int(
                frozen_reference_reuse.get("expected_records") or 0
            ),
            "frozen_reference_questionnaire_ref": (
                frozen_reference_reuse.get("source_manifest_path")
                or (
                    str(manifest_path.resolve())
                    if ipip_scales is not None
                    else None
                )
            ),
            "reused_local_item_count": reuse_summary.get("reused_local_item_count", 0),
            "source_response_reuse": deepcopy(reuse_summary),
            "source_manifest_path": reuse_summary.get("source_manifest_path"),
            "model_id": self.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
            "response_temperature": self.response_temperature,
            "demographics_version": config["demographics_version"],
        }

    async def run_single_item_retest(
        self,
        *,
        state: PSJTState,
        item: Mapping[str, Any],
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> dict[str, Any]:
        """施测一个候选题，复用同一批匹配 facet 被试和随机化条件。

        这是题目返修 Agent 的局部反馈工具，不会修改正式作答目录，也不会
        生成IPIP-NEO参照问卷或整卷重测数据。正式测量仍由 ``run`` 统一完成。
        候选作答按候选题内容缓存，进程中断后不会因为同一个候选重复调用模型。
        """

        context = build_virtual_response_context(state)
        config = context.get("virtual_sample_config")
        respondent_refs = context.get("virtual_respondents")
        if not isinstance(config, Mapping) or config.get(
            "schema_version"
        ) != MATCHED_CONDITION_SCHEMA_VERSION:
            raise ValueError("单题局部复测需要固定三臂 matched-condition 配置")
        if not isinstance(respondent_refs, list) or not respondent_refs:
            raise ValueError("单题局部复测缺少虚拟被试")
        if config.get("sample_size") != len(respondent_refs):
            raise ValueError("单题局部复测的样本量与被试引用数量不一致")
        if not isinstance(item.get("item_id"), str) or not item.get("item_id"):
            raise ValueError("单题局部复测的候选题缺少 item_id")
        item_version = _item_version_value(item)
        if not isinstance(item_version, int) or isinstance(item_version, bool):
            raise ValueError("单题局部复测的候选题缺少有效版本")

        conditions = config.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError("单题局部复测缺少 matched facet 条件")
        top_level_condition_ids = {
            str(row.get("condition_id"))
            for row in conditions
            if isinstance(row, Mapping)
        }
        if top_level_condition_ids != set(MATCHED_CONDITION_IDS):
            raise ValueError(
                "单题局部复测配置的顶层条件必须恰好包含 "
                "target、same_domain、cross_domain"
            )
        condition_groups = flatten_matched_condition_groups(conditions)
        condition_ids = tuple(str(row.get("condition_id")) for row in condition_groups)
        arm_ids = {
            str(row.get("arm_id"))
            for row in condition_groups
            if row.get("arm_id")
        }
        target_groups = [
            row
            for row in condition_groups
            if str(row.get("arm_id")) == "target"
        ]
        if (
            not condition_groups
            or
            set(arm_ids) != set(MATCHED_CONDITION_IDS)
            or len(set(condition_ids)) != len(condition_ids)
            or len(target_groups) != 1
            or str(target_groups[0].get("condition_id")) != "target"
        ):
            raise ValueError(
                "单题局部复测必须使用固定三臂，并且 target 只能有一个 facet group"
            )

        persona_mode = PERSONA_MODE_SCORE_PROFILE
        condition_rows = {
            str(row["condition_id"]): dict(row) for row in condition_groups
        }
        if not matched_condition_sample_is_current(config, respondent_refs):
            raise ValueError("旧协议或不完整的虚拟被试资料不能用于局部复测；请重新配置")
        if config["response_temperature"] != self.response_temperature:
            raise ValueError("局部复测模型温度与冻结配置不一致")
        profiles = resolve_virtual_respondent_profiles(
            respondent_refs, demographics_snapshot=config["demographics_snapshot"],
        )
        all_local_specs = deepcopy(config["score_specs"])
        profile_by_id = {
            str(profile["respondent_id"]): profile for profile in profiles
        }
        persona_prompts = {
            respondent_id: build_persona_prompt(
                profile,
                persona_mode=persona_mode,
                score_specs=all_local_specs,
                demographics_snapshot=config["demographics_snapshot"],
            )
            for respondent_id, profile in profile_by_id.items()
        }
        target_references = [
            reference
            for reference in respondent_refs
            if str(reference.get("condition_id")) == "target"
        ]
        index_by_matched = {
            str(reference.get("matched_subject_id")): index
            for index, reference in enumerate(target_references)
        }
        if not index_by_matched:
            raise ValueError("单题局部复测缺少 target 条件被试")
        if any(
            str(reference.get("matched_subject_id")) not in index_by_matched
            for reference in respondent_refs
        ):
            raise ValueError("所有局部复测条件必须共享 target 的 matched_subject_id")

        cache_dir = (
            Path(output_root)
            / str(state.get("run_id") or "unknown")
            / "item_local_retests"
        )
        cache_configs = [_normalize_concurrency_config(config)]
        for legacy_limit in range(1, 21):
            legacy_config = dict(config)
            legacy_config.pop("concurrency_policy", None)
            legacy_config["max_concurrency"] = legacy_limit
            cache_configs.append(legacy_config)
        cache_candidates: list[tuple[str, Path]] = []
        for cache_config in cache_configs:
            cache_payload = {
                "run_id": state.get("run_id"),
                "item_bank_id": state.get("item_bank_id"),
                "item_bank_version": state.get("item_bank_version"),
                "item_bank_fingerprint": state.get("item_bank_fingerprint"),
                "model_id": self.model_id,
                "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
                "config": cache_config,
                "respondents": list(respondent_refs),
                "item": dict(item),
            }
            candidate_key = sha256(
                json.dumps(
                    cache_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cache_candidates.append(
                (
                    candidate_key,
                    cache_dir / (
                        f"{item['item_id']}-v{item_version}-"
                        f"{candidate_key[:16]}.json"
                    ),
                )
            )
        cache_key, cache_path = cache_candidates[0]
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, Mapping)
                and cached.get("cache_key") == cache_key
                and cached.get("prompt_version") == VIRTUAL_RESPONSE_PROMPT_VERSION
                and cached.get("virtual_sample_config") == _normalize_concurrency_config(config)
                and isinstance(cached.get("records"), list)
            ):
                return {
                    "records": [dict(record) for record in cached["records"]],
                    "cached": True,
                    "scheduled_sjt_api_calls": 0,
                    "response_count": len(cached["records"]),
                    "cache_path": str(cache_path.resolve()),
                }
        for candidate_key, candidate_path in cache_candidates[1:]:
            if not candidate_path.is_file():
                continue
            cached = json.loads(candidate_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, Mapping)
                and cached.get("cache_key") == candidate_key
                and cached.get("prompt_version") == VIRTUAL_RESPONSE_PROMPT_VERSION
                and _normalize_concurrency_config(cached.get("virtual_sample_config") or {}) == _normalize_concurrency_config(config)
                and isinstance(cached.get("records"), list)
            ):
                return {
                    "records": [dict(record) for record in cached["records"]],
                    "cached": True,
                    "scheduled_sjt_api_calls": 0,
                    "response_count": len(cached["records"]),
                    "cache_path": str(candidate_path.resolve()),
                }
        seed = int(config.get("seed", 0))

        def validate_sjt(result: Mapping[str, Any], *, allowed_display_ids: set[str]) -> str:
            selected = result.get("selected_option_id")
            if selected not in allowed_display_ids:
                raise ValueError(f"模型选择了无效展示选项 {selected!r}")
            return str(selected)

        async def run_job(respondent_ref: Mapping[str, Any]) -> dict[str, Any]:
            condition_id = str(respondent_ref["condition_id"])
            matched_subject_id = str(respondent_ref["matched_subject_id"])
            option_ids = [
                str(option["option_id"])
                for option in item.get("response_options") or []
                if isinstance(option, Mapping)
            ]
            ordered_ids = balanced_option_order(
                option_ids,
                respondent_index=index_by_matched[matched_subject_id],
                seed=seed,
                item_id=str(item["item_id"]),
            )
            display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
            display_to_original = dict(zip(display_ids, ordered_ids))
            raw_display = await _invoke_with_retry(
                self.sjt_model,
                build_sjt_messages(
                    persona_prompts[str(respondent_ref["respondent_id"])],
                    item,
                    display_option_order=ordered_ids,
                ),
                semaphore=self.semaphore,
                validator=lambda result: validate_sjt(
                    result,
                    allowed_display_ids=set(display_ids),
                ),
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=(
                    f"SJT local retest {condition_id}/{matched_subject_id}/"
                    f"{item.get('item_id')}"
                ),
            )
            selected_option_id = display_to_original[raw_display]
            score_dimension_id = str(
                item.get("target_dimension_id")
                if condition_id == "target"
                else condition_rows.get(condition_id, {}).get("dimension_id")
                or respondent_ref.get("active_dimension_id")
                or item.get("target_dimension_id")
                or ""
            )
            return {
                "record_type": "sjt_local_item_retest_response",
                "run_id": state["run_id"],
                "respondent_id": respondent_ref["respondent_id"],
                "condition_id": condition_id,
                "arm_id": respondent_ref.get("arm_id")
                or condition_rows.get(condition_id, {}).get("arm_id"),
                "group_id": respondent_ref.get("group_id")
                or condition_rows.get(condition_id, {}).get("group_id"),
                "matched_subject_id": matched_subject_id,
                "active_dimension_id": score_dimension_id,
                "active_score": float(
                    (respondent_ref.get("score_values") or {}).get(
                        score_dimension_id,
                        next(iter((respondent_ref.get("score_values") or {}).values())),
                    )
                ),
                "score_values": dict(respondent_ref.get("score_values") or {}),
                "persona_mode": persona_mode,
                "item_bank_id": state.get("item_bank_id"),
                "item_bank_version": state.get("item_bank_version"),
                "item_bank_fingerprint": state.get("item_bank_fingerprint"),
                "item_id": item.get("item_id"),
                "item_version": item_version,
                "display_option_order": [
                    {
                        "display_option_id": display_id,
                        "option_id": original_id,
                    }
                    for display_id, original_id in zip(display_ids, ordered_ids)
                ],
                "raw_display_option_id": raw_display,
                "selected_option_id": selected_option_id,
                "response_mode": "matched_condition_single_item_local_retest",
                "demographics": deepcopy(respondent_ref["demographics"]),
                **demographics_to_columns(respondent_ref["demographics"], config["demographics_snapshot"]),
                "response_temperature": self.response_temperature,
                "model_id": self.model_id,
                "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            }

        results = await asyncio.gather(
            *(run_job(reference) for reference in respondent_refs),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise RuntimeError(f"单题局部复测失败：{errors[0]}") from errors[0]
        records = [dict(result) for result in results if isinstance(result, Mapping)]
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            cache_path,
            {
                "cache_key": cache_key,
                "created_at": utc_timestamp(),
                "item_id": item.get("item_id"),
                "item_version": item_version,
                "records": records,
                "virtual_sample_config": _normalize_concurrency_config(config),
                "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            },
        )
        return {
            "records": records,
            "cached": False,
            "scheduled_sjt_api_calls": len(records),
            "response_count": len(records),
            "cache_path": str(cache_path.resolve()),
        }

    async def run(
        self,
        *,
        state: PSJTState,
        output_dir: Path,
        ipip_neo_path: str | Path = DEFAULT_IPIP_NEO_PATH,
    ) -> dict[str, Any]:
        profile = state.get("construct_profile")
        if not isinstance(profile, Mapping):
            blueprint = state.get("blueprint")
            if isinstance(blueprint, Mapping):
                candidate = blueprint.get("construct_profile_snapshot")
                if isinstance(candidate, Mapping):
                    profile = candidate
        context = build_virtual_response_context(state)
        config = context.get("virtual_sample_config")
        respondent_refs = context.get("virtual_respondents")
        if not isinstance(config, Mapping):
            raise ValueError("虚拟作答前必须先配置虚拟样本")
        if not isinstance(respondent_refs, list) or not respondent_refs:
            raise ValueError("虚拟作答前必须先选择虚拟被试")
        if config.get("sample_size") != len(respondent_refs):
            raise ValueError("虚拟样本配置人数与被试引用数量不一致")
        if config.get("schema_version") == MATCHED_CONDITION_SCHEMA_VERSION:
            if not isinstance(profile, Mapping) or not isinstance(
                profile.get("domain_id"), str
            ):
                raise ValueError("匹配 facet 虚拟作答缺少目标domain构念档案")
            criterion = {
                "domain_id": str(profile["domain_id"]),
                "domain_name_en": str(
                    profile.get("domain_name_en") or profile["domain_id"]
                ),
            }
            return await self._run_matched_condition_profile(
                state=state,
                output_dir=output_dir,
                context=context,
                config=config,
                respondent_refs=respondent_refs,
                criterion=criterion,
                ipip_neo_path=ipip_neo_path,
            )
        raise ValueError(
            "旧 tier/重复作答虚拟样本配置已停用；请重新配置三臂匹配 facet 条件"
        )

def resolve_virtual_response_output_dir(
    state: Mapping[str, Any],
    output_root: str | Path,
) -> Path:
    """Isolate responses by bank, run protocol, and generation round.

    The matched-condition directory used to contain only the bank and schema
    version.  That allowed a resume after changing the SJT/IPIP response
    protocol to land on the old directory and fail only after opening its
    manifest.  Keep the manifest guard, but make protocol changes select a
    fresh directory up front; old response files remain available for audit.
    """

    from sjt_system.runtime.output_paths import scoped_output
    run_dir = scoped_output("virtual_responses", output_root) / str(state["run_id"])
    fingerprint = str(state.get("item_bank_fingerprint") or "unknown")
    version = state.get("item_bank_version") or 0
    config = state.get("virtual_sample_config") or {}
    schema_version = config.get("schema_version")
    protocol_suffix = (
        "-score-tiers-v3"
        if schema_version == 3
        else (
            # Bind the directory to both batched response protocols.  A short
            # digest keeps paths readable while preventing old independent
            # IPIP facet batches from colliding with the selected-facet batch.
            f"-matched-v{MATCHED_CONDITION_SCHEMA_VERSION}"
            f"-r{int(config.get('generation_round') or 1)}"
            f"-p{sha256((VIRTUAL_RESPONSE_PROMPT_VERSION + '|' + IPIP_NEO_REFERENCE_PROMPT_VERSION).encode('utf-8')).hexdigest()[:12]}"
            if schema_version == MATCHED_CONDITION_SCHEMA_VERSION
            else ""
        )
    )
    versioned_dir = (
        run_dir / f"bank-v{version}-{fingerprint[:12]}{protocol_suffix}"
    )
    if versioned_dir.exists():
        return versioned_dir

    legacy_manifest = run_dir / "manifest.json"
    if legacy_manifest.exists():
        try:
            legacy = json.loads(
                legacy_manifest.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            legacy = {}
        if (
            legacy.get("schema_version") == 3
            and legacy.get("prompt_version") == VIRTUAL_RESPONSE_PROMPT_VERSION
            and legacy.get("item_bank_id") == state.get("item_bank_id")
            and legacy.get("item_bank_version")
            == state.get("item_bank_version")
            and legacy.get("item_bank_fingerprint")
            == state.get("item_bank_fingerprint")
        ):
            return run_dir
    return versioned_dir


async def run_virtual_response_simulation(
    state: PSJTState,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    base_model: Any | None = None,
    ipip_neo_path: str | Path = DEFAULT_IPIP_NEO_PATH,
    retry_delay_seconds: float = 1.0,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """执行虚拟作答并返回允许提交到 State 的轻量更新。"""

    config = state.get("virtual_sample_config") or {}
    if not matched_condition_sample_is_current(config, state.get("virtual_respondents")):
        raise ValueError("虚拟被试协议已更新；请重新配置人口学资料、facet 定义和 temperature=1.5")
    if base_model is None:
        configured_model_id = str(config.get("model_id") or "").strip()
        if configured_model_id:
            thinking = os.getenv("VIRTUAL_RESPONDENT_THINKING", "enabled").strip().lower()
            effort = os.getenv("VIRTUAL_RESPONDENT_REASONING_EFFORT", "high").strip().lower()
            base_model = get_model(
                configured_model_id,
                temperature=config["response_temperature"],
                thinking_type=thinking,
                reasoning_effort=(None if thinking == "disabled" else effort),
            )
    max_concurrency = config.get(
        "max_concurrency",
        DEFAULT_MAX_CONCURRENCY,
    )
    max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
    runner = VirtualResponseRunner(
        base_model=base_model,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        request_timeout_seconds=request_timeout_seconds,
        response_temperature=config["response_temperature"],
    )
    output_dir = resolve_virtual_response_output_dir(state, output_root)
    summary = await runner.run(
        state=state,
        output_dir=output_dir,
        ipip_neo_path=ipip_neo_path,
    )
    frozen_reference_ref = (
        state.get("frozen_reference_questionnaire_ref")
        or summary.get("frozen_reference_questionnaire_ref")
    )
    if not frozen_reference_ref and summary.get("ipip_neo_response_count", 0):
        frozen_reference_ref = summary.get("manifest_path")
    return {
        "state_update": {
            "virtual_response_data_ref": summary["manifest_path"],
            "frozen_reference_questionnaire_ref": frozen_reference_ref,
            "virtual_response_summary": summary,
            "virtual_response_item_bank_id": state["item_bank_id"],
            "virtual_response_item_bank_version": state[
                "item_bank_version"
            ],
        },
        "summary": (
            f"已完成 {summary['respondent_count']} 名虚拟被试、"
            f"每名被试一次性回答全部题目（元数据含 {summary.get('group_count', 0)} 个比较 group），共"
            f" {summary['sjt_response_count']} 条SJT记录；"
            f"target整卷重测 {summary.get('target_form_retest_response_count', 0)} 条；"
            f"复用已有SJT作答 {summary['reused_sjt_records']} 条，"
            f"其中复用局部复测题目 {summary.get('reused_local_item_count', 0)} 道；"
            f"复用target重测记录 {summary.get('reused_target_retest_records', 0)} 条；"
            f"新增主施测SJT调用 {summary['scheduled_sjt_api_calls']} 次，"
            f"新增target重测调用 {summary.get('scheduled_target_form_retest_api_calls', 0)} 次，"
            f"IPIP-NEO所选 {summary.get('ipip_neo_facet_count', 0)} 个facet共 {summary.get('ipip_neo_response_count', 0)} 条（首轮冻结，后续轮次复用 {summary.get('frozen_reference_reused_records', 0)} 条；本轮新增 {summary.get('scheduled_ipip_neo_api_calls', 0)} 次调用），"
            f"并发策略=全量并发，作答温度={summary['response_temperature']}，"
            f"人口学库={summary['demographics_version']}"
        ),
    }


async def run_single_item_virtual_retest(
    state: PSJTState,
    item: Mapping[str, Any],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    base_model: Any | None = None,
    retry_delay_seconds: float = 1.0,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """为单题返修提供局部虚拟施测，不改变正式作答数据。"""

    from sjt_system.evaluation.facet_iteration import is_enabled

    if is_enabled(state):
        raise ValueError(
            "固定三轮facet开发必须通过已提交的批次施测；不能使用旧单题虚拟重测路径"
        )
    from sjt_system.runtime.output_paths import scoped_output
    output_root = scoped_output("virtual_responses", output_root)
    config = state.get("virtual_sample_config") or {}
    if not matched_condition_sample_is_current(config, state.get("virtual_respondents")):
        raise ValueError("局部复测需要新协议的冻结人物资料；请重新配置虚拟被试")
    if base_model is None:
        configured_model_id = str(config.get("model_id") or "").strip()
        if configured_model_id:
            thinking = os.getenv("VIRTUAL_RESPONDENT_THINKING", "enabled").strip().lower()
            effort = os.getenv("VIRTUAL_RESPONDENT_REASONING_EFFORT", "high").strip().lower()
            base_model = get_model(
                configured_model_id,
                temperature=config["response_temperature"],
                thinking_type=thinking,
                reasoning_effort=(None if thinking == "disabled" else effort),
            )
    runner = VirtualResponseRunner(
        base_model=base_model,
        max_concurrency=config.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
        max_retries=config.get("max_retries", DEFAULT_MAX_RETRIES),
        retry_delay_seconds=retry_delay_seconds,
        request_timeout_seconds=request_timeout_seconds,
        response_temperature=config["response_temperature"],
    )
    return await runner.run_single_item_retest(
        state=state,
        item=item,
        output_root=output_root,
    )
