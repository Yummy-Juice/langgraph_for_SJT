"""Concurrent virtual-response generation and persistence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
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
from sjt_system.runtime.telemetry import job_context as telemetry_job_context
from sjt_system.runtime.io import write_json_atomic as _write_json_atomic
from sjt_system.evaluation.respondents import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    MAX_ALLOWED_CONCURRENCY,
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
)
from sjt_system.state import PSJTState
from sjt_system.runtime.trace import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEO_FFI_PATH = PROJECT_ROOT / "docs" / "Neo-FFI.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "virtual_responses"
VIRTUAL_RESPONSE_PROMPT_VERSION = "matched-condition-single-sjt-v1"
PERSONA_SUMMARY_PROMPT_VERSION = "liu-item-response-summary-v2"
IPIP_NEO_REFERENCE_PROMPT_VERSION = "ipip-neo-mussel-five-facets-v3"
# Mussel's five NEO-PI-R facets.  The SJT may target one facet at a time, but
# the same target respondents answer this complete reference set so that
# target and non-target effects use one fixed, human-comparable scope.
MUSSEL_IPIP_FACET_IDS = (
    "neuroticism_self_consciousness",  # N4
    "extraversion_gregariousness",  # E2
    "openness_ideas",  # O5
    "agreeableness_compliance",  # A4
    "conscientiousness_self_discipline",  # C5
)


class SJTSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_option_id: str


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
    intentionally not replaced with a hard-coded target facet: if authoring
    selected five facets, the same target respondents answer all five scales.
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
) -> str:
    """Build the active explicit-score prompt."""

    if persona_mode not in SUPPORTED_PERSONA_MODES:
        raise ValueError(f"不支持的人格提示模式：{persona_mode}")
    if persona_mode == PERSONA_MODE_SCORE_PROFILE:
        values = profile.get("score_values")
        specs = list(score_specs or profile.get("score_specs") or [])
        if not isinstance(values, Mapping):
            raise ValueError("score_profile 模式缺少分数或 score_specs")
        if not specs:
            specs = [
                {
                    "dimension_id": str(profile.get("active_dimension_id") or dimension_id),
                    "level": "facet",
                    "domain_name_en": profile.get("domain_name_en") or "facet",
                    "domain_name": profile.get("domain_name") or "facet",
                    "facet_name_en": profile.get("facet_name_en") or profile.get("active_dimension_id"),
                    "facet_name": profile.get("facet_name") or profile.get("active_dimension_id"),
                }
                for dimension_id in values
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
            score_lines.append(f"{label} | {value:.1f}")
        return (
            "请想象你正在扮演一个特定的人。\n"
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
            "现在请以这个人的真实行为倾向回答一道情境判断题。\n"
            "选择最可能采取的行为，而非理论上最好、最正确或最受赞许的行为。\n"
            "每次作答相互独立，不复述人格名称或分数。\n"
            '仅返回：{"selected_option_id":"选项编号"}。'
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

    if "[PERSONALITY SCORES]" in persona_prompt:
        system_message = persona_prompt
    else:
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
    items: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Build a facet-blind IPIP-NEO batch for one selected facet."""

    if not items:
        raise ValueError("IPIP-NEO参照问卷缺少题目")
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
        f"每题独立判断，并按题目顺序返回{count}个整数。"
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


async def _invoke_with_retry(
    runnable: Any,
    messages: list[tuple[str, str]],
    *,
    semaphore: asyncio.Semaphore,
    validator: Any,
    max_retries: int,
    retry_delay_seconds: float,
    request_timeout_seconds: float,
    job_label: str,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                with telemetry_job_context(job_label, attempt=attempt + 1):
                    raw_result = await asyncio.wait_for(
                        runnable.ainvoke(messages),
                        timeout=request_timeout_seconds,
                    )
            return validator(_structured_result_dict(raw_result))
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                last_error = TimeoutError(
                    f"单次请求超过 {request_timeout_seconds:g} 秒"
                )
            else:
                last_error = exc
            if attempt >= max_retries:
                break
            emit_progress(
                {
                    "type": "request_retry",
                    "retry_kind": (
                        "network_timeout"
                        if isinstance(exc, TimeoutError)
                        else "output_repair"
                    ),
                    "job_label": job_label,
                    "attempt": attempt + 2,
                    "max_attempts": max_retries + 1,
                    "reason": str(last_error),
                }
            )
            await asyncio.sleep(retry_delay_seconds * (2**attempt))
    raise RuntimeError(
        f"{job_label} 在 {max_retries + 1} 次尝试后仍失败：{last_error}"
    ) from last_error


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
        "score_prompt_version": SCORE_PROFILE_PROMPT_VERSION,
        "score_generator_version": SCORE_PROFILE_GENERATOR_VERSION,
        "persona_summary_prompt_version": PERSONA_SUMMARY_PROMPT_VERSION,
        "criterion": dict(criterion),
        "model_id": model_id,
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
) -> dict[str, Any]:
    """Bind a reusable raw response to the new frozen bank metadata."""

    return {
        **dict(record),
        "run_id": state.get("run_id"),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "response_source": response_source,
    }


def _matched_source_is_compatible(
    source_manifest: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    model_id: str,
) -> bool:
    """Check that old raw responses were generated under the same protocol."""

    return all(
        source_manifest.get(field) == expected
        for field, expected in (
            ("schema_version", MATCHED_CONDITION_SCHEMA_VERSION),
            ("status", "completed"),
            ("run_id", state.get("run_id")),
            ("sample_size_per_condition", config.get("sample_size_per_condition")),
            ("conditions", list(conditions)),
            ("persona_modes", [PERSONA_MODE_SCORE_PROFILE]),
            ("model_id", model_id),
            ("prompt_version", VIRTUAL_RESPONSE_PROMPT_VERSION),
            ("score_prompt_version", MATCHED_CONDITION_PROMPT_VERSION),
            ("generator_version", MATCHED_CONDITION_GENERATOR_VERSION),
            ("virtual_sample_config", dict(config)),
        )
    )


def _local_retest_cache_records(
    state: Mapping[str, Any],
    *,
    current_items: Mapping[str, Mapping[str, Any]],
    expected_count: int,
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
            if candidate.get("version") != final_item.get("version"):
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
                and record.get("item_version") == final_item.get("version")
            ]
            if len(normalized) == expected_count:
                found[item_id] = normalized
    return found


def _seed_matched_response_files(
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    model_id: str,
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

    source_ref = state.get("previous_virtual_response_data_ref")
    if not isinstance(source_ref, str) or not source_ref:
        return {"source_manifest_path": None, "source_compatible": False, "reused_sjt_records": 0, "reused_local_item_count": 0, "reused_target_retest_records": 0, "reused_ipip_neo_records": 0}
    source_manifest_path = Path(source_ref).resolve()
    if not source_manifest_path.is_file():
        return {"source_manifest_path": None, "source_compatible": False, "reused_sjt_records": 0, "reused_local_item_count": 0, "reused_target_retest_records": 0, "reused_ipip_neo_records": 0}
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {"source_manifest_path": None, "source_compatible": False, "reused_sjt_records": 0, "reused_local_item_count": 0, "reused_target_retest_records": 0, "reused_ipip_neo_records": 0}
    if not isinstance(source_manifest, Mapping) or not _matched_source_is_compatible(
        source_manifest,
        state=state,
        config=config,
        conditions=conditions,
        model_id=model_id,
    ):
        return {"source_manifest_path": str(source_manifest_path), "source_compatible": False, "reused_sjt_records": 0, "reused_local_item_count": 0, "reused_target_retest_records": 0, "reused_ipip_neo_records": 0}

    current_versions = {
        str(item.get("item_id")): item.get("item_version")
        for item in sjt_items
        if isinstance(item, Mapping) and item.get("item_id")
    }
    current_items = {
        str(item.get("item_id")): item
        for item in state.get("frozen_item_bank") or []
        if isinstance(item, Mapping) and item.get("item_id")
    }
    expected_per_condition = int(config.get("sample_size_per_condition") or 0)
    expected_local_records = expected_per_condition * len(MATCHED_CONDITION_IDS)
    local_records_by_item = _local_retest_cache_records(
        state,
        current_items=current_items,
        expected_count=expected_local_records,
    )
    local_item_ids = set(local_records_by_item)
    source_sjt = _load_jsonl_records(source_manifest_path.parent / "sjt_responses.jsonl")
    reusable_sjt: list[dict[str, Any]] = []
    seen_sjt: set[tuple[Any, ...]] = set()
    for record in source_sjt:
        item_id = str(record.get("item_id") or "")
        key = (
            record.get("respondent_id"),
            record.get("persona_mode"),
            item_id,
            record.get("item_version"),
            record.get("condition_id"),
            record.get("matched_subject_id"),
        )
        if (
            item_id not in current_versions
            # 返修题的作答由 local retest 缓存提供（优先），不重复复制上一版记录
            or item_id in local_item_ids
            or record.get("item_version") != current_versions[item_id]
            or key in seen_sjt
        ):
            continue
        seen_sjt.add(key)
        reusable_sjt.append(
            _rewrite_response_identity(
                record,
                state,
                response_source="reused_unchanged_item",
            )
        )
    for item_id, records in local_records_by_item.items():
        reusable_sjt.extend(records)
    if reusable_sjt:
        _append_jsonl_records(sjt_path, reusable_sjt)

    option_orders: list[dict[str, Any]] = []
    source_option_path = source_manifest.get("option_order_path")
    if isinstance(source_option_path, str) and Path(source_option_path).is_file():
        for record in _load_jsonl_records(Path(source_option_path)):
            item_id = str(record.get("item_id") or "")
            if item_id in current_versions and record.get("item_version") == current_versions[item_id]:
                option_orders.append(_rewrite_response_identity(record, state, response_source="reused_unchanged_item"))
    for records in local_records_by_item.values():
        option_orders.extend(
            {
                key: record.get(key)
                for key in (
                    "run_id", "respondent_id", "condition_id", "matched_subject_id",
                    "item_id", "item_version", "display_option_order",
                    "raw_display_option_id", "selected_option_id",
                )
            }
            for record in records
        )
    if option_orders:
        _append_jsonl_records(option_order_path, option_orders)

    target_retest_records: list[dict[str, Any]] = []
    source_target_meta = source_manifest.get("target_form_retest") or {}
    source_target_path = source_target_meta.get("path") if isinstance(source_target_meta, Mapping) else None
    allowed_administrations = {
        int(value)
        for value in source_target_meta.get("administration_ids") or []
        if isinstance(value, int) and not isinstance(value, bool)
    } if isinstance(source_target_meta, Mapping) else set()
    if isinstance(source_target_path, str) and Path(source_target_path).is_file():
        for record in _load_jsonl_records(Path(source_target_path)):
            item_id = str(record.get("item_id") or "")
            if (
                item_id in current_versions
                and record.get("item_version") == current_versions[item_id]
                and record.get("administration_id") in allowed_administrations
            ):
                target_retest_records.append(
                    _rewrite_response_identity(record, state, response_source="reused_unchanged_item")
                )
    if target_retest_records:
        _append_jsonl_records(target_retest_path, target_retest_records)

    reused_reference_counts: dict[str, int] = {}
    references = source_manifest.get("reference_questionnaires") or {}
    for name, target_path in (("ipip_neo", ipip_neo_path),):
        metadata = references.get(name) if isinstance(references, Mapping) else None
        source_path = metadata.get("path") if isinstance(metadata, Mapping) else None
        records = (
            _load_jsonl_records(Path(source_path))
            if isinstance(source_path, str) and Path(source_path).is_file()
            else []
        )
        if records:
            _append_jsonl_records(
                target_path,
                [
                    _rewrite_response_identity(
                        record,
                        state,
                        response_source=f"reused_{name}",
                    )
                    for record in records
                ],
            )
        reused_reference_counts[name] = len(records)

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
    """以受控并发执行SJT、IPIP-NEO参照问卷与整卷重测。"""

    def __init__(
        self,
        *,
        base_model: Any | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = 1.0,
        request_timeout_seconds: float | None = None,
    ) -> None:
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency < 1
            or max_concurrency > MAX_ALLOWED_CONCURRENCY
        ):
            raise ValueError(
                "max_concurrency 必须是 1 到 "
                f"{MAX_ALLOWED_CONCURRENCY} 之间的整数"
            )
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
            or max_retries > 5
        ):
            raise ValueError("max_retries 必须是 0 到 5 之间的整数")
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

        self.base_model = base_model or get_model()
        self.sjt_model, self.structured_output_method = (
            with_compatible_structured_output(
                self.base_model,
                SJTSelectionOutput,
            )
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
        ):
            raise ValueError("同一模型的结构化输出方式不一致")
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.semaphore = asyncio.Semaphore(max_concurrency)
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
        profiles = resolve_virtual_respondent_profiles(respondent_refs)
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
        # The main SJT target can be one facet, but the external reference
        # scope is fixed to Mussel's five facets for every run.  This is what
        # makes the non-target part of I_g observable instead of silently
        # dropping the other four constructs.
        selected_ipip_facet_ids = list(MUSSEL_IPIP_FACET_IDS)
        # Experiment-only opt-out. Existing CLI runs preserve reference behavior.
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
        condition_specs: dict[str, list[dict[str, Any]]] = {}
        for condition_id, row in condition_rows.items():
            dimension_id = str(row.get("dimension_id") or "")
            condition_specs[condition_id] = [{
                "dimension_id": dimension_id,
                "level": "facet",
                "domain_id": row.get("domain_id"),
                "domain_name_en": row.get("domain_name_en") or row.get("domain_id"),
                "domain_name": row.get("domain_name") or row.get("domain_id"),
                "facet_name_en": row.get("facet_name_en") or dimension_id,
                "facet_name": row.get("facet_name") or dimension_id,
            }]

        output_dir.mkdir(parents=True, exist_ok=True)
        sjt_path = output_dir / "sjt_responses.jsonl"
        target_retest_path = output_dir / "target_form_retest_responses.jsonl"
        ipip_path = output_dir / "ipip_neo_responses.jsonl"
        option_order_path = output_dir / "option_orders.jsonl"
        manifest_path = output_dir / "manifest.json"
        scoring_snapshot_path = output_dir / "scoring_snapshot.json"
        score_profiles_path = output_dir / "score_profiles.json"
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
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest.get("simulation_signature") != signature:
                raise ValueError("输出目录已包含不同配置的虚拟作答，拒绝混合数据")
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

        reuse_summary = {"source_manifest_path": None, "source_compatible": False, "reused_sjt_records": 0, "reused_local_item_count": 0, "reused_target_retest_records": 0, "reused_ipip_neo_records": 0}
        if not manifest_path.exists():
            reuse_summary = _seed_matched_response_files(
                state=state,
                config=config,
                conditions=conditions,
                model_id=self.model_id,
                sjt_items=sjt_items,
                sjt_path=sjt_path,
                target_retest_path=target_retest_path,
                ipip_neo_path=ipip_path,
                option_order_path=option_order_path,
            )

        sjt_keys = _load_jsonl_keys(
            sjt_path,
            ("respondent_id", "persona_mode", "item_id", "item_version", "condition_id", "matched_subject_id"),
        )
        initial_sjt_count = len(sjt_keys)
        expected_sjt_records = len(respondent_refs) * len(sjt_items)
        scheduled_sjt_calls = expected_sjt_records - initial_sjt_count
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
        target_retest_keys = _load_jsonl_keys(
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
        )
        initial_target_retest_count = len(target_retest_keys)
        expected_target_retest_records = (
            len(target_references)
            * len(sjt_items)
            * (target_administration_count - 1)
        )
        scheduled_target_retest_calls = (
            expected_target_retest_records - initial_target_retest_count
        )
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
        _write_json_atomic(score_profiles_path, {
            "schema_version": 2,
            "sampling_design": config.get("sampling_design"),
            "generator_version": config.get("generator_version"),
            "prompt_version": config.get("prompt_version"),
            "score_distribution": config.get("score_distribution"),
            "conditions": deepcopy(conditions),
            "profiles": list(respondent_refs),
        })
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
                    "version": item.get("version"),
                    "target_dimension_id": item.get("target_dimension_id"),
                    "context_category": item.get("context_category"),
                    "option_ids": [option.get("option_id") for option in item.get("response_options") or [] if isinstance(option, Mapping)],
                    "scoring_key": dict(item.get("scoring_key") or {}),
                }
                for item in state["frozen_item_bank"]
            ],
        })
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
            "sjt_response_mode": "matched_condition_single_item_balanced_order",
            "target_form_administration_count": target_administration_count,
            "target_form_retest": {
                "path": str(target_retest_path.resolve()),
                "administration_ids": list(
                    range(2, target_administration_count + 1)
                ),
                "response_mode": "target_form_retest_balanced_order",
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
                    "response_mode": "independent_mussel_five_facet_batches",
                    "scope": "mussel_five_facets_fixed",
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
            "expected_sjt_records": expected_sjt_records,
            "expected_target_form_retest_records": (
                expected_target_retest_records
            ),
            "expected_ipip_neo_records": expected_ipip_records,
            "source_manifest_path": reuse_summary.get("source_manifest_path"),
            "source_response_reuse": deepcopy(reuse_summary),
            "score_profiles_path": str(score_profiles_path.resolve()),
            "scoring_snapshot_path": str(scoring_snapshot_path.resolve()),
            "option_order_path": str(option_order_path.resolve()),
            "started_at": utc_timestamp(),
            "interpretation_limitations": [
                "These are exploratory virtual screening results, not formal human psychometric evidence.",
                "The three top-level arms share one matched score sequence; each facet group is simulated independently and only its named facet is provided to the model.",
                "The target-form retest measures model/prompt response stability under a second balanced option order; it is not human test-retest reliability.",
                "Mussel's five IPIP-NEO facet scales are answered by the same target respondents; this is development-stage convergent/discriminant evidence, not human criterion data.",
            ],
        }
        _write_json_atomic(manifest_path, manifest)
        persona_prompts = {
            respondent_id: build_persona_prompt(
                profile,
                persona_mode=persona_mode,
                score_specs=condition_specs[str(profile["condition_id"])],
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

        async def run_sjt_job(respondent_ref: Mapping[str, Any], item: Mapping[str, Any]) -> None:
            condition_id = str(respondent_ref["condition_id"])
            matched_subject_id = str(respondent_ref["matched_subject_id"])
            key = (respondent_ref["respondent_id"], persona_mode, item.get("item_id"), item.get("item_version"), condition_id, matched_subject_id)
            if key in sjt_keys:
                return
            option_ids = [str(option["option_id"]) for option in item["response_options"]]
            ordered_ids = balanced_option_order(option_ids, respondent_index=index_by_matched[matched_subject_id], seed=seed, item_id=str(item.get("item_id")))
            display_ids = [chr(ord("A") + index) for index in range(len(ordered_ids))]
            display_to_original = dict(zip(display_ids, ordered_ids))
            raw_display = await _invoke_with_retry(
                self.sjt_model,
                build_sjt_messages(persona_prompts[str(respondent_ref["respondent_id"])], item, display_option_order=ordered_ids),
                semaphore=self.semaphore,
                validator=lambda result: validate_sjt(result, allowed_display_ids=set(display_ids)),
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=f"SJT {condition_id}/{matched_subject_id}/{item.get('item_id')}",
            )
            selected_option_id = display_to_original[raw_display]
            display_order = [{"display_option_id": display_id, "option_id": original_id} for display_id, original_id in zip(display_ids, ordered_ids)]
            record = {
                "record_type": "sjt_response",
                "run_id": state["run_id"],
                "respondent_id": respondent_ref["respondent_id"],
                "condition_id": condition_id,
                "arm_id": respondent_ref.get("arm_id") or condition_rows.get(condition_id, {}).get("arm_id"),
                "group_id": respondent_ref.get("group_id") or condition_rows.get(condition_id, {}).get("group_id"),
                "matched_subject_id": matched_subject_id,
                "active_dimension_id": respondent_ref.get("active_dimension_id"),
                "active_score": next(iter(respondent_ref.get("score_values", {}).values())),
                "persona_mode": persona_mode,
                "item_bank_id": state["item_bank_id"],
                "item_bank_version": state["item_bank_version"],
                "item_bank_fingerprint": state.get("item_bank_fingerprint"),
                "item_id": item.get("item_id"),
                "item_version": item.get("item_version"),
                "display_option_order": display_order,
                "raw_display_option_id": raw_display,
                "selected_option_id": selected_option_id,
                "response_mode": "matched_condition_single_item_balanced_order",
                "model_id": self.model_id,
                "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            }
            await self._append_record(sjt_path, record)
            await self._append_record(option_order_path, {key_name: record.get(key_name) for key_name in ("run_id", "respondent_id", "condition_id", "matched_subject_id", "item_id", "item_version", "display_option_order", "raw_display_option_id", "selected_option_id")})
            sjt_keys.add(key)
            await report_completed()

        async def run_target_retest_job(
            respondent_ref: Mapping[str, Any],
            item: Mapping[str, Any],
            administration_id: int,
        ) -> None:
            matched_subject_id = str(respondent_ref["matched_subject_id"])
            respondent_id = str(respondent_ref["respondent_id"])
            key = (
                respondent_id,
                persona_mode,
                item.get("item_id"),
                item.get("item_version"),
                "target",
                matched_subject_id,
                administration_id,
            )
            if key in target_retest_keys:
                return
            option_ids = [
                str(option["option_id"])
                for option in item["response_options"]
            ]
            ordered_ids = balanced_option_order(
                option_ids,
                respondent_index=index_by_matched[matched_subject_id],
                seed=seed + 100_003 * administration_id,
                item_id=str(item.get("item_id")),
            )
            display_ids = [
                chr(ord("A") + index) for index in range(len(ordered_ids))
            ]
            display_to_original = dict(zip(display_ids, ordered_ids))
            raw_display = await _invoke_with_retry(
                self.sjt_model,
                build_sjt_messages(
                    persona_prompts[respondent_id],
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
                    f"SJT target retest {administration_id}/"
                    f"{matched_subject_id}/{item.get('item_id')}"
                ),
            )
            selected_option_id = display_to_original[raw_display]
            record = {
                "record_type": "sjt_target_form_retest_response",
                "run_id": state["run_id"],
                "respondent_id": respondent_id,
                "condition_id": "target",
                "arm_id": "target",
                "group_id": respondent_ref.get("group_id")
                or condition_rows["target"].get("group_id"),
                "matched_subject_id": matched_subject_id,
                "active_dimension_id": respondent_ref.get(
                    "active_dimension_id"
                ),
                "active_score": next(
                    iter(respondent_ref.get("score_values", {}).values())
                ),
                "persona_mode": persona_mode,
                "administration_id": administration_id,
                "item_bank_id": state["item_bank_id"],
                "item_bank_version": state["item_bank_version"],
                "item_bank_fingerprint": state.get("item_bank_fingerprint"),
                "item_id": item.get("item_id"),
                "item_version": item.get("item_version"),
                "display_option_order": [
                    {
                        "display_option_id": display_id,
                        "option_id": original_id,
                    }
                    for display_id, original_id in zip(
                        display_ids, ordered_ids
                    )
                ],
                "raw_display_option_id": raw_display,
                "selected_option_id": selected_option_id,
                "response_mode": "target_form_retest_balanced_order",
                "model_id": self.model_id,
                "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            }
            await self._append_record(target_retest_path, record)
            target_retest_keys.add(key)

        emit_progress({"type": "simulation_stage", "stage": "SJT matched facet response", "status": "started", "total": scheduled_sjt_calls})
        results = await asyncio.gather(*(run_sjt_job(reference, item) for reference in respondent_refs for item in sjt_items), return_exceptions=True)
        sjt_errors = [result for result in results if isinstance(result, Exception)]
        emit_progress({"type": "simulation_stage", "stage": "SJT matched facet response", "status": "completed" if not sjt_errors and len(sjt_keys) == expected_sjt_records else "failed", "total": scheduled_sjt_calls})
        target_retest_errors: list[Exception] = []
        if not sjt_errors:
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "SJT target form retest",
                    "status": "started",
                    "total": scheduled_target_retest_calls,
                }
            )
            target_retest_results = await asyncio.gather(
                *(
                    run_target_retest_job(reference, item, administration_id)
                    for administration_id in range(
                        2, target_administration_count + 1
                    )
                    for reference in target_references
                    for item in sjt_items
                ),
                return_exceptions=True,
            )
            target_retest_errors = [
                result
                for result in target_retest_results
                if isinstance(result, Exception)
            ]
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "SJT target form retest",
                    "status": (
                        "completed"
                        if not target_retest_errors
                        and len(target_retest_keys)
                        == expected_target_retest_records
                        else "failed"
                    ),
                    "total": scheduled_target_retest_calls,
                }
            )
        reference_errors: list[Exception] = []
        scheduled_ipip_calls = 0

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
            ipip_scale: Mapping[str, Any],
        ) -> None:
            nonlocal scheduled_ipip_calls
            respondent_id = str(respondent_ref["respondent_id"])
            facet_code = str(ipip_scale["facet_code"])
            facet_id = str(ipip_scale["facet_id"])
            items = list(ipip_scale.get("items") or [])
            keys = {
                (respondent_id, facet_code, str(item["item_id"]))
                for item in items
            }
            present = keys.intersection(ipip_keys)
            if present:
                if len(present) != len(keys):
                    raise ValueError(
                        f"IPIP-NEO {respondent_id}/{facet_code} 存在不完整批次，拒绝混合恢复"
                    )
                return
            scheduled_ipip_calls += 1
            ratings = await _invoke_with_retry(
                self.ipip_neo_model,
                build_ipip_neo_messages(
                    persona_prompts[respondent_id],
                    items,
                ),
                semaphore=self.semaphore,
                validator=lambda result: validate_ipip_reference(
                    result,
                    expected_count=len(items),
                ),
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=f"IPIP-NEO selected facet reference {respondent_id}/{facet_code}",
            )
            records = []
            for item, rating in zip(items, ratings):
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
                        "response_mode": "selected_reference_facet_batch_10",
                        "model_id": self.model_id,
                        "prompt_version": IPIP_NEO_REFERENCE_PROMPT_VERSION,
                    }
                )
            await self._append_records(ipip_path, records)
            ipip_keys.update(keys)

        if not sjt_errors and not target_retest_errors and ipip_scales is not None:
            ipip_job_count = len(target_references) * len(ipip_scales)
            emit_progress({"type": "simulation_stage", "stage": "IPIP-NEO selected facet response", "status": "started", "total": ipip_job_count})
            ipip_results = await asyncio.gather(
                *(
                    run_ipip_reference_job(reference, scale)
                    for reference in target_references
                    for scale in ipip_scales
                ),
                return_exceptions=True,
            )
            reference_errors.extend(result for result in ipip_results if isinstance(result, Exception))
            emit_progress({"type": "simulation_stage", "stage": "IPIP-NEO selected facet response", "status": "failed" if reference_errors else "completed", "total": ipip_job_count})

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
            "persona_summary_count": 0,
            "sjt_item_count": len(sjt_items),
            "sjt_response_count": len(sjt_keys),
            "target_form_retest_response_count": len(target_retest_keys),
            "ipip_neo_path": str(ipip_path.resolve()),
            "ipip_neo_response_count": len(ipip_keys),
            "scheduled_persona_summary_api_calls": 0,
            "scheduled_sjt_api_calls": scheduled_sjt_calls,
            "scheduled_target_form_retest_api_calls": (
                scheduled_target_retest_calls
            ),
            "scheduled_ipip_neo_api_calls": scheduled_ipip_calls,
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
            "reused_local_item_count": reuse_summary.get("reused_local_item_count", 0),
            "source_response_reuse": deepcopy(reuse_summary),
            "source_manifest_path": reuse_summary.get("source_manifest_path"),
            "model_id": self.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "score_prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
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
        if not isinstance(item.get("version"), int) or isinstance(
            item.get("version"), bool
        ):
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

        profiles = resolve_virtual_respondent_profiles(respondent_refs)
        persona_mode = PERSONA_MODE_SCORE_PROFILE
        condition_rows = {
            str(row["condition_id"]): dict(row) for row in condition_groups
        }
        condition_specs: dict[str, list[dict[str, Any]]] = {}
        for condition_id, row in condition_rows.items():
            dimension_id = str(row.get("dimension_id") or "")
            condition_specs[condition_id] = [
                {
                    "dimension_id": dimension_id,
                    "level": "facet",
                    "domain_id": row.get("domain_id"),
                    "domain_name_en": row.get("domain_name_en") or row.get("domain_id"),
                    "domain_name": row.get("domain_name") or row.get("domain_id"),
                    "facet_name_en": row.get("facet_name_en") or dimension_id,
                    "facet_name": row.get("facet_name") or dimension_id,
                }
            ]
        profile_by_id = {
            str(profile["respondent_id"]): profile for profile in profiles
        }
        persona_prompts = {
            respondent_id: build_persona_prompt(
                profile,
                persona_mode=persona_mode,
                score_specs=condition_specs[str(profile["condition_id"])],
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

        cache_payload = {
            "run_id": state.get("run_id"),
            "item_bank_id": state.get("item_bank_id"),
            "item_bank_version": state.get("item_bank_version"),
            "item_bank_fingerprint": state.get("item_bank_fingerprint"),
            "model_id": self.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "config": dict(config),
            "respondents": list(respondent_refs),
            "item": dict(item),
        }
        cache_key = sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_dir = (
            Path(output_root)
            / str(state.get("run_id") or "unknown")
            / "item_local_retests"
        )
        cache_path = cache_dir / f"{item['item_id']}-v{item['version']}-{cache_key[:16]}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, Mapping)
                and cached.get("cache_key") == cache_key
                and isinstance(cached.get("records"), list)
            ):
                return {
                    "records": [dict(record) for record in cached["records"]],
                    "cached": True,
                    "scheduled_sjt_api_calls": 0,
                    "response_count": len(cached["records"]),
                    "cache_path": str(cache_path.resolve()),
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
                "active_dimension_id": respondent_ref.get("active_dimension_id"),
                "active_score": next(iter(respondent_ref.get("score_values", {}).values())),
                "persona_mode": persona_mode,
                "item_bank_id": state.get("item_bank_id"),
                "item_bank_version": state.get("item_bank_version"),
                "item_bank_fingerprint": state.get("item_bank_fingerprint"),
                "item_id": item.get("item_id"),
                "item_version": item.get("version"),
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
                "item_version": item.get("version"),
                "records": records,
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
    """Isolate responses by frozen-bank identity while supporting old runs."""

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
            f"-matched-v{MATCHED_CONDITION_SCHEMA_VERSION}"
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
    if base_model is None:
        configured_model_id = str(config.get("model_id") or "").strip()
        if configured_model_id:
            base_model = get_model(configured_model_id)
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
    )
    output_dir = resolve_virtual_response_output_dir(state, output_root)
    summary = await runner.run(
        state=state,
        output_dir=output_dir,
        ipip_neo_path=ipip_neo_path,
    )
    return {
        "state_update": {
            "virtual_response_data_ref": summary["manifest_path"],
            "virtual_response_summary": summary,
            "virtual_response_item_bank_id": state["item_bank_id"],
            "virtual_response_item_bank_version": state[
                "item_bank_version"
            ],
        },
        "summary": (
            f"已完成 {summary['respondent_count']} 名虚拟被试、"
            f"{summary.get('condition_count', 3)} 个匹配条件组、主施测每组每人每题1次作答，共"
            f" {summary['sjt_response_count']} 条SJT记录；"
            f"target整卷重测 {summary.get('target_form_retest_response_count', 0)} 条；"
            f"复用已有SJT作答 {summary['reused_sjt_records']} 条，"
            f"其中复用局部复测题目 {summary.get('reused_local_item_count', 0)} 道；"
            f"复用target重测记录 {summary.get('reused_target_retest_records', 0)} 条；"
            f"新增主施测SJT调用 {summary['scheduled_sjt_api_calls']} 次，"
            f"新增target重测调用 {summary.get('scheduled_target_form_retest_api_calls', 0)} 次，"
            f"IPIP-NEO Mussel五个facet共 {summary.get('ipip_neo_response_count', 0)} 条（新增调用 {summary.get('scheduled_ipip_neo_api_calls', 0)} 次），"
            f"最大并发 {summary['max_concurrency']}"
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

    from sjt_system.runtime.output_paths import scoped_output
    output_root = scoped_output("virtual_responses", output_root)
    config = state.get("virtual_sample_config") or {}
    if base_model is None:
        configured_model_id = str(config.get("model_id") or "").strip()
        if configured_model_id:
            base_model = get_model(configured_model_id)
    runner = VirtualResponseRunner(
        base_model=base_model,
        max_concurrency=config.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
        max_retries=config.get("max_retries", DEFAULT_MAX_RETRIES),
        retry_delay_seconds=retry_delay_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
    return await runner.run_single_item_retest(
        state=state,
        item=item,
        output_root=output_root,
    )
