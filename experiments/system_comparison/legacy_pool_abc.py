"""Evaluate the frozen A/B/C forms with the legacy virtual-respondent pool.

This is deliberately separate from the development workflow.  It treats the
virtual respondent as a black box: the SJT and NEO-FFI calls receive the
respondent's item-level personality answers (and the cached legacy summary),
but the evaluator never supplies a target score to either questionnaire.

The module does not calculate the development-only target-recovery R², VTS, or
construct-selectivity S.  It reports only item CITC/option statistics,
Cronbach's alpha, and SJT-total versus NEO-FFI correlations.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.evaluation.simulation import (
    NeoFFIBatchOutput,
    SJTSelectionOutput,
    _invoke_with_retry,
    _load_jsonl_keys,
    _load_jsonl_records,
    build_neo_ffi_messages,
    load_neo_ffi,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import read_ledger, run_context

from .config import fingerprint
from .storage import write_csv, write_json


DEFAULT_LEGACY_PROJECT = Path(r"E:\DR_projects\SJT\Code\langgraph_for_SJT")
DEFAULT_IMPORTED_SUMMARY_POOL = (
    Path(__file__).resolve().parents[2]
    / "experiment_data"
    / "legacy_persona_summary_pool"
)
NEO_DOMAINS = ("E", "N", "O", "A", "C")
OPTION_IDS = ("A", "B", "C", "D")
PROMPT_VERSION = "legacy-pool-abc-black-box-v1"
PERSONA_PROMPT_VERSIONS = {
    "items_plus_summary": "legacy-items-plus-summary-v1",
    "summary_only": "legacy-summary-only-v1",
    "summary_embodied_probability": "legacy-summary-embodied-probability-v2-first-person",
}
METRIC_VERSION = "legacy-pool-abc-ctt-alpha-neo-v1"
CITC_THRESHOLD = 0.20


class ChoiceProbabilityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice_probabilities: dict[str, float]


@dataclass(frozen=True)
class LegacyABCConfig:
    experiment: Path
    legacy_project: Path = DEFAULT_LEGACY_PROJECT
    model_id: str | None = None
    neo_ffi_path: Path | None = None
    max_concurrency: int = 5
    max_retries: int = 2
    timeout_seconds: float | None = None
    respondent_limit: int | None = None
    output: Path | None = None
    persona_mode: str = "items_plus_summary"
    summary_pool: Path | None = None
    respondent_source: str = "legacy_100"
    sampling_seed: int = 20260913

    def validate(self) -> None:
        if self.max_concurrency < 1 or self.max_concurrency > 50:
            raise ValueError("max_concurrency 必须在1至50之间")
        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries 必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须为正数")
        if self.respondent_limit is not None and self.respondent_limit < 3:
            raise ValueError("respondent_limit 至少为3")
        if self.persona_mode not in PERSONA_PROMPT_VERSIONS:
            raise ValueError(
                "persona_mode必须是items_plus_summary、summary_only或"
                "summary_embodied_probability"
            )
        if self.respondent_source not in {"legacy_100", "all_285"}:
            raise ValueError("respondent_source必须是legacy_100或all_285")
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed, int
        ):
            raise ValueError("sampling_seed必须是整数")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_id(model: Any, requested: str | None) -> str:
    if requested:
        return requested
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return os.getenv("MODEL_ID", "deepseek-v4-flash")


def _json_value(value: Any) -> Any:
    """Convert numpy scalars and non-finite values for strict JSON output."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _legacy_form_paths(experiment: Path) -> dict[str, Path]:
    paths = {
        "A": experiment / "A" / "round_01" / "form.json",
        "B": experiment / "B" / "round_01" / "form.json",
        "C": experiment / "C" / "final.json",
    }
    missing = [f"{method}: {path}" for method, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少已冻结的A/B/C问卷：" + "；".join(missing))
    return paths


def _validate_form(method: str, payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{method}问卷没有items：{path}")
    if len(items) != 16:
        raise ValueError(f"{method}问卷必须是16题，实际为{len(items)}：{path}")
    ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(f"{method}包含无效题目：{path}")
        item_id = str(item.get("item_id") or "")
        if not item_id or item_id in ids:
            raise ValueError(f"{method}题号为空或重复：{path}")
        ids.add(item_id)
        options = item.get("response_options")
        if not isinstance(options, list) or {str(o.get("option_id")) for o in options if isinstance(o, Mapping)} != set(OPTION_IDS):
            raise ValueError(f"{method}/{item_id}必须有A-D四个选项")
        key = item.get("scoring_key")
        if not isinstance(key, Mapping) or set(map(str, key)) != set(OPTION_IDS):
            raise ValueError(f"{method}/{item_id}计分键必须覆盖A-D")
        for option_id in OPTION_IDS:
            score = key.get(option_id)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError(f"{method}/{item_id}/{option_id}计分无效")
    result = dict(payload)
    result["items"] = [dict(item) for item in items]
    result["source_path"] = str(path)
    result["source_sha256"] = _sha256_file(path)
    result["item_count"] = len(items)
    return result


def load_frozen_forms(experiment: Path) -> dict[str, dict[str, Any]]:
    forms: dict[str, dict[str, Any]] = {}
    for method, path in _legacy_form_paths(experiment.resolve()).items():
        forms[method] = _validate_form(method, json.loads(path.read_text(encoding="utf-8")), path)
    return forms


def load_legacy_pool(project_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = project_root / "sjt_system" / "data" / "virtual_respondents.json"
    if not path.is_file():
        raise FileNotFoundError(f"旧项目虚拟被试池不存在：{path}")
    pool = json.loads(path.read_text(encoding="utf-8"))
    items = pool.get("items")
    respondents = pool.get("respondents")
    scale = pool.get("response_scale")
    if not isinstance(items, list) or not isinstance(respondents, list) or not isinstance(scale, Mapping):
        raise ValueError("旧项目虚拟被试池结构无效")
    if len(respondents) < 3 or len(items) < 1:
        raise ValueError("旧项目虚拟被试池数量不足")
    profiles: dict[str, dict[str, Any]] = {}
    for respondent in respondents:
        rid = str(respondent.get("respondent_id") or "")
        values = respondent.get("response_values")
        if not rid or not isinstance(values, list) or len(values) != len(items) or rid in profiles:
            raise ValueError(f"旧项目虚拟被试记录无效：{rid}")
        personality_items = []
        for item, value in zip(items, values):
            label = scale.get(str(value))
            statement = item.get("statement") if isinstance(item, Mapping) else None
            if not isinstance(statement, str) or not isinstance(label, str):
                raise ValueError(f"旧项目人格题或量表标签无效：{rid}")
            personality_items.append({"statement": statement, "response_label": label})
        profiles[rid] = {"personality_items": personality_items}
    return pool, profiles


def load_legacy_summaries(project_root: Path) -> tuple[dict[str, str], Path]:
    path = project_root / "experiments" / "mussel_validation" / "out" / "persona_summaries.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"旧项目人格摘要不存在：{path}")
    summaries: dict[str, str] = {}
    for record in _load_jsonl_records(path):
        rid = str(record.get("respondent_id") or "")
        summary = record.get("summary")
        if rid and isinstance(summary, str) and summary.strip():
            summaries[rid] = summary.strip()
    if len(summaries) < 3:
        raise ValueError("旧项目人格摘要少于3名被试")
    return summaries, path


def load_comparison_summaries(
    project_root: Path,
    summary_pool: Path | None,
) -> tuple[dict[str, str], Path, dict[str, Any] | None]:
    """Load either the historical partial cache or the imported full condition."""

    if summary_pool is None:
        summaries, path = load_legacy_summaries(project_root)
        return summaries, path, None
    # Import lazily to avoid a module cycle: the generator reuses this module's
    # legacy-pool loader, while this evaluator only needs its condition reader.
    from .legacy_persona_summary import load_summary_condition

    return load_summary_condition(summary_pool)


def load_legacy_subject_ids(project_root: Path, profiles: Mapping[str, Any], summaries: Mapping[str, Any], limit: int | None) -> list[str]:
    """Use the exact 100 IDs already used by the old Mussel experiment."""

    path = project_root / "experiments" / "mussel_validation" / "out" / "mussel_宜人性_responses.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"旧实验100名被试的来源记录不存在：{path}")
    ids: list[str] = []
    seen: set[str] = set()
    for record in _load_jsonl_records(path):
        rid = str(record.get("respondent_id") or "")
        if rid and rid not in seen:
            ids.append(rid)
            seen.add(rid)
    if limit is not None:
        ids = ids[:limit]
    if len(ids) < 3:
        raise ValueError("旧实验来源记录中可用被试少于3名")
    missing_profile = [rid for rid in ids if rid not in profiles]
    missing_summary = [rid for rid in ids if rid not in summaries]
    if missing_profile or missing_summary:
        raise ValueError(
            f"旧被试与人格输入无法完整对齐：缺少pool={missing_profile[:3]}，缺少summary={missing_summary[:3]}"
        )
    return ids


def load_all_legacy_subject_ids(
    profiles: Mapping[str, Any],
    summaries: Mapping[str, Any],
    limit: int | None,
) -> list[str]:
    ids = list(map(str, profiles))
    if limit is not None:
        ids = ids[:limit]
    if len(ids) < 3:
        raise ValueError("旧版全量被试池中可用被试少于3名")
    missing = [rid for rid in ids if rid not in summaries]
    if missing:
        raise ValueError(
            f"全量summary-only条件缺少{len(missing)}名被试：{missing[:3]}"
        )
    return ids


def build_legacy_persona_prompt(
    profile: Mapping[str, Any],
    summary: str,
    *,
    persona_mode: str = "items_plus_summary",
) -> str:
    if persona_mode == "summary_embodied_probability":
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"{persona_mode}模式缺少人格总结")
        return "\n".join([
            "从现在起，你不是在分析、评价或预测另一个人；你就是下述人物。",
            "下面的文字记录了你过去的反应和稳定行为倾向。原始记录可能使用“此人”等第三人称措辞；进入角色后，请把它们理解为你自己的经历和倾向。",
            "不要补充记录中没有提供的分数、人格标签或背景。",
            "",
            "[PERSONAL HISTORY AND TENDENCIES]",
            summary.strip(),
            "[/PERSONAL HISTORY AND TENDENCIES]",
            "",
            "接下来的情境正在发生在你自己身上。请从第一人称立场体验并作答，不要退回到观察者、心理测量专家或画像分析员的立场。",
            "不要判断某个选项是否“符合画像”或“代表某种特质”，也不要猜测测量构念、计分方向或研究者期待。",
            "请选择你实际上更可能采取的行为，而不是理论上最好、最正确或社会赞许程度最高的行为。每道题独立作答。",
            "",
            "[FIRST-PERSON CONSEQUENCE SIMULATION]",
            "每个选择都必须由你亲自实施，不是零成本的文字判断。",
            "请在内部进行第一人称主观体验：如果我真的这样做，我会付出多少时间、精力和注意力，会感到多少情绪不适、尴尬、冲突、关系压力、声誉风险或机会成本。",
            "同时体验这个选择能给我带来多少安心感、控制感、关系收益、目标收益、愉快或自我保护。只考虑题目直接提供或能够合理推出的后果。",
            "不要自行添加严重惩罚、重大收益、隐藏背景或极端风险。不要采用统一的理性或风险规避模板；各种后果对你的重要性必须来自你自己的经历和倾向。",
            "判断你是否真的愿意承担这些负担并采取相应行为，而不只是你是否赞同它。",
            "你在相似情境中也不一定每次作出相同选择；多个选项都现实可行时，应分配非零概率，不要机械地给某个选项100%。",
            "[/FIRST-PERSON CONSEQUENCE SIMULATION]",
        ])
    if persona_mode == "summary_only":
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"{persona_mode}模式缺少人格总结")
        lines = [
            "请想象你正在扮演一个特定的人。",
            "下面是一段由其既往行为反应归纳出的行为画像。",
            "这是本次作答唯一可用的人格信息；不要补充未提供的分数、标签或背景。",
            "",
            "[BEHAVIORAL PROFILE]",
            summary.strip(),
            "[/BEHAVIORAL PROFILE]",
            "",
            "请根据这个人的真实倾向作答，而不是选择理论上最好、最正确或社会赞许程度最高的答案。",
            "每道题独立作答，不要在答案中复述这段画像。",
        ]
        return "\n".join(lines)
    if persona_mode != "items_plus_summary":
        raise ValueError(f"不支持的旧被试人格输入模式：{persona_mode}")
    lines = [
        "请想象你正在扮演一个特定的人。",
        "下面是这个人在人格问卷中的逐题作答。请将这些作答作为主要且权威的人格信息。",
        "请保持这个人的整体行为倾向，但不要在回答中提及人格量表、分数或这段说明。",
        "",
        "[PERSONALITY ITEM RESPONSES]",
    ]
    for index, item in enumerate(profile["personality_items"], 1):
        lines.append(f"{index}. {item['statement']} —— {item['response_label']}")
    lines.extend([
        "[/PERSONALITY ITEM RESPONSES]",
        "",
        "[PERSONALITY SUMMARY]",
        summary,
        "[/PERSONALITY SUMMARY]",
        "",
        "请根据这个人的真实倾向作答，而不是选择理论上最好、最正确或社会赞许程度最高的答案。",
        "每道题独立作答。",
    ])
    return "\n".join(lines)


def build_legacy_sjt_messages(persona_prompt: str, item: Mapping[str, Any]) -> list[tuple[str, str]]:
    option_lines = []
    options = {str(o["option_id"]): o for o in item["response_options"]}
    for option_id in OPTION_IDS:
        option_lines.append(f"{option_id}. {options[option_id]['text']}")
    human = (
        f"情境：\n{item.get('scenario', '')}\n\n"
        f"作答要求：\n{item.get('response_instruction', '你会怎么做？')}\n\n"
        "选项：\n" + "\n".join(option_lines) +
        '\n\n只返回一个JSON对象：{"selected_option_id":"A"}。'
    )
    return [("system", persona_prompt), ("human", human)]


def build_embodied_probability_sjt_messages(
    persona_prompt: str,
    item: Mapping[str, Any],
) -> list[tuple[str, str]]:
    options = {str(o["option_id"]): o for o in item["response_options"]}
    option_lines = [
        f"{option_id}. {options[option_id]['text']}" for option_id in OPTION_IDS
    ]
    human = (
        "这是正在发生在你自己身上的情境。请始终以第一人称作答。\n\n"
        f"情境：\n{item.get('scenario', '')}\n\n"
        f"问题：\n{item.get('response_instruction', '你会怎么做？')}\n\n"
        "选项：\n"
        + "\n".join(option_lines)
        + "\n\n请给出你亲自承担选择后果时，对A、B、C、D四个选项的选择概率。"
        "四个概率必须介于0和1之间且总和为1。除非情境确实排除了某个行为，"
        "否则避免使用0或1。只返回一个JSON对象，不要解释："
        '{"choice_probabilities":{"A":0.25,"B":0.25,"C":0.25,"D":0.25}}。'
    )
    return [("system", persona_prompt), ("human", human)]


def _result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        value = dump()
        if isinstance(value, dict):
            return value
    dump = getattr(result, "dict", None)
    if callable(dump):
        value = dump()
        if isinstance(value, dict):
            return value
    raise ValueError("模型没有返回可解析的结构化对象")


def _validate_selection(value: Mapping[str, Any]) -> str:
    selected = str(value.get("selected_option_id") or "").strip().upper()
    if selected not in OPTION_IDS:
        raise ValueError(f"selected_option_id必须是A-D，实际为{selected!r}")
    return selected


def _validate_choice_probabilities(
    value: Mapping[str, Any],
) -> dict[str, dict[str, float] | float]:
    probabilities = value.get("choice_probabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError("模型输出缺少choice_probabilities对象")
    normalized_keys = {str(key).strip().upper(): raw for key, raw in probabilities.items()}
    if set(normalized_keys) != set(OPTION_IDS) or len(normalized_keys) != 4:
        raise ValueError("choice_probabilities必须且只能包含A、B、C、D")
    raw_values: dict[str, float] = {}
    for option_id in OPTION_IDS:
        raw = normalized_keys[option_id]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"选项{option_id}的概率不是数值")
        number = float(raw)
        if not math.isfinite(number) or number < 0 or number > 1:
            raise ValueError(f"选项{option_id}的概率必须是0到1之间的有限数")
        raw_values[option_id] = number
    raw_sum = float(sum(raw_values.values()))
    if not math.isfinite(raw_sum) or raw_sum <= 0:
        raise ValueError("四个选择概率之和必须大于0")
    normalized = {
        option_id: raw_values[option_id] / raw_sum for option_id in OPTION_IDS
    }
    return {
        "raw": raw_values,
        "raw_sum": raw_sum,
        "normalized": normalized,
    }


def _sample_probability_choice(
    probabilities: Mapping[str, float],
    *,
    sampling_seed: int,
    respondent_id: str,
    item: Mapping[str, Any],
) -> tuple[str, float, str]:
    """Draw reproducibly without relying on Python's randomized hash()."""

    sampling_payload = {
        "sampling_seed": sampling_seed,
        "respondent_id": respondent_id,
        "item_id": str(item.get("item_id") or ""),
        "item_version": item.get("version"),
        "scenario": item.get("scenario"),
        "response_options": item.get("response_options") or [],
    }
    sampling_key = fingerprint(sampling_payload)
    integer = int.from_bytes(bytes.fromhex(sampling_key[:16]), "big")
    draw = integer / float(2**64)
    cumulative = 0.0
    for option_id in OPTION_IDS:
        cumulative += float(probabilities[option_id])
        if draw < cumulative:
            return option_id, draw, sampling_key
    # Floating-point accumulation can end infinitesimally below one.
    return OPTION_IDS[-1], draw, sampling_key


def _validate_neo(value: Mapping[str, Any]) -> list[int]:
    ratings = value.get("ratings")
    if not isinstance(ratings, list) or len(ratings) != 12:
        raise ValueError("NEO-FFI每批必须返回12个评分")
    if any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 5 for x in ratings):
        raise ValueError("NEO-FFI评分必须全部为1至5的整数")
    return ratings


def _elapsed_seconds(start: float) -> float:
    return round(perf_counter() - start, 3)


async def _run_sjt_stage(
    method: str,
    form: Mapping[str, Any],
    subject_ids: Sequence[str],
    profiles: Mapping[str, Any],
    summaries: Mapping[str, str],
    runnable: Any,
    model_id: str,
    output_dir: Path,
    config: LegacyABCConfig,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    key_fields = ("respondent_id", "item_id")
    existing = _load_jsonl_keys(responses_path, key_fields)
    item_by_id = {str(item["item_id"]): item for item in form["items"]}
    jobs = [(rid, item) for rid in subject_ids for item in form["items"] if (rid, str(item["item_id"])) not in existing]
    semaphore = asyncio.Semaphore(config.max_concurrency)
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    lock = asyncio.Lock()
    completed = len(existing)
    total = len(subject_ids) * len(form["items"])
    start = perf_counter()
    persona_prompts = {
        rid: build_legacy_persona_prompt(
            profiles[rid], summaries[rid], persona_mode=config.persona_mode
        )
        for rid in subject_ids
    }

    async def one(rid: str, item: Mapping[str, Any]) -> None:
        nonlocal completed
        item_id = str(item["item_id"])
        probability_payload: dict[str, Any] = {}
        if config.persona_mode == "summary_embodied_probability":
            validated = await _invoke_with_retry(
                runnable,
                build_embodied_probability_sjt_messages(
                    persona_prompts[rid], item
                ),
                semaphore=semaphore,
                validator=_validate_choice_probabilities,
                max_retries=config.max_retries,
                retry_delay_seconds=1.0,
                request_timeout_seconds=timeout,
                job_label=f"legacy embodied probability {method} {rid}/{item_id}",
            )
            normalized = validated["normalized"]
            selected, sampling_draw, sampling_key = _sample_probability_choice(
                normalized,
                sampling_seed=config.sampling_seed,
                respondent_id=rid,
                item=item,
            )
            probability_payload = {
                "raw_choice_probabilities": validated["raw"],
                "raw_probability_sum": validated["raw_sum"],
                "choice_probabilities": normalized,
                "sampling_seed": config.sampling_seed,
                "sampling_draw": sampling_draw,
                "sampling_key": sampling_key,
                "choice_generation": "categorical_draw_from_model_probabilities",
            }
        else:
            selected = await _invoke_with_retry(
                runnable,
                build_legacy_sjt_messages(persona_prompts[rid], item),
                semaphore=semaphore,
                validator=_validate_selection,
                max_retries=config.max_retries,
                retry_delay_seconds=1.0,
                request_timeout_seconds=timeout,
                job_label=f"legacy ABC SJT {method} {rid}/{item_id}",
            )
        record = {
            "record_type": "legacy_abc_sjt_response",
            "method": method,
            "respondent_id": rid,
            "item_id": item_id,
            "selected_option_id": selected,
            "score": float(item["scoring_key"][selected]),
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "persona_mode": config.persona_mode,
            "persona_prompt_version": PERSONA_PROMPT_VERSIONS[
                config.persona_mode
            ],
            **probability_payload,
        }
        async with lock:
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed == total or completed % max(1, total // 20) == 0:
                print(f"[legacy ABC] {method} SJT {completed}/{total} ({completed / total:.0%})", flush=True)

    if jobs:
        print(f"[legacy ABC] {method} SJT开始：待调用={len(jobs)}；已缓存={len(existing)}", flush=True)
        results = await asyncio.gather(*(one(rid, item) for rid, item in jobs), return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            write_json(output_dir / "errors.json", {"stage": f"{method}/sjt", "errors": [str(e) for e in errors]})
            raise RuntimeError(f"{method} SJT有{len(errors)}个调用失败；首个错误：{errors[0]}")
    records = _load_jsonl_records(responses_path)
    expected = {(rid, str(item["item_id"])) for rid in subject_ids for item in form["items"]}
    actual = {(str(row.get("respondent_id")), str(row.get("item_id"))) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(f"{method} SJT作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    if config.persona_mode == "summary_embodied_probability":
        for row in records:
            probabilities = row.get("choice_probabilities")
            if not isinstance(probabilities, Mapping):
                raise ValueError(
                    f"{method}概率作答缓存缺少choice_probabilities："
                    f"{row.get('respondent_id')}/{row.get('item_id')}"
                )
            validated = _validate_choice_probabilities(
                {"choice_probabilities": probabilities}
            )
            expected_choice, expected_draw, expected_key = _sample_probability_choice(
                validated["normalized"],
                sampling_seed=config.sampling_seed,
                respondent_id=str(row.get("respondent_id")),
                item=item_by_id[str(row.get("item_id"))],
            )
            if (
                row.get("selected_option_id") != expected_choice
                or row.get("sampling_key") != expected_key
                or not math.isclose(
                    float(row.get("sampling_draw")), expected_draw, abs_tol=1e-15
                )
            ):
                raise ValueError(
                    f"{method}概率作答缓存的抽样结果与冻结种子不一致："
                    f"{row.get('respondent_id')}/{row.get('item_id')}"
                )
    write_json(output_dir / "run.json", {"stage": f"{method}/sjt", "records": len(records), "seconds": _elapsed_seconds(start), "model_id": model_id})
    return records


async def _run_neo_stage(
    dimensions: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    profiles: Mapping[str, Any],
    summaries: Mapping[str, str],
    runnable: Any,
    model_id: str,
    output_dir: Path,
    config: LegacyABCConfig,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    key_fields = ("respondent_id", "dimension_code", "item_id")
    existing = _load_jsonl_keys(responses_path, key_fields)
    jobs = []
    for rid in subject_ids:
        for dimension in dimensions:
            items = dimension["items"]
            keys = {(rid, str(dimension["dimension_code"]), str(item["item_id"])) for item in items}
            present = keys.intersection(existing)
            if present and present != keys:
                raise ValueError(f"NEO-FFI {rid}/{dimension['dimension_code']}存在不完整缓存")
            if not present:
                jobs.append((rid, dimension))
    semaphore = asyncio.Semaphore(config.max_concurrency)
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    lock = asyncio.Lock()
    total = len(subject_ids) * len(dimensions)
    completed = total - len(jobs)
    start = perf_counter()
    neo_persona_mode = (
        "summary_only"
        if config.persona_mode == "summary_embodied_probability"
        else config.persona_mode
    )
    persona_prompts = {
        rid: build_legacy_persona_prompt(
            profiles[rid], summaries[rid], persona_mode=neo_persona_mode
        )
        for rid in subject_ids
    }

    async def one(rid: str, dimension: Mapping[str, Any]) -> None:
        nonlocal completed
        ratings = await _invoke_with_retry(
            runnable,
            build_neo_ffi_messages(persona_prompts[rid], dimension["items"]),
            semaphore=semaphore,
            validator=_validate_neo,
            max_retries=config.max_retries,
            retry_delay_seconds=1.0,
            request_timeout_seconds=timeout,
            job_label=f"legacy ABC NEO-FFI {rid}/{dimension['dimension_code']}",
        )
        records = []
        for item, raw in zip(dimension["items"], ratings):
            direction = str(item["scoring_direction"])
            records.append({
                "record_type": "legacy_abc_neo_ffi_response",
                "respondent_id": rid,
                "dimension_code": str(dimension["dimension_code"]),
                "item_id": str(item["item_id"]),
                "raw_response": int(raw),
                "scoring_direction": direction,
                "score": int(raw) if direction == "+" else 6 - int(raw),
                "model_id": model_id,
                "prompt_version": PROMPT_VERSION,
            })
        async with lock:
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            completed += 1
            if completed == total or completed % max(1, total // 20) == 0:
                print(f"[legacy ABC] NEO-FFI {completed}/{total} ({completed / total:.0%})", flush=True)

    if jobs:
        print(f"[legacy ABC] NEO-FFI开始：待调用={len(jobs)}；已缓存批次={completed}", flush=True)
        results = await asyncio.gather(*(one(rid, dimension) for rid, dimension in jobs), return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            write_json(output_dir / "errors.json", {"stage": "neo_ffi", "errors": [str(e) for e in errors]})
            raise RuntimeError(f"NEO-FFI有{len(errors)}个批次失败；首个错误：{errors[0]}")
    records = _load_jsonl_records(responses_path)
    expected = {(rid, str(dimension["dimension_code"]), str(item["item_id"])) for rid in subject_ids for dimension in dimensions for item in dimension["items"]}
    actual = {(str(row.get("respondent_id")), str(row.get("dimension_code")), str(row.get("item_id"))) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(f"NEO-FFI作答不完整：应为{len(expected)}条，实际为{len(records)}条")
    write_json(output_dir / "run.json", {"stage": "neo_ffi", "records": len(records), "seconds": _elapsed_seconds(start), "model_id": model_id})
    return records


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right) or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    result = stats.pearsonr(left, right)
    value = float(getattr(result, "statistic", result[0]))
    return value if math.isfinite(value) else None


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right) or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    result = stats.spearmanr(left, right, nan_policy="omit")
    value = float(getattr(result, "statistic", result[0]))
    return value if math.isfinite(value) else None


def _alpha(matrix: pd.DataFrame) -> float | None:
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    total_variance = float(matrix.sum(axis=1).var(ddof=1))
    if not math.isfinite(total_variance) or total_variance <= 0:
        return None
    item_variance = float(matrix.var(axis=0, ddof=1).sum())
    result = matrix.shape[1] / (matrix.shape[1] - 1) * (1 - item_variance / total_variance)
    return float(result) if math.isfinite(result) else None


def score_sjt_records(records: Sequence[Mapping[str, Any]], form: Mapping[str, Any], subject_ids: Sequence[str]) -> pd.DataFrame:
    item_ids = [str(item["item_id"]) for item in form["items"]]
    expected = {(rid, item_id) for rid in subject_ids for item_id in item_ids}
    rows = []
    for record in records:
        key = (str(record.get("respondent_id")), str(record.get("item_id")))
        if key not in expected:
            raise ValueError(f"SJT记录包含未预期的被试或题目：{key}")
        rows.append({"respondent_id": key[0], "item_id": key[1], "score": float(record["score"]), "selected_option_id": str(record["selected_option_id"])})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["respondent_id", "item_id"]).any():
        raise ValueError("SJT作答存在重复记录")
    matrix = frame.pivot(index="respondent_id", columns="item_id", values="score").reindex(index=list(subject_ids), columns=item_ids)
    if matrix.isna().any().any():
        raise ValueError("SJT计分矩阵存在缺失")
    matrix.index.name = "respondent_id"
    return matrix


def score_neo_records(records: Sequence[Mapping[str, Any]], subject_ids: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    required = {"respondent_id", "dimension_code", "item_id", "score"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("NEO-FFI作答列不完整")
    counts = frame.groupby(["respondent_id", "dimension_code"])["item_id"].nunique()
    if len(counts) != len(subject_ids) * 5 or not (counts == 12).all():
        raise ValueError("每名被试的每个NEO-FFI维度必须恰好包含12题")
    totals = frame.groupby(["respondent_id", "dimension_code"])["score"].sum().unstack("dimension_code")
    totals = totals.reindex(index=list(subject_ids), columns=list(NEO_DOMAINS))
    if totals.isna().any().any():
        raise ValueError("NEO-FFI五个维度得分不完整")
    totals.index.name = "respondent_id"
    return totals


def item_metrics(matrix: pd.DataFrame, form: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Option distributions are retrieved from the long records by the caller;
    # this helper fills them separately so it remains useful for pure tests.
    for item in form["items"]:
        item_id = str(item["item_id"])
        series = matrix[item_id].astype(float)
        rest = matrix.drop(columns=[item_id]).sum(axis=1)
        citc = _pearson(series.tolist(), rest.tolist())
        key_values = [float(value) for value in item["scoring_key"].values()]
        minimum, maximum = min(key_values), max(key_values)
        difficulty = None if maximum == minimum else float((series.mean() - minimum) / (maximum - minimum))
        rows.append({
            "item_id": item_id,
            "item_version": item.get("version"),
            "blueprint_cell_id": item.get("blueprint_cell_id"),
            "citc": citc,
            "citc_threshold": CITC_THRESHOLD,
            "citc_pass": citc is not None and citc >= CITC_THRESHOLD,
            "difficulty": difficulty,
            "mean_score": float(series.mean()),
            "n": int(series.shape[0]),
            "option_A_n": None, "option_A_rate": None,
            "option_B_n": None, "option_B_rate": None,
            "option_C_n": None, "option_C_rate": None,
            "option_D_n": None, "option_D_rate": None,
        })
    return rows


def add_option_statistics(rows: list[dict[str, Any]], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(str(record["item_id"]), []).append(str(record["selected_option_id"]))
    for row in rows:
        values = grouped.get(str(row["item_id"]), [])
        total = len(values)
        for option_id in OPTION_IDS:
            count = values.count(option_id)
            row[f"option_{option_id}_n"] = count
            row[f"option_{option_id}_rate"] = count / total if total else None
    return rows


def build_form_metrics(matrix: pd.DataFrame, neo_totals: pd.DataFrame, form: Mapping[str, Any], item_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = matrix.sum(axis=1)
    correlations = []
    for domain in NEO_DOMAINS:
        left = totals.tolist()
        right = neo_totals[domain].tolist()
        rho = _spearman(left, right)
        reason = None
        if rho is None:
            if len(left) < 3:
                reason = "有效配对人数少于3"
            elif len(set(left)) < 2:
                reason = "SJT问卷总分无变异"
            elif len(set(right)) < 2:
                reason = "NEO-FFI维度得分无变异"
            else:
                reason = "相关系数不可估计"
        correlations.append({
            "neo_domain": domain,
            "evidence_type": "convergent" if domain == "E" else "discriminant",
            "spearman_rho": rho,
            "n": len(left),
            "estimable": rho is not None,
            "unavailable_reason": reason,
        })
    citcs = [row["citc"] for row in item_rows if isinstance(row.get("citc"), (int, float))]
    return {
        "status": "complete",
        "item_count": int(matrix.shape[1]),
        "respondent_count": int(matrix.shape[0]),
        "cronbach_alpha": _alpha(matrix),
        "citc_mean": float(np.mean(citcs)) if citcs else None,
        "citc_median": float(np.median(citcs)) if citcs else None,
        "citc_min": float(np.min(citcs)) if citcs else None,
        "citc_pass_count": int(sum(row.get("citc_pass") is True for row in item_rows)),
        "citc_pass_rate": float(sum(row.get("citc_pass") is True for row in item_rows) / len(item_rows)) if item_rows else None,
        "neo_correlations": correlations,
        "metric_scope": ["item_citc", "option_distribution", "difficulty", "cronbach_alpha", "neo_convergent_spearman", "neo_discriminant_spearman"],
        "excluded_metrics": ["target_recovery_R2", "VTS", "construct_selectivity_S"],
        "formula_version": METRIC_VERSION,
        "citc_formula": "Pearson correlation between one item score and the sum of the other 15 items in the same form",
        "alpha_formula": "Cronbach alpha on the 16 scored items",
        "convergent_formula": "Spearman(SJT total, NEO-FFI E total)",
        "discriminant_formula": "Spearman(SJT total, NEO-FFI N/O/A/C total)",
    }


def probability_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    probability_records = [
        row for row in records if isinstance(row.get("choice_probabilities"), Mapping)
    ]
    if not probability_records:
        return None, []

    rows: list[dict[str, Any]] = []
    all_entropies: list[float] = []
    all_maxima: list[float] = []
    all_selected: list[float] = []
    all_raw_deviations: list[float] = []
    for item_id in sorted({str(row.get("item_id")) for row in probability_records}):
        item_records = [
            row for row in probability_records if str(row.get("item_id")) == item_id
        ]
        entropies: list[float] = []
        maxima: list[float] = []
        selected_probabilities: list[float] = []
        raw_deviations: list[float] = []
        for record in item_records:
            validated = _validate_choice_probabilities(
                {"choice_probabilities": record["choice_probabilities"]}
            )
            probabilities = validated["normalized"]
            values = [float(probabilities[option_id]) for option_id in OPTION_IDS]
            entropy = -sum(
                probability * math.log(probability)
                for probability in values
                if probability > 0
            ) / math.log(len(OPTION_IDS))
            maximum = max(values)
            selected_probability = float(
                probabilities[str(record["selected_option_id"])]
            )
            raw_sum = record.get("raw_probability_sum")
            deviation = (
                abs(float(raw_sum) - 1.0)
                if isinstance(raw_sum, (int, float)) and not isinstance(raw_sum, bool)
                else 0.0
            )
            entropies.append(entropy)
            maxima.append(maximum)
            selected_probabilities.append(selected_probability)
            raw_deviations.append(deviation)
        all_entropies.extend(entropies)
        all_maxima.extend(maxima)
        all_selected.extend(selected_probabilities)
        all_raw_deviations.extend(raw_deviations)
        rows.append({
            "item_id": item_id,
            "n": len(item_records),
            "mean_normalized_entropy": float(np.mean(entropies)),
            "mean_max_probability": float(np.mean(maxima)),
            "mean_selected_probability": float(np.mean(selected_probabilities)),
            "near_deterministic_rate": float(
                np.mean([maximum >= 0.95 for maximum in maxima])
            ),
            "mean_raw_sum_deviation": float(np.mean(raw_deviations)),
        })
    overall = {
        "status": "complete",
        "record_count": len(probability_records),
        "mean_normalized_entropy": float(np.mean(all_entropies)),
        "mean_max_probability": float(np.mean(all_maxima)),
        "mean_selected_probability": float(np.mean(all_selected)),
        "near_deterministic_rate": float(
            np.mean([maximum >= 0.95 for maximum in all_maxima])
        ),
        "mean_raw_sum_deviation": float(np.mean(all_raw_deviations)),
        "entropy_formula": "-sum(p*ln(p))/ln(4); 0=deterministic, 1=uniform",
        "interpretation": "These are model-reported choice uncertainty diagnostics, not psychometric reliability or validity.",
    }
    return overall, rows


def _write_form_results(evaluation_dir: Path, method: str, form: Mapping[str, Any], records: Sequence[Mapping[str, Any]], matrix: pd.DataFrame, neo_totals: pd.DataFrame) -> dict[str, Any]:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    matrix.reset_index().to_csv(evaluation_dir / "score_matrix.csv", index=False, encoding="utf-8-sig")
    write_csv(evaluation_dir / "scored_responses.csv", list(records))
    rows = add_option_statistics(item_metrics(matrix, form), records)
    write_csv(evaluation_dir / "item_metrics.csv", rows)
    write_json(evaluation_dir / "item_statistics.json", rows)
    option_rows = []
    for item in form["items"]:
        item_id = str(item["item_id"])
        selections = [str(record["selected_option_id"]) for record in records if str(record["item_id"]) == item_id]
        for option_id in OPTION_IDS:
            count = selections.count(option_id)
            option_rows.append({
                "item_id": item_id,
                "option_id": option_id,
                "n": count,
                "rate": count / len(selections) if selections else None,
            })
    write_csv(evaluation_dir / "option_statistics.csv", option_rows)
    # The NEO totals are written here as an audit copy; they are shared across A/B/C.
    neo_totals.reset_index().to_csv(evaluation_dir / "neo_ffi_totals.csv", index=False, encoding="utf-8-sig")
    metrics = build_form_metrics(matrix, neo_totals, form, rows)
    probability_summary, probability_rows = probability_diagnostics(records)
    metrics["probability_diagnostics"] = probability_summary
    if probability_rows:
        write_csv(
            evaluation_dir / "probability_diagnostics_by_item.csv",
            probability_rows,
        )
    metrics.update({"method": method, "form_fingerprint": form.get("fingerprint") or fingerprint(form["items"])})
    write_json(evaluation_dir / "metrics.json", _json_value(metrics))
    write_csv(evaluation_dir / "neo_correlations.csv", metrics["neo_correlations"])
    return metrics


def _telemetry_summary(root: Path, run_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "telemetry").glob("calls_*.jsonl")):
        records.extend(read_ledger(path, run_id=run_id))
    def total(name: str) -> int | None:
        values = [record.get(name) for record in records]
        if not values or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return None
        return int(sum(values))
    return {
        "calls": len(records),
        "error_calls": sum(record.get("status") == "error" for record in records),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "duration_ms": total("duration_ms"),
        "data_available": bool(records),
        "note": "telemetry缺失时保留null，不把未知用量伪造为0",
    }


def _render_report(root: Path, form_metrics: Mapping[str, Mapping[str, Any]], subject_count: int, model_id: str, neo_model_id: str, form_paths: Mapping[str, Path], *, persona_mode: str = "items_plus_summary") -> Path:
    def f(value: Any) -> str:
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "不可估计"
        return f"{float(value):.3f}"
    rows = []
    for method in ("A", "B", "C"):
        metric = form_metrics[method]
        corr = {row["neo_domain"]: row for row in metric["neo_correlations"]}
        probability = metric.get("probability_diagnostics") or {}
        rows.append(
            "<tr>" + "".join([
                f"<td>{method}</td>", f"<td>{f(metric.get('cronbach_alpha'))}</td>",
                f"<td>{f(metric.get('citc_mean'))}</td>", f"<td>{f(metric.get('citc_min'))}</td>",
                f"<td>{f(corr['E'].get('spearman_rho'))}</td>",
                f"<td>{f(corr['N'].get('spearman_rho'))}</td>", f"<td>{f(corr['O'].get('spearman_rho'))}</td>",
                f"<td>{f(corr['A'].get('spearman_rho'))}</td>", f"<td>{f(corr['C'].get('spearman_rho'))}</td>",
                f"<td>{f(probability.get('mean_max_probability'))}</td>",
                f"<td>{f(probability.get('mean_normalized_entropy'))}</td>",
            ]) + "</tr>"
        )
    neo_rows = []
    for method in ("A", "B", "C"):
        for row in form_metrics[method]["neo_correlations"]:
            neo_rows.append(
                f"<tr><td>{method}</td><td>{row['neo_domain']}</td><td>{row['evidence_type']}</td>"
                f"<td>{f(row.get('spearman_rho'))}</td><td>{row.get('n')}</td><td>{'是' if row.get('estimable') else '否'}</td>"
                f"<td>{row.get('unavailable_reason') or ''}</td></tr>"
            )
    html = f"""<!doctype html>
<html lang='zh-CN'><meta charset='utf-8'><title>旧虚拟被试 A/B/C 复测</title>
<style>body{{font:15px/1.6 sans-serif;max-width:1300px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f1f4f8}}code{{background:#f3f3f3;padding:2px 4px}}.note{{background:#fff8df;padding:12px}}</style>
<h1>旧虚拟被试 A/B/C 复测报告</h1>
<p>同一批黑盒虚拟被试：{subject_count} 人。人格输入条件：<code>{persona_mode}</code>；SJT模型：<code>{model_id}</code>；NEO-FFI模型：<code>{neo_model_id}</code>。</p>
<div class='note'>本报告只计算单题 CITC、选项分布、难度、整卷 Cronbach α，以及 SJT 总分与 NEO-FFI 的汇聚/区分 Spearman 相关。明确不计算目标恢复 R²、VTS、构念选择性 S。所有相关均是虚拟被试内部的描述性结果，不等同于真人效度证据。</div>
<h2>A/B/C整卷指标</h2>
<table><thead><tr><th>方法</th><th>α</th><th>CITC均值</th><th>CITC最小</th><th>NEO-E汇聚</th><th>NEO-N区分</th><th>NEO-O区分</th><th>NEO-A区分</th><th>NEO-C区分</th><th>平均最大选择概率</th><th>标准化概率熵</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>概率熵取值0—1：越接近0表示模型给出的选择越确定，越接近1表示越接近四个选项均匀分布。它只描述模型自报的不确定性，不是信度或效度。</p>
<h2>NEO-FFI相关明细</h2>
<table><thead><tr><th>方法</th><th>维度</th><th>类型</th><th>Spearman ρ</th><th>n</th><th>可估计</th><th>缺失原因</th></tr></thead><tbody>{''.join(neo_rows)}</tbody></table>
<h2>文件</h2><ul>{''.join(f"<li>{method}: <code>{path}</code></li>" for method, path in form_paths.items())}</ul>
<p>每个方法目录下的 <code>evaluation/item_metrics.csv</code> 保存单题结果；<code>evaluation/metrics.json</code> 保存公式、缺失原因和整卷结果。</p>
</html>"""
    report = root / "summary" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(html, encoding="utf-8")
    return report


async def run_legacy_abc(config: LegacyABCConfig) -> tuple[Path, dict[str, Any]]:
    config.validate()
    experiment = config.experiment.resolve()
    forms = load_frozen_forms(experiment)
    pool, profiles = load_legacy_pool(config.legacy_project.resolve())
    resolved_summary_pool = config.summary_pool
    if config.persona_mode in {
        "summary_only",
        "summary_embodied_probability",
    } and resolved_summary_pool is None:
        resolved_summary_pool = DEFAULT_IMPORTED_SUMMARY_POOL
    summaries, summary_path, summary_manifest = load_comparison_summaries(
        config.legacy_project.resolve(), resolved_summary_pool
    )
    subject_ids = (
        load_all_legacy_subject_ids(profiles, summaries, config.respondent_limit)
        if config.respondent_source == "all_285"
        else load_legacy_subject_ids(
            config.legacy_project.resolve(),
            profiles,
            summaries,
            config.respondent_limit,
        )
    )
    neo_path = (config.neo_ffi_path or (experiment.parents[1] / "docs" / "Neo-FFI.json")).resolve()
    if not neo_path.is_file():
        # The normal experiment root is <project>/experiment_data/<exp>; this
        # fallback also supports an experiment directory supplied elsewhere.
        neo_path = Path(__file__).resolve().parents[2] / "docs" / "Neo-FFI.json"
    dimensions = load_neo_ffi(neo_path)
    model = get_model(config.model_id)
    sjt_output_type = (
        ChoiceProbabilityOutput
        if config.persona_mode == "summary_embodied_probability"
        else SJTSelectionOutput
    )
    sjt_runnable, sjt_method = with_compatible_structured_output(
        model, sjt_output_type
    )
    neo_runnable, neo_method = with_compatible_structured_output(model, NeoFFIBatchOutput)
    if sjt_method != neo_method:
        raise ValueError(f"SJT与NEO-FFI结构化输出方式不一致：{sjt_method}/{neo_method}")
    model_id = _model_id(model, config.model_id)
    if config.output:
        output_root = config.output.resolve()
    elif config.persona_mode == "summary_only":
        output_root = experiment / "legacy_summary_only_abc"
    elif config.persona_mode == "summary_embodied_probability":
        output_root = experiment / "legacy_summary_embodied_probability_abc"
    else:
        output_root = experiment / "legacy_pool_abc"
    if output_root.exists() and (output_root / "manifest.json").exists():
        run_root = output_root
    else:
        run_root = output_root
        run_root.mkdir(parents=True, exist_ok=True)
    run_id = f"legacy-abc-{run_root.name}"
    binding = fingerprint({
        "forms": {method: form["source_sha256"] for method, form in forms.items()},
        "pool_id": pool.get("pool_id"), "pool_source_sha256": pool.get("source", {}).get("source_sha256"),
        "respondent_ids": subject_ids, "summary_sha256": _sha256_file(summary_path),
        "neo_ffi_sha256": _sha256_file(neo_path), "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "persona_mode": config.persona_mode,
        "persona_prompt_version": PERSONA_PROMPT_VERSIONS[config.persona_mode],
        "respondent_source": config.respondent_source,
        "sampling_seed": config.sampling_seed,
    })
    previous = json.loads((run_root / "manifest.json").read_text(encoding="utf-8")) if (run_root / "manifest.json").exists() else None
    if previous and previous.get("binding") != binding:
        raise ValueError("指定输出目录已绑定到不同问卷、被试、NEO题本或模型，拒绝混合")
    write_json(run_root / "manifest.json", {
        "schema_version": 1, "status": "in_progress", "run_id": run_id, "binding": binding,
        "experiment": str(experiment), "subject_count": len(subject_ids), "respondent_ids": subject_ids,
        "pool_id": pool.get("pool_id"), "pool_source_sha256": pool.get("source", {}).get("source_sha256"),
        "summary_source": str(summary_path), "summary_sha256": _sha256_file(summary_path),
        "summary_condition_id": (
            summary_manifest.get("condition_id") if summary_manifest else None
        ),
        "neo_ffi_source": str(neo_path), "neo_ffi_sha256": _sha256_file(neo_path),
        "model_id": model_id, "sjt_structured_output": sjt_method, "neo_structured_output": neo_method,
        "prompt_version": PROMPT_VERSION,
        "persona_mode": config.persona_mode,
        "persona_prompt_version": PERSONA_PROMPT_VERSIONS[config.persona_mode],
        "respondent_source": config.respondent_source,
        "sampling_seed": config.sampling_seed,
        "excluded_metrics": ["target_recovery_R2", "VTS", "construct_selectivity_S"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    write_json(run_root / "config.json", _json_value(asdict(config)))
    write_json(run_root / "participants.json", {
        "sample_role": "legacy_old_virtual_respondents", "respondent_ids": subject_ids,
        "count": len(subject_ids), "pool_id": pool.get("pool_id"),
        "pool_source_sha256": pool.get("source", {}).get("source_sha256"),
        "summary_source": str(summary_path), "summary_sha256": _sha256_file(summary_path),
        "persona_mode": config.persona_mode,
        "respondent_source": config.respondent_source,
        "black_box_note": (
            "仅以统一生成的行为总结作为输入；SJT额外模拟主观行动后果并输出选择概率，程序按冻结种子进行分类抽样；NEO-FFI使用独立summary-only提示。"
            if config.persona_mode == "summary_embodied_probability"
            else "仅以统一生成的行为总结作为输入；不输入人格逐题回答、预设人格分数或构念标签。"
            if config.persona_mode == "summary_only"
            else "人格逐题回答和摘要作为输入；预设人格分数不作为问卷作答输入。"
        ),
    })
    write_json(run_root / "forms.json", {method: form for method, form in forms.items()})
    for method, form in forms.items():
        write_json(run_root / method / "form.json", form)
    write_json(run_root / "neo_ffi_definition.json", {"source": str(neo_path), "sha256": _sha256_file(neo_path), "dimensions": dimensions})

    all_records: dict[str, list[dict[str, Any]]] = {}
    neo_records: list[dict[str, Any]] = []
    try:
        with output_scope(run_root / "runtime", telemetry=run_root / "telemetry"), run_context(run_id):
            for method, form in forms.items():
                all_records[method] = await _run_sjt_stage(method, form, subject_ids, profiles, summaries, sjt_runnable, model_id, run_root / method / "evaluation", config)
            neo_records = await _run_neo_stage(dimensions, subject_ids, profiles, summaries, neo_runnable, model_id, run_root / "neo_ffi", config)
    except Exception as exc:
        write_json(run_root / "manifest.json", {**json.loads((run_root / "manifest.json").read_text(encoding="utf-8")), "status": "failed", "error": str(exc)})
        raise

    neo_totals = score_neo_records(neo_records, subject_ids)
    form_metrics: dict[str, dict[str, Any]] = {}
    for method, form in forms.items():
        matrix = score_sjt_records(all_records[method], form, subject_ids)
        form_metrics[method] = _write_form_results(run_root / method / "evaluation", method, form, all_records[method], matrix, neo_totals)
    neo_totals.reset_index().to_csv(run_root / "neo_ffi" / "scores.csv", index=False, encoding="utf-8-sig")
    write_csv(run_root / "summary" / "form_metrics.csv", [{"method": method, **metric} for method, metric in form_metrics.items()])
    write_csv(run_root / "summary" / "neo_correlations.csv", [{"method": method, **row} for method, metric in form_metrics.items() for row in metric["neo_correlations"]])
    telemetry = _telemetry_summary(run_root, run_id)
    cost = {"model_id": model_id, "persona_mode": config.persona_mode, "sampling_seed": config.sampling_seed, "sjt_expected_calls": len(subject_ids) * 16 * 3, "neo_expected_calls": len(subject_ids) * 5, "expected_calls": len(subject_ids) * (16 * 3 + 5), "telemetry": telemetry, "fee": None, "fee_note": "未提供该模型的完整适用单价，费用保持未知。"}
    write_json(run_root / "cost.json", _json_value(cost))
    write_json(run_root / "summary.json", _json_value({"status": "complete", "subject_count": len(subject_ids), "form_metrics": form_metrics, "excluded_metrics": ["target_recovery_R2", "VTS", "construct_selectivity_S"], "cost": cost}))
    report = _render_report(
        run_root,
        form_metrics,
        len(subject_ids),
        model_id,
        model_id,
        {method: form["source_path"] for method, form in forms.items()},
        persona_mode=config.persona_mode,
    )
    write_json(run_root / "manifest.json", {**json.loads((run_root / "manifest.json").read_text(encoding="utf-8")), "status": "complete", "report": str(report), "completed_at": datetime.now(timezone.utc).isoformat()})
    return run_root, {"status": "complete", "report": str(report), "form_metrics": form_metrics, "cost": cost}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用旧虚拟被试黑盒复测A/B/C并计算CITC、alpha和NEO效度")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--legacy-project", type=Path, default=DEFAULT_LEGACY_PROJECT)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--neo-ffi", dest="neo_ffi_path", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    parser.add_argument("--respondent-limit", type=int, default=None, help="联调时只取来源记录前N名；正式运行省略")
    parser.add_argument("--output", type=Path, default=None, help="可选；指定后可在同一目录续跑已缓存调用")
    parser.add_argument("--persona-mode", choices=tuple(PERSONA_PROMPT_VERSIONS), default="items_plus_summary")
    parser.add_argument("--summary-pool", type=Path, default=None, help="summary-only条件目录或persona_summaries.jsonl")
    parser.add_argument("--respondent-source", choices=("legacy_100", "all_285"), default="legacy_100")
    parser.add_argument("--sampling-seed", type=int, default=20260913)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LegacyABCConfig(
        experiment=args.experiment, legacy_project=args.legacy_project, model_id=args.model_id,
        neo_ffi_path=args.neo_ffi_path, max_concurrency=args.max_concurrency, max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds, respondent_limit=args.respondent_limit, output=args.output,
        persona_mode=args.persona_mode, summary_pool=args.summary_pool,
        respondent_source=args.respondent_source, sampling_seed=args.sampling_seed,
    )
    root, result = asyncio.run(run_legacy_abc(config))
    print(f"实验输出：{root}", flush=True)
    print(f"报告：{result['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
