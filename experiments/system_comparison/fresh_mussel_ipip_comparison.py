"""Fresh full comparison of embodied and explicit-score virtual respondents.

This experiment deliberately does not read prior Mussel or IPIP response files.
It uses only the source persona pool, persona summaries, and frozen questionnaire
definitions, then administers both questionnaires anew under both conditions.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.authoring.construct_registry import construct_selection_catalog
from sjt_system.evaluation.reference_questionnaires import (
    load_mussel_items,
    score_mussel_records,
)
from sjt_system.evaluation.respondents import (
    build_score_dimension_catalog,
    generate_matched_condition_respondent_refs,
    normalize_matched_conditions,
)
from sjt_system.evaluation.simulation import (
    NeoFFIBatchOutput,
    SJTSelectionOutput,
    build_persona_prompt,
    _invoke_with_retry,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import read_ledger, run_context

from .legacy_pool_abc import (
    ChoiceProbabilityOutput,
    DEFAULT_IMPORTED_SUMMARY_POOL,
    OPTION_IDS,
    _alpha,
    _json_value,
    _model_id,
    _pearson,
    _sample_probability_choice,
    _sha256_file,
    _spearman,
    _validate_choice_probabilities,
    _validate_selection,
    build_embodied_probability_sjt_messages,
    build_legacy_persona_prompt,
    build_legacy_sjt_messages,
    load_comparison_summaries,
    load_legacy_pool,
)
from .mussel_ipip_reference import (
    FACET_CODES,
    DIMENSION_TO_IPIP,
    _load_selected_scales,
    _ordered_items,
    _validate_ratings,
    build_ipip_messages,
    score_ipip_records,
)
from .storage import write_csv, write_json
from .virtual_prompt_registry import (
    DEFAULT_PROMPT_ROOT,
    format_extra_parameters,
    get_prompt_spec,
    load_prompt_specs,
    select_prompt_ids_interactively,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSSEL = PROJECT_ROOT / "docs" / "mussel_zh.json"
DEFAULT_IPIP = PROJECT_ROOT / "knowledge_base" / "items" / "ipip_neo_items.json"
DEFAULT_PROMPT_IDS = ("embodied_probability_v1", "score_profile_v1")
PROMPT_VERSION = "fresh-mussel-ipip-two-method-v1"
# Same default seed as the main CLI.  It is a fixed protocol parameter, not
# an extra interactive setting in the Mussel×IPIP launcher.
SCORE_PROFILE_SEED = 7
TARGET_SCORE_DIMENSIONS = tuple(DIMENSION_TO_IPIP)
DEFAULT_TARGET_FACET = "extraversion_gregariousness"
DEFAULT_SAME_DOMAIN_FACET = "extraversion_warmth"
DEFAULT_CROSS_DOMAIN_FACET = "neuroticism_anxiety"


@dataclass(frozen=True)
class FreshComparisonConfig:
    experiment: Path
    respondents: int = 285
    summary_pool: Path = DEFAULT_IMPORTED_SUMMARY_POOL
    mussel_path: Path = DEFAULT_MUSSEL
    ipip_path: Path = DEFAULT_IPIP
    model_id: str = "glm-5.3-flash"
    max_concurrency: int = 30
    max_retries: int = 2
    timeout_seconds: float | None = None
    sampling_seed: int = 20260916
    score_seed: int = SCORE_PROFILE_SEED
    score_mean: float = 50.0
    score_sd: float = 15.0
    target_facet: str = DEFAULT_TARGET_FACET
    same_domain_facet: str = DEFAULT_SAME_DOMAIN_FACET
    cross_domain_facet: str = DEFAULT_CROSS_DOMAIN_FACET
    prompt_ids: str | None = None
    output: Path | None = None

    def validate(self) -> None:
        if self.respondents < 3:
            raise ValueError("respondents至少为3")
        if not 1 <= self.max_concurrency <= 50:
            raise ValueError("max_concurrency必须在1至50之间")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须为正数")
        if not self.model_id.strip():
            raise ValueError("model_id不能为空")
        if not math.isfinite(float(self.score_mean)) or not 0.0 < float(self.score_mean) < 100.0:
            raise ValueError("score_mean必须大于0且小于100")
        if not math.isfinite(float(self.score_sd)) or float(self.score_sd) <= 0:
            raise ValueError("score_sd必须为正数")
        _matched_conditions(self)

    def resolved_prompt_ids(self) -> tuple[str, ...]:
        if self.prompt_ids is None:
            specs = {
                prompt_id: spec
                for prompt_id, spec in load_prompt_specs(DEFAULT_PROMPT_ROOT).items()
                if spec.persona_input != "five_mussel_facets"
            }
            return select_prompt_ids_interactively(
                specs,
                default_ids=DEFAULT_PROMPT_IDS,
            )
        prompt_ids = tuple(
            prompt_id.strip()
            for prompt_id in str(self.prompt_ids).split(",")
            if prompt_id.strip()
        )
        if not prompt_ids:
            raise ValueError("prompt_ids不能为空")
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("prompt_ids不能重复")
        for prompt_id in prompt_ids:
            if get_prompt_spec(prompt_id).persona_input == "five_mussel_facets":
                raise ValueError(
                    "fresh Mussel×IPIP 已停用五个 facet 共同变化路径；"
                    "请使用 score_profile_v1 或其他单facet/具身方法"
                )
        return prompt_ids


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_value(dict(record)), ensure_ascii=False))
            handle.write("\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            rows.append(value)
    return rows


def _facet_catalog() -> dict[str, dict[str, Any]]:
    catalog = {
        str(row["dimension_id"]): dict(row)
        for row in build_score_dimension_catalog(construct_selection_catalog())
        if row.get("level") == "facet"
    }
    return catalog


def _matched_conditions(config: FreshComparisonConfig) -> list[dict[str, Any]]:
    """Use the same three-arm score design as the main CLI."""

    catalog = list(_facet_catalog().values())
    return normalize_matched_conditions(
        [
            {
                "condition_id": "target",
                "role": "target",
                "dimension_id": config.target_facet,
            },
            {
                "condition_id": "same_domain",
                "role": "same_domain_non_target",
                "dimension_id": config.same_domain_facet,
            },
            {
                "condition_id": "cross_domain",
                "role": "cross_domain_non_target",
                "dimension_id": config.cross_domain_facet,
            },
        ],
        dimension_catalog=catalog,
        target_dimension_id=config.target_facet,
    )


def _score_spec(dimension_id: str, *, mean_score: float) -> dict[str, Any]:
    spec = _facet_catalog().get(str(dimension_id))
    if spec is None:
        raise ValueError(f"构念注册表缺少facet：{dimension_id}")
    spec["mean_score"] = float(mean_score)
    return spec


def _uses_score_input(prompt_id: str) -> bool:
    return get_prompt_spec(prompt_id).persona_input == "score_values"


def _prepare_participants(
    config: FreshComparisonConfig,
    *,
    use_matched_score_arms: bool,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    pool, source_profiles = load_legacy_pool(PROJECT_ROOT)
    summaries, summary_path, summary_manifest = load_comparison_summaries(
        PROJECT_ROOT, config.summary_pool.resolve()
    )
    available = [str(rid) for rid in source_profiles if str(rid) in summaries]
    if config.respondents > len(available):
        raise ValueError(
            f"要求{config.respondents}名被试，但画像和总结只能完整对齐{len(available)}名"
        )
    if use_matched_score_arms:
        conditions = _matched_conditions(config)
        score_refs, generation_diagnostics = generate_matched_condition_respondent_refs(
            config.respondents,
            conditions,
            seed=config.score_seed,
            mean_score=config.score_mean,
            standard_deviation=config.score_sd,
        )
        source_by_matched = {
            f"matched-{index:04d}": rid
            for index, rid in enumerate(available[: config.respondents], start=1)
        }
        subject_ids = [str(ref["respondent_id"]) for ref in score_refs]
        score_specs_by_condition = {
            str(row["condition_id"]): [
                _score_spec(str(row["dimension_id"]), mean_score=config.score_mean)
            ]
            for row in generation_diagnostics.get("conditions", [])
        }
    else:
        subject_ids = available[: config.respondents]
        score_refs = [
            {
                "respondent_id": rid,
                "score_values": {},
                "condition_id": None,
                "matched_subject_id": None,
            }
            for rid in subject_ids
        ]
        generation_diagnostics = {
            "generator_version": None,
            "sampling_design": "source_persona_pool",
            "sample_size": len(subject_ids),
        }
        source_by_matched = {}
        score_specs_by_condition = {}
    score_profiles: dict[str, dict[str, Any]] = {}
    participants: list[dict[str, Any]] = []
    for ref in score_refs:
        rid = str(ref["respondent_id"])
        profile = dict(ref)
        source_respondent_id = (
            source_by_matched.get(str(ref.get("matched_subject_id")))
            if use_matched_score_arms
            else rid
        )
        profile["source_respondent_id"] = source_respondent_id
        if use_matched_score_arms:
            profile["score_specs"] = score_specs_by_condition[str(ref["condition_id"])]
        score_profiles[rid] = profile
        if source_respondent_id not in summaries:
            raise ValueError(f"缺少被试画像总结：{source_respondent_id}")
        if use_matched_score_arms:
            score_values = dict(profile.get("score_values") or {})
        else:
            score_values = {}
        participants.append({
            "respondent_id": rid,
            "condition_id": profile.get("condition_id"),
            "arm_id": profile.get("arm_id"),
            "group_id": profile.get("group_id"),
            "matched_subject_id": profile.get("matched_subject_id"),
            "active_dimension_id": profile.get("active_dimension_id"),
            "source_respondent_id": source_respondent_id,
            "score_values": score_values,
            "summary_available": True,
        })
    summary_lookup = {
        str(profile["respondent_id"]): summaries[str(profile["source_respondent_id"])]
        for profile in score_profiles.values()
    }
    metric_subject_ids = (
        [str(ref["respondent_id"]) for ref in score_refs if ref.get("condition_id") == "target"]
        if use_matched_score_arms
        else list(subject_ids)
    )
    metadata = {
        "source_pool_id": pool.get("pool_id"),
        "source_profile_count": len(source_profiles),
        "summary_path": str(summary_path),
        "summary_manifest": summary_manifest,
        "score_specs": [
            _score_spec(config.target_facet, mean_score=config.score_mean)
        ] if use_matched_score_arms else [],
        "score_specs_by_condition": score_specs_by_condition,
        "score_generation_diagnostics": generation_diagnostics,
        "matched_conditions": _matched_conditions(config) if use_matched_score_arms else [],
        "matched_score_arms": bool(use_matched_score_arms),
        "metric_subject_ids": metric_subject_ids,
    }
    return subject_ids, score_profiles, summary_lookup, participants, metadata


def _score_lines(
    profile: Mapping[str, Any],
    score_specs: Sequence[Mapping[str, Any]],
) -> str:
    values = profile.get("score_values")
    if not isinstance(values, Mapping):
        raise ValueError("自定义facet提示词缺少score_values")
    lines: list[str] = []
    for spec in score_specs:
        dimension_id = str(spec.get("dimension_id"))
        if dimension_id not in values:
            continue
        label = (
            f"{spec.get('domain_name_en')} > {spec.get('facet_name_en')}"
            f"（{spec.get('facet_name')}）"
        )
        lines.append(f"{dimension_id} | {label} | {float(values[dimension_id]):.1f}")
    return "\n".join(lines)


def _persona_prompts(
    prompt_id: str,
    subject_ids: Sequence[str],
    score_profiles: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, str],
    score_specs_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    for_ipip: bool,
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    spec = get_prompt_spec(prompt_id)
    for rid in subject_ids:
        profile = score_profiles[rid]
        active_specs = score_specs_by_condition.get(
            str(profile.get("condition_id")),
            profile.get("score_specs") or [],
        )
        if prompt_id == "score_profile_v1":
            prompts[rid] = build_persona_prompt(
                profile, score_specs=active_specs
            )
        elif prompt_id == "embodied_probability_v1" and for_ipip:
            prompts[rid] = build_legacy_persona_prompt(
                {}, summaries[rid], persona_mode="summary_only"
            )
        elif prompt_id == "embodied_probability_v1":
            prompts[rid] = build_legacy_persona_prompt(
                {}, summaries[rid], persona_mode="summary_embodied_probability"
            )
        else:
            prompts[rid] = spec.render(
                persona_summary=summaries[rid],
                score_lines=_score_lines(profile, active_specs),
                extra_parameters=format_extra_parameters(spec, rid),
            )
    return prompts


async def _run_sjt_condition(
    *,
    condition: str,
    subject_ids: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, str],
    participant_metadata: Mapping[str, Mapping[str, Any]],
    sjt_response_mode: str,
    runnable: Any,
    model_id: str,
    config: FreshComparisonConfig,
    output_dir: Path,
    run_id: str,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "responses.jsonl"
    existing = _load_jsonl(path)
    existing_keys = {
        (str(row.get("respondent_id")), str(row.get("item_id")))
        for row in existing
    }
    expected = {
        (str(rid), str(item["item_id"]))
        for rid in subject_ids
        for item in items
    }
    if not existing_keys <= expected or len(existing_keys) != len(existing):
        raise ValueError(f"{condition} Mussel缓存包含重复或非预期记录")
    item_by_id = {str(item["item_id"]): dict(item) for item in items}
    jobs = [
        (str(rid), dict(item))
        for rid in subject_ids
        for item in items
        if (str(rid), str(item["item_id"])) not in existing_keys
    ]
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    lock = asyncio.Lock()
    completed = len(existing)
    total = len(subject_ids) * len(items)

    async def one(rid: str, item: Mapping[str, Any]) -> None:
        nonlocal completed
        item_id = str(item["item_id"])
        if sjt_response_mode == "choice_probability":
            payload = await _invoke_with_retry(
                runnable,
                build_embodied_probability_sjt_messages(prompts[rid], item),
                semaphore=semaphore,
                validator=_validate_choice_probabilities,
                max_retries=config.max_retries,
                retry_delay_seconds=1.0,
                request_timeout_seconds=timeout,
                job_label=f"fresh Mussel probability {rid}/{item_id}",
            )
            selected, draw, sampling_key = _sample_probability_choice(
                payload["normalized"],
                sampling_seed=config.sampling_seed,
                respondent_id=rid,
                item=item,
            )
            extra = {
                "raw_choice_probabilities": payload["raw"],
                "choice_probabilities": payload["normalized"],
                "sampling_draw": draw,
                "sampling_key": sampling_key,
                "choice_generation": "categorical_draw_from_model_probabilities",
            }
        else:
            selected = await _invoke_with_retry(
                runnable,
                build_legacy_sjt_messages(prompts[rid], item),
                semaphore=semaphore,
                validator=_validate_selection,
                max_retries=config.max_retries,
                retry_delay_seconds=1.0,
                request_timeout_seconds=timeout,
                job_label=f"fresh Mussel selection {rid}/{item_id}",
            )
            extra = {"choice_generation": "structured_single_selection"}
        record = {
            "record_type": "fresh_mussel_response",
            "condition": condition,
            "respondent_id": rid,
            "item_id": item_id,
            "selected_option_id": selected,
            "score": int(item["scoring_key"][selected]),
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "prompt_id": condition,
            "condition_id": participant_metadata.get(rid, {}).get("condition_id"),
            "arm_id": participant_metadata.get(rid, {}).get("arm_id"),
            "group_id": participant_metadata.get(rid, {}).get("group_id"),
            "matched_subject_id": participant_metadata.get(rid, {}).get("matched_subject_id"),
            "active_dimension_id": participant_metadata.get(rid, {}).get("active_dimension_id"),
            **extra,
        }
        async with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed == total or completed % max(1, total // 20) == 0:
                print(
                    f"[全量Mussel] {condition} {completed}/{total} ({completed / total:.0%})",
                    flush=True,
                )

    if jobs:
        print(
            f"[全量Mussel] {condition}开始：新调用={len(jobs)}；已缓存={len(existing)}",
            flush=True,
        )
        results = await asyncio.gather(
            *(one(rid, item) for rid, item in jobs), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            write_json(output_dir / "errors.json", {"errors": [str(e) for e in errors]})
            raise RuntimeError(
                f"{condition} Mussel有{len(errors)}个调用失败；首个错误：{errors[0]}"
            )
    records = _load_jsonl(path)
    actual = {
        (str(row.get("respondent_id")), str(row.get("item_id")))
        for row in records
    }
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"{condition} Mussel作答不完整：应为{len(expected)}条，实际为{len(records)}条"
        )
    write_json(output_dir / "run.json", {
        "condition": condition,
        "record_count": len(records),
        "new_calls": len(jobs),
        "model_id": model_id,
    })
    return records


async def _run_ipip_condition(
    *,
    condition: str,
    subject_ids: Sequence[str],
    scales: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, str],
    participant_metadata: Mapping[str, Mapping[str, Any]],
    runnable: Any,
    model_id: str,
    config: FreshComparisonConfig,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "responses.jsonl"
    existing = _load_jsonl(path)
    expected = {
        (str(rid), str(scale["facet_code"]), str(item["item_id"]))
        for rid in subject_ids
        for scale in scales
        for item in scale["items"]
    }
    existing_keys = {
        (str(row.get("respondent_id")), str(row.get("facet_code")), str(row.get("item_id")))
        for row in existing
    }
    if not existing_keys <= expected or len(existing_keys) != len(existing):
        raise ValueError(f"{condition} IPIP缓存包含重复或非预期记录")
    jobs = []
    for rid in subject_ids:
        for scale in scales:
            scale_keys = {
                (str(rid), str(scale["facet_code"]), str(item["item_id"]))
                for item in scale["items"]
            }
            if not scale_keys <= existing_keys:
                jobs.append((str(rid), scale))
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    lock = asyncio.Lock()
    completed = len(existing) // 10
    total = len(subject_ids) * len(scales)

    async def one(rid: str, scale: Mapping[str, Any]) -> None:
        ordered = _ordered_items(scale)
        ratings = await _invoke_with_retry(
            runnable,
            build_ipip_messages(prompts[rid], scale),
            semaphore=semaphore,
            validator=lambda value: _validate_ratings(value, len(ordered)),
            max_retries=config.max_retries,
            retry_delay_seconds=1.0,
            request_timeout_seconds=timeout,
            job_label=f"fresh IPIP {condition} {rid}/{scale['facet_code']}",
        )
        rows = []
        for position, (item, raw) in enumerate(zip(ordered, ratings), 1):
            raw = int(raw)
            polarity = str(item["polarity"])
            rows.append({
                "record_type": "fresh_ipip_response",
                "condition": condition,
                "respondent_id": rid,
                "facet_code": str(scale["facet_code"]),
                "item_id": str(item["item_id"]),
                "prompt_position": position,
                "raw_response": raw,
                "polarity": polarity,
                "score": raw if polarity == "positive" else 6 - raw,
                "model_id": model_id,
                "prompt_version": PROMPT_VERSION,
                "prompt_id": condition,
                "condition_id": participant_metadata.get(rid, {}).get("condition_id"),
                "arm_id": participant_metadata.get(rid, {}).get("arm_id"),
                "group_id": participant_metadata.get(rid, {}).get("group_id"),
                "matched_subject_id": participant_metadata.get(rid, {}).get("matched_subject_id"),
                "active_dimension_id": participant_metadata.get(rid, {}).get("active_dimension_id"),
            })
        async with lock:
            with path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            nonlocal_completed[0] += 1
            if nonlocal_completed[0] == total or nonlocal_completed[0] % max(1, total // 20) == 0:
                print(
                    f"[全量IPIP] {condition} {nonlocal_completed[0]}/{total} ({nonlocal_completed[0] / total:.0%})",
                    flush=True,
                )

    nonlocal_completed = [completed]
    if jobs:
        print(
            f"[全量IPIP] {condition}开始：新调用={len(jobs)}；已缓存批次={completed}",
            flush=True,
        )
        results = await asyncio.gather(
            *(one(rid, scale) for rid, scale in jobs), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            write_json(output_dir / "errors.json", {"errors": [str(e) for e in errors]})
            raise RuntimeError(
                f"{condition} IPIP有{len(errors)}个批次失败；首个错误：{errors[0]}"
            )
    records = _load_jsonl(path)
    actual = {
        (str(row.get("respondent_id")), str(row.get("facet_code")), str(row.get("item_id")))
        for row in records
    }
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"{condition} IPIP作答不完整：应为{len(expected)}条，实际为{len(records)}条"
        )
    write_json(output_dir / "run.json", {
        "condition": condition,
        "record_count": len(records),
        "batch_count": total,
        "new_batch_calls": len(jobs),
        "model_id": model_id,
    })
    return records


def _sjt_metrics(
    condition: str,
    records: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float | None], list[dict[str, Any]]]:
    metric_ids = {str(value) for value in subject_ids}
    metric_records = [
        record for record in records
        if str(record.get("respondent_id")) in metric_ids
    ]
    long_frame, facet_scores = score_mussel_records(
        metric_records, expected_respondent_ids=subject_ids, items=items
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(output_dir / "scored_responses.csv", index=False, encoding="utf-8-sig")
    facet_scores.reset_index().to_csv(output_dir / "facet_scores.csv", index=False, encoding="utf-8-sig")
    item_map = {str(item["item_id"]): dict(item) for item in items}
    item_rows: list[dict[str, Any]] = []
    facet_alphas: dict[str, float | None] = {}
    for dimension in TARGET_SCORE_DIMENSIONS:
        ids = [str(item["item_id"]) for item in items if str(item["target_dimension_id"]) == dimension]
        matrix = (
            long_frame[long_frame["target_dimension_id"] == dimension]
            .pivot(index="respondent_id", columns="item_id", values="score")
            .reindex(index=list(subject_ids), columns=ids)
        )
        if matrix.isna().any().any() or matrix.shape != (len(subject_ids), 22):
            raise ValueError(f"{condition}/{dimension} Mussel计分矩阵不完整")
        total = matrix.sum(axis=1)
        citcs: list[float] = []
        for item_id in ids:
            citc = _pearson(matrix[item_id].tolist(), (total - matrix[item_id]).tolist())
            if citc is not None:
                citcs.append(float(citc))
            choices = long_frame[long_frame["item_id"] == item_id]["selected_option_id"].astype(str)
            row = {
                "condition": condition,
                "target_dimension_id": dimension,
                "item_id": item_id,
                "citc": citc,
                "mean_score": float(matrix[item_id].mean()),
                "n": len(subject_ids),
            }
            for option_id in OPTION_IDS:
                row[f"option_{option_id}_n"] = int((choices == option_id).sum())
                row[f"option_{option_id}_rate"] = float((choices == option_id).mean())
            item_rows.append(row)
        facet_alphas[dimension] = _alpha(matrix)
        write_csv(output_dir / f"{dimension}_item_metrics.csv", [row for row in item_rows if row["target_dimension_id"] == dimension])
    write_csv(output_dir / "item_metrics.csv", item_rows)
    return facet_scores, facet_alphas, item_rows


def _ipip_metrics(
    condition: str,
    records: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    scales: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float | None], list[dict[str, Any]]]:
    metric_ids = {str(value) for value in subject_ids}
    metric_records = [
        record for record in records
        if str(record.get("respondent_id")) in metric_ids
    ]
    scores, reliability = score_ipip_records(metric_records, subject_ids, scales)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.reset_index().to_csv(output_dir / "facet_scores.csv", index=False, encoding="utf-8-sig")
    write_csv(output_dir / "reliability.csv", reliability)
    return scores, {
        str(row["facet_code"]): row.get("virtual_sample_alpha")
        for row in reliability
    }, reliability


def _validity_rows(
    condition: str,
    sjt_scores: pd.DataFrame,
    sjt_alphas: Mapping[str, float | None],
    ipip_scores: pd.DataFrame,
    ipip_alphas: Mapping[str, float | None],
    subject_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, ipip_code in DIMENSION_TO_IPIP.items():
        correlations = {
            code: _spearman(
                sjt_scores.loc[list(subject_ids), dimension].tolist(),
                ipip_scores.loc[list(subject_ids), code].tolist(),
            )
            for code in FACET_CODES
        }
        target = correlations[ipip_code]
        non_target = [abs(float(value)) for code, value in correlations.items() if code != ipip_code and value is not None]
        maximum = max(non_target) if non_target else None
        rows.append({
            "condition": condition,
            "target_dimension_id": dimension,
            "target_ipip_facet": ipip_code,
            "respondent_count": len(subject_ids),
            "sjt_alpha": sjt_alphas.get(dimension),
            "ipip_alpha": ipip_alphas.get(ipip_code),
            "convergent_spearman": target,
            "max_abs_non_target_spearman": maximum,
            "discriminant_gap": target - maximum if target is not None and maximum is not None else None,
            **{f"rho_ipip_{code}": value for code, value in correlations.items()},
        })
    return rows


def _telemetry_summary(root: Path, run_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "telemetry").glob("calls_*.jsonl")):
        records.extend(read_ledger(path, run_id=run_id))
    def total(field: str) -> int | None:
        values = [row.get(field) for row in records]
        if not values or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            return None
        return int(sum(values))
    return {
        "calls": len(records),
        "error_calls": sum(row.get("status") == "error" for row in records),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "duration_ms": total("duration_ms"),
        "data_available": bool(records),
    }


def _render_report(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    respondent_count: int,
    conditions: Sequence[str],
    *,
    matched_score_arms: bool = False,
    metric_respondent_count: int | None = None,
) -> Path:
    def f(value: Any) -> str:
        if value is None:
            return "不可估计"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "不可估计"
        return f"{number:.3f}" if math.isfinite(number) else "不可估计"
    html_rows = "".join(
        "<tr>" + "".join([
            f"<td>{escape(str(row['condition']))}</td>",
            f"<td>{escape(str(row['target_dimension_id']))}</td>",
            f"<td>{escape(str(row['target_ipip_facet']))}</td>",
            f"<td>{f(row.get('sjt_alpha'))}</td>",
            f"<td>{f(row.get('ipip_alpha'))}</td>",
            f"<td>{f(row.get('convergent_spearman'))}</td>",
            f"<td>{f(row.get('max_abs_non_target_spearman'))}</td>",
            f"<td>{f(row.get('discriminant_gap'))}</td>",
        ]) + "</tr>"
        for row in rows
    )
    condition_text = "、".join(escape(str(condition)) for condition in conditions)
    design_text = (
        "分数型方法采用主 CLI 的三臂匹配设计；标准信效度表使用 target 臂，"
        f"target 臂有效样本={metric_respondent_count}，同域和跨域臂保留作控制。"
        if matched_score_arms
        else "本次没有启用匹配分数三臂设计。"
    )
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>全量Mussel×IPIP虚拟被试提示词比较</title>
<style>body{{font:15px/1.6 sans-serif;max-width:1300px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f1f4f8}}.note{{background:#fff8df;padding:12px}}</style>
<h1>全量Mussel×IPIP虚拟被试提示词比较</h1>
<p>全新模型调用；作答记录被试={respondent_count}；Mussel=110题；IPIP=5个facet、每facet 10题；条件={condition_text}。</p>
<div class='note'>{design_text}<br>每个提示词条件使用同一套题目；IPIP作答不读取Mussel作答；该实验是虚拟被试内部比较，不能直接证明真人效度。</div>
<h2>信效度结果</h2><table><thead><tr><th>条件</th><th>Mussel facet</th><th>IPIP facet</th><th>Mussel α</th><th>IPIP α</th><th>汇聚ρ</th><th>最大非目标|ρ|</th><th>区分差值</th></tr></thead><tbody>{html_rows}</tbody></table>
<h2>运行信息</h2><p>模型调用={telemetry.get('calls')}；错误调用={telemetry.get('error_calls')}；总Token={telemetry.get('total_tokens')}。</p>
<p>汇聚效度为Mussel目标facet总分与对应IPIP facet总分的Spearman相关；区分差值=目标相关−其他IPIP facet最大绝对相关。</p></html>"""
    path = root / "summary" / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


async def run_fresh_comparison(config: FreshComparisonConfig) -> tuple[Path, dict[str, Any]]:
    config.validate()
    prompt_ids = config.resolved_prompt_ids()
    prompt_specs = {prompt_id: get_prompt_spec(prompt_id) for prompt_id in prompt_ids}
    experiment = config.experiment.resolve()
    items, mussel_metadata = load_mussel_items(config.mussel_path.resolve())
    if len(items) != 110:
        raise ValueError(f"Mussel必须为110题，实际为{len(items)}")
    scales, ipip_metadata = _load_selected_scales(config.ipip_path.resolve())
    use_matched_score_arms = any(_uses_score_input(prompt_id) for prompt_id in prompt_ids)
    subject_ids, score_profiles, summaries, participants, source_metadata = _prepare_participants(
        config,
        use_matched_score_arms=use_matched_score_arms,
    )
    score_specs_by_condition = source_metadata["score_specs_by_condition"]
    participant_metadata = {
        str(row["respondent_id"]): row
        for row in participants
    }
    output = config.output.resolve() if config.output else experiment / "fresh_mussel_ipip_comparison" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"fresh-mussel-ipip-{output.name}"
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "run_id": run_id,
        "experiment": str(experiment),
        "respondent_count": len(subject_ids),
        "respondent_count_per_matched_arm": config.respondents if use_matched_score_arms else None,
        "respondent_ids": subject_ids,
        "conditions": list(prompt_ids),
        "mussel_item_count": len(items),
        "ipip_facets": list(FACET_CODES),
        "model_id": config.model_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_ids": list(prompt_ids),
        "matched_score_arms": use_matched_score_arms,
        "matched_conditions": source_metadata.get("matched_conditions", []),
        "fresh_response_policy": "do_not_read_prior_mussel_or_ipip_response_files",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_json(output / "config.json", _json_value(asdict(config)))
    write_json(output / "participants.json", participants)
    write_json(output / "mussel_definition.json", {"metadata": mussel_metadata, "items": items, "sha256": _sha256_file(config.mussel_path.resolve())})
    write_json(output / "ipip_definition.json", {"metadata": ipip_metadata, "scales": scales, "sha256": _sha256_file(config.ipip_path.resolve())})
    write_json(output / "source_metadata.json", _json_value(source_metadata))

    model = get_model(config.model_id)
    probability_runnable, _ = with_compatible_structured_output(model, ChoiceProbabilityOutput)
    selection_runnable, _ = with_compatible_structured_output(model, SJTSelectionOutput)
    ipip_runnable, _ = with_compatible_structured_output(model, NeoFFIBatchOutput)
    model_id = _model_id(model, config.model_id)
    sjt_records: dict[str, list[dict[str, Any]]] = {}
    ipip_records: dict[str, list[dict[str, Any]]] = {}
    shared_semaphore = asyncio.Semaphore(config.max_concurrency)

    async def run_condition(condition: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        spec = prompt_specs[condition]
        prompts = _persona_prompts(
            condition,
            subject_ids,
            score_profiles,
            summaries,
            score_specs_by_condition,
            for_ipip=False,
        )
        condition_sjt_records = await _run_sjt_condition(
            condition=condition,
            subject_ids=subject_ids,
            items=items,
            prompts=prompts,
            participant_metadata=participant_metadata,
            sjt_response_mode=spec.sjt_response_mode,
            runnable=probability_runnable
            if spec.sjt_response_mode == "choice_probability"
            else selection_runnable,
            model_id=model_id,
            config=config,
            output_dir=output / condition / "mussel",
            run_id=run_id,
            semaphore=shared_semaphore,
        )
        ipip_prompts = _persona_prompts(
            condition,
            subject_ids,
            score_profiles,
            summaries,
            score_specs_by_condition,
            for_ipip=True,
        )
        condition_ipip_records = await _run_ipip_condition(
            condition=condition,
            subject_ids=subject_ids,
            scales=scales,
            prompts=ipip_prompts,
            participant_metadata=participant_metadata,
            runnable=ipip_runnable,
            model_id=model_id,
            config=config,
            output_dir=output / condition / "ipip",
            semaphore=shared_semaphore,
        )
        return condition, condition_sjt_records, condition_ipip_records

    try:
        with output_scope(output / "runtime", telemetry=output / "telemetry"), run_context(run_id):
            results = await asyncio.gather(
                *(run_condition(condition) for condition in prompt_ids),
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, Exception)]
            if errors:
                raise RuntimeError(
                    f"并发提示词实验有{len(errors)}个条件失败；首个错误：{errors[0]}"
                )
            for result in results:
                condition, condition_sjt_records, condition_ipip_records = result
                sjt_records[condition] = condition_sjt_records
                ipip_records[condition] = condition_ipip_records
    except Exception as exc:
        write_json(manifest_path, {**manifest, "status": "failed", "error": str(exc)})
        raise

    validity: list[dict[str, Any]] = []
    metric_subject_ids = [str(value) for value in source_metadata["metric_subject_ids"]]
    for condition in prompt_ids:
        sjt_scores, sjt_alphas, _ = _sjt_metrics(
            condition,
            sjt_records[condition],
            items,
            metric_subject_ids,
            output / condition / "mussel",
        )
        ipip_scores, ipip_alphas, _ = _ipip_metrics(
            condition,
            ipip_records[condition],
            metric_subject_ids,
            scales,
            output / condition / "ipip",
        )
        validity.extend(
            _validity_rows(
                condition,
                sjt_scores,
                sjt_alphas,
                ipip_scores,
                ipip_alphas,
                metric_subject_ids,
            )
        )
    write_csv(output / "summary" / "validity.csv", validity)
    comparison_rows: list[dict[str, Any]] = []
    by_key = {(str(row["condition"]), str(row["target_dimension_id"])): row for row in validity}
    reference_condition = prompt_ids[0]
    for comparison_condition in prompt_ids[1:]:
        for dimension in TARGET_SCORE_DIMENSIONS:
            reference = by_key[(reference_condition, dimension)]
            comparison = by_key[(comparison_condition, dimension)]
            comparison_rows.append({
                "reference_condition": reference_condition,
                "comparison_condition": comparison_condition,
                "target_dimension_id": dimension,
                "target_ipip_facet": reference["target_ipip_facet"],
                "delta_sjt_alpha_comparison_minus_reference": comparison["sjt_alpha"] - reference["sjt_alpha"],
                "delta_ipip_alpha_comparison_minus_reference": comparison["ipip_alpha"] - reference["ipip_alpha"],
                "delta_convergent_rho_comparison_minus_reference": comparison["convergent_spearman"] - reference["convergent_spearman"],
                "delta_discriminant_gap_comparison_minus_reference": comparison["discriminant_gap"] - reference["discriminant_gap"],
            })
    write_csv(output / "summary" / "method_comparison.csv", comparison_rows)
    telemetry = _telemetry_summary(output, run_id)
    summary = {
        "status": "complete",
        "respondent_count": len(subject_ids),
        "metric_respondent_count": len(metric_subject_ids),
        "respondent_count_per_matched_arm": config.respondents if use_matched_score_arms else None,
        "conditions": list(prompt_ids),
        "expected_model_calls": len(subject_ids) * (len(items) + len(scales)) * len(prompt_ids),
        "matched_score_arms": use_matched_score_arms,
        "matched_conditions": source_metadata.get("matched_conditions", []),
        "validity": validity,
        "comparisons": comparison_rows,
        "telemetry": telemetry,
        "interpretation_boundary": "Virtual-respondent internal comparison only; no human-validity conclusion.",
    }
    write_json(output / "summary" / "summary.json", _json_value(summary))
    report = _render_report(
        output,
        validity,
        telemetry,
        len(subject_ids),
        prompt_ids,
        matched_score_arms=use_matched_score_arms,
        metric_respondent_count=len(metric_subject_ids),
    )
    write_json(manifest_path, {**manifest, "status": "complete", "model_id": model_id, "report": str(report), "completed_at": datetime.now(timezone.utc).isoformat()})
    return output, {**summary, "report": str(report)}


def build_parser() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="全新重跑具身与显式分数虚拟被试的Mussel×IPIP实验")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--respondents", type=int, default=285)
    parser.add_argument("--summary-pool", type=Path, default=DEFAULT_IMPORTED_SUMMARY_POOL)
    parser.add_argument("--mussel", dest="mussel_path", type=Path, default=DEFAULT_MUSSEL)
    parser.add_argument("--ipip", dest="ipip_path", type=Path, default=DEFAULT_IPIP)
    parser.add_argument("--model", dest="model_id", default="glm-5.3-flash")
    parser.add_argument("--max-concurrency", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=20260916)
    parser.add_argument("--score-seed", type=int, default=SCORE_PROFILE_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, result = asyncio.run(run_fresh_comparison(FreshComparisonConfig(**vars(args))))
    print(f"全量Mussel×IPIP双方法实验输出：{output}")
    print(f"被试={result['respondent_count']}；预计模型调用={result['expected_model_calls']}")
    print(f"报告：{result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
