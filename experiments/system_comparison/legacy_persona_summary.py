"""Build a frozen summary-only condition from the legacy respondent pool.

The source pool lives in the historical project.  Generated summaries and a
source snapshot are written under the current project's ``experiment_data``
tree so later comparison runs do not depend on mutable files in that project.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.evaluation.simulation import (
    PersonaSummaryOutput,
    _invoke_with_retry,
    _load_jsonl_records,
)
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context

from .config import fingerprint
from .legacy_pool_abc import (
    DEFAULT_LEGACY_PROJECT,
    _model_id,
    _sha256_file,
    _telemetry_summary,
    load_legacy_pool,
)
from .storage import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiment_data" / "legacy_persona_summary_pool"
SUMMARY_SCHEMA_VERSION = 1
SUMMARY_PROMPT_VERSION = "legacy-item-response-summary-only-v1"
CONDITION_ID = "legacy_summary_only"
_WHITESPACE = re.compile(r"\s+")
_FORBIDDEN_SUMMARY_TERMS = (
    "NEO",
    "大五人格",
    "人格问卷",
    "神经质",
    "外向性",
    "开放性",
    "宜人性",
    "尽责性",
    "分数",
    "得分",
    "计分",
)


def _cjk_character_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


@dataclass(frozen=True)
class LegacySummaryConfig:
    legacy_project: Path = DEFAULT_LEGACY_PROJECT
    output: Path = DEFAULT_OUTPUT
    model_id: str | None = None
    max_concurrency: int = 5
    max_retries: int = 2
    timeout_seconds: float | None = None

    def validate(self) -> None:
        if self.max_concurrency < 1 or self.max_concurrency > 50:
            raise ValueError("max_concurrency 必须在1至50之间")
        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries 必须在0至10之间")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须为正数")


def build_legacy_summary_messages(
    profile: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Use item-level answers only; never expose latent scores or labels."""

    personality_items = profile.get("personality_items")
    if not isinstance(personality_items, list) or not personality_items:
        raise ValueError("旧虚拟被试缺少人格逐题作答")
    lines: list[str] = []
    for item in personality_items:
        if not isinstance(item, Mapping):
            raise ValueError("旧虚拟被试包含无效人格题目")
        statement = item.get("statement")
        response_label = item.get("response_label")
        if not isinstance(statement, str) or not isinstance(response_label, str):
            raise ValueError("旧虚拟被试人格题缺少题干或作答标签")
        lines.append(f"{statement}：{response_label}")
    system_message = (
        "You summarize recurring behavioral response patterns from a virtual "
        "respondent's item-level questionnaire responses. Use only the supplied "
        "statements and response labels. Distinguish directly observed response "
        "patterns from broader inferences, and do not infer beyond what the "
        "responses support. Describe when the person tends to respond in a certain "
        "way, meaningful tensions across situations, and uncertainty. Preserve "
        "contradictions instead of forcing a perfectly coherent profile. Do not "
        "name or restate latent traits, facets, target constructs, or global "
        "personality labels; do not make evaluative judgments. Do not invent "
        "demographics, biography, motives, abilities, resources, diagnoses, life "
        "events, or causal explanations. Do not mention questionnaire names, item "
        "codes, numeric scores, scoring direction, or psychometric jargon. Use "
        "calibrated language such as 'tends to', 'may', or 'in some situations'. "
        "Write one compact Simplified-Chinese paragraph of 100-180 Chinese "
        "characters. Return JSON only: {\"summary\":\"...\"}."
    )
    return [
        ("system", system_message),
        (
            "human",
            "Item-level responses (the only evidence available):\n"
            + "\n".join(lines),
        ),
    ]


def _normalize_summary(value: Mapping[str, Any]) -> str:
    raw = value.get("summary")
    if not isinstance(raw, str):
        raise ValueError("人格总结缺少summary字符串")
    summary = _WHITESPACE.sub("", raw).strip()
    cjk_count = _cjk_character_count(summary)
    if len(summary) < 80 or len(summary) > 260 or cjk_count < 80 or cjk_count > 220:
        raise ValueError(
            "人格总结长度应接近100—180字，"
            f"当前汉字数={cjk_count}、非空白字符数={len(summary)}"
        )
    leaked = [term for term in _FORBIDDEN_SUMMARY_TERMS if term in summary]
    if leaked:
        raise ValueError("人格总结泄露构念、问卷或计分信息：" + "、".join(leaked))
    return summary


def _profile_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint(profile.get("personality_items") or [])


def _read_cached(
    path: Path,
    *,
    expected_ids: set[str],
    expected_model_id: str,
    profile_fingerprints: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl_records(path):
        rid = str(row.get("respondent_id") or "")
        if rid not in expected_ids:
            raise ValueError(f"总结缓存包含来源池外被试：{rid}")
        if rid in records:
            raise ValueError(f"总结缓存包含重复被试：{rid}")
        if row.get("model_id") != expected_model_id:
            raise ValueError(f"总结缓存模型不一致：{rid}")
        if row.get("prompt_version") != SUMMARY_PROMPT_VERSION:
            raise ValueError(f"总结缓存提示词版本不一致：{rid}")
        if row.get("input_fingerprint") != profile_fingerprints[rid]:
            raise ValueError(f"总结缓存输入指纹不一致：{rid}")
        _normalize_summary(row)
        records[rid] = dict(row)
    return records


def load_summary_condition(
    source: str | Path,
    *,
    expected_ids: Sequence[str] | None = None,
) -> tuple[dict[str, str], Path, dict[str, Any] | None]:
    """Load a generated summary-only condition for downstream experiments."""

    source_path = Path(source).resolve()
    if source_path.is_dir():
        root = source_path
        summary_path = root / "persona_summaries.jsonl"
    else:
        summary_path = source_path
        root = source_path.parent
    if not summary_path.is_file():
        raise FileNotFoundError(f"人格总结条件不存在：{summary_path}")
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if manifest is not None:
        if manifest.get("condition_id") != CONDITION_ID:
            raise ValueError("人格总结目录不是legacy_summary_only条件")
        if manifest.get("status") != "complete":
            raise ValueError("人格总结条件尚未完整生成")
    summaries: dict[str, str] = {}
    for row in _load_jsonl_records(summary_path):
        rid = str(row.get("respondent_id") or "")
        if not rid or rid in summaries:
            raise ValueError(f"人格总结被试ID为空或重复：{rid}")
        summaries[rid] = _normalize_summary(row)
    if expected_ids is not None:
        expected = list(map(str, expected_ids))
        missing = [rid for rid in expected if rid not in summaries]
        if missing:
            raise ValueError(
                f"人格总结条件缺少{len(missing)}名指定被试：{missing[:3]}"
            )
    return summaries, summary_path, manifest


def _render_report(
    root: Path,
    *,
    model_id: str,
    source_path: Path,
    records: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
) -> Path:
    cjk_lengths = [_cjk_character_count(str(row["summary"])) for row in records]
    within = sum(100 <= value <= 180 for value in cjk_lengths)
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['respondent_id']))}</td>"
        f"<td>{_cjk_character_count(str(row['summary']))}</td>"
        f"<td>{len(str(row['summary']))}</td>"
        f"<td>{escape(str(row['summary']))}</td>"
        "</tr>"
        for row in records
    )
    html = f"""<!doctype html>
<html lang='zh-CN'><meta charset='utf-8'><title>旧虚拟被试全量人格总结</title>
<style>body{{font:15px/1.65 sans-serif;max-width:1280px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;vertical-align:top}}th{{background:#f1f4f8;position:sticky;top:0}}code{{background:#f3f3f3;padding:2px 4px}}.note{{background:#fff8df;padding:12px}}</style>
<h1>旧虚拟被试全量人格总结</h1>
<p>条件：<code>{CONDITION_ID}</code>；被试={len(records)}；模型=<code>{escape(model_id)}</code>；按汉字计100—180字范围内={within}/{len(records)}。</p>
<div class='note'>每段总结仅由该被试的39道人格逐题作答生成，不输入潜变量名称、维度分数或计分方向。本条件后续只向作答模型提供总结文本，不再同时提供原始逐题作答。它是计算机实验对比条件，不是真人画像。</div>
<p>来源：<code>{escape(str(source_path))}</code>；模型调用={telemetry.get('calls')}；Token={telemetry.get('total_tokens')}。</p>
<table><thead><tr><th>被试ID</th><th>汉字数</th><th>总字符数</th><th>行为画像</th></tr></thead><tbody>{rows}</tbody></table>
</html>"""
    report = root / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


async def run_legacy_summary_pool(
    config: LegacySummaryConfig,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    source_project = config.legacy_project.resolve()
    output = config.output.resolve()
    pool_path = source_project / "sjt_system" / "data" / "virtual_respondents.json"
    pool, profiles = load_legacy_pool(source_project)
    respondent_ids = list(profiles)
    if len(respondent_ids) != 285:
        raise ValueError(
            f"旧版冻结池应包含285名被试，当前为{len(respondent_ids)}；拒绝静默改变条件"
        )
    if len(set(respondent_ids)) != len(respondent_ids):
        raise ValueError("旧版冻结池包含重复被试ID")

    model = get_model(config.model_id)
    runnable, structured_method = with_compatible_structured_output(
        model, PersonaSummaryOutput
    )
    model_id = _model_id(model, config.model_id)
    source_sha256 = _sha256_file(pool_path)
    profile_fingerprints = {
        rid: _profile_fingerprint(profile) for rid, profile in profiles.items()
    }
    binding = fingerprint(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "respondent_ids": respondent_ids,
            "profile_fingerprints": profile_fingerprints,
            "model_id": model_id,
            "prompt_version": SUMMARY_PROMPT_VERSION,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if previous and previous.get("binding") != binding:
        raise ValueError(
            "输出目录已绑定到不同来源、模型或提示词；请指定新目录，禁止混合总结"
        )
    run_id = str((previous or {}).get("run_id") or f"legacy-summary-{binding[:12]}")
    created_at = (previous or {}).get("created_at") or datetime.now(
        timezone.utc
    ).isoformat()
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "in_progress",
        "condition_id": CONDITION_ID,
        "persona_input_mode": "summary_only",
        "run_id": run_id,
        "binding": binding,
        "source_project": str(source_project),
        "source_path": str(pool_path),
        "source_sha256": source_sha256,
        "pool_id": pool.get("pool_id"),
        "respondent_count": len(respondent_ids),
        "respondent_ids": respondent_ids,
        "personality_item_count": len(pool.get("items") or []),
        "model_id": model_id,
        "structured_output_method": structured_method,
        "prompt_version": SUMMARY_PROMPT_VERSION,
        "created_at": created_at,
    }
    write_json(manifest_path, manifest)
    write_json(output / "config.json", {
        **asdict(config),
        "legacy_project": str(source_project),
        "output": str(output),
        "model_id": model_id,
    })
    # Freeze the complete source pool inside the current project for provenance.
    write_json(output / "source_virtual_respondents.json", pool)
    write_json(output / "condition.json", {
        "condition_id": CONDITION_ID,
        "description": "旧版285名虚拟被试的人格逐题作答经统一模型压缩后的summary-only黑盒条件",
        "respondent_count": len(respondent_ids),
        "persona_input_mode": "summary_only",
        "source_pool_id": pool.get("pool_id"),
        "summary_path": str((output / "persona_summaries.jsonl").resolve()),
        "comparison_limitations": [
            "总结由模型从旧人格逐题作答派生，仍可能保留语义匹配线索。",
            "总结是有损压缩，不应解释为真实被试人格测量。",
            "与其他条件比较时必须使用相同被试ID、问卷、作答模型和计分规则。",
        ],
    })

    summary_path = output / "persona_summaries.jsonl"
    cached = _read_cached(
        summary_path,
        expected_ids=set(respondent_ids),
        expected_model_id=model_id,
        profile_fingerprints=profile_fingerprints,
    )
    missing = [rid for rid in respondent_ids if rid not in cached]
    timeout = config.timeout_seconds or get_model_request_timeout_seconds()
    semaphore = asyncio.Semaphore(config.max_concurrency)
    write_lock = asyncio.Lock()
    start = perf_counter()
    completed = len(cached)
    failures: list[dict[str, Any]] = []
    print(
        f"[旧被试总结] 总数={len(respondent_ids)}；已缓存={len(cached)}；"
        f"待生成={len(missing)}；模型={model_id}",
        flush=True,
    )

    async def one(rid: str) -> None:
        nonlocal completed
        try:
            summary = await _invoke_with_retry(
                runnable,
                build_legacy_summary_messages(profiles[rid]),
                semaphore=semaphore,
                validator=_normalize_summary,
                max_retries=config.max_retries,
                retry_delay_seconds=1.0,
                request_timeout_seconds=timeout,
                job_label=f"legacy persona summary {rid}",
            )
            row = {
                "record_type": "persona_summary",
                "condition_id": CONDITION_ID,
                "respondent_id": rid,
                "summary": summary,
                "character_count": len(summary),
                "cjk_character_count": _cjk_character_count(summary),
                "model_id": model_id,
                "prompt_version": SUMMARY_PROMPT_VERSION,
                "input_fingerprint": profile_fingerprints[rid],
                "response_source": "regenerated_from_legacy_item_responses",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            async with write_lock:
                with summary_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                cached[rid] = row
                completed += 1
                if (
                    completed == len(respondent_ids)
                    or completed == 1
                    or completed % max(1, len(respondent_ids) // 20) == 0
                ):
                    print(
                        f"[旧被试总结] {completed}/{len(respondent_ids)} "
                        f"({completed / len(respondent_ids):.0%})",
                        flush=True,
                    )
                write_json(output / "progress.json", {
                    "status": "running",
                    "total": len(respondent_ids),
                    "completed": completed,
                    "remaining": len(respondent_ids) - completed,
                    "failed": len(failures),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:  # noqa: BLE001 - retain every failed respondent
            failure = {
                "respondent_id": rid,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            async with write_lock:
                failures.append(failure)
                with (output / "errors.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    with output_scope(
        output / "runtime", telemetry=output / "telemetry"
    ), run_context(run_id):
        await asyncio.gather(*(one(rid) for rid in missing))

    records_by_id = _read_cached(
        summary_path,
        expected_ids=set(respondent_ids),
        expected_model_id=model_id,
        profile_fingerprints=profile_fingerprints,
    )
    records = [records_by_id[rid] for rid in respondent_ids if rid in records_by_id]
    telemetry = _telemetry_summary(output, run_id)
    cost = {
        "model_id": model_id,
        "scheduled_calls_this_run": len(missing),
        "completed_records": len(records),
        "wall_seconds_this_run": round(perf_counter() - start, 3),
        "telemetry": telemetry,
        "fee": None,
        "fee_note": "未配置适用于该模型的输入、输出和缓存单价，费用不臆测。",
    }
    write_json(output / "cost.json", cost)
    write_csv(
        output / "persona_summaries.csv",
        [
            {
                "respondent_id": row["respondent_id"],
                "character_count": row["character_count"],
                "cjk_character_count": _cjk_character_count(str(row["summary"])),
                "summary": row["summary"],
            }
            for row in records
        ],
    )
    if len(records) != len(respondent_ids):
        write_json(output / "progress.json", {
            "status": "failed",
            "total": len(respondent_ids),
            "completed": len(records),
            "remaining": len(respondent_ids) - len(records),
            "failed": len(failures),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json(manifest_path, {
            **manifest,
            "status": "failed",
            "completed_count": len(records),
            "failed_count": len(failures),
            "cost": cost,
        })
        first = failures[0]["error"] if failures else "未知错误"
        raise RuntimeError(
            f"全量总结未完成：成功{len(records)}/{len(respondent_ids)}；"
            f"本轮失败{len(failures)}；首个错误：{first}。再次运行相同命令可续跑。"
        )

    report = _render_report(
        output,
        model_id=model_id,
        source_path=pool_path,
        records=records,
        telemetry=telemetry,
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    write_json(output / "progress.json", {
        "status": "complete",
        "total": len(respondent_ids),
        "completed": len(records),
        "remaining": 0,
        "failed": 0,
        "updated_at": completed_at,
    })
    write_json(manifest_path, {
        **manifest,
        "status": "complete",
        "completed_count": len(records),
        "summary_sha256": _sha256_file(summary_path),
        "report": str(report.resolve()),
        "cost": cost,
        "completed_at": completed_at,
    })
    return output, {
        "status": "complete",
        "respondent_count": len(records),
        "summary_path": str(summary_path.resolve()),
        "report": str(report.resolve()),
        "cost": cost,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成旧版285名虚拟被试的统一summary-only对比条件")
    parser.add_argument("--legacy-project", type=Path, default=DEFAULT_LEGACY_PROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", dest="timeout_seconds", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root, result = asyncio.run(run_legacy_summary_pool(LegacySummaryConfig(
        legacy_project=args.legacy_project,
        output=args.output,
        model_id=args.model_id,
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )))
    print(f"总结条件输出：{root}", flush=True)
    print(f"被试数：{result['respondent_count']}", flush=True)
    print(f"报告：{result['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
