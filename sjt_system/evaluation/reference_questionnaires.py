"""Reference-questionnaire resources used during virtual form development.

These scores are development-stage reference evidence.  They are not human
criterion-validity data and must not be presented as such.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSSEL_PATH = PROJECT_ROOT / "docs" / "mussel_zh.json"
MUSSEL_SCHEMA_VERSION = "mussel-zh-110-v1"
MUSSEL_SCORING_VERSION = "mussel-ab-high-cd-low-binary-v1"
MUSSEL_HIGH_OPTION_IDS = frozenset({"A", "B"})
MUSSEL_LOW_OPTION_IDS = frozenset({"C", "D"})

MUSSEL_FACET_TO_DIMENSION = {
    "开放性": "openness_ideas",
    "责任心": "conscientiousness_self_discipline",
    "外倾性": "extraversion_gregariousness",
    "宜人性": "agreeableness_compliance",
    "神经质": "neuroticism_self_consciousness",
}


def _numeric_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def mussel_binary_scoring_key(option_ids: Sequence[str]) -> dict[str, int]:
    """Return the source-order binary key used by the translated Mussel bank.

    The source bank presents the two high-trait options first (A/B) and the
    two low-trait options second (C/D).  The key is deliberately explicit so
    a later source-version change cannot silently change the scoring rule.
    """

    normalized = [str(option_id) for option_id in option_ids]
    if normalized != ["A", "B", "C", "D"]:
        raise ValueError(
            "Mussel 二元计分要求每题选项按 A、B、C、D 排列；"
            f"当前为 {normalized!r}"
        )
    return {
        option_id: 1 if option_id in MUSSEL_HIGH_OPTION_IDS else 0
        for option_id in normalized
    }


def load_mussel_items(
    path: str | Path = DEFAULT_MUSSEL_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the 110 translated Mussel items with an explicit 0/1 key."""

    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Mussel 题库必须是对象")
    if set(raw) != set(MUSSEL_FACET_TO_DIMENSION):
        raise ValueError(
            "Mussel 题库 facet 不完整或包含未知 facet："
            + ", ".join(str(key) for key in raw)
        )

    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for facet_key, dimension_id in MUSSEL_FACET_TO_DIMENSION.items():
        facet_rows = raw.get(facet_key)
        if not isinstance(facet_rows, Mapping) or len(facet_rows) != 22:
            raise ValueError(f"Mussel facet {facet_key} 必须包含22道题")
        count = 0
        for raw_item_id, raw_item in sorted(
            facet_rows.items(), key=lambda pair: _numeric_key(pair[0])
        ):
            if not isinstance(raw_item, Mapping):
                raise ValueError(f"Mussel 题目 {facet_key}/{raw_item_id} 结构无效")
            situation = raw_item.get("situation")
            options = raw_item.get("options")
            if not isinstance(situation, str) or not situation.strip():
                raise ValueError(f"Mussel 题目 {facet_key}/{raw_item_id} 缺少情境")
            if not isinstance(options, Mapping):
                raise ValueError(f"Mussel 题目 {facet_key}/{raw_item_id} 缺少选项")
            option_ids = [str(option_id) for option_id in options]
            scoring_key = mussel_binary_scoring_key(option_ids)
            item_number = str(raw_item_id)
            item_id = f"MUSSEL-{dimension_id}-{int(item_number):02d}"
            items.append(
                {
                    "item_id": item_id,
                    "source_item_id": item_number,
                    "facet_key": facet_key,
                    "target_dimension_id": dimension_id,
                    "scenario": situation,
                    "response_instruction": "请选择你最可能采取的行为。",
                    "response_options": [
                        {"option_id": option_id, "text": str(options[option_id])}
                        for option_id in option_ids
                    ],
                    "scoring_key": scoring_key,
                }
            )
            count += 1
        counts[facet_key] = count

    return items, {
        "schema_version": MUSSEL_SCHEMA_VERSION,
        "scoring_version": MUSSEL_SCORING_VERSION,
        "item_count": len(items),
        "items_per_facet": counts,
        "high_trait_option_ids": sorted(MUSSEL_HIGH_OPTION_IDS),
        "low_trait_option_ids": sorted(MUSSEL_LOW_OPTION_IDS),
        "scoring_description": "每题A/B为高特质行为记1，C/D为低特质行为记0。",
        "source_path": str(resolved.resolve()),
    }


def score_mussel_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_respondent_ids: Sequence[str],
    items: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw binary Mussel responses and return long/facet-score frames."""

    item_map = {
        str(item.get("item_id")): dict(item)
        for item in items
        if item.get("item_id")
    }
    expected_ids = {str(value) for value in expected_respondent_ids}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        respondent_id = str(record.get("respondent_id") or "")
        item_id = str(record.get("item_id") or "")
        if respondent_id not in expected_ids:
            raise ValueError(f"Mussel 作答包含未知 respondent_id：{respondent_id}")
        if item_id not in item_map:
            raise ValueError(f"Mussel 作答包含未知 item_id：{item_id}")
        key = (respondent_id, item_id)
        if key in seen:
            raise ValueError(f"Mussel 作答包含重复记录：{key}")
        seen.add(key)
        option_id = str(record.get("selected_option_id") or "")
        scoring_key = item_map[item_id]["scoring_key"]
        if option_id not in scoring_key:
            raise ValueError(f"Mussel 题目 {item_id} 选择了无效选项：{option_id}")
        recorded_version = record.get("scoring_version")
        if recorded_version is not None and recorded_version != MUSSEL_SCORING_VERSION:
            raise ValueError(
                f"Mussel 题目 {item_id} 使用了不兼容的计分版本：{recorded_version}"
            )
        score = int(scoring_key[option_id])
        recorded_score = record.get("score")
        if recorded_score is not None and recorded_score != score:
            raise ValueError(f"Mussel 题目 {item_id} 的记录分数与0/1规则不一致")
        rows.append(
            {
                "respondent_id": respondent_id,
                "matched_subject_id": str(record.get("matched_subject_id") or ""),
                "item_id": item_id,
                "facet_key": item_map[item_id]["facet_key"],
                "target_dimension_id": item_map[item_id]["target_dimension_id"],
                "selected_option_id": option_id,
                "score": score,
                "scoring_version": MUSSEL_SCORING_VERSION,
            }
        )
    long_frame = pd.DataFrame(rows)
    if long_frame.empty:
        raise ValueError("Mussel 没有有效作答记录")
    observed_ids = set(long_frame["respondent_id"].astype(str))
    if observed_ids != expected_ids:
        raise ValueError("Mussel 与目标 SJT 的被试集合不一致")
    expected_count = len(item_map)
    counts = long_frame.groupby("respondent_id").size()
    if len(counts) != len(expected_ids) or not (counts == expected_count).all():
        raise ValueError("每名被试必须完成全部110道 Mussel 题")
    facet_scores = (
        long_frame.groupby(["respondent_id", "target_dimension_id"])["score"]
        .mean()
        .unstack("target_dimension_id")
        .sort_index()
    )
    return long_frame, facet_scores
