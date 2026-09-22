"""Virtual respondent pool loading and deterministic sampling."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_POOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "virtual_respondents.json"
)
DEFAULT_SELECTION_SEED = 7
DEFAULT_MAX_CONCURRENCY = 5
MAX_ALLOWED_CONCURRENCY = 20
DEFAULT_MAX_RETRIES = 2
MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE = 30
MAX_VIRTUAL_SAMPLE_SIZE = 500
PERSONA_MODE_SUMMARY_PLUS_ITEMS = "summary_plus_items"
PERSONA_MODE_SCORE_PROFILE = "score_profile"
SUPPORTED_PERSONA_MODES = (PERSONA_MODE_SCORE_PROFILE,)
DEFAULT_PERSONA_MODES = SUPPORTED_PERSONA_MODES
SCORE_PROFILE_GENERATOR_VERSION = "tiered-symmetric-quantile-v3"
SCORE_PROFILE_PROMPT_VERSION = "score-tier-sjt-v3"
MATCHED_CONDITION_GENERATOR_VERSION = "matched-normal-shared-sequence-v3"
MATCHED_CONDITION_PROMPT_VERSION = "matched-facet-grouped-score-v2"
MATCHED_CONDITION_SCHEMA_VERSION = 7
DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT = 2
MATCHED_CONDITION_IDS = ("target", "same_domain", "cross_domain")
MATCHED_CONDITION_ROLES = {
    "target": "target",
    "same_domain": "same_domain_non_target",
    "cross_domain": "cross_domain_non_target",
}
MINIMUM_SCORE_TIER_COUNT = 1
MAXIMUM_SCORE_TIER_COUNT = 3
DEFAULT_SPREAD_CAP = 15.0
SCORE_SCALE = [0.0, 100.0]


@lru_cache(maxsize=4)
def _load_pool_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        pool = json.load(handle)
    _validate_pool(pool)
    return pool


def _validate_pool(pool: object) -> None:
    if not isinstance(pool, dict):
        raise ValueError("虚拟被试池必须是对象")
    if pool.get("schema_version") != 1:
        raise ValueError("虚拟被试池 schema_version 必须为 1")

    items = pool.get("items")
    respondents = pool.get("respondents")
    response_scale = pool.get("response_scale")
    if not isinstance(items, list) or not items:
        raise ValueError("虚拟被试池缺少人格题目")
    if not isinstance(respondents, list) or not respondents:
        raise ValueError("虚拟被试池缺少被试记录")
    if not isinstance(response_scale, dict):
        raise ValueError("虚拟被试池缺少作答标签")
    if pool.get("item_count") != len(items):
        raise ValueError("虚拟被试池 item_count 与实际题目数不一致")
    if pool.get("respondent_count") != len(respondents):
        raise ValueError("虚拟被试池 respondent_count 与实际人数不一致")

    item_ids = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("虚拟被试池包含无效人格题目")
        item_id = item.get("item_id")
        statement = item.get("statement")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("人格题目缺少 item_id")
        if not isinstance(statement, str) or not statement:
            raise ValueError(f"人格题目 {item_id} 缺少题干")
        item_ids.append(item_id)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("虚拟被试池包含重复人格题目 ID")

    respondent_ids = []
    for respondent in respondents:
        if not isinstance(respondent, dict):
            raise ValueError("虚拟被试池包含无效被试记录")
        respondent_id = respondent.get("respondent_id")
        values = respondent.get("response_values")
        if not isinstance(respondent_id, str) or not respondent_id:
            raise ValueError("虚拟被试记录缺少 respondent_id")
        if not isinstance(values, list) or len(values) != len(items):
            raise ValueError(
                f"虚拟被试 {respondent_id} 的人格作答数量不正确"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > 5
            for value in values
        ):
            raise ValueError(
                f"虚拟被试 {respondent_id} 包含不在 1-5 范围内的作答"
            )
        respondent_ids.append(respondent_id)
    if len(set(respondent_ids)) != len(respondent_ids):
        raise ValueError("虚拟被试池包含重复 respondent_id")

    for value in range(1, 6):
        label = response_scale.get(str(value))
        if not isinstance(label, str) or not label:
            raise ValueError(f"虚拟被试池缺少作答值 {value} 的文本标签")


def load_virtual_respondent_pool(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """读取并校验项目内置的匿名虚拟被试池。"""

    resolved_path = Path(path or DEFAULT_POOL_PATH).resolve()
    return deepcopy(_load_pool_cached(str(resolved_path)))


def build_virtual_pool_summary(
    pool: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """生成适合交互界面展示的被试池摘要。"""

    resolved_pool = dict(pool) if pool is not None else load_virtual_respondent_pool()
    items = resolved_pool["items"]
    facet_counts = Counter(item["facet_code"] for item in items)
    return {
        "pool_id": resolved_pool["pool_id"],
        "available_count": len(resolved_pool["respondents"]),
        "item_count": len(items),
        "facet_item_counts": dict(facet_counts),
        "source_file": resolved_pool["source"]["file_name"],
        "source_sha256": resolved_pool["source"]["source_sha256"],
        "contains_original_identifiers": False,
        "contains_demographics": False,
    }


def recommend_virtual_sample_size(available_count: int) -> int:
    """Recommend the minimum sample used by the automated development loop."""

    if (
        not isinstance(available_count, int)
        or isinstance(available_count, bool)
        or available_count < 1
    ):
        raise ValueError("available_count 必须是正整数")
    return min(available_count, MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE)


def build_virtual_sample_recommendations(
    available_count: int,
) -> list[dict[str, Any]]:
    """提供快速、平衡和全量三个不重复的样本量档位。"""

    recommended = recommend_virtual_sample_size(available_count)
    candidates = [
        (
            min(30, available_count),
            "quick",
            "最低开发样本",
            (
                "达到自动诊断筛选的最低人数；结果仍属于虚拟开发证据，"
                "不替代真实被试验证。"
            ),
        ),
        (
            min(100, available_count),
            "balanced",
            "平衡开发",
            "兼顾调用成本与初步分布检查，仍可能有明显抽样波动。",
        ),
        (
            available_count,
            "full",
            "全量样本",
            (
                "使用当前被试池的全部人格档案，避免额外的子样本选择波动；"
                "当前池容量不大于300人时推荐使用。"
            ),
        ),
    ]
    recommendations = []
    seen_sizes = set()
    for sample_size, code, label, description in candidates:
        if sample_size in seen_sizes:
            continue
        seen_sizes.add(sample_size)
        recommendations.append(
            {
                "code": code,
                "label": label,
                "sample_size": sample_size,
                "description": description,
                "recommended": sample_size == recommended,
            }
        )
    return recommendations


def build_score_dimension_catalog(
    inventories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten the construct registry into scoreable domain/facet rows."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for inventory in inventories:
        inventory_id = str(inventory.get("inventory_id") or "")
        inventory_name = str(inventory.get("inventory_name") or inventory_id)
        for domain in inventory.get("domains") or []:
            if not isinstance(domain, Mapping):
                continue
            domain_id = str(domain.get("domain_id") or "")
            if not domain_id or domain_id in seen:
                continue
            domain_name = str(domain.get("domain_name") or domain_id)
            domain_name_en = str(domain.get("domain_name_en") or domain_id)
            rows.append(
                {
                    "dimension_id": domain_id,
                    "level": "domain",
                    "inventory_id": inventory_id,
                    "inventory_name": inventory_name,
                    "domain_id": domain_id,
                    "domain_name": domain_name,
                    "domain_name_en": domain_name_en,
                    "facet_name": None,
                    "facet_name_en": None,
                    "definition": None,
                    "high_behavior": None,
                    "low_behavior": None,
                    "display_label": (
                        f"Domain | {domain_name_en}（{domain_name}）"
                    ),
                }
            )
            seen.add(domain_id)
            for facet in domain.get("facets") or []:
                if not isinstance(facet, Mapping):
                    continue
                facet_id = str(facet.get("facet_id") or "")
                if not facet_id or facet_id in seen:
                    continue
                facet_name = str(facet.get("facet_name") or facet_id)
                facet_name_en = str(
                    facet.get("facet_name_en") or facet_name
                )
                rows.append(
                    {
                        "dimension_id": facet_id,
                        "level": "facet",
                        "inventory_id": inventory_id,
                        "inventory_name": inventory_name,
                        "domain_id": domain_id,
                        "domain_name": domain_name,
                        "domain_name_en": domain_name_en,
                        "facet_name": facet_name,
                        "facet_name_en": facet_name_en,
                        "definition": facet.get("definition"),
                        "high_behavior": facet.get("high_behavior"),
                        "low_behavior": facet.get("low_behavior"),
                        "display_label": (
                            "Facet | "
                            f"{domain_name_en} > {facet_name_en}"
                            f"（{facet_name}）"
                        ),
                    }
                )
                seen.add(facet_id)
    if not rows:
        raise ValueError("构念注册表没有可设置的 domain/facet")
    return rows


def score_spec_is_target_related(
    score_spec: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    """Return whether a score is the target or its non-contaminating parent."""

    if score_spec.get("dimension_id") == target.get("dimension_id"):
        return True
    target_level = target.get("level")
    if target_level == "facet":
        return (
            score_spec.get("level") == "domain"
            and score_spec.get("dimension_id") == target.get("domain_id")
        )
    if target_level == "domain":
        return score_spec.get("domain_id") == target.get("domain_id")
    return False


def normalize_score_specs(
    raw_scores: Mapping[str, Any],
    *,
    dimension_catalog: Sequence[Mapping[str, Any]],
    target_dimension_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate user means and attach stable construct metadata."""

    if not isinstance(raw_scores, Mapping):
        raise ValueError("score_means 必须是 dimension_id 到均分的对象")
    catalog = {
        str(row.get("dimension_id")): dict(row)
        for row in dimension_catalog
        if isinstance(row, Mapping) and row.get("dimension_id")
    }
    unknown = sorted(str(key) for key in raw_scores if str(key) not in catalog)
    if unknown:
        raise ValueError("未知的 domain/facet：" + "、".join(unknown))

    specs: list[dict[str, Any]] = []
    for row in dimension_catalog:
        dimension_id = str(row.get("dimension_id") or "")
        if dimension_id not in raw_scores:
            continue
        value = raw_scores[dimension_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < SCORE_SCALE[0]
            or float(value) > SCORE_SCALE[1]
        ):
            raise ValueError(f"{dimension_id} 的均分必须是0到100的有限数值")
        if float(value) in {SCORE_SCALE[0], SCORE_SCALE[1]}:
            raise ValueError(
                f"{dimension_id} 的均分为 {float(value):g} 时无法同时保持"
                "精确均值与非零方差；自动迭代请使用0与100之间的分数"
            )
        specs.append({**dict(row), "mean_score": float(value)})
    if not specs:
        raise ValueError("至少需要填写一个 domain 或 facet 均分")

    targets = list(dict.fromkeys(str(value) for value in target_dimension_ids))
    missing_targets = [target for target in targets if target not in raw_scores]
    if missing_targets:
        raise ValueError(
            "自动迭代必须填写所有题目目标维度："
            + "、".join(missing_targets)
        )
    target_rows = []
    for target_id in targets:
        if target_id not in catalog:
            raise ValueError(f"题目目标维度不在构念注册表中：{target_id}")
        target_rows.append(catalog[target_id])
    for target in target_rows:
        if not any(
            not score_spec_is_target_related(spec, target) for spec in specs
        ):
            raise ValueError(
                f"目标 {target['dimension_id']} 至少需要一个非目标"
                "domain/facet，才能计算虚拟目标特异性"
            )
    return specs


def normalize_score_tiers(
    raw_tiers: object,
    *,
    dimension_catalog: Sequence[Mapping[str, Any]],
    target_dimension_ids: Sequence[str],
    non_target_facet_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate 1-3 tiers whose selected facets share one mean per tier."""

    if not isinstance(raw_tiers, list) or not (
        MINIMUM_SCORE_TIER_COUNT <= len(raw_tiers) <= MAXIMUM_SCORE_TIER_COUNT
    ):
        raise ValueError("score_tiers 必须包含1到3个完整分数档")
    catalog = {
        str(row.get("dimension_id")): dict(row)
        for row in dimension_catalog
        if isinstance(row, Mapping) and row.get("dimension_id")
    }
    target_ids = list(dict.fromkeys(str(value) for value in target_dimension_ids))
    selected_non_targets = (
        list(dict.fromkeys(str(value) for value in non_target_facet_ids))
        if non_target_facet_ids is not None
        else None
    )
    if selected_non_targets is not None:
        selected_ids = [*target_ids, *selected_non_targets]
        unknown = sorted(value for value in selected_ids if value not in catalog)
        if unknown:
            raise ValueError("未知的 facet：" + "、".join(unknown))
        non_facets = sorted(
            value for value in selected_ids if catalog[value].get("level") != "facet"
        )
        if non_facets:
            raise ValueError("分数档只能配置 facet，不能配置 domain：" + "、".join(non_facets))
        if set(target_ids) & set(selected_non_targets):
            raise ValueError("non_target_facet_ids 不能包含目标 facet")
    else:
        selected_ids = []

    normalized_tiers: list[dict[str, Any]] = []
    tier_ids: set[str] = set()
    base_specs: list[dict[str, Any]] | None = None
    for index, raw_tier in enumerate(raw_tiers, start=1):
        if not isinstance(raw_tier, Mapping):
            raise ValueError(f"第{index}个分数档必须是对象")
        tier_id = str(raw_tier.get("tier_id") or f"tier-{index}").strip()
        if not tier_id or tier_id in tier_ids:
            raise ValueError("每个分数档必须有唯一的 tier_id")
        if selected_non_targets is not None:
            value = raw_tier.get("facet_score")
            raw_scores = {dimension_id: value for dimension_id in selected_ids}
        else:
            raw_scores = raw_tier.get("score_means")
            if not isinstance(raw_scores, Mapping) or not raw_scores:
                raise ValueError(
                    "旧分数档必须提供 score_means；新配置请提供 non_target_facet_ids "
                    "及每档 facet_score"
                )
            raw_values = list(raw_scores.values())
            try:
                numeric_values = [float(value) for value in raw_values]
            except (TypeError, ValueError) as exc:
                raise ValueError("旧分数档的 score_means 必须是数值") from exc
            if any(not math.isfinite(value) for value in numeric_values) or len(
                {round(value, 12) for value in numeric_values}
            ) != 1:
                raise ValueError(
                    "旧分数档中所有 facet 分数必须完全相同；请重新配置当前分数档"
                )
            if any(catalog.get(str(key), {}).get("level") != "facet" for key in raw_scores):
                raise ValueError("旧分数档包含 domain 分数；请重新选择非目标 facet")
            if not selected_ids:
                selected_ids = [str(key) for key in raw_scores]
                selected_non_targets = [
                    value for value in selected_ids if value not in set(target_ids)
                ]
            elif set(raw_scores) != set(selected_ids):
                raise ValueError("所有旧分数档必须填写完全相同的 facet 集合")

        tier_specs = normalize_score_specs(
            raw_scores,
            dimension_catalog=dimension_catalog,
            target_dimension_ids=target_dimension_ids,
        )
        if base_specs is None:
            base_specs = [
                {key: deepcopy(value) for key, value in spec.items() if key != "mean_score"}
                for spec in tier_specs
            ]
        facet_score = float(tier_specs[0]["mean_score"])
        normalized_tiers.append(
            {
                "tier_id": tier_id,
                "facet_score": facet_score,
            }
        )
        tier_ids.add(tier_id)
    assert base_specs is not None
    target_id_set = set(target_ids)
    targets = {
        str(row.get("dimension_id")): row
        for row in base_specs
        if str(row.get("dimension_id")) in target_id_set
    }
    for target_id, target in targets.items():
        if target.get("level") != "facet":
            raise ValueError(f"目标 {target_id} 必须是 facet，才能拆分同域与跨域VTS")
        same_domain = [
            spec for spec in base_specs
            if spec.get("level") == "facet"
            and spec.get("dimension_id") != target_id
            and spec.get("domain_id") == target.get("domain_id")
        ]
        cross_domain = [
            spec for spec in base_specs
            if spec.get("level") == "facet"
            and spec.get("domain_id") != target.get("domain_id")
        ]
        if not same_domain or not cross_domain:
            raise ValueError(
                f"目标 {target_id} 必须同时配置至少一个同domain非目标facet和"
                "一个不同domain非目标facet"
            )
    return base_specs, normalized_tiers


def _symmetric_normal_scores(
    sample_size: int,
    *,
    mean_score: float,
    spread_cap: float,
) -> np.ndarray:
    quantiles = np.array(
        [
            NormalDist().inv_cdf((index + 0.5) / sample_size)
            for index in range(sample_size)
        ],
        dtype=float,
    )
    quantiles -= float(quantiles.mean())
    sample_sd = float(quantiles.std(ddof=1))
    if sample_sd <= 0:
        raise ValueError("虚拟被试样本不足以产生非零方差")
    quantiles /= sample_sd
    lower_scale = mean_score / abs(float(quantiles.min()))
    upper_scale = (100.0 - mean_score) / float(quantiles.max())
    actual_spread = min(float(spread_cap), lower_scale, upper_scale)
    if actual_spread <= 0:
        raise ValueError("该均分无法生成非零方差的0-100分数")
    values = mean_score + actual_spread * quantiles
    values += mean_score - float(values.mean())
    return values


def _orthogonal_rank_orders(
    sample_size: int,
    dimension_count: int,
    *,
    seed: int,
    maximum_absolute_correlation: float = 0.10,
) -> np.ndarray:
    if dimension_count == 1:
        return np.arange(sample_size, dtype=int).reshape(sample_size, 1)
    if sample_size < dimension_count + 2:
        raise ValueError(
            "为控制输入维度间相关，虚拟被试人数至少应比已填写维度数多2"
        )
    best_orders: np.ndarray | None = None
    best_maximum = math.inf
    for attempt in range(1000):
        attempt_seed = int.from_bytes(
            sha256(f"{seed}:{attempt}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(attempt_seed)
        random_matrix = rng.normal(size=(sample_size, dimension_count))
        random_matrix -= random_matrix.mean(axis=0, keepdims=True)
        orthogonal, _ = np.linalg.qr(random_matrix, mode="reduced")
        orders = np.empty_like(orthogonal, dtype=int)
        for column in range(dimension_count):
            order = np.argsort(orthogonal[:, column], kind="mergesort")
            orders[order, column] = np.arange(sample_size, dtype=int)
        correlations = np.corrcoef(orders, rowvar=False)
        off_diagonal = np.abs(
            correlations[np.triu_indices(dimension_count, k=1)]
        )
        maximum = float(off_diagonal.max()) if len(off_diagonal) else 0.0
        if maximum < best_maximum:
            best_maximum = maximum
            best_orders = orders.copy()
        if maximum <= maximum_absolute_correlation + 1e-12:
            return orders
    raise ValueError(
        "无法在当前样本量下把输入维度间的最大绝对秩相关控制在"
        f"{maximum_absolute_correlation:.2f}以内（最佳={best_maximum:.3f}）；"
        "请增加虚拟被试人数或减少已填写维度"
    )


def generate_score_respondent_refs(
    sample_size: int,
    score_specs: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    spread_cap: float = DEFAULT_SPREAD_CAP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate deterministic 0-100 profiles with exact requested means."""

    if (
        not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or sample_size < MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE
        or sample_size > MAX_VIRTUAL_SAMPLE_SIZE
    ):
        raise ValueError(
            "自动迭代的 sample_size 必须是"
            f"{MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE} 到 "
            f"{MAX_VIRTUAL_SAMPLE_SIZE} 之间的整数"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")
    if (
        isinstance(spread_cap, bool)
        or not isinstance(spread_cap, (int, float))
        or not math.isfinite(float(spread_cap))
        or float(spread_cap) <= 0
    ):
        raise ValueError("spread_cap 必须是正的有限数值")
    specs = [dict(spec) for spec in score_specs]
    if not specs:
        raise ValueError("score_specs 不能为空")
    dimension_ids = [str(spec.get("dimension_id") or "") for spec in specs]
    if any(not value for value in dimension_ids) or len(set(dimension_ids)) != len(
        dimension_ids
    ):
        raise ValueError("score_specs 包含空或重复的 dimension_id")

    rank_orders = _orthogonal_rank_orders(
        sample_size,
        len(specs),
        seed=seed,
    )
    matrix = np.empty((sample_size, len(specs)), dtype=float)
    for column, spec in enumerate(specs):
        base = _symmetric_normal_scores(
            sample_size,
            mean_score=float(spec["mean_score"]),
            spread_cap=float(spread_cap),
        )
        matrix[:, column] = base[rank_orders[:, column]]
    # Boundary means can produce floating-point noise just outside 0-100.
    matrix = np.clip(matrix, SCORE_SCALE[0], SCORE_SCALE[1])

    refs = [
        {
            "respondent_id": f"score-{index + 1:04d}",
            "score_values": {
                dimension_id: float(matrix[index, column])
                for column, dimension_id in enumerate(dimension_ids)
            },
        }
        for index in range(sample_size)
    ]
    actual_correlations = np.corrcoef(rank_orders, rowvar=False)
    correlation_rows = []
    for left in range(len(specs)):
        for right in range(left + 1, len(specs)):
            correlation_rows.append(
                {
                    "dimension_a": dimension_ids[left],
                    "dimension_b": dimension_ids[right],
                    "spearman_rho": float(actual_correlations[left, right]),
                }
            )
    diagnostics = {
        "generator_version": SCORE_PROFILE_GENERATOR_VERSION,
        "sample_size": sample_size,
        "seed": seed,
        "spread_cap": float(spread_cap),
        "dimensions": [
            {
                "dimension_id": dimension_id,
                "requested_mean": float(specs[column]["mean_score"]),
                "actual_mean": float(matrix[:, column].mean()),
                "actual_sample_sd": float(matrix[:, column].std(ddof=1)),
                "actual_minimum": float(matrix[:, column].min()),
                "actual_maximum": float(matrix[:, column].max()),
            }
            for column, dimension_id in enumerate(dimension_ids)
        ],
        "pairwise_rank_correlations": correlation_rows,
        "maximum_absolute_rank_correlation": max(
            (abs(row["spearman_rho"]) for row in correlation_rows),
            default=0.0,
        ),
    }
    return refs, diagnostics


def generate_tiered_score_respondent_refs(
    sample_size_per_tier: int,
    score_specs: Sequence[Mapping[str, Any]],
    score_tiers: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    spread_cap: float = DEFAULT_SPREAD_CAP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate tiered respondents and report, but never gate on, input rho."""

    if not 1 <= len(score_tiers) <= MAXIMUM_SCORE_TIER_COUNT:
        raise ValueError("score_tiers 必须包含1到3个分数档")
    dimension_ids = [str(spec["dimension_id"]) for spec in score_specs]
    all_refs: list[dict[str, Any]] = []
    tier_diagnostics: list[dict[str, Any]] = []
    for tier_index, tier in enumerate(score_tiers, start=1):
        tier_id = str(tier.get("tier_id") or f"tier-{tier_index}")
        facet_score = tier.get("facet_score")
        if (
            isinstance(facet_score, bool)
            or not isinstance(facet_score, (int, float))
            or not math.isfinite(float(facet_score))
            or not SCORE_SCALE[0] < float(facet_score) < SCORE_SCALE[1]
        ):
            raise ValueError(f"分数档 {tier_id} 的 facet_score 必须大于0且小于100")
        tier_specs = [
            {**dict(spec), "mean_score": float(facet_score)}
            for spec in score_specs
        ]
        tier_seed = int.from_bytes(
            sha256(f"{seed}:{tier_id}".encode("utf-8")).digest()[:8], "big"
        ) % (2**31)
        refs, diagnostics = generate_score_respondent_refs(
            sample_size_per_tier,
            tier_specs,
            seed=tier_seed,
            spread_cap=spread_cap,
        )
        for reference in refs:
            local_id = str(reference["respondent_id"])
            all_refs.append(
                {
                    **reference,
                    "respondent_id": f"{tier_id}-{local_id}",
                    "tier_id": tier_id,
                    "tier_respondent_id": local_id,
                }
            )
        tier_diagnostics.append({"tier_id": tier_id, **diagnostics})

    matrix = np.asarray(
        [
            [float(ref["score_values"][dimension_id]) for dimension_id in dimension_ids]
            for ref in all_refs
        ],
        dtype=float,
    )
    correlation_rows: list[dict[str, Any]] = []
    if len(dimension_ids) > 1:
        from scipy import stats as scipy_stats

        correlations = np.asarray(
            scipy_stats.spearmanr(matrix, axis=0).statistic,
            dtype=float,
        )
        for left in range(len(dimension_ids)):
            for right in range(left + 1, len(dimension_ids)):
                correlation_rows.append(
                    {
                        "dimension_a": dimension_ids[left],
                        "dimension_b": dimension_ids[right],
                        "spearman_rho": float(correlations[left, right]),
                    }
                )
    return all_refs, {
        "generator_version": SCORE_PROFILE_GENERATOR_VERSION,
        "sample_size_per_tier": sample_size_per_tier,
        "tier_count": len(score_tiers),
        "total_sample_size": len(all_refs),
        "seed": seed,
        "spread_cap": float(spread_cap),
        "tiers": tier_diagnostics,
        "merged_pairwise_rank_correlations": correlation_rows,
        "merged_maximum_absolute_rank_correlation": max(
            (abs(float(row["spearman_rho"])) for row in correlation_rows),
            default=0.0,
        ),
        "input_correlation_filtering_authority": False,
        "input_correlation_note": "输入相关仅作诊断，不拒绝运行或参与题目筛选。",
    }


def normalize_matched_conditions(
    raw_conditions: object,
    *,
    dimension_catalog: Sequence[Mapping[str, Any]],
    target_dimension_id: str,
) -> list[dict[str, Any]]:
    """Normalize three fixed arms, allowing multiple independently matched groups per arm.

    The target arm has exactly one group. Same-domain and cross-domain arms may
    contain any number of facet groups; each group receives its own condition ID
    so its rho is estimated independently before the arm-level MAX is taken.
    """

    if not isinstance(raw_conditions, list) or len(raw_conditions) != 3:
        raise ValueError("conditions 必须恰好包含 target、same_domain、cross_domain 三组")
    catalog = {
        str(row.get("dimension_id")): dict(row)
        for row in dimension_catalog
        if isinstance(row, Mapping) and row.get("dimension_id")
    }
    target_id = str(target_dimension_id or "")
    target = catalog.get(target_id)
    if not target or target.get("level") != "facet":
        raise ValueError("target_dimension_id 必须是构念注册表中的 facet")
    normalized: list[dict[str, Any]] = []
    seen_dimensions: set[str] = set()
    seen_group_ids: set[str] = set()
    for expected_id in MATCHED_CONDITION_IDS:
        matches = [
            row for row in raw_conditions
            if isinstance(row, Mapping)
            and str(row.get("condition_id") or "") == expected_id
        ]
        if len(matches) != 1:
            raise ValueError(f"conditions 必须恰好包含 {expected_id} 一组")
        row = matches[0]
        condition_id = str(row.get("condition_id"))
        role = str(row.get("role") or "")
        if role != MATCHED_CONDITION_ROLES[expected_id]:
            raise ValueError(f"{condition_id} 的 role 必须为 {MATCHED_CONDITION_ROLES[expected_id]}")
        raw_groups = row.get("groups")
        if raw_groups is None:
            # Backward-compatible input: one facet directly on each arm.
            raw_groups = [row]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(f"{condition_id} 至少需要一个 facet group")
        if expected_id == "target" and len(raw_groups) != 1:
            raise ValueError("target 臂必须只有一个 facet group")
        groups: list[dict[str, Any]] = []
        for index, raw_group in enumerate(raw_groups, start=1):
            if not isinstance(raw_group, Mapping):
                raise ValueError(f"{condition_id} 的 group 必须是对象")
            dimension_id = str(raw_group.get("dimension_id") or "")
            group_id = str(raw_group.get("group_id") or dimension_id or f"group_{index}")
            if not dimension_id:
                raise ValueError(f"{condition_id} group 缺少 dimension_id")
            if group_id in seen_group_ids:
                raise ValueError("group_id 不能重复")
            dimension = catalog.get(dimension_id)
            if not dimension or dimension.get("level") != "facet":
                raise ValueError(f"{condition_id} 必须选择有效的 facet")
            if dimension_id in seen_dimensions:
                raise ValueError("同一个 facet 不能重复配置到多个 matched group")
            if expected_id == "target" and dimension_id != target_id:
                raise ValueError("target 条件必须使用题目目标 facet")
            if expected_id != "target" and dimension_id == target_id:
                raise ValueError("非目标条件不能重复使用目标 facet")
            if expected_id == "same_domain" and dimension.get("domain_id") != target.get("domain_id"):
                raise ValueError("same_domain facet 必须与目标 facet 属于同一 domain")
            if expected_id == "cross_domain" and dimension.get("domain_id") == target.get("domain_id"):
                raise ValueError("cross_domain facet 必须来自不同 domain")
            condition_group_id = "target" if expected_id == "target" else f"{expected_id}__{group_id}"
            groups.append({
                "group_id": group_id,
                "condition_id": condition_group_id,
                "arm_id": expected_id,
                "role": role,
                "dimension_id": dimension_id,
                "domain_id": dimension.get("domain_id"),
                "domain_name": dimension.get("domain_name"),
                "domain_name_en": dimension.get("domain_name_en"),
                "facet_name": dimension.get("facet_name"),
                "facet_name_en": dimension.get("facet_name_en"),
                "definition": dimension.get("definition"),
                "high_behavior": dimension.get("high_behavior"),
                "low_behavior": dimension.get("low_behavior"),
            })
            seen_dimensions.add(dimension_id)
            seen_group_ids.add(group_id)
        arm = {
            "condition_id": expected_id,
            "arm_id": expected_id,
            "role": role,
            "groups": groups,
            "group_count": len(groups),
        }
        # Keep one-group metadata available to older displays and manifests.
        if len(groups) == 1:
            arm.update({key: groups[0].get(key) for key in (
                "dimension_id", "domain_id", "domain_name", "domain_name_en",
                "facet_name", "facet_name_en", "definition", "high_behavior",
                "low_behavior",
            )})
        normalized.append(arm)
    return normalized


def flatten_matched_condition_groups(
    conditions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one metadata row per matched facet group in fixed arm order."""

    flattened: list[dict[str, Any]] = []
    arm_rows = {
        str(arm.get("arm_id") or arm.get("condition_id") or ""): arm
        for arm in conditions
        if isinstance(arm, Mapping)
    }
    ordered_arms = [arm_rows[arm_id] for arm_id in MATCHED_CONDITION_IDS if arm_id in arm_rows]
    ordered_arms.extend(
        arm for arm_id, arm in arm_rows.items() if arm_id not in MATCHED_CONDITION_IDS
    )
    for arm in ordered_arms:
        arm_id = str(arm.get("arm_id") or arm.get("condition_id") or "")
        role = str(arm.get("role") or MATCHED_CONDITION_ROLES.get(arm_id, ""))
        raw_groups = arm.get("groups")
        legacy_flat_arm = not isinstance(raw_groups, list)
        if not isinstance(raw_groups, list) or not raw_groups:
            raw_groups = [arm]
        for group in raw_groups:
            if not isinstance(group, Mapping):
                continue
            row = dict(group)
            row["arm_id"] = arm_id
            row["role"] = str(group.get("role") or role)
            row["group_id"] = str(group.get("group_id") or group.get("dimension_id") or arm_id)
            row["condition_id"] = str(
                group.get("condition_id")
                or (arm_id if legacy_flat_arm else (arm_id if arm_id == "target" else f"{arm_id}__{row['group_id']}"))
            )
            flattened.append(row)
    return flattened


def _matched_normal_scores(
    sample_size: int,
    *,
    mean_score: float,
    standard_deviation: float,
    seed: int,
) -> np.ndarray:
    """Generate one exact-mean/SD normal vector and reject out-of-range draws."""

    if (
        not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or sample_size < MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE
        or sample_size > MAX_VIRTUAL_SAMPLE_SIZE
    ):
        raise ValueError(
            f"sample_size_per_condition 必须是 {MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE} 到 "
            f"{MAX_VIRTUAL_SAMPLE_SIZE} 之间的整数"
        )
    if not math.isfinite(float(mean_score)) or not 0.0 < float(mean_score) < 100.0:
        raise ValueError("score_distribution.mean 必须大于0且小于100")
    if not math.isfinite(float(standard_deviation)) or float(standard_deviation) <= 0:
        raise ValueError("score_distribution.sd 必须是正数")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")
    for attempt in range(1000):
        attempt_seed = int.from_bytes(
            sha256(f"{seed}:matched-normal:{attempt}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(attempt_seed)
        raw = rng.normal(0.0, 1.0, size=sample_size)
        raw -= float(raw.mean())
        raw_sd = float(raw.std(ddof=1))
        if raw_sd <= 0 or not math.isfinite(raw_sd):
            continue
        values = float(mean_score) + float(standard_deviation) * raw / raw_sd
        if (
            np.all(np.isfinite(values))
            and float(values.min()) >= SCORE_SCALE[0]
            and float(values.max()) <= SCORE_SCALE[1]
        ):
            return values.astype(float)
    raise ValueError(
        "当前均值和标准差无法生成全部落在0-100范围内的正态分数；"
        "请减小 SD 或调整均值"
    )


def generate_matched_condition_respondent_refs(
    sample_size_per_condition: int,
    conditions: Sequence[Mapping[str, Any]],
    *,
    mean_score: float = 50.0,
    standard_deviation: float = 15.0,
    seed: int = DEFAULT_SELECTION_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate fixed arms whose individual facet groups share one score sequence."""

    if len(conditions) != 3:
        raise ValueError("conditions 必须包含三个固定臂")
    ordered = {str(row.get("condition_id")): dict(row) for row in conditions}
    if set(ordered) != set(MATCHED_CONDITION_IDS):
        raise ValueError("conditions 顶层必须是 target、same_domain、cross_domain")
    groups = flatten_matched_condition_groups(conditions)
    if not groups or groups[0].get("arm_id") != "target":
        raise ValueError("conditions 缺少 target facet group")
    if sum(row.get("arm_id") == "target" for row in groups) != 1:
        raise ValueError("target 臂必须只有一个 facet group")
    z_values = _matched_normal_scores(
        sample_size_per_condition,
        mean_score=float(mean_score),
        standard_deviation=float(standard_deviation),
        seed=seed,
    )
    refs: list[dict[str, Any]] = []
    condition_diagnostics: list[dict[str, Any]] = []
    for row in groups:
        condition_id = str(row["condition_id"])
        dimension_id = str(row.get("dimension_id") or "")
        condition_refs = []
        for index, value in enumerate(z_values, start=1):
            matched_subject_id = f"matched-{index:04d}"
            condition_refs.append({
                "respondent_id": f"{condition_id}-{matched_subject_id}",
                "condition_id": condition_id,
                "arm_id": row.get("arm_id"),
                "group_id": row.get("group_id"),
                "matched_subject_id": matched_subject_id,
                "active_dimension_id": dimension_id,
                "score_values": {dimension_id: float(value)},
            })
        refs.extend(condition_refs)
        condition_diagnostics.append({
            "condition_id": condition_id,
            "arm_id": row.get("arm_id"),
            "group_id": row.get("group_id"),
            "dimension_id": dimension_id,
            "sample_size": len(condition_refs),
            "actual_mean": float(z_values.mean()),
            "actual_sample_sd": float(z_values.std(ddof=1)),
            "actual_minimum": float(z_values.min()),
            "actual_maximum": float(z_values.max()),
        })
    return refs, {
        "generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
        "sampling_design": "matched_facet_conditions",
        "sample_size_per_condition": int(sample_size_per_condition),
        "condition_count": 3,
        "group_count": len(groups),
        "total_sample_size": len(refs),
        "seed": int(seed),
        "score_distribution": {
            "family": "normal",
            "mean": float(mean_score),
            "sd": float(standard_deviation),
        },
        "conditions": condition_diagnostics,
        "matched_score_sequence": [float(value) for value in z_values],
        "matched_sequence_exact": True,
        "input_correlation_filtering_authority": False,
    }


def build_matched_condition_sample_config(
    sample_size_per_condition: int,
    *,
    conditions: Sequence[Mapping[str, Any]],
    generation_diagnostics: Mapping[str, Any],
    mean_score: float = 50.0,
    standard_deviation: float = 15.0,
    seed: int = DEFAULT_SELECTION_SEED,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Build the matched-facet configuration with one target-form retest."""

    if not 1 <= int(max_concurrency) <= MAX_ALLOWED_CONCURRENCY:
        raise ValueError(f"max_concurrency 必须是1到{MAX_ALLOWED_CONCURRENCY}之间的整数")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 0 <= max_retries <= 5:
        raise ValueError("max_retries 必须是0到5之间的整数")
    normalized_conditions = [deepcopy(dict(row)) for row in conditions]
    group_count = len(flatten_matched_condition_groups(normalized_conditions))
    if group_count < 3:
        raise ValueError("matched conditions 至少需要三个 facet group")
    serialized = json.dumps({
        "sampling_design": "matched_facet_conditions",
        "sample_size_per_condition": sample_size_per_condition,
        "conditions": normalized_conditions,
        "score_distribution": {"family": "normal", "mean": mean_score, "sd": standard_deviation},
        "seed": seed,
        "generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    profile_id = "matched-facet-" + sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": MATCHED_CONDITION_SCHEMA_VERSION,
        "pool_id": profile_id,
        "pool_ref": None,
        "source_file": None,
        "source_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
        "available_count": MAX_VIRTUAL_SAMPLE_SIZE,
        "sample_size": int(sample_size_per_condition) * group_count,
        "sample_size_per_condition": int(sample_size_per_condition),
        "condition_count": 3,
        "group_count": group_count,
        "recommended_sample_size": MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE,
        "automatic_selection_minimum_sample_size": MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE,
        "seed": int(seed),
        "max_concurrency": int(max_concurrency),
        "max_retries": int(max_retries),
        "persona_modes": [PERSONA_MODE_SCORE_PROFILE],
        "sampling_design": "matched_facet_conditions",
        "selection_strategy": "deterministic_matched_normal_generation",
        "persona_method": "MatchedFacetConditions",
        "score_distribution": {"family": "normal", "mean": float(mean_score), "sd": float(standard_deviation)},
        "conditions": normalized_conditions,
        "score_scale": list(SCORE_SCALE),
        "generator_version": MATCHED_CONDITION_GENERATOR_VERSION,
        "prompt_version": MATCHED_CONDITION_PROMPT_VERSION,
        "generation_diagnostics": deepcopy(dict(generation_diagnostics)),
        # Pin the virtual-respondent model in the run configuration.  This
        # lets the main CLI use a separate model from the authoring agents and
        # keeps resumed runs reproducible even if .env changes later.
        "model_id": (
            os.getenv("VIRTUAL_RESPONDENT_MODEL_ID")
            or os.getenv("MODEL_ID")
            or "deepseek-v4-flash"
        ),
        "ipip_neo_reference_enabled": True,
        "ipip_neo_in_main_iteration": False,
        "response_count_per_respondent_item": 1,
        # The primary administration remains the sole authority for item-level
        # screening.  One additional target-only administration supplies the
        # whole-form virtual test-retest stability estimate.
        "target_form_administration_count": (
            DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT
        ),
    }


def matched_condition_sample_is_current(config: object, respondents: object) -> bool:
    """Validate the active matched-facet and target-retest contract."""

    if not isinstance(config, Mapping) or not isinstance(respondents, list):
        return False
    if (
        config.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION
        or config.get("sampling_design") != "matched_facet_conditions"
        or config.get("condition_count") != 3
        or config.get("persona_modes") != [PERSONA_MODE_SCORE_PROFILE]
        or config.get("generator_version") != MATCHED_CONDITION_GENERATOR_VERSION
        or config.get("prompt_version") != MATCHED_CONDITION_PROMPT_VERSION
        or config.get("response_count_per_respondent_item") != 1
        or config.get("target_form_administration_count")
        != DEFAULT_TARGET_FORM_ADMINISTRATION_COUNT
        or config.get("sample_size") != len(respondents)
        or config.get("sample_size_per_condition") is None
    ):
        return False
    n = config.get("sample_size_per_condition")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return False
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or {row.get("condition_id") for row in conditions if isinstance(row, Mapping)} != set(MATCHED_CONDITION_IDS):
        return False
    groups = flatten_matched_condition_groups(conditions)
    expected_conditions = {str(row.get("condition_id")): str(row.get("dimension_id")) for row in groups}
    group_ids = tuple(expected_conditions)
    if config.get("group_count") != len(group_ids) or len(respondents) != n * len(group_ids):
        return False
    grouped: dict[str, dict[str, float]] = {condition_id: {} for condition_id in group_ids}
    respondent_ids: set[str] = set()
    for reference in respondents:
        if not isinstance(reference, Mapping):
            return False
        condition_id = str(reference.get("condition_id") or "")
        matched_id = str(reference.get("matched_subject_id") or "")
        respondent_id = str(reference.get("respondent_id") or "")
        values = reference.get("score_values")
        if condition_id not in expected_conditions or not matched_id or not respondent_id or respondent_id in respondent_ids:
            return False
        if respondent_id != f"{condition_id}-{matched_id}" or not isinstance(values, Mapping) or set(values) != {expected_conditions[condition_id]}:
            return False
        value = next(iter(values.values()))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
            return False
        grouped[condition_id][matched_id] = float(value)
        respondent_ids.add(respondent_id)
    if any(len(group) != n for group in grouped.values()):
        return False
    reference_ids = set(grouped["target"])
    if any(set(group) != reference_ids for group in grouped.values()):
        return False
    target_values = grouped["target"]
    return all(group == target_values for group in grouped.values())


def select_virtual_respondent_refs(
    pool: Mapping[str, Any],
    sample_size: int,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
) -> list[dict[str, Any]]:
    """按固定随机种子选择被试，仅把轻量引用写入工作流 State。"""

    respondents = pool.get("respondents")
    if not isinstance(respondents, list) or not respondents:
        raise ValueError("虚拟被试池为空")
    available_count = len(respondents)
    if (
        not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or sample_size < 1
        or sample_size > available_count
    ):
        raise ValueError(
            f"sample_size 必须是 1 到 {available_count} 之间的整数"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")

    if sample_size == available_count:
        selected_indexes = list(range(available_count))
    else:
        selected_indexes = random.Random(seed).sample(
            range(available_count),
            sample_size,
        )
    return [
        {
            "respondent_id": respondents[index]["respondent_id"],
            "pool_index": index,
        }
        for index in selected_indexes
    ]


def build_virtual_sample_config(
    pool: Mapping[str, Any],
    sample_size: int,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    persona_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """记录虚拟样本选择的可复现配置。"""

    summary = build_virtual_pool_summary(pool)
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
    resolved_persona_modes = list(
        DEFAULT_PERSONA_MODES if persona_modes is None else persona_modes
    )
    if not resolved_persona_modes:
        raise ValueError("persona_modes 不能为空")
    if (
        len(set(resolved_persona_modes)) != len(resolved_persona_modes)
        or any(
            mode not in SUPPORTED_PERSONA_MODES
            for mode in resolved_persona_modes
        )
    ):
        raise ValueError(
            "persona_modes 只能包含不重复的 summary_plus_items"
        )
    recommended = recommend_virtual_sample_size(summary["available_count"])
    return {
        "pool_id": summary["pool_id"],
        "pool_ref": str(DEFAULT_POOL_PATH),
        "source_file": summary["source_file"],
        "source_sha256": summary["source_sha256"],
        "available_count": summary["available_count"],
        "sample_size": sample_size,
        "recommended_sample_size": recommended,
        "automatic_selection_minimum_sample_size": (
            MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE
        ),
        "seed": seed,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "persona_modes": resolved_persona_modes,
        "selection_strategy": (
            "all_in_source_order"
            if sample_size == summary["available_count"]
            else "simple_random_without_replacement"
        ),
        "persona_method": "Liu-SummaryPlusItems",
    }


def build_score_virtual_sample_config(
    sample_size_per_tier: int,
    *,
    score_specs: Sequence[Mapping[str, Any]],
    score_tiers: Sequence[Mapping[str, Any]],
    generation_diagnostics: Mapping[str, Any],
    non_target_facet_ids: Sequence[str] | None = None,
    seed: int = DEFAULT_SELECTION_SEED,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    spread_cap: float = DEFAULT_SPREAD_CAP,
) -> dict[str, Any]:
    """Build the reproducible configuration for score-conditioned personas."""

    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency < 1
        or max_concurrency > MAX_ALLOWED_CONCURRENCY
    ):
        raise ValueError(
            f"max_concurrency 必须是1到{MAX_ALLOWED_CONCURRENCY}之间的整数"
        )
    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 0
        or max_retries > 5
    ):
        raise ValueError("max_retries 必须是0到5之间的整数")
    if not 1 <= len(score_tiers) <= MAXIMUM_SCORE_TIER_COUNT:
        raise ValueError("score_tiers 必须包含1到3个分数档")
    resolved_non_targets = list(
        dict.fromkeys(
            str(value)
            for value in (
                non_target_facet_ids
                if non_target_facet_ids is not None
                else [spec.get("dimension_id") for spec in score_specs[1:]]
            )
            if value
        )
    )
    serialized = json.dumps(
        {
            "score_specs": list(score_specs),
            "score_tiers": list(score_tiers),
            "non_target_facet_ids": resolved_non_targets,
            "sample_size_per_tier": sample_size_per_tier,
            "seed": seed,
            "spread_cap": float(spread_cap),
            "generator_version": SCORE_PROFILE_GENERATOR_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    profile_id = "score-profile-" + sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": 4,
        "pool_id": profile_id,
        "pool_ref": None,
        "source_file": None,
        "source_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
        "available_count": MAX_VIRTUAL_SAMPLE_SIZE,
        "sample_size": sample_size_per_tier * len(score_tiers),
        "sample_size_per_tier": sample_size_per_tier,
        "tier_count": len(score_tiers),
        "recommended_sample_size": MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE,
        "automatic_selection_minimum_sample_size": (
            MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE
        ),
        "seed": seed,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "persona_modes": [PERSONA_MODE_SCORE_PROFILE],
        "selection_strategy": "deterministic_score_profile_generation",
        "persona_method": "ExplicitDomainFacetScores",
        "score_specs": [deepcopy(dict(spec)) for spec in score_specs],
        "score_tiers": [deepcopy(dict(tier)) for tier in score_tiers],
        "non_target_facet_ids": resolved_non_targets,
        "score_scale": list(SCORE_SCALE),
        "spread_cap": float(spread_cap),
        "generator_version": SCORE_PROFILE_GENERATOR_VERSION,
        "prompt_version": SCORE_PROFILE_PROMPT_VERSION,
        "generation_diagnostics": deepcopy(dict(generation_diagnostics)),
        "neo_ffi_in_main_iteration": False,
        "response_count_per_respondent_item": 1,
    }


def score_virtual_sample_is_current(
    config: object,
    respondents: object,
) -> bool:
    """Validate the complete in-memory contract for the active protocol."""

    if not isinstance(config, Mapping) or not isinstance(respondents, list):
        return False
    specs = config.get("score_specs")
    if not isinstance(specs, list) or not specs:
        return False
    dimension_ids = [
        str(spec.get("dimension_id") or "")
        for spec in specs
        if isinstance(spec, Mapping)
    ]
    if (
        len(dimension_ids) != len(specs)
        or any(not value for value in dimension_ids)
        or len(set(dimension_ids)) != len(dimension_ids)
        or config.get("schema_version") != 4
        or config.get("persona_modes") != [PERSONA_MODE_SCORE_PROFILE]
        or config.get("prompt_version") != SCORE_PROFILE_PROMPT_VERSION
        or config.get("generator_version") != SCORE_PROFILE_GENERATOR_VERSION
        or config.get("score_scale") != list(SCORE_SCALE)
        or config.get("neo_ffi_in_main_iteration") is not False
        or config.get("response_count_per_respondent_item") != 1
        or config.get("sample_size") != len(respondents)
        or not respondents
    ):
        return False
    tiers = config.get("score_tiers")
    if not isinstance(tiers, list) or not 1 <= len(tiers) <= MAXIMUM_SCORE_TIER_COUNT:
        return False
    tier_ids = {str(tier.get("tier_id")) for tier in tiers if isinstance(tier, Mapping)}
    sample_size_per_tier = config.get("sample_size_per_tier")
    if (
        len(tier_ids) != len(tiers)
        or config.get("tier_count") != len(tiers)
        or not isinstance(sample_size_per_tier, int)
        or isinstance(sample_size_per_tier, bool)
        or sample_size_per_tier < 1
        or config.get("sample_size") != sample_size_per_tier * len(tiers)
    ):
        return False
    expected = set(dimension_ids)
    non_targets = config.get("non_target_facet_ids")
    if (
        not isinstance(non_targets, list)
        or not non_targets
        or any(str(value) not in expected for value in non_targets)
        or any(
            not isinstance(tier, Mapping)
            or isinstance(tier.get("facet_score"), bool)
            or not isinstance(tier.get("facet_score"), (int, float))
            for tier in tiers
        )
    ):
        return False
    tier_counts = {tier_id: 0 for tier_id in tier_ids}
    for reference in respondents:
        if not isinstance(reference, Mapping) or not isinstance(
            reference.get("respondent_id"), str
        ):
            return False
        tier_counts[str(reference.get("tier_id"))] += 1
        values = reference.get("score_values")
        if (
            not isinstance(values, Mapping)
            or set(values) != expected
            or reference.get("tier_id") not in tier_ids
        ):
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not SCORE_SCALE[0] <= float(value) <= SCORE_SCALE[1]
            for value in values.values()
        ):
            return False
    return all(count == sample_size_per_tier for count in tier_counts.values())


def resolve_virtual_respondent_profiles(
    respondent_refs: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Any] | None = None,
    *,
    score_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """把 State 中的轻量引用解析为后续模型调用所需的人格档案。"""

    if respondent_refs and all(
        isinstance(reference, Mapping)
        and isinstance(reference.get("score_values"), Mapping)
        for reference in respondent_refs
    ):
        specs = [dict(spec) for spec in (score_specs or [])]
        if all(
            reference.get("condition_id") == "target"
            or str(reference.get("condition_id") or "").startswith(("same_domain__", "cross_domain__"))
            for reference in respondent_refs
        ):
            profiles = []
            seen: set[str] = set()
            for reference in respondent_refs:
                respondent_id = reference.get("respondent_id")
                condition_id = reference.get("condition_id")
                matched_subject_id = reference.get("matched_subject_id")
                values = reference.get("score_values")
                if not isinstance(respondent_id, str) or not respondent_id or respondent_id in seen:
                    raise ValueError("匹配 facet 虚拟被试缺少或重复 respondent_id")
                if not (
                    condition_id == "target"
                    or str(condition_id or "").startswith(("same_domain__", "cross_domain__"))
                ) or not isinstance(matched_subject_id, str) or not isinstance(values, Mapping) or len(values) != 1:
                    raise ValueError(f"虚拟被试 {respondent_id} 的匹配条件档案无效")
                dimension_id, value = next(iter(values.items()))
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not SCORE_SCALE[0] <= float(value) <= SCORE_SCALE[1]
                ):
                    raise ValueError(f"虚拟被试 {respondent_id} 的 facet 分数无效")
                profiles.append({
                    "respondent_id": respondent_id,
                    "condition_id": condition_id,
                    "arm_id": reference.get("arm_id") or ("target" if condition_id == "target" else str(condition_id).split("__", 1)[0]),
                    "group_id": reference.get("group_id") or ("target" if condition_id == "target" else str(condition_id).split("__", 1)[-1]),
                    "matched_subject_id": matched_subject_id,
                    "active_dimension_id": str(dimension_id),
                    "score_values": {str(dimension_id): float(value)},
                    "score_specs": deepcopy(specs),
                })
                seen.add(respondent_id)
            return profiles
        expected = {str(spec.get("dimension_id")) for spec in specs}
        if not expected:
            raise ValueError("分数型虚拟被试缺少 score_specs")
        profiles = []
        seen: set[str] = set()
        for reference in respondent_refs:
            respondent_id = reference.get("respondent_id")
            values = reference.get("score_values")
            if not isinstance(respondent_id, str) or not respondent_id:
                raise ValueError("分数型虚拟被试缺少 respondent_id")
            if respondent_id in seen:
                raise ValueError("分数型虚拟被试包含重复 respondent_id")
            if set(values) != expected:
                raise ValueError(
                    f"虚拟被试 {respondent_id} 的分数维度与 score_specs 不一致"
                )
            normalized: dict[str, float] = {}
            for dimension_id, value in values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not SCORE_SCALE[0] <= float(value) <= SCORE_SCALE[1]
                ):
                    raise ValueError(
                        f"虚拟被试 {respondent_id} 的 {dimension_id} 分数无效"
                    )
                normalized[str(dimension_id)] = float(value)
            profiles.append(
                {
                    "respondent_id": respondent_id,
                    "tier_id": reference.get("tier_id"),
                    "score_values": normalized,
                    "score_specs": deepcopy(specs),
                }
            )
            seen.add(respondent_id)
        return profiles

    resolved_pool = pool or load_virtual_respondent_pool()
    respondents = resolved_pool["respondents"]
    items = resolved_pool["items"]
    response_scale = resolved_pool["response_scale"]
    profiles = []
    for respondent_ref in respondent_refs:
        pool_index = respondent_ref.get("pool_index")
        respondent_id = respondent_ref.get("respondent_id")
        if (
            not isinstance(pool_index, int)
            or isinstance(pool_index, bool)
            or pool_index < 0
            or pool_index >= len(respondents)
        ):
            raise ValueError("虚拟被试引用包含无效 pool_index")
        respondent = respondents[pool_index]
        if respondent["respondent_id"] != respondent_id:
            raise ValueError(
                f"虚拟被试引用 {respondent_id!r} 与被试池不一致"
            )
        personality_items = []
        for item, value in zip(items, respondent["response_values"]):
            personality_items.append(
                {
                    **item,
                    "response_value": value,
                    "response_label": response_scale[str(value)],
                }
            )
        profiles.append(
            {
                "respondent_id": respondent_id,
                "personality_items": personality_items,
            }
        )
    return profiles
