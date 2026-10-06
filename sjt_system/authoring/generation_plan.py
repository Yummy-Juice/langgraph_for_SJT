"""Two-way specification table and compact item skeletons."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from collections import Counter
from typing import Any

from sjt_system.authoring.situation_space import (
    BlueprintAgentOutput,
    BlueprintRowDraft,
    INCREMENTAL_CANDIDATES_PER_CELL,
    FacetExpansion,
    expansion_cache_path,
    load_facet_expansion,
)


ANCHOR_LEVELS = ("low", "medium_low", "medium_high", "high")
GENERATION_BLUEPRINT_VERSION = 8
EXPANSION_SITUATION_BUFFER = 10


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def resolve_facet_item_counts(
    specification: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, int]:
    """Resolve explicit retained-item quotas for the selected facets.

    The explicit mapping is the normal path.  A quotient/remainder fallback is
    retained only for older checkpoints that predate ``facet_item_counts``;
    new requirement runs always persist the mapping.
    """

    facet_ids = [
        str(facet["facet_id"])
        for facet in profile.get("facets") or []
        if isinstance(facet, Mapping) and facet.get("facet_id")
    ]
    if not facet_ids:
        raise ValueError("当前构念没有可用 facet")
    final_item_count = specification.get("final_item_count")
    if not _positive_int(final_item_count):
        raise ValueError("final_item_count 必须是正整数")
    raw = specification.get("facet_item_counts")
    if isinstance(raw, Mapping):
        counts = {
            str(key): int(value)
            for key, value in raw.items()
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
            )
        }
        if len(counts) != len(raw):
            counts = {}
        if (
            set(counts) == set(facet_ids)
            and all(value > 0 for value in counts.values())
            and sum(counts.values()) == int(final_item_count)
        ):
            return {facet_id: counts[facet_id] for facet_id in facet_ids}
    if len(facet_ids) == 1:
        return {facet_ids[0]: int(final_item_count)}
    if int(final_item_count) >= len(facet_ids):
        base, remainder = divmod(int(final_item_count), len(facet_ids))
        return {
            facet_id: base + (1 if index < remainder else 0)
            for index, facet_id in enumerate(facet_ids)
        }
    raise ValueError(
        "缺少有效的 facet_item_counts；多 facet 测验必须逐一指定题数，"
        "且各 facet 题数之和等于 final_item_count"
    )


def construct_profile_reference(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "inventory_id": str(profile["inventory_id"]),
        "inventory_name": str(profile["inventory_name"]),
        "inventory_version": str(profile["inventory_version"]),
        "review_status": str(profile["review_status"]),
        "selection_level": str(profile["selection_level"]),
        "domain_id": str(profile["domain_id"]),
        "domain_ids": [
            str(domain_id)
            for domain_id in profile.get("domain_ids") or []
        ],
        "domain_name": str(profile["domain_name"]),
        "selected_facet_ids": [
            str(facet["facet_id"]) for facet in profile["facets"]
        ],
        "profile_hash": str(profile["profile_hash"]),
        "resolution_source": str(profile["resolution_source"]),
    }


def required_generation_total(final_item_count: int) -> int:
    """Return the initial candidate count for incremental development."""
    return int(final_item_count) * INCREMENTAL_CANDIDATES_PER_CELL


def required_expansion_situation_total(final_item_count: int) -> int:
    """Return the fixed situation-pool size before blueprint selection."""
    return required_generation_total(final_item_count) + EXPANSION_SITUATION_BUFFER


def repair_blueprint_proposal(
    proposal: BlueprintAgentOutput | Mapping[str, Any],
    profile: Mapping[str, Any],
    expansions: list[FacetExpansion],
    facet_retention_quotas: Mapping[str, int],
) -> BlueprintAgentOutput:
    """Repair reference bookkeeping without changing construct content.

    Blueprint generation is an LLM selection problem, but the references are
    program-owned IDs.  A long multi-facet response can omit a row or copy a
    reference from a neighboring behavior.  Preserve rows that are valid and
    complete each facet from its own expansion pool, allocating references only
    within the same ``facet_id``/``behavior_id`` measurement unit.
    """

    result = (
        proposal
        if isinstance(proposal, BlueprintAgentOutput)
        else BlueprintAgentOutput.model_validate(proposal)
    )
    facet_order = [
        str(facet["facet_id"])
        for facet in profile.get("facets") or []
        if isinstance(facet, Mapping) and facet.get("facet_id")
    ]
    pools: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for expansion in expansions:
        for behavior in expansion.behavior_expansions:
            key = (str(expansion.facet_id), str(behavior.behavior_id))
            refs = []
            for mechanism in behavior.mechanisms:
                refs.extend(
                    (str(mechanism.mechanism_id), str(situation.situation_id))
                    for situation in mechanism.situations
                )
            pools[key] = refs

    accepted: dict[str, list[Any]] = {facet_id: [] for facet_id in facet_order}
    used: set[tuple[str, str, str, str]] = set()
    seen_rows: set[tuple[str, str, str, str]] = set()

    for row in result.rows:
        facet_id = str(row.facet_id)
        behavior_id = str(row.behavior_id)
        quota = int(facet_retention_quotas.get(facet_id, 0))
        pool = set(pools.get((facet_id, behavior_id), ()))
        raw_refs = [
            (str(reference.mechanism_id), str(reference.situation_id))
            for reference in row.candidate_references
        ]
        keys = [(facet_id, behavior_id, mechanism, situation) for mechanism, situation in raw_refs]
        valid = (
            facet_id in accepted
            and len(accepted[facet_id]) < quota
            and len(raw_refs) == INCREMENTAL_CANDIDATES_PER_CELL
            and len(set(raw_refs)) == len(raw_refs)
            and all(reference in pool for reference in raw_refs)
            and all(key not in used for key in keys)
            and (facet_id, behavior_id, *raw_refs[0]) not in seen_rows
        )
        if not valid:
            continue
        accepted[facet_id].append(row)
        used.update(keys)
        seen_rows.add((facet_id, behavior_id, *raw_refs[0]))

    def remaining_capacity(facet_id: str, rows: list[Any]) -> int:
        consumed: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (facet_id, str(row.behavior_id))
            consumed[key] = consumed.get(key, 0) + INCREMENTAL_CANDIDATES_PER_CELL
        capacity = 0
        for key, pool in pools.items():
            if key[0] != facet_id:
                continue
            capacity += max(
                0,
                (len(pool) - consumed.get(key, 0))
                // INCREMENTAL_CANDIDATES_PER_CELL,
            )
        return capacity

    # Drop only as many preserved rows as needed to make the remaining pool
    # usable; this keeps valid model-selected rows whenever possible.
    for facet_id in facet_order:
        quota = int(facet_retention_quotas[facet_id])
        rows = accepted[facet_id]
        while len(rows) + remaining_capacity(facet_id, rows) < quota:
            if not rows:
                break
            row = rows.pop()
            first_reference = row.candidate_references[0]
            seen_rows.discard(
                (
                    facet_id,
                    str(row.behavior_id),
                    str(first_reference.mechanism_id),
                    str(first_reference.situation_id),
                )
            )
            for reference in row.candidate_references:
                used.discard(
                    (
                        facet_id,
                        str(row.behavior_id),
                        str(reference.mechanism_id),
                        str(reference.situation_id),
                    )
                )

        for (pool_facet, behavior_id), pool in sorted(pools.items()):
            if pool_facet != facet_id or len(rows) >= quota:
                continue
            available = [
                reference
                for reference in pool
                if (facet_id, behavior_id, *reference) not in used
            ]
            index = 0
            while (len(rows) < quota
                   and index + INCREMENTAL_CANDIDATES_PER_CELL <= len(available)):
                references = available[index:index + INCREMENTAL_CANDIDATES_PER_CELL]
                index += INCREMENTAL_CANDIDATES_PER_CELL
                first = references[0]
                primary_key = (facet_id, behavior_id, *first)
                if primary_key in seen_rows:
                    continue
                rows.append(
                    BlueprintRowDraft(
                        facet_id=facet_id,
                        behavior_id=behavior_id,
                        candidate_references=[
                            {"mechanism_id": mechanism, "situation_id": situation}
                            for mechanism, situation in references
                        ],
                    )
                )
                used.update(
                    (facet_id, behavior_id, *reference)
                    for reference in references
                )
                seen_rows.add(primary_key)
        if len(rows) != quota:
            raise ValueError(
                f"facet={facet_id} 的情境引用无法组成 {quota} 个完整测量单元"
            )

    return BlueprintAgentOutput(
        rows=[row for facet_id in facet_order for row in accepted[facet_id]]
    )


def _reference_index(
    profile: Mapping[str, Any],
    expansions: list[FacetExpansion],
) -> tuple[
    set[tuple[str, str]],
    set[tuple[str, str, str]],
    set[tuple[str, str, str, str]],
]:
    behavior_refs: set[tuple[str, str]] = set()
    mechanism_refs: set[tuple[str, str, str]] = set()
    situation_refs: set[tuple[str, str, str, str]] = set()
    facet_ids = {
        str(facet["facet_id"])
        for facet in profile.get("facets") or []
        if isinstance(facet, Mapping)
    }
    for expansion in expansions:
        if expansion.facet_id not in facet_ids:
            continue
        for behavior in expansion.behavior_expansions:
            behavior_refs.add((expansion.facet_id, behavior.behavior_id))
            for mechanism in behavior.mechanisms:
                mechanism_refs.add(
                    (
                        expansion.facet_id,
                        behavior.behavior_id,
                        mechanism.mechanism_id,
                    )
                )
                for situation in mechanism.situations:
                    situation_refs.add(
                        (
                            expansion.facet_id,
                            behavior.behavior_id,
                            mechanism.mechanism_id,
                            situation.situation_id,
                        )
                    )
    return behavior_refs, mechanism_refs, situation_refs


def build_generation_blueprint(
    specification: Mapping[str, Any],
    profile: Mapping[str, Any],
    run_id: str,
    *,
    expansions: list[FacetExpansion],
    proposal: BlueprintAgentOutput | Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one LLM-designed two-way table to program-owned IDs."""

    result = (
        proposal
        if isinstance(proposal, BlueprintAgentOutput)
        else BlueprintAgentOutput.model_validate(proposal)
    )
    retention_total = int(specification["final_item_count"])
    generation_total = required_generation_total(retention_total)
    _, _, situation_refs = _reference_index(profile, expansions)
    if len(situation_refs) < generation_total:
        raise ValueError(
            "rows: Behavior Expansion 的唯一情境引用不足："
            f"需要 {generation_total} 个，实际 {len(situation_refs)} 个"
        )
    row_total = retention_total
    if row_total < 1:
        raise ValueError("rows: Behavior Expansion 没有可用于蓝图的情境引用")
    if len(result.rows) != row_total:
        raise ValueError(
            "rows: 细目表必须返回"
            f" {row_total} 个唯一组合，实际为 {len(result.rows)} 个"
        )
    facet_ids = [str(facet["facet_id"]) for facet in profile.get("facets") or []]
    expected_facet_counts = resolve_facet_item_counts(specification, profile)
    actual_facet_counts = Counter(str(row.facet_id) for row in result.rows)
    if actual_facet_counts != expected_facet_counts:
        raise ValueError(
            "rows: 每个 facet 的保留题数必须符合 facet_item_counts；"
            f"期望 {expected_facet_counts}，实际 {dict(actual_facet_counts)}"
        )
    generation_counts = [INCREMENTAL_CANDIDATES_PER_CELL] * row_total
    retention_counts = [1] * row_total
    blueprint_id = f"bp-{run_id}"
    situation_lookup: dict[tuple[str, str], dict[str, str]] = {}
    mechanism_lookup: dict[str, str] = {}
    for expansion in expansions:
        for behavior in expansion.behavior_expansions:
            for mechanism in behavior.mechanisms:
                mechanism_lookup[mechanism.mechanism_id] = mechanism.activation_mechanism
                for situation in mechanism.situations:
                    situation_lookup[(mechanism.mechanism_id, situation.situation_id)] = {
                        "domain": situation.domain,
                        "actor_relation": situation.actor_relation,
                        "event_class": situation.event_class,
                    }
    cells = []
    slots = []
    for index, row in enumerate(result.rows, start=1):
        candidate_references = [
            reference.model_dump(mode="json")
            for reference in row.candidate_references
        ]
        primary_reference = candidate_references[0]
        cell_id = f"{blueprint_id}-row-{index:02d}"
        sit = situation_lookup.get(
            (
                primary_reference["mechanism_id"],
                primary_reference["situation_id"],
            ),
            {},
        )
        cell = {
            "cell_id": cell_id,
            "facet_id": row.facet_id,
            "behavior_id": row.behavior_id,
            "mechanism_id": primary_reference["mechanism_id"],
            "situation_id": primary_reference["situation_id"],
            "candidate_references": candidate_references,
            "domain": sit.get("domain"),
            "actor_relation": sit.get("actor_relation"),
            "event_class": sit.get("event_class"),
            "activation_mechanism": mechanism_lookup.get(
                primary_reference["mechanism_id"]
            ),
            "planned_generation_count": generation_counts[index - 1],
            "planned_retention_count": retention_counts[index - 1],
        }
        cells.append(cell)
        for slot_index, candidate_reference in enumerate(
            candidate_references, start=1
        ):
            slots.append(
                {
                    "specification_id": f"{cell_id}-slot-{slot_index}",
                    "blueprint_cell_id": cell_id,
                    "candidate_reference": candidate_reference,
                }
            )
    blueprint = {
        "blueprint_id": blueprint_id,
        "version": GENERATION_BLUEPRINT_VERSION,
        "construct_profile_ref": construct_profile_reference(profile),
        "construct_profile_snapshot": deepcopy(dict(profile)),
        "expansion_refs": [
            {
                "facet_id": expansion.facet_id,
                "run_id": run_id,
            }
            for expansion in expansions
        ],
        "cells": cells,
        "slots": slots,
    }
    errors = validate_generation_blueprint(
        blueprint, specification, expansions=expansions
    )
    if errors:
        raise ValueError("；".join(f"{key}: {value}" for key, value in errors.items()))
    return blueprint


def planned_generation_count(blueprint: Mapping[str, Any]) -> int:
    return sum(
        int(cell.get("planned_generation_count", 0))
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping)
    )


def planned_retention_count(blueprint: Mapping[str, Any]) -> int:
    return sum(
        int(cell.get("planned_retention_count", 0))
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping)
    )


def _expansion_models(blueprint: Mapping[str, Any]) -> list[FacetExpansion]:
    return [
        load_facet_expansion(
            expansion_cache_path(row["run_id"], row["facet_id"])
        )
        for row in blueprint.get("expansion_refs") or []
        if isinstance(row, Mapping)
    ]


def resolve_blueprint_design(
    blueprint: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    expansions: list[FacetExpansion] | None = None,
    candidate_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = blueprint["construct_profile_snapshot"]
    facet = next(
        row for row in profile.get("facets") or []
        if row.get("facet_id") == cell.get("facet_id")
    )
    behavior = next(
        row for row in facet.get("behavior_evidence") or []
        if row.get("behavior_id") == cell.get("behavior_id")
    )
    expansion = next(
        row for row in (expansions or _expansion_models(blueprint))
        if row.facet_id == cell.get("facet_id")
    )
    behavior_expansion = next(
        row for row in expansion.behavior_expansions
        if row.behavior_id == cell.get("behavior_id")
    )
    reference = candidate_reference or cell
    mechanism_id = reference.get("mechanism_id")
    situation_id = reference.get("situation_id")
    mechanism = next(
        row for row in behavior_expansion.mechanisms
        if row.mechanism_id == mechanism_id
    )
    situation = next(
        row for row in mechanism.situations
        if row.situation_id == situation_id
    )
    return {
        "facet": deepcopy(facet),
        "behavior_evidence": deepcopy(behavior),
        "activation_mechanism": mechanism.activation_mechanism,
        "situation": situation.model_dump(mode="json"),
    }


def _skeleton_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["骨架必须是对象"]
    expected = {
        "situation_type",
        "stakes_level",
        "social_context",
        "behavioral_tension",
        "option_structure",
    }
    if set(value) != expected:
        return ["骨架字段不符合当前契约"]
    problems = []
    if value.get("stakes_level") not in {"low", "medium", "high"}:
        problems.append("stakes_level 无效")
    for field in ("situation_type", "social_context", "behavioral_tension"):
        if not _text(value.get(field)):
            problems.append(f"{field} 必须是非空文本")
    options = value.get("option_structure")
    if not isinstance(options, list) or len(options) != 4:
        problems.append("option_structure 必须包含四级行为")
        return problems
    levels = {
        row.get("behavioral_level")
        for row in options
        if isinstance(row, Mapping)
    }
    if levels != set(ANCHOR_LEVELS):
        problems.append("option_structure 必须完整覆盖四个行为等级")
    for row in options:
        if not isinstance(row, Mapping) or set(row) != {
            "behavioral_level",
            "behavioral_tendency",
            "psychological_function",
        }:
            problems.append("option_structure 条目字段无效")
            continue
        for field in ("behavioral_tendency", "psychological_function"):
            if not _text(row.get(field)):
                problems.append(f"{field} 必须是非空文本")
    return problems


def classify_compact_skeletons(
    blueprint: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    slot_ids = {
        str(slot["specification_id"])
        for slot in blueprint.get("slots") or []
        if isinstance(slot, Mapping)
    }
    valid = {}
    invalid = {}
    for specification_id, skeleton in candidates.items():
        problems = [] if specification_id in slot_ids else ["未知槽位"]
        problems.extend(_skeleton_problems(skeleton))
        if problems:
            invalid[specification_id] = problems
        else:
            valid[specification_id] = deepcopy(dict(skeleton))
    return {"valid": valid, "invalid": invalid}


def validate_generation_blueprint(
    blueprint: Any,
    specification: Mapping[str, Any] | None,
    *,
    expansions: list[FacetExpansion] | None = None,
) -> dict[str, str]:
    if not isinstance(blueprint, Mapping):
        return {"blueprint": "必须是对象"}
    expected = {
        "blueprint_id",
        "version",
        "construct_profile_ref",
        "construct_profile_snapshot",
        "expansion_refs",
        "cells",
        "slots",
    }
    errors = {}
    if set(blueprint) != expected:
        errors["blueprint.fields"] = "字段不符合当前双向细目表契约"
        return errors
    if blueprint.get("version") != GENERATION_BLUEPRINT_VERSION:
        errors["blueprint.version"] = "版本无效"
    cells = blueprint.get("cells")
    slots = blueprint.get("slots")
    if not isinstance(cells, list) or not cells:
        errors["cells"] = "必须是非空列表"
        return errors
    if not isinstance(slots, list):
        errors["slots"] = "必须是列表"
        return errors
    try:
        expansion_models = expansions or _expansion_models(blueprint)
    except (OSError, ValueError, KeyError) as exc:
        errors["expansion_refs"] = str(exc)
        return errors
    profile = blueprint.get("construct_profile_snapshot") or {}
    profile_facet_ids = [
        str(row.get("facet_id"))
        for row in profile.get("facets") or []
        if isinstance(row, Mapping) and row.get("facet_id")
    ]
    if specification is not None and profile_facet_ids:
        final_count = specification.get("final_item_count")
        if _positive_int(final_count):
            try:
                expected = resolve_facet_item_counts(
                    specification,
                    profile,
                )
            except ValueError as exc:
                errors["retention_facet_quotas"] = str(exc)
                expected = {}
            actual = Counter()
            for cell in cells:
                if isinstance(cell, Mapping):
                    actual[str(cell.get("facet_id"))] += int(
                        cell.get("planned_retention_count") or 0
                    )
            if expected and actual != expected:
                errors["retention_facet_quotas"] = (
                    "每个 facet 的保留题数配额不一致："
                    f"期望 {expected}，实际 {dict(actual)}"
                )
    refs = blueprint.get("expansion_refs")
    if not isinstance(refs, list):
        errors["expansion_refs"] = "必须是列表"
        return errors
    ref_facets = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, Mapping) or set(ref) != {"run_id", "facet_id"}:
            errors[f"expansion_refs[{index}]"] = "字段无效"
            continue
        ref_facets.append(str(ref["facet_id"]))
    if len(ref_facets) != len(set(ref_facets)):
        errors["expansion_refs"] = "facet 引用必须唯一"
    behavior_refs, mechanism_refs, situation_refs = _reference_index(
        profile, expansion_models
    )
    facet_ids = {
        str(row.get("facet_id"))
        for row in profile.get("facets") or []
        if isinstance(row, Mapping)
    }
    expansion_facets = {row.facet_id for row in expansion_models}
    if set(ref_facets) != facet_ids or expansion_facets != facet_ids:
        errors["expansion_refs"] = "必须逐一覆盖当前 facet"
    cell_ids = set()
    combinations = set()
    candidate_combinations: dict[tuple[str, str, str, str], str] = {}
    for index, cell in enumerate(cells):
        prefix = f"cells[{index}]"
        if not isinstance(cell, Mapping):
            errors[prefix] = "必须是对象"
            continue
        required = {
            "cell_id", "facet_id", "behavior_id", "mechanism_id",
            "situation_id", "planned_generation_count",
            "planned_retention_count",
            "domain", "actor_relation", "event_class",
            "activation_mechanism", "candidate_references",
        }
        if set(cell) != required:
            errors[prefix] = "字段无效"
            continue
        cell_id = str(cell.get("cell_id") or "")
        if not cell_id or cell_id in cell_ids:
            errors[f"{prefix}.cell_id"] = "必须唯一"
        cell_ids.add(cell_id)
        facet_id = str(cell.get("facet_id") or "")
        behavior_id = str(cell.get("behavior_id") or "")
        mechanism_id = str(cell.get("mechanism_id") or "")
        situation_id = str(cell.get("situation_id") or "")
        combination = (facet_id, behavior_id, mechanism_id, situation_id)
        if combination in combinations:
            errors[f"{prefix}.reference"] = "同一引用组合不得重复成行"
        combinations.add(combination)
        candidate_references = cell.get("candidate_references")
        if (
            not isinstance(candidate_references, list)
            or len(candidate_references) != INCREMENTAL_CANDIDATES_PER_CELL
            or any(
                not isinstance(reference, Mapping)
                or set(reference) != {"mechanism_id", "situation_id"}
                or not _text(reference.get("mechanism_id"))
                or not _text(reference.get("situation_id"))
                for reference in candidate_references or []
            )
        ):
            errors[f"{prefix}.candidate_references"] = (
                f"必须包含 {INCREMENTAL_CANDIDATES_PER_CELL} 个有效候选引用"
            )
            candidate_references = []
        else:
            candidate_pairs = [
                (
                    str(reference["mechanism_id"]),
                    str(reference["situation_id"]),
                )
                for reference in candidate_references
            ]
            if len(candidate_pairs) != len(set(candidate_pairs)):
                errors[f"{prefix}.candidate_references"] = (
                    "同一测量单元的候选情境不得重复"
                )
            if (mechanism_id, situation_id) != candidate_pairs[0]:
                errors[f"{prefix}.reference"] = (
                    "主引用必须与第一个候选引用一致"
                )
        if facet_id not in facet_ids or (facet_id, behavior_id) not in behavior_refs:
            errors[f"{prefix}.reference"] = "facet或behavior引用无效"
        elif (facet_id, behavior_id, mechanism_id) not in mechanism_refs:
            errors[f"{prefix}.mechanism_id"] = "引用无效"
        elif (
            facet_id, behavior_id, mechanism_id, situation_id
        ) not in situation_refs:
            errors[f"{prefix}.situation_id"] = "引用无效"
        for reference in candidate_references:
            if not isinstance(reference, Mapping):
                continue
            candidate_combination = (
                facet_id,
                behavior_id,
                str(reference.get("mechanism_id") or ""),
                str(reference.get("situation_id") or ""),
            )
            if candidate_combination not in situation_refs:
                errors[f"{prefix}.candidate_references"] = "候选情境引用无效"
            first_cell = candidate_combinations.get(candidate_combination)
            if first_cell is not None and first_cell != prefix:
                errors[f"{prefix}.candidate_references"] = (
                    "候选情境引用 "
                    f"{candidate_combination[2]}/{candidate_combination[3]} "
                    f"已在 {first_cell} 使用，不得在不同测量单元重复"
                )
            elif first_cell is None:
                candidate_combinations[candidate_combination] = prefix
        generated = cell.get("planned_generation_count")
        retained = cell.get("planned_retention_count")
        retention_valid = (
            isinstance(retained, int)
            and not isinstance(retained, bool)
            and retained >= 0
        )
        if (
            not _positive_int(generated)
            or generated < INCREMENTAL_CANDIDATES_PER_CELL
            or not retention_valid
        ):
            errors[f"{prefix}.counts"] = "生成题数必须为正整数，保留题数不得为负"
        elif retained > generated:
            errors[f"{prefix}.counts"] = "保留题数不得超过生成题数"
    expected_generation = sum(
        int(cell.get("planned_generation_count", 0))
        for cell in cells if isinstance(cell, Mapping)
    )
    if len(slots) != expected_generation:
        errors["slots"] = "槽位数必须等于计划生成题数"
    if specification is not None:
        final_count = specification.get("final_item_count")
        if _positive_int(final_count):
            if planned_retention_count(blueprint) != final_count:
                errors["retention_total"] = "保留题数与测验规格不一致"
            if planned_generation_count(blueprint) != required_generation_total(
                final_count
            ):
                errors["generation_total"] = "生成题数与测验规格不一致"
    return errors


def materialize_item_specifications(
    blueprint: Mapping[str, Any],
    item_skeletons: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    expansions: list[FacetExpansion] | None = None,
) -> list[dict[str, Any]]:
    skeletons = item_skeletons or {}
    cells = {
        str(cell["cell_id"]): cell
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping)
    }
    rows = []
    for slot in blueprint.get("slots") or []:
        if not isinstance(slot, Mapping):
            continue
        specification_id = str(slot["specification_id"])
        skeleton = skeletons.get(specification_id)
        if not isinstance(skeleton, Mapping):
            continue
        cell = cells[str(slot["blueprint_cell_id"])]
        candidate_reference = slot.get("candidate_reference")
        if not isinstance(candidate_reference, Mapping):
            candidate_reference = {
                "mechanism_id": cell["mechanism_id"],
                "situation_id": cell["situation_id"],
            }
        design = resolve_blueprint_design(
            blueprint,
            cell,
            expansions=expansions,
            candidate_reference=candidate_reference,
        )
        options = {
            row["behavioral_level"]: row
            for row in skeleton["option_structure"]
        }
        facet = design["facet"]
        situation = design["situation"]
        rows.append(
            {
                "specification_id": specification_id,
                "blueprint_cell_id": cell["cell_id"],
                "target_dimension_id": cell["facet_id"],
                "behavior_evidence_id": cell["behavior_id"],
                "mechanism_id": candidate_reference["mechanism_id"],
                "situation_id": candidate_reference["situation_id"],
                "context_category": situation["domain"],
                "context_seed": situation["event_class"],
                "situation_type": skeleton["situation_type"],
                "stakes_level": skeleton["stakes_level"],
                "social_context": skeleton["social_context"],
                "activation_mechanism": design["activation_mechanism"],
                "core_tension": skeleton["behavioral_tension"],
                "behavioral_anchors": {
                    level: options[level]["behavioral_tendency"]
                    for level in ANCHOR_LEVELS
                },
                "behavioral_functions": {
                    level: options[level]["psychological_function"]
                    for level in ANCHOR_LEVELS
                },
                "contamination_exclusions": deepcopy(
                    facet.get("common_confounds") or []
                ),
                "scenario_constraints": deepcopy(
                    facet.get("inappropriate_conditions") or []
                ),
                "option_constraints": deepcopy(
                    [
                        *(facet.get("forbidden_patterns") or []),
                        *(facet.get("option_design_rules") or []),
                    ]
                ),
                "avoid_scenario_patterns": [],
                "avoid_response_patterns": [],
            }
        )
    return rows
