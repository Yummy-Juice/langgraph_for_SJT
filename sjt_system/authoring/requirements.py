"""Minimal requirements clarification and confirmation rules."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from sjt_system.authoring.construct_registry import (
    ALL_DOMAINS_DOMAIN_ID,
    construct_selection_catalog,
    construct_selection_from_profile,
    resolve_construct_profile,
    resolve_construct_selection,
)
from sjt_system.config import DEFAULT_OUTPUT_LANGUAGE
from sjt_system.state import RequirementInteraction


USER_CONFIRMATION_FIELDS = frozenset(
    {
        "construct_selection",
        "target_population",
        "facet_item_counts",
        "final_item_count",
    }
)
SPECIFICATION_FIELDS = frozenset(
    {*USER_CONFIRMATION_FIELDS, "output_language"}
)
VALID_SPECIFICATION_SOURCES = frozenset(
    {"user", "inferred", "system_default"}
)

_PLACEHOLDER_VALUES = {
    "n/a", "na", "none", "null", "unknown", "不明确", "不确定",
    "未知", "未指定", "待定", "待确认",
}
_LANGUAGE_ALIASES = {
    "简体中文": "zh-CN",
    "simplified chinese": "zh-CN",
    "zh_cn": "zh-CN",
    "zh-cn": "zh-CN",
    "中文": "zh-CN",
    "英文": "en",
    "english": "en",
}
_LANGUAGE_TAG = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_ALL_DOMAINS_ALIASES = frozenset(
    {
        "alldomains",
        "alldomain",
        "all",
        "allfacets",
        "allfacet",
        "全部域",
        "全域",
        "全部分面",
        "所有域",
        "所有分面",
    }
)
_ITEM_COUNT_PATTERN = re.compile(r"(?<!\d)(\d{1,4})\s*(?:道题|题目|道|题)")
_PER_FACET_COUNT_PATTERN = re.compile(
    r"每个\s*(?:facet|分面|维度|题目)\D{0,12}(\d{1,3})\s*(?:道题|题目|道|题)"
    ,
    re.IGNORECASE,
)
_NAMED_FACET_COUNT_PATTERN = re.compile(
    r"(?P<facet>[A-Za-z][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,})"
    r"\s*(?:各有|各为|各|设置为|配置为|为|=|：|:)\s*"
    r"(?P<count>\d{1,3})\s*(?:道题|题目|道|题)?",
    re.IGNORECASE,
)
_GROUPED_NAMED_FACET_COUNT_PATTERN = re.compile(
    r"(?P<labels>[A-Za-z][A-Za-z0-9_.-]*(?:\s*(?:[,，、/和及&]\s*)"
    r"[A-Za-z][A-Za-z0-9_.-]*)+)\s*各\s*"
    r"(?:有|为|设置为|配置为)?\s*(?P<count>\d{1,3})"
    r"\s*(?:道题|题目|道|题)?",
    re.IGNORECASE,
)


def _normalized_text(value: Any) -> str:
    """Normalize user wording using the same matching rule as the registry."""

    return re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        str(value or "").strip().casefold(),
    )


def _normalize_domain_id(value: Any) -> Any:
    """Canonicalize human/model spellings of the cross-domain scope."""

    if not isinstance(value, str):
        return value
    normalized = _normalized_text(value)
    if normalized in _ALL_DOMAINS_ALIASES:
        return ALL_DOMAINS_DOMAIN_ID
    return value.strip()


def _is_valid_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return bool(normalized) and normalized not in _PLACEHOLDER_VALUES


def _normalize_language(value: Any) -> str:
    if not _is_valid_text(value):
        return DEFAULT_OUTPUT_LANGUAGE
    text = str(value).strip()
    return _LANGUAGE_ALIASES.get(text.casefold(), text)


def _canonical_construct_selection(
    specification: Mapping[str, Any],
) -> dict[str, Any] | None:
    selection = specification.get("construct_selection")
    if isinstance(selection, Mapping):
        payload = dict(selection)
        # New state contains only domain_ids. Legacy model/checkpoint payloads
        # may still contain domain_id; the registry resolver accepts that
        # input and the canonical projection below removes it.
        if "domain_ids" in payload:
            payload.pop("domain_id", None)
        elif "domain_id" in payload:
            legacy_domain_id = _normalize_domain_id(payload.get("domain_id"))
            if not legacy_domain_id and payload.get("facet_ids"):
                legacy_domain_id = ALL_DOMAINS_DOMAIN_ID
            payload["domain_id"] = legacy_domain_id
        profile = resolve_construct_selection(payload)
        return construct_selection_from_profile(profile)

    # One-way compatibility for old requirement candidates/checkpoints. New
    # prompts never produce this overloaded free-text field.
    target = specification.get("target_construct")
    if isinstance(target, str) and target.strip():
        return construct_selection_from_profile(
            resolve_construct_profile(target)
        )
    return None


def _selection_facets(
    selection: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(selection, Mapping):
        return []
    try:
        profile = resolve_construct_selection(selection)
    except ValueError:
        return []
    return [
        dict(facet)
        for facet in (profile.get("facets") or [])
        if isinstance(facet, Mapping)
    ]


def _normalize_facet_item_counts(
    value: Any,
    selection: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    """Normalize a facet quota mapping against the selected registry facets."""

    if not isinstance(value, Mapping):
        return None
    facets = _selection_facets(selection)
    aliases: dict[str, str] = {}
    for facet in facets:
        facet_id = str(facet.get("facet_id") or "")
        if not facet_id:
            continue
        aliases[facet_id.casefold()] = facet_id
        for key in ("facet_name", "facet_name_en"):
            alias = facet.get(key)
            if isinstance(alias, str) and alias.strip():
                aliases[alias.strip().casefold()] = facet_id
    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if not isinstance(raw_key, str):
            return None
        facet_id = aliases.get(raw_key.strip().casefold())
        if facet_id is None:
            return None
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
        ):
            return None
        normalized[facet_id] = int(raw_count)
    if set(normalized) != {
        str(facet.get("facet_id")) for facet in facets
    }:
        return None
    return normalized


def _explicit_facet_item_counts(
    texts: tuple[str | None, ...],
    selection: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    """Read explicit per-facet counts from the latest user wording."""

    facets = _selection_facets(selection)
    if not facets:
        return None
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        generic = _PER_FACET_COUNT_PATTERN.search(text)
        if generic is not None:
            count = int(generic.group(1))
            return {
                str(facet["facet_id"]): count
                for facet in facets
            }
        aliases = {
            str(facet["facet_id"]): {
                str(facet["facet_id"]).casefold(),
                str(facet.get("facet_name") or "").casefold(),
                str(facet.get("facet_name_en") or "").casefold(),
            }
            for facet in facets
        }
        parsed: dict[str, int] = {}
        # Natural language commonly groups several named facets before one
        # quota (for example, ``Gregariousness、Warmth、Altruism各10道``).
        # Parse that form before the single-name form so the model cannot
        # leave the previous inventory-wide quotas in place.
        for match in _GROUPED_NAMED_FACET_COUNT_PATTERN.finditer(text):
            labels = re.split(r"[,，、/和及&]", match.group("labels"))
            count = int(match.group("count"))
            for label in labels:
                normalized_label = _normalized_text(label)
                facet_id = next(
                    (
                        facet_id
                        for facet_id, facet_aliases in aliases.items()
                        if _normalized_text(facet_id) == normalized_label
                        or any(
                            _normalized_text(alias) == normalized_label
                            for alias in facet_aliases
                        )
                    ),
                    None,
                )
                if facet_id is not None:
                    parsed[facet_id] = count
        for match in _NAMED_FACET_COUNT_PATTERN.finditer(text):
            label = str(match.group("facet")).strip().casefold()
            facet_id = next(
                (
                    facet_id
                    for facet_id, facet_aliases in aliases.items()
                    if label in facet_aliases
                ),
                None,
            )
            if facet_id is not None:
                parsed[facet_id] = int(match.group("count"))
        if parsed and set(parsed) == set(aliases):
            return parsed
    return None


def _explicit_multi_facet_selection(
    *texts: str | None,
) -> dict[str, Any] | None:
    """Resolve several facet names mentioned together in user wording.

    ``resolve_construct_profile`` intentionally rejects ambiguous multi-facet
    free text. Requirement edits are a different boundary: a user can name a
    deliberate subset such as ``Gregariousness、Warmth、Altruism``. Resolve
    those names against the versioned catalog, then pass the structured IDs to
    the strict registry validator.
    """

    inventories = construct_selection_catalog()
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        normalized_text = _normalized_text(text)
        matches: dict[str, str] = {}
        for catalog in inventories:
            for domain in catalog.get("domains") or []:
                domain_id = str(domain.get("domain_id") or "")
                for facet in domain.get("facets") or []:
                    facet_id = str(facet.get("facet_id") or "")
                    aliases = (
                        facet_id,
                        facet.get("facet_name"),
                        facet.get("facet_name_en"),
                    )
                    if any(
                        alias
                        and _normalized_text(alias) in normalized_text
                        for alias in aliases
                    ):
                        matches[facet_id] = domain_id
        if len(matches) < 2:
            continue
        domain_ids = set(matches.values())
        selection = {
            "inventory_id": str(inventories[0]["inventory_id"]),
            "domain_ids": sorted(domain_ids),
            "facet_ids": list(matches),
        }
        return construct_selection_from_profile(
            resolve_construct_selection(selection)
        )
    return None


def _resolve_facet_item_counts(
    raw_specification: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    texts: tuple[str | None, ...],
    final_item_count: Any,
) -> tuple[dict[str, int], str] | tuple[dict[str, int], None] | tuple[None, None]:
    """Return quotas and whether they were directly supplied or inferred."""

    explicit = _explicit_facet_item_counts(texts, selection)
    if explicit is not None:
        return explicit, "user"
    normalized = _normalize_facet_item_counts(
        raw_specification.get("facet_item_counts"),
        selection,
    )
    if normalized is not None:
        return normalized, None
    facets = _selection_facets(selection)
    if not facets or not isinstance(final_item_count, int) or isinstance(
        final_item_count, bool
    ) or final_item_count < 1:
        return None, None
    facet_ids = [str(facet["facet_id"]) for facet in facets]
    if len(facet_ids) == 1:
        return {facet_ids[0]: int(final_item_count)}, "inferred"
    if final_item_count % len(facet_ids) == 0:
        count = final_item_count // len(facet_ids)
        return {facet_id: count for facet_id in facet_ids}, "inferred"
    return None, None


def _explicit_construct_selection(
    *texts: str | None,
) -> dict[str, Any] | None:
    """Resolve the latest unambiguous construct named by the user."""

    multi_facet = _explicit_multi_facet_selection(*texts)
    if multi_facet is not None:
        return multi_facet
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        normalized = text.casefold()
        # Do not interpret diagnostic text that merely quotes a previous
        # payload (for example ``'domain_id': 'extraversion'``) as a new
        # construct request.
        if "domain_id" in normalized and not any(
            marker in normalized
            for marker in ("改为", "改成", "选择", "测量", "生成", "覆盖", "全部", "所有")
        ):
            continue
        try:
            profile = resolve_construct_profile(text)
        except ValueError:
            continue
        return construct_selection_from_profile(profile)
    return None


def _explicit_item_count(
    texts: tuple[str | None, ...],
    selection: Mapping[str, Any] | None,
) -> int | None:
    """Extract an explicit latest-round item count before model output merge."""

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        matches = _ITEM_COUNT_PATTERN.findall(text)
        if matches:
            return int(matches[-1])
    return None


def canonicalize_requirement_agent_update(
    update: Mapping[str, Any],
    *,
    previous_candidate: Mapping[str, Any] | None = None,
    user_feedback: str | None = None,
    user_request: str | None = None,
) -> dict[str, Any]:
    """Normalize one model candidate into the facet-aware requirement spec."""

    if set(update) != {"test_specification", "specification_sources"}:
        unexpected = set(update) - {"test_specification", "specification_sources"}
        if unexpected:
            raise ValueError(
                "clarify_requirements 只能返回 test_specification 和 "
                "specification_sources"
            )
    raw_specification = update.get("test_specification")
    raw_sources = update.get("specification_sources")
    if raw_specification is None:
        raw_specification = {}
    if raw_sources is None:
        raw_sources = {}
    if not isinstance(raw_specification, Mapping):
        raise ValueError("test_specification 必须是对象")
    if not isinstance(raw_sources, Mapping):
        raise ValueError("specification_sources 必须是对象")

    previous_specification: Mapping[str, Any] = {}
    previous_sources: Mapping[str, Any] = {}
    if isinstance(previous_candidate, Mapping):
        candidate_specification = previous_candidate.get("test_specification")
        candidate_sources = previous_candidate.get("specification_sources")
        if isinstance(candidate_specification, Mapping):
            previous_specification = candidate_specification
        if isinstance(candidate_sources, Mapping):
            previous_sources = candidate_sources
    merged_specification = {**previous_specification, **dict(raw_specification)}
    merged_sources = {**previous_sources, **dict(raw_sources)}

    explicit_selection = _explicit_construct_selection(user_feedback)
    previous_selection = _canonical_construct_selection(previous_specification)
    selection = explicit_selection
    if selection is None and user_feedback and previous_selection is not None:
        # A follow-up that only criticizes the previous candidate must not
        # silently fall back to the original request's old single facet.
        selection = previous_selection
    if selection is None:
        selection = _explicit_construct_selection(user_request)
    if selection is None:
        selection = _canonical_construct_selection(merged_specification)
    user_texts = (user_feedback, user_request)
    raw_item_count = merged_specification.get("final_item_count")
    explicit_item_count = _explicit_item_count(user_texts, selection)
    count_basis = (
        explicit_item_count
        if explicit_item_count is not None
        else raw_item_count
    )
    facet_item_counts, facet_count_source = _resolve_facet_item_counts(
        merged_specification,
        selection,
        user_texts,
        count_basis,
    )
    if facet_item_counts is not None:
        derived_total = sum(facet_item_counts.values())
    else:
        derived_total = count_basis
    specification = {
        "construct_selection": selection,
        "target_population": merged_specification.get("target_population"),
        "facet_item_counts": facet_item_counts or {},
        "final_item_count": derived_total,
        "output_language": _normalize_language(
            merged_specification.get("output_language")
        ),
    }
    sources = {
        field: merged_sources.get(field)
        for field in SPECIFICATION_FIELDS
    }
    if sources.get("construct_selection") is None:
        sources["construct_selection"] = merged_sources.get("target_construct")
    if explicit_selection is not None:
        sources["construct_selection"] = "user"
    if facet_count_source == "user":
        sources["facet_item_counts"] = "user"
        sources["final_item_count"] = "user"
    elif facet_count_source == "inferred":
        sources["facet_item_counts"] = "inferred"
        if explicit_item_count is None:
            sources["final_item_count"] = "inferred"
    elif explicit_item_count is not None:
        sources["final_item_count"] = "user"
    if sources.get("facet_item_counts") is None:
        sources["facet_item_counts"] = (
            "inferred" if facet_item_counts is not None else "inferred"
        )
    if sources.get("output_language") != "user":
        sources["output_language"] = "system_default"
    return {
        "test_specification": specification,
        "specification_sources": sources,
    }


def validate_test_specification(specification: Any) -> dict[str, str]:
    """Return field-level errors; an empty mapping means structurally valid."""

    if not isinstance(specification, Mapping):
        return {"test_specification": "必须是对象"}
    errors: dict[str, str] = {}
    if set(specification) != SPECIFICATION_FIELDS:
        errors["test_specification.fields"] = (
            "必须且只能包含 construct_selection、target_population、"
            "facet_item_counts、final_item_count、output_language"
        )
    profile = None
    try:
        profile = resolve_construct_selection(
            specification.get("construct_selection")
        )
    except ValueError as exc:
        errors["construct_selection"] = str(exc)
    if not _is_valid_text(specification.get("target_population")):
        errors["target_population"] = "必须是非空且明确的文本"
    final_item_count = specification.get("final_item_count")
    if (
        isinstance(final_item_count, bool)
        or not isinstance(final_item_count, int)
        or final_item_count < 1
    ):
        errors["final_item_count"] = "必须是正整数"
    facet_item_counts = specification.get("facet_item_counts")
    if not isinstance(facet_item_counts, Mapping):
        errors["facet_item_counts"] = "必须是 facet_id 到正整数题数的对象"
    elif profile is not None:
        expected_facet_ids = {
            str(facet["facet_id"])
            for facet in profile.get("facets") or []
        }
        actual_facet_ids = set(facet_item_counts)
        if actual_facet_ids != expected_facet_ids:
            errors["facet_item_counts"] = (
                "必须逐一覆盖所选 facet；"
                f"期望 {sorted(expected_facet_ids)}，实际 "
                f"{sorted(map(str, actual_facet_ids))}"
            )
        else:
            invalid_counts = [
                facet_id
                for facet_id, count in facet_item_counts.items()
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 1
                )
            ]
            if invalid_counts:
                errors["facet_item_counts"] = (
                    "每个 facet 的题数必须是正整数："
                    + "、".join(sorted(map(str, invalid_counts)))
                )
            elif (
                isinstance(final_item_count, int)
                and sum(facet_item_counts.values()) != final_item_count
            ):
                errors["facet_item_counts"] = (
                    "各 facet 题数之和必须等于 final_item_count"
                )
    language = specification.get("output_language")
    if not isinstance(language, str) or not _LANGUAGE_TAG.fullmatch(language):
        errors["output_language"] = "必须是规范语言标签，例如 zh-CN 或 en"
    return errors


def build_requirement_interaction(
    result: Mapping[str, Any],
) -> RequirementInteraction:
    """Validate the only two pieces of interaction metadata still needed."""

    if not isinstance(result.get("suggestions"), list):
        raise ValueError("Requirement Agent 输出缺少 suggestions 列表")
    if not isinstance(result.get("questions"), list):
        raise ValueError("Requirement Agent 输出缺少 questions 列表")

    suggestions: list[dict[str, Any]] = []
    for suggestion in result["suggestions"]:
        if not isinstance(suggestion, Mapping):
            raise ValueError("Requirement suggestion 必须是对象")
        if set(suggestion) != {"field", "reason"}:
            raise ValueError("Requirement suggestion 只能包含 field 和 reason")
        field = suggestion.get("field")
        if field not in USER_CONFIRMATION_FIELDS:
            raise ValueError("Requirement suggestion 只能针对用户确认字段")
        if not _is_valid_text(suggestion.get("reason")):
            raise ValueError("Requirement suggestion 缺少有效 reason")
        suggestions.append(dict(suggestion))

    questions: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for question in result["questions"]:
        if not isinstance(question, Mapping) or set(question) != {
            "field", "issue_type", "text"
        }:
            raise ValueError(
                "Requirement question 只能包含 field、issue_type、text"
            )
        field = question.get("field")
        if field not in USER_CONFIRMATION_FIELDS:
            raise ValueError("Requirement question 只能针对用户确认字段")
        if field in seen_fields:
            raise ValueError(f"Requirement question 重复字段：{field}")
        if question.get("issue_type") not in {
            "missing", "ambiguous", "confirm_inference"
        }:
            raise ValueError("Requirement question.issue_type 无效")
        if not _is_valid_text(question.get("text")):
            raise ValueError("Requirement question 缺少有效 text")
        seen_fields.add(str(field))
        questions.append(dict(question))
    if len(questions) > 3:
        raise ValueError("每轮最多提出三个需求问题")
    return {"suggestions": suggestions, "questions": questions}


def build_confirmed_requirement_fields_update(
    result: Mapping[str, Any],
    confirmed_fields: list[str],
) -> dict[str, Any]:
    """Merge explicit user-owned fields into durable confirmation state."""

    state_update = result.get("state_update")
    sources = (
        state_update.get("specification_sources", {})
        if isinstance(state_update, Mapping)
        else {}
    )
    user_fields = {
        field
        for field, source in sources.items()
        if source == "user" and field in USER_CONFIRMATION_FIELDS
    }
    return {
        "confirmed_requirement_fields": sorted(
            ({*confirmed_fields, *user_fields}) & USER_CONFIRMATION_FIELDS
        )
    }


def validate_requirement_confirmation(
    specification: Any,
    interaction: Mapping[str, Any] | None,
    confirmed_fields: list[str],
    specification_sources: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate structural completeness and explicit acceptance of inference."""

    errors = validate_test_specification(specification)
    if not isinstance(interaction, Mapping):
        return {**errors, "interaction": "缺少需求交互状态"}

    if not isinstance(specification_sources, Mapping):
        errors["specification_sources"] = "缺少字段来源记录"
        sources: Mapping[str, Any] = {}
    else:
        sources = specification_sources
        if set(sources) != SPECIFICATION_FIELDS:
            errors["specification_sources"] = "字段来源必须精确覆盖需求规格"
        invalid_sources = sorted(
            field for field, source in sources.items()
            if source not in VALID_SPECIFICATION_SOURCES
        )
        if invalid_sources:
            errors["invalid_specification_sources"] = (
                "以下字段来源无效：" + "、".join(invalid_sources)
            )
        for field in USER_CONFIRMATION_FIELDS:
            if sources.get(field) not in {"user", "inferred"}:
                errors[f"source.{field}"] = "来源必须是 user 或 inferred"
        if sources.get("output_language") not in {"user", "system_default"}:
            errors["source.output_language"] = (
                "输出语言来源必须是 user 或 system_default"
            )

    questions = interaction.get("questions")
    if not isinstance(questions, list):
        errors["questions"] = "必须是结构化问题列表"
    elif questions:
        errors["questions"] = "仍有待解决问题：" + "、".join(
            str(question.get("field"))
            for question in questions
            if isinstance(question, Mapping)
        )

    inferred = {
        field for field, source in sources.items()
        if source == "inferred" and field in USER_CONFIRMATION_FIELDS
    }
    unaccepted = sorted(inferred - set(confirmed_fields))
    if unaccepted:
        errors["inferred_fields"] = (
            "系统推断值尚未被用户接受：" + "、".join(unaccepted)
        )
    return errors
