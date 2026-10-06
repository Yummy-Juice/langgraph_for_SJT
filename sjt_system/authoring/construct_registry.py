"""Versioned construct registry adapted from the legacy project.

Most content is copied from legacy ``resources.py``. Explicit, reviewable SJT
operationalization corrections are applied here when a legacy description
would systematically confound a NEO facet in behavior-only item generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from sjt_system.authoring.legacy_construct_resources import get_inventory


REGISTRY_VERSION = "legacy-openCode-for-SJT-v6"
DEFAULT_INVENTORY_ID = "neo_pi_r"
# A real string is required by the requirement-agent JSON schema. It denotes
# a cross-domain scope; concrete selected domains are stored in ``domain_ids``.
ALL_DOMAINS_DOMAIN_ID = "all_domains"

_FACET_USER_ALIASES: dict[str, tuple[str, ...]] = {
    # Legacy resource names are English for this facet even in the Chinese
    # inventory. Keep common user wording at the resolver boundary.
    "agreeableness_modesty": ("谦逊", "谦虚", "humility"),
}


_FACET_HARD_CONSTRAINTS: dict[str, list[dict[str, Any]]] = {
    "neuroticism_impulsiveness": [
        {
            "rule_id": "impulsiveness-not-low-deliberation",
            "target_levels": ["low", "medium_low", "medium_high", "high"],
            "prohibited_meaning": (
                "把N5冲动性实现为未经充分思考便作决定、草率权衡后果或缺少计划"
            ),
            "required_boundary": (
                "等级必须来自面对明确短期诱惑、欲望或行动冲动时的克制程度；"
                "未经思考的决策属于低C6审慎性"
            ),
        }
    ],
    "extraversion_warmth": [
        {
            "rule_id": "warmth-not-agreeableness-helping",
            "target_levels": ["low", "medium_low", "medium_high", "high"],
            "prohibited_meaning": (
                "通过帮助投入、同情弱者、迁就他人或善意推定来建立E1热情性等级"
            ),
            "required_boundary": (
                "等级必须来自主动建立人际接触以及表达亲近和友好的程度；"
                "不能主要测量A1信任、A3利他或A6慈悲心"
            ),
        }
    ],
    "conscientiousness_competence": [
        {
            "rule_id": "competence-not-assertiveness-or-achievement",
            "target_levels": ["low", "medium_low", "medium_high", "high"],
            "prohibited_meaning": (
                "通过主导发言、主动加码、追求完美、条理拆解或更强专业能力"
                "来建立C1胜任感等级"
            ),
            "required_boundary": (
                "在能力证据和资源相同的前提下，等级只来自面对普通挑战时"
                "有证据支持的现实掌控预期"
            ),
        }
    ],
    "openness_values": [
        {
            "rule_id": "values-no-position-content-scoring",
            "target_levels": ["low", "medium_low", "medium_high", "high"],
            "prohibited_meaning": (
                "根据赞成哪一种政治、宗教、道德或具体价值立场来建立O6价值开放性等级"
            ),
            "required_boundary": (
                "只测量是否愿意审视低敏感度日常规则及其假设，不测量最终选择的"
                "立场内容，也不把违规或谋取私利写成开放性"
            ),
        }
    ],
    "agreeableness_trust": [
        {
            "rule_id": "trust-no-blind-trust",
            "target_levels": ["high"],
            "prohibited_meaning": (
                "把高信任实现为无条件相信、取消正常沟通、完全不检查或不保留基本判断"
            ),
            "required_boundary": (
                "保持对他人的善意预期，同时保留与情境相称的最低限度判断或正常沟通"
            ),
        },
        {
            "rule_id": "trust-no-superior-risk-management-low",
            "target_levels": ["low"],
            "prohibited_meaning": (
                "通过更周全的计划、监督或风险管理优势，把低信任写成明显更理性的方案"
            ),
            "required_boundary": (
                "低信任只表现为较少给予善意预期；不得独占更好的信息、计划或保障"
            ),
        },
    ],
    "agreeableness_straightforwardness": [
        {
            "rule_id": "straightforwardness-no-hurtful-bluntness",
            "target_levels": ["high"],
            "prohibited_meaning": (
                "把高坦诚实现为不顾场合、不顾关系、毫无修饰或带有伤害性的直率"
            ),
            "required_boundary": "如实表达核心信息，同时保持适当、尊重且可执行的表达方式",
        },
        {
            "rule_id": "straightforwardness-no-obvious-lie-low",
            "target_levels": ["low"],
            "prohibited_meaning": "把低坦诚实现为明显说谎或完全否认已知问题",
            "required_boundary": "低坦诚可以策略性保留或委婉表达，但不能成为明显欺骗",
        },
    ],
    "agreeableness_altruism": [
        {
            "rule_id": "altruism-equal-resource-boundary",
            "target_levels": ["low", "medium_low", "medium_high", "high"],
            "prohibited_meaning": (
                "通过额外时间、更多行动、协调更多人员或无限投入来建立利他等级"
            ),
            "required_boundary": (
                "四级面对相同资源和现实成本；只改变主动回应他人需要的程度"
            ),
        },
        {
            "rule_id": "altruism-low-remains-responsive",
            "target_levels": ["low"],
            "prohibited_meaning": "把低利他写成冷漠、不回应或把责任完全推给他人",
            "required_boundary": "低利他应保持现实边界和基本回应，不得成为明显冷漠选项",
        },
    ],
    "agreeableness_compliance": [
        {
            "rule_id": "compliance-no-self-erasure",
            "target_levels": ["high"],
            "prohibited_meaning": (
                "把高顺从实现为完全服从、直接放弃合理意见、偏好或正当边界"
            ),
            "required_boundary": (
                "高顺从应通过体面让步、缓和冲突或建设性妥协表达，同时保留合理边界"
            ),
        }
    ],
    "agreeableness_modesty": [
        {
            "rule_id": "modesty-no-contribution-erasure",
            "target_levels": ["high"],
            "prohibited_meaning": (
                "把高谦逊实现为否认、隐藏或完全不提真实贡献"
            ),
            "required_boundary": (
                "如实承认自己的贡献，同时不过度突出自己并给予他人适当认可"
            ),
        },
        {
            "rule_id": "modesty-low-no-arrogance",
            "target_levels": ["low"],
            "prohibited_meaning": "把低谦逊写成自大、贬低他人或声称自己不可替代",
            "required_boundary": "低谦逊可以积极展示真实成果，但不得贬低他人或夸大不可替代性",
        },
    ],
    "agreeableness_tender_mindedness": [
        {
            "rule_id": "tender-mindedness-not-extra-helping",
            "target_levels": ["low", "medium_low", "medium_high", "high"],
            "prohibited_meaning": (
                "仅通过替他人承担工作、投入更多时间或提供更多实际帮助来建立慈悲心等级"
            ),
            "required_boundary": (
                "选项可以包含助人行为，但等级的主要差异必须来自是否感知、理解并回应"
                "对方的情绪或处境，而不是投入的资源量；后者主要属于A3利他"
            ),
        }
    ],
}


class ConstructResolutionError(ValueError):
    """A user-correctable construct or inventory resolution failure."""

_INVENTORIES = {
    "neo_pi_r": {
        "inventory_id": "neo_pi_r",
        "inventory_name": "NEO-PI-R",
        "source_id": "legacy-neo-pi-r-resources",
        "citation_label": (
            "Costa, P. T., & McCrae, R. R. (1992/2010). "
            "NEO-PI-R Manual."
        ),
    },
}

# These are copied from openCode_for_SJT/interactive_main.py.
_DOMAIN_NAMES = {
    "neuroticism": ("神经质", "Neuroticism"),
    "extraversion": ("外向性", "Extraversion"),
    "openness": ("开放性", "Openness"),
    "agreeableness": ("宜人性", "Agreeableness"),
    "conscientiousness": ("尽责性", "Conscientiousness"),
}

_NEO_FFI_DIMENSION_CODES = {
    "neuroticism": "N",
    "extraversion": "E",
    "openness": "O",
    "agreeableness": "A",
    "conscientiousness": "C",
}

_INVENTORY_ALIASES = {
    "neo_pi_r": "neo_pi_r",
    "neo-pi-r": "neo_pi_r",
    "neopir": "neo_pi_r",
    "neo": "neo_pi_r",
}

def _normalized(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        str(value or "").strip().casefold(),
    )


def _domain_aliases(domain_id: str) -> tuple[str, ...]:
    name_zh, name_en = _DOMAIN_NAMES.get(
        domain_id,
        (domain_id, domain_id),
    )
    return domain_id, name_zh, name_en


def _facet_snapshot(
    facet_id: str,
    legacy_facet: Any,
) -> dict[str, Any]:
    """Project one legacy FacetDefinition without changing its content."""

    snapshot = {
        "facet_id": facet_id,
        "facet_name": legacy_facet.facet_name,
        "facet_name_en": legacy_facet.facet_name,
        "definition": legacy_facet.definition,
        "high_behavior": legacy_facet.high_trait_behavior,
        "low_behavior": legacy_facet.low_trait_behavior,
        "common_confounds": list(legacy_facet.common_confounds),
        "inappropriate_conditions": list(
            legacy_facet.inappropriate_contexts
        ),
        "forbidden_patterns": list(legacy_facet.forbidden_patterns),
        "option_design_rules": list(
            legacy_facet.option_design_rules
        ),
        "hard_constraints": deepcopy(
            _FACET_HARD_CONSTRAINTS.get(facet_id, [])
        ),
    }
    if facet_id == "neuroticism_impulsiveness":
        snapshot.update(
            {
                "definition": (
                    "面对即时诱惑、欲望或行动冲动时，较难抵抗并延迟满足的倾向。"
                ),
                "high_behavior": (
                    "短期诱惑出现时更容易立即满足欲望，即使它与已明确的延迟目标冲突"
                ),
                "low_behavior": (
                    "短期诱惑出现时仍能维持既定延迟目标，并以适度方式调节欲望"
                ),
            }
        )
        snapshot["common_confounds"].extend(
            [
                "未经充分思考便作决定主要涉及低审慎性（C6），不是N5的核心",
                "愤怒爆发主要涉及愤怒敌意（N2），不得作为N5的主要指标",
                "放弃任务可能涉及自律性或责任感，不能单独证明N5冲动性",
            ]
        )
        snapshot["inappropriate_conditions"].extend(
            [
                "需要作出重大或高风险决定的场景",
                "以发脾气、攻击或人际冲突为核心的场景",
                "以履约、罢工或是否坚持任务为核心的场景",
            ]
        )
        snapshot["forbidden_patterns"].extend(
            [
                "不要用未经思考的重大决定代表N5冲动性",
                "不要用发脾气、攻击或直接罢工代表N5冲动性",
            ]
        )
        snapshot["option_design_rules"] = [
            "四个选项面对相同的即时诱惑、延迟目标、信息和现实成本",
            "行为等级只改变对即时诱惑的调节程度，不改变计划能力或责任要求",
            "高冲动选项可以选择较快满足，但不得荒唐、违法、危险或攻击他人",
            "低冲动选项体现适度延迟或调节，不得写成完美自律或完全没有欲望",
        ]
    elif facet_id == "conscientiousness_competence":
        snapshot.update(
            {
                "high_behavior": (
                    "已有证据表明具备基本能力时，对普通挑战保持现实信心并愿意承担相称任务"
                ),
                "low_behavior": (
                    "即使已有相同能力证据，仍较易低估自己的应对能力并犹豫是否承担相称任务"
                ),
                "sjt_applicability": {
                    "status": "conditional",
                    "reason": (
                        "胜任感包含内部自我效能判断，行为型SJT只能间接观察，"
                        "容易与能力、焦虑、果断性、条理性和成就追求混淆"
                    ),
                    "requirements": [
                        "情境必须明确所有选项具有相同且足够的基本能力证据",
                        "四个选项必须保持知识、经验、资源、准备时间和任务难度相同",
                        "不得用主导发言、提高目标、追求完美、有条理拆解或更周全思考表现胜任感",
                        "不得用任务成功或失败结果反向证明胜任感",
                    ],
                },
            }
        )
        snapshot["common_confounds"].extend(
            [
                "易与外倾性果断性共现；不得用抢占话语权或主导他人表现胜任感",
                "易与成就追求共现；不得用主动加码、追求完美或高标准表现胜任感",
                "易与条理性和审慎性共现；不得用更有条理或思考更周全表现胜任感",
                "易与一般能力、知识、经验及神经质焦虑混淆",
            ]
        )
        snapshot["inappropriate_conditions"].extend(
            [
                "不同选项拥有不同知识、经验、资源、指导或准备时间",
                "依靠任务成功或失败结果证明是否胜任",
                "以主导他人、追求高标准或有条理拆解作为关键差异",
            ]
        )
        snapshot["option_design_rules"] = [
            "仅在满足sjt_applicability全部条件时生成该facet骨架",
            "高胜任感体现有证据支持的现实掌控预期，不是不受证据约束的自信",
            "低胜任感体现能力低估或额外确认需要，不得写成无能、逃责或推给别人",
            "选项不得通过果断表达、额外努力、条理性、审慎性或结果优势建立等级",
        ]
    return snapshot


def _build_inventory(inventory_id: str) -> dict[str, Any]:
    metadata = _INVENTORIES[inventory_id]
    legacy_facets = get_inventory(inventory_id)
    domains: dict[str, dict[str, Any]] = {}
    for facet_id, legacy_facet in legacy_facets.items():
        domain_id = legacy_facet.domain
        name_zh, name_en = _DOMAIN_NAMES.get(
            domain_id,
            (domain_id, domain_id),
        )
        domain = domains.setdefault(
            domain_id,
            {
                "domain_id": domain_id,
                "domain_name": name_zh,
                "domain_name_en": name_en,
                "facets": {},
            },
        )
        facet_snapshot = _facet_snapshot(facet_id, legacy_facet)
        # Keep the owning domain on each facet so a cross-domain selection can
        # still be traced after the profile is flattened for generation.
        facet_snapshot["domain_id"] = domain_id
        facet_snapshot["domain_name"] = name_zh
        facet_snapshot["domain_name_en"] = name_en
        domain["facets"][facet_id] = facet_snapshot
    return {
        **metadata,
        "inventory_version": REGISTRY_VERSION,
        "review_status": "legacy_resources_with_curated_sjt_overrides",
        "sources": [
            {
                "source_id": metadata["source_id"],
                "citation_label": metadata["citation_label"],
                "supported_claim": (
                    "Facet definitions copied from "
                    "openCode_for_SJT/sjt_system/resources.py"
                ),
            }
        ],
        "domains": domains,
    }


def _registry() -> dict[str, dict[str, Any]]:
    return {
        inventory_id: _build_inventory(inventory_id)
        for inventory_id in _INVENTORIES
    }


def list_inventories() -> list[dict[str, str]]:
    return [
        {
            "inventory_id": item["inventory_id"],
            "inventory_name": item["inventory_name"],
            "inventory_version": item["inventory_version"],
        }
        for item in _registry().values()
    ]


def construct_selection_catalog() -> list[dict[str, Any]]:
    """Return stable IDs and labels needed by requirement clarification."""

    catalog = []
    for inventory in _registry().values():
        all_facets = [
            facet
            for domain in inventory["domains"].values()
            for facet in domain["facets"].values()
        ]
        catalog.append({
            "inventory_id": inventory["inventory_id"],
            "inventory_name": inventory["inventory_name"],
            "all_domains_selection": {
                "domain_ids": list(inventory["domains"]),
                "facet_ids": [
                    str(facet["facet_id"]) for facet in all_facets
                ],
                "facet_count": len(all_facets),
            },
            "domains": [
                {
                    "domain_id": domain_id,
                    "domain_name": domain["domain_name"],
                    "domain_name_en": domain["domain_name_en"],
                    "facets": [
                        {
                            "facet_id": facet["facet_id"],
                            "facet_name": facet["facet_name"],
                            "facet_name_en": facet["facet_name_en"],
                            "definition": facet.get("definition"),
                            "high_behavior": facet.get("high_behavior"),
                            "low_behavior": facet.get("low_behavior"),
                        }
                        for facet in domain["facets"].values()
                    ],
                }
                for domain_id, domain in inventory["domains"].items()
            ],
        })
    return catalog


def resolve_construct_selection(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one explicit inventory/domain/facet selection by stable IDs."""

    if not isinstance(selection, Mapping):
        raise ConstructResolutionError("construct_selection 必须是对象")
    allowed_fields = {"inventory_id", "domain_ids", "facet_ids", "domain_id"}
    if not set(selection).issubset(allowed_fields) or not {
        "inventory_id",
        "facet_ids",
    }.issubset(selection):
        raise ConstructResolutionError(
            "construct_selection 必须包含 inventory_id、domain_ids、facet_ids"
        )
    inventory_id = selection.get("inventory_id")
    facet_ids = selection.get("facet_ids")
    if not isinstance(inventory_id, str) or not inventory_id:
        raise ConstructResolutionError("construct_selection.inventory_id 无效")
    if not isinstance(facet_ids, list) or not all(
        isinstance(facet_id, str) and facet_id for facet_id in facet_ids
    ):
        raise ConstructResolutionError("construct_selection.facet_ids 必须是字符串列表")
    if len(facet_ids) != len(set(facet_ids)):
        raise ConstructResolutionError("construct_selection.facet_ids 不得重复")

    inventory = _registry().get(inventory_id)
    if inventory is None:
        raise ConstructResolutionError(f"未知构念量表体系：{inventory_id}")

    requested_domain_ids = selection.get("domain_ids")
    if requested_domain_ids is None:
        # Compatibility for pre-domain_ids checkpoints and provider output.
        legacy_domain_id = selection.get("domain_id")
        if not isinstance(legacy_domain_id, str) or not legacy_domain_id:
            raise ConstructResolutionError(
                "construct_selection.domain_ids 必须是非空字符串列表"
            )
        legacy_normalized = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+",
            "",
            legacy_domain_id.casefold(),
        )
        if legacy_normalized in {"alldomains", "allfacets", "全部域", "全部分面"}:
            if facet_ids:
                facet_domain_map = {
                    facet_id: domain_id
                    for domain_id, domain in inventory["domains"].items()
                    for facet_id in domain["facets"]
                }
                requested_domain_ids = list(
                    dict.fromkeys(
                        facet_domain_map[facet_id]
                        for facet_id in facet_ids
                        if facet_id in facet_domain_map
                    )
                )
            else:
                requested_domain_ids = list(inventory["domains"])
        else:
            requested_domain_ids = [legacy_domain_id]
    if not isinstance(requested_domain_ids, list) or not requested_domain_ids:
        raise ConstructResolutionError(
            "construct_selection.domain_ids 必须是非空字符串列表"
        )
    if not all(isinstance(item, str) and item for item in requested_domain_ids):
        raise ConstructResolutionError(
            "construct_selection.domain_ids 必须是字符串列表"
        )
    requested_domain_ids = list(dict.fromkeys(requested_domain_ids))
    unknown_domains = [
        domain_id
        for domain_id in requested_domain_ids
        if domain_id not in inventory["domains"]
    ]
    if unknown_domains:
        raise ConstructResolutionError(
            "量表 " + inventory_id + " 中不存在 domain："
            + "、".join(unknown_domains)
        )
    domain_id = (
        requested_domain_ids[0]
        if len(requested_domain_ids) == 1
        else ALL_DOMAINS_DOMAIN_ID
    )
    if domain_id == ALL_DOMAINS_DOMAIN_ID:
        all_available = {
            facet_id: facet
            for domain in inventory["domains"].values()
            for facet_id, facet in domain["facets"].items()
        }
        available = {
            facet_id: facet
            for facet_id, facet in all_available.items()
            if facet.get("domain_id") in set(requested_domain_ids)
        }
        domain = {
            "domain_id": ALL_DOMAINS_DOMAIN_ID,
            "domain_name": "全部域",
            "domain_name_en": "All domains",
        }
        valid_selection_levels = {"inventory", "facet"}
    else:
        domain = inventory["domains"].get(domain_id)
        if domain is None:
            raise ConstructResolutionError(
                f"量表 {inventory_id} 中不存在 domain：{domain_id}"
            )
        available = domain["facets"]
        valid_selection_levels = {"domain", "facet"}
    unknown = [facet_id for facet_id in facet_ids if facet_id not in available]
    if unknown:
        raise ConstructResolutionError(
            "以下 facet 不属于所选 domain：" + "、".join(unknown)
        )
    selected_facets = (
        [deepcopy(available[facet_id]) for facet_id in facet_ids]
        if facet_ids
        else [deepcopy(facet) for facet in available.values()]
    )
    selection_level = (
        "inventory"
        if domain_id == ALL_DOMAINS_DOMAIN_ID
        and set(requested_domain_ids) == set(inventory["domains"])
        and (not facet_ids or len(selected_facets) == len(available))
        else ("facet" if facet_ids else "domain")
    )
    if selection_level not in valid_selection_levels:
        raise ConstructResolutionError("construct_selection 选择范围无效")
    resolved_domain_ids = list(
        dict.fromkeys(
            str(facet.get("domain_id"))
            for facet in selected_facets
            if facet.get("domain_id")
        )
    )
    if not resolved_domain_ids:
        resolved_domain_ids = (
            list(inventory["domains"])
            if domain_id == ALL_DOMAINS_DOMAIN_ID
            else [domain_id]
        )
    if set(requested_domain_ids) != set(resolved_domain_ids):
        raise ConstructResolutionError(
            "construct_selection.domain_ids 与所选 facet 的实际 domain 不一致"
        )
    snapshot = {
        "inventory_id": inventory_id,
        "inventory_name": inventory["inventory_name"],
        "inventory_version": inventory["inventory_version"],
        "review_status": inventory["review_status"],
        "selection_level": selection_level,
        "domain_id": domain_id,
        "domain_name": domain["domain_name"],
        "domain_name_en": domain["domain_name_en"],
        "facets": selected_facets,
        "domain_ids": resolved_domain_ids,
        "sources": deepcopy(inventory["sources"]),
        "resolution_source": "explicit_structured_selection",
    }
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    snapshot["profile_hash"] = hashlib.sha256(encoded).hexdigest()
    return snapshot


def construct_selection_from_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the canonical selection stored in TestSpecification."""

    domain_ids = list(profile.get("domain_ids") or [])
    if not domain_ids:
        domain_ids = list(
            dict.fromkeys(
                str(facet.get("domain_id"))
                for facet in profile.get("facets") or []
                if facet.get("domain_id")
            )
        )
    return {
        "inventory_id": str(profile["inventory_id"]),
        "domain_ids": [
            str(domain_id) for domain_id in domain_ids
        ],
        "facet_ids": [
            str(facet["facet_id"]) for facet in profile.get("facets") or []
        ] if profile.get("selection_level") in {"facet", "inventory"}
        else [],
    }


def construct_selection_label(selection: Mapping[str, Any]) -> str:
    """Return a concise human label for a validated structured selection."""

    profile = resolve_construct_selection(selection)
    if profile["selection_level"] == "domain":
        target = profile["domain_name"]
    elif profile["selection_level"] == "inventory":
        target = f"全部域（{len(profile.get('facets') or [])} 个 facet）"
    else:
        domains = "、".join(profile.get("domain_ids") or [])
        domain_note = f"（域：{domains}）" if len(profile.get("domain_ids") or []) > 1 else ""
        target = "、".join(
            str(facet["facet_name"]) for facet in profile["facets"]
        ) + domain_note
    return f"{profile['inventory_name']} / {target}"


def resolve_specification_profile(
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the construct profile from the canonical requirement spec.

    The text fallback is intentionally one-way checkpoint compatibility. New
    runtime state stores only ``construct_selection``.
    """

    selection = specification.get("construct_selection")
    if isinstance(selection, Mapping):
        return resolve_construct_selection(selection)
    target = specification.get("target_construct")
    if isinstance(target, str) and target.strip():
        return resolve_construct_profile(target)
    raise ConstructResolutionError("测验规格缺少有效的 construct_selection")


def _explicit_inventory(normalized_target: str) -> str | None:
    matches = {
        inventory_id
        for alias, inventory_id in _INVENTORY_ALIASES.items()
        if _normalized(alias) in normalized_target
    }
    return next(iter(matches), None)


def _facet_matches(
    inventory: Mapping[str, Any],
    normalized_target: str,
) -> list[tuple[str, dict[str, Any]]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for domain_id, domain in inventory["domains"].items():
        for facet in domain["facets"].values():
            aliases = (
                facet["facet_id"],
                facet["facet_name"],
                facet["facet_name_en"],
                *_FACET_USER_ALIASES.get(facet["facet_id"], ()),
            )
            if any(
                _normalized(alias) in normalized_target
                for alias in aliases
            ):
                matches.append((domain_id, facet))
    return matches


def _domain_matches(
    inventory: Mapping[str, Any],
    normalized_target: str,
) -> list[str]:
    return [
        domain_id
        for domain_id in inventory["domains"]
        if any(
            _normalized(alias) in normalized_target
            for alias in _domain_aliases(domain_id)
        )
    ]


def resolve_construct_profile(
    target_construct: str,
    *,
    default_inventory_id: str = DEFAULT_INVENTORY_ID,
) -> dict[str, Any]:
    """Resolve a NEO-PI-R domain or facet from the versioned registry."""

    if not isinstance(target_construct, str) or not target_construct.strip():
        raise ConstructResolutionError("target_construct 必须是非空文本")
    registry = _registry()
    normalized = _normalized(target_construct)
    explicit_inventory = _explicit_inventory(normalized)

    inventory_id = explicit_inventory
    resolution_source = "explicit_inventory"
    implicit_matches: list[tuple[str, str, dict[str, Any]]] = []
    if inventory_id is None:
        for candidate_id, candidate in registry.items():
            implicit_matches.extend(
                (candidate_id, domain_id, facet)
                for domain_id, facet in _facet_matches(
                    candidate,
                    normalized,
                )
            )
        unique_implicit = {
            (candidate_id, facet["facet_id"]): (
                candidate_id,
                domain_id,
                facet,
            )
            for candidate_id, domain_id, facet in implicit_matches
        }
        if len(unique_implicit) == 1:
            inventory_id = next(iter(unique_implicit.values()))[0]
            resolution_source = "unique_legacy_facet_match"
        elif len(unique_implicit) > 1:
            raise ConstructResolutionError("目标文本同时匹配多个 facet")
        else:
            inventory_id = default_inventory_id
            resolution_source = (
                f"legacy_default_inventory:{default_inventory_id}"
            )

    inventory = registry.get(inventory_id)
    if inventory is None:
        raise ConstructResolutionError(
            f"未知构念量表体系：{inventory_id}"
        )

    all_markers = (
        "全部facet", "所有facet", "全部分面", "所有分面",
        "每个facet", "各个facet", "全部维度", "所有维度",
        "每个维度", "各个维度", "allfacets", "entireinventory",
        "wholeinventory", "alldomains",
    )
    inventory_markers = (
        "neopir", "neopi", "量表", "inventory", "domains",
    )
    all_scope_markers = (
        "全部facet", "所有facet", "全部分面", "所有分面",
        "每个facet", "各个facet", "全部维度", "所有维度",
        "每个维度", "各个维度",
    )
    all_inventory_request = (
        any(token in normalized for token in all_markers)
        and (
            any(token in normalized for token in inventory_markers)
            or any(token in normalized for token in all_scope_markers)
        )
    ) or (
        any(token in normalized for token in ("整个", "全量"))
        and any(token in normalized for token in inventory_markers)
    )
    if all_inventory_request:
        all_facets = [
            deepcopy(facet)
            for domain in inventory["domains"].values()
            for facet in domain["facets"].values()
        ]
        snapshot = {
            "inventory_id": inventory_id,
            "inventory_name": inventory["inventory_name"],
            "inventory_version": inventory["inventory_version"],
            "review_status": inventory["review_status"],
            "selection_level": "inventory",
            "domain_id": ALL_DOMAINS_DOMAIN_ID,
            "domain_name": "全部域",
            "domain_name_en": "All domains",
            "domain_ids": list(inventory["domains"]),
            "facets": all_facets,
            "sources": deepcopy(inventory["sources"]),
            "resolution_source": resolution_source,
        }
        encoded = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        snapshot["profile_hash"] = hashlib.sha256(encoded).hexdigest()
        return snapshot

    facet_candidates = _facet_matches(inventory, normalized)
    if facet_candidates:
        unique_facets = {
            facet["facet_id"]: (domain_id, facet)
            for domain_id, facet in facet_candidates
        }
        if len(unique_facets) == 1:
            domain_id, selected_facet = next(iter(unique_facets.values()))
            selection_level = "facet"
            selected_facets = [deepcopy(selected_facet)]
        else:
            matched_domains = {
                str(domain_id)
                for domain_id, _facet in unique_facets.values()
            }
            domain_id = (
                next(iter(matched_domains))
                if len(matched_domains) == 1
                else ALL_DOMAINS_DOMAIN_ID
            )
            selection_level = "facet"
            selected_facets = [
                deepcopy(facet)
                for _domain_id, facet in unique_facets.values()
            ]
    else:
        domain_candidates = list(
            dict.fromkeys(_domain_matches(inventory, normalized))
        )
        if len(domain_candidates) != 1:
            raise ConstructResolutionError(
                "无法识别目标构念；请使用旧版的五个 domain 名称，"
                "或先选择 NEO-PI-R 后选择 facet"
            )
        domain_id = domain_candidates[0]
        selection_level = "domain"
        selected_facets = [
            deepcopy(facet)
            for facet in inventory["domains"][domain_id][
                "facets"
            ].values()
        ]

    if domain_id == ALL_DOMAINS_DOMAIN_ID:
        domain = {
            "domain_id": ALL_DOMAINS_DOMAIN_ID,
            "domain_name": "全部域",
            "domain_name_en": "All domains",
        }
        domain_ids = list(
            dict.fromkeys(
                str(facet.get("domain_id"))
                for facet in selected_facets
                if facet.get("domain_id")
            )
        )
    else:
        domain = inventory["domains"][domain_id]
        domain_ids = [domain_id]
    snapshot = {
        "inventory_id": inventory_id,
        "inventory_name": inventory["inventory_name"],
        "inventory_version": inventory["inventory_version"],
        "review_status": inventory["review_status"],
        "selection_level": selection_level,
        "domain_id": domain_id,
        "domain_name": domain["domain_name"],
        "domain_name_en": domain["domain_name_en"],
        "facets": selected_facets,
        "domain_ids": domain_ids,
        "sources": deepcopy(inventory["sources"]),
        "resolution_source": resolution_source,
    }
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    snapshot["profile_hash"] = hashlib.sha256(encoded).hexdigest()
    return snapshot


def resolve_neo_ffi_criterion(
    *,
    construct_profile: Mapping[str, Any] | None = None,
    target_construct: str | None = None,
) -> dict[str, str]:
    """Resolve the target Big Five domain to its Neo-FFI criterion scale."""

    profile_domain = (
        construct_profile.get("domain_id")
        if isinstance(construct_profile, Mapping)
        else None
    )
    resolved_profile = None
    if isinstance(target_construct, str) and target_construct.strip():
        resolved_profile = resolve_construct_profile(target_construct)
    target_domain = (
        resolved_profile.get("domain_id")
        if isinstance(resolved_profile, Mapping)
        else None
    )
    if (
        isinstance(profile_domain, str)
        and isinstance(target_domain, str)
        and profile_domain != target_domain
    ):
        raise ValueError(
            "construct_profile.domain_id 与 "
            "test_specification.target_construct 不一致"
        )
    domain_id = profile_domain or target_domain
    if not isinstance(domain_id, str) or not domain_id:
        raise ValueError(
            "无法确定目标 Big Five domain；心理测量分析不能猜测 Neo-FFI "
            "参照维度"
        )
    dimension_code = _NEO_FFI_DIMENSION_CODES.get(domain_id)
    if dimension_code is None:
        raise ValueError(
            f"目标 domain {domain_id!r} 没有受支持的 Neo-FFI 映射"
        )
    name_zh, name_en = _DOMAIN_NAMES[domain_id]
    return {
        "domain_id": domain_id,
        "domain_name": name_zh,
        "domain_name_en": name_en,
        "neo_ffi_dimension": dimension_code,
    }


def get_facet(
    profile: dict[str, Any],
    facet_id: str,
) -> dict[str, Any]:
    facet = next(
        (
            candidate
            for candidate in profile.get("facets") or []
            if candidate.get("facet_id") == facet_id
        ),
        None,
    )
    if facet is None:
        raise ValueError(f"构念档案中不存在 facet：{facet_id}")
    return deepcopy(facet)
