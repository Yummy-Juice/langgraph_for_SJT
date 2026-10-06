"""Single-call behavior-evidence extraction and cache orchestration."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

from sjt_system.agent.retry import ainvoke_model_with_schema_repair
from sjt_system.knowledge.behavior_evidence import (
    BehaviorEvidenceAgentOutput,
    BehaviorEvidenceBundle,
    BehaviorEvidenceRecord,
    IPIPCorpus,
    NEO_FACET_CODE_TO_ID,
    NEO_FACET_ID_TO_CODE,
    behavior_source_fingerprint,
    create_behavior_evidence_bundle,
    find_behavior_evidence,
    get_ipip_scale,
)
from sjt_system.prompt.behavior_evidence_prompt import BEHAVIOR_EVIDENCE_PROMPT
from sjt_system.runtime.trace import utc_timestamp


def _facet_context(facet_code: str) -> dict[str, Any]:
    from sjt_system.authoring.construct_registry import resolve_construct_selection

    facet_id = NEO_FACET_CODE_TO_ID[facet_code.upper()]
    domain_id = facet_id.split("_", 1)[0]
    profile = resolve_construct_selection(
        {
            "inventory_id": "neo_pi_r",
            "domain_ids": [domain_id],
            "facet_ids": [facet_id],
        }
    )
    return deepcopy(profile["facets"][0])


def create_behavior_evidence_agent() -> Any:
    from sjt_system.agent.agent_factory import create_agent

    model_id = os.getenv("BEHAVIOR_EVIDENCE_MODEL_ID") or None
    temperature = float(os.getenv("BEHAVIOR_EVIDENCE_TEMPERATURE", "0.2"))
    return create_agent(
        BEHAVIOR_EVIDENCE_PROMPT,
        BehaviorEvidenceAgentOutput,
        model_id=model_id,
        temperature=temperature,
        # The prompt carries its own explicit output contract with a complete
        # example record, so the verbose machine-generated JSON Schema is not
        # appended here.
        include_json_schema=False,
    )


async def mine_behavior_evidence(
    facet_code: str,
    corpus: IPIPCorpus,
    *,
    miner: Any | None = None,
) -> BehaviorEvidenceBundle:
    code = facet_code.upper()
    facet = _facet_context(code)
    scale = get_ipip_scale(corpus, code)
    agent = miner or create_behavior_evidence_agent()
    raw = await ainvoke_model_with_schema_repair(
        agent,
        {
            "input_data": {
                "facet_profile": facet,
                "ipip_scale": scale.model_dump(mode="json"),
            }
        },
        job_label=f"行为证据抽取-{code}",
    )
    return create_behavior_evidence_bundle(
        facet_code=code,
        facet=facet,
        corpus=corpus,
        output=raw,
    )


async def ensure_behavior_evidence(
    facet_id: str,
    corpus: IPIPCorpus,
    *,
    miner: Any | None = None,
    allow_legacy_fallback: bool = False,
) -> BehaviorEvidenceBundle:
    cached = find_behavior_evidence(facet_id)
    if cached is not None:
        return cached
    if allow_legacy_fallback:
        # Multi-facet runs may include registered NEO-PI-R facets that do not
        # yet have a curated evidence-library file. Keep generation moving by
        # building a transparent, registry-backed candidate bundle. It is
        # marked as a fallback and must not be confused with SME-reviewed
        # evidence in reports.
        code = NEO_FACET_ID_TO_CODE.get(str(facet_id))
        if code is None:
            raise ValueError(f"未知 NEO facet：{facet_id}")
        facet = _facet_context(code)
        scale = get_ipip_scale(corpus, code)
        positive = next(
            (item for item in scale.items if item.polarity == "positive"),
            scale.items[0],
        )
        negative = next(
            (item for item in scale.items if item.polarity == "negative"),
            scale.items[-1],
        )
        boundary = (
            "只依据该 facet 的定义和行为边界；"
            + "；".join(str(value) for value in (facet.get("common_confounds") or [])[:3])
        )
        return BehaviorEvidenceBundle(
            schema_version="behavior-evidence-v2",
            facet_code=code,
            facet_id=str(facet_id),
            source_fingerprint=behavior_source_fingerprint(facet, corpus),
            generated_at=f"legacy_registry_fallback:{utc_timestamp()}",
            evidence=[
                BehaviorEvidenceRecord(
                    behavior_id=f"{code}_LEGACY_BE01",
                    behavior_dimension=str(facet.get("facet_name") or facet_id),
                    observable_behavior=str(facet.get("definition") or facet_id),
                    high_expression=str(
                        facet.get("high_behavior")
                        or facet.get("definition")
                        or facet_id
                    ).strip(),
                    low_expression=str(
                        facet.get("low_behavior")
                        or facet.get("definition")
                        or facet_id
                    ).strip(),
                    boundary_condition=boundary,
                    source_item_ids=[positive.item_id, negative.item_id],
                )
            ],
        )
    raise ValueError(
        "unsupported_construct: facet 缺少已审核的 curated Behavior Evidence: "
        f"{facet_id}"
    )
