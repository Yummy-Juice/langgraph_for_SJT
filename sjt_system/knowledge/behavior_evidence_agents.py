"""Single-call behavior-evidence extraction and cache orchestration."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

from sjt_system.agent.retry import ainvoke_model_with_schema_repair
from sjt_system.knowledge.behavior_evidence import (
    BehaviorEvidenceAgentOutput,
    BehaviorEvidenceBundle,
    IPIPCorpus,
    NEO_FACET_CODE_TO_ID,
    create_behavior_evidence_bundle,
    find_behavior_evidence,
    get_ipip_scale,
)
from sjt_system.prompt.behavior_evidence_prompt import BEHAVIOR_EVIDENCE_PROMPT


def _facet_context(facet_code: str) -> dict[str, Any]:
    from sjt_system.authoring.construct_registry import resolve_construct_selection

    facet_id = NEO_FACET_CODE_TO_ID[facet_code.upper()]
    domain_id = facet_id.split("_", 1)[0]
    profile = resolve_construct_selection(
        {
            "inventory_id": "neo_pi_r",
            "domain_id": domain_id,
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
) -> BehaviorEvidenceBundle:
    cached = find_behavior_evidence(facet_id)
    if cached is not None:
        return cached
    raise ValueError(
        "unsupported_construct: facet 缺少已审核的 curated Behavior Evidence: "
        f"{facet_id}"
    )
