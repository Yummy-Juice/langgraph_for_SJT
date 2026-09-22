"""Population-specific behavior expansion and two-way blueprint proposals."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import os
from typing import Any

from pydantic import Field

from sjt_system.agent.agent_factory import mechanism_validation_agent
from sjt_system.agent.retry import ainvoke_model_with_schema_repair
from sjt_system.knowledge.behavior_evidence import StrictModel
from sjt_system.runtime.io import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_KNOWLEDGE_ROOT = PROJECT_ROOT / "outputs" / "run_knowledge"

_MAX_VALIDATION_RETRIES = 2
BLUEPRINT_SEMANTIC_RETRY_ATTEMPTS = 2
INCREMENTAL_CANDIDATES_PER_CELL = 2


class SituationDraft(StrictModel):
    domain: str = Field(min_length=1, max_length=100)
    actor_relation: str = Field(min_length=1, max_length=160)
    event_class: str = Field(min_length=1, max_length=200)


class MechanismDraft(StrictModel):
    activation_mechanism: str = Field(min_length=1, max_length=400)
    situations: list[SituationDraft] = Field(min_length=1)


class BehaviorExpansionDraft(StrictModel):
    behavior_id: str = Field(min_length=1)
    mechanisms: list[MechanismDraft] = Field(min_length=1, max_length=4)


class FacetExpansionAgentOutput(StrictModel):
    behavior_expansions: list[BehaviorExpansionDraft] = Field(min_length=1)


class SituationRecord(SituationDraft):
    situation_id: str = Field(min_length=1)


class MechanismRecord(StrictModel):
    mechanism_id: str = Field(min_length=1)
    activation_mechanism: str = Field(min_length=1, max_length=400)
    situations: list[SituationRecord] = Field(min_length=1)


class BehaviorExpansionRecord(StrictModel):
    behavior_id: str = Field(min_length=1)
    mechanisms: list[MechanismRecord] = Field(min_length=1, max_length=4)


class FacetExpansion(StrictModel):
    facet_id: str = Field(min_length=1)
    target_population: str = Field(min_length=1)
    behavior_expansions: list[BehaviorExpansionRecord] = Field(min_length=1)


class BlueprintCandidateReference(StrictModel):
    mechanism_id: str = Field(min_length=1)
    situation_id: str = Field(min_length=1)


class BlueprintRowDraft(StrictModel):
    facet_id: str = Field(min_length=1)
    behavior_id: str = Field(min_length=1)
    candidate_references: list[BlueprintCandidateReference] = Field(
        min_length=INCREMENTAL_CANDIDATES_PER_CELL,
        max_length=INCREMENTAL_CANDIDATES_PER_CELL,
    )


class BlueprintAgentOutput(StrictModel):
    rows: list[BlueprintRowDraft] = Field(min_length=1)


EXPANSION_PROMPT = """
Expand all supplied behavior evidence for one personality facet into a compact,
theory-grounded, and population-appropriate situation space. Do not write SJT
stems, response options, scores, reviews, rationales, or workflow metadata.

INPUT AUTHORITY

Use the supplied facet_profile to determine:

* the meaning and direction of the target facet;
* its boundaries from adjacent facets, ability, morality, and social
  desirability;
* the psychological processes that are admissible for the target construct.

Use the supplied behavior evidence to determine:

* the observable behavioral difference to be activated;
* the level of abstraction of the expansion;
* which behavioral processes may be represented.

Use supplied theory_notes, when available, to determine:

* which situational cues can plausibly activate the behavior difference;
* which competing goals, priorities, or preferences are theoretically relevant;
* why different expressions of the behavior may emerge.

Use target_population to constrain:

* domains and actor relations;
* realistic authority, resources, and opportunities;
* event frequency and stakes;
* cultural and developmental appropriateness.

Do not invent a psychological conflict merely because it is generally
plausible. A proposed mechanism must be supported by the facet_profile,
theory_notes, or the immediate psychological implications of the supplied
behavior evidence.

OUTPUT SCOPE

The input includes required_situation_count. Across all behavior expansions,
return exactly that many unique situation entries. Distribute the required
situations across supported mechanisms without forcing extra mechanisms. A
mechanism may contain one or more situations. Do not duplicate situations
through superficial wording changes merely to satisfy capacity. Do not return
more or fewer situation entries than requested.

For each supplied behavior_id, return exactly one behavior expansion. Preserve
the supplied behavior_id and do not create new IDs.

Each behavior expansion contains 1–4 activation mechanisms. The number must vary
with the theoretical and behavioral complexity of the evidence. Do not generate
additional mechanisms merely to reach a preferred count.

Each activation mechanism must be one clear sentence that specifies:

1. the trait-relevant situational cue;
2. two competing goals, priorities, or preferences that are both legitimate in
   ordinary circumstances;
3. why the situation leaves meaningful behavioral discretion;
4. the observable decision pattern through which the supplied behavior
   difference may emerge.

An activation mechanism must explain when and why the behavior evidence may be
expressed. It must not merely restate the high and low endpoints.

DISTINCTNESS OF ACTIVATION MECHANISMS

Activation mechanisms must represent substantively different activation
processes.

Two mechanisms are distinct only when at least one of the following differs
meaningfully:

* the trait-relevant cue;
* the competing psychological priorities;
* the reason behavioral discretion is preserved;
* the observable choice through which the trait is expressed.

Do not create separate mechanisms merely by changing:

* location;
* actor identity;
* actor relation;
* organization type;
* academic subject;
* surface event wording.

Merge mechanisms that differ only in setting or surface realization. Prefer one
well-supported mechanism over several semantically redundant mechanisms.

TRAIT ACTIVATION AND SITUATIONAL STRENGTH

Each mechanism must provide an opportunity for the target behavior difference
to be expressed.

Use weak situations in which:

* no explicit rule dictates one response;
* no option is clearly illegal, unsafe, cruel, deceptive, or irresponsible;
* consequences are meaningful but not extreme;
* the focal person has realistic choice;
* both competing paths retain some practical or psychological justification.

Do not use strong situations in which:

* formal rules clearly require one action;
* severe punishment makes one response unavoidable;
* safety or emergency demands determine the response;
* one path is obviously moral and the other obviously harmful;
* one path requires incompetence, recklessness, or implausible bad intent.

Each mechanism must contain two legitimate competing priorities. Do not contrast
a facet-relevant tendency with negligence, dishonesty, cruelty, inability,
general irresponsibility, or an obviously inferior decision.

CONSTRUCT PURITY

Preserve construct purity when converting behavior evidence into activation
mechanisms.

Do not turn the target distinction into:

* lawfulness versus illegality;
* rule compliance versus violation;
* obedience versus defiance;
* morality versus immorality;
* helping versus harming;
* competence versus incompetence;
* emotional maturity versus dysfunction;
* socially approved versus socially condemned behavior.

When the source evidence mentions cheating, rule bending, exploitation,
aggression, irresponsibility, or harm, retain only a directly supported
facet-relevant process. Do not use the misconduct itself as the
situation-defining conflict.

Do not rehabilitate a contaminated source example by inventing an unsupported
process such as information asymmetry, privacy, disclosure, trust, negotiation,
or impression management. If the relevant process is not directly supported by
the behavior evidence or supplied theory, do not use it.

A mechanism must be explained more strongly by the target facet than by another
personality facet, general ability, social skill, professional knowledge,
morality, or situational demand.

OBSERVABILITY

Define mechanisms through observable differences in what the person:

* says;
* discloses;
* withholds;
* emphasizes;
* chooses;
* postpones;
* initiates;
* continues;
* changes across comparable conditions.

Do not define a mechanism through inaccessible motives such as:

* genuine versus false concern;
* sincerity versus insincerity;
* hidden selfishness;
* malicious intent;
* unconscious goals;
* a desire to manipulate.

When such a latent process is relevant, express it through observable indicators
such as:

* changes in communication before and after a request;
* consistency across comparable audiences;
* timing of relational expressions;
* inclusion or omission of decision-relevant information;
* differences between stated reasons and subsequent action.

Indirect communication, politeness, persuasion, selective emphasis, privacy
protection, strategic timing, or negotiation alone is not sufficient evidence
of a low trait expression.

SITUATION ENTRIES

Each activation mechanism may contain one or more situation entries. The total
across this facet, rather than a per-mechanism quota, must satisfy
required_situation_count.

Each situation entry contains exactly:

* domain;
* actor_relation;
* event_class.

domain identifies a broad area of ordinary life relevant to target_population.

actor_relation identifies the relationship and any psychologically relevant
role feature, without inventing unnecessary personal detail.

event_class must describe the trait-relevant cue and decision tension,
not merely name a generic activity. Avoid broad labels such as:

* asking for help;
* working in a group;
* attending an activity;
* giving feedback;
* resolving a conflict.

The event_class must be abstract enough to support multiple later SJT
items but specific enough to preserve:

* the activation cue;
* the competing priorities;
* the behavioral discretion;
* the intended observable distinction.

POPULATION APPROPRIATENESS

Every situation entry must:

* be reasonably likely in the ordinary life of target_population;
* give the focal person realistic authority, information, and access;
* avoid requiring specialized professional expertise unless supplied;
* use realistic stakes, resources, and time pressure;
* avoid rare emergencies, extreme misconduct, or culturally implausible events;
* avoid assuming relationships or responsibilities uncommon in the population.

FINAL INTERNAL CHECK

Before returning, silently verify each activation mechanism:

* Is it directly supported by the behavior evidence, facet profile, or supplied
  theory?
* Does it contain a trait-relevant cue rather than a generic setting?
* Does it contain two legitimate competing priorities?
* Does the situation preserve meaningful behavioral discretion?
* Are the behavioral differences observable?
* Is the mechanism explained more strongly by the target facet than by another
  construct?
* Is it distinct from all other mechanisms for the same behavior_id?
* Can it support a weak-situation SJT without creating an obvious correct
  response?
* Do the situation entries preserve the same activation mechanism rather than
  introduce new conflicts?

Revise, merge, or discard any mechanism that fails a check.

Return only the requested structured object. Do not include commentary,
explanations, citations, construct boundaries, scores, reviews, or text outside
the object.
""".strip()


BLUEPRINT_PROMPT = """
Design a PSJT two-way specification table from the supplied facets, behavior
evidence expansions, and the exact requested final item count. The measurement
axis is facet plus behavior_id. The candidate situation axis is mechanism_id
plus situation_id.

Return exactly row_total unique measurement rows, ordered by construct
coverage. row_total equals the requested final item count. Each row must
contain exactly two candidate_references, so the program can generate two
candidate items for the same measurement cell and retain one later. Expansion
has prepared a situation pool larger than generation_total; select exactly
generation_total unique situation references from it.
Do not return generation or retention counts.

Every candidate reference must cite IDs exactly as supplied. The two
candidates in one row must use different situation references. Prefer
different activation mechanisms when the evidence supports them; when they
share a mechanism, their situation entries must represent substantively
different pressure structures or competing priorities, not merely different
locations, actors, names, or wording. Do not create or infer IDs.

Prioritize construct representation: cover every supplied behavior_id when the
requested count allows, and cover multiple mechanisms within a behavior when
available candidate references allow. Do not concentrate rows merely to
simplify selection. Do not generate item content, explanations, row IDs, or
duplicated combinations.

EVIDENCE-GUIDED SELECTION

Each supplied facet includes its behavior_evidence. Use each evidence
entry's behavior_dimension, high_expression, low_expression, and
boundary_condition to validate every candidate mechanism before
selecting it.

Reject any mechanism whose activation_mechanism:

* describes a behavioral pattern not covered by the evidence's
  high_expression or low_expression — such as persistence versus
  stopping engagement, social participation versus withdrawal,
  directness versus indirectness, or general compliance versus
  defiance;
* relies on a behavioral signal the evidence's boundary_condition
  explicitly excludes;
* would still be selectable if the evidence's high_expression were
  removed — i.e., the mechanism is not actually constrained by the
  facet's defined behavior.

When a mechanism is rejected, do not select any of its situations.
Prefer mechanisms whose activation_mechanism operates on the same
behavioral dimension the evidence defines, so that both candidates in every
row remain construct-pure.

If blueprint_semantic_feedback is supplied in the task input, treat it as a hard
correction to the previous proposal. Replace the cited candidate references
and re-check global uniqueness across all rows before returning the object.

If the available mechanisms cannot supply row_total rows with two unique,
construct-pure candidate references each, return the maximum construct-pure
subset and let the program raise an error rather than padding with out-of-
scope mechanisms or repeating a situation.
""".strip()


def _agent(prompt: str, output_type: type, prefix: str, temperature: float) -> Any:
    from sjt_system.agent.agent_factory import create_agent

    model_id = os.getenv(f"{prefix}_MODEL_ID") or None
    value = float(os.getenv(f"{prefix}_TEMPERATURE", str(temperature)))
    return create_agent(prompt, output_type, model_id=model_id, temperature=value)


def create_expansion_agent() -> Any:
    return _agent(EXPANSION_PROMPT, FacetExpansionAgentOutput, "EXPANSION", 0.5)


def create_blueprint_agent() -> Any:
    return _agent(BLUEPRINT_PROMPT, BlueprintAgentOutput, "BLUEPRINT", 0.3)


def assign_expansion_ids(
    facet_id: str,
    target_population: str,
    output: FacetExpansionAgentOutput | Mapping[str, Any],
) -> FacetExpansion:
    result = (
        output
        if isinstance(output, FacetExpansionAgentOutput)
        else FacetExpansionAgentOutput.model_validate(output)
    )
    expansions = []
    for behavior in result.behavior_expansions:
        mechanisms = []
        for mechanism_index, mechanism in enumerate(behavior.mechanisms, start=1):
            mechanism_id = f"{behavior.behavior_id}_M{mechanism_index:02d}"
            mechanisms.append(
                MechanismRecord(
                    mechanism_id=mechanism_id,
                    activation_mechanism=mechanism.activation_mechanism,
                    situations=[
                        SituationRecord(
                            situation_id=f"{mechanism_id}_S{index:02d}",
                            **situation.model_dump(),
                        )
                        for index, situation in enumerate(
                            mechanism.situations, start=1
                        )
                    ],
                )
            )
        expansions.append(
            BehaviorExpansionRecord(
                behavior_id=behavior.behavior_id,
                mechanisms=mechanisms,
            )
        )
    return FacetExpansion(
        facet_id=facet_id,
        target_population=target_population,
        behavior_expansions=expansions,
    )


def expansion_cache_path(run_id: str, facet_id: str) -> Path:
    from sjt_system.runtime.output_paths import scoped_output
    return scoped_output("run_knowledge", DEFAULT_RUN_KNOWLEDGE_ROOT) / run_id / "expansions" / f"{facet_id}.json"


def load_facet_expansion(path: Path | str) -> FacetExpansion:
    return FacetExpansion.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


async def ensure_facet_expansion(
    *,
    run_id: str,
    facet: Mapping[str, Any],
    behavior_evidence: list[Mapping[str, Any]],
    target_population: str,
    output_language: str,
    required_situation_count: int = 1,
    agent: Any | None = None,
) -> FacetExpansion:
    required_situation_count = max(1, int(required_situation_count))
    path = expansion_cache_path(run_id, str(facet["facet_id"]))
    if path.exists():
        cached = load_facet_expansion(path)
        # 数量满足“至少”即可：蓝图从情境池中选取所需条目，池子偏大不影响
        if expansion_situation_count(cached) >= required_situation_count:
            return cached
    runnable = agent or create_expansion_agent()
    last_count = 0
    for attempt in range(_MAX_VALIDATION_RETRIES + 1):
        revision = ""
        if attempt:
            revision = (
                f"上一候选有 {last_count} 个情境条目，但至少需要 "
                f"{required_situation_count} 个。保持现有 Behavior Evidence 和"
                "实质机制边界；如果数量不足，增加真实不同的情境条目；"
                "不要为凑数创建重复机制或同质情境。"
            )
        expansion = await _generate_expansion(
            runnable,
            facet,
            behavior_evidence,
            target_population,
            output_language,
            required_situation_count,
            extra_context=revision,
        )
        last_count = expansion_situation_count(expansion)
        if last_count < required_situation_count:
            continue
        # 数量满足“至少”即接受（不足才重试，偏大直接用）；
        # Expansion 是离线设计提案，语义质量由统一审题把关。
        break
    else:
        raise ValueError(
            f"Behavior Expansion 最终提供 {last_count} 个情境条目，"
            f"但至少需要 {required_situation_count} 个"
        )
    expected_behavior_ids = {
        str(row["behavior_id"]) for row in behavior_evidence
    }
    actual_behavior_ids = [
        row.behavior_id for row in expansion.behavior_expansions
    ]
    if (
        set(actual_behavior_ids) != expected_behavior_ids
        or len(actual_behavior_ids) != len(expected_behavior_ids)
    ):
        raise ValueError("Expansion 必须逐一引用当前 facet 的行为证据")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expansion.model_dump(mode="json"))
    return expansion


def expansion_situation_count(expansion: FacetExpansion) -> int:
    return sum(
        len(mechanism.situations)
        for behavior in expansion.behavior_expansions
        for mechanism in behavior.mechanisms
    )


async def _generate_expansion(
    runnable: Any,
    facet: Mapping[str, Any],
    behavior_evidence: list[Mapping[str, Any]],
    target_population: str,
    output_language: str,
    required_situation_count: int,
    extra_context: str = "",
) -> FacetExpansion:
    input_data: dict[str, Any] = {
        "facet_profile": dict(facet),
        "behavior_evidence": [dict(row) for row in behavior_evidence],
        "target_population": target_population,
        "output_language": output_language,
        "required_situation_count": required_situation_count,
    }
    if extra_context:
        input_data["revision_instruction"] = extra_context
    raw = await ainvoke_model_with_schema_repair(
        runnable,
        {"input_data": input_data},
        job_label=f"行为情境扩展-{facet['facet_id']}",
    )
    return assign_expansion_ids(
        str(facet["facet_id"]), target_population, raw
    )


async def _validate_and_repair_mechanisms(
    expansion: FacetExpansion,
    runnable: Any,
    facet: Mapping[str, Any],
    behavior_evidence: list[Mapping[str, Any]],
    target_population: str,
    output_language: str,
    required_situation_count: int,
) -> FacetExpansion:
    neighbor_facets = _neighbor_facet_definitions(facet)
    for retry in range(_MAX_VALIDATION_RETRIES + 1):
        failed_mechanisms: list[tuple[int, int, str, str]] = []
        for bi, behavior in enumerate(expansion.behavior_expansions):
            for mi, mechanism in enumerate(behavior.mechanisms):
                result = await _validate_one_mechanism(
                    mechanism.activation_mechanism,
                    facet,
                    neighbor_facets,
                )
                if not result.get("target_is_first"):
                    failed_mechanisms.append((
                        bi, mi,
                        mechanism.activation_mechanism,
                        result.get("reason", ""),
                    ))

        if not failed_mechanisms:
            return expansion

        if retry >= _MAX_VALIDATION_RETRIES:
            _discard_failed_mechanisms(expansion, failed_mechanisms)
            return expansion

        revision = _build_validation_feedback(failed_mechanisms)
        expansion = await _generate_expansion(
            runnable,
            facet,
            behavior_evidence,
            target_population,
            output_language,
            required_situation_count,
            extra_context=revision,
        )
    return expansion


async def _validate_one_mechanism(
    mechanism_text: str,
    facet: Mapping[str, Any],
    neighbor_facets: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = await mechanism_validation_agent.ainvoke({
        "input_data": {
            "target_facet": dict(facet),
            "neighbor_facets": neighbor_facets,
            "mechanism": mechanism_text,
        }
    })
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, Mapping):
        return dict(raw)
    raise ValueError(f"校验 agent 返回了非预期的类型：{type(raw).__name__}")


def _neighbor_facet_definitions(
    target_facet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    confounds = target_facet.get("common_confounds") or []
    if isinstance(confounds, str):
        confounds = [confounds]
    ambiguous_contexts = target_facet.get("inappropriate_conditions") or []
    if isinstance(ambiguous_contexts, str):
        ambiguous_contexts = [ambiguous_contexts]
    return [
        {
            "note": f"以下内容摘自目标 facet 的 common_confounds 和 inappropriate_conditions，"
                    f"是已知的可能与 {target_facet.get('facet_id', '')} 混淆的构念领域",
            "items": [str(item) for item in [*confounds, *ambiguous_contexts]],
        }
    ]


def _build_validation_feedback(
    failed: list[tuple[int, int, str, str]],
) -> str:
    lines = [
        "以下 activation mechanism 未能通过构念纯度校验，"
        "它们描述的行为差异更可能由相邻 facet 而非目标 facet 驱动。"
        "请重写这些 mechanism，确保核心行为差异由目标 facet 驱动。",
        "",
    ]
    for bi, mi, text, reason in failed:
        lines.append(f"失败的 mechanism: {text}")
        lines.append(f"校验判断: {reason}")
        lines.append("")
    lines.append(
        "修改要点：改变行为差异的载体，使高-低差异体现在的目标 facet 的核心行为维度上，"
        "而不是相邻 facet 的行为维度。"
    )
    return "\n".join(lines)


def _discard_failed_mechanisms(
    expansion: FacetExpansion,
    failed: list[tuple[int, int, str, str]],
) -> None:
    discarded_indices: dict[int, set[int]] = {}
    for bi, mi, _text, _reason in failed:
        discarded_indices.setdefault(bi, set()).add(mi)

    for bi in sorted(discarded_indices, reverse=True):
        behavior = expansion.behavior_expansions[bi]
        surviving = [
            deepcopy(m) for mi, m in enumerate(behavior.mechanisms)
            if mi not in discarded_indices.get(bi, set())
        ]
        if not surviving:
            raise ValueError(
                f"behavior_id={behavior.behavior_id} 的所有 mechanism "
                f"均未通过构念纯度校验且已达重试上限。"
                f"请检查 behavior evidence 与 facet 定义是否一致，"
                f"或人工审核该 facet 的行为证据。"
            )
        expansion.behavior_expansions[bi] = BehaviorExpansionRecord(
            behavior_id=behavior.behavior_id,
            mechanisms=surviving,
        )


async def propose_blueprint_rows(
    *,
    profile: Mapping[str, Any],
    expansions: list[FacetExpansion],
    generation_total: int,
    retention_total: int,
    retry_feedback: str = "",
    agent: Any | None = None,
) -> BlueprintAgentOutput:
    runnable = agent or create_blueprint_agent()
    available_reference_count = sum(
        len(mechanism.situations)
        for expansion in expansions
        for behavior in expansion.behavior_expansions
        for mechanism in behavior.mechanisms
    )
    if available_reference_count < generation_total:
        raise ValueError(
            "Behavior Expansion 的唯一情境引用不足："
            f"需要 {generation_total} 个，实际 {available_reference_count} 个"
        )
    row_total = retention_total
    if row_total < 1:
        raise ValueError("Behavior Expansion 没有可用于蓝图的情境引用")
    input_data: dict[str, Any] = {
        "facets": [
            {
                "facet_id": facet["facet_id"],
                "facet_name": facet["facet_name"],
                "behavior_evidence": [
                    {
                        "behavior_id": ev["behavior_id"],
                        "behavior_dimension": ev["behavior_dimension"],
                        "high_expression": ev["high_expression"],
                        "low_expression": ev["low_expression"],
                        "boundary_condition": ev["boundary_condition"],
                    }
                    for ev in (facet.get("behavior_evidence") or [])
                ],
            }
            for facet in profile.get("facets") or []
        ],
        "expansions": [row.model_dump(mode="json") for row in expansions],
        "generation_total": generation_total,
        "retention_total": retention_total,
        "row_total": row_total,
        "candidate_count_per_row": INCREMENTAL_CANDIDATES_PER_CELL,
    }
    if retry_feedback:
        input_data["blueprint_semantic_feedback"] = retry_feedback
    raw = await ainvoke_model_with_schema_repair(
        runnable,
        {
            "input_data": input_data,
        },
        job_label="双向细目表设计",
    )
    return BlueprintAgentOutput.model_validate(raw)
