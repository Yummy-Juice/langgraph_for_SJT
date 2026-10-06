"""
Mussel et al. (2018) PSJT 构念证据抽取 — 两阶段 + 行为/线索库

Stage 1: 逐题独立抽取（一题一调，消除批次 anchoring）
Stage 2: 跨题证据族构建（只聚类/合并，不创造新 Evidence）
Stage 3: 行为表现库抽取
Stage 4: 特质激活线索库抽取

用法：
    python mussel_evidence_extraction.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sjt_system.runtime.concurrency import gather_all
from sjt_system.runtime.io import write_json_atomic

# ═══════════════════════════════════════════════════════════════════════════
# Facet 定义
# ═══════════════════════════════════════════════════════════════════════════

FACET_DEFINITIONS = {
    "openness_to_ideas": {
        "facet_id": "openness_to_ideas",
        "facet_name_zh": "观念开放",
        "facet_name_en": "Openness to Ideas",
        "domain": "openness",
        "neo_code": "O5",
        "definition": (
            "出于兴趣本身主动追求智力活动，并开放地考虑新的甚至非常规的观念。"
            "核心是 intellectual curiosity + active pursuit of intellectual "
            "interests + receptivity to new/unconventional ideas，"
            "不等同于高智力、成就追求或政治自由主义。"
        ),
        "high_pole": "主动投入认知资源，愿意探索不熟悉的或抽象的观念",
        "low_pole": "对非必需的智力探索缺乏兴趣，满足于已有理解，回避抽象讨论",
        "common_confounds": [
            "智力或认知能力",
            "成就追求 (C4)",
            "审美敏感性 (O2)",
            "政治自由主义",
            "教育水平",
        ],
    },
    "self_discipline": {
        "facet_id": "self_discipline",
        "facet_name_zh": "自律",
        "facet_name_en": "Self-discipline",
        "domain": "conscientiousness",
        "neo_code": "C5",
        "definition": (
            "面对即时诱惑、疲劳或分心时，仍能坚持完成已承诺或必要的任务。"
            "核心是 persist in task completion despite competing impulses，"
            "不等同于成就追求 (C4)、条理性 (C2) 或审慎性 (C6)。"
        ),
        "high_pole": "即使有更愉快的替代选择，仍坚持完成不愉快的任务",
        "low_pole": "容易被替代诱惑吸引，推迟或放弃计划中的任务",
        "common_confounds": [
            "成就追求 (C4)",
            "条理性 (C2)",
            "审慎性 (C6)",
            "尽责性总体水平",
            "任务重要性判断",
        ],
    },
    "gregariousness": {
        "facet_id": "gregariousness",
        "facet_name_zh": "乐群",
        "facet_name_en": "Gregariousness",
        "domain": "extraversion",
        "neo_code": "E2",
        "definition": (
            "喜欢与他人相处，主动寻求社交接触和群体活动。"
            "核心是 preference for and active seeking of social contact，"
            "不等同于果断性 (E3)、热情性 (E1) 或积极情感 (E6)。"
        ),
        "high_pole": "主动寻求社交机会，喜欢身处人群之中，偏好多人活动",
        "low_pole": "偏好独处或小范围社交，不主动寻求大量社交接触",
        "common_confounds": [
            "果断性 (E3)",
            "热情性 (E1)",
            "积极情感 (E6)",
            "社交焦虑 (N4)",
            "内向-外向总体水平",
        ],
    },
    "compliance": {
        "facet_id": "compliance",
        "facet_name_zh": "顺从",
        "facet_name_en": "Compliance",
        "domain": "agreeableness",
        "neo_code": "A4",
        "definition": (
            "面对人际冲突或不同意见时，倾向于让步、缓和或避免对抗。"
            "核心是 tendency to defer, accommodate, or inhibit aggressive "
            "responses in interpersonal conflict，"
            "不等同于利他性 (A3)、信任 (A1) 或谦逊 (A5)。"
        ),
        "high_pole": "在冲突中倾向于让步、缓和矛盾并维持和谐",
        "low_pole": "更愿意坚持己见，不轻易在冲突中退让",
        "common_confounds": [
            "利他性 (A3)",
            "信任 (A1)",
            "谦逊 (A5)",
            "果断性 (E3)",
            "权力地位差异",
        ],
    },
    "self_consciousness": {
        "facet_id": "self_consciousness",
        "facet_name_zh": "自我意识",
        "facet_name_en": "Self-consciousness",
        "domain": "neuroticism",
        "neo_code": "N4",
        "definition": (
            "在社交场合中容易感到尴尬、害羞或不自在，对他人评价敏感。"
            "核心是 sensitivity to social evaluation and tendency to feel "
            "embarrassed or inferior in social situations，"
            "不等同于社交焦虑障碍、内向 (E) 或低自尊。"
        ),
        "high_pole": "在社交场合中容易感到不自在，担心他人评价，回避引人注目的情境",
        "low_pole": "在社交场合中从容自在，不太在意他人看法，能坦然面对尴尬",
        "common_confounds": [
            "社交焦虑障碍（临床）",
            "内向-外向 (E)",
            "抑郁 (N3)",
            "低自尊",
            "羞怯气质",
        ],
    },
}

FACET_KEY_MAP = {
    "开放性": "openness_to_ideas",
    "责任心": "self_discipline",
    "外倾性": "gregariousness",
    "宜人性": "compliance",
    "神经质": "self_consciousness",
}

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: 逐题独立抽取  (一题一调)
# ═══════════════════════════════════════════════════════════════════════════


class ItemEvidenceUnit(BaseModel):
    """单道 Mussel 题目的结构化证据抽取。一次只分析一道题。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="原题编号，如 openness_to_ideas_05")
    facet: str = Field(description="目标 facet 标识")

    # —— 情境层 ——
    situation_surface: str = Field(
        description="原题表面情境，用一句话概括（保留但不进入生成规则）"
    )
    trait_relevant_cue: str = Field(
        description=(
            "情境中的什么具体条件使目标 facet 的个体差异可以表现出来？"
            "必须是情境特征（situation property），不是行为描述。"
        )
    )

    # —— 证据层 ——
    evidence_proposition: str = Field(
        description=(
            "什么行为差异可以用于推断该 facet 的高 vs 低水平？"
            "表述为一个可检验的命题：在 [cue] 下，是否 [behavioral tendency]。"
        )
    )

    # —— 行为层 ——
    high_behaviors: list[str] = Field(
        min_length=2, max_length=2,
        description="原题中代表高 trait 的 2 个选项对应的具体行为（可观察动作）"
    )
    low_behaviors: list[str] = Field(
        min_length=2, max_length=2,
        description="原题中代表低 trait 的 2 个选项对应的具体行为"
    )

    # —— 机制层 ——
    behavioral_mechanism: str = Field(
        description=(
            "高低行为的本质心理区别。不描述选项，描述心理过程。"
            "例如：'面对非强制性认知挑战时，主动接近 vs 回避'"
        )
    )

    # —— 效度层 ——
    alternative_explanation: str = Field(
        description="可能由目标 facet 之外的什么因素解释？如无，写'无明显替代解释'"
    )
    competing_facets: list[str] = Field(
        default_factory=list,
        description="可能混淆的 NEO facet 代码，如 ['C4', 'E3']"
    )

    # —— 成本层 ——
    situation_cost: str = Field(
        description="做出高 trait 行为是否有时间/精力/机会/人际成本？具体描述"
    )

    # —— 元信息 ——
    high_option_ids: list[str] = Field(
        description="原题中代表高 trait 水平的选项字母，如 ['A', 'B']"
    )
    low_option_ids: list[str] = Field(
        description="原题中代表低 trait 水平的选项字母，如 ['C', 'D']"
    )


class SingleItemExtractionOutput(BaseModel):
    """Stage 1 输出：一道题的证据单元。"""

    model_config = ConfigDict(extra="forbid")
    evidence_unit: ItemEvidenceUnit


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: 跨题证据族构建  (只聚类，不创造)
# ═══════════════════════════════════════════════════════════════════════════


class MergeDecision(BaseModel):
    """两条相似 Evidence 的关系判断。"""

    model_config = ConfigDict(extra="forbid")

    unit_a: str = Field(description="第一条 evidence 的 item_id")
    unit_b: str = Field(description="第二条 evidence 的 item_id")
    decision: Literal["merge", "separate", "hierarchical", "uncertain"] = Field(
        description="合并 / 分开 / 层级关系 / 证据不足以判断"
    )
    reasoning: str = Field(description="判断理由，引用两条 evidence 的具体内容")


class EvidenceFamily(BaseModel):
    """一个证据族：共享同一心理机制的多个 evidence_unit 的聚类结果。"""

    model_config = ConfigDict(extra="forbid")

    family_id: str = Field(description="EF01, EF02, ...")
    family_name: str = Field(
        description="简洁标签，如'知识缺口主动探索'。不以行为或主题命名。"
    )
    definition: str = Field(
        description="这个证据族的抽象定义。独立于任何具体题目主题。"
    )
    high_trait_evidence: str = Field(
        description="高 trait 者的共同行为倾向（抽象表述，非具体动作）"
    )
    low_trait_evidence: str = Field(
        description="低 trait 者的共同行为倾向"
    )

    supporting_item_ids: list[str] = Field(description="属于这个族的原题 ID")
    support_count: int = Field(description="支持题目数")

    member_evidence_units_summary: list[str] = Field(
        description="本族内各 evidence_unit 的 evidence_proposition 摘要"
    )

    typical_high_manifestations: list[str] = Field(
        description="从原题中提取的典型高 trait 具体行为"
    )
    typical_low_manifestations: list[str] = Field(
        description="从原题中提取的典型低 trait 具体行为"
    )

    common_trait_activating_cues: list[str] = Field(
        description="本族中出现的共同情境线索特征"
    )

    competing_facets: list[str] = Field(description="可能的混淆 facet")
    alternative_explanations: list[str] = Field(description="主要替代解释")
    boundary_condition: str = Field(
        default="",
        description="此 evidence 的构念边界——在什么条件下不应据此推断目标 facet"
    )
    construct_purity: Literal["high", "medium", "low"] = Field(
        description="构念纯度评估"
    )
    status: Literal["stable", "provisional"] = Field(
        description="stable: ≥2 题支持; provisional: 仅 1 题支持"
    )


class ItemEvidenceMapping(BaseModel):
    """每道题到证据族的映射。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    primary_family: str = Field(description="主要归属的证据族 ID")
    secondary_family: str | None = Field(
        default=None, description="如有跨族元素，标注第二归属"
    )
    redundancy_cluster: str | None = Field(
        default=None,
        description="与哪些其他 item_id 测量完全相同的 evidence（冗余聚类）",
    )


class CoverageAnalysis(BaseModel):
    """覆盖分析。"""

    model_config = ConfigDict(extra="forbid")

    overrepresented_families: list[str] = Field(
        default_factory=list,
        description="被过多题目重复测量的 evidence 族"
    )
    underrepresented_families: list[str] = Field(
        default_factory=list,
        description="仅 1-2 题测量、可能需要更多覆盖的 evidence 族"
    )
    high_redundancy_item_clusters: list[list[str]] = Field(
        default_factory=list,
        description="实际上测量同一 evidence 的冗余题目组"
    )
    high_contamination_items: list[str] = Field(
        default_factory=list,
        description="构念污染风险较高的题目"
    )
    possible_construct_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "构念定义中可能存在但原题未充分操作化的内容。"
            "只能标记为'可能存在覆盖不足'，不创造新 Evidence。"
        ),
    )


class EvidenceLibraryOutput(BaseModel):
    """Stage 2 输出：完整的证据库。"""

    model_config = ConfigDict(extra="forbid")

    facet: str
    evidence_library: list[EvidenceFamily]
    merge_decisions: list[MergeDecision] = Field(
        description="所有被判断为相似/可能相似的 evidence 对的处理决定"
    )
    item_evidence_mapping: list[ItemEvidenceMapping]
    coverage_analysis: CoverageAnalysis
    summary: dict[str, Any] = Field(
        description="包含 number_of_items, number_of_evidence_families 等"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: 行为表现库
# ═══════════════════════════════════════════════════════════════════════════


class BehavioralManifestation(BaseModel):
    """一个具体行为表现。"""

    model_config = ConfigDict(extra="forbid")

    manifestation_id: str = Field(description="BM01, BM02, ...")
    description: str = Field(description="可观察的行为描述")
    trait_direction: Literal["high", "low"] = Field(description="高/低 trait 方向")
    source_item_ids: list[str] = Field(description="来源题目")
    linked_evidence_family: str = Field(description="关联的证据族 ID")


class BehavioralManifestationBank(BaseModel):
    """Stage 3 输出：行为表现库。"""

    model_config = ConfigDict(extra="forbid")

    facet: str
    manifestations: list[BehavioralManifestation]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: 特质激活线索库
# ═══════════════════════════════════════════════════════════════════════════


class TraitActivatingCue(BaseModel):
    """一个特质激活线索。"""

    model_config = ConfigDict(extra="forbid")

    cue_id: str = Field(description="C01, C02, ...")
    description: str = Field(description="情境线索的描述")
    linked_evidence_families: list[str] = Field(description="关联的证据族 ID")
    source_item_ids: list[str] = Field(description="来源题目")
    cue_category: str = Field(
        description="线索类别，如'存在未知信息'、'观点受到挑战'、'已有投入与诱惑冲突'"
    )


class TraitActivatingCueBank(BaseModel):
    """Stage 4 输出：特质激活线索库。"""

    model_config = ConfigDict(extra="forbid")

    facet: str
    cues: list[TraitActivatingCue]


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

STAGE1_SINGLE_ITEM_PROMPT = """\
You are a construct-evidence extraction specialist. You analyze ONE PSJT item
at a time. You do not see other items from the same facet — this is intentional
to prevent anchoring bias.

Your ONLY task: extract the latent construct-relevant structure from this
single item. You do NOT judge item quality, rewrite, or compare.

INPUT

You receive:
- facet_definition: the Big Five facet this item measures (definition, high/low
  poles, common confounds)
- item: one SJT item with situation text and four options (A/B = higher trait,
  C/D = lower trait)

EXTRACTION RULES

1. situation_surface: one sentence summarizing the surface situation. This is
   a reference record only — not a generation rule.

2. trait_relevant_cue: identify the specific SITUATIONAL CONDITION that allows
   the target facet's individual differences to manifest. Must describe a
   property of the situation, not a behavior.
   Good: "存在一个非必须解决但有认知挑战性的可选任务"
   Bad: "被试选择是否解题"

3. evidence_proposition: formulate as a testable proposition:
   "在 [cue] 的情况下，是否 [behavioral tendency]"

4. high_behaviors: the two observable behaviors from options A and B.
   Describe actions, not option text. Use verbs.

5. low_behaviors: same for options C and D.

6. behavioral_mechanism: the essential psychological distinction between high
   and low. Not "high vs low openness" but the actual process:
   "面对非强制性认知挑战时，主动接近 vs 回避"

7. alternative_explanation: could something other than the target facet
   explain the response pattern? Consider ability, domain interest, time
   pressure, social desirability, another facet. Be specific.

8. competing_facets: list NEO facet codes that might confound this item
   (e.g., ["C4", "E3"]). Use empty list if none.

9. situation_cost: does choosing the high-trait behavior involve a real cost?

10. high_option_ids / low_option_ids: which option letters are high/low trait.

CRITICAL: Do NOT infer what the item "should have" measured. Only extract what
is actually present in the options. Do NOT add evidence dimensions not directly
supported by the option content.

Return only the structured JSON object.
"""

STAGE2_FAMILY_BUILDER_PROMPT = """\
You are a personality measurement and construct modeling expert.

You receive structured evidence extractions from multiple PSJT items measuring
the SAME Big Five facet. Each evidence_unit was extracted independently (one
item per call) to prevent anchoring.

Your task: cluster, de-duplicate, and abstract these evidence_units into a
stable facet-level Evidence Library.

CRITICAL RULE: You MUST NOT create any new evidence that is not present in
the input evidence_units. You may only cluster, merge, name, and describe
boundaries between existing extractions.

INPUT

- facet_name, facet_definition, high_pole, low_pole
- A list of evidence_units (one per item)

═══ THREE-LAYER DISTINCTION ═══

Before clustering, you must distinguish three layers. Confusing them is the
most common failure mode.

LAYER 1: EVIDENCE (psychological mechanism) — WHY the behavior diagnoses the
facet. This is what you cluster. Examples:
  - "actively resolving unnecessary knowledge gaps"
  - "deep processing of complex information beyond surface comprehension"
  - "receptivity to viewpoints that challenge existing beliefs"
  - "anticipatory avoidance of social-evaluative exposure"
  - "self-referential interpretation of ambiguous social cues"

LAYER 2: TRAIT-ACTIVATING CUE (situation condition) — WHAT feature of the
situation allows the trait to manifest. These go in common_trait_activating_cues.
  - "task is non-mandatory with no external reward"
  - "existing belief is challenged by new evidence"
  - "attention is already drawn to the self in a public setting"

LAYER 3: SURFACE DOMAIN — concrete setting: software, TV, museum, game night,
fatigue state, travel, restaurant. These go in situation_variants only.

═══ THREE HARD RULES ═══

┌─────────────────────────────────────────────────────────────────┐
│ RULE 1: HIGH AND LOW ARE TWO ENDS OF ONE EVIDENCE —             │
│         NEVER SPLIT THEM INTO TWO FAMILIES                      │
└─────────────────────────────────────────────────────────────────┘

An Evidence Family describes ONE psychological continuum along which
individuals differ. High-trait and low-trait behaviors are the two POLES
of this continuum. They must stay in the SAME family.

FAILURE EXAMPLE (from self-consciousness):
  EF01 "社交失误后回避" — high = avoid/escape after embarrassing event
  EF08 "社交失误后主动化解" — low = stay and resolve after embarrassing event
  This is WRONG. EF08 is simply the LOW pole of EF01's mechanism.
  They describe the SAME evidence: "after an embarrassing event, do you
  avoid (high) or engage (low)?"

CORRECT APPROACH:
  One family: "尴尬事件后的社会评价回避"
    high_trait_evidence: 回避、逃离、隐藏、减少曝光
    low_trait_evidence: 留在情境中、坦然面对、主动化解

Before creating a new family, ask: "Is this just describing what the LOW
end of an existing family looks like?" If yes, merge.

┌─────────────────────────────────────────────────────────────────┐
│ RULE 2: EVENT TYPE IS NOT EVIDENCE —                            │
│         DO NOT SPLIT FAMILIES BY WHAT CAUSED THE SITUATION       │
└─────────────────────────────────────────────────────────────────┘

Items that activate the SAME psychological mechanism but through different
triggering events must be in the SAME family.

FAILURE EXAMPLE (from self-consciousness):
  EF01: 自己犯错/滑倒/认错人/打翻酒 → 回避
  EF06: 手机响/买私密物品遇上司/被醉汉指责/遇到亲密情侣 → 回避
  These are the SAME mechanism: "an event that may cause public
  embarrassment triggers avoidance to reduce exposure."
  "Who caused it" and "what type of event" are trait-activating cue
  variants, not different evidence.

CORRECT APPROACH: merge EF01 + EF06 into one family, document the
different event types as cue variants.

Similarly for gregariousness:
  "在葬礼后聚会中融入群体" vs "在火车上与陌生人交谈"
  Both diagnose: "在非强制社交机会中主动扩大接触范围"
  "葬礼" and "火车" are surface domains, not different evidence.

┌─────────────────────────────────────────────────────────────────┐
│ RULE 3: "APPROACH vs AVOID" IS A BEHAVIORAL PATTERN —           │
│         NOT A SUFFICIENT REASON TO MERGE                         │
└─────────────────────────────────────────────────────────────────┘

Many items share the surface pattern "high = approach X, low = avoid X".
This is NOT sufficient for merging. You must verify:
  (a) The PSYCHOLOGICAL OBJECT being approached/avoided is the same
  (b) The DIAGNOSTIC MECHANISM (why behavior diagnoses the facet) is the same

FAILURE EXAMPLE (from openness_to_ideas):
  Item 5: looking up an unfamiliar word → "filling an unnecessary knowledge gap"
  Item 4: understanding a complex scientific theory → "deep processing of complex content"
  Item 6: watching a debate with non-traditional arguments → "considering viewpoints that challenge existing beliefs"
  Item 18: choosing a high-difficulty new game → "preferring higher cognitive demand"
  All four are "approach intellectual stimulation" superficially, but they
  measure FOUR DIFFERENT psychological mechanisms. They must NOT merge.

For self-consciousness:
  "预期自己将成为注意焦点 → 提前回避" (anticipatory avoidance)
  vs
  "已经发生尴尬事件 → 逃离现场" (reactive avoidance)
  These are DIFFERENT mechanisms: one is anticipatory (before exposure),
  the other is reactive (after exposure). They can be separate families IF
  the diagnostic logic differs.

CORRECT TEST before merging:
  "If I changed the surface topic completely, would these two items still
  diagnose the facet through the SAME psychological pathway?"

═══ MERGE DECISION TAXONOMY ═══

For every pair of evidence_units that appear similar, document:
- merge: same mechanism, different surface domains or cues
- separate: different mechanisms — explicitly name WHAT differs
- hierarchical: one is broader, the other a subtype
- uncertain: insufficient evidence

═══ ITEM-TO-FAMILY MAPPING ═══

One item CAN and often SHOULD belong to multiple evidence families.

Example from openness: an item about "a complex new scientific theory
in your area of expertise" may involve BOTH:
  - Primary: deep processing / knowledge integration (understanding every detail)
  - Secondary: non-instrumental intellectual engagement (the theory has no practical use)

Example from self-consciousness: an item about "being mimicked by a
superior at a party" may involve BOTH:
  - Primary: becoming an unintended focus of social attention
  - Secondary: sensitivity to negative evaluation by authority figures

You MUST use both primary_family and secondary_family.
Forcing every item into exactly one family discards valuable diagnostic
information. The secondary_family can be null only when the item truly
measures a single, pure mechanism.

═══ EVIDENCE FAMILY FORMATION ═══

Each family must:
- Be named after the psychological mechanism (not surface domain, not event type, not behavioral direction alone).
  Good: "评价暴露的预期性回避", "社会注意焦点敏感性", "模糊社会线索的自我指向解释"
  Bad: "学习情境", "疲劳时", "社交失误后", "自己犯错", "别人失误"
- Be describable independently of any specific item theme.
- Be directly consistent with the facet definition.
- Include a boundary_condition that prevents construct drift.

>=2 items -> status "stable"
1 item -> status "provisional"

If a provisional family differs from a stable family only by a situational
condition (Layer 2) or event type (Layer 3), MERGE it. Document the
condition as a trait-activating cue, not as a new family.

═══ COMPETING FACETS AND CONSTRUCT PURITY ═══

For each family, identify NEO facet codes that could produce the same
behavioral pattern.

Mark purity "low" when:
- The mechanism is more strongly explained by another facet
- >2 competing facets provide plausible alternatives
- The boundary with a non-facet factor (ability, social norm, role) is unclear

═══ COVERAGE ANALYSIS ═══

- Which families are over/under-represented?
- Which items are redundant (identical evidence)?
- Which items have high contamination risk?
- Are there aspects of the facet definition under-operationalized?
  Flag as "可能存在覆盖不足" only — do NOT create new evidence.
  Must cite specific definition content absent from the items.

═══ OUTPUT ═══

Return only the structured JSON object.
"""

STAGE3_BEHAVIOR_BANK_PROMPT = """\
You are building a Behavioral Manifestation Bank from PSJT items for one
Big Five facet.

INPUT

- evidence_library: the evidence families already established
- item_evidence_units: all per-item extractions with their high_behaviors
  and low_behaviors

TASK

For each evidence family, extract every distinct, observable behavioral
manifestation from the original items. A manifestation is:

- A concrete, observable action (verb + object + context)
- Tied to one trait direction (high or low)
- Traceable to specific source item_ids

Group similar manifestations and de-duplicate:
- "查询词典" and "上网搜索资料" -> both are "主动查找信息", but keep both
  as separate manifestations if they involve different action types
- "跳过不读" and "继续翻杂志" -> different surface actions, but if both
  express "忽略认知需求", keep both as manifestations of the same tendency

Do NOT invent new manifestations not present in the source items.

OUTPUT

Return only the structured JSON object.
"""

STAGE4_CUE_BANK_PROMPT = """\
You are building a Trait-Activating Cue Bank from PSJT items for one Big Five
facet.

INPUT

- evidence_library: the evidence families
- item_evidence_units: all per-item extractions with their trait_relevant_cue

TASK

For each evidence family, extract and de-duplicate the situational conditions
(trait_relevant_cues) that activate the target facet.

A cue describes a feature of the SITUATION, not a behavior:
- Good: "存在非强制但可选的认知挑战任务"
- Good: "已有观点受到有依据的新信息挑战"
- Bad: "被试选择是否学习"

Group cues into categories such as:
- 存在未知信息
- 观点受到挑战
- 已有投入与诱惑冲突
- 存在社交接触机会
- 出现人际摩擦

Each cue must be linked to the evidence families it activates and the source
items where it appears.

OUTPUT

Return only the structured JSON object.
"""

# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════


def load_mussel_items(facet_key: str, path: str = "docs/mussel_zh.json") -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    facet_data = data.get(facet_key)
    if facet_data is None:
        raise KeyError(f"找不到 facet '{facet_key}'，可用: {list(data.keys())}")
    facet_id = FACET_KEY_MAP[facet_key]
    items = []
    for key, value in facet_data.items():
        items.append({
            "item_id": f"{facet_id}_{int(key):02d}",
            "facet": facet_id,
            "situation": value["situation"],
            "options": value["options"],
        })
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════════════


async def run_stage1_single_item(
    facet_key: str,
    item: dict[str, Any],
    agent: Any,
) -> ItemEvidenceUnit:
    """Stage 1: 对一道题进行独立证据抽取。"""
    from sjt_system.agent.retry import ainvoke_model_with_schema_repair

    facet_id = FACET_KEY_MAP[facet_key]
    input_data = {
        "facet_definition": FACET_DEFINITIONS[facet_id],
        "item": item,
    }
    result = await ainvoke_model_with_schema_repair(
        agent,
        {"input_data": input_data},
        job_label=f"stage1/{facet_id}/{item['item_id']}",
    )
    parsed = (
        result if isinstance(result, SingleItemExtractionOutput)
        else SingleItemExtractionOutput.model_validate(result)
    )
    return parsed.evidence_unit


async def run_stage2_family_builder(
    facet_key: str,
    evidence_units: list[ItemEvidenceUnit],
    agent: Any,
) -> EvidenceLibraryOutput:
    """Stage 2: 跨题证据族构建。"""
    from sjt_system.agent.retry import ainvoke_model_with_schema_repair

    facet_id = FACET_KEY_MAP[facet_key]
    defn = FACET_DEFINITIONS[facet_id]
    input_data = {
        "facet_name": defn["facet_name_zh"],
        "facet_definition": defn["definition"],
        "high_pole": defn["high_pole"],
        "low_pole": defn["low_pole"],
        "evidence_units": [unit.model_dump(mode="json") for unit in evidence_units],
    }
    result = await ainvoke_model_with_schema_repair(
        agent,
        {"input_data": input_data},
        job_label=f"stage2/{facet_id}",
    )
    return (
        result if isinstance(result, EvidenceLibraryOutput)
        else EvidenceLibraryOutput.model_validate(result)
    )


async def run_stage3_behavior_bank(
    facet_key: str,
    evidence_library: EvidenceLibraryOutput,
    evidence_units: list[ItemEvidenceUnit],
    agent: Any,
) -> BehavioralManifestationBank:
    """Stage 3: 行为表现库。"""
    from sjt_system.agent.retry import ainvoke_model_with_schema_repair

    facet_id = FACET_KEY_MAP[facet_key]
    input_data = {
        "facet": facet_id,
        "evidence_library": [
            fam.model_dump(mode="json") for fam in evidence_library.evidence_library
        ],
        "item_evidence_units": [
            unit.model_dump(mode="json") for unit in evidence_units
        ],
    }
    result = await ainvoke_model_with_schema_repair(
        agent,
        {"input_data": input_data},
        job_label=f"stage3/{facet_id}",
    )
    return (
        result if isinstance(result, BehavioralManifestationBank)
        else BehavioralManifestationBank.model_validate(result)
    )


async def run_stage4_cue_bank(
    facet_key: str,
    evidence_library: EvidenceLibraryOutput,
    evidence_units: list[ItemEvidenceUnit],
    agent: Any,
) -> TraitActivatingCueBank:
    """Stage 4: 特质激活线索库。"""
    from sjt_system.agent.retry import ainvoke_model_with_schema_repair

    facet_id = FACET_KEY_MAP[facet_key]
    input_data = {
        "facet": facet_id,
        "evidence_library": [
            fam.model_dump(mode="json") for fam in evidence_library.evidence_library
        ],
        "item_evidence_units": [
            unit.model_dump(mode="json") for unit in evidence_units
        ],
    }
    result = await ainvoke_model_with_schema_repair(
        agent,
        {"input_data": input_data},
        job_label=f"stage4/{facet_id}",
    )
    return (
        result if isinstance(result, TraitActivatingCueBank)
        else TraitActivatingCueBank.model_validate(result)
    )


def _source_items_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items_by_id = {str(item["item_id"]): item for item in items}
    if len(items_by_id) != len(items):
        raise ValueError("Mussel source items contain duplicate item_id values")
    return items_by_id


def _validate_source_item_unit(
    value: Any,
    source_item: dict[str, Any],
) -> ItemEvidenceUnit:
    unit = (
        value
        if isinstance(value, ItemEvidenceUnit)
        else ItemEvidenceUnit.model_validate(value)
    )
    if unit.item_id != str(source_item["item_id"]):
        raise ValueError(
            f"Stage 1 cache item_id {unit.item_id!r} does not match "
            f"source item {source_item['item_id']!r}"
        )
    if unit.facet != str(source_item["facet"]):
        raise ValueError(
            f"Stage 1 cache facet {unit.facet!r} does not match "
            f"source facet {source_item['facet']!r}"
        )
    return unit


def _load_complete_stage1_cache(
    path: Path,
    items: list[dict[str, Any]],
    items_by_id: dict[str, dict[str, Any]],
) -> list[ItemEvidenceUnit]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Stage 1 aggregate cache must be a list")
    units_by_id: dict[str, ItemEvidenceUnit] = {}
    for row in raw:
        unit = ItemEvidenceUnit.model_validate(row)
        source_item = items_by_id.get(unit.item_id)
        if source_item is None:
            raise ValueError(f"Stage 1 aggregate contains unknown item_id {unit.item_id!r}")
        _validate_source_item_unit(unit, source_item)
        if unit.item_id in units_by_id:
            raise ValueError(f"Stage 1 aggregate repeats item_id {unit.item_id!r}")
        units_by_id[unit.item_id] = unit
    expected = set(items_by_id)
    if set(units_by_id) != expected:
        missing = sorted(expected - set(units_by_id))
        raise ValueError(f"Stage 1 aggregate is incomplete; missing item_ids: {missing}")
    return [units_by_id[str(item["item_id"])] for item in items]


async def process_facet(
    facet_key: str,
    output_root: Path,
    stage1_agent: Any,
    stage2_agent: Any,
    stage3_agent: Any,
    stage4_agent: Any,
) -> None:
    facet_id = FACET_KEY_MAP[facet_key]
    print(f"\n{'='*60}")
    print(f"处理: {facet_key} ({facet_id})")
    print(f"{'='*60}")

    items = load_mussel_items(facet_key)
    items_by_id = _source_items_by_id(items)
    output_dir = output_root / facet_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: 逐题独立抽取
    evidence_units: list[ItemEvidenceUnit] = []
    stage1_path = output_dir / "stage1_per_item_evidence.json"
    if stage1_path.exists():
        print("  Stage 1: 从缓存加载...")
        evidence_units = _load_complete_stage1_cache(stage1_path, items, items_by_id)
    else:
        item_cache_dir = output_dir / "stage1_items"
        item_cache_dir.mkdir(parents=True, exist_ok=True)
        units_by_id: dict[str, ItemEvidenceUnit] = {}
        missing_items = []
        for index, item in enumerate(items):
            item_id = str(item["item_id"])
            item_cache_path = item_cache_dir / f"{item_id}.json"
            if item_cache_path.exists():
                cached = json.loads(item_cache_path.read_text(encoding="utf-8"))
                units_by_id[item_id] = _validate_source_item_unit(cached, item)
            else:
                missing_items.append((index, item))

        if missing_items:
            if units_by_id:
                print(
                    f"  Stage 1: 恢复逐题缓存 ({len(units_by_id)}/{len(items)} 已缓存，"
                    f"{len(missing_items)} 题待抽取)..."
                )
            else:
                print(f"  Stage 1: 逐题抽取 ({len(items)} 题，每题独立调用)...")

            async def extract_and_cache(item: dict[str, Any]) -> ItemEvidenceUnit:
                unit = _validate_source_item_unit(
                    await run_stage1_single_item(facet_key, item, stage1_agent),
                    item,
                )
                item_cache_path = item_cache_dir / f"{item['item_id']}.json"
                write_json_atomic(item_cache_path, unit.model_dump(mode="json"))
                return unit

            new_units = await gather_all(
                *(extract_and_cache(item) for _, item in missing_items)
            )
            for (index, item), unit in zip(missing_items, new_units):
                units_by_id[str(item["item_id"])] = unit
                print(f"    [{index + 1}/{len(items)}] {item['item_id']} OK")
        else:
            print("  Stage 1: 从逐题缓存加载...")

        evidence_units = [units_by_id[str(item["item_id"])] for item in items]
        stage1_path.write_text(
            json.dumps(
                [u.model_dump(mode="json") for u in evidence_units],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  Stage 1 完成 -> {stage1_path}")

    # Stage 2: 证据族构建
    stage2_path = output_dir / "stage2_evidence_library.json"
    if stage2_path.exists():
        print("  Stage 2: 从缓存加载...")
        raw = json.loads(stage2_path.read_text(encoding="utf-8"))
        library = EvidenceLibraryOutput.model_validate(raw)
    else:
        print("  Stage 2: 构建证据族...")
        library = await run_stage2_family_builder(facet_key, evidence_units, stage2_agent)
        stage2_path.write_text(
            json.dumps(library.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Stage 2 完成: {len(library.evidence_library)} 族 -> {stage2_path}")

    # Stage 3 and Stage 4 only depend on the completed Stage 2 library.
    stage3_path = output_dir / "stage3_behavior_bank.json"
    if stage3_path.exists():
        print("  Stage 3: 从缓存加载...")
    else:
        print("  Stage 3: 构建行为表现库...")

        async def build_stage3() -> None:
            behavior_bank = await run_stage3_behavior_bank(
                facet_key, library, evidence_units, stage3_agent
            )
            stage3_path.write_text(
                json.dumps(behavior_bank.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Stage 3 完成: {len(behavior_bank.manifestations)} 条 -> {stage3_path}")

    # Stage 4: 特质激活线索库
    stage4_path = output_dir / "stage4_cue_bank.json"
    if stage4_path.exists():
        print("  Stage 4: 从缓存加载...")
    else:
        print("  Stage 4: 构建线索库...")

        async def build_stage4() -> None:
            cue_bank = await run_stage4_cue_bank(
                facet_key, library, evidence_units, stage4_agent
            )
            stage4_path.write_text(
                json.dumps(cue_bank.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Stage 4 完成: {len(cue_bank.cues)} 条 -> {stage4_path}")

    pending_libraries = []
    if not stage3_path.exists():
        pending_libraries.append(build_stage3())
    if not stage4_path.exists():
        pending_libraries.append(build_stage4())
    await gather_all(*pending_libraries)


async def main():
    from sjt_system.agent.agent_factory import create_agent

    output_root = Path("knowledge_base/evidence_library")

    stage1_agent = create_agent(STAGE1_SINGLE_ITEM_PROMPT, SingleItemExtractionOutput, temperature=0.1)
    stage2_agent = create_agent(STAGE2_FAMILY_BUILDER_PROMPT, EvidenceLibraryOutput, temperature=0.1)
    stage3_agent = create_agent(STAGE3_BEHAVIOR_BANK_PROMPT, BehavioralManifestationBank, temperature=0.1)
    stage4_agent = create_agent(STAGE4_CUE_BANK_PROMPT, TraitActivatingCueBank, temperature=0.1)

    facet_keys = ["开放性", "责任心", "外倾性", "宜人性", "神经质"]
    await gather_all(*(
        process_facet(
            facet_key, output_root,
            stage1_agent, stage2_agent, stage3_agent, stage4_agent,
        )
        for facet_key in facet_keys
    ))

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
