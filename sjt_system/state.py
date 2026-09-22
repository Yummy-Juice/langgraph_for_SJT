from typing import Any, Literal
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict
# ============================================================
# 1. Router 可以选择的任务
RouteType = Literal[
    "clarify_requirements",          # 分析并补全测验需求
    "build_blueprint",               # 一次性建立程序持有的固定构念—题目细目表
    "generate_item",                 # 根据当前蓝图单元生成题
    "review_item",                   # 多专家审查当前题目
    "revise_item",                   # 定向修改当前题目
    "regenerate_item",               # 废弃当前题目并重新生成
    "simulate_responses",            # 生成虚拟被试作答
    "analyze_psychometrics",         # 计算项目和测验统计指标
    "select_items",                  # 筛选题目并检查筛选后的蓝图覆盖度
    "confirm_psychometric_repair",   # 用户确认当前单题诊断后再执行返修
    "plateau_gap_decision",          # 平台期收卷存在蓝图缺口时用户处置缺口单元
    "psychometric_repair_batch",     # 并发执行整批单题修改—复测闭环
    "assemble_test",                 # 根据筛选结果组卷
    "review_test",                   # 对组卷后的完整测验进行综合审核
    "rescore_test",                  # 调整计分并重新分析
    "generate_reports",              # 生成测验材料和报告
    "finish",                        # 结束流程
]

# ============================================================
# 2. Router
class PSJTRouteDecision(TypedDict):
    # Execute 下一步需要执行的任务
    next_action: RouteType
    # Router 选择该任务的依据，主要用于日志和调试
    reason: str
    # 当前任务操作的题目 ID；
    target_item_id: str | None
    # 当前任务操作的蓝图单元 ID；
    target_blueprint_cell_id: str | None


class TraceEvent(TypedDict, total=False):
    """单个工作流节点产生的结构化调试事件。"""

    event_id: str
    run_id: str
    step: int
    node: str
    action: str
    event_type: Literal["started", "completed", "failed", "waiting"]
    reason: str
    recorded_at: str
    duration_ms: int
    state_changes: dict[str, Any]
    error: str
    approval_source: Literal["user", "system"]


class UserDecision(TypedDict, total=False):
    """用户对单次 Agent 候选结果的决策。"""

    decision: Literal[
        "approve",
        "edit",
        "regenerate",
        "answer",
        "accept_suggestions",
        "confirm",
        "revise",
        "stop",
    ]
    feedback: str | None
    state_patch: dict[str, Any] | None
    approval_source: Literal["user", "system"]


class VirtualRespondentRef(TypedDict):
    """工作流 State 中保存的确定性分数型虚拟被试。"""

    respondent_id: str
    condition_id: str
    matched_subject_id: str
    active_dimension_id: str
    score_values: dict[str, float]


class VirtualSampleConfig(TypedDict):
    """一次虚拟样本选择的可复现配置。"""

    pool_id: str
    schema_version: int
    pool_ref: str | None
    source_file: str | None
    source_sha256: str
    available_count: int
    sample_size: int
    sample_size_per_condition: int
    condition_count: int
    recommended_sample_size: int
    automatic_selection_minimum_sample_size: int
    seed: int
    max_concurrency: int
    max_retries: int
    persona_modes: list[Literal["score_profile"]]
    selection_strategy: Literal["deterministic_matched_normal_generation"]
    sampling_design: Literal["matched_facet_conditions"]
    persona_method: str
    conditions: list[dict[str, Any]]
    score_distribution: dict[str, Any]
    score_scale: list[float]
    response_count_per_respondent_item: Literal[1]
    generator_version: str
    prompt_version: str
    generation_diagnostics: dict[str, Any]
    ipip_neo_reference_enabled: bool
    ipip_neo_in_main_iteration: bool
    neo_ffi_in_main_iteration: NotRequired[bool]

# ============================================================
# 3. 需求 State
class ConstructSelection(TypedDict):
    inventory_id: str
    domain_id: str
    # Empty means the complete domain; otherwise these are explicit facets.
    facet_ids: list[str]


class TestSpecification(TypedDict):
    """需求分析阶段形成的测验规格。"""
    construct_selection: ConstructSelection
    target_population: str
    final_item_count: int
    output_language: str


class RequirementSuggestion(TypedDict):
    field: str
    reason: str


class RequirementQuestion(TypedDict):
    field: str
    issue_type: Literal["missing", "ambiguous", "confirm_inference"]
    text: str


class RequirementInteraction(TypedDict):
    suggestions: list[RequirementSuggestion]
    questions: list[RequirementQuestion]

class RequirementStateUpdate(TypedDict):
    test_specification: TestSpecification
    specification_sources: dict[
        str,
        Literal[
            "user",
            "inferred",
            "system_default",
        ],
    ]

class RequirementResult(TypedDict):
    state_update: RequirementStateUpdate
    suggestions: list[RequirementSuggestion]
    questions: list[RequirementQuestion]


class BehavioralAnchors(TypedDict):
    low: str
    medium_low: str
    medium_high: str
    high: str


# ============================================================
# 5. 蓝图、题目与审题结构化输出
class ItemSpecification(TypedDict):
    specification_id: str
    blueprint_cell_id: str
    target_dimension_id: str
    context_category: str
    context_seed: str
    situation_type: str
    stakes_level: Literal["low", "medium", "high"]
    social_context: str
    behavior_evidence_id: str
    mechanism_id: str
    situation_id: str
    activation_mechanism: str
    core_tension: str
    behavioral_anchors: BehavioralAnchors
    behavioral_functions: BehavioralAnchors
    contamination_exclusions: list[str]
    scenario_constraints: list[str]
    option_constraints: list[str]
    avoid_scenario_patterns: list[str]
    avoid_response_patterns: list[str]


class ItemPatternProfile(TypedDict):
    item_id: str
    context_category: str
    context_signature: str
    scenario: str
    highest_score_strategy: str
    lowest_score_strategy: str
    option_lengths: list[int]
    score_position_pattern: list[int]


class ConstructProfileReference(TypedDict):
    inventory_id: str
    inventory_name: str
    inventory_version: str
    review_status: str
    selection_level: Literal["domain", "facet"]
    domain_id: str
    domain_name: str
    selected_facet_ids: list[str]
    profile_hash: str
    resolution_source: str


class GenerationSlot(TypedDict):
    specification_id: str
    blueprint_cell_id: str
    candidate_reference: dict[str, str]


class GenerationCell(TypedDict):
    cell_id: str
    facet_id: str
    behavior_id: str
    mechanism_id: str
    situation_id: str
    candidate_references: list[dict[str, str]]
    planned_generation_count: int
    planned_retention_count: int


class CompactOptionFunction(TypedDict):
    behavioral_level: Literal[
        "low",
        "medium_low",
        "medium_high",
        "high",
    ]
    behavioral_tendency: str
    psychological_function: str


class CompactItemSkeleton(TypedDict):
    situation_type: str
    stakes_level: Literal["low", "medium", "high"]
    social_context: str
    behavioral_tension: str
    option_structure: list[CompactOptionFunction]


class GenerationBlueprint(TypedDict):
    blueprint_id: str
    version: int
    construct_profile_ref: ConstructProfileReference
    construct_profile_snapshot: dict[str, Any]
    expansion_refs: list[dict[str, str]]
    cells: list[GenerationCell]
    slots: list[GenerationSlot]


class CompactSkeletonStateUpdate(TypedDict):
    """One skeleton payload; the workflow owns and attaches its slot ID."""

    item_skeleton: CompactItemSkeleton


class CompactSkeletonResult(TypedDict):
    state_update: CompactSkeletonStateUpdate
    summary: str


# Compatibility aliases for older checkpoints/imports. New model calls use the
# singular contract above and never ask the model to copy specification IDs.
CompactSkeletonBatchStateUpdate = CompactSkeletonStateUpdate
CompactSkeletonBatchResult = CompactSkeletonResult


class SkeletonReviewFinding(TypedDict):
    severity: Literal["warning", "blocking"]
    issue: str
    evidence: str
    instruction: str | None


class SkeletonReview(TypedDict):
    findings: list[SkeletonReviewFinding]
    evidence_limitations: list[str]
    summary: str


class SJTResponseOption(TypedDict):
    option_id: str
    text: str
    behavioral_level: Literal[
        "low",
        "medium_low",
        "medium_high",
        "high",
    ]


class SJTItem(TypedDict):
    item_id: str
    blueprint_cell_id: str
    target_dimension_id: str
    context_category: str
    context_signature: str
    scenario: str
    response_instruction: str
    response_options: list[SJTResponseOption]
    scoring_key: dict[str, int | float]
    construct_rationale: str
    contamination_risks: list[str]
    version: int


class ItemStateUpdate(TypedDict):
    current_item: SJTItem


class ItemResult(TypedDict):
    state_update: ItemStateUpdate
    summary: str


class ItemOptionTextPatch(TypedDict):
    option_id: str
    text: str


class ItemRevisionStateUpdate(TypedDict):
    option_updates: list[ItemOptionTextPatch]


class ItemRevisionResult(TypedDict):
    state_update: ItemRevisionStateUpdate
    summary: str


class ItemRepairStateUpdate(TypedDict):
    scenario_update: str | None
    option_updates: list[ItemOptionTextPatch]


class ItemRepairResult(TypedDict):
    state_update: ItemRepairStateUpdate
    summary: str


class ItemBehavioralStrategy(TypedDict):
    behavioral_level: Literal["low", "medium_low", "medium_high", "high"]
    text: str


class ItemRealizationStateUpdate(TypedDict):
    scenario: str
    strategies: list[ItemBehavioralStrategy]


class ItemRealizationResult(TypedDict):
    state_update: ItemRealizationStateUpdate
    summary: str


class ItemRegenerationStateUpdate(TypedDict):
    context_signature: str
    scenario: str
    strategies: list[ItemBehavioralStrategy]
    construct_rationale: str
    contamination_risks: list[str]


class ItemRegenerationResult(TypedDict):
    state_update: ItemRegenerationStateUpdate
    summary: str


class PsychometricRepairHypothesis(TypedDict):
    failure_mode: Literal[
        "acceptable_development_quality",
        "borderline_but_content_sound",
        "low_frequency_options",
        "score_concentration",
        "semantic_level_mismatch",
        "weak_trait_activation",
        "construct_contamination",
        "transparent_scoring",
        "non_actionable_statistical_signal",
        "skeleton_incompatibility",
        "simulation_inconsistency",
        "insufficient_evidence",
    ]
    locus: Literal[
        "scenario",
        "response_options",
        "behavioral_level",
        "skeleton",
        "construct",
        "uncertain",
    ]
    affected_option_ids: list[str]
    observed_pattern: str
    textual_evidence: str
    alternative_explanations: list[str]
    minimal_edit_operator: str
    repair_instruction: str
    predicted_change: str
    confidence: Literal["low", "medium", "high"]


class PsychometricRepairDiagnosis(TypedDict):
    decision: Literal[
        "retain",
        "revise_item",
        "revise_options",
        "regenerate_realization",
        "replace_candidate",
        "defer",
    ]
    primary_hypothesis: PsychometricRepairHypothesis
    acceptance_criteria: list[str]
    summary: str


class AtomicDiagnosisDiscrepancy(TypedDict):
    observation_refs: list[str]
    constraint_refs: list[str]
    description: str


class AtomicDiagnosisCandidate(TypedDict):
    diagnosis_id: str
    suspect_components: list[Literal[
        "scenario",
        "response_options",
        "skeleton",
        "activation_mechanism",
        "behavior_evidence",
        "construct",
        "simulation",
        "simulation_or_insufficient_evidence",
        "insufficient_evidence",
    ]]
    affected_option_ids: list[str]
    observation_refs: list[str]
    constraint_refs: list[str]
    textual_evidence: str
    explanation: str
    confidence: Literal["low", "medium", "high"]


class AtomicEdit(TypedDict):
    target_field: Literal["scenario", "response_options"]
    option_ids: list[str]
    problem: str
    instruction: str


class AtomicRepairTask(TypedDict):
    diagnosis_id: str
    atomic_edit: AtomicEdit
    # The workflow reserves the first executable task for target-facet
    # gradient optimization. Every current diagnosis task declares its phase.
    phase: Literal["target_facet_gradient", "other"]


class AtomicRepairAdvice(TypedDict):
    decision: Literal["repair", "defer"]
    observed_discrepancies: list[AtomicDiagnosisDiscrepancy]
    candidate_diagnoses: list[AtomicDiagnosisCandidate]
    repair_tasks: list[AtomicRepairTask]
    summary: str


class MechanismValidationResult(TypedDict):
    ranking: list[str]
    target_is_first: bool
    reason: str


class ItemRequiredEdit(TypedDict):
    field: Literal["scenario", "response_options"]
    option_ids: list[str]
    instruction: str


class ItemReviewFinding(TypedDict):
    criterion: Literal[
        "trait_activation",
        "ecological_plausibility",
        "option_anti_faking",
        "construct_purity",
    ]
    severity: Literal["warning", "blocking"]
    locus: Literal[
        "scenario",
        "response_options",
        "behavioral_level",
        "scoring_key",
        "skeleton",
    ]
    affected_option_ids: list[str]
    evidence: str
    problem: str
    repair_instruction: str
    # 审题提示词要求每个 finding 都返回此字段；没有具体修改任务时使用空列表。
    required_edits: list[ItemRequiredEdit]


class ItemReviewDiagnosis(TypedDict):
    findings: list[ItemReviewFinding]
    summary: str


class RepairTarget(TypedDict):
    field: Literal["scenario", "response_options", "scoring_key"]
    option_ids: list[str]


class RepairTask(TypedDict):
    task_id: str
    source: Literal["construct", "content", "deterministic"]
    targets: list[RepairTarget]
    problem: str
    instruction: str


class UnifiedItemReview(TypedDict):
    findings: list[ItemReviewFinding]
    repair_tasks: list[RepairTask]
    summary: str


# ============================================================
# 6. 共享 State
class PSJTState(TypedDict):
    # --------------------------------------------------------
    # A. 原始输入
    # 用户最初输入的自然语言测验开发需求
    user_request: str
    # 本次自动开发任务的唯一标识
    run_id: str
    # --------------------------------------------------------
    # B. Router 决策结果
    # Router 最近一次输出的结构化路由结果
    route: PSJTRouteDecision | None
    # 系统当前所处的业务阶段，便于观察和恢复流程
    current_phase: Literal[
        "requirements",
        "construct_blueprint",
        "item_development",
        "virtual_simulation",
        "psychometric_analysis",
        "item_selection",
        "test_assembly",
        "reporting",
        "completed",
    ]
    # 蓝图确认后选择的题目开发模式；旧检查点缺失时按未选择处理。
    item_development_mode: Literal["manual", "automatic"] | None
    # 整个工作流的运行状态
    status: Literal[
        "running",       # 正常运行
        "completed",     # 所有结束条件满足
        "failed",        # 出现不可恢复错误
        "stopped",       # 用户主动停止
    ]

    # --------------------------------------------------------
    # B2. Agent 候选结果与用户确认
    # 未经用户确认的 Agent 输出不会直接写入正式业务 State
    pending_action: RouteType | None
    pending_state_update: dict[str, Any] | None
    pending_summary: str | None
    pending_state_changes: dict[str, Any] | None
    pending_interaction: RequirementInteraction | None
    user_decision: UserDecision | None
    # 要求重新生成时传回 Agent 的用户意见
    user_feedback: str | None

    # Requirement Agent 可以多轮提出建议，但只有用户确认后才能进入下游。
    requirements_confirmed: bool
    requirement_conversation: list[dict[str, str]]
    confirmed_requirement_fields: list[str]

    # --------------------------------------------------------
    # C. 测验需求与开发约束
    # 精简测验需求，包括构念、人群、题量及程序固定的作答与计分规则
    test_specification: TestSpecification  | None
    # 记录每项需求来自用户明确指定还是系统推定
    specification_sources: dict[
        str,
        Literal[
            "user",
            "inferred",
            "system_default",
        ],
    ] | None
    # --------------------------------------------------------
    # D. 统一构念—题目细目表
    # 版本化构念档案；构念语义不再由每次运行的 LLM 重写
    construct_profile: dict[str, Any] | None
    # 只包含构念快照、维度配额与固定题号的细目表
    blueprint: GenerationBlueprint | None
    # 当前正在处理的蓝图单元
    current_blueprint_cell: GenerationCell | None
    # 每个蓝图单元已生成、通过、拒绝和缺失的题目数量
    blueprint_progress: dict[str, dict[str, int]]
    # 筛选或删题后对双向细目表的覆盖检查结果
    blueprint_coverage: dict[str, Any] | None
    # Program-owned slot IDs map to anonymous, separately reviewed content.
    item_skeletons: dict[str, CompactItemSkeleton]
    skeleton_reviews: dict[str, dict[str, Any]]
    skeleton_review_history: dict[str, list[dict[str, Any]]]
    skeleton_failures: dict[str, dict[str, Any]]
    skeleton_slot_failure_pending: bool
    # --------------------------------------------------------
    # E. 逐题生成
    # 从统一细目表镜像出的逐题骨架
    item_specifications: list[ItemSpecification]
    # 当前准备生成或正在使用的题目规格
    current_item_specification: dict[str, Any] | None
    # 当前正在生成、审核或修改的题目
    current_item: SJTItem | None
    # 所有通过逐题审核的候选题目
    item_pool: list[dict[str, Any]]
    # 已接受题目的结构化情境与反应模式，供后续出题和跨题审查使用
    item_pattern_profiles: dict[str, ItemPatternProfile]
    # 已接受题目按情境类别计数；蓝图生成配额另见 blueprint.context_quotas
    context_usage: dict[str, int]
    # 被永久拒绝或达到修改上限的题目
    rejected_items: list[dict[str, Any]]

    # --------------------------------------------------------
    # F. 逐题审查与修改
    # 统一审题结果；最终路由由 repair_tasks 和一次修题标记确定性派生
    current_item_review: UnifiedItemReview | None
    review_process_status: str
    item_content_status: str
    current_review_request_id: str | None
    current_review_item_id: str | None
    current_review_item_version: int | None
    current_review_retry_count: int
    current_item_repair_attempted: bool
    current_item_repair_failure: str | None
    # Program-owned counters for the fixed-ID item loop. Three unsuccessful
    # directed revisions trigger a full realization rewrite under the same ID.
    current_item_revision_count: int
    current_item_rewrite_count: int
    current_item_replacement_count: int
    current_skeleton_repair_required: bool
    max_item_revision_attempts: int
    max_item_rewrite_rounds: int
    max_item_replacement_attempts: int
    # 每道题的完整版本、审题结果和修改历史
    item_history: dict[str, list[dict[str, Any]]]
    # 淘汰补题使用新ID时保留稳定的替代关系。
    item_lineage: dict[str, dict[str, Any]]

    # --------------------------------------------------------
    # G. 题库冻结与定向补题
    # 虚拟施测前由程序创建的不可变版本快照及其内容指纹
    item_bank_id: str | None
    item_bank_version: int
    item_bank_fingerprint: str | None
    item_bank_frozen_at: str | None
    frozen_item_bank: list[dict[str, Any]]
    # 被程序强制纳入开发版候选池的题目及其质量警告。
    provisional_item_flags: dict[str, dict[str, Any]]
    # 局部返修候选汇总后、统一正式施测前的程序审计结果。
    candidate_bank_audit: NotRequired[dict[str, Any] | None]
    # 全运行最多开发的不同候选题数相对最终题量的倍数。
    # --------------------------------------------------------
    # H. 虚拟被试设计与数据
    # 虚拟样本的模型、人格水平、重复次数和随机化方案
    virtual_sample_config: VirtualSampleConfig | None
    # 旧虚拟协议恢复时，向用户说明为何必须重新配置
    virtual_sample_reconfiguration_reason: str | None
    # 兼容旧等分档配置时记录迁移证据；旧指标仍必须重算。
    virtual_sample_migration_events: list[dict[str, Any]]
    virtual_analysis_reconfiguration_reason: str | None
    # 虚拟被试的人格参数和模型参数
    virtual_respondents: list[VirtualRespondentRef]
    # 虚拟作答数据文件的位置或数据库标识
    virtual_response_data_ref: str | None
    # 新版题库增量重测时可复用的上一版完整作答 manifest
    previous_virtual_response_data_ref: str | None
    # 虚拟作答数据的摘要，例如样本量和完成率
    virtual_response_summary: dict[str, Any] | None
    # 虚拟作答实际绑定的冻结题库版本
    virtual_response_item_bank_id: str | None
    virtual_response_item_bank_version: int | None

    # --------------------------------------------------------
    # I. 心理测量分析结果
    # 每次成功完成心理测量分析递增；恢复或重复展示不递增。
    psychometric_analysis_round: int
    # 最近一次分析的统一轮次展示结构，供CLI、网页与报告共用。
    psychometric_round_result: dict[str, Any] | None
    # 每一轮临时组卷的虚拟整卷传导指标、题量、Token 与时间记录。
    psychometric_iteration_history: list[dict[str, Any]]
    # 连续多轮整卷指标没有实质改善时的自动停止状态。
    psychometric_plateau_status: NotRequired[dict[str, Any] | None]
    psychometric_plateau_patience: NotRequired[int]
    psychometric_plateau_min_delta: NotRequired[float]
    # 每道题的难度、区分度、项目总分相关和选项功能等
    item_statistics: dict[str, dict[str, Any]]
    # 信度、总分分布、分量表相关和测验信息等
    test_statistics: dict[str, Any] | None
    # 探索性或验证性因子分析结果
    factor_results: dict[str, Any] | None
    # IRT 项目参数、信息函数和条件标准误
    irt_results: dict[str, Any] | None
    # 虚拟群体间的 DIF 分析结果
    dif_results: dict[str, Any] | None

    # --------------------------------------------------------
    # J. 题目筛选
    # 经过多目标优化后保留的题目
    selected_items: list[dict[str, Any]]
    # 质量合格但未被蓝图选入正式测验的备选题目
    reserve_items: list[dict[str, Any]]
    # 心理测量统计异常的外层待处理题目队列；首项可带诊断，其余等待逐题处理
    items_to_revise: list[dict[str, Any]]
    # 本轮需要完全重写的题目
    items_to_regenerate: list[dict[str, Any]]
    # 蓝图覆盖已满足，因此暂不消耗模型调用的可返修候选
    items_deferred_for_revision: list[dict[str, Any]]
    # 筛选过程中删除的题目
    removed_items: list[dict[str, Any]]
    # 每道题被保留或删除的原因
    selection_reasons: dict[str, str]
    # 多目标筛选函数的完整输出
    selection_results: dict[str, Any] | None
    # 心理测量筛选与返修记录
    psychometric_selection_history: list[dict[str, Any]]
    # 同一题目版本首次达到 retain 后锁定其通过资格；后续指标仅监测。
    locked_retained_item_versions: dict[str, int]
    # 历史最佳、满足蓝图的正式组合及其组合级指标。
    best_assembly_candidate: dict[str, Any] | None
    # 当前正在执行的心理测量返修任务
    active_psychometric_repair: dict[str, Any] | None
    # 当前唯一等待用户确认的心理测量返修建议
    psychometric_repair_confirmation: dict[str, Any] | None
    # 各题已经完成的心理测量返修轮数
    psychometric_repair_rounds: dict[str, int]
    # 任意 defer 诊断均自动进入同槽位补题队列；不是静默保留或淘汰。
    psychometric_repair_defer_after_rounds: int
    # 旧检查点兼容字段；新流程不再用它决定保留、淘汰或补题。
    max_psychometric_repair_rounds: int
    # 心理测量返修、淘汰和重新验证历史
    psychometric_repair_history: list[dict[str, Any]]
    # 最近一次并发返修批次的并发度与成功/失败摘要。
    psychometric_repair_batch_summary: NotRequired[dict[str, Any] | None]
    # 已锁定正式题的最新重算指标若漂移，只记录监测警告，不撤销资格。
    psychometric_monitoring_warnings: list[dict[str, Any]]
    psychometric_repair_user_decision: Literal["start"] | None
    item_final_dispositions: dict[str, dict[str, Any]]

    # --------------------------------------------------------
    # K. 组卷、重新计分与测验审核
    # 组卷函数产生的候选正式测验
    assembled_test: dict[str, Any] | None
    # 当前测验级审核结果
    test_review_result: dict[str, Any] | None
    # 当前已经重新组卷的次数
    reassembly_round: int
    # 允许重新组卷的最大次数
    max_reassembly_rounds: int
    # 当前已经调整计分的次数
    rescore_round: int
    # 允许调整计分的最大次数
    max_rescore_rounds: int
    # 重计分后是否必须重新审核、冻结、模拟和分析
    rescore_pending_revalidation: bool
    # 每次重新组卷和调整计分的历史
    test_revision_history: list[dict[str, Any]]


    # --------------------------------------------------------
    # L. 最终输出
    # 最终通过审核的正式候选 PSJT
    final_test: dict[str, Any] | None
    # 包含全部题目及元数据的题目数据库引用
    item_database_ref: str | None
    # 测验开发过程和模拟测量证据的技术报告
    technical_report: dict[str, Any] | None
    # 虚拟被试设计、参数和分析结果报告
    virtual_respondent_report: dict[str, Any] | None

    # --------------------------------------------------------
    # M. 结束条件
    # 每项结束条件是否已经满足
    completion_checks: dict[str, bool]
    # 尚未满足的结束条件及原因
    unmet_completion_conditions: list[str]

    # --------------------------------------------------------
    # N. 流程安全与运行记录
    # Execute 已经执行的总任务次数
    step_count: int
    # 防止 Router–Execute 出现无限循环的最大任务次数
    max_steps: int
    # 执行过程中发生的结构化错误
    errors: list[dict[str, Any]]
    # 结构化执行轨迹；供命令行调试器和后续可视化界面共同使用
    execution_history: list[TraceEvent]


def create_initial_state(
    user_request: str,
    *,
    target_population: str | None = None,
    target_construct: str | None = None,
    requested_item_count: int | None = None,
    max_steps: int = 100,
) -> PSJTState:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if requested_item_count is not None and requested_item_count < 1:
        raise ValueError("requested_item_count must be at least 1")

    supplied_specification = any(
        value is not None
        for value in (
            target_population,
            target_construct,
            requested_item_count,
        )
    )

    test_specification = None
    specification_sources = None
    if supplied_specification:
        from sjt_system.authoring.construct_registry import (
            construct_selection_from_profile,
            resolve_construct_profile,
        )
        from sjt_system.config import DEFAULT_OUTPUT_LANGUAGE

        construct_selection = None
        if target_construct:
            try:
                construct_selection = construct_selection_from_profile(
                    resolve_construct_profile(target_construct)
                )
            except ValueError:
                # The requirement agent will resolve or question the original
                # free-text request against the supplied construct catalog.
                construct_selection = None
        test_specification = {
            "construct_selection": construct_selection,
            "target_population": target_population,
            "final_item_count": requested_item_count,
            "output_language": DEFAULT_OUTPUT_LANGUAGE,
        }
        specification_sources = {
            key: "user"
            for key, value in test_specification.items()
            if value is not None and key != "output_language"
        }
        specification_sources["output_language"] = "system_default"

    return {
        "user_request": user_request,
        "run_id": str(uuid4()),
        "route": None,
        "current_phase": "requirements",
        "item_development_mode": None,
        "status": "running",
        "pending_action": None,
        "pending_state_update": None,
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": None,
        "requirements_confirmed": False,
        "requirement_conversation": [],
        "confirmed_requirement_fields": [],
        "test_specification": test_specification,
        "specification_sources": specification_sources,
        "construct_profile": None,
        "blueprint": None,
        "current_blueprint_cell": None,
        "blueprint_progress": {},
        "blueprint_coverage": None,
        "item_skeletons": {},
        "skeleton_reviews": {},
        "skeleton_review_history": {},
        "skeleton_failures": {},
        "skeleton_slot_failure_pending": False,
        "item_specifications": [],
        "current_item_specification": None,
        "current_item": None,
        "item_pool": [],
        "item_pattern_profiles": {},
        "context_usage": {},
        "rejected_items": [],
        "current_item_review": None,
        "review_process_status": "not_started",
        "item_content_status": "not_evaluated",
        "current_review_request_id": None,
        "current_review_item_id": None,
        "current_review_item_version": None,
        "current_review_retry_count": 0,
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "current_item_revision_count": 0,
        "current_item_rewrite_count": 0,
        "current_item_replacement_count": 0,
        "current_skeleton_repair_required": False,
        "max_item_revision_attempts": 3,
        "max_item_rewrite_rounds": 3,
        "max_item_replacement_attempts": 2,
        "item_history": {},
        "item_lineage": {},
        "item_bank_id": None,
        "item_bank_version": 0,
        "item_bank_fingerprint": None,
        "item_bank_frozen_at": None,
        "frozen_item_bank": [],
        "provisional_item_flags": {},
        "candidate_bank_audit": None,
        "virtual_sample_config": None,
        "virtual_sample_reconfiguration_reason": None,
        "virtual_sample_migration_events": [],
        "virtual_analysis_reconfiguration_reason": None,
        "virtual_respondents": [],
        "virtual_response_data_ref": None,
        "previous_virtual_response_data_ref": None,
        "virtual_response_summary": None,
        "virtual_response_item_bank_id": None,
        "virtual_response_item_bank_version": None,
        "psychometric_analysis_round": 0,
        "psychometric_round_result": None,
        "psychometric_iteration_history": [],
        "psychometric_plateau_status": None,
        "psychometric_plateau_patience": 2,
        "psychometric_plateau_min_delta": 0.01,
        "item_statistics": {},
        "test_statistics": None,
        "factor_results": None,
        "irt_results": None,
        "dif_results": None,
        "selected_items": [],
        "reserve_items": [],
        "items_to_revise": [],
        "items_to_regenerate": [],
        "items_deferred_for_revision": [],
        "removed_items": [],
        "selection_reasons": {},
        "selection_results": None,
        "psychometric_selection_history": [],
        "locked_retained_item_versions": {},
        "best_assembly_candidate": None,
        "active_psychometric_repair": None,
        "psychometric_repair_confirmation": None,
        "psychometric_repair_rounds": {},
        "psychometric_repair_defer_after_rounds": 3,
        "max_psychometric_repair_rounds": 3,
        "psychometric_repair_history": [],
        "psychometric_repair_batch_summary": None,
        "psychometric_monitoring_warnings": [],
        "psychometric_repair_user_decision": None,
        "item_final_dispositions": {},
        "assembled_test": None,
        "test_review_result": None,
        "reassembly_round": 0,
        "max_reassembly_rounds": 3,
        "rescore_round": 0,
        "max_rescore_rounds": 3,
        "rescore_pending_revalidation": False,
        "test_revision_history": [],
        "final_test": None,
        "item_database_ref": None,
        "technical_report": None,
        "virtual_respondent_report": None,
        "completion_checks": {},
        "unmet_completion_conditions": [],
        "step_count": 0,
        "max_steps": max_steps,
        "errors": [],
        "execution_history": [],
    }
