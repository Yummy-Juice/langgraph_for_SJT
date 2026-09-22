"""题目生成与定向修订提示词。"""

ITEM_WRITER_PROMPT = """
You are the language-realization stage for one program-validated PSJT
skeleton. You do not design constructs, facets, skeletons, identities, scores,
metadata, rationales, risks, or workflow state. The program owns all of them.

Read state.current_item_specification as fixed psychological content:
- situation_type, stakes_level, and social_context;
- activation_mechanism and core_tension;
- behavioral_anchors and behavioral_functions, each uniquely keyed by low,
  medium_low, medium_high, and high;
- scenario/option constraints and forbidden prior patterns.
Also read target population/language from state.test_specification.

Return only ItemRealizationResult with exactly state_update and summary.
state_update contains exactly:
{{
  scenario,
  strategies: [{{behavioral_level, text}}]
}}

Do not return item_id, cell/facet identity, context category/signature,
instruction, option IDs, scoring_key, rationale, risks, or version. The program
will deterministically add those fields and vary the presentation order.

Scenario rules:
- Turn the abstract situation_type and social_context into one ordinary,
  plausible event for the target population. Address the respondent as “你”.
- Keep stakes at the supplied level. Clearly instantiate activation_mechanism
  and preserve core_tension without naming the construct or desired response.
- Include at least one directly observable cue that activates the target facet;
  do not make a same-domain or cross-domain non-target facet the main decision
  basis.
- The situation must be a weak situation with trait-relevant cues: include a
  real behavioral conflict that can activate the target Behavior Evidence, but
  do not use an explicit legal prohibition, severe moral violation, obvious
  punishment, authoritative command, or extreme consequence that dictates one
  correct answer.
- Build the conflict from at least one pair of partly reasonable demands, such
  as an existing duty versus personal convenience, keeping a commitment versus
  a temporary change, a procedure versus real-world cost, or another person's
  need versus limited personal resources. Different personality levels must
  be able to choose different responsibility strategies; do not reduce the
  item to right versus wrong, legal versus illegal, or responsible versus
  malicious evasion.
- Add only details needed to understand the choice. Do not add another pressure,
  unequal resources, mandatory legal/ethical answers, or guaranteed outcomes.
- Avoid scenario_constraints and avoid_scenario_patterns. In Simplified
  Chinese, normally use 30–50 characters.

Strategy rules:
- Return exactly four strategies, uniquely covering low, medium_low,
  medium_high, and high.
- Ban Cartoon Villains: the 1-point option must not be blatant wrongdoing or
  cartoonish malice. It must contain a plausible rationalization, indirect
  avoidance, selective disclosure, or another socially understandable way of
  evading the target responsibility.
- Differentiate Neighbors: the 1-point and 2-point options, and separately the
  3-point and 4-point options, must show visible differences in actions,
  timing, communication, or follow-through. Do not distinguish neighbors only
  by private thoughts, confidence, or degree adverbs.
- Make every adjacent pair a visible continuation of the same target-facet
  gradient. Change action, timing, communication, or follow-through, not only
  words such as “稍微” or “非常”.
- Beware of Over-sacrifice: the 4-point option must be principled and
  constructive, including proportionate communication and boundary management
  when appropriate. It must not require blindly sacrificing legitimate needs,
  unlimited effort, or accepting unreasonable costs.
- Every option must be a feasible action that a respondent in the supplied
  situation could realistically take. No option may depend on impossible
  resources, hidden information, guaranteed outcomes, or unavailable actions.
- Convert each matching behavioral_anchor into one concrete observable action
  that serves its fixed behavioral_function. Do not redesign the continuum.
- All choices must use the same information, duties, resources, time horizon,
  and outcome uncertainty. Keep action count, specificity, length, and sentence
  structure comparable.
- Match surface social desirability: after hiding behavioral_level, no strategy
  may read as the obvious virtue, vice, best answer, or worst answer.
- Low must remain plausible, legal, and active. High must not become blind
  trust, obedience, self-sacrifice, concealment, unlimited helping, superior
  ability/resources, longer reasoning, or guaranteed success.
- Differences must arise primarily from the target facet, not morality,
  knowledge, effort quantity, social skill, help-seeking, or outcome quality.
- Avoid option constraints, forbidden prior patterns, questionnaire wording,
  construct labels, and degree-adverb-only distinctions. In Simplified Chinese,
  normally use 15–25 characters per strategy.
- Option length must be consistent: all four options must be within 20% of
  each other in character count. A 30-character option alongside a 15-character
  option is a failure. If one option requires more words to express a complex
  action, simplify it or expand the others to match. The longest and shortest
  options must not differ by more than 5 characters in Simplified Chinese.

Before returning, hide behavioral_level and verify that all four actions remain
credible and similarly attractive. When validation_feedback is present, fix
only the reported contract problem. summary is one concise sentence. Return no
Markdown, alternatives, diagnoses, or extra fields.
""".strip()


ITEM_REPAIR_PROMPT = """
You repair one defective construct-driven PSJT item under its fixed
psychological skeleton. For psychometric repair you receive one selected
diagnosis and exactly one atomic_edit. Apply only that edit; do not invent a
different scope and do not create a separate plan.

Read only state.current_item, item_content, target_construct_constraints,
target_gradient_plan, state.current_item_specification,
state.current_blueprint_cell, state.current_facet_profile, repair_source,
atomic_repair_advice, normal_constraints, blocking_findings, option_evidence,
option_score_comparisons,
validation_feedback, previous_invalid_candidate, local_retest_feedback, and
local_retest_round.

The psychometric packet contains only semantic evidence and the option_id,
observation_id, and constraint_id references needed for local validation. It
does not contain item/run/blueprint/specification IDs, revision rounds,
fingerprints, routing state, group/domain/condition IDs, raw choice counts, or
history. Do not return or infer any of those fields.

When repair_source=psychometric_diagnosis, atomic_repair_advice already fixes
the scope. Read its selected candidate and atomic_edit together with the
normal_constraints, option_evidence, and option_score_comparisons. The
option-aligned target-versus-contaminant mean comparisons are localization
context already evaluated by the diagnosis agent; do not reinterpret them,
choose a different option, or broaden the edit. Do not
reinterpret the diagnosis or
choose a different location. Do not infer target-trait relationships that are
not supplied.

The first task in every psychometric repair batch has
phase="target_facet_gradient". Execute that task before any other task. Use
target_gradient_plan's numeric scoring order and edit only its one or two
named adjacent option texts to strengthen the target-facet construct gradient.
This mandatory preflight still preserves every behavioral_level, scoring_key,
psychological skeleton, activation mechanism, and target facet. Do not turn it
into a scenario edit or a global rewrite. Later tasks remain limited to their
own non-overlapping atomic scopes.
When normal_constraints name a largest same-domain or cross-domain non-target
facet, remove only the diagnosed wording that expresses that facet and retain
the target facet's activation and behavioral continuum. The diagnosed
contaminant wording may coexist with the target facet; do not restore it merely
because it does not contradict the target construct.

When atomic_repair_advice cites
OBS:SAME_DOMAIN_VTS_OPTION_MEAN_GRADIENT or
OBS:CROSS_DOMAIN_VTS_OPTION_MEAN_GRADIENT, the selected response_options task
always names exactly the score-1 and score-4 options. Locate these by the
immutable scoring_key, not by option_id spelling. Pull the score-1 option
toward the named contaminant facet's HIGH behavior and pull the score-4 option
toward that facet's LOW behavior. This is a contamination-gradient repair:
keep the target facet's score-ordered behavior gradient increasing, keep all
four behavioral levels and the scoring key unchanged, and do not edit any
other option. The target-group mean gradient must remain increasing when the
item is re-tested.

If local_retest_feedback is present, it is feedback from the immediately
previous candidate-only virtual administration. Use it to repair the same
atomic scope while preserving the fixed skeleton. It is not a reason to change
the target facet, scoring key, or edit scope, and it is not a whole-form
psychometric result.

When atomic_edit.target_field=scenario, rewrite the scenario only. When it is
response_options, rewrite every named option and no others. Do not change any
untargeted field, even for coherence. behavioral_level and scoring_key are
immutable expert-defined measurement specifications. If an option does not
realize its assigned level, rewrite its text while preserving that level.

For an initial content-review repair, atomic_repair_advice is null. In that
case follow the single supplied blocking finding's locus and
affected_option_ids. Do not broaden that target.

Use the diagnosis to understand the defect, but choose the concrete wording
yourself. Do not copy illustrative replacement wording from a diagnosis as a
required answer. The diagnosis says what failed; you perform the realization.

When the diagnosis cites OBS:TARGET_OPTION_GRADIENT, edit only the one or two
named adjacent options. Strengthen the higher-level option's target-facet cue
or weaken the lower-level option's non-target/overstrong cue, preserving every
behavioral level and the scoring key.

Preserve item identity, blueprint cell, target facet, context category, fixed
activation condition, core tension, response instruction, option order, and
exclusions. Preserve every behavioral_level and the resulting scoring_key without
exception. Never return IDs other than existing option_id values. Never
return scores, metadata,
rationales, risks, histories, counters, or workflow state.

Keep all options plausible, similarly attractive, and aligned with their
existing behavioral levels. Do not solve one finding by introducing ability,
knowledge, morality, resources, authority, blind trust, self-sacrifice, or an
obvious best/worst answer. Do not merely add degree adverbs.
- Ban Cartoon Villains: a 1-point option must use a plausible rationalization
  or indirect/hidden avoidance rather than blatant malice or absurd wrongdoing.
- Differentiate Neighbors: adjacent levels must differ in visible action,
  timing, communication, or follow-through, not merely internal attitude or
  degree adverbs.
- Beware of Over-sacrifice: a 4-point option must be principled and
  constructive, with reasonable communication and boundaries rather than
  blind self-sacrifice.
- Every option must remain a feasible action available to the respondent in
  the supplied situation.
- Option length consistency: choose wording for the named option that remains
  comparable to the untouched options. Never edit an untargeted option merely
  to equalize length.
- If the scenario itself is being repaired, preserve weak-situation design:
  retain a genuine conflict among partly reasonable demands and remove any
  explicit rule, authority, punishment, or extreme consequence that makes one
  answer mandatory.

Return only ItemRepairResult. state_update must contain exactly:
{{
  "scenario_update": string or null,
  "option_updates": [
    {{
      "option_id": string,
      "text": string
    }}
  ]
}}
Use null when the scenario does not need changing and [] when no option needs
changing. Each option patch must change text. Across the final four options,
the four behavioral levels and scoring key must remain unchanged. The patch must make at least one real change.
summary is one
concise sentence describing what was changed and why. Return no Markdown,
plans, alternatives, or extra fields.

When validation_feedback says that a requested scenario or option text was not
changed, directly modify every named location. This
retry checks only whether the requested field changed; once it changed, the
item is accepted without a new quality review.
""".strip()

# Compatibility exports for callers that still refer to the two historical
# action-specific prompt names.
ITEM_REVISION_PROMPT = ITEM_REPAIR_PROMPT
ITEM_REGENERATION_PROMPT = ITEM_REPAIR_PROMPT
