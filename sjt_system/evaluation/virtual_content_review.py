"""Version-bound, virtual-only investigation before evidence-authorized edits."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

from typing_extensions import NotRequired, TypedDict

from sjt_system.evaluation.repair_recovery import MAX_CALLS, journaled_call
from sjt_system.evaluation.scenario_detection import (
    ProbeUnavailable, RepairArchive, validate_candidate, validate_differences,
)
from sjt_system.prompt.virtual_content_review_prompt import PROMPTS
from sjt_system.runtime.trace import utc_timestamp

PROTOCOL = "mte_cosmin_virtual_content_review_v1"
MAX_RETESTS = 3
MAX_MEASUREMENTS = 1 + MAX_RETESTS
EVIDENCE_SCOPE = "exploratory_virtual_development_evidence"
CRITERIA = ("relevance", "comprehensiveness", "comprehensibility")
ITERATION_GATES = (
    "citc_pass",
    "target_hedges_g_pass",
    "target_ipip_spearman_rho_pass",
    "discriminant_delta_min_pass",
)
FINDING_CRITERIA = {*CRITERIA, "trait_activation", "construct_purity", "option_gradient", "scoring_alignment"}
FIELDS = {"scenario", "response_options", "response_instruction", "skeleton"}
PROFILES = (
    ("literal", "limited", "reflective"), ("global", "familiar", "quick"),
    ("detailed", "limited", "quick"), ("literal", "familiar", "reflective"),
    ("global", "limited", "reflective"), ("detailed", "familiar", "quick"),
)
CONTENT_CONSTRAINTS = (
    ("CONTENT:RELEVANCE", "The wording must match the frozen construct, population and use context."),
    ("CONTENT:COMPREHENSIVENESS", "The item set must preserve its planned construct and slot coverage."),
    ("CONTENT:COMPREHENSIBILITY", "The scenario and options must support the intended interpretation without ambiguity."),
)


class Ratings(TypedDict):
    relevance: str
    comprehensiveness: str
    comprehensibility: str


class Finding(TypedDict):
    criterion: str
    field: str
    quote: str
    constraint_id: str
    problem: str
    required_change: str


class ExpertReview(TypedDict):
    criteria: Ratings
    findings: list[Finding]
    needs_scenario_probe: bool
    summary: str


class InterviewAnswer(TypedDict):
    selected_option_id: str
    paraphrase: str
    retrieval: str
    judgment: str
    selection_reason: str
    unclear_quotes: list[str]


class OptionMeaning(TypedDict):
    option_id: str
    meaning: str


class InterviewIssue(TypedDict):
    quote: str
    problem: str


class InterviewProbe(TypedDict):
    interpretation: str
    option_interpretations: list[OptionMeaning]
    issues: list[InterviewIssue]
    summary: str


class RepairTask(TypedDict):
    task_id: str
    field: str
    option_ids: list[str]
    quote: str
    constraint_id: str
    evidence_ids: list[str]
    required_change: str


class ContentDiagnosis(TypedDict):
    decision: str
    summary: str
    replacement_scope: str
    repair_tasks: list[RepairTask]


class OptionText(TypedDict):
    option_id: str
    text: str


class ContentPatch(TypedDict):
    reason: str
    scenario: NotRequired[str]
    response_options: NotRequired[list[OptionText]]


SCHEMAS = {
    "expert_review": ExpertReview, "expert_arbitration": ExpertReview,
    "interview_answer": InterviewAnswer, "interview_probe": InterviewProbe,
    "content_diagnosis": ContentDiagnosis, "content_edit": ContentPatch,
}


def is_enabled(state: Mapping[str, Any]) -> bool:
    return state.get("virtual_content_review_protocol") == PROTOCOL


def iteration_gates_pass(statistics: Mapping[str, Any]) -> bool:
    qualification = statistics.get("qualification") or {}
    return isinstance(qualification, Mapping) and all(
        qualification.get(gate) is True for gate in ITERATION_GATES
    )


def round_label(round_number: int) -> str:
    return "首测" if round_number == 1 else f"第{round_number}批虚拟施测"


def archive_path(state: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    from sjt_system.runtime.output_paths import scoped_output
    return (scoped_output("virtual_content_review", "outputs/virtual_content_review")
            / ("run-" + quote(str(state["run_id"]), safe=""))
            / f"analysis-{int(state.get('psychometric_analysis_round') or 0)}"
            / ("item-" + quote(str(item["item_id"]), safe=""))
            / f"v{int(item['version'])}.json")


def build_virtual_content_repair_entry(state, item, *, revision_round):
    from sjt_system.evaluation.diagnosis import build_scenario_repair_entry
    entry = build_scenario_repair_entry(state, item, revision_round=revision_round)
    entry.update(repair_protocol=PROTOCOL, queue_status="pending_virtual_review",
                 diagnosis_status="virtual_content_review_planned",
                 source_analysis_round=int(state.get("psychometric_analysis_round") or 0),
                 content_review_archive_ref=str(archive_path(state, item)))
    entry.pop("scenario_archive_ref", None)
    entry["atomic_repair_advice"] = {
        "protocol": PROTOCOL, "decision": "investigate", "repair_tasks": [],
        "summary": "先核查材料、虚拟专家盲审和认知访谈，再决定证据支持的修改或暂缓。",
    }
    return entry


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _object(value, fields, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} requires exactly {sorted(fields)}")


def _field_text(item, materials, field):
    if field == "response_options":
        return "\n".join(str(row["text"]) for row in item["response_options"])
    if field == "skeleton":
        return json.dumps(materials.get("fixed_skeleton") or {}, ensure_ascii=False)
    return str(item.get(field) or "")


def material_packet(state, item, evidence):
    if evidence.get("planning_error"):
        raise ValueError(str(evidence["planning_error"]))
    constraints = [deepcopy(dict(row)) for row in evidence.get("normal_constraints") or []
                   if isinstance(row, Mapping) and row.get("constraint_id") and row.get("statement")]
    target = evidence.get("target_construct_constraints") or [
        row for row in constraints if row.get("component") in {"facet", "behavior_evidence"}]
    if not target:
        raise ValueError("virtual content review requires frozen target construct material")
    for identifier, statement in CONTENT_CONSTRAINTS:
        if not any(row["constraint_id"] == identifier for row in constraints):
            constraints.append({"constraint_id": identifier, "component": "content_boundary", "statement": statement})
    population = (state.get("test_specification") or {}).get("target_population") or state.get("target_population")
    _text(population, "target_population")
    return {
        "constraints": constraints, "target_construct_constraints": deepcopy(target),
        "target_population": population, "fixed_skeleton": deepcopy(evidence.get("fixed_skeleton") or {}),
        "use_context": {"blueprint_cell_id": item.get("blueprint_cell_id"),
                        "context_category": item.get("context_category")},
        "coverage": [{"item_id": row.get("item_id"), "item_version": row.get("version"),
                      "facet_id": row.get("target_dimension_id"), "blueprint_cell_id": row.get("blueprint_cell_id"),
                      "scenario": row.get("scenario"), "response_instruction": row.get("response_instruction"),
                      "response_options": deepcopy(row.get("response_options") or [])}
                     for row in state.get("frozen_item_bank") or state.get("item_pool") or [item]],
        "temporary_form_item_ids": list((state.get("psychometric_iteration_history") or [{}])[-1].get("form_item_ids") or []),
        "blueprint_cells": deepcopy((state.get("blueprint") or {}).get("cells") or []),
        "construct_version": evidence.get("construct_version"), "evidence_scope": EVIDENCE_SCOPE,
    }


def validate_expert_review(raw, item, materials):
    _object(raw, ExpertReview.__annotations__, "expert review")
    _object(raw["criteria"], CRITERIA, "content criteria")
    if any(value not in {"supported", "problem", "indeterminate"} for value in raw["criteria"].values()):
        raise ValueError("unsupported virtual content criterion status")
    if not isinstance(raw["needs_scenario_probe"], bool) or not isinstance(raw["findings"], list):
        raise ValueError("invalid expert findings or probe flag")
    identifiers = {row["constraint_id"] for row in materials["constraints"]}
    for finding in raw["findings"]:
        _object(finding, Finding.__annotations__, "expert finding")
        if finding["criterion"] not in FINDING_CRITERIA or finding["field"] not in FIELDS:
            raise ValueError("unsupported expert finding criterion or field")
        if finding["constraint_id"] not in identifiers:
            raise ValueError("expert finding cites an unknown constraint")
        if _text(finding["quote"], "expert quote") not in _field_text(item, materials, finding["field"]):
            raise ValueError("expert quote is not present in the reviewed version")
        _text(finding["problem"], "expert problem")
        _text(finding["required_change"], "expert change")
    _text(raw["summary"], "expert summary")
    return deepcopy(dict(raw))


def interview_packet(item, materials, index):
    options = item["response_options"]
    ordered = options[index % len(options):] + options[:index % len(options)]
    mapping = {f"choice_{number + 1}": row["option_id"] for number, row in enumerate(ordered)}
    reading, experience, decision = PROFILES[index]
    return {
        "role": {"role_id": f"C{index + 1}", "target_population": materials["target_population"],
                 "reading_style": reading, "experience": experience, "decision_style": decision,
                 "synthetic": True},
        "item": {"scenario": item["scenario"], "response_instruction": item.get("response_instruction"),
                 "response_options": [{"option_id": code, "text": row["text"]}
                                      for code, row in zip(mapping, ordered)]},
    }, mapping


def _public_text(packet):
    return "\n".join([str(packet["item"].get("scenario") or ""),
                      str(packet["item"].get("response_instruction") or ""),
                      *(row["text"] for row in packet["item"]["response_options"])])


def validate_interview_answer(raw, packet):
    _object(raw, InterviewAnswer.__annotations__, "interview answer")
    codes = {row["option_id"] for row in packet["item"]["response_options"]}
    if raw["selected_option_id"] not in codes:
        raise ValueError("interview selected an unknown anonymous option")
    for field in ("paraphrase", "retrieval", "judgment", "selection_reason"):
        _text(raw[field], field)
    if not isinstance(raw["unclear_quotes"], list):
        raise ValueError("interview unclear_quotes must be a list")
    for quoted in raw["unclear_quotes"]:
        if _text(quoted, "interview quote") not in _public_text(packet):
            raise ValueError("interview quote is absent from the presented item")
    return deepcopy(dict(raw))


def validate_interview_probe(raw, packet):
    _object(raw, InterviewProbe.__annotations__, "interview probe")
    _text(raw["interpretation"], "interview interpretation")
    _text(raw["summary"], "interview summary")
    expected = {row["option_id"] for row in packet["item"]["response_options"]}
    if not isinstance(raw["option_interpretations"], list):
        raise ValueError("option interpretations must be a list")
    observed = []
    for row in raw["option_interpretations"]:
        _object(row, OptionMeaning.__annotations__, "option interpretation")
        observed.append(row["option_id"])
        _text(row["meaning"], "option meaning")
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("interview must interpret every anonymous option exactly once")
    if not isinstance(raw["issues"], list):
        raise ValueError("interview issues must be a list")
    for issue in raw["issues"]:
        _object(issue, InterviewIssue.__annotations__, "interview issue")
        if _text(issue["quote"], "interview issue quote") not in _public_text(packet):
            raise ValueError("interview issue quote is not in the presented item")
        _text(issue["problem"], "interview issue")
    return deepcopy(dict(raw))


def evidence_registry(reviews, interviews):
    registry = {}
    for index, report in enumerate(reviews):
        for number, finding in enumerate(report["findings"]):
            registry[f"EXPERT:{index + 1}:F{number + 1}"] = deepcopy(finding)
    for interview in interviews:
        role_id = interview["role"]["role_id"]
        registry[f"INTERVIEW:{role_id}:ANSWER"] = deepcopy(interview["answer"])
        for number, issue in enumerate(interview["probe"]["issues"]):
            registry[f"INTERVIEW:{role_id}:I{number + 1}"] = deepcopy(issue)
    return registry


def validate_diagnosis(raw, item, materials, registry):
    _object(raw, ContentDiagnosis.__annotations__, "content diagnosis")
    _text(raw["summary"], "diagnosis summary")
    if raw["decision"] not in {"repair", "replace", "defer"} or not isinstance(raw["repair_tasks"], list):
        raise ValueError("invalid content repair decision")
    if raw["decision"] == "defer":
        if raw["repair_tasks"] or raw["replacement_scope"] != "none":
            raise ValueError("defer cannot authorize edits or a replacement")
        return deepcopy(dict(raw))
    if not 1 <= len(raw["repair_tasks"]) <= 4:
        raise ValueError("repair requires one to four evidence-backed tasks")
    if raw["decision"] == "repair" and raw["replacement_scope"] != "none":
        raise ValueError("ordinary repair cannot change the fixed design")
    if raw["decision"] == "replace" and raw["replacement_scope"] not in {"skeleton", "blueprint"}:
        raise ValueError("replacement requires an explicit design scope")
    constraints = {row["constraint_id"] for row in materials["constraints"]}
    valid_options = {row["option_id"] for row in item["response_options"]}
    task_ids, touched = set(), set()
    for task in raw["repair_tasks"]:
        _object(task, RepairTask.__annotations__, "repair task")
        if _text(task["task_id"], "task ID") in task_ids:
            raise ValueError("duplicate repair task ID")
        task_ids.add(task["task_id"])
        field = task["field"]
        if field not in {"scenario", "response_options", "skeleton"}:
            raise ValueError("task changes a protected field")
        if field == "skeleton" and raw["decision"] != "replace":
            raise ValueError("skeleton changes must create a new candidate")
        if not isinstance(task["option_ids"], list):
            raise ValueError("authorized option IDs must be a list")
        option_ids = task["option_ids"]
        if field == "response_options":
            if not option_ids or len(set(option_ids)) != len(option_ids) or not set(option_ids) <= valid_options:
                raise ValueError("task must explicitly authorize existing option IDs")
            targets = {f"option:{identifier}" for identifier in option_ids}
            text = "\n".join(row["text"] for row in item["response_options"] if row["option_id"] in option_ids)
        else:
            if option_ids:
                raise ValueError("non-option task cannot authorize option IDs")
            targets, text = {field}, _field_text(item, materials, field)
        if touched & targets:
            raise ValueError("repair tasks overlap")
        touched |= targets
        if _text(task["quote"], "task quote") not in text:
            raise ValueError("task quote is absent from its authorized field")
        if task["constraint_id"] not in constraints:
            raise ValueError("task cites an unknown construct/content constraint")
        refs = task["evidence_ids"]
        if not isinstance(refs, list) or not refs or any(ref not in registry for ref in refs):
            raise ValueError("task requires existing virtual expert/interview evidence")
        grounded = False
        for ref in refs:
            finding = registry[ref]
            quote = finding.get("quote")
            if isinstance(quote, str) and (task["quote"] in quote or quote in task["quote"]):
                if ref.startswith("EXPERT:"):
                    grounded |= finding.get("field") == field and finding.get("constraint_id") == task["constraint_id"]
                else:
                    grounded = True
            if task["quote"] in (finding.get("unclear_quotes") or []):
                grounded = True
        if not grounded:
            raise ValueError("task evidence does not identify the quoted content problem in its authorized field")
        _text(task["required_change"], "required change")
    return deepcopy(dict(raw))


def apply_authorized_patch(item, raw, tasks):
    scenario_allowed = any(task["field"] == "scenario" for task in tasks)
    authorized = {identifier for task in tasks for identifier in task["option_ids"]}
    required = {"reason"} | ({"scenario"} if scenario_allowed else set()) | ({"response_options"} if authorized else set())
    _object(raw, required, "authorized content patch")
    _text(raw["reason"], "patch reason")
    candidate = deepcopy(dict(item))
    if scenario_allowed:
        candidate["scenario"] = _text(raw["scenario"], "scenario patch")
    if authorized:
        if not isinstance(raw["response_options"], list):
            raise ValueError("option patch must be a list")
        by_id = {}
        for row in raw["response_options"]:
            _object(row, OptionText.__annotations__, "option patch")
            identifier = row["option_id"]
            if identifier not in authorized or identifier in by_id:
                raise ValueError("unknown or duplicate patched option")
            by_id[identifier] = _text(row["text"], "option text")
        if set(by_id) != authorized:
            raise ValueError("patch must cover the authorized options exactly")
        for row in candidate["response_options"]:
            if row["option_id"] in by_id:
                row["text"] = by_id[row["option_id"]]
    validate_candidate(item, candidate)
    return candidate


class ContentReviewArchive(RepairArchive):
    def __init__(self, path, item, *, source):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if (self.data.get("protocol") != PROTOCOL or self.data.get("baseline_item") != item
                    or self.data.get("source") != source):
                raise ValueError("content review archive inputs changed; legacy evidence cannot be relabeled")
        else:
            self.data = {"protocol": PROTOCOL, "baseline_item": deepcopy(dict(item)),
                         "source": deepcopy(source), "stage": "materials", "status": "running",
                         "calls": [], "errors": [], "rounds": [], "rewrite_count": 0,
                         "expert_reviews": [], "cognitive_interviews": [], "created_at": utc_timestamp()}
            self.save()

    def result(self):
        data = self.data
        return {"protocol": PROTOCOL, "evidence_scope": EVIDENCE_SCOPE,
                "status": data["status"], "stage": data["stage"], "archive_ref": str(self.path),
                "source_analysis_round": data["source"]["analysis_round"],
                "reviewed_item_id": data["baseline_item"]["item_id"],
                "reviewed_item_version": data["baseline_item"]["version"],
                "expert_reviews": deepcopy(data.get("expert_reviews") or []),
                "cognitive_interviews": deepcopy(data.get("cognitive_interviews") or []),
                "diagnosis": deepcopy(data.get("diagnosis")),
                "cosmin_gate": deepcopy(data.get("cosmin_gate")),
                "candidate": deepcopy(data.get("candidate")),
                "staged_design": deepcopy(data.get("staged_design")),
                "requires_remeasurement": data.get("status") in {"ready", "replenished"},
                "auto_replenishment": data.get("auto_replenishment") is True,
                "replenishment_reason": data.get("replenishment_reason"),
                "pause_reason": data.get("pause_reason"), "model_call_count": len(data["calls"])}

    @classmethod
    def load_existing(cls, path, item):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        source = saved.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("content review archive is missing its immutable source")
        return cls(path, item, source=source)


def _validate_completed_archive(data, item, materials):
    reports = data.get("expert_reviews") or []
    interviews = data.get("cognitive_interviews") or []
    if len(reports) not in {2, 3} or len(interviews) != len(PROFILES):
        raise ValueError("completed archive requires two experts and all six cognitive roles")
    for report in reports:
        validate_expert_review(report, item, materials)
    for index, interview in enumerate(interviews):
        packet, mapping = interview_packet(item, materials, index)
        if (interview.get("role") != packet["role"] or interview.get("option_mapping") != mapping
                or interview.get("item_id") != item["item_id"] or interview.get("item_version") != item["version"]
                or interview.get("synthetic") is not True):
            raise ValueError("archived interview is not bound to the reviewed version and virtual role")
        validate_interview_answer(interview["answer"], packet)
        validate_interview_probe(interview["probe"], packet)
    diagnosis = validate_diagnosis(data.get("diagnosis"), item, materials, evidence_registry(reports, interviews))
    candidate, bundle = data.get("candidate"), data.get("staged_design")
    if data["status"] == "deferred":
        if diagnosis["decision"] != "defer" or candidate is not None or bundle is not None:
            raise ValueError("deferred archive cannot contain an authorized modification")
        return
    if diagnosis["decision"] not in {"repair", "replace"} or (diagnosis["decision"] == "replace") != bool(bundle):
        raise ValueError("staged candidate is inconsistent with the evidence diagnosis")
    validate_content_candidate(item, candidate, bundle)
    if not bundle:
        tasks = diagnosis["repair_tasks"]
        patch = {"reason": diagnosis["summary"]}
        if any(task["field"] == "scenario" for task in tasks):
            patch["scenario"] = candidate["scenario"]
        option_ids = {identifier for task in tasks for identifier in task["option_ids"]}
        if option_ids:
            patch["response_options"] = [{"option_id": row["option_id"], "text": row["text"]}
                                         for row in candidate["response_options"] if row["option_id"] in option_ids]
        if apply_authorized_patch(item, patch, tasks) != candidate:
            raise ValueError("archived candidate changed wording outside its authorized tasks")


def make_model_invoker(state):
    from sjt_system.agent.agent_factory import create_agent
    from sjt_system.agent.client import get_model_request_timeout_seconds
    from sjt_system.evaluation.scenario_detection import make_model_invoker as probe_invoker
    from sjt_system.evaluation.scenario_detection import PROMPTS as probe_prompts, SCHEMAS as probe_schemas
    _, original_metadata = probe_invoker(state)
    metadata = {kind: deepcopy(original_metadata["open_action" if kind.startswith("interview_") else "rewrite_options"])
                for kind in SCHEMAS}
    metadata.update({kind: original_metadata[kind] for kind in ("open_action", "compare_adjacent")})
    schemas, prompts = {**probe_schemas, **SCHEMAS}, {**probe_prompts, **PROMPTS}

    async def invoke(kind, payload):
        agent = create_agent(prompts[kind], schemas[kind], **metadata[kind])
        return await asyncio.wait_for(agent.ainvoke({"input_data": json.dumps(payload, ensure_ascii=False)}),
                                      timeout=get_model_request_timeout_seconds())
    return invoke, metadata


async def _diagnostic_probe(archive, evidence, item, call):
    from sjt_system.evaluation.scenario_detection import (
        build_detection_plan, build_roles, build_adjacent_pairs, validate_actions,
        evaluate_differences,
    )
    plan = build_detection_plan(evidence)
    roles, rows = build_roles(plan), []
    for index, role in enumerate(roles):
        action = await call(f"probe/{index}", "open_action", {"scenario": item["scenario"], "role": role["persona"]}, validate_actions)
        rows.append({"action": action})
    if "diagnostic_pairs" not in archive.data:
        archive.data["diagnostic_pairs"] = build_adjacent_pairs(roles, rows)
        archive.save()
    packet = archive.data["diagnostic_pairs"]
    differences = {}

    def validate_pair(raw, pair):
        result, correction = _validate_single_pair_difference(raw, pair)
        if correction:
            correction = {
                "item_id": item["item_id"],
                "key": f"probe/compare/{pair['pair_id']}",
                **correction,
            }
            corrections = archive.data.setdefault("pair_id_corrections", [])
            if correction not in corrections:
                corrections.append(correction)
                archive.save()
        return result

    for pair in packet["records"]:
        output = await call(f"probe/compare/{pair['pair_id']}", "compare_adjacent",
                            {"construct": plan["target_construct"], "pairs": [pair]},
                            lambda raw, pair=pair: validate_pair(raw, pair))
        differences.update(output)
    return {"respondent_count": len(roles), "decision": evaluate_differences(roles, packet["identity"], differences)}


def _validate_single_pair_difference(
    raw: Any, pair: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str] | None]:
    try:
        return validate_differences(raw, [pair]), None
    except ProbeUnavailable as exc:
        if str(exc) != "unknown or duplicate anonymous pair ID":
            raise
    expected_id = pair.get("pair_id")
    records = raw.get("records") if isinstance(raw, Mapping) else None
    if (not isinstance(expected_id, str) or not expected_id.startswith("p")
            or not isinstance(records, list) or len(records) != 1
            or not isinstance(records[0], Mapping)
            or records[0].get("pair_id") != expected_id[1:]):
        raise ProbeUnavailable("unknown or duplicate anonymous pair ID")
    corrected = deepcopy(raw)
    corrected["records"][0]["pair_id"] = expected_id
    return validate_differences(corrected, [pair]), {
        "provided_id": str(records[0]["pair_id"]),
        "canonical_id": expected_id,
        "method": "single_pair_missing_prefix",
    }


async def run_virtual_content_repair(*, state, item, evidence, path, invoke,
                                     model_info=None, rebuild=None, progress=None,
                                     retry_automatically=False):
    analysis_round = int(state.get("psychometric_analysis_round") or 0)
    from sjt_system.evaluation.facet_iteration import is_enabled as fixed_iteration_enabled
    fixed_iteration = fixed_iteration_enabled(state)
    if (analysis_round < 1 or (not fixed_iteration and analysis_round >= MAX_MEASUREMENTS)
            or (fixed_iteration and state["facet_iteration_state"].get("status") != "awaiting_repair")):
        raise ValueError("content repair requires a completed measurement with a remaining retest")
    source = {"run_id": state["run_id"], "analysis_round": analysis_round,
              "evidence": deepcopy(dict(evidence)), "model_info": deepcopy(model_info or {}),
              "target_population": (state.get("test_specification") or {}).get("target_population"),
              "profiles": [list(profile) for profile in PROFILES],
              "material_context": {
                  "test_specification": deepcopy(state.get("test_specification") or {}),
                  "target_population": state.get("target_population"),
                  "blueprint": deepcopy(state.get("blueprint") or {}),
                  "candidate_items": deepcopy(state.get("frozen_item_bank") or state.get("item_pool") or [item]),
                  "temporary_form_item_ids": list((state.get("psychometric_iteration_history") or [{}])[-1].get("form_item_ids") or []),
              }}
    archive = ContentReviewArchive(path, item, source=source)
    data = archive.data

    async def call(key, kind, payload, validate):
        data.update(stage=kind, status="running", pause_reason=None)
        archive.save()
        if progress:
            progress({"status": "running", "stage": kind, "model_call_count": len(data["calls"])})
        return await journaled_call(records=data["calls"], key=key, kind=kind, payload=payload,
                                    invoke=invoke, validate=validate, save=archive.save,
                                    prompt=PROMPTS.get(kind, ""), model=(model_info or {}).get(kind),
                                    retry_automatically=retry_automatically)
    try:
        materials = material_packet(state, item, evidence)
        if data["status"] == "replenished":
            validate_content_candidate(item, data.get("candidate"), data.get("staged_design"))
            return archive.result()
        if data["status"] in {"ready", "deferred"}:
            completed_stage = data["stage"]
            data["stage"] = "cached_validation"
            _validate_completed_archive(data, item, materials)
            data["stage"] = completed_stage
            return archive.result()
        data["materials"] = materials
        data["plan"] = {"constraints": materials["constraints"], "fixed_skeleton": materials["fixed_skeleton"]}
        archive.save()
        review_item = {key: deepcopy(item.get(key)) for key in (
            "scenario", "response_instruction", "response_options", "scoring_key")}
        reviews = []
        for number, expertise in enumerate(("construct_and_content", "sjt_and_measurement")):
            report = await call(f"expert/{number + 1}", "expert_review",
                                {"expertise": expertise, "item": review_item, "materials": materials},
                                lambda raw: validate_expert_review(raw, item, materials))
            reviews.append(report)
            data["expert_reviews"] = deepcopy(reviews)
            archive.save()
        if reviews[0]["criteria"] != reviews[1]["criteria"] or {
                (row["criterion"], row["field"], row["quote"]) for row in reviews[0]["findings"]} != {
                (row["criterion"], row["field"], row["quote"]) for row in reviews[1]["findings"]}:
            arbitration = await call("expert/arbitration", "expert_arbitration",
                                     {"item": review_item, "materials": materials, "independent_reviews": reviews},
                                     lambda raw: validate_expert_review(raw, item, materials))
            reviews.append(arbitration)
            data["expert_reviews"] = deepcopy(reviews)
            archive.save()
        interviews = []
        for index in range(len(PROFILES)):
            packet, mapping = interview_packet(item, materials, index)
            answer = await call(f"interview/C{index + 1}/answer", "interview_answer", packet,
                                lambda raw, packet=packet: validate_interview_answer(raw, packet))
            probe = await call(f"interview/C{index + 1}/probe", "interview_probe", {**packet, "previous_answer": answer},
                               lambda raw, packet=packet: validate_interview_probe(raw, packet))
            interviews.append({"role": packet["role"], "option_mapping": mapping, "answer": answer, "probe": probe,
                               "item_id": item["item_id"], "item_version": item["version"], "synthetic": True})
            data["cognitive_interviews"] = deepcopy(interviews)
            archive.save()
        if any(report["needs_scenario_probe"] for report in reviews):
            data["diagnostic_probe"] = await _diagnostic_probe(archive, evidence, item, call)
            archive.save()
        registry = evidence_registry(reviews, interviews)
        diagnosis = await call("diagnosis", "content_diagnosis",
                               {"item": deepcopy(dict(item)), "materials": materials,
                                "statistical_symptoms": deepcopy(evidence.get("observations") or []),
                                "facet_form_failure": deepcopy(evidence.get("facet_form_failure")),
                                "expert_reviews": reviews, "cognitive_interviews": interviews,
                                "evidence_registry": registry, "diagnostic_probe": data.get("diagnostic_probe"),
                                "prior_repairs": deepcopy(evidence.get("prior_atomic_repairs") or [])},
                               lambda raw: validate_diagnosis(raw, item, materials, registry))
        data["diagnosis"] = diagnosis
        data["cosmin_gate"] = {"evidence_scope": EVIDENCE_SCOPE, "reviewed_item_version": item["version"],
                               "relevance": [report["criteria"]["relevance"] for report in reviews],
                               "comprehensiveness": [report["criteria"]["comprehensiveness"] for report in reviews],
                               "comprehensibility": "virtual_interview_evidence_recorded",
                               "post_edit_comprehensibility": "not_reassessed", "human_content_validity": "not_evaluated"}
        archive.save()
        if diagnosis["decision"] == "defer":
            data.update(status="deferred", stage="diagnosed", candidate=None)
        elif diagnosis["decision"] == "replace":
            if rebuild is None:
                raise ValueError("evidence-authorized replacement requires a fixed-slot rebuilder")
            bundle = await rebuild(archive, 1 if diagnosis["replacement_scope"] == "skeleton" else 2)
            candidate = bundle["item"]
            validate_content_candidate(item, candidate, bundle)
            data.update(candidate=deepcopy(candidate), staged_design=deepcopy(bundle), status="ready", stage="staged")
        else:
            tasks = diagnosis["repair_tasks"]
            candidate = await call("edit", "content_edit",
                                   {"item": deepcopy(dict(item)), "materials": materials,
                                    "repair_tasks": tasks, "evidence_registry": registry},
                                   lambda raw: apply_authorized_patch(item, raw, tasks))
            data.update(candidate=candidate, status="ready", stage="staged")
        archive.save()
    except Exception as exc:
        data["errors"].append({"stage": data["stage"], "message": str(exc), "recorded_at": utc_timestamp()})
        data.update(status="paused", pause_reason=str(exc))
        archive.save()
    return archive.result()


def validate_content_candidate(original, candidate, bundle=None):
    if not bundle:
        validate_candidate(original, candidate)
        return
    if (candidate != bundle.get("item") or candidate.get("item_id") == original["item_id"]
            or bundle.get("replaces_item_id") != original["item_id"]
            or (bundle.get("slot") or {}).get("specification_id") != candidate.get("item_id")):
        raise ValueError("invalid evidence-authorized replacement identity")
    for field in ("target_dimension_id", "blueprint_cell_id", "scoring_key", "response_instruction"):
        if candidate.get(field) != original.get(field):
            raise ValueError(f"replacement changed protected {field}")
    if [(row["option_id"], row["behavioral_level"]) for row in candidate["response_options"]] != [
            (row["option_id"], row["behavioral_level"]) for row in original["response_options"]]:
        raise ValueError("replacement changed option identity, order or level")
    if not candidate.get("scenario") or candidate.get("version") != 1:
        raise ValueError("replacement requires a new version-one item")
    specification = bundle.get("specification") or {}
    if (specification.get("specification_id") != candidate.get("item_id")
            or not isinstance(bundle.get("skeleton"), Mapping)
            or not bundle["skeleton"]
            or not isinstance(bundle.get("replacement_number"), int)
            or bundle["replacement_number"] < 1):
        raise ValueError("replacement requires its own specification, skeleton and lineage ordinal")
