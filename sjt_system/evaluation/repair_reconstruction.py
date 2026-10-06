"""Stage structural replacements without mutating the official bank."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing_extensions import TypedDict

from sjt_system.evaluation.repair_recovery import RepairStopped, journaled_call
from sjt_system.evaluation.scenario_detection import (
    DESIGN_LIMIT_REASON, DETECTION_PROTOCOL, MAX_REWRITES, SCENARIO_LENGTH_INSTRUCTION,
    RepairArchive, design_records, failure_memory,
)


class BlueprintChoice(TypedDict):
    choice_id: str
    reason: str


CHOICE_PROMPT = """根据失败记忆和返修知识，从choices中选择一个尚未尝试的机制/情境组合。
目标facet、行为证据和测量单元不能改变。只输出choice_id和reason，不发明ID。
优先避免前几轮反复出现的混淆节点；知识属于模拟经验，不是已证明因果规律。"""
MAX_REPLENISHMENT_CONTENT_ATTEMPTS = 3


def _content_replacement_context(data, baseline, stage):
    from sjt_system.evaluation.virtual_content_review import evidence_registry
    diagnosis = data.get("diagnosis") or {}
    expected_scope = "skeleton" if stage == 1 else "blueprint"
    if stage not in {1, 2} or diagnosis.get("decision") != "replace" or diagnosis.get("replacement_scope") != expected_scope:
        raise RepairStopped("reconstruction scope is not authorized by the virtual content diagnosis")
    registry = evidence_registry(data.get("expert_reviews") or [], data.get("cognitive_interviews") or [])
    cited_ids = {identifier for task in diagnosis.get("repair_tasks") or [] for identifier in task["evidence_ids"]}
    return {
        "reviewed_item": deepcopy(baseline), "authorized_diagnosis": deepcopy(diagnosis),
        "cited_content_evidence": {identifier: deepcopy(registry[identifier]) for identifier in cited_ids},
        "frozen_constraints": deepcopy(data["plan"]["constraints"]),
        "source_analysis_round": data["source"]["analysis_round"],
    }


def make_rebuilder(state, baseline, knowledge, *, review_generated_candidate=True):
    async def rebuild(archive, stage):
        from sjt_system.agent.agent_factory import (
            create_agent, compact_skeleton_agent, item_writer_agent, item_review_agent,
            PSYCHOMETRIC_REASONING_ROLE_MANIFEST, COMPACT_SKELETON_BATCH_PROMPT,
            ITEM_WRITER_PROMPT, UNIFIED_ITEM_REVIEW_PROMPT,
        )
        from sjt_system.agent.client import get_model_request_timeout_seconds
        from sjt_system.authoring.generation_plan import (
            _expansion_models, resolve_blueprint_design, classify_compact_skeletons,
            materialize_item_specifications,
        )
        from sjt_system.authoring.context import (
            build_item_generation_context, build_item_model_state, build_unified_review_context,
        )
        from sjt_system.authoring.items import (
            canonicalize_item_agent_update, validate_item_agent_update,
            validate_item_review_diagnosis, validate_item_review,
        )
        data = archive.data
        bundles = data.setdefault("reconstruction_jobs", {})
        job = bundles.setdefault(str(stage), {})
        if job.get("bundle"):
            return deepcopy(job["bundle"])
        blueprint = deepcopy(state.get("blueprint") or {})
        cell = next((c for c in blueprint.get("cells", []) if c["cell_id"] == baseline["blueprint_cell_id"]), None)
        source_slot = next((s for s in blueprint.get("slots", []) if s["specification_id"] == baseline["item_id"]), None)
        if cell is None or source_slot is None:
            raise RepairStopped("reconstruction requires original blueprint cell and slot")
        expansions = _expansion_models(blueprint)
        reference = deepcopy(source_slot.get("candidate_reference") or {
            "mechanism_id": cell["mechanism_id"], "situation_id": cell["situation_id"]})
        history = failure_memory(data, archive.path)
        content_context = {}
        if not review_generated_candidate:
            content_context = _content_replacement_context(data, baseline, stage)
            history.append({"type": "evidence_authorized_content_replacement", "scenario": baseline["scenario"],
                            "content_revision": deepcopy(content_context), "archive_ref": str(archive.path)})
        role = PSYCHOMETRIC_REASONING_ROLE_MANIFEST["psychometric_item_repair"]
        metadata = {"model_id": role["model_id"], "temperature": role.get("temperature"),
                    "thinking_type": role.get("thinking"), "reasoning_effort": role.get("reasoning_effort")}
        agents = {"skeleton": compact_skeleton_agent, "realization": item_writer_agent,
                  "review": item_review_agent}
        prompts = {"skeleton": COMPACT_SKELETON_BATCH_PROMPT, "realization": ITEM_WRITER_PROMPT,
                   "review": UNIFIED_ITEM_REVIEW_PROMPT, "blueprint_choice": CHOICE_PROMPT}
        async def invoke(kind, payload):
            agent = agents.get(kind)
            if kind == "blueprint_choice":
                agent = create_agent(CHOICE_PROMPT, BlueprintChoice, **metadata)
            return await asyncio.wait_for(agent.ainvoke({"input_data": payload}),
                                          timeout=get_model_request_timeout_seconds())
        call_prefix = f"rebuild-{stage}/{DETECTION_PROTOCOL}"
        async def call(key, kind, payload, validate):
            return await journaled_call(records=data["calls"], key=f"{call_prefix}/{key}",
                kind=kind, payload=payload, invoke=invoke, validate=validate, save=archive.save,
                prompt=prompts[kind], model=metadata if kind == "blueprint_choice" else {"agent": kind},
                retry_automatically=review_generated_candidate)
        if stage in (2, "budget_replenishment"):
            attempted = {(reference["mechanism_id"], reference["situation_id"])}
            for design in design_records(data):
                prior = (design.get("staged_design") or {}).get("slot", {}).get("candidate_reference")
                if prior:
                    attempted.add((prior["mechanism_id"], prior["situation_id"]))
            expansion = next(e for e in expansions if e.facet_id == cell["facet_id"])
            behavior = next(b for b in expansion.behavior_expansions if b.behavior_id == cell["behavior_id"])
            choices = [{"choice_id": f"{m.mechanism_id}/{s.situation_id}",
                        "mechanism_id": m.mechanism_id, "situation_id": s.situation_id,
                        "activation_mechanism": m.activation_mechanism, "situation": s.model_dump(mode="json")}
                       for m in behavior.mechanisms for s in m.situations
                       if (m.mechanism_id, s.situation_id) not in attempted]
            if not choices and stage == 2:
                raise RepairStopped("no untried blueprint combination for the fixed facet and behavior")
            if choices:
                def validate_choice(raw):
                    if set(raw) != {"choice_id", "reason"} or not isinstance(raw["reason"], str) or not raw["reason"].strip():
                        raise ValueError("blueprint choice requires a choice_id and reason")
                    selected = next((c for c in choices if c["choice_id"] == raw["choice_id"]), None)
                    if selected is None:
                        raise ValueError("blueprint choice is not in the authorized list")
                    return {"mechanism_id": selected["mechanism_id"], "situation_id": selected["situation_id"]}
                reference = await call("choice", "blueprint_choice", {"choices": choices,
                    "failure_memory": history, "repair_knowledge": knowledge,
                    **({"content_revision": content_context} if content_context else {})}, validate_choice)
        if "item_id" not in job:
            lineage = state.get("item_lineage") or {}
            root = (lineage.get(baseline["item_id"]) or {}).get("root_item_id", baseline["item_id"])
            existing = {s["specification_id"] for s in blueprint["slots"]} | set(lineage)
            existing.update(j["item_id"] for j in bundles.values() if j.get("item_id"))
            number = 1
            while f"{root}-R{number}" in existing:
                number += 1
            job.update(item_id=f"{root}-R{number}", root_item_id=root, replacement_number=number)
            archive.save()
        new_id = job["item_id"]
        slot = {"specification_id": new_id, "blueprint_cell_id": cell["cell_id"], "candidate_reference": reference}
        blueprint["slots"].append(slot)
        cell["planned_generation_count"] += 1
        design = resolve_blueprint_design(blueprint, cell, expansions=expansions, candidate_reference=reference)
        def validate_skeleton(raw):
            skeleton = raw.get("state_update", {}).get("item_skeleton")
            if any(skeleton == (d.get("plan") or {}).get("fixed_skeleton") for d in design_records(data)):
                raise ValueError("reconstruction repeated an earlier skeleton")
            all_skeletons = {**deepcopy(state.get("item_skeletons") or {}), new_id: skeleton}
            check = classify_compact_skeletons(blueprint, all_skeletons)
            if new_id not in check["valid"]:
                raise ValueError(str(check["invalid"].get(new_id, "invalid skeleton")))
            return check["valid"][new_id]
        skeleton = await call("skeleton", "skeleton", {
            "test_specification": state.get("test_specification"), "current_facet": design["facet"],
            "behavior_evidence": design["behavior_evidence"], "activation_mechanism": design["activation_mechanism"],
            "situation": design["situation"], "failure_memory": history, "repair_knowledge": knowledge,
            **({"content_revision": content_context} if content_context else {}),
        }, validate_skeleton)
        specification = materialize_item_specifications(blueprint, {new_id: skeleton}, expansions=expansions)[0]
        local = {**state, "blueprint": blueprint, "current_item": None, "active_psychometric_repair": None,
                 "current_item_specification": specification, "current_blueprint_cell": cell,
                 "item_specifications": [*(state.get("item_specifications") or []), specification],
                 "item_skeletons": {**(state.get("item_skeletons") or {}), new_id: skeleton}}
        def validate_item(raw):
            update = canonicalize_item_agent_update("generate_item", raw["state_update"],
                specification=state.get("test_specification"), blueprint_cell=cell,
                item_specification=specification, previous_item=None)
            candidate = update["current_item"]
            texts = {o["behavioral_level"]: o["text"] for o in candidate["response_options"]}
            candidate["response_options"] = [{**deepcopy(o), "text": texts[o["behavioral_level"]]}
                                              for o in baseline["response_options"]]
            candidate["scoring_key"] = deepcopy(baseline["scoring_key"])
            validate_item_agent_update("generate_item", update, target_blueprint_cell_id=cell["cell_id"],
                specification=state.get("test_specification"), blueprint_cell=cell,
                item_specification=specification, previous_item=None)
            if candidate["scenario"].strip() in {h["scenario"].strip() for h in history
                                               if h.get("type") != "evidence_authorized_content_replacement"}:
                raise ValueError("reconstruction repeated a failed scenario")
            return candidate
        while True:
            if stage == "budget_replenishment" and sum(
                row.get("key") == f"{call_prefix}/realization" and row.get("status") == "content_rejected"
                for row in data["calls"]
            ) >= MAX_REPLENISHMENT_CONTENT_ATTEMPTS:
                raise RepairStopped("budget replenishment content review exhausted")
            candidate = await call("realization", "realization", {
                "action": "generate_item", "state": build_item_model_state(local),
                "generation_context": build_item_generation_context(local),
                "failure_memory": history, "repair_knowledge": knowledge,
                **({"content_revision": content_context} if content_context else {}),
                "repair_scenario_instruction": SCENARIO_LENGTH_INSTRUCTION}, validate_item)
            local["current_item"] = candidate
            if not review_generated_candidate:
                review = None
                break
            generation_record = next(r for r in reversed(data["calls"]) if r["key"] == f"{call_prefix}/realization")
            def validate_review(raw):
                validate_item_review_diagnosis(raw, current_item=candidate)
                review = {"findings": raw["findings"], "repair_tasks": [], "summary": raw["summary"]}
                validate_item_review(review, current_item=candidate)
                return review
            review = await call(f"review-{generation_record['attempt']}", "review",
                                build_unified_review_context(local), validate_review)
            blockers = [f for f in review["findings"] if f.get("severity") == "blocking"]
            if not blockers:
                break
            generation_record.update(status="content_rejected", error=f"content review blocking findings: {blockers}",
                                     error_category="invalid_output")
            archive.save()
        constraints = [deepcopy(c) for c in data["plan"]["constraints"]
                       if c.get("component") not in {"skeleton", "situation", "activation_mechanism"}]
        constraints.extend([
            {"constraint_id": f"MECHANISM:{reference['mechanism_id']}", "component": "activation_mechanism", "statement": str(design["activation_mechanism"])},
            {"constraint_id": f"SITUATION:{reference['situation_id']}", "component": "situation", "statement": str(design["situation"])},
            {"constraint_id": f"SKELETON:TENSION:{new_id}", "component": "skeleton", "statement": skeleton["behavioral_tension"]},
        ])
        constraints.extend({"constraint_id": f"SKELETON:OPTION:{row['behavioral_level']}",
                            "component": "skeleton", "statement": row["behavioral_tendency"]}
                           for row in skeleton["option_structure"])
        job["bundle"] = {"item": candidate, "skeleton": skeleton, "specification": specification,
            "slot": slot, "constraints": constraints, "review": review,
            "root_item_id": job["root_item_id"], "replacement_number": job["replacement_number"],
            "replaces_item_id": baseline["item_id"], "design_stage": stage}
        archive.save()
        return deepcopy(job["bundle"])
    return rebuild


async def replenish_exhausted_design(state, baseline, progress, knowledge):
    """Generate one reviewed same-cell replacement after the final design limit."""
    archive_ref = progress.get("archive_ref")
    if not isinstance(archive_ref, str) or not Path(archive_ref).is_file():
        raise RepairStopped("budget replenishment requires the saved scenario archive")
    archive = RepairArchive(Path(archive_ref), baseline)
    data = archive.data
    if data.get("status") == "replenished" and data.get("replenishment_reason") == DESIGN_LIMIT_REASON:
        result = archive.result()
        validate_replacement(baseline, result["candidate"], result)
        return result
    if (progress.get("status") != "paused" or progress.get("pause_reason") != DESIGN_LIMIT_REASON
            or data.get("status") != "paused" or data.get("pause_reason") != DESIGN_LIMIT_REASON
            or data.get("design_stage") != 2 or data.get("rewrite_count", 0) < MAX_REWRITES):
        raise RepairStopped("only an exhausted scenario design can be replenished automatically")

    bundle = await make_rebuilder(state, baseline, knowledge)(archive, "budget_replenishment")
    review = bundle.get("review")
    if not isinstance(review, Mapping) or not isinstance(review.get("findings"), list) or any(
            finding.get("severity") == "blocking" for finding in review["findings"]):
        raise RepairStopped("budget replenishment has no passing content review")
    candidate = bundle["item"]
    validate_replacement(baseline, candidate, {"status": "replenished", "staged_design": bundle})
    data.update(candidate=deepcopy(candidate), staged_design=deepcopy(bundle),
                status="replenished", stage="staged", replenishment_reason=DESIGN_LIMIT_REASON,
                pause_reason=None, terminal_failure=False)
    archive.save()
    return archive.result()


def is_budget_replenishment(progress):
    return (progress.get("status") == "replenished"
            and progress.get("detection_protocol") == DETECTION_PROTOCOL
            and progress.get("replenishment_reason") == DESIGN_LIMIT_REASON
            and isinstance(progress.get("staged_design"), Mapping)
            and isinstance(progress.get("candidate"), Mapping))


def validate_replacement(original, candidate, progress):
    from sjt_system.evaluation.scenario_detection import validate_candidate
    bundle = progress.get("staged_design")
    if not bundle:
        validate_candidate(original, candidate)
        return
    base = bundle["item"]
    if base["item_id"] == original["item_id"] or bundle["replaces_item_id"] != original["item_id"]:
        raise ValueError("invalid replacement identity")
    for key in ("target_dimension_id", "blueprint_cell_id", "scoring_key"):
        if base[key] != original[key]:
            raise ValueError(f"replacement changed protected field {key}")
    if [(o["option_id"], o["behavioral_level"]) for o in base["response_options"]] != [
            (o["option_id"], o["behavioral_level"]) for o in original["response_options"]]:
        raise ValueError("replacement changed option identity or level")
    if progress.get("status") == "replenished":
        if candidate != base:
            raise ValueError("replenishment differs from the reviewed blueprint item")
    else:
        validate_candidate(base, candidate)
