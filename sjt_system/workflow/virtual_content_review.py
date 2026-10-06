"""Atomic publication of evidence-authorized virtual content revisions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from sjt_system.evaluation import virtual_content_review as review
from sjt_system.runtime.progress import emit_progress
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.evaluation.facet_iteration import (
    is_enabled as fixed_facet_iteration_enabled,
    local_repair_capacity_available,
    mark_committed_repairs,
)
from sjt_system.workflow.replacement_policy import (
    replacement_capacity_available as _replacement_capacity_available,
    replacement_quota_limit,
)


def replacement_measurement_pending(state: Mapping[str, Any]) -> bool:
    """Return whether a committed same-slot replacement needs one retest."""

    return bool(state.get("deferred_replacement_measurement_pending"))


def replacement_capacity_available(
    state: Mapping[str, Any], item_id: str,
) -> bool:
    """Check the persistent replacement policy without admitting a deferred item."""

    return _replacement_capacity_available(state, item_id)


def replacement_only_measurement_allowed(state: Mapping[str, Any]) -> bool:
    """Allow a deferred-only replacement batch after the base measurements."""

    if fixed_facet_iteration_enabled(state):
        controller = state["facet_iteration_state"]
        return controller.get("status") not in {"complete", "paused"}
    measurement = int(state.get("psychometric_analysis_round") or 0)
    return measurement < review.MAX_MEASUREMENTS or replacement_measurement_pending(state)


def _call_budget_exhausted(result):
    reason = str((result or {}).get("pause_reason") or "")
    return f"exhausted {review.MAX_CALLS} model calls" in reason


def _failure_replenishment_enabled(state):
    # This is a permanent workflow policy.  The state flag remains an audit
    # marker, but an old checkpoint cannot opt back into admitting failed or
    # deferred items as temporary form content.
    return True


def _history(state, results, *, committed=False, candidates=None):
    records = deepcopy(state.get("virtual_content_review_history") or [])
    for item_id, result in results.items():
        identity = (result.get("source_analysis_round"), item_id, result.get("reviewed_item_version"))
        records = [row for row in records if (
            row.get("source_analysis_round"), row.get("reviewed_item_id"), row.get("reviewed_item_version")) != identity]
        candidate = (candidates or {}).get(item_id)
        records.append({**deepcopy(result), "candidate": None, "staged_design": None,
                        "committed": committed and candidate is not None,
                        "candidate_item_id": candidate.get("item_id") if candidate else None,
                        "new_item_version": candidate.get("version") if candidate else None,
                        "recorded_at": utc_timestamp()})
    return records


def _archive_source(state, item, evidence, model_info):
    return {
        "run_id": state["run_id"],
        "analysis_round": int(state.get("psychometric_analysis_round") or 0),
        "evidence": deepcopy(dict(evidence)),
        "model_info": deepcopy(model_info or {}),
        "target_population": (
            state.get("test_specification") or {}
        ).get("target_population"),
        "profiles": [list(profile) for profile in review.PROFILES],
        "material_context": {
            "test_specification": deepcopy(state.get("test_specification") or {}),
            "target_population": state.get("target_population"),
            "blueprint": deepcopy(state.get("blueprint") or {}),
            "candidate_items": deepcopy(
                state.get("frozen_item_bank") or state.get("item_pool") or [item]
            ),
            "temporary_form_item_ids": list(
                (state.get("psychometric_iteration_history") or [{}])[-1].get(
                    "form_item_ids"
                )
                or []
            ),
        },
    }


async def _stage_deferred_replacement(
    *, state, item, evidence, path, model_info, rebuild, progress,
):
    """Turn a completed defer decision into an audited same-slot candidate."""

    source = _archive_source(state, item, evidence, model_info)
    archive = review.ContentReviewArchive(path, item, source=source)
    data = archive.data
    if data.get("status") == "replenished" and data.get(
        "replenishment_reason"
    ) == "deferred_decision":
        review.validate_content_candidate(
            item, data.get("candidate"), data.get("staged_design")
        )
        return archive.result()
    if not replacement_capacity_available(state, str(item["item_id"])):
        raise ValueError(
            f"题目 {item['item_id']} 无可用的同槽位替代容量"
        )
    materials = data.get("materials")
    if not isinstance(materials, Mapping):
        materials = review.material_packet(state, item, evidence)
        data["materials"] = deepcopy(materials)
    data.setdefault(
        "plan",
        {
            "constraints": deepcopy(materials.get("constraints") or []),
            "fixed_skeleton": deepcopy(materials.get("fixed_skeleton") or {}),
        },
    )
    original_diagnosis = deepcopy(data.get("diagnosis") or {})
    data.update(
        stage="deferred_replenishment",
        status="running",
        pause_reason=None,
        deferred_diagnosis=original_diagnosis,
    )
    archive.save()
    if progress:
        progress(
            {
                "status": "running",
                "stage": "deferred_replenishment",
                "message": "deferred题已删除，正在同一蓝图槽位生成替代题",
            }
        )
    bundle = await rebuild(archive, "budget_replenishment")
    candidate = bundle["item"]
    review.validate_content_candidate(item, candidate, bundle)
    data.update(
        candidate=deepcopy(candidate),
        staged_design=deepcopy(bundle),
        status="replenished",
        stage="replenishment_staged",
        pause_reason=None,
        auto_replenishment=True,
        replenishment_reason="deferred_decision",
        diagnosis={
            "decision": "replace",
            "summary": (
                "原题无法继续返修，已删除并在同一蓝图槽位生成替代题；"
                "替代题将在下一轮虚拟测量。"
            ),
            "replacement_scope": "blueprint",
            "repair_tasks": [],
        },
    )
    archive.save()
    return archive.result()


async def execute_virtual_content_review_batch(state):
    from sjt_system.authoring.context import build_item_pattern_profile
    from sjt_system.evaluation.repair_reconstruction import make_rebuilder
    from sjt_system.runtime.checkpoint import DEFAULT_CHECKPOINT_ROOT, save_run_checkpoint
    from sjt_system.runtime.output_paths import scoped_output

    measurement = int(state.get("psychometric_analysis_round") or 0)
    fixed_iteration = fixed_facet_iteration_enabled(state)
    if fixed_iteration and state["facet_iteration_state"].get("status") != "awaiting_repair":
        raise ValueError("No unfinished facet development round is available for repair")
    queued_entries = [
        entry
        for entry in [
            *(state.get("items_to_revise") or []),
            *(state.get("items_to_regenerate") or []),
        ]
        if isinstance(entry, Mapping)
    ]
    replacement_only_batch = any(
        entry.get("deferred_replacement_only") is True
        for entry in queued_entries
    )
    if not 1 <= measurement or (
        measurement >= review.MAX_MEASUREMENTS and not replacement_only_batch and not fixed_iteration
    ):
        raise ValueError("本轮已没有可用于验证修改的虚拟内容复审额度")
    measurement_record = next((row for row in state.get("psychometric_iteration_history") or []
                               if int(row.get("analysis_round") or 0) == measurement), None)
    if measurement_record is None:
        raise ValueError("必须先保存当轮逐题及临时卷指标，再启动虚拟内容复审返修")
    if not fixed_iteration and (state.get("psychometric_plateau_status") or {}).get("reached"):
        raise ValueError("平台期已收卷，不能重新启动返修")
    items = {str(row["item_id"]): deepcopy(dict(row))
             for row in state.get("item_pool") or state.get("frozen_item_bank") or []}
    if (measurement_record.get("virtual_content_review_protocol") != review.PROTOCOL
            or measurement_record.get("item_snapshots") != items
            or measurement_record.get("item_statistics_snapshot") != (state.get("item_statistics") or {})):
        raise ValueError("当轮测量账本与当前题目版本或原始指标不一致，不能用于返修")
    entries, seen = [], set()
    for entry in [*(state.get("items_to_revise") or []), *(state.get("items_to_regenerate") or [])]:
        item_id = str(entry["item_id"])
        if item_id in seen:
            continue
        seen.add(item_id)
        item = items[item_id]
        root_id = (state.get("item_lineage") or {}).get(item_id, {}).get("root_item_id", item_id)
        rounds = state.get("psychometric_repair_rounds") or {}
        if (
            not fixed_iteration
            and
            max(int(rounds.get(item_id) or 0), int(rounds.get(root_id) or 0))
            >= review.MAX_RETESTS
            and entry.get("deferred_replacement_only") is not True
        ):
            raise ValueError("同一根项三轮返修额度已耗尽，换ID不能重置额度")
        if entry.get("repair_protocol") != review.PROTOCOL:
            raise ValueError("旧返修队列不能静默转换成虚拟内容复审证据")
        if entry.get("source_analysis_round") != measurement:
            raise ValueError("返修队列必须绑定当前已保存的测量轮次")
        facet_failure = fixed_iteration and entry.get("facet_level_failure") is True
        if fixed_iteration and not facet_failure:
            raise ValueError("Fixed-cohort repairs require failed-facet authority")
        if facet_failure:
            facet_id = str(item.get("target_dimension_id"))
            controller = state["facet_iteration_state"]
            if (entry.get("facet_id") != facet_id
                    or entry.get("development_round") != controller["development_round"]
                    or facet_id not in controller["failed_facet_ids"]
                    or facet_id in controller["accepted_facets"]
                    or not local_repair_capacity_available(controller, facet_id)):
                raise ValueError("Facet-local repair authority does not match the failed facet")
        if not facet_failure and (state.get("locked_retained_item_versions") or {}).get(item_id) == item["version"]:
            raise ValueError("永久锁定题目不能进入虚拟返修队列")
        if not facet_failure and review.iteration_gates_pass((state.get("item_statistics") or {}).get(item_id) or {}):
            raise ValueError("当轮四项单题门槛已通过的题目需要失败facet的局部解冻授权")
        if (entry.get("diagnosis_evidence") or {}).get("current_item") != item:
            raise ValueError("返修证据与当前题目版本不一致")
        entries.append(deepcopy(dict(entry)))
    if not entries:
        raise ValueError("没有待处理的虚拟内容复审题目")
    save_run_checkpoint(state, checkpoint_root=scoped_output("run_checkpoints", DEFAULT_CHECKPOINT_ROOT))
    batch_id = f"virtual-content-review/{state['run_id']}/{measurement}"
    progress_by_id = {}

    async def run_one(entry):
        item_id = str(entry["item_id"])
        item = items[item_id]

        def bounded_rebuilder(*, review_generated_candidate):
            rebuild = make_rebuilder(state, item, None,
                                     review_generated_candidate=review_generated_candidate)

            async def bounded(archive, stage):
                if not replacement_capacity_available(state, item_id):
                    raise ValueError("No same-slot replacement capacity is available")
                return await rebuild(archive, stage)

            return bounded

        def progress(event):
            emit_progress({"type": "psychometric_subagent_progress", "batch_id": batch_id,
                           "batch_total": len(entries), "item_id": item_id,
                           "message": "虚拟内容复审：核查材料、审题、访谈与证据定位", **event})
        try:
            invoke, model_info = review.make_model_invoker(state)
            evidence = entry["diagnosis_evidence"]
            archive_ref = entry.get("content_review_archive_ref") or review.archive_path(
                state, item
            )
            review_evidence = evidence
            if entry.get("qualified_item_thaw") is True:
                # A resumed bottom-quartile queue may point at an archive
                # created before the thaw metadata was added.  Reuse that
                # archive's immutable source evidence; the thaw authorization
                # remains auditable on the queue entry and is not relabeled
                # into the archived source.
                try:
                    cached_archive = review.ContentReviewArchive.load_existing(
                        archive_ref, item
                    )
                except (OSError, ValueError):
                    cached_archive = None
                if cached_archive is not None:
                    source_evidence = (cached_archive.data.get("source") or {}).get("evidence")
                    if isinstance(source_evidence, Mapping):
                        review_evidence = deepcopy(dict(source_evidence))
            if entry.get("deferred_replacement_only") is True:
                result = await _stage_deferred_replacement(
                    state=state,
                    item=item,
                    evidence=review_evidence,
                    path=archive_ref,
                    model_info=model_info,
                    rebuild=bounded_rebuilder(review_generated_candidate=True),
                    progress=progress,
                )
            else:
                result = await review.run_virtual_content_repair(
                    state=state,
                    item=item,
                    evidence=review_evidence,
                    path=archive_ref,
                    invoke=invoke,
                    model_info=model_info,
                    rebuild=bounded_rebuilder(review_generated_candidate=False),
                    progress=progress,
                    retry_automatically=_failure_replenishment_enabled(state),
                )
            if (_failure_replenishment_enabled(state)
                    and result.get("status") == "paused"
                    and _call_budget_exhausted(result)):
                progress({"status": "replenishing", "stage": "budget_replenishment",
                          "message": "失败调用预算耗尽，删除原题并在同一蓝图槽位补题"})
                archive = review.ContentReviewArchive.load_existing(result["archive_ref"], item)
                bundle = await bounded_rebuilder(review_generated_candidate=True)(archive, "budget_replenishment")
                candidate = bundle["item"]
                review.validate_content_candidate(item, candidate, bundle)
                archive.data.update(
                    candidate=deepcopy(candidate), staged_design=deepcopy(bundle),
                    status="replenished", stage="replenishment_staged",
                    pause_reason=None, auto_replenishment=True,
                    replenishment_reason="failure_call_budget_exhausted",
                    diagnosis={
                        "decision": "replace", "summary": (
                            f"原题失败调用达到上限（{review.MAX_CALLS}），"
                            "已删除并在同一蓝图槽位重新生成候选题"
                        ), "replacement_scope": "blueprint", "repair_tasks": [],
                    },
                )
                archive.save()
                result = archive.result()
            if result.get("status") == "deferred":
                progress(
                    {
                        "status": "replenishing",
                        "stage": "deferred_replenishment",
                        "message": "deferred题已删除并替换；替代题将在下一轮测量",
                    }
                )
                result = await _stage_deferred_replacement(
                    state=state,
                    item=item,
                    evidence=review_evidence,
                    path=archive_ref,
                    model_info=model_info,
                    rebuild=bounded_rebuilder(review_generated_candidate=True),
                    progress=progress,
                )
        except Exception as exc:
            result = {"protocol": review.PROTOCOL, "status": "paused", "stage": "setup",
                      "reviewed_item_id": item_id, "reviewed_item_version": item["version"],
                      "source_analysis_round": measurement, "pause_reason": str(exc),
                      "evidence_scope": review.EVIDENCE_SCOPE}
        progress_by_id[item_id] = result
        progress({"status": "completed" if result["status"] in {"ready", "deferred", "replenished"} else "failed",
                  "stage": result.get("stage"), "archive_ref": result.get("archive_ref")})
        return item_id, result

    outcomes = await asyncio.gather(*(run_one(entry) for entry in entries))
    drafts, deferred, failures, occupied = {}, {}, {}, set(items)
    for item_id, result in outcomes:
        try:
            bundle = result.get("staged_design")
            if bundle:
                root_id = (state.get("item_lineage") or {}).get(item_id, {}).get("root_item_id", item_id)
                slots = [slot for slot in (state.get("blueprint") or {}).get("slots") or []
                         if slot.get("specification_id") == item_id]
                replacement_limit = replacement_quota_limit(state)
                if (not replacement_capacity_available(state, item_id)
                        or bundle.get("root_item_id") != root_id or len(slots) != 1
                        or (bundle.get("slot") or {}).get("blueprint_cell_id") != items[item_id].get("blueprint_cell_id")
                        or not isinstance(bundle.get("replacement_number"), int)
                        or (replacement_limit is not None
                            and bundle["replacement_number"] > replacement_limit)):
                    raise ValueError("Replacement changed its fixed slot or exceeded the root quota")
            if result.get("auto_replenishment") is True:
                if (result.get("status") != "replenished"
                        or result.get("protocol") != review.PROTOCOL
                        or result.get("reviewed_item_id") != item_id
                        or result.get("reviewed_item_version") != items[item_id]["version"]
                        or result.get("source_analysis_round") != measurement
                        or not result.get("archive_ref")
                        or not isinstance(result.get("candidate"), Mapping)
                        or not isinstance(result.get("staged_design"), Mapping)):
                    raise ValueError("失败调用预算补题缺少版本绑定的暂存候选")
                candidate = result["candidate"]
                bundle = result["staged_design"]
                review.validate_content_candidate(items[item_id], candidate, bundle)
                if candidate["item_id"] in occupied:
                    raise ValueError("失败调用预算补题ID已存在，不能复用")
                occupied.add(candidate["item_id"])
                drafts[item_id] = deepcopy(candidate)
                continue
            if result["status"] not in {"ready"} or result.get("protocol") != review.PROTOCOL:
                raise ValueError(result.get("pause_reason") or "虚拟内容复审尚未完成")
            if (result.get("reviewed_item_id") != item_id
                    or result.get("reviewed_item_version") != items[item_id]["version"]
                    or result.get("source_analysis_round") != measurement
                    or len(result.get("expert_reviews") or []) not in {2, 3}
                    or len(result.get("cognitive_interviews") or []) != 6
                    or not result.get("archive_ref")):
                raise ValueError("暂存候选缺少同版本的虚拟审题和访谈证据")
            diagnosis = result.get("diagnosis") or {}
            if result.get("requires_remeasurement") is not True or diagnosis.get("decision") not in {"repair", "replace"}:
                raise ValueError("暂存修改必须通过证据诊断并要求后续重测")
            candidate = result["candidate"]
            bundle = result.get("staged_design")
            if (diagnosis["decision"] == "replace") != bool(bundle):
                raise ValueError("新候选必须与诊断授权的替换方式一致")
            review.validate_content_candidate(items[item_id], candidate, bundle)
            if bundle:
                root_id = (state.get("item_lineage") or {}).get(item_id, {}).get("root_item_id", item_id)
                original_slots = [slot for slot in (state.get("blueprint") or {}).get("slots") or []
                                  if slot.get("specification_id") == item_id]
                if (bundle.get("root_item_id") != root_id or len(original_slots) != 1
                        or (bundle.get("slot") or {}).get("blueprint_cell_id") != items[item_id].get("blueprint_cell_id")):
                    raise ValueError("新候选必须保持根项额度并替换原有固定蓝图槽位")
            if candidate["item_id"] != item_id and candidate["item_id"] in occupied:
                raise ValueError("新候选ID已存在，不能复用")
            occupied.add(candidate["item_id"])
            drafts[item_id] = deepcopy(candidate)
        except (KeyError, TypeError, ValueError) as exc:
            failures[item_id] = str(exc)
    if failures:
        return {"state_update": {
            "scenario_repair_pause": {"protocol": review.PROTOCOL, "recovery_version": 2,
                                      "batch_id": batch_id, "failures": failures, "limit_reached": False,
                                      "message": "虚拟内容复审暂停；原题库及作答未更新，完成步骤已归档。"},
            "scenario_repair_progress": progress_by_id, "scenario_repair_staged": drafts,
            "virtual_content_review_history": _history(state, progress_by_id),
            "items_to_revise": [{**entry, "content_review_archive_ref":
                                 progress_by_id[str(entry["item_id"])].get("archive_ref")
                                 or entry.get("content_review_archive_ref"),
                                 "queue_status": progress_by_id[str(entry["item_id"])]["status"]} for entry in entries],
            "items_to_regenerate": [],
            "psychometric_repair_batch_summary": {"batch_id": batch_id, "status": "paused",
                                                  "batch_total": len(entries), "completed_count": len(drafts),
                                                  "deferred_count": len(deferred), "failed_count": len(failures)},
        }, "summary": "虚拟内容复审失败，整批不提交并保留检查点", "repair_attempt_count": 0}

    lineage = deepcopy(state.get("item_lineage") or {})
    blueprint = deepcopy(state.get("blueprint") or {})
    specifications = deepcopy(state.get("item_specifications") or [])
    skeletons = deepcopy(state.get("item_skeletons") or {})
    skeleton_history = deepcopy(state.get("skeleton_review_history") or {})
    generations = set(state.get("item_generation_completed_ids") or [])
    dispositions = deepcopy(state.get("item_final_dispositions") or {})
    content_evidence = deepcopy(state.get("item_content_evidence") or {})
    rounds = deepcopy(state.get("psychometric_repair_rounds") or {})
    item_history = deepcopy(state.get("item_history") or {})
    repair_history = deepcopy(state.get("psychometric_repair_history") or [])
    profiles = deepcopy(state.get("item_pattern_profiles") or {})
    for entry in entries:
        old_id = str(entry["item_id"])
        result = progress_by_id[old_id]
        base = items[old_id]
        root_id = (lineage.get(old_id) or {}).get("root_item_id", old_id)
        if old_id in deferred:
            dispositions[old_id] = {"status": "deferred_decision", "item_version": base["version"],
                                    "source_analysis_round": measurement, "diagnosis_status": "insufficient_content_evidence",
                                    "reason": result["diagnosis"]["summary"]}
            content_evidence[old_id] = {"status": "investigated_deferred", "item_version": base["version"],
                                       "last_reviewed_item_id": old_id, "last_reviewed_version": base["version"],
                                       "source_analysis_round": measurement, "archive_ref": result["archive_ref"],
                                       "evidence_scope": review.EVIDENCE_SCOPE}
            repair_history.append({"event": "virtual_content_review_deferred", "item_id": old_id,
                                   "source_analysis_round": measurement, "item_version": base["version"],
                                   "content_review_archive_ref": result["archive_ref"],
                                   "reason": result["diagnosis"]["summary"], "recorded_at": utc_timestamp()})
            continue
        candidate = drafts[old_id]
        bundle = result.get("staged_design")
        candidate["version"] = 1 if bundle else int(base["version"]) + 1
        new_id = candidate["item_id"]
        completed = max(int(rounds.get(root_id) or 0), int(rounds.get(old_id) or 0)) + 1
        rounds[root_id] = rounds[old_id] = rounds[new_id] = completed
        dispositions.pop(old_id, None)
        if bundle:
            for index, slot in enumerate(blueprint.get("slots") or []):
                if slot.get("specification_id") == old_id:
                    blueprint["slots"][index] = deepcopy(bundle["slot"])
                    break
            specifications = [row for row in specifications if row.get("specification_id") != old_id]
            specifications.append(deepcopy(bundle["specification"]))
            skeletons.pop(old_id, None)
            skeletons[new_id] = deepcopy(bundle["skeleton"])
            skeleton_history[new_id] = [{"mode": "evidence_authorized_replacement", "skeleton": deepcopy(bundle["skeleton"]),
                                         "content_review": None, "content_status": "not_reassessed"}]
            generations.discard(old_id)
            generations.add(new_id)
            lineage[old_id] = {**lineage.get(old_id, {}), "root_item_id": root_id,
                               "status": "replaced", "replaced_by_item_id": new_id}
            lineage[new_id] = {"root_item_id": root_id, "replaces_item_id": old_id,
                               "replacement_number": bundle["replacement_number"]}
            dispositions[old_id] = {"status": "replaced", "item_version": base["version"], "replacement_item_id": new_id}
        content_evidence[new_id] = {"status": "pending_remeasurement", "item_version": candidate["version"],
                                   "last_reviewed_item_id": old_id, "last_reviewed_version": base["version"],
                                   "source_analysis_round": measurement, "archive_ref": result["archive_ref"],
                                   "post_edit_comprehensibility": "not_reassessed", "evidence_scope": review.EVIDENCE_SCOPE}
        profiles[new_id] = build_item_pattern_profile(candidate, next(
            (row for row in specifications if row.get("specification_id") == new_id), None))
        item_history.setdefault(new_id, []).append({"event": "reconstructed" if bundle else "revised",
                                                   "item": deepcopy(candidate), "previous_item_id": old_id,
                                                   "previous_version": base["version"], "source_analysis_round": measurement,
                                                   "content_review_archive_ref": result["archive_ref"], "recorded_at": utc_timestamp()})
        repair_history.append({"event": "psychometric_item_repaired", "item_id": old_id,
                               "candidate_item_id": new_id, "revision_round": completed,
                               "source_analysis_round": measurement, "new_item_version": candidate["version"],
                               "baseline_item": deepcopy(base), "baseline_metrics": deepcopy(entry.get("baseline_metrics") or {}),
                               "repair_protocol": review.PROTOCOL, "resolution": "replaced" if bundle else "repaired",
                               "atomic_repair_advice": deepcopy(result["diagnosis"]),
                               "content_review_archive_ref": result["archive_ref"], "recorded_at": utc_timestamp()})

    update = {"scenario_repair_pause": None, "scenario_repair_staged": {}, "scenario_repair_progress": progress_by_id,
              "virtual_content_review_history": _history(state, progress_by_id, committed=True, candidates=drafts),
              "item_content_evidence": content_evidence, "item_final_dispositions": dispositions,
              "items_to_revise": [], "items_to_regenerate": [], "items_deferred_for_revision": list(deferred),
              "psychometric_repair_confirmation": None, "active_psychometric_repair": None,
              "selection_results": None, "psychometric_repair_history": repair_history,
              "psychometric_repair_batch_summary": {"batch_id": batch_id, "status": "completed",
                                                    "batch_total": len(entries), "completed_count": len(drafts),
                                                    "deferred_count": len(deferred), "failed_count": 0}}
    deferred_replacement_committed = any(
        result.get("auto_replenishment") is True
        and result.get("replenishment_reason") == "deferred_decision"
        for result in progress_by_id.values()
        if isinstance(result, Mapping)
    )
    if drafts:
        if fixed_iteration:
            update["facet_iteration_state"] = mark_committed_repairs(state, {
                item_id: {
                    "facet_id": str(items[item_id]["target_dimension_id"]),
                    "old_version": items[item_id]["version"],
                    "new_item_id": candidate["item_id"], "new_version": candidate["version"],
                    "reason": (progress_by_id[item_id].get("replenishment_reason")
                               or "evidence_authorized_facet_local_change"),
                    "evidence_archive_ref": progress_by_id[item_id].get("archive_ref"),
                    "authorized_diagnosis": deepcopy(progress_by_id[item_id].get("diagnosis")),
                    "previous_response_data_ref": state.get("virtual_response_data_ref"),
                    "was_qualified_locked": (state.get("locked_retained_item_versions") or {}).get(item_id) == items[item_id]["version"],
                }
                for item_id, candidate in drafts.items()
            })
        update.update(virtual_content_review_stop_reason=None,
                      deferred_replacement_measurement_pending=deferred_replacement_committed,
                      item_pool=[deepcopy(drafts.get(identifier, original)) for identifier, original in items.items()],
                      blueprint=blueprint, item_specifications=specifications, item_skeletons=skeletons,
                      skeleton_review_history=skeleton_history, item_lineage=lineage,
                      item_generation_completed_ids=sorted(generations), psychometric_repair_rounds=rounds,
                      item_history=item_history, item_pattern_profiles=profiles,
                      previous_virtual_response_data_ref=state.get("virtual_response_data_ref"),
                      current_item=None, current_item_review=None, current_item_specification=None,
                      current_blueprint_cell=None, candidate_bank_audit=None, selected_items=[], reserve_items=[],
                      selection_reasons={}, blueprint_coverage=None, assembled_test=None, test_review_result=None,
                      final_test=None, item_database_ref=None, technical_report=None, virtual_respondent_report=None,
                      virtual_response_data_ref=None, virtual_response_summary=None,
                      virtual_response_item_bank_id=None, virtual_response_item_bank_version=None,
                      item_statistics={}, psychometric_round_result=None, test_statistics=None,
                      factor_results=None, irt_results=None, dif_results=None, best_assembly_candidate=None)
    else:
        update["virtual_content_review_stop_reason"] = "no_evidence_supported_edit"
    return {"state_update": update, "repair_attempt_count": 0,
            "summary": f"虚拟内容复审完成：提交 {len(drafts)} 道修改，暂缓 {len(deferred)} 道；修改后统一重测"}
