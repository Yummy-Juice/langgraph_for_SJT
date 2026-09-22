"""Independent, form-scoped evaluation; never invokes a writer or selector."""
from copy import deepcopy
from pathlib import Path
import shutil

from sjt_system.authoring.bank import build_item_bank_freeze_update
from sjt_system.evaluation.simulation import run_virtual_response_simulation, VIRTUAL_RESPONSE_PROMPT_VERSION
from sjt_system.evaluation.psychometrics import run_psychometric_analysis
from sjt_system.evaluation.form_metrics import build_provisional_form_metrics, form_quality_summary
from sjt_system.evaluation.round_results import build_psychometric_round_result
from sjt_system.runtime.output_paths import output_scope
from sjt_system.runtime.telemetry import run_context
from sjt_system.state import create_initial_state

from .config import fingerprint
from .storage import write_csv


def evaluation_state(store, key):
    form = store.read(f"{key}/form.json")
    if not form or fingerprint(form["items"]) != form["fingerprint"]:
        raise ValueError(f"问卷快照缺失或已被更改：{key}")
    sample = store.read("participants/evaluation.json")
    state = create_initial_state("独立冻结问卷评估")
    state.update({"run_id": store.root.name + "-evaluation",
                  "construct_profile": store.read("shared/construct_profile.json"),
                  "item_pool": deepcopy(form["items"]),
                  "virtual_respondents": deepcopy(sample["respondents"]),
                  "virtual_sample_config": deepcopy(sample["config"])})
    state["virtual_sample_config"]["model_id"] = store.config.evaluation_model_id
    state["virtual_sample_config"]["experiment_protocol"] = {
        "model": store.config.evaluation_model_id, "sample": "evaluation", "temperature": 1.0}
    state.update(build_item_bank_freeze_update(state))
    return state


def flatten_items(state, key):
    clean = deepcopy(state)
    clean["locked_retained_item_versions"] = {}
    clean["item_final_dispositions"] = {}
    statistics = clean.setdefault("item_statistics", {})
    for item in clean.get("frozen_item_bank", []):
        statistics.setdefault(item["item_id"], {"quality_evaluation": {}, "qualification": {}})
    result = build_psychometric_round_result(clean)
    names = {"citc_pass": "citc", "target_rho_pass": "target_rho",
             "same_domain_vts_pass": "same_domain_vts", "cross_domain_vts_pass": "cross_domain_vts"}
    rows = []
    for item in result["items"]:
        row = {"method": key.split("/")[0], "round": key.split("/")[1],
               "item_id": item["item_id"], "item_version": item["item_version"],
               "item_content_hash": fingerprint(item["item"]),
               "facet": item["item"].get("target_dimension_id"),
               "qualified": item["qualified"] and not item["failed_thresholds"],
               "failed_gates": item["failed_thresholds"],
               "unavailable_reason": "缺少作答或分数无变异；详见item_statistics.json" if any(not g["estimable"] for g in item["gates"]) else None}
        for gate in item["gates"]:
            name = names[gate["gate_id"]]
            row.update({name: gate["value"], name + "_threshold": gate["threshold"],
                        name + "_passes": gate["passes"], name + "_estimable": gate["estimable"]})
        for name in ("same_domain", "cross_domain"):
            contaminant = item["max_contaminants"][name]
            row[name + "_facet"] = contaminant.get("dimension_id")
            row[name + "_rho"] = contaminant.get("rho")
            row[name + "_all_correlations"] = contaminant.get("non_target_spearman")
        rows.append(row)
    return rows


def _copy_tree_rebased(source, target, *, skip_files=()):
    """Copy self-contained response artifacts, relocating embedded path references."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        return
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*skip_files))
    replacements = ((str(source), str(target)), (source.as_posix(), target.as_posix()))
    def replace(value):
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        return value
    import json
    from .storage import write_json
    for file in target.rglob("*.json"):
        write_json(file, replace(json.loads(file.read_text(encoding="utf-8"))))


async def evaluate_form(store, key, base_model=None):
    state = evaluation_state(store, key)
    form = store.read(f"{key}/form.json")
    binding = fingerprint({"form": form["fingerprint"], "sample": state["virtual_sample_config"],
                           "respondents": state["virtual_respondents"],
                           "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION})
    old = store.read(f"{key}/evaluation/metrics.json")
    if old and old.get("status") == "complete":
        if old.get("binding") != binding:
            raise ValueError("已评估数据与当前题目或配置不一致")
        return old
    # Exact B/baseline aliases can share a completed evaluation, not development scores.
    reused_from = None
    for other in store.forms():
        if other == key:
            continue
        metrics = store.read(f"{other}/evaluation/metrics.json", {})
        if metrics.get("status") == "complete" and metrics.get("binding") == binding:
            store.write(f"{key}/evaluation/cost.json", {"wall_seconds": 0, "reused_from": other})
            _copy_tree_rebased(store.path(f"{other}/evaluation"), store.path(f"{key}/evaluation"),
                               skip_files=("metrics.json", "cost.json"))
            reused_from = other
            state = store.read(f"{key}/evaluation/checkpoint.json")
            break
    if reused_from is None:
        # Reuse only previous independent evaluation records when the core cache
        # validator accepts config, model, item content, scores and administration.
        for other in reversed(store.forms()):
            if other == key:
                continue
            previous = store.read(f"{other}/evaluation/checkpoint.json", {})
            completed = store.read(f"{other}/evaluation/metrics.json", {})
            if completed.get("status") == "complete" and previous.get("virtual_response_data_ref"):
                state["previous_virtual_response_data_ref"] = previous["virtual_response_data_ref"]
                break
        if base_model is None:
            from sjt_system.agent.client import get_model
            base_model = get_model(store.config.evaluation_model_id)
        with store.timer(f"{key}/evaluation"), output_scope(store.path(f"{key}/evaluation")), run_context(state["run_id"]):
            simulation = await run_virtual_response_simulation(state, base_model=base_model)
            state.update(simulation["state_update"])
            store.write(f"{key}/evaluation/checkpoint.json", state)
    # Recompute all item statistics in the current form, even when raw responses were reused.
    analysis = run_psychometric_analysis(state)
    state.update(analysis["state_update"])
    form_metrics = build_provisional_form_metrics(state, [i["item_id"] for i in state["frozen_item_bank"]])
    rows = flatten_items(state, key)
    store.write(f"{key}/evaluation/checkpoint.json", state)
    store.write(f"{key}/evaluation/item_statistics.json", state["item_statistics"])
    store.write(f"{key}/evaluation/items.json", rows)
    write_csv(store.path(f"{key}/evaluation/item_metrics.csv"), rows)
    outputs = state["test_statistics"]["output_files"]
    for name, filename in (("option_statistics", "option_statistics.csv"),
                            ("option_choice_diagnostics", "option_choice_diagnostics.json"),
                            ("scored_matched_condition_sjt_responses", "scored_responses.csv"),
                            ("analysis_manifest", "analysis_manifest.json")):
        source = Path(outputs[name])
        target = store.path(f"{key}/evaluation/{filename}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    # Export explicit respondent x item matrices as well as the long score table.
    import pandas as pd
    item_order = [i["item_id"] for i in state["frozen_item_bank"]]
    for output_name, group_column, prefix in (
        ("scored_matched_condition_sjt_responses", "condition_id", ""),
        ("scored_target_form_retest_sjt_responses", "administration_id", "target_retest_"),
    ):
        frame = pd.read_csv(outputs[output_name])
        for group, scores in frame.groupby(group_column):
            wide = scores.pivot(index="matched_subject_id", columns="item_id", values="score").reindex(columns=item_order)
            wide = wide.where(pd.notna(wide), None).reset_index()
            write_csv(store.path(f"{key}/evaluation/score_matrices/{prefix}{group}.csv"), wide.to_dict("records"))
    qualified = sum(r["qualified"] for r in rows)
    metrics = {"status": "complete", "binding": binding, "reused_from": reused_from,
               "item_count": len(rows), "qualified_item_count": qualified,
               "qualification_rate": qualified / len(rows) if rows else None,
               "quality": form_quality_summary(form_metrics), "form_metrics": form_metrics,
               "formula_version": state["test_statistics"].get("formula_version"),
               "thresholds": state["test_statistics"].get("qualification_criteria")}
    if reused_from:
        store.write(f"{key}/evaluation/cost.json", {"wall_seconds": 0, "reused_from": reused_from,
                                                    "additional_model_calls": 0})
    store.write(f"{key}/evaluation/metrics.json", metrics)
    return metrics
