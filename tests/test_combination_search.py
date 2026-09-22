"""Offline equivalence, provenance, missingness and selection contracts."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from experiments.system_comparison.combination_data import (
    SearchData, content_hash, inside, load_search_data,
)
from experiments.system_comparison.combination_search import (
    BatchMetrics, SearchConfig, blueprint_groups, combination_indices,
    ranked, row_correlations, run_combination_search,
)
from sjt_system.evaluation.form_metrics import (
    _cronbach_alpha, _cross_validated_target_recovery, _extreme_group_effect,
    _icc_absolute_agreement_single, _construct_selectivity_value,
)


def item(j):
    return {"item_id": f"item-{j}", "version": 1, "blueprint_cell_id": f"cell-{j // 2}",
            "target_dimension_id": "extraversion_gregariousness", "scenario": f"情境-{j}",
            "response_instruction": "你会怎么做？", "scoring_key": dict(zip("ABCD", (1, 2, 3, 4))),
            "response_options": [{"option_id": k, "text": f"行为{k}-{j}"} for k in "ABCD"]}


def make_data(tmp_path):
    rng = np.random.default_rng(948)
    n, m = 30, 6
    subjects = [f"evaluation-matched-{i:04d}" for i in range(n)]
    conditions = ["target", "same_domain__warmth", "cross_domain__anxiety"]
    x = {c: rng.integers(1, 5, (n, m)).astype(float) for c in conditions}
    active = {c: rng.normal(50, 15, n) for c in conditions}
    bank = [item(j) for j in range(m)]
    blueprint = {"cells": [{"cell_id": f"cell-{j}", "planned_retention_count": 1,
                            "facet_id": "extraversion_gregariousness"} for j in range(m // 2)]}
    reference = pd.DataFrame(rng.integers(15, 55, (n, 5)), index=subjects, columns=list("ENOAC"))
    reference["N"] = 36
    return SearchData(tmp_path, bank, blueprint, subjects, x, active, x["target"].copy(), reference,
                      {"item_sources": {i["item_id"]: {} for i in bank}})


def test_vector_metrics_match_existing_per_form_calculations(tmp_path):
    data = make_data(tmp_path)
    groups, _ = blueprint_groups(data, available_only=False)
    indices = list(combination_indices(groups))
    actual = BatchMetrics(data).calculate(indices)
    for row, selected in enumerate(indices):
        x = pd.DataFrame(data.matrices["target"][:, selected], index=data.subjects)
        y = pd.Series(data.active["target"], index=data.subjects)
        total = x.mean(axis=1)
        r2 = _cross_validated_target_recovery(x, y)["cross_validated_r2"]
        t = _extreme_group_effect(total, y, denominator_sd=total.std(ddof=1))["standardized_effect"]
        leak = []
        for c in data.matrices:
            if c != "target":
                scores = pd.Series(data.matrices[c][:, selected].mean(axis=1), index=data.subjects)
                leak.append(abs(_extreme_group_effect(scores, pd.Series(data.active[c], index=data.subjects),
                                                     denominator_sd=total.std(ddof=1))["standardized_effect"]))
        s = _construct_selectivity_value(t, max(leak))
        assert actual["alpha"][row] == pytest.approx(_cronbach_alpha(x), abs=1e-12)
        repeat = pd.DataFrame({"1": total, "2": data.retest[:, selected].mean(axis=1)}, index=data.subjects)
        assert actual["icc"][row] == pytest.approx(_icc_absolute_agreement_single(repeat), abs=1e-12)
        assert actual["target_recovery_r2"][row] == pytest.approx(r2, abs=1e-12)
        assert actual["construct_selectivity"][row] == pytest.approx(s, abs=1e-12)
        assert actual["q"][row] == pytest.approx(np.sqrt(np.clip(r2, 0, 1) * s), abs=1e-12)
        assert actual["neo_E_rho"][row] == pytest.approx(total.corr(data.reference.E, method="spearman"))
        assert np.isnan(actual["neo_N_rho"][row])


def test_missing_retest_and_constant_scores_are_not_zero_or_pass(tmp_path):
    data = make_data(tmp_path)
    data.retest[:, 0] = np.nan
    actual = BatchMetrics(data).calculate([[0, 2, 4]])
    assert np.isnan(actual["icc"][0])
    data.matrices["target"][:] = 2
    actual = BatchMetrics(data).calculate([[0, 2, 4]])
    assert np.isnan(actual["alpha"][0])
    assert np.isnan(actual["q"][0])
    assert np.isnan(actual["neo_E_rho"][0])


def test_single_item_alpha_missing_and_negative_r2_not_clipped(tmp_path):
    data = make_data(tmp_path)
    actual = BatchMetrics(data).calculate([[0]])
    assert np.isnan(actual["alpha"][0])
    assert actual["target_recovery_r2"][0] < 0
    assert actual["q"][0] == 0


def test_spearman_ties_constant_and_nonfinite():
    x = np.array([[1, 1, 2, 3], [2, 2, 2, 2], [1, 2, np.nan, 4]])
    actual = row_correlations(x, np.array([4, 4, 2, 1]))
    assert actual[0] == pytest.approx(-1)
    assert np.isnan(actual[1:]).all()


def test_ranking_uses_actual_neo_not_q():
    frame = pd.DataFrame({"eligible": [True, True, False], "combination_id": [0, 1, 2],
                          "neo_E_rho": [.91, .94, .99], "q": [.8, .6, .9]})
    assert ranked(frame, "neo", SearchConfig()).iloc[0].combination_id == 1
    assert ranked(frame, "q", SearchConfig()).iloc[0].combination_id == 0


def test_blueprint_65536_and_partial_space(tmp_path):
    data = make_data(tmp_path)
    data.bank = [item(j) for j in range(32)]
    data.blueprint["cells"] = [{"cell_id": f"cell-{j}", "planned_retention_count": 1} for j in range(16)]
    data.audit["item_sources"] = {i["item_id"]: {} for i in data.bank}
    groups, _ = blueprint_groups(data, available_only=True)
    assert sum(1 for _ in combination_indices(groups)) == 65536
    for j in range(9):
        del data.audit["item_sources"][f"item-{j * 2 + 1}"]
    groups, _ = blueprint_groups(data, available_only=True)
    assert sum(1 for _ in combination_indices(groups)) == 128


def write_fixture(root):
    data = make_data(root)
    def write(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    respondents = [{"condition_id": c, "matched_subject_id": s, "active_dimension_id": c,
                    "score_values": {c: float(data.active[c][r])}}
                   for c in data.matrices for r, s in enumerate(data.subjects)]
    write("shared/initial_bank.json", data.bank)
    write("shared/blueprint.json", data.blueprint)
    write("participants/evaluation.json", {"respondents": respondents, "config": {"source_sha256": "sample-hash"}})
    folder = root / "B/round_01/evaluation"
    write("B/round_01/form.json", {"items": data.bank})
    write("B/round_01/evaluation/checkpoint.json", {
        "virtual_response_data_ref": "B/round_01/evaluation/manifest.json",
        "test_statistics": {"output_files": {"scored_target_form_retest_sjt_responses": "B/round_01/evaluation/retest.csv"}}})
    write("B/round_01/evaluation/manifest.json", {"status": "completed", "source_sha256": "sample-hash",
                                               "model_id": "offline-model", "prompt_version": "v1"})
    records = [{"item_id": i["item_id"], "item_version": i["version"], "condition_id": c,
                "matched_subject_id": s, "active_score": data.active[c][r], "score": data.matrices[c][r, j],
                "selected_option_id": "ABCD"[int(data.matrices[c][r, j]) - 1]}
               for c in data.matrices for r, s in enumerate(data.subjects) for j, i in enumerate(data.bank)]
    scores = pd.DataFrame(records)
    scores.to_csv(folder / "scored_responses.csv", index=False)
    scores[scores.condition_id == "target"].assign(administration_id=2).to_csv(folder / "retest.csv", index=False)
    (root / "reference/neo_ffi").mkdir(parents=True)
    data.reference.rename_axis("matched_subject_id").reset_index().to_csv(root / "reference/neo_ffi/scores.csv", index=False)
    write("reference/neo_ffi/manifest.json", {"status": "completed", "model_id": "offline-model"})
    return data


def test_frozen_versions_and_namespaces_are_validated(tmp_path):
    write_fixture(tmp_path)
    data = load_search_data(tmp_path)
    assert data.audit["available_count"] == 6
    assert data.audit["reference"]["unavailable_domains"] == {"N": "NEO维度分数无变异"}
    path = tmp_path / "B/round_01/evaluation/scored_responses.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "matched_subject_id"] = "development-matched-0000"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="混用"):
        load_search_data(tmp_path)


@pytest.mark.parametrize("field,value,message", [("item_version", 2, "版本"),
                                                   ("score", 99, "计分"), ("active_score", 1000, "人格")])
def test_reject_corrupted_score_rows(tmp_path, field, value, message):
    write_fixture(tmp_path)
    path = tmp_path / "B/round_01/evaluation/scored_responses.csv"
    frame = pd.read_csv(path)
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_search_data(tmp_path)


def test_same_id_changed_wording_is_not_reused(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / "shared/initial_bank.json"
    bank = json.loads(path.read_text(encoding="utf-8"))
    bank[0]["scenario"] = "改过的内容"
    path.write_text(json.dumps(bank), encoding="utf-8")
    data = load_search_data(tmp_path)
    assert data.audit["available_count"] == 5
    assert data.audit["missing_item_ids"] == ["item-0"]


def test_partial_data_requires_opt_in_and_exports_honest_report(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / "B/round_01/evaluation/scored_responses.csv"
    frame = pd.read_csv(path)
    frame = frame[~((frame.item_id == "item-0") & (frame.condition_id == "target"))]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="allow-partial-bank"):
        run_combination_search(tmp_path, progress=lambda _: None)
    folder, summary = run_combination_search(tmp_path, SearchConfig(allow_partial_bank=True, alpha_min=0), progress=lambda _: None)
    assert summary["evaluated_combinations"] == 4
    assert summary["theoretical_combinations"] == 8
    csv = pd.read_csv(folder / "combinations.csv")
    assert csv.neo_N_rho.isna().all()
    assert csv.max_abs_non_target_rho.isna().all()
    assert not csv.discriminant_complete.any()
    assert summary["additional_tokens"] == 0
    assert "不是独立验证" in (folder / "report.html").read_text(encoding="utf-8")
    assert (folder / "data_audit.json").exists()


def test_strict_discrimination_does_not_allow_missing_domains(tmp_path):
    write_fixture(tmp_path)
    folder, summary = run_combination_search(tmp_path, SearchConfig(alpha_min=0, max_non_target_rho=1), progress=lambda _: None)
    assert summary["status"] == "no_eligible_form"
    assert not (folder / "best_neo/form.json").exists()


def test_search_does_not_overwrite_inputs_and_is_reproducible(tmp_path):
    write_fixture(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    folder, first = run_combination_search(tmp_path, SearchConfig(alpha_min=0, batch_size=3), progress=lambda _: None)
    other, second = run_combination_search(tmp_path, SearchConfig(alpha_min=0, batch_size=8), progress=lambda _: None)
    assert folder != other
    for key, content in before.items():
        assert (tmp_path / key).read_bytes() == content
    assert first["leaders"]["best_neo"]["combination_id"] == second["leaders"]["best_neo"]["combination_id"]
    assert first["leaders"]["best_q"]["q"] == pytest.approx(second["leaders"]["best_q"]["q"], abs=1e-12)


def test_path_escape_and_content_key_changes(tmp_path):
    with pytest.raises(ValueError, match="越出"):
        inside(tmp_path, "../secrets.json")
    original = item(0)
    changed = deepcopy(original)
    changed["scoring_key"]["A"] = 4
    assert content_hash(original) != content_hash(changed)


def test_cli_combine_does_not_open_main_experiment_store(tmp_path, monkeypatch):
    from experiments.system_comparison import __main__ as cli
    write_fixture(tmp_path)
    def forbidden_store(*args, **kwargs):
        raise AssertionError("Offline combinations must not open the main experiment runner/store")
    monkeypatch.setattr(cli, "ExperimentStore", forbidden_store)
    assert cli.main(["combine", "--experiment", str(tmp_path), "--alpha-min", "0"]) == 0


def test_duplicate_scenarios_are_excluded(tmp_path):
    from experiments.system_comparison.combination_search import duplicate_pairs
    bank = [item(j) for j in range(4)]
    bank[3]["scenario"] = " 情境-0 \n"
    assert duplicate_pairs(bank) == [(0, 3)]


def test_search_cap_fails_before_metric_calculation(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    def forbidden_calculate(*args, **kwargs):
        raise AssertionError("Search limit must precede numerical work")
    monkeypatch.setattr(BatchMetrics, "calculate", forbidden_calculate)
    with pytest.raises(ValueError, match="超过安全上限"):
        run_combination_search(tmp_path, SearchConfig(max_combinations=7), progress=lambda _: None)
