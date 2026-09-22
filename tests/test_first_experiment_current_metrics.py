from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from experiments.system_comparison.first_experiment_current_metrics import (
    METHOD_PATHS,
    _extreme_group_masks,
    compute_form_metrics,
    load_experiment,
    paired_bootstrap,
    plot_metrics,
    run_analysis,
)


def test_extreme_group_masks_select_exact_33_with_ties() -> None:
    criterion = np.array([30.0] * 26 + [36.0] * 30 + [41.0] * 33 + [50.0] * 3)

    low, high = _extreme_group_masks(criterion, group_size=33)

    assert low.sum() == 33
    assert high.sum() == 33
    assert not np.any(low & high)
    assert np.all(criterion[low] <= 36.0)
    assert np.all(criterion[high] >= 41.0)


def _synthetic_form() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    target = np.arange(12, dtype=float)
    first = np.column_stack(
        [
            target,
            target * 0.8 + np.array([0, 1] * 6, dtype=float),
            target * 1.2 + np.array([1, 0, 0] * 4, dtype=float),
        ]
    )
    references = pd.DataFrame(
        {
            "E": first.mean(axis=1),
            "N": [0, 3, 1, 2] * 3,
            "O": [1, 0, 3, 2] * 3,
            "A": [2, 1, 0, 3] * 3,
            "C": [3, 1, 2, 0] * 3,
        }
    )
    return first, first.copy(), references


def test_compute_form_metrics_returns_current_five_metrics() -> None:
    first, retest, references = _synthetic_form()

    result = compute_form_metrics(first, retest, references, target_column="E")

    assert set(result) == {
        "cronbach_alpha",
        "icc_a1",
        "target_spearman",
        "delta_min",
        "target_hedges_g",
        "non_target_spearman",
    }
    assert np.isfinite(result["cronbach_alpha"])
    assert np.isclose(result["icc_a1"], 1.0)
    assert np.isclose(result["target_spearman"], 1.0)
    assert np.isclose(
        result["delta_min"],
        result["target_spearman"]
        - max(abs(value) for value in result["non_target_spearman"].values()),
    )
    assert np.isfinite(result["target_hedges_g"])


def test_compute_form_metrics_does_not_turn_unavailable_values_into_zero() -> None:
    first, retest, references = _synthetic_form()
    references.loc[:, "E"] = 50.0

    result = compute_form_metrics(first, retest, references, target_column="E")

    assert np.isnan(result["target_spearman"])
    assert np.isnan(result["delta_min"])
    assert np.isnan(result["target_hedges_g"])


def _write_experiment_fixture(root: Path) -> None:
    subject_ids = [f"s{index:02d}" for index in range(12)]
    first, retest, references = _synthetic_form()
    for method_path in METHOD_PATHS.values():
        score_dir = root / method_path / "evaluation" / "score_matrices"
        score_dir.mkdir(parents=True, exist_ok=True)
        first_frame = pd.DataFrame(first, columns=["i1", "i2", "i3"])
        first_frame.insert(0, "matched_subject_id", subject_ids)
        retest_frame = pd.DataFrame(retest, columns=["i1", "i2", "i3"])
        retest_frame.insert(0, "matched_subject_id", subject_ids)
        first_frame.to_csv(score_dir / "target.csv", index=False)
        retest_frame.to_csv(score_dir / "target_retest_2.csv", index=False)
    reference_dir = root / "reference" / "neo_ffi"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_frame = references.copy()
    reference_frame.insert(0, "matched_subject_id", subject_ids)
    reference_frame.to_csv(reference_dir / "scores.csv", index=False)


def test_load_experiment_uses_frozen_final_method_paths(tmp_path: Path) -> None:
    _write_experiment_fixture(tmp_path)

    methods, references, subject_ids = load_experiment(tmp_path)

    assert METHOD_PATHS == {
        "A": "A/round_01",
        "B": "B/round_01",
        "C": "C/round_02",
    }
    assert list(methods) == ["A", "B", "C"]
    assert methods["C"]["source_path"] == "C/round_02"
    assert methods["A"]["first"].shape == (12, 3)
    assert methods["A"]["retest"].shape == (12, 3)
    assert references.shape == (12, 5)
    assert subject_ids == [f"s{index:02d}" for index in range(12)]


def test_paired_bootstrap_reuses_the_same_draw_for_identical_methods(
    tmp_path: Path,
) -> None:
    _write_experiment_fixture(tmp_path)
    methods, references, _ = load_experiment(tmp_path)

    result = paired_bootstrap(
        methods,
        references,
        replications=50,
        seed=123,
    )

    assert len(result) == 15
    assert set(result["method"]) == {"A", "B", "C"}
    assert set(result["metric"]) == {
        "cronbach_alpha",
        "icc_a1",
        "target_spearman",
        "delta_min",
        "target_hedges_g",
    }
    a = result[result["method"] == "A"].set_index("metric")
    b = result[result["method"] == "B"].set_index("metric")
    pd.testing.assert_series_equal(a["estimate"], b["estimate"], check_names=False)
    pd.testing.assert_series_equal(a["ci_lower_95"], b["ci_lower_95"], check_names=False)
    pd.testing.assert_series_equal(a["ci_upper_95"], b["ci_upper_95"], check_names=False)


def test_plot_metrics_exports_five_panel_figure(tmp_path: Path) -> None:
    rows = []
    for metric_index, metric in enumerate(
        (
            "cronbach_alpha",
            "icc_a1",
            "target_spearman",
            "delta_min",
            "target_hedges_g",
        )
    ):
        for method_index, method in enumerate(("A", "B", "C")):
            estimate = 0.2 + metric_index + method_index * 0.1
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower_95": estimate - 0.05,
                    "ci_upper_95": estimate + 0.05,
                }
            )
    results = pd.DataFrame(rows)
    output_stem = tmp_path / "current_metrics"

    plot_metrics(results, output_stem)

    for suffix in (".svg", ".png", ".pdf"):
        output = output_stem.with_suffix(suffix)
        assert output.is_file()
        assert output.stat().st_size > 0
    svg = output_stem.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Cronbach" in svg
    assert "ICC(A,1)" in svg
    assert "Hedges" in svg
    assert "Error bars" not in svg
    assert "95% percentile CIs" not in svg
    assert "uncertainty not displayed" in svg


def test_run_analysis_writes_traceable_output_bundle(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    output = tmp_path / "output"
    _write_experiment_fixture(experiment)

    result = run_analysis(experiment, output, replications=10, seed=7)

    assert len(result) == 15
    assert (output / "first_experiment_current_metrics.csv").is_file()
    assert (output / "first_experiment_current_metrics.svg").is_file()
    assert (output / "first_experiment_current_metrics.png").is_file()
    assert (output / "first_experiment_current_metrics.pdf").is_file()
    manifest = (output / "first_experiment_current_metrics_manifest.json").read_text(
        encoding="utf-8"
    )
    assert '"paired": true' in manifest
    assert '"criterion_scope": "NEO-FFI domain-level"' in manifest
    assert '"C": "C/round_02"' in manifest
