import asyncio
from argparse import Namespace
import json
from types import SimpleNamespace

import pytest

from sjt_system.runtime.concurrency import UnlimitedConcurrency


def _patch_model_inputs(monkeypatch, runner, loader_name):
    monkeypatch.setattr(runner, "get_model", lambda: SimpleNamespace(model_name="fake"))
    monkeypatch.setattr(
        runner,
        "with_compatible_structured_output",
        lambda model, _schema: (model, "fake"),
    )
    monkeypatch.setattr(runner, "get_model_request_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(
        runner,
        "generate_score_respondent_refs",
        lambda sample_size, specs, seed: (
            [
                {
                    "respondent_id": f"score-{index:04d}",
                    "score_values": {specs[0]["dimension_id"]: 50.0},
                }
                for index in range(1, sample_size + 1)
            ],
            {},
        ),
    )
    monkeypatch.setattr(runner, loader_name, lambda: ([], {}))


@pytest.mark.parametrize(
    ("module_name", "group_arg", "group_field", "summary_field", "groups"),
    [
        (
            "tools.run_mussel_conductance_check",
            "groups",
            "facet_id",
            "facet_id",
            ["extraversion_gregariousness", "openness_ideas"],
        ),
        (
            "tools.run_cibol_conductance_check",
            "domains",
            "domain_id",
            "domain_id",
            ["extraversion", "openness"],
        ),
    ],
)
def test_conductance_groups_fan_out_and_summaries_keep_input_order(
    monkeypatch, tmp_path, module_name, group_arg, group_field, summary_field, groups
):
    import importlib

    runner = importlib.import_module(module_name)
    loader_name = "load_mussel_items" if group_arg == "groups" else "load_cibol_items"
    _patch_model_inputs(monkeypatch, runner, loader_name)

    all_started = asyncio.Event()
    started = []
    completed = []

    async def answer_group(**kwargs):
        group_id = kwargs[group_field]
        started.append(group_id)
        assert isinstance(kwargs["semaphore"], UnlimitedConcurrency)
        if len(started) == len(groups):
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        await asyncio.sleep((len(groups) - groups.index(group_id)) * 0.01)
        completed.append(group_id)
        return {
            summary_field: group_id,
            "respondent_count": len(kwargs["refs"]),
            "completed_calls": 1,
            "error_count": 0,
            "sample_errors": [],
        }

    monkeypatch.setattr(
        runner,
        "_answer_mussel_group" if group_arg == "groups" else "_answer_cibol_group",
        answer_group,
    )
    values = {
        "output_root": str(tmp_path),
        "sample_size": 1,
        "concurrency": 1,
        "max_retries": 0,
        "seed": 3,
        group_arg: groups,
    }

    asyncio.run(runner._main(Namespace(**values)))

    assert set(started) == set(groups)
    assert completed == list(reversed(groups))
    output_dir = next(tmp_path.iterdir())
    summaries = json.loads((output_dir / "group_summaries.json").read_text(encoding="utf-8"))
    assert [row[summary_field] for row in summaries] == groups


def test_same_domain_arms_fan_out_and_summaries_keep_mapping_order(monkeypatch, tmp_path):
    import tools.run_same_domain_arms as runner

    _patch_model_inputs(monkeypatch, runner, "load_cibol_items")
    pairs = list(runner.CIBOL_SAME_DOMAIN.items())
    facets = [same_facet for _target, same_facet in pairs]
    all_started = asyncio.Event()
    started = []
    completed = []

    async def answer_group(**kwargs):
        facet_id = kwargs["facet_id"]
        started.append(facet_id)
        assert isinstance(kwargs["semaphore"], UnlimitedConcurrency)
        if len(started) == len(facets):
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        await asyncio.sleep((len(facets) - facets.index(facet_id)) * 0.01)
        completed.append(facet_id)
        return {
            "facet_id": facet_id,
            "respondent_count": len(kwargs["refs"]),
            "completed_calls": 1,
            "error_count": 0,
            "sample_errors": [],
        }

    monkeypatch.setattr(runner, "_answer_mussel_group", answer_group)
    asyncio.run(
        runner._main(
            Namespace(
                bank="cibol",
                dir=str(tmp_path),
                sample_size=1,
                concurrency=1,
                max_retries=0,
                seed=3,
            )
        )
    )

    assert set(started) == set(facets)
    assert completed == list(reversed(facets))
    summaries = json.loads(
        (tmp_path / "same_domain_summaries.json").read_text(encoding="utf-8")
    )
    assert [row["target"] for row in summaries] == [target for target, _facet in pairs]
    assert [row["same_domain_facet"] for row in summaries] == facets
