from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sjt_system.evaluation import simulation
from sjt_system.evaluation.respondents import (
    PERSONA_MODE_SCORE_PROFILE,
    build_matched_condition_sample_config,
    generate_matched_condition_respondent_refs,
    matched_condition_sample_is_current as validate_matched_condition_sample,
    matched_score_specs,
    normalize_matched_conditions,
)
from sjt_system.evaluation.demographics import generate_demographics, load_demographics_snapshot
from sjt_system.runtime.concurrency import UnlimitedConcurrency, validate_max_concurrency


@pytest.fixture(autouse=True)
def isolate_concurrency_from_sample_size_validation(monkeypatch):
    # These execution tests use one subject; full protocol validation is covered separately.
    monkeypatch.setattr(simulation, "matched_condition_sample_is_current", lambda config, refs: True)


@pytest.mark.parametrize("legacy_limit", [0, 1, 5, 20])
def test_runner_uses_unlimited_gate_for_legacy_limits(monkeypatch, legacy_limit: int) -> None:
    monkeypatch.setattr(
        simulation,
        "with_compatible_structured_output",
        lambda model, _output_type: (model, "plain_json"),
    )
    monkeypatch.setattr(
        simulation,
        "get_model_request_timeout_seconds",
        lambda: 10.0,
    )

    class Model:
        model_name = "local-fake"

    runner = simulation.VirtualResponseRunner(
        base_model=Model(),
        max_concurrency=legacy_limit,
        max_retries=0,
        request_timeout_seconds=10,
    )
    assert runner.max_concurrency == 0
    assert isinstance(runner.semaphore, UnlimitedConcurrency)

    async def run_barrier() -> int:
        active = 0
        peak = 0
        started = asyncio.Event()
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal active, peak
            async with runner.semaphore:
                async with lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 6:
                        started.set()
                await asyncio.wait_for(started.wait(), timeout=0.5)
                async with lock:
                    active -= 1

        await asyncio.wait_for(
            asyncio.gather(*(worker() for _ in range(6))),
            timeout=1.0,
        )
        return peak

    assert asyncio.run(run_barrier()) == 6


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_max_concurrency_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_max_concurrency(value)  # type: ignore[arg-type]


def test_simulation_request_attempt_budget_is_durable_and_caps_at_five_calls():
    class FailingRunnable:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, _messages):
            self.calls += 1
            raise RuntimeError("injected request failure")

    runnable = FailingRunnable()
    ledger: dict[str, dict] = {}
    writes = 0

    def persist():
        nonlocal writes
        writes += 1

    async def call():
        return await simulation._invoke_with_retry(
            runnable,
            [],
            semaphore=UnlimitedConcurrency(),
            validator=lambda result: result,
            max_retries=4,
            retry_delay_seconds=0,
            request_timeout_seconds=1,
            job_label="test request",
            attempt_ledger=ledger,
            attempt_key='{"stage":"sjt_main","respondent_id":"r1"}',
            persist_attempt_ledger=persist,
        )

    with pytest.raises(RuntimeError, match="累计 5 次"):
        asyncio.run(call())
    with pytest.raises(RuntimeError, match="耗尽 5 次持久化请求预算"):
        asyncio.run(call())
    request = next(iter(ledger.values()))
    assert runnable.calls == 5
    assert request["attempts_used"] == request["max_attempts"] == 5
    assert [row["status"] for row in request["history"]] == ["error"] * 5
    assert writes == 10
    with pytest.raises(ValueError, match="最多5次"):
        asyncio.run(simulation._invoke_with_retry(
            runnable,
            [],
            semaphore=UnlimitedConcurrency(),
            validator=lambda result: result,
            max_retries=5,
            retry_delay_seconds=0,
            request_timeout_seconds=1,
            job_label="over-budget request",
        ))


def test_qualified_response_source_overrides_old_baseline_version():
    item = {"item_id": "changed-item", "version": 3}
    state = {
        "item_final_dispositions": {
            "changed-item": {
                "status": "qualified_locked",
                "item_version": 3,
                "qualification_analysis_round": 2,
                "qualification_response_data_ref": "round-2/manifest.json",
            }
        },
        "facet_iteration_state": {
            "baseline_facets": {
                "failed-facet": {
                    "response_data_ref": "round-1/manifest.json",
                    "item_versions": {"changed-item": 2},
                    "selected_item_ids": ["changed-item"],
                }
            },
            "accepted_facets": {},
        },
    }
    assert simulation._facet_iteration_response_refs(
        state, item_versions={"changed-item": 3}
    ) == {}
    assert simulation._qualified_item_source_round(state, item) == (
        2,
        "round-2/manifest.json",
    )


def test_later_round_rejects_source_manifest_with_a_different_cohort(
    tmp_path: Path,
) -> None:
    state, config, references, items = _profile_run_inputs()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    profiles_path = source_dir / "score_profiles.json"
    profiles_path.write_text(
        json.dumps({
            "profiles": references,
            "demographics_snapshot": config["demographics_snapshot"],
            "demographics_version": config["demographics_version"],
            "response_temperature": config["response_temperature"],
            "score_specs": config["score_specs"],
        }),
        encoding="utf-8",
    )
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "status": "completed",
            "schema_version": simulation.MATCHED_CONDITION_SCHEMA_VERSION,
            "run_id": state["run_id"],
            "sample_size_per_condition": config["sample_size_per_condition"],
            "conditions": config["conditions"],
            "persona_modes": [PERSONA_MODE_SCORE_PROFILE],
            "model_id": config["model_id"],
            "prompt_version": simulation.VIRTUAL_RESPONSE_PROMPT_VERSION,
            "score_noise_method": "fixed_scores_temperature_sampling",
            "response_temperature": config["response_temperature"],
            "demographics_version": config["demographics_version"],
            "score_prompt_version": simulation.MATCHED_CONDITION_PROMPT_VERSION,
            "generator_version": simulation.MATCHED_CONDITION_GENERATOR_VERSION,
            "virtual_sample_config": {**config, "generation_round": 1},
            "score_profiles_path": str(profiles_path),
        }),
        encoding="utf-8",
    )
    state["previous_virtual_response_data_ref"] = str(manifest_path)
    next_config = {**config, "generation_round": 2}
    changed_references = json.loads(json.dumps(references))
    changed_references[0]["score_values"]["gregariousness"] += 1
    output_dir = tmp_path / "round-2"

    with pytest.raises(ValueError, match="score_profiles|cohort"):
        simulation._seed_matched_response_files(
            state=state,
            output_dir=output_dir,
            config=next_config,
            conditions=next_config["conditions"],
            model_id=next_config["model_id"],
            respondent_refs=changed_references,
            sjt_items=items,
            sjt_path=output_dir / "sjt_responses.jsonl",
            target_retest_path=output_dir / "target_form_retest_responses.jsonl",
            ipip_neo_path=output_dir / "ipip_neo_responses.jsonl",
            option_order_path=output_dir / "option_orders.jsonl",
        )


def test_fixed_facet_iteration_cannot_use_legacy_single_item_retest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sjt_system.evaluation import facet_iteration

    monkeypatch.setattr(facet_iteration, "is_enabled", lambda _state: True)
    with pytest.raises(ValueError, match="已提交的批次施测"):
        asyncio.run(
            simulation.run_single_item_virtual_retest(
                {"facet_iteration_state": {"status": "awaiting_measurement"}},
                {"item_id": "item-1", "version": 2},
                output_root=tmp_path,
            )
        )


def test_legacy_manifest_and_source_configs_ignore_only_concurrency_metadata() -> None:
    state, legacy_config, _references, _items = _profile_run_inputs()
    new_config = {
        **legacy_config,
        "max_concurrency": 0,
        "concurrency_policy": "all_at_once",
    }
    manifest = {
        "status": "in_progress",
        "schema_version": simulation.MATCHED_CONDITION_SCHEMA_VERSION,
        "run_id": state["run_id"],
        "item_bank_id": state["item_bank_id"],
        "item_bank_version": state["item_bank_version"],
        "item_bank_fingerprint": state["item_bank_fingerprint"],
        "sample_size_per_condition": legacy_config["sample_size_per_condition"],
        "pool_id": legacy_config["pool_id"],
        "source_sha256": legacy_config["source_sha256"],
        "persona_modes": [PERSONA_MODE_SCORE_PROFILE],
        "model_id": legacy_config["model_id"],
        "prompt_version": simulation.VIRTUAL_RESPONSE_PROMPT_VERSION,
        "score_noise_method": "fixed_scores_temperature_sampling",
        "response_temperature": 1.5,
        "demographics_version": "demographics-v1",
        "score_prompt_version": simulation.MATCHED_CONDITION_PROMPT_VERSION,
        "generator_version": simulation.MATCHED_CONDITION_GENERATOR_VERSION,
        "virtual_sample_config": legacy_config,
        "conditions": legacy_config["conditions"],
    }

    assert simulation._matched_manifest_core_is_compatible(
        manifest,
        state=state,
        config=new_config,
        conditions=new_config["conditions"],
        persona_modes=[PERSONA_MODE_SCORE_PROFILE],
        model_id=legacy_config["model_id"],
    )
    assert simulation._matched_source_is_compatible(
        manifest,
        state=state,
        config=new_config,
        conditions=new_config["conditions"],
        model_id=legacy_config["model_id"],
    )

    changed_config = {**new_config, "max_retries": 2}
    assert not simulation._matched_source_is_compatible(
        manifest,
        state=state,
        config=changed_config,
        conditions=changed_config["conditions"],
        model_id=legacy_config["model_id"],
    )


def _profile_run_inputs() -> tuple[dict, dict, list[dict], dict]:
    conditions = [
        {
            "condition_id": condition_id,
            "arm_id": condition_id,
            "role": condition_id,
            "group_id": dimension_id,
            "dimension_id": dimension_id,
            "domain_id": domain_id,
            "domain_name": domain_id,
            "domain_name_en": domain_id.title(),
            "facet_name": dimension_id,
            "facet_name_en": dimension_id.title(),
            "definition": f"Definition of {dimension_id}",
        }
        for condition_id, dimension_id, domain_id in (
            ("target", "gregariousness", "extraversion"),
            ("same_domain", "assertiveness", "extraversion"),
            ("cross_domain", "orderliness", "conscientiousness"),
        )
    ]
    snapshot = load_demographics_snapshot()
    references = [
        {
            "respondent_id": "target-matched-1",
            "condition_id": "target",
            "arm_id": "target",
            "group_id": "gregariousness",
            "matched_subject_id": "matched-1",
            "active_dimension_id": "gregariousness",
            "demographics": generate_demographics(17, snapshot),
            "score_values": {
                "gregariousness": 60.0,
                "assertiveness": 45.0,
                "orderliness": 50.0,
            },
        }
    ]
    config = {
        "schema_version": simulation.MATCHED_CONDITION_SCHEMA_VERSION,
        "demographics_version": "demographics-v1",
        "demographics_seed": 17,
        "demographics_snapshot": snapshot,
        "response_temperature": 1.5,
        "score_specs": matched_score_specs(conditions),
        "conditions": conditions,
        "persona_modes": [PERSONA_MODE_SCORE_PROFILE],
        "sample_size": 1,
        "sample_size_per_condition": 1,
        "pool_id": "local-test-pool",
        "source_sha256": "local-test-source",
        "model_id": "local-fake",
        "seed": 17,
        "max_concurrency": 1,
        "max_retries": 0,
        "response_count_per_respondent_item": 1,
        "target_form_administration_count": 2,
    }
    items = [
        {
            "item_id": f"item-{index}",
            "version": 1,
            "target_dimension_id": "gregariousness",
            "scenario": f"团队合作情境 {index}",
            "response_instruction": "你会怎么做？",
            "response_options": [
                {"option_id": f"O{option}", "text": f"行为 {option}"}
                for option in range(1, 5)
            ],
            "scoring_key": {f"O{option}": option for option in range(1, 5)},
        }
        for index in range(1, 4)
    ]
    state = {
        "run_id": "local-concurrency-test",
        "item_bank_id": "bank-local",
        "item_bank_version": 1,
        "item_bank_fingerprint": "local-fingerprint",
        "frozen_item_bank": items,
    }
    return state, config, references, items


def test_independent_stages_and_retest_retries_fan_out_and_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, config, references, items = _profile_run_inputs()
    progress_events: list[dict] = []
    monkeypatch.setattr(simulation, "emit_progress", progress_events.append)
    monkeypatch.setattr(
        simulation,
        "resolve_ipip_neo_reference_facet_ids",
        lambda _state, _items: ["gregariousness"],
    )
    monkeypatch.setattr(
        simulation,
        "load_ipip_neo_facet_scales",
        lambda _facet_ids, _path: [
            {
                "facet_id": "gregariousness",
                "facet_code": "E1",
                "alpha": 0.8,
                "corpus_hash": "local-corpus",
                "source_file": "local-corpus.json",
                "items": [
                    {
                        "item_id": f"ipip-{index}",
                        "text": f"本地参照项目 {index}",
                        "polarity": "positive",
                    }
                    for index in range(1, 11)
                ],
            }
        ],
    )
    monkeypatch.setattr(
        simulation,
        "_seed_matched_response_files",
        lambda **_kwargs: {
            "source_manifest_path": None,
            "source_compatible": False,
            "reused_sjt_records": 0,
            "reused_local_item_count": 0,
            "reused_target_retest_records": 0,
            "reused_ipip_neo_records": 0,
        },
    )
    runner = object.__new__(simulation.VirtualResponseRunner)
    runner.sjt_batch_model = "sjt"
    runner.ipip_neo_model = "ipip"
    runner.max_concurrency = 0
    runner.max_retries = 0
    runner.retry_delay_seconds = 0.0
    runner.request_timeout_seconds = 10.0
    runner.semaphore = UnlimitedConcurrency()
    runner.write_lock = asyncio.Lock()
    runner.model_id = "local-fake"
    runner.response_temperature = 1.5

    first_wave = asyncio.Event()
    retry_wave = asyncio.Event()
    first_seen: set[str] = set()
    retry_seen: set[str] = set()
    fail_item_two = True
    resuming = False
    active = 0
    first_wave_peak = 0
    retry_wave_peak = 0

    async def fake_invoke(
        runnable,
        _messages,
        *,
        validator,
        job_label,
        **_kwargs,
    ):
        nonlocal active, first_wave_peak, retry_wave_peak
        if runnable == "ipip":
            stage = "ipip"
        elif "item retest" in job_label:
            stage = "retry"
        elif "batch retest" in job_label:
            stage = "retest"
        else:
            stage = "sjt"

        if stage == "retry":
            item_id = job_label.rsplit("/", 1)[-1]
            retry_seen.add(item_id)
            active += 1
            retry_wave_peak = max(retry_wave_peak, active)
            if retry_seen == {"item-2", "item-3"}:
                retry_wave.set()
            try:
                await asyncio.wait_for(retry_wave.wait(), timeout=0.5)
            finally:
                active -= 1
            if item_id == "item-2" and fail_item_two:
                raise RuntimeError("injected item retest failure")
            return validator(
                {
                    "responses": [
                        {"item_id": item_id, "selected_option_id": "A"}
                    ]
                }
            )

        if not resuming:
            first_seen.add(stage)
            active += 1
            first_wave_peak = max(first_wave_peak, active)
            if first_seen == {"sjt", "retest", "ipip"}:
                first_wave.set()
            try:
                await asyncio.wait_for(first_wave.wait(), timeout=0.5)
            finally:
                active -= 1

        if stage == "ipip":
            return validator({"ratings": [4] * 10})
        if stage == "retest" and not resuming:
            try:
                return validator(
                    {
                        "responses": [
                            {"item_id": "item-1", "selected_option_id": "A"}
                        ]
                    }
                )
            except simulation.SJTBatchValidationError as exc:
                raise RuntimeError("partial target retest batch") from exc
        response_ids = ["item-2"] if resuming and stage == "retest" else [
            "item-1", "item-2", "item-3"
        ]
        return validator(
            {
                "responses": [
                    {"item_id": item_id, "selected_option_id": "A"}
                    for item_id in response_ids
                ]
            }
        )

    monkeypatch.setattr(simulation, "_invoke_with_retry", fake_invoke)

    async def run_once():
        return await runner._run_matched_condition_profile(
            state=state,
            output_dir=tmp_path / "responses",
            context={"items": items},
            config=config,
            respondent_refs=references,
            criterion={
                "domain_id": "extraversion",
                "domain_name_en": "Extraversion",
            },
            ipip_neo_path=tmp_path / "unused.json",
        )

    with pytest.raises(RuntimeError, match="虚拟作答有"):
        asyncio.run(run_once())

    assert first_seen == {"sjt", "retest", "ipip"}
    assert first_wave_peak == 3
    assert retry_seen == {"item-2", "item-3"}
    assert retry_wave_peak == 2
    response_dir = tmp_path / "responses"
    assert len(
        (response_dir / "sjt_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 3
    assert len(
        (response_dir / "target_form_retest_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2
    assert len(
        (response_dir / "ipip_neo_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 10
    failed_manifest = json.loads(
        (response_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["status"] == "failed"

    fail_item_two = False
    resuming = True
    config.update(max_concurrency=0, concurrency_policy="all_at_once")
    retry_seen.clear()
    result = asyncio.run(run_once())
    assert result["sjt_response_count"] == 3
    assert result["target_form_retest_response_count"] == 3
    assert result["ipip_neo_response_count"] == 10
    assert result["reused_sjt_records"] == 3
    assert result["reused_target_retest_records"] == 2
    assert result["reused_ipip_neo_records"] == 10
    assert retry_seen == set()
    assert result["max_concurrency"] == 0
    assert result["concurrency_policy"] == "all_at_once"

    final_manifest = json.loads(
        (response_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["status"] == "completed"
    started_stages = [
        event["stage"]
        for event in progress_events
        if event.get("type") == "simulation_stage"
        and event.get("status") == "started"
    ]
    assert started_stages[:3] == [
        "SJT matched facet response",
        "SJT target form retest",
        "IPIP-NEO selected facet response",
    ]


def test_later_round_reuses_frozen_first_round_ipip_without_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, config, references, items = _profile_run_inputs()
    monkeypatch.setattr(
        simulation,
        "resolve_ipip_neo_reference_facet_ids",
        lambda _state, _items: ["gregariousness"],
    )
    monkeypatch.setattr(
        simulation,
        "load_ipip_neo_facet_scales",
        lambda _facet_ids, _path: [
            {
                "facet_id": "gregariousness",
                "facet_code": "E1",
                "alpha": 0.8,
                "corpus_hash": "local-corpus",
                "source_file": "local-corpus.json",
                "items": [
                    {
                        "item_id": f"ipip-{index}",
                        "text": f"本地参照项目 {index}",
                        "polarity": "positive",
                    }
                    for index in range(1, 11)
                ],
            }
        ],
    )
    monkeypatch.setattr(
        simulation,
        "_seed_matched_response_files",
        lambda **_kwargs: {
            "source_manifest_path": None,
            "source_compatible": False,
            "reused_sjt_records": 0,
            "reused_local_item_count": 0,
            "reused_target_retest_records": 0,
            "reused_ipip_neo_records": 0,
        },
    )
    runner = object.__new__(simulation.VirtualResponseRunner)
    runner.sjt_batch_model = "sjt"
    runner.ipip_neo_model = "ipip"
    runner.max_concurrency = 0
    runner.max_retries = 0
    runner.retry_delay_seconds = 0.0
    runner.request_timeout_seconds = 10.0
    runner.semaphore = UnlimitedConcurrency()
    runner.write_lock = asyncio.Lock()
    runner.model_id = "local-fake"
    runner.response_temperature = 1.5
    ipip_calls: list[str] = []

    async def fake_invoke(
        runnable,
        _messages,
        *,
        validator,
        job_label,
        **_kwargs,
    ):
        if runnable == "ipip":
            ipip_calls.append(job_label)
            return validator({"ratings": [4] * 10})
        return validator(
            {
                "responses": [
                    {"item_id": item_id, "selected_option_id": "A"}
                    for item_id in ["item-1", "item-2", "item-3"]
                ]
            }
        )

    monkeypatch.setattr(simulation, "_invoke_with_retry", fake_invoke)

    async def run_once(run_state, run_config, output_dir):
        return await runner._run_matched_condition_profile(
            state=run_state,
            output_dir=output_dir,
            context={"items": items},
            config=run_config,
            respondent_refs=references,
            criterion={
                "domain_id": "extraversion",
                "domain_name_en": "Extraversion",
            },
            ipip_neo_path=tmp_path / "unused.json",
        )

    first = asyncio.run(run_once(state, config, tmp_path / "round1"))
    assert first["scheduled_ipip_neo_api_calls"] == 1
    assert len(ipip_calls) == 1

    second_state = {
        **state,
        "frozen_reference_questionnaire_ref": first["manifest_path"],
    }
    second = asyncio.run(
        run_once(
            second_state,
            {**config, "generation_round": 2},
            tmp_path / "round2",
        )
    )
    assert second["scheduled_ipip_neo_api_calls"] == 0
    assert second["frozen_reference_reused"] is True
    assert second["frozen_reference_reused_records"] == 10
    assert second["reused_ipip_neo_records"] == 10
    assert second["frozen_reference_questionnaire_ref"] == first["manifest_path"]
    assert len(ipip_calls) == 1
    copied_records = [
        json.loads(line)
        for line in (tmp_path / "round2" / "ipip_neo_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(copied_records) == 10
    assert {
        record["response_source"] for record in copied_records
    } == {"reused_frozen_first_round_reference"}


@pytest.mark.parametrize("ipip_corruption", [None, "missing", "identity"])
def test_later_bank_reuses_only_unchanged_versions_and_freezes_ipip(
    tmp_path: Path,
    monkeypatch,
    ipip_corruption: str | None,
) -> None:
    state, seed_config, _seed_references, items = _profile_run_inputs()
    dimension_catalog = [
        {
            "dimension_id": row["dimension_id"],
            "level": "facet",
            "domain_id": row["domain_id"],
            "domain_name": row["domain_name"],
            "domain_name_en": row["domain_name_en"],
            "facet_name": row["facet_name"],
            "facet_name_en": row["facet_name_en"],
            "definition": row["definition"],
            "high_behavior": f"High behavior for {row['dimension_id']}",
            "low_behavior": f"Low behavior for {row['dimension_id']}",
        }
        for row in seed_config["conditions"]
    ]
    raw_conditions = [
        {
            "condition_id": "target",
            "role": "target",
            "groups": [{"group_id": "target", "dimension_id": "gregariousness"}],
        },
        {
            "condition_id": "same_domain",
            "role": "same_domain_non_target",
            "groups": [{
                "group_id": "assertiveness",
                "dimension_id": "assertiveness",
                "comparison_target_dimension_id": "gregariousness",
            }],
        },
        {
            "condition_id": "cross_domain",
            "role": "cross_domain_non_target",
            "groups": [{
                "group_id": "orderliness",
                "dimension_id": "orderliness",
                "comparison_target_dimension_id": "gregariousness",
            }],
        },
    ]
    conditions = normalize_matched_conditions(
        raw_conditions,
        dimension_catalog=dimension_catalog,
        target_dimension_id="gregariousness",
        shared_score_distribution={"family": "normal", "mean": 50.0, "sd": 15.0},
    )
    references, generation_diagnostics = generate_matched_condition_respondent_refs(
        30,
        conditions,
        generation_round=1,
        seed=17,
    )
    config = build_matched_condition_sample_config(
        30,
        conditions=conditions,
        generation_diagnostics=generation_diagnostics,
        mean_score=50.0,
        standard_deviation=15.0,
        generation_round=1,
        seed=17,
        max_retries=0,
        response_temperature=1.5,
    )
    config["model_id"] = "local-fake"
    monkeypatch.setattr(
        simulation,
        "matched_condition_sample_is_current",
        validate_matched_condition_sample,
    )
    monkeypatch.setattr(
        simulation,
        "resolve_ipip_neo_reference_facet_ids",
        lambda _state, _items: ["gregariousness"],
    )
    scales = [{
        "facet_id": "gregariousness",
        "facet_code": "E1",
        "alpha": 0.8,
        "corpus_hash": "local-corpus",
        "source_file": "local-corpus.json",
        "items": [
            {"item_id": f"ipip-{index}", "text": f"本地参照项目 {index}", "polarity": "positive"}
            for index in range(1, 11)
        ],
    }]
    monkeypatch.setattr(
        simulation,
        "load_ipip_neo_facet_scales",
        lambda _facet_ids, _path: deepcopy(scales),
    )
    runner = object.__new__(simulation.VirtualResponseRunner)
    runner.sjt_batch_model = "sjt"
    runner.ipip_neo_model = "ipip"
    runner.max_concurrency = 0
    runner.max_retries = 0
    runner.retry_delay_seconds = 0.0
    runner.request_timeout_seconds = 10.0
    runner.semaphore = UnlimitedConcurrency()
    runner.write_lock = asyncio.Lock()
    runner.model_id = "local-fake"
    runner.response_temperature = 1.5

    original_builder = simulation.build_sjt_batch_messages
    row_ids_by_messages: dict[int, list[str]] = {}
    retained_messages: list[list[tuple[str, str]]] = []

    def capture_sjt_rows(rows):
        messages = original_builder(rows)
        row_ids_by_messages[id(messages)] = [str(row[0]["item_id"]) for row in rows]
        retained_messages.append(messages)
        return messages

    monkeypatch.setattr(simulation, "build_sjt_batch_messages", capture_sjt_rows)
    calls: list[tuple[int, str, list[str], str]] = []
    active_round = 1

    async def fake_invoke(runnable, messages, *, validator, job_label, **_kwargs):
        if runnable == "ipip":
            calls.append((active_round, "ipip", [], repr(messages)))
            return validator({"ratings": [4] * 10})
        stage = "retest" if "retest" in job_label else "main"
        ids = row_ids_by_messages[id(messages)]
        calls.append((active_round, stage, ids, repr(messages)))
        answer = "D" if stage == "retest" else "A"
        return validator({"responses": [
            {"item_id": item_id, "selected_option_id": answer} for item_id in ids
        ]})

    monkeypatch.setattr(simulation, "_invoke_with_retry", fake_invoke)

    async def run_once(run_state, run_config, run_items, output_dir):
        return await runner._run_matched_condition_profile(
            state=run_state,
            output_dir=output_dir,
            context={"items": run_items},
            config=run_config,
            respondent_refs=references,
            criterion={"domain_id": "extraversion", "domain_name_en": "Extraversion"},
            ipip_neo_path=tmp_path / "unused.json",
        )

    first = asyncio.run(run_once(state, config, items, tmp_path / "round-1"))
    respondent_count = len(references)
    assert len([row for row in calls if row[0] == 1 and row[1] == "main"]) == respondent_count
    assert len([row for row in calls if row[0] == 1 and row[1] == "retest"]) == respondent_count
    assert len([row for row in calls if row[0] == 1 and row[1] == "ipip"]) == respondent_count

    first_manifest = Path(first["manifest_path"])
    if ipip_corruption == "missing":
        Path(first["ipip_neo_path"]).unlink()
    elif ipip_corruption == "identity":
        ipip_path = Path(first["ipip_neo_path"])
        ipip_rows = [json.loads(line) for line in ipip_path.read_text(encoding="utf-8").splitlines()]
        ipip_rows[0]["matched_subject_id"] = "wrong-cohort"
        ipip_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ipip_rows),
            encoding="utf-8",
        )

    second_state = deepcopy(state)
    next_items = [
        {**deepcopy(item), "version": 2} if item["item_id"] == "item-2" else deepcopy(item)
        for item in items
    ]
    second_state.update(
        item_bank_id="bank-local-v2",
        item_bank_version=2,
        item_bank_fingerprint="local-fingerprint-v2",
        previous_virtual_response_data_ref=str(first_manifest),
        frozen_reference_questionnaire_ref=str(first_manifest),
        frozen_item_bank=next_items,
        item_final_dispositions={
            "item-1": {
                "status": "qualified_locked",
                "item_version": 1,
                "qualification_analysis_round": 1,
                "qualification_response_data_ref": str(first_manifest),
            },
        },
        facet_iteration_state={
            "policy_version": "fixed_cohort_facet_iteration_v2",
            "status": "awaiting_measurement",
            "development_round": 2,
            "completed_rounds": 1,
            "measurement_batch": 1,
            "baseline_facets": {
                "gregariousness": {
                    "response_data_ref": str(first_manifest),
                    "selected_item_ids": ["item-1", "item-2", "item-3"],
                    "item_versions": {"item-1": 1, "item-2": 1, "item-3": 1},
                    "item_snapshots": {item["item_id"]: deepcopy(item) for item in items},
                }
            },
            "accepted_facets": {
                "stable-facet": {
                    "response_data_ref": str(first_manifest),
                    "selected_item_ids": ["item-1"],
                    "item_versions": {"item-1": 1},
                    "item_snapshots": {"item-1": deepcopy(items[0])},
                    "analysis_round": 1,
                }
            },
            "local_attempts": {"gregariousness": 1},
            "failed_facet_ids": ["gregariousness"],
            "pending_facet_ids": ["gregariousness"],
            "cohort_source_ref": str(first_manifest),
            "first_round_reference_ref": str(first_manifest),
            "thawed_item_versions": {},
        },
    )
    calls_before_second = len(calls)
    active_round = 2
    later_config = {
        **config,
        "generation_round": 2,
        "reference_questionnaire_freeze_policy": "freeze_after_first_measurement",
        "frozen_reference_generation_round": 1,
        "frozen_reference_questionnaire_ref": str(first_manifest),
    }
    if ipip_corruption:
        with pytest.raises(ValueError, match="IPIP|校标|cohort"):
            asyncio.run(run_once(
                second_state,
                later_config,
                next_items,
                tmp_path / "round-2",
            ))
        assert len(calls) == calls_before_second
        return

    second = asyncio.run(run_once(
        second_state,
        later_config,
        next_items,
        tmp_path / "round-2",
    ))
    second_calls = calls[calls_before_second:]
    assert len([row for row in second_calls if row[1] == "main"]) == respondent_count
    assert len([row for row in second_calls if row[1] == "retest"]) == respondent_count
    assert not any(row[1] == "ipip" for row in second_calls)
    assert all(
        row[2] == ["item-2"]
        for row in second_calls
        if row[1] in {"main", "retest"}
    )
    assert second["scheduled_sjt_api_calls"] == respondent_count
    assert second["scheduled_target_form_retest_api_calls"] == respondent_count
    assert second["scheduled_ipip_neo_api_calls"] == 0
    assert second["reused_sjt_records"] == respondent_count * 2
    assert second["reused_target_retest_records"] == respondent_count * 2
    assert second["frozen_reference_reused_records"] == respondent_count * 10

    main_rows = [
        json.loads(line)
        for line in Path(second["sjt_path"]).read_text(encoding="utf-8").splitlines()
    ]
    retest_rows = [
        json.loads(line)
        for line in Path(second["target_form_retest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(main_rows) == respondent_count * 3
    assert len(retest_rows) == respondent_count * 3
    assert {row["item_id"] for row in main_rows} == {"item-1", "item-2", "item-3"}
    assert {row["item_id"] for row in retest_rows} == {"item-1", "item-2", "item-3"}
    assert {row["administration_id"] for row in retest_rows} == {2}
    assert {row["item_version"] for row in retest_rows if row["item_id"] == "item-2"} == {2}
    assert {row["item_version"] for row in retest_rows if row["item_id"] != "item-2"} == {1}
    assert all(
        row.get("reuse_source_manifest_path") == str(first_manifest)
        for row in [*main_rows, *retest_rows]
        if row["item_id"] != "item-2"
    )
    assert all(row[2] == ["item-2"] for row in second_calls if row[1] == "retest")
    dirty_main = next(row for row in main_rows if row["item_id"] == "item-2")
    dirty_answer = dirty_main["selected_option_id"]
    retest_prompts = [row[3] for row in second_calls if row[1] == "retest"]
    assert dirty_main["raw_display_option_id"] == "A"
    assert all(dirty_answer not in prompt for prompt in retest_prompts)


def test_later_sample_round_keeps_the_frozen_target_cohort(monkeypatch) -> None:
    from sjt_system.workflow import interaction_nodes

    references = [
        {
            "respondent_id": "target-round1-matched-0001",
            "matched_subject_id": "round1-matched-0001",
            "condition_id": "target",
            "score_values": {"gregariousness": 62.0},
            "demographics": {"age": 31},
        }
    ]
    frozen_ref = "outputs/round1/manifest.json"
    config = {
        "generation_round": 1,
        "target_dimension_ids": ["gregariousness"],
        "sample_size": 1,
    }
    monkeypatch.setattr(
        interaction_nodes,
        "matched_condition_sample_is_current",
        lambda _config, _respondents: True,
    )
    monkeypatch.setattr(
        interaction_nodes,
        "interrupt",
        lambda _payload: (_ for _ in ()).throw(
            AssertionError("frozen later rounds must not ask for a new cohort")
        ),
    )
    state = {
        "virtual_sample_config": config,
        "virtual_respondents": references,
        "frozen_item_bank": [{"target_dimension_id": "gregariousness"}],
        "psychometric_analysis_round": 1,
        "frozen_reference_questionnaire_ref": frozen_ref,
        "execution_history": [],
        "run_id": "frozen-cohort-test",
        "step_count": 2,
    }

    result = interaction_nodes.virtual_sample_selection_node(state)

    assert result["virtual_sample_config"]["generation_round"] == 2
    assert result["virtual_sample_config"][
        "frozen_reference_questionnaire_ref"
    ] == frozen_ref
    assert result["virtual_respondents"] == references


@pytest.mark.parametrize("legacy_limit", [1, 20])
def test_local_retest_cache_accepts_legacy_concurrency_config(
    tmp_path: Path,
    monkeypatch,
    legacy_limit: int,
) -> None:
    state, config, references, items = _profile_run_inputs()
    config.update(max_concurrency=legacy_limit)
    state = {
        **state,
        "virtual_sample_config": config,
        "virtual_respondents": references,
    }
    monkeypatch.setattr(
        simulation,
        "build_virtual_response_context",
        lambda _state: {
            "virtual_sample_config": _state["virtual_sample_config"],
            "virtual_respondents": _state["virtual_respondents"],
        },
    )
    monkeypatch.setattr(
        simulation,
        "resolve_virtual_respondent_profiles",
        lambda refs, **_kwargs: [
            {"respondent_id": ref["respondent_id"]}
            for ref in refs
        ],
    )
    monkeypatch.setattr(simulation, "build_persona_prompt", lambda *_a, **_k: "persona")
    monkeypatch.setattr(simulation, "build_sjt_messages", lambda *_a, **_k: [])

    runner = object.__new__(simulation.VirtualResponseRunner)
    runner.sjt_model = "local-fake"
    runner.max_retries = 0
    runner.retry_delay_seconds = 0.0
    runner.request_timeout_seconds = 10.0
    runner.semaphore = UnlimitedConcurrency()
    runner.model_id = "local-fake"
    runner.response_temperature = 1.5
    call_count = 0

    async def fake_invoke(_model, _messages, *, validator, **_kwargs):
        nonlocal call_count
        call_count += 1
        return validator({"selected_option_id": "A"})

    monkeypatch.setattr(simulation, "_invoke_with_retry", fake_invoke)

    first = asyncio.run(
        runner.run_single_item_retest(
            state=state,
            item=items[0],
            output_root=tmp_path,
        )
    )
    assert first["cached"] is False
    assert call_count == len(references)

    config.update(max_concurrency=0, concurrency_policy="all_at_once")
    second = asyncio.run(
        runner.run_single_item_retest(
            state=state,
            item=items[0],
            output_root=tmp_path,
        )
    )
    assert second["cached"] is True
    assert second["scheduled_sjt_api_calls"] == 0
    assert call_count == len(references)
