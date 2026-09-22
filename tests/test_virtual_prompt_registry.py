from pathlib import Path

from experiments.system_comparison.virtual_prompt_registry import (
    DEFAULT_PROMPT_ROOT,
    VirtualPromptSpec,
    format_extra_parameters,
    get_prompt_spec,
    load_prompt_specs,
)


def test_builtin_prompt_specs_are_valid() -> None:
    specs = load_prompt_specs(DEFAULT_PROMPT_ROOT)
    assert set(specs) == {
        "score_profile_v1",
        "five_facet_profile_v1",
        "embodied_probability_v1",
    }
    assert specs["score_profile_v1"].persona_input == "score_values"
    assert specs["score_profile_v1"].persona_information
    assert specs["score_profile_v1"].response_behavior
    assert specs["five_facet_profile_v1"].sjt_response_mode == "single_selection"
    assert specs["embodied_probability_v1"].sjt_response_mode == "choice_probability"


def test_custom_prompt_template_renders_persona_fields() -> None:
    spec = get_prompt_spec("five_facet_profile_v1")
    rendered = spec.render(score_lines="N4=50\nE2=70")
    assert "N4=50" in rendered
    assert "E2=70" in rendered
    assert "{score_lines}" not in rendered


def test_prompt_selection_defaults_to_one_method_without_input(monkeypatch) -> None:
    from experiments.system_comparison.virtual_prompt_registry import (
        select_prompt_ids_interactively,
    )

    monkeypatch.setattr("builtins.input", lambda _: "")
    specs = load_prompt_specs(DEFAULT_PROMPT_ROOT)
    assert select_prompt_ids_interactively(
        specs,
        default_ids=("score_profile_v1", "five_facet_profile_v1"),
    ) == ("score_profile_v1",)


def test_prompt_selection_accepts_one_method(monkeypatch) -> None:
    from experiments.system_comparison.virtual_prompt_registry import (
        select_prompt_ids_interactively,
    )

    monkeypatch.setattr("builtins.input", lambda _: "2")
    specs = load_prompt_specs(DEFAULT_PROMPT_ROOT)
    assert select_prompt_ids_interactively(specs) == ("five_facet_profile_v1",)


def test_custom_parameter_values_are_rendered_per_respondent(tmp_path: Path) -> None:
    values_file = tmp_path / "parameters.json"
    values_file.write_text(
        '{"respondent-1": {"awakening_level": 80}}',
        encoding="utf-8",
    )
    spec = VirtualPromptSpec.from_mapping(
        {
            "prompt_id": "custom",
            "label": "custom",
            "description": "custom",
            "persona_information": "scores",
            "response_behavior": "first person",
            "persona_input": "score_values",
            "sjt_response_mode": "single_selection",
            "ipip_response_mode": "likert_1_5",
            "version": "v1",
            "extra_parameters": [
                {
                    "key": "awakening_level",
                    "label": "个人觉醒水平",
                    "description": "0到100",
                    "default": 50,
                }
            ],
            "parameter_values_file": str(values_file),
            "template": "{score_lines}\n{extra_parameters}",
        },
        source=tmp_path / "custom.json",
    )
    rendered = spec.render(
        score_lines="E2=70",
        extra_parameters=format_extra_parameters(
            spec,
            "respondent-1",
            root=tmp_path,
        ),
    )
    assert "E2=70" in rendered
    assert "个人觉醒水平：80" in rendered
