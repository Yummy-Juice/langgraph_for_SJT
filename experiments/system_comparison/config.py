"""Frozen, non-secret experiment parameters and matched participant samples."""
from hashlib import sha256
import json
import os
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sjt_system.authoring.construct_registry import (
    construct_selection_catalog, resolve_construct_profile,
)
from sjt_system.evaluation.respondents import (
    build_score_dimension_catalog, normalize_matched_conditions,
    generate_matched_condition_respondent_refs, build_matched_condition_sample_config,
    MATCHED_CONDITION_ROLES,
)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_facet: str = "extraversion_gregariousness"
    same_domain_facet: str = "extraversion_warmth"
    cross_domain_facet: str = "neuroticism_anxiety"
    target_population: str = "大学生"
    final_item_count: int = Field(default=16, ge=1, le=100, strict=True)
    sample_size: int = Field(default=100, ge=30, le=500, strict=True)
    seed: int = Field(default=7, ge=0, strict=True)
    mean: float = 50.0
    sd: float = Field(default=15.0, gt=0)
    max_concurrency: int = Field(default=5, ge=1, le=20, strict=True)
    max_repair_rounds: int = Field(default=3, ge=1, le=20, strict=True)
    plateau_patience: int = Field(default=2, ge=1, strict=True)
    plateau_min_delta: float = Field(default=0.01, ge=0)
    stability_minimum: float = Field(default=0.80, ge=0, le=1)
    # ``model_id`` is the authoring model for A/B/C: generation, review,
    # diagnosis, repair and form assembly.  Virtual respondents are configured
    # separately so a model change in the evaluator cannot silently change the
    # authoring treatment.
    model_id: str | None = None
    virtual_respondent_model_id: str | None = None
    evaluation_model_id: str | None = None
    input_price_per_million: float | None = Field(default=None, ge=0)
    cached_input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_constructs(self):
        conditions_for(self)
        if self.stability_minimum != 0.80:
            raise ValueError("首版保持主系统ICC门槛0.80")
        return self

    def resolved(self):
        # get_model imports dotenv; do not serialize API_KEY or the endpoint URL.
        from sjt_system.agent.client import get_model  # noqa: F401
        model = self.model_id or os.getenv("MODEL_ID", "deepseek-v4-flash")
        virtual_model = (
            self.virtual_respondent_model_id
            or os.getenv("VIRTUAL_RESPONDENT_MODEL_ID")
            or model
        )
        return self.model_copy(
            update={
                "model_id": model,
                "virtual_respondent_model_id": virtual_model,
                "evaluation_model_id": self.evaluation_model_id or virtual_model,
            }
        )


def conditions_for(config):
    return normalize_matched_conditions(
        [{"condition_id": arm, "role": MATCHED_CONDITION_ROLES[arm], "dimension_id": facet}
         for arm, facet in zip(("target", "same_domain", "cross_domain"),
                               (config.target_facet, config.same_domain_facet, config.cross_domain_facet))],
        dimension_catalog=build_score_dimension_catalog(construct_selection_catalog()),
        target_dimension_id=config.target_facet,
    )


def make_participants(config, role):
    if role not in {"development", "evaluation"}:
        raise ValueError("unknown sample role")
    seed = int.from_bytes(sha256(f"{config.seed}:{role}".encode()).digest()[:4], "big")
    conditions = conditions_for(config)
    refs, diagnostics = generate_matched_condition_respondent_refs(
        config.sample_size, conditions, seed=seed, mean_score=config.mean,
        standard_deviation=config.sd)
    for ref in refs:
        ref["matched_subject_id"] = f"{role}-{ref['matched_subject_id']}"
        ref["respondent_id"] = f"{ref['condition_id']}-{ref['matched_subject_id']}"
    sample_config = build_matched_condition_sample_config(
        config.sample_size, conditions=conditions, generation_diagnostics=diagnostics,
        mean_score=config.mean, standard_deviation=config.sd, seed=seed,
        max_concurrency=config.max_concurrency)
    sample_config["experiment_sample_role"] = role
    sample_config["reference_questionnaires_enabled"] = False
    return {"role": role, "config": sample_config, "respondents": refs}


def profile_for(config):
    return resolve_construct_profile(config.target_facet)


def fingerprint(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
