"""Pin and scope existing model roles for this single-process experiment.

The main system constructs agents at import time. Changing MODEL_ID alone cannot
reconfigure them. Rebind the actual consumers temporarily, without modifying the
main CLI defaults. Do not run two experiments concurrently in the same process.
"""
from contextlib import contextmanager
from copy import deepcopy
import os


def model_settings(config):
    from sjt_system.agent import agent_factory as factory
    base = {"model_id": config.model_id}
    roles = {
        "requirement_agent": {**base, "include_json_schema": False},
        "item_writer_agent": dict(base), "item_review_agent": dict(base),
        "item_repair_agent": dict(base),
        "mechanism_validation_agent": {**base, "temperature": .1},
    }
    _, temperature = factory._task_model_parameters("SKELETON_GENERATION", default_temperature=.7)
    roles["compact_skeleton_agent"] = {"model_id": config.model_id, "temperature": temperature}
    for role, prefix in (("psychometric_repair_diagnosis_agent", "PSYCHOMETRIC_DIAGNOSIS"),
                         ("psychometric_item_repair_agent", "PSYCHOMETRIC_ITEM_REPAIR")):
        _, temperature, thinking, effort = factory._reasoning_role_parameters(
            prefix,
            default_model_id=config.model_id,
        )
        roles[role] = {"model_id": config.model_id, "temperature": temperature,
                       "thinking_type": thinking, "reasoning_effort": effort}
    roles["psychometric_repair_diagnosis_agent"]["include_json_schema"] = False
    virtual_model = config.virtual_respondent_model_id or config.model_id
    evaluation_model = config.evaluation_model_id or virtual_model
    roles["development_virtual_respondent"] = {"model_id": virtual_model}
    roles["evaluation_virtual_respondent"] = {"model_id": evaluation_model}
    environment = {
        "MODEL_ID": config.model_id,
        "VIRTUAL_RESPONDENT_MODEL_ID": virtual_model,
    }
    for prefix in ("EXPANSION", "BLUEPRINT", "FORM_OPTIMIZER"):
        # A/B/C authoring must use one frozen model.  Role-specific model IDs
        # from the main CLI are deliberately not inherited by this experiment.
        environment[prefix + "_MODEL_ID"] = config.model_id
        if prefix != "FORM_OPTIMIZER":
            environment[prefix + "_TEMPERATURE"] = os.getenv(prefix + "_TEMPERATURE")
    environment["STRUCTURED_OUTPUT_METHOD"] = os.getenv("STRUCTURED_OUTPUT_METHOD")
    return {"roles": roles, "environment": environment}


@contextmanager
def workflow_model_scope(store):
    from sjt_system.agent import agent_factory as factory
    from sjt_system.authoring import situation_space
    from sjt_system.workflow import executor
    settings = store.read("model_roles.json")
    specifications = {
        "requirement_agent": (factory.REQUIREMENT_PROMPT, factory.RequirementResult),
        "compact_skeleton_agent": (factory.COMPACT_SKELETON_BATCH_PROMPT, factory.CompactSkeletonResult),
        "item_writer_agent": (factory.ITEM_WRITER_PROMPT, factory.ItemRealizationResult),
        "item_review_agent": (factory.UNIFIED_ITEM_REVIEW_PROMPT, factory.ItemReviewDiagnosis),
        "item_repair_agent": (factory.ITEM_REPAIR_PROMPT, factory.ItemRepairResult),
        "mechanism_validation_agent": (factory.MECHANISM_VALIDATION_PROMPT, factory.MechanismValidationResult),
        "psychometric_repair_diagnosis_agent": (factory.PSYCHOMETRIC_REPAIR_DIAGNOSIS_PROMPT, factory.AtomicRepairAdvice),
        "psychometric_item_repair_agent": (factory.ITEM_REPAIR_PROMPT, factory.ItemRepairResult),
    }
    previous_env = {k: os.environ.get(k) for k in settings["environment"]}
    previous = {}
    previous_map = dict(executor.AGENT_MAP)
    try:
        for name, value in settings["environment"].items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if hasattr(factory.create_agent, "cache_clear"):
            factory.create_agent.cache_clear()
        agents = {name: factory.create_agent(prompt, schema, **settings["roles"][name])
                  for name, (prompt, schema) in specifications.items()}
        agents["revision_agent"] = agents["item_repair_agent"]
        agents["item_regeneration_agent"] = agents["item_repair_agent"]
        for module in (executor, situation_space):
            for name, agent in agents.items():
                if hasattr(module, name):
                    previous[(module, name)] = getattr(module, name)
                    setattr(module, name, agent)
        executor.AGENT_MAP = {"clarify_requirements": agents["requirement_agent"],
                              "generate_item": agents["item_writer_agent"],
                              "revise_item": agents["revision_agent"],
                              "regenerate_item": agents["item_regeneration_agent"]}
        previous[(executor, "PSYCHOMETRIC_REASONING_ROLE_MANIFEST")] = executor.PSYCHOMETRIC_REASONING_ROLE_MANIFEST
        manifest = deepcopy(executor.PSYCHOMETRIC_REASONING_ROLE_MANIFEST)
        for name, agent in (("psychometric_diagnosis", "psychometric_repair_diagnosis_agent"),
                            ("psychometric_item_repair", "psychometric_item_repair_agent")):
            parameters = settings["roles"][agent]
            manifest[name].update({"model_id": parameters["model_id"], "temperature": parameters["temperature"],
                                   "thinking": parameters["thinking_type"], "reasoning_effort": parameters["reasoning_effort"]})
        executor.PSYCHOMETRIC_REASONING_ROLE_MANIFEST = manifest
        yield
    finally:
        if hasattr(factory.create_agent, "cache_clear"):
            factory.create_agent.cache_clear()
        for (module, name), value in previous.items():
            setattr(module, name, value)
        executor.AGENT_MAP = previous_map
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
