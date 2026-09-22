import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import TypeAdapter, ValidationError as PydanticValidationError
from sjt_system.agent.json_parsing import parse_model_json_response
from sjt_system.runtime.telemetry import HANDLER as TELEMETRY_HANDLER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_model(
    model_id: str | None = None,
    *,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    thinking_type: str | None = None,
) -> ChatOpenAI:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY environment variable.")
    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.strip().lower()
        if reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError(
                "reasoning_effort must be low, medium, high, or max"
            )
    if thinking_type is not None:
        thinking_type = thinking_type.strip().lower()
        if thinking_type not in {"enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled or disabled")
    if reasoning_effort is not None and thinking_type == "disabled":
        raise ValueError(
            "thinking_type cannot be disabled when reasoning_effort is set"
        )

    # Preserve the historical temperature=1.0 default for ordinary agents.
    # Reasoning agents omit temperature unless the role explicitly sets it,
    # because reasoning endpoints do not consistently accept sampling controls.
    resolved_temperature = (
        None
        if temperature is None and (reasoning_effort or thinking_type)
        else 1.0 if temperature is None else temperature
    )
    if resolved_temperature is not None and (
        not isinstance(resolved_temperature, (int, float))
        or isinstance(resolved_temperature, bool)
        or resolved_temperature < 0
        or resolved_temperature > 2
    ):
        raise ValueError("model temperature must be between 0 and 2")
    return ChatOpenAI(
        model=model_id or os.getenv("MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        base_url=os.getenv("BASE_URL") or None,
        temperature=(
            float(resolved_temperature)
            if resolved_temperature is not None
            else None
        ),
        reasoning_effort=reasoning_effort,
        extra_body=(
            {"thinking": {"type": thinking_type}}
            if thinking_type is not None
            else None
        ),
        timeout=get_model_request_timeout_seconds(),
        max_retries=0,
        # Observation-only telemetry: token usage and latency per model call.
        callbacks=[TELEMETRY_HANDLER],
    )


SUPPORTED_STRUCTURED_OUTPUT_METHODS = {
    "function_calling",
    "json_mode",
    "json_schema",
    "plain_json",
}
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_MODEL_REQUEST_MAX_ATTEMPTS = 2


class LocalJSONSchemaError(ValueError):
    """A parsed JSON candidate that failed local schema validation."""

    def __init__(
        self,
        message: str,
        *,
        candidate: Any,
        field_path: str,
    ) -> None:
        super().__init__(message)
        self.candidate = candidate
        self.field_path = field_path


_OUTPUT_ENVELOPE_KEYS = ("result", "output", "data", "response")
_STATE_UPDATE_ALIAS_KEYS = ("update", "patch", "item_patch", "state")


def _schema_node(
    schema: Mapping[str, Any],
    node: Any,
) -> Mapping[str, Any]:
    """Resolve one local JSON-Schema reference without following arbitrary URIs."""

    if not isinstance(node, Mapping):
        return {}
    reference = node.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return node
    current: Any = schema
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            return {}
        current = current[part]
    return current if isinstance(current, Mapping) else {}


def normalize_model_output_shape(value: Any, output_type: type) -> Any:
    """Normalize common transport shapes before strict schema validation.

    This adapter changes containers only. It never invents domain values,
    rewrites item text, or relaxes downstream business validation. Supported
    compatibility cases are deliberately narrow:

    * a single conventional envelope such as ``result`` or ``output``;
    * a known alias for ``state_update``;
    * state-update fields returned directly at the top level.
    """

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
    if not isinstance(value, Mapping):
        return value

    schema = TypeAdapter(output_type).json_schema()
    root_properties = schema.get("properties")
    if not isinstance(root_properties, Mapping):
        return dict(value)
    root_keys = {str(key) for key in root_properties}
    candidate = dict(value)
    state_property = _schema_node(schema, root_properties.get("state_update"))
    state_properties = state_property.get("properties")
    state_keys = (
        {str(key) for key in state_properties}
        if isinstance(state_properties, Mapping)
        else set()
    )

    # Some compatible endpoints wrap the requested object in one generic
    # transport envelope. Preserve any valid root-level sibling (for example
    # ``summary``) while unwrapping the nested object.
    if not (root_keys & set(candidate)):
        for envelope_key in _OUTPUT_ENVELOPE_KEYS:
            nested = candidate.get(envelope_key)
            if isinstance(nested, Mapping):
                candidate = dict(nested)
                break
    else:
        for envelope_key in _OUTPUT_ENVELOPE_KEYS:
            nested = candidate.get(envelope_key)
            if not isinstance(nested, Mapping):
                continue
            nested_keys = set(nested)
            if not (nested_keys & (root_keys | state_keys)):
                continue
            merged = dict(nested)
            for key in root_keys:
                if key in candidate and key not in merged:
                    merged[key] = candidate[key]
            candidate = merged
            break

    if "state_update" not in root_keys or "state_update" in candidate:
        return candidate

    # Accept only known structural aliases. The aliased payload still has to
    # satisfy the exact state_update schema below.
    for alias in _STATE_UPDATE_ALIAS_KEYS:
        nested = candidate.get(alias)
        if isinstance(nested, Mapping):
            candidate["state_update"] = dict(nested)
            candidate.pop(alias, None)
            return candidate

    # Models frequently omit the state_update envelope while returning the
    # correct patch fields. Move only schema-declared fields into the envelope;
    # missing required fields remain missing and will be rejected normally.
    direct_state_keys = [key for key in state_keys if key in candidate]
    if direct_state_keys:
        candidate["state_update"] = {
            key: candidate.pop(key) for key in direct_state_keys
        }
    return candidate


def _local_output_validator(output_type: type):
    output_adapter = TypeAdapter(output_type)

    def validate_local_output(value: Any) -> Any:
        normalized = normalize_model_output_shape(value, output_type)
        try:
            validated = output_adapter.validate_python(normalized)
        except PydanticValidationError as exc:
            issues = exc.errors(include_url=False)
            issue = issues[0]
            field_path = "$"
            for part in issue.get("loc", ()):
                if isinstance(part, int):
                    field_path += f"[{part}]"
                else:
                    field_path += f".{part}"
            details: list[str] = []
            for row in issues[:5]:
                path = "$"
                for part in row.get("loc", ()):
                    path += f"[{part}]" if isinstance(part, int) else f".{part}"
                details.append(f"{path}: {row.get('msg', 'invalid value')}")
            raise LocalJSONSchemaError(
                "本地 JSON Schema 校验失败：" + "；".join(details),
                candidate=normalized,
                field_path=field_path,
            ) from exc
        model_dump = getattr(validated, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return validated

    return validate_local_output


def get_model_request_timeout_seconds() -> float:
    """Read and validate the per-request model timeout."""

    raw_value = os.getenv(
        "MODEL_REQUEST_TIMEOUT_SECONDS",
        str(DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "MODEL_REQUEST_TIMEOUT_SECONDS must be a number"
        ) from exc
    if timeout_seconds <= 0 or timeout_seconds > 1800:
        raise ValueError(
            "MODEL_REQUEST_TIMEOUT_SECONDS must be greater than 0 "
            "and no greater than 1800"
        )
    return timeout_seconds


def get_model_request_max_attempts() -> int:
    """Read and validate the total attempts for one model request."""

    raw_value = os.getenv(
        "MODEL_REQUEST_MAX_ATTEMPTS",
        str(DEFAULT_MODEL_REQUEST_MAX_ATTEMPTS),
    )
    try:
        max_attempts = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "MODEL_REQUEST_MAX_ATTEMPTS must be an integer"
        ) from exc
    if max_attempts < 1 or max_attempts > 5:
        raise ValueError(
            "MODEL_REQUEST_MAX_ATTEMPTS must be between 1 and 5"
        )
    return max_attempts


def resolve_structured_output_method(model: Any) -> str:
    """Select a structured-output method supported by the configured model."""

    configured = os.getenv("STRUCTURED_OUTPUT_METHOD", "").strip().lower()
    if configured:
        if configured not in SUPPORTED_STRUCTURED_OUTPUT_METHODS:
            allowed = ", ".join(sorted(SUPPORTED_STRUCTURED_OUTPUT_METHODS))
            raise ValueError(
                "STRUCTURED_OUTPUT_METHOD must be one of: "
                f"{allowed}"
            )
        return configured

    model_id = str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or ""
    ).lower()
    if "deepseek" in model_id:
        return "plain_json"
    return "json_schema"


def build_json_output_instruction(output_type: type) -> str:
    """Build the schema instruction required when the API uses JSON mode."""

    parameters = TypeAdapter(output_type).json_schema()
    serialized_schema = json.dumps(
        parameters,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Return exactly one valid JSON object and no Markdown or explanatory "
        "text. The JSON object must match this JSON Schema: "
        f"{serialized_schema}"
    )


def with_compatible_structured_output(
    model: Any,
    output_type: type,
) -> tuple[Any, str]:
    """Wrap a model for structured output and return the selected method."""

    method = resolve_structured_output_method(model)
    validate_local_output = _local_output_validator(output_type)
    if method == "plain_json":
        runnable = (
            model
            | RunnableLambda(parse_model_json_response)
            | RunnableLambda(validate_local_output)
        )
        return runnable, method
    runnable = model.with_structured_output(
        output_type,
        method=method,
    )
    # Native structured-output implementations vary in strictness across
    # OpenAI-compatible providers. Revalidate and normalize their returned
    # Python object so every provider reaches the workflow in one shape.
    return runnable | RunnableLambda(validate_local_output), method
