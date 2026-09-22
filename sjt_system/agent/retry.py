import asyncio
import json
from typing import Any

from httpx import TransportError
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from openai import APIConnectionError, APIStatusError

from sjt_system.agent.client import (
    LocalJSONSchemaError,
    get_model_request_max_attempts,
    get_model_request_timeout_seconds,
)
from sjt_system.runtime.progress import emit_progress
from sjt_system.runtime.telemetry import job_context as telemetry_job_context


# One targeted correction after the initial response. Network retries are
# counted separately by ``ainvoke_model_with_retry``. Keeping this bounded
# prevents a persistent schema mismatch from becoming an open-ended token loop.
MAX_LOCAL_JSON_REPAIR_ATTEMPTS = 1
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}


class ModelOutputValidationError(ValueError):
    """A final model-output failure that preserves the rejected candidate."""

    def __init__(
        self,
        message: str,
        *,
        candidate: Any,
        error_kind: str,
    ) -> None:
        super().__init__(message)
        self.candidate = candidate
        self.error_kind = error_kind


def _is_retryable_status_error(exc: APIStatusError) -> bool:
    status_code = exc.status_code
    return (
        status_code in RETRYABLE_HTTP_STATUS_CODES
        or status_code >= 500
    )


async def _sleep_before_retry(
    backoff_seconds: float,
    attempt: int,
) -> None:
    if backoff_seconds:
        await asyncio.sleep(backoff_seconds * (2 ** (attempt - 1)))


async def ainvoke_model_with_retry(
    agent: Runnable,
    input_data: dict[str, Any],
    *,
    job_label: str,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
    backoff_seconds: float = 1.0,
) -> Any:
    """Invoke a model with a per-attempt timeout and bounded retries."""

    timeout_seconds = (
        get_model_request_timeout_seconds()
        if timeout_seconds is None
        else timeout_seconds
    )
    max_attempts = (
        get_model_request_max_attempts()
        if max_attempts is None
        else max_attempts
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with telemetry_job_context(job_label, attempt=attempt):
                return await asyncio.wait_for(
                    agent.ainvoke(input_data),
                    timeout=timeout_seconds,
                )
        except TimeoutError as exc:
            if attempt >= max_attempts:
                emit_progress(
                    {
                        "type": "request_timeout",
                        "job_label": job_label,
                        "timeout_seconds": timeout_seconds,
                    }
                )
                raise TimeoutError(
                    f"{job_label} 请求超过 {timeout_seconds:g} 秒，"
                    f"已尝试 {max_attempts} 次"
                ) from exc
            emit_progress(
                {
                    "type": "request_retry",
                    "retry_kind": "network_timeout",
                    "job_label": job_label,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "reason": f"单次请求超过 {timeout_seconds:g} 秒",
                }
            )
            await _sleep_before_retry(backoff_seconds, attempt)
        except (APIConnectionError, TransportError) as exc:
            if attempt >= max_attempts:
                raise ConnectionError(
                    f"{job_label} 连接模型服务失败，"
                    f"已尝试 {max_attempts} 次：{exc}"
                ) from exc
            emit_progress(
                {
                    "type": "request_retry",
                    "retry_kind": "network_connection",
                    "job_label": job_label,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "reason": f"模型服务连接失败：{exc}",
                }
            )
            await _sleep_before_retry(backoff_seconds, attempt)
        except APIStatusError as exc:
            if not _is_retryable_status_error(exc):
                raise
            if attempt >= max_attempts:
                raise ConnectionError(
                    f"{job_label} 模型服务暂时不可用，"
                    f"已尝试 {max_attempts} 次：HTTP {exc.status_code}"
                ) from exc
            emit_progress(
                {
                    "type": "request_retry",
                    "retry_kind": "network_status",
                    "job_label": job_label,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "reason": (
                        "模型服务返回可重试状态："
                        f"HTTP {exc.status_code}"
                    ),
                }
            )
            await _sleep_before_retry(backoff_seconds, attempt)


async def ainvoke_model_with_schema_repair(
    agent: Runnable,
    input_data: dict[str, Any],
    *,
    job_label: str,
    max_schema_repair_attempts: int = MAX_LOCAL_JSON_REPAIR_ATTEMPTS,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
) -> Any:
    """Repair locally parsed JSON that violates the requested schema."""

    request_data = input_data
    last_error: LocalJSONSchemaError | OutputParserException | None = None
    last_feedback: str | None = None
    last_candidate: Any = None
    last_error_kind = "json_syntax"
    previous_failure_signature: tuple[str, str] | None = None
    for repair_attempt in range(max_schema_repair_attempts + 1):
        try:
            return await ainvoke_model_with_retry(
                agent,
                request_data,
                job_label=job_label,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
            )
        except (LocalJSONSchemaError, OutputParserException) as exc:
            last_error = exc
            if isinstance(exc, LocalJSONSchemaError):
                candidate = exc.candidate
                feedback = str(exc)
                error_kind = "json_schema"
            else:
                candidate = exc.llm_output
                feedback = (
                    "JSON 解析失败：输出不是合法 JSON。"
                    '字符串内部的双引号必须写成 \\"，'
                    "也可以改用中文引号。"
                )
                error_kind = "json_syntax"
            last_feedback = feedback
            last_candidate = candidate
            last_error_kind = error_kind
            try:
                serialized_candidate = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            except (TypeError, ValueError):
                serialized_candidate = repr(candidate)
            failure_signature = (error_kind, serialized_candidate)
            if failure_signature == previous_failure_signature:
                emit_progress(
                    {
                        "type": "output_repair_stalled",
                        "retry_kind": error_kind,
                        "job_label": job_label,
                        "attempt": repair_attempt + 1,
                        "max_attempts": max_schema_repair_attempts + 1,
                        "reason": "模型连续返回相同的无效结构，停止继续抽样",
                    }
                )
                break
            previous_failure_signature = failure_signature
            if repair_attempt >= max_schema_repair_attempts:
                break
            task_input = request_data.get("input_data")
            if not isinstance(task_input, dict):
                raise
            emit_progress(
                {
                    "type": "output_repair",
                    "retry_kind": error_kind,
                    "job_label": job_label,
                    "attempt": repair_attempt + 2,
                    "max_attempts": max_schema_repair_attempts + 1,
                    "reason": feedback,
                }
            )
            request_data = {
                **request_data,
                "input_data": {
                    **task_input,
                    "validation_feedback": feedback,
                    "previous_invalid_candidate": candidate,
                },
            }

    raise ModelOutputValidationError(
        "JSON 输出自动修复"
        f"{max_schema_repair_attempts}次后仍未通过："
        f"{last_feedback or last_error}",
        candidate=last_candidate,
        error_kind=last_error_kind,
    ) from last_error
