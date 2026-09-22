"""Model-call telemetry: token usage, latency, and job attribution.

Every LLM call made through ``get_model()`` (both the main workflow agents
and the virtual-respondent simulation path) is captured by one global
LangChain callback handler and appended to a per-session JSONL ledger under
``outputs/run_telemetry/``.

Attribution is done with ContextVars so concurrent asyncio tasks (virtual
respondents) keep their own job label without cross-talk:

* ``job_context(job_label, attempt=..., iteration=...)`` wraps a single model call.
* ``run_context(run_id)`` is set once per workflow turn at the runner entry.
* ``iteration_context(iteration)`` is copied into the per-call record.

Aggregation helpers at the bottom of this module produce the token / latency
reports used by the engineering dashboard. This module is observation-only:
it never changes workflow state or model output.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from sjt_system.runtime.output_paths import scoped_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEMETRY_ROOT = PROJECT_ROOT / "outputs" / "run_telemetry"

_JOB_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "sjt_telemetry_job",
    default=None,
)
_RUN_CONTEXT: ContextVar[str | None] = ContextVar(
    "sjt_telemetry_run",
    default=None,
)
_ITERATION_CONTEXT: ContextVar[int | None] = ContextVar(
    "sjt_telemetry_iteration",
    default=None,
)

_write_lock = threading.Lock()
_start_times: dict[str, float] = {}
_start_lock = threading.Lock()
_process_session: str | None = None


def current_run_id() -> str | None:
    return _RUN_CONTEXT.get()


@contextmanager
def job_context(
    job_label: str,
    *,
    attempt: int = 1,
    iteration: int | None = None,
) -> Any:
    """Attribute one model call to its job, attempt, and development round."""

    resolved_iteration = (
        _ITERATION_CONTEXT.get()
        if iteration is None
        else int(iteration)
    )
    token = _JOB_CONTEXT.set(
        {
            "job_label": str(job_label),
            "attempt": int(attempt),
            "iteration": resolved_iteration,
        }
    )
    try:
        yield
    finally:
        _JOB_CONTEXT.reset(token)


@contextmanager
def run_context(run_id: str | None) -> Any:
    """Attribute ledger records to one workflow run for its duration."""
    if not run_id:
        yield
        return
    token = _RUN_CONTEXT.set(str(run_id))
    try:
        yield
    finally:
        _RUN_CONTEXT.reset(token)


@contextmanager
def iteration_context(iteration: int | None) -> Any:
    """Attribute model calls to one development iteration."""

    if iteration is None:
        yield
        return
    token = _ITERATION_CONTEXT.set(int(iteration))
    try:
        yield
    finally:
        _ITERATION_CONTEXT.reset(token)


def _telemetry_session() -> str:
    """Session id for the ledger filename; stable within one process."""
    global _process_session
    configured = os.getenv("TELEMETRY_SESSION", "").strip()
    if configured:
        return configured
    if _process_session is None:
        _process_session = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    return _process_session


def ledger_path(
    *,
    session: str | None = None,
    root: str | Path = DEFAULT_TELEMETRY_ROOT,
) -> Path:
    root_path = scoped_output("telemetry", root)
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path / f"calls_{session or _telemetry_session()}.jsonl"


def _append_record(record: dict[str, Any]) -> None:
    path = ledger_path()
    with _write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TelemetryHandler(BaseCallbackHandler):
    """Capture token usage and latency from every ChatOpenAI call."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            with _start_lock:
                _start_times[run_id] = time.perf_counter()

    def _finish(
        self,
        run_id: str,
        *,
        status: str,
        error_kind: str | None,
        llm_output: dict[str, Any] | None = None,
    ) -> None:
        if not run_id:
            return
        with _start_lock:
            started_at = _start_times.pop(run_id, None)
        duration_ms = (
            round((time.perf_counter() - started_at) * 1000)
            if started_at is not None
            else None
        )
        token_usage = (llm_output or {}).get("token_usage") or {}
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        total_tokens = token_usage.get("total_tokens")
        job = _JOB_CONTEXT.get() or {}
        iteration = job.get("iteration")
        if iteration is None:
            iteration = _ITERATION_CONTEXT.get()
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": _RUN_CONTEXT.get(),
            "iteration": iteration,
            "job_label": job.get("job_label"),
            "attempt": job.get("attempt"),
            "model_id": (llm_output or {}).get("model_name"),
            "status": status,
            "error_kind": error_kind,
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": (
                (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                if (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens") is not None
                else token_usage.get("prompt_cache_hit_tokens")
            ),
        }
        _append_record(record)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        llm_output = response.llm_output or {}
        self._finish(run_id, status="success", error_kind=None, llm_output=llm_output)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        self._finish(
            run_id,
            status="error",
            error_kind=type(error).__name__,
        )


# One shared handler; attaching it to every model keeps attribution uniform.
HANDLER = TelemetryHandler()


# ============================================================================
# Aggregation helpers
# ============================================================================


def read_ledger(
    path: str | Path | None = None,
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read one ledger file (or the current session file) into records."""
    if path is None:
        path = ledger_path()
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    if run_id is not None:
        records = [r for r in records if r.get("run_id") == run_id]
    return records


def _safe_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def aggregate_model_calls(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate ledger records by job label, run, and overall status."""
    total = len(records)
    errors = [r for r in records if r.get("status") == "error"]
    by_label: dict[str, dict[str, Any]] = {}
    by_run: dict[str, dict[str, Any]] = {}
    for record in records:
        label = str(record.get("job_label") or "(unknown)")
        run_id = str(record.get("run_id") or "(no-run)")
        for bucket in (by_label.setdefault(label, {}), by_run.setdefault(run_id, {})):
            bucket["calls"] = bucket.get("calls", 0) + 1
            bucket["errors"] = bucket.get("errors", 0) + int(record.get("status") == "error")
            bucket["prompt_tokens"] = bucket.get("prompt_tokens", 0.0) + _safe_number(
                record.get("prompt_tokens")
            )
            bucket["completion_tokens"] = bucket.get("completion_tokens", 0.0) + _safe_number(
                record.get("completion_tokens")
            )
            bucket["total_tokens"] = bucket.get("total_tokens", 0.0) + _safe_number(
                record.get("total_tokens")
            )
            bucket["duration_ms"] = bucket.get("duration_ms", 0.0) + _safe_number(
                record.get("duration_ms")
            )
    return {
        "total_calls": total,
        "error_calls": len(errors),
        "total_prompt_tokens": sum(_safe_number(r.get("prompt_tokens")) for r in records),
        "total_completion_tokens": sum(
            _safe_number(r.get("completion_tokens")) for r in records
        ),
        "total_tokens": sum(_safe_number(r.get("total_tokens")) for r in records),
        "total_duration_ms": sum(_safe_number(r.get("duration_ms")) for r in records),
        "by_job_label": by_label,
        "by_run": by_run,
    }


@dataclass
class ModelCallSummary:
    """One aggregated row for the CSV report."""

    key: str
    calls: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def flatten_aggregation(
    aggregation: dict[str, Any],
    *,
    group_key: str,
) -> list[ModelCallSummary]:
    """Flatten a ``by_*`` bucket into summary rows, sorted by token spend."""
    rows = []
    for key, bucket in aggregation.items():
        rows.append(
            ModelCallSummary(
                key=str(key),
                calls=int(bucket.get("calls", 0)),
                errors=int(bucket.get("errors", 0)),
                prompt_tokens=int(bucket.get("prompt_tokens", 0)),
                completion_tokens=int(bucket.get("completion_tokens", 0)),
                total_tokens=int(bucket.get("total_tokens", 0)),
                duration_ms=int(bucket.get("duration_ms", 0)),
            )
        )
    rows.sort(key=lambda row: row.total_tokens, reverse=True)
    return rows


def aggregate_iteration_calls(
    records: list[dict[str, Any]],
    *,
    iteration: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Summarize token and latency usage for one development iteration."""

    scoped = [
        record
        for record in records
        if record.get("iteration") == int(iteration)
        and (run_id is None or record.get("run_id") == run_id)
    ]
    return {
        "iteration": int(iteration),
        "calls": len(scoped),
        "error_calls": sum(record.get("status") == "error" for record in scoped),
        "prompt_tokens": int(sum(_safe_number(record.get("prompt_tokens")) for record in scoped)),
        "completion_tokens": int(sum(_safe_number(record.get("completion_tokens")) for record in scoped)),
        "total_tokens": int(sum(_safe_number(record.get("total_tokens")) for record in scoped)),
        "duration_ms": int(sum(_safe_number(record.get("duration_ms")) for record in scoped)),
        "data_available": bool(scoped),
    }
