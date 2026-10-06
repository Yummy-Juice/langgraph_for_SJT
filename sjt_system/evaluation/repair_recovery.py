"""Bounded, journaled recovery for repair-only model calls."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone

POLICY = "bounded_repair_recovery_v2"
MAX_CALLS = 5


class RepairStopped(RuntimeError):
    """A durable terminal boundary, not another automatic retry request."""


def error_category(exc):
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429} or isinstance(status, int) and status >= 500:
        return "transient_service"
    if isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
        "APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
    }:
        return "transient_service"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "invalid_output"
    return "unrecoverable"


def now():
    return datetime.now(timezone.utc).isoformat()


async def journaled_call(*, records, key, kind, payload, invoke, validate, save,
                         prompt="", model=None, retry_automatically=True):
    """Reuse results and persist each attempt before its next await.

    Legacy and current journal entries share one durable five-call budget.
    """
    previous = [r for r in records if r.get("key") == key]
    for record in reversed(previous):
        if record.get("status") == "completed":
            return deepcopy(record["validated_output"])
        if "raw_output" in record and record.get("status") in {"validating", "received", "error"}:
            try:
                result = validate(deepcopy(record["raw_output"]))
            except (ValueError, KeyError, TypeError):
                continue
            record.update(status="completed", validated_output=deepcopy(result), recovered_at=now())
            save()
            return result
    # A legacy archive may contain a raw model response but no v2 attempt
    # records. It is intentionally offered once as correction input, then the
    # durable v2 budget owns all subsequent retries.
    legacy_raw = next((r.get("raw_output") for r in reversed(previous)
                       if r.get("raw_output") is not None), None)
    call_statuses = {
        "running", "interrupted", "validating", "received", "error",
        "invalid", "content_rejected", "completed",
    }
    call_records = [r for r in previous if r.get("status") in call_statuses]
    policy_attempts = [r for r in call_records if r.get("recovery_policy") == POLICY]
    if call_records and call_records[-1].get("error_category") == "unrecoverable":
        raise RepairStopped(call_records[-1].get("error", "unrecoverable repair error"))
    feedback = previous[-1] if previous else None
    if legacy_raw is not None and not policy_attempts:
        feedback = {"error": "legacy output requires v2 validation", "raw_output": legacy_raw,
                    "status": "error", "error_category": "invalid_output"}
    for attempt in range(len(call_records) + 1, MAX_CALLS + 1):
        request = deepcopy(payload)
        if feedback and feedback.get("error") and feedback.get("status") != "interrupted" and feedback.get("error_category") != "transient_service":
            request["validation_feedback"] = {"error": feedback["error"],
                "previous_invalid_output": deepcopy(feedback.get("raw_output")),
                "instruction": "Correct only the reported output contract; do not invent evidence."}
            if "allowed_kinds" in payload:
                request["validation_feedback"]["allowed_kinds"] = payload["allowed_kinds"]
        record = {"key": key, "kind": kind, "input": request, "prompt": prompt,
                  "model": deepcopy(model or {}), "recovery_policy": POLICY,
                  "attempt": attempt, "status": "running", "started_at": now()}
        records.append(record)
        save()
        try:
            raw = await invoke(kind, request)
            record.update(raw_output=deepcopy(raw), status="validating")
            save()
            result = validate(raw)
            record.update(status="completed", validated_output=deepcopy(result), completed_at=now())
            save()
            return result
        except asyncio.CancelledError:
            record.update(status="interrupted", error="cancelled", completed_at=now())
            save()
            raise
        except Exception as exc:
            if getattr(exc, "candidate", None) is not None:
                record["raw_output"] = deepcopy(exc.candidate)
            category = error_category(exc)
            record.update(status="error", error=str(exc), error_type=type(exc).__name__,
                          error_category=category, completed_at=now())
            save()
            if category == "unrecoverable":
                raise RepairStopped(str(exc)) from exc
            if not retry_automatically:
                raise RepairStopped(f"{kind}: {exc}; saved for explicit resume") from exc
            feedback = record
            if attempt < MAX_CALLS:
                from sjt_system.runtime.progress import emit_progress
                emit_progress({"type": "output_repair", "retry_kind": category,
                               "job_label": kind, "attempt": attempt + 1,
                               "max_attempts": MAX_CALLS, "reason": str(exc)})
                if category == "transient_service":
                    await asyncio.sleep(2 ** attempt)
    raise RepairStopped(f"{kind}: exhausted {MAX_CALLS} model calls; {feedback.get('error') if feedback else 'interrupted'}")
