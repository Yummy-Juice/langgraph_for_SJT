"""Three tiny concurrent requests; no retries or research workflow execution."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from sjt_system.agent.client import get_model
from sjt_system.agent.json_parsing import parse_model_json_response
from sjt_system.runtime.io import write_json_atomic


async def run(output: Path) -> int:
    model = get_model(temperature=0, thinking_type="disabled")
    runnable = model.bind(max_tokens=64)
    started = perf_counter()
    active = 0
    peak = 0
    rows = []

    async def probe(index: int) -> dict:
        nonlocal active, peak
        row = {"probe": index, "started_seconds": perf_counter() - started}
        active += 1
        peak = max(peak, active)
        try:
            response = await asyncio.wait_for(runnable.ainvoke([
                ("system", "Return only the exact requested JSON, with no other text."),
                ("human", json.dumps({"ok": True, "probe": index})),
            ]), timeout=60)
            parsed = parse_model_json_response(response)
            row["passed"] = parsed == {"ok": True, "probe": index}
            row["usage"] = response.usage_metadata
            if not row["passed"]:
                row["error"] = "Model output did not match the tiny JSON contract"
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            active -= 1
            row["finished_seconds"] = perf_counter() - started
        return row

    try:
        rows = await asyncio.gather(*(probe(index) for index in range(1, 4)))
    finally:
        await model.http_async_client.aclose()
        model.http_client.close()
    report = {
        "model": model.model_name, "requested_calls": 3, "retries": 0,
        "max_output_tokens_per_call": 64, "request_timeout_seconds": 60,
        "peak_in_flight": peak,
        "all_started_before_first_completion": max(row["started_seconds"] for row in rows)
        < min(row["finished_seconds"] for row in rows),
        "passed": peak == 3 and all(row["passed"] for row in rows),
        "elapsed_seconds": perf_counter() - started, "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return asyncio.run(run(parser.parse_args().output))


if __name__ == "__main__":
    raise SystemExit(main())
