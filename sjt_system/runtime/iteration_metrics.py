"""Durable per-iteration metric snapshots."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from sjt_system.evaluation.round_results import build_iteration_metrics_snapshot
from sjt_system.runtime.io import write_json_atomic


ITERATION_METRICS_HISTORY_SCHEMA_VERSION = 1

MEASUREMENT_FIELDS = (
    "analysis_round", "development_round", "development_round_batch",
    "measurement_batch", "candidate_count", "item_count", "form_status",
    "item_metrics", "facet_metrics", "selected_item_ids", "form_metrics",
    "item_snapshots", "item_statistics_snapshot", "item_bank_id", "item_bank_version",
    "response_data_ref", "virtual_sample_config",
)


ITEM_METRIC_ANNOTATION_FIELDS = frozenset({
    # These fields are assigned or recomputed after the measurement itself.
    # ``qualified`` is retained in the rendered snapshot for compatibility,
    # but the immutable comparison uses ``measurement_qualified`` below.
    "frozen_qualified", "retention_basis", "qualified",
})


def _measurement_value(field: str, value: Any) -> Any:
    """Exclude disposition annotations derived after measurement commits."""

    if field != "item_metrics" or not isinstance(value, Mapping):
        return value
    normalized: dict[str, Any] = {}
    for item_id, row in value.items():
        if not isinstance(row, Mapping):
            normalized[str(item_id)] = row
            continue
        comparable = {
            key: item_value
            for key, item_value in row.items()
            if key not in ITEM_METRIC_ANNOTATION_FIELDS
        }
        # Older snapshots did not carry the raw qualification separately.
        # Preserve their best available value while making newly written
        # snapshots immune to later disposition-only display changes.
        comparable.setdefault("measurement_qualified", row.get("qualified"))
        normalized[str(item_id)] = comparable
    return normalized


def validate_same_measurement(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Allow annotations, never replace a committed round's measured values."""
    if previous.get("virtual_content_review_protocol") != "mte_cosmin_virtual_content_review_v1":
        raise ValueError("旧轮次指标不能静默包装成虚拟内容复审证据")
    for field in MEASUREMENT_FIELDS:
        if _measurement_value(field, previous.get(field)) != _measurement_value(
            field, current.get(field)
        ):
            raise ValueError(f"已保存轮次的测量结果不可覆盖：{field}")


def _iteration_metrics_output_dir(state: Mapping[str, Any]) -> Path | None:
    """Resolve one run-level directory shared by all response-bank rounds."""

    reference = state.get("virtual_response_data_ref")
    if not isinstance(reference, str) or not reference:
        return None
    manifest_path = Path(reference).expanduser().resolve()
    if not manifest_path.is_file():
        return None
    # Each analysis round can create a new bank-vN directory. Keep the compact
    # iteration ledger above those bank-specific directories so all rounds
    # remain together and no later bank can hide an earlier snapshot.
    return manifest_path.parent.parent / "iteration_metrics"


def persist_iteration_metrics_snapshot(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
) -> str | None:
    """Persist one round and atomically refresh the cumulative metric ledger.

    The checkpoint remains the source of truth for workflow state. These small
    JSON artifacts make the item/facet metric history independently inspectable
    while the workflow is still running and prevent later rounds from hiding
    earlier values.
    """

    output_dir = _iteration_metrics_output_dir(state)
    if output_dir is None:
        return None
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    round_number = int(record.get("analysis_round") or 0)
    if round_number < 1:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_iteration_metrics_snapshot(record, run_id=run_id)
    round_path = output_dir / f"iteration_metrics_round_{round_number:02d}.json"
    immutable = snapshot.get("schema_version") == 2
    if immutable and round_path.is_file():
        previous = json.loads(round_path.read_text(encoding="utf-8"))
        validate_same_measurement(previous, snapshot)

    history_path = output_dir / "iteration_metrics_history.json"
    rounds: list[dict[str, Any]] = []
    if history_path.is_file():
        try:
            existing = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"无法读取已有轮次指标历史：{history_path}"
            ) from exc
        if not isinstance(existing, Mapping):
            raise ValueError(f"轮次指标历史必须是对象：{history_path}")
        if existing.get("run_id") not in (None, run_id):
            raise ValueError("轮次指标历史与当前运行ID不一致")
        if immutable:
            for row in existing.get("rounds") or []:
                if int(row.get("analysis_round") or 0) == round_number:
                    validate_same_measurement(row, snapshot)
        rounds = [
            dict(row)
            for row in existing.get("rounds") or []
            if isinstance(row, Mapping)
            and int(row.get("analysis_round") or 0) != round_number
        ]
    rounds.append(snapshot)
    rounds.sort(key=lambda row: int(row.get("analysis_round") or 0))
    write_json_atomic(round_path, snapshot)
    write_json_atomic(
        history_path,
        {
            "schema_version": 2 if immutable else ITERATION_METRICS_HISTORY_SCHEMA_VERSION,
            "run_id": run_id,
            "rounds": rounds,
        },
    )
    return str(round_path.resolve())
