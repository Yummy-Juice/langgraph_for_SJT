"""Refresh stale complete-profile option evidence from saved SJT responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2
from uuid import uuid4

from sjt_system.evaluation.psychometrics import (
    OPTION_CHOICE_DIAGNOSTICS_VERSION,
    refresh_saved_option_diagnostics,
)
from sjt_system.runtime.checkpoint import save_run_checkpoint


def refresh_checkpoint(path: Path) -> dict[str, object]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload["state"]
    run_id = state["run_id"]
    if payload.get("run_id") != run_id or path != (path.parent / f"{run_id}.json").resolve():
        raise ValueError("检查点文件名或运行ID不匹配")
    manifest_path = Path(state["virtual_response_data_ref"])
    artifact_dir = manifest_path.parent / "psychometrics"
    artifacts = (
        artifact_dir / "option_choice_diagnostics.json",
        artifact_dir / "option_statistics.csv",
        artifact_dir / "analysis_manifest.json",
    )
    if not all(source.is_file() for source in artifacts):
        raise ValueError("分析产物不完整，不能刷新旧选项诊断")
    statistics = state.get("item_statistics") or {}
    stale_count = sum(
        isinstance(row, dict)
        and (row.get("quality_evaluation") or {}).get("virtual_target_specificity", {}).get(
            "correlation_scope"
        ) == "complete_matched_profile"
        and (row.get("option_choice_diagnostics") or {}).get("version")
        != OPTION_CHOICE_DIAGNOSTICS_VERSION
        for row in statistics.values()
    )
    if not stale_count:
        return {"run_id": run_id, "refreshed_items": 0, "backup_dir": None}

    original_stat = path.stat()
    backup_dir = path.parent / "option_diagnostics_backups" / f"{run_id}-{uuid4().hex[:8]}"
    backup_dir.mkdir(parents=True)
    for source in (path, *artifacts):
        copy2(source, backup_dir / source.name)
    if (path.stat().st_mtime_ns, path.stat().st_size) != (
        original_stat.st_mtime_ns, original_stat.st_size
    ):
        raise ValueError("备份期间检查点发生变化，已停止刷新")

    refreshed = refresh_saved_option_diagnostics(state, artifact_dir=artifact_dir)
    if (path.stat().st_mtime_ns, path.stat().st_size) != (
        original_stat.st_mtime_ns, original_stat.st_size
    ):
        raise ValueError("刷新期间检查点发生变化，已停止覆盖检查点；备份已保留")
    save_run_checkpoint(refreshed, checkpoint_root=path.parent)
    return {"run_id": run_id, "refreshed_items": stale_count, "backup_dir": str(backup_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(refresh_checkpoint(args.checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
