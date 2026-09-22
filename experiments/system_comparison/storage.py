"""Atomic experiment saves; immutable item snapshots and editable progress."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import csv
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter, sleep
from uuid import uuid4

from .config import ExperimentConfig, fingerprint, make_participants, profile_for


def _replace_with_retry(temporary, path, *, attempts=8, base_delay=0.1):
    """Retry atomic replace on transient Windows file locks.

    Antivirus and the search indexer can briefly hold a handle on the target
    file, which makes os.replace fail with WinError 5 (access denied) or 32
    (sharing violation). Retry briefly instead of failing the experiment.
    """
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            transient = isinstance(exc, PermissionError) or winerror in (5, 32)
            if not transient or attempt == attempts - 1:
                raise
            sleep(base_delay * (attempt + 1))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    _replace_with_retry(temporary, path)


class ExperimentStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._timers = {}
        raw_config = self.read("config.json")
        self.config = ExperimentConfig.model_validate(raw_config)
        # Hash the frozen file rather than the current model dump.  This keeps
        # historical experiments readable when a new optional config field is
        # introduced, while still rejecting any edit to their saved config.
        if fingerprint(raw_config) != self.read("progress.json")["config_hash"]:
            raise ValueError("实验配置已改变；请创建新实验，不得覆盖原实验")
        for role, expected in self.read("progress.json").get("sample_hashes", {}).items():
            if fingerprint(self.read(f"participants/{role}.json")) != expected:
                raise ValueError("已冻结的虚拟被试数据被更改；不得静默继续实验")
        if fingerprint(self.read("model_roles.json")) != self.read("progress.json").get("model_roles_hash"):
            raise ValueError("已冻结的模型角色配置被更改")

    @classmethod
    def create(cls, output_root, config):
        config = config.resolved()
        # Validate both samples before creating an experiment directory.
        samples = {role: make_participants(config, role) for role in ("development", "evaluation")}
        from .models import model_settings
        roles = model_settings(config)
        root = Path(output_root).resolve() / ("exp_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
        root.mkdir(parents=True)
        write_json(root / "config.json", config.model_dump())
        write_json(root / "model_roles.json", roles)
        write_json(root / "progress.json", {
            "schema_version": 1, "experiment_id": root.name, "stage": "A", "status": "ready",
            "config_hash": fingerprint(config.model_dump()), "completed_stages": [],
            "sample_hashes": {role: fingerprint(sample) for role, sample in samples.items()},
            "model_roles_hash": fingerprint(roles),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        for role, sample in samples.items():
            write_json(root / "participants" / f"{role}.json", sample)
        write_json(root / "shared/construct_profile.json", profile_for(config))
        # A receives this predeclared content contract; B/C share the detailed generated blueprint.
        write_json(root / "shared/content_blueprint.json", {
            "target_facet": config.target_facet, "target_population": config.target_population,
            "final_item_count": config.final_item_count,
            "response_instruction": "你会怎么做？", "score_levels": [1, 2, 3, 4],
            "requirements": ["每题仅测目标facet", "不同情境和决策张力", "避免重复题与社会赞许线索"],
        })
        project = Path(__file__).resolve().parents[2]
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                                      capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        source_files = [*Path(__file__).parent.glob("*.py"),
                        *(project / "sjt_system").rglob("*.py")]
        write_json(root / "provenance.json", {
            "git_revision": revision, "source_hashes": {
                str(p.relative_to(project)): __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                for p in source_files},
            "inference_note": "本地样本种子可复现；远程模型输出不保证逐字复现。",
        })
        return cls(root)

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("实验路径越界")
        return path

    def read(self, relative, default=None):
        path = self.path(relative)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else deepcopy(default)

    def write(self, relative, value):
        write_json(self.path(relative), value)

    def progress(self, **updates):
        current = self.read("progress.json")
        current.update(updates)
        self.write("progress.json", current)

    def save_form(self, key, items, **metadata):
        items = deepcopy(items)
        payload = {"method": key.split("/")[0], "round": key.split("/")[1],
                   "items": items, "fingerprint": fingerprint(items), **metadata}
        previous = self.read(f"{key}/form.json")
        if previous and previous["fingerprint"] != payload["fingerprint"]:
            raise ValueError(f"禁止覆盖已冻结问卷：{key}")
        if not previous:
            self.write(f"{key}/form.json", payload)
        html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
                '<title>情境问卷</title><style>body{max-width:850px;margin:40px auto;font:17px/1.8 sans-serif;padding:20px}section{margin:30px 0}</style>',
                '<h1>情境问卷</h1><p>请选择最符合您实际行为的一项。</p>']
        for index, item in enumerate(items, 1):
            html.append(f"<section><b>{index}. {escape(item.get('scenario', ''))}</b><p>{escape(item.get('response_instruction', ''))}</p>")
            for option in item.get("response_options", []):
                html.append(f"<div>{escape(str(option['option_id']))}. {escape(option['text'])}</div>")
            html.append("</section>")
        html.append("</html>")
        self.path(f"{key}/form.html").write_text("\n".join(html), encoding="utf-8")

    def forms(self):
        return [str(p.parent.relative_to(self.root)).replace("\\", "/")
                for method in ("A", "B", "C")
                for p in sorted(self.path(method).glob("*/form.json"))]

    @contextmanager
    def timer(self, key):
        path = f"{key}/cost.json"
        previous = self.read(path, {})
        start = perf_counter()
        self._timers[key] = (previous.get("wall_seconds", 0), start)
        try:
            yield
        finally:
            current = self.read(path, {})
            current["wall_seconds"] = previous.get("wall_seconds", 0) + perf_counter() - start
            self.write(path, current)
            self._timers.pop(key, None)

    def elapsed(self, key):
        if key in self._timers:
            previous, start = self._timers[key]
            return previous + perf_counter() - start
        return self.read(f"{key}/cost.json", {}).get("wall_seconds")
