"""Durable InMemorySaver adapter using SQLite and LangGraph's own serializer.

Persists checkpoint blobs and pending writes, including model results preceding
human interrupts. No pickle, CLI state reset, or model replay on normal resume.
"""
from base64 import b64encode, b64decode
import json
from pathlib import Path
import sqlite3
from threading import RLock

from langgraph.checkpoint.memory import InMemorySaver


def _encode(value):
    if isinstance(value, bytes):
        return {"bytes": b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"tuple": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict) and "bytes" in value:
        return b64decode(value["bytes"])
    if isinstance(value, dict) and "tuple" in value:
        return tuple(_decode(v) for v in value["tuple"])
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


class DiskSaver(InMemorySaver):
    def __init__(self, path):
        super().__init__()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = RLock()
        self.db.execute("CREATE TABLE IF NOT EXISTS checkpoints (kind TEXT, key TEXT, value TEXT, PRIMARY KEY(kind,key))")
        for kind, key, value in self.db.execute("SELECT kind,key,value FROM checkpoints"):
            key, value = _decode(json.loads(key)), _decode(json.loads(value))
            if kind == "storage":
                self.storage[key[0]][key[1]][key[2]] = value
            elif kind == "writes":
                self.writes[key[0]][key[1]] = value
            elif kind == "blobs":
                self.blobs[key] = value

    def _save(self, kind, key, value):
        self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)",
                        (kind, json.dumps(_encode(key)), json.dumps(_encode(value))))

    def put(self, config, checkpoint, metadata, new_versions):
        with self.lock, self.db:
            result = super().put(config, checkpoint, metadata, new_versions)
            tid = config["configurable"]["thread_id"]
            ns = config["configurable"].get("checkpoint_ns", "")
            cid = checkpoint["id"]
            self._save("storage", (tid, ns, cid), self.storage[tid][ns][cid])
            for channel, version in new_versions.items():
                key = (tid, ns, channel, version)
                self._save("blobs", key, self.blobs[key])
            return result

    def put_writes(self, config, writes, task_id, task_path=""):
        with self.lock, self.db:
            super().put_writes(config, writes, task_id, task_path)
            cfg = config["configurable"]
            outer = (cfg["thread_id"], cfg.get("checkpoint_ns", ""), cfg["checkpoint_id"])
            for inner, value in self.writes[outer].items():
                self._save("writes", (outer, inner), value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()
