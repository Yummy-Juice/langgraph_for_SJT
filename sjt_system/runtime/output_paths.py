"""Opt-in, task-local output isolation for experiment adapters."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ROOT = ContextVar("sjt_output_root", default=None)
_OVERRIDES = ContextVar("sjt_output_overrides", default={})


@contextmanager
def output_scope(root, **overrides):
    token = _ROOT.set(Path(root).resolve())
    override_token = _OVERRIDES.set({k: Path(v).resolve() for k, v in overrides.items()})
    try:
        yield
    finally:
        _OVERRIDES.reset(override_token)
        _ROOT.reset(token)


def scoped_output(category, default):
    if category in _OVERRIDES.get():
        return _OVERRIDES.get()[category]
    root = _ROOT.get()
    return root / category if root is not None else Path(default)
