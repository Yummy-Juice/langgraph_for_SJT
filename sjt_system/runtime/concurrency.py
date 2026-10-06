"""Unbounded fan-out for independent external-model requests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar


T = TypeVar("T")


class UnlimitedConcurrency:
    """Compatibility gate that never queues or limits a request."""

    async def __aenter__(self) -> UnlimitedConcurrency:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


def validate_max_concurrency(value: int) -> None:
    """Accept legacy positive limits, but treat scheduling as unbounded."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_concurrency must be a non-negative integer (0 = unlimited)")


async def gather_all(*requests: Awaitable[T]) -> list[T]:
    """Launch every request, drain siblings, then propagate the first error."""

    outcomes = await asyncio.gather(*requests, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes
