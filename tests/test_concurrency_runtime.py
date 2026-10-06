from __future__ import annotations

import asyncio

import pytest


def test_gather_all_starts_every_request_before_any_completes() -> None:
    from sjt_system.runtime.concurrency import gather_all

    async def exercise() -> None:
        count = 8
        started: list[int] = []
        finished: list[int] = []
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def request(index: int) -> int:
            started.append(index)
            if len(started) == count:
                all_started.set()
            await release.wait()
            finished.append(index)
            return index * 10

        task = asyncio.create_task(
            gather_all(*(request(index) for index in range(count)))
        )
        started_together = False
        try:
            await asyncio.wait_for(all_started.wait(), timeout=1)
            started_together = True
            assert finished == []
        except TimeoutError:
            pass
        finally:
            release.set()

        results = await asyncio.wait_for(task, timeout=1)
        assert started_together
        assert started == list(range(count))
        assert finished == list(range(count))
        assert results == [index * 10 for index in range(count)]

    asyncio.run(exercise())


def test_gather_all_drains_siblings_before_propagating_a_child_error() -> None:
    from sjt_system.runtime.concurrency import gather_all

    async def exercise() -> None:
        all_started = asyncio.Event()
        release_sibling = asyncio.Event()
        sibling_finished = asyncio.Event()
        started = 0
        expected_error = RuntimeError("request failed")

        async def mark_started() -> None:
            nonlocal started
            started += 1
            if started == 2:
                all_started.set()

        async def fail() -> None:
            await mark_started()
            await all_started.wait()
            raise expected_error

        async def sibling() -> str:
            await mark_started()
            try:
                await release_sibling.wait()
                return "drained"
            finally:
                sibling_finished.set()

        task = asyncio.create_task(gather_all(fail(), sibling()))
        try:
            await asyncio.wait_for(all_started.wait(), timeout=1)
            await asyncio.sleep(0)
            assert not task.done()
            assert not sibling_finished.is_set()
        finally:
            release_sibling.set()

        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(task, timeout=1)
        assert raised.value is expected_error
        assert sibling_finished.is_set()

    asyncio.run(exercise())


def test_gather_all_propagates_outer_cancellation_to_all_requests() -> None:
    from sjt_system.runtime.concurrency import gather_all

    async def exercise() -> None:
        count = 4
        all_started = asyncio.Event()
        all_cancelled = asyncio.Event()
        started = 0
        cancelled: list[int] = []

        async def request(index: int) -> None:
            nonlocal started
            started += 1
            if started == count:
                all_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(index)
                if len(cancelled) == count:
                    all_cancelled.set()

        task = asyncio.create_task(
            gather_all(*(request(index) for index in range(count)))
        )
        await asyncio.wait_for(all_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert all_cancelled.is_set()
        assert sorted(cancelled) == list(range(count))

    asyncio.run(exercise())


def test_unlimited_concurrency_enters_all_contexts_without_a_cap() -> None:
    from sjt_system.runtime.concurrency import UnlimitedConcurrency

    async def exercise() -> None:
        count = 12
        gate = UnlimitedConcurrency()
        all_entered = asyncio.Event()
        release = asyncio.Event()
        entered = 0

        async def request() -> None:
            nonlocal entered
            async with gate:
                entered += 1
                if entered == count:
                    all_entered.set()
                await release.wait()

        async def run_all() -> None:
            await asyncio.gather(*(request() for _ in range(count)))

        task = asyncio.create_task(run_all())
        try:
            await asyncio.wait_for(all_entered.wait(), timeout=1)
            assert entered == count
        finally:
            release.set()
        await task

    asyncio.run(exercise())


def test_max_concurrency_compatibility_validation_is_integer_and_non_negative() -> None:
    from sjt_system.runtime.concurrency import validate_max_concurrency

    for value in (0, 1, 50_000):
        validate_max_concurrency(value)

    for value in (-1, True, False, 1.0, 2.5):
        with pytest.raises(ValueError):
            validate_max_concurrency(value)


def test_get_model_configures_unlimited_active_http_connections(monkeypatch) -> None:
    import sjt_system.agent.client as client

    constructed_clients: list[tuple[str, object]] = []

    def capture_http_client(kind: str):
        def factory(*, limits):
            constructed_clients.append((kind, limits))
            return {"kind": kind, "limits": limits}

        return factory

    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr(
        client,
        "DefaultHttpxClient",
        capture_http_client("sync"),
    )
    monkeypatch.setattr(
        client,
        "DefaultAsyncHttpxClient",
        capture_http_client("async"),
    )
    monkeypatch.setattr(client, "ChatOpenAI", lambda **kwargs: kwargs)

    model = client.get_model("test-model")

    assert {kind for kind, _ in constructed_clients} == {"sync", "async"}
    for _, limits in constructed_clients:
        assert limits.max_connections is None
        assert limits.max_keepalive_connections == 20
    assert model["http_client"]["limits"].max_connections is None
    assert model["http_async_client"]["limits"].max_connections is None
