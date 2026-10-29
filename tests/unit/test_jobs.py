"""Unit tests for the SQLite job store (submit/get_status/cancel trio semantics).

Covers the full lifecycle and the guarded transitions: terminal-state immutability,
cancel only from pending/running, unknown-ID handling, and reset_for_test file removal.
"""

import json
from typing import Any

import pytest

from src.errors import JobStateError
from src.tools.jobs import JobStatus, JobsStore


async def _fresh_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> JobsStore:
    """Point the store at a per-test DB and return a fresh singleton."""
    monkeypatch.setenv("JOBS_DB", str(tmp_path / "jobs.db"))
    await JobsStore.reset_for_test()
    return await JobsStore.get_instance()


async def test_lifecycle_create_pending_then_running(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)

    job_id = await store.create_job(json.dumps({"description": "build a thing"}), description="demo job")

    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.PENDING
    assert job["payload"] == json.dumps({"description": "build a thing"})
    assert job["description"] == "demo job"
    assert job["result"] is None
    assert job["error"] is None
    assert not await store.is_cancelled(job_id)

    await store.set_running(job_id)
    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.RUNNING


async def test_set_completed_stores_result_json(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)
    job_id = await store.create_job("{}")
    await store.set_running(job_id)

    result_payload = json.dumps({"overview": {"reasoning": "won on fit"}})
    await store.set_completed(job_id, result_payload)

    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.COMPLETED
    # Re-read returns the stored result and it parses back to the original JSON.
    assert job["result"] == result_payload
    assert json.loads(str(job["result"])) == {"overview": {"reasoning": "won on fit"}}


async def test_set_failed_stores_error(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)
    job_id = await store.create_job("{}")
    await store.set_running(job_id)

    await store.set_failed(job_id, "ERR_007: generation failed after retry")

    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.FAILED
    assert job["error"] == "ERR_007: generation failed after retry"


async def test_cancel_from_pending_and_running(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)

    pending_id = await store.create_job("{}")
    await store.set_cancelled(pending_id)
    assert await store.is_cancelled(pending_id)
    pending = await store.get_job(pending_id)
    assert pending is not None
    assert pending["status"] == JobStatus.CANCELLED

    running_id = await store.create_job("{}")
    await store.set_running(running_id)
    await store.set_cancelled(running_id)
    assert await store.is_cancelled(running_id)


async def test_cancel_after_completion_raises_and_status_stays(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)
    job_id = await store.create_job("{}")
    await store.set_running(job_id)
    await store.set_completed(job_id, '{"done": true}')

    with pytest.raises(JobStateError):
        await store.set_cancelled(job_id)

    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.COMPLETED
    assert not await store.is_cancelled(job_id)


async def test_set_running_on_cancelled_job_raises(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)
    job_id = await store.create_job("{}")
    await store.set_cancelled(job_id)

    with pytest.raises(JobStateError):
        await store.set_running(job_id)

    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.CANCELLED


async def test_terminal_states_are_immutable(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)
    job_id = await store.create_job("{}")
    await store.set_running(job_id)
    await store.set_failed(job_id, "boom")

    with pytest.raises(JobStateError):
        await store.set_running(job_id)
    with pytest.raises(JobStateError):
        await store.set_completed(job_id, '{"late": true}')
    with pytest.raises(JobStateError):
        await store.set_failed(job_id, "again")
    with pytest.raises(JobStateError):
        await store.set_cancelled(job_id)

    job = await store.get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.FAILED
    assert job["error"] == "boom"


async def test_get_job_unknown_id_returns_none(monkeypatch, tmp_path):
    store = await _fresh_store(monkeypatch, tmp_path)
    assert await store.get_job("nonexistent-id") is None
    assert not await store.is_cancelled("nonexistent-id")


async def test_reset_for_test_removes_db_file(monkeypatch, tmp_path):
    db_file = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB", str(db_file))
    await JobsStore.reset_for_test()

    store = await JobsStore.get_instance()
    await store.create_job("{}")
    assert db_file.exists()

    await JobsStore.reset_for_test()
    assert not db_file.exists()

    # After reset the singleton is gone; a fresh store sees no jobs.
    fresh = await JobsStore.get_instance()
    assert await fresh.get_job("anything") is None


async def test_configure_is_used_when_env_unset(monkeypatch, tmp_path):
    db_file = tmp_path / "configured" / "jobs.db"
    monkeypatch.delenv("JOBS_DB", raising=False)
    await JobsStore.reset_for_test()
    JobsStore.configure(str(db_file))
    try:
        store = await JobsStore.get_instance()
        job_id = await store.create_job("{}")
        job = await store.get_job(job_id)
        assert job is not None
        assert db_file.exists()
    finally:
        JobsStore.configure(str(tmp_path / "discard.db"))
        await JobsStore.reset_for_test()
async def test_close_is_idempotent_and_drops_the_singleton(monkeypatch, tmp_path):
    """Teardown contract: close() releases the connection, and a second close() is a no-op.

    The server's cleanup path calls close() unconditionally, so it must survive both the
    "connection is open" and the "already closed" state without raising.
    """
    store = await _fresh_store(monkeypatch, tmp_path)
    assert store._db is not None

    await store.close()
    assert store._db is None
    assert JobsStore._instance is None

    await store.close()  # must not raise on an already-closed store
    assert store._db is None
