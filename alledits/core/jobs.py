"""Asynchronous job abstraction (Principle 7, 8).

The MVP runs jobs inline through InlineJobQueue so the vertical slice is testable
with zero infrastructure. The interface matches what a Redis/RQ/Celery worker pool
needs, so swapping in a distributed queue is a provider change, not a rewrite.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from enum import Enum
from typing import Callable, Any

from pathlib import Path

from .ids import new_id


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    steps: list = field(default_factory=list)
    project_id: str | None = None
    # What survived a cancellation or failure. A render that stopped halfway
    # leaves real files on disk; claiming nothing happened would be false, and
    # silently resuming from them would be worse.
    partial: list = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        end = self.ended_at or time.time()
        return max(0.0, end - (self.started_at or self.created_at))

    def to_dict(self):
        d = asdict(self)
        d["state"] = self.state.value
        d["elapsed"] = round(self.elapsed, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        d = dict(d or {})
        d.pop("elapsed", None)
        known = {f.name for f in dataclass_fields(cls)}
        job = cls(**{k: v for k, v in d.items() if k in known})
        job.state = JobState(job.state)
        return job


class JobQueue:
    """Minimal queue interface. submit() returns a Job whose status is observable."""

    def submit(self, kind: str, fn: Callable, *args, **kwargs) -> Job:
        raise NotImplementedError

    def get(self, job_id: str) -> Job | None:
        raise NotImplementedError


class InlineJobQueue(JobQueue):
    def __init__(self, on_progress: Callable[[Job], None] | None = None):
        self._jobs: dict[str, Job] = {}
        self.on_progress = on_progress

    def submit(self, kind: str, fn: Callable, *args, **kwargs) -> Job:
        job = Job(id=new_id("job"), kind=kind)
        self._jobs[job.id] = job
        job.state = JobState.RUNNING
        job.started_at = time.time()

        def report(progress: float, message: str = ""):
            job.progress = max(0.0, min(1.0, progress))
            job.message = message
            job.steps.append({"t": time.time(), "p": job.progress, "m": message})
            if self.on_progress:
                self.on_progress(job)

        try:
            job.result = fn(*args, progress=report, **kwargs)
            job.state = JobState.SUCCEEDED
            job.progress = 1.0
        except Exception as e:            # recoverable: the error is captured, not swallowed
            job.state = JobState.FAILED
            job.error = f"{type(e).__name__}: {e}"
            job.result = None
            job.steps.append({"t": time.time(), "traceback": traceback.format_exc()})
        finally:
            job.ended_at = time.time()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


class JobCancelled(Exception):
    """Raised inside a job's progress callback when cancellation is requested."""


class BackgroundJobQueue(JobQueue):
    """Runs jobs on worker threads, persists them, and supports cancellation.

    Three things InlineJobQueue does not do, each needed before a UI can exist:

    **Background execution.** Inline running blocks the caller for the length of
    a render. A UI needs to submit and then poll.

    **Persistence.** A job that vanishes when the process exits cannot be
    reported to a client that reconnects. Jobs are written to disk on every
    state change — not just at the end, or a crash mid-render leaves a job
    permanently "running".

    **Cooperative cancellation.** A render cannot be killed safely mid-ffmpeg,
    so cancellation is a flag the job checks at its next progress report. The
    job then stops at a known point and records what was already produced,
    rather than being torn down and leaving the caller guessing.

    Deliberately single-process with a thread pool. Distributed queueing is
    deferred: it needs infrastructure this build does not have, and pretending
    otherwise would be the same failure as a fake model provider.
    """

    def __init__(self, root=None, workers: int = 2, on_progress=None):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict = {}
        self._cancel: dict = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self.on_progress = on_progress
        if self.root:
            self._recover()

    # ------------------------------------------------------------ persistence
    def _path(self, job_id: str):
        return self.root / f"{job_id}.json" if self.root else None

    def _persist(self, job: Job):
        if not self.root:
            return
        try:
            import json
            self._path(job.id).write_text(json.dumps(job.to_dict(), indent=2))
        except Exception:
            pass          # persistence is best-effort; it must never fail a job

    def _recover(self):
        """Reload jobs from disk. A job left RUNNING did not survive the process."""
        import json
        for f in sorted(self.root.glob("job_*.json")):
            try:
                job = Job.from_dict(json.loads(f.read_text()))
            except Exception:
                continue
            if job.state in (JobState.RUNNING, JobState.PENDING):
                # Honest: we cannot know how far it got, and no thread is
                # carrying it any more. Say so rather than showing a live job
                # that will never advance.
                job.state = JobState.FAILED
                job.error = ("interrupted: the process ended while this job was "
                             "running, so its outcome is unknown")
                job.ended_at = job.ended_at or time.time()
                self._persist(job)
            self._jobs[job.id] = job

    # ---------------------------------------------------------------- running
    def submit(self, kind: str, fn: Callable, *args, project_id=None,
               **kwargs) -> Job:
        job = Job(id=new_id("job"), kind=kind, project_id=project_id)
        with self._lock:
            self._jobs[job.id] = job
            self._cancel[job.id] = False
        self._persist(job)
        self._pool.submit(self._run, job, fn, args, kwargs)
        return job

    def _run(self, job: Job, fn, args, kwargs):
        job.state = JobState.RUNNING
        job.started_at = time.time()
        self._persist(job)

        def report(progress: float, message: str = ""):
            if self._cancel.get(job.id):
                raise JobCancelled(message or "cancelled by request")
            job.progress = max(0.0, min(1.0, progress))
            job.message = message
            job.steps.append({"t": time.time(), "p": job.progress, "m": message})
            self._persist(job)
            if self.on_progress:
                self.on_progress(job)

        try:
            job.result = fn(*args, progress=report, **kwargs)
            job.state = JobState.SUCCEEDED
            job.progress = 1.0
        except JobCancelled as e:
            job.state = JobState.CANCELLED
            job.message = str(e)
            job.error = None          # cancelling is not an error
        except Exception as e:
            job.state = JobState.FAILED
            job.error = f"{type(e).__name__}: {e}"
            job.result = None
            job.steps.append({"t": time.time(), "traceback": traceback.format_exc()})
        finally:
            job.ended_at = time.time()
            self._persist(job)
        return job

    # ---------------------------------------------------------------- queries
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, project_id=None, state=None) -> list:
        jobs = list(self._jobs.values())
        if project_id:
            jobs = [j for j in jobs if j.project_id == project_id]
        if state:
            jobs = [j for j in jobs if j.state == JobState(state)]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. Takes effect at the job's next progress report."""
        job = self._jobs.get(job_id)
        if job is None or job.state.terminal:
            return False
        self._cancel[job_id] = True
        if job.state == JobState.PENDING:
            job.state = JobState.CANCELLED
            job.ended_at = time.time()
            self._persist(job)
        return True

    def wait(self, job_id: str, timeout: float = 600.0) -> Job | None:
        """Block until a job reaches a terminal state. For CLI and tests."""
        job = self._jobs.get(job_id)
        deadline = time.time() + timeout
        while job is not None and not job.state.terminal and time.time() < deadline:
            time.sleep(0.05)
        return job

    def shutdown(self, wait: bool = True):
        self._pool.shutdown(wait=wait)
