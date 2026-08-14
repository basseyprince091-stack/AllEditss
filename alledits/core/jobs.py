"""Asynchronous job abstraction (Principle 7, 8).

The MVP runs jobs inline through InlineJobQueue so the vertical slice is testable
with zero infrastructure. The interface matches what a Redis/RQ/Celery worker pool
needs, so swapping in a distributed queue is a provider change, not a rewrite.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Any

from .ids import new_id


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    steps: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["state"] = self.state.value
        return d


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
