"""Server-owned App + Python physical-footprint monitoring, independent of MLX.

Retain the peak between UI polls. Work requests switch to 10 ms sampling;
status/health polling and idle time do not. All timings are nominal.
"""
from contextlib import contextmanager
import threading
import uuid

from .app_memory import physical_footprint, app_process_ids, SAMPLE_INTERVAL_SECONDS

IDLE_INTERVAL_SECONDS = 1.0


class StatusMemorySampler:
    def __init__(self, *, reader=None, pids=None):
        self.pids = app_process_ids() if pids is None else tuple(dict.fromkeys(pids))
        self.scope = "app" if len(self.pids) > 1 else "process"
        self._reader = physical_footprint if reader is None else reader
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._thread = None
        self._epoch = uuid.uuid4().hex
        self._current = None
        self._peak = None
        self._failed = False

    def _sample_locked(self):
        try:
            values = [self._reader(pid) for pid in self.pids]
            if not values or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("physical footprint unavailable")
            self._current = sum(values)
            self._peak = max(self._peak or 0, self._current)
        except Exception:
            self._current = None
            # Once a sample is missed the maximum is unknown until reset, even
            # if later current readings recover. Never substitute RSS or zero.
            self._failed = True

    def _snapshot_locked(self):
        return dict(current_app_memory_bytes=self._current,
                    peak_app_memory_bytes=None if self._failed else self._peak,
                    memory_scope=self.scope, epoch=self._epoch,
                    sample_interval_seconds=SAMPLE_INTERVAL_SECONDS if self._active else IDLE_INTERVAL_SECONDS,
                    active_requests=self._active)

    def snapshot(self):
        # Reading the status endpoint must not itself trigger extra sampling.
        with self._condition:
            return self._snapshot_locked()

    def reset(self):
        with self._condition:
            self._epoch = uuid.uuid4().hex
            self._current = self._peak = None
            self._failed = False
            if not self._closed:
                self._sample_locked()
            return self._snapshot_locked()

    def start(self):
        with self._condition:
            if self._closed or self._thread is not None:
                raise RuntimeError("memory monitor cannot be started twice")
            self._sample_locked()
            self._thread = threading.Thread(target=self._run, name="status-memory-sampler", daemon=True)
            self._thread.start()
        return self

    def _run(self):
        with self._condition:
            while not self._closed:
                interval = SAMPLE_INTERVAL_SECONDS if self._active else IDLE_INTERVAL_SECONDS
                if self._condition.wait(interval):
                    # Activity transitions wake an idle wait immediately.
                    continue
                if not self._closed:
                    self._sample_locked()

    @contextmanager
    def activity(self):
        with self._condition:
            self._active += 1
            if not self._closed:
                self._sample_locked()
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                if not self._closed:
                    self._sample_locked()
                self._active -= 1
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join()
