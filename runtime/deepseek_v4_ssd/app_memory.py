"""Sample macOS physical footprint, not RSS or MLX allocator statistics.

ABI: macOS SDK sys/resource.h (rusage_info_v0), libproc.h (proc_pid_rusage).
The App supplies its PID in the child environment; never count a shell parent.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading

SAMPLE_INTERVAL_SECONDS = 0.01


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
        )
    ]


if sys.platform == "darwin":
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    _pid_rusage = _libproc.proc_pid_rusage
    _pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _pid_rusage.restype = ctypes.c_int
else:
    _pid_rusage = None


def physical_footprint(pid: int) -> int | None:
    if _pid_rusage is None:
        return None
    info = _RUsageInfoV0()
    if _pid_rusage(pid, 0, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint)


def app_process_ids() -> tuple[int, ...]:
    runtime_pid = os.getpid()
    try:
        app_pid = int(os.environ.get("WHALLM_APP_PID", ""))
    except ValueError:
        return (runtime_pid,)
    if app_pid > 1 and app_pid != runtime_pid and app_pid == os.getppid():
        return (runtime_pid, app_pid)
    return (runtime_pid,)


class AppMemorySampler:
    """One trial, including loading; sum each sample before taking the maximum.

    A failed read invalidates the measurement instead of reporting a partial
    process total. Sequential PID reads approximate a simultaneous sample.
    """

    def __init__(self, *, reader=None, pids=None, interval=SAMPLE_INTERVAL_SECONDS):
        self.pids = app_process_ids() if pids is None else tuple(dict.fromkeys(pids))
        self.scope = "app" if len(self.pids) > 1 else "process"
        self._reader = physical_footprint if reader is None else reader
        self._interval = interval
        self._peak = 0
        self._failed = False
        self._stop = threading.Event()
        self._thread = None

    @property
    def peak_bytes(self) -> int | None:
        return None if self._failed else self._peak

    def sample(self):
        try:
            values = [self._reader(pid) for pid in self.pids]
            if not values or any(value is None or value < 0 for value in values):
                self._failed = True
            else:
                self._peak = max(self._peak, sum(values))
        except Exception:
            self._failed = True

    def _run(self):
        while not self._stop.wait(self._interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, name="app-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def finish(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self.sample()
        return self.peak_bytes

    def __exit__(self, *_):
        self.finish()
