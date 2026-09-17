from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from deepseek_v4_ssd.app_memory import (
    AppMemorySampler, _RUsageInfoV0, app_process_ids, physical_footprint,
)


class AppMemoryTests(unittest.TestCase):
    def test_default_sampling_wait_is_ten_milliseconds(self):
        sampler = AppMemorySampler(reader=lambda _: 50, pids=(11,))
        with patch.object(sampler._stop, "wait", side_effect=[False, True]) as wait:
            sampler._run()
        self.assertEqual(wait.call_count, 2)
        for call in wait.call_args_list:
            self.assertEqual(call.args, (0.01,))
        self.assertEqual(sampler.peak_bytes, 50)

    def test_sums_each_sample_not_independent_process_peaks(self):
        values = iter([100, 10, 20, 200, 30, 40])
        sampler = AppMemorySampler(reader=lambda _: next(values), pids=(11, 22))
        for _ in range(3):
            sampler.sample()
        self.assertEqual(sampler.peak_bytes, 220)  # Not 100 + 200.
        self.assertEqual(sampler.scope, "app")

    def test_read_failure_invalidates_total_even_after_success(self):
        for failure in (None, -1):
            values = iter([100, 10, 200, failure, 100, 10])
            sampler = AppMemorySampler(reader=lambda _: next(values), pids=(11, 22))
            for _ in range(3):
                sampler.sample()
            self.assertIsNone(sampler.peak_bytes)
        sampler = AppMemorySampler(reader=lambda _: 1 / 0, pids=(11,))
        sampler.sample()
        self.assertIsNone(sampler.peak_bytes)

    def test_initial_final_samples_and_new_trial_reset(self):
        values = iter([10, 100, 20, 30])
        reader = lambda _: next(values)
        with AppMemorySampler(reader=reader, pids=(11,), interval=60) as first:
            self.assertEqual(first.peak_bytes, 10)
        self.assertEqual(first.peak_bytes, 100)
        self.assertEqual(first.finish(), 100)  # Idempotent cleanup.
        with AppMemorySampler(reader=reader, pids=(11,), interval=60) as second:
            pass
        self.assertEqual(second.peak_bytes, 30)
        self.assertEqual(second.scope, "process")

    def test_background_sampling_and_exception_cleanup(self):
        sampled = threading.Event()
        def reader(_):
            if threading.current_thread().name == "app-memory-sampler":
                sampled.set()
            return 50
        sampler = AppMemorySampler(reader=reader, pids=(11,), interval=0.001)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with sampler:
                worker = sampler._thread
                self.assertTrue(sampled.wait(2))
                raise RuntimeError("cancelled")
        self.assertFalse(worker.is_alive())
        self.assertEqual(sampler.peak_bytes, 50)

    def test_parent_is_only_counted_when_explicitly_marked_by_app(self):
        with patch("os.getpid", return_value=22), patch("os.getppid", return_value=11):
            for value in ("", "bad", "1", "0", "-1", "22", "33"):
                with patch.dict(os.environ, {"WHALLM_APP_PID": value}):
                    self.assertEqual(app_process_ids(), (22,))
            with patch.dict(os.environ, {"WHALLM_APP_PID": "11"}):
                self.assertEqual(app_process_ids(), (22, 11))

    def test_libproc_failure_is_not_zero_or_rss(self):
        with patch("deepseek_v4_ssd.app_memory._pid_rusage", return_value=-1):
            self.assertIsNone(physical_footprint(11))
        with patch("deepseek_v4_ssd.app_memory._pid_rusage", None):
            self.assertIsNone(physical_footprint(11))

    @unittest.skipUnless(sys.platform == "darwin", "macOS physical footprint")
    def test_real_libproc_and_app_parent_in_child_process(self):
        self.assertEqual(ctypes.sizeof(_RUsageInfoV0), 96)
        self.assertGreater(physical_footprint(os.getpid()), 0)
        self.assertIsNone(physical_footprint(-1))
        code = """
import os
from deepseek_v4_ssd.app_memory import AppMemorySampler
with AppMemorySampler() as sample:
    assert sample.pids == (os.getpid(), os.getppid()), sample.pids
    assert sample.scope == 'app'
assert sample.peak_bytes > 0, sample.peak_bytes
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=10,
                       env={**os.environ, "WHALLM_APP_PID": str(os.getpid())})
