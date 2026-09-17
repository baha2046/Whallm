import threading
import unittest

from deepseek_v4_ssd.status_memory import StatusMemorySampler


class StatusMemoryTests(unittest.TestCase):
    def test_sums_contemporaneous_footprints_and_retains_idle_peak(self):
        values = {1: 100, 2: 10}
        sampler = StatusMemorySampler(reader=values.get, pids=(1, 2))
        first = sampler.reset()
        values.update({1: 20, 2: 200})
        with sampler.activity():
            state = sampler.snapshot()
            self.assertEqual(state['current_app_memory_bytes'], 220)
            self.assertEqual(state['peak_app_memory_bytes'], 220)  # Not 100 + 200.
            self.assertEqual(state['memory_scope'], 'app')
            self.assertEqual(state['sample_interval_seconds'], 0.01)
        values.update({1: 1, 2: 2})
        with sampler.activity():
            pass
        state = sampler.snapshot()
        self.assertEqual(state['current_app_memory_bytes'], 3)
        self.assertEqual(state['peak_app_memory_bytes'], 220)
        self.assertEqual(state['sample_interval_seconds'], 1)
        self.assertEqual(state['active_requests'], 0)
        reset = sampler.reset()
        self.assertNotEqual(reset['epoch'], first['epoch'])
        self.assertEqual(reset['peak_app_memory_bytes'], 3)

    def test_failed_read_never_falls_back_or_hides_a_missing_peak(self):
        values = {1: 100, 2: None}
        sampler = StatusMemorySampler(reader=values.get, pids=(1, 2))
        state = sampler.reset()
        self.assertIsNone(state['current_app_memory_bytes'])
        self.assertIsNone(state['peak_app_memory_bytes'])
        values[2] = 10
        with sampler.activity():
            state = sampler.snapshot()
            self.assertEqual(state['current_app_memory_bytes'], 110)
            self.assertIsNone(state['peak_app_memory_bytes'])
        self.assertEqual(sampler.reset()['peak_app_memory_bytes'], 110)
        self.assertEqual(StatusMemorySampler(reader=lambda _: 7, pids=(1,)).reset()['memory_scope'], 'process')
        self.assertIsNone(StatusMemorySampler(reader=lambda _: 7, pids=()).reset()['peak_app_memory_bytes'])

    def test_nested_and_failed_requests_restore_idle_only_after_last_exit(self):
        sampler = StatusMemorySampler(reader=lambda _: 5, pids=(1, 2))
        with sampler.activity():
            with self.assertRaisesRegex(ValueError, 'injected'):
                with sampler.activity():
                    self.assertEqual(sampler.snapshot()['active_requests'], 2)
                    self.assertEqual(sampler.reset()['sample_interval_seconds'], 0.01)
                    self.assertEqual(sampler.snapshot()['active_requests'], 2)
                    raise ValueError('injected')
            self.assertEqual(sampler.snapshot()['active_requests'], 1)
            self.assertEqual(sampler.snapshot()['sample_interval_seconds'], 0.01)
        self.assertEqual(sampler.snapshot()['active_requests'], 0)
        self.assertEqual(sampler.snapshot()['sample_interval_seconds'], 1)

    def test_active_transition_wakes_idle_wait_and_preserves_unpolled_spike(self):
        high_seen = threading.Event()
        low_seen = threading.Event()
        value = [1]
        reads = [0]
        def read(_):
            reads[0] += 1
            if value[0] == 80:
                high_seen.set()
            elif high_seen.is_set():
                low_seen.set()
            return value[0]
        sampler = StatusMemorySampler(reader=read, pids=(1,)).start()
        try:
            # GET-style snapshots do not wake the worker or read processes.
            initial = reads[0]
            for _ in range(100):
                sampler.snapshot()
            self.assertEqual(reads[0], initial)
            with sampler.activity():
                value[0] = 80
                self.assertTrue(high_seen.wait(0.5))  # Shorter than idle's 1 second.
                value[0] = 1
                self.assertTrue(low_seen.wait(0.5))
            state = sampler.snapshot()
            self.assertEqual(state['current_app_memory_bytes'], 1)
            self.assertEqual(state['peak_app_memory_bytes'], 80)
            self.assertEqual(state['sample_interval_seconds'], 1)
        finally:
            sampler.close()
        self.assertFalse(sampler._thread.is_alive())
        sampler.close()
        with self.assertRaises(RuntimeError):
            sampler.start()

    def test_idle_sampling_continues_without_status_requests(self):
        sampled = threading.Event()
        reads = [0]
        def read(_):
            reads[0] += 1
            if reads[0] >= 2:
                sampled.set()
            return reads[0]
        sampler = StatusMemorySampler(reader=read, pids=(1,)).start()
        try:
            self.assertFalse(sampled.wait(0.05))
            self.assertTrue(sampled.wait(1.5))
            self.assertGreaterEqual(sampler.snapshot()['current_app_memory_bytes'], 2)
        finally:
            sampler.close()


if __name__ == '__main__':
    unittest.main()
