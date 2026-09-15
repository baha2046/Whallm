import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mlx.core as mx
from deepseek_v4_ssd.route_cache import RouteCachePolicy
from deepseek_v4_ssd.expert_cache import ExpertCache
from runtime.tests.test_expert_eviction_policy import fixture


class RouteCacheTests(unittest.TestCase):
    def test_history_survives_eviction_and_prefill_slot_release(self):
        with tempfile.TemporaryDirectory() as d:
            with ExpertCache(fixture(Path(d)), slots=1, eviction_policy='route') as cache:
                for _ in range(12): cache.get_many(0, [0])
                cache.get_many(0, [1])
                self.assertNotIn((0, 0), cache._entries)
                self.assertGreater(cache._route_policy.short[0, 0], 0)
                history = cache._route_policy.long.copy()
                cache.release_prefill_slots()
                np.testing.assert_array_equal(cache._route_policy.long, history)
                cache.get_many(0, [0])
                self.assertGreater(cache._route_policy.long[0, 0], history[0, 0])

    def test_short_history_adapts_and_long_history_remembers(self):
        p = RouteCachePolicy(1, 4, 2, 1)
        p.observe(0, [0] * 256, 'decode')
        p.observe(0, [1] * 32, 'decode')
        self.assertGreater(p.short[0, 1], p.short[0, 0])
        self.assertGreater(p.long[0, 0], p.long[0, 1])
        p.begin_prefill()
        old = p.long.copy()
        p.observe(0, [3] * 10000, 'prefill')
        np.testing.assert_array_equal(p.long, old)
        self.assertLessEqual(p.prefill[0, 3] / p.prefill_tokens[0], 1)

    def test_split_prefill_keeps_one_prior_and_next_request_resets_only_prior(self):
        with tempfile.TemporaryDirectory() as d:
            with ExpertCache(fixture(Path(d)), slots=2, eviction_policy='route') as cache:
                cache.get_many(0, [0])
                history = cache._route_policy.long.copy()
                cache.begin_route_request()
                for routes in ([[1], [2]], [[3]]):
                    with cache.trace_routes('prefill'):
                        cache.observe_batched_routes(0, routes)
                self.assertEqual(cache._route_policy.prefill_tokens.tolist(), [3])
                np.testing.assert_array_equal(cache._route_policy.prefill, [[0, 1, 1, 1]])
                cache.begin_route_request()
                self.assertEqual(cache._route_policy.prefill.sum(), 0)
                np.testing.assert_array_equal(cache._route_policy.long, history)

    def test_batch_observation_matches_tokenwise_and_quota_follows_marginal_benefit(self):
        a, b = [RouteCachePolicy(2, 8, 4, 2) for _ in range(2)]
        routes = np.array([[0, 1], [1, 2], [0, 2]])
        a.observe(0, routes, 'decode')
        for row in routes: b.observe(0, row, 'decode')
        np.testing.assert_allclose(a.short, b.short)
        np.testing.assert_allclose(a.long, b.long)
        # One highly predictable expert needs only one slot; diffuse layer earns three.
        a.short[:] = [[1, 0, 0, 0, 0, 0, 0, 0], [.3, .3, .3, .1, 0, 0, 0, 0]]
        a.long.fill(0);a.rebalance()
        self.assertEqual(a.quotas.tolist(), [1, 3])
        a.short[:] = a.short[::-1].copy();a.rebalance()
        self.assertEqual(a.quotas.tolist(), [3, 1])
        self.assertEqual(int(a.quotas.sum()), 4)
        # All-zero cold start is balanced; no layer is arbitrarily allocated everything.
        self.assertEqual(RouteCachePolicy(3, 8, 8, 1).quotas.tolist(), [3, 3, 2])

    def test_pinning_failure_retry_and_phase_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            with ExpertCache(fixture(Path(d)), slots=3, eviction_policy='route') as cache:
                cache.get_many(0, [0, 1, 2])
                cache._pinned_expert_keys.add((0, 0))
                cache.get_many(0, [1, 3])
                self.assertIn((0, 0), cache._entries)
                cache._pinned_expert_keys.clear()
                with patch.object(cache, '_read_expert_into_slot', side_effect=OSError('read failed')):
                    with self.assertRaises(OSError): cache.get_many(0, [2])
                cache.get_many(0, [2])
                with self.assertRaises(RuntimeError):
                    with cache.trace_routes('prefill'):
                        cache.observe_batched_routes(0, np.array([[2]]))
                        raise RuntimeError('cancelled')
                self.assertEqual(cache._route_phase, 'decode')
                self.assertEqual(len(set(e.slot for e in cache._entries.values())), cache.resident_count)
                self.assertLessEqual(cache.resident_count, 3)

    def test_metadata_bounded_and_invalid_routes_do_not_update(self):
        p = RouteCachePolicy(48, 512, 3072, 10)
        self.assertLess(p.snapshot()['metadata_array_bytes'], 1_000_000)
        for route in ([-1], [512]):
            with self.assertRaises(ValueError): p.observe(0, route, 'decode')
        self.assertEqual(p.decode_tokens.sum(), 0)
        p.record_read(0, float('nan'))
        self.assertEqual(p.read_samples.sum(), 0)

    def test_layer_budget_moves_slots_and_preserves_bytes_under_pressure(self):
        with tempfile.TemporaryDirectory() as d:
            model = fixture(Path(d), layers=2)
            for staged in (False, True):
                with ExpertCache(model, slots=4, eviction_policy='route',
                                 staged_expert_streaming=staged) as cache:
                    cache.get_many(0, [0, 1, 2])
                    cache.get_many(1, [0])
                    cache._route_policy.short[:] = [[1, 0, 0, 0], [.3, .3, .3, 0]]
                    cache._route_policy.long.fill(0)
                    cache._route_policy.rebalance()
                    cache._rebuild_route_heap_locked()
                    cache.get_many(1, [0, 1, 2])
                    self.assertEqual(cache._layer_counts, [1, 3])
                    for step in range(100):
                        layer, expert = step % 2, (step // 2) % 4
                        if staged:
                            for handle in cache.iter_staged_ready(layer, [expert]):
                                handle.finish_w2()
                                weights = handle.weights
                                self.assertEqual(weights.w1.view(mx.uint8).tolist(),
                                                 [[expert * 15 + x for x in range(4)]])
                        else:
                            for _, weights in cache.iter_ready(layer, [expert]):
                                self.assertEqual(weights.w1.view(mx.uint8).tolist(),
                                                 [[expert * 15 + x for x in range(4)]])
                        self.assertEqual(sum(cache._layer_counts), cache.resident_count)
                        self.assertEqual(len({e.slot for e in cache._entries.values()}), cache.resident_count)
                    self.assertGreater(cache._route_policy.read_samples.sum(), 10)
                    self.assertLess(sum(map(len, cache._route_heaps)), 64)

    def test_heap_selection_matches_exhaustive_policy_with_protected_experts(self):
        with tempfile.TemporaryDirectory() as d:
            with ExpertCache(fixture(Path(d), layers=3), slots=5, eviction_policy='route') as cache:
                original = cache._evict_route_locked
                checked = []
                def verify(protected, incoming_layer):
                    choices = []
                    for key, entry in cache._entries.items():
                        if key in protected or key in cache._pinned_expert_keys:
                            continue
                        layer, expert = key
                        count, quota = cache._layer_counts[layer], cache._route_policy.quotas[layer]
                        over = count > quota or (layer == incoming_layer and count >= quota)
                        tier = 2 if layer in cache._pinned_layers else (0 if over else 1)
                        choices.append((tier, cache._route_policy.scores[key], entry.last_access,
                                        entry.version, layer, expert))
                    expected = min(choices)[-2:]
                    actual = original(protected, incoming_layer)
                    self.assertEqual(actual[0], expected)
                    checked.append(expected)
                    return actual
                rng = np.random.default_rng(4)
                with patch.object(cache, '_evict_route_locked', side_effect=verify):
                    for step in range(160):
                        cache._pinned_layers = {step % 3} if step % 2 else set()
                        cache._pinned_expert_keys = set(list(cache._entries)[:1])
                        cache.get_many(step % 3, rng.choice(4, 2, replace=False).tolist())
                self.assertGreater(len(checked), 100)


if __name__ == '__main__': unittest.main()
