from __future__ import annotations

import unittest

from Scripts.simulate_qwen_lfu import simulate, summarize


class QwenLFUSimulatorTests(unittest.TestCase):
    def test_belady_oracle_is_event_atomic_and_beats_lru(self):
        trace = {
            "layer_count": 1,
            "expert_blob_size": 10,
            "decode_routes": [[[0], [1], [2], [0], [1]]],
            "decode_misses": [[[1], [1], [1], [1], [1]]],
        }

        result = summarize(trace, "fixture", slots=2, target_tokens_per_second=5)
        policies = {row["policy"]: row for row in result["replacement_policies"]}

        self.assertEqual(policies["lru"]["misses"], 5)
        self.assertEqual(policies["belady_oracle"]["misses"], 4)
        self.assertEqual(policies["belady_oracle"]["expert_bytes_per_token"], 8)
        self.assertEqual(
            policies["belady_oracle"]["required_expert_bandwidth_gbps"],
            0.00000004,
        )

    def test_event_protects_all_selected_experts(self):
        trace = {
            "layer_count": 1,
            "selected_expert_count": 2,
            "expert_blob_size": 1,
            "decode_routes": [[[0, 1], [2, 3], [0, 1]]],
        }

        result = simulate(trace, slots=2, policy="lru")

        self.assertEqual(result["misses"], 6)
        self.assertEqual(result["evictions"], 4)

    def test_replays_recorded_prefill_cache_accesses(self):
        trace = {
            "layer_count": 1,
            "selected_expert_count": 1,
            "expert_blob_size": 1,
            "prefill_cache_accesses": [
                {"layer": 0, "histogram": [2, 1, 0]},
            ],
            "decode_routes": [[[0], [1], [2]]],
            "decode_misses": [[[0], [0], [1]]],
        }

        result = summarize(trace, "fixture", slots=3)

        self.assertTrue(result["current_matches_recorded"])
        self.assertEqual(result["current_simulated_misses"], 1)
        self.assertEqual(result["policies"][0]["prefill_cache_calls"], 1)

    def test_prefill_guided_policy_uses_observed_frequency_without_future_routes(self):
        trace = {
            "layer_count": 1,
            "expert_count": 3,
            "selected_expert_count": 1,
            "expert_blob_size": 1,
            "prefill_histograms": [[2, 1, 0]],
            "prefill_cache_accesses": [
                {"layer": 0, "histogram": [1, 1, 0]},
            ],
            "decode_routes": [[[0], [1], [2], [0]]],
        }

        lru = simulate(trace, slots=2, policy="lru")
        candidate = simulate(trace, slots=2, policy="prefill_guided_24")

        self.assertEqual(lru["misses"], 2)
        self.assertEqual(candidate["misses"], 1)
        self.assertFalse(candidate["uses_future_routes"])

    def test_rejects_nonpositive_target_rate(self):
        trace = {
            "layer_count": 1,
            "expert_blob_size": 1,
            "decode_routes": [[[0]]],
        }

        with self.assertRaisesRegex(ValueError, "target tokens"):
            summarize(trace, "fixture", slots=1, target_tokens_per_second=0)


if __name__ == "__main__":
    unittest.main()
