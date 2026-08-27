from __future__ import annotations

import unittest

from Scripts.benchmark_expert_file_cache_bypass import _range_contract_passed


def _row(blob_size: int = 100) -> dict:
    return {
        "before": {"resident_bytes": 0, "nonresident_bytes": blob_size},
        "bypass_read": {"bytes_read": blob_size, "sha256": "same"},
        "after_bypass": {
            "resident_bytes": 0,
            "nonresident_bytes": blob_size,
        },
        "cached_read": {"bytes_read": blob_size, "sha256": "same"},
        "after_cached": {
            "resident_bytes": blob_size,
            "nonresident_bytes": 0,
        },
    }


class ExpertFileCacheBypassBenchmarkTests(unittest.TestCase):
    def test_range_contract_accepts_exact_bypass_then_cached_transition(self):
        self.assertTrue(_range_contract_passed(_row(), 100))

    def test_range_contract_rejects_cache_pollution_after_bypass(self):
        row = _row()
        row["after_bypass"]["resident_bytes"] = 4
        row["after_bypass"]["nonresident_bytes"] = 96

        self.assertFalse(_range_contract_passed(row, 100))

    def test_range_contract_rejects_byte_hash_mismatch(self):
        row = _row()
        row["cached_read"]["sha256"] = "different"

        self.assertFalse(_range_contract_passed(row, 100))


if __name__ == "__main__":
    unittest.main()
