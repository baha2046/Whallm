from __future__ import annotations

import unittest

from Scripts.benchmark_hash_prefetch_page_cache import (
    _accounting_exact,
    _partition,
)


def _exact_partition(logical: int = 10) -> dict:
    return _partition(
        logical=logical,
        classified=logical,
        resident=4 * logical // 10,
        nonresident=logical - 4 * logical // 10,
        unclassified=0,
    )


class HashPrefetchPageCacheBenchmarkTests(unittest.TestCase):
    def test_partition_requires_both_accounting_identities(self):
        exact = _exact_partition()
        incomplete = _partition(
            logical=10,
            classified=9,
            resident=4,
            nonresident=5,
            unclassified=0,
        )

        self.assertTrue(exact["classified_partition_exact"])
        self.assertTrue(exact["logical_coverage_exact"])
        self.assertTrue(incomplete["classified_partition_exact"])
        self.assertFalse(incomplete["logical_coverage_exact"])

    def test_hash_accounting_requires_useful_wasted_closure(self):
        row = {
            "mode": "hash",
            "main_probe_failures": 0,
            "draft_probe_failures": 0,
            "main_target": _exact_partition(),
            "draft": _exact_partition(),
            "hash_prefetch": _exact_partition(),
            "useful": _exact_partition(6),
            "wasted": _exact_partition(4),
            "hash_useful_wasted_partition_exact": True,
        }

        self.assertTrue(_accounting_exact(row))
        row["hash_useful_wasted_partition_exact"] = False
        self.assertFalse(_accounting_exact(row))


if __name__ == "__main__":
    unittest.main()
