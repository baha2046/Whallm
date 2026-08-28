from __future__ import annotations

import unittest

from Scripts.benchmark_page_cache_probe import _probe_accounting


def _row(
    *,
    logical: int,
    classified: int,
    resident: int,
    nonresident: int,
    unclassified: int = 0,
    failures: int = 0,
) -> dict:
    return {
        "id": "fixture",
        "metrics": {
            "request_expert_bytes_read": logical,
            "request_expert_page_cache_classified_bytes": classified,
            "request_expert_page_cache_resident_bytes_before_read": resident,
            "request_expert_page_cache_nonresident_bytes_before_read": nonresident,
            "request_expert_page_cache_unclassified_bytes": unclassified,
            "request_expert_page_cache_probe_calls": 2,
            "request_expert_page_cache_probe_failures": failures,
            "request_process_disk_bytes_read": 7,
        },
    }


class PageCacheProbeBenchmarkTests(unittest.TestCase):
    def test_accounting_accepts_an_exact_partition(self):
        accounting = _probe_accounting(
            _row(logical=100, classified=100, resident=40, nonresident=60)
        )

        self.assertTrue(accounting["classified_partition_exact"])
        self.assertTrue(accounting["logical_coverage_exact"])

    def test_accounting_keeps_unclassified_bytes_visible(self):
        accounting = _probe_accounting(
            _row(
                logical=100,
                classified=80,
                resident=30,
                nonresident=50,
                unclassified=20,
                failures=1,
            )
        )

        self.assertTrue(accounting["classified_partition_exact"])
        self.assertTrue(accounting["logical_coverage_exact"])
        self.assertEqual(accounting["unclassified_bytes"], 20)
        self.assertEqual(accounting["probe_failures"], 1)


if __name__ == "__main__":
    unittest.main()
