import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts"))
from benchmark_qwen_nohint_validation import pair_contract
from analyze_qwen_nohint_validation import summarize
from benchmark_qwen_nohint_cache import matching_entry, tensor_payload
from benchmark_qwen_nohint_integration import performance_summary


class PairContractTests(unittest.TestCase):
    def test_incomplete_integration_cannot_pass(self):
        with self.assertRaises(ValueError):
            performance_summary([None] * 7)

    def test_tensor_identity_ignores_file_order_but_not_values(self):
        with tempfile.TemporaryDirectory() as temp:
            results = []
            for i, order in enumerate((("a", "b"), ("b", "a"), ("a", "b"))):
                header = {"__metadata__": {"state": "{}"}}
                for offset, name in enumerate(order):
                    header[name] = {"dtype": "U8", "shape": [1], "data_offsets": [offset, offset+1]}
                encoded = json.dumps(header).encode()
                path = Path(temp) / str(i)
                values = b"ac" if i == 2 else "".join(order).encode()
                path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + values)
                results.append(tensor_payload(path))
            self.assertEqual(results[0], results[1])
            self.assertNotEqual(results[0], results[2])

    def test_cache_match_requires_a_strict_prefix(self):
        entries = [SimpleNamespace(tokens=ids) for ids in ([1, 2], [1, 2, 3], [1, 9], [1, 2, 3, 4])]
        self.assertEqual(matching_entry(entries, [1, 2, 3, 4]), 3)
        self.assertEqual(matching_entry(entries, [7, 2, 3, 4]), 0)

    def test_incomplete_extension_cannot_pass(self):
        for status, count in (("running", 32), ("stopped_on_contract", 2), ("completed", 31)):
            with self.assertRaises(ValueError):
                summarize({"status": status, "runs": [None] * count, "prompts": [None] * 8})

    def test_output_and_io_must_independently_match(self):
        control = {"metrics": {"generated_token_ids": [2, 3, 5],
                               "request_expert_bytes_read": 1024,
                               "request_gather_qmm_calls": 96}}
        self.assertTrue(all(pair_contract(control, control).values()))
        for field, value in (("generated_token_ids", [2, 3, 7]),
                             ("request_expert_bytes_read", 1023),
                             ("request_gather_qmm_calls", 94)):
            candidate = copy.deepcopy(control)
            candidate["metrics"][field] = value
            self.assertEqual(sum(pair_contract(control, candidate).values()), 2)


if __name__ == "__main__":
    unittest.main()
