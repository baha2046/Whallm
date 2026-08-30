from __future__ import annotations

import json
import unittest

from Scripts.audit_qwen_mtp_checkpoint import (
    AuditError,
    _aligned_payload_bytes,
    _parse_safetensors_header,
)


class QwenMTPCheckpointAuditTests(unittest.TestCase):
    def test_safetensors_header_parser(self):
        header = json.dumps(
            {
                "mtp.fc_hidden.weight": {
                    "dtype": "BF16",
                    "shape": [2560, 2560],
                    "data_offsets": [0, 13107200],
                }
            }
        ).encode()

        parsed = _parse_safetensors_header(
            len(header).to_bytes(8, "little"), header, "test"
        )

        self.assertEqual(parsed["mtp.fc_hidden.weight"]["dtype"], "BF16")

    def test_safetensors_header_parser_rejects_size_mismatch(self):
        with self.assertRaises(AuditError):
            _parse_safetensors_header((2).to_bytes(8, "little"), b"{}\n", "test")

    def test_aligned_payload_bytes_sorts_by_tensor_name(self):
        tensors = [
            {"name": "b", "bytes": 2},
            {"name": "a", "bytes": 3},
        ]

        self.assertEqual(_aligned_payload_bytes(tensors, 4), 6)


if __name__ == "__main__":
    unittest.main()
