from __future__ import annotations

import unittest

from Scripts.audit_mtp1_contract import (
    INPUT_ADAPTER_SUFFIXES,
    OUTPUT_HEAD_SUFFIXES,
    SOURCE_CONTRACT_SNIPPETS,
    _source_contract,
    _stage_contract,
)


def _names(stage: int, suffixes: set[str]) -> list[str]:
    return [f"mtp.{stage}.{suffix}" for suffix in sorted(suffixes)]


class MTP1ContractAuditTests(unittest.TestCase):
    def test_split_input_and_output_ownership_is_not_self_contained(self):
        names = [
            *_names(0, INPUT_ADAPTER_SUFFIXES),
            "mtp.0.attn.wq_a.weight",
            "mtp.1.attn.wq_a.weight",
            *_names(2, OUTPUT_HEAD_SUFFIXES),
            "mtp.2.attn.wq_a.weight",
        ]

        stages = _stage_contract(names, 3)

        self.assertTrue(stages[0]["input_adapter_complete"])
        self.assertFalse(stages[0]["output_head_complete"])
        self.assertFalse(stages[1]["self_contained_single_stage"])
        self.assertFalse(stages[2]["input_adapter_complete"])
        self.assertTrue(stages[2]["output_head_complete"])
        self.assertFalse(any(stage["self_contained_single_stage"] for stage in stages))

    def test_complete_one_stage_contract_is_detected(self):
        names = _names(0, INPUT_ADAPTER_SUFFIXES | OUTPUT_HEAD_SUFFIXES)

        stages = _stage_contract(names, 1)

        self.assertTrue(stages[0]["self_contained_single_stage"])

    def test_reference_graph_contract_requires_all_stage_edges(self):
        source = "\n".join(SOURCE_CONTRACT_SNIPPETS.values())

        checks = _source_contract(source)

        self.assertTrue(all(checks.values()))
        self.assertTrue(checks["does_not_consume_num_nextn_predict_layers"])

    def test_transformers_metadata_is_not_an_inference_graph_edge(self):
        source = "\n".join(SOURCE_CONTRACT_SNIPPETS.values())
        source += "\nnum_nextn_predict_layers"

        checks = _source_contract(source)

        self.assertFalse(checks["does_not_consume_num_nextn_predict_layers"])


if __name__ == "__main__":
    unittest.main()
