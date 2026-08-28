from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts"))

from audit_plan_prerequisites import _decision_matrix  # noqa: E402


def _evidence(**overrides) -> dict:
    evidence = {
        "trained_candidate_artifacts": [],
        "approved_training_contracts": [],
        "coreml_model_artifacts": [],
        "current_trace_files": [],
        "physical_device_claim_fields": [],
        "storage_native_project_spec": [],
    }
    evidence.update(overrides)
    return evidence


class PlanPrerequisiteAuditTests(unittest.TestCase):
    def test_missing_candidates_close_training_and_platform_scopes(self):
        decisions = _decision_matrix(_evidence())

        self.assertEqual(
            decisions["p2_dense_drafter"]["state"],
            "prerequisite_closed_missing_trained_candidate_or_data_contract",
        )
        self.assertEqual(
            decisions["p3_ane"]["state"],
            "upstream_blocked_no_resident_dense_coreml_candidate",
        )
        self.assertEqual(
            decisions["p0_physical_io"]["state"],
            "instrumentation_boundary_complete_no_attributable_device_counter",
        )
        self.assertTrue(
            all(item["current_scope_closed"] for item in decisions.values())
        )

    def test_tools_or_training_contract_alone_do_not_invent_candidates(self):
        decisions = _decision_matrix(
            _evidence(approved_training_contracts=["research/contract.md"])
        )

        self.assertEqual(
            decisions["p2_dense_drafter"]["state"],
            "prerequisite_closed_missing_trained_candidate_or_data_contract",
        )
        self.assertEqual(
            decisions["p3_ane"]["state"],
            "upstream_blocked_no_resident_dense_coreml_candidate",
        )
        self.assertFalse(
            decisions["p4_router_locality"]["current_scope_closed"]
        )

    def test_candidate_and_contract_open_only_the_corresponding_gates(self):
        decisions = _decision_matrix(
            _evidence(
                trained_candidate_artifacts=["candidate.ckpt"],
                approved_training_contracts=["research/contract.md"],
                coreml_model_artifacts=["candidate.mlpackage"],
                current_trace_files=["candidate.trace"],
                physical_device_claim_fields=["device.json"],
                storage_native_project_spec=["research/storage.md"],
            )
        )

        self.assertEqual(
            decisions["p2_dense_drafter"]["state"],
            "training_candidate_ready",
        )
        self.assertEqual(
            decisions["p3_ane"]["state"],
            "candidate_ready_for_coreml_gate",
        )
        self.assertEqual(
            decisions["p3_native_zero_allocation"]["state"],
            "boundary_ready_for_prototype",
        )
        self.assertEqual(
            decisions["p0_physical_io"]["state"],
            "ready_for_attributable_device_validation",
        )


if __name__ == "__main__":
    unittest.main()
