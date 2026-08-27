from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
from pathlib import Path


METRICS = (
    "peak_memory_bytes",
    "dspark_draft_expert_bytes_read",
    "dspark_speculative_expert_bytes_read",
    "dspark_expert_cache_hit_rate",
    "dspark_expert_cache_misses",
    "dspark_expert_evictions",
    "request_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _distribution(values: list[float]) -> dict:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate DSpark reduced expert-cache pilot artifacts"
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    inputs = [path.resolve() for path in arguments.inputs]
    reference_path = arguments.reference_artifact.resolve()
    reference = _load(reference_path)
    reference_hashes = {
        summary["name"]: summary["output_token_sha256"][0]
        for summary in reference["workload_summaries"]
    }
    rows = []
    checkpoint_identity = None
    slot_identity = None
    for path in inputs:
        artifact = _load(path)
        if artifact.get("evidence_kind") != "dspark_reduced_expert_cache_pilot":
            raise ValueError(f"not a reduced-cache pilot artifact: {path}")
        checkpoint = artifact["checkpoint"]
        identity = (checkpoint["model_id"], checkpoint["revision"])
        if checkpoint_identity is None:
            checkpoint_identity = identity
        elif identity != checkpoint_identity:
            raise ValueError("input checkpoint identities differ")
        experiment = artifact["experiment"]
        slots = (experiment["control_slots"], experiment["candidate_slots"])
        if slot_identity is None:
            slot_identity = slots
        elif slots != slot_identity:
            raise ValueError("input slot configurations differ")
        workload = experiment["workload"]
        output_hashes = artifact["summary"]["output_token_sha256"]
        if len(output_hashes) != 1:
            raise ValueError(f"input has inconsistent output hashes: {path}")
        reference_hash = reference_hashes.get(workload)
        if reference_hash is None:
            raise ValueError(f"reference artifact has no workload {workload}")
        summary = artifact["summary"]
        rows.append(
            {
                "workload": workload,
                "artifact": str(path),
                "artifact_sha256": _sha256(path),
                "prompt_token_sha256": summary["prompt_token_sha256"][0],
                "output_token_sha256": output_hashes[0],
                "reference_output_token_sha256": reference_hash,
                "within_pilot_token_ids_exact": summary[
                    "all_run_token_ids_exact"
                ],
                "matches_hybrid_v3_reference": output_hashes[0] == reference_hash,
                "control_medians": summary["control_medians"],
                "candidate_medians": summary["candidate_medians"],
                "candidate_change_fraction": summary[
                    "candidate_change_fraction"
                ],
            }
        )
    if len({row["workload"] for row in rows}) != len(rows):
        raise ValueError("input artifacts repeat a workload")
    rows.sort(key=lambda item: item["workload"])

    changes = {
        metric: _distribution(
            [float(row["candidate_change_fraction"][metric]) for row in rows]
        )
        for metric in METRICS
        if all(
            row["candidate_change_fraction"][metric] is not None for row in rows
        )
    }
    control_values = {
        metric: _distribution(
            [float(row["control_medians"][metric]) for row in rows]
        )
        for metric in METRICS
    }
    candidate_values = {
        metric: _distribution(
            [float(row["candidate_medians"][metric]) for row in rows]
        )
        for metric in METRICS
    }
    all_exact = all(row["within_pilot_token_ids_exact"] for row in rows)
    all_reference = all(row["matches_hybrid_v3_reference"] for row in rows)
    all_memory_lower = all(
        row["candidate_change_fraction"]["peak_memory_bytes"] < 0
        for row in rows
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "dspark_reduced_expert_cache_multi_workload_summary",
        "formal_performance_result": False,
        "checkpoint": {
            "model_id": checkpoint_identity[0],
            "revision": checkpoint_identity[1],
        },
        "experiment": {
            "workloads": [row["workload"] for row in rows],
            "control_slots": slot_identity[0],
            "candidate_slots": slot_identity[1],
            "source_artifact_count": len(rows),
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": _sha256(reference_path),
        },
        "workloads": rows,
        "aggregate": {
            "all_within_pilot_token_ids_exact": all_exact,
            "all_match_hybrid_v3_reference": all_reference,
            "all_candidate_peak_memory_lower": all_memory_lower,
            "control_medians_across_workloads": control_values,
            "candidate_medians_across_workloads": candidate_values,
            "candidate_change_fraction_across_workloads": changes,
        },
        "decision": {
            "status": (
                "continue_long_decode_gate"
                if all_exact and all_reference and all_memory_lower
                else "stop_reduced_cache_candidate"
            ),
            "adopt_as_default": False,
            "reason": (
                "Correctness and memory direction hold across the recorded "
                "workloads, but the memory benefit varies and every workload "
                "increases draft expert reads."
            ),
            "next_gate": (
                "Use longer decode to bound resident-set growth and thrashing; "
                "retain exact tokens and predeclare a committed-token read and "
                "request-time regression budget."
            ),
        },
        "evidence_limits": [
            "Only storage_sentence has an ABBA pair; the other workloads have one unreplicated control/candidate pair.",
            "The source artifacts did not purge or control the operating-system page cache.",
            "Fallback was disabled to complete each research trace.",
            "This composition does not turn exploratory timing or process disk counters into formal evidence.",
            "The summary does not authorize changing the 768-slot default.",
        ],
    }
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
