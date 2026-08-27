from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


IGNORED_DIRECTORY_NAMES = {
    ".build",
    ".git",
    ".venv",
    "__pycache__",
    "scratch",
}
TRAINED_ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".mlmodel",
    ".mlpackage",
    ".onnx",
    ".pt",
    ".pth",
}
APPROVED_CONTRACT_PATHS = (
    "research/TRAINING_DATA_RIGHTS.md",
    "research/DRAFTER_TRAINING_CONTRACT.md",
    "research/APPROXIMATE_MODE_CONTRACT.md",
)
STORAGE_NATIVE_SPEC_PATH = "research/STORAGE_NATIVE_MODEL_PROJECT.md"
FULL_XCODE = Path("/Applications/Xcode-26.6.0.app/Contents/Developer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_state(project_root: Path, protocol: Path) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], cwd=project_root),
        "working_tree_dirty": bool(
            _command_output(["git", "status", "--porcelain"], cwd=project_root)
        ),
        "tracked_diff_sha256_at_run": hashlib.sha256(diff).hexdigest(),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "predeclared_protocol_sha256_at_run": _sha256(protocol),
    }


def _project_files(project_root: Path) -> list[Path]:
    files = []
    for path in project_root.rglob("*"):
        relative = path.relative_to(project_root)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def _relative_paths(project_root: Path, paths: list[Path]) -> list[str]:
    return sorted(path.relative_to(project_root).as_posix() for path in paths)


def _find_physical_device_claim_fields(paths: list[Path]) -> list[Path]:
    matches = []
    needles = (
        '"physical_device_bytes_read"',
        '"attributable_physical_ssd_bytes"',
    )
    for path in paths:
        if path.suffix != ".json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(needle in text for needle in needles):
            matches.append(path)
    return matches


def _flatten_manifest_tensor_names(manifest: dict[str, Any]) -> list[str]:
    names = [str(item["name"]) for item in manifest.get("commonTensors", [])]
    names.extend(
        str(item["name"])
        for item in manifest.get("dspark", {}).get("commonTensors", [])
    )
    return names


def _tool_path(tool: str, project_root: Path) -> str | None:
    environment = dict(os.environ)
    if FULL_XCODE.is_dir():
        environment["DEVELOPER_DIR"] = str(FULL_XCODE)
    return _command_output(
        ["xcrun", "--find", tool],
        cwd=project_root,
        environment=environment,
    )


def _decision_matrix(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trained_candidate = bool(evidence["trained_candidate_artifacts"])
    training_contract = bool(evidence["approved_training_contracts"])
    coreml_candidate = bool(evidence["coreml_model_artifacts"])
    current_trace = bool(evidence["current_trace_files"])
    physical_counter = bool(evidence["physical_device_claim_fields"])
    storage_spec = bool(evidence["storage_native_project_spec"])
    return {
        "p0_physical_io": {
            "state": (
                "ready_for_attributable_device_validation"
                if physical_counter
                else "instrumentation_boundary_complete_no_attributable_device_counter"
            ),
            "current_scope_closed": not physical_counter,
            "reopen_requires": "a lower-level per-request attributable device-byte source",
        },
        "p0_gpu_boundary": {
            "state": (
                "current_trace_available"
                if current_trace
                else "measurement_deferred_no_current_candidate_bound_trace"
            ),
            "current_scope_closed": not current_trace,
            "reopen_requires": "a named changed boundary, workload/token hash, and current trace",
        },
        "p1_probation_deadline": {
            "state": "upstream_blocked_no_eligible_trained_predictor",
            "current_scope_closed": True,
            "reopen_requires": "an eligible trained predictor that passes held-out useful/wasted byte gates",
        },
        "p2_dense_drafter": {
            "state": (
                "training_candidate_ready"
                if trained_candidate and training_contract
                else "prerequisite_closed_missing_trained_candidate_or_data_contract"
            ),
            "current_scope_closed": not (trained_candidate and training_contract),
            "reopen_requires": "a V4-specific trained artifact plus approved data/split/training contract",
        },
        "p3_ane": {
            "state": (
                "candidate_ready_for_coreml_gate"
                if trained_candidate and coreml_candidate
                else "upstream_blocked_no_resident_dense_coreml_candidate"
            ),
            "current_scope_closed": not (trained_candidate and coreml_candidate),
            "reopen_requires": "a resident dense candidate converted to Core ML, then placement/concurrency measurements",
        },
        "p3_native_zero_allocation": {
            "state": (
                "boundary_ready_for_prototype"
                if current_trace
                else "measurement_deferred_no_selected_trace_boundary"
            ),
            "current_scope_closed": not current_trace,
            "reopen_requires": "one trace-selected stable boundary before any native rewrite",
        },
        "p4_router_locality": {
            "state": (
                "non_equivalent_training_scope_defined"
                if training_contract
                else "non_equivalent_prerequisite_closed_no_training_or_safety_contract"
            ),
            "current_scope_closed": not training_contract,
            "reopen_requires": "approved data, train/validation/held-out splits, immutable weights, capability and safety gates",
        },
        "p4_mixed_miss_experts": {
            "state": (
                "approximate_mode_contract_present"
                if training_contract
                else "non_equivalent_prerequisite_closed_no_approximate_mode_contract"
            ),
            "current_scope_closed": not training_contract,
            "reopen_requires": "an explicit approximate-mode contract and quality/safety evaluation",
        },
        "p4_storage_native_model": {
            "state": (
                "separate_project_spec_present"
                if storage_spec
                else "separate_project_prerequisite_closed_no_model_program_spec"
            ),
            "current_scope_closed": not storage_spec,
            "reopen_requires": "a separate data/compute/architecture/objective/evaluation program",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit prerequisites for the remaining PLAN research directions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    protocol = Path(arguments.protocol).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    manifest_path = model / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_files = _project_files(project_root)
    coreml_package_directories = [
        path
        for path in project_root.rglob("*.mlpackage")
        if path.is_dir()
        and not any(
            part in IGNORED_DIRECTORY_NAMES
            for part in path.relative_to(project_root).parts
        )
    ]
    tensor_names = _flatten_manifest_tensor_names(manifest)
    trained_artifacts = [
        path
        for path in project_files
        if path.suffix.lower() in TRAINED_ARTIFACT_SUFFIXES
    ]
    coreml_artifacts = [
        path
        for path in project_files
        if path.suffix.lower() in {".mlmodel", ".mlpackage"}
    ] + coreml_package_directories
    trained_artifacts.extend(coreml_package_directories)
    trace_files = [path for path in project_files if path.suffix == ".trace"]
    approved_contracts = [
        project_root / relative
        for relative in APPROVED_CONTRACT_PATHS
        if (project_root / relative).is_file()
    ]
    physical_claim_fields = _find_physical_device_claim_fields(project_files)
    storage_spec = project_root / STORAGE_NATIVE_SPEC_PATH
    direct_transfer_artifact = (
        project_root
        / "docs/benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json"
    )
    route_labels = None
    if direct_transfer_artifact.is_file():
        raw = json.loads(direct_transfer_artifact.read_text(encoding="utf-8"))
        candidates = raw.get("gate", {}).get("candidates", [])
        route_labels = (
            candidates[0].get("target_assignments") if candidates else None
        )

    evidence = {
        "trained_candidate_artifacts": _relative_paths(
            project_root, trained_artifacts
        ),
        "coreml_model_artifacts": _relative_paths(project_root, coreml_artifacts),
        "approved_training_contracts": _relative_paths(
            project_root, approved_contracts
        ),
        "current_trace_files": _relative_paths(project_root, trace_files),
        "physical_device_claim_fields": _relative_paths(
            project_root, physical_claim_fields
        ),
        "storage_native_project_spec": (
            [STORAGE_NATIVE_SPEC_PATH] if storage_spec.is_file() else []
        ),
        "direct_transfer_feasibility_artifact": (
            direct_transfer_artifact.relative_to(project_root).as_posix()
            if direct_transfer_artifact.is_file()
            else None
        ),
        "direct_transfer_assignment_count": route_labels,
    }
    decisions = _decision_matrix(evidence)
    tool_paths = {
        tool: _tool_path(tool, project_root)
        for tool in ("xctrace", "metal", "coremlcompiler")
    }
    packages = {
        name: bool(importlib.util.find_spec(name))
        for name in ("coremltools", "torch", "datasets", "accelerate")
    }
    dense_names = [
        name
        for name in tensor_names
        if any(
            marker in name.lower()
            for marker in ("drafter", "dflash", "redrafter", "hyperdflash", "predictor")
        )
    ]
    criteria = {
        "installed_manifest_read": bool(manifest.get("revision")),
        "dspark_contract_recorded": bool(manifest.get("dspark")),
        "all_remaining_directions_classified": len(decisions) == 9,
        "reopen_requirement_present_for_every_direction": all(
            bool(item.get("reopen_requires")) for item in decisions.values()
        ),
        "local_evidence_does_not_claim_a_trained_dense_candidate": not trained_artifacts
        and not dense_names,
        "tool_presence_kept_separate_from_candidate_readiness": bool(
            tool_paths.get("coremlcompiler")
        )
        and decisions["p3_ane"]["state"]
        == "upstream_blocked_no_resident_dense_coreml_candidate",
    }
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_kind": "local prerequisite feasibility audit",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol),
        "checkpoint": {
            "model_path": str(model),
            "model_id": manifest.get("modelID"),
            "revision": manifest.get("revision"),
            "manifest_sha256": _sha256(manifest_path),
            "expert_blob_size": manifest.get("expertBlobSize"),
            "expert_count": manifest.get("expertCount"),
            "selected_expert_count": manifest.get("selectedExpertCount"),
            "dspark_layer_count": manifest.get("dspark", {}).get("layerCount"),
            "dspark_block_size": manifest.get("dspark", {}).get("blockSize"),
            "separately_named_dense_drafter_tensor_count": len(dense_names),
            "separately_named_dense_drafter_tensors": dense_names,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "full_xcode_developer_dir": (
                str(FULL_XCODE) if FULL_XCODE.is_dir() else None
            ),
            "tool_paths": tool_paths,
            "system_observation_tools": {
                tool: shutil.which(tool)
                for tool in ("fs_usage", "iostat", "powermetrics")
            },
            "python_packages_present": packages,
        },
        "local_evidence": evidence,
        "decisions": decisions,
        "gate": {"criteria": criteria, "passed": all(criteria.values())},
        "decision": (
            "remaining_plan_directions_prerequisite_closed"
            if all(criteria.values())
            else "prerequisite_audit_incomplete"
        ),
        "evidence_limits": [
            "Absence is established only for this checkout, installed model, and named contract paths at audit time.",
            "Tool availability does not prove ANE placement, performance, energy benefit, or concurrent GPU behavior.",
            "The 6000 direct-transfer labels are a bounded feasibility artifact, not a rights-approved training dataset.",
            "No physical-device-byte or current per-operation GPU timing claim is created by this audit.",
            "Prerequisite closure is a scoped stop/defer decision, not a claim that the broad research direction is impossible.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    if not artifact["gate"]["passed"]:
        raise SystemExit("PLAN prerequisite audit failed")


if __name__ == "__main__":
    main()
