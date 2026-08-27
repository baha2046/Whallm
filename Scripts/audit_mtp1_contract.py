from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path


PINNED_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
PINNED_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
REFERENCE_FILES = (
    "config.json",
    "inference/config.json",
    "inference/model.py",
    "model.safetensors.index.json",
)
INPUT_ADAPTER_SUFFIXES = {
    "main_norm.weight",
    "main_proj.scale",
    "main_proj.weight",
}
OUTPUT_HEAD_SUFFIXES = {
    "confidence_head.proj.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "norm.weight",
}
SOURCE_CONTRACT_SNIPPETS = {
    "first_stage_owns_input_adapter": "if stage_id == 0:",
    "last_stage_owns_output_heads": (
        "if stage_id == args.n_mtp_layers - 1:"
    ),
    "constructs_configured_stage_count": (
        "for layer_id in range(args.n_mtp_layers):"
    ),
    "enters_through_stage_zero": (
        "h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)"
    ),
    "executes_every_stage": "for layer in self.mtp:",
    "exits_through_last_stage": (
        "output_ids, logits, confidence = self.mtp[-1].forward_head(h, input_ids)"
    ),
}


class AuditError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "DeepSeekV4SSD-MTP1-Contract-Audit/1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise AuditError(f"reference request returned HTTP {response.status}: {url}")
        return response.read()


def _load_json_bytes(data: bytes, source: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot decode JSON from {source}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object from {source}")
    return value


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object in {path}")
    return value


def _stage_suffix(name: str) -> tuple[int, str]:
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "mtp" or not parts[1].isdigit():
        raise AuditError(f"invalid MTP tensor name: {name}")
    return int(parts[1]), parts[2]


def _stage_contract(names: list[str], stage_count: int) -> list[dict]:
    by_stage: dict[int, list[str]] = {stage: [] for stage in range(stage_count)}
    for name in names:
        stage, suffix = _stage_suffix(name)
        if stage not in by_stage:
            raise AuditError(f"MTP tensor references unexpected stage {stage}: {name}")
        by_stage[stage].append(suffix)

    stages = []
    for stage in range(stage_count):
        suffixes = set(by_stage[stage])
        input_present = sorted(INPUT_ADAPTER_SUFFIXES.intersection(suffixes))
        output_present = sorted(OUTPUT_HEAD_SUFFIXES.intersection(suffixes))
        stages.append(
            {
                "stage": stage,
                "tensor_count": len(by_stage[stage]),
                "input_adapter_tensors_present": input_present,
                "input_adapter_complete": set(input_present) == INPUT_ADAPTER_SUFFIXES,
                "output_head_tensors_present": output_present,
                "output_head_complete": set(output_present) == OUTPUT_HEAD_SUFFIXES,
                "self_contained_single_stage": (
                    set(input_present) == INPUT_ADAPTER_SUFFIXES
                    and set(output_present) == OUTPUT_HEAD_SUFFIXES
                ),
            }
        )
    return stages


def _source_contract(source: str) -> dict:
    checks = {
        name: snippet in source
        for name, snippet in SOURCE_CONTRACT_SNIPPETS.items()
    }
    checks["does_not_consume_num_nextn_predict_layers"] = (
        "num_nextn_predict_layers" not in source
    )
    return checks


def _remote_url(model_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{path}"


def _installed_file_records(
    model_root: Path,
    manifest: dict,
    *,
    verify_hashes: bool,
) -> list[dict]:
    declared = {
        item["path"]: item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and str(item.get("path", "")).startswith("dspark/")
    }
    expected = [
        "dspark/common.bin",
        *[
            f"dspark/experts/layer_{stage:02d}.bin"
            for stage in range(int(manifest["dspark"]["layerCount"]))
        ],
    ]
    if set(declared) != set(expected):
        raise AuditError(
            "installed manifest DSpark file set does not match its layer count"
        )

    records = []
    for relative in expected:
        descriptor = declared[relative]
        path = model_root / relative
        if not path.is_file():
            raise AuditError(f"installed DSpark file is missing: {path}")
        actual_size = path.stat().st_size
        expected_size = int(descriptor["size"])
        if actual_size != expected_size:
            raise AuditError(
                f"installed DSpark file has size {actual_size}; "
                f"expected {expected_size}: {path}"
            )
        actual_sha256 = _sha256(path) if verify_hashes else None
        expected_sha256 = descriptor["sha256"]
        if actual_sha256 is not None and actual_sha256 != expected_sha256:
            raise AuditError(f"installed DSpark file hash mismatch: {path}")
        records.append(
            {
                "path": relative,
                "bytes": actual_size,
                "manifest_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "hash_verified": actual_sha256 is not None,
            }
        )
    return records


def audit(
    model_root: Path,
    repo_root: Path,
    *,
    verify_hashes: bool,
) -> dict:
    manifest_path = model_root / "manifest.json"
    config_path = model_root / "config.json"
    inference_config_path = model_root / "inference/config.json"
    manifest = _load_json(manifest_path)
    local_config = _load_json(config_path)
    local_inference_config = _load_json(inference_config_path)

    model_id = manifest.get("modelID")
    revision = manifest.get("revision")
    if model_id != PINNED_MODEL_ID or revision != PINNED_REVISION:
        raise AuditError(
            "installed model does not match the pinned checkpoint contract: "
            f"{model_id}@{revision}"
        )
    dspark = manifest.get("dspark")
    if not isinstance(dspark, dict):
        raise AuditError("installed model has no DSpark descriptor")

    fetched: dict[str, bytes] = {}
    remote_sources = []
    for relative in REFERENCE_FILES:
        url = _remote_url(model_id, revision, relative)
        data = _fetch(url)
        fetched[relative] = data
        remote_sources.append(
            {
                "path": relative,
                "url": url,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )

    remote_config = _load_json_bytes(fetched["config.json"], "remote config.json")
    remote_inference_config = _load_json_bytes(
        fetched["inference/config.json"],
        "remote inference/config.json",
    )
    index = _load_json_bytes(
        fetched["model.safetensors.index.json"],
        "remote model.safetensors.index.json",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AuditError("remote checkpoint index has no weight_map object")
    mtp_names = sorted(name for name in weight_map if name.startswith("mtp."))
    if not mtp_names:
        raise AuditError("remote checkpoint index has no mtp.* tensors")

    stage_count = int(dspark["layerCount"])
    stage_counts = Counter(_stage_suffix(name)[0] for name in mtp_names)
    index_stages = sorted(stage_counts)
    if index_stages != list(range(stage_count)):
        raise AuditError(
            f"remote checkpoint MTP stages {index_stages} do not match "
            f"installed layer count {stage_count}"
        )

    common_names = sorted(item["name"] for item in dspark["commonTensors"])
    expert_regions = [item["name"] for item in manifest["expertRegions"]]
    expected_expert_names = {
        f"mtp.{stage}.ffn.experts.{expert}.{region}"
        for stage in range(stage_count)
        for expert in range(int(manifest["expertCount"]))
        for region in expert_regions
    }
    installed_tensor_contract = set(common_names).union(expected_expert_names)
    remote_tensor_contract = set(mtp_names)
    missing_from_install = sorted(remote_tensor_contract - installed_tensor_contract)
    unexpected_in_install = sorted(installed_tensor_contract - remote_tensor_contract)
    if missing_from_install or unexpected_in_install:
        raise AuditError(
            "installed DSpark tensor contract differs from the pinned checkpoint index"
        )

    source_text = fetched["inference/model.py"].decode("utf-8")
    source_checks = _source_contract(source_text)
    if not all(source_checks.values()):
        failed = sorted(name for name, passed in source_checks.items() if not passed)
        raise AuditError(f"pinned reference graph checks failed: {failed}")

    stages = _stage_contract(mtp_names, stage_count)
    self_contained = [
        item["stage"] for item in stages if item["self_contained_single_stage"]
    ]
    local_config_matches = local_config == remote_config
    local_inference_matches = local_inference_config == remote_inference_config
    if not local_config_matches or not local_inference_matches:
        raise AuditError("installed configuration files differ from the pinned checkpoint")

    contract_values = {
        "transformers_num_nextn_predict_layers": remote_config.get(
            "num_nextn_predict_layers"
        ),
        "inference_n_mtp_layers": remote_inference_config.get("n_mtp_layers"),
        "installed_dspark_layer_count": stage_count,
        "installed_dspark_block_size": int(dspark["blockSize"]),
        "installed_dspark_target_layer_ids": dspark["targetLayerIDs"],
        "installed_dspark_markov_rank": int(dspark["markovRank"]),
    }
    expected_values = {
        "transformers_num_nextn_predict_layers": 1,
        "inference_n_mtp_layers": 3,
        "installed_dspark_layer_count": 3,
        "installed_dspark_block_size": 5,
        "installed_dspark_target_layer_ids": [40, 41, 42],
        "installed_dspark_markov_rank": 256,
    }
    if contract_values != expected_values:
        raise AuditError(
            f"pinned MTP/DSpark contract changed: {contract_values}"
        )
    if self_contained:
        raise AuditError(
            f"unexpected self-contained single MTP stage found: {self_contained}"
        )

    installed_files = _installed_file_records(
        model_root,
        manifest,
        verify_hashes=verify_hashes,
    )
    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "evidence_kind": "mtp1_checkpoint_contract_audit",
        "formal_performance_result": False,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "subject": {
            "model_id": model_id,
            "revision": revision,
            "installed_model": str(model_root),
        },
        "source_provenance": {
            "git_commit": _command_output(["git", "rev-parse", "HEAD"], repo_root),
            "git_status_short": _command_output(
                ["git", "status", "--short"], repo_root
            ),
            "audit_script": str(script_path.relative_to(repo_root)),
            "audit_script_sha256": _sha256(script_path),
            "installed_sources": [
                {
                    "path": str(path.relative_to(model_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in (manifest_path, config_path, inference_config_path)
            ],
            "pinned_remote_sources": remote_sources,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "contract": {
            "values": contract_values,
            "local_config_matches_pinned_remote": local_config_matches,
            "local_inference_config_matches_pinned_remote": local_inference_matches,
            "checkpoint_index_mtp_tensor_count": len(mtp_names),
            "checkpoint_index_mtp_stages": index_stages,
            "checkpoint_index_tensor_counts_by_stage": [
                [stage, stage_counts[stage]] for stage in index_stages
            ],
            "installed_common_tensor_count": len(common_names),
            "installed_expert_tensor_count": len(expected_expert_names),
            "installed_tensor_contract_matches_checkpoint_index": True,
            "missing_from_installed_tensor_contract": missing_from_install,
            "unexpected_in_installed_tensor_contract": unexpected_in_install,
            "stage_ownership": stages,
            "self_contained_single_stage_candidates": self_contained,
            "reference_graph_checks": source_checks,
            "installed_files": installed_files,
            "installed_dspark_total_bytes": sum(
                item["bytes"] for item in installed_files
            ),
        },
        "decision": {
            "status": "faithful_native_mtp1_not_exposed_by_pinned_checkpoint",
            "native_mtp1_checkpoint_exposed": False,
            "runtime_stage_truncation_is_checkpoint_faithful": False,
            "reason": [
                (
                    "num_nextn_predict_layers=1 is not consumed by the pinned "
                    "inference graph; that graph uses n_mtp_layers=3."
                ),
                (
                    "The pinned graph enters through mtp.0, executes all three "
                    "stages, and exits through mtp.2."
                ),
                (
                    "mtp.0 alone has the target-hidden input adapter but no output "
                    "head; mtp.2 has the output head but no input adapter."
                ),
                (
                    "Stitching mtp.0 directly to mtp.2 heads or changing block size "
                    "would define a graph outside the pinned reference contract, "
                    "not a faithful MTP-1 baseline."
                ),
            ],
            "adoption": "reject_runtime_truncation",
            "next_valid_options": [
                (
                    "Obtain a checkpoint whose reference contract exposes a "
                    "self-contained one-stage MTP drafter."
                ),
                (
                    "Train or distill a new one-stage resident drafter and classify "
                    "it as a new checkpoint candidate."
                ),
                (
                    "Continue optimizing the complete three-stage DSpark graph "
                    "without labeling it MTP-1."
                ),
            ],
        },
        "evidence_limits": [
            "This is a checkpoint and reference-graph contract audit, not a timing result.",
            "It does not claim that a separately trained MTP-1 checkpoint is impossible.",
            "It does not evaluate the quality of an intentionally approximate stage truncation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the pinned installed checkpoint exposes a faithful, "
            "self-contained native MTP-1 drafter"
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-dspark-file-hashes",
        action="store_true",
        help="validate DSpark file sizes but skip hashing the 10+ GiB payload",
    )
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = audit(
        arguments.model.expanduser().resolve(),
        repo_root,
        verify_hashes=not arguments.skip_dspark_file_hashes,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        sys.stdout.write(rendered)
    else:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(output)


if __name__ == "__main__":
    main()
