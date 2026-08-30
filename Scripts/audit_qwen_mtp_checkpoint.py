from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path


PINNED_MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
PINNED_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"


class AuditError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _url(path: str) -> str:
    return (
        f"https://huggingface.co/{PINNED_MODEL_ID}/resolve/"
        f"{PINNED_REVISION}/{path}"
    )


def _request(url: str, byte_range: tuple[int, int] | None = None):
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "Whallm-Qwen-MTP-Checkpoint-Audit/1",
    }
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), dict(response.headers.items()), response.status


def _json_object(data: bytes, source: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot decode JSON from {source}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object from {source}")
    return value


def _parse_safetensors_header(prefix: bytes, header: bytes, source: str) -> dict:
    if len(prefix) != 8:
        raise AuditError(f"invalid safetensors prefix from {source}")
    expected = int.from_bytes(prefix, "little")
    if expected != len(header) or expected <= 0 or expected > 64 * 1024 * 1024:
        raise AuditError(
            f"invalid safetensors header size from {source}: {expected}"
        )
    return _json_object(header, source)


def _total_file_bytes(headers: dict[str, str]) -> int | None:
    value = headers.get("Content-Range") or headers.get("content-range")
    if value is None:
        return None
    match = re.fullmatch(r"bytes \d+-\d+/(\d+)", value)
    return int(match.group(1)) if match else None


def _aligned_payload_bytes(tensors: list[dict], alignment: int) -> int:
    offset = 0
    for tensor in sorted(tensors, key=lambda item: item["name"]):
        offset = (offset + alignment - 1) // alignment * alignment
        offset += int(tensor["bytes"])
    return offset


def _read_shard(shard: str, expected_names: set[str]) -> tuple[dict, list[dict]]:
    url = _url(shard)
    prefix, headers, status = _request(url, (0, 7))
    if status != 206:
        raise AuditError(f"range request returned HTTP {status}: {url}")
    header_size = int.from_bytes(prefix, "little")
    header_bytes, _, status = _request(url, (8, 8 + header_size - 1))
    if status != 206:
        raise AuditError(f"header request returned HTTP {status}: {url}")
    header = _parse_safetensors_header(prefix, header_bytes, shard)

    tensors = []
    for name in sorted(expected_names):
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise AuditError(f"{shard} has no header metadata for {name}")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[1] < offsets[0]
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(dtype, str)
        ):
            raise AuditError(f"invalid tensor metadata for {name}")
        tensors.append(
            {
                "name": name,
                "shard": shard,
                "dtype": dtype,
                "shape": shape,
                "bytes": offsets[1] - offsets[0],
                "category": "routed_expert" if ".experts." in name else "common",
            }
        )
    return (
        {
            "path": shard,
            "url": url,
            "file_bytes": _total_file_bytes(headers),
            "header_bytes": header_size,
            "header_sha256": _sha256(header_bytes),
            "mtp_tensor_count": len(tensors),
            "mtp_tensor_bytes": sum(item["bytes"] for item in tensors),
        },
        tensors,
    )


def audit(installed_model: Path, repo_root: Path) -> dict:
    config_bytes, _, config_status = _request(_url("config.json"))
    index_bytes, _, index_status = _request(_url("model.safetensors.index.json"))
    if config_status != 200 or index_status != 200:
        raise AuditError("checkpoint config or index request failed")
    config = _json_object(config_bytes, "config.json")
    index = _json_object(index_bytes, "model.safetensors.index.json")
    text_config = config.get("text_config")
    weight_map = index.get("weight_map")
    if not isinstance(text_config, dict) or not isinstance(weight_map, dict):
        raise AuditError("checkpoint config or index has no required object")
    if text_config.get("mtp_num_hidden_layers") != 1:
        raise AuditError("pinned checkpoint no longer declares one MTP layer")
    if text_config.get("mtp_use_dedicated_embeddings") is not False:
        raise AuditError("pinned checkpoint embedding contract changed")

    names_by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        if isinstance(name, str) and name.startswith("mtp.") and isinstance(shard, str):
            names_by_shard.setdefault(shard, set()).add(name)
    if not names_by_shard:
        raise AuditError("checkpoint index has no mtp.* tensors")

    shard_records = []
    tensors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_read_shard, shard, names): shard
            for shard, names in names_by_shard.items()
        }
        for future in concurrent.futures.as_completed(futures):
            shard, shard_tensors = future.result()
            shard_records.append(shard)
            tensors.extend(shard_tensors)
    shard_records.sort(key=lambda item: item["path"])
    tensors.sort(key=lambda item: item["name"])

    manifest_path = installed_model / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise AuditError(f"cannot read installed manifest: {error}") from error
    manifest = _json_object(manifest_bytes, str(manifest_path))
    if (
        manifest.get("modelID") != PINNED_MODEL_ID
        or manifest.get("revision") != PINNED_REVISION
    ):
        raise AuditError("installed model does not match the pinned checkpoint")
    manifest_names = {
        item.get("name")
        for item in manifest.get("commonTensors", [])
        if isinstance(item, dict)
    }
    installed_files = manifest.get("files", [])
    installed_total_bytes = sum(
        int(item.get("size", 0)) for item in installed_files if isinstance(item, dict)
    )

    dtype_counts = Counter(item["dtype"] for item in tensors)
    dtype_bytes = Counter()
    category_counts = Counter(item["category"] for item in tensors)
    category_bytes = Counter()
    for item in tensors:
        dtype_bytes[item["dtype"]] += item["bytes"]
        category_bytes[item["category"]] += item["bytes"]

    script_path = Path(__file__).resolve()
    mtp_bytes = sum(item["bytes"] for item in tensors)
    mtp_common = [item for item in tensors if item["category"] == "common"]
    common_alignment = 256
    projected_common_bytes = _aligned_payload_bytes(mtp_common, common_alignment)
    expert_blob_bytes = int(manifest["expertBlobSize"])
    expert_count = int(manifest["expertCount"])
    selected_expert_count = int(manifest["selectedExpertCount"])
    projected_expert_file_bytes = expert_blob_bytes * expert_count
    bf16_mtp_cache_bytes_per_token = 2_560
    return {
        "schema_version": 1,
        "evidence_kind": "qwen_mtp_checkpoint_contract_audit",
        "formal_performance_result": False,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "subject": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_REVISION,
            "installed_model": str(installed_model),
        },
        "source_provenance": {
            "git_commit": _command_output(["git", "rev-parse", "HEAD"], repo_root),
            "git_status_short": _command_output(["git", "status", "--short"], repo_root),
            "audit_script": str(script_path.relative_to(repo_root)),
            "audit_script_sha256": _sha256(script_path.read_bytes()),
            "checkpoint_config": {
                "url": _url("config.json"),
                "bytes": len(config_bytes),
                "sha256": _sha256(config_bytes),
            },
            "checkpoint_index": {
                "url": _url("model.safetensors.index.json"),
                "bytes": len(index_bytes),
                "sha256": _sha256(index_bytes),
            },
            "shard_headers": shard_records,
            "installed_manifest": {
                "path": str(manifest_path),
                "bytes": len(manifest_bytes),
                "sha256": _sha256(manifest_bytes),
            },
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "contract": {
            "mtp_num_hidden_layers": text_config["mtp_num_hidden_layers"],
            "mtp_use_dedicated_embeddings": text_config[
                "mtp_use_dedicated_embeddings"
            ],
            "mtp_config": text_config.get("mtp"),
            "tensor_count": len(tensors),
            "shard_count": len(shard_records),
            "tensor_count_by_category": dict(sorted(category_counts.items())),
            "tensor_bytes_by_category": dict(sorted(category_bytes.items())),
            "tensor_count_by_dtype": dict(sorted(dtype_counts.items())),
            "tensor_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
            "checkpoint_mtp_bytes": mtp_bytes,
            "checkpoint_mtp_gib": mtp_bytes / 1024**3,
            "tensors": tensors,
        },
        "installed_model_gap": {
            "format_version": manifest.get("formatVersion"),
            "installed_total_declared_bytes": installed_total_bytes,
            "installed_manifest_mtp_tensor_count": sum(
                isinstance(name, str) and name.startswith("mtp.")
                for name in manifest_names
            ),
            "installed_manifest_has_mtp_file": any(
                isinstance(item, dict)
                and str(item.get("path", "")).startswith("mtp")
                for item in installed_files
            ),
            "minimum_extra_raw_checkpoint_bytes": mtp_bytes,
        },
        "prototype_bounds": {
            "common_alignment_bytes": common_alignment,
            "projected_common_file_bytes": projected_common_bytes,
            "projected_mxfp4_expert_file_bytes": projected_expert_file_bytes,
            "projected_installed_payload_bytes": (
                projected_common_bytes + projected_expert_file_bytes
            ),
            "minimum_selected_expert_slots": selected_expert_count,
            "minimum_selected_expert_slot_bytes": (
                selected_expert_count * expert_blob_bytes
            ),
            "bf16_mtp_cache_bytes_per_token": bf16_mtp_cache_bytes_per_token,
            "bf16_mtp_cache_bytes_at_context": {
                str(context): context * bf16_mtp_cache_bytes_per_token
                for context in (4_096, 32_768, 131_072, 262_144)
            },
            "assumptions": [
                "MTP routed experts use the current Qwen MXFP4 expert blob layout.",
                "MTP common tensors remain BF16 and use the current 256-byte common tensor alignment.",
                "The minimum slot bound covers one token's top-10 routed experts only.",
                "The cache bound reuses the current QSA CacheList with one BF16 main KV cache and one BF16 index KV cache.",
                "The output head and token embedding remain shared with the main model.",
            ],
        },
        "evidence_limits": [
            "This audit reads checkpoint metadata only. It does not download tensor payloads.",
            "The raw FP8/BF16 byte count is not an installed MXFP4 size or a runtime memory result.",
            "This artifact does not validate an MTP inference graph or token distribution.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the pinned Qwen MTP checkpoint tensor contract"
    )
    parser.add_argument("--installed-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = audit(arguments.installed_model.expanduser().resolve(), repo_root)
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
