from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.model import RuntimeConfig


SHARED_POLICY = (
    "Shared repository policy: preserve exact outputs and validate every cache "
    "boundary.\n"
) * 24
BRANCH_A = "Branch A task: inspect alpha and return one deterministic token."
BRANCH_B = "Branch B task: inspect beta and return one deterministic token."
SHARED_CHECKPOINT_TOKENS = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def _runtime_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_state(project_root: Path, protocol: Path) -> dict:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
        "working_tree_dirty": bool(
            _command_output(["git", "status", "--porcelain"], project_root)
        ),
        "tracked_diff_sha256_at_run": hashlib.sha256(diff).hexdigest(),
        "runtime_python_tree_sha256": _runtime_tree_sha256(
            project_root / "runtime" / "deepseek_v4_ssd"
        ),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "predeclared_protocol_sha256_at_run": _sha256(protocol),
    }


def _config(cache_root: Path, *, persistent: bool) -> RuntimeConfig:
    return RuntimeConfig(
        persistent_prompt_cache=persistent,
        prompt_cache_directory=str(cache_root),
        layer_major_prefill=False,
        expert_file_cache_policy="bypass",
    )


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    common = 0
    for left, right in zip(first, second):
        if left != right:
            break
        common += 1
    return common


def _run_request(runtime: ModelRuntime, prompt: str, run_id: str) -> dict:
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    started = time.perf_counter()
    pieces = list(
        runtime.stream(
            prompt,
            GenerationOptions(max_tokens=1, temperature=0, top_p=1),
        )
    )
    wall_seconds = time.perf_counter() - started
    tokens = [int(piece.token) for piece in pieces]
    return {
        "id": run_id,
        "generated_token_ids": tokens,
        "token_sha256": _token_sha256(tokens),
        "wall_seconds": wall_seconds,
        "metrics": runtime.metrics.snapshot(),
    }


def _shared_bundle(cache_directory: Path, tokens: list[int]) -> dict:
    matches = []
    for metadata_path in cache_directory.glob("*.normal.v4.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("tokens") == tokens:
            matches.append((metadata_path, metadata))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one shared checkpoint payload, found {len(matches)}"
        )
    metadata_path, metadata = matches[0]
    data_path = cache_directory / metadata["data"]
    access_path = cache_directory / (
        f"{metadata['cacheKey']}.normal.v4.access"
    )
    access = (
        json.loads(access_path.read_text(encoding="utf-8"))
        if access_path.is_file()
        else {}
    )
    return {
        "metadata_path": metadata_path.name,
        "metadata_sha256": _sha256(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "data_path": data_path.name,
        "data_sha256": _sha256(data_path),
        "data_bytes": data_path.stat().st_size,
        "access_path": access_path.name,
        "access": access,
        "metadata": metadata,
    }


def _metadata_contract(bundle: dict, revision: str) -> dict[str, bool]:
    metadata = bundle["metadata"]
    contract = metadata.get("contract", {})
    blocks = metadata.get("blocks", [])
    return {
        "mode_exact": metadata.get("mode") == "normal",
        "format_exact": metadata.get("format") == 4,
        "revision_exact": metadata.get("revision") == revision,
        "contract_revision_exact": contract.get("revision") == revision,
        "model_config_sha256_present": len(
            str(contract.get("modelConfigSHA256", ""))
        )
        == 64,
        "rope_contract_present": isinstance(contract.get("rope"), dict),
        "kv_contract_exact": contract.get("kvFormat", {}).get(
            "compressedAttention"
        )
        == "mxfp8",
        "attention_contract_present": isinstance(
            contract.get("attentionMode"), dict
        ),
        "block_size_exact": contract.get("tokenBlockSize")
        == SHARED_CHECKPOINT_TOKENS,
        "one_complete_block": len(blocks) == 1
        and blocks[0].get("length") == SHARED_CHECKPOINT_TOKENS,
        "terminal_key_exact": bool(blocks)
        and blocks[-1].get("key") == metadata.get("cacheKey"),
        "data_file_present": bool(bundle["data_bytes"]),
    }


def _gate(
    runs: list[dict],
    common_prefix_tokens: int,
    before: dict,
    after: dict,
    metadata_contract: dict[str, bool],
) -> dict:
    first, persistent, cold = runs
    criteria = {
        "prompts_share_at_least_one_complete_block": common_prefix_tokens
        >= SHARED_CHECKPOINT_TOKENS,
        "cold_seed_reuses_zero_tokens": int(
            first["metrics"]["prompt_cache_reused_tokens"]
        )
        == 0,
        "restart_branch_reuses_exact_checkpoint": int(
            persistent["metrics"]["prompt_cache_reused_tokens"]
        )
        == SHARED_CHECKPOINT_TOKENS,
        "isolated_cold_branch_reuses_zero_tokens": int(
            cold["metrics"]["prompt_cache_reused_tokens"]
        )
        == 0,
        "persistent_and_cold_branch_tokens_exact": persistent[
            "generated_token_ids"
        ]
        == cold["generated_token_ids"],
        "persistent_and_cold_branch_hash_exact": persistent["token_sha256"]
        == cold["token_sha256"],
        "shared_metadata_immutable": before["metadata_sha256"]
        == after["metadata_sha256"],
        "shared_data_immutable": before["data_sha256"]
        == after["data_sha256"],
        "persistent_hit_recorded": int(
            after.get("access", {}).get("reuseCount", 0)
        )
        >= 1,
        "metadata_contract_exact": all(metadata_contract.values()),
        "all_cache_writes_succeeded": all(
            int(run["metrics"]["prompt_cache_write_errors"]) == 0
            for run in runs
        ),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate block-granular immutable normal prompt caching"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    manifest_path = Path(arguments.prompt_manifest).expanduser().resolve()
    protocol = Path(arguments.protocol).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    prompt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_entry = next(
        entry
        for entry in prompt_manifest["prompts"]
        if entry["name"] == "repeated"
    )
    prompt_path = manifest_path.parent / prompt_entry["file"]
    if _sha256(prompt_path) != prompt_entry["text_sha256"]:
        raise RuntimeError("prompt text SHA-256 does not match manifest")
    base_prompt = prompt_path.read_text(encoding="utf-8")
    shared = f"{base_prompt}\n\n{SHARED_POLICY}"
    first_prompt = f"{shared}\n{BRANCH_A}"
    branch_prompt = f"{shared}\n{BRANCH_B}"

    with tempfile.TemporaryDirectory(prefix="block-prompt-cache-") as temporary:
        cache_root = Path(temporary) / "persistent"
        runtime_a = ModelRuntime.open(str(model), _config(cache_root, persistent=True))
        try:
            first_tokens = runtime_a._encode_prompt(first_prompt)
            branch_tokens = runtime_a._encode_prompt(branch_prompt)
            common_prefix_tokens = _common_prefix_length(first_tokens, branch_tokens)
            if common_prefix_tokens < SHARED_CHECKPOINT_TOKENS:
                raise RuntimeError("constructed prompts do not share one token block")
            runs = [_run_request(runtime_a, first_prompt, "runtime-a-cold-seed")]
            installed = runtime_a.installed
            revision = installed.revision
        finally:
            runtime_a.close()
        del runtime_a

        cache_directory = cache_root / revision
        before = _shared_bundle(
            cache_directory,
            first_tokens[:SHARED_CHECKPOINT_TOKENS],
        )
        mx.clear_cache()
        gc.collect()

        runtime_b = ModelRuntime.open(str(model), _config(cache_root, persistent=True))
        try:
            runs.append(
                _run_request(runtime_b, branch_prompt, "runtime-b-restart-branch")
            )
        finally:
            runtime_b.close()
        del runtime_b
        after = _shared_bundle(
            cache_directory,
            first_tokens[:SHARED_CHECKPOINT_TOKENS],
        )
        shared_payload_count = sum(
            json.loads(path.read_text(encoding="utf-8")).get("tokens")
            == first_tokens[:SHARED_CHECKPOINT_TOKENS]
            for path in cache_directory.glob("*.normal.v4.json")
        )
        mx.clear_cache()
        gc.collect()

        cold_root = Path(temporary) / "cold-disabled"
        runtime_c = ModelRuntime.open(str(model), _config(cold_root, persistent=False))
        try:
            runs.append(
                _run_request(runtime_c, branch_prompt, "runtime-c-isolated-cold-branch")
            )
        finally:
            runtime_c.close()
        del runtime_c

    metadata_contract = _metadata_contract(after, revision)
    metadata_contract["shared_payload_count_exact"] = shared_payload_count == 1
    gate = _gate(
        runs,
        common_prefix_tokens,
        before,
        after,
        metadata_contract,
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_kind": "functional integration gate",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol),
        "checkpoint": {
            "model_path": str(model),
            "model_id": installed.model_id,
            "revision": revision,
            "installed_manifest_sha256": _sha256(model / "manifest.json"),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
            "chip": _command_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], project_root
            ),
            "memory_bytes": _command_output(
                ["sysctl", "-n", "hw.memsize"], project_root
            ),
        },
        "experiment": {
            "base_prompt_file": prompt_entry["file"],
            "base_prompt_text_sha256": prompt_entry["text_sha256"],
            "prompt_manifest_sha256": _sha256(manifest_path),
            "first_prompt_tokens": len(first_tokens),
            "first_prompt_token_sha256": _token_sha256(first_tokens),
            "branch_prompt_tokens": len(branch_tokens),
            "branch_prompt_token_sha256": _token_sha256(branch_tokens),
            "common_prefix_tokens": common_prefix_tokens,
            "shared_checkpoint_tokens": SHARED_CHECKPOINT_TOKENS,
            "max_output_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "batch_size": 1,
            "layer_major_prefill": False,
            "fp8_kv_cache": True,
            "fp4_index_cache": True,
            "expert_file_cache_policy": "bypass",
            "cache_directory": "new isolated temporary directory",
            "run_order": [run["id"] for run in runs],
        },
        "runs": runs,
        "shared_bundle_before_restart_hit": before,
        "shared_bundle_after_restart_hit": after,
        "metadata_contract": metadata_contract,
        "gate": gate,
        "decision": (
            "block_prompt_cache_functionally_validated"
            if gate["passed"]
            else "stop_block_prompt_cache_candidate"
        ),
        "evidence_limits": [
            "This is a three-request functional gate, not a balanced performance result.",
            "Wall time, TTFT, and logical expert bytes are descriptive only.",
            "The gate covers one checkpoint, one synthetic suffix branch, batch size 1, and one restart boundary.",
            "Content-addressed snapshots share an immutable cumulative cache payload; physical per-layer KV-delta deduplication is not implemented.",
            "Passing does not prove concurrent multi-request safety or cold-prefill improvement.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    if not gate["passed"]:
        raise SystemExit("block prompt-cache gate failed")


if __name__ == "__main__":
    main()
