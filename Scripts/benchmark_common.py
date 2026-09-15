"""Shared measurement helpers; no experiment selection or runtime patches."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

MODE_ARGUMENTS = {
    "normal": (),
    "fixed": ("--dspark",),
}

ROW_METRICS = (
    "generated_token_ids",
    "expert_file_cache_policy",
    "expert_file_direct_io_alignment_bytes",
    "prompt_token_sha256",
    "token_sha256",
    "prompt_tokens",
    "generated_tokens",
    "request_seconds",
    "time_to_first_token_seconds",
    "decode_tokens_per_second",
    "peak_memory_bytes",
    "request_expert_bytes_read",
    "request_process_disk_bytes_read",
    "request_expert_page_cache_probe_calls",
    "request_expert_page_cache_probe_failures",
    "request_expert_page_cache_classified_bytes",
    "request_expert_page_cache_resident_bytes_before_read",
    "request_expert_page_cache_nonresident_bytes_before_read",
    "request_expert_page_cache_unclassified_bytes",
    "request_expert_page_cache_resident_fraction_before_read",
    "request_expert_page_cache_nonresident_fraction_before_read",
    "request_expert_page_cache_nonresident_bytes_per_generated_token",
    "request_expert_evictions",
    "dspark_rounds",
    "dspark_proposed_tokens",
    "dspark_accepted_tokens",
    "dspark_committed_tokens",
    "dspark_output_budget_trimmed_tokens",
    "dspark_fallback",
    "dspark_fallback_would_trigger_rounds",
    "dspark_fallback_triggered_rounds",
    "dspark_fallback_cost_ratios",
    "dspark_last_fallback_target_step_seconds",
    "dspark_last_fallback_speculative_seconds",
    "dspark_last_fallback_break_even_seconds",
    "dspark_last_fallback_cost_ratio",
    "dspark_round_trace",
    "dspark_block_verification_rounds",
    "dspark_sequential_verification_rounds",
    "dspark_last_verification_mode",
    "dspark_last_sequential_position_seconds",
    "dspark_draft_seconds",
    "dspark_verification_seconds",
    "dspark_verification_expert_union_calls",
    "dspark_verification_routed_expert_assignments",
    "dspark_verification_expert_union_experts",
    "dspark_verification_expert_union_reused_assignments",
    "dspark_verification_expert_union_reuse_rate",
    "dspark_verification_expert_union_misses",
    "dspark_target_expert_bytes_read",
    "dspark_verification_expert_bytes_read",
    "dspark_replay_expert_bytes_read",
    "dspark_target_expert_read_seconds",
    "dspark_last_verification_expert_union_layer_ids",
    "dspark_last_verification_expert_assignments_by_layer",
    "dspark_last_verification_expert_union_by_layer",
    "dspark_last_verification_expert_misses_by_layer",
    "dspark_last_verification_expert_bytes_read",
    "dspark_draft_expert_bytes_read",
    "dspark_draft_page_cache_probe_calls",
    "dspark_draft_page_cache_probe_failures",
    "dspark_draft_page_cache_classified_bytes",
    "dspark_draft_page_cache_resident_bytes_before_read",
    "dspark_draft_page_cache_nonresident_bytes_before_read",
    "dspark_draft_page_cache_unclassified_bytes",
    "dspark_draft_page_cache_nonresident_bytes_per_committed_token",
    "dspark_speculative_expert_bytes_read",
    "dspark_draft_expert_bytes_per_committed_token",
    "dspark_target_expert_bytes_per_committed_token",
    "dspark_speculative_expert_bytes_per_committed_token",
    "dspark_expert_cache_hit_rate",
    "dspark_expert_cache_hits",
    "dspark_expert_cache_misses",
    "dspark_expert_evictions",
    "dspark_expert_bytes_read",
    "dspark_expert_read_seconds",
    "dspark_expert_upload_seconds",
    "dspark_expert_pack_seconds",
    "dspark_expert_resident_slots",
    "dspark_expert_capacity_slots",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _runtime_tree_sha256(runtime_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(runtime_root.rglob("*.py")):
        digest.update(path.relative_to(runtime_root).as_posix().encode())
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

def _sysctl(name: str, cwd: Path) -> str | None:
    return _command_output(["sysctl", "-n", name], cwd)

def _load_prompts(manifest_path: Path, names: tuple[str, ...]) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    by_name = {prompt["name"]: prompt for prompt in manifest["prompts"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"prompt manifest is missing: {', '.join(missing)}")
    prompts = []
    for name in names:
        prompt = dict(by_name[name])
        path = manifest_path.parent / prompt["file"]
        if not path.is_file():
            raise ValueError(f"prompt file is missing: {path}")
        if _sha256(path) != prompt["text_sha256"]:
            raise ValueError(f"prompt text hash mismatch: {path}")
        prompt["path"] = str(path)
        prompts.append(prompt)
    return prompts

def _run(
    *,
    project_root: Path,
    python: Path,
    model: Path,
    prompt: dict,
    mode: str,
    run_id: str,
    raw_directory: Path,
    max_tokens: int,
    resume: bool,
    dspark_fallback_enabled: bool,
    layer_major_prefill_enabled: bool,
    sequential_verification_enabled: bool,
    dspark_slots: int | None = None,
    expert_page_cache_probe_enabled: bool = False,
    expert_file_cache_policy: str = "cached",
) -> dict:
    metrics_path = raw_directory / f"{run_id}.metrics.json"
    stdout_path = raw_directory / f"{run_id}.stdout.txt"
    stderr_path = raw_directory / f"{run_id}.stderr.txt"
    if not (resume and metrics_path.is_file()):
        command = [
            str(python),
            "-m",
            "deepseek_v4_ssd.cli",
            "--model",
            str(model),
            "--prompt-file",
            prompt["path"],
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--no-persistent-prompt-cache",
            "--no-separate-prefill-io",
            "--expert-file-cache-policy",
            expert_file_cache_policy,
            *(
                ("--expert-page-cache-probe",)
                if expert_page_cache_probe_enabled
                else ()
            ),
            *(("--no-layer-major-prefill",) if not layer_major_prefill_enabled else ()),
            *MODE_ARGUMENTS[mode],
            *(
                ("--dspark-sequential-verification",)
                if mode != "normal" and sequential_verification_enabled
                else ()
            ),
            *(
                ("--dspark-confidence-threshold", "0")
                if mode != "normal"
                else ()
            ),
            *(
                ("--dspark-slots", str(dspark_slots))
                if mode != "normal" and dspark_slots is not None
                else ()
            ),
            *(
                ("--no-dspark-fallback",)
                if mode != "normal" and not dspark_fallback_enabled
                else ()
            ),
            "--metrics-json",
            str(metrics_path),
        ]
        environment = os.environ.copy()
        runtime = str(project_root / "runtime")
        environment["PYTHONPATH"] = (
            runtime
            if not environment.get("PYTHONPATH")
            else runtime + os.pathsep + environment["PYTHONPATH"]
        )
        environment["TOKENIZERS_PARALLELISM"] = "false"
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode:
            tail = stderr_path.read_text(encoding="utf-8")[-2_000:]
            raise RuntimeError(f"{run_id} failed ({completed.returncode}):\n{tail}")
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if metrics.get("prompt_token_sha256") != prompt["prompt_token_sha256"]:
        raise RuntimeError(f"{run_id} prompt token hash does not match the manifest")
    return {
        "id": run_id,
        "workload": prompt["name"],
        "mode": mode,
        "raw_metrics_sha256": _sha256(metrics_path),
        "metrics": {key: metrics.get(key) for key in ROW_METRICS},
    }

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
