"""N2: isolated cache branches and cross-version restart for no-hint grouping."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_research_baseline import digest
from prepare_r0_prompts import encode_prompt, token_sha256


def matching_entry(entries, tokens):
    lengths = [len(e.tokens) for e in entries if len(e.tokens) < len(tokens)
               and tokens[:len(e.tokens)] == e.tokens]
    return max(lengths, default=0)


def tensor_payload(path):
    """Compare tensor bytes independently of safetensors header/storage order."""
    with path.open("rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        if not 0 < header_size <= min(16 * 1024**2, path.stat().st_size - 8):
            raise ValueError("invalid safetensors header size")
        header = json.loads(file.read(header_size))
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            assert 0 <= start <= end <= path.stat().st_size - 8 - header_size
            file.seek(8 + header_size + start)
            tensors[name] = dict(dtype=info["dtype"], shape=info["shape"],
                sha256=hashlib.sha256(file.read(end-start)).hexdigest())
    return dict(schema=json.loads(header["__metadata__"]["state"]), tensors=tensors)


def shared_bundle(cache_root, shared_tokens):
    matches = []
    for path in cache_root.rglob("*.normal.v5.json"):
        metadata = json.loads(path.read_text())
        if metadata["tokens"] == shared_tokens:
            matches.append(dict(metadata=metadata, metadata_sha256=digest(path),
                data_sha256=digest(path.parent / metadata["data"]),
                payload=tensor_payload(path.parent / metadata["data"])))
    if len(matches) != 1:
        raise ValueError("expected exactly one shared checkpoint")
    return matches[0]


def worker(plan, output):
    from deepseek_v4_ssd import qwen4_exp as qwen
    from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
    from deepseek_v4_ssd.model import RuntimeConfig
    from research_qwen_sorted_experts import sorted_expert_output

    original = qwen.StreamingExperts.__call__
    calls = 0

    def grouped(self, value, indices):
        nonlocal calls
        batched = self.cache.current_batched(self.layer)
        if not isinstance(batched, qwen.QwenBatchedExperts) or indices.size < 64:
            return original(self, value, indices)
        result = sorted_expert_output(value, indices, batched, sorting_hint=False)
        self.cache.record_gather_qmm(2)
        calls += 1
        return result

    config = RuntimeConfig(slots=4096, memory_limit_gib=48, mtp_enabled=False,
        persistent_prompt_cache=plan["action"] != "cold", prompt_cache_directory=plan["cache_root"])
    assert digest(Path(plan["model"]) / "manifest.json") == plan["manifest_sha256"]
    if plan["variant"] == "candidate":
        qwen.StreamingExperts.__call__ = grouped
    runtime = None
    result = dict(status="running", plan=plan, config=asdict(config), runs=[])
    try:
        runtime = ModelRuntime.open(plan["model"], config)
        result["cache_contract"] = runtime._normal_prompt_cache_contract()
        shared = plan["suite"]["shared_token_ids"]
        if plan["action"] == "seed":
            before = calls
            count = runtime.warm_prompt(plan["suite"]["warm_prompt"])
            assert count == len(shared)
            result["warm_modified_calls"] = calls-before
            result["shared_before"] = shared_bundle(Path(plan["cache_root"]), shared)
            shutil.copytree(plan["cache_root"], plan["seed_copy"])
        elif plan["action"] == "restart":
            result["shared_before"] = shared_bundle(Path(plan["cache_root"]), shared)
        for branch in plan["branches"]:
            prompt = plan["suite"]["branches"][branch]
            tokens = runtime._encode_prompt(prompt["text"])
            assert token_sha256(tokens) == prompt["token_sha256"]
            memory = matching_entry(runtime._prompt_caches, tokens)
            persistent = matching_entry(runtime._persistent_prompt_caches, tokens)
            source = "memory" if memory else "persistent" if persistent else "none"
            expected_reuse = memory or persistent
            before = calls
            pieces = list(runtime.stream(prompt["text"], GenerationOptions(max_tokens=128, temperature=0, top_p=1)))
            metrics = runtime.metrics.snapshot()
            ids = [int(p.token) for p in pieces]
            row = dict(branch=branch, metrics=metrics, generated_token_ids=ids,
                token_sha256=token_sha256(ids), text="".join(p.text for p in pieces),
                prompt_token_sha256=token_sha256(tokens), modified_calls=calls-before,
                matching_memory_tokens=memory, matching_persistent_tokens=persistent,
                expected_reuse_source=source, expected_reused_tokens=expected_reuse)
            result["runs"].append(row)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
            assert metrics["prompt_cache_reused_tokens"] == expected_reuse
            assert not metrics["prompt_cache_write_errors"] and not metrics["mtp_enabled"]
            assert metrics["approximation_mode"] == "exact"
            assert (expected_reuse == 0 if plan["action"] == "cold" else expected_reuse >= len(shared))
        if plan["action"] != "cold":
            result["shared_after"] = shared_bundle(Path(plan["cache_root"]), shared)
            assert result["shared_before"] == result["shared_after"]
        controller = getattr(runtime.model, "ane_prefill", None)
        result["ane_state"] = controller.snapshot() if controller is not None else None
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        qwen.StreamingExperts.__call__ = original
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


def prepare_suite(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model / "tokenizer", trust_remote_code=True)
    tasks = {
        "A": "Write Python code for merge_intervals(intervals). Input endpoints are integers and each pair is inclusive. Normalize reversed endpoints, merge overlapping or adjacent intervals, sort the result, and preserve no duplicates. Include type hints and explain the invariant, followed by six edge-case examples. Return the full implementation before the explanation.",
        "B": "Write Python code for rotate_groups(items, group_size). Rotate each consecutive full group one position to the right, leaving a final incomplete group unchanged. Reject nonpositive group_size and return a new list. Include type hints, the implementation, and six examples covering empty input, singleton groups and a final partial group. Explain the loop invariant.",
    }
    suite = dict(branches={})
    for branch, task in tasks.items():
        padding = 0
        for _ in range(10):
            text = tokenizer.apply_chat_template([dict(role="user", content=
                "Ignore the following shared padding; it contains no task data:\n" + " pad" * padding +
                "\nEnd shared padding.\n" + task)], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = encode_prompt(tokenizer, text)
            if len(ids) == 2048:
                break
            padding += 2048-len(ids)
            assert padding >= 0
        else:
            raise ValueError("cannot construct branch length")
        suite["branches"][branch] = dict(text=text, token_ids=ids, token_sha256=token_sha256(ids))
    warm_ids = suite["branches"]["A"]["token_ids"][:1025]
    suite["warm_prompt"] = tokenizer.decode(warm_ids, clean_up_tokenization_spaces=False)
    assert encode_prompt(tokenizer, suite["warm_prompt"]) == warm_ids
    suite["shared_token_ids"] = warm_ids[:-1]
    assert all(p["token_ids"][:1024] == warm_ids[:-1] for p in suite["branches"].values())
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n1", type=Path)
    parser.add_argument("--worker-plan", type=Path)
    args = parser.parse_args()
    if args.worker_plan:
        worker(json.loads(args.worker_plan.read_text()), args.output)
        return
    if not args.n1:
        parser.error("N1 completed results are required")
    from analyze_qwen_nohint_validation import summarize
    n1 = json.loads(args.n1.read_text())
    assert summarize(n1)["passed"], "N1 must pass before N2"
    root = Path(__file__).resolve().parents[1]
    for name, sha in n1["source"]["files_sha256"].items():
        assert digest(root / name) == sha, f"source changed since N1: {name}"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = Path(n1["installed_model"]["path"])
    suite = prepare_suite(model)
    (output / "suite.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2)+"\n")
    files = [*sorted((root / "runtime/deepseek_v4_ssd").glob("*.py")), Path(__file__),
        root / "Scripts/research_qwen_sorted_experts.py", root / "Scripts/prepare_r0_prompts.py",
        root / "Scripts/benchmark_research_baseline.py", root / "Scripts/analyze_qwen_nohint_validation.py",
        root / "research/QWEN_NOHINT_CACHE_2026-09-06.md",
        root / "Native/ANEBridge/WhallmANE.m", root / "Native/ANEBridge/LICENSE",
        root / "Scripts/build-ane-bridge.sh", root / ".build/native/libWhallmANE.dylib"]
    source = {str(p.relative_to(root)): digest(p) for p in files}
    for path in files:
        dest = output / "source-at-start" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    artifact = dict(status="running", formal_performance_result=False, n1_sha256=digest(args.n1),
        started_at=datetime.now(timezone.utc).isoformat(), source=dict(commit=n1["source"]["commit"], files_sha256=source),
        environment=n1["environment"], installed_model=n1["installed_model"], suite=suite, workers=[])
    plans = [(f"cold-{v}-{b}", v, "cold", [b], None) for b in ("A", "B") for v in ("control", "candidate")]
    plans += [(f"seed-{v}", v, "seed", ["A", "B", "A"], None) for v in ("control", "candidate")]
    plans += [(f"restart-{writer}-to-{reader}", reader, "restart", ["B", "A"], writer)
              for writer in ("control", "candidate") for reader in ("control", "candidate")]
    cold = {}
    cold_text = {}
    try:
        for number, (name, variant, action, branches, writer) in enumerate(plans, 1):
            assert all(digest(root / name) == sha for name, sha in source.items())
            cache = output / f"{name}-cache"
            if writer:
                shutil.copytree(output / f"seed-{writer}-snapshot", cache)
            plan = dict(model=str(model), manifest_sha256=n1["installed_model"]["manifest_sha256"],
                variant=variant, action=action, branches=branches, suite=suite, cache_root=str(cache),
                seed_copy=str(output / f"seed-{variant}-snapshot"))
            plan_path = output / f"{name}.plan.json"
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n")
            result_path = output / f"{name}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(plan_path), "--output", str(result_path)]
            print(f"START {number}/{len(plans)} {name}", flush=True)
            with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
                completed = subprocess.run(command, cwd=root, env=os.environ | {"PYTHONPATH": str(root / "runtime")},
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            result = json.loads(result_path.read_text()) if result_path.exists() else dict(status="failed", runs=[])
            artifact["workers"].append(dict(name=name, command=command, result=result))
            assert completed.returncode == 0 and result["status"] == "completed", name
            for row in result["runs"]:
                assert token_sha256(row["generated_token_ids"]) == row["token_sha256"]
                branch = row["branch"]
                if action == "cold" and variant == "control":
                    cold[branch] = row["generated_token_ids"]
                    cold_text[branch] = row["text"]
                assert row["generated_token_ids"] == cold[branch], f"{name}/{branch}: output differs from cold control"
                assert row["text"] == cold_text[branch], f"{name}/{branch}: decoded text differs from cold control"
                if variant == "candidate" and row["expected_reused_tokens"] <= 1024:
                    assert row["modified_calls"] > 0
            if action == "seed" and variant == "candidate":
                control = next(w["result"] for w in artifact["workers"] if w["name"] == "seed-control")
                assert result["warm_modified_calls"] > 0
                assert result["shared_before"]["metadata"] == control["shared_before"]["metadata"]
                assert result["shared_before"]["payload"] == control["shared_before"]["payload"], "shared checkpoint tensors differ between variants"
            (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n")
            print(f"DONE {name}", flush=True)
        artifact["status"] = "completed"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
