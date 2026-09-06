"""Small, frozen task-accuracy screen for the sorted-expert hint candidate.

This checks structured answers, not general quality or statistical non-inferiority.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from benchmark_research_baseline import digest
from prepare_r0_prompts import encode_prompt, token_sha256


def score(text, expected):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in b)
        if isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    try:
        return equal(json.loads(text.strip(), object_pairs_hook=unique), expected)
    except (ValueError, TypeError):
        return False


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((root / "docs/benchmarks/2026-09-05-qwen-research-baseline-m5-pro.json").read_text())
    suite = json.loads(args.suite.read_text())
    model = Path(baseline["installed_model"]["path"])
    assert digest(model / "manifest.json") == baseline["installed_model"]["manifest_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(model / "tokenizer", trust_remote_code=True)
    protocol = root / "research/QWEN_TASK_ACCURACY_2026-09-06.md"
    (output / "protocol-at-start.md").write_bytes(protocol.read_bytes())
    (output / "suite.json").write_bytes(args.suite.read_bytes())
    sources = [*sorted((root / "runtime/deepseek_v4_ssd").glob("*.py")), Path(__file__),
               root / "Scripts/research_qwen_sorted_experts.py", protocol, args.suite.resolve()]
    artifact = dict(status="running", formal_performance_result=False, adoption_passed=False,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    environment=dict(platform=platform.platform(), python=sys.version,
                        chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                        memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
                        packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy")}),
                    source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                files_sha256={str(p.relative_to(root)): digest(p) for p in sources}),
                    installed_model=baseline["installed_model"], cache_state=baseline["cache_state"],
                    runs=[], pairs=[], limits=suite["limits"])
    # Freeze every rendered prompt before generating the first answer.
    prompts = {}
    for case in suite["cases"]:
        pad = 0
        for _ in range(10):
            messages = [{"role": "user", "content":
                "Ignore the following padding; it is not task data:\n" + " pad" * pad +
                "\nEnd padding.\n" + case["prompt"] +
                "\nReturn only the requested JSON, without Markdown or explanation."}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
            ids = encode_prompt(tokenizer, prompt)
            if len(ids) == 1024:
                break
            pad += 1024 - len(ids)
            assert pad >= 0
        else:
            raise ValueError("cannot construct a 1024-token prompt")
        path = output / (case["id"] + ".prompt.txt")
        path.write_text(prompt)
        prompts[case["id"]] = dict(path=path, token_hash=token_sha256(ids), text_hash=digest(path))
    try:
        for index, case in enumerate(suite["cases"]):
            pair = {}
            for variant in (("control", "candidate") if index % 2 == 0 else ("candidate", "control")):
                stem = output / f'{case["id"]}-{variant}'
                command = baseline["runs"][0]["command"].copy()
                command[0] = sys.executable
                for flag, value in (("--prompt-file", prompts[case["id"]]["path"]),
                                    ("--max-tokens", 128), ("--metrics-json", stem.with_suffix(".json"))):
                    command[command.index(flag) + 1] = str(value)
                if variant == "candidate":
                    command = [sys.executable, str(root / "Scripts/research_qwen_sorted_experts.py"),
                               "--record", str(stem.with_suffix(".record.json")), "--", *command[3:]]
                print(f'START {index + 1}/{len(suite["cases"])} {case["id"]} {variant}', flush=True)
                with stem.with_suffix(".txt").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
                    subprocess.run(command, cwd=root, env=os.environ | {"PYTHONPATH": str(root / "runtime")},
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=True)
                metrics = json.loads(stem.with_suffix(".json").read_text())
                assert metrics["token_sha256"] == token_sha256(metrics["generated_token_ids"])
                assert metrics["prompt_token_sha256"] == prompts[case["id"]]["token_hash"]
                assert metrics["prompt_tokens"] == 1024 and not metrics["mtp_enabled"]
                assert metrics["approximation_mode"] == "exact" and metrics["prompt_cache_reused_tokens"] == 0
                record = json.loads(stem.with_suffix(".record.json").read_text()) if variant == "candidate" else None
                if record:
                    expected_calls = metrics["request_batched_expert_layers"] * len(metrics["prefill_attention_chunk_sizes"])
                    assert expected_calls == 192
                    assert record["status"] == "completed" and record["sorting_hint"] and record["modified_calls"] == expected_calls
                text = stem.with_suffix(".txt").read_text()
                row = dict(case=case["id"], category=case["category"], variant=variant,
                           command=command, metrics=metrics, record=record, text=text,
                           text_sha256=digest(stem.with_suffix(".txt")),
                           prompt_text_sha256=prompts[case["id"]]["text_hash"],
                           truncated=metrics["generated_tokens"] >= 128,
                           passed=metrics["generated_tokens"] < 128 and score(text, case["expected"]))
                artifact["runs"].append(row)
                pair[variant] = row
                print(f'DONE passed={row["passed"]} tokens={metrics["generated_tokens"]}', flush=True)
                (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
            loss = pair["control"]["passed"] and not pair["candidate"]["passed"]
            artifact["pairs"].append(dict(case=case["id"], category=case["category"], regression=loss,
                control_passed=pair["control"]["passed"], candidate_passed=pair["candidate"]["passed"],
                output_parity=pair["control"]["metrics"]["token_sha256"] == pair["candidate"]["metrics"]["token_sha256"]))
            if loss:
                artifact["status"] = "stopped_on_regression"
                break
        else:
            artifact["status"] = "screen_completed"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
