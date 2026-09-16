"""Frozen new-prompt, long-decode extension for the existing no-hint candidate."""
from __future__ import annotations
if __package__:
    from .archived_evidence import archived_path
else:
    from archived_evidence import archived_path

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_research_baseline import digest
from prepare_r0_prompts import encode_prompt, token_sha256


TASKS = {
    "code": "Write a complete Python implementation of an in-memory event scheduler. Events have an id, integer due time, priority and callback name. Support schedule, cancel, reschedule and pop_due(now), preserving insertion order when due time and priority tie. Explain the invariant, time complexity, and include at least 12 executable unittest cases covering replacement, cancellation, tie order, empty queues and repeated calls. Use only the standard library. Provide the full code and then the explanations; do not abbreviate the tests.",
    "zh": "請用繁體中文寫一份社區圖書交換站的完整運作手冊。條件：每週三與週六開放，每人最多借三本，借期十四天，預約依申請順序，逾期只暂停新借不收費，兒童繪本須由成人代借。請分別說明登記、借還、預約、逾期、遺失、志工交班及個資保存，列出十二個具體問答，最後附一份可直接使用的交班表。各段都要有實際例子，全文至少一千五百字。遇到未提供的規則請標明為建議，不要假稱既定政策。",
    "math": "For every integer n from 2 through 35, solve x+y=n and x^2+y^2=n^2-2n over the real numbers. Derive the general formula, say exactly which n admit real solutions, and enumerate each n with both ordered solutions or a proof of impossibility. Show substitution checks for n=4, n=9 and n=25. Then explain why the discriminant criterion is necessary and sufficient. Include all 34 cases without ellipses and show intermediate arithmetic.",
    "tool": 'Produce only a JSON array of exactly 40 planned tool calls; do not execute any calls. Call i for i=1..40 uses name "record_inspection" and arguments {"item_id":"BOX-XXX","zone":...,"checks":["seal","label","weight"],"requires_photo":...}. XXX is i zero-padded to 3 digits. zone is "north" if i mod 3=1, "south" if i mod 3=2, otherwise "west". requires_photo is true exactly when i is divisible by 4 or 7. Each call also has "sequence":i and "reason": a full sentence stating its zone and whether a photo is required. Output every entry in full; no ellipses or prose outside the JSON.',
}


def pair_contract(control, candidate):
    a, b = control["metrics"], candidate["metrics"]
    return {
        "output_parity": a["generated_token_ids"] == b["generated_token_ids"],
        "logical_bytes_equal": a["request_expert_bytes_read"] == b["request_expert_bytes_read"],
        "qmm_calls_equal": a["request_gather_qmm_calls"] == b["request_gather_qmm_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm", type=Path, help="Stopped results to confirm, in reversed order")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((archived_path("docs/benchmarks/2026-09-05-qwen-research-baseline-m5-pro.json")).read_text())
    model = Path(baseline["installed_model"]["path"])
    assert digest(model / "manifest.json") == baseline["installed_model"]["manifest_sha256"]
    protocol = archived_path("research/QWEN_NOHINT_VALIDATION_2026-09-06.md")
    files = [*sorted((root / "runtime/deepseek_v4_ssd").glob("*.py")), Path(__file__), protocol,
             root / "Scripts/research_qwen_sorted_experts.py", root / "Scripts/benchmark_research_baseline.py",
             root / "Scripts/prepare_r0_prompts.py"]
    artifact = dict(status="running", formal_performance_result=False,
        candidate_kind="sorted_experts_without_hint", started_at=datetime.now(timezone.utc).isoformat(),
        source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    files_sha256={str(p.relative_to(root)): digest(p) for p in files}),
        environment=dict(platform=platform.platform(), python=sys.version,
            chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
            packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy")}),
        installed_model=baseline["installed_model"], cache_state=baseline["cache_state"],
        max_tokens=1024, prompts=[], runs=[], pairs=[],
        limits=["New authored tasks with inert padding, not a general capability or long-document understanding test.",
                "Two reversed pairs per case; OS cache not purged; generation may end at the token cap.",
                "Research process only; defaults unchanged."])
    for path in files:
        dest = output / "source-at-start" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    if args.confirm:
        previous = json.loads(args.confirm.read_text())
        assert previous["status"] == "stopped_on_contract"
        assert previous["source"]["files_sha256"] == artifact["source"]["files_sha256"]
        last = previous["pairs"][-1]
        prompt = next(p for p in previous["prompts"] if p["case"] == last["case"])
        artifact["prompts"] = [prompt]
        prior = [r for r in previous["runs"] if r["case"] == last["case"] and r["wave"] == last["wave"]]
        plan = [(0, prompt, list(reversed([r["variant"] for r in prior])))]
        artifact["confirmation_of_sha256"] = digest(args.confirm)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model / "tokenizer", trust_remote_code=True)
        for length in (1024, 16384):
            for category, task in TASKS.items():
                padding = 0
                for _ in range(12):
                    prompt = tokenizer.apply_chat_template([dict(role="user", content=
                        "Ignore this padding; it is not task data:\n" + " pad" * padding +
                        "\nEnd padding.\nTask:\n" + task)], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    ids = encode_prompt(tokenizer, prompt)
                    if len(ids) == length:
                        break
                    padding += length - len(ids)
                    assert padding >= 0
                else:
                    raise ValueError("cannot prepare exact prompt length")
                case = f"{category}-{length}"
                path = output / f"{case}.prompt.txt"
                path.write_text(prompt)
                artifact["prompts"].append(dict(case=case, category=category, length=length,
                    path=str(path), text_sha256=digest(path), token_sha256=token_sha256(ids), task=task))
        plan = [(wave, p, ["control", "candidate"] if (wave + i) % 2 == 0 else ["candidate", "control"])
                for wave in (0, 1) for i, p in enumerate(artifact["prompts"])]
    (output / "manifest.json").write_text(json.dumps(artifact["prompts"], ensure_ascii=False, indent=2) + "\n")
    try:
        for number, (wave, prompt, order) in enumerate(plan, 1):
            pair = {}
            for variant in order:
                assert all(digest(root / name) == sha for name, sha in artifact["source"]["files_sha256"].items())
                assert digest(Path(prompt["path"])) == prompt["text_sha256"]
                stem = output / f'{wave}-{prompt["case"]}-{variant}'
                command = baseline["runs"][0]["command"].copy()
                command.append("--no-qwen-grouped-experts")
                command[0] = sys.executable
                for flag, value in (("--prompt-file", prompt["path"]), ("--max-tokens", 1024),
                                    ("--metrics-json", stem.with_suffix(".json"))):
                    command[command.index(flag) + 1] = str(value)
                if variant == "candidate":
                    command = [sys.executable, str(root / "Scripts/research_qwen_sorted_experts.py"),
                        "--record", str(stem.with_suffix(".record.json")), "--no-sorting-hint", "--", *command[3:]]
                print(f'START {number}/{len(plan)} wave={wave} {prompt["case"]} {variant}', flush=True)
                start = time.perf_counter()
                with stem.with_suffix(".txt").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
                    subprocess.run(["/usr/bin/time", "-l", *command], cwd=root,
                        env=os.environ | {"PYTHONPATH": str(root / "runtime")},
                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=True)
                metrics = json.loads(stem.with_suffix(".json").read_text())
                assert metrics["token_sha256"] == token_sha256(metrics["generated_token_ids"])
                assert metrics["prompt_token_sha256"] == prompt["token_sha256"]
                assert metrics["prompt_tokens"] == prompt["length"] and not metrics["mtp_enabled"]
                assert metrics["approximation_mode"] == "exact" and metrics["prompt_cache_reused_tokens"] == 0
                record = json.loads(stem.with_suffix(".record.json").read_text()) if variant == "candidate" else None
                if record:
                    expected = metrics["request_batched_expert_layers"] * len(metrics["prefill_attention_chunk_sizes"])
                    assert record["status"] == "completed" and not record["sorting_hint"]
                    assert record["modified_calls"] == expected and expected > 0
                rss = re.search(r"(\d+)\s+maximum resident set size", stem.with_suffix(".stderr").read_text())
                row = dict(wave=wave, case=prompt["case"], variant=variant, command=command,
                    metrics=metrics, record=record, process_seconds=time.perf_counter()-start,
                    max_rss_bytes=int(rss[1]) if rss else None, text=stem.with_suffix(".txt").read_text(),
                    text_sha256=digest(stem.with_suffix(".txt")), metrics_sha256=digest(stem.with_suffix(".json")))
                artifact["runs"].append(row)
                pair[variant] = row
                (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
                print(f'DONE tokens={metrics["generated_tokens"]} TTFT={metrics["time_to_first_token_seconds"]:.2f} decode={metrics["decode_tokens_per_second"]:.2f}', flush=True)
            contract = pair_contract(pair["control"], pair["candidate"])
            artifact["pairs"].append(dict(wave=wave, case=prompt["case"], **contract))
            print(f"PAIR {contract}", flush=True)
            if not all(contract.values()):
                artifact["status"] = "stopped_on_contract"
                break
        else:
            artifact["status"] = "completed"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
