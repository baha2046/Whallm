"""Bounded, isolated API reproduction against the installed Whallm bundle.

Exploratory quality evidence, not a formal performance benchmark. No bundle edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import socket
import subprocess
import sys
import threading
import time
import urllib.request

LONG = """日本語の小説の第一章を、約5000字で書いてください。舞台は秋の海辺の町です。十年ぶりに帰郷した翻訳家の遥が、亡くなった祖母の書店を片付けていると、差出人のない手紙を見つけます。次の四つの場面を順番に描いてください。1. 雨の駅に着き、幼なじみの修と再会する。2. 古い書店を片付け、祖母との思い出を振り返る。3. 本に挟まれた手紙を見つけ、祖母が隠していた約束を知る。4. 夕暮れの港で修と話し、明日その約束の相手を訪ねると決める。各場面を十分な長さで描き、自然な会話、具体的な情景、人物の感情の変化を入れてください。あらすじではなく、小説の本文を書いてください。"""
SHORT = """次の文章は「〜ている」という文末が続いて単調です。意味を変えず、自然な日本語になるように文末に変化をつけて書き直してください。修正後の文章だけを答えてください。
「窓の外では雨が降っている。彼女は机に向かっている。机の上には古い手紙が置かれている。」"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["long-thinking", "long-chat"])
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = Path("/Applications/Whallm.app/Contents")
    runtime = app / "Resources/runtime"
    model = Path("/Users/yanun/.dsmodel/qwen3.8-flash-next.dsv4")
    env = dict(os.environ, PYTHONHOME=str(app / "Frameworks/Python.framework/Versions/Current"),
               PYTHONPATH=f"{runtime}:{app / 'Resources/python/site-packages'}",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env.pop("DEEPSEEK_API_KEY", None)
    metadata = dict(kind="exploratory issue-6 quality reproduction", platform=platform.platform(),
                    macos=subprocess.check_output(["sw_vers"], text=True),
                    hardware=subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True).split("Serial Number")[0],
                    app_version=plistlib.loads((app / "Info.plist").read_bytes())["CFBundleShortVersionString"],
                    project_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    runtime_source="installed 1.1.4 bundle; project working tree is not executed",
                    source_hashes={str(x.relative_to(runtime)): digest(x.read_bytes()) for x in runtime.rglob("*.py")},
                    model_path=str(model), manifest_sha256=digest((model / "manifest.json").read_bytes()),
                    cache="fresh server and empty private persistent prompt cache for every case; OS page cache not purged; disk not RAM disk",
                    limitations=["Original issue prompts are unavailable; substitute Japanese prompts.",
                                 "Single stochastic sample per case; API seed unsupported, no seed injected.",
                                 "API text hashes are not generated token ID hashes; timings are exploratory.",
                                 "No full model checksum audit repeated."],
                    timeout_seconds=args.timeout)
    write_json(args.output / "environment.json", metadata)
    for name in args.cases:
        if name not in {"long-thinking", "long-chat", "short-thinking", "short-chat"}:
            raise ValueError(name)
        case = args.output / name
        case.mkdir()
        thinking = name.endswith("thinking")
        payload = dict(model="qwen3.8-flash-next-fp8", messages=[dict(role="user", content=LONG if name.startswith("long") else SHORT)],
                       reasoning_effort="medium" if thinking else "none", stream=True,
                       stream_options=dict(include_usage=True), max_tokens=8000 if name.startswith("long") else 4096)
        write_json(case / "request.json", payload)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [str(app / "MacOS/python3"), "-m", "deepseek_v4_ssd.server", "--model", str(model),
                   "--host", "127.0.0.1", "--port", str(port), "--slots", "8192", "--prompt-cache-entries", "2",
                   "--prompt-cache-directory", str(case.resolve() / "prompt-cache")]
        write_json(case / "command.json", command)
        result = dict(case=name, state="starting", content_chars=0, reasoning_chars=0, finish_reason=None, usage=None)
        content, reasoning = [], []
        start = time.monotonic()
        def save():
            result.update(elapsed_seconds=round(time.monotonic() - start, 2),
                          content_chars=sum(map(len, content)), reasoning_chars=sum(map(len, reasoning)))
            write_json(case / "status.json", result)
        save()
        print(f"START {name}", flush=True)
        with (case / "server.log").open("w") as log:
            proc = subprocess.Popen(command, env=env, cwd=runtime, stdout=log, stderr=subprocess.STDOUT)
            write_json(case / "process.json", dict(pid=proc.pid, port=port))
            expired = threading.Event()
            def timeout():
                expired.set()
                if proc.poll() is None:
                    proc.kill()
            timer = threading.Timer(args.timeout, timeout)
            timer.start()
            try:
                for _ in range(120):
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exited {proc.returncode}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        time.sleep(1)
                else:
                    raise TimeoutError("startup health check")
                result["state"] = "generating"
                save()
                request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                last_save = last_print = time.monotonic()
                with urllib.request.urlopen(request, timeout=args.timeout) as response, (case / "events.jsonl").open("w") as events, (case / "content.txt").open("w") as body, (case / "reasoning.txt").open("w") as thought:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            result["done_event"] = True
                            break
                        event = json.loads(data)
                        events.write(json.dumps(dict(seconds=time.monotonic()-start, event=event), ensure_ascii=False) + "\n")
                        events.flush()
                        if event.get("usage"):
                            result["usage"] = event["usage"]
                        if event.get("error"):
                            result["api_error"] = event["error"]
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            for key, chunks, handle in [("content", content, body), ("reasoning_content", reasoning, thought)]:
                                if delta.get(key):
                                    result.setdefault("first_text_seconds", time.monotonic()-start)
                                    chunks.append(delta[key]); handle.write(delta[key]); handle.flush()
                            if choice.get("finish_reason"):
                                result["finish_reason"] = choice["finish_reason"]
                        if time.monotonic() - last_save >= 10:
                            save(); last_save = time.monotonic()
                        if time.monotonic() - last_print >= 30:
                            print(json.dumps(result, ensure_ascii=False), flush=True)
                            last_print = time.monotonic()
                result["state"] = "complete" if result.get("done_event") else "incomplete-stream"
            except Exception as exc:
                result.update(state="error", error=repr(exc))
            finally:
                timer.cancel()
                if expired.is_set():
                    result["state"] = "wall-time-limit"
                if proc.poll() is None:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as response:
                            write_json(case / "runtime-status-final.json", json.load(response))
                    except Exception as exc:
                        result["final_status_error"] = repr(exc)
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                result["server_exit_code_after_cleanup"] = proc.returncode
                for label, chunks in [("content", content), ("reasoning", reasoning)]:
                    result[label + "_sha256"] = digest("".join(chunks).encode())
                save()
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
