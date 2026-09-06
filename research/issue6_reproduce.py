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
    p.add_argument("--runtime-path", type=Path, help="optional candidate runtime directory; still uses bundled Python/dependencies")
    p.add_argument("--dependency-path", type=Path, help="isolated dependency overlay ahead of bundled packages")
    p.add_argument("--python-executable", type=Path, help="development Python for unsigned dependency overlay")
    p.add_argument("--payload-overrides", type=json.loads, default={})
    p.add_argument("--requests-json", type=Path, help="named custom API payloads for regression coverage")
    p.add_argument("--record-tokens", action="store_true", help="record raw generated IDs with a research-only iterator wrapper")
    p.add_argument("--research-seed", type=int, help="seed the runtime in the observer; not an API feature")
    p.add_argument("--no-persistent-cache", action="store_true", help="diagnostic control disabling disk prompt cache")
    p.add_argument("--uncompiled-categorical", action="store_true", help="research-only sampler control on the original MLX version")
    args = p.parse_args()
    if args.research_seed is not None and not args.record_tokens:
        p.error("--research-seed requires --record-tokens")
    if args.uncompiled_categorical and not args.record_tokens:
        p.error("--uncompiled-categorical requires --record-tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    app = Path("/Applications/Whallm.app/Contents")
    python = str(args.python_executable.absolute()) if args.python_executable else str(app / "MacOS/python3")
    runtime = args.runtime_path.resolve() if args.runtime_path else app / "Resources/runtime"
    model = Path("/Users/yanun/.dsmodel/qwen3.8-flash-next.dsv4")
    env = dict(os.environ, PYTHONHOME=str(app / "Frameworks/Python.framework/Versions/Current"),
               PYTHONPATH=f"{runtime}:{app / 'Resources/python/site-packages'}",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env.pop("DEEPSEEK_API_KEY", None)
    if args.dependency_path:
        env["PYTHONPATH"] = f"{args.dependency_path.resolve()}:{env['PYTHONPATH']}"
    metadata = dict(kind="exploratory issue-6 quality reproduction", platform=platform.platform(),
                    macos=subprocess.check_output(["sw_vers"], text=True),
                    hardware=subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True).split("Serial Number")[0],
                    app_version=plistlib.loads((app / "Info.plist").read_bytes())["CFBundleShortVersionString"],
                    project_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    runtime_source=str(runtime), driver_sha256=digest(Path(__file__).read_bytes()),
                    source_hashes={str(x.relative_to(runtime)): digest(x.read_bytes()) for x in runtime.rglob("*.py")},
                    model_path=str(model), manifest_sha256=digest((model / "manifest.json").read_bytes()),
                    cache=("fresh server; persistent prompt cache disabled" if args.no_persistent_cache else
                           "fresh server and empty private persistent prompt cache for every case") +
                          "; OS page cache not purged; disk not RAM disk",
                    limitations=["Original issue prompts are unavailable; substitute Japanese prompts.",
                                 ("Single sample per case; seed injected by research observer, not an API feature."
                                  if args.research_seed is not None else
                                  "Single stochastic sample per case; API seed unsupported, no seed injected."),
                                 "API text hashes are not generated token ID hashes; timings are exploratory.",
                                 "No full model checksum audit repeated."],
                    timeout_seconds=args.timeout, research_seed=args.research_seed,
                    persistent_prompt_cache=not args.no_persistent_cache)
    write_json(args.output / "environment.json", metadata)
    metadata["dependency_overlay"] = str(args.dependency_path.resolve()) if args.dependency_path else None
    metadata["python_executable"] = python
    metadata["research_uncompiled_categorical"] = args.uncompiled_categorical
    metadata["packages"] = json.loads(subprocess.check_output(
        [python, "-c",
         "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['mlx','mlx-metal','mlx-lm','numpy','transformers']}))"],
        env=env, text=True))
    write_json(args.output / "environment.json", metadata)
    custom = json.loads(args.requests_json.read_text()) if args.requests_json else None
    for name in custom if custom is not None else args.cases:
        if (args.output / "STOP").exists():
            print("Remaining cases cancelled by STOP marker", flush=True)
            break
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(name)
        if custom is None and name not in {"long-thinking", "long-chat", "short-thinking", "short-chat"}:
            raise ValueError(name)
        case = args.output / name
        case.mkdir()
        thinking = name.endswith("thinking")
        payload = dict(model="qwen3.8-flash-next-fp8", messages=[dict(role="user", content=LONG if name.startswith("long") else SHORT)],
                       reasoning_effort="medium" if thinking else "none", stream=True,
                       stream_options=dict(include_usage=True), max_tokens=8000 if name.startswith("long") else 4096)
        if custom is not None:
            payload = custom[name].copy()
        payload.update(args.payload_overrides)
        write_json(case / "request.json", payload)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [python, "-m", "deepseek_v4_ssd.server", "--model", str(model),
                   "--host", "127.0.0.1", "--port", str(port), "--slots", "8192", "--prompt-cache-entries", "2",
                   "--prompt-cache-directory", str(case.resolve() / "prompt-cache")]
        if args.no_persistent_cache:
            command.append("--no-persistent-prompt-cache")
        if args.record_tokens:
            launcher = case.resolve() / "recording_server.py"
            launcher.write_text(
                "import json\nfrom dataclasses import asdict\nimport mlx.core as mx\nfrom deepseek_v4_ssd.generation import ModelRuntime\n"
                "from deepseek_v4_ssd.server import main\noriginal = ModelRuntime.stream\n"
                + ("from mlx_lm import sample_utils\nsample_utils.categorical_sampling = lambda logits, temp: mx.random.categorical(logits * (1 / temp))\n"
                   if args.uncompiled_categorical else "")
                +
                "def record(self, *args, **kwargs):\n"
                + (f"    with mx.stream(self._generation_stream):\n        mx.random.seed({args.research_seed})\n" if args.research_seed is not None else "")
                + f"    with open({str(case.resolve() / 'runtime-request.json')!r}, 'w') as request:\n"
                "        json.dump({'prompt_tokens': self._encode_prompt(args[0]), 'options': asdict(args[1]), 'config': asdict(self.config)}, request)\n"
                "    iterator = original(self, *args, **kwargs)\n"
                f"    with open({str(case.resolve() / 'tokens.jsonl')!r}, 'a') as output:\n"
                "        try:\n            for piece in iterator:\n"
                "                output.write(json.dumps({'token': int(piece.token), 'index': piece.generation_tokens, 'finish': piece.finish_reason}) + '\\n')\n"
                "                output.flush()\n                yield piece\n"
                "        finally:\n            iterator.close()\n"
                "ModelRuntime.stream = record\nmain()\n"
            )
            command = [command[0], str(launcher), *command[3:]]
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
