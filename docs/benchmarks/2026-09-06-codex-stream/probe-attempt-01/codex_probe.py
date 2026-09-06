"""Exercise the installed Codex SSE client with slow, deterministic model output."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import tempfile
import threading
import time

from runtime.tests.test_server import FakeRuntime
from deepseek_v4_ssd import server as api
from deepseek_v4_ssd.generation import GeneratedPiece
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.tool_codec import AssistantTurn, ToolCall

OUT = Path(__file__).resolve().parent


def run(label, heartbeat):
    runtime = FakeRuntime()
    runtime.installed = SimpleNamespace(root=Path('/tmp/model'), is_qwen=True, has_dspark=False)
    runtime.calls = 0
    def stream(prompt, options):
        runtime.calls += 1
        time.sleep(1.5)
        runtime.parsed_turn = (
            AssistantTurn('', '', (ToolCall('exec_command', '{"cmd":"printf stream-ok"}'),))
            if runtime.calls == 1 else AssistantTurn('stream-ok', '', ())
        )
        yield GeneratedPiece('fixture', 1, 5, 1, 'stop')
    runtime.stream = stream
    manager = ModelManager([
        ModelSpec('qwen3.8-flash-next-fp8', None, '/tmp/model', 'qwen3.8-flash-next',
                  RuntimeConfig(), ModelDefaults(64, 0, 1, 0))
    ], runtime_loader=lambda _: runtime, clear_cache=lambda: None)
    api.RESPONSE_HEARTBEAT_SECONDS = heartbeat
    server = api.OpenAIServer(('127.0.0.1', 0), manager, log_level='error')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = ('{name="Whallm probe",base_url="http://127.0.0.1:' + str(server.server_port)
                + '/v1",wire_api="responses",stream_idle_timeout_ms=500,stream_max_retries=0,request_max_retries=0,supports_websockets=false}')
    with tempfile.TemporaryDirectory(prefix='whallm-codex-probe-') as work:
        command = ['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
                   '--sandbox','read-only','--json','--color','never','--cd',work,
                   '-m','qwen3.8-flash-next-fp8','-c','model_provider="fixture"',
                   '-c','model_providers.fixture='+provider,
                   'Run printf stream-ok once, then reply stream-ok.']
        started = time.monotonic()
        result = subprocess.run(command,capture_output=True,text=True,timeout=35)
        (OUT/f'{label}.stdout.jsonl').write_text(result.stdout)
        (OUT/f'{label}.stderr.log').write_text(result.stderr)
        record = {'label':label,'command':command,'exit_code':result.returncode,
                  'seconds':time.monotonic()-started,'model_requests':runtime.calls,
                  'heartbeat_seconds':heartbeat,'client_idle_timeout_ms':500,
                  'model_delay_seconds_per_request':1.5}
    server.shutdown(); server.server_close(); thread.join(); manager.close()
    return record

records = [run('codex-no-heartbeat', 3600), run('codex-heartbeat', 0.1)]
(OUT/'codex-probe.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
assert records[0]['exit_code'] != 0
assert records[1]['exit_code'] == 0
