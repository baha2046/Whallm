"""Actual Codex SIGINT -> Whallm cancellation, with fixture or installed Qwen.

Binds an ephemeral loopback port and never accesses the user's API key/config.
"""
from dataclasses import asdict
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from deepseek_v4_ssd.cancellation import check_cancelled
from deepseek_v4_ssd.generation import ModelRuntime, GeneratedPiece
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.server import OpenAIServer
from deepseek_v4_ssd.tool_codec import AssistantTurn
from runtime.tests.test_server import FakeRuntime

OUT = Path(__file__).resolve().parent
REAL = '--real' in sys.argv
LABEL = 'qwen' if REAL else 'fixture'
MODEL = 'qwen3.8-flash-next-fp8'
config = RuntimeConfig(slots=1152, persistent_prompt_cache=False)
entered = threading.Event()
closed = threading.Event()
state = {'pieces': 0, 'tokens': []}


def loader(spec):
    runtime = ModelRuntime.open(spec.path, spec.runtime) if REAL else FakeRuntime()
    if not REAL:
        runtime.installed = SimpleNamespace(root=Path('/tmp/model'), is_qwen=True, has_dspark=False)
        def fake(prompt, options):
            if not state.get('recovering'):
                for _ in range(3000):
                    check_cancelled()
                    time.sleep(0.01)
            runtime.parsed_turn = AssistantTurn('OK', '', ())
            yield GeneratedPiece('OK', 1, 5, 1, 'stop')
        runtime.stream = fake
    original = runtime.stream
    def tracked(prompt, options):
        entered.set()
        try:
            for piece in original(prompt, options):
                state['pieces'] += 1
                state['tokens'].append(piece.token)
                yield piece
        finally:
            closed.set()
    runtime.stream = tracked
    return runtime

manager = ModelManager([ModelSpec(MODEL, None,
    str(Path.home()/'.dsmodel/qwen3.8-flash-next.dsv4'), 'qwen3.8-flash-next',
    config, ModelDefaults(128, 0, 1, 0))], runtime_loader=loader)
server = OpenAIServer(('127.0.0.1', 0), manager, log_level='error')
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
record = {'model': MODEL if REAL else 'deterministic fixture', 'runtime': asdict(config),
          'machine': platform.platform(), 'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
          'codex_version': subprocess.check_output(['codex','--version'], text=True).strip(),
          'cache_state': 'new runtime; persistent prompt cache disabled', 'cases': []}


def wait_idle(timeout=30):
    assert manager._generation_lock.acquire(timeout=timeout), 'generation lock still held'
    manager._generation_lock.release()
    assert not server.metrics.snapshot()['generating']


def recover():
    state['recovering'] = True
    state['tokens'] = []
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=120)
    conn.request('POST', '/v1/responses', json.dumps({'model': MODEL,
        'input': 'Reply with exactly OK.', 'stream': True, 'max_output_tokens': 8,
        'temperature': 0, 'reasoning': {'effort': 'none'}}), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    assert response.status == 200
    events = [json.loads(line[6:]) for line in response.read().splitlines() if line.startswith(b'data: ')]
    response.close(); conn.close()
    assert events[-1]['type'] == 'response.completed', events[-1]
    wait_idle()
    result = {'response': events[-1]['response'], 'output_token_hash': hashlib.sha256(json.dumps(state['tokens']).encode()).hexdigest()}
    state['recovering'] = False
    return result

try:
    with tempfile.TemporaryDirectory(prefix='whallm-cancel-codex-') as work:
        home = Path(work)/'codex-home'; home.mkdir()
        provider = ('{name="Whallm cancel probe",base_url="http://127.0.0.1:' + str(server.server_port)
                    + '/v1",wire_api="responses",stream_max_retries=0,request_max_retries=0,supports_websockets=false}')
        command = ['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
                   '--sandbox','read-only','--json','--color','never','--cd',work,
                   '-m',MODEL,'-c','model_provider="fixture"','-c','model_context_window=272000',
                   '-c','model_auto_compact_token_limit=240000','-c','model_providers.fixture='+provider,
                   'Write a detailed explanation of HTTP request cancellation. Do not run tools.']
        with (OUT/f'{LABEL}-codex.stdout.jsonl').open('w') as stdout, (OUT/f'{LABEL}-codex.stderr.log').open('w') as stderr:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                env={**os.environ, 'CODEX_HOME': str(home)}, start_new_session=True)
            try:
                assert entered.wait(180), 'Codex did not start generation'
                time.sleep(0.5)
                before = state['pieces']
                started = time.monotonic()
                # Terminal Ctrl+C delivers SIGINT to the foreground process group.
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=10)
                assert closed.wait(30), 'runtime did not cancel'
                wait_idle()
                record['cases'].append({'case':'codex_sigint', 'exit_code': proc.returncode,
                    'idle_seconds_after_sigint': time.monotonic()-started,
                    'pieces_before_interrupt': before, 'pieces_after_interrupt': state['pieces']-before})
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    first = recover()
    record['recovery_after_codex'] = first
    if REAL:
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=120)
        conn.request('POST','/v1/responses',json.dumps({'model':MODEL,'input':'Count from 1 to 1000, one number per line.',
            'stream':True,'max_output_tokens':128,'temperature':0,'reasoning':{'effort':'none'}}),{'Content-Type':'application/json'})
        response=conn.getresponse(); assert response.status==200
        for line in response:
            if line.startswith(b'data: ') and json.loads(line[6:])['type']=='response.output_text.delta':
                break
        else:
            raise AssertionError('no decode output before EOF')
        before = state['pieces']; closed.clear(); started=time.monotonic()
        response.close(); conn.close()
        assert closed.wait(30), 'decode did not cancel'
        wait_idle()
        record['cases'].append({'case':'http_close_during_decode', 'idle_seconds_after_close':time.monotonic()-started,
            'pieces_before_interrupt':before, 'pieces_after_interrupt':state['pieces']-before})
        second = recover()
        record['recovery_after_decode'] = second
        assert first['output_token_hash'] == second['output_token_hash'], 'recovery token outputs differ'
    record['status'] = 'PASSED'
finally:
    (OUT/f'{LABEL}-probe.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    server.shutdown(); server.server_close(); worker.join(); manager.close()
