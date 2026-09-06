from __future__ import annotations

import http.client
import json
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

from runtime.tests.test_server import FakeRuntime
from deepseek_v4_ssd.cancellation import check_cancelled
from deepseek_v4_ssd.generation import GeneratedPiece
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.server import OpenAIServer
from deepseek_v4_ssd.tool_codec import AssistantTurn, ToolCall


class ResponseLivenessTests(unittest.TestCase):
    def setUp(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.closed = threading.Event()
        self.runtime = FakeRuntime()
        self.runtime.parsed_turn = AssistantTurn('', '', (ToolCall('lookup', '{}'),))
        def stream(prompt, options):
            self.entered.set()
            try:
                if not self.release.wait(3):
                    raise RuntimeError('test generation gate timed out')
                yield GeneratedPiece('tool', 1, 5, 1, 'stop')
            finally:
                self.closed.set()
        self.runtime.stream = stream
        self.manager = ModelManager([
            ModelSpec('deepseek-v4-flash-0731', None, '/tmp/model', 'deepseek-v4',
                      RuntimeConfig(), ModelDefaults(64, 0, 1, 0))
        ], runtime_loader=lambda _: self.runtime, clear_cache=lambda: None)
        self.interval = patch('deepseek_v4_ssd.server.RESPONSE_HEARTBEAT_SECONDS', 0.02, create=True)
        self.interval.start()
        self.server = OpenAIServer(('127.0.0.1', 0), self.manager, log_level='error')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=0.5)

    def tearDown(self):
        self.release.set()
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()
        self.interval.stop()

    def start(self, *, tools=True):
        payload = {'model':'deepseek-v4-flash-0731', 'input':'Hello', 'stream':True, 'max_output_tokens':64}
        if tools:
            payload.update(tools=[{'type':'function','name':'lookup','parameters':{'type':'object','properties':{}}}], tool_choice='required')
        self.connection.request('POST', '/v1/responses', json.dumps(payload), {'Content-Type':'application/json'})
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        first = self.event(response)
        self.assertEqual(first['type'], 'response.created')
        self.assertTrue(self.entered.wait(1))
        return response, first

    def event(self, response):
        while line := response.readline():
            if line.startswith(b'data: '):
                result = json.loads(line[6:])
                self.assertEqual(response.readline(), b'\n')
                return result
        self.fail('Stream closed before the next event')

    def test_buffered_tool_validation_stays_live_without_exposing_partial_tools(self):
        response, first = self.start()
        heartbeat = self.event(response)
        self.assertEqual(heartbeat['type'], 'response.in_progress')
        self.assertEqual(heartbeat['response']['id'], first['response']['id'])
        self.assertEqual(heartbeat['response']['output'], [])
        self.release.set()
        rest = [json.loads(line[6:]) for line in response.read().splitlines() if line.startswith(b'data: ')]
        events = [first, heartbeat, *rest]
        self.assertEqual([e['sequence_number'] for e in events], list(range(len(events))))
        self.assertEqual(events[-1]['type'], 'response.completed')
        self.assertEqual(events[-1]['response']['output'][0]['name'], 'lookup')

    def test_prefill_wait_without_tools_stays_live(self):
        response, _ = self.start(tools=False)
        self.assertEqual(self.event(response)['type'], 'response.in_progress')
        self.release.set()
        self.assertIn(b'"type":"response.completed"', response.read())

    def test_generation_error_has_a_terminal_response_instead_of_bare_eof(self):
        def broken(prompt, options):
            raise RuntimeError('injected generation failure')
            yield
        self.runtime.stream = broken
        payload = {'model':'deepseek-v4-flash-0731','input':'Hello','stream':True}
        self.connection.request('POST','/v1/responses',json.dumps(payload),{'Content-Type':'application/json'})
        response = self.connection.getresponse()
        body = response.read()
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b'data: ')]
        self.assertEqual(events[-1]['type'], 'response.failed')
        self.assertEqual(events[-1]['response']['status'], 'failed')
        self.assertNotIn(b'HTTP/1.1', body)

    def test_disconnect_cancels_buffered_generation_and_releases_model(self):
        produced = []
        def stream(prompt, options):
            self.entered.set()
            try:
                self.release.wait(3)
                for i in range(100):
                    produced.append(i)
                    yield GeneratedPiece('x', i, 5, i + 1, None)
            finally:
                self.closed.set()
        self.runtime.stream = stream
        response, _ = self.start()
        sock = response.fp.raw._sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        response.close()
        self.connection.close()
        # Wait for the heartbeat writer to observe the reset before generation resumes.
        threading.Event().wait(0.15)
        self.release.set()
        self.assertTrue(self.closed.wait(1))
        self.assertLess(len(produced), 100)
        self.assertTrue(self.manager._generation_lock.acquire(timeout=1))
        self.manager._generation_lock.release()

    def test_normal_close_cancels_tools_without_waiting_for_heartbeat(self):
        def stream(prompt, options):
            self.entered.set()
            try:
                for i in range(1000):
                    time.sleep(0.005)
                    yield GeneratedPiece('x', i, 5, i + 1, None)
            finally:
                self.closed.set()
        self.runtime.stream = stream
        with patch('deepseek_v4_ssd.server.RESPONSE_HEARTBEAT_SECONDS', 3600):
            response, _ = self.start()
            # Normal FIN, not the artificial RST used by the older test.
            response.close()
            self.connection.close()
            self.assertTrue(self.closed.wait(1), 'generation continued after client exit')
            self.assertTrue(self.manager._generation_lock.acquire(timeout=1))
            self.manager._generation_lock.release()

    def test_disconnected_queued_request_never_loads_or_generates(self):
        with patch.object(self.manager, '_ensure_loaded', wraps=self.manager._ensure_loaded) as load:
            self.manager._generation_lock.acquire()
            try:
                self.connection.request('POST', '/v1/responses', json.dumps({
                    'model': 'deepseek-v4-flash-0731', 'input': 'Hello', 'stream': True,
                }), {'Content-Type': 'application/json'})
                time.sleep(0.2)
                self.connection.close()
                time.sleep(0.3)
            finally:
                self.manager._generation_lock.release()
            self.assertFalse(self.entered.wait(0.3), 'abandoned queued request generated')
            load.assert_not_called()

    def test_disconnect_during_prefill_stops_before_first_token_then_recovers(self):
        def stream(prompt, options):
            self.entered.set()
            try:
                while not self.release.is_set():
                    check_cancelled()
                    time.sleep(0.005)
                yield GeneratedPiece('unused', 1, 5, 1, 'stop')
            finally:
                self.closed.set()
        self.runtime.stream = stream
        with patch('deepseek_v4_ssd.server.RESPONSE_HEARTBEAT_SECONDS', 3600):
            response, _ = self.start()
            response.close()
            self.connection.close()
            self.assertTrue(self.closed.wait(1))
            self.assertTrue(self.manager._generation_lock.acquire(timeout=1))
            self.manager._generation_lock.release()
        # Cancellation belongs to one request; the next request must still work.
        self.release.set()
        response, _ = self.start()
        self.assertIn(b'"type":"response.completed"', response.read())
