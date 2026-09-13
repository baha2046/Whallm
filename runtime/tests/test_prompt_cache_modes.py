from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd import server
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.model import RuntimeConfig, _apply_prompt_cache_mode
from deepseek_v4_ssd.model_support import get_support


def raw_model(kind):
    return {
        "id": get_support(kind).descriptor.api_model_id,
        "alias": None,
        "path": f"/tmp/{kind}",
        "model_kind": kind,
        "runtime": asdict(RuntimeConfig()),
        "defaults": {"max_tokens": 256, "temperature": 0.2, "top_p": 0.98, "top_k": 0},
        "warmup_prompt_path": None,
    }


class PromptCacheModeTests(unittest.TestCase):
    def test_modes_reuse_only_allowed_state_and_disk_survives_restart(self):
        class Cache:
            def __init__(self):
                self.state = [mx.array([0], dtype=mx.int32)]
                self.nbytes = 4

        for mode in ('off', 'memory', 'disk'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                received, restored = [], []

                def generate(_model, _tokenizer, prompt, **options):
                    received.append(list(prompt))
                    cache = options['prompt_cache'][0]
                    restored.append(int(cache.state[0].item()))
                    callback = options['prompt_progress_callback']
                    callback(0, len(prompt))
                    if len(prompt) > 1:
                        cache.state[0] = mx.array([47], dtype=mx.int32)
                        callback(len(prompt) - 1, len(prompt))
                        # Subsequent generation must not overwrite a saved prefix.
                        cache.state[0] = mx.array([999], dtype=mx.int32)
                    yield SimpleNamespace(text='A', token=42, generation_tokens=1, finish_reason='length')

                installed = SimpleNamespace(root=root / 'model', revision='test', model_id='fixture/model')
                config = _apply_prompt_cache_mode(replace(RuntimeConfig(), layer_major_prefill=False,
                    prompt_cache_directory=str(root / 'cache')), mode)
                tokenizer = SimpleNamespace(bos_token=None, encode=lambda *_args, **_kwargs: list(range(48)))
                with patch('deepseek_v4_ssd.generation.load_model', return_value=(object(), SimpleNamespace(close=lambda: None))), patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained', return_value=tokenizer), patch('deepseek_v4_ssd.model_support.state.make_prompt_cache', side_effect=lambda _: [Cache()]), patch('deepseek_v4_ssd.generation.stream_generate', side_effect=generate):
                    runtime = ModelRuntime(installed, config)
                    for _ in range(2):
                        list(runtime.stream('same', GenerationOptions(max_tokens=1)))
                    if mode == 'off':
                        self.assertEqual(runtime._prompt_caches, [])
                    runtime.close()
                    restarted = ModelRuntime(installed, config)
                    list(restarted.stream('same', GenerationOptions(max_tokens=1)))
                    restarted.close()
                self.assertEqual([len(p) for p in received], {
                    'off': [48, 48, 48], 'memory': [48, 1, 48], 'disk': [48, 1, 1],
                }[mode])
                self.assertEqual(restored, {'off': [0, 0, 0], 'memory': [0, 47, 0], 'disk': [0, 47, 47]}[mode])
                self.assertEqual((root / 'cache').exists(), mode == 'disk')

    def test_catalog_uses_explicit_cli_overrides_and_preserves_other_settings(self):
        for flags in ([], ['--no-persistent-prompt-cache', '--prompt-cache-directory=/tmp/custom', '--prompt-cache-entries', '7', '--slots', '900', '--default-temperature', '0.4'], ['--prompt-cache', 'off'], ['--prompt-cache', 'memory'], ['--prompt-cache', 'disk']):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                model = raw_model('qwen3.8-flash-next')
                model['runtime'].update(slots=4096, persistent_prompt_cache=True)
                path = Path(temporary) / 'catalog.json'
                path.write_text(json.dumps({'version': 1, 'models': [model]}))
                with patch.object(sys, 'argv', ['server', '--model-catalog', str(path), *flags]), patch.object(server, 'ModelManager') as manager, patch.object(server, 'OpenAIServer'), contextlib.redirect_stdout(io.StringIO()):
                    server.main()
                spec = manager.call_args.args[0][0]
                config = spec.runtime
                if flags and flags[0] == '--no-persistent-prompt-cache':
                    self.assertFalse(config.persistent_prompt_cache)
                    self.assertEqual(config.prompt_cache_directory, '/tmp/custom')
                    self.assertEqual(config.prompt_cache_entries, 7)
                    self.assertEqual(config.slots, 900)
                    self.assertEqual(spec.defaults.temperature, 0.4)
                else:
                    self.assertEqual(config.slots, 4096)
                    self.assertEqual(config.persistent_prompt_cache, not flags or flags[-1] == 'disk')
                    self.assertEqual(config.prompt_cache_entries, 0 if flags == ['--prompt-cache', 'off'] else 2)

    def test_catalog_validates_overrides_against_inherited_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = raw_model('deepseek-v4')
            model['runtime']['dspark_enabled'] = True
            path = Path(temporary) / 'catalog.json'
            path.write_text(json.dumps({'version': 1, 'models': [model]}))
            with (
                patch.object(sys, 'argv', ['server', '--model-catalog', str(path), '--dspark-prompt-cache']),
                patch.object(server, 'ModelManager') as manager,
                patch.object(server, 'OpenAIServer'),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                server.main()
            config = manager.call_args.args[0][0].runtime
            self.assertTrue(config.dspark_enabled)
            self.assertTrue(config.dspark_prompt_cache)
            with (
                patch.object(sys, 'argv', ['server', '--model-catalog', str(path), '--slots', '0']),
                patch.object(server, 'OpenAIServer') as http_server,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                server.main()
            http_server.assert_not_called()

    def test_default_is_memory_and_conflicting_modes_are_rejected(self):
        self.assertFalse(RuntimeConfig().persistent_prompt_cache)
        self.assertEqual(RuntimeConfig().prompt_cache_entries, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            server._parser().parse_args(['--prompt-cache', 'disk', '--no-persistent-prompt-cache'])
