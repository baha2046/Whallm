"""Keep unrelated benchmark callers working after removing experiment runners."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from Scripts.benchmark_common import _run


class CommonBenchmarkTests(unittest.TestCase):
    def test_normal_and_dspark_commands_use_supported_flags_and_check_prompt_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(command, **kwargs):
                self.assertIn('--no-separate-prefill-io', command)
                self.assertFalse({'--dspark-hash-prefetch', '--dspark-adaptive-block',
                                  '--dspark-hybrid-verification'} & set(command))
                self.assertEqual('--dspark' in command, mode == 'fixed')
                metrics = Path(command[command.index('--metrics-json') + 1])
                metrics.write_text(json.dumps({'prompt_token_sha256':'prompt', 'token_sha256':'output'}))
                return SimpleNamespace(returncode=0)
            for mode in ('normal', 'fixed'):
                with patch('Scripts.benchmark_common.subprocess.run', side_effect=run):
                    row = _run(project_root=root, python=Path('/python'), model=root,
                               prompt={'path':str(root/'prompt.txt'), 'name':'fixture',
                                       'prompt_token_sha256':'prompt'}, mode=mode, run_id=mode,
                               raw_directory=root, max_tokens=32, resume=False,
                               dspark_fallback_enabled=True, layer_major_prefill_enabled=True,
                               sequential_verification_enabled=False)
                    self.assertEqual(row['metrics']['token_sha256'], 'output')
            with self.assertRaisesRegex(RuntimeError, 'prompt token hash'):
                _run(project_root=root, python=Path('/python'), model=root,
                     prompt={'path':'unused', 'name':'fixture', 'prompt_token_sha256':'changed'},
                     mode='normal', run_id='normal', raw_directory=root, max_tokens=32,
                     resume=True, dspark_fallback_enabled=True, layer_major_prefill_enabled=True,
                     sequential_verification_enabled=False)
