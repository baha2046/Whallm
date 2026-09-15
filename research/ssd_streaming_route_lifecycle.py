"""Real-model route-history lifetime and cancellation checks; not a timing benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import subprocess

import numpy as np

from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model import RuntimeConfig


def token_hash(tokens):
    return hashlib.sha256(','.join(map(str, tokens)).encode()).hexdigest()


def run(code_suite, changed_suite, output):
    baseline = json.loads((code_suite / 'summary.json').read_text())
    changed = json.loads((changed_suite / 'summary.json').read_text())
    installed = InstalledModel.open(Path(baseline['installed_model']['path']))
    config = RuntimeConfig(slots=baseline['slots'], expert_eviction_policy='route',
                           memory_limit_gib=48, prompt_cache_entries=0,
                           qwen_grouped_experts=baseline['runs'][0]['metrics']['qwen_grouped_experts'],
                           persistent_prompt_cache=False)
    options = GenerationOptions(max_tokens=32, temperature=0, top_p=1, top_k=0)
    expected = [next(r['metrics']['generated_token_ids'][:32] for r in suite['runs']
                     if r['variant'] == 'control') for suite in (baseline, changed)]
    prompts = [(suite / 'prompt.txt').read_text() for suite in (code_suite, changed_suite)]
    result = {'evidence_kind': 'full_model_lifecycle_correctness',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'installed_model': baseline['installed_model'], 'config': asdict(config),
              'environment': baseline['environment'],
              'source_suite': str(code_suite), 'checks': []}
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    with ModelRuntime(installed, config) as runtime:
        policy = runtime.expert_cache._route_policy
        previous = 0
        first_prefill = None
        for name, prompt_index in [('first', 0), ('repeat', 0), ('changed_topic', 1)]:
            tokens = [int(piece.token) for piece in runtime.stream(prompts[prompt_index], options)]
            observed = int(policy.decode_tokens.sum())
            row = {'name': name, 'tokens': tokens, 'token_sha256': token_hash(tokens),
                   'expected_token_sha256': token_hash(expected[prompt_index]),
                   'output_exact': tokens == expected[prompt_index],
                   'history_advanced': observed > previous,
                   'route_cache': runtime.expert_cache.route_cache_snapshot()}
            if name == 'first':
                first_prefill = policy.prefill_tokens.copy()
            if name == 'repeat':
                row['prefill_reset_once'] = bool(np.array_equal(policy.prefill_tokens, first_prefill))
                assert row['prefill_reset_once']
            result['checks'].append(row)
            save()
            assert row['output_exact'] and row['history_advanced'], name
            previous = observed
        # Closing an in-progress request is the runtime's normal cancellation cleanup.
        response = runtime.stream(prompts[0], options)
        try:
            cancelled = [int(next(response).token) for _ in range(5)]
        finally:
            response.close()
        before_release = policy.long.copy()
        runtime.expert_cache.release_prefill_slots()
        survived = np.array_equal(policy.long, before_release)
        tokens = [int(piece.token) for piece in runtime.stream(prompts[0], options)]
        result['checks'].append({'name': 'cancel_release_retry',
            'cancelled_tokens': cancelled, 'history_survived_slot_release': survived,
            'tokens': tokens, 'token_sha256': token_hash(tokens),
            'expected_token_sha256': token_hash(expected[0]), 'output_exact': tokens == expected[0],
            'phase_restored': runtime.expert_cache._route_phase == 'decode',
            'route_cache': runtime.expert_cache.route_cache_snapshot()})
        save()
        assert survived and tokens == expected[0] and runtime.expert_cache._route_phase == 'decode'
    result['passed'] = True
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-suite', type=Path, required=True)
    parser.add_argument('--changed-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.code_suite, args.changed_suite, args.output)
