import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('scratch/qwen-decode-2026-09-06/attempt-01')
assert json.loads((root/'quality/summary.json').read_text())['status'].startswith('passed-')
output = root/'zh-extended-summary.json'
assert not output.exists()
prompt = Path('docs/benchmarks/prompts/2026-09-06-qwen-decode-zh-extended.txt')
pair = dict(workload='zh_extended', prompt_length_label=4096, output_limit=256,
            prompt_file_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(), runs=[], status='running')
output.write_text(json.dumps(pair, indent=2)+'\n')
for mode in ('control', 'arena'):
    name = 'zh_extended-4096-'+mode
    metrics = root/'quality'/f'{name}.json'
    status = root/'quality'/f'{name}-status.json'
    assert not metrics.exists() and not status.exists()
    cmd = ['/usr/bin/time', '-l', sys.executable, 'research/qwen_decode_run.py', '--mode', mode,
           '--status', str(status), '--', '--model', '/Users/yanun/.dsmodel/qwen3.8-flash-next.dsv4',
           '--prompt-file', str(prompt), '--max-tokens', '256', '--temperature', '0', '--top-p', '1',
           '--top-k', '0', '--slots', '1152', '--no-persistent-prompt-cache', '--metrics-json', str(metrics)]
    print('START '+name, flush=True)
    start = time.monotonic()
    with (root/'quality'/f'{name}.log').open('w') as log:
        run = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH':'runtime:research'}, stdout=log, stderr=subprocess.STDOUT)
    pair['runs'].append(dict(mode=mode, command=cmd, metrics=metrics.name, status=status.name,
                             exit_code=run.returncode, client_wall_seconds=time.monotonic()-start))
    if run.returncode:
        pair['status']='failed-process';output.write_text(json.dumps(pair,indent=2)+'\n');sys.exit(1)
    observed=json.loads(status.read_text())
    assert observed['ane'] and all(a['active'] and not a['fallbacks'] for a in observed['ane'])
    if mode=='arena':assert observed['arena_state']['grouped_calls']>0
    output.write_text(json.dumps(pair, indent=2)+'\n')
a,b=[json.loads((root/'quality'/r['metrics']).read_text()) for r in pair['runs']]
pair['prompt_exact']=a['prompt_token_sha256']==b['prompt_token_sha256']
pair['output_exact']=a['generated_token_ids']==b['generated_token_ids']
pair['actual_prompt_tokens']=a['prompt_tokens']
pair['actual_output_tokens']=[a['generated_tokens'],b['generated_tokens']]
pair['long_decode_covered']=min(pair['actual_output_tokens'])>=256
pair['first_divergence_zero_based']=next((i for i,(x,y) in enumerate(zip(a['generated_token_ids'],b['generated_token_ids'])) if x!=y),None)
pair['status']='passed-exact-long-coverage' if pair['prompt_exact'] and pair['output_exact'] and pair['long_decode_covered'] else 'stopped-coverage-gate'
output.write_text(json.dumps(pair,indent=2)+'\n')
print(json.dumps({k:pair[k] for k in ['status','actual_prompt_tokens','actual_output_tokens','output_exact','first_divergence_zero_based']}),flush=True)
