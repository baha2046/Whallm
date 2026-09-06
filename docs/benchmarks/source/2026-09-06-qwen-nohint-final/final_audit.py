import hashlib
import json
import shutil
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
sys.path[:0] = ['.', 'Scripts', 'runtime']
from analyze_qwen_nohint_validation import summarize, token_hash
from benchmark_qwen_nohint_integration import performance_summary
root = Path.cwd()
base = Path('docs/benchmarks')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name): return json.loads((base / name).read_text())
n1 = read('2026-09-06-qwen-nohint-n1-runs-m5-pro.json')
n2 = read('2026-09-06-qwen-nohint-n2-cache-m5-pro.json')
n3 = read('2026-09-06-qwen-nohint-n3-integration-m5-pro.json')
indexes = {}
for stage in ('n1', 'n2', 'n3'):
 p = base / f'2026-09-06-qwen-nohint-{stage}-index.json'
 d = json.loads(p.read_text())
 for name, expected in d['files_sha256'].items(): assert sha(name) == expected, name
 indexes[str(p)] = {'sha256': sha(p), 'verified_files': len(d['files_sha256'])}
assert summarize(n1) == read('2026-09-06-qwen-nohint-n1-gate-m5-pro.json')['summary']
assert performance_summary(n3['performance_runs']) == n3['performance_summary']
assert n3['performance_summary']['passed']
assert len(n3['performance_runs']) == 8 and len(n3['cache_workers']) == 4
for row in n3['performance_runs']:
 m = row['metrics']; old = next(r for r in n1['runs'] if r['case'] == row['case'])
 assert m['generated_token_ids'] == old['metrics']['generated_token_ids'][:256]
 assert m['token_sha256'] == token_hash(m['generated_token_ids'])
 assert row['text_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
 assert m['prompt_token_sha256'] == old['metrics']['prompt_token_sha256']
 assert row['observation']['sorting_calls'] == (192 if row['case'].endswith('1024') else 768) * (row['variant'] == 'candidate')
for case in ('code-1024','code-16384'):
 for wave in (0,1):
  a,b = [r['metrics'] for r in n3['performance_runs'] if r['case']==case and r['wave']==wave]
  for k in ('generated_token_ids','request_expert_bytes_read','request_gather_qmm_calls'): assert a[k] == b[k]
cold = {b: next(w['result']['runs'][0] for w in n2['workers'] if w['name'] == f'cold-control-{b}') for b in ('A','B')}
cache_count = 0
for worker in n2['workers'] + n3['cache_workers']:
 r = worker['result']; assert r['status'] == 'completed'
 if 'shared_before' in r: assert r['shared_before'] == r['shared_after']
 for row in r['runs']:
  ref = cold[row['branch']]; m = row['metrics']; cache_count += 1
  assert row['generated_token_ids'] == ref['generated_token_ids'] and row['text'] == ref['text']
  assert row['token_sha256'] == token_hash(row['generated_token_ids'])
  assert row['prompt_token_sha256'] == n2['suite']['branches'][row['branch']]['token_sha256']
  assert m['prompt_cache_reused_tokens'] == row['expected_reused_tokens']
  assert not m['prompt_cache_write_errors'] and not m['mtp_enabled']
  if r['plan']['action'] != 'cold': assert m['prompt_cache_reused_tokens'] >= 1024
  if r['plan'].get('integrated'):
   enabled = r['plan']['variant'] == 'candidate'
   assert r['config']['qwen_grouped_experts'] == enabled
   assert (row['modified_calls'] > 0) == enabled
seed = [w['result'] for w in n2['workers'] if w['name'].startswith('seed-')]
assert seed[0]['shared_before']['metadata'] == seed[1]['shared_before']['metadata']
assert seed[0]['shared_before']['payload'] == seed[1]['shared_before']['payload']
assert len(n1['runs']) + len(n3['performance_runs']) + cache_count == 66
source = n3['source']['files_sha256']
changed_runtime = [n for n,h in source.items() if n.startswith('runtime/deepseek_v4_ssd/') and sha(n) != h]
assert changed_runtime == ['runtime/deepseek_v4_ssd/model_manager.py'], changed_runtime
modules = ['test_qwen','test_runtime','test_cli','test_block_prompt_cache','test_qwen_grouped_experts','test_qwen_nohint_validation','test_model_manager','test_server']
def ids(suite):
 for test in suite:
  if isinstance(test, unittest.TestSuite): yield from ids(test)
  else: yield test.id()
test_ids = list(ids(unittest.defaultTestLoader.loadTestsFromNames(['runtime.tests.'+m for m in modules])))
assert len(test_ids) == len(set(test_ids)) == 220, len(test_ids)
dest = base / 'source/2026-09-06-qwen-nohint-final'
assert not dest.exists()
files = ['runtime/deepseek_v4_ssd/'+n+'.py' for n in ('model','qwen4_exp','cli','model_manager')]
files += ['runtime/tests/'+m+'.py' for m in modules]
for name in files:
 target = dest / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name,target)
shutil.copy2(__file__, dest / 'final_audit.py')
result = dict(verified_at=datetime.now(timezone.utc).isoformat(), formal_performance_result=False,
 source_commit=n3['source']['commit'], working_tree_dirty=True, default_enabled=False,
 model_requests={'n1':32,'n2':18,'n3_performance':8,'n3_cache':8,'total':66,'additional_cache_warmups':2},
 artifact_indexes=indexes, all_output_hashes_and_cache_contracts_passed=True,
 n1_gate_recomputed=True,n3_gate_recomputed=True,
 post_measurement_change={'files':changed_runtime,'scope':'Only catalog parsing: accept omitted new boolean as false and validate explicit types. CLI and direct ModelRuntime measurements bypass this parser.',
  'unchanged_generation_sources':{n:h for n,h in source.items() if n.startswith('runtime/deepseek_v4_ssd/') and n not in changed_runtime}},
 tests={'unique_passed':220,'ids':test_ids,'recorded_runs':[
  {'command':'PYTHONPATH=runtime .venv/bin/python -m unittest '+' '.join('runtime.tests.'+m for m in modules[:6]),'passed':139,'seconds':0.236,'note':'Before addition of the incomplete N3 evidence guard.'},
  {'command':'PYTHONPATH=runtime .venv/bin/python -m unittest runtime.tests.test_qwen_nohint_validation','passed':5,'note':'Includes four previously counted tests and one new incomplete N3 evidence guard.'},
  {'command':'PYTHONPATH=runtime .venv/bin/python -m unittest runtime.tests.test_model_manager runtime.tests.test_server','passed':80,'seconds':3.791,'note':'After catalog compatibility fix.'}],
  'audit_action':'Test identities enumerated without re-executing model or test runs.'},
 final_source_sha256={str(p):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},
 limits=['Single M5 Pro, fixed prompts, two reversed pairs per performance case; OS cache not purged.',
 'N1 native binary provenance first recorded after request 15, checked unchanged at end; Python source frozen before start.',
 'N1 capped 1024-token generation validates output parity and work, not completed-task correctness.',
 'N3 measured source retained separately; catalog-only compatibility fix validated afterward.'])
(base/'2026-09-06-qwen-nohint-final-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verified_indexes':indexes,'unique_test_ids':len(test_ids),'requests':66,'post_n3_runtime_changes':changed_runtime},indent=2))
