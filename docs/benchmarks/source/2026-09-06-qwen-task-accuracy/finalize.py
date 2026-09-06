from pathlib import Path
import hashlib,json,sys
from collections import Counter
root=Path.cwd()
sys.path.insert(0,str(root/'Scripts'))
from benchmark_qwen_task_accuracy import score
raw=root/'scratch/qwen-task-accuracy-2026-09-06-v2'
d=json.loads((raw/'results.json').read_text())
assert d['status']=='screen_completed'
suite=json.loads((raw/'suite.json').read_text())
cases={c['id']:c for c in suite['cases']}
assert len(d['runs'])==24 and len(d['pairs'])==12
assert len({(r['case'],r['variant']) for r in d['runs']})==24
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for r in d['runs']:
 assert r['passed']==score(r['text'],cases[r['case']]['expected'])
 assert not r['truncated']
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 assert hashlib.sha256(','.join(map(str,r['metrics']['generated_token_ids'])).encode()).hexdigest()==r['metrics']['token_sha256']
 if r['variant']=='candidate':
  assert r['record']['modified_calls']==192 and r['record']['sorting_hint']
for p in d['pairs']:
 rows={r['variant']:r for r in d['runs'] if r['case']==p['case']}
 assert rows['control']['metrics']['prompt_token_sha256']==rows['candidate']['metrics']['prompt_token_sha256']
 assert p['regression']==(rows['control']['passed'] and not rows['candidate']['passed'])
summary={}
for category in ['code','math','zh','tool']:
 pairs=[p for p in d['pairs'] if p['category']==category]
 summary[category]=dict(total=len(pairs),control_correct=sum(p['control_passed'] for p in pairs),candidate_correct=sum(p['candidate_passed'] for p in pairs),regressions=sum(p['regression'] for p in pairs),gains=sum(not p['control_passed'] and p['candidate_passed'] for p in pairs),both_wrong=sum(not p['control_passed'] and not p['candidate_passed'] for p in pairs),output_parity=sum(p['output_parity'] for p in pairs))
d['category_summary']=summary
d['quality_decision']='no_regression_observed_in_small_screen_not_adoption_evidence'
d['raw_directory']=str(raw.relative_to(root))
d['raw_results_sha256']=sha(raw/'results.json')
d['finalizer_sha256']=sha(Path(__file__))
dest=root/'docs/benchmarks/2026-09-06-qwen-task-accuracy-screen-m5-pro.json'
dest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
prompts=root/'docs/benchmarks/prompts/2026-09-06-qwen-task-accuracy'
prompts.mkdir(exist_ok=True)
manifest=[]
for case in suite['cases']:
 p=raw/(case['id']+'.prompt.txt')
 (prompts/p.name).write_bytes(p.read_bytes())
 manifest.append(dict(case=case['id'],file=p.name,text_sha256=sha(p)))
(prompts/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sources=root/'docs/benchmarks/source/2026-09-06-qwen-task-accuracy'
sources.mkdir(exist_ok=True)
for name in ['Scripts/benchmark_qwen_task_accuracy.py','Scripts/research_qwen_sorted_experts.py','research/QWEN_TASK_ACCURACY_2026-09-06.md']:
 p=raw/'source-at-start'/name
 assert sha(p)==d['source']['files_sha256'][name]
 (sources/p.name).write_bytes(p.read_bytes())
(sources/'finalize.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary,indent=2))
print('All 24 runs, 12 pairs, prompt hashes, output hashes and scores verified')
