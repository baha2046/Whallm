import hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd(); folder=root/'scratch/qwen-nohint-validation-2026-09-06/default-on'
folder.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
n3=json.loads((root/'docs/benchmarks/2026-09-06-qwen-nohint-n3-integration-m5-pro.json').read_text())
old=next(r for r in n3['performance_runs'] if r['case']=='code-1024' and r['variant']=='candidate')
args=old['command'][old['command'].index('--')+1:]
args=[v for v in args if v not in ('--qwen-grouped-experts','--no-qwen-grouped-experts')]
args[args.index('--metrics-json')+1]=str(folder/'metrics.json')
command=[sys.executable,str(root/'Scripts/benchmark_qwen_nohint_integration.py'),'--cli-record',str(folder/'observation.json'),'--',*args]
files=list((root/'runtime/deepseek_v4_ssd').glob('*.py'))+[root/'Scripts/benchmark_qwen_nohint_integration.py']
source={str(p.relative_to(root)):sha(p) for p in files}
record=dict(status='running',formal_performance_result=False,purpose='Verify the newly enabled default without an explicit flag against the previously validated 256-token N3 output. No new speed claim.',started_at=datetime.now(timezone.utc).isoformat(),command=command,source_sha256=source,environment=n3['environment'],model=n3['installed_model'],cache_state=n3['cache_state'],reference_token_sha256=old['metrics']['token_sha256'])
(folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
with (folder/'output.txt').open('w') as stdout,(folder/'stderr.txt').open('w') as stderr:
 subprocess.run(command,env=os.environ|{'PYTHONPATH':str(root/'runtime')},stdout=stdout,stderr=stderr,check=True)
m=json.loads((folder/'metrics.json').read_text());obs=json.loads((folder/'observation.json').read_text())
assert all(sha(root/n)==h for n,h in source.items())
assert m['qwen_grouped_experts'] and not m['mtp_enabled']
assert m['generated_token_ids']==old['metrics']['generated_token_ids'] and m['token_sha256']==old['metrics']['token_sha256']
assert m['prompt_token_sha256']==old['metrics']['prompt_token_sha256'] and not m['prompt_cache_reused_tokens']
assert obs['sorting_calls']==192 and obs['status']=='completed'
record.update(status='completed',ended_at=datetime.now(timezone.utc).isoformat(),metrics=m,observation=obs,output=(folder/'output.txt').read_text(),files_sha256={p.name:sha(p) for p in folder.iterdir() if p.name!='result.json'})
(folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'default_enabled':m['qwen_grouped_experts'],'sorting_calls':obs['sorting_calls'],'generated_tokens':m['generated_tokens'],'output_matches_n3':True}))
