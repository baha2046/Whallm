import argparse, dataclasses, hashlib, importlib.metadata, json, platform, subprocess, sys, time
from pathlib import Path
import mlx.core as mx
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions
from deepseek_v4_ssd.throughput import run_trial
p=argparse.ArgumentParser();p.add_argument('--lengths',type=int,nargs='+',default=[1024,4096,8192]);p.add_argument('--moe-step',type=int,default=0);p.add_argument('--output',required=True);p.add_argument('--tokens',type=int,default=128);p.add_argument('--profile',action='store_true');a=p.parse_args()
config=RuntimeConfig(slots=1024,prefill_step_size=128,moe_prefill_step_size=a.moe_step,prompt_cache_entries=0,expert_eviction_policy='lru')
options=GenerationOptions(max_tokens=a.tokens,temperature=0.6,top_p=0.95)
artifact={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'environment':{'platform':platform.platform(),'device':mx.device_info(),'packages':{k:importlib.metadata.version(k) for k in ['mlx','mlx-lm']}},'config':dataclasses.asdict(config),'options':dataclasses.asdict(options),'cache_state':'new runtime; consecutive requests; prompt cache off; OS page cache not purged','formal_performance_result':False,'instrumented':a.profile,'runs':[]}
source=Path('runtime/deepseek_v4_ssd');artifact['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'model.py',source/'generation.py',source/'expert_cache.py',source/'throughput.py']}
runtime=ModelRuntime.open('/Users/yanun/.dsmodel/deepseek-v4-flash-0731.dsv4',config)
trace=[]
def tracer(frame,event,arg):
 if frame.f_code.co_name != 'layer_major_prefill':return None
 if event=='line' and frame.f_lineno in (259,342,395,408,411):
  trace.append({'line':frame.f_lineno,'layer':frame.f_locals.get('layer_index'),'active':mx.get_active_memory(),'peak':mx.get_peak_memory()})
 return tracer
try:
 for length in a.lengths:
  trace=[];mx.random.seed(41)
  before={'active':mx.get_active_memory(),'resident_experts':runtime.expert_cache.resident_count};print('START',length,'moe_step',a.moe_step, 'before', before,flush=True)
  if a.profile:sys.settrace(tracer)
  r=run_trial(runtime,options,length,lambda x:x,lambda n:None)
  sys.settrace(None)
  r['before']=before;r['active_after']=mx.get_active_memory();r['runtime_metrics']=runtime.metrics.snapshot();r['trace']=trace
  artifact['runs'].append(r);Path(a.output).write_text(json.dumps(artifact,indent=2))
  print('RESULT',length,'peak GiB',r['peak_memory_bytes']/1024**3,'active GiB',r['active_after']/1024**3,'ttft',r['ttft_ms'],flush=True)
finally:runtime.close()
