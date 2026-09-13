"""One full-model diagnostic pair per fresh process; not a production patch."""
import argparse, dataclasses, gc, hashlib, importlib.metadata, json, platform, subprocess, time
from pathlib import Path
import mlx.core as mx
import deepseek_v4_ssd.model as model_module
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.expert_cache import _SlotPool
from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions
from deepseek_v4_ssd.throughput import run_trial
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','release-slots','individual-256'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
config=RuntimeConfig(slots=1024,prefill_step_size=128,prompt_cache_entries=0,expert_eviction_policy='lru')
options=GenerationOptions(max_tokens=128,temperature=0.6,top_p=0.95)
root=Path('runtime/deepseek_v4_ssd');installed=Path('/Users/yanun/.dsmodel/deepseek-v4-flash-0731.dsv4')
artifact={'mode':a.mode,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'device':mx.device_info(),'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['mlx','mlx-lm']},'config':dataclasses.asdict(config),'options':dataclasses.asdict(options),'seed':41,'formal_performance_result':False,'cache_state':'new process; identical 1024/128 priming request; OS page cache not purged','source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'model.py',root/'expert_cache.py',root/'generation.py',root/'throughput.py',Path(__file__)]},'manifest_sha256':hashlib.sha256((installed/'manifest.json').read_bytes()).hexdigest()}
runtime=ModelRuntime.open(str(installed),config)
original=model_module.layer_major_prefill

def released_prefill(model,tokens,prompt_cache,step,cache,*args,**kwargs):
    # This isolated prototype is limited to ordinary V4 slots with no active request work.
    mx.synchronize()
    before=mx.get_active_memory();started=time.perf_counter()
    with cache._lock:
        assert type(cache._pool) is _SlotPool
        assert not cache._prefetched_layers and cache._batched_layer is None
        assert not cache._pinned_layers and not cache._speculative_pinned_keys
        assert cache._active_speculative_prefetch is None
        cache._entries.clear();cache._free_slots=list(reversed(range(cache.slots)))
        cache._heap.clear();cache._layer_counts=[0]*cache.layer_count
        cache._clock=0;cache._last_decay=0
        cache._pool=_SlotPool(cache.model,cache.slots)
    gc.collect();mx.clear_cache()
    artifact['release']={'before_active':before,'after_active':mx.get_active_memory(),'seconds':time.perf_counter()-started}
    return original(model,tokens,prompt_cache,step,cache,*args,**kwargs)

try:
    mx.random.seed(41);print('PRIME',a.mode,flush=True)
    artifact['prime']=run_trial(runtime,options,1024,lambda x:x,lambda n:None)
    print('PRIME DONE',a.mode,flush=True)
    artifact['before']={'active':mx.get_active_memory(),'resident_experts':runtime.expert_cache.resident_count,'resident_keys_sha256':hashlib.sha256(repr(sorted(runtime.expert_cache._entries.keys())).encode()).hexdigest()}
    if a.mode=='release-slots':model_module.layer_major_prefill=released_prefill
    if a.mode=='individual-256':runtime.config=dataclasses.replace(config,batched_expert_prefill=False,moe_prefill_step_size=256)
    artifact['trial_config']=dataclasses.asdict(runtime.config)
    mx.random.seed(41);print('TRIAL',a.mode,flush=True)
    artifact['result']=run_trial(runtime,options,4096,lambda x:x,lambda n:None)
    artifact['metrics']=runtime.metrics.snapshot();artifact['after_active']=mx.get_active_memory()
    Path(a.output).write_text(json.dumps(artifact,indent=2)+'\n')
    r=artifact['result'];print('RESULT',a.mode,'peak',r['peak_memory_bytes']/1024**3,'ttft',r['ttft_ms']/1000,'decode',r['decode_tps'],'total',r['elapsed_seconds'],flush=True)
finally:
    model_module.layer_major_prefill=original
    runtime.close()
