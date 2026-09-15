"""Fresh-process full-model screening for two independent research candidates."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from qwen_packed_select_probe import make_attention
from ssd_streaming_ablation import swapouts, digest, summarize

ROOT=Path(__file__).resolve().parents[1]


def summarize_screen(rows):
    result=summarize(rows)
    # The reused SSD helper classifies unknown variants by Prefill/request time.
    # This study predeclares request OR Decode throughput as its speed signal.
    for entry in result["variants"].values():
        change=entry["median_change_percent"]
        entry["screening_speed_signal"]=(entry["pairs"]>=2 and entry["valid"] and
            (change["request_seconds"]<=-5 or change["decode_tokens_per_second"]>=5))
    return result


def worker(variant):
    calls=Counter()
    if variant == "resident_batch":
        source=textwrap.dedent(inspect.getsource(qwen.StreamingExperts.__call__))
        old="        outputs = {}\n        for expert, weights in self.cache.iter_ready"
        new="""        outputs = {}
        with self.cache._lock:
            resident_count = sum((self.layer, int(e)) in self.cache._entries for e in selected.reshape(-1))
        _calls[resident_count] += 1
        pending = []
        for expert, weights in self.cache.iter_ready"""
        assert source.count(old)==1
        source=source.replace(old,new)
        old="            mx.async_eval(output)"
        new="""            if len(outputs) < resident_count:
                pending.append(output)
                if len(pending) == resident_count:
                    mx.async_eval(pending)
            else:
                mx.async_eval(output)"""
        assert source.count(old)==1
        source=source.replace(old,new)
        namespace=dict(vars(qwen),_calls=calls)
        exec(compile(source,"<resident-submit-only>","exec"),namespace)
        qwen.StreamingExperts.__call__=namespace["__call__"]
    elif variant == "packed_select":
        from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
        from mlx_lm.models.cache import QuantizedKVCache
        original_update=QSAQuantizedCache.update_and_fetch
        original_attention=qwen.QSAAttention._bounded_attention
        candidate,_=make_attention()
        def update(cache,keys,values):
            if cache.dimension==256 and keys.shape[2]==1 and cache.offset+1>2048:
                pairs=QuantizedKVCache.update_and_fetch(cache,keys,values)
                return tuple(SimpleNamespace(shape=(1,2,cache.offset,256),packed=p,bits=cache.bits) for p in pairs)
            return original_update(cache,keys,values)
        def attention(self,query,key,value,index_query,raw_index_keys,offset):
            if not hasattr(key,"packed"):
                return original_attention(self,query,key,value,index_query,raw_index_keys,offset)
            assert key.bits==value.bits==8 and query.shape[2]==1
            self._packed_key,self._packed_value,self._bits=key.packed,value.packed,key.bits
            calls["selected_attention_calls"]+=1
            try:
                return candidate(self,query,key,value,index_query,raw_index_keys,offset)
            finally:
                del self._packed_key,self._packed_value,self._bits
        QSAQuantizedCache.update_and_fetch=update
        qwen.QSAAttention._bounded_attention=attention
    metrics=Path(sys.argv[sys.argv.index("--metrics-json")+1])
    from deepseek_v4_ssd.cli import main
    try:
        main()
    finally:
        Path(str(metrics)+".research.json").write_text(json.dumps(dict(variant=variant,calls=calls,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--candidate",choices=("resident_batch","packed_select"),required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    prompt=(ROOT/"docs/benchmarks/prompts/2026-09-05-qwen-research-baseline/zh_technical-4096.txt"
            if args.candidate=="resident_batch" else
            ROOT/"docs/benchmarks/prompts/2026-09-06-qwen-nohint-n1/code-16384.prompt.txt")
    model=Path.home()/".dsmodel/qwen3.8-flash-next.dsv4"
    artifact=dict(status="running",formal_performance_result=False,evidence_kind="paired_full_model_screen",
                  candidate=args.candidate,commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  device=mx.device_info(),prompt=str(prompt),prompt_sha256=digest(prompt),
                  manifest_sha256=digest(model/"manifest.json"),
                  sources={str(p.relative_to(ROOT)):digest(p) for p in
                           [Path(__file__).resolve(),ROOT/"research/qwen_packed_select_probe.py",
                            ROOT/"runtime/deepseek_v4_ssd/qwen4_exp.py",ROOT/"runtime/deepseek_v4_ssd/expert_cache.py",
                            ROOT/"runtime/deepseek_v4_ssd/qwen_quantized_cache.py"]},
                  cache_state="Fresh process, empty expert slots, prompt cache off; OS cache not purged or identically conditioned.",
                  stop="Reject speed claim on token mismatch, swapout, extra MLX/RSS >1 GB; 5% request/decode screening signal.",runs=[])
    def save():
        (args.output/"result.json").write_text(json.dumps(artifact,indent=2)+"\n")
    save()
    try:
        for index,variant in enumerate(["control",args.candidate,args.candidate,"control"]):
            prefix=args.output/f"{index}-{variant}"
            command=[sys.executable,str(Path(__file__).resolve()),"--worker",variant,
                     "--model",str(model),"--prompt-file",str(prompt),"--max-tokens","64",
                     "--temperature","0","--top-p","1","--top-k","0","--approximation","exact",
                     "--slots","3072","--expert-eviction-policy","lru","--read-workers","4",
                     "--prefetch-read-workers","2","--memory-limit-gib","48","--prompt-cache","off",
                     "--separate-prefill-io","--metrics-json",str(prefix)+".json"]
            if args.candidate=="packed_select":command += ["--qwen-quantized-kv","--no-qwen-quantized-index"]
            before=swapouts();start=time.time()
            print(f"START {args.candidate} {index} {variant}",flush=True)
            with Path(str(prefix)+".stdout").open("w") as out,Path(str(prefix)+".stderr").open("w") as err:
                subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/"runtime")),
                               stdout=out,stderr=err,check=True,timeout=1800)
            metrics=json.loads(Path(str(prefix)+".json").read_text())
            row=dict(wave=index//2,variant=variant,command=command,process_seconds=time.time()-start,
                     swapout_delta=swapouts()-before,metrics=metrics,
                     effective_worker=json.loads(Path(str(prefix)+".json.research.json").read_text()))
            artifact["runs"].append(row);save()
            print(f"DONE TTFT={metrics['time_to_first_token_seconds']:.3f}s Decode={metrics['decode_tokens_per_second']:.3f}tok/s swapout={row['swapout_delta']}",flush=True)
            if metrics['generated_tokens']!=64:raise RuntimeError("unequal output workload")
            if len({r['metrics']['token_sha256'] for r in artifact['runs']})>1:
                raise RuntimeError("full-model token exactness gate failed")
            if row['swapout_delta']:raise RuntimeError("swapout gate failed")
        artifact["comparison"]=summarize_screen(artifact["runs"])
        artifact["status"]="complete"
    except BaseException as error:
        artifact["status"]="stopped";artifact["error"]=str(error)
        raise
    finally:save()


if __name__=="__main__":
    if len(sys.argv)>2 and sys.argv[1]=="--worker":
        variant=sys.argv[2];sys.argv=[sys.argv[0],*sys.argv[3:]];worker(variant)
    else:main()
