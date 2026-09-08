"""Research-only demand-first native Decode groups with shared-expert overlap."""
from collections import Counter
from concurrent.futures import wait,FIRST_COMPLETED,ALL_COMPLETED
from contextlib import contextmanager
import time
from unittest.mock import patch
import mlx.core as mx
import numpy as np
from deepseek_v4_ssd.cancellation import check_cancelled
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import SparseMoE
from research.qwen_resident_block import BoundedArenaPool,direct_project


@contextmanager
def demand_lease(cache,layer,selected):
    """Reserve/touch once, protect all source slots through the final GPU fence."""
    check_cancelled()
    if cache._active_speculative_prefetch is not None:raise ValueError('speculative scratch composition unsupported')
    frequencies=Counter(selected);unique=sorted(frequencies)
    if not 0<=layer<cache.layer_count or len(unique)>cache.slots or any(not 0<=e<cache.model.expert_count for e in unique):raise ValueError('invalid expert demand')
    protected={(layer,e) for e in unique};assigned={};futures={};added=set();success=False
    missing=[];resident=[];started=None
    try:
        with cache._lock:
            added=protected-cache._speculative_pinned_keys
            cache._speculative_pinned_keys.update(added)
            for expert in unique:
                entry=cache._entries.get((layer,expert))
                if entry is None:cache.metrics.misses+=1;missing.append(expert)
                else:
                    cache.metrics.hits+=1;cache._touch(layer,expert,entry,frequencies[expert]);resident.append(expert)
            cache._record_expert_union_locked(layer,len(selected),len(unique),len(missing))
            assigned=cache._reserve_slots(layer,missing,frequencies,protected)
        cache._record_residency(layer,selected,missing)
        started=time.perf_counter()
        for expert in missing:
            futures[expert]=cache._executor.submit(cache._read_expert_into_slot,layer,expert,assigned[expert])
        yield resident,futures,assigned
        success=True
    finally:
        finished=[]
        try:
            for f in futures.values():
                try:finished.append(f.result())
                except Exception:pass
            mx.synchronize()
        finally:
            with cache._lock:
                cache.metrics.bytes_read+=len(finished)*cache.model.expert_blob_size
                if finished and started is not None:cache.metrics.read_seconds+=max(finished)-started
                if not success:cache._release_slots(layer,assigned)
                else:cache._decay_if_needed()
                cache._speculative_pinned_keys.difference_update(added)


def ready_experts(value,indices,cache,layer,shared,state):
    selected=np.asarray(indices,dtype=np.int32).reshape(-1).tolist()
    outputs=[];positions=[];groups=0
    with demand_lease(cache,layer,selected) as (resident,futures,assigned):
        mx.async_eval(shared)
        remaining=set(futures);ready=set(resident)
        while remaining or ready:
            check_cancelled()
            ready.update(e for e in remaining if futures[e].done())
            if not ready:
                wait([futures[e] for e in remaining],return_when=ALL_COMPLETED if groups>=2 else FIRST_COMPLETED)
                ready.update(e for e in remaining if futures[e].done())
            if groups>=2 and remaining-ready:
                wait([futures[e] for e in remaining],return_when=ALL_COMPLETED);ready.update(remaining)
            for e in sorted(ready & remaining):
                futures[e].result();cache._pool.mark_loaded(assigned[e])
            remaining.difference_update(ready)
            order=[i for i,e in enumerate(selected) if e in ready]
            with cache._lock:slots=[cache._entries[(layer,selected[i])].slot for i in order]
            output=direct_project(value,cache._pool,slots,len(order));mx.async_eval(output)
            outputs.append(output);positions.extend(order);groups+=1;ready.clear()
        result=mx.take(mx.concatenate(outputs,axis=-2),mx.array(np.argsort(positions)),axis=-2)
        mx.eval(result)
        cache.record_gather_qmm(2*groups)
        state['calls']+=1;state['groups']+=groups;state['max_groups']=max(state['max_groups'],groups)
    return result


@contextmanager
def install_ready_groups(mode):
    if mode not in ('control','native','ready'):raise ValueError('unknown mode')
    state=dict(calls=0,groups=0,max_groups=0)
    original_init=ExpertCache.__init__;original_moe=SparseMoE.__call__
    def initialize(self,*a,**kw):
        original_init(self,*a,**kw)
        if mode!='control':
            try:
                if self.staged_expert_streaming or self.qwen_grouped_decode:raise ValueError('unsupported cache mode')
                self._pool=BoundedArenaPool(self.model,self.slots)
            except BaseException:self.close();raise
    def call(self,value):
        if mode=='control' or value.shape!=(1,1,2560) or self.cache.current_batched(self.layer) is not None:return original_moe(self,value)
        probabilities=mx.softmax(self.gate(value),axis=-1,precise=True)
        indices=mx.argpartition(probabilities,kth=-self.top_k,axis=-1)[...,-self.top_k:]
        scores=mx.take_along_axis(probabilities,indices,axis=-1)
        if self.norm_topk_prob:scores=scores/scores.sum(axis=-1,keepdims=True)
        if self.cache.route_trace_enabled:self.cache.record_routes(self.layer,np.asarray(indices,dtype=np.int32))
        shared=mx.sigmoid(self.shared_expert_gate(value))*self.shared_expert(value)
        if mode=='ready':routed=ready_experts(value,indices,self.cache,self.layer,shared,state)
        else:
            selected=np.asarray(indices,dtype=np.int32).reshape(-1).tolist()
            self.cache.get_many(self.layer,selected)
            with self.cache._lock:
                slots=[self.cache._entries[(self.layer,e)].slot for e in selected]
                routed=direct_project(value,self.cache._pool,slots,len(slots));mx.eval(routed)
            self.cache.record_gather_qmm(2);state['calls']+=1;state['groups']+=1;state['max_groups']=1
        return (routed*scores[...,None].astype(routed.dtype)).sum(axis=-2)+shared
    with patch.object(ExpertCache,'__init__',initialize),patch.object(SparseMoE,'__call__',call):yield state
