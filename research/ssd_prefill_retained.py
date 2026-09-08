"""Research-only layer-local persistent pages for multi-chunk Prefill.

Uses the stable pipeline's compute callback. Pages are never overwritten within
one layer; reuse across layers follows an explicit GPU fence and reader drain.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import os
import threading
import time
import mlx.core as mx
import numpy as np
from deepseek_v4_ssd.expert_cache import _SlotPool
from deepseek_v4_ssd.cancellation import check_cancelled
from research.ssd_prefill_pipeline import preadv_exact


class RetainedBanks:
    def __init__(self,model,bank_experts,budget_bytes,reader=preadv_exact):
        if not model.is_qwen or bank_experts<1 or budget_bytes<model.expert_count*model.expert_blob_size:
            raise ValueError('retained pages require one canonical Qwen layer payload budget')
        self.model,self.bank_experts,self.reader=model,bank_experts,reader
        self.pool=_SlotPool(replace(model,expert_count=bank_experts),0)
        self.pages={};self.mapping={};self.loaded=set();self.rows=[]
        self.executor=ThreadPoolExecutor(max_workers=2)
        self._running=threading.Lock();self._layer=None;self.closed=False;self.fd=None

    def _prepare(self,page):
        if page in self.pages:return
        # Full last page is permitted only for canonical divisible page layouts.
        count=min(self.bank_experts,self.model.expert_count-page*self.bank_experts)
        if count!=self.bank_experts: raise ValueError('partial physical page layout unsupported')
        buf=mx.empty((count*self.model.expert_blob_size//4,),mx.uint32);mx.eval(buf)
        weights=self.pool.batched(buf);mx.eval(*vars(weights).values())
        views=[self.pool.write_views(memoryview(buf).cast('B'),i) for i in range(count)]
        self.pages[page]=(buf,weights,views)

    @contextmanager
    def layer(self,layer):
        if self.closed or self._layer is not None:raise RuntimeError('layer scope unavailable')
        if not 0<=layer<self.model.layer_count:raise ValueError('invalid layer')
        self._layer=layer;self.mapping.clear();self.loaded.clear()
        try:
            self.fd=os.open(self.model.root/'experts'/f'layer_{layer:02d}.bin',os.O_RDONLY)
            yield
        finally:
            mx.synchronize()
            if self.fd is not None:os.close(self.fd)
            self.fd=None;self._layer=None;self.mapping.clear();self.loaded.clear()

    def _read(self,page,items,row):
        row['read_start']=time.perf_counter()
        try:
            for expert,slot in items:
                row['bytes_read']+=self.reader(self.fd,self.pages[page][2][slot],expert*self.model.expert_blob_size)
        finally:row['read_complete']=time.perf_counter()

    def run(self,layer,waves,consume,cancelled=check_cancelled):
        if self.closed or self._layer!=layer:raise RuntimeError('active layer scope required')
        if not self._running.acquire(False):raise RuntimeError('concurrent consumption unsupported')
        pending={};self.rows.clear()
        try:
            cancelled()
            positions_by_expert={}
            for ids,positions,local in waves:
                for i,expert in enumerate(ids):
                    if not 0<=expert<self.model.expert_count or expert in positions_by_expert:raise ValueError('invalid/duplicate wave expert')
                    positions_by_expert[expert]=np.asarray(positions)[np.asarray(local)==i]
            missing={}
            for expert in sorted(positions_by_expert):
                if expert not in self.mapping:
                    page,slot=divmod(len(self.mapping),self.bank_experts)
                    self._prepare(page);self.mapping[expert]=(page,slot)
                    missing.setdefault(page,[]).append((expert,slot))
                elif expert not in self.loaded:raise RuntimeError('failed read cannot be reused')
            groups={}
            for expert,pos in positions_by_expert.items():
                page,slot=self.mapping[expert]
                groups.setdefault(page,[]).append((expert,pos,slot))
            rows={}
            for page in groups:
                row=dict(layer=layer,page=page,bytes_read=0,read_start=time.perf_counter(),read_complete=time.perf_counter())
                rows[page]=row;self.rows.append(row)
            for page,items in missing.items():
                cancelled();pending[page]=self.executor.submit(self._read,page,items,rows[page])
            # Fully resident pages can compute while missing pages are being read.
            order=sorted(groups,key=lambda p:(p in missing,p))
            for page in order:
                cancelled()
                if page in pending:
                    pending[page].result()
                    self.loaded.update(e for e,_ in missing[page])
                row=rows[page];row['consume_start']=time.perf_counter()
                pos=np.concatenate([x[1] for x in groups[page]]).astype(np.int32)
                local=np.concatenate([np.full(len(x[1]),x[2],np.uint32) for x in groups[page]])
                output=consume(self.pages[page][1],pos,local)
                if output is not None:mx.eval(output)
                mx.synchronize();row['consumer_fence_complete']=time.perf_counter()
        finally:
            try:
                for f in pending.values():
                    try:f.result()
                    except Exception:pass
                mx.synchronize()
            finally:self._running.release()

    def close(self):
        with self._running:
            if not self.closed:
                self.executor.shutdown(wait=True);mx.synchronize();self.pages.clear();self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
