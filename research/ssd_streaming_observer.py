"""Per-layer I/O observer. Its timings are excluded from speed comparisons."""
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def install_observer():
    from deepseek_v4_ssd.expert_cache import ExpertCache
    initialize, close = ExpertCache.__init__, ExpertCache.close
    read_one, read_ids = ExpertCache._read_expert_into_pool, ExpertCache._read_expert_ids
    ready, many = ExpertCache.iter_ready, ExpertCache.get_many

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._observed = {}
        self._observer_lock = threading.Lock()

    def add(self, layer, field, value):
        phase = self._route_trace._phase if self._route_trace else None
        key = (phase or "outside_request", layer)
        with self._observer_lock:
            row = self._observed.setdefault(key, {"phase": key[0], "layer": layer})
            row[field] = row.get(field, 0) + value

    def one(self, pool, layer, expert, slot):
        started = time.perf_counter()
        result = read_one(self, pool, layer, expert, slot)
        add(self, layer, "summed_worker_read_seconds", time.perf_counter() - started)
        add(self, layer, "logical_bytes", self.model.expert_blob_size)
        add(self, layer, "expert_read_count", 1)
        return result

    def ids(self, layer, packed, experts):
        started = time.perf_counter()
        result = read_ids(self, layer, packed, experts)
        add(self, layer, "summed_worker_read_seconds", time.perf_counter() - started)
        add(self, layer, "logical_bytes", len(experts) * self.model.expert_blob_size)
        add(self, layer, "expert_read_count", len(experts))
        return result

    def iter_ready(self, layer, experts):
        iterator = ready(self, layer, experts)
        try:
            while True:
                start = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    add(self, layer, "foreground_acquisition_seconds", time.perf_counter() - start)
                    break
                add(self, layer, "foreground_acquisition_seconds", time.perf_counter() - start)
                yield item
        finally:
            iterator.close()

    def get_many(self, layer, experts):
        started = time.perf_counter()
        result = many(self, layer, experts)
        add(self, layer, "foreground_acquisition_seconds", time.perf_counter() - started)
        return result

    def finish(self):
        close(self)
        if self._route_trace_path is None:
            return
        trace = json.loads(Path(self._route_trace_path).read_text())
        rows = list(self._observed.values())
        for layer, routes in enumerate(trace["decode_routes"]):
            masks = trace["decode_misses"][layer]
            assignments = sum(map(len, masks))
            misses = sum(sum(mask) for mask in masks)
            row = next((r for r in rows if r["phase"] == "decode" and r["layer"] == layer), None)
            if row is None:
                row = {"phase": "decode", "layer": layer}
                rows.append(row)
            row.update({"routed_tokens": len(routes), "assignments": assignments,
                        "misses": misses, "hit_rate": 1 - misses / assignments if assignments else None,
                        "misses_per_token": misses / len(routes) if routes else None,
                        "logical_bytes_per_token": row.get("logical_bytes", 0) / len(routes) if routes else None,
                        "gpu_idle_due_to_io_seconds": None})
        artifact = {"evidence_kind": "instrumented_layer_observer", "layers": rows,
                    "prefill_overlap": trace["prefetch_events"],
                    "limits": ["Worker read seconds sum concurrent reads; not wall time.",
                               "Foreground acquisition includes CPU bookkeeping and waiting for weights; not measured GPU idle.",
                               "Logical bytes are not physical SSD bytes.",
                               "Full-layer Prefill has no ordinary slot hit-rate denominator.",
                               "Do not use this instrumented run as the speed baseline."]}
        Path(str(self._route_trace_path) + ".layers.json").write_text(json.dumps(artifact, indent=2) + "\n")

    ExpertCache.__init__, ExpertCache.close = init, finish
    ExpertCache._read_expert_into_pool, ExpertCache._read_expert_ids = one, ids
    ExpertCache.iter_ready, ExpertCache.get_many = iter_ready, get_many
