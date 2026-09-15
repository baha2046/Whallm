"""Allocate the shared expert pool using decayed use count times observed miss cost."""
import heapq


def install_cost_hotness():
    from deepseek_v4_ssd.expert_cache import ExpertCache
    initialize = ExpertCache.__init__
    touch = ExpertCache._touch
    read = ExpertCache._read_expert_into_pool

    def init(self, *args, **kwargs):
        kwargs["eviction_policy"] = "lfu"
        initialize(self, *args, **kwargs)
        self._layer_reserve = 0
        self._cost_active = [1.0] * self.layer_count
        self._cost_sum = [0.0] * self.layer_count
        self._cost_count = [0] * self.layer_count

    def touched(self, layer, expert, entry, count):
        entry.ablation_layer = layer
        touch(self, layer, expert, entry, count)

    def rank(self, entry):
        return entry.frequency * self._cost_active[entry.ablation_layer]

    def measured_read(self, pool, layer, expert, slot):
        result = read(self, pool, layer, expert, slot)
        with self._lock:
            self._cost_sum[layer] += result.finished - result.started
            self._cost_count[layer] += 1
        return result

    def decay(self):
        interval = 16 * self.layer_count * self.model.selected_expert_count
        if self._clock - self._last_decay < interval:
            return
        self._last_decay = self._clock
        # Prices change only with the complete heap rebuild; no stale weighted ranks.
        total = sum(self._cost_count)
        average = sum(self._cost_sum) / total if total else 1.0
        for layer, count in enumerate(self._cost_count):
            estimate = self._cost_sum[layer] / count if count else average
            self._cost_active[layer] = max(0.25, min(4.0, estimate / max(average, 1e-9)))
            self._cost_sum[layer] *= 0.5
            self._cost_count[layer] *= 0.5
        self._heap = []
        for (layer, expert), entry in self._entries.items():
            entry.frequency = max(1, entry.frequency // 2)
            entry.version += 1
            self._heap.append((rank(self, entry), entry.last_access, entry.version, layer, expert))
        heapq.heapify(self._heap)

    ExpertCache.__init__, ExpertCache._touch = init, touched
    ExpertCache._eviction_rank, ExpertCache._decay_if_needed = rank, decay
    ExpertCache._read_expert_into_pool = measured_read
