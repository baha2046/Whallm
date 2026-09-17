"""Budget planner used by the opt-in Qwen phase-memory policy.

The extra prefill workspace is reserved *inside* the existing expert-byte
ceiling. Other model, KV, MTP and DSpark budgets are unchanged. This does not
estimate macOS availability or guarantee total physical footprint.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class QwenPhaseBudget:
    expert_ceiling_bytes: int
    expert_blob_bytes: int
    selected_experts: int
    prefill_workspace_bytes: int

    def __post_init__(self):
        for name in ("expert_ceiling_bytes", "expert_blob_bytes", "selected_experts", "prefill_workspace_bytes"):
            value = getattr(self, name)
            minimum = 0 if name == "prefill_workspace_bytes" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.prefill_slots < self.selected_experts:
            raise ValueError("prefill workspace leaves insufficient routed-expert capacity")

    @property
    def prefill_slots(self):
        return (self.expert_ceiling_bytes - self.prefill_workspace_bytes) // self.expert_blob_bytes

    @property
    def decode_slots(self):
        return self.expert_ceiling_bytes // self.expert_blob_bytes

    @property
    def prefill_expert_bytes(self):
        return self.prefill_slots * self.expert_blob_bytes

    @property
    def decode_expert_bytes(self):
        return self.decode_slots * self.expert_blob_bytes
