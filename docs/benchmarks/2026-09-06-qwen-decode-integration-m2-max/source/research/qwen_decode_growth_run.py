"""Observe allocation immediately after model load, without changing graph math."""
import json
from pathlib import Path
import sys

import mlx.core as mx
from deepseek_v4_ssd.generation import ModelRuntime
import qwen_a_run


def main():
    observed = []
    original = ModelRuntime.__init__
    def initialize(runtime, installed, config):
        original(runtime, installed, config)
        raw = sum(t.length for t in installed.common_tensors)
        observed.append({"common_manifest_bytes": raw, "planning_floor_bytes": raw * 9 // 10,
                         "active_after_load_bytes": mx.get_active_memory(),
                         "floor_satisfied": mx.get_active_memory() >= raw * 9 // 10,
                         "page_counts": getattr(runtime.expert_cache._pool, "page_counts", None)})
    ModelRuntime.__init__ = initialize
    status_path = Path(sys.argv[sys.argv.index("--status") + 1])
    try:
        qwen_a_run.main()
    finally:
        if status_path.exists():
            status = json.loads(status_path.read_text())
            status["load_allocation"] = observed
            status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
