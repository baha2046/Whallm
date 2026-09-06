"""Fresh-process, explicit research kernel activation and execution observer."""
import json
import sys
from pathlib import Path

import qwen_decode_cross_arena as candidate
import qwen_decode_growth_run


if __name__ == "__main__":
    calls = [0]
    status = Path(sys.argv[sys.argv.index("--status") + 1])
    active = "--qwen-grouped-decode" in sys.argv
    if active:
        original = candidate.cross_arena
        def observe(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)
        candidate.cross_arena = observe
        candidate.install()
    try:
        qwen_decode_growth_run.main()
    finally:
        if status.exists():
            data = json.loads(status.read_text())
            data["cross_arena_calls"] = calls[0]
            data["cross_arena_requested"] = active
            status.write_text(json.dumps(data, indent=2) + "\n")
