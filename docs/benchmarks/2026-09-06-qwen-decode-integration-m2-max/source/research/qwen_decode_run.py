"""Isolated control/arena runner using the ordinary CLI and ANE status hook."""
import argparse
import json
from pathlib import Path
import sys

import qwen_a_run
from qwen_decode_arena import install


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("control", "arena"), required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.status.exists() or not args.cli or args.cli[0] != "--":
        parser.error("new status file and CLI arguments after -- required")
    state = install() if args.mode == "arena" else None
    sys.argv = ["qwen_a_run", "--mode", "control", "--status", str(args.status), *args.cli]
    try:
        qwen_a_run.main()
    finally:
        if args.status.exists():
            status = json.loads(args.status.read_text())
            status["decode_mode"] = args.mode
            status["arena_state"] = state
            args.status.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
