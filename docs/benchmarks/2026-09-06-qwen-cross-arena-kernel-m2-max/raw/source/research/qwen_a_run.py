"""Fresh-process CLI runner for an isolated Qwen A control/candidate.

No production flags/defaults or persistent cache formats are changed. All
arguments after -- go to the ordinary CLI. Capture ANE status before teardown.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.ane_prefill import ANEPrefillController
from qwen_a_qsa_candidate import build_direct


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode",choices=("control","direct-prefill"),required=True)
    parser.add_argument("--status",type=Path,required=True)
    parser.add_argument("cli",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.cli or args.cli[0]!="--" or args.status.exists():
        parser.error("use new --status path and CLI arguments after --")
    if "--no-persistent-prompt-cache" not in args.cli:
        parser.error("research runner requires persistent prompt cache disabled")
    statuses=[]
    calls={"candidate":0,"control":0}
    close=ANEPrefillController.close

    def record_close(controller):
        statuses.append(controller.snapshot())
        close(controller)

    ANEPrefillController.close=record_close
    if args.mode=="direct-prefill":
        original=qwen.QSAAttention._bounded_attention
        candidate=build_direct()

        def dispatch(self,query,*a,**kw):
            active=query.shape[2]>1
            calls["candidate" if active else "control"]+=1
            return (candidate if active else original)(self,query,*a,**kw)

        qwen.QSAAttention._bounded_attention=dispatch
    sys.argv=["deepseek_v4_ssd.cli",*args.cli[1:]]
    status="failed"
    try:
        runpy.run_module("deepseek_v4_ssd.cli",run_name="__main__")
        status="completed"
    finally:
        args.status.write_text(json.dumps(dict(status=status,mode=args.mode,ane=statuses,calls=calls),indent=2)+"\n")


if __name__=="__main__":
    main()
