import os,time,subprocess,sys,json
from pathlib import Path
root=Path("docs/benchmarks/2026-09-06-issue6-causal-m2-max")
while True:
    try: os.kill(78947,0)
    except ProcessLookupError: break
    time.sleep(5)
base=[sys.executable,"research/issue6_reproduce.py"]
cases=[
 ["--output",str(root/"sampler-only-http-seed-20260906"),"--runtime-path",str(root/"baseline-source"),"--cases","long-thinking","--payload-overrides",json.dumps({"max_tokens":64}),"--research-seed","20260906","--no-persistent-cache","--record-tokens","--uncompiled-categorical"],
 ["--output",str(root/"current-source-seed-20260907"),"--runtime-path","runtime","--dependency-path",str(root/"mlx-0.32.1"),"--python-executable",".venv/bin/python","--cases","long-chat","short-thinking","short-chat","long-thinking","--research-seed","20260907","--no-persistent-cache","--record-tokens"],
 ["--output",str(root/"current-source-regression"),"--runtime-path","runtime","--dependency-path",str(root/"mlx-0.32.1"),"--python-executable",".venv/bin/python","--requests-json",str(root/"regression-requests.json"),"--research-seed","20260907","--no-persistent-cache","--record-tokens"]]
for case in cases:
    if (root/"STOP-FOLLOWON").exists(): break
    subprocess.run(base+case,check=True)
