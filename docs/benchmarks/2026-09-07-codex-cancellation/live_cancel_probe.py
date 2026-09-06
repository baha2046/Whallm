"""Check the active packaged server; its existing credential stays in memory."""
import hashlib
import http.client
import json
from pathlib import Path
import re
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
pid = int(sys.argv[1])
raw = subprocess.check_output(['ps','eww','-p',str(pid),'-o','command='],text=True)
match = re.search(r'(?:^| )DEEPSEEK_API_KEY=([^ ]+)',raw)
headers = {'Content-Type':'application/json'}
if match: headers['Authorization'] = 'Bearer ' + match.group(1)

def status():
    conn = http.client.HTTPConnection('127.0.0.1',11434,timeout=10)
    conn.request('GET','/api/status',headers=headers)
    response=conn.getresponse(); assert response.status==200
    result=json.loads(response.read()); conn.close()
    return result

before=status()
assert not before['performance']['generating'] and not before['loading_model'], 'Server is busy'
conn=http.client.HTTPConnection('127.0.0.1',11434,timeout=180)
conn.request('POST','/v1/responses',json.dumps({'model':'qwen3.8-flash-next-fp8',
    'input':'Count from 1 to 1000, one number per line.','stream':True,
    'max_output_tokens':128,'temperature':0,'reasoning':{'effort':'none'}}),headers)
response=conn.getresponse(); assert response.status==200
for line in response:
    if line.startswith(b'data: ') and json.loads(line[6:])['type']=='response.output_text.delta': break
else: raise AssertionError('No text before EOF')
started=time.monotonic()
response.close(); conn.close()
while True:
    after=status()
    if not after['performance']['generating']: break
    assert time.monotonic()-started<30, 'Server still generating after cancellation'
    time.sleep(0.05)
seconds=time.monotonic()-started
conn=http.client.HTTPConnection('127.0.0.1',11434,timeout=120)
conn.request('POST','/v1/responses',json.dumps({'model':'qwen3.8-flash-next-fp8',
    'input':'Reply with exactly OK.','stream':True,'max_output_tokens':8,
    'temperature':0,'reasoning':{'effort':'none'}}),headers)
response=conn.getresponse(); assert response.status==200
items=[json.loads(line[6:]) for line in response.read().splitlines() if line.startswith(b'data: ')]
response.close(); conn.close()
assert items[-1]['type']=='response.completed'
final=status(); assert not final['performance']['generating']
record={'status':'PASSED','server_pid':pid,'url':'http://127.0.0.1:11434/v1/responses',
    'idle_seconds_after_close':seconds,'performance_after_cancel':after['performance'],
    'recovery_response':items[-1]['response'], 'final_server_idle':True,
    'packaged_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        Path('dist/Whallm.app/Contents/Resources/runtime/deepseek_v4_ssd').glob('*.py')}}
(out/'live-probe.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','server_pid','idle_seconds_after_close','final_server_idle']},indent=2))
