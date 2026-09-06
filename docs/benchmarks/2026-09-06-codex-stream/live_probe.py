"""Bounded localhost Responses probe; credentials stay in process memory."""
import hashlib
import http.client
import json
from pathlib import Path
import re
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
pid = sys.argv[1]
raw = subprocess.check_output(['ps','eww','-p',pid,'-o','command='],text=True)
match = re.search(r'(?:^| )DEEPSEEK_API_KEY=([^ ]+)',raw)
headers = {'Content-Type':'application/json'}
if match:
    headers['Authorization'] = 'Bearer ' + match.group(1)
connection = http.client.HTTPConnection('127.0.0.1',11434,timeout=30)
connection.request('GET','/api/status',headers=headers)
status_response = connection.getresponse()
assert status_response.status == 200
status = json.loads(status_response.read())
assert not status['performance']['generating'] and not status['loading_model'], 'Server is busy'
connection.close()
context = '\n'.join(f'Reference {i}: Taipei is the requested city; this reference is background context only.' for i in range(100))
payload = {'model':'qwen3.8-flash-next-fp8','input':context+'\nCall get_weather with city Taipei exactly once. Do not answer in prose.',
           'tools':[{'type':'function','name':'get_weather','description':'Return weather for a city.',
                     'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}],
           'tool_choice':{'type':'function','name':'get_weather'},'reasoning':{'effort':'none'},
           'stream':True,'max_output_tokens':128,'temperature':0}
(out/'live-request.json').write_text(json.dumps(payload,indent=2)+'\n')
connection = http.client.HTTPConnection('127.0.0.1',11434,timeout=30)
started = time.monotonic()
connection.request('POST','/v1/responses',json.dumps(payload),headers)
response = connection.getresponse()
assert response.status == 200, response.status
events=[]
with (out/'live-events.jsonl').open('w') as log:
    for line in response:
        if time.monotonic()-started > 300:
            raise TimeoutError('Probe exceeded 300 seconds')
        if not line.startswith(b'data: '): continue
        event=json.loads(line[6:])
        record={'seconds':time.monotonic()-started,'event':event}
        events.append(record)
        log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
connection.close()
last=events[-1]['event']
assert last['type']=='response.completed' and last['response']['status']=='completed',last
calls=[item for item in last['response']['output'] if item['type']=='function_call']
assert len(calls)==1 and calls[0]['name']=='get_weather' and json.loads(calls[0]['arguments'])=={'city':'Taipei'},calls
sequence=[item['event']['sequence_number'] for item in events]
assert sequence==list(range(len(sequence)))
heartbeat=[item for item in events if item['event']['type']=='response.in_progress']
assert heartbeat, 'Workload finished without exercising heartbeat'
summary={'status':'PASSED','server_pid':int(pid),'endpoint':'http://127.0.0.1:11434/v1/responses',
         'seconds':events[-1]['seconds'],'events':len(events),'heartbeats':len(heartbeat),
         'first_event_seconds':events[0]['seconds'],
         'max_event_gap_seconds':max(b['seconds']-a['seconds'] for a,b in zip(events,events[1:])),
         'usage':last['response']['usage'],'tool_calls':calls,'model':'qwen3.8-flash-next-fp8',
         'tool_executed':False,'source_sha256':hashlib.sha256(Path('runtime/deepseek_v4_ssd/server.py').read_bytes()).hexdigest()}
(out/'live-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
