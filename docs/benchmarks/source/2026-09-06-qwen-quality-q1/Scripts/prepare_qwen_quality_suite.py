"""Freeze Q1A cases before generation; official samples plus local state tasks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'scratch/qwen-quality-q1-2026-09-06/setup'


def main():
    code = json.loads((SETUP / 'humaneval-selected.json').read_text())
    math = json.loads((SETUP / 'gsm8k-selected.json').read_text())
    zh = [
        ('甲庫原有120件零件。週一移出35件給乙庫；乙庫驗出其中5件瑕疵，當天退回甲庫。週二甲庫又收到新貨18件。瑕疵品必須另外封存，不能計入可用庫存。乙庫尚未使用零件。', '回傳 {"甲庫實體總數":整數,"甲庫可用數":整數,"乙庫可用數":整數}。', {'甲庫實體總數':108,'甲庫可用數':103,'乙庫可用數':30}),
        ('出貨規則：一般件在週二與週五出貨；冷藏普通件只在週五出貨。未付款一律暫緩。急件可在週二額外出貨，不受冷藏限制，但不能略過付款條件。今天週二。A是已付款冷藏急件，B是未付款一般急件，C是已付款一般件，D是已付款冷藏普通件。', '按名稱排序回傳 {"今天出貨":[名稱],"暫緩":[名稱]}。', {'今天出貨':['A','C'],'暫緩':['B','D']}),
        ('團隊有四件工作。P無前置條件，需2天；Q須等P完成，需3天；R須等P完成，需1天；S須等Q及R都完成，需2天。人力足夠，能平行工作。從第0天開始，工作完成當下下一項即可開始。', '回傳 {"最早完成日":整數,"S最早開始日":整數,"關鍵順序":[工作名稱]}。', {'最早完成日':7,'S最早開始日':5,'關鍵順序':['P','Q','S']}),
        ('公告甲：9月10日上午會議在三樓。公告乙在甲之後發布：9月10日上午會議改到五樓，時間不變。公告丙最新：只有下午場改為線上，上午場仍按公告乙。張同學報名上午場，李同學報名下午場。', '回傳 {"張同學地點":字串,"李同學地點":字串,"上午時間是否改變":布林值}。', {'張同學地點':'五樓','李同學地點':'線上','上午時間是否改變':False}),
        ('系統規則：管理員可修改及刪除資料；編輯者可修改但不能刪除；停權帳號不論角色均無操作權。小安是管理員但已停權；小白是啟用中的編輯者；小成是啟用中的管理員。', '名單固定依小安、小白、小成的順序排列，回傳 {"可刪除":[姓名],"可修改":[姓名]}。', {'可刪除':['小成'],'可修改':['小白','小成']}),
        ('三台機器的紀錄如下：A運作8小時，其中停機2小時；B排定8小時，全程運作；C排定6小時，其中停機1小時。本紀錄中的A運作8小時是排定時段長度，不是扣除停機後的時間。有效工時等於排定時數減停機時數。', '回傳 {"有效總工時":整數,"有效工時最多的機器":名稱,"A有效工時":整數}。', {'有效總工時':19,'有效工時最多的機器':'B','A有效工時':6}),
        ('報名名額為3位。候補依登記順序遞補：甲、乙、丙、丁、戊。最初甲乙丙錄取；乙取消後由丁遞補；甲取消後由戊遞補；乙重新登記，排在現有候補最後，不恢復原資格。', '按原始登記順序回傳 {"目前錄取":[名稱],"目前候補":[名稱]}。', {'目前錄取':['丙','丁','戊'],'目前候補':['乙']}),
        ('專案預算500元。已支付設備180元、材料95元。另有尚未支付但已承諾的費用75元。可自由運用餘額須扣除已支付與已承諾費用。有一筆40元的草案尚未核准，不算承諾；之後材料退費20元已入帳。', '回傳 {"已支付淨額":整數,"承諾未付":整數,"可自由運用餘額":整數}。', {'已支付淨額':255,'承諾未付':75,'可自由運用餘額':170}),
    ]
    tools = [
        ('User: Reserve 3 units of SKU A; if unavailable, reserve 2 of SKU B.\nAssistant: inventory({"sku":"A"})\nTool: {"available":1}\nAssistant: inventory({"sku":"B"})\nTool: {"available":4}', 'reserve(sku:string, quantity:integer)', {'name':'reserve','arguments':{'sku':'B','quantity':2}}),
        ('User: Move report.txt to folder archive.\nAssistant: find_file({"name":"report.txt"})\nTool: {"id":"f7","parent":"inbox"}\nAssistant: find_folder({"name":"archive"})\nTool: {"id":"d9"}', 'move_file(file_id:string, destination_id:string)', {'name':'move_file','arguments':{'file_id':'f7','destination_id':'d9'}}),
        ('User: Cancel my most recent pending order, leave shipped orders alone.\nAssistant: list_orders({})\nTool: [{"id":"o3","status":"shipped","created":3},{"id":"o2","status":"pending","created":2},{"id":"o1","status":"pending","created":1}]', 'cancel_order(order_id:string)', {'name':'cancel_order','arguments':{'order_id':'o2'}}),
        ('User: Set room temperature to 22 C, but do not change it if the room is occupied.\nAssistant: room_status({"room":"R1"})\nTool: {"occupied":true,"temperature_c":25}', 'set_temperature(room:string, celsius:integer), finish(reason:string). For an occupied room, finish with reason "occupied".', {'name':'finish','arguments':{'reason':'occupied'}}),
        ('User: Add 2 apples to my basket, then check out with standard delivery.\nAssistant: add_item({"sku":"apple","quantity":2})\nTool: {"ok":true,"basket_id":"b8","items":[{"sku":"apple","quantity":2}]}', 'checkout(basket_id:string, delivery:string)', {'name':'checkout','arguments':{'basket_id':'b8','delivery':'standard'}}),
        ('User: Rename task t4 to Review and mark it done.\nAssistant: rename_task({"id":"t4","title":"Review"})\nTool: {"ok":false,"error":"version_conflict","current_version":7}\nThe tool requires retrying the failed change with current_version before any later change.', 'rename_task(id:string, title:string, version:integer), complete_task(id:string)', {'name':'rename_task','arguments':{'id':'t4','title':'Review','version':7}}),
        ('User: Book the earliest available slot after 13:00 on 2026-09-10, lasting 30 minutes.\nAssistant: list_slots({"date":"2026-09-10"})\nTool: [{"id":"s1","start":"13:00","duration_minutes":30,"available":true},{"id":"s2","start":"13:15","duration_minutes":30,"available":false},{"id":"s3","start":"13:30","duration_minutes":30,"available":true},{"id":"s4","start":"14:00","duration_minutes":30,"available":true}]\nHere after means strictly later.', 'book_slot(slot_id:string)', {'name':'book_slot','arguments':{'slot_id':'s3'}}),
        ('User: Transfer 4 units from warehouse W1 to W2.\nAssistant: prepare_transfer({"source":"W1","destination":"W2","quantity":4})\nTool: {"status":"prepared","transfer_id":"x6","version":2}\nAssistant: commit_transfer({"transfer_id":"x6","version":2})\nTool: {"status":"already_committed","transfer_id":"x6"}\nNever create a second transfer for an already committed request.', 'commit_transfer(transfer_id:string, version:integer), finish(reason:string). For an already committed transfer, finish with reason "completed".', {'name':'finish','arguments':{'reason':'completed'}}),
    ]
    cases=[]
    for i in range(8):
        c,m=code[i],math[i]
        cases.append(dict(id=f'code-{c["index"]:03}',category='code',grader='python',prompt='Write a complete Python module implementing the specification below. Include all required imports and helper functions. Return code only.\n'+c['prompt'],response_instruction='',test=c['test'],entry_point=c['entry_point'],reference=c['prompt']+c['canonical_solution'],source_id=c['task_id']))
        cases.append(dict(id=f'math-{m["index"]:04}',category='math',grader='math',prompt=m['question']+'\nSolve carefully. End with a line exactly of the form Answer: number. A brief explanation before it is allowed.',response_instruction='',expected=m['answer'].rsplit('####',1)[1].strip(),reference=m['answer'],source_id=f'GSM8K/test/{m["index"]}'))
        body,query,expected=zh[i]
        cases.append(dict(id=f'zh-{i+1:02}',category='zh',grader='json',prompt='只根據以下紀錄作答，不補充未提供的事實。\n'+body+'\n'+query,expected=expected))
        transcript,schema,expected=tools[i]
        cases.append(dict(id=f'tool-{i+1:02}',category='tool',grader='json',prompt='Continue this tool-use transcript with exactly one next action. These are simulated tools.\n'+transcript+'\nAllowed calls: '+schema+'\nReturn {"name":tool_name,"arguments":arguments_object}.',expected=expected))
    suite=dict(schema_version=2,quality_q1=True,target_tokens=4096,max_tokens=1024,protocol='research/QWEN_QUALITY_Q1_2026-09-06.md',cases=cases,sources={n:json.loads((SETUP/(n+'-source.json')).read_text()) for n in ['humaneval','gsm8k']},limits=['32-task expanded objective screen; not a full public benchmark or adoption result.','Code and math use fixed hash-selected public test examples; checkpoint training overlap is unknown.','Chinese and tool tasks are locally authored; tool cases continue fixed multistep transcripts, not live agent episodes.','4K includes marked inert padding; this does not validate real long-document understanding or long-form answer quality.','Greedy, thinking off, one output per task; sampling and thinking mode remain untested.','Stop on the first control-correct/candidate-wrong pair. Confirm it with one reversed pair, without changing questions or code.','Incomplete and early-stopped samples receive no fixed-sample confidence interval; no category offsets another.','The code child denies home/project reads, file writes, networking, fork, signals and Mach IPC, with CPU/time/output limits; it is not a VM and does not claim a hard memory cap.'])
    dest=ROOT/'docs/benchmarks/prompts/2026-09-06-qwen-quality-q1.json'
    dest.write_text(json.dumps(suite,ensure_ascii=False,indent=2)+'\n')
    print(dest)


if __name__=='__main__':
    main()
