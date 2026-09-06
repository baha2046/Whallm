# Qwen 任務正確率：Q1 擴充評估

沿用使用者指定的「各類任務正確率不下降」。Q0 的 12 題不重用，也不調整候選。
本輪先完成 Q1A 可自動核對的能力測試；Q1B 的長篇回答與實際多回合工具評估另列，不能混稱完成。

## Q1A 固定規則（執行前）

固定 32 題，每類 8 題，按程式、數學、繁中、工具交錯執行。
每題一組獨立 process 配對，相鄰題交替原版→候選、候選→原版。
候選保持 `sorted_indices=True`；Slot 4096、MTP false、thinking false、greedy。
輸入固定 4096 tokens，含明確標記的無關填充；生成上限 1024 tokens。
停用 persistent prompt cache，OS page cache 不清空；不同 GPU 測試不並行。
核對 prompt/output hashes、48 layers × 4 chunks = 192 次候選呼叫。
截斷、缺資料、評分器失敗都不能算通過。

### 題目與評分

- 程式：從 [HumanEval 官方資料](https://github.com/openai/human-eval) 抽 8 題，
  要求完整 Python module，用原始功能測試計分，每題全數通過才得分。
  允許整段 Python Markdown fence；不從長篇說明中猜測程式碼。
- 數學：從 [GSM8K 官方 test split](https://github.com/openai/grade-school-math) 抽 8 題，
  允許解釋，最後須有唯一的 `Answer: number` 行；以十進位數值與官方答案比較。
- 繁中：8 個新編、可確定答案的閱讀／例外規則／狀態更新問題，完整 JSON 值與型別計分。
- 工具：8 個新編多步工具紀錄，包含失敗後重試、已提交操作、條件分支、名稱轉 ID 與選時間。
  評分下一個呼叫的名稱、schema 和參數；這是固定紀錄續接，不是實際執行整個 agent 流程。

公開資料先固定 commit、原始 hash 與 MIT 授權。
選题規則是以 `sha256("whallm-q1-2026-09-06:{dataset}:{zero_based_index}")` 排序取前 8 題，
不依模型回答換題。HumanEval indices：44、39、95、154、57、56、50、151；
GSM8K：678、1178、394、1181、1146、644、926、757。
這些資料可能出現在 checkpoint 的訓練資料中，不宣稱對模型未見過。

題組、參考解答、測試與来源在
[固定題組](../docs/benchmarks/prompts/2026-09-06-qwen-quality-q1.json)。
執行前必須確認 8 個程式參考解答都通過同一評分環境，並檢查錯誤實作會失敗。
評分環境不允許讀 home／專案檔案、寫檔、連網、fork、signal 或 Mach IPC。
限制 CPU 3 秒、wall time 10 秒、輸出 1 MiB；這不是 VM，也不宣稱已設置硬性記憶體上限。
程式執行和 GPU 生成位於不同 process。

### 停止條件與不確定性

每題記錄兩版皆對、兩版皆錯、原版對／候選錯、原版錯／候選對。
第一個原版對／候選錯即停止後續題目，不以其他題的提升抵銷。
先核對評分和標準答案，再用相反順序重跑同一對一次；重跑只作重現確認，不重複計入樣本數。
若重現，標記候選未通過本輪品質篩選；若未重現，標記不穩定，仍不放行。
若沒有回退，才繼續 Q1B；Q1A 結束不代表通過採用條件。

樣本數在生成前固定為每類 8 題。本階段用來尋找回退，沒有以它證明微小差異的統計能力。
完整執行且零回退時，附上零事件的條件式上界 `1-(0.05/4)^(1/8)`，約 42.2%。
這是假設同類題目獨立、代表目標分布時，四類合計 95% 的保守回退機率上界；
本地編題未滿足代表性假設，所以數字只展示小樣本的限制，不當成模型品質信賴區間。
若提前停止，不套用這個固定樣本計算。一般任務正確率的下界仍屬未定。

### 重現命令

```sh
PYTHONPATH=runtime:Scripts .venv/bin/python -m unittest \
  runtime.tests.test_qwen_quality_grading runtime.tests.test_qwen_task_accuracy
PYTHONPATH=runtime:Scripts .venv/bin/python Scripts/benchmark_qwen_task_accuracy.py \
  --suite docs/benchmarks/prompts/2026-09-06-qwen-quality-q1.json \
  --output scratch/qwen-quality-q1-2026-09-06/runs
```

## Q1B 尚待覆蓋的能力

Q1A 沒有測長篇繁中回答的完整性／無根據主張，也没有真實長文件與實際多回合工具操作。
若 Q1A 無回退，先固定長篇回答的逐條事實與指令評分表、盲化版本和歧義處理方法；
人評或獨立審查未完成前，這一項保持未定，不以模型自評取代。
工具須接入本地模擬器，逐步接收實際回傳並核對最終狀態、失敗重試和重複執行。
再擴大保留題組及樣本數，為正式評估固定配對正確率差的統計方法；不自行放寬零退步門檻。

## 當前狀態

已固定 Q1A 題組與規則，評分器檢查通過；尚未完成模型配對。
