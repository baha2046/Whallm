# Issue #6：其他框架的重複輸出處理方式

日期：2026-09-06。狀態：**保留外部原始碼評估與離線重播的階段紀錄**。
後續依下列根因優先順序，已定位 MLX 0.32.0 的背景執行緒抽樣缺陷，
改採上游 MLX 0.32.1 修復；目前證據與驗收見
[因果調查](ISSUE_6_CAUSAL_INVESTIGATION_2026-09-06.md)。以下描述評估當時狀態。
使用者要求先評估其他框架，因此已停止 candidate 08 的長文生成；該次輸出不完整，
不能判為通過或失敗。本次沒有修改 runtime、安裝其他框架或部署 App。
當時未完成的修復候選，後續已另存研究資料，未併入根因修復。

## 結論與範圍

值得借用的是三種不同機制：正確實作取樣參數、針對重複片段降低機率、偵測失控後
提早停止。另以獨立 thinking budget 處理「思考內容正常但耗時過長」。
這些機制不能證明 Whallm 的模型計算正確，也不能證明相同題目換框架就會正常。
本次未在 vLLM、SGLang 或 llama.cpp 執行 Qwen 全模型推論。

Issue 的正文循環、reasoning 循環、正常但冗長的 reasoning 應分開驗收。
原始 issue 與後續更正見 [Issue #6](https://github.com/yanun0323/Whallm/issues/6)。

## 原始碼比較

| 框架 | 已核對行為 | 對 Whallm 的意義 |
| --- | --- | --- |
| vLLM | Presence/frequency 計算已生成的完整歷史；repetition penalty 包含 prompt 與生成內容。另有預設關閉的 `repetition_detection`，可設定片段長度與連續次數。Scheduler 命中後設為 `FINISHED_REPETITION`，`stop_reason="repetition_detected"`。 | 參數語意可作相容基準；偵測適合避免把額度耗在循環，不能當成正常完成。 |
| SGLang | 支援 presence、frequency、repetition、min-p；目前 repetition penalizer 累積 output tokens。Presence/frequency 是完整生成歷史。參數可由 request 覆寫，部分取樣預設來自模型的 `generation_config.json`。 | 相同參數名稱不保證與 vLLM 完全同義，尤其 repetition 是否涵蓋 prompt。要明文決定合約。 |
| llama.cpp | 一般重複懲罰有可調視窗；DRY 對「會延長既有重複片段」的候選 token 施加隨片段長度增加的懲罰。DRY 預設關閉；本次 pinned 版本的視窗預設 64、allowed length 2、base 1.75。換行等 sequence breakers 會限制片段比對。 | DRY 值得作為正文片段循環的候選，但必須驗證日文、跨段重複、程式碼和刻意重複。不能直接搬預設值並宣稱有效。 |
| MLX-LM | 目前上游與本機使用的 `make_logits_processors` 都提供三類 penalty，預設視窗各為 20 tokens。底層有能力處理更長歷史；Whallm 原本未指定視窗，且 API 部分參數被拒絕或忽略。 | API 接線是已確認缺陷；20-token 視窗與其他框架語意不同，但 thinking 的 presence 原本為 0，所以只加長視窗不會修好 stock thinking。 |

來源：[vLLM 參數](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)、
[vLLM scheduler](https://github.com/vllm-project/vllm/blob/144e79c8106da23141ac010394b782f730cc7fe8/vllm/v1/core/sched/utils.py)、
[SGLang 文件](https://docs.sglang.io/docs/basic_usage/sampling_params)、
[SGLang repetition 實作](https://github.com/sgl-project/sglang/blob/e3f709759138f037d67ee89b48d95b5c81440443/python/sglang/srt/sampling/penaltylib/repetition_penalty.py)、
[llama.cpp server 文件](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/tools/server/README.md)、
[DRY 實作](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/src/llama-sampler.cpp)、
[MLX-LM 取樣程式](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)。

vLLM 的 `thinking_token_budget` 會在超過思考額度時強制結束思考標記；llama.cpp
提供 `--reasoning-budget`，並可設定結束前插入的訊息。這適合限制短改寫任務的
思考時間，無法處理已進入正文的循環，也不保證縮短思考後的答案品質。
[vLLM 實作](https://github.com/vllm-project/vllm/blob/144e79c8106da23141ac010394b782f730cc7fe8/vllm/v1/sample/thinking_budget_state.py)。

Qwen 官方仍推薦 thinking 的 presence 0、chat 1.5；也說明提高 presence 可以減少
無止境重複，但可能導致語言混雜與品質下降。因此沒有外部依據支持「懲罰越強越好」。
[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices)。

## 本機離線重播

只執行已檢查的 pinned vLLM `check_sequence_repetition` 與 `_has_repeating_pattern`
兩個函式，不啟動 vLLM 引擎。探索設定：片段長度 1–64 tokens、連續 8 次；
這不是上游預設，也不是已決定的 Whallm 設定。

| 保存的輸出 | 首次偵測位置（generated token，1 起算） | 解讀 |
| --- | ---: | --- |
| candidate 04 | 554 | 命中 `で` 循環。 |
| candidate 06 | 1,404 | 命中 `の絵` 循環。 |
| candidate 07 | 1,089 | 命中祖母說話句子的循環。 |
| candidate 08 已產生的 788 tokens | 未命中 | 被使用者要求暫停；不能推論後續品質。 |
| 先前 chat 正文重新編碼的 3,827 tokens | 未命中 | 該正文結尾本來仍有三次重複段落；此偵測設定不涵蓋所有重複。 |
| 合成的刻意重複 12 行 | 56 | 必須有 opt-out／應用政策；不能一律視為模型故障。 |

前三列使用記錄的原始 token IDs，包含 reasoning；chat 列是保存文字重新編碼，
不是原始 decode IDs。這只證明偵測能力，不是防止循環、誤判率或效能量測。
詳細資料與來源 hash 見
[detector-replay.json](../docs/benchmarks/2026-09-06-issue6-framework-assessment/detector-replay.json)；
固定外部 commit 與檔案 SHA-256 見
[sources.json](../docs/benchmarks/2026-09-06-issue6-framework-assessment/sources.json)。

## 建議的修復順序

使用者追問後重新評估：先前把 API、循環中止、DRY 放在根因比較之前，適合降低
失敗成本，但不足以作為「讓長文正常完成」的優先順序。以下取代原建議；
後續執行結果另見上方因果調查。

1. **固定基準，分離模式與取樣因素。** 從隔離的 v1.1.4 基準開始，保留全部失敗
   候選。明確設定所有取樣參數，做 thinking/chat × thinking/chat 取樣設定的四組
   對照，記錄實際 prompt token IDs、思考與正文長度、重複起點。使用同框架研究用
   固定亂數控制並重複樣本；相同 seed 不代表跨框架等價。四組對照只能縮小嫌疑，
   不能單憑模式差異認定根因，因思考內容與歷史長度也不同。
2. **用固定文字前綴檢查計算路徑。** 在已保存失敗輸出的幾個關鍵位置，固定完全
   相同的輸入 token IDs，比較逐 token 累積狀態與從頭 prefill 的下一步預測；
   再對可疑元件使用獨立參考運算。記錄數值誤差與候選 token 機率，不把任何微小
   浮點差异或單一 argmax 分歧直接判為 bug。這能避免兩次自由生成分岔後，無法
   分清差異來自運算還是隨機取樣。
3. **依證據修最小範圍，再驗收。** 若發現模板、狀態更新或運算錯誤，先修該處；
   若參考路徑也重複，再分離量化與模型／取樣行為。Whallm API ID 雖含 FP8，
   installed routed experts 是 MXFP4；直接與官方 FP8 服務比較會混入量化差異。
   先前 FP32 router/GDN/norm 候選未通過長文，不能從元件單元測試推導出根因修復。
   驗收需多次自然完成且內容連貫，另檢查短改寫、程式碼、JSON、工具呼叫及刻意重複。
4. **獨立處理已確認的 API 缺陷與失敗成本。** API 參數失效可以獨立修復；需明確
   定義 penalty 是否包含 prompt、reasoning、正文及 cache 命中後的歷史。目前
   候選的 repetition 排除 prompt，不能宣稱完全符合 vLLM。循環偵測、明確中止原因、
   client 取消後能否釋放 generation，以及 thinking budget 是獨立工作；不得將中止
   當成長文品質通過，也不應讓這些工作延後根因調查。
5. **最後才決定是否採用 DRY。** 僅在運算路徑核對後，將 DRY 作為可選品質策略
   單獨驗證。不要再以硬性禁止重複片段替代品質驗證，先前 candidate 02 已出現
   無意義文字與過早結束。尚無本機證據證明 DRY 能修復此案例。

修復實驗輸出保存在
[`2026-09-06-issue6-fix-m2-max`](../docs/benchmarks/2026-09-06-issue6-fix-m2-max/)。
本次評估不批准任何候選成為發布版本。
