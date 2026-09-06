# Issue #6：固定前綴與模式／取樣對照

狀態：已定位並修復背景執行緒抽樣缺陷；MLX 0.32.1 通過回歸測試及本輪三篇長文的循環／自然終止檢查。文字品質限制另列。依使用者同意的
[調整後順序](ISSUE_6_FRAMEWORK_ASSESSMENT_2026-09-06.md) 執行。

## 基準與證據

使用 installed Whallm v1.1.4 runtime 的獨立副本、原 installed model、bundled Python
與套件，M2 Max 64 GB。MTP、DSpark 關閉；8,192 expert slots、prefill chunk 128。
本輪 direct-runtime 診斷不使用 persistent prompt cache；ANE 設定保留預設，但
128-token chunk 不滿足其 1,024-token projection shape。
權重與已安裝 App 未修改，未發布、未提交 Git commit。

進入調查時 base commit 為 `bab462f495ec9b070d7546452349623d09c01289`。
既有修改已保存成 patch。先前未通過長文的 router/GDN/norm 數值候選已從 production
diff 移出，保存於獨立研究目錄；API／全歷史 penalty 候選亦已移入 `unadopted-api/` 保存，沒有併入根因修復。

環境、副本來源 hash、原修改與實驗輸出位於
[`2026-09-06-issue6-causal-m2-max`](../docs/benchmarks/2026-09-06-issue6-causal-m2-max/)。

## 固定文字前綴對照：完成

腳本：[issue6_prefix_parity.py](issue6_prefix_parity.py)。
固定原長文 prompt 的 188 tokens，接續先前 candidate 06 保存的原始 token IDs。
這些 ID 是固定測試輸入，並非宣稱由 release baseline 生成。沒有取樣。
先從 prompt 開始逐 token 讀入，保存四個檢查點；再用空白狀態、layer-major prefill
重讀完全相同的前綴，比較下一步 logits 與 cache arrays。

| 已接續 token 數 | logits RMS 差異 | 預測分布 L1 距離 | 兩路徑的最高機率 token |
| ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 相同：`The` |
| 128 | 0.1663 | 0.00000093 | 相同：`4` |
| 512 | 0.0740 | 0.01758 | 相同：`つき` |
| 1,280 | 0.0683 | 0.06563 | 相同：`所` |

L1 距離是整個詞彙表的機率絕對差總和，範圍 0–2，不是錯誤率。
初始完全相同的對照得到零差異。後續有數值差異，不能宣稱完整一致；四個位置的
top-1 相同也不能證明整個 runtime 正確。但目前未見「重讀同一內容使預測大幅
恢復」的證據，不足以把持續重複歸因於累積狀態損壞。
這仍是同一 runtime 的兩條路徑，不是獨立全模型 reference，亦未涵蓋所有生成位置。

完整逐層 state 差異與 logits 見
[results.json](../docs/benchmarks/2026-09-06-issue6-causal-m2-max/prefix-baseline-01/results.json)。
本次同步保存全部 state arrays，不把時間當成正式效能結果。

## 模式與取樣設定交叉對照：暫停，改做執行緒因果對照

腳本：[issue6_mode_matrix.py](issue6_mode_matrix.py)。
四組為 thinking/chat 模式，各套用 thinking/chat 的完整取樣設定。
兩個研究用固定 seed：20260906、20260907；每組全新 runtime。
保留 release 的原始 20-token penalty window。直接傳入 GenerationOptions，
隔離 release API 不支援部分參數的干擾。相同 seed 不代表跨框架等價。

每組先設 2,048-token 診斷上限，不是原 8,000-token 長文驗收。
若看到至少 128 個 token、至少八次的完整連續重複，研究腳本會保存並停止該組；
這是實驗節省資源的措施，不是產品功能，也不代表正常完成。

首個 seed 的 thinking 模式＋thinking 設定、chat 模式＋thinking 設定，均到達
診斷上限而未觸發持續重複偵測；不能由此推論完整長文會完成或問題已消失。
第三組 thinking 模式＋chat 設定亦到上限，未觸發偵測。
隨後相同 seed、188 prompt token IDs、完整 GenerationOptions 的 HTTP baseline
在第 7 個 token（零起算 index 6）即與 direct runtime 不同。進一步發現下節的
執行緒抽樣缺陷，因此停止其餘 matrix：主執行緒結果無法代表有缺陷的 HTTP 路徑。

原始 matrix observer 將 `finish_reason=length` 的末個 token 留在 token JSONL，
但未納入 result.json 的 `generated_tokens`／token hash，故上限案例顯示 2,047。
後續彙整必須納入此末個有效 token；這是觀測欄位的偏差，不影響模型的生成內容。

## 根因：編譯抽樣器捕獲錯誤執行緒的亂數狀態

Primary sources：
- [MLX-LM #1675](https://github.com/ml-explore/mlx-lm/issues/1675)：相同 pinned 組合
  MLX 0.32.0／MLX-LM 0.31.3 的無模型重現。
- [MLX #3828](https://github.com/ml-explore/mlx/pull/3828)：compiled random state 修復。
- [MLX-LM #1753](https://github.com/ml-explore/mlx-lm/pull/1753)：上游採用 MLX 0.32.1。
- [MLX-LM #1593](https://github.com/ml-explore/mlx-lm/pull/1593) 曾提議移除 import-level
  compile，最後關閉並改採上游 MLX 修復；不是本專案需要另行維護的 sampler fork。

本機 `sampler-thread-probe.json` 在同一組固定 logits／seed 下抽樣 32 次：
MLX 0.32.0 主執行緒抽到 5 種 token、32 種 RNG state；背景執行緒只抽到 1 種
 token、1 種 RNG state。重設 seed 亦無法恢復正常前進。
在原 bundled 0.32.0 **只移除 categorical_sampling 的 compile 包裝**，保留其餘
filters，主／背景執行緒的全部 32 個結果完全相同且正常變化，見
`sampler-only-control.json`。這個小測試隔離了抽樣器，不涉及模型權重或數值算式。
再以原 bundled Python／MLX 0.32.0／frozen v1.1.4 runtime 做真實 HTTP 控制：
只取消 categorical compile，同 seed／同 188 prompt tokens，前 64 個生成 tokens
全部與原主執行緒 baseline 相同；原 HTTP 在 index 6 即不同。見
`sampler-only-causality.json`。64-token cap 僅用於路徑因果驗證，不算完整回覆。

MLX 0.32.1 的主／背景執行緒均有 32 種 RNG state，結果完全相同；同 seed 可重現，
不同 seed 得到不同序列，見 `sampler-thread-probe-0.32.1.json`。
**相同 seed 不保證跨 MLX 版本的 token 相同**；本次兩版本的主執行緒結果也不同。
此缺陷會讓隨機選擇固定於同一個亂數樣本；模型 logits 仍會改變，所以不表示每次
輸出永遠是同一個 token，也不表示每個 prompt 都會失敗。

新增 `runtime/tests/test_sampling.py`：背景執行緒建構／跨執行緒呼叫、seed、
top-p／top-k／min-p、greedy 和真實 `generate_step` 迴圈。舊環境產生四個 failure
（含 subtests），新版三個測試全通過。完整 Python suite 在隔離 overlay 與更新後
本機 .venv 各通過 308 項。這些不是全模型文字品質或效能測試。

## 採用範圍與長文驗證

目前最小產品修正為 `requirements.txt` 的 MLX 0.32.0 → 0.32.1，及打包前確認
`mlx`／`mlx-metal` 與 pinned 版本相符。舊 .venv 的 package command 已在任何
build／output mutation 前拒絕；新 .venv 的相同版本檢查通過。未執行完整打包。
其餘 API、預設 penalty／window、模型算式、MTP 和 DSpark 設定都保持原樣。

長文 driver：`research/issue6_reproduce.py`。MLX 0.32.1 的首次 overlay 嘗試被
installed Python 的 Team ID library validation 擋在 import 階段，四個 case 均未
生成，完整錯誤保留於 `mlx0321-seed-20260906/`。重試使用相同 3.14.7 開發用 Python
啟動器、bundled Python home／其他套件、isolated MLX overlay；未更改 App 簽章。
故此為開發環境驗證，不能宣稱 signed App 驗證。Frozen v1.1.4 runtime、原模型、
原設定，seed 20260906，8,000-token 長文上限、4,096-token 短句上限，900 秒期限。
實際 source／package versions／request／cache／token IDs／正文均保存在每個 case。

第一個新版 frozen-release thinking 案例自然 `stop`：2,949 completion tokens、
3,947 正文字元、1,336 reasoning 字元，四個場景與章末結束都有寫出；自動相鄰
重複／三次重複段落掃描未發現循環。全文檢查仍見用字瑕疵與部分對話邏輯不自然，
也未達題目約 5,000 字的要求，不能宣稱完整文字品質／長度遵循通過。
目前工作樹 seed 20260907 的 thinking／chat 長文均自然 `stop`，正文各為 5,556／
5,544 字元。短句兩模式亦自然 `stop`，正文 40／39 字元；thinking reasoning 仍有
4,629 字元，屬思考長度問題，這次未調整 thinking budget。三篇長文皆無研究中止、
循環中止、token cap 或 wall-time limit，掃描亦無持續相鄰重複或三次重複段落。
第二篇 thinking 仍有時間線、用詞與簡體中文字形瑕疵；詳見 `quality-review.json`。
本輪證據支持修復已重現的抽樣缺陷與長文循環，但不能保證所有題目、字數遵循、
日文文法、事實或小說品質。
程式碼／JSON／tool 三個案例亦正常完成。程式碼移除 Markdown fence 後，函式與
三組輸入檢查通過，但「只有原始 code」格式要求未完全遵守；JSON 完全符合指定
物件；stochastic tool call 正確產生 `get_weather` 與 `{"city":"Taipei"}`，沒有
執行外部工具。總計 8 個 HTTP 案例及其完整 raw token IDs hash，見
[`summary.json`](../docs/benchmarks/2026-09-06-issue6-causal-m2-max/summary.json)。
不能把 `length`、研究停止、逾時或只是不再重複，當成完整品質通過。
尚未進行完整 FP8 reference、M5 Pro、DeepSeek 全模型、正式速度／能源或打包驗證。
