# Throughput

Throughput 頁面提供單次請求的吞吐量測試。`v1.1.7` 正式版包含下列取消、卸載與記憶體修正。
本機 `dist` App 與 ZIP 為已簽章、公證的正式成品，排除 Dry run。
封裝與隔離檢查見 [發布驗證](VALIDATION.md#2026-09-14whallm-117-正式發布)。

## 操作

2026-09-14 原始碼另補齊 Prefill 中途取消：V4 在分批／層間停止，
V4.1 增加層間檢查，並共用 expert 讀取取消與安全收尾。
此項已納入 `v1.1.7`。`Unloading model…` 仍可能包含等待當前運算／讀取
收尾的時間；詳見 [共用取消流程](MODEL_PACKAGES.md#prefill-取消)。

- Model 列出可用的 installed model，選單沿用模型 Alias。Run Benchmark 會在需要時
  啟動 App server，並自動載入所選模型。模型設定沿用 server catalog。
- **只有 local build** 的 Model 選單提供 **Dry run** 除錯選項。選取後立即依目前輸入長度、生成長度與 Code／Novel
  產生固定的模擬結果，不需要 installed model、不啟動 server、不發出推論請求。
  調整選項後可按「產生結果」重新生成。模擬 slots 固定 2304；畫面標示模擬結果，
  三種文字輸出皆以 `dry-run` 識別模型，JSON 的 finish reason 是 `dry_run`、
  素材與輸出 hash 欄位也標示 `dry-run`，不代表真實測量或有效 hash。
  `make build`／`make run`／`make test` 與 `make package` 啟用 `WHALLM_LOCAL_BUILD` 編譯旗標。
  stable 與 dev 發布均由 `Scripts/release.sh` 強制 `WHALLM_BUILD_FLAVOR=distribution`，
  不編入模擬結果生成程式，也不顯示選項。直接呼叫 `package-app.sh` 時預設 distribution；
  local 封裝入口為 `make package`。封裝驗證從執行檔啟動標記檢查旗標；發布前與下載後
  均要求 local 功能關閉，防止使用到錯誤組態的 App。
- Benchmark Context 下拉選單提供 **Code／Novel**，預設 Code。
  Code 是 Whallm 多個 Swift／Python 檔案的固定副本；Novel 是《白鯨記》英文原文。
  兩者以 gzip 內建於 App，合計 **697665 bytes**（約 698 KB，不含來源與授權記錄）。
  執行時不讀取使用者專案，也不下載外部資料；每份素材只使用一次原文序列。
  詳細大小、token 數與來源比較見下方「素材評估」。
- Context lengths 可多選 **1K／4K／8K／16K／32K／64K／128K／200K**，
  K = 1024，200K = 204800。預設選 **4K／8K／16K**，依小到大各執行一次。
- Generation length 單選 **128／1024／4096**，預設 **128**。這是輸出上限；
  模型可提早結束，結果顯示實際數量。
- 不提供 Warm-up、ANE-aligned prompts 或 Batch sizes 控制項，也不額外執行
  benchmark warm-up。模型原有的載入設定仍生效。
- 執行中可取消，已完成的列保留；切換頁面保留本輪狀態。關閉視窗會取消。
  新一輪 Run 會清除上一輪結果；目前不跨 App 重啟保存。
- 目前原始碼會在整輪完成、取消或請求失敗後，自動卸載本次模型；各輸入長度之間
  保持載入。取消後會先等待生成停止，再卸載，清理完成前不能重新 Run。
  卸載失敗會顯示錯誤並保留結果；尚未發出模型請求或 Dry run 不執行卸載。
  此修改已隨 `v1.1.7` 發布；較早的 `v1.1.7-dev.2` 成品不含此行為。
- 結果底下的 Result output 可切換 **純文字／JSON／Markdown table**，文字可選取複製。
  已完成的每筆結果即時加入輸出；取消或失敗後仍可複製已完成部分。
  純文字使用等寬空格對齊欄位，Markdown 是可直接貼上的表格原文；兩者包括模型、素材、
  slots、輸出上限及畫面中的全部指標。JSON 保留未四捨五入的原始數值、hash 及完成原因。

## 版面

Throughput 使用較緊湊的設定列：內容寬度上限 760 pt，區塊左右內距 16 pt，
每列上下內距 12 pt。標題與說明間距 2 pt，標題 14 pt、說明 13 pt。
輸入長度選項靠左排列、間距 6 pt，視窗不足時換行。Model、Benchmark Context、
Generation length 與 Run／Cancel 共用同一條右邊界，操作按鈕位於設定區塊底部。
結果欄位展開填滿卡片可用寬度，保留一致的 16 pt 內距；窄視窗改用水平捲動。
結果標題在模型與 Code／Novel 後顯示 runtime 回傳的本次 slots。
若各筆 slots 不同，列出不同值；連接未提供此欄位的舊 server 時顯示破折號，不猜測設定值。

## 測試方式

素材經所選模型的 tokenizer 編碼一次，直接截取前 N 個 token ID，**不重複補長**。
這是原始 completion，不額外加聊天模板。不經 decode／重新編碼，以保持精確長度。
若素材對所選模型不足 N tokens，回報可用數量並要求縮短輸入，不自動填充。
輸入加生成上限超過 checkpoint context limit 時會回報錯誤。

`POST /api/benchmark/throughput` 使用既有 API key 驗證，接收：

```json
{"model":"qwen3.8-flash-next-fp8","benchmark_context":"code","context_length":1024,"generation_length":128}
```

每個請求執行一筆測試，以 SSE 傳回 loading／running、生成進度、result、`[DONE]`。
載入或生成失敗以串流 error 回報。App 逐筆呼叫，完成一筆便更新表格。
`benchmark_context` 只接受 `code`／`novel`，省略時預設 `code`；不合法值在載入模型前
回 HTTP 400。App 在 Run 時固定本輪素材，結果標題沿用實際結果，不隨後續選單改變。

每筆測試持有既有模型請求鎖，與聊天、載入／卸載和設定變更依序執行。
測試期間停用一般與 DSpark 的跨請求 prompt cache 重用及寫入，完成、失敗或
取消後恢復原設定；不刪除已有的對話快取。作業系統檔案快取仍保留；目前原始碼在
V4 batched／Qwen layer-major prefill 前釋放舊 expert Slots，生成時再填回，其他路徑仍保留。
不同順序、前次使用情況仍可能影響結果。取消透過關閉請求連線傳到 runtime，
正在執行的 GPU 工作仍須完成安全收尾。

**修正前的 Peak MLX 差異。** DeepSeek V4 的 slots 按需配置，第一筆可從空 slots 開始，
下一筆則同時持有已填滿的 slots 與 prefill 暫存。2026-09-14 本機交換順序確認：
4K 首筆為 21.20 GiB，放在 1K 後為 32.56 GiB；1K 首筆為 21.18 GiB，放在 4K 後為
31.20 GiB。因此不能把第一列與後續列的全部差額歸因於輸入長度。
本次設定、輸出一致性及限制見 [記憶體排查](benchmarks/2026-09-14-throughput-memory/README.md)。
`v1.1.7` 已移除上述兩條路徑的舊 Slots 重疊；較早的 Alpha 仍有此行為。

## 結果欄位

| 欄位 | 定義 |
| --- | --- |
| Input / Output | 本次輸入 token 數／實際輸出 token 數 |
| TTFT (ms) | runtime 記錄的第一個 token 等待時間 |
| TPOT (ms) | `1000 / TG tok/s`；沒有可計算的生成速度時顯示破折號 |
| PP tok/s | runtime 輸入處理速度，沿用 Metric 頁面的計算 |
| TG tok/s | runtime 生成速度，沿用 Metric 頁面的計算 |
| Total (s) | 從生成 iterator 開始到完成收尾的時間；不含載入與素材編碼 |
| Throughput | `(輸入 tokens + 實際輸出 tokens) / Total`，單位 tok/s |
| Peak MLX | 本筆測試的 MLX 配置記憶體峰值，GB 以 1024³ bytes 換算；不是 RSS 或整機記憶體 |

每筆測試在生成前重設 MLX peak 計數，因此 server status 的 MLX peak 也從這筆測試
重新累計。SSE result 同時包含 API model ID、生成上限、finish reason、prompt cache
重用數及輸出 token ID 序列的 SHA-256（每個 ID 以十進位加換行編碼）。
另包含 `benchmark_context` 及 `corpus_sha256`（完整解壓後 UTF-8 素材的 SHA-256）。
`slots` 取自持有模型請求鎖時的 `runtime.config.slots`，與本筆數據一起保存；
後續修改模型設定不會改變已完成結果或其文字輸出。

表內數據適合檢查本機設定。素材內容、沒有額外預熱和不同快取狀態可能影響速度，
不可直接當作一般工作負載或不同電腦的效能結論。正式結果仍須另外依 `docs/benchmarks/`
規定記錄 commit、環境、工作負載、設定、快取狀態與輸出 token hash。

## 素材評估（2026-09-13）

以下是離線 tokenizer 計數，**不是生成速度測試**。200K 指 204800 tokens。
大小使用十進位 KB／MB。候選程式庫按固定 commit 收集 `.py`／`.go`（包含測試，
排除 vendor），每個檔案只加入一次；Whallm 使用 23 個 Swift 與 22 個 Python 檔案，
排除測試及巢狀第三方原始碼目錄。作品以 Gutenberg START／END 標記之間的原文計數。

| 素材 | 原始大小 | gzip 大小 | Qwen tokens | DeepSeek V4.1 tokens | 覆蓋 200K |
| --- | ---: | ---: | ---: | ---: | --- |
| 舊內建片段 | 9.2 KB | 2.7 KB | 2254 | 2302 | 否 |
| **Whallm 原始碼（Code）** | 1.04 MB | 198 KB | 252447 | 249700 | 是 |
| [mlx-lm](https://github.com/yanun0323/mlx-lm) | 2.13 MB | 341 KB | 558650 | 569349 | 是 |
| [clean_architecture](https://github.com/yanun0323/clean_architecture) | 27.4 KB | 6.2 KB | 8477 | 8390 | 否 |
| [sutando](https://github.com/yanun0323/sutando) | 83.7 KB | 16.1 KB | 27494 | 26767 | 否 |
| [gollection](https://github.com/yanun0323/gollection) | 79.5 KB | 13.4 KB | 29643 | 28303 | 否 |
| [Hamlet](https://www.gutenberg.org/ebooks/1524) | 180 KB | 71.6 KB | 52215 | 49265 | 否 |
| **[Moby-Dick（Novel）](https://www.gutenberg.org/ebooks/2701)** | 1.23 MB | 500 KB | 311730 | 309191 | 是 |
| [莎士比亞全集](https://www.gutenberg.org/ebooks/100) | 5.42 MB | 2.10 MB | 1567778 | 1469759 | 是 |

選用 Whallm 與 Moby-Dick：兩者各自足以覆蓋 200K，無須加入其他程式庫。
另使用 DeepSeek V4 的固定 tokenizer revision 驗證，Code 為 **249700**、Novel 為
**309191** tokens，亦足夠。來源本身的自然重複（例如程式碼樣板）保留，測試器不再
人工複製片段。不同長度使用同一原文的前綴，便於在固定素材下比較。

資源位於 `runtime/deepseek_v4_ssd/benchmark_contexts/`。`manifest.json` 記錄固定副本的
來源、日期、完整文字 SHA-256、原始／壓縮大小，Code 另外保存各原始檔案 hash。
此副本包含當時尚未 commit 的功能程式碼，不能只靠 base commit 重建；封裝時不會
自動更新素材。Code 保留專案 MIT 授權；Novel 的來源說明與完整 Gutenberg 授權文字
另存 `NOVEL-NOTICE.txt`，不加入模型輸入。

候選 commit、tokenizer 檔案 hash、套件版本、兩份素材在三種 tokenizer 下的完整計數及
200K 輸入 token hash，見 [機器可讀紀錄](benchmarks/2026-09-13-throughput-contexts/coverage.json)。
這些資料只確認素材容量與輸入組成，未執行 200K 模型推論。
