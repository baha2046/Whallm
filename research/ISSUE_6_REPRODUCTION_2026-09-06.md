# Issue #6：v1.1.4 本機重現

2026-09-06 在 Mac Studio M2 Max／64 GB、macOS 26.6.2 上，使用已安裝的
Whallm 1.1.4 和 Qwen FP8，重現了思考模式的答案正文連續重複。
這是單次抽樣的品質探索，不是正式效能量測，也不是根因或修復驗證。

原始回報：[Whallm #6](https://github.com/yanun0323/Whallm/issues/6)。
Issue 本文與留言的讀取快照、完整 requests、逐事件輸出、reasoning、正文、
來源雜湊、環境及結果，保存在
[`2026-09-06-issue6-reproduction-m2-max`](../docs/benchmarks/2026-09-06-issue6-reproduction-m2-max/summary.json)。

## 結果

| 自建日文題目 | 模式 | 思考字元 | 正文字元 | 結果 |
| --- | --- | ---: | ---: | --- |
| 四場景、約 5,000 字小說 | thinking | 1,932 | 2,136 | 正文第 641 個字元開始，`三番目に初めて、` 連續重複 187 次；人工停止。 |
| 相同小說題目 | chat | 0 | 6,052 | 3,829 output tokens，`finish_reason=stop`；結尾有重複段落。 |
| 約 43 字短句改寫 | thinking | 3,569 | 43 | 1,310 output tokens，`stop`；反覆推敲相同選項後完成。 |
| 相同短句改寫 | chat | 0 | 39 | 24 output tokens，`stop`。 |

字元計數包含換行，使用 Python Unicode 字串長度。長文 thinking 的重複區間是
`[640, 2136)`（zero-based），長 1,496 字元，基本單位長 8 字元。
人工停止前的 server snapshot 記錄 1,793 個已產生 token，包含 reasoning。
該輪沒有自然的 finish event；`incomplete-stream` 是測試者停止服務造成的，
不能當成另外一個 server 錯誤或客戶端斷線取消失敗。
沒有等待該輪耗盡 8,000 tokens，因此未重現原回報的 `finish_reason=length` 終態。

Chat 長文沒有同樣的短語連續失控，但五條至少 30 字元的完整句子各出現三次，
結尾一大段也重複後才停止。不能宣稱「chat 完全不重複」或答案品質通過。

短句 thinking 約 160.14 秒，chat 約 25.99 秒，這兩個 wall times 包含
啟動隔離服務及首次載入。Server 自己記錄的 request time 分別是約 141.03 秒與
6.92 秒。這些是單輪探索性觀察，沒有隨機化執行順序或控制 OS page cache，
不能當作正式速度比較。兩種改寫都改動了原文時態，沒有做語意保留的品質驗收。

短句 reasoning 有反覆列舉相同選項，但最後正常結束，沒有觀察到原留言描述的
持續短 token 迴圈。`presence_penalty=1.5` 的修正效果、斷線後是否繼續運算及
根因都尚未驗證。

## 執行邊界

- 執行 `/Applications/Whallm.app` 內的 Python 和 runtime；沒有執行目前開發工作樹。
- 已核對 bundled runtime 的 15 個 Python source files 與 tag `v1.1.4` 完全相同。
  Tag commit：`b25ae07e9df143dd744405b741628daa5e8705c1`。
- 工作樹當時 HEAD：`bab462f495ec9b070d7546452349623d09c01289`。
- Python 3.14.7、MLX 0.32.0、mlx-lm 0.31.3、numpy 2.3.5、transformers 5.12.1。
- Installed model：`/Users/yanun/.dsmodel/qwen3.8-flash-next.dsv4`。
  保存 manifest SHA-256，沒有重新讀取所有模型檔案做完整 checksum audit。
- `slots=8192`、read workers 4、prompt cache entries 2、persistent entries 8；
  MTP／DSpark 關閉，ANE Prefill 使用 packaged 預設值。
- 每個 case 都是新 server、新的私有 persistent prompt-cache 目錄與第一個生成請求。
  四輪 status evidence 都記錄 `prompt_cache_reused_tokens=0`。
- 快取放在磁碟，不是回報者的 RAM disk；OS page cache 沒有清除。
- 每輪都保留預設的模式取樣設定，只透過 `reasoning_effort=medium/none` 切換模式。
  API 不支援 seed；沒有私下注入固定亂數種子或修改防重複設定。
- 長文 `max_tokens=8000`、每輪 wall limit 900 秒；短句 `max_tokens=4096`、
  wall limit 240 秒。長文 thinking 確認持續重複後，在約 215 秒手動停止。
- 所有測試服務已停止。沒有修改 App bundle、模型或 production runtime，也沒有
  發送 GitHub 留言。

原作者的完整 prompt 沒有附在 issue；本次使用自行撰寫的相近題目，因此是
同類症狀重現，而非原題逐字重跑。每題每模式只測一次，不能估計發生機率，
也不能區分模型本身、提示格式、取樣實作或數值路徑等根因。

## 重現方式

在專案根目錄執行，output 必須使用新的目錄：

```sh
.venv/bin/python research/issue6_reproduce.py --output /tmp/issue6-new-long --cases long-thinking long-chat --timeout 900
.venv/bin/python research/issue6_reproduce.py --output /tmp/issue6-new-short --cases short-thinking short-chat --timeout 240
.venv/bin/python research/issue6_analyze.py /tmp/issue6-new-long
```

Script 使用本機上述 App 與 installed model 路徑。它會順序執行 cases、保存原始
SSE events 和拆開的 reasoning／正文，並在成功、錯誤或到時後停止自己啟動的 server。
它不會自動因為偵測到重複而提前停止；本次長文人工停止另存 `manual-stop.json`。
`issue6_analyze.py` 檢查至少四次相鄰重複、合計至少 120 字元的 1–512 字元單位，
並列出出現至少三次的長句。沒有被規則抓到不代表沒有語意重複，仍需讀取正文。
本次分析另做了普通句子及已知重複字串的基本檢查。

Artifact 內的 prompt-cache 是本次測試產生、可拋棄的快取，已由該目錄的
`.gitignore` 排除；原始文字、JSON、logs 和 hashes 保留作為證據。

下一步若要修復，應使用這份已失敗的 prompt 作為回歸案例，再對照相同 checkpoint
的參考實作、檢查取樣／reasoning 切換及數值路徑，最後驗證候選修正。單純改參數
後某一輪沒有重複，不足以宣告修復。
