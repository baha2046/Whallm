# DeepSeek V4 Throughput 記憶體差異排查

以下為修正前的 2026-09-14 本機實測。後續已採用 prefill 前釋放舊 Slots，
目前 V4／Qwen 原始碼已修正，尚未打包。歷史結論：主要差異來自測試順序與前一筆留下的 expert slots，
不是 1K 增至 4K 就需要多約 11 GiB 的注意力快取。

## 實測

Apple M5 Pro、64 GiB、macOS 26.6.2、MLX 0.32.2、mlx-lm 0.31.3。
模型 `deepseek-v4-flash-0731`、1024 slots、Code、128 output、attention 分批 128、
MoE 分批 Auto、layer-major 門檻 1024、FP8 KV、LRU；Prompt Cache、DSpark 關閉。
每筆 seed 41、temperature 0.6、top-p 0.95。這些主要設定與檢查時 App 儲存值一致。

| 獨立程序 | 順序 | 輸入 | Peak MLX（GiB） | 請求結束後 active（GiB） |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 1024 | 21.1753 | 21.1501 |
| A | 2 | 4096 | 32.5608 | 21.1501 |
| A | 3 | 8192 | 32.9677 | 21.1501 |
| B | 1 | 4096 | 21.2026 | 21.1501 |
| B | 2 | 1024 | 31.1977 | 21.1501 |

五筆都實際生成 128 tokens，沒有重用 Prompt Cache。同長度的 A／B 輸出 token
SHA-256 完全相同。程序 B 的第一筆開始時 resident experts = 0；第二筆 = 1024。
只改順序，就能讓 4K 從 32.56 降至 21.20，讓 1K 從 21.18 升至 31.20。

## 原因

1. Slot 是按需分配的。剛載入模型時，1024 slots 尚未全部占用記憶體。
   第一筆生成逐步填滿 slots，之後保持到卸載；一個 expert blob 是 13,369,344 bytes，
   1024 個恰為 **12.75 GiB**。這次 active 從載入後約 8.40 變成約 21.15 GiB。
2. Layer-major prefill 另外載入整層 256 experts，並預讀下一層。一層 **3.1875 GiB**，
   兩層共 **6.375 GiB**；它們不屬於 1024 slots 的容量。
   矩陣運算還需要權重整理和中間輸出。MLX 的 sorted `gather_qmm` 路徑會整理為連續陣列，
   詳見 [MLX 0.32.2 原始碼](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp#L1371-L1399)。
   此次沒有逐筆追蹤底層配置，不能把所有暫存精確分攤到單一 kernel。
3. 第一筆的 prefill 發生在 slots 尚空時；後面的 prefill 則同時持有已填滿的 slots。
   程序 B 第一筆 4K 的 prefill 峰值約 **19.8080 GiB**；加上已填滿 slots 的
   **12.75 GiB** 是 **32.5580 GiB**，與程序 A 第二筆 4K 的 **32.5608 GiB** 相符。
   第一筆整體峰值較 prefill 高，是後續生成填滿 slots 的結果。
4. `mx.reset_peak_memory()` 只重新計數，Throughput 沒有清空 expert slots。
   因此同一張表的第一列與後續列可能使用不同起始記憶體狀態。

1K／4K／8K 都已走 layer-major，attention 分批同為 128；並非 4K 才啟用該路徑。
輸入長度仍影響注意力狀態和中間輸出，但不能解釋主要跳升。
MoE Auto 分批上限是 4096；這三筆實際 prefill 1023／4095／8191 tokens，
MoE 每批最多分別為 1023／4095／4096，因此從 4K 到 8K 不會讓單批 MoE 暫存翻倍。

這些實測每次結束都回到約 21.15 GiB，符合暫存釋放、expert slots 保留的行為。
這是本次測試的結論，不能據此宣稱所有路徑均無記憶體洩漏。

## 結果解讀與修改範圍

- App 欄位標示 GB，實際除以 `1024^3`，這份紀錄明確寫作 GiB。
- 21.18 不代表 1024 slots 在所有請求中的最高需求；本次連續使用的峰值可達 32–33 GiB。
- 整輪完成／取消自動卸載的 App 修改，能釋放整輪結束後的模型；它不會消除同一輪
  各 context 長度之間的快取差異。公平比較長度時，需要統一起始 expert-cache 狀態。
- 此份紀錄測量時只排查原因；後續已另行採用 prefill 前釋放舊 Slots，詳見目前模型文件。
  自動卸載另有 Swift／HTTP 測試，尚未打包或發布。

## 重現與證據限制

以下是當時的重現命令，需搭配 artifact 中 source hash 對應的修改前 runtime
（base commit `6821af3`）。目前原始碼已修正，直接重跑不會再得到相同跳升。
兩個命令應依序執行，避免兩個模型同時占用記憶體：

```sh
PYTHONPATH=runtime .venv/bin/python docs/benchmarks/2026-09-14-throughput-memory/probe.py --output /tmp/throughput-memory-baseline.json
PYTHONPATH=runtime .venv/bin/python docs/benchmarks/2026-09-14-throughput-memory/probe.py --lengths 4096 1024 --profile --output /tmp/throughput-memory-reversed.json
```

腳本使用此機 `/Users/yanun/.dsmodel/deepseek-v4-flash-0731.dsv4`；其他機器需修改路徑。
每個命令新開一個 runtime，每個命令內依序保留 expert slots。沒有清除 OS page cache。
第二組用 Python line trace 讀取 MLX 記憶體計數，不增加 MLX 陣列或額外 GPU 同步。
這是單次成對的記憶體診斷，有觀測成本與未控制的桌面工作，不是正式速度比較。

[baseline.json](baseline.json) 與 [reversed.json](reversed.json) 保存 commit、原始碼 hash、
模型 manifest hash、環境、完整設定、素材 hash、輸出 hash、runtime 指標與 B 的各層記憶體觀測。
base commit `6821af3`，工作目錄另有尚未提交的 App 卸載／Qwen 預設／V4.1 Prompt Cache 修改；
本次使用的 V4 `model.py`、`generation.py`、`expert_cache.py`、`throughput.py`、
`model_support/deepseek_v4.py` 已逐檔確認與 `/Applications/Whallm.app` 相同。
原始命令輸出位於 `scratch/throughput-memory-2026-09-14/`。
