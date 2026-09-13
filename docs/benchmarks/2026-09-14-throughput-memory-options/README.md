# DeepSeek V4 低峰值方案比較

2026-09-14 採用前的單輪篩選。測量當時未修改正式 runtime 或 App 設定。
使用者後續已採用方案 1，目前 V4／Qwen 原始碼已加入清理；本頁保留當時原型結果。
這是完整模型的記憶體與行為診斷，不是正式速度基準或通用品質驗證。

## 條件

M5 Pro 64 GiB、macOS 26.6.2、MLX 0.32.2、mlx-lm 0.31.3。
`deepseek-v4-flash-0731`，1024 slots、LRU、FP8 KV、attention 分批 128，
layer-major 門檻 1024，Prompt Cache／DSpark 關閉，temperature 0.6、top-p 0.95。
每個方案使用獨立程序；先跑 Code 1024 input／128 output 填滿 Slots，再測
Code 4096 input／128 output，每筆 seed 41。

三組預備請求的輸出 hash、正式測量前的 1024 個 resident expert keys hash 均相同。
候選 1 在正式請求的 layer-major prefill 入口同步 GPU、確認沒有背景 expert 作業，
再真正釋放 slot buffers 與重設索引。常駐模型、tokenizer 與檔案描述符保持原樣。
候選 2 沿用現有 individual expert 計算，`batched_expert_prefill=False`、
`moe_prefill_step_size=256`；是「停用整層預載＋小批計算」的具體組合。

## 結果

| 方案 | Peak MLX（GiB） | 首字等待（s） | 整筆時間（s） | 生成（tok/s） | 邏輯 expert 讀取（GiB） | 與原本輸出 token 相同 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 原本 | 32.5608 | 22.0324 | 47.9723 | 4.8965 | 349.2678 | 基準 |
| 1：prefill 前釋放 Slots | 21.2026 | 21.9777 | 49.6414 | 4.5914 | 350.9363 | 是 |
| 2：individual experts／MoE 256 | 21.9007 | 43.6182 | 70.3739 | 4.7472 | 338.9209 | **否** |

每筆正式請求都生成 128 tokens、沒有 Prompt Cache 重用；結束 active 均約 21.15 GiB。
方案 1 實際釋放恰好 **12.75 GiB**，清理約 **0.0765 s**；相較基準峰值下降
**11.3582 GiB（34.88%）**。此次整筆慢約 3.48%，首字等待幾乎相同；
多 134 次 expert cache miss，增加 1.6685 GiB 邏輯讀取。
這證明清掉 12.75 GiB Slots 不等於本筆固定增加 12.75 GiB 讀取。

方案 2 首字等待約為原本 1.98 倍。其完整輸出 token hash 與原本不同，
因此 **生成速度、整筆時間與整筆讀取量不是等價輸出的比較**。
輸出不同尚未定位至第一個分歧 token 或 logits；不能直接歸因為品質下降，也不能
視為已通過等價性。它並未更改權重量化或選用近似 experts，仍須檢查分批及計算路徑的數值差異。

這次結果支持先將方案 1 作為實作候選；尚未完成正式採用所需的多工作負載、
重複交錯測量、長 context、Prompt Cache 啟用、取消及下一筆恢復驗證。

## 證據與重現

[summary.json](summary.json) 為摘要；三個 JSON 保存完整設定、環境、commit、
runtime 與 harness hash、模型 manifest hash、前置請求、初始快取 hash、輸出 hash 和 metrics。
base commit `6821af3`，工作目錄另有先前未提交的 App／Qwen／V4.1 修改。
本次未更動 V4 `model.py`、`expert_cache.py`、`generation.py`。

以下為測量當時的重現命令（不可平行啟動模型）。需使用 JSON 中 source hash 對應的
修改前 runtime（base commit `6821af3`）；目前修正後的原始碼已自動釋放 Slots，
因此直接重跑不再代表下方歷史基準。

```sh
PYTHONPATH=runtime .venv/bin/python docs/benchmarks/2026-09-14-throughput-memory-options/compare-baseline.py --mode baseline --output /tmp/memory-baseline.json
PYTHONPATH=runtime .venv/bin/python docs/benchmarks/2026-09-14-throughput-memory-options/compare.py --mode release-slots --output /tmp/memory-release-slots.json
PYTHONPATH=runtime .venv/bin/python docs/benchmarks/2026-09-14-throughput-memory-options/compare.py --mode individual-256 --output /tmp/memory-individual-256.json
```

腳本固定使用此機的 `/Users/yanun/.dsmodel/deepseek-v4-flash-0731.dsv4`。
`compare-baseline.py` 是基準當時的 harness；`compare.py` 額外加入首 token 日誌與
180 秒停止條件，兩候選均在期限內完成。兩個版本皆與各自 artifact 的 harness hash 相符。
方案 1 在隔離程序內操作 private cache state，僅供本次原型測試，不能直接當成正式 API 使用。
原始 log：`scratch/throughput-memory-options-2026-09-14/`。

只有一筆正式測量／方案，OS page cache 未清、桌面工作與方案順序未控制。
時間差不能作為普遍速度承諾；logical expert bytes 也不等於物理 SSD 讀取。
沒有測試 Novel、其他長度、Qwen、V4.1、DSpark 或開啟 Prompt Cache 的情況。
