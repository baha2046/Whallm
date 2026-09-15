# 三模型分開 Prefill／Decode 讀取

日期：2026-09-15。原始碼功能已接入三模型；完整模型速度測試為 V4、Qwen。
V4.1 只完成正式權重格式的讀取驗證，尚無完整 checkpoint。

## 本輪結果

同一個中文技術輸入，4,096 個輸入 tokens、128 個 greedy 輸出 tokens。
每模型兩輪反向順序配對；下表是每輪百分比變化的中位數。
時間負值代表等待減少，生成速度正值代表更快。

| 模型 | 首字時間 | 整次請求時間 | 生成速度 | 判讀 |
| --- | ---: | ---: | ---: | --- |
| V4 | −10.95% | −4.63% | −2.67% | 輸入處理較快；整次請求兩輪均縮短 |
| Qwen | +0.73% | −1.08% | +6.15% | 首字與整體等待接近持平；未達預先設定的首字／整次 5% 門檻 |

V4 整次請求的兩輪變化是 −4.10%／−5.15%；Qwen 是 −2.26%／+0.10%。
Qwen 不能宣稱已有穩定的整體加速。兩模型的有效專家讀取 bytes 都不變。

八次計時生成均與其對照輸出完全一致，無新增 swapout，MLX／RSS 差異均小於 1 GB。
每次計時前另有相同控制請求預熱，共八次預熱。
V4 全走直接路徑，未配置暫存區；Qwen 用四個暫存區，共 10,575,872 bytes（約 10.1 MiB）。
暫存區容量不等於整個程序的 RSS 增幅。

## 條件與限制

- Apple M5 Pro、64 GiB；Python／MLX／mlx-lm 版本見兩模型 JSON 的 `environment`。
- Slots：V4 1,152、Qwen 3,072；LRU，4 個讀取工作者、2 個 Prefill 預讀工作者；MLX 上限 48 GiB。
- 每次為新程序，專家 Slots 為空、Prompt Cache 關閉；作業系統頁面快取不清除，以相同預熱及兩輪反向順序控制差異。
- 只改 `separate_prefill_io`。沒有啟用 route 策略、DSpark 或 MTP；不能與路由感知快取的既有百分比相加。
- 只有一個負載、兩組配對，沒有電力、GPU idle 或實體 SSD 流量歸因。
- 速度快照保存於原始資料。最後回歸版本另還原了 DSpark 拒絕草稿時的快取評估收尾，並更新測試／空白；兩組速度測試均未啟用 DSpark。自動核對其餘 runtime 的語法結構未變，相關差異已列入索引。
- 本次沒有打包、簽章或發布 App。

## 驗證與資料

- Python **395 項**通過，包含三種正式 blob 格式、未對齊區段、EOF、短讀、Prefill 取消、重試及設定相容性。
- Swift **93 項**通過，涵蓋三模型設定及 App 原始碼編譯。
- [總索引與雜湊](summary.json)
- [V4 對照](v4.json)／[Qwen 對照](qwen.json)
- [原始碼、預熱、計時輸出與最終驗證原始碼](raw-evidence.tar.gz)
- [Python 測試](runtime-tests.log)／[Swift 測試](swift-all-tests.log)

功能契約見[分開讀取](../../PREFILL_IO.md)，移除範圍見[封存說明](../../../research/archive/SSD_DIRECTIONS_RETIRED_2026-09-15.md)。

## 重跑

使用已安裝的兩個模型，以保存的原始碼快照或目前同一讀取實作執行：

```sh
PYTHONPATH=runtime .venv/bin/python research/ssd_streaming_ablation.py --model v4 --variants control,split_io --input-tokens 4096 --output-tokens 128 --workload zh_technical --waves 2 --prime --output scratch/prefill-io-v4
PYTHONPATH=runtime .venv/bin/python research/ssd_streaming_ablation.py --model qwen --variants control,split_io --input-tokens 4096 --output-tokens 128 --workload zh_technical --waves 2 --prime --output scratch/prefill-io-qwen
```
