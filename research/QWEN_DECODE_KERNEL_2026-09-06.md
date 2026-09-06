# Qwen Decode 跨區塊 kernel：M2 Max 完整研究 gate 通過

2026-09-06。使用者選 **B：維持原門檻，研究跨區塊 kernel／dispatch**。
本輪已完成研究原型與 M2 Max 驗證，**沒有放寬 gate，尚未產品採用**。
原 CLI `--qwen-grouped-decode` 仍執行先前的分頁 QMM；新 kernel 只由 research
runner 明確啟用。App 預設保持關閉，沒有 commit、release 或模型改寫。

14 個正式波次／56 次 fresh-process requests：Decode +9.59–16.42%，整段 request
縮短 2.35–5.12%，每輪各自通過；11,264 個正式輸出 token 的配對與跨波次一致性
均通過。另外完成 12 次 cold allocation requests、16 次 lifecycle／restart requests。
共 84 次完整模型請求；元件 replay 另計。這些是指定 workload 的本機結果，不是通用
速度或模型能力宣稱。M5 Pro 重現、能耗與 production integration 尚未完成。

[333 份證據索引與原始結果](../docs/benchmarks/2026-09-06-qwen-cross-arena-kernel-m2-max/summary.json)
包含完整 metrics、token IDs／hashes、環境、來源 snapshot、執行命令與
[獨立稽核](../docs/benchmarks/2026-09-06-qwen-cross-arena-kernel-m2-max/raw/final-audit.json)。
上一階段各種 arena 的失敗紀錄保留於 [整合研究](QWEN_DECODE_INTEGRATION_2026-09-06.md)，
沒有把舊不合格結果改寫成通過。

## 假設、論文與實作依據

[DeepSpeed Inference §III](https://arxiv.org/html/2207.00032v1#S3) 說明小 batch
推論受 kernel invocation、資料搬移與矩陣運算實作影響。本輪研究推論：一次 kernel
直接讀取多個 resident buffers，可能消除分頁 QMM 額外呼叫與結果合併成本。
論文沒有驗證 Apple／MXFP4 的此一設計；收益與 exactness 必須本機實測。

[MLX v0.32.0 quantized.cpp](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/quantized.cpp)
與 [fp_quantized.h](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/fp_quantized.h)
是本機版本對應的一手實作依據。固定上游 commit
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`；MIT license 保留於
`research/kernels/MLX-LICENSE.txt`。本機 wheel 版本相同，但未自行重建 wheel。

Gate/up 的 2560 寬採 MLX fast QMV，down 的 640 寬採一般 QMV。保留 FP32 累加、
SIMD reduction、BF16 中間輸出與原 SiLU／乘法；buffer 選擇改為每 expert 使用
(page, local slot)，輸出直接寫入原 expert 次序，然後由既有 MoE 程式加權。
不需要逐頁 concatenate／take，跨頁仍為兩次投影呼叫。

源碼稽核確認 QMV 運算區段與上游相同；整數維度參數從 constant address-space
參照改為值參數，由 wrapper 傳入固定 K／N。因此也包含 shape specialization，
不能把全部收益歸因於單一 dispatch 因素。原型明確限制此 checkpoint 的單列
BF16／2560 寬與 canonical blob offsets；不提供其他模型或 MLX 版本的通用保證。

## 前置條件與原停止門檻

固定 checkpoint、MXFP4 bytes、top-10、LFU、greedy、4 read workers／2 prefetch、
ANE requested=true／0.25、原 arena growth plan。4096 slots 的區塊數量仍是
`[1024,640,736,832,864]`，沒有增加 RAM 預算。前一階段已證明：2048-slot pages
cold peak +29.85% 不合格；512-slot pages 的 Decode +3.28% 不合格；growth plan
長工具 +4.58% 也不合格。本輪改變 kernel，沒有重跑相同設計來碰運氣。

正式效能 gate：每輪 Decode 至少 +5%；TTFT 至多 +5%、p95 至多 +10%、
MLX peak 至多 +15%、expert bytes 至多 +5%；prompt／output token IDs 必須相同。
任何 exact mismatch 或正式 wave 失敗即停止。工程門檻不是論文的最佳參數。

先驗第 0／23／47 層真實 capture，再驗實際 4096-slot sparse locations 的
pointer alias、overwrite／restore，及單頁、多頁、反序、重複 expert。接著依序
執行長工具 ABBA、cold edges、lifecycle、反向長工具、App／CLI 短 code、四類其他
長 workload 正反兩輪。沒有同時執行多個 GPU inference。

## 正式效能結果

Apple M2 Max／64 GiB、macOS 26.6.2、Python 3.14.7、MLX 0.32.0、mlx-lm 0.31.3。
HEAD `7dc9cf8f050c75def77c0563cd7b8ac03f2d8435` 加來源 snapshots；runtime 與
kernel／執行 runner 的 hashes 經逐波核對。Installed manifest SHA-256
`a71f38985d7b46919e4ba5abd5ca37f209c6864e6a51fe635e48cd45788326dc`。
驗證 manifest 與 59 個檔案大小，不宣稱本輪逐 byte 重算整套約 127 GB 模型。

Persistent prompt cache 在效能波次關閉；每次為 fresh process，OS file cache 和
Metal compiler cache 未清除，不能稱整台機器完全 cold。各模式兩次結果取 median，
A 是原始路徑，B 是 cross-arena research runner。下表百分比以各波自己的 A 為基準。
短 code 為 143 input／64 outputs；其他為約 4K input／256 outputs，實際 prompt
counts 與 hashes 均在 metrics。中文使用補充 prompt，沒有把早期 9-token EOS 算作
256-token 覆蓋。除明列 1152 slots 的兩輪外，均為 4096 slots。

| Workload／順序 | Decode | Request | TTFT | p95 |
| --- | ---: | ---: | ---: | ---: |
| 工具／ABBA | +11.90% | -4.33% | -0.91% | -14.15% |
| 長 code／ABBA | +11.35% | -3.39% | +0.30% | -11.22% |
| 長 code／BAAB | +9.59% | -2.46% | +0.96% | -11.10% |
| 短 code／1152／ABBA | +11.47% | -2.35% | +2.60% | -8.53% |
| 短 code／1152／BAAB | +11.08% | -3.63% | +0.21% | -8.07% |
| 短 code／4096／ABBA | +10.94% | -2.47% | +1.91% | -4.21% |
| 短 code／4096／BAAB | +10.96% | -3.32% | +0.56% | -5.30% |
| 數學／ABBA | +11.67% | -3.66% | +0.14% | -11.17% |
| 數學／BAAB | +13.73% | -4.44% | -0.09% | -14.33% |
| 重複文本／ABBA | +16.42% | -3.27% | +1.58% | -17.56% |
| 重複文本／BAAB | +15.21% | -4.03% | 0.00% | -14.67% |
| 工具／BAAB | +12.38% | -4.02% | -0.19% | -13.96% |
| 中文／ABBA | +11.56% | -5.12% | -1.92% | -9.47% |
| 中文／BAAB | +10.25% | -3.68% | -0.24% | -9.38% |

所有波次 expert bytes 相同，MLX peak 下降 0.09–0.21%，ANE active 且無 fallback。
TTFT 最大增幅 +2.60%，p95 每輪都下降；沒有用跨 workload 平均掩蓋失敗。
六組相同 prompt／output 長度的跨波次 tokens 也一致，包含短 code 的兩種容量。

## Cold、lifecycle 與元件證據

Cold `Hi`／4096 slots：1／2／3／4／8／16 outputs 各一個 fresh-process 配對，
全部 output、prompt hashes、expert bytes 一致。Peak 增幅依序為 +7.26%、+2.03%、
+11.50%、+8.40%、+9.92%、+1.93%，均低於原 15%。模型剛載入的 common memory
仍高於規劃下限。這些是 cold allocation gate；單次 latency 不作正式加速宣稱。

16 次 lifecycle requests 涵蓋 cold、memory repeat、branch EOS、實際輸出 5 tokens
後取消、取消後重用、單輸出、單輸出後重用及 fresh-process restart。所有 paired
outputs、cache reuse、expert bytes 一致。Warm／restart 重用 142 tokens；取消後
狀態清除，輸出前綴與原 cold request 一致，沒有把 branch 的 1-token EOS 當成取消。

三個真實 capture 的完整 tensors、finite、跨頁 overwrite visible、restore exact
全部通過；八組排列邊界也全 exact。Component replay 的跨區塊 kernel 約
0.325–0.349 ms，分頁路徑約 0.484–0.520 ms；不含 SSD／完整生成，不能換算端到端
收益。獨立 audit 從原始 metrics 重算全部正式門檻、核對實際新 kernel 呼叫數、
驗證跨波次 outputs 與 lifecycle 前綴，結果通過。Research Python 語法檢查通過；
本輪沒有變更 production runtime，未重跑此前已通過的 305 項 Python suite。

## 下一階段需要選擇的範圍

M2 Max 本機矩陣已完成。原研究列出的 M5 Pro 重現不能由更多 M2 Max 測試替代。
要符合原硬體採用範圍，需要在 M5 Pro 環境重現；目前環境不是 M5 Pro。
也可以先做 default-off production 整合，但必須保留 layout／MLX 限制和 lifecycle
護欄，重新驗證最終執行路徑；這不等於可以開 App 預設或替代 M5 驗證。

另一個尚未實測的方向，是讓同一次 kernel 直接讀現有個別 expert slot，嘗試省去
arena 的額外預留。它延續 DeepSpeed Inference 的小 batch dispatch 機制，並以
[PagedAttention](https://arxiv.org/abs/2309.06180) 避免未使用配置的觀點作 expert
storage 的研究推論；兩篇論文均沒有驗證此 Qwen／Apple buffer 設計。可能增加
kernel buffer 參數與選擇成本，尚無本機正確性或效能結果，不可宣稱優於本輪。
