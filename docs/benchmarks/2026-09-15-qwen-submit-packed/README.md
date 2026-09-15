# Qwen：Decode 提交與 packed cache 先選列

2026-09-15，M5 Pro／64 GiB／macOS 26.6.2／MLX 0.32.2，起點 `872d6fd`。
這是研究測量，沒有修改 runtime、App 預設或發布成品。

| 分開測量的候選 | 條件 | Decode 配對變化 | 整次請求時間變化 | 本輪結論 |
| --- | --- | ---: | ---: | --- |
| 常駐專家合併提交 | 4K 繁中、普通 K/V | +5.96%／+1.24% | 中位數 −0.19% | 未達 5% 延伸線 |
| packed K/V 先選列再還原 | 16K code、主 K/V packed 8-bit | +15.12%／+15.49% | 中位數 −1.45% | 值得延伸 |

每個方向 A–B–B–A，共 8 次新程序請求；LRU 3072 slots、greedy 64 outputs、
Prompt Cache off、分開 Prefill I/O 開啟。各方向的四次輸出 token hash 完全相同、
讀取 expert bytes 相同，8 次均無新增 swapout。OS page cache 未清空或逐次相同預熱。
兩對不足以代表所有工作量；數字不是信賴區間。主 K/V packed 預設關閉，
本輪沒有與普通 BF16 作同 prompt 的品質比較。

完整方法、原生 GPU 時間軸、停止條件與後續工作見
[研究報告](../../../research/DECODE_SUBMISSION_PACKED_SELECT_2026-09-15.md)。

## 機器可讀資料

- [完整 packed 比較](full-packed.json)：逐次 tokens、metrics、環境、來源、配對結果。
- [完整提交比較](full-submit.json)：相同格式，另有命中分布。
- [packed 部件](packed-component.json)：24 組數值、耗時、記憶體測量；4-bit 主 K/V 為額外研究模式。
- [提交部件](submission-component.json)：全命中單層改善 20.75%，不等於完整模型改善。
- [原版時間軸](submission-timeline.json)：有 ready 工作時的目標程序 GPU 空檔上限，不是可直接消除的時間。
- [批次數比較](submission-batches.json)：原生 GPU 區間，保留 command buffer 層級限制。
- [環境與來源副本](provenance.json)：完整 raw source snapshots 保留在下方 scratch。

原始 `.trace`、XML、逐次 stdout／stderr、source snapshots 與 JSON：
`scratch/prefill-decode-2-3-2026-09-15/`。原始 summary 的分類修正另有留存，未修改測量值。
