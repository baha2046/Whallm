# B 路線第一輪：Qwen 快取與 expert 共用權重

2026-09-05 使用者選 B：同時研究保持輸出一致與可能改變品質的路線。
本輪不改 App 預設，不訓練新模型，不把少量樣本相同當成品質保證。

## 第一輪選擇

先以 Qwen 為主，因 H3 的舊實驗就是 Qwen，H9 的單個 expert 也比較小。
兩個方向使用相同 installed model，較容易核對資料和測試成本。
DeepSeek 與 Prefill kernel 方向仍在總計畫中。

- H3：沿用 `prefill_guided_24`，重新測現在的 4,096 slots；先確認 offline replay 與 runtime miss 一致，再評估候選。論文依據是 [HOBBIT §3.4](https://arxiv.org/html/2411.01433v2#S3.SS4)。
- H9：先查早／中／晚層的實際 MXFP4 expert。量單一或局部 base 加低秩 delta 在相同儲存預算下的誤差，不先接入生成。依據是 [LorExperts §2–3](https://arxiv.org/html/2608.07814v1) 與 [D²-MoE](https://arxiv.org/html/2502.17298v1)。

## 先登記的條件

1. R2 使用目前 checkout、Qwen 4,096 slots、greedy、exact、MTP off、4 read workers、2 prefetch workers、ANE ratio 0.25、memory limit 48 GiB。
2. 使用既有 9 月 1 日 cache-oracle 的長回答 chat 題目，加上 R0 seed 的背景範例補足長度；code／繁中技術／數學／tool／repeated 各 1K 和 4K，256 output tokens。這些輸入含重複背景，只是可重現的診斷基準；採用前另加真實且未用於挑選候選的題目。
3. 每個 request 是獨立 process，expert／memory prompt cache 從空開始，persistent cache 關閉；OS page cache 不清除，標 `not purged`。
4. 正常基準不開 route trace；另跑五個 4K observer requests，逐 token hash 須與正常基準一致。code 4K 再跑一個正常 request 檢查重現。
5. 保存完整命令、runtime tree／script／prompt／manifest hash、MLX 版本、output token IDs／hash、讀取與時間計數；CLI process wall 包含載入，runtime TTFT 不包含載入，分開記錄。
6. H3 的新離線入口門檻為非 repeated workloads miss 中位至少降低 5%，且任何 workload 不增加超過 2%；這只決定是否值得做時間實驗，不是速度通過線，也不改寫舊 50%-Belady gate。
7. H9 的第一步只評估表示法的空間與誤差。所有大小均對照 installed MXFP4，計入 base、兩個低秩因子與 scales；單層權重誤差沒有能力／速度結論。若相同預算需要大幅逼近原矩陣大小，記錄為該候選不利，不推論整個方向不可行。
8. 任何 route replay／token parity 不一致都先停止依賴該證據的步驟，定位原因；不能以錯的 baseline 繼續。

## 狀態

R1 完成。R2 進行中。H3／H9 尚未產生結果。
原始資料預定放 `scratch/prefill-decode-b-2026-09-05/`，正式整理的 artifact 放 `docs/benchmarks/`。

第一個 R2 嘗試的 repeated 1K／4K 完成，但 code 1K 在 16 tokens 自然結束；hash 正確。
這表示純片段續寫不適合固定 256-token Decode 基準，不是 runtime 正確性失敗。
保留原始失敗紀錄，更新 prompt 建構後以 `scratch/prefill-decode-b-2026-09-05-chat/` 重跑整輪，不混用兩批時間。
修正版仍保留正常 EOS 行為；沒有強迫模型繼續輸出。
