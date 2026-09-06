# B 路線第一輪：Qwen 快取與 expert 共用權重

本文件保存 9 月 5 日第一輪的測量與執行前狀態。使用者已於 9 月 6 日允許持續使用 GPU，
下文當時等待 GPU 的記錄不再是目前的限制。
後續 expert 候選與品質停止判定見 [研究入口](README.md) 和
[Q1A 結果](QWEN_QUALITY_Q1_2026-09-06.md)。

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

後續更新：使用者已於 2026-09-06 允許恢復 GPU 研究；H2 計時完成，見
[Gated Delta 診斷](GATED_DELTA_PREFILL_2026-09-06.md)。下文 GPU 暫停敘述保留為 9 月 5 日的歷史狀態。

R1 完成；R2 的 Qwen 1K／4K 第一批基準、H3 離線篩選與 H9 固定 base 篩選完成。
兩個候選都沒有進入 runtime，也沒有取得速度改善結論。
完整模型、其他模型與長輸入的驗證仍未完成。

使用者隨後要求保留 GPU：後續任何需要 GPU 的指令，必須先告知並等待明確允許。
檢查時最後一個權重分析已自行完成；沒有本輪啟動的 GPU 工作仍在執行。
下一個 GPU 實驗目前暫停，不能把之前選 B 當成新的 GPU 使用許可。

第一個 R2 嘗試的 repeated 1K／4K 完成，但 code 1K 在 16 tokens 自然結束；hash 正確。
這表示純片段續寫不適合固定 256-token Decode 基準，不是 runtime 正確性失敗。
保留原始失敗紀錄，更新 prompt 建構後以 `scratch/prefill-decode-b-2026-09-05-chat/` 重跑整輪，不混用兩批時間。
修正版仍保留正常 EOS 行為；沒有強迫模型繼續輸出。
第一次 chat prompt 建構因 token 邊界無法收斂而在載入模型前停止；改用固定背景與單 token padding 補齊，完整量測改存 `scratch/prefill-decode-b-2026-09-05-chat-v2/`。

## H9 事先固定的範圍

讀取 code observer 前先固定層 0／23／47；每層以 code 前 128 個 Decode steps 的最高使用次數選 base，再選與它共現最多的另一個 expert。
後半段只記錄共現是否持續，不拿來改選。
比較完整 `gate_up` 和 `down` 矩陣，rank 固定為 0／16／32／64／96／128／192／256／384／512／640 等，不依結果挑選層或 expert。
SVD 量的是固定 base 下最好的未加權矩陣誤差；另外量沒有 base、直接低秩分解的對照。
儲存估算列 native MXFP4 base 與 FP16 base 兩種；每八個 experts 共用一個 base 是假設，不是已完成分群的實測容量。
因子使用 FP16 大小，但誤差暫不計入因子量化；metadata 也未計，因此容量與誤差均是偏樂觀的篩選。
本輪沒有做 LorExperts 的 permutation alignment、activation fitting、完整分群或微調，故只能回答這個簡化候選，不能宣稱重現或否定整篇論文。

## 第一輪結果

### 可重現的基準

16 個 fresh-process requests 全部完成，各產生 256 tokens。
五個 4K observer 與一個 code 4K repeat 的完整 token hash 均與各自基準相同。
這驗證本批 route 記錄可用，不能外推為其他輸入的品質保證。

以下只列不開 route 記錄的首次基準，每個格子只有一次量測。
TTFT 是模型載入後到首 token 的時間；Decode 是 runtime 的 tokens/s。

| 題目 | 1K TTFT（秒） | 4K TTFT（秒） | 1K Decode | 4K Decode |
| --- | ---: | ---: | ---: | ---: |
| repeated 控制 | 18.37 | 42.68 | 10.18 | 10.02 |
| 程式碼 | 17.60 | 52.22 | 8.52 | 8.35 |
| 繁中技術 | 17.62 | 44.07 | 8.38 | 9.07 |
| 數學 | 17.30 | 43.55 | 8.47 | 8.02 |
| tool 格式 | 17.63 | 42.53 | 8.25 | 5.16 |

同一個 code 4K 正常重跑，Decode 從 8.35 變成 6.57 tokens/s，輸出完全一致。
因此這些數字適合建立目前的參考與檢查輸出，不適合拿單次差異宣稱優化成功。
後續速度比較仍需交錯順序、成對重複測試；observer 的時間不混入正常基準。
這批保留 process RSS 和 MLX memory，但未逐次量系統記憶體壓力或實際 ANE 執行次數；
也沒有把 read timers 解讀成暴露等待。R2 的完整診斷欄位仍待補齊。

### H3：舊候選在目前 4,096 slots 下不值得直接接入

五題的目前 LFU 離線重播，全部精確對上 runtime 記錄的 Decode misses。
使用同一路徑比較 `prefill_guided_24`，結果如下；miss 表示必須重新取得一個 expert，並非實體 SSD 讀取次數。

| 題目 | 目前 misses | 候選 misses | 候選變化 |
| --- | ---: | ---: | ---: |
| repeated 控制 | 5,363 | 5,353 | −0.19% |
| 程式碼 | 35,636 | 35,165 | −1.32% |
| 繁中技術 | 23,001 | 24,589 | +6.90% |
| 數學 | 28,331 | 31,630 | +11.64% |
| tool 格式 | 32,186 | 36,012 | +11.89% |

非 repeated 題目的 miss 變化中位數是 **增加 9.27%**，沒有通過事先登記的入口門檻。
本候選停止於離線篩選。這不是快取研究整體失敗，也不是 HOBBIT 論文失敗；
本輪只測先前的 `prefill_guided_24`，沒有完整重現 HOBBIT 的多因素與多精度策略。

得到的經驗是：小快取時少讀資料的規則，不能直接搬到較大的快取。
若重啟，需先提出新的分配或淘汰規則與論文依據，用獨立題目檢查；不能只調到這五題好看。

### H9：直接共用未對齊的權重，這個簡化方法不利

事先固定的三層各取一對 experts；前 128 步挑選，後半段只觀察共現。
選中的配對為 layer 0 的 93／7、layer 23 的 387／147、layer 47 的 333／323。
後半段共現分別有 15／72／60 次，但六個權重矩陣與 base 的 cosine 都接近零（−0.00091 至 0.00411）。
因此在這三對資料中，常被一起選中不代表未對齊的權重本身相似。

| 每個修正矩陣的 rank | 假設八個 experts 共用 base，平均容量相對目前 MXFP4 | 六個矩陣的相對重建誤差 |
| --- | ---: | ---: |
| 64 | 少 52.99% | 124.0%–136.0% |
| 128 | 少 18.48% | 110.1%–125.2% |
| 192 | 多 16.03% | 已超過原有容量，不作省空間候選 |

相對重建誤差是「權重差距的大小 ÷ 原權重的大小」。
100% 等同於用全零矩陣替代原矩陣的這項誤差；**它不是回答錯誤率，也不是能力下降百分比**。
rank 64 的直接低秩分解對照誤差為 88.3%–92.3%，也比加這個固定 base 小。
這只是同 rank 的對照；完整同容量設計還須計入 base、因子精度、metadata 與執行成本。

本輪停止「共現配對＋未對齊固定 base」這個候選，不建立完整近似模型。
[LorExperts](https://arxiv.org/html/2608.07814v1) 還包含排列對齊、分群和重建微調，
[D²-MoE](https://arxiv.org/html/2502.17298v1) 的完整方法也不能由這個簡化測試取代。
若重啟 H9，需把通道對齊和實際輸入下的誤差評估納入，先在單層驗證；
容量仍須與目前已經很小的 MXFP4 比較。GPU 部分等待使用者允許。

### Prefill 的下一個問題

本輪另核對已安裝的 mlx-lm 0.31.3：Qwen 的 GatedDeltaNet 呼叫
`gated_delta_update`，其 Metal kernel 在一次 dispatch 內依序處理 T 個位置。
它已避免每個 token 都從 Python 啟動一次運算，但不是
[Gated Delta Networks §3.2](https://arxiv.org/html/2412.06464v1) 的分塊矩陣演算法。
這支持把分塊 Prefill 留作 H2 候選，**尚不能證明該段是瓶頸或改寫就會更快**。

下一個需要 GPU 的步驟是：以實際 Qwen 形狀量出這段運算的時間，
再決定是否值得做單層分塊版本；比較輸出與 recurrent state，才進入完整生成。
改變計算順序也可能改變浮點結果，不能僅靠公式相同就宣稱輸出一致。
**9 月 5 日當時停在執行前；9 月 6 日已獲 GPU 授權並完成後續診斷，現況見文件開頭。**

## 檔案與重做方法

- [證據索引](../docs/benchmarks/2026-09-05-qwen-research-round1-index.json)：commit、環境、hash、保存路徑與限制。
- [16 組基準](../docs/benchmarks/2026-09-05-qwen-research-baseline-m5-pro.json)：完整命令、tokens、時間與記憶體欄位。
- [快取篩選](../docs/benchmarks/2026-09-05-qwen-causal-cache-4096-screen-m5-pro.json) 與 [權重篩選](../docs/benchmarks/2026-09-05-qwen-shared-base-spectral-screen-m5-pro.json)。
- [Prefill 原始碼查核](../docs/benchmarks/2026-09-05-qwen-gated-delta-source-audit.json)：已安裝依賴的 hash；不是速度證據。
- [量測開始時的規則快照](../docs/benchmarks/2026-09-05-qwen-research-baseline-protocol.md)：hash 與基準 artifact 相符；之後追加的 H9 選取條件保留在本文件。
- [十個輸入與 manifest](../docs/benchmarks/prompts/2026-09-05-qwen-research-baseline/manifest.json)；五個 gzip 路徑記錄的保存位置與解壓後 hash 見證據索引。

完整 stdout／stderr 與第一、二次未完成嘗試均保留於 `scratch/`；沒有刪除原始證據。
`Scripts/analyze_research_cache.py` 可使用原始 baseline 重算，僅需 CPU；
移機時先將保存的 gzip traces 解壓至新路徑，另存一份 baseline 路徑映射，保留原 artifact 不改。
`Scripts/benchmark_research_baseline.py` 與 `Scripts/analyze_expert_shared_base.py` 會用 GPU；
使用者已於 9 月 6 日授權持續研究，重跑時仍須避免不同 GPU 實驗並行。

收尾核對：7 個 CPU 單元測試通過，涵蓋基準輸出檢查、低秩誤差／容量公式與既有快取模擬器。
保存檔案的 hashes、五個解壓後 trace hashes、研究文件連結與 `git diff --check` 均通過。
此次沒有重跑完整 runtime 或 App 測試，也沒有改動 runtime 程式。
