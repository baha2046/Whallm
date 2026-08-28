# 結論

這個專案目前最主要的瓶頸不是單一矩陣乘法算子，也不是單純「SSD 不夠快」，而是以下三者共同形成的序列化臨界路徑：

1. **每個解碼 token 需要的專家權重搬運量太大。**
2. **專家讀取、MLX 圖執行、統一記憶體與 CPU／GPU 同步之間仍存在大量氣泡。**
3. **現有 DSpark 雖然能一次接受多個 token，但它的草稿模型成本、獨立快取與預填充成本抵消了收益。**

真正最值得研究的方向不是單獨換成 DFlash2、MTP 或 ANE，而是打造一個：

> **儲存感知推測解碼器（storage-aware speculative decoder）**

讓草稿模型不只負責猜 token，還同時預測下一輪會用到哪些專家；驗證階段則依照整個 token block 的**專家聯集（expert union）**去重、合併讀取，並用 SSD 流量而不是只用接受率決定草稿長度。

這是目前最可能把解碼速度從約 6.5～8 tok/s 推到另一個級別的方向。

---

# 一、目前真正的瓶頸在哪裡？

## 1. 解碼首先受限於專家權重流量

DeepSeek-V4-Flash 有：

- 43 層
- 每層 256 個 routed experts
- 每個 token 每層選擇 6 個 experts
- 每個 expert blob 約 13.37 MB
- 主模型共有 11,008 個 routed expert blobs
- 目前 cache 只有 1,152 slots，也就是最多容納約 **10.5%** 的 routed experts

若完全沒有快取命中，一個 token 最壞要碰：

\[
43 \times 6 = 258\text{ 個 expert blobs}
\]

對應約：

\[
258 \times 13.37\text{ MB} \approx 3.21\text{ GiB/token}
\]

目前 runtime 已使用全域最少使用頻率（Least Frequently Used, LFU）快取、ready-expert 執行與 `preadv`，但一般非重複 prompt 仍有約 24,000～29,000 次 miss；重複 prompt 可達 12.22 tok/s，而一般 prompt 只有約 6.44～8.28 tok/s，顯示專家工作集與 SSD／快取壓力會直接決定解碼速度。

從 R0 的 4K 測試粗估，扣除 layer-major 預填充讀取的 42 個完整 expert layers 後，每個生成 token 仍對應約 **1.29 GB 的邏輯 expert read 流量**。這不是實體 SSD 一定真的讀了 1.29 GB，因為其中可能包含作業系統頁面快取、重疊讀取與邏輯計數；但它說明未來最該新增的指標是：

- 實體儲存讀取 bytes
- 作業系統頁面快取命中
- runtime expert cache 命中
- speculative prefetch 的 useful／wasted bytes
- 每個 committed token 的總讀取量

現在只看 `expert_bytes`，還無法完整區分這幾層。

---

## 2. GPU 在真實解碼工作負載下沒有吃滿

Metal 系統追蹤結果顯示：

| 工作負載 | 解碼 GPU busy |
|---|---:|
| Repeated 4K | 57.8% |
| Tool-like 4K | 35.5% |
| Tool-like 14K | 37.1% |

Tool-like 工作負載同時有更多 expert miss、`preadv` CPU sample 與更長 GPU idle boundary。這表示一般解碼主要是在等待：

- 專家 ID 從 GPU 回到 CPU
- 快取查詢與 slot admission
- SSD 讀取
- MLX array ready
- CPU／GPU synchronization
- 下一批 Metal 工作提交

而不是 GPU 算不完。

這也解釋了為什麼現在立刻重寫一個超快 FP4 Metal 核心，不一定能大幅改善解碼：GPU 本來就有約六成時間沒有工作。

---

## 3. 增加快取反而可能讓同步成本爆炸

將 expert cache 從 1,152 slots 增加到 2,048 slots：

- expert bytes 降低 21.73%
- miss 降低 31.34%
- eviction 降低 36.77%

但 Code 4K 解碼卻從 8.25 tok/s 掉到 2.39 tok/s，routing synchronization boundary 從 9.829 秒增加到 111.488 秒，記憶體峰值增加 11.156 GiB。

這代表目前不只是快取演算法問題，還可能包含：

- MLX slot 首次配置與 lazy evaluation
- 大量 MLX array ownership
- 統一記憶體壓力
- wired memory 增加
- cache lookup／heap 操作
- slot reuse 前的同步
- GPU command dependency 增加

因此「塞更多 experts 進 RAM」不是解答。下一步應該先建立**固定大小、零配置（zero-allocation）的 expert slot hot path**，將 slot ownership 與 Metal fence 明確化，再重新測快取容量。

BaseRT 在 Apple 晶片上採用完全預先配置的解碼緩衝區、零配置解碼迴圈、原生 Metal dispatch 與融合反量化核心；在一般常駐模型上相較 MLX 最高有約 1.35 倍解碼提升。但在大型 MoE 上，因記憶體頻寬成為主要限制，解碼差距只剩約 1～7%，也再次說明單純換 runtime 不會消除 expert streaming 問題。

---

## 4. 預填充是一個不同的瓶頸

4K 以上的 layer-major 預填充目前會：

1. 計算當層 attention
2. 預取下一層
3. 將當層全部 256 個 routed experts 讀進連續 buffer
4. 使用 4,096-token tile 執行 `gather_qmm`
5. 逐層執行

cache-only prefill 通常會完整讀取 42 個 expert layers，也就是：

\[
42 \times 3.19\text{ GiB} \approx 133.9\text{ GiB}
\]

這對 4K 冷啟動預填充來說，已經相當接近「大量連續 SSD 串流」工作負載。

但預填充也不是純 SSD-bound：

- 4K Tool-like 的 GPU busy 是 60.7%
- 14K Tool-like 上升到 70.2%
- `gather_qmm` 只佔 4K request 約 13.99%
- 24.53% 的 expert runs 完全沒有 rows
- 27.37% 只有 1～15 rows

因此短預填充較偏 SSD 與流水線瓶頸；長預填充則逐漸變成 attention、routing、expert GEMM 與 SSD 的混合瓶頸。

由 24.53% 的空 expert runs 推算，即使能無成本地完全不讀空 experts，expert bytes 的理論上限也只有約：

\[
\frac{1}{1-0.2453}\approx1.325\times
\]

而且這只是 expert bytes，不是端到端預填充速度。因為 selective read 會失去部分連續讀取與 next-layer overlap，實際提升一定低於 1.325 倍。

---

## 5. 現有 DSpark 的問題不只是接受率

目前 DSpark 測試非常關鍵：

| 模式 | Request | TTFT | 總吞吐 | 回報 Decode | Peak |
|---|---:|---:|---:|---:|---:|
| Normal | 31.152 s | 22.072 s | 2.054 tok/s | 6.94 tok/s | 23.274 GiB |
| DSpark | 41.301 s | 33.722 s | 1.549 tok/s | 8.32 tok/s | 27.257 GiB |

DSpark 的兩次測試都完整接受了 5 個 draft tokens，但：

- request 慢 32.58%
- TTFT 慢 52.78%
- 總 throughput 下降 24.60%
- 額外使用約 4 GiB peak memory
- 第一輪後就 fallback

所以這不是「draft 猜得不準」。它甚至完整接受了五個 token。問題是 draft 與 runtime 成本太高。

目前 DSpark 還有：

- 10.12 GiB packed weights
- 獨立 768-slot expert cache
- 約 9.56 GiB 的 expert cache 容量
- 不使用一般 prompt cache
- 不使用主模型 layer-major prefill
- 與主模型形成兩套 expert 工作集

這會在 unified memory、SSD、MLX allocator 與 prompt prefill 上直接和 target model 競爭。

官方 DSpark 的 60～85% per-user speedup 是在多請求、高併發、matched aggregate throughput 的 production serving 環境中得到的。它的 confidence scheduler 主要在避免高併發時浪費 verification batch capacity，不能直接套成單一 Mac、batch size 1 的倍率。

---

# 二、最值得做的核心設計：SSD-aware speculative decoding

我建議設計一個專用的 **V4-FlashSpecSSD** 流水線。

它的資料流應該是：

**目前已提交的 target hidden state / KV cache**

→ 常駐草稿模型產生 4～8 個候選 token  
→ 同時預測每層可能使用的 target experts  
→ Metal I/O 依 deadline 預取 experts  
→ target 一次驗證整個 token block  
→ 每層依 expert union 去重與合併讀取  
→ commit accepted prefix  
→ 只 replay correction 所需的最短部分

這裡最重要的不是平均接受長度，而是：

\[
C_{\text{token}}
=
\frac{
T_{\text{draft}}
+
T_{\text{unhidden I/O}}
+
T_{\text{verify}}
+
T_{\text{replay}}
}{
N_{\text{committed}}
}
\]

以及：

\[
B_{\text{token}}
=
\frac{
B_{\text{useful prefetch}}
+
B_{\text{on-demand}}
+
B_{\text{wasted prefetch}}
}{
N_{\text{committed}}
}
\]

對這個專案而言，**每個 committed token 的 SSD bytes**，通常比單純的 draft acceptance rate 更重要。

SP-MoE 的研究正好指出，一般推測解碼直接套到 offloaded MoE 上，會因多 token verification 啟用更多 experts，增加頻寬競爭；被拒絕 token 所載入的 experts 還會浪費 I/O。它提出在 drafting 階段用 draft hidden state 加 target router 預測 experts，並限制預取深度，避免 cache thrashing。

## 這個架構應包含五個關鍵技巧

### 1. 三個 Hash-routing layers 做完全精準預取

DeepSeek-V4 的前三個 MoE layers 使用 Hash routing，expert ID 只由 token ID 與固定 hash function 決定。因此 draft tokens 一產生，前三層需要的 experts 已經完全確定，不需要模型預測。

可以在 target 還在完成上一輪時立即預取：

- 第 0～2 層 draft token expert union
- 去除 cache resident experts
- 合併重複 experts
- 依 block 內第一個使用位置排序 deadline

這只覆蓋 3/43 層，直接上限大約只有 7%，但它能讓每一輪 verification 一開始就保持 GPU busy，並為後續 predicted prefetch 建立流水線。

---

### 2. 對其餘層建立 draft-to-target router predictor

對第 3 層之後，可將草稿模型每層的 attention output 或壓縮 hidden state，送進 frozen target router，預測 target 可能使用的 top-k experts。

SP-MoE 在其他 MoE 模型上觀察到相鄰 draft tokens 有高度重疊的 expert sets，使用 draft features 加 target gates 可取得約 88% 的 top-1 預測準確率；但預取超過三層後開始明顯增加 thrashing，超過五層甚至可能失敗。這些數字不能直接視為 V4 結果，但它提供了很合理的初始 cutoff：先只預取前 2～3 層。

HOBBIT 在 Mixtral 上曾觀察下一層 top-1 expert 可達約 96% 預測準確率，但解碼實際只改善約 1.05 倍，因為錯誤預取的懲罰很高。這說明 predictor accuracy 本身不是成功標準，必須直接量測 useful prefetch bytes 與 wasted bytes。

---

### 3. Verification 必須以 expert union 為基本單位

假設 draft block 有 5 個 token，不能再按照「5 個 token × 各自 6 experts」分別處理。

每層應先建立：

\[
U_l(k)=\bigcup_{t=1}^{k}Experts(l,t)
\]

然後：

1. 每個 expert blob 最多載入一次
2. 收集所有對應 token rows
3. 一次執行 grouped／gather FP4 QMM
4. resident 與 missing experts 分成不同工作佇列
5. missing expert 還在讀時，先執行 resident experts
6. speculative-only experts 可寫入 temporary verification scratch，不一定污染長期 LFU cache

目前文件顯示 verifier 已經能一次處理完整 token block，但沒有足夠證據顯示 SSD I/O、cache admission 與 QMM dispatch 已完全以 per-layer expert union 去重。這應該是第一個需要確認的地方。

---

### 4. Draft block 長度必須由 SSD 成本動態決定

一般 DSpark confidence scheduler 根據 prefix survival probability 與 serving throughput 決定 verification length。對本專案，scheduler 還應納入：

- 預測 expert union 數量
- 已在 cache 的比例
- 預估 SSD bytes
- prefetch queue depth
- GPU idle ratio
- 目前 memory pressure
- 草稿每個 position 的存活機率

可以定義：

\[
Score(k)
=
\frac{
E[\text{committed tokens}\mid k]
}{
T_{\text{draft}}(k)+
T_{\text{verify}}(k)+
\max(0,T_{\text{I/O}}(k)-T_{\text{hidden}})
}
\]

不是固定 block size 5，而是在 1、2、4、6、8 之間動態選擇。

當 predicted expert union 太大時，直接退回 MTP-1 或 autoregressive decode；當 code、JSON、重複語法使接受率與 expert reuse 很高時，才放大 block。

---

### 5. 做 storage-aware candidate path selection

DFlash2 不是只在每個位置留一個 token，而是保留多個候選，再由輕量 selector 找出一條連貫路徑；它還使用 dynamic convolution 改善 block 尾端的品質。官方 Qwen3.8-27B DFlash2 模型卡在單張 H200、concurrency 1 上報告約 2.67～3.43 倍 resident-GPU speedup，但這不能直接套用到 SSD-offloaded V4。

可以把 selector 的目標從：

> 最大化 draft probability

改為：

> 最大化預期 accepted tokens／預測 expert-union bytes

例如：

\[
PathScore
=
\log P_{\text{draft}}(\text{path})
-
\lambda B_{\text{expert-union}}
-
\mu B_{\text{not-resident}}
\]

對 greedy decoding，只要 target 仍逐 token 驗證，輸出 token 可以保持 target-equivalent。對 sampling，則必須保留正確 proposal probability 並實作完整 rejection sampling，不能只用任意 heuristic 選路徑。

這是很值得研究、且目前文獻中較少直接針對 SSD MoE 設計的組合。

---

# 三、草稿模型應該怎麼選？

| 方案 | 優點 | 問題 | 對本專案的判斷 |
|---|---|---|---|
| 原生 MTP-1 | 最低額外成本、target-aligned、可能直接重用 target features | 長期 draft accuracy 容易衰退 | **第一個 baseline** |
| 現有 DSpark | 已有官方權重，semi-autoregressive + confidence head | 10.12 GiB、獨立 MoE cache、沒有一般 prompt reuse | 架構正確，但目前 runtime integration 不適合 |
| HyperDFlash | 專門對齊 V4 的 mHC residual，接受長度高 | 需要自行訓練與新 runtime | **最適合 V4 的高潛力方向** |
| DFlash2 | 一次產生整塊、candidate path selector、尾端衰退較小 | 目前公開 checkpoint 是 Qwen3.8，不可直接套 V4；公開資料仍很新 | 適合作為 drafter backbone 設計 |
| ReDrafter | 小型 RNN、target hidden-conditioned，Apple 已有 MLX 實作 | 仍需 V4-specific 訓練 | **很適合先做低成本 Apple 原型** |
| 獨立小型 LLM | 容易取得 | 重複 prefill、KV、tokenizer、記憶體與同步成本 | 不建議 |

DeepSeek-V4 的原生 MTP depth 是 1，因此先確認 0731 checkpoint 是否包含可直接執行的 MTP inference weights。若保留，應先實作 MTP-1，建立「幾乎零額外 prompt prefill」的最低成本基線。

Apple 的 ReDrafter 使用 target hidden states 驅動小型 RNN drafter，搭配 dynamic tree attention 與知識蒸餾，在 M2 Ultra 的 MLX／Metal 上曾取得最高約 2.3 倍 resident-model speedup。這個倍率同樣不能直接搬到 SSD V4，但它證明「target-conditioned tiny drafter」很適合 Apple 晶片。

## 我認為最合理的新草稿模型

可以設計一個 **V4-FlashDraft-ANE**：

- 輸入：target 最終 pre-collapse mHC residual
- reducer：直接繼承 target 的 `hc_head` gated reducer
- backbone：1～3 個小型 dense blocks，不使用 MoE
- block size：支援固定 2／4／8 三種 shape
- tail modeling：DFlash2 式 dynamic convolution，或 DSpark 的低秩 Markov head
- confidence head：預測每個 position 的 prefix survival probability
- router heads：預測 43 層的 expert sets
- 記憶體目標：完全常駐，不允許自己的 SSD expert cache
- 量化目標：依 Core ML／Metal 實測選 FP16、W8 或 W4，而不是先假設最低 bit 一定最快

HyperDFlash 特別值得參考。它發現普通 DFlash 與 DeepSeek-V4 的流形約束超連接（manifold-constrained hyper-connections, mHC）不對齊，因此改用 target 已為 MTP 保存的 final pre-collapse residual，並將一般 67M 參數 reducer 換成由 target `hc_head` 初始化的 65K gated reducer。其 vLLM resident-GPU 實驗中，平均接受長度約從 MTP-3 的 2.93 提升到 3.69，平均解碼加速從 2.25 倍提升至 2.80 倍。

不過 HyperDFlash 使用約 300K general examples 加 150K task-oriented examples，並在 8 張 H20 上各訓練 5 epochs。這對個人專案偏重，因此可先縮小為：

1. 使用 target-generated code／tool／chat 資料。
2. 只儲存 tapped residual、top-k logits 與 expert IDs，不儲存完整 target KV。
3. 先訓練 block size 4。
4. KL 蒸餾只放前兩個 draft positions。
5. 額外加入 expert-route prediction loss。
6. 以實際 SSD cost 作為 candidate selector 的訓練或排序訊號。

---

# 四、ANE 到底能不能幫忙？

## 不建議：把主模型 routed experts 丟到 ANE

Apple 神經網路引擎（Apple Neural Engine, ANE）目前不適合直接接管這個專案的動態 routed expert path，原因是：

- ANE 偏好靜態 graph 與靜態 shape
- MoE 需要動態 top-k、scatter、gather 與 variable expert load
- 每層 CPU／ANE／GPU 的切換會產生 synchronization
- 主模型 routed experts 是 checkpoint-native FP4
- Core ML path 不一定能直接以相同 FP4 layout 執行
- 145 GiB expert weights 仍然無法常駐
- SSD streaming 問題不會因換成 ANE 消失

NPUMoE 的 Apple ANE 研究也主要假設所有 expert weights 已經常駐統一記憶體，鎖定長 context prefill；它明確指出單 token decode 無法有效攤平 CPU–NPU coordination overhead，而且測試中的 GPU 仍比 ANE 快 1.26～5.4 倍。

## 建議：讓 ANE 執行草稿模型與 predictor

ANE 最合理的用途是：

- 常駐 dense draft model
- confidence head
- route predictor
- candidate path selector
- 小型 target-hidden reducer

這些工作具有：

- 固定 block size
- 固定 hidden dimension
- 無動態 expert weights
- 相對小型
- 可與 GPU target verification 並行

理想的異質流水線是：

- **ANE**：產生下一個 draft block 與 expert prediction
- **Metal I/O**：依 prediction 預取 experts
- **GPU**：驗證上一個 draft block
- **CPU**：管理 cache、deadline 與 commit

但必須實測 ANE 與 GPU 同時執行時的統一記憶體頻寬競爭。統一記憶體代表不需要傳統 PCIe copy，不代表 CPU、GPU、ANE 各自擁有獨立無限頻寬。

---

## M5 GPU Neural Accelerators 與 ANE 是兩件事

M5 的 GPU 內另有 Neural Accelerators，MLX 可經由 Metal 4 的 Tensor Operations 與 Metal Performance Primitives 使用；這不是 Core ML 的 ANE。

Apple 的測試顯示，M5 在大型矩陣乘法主導的 TTFT／prefill 上可比 M4 提升約 3～4 倍，但 resident model 的後續 token generation 主要仍受記憶體頻寬限制，只提升約 19～27%。

對 DeepSeekV4SSD 的含意是：

- M5 GPU Neural Accelerators 可能明顯幫助長預填充與 block verification。
- 不會直接解決 SSD expert miss。
- 需要確認 `mxfp4_gather_qmm` 是否真的走 TensorOps／Neural Accelerator path。
- 若目前 custom gathered FP4 path 沒有使用 TensorOps，可能值得做專門的 grouped FP4 verification kernel。

---

# 五、Apple 統一記憶體與 Metal I/O 還能怎麼利用？

## 1. 將 `preadv → MLX array` 改成明確的 Metal I/O pipeline

Metal 快速資源載入（fast resource loading）可以：

- 排入大量非同步檔案讀取
- 直接寫入 Metal buffers
- 使用 Metal event／fence 與 GPU 工作同步
- 使用獨立 I/O command queue
- 減少 staging 與 CPU command overhead

Apple 官方特別強調其可直接載入 Metal buffer，並讓 I/O 與 GPU command 使用相同同步模型。

但 MTLIO 不是免費加速器。它能改善的是：

- CPU dispatch
- buffer ownership
- copy
- synchronization
- I/O queue depth
- latency consistency

它無法突破：

- 實體 SSD throughput
- expert ID 尚未決定
- 錯誤 prefetch
- unified memory bandwidth

正確做法是先做 microbenchmark：

| 測試 | 比較 |
|---|---|
| 單一 13.37 MB expert | `preadv` 對 MTLIO |
| 6 個並行 experts | p50／p95 latency |
| 32～128 experts queue | aggregate bandwidth |
| Shared MTLBuffer | 是否有額外同步 |
| Private MTLBuffer | 是否產生 copy |
| GPU 讀取重疊 | 真正 hidden I/O 時間 |
| GPU＋ANE 並行 | memory bandwidth contention |

---

## 2. staged `w13/w2` streaming

目前 expert blob 實體排列為：

- `w1`
- `w1 scale`
- `w2`
- `w2 scale`
- `w3`
- `w3 scale`

runtime 已會建立 fused `w13` view。

因此可考慮：

1. 優先讀取 `w1 + w3`
2. 立即執行 gate／up projection
3. 同時讀取 `w2`
4. activation 完成後直接執行 down projection
5. 使用雙 buffer overlap 下一個 expert

單 token decode 的計算量可能太小，無法完全隱藏 `w2` read；但在 4～8 token block verification 時，QMM 計算時間較長，這個技巧會更有價值。

所以 staged streaming 最好和 speculative block verification 一起做，而不是單獨做。

---

## 3. Cache admission 應區分長期與 speculative experts

目前全域 LFU 對所有 expert read 使用相似 admission 行為，但 speculative verification 會產生大量「可能只使用一次，甚至最後被 reject」的 experts。

建議分成：

- **Resident cache**：長期 LFU／recency cache
- **Verification scratch**：本輪 block 專用
- **Prefetch probation**：先讀取但未確認使用
- **Pinned deadline set**：未來 1～3 層一定會用
- **Bypass path**：低重用預期的 expert 用完即丟

這能避免長 draft block 將真正的 hot experts 驅逐出去。

---

# 六、預填充最有機會的改法

## 1. Adaptive full-layer / selective expert loading

現在 4K 以上固定載入當層全部 256 experts。可以改為：

- router union 比例高於門檻：維持 full-layer sequential read
- union 比例較低：只讀取 selected experts
- 預測準確且能 overlap：使用 predicted selective read
- 預測不穩定：退回 full-layer read

例如先測試門檻 70%、80%、90%，但不能直接假定 75% 最好。每個 expert blob 已有 13.37 MB，並不是 4 KB 級小型隨機讀取，因此 queue depth 充足時 selective I/O 仍可能有效。

由目前 24.53% 空 expert run 推算，這個方向對 expert bytes 的理論空間約 1.325 倍；端到端比較合理的目標應該是雙位數百分比，而不是期待兩倍。

---

## 2. 不要先把全部時間投入 `gather_qmm`

目前 `gather_qmm` 只佔 request 約 13.99%。即使將它加速兩倍，依阿姆達爾定律（Amdahl’s law），端到端最高也只有約：

\[
\frac{1}{0.86+0.14/2}\approx1.075\times
\]

也就是大約 7.5%。

因此 custom Metal `gather_qmm` 應排在：

1. 減少 expert bytes
2. 增加 I/O overlap
3. 移除 synchronization bubbles
4. 固定 buffer ownership

之後。

原生 Metal runtime 對短 MoE prefill 的確可能有效；BaseRT 在 Qwen3-30B-A3B 的短 prefill 相較 MLX 曾達到約 1.78 倍，但 prompt 變長後差距逐漸縮小，因此不能直接推論 DeepSeekV4SSD 也會有同樣倍率。

---

## 3. 強化 block-granular persistent prefix cache

對 Codex、工具呼叫與 agent 工作負載，以下內容通常高度重複：

- system prompt
- tool schema
- repository instructions
- conversation prefix
- coding agent policy

目前 runtime 已有兩個 memory prompt cache timelines 與最多八個 persistent entries。可以進一步改成：

- 內容雜湊的 block-level prefix index
- 支援非完整 prompt match
- 多 request 共用 immutable KV blocks
- 依 revision、rope、KV format、attention mode 做完整 cache key
- 優先保留高重用 system/tool prefixes

這無法改善 cold prefill，但對實際 Codex 體感 TTFT 可能比任何 10% kernel optimization 更有價值。

---

# 七、願意修改模型時，還有更激進的方案

以下不再保持原始 checkpoint 的完全等價性。

## 1. Router locality fine-tuning

ReMoE 只微調 router，不修改 experts，加入 temporal reuse 與 trust-KL 約束，使相鄰 token 更傾向重用近期 experts。

它在 DeepSeek-V2-Lite 上將 expert overlap 提高 26.4%；在 Jetson Orin SSD-backed 測試中，解碼速度提高約 1.77～1.99 倍。這不是 V4 結果，但它證明對 SSD MoE 而言，**訓練 router 讓工作集更 cache-friendly，可能比換 cache 演算法更有效**。

適合 DeepSeek-V4SSD 的版本可以：

- 凍結全部 expert、attention 與 embedding
- 只微調後 40 層 learned routers
- 前三個 hash layers 不動
- loss 同時約束原 router distribution、expert balance 與短期 reuse
- 以實際 1,152-slot simulator 計算 miss cost
- 加入 code／tool／chat domain 混合資料

但這會改變模型輸出，必須重新跑完整能力與安全評估。

不應直接在 inference 時強迫選 cache 內的 experts。ReMoE 的實驗中，過強 cache-aware rerouting 雖將 unique hit rate 推到 63.88%，perplexity 卻從 6.35 崩壞到 3607.92。

---

## 2. Mixed-precision miss experts

可以保留：

- 高重要度或 cache-hot experts：checkpoint-native FP4
- 低 router score 且 cache miss 的 experts：更低精度
- 最低分 experts：選擇性略過

HOBBIT 在其他模型上以 router score 估計 expert importance，混合 FP16／INT4 並略過極低重要度 expert，可降低載入量並取得約 1.19～1.57 倍的部分加速。

但 DeepSeek-V4 experts 已經是 FP4，再往下通常要：

- 2-bit／ternary
- residual correction
- bit-sliced progressive loading
- quantization-aware training

這會是高風險 approximate mode，不適合先做。

---

## 3. 從模型架構重新設計 storage-native MoE

真正為 SSD 設計的新模型可以考慮：

- top-6 降為 top-4
- sticky／temporally stable routing
- shared base expert + small expert delta
- experts 依共現關係分群
- hash routing 比例增加
- 每組 expert 使用相同低秩 basis
- router loss 直接加入 cache miss cost
- 草稿模型與 target 共同訓練 expert predictability

這可能帶來最大幅度改善，但已經不再是「讓 DeepSeek-V4-Flash checkpoint 跑更快」，而是訓練一個新的 storage-native MoE。

---

# 八、建議的研究優先順序

| 優先級 | 研究項目 | 是否保持 target 等價 | 潛力 |
|---:|---|---:|---|
| P0 | 實體 I/O、expert union、wasted prefetch、GPU boundary 完整量測 | 是 | 必要基礎 |
| P1 | Hash-layer exact prefetch | 是 | 低至中 |
| P1 | Block verification 的 expert union／dedupe／batched I/O | 是 | 高 |
| P1 | Speculative expert cache bypass／probation | 是 | 中至高 |
| P1 | Adaptive block size，目標改為 committed token／SSD byte | 是 | 高 |
| P2 | 原生 MTP-1 baseline | 是 | 中 |
| P2 | 移除 DSpark 獨立 prompt prefill 與獨立大 cache | 是 | 高 |
| P2 | HyperDFlash／ReDrafter 式 resident dense drafter | 是 | 很高 |
| P2 | `w13/w2` staged streaming | 是 | 中，與 block verify 結合較高 |
| P2 | MTLIO + fixed Metal buffers + explicit fences | 是 | 中 |
| P3 | ANE 執行 drafter／predictor／confidence head | 是 | 中至高 |
| P3 | Native Metal zero-allocation decode path | 是 | 中 |
| P4 | Router locality fine-tuning | 否 | 很高 |
| P4 | 低精度 miss experts／expert dropping | 否 | 高，但風險高 |

---

# 最終判斷

**解碼方面，最值得投注的不是單獨優化 FP4 kernel，而是：**

> **MHC-aligned resident draft model + draft-time expert prediction + per-layer expert union verification + SSD-aware adaptive scheduler。**

其中最有創新價值的組合是：

1. 前三個 hash layers 完全精準預取。
2. 其餘 layers 使用 draft hidden state 加 target routers 預測。
3. DFlash2／HyperDFlash 的候選 selector 同時考慮接受機率與 expert-union bytes。
4. speculative experts 使用 temporary scratch，避免污染 LFU。
5. ANE 執行 draft、confidence 與 router prediction。
6. GPU 驗證上一個 block。
7. MTLIO 預取下一個 block。
8. block size 依實際 SSD bytes／committed token 自動調整。

**預填充方面，冷啟動的上限相對明確：**

- 4K path 每次約讀 133.9 GiB routed expert data。
- selective expert loading 的 expert-byte 理論空間目前約 1.325 倍。
- 長 prompt 則會逐漸轉向 GPU compute 與 attention。
- 對實際 Codex 使用，block-level persistent prefix cache 可能比再磨 5～10% kernel 更有價值。

因此最合理的第一個大型原型不是「把 DSpark 打開」，也不是「把全部 experts 移到 ANE」，而是先實作：

> **基於現有 verifier 的 expert-union profiler、Hash exact prefetch、speculative cache bypass，以及以 SSD bytes／committed token 為核心的 adaptive block scheduler。**

這四項不需要先訓練新模型，就能驗證整條研究路線是否真的具備突破目前 6.5～8 tok/s 的條件。