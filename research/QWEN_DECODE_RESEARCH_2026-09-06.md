# Qwen Decode：資料、同步與呼叫成本

2026-09-06，使用者已選先轉攻 Decode。沿用 M2 Max／64 GiB、現行 weights／top-10、
CLI 1,152 slots、ANE requested=true／0.25、greedy、persistent cache 關閉。
這是新一輪有界研究；不重跑已拒絕的 Prefill kernel、ready-expert、LFU、
next-layer prefetch 或 staged loading。先前證據見
[第一輪報告](QWEN_PREFILL_DECODE_RESEARCH_2026-09-06.md)。

本輪狀態：**找到有重複測量支持的 Decode 研究原型，尚未產品採用**。
後續已開始 [default-off runtime 整合](QWEN_DECODE_INTEGRATION_2026-09-06.md)；
下文「未修改 runtime」描述的是第一輪結束時的歷史狀態。
短 code 的 Decode 改善約 11–12%，完整 request 改善約 3.5–3.9%；
五類短／長輸出覆蓋均與原版一致，詳細範圍與未完成的採用 gate 見下文。

## 論文支持的問題與本機假設

[DeepSpeed Inference，SC22](https://arxiv.org/pdf/2207.00032)，§III，將小 batch
推論的權重搬移、kernel launch 與中間資料流列為優化對象。這支持檢查本機單 token
expert path 的 graph 建構／資料整理成本；不代表 CUDA 的融合或速度可直接移植。
[MoE-Infinity](https://arxiv.org/abs/2401.14361) 支持依實際 expert 活動量研究資料供應，
本輪先量現行讀取等待，沒有假設改 cache policy 會成功。

源碼事實：`StreamingExperts.__call__` 的非 batched path 在每層 materialize router
indices、等待 `get_many`，再為每個 expert 建立 `mx.take`、QMM、concat／restore。
單 token 的十個 experts 都使用同一份 input row。可檢驗的假設是：保持 `_one`、
expert 執行順序與還原順序，省去單 row 的重複 gather／NumPy 分組，可能降低 CPU
dispatch 與 GPU 小資料操作；只有測量能確認它是否占足夠時間。

## 執行前 gate

1. 先以歷史 code-128 text／64 output 跑 fresh control，記錄 Qwen 實際 token 數。
   再做保持計算原路徑的 observer，區分 router materialization、`get_many` 與
   graph 建構 wall time。Materialization 含前一段 lazy GPU 工作，不能全歸為 router；
   graph timer 是提交成本，不能冒稱 GPU compute time。
2. 保存少量真實 expert inputs／weights，capture 另加同步與 I/O，排除最初四次
   single-token calls 的統計；observer 總速度僅作診斷。Token IDs 必須與 control 相同。
3. 只對有量測依據的不同設計做 component replay；相同 input／weights，要求全部
   tensor-exact、finite，然後才做完整模型。正式主要指標 Decode median ≥5% 改善，
   TTFT regression ≤5%、P95 ≤10%、peak MLX allocation ≤15%、expert bytes ≤5%。
4. 完整模型先單一 workload ABBA，失敗即停止；通過再 BAAB、其餘四類與 4K／256、
   短 prompt／cache 路徑。Exact output 必須逐 token 相同。不用 observer timing 採用。
5. 原始 attempts 保存在 `scratch/qwen-decode-2026-09-06/` 新目錄；失敗不覆寫。
   Machine-readable results 整理至 `docs/benchmarks/`，保存 commit、環境、設定、
   cache state、source／manifest hashes 與 output IDs。未通過不改產品預設。

## 第一輪成本與候選篩選

Code text 經本機 Qwen tokenizer 為 143 input tokens。Fresh control／64 output：
request 23.435 s、TTFT 14.662 s、Decode 7.184 tok/s。Observer 的 64 個 output IDs
全部相同。ANE active、0 evaluations／0 fallbacks；143-token prompt 沒有觸發
固定 1,024-token ANE chunk，不能把 active 說成有 ANE 執行收益。

排除每層前四個 single-token calls 後，2,928 layer calls（61 組 ×48 層）的診斷：

| 邊界 | 每組 48 層的 wall time | 解讀 |
| --- | --- | --- |
| Router indices materialization | 66.433 ms | 含前面 lazy GPU 工作與同步，不能全算 router compute |
| `get_many` expert acquisition | 55.173 ms | 含排程、讀取等待與權重 view 整理，非純 physical SSD latency |
| 後續 Python graph construction | 4.516 ms | 未計其後的 GPU 執行 |
| Logical expert bytes | 約 736 MB | 非 physical SSD bytes；OS page cache 未清除 |

`qwen_decode_probe.py` 保存第 0／23／47 層真實 expert inputs／weights。
Replay 的 baseline 必須先重現保存的 output tensor。三組單列 fast path 均 tensor-exact，
但 component 只約 -0.17% 至 +2.20% 改善，停止該候選，不進 full-model。

同時檢查一個實質不同的設計：把已 resident experts 合併至 grouped QMM，減少
per-expert dispatch。這與舊 ready-expert 的「何時開始計算」不同；也不改 LFU、
預測 router 或 staged read。預先組好的權重 oracle 三組 tensor-exact，component
改善約 31.19–34.18%。Oracle 不含 packing，因此不代表可直接採用。

特別保留 [既有 arena 失敗](../docs/RESEARCH.md)：舊 ready-expert 路徑的單一 15 GB
buffer 曾比 per-slot buffers 慢約 10.1%。本輪不是把該配置原樣重跑，也不是宣稱
「單一 buffer 一定比較快」；差別是 Qwen 的 resident grouped QMM，以跨 slot views
取代每個 expert 各自的計算提交。新設計必須自行通過 gate，舊結論的適用範圍不變。

為檢驗可否消除 packing，研究用 `ArenaSlotPool` 保持 canonical blob regions、
`get_many`／read workers／eviction，將 1,152 slots 放在一個 allocation；以 physical
slot IDs 索引跨 slot 的 strided views。只在第一個 single-token model call 之後的
single-token path 使用 grouped QMM；Prefill 計算仍走原方法。

Infrastructure attempts 完整保留：一維 U8 的 3,008,102,400 元素超過 MLX 維度上限，
改為二維 U32；普通 row slice 又產生副本，使 CPU slot 寫入與 GPU arena 讀取脫節。
這是 storage alias 實作錯誤，不是放寬 numerical tolerance。改用 explicit
`as_strided` 並檢查實際位址後，1,152-slot real capture output 完全一致，分散 slots
包含 1,151；覆寫可見、還原 exact。Component 約 0.388 ms，peak MLX allocation
約 3.03 GB。尚未證明整個 runtime 的 lifetime 與效能。

開始 code143／64 fresh-process ABBA；每個 run 記錄完整 command、ANE status、
candidate call count、token IDs、client wall 與 `/usr/bin/time -l` RSS。任何 token、
資源 regression 或 candidate 未實際執行均停止。小測試成功不抵銷這個 gate。

## 交錯重複與覆蓋進度

Code143／64 的 ABBA 與 BAAB 共八次全部 output IDs 相同，兩波分別為：

| 指標 | ABBA control → candidate | BAAB control → candidate |
| --- | --- | --- |
| Decode tok/s median | 7.212 → 8.019（+11.18%） | 7.210 → 8.069（+11.91%） |
| Request seconds median | 23.203 → 22.387（-3.52%） | 23.159 → 22.246（-3.94%） |
| TTFT seconds median | 14.465 → 14.527（+0.43%） | 14.419 → 14.436（+0.12%） |
| Token P95 seconds median | 0.18510 → 0.17024（-8.03%） | 0.18577 → 0.16872（-9.18%） |

Expert bytes 完全相同，peak MLX allocation 約 13.17 → 13.16 GB；沒有因為少讀
experts 或放大 cache 而取得速度。這是 143-input／64-output、M2 Max／1,152-slot
的重複證據，不是所有 workload／硬體都會加速 11% 的宣稱。

其他四種短 prompt、五種長 prompt 先各做一組 exact-token coverage，不能用
單一配對的 timing 取代完整效能 gate。特別地，歷史 long zh_technical prompt
雖設定 max_tokens=256，兩個模式都在 9 個 tokens 後 EOS，因此只證明 9-token parity。
另外建立
[長篇中文補充 prompt](../docs/benchmarks/prompts/2026-09-06-qwen-decode-zh-extended.txt)，
保留原文並加上二十段技術教學的明確要求，用自然生成補足長 Decode 覆蓋；不忽略 EOS、
不更改原始 prompt／原始結果，不以 output 上限代替實際 token 數。

研究 runner 的 status `decode_mode` 與 `arena_state` 才是此候選的模式與呼叫證據；
通用 runner 的 `mode=control` 表示未啟用先前的 Prefill QSA 候選。
目前只有 fresh-process single-request 合約；尚未在產品整合 request reset、
多輪 cache／重啟 cache 與 App 4,096-slot 測試。功耗／能耗未測量。

## 本輪完成結果與證據

新增長篇中文實際為 4,155 input／256 output tokens，兩個模式完整一致。
Code 的八次交錯測量，加上十組覆蓋配對（二十次完整模型執行），均符合對應基準。
另外的 initial control／observer 只作成本診斷，不混入以下效能結論。

下表是 **單次配對** 的延伸 screening，不能據此宣稱已完成每個 workload 的
重複效能驗收。輸出相同且 expert bytes 相同；觀察到的 peak MLX allocation
變化最高約 +3.58%、TTFT 最高約 +3.30%、P95 均下降，仍需重複驗證泛化。

| Workload | 實際 input／output tokens | Output IDs | Decode 單次差異 | Request 單次差異 |
| --- | --- | --- | --- | --- |
| 短中文 | 128／64 | exact | +12.44% | -2.37% |
| 短工具訊息 | 142／64 | exact | +10.54% | -4.44% |
| 短數學 | 134／64 | exact | +10.24% | -3.80% |
| 短重複文字 | 128／64 | exact | +14.16% | -1.81% |
| 長工具訊息 | 4,539／256 | exact | +10.93% | -4.26% |
| 原長中文，提早 EOS | 4,096／9 | exact；不是 256-token 覆蓋 | +10.16% | +1.09% |
| 長數學 | 4,291／256 | exact | +9.59% | -3.83% |
| 長程式碼 | 4,577／256 | exact | +11.34% | -4.21% |
| 長重複文字 | 4,096／256 | exact | +14.40% | -3.57% |
| 補充長篇中文 | 4,155／256 | exact | +11.25% | -5.07% |

較快的 Decode 不保證每個短輸出 request 都更快：原長中文只有九個輸出，該單次
request 仍慢約 1.09%。此處不以平均掩蓋這個結果，也不宣稱任務品質或省電改善。

Durable evidence：
[summary 與 110 份原始證據 index](../docs/benchmarks/2026-09-06-qwen-decode-arena-m2-max/summary.json)。
其中三份大型 safetensors captures 留在本機 scratch，index 保存位置、大小與 hash；
其餘 raw logs／JSON／失敗版本 source 已複製到 benchmark 目錄。包含 commit、
dependency／native ANE bridge hashes、環境、power／swap、CLI commands、cache state、
完整 token IDs／hash、RSS、MLX memory 與 expert bytes。

兩項停止的設計：單 row gather 精簡收益不足；原版 slicing 的 arena 實作不能共享
CPU／GPU bytes。後者以 explicit strided alias 與位址／overwrite／restore 檢查修復，
不是放寬 tolerance。成功的是修復後的 **arena＋resident grouped QMM 組合**。

本輪研究交付已完成；沒有修改 production runtime、模型、預設值或 cache format。
若進入產品整合，下一個必要階段是 request lifecycle／warm-restart cache、其他
workload 的重複效能 gate、App 4,096 slots，以及 M5 Pro 重現。這些尚未完成，
不能將目前獨立 runner 視為已可替換整個 App 的正式版本。
