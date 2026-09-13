# 模組化 runtime 設計研究

日期：2026-09-12。使用者已選定選項 2 並完成第一輪程式改造。
本文保留當時研究與較長期驗收提案；目前程式以
[模型支援套件](../docs/MODEL_PACKAGES.md) 為準。
目前 runtime 行為仍以 [docs](../docs/README.md) 為準。
已接受自有 fork 與模型支援套件，見 [fork ADR](../docs/adr/0001-owned-mlx-lm-fork.md)
及 [模型套件 ADR](../docs/adr/0002-model-support-packages.md)。

## 問題與結論

目標是讓貢獻者加入模型時，集中實作模型差異，共用 Whallm 的安裝、SSD 讀取、
生成請求、取消、API 和 UI。可行，但新的運算架構仍需要對應的模型程式。
同架構的新 checkpoint 才有機會只新增資料、映射規則及驗證案例。

建議採用選項 2：每個模型提供一個支援套件，對外維持少數入口，
內部重用現有的儲存、量化和對話格式模組。不要要求每個模型都實作十多個介面。

## 現況證據

核對 Whallm `codex/pr10-fixes`，HEAD `5565c2396465724dc4372e91720956b89cd34200`，
含當輪 PR #10 本機修正；這是程式檢查，並非效能量測。

| 現在分散的模型知識 | 程式位置 | 造成的限制 |
|---|---|---|
| 三種模型使用三個 manifest 版本 | [manifest.py](../runtime/deepseek_v4_ssd/manifest.py) `InstalledModel.open` | 格式版本同時承擔模型分派 |
| loader 選 V4、Qwen、V4.1 | [model.py](../runtime/deepseek_v4_ssd/model.py) `load_model` | 新模型繼續增加分支 |
| Prefill、工具格式、狀態重用按模型分支 | [generation.py](../runtime/deepseek_v4_ssd/generation.py) | 生成流程知道模型內部細節 |
| slot layout 知道 `w1/w2/w3` 和 `gate_up/down` | [expert_cache.py](../runtime/deepseek_v4_ssd/expert_cache.py) | 儲存與權重解碼混在一起 |
| V4、V4.1 透過修改 module globals 替換元件 | [model.py](../runtime/deepseek_v4_ssd/model.py)、[deepseek_v41_ssd.py](../runtime/deepseek_v4_ssd/deepseek_v41_ssd.py) | 上游同步及多模型生命週期脆弱 |
| Swift 模型 enum、安裝、修復、UI 各自分派 | [Model.swift](../Sources/DeepSeekRepack/Model.swift)、[ModelLibrary.swift](../Sources/DeepSeekV4SSDApp/ModelLibrary.swift)、[ContentView.swift](../Sources/DeepSeekV4SSDApp/ContentView.swift) | 只整理 Python 仍無法降低完整接入成本 |

最不能硬套成共用結構的是模型狀態：

| 模型 | 必須保留的差異 | 遷移時的限制 |
|---|---|---|
| DeepSeek V4 | 壓縮 attention cache、SSD experts、DSpark | 保留既有生成與快取行為 |
| Qwen3.8 | QSA 雙分支、linear attention state、N-gram store | 狀態還原必須涵蓋所有分支；保留既有 Prefill、MTP 行為 |
| DeepSeek V4.1 | 全模型 cache、跨層共享 KV、Engram store | 目前不支援 prompt-cache restore/trim、MTP 或 layer-major Prefill |

V4.1 的完整 checkpoint 安裝和生成仍未驗證，詳見 [目前支援範圍](../docs/DEEPSEEK_V41.md)。
抽出介面不會自動擴大這個支援範圍。

## 三個選項

| 選項 | 做法 | 優點 | 代價 |
|---|---|---|---|
| 1. 輕量封裝 | 只把三個 loader 收入同一入口 | 改動小，容易維持行為 | 安裝、UI、快取與生成仍有分支；貢獻者仍須理解多處程式 |
| 2. 模型支援套件（建議） | 共用模型描述資料，加安裝 adapter 與 runtime adapter；模型自己管理狀態 | 接入範圍清楚，現有三種模型可逐一遷移 | 首次須整理 Swift/Python 共用資料，以及儲存、狀態的責任 |
| 3. 細粒度組件框架 | attention、state、expert、排程等各自註冊、協商版本 | 適合大量架構組合與長期框架開發 | 介面和版本組合多，貢獻門檻及驗證成本最高 |

選項 2 將選項 3 的少數必要組件留在內部，等第二個實際用途出現才抽出。
本研究不建議以 JSON 描述整個運算圖：處理新 attention、共享 state 和條件邏輯時，
它會變成另一種程式語言，仍需要實作與除錯。

## 選項 2 的責任分配（提案）

```text
模型支援套件
  ├─ 模型描述資料 ─── 名稱、checkpoint、安裝格式、可用設定
  ├─ 安裝 adapter ─── tensor 映射、repack plan、嚴格驗證
  └─ runtime adapter ─ 運算接線、狀態、可用功能
            │
            ▼
Whallm 共用核心
  安裝與修復 │ SSD / slot │ 請求與取消 │ 抽樣與串流 │ API / UI
            │
            ▼
自己的 mlx-lm fork
  ModelArgs / Model │ attention 等模型運算 │ 明確的元件注入入口
```

fork 保留官方模型慣例和一般記憶體內推論路徑；Whallm 專用的檔案路徑、slot policy、
安裝格式和 HTTP 行為留在 Whallm。不要將 Whallm 的 ModelSession 介面也搬進 fork。
先用建構參數注入 routed expert、row lookup 等元件，逐一取代修改 module globals。
效能敏感的函式在載入時接好，不在每個 token 或矩陣乘法時查註冊表。

模型描述資料是一份 Swift/Python 共用、隨 App 發布的資料檔，包含穩定 ID、
API model ID、固定 checkpoint revision、安裝規則 ID、架構 ID、工具格式及設定限制。
它只能引用專案內已註冊的實作，不能用任意 Python import path 指定程式。
設定只引用 App 已支援的控制項與翻譯鍵；新增現有種類的設定值不必改 SwiftUI。

第一版保留兩個語言各自的明確入口，不把 Swift repacker 改寫成 Python：

```swift
// 草案，名稱與型別待第一個遷移案例確認。
protocol CheckpointAdapter {
    func makePlan(_ checkpoint: CheckpointIndex) throws -> RepackPlan
    func validate(_ installed: InstalledModelMetadata) throws
}
```

```python
# 草案；尚無這些公開 API。
class ModelAdapter(Protocol):
    def open(self, installed, resources, options) -> ModelSession: ...

class ModelSession(Protocol):
    def begin(self, request) -> ModelRun: ...
    def close(self) -> None: ...

class ModelRun(Protocol):
    def prefill(self, token_ids) -> ForwardResult: ...
    def decode(self, token_ids) -> ForwardResult: ...
    def close(self) -> None: ...
```

`ForwardResult` 交回 logits 與共用計量資料；模型 cache 留在 `ModelRun` 內。
共用核心負責抽樣、停止條件、串流與請求管理，不讀取模型的 `keys`、`buf_kv` 等欄位。
可選功能在載入時綁定：對話格式、prompt-state 保存/還原、特殊 Prefill。
不支援的功能在載入或請求驗證階段明確拒絕，不放入假實作。
DSpark、Qwen MTP 和 ANE 先保留在對應 adapter 內；遷移要保持其現有功能，
不能只抽出普通 decode 後就宣稱完成。

SSD 共用層管理檔案讀取、slot 與其使用期限；expert codec 決定 expert blob
如何解成模型權重。N-gram 和 Engram 可共享 row 讀取資源管理，但保留各自解碼規則。
一般 dense 模型可不使用 expert store。

## 安全性、狀態與相容規則

1. 驗證固定 revision、tensor 名稱、shape、dtype、packing、檔案路徑及大小後，
   才配置大型 buffer。安裝 byte ranges 的邊界、覆蓋及轉換規則仍須逐項檢查。
2. slot 在 GPU 使用完成前不得被覆寫；第一版沿用已有同步方式，
   不以尚未驗證的外部 Metal event API 作必要條件。
3. 每個載入的模型先維持一次一個 active request；取消或失敗需清理 request 資源，
   隨後第二個 request 必須可執行。`close()` 可重複呼叫，部分載入失敗會釋放已開資源。
4. 可重用狀態必須包含全部 attention/SSM/共享 buffer，保持引用關係；
   只能還原至相容 checkpoint、量化、adapter 及 state 版本。V4.1 初版不提供此功能。
5. 描述檔不能自己宣告驗證規則再自行通過；規則來自隨 App 發布的可信實作。
   第一版接受經專案審查與發版的貢獻，不新增即時下載並執行第三方套件的功能。

manifest 格式版本與模型身分應分離，但不急著改寫已安裝資料。
先保留 v1/v2/v3 readers，轉成共用記憶體內描述；維持舊資料的驗證和修復。
若需要新格式，再加入獨立的 `packageID`、contract version/digest，
並讓 repack plan 和中斷續傳紀錄使用同一身分。
Swift 與 Python 都能讀取後，才啟用新 writer；不得因開啟模型就自動改寫 manifest。

## 貢獻者實際要做什麼

| 新增內容 | 需要提供 | 共用部分 |
|---|---|---|
| 同架構、同格式的新 checkpoint | 描述資料、固定 revision 驗證、tensor mapping fixture | 運算、SSD、UI、server |
| 使用現有儲存格式的新架構 | 上述內容、fork 中的模型實作、runtime adapter；必要時新增對話 codec | 安裝執行器、SSD 讀取與 slot、請求管理、API |
| 新量化格式或新儲存需求 | 另補 expert/row codec 或運算 primitive，及對照測試 | 仍沿用已適用的部分 |

若 tensor 命名或轉換已超出既有規則，即使是同架構也需要安裝 adapter；
不保證每個 checkpoint 都只加一份 JSON。新架構只需寫新增的數學，
不必重寫下載、記憶體管理、聊天 server 和 App。

## 分階段 TODO 與驗收

以下保留原始分階段提案。實際先完成不依賴 fork 發布的共用描述、
套件入口、expert layout 與第四模型接入檢查；fork 發布、建構注入和完整模型
量測尚未完成。驗證邊界以目前 docs 為準。

| 階段 | 交付內容 | 驗收與停止條件 |
|---|---|---|
| P0：fork 相容基礎 | 自有 fork、本機相容 patch、來源及測試紀錄 | 已完成本機準備；尚未提交/推送、未切 dependency |
| P1：固定基準與切換來源 | 已發布的自有 commit、重現環境、dependency pin、包裝檢查 | 現有測試及小模型對照通過；完整模型行為另驗證，不以單元測試替代 |
| P2：共用模型描述 | 三份描述資料、Swift/Python 同一資料來源、舊 manifest 正規化 | 三種 catalog、API ID、設定預設、安装/repack/repair fixtures 不變；不改 writer |
| P3：模型入口和狀態 | 逐一遷移 V4、Qwen、V4.1 adapter，移除共用 generation 的模型判斷 | Prefill/decode logits、cache、取消後下一請求、關閉資源一致；既有可選功能不退步 |
| P4：共用儲存與 fork 注入 | expert codec、row lookup、明確建構入口，逐一搬移模型運算 | 小尺寸 resident/SSD 對照通過，沒有全域類別替換或過早覆寫 slot |
| P5：貢獻路徑驗證 | 接入範例、測試工具、貢獻文件 | 一個同架構新描述及第四個小型架構案例只修改套件/註冊資料，不改 server/UI/核心生成 |

P1 的來源切換不得指向本機絕對路徑或可變分支。
相容 patch 先保持現有實作不變，上游版本升級另作獨立變更。
搬移 Qwen 與 V4.1 時保留來源和授權；V4.1 原始來源及 Apache-2.0
紀錄見 [UPSTREAM.md](../runtime/deepseek_v4_ssd/deepseek_v41/UPSTREAM.md)。

必要檢查依次為：描述資料與安裝 fixtures、三種模型的錯誤拒絕案例、
小尺寸 logits/state 對照、codec 串流對照、取消/重用/unload、完整 pinned checkpoint。
小模型採用固定 seed、輸入、量化與運算順序；既有實作純搬移要求相同 logits/state。
若需改變運算順序，另訂數值容差並獨立驗證，不把差異藏在介面遷移中。

正式效能比較前固定 commit、機器、checkpoint、slots、功能開關、prompt、
輸出長度與冷暖 cache 條件，採交錯配對並記錄 output token hash。
提議重構保護線：TTFT/decode 中位數退步不超過 3%，峰值記憶體增加不超過
max(128 MiB, 2%)；這些是待接受的驗收門檻，沒有已達標的宣稱。
遇到持續 swap 或測量波動大到無法辨別 3% 時，停止並標示未定。
原始測試放 `scratch/`，正式可重現結果再放 `docs/benchmarks/`。

若新案例仍迫使共用核心加入模型名稱判斷，表示介面位置不對，停止擴大遷移。
若描述資料開始承擔任意運算邏輯，改由模型程式表達，不增加 JSON 指令。
每階段完成後才更新適用的 `docs/`；尚未完成的部分保留在本研究文件。

## 外部做法與證據範圍

- 官方 mlx-lm 以 `model_type` 找到 `Model`/`ModelArgs`，並提供模型載入、
  `sanitize` 和 cache 建立等慣例。可以延伸既有入口，無須自行重寫整套模型框架。
  來源：[官方 utils.py（固定版本）](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/utils.py)。
- vLLM 將模型註冊與模型實作分開，提供外部註冊途徑；它支持「明確模型入口」這個方向，
  但不表示其 GPU 執行框架適合直接移植到 Whallm。
  來源：[Model Registration](https://docs.vllm.ai/en/latest/contributing/model/registration/)。
- llama.cpp 新模型流程仍要處理 checkpoint 轉換與模型圖，說明共用底層可以減少
  重複工程，但新的架構不會因此省去運算實作。
  來源：[HOWTO-add-model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)。

上述來源用於架構比較；本文件的介面與分階段安排是 Whallm 的設計提案。
初期研究只執行 fork 相容測試。之後採用選項 2，已接入模型支援套件並完成小型模型測試；
現行實作與驗證以 [模型支援套件文件](../docs/MODEL_PACKAGES.md) 為準。
完整模型與效能實驗尚未執行。
