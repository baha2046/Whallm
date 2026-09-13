# 驗證紀錄

## 2026-09-14：共用 Prefill 中途取消

- 來源為 `6821af3` 加目前工作樹修改。原先 V4 逐層 Prefill 未檢查取消旗標；
  小型重現涵蓋 attention／MoE／fallback 及兩種分批長度，六個案例皆因
  `GenerationCancelled not raised` 失敗。加入分批／層間檢查後通過。
- V4.1 增加每層取消檢查；共用 `wait_for_futures` 在等待 expert 讀取時檢查取消，
  取消尚未開始的工作並等待正在寫入的工作完成。未完成請求清除未使用的
  整層預讀，再等待 GPU 工作完成；不因取消而提前重用仍在寫入的 slots。
- 新增五項測試，涵蓋 V4 中途停止後恢復、V4.1 層間停止後恢復、讀取取消與收尾、
  slots 回收及不暴露未完成的 batched weights。一般 server 的斷線測試改用真正的
  V4 Prefill 迴圈搭配小型替身，確認只執行第一批、不產生第一個 token，下一筆成功。
- `PYTHONPATH=runtime:runtime/tests .venv/bin/python -m unittest discover -s runtime/tests -q`：
  **427 項通過**。包含 Qwen 既有取消、Prompt Cache、數值一致性、一般 HTTP request
  與 Throughput 停止後卸載的測試。修改檔案的 `git diff --check` 通過。
- 僅執行小型模型／buffer 與測試 server，未載入完整 checkpoint，未量測完整模型
  取消延遲，未重新打包或發布。50 ms 是讀取等待的旗標檢查間隔，並非停止時間保證。

## 2026-09-14：Qwen3.8 Slots 預設 3072

- 共用模型描述的 Slots 預設由 2048 改為 **3072**，Advanced Settings 三語建議值
  讀取相同預設；已儲存的自訂值保留，DeepSeek V4／V4.1 維持 1152。
- Swift 兩項設定測試通過，涵蓋預設、三語文案、設定保存與舊設定遷移；
  Python 模型支援七項測試、三語 strings 格式與修改空白檢查均通過。
- logs：`scratch/qwen-slots-3072-2026-09-14/`。本次尚未重新打包或發布，
  下方 local build 紀錄與現有 `dist` 成品的 Qwen 預設仍為 2048。

## 2026-09-14：最新修正 build local

- 來源為 `develop` 的 `6821af3` 加當前未提交修改；包含 V4／Qwen 長輸入前釋放舊 Slots、
  Throughput 結束後卸載、V4.1 Prompt Cache、Qwen Slots 2048 與三語建議值。
- Swift **89 項通過**，排除既有互動式 Keychain round-trip 案例。
  Python **422 項通過**；首次執行漏設 `PYTHONPATH=runtime`，兩個模組無法載入，
  補正環境後完整重跑通過，兩次 logs 均保留。
- 執行 `make package APP_VERSION=1.1.7 BUILD_VERSION=1.1.7d2 CODE_SIGN_IDENTITY=- NOTARY_PROFILE=`，
  建立 `dist/Whallm.app` 與 `dist/Whallm-macOS-arm64.zip`。版本沿用現有值，
  使用 ad hoc 簽章，未建立 tag、公證或上傳；此成品與公開 Alpha 內容不同。
- App 與 ZIP 解壓副本均通過 `codesign --verify --deep --strict`。
  en／zh-Hans／zh-Hant 各自禁止專案 `.build` 與 Swift module bundle 存取時，
  仍從 `Contents/Resources` 完成 L10n 初始化並存活；六次啟動均確認 local 功能旗標為 1。
- 包內 Python 在禁止專案 `.build`、`.venv`、`runtime`、`Sources` 的沙盒下，
  **26 項測試通過**：Slots 釋放 6、V4.1 Prompt Cache 4、Throughput 9、模型支援 7。
  包內五個修改後的 runtime 檔案與原始碼逐位元一致；模型描述確認 Qwen Slots 2048、
  所有模型 Max tokens 8192、V4.1 Prompt Cache 選項已包含。
- ZIP：186331188 bytes；SHA-256：
  `5e2ff10e99edb5816ed9d0cf02c56145dc2fabf77275de0c83df4e9b5e1bd63b`。
  logs、驗證腳本與 runtime hashes：`scratch/local-build-2026-09-14/`。
- 未另外執行完整模型重現或新增效能測量；下方較早紀錄中的「尚未打包」指當時狀態。

## 2026-09-14：長輸入前釋放 Slots 與三模型審查

- 使用者採用方案 1 後，原始碼在 V4 batched layer-major、Qwen layer-major 入口
  先釋放舊 expert Slots，再建立輸入暫存。正式 server、CLI、Throughput 共用此修正。
- 共用清理等待 GPU 與中斷請求殘留的 expert 預讀；保留 direct／staged／Qwen grouped／
  bounded 儲存類型、模型、Prompt Cache、設定、檔案描述符及累計指標。
  拒絕清除使用中或 speculative prefetch 保護中的 buffers。
- V4.1 使用既有 Slots 做分批計算，程式未建立同類整層 expert 預載，沒有加入無效清理。
- 六項新增回歸測試檢查實際 buffers 釋放、原 pool 釋放、重填與命中、所有儲存類型、
  使用中保護、殘留預讀等待、embedding 前清理、非 batched／空輸入保留與 Qwen 入口。
  修改前針對性案例失敗，修改後通過；完整 Python **422 項通過**。
- 依使用者要求，未另外執行完整模型重現。上輪 V4 原型的 32.56 → 21.20 GiB 是
  採用前的獨立方案評估，詳見 [原型數據](benchmarks/2026-09-14-throughput-memory-options/README.md)。
  不是本次 Qwen／V4.1 或最終程式的完整模型效能驗收。
- logs：`scratch/prefill-slot-release-2026-09-14/`。本次未改 Swift；沿用上一輪 89 項檢查。
  尚未打包或發布。

## 2026-09-14：DeepSeek V4 Throughput 記憶體排查

- M5 Pro 64 GiB、1024 slots、Code、每筆實際生成 128 tokens、Prompt Cache 關閉。
  五筆完整模型推論重現 Peak MLX 差異：依序 1K／4K／8K 為
  **21.1753／32.5608／32.9677 GiB**；另一個新程序反向 4K／1K 為
  **21.2026／31.1977 GiB**。同長度交換順序前後的輸出 token hash 相同。
- 確認主因是 expert slots 按需配置並跨請求保留，1024 slots 共 12.75 GiB。
  第一筆 prefill 在 slots 尚空時執行；後續 prefill 同時持有已填滿 slots 與計算暫存。
  每筆結束 active 均約 21.15 GiB；不是 1K 增至 4K 自然需要額外約 11 GiB 的結論。
- 這是記憶體診斷；第二組有逐層觀測、未清 OS page cache，不作速度比較。
  完整條件、重現腳本、原始數值與 source／output hashes 見
  [紀錄](benchmarks/2026-09-14-throughput-memory/README.md)。未更動推論或每筆快取策略。

## 2026-09-14：Throughput 結束後卸載

- 整輪完成、取消或請求失敗後呼叫本次模型的卸載 API；使用 canonical model ID，
  等待取消的生成離開鎖後才關閉模型。各 context 長度之間不卸載。
- 清理不繼承取消狀態，清理期間禁止再次取消或 Run；卸載錯誤可見，已完成結果保留。
  server 啟動失敗、尚未發出請求與 Dry run 不卸載。
- Swift **89 項通過**，排除既有互動式 Keychain 案例。其中 Throughput **9 項**，
  覆蓋完成、取消、錯誤、清理失敗、清理期間重跑及 Dry run。
- Python Throughput **9 項通過**；HTTP 測試確認斷線取消後立即要求卸載，
  runtime 確實關閉、loaded model 清空、設定還原且生成狀態結束。
- logs：`scratch/throughput-unload-2026-09-14/`。
  尚未打包或發布，未以完整模型驗證 App 端取消與卸載。

## 2026-09-14：DeepSeek V4.1 Prompt Cache

- 原始碼開放 V4.1 的 Off／Memory／Disk，新的 Advanced Settings 預設 Memory，
  已保存的模式保留；CLI／Server 沿用既有快取參數。尚未打包或發布。
- V4.1 套件新增完整狀態保存與還原，包含 window KV、compressed KV、index keys、
  compressor partial group、offset、capacity 與 Engram token history；還原檢查格式、
  位置、形狀及型別，分支互相獨立，記憶體計量包含 CPU Engram 歷史。
- 四項新增測試使用非零隨機權重的四層小型 V4.1 真實運算，涵蓋共享注意力、
  ratio 1／2、partial group、ring wrap、擴容、磁碟保存與重開、取消後重試及無效狀態。
  clone／磁碟還原後的後續 logits 完全一致；runtime 開關快取生成的 token 完全一致，
  Memory／Disk 第二次請求皆確認實際重用了前文，Memory 不建立磁碟目錄。
  刻意清掉 Engram history 或 compressor partial state 會改變輸出，證明案例有測到這些狀態。
- Python 完整 **416 項通過**；Swift **85 項通過**，排除互動式 Keychain round-trip 案例。
  Swift 覆蓋 V4.1 模式保存、舊資料遷移與 server catalog 傳遞。
  完整測試後另補強 Memory 不落盤檢查，四項針對性測試再次通過。
- logs：`scratch/v41-prompt-cache-2026-09-14/`。未驗證完整 V4.1 checkpoint 的生成、
  長 context、效能或記憶體用量；上述是小型模型的功能驗證。

## 2026-09-14：Qwen3.8 Slots 預設 2048

- 共用模型描述的 Qwen Slots 預設改為 **2048**，Advanced Settings 的三語建議值
  讀取相同預設；已儲存的自訂 Slots 保留。DeepSeek 預設維持 1152。
- 兩項 Swift 設定測試通過，涵蓋預設、三語建議、設定保存及舊設定遷移；
  Python model-support 七項測試、三語 strings 格式及 diff 空白檢查通過。
  文案測試按數字核對，接受系統加入千分位分隔符號。
- 本次僅修改原始碼與文件，尚未打包或發布；既有 Alpha 成品預設仍為 4096。

## 2026-09-14：Whallm 1.1.7-dev.2 Alpha 發布

- [GitHub Release](https://github.com/yanun0323/Whallm/releases/tag/v1.1.7-dev.2)：
  `Whallm 1.1.7-dev.2 (Alpha)`，pre-release，非 latest。
  tag 指向 `f4eb369f6ea612d01a1cb01263f0ad87da70443d`；App 版本 `1.1.7`／`1.1.7d2`。
- `make release VERSION=1.1.7 CHANNEL=dev DEV_BUILD=2` 完成；使用 Developer ID
  `Yanun Yang (Y366CJ66L6)` 與 `deepseek_ssd` 公證 profile。
  Apple submission `94717c72-0ecd-4cee-9119-35e5a3509c78` 為 Accepted。
- 發布前的 App／ZIP 解壓副本與發布後從 GitHub 下載的 ZIP 均通過 strict code signature、
  stapled notarization、Gatekeeper，以及 en／zh-Hans／zh-Hant 的隔離啟動。
  拒絕 `.build` 與 Swift module bundle 存取時仍完成 L10n 初始化並存活；
  每次啟動皆確認 `WHALLM_LOCAL_BUILD:0`，發布包排除 Dry run。
- 包內 Python 在禁止專案 `.build`、`.venv`、`runtime`、`Sources` 的沙盒下，
  **9 項 Throughput 測試通過**。完整開發測試結果見下方發布前檢查。
- Release 含 ZIP 與 `appcast.xml`；英文說明與原稿一致，Sparkle feed 簽章通過。
  feed 內 dev 為 `1.1.7d2`，stable 保留 `1.1.6`；`gh-pages` 提交 `0be6544`。
  公開 Pages 清單與 Release 附件 SHA-256 一致；GitHub latest 仍為 `v1.1.6`。
  已開啟最終 Alpha App，實際檢查 Throughput 的 Model 選單沒有 Dry run。
- GitHub 下載 ZIP SHA-256：
  `4942481fd460213748ed7279ee5b3b7b53d30178eee86242ae146d1ce51312a0`。
  GitHub 下載 appcast SHA-256：
  `dda5b35fbe03d6a06f82ce87e852f441e1c9a1692f132cc369910a510361f59e`。
- 原始 logs：`scratch/release-1.1.7-dev.2/`。本輪未執行 Sparkle 原地安裝，
  未重新測量真實模型吞吐量，200K 素材覆蓋不代表已完成 200K 模型生成測試。

## 2026-09-14：1.1.7-dev.2 Alpha 發布前檢查

- 版本為 `1.1.7`／`1.1.7d2`，tag `v1.1.7-dev.2`，GitHub pre-release，更新來源 dev。
- 重新執行 Python **412 項通過**、local Swift **82 項通過、3 項 fixture 略過**，
  排除需要互動的 Keychain round-trip 案例。shell 腳本語法及 diff 空白檢查通過。
- 本機封裝前置條件已由下方 local-only Dry run 紀錄完成；發布改用 distribution，
  需在上傳前及下載後再次檢查公證、三語隔離啟動及 local 功能關閉。
- Developer ID 與 `deepseek_ssd` 公證 profile 已於本輪確認可用。
  原始 logs：`scratch/release-1.1.7-dev.2/`。本節是發布前檢查，公開成品結果另記。

## 2026-09-14：local-only Dry run 與 Max tokens 8192

- Dry run 以 `WHALLM_LOCAL_BUILD` 條件編譯；`make build/run/test/package` 啟用，
  stable／dev 共用的 release 腳本強制 distribution，不編入模擬生成、不顯示選項。
  發布組態的最佳化 binary 已編譯，啟動標記為 0，實際 Model 選單無 Dry run。
- App 各模型 Advanced Settings 的 Max tokens 預設均為 **8192**，描述資料及設定
  初始值一致。程式載入仍保留已儲存設定；依本次要求，也透過 UI 將這台電腦的
  DeepSeek V4／V4.1／Qwen 現有 Max tokens 設為 8192，重新讀取偏好設定確認三者皆相符。
- local 組態 Swift **82 項通過、3 項 fixture 略過**，另排除既有 Keychain 互動案例。
  distribution 組態 **6 項針對性測試通過**，涵蓋禁止模擬生成及所有模型的新預設值；
  Python model-support **7 項通過**。本輪未更動 Python runtime，未重跑其完整 suite。
- `make package` 完成，沿用 1.1.7／1.1.7d1、ad hoc 簽章。App／ZIP 解壓副本通過
  嚴格簽章、模型描述及六次三語隔離啟動，並從 binary 標記確認 local 功能為 1。
  `.build` 與 module resource bundle 被拒絕時，L10n 仍從 `Contents/Resources` 初始化。
- 故意以 `EXPECTED_LOCAL_BUILD=0` 驗證 local 成品，確認因組態不符而拒絕；發布前及
  下載後均加入此檢查。僅驗證組態和阻擋機制，未執行公證、tag 或公開發布。
- source、App、ZIP 三處描述資料及目前偏好設定的每個 Max tokens 皆為 8192。
  已重新開啟成品。logs：`scratch/local-build-gate-2026-09-14/`。
  ZIP SHA-256：`31a88634dc1b08bb7b322525f65534e98e21a7b4e25251561ec115ec86050682`。

## 2026-09-14：Throughput Dry run build local

- Model 新增 Dry run；選取立即依目前選項產生模擬结果，沒有 installed model 也可用。
  可按「產生結果」重新生成，畫面與三種輸出皆標示 dry-run；模擬 slots 固定 2304。
  不啟動 server、不載入模型，不屬於效能測量。
- Swift **81 項通過、3 項 fixture 略過**，另排除既有 Keychain 互動案例。
  新增測試確認空 catalog 下不啟動 server、選項正確、結果固定、三種輸出保留 dry-run
  標記及空長度處理。既有 Chat 定時等待案例一輪失敗後，完整重跑通過；未修改該案例。
  本輪未修改 Python，未重跑其 suite。
- 無模型的開發 App 與最終封裝 App 均已確認選取 Dry run 後立即出現三筆結果。
- `make package` 完成，沿用 1.1.7／1.1.7d1、ad hoc 簽章；App／ZIP 解壓副本通過
  嚴格簽章、模型描述與六次三語隔離啟動。禁止 `.build` 及 module resource bundle 時，
  L10n 仍從 `Contents/Resources` 初始化並存活。已重新開啟 App，未公證或發布。
- 原始 logs：`scratch/throughput-dry-run-2026-09-14/`。
  ZIP SHA-256：`9a71ea3e4aa60387d052c2894a50921b51d2dd27b832412f0a268114536bb40e`。

## 2026-09-14：Throughput 結果排版與文字輸出 build local

- 結果表格欄位填滿卡片寬度，窄視窗保留水平捲動。標題加入 runtime 回傳的當次 slots，
  不讀取後續設定值。每筆結果均保存自己的 slots；舊 server 未提供時顯示破折號。
- 新增純文字／JSON／Markdown table 輸出，使用可選取、不可編輯的原生文字區域。
  純文字以空格對齊欄位；JSON 保留原始數值、素材與輸出 hash 及 finish reason。
- 使用明確標示 UI preview 的三筆合成結果檢查滿寬表格、slots 標題、格式切換、
  原生文字全選與長行捲動。預覽資料只存在 scratch App；已確認正式 binary 不含 fixture。
  本輪未重新測量模型吞吐量，也未完成所有窄視窗與語言的視覺檢查。
- Swift **80 項通過、3 項 fixture 略過**，另排除既有 Keychain 互動案例。
  一輪完整測試中既有 Chat 導航案例在定時等待後未取得預期狀態；未修改該案例，
  重跑完整 suite 通過。Python **412 項通過**，含非預設 slots 回傳與設定恢復檢查。
- `make package` 完成，版本沿用 1.1.7／1.1.7d1、ad hoc 簽章。App 與 ZIP 解壓副本
  的嚴格簽章、模型描述檢查及六次三語隔離啟動均通過；禁止讀取 `.build` 與 module
  resource bundle 時，L10n 從 `Contents/Resources` 初始化並持續存活。
- 使用 App 內建 Python，禁止讀取專案 `.build`、`.venv`、`runtime`、`Sources`，
  **9 項 Throughput 測試通過**。封裝 runtime 與 source 相符。已開啟最終 App；未公證或發布。
- 原始 logs：`scratch/throughput-output-2026-09-14/`。
  ZIP SHA-256：`223f4725becf21a0d1840d77dea67cc9005c6e171165a8bb41240388be3aa812`。

## 2026-09-13：Code／Novel 素材與 build local

- Benchmark Context 改為 Code／Novel 下拉選單，選單、模型、生成選項及 Run 的
  可見右緣保持對齊。已在開發 App 與最終 `dist/Whallm.app` 切換 Novel 並檢查畫面。
- 移除 9 KB 重複片段，改用 Whallm 的 45 個原始碼檔案固定副本與《白鯨記》英文原文。
  gzip 合計 **697665 bytes**，另附來源、授權與 hash。三種模型 tokenizer 均可直接
  截取 204800 tokens；素材不足會報錯。完整來源比較見 [Throughput](THROUGHPUT.md)。
- Swift **79 項通過、3 項 fixture 略過**，另排除既有 Keychain 互動案例；
  Python **412 項通過**。涵蓋兩種素材的原文前綴、短素材拒絕、API 選擇與素材 hash、
  非法選項在載入前拒絕，以及既有取消／錯誤後設定恢復。
- `make package` 完成，版本 1.1.7／1.1.7d1、ad hoc 簽章。
  App 及 ZIP 解壓副本通過嚴格簽章、模型描述檢查與六次三語隔離啟動；禁止讀取
  `.build` 及 module resource bundle 時，三語皆從 `Contents/Resources` 初始化並存活。
- App／ZIP 中兩份 gzip、來源 manifest 及兩份授權檔與 workspace 逐 byte 相符。
  以 App 內建 Python 執行 **9 項 Throughput 測試通過**，禁止讀取專案 `.build`、
  `.venv`、`runtime` 與 `Sources`，確認素材與功能不依賴開發目錄。
- 已重新開啟最終 App。未公證、發布或進行 200K 模型生成；token 計數不代表測速結果。
  [容量與輸入 hash 紀錄](benchmarks/2026-09-13-throughput-contexts/coverage.json)；
  原始檢查 log：`scratch/throughput-corpora-2026-09-13/`。
- ZIP SHA-256：`ff12c73dcd82987c39de517543cfcb12cc210cc631243a70d75783cc7008e1b4`。

## 2026-09-13：Throughput 間距與右側對齊 build local

- 依 oMLX 參考圖收緊 Throughput 版面：文字間距 2 pt、列內距 12 pt、
  區塊左右內距 16 pt、內容寬度上限 760 pt；輸入選項靠左、間距 6 pt。
  素材說明移至左側標題下，Run／Cancel 移入設定區塊底部。
- 模型選單、Whallm Code、生成選項、Run／Cancel 的可見右緣對齊。
  已檢查英文、繁體中文畫面及約 800 pt 寬的英文視窗；未完成所有窄視窗、
  文字放大或其他語言的視覺檢查。測試流程與 runtime 未改動。
- Swift **79 項通過、3 項 fixture 略過**，另排除既有 Keychain 互動案例；
  Python **410 項通過**。
- `make package` 完成，版本沿用 1.1.7／1.1.7d1、ad hoc 簽章。
  App 與 ZIP 解壓副本的嚴格簽章、模型描述及六次三語隔離啟動全部通過；
  禁止讀取 `.build` 及 module resource bundle 時仍能完成 L10n 初始化並持續存活。
- 已重新開啟 `dist/Whallm.app`，確認最終 Throughput 排版與右緣一致。
  未公證、發布或重新執行完整模型測速。
- 原始 log 與 ZIP SHA-256：`scratch/throughput-spacing-build-2026-09-13/`。

## 2026-09-13：Throughput build local

- 使用 `make package APP_VERSION=1.1.7 BUILD_VERSION=1.1.7d1 CODE_SIGN_IDENTITY=- NOTARY_PROFILE=`
  完成本機封裝；`dist/Whallm.app` 與 `dist/Whallm-macOS-arm64.zip` 已包含 Throughput。
  版本號沿用 1.1.7／1.1.7d1，本次為 ad hoc 簽章，未公證或發布。
- 封裝前 Python **410 項通過**；Swift **79 項通過、3 項 fixture 略過**，
  排除 `ServerConfigurationTests.testAPIKeyRoundTripsThroughIsolatedKeychainItem`。
- App 與 ZIP 的臨時解壓副本均通過 `codesign --verify --deep --strict`。
- 兩個副本均禁止讀取專案 `.build` 及 Swift module resource bundle，
  分別啟動英文、簡體中文、繁體中文；六次均完成 L10n 與模型描述初始化，
  並持續存活至檢查結束。三語檔案皆位於 `Contents/Resources`。
- 模型描述的包內 Python 檢查也禁止讀取 `.build`、`.venv` 與 `Sources`，均通過。
- 內建 Throughput 素材為 **9217 bytes**，與原始碼中的素材一致；使用 App 內建 Python，
  禁止讀取專案 `.build`、`.venv`、`runtime`、`Sources` 後，**七項 Throughput 測試通過**。
  測試後再次確認 App 簽章有效。
- 已正常開啟 `dist/Whallm.app`，確認 Throughput 頁面及全部選項可見；server 保持停止。
  本次未重新執行完整模型測速，前一節的開發版實測範圍仍適用。
- 原始 log 及 ZIP SHA-256：`scratch/throughput-build-local-2026-09-13/`。

## 2026-09-13：Throughput 頁面

基準為 `develop`、`0204409` 加本次未提交修改。功能與數據定義見
[Throughput](THROUGHPUT.md)。

- `swift test --skip ServerConfigurationTests.testAPIKeyRoundTripsThroughIsolatedKeychainItem`：
  **79 項通過、3 項缺少 fixture 略過**，另排除一項既有 Keychain 互動測試。
  新增三項涵蓋結果解析、串流進度／錯誤及三語文字查找。
- Python 全套 **410 項通過**。新增七項涵蓋固定素材大小與八種精確輸入長度、
  模型自動載入、實際輸出計數、設定恢復、長度與授權檢查、生成／素材錯誤、
  checkpoint context limit 和 Prefill 期間斷線取消。
- 三語 `Localizable.strings` 的 `plutil -lint` 及 `git diff --check` 通過。
- 使用由 debug binary 建立的開發 App 副本，確認 Throughput 頁面的模型選單、
  輸入多選、生成長度切換與執行中鎖定。這個副本僅用於開發畫面檢查，
  不是 `build local` 的封裝成品。
- 在本機 64 GB 主機、外接磁碟 installed model 上，透過 App 的 Run 完成
  **Qwen 1K 輸入／128 輸出**；從停止的 server 自動啟動、載入模型、顯示進度，
  最後表格顯示實際 1024／128 及本次各欄數據。
- 另一筆 1K／1024 在 Prefill 期間按 Cancel，畫面顯示取消，收尾後 Run 恢復可用。
  最後關閉開發副本，確認其 App 與 server 程序均已結束。
- 實機檢查使用預設模型設定，目的是確認操作與生成可完成，**不是正式效能比較**。
  尚未逐一實跑其他模型、全部長度及生成上限，也未完成正式 benchmark artifact。
  完整流程實測後另將進度更新改為最多每 0.25 秒一次，並補上素材讀取錯誤回報；
  最終程式已重新通過上述測試，未再重跑完整模型。
- 測試 log 保存在 `scratch/throughput-validation-2026-09-13/`。
  本次未重新封裝 `dist`、公證或發布。

## 2026-09-13：Whallm 1.1.7-dev 預發布

已發布 [v1.1.7-dev](https://github.com/yanun0323/Whallm/releases/tag/v1.1.7-dev)，
Release 名稱 `Whallm 1.1.7-dev`，標記為 pre-release；最新正式版仍為 `v1.1.6`。
Tag 對應 `32de5c8c33d76b36f4c2a0be7d2e9120e576a63d`；App 版本 `1.1.7`、
內部 build `1.1.7d1`，Sparkle channel `dev`。

- Python 403 項、Swift 77 項通過；Swift 3 項缺少 installed-model fixture 略過。
  三種模型描述檢查及 CLI 四項離線安裝分派檢查通過。
- `build local` 先以 ad-hoc 簽章完成，App／ZIP 的簽章、三語系隔離啟動及包內
  模型描述檢查通過；再以 Developer ID 簽章並透過 `deepseek_ssd` 完成 Apple 公證。
- 公證 submission `c3749b71-661f-4808-855f-ab410d0b7692` 為 Accepted。
  上傳前及 GitHub 重新下載後，App／ZIP 解壓副本均通過嚴格簽章、公證票據、
  Gatekeeper、三語系檔案與禁止讀取專案 `.build`／Swift resource bundle 的啟動檢查。
- Release commit、乾淨工作目錄、本機 tag 與遠端 tag 在打包前和上傳前核對一致；
  GitHub Release 使用 `--verify-tag --prerelease --latest=false` 建立。
- GitHub Release body、Sparkle 內嵌 Markdown 與
  `Packaging/ReleaseNotes/1.1.7.md` 英文內容相同。
- `gh-pages` commit `d34e08a` 已部署，公開 appcast 與 Release 附件逐位元相同且
  Sparkle 簽章有效；保留 stable 1.1.6 與 dev 1.1.7d1。
- 使用獨立 bundle ID、版本設為 1.1.6 的 App 副本實測：stable 顯示已是最新版，
  dev 找到 1.1.7 更新並顯示完整英文 release notes；未執行 Sparkle 原地安裝。
  未修改已安裝 App 的來源選擇，驗證副本已關閉。
- 未重跑完整模型生成或效能量測；V4.1 完整安裝與生成仍未驗證。

SHA-256：

- `Whallm-macOS-arm64.zip`：`78a0f7fbca928e3dd4c8111705c612f55479fd8bfafc3ca66200637e4610bd6c`
- `appcast.xml`：`ddb9a03cdbdea05dead56a9bc7da1fb0a0786df7f7075e6715123384f994e72a`

原始測試、公證／發布記錄、下載副本與核對摘要位於 `scratch/release-1.1.7-dev/`。

## 2026-09-13：Updates 區塊與靠右對齊

- Settings 將更新設定移至獨立 Updates 區塊；stable／dev 的固定寬度容器改為靠右對齊。
- 使用獨立 App 副本查看實際畫面，確認分組及來源選項與按鈕、開關右緣對齊。
- Swift 77 項通過、3 項缺少 fixture 略過；`git diff --check` 通過。
- 重新執行本機 `make package`，App 與 ZIP 解壓副本的簽章、三語隔離啟動及
  包內模型描述驗證皆通過。未公證或發布新版。
- 原始記錄位於 `scratch/update-layout-2026-09-13/`。

## 2026-09-13：更新來源與自動檢查

基準為 `codex/pr10-fixes`、`5565c23` 加目前未提交修改。

- Settings 新增 stable／dev 來源與自動檢查開關；App 選單與設定頁共用同一個
  Sparkle updater。Swift 77 項通過、3 項缺少 fixture 略過，包含四項新增更新設定測試。
- 實際使用獨立 bundle ID 的封裝副本，確認 stable／dev 及自動檢查開／關皆可操作，
  偏好確實保存；關閉自動檢查後仍能手動檢查。對公開 Pages 清單檢查得到
  「1.1.6 已是最新版本」，沒有下載或安裝其他 App。驗證副本已關閉。
- 真正執行 Sparkle `generate_appcast --channel dev --maximum-versions 1`，
  確認保留既有正式版並加入 dev，重新生成清單的簽章驗證通過。
  此測試使用本機 ZIP 與 `example.invalid` 下載網址，未發布測試清單。
- 發布腳本離線替身檢查涵蓋 stable、dev、打包失敗停止發布，以及下載驗證完成後
  才推送 Pages；不代表實際完成一次正式或 dev 發布。錯誤 channel、超出範圍的
  DEV_BUILD、不相符的 dev BUILD_VERSION 都會提早拒絕。zsh 語法與 make dry run 通過。
- `gh-pages` 已建立（`d54e64b`），Pages 根目錄已部署；從公開網址下載的清單與
  已發布 v1.1.6 清單逐位元相同，Sparkle 簽章驗證通過。SHA-256：
  `5d08e384ebf2507f9f7c4143969c52bcff7882e0d3663c2d2e0d34072730cc3c`。
- 本機 `make package` 成功；App 與 ZIP 解壓副本的嚴格簽章、三語系隔離啟動、
  包內模型描述驗證全部通過。只使用 ad-hoc 簽章，未發布新版 App。
- 沒有已發布 dev App，因此尚未實測 dev ZIP 的完整下載／安裝。

流程與參數見 [App 更新](UPDATES.md)。原始測試、封裝與清單記錄位於
`scratch/update-channels-2026-09-13/`。

## 2026-09-13：V4.1 slots 1152 與本機封裝

基準為 `codex/pr10-fixes`、`5565c23` 加目前未提交修改。

- V4.1 Advanced Settings 的 slots 預設值改為 1152；三語建議文案讀取同一份
  模型描述。已確認包內英文、簡中、繁中文案皆為 1152，既有個人設定保留。
- `make test`：73 項通過、3 項缺少 fixture 略過；Python 完整 suite 403 項通過。
- `env -u NOTARY_PROFILE CODE_SIGN_IDENTITY=- make package` 成功，產生
  `dist/Whallm.app` 與 `dist/Whallm-macOS-arm64.zip`。
- App 與 ZIP 解壓副本均通過 `codesign --verify --deep --strict`，並在禁止讀取
  專案 `.build` 與 Swift resource bundle 時完成三語 L10n 初始化且保持運行。
  語系檔由 `Contents/Resources` 載入，優先於 `Bundle.module`。
- 兩份 App 的 Swift／Python 模型描述一致；包內 Python 在禁止讀取專案
  `.build`、`.venv`、`Sources` 時仍可載入三種模型支援套件。
- 使用本機 ad-hoc 簽章；未公證或發布，未執行完整模型生成或效能量測。

測試與封裝原始記錄：`scratch/build-local-v41-slots-1152-2026-09-13/`。

## 2026-09-13：build local

基準為 `codex/pr10-fixes`、`5565c23` 加目前未提交修改，包含最新模型進階設定。
同日再次封裝，已包含 V4.1 固定顯示大小 501,382,643,728 bytes；
重新執行下列完整測試與 App／ZIP 驗證，全部通過。

- Python 完整 suite 403 項通過；`swift test` 73 項通過、3 項缺少 fixture 略過。
- `env -u NOTARY_PROFILE CODE_SIGN_IDENTITY=- make package` 成功，產生
  `dist/Whallm.app` 與 `dist/Whallm-macOS-arm64.zip`，使用本機 ad-hoc 簽章。
- App 與 ZIP 解壓版本均通過 `codesign --verify --deep --strict`。
- 兩份 App 均在禁止讀取專案 `.build` 與 Swift resource bundle 的環境下，
  分別以英文、簡中、繁中啟動並維持運行，輸出 L10n 初始化完成標記；
  確認直接使用 `Contents/Resources` 語系檔，沒有進入 `Bundle.module`。
- 兩份 App 的 Swift／Python 模型描述一致，包內 Python 在禁止讀取專案
  `.build`、`.venv`、`Sources` 時仍能載入三種模型支援套件。
- 未建立 tag、未公證、未上傳或發布；沒有重新進行完整模型生成或效能量測。

## 2026-09-13：模型進階設定

基準為 `codex/pr10-fixes`、`5565c23` 加目前未提交修改。

- Python 完整 suite：403 項通過。Swift `swift test`：73 項通過、3 項缺少 fixture 略過，無失敗。
- 自動測試覆蓋舊設定遷移、新欄位保存與 catalog 傳遞、輸入範圍驗證、
  Qwen 聊天／思考的自動與手動值、API 明確值優先、DeepSeek 預設 Exact 及 DSpark 限制。
- 使用獨立 bundle ID 的開發 App，確認 Qwen 自動取樣預設開啟、關閉後可編輯，
  再開啟會保留手動值；LRU／LFU 選單可切換。修改預讀工作數為 3、MoE 分批大小為 64，
  返回後重新開啟仍保留；DeepSeek 近似模式預設關閉且可切換。已查看實際畫面。
- 本次沒有載入完整模型做生成、品質或效能量測，也沒有重新封裝或發布。

## 2026-09-12：快取模式與設定介面

基準為 `codex/pr10-fixes`、`5565c23` 加目前未提交修改。這是功能驗證，
沒有重新量測完整 Qwen 模型的速度、記憶體峰值或 SSD 寫入量，也沒有重新封裝或發布。

- Python：`PYTHONPATH=runtime:. .venv/bin/python -m unittest discover -s runtime/tests -p 'test_*.py'`，
  **400 項通過**。
- Swift：`swift test --skip AppKeychainTests`，**72 項通過、3 項缺少 fixture 略過**。
  目前沒有名為 AppKeychainTests 的 suite，因此該 skip 沒有排除案例；
  `ServerConfigurationTests.testAPIKeyRoundTripsThroughIsolatedKeychainItem` 本次也通過。
- `git diff --check` 通過。
- 新的 runtime 測試逐一驗證 off／memory／disk：同程序重複輸入及重啟後的未快取
  token 數分別為 `[48,48,48]`、`[48,1,48]`、`[48,1,1]`。每次重用還原的是
  checkpoint 當下值 47，不是後續改寫的 999；off／memory 不建立快取目錄。
  這是小型可控模型替身，不是完整 Qwen 的新量測。
- catalog 測試覆蓋 `--no-persistent-prompt-cache`、目錄、筆數、slots、取樣參數與
  三種 mode 的明確覆寫；未指定的 4096 slots 保留。驗證依合併後的模型設定執行，
  能沿用 JSON 的 DSpark 開啟值，也會拒絕非法覆寫。
- Swift 測試覆蓋快取模式的舊偏好遷移、三種模式保存與 JSON 傳遞、V4.1 固定關閉、
  Qwen 合批新預設關閉，以及已儲存 true／false 的保留；三語 label 檢查通過。
- 使用獨立 bundle ID 的開發預覽，實際檢查 Log 頁面的 Debug／Info 切換、Server
  頁面移除 Log Level、Qwen 合批預設關閉、Memory 預設選取、Off 隱藏容量欄位，
  Disk 恢復容量欄位，最後切回 Memory。已查看實際畫面；沒有啟動模型生成服務。
  預覽使用獨立偏好，不修改已安裝 App 的設定，完成後已關閉。
  補充啟動失敗紀錄：最初的臨時預覽於 23:35:55、23:35:59、23:36:29 三次
  因找不到 `@rpath/Sparkle.framework/Versions/B/Sparkle` 而在進入 App 前中止
  （macOS report：DYLD / Library missing / SIGABRT）。原因是臨時 bundle 的
  framework 放置位置不在該 debug executable 的搜尋路徑內；後續以
  `DYLD_FRAMEWORK_PATH` 指向開發框架目錄才完成上述 UI 檢查。
  這項啟動方式不能視為可分發 App 的封裝驗證。

功能與入口的完整核對見 [命令列與 UI 功能對照](FEATURE_MATRIX.md)。
測試檔為 `runtime/tests/test_prompt_cache_modes.py` 與
`Tests/DeepSeekV4SSDAppTests/ServerConfigurationTests.swift`。

## 2026-09-12：模型支援套件

在 `codex/pr10-fixes`、基底 `5565c2396465724dc4372e91720956b89cd34200`
加本機未提交修改，完成 [模型支援套件](MODEL_PACKAGES.md) 的接入驗證。

- Python runtime **396 項通過**，涵蓋既有三個模型與新增套件測試。
- Swift **70 項通過、3 項缺少 installed model fixture 略過**；
  另排除 `testAPIKeyRoundTripsThroughIsolatedKeychainItem`，不列為通過。
- CLI 四項離線安裝計畫分派檢查通過。
- 第四個小型 MLX 模型只加入描述與套件註冊，沿用 manifest 結構版本 1，
  經 `ModelManager` 完成載入、生成、串流中斷、後續請求及卸載；資源僅關閉一次。
- 描述資料重複 ID、不安全路徑、未知實作、manifest 身分不符，以及 V4.1
  不支援的快取還原與近似模式均有拒絕檢查。V4.1 預設請求改用 exact。
- 設定 UI 改由模型描述控制預設值與可見選項；slots 提示使用模型預設值，
  修正 V4.1 顯示 1152 而實際預設 768 的差異。
- `make package` 產生本機 ad hoc 簽章的 1.1.6 測試成品，位於
  `scratch/model-packages-2026-09-12/package/`；既有 `dist/` 與已安裝 App 未替換。
  App 與 ZIP 解壓副本均通過 deep/strict 簽章、三語系資源與六次隔離啟動。
  Swift 在禁止讀取專案 `.build` 的條件下載入模型描述與主 App 語系資源；
  包內 Python 在禁止讀取專案 `.build`、`.venv`、`Sources` 的條件下確認
  三個套件完整註冊，兩份包內描述資料逐 byte 相同。未公證、建立 tag 或上傳。

原始記錄與來源 SHA-256 保留於 `scratch/model-packages-2026-09-12/`。
機器可讀摘要為該目錄的 `summary.json`，包含 ZIP SHA-256；包內 43 個 runtime
Python 檔案與本次工作目錄逐檔相同。
這是功能與打包驗證；沒有新的完整 checkpoint、輸出品質、速度或記憶體測量。
自有 mlx-lm fork 的 dependency 切換尚未完成，見 [fork 狀態](MLX_LM_FORK.md)。

## 2026-09-12：PR #10 修正版本 build local

在 `codex/pr10-fixes`（基底 `5565c2396465724dc4372e91720956b89cd34200`
加本機未提交修正）完成本機封裝，App 與 build version 沿用 `1.1.6`。
執行 `make package APP_VERSION=1.1.6 BUILD_VERSION=1.1.6 CODE_SIGN_IDENTITY=- NOTARY_PROFILE=`，
產生 `dist/Whallm.app` 和 `dist/Whallm-macOS-arm64.zip`。

- 封裝前 Python runtime 389 項通過；Swift 68 項通過、3 項因缺少 installed model
  fixture 略過，另排除 `testAPIKeyRoundTripsThroughIsolatedKeychainItem`。
- CLI 四項離線分派檢查通過。
- App 與 ZIP 暫存解壓副本均通過 `codesign --verify --deep --strict`。
- 兩份 App 的英文、簡中、繁中資源均存在且通過 plist 檢查。六次隔離啟動
  均禁止讀取專案 `.build` 與包內 Swift module bundle，完成實際 L10n 查找、
  輸出對應語言的 ready marker，持續存活三秒後由檢查程序結束。
  這確認各語言先從 `Contents/Resources` 載入，不需存取 `Bundle.module`。
- 包內 Python 在禁止讀取專案 `.build` 與 `.venv` 的條件下，V4.1 10 項測試通過；
  測試後 App 簽章再次驗證通過。

ZIP SHA-256：`e2044db213402bf67d5a95716ce6a093b3b5df67e464a6d4072ee7b612b736a4`。
原始記錄保留於 `scratch/build-local-pr10-2026-09-12/`。
本次為本機 ad hoc 簽章成品，未建立 tag、公證或上傳，未替換已安裝 App。
完整 V4.1 checkpoint 安裝與模型生成仍未驗證。

## 2026-09-12：PR #10 本機修正與驗證

以 PR #10 的 `5565c2396465724dc4372e91720956b89cd34200` 為基礎，
本地分支 `codex/pr10-fixes` 修正以下兩項問題：

- CLI `repack --model deepseek-v4.1` 與讀取 V4.1 `--plan` 共用依
  `modelKind` 選擇 repacker 的流程，避免落入舊版 V4；舊 plan 未提供
  `modelKind` 時仍使用 V4。
- Swift 測試先將 `switch` 結果存入區域變數，再傳給 `modelID`，修正編譯錯誤。

本機 Apple Swift 6.3.3 驗證：

- `swift test --skip ServerConfigurationTests.testAPIKeyRoundTripsThroughIsolatedKeychainItem`：
  執行 71 項，68 項通過、3 項 fixture 測試略過、零失敗；另排除一項 Keychain
  互動測試。App 與 CLI targets 均在此次測試中完成編譯。
- `.venv/bin/python Scripts/check_repack_cli.py .build/debug/dsv4-repack`：
  舊格式 V4、明確指定 V4、V4.1、Qwen 共四項離線檢查通過。相同檢查對
  PR 原始 CLI 在 V4.1 案例失敗，確認能偵測本次修正的分派錯誤。
  檢查使用故意不相容的 plan，確認由正確模型的驗證器拒絕；不下載權重。
- 本次 review 在未修改的 PR revision 執行 Python runtime suite，389 項通過；
  後續修正未變動 Python runtime。

UI 程式已接入 V4.1 模型列、下載／修復、載入／卸載及進階設定。
此次只確認程式接線、編譯及上述測試，未做完整模型安裝、生成或 UI 操作驗證，
也未重新封裝 App。

## 2026-09-11：DeepSeek V4.1 text-only working tree

目前 working tree 新增固定 `deepseek-ai/DeepSeek-V4.1-Flash` revision
`dba1be0a40aa45a94ad051997016db3960a90277` 的 text-only 支援。
MLX 架構來源是同日公開、Apache-2.0 的 `deepseek_v41` port；Whallm vendored
版本保留第三方授權，並將 routed experts 與兩個 Engram table 改為 SSD lookup。

已通過：

- Python 全 runtime AST parse。
- App 內附 Python、MLX 0.32.2 與 mlx-lm 0.31.3 執行 V4.1 manifest、
  native FP8/MXFP8、strict quantized load、SSD Engram lookup、cache ceiling
  與單字元分片 DSML parser；相關 focused suite **37 項通過**。
- Kiro review 後新增 warmup、approximation、RoPE growth、2-D ready-expert、
  token-map immutability、strict shape validation 與 tiny adapter forward 回歸；
  focused regression subset **14 項通過**。
- 同一套 App 內附 Python 執行完整 runtime suite，**389 項通過**。
- `swift build --target DeepSeekRepack`。
- `swift build --target dsv4-repack`。

本機 command-line developer tools 缺少可解析的 `XCTest` 與 `SwiftUIMacros`
plugin，因此完整 Swift test 與 App target build 無法在此 host 完成；這是環境阻擋，
不是通過證據。所有 Swift source 已另以 compiler frontend 完成 syntax parse。
tiny synthetic V4.1 adapter 已走過 production loader 與真實 forward；尚未下載或
repack 完整 checkpoint，也尚未執行 full-model generation、記憶體或效能測量。
vision、MTP/DSpark、layer-major prefill、persistent prompt cache 與 prompt-cache reuse
不在目前 V4.1 支援範圍。

## 2026-09-08：Whallm 1.1.6 發布

本機 master 已 fast-forward 合併 `origin/codex/ssd-prefill-pipeline`，來源 commit
`8e69efb3e5f9ebb4bc31dd88c1c778ee5cf9a123`；使用分支既有的
[1.1.6 Release notes](../Packaging/ReleaseNotes/1.1.6.md)。

首次 Python 測試執行 358 項、有 8 個 import errors：分支忽略 research 目錄，
六個必要 Python 來源未追蹤。從原開發機同一 commit 的工作目錄
取回這六檔及既有 kernel／授權檔，八檔兩端 SHA-256 完全一致。
六個缺檔加入版本控制，既有兩檔內容不變；沒有改動原始內容。
補齊後 `PYTHONPATH=runtime:. .venv/bin/python -m unittest discover -s runtime/tests
-p 'test_*.py'` 完整 **375 項通過**。本機 MLX／MLX Metal 依 requirements 升至 0.32.2。

Swift 使用 `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer make test`，
**66 項通過**，排除以下四項既有 Keychain 互動案例，未將其列為通過：
`testAPIKeyRoundTripsThroughIsolatedKeychainItem`、
`testGenerationContinuesAfterLeavingAndReturningToChat`、
`testRapidStreamingCoalescesMessagePublications`、
`testStreamingFlushesPendingTextBeforeReportingAnError`。

`make package APP_VERSION=1.1.6 BUILD_VERSION=1.1.6` 已完成本機 ad hoc 封裝。
App 與 ZIP 解壓副本的 strict/deep 簽章、英文／簡中／繁中資源及六次隔離啟動通過；
啟動禁止讀取專案 `.build` 與 Swift module bundle，L10n 均從 App Resources 初始化。
包內 MLX／MLX Metal 0.32.2，Python **371 項通過**；四項研究 grader 測試因既有
沙盒禁止包內 Python 路徑而排除，這四項已在本次開發環境全套測試通過。
`pip check` 通過。未替換 `/Applications/Whallm.app`。

本機驗證後執行 `make release VERSION=1.1.6`，使用 Developer ID 與
`NOTARY_PROFILE=deepseek_ssd`。Apple 公證 submission
`97e1c66d-c6a3-436f-a3e8-90c4a3c35e6f` 為 Accepted。
標籤 `v1.1.6` 對應 `195420b2edbf409519c8ec34ef0230f8d530ba9c`，已推送 master。
包內 21 個 Git 追蹤 runtime 檔案與此 tag 內容完全一致。

[GitHub Release](https://github.com/yanun0323/Whallm/releases/tag/v1.1.6)
已公開為正式版本，包含 `Whallm-macOS-arm64.zip` 與 `appcast.xml`。
Release notes 與分支原稿逐字一致。上傳前及 GitHub 下載後的 App／ZIP 副本
皆通過 strict deep signature、公證 ticket、Gatekeeper、三語資源與隔離啟動檢查。
目前 `dist` 已更新為 1.1.6 簽章及公證成品。

下載檔案 SHA-256：

- `Whallm-macOS-arm64.zip`：`3732e5b741145bd4eec6f44d9cd614d043555c525dc68e29548c6ff3289426d6`
- `appcast.xml`：`5d08e384ebf2507f9f7c4143969c52bcff7882e0d3663c2d2e0d34072730cc3c`

本次原始日誌與取回來源 SHA-256 保存在本機 `scratch/release-1.1.6/`。
上述為目前測試結果，不是重新執行歷史真實模型或效能矩陣。

## 2026-09-08：MLX 0.32.2 升級與本機成品驗證

依使用者授權，requirements.txt 與開發環境 MLX／MLX Metal 升至 0.32.2，
mlx-lm revision 及其他 dependency pins 保留；未加入 finish()/clear_streams()。
既有 packaging guard 直接讀 requirements，已確認兩個套件版本皆符合新 pin。

- 開發環境完整 Python **375 項通過**，pip check 通過。
- `make package APP_VERSION=1.1.4 BUILD_VERSION=1.1.4 CODE_SIGN_IDENTITY=- NOTARY_PROFILE=`
  通過。App 與 ZIP 解壓副本均通過 deep/strict 簽章、資源檢查及三語系隔離啟動，
  六次啟動禁止讀取專案 .build 與 Bundle.module 資源。未 notarize 或發布。
- 簽章包內 Python 實際載入 MLX／MLX Metal 0.32.2，沒有外部 overlay，100 次 tuple
  worker lifecycle 通過；包內 21 個 runtime 來源檔與工作樹一致。
- 包內完整 375 項嘗試有 10 個 failure（含 subtests，涉及三個 test methods）：
  研究評分工具 `grade_qwen_quality.py` 的沙盒禁止讀 /Users，故無法執行位於 App
  內的子 Python。完整 log 保留；未修改沙盒規則。將該研究工具整個四項測試模組
  排除後，包內 **371 項 runtime 測試通過**。這四項在開發環境已全部通過。
- 包內真實 DeepSeek 六次獨立 HTTP 請求與 health 全部 200，輸出一致且與先前隔離
  基準相同；實際前綴重用 265 tokens，SSE 斷線後 idle／後續請求恢復通過。
- 包內 Qwen 四次各 32 tokens 抽樣請求及 health 全部 200。每個 server 在 harness
  cleanup 前仍存活，最後已終止。沒有新增通用速度或記憶體結論。

成品為 `dist/Whallm.app` 與 `dist/Whallm-macOS-arm64.zip`，App 版本號保留 1.1.4、
本機 ad hoc signature；未替換 `/Applications` 的 App。前版本機成品保存在
`scratch/issue7-mlx0322-upgrade-2026-09-08/previous-dist`。
機器、來源及成品 hashes、命令與完整通過／失敗 log：
[summary.json](benchmarks/2026-09-08-mlx0322-upgrade/summary.json)。

## 2026-09-08：Issue #7 MLX 0.32.2 隔離功能測試通過

未加 finish()/clear_streams()。同一本機 Python 3.14.7，MLX 0.32.1 tuple probe
仍 SIGSEGV，0.32.2 完成 100 次 worker lifecycle。目前工作樹 Python 375 項通過。
公開 1.1.5 runtime：DeepSeek 六次連續 HTTP 200 且輸出一致、實際重用 265 tokens、
SSE 斷線後恢復成功；Qwen 四次各 32 tokens 抽樣全部 HTTP 200。

bundled Python 直接載入新版 wheel 遭 Team ID 驗證拒絕，改用同版本本機 Python。
錯誤 Qwen API model ID 的初次 400 亦保留。正式 dependency、App、權重未修改，
簽章成品、長文、完整新功能組合及成對效能／記憶體尚未驗證。
見 [報告](../research/ISSUE_7_MLX_0322_2026-09-08.md) 與
[機器可讀證據](benchmarks/2026-09-08-issue7-mlx0322/summary.json)。

## 2026-09-08：Issue #7 回報者的 finish() 修法未通過

在公開 1.1.5 的隔離 runtime 副本，照回報者方法於 `OpenAIHandler.finish()` 的
`finally` 呼叫 `mx.clear_streams()`。原版第一筆 HTTP 200 後 SIGSEGV；修正版第一筆
成功且 server 存活，但第二筆出現 `There is no Stream(gpu, 0) in current thread.`。
三個修正版 fresh processes（close、要求 keep-alive、要求 keep-alive 且無中間健康檢查）
皆在第二筆失敗；四組第一筆內容相同。公開 server 強制 `Connection: close`，所以
keep-alive header 不代表實際同連線重用。完整 gate 狀態為 `REJECTED_BASIC_REPEAT_GATE`。

條件為 M2 Max 64 GiB、bundled Python 3.14.7／MLX 0.32.1、DeepSeek、DSpark off、
18-input／16-output、temperature 0、persistent prompt cache off。公開 ZIP、原版
server 與 libmlx hashes 已核對。未改正式 runtime／App／weights，未發布。
基本 gate 失敗後停止長前綴重用、取消恢復、Qwen／MTP 及打包採用測試。
[完整結果](../research/ISSUE_7_HANDLER_FINISH_2026-09-08.md)；
[原始資料與來源](benchmarks/2026-09-08-issue7-handler-finish/summary.json)。

## 2026-09-08：Model Advanced Settings 與合批生成

UI 增加預設開啟的 LRU 快取及 Qwen 最多四字詞合批；英／繁中／簡中標籤、
舊設定缺值遷移、明確關閉保存、模型限制與 catalog snake-case 編碼已驗證。
狀態位元複製修正始終生效。既有 Prefill 加速選項保留。

Python **375 項通過**；Swift **69 項通過**（63 + 6 個不同案例）。Swift 首輪因
Command Line Tools 缺少 XCTest 無法執行，改用本機實際存在的
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` 後編譯成功。
全套執行遇到既有聊天導航測試的 Keychain 等待，停止該程序；此一案例未完成，
其餘分組測試通過。沒有改 Keychain 權限，也没有把等待案例算成通過。

實際模型：M2 Max 64 GiB、MLX 0.32.1、Qwen3.8-Flash-Next、4096 slots、兩邊
皆 LRU、MTP 關閉、ANE 設定維持預設，persistent prompt cache 關閉。
在各自新程序依序執行繁中重複文字（64 outputs、greedy）、自然問答（32 outputs、
temperature 0.7／top-p 0.8／top-k 20）、精確 token 前綴重用（8 outputs、greedy）。
三組 output token、cached token、完整 state schema／逐陣列 raw-byte hash 全部一致；
沒有新增 system swapout。重複文字的 16 次合批提出 48 個候選並全部接受；自然問答
提出 3 個候選、接受 0 個，之後暫停猜測。候選不是未確認就輸出。

第一版 128-slot pages 的 matched 額外 MLX peak 為 **1,025,613,207 bytes**，
超過使用者 +1 GB 上限，因此未保留為最終實作；完整結果保存在 attempt-01。
改為 32-slot pages 後，matched 額外 MLX peak 最大 **703,732,987 bytes**；
process peak RSS 沒有增加。這是上述短輸入功能驗證，不代表所有長度的實測上限。
state fork 的保守計帳超過 300 MB 時不合批；缺候選或不支援模式亦逐字計算。
取消發生於 private verification、注入讀取失敗、接受／拒絕、各位置 EOS／長度停止、
隨機抽樣、跨 24 arenas、反覆 expert 頻率與一次 SSD acquisition 均有測試。

單次 request 時間：重複文字 22.889 → 21.850 s；自然問答 6.809 → 7.390 s；
prefix reuse 1.414 → 1.424 s。這些不是平衡交錯效能 gate，不宣稱普遍加速。
預設開啟遵循最新使用者要求，並未把先前未達效能門檻的結果改判。

證據與來源 hashes：[UI 整合記錄](benchmarks/2026-09-08-ssd-ui/summary.json)。
已編譯開發版 UI；未重新打包或替換 `/Applications/Whallm.app`。

## 2026-09-07：retained Prefill 的 8K／16K 延伸測量

精確 8192／16384-token 輸入，各兩個完整模型 worker 的 384／768 次 Prefill
observations、hidden、32 步 logits、前後 122 個 state arrays/schema exact。
各完成八次有效 `ABBABAAB` 新程序測量、固定 64 outputs、warm common file。
8K request **1.00393x**，僅微幅改善；16K **0.95716x**，速度 gate 拒絕。
MLX peak 最保守增幅 **230,392／442,364 bytes**，RSS **802,816／1,261,568 bytes**；
兩長度均符合 +1 GB，p95、ANE、輸出與無新增 swapout gate 通過。

8K 首輪第五次／16K 首輪第二次分別新增 140／88 頁 system swapout，首輪停止保存。
依事前規則各在新安靜期後完整重測一次；有效結果不混入首輪資料。
獨立重算與來源核對 **112 項通過**，16K 明確為速度失敗，不是數值或記憶體失敗。
沒有修改 production／App，沒有重跑未變更的既有 Python unit suite。
[結果與限制](../research/SSD_RETAINED_LONG_RESULTS_2026-09-07.md)；
[機器可讀證據及 archive hashes](benchmarks/2026-09-07-ssd-retained-long/summary.json)。

## 2026-09-07：長 Prefill 保留 pages／Decode ready groups

Python **360 項通過**。新增六項測試驗證跨 chunk 重用、資料生命週期、
取消／失敗收尾、先算 resident、LFU 一致與 reader 失敗後重試。
長 Prefill 在 4K／8K 的 expert tensors、hidden、32 步 logits、前後各 122 個
state arrays 及 schema exact；4K 八次交錯測量 request **1.0907x**、TTFT
**1.1059x**，MLX／RSS 最大候選減最小原版為 **450,504／966,656 bytes**。
輸出一致、ANE 無 fallback、無新增 swapout。8K 沒有效能／記憶體 gate。
真實 4K 五對取消／恢復／重用情境也通過，26 個對照輸出 token 與 reuse 相同。

Decode native wait-all 與 ready groups 兩版的 32 步 logits/state exact。
原定三版 12 次效能測量在第 10 次原版對照出現 80 頁 system swapout，
狀態 **REJECTED_SWAPOUT**，停止後兩次；已完成十次輸出與 logical reads 相同。
這是效能驗證無效，不是速度通過，也不能判為候選自身記憶體失敗。

50 項獨立證據 audit 通過；三份舊 source archive 未變，新 source／raw evidence
另封存並讀回逐檔 hash。兩個早期 harness 假設錯誤與修正 log 保留，未放寬數值要求。
production／App 預設未修改，沒有新增完整生成 2x 或跨模型認證。
[結果與限制](../research/SSD_RETAINED_READY_RESULTS_2026-09-07.md)；
[完整證據及封存 hashes](benchmarks/2026-09-07-ssd-retained-ready/summary.json)。

## 2026-09-07：直接 resident 短 block

Python **354 項通過**。新增四項 native 跨 page／重排重複 routes、slot 重寫可見性、
budget guard／hook 恢復及輸入形狀驗證。完整模型兩個 shape 的精確 logits/state
初驗通過，再完成 16 次原版配對與 8 次同 native 歸因配對，候選 fork 隔離正確。
兩字 0.9374x、四字 1.0530x；MLX/RSS peak 差額皆低於 +1 GB。只有四字
通過 verifier 成本初篩，沒有真正 speculative generation 或端到端 2x 證據。
既有穩定 Prefill／失敗候選的封存內容與 executable sources 校驗通過。
[證據及 source snapshot](benchmarks/2026-09-07-qwen-resident-block/summary.json)；
[限制與結果](../research/QWEN_RESIDENT_BLOCK_RESULTS_2026-09-07.md)。

## 2026-09-07：Prefill lifecycle 封存與短 block verifier 成本拒絕

最新 Python **350 項通過**。Prefill 的獨立 lifecycle gate 通過 **23 對情境**，
680 個對照輸出 token、prompt 與逐次 reuse 序列相同；涵蓋多輪、4K fallback、
256-token 生成、Prefill 途中取消、Decode 第 5 token 中止、恢復、12 次重用，
以及重啟後互讀另一版 persistent cache。所有 banks、batched scope 與 lock 均釋放，
最後 5 次 active MLX 波動兩版皆 16 KiB，MLX/RSS peak 差額符合 +1 GB。
最初 control 的「每次都必須 cache hit」harness 失敗保留，修正後使用新目錄。
[穩定性證據與封存 source](benchmarks/2026-09-07-ssd-prefill-stability/summary.json)
保存逐次結果及已驗證的 tar SHA-256。

短 block 原型維持 token-shaped dense/state，僅合併 expert union。2 / 4-token
完整 logits、122 個 cache arrays 與 metadata exact，fork 未修改原 cache。
各 8 次 ABBA/BAAB 的 verifier 速度為 **0.944x / 1.001x**，logical reads 均未下降；
新增打包／同步成本下未達 1.05x gate，狀態 **REJECTED_VERIFIER_COST**。
沒有疊加 drafter、跑正式 speculative generation 或改 App 預設。
350 項功能測試通過不能覆蓋這個效能拒絕。
[短 block 證據](benchmarks/2026-09-07-qwen-short-block/summary.json)
與 [完整研究紀錄](../research/SSD_STABILITY_AND_BLOCK_2026-09-07.md) 保存限制和停止條件。

## 2026-09-07：SSD expert Prefill 流水線研究原型

Python **347 項通過**，包含新增 9 項雙 buffer、partial read、取消／drain、
重用、GPU fence、workspace guard、trace 記憶體有界與輸出精確一致測試。
初次 API 錯誤及 trace 累積的重現失敗各自保留，未覆寫。
Qwen 真實權重 component 的 128 / 1024-token gate 通過；143 / 1024-token
完整模型 observer 的 48 層 input/routes/expert output、32 個生成 token 均 exact。
收尾修正 trace 後，再次驗證 1K observer、生成 token 與讀取量相同。

原始不受控 startup 的 full gate 拒絕，使用者更新為 +1 GB 記憶體上限
後也沒有將其他失敗抹除。後續兩種長度各 8 次 warm-common ABBA/BAAB
通過 bounded screen，64-token 輸出一致、ANE 無 fallback、無 swapout。
request 加速 **1.77x / 1.36x**，不是普遍 2x 的結果。
沒有修改 production/App、模型權重或建立 commit。
規則與限制見 [研究紀錄](../research/SSD_PREFILL_PIPELINE_2026-09-07.md)，
source/raw hashes 和逐次資料見 [證據摘要](benchmarks/2026-09-07-ssd-prefill-pipeline/summary.json)。

## 2026-09-07：Whallm 1.1.5 發布

[GitHub Release](https://github.com/yanun0323/Whallm/releases/tag/v1.1.5)
的 tag `v1.1.5` 指向 `13d44524c9c12a71f26bee79b33bc345a5163d7f`。
發布內容見 [Release notes](../Packaging/ReleaseNotes/1.1.5.md)。

本次在 `/Users/Shared/Project/test/llm_ssd` 將本機 MLX／MLX Metal 從
0.32.0 更新為 requirements 指定的 0.32.1 後，Python 338 項通過；
Swift 65 項通過、零略過，另外四項既有 Keychain 互動測試排除。
先完成 `make package APP_VERSION=1.1.5 BUILD_VERSION=1.1.5` 本機驗證，
再執行 `make release VERSION=1.1.5`，使用 Developer ID 與
`NOTARY_PROFILE=deepseek_ssd`。Apple 公證 submission
`ace7c95c-4144-461c-9917-f6919c18ceba` 為 Accepted。

上傳前與 GitHub 下載後均通過 strict deep signature、公證 ticket、
Gatekeeper、英文／簡中／繁中資源及隔離啟動檢查。隔離啟動禁止讀取
專案 `.build` 和 Swift module bundle，六次 L10n 初始化均完成。
GitHub Release 包含 ZIP 與 `appcast.xml`，發布筆記與本機原稿一致。
本次原始日誌保留於本機 `scratch/release-1.1.5/`。

下載檔案 SHA-256：

- `Whallm-macOS-arm64.zip`：`5ba28945e1f621b452667690a8d4ca01e4dc50c80de02a5eb4b6e83d36a0ddf5`
- `appcast.xml`：`79f2a851137b738de2e8a5f51e6ae205c7498eb007766ae47ead1037a04d29c1`

## 2026-09-07：最終 build local 與 dist 清理

以 `ec39204` 基礎及既有修復工作樹重新執行測試和 `make package`，App
version／build 均維持 **1.1.4**。Python **338 通過**；Swift **62 通過、
3 項 fixture 略過、4 項 Keychain 等待案例排除**，零失敗。
App 與 ZIP 解壓副本皆通過 strict deep signature、三語共六次明確 L10n
初始化（禁止存取 `.build` 與 Swift module bundle）、包內抽樣各三項，
以及各 127 個 Mach-O 的依賴檢查。三語皆優先由 `Contents/Resources` 載入。

驗證後依使用者要求清理 `dist`，僅留下 `Whallm.app` 與
`Whallm-macOS-arm64.zip`。下方歷史紀錄提及的 `dist` 備份目錄已移除；
原始驗證資料仍保留在 `docs/benchmarks/`。重新開啟最終 App，維持本次
build 前的 server 停止狀態，未替換 `/Applications/Whallm.app` 或發布版本。
ZIP 大小 `120426105` bytes，SHA-256：
`cb7b8a16b2b254bd25757cc67027c5ca6909d395f829dec95c4ed49bcf0f53ac`。
命令、source hash、測試和清理清單見
[最終本機打包紀錄](benchmarks/2026-09-07-final-build-local/summary.json)。

## 2026-09-07：Codex Ctrl+C 取消 Qwen 生成

在串流保活修復上繼續重現：正常 TCP close 後工具生成未於一秒內退出；
已斷線的排隊請求仍進入模型載入。修復前兩項測試皆失敗，原始 log 已保留。
新增 request-scoped cancellation、50 ms 連線觀察、可取消的 request lock 等待，
以及 Qwen Prefill layer／chunk、Decode／MTP 的合作取消檢查。
所有 MLX 運算與收尾保持在原 generation thread；iterator 關閉且已提交工作
完成後才釋放 request lock。中途取消不將未完成 generation cache 提升為可重用結果。

Python **338 項通過**，包含正常 close、不依賴保活的工具取消、排隊取消、
首 token 前取消／下一請求恢復、Qwen chunk 中止及 expert scope 釋放。
Swift **62 通過、3 項 fixture 略過**，四項既有 Keychain 等待案例排除，
零失敗；首次 make ARGS quoting 錯誤及修正後 log 分別保留。

本機 M2 Max／64 GiB、MLX 0.32.1、Codex CLI **0.153.4**，獨立 loopback
server 與隔離 CODEX_HOME，不讀取或更動使用者 API key。使用 1,152 slots、
預設 Qwen grouped Prefill 開啟、Decode 實驗與 MTP 關閉、persistent cache
停用的新 runtime，實際將 SIGINT 傳至 Codex 前景程序群組（Ctrl+C 的訊號）：

- 可控制模型於約 **0.036 秒**釋放生成，首 token 前停止。
- 真實 installed Qwen 於約 **0.078 秒**釋放生成，首 token 前停止。
- 真實 Qwen 開始輸出後關閉 HTTP 連線，約 **0.008 秒**釋放生成。
- 上述真實取消後皆未新增輸出 piece；兩次後續請求正常回答 `OK`，
  token hash 同為 `54f9f9a9d7ec1d59a05429914544c095bae6fedf15abfca4e2a5aac9d06da9d0`。

這些是個別取消／恢復的功能檢查，不是停止延遲上界、效能基準或長期可靠率。
已執行 kernel、同步 I/O、首次模型載入仍須收尾；proxy 若保留上游連線，
server 無法單靠 TCP 得知 UI 已停止。MTP 取消檢查未執行實機驗證。
下方保活修復的「等寫入失敗，再等第一個 token」限制已被本次修復取代。
原始測試、Codex stdout／stderr、實機配置與恢復輸出見
[取消驗證資料](benchmarks/2026-09-07-codex-cancellation/qwen-probe.json)。

本次 `make package` 產生 App 1.1.4／ZIP，原件與解壓副本皆通過 strict deep
signature、英文／簡中／繁中共六次明確 L10n 初始化（拒絕讀取 `.build` 與
Swift module bundle），包內抽樣回歸各三項及各 127 個 Mach-O 依賴檢查。
驗證模式不建立 ContentView、不讀 API key。ZIP SHA-256：
`f7607ce93966c208849017d4a7575ab6ab8a2a7da9f21e2d80ecfcd0a4a5416d`。
已替換本機 `dist` 並啟動服務，舊包保留於
`dist/before-codex-cancellation-fix-20260907/`。未替換 `/Applications/Whallm.app`，
未建立 commit、tag 或公開 release。下方舊紀錄的 Keychain 啟動等待已解除。
對目前打包 App 的 `127.0.0.1:11434` 再測一次真實 Qwen：輸出中關閉連線，
約 **0.058 秒**觀察到 server 回到 idle，下一請求正常完成，最終維持 idle。
此測試保留原有模型設定與認證，API key 僅在測試程序記憶體內使用；資料與
完整 source hash 見 [本次摘要](benchmarks/2026-09-07-codex-cancellation/summary.json)。

## 2026-09-07：Codex Responses 串流保活修復

在 `ec39204` 基礎工作樹重現四項串流問題：工具結果暫存與 Prefill 等待時
沒有 SSE 事件、client 已斷線仍產生完整結果、生成錯誤只留下 `response.created`
後的 EOF。修復前四項測試為 2 failure／2 timeout error；修復後全部通過。
完整 Python **334 項通過**；Swift **62 通過、3 項 fixture 略過**，另排除
四項既有 Keychain 等待案例，零失敗。

以本機 Codex CLI **0.153.4**、隔離的 `CODEX_HOME`、明確 272,000 context、
500 ms SSE idle timeout，搭配每次回覆延遲 1.5 秒的可控制模型測試：
沒有保活時出現 `idle timeout waiting for SSE`；100 ms 保活時完成
`exec_command` 的 `printf stream-ok` 及最終答案，兩次模型請求、exit 0。
正式 runtime 使用 **10 秒**保活，事件為 `response.in_progress`；不改生成算式
或 MLX 執行緒，也不提早公開未驗證工具參數。

使用者貼出的 `error decoding response body` 原始連線尚未擷取；本輪證明的是
一條可重現的 idle timeout 路徑，不能宣稱所有 network error 都已消除。
第一輪 Codex 測試繼承技能且觸發人工測試的 context compaction，保留該紀錄；
隔離設定後重跑，確認最終答案與工具執行都符合預期。

修復版 App／ZIP 已通過 strict deep signature、原件與解壓副本各三語隔離啟動、
包內 MLX 0.32.1 抽樣回歸各三項、127 個 Mach-O 的絕對依賴檢查。
本機 `dist` 已換為此工作樹的修復版，原件保留在
`dist/before-codex-stream-fix-20260907/`。
使用者完成 Keychain 授權後，已啟動串流修復版並恢復服務。真實 Qwen 的
2,226 input／27 output token 工具請求在 55.70 秒完成，收到四次保活，
SSE 最大事件間隔 10.006 秒，正確回傳 `get_weather({"city":"Taipei"})`；
未實際執行工具。這是 HTTP Responses 的實機功能檢查，Codex CLI 對照則使用
可控制模型；不是長時間網路可靠率或正式效能量測。

使用者反映大量 Keychain 提示後，另修正兩個重複存取來源：SwiftUI 重建
ContentView 時不再讀 API key，第一次 view task 才載入；一般設定變更不再
存取 Keychain，僅 API key 本身變更才保存。
打包驗證使用不建立 ContentView 的 `--verify-localizations` 模式，並要求
實際 L10n lookup 完成後的明確 marker。先前「只確認程序存活」不足以區分
初始化完成與卡在 Keychain，舊紀錄不構成已通過這個更嚴格 gate 的證據。
第一版 marker 放在 onAppear，直接啟動的副本沒有建立視窗而未觸發；失敗 log
已保留，改為 App init 內 lookup 與 marker，後續重新打包驗證。
新簽章 App 的正常啟動仍可能需要一次 macOS 授權；這不是移除 Keychain 保護。
最終完整包已完成六次明確 L10n marker 驗證，原件與 ZIP 副本的簽章、包內抽樣
及依賴檢查均通過。ZIP 大小 `120423726` bytes，SHA-256 為
`5edf20a7d1df44e38f4810ed81cf620decdb4d6e46a189b92e80e4032a678f5c`。
最終 App 的 server.py 與上述實機 Qwen 驗證使用的檔案 hash 相同。
目前完整包已放至本機 `dist`，正常啟動仍等待 macOS Keychain 授權後恢復服務。
完整命令、source hash、失敗與通過紀錄見
[串流修復紀錄](benchmarks/2026-09-06-codex-stream/summary.json)。

## 2026-09-06：合併後 build local 完成

以 `master` source commit `ec39204f2afc9aec4386236ffc775ce8d1a9cb7a` 完成
`make package`，本機 App version／build 均為 `1.1.4`，採 ad hoc signature。
成品包含 Prefill 加速設定與 MLX／MLX Metal **0.32.1**。
打包前重跑 Python **330 項通過**；Swift **62 通過、3 項缺少 fixture 略過、
零失敗**，另排除四個先前會等待 Keychain 的案例，不算通過。

App 與 ZIP 解壓副本均通過 `codesign --verify --deep --strict`，各完成英文、
簡中、繁中隔離啟動，共六次，各存活三秒。測試禁止讀取專案 `.build` 和
各副本的 Swift resource bundle，沒有發生 L10n fallback trap；額外讀取控制
確認 sandbox 的禁止規則生效。確認 L10n 先讀 `.main` 再讀 `.module` 的順序，
`Contents/Resources` 的三語檔案皆與 source 相同。

兩份成品的包內 Python 各通過三項抽樣回歸測試，測試後簽章仍有效；各 127 個
Mach-O 檔案未發現非系統的絕對依賴路徑，runtime source 與原生檔案完全一致。
ZIP 大小 `120415290` bytes，SHA-256：
`bc294575aa4f5cc56da2c010cf0528c0d26aa18f691429fcf9959fc8a46a4856`。
命令與原始 log 見 [本機打包紀錄](benchmarks/2026-09-06-merged-build-local/summary.json)。
未建立 tag、notarize、upload、替換已安裝 App 或重新量測模型效能。

## 2026-09-06：合併 feat/optimize_qwen 至 master

將遠端 `87ffb59845aef94555fea703b511ab7249cf836b` 合併至本機
`a3572c3c59876edd5ec0ca5ee488fa7aee259642`，保留 MLX **0.32.1** 修復、
Qwen 預設開啟的 `qwen_grouped_experts` Prefill 加速與預設關閉的
`qwen_grouped_decode` 實驗。兩個 catalog 欄位可各自省略；另補測兩者皆缺少的
舊 Qwen／DeepSeek catalog，確認可載入並套用各自預設值。

合併後 Python **330 項通過**；Swift 執行 65 項，**62 通過、3 項缺少 fixture
略過、零失敗**。四個先前會等待 Keychain 的案例明確排除，不算通過。
測試使用本機 M2 Max、MLX 0.32.1 與完整 Xcode；命令與原始 log 見
[合併驗證 artifact](benchmarks/2026-09-06-merge-optimize-qwen/summary.json)。

下方保留兩個分支各自的歷史測試、打包與效能紀錄。遠端分支的 MLX 0.32.0／
M5 Pro 效能結果並非合併後 MLX 0.32.1 的重新量測；本輪沒有執行模型效能測試。
合併當時未重新打包，當時本機 `dist` 是下述 `5245d42` 的 Issue #6 build。
後續已完成上方 `ec39204` 的 build local，包含此次合併的 Prefill 設定介面。
未推送遠端或替換已安裝 App。

## Issue #6 重現、修復與合併前本機打包

2026-09-06 另以已安裝的 Whallm **v1.1.4**（不是目前開發工作樹）完成
[#6 的四輪探索性重現](../research/ISSUE_6_REPRODUCTION_2026-09-06.md)：
Qwen thinking 的日文長文在正文第 641 個字元起，連續重複同一短語 187 次後
由測試者停止。相同題目的 chat 寫出 6,052 字元並自行 `stop`，但結尾仍有重複段落。
短句改寫兩種模式都完成，thinking 先產生 3,569 字元 reasoning。
這確認已發布版本的一個正文重複案例；不代表根因、修復、發生率或正式效能已驗證。
完整輸出與環境見 [重現 artifact](benchmarks/2026-09-06-issue6-reproduction-m2-max/summary.json)。

Issue #6 根因調查在同機重現 MLX 0.32.0 的 compiled sampler 背景執行緒缺陷：
固定分布抽樣 32 次，亂數 state 完全不前進。換成 MLX 0.32.1 後正常前進，
相同 seed 的主／背景執行緒輸出一致；保留 0.32.0、僅取消 categorical sampler
編譯的隔離控制亦恢復正常。與 [上游缺陷](https://github.com/ml-explore/mlx-lm/issues/1675)
和 [上游修復](https://github.com/ml-explore/mlx/pull/3828) 相符。
目前工作樹更新 pinned MLX 為 0.32.1，加入打包版本檢查與三項抽樣回歸測試；
完整 Python suite **308 項通過**，舊版可觀察到四個 failure（含 subtests）。
本輪 M2 Max 開發環境的 **8 個 HTTP 案例正常結束**：三篇長文（兩個 thinking、
一個 chat）、兩個短句、程式碼、JSON、tool call。三篇長文正文分別 3,947、5,556、
5,544 字元，均自然 `stop`，沒有 token cap／逾時／循環中止；全文檢查與重複掃描
未見持續循環。第一篇使用 frozen v1.1.4 runtime＋新版 MLX，其他使用目前 source。
仍有一篇字數偏短、用字／時間線瑕疵；程式碼功能通過三組輸入，但多了 Markdown
圍欄。這是抽樣根因與本輪循環／終止檢查通過，**不是完整文字品質驗收或發生率估計**。
JSON 解析符合指定物件；tool 正確輸出 `get_weather({"city":"Taipei"})`，未實際呼叫
外部工具。命令、環境、source hash、raw token IDs／hash 與全文見
[機器可讀彙整](benchmarks/2026-09-06-issue6-causal-m2-max/summary.json)。
後續已將修復提交至 `master` 並完成下列本機打包；未替換已安裝 App，未發布。
詳細因果對照與證據見 [調查紀錄](../research/ISSUE_6_CAUSAL_INVESTIGATION_2026-09-06.md)。

先前 [框架評估](../research/ISSUE_6_FRAMEWORK_ASSESSMENT_2026-09-06.md) 的 vLLM
離線循環偵測不是文字品質修復；API／penalty 與數值算式候選均未採用。

2026-09-06 Issue #6 **build local 完成**。打包 source commit 是
`5245d426732a8811d6210a844069131b15aa8039`，branch `master`；App version／build
均為 `1.1.4`，ad hoc signature，包內 MLX／MLX Metal 均為 `0.32.1`。
重新執行 Python 308 項通過；Swift 執行 64 項，61 通過、3 項缺少 DeepSeek fixture
而略過、零失敗。四個先前會等待 Keychain 的案例明確排除，不算通過。
最初選用 CommandLineTools 的 Swift command 缺少 XCTest，改以完整 Xcode 的
`DEVELOPER_DIR` 重跑後通過；未修改全域 Xcode 選擇。

`dist/Whallm.app` 與 ZIP 解壓副本均通過 strict deep signature、三語 localization
檔案與 source hash 比對，以及各三個語言的隔離啟動。共六次啟動均存活三秒，
sandbox 禁止讀取專案 `.build` 和各副本的 Swift resource bundle；另以實際檔案
讀取確認 sandbox 確實拒絕兩種路徑。`L10n` 先讀 `.main` 的邏輯與 resource 檔案
保持一致，未發生 module fallback trap。兩份 App 的 packaged Python 各通過三項
背景抽樣回歸測試，測試後簽章仍有效。每份 127 個 Mach-O 檔案未發現非系統的
絕對依賴路徑，兩份 runtime source／native payload 完全一致。

ZIP 大小 `120401377` bytes，SHA-256：
`fbeb8dfb59c5286d3f46e0587948765e50c01d97ce2e6d2d49bc0db2a99b0f8f`。
完整命令、各輪 log 與檢查結果見
[build local artifact](benchmarks/2026-09-06-issue6-build-local/summary.json)。
沒有建立 tag、notarize、upload 或替換 `/Applications/Whallm.app`。

## 2026-09-06：Qwen「Prefill 加速」開關與本機 App 打包

Qwen 的 Model Advanced Settings 新增預設開啟的「Prefill 加速」開關。
選擇會儲存，並透過 model catalog／configure 傳入 `qwen_grouped_experts`；
未載入模型下次載入時生效。舊設定缺少欄位仍預設開啟，明確 false 不會被覆蓋。
Loaded／Loading 模型維持鎖定；MTP 開啟或 layer-major prefill 關閉時停用此開關。
英文、簡中、繁中名稱與說明皆已加入；DeepSeek 不顯示此選項。

`make test` 的 69 個 Swift 測試，以及模型管理／API／CLI／分組計算的 87 個 Python 測試通過。
執行 `env -u NOTARY_PROFILE CODE_SIGN_IDENTITY=- make package` 完成本機打包。
`dist/Whallm.app` 與 ZIP 解壓後的 App 均通過 `codesign --verify --deep --strict`。
兩份 App 都在禁止讀取專案 `.build` 與 Swift module 資源 bundle 的情況下，
以 en／zh-Hans／zh-Hant 各啟動 3 秒並保持運作。
三語資源位於 `Contents/Resources`，`L10n` 先讀 `.main` 再讀 `.module` 的順序已核對。
包內 Python 另在禁止讀取專案 `.build`、`.venv` 和 `runtime` 時完成匯入，
確認 Qwen 預設開啟、明確 false 可關閉，以及舊 DeepSeek catalog 的補值不變。

App 採本機 ad-hoc 簽章；沒有公證、建立 tag 或上傳。
[打包證據、檔案雜湊與測試紀錄](benchmarks/2026-09-06-qwen-prefill-ui-local-package.json) 已保存。
介面使用有本地化名稱的系統 Toggle；本輪未執行互動式鍵盤／VoiceOver 操作測試。

## 2026-09-06：Qwen 無提示排序的新題目、長生成與快取驗證

N1 使用四類新編題目、1K／16K 輸入、每次 1,024-token 生成，兩輪反向配對共 32 次執行。
16 對完整 outputs、logical expert bytes 和 QMM 呼叫數相同；同題四次重跑的輸出也相同。
跨四題首次回覆改善中位數為 1K **7.04%**、16K **13.46%**。
逐題兩對的 Decode／p95／MLX peak／RSS 均通過原訂門檻，code 長生成沒有重現先前的小幅回退。

N2 使用兩個 2,048-token 分支、1,024-token shared checkpoint 和 128-token 生成。
10 個 processes 的 18 次回答與各分支未快取對照相同；其中 14 次實際重用快取。
A→B→A 的 reuse 為 1,024／1,024／2,047 tokens；四種 writer／reader 重啟組合都通過。
兩版 format-5 checkpoint 的 122 個 tensors、state schema 與 metadata 相同，分支執行後也未改變。

保存 [N1 完整量測](benchmarks/2026-09-06-qwen-nohint-n1-runs-m5-pro.json)、
[N1 門檻](benchmarks/2026-09-06-qwen-nohint-n1-gate-m5-pro.json)、
[N1 來源索引](benchmarks/2026-09-06-qwen-nohint-n1-index.json)、
[N2 完整合約](benchmarks/2026-09-06-qwen-nohint-n2-cache-m5-pro.json)
與 [N2 來源索引](benchmarks/2026-09-06-qwen-nohint-n2-index.json)。
N1 的 native binary 額外雜湊是在第 15 次 request 後補記，結束時再次核對相同；
Python 原始碼則在開始前固定。N2 已在執行前固定 native binary／source。
N1 全部回答均達生成上限，這是等長運算與輸出比較，不是完成任務的正確率成績。
N2 是功能驗證，不以其時間宣稱快取加速；本組 ANE evaluation counter 為零。

當時預設關閉的 `qwen_grouped_experts` 已通過 [N3 實際 runtime 對照](../research/QWEN_NOHINT_INTEGRATION_2026-09-06.md)：
八次效能執行與八次快取互讀均通過，1K／16K 首次回覆改善 7.59%／13.30%，
Decode 變化 +0.17%／−0.14%，輸出與 N1 對應片段一致。N1–N3 共 66 次模型請求。
[N3 完整量測](benchmarks/2026-09-06-qwen-nohint-n3-integration-m5-pro.json) 與
[來源索引](benchmarks/2026-09-06-qwen-nohint-n3-index.json) 已保存。

相關 runtime／Qwen／CLI／cache 測試共 140 個通過。
N3 後先修正 model catalog 對舊設定檔的相容性：新 boolean 可省略，當時預設 false；
80 個模型管理／API 測試通過，累計 220 個。設定解析修正未改動 N3 的生成計算程式，
詳見 [最後核對](benchmarks/2026-09-06-qwen-nohint-final-verification.json)。
使用者後續選擇預設開啟；現在 Qwen 省略新欄位時為 true，明確 false 仍可關閉。
採用階段新增的預設／關閉／MTP 保護檢查通過；本次 225 個 Python 測試與 24 個 App 設定測試通過。
[採用驗證](benchmarks/2026-09-06-qwen-grouped-experts-default-on.json) 保存目前來源與不帶開啟參數的實際模型檢查。
原始量測來源與結果保留不變；當輪尚未打包或發布。
後續 UI 開關與本機打包驗證見本文最新紀錄。

## 2026-09-06：Qwen 開排序提示候選在 Q1A 因回退停止

預定四類各 8 題，在第 27 題出現原版對／候選錯，依事前規則停止後續 5 題。
已執行 27 對、54 個模型 process，另以相反順序重跑回退題 2 個 process。
數學題 `math-0926` 原版在 1024-token 上限截斷，另列未定；26 對有完整答案可比較。
其程式功能原版／候選各 7/7；數學各 6/6；繁中 6/7 對 5/7；工具紀錄續接各 6/6。
這不是完整公開能力基準成績，也不外推一般任務正確率。

`zh-07` 正確錄取名單為丙、丁、戊，候補為乙；候選漏掉丙並錯列候補。
反向重跑仍是原版對、候選錯，兩版各自的 output token hash 與第一次相同。
兩次候選均有 192 次預期 Prefill expert 呼叫。
因此目前開提示版本未通過「任務正確率不下降」門檻，未採用，Q1B 和截斷題延長生成補測暫緩。
無提示候選的既有輸出一致性／速度篩選不受此判定覆蓋。

保存 [完整結果與反向確認](benchmarks/2026-09-06-qwen-quality-q1-gate-m5-pro.json)、
[證據索引](benchmarks/2026-09-06-qwen-quality-q1-index.json)
與 [規則、題目來源、未覆蓋範圍](../research/QWEN_QUALITY_Q1_2026-09-06.md)。
已重新計分並核對 hashes；8 個相關 CPU 單元測試和固定 seed 複核通過。
runtime、App 和模型預設未變更。

## 2026-09-06：Qwen 開排序提示候選的任務初測

以新編固定題組完成 24 個獨立 process／12 題配對，1K 輸入（含填充）、greedy、thinking off、
4096 slots、MTP off、persistent prompt cache off，生成上限 128 tokens，沒有截斷。
程式理解原版／候選各 2/3；數學各 2/3；繁中理解各 3/3；工具參數 JSON 各 3/3。
兩版皆 10/12，沒有原版答對／候選答錯；11/12 題輸出 hash 相同。
唯一文字不同的程式題兩版均錯，不能把文字差異直接當成能力下降。
每次候選都有 192 次開排序提示的 Prefill expert 呼叫。

這是小型正確性初測，不是正式能力基準或速度採用結果；每類只有三題，
未涵蓋程式生成、多步工具與真實長內容，尚未證明「任務正確率不下降」的完整採用條件。
保存 [逐題結果與來源紀錄](benchmarks/2026-09-06-qwen-task-accuracy-screen-m5-pro.json)
與 [規則、共同錯誤和下一步](../research/QWEN_TASK_ACCURACY_2026-09-06.md)。
24 次執行的 hashes／評分已重新核對；評分器與相關 4 個 CPU 測試通過。
runtime 與 App 預設未變更。

## 2026-09-06：Qwen QSA 批次與 expert 分組

QSA query chunk 4→16 完成五類 4K／256-token、兩組相反順序配對，共 20 個獨立 process。
10 對輸出 tokens 與 logical expert bytes 全部相同。
主要四類 TTFT 改善中位數 2.64%，未達 5% 門檻；Decode、p95、記憶體保護條件通過。
此候選未採用。保存 [完整配對](benchmarks/2026-09-06-qwen-qsa-chunk-paired-runs-m5-pro.json)
與 [門檻結果](benchmarks/2026-09-06-qwen-qsa-chunk-paired-gate-m5-pro.json)。

Expert 分組的初次候選開啟排序提示，在 code 4K／32 從第 5 個 token 起分歧，未通過精確輸出條件。
layer 0／23／47 的排序還原檢查全部通過；關閉提示後，三層 expert 輸出元素完全相同。
無提示候選的 code 4K／32 完整輸出和反向短測試也通過。
隨後五類 4K／256-token 的 20 組完整配對，主要四類 TTFT 改善中位數 15.79%；
全部 output tokens、logical expert bytes 與 gather_qmm 計數相同，Decode／p95／記憶體保護條件通過。
程式碼題目的 Decode 中位數 −2.20%，需在長生成另驗；本輪不是 Decode 加速結論。
保存 [完整配對](benchmarks/2026-09-06-qwen-grouped-experts-nohint-paired-runs-m5-pro.json)
與 [門檻結果](benchmarks/2026-09-06-qwen-grouped-experts-nohint-paired-gate-m5-pro.json)。
尚無獨立 held-out、長輸入或 cache contract 驗證，未採用；不混用初次失敗候選。
數值診斷見 [artifact](benchmarks/2026-09-06-qwen-grouped-experts-numerical-diagnosis-m5-pro.json)。

上述都是指定輸入的研究驗證，runtime 程式和模型預設未改。

## 2026-09-06：Qwen Prefill recurrent kernel 診斷

沿用前一輪 runtime source hashes、Qwen 4,096 slots、exact／MTP off、48 GiB memory limit，
以 code 4K 跑正常基準、獨立同步 observer、正常重跑，各 32 tokens，均與前一輪前 32 tokens 相同。
144 次 Prefill kernel 呼叫共 0.5528 秒；扣除兩種形狀首次呼叫後為 0.5437 秒。
兩次正常 TTFT 是 69.910／43.981 秒，故組件計時約為其 0.78%／1.24%。
這是受同步影響的投入排序參考，非嚴格瓶頸占比，也沒有候選加速結果。
完整分塊改寫在此案例暫緩，不能推論其他輸入長度或整個研究方向失敗。

保存 [逐次計時、三次完整 metrics 與來源 hashes](benchmarks/2026-09-06-qwen-gated-delta-prefill-diagnosis-m5-pro.json)。
方法與限制見 [H2 研究紀錄](../research/GATED_DELTA_PREFILL_2026-09-06.md)。
檢查了輸出 hashes、呼叫數、來源／樣本 hashes 與 Python 編譯；runtime 未修改，未重跑完整 App／runtime 測試。

## 2026-09-05：Qwen 研究第一輪

基準 commit 為 `7dc9cf8f050c75def77c0563cd7b8ac03f2d8435`，
Apple M5 Pro／64 GiB、MLX 0.32.0、mlx-lm 0.31.3。
Qwen 使用 4,096 slots、exact、MTP off、48 GiB memory limit；
每個 request 是獨立 process，prompt cache 不重用，OS page cache 未清除。

五類合成診斷輸入各有 1K／4K 基準，另加一次 code 4K 重跑與五次獨立路徑記錄，
共 16 組，每組 256 output tokens。重跑與路徑記錄的完整 token hashes 均對上基準。
五組目前 LFU 的離線 Decode miss 數也全部對上 runtime 記錄。
這是本次工作樹的診斷驗證，不是候選優化的正式速度或能力驗證。
同題正常重跑仍有速度波動，未逐次記錄系統記憶體壓力或實際 ANE 執行次數。

`prefill_guided_24` 的四類主要題目 miss 變化中位數為 +9.27%，未通過離線入口門檻。
三層、每層一對 experts 的固定 base 權重篩選也完成，但不利於直接採用；
它不含通道排列對齊、微調或生成品質測試，不能否定完整的共用權重方法。
兩個候選均未接入 runtime，App 預設不變。

完整 commit、環境、設定、prompt／output hashes、原始資料和保存檔案的 hashes 見
[證據索引](benchmarks/2026-09-05-qwen-research-round1-index.json)。
結果與方法限制見 [B 第一輪報告](../research/PREFILL_DECODE_B_ROUND1_2026-09-05.md)。
本輪另通過 7 個 CPU 單元測試、保存檔案／解壓 trace 的 hash 核對與 diff 檢查；
沒有重跑完整 runtime 或 App 測試，runtime 程式未改動。

## 既有驗證背景

本文件分開記錄目前驗證和歷史量測。
以下為本次依賴修正前的自動測試：2026-09-06、base commit `7dc9cf8f050c75def77c0563cd7b8ac03f2d8435`
加上 default-off Qwen grouped Decode 工作樹。Python 305 項通過；Swift 成功編譯，
排除四項 Keychain-dependent 案例後，64 項執行、61 通過、三項缺少 DeepSeek fixture
而略過、零失敗。全套 Swift 曾在 `SecItemCopyMatching` 等待，未宣稱全套通過。
CLI／APP capacity、cold allocation、multi-request／cancel／restart 與各失敗設計的
完整 logs、source snapshots、commands、token IDs／hashes 位於
[2026-09-06 整合 artifact](benchmarks/2026-09-06-qwen-decode-integration-m2-max/summary.json)。
最新候選的長工具 Decode +4.58% 未達預定 5% gate，停止採用並維持預設關閉。

下文較早的自動測試使用 2026-08-28 的合併工作樹。
較早的 Qwen 支援測試紀錄使用 base commit
`9e7f2f377662e59a44275ff632f5c434adc2d5c4` 和本文件描述的工作樹變更。
較早的 DeepSeek runtime 研究測試紀錄使用 base commit
`997e2ca756d3d6c8ae97aa4df3effcf566ed449f` 加上本文件列出的 dirty research
working tree。
完整 installed model 驗證和正式量測使用 commit
`57e440e48b79fb0399beae7e9965481f74b35b25`。
每個探索性 artifact 另行記錄 working tree source hash。
歷史量測只用來說明演進和決策。

機器可讀的目前結果如下。

- [`benchmarks/2026-08-10-m5-pro.json`](benchmarks/2026-08-10-m5-pro.json)：clean checkout 的單一 repeated prompt 驗證。
- [`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)：目前 checkout 的 R0 多 prompt baseline、route profile 和 Metal capture。
- [`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)：R2a 1,152/2,048-slot 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)：R2b 4/2 read-worker 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)：R2b 4/8 read-worker 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)：R2c 2/4 prefetch-worker TTFT 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)：R2c 2/1 prefetch-worker TTFT 探索性 ABBA。
- [`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)：R3 normal/DSpark 端到端探索性 ABBA。
- [`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)：M1 Prefill file-cache 端到端探索性 ABBA。
- [`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)：M1 Tool-like 4K 獨立探索性 ABBA。
- [`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)：M1 正式效能評估 Wave 5。
- [`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)：M1 五輪正式統計與最終決定。
- [`benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json`](benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json)：P3 `gather_qmm` shader profile gate。
- [`benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json`](benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json)：P4 attention step 探索、記憶體與 expert route 比較。
- [`benchmarks/2026-08-11-d2-top6-profile-m5-pro.json`](benchmarks/2026-08-11-d2-top6-profile-m5-pro.json)：D2 routed expert top-6 dispatch 與 shader profile gate。
- [`benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json`](benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json)：D3 相鄰 Decode CSA row 與 `gather` profile gate。
- [`benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json`](benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json)：Qwen 完整安裝、API、4K prompt cache 與 packaged App 驗證。
- [`benchmarks/2026-09-02-qwen-private-ane-prefill-exploratory-m5-pro.json`](benchmarks/2026-09-02-qwen-private-ane-prefill-exploratory-m5-pro.json)：Qwen private ANE Prefill 的 native component、回退與 4,097-token 探索性 ABBA。
- [`benchmarks/2026-09-04-130402-api-deepseek-v4-flash-0731.json`](benchmarks/2026-09-04-130402-api-deepseek-v4-flash-0731.json)：v1.1.4 DeepSeek mixed SPEED-Bench 1K／2K／8K／16K input、64-token output、每個 input size 三次的探索性 API benchmark。
- [`benchmarks/2026-09-04-132210-api-qwen3-8-flash-next-fp8.json`](benchmarks/2026-09-04-132210-api-qwen3-8-flash-next-fp8.json)：v1.1.4 Qwen mixed SPEED-Bench 1K／2K／8K／16K input、64-token output、每個 input size 三次的探索性 API benchmark。
- [`benchmarks/2026-08-26-hash-exact-prefetch-smoke-m2-max.json`](benchmarks/2026-08-26-hash-exact-prefetch-smoke-m2-max.json)：完整 installed model 的 hash exact prefetch correctness 與 logical-byte smoke。
- [`benchmarks/2026-08-26-adaptive-block-smoke-m2-max.json`](benchmarks/2026-08-26-adaptive-block-smoke-m2-max.json)：storage-aware adaptive block correctness 與 metric-plumbing smoke。
- [`benchmarks/2026-08-26-adaptive-block-calibration-wave1-m2-max.json`](benchmarks/2026-08-26-adaptive-block-calibration-wave1-m2-max.json)：五類 128-token 首輪校準；保存已拒絕的 over-trimming policy 與 output-budget metric 問題。
- [`benchmarks/2026-08-26-adaptive-block-calibration-wave2-m2-max.json`](benchmarks/2026-08-26-adaptive-block-calibration-wave2-m2-max.json)：0.90 高信心完整 block 護欄的五類修正後 ABBA。
- [`benchmarks/2026-08-26-adaptive-block-repair-validation-m2-max.json`](benchmarks/2026-08-26-adaptive-block-repair-validation-m2-max.json)：低信心 storage-score path 與 output-budget cap 的 full-model repair validation。
- [`benchmarks/2026-08-26-adaptive-block-4k256-survey-m2-max.json`](benchmarks/2026-08-26-adaptive-block-4k256-survey-m2-max.json)：五類 R0 4K／256-output selector survey、歷史 fixed hash parity audit 與 paired-wave stop decision。
- [`benchmarks/2026-08-26-adaptive-block-discovery-4k16-m2-max.json`](benchmarks/2026-08-26-adaptive-block-discovery-4k16-m2-max.json)：五類低信心 4K／16-output screening；全部命中 storage score，也全部由正式 fallback 在第一輪停止。
- [`benchmarks/2026-08-26-adaptive-block-storage-4k16-fallback-m2-max.json`](benchmarks/2026-08-26-adaptive-block-storage-4k16-fallback-m2-max.json)：`storage_sentence` 正常 fallback 重跑與 1.301 cost-ratio audit。
- [`benchmarks/2026-08-26-adaptive-block-storage-4k16-no-fallback-m2-max.json`](benchmarks/2026-08-26-adaptive-block-storage-4k16-no-fallback-m2-max.json)：相同 prompt 的 research-only no-fallback control；保存 would-trigger 與 triggered 的差異。
- [`benchmarks/2026-08-26-adaptive-block-random-hex-4k256-no-fallback-m2-max.json`](benchmarks/2026-08-26-adaptive-block-random-hex-4k256-no-fallback-m2-max.json)：低信心 adaptive 長 decode；selector 行為成立，但 output parity 失敗，不能作效能證據。
- [`benchmarks/2026-08-26-fixed-block-random-hex-4k256-no-fallback-m2-max.json`](benchmarks/2026-08-26-fixed-block-random-hex-4k256-no-fallback-m2-max.json)：上述 4K／256 adaptive run 的 fixed control；output hash 不同，因此 comparison 無效。
- [`benchmarks/2026-08-26-random-hex-4k32-normal-layer-major-m2-max.json`](benchmarks/2026-08-26-random-hex-4k32-normal-layer-major-m2-max.json)、[`sequential-prefill reference`](benchmarks/2026-08-26-random-hex-4k32-normal-sequential-prefill-m2-max.json) 與 [`fixed／adaptive exact-token reproduction`](benchmarks/2026-08-26-random-hex-4k32-fixed-adaptive-failed-m2-max.json)：保存四條 32-token 路徑及第一個 token-ID 分歧位置。
- [`benchmarks/2026-08-26-dspark-block-parity-diagnostic-m2-max.json`](benchmarks/2026-08-26-dspark-block-parity-diagnostic-m2-max.json)：從 exact common prefix 比較兩個 sequential target steps 與一個 two-token block；只作 near-tie correctness diagnosis。
- [`benchmarks/2026-08-26-random-hex-4k32-sequential-fixed-oracle-m2-max.json`](benchmarks/2026-08-26-random-hex-4k32-sequential-fixed-oracle-m2-max.json) 與 [`adaptive oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-adaptive-oracle-m2-max.json)：逐 token target verification／replay correctness oracle；兩者都精確匹配 sequential-prefill reference，不是效能結果。
- [`benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json`](benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json)：在同一 exact common prefix 逐層比較 sequential 與 two-token block 的 post-attention、FFN input、router top-k 與 post-layer states；只作 correctness diagnosis。
- [`benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json`](benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json)：拆解 layer 0 HyperConnection、Q／KV projection、RoPE、cache update、attention 與 output projection；只作 correctness diagnosis。
- [`benchmarks/2026-08-27-dspark-layer0-ffn-component-diagnostic-m2-max.json`](benchmarks/2026-08-27-dspark-layer0-ffn-component-diagnostic-m2-max.json)：拆解 layer 0 FFN HyperConnection、router、shared／routed experts、reduction 與 final expand；只作 correctness diagnosis。
- [`benchmarks/2026-08-27-dspark-hybrid-verifier-diagnostic-m2-max.json`](benchmarks/2026-08-27-dspark-hybrid-verifier-diagnostic-m2-max.json)：單一 exact cache state 的 token-attention／block-FFN hybrid diagnostic。
- [`benchmarks/2026-08-27-dspark-hybrid-random-hex-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-hybrid-random-hex-4k32-m2-max.json)：hybrid v1 多 round parity stop。
- [`benchmarks/2026-08-27-dspark-hybrid-v2-discovery-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-hybrid-v2-discovery-4k32-m2-max.json)：hybrid v2 五組 correctness survey；2/5 workloads 分歧。
- [`benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json)：hybrid v3 五組 normal／fixed exact-token gate。
- [`benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json)：hybrid v3 與 hash exact prefetch 的五組 correctness／logical-byte gate。
- [`benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json)：hybrid v3、hash prefetch 與 adaptive selector 的五組 normal／fixed／adaptive gate。
- [`benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json`](benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json)：pinned configs、official reference graph、checkpoint index 與完整 installed DSpark payload 的 MTP-1 feasibility audit；不是效能結果。
- [`benchmarks/2026-08-27-dspark-slots-768-vs-96-storage-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-slots-768-vs-96-storage-4k32-m2-max.json)：完整 installed model 上的 normal + 768/96/96/768 reduced-cache pilot；保存 exact tokens、memory 與 draft-read tradeoff，非正式效能結果。
- [`benchmarks/2026-08-27-dspark-slots-768-vs-96-multilingual-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-slots-768-vs-96-multilingual-4k32-m2-max.json) 與 [`random-hex 4K/32`](benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k32-m2-max.json)：96-slot multi-workload correctness／memory-direction single pairs。
- [`benchmarks/2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json`](benchmarks/2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json)：三 workload composition；保存 reference hash 比對與跨 workload tradeoff，不升格 timing 為正式證據。
- [`benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json`](benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json)：預先宣告的長解碼 gate；exactness、memory 與總 speculative-read 條件通過，但 draft bytes/committed token +83.17% 超過 +50% 停止線。
- [`benchmarks/2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json`](benchmarks/2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json)：research-only `mincore` probe contract ABBA；四條 tokens exact，兩個 probe runs 零 failure／零 unclassified 並完整覆蓋 logical bytes。
- [`benchmarks/2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json`](benchmarks/2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json)：短 hybrid v3 + hash page-residency composition；所有 partitions exact，但 prefetch 100% useful。
- [`benchmarks/2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json`](benchmarks/2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json)：預先宣告的 4K useful/wasted coverage gate；normal/hash exact，logical／resident／nonresident useful/wasted partitions 全部閉合。
- [`benchmarks/2026-08-27-expert-file-cache-bypass-selection-race-pilot-m2-max.json`](benchmarks/2026-08-27-expert-file-cache-bypass-selection-race-pilot-m2-max.json)：保留原先一次預選六個 ranges 的 5/6 pilot；cached read-ahead 讓最後 range 在 just-before-read checkpoint 已 resident，促成 just-in-time selection 修正。
- [`benchmarks/2026-08-27-expert-file-cache-bypass-contract-m2-max.json`](benchmarks/2026-08-27-expert-file-cache-bypass-contract-m2-max.json)：六個 installed expert ranges 的 aligned bypass／cached byte 與 residency contract；6/6 通過。
- [`benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json`](benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json)：explicit bypass policy 下的 4K／32 三波 normal／fixed／adaptive performance gate 與獨立 observer wave；correctness/accounting 通過，兩個候選效能門檻失敗。
- [`benchmarks/2026-08-27-verifier-union-tradeoff-attribution-label-pilot-m2-max.json`](benchmarks/2026-08-27-verifier-union-tradeoff-attribution-label-pilot-m2-max.json)：第一個完整 union tradeoff wave；量測有效但結果 label 過度歸因 QMM，僅保留 traceability，不使用 timing。
- [`benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json`](benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json)：用全新 raw 重跑的 corrected four-wave sequential／grouped／hybrid diagnostic；19 outputs exact，union contract 與 observer accounting 通過，aggregate token-shaped execution cost material。
- [`benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json`](benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json)：atomic target-KV + 三層 DSpark-context 的首次／memory／restart-persistent functional gate；三條 outputs、reuse source／token count、metadata 與 logical-byte stop 全部通過，非正式效能結果。
- [`benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json`](benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json)：native `preadv`／MTLIO bytes／shared／private 1／6／32／128 expert ownership、shared-event visibility、cancellation 與 installed MLX public handoff gate；native exact，runtime integration 因 dependency handoff 缺失而停止，非正式效能結果。
- [`benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json`](benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json)：fixed full／split arenas 的 1／4／8-row `w13` compute 與 `w2` read overlap gate；byte/output exact，4／8-row continuation 通過，非正式效能結果。
- [`benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json`](benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json)：default-off split-slot runtime 的四波 128／32 fresh-process gate；八條 tokens 與 logical-byte／eviction accounting exact，但 request +2.17%、Decode -4.26%，candidate 停止。
- [`benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json`](benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json)：五種 exact 4K prompts 的 42-layer route-union eligibility；reference／trace tokens 與 histogram accounting exact，70%／80%／90% 全部通過 logical-byte continuation gate，timing 不可解讀。
- [`benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json`](benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json)：internal fixed-buffer selected-row runtime 的兩組反向 fresh-process gate；tokens、union rows、adaptive batched bytes 與 request bytes exact，bytes -66.78%、memory -15.78%，但 TTFT +16.87%、p95 +17.02%，candidate 停止。
- [`benchmarks/2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json`](benchmarks/2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json)：五組 128-token fresh-process coherent Markov path gate；actual DSpark draft reconstruction、top-4 transitions、sequential target truth 與 exact hash routes 全部通過，但 0/5 storage-selected paths 維持 baseline acceptance，candidate 停止。
- [`benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json`](benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json)：6,000-label learned-router direct-transfer gate；四個 DSpark taps × top-6／12／24／48 的 route／union accounting exact，但最佳 top-24 只有 17.15% assignment recall 與 11.15% useful rate，training prerequisite stop。
- [`benchmarks/2026-08-27-block-prompt-cache-partial-restart-m2-max.json`](benchmarks/2026-08-27-block-prompt-cache-partial-restart-m2-max.json)：normal format-4 content-addressed prefix checkpoint functional gate；兩個 453-token prompts 共享 442 tokens，restart 精確重用 128 tokens 且與 isolated cold branch output hash 相同，immutable payload／metadata、contract、sharing 與 access sidecar 全部通過，非正式效能結果。
- [`benchmarks/2026-08-27-plan-prerequisite-closure-m2-max.json`](benchmarks/2026-08-27-plan-prerequisite-closure-m2-max.json)：剩餘九個 PLAN directions 的 local feasibility audit；trained/Core ML candidates、approved contracts、current traces、attributable physical-device fields 與 storage-native spec 均不存在，Xcode／system tool availability 分開記錄，全部 scoped stop/defer 與 reopening rules 通過。
- [`benchmarks/2026-09-01-expert-blob-compression-m5-pro.json`](benchmarks/2026-09-01-expert-blob-compression-m5-pro.json)：DeepSeek FP4 與 Qwen MXFP4 的 LZ4／LZFSE microbenchmark、byte hash、nonresident page probe、Metal contention 和 Phase 4 stop decision。
- [`benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json)：DeepSeek top-5 learned-routing candidate 的 10-case Phase 6A quality／safety entry smoke；不是效能結果。
- [`benchmarks/2026-09-01-approximate-expert-drop-component-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-component-m5-pro.json)：Phase 6B fresh-worker next-token logits、route width、logical bytes 和 Peak RSS component gate。
- [`benchmarks/2026-09-01-approximate-expert-drop-4k32-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-4k32-m5-pro.json)：Phase 6C 五種 4K／32 pilot；10 條 token outputs 完全相同，timing 只作探索。
- [`benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json)：Phase 6E 五種 workload、兩輪反向順序、20 個 fresh-process 4K／256 formal gate 和人工 parity review。
- [`benchmarks/prompts/2026-08-26-adaptive-128/manifest.json`](benchmarks/prompts/2026-08-26-adaptive-128/manifest.json)：兩輪五類校準使用的精確 128-token prompt manifest、seed、文字 SHA 與 token SHA；同目錄保存無尾端換行的原始 prompt bytes。
- [`benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json`](benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json)：重建並逐一匹配歷史 R0 token hash 的五類 4,096-token prompt manifest。
- [`benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json`](benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json)：五類低信心 discovery prompts；保存固定 instruction suffix、文字 SHA 與精確 4,096-token SHA。

## 證據標記

| 標記 | 意義 |
| --- | --- |
| 目前驗證 | 本次使用目前程式碼重跑。 |
| 歷史量測 | 舊版本留下的本機結果。沒有使用目前程式碼重跑。 |
| 單元測試 | 驗證局部邏輯。單元測試不代表 full-model 速度或品質。 |
| 未驗證 | 程式可能允許該設定，但本專案沒有完整證據。 |

## 正式量測環境

| 項目 | 值 |
| --- | --- |
| 日期 | 2026-08-10 至 2026-08-11 |
| Commit | `57e440e48b79fb0399beae7e9965481f74b35b25` |
| Mac | MacBook Pro `Mac17,8` |
| SoC | Apple M5 Pro |
| CPU / GPU | 18 cores / 20 cores |
| 統一記憶體 | 64 GB |
| Metal | Metal 4 |
| 模型磁碟 | 內接 APFS SSD，Apple Fabric，FileVault 開啟 |
| macOS | 27.0 build `26A5388g` |
| Xcode | 27.0 build `27A5218g` |
| Python | 3.14.7 |
| MLX | 0.32.0 |
| mlx-lm package | 0.31.3 |
| mlx-lm source | `Blaizzy/mlx-lm@5c10538136b9038b9626c134612b08afc18d697a` |
| Transformers | 5.12.1 |

目前 mlx-lm 來源是固定 fork。
目前 mlx-lm 來源不是相同版本的 upstream release。

## 自動測試

2026-08-27 使用當時的工作樹執行 Swift 測試。
2026-08-28 在 Codex 和 Qwen request 格式修正後，重新執行 Python 測試。
2026-08-28 在兩個分支合併後，重新執行兩組完整測試。
2026-08-30 在 sampled MTP 實作後，重新執行兩組完整測試。
2026-09-01 在 DeepSeek approximate mode 改為預設開啟後，重新執行兩組完整測試。
2026-09-02 在未載入模型支援即時更新進階設定後，重新執行兩組完整測試。
2026-09-03 在 Codex 長 namespace tool name、model metadata 與 GitHub issue #5
persistent prompt-cache snapshot 修正後，重新執行兩組完整測試。
2026-09-03 在 Qwen Codex tool-first、required 結果驗證與單次重試實作後，
再次執行兩組完整測試。
2026-09-04 修正 Responses tool SSE 的重複 parser 誤判，並加入 server log 層級後，
再次執行兩組完整測試。

```sh
make test
PYTHONPATH=runtime:. .venv/bin/python -m unittest discover -s runtime/tests -p 'test_*.py'
```

下表保留 2026-09-04 的 suite 結果；最新 Python 308 項與 Swift 驗證邊界見本頁開頭。

| Suite | 通過 | 失敗 | 略過 |
| --- | ---: | ---: | ---: |
| Swift `DeepSeekRepackTests` | 17 | 0 | 0 |
| Swift `DeepSeekV4SSDAppTests` | 51 | 0 | 0 |
| Python runtime 與 server | 294 | 0 | 0 |
| 合計 | 362 | 0 | 0 |

該輪沒有略過測試。

2026-09-04 的目前工作樹已執行 `make package`。HEAD 是
`f585c2d7fe500d875d7083c92428be18e82afcff`，並包含目前未提交的 Codex namespace
tool Alias、Codex model metadata，以及 GitHub issue #5 immutable persistent checkpoint／
normal format-5 修正，也包含 Qwen Codex tool-first、required 結果驗證與一次受控重試。
本次封裝也包含 Responses tool SSE 只採用完整 parser 結果的修正，以及 Debug／Info／Error
server log 層級。
App 版本和 build 都是 `1.0.0`，使用 ad hoc signature。
ZIP SHA-256 是
`cfaa432d474fafc137b72619192b4b29bc82a4a43242d508eb76a1bb2bd142fb`。
本次沒有 notarize、建立 Git tag 或上傳檔案。

本機 App 與重新解壓的 App 都通過 `codesign --verify --deep --strict`。兩個 App 都包含
英文、簡體中文與繁體中文 localization，且檔案與 source 一致。兩個 App 分別以三種語言
在 sandbox 內啟動；sandbox 禁止讀取專案 `.build` 與各自的 Swift resource bundle。
六次啟動都持續執行，沒有 `L10n` trap，因此確認 localization 先從
`Contents/Resources` 載入。Mach-O dependency audit 沒有找到專案 `.build` 或
`/opt/homebrew` absolute dependency。Packaged `libWhallmANE.dylib` SHA-256 是
`b4c115c5d540e8d3aae8c8e0820ff38df60a35d2742928faf2beb94e44a73cdc`。

2026-09-02 的目前工作樹已執行 `make package`。
HEAD 是 `f585c2d7fe500d875d7083c92428be18e82afcff`，並包含目前尚未提交的
DeepSeek approximate mode、SSD streaming 研究變更，以及 APP model catalog 的
`qwen_next_layer_prefetch=false` contract 修正。此 build 也包含 Chat streaming
delta 50 ms 合併更新修正、預設啟用的 Qwen QSA Grouped-KV runtime 路徑，
以及預設啟用的 Qwen private ANE Prefill projection。
App 版本和 build 都是 `1.0.0`。
本機 App 使用 ad hoc signature。
ZIP SHA-256 是
`f9bc6fd2a8b6ae59883f83d4c232546ee76e0423e320b26436716bb5eb558107`。
本次驗證沒有 notarize、建立 Git tag 或上傳檔案。
本機 App 和重新解壓的 App 都通過 strict deep signature 驗證。
兩個 App 都包含英文、簡體中文和繁體中文 localization。
隔離啟動檢查禁止兩個 App 讀取專案 `.build` 目錄。
隔離啟動檢查也禁止兩個 App 讀取 Swift resource bundle。
兩個 App 都從 `Contents/Resources` 載入三種 localization。
兩個 App 在每種語言下都持續執行，且沒有在 `L10n` 初始化時停止。
Packaged App 包含已簽署的 `Contents/Frameworks/libWhallmANE.dylib`。
該 dylib 的 SHA-256 是
`b4c115c5d540e8d3aae8c8e0820ff38df60a35d2742928faf2beb94e44a73cdc`。
Packaged Python 已使用完整 Qwen installed model 載入 12 個 ANE projection。
`ane_prefill_ratio=0.5` 的狀態是 active，沒有 interface error。
實際分配是 ANE 6,144 channels 和 GPU 6,144 channels。

### Qwen private ANE Prefill 探索性 gate

Native 64×64 projection 已直接執行 private `AppleNeuralEngine.framework`。
相對 CPU 參考的最大絕對誤差是 0.0001411438。

完整 model gate 使用 4,097-token synthetic prompt、1 個 output token、1,152 slots、
關閉 persistent prompt cache，以及 control、ANE、ANE、control 順序。
兩個 ANE run 各執行 48 次 evaluate，沒有 fallback。
四次首個 output token 的 SHA-256 都是
`b531a2e37cafff6750caed50b5ad3881f984a6b74078caa4380b6d3be8fee412`。
Paired median Prefill 是 control 101.37 tok/s、ANE 109.82 tok/s。
第一組受到未清除的 file cache 影響。
反向順序的穩態結果接近相同。
因此這是實作與正確性證據，不是正式速度結論。

2026-08-28 的目前工作樹已執行 `make package`。
該 package 包含 2026-08-28 的 `QwenToolCodec` 修正。
App 版本是 `1.1.0`，build 是 `1.1.1`。
Apple notary submission `1994e20c-5298-46c5-bca6-852e8cfdb1cc`
的狀態是 `Accepted`。
本機 App 和解壓後的 ZIP 都通過 Developer ID、stapled ticket 和
Gatekeeper 檢查。
ZIP SHA-256 是
`da6a638ad738da811f2ab5a62c1b7c0e7d1d99f29475ed59664c9e1c06e0fd50`。
本次驗證沒有建立 Git tag 或 GitHub Release。
兩個 App 都包含英文、簡體中文和繁體中文 localization。
隔離啟動檢查禁止 App 讀取專案 `.build` 目錄。
隔離啟動檢查也禁止 App 讀取 Swift resource bundle。
三種 localization 都直接從 `Contents/Resources` 載入。
App 在每種語言下都持續執行，且沒有在 `L10n` 初始化時停止。

2026-08-27 另在 base commit
`997e2ca756d3d6c8ae97aa4df3effcf566ed449f` 的 storage-aware profiling
與 hash exact prefetch prototype working tree 執行相同兩組測試。Python runtime
與 server 在 sequential verification／replay oracle、layer-wise 與 layer 0 component
diagnostic、MTP-1 contract audit、DSpark 96-slot gate 與 expert-file page-cache
residency proxy、descriptor bypass policy、bypass benchmark gates 與 verifier-union
tradeoff gate、atomic DSpark prompt-context snapshot、native MTLIO gate 與 staged
fixed-arena／runtime decision gates、adaptive prefill route／runtime gates，以及 normal
format-4 block prompt-cache gate與剩餘 prerequisite closure audit 完成後是 173 個通過（包含 selected-row offsets、route
reuse、configuration isolation、actual batched-byte closure、adaptive-prefill early-stop、
coherent path stable-top-k／selector continuation、learned-router recall／union decision，
prefix contract／partial restart／sharing／eviction／MXFP8 snapshot tests，以及 prerequisite
decision-matrix tests）；Swift 是 28 個
通過、0 個略過。
合計是 201 個通過、0 個失敗、
0 個略過。
Swift 測試使用
`DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer`；目前系統
`xcode-select` 指向 Command Line Tools，直接執行時找不到 `XCTest`。
App model-discovery tests 會先使用 APP 預設的
`~/.dsmodel/deepseek-v4-flash-0731.dsv4`，再以專案 `scratch/` 路徑作 fallback；
本次三個 installed-model tests 都使用完整 `~/.dsmodel` model 通過。
本次不是 `build local` workflow，因此沒有為 working tree 重跑 packaging 驗證。
新增 fixture 覆蓋 checkpoint `tid2eid` exact route plan、first-use union、
committed-prefix useful／wasted bytes、speculative scratch `preadv`、resident pin、
replay transaction wiring，以及 scratch expert 不進入主 LFU cache。這些測試不包含
完整 installed model，因此不是 throughput、process disk I/O 或 peak-memory 證據。
新增 adaptive fixture 另覆蓋 1／2／4／max candidate selection、confidence survival、
resident-aware missing bytes、0.90 高信心完整 block 護欄、prefix truncation、
output-budget cap、實際 committed-token 計數、正常 fallback 與 no-fallback
would-trigger 診斷、逐 token verification／replay、oracle 與 hash scratch 互斥，
hybrid default-off／互斥／replay／metrics、每層一次 union acquisition 的 token-shaped
MoE，以及 fixture greedy parity。
MTP-1 audit 的四個新增測試覆蓋 split stage ownership、self-contained stage
判定、official graph edge 檢查，以及不把 Transformers metadata 當成 inference edge。
Candidate-path 的五個新增測試覆蓋 stable top-k tie break、accepted-prefix first mismatch、
storage-weight selection、acceptance regression rejection 與五-workload continuation stop。
Learned-router baseline 的四個新增測試覆蓋 stable top-k、useful／wasted／missed union
closure、最小 eligible k selection 與 zero-recall stop。

測試覆蓋下列關鍵行為。

- repack plan 和 expert blob layout。
- 無效 expert shape 和缺少 tensor 的拒絕行為。
- 8 MiB bounded copy、續傳 receipt、SHA failure 和 repair。
- APP installed model 掃描、server 設定、設定儲存、Keychain 和測試對話儲存。
- Power Saving Mode Slider 的 legend 節點對齊。
- MXFP4 個別與 batched expert output。
- ready expert decode 的 router 順序。
- MXFP8 cache chunk、gather、index 和 persistence round-trip。
- layer-major prefill 和一般 path 的 next-token logits。
- 記憶體與 persistent prompt cache reuse。
- Normal format-5 model／RoPE／KV／attention contract、128-token block-chain identity、
  suffix-divergence partial restart、immutable payload sharing、frequency-aware eviction、
  incompatible-contract／format-4 rejection，以及 mutable `ArraysCache`、MXFP8 active
  remainder／chunk-list snapshot isolation。
- Output token 的 process 累計值。
- DSpark greedy、sampling、verification、replay 和 fallback 邏輯。
- DSpark per-layer expert union、verification／replay logical bytes 和
  expert bytes／committed token。
- Darwin process disk-I/O counter 的 request delta。
- Darwin `mincore` 的 partial-page residency classification、失敗時 unclassified
  accounting、speculative useful／wasted composition，以及 server research opt-in。
- Darwin expert descriptor 的 cached no-op、`F_NOCACHE`／no-read-ahead calls、
  direct-read alignment rejection、CLI/server policy reporting，以及 bypass benchmark gates。
- Verifier union-call accumulation、258／43／43 shape contracts、observer partition 與
  aggregate token-shaped execution-cost decision helper。
- Atomic DSpark target/context cloning、prefix/revision/layer contract、memory namespace、
  format-3 restart round-trip、malformed／partial bundle rejection、eviction／close lifetime、
  resumed prefill alignment、reuse-source metrics 與 full-gate decision helper。
- MTLIO native method/count matrix、byte/event/private-copy/cancellation gate 與缺少 MLX
  external-event dependency 時的 stop decision helper。
- Adaptive prefill threshold boundary、internal configuration isolation、selected-row original
  offsets、不同 prefetch union 的 reuse rejection、router tensor single-use、metric delta、
  successful batched-byte closure、五類 histogram eligibility 與兩-pair early-stop helper。
- Coherent candidate-path stable top-k、accepted-prefix、storage-feasibility selector 與
  five-workload aggregate stop helper。
- Learned-router stable top-k、assignment／union recall、logical-byte closure 與 aggregate
  continuation／training-required stop helper。
- Remaining-PLAN prerequisite matrix：缺少 trained candidate／data contract 時的 scoped
  closure、只有 tool 或 contract 不會虛構 candidate，以及 candidate／trace／counter 各自只
  開啟對應 gate。
- Chat Completions、Responses、Text Completions、tool 和 SSE。
- SPEED-Bench prompt 的精確 input token 數和完整 chat template 尾端。
- Codex Responses request 到 Qwen chat template 的完整 codec 格式轉換。
- 空 model catalog、`0.0.0.0` IPv4 wildcard bind、API model ID、Alias、未知模型和模型載入失敗重試。
- 延遲載入、runtime 重用、關閉後切換、generation request 排隊和累計計數。
- 手動模型載入與卸載、所有 `/v1/*` generation route 和 `/api/status` 的 Bearer 驗證，以及卸載後的 status。
- DeepSeek layer-major Prefill 門檻預設值、邊界、catalog 編碼、舊設定遷移和正整數驗證。
- DeepSeek approximate mode 的 strict request validation、Qwen／DSpark rejection、
  router request-scope restore、prompt-cache mode isolation、persistent-cache exclusion、
  response 和 status actual-mode reporting。
- 模型載入期間的 `/healthz` 和 `/api/status` 回應。
- APP Alias 儲存與遷移、完整 RuntimeConfig model catalog、Loaded 模型辨識、Chat 模型選擇、訊息模型名稱、快速 streaming delta 合併，以及切換頁面期間持續接收 Chat 回覆。
- Server 執行期間的未載入模型進階設定更新、Loaded／Loading 鎖定，以及載入時套用最新 model catalog entry。
- 未載入、載入中與已載入的 status decoding，以及模型切換後清除 Metric 歷史。

2026-08-28 使用目前合併工作樹重跑下列測試：

- `swift test`：61 項通過。
- `PYTHONPATH=runtime .venv/bin/python -m unittest discover -s runtime/tests`：239 項通過。

2026-09-01 完成 expert blob 壓縮與 approximate Phase 6 後重跑 Python tests：

- `PYTHONPATH=runtime:. .venv/bin/python -m unittest discover -s runtime/tests -p 'test_*.py'`：275 項通過。

這些測試多數使用 fixture 或 mock model。
這些測試不取代 full-model benchmark。
多模型生命週期測試使用 fake runtime。
目前工作區沒有兩個完整 installed model。
本次驗證沒有執行真實雙模型切換或 RSS 量測。

2026-08-28 也使用目前 installed Qwen tokenizer 驗證完整格式轉換路徑。
驗證路徑是 Codex Responses request、server normalization、`QwenToolCodec` 和
Qwen `chat_template.jinja`。
驗證內容包含 `instructions`、`developer` message、text content item array、
後置 instruction、沒有 user 的 history、namespace tool、`web_search`、
巢狀 arguments、`function_call_output` 和 Qwen 保留標記。
role sequence 矩陣涵蓋 255 組 server 接受的非 tool history。
installed tokenizer 也通過含 tool history 的 prompt boundary 檢查。
這次驗證沒有載入完整模型權重，也沒有執行 generation。

2026-09-03 另用 Codex CLI `0.152.1` 重現目前的 Responses request。
Request 有 15 個 top-level tools，並包含 Codex 內建 app namespace tools。
修正前，`namespace__name` 最長是 67 個字元；server 在 generation 前以
`tools.13.function.name` 回傳 HTTP 400。
修正後，超過 64 字元的 prompt 內部名稱會使用固定 hash Alias。
完整 Qwen installed model 先產生 `exec_command`，Codex 實際執行一次
`/bin/zsh -lc pwd`，再把結果送回第二個 Responses request；turn 正常完成，沒有
`invalid_tool_call`。
該輪長名稱修正驗證時，Codex 仍因 `/v1/models` 只有 OpenAI 相容的 `data`
array 而顯示 model metadata fallback warning；該 warning 沒有阻止兩輪 tool flow。

2026-09-03 後續依 Codex `rust-v0.152.1` 的 `ModelsResponse` 和 `ModelInfo`
schema，讓 `/v1/models` 保留標準 `data` array，並增加 Codex `models` metadata
array。使用 npm `@openai/codex@0.152.1` 與完整 Qwen installed model 實測：
`GET /v1/models?client_version=0.152.1` 和 `POST /v1/responses` 都回傳 HTTP 200，
Codex 沒有再顯示 model metadata fallback warning，Qwen 正常回覆 `OK`。另一輪要求
Codex 只執行一次 `pwd`；Qwen 產生 tool call、Codex 執行命令並送回 tool result，第二個
Responses request 完成且輸出正確工作目錄。
這是功能驗證，不是效能結果。

2026-09-03 的自動 endpoint 測試另覆蓋 Qwen Codex tool-first 行為：首輪
`tool_choice: auto` 在有 `exec_command` 且沒有 tool result 時改用 `required`；第一次
只輸出狀態文字時不把該文字送給 client，並重試一次；第二次的有效 `exec_command`
call 正常送出。另驗證連續兩次缺少 required call 時以
`tool_choice_not_satisfied` 結束、指定 function 第一次選錯時只重試一次，以及收到
`function_call_output` 後恢復 `auto` 並可輸出最終答案。

這些案例使用 server 測試 runtime，沒有載入完整 Qwen 權重，也不是完整 Codex CLI
端到端 generation 證據。

2026-09-04 使用 npm Codex CLI `0.153.0`、最終封裝內的 Python runtime 和完整 Qwen
installed model，重現使用者的專案 review request。修正前，首輪 `required` response
雖然完整 parser 已取得 `exec_command`，streaming parser 仍誤判，接著把錯誤 delta
送到已完成的 message item；Codex 顯示 `OutputTextDelta without active item` 並留下空白。
只修首輪後，第一個 tool flow 成功，但 tool result 後的 `auto` response 仍會被相同的
重複判定中止。

最終修正讓所有 Responses tool SSE 都直接採用完整 parser 結果。相同 request 的首個
response 產生兩個 command call，Codex 都實際執行；收到結果後的 `auto` response 又產生
兩個 command call，也都實際執行。過程沒有 `invalid_tool_call` 或
`OutputTextDelta without active item`。第三個 response 因完整 review 還會繼續讀取更多
檔案而由測試者手動停止；這項驗證覆蓋原本兩個失敗點，不代表完整 review 品質評估。

同一個最終封裝另以 `--log-level debug` 啟動，送入含 nested object 的 JSON request。
Server Log 完整顯示 compact JSON body 與 HTTP 400 access log。自動測試另驗證預設 Info、
Error 隱藏 HTTP 2xx 並保留 4xx，以及舊的 server 設定缺少 log 層級時會遷移成 Info。

2026-09-03 也重現並修正 GitHub issue #5。根因不是 128-token block boundary，
而是 non-layer-major prefill checkpoint 保留 Qwen `ArraysCache.state` 可變 list 的別名；
後續 final-token evaluation 和 decode 會改寫 checkpoint state，但 metadata 仍保留較短的
token prefix。修正後 checkpoint 先深複製並 materialize state。Regression test 模擬 mutable
state 在 callback 後被改寫，並驗證 restart restore 的仍是 callback 當下值。Normal
persistent format 同時從 4 升到 5；scanner 拒絕 format 4，避免升級後載入磁碟上既有的
poisoned checkpoint。

完整 Qwen installed model 使用隔離的暫存 cache 目錄驗證 37-token prompt。Disk 上同樣
產生 36-token checkpoint；cold request、同 process 第二次相同 request，以及 server
restart 後第三次 request 都完成 64 個 output tokens，文字 SHA-256 都是
`cb67ebf069f0774e140a48a66062f6ae61a84dde0557604d2448404dc0676f63`，沒有
`<|endoftext|>` 或 `<|im_start|>`。這是 correctness 驗證，不是效能結果。

## Installed model 完整驗證

目前驗證執行：

```sh
swift run dsv4-repack verify \
  --model ~/.dsmodel/deepseek-v4-flash-0731.dsv4
```

結果是通過。
驗證會重新計算每個 manifest file 的 SHA-256。

| 欄位 | 結果 |
| --- | ---: |
| checkpoint revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| manifest files | 54 |
| manifest file bytes | 166,884,980,648 |
| main model packed weight bytes | 156,015,738,880 |
| common tensor count | 1,564 |
| main model `common.bin` | 8,846,000,128 bytes |
| DSpark packed weight bytes | 10,862,841,600 |
| file size 驗證 | 通過 |
| SHA-256 驗證 | 通過 |

## 目前 SSD microbenchmark

目前驗證執行：

```sh
swift run dsv4-repack benchmark \
  --model ~/.dsmodel/deepseek-v4-flash-0731.dsv4 \
  --samples 32
```

每列讀取 32 個 expert blob。
每個 expert blob 是 13,369,344 bytes。

`direct` 對 file descriptor 設定 `F_NOCACHE`。
`cached` 先讀取一次相同 expert blob，然後重複讀取該 blob。

| Mode | Workers | GiB/s | Mean read time |
| --- | ---: | ---: | ---: |
| Direct | 1 | 10.76 | 1.16 ms |
| Direct | 2 | 12.94 | 1.90 ms |
| Direct | 4 | 12.89 | 3.69 ms |
| Direct | 8 | 12.95 | 6.87 ms |
| Cached | 1 | 25.21 | 0.49 ms |
| Cached | 2 | 42.87 | 0.57 ms |
| Cached | 4 | 60.40 | 0.79 ms |
| Cached | 8 | 72.79 | 1.28 ms |

本次 direct throughput 在 2、4 和 8 workers 間很接近。
8 workers 的 mean read time 最高。
目前 runtime 保留 4 個一般 read workers 和 2 個 prefetch workers。
這個預設值來自多次端到端量測，不只來自這次 microbenchmark。

## Expert blob 壓縮 microbenchmark

2026-09-01 在 M5 Pro 執行下列命令：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_expert_blob_compression.py \
  --deepseek-model ~/.dsmodel/deepseek-v4-flash-0731.dsv4 \
  --qwen-model ~/.dsmodel/qwen3.8-flash-next.dsv4 \
  --output docs/benchmarks/2026-09-01-expert-blob-compression-m5-pro.json \
  --scratch-directory scratch/expert-blob-compression-2026-09-01 \
  --samples 9 \
  --expert-samples 3 \
  --codecs lz4 lzfse
```

Benchmark 使用 Apple Compression。
專案沒有新增第三方 dependency。
測試包含 64 KiB、256 KiB、1 MiB、4 MiB 和完整 expert blob。
DeepSeek FP4 和 Qwen MXFP4 各測三個 expert 位置。
Qwen 的 4 MiB case 會跨相鄰 expert blob。
完整 expert case 不會跨 expert blob。

每個 baseline 與 candidate 都使用新的 worker process。
讀取使用 `F_NOCACHE`，並關閉 read-ahead。
頁面 residency probe 確認 60 個 case 的所有輸入在讀取前都是 nonresident。
Metal control 使用重複的 2048 x 2048 FP16 MLX matrix multiplication。

| Gate | 結果 |
| --- | --- |
| 解壓縮 byte hash 完全相同 | 通過，60/60 |
| 無 Metal 的 read + decode 改善至少 5% | 未通過 |
| 有 Metal 的 read + decode 改善至少 5% | 未通過 |
| Peak RSS 增加不超過 5% | 通過 |

完整 expert blob 的中位數如下。
時間改善的正值代表較快。負值代表較慢。

| Model | Codec | 儲存減少 | Decode output | 無 Metal 時間改善 | 有 Metal 時間改善 |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek FP4 | LZ4 | 2.43% | 21.10 GB/s | -69.91% | -71.24% |
| DeepSeek FP4 | LZFSE | 7.11% | 1.29 GB/s | -974.74% | -1036.27% |
| Qwen MXFP4 | LZ4 | 2.47% | 26.11 GB/s | +0.36% | -5.08% |
| Qwen MXFP4 | LZFSE | 4.92% | 1.22 GB/s | -564.51% | -591.75% |

Phase 4 gate 未通過。
Runtime 沒有接入壓縮。
Phase 5 ready／compact／cold cache 不執行。

正式 artifact 位於
[`benchmarks/2026-09-01-expert-blob-compression-m5-pro.json`](benchmarks/2026-09-01-expert-blob-compression-m5-pro.json)。

`F_NOCACHE` 和 residency probe 不能證明每次讀取都由 physical SSD 提供。
Metal control 是合成負載，不是完整模型 inference。

## Approximate expert drop entry smoke

2026-09-01 在 M5 Pro 執行：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_approximate_expert_drop.py \
  --model ~/.dsmodel/deepseek-v4-flash-0731.dsv4 \
  --suite docs/benchmarks/prompts/2026-09-01-approximate-entry/manifest.json \
  --protocol research/APPROXIMATE_MODE_2026-09-01.md \
  --output docs/benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json
```

Phase 6A 執行時，candidate 只修改 benchmark process 內的 learned router `top_k`。

| Gate | Exact | Candidate |
| --- | ---: | ---: |
| 品質 smoke | 5/5 | 5/5 |
| 安全 smoke | 5/5 | 5/5 |
| 相對 exact output token | 不適用 | 10/10 case 完全相同 |

Candidate 的 selected expert 理論減少量是 15.50%。
這不是 measured SSD bytes。
Exact 和 candidate 共用一個 loaded runtime 和未平衡 expert cache。
因此，artifact 內的時間只作描述，不能作效能比較。

Phase 6A 通過。
Phase 6B 接著使用分離的 fresh workers 執行 component gate。

### Phase 6B 至 Phase 6E

Phase 6B 的 10/10 next-token top-1 相同。
最大 exact-to-candidate KL divergence 是 0.00009246。
Logical expert bytes 減少 12.73%。
Peak RSS 增加 0.83%。

Phase 6C 使用五種 4K／32 workloads。
10 個 runs 都完成。
五個 pairs 都是 32/32 token 相同。
扣除 42 個 full-layer Prefill reads 後，aggregate Decode logical bytes 減少 11.26%。
這一階段的 timing 只作探索。

Phase 6D 加入 default-off API prototype。
Phase 6D 驗證時，沒有 `approximation` 欄位會使用 exact。
DeepSeek 可以明確指定 `learned-route-drop-lowest-1`。
Qwen、DSpark、malformed mode 和未知 mode 都會拒絕。
Exact 與 approximate prompt cache 已隔離。
Approximate prompt cache 不會持久化。

Phase 6E 的重現命令如下：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_approximate_expert_drop_formal.py run \
  --model ~/.dsmodel/deepseek-v4-flash-0731.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json \
  --protocol research/APPROXIMATE_MODE_2026-09-01.md \
  --safety-artifact docs/benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json \
  --raw-directory /new/approximate-expert-drop-4k256-formal \
  --output /new/approximate-expert-drop-4k256-formal.json
```

Phase 6E 使用兩個 reversed-order waves。
五種 workload 各執行兩個 exact/candidate pairs。
總共有 20 個 fresh-process runs。

| Formal gate | 結果 |
| --- | ---: |
| 10-pair token agreement | 100% |
| Aggregate Decode logical expert bytes | -15.14% |
| Decode throughput change 中位數 | +8.37% |
| Repeated workload p95 change 中位數 | -1.87% |
| Code workload p95 change 中位數 | -9.53% |
| Traditional Chinese workload p95 change 中位數 | -7.95% |
| Mixed math workload p95 change 中位數 | -9.61% |
| Tool-like workload p95 change 中位數 | -22.11% |

所有 machine gates 都通過。
人工 review 確認 10 個 paired texts 逐字相同。
五個 prompt 是重複型壓力 workload。
因此，結果只證明這個固定 suite 的相對 parity。
結果不證明一般能力，也不等於 physical SSD bytes。

Phase 6 完成。
後續使用者已授權把 candidate 設為一般 DeepSeek request 的預設模式。
API、CLI 和 APP request 未指定 mode 時使用 `learned-route-drop-lowest-1`。
明確指定 `exact` 時仍使用完整模式。
Qwen 和啟用 DSpark 的 DeepSeek 維持 exact。

## 目前端到端量測

本次量測使用一個新 `ModelRuntime`。
expert cache 在開始時是空的。
persistent prompt cache 已停用。
作業系統 page cache 沒有清除。
完整 SHA-256 和 SSD microbenchmark 在本次量測前已執行。

prompt 產生方式如下。

1. 重複字串 ` test`。
2. tokenizer 不加入 special token。
3. 保留前 4,096 個 token。
4. tokenizer 把 token 解碼成 prompt。

generation 使用 batch size 1、`temperature=0`、`top_p=1` 和 64 個 output token。
DSpark 已停用。

| 指標 | 結果 |
| --- | ---: |
| Runtime prompt tokens | 4,096 |
| Generated tokens | 64 |
| Wall time | 30.055 s |
| Time to first token | 24.703 s |
| Prefill | 165.81 Tok/s |
| Decode end-to-end time | 5.348 s |
| Decode | 11.78 Tok/s |
| Decode p50 | 74.30 ms/token |
| Decode p95 | 132.20 ms/token |
| Expert bytes read | 156,688,711,680 |
| Expert read timer | 15.648 s |
| Routing synchronization boundary | 2.367 s |
| Batched expert layers | 42 |
| `gather_qmm` calls | 84 |
| Prefetched layer hits | 41 |
| Expert cache hits / misses | 15,802 / 968 |
| Expert evictions | 0 |
| MLX active memory | 20.67 GiB |
| MLX peak memory | 20.67 GiB |
| Token SHA-256 | `90760d2b...fdd9b9` |

完整 token hash 是
`90760d2bc22d73915dfadf7cfa5b064514d4a56d92cf2868c24614d6ddfdd9b9`。

### 量測限制

- 這是一次端到端量測。這個結果不是 throughput 保證。
- `prefill_tokens_per_second` 使用未快取 token 除以 time to first token。
- expert read timer 和 GPU work 可以重疊。兩者不能直接相加。
- routing synchronization timer 包含產生 router index 之前的 pending graph work。
- expert cache hit rate 不包含 full-layer batched prefill 讀取。
- MLX peak memory 不等於 APP 顯示的 process RSS。

指標的完整語意請見[效能與瓶頸](PERFORMANCE.md)。

## R0 多 prompt baseline

R0 使用五種固定 prompt。
prompt 類型是 repeated、code、繁體中文技術文字、mixed math 和 Tool-like。
每種 prompt 都有 4,096、8,192 和 14,363 token。
每個 run 都建立新的 process。
expert cache 和 prompt cache 在開始時都是空的。
persistent prompt cache 和 DSpark 已停用。
每個 run 使用 greedy generation，並產生 256 個 output token。
作業系統 page cache 沒有清除。

下表是每個長度跨五種 prompt 的中位數。

| Prompt token | TTFT | Prefill | Decode | Expert bytes | Expert read | Misses | Evictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 21.611 s | 189.53 Tok/s | 8.00 Tok/s | 473.970 GB | 28.844 s | 24,700 | 23,548 |
| 8,192 | 39.124 s | 209.39 Tok/s | 7.76 Tok/s | 477.393 GB | 27.116 s | 24,956 | 23,804 |
| 14,363 | 67.466 s | 212.89 Tok/s | 7.43 Tok/s | 482.593 GB | 27.931 s | 25,345 | 24,193 |

| Prompt token | MLX peak | Maximum RSS | Peak footprint |
| ---: | ---: | ---: | ---: |
| 4,096 | 22.969 GiB | 22.897 GiB | 23.297 GiB |
| 8,192 | 23.014 GiB | 22.904 GiB | 23.360 GiB |
| 14,363 | 22.939 GiB | 22.918 GiB | 23.267 GiB |

R0 只對每種 prompt 和長度執行一次。
因此，跨 prompt p95 只表示 workload 差異。
跨 prompt p95 不表示重複執行的變異。

### Repeated prompt 的證據邊界

4K repeated prompt 與四種非 repeated prompt 的 Decode 行為不同。

| Workload | Decode | Expert misses | Evictions | Expert bytes |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 12.22 Tok/s | 1,535 | 383 | 164.269 GB |
| 四種非 repeated prompt 的範圍 | 6.44–8.28 Tok/s | 24,307–28,765 | 23,155–27,613 | 468.716–528.316 GB |

Repeated prompt 的 route 分布也較集中。
Repeated 4K 的每個 Prefill 區塊有 84 個 active expert 中位數。
其他 4K prompt 的中位數是 177 至 202。
因此，repeated prompt 不能代表一般 Decode 的 SSD 與 slot 壓力。

### R1 Metal boundary capture

下表來自獨立的 `Metal System Trace` 診斷 run。
capture 使用 64 個 output token。
capture 的 wall time 不是正式效能結果。

| Workload | Prefill GPU busy | Decode GPU busy | Decode GPU idle |
| --- | ---: | ---: | ---: |
| Repeated 4K | 57.6% | 57.8% | 2.604 s |
| Tool-like 4K | 60.7% | 35.5% | 7.968 s |
| Tool-like 14K | 70.2% | 37.1% | 8.114 s |

Tool-like Decode 的 GPU idle 時間較長。
同一批 capture 也記錄較多 expert misses 和 `preadv` CPU samples。
這個結果支持先測 slot 和 worker 設定。

Metal System Trace 沒有提供 shader timing sample。
目前 capture 不能把 GPU time 分配到 attention、routed expert 和 shared expert。
request 結束時間是依最後一個 Python GPU interval 推定。
CPU sampled running time 是跨 thread 的總和，不是 wall time。

完整設定、prompt hash、output token hash、route profile 和限制位於
[`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)。

## R2a slot 探索性 ABBA

R2a 比較 `slots=1152` 與 `slots=2048`。
其他 runtime 設定保持不變。
每個 run 使用新的 process、4K prompt 和 256 個 output token。

每個 prompt 的順序是 A、B、B、A。
A 是 1,152 slots。
B 是 2,048 slots。

| Prompt | Slots | Decode 中位數 | Expert bytes | Misses | Evictions | Routing sync | MLX peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated 4K | 1,152 | 14.42 Tok/s | 164.269 GB | 1,535 | 383 | 8.906 s | 22.969 GiB |
| Repeated 4K | 2,048 | 13.95 Tok/s | 162.264 GB | 1,385 | 0 | 9.226 s | 25.870 GiB |
| Code 4K | 1,152 | 8.25 Tok/s | 468.716 GB | 24,307 | 23,155 | 9.829 s | 22.969 GiB |
| Code 4K | 2,048 | 2.39 Tok/s | 366.868 GB | 16,689 | 14,641 | 111.488 s | 34.125 GiB |

2,048 slots 讓 repeated Decode 降低 3.31%。
2,048 slots 讓 code Decode 降低 71.09%。
Code 的兩個 2,048-slot run 分別是 3.49 和 1.28 Tok/s。
同一組 ABBA 的兩個 1,152-slot run 是 8.46 和 8.05 Tok/s。

Code 的 expert bytes 降低 21.73%。
Code 的 miss 降低 31.34%。
Code 的 eviction 降低 36.77%。
但是，expert read timer 增加 2.01%。
Routing synchronization boundary 從 9.829 秒增加到 111.488 秒。
這個 boundary 包含 pending upstream graph work。
這個 boundary 不是純 router 時間。

Code 的 MLX peak memory 增加 11.156 GiB。
測試前後的 system swap 都是 361.12 MiB。
八個完整 run 的 output token hash 都在相同 prompt 內一致。

R2a 在 code ABBA 完成後觸發停止條件。
後面三種 prompt 沒有繼續執行。
這不是正式五個 workload、五輪 ABBA 結果。
這個探索性結果足以拒絕 2,048-slot 候選設定。
目前預設值維持 1,152 slots。

完整資料位於
[`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

## R2b 2-worker 探索性 ABBA

R2b 使用 code 4K prompt。
R2b 比較 `read_workers=4` 與 `read_workers=2`。
每個 run 使用 1,152 slots、2 個 prefetch worker 和 256 個 output token。
執行順序是 4、2、2、4 workers。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read | External wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.29 Tok/s | 110.58 ms | 172.12 ms | 28.042 s | 57.96 s |
| 2 | 7.94 Tok/s | 116.54 ms | 182.41 ms | 29.435 s | 59.34 s |

2 workers 讓 Decode 中位數降低 4.20%。
兩個配對差異是 -4.12% 和 -4.27%。
Decode p50 增加 5.39%。
Decode p95 增加 5.98%。
Expert read timer 增加 4.97%。

Expert bytes、miss、eviction 和 token hash 都相同。
Routing synchronization boundary 和 peak memory 沒有明顯差異。
測試前後的 system swap 都是 337.12 MiB。

這是一個 workload、一輪 ABBA 的探索性結果。
這個結果足以拒絕 2-worker 候選設定。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

## R2b 8-worker 探索性 ABBA

R2b 使用相同的 code 4K prompt。
R2b 比較 `read_workers=4` 與 `read_workers=8`。
執行順序是 4、8、8、4 workers。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read | External wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.27 Tok/s | 110.88 ms | 173.60 ms | 28.202 s | 58.36 s |
| 8 | 8.29 Tok/s | 110.61 ms | 173.13 ms | 27.875 s | 58.19 s |

8 workers 讓 Decode 中位數增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
Expert read timer 降低 1.16%。
這些差異低於 3% 採用門檻。

Expert bytes、miss、eviction 和 token hash 都相同。
Peak memory 沒有明顯差異。
測試前後的 system swap 都是 337.12 MiB。

這是一個 workload、一輪 ABBA 的探索性結果。
8-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

## R2c 4-prefetch-worker 探索性 ABBA

R2c 使用 code 4K prompt 和一個 output token。
R2c 比較 `prefetch_read_workers=2` 與 `prefetch_read_workers=4`。
執行順序是 2、4、4、2 workers。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 21.956 s | 186.56 Tok/s | 14.478 s | 40 |
| 4 | 22.061 s | 185.67 Tok/s | 12.009 s | 41 |

4 workers 讓 expert read timer 降低 17.05%。
但是，4 workers 讓 TTFT 增加 0.48%。
兩個 TTFT 配對改善是 -1.01% 和 0.05%。
結果低於 3% 採用門檻。

Expert bytes、`gather_qmm` call 和 token hash 都相同。
測試前後的 system swap 都是 337.12 MiB。

這是一個 workload、一輪 ABBA 的探索性結果。
4-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

## R2c 1-prefetch-worker 探索性 ABBA

R2c 使用相同的 code 4K prompt 和一個 output token。
R2c 比較 `prefetch_read_workers=2` 與 `prefetch_read_workers=1`。
執行順序是 2、1、1、2 workers。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 22.037 s | 185.87 Tok/s | 14.531 s | 39.5 |
| 1 | 23.740 s | 172.54 Tok/s | 19.394 s | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
兩個 TTFT 配對改善是 -7.68% 和 -7.78%。
Expert read timer 增加 33.46%。
Prefetch hit 中位數從 39.5 降到 0.5。

Expert bytes、`gather_qmm` call 和 token hash 都相同。
測試前後的 system swap 都是 337.12 MiB。

1-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。
R2 設定 sweep 已完成。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

## R3 DSpark 探索性 ABBA

R3 使用 code 4K prompt 和 64 個 output token。
R3 比較 normal generation 與 DSpark。
執行順序是 normal、DSpark、DSpark、normal。
每個 run 都使用新的 process。
每個 run 都停用 persistent prompt cache。

| Mode | Request | TTFT | 總 throughput | 回報的 Decode | Main model expert bytes | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 31.152 s | 22.072 s | 2.054 Tok/s | 6.94 Tok/s | 239.659 GB | 23.274 GiB |
| DSpark | 41.301 s | 33.722 s | 1.549 Tok/s | 8.32 Tok/s | 316.666 GB | 27.257 GiB |

DSpark 讓 request 中位數增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都完成一個 speculative round。
每個 round 提議 5 個 token，接受 5 個 token，並提交 6 個 token。
兩個 run 的 acceptance 都是 100%。
兩個 run 之後都觸發 fallback。

回報的 Decode 從 6.94 增加到 8.32 Tok/s。
這個指標只涵蓋第一個 token 之後的 boundary。
這個指標不包含 DSpark 增加的 TTFT。
DSpark 也在一個 round 後停止 speculative decoding。
因此，這個 Decode 指標不能推翻 request 與總 throughput 的結果。

Normal run 記錄 42 個 batched expert layer 和 84 次 `gather_qmm`。
DSpark run 的兩個欄位都是 0。
DSpark 的 main model expert bytes 增加 32.13%。
DSpark 的 main model cache miss 增加 230.16%。

四個 output token hash 都相同。
測試前後的 system swap 都是 337.12 MiB。
System compressor pages 從 82,443 增加到 114,791。
作業系統 page cache 沒有清除。

R3 達到停止條件。
研究沒有擴大到五個 prompt 和 256 個 output token。
這個結果足以拒絕目前 DSpark 預設啟用。
DSpark 預設值維持停用。

完整資料位於
[`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

## M1 full-model smoke test

本節到正式結論記錄已移除 prototype 的歷史測試。
目前 runtime 不提供 M1 開關。

M1 smoke test 使用 code 4K prompt 和一個 output token。
當時的 runtime 啟用 `--prefill-no-file-cache`。
Persistent prompt cache 和 DSpark 都停用。

| 指標 | 結果 |
| --- | ---: |
| TTFT | 20.412 s |
| Prefill | 200.66 Tok/s |
| Expert bytes | 149.991 GB |
| Expert read | 11.574 s |
| Batched expert layers | 42 |
| `gather_qmm` calls | 84 |
| Prefetched layer hits | 41 |
| Peak footprint | 20.352 GiB |

Output token hash 是
`2a7185fd600ecd7bd4b998bc7dbf460bd0dd2f1b9529ebdf0f662a48be746642`。
這個 hash 與既有 code 4K one-token control 相同。

這次 smoke test 只驗證 full-model 路徑可以執行。
這次 smoke test 沒有同輪 control。
這次 smoke test 不是效能結果。
後續探索性 ABBA 位於下一節。

## M1 Prefill file-cache 探索性 ABBA

M1 使用 code 4K prompt 和 256 個 output token。
M1 比較預設 file descriptor 與 Prefill 專用 `F_NOCACHE` file descriptor。
執行順序是 Control、M1、M1、Control。
每個 run 都使用新的 process。
Persistent prompt cache 和 DSpark 都停用。
作業系統 page cache 沒有清除。

| Mode | Request | TTFT | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 52.508 s | 21.884 s | 187.17 Tok/s | 8.33 Tok/s | 27.945 s | 23.296 GiB |
| M1 | 50.239 s | 20.221 s | 202.56 Tok/s | 8.50 Tok/s | 24.542 s | 23.297 GiB |

M1 讓 request 中位數改善 4.32%。
M1 讓 TTFT 改善 7.60%。
M1 讓 Prefill throughput 增加 8.22%。
M1 讓 Decode throughput 增加 2.02%。
M1 讓 expert read timer 降低 12.18%。

兩個 request 配對分別改善 4.18% 和 4.46%。
兩個 TTFT 配對分別改善 7.27% 和 7.93%。
四個 output token hash 都相同。
Expert bytes、cache hit、miss 和 eviction 都相同。

M1 的 process peak footprint 增加 1.77 MiB。
這個差異是 0.0074%，可視為沒有變化。
MLX peak memory 也沒有實質變化。
測試沒有出現可重現的 memory pressure。
測試前後的 system swap 都是 337.12 MiB。

M1 沒有達到預先定義的記憶體接受條件。
M1 的記憶體假設尚未成立。
專案不把 M1 採用為記憶體最佳化。
這個 pilot 當時沒有改變 runtime 預設值。

System pagein 計數的配對中位數降低 61.88%。
這個計數包含其他 process，也受到執行順序影響。
這個計數不等於 expert bytes，也不能證明 process 記憶體降低。

外部 wall time 的第一個配對增加 0.83%。
外部 wall time 的第二個配對改善 4.03%。
本次測試只有一種 workload 和一輪 ABBA。
本次測試沒有 bootstrap 信賴區間。
因此，本次測試不能支持正式效能採用。

兩個 runtime request 配對都有超過 3% 的同向改善。
專案當時暫時保留 opt-in prototype。
後續 Tool-like 4K 獨立效能重跑位於下一節。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)。

## M1 Tool-like 4K 獨立探索性 ABBA

這次重跑使用 Tool-like 4K prompt 和 256 個 output token。
執行順序是 Control、M1、M1、Control。
其他設定與 code 4K pilot 相同。

| Mode | Request | TTFT | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 62.574 s | 22.860 s | 179.33 Tok/s | 6.42 Tok/s | 36.282 s | 23.296 GiB |
| M1 | 60.380 s | 20.966 s | 195.38 Tok/s | 6.47 Tok/s | 33.173 s | 23.297 GiB |

M1 讓 request 中位數改善 3.51%。
M1 讓 TTFT 改善 8.29%。
M1 讓 Prefill throughput 增加 8.95%。
M1 讓 Decode throughput 增加 0.75%。
M1 讓 expert read timer 降低 8.57%。

兩個 request 配對分別改善 2.34% 和 4.63%。
兩個 TTFT 配對分別改善 6.31% 和 10.15%。
外部 wall time 中位數改善 3.15%。
四個 output token hash 都相同。
Expert bytes、cache hit、miss 和 eviction 都相同。

M1 的 process peak footprint 增加 0.98 MiB。
MLX peak memory 沒有實質變化。
System swap 在 B2 後增加 2.13 MiB。
System swap 在 A2 後維持相同數值。
System swap 是 system-wide 指標。
本次測試不能把這個變化歸因於 M1。

A1 的 CLI 已完成，並保存完整 metrics 和 time output。
量測包裝指令在 CLI 結束後使用 zsh 保留變數，因此回傳錯誤。
A1 的 runtime 和 process 指標有效。
A1 的 post-run system snapshot 比其他 run 晚。
System pagein 比較只保留為診斷資料。

Code 與 Tool-like pilot 的 request 中位數都改善超過 3%。
兩種 workload 的四個 request 配對都同向改善。
這個結果確認 pilot 等級的效能訊號。

兩種 workload 都只有一輪 ABBA。
這兩個 pilot 當時沒有 bootstrap 信賴區間。
專案因此進入正式多 workload 重複 ABBA。
正式測試監控 swap，並套用既有停止條件。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

## M1 正式效能評估：Wave 1

Wave 1 使用全部五個 R0 4K prompt。
Prompt 順序是 Repeated、Code、繁體中文技術文字、Mixed math、Tool-like。
每個 prompt 先執行一個 Control warm-up。
五個 warm-up 都不納入統計。
每個 prompt 接著執行一輪 Control、M1、M1、Control。
Wave 1 共有 20 個有效量測 run。

第一個 runner 在 Repeated B2 後發生 `awk` 比較錯誤。
Runner 在第一個 Repeated A2 產生 93 個 token 時停止。
專案移出第一次 Repeated 序列的完整檔案和部分 A2 檔案。
正式序列接著從新的 Repeated warm-up 完整重跑。
下表和 artifact 都不包含第一次序列。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 3.03% | 8.23% | 8.98% | -2.93% | 21.91% | 0.72 MiB |
| Code | 4.54% | 8.55% | 9.34% | 1.73% | 12.34% | 4.01 MiB |
| 繁體中文技術文字 | 3.40% | 8.12% | 8.84% | 0.13% | 10.74% | 0.53 MiB |
| Mixed math | 3.69% | 8.24% | 8.98% | 0.57% | 10.85% | 0.20 MiB |
| Tool-like | 2.70% | 7.97% | 8.66% | -0.32% | 8.24% | 0.05 MiB |

四個主要 workload 的 request 改善中位數是 3.54%。
TTFT 改善中位數是 8.18%。
Prefill throughput 增加中位數是 8.91%。
Decode throughput 增加中位數是 0.35%。
Expert read timer 改善中位數是 10.80%。
外部 wall time 改善中位數是 3.26%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
每個 workload 的五個 output token hash 都相同。
Expert bytes、cache hit、miss 和 eviction 在同一 workload 內都相同。

四個主要 workload 的 process peak footprint 增加中位數是 0.36 MiB。
最大的 workload 差異是 Code 增加 4.01 MiB。
這個差異是 0.017%。
MLX peak memory 沒有實質變化。
正式序列的 system swap 從 291.25 MiB 降到 283.25 MiB。
Compressor 使用量從 106,138 pages 降到 102,407 pages。
測試沒有出現 thermal 或 performance warning。

Repeated Decode 在 Wave 1 降低 2.93%。
這個結果高於 2% regression 門檻。
但是，Wave 1 只有兩個 Repeated candidate run。
目前沒有五輪 p95 或 95% bootstrap 信賴區間。
專案必須在後續 wave 繼續檢查這個 regression。

Wave 1 單獨不能支持正式採用決定。
Wave 1 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

## M1 正式效能評估：Wave 2

Wave 2 輪替 prompt 順序。
順序是 Code、繁體中文技術文字、Mixed math、Tool-like、Repeated。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
Wave 2 共有 20 個有效量測 run。

第一次 Code warm-up 完成後，runner 的 Swap parser 使用了 macOS `awk` 不接受的變數名稱。
這個錯誤發生在任何正式量測 run 開始前。
專案保留並排除第一次 warm-up。
Parser 修正並通過相等值與增加值檢查後，完整 Wave 2 從新的 Code warm-up 重跑。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 7.63% | 11.05% | 12.48% | 3.64% | 23.24% | 1.60 MiB |
| Code | 3.78% | 7.83% | 8.49% | 0.93% | 11.34% | -0.72 MiB |
| 繁體中文技術文字 | 3.66% | 8.26% | 9.01% | 0.47% | 10.72% | -1.01 MiB |
| Mixed math | 3.57% | 8.51% | 9.30% | 0.17% | 10.69% | 0.01 MiB |
| Tool-like | 2.61% | 7.77% | 8.43% | -0.36% | 8.04% | 0.63 MiB |

四個主要 workload 的 request 改善中位數是 3.62%。
TTFT 改善中位數是 8.05%。
Prefill throughput 增加中位數是 8.75%。
Decode throughput 增加中位數是 0.32%。
Expert read timer 改善中位數是 10.70%。
外部 wall time 改善中位數是 3.26%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
Runtime Python tree SHA-256 也與 Wave 1 相同。

四個主要 workload 的 process peak footprint 差異中位數是 -0.36 MiB。
最大的 workload 增加是 Repeated 的 1.60 MiB。
這些 process 差異沒有實質變化。
正式序列的 system swap 從 283.25 MiB 降到 275.25 MiB。
System-wide compressor 增加 209.67 MiB。
這個增加主要發生在 warm-up 與 Control 區段。
本次測試不能把 system-wide compressor 變化歸因於 M1。
測試沒有出現 thermal 或 performance warning。

Wave 1 的 Repeated Decode regression 沒有在 Wave 2 重現。
Repeated Decode 在 Wave 2 增加 3.64%。

## M1 正式效能評估：Wave 3

Wave 3 再次輪替 prompt 順序。
順序是繁體中文技術文字、Mixed math、Tool-like、Repeated、Code。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
Wave 3 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 7.93% | 11.87% | 13.45% | 3.25% | 23.77% | -11.73 MiB |
| Code | 4.08% | 7.71% | 8.35% | 1.53% | 11.53% | 0.04 MiB |
| 繁體中文技術文字 | 4.28% | 8.68% | 9.50% | 1.22% | 11.82% | 0.24 MiB |
| Mixed math | 3.34% | 6.97% | 7.52% | 0.84% | 10.72% | -1.48 MiB |
| Tool-like | 2.07% | 8.48% | 9.17% | -1.64% | 8.23% | 0.99 MiB |

四個主要 workload 的 request 改善中位數是 3.71%。
TTFT 改善中位數是 8.09%。
Prefill throughput 增加中位數是 8.76%。
Decode throughput 增加中位數是 1.03%。
Expert read timer 改善中位數是 11.12%。
外部 wall time 改善中位數是 3.35%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
Runtime Python tree SHA-256 也與 Wave 1 相同。

四個主要 workload 的 process peak footprint 增加中位數是 0.14 MiB。
最大的 workload 增加是 Tool-like 的 0.99 MiB。
這些 process 差異沒有實質變化。
正式序列的 system swap 維持 275.25 MiB。
System-wide compressor 增加 244.77 MiB。
本次測試不能把 system-wide compressor 變化歸因於 M1。
測試沒有出現 thermal 或 performance warning。

Tool-like Decode 在 Wave 3 降低 1.64%。
這個差異低於 2% regression 門檻。

## M1 正式效能評估：Wave 4

Wave 4 再次輪替 prompt 順序。
順序是 Mixed math、Tool-like、Repeated、Code、繁體中文技術文字。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
Wave 4 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 7.76% | 10.82% | 12.18% | 4.34% | 23.05% | -2.16 MiB |
| Code | 3.31% | 5.81% | 6.17% | 1.53% | 10.36% | 10.54 MiB |
| 繁體中文技術文字 | 3.56% | 7.95% | 8.63% | 0.48% | 11.09% | -2.28 MiB |
| Mixed math | 3.17% | 7.75% | 8.42% | 0.01% | 10.65% | 13.66 MiB |
| Tool-like | 3.32% | 7.90% | 8.56% | 0.63% | 9.02% | -0.45 MiB |

四個主要 workload 的 request 改善中位數是 3.32%。
TTFT 改善中位數是 7.82%。
Prefill throughput 增加中位數是 8.49%。
Decode throughput 增加中位數是 0.56%。
Expert read timer 改善中位數是 10.51%。
外部 wall time 改善中位數是 2.90%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
Runtime Python tree SHA-256 也與 Wave 1 相同。

四個主要 workload 的 process peak footprint 增加中位數是 5.05 MiB。
最大的 workload 增加是 Mixed math 的 13.66 MiB。
這些 process 差異沒有實質變化。
正式序列的 system swap 維持 275.25 MiB。
System-wide compressor 增加 340.41 MiB。
本次測試不能把 system-wide compressor 變化歸因於 M1。
測試沒有出現 thermal 或 performance warning。

## M1 正式效能評估：Wave 5

Wave 5 的 prompt 順序是 Tool-like、Repeated、Code、繁體中文技術文字、Mixed math。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
有效序列共有 20 個量測 run。

第一次序列在 Tool-like B1 啟動時失去執行 session。
Tool-like A1 已完成，但 B1 沒有完成，也沒有寫入 metrics。
專案保存並排除整個不完整序列。
專案接著從新的 Tool-like warm-up 完整重跑 Wave 5。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 8.36% | 12.44% | 14.24% | 3.89% | 23.84% | -0.34 MiB |
| Code | 2.85% | 6.26% | 6.68% | 0.51% | 10.98% | 17.00 MiB |
| 繁體中文技術文字 | 3.44% | 9.36% | 10.32% | -0.69% | 10.85% | -9.09 MiB |
| Mixed math | 3.68% | 8.83% | 9.66% | 0.14% | 10.88% | -6.55 MiB |
| Tool-like | 2.36% | 6.32% | 6.82% | 0.10% | 9.22% | -3.77 MiB |

四個主要 workload 的 request 改善中位數是 3.15%。
TTFT 改善中位數是 7.58%。
Prefill throughput 增加中位數是 8.24%。
Decode throughput 增加中位數是 0.12%。
八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與前四輪相同。

四個主要 workload 的 process peak footprint 差異中位數是降低 5.16 MiB。
最大的 workload 增加是 Code 的 17.00 MiB。
正式序列的 system swap 降低 8.56 MiB。
System-wide compressor 增加 1,236.50 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)。

### 五輪正式結論

五輪共有 100 個有效量測 run 和 50 個配對。
四十個主要 workload request 配對都改善。
包含 Repeated 後，五十個 request 配對都改善。
所有正確性與執行停止條件都通過。

四個主要 workload 的正式結果如下。

| 指標 | 結果 |
| --- | ---: |
| Request 改善中位數 | 3.53% |
| Request 95% bootstrap 信賴區間 | 3.27% 至 3.76% |
| TTFT 改善中位數 | 7.99% |
| Prefill throughput 增加中位數 | 8.69% |
| Decode throughput 增加中位數 | 0.23% |
| Expert read 改善中位數 | 10.88% |
| 外部 wall time 改善中位數 | 3.25% |

Request 中位數與信賴區間通過效能條件。
主要 workload 的 Decode p95 regression 最大值是 1.71%。
這個結果通過 2% 條件。

Repeated 是 regression control。
Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。

Process peak footprint 的配對中位數增加 0.32 MiB。
配對 p95 增加 19.24 MiB。
配對最大值增加 21.67 MiB。
這些結果通過 256 MiB 條件。
每一輪的 system swap 都沒有增加。

部分 wave 的 system-wide compressor 總量增加。
Snapshot 包含 warm-up、Control、M1 和其他 process。
因此，專案不能把增加歸因於 M1。
但是，這組證據沒有證明 compressor 不增加。

M1 沒有通過全部正式採用條件。
專案拒絕預設啟用 M1。
專案依研究計畫移除 M1 prototype。
目前 Prefill 與 Decode 共用一般 file descriptor。

完整正式結果位於
[`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

## 歷史 full-model 結果

以下結果來自舊版本的本機紀錄。
以下結果沒有在本次文件整理中全部重跑。

### Release 摘要

| Runtime | 測試 | Prefill | Decode | 記錄的峰值記憶體 |
| --- | --- | ---: | ---: | ---: |
| v1.0.3 | 14,000-token Codex request | 180 Tok/s | 6.5 Tok/s | 30 GB |
| v1.0.2 | 4,096-token prompt，1 output | 144.53 Tok/s | — | 15.56 GiB |
| v1.0.2 | 短 prompt，同 runtime 第二次 | — | 6.41 Tok/s | 15.05 GiB |

這些 row 使用不同 prompt、output 長度和 cache 狀態。
這些 row 不能互相比較。
repo 沒有保存三個 release row 的完整 raw artifact。

### Prefill 演進

早期 32-token chunk 測試使用 deterministic repeated prompt。

| Context | KV cache | Time | Expert bytes | 結果 |
| ---: | --- | ---: | ---: | --- |
| 4,096 | BF16 | 124.0 s | 572.9 GB | greedy token 是 ` test` |
| 8,192 | MXFP8 | 264.1 s | 1,340.2 GB | greedy token 是 ` test` |
| 8,192 | BF16 | 264.6 s | 1,377.7 GB | greedy token 是 ` test` |

128-token chunk 和 grouped routed expert path 後的結果如下。

| Context | KV cache | Time | Expert bytes | 結果 |
| ---: | --- | ---: | ---: | --- |
| 4,096 | BF16 | 30.74 s | 219.1 GB | 與早期 greedy token 相同 |
| 8,192 | MXFP8 | 65.47 s | 488.8 GB | 與早期 greedy token 相同 |

14,363-token Tool-like prompt 的歷史 path 比較如下。

| Path | Time | Expert bytes |
| --- | ---: | ---: |
| Chunk-major，512-token step | 386.65 s | 2,290.6 GB |
| Layer-major，512-token step | 105.81 s | 131.5 GB |
| Batched layer-local | 76.54 s | 150.41 GB |

layer-major path 相對 chunk-major path 減少約 94.3% expert bytes。
batched layer-local path 再減少大量個別 matrix operation。
這些數字來自同類 Tool-like prompt，但 repo 沒有保存完整 prompt artifact。

### Prompt cache

一組同 process 的 4K continuation 重用 4,097 / 4,098 prompt token。
time to first token 從 20.89 秒降到 0.56 秒。

歷史 persistent prompt cache v2 的短 prompt artifact 是 6.46 MB，寫入時間是 4.4 ms，
round-trip 測試還原 62 個 quantized pooling cache item。Format v2 現在不再載入，因為它
沒有足以證明 RoPE／KV／attention 相容性的 contract。

目前 normal format 5 自動測試覆蓋完整 contract、content-address block chain、restart、
suffix partial match、immutable sharing、frequency-aware eviction、quantized round-trip、
mutable-state snapshot isolation、format-4 rejection，以及 MXFP8 snapshot isolation。
Installed-model functional gate 另驗證 128-token partial
restart reuse 與 cold branch token parity；詳見下方獨立小節。

### Ready expert decode

五組 4,096-token prompt 和 256-token output 使用配對 A/B。
prompt 類型是 repeated text、code、繁體中文技術文字、English prose 和 mixed math。

| 結果 | 歷史量測 |
| --- | ---: |
| Median Decode Tok/s 改善 | 12.9% |
| Minimum 改善 | 8.1% |
| Maximum 改善 | 14.0% |
| Aggregate decode-time 降低 | 10.5% |
| 相同 256-token hash | 5 / 5 |

2,000-token stability pair 的 baseline 是 10.67 Tok/s。
ready expert decode 是 11.92 Tok/s。
兩個 run 的完整 token hash 相同。

原始本機 artifact 位於被忽略的 `scratch/expert-streaming-experiment`。
這些數字不是 release throughput 保證。

### DSpark

2026-08-09 的舊 runtime 有三組配對量測。
normal decode 中位數是 5.39 Tok/s。
DSpark 中位數是 5.14 Tok/s。
DSpark 在該測試中慢約 4.6%。

該測試早於目前的 round-level cache fork 和 block verification。
該測試不能代表目前 DSpark 速度。

較新的 16-token smoke pair 產生相同 token hash。
DSpark 第一輪接受 0 個 draft token，然後正確 fallback。

目前 checkout 的 R3 探索性 ABBA 使用 code 4K prompt。
R3 的 request 增加 32.58%。
R3 的總 throughput 降低 24.60%。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
因此，runtime 繼續預設停用 DSpark。

### Hash exact prefetch 與 adaptive block smoke

2026-08-26 在 Mac Studio M2 Max 與完整 installed model 上執行 7-token prompt、
8-token greedy output。Hash prefetch 使用 off、on、on、off 順序；每個 run 都使用新
process、停用 persistent prompt cache，但沒有清除 OS page cache。

- 四次 prompt token hash 都是
  `684104d3236aa5b8d7427d90c2e08e0a1e9571f0f400d9902117ed3aa768deb2`。
- 四次 output token hash 都是
  `749bdfdbeeb3f8dbe4777f1725f27901c9881e2517cedb5113143736d4aca696`。
- Fixed 5-token prefetch 的兩次 on run 都讀 79 個 scratch experts，其中 47 個 useful、
  32 個 wasted；useful rate 是 59.49%。
- 相對 off，fixed prefetch 的 replay logical bytes 少 13 個 expert blobs，整個 request
  少 19 個 logical expert blobs，peak memory 則固定增加 1,443,889,152 bytes。
- 兩次 timing 中位數受第一個 cold-ish off run 影響；on-2 對 off-2 的回程 pair 反而
  regression，因此速度結果標記為不確定。

Adaptive follow-up 同樣產生相同 output token hash。兩次 decision 都把 5-token 草稿
選成 1-token prefix；總共 committed 3 tokens。相對 fixed prefetch 的 deterministic
counters，request logical read 少 249 個 expert blobs、target 少 223 個、wasted
prefetch 少 14 個，useful rate 升至 73.13%，但 replay 多 4 個。這個 adaptive 結果
只有一個 warm run，不能採用 request time、process disk bytes 或 peak memory 差異。

### Adaptive block 五類校準與 repair

同日另以 repeated、code、zh_technical、mixed_math、tool_like 五類各 128-token prompt
執行一個 warmup，再對 fixed／adaptive／adaptive／fixed 做 fresh-process ABBA；每次產生
8 個 greedy output token，停用 persistent prompt cache，未清除 OS page cache。

第一輪五類都通過 prompt 與 output token hash parity，但原始 storage-only score
觸發停止條件：跨 workload 中位數的 request time 增加 2.013%、Decode 降低 14.843%，
request logical expert bytes 增加 0.593%。Adaptive 雖在 3/5 類降低 target logical
bytes，卻沒有任何 workload 降低整個 request logical bytes。Code、zh_technical 與
mixed_math 的完整候選預期利用率接近 100%，實際 fixed block 也接受 5/5 tokens，原始
score 仍分別縮成 2、1、1 token。Tool-like 另暴露最後一輪驗證超出剩餘 output budget，
使舊 metric 回報 11 committed tokens，超過實際 8-token output；該 committed 值已標記
無效。第一輪 artifact 保留作為已失敗 policy 的證據。

修正後，完整候選預期利用率至少 0.90 時保留完整 block；每輪最多驗證
`remaining output tokens - 1` 個 draft token，且 committed metrics 只計入輸出預算內
可提交的 token。第二輪五類 ABBA 再次全部通過 hash parity；10/10 adaptive timed runs
都由高信心護欄選擇完整 5-token block，實際全部接受 5、committed 6。五類的 fixed 與
adaptive request／target logical counters 完全相同；跨 workload request-time 差異中位數
是 +0.0495%，Decode 差異中位數是 +0.2164%。這只證明已消除首輪一致性退化，不能視為
加速，因 candidate 在這五類工作負載上等同 control，且每類只有兩個 timed observations。

另一組 7-token 低信心 prompt 的 A/B/B/A repair validation 證明 storage-score path
仍可執行：兩個 adaptive run 都有 2 個 `storage_score` decisions、0 個高信心 decision，
最後一輪各裁掉 2 個超出 output budget 的 draft token，並正確回報 3 committed tokens。
四次 output hash 相同；adaptive deterministic request logical bytes 少 11.02%，target
logical bytes 少 33.09%。OS page cache 未控制，因此不採用這組 timing。

原始數值、source hash 與證據限制位於
[`第一輪`](benchmarks/2026-08-26-adaptive-block-calibration-wave1-m2-max.json)、
[`第二輪`](benchmarks/2026-08-26-adaptive-block-calibration-wave2-m2-max.json)
與
[`repair validation`](benchmarks/2026-08-26-adaptive-block-repair-validation-m2-max.json)。

### Adaptive block 五類 4K／256 decision survey

同日使用既有 generator 重建五類 4,096-token prompts。五個 prompt token hash 都與
2026-08-10 R0 artifact 相同。每個 workload 以 fresh process 執行一次 adaptive
256-token greedy decode；沒有 same-prompt warmup，persistent prompt cache 停用，OS
page cache 未清除。

五個 output token hash 與 generated-token count 都匹配歷史 fixed R0 baseline。
五類合計 54 個 adaptive decisions，全數是 high-confidence full-block decisions；
storage-score decision 是 0。Selected-length histogram 是 `[[1, 1], [5, 53]]`，其中
1-token decision 是中文最後一輪 output-budget cap 後的完整 block。四個 workload
最終 fallback；中文 workload 完成 43 rounds 而未 fallback。

這輪通過 correctness、counter 與 selector-behavior gate，但沒有 control/candidate
策略差異。Artifact 因此記錄 stop decision，沒有接著執行 fixed／adaptive A/B/B/A；
單次 request time、process disk bytes 與 peak memory 不作效能結論。原始數值與 audit
位於
[`4K／256 survey`](benchmarks/2026-08-26-adaptive-block-4k256-survey-m2-max.json)。

### 低信心 discovery、fallback control 與 parity stop

為建立 R0 survey 缺少的 storage-score workload，同日以固定 instruction suffix 產生
`storage_sentence`、`creative_metaphor`、`balanced_choice`、`multilingual_choice` 與
`random_hex` 五個精確 4,096-token prompts。16-output screening 的 5/5 runs 都選擇
1-token prefix，也都在第一個 speculative round 觸發正式 fallback；
`balanced_choice` 在 5 tokens 時 EOS。

`storage_sentence` 的正常 fallback 重跑保持相同 16-token hash。該 round 的 target step
是 0.643 秒、speculative path 是 0.837 秒，cost ratio 是 1.301，支持正式 runtime
繼續 fallback。明確停用 fallback 的研究控制保持相同 output hash，並完成 8 rounds；其中
5 rounds 的 `fallback_would_trigger` 為 true，但 `fallback_triggered` 維持 false。這只驗證
控制與診斷，不是部署設定或速度證據。

`random_hex` 的 research-only no-fallback 4K／256 adaptive run 產生 256 tokens 與
86 個 decisions：66 次走 storage score、20 次走 high-confidence full；selected length
1／2／4／5 分別出現 55／6／1／24 次。它建立了需要的 selector 分布，但 fixed control
與 adaptive 的 output token hash 不同，所有 timing、logical bytes 與 memory comparison
因此標記無效。

32-token reproduction 保存 exact generated token IDs。相對 sequential-prefill normal
reference，layer-major normal 在 index 6 首次分歧，fixed 與 adaptive block 都在 index
15 首次分歧；fixed 和 adaptive 在 index 17 彼此分歧。共同 prefix diagnostic 顯示第一個
block position 的 sequential／block top token 相同；第二個 position 的 sequential top-2
margin 是 0.125，而 block top logits 打平，最大 absolute logit delta 是 1.9375。這只支持
near-tie 對 block shape／低精度數值敏感的解釋，不建立全 workload 安全 tolerance。

Parity stop gate 因此失敗，adaptive calibration 停止。沒有採用單案例的 margin heuristic；
DSpark、hash exact prefetch 與 adaptive block 保持預設關閉，fallback 保持預設啟用。完整
失敗 trace 與 diagnostic 位於
[`4K／32 reproduction`](benchmarks/2026-08-26-random-hex-4k32-fixed-adaptive-failed-m2-max.json)
和
[`block diagnostic`](benchmarks/2026-08-26-dspark-block-parity-diagnostic-m2-max.json)。

### Sequential verification oracle

同日新增預設關閉的 `--dspark-sequential-verification` correctness oracle。它保留
round-level cache fork，但 target verification 每次只執行一個 token；若 draft 被拒絕，
committed prefix 也逐 token replay。Oracle 禁止同時使用 hash exact prefetch，讓這個
實驗只比較 verifier shape，不混入 speculative scratch 的 expert execution path。

以相同 `random_hex` 4K／32 prompt、停用 fallback 並停用 layer-major prefill：

| 路徑 | Rounds | Storage-score decisions | 與 sequential reference exact IDs | 正常 policy would fallback |
| --- | ---: | ---: | ---: | ---: |
| Fixed sequential oracle | 13 | 0 | 通過 | 11 |
| Adaptive sequential oracle | 17 | 16 | 通過 | 5 |

三條路徑的 32-token hash 都是
`e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`。
Adaptive oracle 的 selected-length histogram 是 `[[1, 15], [2, 1]]`，證明 selector
仍有實際作用；exact parity 不是因 candidate 回到 fixed block。Fixture 另驗證 oracle
fork 的 logits／hidden 與逐 token target steps 完全相同，且 rejected committed prefix
也使用逐 token replay。

這項結果把該 workload 的 correctness boundary 縮小到 block-shaped verification，
但 oracle 移除了 batching 與 expert-union verification 的主要效能價值。No-fallback 只為
完成 trace；request time、Decode、bytes 與 memory 都不作效能比較。原始數值位於
[`fixed oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-fixed-oracle-m2-max.json)
與
[`adaptive oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-adaptive-oracle-m2-max.json)。

### Layer-wise block verification diagnosis

2026-08-27 以相同 common prefix、anchor index 13 與兩個 target tokens 重跑完整
43-layer main model。診斷先執行未攔截的 sequential／block verifier，再攔截每層的
post-attention、FFN input、router top-k 與 post-layer state；86 個 sequential layer
positions 與 43 個 block layers 全部捕獲，攔截前後兩條路徑的 logits 都逐值完全相同。

兩個位置的 16,384-element broadcast embeddings 都完全相同。最早的非 exact boundary
出現在第 0 層 `LocalAttention` 後：position 0 有 112 個元素不同、最大 absolute delta
是 0.0009765625；position 1 有 46 個元素不同、最大 delta 是 0.00048828125。第 0 層
router 的 expert IDs、順序與 selected weights 在兩條路徑都完全相同，因此 routed MoE
selection 不是最早觀測到的差異來源。

誤差累積後，position 1 的 router 在第 11 層首次只改變順序，第 12 層首次改變 expert
set；position 0 到第 16 層才同時首次改變順序與 set。第 42 層 post-layer 最大 delta
分別達 3.0 與 3.75，最終 logits 精確重現既有 diagnostic：position 0 top token 相同，
position 1 的 sequential／block top token 不同。這項單案例證據把最早觀測邊界縮到
第 0 層 attention branch，並顯示 learned router 會在後續放大差異；它尚未分辨
`LocalAttention` 內的哪一個 projection、attention kernel 或 output path 是單獨原因。
原始逐層數值與 instrumentation audit 位於
[`layer parity diagnostic`](benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json)。

### Layer 0 component diagnosis

同日使用相同 exact prefix、anchor index 13 與 two-token block，逐段攔截 layer 0 的
HyperConnection、Q／KV projection、RoPE、rotating cache、attention output 與 output
projection。診斷先保存未攔截 reference，再要求兩次 sequential layer 0 call 與一次
block call 全部捕獲；攔截後的 sequential／block logits 都逐值精確匹配各自 reference。

兩個位置的 input hidden、HyperConnection collapsed value、attention norm、`wq_a`、
`q_norm`、`wkv`、KV norm 與新 KV RoPE 都 exact。嚴格 stage order 最先在
HyperConnection `post`／`combine` 出現極小 shape-dependent 差異：`post` 最大 delta
分別是 6.821210263296962e-12 與 2.9558577807620168e-12。`wq_b` 在 exact `q_norm`
input 下也非 exact，position 0／1 分別有 3／1 個值不同，最大 delta 是
0.00006103515625／2.9802322387695312e-8。KV path 在捕獲邊界維持 exact。

Sequential cache 走兩次 one-token in-place update，offset 由 4109 增至 4111，兩次 mask
都是 `None`；block 走一次 two-token concatenate update，使用 2x129 float32 mask。
Raw fetched cache shape 分別是 1x1x128x512 與 1x1x129x512，不能直接逐值比較；轉成
temporal order 後，共同 128-token suffix 的 65,536 個值 exact。Attention output 最大
delta 是 0.001953125／0.0001220703125；output projection 後的 `post_attention` 完整
重現逐層 artifact 的 112／46 個不同值與 0.0009765625／0.00048828125 最大 delta。
最終 logits 也重現 position 0 top token 相同、position 1 top token 不同。

這項單案例證據排除 input、collapsed attention value、attention norm、KV projection 與
共同 temporal cache suffix 是最早差異來源；它沒有證明 HyperConnection、Q projection、
mask/cache layout 或 SDPA 中任何單一 kernel 是獨立原因。Artifact 明確標記
`formal_performance_result=false`，沒有 timing 或 I/O 結論。原始 component values、source
hash、cache audit 與 evidence limits 位於
[`layer 0 attention component diagnostic`](benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json)。

## Hybrid verifier 與 composition 驗證

Hybrid v1 在單一 exact cache state 恢復兩個 sequential top tokens，但多 round
`random_hex` 4K／32 在 index 15 分歧。Layer 0 FFN diagnostic 隨後確認 router
IDs／scores、routed selected outputs、routed reduction 與 MoE output exact，最先保留的
差異位於 FFN HyperConnection `post`／`combine`。Hybrid v2 將這些 boundary
token-shaped 後仍只有 3/5 workloads exact。

Hybrid v3 讓完整 target math 維持 one-token shape，但先收集一層內的所有 route IDs，
只 acquire 一次 expert union。五組 4,096-token discovery prompts 的結果如下。

| Workload | Normal／v3 | Normal／v3+hash | Normal／adaptive | Output tokens |
| --- | --- | --- | --- | ---: |
| `storage_sentence` | exact | exact | exact | 32 |
| `multilingual_choice` | exact | exact | exact | 32 |
| `random_hex` | exact | exact | exact | 32 |
| `creative_metaphor` | exact | exact | exact | 32 |
| `balanced_choice` | exact | exact | exact | 5 |

Hybrid v3 fixed runs 的 expert-union assignment reuse rate 是 37.08%–46.90%。加入 hash
prefetch 後，五組 `useful + wasted = bytes_read`，useful rate 是 23.53%–56.23%。加入
adaptive 後共有 63 decisions，selected length 分布 `[[1,56],[2,4],[3,1],[4,1],[5,1]]`；
useful rate 是 65.22%–91.31%，union reuse rate 是 15.59%–23.98%。這是 correctness、
selector behavior 與 logical-byte accounting evidence。

所有 survey 都是 fresh process、persistent prompt cache 關閉、temperature 0、top-p 1、
無 warmup、OS page cache 未控制，並以 `--no-dspark-fallback` 完成 trace。每模式只有一
run，`formal_performance_result=false`；timing、process disk bytes、memory 與 power
不得作 adoption 結論。Hybrid、hash prefetch 與 adaptive 全部仍預設關閉。

## DSpark 96-slot reduced-cache gate

三組 4K／32 workload 的 normal／768／96 token IDs 全部 exact，且都匹配既有 hybrid
v3 reference hash。96-slot peak-memory change 是 -0.40% 至 -3.31%，draft-read change
是 +16.56% 至 +27.23%，all speculative-read change 是 +0.216% 至 +0.515%。只有
`storage_sentence` 有 ABBA；其餘是未重複 single pairs。

下一個 `random_hex` 4K／128 gate 在執行前固定五條條件：exact tokens、peak memory
至少 -1%、draft bytes/committed token 不超過 +50%、all speculative bytes/committed
token 不超過 +2%、request time 不超過 +5%。Normal、768、96 的 128-token IDs 全部
exact；96 slots 的 peak memory -7.35%、all speculative bytes/committed +0.91%、request
time -0.14%，但 draft bytes/committed +83.17%，並產生 470 evictions。候選因此停止，
預設維持 768 slots。所有 timing 仍受 single pair 與未控制 OS page cache 限制。

## Expert-file page-cache residency proxy

Default-off probe 使用 `mincore` 在 expert `preadv` 前分類 VM page residency。`code`
128/32 的 control/probe/probe/control 四條 output exact；兩個 probe runs 各分類全部
90,323,288,064 logical bytes，6,756 calls、零 failure、零 unclassified。Median
resident／nonresident fractions 是 32.10%／67.90%。Probe 對 request time 的 +3.89% 與
reported decode throughput 的 -10.77% 只表示 observer overhead。

4K `balanced_choice` normal/hash single pair 進一步覆蓋非零 useful 與 wasted 分支。
兩條 output exact；useful logical/resident/nonresident bytes 是
802,160,640／120,455,168／681,705,472，wasted 是
2,607,022,080／989,462,528／1,617,559,552。Logical useful rate 23.53%，nonresident
useful rate 29.65%。所有 main、draft、prefetch、useful／wasted identities 都閉合，零
failure／unclassified。這仍只是 page-cache-miss proxy，不是 physical SSD bytes；OS
page cache 未控制，timing 不作採用證據。

## Expert-file cache-bypass contract 與 candidate gate

本機帳號執行 `/usr/sbin/purge` 回傳
`Unable to purge disk buffers: Operation not permitted`，所以沒有將未控制的 run 標成
cold。Research-only `expert_file_cache_policy=bypass` 改用每個 expert descriptor 的
`F_NOCACHE=1`、`F_RDAHEAD=0`，並驗證 APFS 4,096-byte direct-read alignment；預設仍是
`cached`。

Installed-range gate 以 just-in-time residency recheck 選出 layer 42 experts 12、15、
16、19、20、24。六個 13,369,344-byte ranges 都在 bypass 前 fully nonresident；
bypass／cached SHA-256 exact，bypass 後 resident bytes 仍為 0，cached control 後全部
13,369,344 bytes resident。六個 bypass reads 的 process disk delta 也各是
13,369,344 bytes。Contract artifact SHA-256 是
`fd1a9eb680267321c264c00d8d397983a115e85bf0692bbc7b823304f2746f71`。

Full-model gate 固定 `random_hex` 4,096-token prompt、32 greedy tokens、fresh process、
persistent prompt cache off、layer-major off、fallback off 與 bypass policy。Performance
採三波 `normal/fixed/adaptive` Latin square，共九 runs；另三個 probe-on observer runs
不計 timing。12 條 token IDs exact，token SHA-256 都是
`e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`，所有 observer
partitions 零 failure／unclassified 並完整閉合。

| Mode | Request median | Decode median | Peak memory median | Process disk bytes/token median |
| --- | ---: | ---: | ---: | ---: |
| Normal | 85.864 s | 3.032 Tok/s | 26.799 GB | 9.615 GB |
| Fixed hybrid + hash | 101.487 s | 1.223 Tok/s | 29.227 GB | 10.878 GB |
| Adaptive hybrid + hash | 90.171 s | 2.256 Tok/s | 29.374 GB | 9.674 GB |

Fixed 的 request／Decode 變化是 +18.19%／-59.67%，adaptive 是
+5.02%／-25.58%。兩者都未達 request 至少 -5%、Decode 至少 +5% 的預先門檻；fixed
另超過 process disk bytes/token +5% 上限。Adaptive speculative bytes/committed token
比 fixed 少 42.96%，但仍不足以通過端到端 gate。決策是拒絕這兩個 exact candidates，
不變更任何預設。Artifact SHA-256 是
`0810de125a00b5edfe3720b3170226318dd6ef00d0245c5d6f44b436813ca36e`；它標記
`formal_performance_result=false`，不能推廣成多 workload 或 system-wide cold-cache
結果。

## Verifier expert-union tradeoff gate

Runtime 新增 `dspark_verification_expert_union_calls`，直接累計 target
verification／replay 的 expert acquisitions。Gate 固定 128-token `repeated` prompt、
8 greedy outputs、bypass policy、fallback／hash／adaptive off，並以四波
normal／sequential／grouped／hybrid Latin square 加三個 observer runs。第一輪完整
artifact 的量測有效，但 decision label 過度歸因 routed QMM；它被保留為 pilot，隨後
以修正的 aggregate execution-shape label 和全新 raw directory 重跑全部 19 runs。

Corrected run 的 19 條 outputs 全部匹配 SHA-256
`03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3`。每個 DSpark
run 都是 one round、5 proposed、5 accepted、6 committed、zero replay；observer
main／draft partitions 全部閉合。

| Mode | Calls | Assignments | Union experts | Misses | Target bytes | Verification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential | 258 | 1,548 | 1,548 | 195 | 2,607,022,080 | 0.8524 s |
| Grouped | 43 | 1,548 | 660 | 146 | 1,951,924,224 | 0.5331 s |
| Hybrid v3 | 43 | 1,548 | 675 | 171 | 2,286,157,824 | 1.0001 s |

Hybrid 的 calls／union experts／target bytes 相對 sequential 分別是 -83.33%／-56.40%／
-12.31%，但 verification time +17.34%；相對 grouped +87.59%。One-per-layer union
contract 通過，但 aggregate token-shaped target cost 超過預先宣告的 20% material
line。Grouped 與 hybrid 的 internal unions 不同，且兩條 path 同時改變多個 execution
shapes，所以不能把差距宣稱為 QMM-only。Corrected artifact SHA-256 是
`9d6a12028b85bdf492e82292a651f5a203c30d4fedd29ab1255dd52a4c2073a8`；pilot SHA-256
是 `1f1ef076b960ff87eb43a313bb74e29cd1b918d3f20727e150a8c70c8e52a700`。

## Atomic DSpark prompt-context snapshot gate

P2 integration candidate 新增 default-off `dspark_prompt_cache`，並以獨立 format 3 entry
同時保存 target KV、三個 DSpark context states、token prefix、revision 與 target layers。
當時的 normal format 1／2 與 DSpark format 3 scanners／eviction 分離；normal namespace
當時已升級為 format 4，DSpark gate 的 format-3 artifact 與 contract 不變。Full-model gate 使用
128-token `repeated` prompt、8 greedy outputs、hybrid v3、fallback/hash/adaptive off、
persistent on、bypass policy 與全新隔離 cache directory。

| Request | Source | Reused | Main logical bytes | Draft logical bytes | Combined logical bytes | Request | TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime A first | `none` | 0 | 28,957,999,104 | 454,557,696 | 29,412,556,800 | 6.979 s | 5.750 s |
| Runtime A memory | `memory` | 127 | 2,286,157,824 | 0 | 2,286,157,824 | 1.214 s | 0.117 s |
| Runtime B restart | `persistent` | 127 | 9,666,035,712 | 454,557,696 | 10,120,593,408 | 2.814 s | 0.751 s |

三條 output token SHA-256 全部是
`03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3`。
兩個 reuse run 的 main／draft components 都不高於首次；combined logical bytes 分別
降低 92.23% 與 65.59%，通過至少 -50% gate。Format-3 safetensors 是 12,985,677 bytes，
metadata 是 850 bytes；mode、format、revision、layers、127 tokens 和 data path 全部 exact。
Artifact SHA-256 是
`2694fb9bf1f77b15a2547babaa961854c3ec487528ff21013ebf0e964e350a25`。

這是三-request functional integration gate。Runtime A 的 memory run 保留既有 expert
slots，runtime B 則重新建立 expert caches；prompt-cache acquisition/load 也不在 runtime
TTFT 內。因此 timing 只作描述，不能宣稱採用或淨加速。功能、DSpark、hybrid 與 bypass
都維持預設關閉。

## Normal block prompt-cache partial-restart gate

Format 4 normal entries 以 checkpoint revision、完整 model-config hash、顯式 RoPE、KV／
index formats、attention mode 與 128-token parent chain 建立 content key。Non-layer-major
prefill 保存 bounded first／final checkpoints；reuse counter 與 access time 位於獨立
sidecar，immutable data／metadata 不因命中改寫。Format 1／2 不再載入，DSpark format 3
仍在獨立 namespace。

Fixture gate 先通過 contract/key 變動、相同 prefix 單一 payload、suffix 分岔後 restart、
frequency-aware eviction、incompatible KV contract rejection、namespace isolation 與 MXFP8
active-remainder isolation。第一次 installed-model run 隨後在 restart 捕獲 mutable
`_chunks`／`_index_chunks` list snapshot bug；該次沒有寫 artifact。改成複製 collection 並
加入 completed-chunk regression 後，才從全新 cache directory 重跑。

修正後兩個 453-token prompts 有 442 個共同 token。Runtime B 重啟只匹配 first 128-token
checkpoint；同一 branch 的 Runtime C 完全停用 persistent cache。

| Request | Reused | Output token/hash | Logical expert bytes | Request | TTFT |
| --- | ---: | --- | ---: | ---: | ---: |
| Runtime A cold seed | 0 | 36,363／`6c094c63…20d51` | 146,902,351,872 | 28.487 s | 28.466 s |
| Runtime B restart branch | 128 | 36,363／`6c094c63…20d51` | 125,070,213,120 | 23.948 s | 23.918 s |
| Runtime C isolated cold branch | 0 | 36,363／`6c094c63…20d51` | 146,848,874,496 | 29.537 s | 29.536 s |

Restart／cold branch output exact。7,002,850-byte shared data 與 2,148-byte metadata 在 hit
前後 SHA 相同、同 key 只有一份，sidecar reuse count 0→1，所有 contract 與 write
criteria 通過。Artifact SHA-256 是
`678034d8a68278f87a9071fa65f662140a8479e9d4c591e0f189c5db3d367620`。

這是 functional integration gate，不是 balanced performance result；persistent load 不在
runtime metrics 內，OS cache 也未平衡。它證明 cumulative immutable checkpoint 的 exact
partial reuse，不證明 per-layer KV delta dedupe、concurrent requests 或 cold-prefill 改善。

## Remaining PLAN prerequisite-closure audit

Local audit 讀取同一 installed manifest 與 checkout，將九個尚需 training／platform／
lower-level instrumentation 的方向逐項分類。Manifest 有三層 block-5 DSpark，但 separately
named dense drafter tensors 是 0。Repo 中 trained candidate、Core ML model、approved
training／approximate-mode contract、current `.trace`、attributable physical-device-byte
field 與 storage-native project spec 也都是 0；現有 learned-router evidence 是 6,000-label
untrained direct-transfer feasibility artifact。

完整 Xcode 的 `xctrace`／`metal`／`coremlcompiler` 和系統 `fs_usage`／`iostat`／
`powermetrics` 都存在；`.venv` 的 `coremltools`／PyTorch／datasets／accelerate 都不存在。
Audit 明確把工具可用性與 candidate／measurement readiness 分開。九個 directions 全部有
`current_scope_closed=true`、不同 decision state 與 reopening requirement；六項 audit
criteria 全通過。Artifact SHA-256 是
`213087121a7dc7b0c39d1e15066c92f470dfaf6c5326dfcf9889a6d8a6fafcfa`。

這個結果不宣稱 ANE、trained drafter、router fine-tune、sub-FP4、native loop 或
storage-native model 已實作，也不宣稱那些方向不可能。它只證明在目前 checkout／installed
model 下沒有合法、可驗證的 upstream candidate，並固定未來重新開啟每列所需的證據。

## Native MTLIO ownership/copy-path gate

2026-08-27 在 Mac14,13／Apple M2 Max 64 GiB、macOS 26.6.2、Xcode 26.6／SDK 26.5
執行 standalone Swift probe。Installed model revision 是
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`，expert blob 是 13,369,344 bytes。
四個方法各跑 1／6／32／128 ranges，共 16 rows、8,930,721,792 candidate bytes。

Native gate 的六項 criteria 全部通過：method/count matrix 完整、所有 completed bytes
exact、shared/private external shared-event GPU visibility exact、shared direct CPU view
exact、private CPU validation copy 明確標記，以及 cancellation destination admission
safe。Cancel command 實際回傳 `cancelled`，candidate digest 是 `null`，destination
沒有 admission。

| Count | `preadv` GiB/s | MTLIO bytes | Shared | Private |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.824 | 4.963 | 5.021 | 2.995 |
| 6 | 5.968 | 5.888 | 5.861 | 3.553 |
| 32 | 5.960 | 5.949 | 6.007 | 5.636 |
| 128 | 6.054 | 6.078 | 6.041 | 6.026 |

MLX 0.32.0 public audit 的 raw-pointer／DLPack resource candidate 存在，但 external
`MTLSharedEvent` import 不存在；observed `from_dlpack` provider call 也沒有 stream／
event argument。因此 resource + dependency conjunction 失敗，decision 是
`stop_no_supported_mtlio_to_mlx_dependency_handoff`。32／128 aggregate 沒有 10%
directional improvement，六 expert p95 另有超過 50% regression。

每列前 `mincore` 至少觀測 99.9885% selected bytes nonresident，但 OS page cache 沒有
purge、沒有 repeated balanced waves，也沒有 target-model compute overlap。Artifact
的 `formal_performance_result` 是 `false`；這只驗證 native mechanism 與 integration
stop，不是 physical SSD performance。Artifact SHA-256 是
`891d2b9635157837c6107f77b928be09063e391d1efa1a0cd410f6783074aaa9`。

## Staged `w13`／`w2` streaming gates

Fixed-arena gate 在同一台 Mac14,13／Apple M2 Max 64 GiB、MLX 0.32.0 上執行
1／4／8 rows 各 12 paired samples，共 36 pairs。Control 固定 arena 是
13,369,344 bytes；candidate 固定 `w13`／`w2` arenas 是 8,912,896／4,456,448 bytes。
所有六個 canonical region hashes、float32 output hashes、destination budgets、aligned
bypass reads 與 post-read nonresidency checks exact。

| Rows | Complete-wall median | p95 | Hidden `w2` read | Gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | -8.06% | +31.02% | 49.24% | p95 stop |
| 4 | -10.74% | -10.24% | 80.98% | continue |
| 8 | -13.67% | -12.67% | 87.90% | continue |

Artifact SHA-256 是
`fa764c02403502fc4a50adbdbb37edd78e612774547ce28c6f8d898da3855f9f`。
4／8 rows 全部集中在單一 routed expert，是 optimistic component gate，不是正式速度。

後續 default-off split-slot runtime 用四個 fresh-process waves 執行 128-token
`repeated`／32 greedy outputs，order 是 control/staged、staged/control、control/staged、
staged/control。八個 outputs 全部共享 token SHA-256
`06de0413ef6f0d6f67a8328b6cfbea9cf97fe28689b6503ad9975367a59d3fa7`。
每一 pair 的 logical expert bytes 都是 40,121,401,344、evictions 是 1,849、peak MLX
memory 是 24,665,121,392 bytes。每個 staged run 的 1,024 reads 都閉合為
9,126,805,504 `w13` + 4,563,402,752 `w2` bytes；control staged counters 是 0。

Paired median request +2.17%、Decode throughput -4.26%、decode p95 +2.12%、memory 0%。
Correctness、p95、memory gates 通過，但必要的 request -5% 或 Decode +5% 失敗，
decision 是 `stop_runtime_candidate_keep_default_off`。Artifact SHA-256 是
`3dd265bf3fb9a78b23d596bbf17adc4f522ad211e2317b75a87a660161198542`。
Prototype 預設 `false`、拒絕 DSpark、沒有 CLI／server／APP opt-in。四 pairs 與單一
短 workload 不是正式效能證據，bypass 也不是 physical SSD measurement。

Adaptive prefill route eligibility 先使用五種 exact 4,096-token prompts；所有十個
reference／trace processes 的 prompt／output hashes exact，每層 24,570-assignment
histogram 與 cumulative chunks 閉合。70%／80%／90% threshold 的跨 workload logical
prefill-byte estimate 中位數是 -23.19%／-33.38%／-34.24%，三者通過 continuation。
Artifact SHA-256 是
`bbd003adf9ba24df977e88de79ce5e616902a7496289b6d5743efaf808ae79f4`。

Internal fixed-buffer runtime 再以 `repeated` 4K／2 greedy outputs 跑兩組反向
adaptive/control pairs。四條 outputs 都是 `[1950, 1950]`，token SHA-256
`dd22c238773ee5642280c221b7a7a51094e6dcec6882cce0f749378ee01e4891`。每個 candidate
42 層全 selective，union/read experts 3,309，actual adaptive batched bytes
44,239,159,296；加 393 ordinary misses 後 request bytes 49,493,311,488。Control 是
10,752 full-prefill rows 加相同 393 misses，共 149,001,338,880 bytes；avoided bytes
99,508,027,392。所有 accounting predicates exact。

Paired median request +16.78%、TTFT +16.87%、request expert bytes -66.78%、peak memory
-15.78%；TTFT p95 +17.02%。`repeated` 在 70%／80%／90% 的所有 42 個 decisions
相同，因此三 threshold 都無法通過 per-workload p95 guard。Decision 是
`stop_runtime_candidate_keep_default_off`，未執行完整矩陣與 cached observer。Runtime
artifact SHA-256 是
`9138b15655c45e7493d1b40aa01b61b2e8afb908225641a11034e1fb6d9ff99f`。
兩個 artifacts 都不是正式效能或 physical SSD measurement。

## DSpark coherent candidate-path gate

五個 128-token workloads 各由 fresh process 建立第一個 speculative round。Script 從同一
DSpark backbone 的五個 fixed position logits，加上前一 token conditioned rank-256
Markov bias，產生 branch-4／beam-8 coherent paths；baseline 另由實際
`DSparkModel.draft` 重跑並從相同 context snapshot 復原。五個 workers 的 baseline
tokens、confidence、候選 top-4 transitions、main hash layers 0--2 routes、六步 sequential
target truth 與 committed prefix criteria 全部通過。

| Workload | Baseline accepted | Baseline requested／missing | Storage-selected alternative | Alternative accepted |
| --- | ---: | ---: | ---: | ---: |
| `repeated` | 5 | 18／0 | 無 | -- |
| `code` | 5 | 104／75 | 87／59 | 4 |
| `zh_technical` | 5 | 102／67 | 無 | -- |
| `mixed_math` | 5 | 103／38 | 無 | -- |
| `tool_like` | 5 | 84／52 | 68／38 | 1 |

每個 workload 都有 64 組只讀 draft probability、requested hash union 與 target-LFU
snapshot misses 的 selector weights。Target acceptance 不參與 selection。雖然 `code`／
`tool_like` alternatives 分別少 17／16 requested blobs，兩者都降低 acceptance；其餘
workloads 的所有 weights 都保留 baseline。Continuation 是 0/5，decision 為
`stop_current_markov_beam_candidate`。Artifact SHA-256 是
`de2ee382fff19f030a3c5150fe1251c352e51d6e5928fb2bfa7c5e40211d29ce`。

這是 candidate-feasibility，不是 timing、steady-state cache、learned-router 或 physical
SSD evidence。沒有 runtime／CLI／server／APP option。Sampling 仍禁止，因 storage
selector 改變 proposal distribution 而目前沒有 selected-path rejection proof。

## DSpark learned-router frozen-transfer gate

相同五組 128-token prompts 各以 fresh process 擷取第一個完整 draft。Sequential verifier
每層保存 anchor + 五個 draft inputs 的 target top-6 routes；predictor scoring 排除 anchor，
留下 layers 3--42 的 6,000 assignments。四個 feature sources 是三個 DSpark layer 的
`hc_head` outputs 與 final output norm；每個 source 直接通過 target layer frozen
`ffn_norm`／router，量 top-6／12／24／48。

五個 prompt contracts、actual／reconstructed drafts、六-position route traces、40-layer
prediction matrices、stable unique top-k 與 assignment／union bytes identities 全部通過。
各 k 的最佳 source 都是 `dspark_final_norm`：

| k | Assignment recall | Union recall | Useful rate | Full target-set positions | Layers >=75% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 6.27% | 12.18% | 13.50% | 0% | 0/40 |
| 12 | 10.13% | 20.56% | 12.40% | 0% | 0/40 |
| 24 | 17.15% | 33.03% | 11.15% | 0% | 0/40 |
| 48 | 28.93% | 51.26% | 10.03% | 0% | 0/40 |

Top-24 的五 workload assignment recall 是 12.42%--22.42%，沒有通過 80% aggregate、
70% floor、30-layer 或 50% useful-rate gate；top-48 依 protocol 不得 continuation，且
約 90% predicted union 仍 wasted。Decision 是
`stop_direct_frozen_router_transfer_training_required`。Artifact SHA-256 是
`4eb2574b7a9e6b613784288833a3d6ff327e71af145e6d822c2b501d387f7739`。

這些 bytes 是 offline labels，不是 executed prefetch 或 physical SSD traffic。Runtime
沒有新增 scratch／probation／deadline pinning。失敗只拒絕 direct transfer，不拒絕經
明確 data collection、split 與 artifact contract 訓練的 reducer/predictor。

## 驗證矩陣

### Qwen 第一版驗證結果

2026-08-27 已執行官方 FP8 header inspection。
inspection 沒有下載 tensor payload。
結果為 48 層、每層 512 個 routed expert、每個 expert blob 2,611,200 bytes，
181,906,343,706 checkpoint tensor payload bytes，
以及 125,268,506,112 installed weight bytes。

Swift 輕量測試已覆蓋 format 2 planner、vision/MTP 排除、MXFP4 固定向量、
Qwen expert 轉換、installed artifact file 續傳與損壞 layer repair。Python 輕量測試已覆蓋 QSA、N-gram、PLE 資料路徑、
Qwen expert layout、layer-major prefill 和 XML tool parser。

2026-08-27 Qwen 第一版工作樹的自動測試結果如下：

| Suite | 通過 | 失敗 | 略過 |
| --- | ---: | ---: | ---: |
| Swift `DeepSeekRepackTests` | 16 | 0 | 0 |
| Swift `DeepSeekV4SSDAppTests` | 39 | 0 | 3 |
| Python runtime 與 server | 113 | 0 | 0 |
| 合計 | 168 | 0 | 3 |

三個 Swift App 測試需要現有的完整 DeepSeek installed model。
使用者已把該模型移到外接硬碟。
本次驗證沒有存取外接硬碟，並略過這三項。

Swift App 測試也確認空的模型庫仍提供 DeepSeek 與 Qwen 選項。
Swift App 測試確認模型下載會比較所需空間與可用空間。
Swift App 測試確認 First Token wait time 會在 Prefill 期間即時更新。
效能歷史只記錄每個 request 的最終 First Token wait time。
Python 測試使用 fake API server 驗證 API benchmark 的精確 input token、
指標收集、Peak、P95 和 ASCII 表格。
Python 測試也驗證 installed model discovery、Server 自動啟動和 Server 清理。
Python 測試確認 APP 的 `/api/status` polling 不會寫入 access log。
本機 Qwen installed model 已通過 Server 自動啟動、API ready 和自動清理檢查。
Swift repack 測試確認 direct installed artifact file 可以從現有 file size 續傳，
並在完成時驗證 SHA-256。

Whallm 更名工作樹已執行 `make package`。
流程建立 `dist/Whallm.app` 和 `dist/Whallm-macOS-arm64.zip`。
App 與解壓後的 App 都通過簽章、三種 localization 和隔離啟動檢查。

Published installed model 位於
[`Yanun/Qwen3.8-Flash-Next-MXFP4`](https://huggingface.co/Yanun/Qwen3.8-Flash-Next-MXFP4)。
程式碼固定 commit `753d0aa57059fad70a5f7e6cc249f25df56bbd34`。
Hugging Face API 列出 61 個 repository files。
遠端 `manifest.json` 的 SHA-256 是
`3f4cb52a88335591cfb8233778eb56396e9d2756c153485e4dcfb0a12daed0ce`。
這個 SHA-256 與本機 `manifest.json` 相同。
遠端 `common.bin` 的 1 MiB range 回傳 HTTP 206 和 1,048,576 bytes。
遠端 range 的 SHA-256 與本機相同。
遠端 `config.json` 的 SHA-256 也與本機相同。
本次驗證沒有重新執行完整 125 GB 的 App direct download。

Qwen installed model 位於內建 SSD。
完整 SHA-256 驗證已通過 57 個 manifest files。
manifest files 合計是 125,291,490,955 bytes。
common tensor 共 1,069 個。

完整模型 API 驗證結果如下：

| 項目 | 結果 |
| --- | --- |
| thinking 關閉 | 回覆 `OK`，`stop`。 |
| thinking 開啟 | reasoning 與答案 `4` 正確，`stop`。 |
| forced tool call | `get_weather`，`{"city":"Taipei"}`。 |
| streaming tool call | function 名稱與 arguments 和完整 response 相同。 |
| packaged App | App、ZIP、三種 localization、隔離啟動和 runtime import 通過。 |

CLI 也使用預設 `memory_limit_gib=0` 執行完整模型。
runtime 套用 48 GiB 自動上限。
5-token prompt 產生 ` Paris`。
該 request 從 SSD 讀取 4,658,380,800 bytes 的 routed expert。
MLX peak memory 是 13,048,438,112 bytes。
request 結束時的 active memory 是 13,040,928,562 bytes。
修正前的 MLX peak memory 是 74,069,529,060 bytes。
修正前後的 output token SHA-256 相同。

4K 使用 4,096-token repeated prompt、greedy generation 和 1 個 output token。
runtime memory limit 是 48 GiB。

| Cache state | 重用 prompt tokens | Wall time | Expert bytes read | Output token SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Cold | 0 | 65.42 s | 66,081,638,400 | `6dfb9763…b3b2bd1` |
| Warm | 4,095 | 0.214 s | 0 | `6dfb9763…b3b2bd1` |

兩次 output token ID 都是 `1228`。
兩次完整 SHA-256 都是
`6dfb97632210ac38a071667cf8be7df83a16178e12f1248e45b2a3d24b3b2bd1`。
這些時間是一個本機驗證結果，不是效能保證。
完整 artifact 位於
[`benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json`](benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json)。

| 能力 | 狀態 | 證據邊界 |
| --- | --- | --- |
| 固定 checkpoint 合約 | 通過 | 程式檢查、測試和完整 manifest 驗證。 |
| Installed model SHA-256 | 通過 | Qwen 目前 57 個 manifest files。 |
| Repack 續傳與 repair | 通過 | Fixture 自動測試。 |
| MXFP4 routed expert | 通過 | Dequantized reference 和 batched parity 單元測試。 |
| Layer-major prefill parity | 局部通過 | Fixture next-token logits 與既有歷史 full-model token 通過；低信心 `random_hex` 4K／32 相對 sequential-prefill reference 在 index 6 分歧，需另行定位數值穩定性。 |
| MXFP8 cache | 通過 | 單元測試、8K 歷史 greedy token。 |
| Persistent prompt cache | Format-4 functional gate 通過 | 完整 model／RoPE／KV／attention contract、content-address block chain、quantized round-trip、suffix partial restart、immutable sharing 與 frequency-aware eviction tests；453-token installed-model branch gate 重用 128 tokens 並保持 cold output exact。不是 balanced speed result，也沒有 per-layer KV delta dedupe。 |
| Atomic DSpark prompt-context cache | Functional gate 通過、預設關閉 | Memory／restart hits exact 重用 127 tokens，三條 output hash exact，combined logical bytes -92.23%／-65.59%；僅單一短 workload、非 balanced performance result。 |
| Native MTLIO ownership/copy path | Native gate 通過、runtime integration 停止 | 16/16 rows byte exact，shared/private event visibility 與 cancellation safe；MLX 0.32.0 public external-event handoff 缺失，exploratory timing 也不是採用證據。 |
| Staged `w13`／`w2` streaming | Component exact、runtime candidate 停止 | 36-pair fixed-arena gate 允許 follow-up；四波 full-model gate 的八條 tokens 與 bytes/evictions exact，但 request +2.17%、Decode -4.26%，未達 5% improvement gate。預設關閉且無 public opt-in。 |
| Adaptive full/selective prefill | Route與 runtime correctness 通過、candidate 停止 | 五類 4K route gate 讓 70/80/90% eligible；兩組反向 full-model pairs 的 selected rows、batched/request bytes 與 tokens exact，但 `repeated` TTFT +16.87%、p95 +17.02%。三 thresholds decisions 相同，full-layer 保持預設。 |
| Ready expert decode | 通過 | 五組歷史 hash 和目前單元測試。 |
| API、tool、SSE | 通過 | Python server tests，包括無效 tool call 的 Codex 終止 event。 |
| Expert-file page-cache proxy | Research contract 通過 | Darwin syscall、partial-page、failure/unclassified 與 full-model accounting tests 通過；4K useful/wasted residency coverage exact。預設關閉，且不等同 physical SSD bytes。 |
| Expert-file bypass policy | Research contract 通過、候選效能拒絕 | 六個 fully nonresident installed ranges 通過 byte/residency contract；policy 預設 `cached`。4K／32 repeated gate 的 fixed/adaptive 都未達 request／Decode 門檻，且 bypass 不等同 system-wide purge。 |
| Verifier expert union | Acquisition contract 通過、目前 hybrid 效能候選拒絕 | 128/8 四波 gate 的 19 outputs exact，sequential/grouped/hybrid calls 258/43/43。Hybrid bytes 較 sequential 低但 verification +17.34%，相對 grouped +87.59%；grouped 仍有既有 correctness stop。 |
| DSpark 邏輯 | 局部通過 | Fixture greedy、sampling、replay、fallback、sequential oracle 與 hybrid tests 通過；原 block verifier 在 near tie 失敗，hybrid v3 的五組 4K／32 gate 與一組 `random_hex` 4K／128 normal／768／96 exact gate 通過。Full-model sampling 與淨加速仍未驗證。 |
| Hash exact prefetch prototype | Correctness 通過、目前效能候選拒絕 | Fixture 與五組 4K correctness 通過；bypass-policy 4K／32 repeated fixed gate 的 request +18.19%、Decode -59.67%，未達 adoption stops。預設關閉。 |
| Adaptive block prototype | Correctness 通過、目前效能候選拒絕 | 五組 composition exact；bypass-policy 4K／32 repeated adaptive gate 雖使 speculative bytes/committed -42.96%，request +5.02%、Decode -25.58%。預設關閉。 |
| DSpark coherent candidate path | Structural gate 通過、candidate 停止 | 五組 first-round baseline 全部 target-accepted 5/5；storage weights 選到的兩個 lower-union alternatives 分別只接受 4／1，0/5 通過 continuation。Greedy-only research script，無 runtime setting，sampling 未證明。 |
| DSpark learned-router predictor | Fixed-label baseline exact、direct transfer 停止 | 6,000 assignments 與 top-k union accounting exact；最佳 top-24 只有 17.15% assignment recall、33.03% union recall、11.15% useful rate，0/40 layers 達 75%。沒有執行 prefetch；trained follow-up 需新資料／split／artifact protocol。 |
| Remaining PLAN prerequisites | Local audit 通過、scoped stop/defer | 九個 training／DeepSeek dense ANE／trace／physical-I/O／non-equivalent directions 全部有 local absence evidence 與 reopening rule；Xcode/system tools 可用不等同 candidate 或 measurement readiness。沒有實作或授權 training／approximate mode。 |
| Native MTP-1 baseline | Pinned checkpoint feasibility 拒絕 | Official graph 需要三個 `mtp.*` stages；4,705 tensors 與完整 installed payload 通過 contract／SHA audit，但沒有 self-contained one-stage path。Runtime 不做 checkpoint-faithful 裁層。 |
| DSpark 96-slot reduced cache | 停止 | 三組 4K／32 exact 且 peak memory 較低；4K／128 也 exact、peak -7.35%，但 draft bytes/committed +83.17%，未通過預先宣告的 +50% gate。保留 768-slot 預設。 |
| DSpark 淨加速 | 未通過 | 目前 checkout 的 R3 探索性 ABBA 已觸發停止條件。 |
| 14K context | 歷史通過 | Tool-like prompt，greedy，1 output token。 |
| 1M context | 未驗證 | checkpoint 合約值不等於本機證據。 |
| Full-model sampling parity | 未驗證 | 目前沒有固定 reference artifact。 |
| 正式模型能力基準 | 未執行 | 2026-09-06 完成 Q0 初測，Q1A 在第 27/32 題因回退停止；不宣稱一般能力評分，詳見本文件開頭。 |

## 重跑規則

每個正式 performance row 必須保存：

1. commit、dependency、checkpoint revision、硬體和 OS。
2. prompt 產生方式、input token 和 output token。
3. batch、temperature、top-p 和停止條件。
4. slot、worker、prefill、KV cache、prompt cache 和 DSpark 設定。
5. fresh/warm compile、OS page cache 和 expert cache 狀態。
6. wall time、TTFT、decode time、SSD bytes、read timer 和 memory 指標。
7. 完整 output token hash。

設定比較必須使用相同 output token 數。
greedy 比較必須使用相同 token hash。
正式 A/B 應至少使用五個 prompt，並交錯執行順序。

## 重跑 command

記錄版本：

```sh
git rev-parse HEAD
sw_vers
.venv/bin/python --version
.venv/bin/python -m pip show mlx mlx-lm transformers
```

驗證 installed model：

```sh
swift run dsv4-repack verify --model /path/to/model.dsv4
```

量測 expert blob I/O：

```sh
swift run dsv4-repack benchmark \
  --model /path/to/model.dsv4 \
  --samples 32
```

CLI 可以使用 `--metrics-json` 保存 runtime 指標。
CLI 也會保存完整 prompt token ID 的 `prompt_token_sha256`。
CLI 也會保存完整 output token ID 的 `token_sha256`。
hash 輸入是以逗號連接的十進位 token ID。
`Scripts/prepare_r0_prompts.py` 可以建立 4K、8K 和 14K 固定 prompt 檔。
工具會把 prompt hash 和重建規則寫入 manifest。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file fixed-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --top-p 1 \
  --no-persistent-prompt-cache \
  --metrics-json result.json
```

逐層重跑 sequential／two-token block correctness diagnostic：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/diagnose_layer_parity.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json \
  --workload random_hex \
  --token-metrics docs/benchmarks/2026-08-26-random-hex-4k32-normal-sequential-prefill-m2-max.json \
  --anchor-index 13 \
  --output /path/to/layer-parity-diagnostic.json
```

診斷會先要求 exact prefix reconstruction，並驗證攔截前後的 sequential 與 block logits
完全相同；任一 audit 失敗就不寫出 artifact。

重跑 layer 0 component diagnostic：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/diagnose_layer0_attention.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json \
  --workload random_hex \
  --token-metrics docs/benchmarks/2026-08-26-random-hex-4k32-normal-sequential-prefill-m2-max.json \
  --anchor-index 13 \
  --output /path/to/layer0-attention-component-diagnostic.json
```

Component diagnostic 同樣要求 exact prefix、完整 layer 0 capture 與 instrumentation-preserved
reference logits。Raw rotating-cache 長度不同時只比較共同 temporal-order suffix，不能把
不同 shape 的 raw arrays 宣稱為 exact 或 non-exact。

重跑 hybrid v3、hash 與 adaptive 五組 correctness survey：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_dspark_adaptive.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json \
  --raw-directory /new/raw/directory \
  --output /new/artifact.json \
  --max-tokens 32 \
  --workloads storage_sentence multilingual_choice random_hex creative_metaphor balanced_choice \
  --run-pattern normal fixed adaptive \
  --skip-warmup --no-dspark-fallback --no-layer-major-prefill \
  --hybrid-verification --hybrid-hash-prefetch
```

重跑 expert-file bypass installed-range contract：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_expert_file_cache_bypass.py \
  --model /path/to/model.dsv4 \
  --sample-count 6 \
  --output /new/cache-bypass-contract.json
```

只有 contract 的 `contract_passed=true` 才可重跑完整 candidate gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_cache_bypass_dspark.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json \
  --raw-directory /new/raw/directory \
  --output /new/cache-bypass-full-model.json \
  --cache-bypass-contract /new/cache-bypass-contract.json \
  --python .venv/bin/python \
  --workload random_hex \
  --max-tokens 32
```

重跑 verifier union／execution-shape gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_verifier_union_tradeoff.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json \
  --raw-directory /new/union/raw/directory \
  --output /new/verifier-union-tradeoff.json \
  --cache-bypass-contract /new/cache-bypass-contract.json \
  --python .venv/bin/python \
  --workload repeated \
  --max-tokens 8
```

正式保存時必須使用全新的 raw directory 與 artifact 路徑。只有在 source tree、runtime
設定與既有 raw 完全相同時才能使用 `--resume`；不得把舊候選 raw 重新標成新 runtime
結果。

重跑 native MTLIO ownership/copy-path gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_mtlio_experts.py \
  --model /path/to/model.dsv4 \
  --raw-directory /new/mtlio/raw/directory \
  --output /new/mtlio-expert-streaming.json
```

此 probe 需要完整 Xcode；預設使用
`/Applications/Xcode-26.6.0.app/Contents/Developer`，可用
`--developer-directory` 指定其他安裝。Raw directory 必須是新的；結果固定標記 OS
page cache `not purged` 與 `formal_performance_result=false`。Native gate 通過也不能略過
artifact 的 MLX resource + dependency conjunction。

重跑 staged fixed-arena gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_staged_expert_streaming.py \
  --model /path/to/model.dsv4 \
  --output /new/staged-w13-w2-overlap.json
```

只有 fixed-arena artifact 的 correctness 與 4／8-row continuation gate 通過，才可重跑
default-off runtime follow-up：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_staged_expert_runtime.py run \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json \
  --raw-directory /new/staged-runtime/raw/directory \
  --output /new/staged-expert-runtime.json \
  --python .venv/bin/python \
  --max-tokens 32
```

現有 split-slot schedule 已停止；上述命令只用於重現，不得把重跑包裝成新候選。
任何新 gate 必須先有 materially different first-stage kernel 或 I/O schedule，並另行
預先登記。Raw directory 與 output path 都必須是新的。

重現 adaptive prefill route eligibility：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_adaptive_expert_prefill.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json \
  --raw-directory /new/adaptive-prefill-route/raw/directory \
  --output /new/adaptive-prefill-route.json \
  --python .venv/bin/python \
  --max-tokens 2
```

只有 route artifact 的 continuation gate 通過，才可重現 internal runtime early gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_adaptive_expert_prefill_runtime.py run \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json \
  --eligibility-artifact /new/adaptive-prefill-route.json \
  --raw-directory /new/adaptive-prefill-runtime/raw/directory \
  --output /new/adaptive-prefill-runtime.json \
  --python .venv/bin/python \
  --max-tokens 2
```

目前 post-attention schedule 已停止；命令只供重現。不得把 logical avoided bytes 寫成
physical SSD bytes，也不得在沒有新 overlap／prediction design 的情況下把相同
70/80/90 matrix 當成新候選。

重現 DSpark coherent candidate-path first-round gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_dspark_candidate_paths.py run \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json \
  --raw-directory /new/candidate-path/raw/directory \
  --output /new/dspark-candidate-paths.json \
  --python .venv/bin/python
```

現有 branch-4／beam-8 candidate 已停止；命令只供重現，raw directory 與 output 必須
全新。不得把第一輪 feasibility 的 wall time 當成 speed result，也不得啟用 sampled
selector。新 candidate 需要 materially different generator 或 trained selector/drafter，
並先建立新的 predeclared protocol。

重現 DSpark-hidden learned-router frozen-transfer gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_dspark_router_predictor.py run \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json \
  --raw-directory /new/router-predictor/raw/directory \
  --output /new/dspark-router-predictor.json \
  --python .venv/bin/python
```

Direct frozen-router transfer 已停止；這個命令只重現 fixed labels／offline top-k gate，
不執行 prefetch。任何 trained follow-up 必須先另寫 data rights、collection、
train／validation／held-out splits 與 immutable dataset/model hashes，不得沿用同一
artifact 名稱冒充 trained candidate。

重現 normal block prompt-cache partial-restart functional gate：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_block_prompt_cache.py \
  --model /path/to/model.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json \
  --protocol research/BLOCK_PROMPT_CACHE_2026-08-27.md \
  --output /new/block-prompt-cache.json
```

Output 必須是新路徑；script 內部也使用新的 temporary cache directory。這是一次
restart／partial-match／parity functional gate，不能把三條 wall time 當 balanced speed
result。若修改 protocol，artifact 必須記錄 run 當下的 protocol SHA。

重現 remaining-PLAN local prerequisite audit：

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/audit_plan_prerequisites.py \
  --model /path/to/model.dsv4 \
  --protocol research/PLAN_PREREQUISITE_CLOSURE_2026-08-27.md \
  --output /new/plan-prerequisite-audit.json
```

這個 audit 只讀本機 checkout、installed manifest、工具與 package availability。它不會
下載模型、安裝 training stack、執行 training、建立 ANE candidate 或產生 device-byte／
GPU timing。任何 prerequisite 狀態改變時都必須用新 output 重跑，不能覆寫舊 artifact。

不要只保存 `tokens_per_second`。
請同時保存 prompt token、`prompt_token_sha256`、generated token、
`token_sha256` 和 runtime metrics。


## 2026-09-06 Qwen cross-arena kernel：M2 Max 研究驗證

[完整證據索引](benchmarks/2026-09-06-qwen-cross-arena-kernel-m2-max/summary.json)
與 [研究條件／來源](../research/QWEN_DECODE_KERNEL_2026-09-06.md)。
14 個 fresh-process ABBA／BAAB 波次、56 次效能 requests、11,264 個正式 output
tokens：配對與跨波次 outputs exact；逐波重算 Decode 至少 +5%、TTFT 至多 +5%、
p95 至多 +10%、MLX peak 至多 +15%、expert bytes 至多 +5%，全數通過。
實際 Decode +9.59–16.42%、request 縮短 2.35–5.12%、TTFT 最大增幅 +2.60%；
p95 每輪下降，expert bytes 相同。

另有 12 次 cold allocation requests（peak 最大 +11.50%）、16 次 lifecycle／
restart requests，以及三層真實 capture／八組排列 component gates。獨立 audit
核對最終 kernel、runtime、runner hashes、ANE active／無 fallback、新 kernel
實際執行次數、全部 output 長度與跨 request 前綴，通過。未清 OS 或 Metal compiler
cache；cold allocation 單次 latency 不是正式 throughput 結論。

本輪僅新增 research 程式，未改 production runtime；research Python 語法檢查
通過，未重跑先前 305 項 Python suite。M5 Pro、能耗與 production integration
均未完成。原 CLI grouped flag 不會啟用新 kernel，App 預設仍關閉。

## 2026-09-07：Issue #7 公開 1.1.5 崩潰重現

M2 Max 64 GB／macOS 26.6.2 使用公開 1.1.5 套件，DeepSeek 第一筆 HTTP
request 回 200 後，Python 在 request thread 的 MLX compile-cache cleanup
SIGSEGV，後續連線被拒絕。與本機當日 installed App crash 堆疊相符；#7 回報者
尚無 traceback／模型資訊，未確認 M1 Ultra 是否同因。

無模型 tuple compile probe：公開 1.1.4 100 次 thread lifecycle 通過；
1.1.5 三個 fresh processes 全部 exit 139。Qwen 12 次 bounded requests 未崩潰，
這不代表所有模式安全。隔離 clear_streams 候選通過最小測試，但完整 DeepSeek
第二筆 request 出現 missing GPU Stream，因此拒絕採用。Production、installed
App 及模型未修改，沒有 build／release。此為崩潰驗證，不是效能結果。

詳見 [調查](../research/ISSUE_7_REPRODUCTION_2026-09-07.md) 與
[機器可讀摘要](benchmarks/2026-09-07-issue7-reproduction/summary.json)。

## 2026-09-07：SSD 功能實作 gate

最終 runtime **367 tests 通過**。新增七項覆蓋 eviction ranking、保留配額／
heap rebuild、protected／speculative pinned keys、讀取失敗與恢復、舊 catalog
及合法值、B4 次數保留／去重 reads／native exact，以及 BF16／FP16／FP32
state clone 的原始位元與實體 buffer 隔離（含負零、subnormal、NaN payload）。
CLI、server 與 model catalog 新選項預設 LFU；status／metrics 可觀測策略。

LRU 八次 128-token 完整請求、32 full-logit／122 state-array 比對通過；完整
request 只縮短 3.17%，未達門檻，未升預設。B4 首次 warm gate 只有一個負零
sign bit 改變，logits／tokens 一致但 hash gate 仍拒絕。兩次 dump 確認原因後，
byte-copy 修正重新跑 exact 及八次交錯成本測試，warm state／expert state、
32 full logits／outputs、posterior schema／122 arrays 與 base isolation 全通過。
有效的 LRU 測速與修正後 warm 成本樣本均無 system swapout；LRU 並未升為預設。
MLX／RSS 皆低於 +1 GB，兩份 raw read-back audit 通過。

LRU 成本與暖 B4 成本的分母不同；後者不含 drafter、拒絕恢復或輸出串流 P95。
未執行其他模型或 App build。完整 failed／passed logs、source versions 與限制見
[實作結果](../research/SSD_FEATURE_IMPLEMENTATION_RESULTS_2026-09-07.md) 和
[證據索引](benchmarks/2026-09-07-ssd-feature-implementation/summary.json)。

## 2026-09-07：SSD 三方向第一輪 gate

新 sparse-fill 原型的 8K／16K 數值比對分別通過 384／768 個 Prefill 中間結果、
32 步完整 logits／token，以及 122 個 cache-state arrays／schema。
每長度八次有效 ABBA／BAAB 皆為 64 tokens、輸出完全一致，ANE active 無 fallback、
system swapout 為零；獨立重新讀取 metrics、status、prompt hash、medians、P95、
次序分半與 peak 差額通過。8K 第一輪六次紀錄因 control 的 4 pages swapout
整輪排除，原資料保留，再於 quiet window 後重跑。速度未達 5%，未採用。

MLX max(candidate)-min(control)：8K +147,468 bytes；16K +507,908 bytes。
RSS 同算法：8K +573,440 bytes；16K +212,992 bytes；全部小於 +1,000,000,000 bytes。
Component 覆蓋 exact QMM、跨 chunk／layer 重用、read error drain／恢復、取消前
ownership release。新候選因速度 gate 失敗，未宣稱完成完整 runtime cancellation
與多 request lifecycle 驗收。

後兩方向為 CPU-only trace 分析：三份 baseline Decode miss-mask bits 共 90,720
全部重現；政策／配額四組獨立 ablation 與 OrderedDict LRU 六組重算一致。
這些檢查不證明實際 SSD 裝置流量、wall-time、speculation 回復或全流程峰值。
本輪 runtime unittest **360 項通過**，研究 component **3 項通過**；source hashes
與開始時一致，未改 production。詳見
[原始資料與 source archive](benchmarks/2026-09-07-ssd-three-directions/summary.json)。
