# 模型支援套件

2026-09-12 採用選項 2。本文描述已接入的程式；早期介面草案見
[設計研究](../research/MODULAR_RUNTIME_2026-09-12.md)。

## 目前結構

```text
App / CLI / HTTP
       |
       +-- 共用模型描述資料：ModelPackages.json
       |
       +-- Swift ModelPackage：安裝、修復、驗證
       |
       +-- Python ModelSupport：載入、Prefill、狀態、對話格式
                 |
                 +-- DeepSeek V4
                 +-- DeepSeek V4.1
                 +-- Qwen3.8
                 |
                 +-- 共用 SSD / slot 管理、expert layout、MLX 運算
```

Swift [ModelPackage](../Sources/DeepSeekRepack/ModelPackage.swift) 統一 App 與 CLI 的
安裝入口；三個實作位於 [ModelPackages](../Sources/DeepSeekRepack/ModelPackages/)。
原有 checkpoint 檢查、repack 和 Qwen artifact 下載程式繼續使用。
V4／V4.1 的完整安裝驗證仍由 App 呼叫；Qwen artifact 安裝與修復自行完成驗證。

Python [model_support](../runtime/deepseek_v4_ssd/model_support/) 集中每個模型的
manifest contract、載入、Prefill、對話解析、抽樣與 reasoning 規則。
共同 generation 透過套件建立、複製、保存、還原和評估 cache，
不再依 `_is_qwen`／`_is_deepseek_v41` 選擇這些行為。
V4.1 明確不提供對話快取重用和持久化，預設請求使用 exact 模式。
顯式要求不支援的近似模式時，API 在生成前拒絕。

[expert_layout.py](../runtime/deepseek_v4_ssd/expert_layout.py) 提供兩個實際使用的格式：
V4／V4.1 共用分離 `w1/w2/w3` 到 fused slot 的映射；Qwen 使用 `gate_up/down`。
套件選擇格式，`ExpertCache` 共用檔案讀取、slot、eviction 和 buffer 生命週期。
模型運算與量化算式維持既有實作，沒有新增運算圖解譯器。

## 共用描述資料

唯一來源是 [ModelPackages.json](../Sources/DeepSeekRepack/Resources/ModelPackages.json)。
Swift 與 Python 使用相同 bytes，包含：

- 穩定 model kind、API model ID、顯示名稱與 owner。
- 固定 checkpoint ID／revision、manifest 結構版本、預設安裝目錄。
- App 快速檢查所需檔案、可用功能、可編輯設定及預設值。
- 模型自動記憶體上限與 Prefill 門檻。

`ModelKind` 改成經 catalog 驗證的字串型別；JSON 編碼仍是一個字串，
既有三個常數、API ID、Alias、偏好設定鍵保持相容。
App 模型列表、進階設定可見性、預設值和 Python catalog 使用描述資料。
描述檔不是完整的 tensor 驗證規則；嚴格檢查仍在各模型套件裡。

新模型以 `modelKind` 選擇支援，並核對套件接受的 `formatVersion`。
不同模型可以共用結構版本；只有舊 manifest 缺少 model kind 時才使用固定的
1→V4、2→Qwen、3→V4.1 映射。現有 v1/v2/v3 reader、writer 和安裝檔保持相容，
沒有自動重寫 installed model。

描述檔不接受 Python module 路徑。可執行實作由 Swift 與 Python 的內建註冊表指定，
隨 App 發布；沒有新增即時下載第三方程式並執行的功能。

## 新增模型

1. 在描述資料加入一筆模型，固定 checkpoint revision，選用已有設定與功能。
2. 在 Swift 的 `ModelPackages` 目錄加入安裝實作，登記至 `ModelPackages.builtins`。
   重用已有 checkpoint 或 artifact 下載器；需要新轉換方式時，再實作對應規則。
3. 在 Python `model_support` 加入 `ModelSupport` 子類別，登記至 `support_types()`。
   一般需提供 `manifest_contract`、`load`、`prefill`、`open_codec`、
   `make_tool_stream_parser`；使用 SSD experts 的模型另選擇 `expert_layout`。
4. 若模型使用不同 cache 結構，覆寫套件的 cache 操作，並驗證保存／還原完整性。
   沒有完整還原實作時，不宣告 `promptCache`。
5. 提供固定 checkpoint 的小型安裝 fixture、錯誤拒絕案例和數值對照，
   再進行完整模型驗證。相似架構可直接重用運算元件，不需建立繼承鏈。

共用 `ModelSupport` 已提供標準 MLX cache、一般抽樣、資源釋放及設定驗證。
模型的運算仍使用 MLX／mlx-lm 的 `Model` 呼叫慣例；這不是任意推論引擎的外掛規格。
新的運算、量化格式、特殊設定或 sidecar 行為仍需要對應程式與驗證，
不能只靠描述檔宣告就取得支援。

可執行檢查：

```sh
PYTHONPATH=runtime:. .venv/bin/python -m deepseek_v4_ssd.model_support
PYTHONPATH=runtime:. .venv/bin/python -m unittest runtime.tests.test_model_support
PYTHONPATH=runtime:. .venv/bin/python -m unittest discover -s runtime/tests -p 'test_*.py'
swift test --skip ServerConfigurationTests.testAPIKeyRoundTripsThroughIsolatedKeychainItem
.venv/bin/python Scripts/check_repack_cli.py .build/debug/dsv4-repack
```

[第四模型測試](../runtime/tests/test_model_support.py) 示範只登記套件與描述，
經 catalog、`ModelManager`、manifest、loader 執行真正的小型 MLX 生成，
中斷後再次請求並卸載。它沿用 manifest 結構版本 1，不為新模型另加核心分支。
這是接入與生命週期驗證，並非第四個正式模型的支援或完整 checkpoint 品質證明。

## 打包與相容性

打包將同一份描述檔放在 `Contents/Resources/ModelPackages.json` 及包內 Python
套件旁。Swift 先讀主 App resources，再考慮 `Bundle.module`；Python 先讀包內副本，
開發 checkout 才使用相對來源路徑。封裝檢查確認兩份資料相同、三個套件已註冊，
並在禁止存取專案 `.build`、`.venv` 或來源資料的條件下實際載入。

部分舊 research helper import 保留相容入口；目前 SSD Prefill 研究 wrapper
已改接 Qwen 套件的 Prefill 函式。歷史量測不因移動檔案而變成新量測。

自有 mlx-lm fork 的發布／dependency 切換仍未完成，見 [fork 狀態](MLX_LM_FORK.md)。
本次保留既有模型運算來源與 V4/V4.1 的底層建構方式；消除其 module globals
替換、搬移完整架構到 fork、進一步共用 row store，均需另行驗證。
本次沒有新的完整模型記憶體或效能結論；驗證結果見 [VALIDATION](VALIDATION.md)。
