# Qwen 無提示排序：快取驗證事前規則

僅在 [N1 新輸入與長生成驗證](QWEN_NOHINT_VALIDATION_2026-09-06.md) 通過後執行。
此文件目前是下一階段的準備，不表示已驗證或已採用。

使用同一段新編 shared prefix，加上兩個不同的程式碼任務 A／B。
執行前固定 rendered prompts 與 token hashes；A／B 各 2,048 tokens，
shared checkpoint 是前 1,024 tokens，每個分支未快取部分也是 1,024 tokens，
讓分支也實際經過 batched expert 路徑。
每次生成最多 128 tokens，greedy、thinking／MTP 關閉、4,096 slots、48 GiB、ANE 比例 0.25。
維持一般記憶體快取兩筆與磁碟快取八筆限制，使用全新隔離目錄。

固定順序：

1. 原版與候選分別在獨立 process 執行未快取 A、B，作為各分支輸出對照。
2. 原版與候選各在新 process 用 `warm_prompt` 保存 shared checkpoint，
   在任何分支執行前複製該 checkpoint 為不可變的重啟輸入。
   隨後同 process 執行 A、B、A，記錄每次實際重用來源與長度。
3. 從兩份未經分支改寫的 checkpoint 各複製新工作目錄；
   讓原版、候選分別在新 process 讀取，執行 B、A。
   這同時涵蓋同版本重啟、原版寫入／候選讀取、候選寫入／原版讀取。

所有分支輸出須與其未快取對照完全相同，兩版本未快取對照也須相同。
每次快取 request 必須重用至少 shared checkpoint 長度。
每個候選的 cold、warm prefix 和第一個分支都必須有實際修改呼叫；
重用完整分支而僅重算最後一個 token 時，修改呼叫為零屬預期行為，另行記錄。
保留每次呼叫前可匹配的 memory／persistent entry 和實際 metrics 以核對來源，
不以「啟用 cache」代替命中證據。

Shared checkpoint 的 tokens、format 5、contract 與 tensor payload hashes 必須兩版相同，
且分支執行後該 checkpoint 內容不能變。
若原版冷／熱輸出已不同，先定位既有快取或分塊差異；不能歸因為候選退步。
任何合約未通過都停止相依的 runtime 整合，保留失敗證據後診斷。
此輪只判定功能與狀態一致性，不宣稱快取效能改善，也不是完整 tool-agent 品質測試。

## N2 結果

10 個獨立 processes、18 次 128-token 回答全部通過。
其中 4 次為未快取對照，14 次實際重用快取；每個分支的所有回答都與其冷啟動對照相同。

原版與候選的 A→B→A 次序均實際重用 1,024／1,024／2,047 tokens，
來源依序為 memory／persistent／persistent。
四種 writer／reader 組合的重啟 B→A，兩次均重用 1,024 tokens。
這確認了 default memory 兩筆限制下的分支與重啟行為，沒有為測試加大快取容量。

兩版 shared checkpoint 都是 format 5、1,024 tokens、8 個內容識別區塊，
122 個 tensor 的名稱、dtype、shape、原始數值 bytes 和狀態 schema 全部相同。
分支執行前後 shared checkpoint 的檔案內容也未改變。
所有 ANE controller 都維持設定比例 0.25、沒有 fallback；
本組 chunk 未呼叫 ANE projection，evaluation counter 均為零，不把它宣稱為 ANE 執行驗證。

- [18 次回答與完整 cache 合約](../docs/benchmarks/2026-09-06-qwen-nohint-n2-cache-m5-pro.json)。
- [來源與證據索引](../docs/benchmarks/2026-09-06-qwen-nohint-n2-index.json)。

N2 通過後，預設關閉的 runtime 整合與 [N3](QWEN_NOHINT_INTEGRATION_2026-09-06.md) 也已通過。
執行前規則與當時程式保存在上述索引中；目前 worker 新增的 `integrated` 分支供 N3 使用，
不能把這個後續版本的雜湊冒充 N2 執行時的版本。

使用者後續選擇預設開啟，目前已採用為 Qwen 預設；此處保留 N2 執行當時的設定。
