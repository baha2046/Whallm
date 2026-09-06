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
