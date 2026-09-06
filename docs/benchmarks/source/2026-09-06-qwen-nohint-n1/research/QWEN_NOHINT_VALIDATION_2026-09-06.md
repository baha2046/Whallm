# Qwen 無提示 expert 排序：新輸入與長生成驗證

## 執行前固定規則

接續既有無提示候選；不重新測試已因品質回退停止的開提示版本。
使用者要求持續研究，只有需要決策才停止。此輪仍不改 runtime 預設。

先執行 N1：四個新編題目（程式碼、繁中、數學、工具格式），每題分別做
1,024 與 16,384 token 輸入、最多 1,024 token 生成。
題目在本輪首次生成之前固定，沒有用於候選選擇；不是公開能力測驗，也不是長文件理解測試。
長輸入以明確排除於任務外的 padding 填滿；長生成用於比較輸出與 Decode 時間，
截斷可以比較相同長度的運算，但不能算成已完成任務或正確回答。
每題兩輪相反順序的原版／候選，共 32 個獨立 process。
每次固定 greedy、thinking 關閉、MTP 關閉、4,096 slots、48 GiB、ANE 比例 0.25。
Prompt cache 停用，expert cache 從空開始，OS page cache 不清除。

候選只使用既有 `research_qwen_sorted_experts.py --no-sorting-hint`。
每次核對 source、installed manifest、prompt/output hashes、修改呼叫數、expert bytes 和 QMM 呼叫數。
任何一對完整 token 序列不同或 logical expert bytes／QMM 呼叫數不同，立即停止 N1，
以同一題反向順序再跑一對確認；不拿速度抵銷差異。
若確認不同，回到數值診斷；不接著做 cache 或 runtime 整合，也不直接判成品質下降。

完成 N1 後，以每題兩對相對變化的中位數判定：每個長度跨四題 TTFT 改善中位數至少 5%；
每題 Decode 不得退步超過 5%，Decode p95 不得增加超過 10%；
每題 MLX peak 與 process RSS 中位數不得增加超過 5%。
至少程式碼的兩個長度都須實際生成 512 tokens，才能結束較長 Decode 的待查項。
只要某個長度未通過，先解釋原因與適用範圍，不將它直接擴成所有長度預設。
兩對只作延伸驗證，不能提供可靠的尾端分位數信賴區間。

N1 通過後執行 N2：同一 shared prefix 的 A/B 分支、返回 A、跨 process 重啟，
以及原版寫入／候選讀取與候選寫入／原版讀取；對照各分支未快取輸出。
必須觀察實際 cache reuse 且輸出完全相同，記錄 checkpoint／reuse 長度與格式合約。
N2 的具體輸入、次序及預期 cache 邊界在執行前另固定。

N1/N2 通過才進入預設關閉的 runtime 整合與針對性測試；整合後另跑反向 A/B。
是否改成預設是最後需要使用者決策的步驟，屆時提供已驗證結果、適用範圍與三個選項。

## 證據

N1 原始輸出保存於 `scratch/qwen-nohint-validation-2026-09-06/n1/`。
執行工具會保存固定 prompts、完整 source snapshot、命令、版本、設定與逐次 metrics。
驗證完成後才將結果與可重現來源移入 `docs/benchmarks/`。
