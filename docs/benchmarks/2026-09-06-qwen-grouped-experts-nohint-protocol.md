# Qwen expert 計算：先按 expert 排序

## 假設與目前證據

4K code 的同步 observer 量到 MoE Prefill 約 19.8 秒，是本輪已量組件中最大的一項。
它不含外部 full-layer 讀取，不能把時間直接叫做 SSD 等待。
目前 Qwen 的 batched Prefill 直接將未排序的 assignments 傳入 `gather_qmm`。
同專案的 DeepSeek 已使用 MLX-LM 的 `_gather_sort`／`_scatter_unsort` 與 `sorted_indices`。
因此先研究：將分配到相同 expert 的輸入放在一起，能否改善 Qwen Prefill 的計算效率。

論文依據是 [ScatterMoE §2.1–2.2](https://arxiv.org/html/2403.08245v1)
與 [MegaBlocks](https://arxiv.org/abs/2211.15841)：按 expert 組織運算與資料搬移的取捨。
本候選只重用現有 MLX 排序／還原工具，會建立額外輸入副本；不是這兩篇論文 kernel 的重現。
論文不證明 native MXFP4、Apple GPU、本模型的收益。

## 事前規則

先獨立測 code 4K／32，使用與 QSA 實驗相同的正常設定與輸入。
只改 batched Prefill、assignments 至少 64 時的 expert 計算順序，還原後才按原 scores 加總。
路由、top-k、expert bytes、gate/up 順序、量化、Decode expert 路線都維持原合約。
必須完整 output token hash 相同，否則停止精確路線，不以「公式相同」放行。
預期 192 次 Prefill expert 呼叫經過候選，記錄實際計數核對。
通過後，與控制做相反順序的重複測試，再進入五類／256-token 評估；
速度／記憶體門檻沿用本輪 QSA 計畫，兩個候選先分開，不混在同一個實驗。

## 初次篩選與後續診斷

第一次 code 4K／32 已完成，192 次 Prefill 呼叫經過候選。
TTFT 為 24.364 秒，但與控制從第 5 個 token 起分歧，因此**沒有通過精確輸出條件**。
這只是速度信號，不是可採用的加速結果；不同文字也尚不能判定能力變好或變差。
初次候選腳本的完整原文與 hash 保存在 `scratch/qwen-grouped-experts-2026-09-06/first-candidate-source.py`。

接著先固定 layer 0／23／47 的第一個 Prefill chunk：核對排序／還原的 expert IDs 與輸入列完全相等；
比較開／關 `sorted_indices` 兩個版本與原始輸出的相對誤差。
這個 observer 一律回傳原始輸出，不把候選誤差帶入後續模型。
若關閉提示即可消除誤差，再測其完整輸出及速度；若仍不同，先定位差異，不進入品質採用。

本輪查閱的 [MLX v0.32.0 原始碼](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/quantized.cpp)
在 `GatherQMM::eval_gpu` 對右側已排序、M=1、B≥16、B/E≥4 使用不同的 grouped 路徑；
目前 1,024×10 assignments／512 experts 的形狀符合此條件。
這支持把排序提示與運算路徑分開診斷，但尚未以 GPU trace 證明本機 wheel 的內部 dispatch。
上述上游版本檔案已保存 URL 對應的 raw 內容；安裝 wheel 的版本是 0.32.0。

狀態：QSA 成對測試執行中，expert 三層診斷已準備，依序執行以免 GPU 測試互相干擾。

## 若必須改用任務品質判定

目前尚未決定可容忍的品質退步，也未開始完整近似候選的能力評測。
輸出文字不同不能換算成品質下降比例；先完成上面的實作／數值診斷。
如果速度收益依賴會改變輸出的計算路徑，採用前需由使用者決定品質門檻。

評估方法的論文依據可包括 [HumanEval](https://arxiv.org/abs/2107.03374) 的程式功能正確性、
[GSM8K](https://arxiv.org/abs/2110.14168) 的數學答案正確性，與
[C-Eval](https://arxiv.org/abs/2305.08322) 的中文知識／理解評估。
這些各有涵蓋範圍：C-Eval 不能單獨代表繁中對話品質；工具呼叫還需本專案 schema／參數測試。
它們是測量方法的來源，不是本候選已通過的證據。
對照與候選必須使用相同的 held-out 題目與生成條件，按能力類別分開報告，
不以某一類的提升抵銷另一類的退步，也不把少量題目零回退宣稱成普遍品質保證。
