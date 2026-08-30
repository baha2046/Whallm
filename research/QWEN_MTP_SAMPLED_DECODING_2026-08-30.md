# Qwen MTP sampled decoding

## 目標

移除 Qwen MTP 的 `temperature=0` 限制。
MTP 必須保持主模型調整後的 sampling 分布。
MTP 維持預設關閉。

## 外部依據

Leviathan、Kalman 和 Matias 的
[Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
定義 exact speculative sampling。

對 draft token `x ~ q`，接受機率是 `min(1, p(x) / q(x))`。
拒絕 draft token 時，修正分布是 `normalize(max(p - q, 0))`。
全部 draft token 都被接受時，額外 token 從最後一個 target 分布抽樣。
該演算法保持 target model 的輸出分布。

MLX LM 的
[`make_sampler`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)
依序套用 Top P、Min P 和 Top K，再使用 Temperature 執行 categorical sampling。
本專案使用相同順序建立 MTP 與 main model 的調整後分布。

## Runtime 合約

Qwen MTP 對 draft distribution `q` 和 target distribution `p` 套用相同設定：

- Temperature。
- Top P。
- Min P。
- Top K。
- request 的 logit processor。

每個位置使用該位置可見的 token prefix。
Runtime 只提交已接受的 draft prefix。
Runtime 在第一個拒絕位置提交 correction token。
Runtime 在全部接受時提交 target bonus token。
Runtime 回復未提交的 MTP cache 和 target cache state。

Temperature 是 0 時，runtime 保留原本的 argmax 驗證路徑。

## 假設

- MTP 與 main model 使用相同 tokenizer vocabulary。
- Sampling filter 不會移除全部 token。
- MLX categorical sampling 和現有 DSpark verifier 的亂數行為可重用。
- 一個 zero-acceptance round 仍會觸發 request-local normal fallback。

## 驗證與停止條件

單元測試必須證明：

1. 相同的 draft 與 target 分布會接受 sampled draft token。
2. 不相交的分布會拒絕 draft token，並從 correction 分布抽樣。
3. Greedy MTP 的既有 token 與 cache 測試維持通過。
4. Model catalog 接受非零的 MTP default Temperature。
5. App 不再把 MTP 的 Temperature 強制設為 0。

若 sampled MTP 產生 NaN、無效 token、cache offset 錯誤或 greedy regression，
則停止採用。

## 證據限制

2026-08-30 的 Apple M5 Pro 64 GB smoke test 使用目前 working tree、
完整 installed model、Temperature 0.7、Top P 0.8、Top K 20、32 MTP slots、
8 個 prompt token 和 6 個 output token。
Runtime 成功產生 6 個 token。
Output token SHA-256 是
`09dd33f15b02bac1782daf7a802d6f5c981116b78facc8250d3a970e050ae356`。
第一個 MTP round 提出 4 個 token，接受 0 個 token，並正常 fallback。
這是 sampled execution smoke test，不是 output distribution 或效能證據。

目前單元測試證明 sampling contract 和 cache 行為。
目前沒有完整 installed model 的 sampled output distribution 統計檢定。
目前沒有 sampled MTP 的正式效能結果。
既有 greedy output parity 和效能結果不能作為 sampled MTP 的效能證據。
