# Qwen3.8-Flash-Next 來源查核

本文件記錄外部來源和目前證據限制。
本文件不是 runtime 結果。

## 一手來源

1. [固定 revision 的官方 FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/bcd9f01ddc9cff2316eb84281bebcd5b058bddce)
   提供 config、generation config、tokenizer、chat template、index 和 safetensors。
2. [Transformers `Qwen4Exp` 實作](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
   定義 Gated DeltaNet、QSA、MoE、hyper-connection、N-gram 和 PLE 行為。
3. [MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.quantize.html)
   定義 MLX MXFP4 quantization 入口。

## 已採用的結論

- checkpoint revision 必須固定。
- vision 和 MTP tensor 不可進入第一版 installed model。
- routed expert 由 F8_E4M3 和 128×128 BF16 inverse scale 轉成 MXFP4。
- common tensor 保留 checkpoint dtype。N-gram table 保留 F8_E4M3。
- MXFP4 正確性的停止條件是固定測試向量與 MLX byte-identical。
- 262,144 是 checkpoint 合約值。它不是本機驗證結果。

## 證據限制

- MXFP4 會改變 routed expert 數值。輸出不保證等同官方 FP8。
- 完整模型已通過本機功能與 4K 驗證，但沒有官方 FP8 output parity 證據。
- 目前只有一組 M5 Pro 本機量測。該結果不是效能保證。
- 官方 FP8 checkpoint 是目前採用的固定合約。

## 完整驗證停止條件

完整驗證必須記錄 commit、環境、設定、cache state、SSD bytes、記憶體和
output token hash。完整驗證必須包含完整安裝與 SHA-256、文字對話、thinking
開關、forced tool call、greedy 4K，以及 cold/warm output token hash。
