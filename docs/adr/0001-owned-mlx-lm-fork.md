---
status: accepted
---

# 自行維護官方 mlx-lm fork

2026-09-12 使用者選擇以官方 mlx-lm 為基礎維護自己的 fork，
沿用現有模型實作和 MLX 運算，逐步建立 Whallm 需要的擴充入口。
相較從零重寫底層，此路線保留已有驗證和上游相容性，但需要自行維護差異與追蹤上游。

目前僅完成 [fork 與本機相容準備](../MLX_LM_FORK.md)，正式 dependency 尚未切換。
模組化架構已採用 [模型支援套件](0002-model-support-packages.md)，並完成 Whallm 端的第一階段實作；
模型運算搬移與 fork 建構入口仍待完成。
