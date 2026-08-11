# 歷史研究封存

本目錄保存 2026-08-06 至 2026-08-11 的研究和計畫。
這些文件保留當時的問題、估計和本機量測。
這些文件不描述目前 runtime。

目前文件如下。

- [文件索引](../../docs/README.md)
- [架構](../../docs/ARCHITECTURE.md)
- [驗證紀錄](../../docs/VALIDATION.md)
- [效能與瓶頸](../../docs/PERFORMANCE.md)
- [研究結論](../../docs/RESEARCH.md)

## 封存內容

| 文件 | 原始目的 | 目前狀態 |
| --- | --- | --- |
| [deepseek-v4-flash-0731-turbofieldfare-feasibility.md](deepseek-v4-flash-0731-turbofieldfare-feasibility.md) | 初始可行性與容量估算。 | 主模型、repacker 和 DSpark 已實作。估算不是目前量測。 |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | M1 至 M4 的實作計畫。 | M1 至 M4 已完成。 |
| [RUNTIME_RESEARCH_2026-08-07.md](RUNTIME_RESEARCH_2026-08-07.md) | 初始 runtime 可行性和支援範圍。 | 依賴和 DSpark 狀態已過時。 |
| [RUNTIME_SPEED_RESEARCH_2026-08-07.md](RUNTIME_SPEED_RESEARCH_2026-08-07.md) | Layer-major 後的效能方向。 | 部分方向已實作。估計值不是結果。 |
| [DSPARK_FIRST_PLAN_2026-08-08.md](DSPARK_FIRST_PLAN_2026-08-08.md) | DSpark-first 計畫。 | 安裝和 runtime path 已實作。 |
| [DSPARK_RUNTIME_OPTIMIZATION_2026-08-08.md](DSPARK_RUNTIME_OPTIMIZATION_2026-08-08.md) | DSpark cache 和 scheduler 研究。 | 部分 cache 問題已修正。 |
| [PREFILL_DECODE_RESEARCH_2026-08-08.md](PREFILL_DECODE_RESEARCH_2026-08-08.md) | Prefill 和 decode profiling 計畫。 | 多項 P0/P1 已實作。 |
| [DSPARK_80_PERCENT_OPTIMIZATION_RESEARCH_2026-08-09.md](DSPARK_80_PERCENT_OPTIMIZATION_RESEARCH_2026-08-09.md) | DSpark 80% 目標查核。 | 舊 verification path 已被取代。 |
| [EXPERT_STREAMING_RESEARCH_2026-08-09.md](EXPERT_STREAMING_RESEARCH_2026-08-09.md) | Handoff、MTLIO 和 staged streaming。 | Ready expert 已實作。其餘多為研究假設。 |
| [PREFILL_DECODE_RUNTIME_OPTIMIZATION_2026-08-09.md](PREFILL_DECODE_RUNTIME_OPTIMIZATION_2026-08-09.md) | Runtime 修改和實機結果。 | 是重要歷史紀錄，但預設值和測試數量已變更。 |
| [RUNTIME_PREFILL_DECODE_RESEARCH_PLAN_2026-08-10.md](RUNTIME_PREFILL_DECODE_RESEARCH_PLAN_2026-08-10.md) | R0 至 D3 的研究計畫、profile gate 和停止條件。 | 計畫已完成。所有 active trace 與 profile 工作已結束。 |
| [RUNTIME_SPEED_OPTIMIZATION_ANALYSIS_2026-08-10.md](RUNTIME_SPEED_OPTIMIZATION_ANALYSIS_2026-08-10.md) | Runtime 速度方向的推論分析與執行順序。 | 分析已完成。候選決定已整理到目前研究結論。 |

## 使用規則

- 引用歷史數字時，必須附日期、prompt、設定和版本。
- 不得把估計改善寫成目前結果。
- 不得把其他硬體或其他模型的結果寫成本專案結果。
- 如果歷史內容與目前程式碼衝突，請使用目前文件。
