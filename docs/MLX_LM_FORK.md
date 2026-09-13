# 自有 mlx-lm fork 準備狀態

核對日期：2026-09-12。
[已接受決定](adr/0001-owned-mlx-lm-fork.md)：以官方 mlx-lm 建立自有 fork。
GitHub repo：[yanun0323/mlx-lm](https://github.com/yanun0323/mlx-lm)。

## 目前狀態

- Whallm 的 `requirements.txt` 仍固定
  `Blaizzy/mlx-lm@5c10538136b9038b9626c134612b08afc18d697a`。
- 本機 fork 位於 `/Users/Shared/Project/test/mlx-lm`，
  分支 `codex/whallm-compat`，基底為官方 v0.31.3
  `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`。
- 本機已套入相對該官方基底的六檔相容修改，尚未 commit 或 push。
  GitHub 上新建的 fork 尚不含這份相容修改。
- 本機 `.venv` 與目前 `dist` 未切换來源；驗證使用 `PYTHONPATH` 暫時載入自有副本。

沒有直接使用當時官方 main `dcbcf786c0cf56f9a12fabe9468c887781431ae2`，
因為首次更換維護來源應維持現有行為，上游升級需獨立驗證。

## 相容差異與驗證

六個檔案：`mlx_lm/generate.py`、`mlx_lm/models/cache.py`、
`mlx_lm/models/deepseek_v4.py`、`mlx_lm/models/hyper_connection.py`、
`mlx_lm/utils.py`、`tests/test_models.py`。
相對官方基底的來源差異共 2,192 行新增、2 行移除。
184 個 package 檔案與目前 Blaizzy 固定 commit 逐檔相同；
安裝副本未包含的七個 examples 檔案另列，沒有套件運算程式差異。

| 檢查 | 結果 | 限制 |
|---|---|---|
| 確認 import path | `mlx_lm` 與 `deepseek_v4` 來自本機 fork | 暫時載入，不是已發行 dependency |
| Whallm runtime tests | 389 項通過 | 不等於完整模型驗證 |
| mlx-lm DeepSeek V4 小模型測試 | 1 項通過 | 未執行整套上游模型測試 |

證據保存在 [scratch 目錄](../scratch/modular-runtime-2026-09-12/)：
`fork-audit.json`、`runtime-tests.log`、`deepseek-v4-test.log`、
`mlx-lm-compat.patch`。Patch SHA-256：
`4759d3bd106d6b09f1dc8ca007df46a7384a8d44650ed07ddb9fae611eb0ab4c`。
這是功能與來源驗證，沒有新的完整模型輸出或效能結果。

## 下一步

1. 將本機相容修改提交並發布到自有 fork 的固定 commit。
2. Whallm dependency 切到該 commit，驗證乾淨安裝、既有測試、打包與隔離啟動。
3. [模型支援套件](MODEL_PACKAGES.md) 已接入 Whallm；後續在 fork 加入建構注入入口，
   逐一移入模型運算，取代目前 V4／V4.1 的全域類別替換。

檢查到的 [上游 AGENTS.md](https://github.com/ml-explore/mlx-lm/blob/dcbcf786c0cf56f9a12fabe9468c887781431ae2/AGENTS.md)
寫明「Do NOT run `git push` or create a PR on behalf of the user」。
本輪因此停在可檢查的本機修改，沒有替使用者推送或建立 PR。
未來若向官方提交新模型，使用 [new_model.md 範本](https://github.com/ml-explore/mlx-lm/compare?template=new_model.md)。
