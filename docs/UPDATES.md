# App 更新

Settings 的獨立 **Updates** 區塊提供下列設定；Preferences 保留語言設定。
更新來源選項靠齊卡片右側，與檢查更新按鈕及自動檢查開關對齊。


- **stable**（預設）：只接收正式版本。
- **dev**：另外接收標記為 `dev` 的測試版本；較新的正式版本仍可更新。
- **Automatically check for updates**：控制 Sparkle 的背景檢查。新安裝預設開啟；
  已有使用者沿用 Sparkle 保存的選擇。關閉後仍可手動檢查。

來源選擇保存於 `updateChannel`。自動檢查直接使用 Sparkle 的
`automaticallyChecksForUpdates`，不另存第二份開關。檢查期間暫停切換來源。
切換回 stable 不會降級已安裝的 dev；需等正式版的版本編號更高。

## 更新清單

共用網址：<https://yanun0323.github.io/Whallm/appcast.xml>。
GitHub Pages 使用 `gh-pages` 分支根目錄；該分支只保存已簽章的
`appcast.xml` 與 `.nojekyll`，App ZIP 仍放在 GitHub Releases。

正式版 item 不設定 `sparkle:channel`；測試版設定為 `dev`。
App 在 stable 回傳空的 allowed channels，在 dev 回傳 `dev`。
清單、下載檔的 Sparkle 簽章及 App 公證要求保持啟用。
舊版 App 仍使用最新正式 GitHub Release 的 appcast 附件。

## 發布

先完成專案要求的本機封裝與驗證，並準備對應版本的 release notes。
沿用 Developer ID、Keychain 公證 profile 與 Sparkle signing account `deepseek_ssd`。
發布前須提交全部版本內容，建立並推送對應 tag；腳本在打包前與上傳前確認
工作目錄乾淨、HEAD 及本機／遠端 tag 都指向同一 commit，並以 `--verify-tag` 發布。

```sh
# 正式版（完成版本 commit 後）
git tag v1.1.7
git push origin refs/tags/v1.1.7
make release VERSION=1.1.7

# 同一版本的第 1 個 dev build（完成版本 commit 後）
git tag v1.1.7-dev.1
git push origin refs/tags/v1.1.7-dev.1
make release VERSION=1.1.7 CHANNEL=dev DEV_BUILD=1
```

上述版本僅為命令範例，不代表已發布。
第一個 dev 指定 tag 為 `v1.1.7-dev`，命令另外加入 `TAG=v1.1.7-dev`；Release 標題為
`Whallm 1.1.7-dev`，內部 build 仍為 `1.1.7d1`。

第二個 dev 的 Alpha 使用 `VERSION=1.1.7 CHANNEL=dev DEV_BUILD=2`，
tag 為 `v1.1.7-dev.2`、內部 build 為 `1.1.7d2`。
`RELEASE_TITLE='Whallm 1.1.7-dev.2 (Alpha)'` 可指定 GitHub 標題，
`RELEASE_NOTES_FILE=Packaging/ReleaseNotes/1.1.7-dev.2.md` 指定獨立說明。
Alpha 仍屬 dev 來源，不新增更新來源。發布一律使用 distribution build，
排除 local-only Dry run；上傳前與下載後的隔離啟動都驗證此設定。

Dev 的 Git tag 預設為 `v1.1.7-dev.1`，`CFBundleShortVersionString` 為 `1.1.7`，
`CFBundleVersion` 為 `1.1.7d1`。`DEV_BUILD` 可用 1–255；提高它來發布下一個測試版。
Sparkle 排序為 `1.1.6 < 1.1.7d1 < 1.1.7d2 < 1.1.7`，因此之後的正式版能取代 dev。
Dev 不接受不同於計算結果的 `BUILD_VERSION` 覆寫；GitHub 標記為 pre-release，
並設定 `--latest=false`。

發布腳本在打包前讀取 `gh-pages` 的已簽章清單，再由 Sparkle `generate_appcast`
加入本次版本，保留各來源的最新項目並重新簽章。Release 的 App／ZIP 驗證維持原流程。
下載已發布 ZIP 再次驗證成功後，才把相同 appcast 提交到 `gh-pages`。

Pages checkout 保留打包前的基底；若有人同時更新清單，普通 git push 會拒絕覆蓋。
若 Release 已發布而 Pages push 失敗，先核對最新 `gh-pages`，從該清單重新生成並
簽章，保留其他人的新版本；不要 force push，也不要刪掉 Release 重新發布。
GitHub Pages 部署完成後才會對外提供新清單。

## 驗證範圍

目前已發布 `v1.1.7` 正式版（build `1.1.7`），GitHub 下載 ZIP 的簽章、公證、
三語隔離啟動及 local-only 功能關閉檢查通過；stable 與 dev 使用者均可更新到 `1.1.7`。
完整成品 hash 與測試範圍見 [發布驗證](VALIDATION.md#2026-09-14whallm-117-正式發布)。
較早的 `v1.1.7-dev.2` Alpha 驗證另見 [Alpha 發布紀錄](VALIDATION.md#2026-09-14whallm-117-dev2-alpha-發布)。

Swift 測試覆蓋來源保存、stable／dev 篩選、自動檢查與 Sparkle 同步、dev 版本排序及三語文案。
以下是前一版 `v1.1.7-dev`（build `1.1.7d1`）的驗證：GitHub 下載 ZIP 的簽章、公證、三語系與
隔離啟動均通過；公開 Pages 清單與 Release 附件一致，英文 release notes 內容也一致。
獨立 App 副本實測 stable 不提供 dev，dev 找到本次版本並顯示英文說明。
此次沒有透過 Sparkle 執行原地安裝替換。

參考：[Sparkle channels](https://sparkle-project.org/documentation/publishing/#channels)、
[Sparkle 設定 API](https://sparkle-project.org/documentation/api-reference/Classes/SPUUpdater.html)、
[GitHub Pages 分支來源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
