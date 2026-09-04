# Whallm

<p align="center">
  <img src="Packaging/AppIcon.png" alt="Whallm アプリアイコン" width="160">
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

> [!NOTE]
> Whallm の旧称は DeepSeekV4SSD です。改名前のリリースでは旧称のアプリ名と ZIP 名を使用しています。

Whallm は、[Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare) から着想を得て、
SSD から routed expert をストリーミングします。M シリーズ Mac で
`DeepSeek-V4-Flash-0731` の全 2840 億パラメーターを実行でき、
`Qwen3.8-Flash-Next-FP8` text checkpoint にも対応します。

## ピークメモリーの目安

| モデル | 測定されたピークメモリー |
| --- | ---: |
| `DeepSeek-V4-Flash-0731` | 32.84–35.70 GiB |
| `Qwen3.8-Flash-Next-FP8` | 20.92–22.75 GiB |

v1.1.4 では、1,024～16,384 input token の chat prompt を測定しました。
これらは測定値であり、最小メモリー要件や性能を保証するものではありません。
prompt の長さ、tool、cache の状態、runtime 設定によってピークメモリーは変わります。
[完全なベンチマーク](BENCHMARK.md)と[検証記録](docs/VALIDATION.md)を参照してください。

## ベンチマーク

v1.1.4 の測定環境は、Apple M5 Pro、64 GB ユニファイドメモリー、1 TB ストレージを
搭載した MacBook Pro です。SPEED-Bench の mixed prompts を使用し、各 input size を
3 回実行、output 上限を 64 tokens としました。3 回の nearest-rank P95 は最大値と
同じです。TTFT は最初の token までの待機時間です。括弧内は v1.1.0 比の改善率で、
正の値は改善、負の値は悪化を表します。合計時間、TTFT、メモリーは低いほど良く、
Prefill と Decode は高いほど良い値です。

### DeepSeek V4 Flash 0731

| Input token | P95 合計時間 | P95 TTFT | P95 Prefill | P95 Decode | ピークメモリー |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 27.10 s(+42.0%) | 17.75 s(+52.4%) | 59.9 tok/s(+188.7%) | 7.8 tok/s(+15.6%) | 32.84 GiB(−42.6%) |
| 2,048 | 27.35 s(−1.0%) | 17.60 s(−4.0%) | 117.3 tok/s(+8.5%) | 7.3 tok/s(+11.7%) | 33.26 GiB(−0.2%) |
| 8,192 | 50.12 s(−5.2%) | 40.47 s(−6.6%) | 206.3 tok/s(−1.5%) | 7.4 tok/s(+10.3%) | 34.50 GiB(+0.7%) |
| 16,384 | 88.27 s(−2.6%) | 78.61 s(−3.0%) | 209.0 tok/s(−1.5%) | 7.2 tok/s(+8.0%) | 35.70 GiB(−0.2%) |

### Qwen3.8 Next Flash FP8

v1.1.4 では、expert cache のヒット率を高めるため、slot 数を 1,152 から 4,096 に
増やしました。その代わり、メモリー使用量も増加します。

| Input token | P95 合計時間 | P95 TTFT | P95 Prefill | P95 Decode | ピークメモリー |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 22.60 s(+7.6%) | 15.65 s(+9.0%) | 69.2 tok/s(+15.7%) | 10.4 tok/s(+11.6%) | 20.92 GiB(−37.7%) |
| 2,048 | 31.42 s(+22.3%) | 24.90 s(+23.7%) | 87.5 tok/s(+34.7%) | 9.8 tok/s(+18.1%) | 21.26 GiB(−29.6%) |
| 8,192 | 84.88 s(+40.4%) | 77.74 s(+42.2%) | 111.9 tok/s(+81.4%) | 10.4 tok/s(+25.0%) | 21.90 GiB(−22.3%) |
| 16,384 | 157.89 s(+42.9%) | 150.33 s(+43.9%) | 113.3 tok/s(+83.4%) | 9.7 tok/s(+23.9%) | 22.75 GiB(−20.3%) |

性能は prompt、SSD の速度、cache の状態によって変わります。
[完全なベンチマーク](BENCHMARK.md)と[検証記録](docs/VALIDATION.md)を参照してください。

## 使い方

**アプリをダウンロード → アプリを開く → モデルを選択してダウンロード →
server を起動 → アプリで対話または Codex に接続**

> [!IMPORTANT]
> 以前の Sparkle signing key が使用できないため、バージョン 1.0.3 から
> 1.0.4 へ自動更新できません。アプリを終了し、
> [1.0.4 release](https://github.com/yanun0323/Whallm/releases/tag/v1.0.4)から
> `DeepSeekV4SSD-macOS-arm64.zip` をダウンロードして、既存のアプリを手動で
> 置き換えてください。1.0.4 のインストール後は自動更新を使用できます。

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases/latest) から
   最新の `Whallm-macOS-arm64.zip` をダウンロードします。
2. ZIP を展開して `Whallm.app` を開きます。
3. **Model** ページを開きます。DeepSeek または Qwen を選択し、
   **Download Model** を選択します。アプリが必要なストレージを確認します。
   Qwen では、公開済みの MXFP4 installed model をダウンロードします。
   ダウンロードは中断して後で再開できます。
4. **Server** ページを開き、**Start Server** を選択します。server は installed model
   がなくても起動できますが、生成には installed model が必要です。
5. chat を開いてモデルを選択します。下記の設定で Codex を接続することもできます。

既定のローカル server アドレスは `http://127.0.0.1:11434` です。

![Whallm アプリ](docs/assets/deepseekv4ssd-app.png)

## 必要環境

| 項目 | 要件 |
| --- | --- |
| Mac | Apple Silicon M シリーズ Mac |
| macOS | macOS 15 以降 |
| ユニファイドメモリー | 64 GiB 以上 |
| 空きストレージ | アプリが選択したモデルと既存の部分データを確認 |
| モデル用ストレージ | 高速な内蔵、Thunderbolt、または USB4 SSD |
| インターネット | モデルとアプリの更新のダウンロードに必要 |

> [!IMPORTANT]
> Whallm は実験的なソフトウェアです。アプリにモデルの重みは含まれません。
> 他の機器から接続する必要がない場合は、既定のローカルアドレスを使用してください。

## Codex `config.toml` 設定

Whallm で server を起動します。次の設定を `~/.codex/config.toml` に追加します。

```toml
model = "deepseek-v4-flash-0731"
model_provider = "deepseek-v4-ssd"
model_reasoning_effort = "high"

[model_providers.deepseek-v4-ssd]
name = "Whallm"
base_url = "http://127.0.0.1:11434/v1"
wire_api = "responses"
requires_openai_auth = false
```

ファイルを保存した後、Codex を再起動してください。ローカルアドレスでは API key は
不要です。provider 設定はユーザーレベルの設定ファイルに配置してください。他の設定は
[Codex 公式設定リファレンス](https://developers.openai.com/codex/config-reference/) を参照してください。

## その他の技術情報

### 動作方式

- main model は全 2840 億パラメーターを持ち、1 token あたり約 130 億パラメーターが有効になります。
- common tensor はユニファイドメモリーに常駐します。
- routed expert は checkpoint ネイティブの FP4 重みを使用します。runtime は必要な
  routed expert を SSD から読み込みます。
- runtime は FP8 KV cache と上限付き expert cache でメモリー使用量を制御します。
- installed model は固定された checkpoint revision に対して検証されます。
- server の起動時には installed model の一覧だけを読み取り、モデルの重みは
  読み込みません。最初の generation request が指定したモデルを読み込みます。
- server が保持する読み込み済みモデルは 1 つです。request が別のモデルを指定すると、
  古い runtime を終了してから新しい runtime を読み込みます。
- **Model** ページでモデルを読み込むか、解放できます。読み込んだモデルは
  **Loaded** セクションに移動します。
- DeepSeek layer-major prefill のしきい値は変更できます。既定値は、cache にない
  1,024 prompt token です。

### モデル容量と DSpark

- main model は約 145 GiB を使用します。
- DSpark は約 10.12 GiB を追加します。新しい DeepSeek のダウンロードには常に DSpark が含まれます。
- DSpark はインストールしても自動で有効になりません。speculative decoding を試すときは、
  runtime 設定で **Use DSpark** を有効にします。
- main model を再インストールせずに DSpark を削除できます。
- Qwen installed weight ファイルは 125,268,506,112 bytes を使用します。
  Qwen は DSpark をサポートしません。
- Qwen では検証済みの MXFP4 installed model をダウンロードします。モデルの
  インストール時にユーザーの Mac で Qwen checkpoint を量子化することはありません。

### OpenAI 互換 server

server は次の endpoint をサポートします。

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /api/models/load`
- `POST /api/models/unload`

固定 API model ID は `deepseek-v4-flash-0731` と
`qwen3.8-flash-next-fp8` です。各モデルの **Advanced Settings** で任意の Alias を
設定できます。有効な変更は自動で保存されます。generation request は API model ID
または Alias を受け付けます。chat のモデル選択には、server の起動時に使用できた
installed model だけが表示されます。server の実行中にダウンロードが完了した場合は、
server を再起動してください。

Responses API は Codex tool と OpenAI function tool をサポートします。API client が
各 tool を実行し、結果を server に返す必要があります。field、例、現在の制限は
[API ガイド](docs/API.md)を参照してください。

### メトリクスとプライバシー

アプリは prefill 速度、decode 速度、token 数、メモリー使用量、SSD 読み込み速度、
cache hit rate、最初の token までの時間、完了時間を表示します。読み込み済みの
モデルが変わると、アプリは metric 履歴を消去します。

推論は Mac 上で実行されます。接続した client が別の場所に送信しない限り、prompt と
生成文はローカル runtime に残ります。アプリは、モデルのダウンロード、更新の確認、
設定済み API request の受け付けにネットワークを使用します。

### 現在の制限

- runtime は現在のドキュメントに記載された 2 つの固定 checkpoint revision だけを
  サポートします。
- Qwen は text のみをサポートします。Qwen vision、video、MTP、DSpark は
  サポートしません。
- 記録済みの M5 Pro 環境で、Qwen の full-model SHA-256、text、thinking、tool call、
  greedy 4K、prompt cache、packaged App の検証に合格しました。詳細は
  [Qwen サポート状況](docs/QWEN.md)を参照してください。
- server が保持する読み込み済みモデルは 1 つで、一度に 1 つの generation request を
  処理します。他の generation request は現在の request stream がすべて終了するまで
  待機します。
- 画像、音声、logprobs、`response_format`、`stop` はサポートしません。
- request body の上限は 1 MiB です。
- 長い input と output にはより多くの KV cache メモリーが必要です。
- 性能は SSD の速度、input 長、cache の状態で変わります。

モデル契約、runtime 設計、検証、性能、研究結果については、
[最新ドキュメント](docs/README.md) を参照してください。

Whallm は DeepSeek と提携していません。モデルをダウンロードして使用する前に、
モデルの利用条件を確認してください。

## ライセンス

Whallm のソースコードは [MIT License](LICENSE) で公開されています。
モデルウェイトは含まれておらず、独自の規約が適用されます。
