# DeepSeekV4SSD

[![English](https://img.shields.io/badge/English-Click-yellow)](README.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-點擊查看-orange)](README-tw.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](README-cn.md)
[![日本語](https://img.shields.io/badge/日本語-クリック-青)](README-ja.md)
[![한국어](https://img.shields.io/badge/한국어-클릭-yellow)](README-ko.md)

DeepSeekV4SSD は、[Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare) から着想を得て、
SSD から routed expert をストリーミングし、M シリーズ Mac で約 30 GB のメモリーを
使って `DeepSeek-V4-Flash-0731` の全 2840 億パラメーターを実行します。

![DeepSeekV4SSD アプリ](docs/assets/deepseekv4ssd-app.png)

## ベンチマーク

測定環境は、Apple M5 Pro、18 コア CPU、20 コア GPU、64 GiB ユニファイドメモリーを
搭載した MacBook Pro です。DSpark は無効です。

| テスト | Prefill | Decode | ピークメモリー |
| --- | ---: | ---: | ---: |
| 14,000 input token の Codex request | 180 Tok/s | 6.5 Tok/s | 30 GB |
| 4,096-token prompt、1 token を生成 | 144.53 Tok/s | — | 15.56 GiB |
| 短い prompt、同じ runtime で 2 回目の実行 | — | 6.41 Tok/s | 15.05 GiB |

最初の 2 行は、それぞれ `v1.0.3` と `v1.0.2` runtime で測定しました。性能は
prompt、SSD の速度、cache の状態で変わります。詳細は
[検証記録](docs/VALIDATION.md) を参照してください。

## 使い方

**アプリをダウンロード → アプリを開く → 167 GB の全パラメーターモデルをダウンロード →
server を起動 → アプリで対話または Codex に接続**

1. [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest) から
   最新の `DeepSeekV4SSD-macOS-arm64.zip` をダウンロードします。
2. ZIP を展開します。`DeepSeekV4SSD.app` を開きます。
3. **Download Model** を選択します。既定のインストールは DSpark を含み、約 167 GB を
   使用します。ダウンロードは中断して後で再開できます。
4. モデルの準備が完了したら、**Start Server** を選択します。
5. アプリ内で対話するか、下記の設定で Codex を接続します。

既定のローカル server アドレスは `http://127.0.0.1:11434` です。

## 必要環境

| 項目 | 要件 |
| --- | --- |
| Mac | Apple Silicon M シリーズ Mac |
| macOS | macOS 15 以降 |
| ユニファイドメモリー | 64 GiB 以上 |
| 空きストレージ | 約 172 GB（160 GiB） |
| モデル用ストレージ | 高速な内蔵、Thunderbolt、または USB4 SSD |
| インターネット | モデルとアプリの更新のダウンロードに必要 |

> [!IMPORTANT]
> DeepSeekV4SSD は実験的なソフトウェアです。アプリにモデルの重みは含まれません。
> 他の機器から接続する必要がない場合は、既定のローカルアドレスを使用してください。

## Codex `config.toml` 設定

DeepSeekV4SSD で server を起動します。次の設定を `~/.codex/config.toml` に追加します。

```toml
model = "deepseek-v4-flash-0731"
model_provider = "deepseek-v4-ssd"
model_reasoning_effort = "high"

[model_providers.deepseek-v4-ssd]
name = "DeepSeekV4SSD"
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

### モデル容量と DSpark

- main model は約 145 GiB を使用します。
- DSpark は約 10.12 GiB を追加します。既定のダウンロードには DSpark が含まれます。
- DSpark はインストールしても自動で有効になりません。speculative decoding を試すときは、
  runtime 設定で **Use DSpark** を有効にします。
- main model を再インストールせずに DSpark を削除できます。

### OpenAI 互換 server

server は次の endpoint をサポートします。

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Responses API は Codex tool と OpenAI function tool をサポートします。API client が tool を
実行し、結果を server に返す必要があります。詳細は [API ガイド](docs/API.md) を参照してください。

### メトリクスとプライバシー

アプリは prefill 速度、decode 速度、token 数、メモリー使用量、SSD 読み込み速度、
cache hit rate、最初の token までの時間、完了時間を表示します。

推論は Mac 上で実行されます。prompt と生成文はローカル runtime に残ります。接続した
API client がデータを外部に送信する場合はあります。

### 現在の制限

- runtime は固定された `DeepSeek-V4-Flash-0731` checkpoint のみをサポートします。
- runtime は同時に 1 つの生成 request だけを処理します。
- 画像、音声、logprobs、`response_format`、`stop` はサポートしません。
- 長い input と output にはより多くの KV cache メモリーが必要です。
- 性能は SSD の速度、input 長、cache の状態で変わります。

モデル契約、runtime 設計、測定結果に基づく技術判断は、
[ランタイム調査](docs/RUNTIME_RESEARCH_2026-08-07.md) と
[実装計画](docs/IMPLEMENTATION_PLAN.md) を参照してください。

DeepSeekV4SSD は DeepSeek と提携していません。モデルをダウンロードして使用する前に、
モデルの利用条件を確認してください。

## ライセンス

DeepSeekV4SSD のソースコードは [MIT License](LICENSE) で公開されています。
モデルウェイトは含まれておらず、独自の規約が適用されます。
