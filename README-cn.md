# DeepSeekV4SSD

[![English](https://img.shields.io/badge/English-Click-yellow)](README.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-點擊查看-orange)](README-tw.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](README-cn.md)
[![日本語](https://img.shields.io/badge/日本語-クリック-青)](README-ja.md)
[![한국어](https://img.shields.io/badge/한국어-클릭-yellow)](README-ko.md)

DeepSeekV4SSD 让 M 系列 Mac 使用约 30 GB 内存运行
`DeepSeek-V4-Flash-0731` 的全部 284B 参数，并从 SSD 流式读取 routed expert
（灵感来自 [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare)）。

![DeepSeekV4SSD APP 截图](docs/assets/deepseekv4ssd-app.png)

## Benchmark

测试机器是 MacBook Pro。MacBook Pro 配备 Apple M5 Pro、18 核 CPU、20 核
GPU 和 64 GiB 统一内存。测试已禁用 DSpark。

| 测试 | Prefill | Decode | 峰值内存 |
| --- | ---: | ---: | ---: |
| Codex request，14,000 个 input token | 180 Tok/s | 6.5 Tok/s | 30 GB |
| 4,096-token prompt，生成 1 个 token | 144.53 Tok/s | — | 15.56 GiB |
| 短 prompt，在同一 runtime 中第二次运行 | — | 6.41 Tok/s | 15.05 GiB |

前两项测试分别使用 `v1.0.3` 和 `v1.0.2` runtime。prompt 内容、SSD 速度
和 cache 状态会改变性能。[完整验证记录](docs/VALIDATION.md)包含详细测试数据。

## 如何使用

**下载 APP → 打开 APP → 下载 167 GB 全参数模型 → 启动 server →
在 APP 中对话或连接 Codex**

1. 从 [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest)
   下载最新的 `DeepSeekV4SSD-macOS-arm64.zip`。
2. 解压 ZIP。打开 `DeepSeekV4SSD.app`。
3. 选择“下载模型”。默认安装包含 DSpark。installed model 约使用 167 GB。
   你可以停止下载，以后再继续下载。
4. 模型 ready 后，选择“启动 server”。
5. 使用 APP 的对话功能，或使用下方配置连接 Codex。

默认本机 server 地址是 `http://127.0.0.1:11434`。

## 使用要求

| 项目 | 要求 |
| --- | --- |
| Mac | Apple Silicon M 系列 Mac |
| macOS | macOS 15 或更高版本 |
| 统一内存 | 64 GiB 或更多 |
| 可用存储空间 | 约 172 GB（160 GiB） |
| 模型存储设备 | 高速内置、Thunderbolt 或 USB4 SSD |
| 网络 | 下载模型和 APP 更新时需要网络 |

> [!IMPORTANT]
> DeepSeekV4SSD 是实验性软件。APP 不包含模型权重。除非其他设备必须连接，
> 否则请保留默认本机 server 地址。

## Codex `config.toml` 配置

先在 DeepSeekV4SSD 中启动 server。然后将以下配置添加到
`~/.codex/config.toml`：

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

保存文件后，请重新启动 Codex。本机地址不需要 API key。provider 配置必须
放在用户级配置文件中。[Codex 官方配置参考](https://developers.openai.com/codex/config-reference/)
包含其他配置。

## 其他技术细节

### 运行方式

- main model 有 284B 总参数。每个 token 约会启用 13B 参数。
- runtime 会将 common tensor 保留在统一内存中。
- routed expert 使用 checkpoint 原生 FP4 权重。runtime 会在需要时从 SSD 读取
  routed expert。
- runtime 使用 FP8 KV cache 和有容量上限的 expert cache 控制内存用量。
- installed model 必须通过固定 checkpoint revision 的验证。

### 模型存储空间与 DSpark

- main model 约使用 145 GiB。
- DSpark 会增加约 10.12 GiB。默认下载会安装 DSpark。
- 安装 DSpark 不会启用 DSpark。你可以在 runtime 设置中启用“使用 DSpark”来
  测试 speculative decoding。
- 你可以移除 DSpark。移除 DSpark 不需要重新安装 main model。

### OpenAI 兼容 server

server 支持以下 endpoint：

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Responses API 支持 Codex tool 和 OpenAI function tool。API client 必须执行 tool，
然后将结果发回 server。[API 指南](docs/API.md)包含 request 字段、示例和当前限制。

### 指标与隐私

APP 会显示 prefill 速度、decode 速度、token 数量、内存用量、SSD 读取速度、
cache hit rate、第一个 token 等待时间和完成时间。

runtime 会在 Mac 上执行推理。prompt 和生成文本会保留在本机 runtime 中。
连接的 API client 仍可能将数据发送到其他位置。APP 会使用网络下载模型、
检查更新和接收已配置的 API request。

### 当前限制

- runtime 只支持固定 checkpoint revision 的 `DeepSeek-V4-Flash-0731`。
- runtime 一次只处理一个生成 request。
- API 不支持图片、音频、logprobs、`response_format` 和 `stop`。
- 很长的 input 和 output 需要更多 KV cache 内存。
- 性能取决于 SSD 速度、input 长度和 cache 状态。

[运行时研究](docs/RUNTIME_RESEARCH_2026-08-07.md)和
[实现计划](docs/IMPLEMENTATION_PLAN.md)包含模型合约、runtime 设计和已测量的工程决策。

DeepSeekV4SSD 与 DeepSeek 没有从属关系。下载和使用模型前，请先阅读模型条款。
