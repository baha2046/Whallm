# Whallm

<p align="center">
  <img src="Packaging/AppIcon.png" alt="Whallm APP Icon" width="160">
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

> [!NOTE]
> Whallm 的旧名称是 DeepSeekV4SSD。更名前发布的 release 仍使用旧 APP 和 ZIP 名称。

Whallm 让 M 系列 Mac 运行 `DeepSeek-V4-Flash-0731` 的全部 284B 参数。
Whallm 从 SSD 流式读取 routed expert。
Whallm 也支持 `Qwen3.8-Flash-Next-FP8` text checkpoint。
本项目的设计灵感来自 [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare)。

## 峰值内存参考

| 模型 | 测量到的峰值内存 |
| --- | ---: |
| `DeepSeek-V4-Flash-0731` | 32.84–35.70 GiB |
| `Qwen3.8-Flash-Next-FP8` | 20.92–22.75 GiB |

v1.1.4 测试使用 1,024 到 16,384 个 input token 的对话 prompt。
这些结果是测量值，不是最低内存要求或性能保证。prompt 长度、tool、cache
状态和 runtime 设置会改变峰值内存。请参阅[完整 Benchmark](BENCHMARK.md)和
[完整验证记录](docs/VALIDATION.md)。

## Benchmark

v1.1.4 测试在配备 Apple M5 Pro、64 GB 统一内存和 1 TB 存储空间的
MacBook Pro 上运行。测试使用 SPEED-Bench mixed prompts，每个 input size 运行三次，
output 上限为 64 tokens。三次测试采用 nearest-rank P95 时，P95 等于最大值。
TTFT 表示第一个 token 的等待时间。

### DeepSeek V4 Flash 0731

| Input token | P95 总时间 | P95 TTFT | P95 Prefill | P95 Decode | 峰值内存 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 27.10 s | 17.75 s | 59.9 tok/s | 7.8 tok/s | 32.84 GiB |
| 2,048 | 27.35 s | 17.60 s | 117.3 tok/s | 7.3 tok/s | 33.26 GiB |
| 8,192 | 50.12 s | 40.47 s | 206.3 tok/s | 7.4 tok/s | 34.50 GiB |
| 16,384 | 88.27 s | 78.61 s | 209.0 tok/s | 7.2 tok/s | 35.70 GiB |

### Qwen3.8 Next Flash FP8

| Input token | P95 总时间 | P95 TTFT | P95 Prefill | P95 Decode | 峰值内存 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 22.60 s | 15.65 s | 69.2 tok/s | 10.4 tok/s | 20.92 GiB |
| 2,048 | 31.42 s | 24.90 s | 87.5 tok/s | 9.8 tok/s | 21.26 GiB |
| 8,192 | 84.88 s | 77.74 s | 111.9 tok/s | 10.4 tok/s | 21.90 GiB |
| 16,384 | 157.89 s | 150.33 s | 113.3 tok/s | 9.7 tok/s | 22.75 GiB |

prompt、SSD 速度和 cache 状态会改变性能。请参阅
[完整 Benchmark](BENCHMARK.md)和[完整验证记录](docs/VALIDATION.md)。

## 如何使用

**下载 APP → 打开 APP → 选择并下载模型 → 启动 server →
在 APP 中对话或连接 Codex**

> [!IMPORTANT]
> 1.0.3 版无法通过自动更新安装 1.0.4 版，因为旧的 Sparkle signing key
> 已无法使用。请退出 APP，从
> [1.0.4 release](https://github.com/yanun0323/Whallm/releases/tag/v1.0.4)
> 下载 `DeepSeekV4SSD-macOS-arm64.zip`，然后手动替换现有 APP。
> 安装 1.0.4 版后，自动更新会恢复正常。

1. 从 [GitHub Releases](https://github.com/yanun0323/Whallm/releases/latest)
   下载最新的 `Whallm-macOS-arm64.zip`。
2. 解压 ZIP。打开 `Whallm.app`。
3. 打开“模型”页面。选择 DeepSeek 或 Qwen，然后选择“下载模型”。APP 会检查
   所需存储空间。Qwen 会下载已发布的 MXFP4 installed model。你可以停止下载，
   以后再继续下载。
4. 打开“Server”页面，然后选择“启动 server”。server 可以在没有 installed model
   时启动，但 generation request 需要 installed model。
5. 打开对话页面并选择模型。你也可以使用下方配置连接 Codex。

默认本机 server 地址是 `http://127.0.0.1:11434`。

![Whallm APP 截图](docs/assets/deepseekv4ssd-app.png)

## 使用要求

| 项目 | 要求 |
| --- | --- |
| Mac | Apple Silicon M 系列 Mac |
| macOS | macOS 15 或更高版本 |
| 统一内存 | 64 GiB 或更多 |
| 可用存储空间 | APP 会检查所选模型和现有的部分下载 |
| 模型存储设备 | 高速内置、Thunderbolt 或 USB4 SSD |
| 网络 | 下载模型和 APP 更新时需要网络 |

> [!IMPORTANT]
> Whallm 是实验性软件。APP 不包含模型权重。除非其他设备必须连接，
> 否则请保留默认本机 server 地址。

## Codex `config.toml` 配置

先在 Whallm 中启动 server。然后将以下配置添加到
`~/.codex/config.toml`：

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
- server 启动时只读取 installed model 清单，不会加载模型权重。第一个 generation
  request 会加载指定模型。
- server 一次只保留一个已加载模型。request 指定另一个模型时，server 会先关闭
  旧 runtime，然后加载新 runtime。
- 你可以在“模型”页面加载或卸载模型。已加载的模型会移到“已加载”区域。
- 你可以调整 DeepSeek layer-major Prefill 阈值。默认值是 1,024 个未缓存的
  prompt token。

### 模型存储空间与 DSpark

- main model 约使用 145 GiB。
- DSpark 会增加约 10.12 GiB。每次新的 DeepSeek 下载都会安装 DSpark。
- 安装 DSpark 不会启用 DSpark。你可以在 runtime 设置中启用“使用 DSpark”来
  测试 speculative decoding。
- 你可以移除 DSpark。移除 DSpark 不需要重新安装 main model。
- Qwen installed weight 文件使用 125,268,506,112 bytes。Qwen 不支持 DSpark。
- Qwen 会下载已验证的 MXFP4 installed model。模型安装程序不会在用户的 Mac
  上量化 Qwen checkpoint。

### OpenAI 兼容 server

server 支持以下 endpoint：

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /api/models/load`
- `POST /api/models/unload`

固定 API model ID 是 `deepseek-v4-flash-0731` 和
`qwen3.8-flash-next-fp8`。每个模型的“进阶设置”页面可设置可选 Alias。
有效更改会自动保存。generation request 接受 API model ID 或 Alias。
对话模型选择器只显示 server 启动时可用的 installed model。如果下载在 server
运行期间完成，请重新启动 server。

Responses API 支持 Codex tool 和 OpenAI function tool。API client 必须执行 tool，
然后将结果发回 server。[API 指南](docs/API.md)包含 request 字段、示例和当前限制。

### 指标与隐私

APP 会显示 prefill 速度、decode 速度、token 数量、内存用量、SSD 读取速度、
cache hit rate、第一个 token 等待时间和完成时间。加载的模型切换时，APP 会清除
指标历史。

runtime 会在 Mac 上执行推理。prompt 和生成文本会保留在本机 runtime 中。
连接的 API client 仍可能将数据发送到其他位置。APP 会使用网络下载模型、
检查更新和接收已配置的 API request。

### 当前限制

- runtime 只支持当前文档中两个固定的 checkpoint revision。
- Qwen 只支持文本。Qwen 不支持 vision、video、MTP 和 DSpark。
- Qwen 已在记录的 M5 Pro 环境中通过完整模型 SHA-256、文本、thinking、tool call、
  greedy 4K、prompt cache 和 packaged APP 验证。请参阅
  [Qwen 支持状态](docs/QWEN.md)。
- server 一次只保留一个已加载模型，并且一次只处理一个 generation request。
  其他 generation request 会等待当前 request stream 完全结束。
- API 不支持图片、音频、logprobs、`response_format` 和 `stop`。
- server 将 request body 限制为 1 MiB。
- 很长的 input 和 output 需要更多 KV cache 内存。
- 性能取决于 SSD 速度、input 长度和 cache 状态。

[当前文档](docs/README.md)包含模型合约、runtime 设计、验证、性能和研究结论。

Whallm 与 DeepSeek 没有从属关系。下载和使用模型前，请先阅读模型条款。

## 许可证

Whallm 源代码使用 [MIT 许可证](LICENSE)开放。项目不包含模型权重。
模型权重适用其自身条款。
