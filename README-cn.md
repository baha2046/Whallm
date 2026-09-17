# Whallm

<p align="center">
  <a href="."><img height="160" src="Packaging/AppIcon.png" alt="Whallm"></a>
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

Whallm 让 Apple Silicon Mac 从 SSD 读取所需的专家权重，在本地运行大语言模型。支持 DeepSeek V4、DeepSeek V4.1 和 Qwen3.8，提供内置聊天和 OpenAI 兼容 API。

## 性能摘要

| 模型 | 芯片 | Prefill | Decode | 峰值内存 | 专家缓存 Slots |
| --- | --- | ---: | ---: | ---: | ---: |
| `DeepSeek-V4-Flash-0731` | M5 Pro | 53.6–201.0 tok/s | 5.9–7.7 tok/s | 23 GiB | 1152 |
| `Qwen3.8-Flash-Next-FP8` | M5 Pro | 99.1–153.7 tok/s | 8.5–10.6 tok/s | 18 GiB | 3072 |
| `DeepSeek-V4.1-Flash` | M2 Max | 13.9–65.9 tok/s | 1.8–2.2 tok/s | 33 GiB | 1152 |

> 使用 v1.1.7 内置 Throughput 性能测试，输入长度为 1,024 至 16,384 tokens。
>
> 详细数据见[完整性能测试](#性能测试)。

## 快速开始

1. 从 [GitHub Releases](https://github.com/yanun0323/Whallm/releases) 下载 `Whallm-macOS-arm64.zip`，解压后打开 `Whallm.app`。
2. 打开 **Model**，选择模型并点击 **Download Model**。App 会检查存储空间，中断的下载可以继续。
3. 打开 **Server**，点击 **Start Server**。
4. 到 **Chat** 选择模型开始对话，或按下方示例连接 API 客户端。

默认地址为 `http://127.0.0.1:11434`。模型在首次使用时加载；服务器一次保留一个模型，依次处理生成请求。如果聊天列表中没有刚安装的模型，请重启服务器。

公开下载版的功能可能少于本文所述源码。

## 使用要求

| 项目 | 要求 |
| --- | --- |
| Mac | Apple Silicon，macOS 15 或更新版本 |
| 统一内存 | 建议 64 GiB；实际用量取决于模型和设置 |
| 存储设备 | 高速内置、Thunderbolt 或 USB4 SSD |
| 可用空间 | App 会根据模型和现有的部分下载计算需求 |
| 网络 | 下载模型和 App 更新时需要 |

App 不包含模型权重。DeepSeek V4.1 的专家和 Engram 文件就需要约 **458 GiB**，还需常驻权重和元数据的空间。

## 模型与默认值

| 模型 | API model ID | 专家缓存 Slots | 可选候选模型 |
| --- | --- | ---: | --- |
| DeepSeek-V4-Flash-0731 | `deepseek-v4-flash-0731` | 1152 | DSpark |
| DeepSeek-V4.1-Flash | `deepseek-v4.1-flash` | 1152 | DSpark |
| Qwen3.8-Flash-Next-FP8 | `qwen3.8-flash-next-fp8` | 3072 | MTP |

一个 Slot 可以保存一个专家的权重。增加 Slots 会占用更多内存，也可能减少 SSD 读取。已保存的自定义设置会保留。

目前源代码已将主模型、MTP 与 DSpark 的 slots 设置改为“专家缓存 GiB”，按已安装模型的专家大小向下换算，保留旧容量。高级设置的固定标题栏并排显示 **64K 与 128K 峰值内存估算**，上方小字为长度，下方为 GiB 数值。公式保留完整专家容量与对话缓存预留，比较加载、输入处理、生成三个阶段并取最高值。V4.1 的输入处理已改变临时数据的存活方式，目前采用结构估算，不再套用之前的校准结果。

固定估算标题栏、精简说明与 Playground／About／Status 导航调整已包含在通过验证的本地打包产物中，尚未公开发布。
128K 数值是外推估算，不是实测峰值或保证上限。单次请求可能未填满专家缓存，因此实际用量可低于容量规划值。

在 **Model → Advanced Settings** 中：

- 三个模型的 **Max tokens** 均默认为 **8192**。
- **Prompt cache** 默认为 **Memory**。**Disk** 可在重启后恢复缓存；**Off** 每次重新处理输入。
- **Use approximate mode** 默认**关闭**。开启后少用一个选中的专家，可能降低输出质量。
- **DSpark / MTP** 默认**关闭**，需要额外权重。它们先提出候选 token，再由主模型确认，不能与近似模式同时使用。
- **Alias** 可设置 API 请求使用的别名。其他设置在下次加载模型时生效；模型加载期间不能编辑。

新下载的 DeepSeek V4 包含 DSpark 权重。V4.1 DSpark 和 Qwen MTP 可以单独安装。Qwen 下载的是已转换好的 MXFP4 模型，不会在安装时于你的 Mac 上量化。App 使用 DSpark／MTP 生成时，不复用跨请求的 Prompt cache。

## 功能

Whallm 将共用权重保留在内存，从 SSD 读取选中的专家。DeepSeek V4.1 的 Engram 和 Qwen 的 N-gram 数据也按需读取。

| 模型 | 加速选项 |
| --- | --- |
| DeepSeek V4 | 按层处理输入、专家批量计算、FP8 KV 缓存、可选 ANE 投影运算和 DSpark |
| DeepSeek V4.1 | 按层处理输入、专家批量计算、压缩 KV／索引缓存、只计算候选索引、CED 输入处理、ANE 投影运算和 DSpark |
| Qwen3.8 | 输入阶段的专家分组、专家读取完成后立即计算、QSA 缓存压缩、下一层预读、ANE 投影运算和 MTP |

目前源代码中，V4.1 App 新设置或恢复默认值时，会开启按层处理输入、专家批量计算与下一层预读。Attention 保留原分批顺序；专家运算默认每批最多 4096 tokens，使用连续排列的融合权重，最多复用两层缓冲区。DSpark 可使用这条路径，但仍不能搭配 CED。已保存的设置保留。Python、CLI 与独立 server 现在也默认开启 V4.1 按层处理输入及下一层预读；可用 `--no-v41-layer-major-prefill` 或 `--no-v41-next-layer-prefetch` 关闭。这两个 V4.1 选项不会开启 V4 或 Qwen 的预读。

Qwen App 默认开启按层处理输入与专家加载后立即计算，专家批量计算及下一层预读仍关闭。V4 的这些选项仍默认关闭。缓存压缩、候选索引、CED 与 DeepSeek ANE 仍默认关闭。CED 会跳过后续层不再需要的旧 token。Qwen 缓存压缩与 ANE 可能改变运算结果。UI 会禁用不兼容的组合；这些选项不保证一定加速。

合并后的 V4.1 修改尚未重新测量完整模型，也未重新打包 App。下方性能表是修改前的记录；[BENCHMARK.md](BENCHMARK.md) 另保留 PR 作者的合并前测量，并非合并后 runtime 的验证结果。

## 性能测试

以下 v1.1.7 记录使用 **Code** 素材，输出上限为 **128 tokens**。表格未附各次程序版本和缓存状态，因此仅供参考，不能作为新增加速选项的控制变量比较。

TTFT 是等待首个 token 的时间。Prefill 为输入处理速度，Decode 为输出生成速度，单位均为 tokens／秒。Peak MLX 是以 **GiB** 表示的 MLX 分配量，不是整台 Mac 的内存用量。App 导出虽标为 GB，实际以 bytes 除以 1024³ 计算。

### M5 Pro

| 模型 | Slots | 输入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |
| Qwen3.8 | 3072 | 1024 | 10336.3 | 99.1 | 10.6 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 28831.0 | 142.1 | 9.2 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 53303.2 | 153.7 | 10.1 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 112353.1 | 145.8 | 8.5 | 17.18 |

### M2 Max

| 模型 | Slots | 输入 tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

要测试自己的 Mac，打开 **Throughput**，选择已安装模型、**Code** 或 **Novel** 素材、**1K–200K** 输入长度，以及 **128、1024 或 4096** 的输出上限。结果可以复制为纯文本、JSON 或 Markdown。整轮完成或取消后，App 会卸载测试模型。本地打包版另有 **Dry run**，只生成模拟结果。

SSD 速度、输入长度、缓存状态和设置都会影响结果。

## 连接 Codex

先启动 Whallm 服务器，再将以下内容加入用户级 `~/.codex/config.toml`：

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

将 `model` 设为上表中的 API model ID 或 Alias，保存后重启 Codex。此示例使用默认本地地址，且未设置 API key。如果在 Whallm 设置了密钥，客户端也要设置相同密钥。详见 [Codex 配置参考](https://developers.openai.com/codex/config-reference/)。

## API 与隐私

API 支持文本流式输出和工具调用，提供以下端点：

- `GET /healthz` 和 `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions` 和 `POST /v1/completions`
- `POST /api/models/load` 和 `POST /api/models/unload`

当前源码的三个生成端点支持可选 `seed`，接受 `0` 到 `4294967295` 的整数；省略或传入 `null` 时，每次请求使用新的随机值。Playground Chat 的 Seed 仅应用于下一条消息，发送后清空。固定 seed 有助于在相同输入、模型、设置与运行环境下复现结果，但不保证跨版本、缓存状态或加速设置仍逐字一致。`temperature=0` 仍选择概率最高的结果。此变更尚未打包或发布。

工具由客户端执行，再返回结果。不支持图片、音频、`logprobs`、`response_format` 和 `stop`。请求体上限为 **1 MiB**。

推理在你的 Mac 上运行。下载、更新和 API 连接会使用网络。连接的客户端可能将数据发送到其他服务；**Debug** 日志可能包含完整输入和工具结果。

## 验证与限制

v1.1.7 本地打包版通过 **424 项 Python 测试**、**91 项 Swift 测试**和 **12 项包内运行测试**。App 和 ZIP 解压副本均通过签名，以及英文、简体中文、繁体中文的隔离启动检查。

目前仅支持这三个固定版本的文本模型。新增加速路径已通过小模型和组件测试；完整模型的速度与质量比较仍待验证。很长的输入需要更多缓存内存。

## 许可证

Whallm 采用 [MIT License](LICENSE)。模型权重另有使用条款。Whallm 与 DeepSeek、Qwen 无隶属关系。
