# Whallm

<p align="center">
  <img src="Packaging/AppIcon.png" alt="Whallm 앱 아이콘" width="160">
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

> [!NOTE]
> Whallm의 이전 이름은 DeepSeekV4SSD입니다. 이름 변경 전 릴리스는 이전 앱 및 ZIP 이름을 사용합니다.

Whallm은 [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare)에서 영감을 받아
SSD에서 routed expert를 스트리밍합니다. M 시리즈 Mac에서
`DeepSeek-V4-Flash-0731`의 전체 2,840억 파라미터를 실행하며,
`Qwen3.8-Flash-Next-FP8` text checkpoint도 지원합니다.

## 최대 메모리 가이드

| 모델 | 측정된 최대 메모리 |
| --- | ---: |
| `DeepSeek-V4-Flash-0731` | 23.03–35.64 GiB |
| `Qwen3.8-Flash-Next-FP8` | 15.19–18.92 GiB |

v1.1.0에서는 input token 1,024개에서 16,384개인 chat prompt를 측정했습니다.
이 값은 측정값이며 최소 메모리 요구 사항이나 성능을 보장하지 않습니다. prompt 길이,
tool, cache 상태, runtime 설정에 따라 최대 메모리가 달라질 수 있습니다.
[전체 벤치마크](BENCHMARK.md)와 [검증 기록](docs/VALIDATION.md)을 확인하세요.

## 벤치마크

v1.1.0은 Apple M5 Pro, 64 GB 통합 메모리, 1 TB 저장 공간을 탑재한
MacBook Pro에서 측정했습니다. 두 모델 모두 `reasoning_effort: low`와
`thinking_mode: chat`을 사용했습니다. TTFT는 첫 token까지의 대기 시간입니다.

### DeepSeek V4 Flash 0731

| Input token | P95 총 시간 | P95 TTFT | P95 Prefill | P95 Decode | 최대 메모리 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 46.72 s | 37.26 s | 20.7 tok/s | 6.7 tok/s | 23.03 GiB |
| 2,048 | 27.08 s | 16.91 s | 108.1 tok/s | 6.6 tok/s | 33.19 GiB |
| 8,192 | 47.66 s | 37.96 s | 209.5 tok/s | 6.7 tok/s | 34.74 GiB |
| 16,384 | 86.01 s | 76.34 s | 212.3 tok/s | 6.7 tok/s | 35.64 GiB |

### Qwen3.8 Next Flash FP8

| Input token | P95 총 시간 | P95 TTFT | P95 Prefill | P95 Decode | 최대 메모리 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 24.45 s | 17.20 s | 59.8 tok/s | 9.3 tok/s | 15.19 GiB |
| 2,048 | 40.45 s | 32.63 s | 65.0 tok/s | 8.3 tok/s | 16.41 GiB |
| 8,192 | 142.39 s | 134.44 s | 61.7 tok/s | 8.3 tok/s | 17.90 GiB |
| 16,384 | 276.41 s | 267.88 s | 61.8 tok/s | 7.8 tok/s | 18.92 GiB |

성능은 prompt, SSD 속도, cache 상태에 따라 달라집니다.
[전체 벤치마크](BENCHMARK.md)와 [검증 기록](docs/VALIDATION.md)을 확인하세요.

## 사용 방법

**앱 다운로드 → 앱 열기 → 모델 선택 및 다운로드 → server 시작 →
앱에서 대화하거나 Codex 연결**

> [!IMPORTANT]
> 이전 Sparkle signing key를 더 이상 사용할 수 없으므로 버전 1.0.3에서
> 1.0.4로 자동 업데이트할 수 없습니다. 앱을 종료하고
> [1.0.4 release](https://github.com/yanun0323/Whallm/releases/tag/v1.0.4)에서
> `DeepSeekV4SSD-macOS-arm64.zip`을 다운로드한 후 기존 앱을 직접 교체하세요.
> 1.0.4를 설치하면 자동 업데이트를 다시 사용할 수 있습니다.

1. [GitHub Releases](https://github.com/yanun0323/Whallm/releases/latest)에서 최신
   `Whallm-macOS-arm64.zip`을 다운로드합니다.
2. ZIP 파일의 압축을 풀고 `Whallm.app`을 엽니다.
3. **Model** 페이지를 엽니다. DeepSeek 또는 Qwen을 선택한 다음
   **Download Model**을 선택합니다. 앱이 필요한 저장 공간을 확인합니다. Qwen은
   배포된 MXFP4 installed model을 다운로드합니다. 다운로드를 중지한 뒤 나중에
   다시 시작할 수 있습니다.
4. **Server** 페이지를 열고 **Start Server**를 선택합니다. server는 installed model이
   없어도 시작할 수 있지만, 생성하려면 installed model이 필요합니다.
5. chat을 열고 모델을 선택합니다. 아래 설정으로 Codex를 연결할 수도 있습니다.

기본 로컬 server 주소는 `http://127.0.0.1:11434`입니다.

![Whallm 앱](docs/assets/deepseekv4ssd-app.png)

## 요구 사항

| 항목 | 요구 사항 |
| --- | --- |
| Mac | Apple Silicon M 시리즈 Mac |
| macOS | macOS 15 이상 |
| 통합 메모리 | 64 GiB 이상 |
| 여유 저장 공간 | 앱이 선택한 모델과 기존 부분 데이터를 확인 |
| 모델 저장 장치 | 고속 내장, Thunderbolt 또는 USB4 SSD |
| 인터넷 | 모델 및 앱 업데이트 다운로드에 필요 |

> [!IMPORTANT]
> Whallm은 실험적인 소프트웨어입니다. 앱에는 모델 가중치가 포함되지 않습니다.
> 다른 장치에서 연결해야 하는 경우가 아니면 기본 로컬 주소를 사용하세요.

## Codex `config.toml` 설정

Whallm에서 server를 시작합니다. 다음 설정을 `~/.codex/config.toml`에 추가합니다.

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

파일을 저장한 뒤 Codex를 다시 시작하세요. 로컬 주소에는 API key가 필요하지 않습니다.
provider 설정은 사용자 수준 설정 파일에 넣어야 합니다. 다른 옵션은
[Codex 공식 설정 참조](https://developers.openai.com/codex/config-reference/)를 확인하세요.

## 기타 기술 세부 정보

### 작동 방식

- main model은 전체 2,840억 파라미터를 가지며 token당 약 130억 파라미터가 활성화됩니다.
- common tensor는 통합 메모리에 유지됩니다.
- routed expert는 checkpoint 기본 FP4 가중치를 사용합니다. runtime은 필요한 routed expert를
  SSD에서 읽습니다.
- runtime은 FP8 KV cache와 용량이 제한된 expert cache로 메모리 사용량을 제어합니다.
- installed model은 고정된 checkpoint revision에 대해 검증됩니다.
- server를 시작할 때 installed model 목록만 읽고 모델 가중치는 로드하지 않습니다.
  첫 generation request가 지정한 모델을 로드합니다.
- server는 로드된 모델 하나만 유지합니다. request가 다른 모델을 지정하면 이전
  runtime을 종료한 후 새 runtime을 로드합니다.
- **Model** 페이지에서 모델을 로드하거나 언로드할 수 있습니다. 로드된 모델은
  **Loaded** 섹션으로 이동합니다.
- DeepSeek layer-major prefill 임계값을 변경할 수 있습니다. 기본값은 cache에 없는
  prompt token 1,024개입니다.

### 모델 저장 공간과 DSpark

- main model은 약 145 GiB를 사용합니다.
- DSpark는 약 10.12 GiB를 추가합니다. 새 DeepSeek 다운로드에는 항상 DSpark가 포함됩니다.
- DSpark를 설치해도 자동으로 활성화되지 않습니다. speculative decoding을 테스트하려면
  runtime 설정에서 **Use DSpark**를 활성화하세요.
- main model을 다시 설치하지 않고 DSpark를 제거할 수 있습니다.
- Qwen installed weight 파일은 125,268,506,112 bytes를 사용합니다.
  Qwen은 DSpark를 지원하지 않습니다.
- Qwen은 검증된 MXFP4 installed model을 다운로드합니다. 모델을 설치할 때 사용자의
  Mac에서 Qwen checkpoint를 양자화하지 않습니다.

### OpenAI 호환 server

server는 다음 endpoint를 지원합니다.

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /api/models/load`
- `POST /api/models/unload`

고정 API model ID는 `deepseek-v4-flash-0731` 및
`qwen3.8-flash-next-fp8`입니다. 각 모델의 **Advanced Settings**에서 선택 사항인
Alias를 설정할 수 있습니다. 유효한 변경 사항은 자동으로 저장됩니다. generation request는
API model ID 또는 Alias를 받습니다. chat 모델 선택기에는 server를 시작할 때 사용할 수
있었던 installed model만 표시됩니다. server 실행 중에 다운로드가 끝나면 server를
다시 시작하세요.

Responses API는 Codex tool과 OpenAI function tool을 지원합니다. API client가 각 tool을
실행하고 결과를 server에 보내야 합니다. field, 예제, 현재 제한 사항은
[API 가이드](docs/API.md)를 확인하세요.

### 지표와 개인정보 보호

앱은 prefill 속도, decode 속도, token 수, 메모리 사용량, SSD 읽기 속도, cache hit rate,
첫 token 대기 시간, 완료 시간을 표시합니다. 로드된 모델이 바뀌면 앱이 metric 기록을
지웁니다.

추론은 Mac에서 실행됩니다. 연결된 client가 다른 위치로 보내지 않는 한 prompt와
생성된 텍스트는 로컬 runtime에 남습니다. 앱은 모델 다운로드, 업데이트 확인,
설정된 API request 수신에 네트워크를 사용합니다.

### 현재 제한 사항

- runtime은 현재 문서에 있는 고정된 checkpoint revision 두 개만 지원합니다.
- Qwen은 text만 지원합니다. Qwen vision, video, MTP, DSpark는 지원하지 않습니다.
- 기록된 M5 Pro 환경에서 Qwen full-model SHA-256, text, thinking, tool call,
  greedy 4K, prompt cache, packaged App 검증을 통과했습니다. 자세한 내용은
  [Qwen 지원 상태](docs/QWEN.md)를 확인하세요.
- server는 로드된 모델 하나만 유지하며 한 번에 하나의 generation request를
  처리합니다. 다른 generation request는 현재 request stream이 모두 끝날 때까지
  기다립니다.
- 이미지, 오디오, logprobs, `response_format`, `stop`은 지원하지 않습니다.
- request body의 최대 크기는 1 MiB입니다.
- 매우 긴 input과 output에는 더 많은 KV cache 메모리가 필요합니다.
- 성능은 SSD 속도, input 길이, cache 상태에 따라 달라집니다.

모델 계약, runtime 설계, 검증, 성능, 연구 결론은
[현재 문서](docs/README.md)를 확인하세요.

Whallm은 DeepSeek와 제휴하지 않습니다. 모델을 다운로드하고 사용하기 전에 모델 약관을
확인하세요.

## 라이선스

Whallm 소스 코드는 [MIT License](LICENSE)로 공개됩니다. 모델 가중치는 포함되지
않으며 별도 약관이 적용됩니다.
