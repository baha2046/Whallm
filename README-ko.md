# DeepSeekV4SSD

<p align="center">
  <img src="Packaging/AppIcon.png" alt="DeepSeekV4SSD 앱 아이콘" width="160">
</p>

[![English](https://img.shields.io/badge/English-Click-yellow)](README.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-點擊查看-orange)](README-tw.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](README-cn.md)
[![日本語](https://img.shields.io/badge/日本語-クリック-青)](README-ja.md)
[![한국어](https://img.shields.io/badge/한국어-클릭-yellow)](README-ko.md)

DeepSeekV4SSD는 [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare)에서 영감을 받아
SSD에서 routed expert를 스트리밍하고, M 시리즈 Mac에서 약 30 GB 메모리로
`DeepSeek-V4-Flash-0731`의 전체 2,840억 파라미터를 실행합니다.

## 벤치마크

측정 장치는 Apple M5 Pro, 18코어 CPU, 20코어 GPU, 64 GiB 통합 메모리를 탑재한
MacBook Pro입니다. DSpark는 비활성화했습니다.

| 테스트 | Prefill | Decode | 최대 메모리 |
| --- | ---: | ---: | ---: |
| input token 14,000개의 Codex request | 180 Tok/s | 6.5 Tok/s | 30 GB |
| 4,096-token prompt에서 token 1개 생성 | 144.53 Tok/s | — | 15.56 GiB |
| 짧은 prompt, 같은 runtime의 두 번째 실행 | — | 6.41 Tok/s | 15.05 GiB |

첫 두 행은 각각 `v1.0.3` 및 `v1.0.2` runtime으로 측정했습니다. 성능은 prompt,
SSD 속도, cache 상태에 따라 달라집니다. 자세한 내용은
[검증 기록](docs/VALIDATION.md)을 확인하세요.

## 사용 방법

**앱 다운로드 → 앱 열기 → 167 GB 전체 파라미터 모델 다운로드 → server 시작 →
앱에서 대화하거나 Codex 연결**

1. [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest)에서 최신
   `DeepSeekV4SSD-macOS-arm64.zip`을 다운로드합니다.
2. ZIP 파일의 압축을 풀고 `DeepSeekV4SSD.app`을 엽니다.
3. **Download Model**을 선택합니다. 기본 설치에는 DSpark가 포함되며 약 167 GB를
   사용합니다. 다운로드를 중지한 뒤 나중에 다시 시작할 수 있습니다.
4. 모델 준비가 끝나면 **Start Server**를 선택합니다.
5. 앱에서 대화하거나 아래 설정으로 Codex를 연결합니다.

기본 로컬 server 주소는 `http://127.0.0.1:11434`입니다.

![DeepSeekV4SSD 앱](docs/assets/deepseekv4ssd-app.png)

## 요구 사항

| 항목 | 요구 사항 |
| --- | --- |
| Mac | Apple Silicon M 시리즈 Mac |
| macOS | macOS 15 이상 |
| 통합 메모리 | 64 GiB 이상 |
| 여유 저장 공간 | 약 172 GB(160 GiB) |
| 모델 저장 장치 | 고속 내장, Thunderbolt 또는 USB4 SSD |
| 인터넷 | 모델 및 앱 업데이트 다운로드에 필요 |

> [!IMPORTANT]
> DeepSeekV4SSD는 실험적인 소프트웨어입니다. 앱에는 모델 가중치가 포함되지 않습니다.
> 다른 장치에서 연결해야 하는 경우가 아니면 기본 로컬 주소를 사용하세요.

## Codex `config.toml` 설정

DeepSeekV4SSD에서 server를 시작합니다. 다음 설정을 `~/.codex/config.toml`에 추가합니다.

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

### 모델 저장 공간과 DSpark

- main model은 약 145 GiB를 사용합니다.
- DSpark는 약 10.12 GiB를 추가합니다. 기본 다운로드에 DSpark가 포함됩니다.
- DSpark를 설치해도 자동으로 활성화되지 않습니다. speculative decoding을 테스트하려면
  runtime 설정에서 **Use DSpark**를 활성화하세요.
- main model을 다시 설치하지 않고 DSpark를 제거할 수 있습니다.

### OpenAI 호환 server

server는 다음 endpoint를 지원합니다.

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Responses API는 Codex tool과 OpenAI function tool을 지원합니다. API client가 tool을 실행하고
결과를 server에 보내야 합니다. 자세한 필드, 예제, 제한 사항은 [API 가이드](docs/API.md)를 확인하세요.

### 지표와 개인정보 보호

앱은 prefill 속도, decode 속도, token 수, 메모리 사용량, SSD 읽기 속도, cache hit rate,
첫 token 대기 시간, 완료 시간을 표시합니다.

추론은 Mac에서 실행됩니다. prompt와 생성된 텍스트는 로컬 runtime에 남습니다. 연결된
API client가 데이터를 다른 위치로 전송할 수는 있습니다.

### 현재 제한 사항

- runtime은 고정된 `DeepSeek-V4-Flash-0731` checkpoint만 지원합니다.
- runtime은 한 번에 하나의 생성 request만 처리합니다.
- 이미지, 오디오, logprobs, `response_format`, `stop`은 지원하지 않습니다.
- 매우 긴 input과 output에는 더 많은 KV cache 메모리가 필요합니다.
- 성능은 SSD 속도, input 길이, cache 상태에 따라 달라집니다.

모델 계약, runtime 설계, 측정 결과에 따른 기술 판단은
[runtime 연구](docs/RUNTIME_RESEARCH_2026-08-07.md)와
[구현 계획](docs/IMPLEMENTATION_PLAN.md)을 확인하세요.

DeepSeekV4SSD는 DeepSeek와 제휴하지 않습니다. 모델을 다운로드하고 사용하기 전에 모델 약관을
확인하세요.

## 라이선스

DeepSeekV4SSD 소스 코드는 [MIT License](LICENSE)로 공개됩니다. 모델 가중치는 포함되지
않으며 별도 약관이 적용됩니다.
