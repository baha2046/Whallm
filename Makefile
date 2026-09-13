-include Makefile.env
export

.PHONY: $(wildcard *)

MODEL ?= $(HOME)/.dsmodel/deepseek-v4-flash-0731.dsv4
HOST ?= 127.0.0.1
PORT ?= 11434
SPEED_BENCH_DIR ?= scratch/speed-bench
SPARKLE_FRAMEWORK_PATH := $(CURDIR)/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64

ARGS := $(word 2,$(MAKECMDGOALS))

## help: show help
help:
	@echo ""
	@echo "Usage:"
	@echo ""
	@sed -n 's/^## //p' Makefile | column -t -s ':' | sed -e 's/^/\t/'
	@echo ""

## run: start the SwiftUI macOS app
run:
	swift run -Xswiftc -DWHALLM_LOCAL_BUILD dsv4-app $(ARGS)

## build: build the Swift package
build:
	swift build -Xswiftc -DWHALLM_LOCAL_BUILD $(ARGS)

## test: run all Swift tests
test:
	mkdir -p .build/debug/PackageFrameworks
	ln -sfn "$(SPARKLE_FRAMEWORK_PATH)/Sparkle.framework" .build/debug/PackageFrameworks/Sparkle.framework
	swift test -Xswiftc -DWHALLM_LOCAL_BUILD $(ARGS)

## package: build a local macOS app with Python, runtime, and local debug features
package:
	WHALLM_BUILD_FLAVOR=local ./Scripts/package-app.sh

## ane-bridge: build the private Apple Neural Engine runtime bridge
ane-bridge:
	./Scripts/build-ane-bridge.sh

## release: publish a signed update (CHANNEL=stable or dev, DEV_BUILD=1)
release:
	./Scripts/release.sh

## server: start the OpenAI-compatible API server directly
server: ane-bridge
	PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.server \
		--model "$(MODEL)" --host "$(HOST)" --port "$(PORT)" $(ARGS)

## benchmark-dsv4: benchmark DeepSeek with SPEED-Bench mixed inputs
benchmark-dsv4:
	.venv/bin/python Scripts/benchmark_api.py --model deepseek-v4-flash-0731 \
		--speed-bench-dir "$(SPEED_BENCH_DIR)" --runs 3 $(ARGS)

## benchmark-qwen: benchmark Qwen with SPEED-Bench mixed inputs
benchmark-qwen:
	.venv/bin/python Scripts/benchmark_api.py --model qwen3.8-flash-next-fp8 \
		--speed-bench-dir "$(SPEED_BENCH_DIR)" --runs 3 $(ARGS)

%:
	@:
