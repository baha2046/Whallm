-include Makefile.env
export

.PHONY: $(wildcard *)

MODEL ?= scratch/deepseek-v4-flash-0731.dsv4
HOST ?= 127.0.0.1
PORT ?= 11434
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
	swift run dsv4-app $(ARGS)

## build: build the Swift package
build:
	swift build $(ARGS)

## test: run all Swift tests
test:
	mkdir -p .build/debug/PackageFrameworks
	ln -sfn "$(SPARKLE_FRAMEWORK_PATH)/Sparkle.framework" .build/debug/PackageFrameworks/Sparkle.framework
	swift test $(ARGS)

## package: build a distributable macOS app with Python and runtime
package:
	./Scripts/package-app.sh

## release: sign, notarize, and publish a Sparkle update to GitHub Releases
release:
	./Scripts/release.sh

## server: start the OpenAI-compatible API server directly
server:
	PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.server \
		--model "$(MODEL)" --host "$(HOST)" --port "$(PORT)" $(ARGS)

## benchmark-dsv4: benchmark an API model with 1K, 4K, 16K, and 32K inputs
benchmark-dsv4:
	.venv/bin/python Scripts/benchmark_api.py $(ARGS) --model deepseek-v4-flash-0731 --runs 3

## benchmark-qwen: benchmark an API model with 1K, 4K, 16K, and 32K inputs
benchmark-qwen:
	.venv/bin/python Scripts/benchmark_api.py $(ARGS) --model Qwen/Qwen3.8-Flash-Next-FP8 --runs 3

%:
	@:
