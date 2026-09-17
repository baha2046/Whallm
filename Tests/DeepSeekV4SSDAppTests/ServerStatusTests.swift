import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ServerStatusTests: XCTestCase {
  func testMemoryMaximumUsesServerPeakNotStatusPollSamples() throws {
    let memory = ServerStatus.AppMemory(currentAppMemoryBytes: 100, peakAppMemoryBytes: 900,
      memoryScope: "app", sampleIntervalSeconds: 0.01, activeRequests: 1, epoch: "window")
    var live = LivePerformance(snapshot: PerformanceSnapshot(memoryUsage: 100), appMemory: memory)
    var history = PerformanceHistory()
    history.record(live.snapshot)
    XCTAssertEqual(history[.memoryUsage]?.maximum, 100)
    XCTAssertEqual(live.memoryMaximum, 900)
    live.memoryResetFailed = true
    XCTAssertNil(live.memoryMaximum) // Never restore the old peak after a failed Clear.
    live.memoryResetFailed = false
    live.appMemory = nil
    XCTAssertNil(live.memoryMaximum) // No RSS/MLX/local-history fallback for old servers.
  }

  func testStatusMemoryDecodingKeepsScopeRateAndUnavailableValues() throws {
    let base = try statusFixture(model: nil, sourceModel: nil, modelPath: nil, runtime: nil,
                             loadedModel: nil, loadingModel: nil)
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: base) as? [String: Any])
    XCTAssertNil(try ServerStatus.decode(base).appMemory)
    json["app_memory"] = ["current_app_memory_bytes": 1234, "peak_app_memory_bytes": 5678,
      "memory_scope": "app", "sample_interval_seconds": 0.01, "active_requests": 1, "epoch": "a"]
    let active = try XCTUnwrap(ServerStatus.decode(JSONSerialization.data(withJSONObject: json)).appMemory)
    XCTAssertEqual(active.currentAppMemoryBytes, 1234)
    XCTAssertEqual(active.peakAppMemoryBytes, 5678)
    XCTAssertEqual(active.sampleIntervalSeconds, 0.01)
    XCTAssertEqual(active.memoryScope, "app")
    json["app_memory"] = ["current_app_memory_bytes": NSNull(), "peak_app_memory_bytes": NSNull(),
      "memory_scope": "process", "sample_interval_seconds": 1, "active_requests": 0, "epoch": "b"]
    let idle = try XCTUnwrap(ServerStatus.decode(JSONSerialization.data(withJSONObject: json)).appMemory)
    XCTAssertNil(idle.currentAppMemoryBytes)
    XCTAssertNil(idle.peakAppMemoryBytes)
    XCTAssertEqual(idle.sampleIntervalSeconds, 1)
    XCTAssertEqual(idle.activeRequests, 0)
    XCTAssertNotEqual(idle.epoch, active.epoch)
  }

  func testStatusMemoryHelpIsLocalized() {
    let keys = ["Memory uses App + inference process physical footprint (inference process only for standalone servers): every 10 ms during model work, every second while idle. Maximum is retained since server start or clearing metric history, including loading and idle time. Memory P95 uses the slower Status refresh samples. Brief peaks may be missed; — means unavailable.",
                "Unable to reset the memory maximum. Try clearing metric history again."]
    for language in [AppLanguage.traditionalChinese, .simplifiedChinese] {
      for key in keys { XCTAssertNotEqual(L10n.string(key, language: language), key) }
    }
  }

  func testStatusDecodesPerformanceMetrics() throws {
    let data = Data(
      #"{"performance":{"generating":true,"runtime_prompt_tokens":120,"runtime_generation_tokens":12,"accumulated_generation_tokens":42,"completed_request_count":2,"request_seconds":3.5,"time_to_first_token_seconds":0.5,"prefill_tokens_per_second":200,"decode_tokens_per_second":4,"request_ssd_read_bytes_per_second":4194304,"request_expert_cache_hit_rate":0.8,"ssd_bytes_read":1048576,"active_parameters_cache":{"hit_rate":0.75,"hits":3,"misses":1,"resident_slots":3,"capacity_slots":1024}}}"#
        .utf8
    )

    let status = try ServerStatus.decode(data)

    XCTAssertTrue(status.performance.generating)
    XCTAssertEqual(status.performance.runtimePromptTokens, 120)
    XCTAssertEqual(status.performance.runtimeGenerationTokens, 12)
    XCTAssertEqual(status.performance.accumulatedGenerationTokens, 42)
    XCTAssertEqual(status.performance.completedRequestCount, 2)
    XCTAssertEqual(status.performance.prefillTokensPerSecond, 200)
    XCTAssertEqual(status.performance.decodeTokensPerSecond, 4)
    XCTAssertEqual(status.performance.requestSsdReadBytesPerSecond, 4_194_304)
    XCTAssertEqual(status.performance.ssdBytesRead, 1_048_576)
    XCTAssertEqual(status.performance.activeParametersCache.hitRate, 0.75)
    XCTAssertEqual(status.performance.activeParametersCache.capacitySlots, 1_024)
  }

  func testStatusDecodesUnloadedLoadingAndLoadedModels() throws {
    let unloaded = try ServerStatus.decode(statusFixture(
      model: nil,
      sourceModel: nil,
      modelPath: nil,
      runtime: nil,
      loadedModel: nil,
      loadingModel: nil
    ))
    XCTAssertNil(unloaded.model)
    XCTAssertNil(unloaded.runtime)
    XCTAssertNil(unloaded.loadedModel)
    XCTAssertNil(unloaded.loadingModel)

    let loading = try ServerStatus.decode(statusFixture(
      model: nil,
      sourceModel: nil,
      modelPath: nil,
      runtime: nil,
      loadedModel: nil,
      loadingModel: "qwen3.8-flash-next-fp8"
    ))
    XCTAssertNil(loading.loadedModel)
    XCTAssertEqual(loading.loadingModel, "qwen3.8-flash-next-fp8")

    let loaded = try ServerStatus.decode(statusFixture(
      model: "work-model",
      sourceModel: "deepseek-ai/DeepSeek-V4-Flash-0731",
      modelPath: "/tmp/model.dsv4",
      runtime: [:],
      loadedModel: "deepseek-v4-flash-0731",
      loadingModel: nil
    ))
    XCTAssertEqual(loaded.model, "work-model")
    XCTAssertEqual(loaded.sourceModel, "deepseek-ai/DeepSeek-V4-Flash-0731")
    XCTAssertEqual(loaded.modelPath, "/tmp/model.dsv4")
    XCTAssertNotNil(loaded.runtime)
    XCTAssertEqual(loaded.loadedModel, "deepseek-v4-flash-0731")
    XCTAssertNil(loaded.loadingModel)
  }

  func testPerformanceHistoryTracksAndClearsStatistics() {
    var history = PerformanceHistory()
    history.record(
      PerformanceSnapshot(
        prefillTokensPerSecond: 10,
        decodeTokensPerSecond: 2,
        inputTokens: 100,
        outputTokens: 20,
        memoryUsage: 1_000,
        ssdReadSpeed: 3_000,
        cacheHitRate: 0.5,
        firstTokenWaitTime: 4,
        completionTime: 8
      ))
    history.record(
      PerformanceSnapshot(
        prefillTokensPerSecond: 30,
        decodeTokensPerSecond: 6,
        inputTokens: 300,
        outputTokens: 60,
        memoryUsage: 2_000,
        ssdReadSpeed: 5_000,
        cacheHitRate: 0.9,
        firstTokenWaitTime: 2,
        completionTime: 12
      ))

    let prefill = history[.prefillTokensPerSecond]
    XCTAssertEqual(prefill?.minimum, 10)
    XCTAssertEqual(prefill?.average, 20)
    XCTAssertEqual(prefill?.maximum, 30)
    XCTAssertEqual(prefill?.p95, 30)
    XCTAssertEqual(history[.firstTokenWaitTime]?.minimum, 2)
    XCTAssertEqual(history[.firstTokenWaitTime]?.maximum, 4)

    history.clear()
    XCTAssertTrue(history.isEmpty)
  }

  func testMetricStatisticsUsesNearestRankP95() {
    var statistics = MetricStatistics()
    for value in 1...20 {
      statistics.record(Double(value))
    }

    XCTAssertEqual(statistics.p95, 19)
  }

  func testLiveFirstTokenWaitTimeUsesRequestElapsedUntilFirstToken() {
    let waiting = LivePerformance(
      generating: true,
      snapshot: PerformanceSnapshot(
        outputTokens: 0,
        firstTokenWaitTime: 0,
        completionTime: 12.4
      )
    )
    let generating = LivePerformance(
      generating: true,
      snapshot: PerformanceSnapshot(
        outputTokens: 1,
        firstTokenWaitTime: 12.8,
        completionTime: 13.2
      )
    )

    XCTAssertEqual(waiting.liveFirstTokenWaitTime, 12.4)
    XCTAssertEqual(generating.liveFirstTokenWaitTime, 12.8)
  }

  @MainActor
  func testPerformanceHistoryRecordsFinalFirstTokenWaitTimeOnly() {
    let controller = ServerController()

    controller.recordPerformanceSample(
      LivePerformance(
        generating: true,
        snapshot: PerformanceSnapshot(
          outputTokens: 0,
          firstTokenWaitTime: 0,
          completionTime: 1
        )
      ))
    controller.recordPerformanceSample(
      LivePerformance(
        generating: true,
        snapshot: PerformanceSnapshot(
          outputTokens: 1,
          firstTokenWaitTime: 2,
          completionTime: 3
        )
      ))

    XCTAssertNil(controller.performanceHistory[.firstTokenWaitTime])

    controller.recordPerformanceSample(
      LivePerformance(
        completedRequestCount: 1,
        snapshot: PerformanceSnapshot(firstTokenWaitTime: 2, completionTime: 4)
      ))

    let firstTokenWait = controller.performanceHistory[.firstTokenWaitTime]
    XCTAssertEqual(firstTokenWait?.count, 1)
    XCTAssertEqual(firstTokenWait?.minimum, 2)
    XCTAssertEqual(firstTokenWait?.maximum, 2)
  }

  @MainActor
  func testPerformanceHistoryRecordsEachGeneratingSecondAcrossRequests() {
    let controller = ServerController()

    controller.recordPerformanceSample(
      LivePerformance(
        generating: true,
        snapshot: PerformanceSnapshot(prefillTokensPerSecond: 10)
      ))
    controller.recordPerformanceSample(
      LivePerformance(
        generating: true,
        snapshot: PerformanceSnapshot(prefillTokensPerSecond: 30)
      ))
    controller.recordPerformanceSample(
      LivePerformance(
        completedRequestCount: 1,
        snapshot: PerformanceSnapshot(prefillTokensPerSecond: 20)
      ))
    controller.recordPerformanceSample(
      LivePerformance(
        completedRequestCount: 1,
        snapshot: PerformanceSnapshot(prefillTokensPerSecond: 0)
      ))
    controller.recordPerformanceSample(
      LivePerformance(
        generating: true,
        completedRequestCount: 1,
        snapshot: PerformanceSnapshot(prefillTokensPerSecond: 40)
      ))

    let prefill = controller.performanceHistory[.prefillTokensPerSecond]
    XCTAssertEqual(prefill?.count, 4)
    XCTAssertEqual(prefill?.minimum, 10)
    XCTAssertEqual(prefill?.average, 25)
    XCTAssertEqual(prefill?.maximum, 40)
  }

  @MainActor
  func testChangingTheLoadedModelClearsMetricHistory() {
    let controller = ServerController()
    controller.updateLoadedModel("deepseek-v4-flash-0731", completedRequestCount: 0)
    controller.recordPerformanceSample(
      LivePerformance(
        generating: true,
        snapshot: PerformanceSnapshot(prefillTokensPerSecond: 10)
      ))
    XCTAssertFalse(controller.performanceHistory.isEmpty)

    controller.updateLoadedModel("qwen3.8-flash-next-fp8", completedRequestCount: 1)

    XCTAssertTrue(controller.performanceHistory.isEmpty)
  }
}

private func statusFixture(
  model: String?,
  sourceModel: String?,
  modelPath: String?,
  runtime: [String: Any]?,
  loadedModel: String?,
  loadingModel: String?
) throws -> Data {
  func jsonValue(_ value: Any?) -> Any { value ?? NSNull() }
  return try JSONSerialization.data(withJSONObject: [
    "model": jsonValue(model),
    "source_model": jsonValue(sourceModel),
    "model_path": jsonValue(modelPath),
    "runtime": jsonValue(runtime),
    "loaded_model": jsonValue(loadedModel),
    "loading_model": jsonValue(loadingModel),
    "performance": [
      "generating": false,
      "runtime_prompt_tokens": 0,
      "runtime_generation_tokens": 0,
      "accumulated_generation_tokens": 0,
      "completed_request_count": 0,
      "request_seconds": 0,
      "time_to_first_token_seconds": 0,
      "prefill_tokens_per_second": 0,
      "decode_tokens_per_second": 0,
      "request_ssd_read_bytes_per_second": 0,
      "request_expert_cache_hit_rate": 0,
      "ssd_bytes_read": 0,
      "active_parameters_cache": [
        "hit_rate": 0,
        "hits": 0,
        "misses": 0,
        "resident_slots": 0,
        "capacity_slots": 0,
      ],
    ],
  ])
}
