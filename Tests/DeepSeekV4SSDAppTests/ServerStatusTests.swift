import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ServerStatusTests: XCTestCase {
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
}
