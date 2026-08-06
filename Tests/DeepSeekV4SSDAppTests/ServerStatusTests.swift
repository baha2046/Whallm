import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ServerStatusTests: XCTestCase {
  func testStatusDecodesPerformanceMetrics() throws {
    let data = Data(
      #"{"performance":{"generating":true,"generation_tokens":12,"tokens_per_second":1.5,"ssd_bytes_read":1048576,"active_parameters_cache":{"hit_rate":0.75,"hits":3,"misses":1,"resident_slots":3,"capacity_slots":1024}}}"#
        .utf8
    )

    let status = try ServerStatus.decode(data)

    XCTAssertTrue(status.performance.generating)
    XCTAssertEqual(status.performance.generationTokens, 12)
    XCTAssertEqual(status.performance.tokensPerSecond, 1.5)
    XCTAssertEqual(status.performance.ssdBytesRead, 1_048_576)
    XCTAssertEqual(status.performance.activeParametersCache.hitRate, 0.75)
    XCTAssertEqual(status.performance.activeParametersCache.capacitySlots, 1_024)
  }
}
