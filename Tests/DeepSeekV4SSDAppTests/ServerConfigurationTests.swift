import XCTest

@testable import DeepSeekV4SSDApp

final class ServerConfigurationTests: XCTestCase {
  func testConfigurationBuildsServerArgumentsWithoutExposingAPIKey() {
    var configuration = ServerConfiguration.localDefault
    configuration.host = "0.0.0.0"
    configuration.port = 9000
    configuration.apiKey = "secret"
    configuration.bf16KVCache = true

    XCTAssertEqual(configuration.baseURL?.absoluteString, "http://127.0.0.1:9000")
    XCTAssertTrue(configuration.arguments.contains("--bf16-kv-cache"))
    XCTAssertTrue(configuration.arguments.contains("9000"))
    XCTAssertFalse(configuration.arguments.contains("secret"))
  }
}
