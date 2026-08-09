import XCTest

@testable import DeepSeekV4SSDApp

final class ServerConfigurationTests: XCTestCase {
  func testLocalDefaultsUseRequestedGenerationValues() {
    let configuration = ServerConfiguration.localDefault

    XCTAssertEqual(configuration.port, 11_434)
    XCTAssertEqual(configuration.slots, 1_152)
    XCTAssertEqual(configuration.readWorkers, 4)
    XCTAssertEqual(configuration.defaultMaxTokens, 272_000)
    XCTAssertEqual(configuration.defaultTemperature, 0.2)
    XCTAssertEqual(configuration.defaultTopP, 0.98)
    XCTAssertEqual(configuration.dsparkSlots, 768)
    XCTAssertTrue(configuration.arguments.contains("272000"))
    XCTAssertTrue(configuration.arguments.contains("0.2"))
    XCTAssertTrue(configuration.arguments.contains("0.98"))
    XCTAssertTrue(configuration.arguments.contains("11434"))
    XCTAssertTrue(configuration.arguments.contains("--dspark-slots"))
  }

  func testLocalizationSupportsAllSelectableLanguages() {
    XCTAssertEqual(AppLanguage.appDefault, .english)
    XCTAssertEqual(L10n.string("Stopped", language: .english), "Stopped")
    XCTAssertEqual(L10n.string("Language", language: .simplifiedChinese), "语言")
    XCTAssertEqual(L10n.string("Language", language: .traditionalChinese), "語言")
    XCTAssertEqual(L10n.string("Live", language: .simplifiedChinese), "实时")
    XCTAssertEqual(L10n.string("Live", language: .traditionalChinese), "即時")
    XCTAssertEqual(
      L10n.string(
        "A higher value increases output variation.", language: .traditionalChinese),
      "較高的值會增加輸出變化。")
    XCTAssertEqual(L10n.string("Memory usage", language: .simplifiedChinese), "内存用量")
  }

  func testConfigurationBuildsServerArgumentsWithoutExposingAPIKey() {
    var configuration = ServerConfiguration.localDefault
    configuration.host = "0.0.0.0"
    configuration.port = 9000
    configuration.apiKey = "secret"
    configuration.bf16KVCache = true

    XCTAssertEqual(configuration.baseURL?.absoluteString, "http://127.0.0.1:9000")
    XCTAssertTrue(configuration.arguments.contains("--bf16-kv-cache"))
    XCTAssertTrue(configuration.arguments.contains("--prompt-cache-entries"))
    XCTAssertTrue(configuration.arguments.contains("--prompt-cache-memory-gib"))
    XCTAssertFalse(configuration.arguments.contains("--no-layer-major-prefill"))
    XCTAssertTrue(configuration.arguments.contains("9000"))
    XCTAssertFalse(configuration.arguments.contains("secret"))
  }
}
