import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ThroughputTests: XCTestCase {
  @MainActor
  func testDryRunUsesSelectedOptionsWithoutStartingServer() throws {
    let session = ThroughputSession()
    let server = ServerController()
    let originalState = server.state
    session.model = ThroughputSession.dryRunModel
    session.contextLengths = [8192, 1024]
    session.generationLength = 4096
    session.benchmarkContext = .novel
    session.start(configuration: .localDefault, server: server, catalog: ModelCatalog(models: []))
    XCTAssertEqual(server.state, originalState)
    XCTAssertFalse(session.isRunning)
    XCTAssertNil(session.error)
    #if !WHALLM_LOCAL_BUILD
    XCTAssertFalse(ThroughputSession.dryRunAvailable)
    session.runDryRun()
    XCTAssertTrue(session.results.isEmpty)
    return
    #else
    XCTAssertTrue(ThroughputSession.dryRunAvailable)
    XCTAssertEqual(session.results.map(\.contextTokens), [1024, 8192])
    XCTAssertEqual(session.results.map(\.generationTokens), [4096, 4096])
    XCTAssertTrue(session.results.allSatisfy { $0.benchmarkContext == .novel && $0.model == "dry-run" && $0.slots == 2304 })
    let firstExport = try ThroughputOutputFormat.json.render(session.results)
    session.runDryRun()
    XCTAssertEqual(try ThroughputOutputFormat.json.render(session.results), firstExport)
    for format in ThroughputOutputFormat.allCases {
      XCTAssertTrue(try format.render(session.results).contains("dry-run"))
    }
    session.contextLengths = []
    session.runDryRun()
    XCTAssertNotNil(session.error)
    XCTAssertEqual(server.state, originalState)
    #endif
  }

  func testBenchmarkLabelsLoadInAllThreeLanguages() {
    XCTAssertEqual(L10n.string("Run Benchmark", language: .english), "Run Benchmark")
    XCTAssertEqual(L10n.string("Run Benchmark", language: .traditionalChinese), "執行測試")
    XCTAssertEqual(L10n.string("Run Benchmark", language: .simplifiedChinese), "运行测试")
  }

  func testDecodesActualCountsAndOptionalTimePerToken() throws {
    let event = try XCTUnwrap(ThroughputEvent.decode(#"data: {"result":{"model":"test-model","benchmark_context":"novel","corpus_sha256":"def","context_tokens":4096,"generation_tokens":1,"generation_limit":128,"ttft_ms":500,"tpot_ms":null,"prefill_tps":8192,"decode_tps":0,"elapsed_seconds":0.6,"throughput_tps":6828.3,"peak_memory_bytes":1073741824,"output_token_sha256":"abc","prompt_cache_reused_tokens":0}}"#))
    let result = try XCTUnwrap(event.result)
    XCTAssertEqual(result.contextTokens, 4096)
    XCTAssertEqual(result.generationTokens, 1)
    XCTAssertEqual(result.generationLimit, 128)
    XCTAssertNil(result.tpotMs)
    XCTAssertEqual(result.peakMemoryBytes, 1073741824)
    XCTAssertEqual(result.outputTokenSha256, "abc")
    XCTAssertEqual(result.benchmarkContext, .novel)
    XCTAssertEqual(result.corpusSha256, "def")
    XCTAssertNil(result.slots) // Older servers must not invent a current settings value.
  }

  func testResultExportsPreserveRunSettingsAndNumbers() throws {
    let event = try XCTUnwrap(ThroughputEvent.decode(#"data: {"result":{"model":"qwen-test","slots":2304,"benchmark_context":"code","corpus_sha256":"def","context_tokens":4096,"generation_tokens":128,"generation_limit":128,"ttft_ms":500,"tpot_ms":125,"prefill_tps":8192,"decode_tps":8,"elapsed_seconds":16.5,"throughput_tps":256,"peak_memory_bytes":1073741824,"output_token_sha256":"abc","prompt_cache_reused_tokens":0,"finish_reason":"length"}}"#))
    let result = try XCTUnwrap(event.result)
    let json = try ThroughputOutputFormat.json.render([result])
    let rows = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
    XCTAssertEqual(rows[0]["slots"] as? Int, 2304)
    XCTAssertEqual(rows[0]["context_tokens"] as? Int, 4096)
    XCTAssertEqual(rows[0]["benchmark_context"] as? String, "code")
    XCTAssertEqual(rows[0]["finish_reason"] as? String, "length")
    XCTAssertEqual(rows[0]["corpus_sha256"] as? String, "def")
    let plain = try ThroughputOutputFormat.plainText.render([result])
    XCTAssertTrue(plain.contains("qwen-test"))
    XCTAssertTrue(plain.contains("2304"))
    XCTAssertTrue(plain.contains("4096 / 128"))
    XCTAssertTrue(plain.contains("500.0"))
    XCTAssertFalse(plain.contains("\t"))
    XCTAssertEqual(plain.split(separator: "\n").count, 2)
    let markdown = try ThroughputOutputFormat.markdown.render([result])
    XCTAssertTrue(markdown.hasPrefix("| Model | Context | Slots | Output limit |"))
    XCTAssertTrue(markdown.contains("| qwen-test | Code | 2304 | 128 | 4096 / 128 |"))
    XCTAssertEqual(markdown.split(separator: "\n").count, 3)
    let empty = try ThroughputOutputFormat.json.render([])
    XCTAssertEqual(try XCTUnwrap(JSONSerialization.jsonObject(with: Data(empty.utf8)) as? [Any]).count, 0)
  }

  func testDecodesProgressErrorsAndStreamFraming() throws {
    let progress = try XCTUnwrap(ThroughputEvent.decode(#"data: {"phase":"running","generated":32}"#))
    XCTAssertEqual(progress.phase, "running")
    XCTAssertEqual(progress.generated, 32)
    let error = try XCTUnwrap(ThroughputEvent.decode(#"data: {"error":{"message":"Model load failed"}}"#))
    XCTAssertEqual(error.error?.message, "Model load failed")
    XCTAssertNil(try ThroughputEvent.decode(": heartbeat"))
    XCTAssertNil(try ThroughputEvent.decode("data: [DONE]"))
    XCTAssertThrowsError(try ThroughputEvent.decode("data: invalid"))
  }
}
