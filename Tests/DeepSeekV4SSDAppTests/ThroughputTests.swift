import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ThroughputTests: XCTestCase {
  @MainActor
  func testCompletionUnloadsOnceAfterAllTrialsAndKeepsResults() async throws {
    let session = ThroughputSession()
    session.model = "test-model"
    session.contextLengths = [1024, 4096]
    var events: [String] = []
    let finished = expectation(description: "unloaded")
    session.start(prepare: {}, run: { length, _ in
      events.append("run \(length)")
      return try Self.result(length)
    }, unload: {
      XCTAssertTrue(session.isRunning)
      XCTAssertTrue(session.isUnloading)
      events.append("unload")
      finished.fulfill()
    })
    await fulfillment(of: [finished], timeout: 2)
    await Self.waitForFinish(session)
    XCTAssertEqual(events, ["run 1024", "run 4096", "unload"])
    XCTAssertEqual(session.results.map(\.contextTokens), [1024, 4096])
    XCTAssertEqual(session.phase, "Benchmark complete")
  }

  @MainActor
  func testCancellationUnloadsWithoutCancellingCleanupOrStartingAnotherRun() async throws {
    let session = ThroughputSession()
    session.model = "test-model"
    session.contextLengths = [1024, 4096]
    let entered = expectation(description: "second trial")
    let unloaded = expectation(description: "unloaded")
    var cleanupCount = 0
    session.start(prepare: {}, run: { length, _ in
      if length == 4096 {
        entered.fulfill()
        try await Task.sleep(for: .seconds(60))
      }
      return try Self.result(length)
    }, unload: {
      cleanupCount += 1
      XCTAssertFalse(Task.isCancelled)
      XCTAssertTrue(session.isRunning)
      session.cancel()
      session.start(prepare: { XCTFail("Must wait for unload") },
                    run: { length, _ in try Self.result(length) }, unload: {})
      try await Task.sleep(for: .milliseconds(10))
      XCTAssertFalse(Task.isCancelled)
      unloaded.fulfill()
    })
    await fulfillment(of: [entered], timeout: 2)
    session.cancel()
    await fulfillment(of: [unloaded], timeout: 2)
    await Self.waitForFinish(session)
    XCTAssertEqual(cleanupCount, 1)
    XCTAssertEqual(session.results.map(\.contextTokens), [1024])
    XCTAssertEqual(session.phase, "Benchmark cancelled")
    XCTAssertNil(session.error)
  }

  @MainActor
  func testTrialFailureUnloadsAndReportsCleanupFailure() async {
    let session = ThroughputSession()
    session.model = "test-model"
    session.contextLengths = [1024]
    let finished = expectation(description: "cleanup attempted")
    session.start(prepare: {}, run: { _, _ in throw TestFailure(message: "trial failed") }, unload: {
      finished.fulfill()
      throw TestFailure(message: "unload failed")
    })
    await fulfillment(of: [finished], timeout: 2)
    await Self.waitForFinish(session)
    XCTAssertTrue(session.error?.contains("trial failed") == true)
    XCTAssertTrue(session.error?.contains("unload failed") == true)
    XCTAssertEqual(session.phase, "Benchmark failed")
  }

  @MainActor
  func testPreparationFailureAndDryRunDoNotUnloadExistingModel() async {
    let session = ThroughputSession()
    session.model = "test-model"
    let entered = expectation(description: "prepare failed")
    session.start(prepare: {
      entered.fulfill()
      throw TestFailure(message: "server failed")
    }, run: { _, _ in XCTFail("Must not run"); return try Self.result(1024) },
       unload: { XCTFail("No benchmark model was requested") })
    await fulfillment(of: [entered], timeout: 2)
    await Self.waitForFinish(session)
    session.model = ThroughputSession.dryRunModel
    session.start(prepare: { XCTFail("Dry run must not start the server") },
                  run: { _, _ in XCTFail("Dry run must not request a model"); return try Self.result(1024) },
                  unload: { XCTFail("Dry run must not unload a model") })
    XCTAssertFalse(session.isRunning)
  }

  private struct TestFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
  }

  @MainActor
  private static func waitForFinish(_ session: ThroughputSession) async {
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while session.isRunning && ContinuousClock.now < deadline { await Task.yield() }
    XCTAssertFalse(session.isRunning)
  }

  private static func result(_ length: Int) throws -> ThroughputResult {
    let data = "data: {\"result\":{\"model\":\"test-model\",\"context_tokens\":\(length),\"generation_tokens\":128,\"generation_limit\":128,\"ttft_ms\":500,\"tpot_ms\":10,\"prefill_tps\":100,\"decode_tps\":100,\"elapsed_seconds\":2,\"throughput_tps\":100,\"peak_app_memory_bytes\":1024,\"memory_scope\":\"app\",\"output_token_sha256\":\"test\",\"prompt_cache_reused_tokens\":0,\"benchmark_context\":\"code\",\"corpus_sha256\":\"test\"}}"
    return try XCTUnwrap(ThroughputEvent.decode(data)?.result)
  }

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
    XCTAssertEqual(L10n.string("Peak Memory", language: .english), "Peak Memory")
    XCTAssertEqual(L10n.string("Peak Memory", language: .traditionalChinese), "記憶體峰值")
    XCTAssertEqual(L10n.string("Peak Memory", language: .simplifiedChinese), "内存峰值")
    XCTAssertEqual(L10n.string("Run Benchmark", language: .english), "Run Benchmark")
    XCTAssertEqual(L10n.string("Run Benchmark", language: .traditionalChinese), "執行測試")
    XCTAssertEqual(L10n.string("Run Benchmark", language: .simplifiedChinese), "运行测试")
  }

  func testDecodesActualCountsAndOptionalTimePerToken() throws {
    let event = try XCTUnwrap(ThroughputEvent.decode(#"data: {"result":{"model":"test-model","benchmark_context":"novel","corpus_sha256":"def","context_tokens":4096,"generation_tokens":1,"generation_limit":128,"ttft_ms":500,"tpot_ms":null,"prefill_tps":8192,"decode_tps":0,"elapsed_seconds":0.6,"throughput_tps":6828.3,"peak_app_memory_bytes":1073741824,"memory_scope":"app","output_token_sha256":"abc","prompt_cache_reused_tokens":0}}"#))
    let result = try XCTUnwrap(event.result)
    XCTAssertEqual(result.contextTokens, 4096)
    XCTAssertEqual(result.generationTokens, 1)
    XCTAssertEqual(result.generationLimit, 128)
    XCTAssertNil(result.tpotMs)
    XCTAssertEqual(result.peakAppMemoryBytes, 1073741824)
    XCTAssertEqual(result.memoryScope, "app")
    XCTAssertEqual(result.cells.last, "1.00 GiB")
    XCTAssertEqual(result.outputTokenSha256, "abc")
    XCTAssertEqual(result.benchmarkContext, .novel)
    XCTAssertEqual(result.corpusSha256, "def")
    XCTAssertNil(result.slots) // Older servers must not invent a current settings value.
  }

  func testResultExportsPreserveRunSettingsAndNumbers() throws {
    let event = try XCTUnwrap(ThroughputEvent.decode(#"data: {"result":{"model":"qwen-test","slots":2304,"benchmark_context":"code","corpus_sha256":"def","context_tokens":4096,"generation_tokens":128,"generation_limit":128,"ttft_ms":500,"tpot_ms":125,"prefill_tps":8192,"decode_tps":8,"elapsed_seconds":16.5,"throughput_tps":256,"peak_app_memory_bytes":1073741824,"memory_scope":"app","output_token_sha256":"abc","prompt_cache_reused_tokens":0,"finish_reason":"length"}}"#))
    let result = try XCTUnwrap(event.result)
    let json = try ThroughputOutputFormat.json.render([result])
    let rows = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
    XCTAssertEqual(rows[0]["peak_app_memory_bytes"] as? Double, 1073741824)
    XCTAssertEqual(rows[0]["memory_scope"] as? String, "app")
    XCTAssertNil(rows[0]["peak_memory_bytes"])
    XCTAssertEqual(rows[0]["slots"] as? Int, 2304)
    XCTAssertEqual(rows[0]["context_tokens"] as? Int, 4096)
    XCTAssertEqual(rows[0]["benchmark_context"] as? String, "code")
    XCTAssertEqual(rows[0]["finish_reason"] as? String, "length")
    XCTAssertEqual(rows[0]["corpus_sha256"] as? String, "def")
    let plain = try ThroughputOutputFormat.plainText.render([result])
    XCTAssertTrue(plain.contains("Peak Memory"))
    XCTAssertFalse(plain.contains("MLX"))
    XCTAssertTrue(plain.contains("qwen-test"))
    XCTAssertTrue(plain.contains("2304"))
    XCTAssertTrue(plain.contains("4096 / 128"))
    XCTAssertTrue(plain.contains("500.0"))
    XCTAssertFalse(plain.contains("\t"))
    XCTAssertEqual(plain.split(separator: "\n").count, 2)
    let markdown = try ThroughputOutputFormat.markdown.render([result])
    XCTAssertTrue(markdown.contains("| Peak Memory |"))
    XCTAssertTrue(markdown.hasPrefix("| Model | Context | Slots | Output limit |"))
    XCTAssertTrue(markdown.contains("| qwen-test | Code | 2304 | 128 | 4096 / 128 |"))
    XCTAssertEqual(markdown.split(separator: "\n").count, 3)
    let empty = try ThroughputOutputFormat.json.render([])
    XCTAssertEqual(try XCTUnwrap(JSONSerialization.jsonObject(with: Data(empty.utf8)) as? [Any]).count, 0)
  }

  func testUnavailableMemoryAndLegacyMLXAreNotReportedAsAppMemory() throws {
    for memory in ["\"peak_app_memory_bytes\":null", "\"peak_memory_bytes\":1073741824"] {
      let line = "data: {\"result\":{\"model\":\"test\",\"benchmark_context\":\"code\",\"corpus_sha256\":\"test\",\"context_tokens\":1024,\"generation_tokens\":1,\"generation_limit\":128,\"ttft_ms\":1,\"prefill_tps\":1,\"decode_tps\":1,\"elapsed_seconds\":1,\"throughput_tps\":1,\(memory),\"output_token_sha256\":\"test\",\"prompt_cache_reused_tokens\":0}}"
      let result = try XCTUnwrap(ThroughputEvent.decode(line)?.result)
      XCTAssertNil(result.peakAppMemoryBytes)
      XCTAssertEqual(result.cells.last, "—")
      let json = try ThroughputOutputFormat.json.render([result])
      XCTAssertFalse(json.contains("peak_memory_bytes"))
    }
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
