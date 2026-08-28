import Foundation
import DeepSeekRepack
import XCTest

@testable import DeepSeekV4SSDApp

final class ServerConfigurationTests: XCTestCase {
  func testActiveDownloadDoesNotReserveExtraModelListHeight() {
    XCTAssertFalse(
      shouldShowModelDownloadReason(
        modelIsInstalled: false,
        modelIsDownloading: true,
        hasReason: true
      )
    )
    XCTAssertTrue(
      shouldShowModelDownloadReason(
        modelIsInstalled: false,
        modelIsDownloading: false,
        hasReason: true
      )
    )
  }

  func testDownloadProgressHeightIncludesBottomPadding() {
    XCTAssertEqual(modelDownloadProgressExtraHeight(hasProgressFraction: false), 40)
    XCTAssertEqual(modelDownloadProgressExtraHeight(hasProgressFraction: true), 72)
  }

  @MainActor
  func testLoadedModelIDMapsToItsModelKind() {
    XCTAssertEqual(modelKind(withAPIModelID: "deepseek-v4-flash-0731"), .deepSeekV4)
    XCTAssertEqual(modelKind(withAPIModelID: "qwen3.8-flash-next-fp8"), .qwen3_8FlashNext)
    XCTAssertNil(modelKind(withAPIModelID: "unknown"))
    XCTAssertNil(modelKind(withAPIModelID: nil))
  }

  func testServerAndModelControlsRemainIndependent() {
    XCTAssertFalse(
      modelDownloadIsDisabled(
        serverIsActive: true,
        operationIsBusy: false,
        canStartDownload: true,
        hasPartialDownload: false,
        targetHasPartialDownload: false
      )
    )
    XCTAssertFalse(
      modelSelectionIsLocked(
        serverIsActive: true,
        operationIsBusy: true,
        downloadIsActive: true,
        hasPartialDownload: true
      )
    )
    XCTAssertTrue(
      modelSelectionIsLocked(
        serverIsActive: false,
        operationIsBusy: true,
        downloadIsActive: false,
        hasPartialDownload: false
      )
    )
  }

  func testPowerSavingLegendAnchorsAlignWithSliderNodes() {
    let count = ServerConfiguration.powerSavingLimitOptionsGBps.count
    let totalWidth = CGFloat(700)
    let nodeSpacing = totalWidth / CGFloat(count - 1)

    for index in 1..<(count - 1) {
      let frame = powerSavingLegendFrame(index: index, count: count, totalWidth: totalWidth)
      XCTAssertEqual(frame.midX, CGFloat(index) * nodeSpacing, accuracy: 0.001)
    }
    XCTAssertEqual(
      powerSavingLegendFrame(index: 0, count: count, totalWidth: totalWidth).minX,
      0,
      accuracy: 0.001
    )
    XCTAssertEqual(
      powerSavingLegendFrame(index: count - 1, count: count, totalWidth: totalWidth).maxX,
      totalWidth,
      accuracy: 0.001
    )
  }

  func testServerConfigurationContainsOnlyServerArguments() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    var configuration = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "secret")
    configuration.host = "0.0.0.0"
    configuration.port = 9_000
    configuration.powerSavingLimitGBps = 2

    let arguments = configuration.arguments(modelCatalogPath: "/tmp/catalog.json")

    XCTAssertEqual(configuration.baseURL?.absoluteString, "http://127.0.0.1:9000")
    XCTAssertEqual(
      arguments,
      [
        "-m", "deepseek_v4_ssd.server",
        "--model-catalog", "/tmp/catalog.json",
        "--host", "0.0.0.0",
        "--port", "9000",
      ]
    )
    XCTAssertFalse(arguments.contains("secret"))
    XCTAssertFalse(arguments.contains("--model"))
    XCTAssertFalse(arguments.contains("--public-model"))
  }

  func testServerConfigurationPersistenceDoesNotStoreAPIKey() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    var configuration = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "")
    configuration.host = "0.0.0.0"
    configuration.port = 9_000
    configuration.apiKey = "secret"
    configuration.powerSavingLimitGBps = 0.5

    configuration.save(defaults: isolated.defaults)
    let restored = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "secret")

    XCTAssertEqual(restored, configuration)
    let storedData = try XCTUnwrap(
      isolated.defaults.data(forKey: ServerConfiguration.preferenceKey))
    XCTAssertFalse(String(decoding: storedData, as: UTF8.self).contains("secret"))
  }

  func testAdvancedSettingsUsePerModelDefaultsAndNormalization() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }

    var deepSeek = ModelAdvancedSettings.defaults(for: .deepSeekV4)
    deepSeek.slots = 700
    deepSeek.bf16KVCache = true
    deepSeek.dsparkEnabled = true
    deepSeek.save(for: .deepSeekV4, defaults: isolated.defaults)

    var qwen = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    qwen.slots = 900
    qwen.bf16KVCache = true
    qwen.dsparkEnabled = true
    qwen.defaultTemperature = 1.0
    qwen.defaultTopP = 0.95
    qwen.defaultTopK = 3
    isolated.defaults.set(
      try JSONEncoder().encode(qwen),
      forKey: "modelAdvancedSettings.qwen3.8-flash-next"
    )

    let restoredDeepSeek = ModelAdvancedSettings.loadOrDefault(
      for: .deepSeekV4, defaults: isolated.defaults)
    let restoredQwen = ModelAdvancedSettings.loadOrDefault(
      for: .qwen3_8FlashNext, defaults: isolated.defaults)

    XCTAssertEqual(restoredDeepSeek.slots, 700)
    XCTAssertTrue(restoredDeepSeek.bf16KVCache)
    XCTAssertTrue(restoredDeepSeek.dsparkEnabled)
    XCTAssertEqual(restoredDeepSeek.layerMajorPrefillThreshold, 1_024)
    XCTAssertEqual(restoredQwen.slots, 900)
    XCTAssertEqual(restoredQwen.defaultMaxTokens, 262_144)
    XCTAssertEqual(restoredQwen.defaultTemperature, 0.7)
    XCTAssertEqual(restoredQwen.defaultTopP, 0.8)
    XCTAssertEqual(restoredQwen.defaultTopK, 20)
    XCTAssertFalse(restoredQwen.bf16KVCache)
    XCTAssertFalse(restoredQwen.dsparkEnabled)
    XCTAssertEqual(restoredQwen.layerMajorPrefillThreshold, 1_024)
  }

  func testSavedAdvancedSettingsWithoutPrefillThresholdUseTheNewDefault() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let encoded = try JSONEncoder().encode(
      ModelAdvancedSettings.defaults(for: .deepSeekV4))
    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    object.removeValue(forKey: "layerMajorPrefillThreshold")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: object),
      forKey: "modelAdvancedSettings.deepseek-v4"
    )

    let restored = ModelAdvancedSettings.loadOrDefault(
      for: .deepSeekV4,
      defaults: isolated.defaults
    )

    XCTAssertEqual(restored.layerMajorPrefillThreshold, 1_024)
  }

  func testLegacyAdvancedSettingsMigrateOnlyToTheCurrentModel() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    isolated.defaults.set("deepseek-v4", forKey: "selectedInstallModelKind")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: legacyConfiguration(publicModel: "custom")),
      forKey: ServerConfiguration.preferenceKey
    )

    let deepSeek = ModelAdvancedSettings.loadOrDefault(
      for: .deepSeekV4, defaults: isolated.defaults)
    isolated.defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    let qwen = ModelAdvancedSettings.loadOrDefault(
      for: .qwen3_8FlashNext, defaults: isolated.defaults)

    XCTAssertEqual(deepSeek.slots, 640)
    XCTAssertEqual(deepSeek.defaultTemperature, 0.7)
    XCTAssertEqual(deepSeek.layerMajorPrefillThreshold, 1_024)
    XCTAssertEqual(qwen.slots, 1_152)
    XCTAssertEqual(qwen.defaultTemperature, 0.7)
  }

  @MainActor
  func testAliasCanBeSavedTrimmedClearedAndSetBeforeInstall() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertEqual(
      try library.saveAlias("  work-model  ", for: .deepSeekV4),
      "work-model"
    )
    XCTAssertEqual(library.alias(for: .deepSeekV4), "work-model")
    XCTAssertNil(library.usableModel(for: .deepSeekV4))

    XCTAssertEqual(try library.saveAlias("   ", for: .deepSeekV4), "")
    XCTAssertEqual(library.alias(for: .deepSeekV4), "")
    XCTAssertNil(
      isolated.defaults.string(forKey: ModelLibrary.aliasPreferenceKey(for: .deepSeekV4)))
  }

  @MainActor
  func testAliasValidationIsCaseSensitiveAndRejectsOtherNames() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertNoThrow(
      try library.saveAlias("deepseek-v4-flash-0731", for: .deepSeekV4))
    XCTAssertNoThrow(try library.saveAlias("Work", for: .deepSeekV4))
    XCTAssertNoThrow(try library.saveAlias("work", for: .qwen3_8FlashNext))
    XCTAssertThrowsError(try library.saveAlias("Work", for: .qwen3_8FlashNext))
    XCTAssertThrowsError(
      try library.saveAlias("deepseek-v4-flash-0731", for: .qwen3_8FlashNext))
  }

  @MainActor
  func testLegacyCustomPublicModelMigratesToCurrentAlias() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    isolated.defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: ["publicModel": "  old-alias  "]),
      forKey: ServerConfiguration.preferenceKey
    )

    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertEqual(library.alias(for: .qwen3_8FlashNext), "old-alias")
  }

  @MainActor
  func testLegacyQwenDefaultDoesNotMigrateToAlias() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    isolated.defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    isolated.defaults.set(
      try JSONSerialization.data(
        withJSONObject: ["publicModel": "Qwen/Qwen3.8-Flash-Next-FP8"]),
      forKey: ServerConfiguration.preferenceKey
    )

    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertEqual(library.alias(for: .qwen3_8FlashNext), "")
  }

  @MainActor
  func testCatalogUsesFixedIDsAliasesAndCompleteSnakeCaseRuntime() throws {
    var deepSeek = ModelAdvancedSettings.defaults(for: .deepSeekV4)
    deepSeek.slots = 700
    deepSeek.defaultTemperature = 0.4
    deepSeek.dsparkEnabled = true
    var qwen = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    qwen.slots = 900
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [
        installedModel(
          .deepSeekV4,
          issues: [InstalledFileIssue(path: "common.bin", kind: .checksumMismatch)]
        ),
        installedModel(.qwen3_8FlashNext),
        installedModel(.deepSeekV4, hasDSpark: true),
      ],
      aliases: [.deepSeekV4: "work-model"],
      settings: [.deepSeekV4: deepSeek, .qwen3_8FlashNext: qwen],
      powerSavingLimitGBps: 2
    )

    XCTAssertEqual(
      catalog.models.map(\.id),
      ["deepseek-v4-flash-0731", "qwen3.8-flash-next-fp8"]
    )
    XCTAssertEqual(catalog.models[0].alias, "work-model")
    XCTAssertTrue(catalog.models[0].runtime.dsparkEnabled)
    XCTAssertEqual(catalog.models[0].runtime.powerSavingLimitGBps, 2)
    XCTAssertEqual(catalog.models[1].defaults.temperature, 0.7)
    XCTAssertEqual(catalog.models[1].defaults.topP, 0.8)
    XCTAssertEqual(catalog.models[1].defaults.topK, 20)
    XCTAssertEqual(catalog.availableModels[0].requestName, "work-model")

    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: catalog.encoded()) as? [String: Any])
    let models = try XCTUnwrap(object["models"] as? [[String: Any]])
    let runtime = try XCTUnwrap(models[0]["runtime"] as? [String: Any])
    XCTAssertEqual(
      Set(runtime.keys),
      [
        "slots", "read_workers", "prefetch_read_workers", "prefill_step_size",
        "fp8_kv_cache", "memory_limit_gib", "layer_major_prefill",
        "layer_major_prefill_threshold",
        "prompt_cache_entries", "prompt_cache_memory_gib", "persistent_prompt_cache",
        "persistent_prompt_cache_entries", "prompt_cache_directory",
        "moe_prefill_step_size", "batched_expert_prefill", "fp4_index_cache",
        "dspark_enabled", "dspark_prompt_cache", "dspark_confidence_threshold",
        "dspark_slots", "dspark_hash_prefetch", "dspark_adaptive_block",
        "dspark_fallback_enabled", "dspark_sequential_verification",
        "dspark_hybrid_verification", "expert_route_trace", "expert_page_cache_probe",
        "expert_file_cache_policy", "ready_expert_decode", "staged_expert_streaming",
        "adaptive_expert_prefill_threshold", "power_saving_limit_gbps",
      ]
    )
    XCTAssertEqual(runtime["layer_major_prefill_threshold"] as? Int, 1_024)
    XCTAssertEqual(models[0]["model_kind"] as? String, "deepseek-v4")
    XCTAssertTrue(models[0]["warmup_prompt_path"] is NSNull)
  }

  @MainActor
  func testEmptyCatalogAndTemporaryFileCleanup() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [], aliases: [:], settings: [:], powerSavingLimitGBps: nil)
    let temporary = try TemporaryModelCatalog(catalog: catalog, directory: root)

    XCTAssertTrue(catalog.models.isEmpty)
    XCTAssertTrue(FileManager.default.fileExists(atPath: temporary.url.path))
    let attributes = try FileManager.default.attributesOfItem(atPath: temporary.url.path)
    XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    temporary.remove()
    XCTAssertFalse(FileManager.default.fileExists(atPath: temporary.url.path))
  }

  @MainActor
  func testServerStartFailureRemovesTheTemporaryCatalog() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let runtime = root.appending(path: "runtime")
    let package = runtime.appending(path: "deepseek_v4_ssd")
    let executable = root.appending(path: "invalid-python")
    try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
    try Data().write(to: package.appending(path: "server.py"))
    try Data("#!/missing/whallm-python\n".utf8).write(to: executable)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o700],
      ofItemAtPath: executable.path
    )
    defer { try? FileManager.default.removeItem(at: root) }

    let configuration = ServerConfiguration(
      runtimeDirectory: runtime.path,
      pythonExecutable: executable.path,
      pythonHome: nil,
      sitePackages: nil,
      host: "127.0.0.1",
      port: 11_434,
      apiKey: "",
      powerSavingLimitGBps: nil
    )
    let before = temporaryCatalogNames()
    let controller = ServerController()

    controller.start(configuration, catalog: ModelCatalog(models: []))

    guard case .failed = controller.state else {
      return XCTFail("Expected server startup to fail.")
    }
    XCTAssertEqual(temporaryCatalogNames(), before)
  }

  func testLocalizationSupportsAllSelectableLanguages() {
    XCTAssertEqual(AppLanguage.appDefault, .system)
    XCTAssertEqual(L10n.string("Stopped", language: .english), "Stopped")
    XCTAssertEqual(L10n.string("Language", language: .simplifiedChinese), "语言")
    XCTAssertEqual(L10n.string("Language", language: .traditionalChinese), "語言")
    XCTAssertEqual(L10n.string("Model", language: .traditionalChinese), "模型")
    XCTAssertEqual(L10n.string("Alias", language: .traditionalChinese), "Alias")
    XCTAssertEqual(
      L10n.string(
        "Optional request name for this model. Changes are saved automatically.",
        language: .traditionalChinese
      ),
      "此模型的選用 request 名稱。變更會自動儲存。"
    )
    XCTAssertEqual(L10n.string("Assistant", language: .traditionalChinese), "助理")
  }

  func testSystemLanguageUsesSupportedLanguageOrFallsBackToEnglish() {
    XCTAssertEqual(
      AppLanguage.systemDefault(preferredLanguages: ["zh-Hant-TW"]), .traditionalChinese)
    XCTAssertEqual(
      AppLanguage.systemDefault(preferredLanguages: ["zh-Hans-CN"]), .simplifiedChinese)
    XCTAssertEqual(AppLanguage.systemDefault(preferredLanguages: ["en-US"]), .english)
    XCTAssertEqual(AppLanguage.systemDefault(preferredLanguages: ["ja-JP"]), .english)
  }

  func testAPIKeyRoundTripsThroughIsolatedKeychainItem() {
    let service = "ServerConfigurationTests.\(UUID().uuidString)"
    let account = "api-key"
    defer { AppKeychain.saveAPIKey("", service: service, account: account) }

    AppKeychain.saveAPIKey("secret", service: service, account: account)

    XCTAssertEqual(AppKeychain.readAPIKey(service: service, account: account), "secret")
  }
}

private func isolatedDefaults() throws -> (defaults: UserDefaults, suite: String) {
  let suite = "ServerConfigurationTests.\(UUID().uuidString)"
  return (try XCTUnwrap(UserDefaults(suiteName: suite)), suite)
}

private func temporaryCatalogNames() -> Set<String> {
  let names =
    (try? FileManager.default.contentsOfDirectory(
      atPath: FileManager.default.temporaryDirectory.path)) ?? []
  return Set(names.filter { $0.hasPrefix("whallm-model-catalog-") })
}

private func installedModel(
  _ modelKind: ModelKind,
  hasDSpark: Bool = false,
  issues: [InstalledFileIssue] = []
) -> InstalledModelInfo {
  InstalledModelInfo(
    url: URL(fileURLWithPath: "/tmp/\(modelKind.rawValue).dsv4"),
    size: 1,
    quickIssues: issues,
    hasDSpark: hasDSpark,
    modelKind: modelKind,
    modelID: modelKind == .deepSeekV4
      ? "deepseek-ai/DeepSeek-V4-Flash-0731"
      : "Qwen/Qwen3.8-Flash-Next-FP8"
  )
}

private func legacyConfiguration(publicModel: String) -> [String: Any] {
  [
    "publicModel": publicModel,
    "slots": 640,
    "readWorkers": 3,
    "memoryLimitGiB": 0,
    "prefillStepSize": 0,
    "layerMajorPrefill": true,
    "promptCacheEntries": 2,
    "promptCacheMemoryGiB": 8,
    "warmupPromptPath": "",
    "bf16KVCache": false,
    "dsparkEnabled": false,
    "dsparkSlots": 768,
    "dsparkConfidenceThreshold": 0.6,
    "defaultMaxTokens": 4_096,
    "defaultTemperature": 0.7,
    "defaultTopP": 0.9,
    "defaultTopK": 0,
  ]
}
