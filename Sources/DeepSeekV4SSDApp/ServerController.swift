import Darwin
import DeepSeekRepack
import Foundation
import Security

struct ServerStatus: Decodable {
  struct Runtime: Decodable {}

  struct Performance: Decodable {
    struct ActiveParametersCache: Decodable {
      let hitRate: Double
      let hits: Int
      let misses: Int
      let residentSlots: Int
      let capacitySlots: Int
    }

    let generating: Bool
    let runtimePromptTokens: Int
    let runtimeGenerationTokens: Int
    let accumulatedGenerationTokens: Int
    let completedRequestCount: Int
    let requestSeconds: Double
    let timeToFirstTokenSeconds: Double
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let requestSsdReadBytesPerSecond: Double
    let requestExpertCacheHitRate: Double
    let dsparkEnabled: Bool?
    let dsparkAcceptanceRate: Double?
    let dsparkAverageAcceptedLength: Double?
    let ssdBytesRead: UInt64
    let activeParametersCache: ActiveParametersCache
  }

  let model: String?
  let sourceModel: String?
  let modelPath: String?
  let runtime: Runtime?
  let loadedModel: String?
  let loadingModel: String?
  let performance: Performance

  static func decode(_ data: Data) throws -> ServerStatus {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(ServerStatus.self, from: data)
  }
}

enum PerformanceMetric: String, CaseIterable, Identifiable {
  case prefillTokensPerSecond
  case decodeTokensPerSecond
  case inputTokens
  case outputTokens
  case memoryUsage
  case ssdReadSpeed
  case cacheHitRate
  case firstTokenWaitTime
  case completionTime

  var id: String { rawValue }
}

struct PerformanceSnapshot: Equatable {
  var prefillTokensPerSecond = 0.0
  var decodeTokensPerSecond = 0.0
  var inputTokens = 0.0
  var outputTokens = 0.0
  var memoryUsage = 0.0
  var ssdReadSpeed = 0.0
  var cacheHitRate = 0.0
  var firstTokenWaitTime = 0.0
  var completionTime = 0.0

  subscript(metric: PerformanceMetric) -> Double {
    switch metric {
    case .prefillTokensPerSecond: prefillTokensPerSecond
    case .decodeTokensPerSecond: decodeTokensPerSecond
    case .inputTokens: inputTokens
    case .outputTokens: outputTokens
    case .memoryUsage: memoryUsage
    case .ssdReadSpeed: ssdReadSpeed
    case .cacheHitRate: cacheHitRate
    case .firstTokenWaitTime: firstTokenWaitTime
    case .completionTime: completionTime
    }
  }
}

struct MetricStatistics: Equatable {
  private(set) var count = 0
  private(set) var minimum = 0.0
  private(set) var total = 0.0
  private(set) var maximum = 0.0
  private var sortedValues: [Double] = []

  var average: Double { count == 0 ? 0 : total / Double(count) }
  var p95: Double {
    guard count > 0 else { return 0 }
    return sortedValues[Int(ceil(Double(count) * 0.95)) - 1]
  }

  mutating func record(_ value: Double) {
    guard value.isFinite else { return }
    if count == 0 {
      minimum = value
      maximum = value
    } else {
      minimum = Swift.min(minimum, value)
      maximum = Swift.max(maximum, value)
    }
    count += 1
    total += value
    var lowerBound = 0
    var upperBound = sortedValues.count
    while lowerBound < upperBound {
      let middle = lowerBound + (upperBound - lowerBound) / 2
      if sortedValues[middle] < value {
        lowerBound = middle + 1
      } else {
        upperBound = middle
      }
    }
    sortedValues.insert(value, at: lowerBound)
  }
}

struct PerformanceHistory: Equatable {
  private(set) var values: [PerformanceMetric: MetricStatistics] = [:]

  var isEmpty: Bool { values.isEmpty }

  subscript(metric: PerformanceMetric) -> MetricStatistics? { values[metric] }

  mutating func record(
    _ snapshot: PerformanceSnapshot,
    excluding excludedMetric: PerformanceMetric? = nil
  ) {
    for metric in PerformanceMetric.allCases where metric != excludedMetric {
      values[metric, default: MetricStatistics()].record(snapshot[metric])
    }
  }

  mutating func clear() {
    values.removeAll()
  }
}

struct LivePerformance: Equatable {
  var hasStatus = false
  var generating = false
  var completedRequestCount = 0
  var accumulatedOutputTokens = 0
  var snapshot = PerformanceSnapshot()
  var dsparkEnabled = false
  var dsparkAcceptanceRate = 0.0
  var dsparkAverageAcceptedLength = 0.0
  var loadedModel: String?
  var loadingModel: String?

  var liveFirstTokenWaitTime: Double {
    if generating && snapshot.outputTokens == 0 {
      return snapshot.completionTime
    }
    return snapshot.firstTokenWaitTime
  }
}

private struct RuntimeEnvironment {
  let runtimeDirectory: URL
  let pythonExecutable: URL
  let pythonHome: URL?
  let sitePackages: URL?

  static var current: RuntimeEnvironment {
    let bundle = Bundle.main.bundleURL
    let bundledPython = bundle.appending(path: "Contents/MacOS/python3")
    let resources = bundle.appending(path: "Contents/Resources")
    if FileManager.default.isExecutableFile(atPath: bundledPython.path) {
      return RuntimeEnvironment(
        runtimeDirectory: resources.appending(path: "runtime", directoryHint: .isDirectory),
        pythonExecutable: bundledPython,
        pythonHome: bundle.appending(
          path: "Contents/Frameworks/Python.framework/Versions/Current",
          directoryHint: .isDirectory
        ),
        sitePackages: resources.appending(
          path: "python/site-packages",
          directoryHint: .isDirectory
        )
      )
    }

    let project = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    return RuntimeEnvironment(
      runtimeDirectory: project.appending(path: "runtime", directoryHint: .isDirectory),
      pythonExecutable: project.appending(path: ".venv/bin/python"),
      pythonHome: nil,
      sitePackages: nil
    )
  }
}

struct ModelAdvancedSettings: Codable, Equatable, Sendable {
  private static let legacyModelPreference = "modelAdvancedSettingsLegacyModelKind"

  var slots = 1_152
  var readWorkers = 4
  var memoryLimitGiB = 0
  var prefillStepSize = 0
  var layerMajorPrefill = true
  var promptCacheEntries = 2
  var promptCacheMemoryGiB = 8
  var warmupPromptPath = ""
  var bf16KVCache = false
  var dsparkEnabled = false
  var dsparkSlots = 768
  var dsparkConfidenceThreshold = 0.6
  var defaultMaxTokens = 272_000
  var defaultTemperature = 0.2
  var defaultTopP = 0.98
  var defaultTopK = 0

  static func defaults(for modelKind: ModelKind) -> ModelAdvancedSettings {
    var settings = ModelAdvancedSettings()
    if modelKind == .qwen3_8FlashNext {
      settings.defaultMaxTokens = 262_144
      settings.defaultTemperature = 0.7
      settings.defaultTopP = 0.8
      settings.defaultTopK = 20
    }
    return settings
  }

  private init() {}

  func normalized(for modelKind: ModelKind) -> ModelAdvancedSettings {
    var settings = self
    if modelKind == .qwen3_8FlashNext {
      settings.bf16KVCache = false
      settings.dsparkEnabled = false
      settings.defaultTemperature = 0.7
      settings.defaultTopP = 0.8
      settings.defaultTopK = 20
    }
    return settings
  }

  static func load(
    for modelKind: ModelKind,
    defaults: UserDefaults
  ) -> ModelAdvancedSettings? {
    guard let data = defaults.data(forKey: preferenceKey(for: modelKind)) else { return nil }
    return try? JSONDecoder().decode(ModelAdvancedSettings.self, from: data)
  }

  static func loadOrDefault(
    for modelKind: ModelKind,
    defaults: UserDefaults = .standard
  ) -> ModelAdvancedSettings {
    let settings = load(for: modelKind, defaults: defaults)
      ?? legacySettings(for: modelKind, defaults: defaults)
      ?? ModelAdvancedSettings.defaults(for: modelKind)
    let normalized = settings.normalized(for: modelKind)
    normalized.save(for: modelKind, defaults: defaults)
    return normalized
  }

  func save(for modelKind: ModelKind, defaults: UserDefaults = .standard) {
    guard let data = try? JSONEncoder().encode(normalized(for: modelKind)) else { return }
    defaults.set(data, forKey: Self.preferenceKey(for: modelKind))
  }

  func validate(for modelKind: ModelKind) throws {
    guard slots >= 6 else {
      throw ConfigurationError(L10n.string("Slots must be at least 6."))
    }
    guard readWorkers >= 1, memoryLimitGiB >= 0, prefillStepSize >= 0 else {
      throw ConfigurationError(
        L10n.string(
          "Read workers must be greater than 0. Memory limit and prefill step size must be 0 or greater."
        ))
    }
    guard promptCacheEntries >= 1, promptCacheMemoryGiB >= 1 else {
      throw ConfigurationError(
        L10n.string("Prompt cache entries and the memory limit must be greater than 0."))
    }
    guard dsparkSlots >= 30, (0...1).contains(dsparkConfidenceThreshold) else {
      throw ConfigurationError(L10n.string("Correct the default generation parameters."))
    }
    guard (1...272_000).contains(defaultMaxTokens),
      (0...2).contains(defaultTemperature),
      (0.000_001...1).contains(defaultTopP),
      (0...248_320).contains(defaultTopK)
    else {
      throw ConfigurationError(L10n.string("Correct the default generation parameters."))
    }
    if !warmupPromptPath.isEmpty,
      !FileManager.default.isReadableFile(atPath: warmupPromptPath)
    {
      throw ConfigurationError(L10n.string("The warmup prompt file cannot be read."))
    }
    if modelKind == .qwen3_8FlashNext, dsparkEnabled {
      throw ConfigurationError(L10n.string("Qwen3.8-Flash-Next does not support DSpark."))
    }
  }

  private struct Legacy: Decodable {
    let publicModel: String?
    let slots: Int
    let readWorkers: Int
    let memoryLimitGiB: Int?
    let prefillStepSize: Int
    let layerMajorPrefill: Bool
    let promptCacheEntries: Int
    let promptCacheMemoryGiB: Int
    let warmupPromptPath: String
    let bf16KVCache: Bool
    let dsparkEnabled: Bool
    let dsparkSlots: Int
    let dsparkConfidenceThreshold: Double
    let defaultMaxTokens: Int
    let defaultTemperature: Double
    let defaultTopP: Double
    let defaultTopK: Int?
  }

  private static func legacySettings(
    for modelKind: ModelKind,
    defaults: UserDefaults
  ) -> ModelAdvancedSettings? {
    guard let data = defaults.data(forKey: ServerConfiguration.preferenceKey),
      let legacy = try? JSONDecoder().decode(Legacy.self, from: data)
    else { return nil }
    let identifiedKind: ModelKind
    if let savedKind = defaults.string(forKey: legacyModelPreference)
      .flatMap(ModelKind.init(rawValue:))
    {
      identifiedKind = savedKind
    } else {
      let selectedKind = defaults.string(forKey: "selectedInstallModelKind")
        .flatMap(ModelKind.init(rawValue:)) ?? .deepSeekV4
      switch legacy.publicModel {
      case "Qwen/Qwen3.8-Flash-Next-FP8", "qwen3.8-flash-next-fp8":
        identifiedKind = .qwen3_8FlashNext
      case "deepseek-v4-flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731":
        identifiedKind = .deepSeekV4
      default:
        identifiedKind = selectedKind
      }
      defaults.set(identifiedKind.rawValue, forKey: legacyModelPreference)
    }
    guard identifiedKind == modelKind else { return nil }
    var settings = ModelAdvancedSettings.defaults(for: modelKind)
    settings.slots = legacy.slots
    settings.readWorkers = legacy.readWorkers
    settings.memoryLimitGiB = legacy.memoryLimitGiB ?? 0
    settings.prefillStepSize = legacy.prefillStepSize
    settings.layerMajorPrefill = legacy.layerMajorPrefill
    settings.promptCacheEntries = legacy.promptCacheEntries
    settings.promptCacheMemoryGiB = legacy.promptCacheMemoryGiB
    settings.warmupPromptPath = legacy.warmupPromptPath
    settings.bf16KVCache = legacy.bf16KVCache
    settings.dsparkEnabled = legacy.dsparkEnabled
    settings.dsparkSlots = legacy.dsparkSlots
    settings.dsparkConfidenceThreshold = legacy.dsparkConfidenceThreshold
    settings.defaultMaxTokens = legacy.defaultMaxTokens
    settings.defaultTemperature = legacy.defaultTemperature
    settings.defaultTopP = legacy.defaultTopP
    settings.defaultTopK = legacy.defaultTopK ?? 0
    return settings
  }

  private static func preferenceKey(for modelKind: ModelKind) -> String {
    "modelAdvancedSettings.\(modelKind.rawValue)"
  }
}

struct ServerConfiguration: Codable, Equatable {
  static let preferenceKey = "serverConfiguration"
  static let powerSavingLimitOptionsGBps: [Double?] = [0.5, 1, 2, 3, 5, 10, 25, nil]

  var runtimeDirectory: String
  var pythonExecutable: String
  var pythonHome: String?
  var sitePackages: String?
  var host: String
  var port: Int
  var apiKey: String
  var powerSavingLimitGBps: Double?

  static var localDefault: ServerConfiguration {
    load(defaults: .standard, apiKey: AppKeychain.readAPIKey())
  }

  static func load(defaults: UserDefaults, apiKey: String) -> ServerConfiguration {
    let runtime = RuntimeEnvironment.current
    var configuration = ServerConfiguration(
      runtimeDirectory: runtime.runtimeDirectory.path,
      pythonExecutable: runtime.pythonExecutable.path,
      pythonHome: runtime.pythonHome?.path,
      sitePackages: runtime.sitePackages?.path,
      host: "127.0.0.1",
      port: 11_434,
      apiKey: "",
      powerSavingLimitGBps: nil
    )
    if let data = defaults.data(forKey: preferenceKey),
      var saved = decodeSavedConfiguration(data)
    {
      saved.runtimeDirectory = configuration.runtimeDirectory
      saved.pythonExecutable = configuration.pythonExecutable
      saved.pythonHome = configuration.pythonHome
      saved.sitePackages = configuration.sitePackages
      saved.apiKey = apiKey
      if !powerSavingLimitOptionsGBps.contains(where: {
        $0 == saved.powerSavingLimitGBps
      }) {
        saved.powerSavingLimitGBps = nil
      }
      configuration = saved
    } else {
      configuration.apiKey = apiKey
    }
    return configuration
  }

  private static func decodeSavedConfiguration(_ data: Data) -> ServerConfiguration? {
    try? JSONDecoder().decode(ServerConfiguration.self, from: data)
  }

  func save(defaults: UserDefaults = .standard) {
    var saved = self
    saved.runtimeDirectory = ""
    saved.pythonExecutable = ""
    saved.pythonHome = nil
    saved.sitePackages = nil
    saved.apiKey = ""
    guard let data = try? JSONEncoder().encode(saved) else { return }
    defaults.set(data, forKey: Self.preferenceKey)
  }

  var baseURL: URL? {
    let clientHost = ["0.0.0.0", "::"].contains(host) ? "127.0.0.1" : host
    let formattedHost = clientHost.contains(":") ? "[\(clientHost)]" : clientHost
    return URL(string: "http://\(formattedHost):\(port)")
  }

  func arguments(modelCatalogPath: String) -> [String] {
    [
      "-m", "deepseek_v4_ssd.server",
      "--model-catalog", modelCatalogPath,
      "--host", host,
      "--port", String(port),
    ]
  }

  func validate() throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: runtimeDirectory, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw ConfigurationError(L10n.string("The app runtime is missing. Install the app again."))
    }
    guard FileManager.default.isExecutableFile(atPath: pythonExecutable) else {
      throw ConfigurationError(
        L10n.string("Python is missing from the app. Install the app again."))
    }
    guard
      FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: runtimeDirectory)
          .appending(path: "deepseek_v4_ssd/server.py").path
      )
    else {
      throw ConfigurationError(L10n.string("The app runtime is incomplete. Install the app again."))
    }
    guard !host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw ConfigurationError(L10n.string("Host cannot be empty."))
    }
    guard (1...65_535).contains(port) else {
      throw ConfigurationError(L10n.string("Port must be from 1 through 65535."))
    }
    guard ["127.0.0.1", "::1", "localhost"].contains(host) || !apiKey.isEmpty else {
      throw ConfigurationError(L10n.string("An API key is required for a non-local host."))
    }
  }
}

struct CatalogModel: Identifiable, Equatable, Sendable {
  let id: String
  let alias: String?

  var requestName: String { alias ?? id }
}

struct ModelCatalog: Codable, Equatable, Sendable {
  struct Entry: Codable, Equatable, Sendable {
    struct Runtime: Codable, Equatable, Sendable {
      let slots: Int
      let readWorkers: Int
      let prefetchReadWorkers: Int
      let prefillStepSize: Int
      let fp8KVCache: Bool
      let memoryLimitGiB: Int
      let layerMajorPrefill: Bool
      let promptCacheEntries: Int
      let promptCacheMemoryGiB: Int
      let persistentPromptCache: Bool
      let persistentPromptCacheEntries: Int
      let promptCacheDirectory: String?
      let moePrefillStepSize: Int
      let batchedExpertPrefill: Bool
      let fp4IndexCache: Bool
      let dsparkEnabled: Bool
      let dsparkPromptCache: Bool
      let dsparkConfidenceThreshold: Double
      let dsparkSlots: Int
      let dsparkHashPrefetch: Bool
      let dsparkAdaptiveBlock: Bool
      let dsparkFallbackEnabled: Bool
      let dsparkSequentialVerification: Bool
      let dsparkHybridVerification: Bool
      let expertRouteTrace: String?
      let expertPageCacheProbe: Bool
      let expertFileCachePolicy: String
      let readyExpertDecode: Bool
      let stagedExpertStreaming: Bool
      let adaptiveExpertPrefillThreshold: Double?
      let powerSavingLimitGBps: Double?

      enum CodingKeys: String, CodingKey {
        case slots
        case readWorkers = "read_workers"
        case prefetchReadWorkers = "prefetch_read_workers"
        case prefillStepSize = "prefill_step_size"
        case fp8KVCache = "fp8_kv_cache"
        case memoryLimitGiB = "memory_limit_gib"
        case layerMajorPrefill = "layer_major_prefill"
        case promptCacheEntries = "prompt_cache_entries"
        case promptCacheMemoryGiB = "prompt_cache_memory_gib"
        case persistentPromptCache = "persistent_prompt_cache"
        case persistentPromptCacheEntries = "persistent_prompt_cache_entries"
        case promptCacheDirectory = "prompt_cache_directory"
        case moePrefillStepSize = "moe_prefill_step_size"
        case batchedExpertPrefill = "batched_expert_prefill"
        case fp4IndexCache = "fp4_index_cache"
        case dsparkEnabled = "dspark_enabled"
        case dsparkPromptCache = "dspark_prompt_cache"
        case dsparkConfidenceThreshold = "dspark_confidence_threshold"
        case dsparkSlots = "dspark_slots"
        case dsparkHashPrefetch = "dspark_hash_prefetch"
        case dsparkAdaptiveBlock = "dspark_adaptive_block"
        case dsparkFallbackEnabled = "dspark_fallback_enabled"
        case dsparkSequentialVerification = "dspark_sequential_verification"
        case dsparkHybridVerification = "dspark_hybrid_verification"
        case expertRouteTrace = "expert_route_trace"
        case expertPageCacheProbe = "expert_page_cache_probe"
        case expertFileCachePolicy = "expert_file_cache_policy"
        case readyExpertDecode = "ready_expert_decode"
        case stagedExpertStreaming = "staged_expert_streaming"
        case adaptiveExpertPrefillThreshold = "adaptive_expert_prefill_threshold"
        case powerSavingLimitGBps = "power_saving_limit_gbps"
      }

      func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(slots, forKey: .slots)
        try values.encode(readWorkers, forKey: .readWorkers)
        try values.encode(prefetchReadWorkers, forKey: .prefetchReadWorkers)
        try values.encode(prefillStepSize, forKey: .prefillStepSize)
        try values.encode(fp8KVCache, forKey: .fp8KVCache)
        try values.encode(memoryLimitGiB, forKey: .memoryLimitGiB)
        try values.encode(layerMajorPrefill, forKey: .layerMajorPrefill)
        try values.encode(promptCacheEntries, forKey: .promptCacheEntries)
        try values.encode(promptCacheMemoryGiB, forKey: .promptCacheMemoryGiB)
        try values.encode(persistentPromptCache, forKey: .persistentPromptCache)
        try values.encode(persistentPromptCacheEntries, forKey: .persistentPromptCacheEntries)
        if let promptCacheDirectory {
          try values.encode(promptCacheDirectory, forKey: .promptCacheDirectory)
        } else {
          try values.encodeNil(forKey: .promptCacheDirectory)
        }
        try values.encode(moePrefillStepSize, forKey: .moePrefillStepSize)
        try values.encode(batchedExpertPrefill, forKey: .batchedExpertPrefill)
        try values.encode(fp4IndexCache, forKey: .fp4IndexCache)
        try values.encode(dsparkEnabled, forKey: .dsparkEnabled)
        try values.encode(dsparkPromptCache, forKey: .dsparkPromptCache)
        try values.encode(dsparkConfidenceThreshold, forKey: .dsparkConfidenceThreshold)
        try values.encode(dsparkSlots, forKey: .dsparkSlots)
        try values.encode(dsparkHashPrefetch, forKey: .dsparkHashPrefetch)
        try values.encode(dsparkAdaptiveBlock, forKey: .dsparkAdaptiveBlock)
        try values.encode(dsparkFallbackEnabled, forKey: .dsparkFallbackEnabled)
        try values.encode(
          dsparkSequentialVerification, forKey: .dsparkSequentialVerification)
        try values.encode(dsparkHybridVerification, forKey: .dsparkHybridVerification)
        if let expertRouteTrace {
          try values.encode(expertRouteTrace, forKey: .expertRouteTrace)
        } else {
          try values.encodeNil(forKey: .expertRouteTrace)
        }
        try values.encode(expertPageCacheProbe, forKey: .expertPageCacheProbe)
        try values.encode(expertFileCachePolicy, forKey: .expertFileCachePolicy)
        try values.encode(readyExpertDecode, forKey: .readyExpertDecode)
        try values.encode(stagedExpertStreaming, forKey: .stagedExpertStreaming)
        if let adaptiveExpertPrefillThreshold {
          try values.encode(
            adaptiveExpertPrefillThreshold, forKey: .adaptiveExpertPrefillThreshold)
        } else {
          try values.encodeNil(forKey: .adaptiveExpertPrefillThreshold)
        }
        if let powerSavingLimitGBps {
          try values.encode(powerSavingLimitGBps, forKey: .powerSavingLimitGBps)
        } else {
          try values.encodeNil(forKey: .powerSavingLimitGBps)
        }
      }
    }

    struct Defaults: Codable, Equatable, Sendable {
      let maxTokens: Int
      let temperature: Double
      let topP: Double
      let topK: Int

      enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
      }
    }

    let id: String
    let alias: String?
    let path: String
    let modelKind: String
    let runtime: Runtime
    let defaults: Defaults
    let warmupPromptPath: String?

    enum CodingKeys: String, CodingKey {
      case id, alias, path, runtime, defaults
      case modelKind = "model_kind"
      case warmupPromptPath = "warmup_prompt_path"
    }

    func encode(to encoder: Encoder) throws {
      var values = encoder.container(keyedBy: CodingKeys.self)
      try values.encode(id, forKey: .id)
      if let alias {
        try values.encode(alias, forKey: .alias)
      } else {
        try values.encodeNil(forKey: .alias)
      }
      try values.encode(path, forKey: .path)
      try values.encode(modelKind, forKey: .modelKind)
      try values.encode(runtime, forKey: .runtime)
      try values.encode(defaults, forKey: .defaults)
      if let warmupPromptPath {
        try values.encode(warmupPromptPath, forKey: .warmupPromptPath)
      } else {
        try values.encodeNil(forKey: .warmupPromptPath)
      }
    }
  }

  let version: Int
  let models: [Entry]

  init(models: [Entry]) {
    version = 1
    self.models = models
  }

  var availableModels: [CatalogModel] {
    models.map { CatalogModel(id: $0.id, alias: $0.alias) }
  }

  func encoded() throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(self)
  }
}

struct TemporaryModelCatalog {
  let url: URL

  init(
    catalog: ModelCatalog,
    directory: URL = FileManager.default.temporaryDirectory
  ) throws {
    url = directory.appending(path: "whallm-model-catalog-\(UUID().uuidString).json")
    do {
      try catalog.encoded().write(to: url, options: .atomic)
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o600],
        ofItemAtPath: url.path
      )
    } catch {
      try? FileManager.default.removeItem(at: url)
      throw error
    }
  }

  func remove() {
    try? FileManager.default.removeItem(at: url)
  }
}

enum AppKeychain {
  private static let service = "com.deepseekv4ssd.app"
  private static let account = "server-api-key"

  static func readAPIKey(service: String = service, account: String = account) -> String {
    var query = baseQuery(service: service, account: account)
    query[kSecReturnData as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var result: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
      let data = result as? Data
    else { return "" }
    return String(data: data, encoding: .utf8) ?? ""
  }

  static func saveAPIKey(
    _ value: String,
    service: String = service,
    account: String = account
  ) {
    let query = baseQuery(service: service, account: account)
    if value.isEmpty {
      SecItemDelete(query as CFDictionary)
      return
    }
    guard readAPIKey(service: service, account: account) != value else { return }
    let data = Data(value.utf8)
    let status = SecItemUpdate(
      query as CFDictionary,
      [kSecValueData as String: data] as CFDictionary
    )
    if status == errSecItemNotFound {
      var item = query
      item[kSecValueData as String] = data
      SecItemAdd(item as CFDictionary, nil)
    }
  }

  private static func baseQuery(service: String, account: String) -> [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
    ]
  }
}

private struct ConfigurationError: LocalizedError {
  let message: String

  init(_ message: String) {
    self.message = message
  }

  var errorDescription: String? { message }
}

@MainActor
final class ServerController: ObservableObject {
  enum State: Equatable {
    case stopped
    case starting
    case running
    case stopping
    case failed(String)

    var label: String {
      switch self {
      case .stopped: L10n.string("Stopped")
      case .starting: L10n.string("Starting")
      case .running: L10n.string("Running")
      case .stopping: L10n.string("Stopping")
      case .failed: L10n.string("Start failed")
      }
    }

    var symbol: String {
      switch self {
      case .running: "checkmark.circle.fill"
      case .starting, .stopping: "clock.fill"
      case .failed: "exclamationmark.triangle.fill"
      case .stopped: "stop.circle.fill"
      }
    }
  }

  @Published private(set) var state: State = .stopped
  @Published private(set) var log = ""
  @Published private(set) var performance = LivePerformance()
  @Published private(set) var performanceHistory = PerformanceHistory()
  @Published private(set) var catalogModels: [CatalogModel] = []

  private var process: Process?
  private var outputTask: Task<Void, Never>?
  private var monitorTask: Task<Void, Never>?
  private var monitorConfiguration: ServerConfiguration?
  private var temporaryModelCatalog: TemporaryModelCatalog?
  private var previousSSDBytes: UInt64?
  private var previousSSDTime: ContinuousClock.Instant?
  private var lastRecordedCompletedRequestCount = 0
  private var lastLoadedModel: String?

  var isActive: Bool {
    switch state {
    case .starting, .running, .stopping: true
    case .stopped, .failed: false
    }
  }

  func start(_ configuration: ServerConfiguration, catalog: ModelCatalog) {
    guard !isActive else { return }
    do {
      try configuration.validate()
      let temporaryModelCatalog = try TemporaryModelCatalog(catalog: catalog)
      self.temporaryModelCatalog = temporaryModelCatalog
      let process = Process()
      let output = Pipe()
      let runtimeURL = URL(fileURLWithPath: configuration.runtimeDirectory)
      process.executableURL = URL(fileURLWithPath: configuration.pythonExecutable)
      process.currentDirectoryURL = runtimeURL
      process.arguments = configuration.arguments(
        modelCatalogPath: temporaryModelCatalog.url.path)
      process.standardOutput = output
      process.standardError = output
      var environment = ProcessInfo.processInfo.environment
      environment["PYTHONPATH"] = [configuration.runtimeDirectory, configuration.sitePackages]
        .compactMap { $0 }
        .joined(separator: ":")
      environment["PYTHONDONTWRITEBYTECODE"] = "1"
      if let pythonHome = configuration.pythonHome {
        environment["PYTHONHOME"] = pythonHome
      }
      if configuration.apiKey.isEmpty {
        environment.removeValue(forKey: "DEEPSEEK_API_KEY")
      } else {
        environment["DEEPSEEK_API_KEY"] = configuration.apiKey
      }
      process.environment = environment
      process.terminationHandler = { [weak self] finished in
        let status = finished.terminationStatus
        Task { @MainActor in self?.didTerminate(status: status) }
      }

      log = ""
      state = .starting
      try process.run()
      self.process = process
      catalogModels = catalog.availableModels
      monitorConfiguration = configuration
      readOutput(output.fileHandleForReading)
      startMonitoring()
    } catch {
      temporaryModelCatalog?.remove()
      temporaryModelCatalog = nil
      catalogModels = []
      state = .failed(error.localizedDescription)
      appendLog(L10n.string("Error: %@", error.localizedDescription) + "\n")
    }
  }

  func stop() {
    guard let process, process.isRunning else {
      temporaryModelCatalog?.remove()
      temporaryModelCatalog = nil
      catalogModels = []
      state = .stopped
      return
    }
    state = .stopping
    process.interrupt()
  }

  func clearPerformanceHistory() {
    performanceHistory.clear()
    lastRecordedCompletedRequestCount = performance.completedRequestCount
  }

  private func readOutput(_ handle: FileHandle) {
    outputTask?.cancel()
    outputTask = Task.detached(priority: .utility) { [weak self] in
      while !Task.isCancelled {
        let data = handle.availableData
        guard !data.isEmpty else { break }
        let text = String(decoding: data, as: UTF8.self)
        await self?.received(text)
      }
    }
  }

  private func received(_ text: String) {
    appendLog(text)
    if case .starting = state, log.contains("Ready:") {
      state = .running
    }
  }

  private func startMonitoring() {
    monitorTask?.cancel()
    monitorTask = Task { [weak self] in
      while !Task.isCancelled {
        guard let self else { return }
        await self.refreshPerformance()
        do {
          try await Task.sleep(for: .seconds(1))
        } catch {
          return
        }
      }
    }
  }

  private func refreshPerformance() async {
    guard let process, process.isRunning else { return }
    let memoryBytes = residentMemoryBytes(process.processIdentifier)
    performance.snapshot.memoryUsage = Double(memoryBytes)
    guard case .running = state,
      let configuration = monitorConfiguration,
      let baseURL = configuration.baseURL
    else { return }

    var request = URLRequest(url: baseURL.appending(path: "api/status"))
    request.timeoutInterval = 2
    if !configuration.apiKey.isEmpty {
      request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "Authorization")
    }
    do {
      let (data, response) = try await URLSession.shared.data(for: request)
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
      let status = try ServerStatus.decode(data)
      updateLoadedModel(
        status.loadedModel,
        completedRequestCount: status.performance.completedRequestCount
      )
      let now = ContinuousClock.now
      let bytesPerSecond: Double
      if let previousSSDBytes, let previousSSDTime,
        status.performance.ssdBytesRead >= previousSSDBytes
      {
        let elapsed = seconds(from: previousSSDTime.duration(to: now))
        bytesPerSecond =
          elapsed > 0 ? Double(status.performance.ssdBytesRead - previousSSDBytes) / elapsed : 0
      } else {
        bytesPerSecond = 0
      }
      previousSSDBytes = status.performance.ssdBytesRead
      previousSSDTime = now
      let cache = status.performance.activeParametersCache
      let requestCompleted = status.performance.completedRequestCount > 0
      let cacheHitRate =
        status.performance.generating || !requestCompleted
        ? cache.hitRate : status.performance.requestExpertCacheHitRate
      let live = LivePerformance(
        hasStatus: true,
        generating: status.performance.generating,
        completedRequestCount: status.performance.completedRequestCount,
        accumulatedOutputTokens: status.performance.accumulatedGenerationTokens,
        snapshot: PerformanceSnapshot(
          prefillTokensPerSecond: status.performance.prefillTokensPerSecond,
          decodeTokensPerSecond: status.performance.decodeTokensPerSecond,
          inputTokens: Double(status.performance.runtimePromptTokens),
          outputTokens: Double(status.performance.runtimeGenerationTokens),
          memoryUsage: Double(memoryBytes),
          ssdReadSpeed: bytesPerSecond,
          cacheHitRate: cacheHitRate,
          firstTokenWaitTime: status.performance.timeToFirstTokenSeconds,
          completionTime: status.performance.requestSeconds
        ),
        dsparkEnabled: status.performance.dsparkEnabled ?? false,
        dsparkAcceptanceRate: status.performance.dsparkAcceptanceRate ?? 0,
        dsparkAverageAcceptedLength: status.performance.dsparkAverageAcceptedLength ?? 0,
        loadedModel: status.loadedModel,
        loadingModel: status.loadingModel
      )
      performance = live
      recordPerformanceSample(live)
    } catch {
      return
    }
  }

  func recordPerformanceSample(_ live: LivePerformance) {
    if live.completedRequestCount < lastRecordedCompletedRequestCount {
      lastRecordedCompletedRequestCount = live.completedRequestCount
    }

    let requestCompleted = live.completedRequestCount > lastRecordedCompletedRequestCount
    if live.generating || requestCompleted {
      performanceHistory.record(
        live.snapshot,
        excluding: requestCompleted ? nil : .firstTokenWaitTime
      )
    }
    if requestCompleted {
      lastRecordedCompletedRequestCount = live.completedRequestCount
    }
  }

  func updateLoadedModel(_ model: String?, completedRequestCount: Int) {
    guard let model, model != lastLoadedModel else { return }
    performanceHistory.clear()
    lastRecordedCompletedRequestCount = completedRequestCount
    previousSSDBytes = nil
    previousSSDTime = nil
    lastLoadedModel = model
  }

  private func residentMemoryBytes(_ processID: Int32) -> UInt64 {
    var information = proc_taskinfo()
    let size = Int32(MemoryLayout<proc_taskinfo>.size)
    let result = proc_pidinfo(processID, PROC_PIDTASKINFO, 0, &information, size)
    return result == size ? information.pti_resident_size : 0
  }

  private func seconds(from duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }

  private func appendLog(_ text: String) {
    log += text
    if log.count > 40_000 {
      log.removeFirst(log.count - 40_000)
    }
  }

  private func didTerminate(status: Int32) {
    outputTask?.cancel()
    outputTask = nil
    monitorTask?.cancel()
    monitorTask = nil
    monitorConfiguration = nil
    temporaryModelCatalog?.remove()
    temporaryModelCatalog = nil
    catalogModels = []
    previousSSDBytes = nil
    previousSSDTime = nil
    performance = LivePerformance()
    lastLoadedModel = nil
    process = nil
    if case .stopping = state {
      state = .stopped
    } else if status == 0 {
      state = .stopped
    } else {
      let message = L10n.string("The server stopped with exit code %d.", status)
      state = .failed(message)
      appendLog("\n\(message)\n")
    }
  }
}
