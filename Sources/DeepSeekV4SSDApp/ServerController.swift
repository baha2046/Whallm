import Darwin
import Foundation
import Security

struct ServerStatus: Decodable {
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

  mutating func record(_ snapshot: PerformanceSnapshot) {
    for metric in PerformanceMetric.allCases {
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

struct ServerConfiguration: Codable, Equatable {
  private static let preferenceKey = "serverConfiguration"
  static let powerSavingLimitOptionsGBps: [Double?] = [0.5, 1, 2, 3, 5, 10, 25, nil]

  var runtimeDirectory: String
  var pythonExecutable: String
  var pythonHome: String?
  var sitePackages: String?
  var modelPath: String
  var host: String
  var port: Int
  var apiKey: String
  var publicModel: String
  var slots: Int
  var readWorkers: Int
  var powerSavingLimitGBps: Double?
  var prefillStepSize: Int
  var layerMajorPrefill: Bool
  var promptCacheEntries: Int
  var promptCacheMemoryGiB: Int
  var warmupPromptPath: String
  var bf16KVCache: Bool
  var dsparkEnabled: Bool
  var dsparkSlots: Int
  var dsparkConfidenceThreshold: Double
  var defaultMaxTokens: Int
  var defaultTemperature: Double
  var defaultTopP: Double

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
      modelPath: UserDefaults.standard.string(forKey: "selectedModelPath") ?? "",
      host: "127.0.0.1",
      port: 11_434,
      apiKey: "",
      publicModel: "deepseek-v4-flash-0731",
      slots: 1_152,
      readWorkers: 4,
      powerSavingLimitGBps: nil,
      prefillStepSize: 0,
      layerMajorPrefill: true,
      promptCacheEntries: 2,
      promptCacheMemoryGiB: 8,
      warmupPromptPath: "",
      bf16KVCache: false,
      dsparkEnabled: false,
      dsparkSlots: 768,
      dsparkConfidenceThreshold: 0.6,
      defaultMaxTokens: 272_000,
      defaultTemperature: 0.2,
      defaultTopP: 0.98
    )
    if let data = defaults.data(forKey: preferenceKey),
      var saved = try? JSONDecoder().decode(ServerConfiguration.self, from: data)
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

  func save(defaults: UserDefaults = .standard) {
    var saved = self
    saved.runtimeDirectory = ""
    saved.pythonExecutable = ""
    saved.pythonHome = nil
    saved.sitePackages = nil
    saved.apiKey = ""
    guard let data = try? JSONEncoder().encode(saved) else { return }
    defaults.set(data, forKey: Self.preferenceKey)
    defaults.set(modelPath, forKey: "selectedModelPath")
  }

  var baseURL: URL? {
    let clientHost = ["0.0.0.0", "::"].contains(host) ? "127.0.0.1" : host
    let formattedHost = clientHost.contains(":") ? "[\(clientHost)]" : clientHost
    return URL(string: "http://\(formattedHost):\(port)")
  }

  var arguments: [String] {
    var values = [
      "-m", "deepseek_v4_ssd.server",
      "--model", modelPath,
      "--host", host,
      "--port", String(port),
      "--public-model", publicModel,
      "--slots", String(slots),
      "--read-workers", String(readWorkers),
      "--prefill-step-size", String(prefillStepSize),
      "--prompt-cache-entries", String(promptCacheEntries),
      "--prompt-cache-memory-gib", String(promptCacheMemoryGiB),
      "--default-max-tokens", String(defaultMaxTokens),
      "--default-temperature", String(defaultTemperature),
      "--default-top-p", String(defaultTopP),
    ]
    if let powerSavingLimitGBps {
      values += ["--power-saving-limit-gbps", String(powerSavingLimitGBps)]
    }
    if !layerMajorPrefill { values.append("--no-layer-major-prefill") }
    if !warmupPromptPath.isEmpty {
      values += ["--warmup-prompt-file", warmupPromptPath]
    }
    if bf16KVCache { values.append("--bf16-kv-cache") }
    if dsparkEnabled { values.append("--dspark") }
    values += ["--dspark-slots", String(dsparkSlots)]
    values += ["--dspark-confidence-threshold", String(dsparkConfidenceThreshold)]
    return values
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
    guard InstalledModelDiscovery.inspect(URL(fileURLWithPath: modelPath))?.isUsable == true else {
      throw ConfigurationError(L10n.string("Select a usable installed model."))
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
    guard !publicModel.isEmpty else {
      throw ConfigurationError(L10n.string("Model ID cannot be empty."))
    }
    guard slots >= 6 else {
      throw ConfigurationError(L10n.string("Slots must be at least 6."))
    }
    guard dsparkSlots >= 30 else {
      throw ConfigurationError(L10n.string("DSpark slots must be at least 30."))
    }
    guard readWorkers >= 1, prefillStepSize >= 0 else {
      throw ConfigurationError(
        L10n.string("Read workers must be greater than 0. Prefill step size must be 0 or greater."))
    }
    guard promptCacheEntries >= 1, promptCacheMemoryGiB >= 1 else {
      throw ConfigurationError(
        L10n.string("Prompt cache entries and the memory limit must be greater than 0."))
    }
    if !warmupPromptPath.isEmpty {
      guard FileManager.default.isReadableFile(atPath: warmupPromptPath) else {
        throw ConfigurationError(L10n.string("The warmup prompt file cannot be read."))
      }
    }
    guard (1...272_000).contains(defaultMaxTokens),
      (0...2).contains(defaultTemperature),
      (0.000_001...1).contains(defaultTopP),
      (0...1).contains(dsparkConfidenceThreshold)
    else {
      throw ConfigurationError(L10n.string("Correct the default generation parameters."))
    }
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

  private var process: Process?
  private var outputTask: Task<Void, Never>?
  private var monitorTask: Task<Void, Never>?
  private var monitorConfiguration: ServerConfiguration?
  private var previousSSDBytes: UInt64?
  private var previousSSDTime: ContinuousClock.Instant?
  private var lastRecordedCompletedRequestCount = 0

  var isActive: Bool {
    switch state {
    case .starting, .running, .stopping: true
    case .stopped, .failed: false
    }
  }

  func start(_ configuration: ServerConfiguration) {
    guard !isActive else { return }
    do {
      try configuration.validate()
      let process = Process()
      let output = Pipe()
      let runtimeURL = URL(fileURLWithPath: configuration.runtimeDirectory)
      process.executableURL = URL(fileURLWithPath: configuration.pythonExecutable)
      process.currentDirectoryURL = runtimeURL
      process.arguments = configuration.arguments
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
      monitorConfiguration = configuration
      readOutput(output.fileHandleForReading)
      startMonitoring()
    } catch {
      state = .failed(error.localizedDescription)
      appendLog(L10n.string("Error: %@", error.localizedDescription) + "\n")
    }
  }

  func stop() {
    guard let process, process.isRunning else {
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
        dsparkAverageAcceptedLength: status.performance.dsparkAverageAcceptedLength ?? 0
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
      performanceHistory.record(live.snapshot)
    }
    if requestCompleted {
      lastRecordedCompletedRequestCount = live.completedRequestCount
    }
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
    previousSSDBytes = nil
    previousSSDTime = nil
    performance = LivePerformance()
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
