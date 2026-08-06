import Darwin
import Foundation

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
    let generationTokens: Int
    let tokensPerSecond: Double
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

struct LivePerformance: Equatable {
  var hasStatus = false
  var generating = false
  var generationTokens = 0
  var tokensPerSecond = 0.0
  var memoryBytes: UInt64 = 0
  var ssdBytesPerSecond = 0.0
  var cacheHitRate = 0.0
  var cacheResidentSlots = 0
  var cacheCapacitySlots = 0
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

struct ServerConfiguration {
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
  var prefillStepSize: Int
  var bf16KVCache: Bool
  var defaultMaxTokens: Int
  var defaultTemperature: Double
  var defaultTopP: Double

  static var localDefault: ServerConfiguration {
    let runtime = RuntimeEnvironment.current
    return ServerConfiguration(
      runtimeDirectory: runtime.runtimeDirectory.path,
      pythonExecutable: runtime.pythonExecutable.path,
      pythonHome: runtime.pythonHome?.path,
      sitePackages: runtime.sitePackages?.path,
      modelPath: UserDefaults.standard.string(forKey: "selectedModelPath") ?? "",
      host: "127.0.0.1",
      port: 8000,
      apiKey: "",
      publicModel: "deepseek-v4-flash-0731",
      slots: 1_024,
      readWorkers: 4,
      prefillStepSize: 32,
      bf16KVCache: false,
      defaultMaxTokens: 32,
      defaultTemperature: 0,
      defaultTopP: 1
    )
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
      "--default-max-tokens", String(defaultMaxTokens),
      "--default-temperature", String(defaultTemperature),
      "--default-top-p", String(defaultTopP),
    ]
    if bf16KVCache { values.append("--bf16-kv-cache") }
    return values
  }

  func validate() throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: runtimeDirectory, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw ConfigurationError("App 缺少 runtime。請重新安裝 App。")
    }
    guard FileManager.default.isExecutableFile(atPath: pythonExecutable) else {
      throw ConfigurationError("App 缺少 Python。請重新安裝 App。")
    }
    guard
      FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: runtimeDirectory)
          .appending(path: "deepseek_v4_ssd/server.py").path
      )
    else {
      throw ConfigurationError("App 的 runtime 不完整。請重新安裝 App。")
    }
    guard InstalledModelDiscovery.inspect(URL(fileURLWithPath: modelPath))?.isUsable == true else {
      throw ConfigurationError("請選擇可使用的 installed model。")
    }
    guard !host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw ConfigurationError("Host 不可空白。")
    }
    guard (1...65_535).contains(port) else {
      throw ConfigurationError("Port 必須介於 1 和 65535。")
    }
    guard ["127.0.0.1", "::1", "localhost"].contains(host) || !apiKey.isEmpty else {
      throw ConfigurationError("非本機 Host 必須設定 API key。")
    }
    guard !publicModel.isEmpty else {
      throw ConfigurationError("Model ID 不可空白。")
    }
    guard slots >= 6 else {
      throw ConfigurationError("Slot 數量至少需要 6。")
    }
    guard readWorkers >= 1, prefillStepSize >= 1 else {
      throw ConfigurationError("Read workers 和 prefill step size 必須大於 0。")
    }
    guard (1...32_768).contains(defaultMaxTokens),
      (0...2).contains(defaultTemperature),
      (0.000_001...1).contains(defaultTopP)
    else {
      throw ConfigurationError("請修正預設生成參數。")
    }
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
      case .stopped: "已停止"
      case .starting: "正在啟動"
      case .running: "執行中"
      case .stopping: "正在停止"
      case .failed: "啟動失敗"
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

  private var process: Process?
  private var outputTask: Task<Void, Never>?
  private var monitorTask: Task<Void, Never>?
  private var monitorConfiguration: ServerConfiguration?
  private var previousSSDBytes: UInt64?
  private var previousSSDTime: ContinuousClock.Instant?

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
      appendLog("錯誤：\(error.localizedDescription)\n")
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
    performance.memoryBytes = memoryBytes
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
      performance = LivePerformance(
        hasStatus: true,
        generating: status.performance.generating,
        generationTokens: status.performance.generationTokens,
        tokensPerSecond: status.performance.tokensPerSecond,
        memoryBytes: memoryBytes,
        ssdBytesPerSecond: bytesPerSecond,
        cacheHitRate: cache.hitRate,
        cacheResidentSlots: cache.residentSlots,
        cacheCapacitySlots: cache.capacitySlots
      )
    } catch {
      return
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
      let message = "Server 已停止，結束碼為 \(status)。"
      state = .failed(message)
      appendLog("\n\(message)\n")
    }
  }
}
