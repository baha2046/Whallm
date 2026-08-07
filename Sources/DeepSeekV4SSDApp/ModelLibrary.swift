import AppKit
import Combine
import DeepSeekRepack
import Foundation

struct InstalledModelInfo: Identifiable, Equatable, Sendable {
  let url: URL
  let size: UInt64
  let quickIssues: [InstalledFileIssue]

  var id: String { url.path }
  var name: String { url.deletingPathExtension().lastPathComponent }
  var isUsable: Bool { quickIssues.isEmpty }
}

struct ModelDiscoveryResult: Sendable {
  let models: [InstalledModelInfo]
  let invalidModelURLs: [URL]
}

enum InstalledModelDiscovery {
  static func find(in root: URL) -> ModelDiscoveryResult {
    let root = root.standardizedFileURL
    let children =
      (try? FileManager.default.contentsOfDirectory(
        at: root,
        includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
        options: [.skipsHiddenFiles]
      )) ?? []
    let directories = children.filter {
      let values = try? $0.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
      return values?.isDirectory == true && values?.isSymbolicLink != true
        && $0.pathExtension != "partial"
    }
    var models: [InstalledModelInfo] = []
    var invalidModelURLs: [URL] = []
    for candidate in [root] + directories
    where FileManager.default.fileExists(atPath: candidate.appending(path: "manifest.json").path) {
      if let model = inspect(candidate) {
        models.append(model)
      } else {
        invalidModelURLs.append(candidate)
      }
    }
    return ModelDiscoveryResult(
      models: models.sorted {
        $0.name.localizedStandardCompare($1.name) == .orderedAscending
      },
      invalidModelURLs: invalidModelURLs.sorted {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
      }
    )
  }

  static func inspect(_ root: URL) -> InstalledModelInfo? {
    let root = root.standardizedFileURL.resolvingSymlinksInPath()
    guard let manifest = try? InstalledModel.loadManifest(at: root) else { return nil }
    let paths = Set(manifest.files.map(\.path))
    let requiredPaths = [
      "common.bin",
      "config.json",
      "encoding/encoding_dsv4.py",
      "tokenizer/tokenizer.json",
    ]
    guard requiredPaths.allSatisfy(paths.contains) else { return nil }
    let expectedLayerSize = UInt64(manifest.expertCount) * manifest.expertBlobSize
    var totalSize: UInt64 = 0
    var issues: [InstalledFileIssue] = []
    for file in manifest.files {
      guard !file.path.hasPrefix("/"), !file.path.split(separator: "/").contains("..") else {
        return nil
      }
      let target = root.appending(path: file.path).standardizedFileURL.resolvingSymlinksInPath()
      guard target.path.hasPrefix(root.path + "/") else { return nil }
      let attributes = try? FileManager.default.attributesOfItem(atPath: target.path)
      if (attributes?[.type] as? FileAttributeType) != .typeRegular {
        issues.append(InstalledFileIssue(path: file.path, kind: .missing))
      } else if (attributes?[.size] as? NSNumber)?.uint64Value != file.size
        || (file.path.hasPrefix("experts/layer_") && file.size != expectedLayerSize)
      {
        issues.append(InstalledFileIssue(path: file.path, kind: .sizeMismatch))
      }
      let sum = totalSize.addingReportingOverflow(file.size)
      guard !sum.overflow else { return nil }
      totalSize = sum.partialValue
    }
    return InstalledModelInfo(url: root, size: totalSize, quickIssues: issues)
  }
}

enum PreflightStatus: Sendable {
  case passed
  case warning
  case failed
}

struct PreflightCheck: Identifiable, Sendable {
  let id: String
  let title: String
  let detail: String
  let status: PreflightStatus
  let blocksDownload: Bool
}

enum ModelOperationPhase: Equatable {
  case idle
  case preparingDownload
  case downloading
  case cancelling
  case verifying
  case preparingRepair
  case repairing

  var label: String {
    switch self {
    case .idle: ""
    case .preparingDownload: "正在準備下載"
    case .downloading: "正在下載並安裝模型"
    case .cancelling: "正在停止"
    case .verifying: "正在完整驗證模型"
    case .preparingRepair: "正在準備修復"
    case .repairing: "正在重新下載損壞資料"
    }
  }
}

struct ModelOperationProgress: Equatable {
  let completedBytes: UInt64
  let totalBytes: UInt64
  let bytesPerSecond: Double?
  let estimatedSecondsRemaining: Double?

  var fraction: Double? {
    guard totalBytes > 0 else { return nil }
    return min(1, Double(completedBytes) / Double(totalBytes))
  }
}

@MainActor
final class ModelLibrary: ObservableObject {
  static let rootPreference = "modelLibraryRoot"
  private static let activeDownloadPreference = "modelDownloadWasActive"
  private static let activeDestinationPreference = "modelDownloadDestination"
  nonisolated private static let minimumMemoryBytes: UInt64 = 64 * 1_024 * 1_024 * 1_024
  nonisolated private static let requiredStorageBytes: UInt64 = 160 * 1_024 * 1_024 * 1_024

  @Published private(set) var rootURL: URL
  @Published private(set) var models: [InstalledModelInfo] = []
  @Published private(set) var invalidModelURLs: [URL] = []
  @Published private(set) var preflightChecks: [PreflightCheck] = []
  @Published private(set) var isScanning = false
  @Published private(set) var operationPhase = ModelOperationPhase.idle
  @Published private(set) var operationProgress: ModelOperationProgress?
  @Published private(set) var message: String?
  @Published private(set) var verificationModelPath: String?
  @Published private(set) var verificationIssues: [InstalledFileIssue]?

  private let defaults: UserDefaults
  private var operationTask: Task<Void, Never>?
  private var downloadStart: ContinuousClock.Instant?

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    if let savedPath = defaults.string(forKey: Self.rootPreference) {
      rootURL = URL(fileURLWithPath: savedPath, isDirectory: true)
    } else {
      rootURL = FileManager.default.homeDirectoryForCurrentUser.appending(
        path: ".dsmodel", directoryHint: .isDirectory)
    }
  }

  var usableModels: [InstalledModelInfo] { models.filter(\.isUsable) }
  var damagedModels: [InstalledModelInfo] { models.filter { !$0.isUsable } }
  var isBusy: Bool { operationPhase != .idle }
  var canDownload: Bool {
    !preflightChecks.contains { $0.blocksDownload && $0.status == .failed }
  }
  var canStartDownload: Bool {
    let destination =
      hasPartialDownload
      ? partialDownloadURL.deletingPathExtension() : defaultDownloadDestination
    return canDownload && !FileManager.default.fileExists(atPath: destination.path)
  }
  var hasPartialDownload: Bool {
    FileManager.default.fileExists(atPath: partialDownloadURL.path)
  }

  func model(at path: String) -> InstalledModelInfo? {
    models.first { $0.url.path == path }
  }

  func canUseModel(at path: String) -> Bool {
    guard model(at: path)?.isUsable == true else { return false }
    return verificationModelPath != path || verificationIssues?.isEmpty != false
  }

  func setRoot(_ url: URL) async {
    rootURL = url.standardizedFileURL
    defaults.set(rootURL.path, forKey: Self.rootPreference)
    verificationModelPath = nil
    verificationIssues = nil
    await scan()
  }

  func scan() async {
    isScanning = true
    message = nil
    do {
      try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
      let root = rootURL
      let result = await Task.detached(priority: .utility) {
        InstalledModelDiscovery.find(in: root)
      }.value
      models = result.models
      invalidModelURLs = result.invalidModelURLs
      refreshPreflight()
      if models.isEmpty && invalidModelURLs.isEmpty && !hasPartialDownload {
        message = "尚未安裝模型。請下載模型或選擇其他資料夾。"
      }
    } catch {
      models = []
      invalidModelURLs = []
      refreshPreflight()
      message = "無法讀取模型資料夾。請選擇可寫入的資料夾。"
    }
    isScanning = false
  }

  func refreshPreflight() {
    preflightChecks = Self.makePreflightChecks(root: rootURL, partial: partialDownloadURL)
  }

  func resumeDownloadIfNeeded() {
    guard defaults.bool(forKey: Self.activeDownloadPreference), !isBusy else { return }
    let saved = defaults.string(forKey: Self.activeDestinationPreference)
    let destination =
      saved.map { URL(fileURLWithPath: $0, isDirectory: true) }
      ?? defaultDownloadDestination
    guard FileManager.default.fileExists(atPath: destination.appendingPathExtension("partial").path)
    else {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      return
    }
    startDownload(to: destination)
  }

  func startDownload(to destination: URL? = nil) {
    guard !isBusy else { return }
    refreshPreflight()
    guard canDownload else {
      message = "請先修正下載前檢查項目。"
      return
    }
    let destination =
      destination
      ?? (hasPartialDownload
        ? partialDownloadURL.deletingPathExtension() : defaultDownloadDestination)
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      message = "此位置已有模型。請先驗證並修復現有模型。"
      return
    }
    defaults.set(true, forKey: Self.activeDownloadPreference)
    defaults.set(destination.path, forKey: Self.activeDestinationPreference)
    operationPhase = .preparingDownload
    operationProgress = nil
    downloadStart = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performDownload(to: destination)
    }
  }

  func startVerification(_ model: InstalledModelInfo) {
    guard !isBusy else { return }
    operationPhase = .verifying
    operationProgress = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performVerification(model.url)
    }
  }

  func startRepair(_ model: InstalledModelInfo) {
    guard !isBusy else { return }
    defaults.set(true, forKey: Self.activeDownloadPreference)
    defaults.set(model.url.path, forKey: Self.activeDestinationPreference)
    operationPhase = .verifying
    operationProgress = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performRepair(model.url)
    }
  }

  func reinstall(_ url: URL) {
    guard !isBusy else { return }
    refreshPreflight()
    guard canDownload else {
      message = "請先修正下載前檢查項目。"
      return
    }
    do {
      var trashedURL: NSURL?
      try FileManager.default.trashItem(at: url, resultingItemURL: &trashedURL)
      startDownload(to: url)
    } catch {
      message = "無法將損壞的模型移到垃圾桶。請檢查資料夾權限。"
    }
  }

  func cancelOperation() {
    guard isBusy else { return }
    defaults.set(false, forKey: Self.activeDownloadPreference)
    operationPhase = .cancelling
    operationTask?.cancel()
  }

  func reveal(_ url: URL) {
    NSWorkspace.shared.activateFileViewerSelecting([url])
  }

  private var defaultDownloadDestination: URL {
    rootURL.appending(path: "deepseek-v4-flash-0731.dsv4", directoryHint: .isDirectory)
  }

  private var partialDownloadURL: URL {
    let saved = defaults.string(forKey: Self.activeDestinationPreference)
    let destination =
      saved.map { URL(fileURLWithPath: $0, isDirectory: true) }
      ?? defaultDownloadDestination
    return destination.appendingPathExtension("partial")
  }

  private func performDownload(to destination: URL) async {
    do {
      _ = try await DeepSeekV4Checkpoint().repack(to: destination) { [weak self] progress in
        Task { @MainActor in self?.updateRepackProgress(progress, phase: .downloading) }
      }
      try Task.checkCancellation()
      let verification = try await audit(destination)
      verificationModelPath = destination.path
      verificationIssues = verification.issues
      let resultMessage =
        verification.isValid
        ? "模型已安裝並通過完整驗證。"
        : "模型安裝完成，但有 \(verification.issues.count) 個檔案無法通過驗證。"
      defaults.set(false, forKey: Self.activeDownloadPreference)
      await scan()
      message = resultMessage
    } catch is CancellationError {
      message = "下載已停止。App 已保留進度。"
    } catch {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      message = "無法下載模型。請檢查網路後再試一次。\n\(String(describing: error))"
    }
    finishOperation()
  }

  private func performVerification(_ url: URL) async {
    do {
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      message =
        verification.isValid
        ? "模型已通過完整驗證。"
        : "模型有 \(verification.issues.count) 個檔案需要修復。"
    } catch is CancellationError {
      message = "驗證已停止。"
    } catch {
      message = "無法驗證模型。\(String(describing: error))"
    }
    finishOperation()
  }

  private func performRepair(_ url: URL) async {
    do {
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      guard !verification.isValid else {
        message = "模型不需要修復。"
        defaults.set(false, forKey: Self.activeDownloadPreference)
        finishOperation()
        return
      }
      operationPhase = .preparingRepair
      operationProgress = nil
      downloadStart = nil
      _ = try await DeepSeekV4Checkpoint().repair(
        at: url,
        invalidFiles: Set(verification.issues.map(\.path))
      ) { [weak self] progress in
        Task { @MainActor in self?.updateRepackProgress(progress, phase: .repairing) }
      }
      try Task.checkCancellation()
      let repaired = try await audit(url)
      verificationIssues = repaired.issues
      let resultMessage =
        repaired.isValid
        ? "模型已修復並通過完整驗證。"
        : "仍有 \(repaired.issues.count) 個檔案需要修復。"
      defaults.set(false, forKey: Self.activeDownloadPreference)
      await scan()
      message = resultMessage
    } catch is CancellationError {
      await scan()
      message = "修復已停止。App 已保留進度。"
    } catch {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      message = "無法修復模型。請檢查網路與儲存空間。\n\(String(describing: error))"
      await scan()
    }
    finishOperation()
  }

  private func audit(_ url: URL) async throws -> InstalledModelVerification {
    operationPhase = .verifying
    operationProgress = nil
    let worker = Task.detached(priority: .utility) { [weak self] in
      try InstalledModel.audit(at: url) { progress in
        Task { @MainActor in
          self?.operationProgress = ModelOperationProgress(
            completedBytes: progress.checkedBytes,
            totalBytes: progress.totalBytes,
            bytesPerSecond: nil,
            estimatedSecondsRemaining: nil
          )
        }
      }
    }
    return try await withTaskCancellationHandler {
      try await worker.value
    } onCancel: {
      worker.cancel()
    }
  }

  private func updateRepackProgress(_ progress: RepackProgress, phase: ModelOperationPhase) {
    operationPhase = phase
    let now = ContinuousClock.now
    if progress.downloadedBytes > 0, downloadStart == nil { downloadStart = now }
    let speed = downloadStart.map {
      let elapsed = Self.seconds(from: $0.duration(to: now))
      return elapsed > 0 ? Double(progress.downloadedBytes) / elapsed : 0
    }
    let remaining =
      progress.totalBytes > progress.copiedBytes
      ? progress.totalBytes - progress.copiedBytes : 0
    operationProgress = ModelOperationProgress(
      completedBytes: progress.copiedBytes,
      totalBytes: progress.totalBytes,
      bytesPerSecond: speed,
      estimatedSecondsRemaining: speed.flatMap { $0 > 0 ? Double(remaining) / $0 : nil }
    )
  }

  private func finishOperation() {
    operationTask = nil
    operationPhase = .idle
    operationProgress = nil
    downloadStart = nil
    refreshPreflight()
  }

  nonisolated private static func makePreflightChecks(root: URL, partial: URL) -> [PreflightCheck] {
    let fileManager = FileManager.default
    try? fileManager.createDirectory(at: root, withIntermediateDirectories: true)

    #if arch(arm64)
      let architecture = PreflightCheck(
        id: "architecture", title: "Apple Silicon", detail: "此 Mac 使用 Apple Silicon。",
        status: .passed, blocksDownload: true)
    #else
      let architecture = PreflightCheck(
        id: "architecture", title: "Apple Silicon", detail: "此 runtime 不支援 Intel Mac。",
        status: .failed, blocksDownload: true)
    #endif

    let memory = ProcessInfo.processInfo.physicalMemory
    let memoryCheck = PreflightCheck(
      id: "memory",
      title: "記憶體",
      detail: memory >= minimumMemoryBytes
        ? "此 Mac 有至少 64 GiB 記憶體。"
        : "此 runtime 需要至少 64 GiB 記憶體。",
      status: memory >= minimumMemoryBytes ? .passed : .failed,
      blocksDownload: true
    )

    let probe = root.appending(path: ".write-check-\(UUID().uuidString)")
    let writable: Bool
    do {
      try Data().write(to: probe, options: .atomic)
      try fileManager.removeItem(at: probe)
      writable = true
    } catch {
      writable = false
    }
    let writableCheck = PreflightCheck(
      id: "writable",
      title: "模型資料夾",
      detail: writable ? "App 可以寫入 \(root.path)。" : "App 無法寫入此資料夾。請選擇其他資料夾。",
      status: writable ? .passed : .failed,
      blocksDownload: true
    )

    let allocated = allocatedBytes(at: partial)
    let required = requiredStorageBytes > allocated ? requiredStorageBytes - allocated : 0
    let available =
      (try? root.resourceValues(
        forKeys: [.volumeAvailableCapacityForImportantUsageKey]
      ).volumeAvailableCapacityForImportantUsage) ?? nil
    let hasStorage = available.map { $0 >= 0 && UInt64($0) >= required } ?? false
    let storageCheck = PreflightCheck(
      id: "storage",
      title: "儲存空間",
      detail: hasStorage
        ? "可用空間足以完成安裝。"
        : "此資料夾所在磁碟需要至少 \(formattedBytes(required)) 可用空間。",
      status: hasStorage ? .passed : .failed,
      blocksDownload: true
    )

    let isInternal =
      (try? root.resourceValues(forKeys: [.volumeIsInternalKey]).volumeIsInternal)
      ?? nil
    let storageTypeCheck = PreflightCheck(
      id: "ssd",
      title: "SSD",
      detail: isInternal == false
        ? "請確認外接磁碟是高速 SSD。慢速磁碟會降低生成速度。"
        : "建議使用高速 SSD。",
      status: isInternal == false ? .warning : .passed,
      blocksDownload: false
    )
    return [architecture, memoryCheck, writableCheck, storageCheck, storageTypeCheck]
  }

  nonisolated private static func allocatedBytes(at root: URL) -> UInt64 {
    guard
      let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey],
        options: [.skipsHiddenFiles]
      )
    else { return 0 }
    var total: UInt64 = 0
    for case let url as URL in enumerator {
      let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey])
      guard values?.isRegularFile == true, let size = values?.totalFileAllocatedSize else {
        continue
      }
      let sum = total.addingReportingOverflow(UInt64(size))
      if sum.overflow { return UInt64.max }
      total = sum.partialValue
    }
    return total
  }

  nonisolated private static func formattedBytes(_ bytes: UInt64) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .file)
  }

  nonisolated private static func seconds(from duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }
}
