import AppKit
import Combine
import DeepSeekRepack
import Foundation

struct InstalledModelInfo: Identifiable, Equatable, Sendable {
  let url: URL
  let size: UInt64
  let quickIssues: [InstalledFileIssue]
  let hasDSpark: Bool

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
    return InstalledModelInfo(
      url: root,
      size: totalSize,
      quickIssues: issues,
      hasDSpark: manifest.dspark != nil
    )
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
  case installingDSpark

  var label: String {
    switch self {
    case .idle: ""
    case .preparingDownload: L10n.string("Preparing download")
    case .downloading: L10n.string("Downloading and installing the model")
    case .cancelling: L10n.string("Stopping")
    case .verifying: L10n.string("Verifying the complete model")
    case .preparingRepair: L10n.string("Preparing repair")
    case .repairing: L10n.string("Downloading damaged data again")
    case .installingDSpark: L10n.string("Installing DSpark")
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
  @Published var installDSparkWithModel = true

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
        message = L10n.string("No model is installed. Download a model or select another folder.")
      }
    } catch {
      models = []
      invalidModelURLs = []
      refreshPreflight()
      message = L10n.string("The model folder cannot be read. Select a writable folder.")
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
      message = L10n.string("Correct the failed download checks first.")
      return
    }
    let destination =
      destination
      ?? (hasPartialDownload
        ? partialDownloadURL.deletingPathExtension() : defaultDownloadDestination)
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      message = L10n.string(
        "A model already exists in this location. Verify and repair the existing model first.")
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

  func startDSparkInstallation(_ model: InstalledModelInfo) {
    guard !isBusy, !model.hasDSpark else { return }
    operationPhase = .installingDSpark
    operationProgress = nil
    downloadStart = nil
    message = nil
    operationTask = Task { [weak self] in
      await self?.performDSparkInstallation(model.url)
    }
  }

  func removeDSpark(_ model: InstalledModelInfo) {
    guard !isBusy, model.hasDSpark else { return }
    do {
      _ = try InstalledModel.removeDSpark(at: model.url)
      Task { [weak self] in
        await self?.scan()
        self?.message = L10n.string("DSpark was removed.")
      }
    } catch {
      message = L10n.string("DSpark could not be removed. %@", String(describing: error))
    }
  }

  func reinstall(_ url: URL) {
    guard !isBusy else { return }
    refreshPreflight()
    guard canDownload else {
      message = L10n.string("Correct the failed download checks first.")
      return
    }
    do {
      var trashedURL: NSURL?
      try FileManager.default.trashItem(at: url, resultingItemURL: &trashedURL)
      startDownload(to: url)
    } catch {
      message = L10n.string(
        "The damaged model could not be moved to Trash. Check the folder permissions.")
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
      _ = try await DeepSeekV4Checkpoint().repack(
        to: destination,
        includeDSpark: installDSparkWithModel
      ) { [weak self] progress in
        Task { @MainActor in self?.updateRepackProgress(progress, phase: .downloading) }
      }
      try Task.checkCancellation()
      let verification = try await audit(destination)
      verificationModelPath = destination.path
      verificationIssues = verification.issues
      let resultMessage =
        verification.isValid
        ? L10n.string("The model is installed and passed complete verification.")
        : L10n.string(
          "The model is installed, but %lld files failed verification.",
          Int64(verification.issues.count))
      defaults.set(false, forKey: Self.activeDownloadPreference)
      await scan()
      message = resultMessage
    } catch is CancellationError {
      message = L10n.string("The download stopped. The app kept the progress.")
    } catch {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      message = L10n.string(
        "The model could not be downloaded. Check the network and try again.\n%@",
        String(describing: error))
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
        ? L10n.string("The model passed complete verification.")
        : L10n.string("%lld model files need repair.", Int64(verification.issues.count))
    } catch is CancellationError {
      message = L10n.string("Verification stopped.")
    } catch {
      message = L10n.string("The model could not be verified. %@", String(describing: error))
    }
    finishOperation()
  }

  private func performRepair(_ url: URL) async {
    do {
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      guard !verification.isValid else {
        message = L10n.string("The model does not need repair.")
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
        ? L10n.string("The model was repaired and passed complete verification.")
        : L10n.string("%lld files still need repair.", Int64(repaired.issues.count))
      defaults.set(false, forKey: Self.activeDownloadPreference)
      await scan()
      message = resultMessage
    } catch is CancellationError {
      await scan()
      message = L10n.string("Repair stopped. The app kept the progress.")
    } catch {
      defaults.set(false, forKey: Self.activeDownloadPreference)
      message = L10n.string(
        "The model could not be repaired. Check the network and storage.\n%@",
        String(describing: error))
      await scan()
    }
    finishOperation()
  }

  private func performDSparkInstallation(_ url: URL) async {
    do {
      _ = try await DeepSeekV4Checkpoint().installDSpark(at: url) { [weak self] progress in
        Task { @MainActor in
          self?.updateRepackProgress(progress, phase: .installingDSpark)
        }
      }
      try Task.checkCancellation()
      let verification = try await audit(url)
      verificationModelPath = url.path
      verificationIssues = verification.issues
      await scan()
      message =
        verification.isValid
        ? L10n.string("DSpark is installed and ready.")
        : L10n.string("DSpark installation did not pass verification.")
    } catch is CancellationError {
      await scan()
      message = L10n.string("DSpark installation stopped. The app kept the progress.")
    } catch {
      await scan()
      message = L10n.string(
        "DSpark could not be installed. Check the network and storage.\n%@",
        String(describing: error)
      )
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
        id: "architecture", title: L10n.string("Apple Silicon"),
        detail: L10n.string("This Mac uses Apple Silicon."),
        status: .passed, blocksDownload: true)
    #else
      let architecture = PreflightCheck(
        id: "architecture", title: L10n.string("Apple Silicon"),
        detail: L10n.string("This runtime does not support Intel Mac."),
        status: .failed, blocksDownload: true)
    #endif

    let memory = ProcessInfo.processInfo.physicalMemory
    let memoryCheck = PreflightCheck(
      id: "memory",
      title: L10n.string("Memory"),
      detail: memory >= minimumMemoryBytes
        ? L10n.string("This Mac has at least 64 GiB of memory.")
        : L10n.string("This runtime needs at least 64 GiB of memory."),
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
      title: L10n.string("Model folder"),
      detail: writable
        ? L10n.string("The app can write to %@.", root.path)
        : L10n.string("The app cannot write to this folder. Select another folder."),
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
      title: L10n.string("Storage"),
      detail: hasStorage
        ? L10n.string("There is enough free space to complete installation.")
        : L10n.string(
          "The disk for this folder needs at least %@ of free space.", formattedBytes(required)),
      status: hasStorage ? .passed : .failed,
      blocksDownload: true
    )

    let isInternal =
      (try? root.resourceValues(forKeys: [.volumeIsInternalKey]).volumeIsInternal)
      ?? nil
    let storageTypeCheck = PreflightCheck(
      id: "ssd",
      title: L10n.string("SSD"),
      detail: isInternal == false
        ? L10n.string(
          "Make sure that the external disk is a high-speed SSD. A slow disk reduces generation speed."
        )
        : L10n.string("Use a high-speed SSD."),
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
