import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ModelDiscoveryTests: XCTestCase {
  private func availableInstalledModel() -> URL? {
    let fileManager = FileManager.default
    let project = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    let candidates = [
      fileManager.homeDirectoryForCurrentUser
        .appending(path: ".dsmodel", directoryHint: .isDirectory)
        .appending(path: "deepseek-v4-flash-0731.dsv4", directoryHint: .isDirectory),
      project.appending(path: "scratch/deepseek-v4-flash-0731.dsv4"),
    ]
    return candidates.first {
      fileManager.fileExists(atPath: $0.appending(path: "manifest.json").path)
    }
  }

  func testDiscoveryAcceptsCompleteInstalledModelAndRejectsMissingModel() throws {
    guard let model = availableInstalledModel() else {
      throw XCTSkip("The complete installed model is not available.")
    }

    XCTAssertEqual(InstalledModelDiscovery.inspect(model)?.url, model.standardizedFileURL)

    let empty = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }
    XCTAssertNil(InstalledModelDiscovery.inspect(empty))
  }

  func testDiscoveryKeepsModelWithMissingFilesForRepair() throws {
    guard let source = availableInstalledModel()?.appending(path: "manifest.json") else {
      throw XCTSkip("The installed model manifest is not available.")
    }
    let model = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: model) }
    try FileManager.default.copyItem(at: source, to: model.appending(path: "manifest.json"))

    let discovered = try XCTUnwrap(InstalledModelDiscovery.inspect(model))
    XCTAssertFalse(discovered.isUsable)
    XCTAssertFalse(discovered.quickIssues.isEmpty)
  }

  func testDiscoveryReportsInvalidManifest() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let model = root.appending(path: "broken.dsv4")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data("{}".utf8).write(to: model.appending(path: "manifest.json"))

    let result = InstalledModelDiscovery.find(in: root)
    XCTAssertTrue(result.models.isEmpty)
    XCTAssertEqual(result.invalidModelURLs.map(\.lastPathComponent), ["broken.dsv4"])
  }

  func testDiscoveryRejectsModelWithoutOfficialEncoder() throws {
    guard let source = availableInstalledModel()?.appending(path: "manifest.json"),
      let data = try? Data(contentsOf: source)
    else {
      throw XCTSkip("The installed model manifest is not available.")
    }
    var manifest = try XCTUnwrap(
      JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    let files = try XCTUnwrap(manifest["files"] as? [[String: Any]])
    manifest["files"] = files.filter { $0["path"] as? String != "encoding/encoding_dsv4.py" }

    let model = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: model) }
    try JSONSerialization.data(withJSONObject: manifest).write(
      to: model.appending(path: "manifest.json")
    )

    XCTAssertNil(InstalledModelDiscovery.inspect(model))
  }

  @MainActor
  func testModelLibraryDefaultsToHiddenHomeFolder() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }

    XCTAssertEqual(
      ModelLibrary(defaults: defaults).rootURL.path,
      FileManager.default.homeDirectoryForCurrentUser.appending(path: ".dsmodel").path
    )
  }

  @MainActor
  func testModelLibraryRestoresDSparkInstallPreference() {
    let suite = "ModelDiscoveryTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: defaults)

    library.installDSparkWithModel = false

    XCTAssertFalse(ModelLibrary(defaults: defaults).installDSparkWithModel)
  }
}
