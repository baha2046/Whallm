import Foundation
import XCTest

@testable import DeepSeekV4SSDApp

final class ModelDiscoveryTests: XCTestCase {
  func testDiscoveryAcceptsCompleteInstalledModelAndRejectsMissingModel() throws {
    let project = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let model = project.appending(path: "scratch/deepseek-v4-flash-0731.dsv4")
    guard FileManager.default.fileExists(atPath: model.appending(path: "manifest.json").path) else {
      throw XCTSkip("The complete installed model is not available.")
    }

    XCTAssertEqual(InstalledModelDiscovery.inspect(model)?.url, model.standardizedFileURL)

    let empty = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }
    XCTAssertNil(InstalledModelDiscovery.inspect(empty))
  }

  func testDiscoveryKeepsModelWithMissingFilesForRepair() throws {
    let project = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let source = project.appending(path: "scratch/deepseek-v4-flash-0731.dsv4/manifest.json")
    guard FileManager.default.fileExists(atPath: source.path) else {
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
}
