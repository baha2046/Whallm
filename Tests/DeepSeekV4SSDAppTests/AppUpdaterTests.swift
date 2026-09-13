import Foundation
import Sparkle
import XCTest

@testable import DeepSeekV4SSDApp

@MainActor
final class AppUpdaterTests: XCTestCase {
  func testChannelSelectionPersistsAndOnlyDevOptsIntoPrereleases() throws {
    let suite = "WhallmUpdateTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let app = AppUpdater(startingUpdater: false, defaults: defaults)
    XCTAssertEqual(app.channel, .stable)
    XCTAssertEqual(app.allowedChannels(for: app.updater), [])
    app.channel = .dev
    XCTAssertEqual(app.allowedChannels(for: app.updater), ["dev"])
    let restored = AppUpdater(startingUpdater: false, defaults: defaults)
    XCTAssertEqual(restored.channel, .dev)
    restored.channel = .stable
    XCTAssertEqual(restored.allowedChannels(for: restored.updater), [])
    defaults.set("unknown", forKey: AppUpdater.channelPreferenceKey)
    XCTAssertEqual(AppUpdater(startingUpdater: false, defaults: defaults).channel, .stable)
  }

  func testAutomaticChecksUseSparklesPreferenceAndObserveExternalChanges() {
    let app = AppUpdater(startingUpdater: false)
    let original = UserDefaults.standard.object(forKey: "SUEnableAutomaticChecks")
    defer {
      app.setAutomaticallyChecksForUpdates(original as? Bool ?? false)
      if let original { UserDefaults.standard.set(original, forKey: "SUEnableAutomaticChecks") }
      else { UserDefaults.standard.removeObject(forKey: "SUEnableAutomaticChecks") }
    }
    app.setAutomaticallyChecksForUpdates(true)
    XCTAssertTrue(app.updater.automaticallyChecksForUpdates)
    XCTAssertTrue(app.automaticallyChecksForUpdates)
    app.updater.automaticallyChecksForUpdates = false
    XCTAssertFalse(app.automaticallyChecksForUpdates)
    XCTAssertFalse(UserDefaults.standard.bool(forKey: "SUEnableAutomaticChecks"))
  }

  func testDevBuildsSortBeforeTheirStableRelease() {
    let comparator = SUStandardVersionComparator.default
    let versions = ["1.1.6", "1.1.7d1", "1.1.7d2", "1.1.7d10", "1.1.7", "1.1.8d1"]
    for (older, newer) in zip(versions, versions.dropFirst()) {
      XCTAssertEqual(comparator.compareVersion(older, toVersion: newer), .orderedAscending)
    }
  }

  func testUpdateSettingsAreLocalized() {
    for language in [AppLanguage.simplifiedChinese, .traditionalChinese] {
      for key in ["Update source", "Automatically check for updates",
                  "Stable includes official releases. Dev also includes test versions.",
                  "Check for new versions in the background."] {
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
    }
  }
}
