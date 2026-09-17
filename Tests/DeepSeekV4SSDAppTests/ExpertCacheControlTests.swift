import Foundation
import AppKit
import SwiftUI
import DeepSeekRepack
import XCTest
@testable import DeepSeekV4SSDApp

final class ExpertCacheControlTests: XCTestCase {
  @MainActor
  func testCacheSliderRendersWithoutFooterIndicators() throws {
    _ = NSApplication.shared
    let host = NSHostingView(rootView: CacheBudgetSlider(value: .constant(7.5), range: 0.1...64)
      .padding(24).frame(width: 520, height: 90))
    let window = NSWindow(contentRect: NSRect(x: -10000, y: -10000, width: 520, height: 90),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    defer { window.close() }
    host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    // Modern SwiftUI draws its slider directly; there need not be an NSSlider
    // subview. Check the rendered strip below the track rather than AppKit internals.
    let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
    host.cacheDisplay(in: host.bounds, to: bitmap)
    var maximumFooterAlpha: CGFloat = 0
    for y in Int(Double(bitmap.pixelsHigh) * 0.70)..<Int(Double(bitmap.pixelsHigh) * 0.85) {
      for x in 0..<bitmap.pixelsWide {
        maximumFooterAlpha = max(maximumFooterAlpha, try XCTUnwrap(bitmap.colorAt(x: x, y: y)).alphaComponent)
      }
    }
    // The system thumb shadow can leave a one-byte alpha tail, not a tick.
    XCTAssertLessThanOrEqual(maximumFooterAlpha, 1.0 / 255.0 + 0.000001)
    if let path = ProcessInfo.processInfo.environment["WHALLM_CAPTURE_CACHE_SLIDER"] {
      try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        .write(to: URL(fileURLWithPath: path))
    }
  }

  @MainActor
  func testTicklessSliderAdjustmentKeepsOneDecimalAndBounds() {
    let range = 0.1...64.0
    XCTAssertEqual(CacheBudgetSlider.adjusted(7.5, by: 0.1, in: range), 7.6)
    XCTAssertEqual(CacheBudgetSlider.adjusted(7.5, by: -0.1, in: range), 7.4)
    XCTAssertEqual(CacheBudgetSlider.adjusted(64, by: 0.1, in: range), 64)
    XCTAssertEqual(CacheBudgetSlider.adjusted(0.1, by: -0.1, in: range), 0.1)
  }

  func testEveryControlUsesItsOwnBudgetAndClearsItWhenSlotsAreEdited() throws {
    let blob = ExpertMemory.blobBytes(for: .qwen3_8FlashNext)
    for control in ExpertCacheControl.allCases {
      var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
      settings[keyPath: control.budgetKey] = 1.2
      let before = settings
      XCTAssertEqual(control.slots(in: settings, blobBytes: blob),
        try ExpertMemory.capacity(gib: 1.2, blobBytes: blob, minimum: 1))
      XCTAssertEqual(settings, before) // Switching presentation only reads the value.
      control.setSlots(123, in: &settings)
      XCTAssertNil(settings[keyPath: control.budgetKey])
      XCTAssertEqual(control.slots(in: settings, blobBytes: blob), 123)
      XCTAssertEqual(control.legacySlots(in: settings), 123)
      for other in ExpertCacheControl.allCases where other != control {
        XCTAssertEqual(settings[keyPath: other.budgetKey], before[keyPath: other.budgetKey])
        XCTAssertEqual(other.legacySlots(in: settings), other.legacySlots(in: before))
      }
      let decoded = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONEncoder().encode(settings))
      XCTAssertEqual(control.slots(in: decoded, blobBytes: blob), 123)
      XCTAssertNil(decoded[keyPath: control.budgetKey])
    }
  }

  func testEverySliderClampsRoundsAndReplacesOldSlotCapacity() throws {
    let blob = ExpertMemory.blobBytes(for: .qwen3_8FlashNext)
    let range = ExpertMemory.sliderRange(physicalMemory: 64 * 1_073_741_824)
    for control in ExpertCacheControl.allCases {
      var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
      control.setSlots(-1, in: &settings)
      XCTAssertThrowsError(try settings.validate(for: .qwen3_8FlashNext))
      control.setGiB(1.26, in: &settings, blobBytes: blob, range: range)
      XCTAssertEqual(settings[keyPath: control.budgetKey], 1.3)
      let expected = try ExpertMemory.capacity(gib: 1.3, blobBytes: blob, minimum: 1)
      XCTAssertEqual(control.slots(in: settings, blobBytes: blob), expected)
      XCTAssertEqual(control.legacySlots(in: settings), expected)
      XCTAssertNoThrow(try settings.validate(for: .qwen3_8FlashNext))
      control.setGiB(100, in: &settings, blobBytes: blob, range: range)
      XCTAssertEqual(settings[keyPath: control.budgetKey], 64)
      control.setGiB(0, in: &settings, blobBytes: blob, range: range)
      XCTAssertEqual(settings[keyPath: control.budgetKey], 0.1)
    }
  }

  func testPreferenceIsOffByDefaultPersistsAndDoesNotChangeModelSettings() throws {
    let name = "ExpertCacheControlTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: name))
    defer { store.removePersistentDomain(forName: name) }
    let settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    settings.save(for: .qwen3_8FlashNext, defaults: store)
    let saved = ModelAdvancedSettings.load(for: .qwen3_8FlashNext, defaults: store)
    XCTAssertFalse(store.bool(forKey: ExpertCacheControl.slotsPreferenceKey))
    store.set(true, forKey: ExpertCacheControl.slotsPreferenceKey)
    XCTAssertTrue(try XCTUnwrap(UserDefaults(suiteName: name)).bool(forKey: ExpertCacheControl.slotsPreferenceKey))
    XCTAssertEqual(ModelAdvancedSettings.load(for: .qwen3_8FlashNext, defaults: store), saved)
    store.set(false, forKey: ExpertCacheControl.slotsPreferenceKey)
    XCTAssertEqual(ModelAdvancedSettings.load(for: .qwen3_8FlashNext, defaults: store), saved)
  }

  func testPreferenceCopyIsLocalized() {
    for language in [AppLanguage.traditionalChinese, .simplifiedChinese] {
      for key in ["Edit expert caches in slots",
                  "Use integer slot inputs instead of GiB sliders for expert, MTP, and DSpark caches. Switching does not change saved capacity.",
                  "Enter the number of experts to retain. Applies on next model load."] {
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
    }
  }
}
