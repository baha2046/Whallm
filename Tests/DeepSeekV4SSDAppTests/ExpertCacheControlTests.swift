import Foundation
import DeepSeekRepack
import XCTest
@testable import DeepSeekV4SSDApp

final class ExpertCacheControlTests: XCTestCase {
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
