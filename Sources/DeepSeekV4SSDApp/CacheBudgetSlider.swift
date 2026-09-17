import SwiftUI

/// Continuous native track: `Slider(step:)` adds automatic tick dots on macOS.
/// Quantize in the binding instead, retaining explicit 0.1 GiB keyboard steps.
struct CacheBudgetSlider: View {
  @Binding var value: Double
  let range: ClosedRange<Double>
  @Environment(\.isEnabled) private var isEnabled

  static func adjusted(_ value: Double, by delta: Double, in range: ClosedRange<Double>) -> Double {
    ExpertMemory.sliderValue(value + delta, in: range)
  }

  var body: some View {
    Slider(value: Binding(
      get: { value },
      set: { value = ExpertMemory.sliderValue($0, in: range) }
    ), in: range)
    .onKeyPress(.leftArrow) { adjust(-0.1) }
    .onKeyPress(.rightArrow) { adjust(0.1) }
    .onKeyPress(.downArrow) { adjust(-0.1) }
    .onKeyPress(.upArrow) { adjust(0.1) }
    .accessibilityAdjustableAction { direction in
      switch direction {
      case .increment: _ = adjust(0.1)
      case .decrement: _ = adjust(-0.1)
      @unknown default: break
      }
    }
  }

  private func adjust(_ delta: Double) -> KeyPress.Result {
    guard isEnabled else { return .ignored }
    value = Self.adjusted(value, by: delta, in: range)
    return .handled
  }
}
