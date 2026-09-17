import SwiftUI
import DeepSeekRepack

struct ModelSettingsResetConfirmation {
  enum Stage { case idle, first, final }
  private(set) var stage: Stage = .idle
  private var target: ModelKind?

  mutating func begin(for modelKind: ModelKind, locked: Bool) {
    guard !locked else { cancel(); return }
    target = modelKind
    stage = .first
  }

  mutating func advance() {
    guard stage == .first else { return }
    stage = .final
  }

  mutating func cancel() {
    stage = .idle
    target = nil
  }

  mutating func confirm(for modelKind: ModelKind, locked: Bool) -> ModelAdvancedSettings? {
    defer { cancel() }
    guard !locked, stage == .final, target == modelKind else { return nil }
    return .defaults(for: modelKind)
  }
}

struct ModelSettingsResetRow: View {
  @Binding var settings: ModelAdvancedSettings
  let modelKind: ModelKind
  let settingsLocked: Bool
  let language: AppLanguage
  @State private var confirmation = ModelSettingsResetConfirmation()

  private func text(_ key: String) -> String { L10n.string(key, language: language) }

  var body: some View {
    SettingRow(
      "Default settings",
      hint: text("Restore all advanced settings for this model, including generation settings. Alias and model files are kept."),
      language: language
    ) {
      Button(text("Restore defaults"), role: .destructive) {
        confirmation.begin(for: modelKind, locked: settingsLocked)
      }
      .buttonStyle(.bordered)
      .tint(.red)
      .foregroundStyle(settingsLocked ? Color.secondary : Color.red)
      .disabled(settingsLocked)
    }
    .sheet(isPresented: Binding(
      get: { confirmation.stage != .idle },
      set: { if !$0 { confirmation.cancel() } }
    )) {
      VStack(alignment: .leading, spacing: 16) {
        Text(text(confirmation.stage == .first
          ? "Restore this model's defaults?" : "Confirm restoring defaults"))
          .font(.headline)
          .accessibilityAddTraits(.isHeader)
        Text(modelKind.displayName).font(.subheadline)
        Text(text(confirmation.stage == .first
          ? "Restore all advanced settings for this model, including generation settings. Alias and model files are kept."
          : "Your custom advanced settings will be replaced and saved immediately. This cannot be undone."))
          .fixedSize(horizontal: false, vertical: true)
        HStack {
          Spacer()
          Button(text("Cancel"), role: .cancel) { confirmation.cancel() }
            .keyboardShortcut(.cancelAction)
          if confirmation.stage == .first {
            Button(text("Continue")) { confirmation.advance() }
          } else {
            Button(text("Restore defaults"), role: .destructive) {
              if let defaults = confirmation.confirm(for: modelKind, locked: settingsLocked) {
                settings = defaults
              }
            }
            .tint(.red)
            .foregroundStyle(settingsLocked ? Color.secondary : Color.red)
            .disabled(settingsLocked)
          }
        }
      }
      .padding(24)
      .frame(width: 440)
    }
    .onChange(of: settingsLocked) { _, locked in
      if locked { confirmation.cancel() }
    }
    .onChange(of: modelKind) { _, _ in confirmation.cancel() }
  }
}
