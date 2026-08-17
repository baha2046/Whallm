import AppKit
import Sparkle
import SwiftUI

@main
struct DeepSeekV4SSDApp: App {
  @StateObject private var server = ServerController()
  @AppStorage(L10n.preferenceKey) private var languageCode = AppLanguage.appDefault.rawValue
  private let updaterController: SPUStandardUpdaterController

  init() {
    updaterController = SPUStandardUpdaterController(
      startingUpdater: true,
      updaterDelegate: nil,
      userDriverDelegate: nil)
    NSApplication.shared.setActivationPolicy(.regular)
  }

  var body: some Scene {
    WindowGroup {
      ContentView(server: server) {
        updaterController.checkForUpdates(nil)
      }
      .font(.body)
      .dynamicTypeSize(.xLarge ... .accessibility5)
      .controlSize(.large)
      .frame(minWidth: 960, minHeight: 760)
      .environment(\.locale, selectedLanguage.locale)
      .onAppear { NSApplication.shared.activate() }
      .onDisappear { server.stop() }
    }
    .defaultSize(width: 1_440, height: 900)
    .windowStyle(.hiddenTitleBar)
    .commands {
      CommandGroup(after: .appInfo) {
        Button(L10n.string("Check for Updates…")) {
          updaterController.checkForUpdates(nil)
        }
        .disabled(!updaterController.updater.canCheckForUpdates)
      }
    }
  }

  private var selectedLanguage: AppLanguage {
    (AppLanguage(rawValue: languageCode) ?? .appDefault).resolved
  }
}
