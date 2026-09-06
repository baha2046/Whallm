import AppKit
import Sparkle
import SwiftUI

@main
struct DeepSeekV4SSDApp: App {
  @StateObject private var server = ServerController()
  @AppStorage(L10n.preferenceKey) private var languageCode = AppLanguage.appDefault.rawValue
  private let updaterController: SPUStandardUpdaterController
  private let verifiesLocalizations: Bool

  init() {
    verifiesLocalizations = CommandLine.arguments.contains("--verify-localizations")
    updaterController = SPUStandardUpdaterController(
      startingUpdater: !verifiesLocalizations,
      updaterDelegate: nil,
      userDriverDelegate: nil)
    if verifiesLocalizations {
      // A directly launched secondary process may never present a window.
      // Complete the real lookup in App initialization, not in a view callback.
      let language = L10n.selectedLanguage
      _ = L10n.string("Server", language: language)
      let marker = "WHALLM_LOCALIZATION_READY:\(language.rawValue)\n"
      FileHandle.standardOutput.write(Data(marker.utf8))
    }
    NSApplication.shared.setActivationPolicy(.regular)
  }

  var body: some Scene {
    WindowGroup {
      Group {
        if verifiesLocalizations {
          // Exercise real L10n lookup without constructing views that read Keychain.
          Text(L10n.string("Server", language: selectedLanguage))
        } else {
          ContentView(server: server) {
            updaterController.checkForUpdates(nil)
          }
        }
      }
      .font(.body)
      .dynamicTypeSize(.xLarge ... .accessibility5)
      .controlSize(.large)
      .frame(minWidth: 640, minHeight: 320)
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
