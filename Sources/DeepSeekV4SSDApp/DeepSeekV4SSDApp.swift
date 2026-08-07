import AppKit
import Sparkle
import SwiftUI

@main
struct DeepSeekV4SSDApp: App {
  @StateObject private var server = ServerController()
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
      ContentView(server: server)
        .frame(minWidth: 980, minHeight: 680)
        .onAppear { NSApplication.shared.activate() }
        .onDisappear { server.stop() }
    }
    .defaultSize(width: 1_280, height: 800)
    .commands {
      CommandGroup(after: .appInfo) {
        Button("檢查更新…") {
          updaterController.checkForUpdates(nil)
        }
        .disabled(!updaterController.updater.canCheckForUpdates)
      }
    }
  }
}
