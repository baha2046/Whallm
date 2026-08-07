import AppKit
import SwiftUI

@main
struct DeepSeekV4SSDApp: App {
  @StateObject private var server = ServerController()

  init() {
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
  }
}
