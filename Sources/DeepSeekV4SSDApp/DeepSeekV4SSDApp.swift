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
        .frame(minWidth: 820, minHeight: 620)
        .onAppear { NSApplication.shared.activate() }
        .onDisappear { server.stop() }
    }
    .defaultSize(width: 1_020, height: 760)
  }
}
