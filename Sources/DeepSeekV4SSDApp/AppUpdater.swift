import Combine
import Foundation
import Sparkle

enum UpdateChannel: String, CaseIterable, Identifiable {
  case stable
  case dev

  var id: String { rawValue }
  var allowedChannels: Set<String> { self == .dev ? ["dev"] : [] }
}

@MainActor
final class AppUpdater: NSObject, ObservableObject, SPUUpdaterDelegate {
  static let channelPreferenceKey = "updateChannel"
  private let defaults: UserDefaults
  private var controller: SPUStandardUpdaterController!
  private var observations: [NSKeyValueObservation] = []

  @Published var channel: UpdateChannel {
    didSet {
      guard channel != oldValue else { return }
      defaults.set(channel.rawValue, forKey: Self.channelPreferenceKey)
      updater.resetUpdateCycleAfterShortDelay()
    }
  }
  @Published private(set) var canCheckForUpdates = false
  @Published private(set) var automaticallyChecksForUpdates = false

  var updater: SPUUpdater { controller.updater }

  init(startingUpdater: Bool = true, defaults: UserDefaults = .standard) {
    self.defaults = defaults
    channel = UpdateChannel(rawValue: defaults.string(forKey: Self.channelPreferenceKey) ?? "") ?? .stable
    super.init()
    controller = SPUStandardUpdaterController(
      startingUpdater: startingUpdater, updaterDelegate: self, userDriverDelegate: nil)
    observations = [
      updater.observe(\.canCheckForUpdates, options: [.initial, .new]) { [weak self] updater, _ in
        MainActor.assumeIsolated { self?.canCheckForUpdates = updater.canCheckForUpdates }
      },
      updater.observe(\.automaticallyChecksForUpdates, options: [.initial, .new]) { [weak self] updater, _ in
        MainActor.assumeIsolated { self?.automaticallyChecksForUpdates = updater.automaticallyChecksForUpdates }
      },
    ]
  }

  func setAutomaticallyChecksForUpdates(_ enabled: Bool) {
    updater.automaticallyChecksForUpdates = enabled
  }

  func checkForUpdates() {
    controller.checkForUpdates(nil)
  }

  func allowedChannels(for updater: SPUUpdater) -> Set<String> {
    channel.allowedChannels
  }
}
