import AppKit
import SwiftUI

struct ContentView: View {
  @ObservedObject var server: ServerController
  let checkForUpdates: () -> Void
  @StateObject private var modelLibrary = ModelLibrary()
  @State private var configuration = ServerConfiguration.localDefault
  @AppStorage("selectedAppPage") private var selectedPage = AppPage.server
  @AppStorage(L10n.preferenceKey) private var languageCode = AppLanguage.appDefault.rawValue

  var body: some View {
    NavigationSplitView {
      List(selection: $selectedPage) {
        Section {
          ForEach(AppPage.primaryPages) { page in
            Label(page.title(language: selectedLanguage), systemImage: page.icon)
              .padding(.vertical, 6)
              .tag(page)
          }
        }
        Section(L10n.string("General", language: selectedLanguage)) {
          Label(
            AppPage.settings.title(language: selectedLanguage),
            systemImage: AppPage.settings.icon
          )
          .padding(.vertical, 6)
          .tag(AppPage.settings)
        }
      }
      .listStyle(.sidebar)
      .scrollContentBackground(.hidden)
      .background(AppTheme.sidebarBackground)
      .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 250)
    } detail: {
      VStack(spacing: 0) {
        HStack {
          Text(selectedPage.title(language: selectedLanguage))
            .font(.title2.bold())
          Spacer()
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 16)

        Divider()

        ZStack {
          ServerView(
            configuration: $configuration,
            server: server,
            modelLibrary: modelLibrary,
            language: selectedLanguage
          )
          .pageVisibility(selectedPage == .server)

          AdvancedView(
            configuration: $configuration,
            serverActive: server.isActive,
            dsparkAvailable: selectedModel?.hasDSpark == true,
            language: selectedLanguage
          )
          .pageVisibility(selectedPage == .advanced)

          ChatView(configuration: configuration, server: server, language: selectedLanguage)
            .pageVisibility(selectedPage == .chat)

          MetricView(
            model: configuration.publicModel,
            state: server.state,
            performance: server.performance,
            history: server.performanceHistory,
            language: selectedLanguage,
            clearHistory: server.clearPerformanceHistory
          )
          .pageVisibility(selectedPage == .metric)

          LogsView(server: server, language: selectedLanguage)
            .pageVisibility(selectedPage == .logs)

          SettingsView(
            languageCode: $languageCode,
            language: selectedLanguage,
            checkForUpdates: checkForUpdates
          )
          .pageVisibility(selectedPage == .settings)
        }
      }
      .background(AppTheme.pageBackground)
    }
    .navigationSplitViewStyle(.balanced)
    .background(AppTheme.pageBackground)
    .preferredColorScheme(.dark)
    .environment(\.locale, selectedLanguage.locale)
    .task {
      await modelLibrary.scan()
      selectDetectedModel()
      modelLibrary.resumeDownloadIfNeeded()
    }
    .onChange(of: modelLibrary.models) { selectDetectedModel() }
    .onChange(of: configuration) {
      configuration.save()
      AppKeychain.saveAPIKey(configuration.apiKey)
    }
    .onChange(of: languageCode) { modelLibrary.refreshPreflight() }
  }

  private var selectedLanguage: AppLanguage {
    (AppLanguage(rawValue: languageCode) ?? .appDefault).resolved
  }

  private var selectedModel: InstalledModelInfo? {
    modelLibrary.usableModels.first { $0.url.path == configuration.modelPath }
  }

  private func selectDetectedModel() {
    guard
      !modelLibrary.usableModels.contains(where: { $0.url.path == configuration.modelPath })
    else {
      return
    }
    configuration.modelPath = modelLibrary.usableModels.first?.url.path ?? ""
  }
}

private enum AppPage: String, CaseIterable, Identifiable {
  case server
  case advanced
  case chat
  case metric
  case logs
  case settings

  var id: String { rawValue }

  static let primaryPages: [AppPage] = [.server, .advanced, .chat, .metric, .logs]

  var icon: String {
    switch self {
    case .server: "externaldrive"
    case .advanced: "slider.horizontal.3"
    case .chat: "bubble"
    case .metric: "gauge.with.dots.needle.50percent"
    case .logs: "doc.text"
    case .settings: "gearshape"
    }
  }

  func title(language: AppLanguage) -> String {
    switch self {
    case .server: L10n.string("Server", language: language)
    case .advanced: L10n.string("Advance", language: language)
    case .chat: L10n.string("Chat", language: language)
    case .metric: L10n.string("Metric", language: language)
    case .logs: L10n.string("Logs", language: language)
    case .settings: L10n.string("Settings", language: language)
    }
  }
}

private enum AppTheme {
  static let pageBackground = Color(red: 0.095, green: 0.095, blue: 0.1)
  static let sidebarBackground = Color(red: 0.12, green: 0.12, blue: 0.125)
  static let cardBackground = Color(red: 0.15, green: 0.15, blue: 0.155)
  static let fieldBackground = Color(red: 0.075, green: 0.075, blue: 0.08)
  static let cardRadius: CGFloat = 16
  static let fieldRadius: CGFloat = 8
}

private struct SectionHeader: View {
  let title: String

  var body: some View {
    Text(title.uppercased())
      .font(.callout.weight(.semibold))
      .tracking(1.1)
      .foregroundStyle(.secondary)
      .accessibilityAddTraits(.isHeader)
  }
}

private struct AppCardModifier: ViewModifier {
  let padding: CGFloat

  func body(content: Content) -> some View {
    content
      .padding(padding)
      .background(AppTheme.cardBackground, in: RoundedRectangle(cornerRadius: AppTheme.cardRadius))
      .overlay(
        RoundedRectangle(cornerRadius: AppTheme.cardRadius)
          .stroke(Color.primary.opacity(0.06))
      )
  }
}

private struct AppInputModifier: ViewModifier {
  let width: CGFloat?

  func body(content: Content) -> some View {
    content
      .textFieldStyle(.plain)
      .padding(.horizontal, 11)
      .frame(width: width)
      .frame(minHeight: 34)
      .background(AppTheme.fieldBackground, in: RoundedRectangle(cornerRadius: AppTheme.fieldRadius))
      .overlay(
        RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
          .stroke(Color.primary.opacity(0.12))
      )
  }
}

extension View {
  fileprivate func pageVisibility(_ isVisible: Bool) -> some View {
    opacity(isVisible ? 1 : 0)
      .allowsHitTesting(isVisible)
      .disabled(!isVisible)
      .accessibilityHidden(!isVisible)
  }

  fileprivate func appCard(padding: CGFloat = 16) -> some View {
    modifier(AppCardModifier(padding: padding))
  }

  fileprivate func appInput(width: CGFloat? = nil) -> some View {
    modifier(AppInputModifier(width: width))
  }
}

private struct ServerView: View {
  @Binding var configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  @ObservedObject var modelLibrary: ModelLibrary
  let language: AppLanguage
  @State private var confirmsDownload = false
  @State private var confirmsRepair = false
  @State private var repairTarget: InstalledModelInfo?
  @State private var confirmsReinstall = false
  @State private var reinstallTarget: URL?

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        serverSummaryCard

        SectionHeader(title: L10n.string("Model", language: language))
          .padding(.top, 10)

        if modelLibrary.usableModels.isEmpty {
          onboardingPanel
        } else {
          modelPanel
        }

        if modelLibrary.isBusy {
          operationPanel
        }

        if let message = modelLibrary.message {
          Label(
            message,
            systemImage: message.hasPrefix("無法") ? "exclamationmark.triangle.fill" : "info.circle"
          )
          .foregroundStyle(message.hasPrefix("無法") ? Color.red : Color.secondary)
          .textSelection(.enabled)
          .accessibilityLabel(L10n.string("Model status: %@", language: language, message))
        }

        if !modelLibrary.damagedModels.isEmpty || !modelLibrary.invalidModelURLs.isEmpty {
          damagedModelsPanel
        }

        SectionHeader(title: L10n.string("Server", language: language))
          .padding(.top, 10)
        serverPanel
      }
      .frame(maxWidth: 980)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
    .confirmationDialog(
      L10n.string("Download and install the model?"),
      isPresented: $confirmsDownload,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Download Model")) { modelLibrary.startDownload() }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(
        L10n.string(
          "The app will install the model in %@. You can resume an interrupted download.",
          modelLibrary.rootURL.path))
    }
    .confirmationDialog(
      L10n.string("Verify and repair the model?"),
      isPresented: $confirmsRepair,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Verify and Repair")) {
        if let repairTarget { modelLibrary.startRepair(repairTarget) }
      }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(
        L10n.string(
          "The app will verify the complete model. It will download only missing or damaged data."))
    }
    .confirmationDialog(
      L10n.string("Download the model again?"),
      isPresented: $confirmsReinstall,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Download Again"), role: .destructive) {
        if let reinstallTarget { modelLibrary.reinstall(reinstallTarget) }
      }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(
        L10n.string(
          "The app will move the damaged model to Trash. It will then download the complete model.")
      )
    }
  }

  private var serverSummaryCard: some View {
    HStack(spacing: 18) {
      Image(nsImage: NSImage(named: NSImage.applicationIconName) ?? NSImage())
        .resizable()
        .interpolation(.high)
        .frame(width: 56, height: 56)
        .accessibilityHidden(true)

      VStack(alignment: .leading, spacing: 5) {
        HStack(spacing: 10) {
          Text(selectedModel?.name ?? configuration.publicModel)
            .font(.title3.bold())
            .lineLimit(1)
          Label(server.state.label, systemImage: "circle.fill")
            .font(.callout.weight(.semibold))
            .foregroundStyle(statusColor)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(statusColor.opacity(0.12), in: Capsule())
        }
        Text(configuration.baseURL?.absoluteString ?? L10n.string("Invalid Base URL"))
          .font(.callout.monospaced())
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
      }
      .accessibilityElement(children: .combine)

      Spacer(minLength: 20)

      Button {
        if server.isActive {
          server.stop()
        } else {
          server.start(configuration)
        }
      } label: {
        Label(
          L10n.string(server.isActive ? "Stop Server" : "Start Server", language: language),
          systemImage: server.isActive ? "stop.fill" : "play.fill"
        )
      }
      .buttonStyle(.borderedProminent)
      .tint(.blue)
      .controlSize(.large)
      .disabled(!server.isActive && !modelLibrary.canUseModel(at: configuration.modelPath))
      .keyboardShortcut(server.isActive ? "." : "\r", modifiers: .command)
    }
    .appCard(padding: 22)
  }

  private var serverPanel: some View {
    VStack(alignment: .leading, spacing: 0) {
      if case .failed(let message) = server.state {
        Label(message, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
          .accessibilityLabel(L10n.string("Error: %@", language: language, message))
          .padding(.bottom, 12)
      }

      SettingRow(
        "Listen Address",
        hint: "Choose which devices can connect to the server.",
        language: language
      ) {
        Picker(L10n.string("Listen Address", language: language), selection: $configuration.host) {
          Text(L10n.string("127.0.0.1 (Local only)", language: language))
            .tag("127.0.0.1")
          Text(L10n.string("0.0.0.0 (All networks)", language: language))
            .tag("0.0.0.0")
        }
        .labelsHidden()
        .frame(width: 290)
      }
      .disabled(server.isActive)

      Divider()

      SettingRow(
        "Port",
        hint: "Default 11434. Restart the server after changing it.",
        language: language
      ) {
        TextField("11434", value: $configuration.port, format: .number.grouping(.never))
          .appInput(width: 120)
          .accessibilityLabel(L10n.string("Port", language: language))
      }
      .disabled(server.isActive)

      Divider()

      SettingRow(
        "API key",
        hint: "Optional for local use",
        language: language
      ) {
        SecureField(L10n.string("Optional for local use"), text: $configuration.apiKey)
          .appInput(width: 320)
      }
      .disabled(server.isActive)

      Divider()

      SettingRow(
        "Model ID",
        hint: "Model name exposed by the OpenAI-compatible API.",
        language: language
      ) {
        TextField("deepseek-v4-flash-0731", text: $configuration.publicModel)
          .appInput(width: 320)
      }
      .disabled(server.isActive)
    }
    .appCard()
  }

  private var onboardingPanel: some View {
    VStack(alignment: .leading, spacing: 24) {
      VStack(alignment: .leading, spacing: 6) {
        Text(
          L10n.string(
            modelLibrary.hasPartialDownload ? "Continue Model Installation" : "Install Model")
        )
        .font(.title2.bold())
        Text(
          L10n.string(
            "Download a model or select an existing model folder. You can then start the local server."
          )
        )
        .foregroundStyle(.secondary)
      }

      preflightPanel

      Toggle(
        L10n.string("Install DSpark with the model (adds 10.12 GiB)"),
        isOn: $modelLibrary.installDSparkWithModel
      )
      .disabled(modelLibrary.isBusy)

      HStack(spacing: 12) {
        Button(L10n.string("Select Model Folder")) { chooseModelDirectory() }
          .disabled(modelLibrary.isBusy)
        Button(L10n.string(modelLibrary.hasPartialDownload ? "Resume Download" : "Download Model"))
        {
          if modelLibrary.hasPartialDownload {
            modelLibrary.startDownload()
          } else {
            confirmsDownload = true
          }
        }
        .buttonStyle(.borderedProminent)
        .tint(.blue)
        .disabled(modelLibrary.isBusy || !modelLibrary.canStartDownload)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .appCard()
  }

  private var preflightPanel: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text(L10n.string("Checks Before Download"))
        .font(.headline)
      ForEach(modelLibrary.preflightChecks) { check in
        HStack(alignment: .top, spacing: 10) {
          Image(systemName: preflightSymbol(check.status))
            .foregroundStyle(preflightColor(check.status))
            .accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 2) {
            Text(check.title).fontWeight(.medium)
            Text(check.detail)
              .font(.callout)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
        }
        .accessibilityElement(children: .combine)
      }
    }
  }

  private var modelPanel: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(spacing: 12) {
        if modelLibrary.isScanning {
          ProgressView().controlSize(.small)
        } else {
          Image(systemName: selectedModel == nil ? "externaldrive" : "checkmark.circle.fill")
            .foregroundStyle(selectedModel == nil ? Color.secondary : Color.green)
            .accessibilityHidden(true)
        }

        Text(L10n.string("Installed model"))
          .font(.body.weight(.medium))
        Spacer()
        Picker(L10n.string("Installed model"), selection: $configuration.modelPath) {
          ForEach(modelLibrary.usableModels) { model in
            Text(L10n.string("%@ · %@", model.name, formattedBytes(model.size)))
              .tag(model.url.path)
          }
        }
        .labelsHidden()
        .frame(width: 420)
      }

      Divider()

      HStack(spacing: 10) {
        Button(L10n.string("Show in Finder")) {
          if let selectedModel { modelLibrary.reveal(selectedModel.url) }
        }
        Button(L10n.string("Verify Complete Model")) {
          if let selectedModel { modelLibrary.startVerification(selectedModel) }
        }
        .disabled(server.isActive || modelLibrary.isBusy)
        if let selectedModel, !selectedModel.hasDSpark {
          Button(L10n.string("Install DSpark (10.12 GiB)")) {
            modelLibrary.startDSparkInstallation(selectedModel)
          }
          .disabled(server.isActive || modelLibrary.isBusy)
        }
        Spacer()
        Button(L10n.string("Select Another Folder")) { chooseModelDirectory() }
          .disabled(server.isActive || modelLibrary.isBusy)
      }

      Text(modelLibrary.rootURL.path)
        .font(.callout.monospaced())
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .truncationMode(.middle)
        .textSelection(.enabled)

      if modelLibrary.verificationModelPath == configuration.modelPath,
        let issues = modelLibrary.verificationIssues
      {
        if issues.isEmpty {
          Label(L10n.string("Complete verification passed"), systemImage: "checkmark.seal.fill")
            .foregroundStyle(.green)
        } else {
          Label(
            L10n.string("%lld files need repair", Int64(issues.count)),
            systemImage: "exclamationmark.triangle.fill"
          )
          .foregroundStyle(.red)
          Button(L10n.string("Verify and Repair")) {
            repairTarget = selectedModel
            confirmsRepair = true
          }
          .disabled(modelLibrary.isBusy || server.isActive)
        }
      }
    }
    .appCard()
  }

  private var operationPanel: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text(modelLibrary.operationPhase.label).font(.headline)
        if let progress = modelLibrary.operationProgress, let fraction = progress.fraction {
          ProgressView(value: fraction)
            .accessibilityLabel(modelLibrary.operationPhase.label)
            .accessibilityValue(fraction.formatted(.percent.precision(.fractionLength(0))))
          HStack(spacing: 16) {
            Text(
              L10n.string(
                "%@ / %@", formattedBytes(progress.completedBytes),
                formattedBytes(progress.totalBytes)))
            if let speed = progress.bytesPerSecond, speed > 0 {
              Text(L10n.string("%@/s", formattedBytes(UInt64(speed))))
            }
            if let seconds = progress.estimatedSecondsRemaining, seconds.isFinite {
              Text(L10n.string("About %@ remaining", formattedDuration(seconds)))
            }
          }
          .font(.callout.monospacedDigit())
          .foregroundStyle(.secondary)
        } else {
          ProgressView()
            .accessibilityLabel(modelLibrary.operationPhase.label)
        }
        Button(L10n.string("Stop Current Operation")) { modelLibrary.cancelOperation() }
          .disabled(modelLibrary.operationPhase == .cancelling)
    }
    .appCard()
  }

  private var damagedModelsPanel: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text(L10n.string("Models That Need Attention")).font(.headline)
        ForEach(modelLibrary.damagedModels) { model in
          HStack(alignment: .top, spacing: 12) {
            Label(
              L10n.string(
                "%@: %lld files are missing or have the wrong size", model.name,
                Int64(model.quickIssues.count)),
              systemImage: "exclamationmark.triangle.fill"
            )
            .foregroundStyle(.red)
            Spacer()
            Button(L10n.string("Show in Finder")) { modelLibrary.reveal(model.url) }
            Button(L10n.string("Verify and Repair")) {
              repairTarget = model
              confirmsRepair = true
            }
            .disabled(modelLibrary.isBusy || server.isActive)
          }
        }
        ForEach(modelLibrary.invalidModelURLs, id: \.path) { url in
          HStack(alignment: .top, spacing: 12) {
            Label(
              L10n.string("%@: The manifest cannot be read", url.lastPathComponent),
              systemImage: "xmark.octagon.fill"
            )
            .foregroundStyle(.red)
            Spacer()
            Button(L10n.string("Show in Finder")) { modelLibrary.reveal(url) }
            Button(L10n.string("Download Again")) {
              reinstallTarget = url
              confirmsReinstall = true
            }
            .disabled(modelLibrary.isBusy || server.isActive || !modelLibrary.canDownload)
          }
        }
    }
    .appCard()
  }

  private var selectedModel: InstalledModelInfo? {
    modelLibrary.usableModels.first { $0.url.path == configuration.modelPath }
  }

  private var statusColor: Color {
    switch server.state {
    case .running: .green
    case .failed: .red
    case .starting, .stopping: .orange
    case .stopped: .secondary
    }
  }

  private func chooseModelDirectory() {
    let panel = NSOpenPanel()
    panel.title = L10n.string("Select Model Folder")
    panel.prompt = L10n.string("Select Folder")
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK, let url = panel.url {
      Task {
        await modelLibrary.setRoot(url)
        configuration.modelPath = modelLibrary.usableModels.first?.url.path ?? ""
      }
    }
  }

  private func formattedBytes(_ bytes: UInt64) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .file)
  }

  private func formattedDuration(_ seconds: Double) -> String {
    let totalMinutes = max(1, Int(seconds / 60))
    if totalMinutes < 60 { return L10n.string("%lld min", Int64(totalMinutes)) }
    return L10n.string(
      "%lld hr %lld min", Int64(totalMinutes / 60), Int64(totalMinutes % 60))
  }

  private func preflightSymbol(_ status: PreflightStatus) -> String {
    switch status {
    case .passed: "checkmark.circle.fill"
    case .warning: "exclamationmark.circle.fill"
    case .failed: "xmark.circle.fill"
    }
  }

  private func preflightColor(_ status: PreflightStatus) -> Color {
    switch status {
    case .passed: .green
    case .warning: .orange
    case .failed: .red
    }
  }
}

private struct AdvancedView: View {
  @Binding var configuration: ServerConfiguration
  let serverActive: Bool
  let dsparkAvailable: Bool
  let language: AppLanguage

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        SectionHeader(title: L10n.string("Generate", language: language))
        VStack(spacing: 0) {
          integerField(
            "Max tokens",
            hint: "Default token limit for each request.",
            value: $configuration.defaultMaxTokens
          )
          Divider()
          doubleField(
            "Temperature",
            hint: "A higher value increases output variation.",
            value: $configuration.defaultTemperature
          )
          Divider()
          doubleField(
            "Top P",
            hint: "A lower value reduces the candidate token range.",
            value: $configuration.defaultTopP
          )
        }
        .appCard()
        .disabled(serverActive)

        SectionHeader(title: L10n.string("Power Saving Mode", language: language))
          .padding(.top, 12)
        powerSavingPanel
          .disabled(serverActive)

        SectionHeader(title: L10n.string("Runtime", language: language))
          .padding(.top, 12)
        VStack(spacing: 0) {
          integerField(
            "Slots",
            hint:
              "Number of routed experts in the Active Parameters Cache. The recommended value is 1152.",
            value: $configuration.slots
          )
          Divider()
          integerField(
            "Read workers",
            hint:
              "Number of workers that read expert blobs at the same time. The recommended value is 4.",
            value: $configuration.readWorkers
          )
          Divider()
          integerField(
            "Prefill step size",
            hint: "0 selects 128, 256, or 1024 based on the prompt length.",
            value: $configuration.prefillStepSize
          )
          Divider()
          toggleField(
            "Use layer-major prefill",
            hint: "Loads routed experts by layer during prefill.",
            value: $configuration.layerMajorPrefill
          )
          Divider()
          integerField(
            "Prompt cache entries",
            hint: "Number of linear conversations to keep. The recommended value is 2.",
            value: $configuration.promptCacheEntries
          )
          Divider()
          integerField(
            "Prompt cache GiB",
            hint: "Memory limit for all prompt caches. The recommended value is 8.",
            value: $configuration.promptCacheMemoryGiB
          )
          Divider()
          SettingRow(
            "Warmup prompt",
            hint: "Optional UTF-8 prompt file path",
            language: language
          ) {
            TextField(
              L10n.string("Optional UTF-8 prompt file path", language: language),
              text: $configuration.warmupPromptPath
            )
            .appInput(width: 340)
          }
          Divider()
          toggleField(
            "Use BF16 KV cache",
            hint: "Stores the KV cache in BF16 format.",
            value: $configuration.bf16KVCache
          )
          Divider()
          toggleField(
            "Use DSpark",
            hint: "Uses DSpark speculative decoding when it is installed.",
            value: $configuration.dsparkEnabled
          )
          .disabled(!dsparkAvailable)
          Divider()
          integerField(
            "DSpark slots",
            hint:
              "Number of DSpark routed experts kept in memory. The recommended value is 768.",
            value: $configuration.dsparkSlots
          )
          .disabled(!configuration.dsparkEnabled || !dsparkAvailable)
          Divider()
          doubleField(
            "DSpark confidence threshold",
            hint:
              "0 keeps all draft tokens. A higher value rejects low-confidence draft tokens early.",
            value: $configuration.dsparkConfidenceThreshold
          )
          .disabled(!configuration.dsparkEnabled || !dsparkAvailable)
        }
        .appCard()
        .disabled(serverActive)
      }
      .frame(maxWidth: 980)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }

  private var powerSavingPanel: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(alignment: .firstTextBaseline) {
        SettingLabel(
          "SSD read limit",
          hint: "A lower SSD read limit can reduce generation speed.",
          language: language
        )
        Spacer()
        Text(powerSavingLimitLabel(configuration.powerSavingLimitGBps))
          .font(.body.weight(.semibold).monospacedDigit())
      }

      VStack(spacing: 6) {
        HStack {
          Text(L10n.string("Power saving", language: language))
          Spacer()
          Text(L10n.string("Performance", language: language))
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(.secondary)

        Slider(
          value: powerSavingSelection,
          in: 0...Double(ServerConfiguration.powerSavingLimitOptionsGBps.count - 1),
          step: 1
        )
        .accessibilityLabel(L10n.string("SSD read limit", language: language))
        .accessibilityValue(powerSavingLimitLabel(configuration.powerSavingLimitGBps))

        HStack(spacing: 0) {
          ForEach(
            Array(ServerConfiguration.powerSavingLimitOptionsGBps.enumerated()),
            id: \.offset
          ) { index, limit in
            Text(powerSavingLimitLabel(limit))
              .font(.caption.monospacedDigit())
              .foregroundStyle(.secondary)
              .frame(
                maxWidth: .infinity,
                alignment: index == 0
                  ? .leading
                  : index == ServerConfiguration.powerSavingLimitOptionsGBps.count - 1
                    ? .trailing : .center
              )
          }
        }
        .accessibilityHidden(true)
      }
    }
    .appCard()
  }

  private var powerSavingSelection: Binding<Double> {
    Binding(
      get: {
        Double(
          ServerConfiguration.powerSavingLimitOptionsGBps.firstIndex {
            $0 == configuration.powerSavingLimitGBps
          } ?? ServerConfiguration.powerSavingLimitOptionsGBps.count - 1
        )
      },
      set: { value in
        let index = min(
          max(Int(value.rounded()), 0),
          ServerConfiguration.powerSavingLimitOptionsGBps.count - 1
        )
        configuration.powerSavingLimitGBps =
          ServerConfiguration.powerSavingLimitOptionsGBps[index]
      }
    )
  }

  private func powerSavingLimitLabel(_ limit: Double?) -> String {
    guard let limit else { return L10n.string("Unlimited", language: language) }
    if limit == 0.5 { return L10n.string("500 MB/s", language: language) }
    return L10n.string("%lld GB/s", language: language, Int64(limit))
  }

  private func integerField(_ label: String, hint: String, value: Binding<Int>) -> some View {
    SettingRow(label, hint: hint, language: language) {
      TextField(
        L10n.string(label, language: language),
        value: value,
        format: .number.grouping(.never)
      )
      .labelsHidden()
      .appInput(width: 120)
    }
  }

  private func doubleField(_ label: String, hint: String, value: Binding<Double>) -> some View {
    SettingRow(label, hint: hint, language: language) {
      TextField(
        L10n.string(label, language: language),
        value: value,
        format: .number.precision(.fractionLength(0...6))
      )
      .labelsHidden()
      .appInput(width: 120)
    }
  }

  private func toggleField(_ label: String, hint: String, value: Binding<Bool>) -> some View {
    SettingRow(label, hint: hint, language: language) {
      Toggle(L10n.string(label, language: language), isOn: value)
        .labelsHidden()
        .accessibilityLabel(L10n.string(label, language: language))
    }
  }
}

private struct LogsView: View {
  @ObservedObject var server: ServerController
  let language: AppLanguage

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      SectionHeader(title: L10n.string("Server log", language: language))
      ScrollView {
        Text(
          server.log.isEmpty
            ? L10n.string(
              "The log will appear here after the server starts.", language: language)
            : server.log
        )
        .font(.system(.callout, design: .monospaced))
        .foregroundStyle(server.log.isEmpty ? .secondary : .primary)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .topLeading)
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .appCard()
      .accessibilityLabel(L10n.string("Server log", language: language))
    }
    .frame(maxWidth: 980)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding(.horizontal, 40)
    .padding(.vertical, 24)
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }
}

private struct SettingsView: View {
  @Binding var languageCode: String
  let language: AppLanguage
  let checkForUpdates: () -> Void

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        HStack(spacing: 20) {
          Image(nsImage: NSImage(named: NSImage.applicationIconName) ?? NSImage())
            .resizable()
            .interpolation(.high)
            .frame(width: 76, height: 76)
            .accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 5) {
            Text("DeepSeekV4SSD")
              .font(.title.bold())
            Text(L10n.string("Local DeepSeek inference from SSD.", language: language))
              .font(.title3)
              .foregroundStyle(.secondary)
            Text(
              L10n.string(
                "Version %@ · build %@",
                language: language,
                appVersion,
                buildVersion
              )
            )
            .font(.callout.monospacedDigit())
            .foregroundStyle(.tertiary)
            .textSelection(.enabled)
          }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCard(padding: 22)

        SectionHeader(title: L10n.string("Preferences", language: language))
          .padding(.top, 12)

        VStack(alignment: .leading, spacing: 0) {
          SettingRow(
            "Language",
            hint: "Select the language used by the app.",
            language: language
          ) {
            Picker(L10n.string("Language", language: language), selection: $languageCode) {
              ForEach(AppLanguage.allCases) { option in
                Text(option.displayName(language: language)).tag(option.rawValue)
              }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(minWidth: 180)
          }

          Divider()

          SettingRow(
            "Software Updates",
            hint: "Check GitHub Releases for a newer app version.",
            language: language
          ) {
            Button(action: checkForUpdates) {
              Label(
                L10n.string("Check for Updates…", language: language),
                systemImage: "arrow.clockwise"
              )
            }
          }
        }
        .appCard()

        SectionHeader(title: L10n.string("Project", language: language))
          .padding(.top, 12)

        VStack(spacing: 0) {
          projectLink(
            title: "GitHub Repository",
            note: "Source, issues, and roadmap",
            icon: "chevron.left.forwardslash.chevron.right",
            url: "https://github.com/yanun0323/deepseek_ssd"
          )
          Divider()
          projectLink(
            title: "Releases",
            note: "Download the latest macOS app",
            icon: "shippingbox",
            url: "https://github.com/yanun0323/deepseek_ssd/releases"
          )
          Divider()
          projectLink(
            title: "Documentation",
            note: "Setup, model management, and API usage",
            icon: "book.closed",
            url: "https://github.com/yanun0323/deepseek_ssd#readme"
          )
          Divider()
          projectLink(
            title: "Report an Issue",
            note: "Report bugs and request features on GitHub",
            icon: "exclamationmark.bubble",
            url: "https://github.com/yanun0323/deepseek_ssd/issues"
          )
        }
        .appCard()

        SectionHeader(title: L10n.string("License", language: language))
          .padding(.top, 12)

        VStack(alignment: .leading, spacing: 6) {
          Label(L10n.string("MIT License", language: language), systemImage: "point.3.connected.trianglepath.dotted")
            .font(.headline)
          Text(
            L10n.string(
              "Copyright © 2026 Yanun. See the LICENSE file in the repository for the full text.",
              language: language
            )
          )
          .font(.callout)
          .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCard()
      }
      .frame(maxWidth: 980)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }

  private var appVersion: String {
    Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "-"
  }

  private var buildVersion: String {
    Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "-"
  }

  private func projectLink(title: String, note: String, icon: String, url: String) -> some View {
    Link(destination: URL(string: url)!) {
      HStack(spacing: 14) {
        Image(systemName: icon)
          .font(.title3)
          .foregroundStyle(.secondary)
          .frame(width: 28)
          .accessibilityHidden(true)
        VStack(alignment: .leading, spacing: 2) {
          Text(L10n.string(title, language: language))
            .font(.body.weight(.semibold))
          Text(L10n.string(note, language: language))
            .font(.callout)
            .foregroundStyle(.secondary)
        }
        Spacer()
        Image(systemName: "arrow.up.right.square")
          .foregroundStyle(.secondary)
          .accessibilityHidden(true)
      }
      .padding(.vertical, 9)
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .accessibilityLabel(L10n.string(title, language: language))
    .accessibilityHint(L10n.string(note, language: language))
  }
}

private struct MetricView: View {
  let model: String
  let state: ServerController.State
  let performance: LivePerformance
  let history: PerformanceHistory
  let language: AppLanguage
  let clearHistory: () -> Void

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        HStack(alignment: .center) {
          SectionHeader(title: localized("Serving Stats"))
          Spacer()
          Button(action: clearHistory) {
            Label(localized("Clear metric history"), systemImage: "trash")
              .labelStyle(.iconOnly)
          }
          .buttonStyle(.borderless)
          .frame(width: 36, height: 36)
          .disabled(history.isEmpty)
          .help(localized("Clear metric history"))
          .accessibilityLabel(localized("Clear metric history"))
        }

        LazyVGrid(
          columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 3),
          spacing: 14
        ) {
          metricCard(.inputTokens)
          metricCard(.outputTokens)
          metricCard(.cacheHitRate)
        }

        SectionHeader(title: localized("Average Speed"))
          .padding(.top, 10)

        LazyVGrid(
          columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 2),
          spacing: 14
        ) {
          metricCard(.prefillTokensPerSecond)
          metricCard(.decodeTokensPerSecond)
        }

        SectionHeader(title: localized("Active Now"))
          .padding(.top, 10)
        activeCard

        SectionHeader(title: localized("System"))
          .padding(.top, 10)
        systemCard
      }
      .frame(maxWidth: 980)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .accessibilityElement(children: .contain)
    .environment(\.locale, language.locale)
  }

  private func metricCard(_ metric: PerformanceMetric) -> some View {
    let live = formattedValue(performance.snapshot[metric], for: metric, live: true)
    let statistics = history[metric]
    let maximum = statistics.map { formattedValue($0.maximum, for: metric) } ?? "-"
    let p95 = statistics.map { formattedValue($0.p95, for: metric) } ?? "-"
    return VStack(spacing: 14) {
      Text(localized(metricTitle(metric)))
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
      Text(live)
        .font(.title2.weight(.semibold).monospacedDigit())
        .lineLimit(1)
      HStack(spacing: 16) {
        statisticLabel(localized("Maximum"), value: maximum)
        Divider().frame(height: 28)
        statisticLabel("P95", value: p95)
      }
    }
    .frame(maxWidth: .infinity, minHeight: 126)
    .appCard(padding: 18)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(
      "\(localized(metricTitle(metric)))。\(localized("Live"))：\(live)。"
        + "\(localized("Maximum"))：\(maximum)。P95：\(p95)"
    )
  }

  private func statisticLabel(_ title: String, value: String) -> some View {
    VStack(spacing: 2) {
      Text(title)
        .font(.caption)
        .foregroundStyle(.secondary)
      Text(value)
        .font(.callout.monospacedDigit())
        .lineLimit(1)
    }
    .frame(maxWidth: .infinity)
  }

  private var activeCard: some View {
    HStack(spacing: 12) {
      Circle()
        .fill(statusColor)
        .frame(width: 9, height: 9)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 3) {
        Text(model)
          .font(.headline)
          .lineLimit(1)
        Text(localizedStateLabel)
          .font(.callout)
          .foregroundStyle(.secondary)
      }
      Spacer()
      if performance.dsparkEnabled {
        Text(
          L10n.string(
            "DSpark · %@ accepted · %@ tokens per round",
            language: language,
            performance.dsparkAcceptanceRate.formatted(
              .percent.precision(.fractionLength(1))),
            performance.dsparkAverageAcceptedLength.formatted(
              .number.precision(.fractionLength(1)))
          )
        )
        .font(.callout.monospacedDigit())
        .foregroundStyle(.secondary)
      }
    }
    .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
    .appCard()
    .accessibilityElement(children: .combine)
  }

  private var systemCard: some View {
    VStack(spacing: 0) {
      HStack {
        Text(localized("Metric"))
        Spacer()
        Text(localized("Live")).frame(width: 120, alignment: .trailing)
        Text(localized("Maximum")).frame(width: 120, alignment: .trailing)
        Text("P95").frame(width: 120, alignment: .trailing)
      }
      .font(.callout.weight(.semibold))
      .foregroundStyle(.secondary)
      .padding(.bottom, 12)

      ForEach(
        [
          PerformanceMetric.memoryUsage,
          .ssdReadSpeed,
          .firstTokenWaitTime,
          .completionTime,
        ]
      ) { metric in
        Divider()
        metricRow(metric)
      }
    }
    .appCard()
  }

  private func metricRow(_ metric: PerformanceMetric) -> some View {
    let live = formattedValue(performance.snapshot[metric], for: metric, live: true)
    let statistics = history[metric]
    let maximum = statistics.map { formattedValue($0.maximum, for: metric) } ?? "-"
    let p95 = statistics.map { formattedValue($0.p95, for: metric) } ?? "-"
    return HStack {
      Text(localized(metricTitle(metric)))
        .font(.body.weight(.medium))
      Spacer()
      Text(live).frame(width: 120, alignment: .trailing)
      Text(maximum).frame(width: 120, alignment: .trailing)
      Text(p95).frame(width: 120, alignment: .trailing)
    }
    .font(.body.monospacedDigit())
    .padding(.vertical, 10)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(
      "\(localized(metricTitle(metric)))。\(localized("Live"))：\(live)。"
        + "\(localized("Maximum"))：\(maximum)。P95：\(p95)"
    )
  }

  private func localized(_ key: String) -> String {
    L10n.string(key, language: language)
  }

  private var localizedStateLabel: String {
    switch state {
    case .stopped: localized("Stopped")
    case .starting: localized("Starting")
    case .running: localized("Running")
    case .stopping: localized("Stopping")
    case .failed: localized("Start failed")
    }
  }

  private func metricTitle(_ metric: PerformanceMetric) -> String {
    switch metric {
    case .prefillTokensPerSecond: "Prefill Tok/s"
    case .decodeTokensPerSecond: "Decode Tok/s"
    case .inputTokens: "Input Tokens"
    case .outputTokens: "Output Tokens"
    case .memoryUsage: "Memory usage"
    case .ssdReadSpeed: "SSD read speed"
    case .cacheHitRate: "Cache Hit rate"
    case .firstTokenWaitTime: "First Token wait time"
    case .completionTime: "Completion time"
    }
  }

  private func formattedValue(
    _ value: Double,
    for metric: PerformanceMetric,
    live: Bool = false
  ) -> String {
    if value == 0 { return "-" }
    if live && !hasLiveValue(metric, value: value) { return "-" }
    switch metric {
    case .prefillTokensPerSecond, .decodeTokensPerSecond:
      return value.formatted(.number.precision(.fractionLength(1)))
    case .inputTokens, .outputTokens:
      return Int64(value.rounded()).formatted()
    case .memoryUsage:
      return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    case .ssdReadSpeed:
      return "\(ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .file))/s"
    case .cacheHitRate:
      return value.formatted(.percent.precision(.fractionLength(1)))
    case .firstTokenWaitTime, .completionTime:
      if value < 1 {
        return "\((value * 1_000).formatted(.number.precision(.fractionLength(0)))) ms"
      }
      return "\(value.formatted(.number.precision(.fractionLength(2)))) s"
    }
  }

  private func hasLiveValue(_ metric: PerformanceMetric, value: Double) -> Bool {
    guard performance.hasStatus else { return false }
    switch metric {
    case .memoryUsage:
      return value > 0
    case .prefillTokensPerSecond, .firstTokenWaitTime:
      return value > 0
    case .inputTokens, .outputTokens, .decodeTokensPerSecond, .completionTime:
      return performance.snapshot.inputTokens > 0
    case .ssdReadSpeed, .cacheHitRate:
      return true
    }
  }

  private var statusColor: Color {
    switch state {
    case .running: .green
    case .failed: .red
    case .starting, .stopping: .orange
    case .stopped: .secondary
    }
  }

}

private struct SettingLabel: View {
  let title: String
  let hint: String
  let language: AppLanguage

  init(_ title: String, hint: String, language: AppLanguage) {
    self.title = title
    self.hint = hint
    self.language = language
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(L10n.string(title, language: language))
        .font(.body.weight(.medium))
      Text(L10n.string(hint, language: language))
        .font(.callout)
        .foregroundStyle(.secondary)
    }
    .help(L10n.string(hint, language: language))
    .accessibilityElement(children: .combine)
  }
}

private struct SettingRow<Value: View>: View {
  let title: String
  let hint: String
  let language: AppLanguage
  let value: Value

  init(
    _ title: String,
    hint: String,
    language: AppLanguage,
    @ViewBuilder value: () -> Value
  ) {
    self.title = title
    self.hint = hint
    self.language = language
    self.value = value()
  }

  var body: some View {
    HStack(alignment: .center, spacing: 32) {
      SettingLabel(title, hint: hint, language: language)
        .frame(maxWidth: .infinity, alignment: .leading)

      value
        .fixedSize(horizontal: true, vertical: false)
    }
    .frame(maxWidth: .infinity)
    .padding(.vertical, 8)
  }
}

private struct ChatView: View {
  let configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  let language: AppLanguage
  @State private var messages = ChatHistory.load()
  @AppStorage("chatDraft") private var input = ""
  @AppStorage("chatThinkingMode") private var thinkingMode = "chat"
  @State private var isSending = false
  @State private var generationTask: Task<Void, Never>?
  @State private var errorMessage: String?
  @State private var showingClearConfirmation = false

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack(spacing: 12) {
        HStack(spacing: 8) {
          Text(localized("Decode Tok/s"))
            .foregroundStyle(.secondary)
          Text(liveDecodeRate)
            .font(.headline.monospacedDigit())
        }
        .accessibilityElement(children: .combine)
        Spacer()
        Picker(localized("Mode"), selection: $thinkingMode) {
          Text(localized("Chat")).tag("chat")
          Text(localized("Thinking")).tag("thinking")
        }
        .pickerStyle(.segmented)
        .tint(.blue)
        .frame(width: 220)
      }
      .appCard(padding: 16)

      VStack(spacing: 0) {
        ScrollViewReader { scroll in
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
              if messages.isEmpty {
                ContentUnavailableView(
                  localized("No Test Messages"),
                  systemImage: "bubble.left",
                  description: Text(localized("Start the server. Then send a message."))
                )
                .frame(maxWidth: .infinity, minHeight: 280)
              } else {
                ForEach(messages) { message in
                  VStack(alignment: .leading, spacing: 6) {
                    Text(message.role == "user" ? localized("You") : "DeepSeek")
                      .font(.callout.bold())
                      .foregroundStyle(.secondary)
                    if !message.reasoningContent.isEmpty {
                      VStack(alignment: .leading, spacing: 4) {
                        Text(localized("Reasoning"))
                          .font(.callout.bold())
                          .foregroundStyle(.secondary)
                        Text(message.reasoningContent)
                          .foregroundStyle(.secondary)
                          .textSelection(.enabled)
                      }
                    }
                    if !message.content.isEmpty {
                      Text(message.content)
                        .textSelection(.enabled)
                    }
                    ForEach(message.toolCalls) { toolCall in
                      GroupBox(localized("Tool call")) {
                        VStack(alignment: .leading, spacing: 8) {
                          LabeledContent(
                            localized("Function"), value: toolCall.function.name)
                          VStack(alignment: .leading, spacing: 3) {
                            Text(localized("Arguments"))
                              .foregroundStyle(.secondary)
                            Text(toolCall.function.arguments)
                              .font(.system(.body, design: .monospaced))
                              .textSelection(.enabled)
                          }
                          Text(localized("The app does not run this tool."))
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                      }
                    }
                    if message.role == "assistant" && message.content.isEmpty
                      && message.reasoningContent.isEmpty && message.toolCalls.isEmpty
                    {
                      ProgressView(localized("Generating"))
                        .controlSize(.small)
                    }
                  }
                  .padding(14)
                  .frame(maxWidth: .infinity, alignment: .leading)
                  .background(
                    message.role == "user"
                      ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 10)
                  )
                  .accessibilityElement(children: .combine)
                }
              }
              Color.clear.frame(height: 1).id("chat-bottom")
            }
            .padding(8)
          }
          .onChange(of: streamedCharacterCount) {
            scroll.scrollTo("chat-bottom", anchor: .bottom)
          }
        }
      }
      .frame(maxHeight: .infinity)
      .appCard(padding: 10)

      if let errorMessage {
        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
          .accessibilityLabel(L10n.string("Error: %@", language: language, errorMessage))
      }

      VStack(alignment: .leading, spacing: 10) {
        ZStack(alignment: .topLeading) {
          if input.isEmpty {
            Text(localized("Enter a message…"))
              .foregroundStyle(.tertiary)
              .padding(.horizontal, 5)
              .padding(.vertical, 8)
              .allowsHitTesting(false)
          }
          TextEditor(text: $input)
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(8)
            .frame(minHeight: 90, maxHeight: 180)
            .background(
              AppTheme.fieldBackground,
              in: RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
            )
            .overlay(
              RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
                .stroke(Color.primary.opacity(0.12))
            )
            .accessibilityLabel(localized("Test message"))
        }

        HStack(spacing: 10) {
          if server.state != .running {
            Label(localized("Start the server first"), systemImage: "server.rack")
              .font(.callout)
              .foregroundStyle(.secondary)
          }
          Spacer()
          Button {
            showingClearConfirmation = true
          } label: {
            Label(localized("Clear Chat"), systemImage: "trash")
          }
          .disabled(messages.isEmpty || isSending)
          if isSending {
            Button(action: stopGenerating) {
              Label(localized("Stop Generating"), systemImage: "stop.fill")
            }
            .controlSize(.large)
          } else {
            Button {
              send()
            } label: {
              Label(localized("Generate"), systemImage: "arrow.up")
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            .controlSize(.large)
            .disabled(server.state != .running)
            .keyboardShortcut(.return, modifiers: .command)
          }
        }
      }
      .appCard(padding: 16)
    }
    .frame(maxWidth: 980)
    .frame(maxWidth: .infinity)
    .padding(.horizontal, 40)
    .padding(.vertical, 24)
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
    .onDisappear { generationTask?.cancel() }
    .confirmationDialog(
      localized("Clear the test chat?"),
      isPresented: $showingClearConfirmation,
      titleVisibility: .visible
    ) {
      Button(localized("Clear Chat"), role: .destructive) {
        messages.removeAll()
        ChatHistory.save(messages)
        input = ""
        errorMessage = nil
      }
      Button(localized("Cancel"), role: .cancel) {}
    } message: {
      Text(
        localized(
          "The app will clear only the local test chat. The server and other API clients are not affected."
        ))
    }
  }

  private var streamedCharacterCount: Int {
    guard let message = messages.last else { return 0 }
    return message.content.count + message.reasoningContent.count
      + message.toolCalls.reduce(0) {
        $0 + $1.function.name.count + $1.function.arguments.count
      }
  }

  private var liveDecodeRate: String {
    let value = server.performance.snapshot.decodeTokensPerSecond
    guard server.performance.generating, value > 0 else { return "-" }
    return value.formatted(.number.precision(.fractionLength(1)))
  }

  private func localized(_ key: String) -> String {
    L10n.string(key, language: language)
  }

  private func send() {
    let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, let baseURL = configuration.baseURL else { return }
    let userMessage = ChatMessage(role: "user", content: text)
    messages.append(userMessage)
    input = ""
    errorMessage = nil
    isSending = true
    let requestMessages = messages
    let assistantID = UUID()
    messages.append(ChatMessage(id: assistantID, role: "assistant", content: ""))
    ChatHistory.save(messages)
    generationTask = Task {
      defer {
        ChatHistory.save(messages)
        isSending = false
        generationTask = nil
      }
      do {
        _ = try await ChatClient.stream(
          messages: requestMessages,
          baseURL: baseURL,
          apiKey: configuration.apiKey,
          model: configuration.publicModel,
          thinkingMode: thinkingMode,
          enableTestTool: false
        ) { delta in
          guard let index = messages.firstIndex(where: { $0.id == assistantID }) else { return }
          messages[index].append(delta)
        }
      } catch {
        if Task.isCancelled {
          removeEmptyAssistantMessage(id: assistantID)
          return
        }
        if let index = messages.firstIndex(where: { $0.id == assistantID }),
          messages[index].content.isEmpty,
          messages[index].reasoningContent.isEmpty,
          messages[index].toolCalls.isEmpty
        {
          messages.remove(at: index)
        }
        errorMessage = L10n.string(
          "Could not get a response. %@", language: language, error.localizedDescription)
      }
    }
  }

  private func stopGenerating() {
    generationTask?.cancel()
  }

  private func removeEmptyAssistantMessage(id: UUID) {
    guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
    let message = messages[index]
    if message.content.isEmpty && message.reasoningContent.isEmpty && message.toolCalls.isEmpty {
      messages.remove(at: index)
    }
  }
}
