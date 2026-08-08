import AppKit
import SwiftUI

struct ContentView: View {
  @ObservedObject var server: ServerController
  @StateObject private var modelLibrary = ModelLibrary()
  @State private var configuration = ServerConfiguration.localDefault
  @AppStorage(L10n.preferenceKey) private var languageCode = AppLanguage.appDefault.rawValue

  var body: some View {
    HSplitView {
      ChatView(configuration: configuration, server: server, language: selectedLanguage)
        .frame(minWidth: 680, idealWidth: 880, maxWidth: .infinity)
      ServerView(
        configuration: $configuration,
        languageCode: $languageCode,
        server: server,
        modelLibrary: modelLibrary
      )
      .frame(minWidth: 480, idealWidth: 540, maxWidth: 680)
    }
    .environment(\.locale, selectedLanguage.locale)
    .task {
      await modelLibrary.scan()
      selectDetectedModel()
      modelLibrary.resumeDownloadIfNeeded()
    }
    .onChange(of: modelLibrary.models) { selectDetectedModel() }
    .onChange(of: configuration.modelPath) {
      UserDefaults.standard.set(configuration.modelPath, forKey: "selectedModelPath")
    }
  }

  private var selectedLanguage: AppLanguage {
    AppLanguage(rawValue: languageCode) ?? .appDefault
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

private struct ServerView: View {
  @Binding var configuration: ServerConfiguration
  @Binding var languageCode: String
  @ObservedObject var server: ServerController
  @ObservedObject var modelLibrary: ModelLibrary
  @State private var showsLog = false
  @State private var confirmsDownload = false
  @State private var confirmsRepair = false
  @State private var repairTarget: InstalledModelInfo?
  @State private var confirmsReinstall = false
  @State private var reinstallTarget: URL?
  @State private var confirmsDSparkRemoval = false

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      serverHeader
      languagePicker

      if case .failed(let message) = server.state {
        Label(message, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
          .accessibilityLabel(L10n.string("Error: %@", message))
      }

      ScrollView {
        VStack(alignment: .leading, spacing: 28) {
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
            .accessibilityLabel(L10n.string("Model status: %@", message))
          }

          if !modelLibrary.damagedModels.isEmpty || !modelLibrary.invalidModelURLs.isEmpty {
            damagedModelsPanel
          }

          if selectedModel != nil {
            advancedSettings
              .disabled(server.isActive)

            DisclosureGroup(L10n.string("Server log"), isExpanded: $showsLog) {
              ScrollView {
                Text(
                  server.log.isEmpty
                    ? L10n.string("The log will appear here after the server starts.") : server.log
                )
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(server.log.isEmpty ? .secondary : .primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(12)
              }
              .frame(minHeight: 140, maxHeight: 260)
              .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
              .accessibilityLabel(L10n.string("Server log"))
              .padding(.top, 12)
            }
          }
        }
        .padding(.vertical, 2)
      }
    }
    .padding(22)
    .background(Color(nsColor: .controlBackgroundColor).opacity(0.35))
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
    .confirmationDialog(
      L10n.string("Remove DSpark?"),
      isPresented: $confirmsDSparkRemoval,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Remove DSpark"), role: .destructive) {
        if let selectedModel { modelLibrary.removeDSpark(selectedModel) }
      }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(L10n.string("The main model will remain installed."))
    }
    .onChange(of: languageCode) { modelLibrary.refreshPreflight() }
  }

  private var languagePicker: some View {
    HStack(spacing: 12) {
      Text(L10n.string("Language"))
        .foregroundStyle(.secondary)
      Spacer()
      Picker(L10n.string("Language"), selection: $languageCode) {
        ForEach(AppLanguage.allCases) { language in
          Text(language.displayName).tag(language.rawValue)
        }
      }
      .labelsHidden()
      .pickerStyle(.menu)
      .frame(minWidth: 160, alignment: .trailing)
    }
  }

  private var serverHeader: some View {
    HStack(spacing: 12) {
      Image(systemName: server.state.symbol)
        .foregroundStyle(statusColor)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 2) {
        Text(server.state.label).font(.headline)
        Text(configuration.baseURL?.absoluteString ?? L10n.string("Invalid Base URL"))
          .font(.callout.monospaced())
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
      }
      .accessibilityElement(children: .combine)
      Spacer()
      if server.isActive {
        Button(L10n.string("Stop Server"), action: server.stop)
          .keyboardShortcut(".", modifiers: .command)
      } else {
        Button(L10n.string("Start Server")) { server.start(configuration) }
          .buttonStyle(.borderedProminent)
          .disabled(!modelLibrary.canUseModel(at: configuration.modelPath))
          .keyboardShortcut(.return, modifiers: .command)
      }
    }
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
        .disabled(modelLibrary.isBusy || !modelLibrary.canStartDownload)
      }
    }
    .padding(20)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
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
    GroupBox(L10n.string("Model")) {
      VStack(alignment: .leading, spacing: 12) {
        HStack(spacing: 12) {
          if modelLibrary.isScanning {
            ProgressView().controlSize(.small)
          } else {
            Image(systemName: selectedModel == nil ? "externaldrive" : "checkmark.circle.fill")
              .foregroundStyle(selectedModel == nil ? Color.secondary : Color.green)
              .accessibilityHidden(true)
          }

          Picker(L10n.string("Installed model"), selection: $configuration.modelPath) {
            ForEach(modelLibrary.usableModels) { model in
              Text(L10n.string("%@ · %@", model.name, formattedBytes(model.size)))
                .tag(model.url.path)
            }
          }
          .labelsHidden()
          .frame(maxWidth: .infinity)
        }

        VStack(alignment: .trailing, spacing: 10) {
          HStack(spacing: 12) {
            Spacer()
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
            } else if selectedModel?.hasDSpark == true {
              Button(L10n.string("Remove DSpark"), role: .destructive) {
                confirmsDSparkRemoval = true
              }
              .disabled(server.isActive || modelLibrary.isBusy)
            }
          }
          HStack {
            Spacer()
            Button(L10n.string("Select Another Folder")) { chooseModelDirectory() }
              .disabled(server.isActive || modelLibrary.isBusy)
          }
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
      .padding(8)
    }
  }

  private var operationPanel: some View {
    GroupBox(modelLibrary.operationPhase.label) {
      VStack(alignment: .leading, spacing: 10) {
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
      .padding(8)
    }
  }

  private var damagedModelsPanel: some View {
    GroupBox(L10n.string("Models That Need Attention")) {
      VStack(alignment: .leading, spacing: 16) {
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
      .padding(8)
    }
  }

  private var advancedSettings: some View {
    VStack(alignment: .leading, spacing: 18) {
      Text(L10n.string("Server"))
        .font(.headline)
      GroupBox {
        VStack(spacing: 12) {
          LabeledContent(L10n.string("Host")) {
            TextField("127.0.0.1", text: $configuration.host)
              .textFieldStyle(.roundedBorder)
          }
          LabeledContent(L10n.string("Port")) {
            TextField("11434", value: $configuration.port, format: .number.grouping(.never))
              .textFieldStyle(.roundedBorder)
              .frame(width: 100)
          }
          LabeledContent(L10n.string("API key")) {
            SecureField(L10n.string("Optional for local use"), text: $configuration.apiKey)
              .textFieldStyle(.roundedBorder)
          }
          LabeledContent(L10n.string("Model ID")) {
            TextField("deepseek-v4-flash-0731", text: $configuration.publicModel)
              .textFieldStyle(.roundedBorder)
          }
        }
        .padding(6)
      }

      Text(L10n.string("Runtime"))
        .font(.headline)
      GroupBox {
        VStack(spacing: 12) {
          integerField(
            "Slots",
            hint:
              "Number of routed experts in the Active Parameters Cache. The recommended value is 512.",
            value: $configuration.slots)
          integerField(
            "Read workers",
            hint:
              "Number of workers that read expert blobs at the same time. The recommended value is 4.",
            value: $configuration.readWorkers)
          integerField(
            "Prefill step size", hint: "0 selects 128, 256, or 1024 based on the prompt length.",
            value: $configuration.prefillStepSize)
          Toggle(
            L10n.string("Use layer-major prefill"),
            isOn: $configuration.layerMajorPrefill)
          integerField(
            "Prompt cache entries",
            hint: "Number of linear conversations to keep. The recommended value is 2.",
            value: $configuration.promptCacheEntries)
          integerField(
            "Prompt cache GiB",
            hint: "Memory limit for all prompt caches. The recommended value is 8.",
            value: $configuration.promptCacheMemoryGiB)
          LabeledContent(L10n.string("Warmup prompt")) {
            TextField(
              L10n.string("Optional UTF-8 prompt file path"),
              text: $configuration.warmupPromptPath
            )
            .textFieldStyle(.roundedBorder)
          }
          Toggle(L10n.string("Use BF16 KV cache"), isOn: $configuration.bf16KVCache)
          Toggle(L10n.string("Use DSpark"), isOn: $configuration.dsparkEnabled)
            .disabled(selectedModel?.hasDSpark != true)
          integerField(
            "DSpark slots",
            hint:
              "Number of DSpark routed experts kept in memory. The recommended value is 256.",
            value: $configuration.dsparkSlots
          )
          .disabled(!configuration.dsparkEnabled || selectedModel?.hasDSpark != true)
          doubleField(
            "DSpark confidence threshold",
            hint:
              "0 keeps all draft tokens. A higher value rejects low-confidence draft tokens early.",
            value: $configuration.dsparkConfidenceThreshold
          )
          .disabled(!configuration.dsparkEnabled || selectedModel?.hasDSpark != true)
        }
        .padding(6)
      }

      Text(L10n.string("Generate"))
        .font(.headline)
      GroupBox {
        VStack(spacing: 12) {
          integerField(
            "Max tokens", hint: "Default token limit for each request.",
            value: $configuration.defaultMaxTokens)
          doubleField(
            "Temperature", hint: "A higher value increases output variation.",
            value: $configuration.defaultTemperature)
          doubleField(
            "Top P", hint: "A lower value reduces the candidate token range.",
            value: $configuration.defaultTopP)
        }
        .padding(6)
      }
    }
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

  @ViewBuilder
  private func integerField(_ label: String, hint: String, value: Binding<Int>) -> some View {
    LabeledContent {
      TextField(
        L10n.string(label, language: selectedLanguage),
        value: value,
        format: .number.grouping(.never)
      )
      .labelsHidden()
      .textFieldStyle(.roundedBorder)
      .frame(width: 120)
    } label: {
      SettingLabel(label, hint: hint, language: selectedLanguage)
    }
  }

  @ViewBuilder
  private func doubleField(_ label: String, hint: String, value: Binding<Double>) -> some View {
    LabeledContent {
      TextField(
        L10n.string(label, language: selectedLanguage), value: value,
        format: .number.precision(.fractionLength(0...6))
      )
      .labelsHidden()
      .textFieldStyle(.roundedBorder)
      .frame(width: 120)
    } label: {
      SettingLabel(label, hint: hint, language: selectedLanguage)
    }
  }

  private var selectedLanguage: AppLanguage {
    AppLanguage(rawValue: languageCode) ?? .appDefault
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

private struct PerformancePanel: View {
  let model: String
  let state: ServerController.State
  let performance: LivePerformance
  let history: PerformanceHistory
  let language: AppLanguage
  let clearHistory: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(spacing: 12) {
        HStack(spacing: 9) {
          Circle()
            .fill(statusColor)
            .frame(width: 9, height: 9)
            .accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 2) {
            Text(model)
              .font(.headline)
              .lineLimit(1)
            Text(localizedStateLabel)
              .font(.callout)
              .foregroundStyle(.secondary)
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
              .font(.callout)
              .foregroundStyle(.secondary)
            }
          }
        }
        .accessibilityElement(children: .combine)

        Spacer(minLength: 8)

        Button(action: clearHistory) {
          Label(localized("Clear metric history"), systemImage: "trash")
        }
        .controlSize(.small)
        .disabled(history.isEmpty)
      }

      ScrollView(.horizontal) {
        Grid(alignment: .trailing, horizontalSpacing: 24, verticalSpacing: 9) {
          GridRow {
            Text(localized("Metric"))
              .gridColumnAlignment(.leading)
            Text(localized("Live"))
            Text(localized("Minimum"))
            Text(localized("Average"))
            Text(localized("Maximum"))
          }
          .font(.callout.weight(.semibold))
          .foregroundStyle(.secondary)

          ForEach(PerformanceMetric.allCases) { metric in
            metricRow(metric)
          }
        }
        .padding(.horizontal, 1)
      }
    }
    .padding(.horizontal, 18)
    .padding(.vertical, 14)
    .background(Color.secondary.opacity(0.07), in: RoundedRectangle(cornerRadius: 16))
    .overlay(
      RoundedRectangle(cornerRadius: 16)
        .stroke(Color.primary.opacity(0.06))
    )
    .accessibilityElement(children: .contain)
    .environment(\.locale, language.locale)
  }

  private func metricRow(_ metric: PerformanceMetric) -> some View {
    let live = formattedValue(performance.snapshot[metric], for: metric, live: true)
    let statistics = history[metric]
    let minimum = statistics.map { formattedValue($0.minimum, for: metric) } ?? "—"
    let average = statistics.map { formattedValue($0.average, for: metric) } ?? "—"
    let maximum = statistics.map { formattedValue($0.maximum, for: metric) } ?? "—"
    return GridRow {
      Text(localized(metricTitle(metric)))
        .font(.callout.weight(.medium))
        .foregroundStyle(.secondary)
        .gridColumnAlignment(.leading)
      metricValue(live)
      metricValue(minimum)
      metricValue(average)
      metricValue(maximum)
    }
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(
      "\(localized(metricTitle(metric)))。\(localized("Live"))：\(live)。"
        + "\(localized("Minimum"))：\(minimum)。\(localized("Average"))：\(average)。"
        + "\(localized("Maximum"))：\(maximum)"
    )
  }

  private func metricValue(_ value: String) -> some View {
    Text(value)
      .font(.callout.monospacedDigit())
      .lineLimit(1)
      .frame(minWidth: 88, alignment: .trailing)
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
    if live && !hasLiveValue(metric, value: value) { return "—" }
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
    HStack(spacing: 4) {
      Text(L10n.string(title, language: language))
      Image(systemName: "info.circle")
        .foregroundStyle(.secondary)
        .accessibilityHidden(true)
    }
    .help(L10n.string(hint, language: language))
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(L10n.string(title, language: language))
    .accessibilityHint(L10n.string(hint, language: language))
  }
}

private struct ChatView: View {
  let configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  let language: AppLanguage
  @State private var messages: [ChatMessage] = []
  @State private var input = ""
  @State private var thinkingMode = "chat"
  @State private var isSending = false
  @State private var errorMessage: String?
  @State private var showingClearConfirmation = false

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      PerformancePanel(
        model: configuration.publicModel,
        state: server.state,
        performance: server.performance,
        history: server.performanceHistory,
        language: language,
        clearHistory: server.clearPerformanceHistory
      )

      HStack(spacing: 12) {
        Text(localized("Chat"))
          .font(.title2.bold())
        Spacer()
        Picker(localized("Thinking mode"), selection: $thinkingMode) {
          Text(localized("Chat")).tag("chat")
          Text(localized("Thinking")).tag("thinking")
        }
        .pickerStyle(.segmented)
        .frame(width: 220)
      }

      GroupBox {
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
            .frame(minHeight: 90, maxHeight: 180)
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
          Button {
            send()
          } label: {
            if isSending {
              ProgressView()
                .controlSize(.small)
                .frame(minWidth: 72)
            } else {
              Label(localized("Generate"), systemImage: "arrow.up")
            }
          }
          .buttonStyle(.borderedProminent)
          .controlSize(.large)
          .disabled(isSending || server.state != .running)
          .keyboardShortcut(.return, modifiers: .command)
        }
      }
      .padding(14)
      .background(Color.secondary.opacity(0.07), in: RoundedRectangle(cornerRadius: 16))
      .overlay(
        RoundedRectangle(cornerRadius: 16)
          .stroke(Color.primary.opacity(0.06))
      )
    }
    .padding(22)
    .environment(\.locale, language.locale)
    .confirmationDialog(
      localized("Clear the test chat?"),
      isPresented: $showingClearConfirmation,
      titleVisibility: .visible
    ) {
      Button(localized("Clear Chat"), role: .destructive) {
        messages.removeAll()
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
    Task {
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
      isSending = false
    }
  }
}
