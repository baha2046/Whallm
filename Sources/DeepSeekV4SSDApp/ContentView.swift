import AppKit
import SwiftUI

struct ContentView: View {
  @ObservedObject var server: ServerController
  @StateObject private var modelLibrary = ModelLibrary()
  @State private var configuration = ServerConfiguration.localDefault

  var body: some View {
    TabView {
      ServerView(configuration: $configuration, server: server, modelLibrary: modelLibrary)
        .tabItem { Label("Server", systemImage: "server.rack") }
      if !modelLibrary.usableModels.isEmpty {
        ChatView(configuration: configuration, server: server)
          .tabItem { Label("測試對話", systemImage: "bubble.left.and.bubble.right") }
      }
    }
    .padding(16)
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
  @ObservedObject var server: ServerController
  @ObservedObject var modelLibrary: ModelLibrary
  @State private var showsAdvancedSettings = false
  @State private var showsLog = false
  @State private var confirmsDownload = false
  @State private var confirmsRepair = false
  @State private var repairTarget: InstalledModelInfo?
  @State private var confirmsReinstall = false
  @State private var reinstallTarget: URL?

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      if selectedModel != nil { serverHeader }

      if case .failed(let message) = server.state {
        Label(message, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
          .accessibilityLabel("錯誤：\(message)")
      }

      ScrollView {
        VStack(alignment: .leading, spacing: 24) {
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
            .accessibilityLabel("模型狀態：\(message)")
          }

          if !modelLibrary.damagedModels.isEmpty || !modelLibrary.invalidModelURLs.isEmpty {
            damagedModelsPanel
          }

          if server.isActive || server.performance.hasStatus {
            PerformancePanel(performance: server.performance)
          }

          if selectedModel != nil {
            DisclosureGroup("進階設定", isExpanded: $showsAdvancedSettings) {
              advancedSettings
                .padding(.top, 12)
            }
            .disabled(server.isActive)

            DisclosureGroup("Server 日誌", isExpanded: $showsLog) {
              ScrollView {
                Text(server.log.isEmpty ? "啟動 server 後，日誌會顯示在這裡。" : server.log)
                  .font(.system(.caption, design: .monospaced))
                  .foregroundStyle(server.log.isEmpty ? .secondary : .primary)
                  .textSelection(.enabled)
                  .frame(maxWidth: .infinity, alignment: .topLeading)
                  .padding(12)
              }
              .frame(minHeight: 140, maxHeight: 260)
              .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
              .accessibilityLabel("Server 日誌")
              .padding(.top, 12)
            }
          }
        }
        .padding(.vertical, 4)
      }
    }
    .confirmationDialog(
      "下載並安裝模型？",
      isPresented: $confirmsDownload,
      titleVisibility: .visible
    ) {
      Button("下載模型") { modelLibrary.startDownload() }
      Button("取消", role: .cancel) {}
    } message: {
      Text("App 會將模型安裝到 \(modelLibrary.rootURL.path)。下載中斷後可以繼續。")
    }
    .confirmationDialog(
      "驗證並修復模型？",
      isPresented: $confirmsRepair,
      titleVisibility: .visible
    ) {
      Button("驗證並修復模型") {
        if let repairTarget { modelLibrary.startRepair(repairTarget) }
      }
      Button("取消", role: .cancel) {}
    } message: {
      Text("App 會先完整驗證模型，然後只重新下載缺少或損壞的資料。")
    }
    .confirmationDialog(
      "重新下載模型？",
      isPresented: $confirmsReinstall,
      titleVisibility: .visible
    ) {
      Button("重新下載模型", role: .destructive) {
        if let reinstallTarget { modelLibrary.reinstall(reinstallTarget) }
      }
      Button("取消", role: .cancel) {}
    } message: {
      Text("App 會將目前的損壞模型移到垃圾桶，然後重新下載完整模型。")
    }
  }

  private var serverHeader: some View {
    HStack(spacing: 12) {
      Image(systemName: server.state.symbol)
        .foregroundStyle(statusColor)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 2) {
        Text(server.state.label).font(.headline)
        Text(configuration.baseURL?.absoluteString ?? "Base URL 無效")
          .font(.caption.monospaced())
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
      }
      .accessibilityElement(children: .combine)
      Spacer()
      if server.isActive {
        Button("停止 server", action: server.stop)
          .keyboardShortcut(".", modifiers: .command)
      } else {
        Button("啟動 server") { server.start(configuration) }
          .buttonStyle(.borderedProminent)
          .disabled(!modelLibrary.canUseModel(at: configuration.modelPath))
          .keyboardShortcut(.return, modifiers: .command)
      }
    }
  }

  private var onboardingPanel: some View {
    VStack(alignment: .leading, spacing: 24) {
      VStack(alignment: .leading, spacing: 6) {
        Text(modelLibrary.hasPartialDownload ? "繼續安裝模型" : "安裝模型")
          .font(.title2.bold())
        Text("下載模型或選擇現有的模型資料夾。完成後即可啟動本機 server。")
          .foregroundStyle(.secondary)
      }

      preflightPanel

      HStack(spacing: 12) {
        Button("選擇模型資料夾") { chooseModelDirectory() }
          .disabled(modelLibrary.isBusy)
        Button(modelLibrary.hasPartialDownload ? "繼續下載" : "下載模型") {
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
      Text("下載前檢查")
        .font(.headline)
      ForEach(modelLibrary.preflightChecks) { check in
        HStack(alignment: .top, spacing: 10) {
          Image(systemName: preflightSymbol(check.status))
            .foregroundStyle(preflightColor(check.status))
            .accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 2) {
            Text(check.title).fontWeight(.medium)
            Text(check.detail)
              .font(.caption)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
        }
        .accessibilityElement(children: .combine)
      }
    }
  }

  private var modelPanel: some View {
    GroupBox("模型") {
      VStack(alignment: .leading, spacing: 12) {
        HStack(spacing: 12) {
          if modelLibrary.isScanning {
            ProgressView().controlSize(.small)
          } else {
            Image(systemName: selectedModel == nil ? "externaldrive" : "checkmark.circle.fill")
              .foregroundStyle(selectedModel == nil ? Color.secondary : Color.green)
              .accessibilityHidden(true)
          }

          Picker("Installed model", selection: $configuration.modelPath) {
            ForEach(modelLibrary.usableModels) { model in
              Text("\(model.name) · \(formattedBytes(model.size))").tag(model.url.path)
            }
          }
          .labelsHidden()
          .frame(maxWidth: 420)
          Spacer()
          Button("在 Finder 中顯示") {
            if let selectedModel { modelLibrary.reveal(selectedModel.url) }
          }
          Button("完整驗證模型") {
            if let selectedModel { modelLibrary.startVerification(selectedModel) }
          }
          .disabled(server.isActive || modelLibrary.isBusy)
          Button("選擇其他資料夾") { chooseModelDirectory() }
            .disabled(server.isActive || modelLibrary.isBusy)
        }

        Text(modelLibrary.rootURL.path)
          .font(.caption.monospaced())
          .foregroundStyle(.secondary)
          .lineLimit(1)
          .truncationMode(.middle)
          .textSelection(.enabled)

        if modelLibrary.verificationModelPath == configuration.modelPath,
          let issues = modelLibrary.verificationIssues
        {
          if issues.isEmpty {
            Label("已通過完整驗證", systemImage: "checkmark.seal.fill")
              .foregroundStyle(.green)
          } else {
            Label("有 \(issues.count) 個檔案需要修復", systemImage: "exclamationmark.triangle.fill")
              .foregroundStyle(.red)
            Button("驗證並修復模型") {
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
              "\(formattedBytes(progress.completedBytes)) / \(formattedBytes(progress.totalBytes))")
            if let speed = progress.bytesPerSecond, speed > 0 {
              Text("\(formattedBytes(UInt64(speed)))/s")
            }
            if let seconds = progress.estimatedSecondsRemaining, seconds.isFinite {
              Text("約剩 \(formattedDuration(seconds))")
            }
          }
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
        } else {
          ProgressView()
            .accessibilityLabel(modelLibrary.operationPhase.label)
        }
        Button("停止目前操作") { modelLibrary.cancelOperation() }
          .disabled(modelLibrary.operationPhase == .cancelling)
      }
      .padding(8)
    }
  }

  private var damagedModelsPanel: some View {
    GroupBox("需要處理的模型") {
      VStack(alignment: .leading, spacing: 16) {
        ForEach(modelLibrary.damagedModels) { model in
          HStack(alignment: .top, spacing: 12) {
            Label(
              "\(model.name)：缺少或大小錯誤的檔案有 \(model.quickIssues.count) 個",
              systemImage: "exclamationmark.triangle.fill"
            )
            .foregroundStyle(.red)
            Spacer()
            Button("在 Finder 中顯示") { modelLibrary.reveal(model.url) }
            Button("驗證並修復模型") {
              repairTarget = model
              confirmsRepair = true
            }
            .disabled(modelLibrary.isBusy || server.isActive)
          }
        }
        ForEach(modelLibrary.invalidModelURLs, id: \.path) { url in
          HStack(alignment: .top, spacing: 12) {
            Label(
              "\(url.lastPathComponent)：manifest 無法讀取",
              systemImage: "xmark.octagon.fill"
            )
            .foregroundStyle(.red)
            Spacer()
            Button("在 Finder 中顯示") { modelLibrary.reveal(url) }
            Button("重新下載模型") {
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
    Form {
      Section("API server") {
        LabeledContent("Host") {
          TextField("127.0.0.1", text: $configuration.host)
            .textFieldStyle(.roundedBorder)
        }
        LabeledContent("Port") {
          TextField("8000", value: $configuration.port, format: .number.grouping(.never))
            .textFieldStyle(.roundedBorder)
            .frame(width: 100)
        }
        LabeledContent("API key") {
          SecureField("本機使用時可留空", text: $configuration.apiKey)
            .textFieldStyle(.roundedBorder)
        }
        LabeledContent("Model ID") {
          TextField("deepseek-v4-flash-0731", text: $configuration.publicModel)
            .textFieldStyle(.roundedBorder)
        }
      }

      Section("Runtime") {
        integerField(
          "Slots", hint: "Active Parameters Cache 的 routed expert 數量。推薦值是 1024。",
          value: $configuration.slots)
        integerField(
          "Read workers", hint: "同時讀取 expert blob 的工作數量。推薦值是 4。", value: $configuration.readWorkers)
        integerField(
          "Prefill step size", hint: "每次處理的 prompt token 數量。推薦值是 32。",
          value: $configuration.prefillStepSize)
        Toggle("使用 BF16 KV cache", isOn: $configuration.bf16KVCache)
      }

      Section("生成") {
        integerField(
          "Max tokens", hint: "每次 request 的預設 token 上限。", value: $configuration.defaultMaxTokens)
        doubleField("Temperature", hint: "0 會產生穩定結果。", value: $configuration.defaultTemperature)
        doubleField("Top P", hint: "推薦值是 1。", value: $configuration.defaultTopP)
      }
    }
    .formStyle(.grouped)
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
      TextField(label, value: value, format: .number.grouping(.never))
        .labelsHidden()
        .textFieldStyle(.roundedBorder)
        .frame(width: 100)
    } label: {
      SettingLabel(label, hint: hint)
    }
  }

  @ViewBuilder
  private func doubleField(_ label: String, hint: String, value: Binding<Double>) -> some View {
    LabeledContent {
      TextField(label, value: value, format: .number.precision(.fractionLength(0...6)))
        .labelsHidden()
        .textFieldStyle(.roundedBorder)
        .frame(width: 100)
    } label: {
      SettingLabel(label, hint: hint)
    }
  }

  private func chooseModelDirectory() {
    let panel = NSOpenPanel()
    panel.title = "選擇模型資料夾"
    panel.prompt = "選擇資料夾"
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
    if totalMinutes < 60 { return "\(totalMinutes) 分鐘" }
    return "\(totalMinutes / 60) 小時 \(totalMinutes % 60) 分鐘"
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
  let performance: LivePerformance

  var body: some View {
    HStack(spacing: 12) {
      MetricCard(
        title: "Token 生成速度",
        value: performance.hasStatus
          ? "\(performance.tokensPerSecond.formatted(.number.precision(.fractionLength(2)))) token/sec"
          : "—",
        detail: performance.hasStatus
          ? (performance.generating ? "正在生成" : "最近一次生成") : "尚無資料",
        systemImage: "gauge.with.dots.needle.67percent"
      )
      MetricCard(
        title: "記憶體使用量",
        value: performance.memoryBytes > 0 ? formattedBytes(performance.memoryBytes) : "—",
        detail: "Python server 目前常駐的記憶體",
        systemImage: "memorychip"
      )
      MetricCard(
        title: "SSD 讀取速度",
        value: performance.hasStatus
          ? "\(formattedBytes(UInt64(max(0, performance.ssdBytesPerSecond))))/s"
          : "—",
        detail: "最近一秒的 expert blob 讀取量",
        systemImage: "externaldrive"
      )
      MetricCard(
        title: "Active Parameters Cache",
        value: performance.hasStatus
          ? performance.cacheHitRate.formatted(.percent.precision(.fractionLength(1)))
          : "—",
        detail: performance.hasStatus
          ? "\(performance.cacheResidentSlots) / \(performance.cacheCapacitySlots) slots"
          : "尚無資料",
        systemImage: "shippingbox"
      )
    }
    .accessibilityElement(children: .contain)
  }

  private func formattedBytes(_ bytes: UInt64) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .memory)
  }
}

private struct MetricCard: View {
  let title: String
  let value: String
  let detail: String
  let systemImage: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Label(title, systemImage: systemImage)
        .font(.caption)
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
      Text(value)
        .font(.title3.bold())
        .monospacedDigit()
        .lineLimit(1)
        .minimumScaleFactor(0.75)
      Text(detail)
        .font(.caption2)
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
    }
    .padding(12)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("\(title)：\(value)。\(detail)")
  }
}

private struct SettingLabel: View {
  let title: String
  let hint: String

  init(_ title: String, hint: String) {
    self.title = title
    self.hint = hint
  }

  var body: some View {
    HStack(spacing: 4) {
      Text(title)
      Image(systemName: "info.circle")
        .foregroundStyle(.secondary)
        .accessibilityHidden(true)
    }
    .help(hint)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(title)
    .accessibilityHint(hint)
  }
}

private struct ChatView: View {
  let configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  @State private var messages: [ChatMessage] = []
  @State private var input = ""
  @State private var thinkingMode = "chat"
  @State private var isSending = false
  @State private var errorMessage: String?
  @State private var lastMetrics: ChatMetrics?
  @State private var showingClearConfirmation = false

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        VStack(alignment: .leading, spacing: 2) {
          Text("測試對話").font(.title2.bold())
          Text("這個畫面只用來確認本機 server 可以回應。")
            .foregroundStyle(.secondary)
        }
        Spacer()
        Button {
          showingClearConfirmation = true
        } label: {
          Label("清空對話", systemImage: "trash")
        }
        .disabled(messages.isEmpty || isSending)
        Picker("Thinking mode", selection: $thinkingMode) {
          Text("Chat").tag("chat")
          Text("Thinking").tag("thinking")
        }
        .pickerStyle(.segmented)
        .frame(width: 220)
        if let lastMetrics {
          Label(
            "\(lastMetrics.tokensPerSecond.formatted(.number.precision(.fractionLength(2)))) token/sec",
            systemImage: "gauge.with.dots.needle.67percent"
          )
          .monospacedDigit()
          .help("\(lastMetrics.completionTokens) completion tokens")
        }
      }

      GroupBox {
        ScrollViewReader { scroll in
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
              if messages.isEmpty {
                ContentUnavailableView(
                  "尚無測試訊息",
                  systemImage: "bubble.left",
                  description: Text("啟動 server，然後送出一則訊息。")
                )
                .frame(maxWidth: .infinity, minHeight: 280)
              } else {
                ForEach(messages) { message in
                  VStack(alignment: .leading, spacing: 6) {
                    Text(message.role == "user" ? "你" : "DeepSeek")
                      .font(.caption.bold())
                      .foregroundStyle(.secondary)
                    if !message.reasoningContent.isEmpty {
                      VStack(alignment: .leading, spacing: 4) {
                        Text("思考過程")
                          .font(.caption.bold())
                          .foregroundStyle(.secondary)
                        Text(message.reasoningContent)
                          .foregroundStyle(.secondary)
                          .textSelection(.enabled)
                      }
                    }
                    if !message.content.isEmpty {
                      Text(message.content)
                        .textSelection(.enabled)
                    } else if message.role == "assistant" && message.reasoningContent.isEmpty {
                      ProgressView("正在生成")
                        .controlSize(.small)
                    }
                  }
                  .padding(12)
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
          .accessibilityLabel("錯誤：\(errorMessage)")
      }

      HStack(alignment: .bottom, spacing: 12) {
        TextEditor(text: $input)
          .font(.body)
          .frame(minHeight: 72, maxHeight: 140)
          .overlay(
            RoundedRectangle(cornerRadius: 8)
              .stroke(Color.secondary.opacity(0.35))
          )
          .accessibilityLabel("測試訊息")
        Button {
          send()
        } label: {
          if isSending {
            ProgressView().controlSize(.small)
          } else {
            Label("送出訊息", systemImage: "paperplane.fill")
          }
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(isSending || server.state != .running)
        .keyboardShortcut(.return, modifiers: .command)
      }

      if server.state != .running {
        Text("請先在 Server 畫面啟動 server。")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
    .padding(4)
    .confirmationDialog(
      "要清空測試對話嗎？",
      isPresented: $showingClearConfirmation,
      titleVisibility: .visible
    ) {
      Button("清空對話", role: .destructive) {
        messages.removeAll()
        input = ""
        errorMessage = nil
        lastMetrics = nil
      }
      Button("取消", role: .cancel) {}
    } message: {
      Text("App 只會清除本機測試對話。Server 和其他 API client 不受影響。")
    }
  }

  private var streamedCharacterCount: Int {
    guard let message = messages.last else { return 0 }
    return message.content.count + message.reasoningContent.count
  }

  private func send() {
    let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, let baseURL = configuration.baseURL else { return }
    let userMessage = ChatMessage(role: "user", content: text)
    messages.append(userMessage)
    input = ""
    errorMessage = nil
    lastMetrics = nil
    isSending = true
    let requestMessages = messages
    let assistantID = UUID()
    messages.append(ChatMessage(id: assistantID, role: "assistant", content: ""))
    Task {
      do {
        lastMetrics = try await ChatClient.stream(
          messages: requestMessages,
          baseURL: baseURL,
          apiKey: configuration.apiKey,
          model: configuration.publicModel,
          thinkingMode: thinkingMode
        ) { delta in
          guard let index = messages.firstIndex(where: { $0.id == assistantID }) else { return }
          messages[index].append(delta)
        }
      } catch {
        if let index = messages.firstIndex(where: { $0.id == assistantID }),
          messages[index].content.isEmpty,
          messages[index].reasoningContent.isEmpty
        {
          messages.remove(at: index)
        }
        errorMessage = "無法取得回應。\(error.localizedDescription)"
      }
      isSending = false
    }
  }
}
