import AppKit
import Combine
import SwiftUI
import XCTest

@testable import DeepSeekV4SSDApp

final class ChatStreamDecoderTests: XCTestCase {
  func testChatModelSelectionUsesAliasAndRestoresSavedSelection() {
    let models = [
      CatalogModel(id: "deepseek-v4-flash-0731", alias: "work-model"),
      CatalogModel(id: "qwen3.8-flash-next-fp8", alias: nil),
    ]

    XCTAssertEqual(
      resolvedChatModelName(savedName: "deepseek-v4-flash-0731", models: models),
      "work-model"
    )
    XCTAssertEqual(
      resolvedChatModelName(savedName: "qwen3.8-flash-next-fp8", models: models),
      "qwen3.8-flash-next-fp8"
    )
    XCTAssertEqual(
      resolvedChatModelName(savedName: "missing", models: models),
      "work-model"
    )
    XCTAssertNil(resolvedChatModelName(savedName: "work-model", models: []))
  }

  func testDecoderPreservesThinkingAndAnswerDeltas() throws {
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}"#),
      .delta(ChatDelta(content: "", reasoningContent: "plan"))
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[{"delta":{"content":"Hello"}}]}"#),
      .delta(ChatDelta(content: "Hello", reasoningContent: ""))
    )
    let toolCall = ChatToolCall(
      id: "call_1",
      type: "function",
      function: .init(
        name: "get_current_time",
        arguments: #"{"time_zone":"Asia/Taipei"}"#
      )
    )
    let toolDelta = ChatToolCallDelta(
      index: 0,
      id: "call_1",
      type: "function",
      function: .init(
        name: "get_current_time",
        arguments: #"{"time_zone":"Asia/Taipei"}"#
      )
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line:
          #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_current_time","arguments":"{\"time_zone\":\"Asia/Taipei\"}"}}]}}]}"#
      ),
      .delta(ChatDelta(content: "", reasoningContent: "", toolCalls: [toolDelta]))
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[],"usage":{"prompt_tokens":34,"completion_tokens":12}}"#),
      .usage(promptTokens: 34, completionTokens: 12)
    )
    XCTAssertEqual(try ChatStreamDecoder.decode(line: "data: [DONE]"), .done)
    XCTAssertNil(try ChatStreamDecoder.decode(line: ""))
    XCTAssertEqual(
      ChatMetrics(
        promptTokens: 34,
        completionTokens: 12,
        elapsedSeconds: 3,
        firstTokenSeconds: 1
      ).tokensPerSecond,
      6
    )

    var message = ChatMessage(role: "assistant", content: "")
    for line in [
      #"data: {"choices":[{"delta":{"reasoning_content":"p"}}]}"#,
      #"data: {"choices":[{"delta":{"reasoning_content":"lan"}}]}"#,
      #"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
      #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_current_time","arguments":"{\"time_zone\":\"Asia/Taipei\"}"}}]}}]}"#,
    ] {
      if case .delta(let delta) = try ChatStreamDecoder.decode(line: line) {
        message.append(delta)
      }
    }
    XCTAssertEqual(message.reasoningContent, "plan")
    XCTAssertEqual(message.content, "Hello")
    XCTAssertEqual(message.toolCalls, [toolCall])

    XCTAssertThrowsError(
      try ChatStreamDecoder.decode(
        line: #"data: {"error":{"message":"Tool call 格式無效。"}}"#
      )
    ) { error in
      XCTAssertEqual(error.localizedDescription, "Tool call 格式無效。")
    }

    if case .delta(let delta) = try ChatStreamDecoder.decode(
      line:
        #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tai"}}]}}]}"#
    ) {
      message.append(delta)
    }
    XCTAssertEqual(
      message.toolCalls[0].function.arguments,
      #"{"time_zone":"Asia/Taipei"}Tai"#
    )
  }

  func testChatHistoryPersistenceRestoresLocalFields() throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let message = ChatMessage(
      role: "assistant",
      content: "answer",
      reasoningContent: "reason",
      toolCalls: [
        ChatToolCall(
          id: "call_1",
          type: "function",
          function: .init(name: "get_current_time", arguments: "{}")
        )
      ],
      modelName: "work-model"
    )

    ChatHistory.save([message], defaults: defaults)
    let restored = try XCTUnwrap(ChatHistory.load(defaults: defaults).first)

    XCTAssertEqual(restored.id, message.id)
    XCTAssertEqual(restored.role, message.role)
    XCTAssertEqual(restored.content, message.content)
    XCTAssertEqual(restored.reasoningContent, message.reasoningContent)
    XCTAssertEqual(restored.toolCalls, message.toolCalls)
    XCTAssertEqual(restored.modelName, "work-model")
  }

  func testOldChatHistoryLoadsWithoutAModelName() throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let id = UUID()
    defaults.set(
      try JSONSerialization.data(
        withJSONObject: [
          [
            "id": id.uuidString,
            "role": "assistant",
            "content": "old answer",
            "reasoningContent": "",
            "toolCalls": [],
          ]
        ]),
      forKey: "chatMessages"
    )

    let message = try XCTUnwrap(ChatHistory.load(defaults: defaults).first)

    XCTAssertEqual(message.id, id)
    XCTAssertNil(message.modelName)
  }

  @MainActor
  func testGenerationContinuesAfterLeavingAndReturningToChat() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let session = ChatSession(
      defaults: defaults,
      stream: { _, _, _, _, _, receive in
        receive(ChatDelta(content: "first", reasoningContent: ""))
        try await Task.sleep(for: .milliseconds(500))
        receive(ChatDelta(content: " second", reasoningContent: ""))
      }
    )
    let visibility = ChatVisibility()
    let hostingView = NSHostingView(
      rootView: ChatNavigationHarness(
        visibility: visibility,
        server: ServerController(),
        session: session
      ))
    hostingView.frame = NSRect(x: 0, y: 0, width: 1_000, height: 700)
    hostingView.layoutSubtreeIfNeeded()

    XCTAssertTrue(
      session.send(
        text: "Hello",
        configuration: .localDefault,
        model: "test-model",
        thinkingMode: "chat",
        language: .english
      ))
    try await Task.sleep(for: .milliseconds(100))
    XCTAssertEqual(session.messages.last?.content, "first")

    visibility.showsChat = false
    try await Task.sleep(for: .milliseconds(100))
    hostingView.layoutSubtreeIfNeeded()
    visibility.showsChat = true
    try await Task.sleep(for: .milliseconds(100))
    hostingView.layoutSubtreeIfNeeded()
    XCTAssertTrue(session.isSending)
    try await Task.sleep(for: .milliseconds(400))

    XCTAssertEqual(session.messages.last?.content, "first second")
    XCTAssertFalse(session.isSending)
  }

  @MainActor
  func testRapidStreamingCoalescesMessagePublications() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let session = ChatSession(
      defaults: defaults,
      stream: { _, _, _, _, _, receive in
        for _ in 0..<80 {
          receive(ChatDelta(content: "x", reasoningContent: ""))
        }
      }
    )
    var messagePublicationCount = 0
    let observation = session.$messages.dropFirst().sink { _ in
      messagePublicationCount += 1
    }
    defer { observation.cancel() }

    XCTAssertTrue(
      session.send(
        text: "Hello",
        configuration: .localDefault,
        model: "test-model",
        thinkingMode: "chat",
        language: .english
      ))
    for _ in 0..<100 where session.isSending {
      try await Task.sleep(for: .milliseconds(10))
    }

    XCTAssertFalse(session.isSending)
    XCTAssertEqual(session.messages.last?.content, String(repeating: "x", count: 80))
    XCTAssertLessThanOrEqual(messagePublicationCount, 10)
  }

  @MainActor
  func testStreamingFlushesPendingTextBeforeReportingAnError() async throws {
    let suite = "ChatStreamDecoderTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let session = ChatSession(
      defaults: defaults,
      stream: { _, _, _, _, _, receive in
        receive(ChatDelta(content: "partial", reasoningContent: ""))
        throw NSError(domain: "ChatStreamDecoderTests", code: 1)
      }
    )

    XCTAssertTrue(
      session.send(
        text: "Hello",
        configuration: .localDefault,
        model: "test-model",
        thinkingMode: "chat",
        language: .english
      ))
    for _ in 0..<100 where session.isSending {
      try await Task.sleep(for: .milliseconds(10))
    }

    XCTAssertFalse(session.isSending)
    XCTAssertEqual(session.messages.last?.content, "partial")
    XCTAssertNotNil(session.errorMessage)
  }
}

@MainActor
private final class ChatVisibility: ObservableObject {
  @Published var showsChat = true
}

private struct ChatNavigationHarness: View {
  @ObservedObject var visibility: ChatVisibility
  @ObservedObject var server: ServerController
  @ObservedObject var session: ChatSession

  var body: some View {
    Group {
      if visibility.showsChat {
        ChatView(
          configuration: .localDefault,
          server: server,
          session: session,
          language: .english
        )
      } else {
        Text("Other")
      }
    }
  }
}
