import XCTest

@testable import DeepSeekV4SSDApp

final class ChatStreamDecoderTests: XCTestCase {
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
        line: #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_current_time","arguments":"{\"time_zone\":\"Asia/Taipei\"}"}}]}}]}"#
      ),
      .delta(ChatDelta(content: "", reasoningContent: "", toolCalls: [toolDelta]))
    )
    XCTAssertEqual(
      try ChatStreamDecoder.decode(
        line: #"data: {"choices":[],"usage":{"completion_tokens":12}}"#),
      .usage(completionTokens: 12)
    )
    XCTAssertEqual(try ChatStreamDecoder.decode(line: "data: [DONE]"), .done)
    XCTAssertNil(try ChatStreamDecoder.decode(line: ""))
    XCTAssertEqual(
      ChatMetrics(completionTokens: 12, elapsedSeconds: 2).tokensPerSecond,
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
      line: #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tai"}}]}}]}"#
    ) {
      message.append(delta)
    }
    XCTAssertEqual(
      message.toolCalls[0].function.arguments,
      #"{"time_zone":"Asia/Taipei"}Tai"#
    )
  }
}
