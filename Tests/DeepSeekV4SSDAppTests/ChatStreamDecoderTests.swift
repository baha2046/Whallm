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
    ] {
      if case .delta(let delta) = try ChatStreamDecoder.decode(line: line) {
        message.append(delta)
      }
    }
    XCTAssertEqual(message.reasoningContent, "plan")
    XCTAssertEqual(message.content, "Hello")
  }
}
