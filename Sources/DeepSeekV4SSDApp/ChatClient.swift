import Foundation

struct ChatToolCall: Codable, Equatable, Identifiable, Sendable {
  struct Function: Codable, Equatable, Sendable {
    var name: String
    var arguments: String
  }

  var id: String
  var type: String
  var function: Function
}

struct ChatToolCallDelta: Decodable, Equatable, Sendable {
  struct Function: Decodable, Equatable, Sendable {
    let name: String?
    let arguments: String?
  }

  let index: Int
  let id: String?
  let type: String?
  let function: Function
}

struct ChatMessage: Encodable, Identifiable, Sendable {
  let id: UUID
  let role: String
  var content: String
  var reasoningContent: String
  var toolCalls: [ChatToolCall]

  init(
    id: UUID = UUID(),
    role: String,
    content: String,
    reasoningContent: String = "",
    toolCalls: [ChatToolCall] = []
  ) {
    self.id = id
    self.role = role
    self.content = content
    self.reasoningContent = reasoningContent
    self.toolCalls = toolCalls
  }

  enum CodingKeys: String, CodingKey {
    case role, content
    case toolCalls = "tool_calls"
  }

  func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(role, forKey: .role)
    try values.encode(content, forKey: .content)
    if !toolCalls.isEmpty {
      try values.encode(toolCalls, forKey: .toolCalls)
    }
  }

  mutating func append(_ delta: ChatDelta) {
    content += delta.content
    reasoningContent += delta.reasoningContent
    for update in delta.toolCalls {
      if toolCalls.indices.contains(update.index) {
        if let id = update.id { toolCalls[update.index].id = id }
        if let type = update.type { toolCalls[update.index].type = type }
        if let name = update.function.name {
          toolCalls[update.index].function.name += name
        }
        toolCalls[update.index].function.arguments += update.function.arguments ?? ""
      } else if update.index == toolCalls.count,
        let id = update.id,
        let type = update.type,
        let name = update.function.name
      {
        toolCalls.append(
          ChatToolCall(
            id: id,
            type: type,
            function: .init(name: name, arguments: update.function.arguments ?? "")
          )
        )
      }
    }
  }
}

struct ChatDelta: Decodable, Equatable, Sendable {
  let content: String
  let reasoningContent: String
  let toolCalls: [ChatToolCallDelta]

  init(content: String, reasoningContent: String, toolCalls: [ChatToolCallDelta] = []) {
    self.content = content
    self.reasoningContent = reasoningContent
    self.toolCalls = toolCalls
  }

  enum CodingKeys: String, CodingKey {
    case content
    case reasoningContent = "reasoning_content"
    case toolCalls = "tool_calls"
  }

  init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    content = try values.decodeIfPresent(String.self, forKey: .content) ?? ""
    reasoningContent =
      try values.decodeIfPresent(String.self, forKey: .reasoningContent) ?? ""
    toolCalls = try values.decodeIfPresent([ChatToolCallDelta].self, forKey: .toolCalls) ?? []
  }
}

enum ChatStreamEvent: Equatable {
  case delta(ChatDelta)
  case usage(completionTokens: Int)
  case done
}

enum ChatStreamDecoder {
  private struct Chunk: Decodable {
    struct Choice: Decodable {
      let delta: ChatDelta
    }

    struct Usage: Decodable {
      let completionTokens: Int

      enum CodingKeys: String, CodingKey {
        case completionTokens = "completion_tokens"
      }
    }

    struct Failure: Decodable {
      let message: String
    }

    let choices: [Choice]?
    let usage: Usage?
    let error: Failure?
  }

  static func decode(line: String) throws -> ChatStreamEvent? {
    guard line.hasPrefix("data:") else { return nil }
    let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
    if payload == "[DONE]" { return .done }
    let chunk = try JSONDecoder().decode(Chunk.self, from: Data(payload.utf8))
    if let error = chunk.error { throw ChatError(error.message) }
    if let usage = chunk.usage {
      return .usage(completionTokens: usage.completionTokens)
    }
    guard let delta = chunk.choices?.first?.delta else { return nil }
    return .delta(delta)
  }
}

struct ChatMetrics: Equatable, Sendable {
  let completionTokens: Int
  let elapsedSeconds: Double

  var tokensPerSecond: Double {
    elapsedSeconds > 0 ? Double(completionTokens) / elapsedSeconds : 0
  }
}

enum ChatClient {
  private struct Tool: Encodable {
    struct Function: Encodable {
      struct Parameters: Encodable {
        struct Property: Encodable {
          let type = "string"
          let description = "IANA time zone, such as Asia/Taipei."
        }

        let type = "object"
        let properties = ["time_zone": Property()]
        let required = ["time_zone"]
        let additionalProperties = false

        enum CodingKeys: String, CodingKey {
          case type, properties, required
          case additionalProperties = "additionalProperties"
        }
      }

      let name = "get_current_time"
      let description = "Get the current date and time for a time zone."
      let parameters = Parameters()
    }

    let type = "function"
    let function = Function()
  }

  private struct Payload: Encodable {
    struct StreamOptions: Encodable {
      let includeUsage = true

      enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
      }
    }

    let model: String
    let messages: [ChatMessage]
    let thinkingMode: String
    let tools: [Tool]?
    let stream = true
    let streamOptions = StreamOptions()

    enum CodingKeys: String, CodingKey {
      case model, messages, stream
      case thinkingMode = "thinking_mode"
      case streamOptions = "stream_options"
      case tools
    }
  }

  private struct ErrorResponse: Decodable {
    struct Detail: Decodable {
      let message: String
    }
    let error: Detail
  }

  static func stream(
    messages: [ChatMessage],
    baseURL: URL,
    apiKey: String,
    model: String,
    thinkingMode: String,
    enableTestTool: Bool,
    receive: @MainActor @escaping (ChatDelta) -> Void
  ) async throws -> ChatMetrics {
    var request = URLRequest(url: baseURL.appending(path: "v1/chat/completions"))
    request.httpMethod = "POST"
    request.timeoutInterval = 3_600
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if !apiKey.isEmpty {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONEncoder().encode(
      Payload(
        model: model,
        messages: messages,
        thinkingMode: thinkingMode,
        tools: enableTestTool ? [Tool()] : nil
      ))

    let clock = ContinuousClock()
    let start = clock.now
    let (bytes, response) = try await URLSession.shared.bytes(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw ChatError("Server 未傳回 HTTP response。")
    }
    guard (200..<300).contains(http.statusCode) else {
      var data = Data()
      for try await byte in bytes { data.append(byte) }
      let detail = try? JSONDecoder().decode(ErrorResponse.self, from: data)
      throw ChatError(detail?.error.message ?? "Server 傳回 HTTP \(http.statusCode)。")
    }

    var completionTokens = 0
    var firstDelta: ContinuousClock.Instant?
    for try await line in bytes.lines {
      guard let event = try ChatStreamDecoder.decode(line: line) else { continue }
      switch event {
      case .delta(let delta):
        if !delta.content.isEmpty || !delta.reasoningContent.isEmpty || !delta.toolCalls.isEmpty {
          firstDelta = firstDelta ?? clock.now
          await receive(delta)
        }
      case .usage(let tokens):
        completionTokens = tokens
      case .done:
        return ChatMetrics(
          completionTokens: completionTokens,
          elapsedSeconds: seconds(from: (firstDelta ?? start).duration(to: clock.now))
        )
      }
    }
    throw ChatError("Streaming response 在 [DONE] 前中斷。")
  }

  private static func seconds(from duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }
}

private struct ChatError: LocalizedError {
  let message: String

  init(_ message: String) {
    self.message = message
  }

  var errorDescription: String? { message }
}
