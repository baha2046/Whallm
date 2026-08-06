import Foundation

protocol CheckpointSource: Sendable {
  func data(path: String) async throws -> Data
  func data(path: String, range: Range<UInt64>) async throws -> Data
}

struct HuggingFaceSource: CheckpointSource {
  let modelID: String
  let revision: String
  let session: URLSession

  init(modelID: String, revision: String, session: URLSession = .shared) {
    self.modelID = modelID
    self.revision = revision
    self.session = session
  }

  func data(path: String) async throws -> Data {
    let (data, response) = try await session.data(for: request(path: path))
    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
      throw RepackError.badResponse("checkpoint request failed for \(path)")
    }
    return data
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    guard !range.isEmpty else { return Data() }
    var request = request(path: path)
    request.setValue(
      "bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
    let (data, response) = try await session.data(for: request)
    guard let http = response as? HTTPURLResponse, http.statusCode == 206 else {
      throw RepackError.badResponse("checkpoint did not honor the byte range for \(path)")
    }
    guard UInt64(data.count) == range.count else {
      throw RepackError.badResponse(
        "checkpoint returned \(data.count) bytes for \(path); expected \(range.count)"
      )
    }
    return data
  }

  private func request(path: String) -> URLRequest {
    let encodedPath = path.split(separator: "/").map(String.init).map {
      $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? $0
    }.joined(separator: "/")
    let url = URL(string: "https://huggingface.co/\(modelID)/resolve/\(revision)/\(encodedPath)")!
    var request = URLRequest(url: url)
    request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
    request.setValue("DeepSeekV4SSD/1", forHTTPHeaderField: "User-Agent")
    return request
  }
}

public struct DeepSeekV4Checkpoint: Sendable {
  private let source: any CheckpointSource

  public init() {
    source = HuggingFaceSource(modelID: ModelContract.modelID, revision: ModelContract.revision)
  }

  init(source: any CheckpointSource) {
    self.source = source
  }

  public func makeRepackPlan() async throws -> RepackPlan {
    let configData = try await source.data(path: "config.json")
    let config: ModelConfig
    do {
      config = try JSONDecoder().decode(ModelConfig.self, from: configData)
    } catch {
      throw RepackError.incompatibleModel("cannot decode model config: \(error)")
    }
    try ModelContract.validate(config)

    let indexData = try await source.data(path: "model.safetensors.index.json")
    let index = try CheckpointIndex.decode(indexData)
    let tensors = try await readTensors(index: index)
    return try RepackPlanner.makePlan(index: index, tensors: tensors)
  }

  public func repack(
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let plan = try await makeRepackPlan()
    return try await repack(plan: plan, to: output, progress: progress)
  }

  public func repack(
    plan: RepackPlan,
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    guard plan.formatVersion == 1,
      plan.modelID == ModelContract.modelID,
      plan.revision == ModelContract.revision,
      plan.layerCount == ModelContract.layerCount,
      plan.expertCount == ModelContract.expertCount,
      plan.selectedExpertCount == ModelContract.selectedExpertCount,
      plan.expertBlobSize == ModelContract.expertBlobSize,
      plan.expertRegions == ModelContract.expertRegions
    else {
      throw RepackError.incompatibleModel("repack plan does not match the pinned model contract")
    }
    return try await Repacker(source: source).run(plan: plan, output: output, progress: progress)
  }

  public func repair(
    at output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    _ = try InstalledModel.loadManifest(at: output)
    let plan = try await makeRepackPlan()
    return try await Repacker(source: source).repair(
      plan: plan,
      output: output,
      invalidFiles: invalidFiles,
      progress: progress
    )
  }

  private func readTensors(index: CheckpointIndex) async throws -> [String: SafeTensor] {
    var headers: [String: (base: UInt64, header: SafeTensorsHeader)] = [:]
    // ponytail: Header reads are serial. Add bounded parallel reads only if inspect latency matters.
    for shard in Set(index.weightMap.values).sorted() {
      let prefix = try await source.data(path: shard, range: 0..<8)
      let headerLength = try littleEndianUInt64(prefix)
      guard headerLength > 1, headerLength <= 64 * 1_024 * 1_024 else {
        throw RepackError.invalidSafeTensors("invalid header length \(headerLength) in \(shard)")
      }
      let data = try await source.data(path: shard, range: 8..<(8 + headerLength))
      headers[shard] = (8 + headerLength, try SafeTensorsHeader.decode(data))
    }

    var tensors: [String: SafeTensor] = [:]
    tensors.reserveCapacity(index.weightMap.count)
    for (name, shard) in index.weightMap {
      guard let sourceHeader = headers[shard], let entry = sourceHeader.header.entries[name] else {
        throw RepackError.invalidIndex("index tensor \(name) is missing from \(shard)")
      }
      tensors[name] = SafeTensor(
        name: name,
        sourceFile: shard,
        dtype: entry.dtype,
        shape: entry.shape,
        sourceOffset: sourceHeader.base + entry.dataStart,
        length: entry.dataEnd - entry.dataStart
      )
    }
    return tensors
  }
}
