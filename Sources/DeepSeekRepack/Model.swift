import Foundation

struct CompanionFile: Sendable {
  let source: String
  let destination: String
}

enum ModelContract {
  static let modelID = "deepseek-ai/DeepSeek-V4-Flash-0731"
  static let revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
  static let layerCount = 43
  static let expertCount = 256
  static let selectedExpertCount = 6
  static let hiddenSize = 4_096
  static let expertIntermediateSize = 2_048
  static let hashLayerCount = 3
  static let maximumContext = 1_048_576
  static let commonAlignment: UInt64 = 256
  static let companions = [
    CompanionFile(source: "config.json", destination: "config.json"),
    CompanionFile(source: "generation_config.json", destination: "generation_config.json"),
    CompanionFile(source: "tokenizer.json", destination: "tokenizer/tokenizer.json"),
    CompanionFile(
      source: "tokenizer_config.json", destination: "tokenizer/tokenizer_config.json"),
    CompanionFile(source: "encoding/encoding_dsv4.py", destination: "encoding/encoding_dsv4.py"),
  ]
  static let companionPaths = companions.map(\.destination)

  static let expertRegions: [ExpertRegion] = {
    var offset: UInt64 = 0
    return [
      makeRegion("w1.weight", dtype: "I8", shape: [2_048, 2_048], offset: &offset),
      makeRegion("w1.scale", dtype: "F8_E8M0", shape: [2_048, 128], offset: &offset),
      makeRegion("w2.weight", dtype: "I8", shape: [4_096, 1_024], offset: &offset),
      makeRegion("w2.scale", dtype: "F8_E8M0", shape: [4_096, 64], offset: &offset),
      makeRegion("w3.weight", dtype: "I8", shape: [2_048, 2_048], offset: &offset),
      makeRegion("w3.scale", dtype: "F8_E8M0", shape: [2_048, 128], offset: &offset),
    ]
  }()

  static let expertBlobSize = expertRegions.reduce(0) { $0 + $1.length }

  private static func makeRegion(
    _ name: String,
    dtype: String,
    shape: [Int],
    offset: inout UInt64
  ) -> ExpertRegion {
    let length = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
    let region = ExpertRegion(
      name: name, dtype: dtype, shape: shape, offset: offset, length: length)
    offset += length
    return region
  }

  static func validate(_ config: ModelConfig) throws {
    let actual: [(String, String, String)] = [
      ("architecture", config.architectures.joined(separator: ","), "DeepseekV4ForCausalLM"),
      ("expert_dtype", config.expertDType, "fp4"),
      ("hidden_size", String(config.hiddenSize), String(hiddenSize)),
      ("moe_intermediate_size", String(config.moeIntermediateSize), String(expertIntermediateSize)),
      ("n_routed_experts", String(config.routedExpertCount), String(expertCount)),
      ("n_shared_experts", String(config.sharedExpertCount), "1"),
      ("num_experts_per_tok", String(config.selectedExpertCount), String(selectedExpertCount)),
      ("num_hidden_layers", String(config.hiddenLayerCount), String(layerCount)),
      ("num_hash_layers", String(config.hashLayerCount), String(hashLayerCount)),
      ("max_position_embeddings", String(config.maximumContext), String(maximumContext)),
    ]
    if let mismatch = actual.first(where: { $0.1 != $0.2 }) {
      throw RepackError.incompatibleModel("\(mismatch.0) is \(mismatch.1); expected \(mismatch.2)")
    }
  }
}

struct ModelConfig: Decodable, Sendable {
  let architectures: [String]
  let expertDType: String
  let hiddenSize: Int
  let moeIntermediateSize: Int
  let routedExpertCount: Int
  let sharedExpertCount: Int
  let selectedExpertCount: Int
  let hiddenLayerCount: Int
  let hashLayerCount: Int
  let maximumContext: Int

  enum CodingKeys: String, CodingKey {
    case architectures
    case expertDType = "expert_dtype"
    case hiddenSize = "hidden_size"
    case moeIntermediateSize = "moe_intermediate_size"
    case routedExpertCount = "n_routed_experts"
    case sharedExpertCount = "n_shared_experts"
    case selectedExpertCount = "num_experts_per_tok"
    case hiddenLayerCount = "num_hidden_layers"
    case hashLayerCount = "num_hash_layers"
    case maximumContext = "max_position_embeddings"
  }
}

struct SafeTensor: Equatable, Sendable {
  let name: String
  let sourceFile: String
  let dtype: String
  let shape: [Int]
  let sourceOffset: UInt64
  let length: UInt64
}

public struct ExpertRegion: Codable, Equatable, Sendable {
  public let name: String
  public let dtype: String
  public let shape: [Int]
  public let offset: UInt64
  public let length: UInt64
}

public struct InstalledTensor: Codable, Equatable, Sendable {
  public let name: String
  public let dtype: String
  public let shape: [Int]
  public let offset: UInt64
  public let length: UInt64
}

public struct PlannedFile: Codable, Equatable, Sendable {
  public let path: String
  public let size: UInt64
}

public struct TensorCopy: Codable, Equatable, Sendable {
  public let tensor: String
  public let sourceFile: String
  public let sourceOffset: UInt64
  public let length: UInt64
  public let destinationFile: String
  public let destinationOffset: UInt64
}

public struct RepackPlan: Codable, Equatable, Sendable {
  public let formatVersion: Int
  public let modelID: String
  public let revision: String
  public let layerCount: Int
  public let expertCount: Int
  public let selectedExpertCount: Int
  public let expertBlobSize: UInt64
  public let checkpointTensorBytes: UInt64
  public let files: [PlannedFile]
  public let commonTensors: [InstalledTensor]
  public let expertRegions: [ExpertRegion]
  public let copies: [TensorCopy]

  public var installedBytes: UInt64 { files.reduce(0) { $0 + $1.size } }
}

public struct InstalledFile: Codable, Equatable, Sendable {
  public let path: String
  public let size: UInt64
  public let sha256: String
}

public struct InstalledManifest: Codable, Equatable, Sendable {
  public let formatVersion: Int
  public let modelID: String
  public let revision: String
  public let layerCount: Int
  public let expertCount: Int
  public let selectedExpertCount: Int
  public let expertBlobSize: UInt64
  public let files: [InstalledFile]
  public let commonTensors: [InstalledTensor]
  public let expertRegions: [ExpertRegion]
}

public struct RepackProgress: Sendable {
  public let copiedBytes: UInt64
  public let downloadedBytes: UInt64
  public let totalBytes: UInt64
}

public enum InstalledFileIssueKind: String, Equatable, Sendable {
  case missing
  case sizeMismatch
  case checksumMismatch
}

public struct InstalledFileIssue: Equatable, Sendable {
  public let path: String
  public let kind: InstalledFileIssueKind

  public init(path: String, kind: InstalledFileIssueKind) {
    self.path = path
    self.kind = kind
  }
}

public struct InstalledModelVerification: Sendable {
  public let manifest: InstalledManifest
  public let issues: [InstalledFileIssue]

  public var isValid: Bool { issues.isEmpty }
}

public struct VerificationProgress: Sendable {
  public let checkedBytes: UInt64
  public let totalBytes: UInt64
}

public enum RepackError: Error, CustomStringConvertible, Equatable {
  case badResponse(String)
  case incompatibleModel(String)
  case invalidIndex(String)
  case invalidSafeTensors(String)
  case invalidPlan(String)
  case destinationExists(String)
  case insufficientStorage(required: UInt64, available: Int64)

  public var description: String {
    switch self {
    case .badResponse(let message), .incompatibleModel(let message), .invalidIndex(let message),
      .invalidSafeTensors(let message), .invalidPlan(let message):
      return message
    case .destinationExists(let path):
      return "destination already exists: \(path)"
    case .insufficientStorage(let required, let available):
      return "insufficient storage: need \(required) bytes; \(available) bytes are available"
    }
  }
}

func aligned(_ value: UInt64, to alignment: UInt64) -> UInt64 {
  (value + alignment - 1) / alignment * alignment
}
