import Foundation

enum RepackPlanner {
  static func makePlan(index: CheckpointIndex, tensors: [String: SafeTensor]) throws -> RepackPlan {
    guard tensors.count == index.weightMap.count else {
      throw RepackError.invalidIndex(
        "resolved \(tensors.count) tensors; index contains \(index.weightMap.count)"
      )
    }

    let expectedExpertNames = Set(expectedExperts().map(\.name))
    let actualExpertNames = Set(tensors.keys.filter(isMainExpert))
    let missing = expectedExpertNames.subtracting(actualExpertNames)
    let unexpected = actualExpertNames.subtracting(expectedExpertNames)
    if let name = missing.sorted().first {
      throw RepackError.invalidPlan("missing routed expert tensor \(name)")
    }
    if let name = unexpected.sorted().first {
      throw RepackError.invalidPlan("unexpected routed expert tensor \(name)")
    }

    var copies: [TensorCopy] = []
    copies.reserveCapacity(tensors.count)
    var commonTensors: [InstalledTensor] = []
    var commonOffset: UInt64 = 0

    let common = tensors.values
      .filter { !$0.name.hasPrefix("mtp.") && !isMainExpert($0.name) }
      .sorted { $0.name < $1.name }
    for tensor in common {
      commonOffset = aligned(commonOffset, to: ModelContract.commonAlignment)
      commonTensors.append(
        InstalledTensor(
          name: tensor.name,
          dtype: tensor.dtype,
          shape: tensor.shape,
          offset: commonOffset,
          length: tensor.length
        ))
      copies.append(
        TensorCopy(
          tensor: tensor.name,
          sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset,
          length: tensor.length,
          destinationFile: "common.bin",
          destinationOffset: commonOffset
        ))
      commonOffset += tensor.length
    }
    guard !commonTensors.isEmpty else {
      throw RepackError.invalidPlan("main model has no common tensors")
    }

    for expected in expectedExperts() {
      guard let tensor = tensors[expected.name] else {
        throw RepackError.invalidPlan("missing routed expert tensor \(expected.name)")
      }
      guard tensor.dtype == expected.region.dtype,
        tensor.shape == expected.region.shape,
        tensor.length == expected.region.length
      else {
        throw RepackError.invalidPlan(
          "invalid layout for \(tensor.name): \(tensor.dtype) \(tensor.shape) \(tensor.length) bytes"
        )
      }
      copies.append(
        TensorCopy(
          tensor: tensor.name,
          sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset,
          length: tensor.length,
          destinationFile: layerPath(expected.layer),
          destinationOffset: UInt64(expected.expert) * ModelContract.expertBlobSize
            + expected.region.offset
        ))
    }

    let layerSize = UInt64(ModelContract.expertCount) * ModelContract.expertBlobSize
    let files =
      [PlannedFile(path: "common.bin", size: commonOffset)]
      + (0..<ModelContract.layerCount).map { PlannedFile(path: layerPath($0), size: layerSize) }
    let checkpointTensorBytes = copies.reduce(UInt64(0)) { $0 + $1.length }

    return RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: checkpointTensorBytes,
      files: files,
      commonTensors: commonTensors,
      expertRegions: ModelContract.expertRegions,
      copies: copies
    )
  }

  private struct ExpectedExpert {
    let name: String
    let layer: Int
    let expert: Int
    let region: ExpertRegion
  }

  private static func expectedExperts() -> [ExpectedExpert] {
    var result: [ExpectedExpert] = []
    result.reserveCapacity(
      ModelContract.layerCount * ModelContract.expertCount * ModelContract.expertRegions.count)
    for layer in 0..<ModelContract.layerCount {
      for expert in 0..<ModelContract.expertCount {
        for region in ModelContract.expertRegions {
          result.append(
            ExpectedExpert(
              name: "layers.\(layer).ffn.experts.\(expert).\(region.name)",
              layer: layer,
              expert: expert,
              region: region
            ))
        }
      }
    }
    return result
  }

  private static func isMainExpert(_ name: String) -> Bool {
    name.hasPrefix("layers.") && name.contains(".ffn.experts.")
  }

  private static func layerPath(_ layer: Int) -> String {
    String(format: "experts/layer_%02d.bin", layer)
  }
}
