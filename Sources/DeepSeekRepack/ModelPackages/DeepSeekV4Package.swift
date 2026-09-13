import Foundation

struct DeepSeekV4Package: ModelPackage {
  let verifiesInstallation = false
  let installationLabel: String? = nil
  var companions: [CompanionFile] { ModelContract.companions }
  func installedBytes() async throws -> UInt64 { try await makeRepackPlan().installedBytes }
  func makeRepackPlan() async throws -> RepackPlan {
    try await DeepSeekV4Checkpoint().makeRepackPlan()
  }
  func repack(plan: RepackPlan, to output: URL, progress: RepackProgressHandler?) async throws
    -> InstalledManifest
  { try await DeepSeekV4Checkpoint().repack(plan: plan, to: output, progress: progress) }
  func install(to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await DeepSeekV4Checkpoint().repack(to: output, progress: progress)
  }
  func repair(at output: URL, invalidFiles: Set<String>, progress: RepackProgressHandler?)
    async throws -> InstalledManifest
  { try await DeepSeekV4Checkpoint().repair(at: output, invalidFiles: invalidFiles, progress: progress) }
  func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    try ModelContract.validate(manifest)
  }
}


extension ModelContract {
  static func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    guard manifest.formatVersion == 1,
      manifest.modelID == ModelContract.modelID,
      manifest.revision == ModelContract.revision,
      manifest.layerCount == ModelContract.layerCount,
      manifest.expertCount == ModelContract.expertCount,
      manifest.selectedExpertCount == ModelContract.selectedExpertCount,
      manifest.expertBlobSize == ModelContract.expertBlobSize,
      manifest.expertRegions == ModelContract.expertRegions
    else {
      throw RepackError.incompatibleModel(
        "installed manifest does not match the pinned model contract")
    }
    let mainPaths = Set(
      [
        "common.bin", "config.json", "generation_config.json", "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json", "encoding/encoding_dsv4.py",
      ]
        + (0..<ModelContract.layerCount).map {
          String(format: "experts/layer_%02d.bin", $0)
        })
    let optionalMainPaths: Set<String> = ["inference/config.json"]
    let dsparkPaths = Set(
      ["dspark/common.bin", "inference/config.json"]
        + (0..<ModelContract.dsparkLayerCount).map {
          String(format: "dspark/experts/layer_%02d.bin", $0)
        })
    let actualPaths = Set(manifest.files.map(\.path))
    let requiredPaths = mainPaths.union(manifest.dspark == nil ? [] : dsparkPaths)
    let allowedPaths = requiredPaths.union(optionalMainPaths)
    guard requiredPaths.isSubset(of: actualPaths),
      actualPaths.isSubset(of: allowedPaths),
      manifest.files.count == actualPaths.count,
      !manifest.commonTensors.isEmpty
    else {
      throw RepackError.invalidPlan("installed manifest has an incomplete file set")
    }
    guard let commonSize = manifest.files.first(where: { $0.path == "common.bin" })?.size else {
      throw RepackError.invalidPlan("installed manifest has no common tensor file")
    }
    var commonNames = Set<String>()
    for tensor in manifest.commonTensors {
      guard !tensor.name.isEmpty,
        commonNames.insert(tensor.name).inserted,
        tensor.offset <= commonSize,
        tensor.length <= commonSize - tensor.offset
      else {
        throw RepackError.invalidPlan("invalid common tensor \(tensor.name)")
      }
    }
    if let dspark = manifest.dspark {
      guard dspark.layerCount == ModelContract.dsparkLayerCount,
        dspark.blockSize == ModelContract.dsparkBlockSize,
        dspark.noiseTokenID == ModelContract.dsparkNoiseTokenID,
        dspark.targetLayerIDs == ModelContract.dsparkTargetLayerIDs,
        dspark.markovRank == ModelContract.dsparkMarkovRank,
        !dspark.commonTensors.isEmpty,
        let dsparkCommonSize = manifest.files.first(where: {
          $0.path == "dspark/common.bin"
        })?.size
      else {
        throw RepackError.invalidPlan("installed manifest has an invalid DSpark contract")
      }
      var dsparkNames = Set<String>()
      for tensor in dspark.commonTensors {
        guard tensor.name.hasPrefix("mtp."),
          dsparkNames.insert(tensor.name).inserted,
          tensor.offset <= dsparkCommonSize,
          tensor.length <= dsparkCommonSize - tensor.offset
        else {
          throw RepackError.invalidPlan("invalid DSpark common tensor \(tensor.name)")
        }
      }
      let layerSize = UInt64(ModelContract.expertCount) * ModelContract.expertBlobSize
      for layer in 0..<ModelContract.dsparkLayerCount {
        let path = String(format: "dspark/experts/layer_%02d.bin", layer)
        guard manifest.files.first(where: { $0.path == path })?.size == layerSize else {
          throw RepackError.invalidPlan("installed DSpark expert layer has an invalid size")
        }
      }
    }
    return manifest
  }

}
