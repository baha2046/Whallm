import Foundation
import XCTest

@testable import DeepSeekRepack

final class DeepSeekRepackTests: XCTestCase {
  func testPlannerCreatesCanonicalExpertLayout() throws {
    let fixture = makePlannerFixture()
    let plan = try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors)

    XCTAssertEqual(plan.expertBlobSize, 13_369_344)
    XCTAssertEqual(plan.files.count, 44)
    XCTAssertEqual(plan.copies.count, 43 * 256 * 6 + 1)
    XCTAssertEqual(plan.files[1].path, "experts/layer_00.bin")
    XCTAssertEqual(plan.files[1].size, 256 * 13_369_344)

    let first = try XCTUnwrap(
      plan.copies.first {
        $0.tensor == "layers.0.ffn.experts.0.w1.weight"
      })
    XCTAssertEqual(first.destinationFile, "experts/layer_00.bin")
    XCTAssertEqual(first.destinationOffset, 0)

    let secondExpert = try XCTUnwrap(
      plan.copies.first {
        $0.tensor == "layers.0.ffn.experts.1.w1.weight"
      })
    XCTAssertEqual(secondExpert.destinationOffset, 13_369_344)
  }

  func testPlannerRejectsMissingExpertTensor() throws {
    var fixture = makePlannerFixture()
    let missing = "layers.42.ffn.experts.255.w3.scale"
    fixture.tensors.removeValue(forKey: missing)
    fixture.index = CheckpointIndex(
      totalSize: fixture.index.totalSize,
      weightMap: fixture.index.weightMap.filter { $0.key != missing }
    )

    XCTAssertThrowsError(try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors))
    {
      XCTAssertTrue(String(describing: $0).contains("missing routed expert tensor"))
    }
  }

  func testPlannerRejectsInvalidExpertShape() throws {
    var fixture = makePlannerFixture()
    let name = "layers.0.ffn.experts.0.w1.weight"
    let original = try XCTUnwrap(fixture.tensors[name])
    fixture.tensors[name] = SafeTensor(
      name: name,
      sourceFile: original.sourceFile,
      dtype: original.dtype,
      shape: [1, 1],
      sourceOffset: original.sourceOffset,
      length: original.length
    )

    XCTAssertThrowsError(try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors))
    {
      XCTAssertTrue(String(describing: $0).contains("invalid layout"))
    }
  }

  func testPlannerAddsOptionalDSparkLayout() throws {
    let fixture = makePlannerFixture(includeDSpark: true)
    let plan = try RepackPlanner.makePlan(
      index: fixture.index,
      tensors: fixture.tensors,
      includeDSpark: true
    )

    XCTAssertEqual(plan.files.count, 48)
    XCTAssertEqual(plan.dspark?.layerCount, 3)
    XCTAssertEqual(plan.dspark?.blockSize, 5)
    XCTAssertEqual(plan.dspark?.targetLayerIDs, [40, 41, 42])
    XCTAssertEqual(plan.dspark?.commonTensors.map(\.name), ["mtp.0.main_proj.weight"])
    XCTAssertEqual(
      plan.files.first { $0.path == "dspark/experts/layer_00.bin" }?.size,
      256 * 13_369_344
    )
    let expert = try XCTUnwrap(
      plan.copies.first { $0.tensor == "mtp.2.ffn.experts.255.w3.scale" }
    )
    XCTAssertEqual(expert.destinationFile, "dspark/experts/layer_02.bin")
  }

  func testSafeTensorsHeaderDecodesTensorMetadata() throws {
    let data = Data(
      #"{"tensor":{"dtype":"I8","shape":[2,3],"data_offsets":[4,10]},"__metadata__":{"format":"pt"}}"#
        .utf8)
    let header = try SafeTensorsHeader.decode(data)
    XCTAssertEqual(
      header.entries["tensor"],
      SafeTensorsHeader.Entry(dtype: "I8", shape: [2, 3], dataStart: 4, dataEnd: 10)
    )
  }

  func testRepackerCopiesOnlyPlannedByteRanges() async throws {
    var files = companionFiles()
    files["shard"] = Data([0, 1, 2, 3, 4, 5, 6, 7])
    let source = MemoryCheckpointSource(files: files)
    let plan = RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: 4,
      files: [PlannedFile(path: "common.bin", size: 6)],
      commonTensors: [InstalledTensor(name: "test", dtype: "I8", shape: [4], offset: 1, length: 4)],
      expertRegions: ModelContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "test",
          sourceFile: "shard",
          sourceOffset: 2,
          length: 4,
          destinationFile: "common.bin",
          destinationOffset: 1
        )
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("model.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    let manifest = try await Repacker(source: source).run(plan: plan, output: output, progress: nil)
    XCTAssertEqual(manifest.files.count, 1 + ModelContract.companionPaths.count)
    XCTAssertEqual(
      try Data(contentsOf: output.appendingPathComponent("common.bin")),
      Data([0, 2, 3, 4, 5, 0])
    )
  }

  func testRepackerResumesValidatedChunks() async throws {
    var files = companionFiles()
    var shard = Data(repeating: 0, count: 100_004)
    shard.replaceSubrange(0..<4, with: [1, 2, 3, 4])
    shard.replaceSubrange(100_000..<100_004, with: [5, 6, 7, 8])
    files["shard"] = shard
    let source = FailingCheckpointSource(files: files, failAt: 100_000, failures: 3)
    let plan = RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: 8,
      files: [PlannedFile(path: "common.bin", size: 8)],
      commonTensors: [
        InstalledTensor(
          name: "test", dtype: "I8", shape: [8], offset: 0, length: 8
        )
      ],
      expertRegions: ModelContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "a", sourceFile: "shard", sourceOffset: 0, length: 4,
          destinationFile: "common.bin", destinationOffset: 0),
        TensorCopy(
          tensor: "b", sourceFile: "shard", sourceOffset: 100_000, length: 4,
          destinationFile: "common.bin", destinationOffset: 4),
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("model.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    do {
      _ = try await Repacker(source: source).run(plan: plan, output: output, progress: nil)
      XCTFail("first repack must fail")
    } catch {
      XCTAssertTrue(FileManager.default.fileExists(atPath: output.path + ".partial"))
    }

    _ = try await Repacker(source: source).run(plan: plan, output: output, progress: nil)
    XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("common.bin")), Data(1...8))
    let firstRangeReads = await source.readCount(at: 0)
    XCTAssertEqual(firstRangeReads, 1)
  }

  func testAuditFindsChecksumFailureAndRepairDownloadsInvalidFile() async throws {
    var files = companionFiles()
    files["shard"] = Data([1, 2, 3, 4])
    let source = MemoryCheckpointSource(files: files)
    let plan = RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: 4,
      files: [PlannedFile(path: "common.bin", size: 4)],
      commonTensors: [InstalledTensor(name: "test", dtype: "I8", shape: [4], offset: 0, length: 4)],
      expertRegions: ModelContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "test",
          sourceFile: "shard",
          sourceOffset: 0,
          length: 4,
          destinationFile: "common.bin",
          destinationOffset: 0
        )
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("model.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    let manifest = try await Repacker(source: source).run(
      plan: plan, output: output, progress: nil)
    try Data([9, 9, 9, 9]).write(to: output.appendingPathComponent("common.bin"))

    let failed = try InstalledModel.audit(manifest: manifest, at: output)
    XCTAssertEqual(
      failed.issues,
      [InstalledFileIssue(path: "common.bin", kind: .checksumMismatch)]
    )

    let repaired = try await Repacker(source: source).repair(
      plan: plan,
      output: output,
      invalidFiles: ["common.bin"],
      progress: nil
    )
    XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("common.bin")), Data(1...4))
    XCTAssertTrue(try InstalledModel.audit(manifest: repaired, at: output).isValid)
  }
}

private struct PlannerFixture {
  var index: CheckpointIndex
  var tensors: [String: SafeTensor]
}

private func makePlannerFixture(includeDSpark: Bool = false) -> PlannerFixture {
  var tensors: [String: SafeTensor] = [:]
  var weightMap: [String: String] = [:]
  var offset: UInt64 = 0

  let common = SafeTensor(
    name: "embed.weight",
    sourceFile: "common.safetensors",
    dtype: "BF16",
    shape: [2, 2],
    sourceOffset: 100,
    length: 8
  )
  tensors[common.name] = common
  weightMap[common.name] = common.sourceFile

  for layer in 0..<ModelContract.layerCount {
    for expert in 0..<ModelContract.expertCount {
      for region in ModelContract.expertRegions {
        let name = "layers.\(layer).ffn.experts.\(expert).\(region.name)"
        let tensor = SafeTensor(
          name: name,
          sourceFile: "experts.safetensors",
          dtype: region.dtype,
          shape: region.shape,
          sourceOffset: offset,
          length: region.length
        )
        tensors[name] = tensor
        weightMap[name] = tensor.sourceFile
        offset += tensor.length
      }
    }
  }
  if includeDSpark {
    let dsparkCommon = SafeTensor(
      name: "mtp.0.main_proj.weight",
      sourceFile: "dspark.safetensors",
      dtype: "BF16",
      shape: [2, 2],
      sourceOffset: offset,
      length: 8
    )
    tensors[dsparkCommon.name] = dsparkCommon
    weightMap[dsparkCommon.name] = dsparkCommon.sourceFile
    offset += dsparkCommon.length
    for layer in 0..<ModelContract.dsparkLayerCount {
      for expert in 0..<ModelContract.expertCount {
        for region in ModelContract.expertRegions {
          let name = "mtp.\(layer).ffn.experts.\(expert).\(region.name)"
          let tensor = SafeTensor(
            name: name,
            sourceFile: "dspark.safetensors",
            dtype: region.dtype,
            shape: region.shape,
            sourceOffset: offset,
            length: region.length
          )
          tensors[name] = tensor
          weightMap[name] = tensor.sourceFile
          offset += tensor.length
        }
      }
    }
  }
  return PlannerFixture(
    index: CheckpointIndex(totalSize: offset + common.length, weightMap: weightMap),
    tensors: tensors
  )
}

private struct MemoryCheckpointSource: CheckpointSource {
  let files: [String: Data]

  func data(path: String) async throws -> Data {
    guard let data = files[path] else { throw RepackError.badResponse("missing \(path)") }
    return data
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    guard let data = files[path], range.upperBound <= UInt64(data.count) else {
      throw RepackError.badResponse("invalid fixture range for \(path)")
    }
    return Data(data[Int(range.lowerBound)..<Int(range.upperBound)])
  }
}

private actor FailingCheckpointSource: CheckpointSource {
  let files: [String: Data]
  let failAt: UInt64
  var failures: Int
  var reads: [UInt64: Int] = [:]

  init(files: [String: Data], failAt: UInt64, failures: Int) {
    self.files = files
    self.failAt = failAt
    self.failures = failures
  }

  func data(path: String) async throws -> Data {
    guard let data = files[path] else { throw RepackError.badResponse("missing \(path)") }
    return data
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    reads[range.lowerBound, default: 0] += 1
    if range.lowerBound == failAt, failures > 0 {
      failures -= 1
      throw RepackError.badResponse("fixture interruption")
    }
    guard let data = files[path], range.upperBound <= UInt64(data.count) else {
      throw RepackError.badResponse("invalid fixture range for \(path)")
    }
    return Data(data[Int(range.lowerBound)..<Int(range.upperBound)])
  }

  func readCount(at offset: UInt64) -> Int {
    reads[offset, default: 0]
  }
}

private func companionFiles() -> [String: Data] {
  Dictionary(
    uniqueKeysWithValues: ModelContract.companions.map {
      ($0.source, Data($0.source.utf8))
    })
}
