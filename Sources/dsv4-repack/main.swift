import DeepSeekRepack
import Foundation

@main
struct CLI {
  static func main() async {
    do {
      try await run(Array(CommandLine.arguments.dropFirst()))
    } catch {
      FileHandle.standardError.write(Data("error: \(error)\n".utf8))
      Foundation.exit(1)
    }
  }

  private static func run(_ arguments: [String]) async throws {
    guard let command = arguments.first else {
      printUsage()
      return
    }
    switch command {
    case "inspect":
      let plan = try await DeepSeekV4Checkpoint().makeRepackPlan()
      printSummary(plan)
    case "plan":
      let output = try value(after: "--output", in: arguments)
      let plan = try await DeepSeekV4Checkpoint().makeRepackPlan()
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let url = URL(fileURLWithPath: output).standardizedFileURL
      guard !FileManager.default.fileExists(atPath: url.path) else {
        throw RepackError.destinationExists(url.path)
      }
      try encoder.encode(plan).write(to: url, options: .atomic)
      printSummary(plan)
      print("plan: \(url.path)")
    case "repack":
      let output = try value(after: "--output", in: arguments)
      let printer = ProgressPrinter()
      let checkpoint = DeepSeekV4Checkpoint()
      let outputURL = URL(fileURLWithPath: output)
      let progress: @Sendable (RepackProgress) -> Void = { printer.update($0) }
      let manifest: InstalledManifest
      if let planPath = optionalValue(after: "--plan", in: arguments) {
        let data = try Data(contentsOf: URL(fileURLWithPath: planPath))
        let plan = try JSONDecoder().decode(RepackPlan.self, from: data)
        manifest = try await checkpoint.repack(plan: plan, to: outputURL, progress: progress)
      } else {
        manifest = try await checkpoint.repack(to: outputURL, progress: progress)
      }
      print("installed: \(output)")
      print("files: \(manifest.files.count)")
    case "verify":
      let model = try value(after: "--model", in: arguments)
      let manifest = try InstalledModel.verify(at: URL(fileURLWithPath: model))
      print("verified: \(model)")
      print("files: \(manifest.files.count)")
    case "benchmark":
      let model = try value(after: "--model", in: arguments)
      let samples = try integer(after: "--samples", in: arguments, default: 32)
      let results = try ExpertIOBenchmark.run(
        at: URL(fileURLWithPath: model),
        samples: samples
      )
      printBenchmark(results)
    case "help", "--help", "-h":
      printUsage()
    default:
      throw RepackError.invalidPlan("unknown command \(command)")
    }
  }

  private static func value(after option: String, in arguments: [String]) throws -> String {
    guard let index = arguments.firstIndex(of: option), arguments.indices.contains(index + 1) else {
      throw RepackError.invalidPlan("missing \(option)")
    }
    return arguments[index + 1]
  }

  private static func optionalValue(after option: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: option), arguments.indices.contains(index + 1)
    else {
      return nil
    }
    return arguments[index + 1]
  }

  private static func integer(
    after option: String,
    in arguments: [String],
    default defaultValue: Int
  ) throws -> Int {
    guard let index = arguments.firstIndex(of: option) else { return defaultValue }
    guard arguments.indices.contains(index + 1), let value = Int(arguments[index + 1]) else {
      throw RepackError.invalidPlan("invalid \(option)")
    }
    return value
  }

  private static func printBenchmark(_ results: [ExpertIOBenchmarkResult]) {
    print("mode\tworkers\treads\tGiB/s\tmean ms")
    for result in results {
      print(
        "\(result.cacheMode.rawValue)\t\(result.concurrency)\t\(result.reads)\t"
          + String(format: "%.2f\t%.2f", result.gibibytesPerSecond, result.meanMilliseconds)
      )
    }
  }

  private static func printSummary(_ plan: RepackPlan) {
    print("model: \(plan.modelID)")
    print("revision: \(plan.revision)")
    print("layers: \(plan.layerCount)")
    print("experts per layer: \(plan.expertCount)")
    print("expert blob: \(plan.expertBlobSize) bytes")
    print("installed weight bytes: \(plan.installedBytes)")
    print("copy operations: \(plan.copies.count)")
  }

  private static func printUsage() {
    print(
      """
      Usage:
        dsv4-repack inspect
        dsv4-repack plan --output plan.json
        dsv4-repack repack --output deepseek-v4-flash-0731.dsv4 [--plan plan.json]
        dsv4-repack verify --model deepseek-v4-flash-0731.dsv4
        dsv4-repack benchmark --model deepseek-v4-flash-0731.dsv4 [--samples 32]
      """)
  }
}

private final class ProgressPrinter: @unchecked Sendable {
  private let lock = NSLock()
  private var lastPercent = -1

  func update(_ progress: RepackProgress) {
    let percent =
      progress.totalBytes == 0 ? 100 : Int(progress.copiedBytes * 100 / progress.totalBytes)
    lock.lock()
    defer { lock.unlock() }
    guard percent != lastPercent else { return }
    lastPercent = percent
    print("repack: \(percent)%")
  }
}
