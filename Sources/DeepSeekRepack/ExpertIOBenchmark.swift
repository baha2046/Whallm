import Darwin
import Foundation

public enum ExpertIOCacheMode: String, Codable, Sendable {
  case cached
  case direct
}

public struct ExpertIOBenchmarkResult: Codable, Sendable {
  public let cacheMode: ExpertIOCacheMode
  public let concurrency: Int
  public let reads: Int
  public let bytes: UInt64
  public let seconds: Double
  public let gibibytesPerSecond: Double
  public let meanMilliseconds: Double
}

public enum ExpertIOBenchmark {
  public static func run(
    at root: URL,
    samples: Int = 32,
    concurrencies: [Int] = [1, 2, 4, 8]
  ) throws -> [ExpertIOBenchmarkResult] {
    guard samples > 0 else {
      throw RepackError.invalidPlan("benchmark samples must be greater than zero")
    }
    guard !concurrencies.isEmpty, concurrencies.allSatisfy({ $0 > 0 }) else {
      throw RepackError.invalidPlan("benchmark concurrency must be greater than zero")
    }

    let root = root.standardizedFileURL
    let manifest = try InstalledModel.loadManifest(at: root)
    guard manifest.expertBlobSize <= UInt64(Int.max) else {
      throw RepackError.invalidPlan("expert blob is too large for this process")
    }

    var results: [ExpertIOBenchmarkResult] = []
    for mode in [ExpertIOCacheMode.direct, .cached] {
      for concurrency in concurrencies {
        results.append(
          try measure(
            root: root,
            manifest: manifest,
            mode: mode,
            samples: samples,
            concurrency: concurrency
          ))
      }
    }
    return results
  }

  private static func measure(
    root: URL,
    manifest: InstalledManifest,
    mode: ExpertIOCacheMode,
    samples: Int,
    concurrency: Int
  ) throws -> ExpertIOBenchmarkResult {
    let accumulator = BenchmarkAccumulator()
    let descriptorCount = mode == .cached ? 1 : manifest.layerCount
    var descriptors: [Int32] = []
    defer {
      for descriptor in descriptors {
        Darwin.close(descriptor)
      }
    }
    for layer in 0..<descriptorCount {
      let file = root.appendingPathComponent(
        String(format: "experts/layer_%02d.bin", layer))
      let descriptor = Darwin.open(file.path, O_RDONLY)
      guard descriptor >= 0 else {
        throw POSIXReadError(operation: "open", path: file.path, code: errno)
      }
      if mode == .direct, Darwin.fcntl(descriptor, F_NOCACHE, 1) == -1 {
        let code = errno
        Darwin.close(descriptor)
        throw POSIXReadError(operation: "F_NOCACHE", path: file.path, code: code)
      }
      descriptors.append(descriptor)
    }
    let readDescriptors = descriptors
    if mode == .cached {
      let count = Int(manifest.expertBlobSize)
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 16_384)
      defer { buffer.deallocate() }
      try preadFully(descriptor: readDescriptors[0], buffer: buffer, count: count, offset: 0)
    }
    let start = DispatchTime.now().uptimeNanoseconds

    DispatchQueue.concurrentPerform(iterations: concurrency) { worker in
      do {
        let count = Int(manifest.expertBlobSize)
        var buffer: UnsafeMutableRawPointer?
        let allocation = posix_memalign(&buffer, 16_384, count)
        guard allocation == 0, let buffer else {
          throw POSIXReadError(operation: "posix_memalign", path: "expert buffer", code: allocation)
        }
        defer { free(buffer) }

        for sample in stride(from: worker, to: samples, by: concurrency) {
          let layer = mode == .cached ? 0 : sample % manifest.layerCount
          let expert = mode == .cached ? 0 : (sample / manifest.layerCount) % manifest.expertCount
          let offset = UInt64(expert) * manifest.expertBlobSize
          let readStart = DispatchTime.now().uptimeNanoseconds
          try preadFully(
            descriptor: readDescriptors[layer],
            buffer: buffer,
            count: count,
            offset: offset
          )
          let readEnd = DispatchTime.now().uptimeNanoseconds
          accumulator.record(nanoseconds: readEnd - readStart)
        }
      } catch {
        accumulator.record(error: error)
      }
    }

    if let error = accumulator.firstError { throw error }
    let end = DispatchTime.now().uptimeNanoseconds
    let seconds = Double(end - start) / 1_000_000_000
    let bytes = UInt64(samples) * manifest.expertBlobSize
    let gibibytesPerSecond = Double(bytes) / Double(1 << 30) / seconds
    let meanMilliseconds = Double(accumulator.totalNanoseconds) / Double(samples) / 1_000_000
    return ExpertIOBenchmarkResult(
      cacheMode: mode,
      concurrency: concurrency,
      reads: samples,
      bytes: bytes,
      seconds: seconds,
      gibibytesPerSecond: gibibytesPerSecond,
      meanMilliseconds: meanMilliseconds
    )
  }
}

private final class BenchmarkAccumulator: @unchecked Sendable {
  private let lock = NSLock()
  private var nanoseconds: UInt64 = 0
  private var error: Error?

  var totalNanoseconds: UInt64 {
    lock.withLock { nanoseconds }
  }

  var firstError: Error? {
    lock.withLock { error }
  }

  func record(nanoseconds: UInt64) {
    lock.withLock { self.nanoseconds += nanoseconds }
  }

  func record(error: Error) {
    lock.withLock {
      if self.error == nil { self.error = error }
    }
  }
}

private struct POSIXReadError: Error, CustomStringConvertible {
  let operation: String
  let path: String
  let code: Int32

  var description: String {
    "\(operation) failed for \(path): \(String(cString: strerror(code)))"
  }
}

private func preadFully(
  descriptor: Int32,
  buffer: UnsafeMutableRawPointer,
  count: Int,
  offset: UInt64
) throws {
  var position = 0
  while position < count {
    let result = Darwin.pread(
      descriptor,
      buffer.advanced(by: position),
      count - position,
      off_t(offset + UInt64(position))
    )
    if result > 0 {
      position += result
    } else if result == -1, errno == EINTR {
      continue
    } else if result == 0 {
      throw RepackError.invalidPlan("unexpected end of expert file")
    } else {
      throw POSIXReadError(operation: "pread", path: "expert file", code: errno)
    }
  }
}
