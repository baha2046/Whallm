import CryptoKit
import Darwin
import Dispatch
import Foundation
import Metal


enum ProbeError: Error, CustomStringConvertible {
    case argument(String)
    case system(String)
    case metal(String)

    var description: String {
        switch self {
        case .argument(let message), .system(let message), .metal(let message):
            return message
        }
    }
}


struct Arguments {
    let file: String
    let blobSize: Int
    let count: Int
    let startExpert: Int
    let method: String
    let readWorkers: Int

    static func parse() throws -> Arguments {
        var values: [String: String] = [:]
        var index = 1
        while index < CommandLine.arguments.count {
            let key = CommandLine.arguments[index]
            guard key.hasPrefix("--"), index + 1 < CommandLine.arguments.count else {
                throw ProbeError.argument("invalid argument at index \(index)")
            }
            values[String(key.dropFirst(2))] = CommandLine.arguments[index + 1]
            index += 2
        }
        guard let file = values["file"], let method = values["method"] else {
            throw ProbeError.argument("--file and --method are required")
        }
        guard
            let blobSize = Int(values["blob-size"] ?? ""), blobSize > 0,
            let count = Int(values["count"] ?? ""), count > 0,
            let startExpert = Int(values["start-expert"] ?? ""), startExpert >= 0,
            let readWorkers = Int(values["read-workers"] ?? "4"), readWorkers > 0
        else {
            throw ProbeError.argument("invalid numeric argument")
        }
        let methods = [
            "preadv",
            "mtlio_bytes",
            "mtlio_shared",
            "mtlio_private",
            "mtlio_cancel",
        ]
        guard methods.contains(method) else {
            throw ProbeError.argument("unsupported method: \(method)")
        }
        return Arguments(
            file: file,
            blobSize: blobSize,
            count: count,
            startExpert: startExpert,
            method: method,
            readWorkers: readWorkers
        )
    }
}


func now() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}


func seconds(_ start: UInt64, _ end: UInt64? = nil) -> Double {
    Double((end ?? now()) - start) / 1_000_000_000
}


func percentile(_ values: [Double], _ fraction: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let index = Int(ceil(Double(sorted.count) * fraction)) - 1
    return sorted[max(0, min(index, sorted.count - 1))]
}


func hexDigest(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
}


func pointerDigest(_ pointer: UnsafeRawPointer, count: Int) -> String {
    var hash = SHA256()
    hash.update(bufferPointer: UnsafeRawBufferPointer(start: pointer, count: count))
    return hexDigest(hash.finalize())
}


func fileRangeDigest(
    descriptor: Int32,
    blobSize: Int,
    count: Int,
    startExpert: Int
) throws -> String {
    let chunkSize = 4 * 1024 * 1024
    let buffer = UnsafeMutableRawPointer.allocate(
        byteCount: chunkSize,
        alignment: 4096
    )
    defer { buffer.deallocate() }
    var hash = SHA256()
    for expert in 0..<count {
        var remaining = blobSize
        var relative = 0
        let base = (startExpert + expert) * blobSize
        while remaining > 0 {
            let amount = min(chunkSize, remaining)
            let received = pread(
                descriptor,
                buffer,
                amount,
                off_t(base + relative)
            )
            guard received == amount else {
                throw ProbeError.system(
                    "reference pread returned \(received), expected \(amount)"
                )
            }
            hash.update(
                bufferPointer: UnsafeRawBufferPointer(
                    start: buffer,
                    count: amount
                )
            )
            remaining -= amount
            relative += amount
        }
    }
    return hexDigest(hash.finalize())
}


func allocateAligned(_ size: Int) throws -> UnsafeMutableRawPointer {
    var pointer: UnsafeMutableRawPointer?
    let result = posix_memalign(&pointer, 4096, size)
    guard result == 0, let pointer else {
        throw ProbeError.system("posix_memalign failed: \(result)")
    }
    return pointer
}


func ioStatusName(_ status: MTLIOStatus) -> String {
    switch status {
    case .pending: return "pending"
    case .cancelled: return "cancelled"
    case .error: return "error"
    case .complete: return "complete"
    @unknown default: return "unknown"
    }
}


func makeIOQueue(device: MTLDevice, count: Int) throws -> MTLIOCommandQueue {
    let descriptor = MTLIOCommandQueueDescriptor()
    descriptor.type = .concurrent
    descriptor.priority = .high
    descriptor.maxCommandBufferCount = max(2, count + 1)
    descriptor.maxCommandsInFlight = count
    do {
        return try device.makeIOCommandQueue(descriptor: descriptor)
    } catch {
        throw ProbeError.metal("makeIOCommandQueue failed: \(error)")
    }
}


func makeIOHandle(device: MTLDevice, file: String) throws -> MTLIOFileHandle {
    do {
        return try device.makeIOFileHandle(url: URL(fileURLWithPath: file))
    } catch {
        throw ProbeError.metal("makeIOFileHandle failed: \(error)")
    }
}


func runPreadv(
    arguments: Arguments,
    descriptor: Int32,
    totalBytes: Int
) throws -> [String: Any] {
    let allocationStarted = now()
    let arena = try allocateAligned(totalBytes)
    let allocationSeconds = seconds(allocationStarted)
    defer { free(arena) }

    let semaphore = DispatchSemaphore(value: arguments.readWorkers)
    let group = DispatchGroup()
    let queue = DispatchQueue.global(qos: .userInitiated)
    let lock = NSLock()
    var samples = Array(repeating: 0.0, count: arguments.count)
    var failures: [String] = []
    let started = now()
    for expert in 0..<arguments.count {
        group.enter()
        queue.async {
            semaphore.wait()
            defer {
                semaphore.signal()
                group.leave()
            }
            let commandStarted = now()
            var vector = iovec(
                iov_base: arena.advanced(by: expert * arguments.blobSize),
                iov_len: arguments.blobSize
            )
            let offset = (arguments.startExpert + expert) * arguments.blobSize
            let received = preadv(descriptor, &vector, 1, off_t(offset))
            let elapsed = seconds(commandStarted)
            lock.lock()
            samples[expert] = elapsed
            if received != arguments.blobSize {
                failures.append(
                    "expert \(expert): preadv returned \(received), expected \(arguments.blobSize)"
                )
            }
            lock.unlock()
        }
    }
    group.wait()
    let ioSeconds = seconds(started)
    guard failures.isEmpty else {
        throw ProbeError.system(failures.joined(separator: "; "))
    }
    let candidate = pointerDigest(arena, count: totalBytes)
    return [
        "allocation_seconds": allocationSeconds,
        "io_seconds": ioSeconds,
        "gpu_visible_seconds": NSNull(),
        "gpu_copy_seconds": NSNull(),
        "per_command_seconds": samples,
        "latency_p50_seconds": percentile(samples, 0.50),
        "latency_p95_seconds": percentile(samples, 0.95),
        "candidate_sha256": candidate,
        "status": "complete",
        "error": NSNull(),
        "storage_mode": "cpu_aligned",
        "cpu_visible_destination": true,
        "explicit_gpu_validation_copy": false,
        "shared_event_ordering": false,
    ]
}


func runMTLIOBytes(
    arguments: Arguments,
    device: MTLDevice,
    totalBytes: Int
) throws -> [String: Any] {
    let allocationStarted = now()
    let arena = try allocateAligned(totalBytes)
    let allocationSeconds = seconds(allocationStarted)
    defer { free(arena) }
    let queue = try makeIOQueue(device: device, count: arguments.count)
    let handle = try makeIOHandle(device: device, file: arguments.file)
    var commandBuffers: [MTLIOCommandBuffer] = []
    var samples = Array(repeating: 0.0, count: arguments.count)
    var starts = Array(repeating: UInt64(0), count: arguments.count)
    let lock = NSLock()
    for expert in 0..<arguments.count {
        let command = queue.makeCommandBuffer()
        command.loadBytes(
            arena.advanced(by: expert * arguments.blobSize),
            size: arguments.blobSize,
            sourceHandle: handle,
            sourceHandleOffset: (arguments.startExpert + expert) * arguments.blobSize
        )
        command.addCompletedHandler { _ in
            let elapsed = seconds(starts[expert])
            lock.lock()
            samples[expert] = elapsed
            lock.unlock()
        }
        commandBuffers.append(command)
    }
    let started = now()
    for expert in commandBuffers.indices {
        starts[expert] = now()
        commandBuffers[expert].commit()
    }
    for command in commandBuffers {
        command.waitUntilCompleted()
    }
    let ioSeconds = seconds(started)
    let errors = commandBuffers.compactMap { command -> String? in
        guard command.status != .complete else { return nil }
        return "\(ioStatusName(command.status)): \(command.error?.localizedDescription ?? "unknown")"
    }
    guard errors.isEmpty else {
        throw ProbeError.metal(errors.joined(separator: "; "))
    }
    return [
        "allocation_seconds": allocationSeconds,
        "io_seconds": ioSeconds,
        "gpu_visible_seconds": NSNull(),
        "gpu_copy_seconds": NSNull(),
        "per_command_seconds": samples,
        "latency_p50_seconds": percentile(samples, 0.50),
        "latency_p95_seconds": percentile(samples, 0.95),
        "candidate_sha256": pointerDigest(arena, count: totalBytes),
        "status": "complete",
        "error": NSNull(),
        "storage_mode": "cpu_aligned",
        "cpu_visible_destination": true,
        "explicit_gpu_validation_copy": false,
        "shared_event_ordering": false,
    ]
}


func runMTLIOBuffer(
    arguments: Arguments,
    device: MTLDevice,
    totalBytes: Int,
    storageMode: MTLStorageMode
) throws -> [String: Any] {
    let allocationStarted = now()
    let options: MTLResourceOptions = storageMode == .shared
        ? .storageModeShared
        : .storageModePrivate
    guard
        let target = device.makeBuffer(length: totalBytes, options: options),
        let validation = device.makeBuffer(
            length: totalBytes,
            options: .storageModeShared
        ),
        let gpuQueue = device.makeCommandQueue(),
        let event = device.makeSharedEvent()
    else {
        throw ProbeError.metal("Metal buffer, queue, or event allocation failed")
    }
    let allocationSeconds = seconds(allocationStarted)
    let ioQueue = try makeIOQueue(device: device, count: arguments.count)
    let handle = try makeIOHandle(device: device, file: arguments.file)
    var commandBuffers: [MTLIOCommandBuffer] = []
    var samples = Array(repeating: 0.0, count: arguments.count)
    var starts = Array(repeating: UInt64(0), count: arguments.count)
    let lock = NSLock()
    for expert in 0..<arguments.count {
        let command = ioQueue.makeCommandBuffer()
        command.load(
            target,
            offset: expert * arguments.blobSize,
            size: arguments.blobSize,
            sourceHandle: handle,
            sourceHandleOffset: (arguments.startExpert + expert) * arguments.blobSize
        )
        command.addCompletedHandler { _ in
            let elapsed = seconds(starts[expert])
            lock.lock()
            samples[expert] = elapsed
            lock.unlock()
        }
        commandBuffers.append(command)
    }
    let signal = ioQueue.makeCommandBuffer()
    guard let gpuCommand = gpuQueue.makeCommandBuffer() else {
        throw ProbeError.metal("Metal command buffer allocation failed")
    }
    signal.signalEvent(event, value: 1)
    gpuCommand.encodeWaitForEvent(event, value: 1)
    guard let blit = gpuCommand.makeBlitCommandEncoder() else {
        throw ProbeError.metal("Metal blit encoder allocation failed")
    }
    blit.copy(
        from: target,
        sourceOffset: 0,
        to: validation,
        destinationOffset: 0,
        size: totalBytes
    )
    blit.endEncoding()

    let started = now()
    for expert in commandBuffers.indices {
        starts[expert] = now()
        commandBuffers[expert].commit()
    }
    ioQueue.enqueueBarrier()
    signal.commit()
    gpuCommand.commit()
    gpuCommand.waitUntilCompleted()
    let gpuVisibleSeconds = seconds(started)
    for command in commandBuffers {
        command.waitUntilCompleted()
    }
    signal.waitUntilCompleted()
    let ioSeconds = commandBuffers.indices.map {
        samples[$0]
    }.max() ?? 0
    let ioErrors = commandBuffers.compactMap { command -> String? in
        guard command.status == .complete else {
            return "\(ioStatusName(command.status)): \(command.error?.localizedDescription ?? "unknown")"
        }
        return nil
    }
    guard ioErrors.isEmpty, signal.status == .complete else {
        throw ProbeError.metal(
            (ioErrors + [signal.error?.localizedDescription].compactMap { $0 })
                .joined(separator: "; ")
        )
    }
    guard gpuCommand.status == .completed else {
        throw ProbeError.metal(
            "GPU visibility command failed: \(gpuCommand.error?.localizedDescription ?? "unknown")"
        )
    }
    let gpuCopySeconds = gpuCommand.gpuEndTime > gpuCommand.gpuStartTime
        ? gpuCommand.gpuEndTime - gpuCommand.gpuStartTime
        : 0
    let validationDigest = pointerDigest(
        validation.contents(),
        count: totalBytes
    )
    let directDigest: Any = storageMode == .shared
        ? pointerDigest(target.contents(), count: totalBytes)
        : NSNull()
    return [
        "allocation_seconds": allocationSeconds,
        "io_seconds": ioSeconds,
        "gpu_visible_seconds": gpuVisibleSeconds,
        "gpu_copy_seconds": gpuCopySeconds,
        "per_command_seconds": samples,
        "latency_p50_seconds": percentile(samples, 0.50),
        "latency_p95_seconds": percentile(samples, 0.95),
        "candidate_sha256": validationDigest,
        "direct_cpu_sha256": directDigest,
        "status": "complete",
        "error": NSNull(),
        "storage_mode": storageMode == .shared ? "shared" : "private",
        "cpu_visible_destination": storageMode == .shared,
        "explicit_gpu_validation_copy": true,
        "validation_copy_required_for_cpu_access": storageMode == .private,
        "shared_event_ordering": true,
        "event_signaled_value": event.signaledValue,
    ]
}


func runCancellation(
    arguments: Arguments,
    device: MTLDevice
) throws -> [String: Any] {
    let queue = try makeIOQueue(device: device, count: 1)
    let handle = try makeIOHandle(device: device, file: arguments.file)
    guard let target = device.makeBuffer(
        length: arguments.blobSize,
        options: .storageModeShared
    ) else {
        throw ProbeError.metal("cancellation target allocation failed")
    }
    let command = queue.makeCommandBuffer()
    command.load(
        target,
        offset: 0,
        size: arguments.blobSize,
        sourceHandle: handle,
        sourceHandleOffset: arguments.startExpert * arguments.blobSize
    )
    let started = now()
    command.commit()
    command.tryCancel()
    command.waitUntilCompleted()
    let status = ioStatusName(command.status)
    let admitted = command.status == .complete
    let digest: Any = admitted
        ? pointerDigest(target.contents(), count: arguments.blobSize)
        : NSNull()
    return [
        "allocation_seconds": 0,
        "io_seconds": seconds(started),
        "gpu_visible_seconds": NSNull(),
        "gpu_copy_seconds": NSNull(),
        "per_command_seconds": [],
        "latency_p50_seconds": 0,
        "latency_p95_seconds": 0,
        "candidate_sha256": digest,
        "status": status,
        "error": command.error?.localizedDescription ?? NSNull(),
        "storage_mode": "shared",
        "cpu_visible_destination": true,
        "explicit_gpu_validation_copy": false,
        "shared_event_ordering": false,
        "destination_admitted": admitted,
    ]
}


func main() throws {
    let arguments = try Arguments.parse()
    let descriptor = open(arguments.file, O_RDONLY)
    guard descriptor >= 0 else {
        throw ProbeError.system("open failed: \(String(cString: strerror(errno)))")
    }
    defer { close(descriptor) }
    let totalBytes = arguments.blobSize * arguments.count
    let fileSize = Int(lseek(descriptor, 0, SEEK_END))
    guard
        arguments.startExpert * arguments.blobSize + totalBytes <= fileSize
    else {
        throw ProbeError.argument("expert ranges exceed file size")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ProbeError.metal("no Metal device")
    }
    let result: [String: Any]
    switch arguments.method {
    case "preadv":
        result = try runPreadv(
            arguments: arguments,
            descriptor: descriptor,
            totalBytes: totalBytes
        )
    case "mtlio_bytes":
        result = try runMTLIOBytes(
            arguments: arguments,
            device: device,
            totalBytes: totalBytes
        )
    case "mtlio_shared":
        result = try runMTLIOBuffer(
            arguments: arguments,
            device: device,
            totalBytes: totalBytes,
            storageMode: .shared
        )
    case "mtlio_private":
        result = try runMTLIOBuffer(
            arguments: arguments,
            device: device,
            totalBytes: totalBytes,
            storageMode: .private
        )
    case "mtlio_cancel":
        result = try runCancellation(arguments: arguments, device: device)
    default:
        throw ProbeError.argument("unreachable method")
    }
    let reference = try fileRangeDigest(
        descriptor: descriptor,
        blobSize: arguments.blobSize,
        count: arguments.count,
        startExpert: arguments.startExpert
    )
    let candidate = result["candidate_sha256"] as? String
    let admitted = (result["destination_admitted"] as? Bool) ?? true
    let exact = admitted ? candidate == reference : true
    var output: [String: Any] = [
        "schema_version": 1,
        "method": arguments.method,
        "file": arguments.file,
        "file_size": fileSize,
        "blob_size": arguments.blobSize,
        "count": arguments.count,
        "start_expert": arguments.startExpert,
        "total_bytes": totalBytes,
        "read_workers": arguments.readWorkers,
        "device": device.name,
        "max_buffer_length": device.maxBufferLength,
        "reference_sha256": reference,
        "exact": exact,
        "aggregate_bytes_per_second": (
            Double(totalBytes) / max(result["io_seconds"] as? Double ?? 0, 1e-12)
        ),
    ]
    for (key, value) in result {
        output[key] = value
    }
    let data = try JSONSerialization.data(
        withJSONObject: output,
        options: [.prettyPrinted, .sortedKeys]
    )
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
    if !exact {
        throw ProbeError.system("candidate bytes do not match installed file range")
    }
}


do {
    try main()
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
