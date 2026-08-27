import Foundation

struct CheckpointIndex: Sendable {
  let totalSize: UInt64
  let weightMap: [String: String]

  static func decode(_ data: Data) throws -> CheckpointIndex {
    struct RawIndex: Decodable {
      struct Metadata: Decodable {
        let totalSize: UInt64

        enum CodingKeys: String, CodingKey {
          case totalSize = "total_size"
        }
      }
      let metadata: Metadata
      let weightMap: [String: String]

      enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
      }
    }

    do {
      let raw = try JSONDecoder().decode(RawIndex.self, from: data)
      guard !raw.weightMap.isEmpty else {
        throw RepackError.invalidIndex("checkpoint index has no tensors")
      }
      return CheckpointIndex(totalSize: raw.metadata.totalSize, weightMap: raw.weightMap)
    } catch let error as RepackError {
      throw error
    } catch {
      throw RepackError.invalidIndex("cannot decode checkpoint index: \(error)")
    }
  }
}

struct SafeTensorsHeader: Sendable {
  struct Entry: Equatable, Sendable {
    let dtype: String
    let shape: [Int]
    let dataStart: UInt64
    let dataEnd: UInt64
  }

  let entries: [String: Entry]

  static func decode(_ data: Data) throws -> SafeTensorsHeader {
    let object: Any
    do {
      object = try JSONSerialization.jsonObject(with: data)
    } catch {
      throw RepackError.invalidSafeTensors("cannot decode safetensors header: \(error)")
    }
    guard let dictionary = object as? [String: Any] else {
      throw RepackError.invalidSafeTensors("safetensors header is not an object")
    }

    var entries: [String: Entry] = [:]
    for (name, value) in dictionary where name != "__metadata__" {
      guard let tensor = value as? [String: Any],
        let dtype = tensor["dtype"] as? String,
        let rawShape = tensor["shape"] as? [NSNumber],
        let rawOffsets = tensor["data_offsets"] as? [NSNumber],
        rawOffsets.count == 2
      else {
        throw RepackError.invalidSafeTensors("invalid tensor metadata for \(name)")
      }
      let start = rawOffsets[0].uint64Value
      let end = rawOffsets[1].uint64Value
      guard end >= start else {
        throw RepackError.invalidSafeTensors("invalid byte range for \(name)")
      }
      entries[name] = Entry(
        dtype: dtype,
        shape: rawShape.map(\.intValue),
        dataStart: start,
        dataEnd: end
      )
    }
    return SafeTensorsHeader(entries: entries)
  }
}

func littleEndianUInt64(_ data: Data) throws -> UInt64 {
  guard data.count == 8 else {
    throw RepackError.invalidSafeTensors("safetensors length prefix is not 8 bytes")
  }
  return data.enumerated().reduce(UInt64(0)) { value, pair in
    value | (UInt64(pair.element) << UInt64(pair.offset * 8))
  }
}
