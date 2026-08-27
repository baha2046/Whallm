import Foundation

public enum MXFP4 {
  public static func quantizeBF16(
    _ source: Data,
    rows: Int,
    columns: Int
  ) throws -> (weights: Data, scales: Data) {
    guard rows > 0, columns > 0, columns.isMultiple(of: 32),
      source.count == rows * columns * 2
    else {
      throw RepackError.invalidPlan("MXFP4 source must contain complete BF16 rows")
    }

    let groupsPerRow = columns / 32
    let input = [UInt8](source)
    var weights = [UInt8](repeating: 0, count: rows * columns / 2)
    var scales = [UInt8](repeating: 0, count: rows * groupsPerRow)
    for row in 0..<rows {
      for group in 0..<groupsPerRow {
        let inputStart = row * columns + group * 32
        var maximum: Float = 0
        for index in 0..<32 {
          maximum = max(maximum, abs(bfloat16(input, at: inputStart + index)))
        }
        let scaleIndex = row * groupsPerRow + group
        guard maximum != 0 else { continue }
        let exponent = max(-127, min(128, maximum.exponent - 2))
        scales[scaleIndex] = UInt8(exponent + 127)
        let factor = scalbnf(1, Int32(-exponent))
        let outputStart = row * columns / 2 + group * 16
        for pair in 0..<16 {
          let low = fp4Code(bfloat16(input, at: inputStart + pair * 2) * factor)
          let high = fp4Code(bfloat16(input, at: inputStart + pair * 2 + 1) * factor)
          weights[outputStart + pair] = low | (high << 4)
        }
      }
    }
    return (Data(weights), Data(scales))
  }

  public static func quantizeFP8(
    _ source: Data,
    inverseScales: Data,
    rows: Int,
    columns: Int
  ) throws -> (weights: Data, scales: Data) {
    guard rows > 0, columns > 0, rows.isMultiple(of: 128),
      columns.isMultiple(of: 128), source.count == rows * columns,
      inverseScales.count == rows / 128 * (columns / 128) * 2
    else {
      throw RepackError.invalidPlan(
        "MXFP4 FP8 source must contain complete 128 by 128 scale blocks")
    }

    let input = [UInt8](source)
    let sourceScales = [UInt8](inverseScales)
    let groupsPerRow = columns / 32
    let scaleColumns = columns / 128
    var weights = [UInt8](repeating: 0, count: rows * columns / 2)
    var scales = [UInt8](repeating: 0, count: rows * groupsPerRow)
    var values = [Float](repeating: 0, count: 32)
    for row in 0..<rows {
      for group in 0..<groupsPerRow {
        let inputStart = row * columns + group * 32
        let sourceScaleIndex = (row / 128) * scaleColumns + group / 4
        let sourceScale = bfloat16(sourceScales, at: sourceScaleIndex)
        var maximum: Float = 0
        for index in 0..<32 {
          let value = roundedBFloat16(float8(input[inputStart + index]) * sourceScale)
          values[index] = value
          maximum = max(maximum, abs(value))
        }
        let scaleIndex = row * groupsPerRow + group
        guard maximum != 0 else { continue }
        let exponent = max(-127, min(128, maximum.exponent - 2))
        scales[scaleIndex] = UInt8(exponent + 127)
        let factor = scalbnf(1, Int32(-exponent))
        let outputStart = row * columns / 2 + group * 16
        for pair in 0..<16 {
          let low = fp4Code(values[pair * 2] * factor)
          let high = fp4Code(values[pair * 2 + 1] * factor)
          weights[outputStart + pair] = low | (high << 4)
        }
      }
    }
    return (Data(weights), Data(scales))
  }

  private static func bfloat16(_ bytes: [UInt8], at index: Int) -> Float {
    let byte = index * 2
    let bits = UInt16(bytes[byte]) | UInt16(bytes[byte + 1]) << 8
    return Float(bitPattern: UInt32(bits) << 16)
  }

  private static func float8(_ bits: UInt8) -> Float {
    let sign: Float = bits & 0x80 == 0 ? 1 : -1
    let exponent = Int((bits >> 3) & 0x0f)
    let mantissa = Int(bits & 0x07)
    if exponent == 0 {
      return sign * Float(mantissa) * scalbnf(1, -9)
    }
    if exponent == 15, mantissa == 7 {
      return .nan
    }
    return sign * (1 + Float(mantissa) / 8) * scalbnf(1, Int32(exponent - 7))
  }

  private static func roundedBFloat16(_ value: Float) -> Float {
    let bits = value.bitPattern
    let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
    return Float(bitPattern: rounded & 0xffff_0000)
  }

  private static func fp4Code(_ value: Float) -> UInt8 {
    let negative = value.sign == .minus
    let magnitude = min(abs(value), 6)
    let values: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    var selected = 0
    var distance = Float.infinity
    for (index, candidate) in values.enumerated() {
      let candidateDistance = abs(magnitude - candidate)
      if candidateDistance < distance
        || (candidateDistance == distance && index.isMultiple(of: 2))
      {
        selected = index
        distance = candidateDistance
      }
    }
    return UInt8(selected) | (negative ? 8 : 0)
  }
}
