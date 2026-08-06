// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "DeepSeekV4SSD",
  platforms: [.macOS(.v15)],
  products: [
    .library(name: "DeepSeekRepack", targets: ["DeepSeekRepack"]),
    .executable(name: "dsv4-repack", targets: ["dsv4-repack"]),
    .executable(name: "dsv4-app", targets: ["DeepSeekV4SSDApp"]),
  ],
  targets: [
    .target(name: "DeepSeekRepack"),
    .executableTarget(name: "dsv4-repack", dependencies: ["DeepSeekRepack"]),
    .executableTarget(name: "DeepSeekV4SSDApp", dependencies: ["DeepSeekRepack"]),
    .testTarget(name: "DeepSeekRepackTests", dependencies: ["DeepSeekRepack"]),
    .testTarget(name: "DeepSeekV4SSDAppTests", dependencies: ["DeepSeekV4SSDApp"]),
  ]
)
