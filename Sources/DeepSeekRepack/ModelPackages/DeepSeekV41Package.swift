import Foundation

struct DeepSeekV41Package: ModelPackage {
  let verifiesInstallation = false
  let installationLabel: String? = nil
  var companions: [CompanionFile] { DeepSeekV41Contract.companions }
  func installedBytes() async throws -> UInt64 { try await makeRepackPlan().installedBytes }
  func makeRepackPlan() async throws -> RepackPlan {
    try await DeepSeekV41Checkpoint().makeRepackPlan()
  }
  func repack(plan: RepackPlan, to output: URL, progress: RepackProgressHandler?) async throws
    -> InstalledManifest
  { try await DeepSeekV41Checkpoint().repack(plan: plan, to: output, progress: progress) }
  func install(to output: URL, progress: RepackProgressHandler?) async throws -> InstalledManifest {
    try await DeepSeekV41Checkpoint().repack(to: output, progress: progress)
  }
  func repair(at output: URL, invalidFiles: Set<String>, progress: RepackProgressHandler?)
    async throws -> InstalledManifest
  { try await DeepSeekV41Checkpoint().repair(at: output, invalidFiles: invalidFiles, progress: progress) }
  func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    try DeepSeekV41Contract.validate(manifest)
  }
}
