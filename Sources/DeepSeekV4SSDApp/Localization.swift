import Foundation

enum AppLanguage: String, CaseIterable, Identifiable {
  case system
  case english = "en"
  case simplifiedChinese = "zh-Hans"
  case traditionalChinese = "zh-Hant"

  static let appDefault = AppLanguage.system

  var id: String { rawValue }

  func displayName(language: AppLanguage) -> String {
    switch self {
    case .system: L10n.string("Follow System", language: language)
    case .english: "English"
    case .simplifiedChinese: "简体中文"
    case .traditionalChinese: "繁體中文"
    }
  }

  var resolved: AppLanguage {
    self == .system ? Self.systemDefault() : self
  }

  var locale: Locale { Locale(identifier: resolved.rawValue) }

  static func systemDefault(preferredLanguages: [String] = Locale.preferredLanguages) -> AppLanguage
  {
    guard let identifier = preferredLanguages.first else { return .english }
    let parts = identifier.replacingOccurrences(of: "_", with: "-")
      .split(separator: "-")
      .map { $0.lowercased() }
    guard let language = parts.first else { return .english }
    if language == "en" { return .english }
    guard language == "zh" else { return .english }
    if parts.contains("hant") || parts.contains("tw") || parts.contains("hk")
      || parts.contains("mo")
    {
      return .traditionalChinese
    }
    return .simplifiedChinese
  }
}

enum L10n {
  static let preferenceKey = "appLanguage"

  static var selectedLanguage: AppLanguage {
    let value = UserDefaults.standard.string(forKey: preferenceKey)
    return (AppLanguage(rawValue: value ?? "") ?? .appDefault).resolved
  }

  static func string(
    _ key: String,
    language: AppLanguage = selectedLanguage,
    _ arguments: CVarArg...
  ) -> String {
    let format = bundle(for: language).localizedString(forKey: key, value: key, table: nil)
    guard !arguments.isEmpty else { return format }
    return String(format: format, locale: language.locale, arguments: arguments)
  }

  private static func bundle(for language: AppLanguage) -> Bundle {
    let resolvedLanguage = language.resolved
    for base in [Bundle.main, Bundle.module] {
      for resourceName in [resolvedLanguage.rawValue, resolvedLanguage.rawValue.lowercased()] {
        guard let path = base.path(forResource: resourceName, ofType: "lproj"),
          let bundle = Bundle(path: path)
        else { continue }
        return bundle
      }
    }
    return .main
  }
}
