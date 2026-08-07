import Foundation

enum AppLanguage: String, CaseIterable, Identifiable {
  case english = "en"
  case simplifiedChinese = "zh-Hans"
  case traditionalChinese = "zh-Hant"

  static let appDefault = AppLanguage.english

  var id: String { rawValue }

  var displayName: String {
    switch self {
    case .english: "English"
    case .simplifiedChinese: "简体中文"
    case .traditionalChinese: "繁體中文"
    }
  }

  var locale: Locale { Locale(identifier: rawValue) }
}

enum L10n {
  static let preferenceKey = "appLanguage"

  static var selectedLanguage: AppLanguage {
    let value = UserDefaults.standard.string(forKey: preferenceKey)
    return AppLanguage(rawValue: value ?? "") ?? .appDefault
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
    for base in [Bundle.main, Bundle.module] {
      guard let path = base.path(forResource: language.rawValue, ofType: "lproj"),
        let bundle = Bundle(path: path)
      else { continue }
      return bundle
    }
    return .main
  }
}
