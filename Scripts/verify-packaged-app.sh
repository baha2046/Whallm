#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
app_path=${1:?Usage: verify-packaged-app.sh APP_PATH ZIP_PATH}
zip_path=${2:?Usage: verify-packaged-app.sh APP_PATH ZIP_PATH}
require_notarization=${REQUIRE_NOTARIZATION:-0}
verification_root=$(mktemp -d)
trap 'rm -rf "$verification_root"' EXIT

verify_signature() {
  local target=$1
  codesign --verify --deep --strict "$target"
  if [[ $require_notarization == 1 ]]; then
    xcrun stapler validate "$target"
    spctl --assess --type execute --verbose=2 "$target"
  fi
}

verify_localizations() {
  local target=$1
  local language
  for language in en zh-Hans zh-Hant; do
    local strings=$target/Contents/Resources/$language.lproj/Localizable.strings
    [[ -f $strings ]] || {
      print -u2 "Localization is missing: $strings"
      exit 1
    }
    plutil -lint "$strings" >/dev/null
  done
}

launch_without_module_bundle() {
  local target=$1
  local launch_app=$verification_root/launch-${target:t}
  local module_bundle=$launch_app/Contents/Resources/DeepSeekV4SSD_DeepSeekV4SSDApp.bundle
  local build_path=$project_root/.build
  local sandbox_profile
  local language

  ditto "$target" "$launch_app"
  [[ -d $module_bundle ]] || {
    print -u2 "Swift resource bundle is missing: $module_bundle"
    exit 1
  }
  rm -rf "$module_bundle"
  sandbox_profile="(version 1)(allow default)(deny file-read* (subpath \"$build_path\"))"

  for language in en zh-Hans zh-Hant; do
    local log_path=$verification_root/launch-$language.log
    sandbox-exec -p "$sandbox_profile" \
      "$launch_app/Contents/MacOS/dsv4-app" \
      -appLanguage "$language" >"$log_path" 2>&1 &
    local app_pid=$!
    sleep 3
    if ! kill -0 "$app_pid" 2>/dev/null; then
      wait "$app_pid" || true
      print -u2 "Packaged App stopped during $language localization startup."
      sed -n '1,160p' "$log_path" >&2
      exit 1
    fi
    kill -TERM "$app_pid"
    wait "$app_pid" || true
  done
}

[[ -d $app_path && -f $zip_path ]] || {
  print -u2 "Packaged App or ZIP is missing."
  exit 1
}

verify_signature "$app_path"
verify_localizations "$app_path"

ditto -x -k "$zip_path" "$verification_root/extracted"
extracted_app=$verification_root/extracted/DeepSeekV4SSD.app
[[ -d $extracted_app ]] || {
  print -u2 "The ZIP does not contain DeepSeekV4SSD.app."
  exit 1
}
verify_signature "$extracted_app"
verify_localizations "$extracted_app"
launch_without_module_bundle "$extracted_app"

print "Packaged App verification passed: $app_path"
print "Extracted ZIP verification passed: $zip_path"
