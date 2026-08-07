#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
version=${VERSION:?Set VERSION, for example: make release VERSION=1.1.0}
tag=${TAG:-v$version}
repository=${GITHUB_REPOSITORY:-yanun0323/deepseek_ssd}
archive_name=DeepSeekV4SSD-macOS-arm64.zip
archive_path=$project_root/dist/$archive_name
sparkle_tools=$project_root/.build/artifacts/sparkle/Sparkle/bin
release_root=$(mktemp -d)
trap 'rm -rf "$release_root"' EXIT

if ! command -v gh >/dev/null || ! gh auth status >/dev/null 2>&1; then
  print -u2 "GitHub CLI is not signed in."
  exit 1
fi

if [[ -z ${CODE_SIGN_IDENTITY:-} || -z ${NOTARY_PROFILE:-} ]]; then
  print -u2 "CODE_SIGN_IDENTITY and NOTARY_PROFILE are required for a release."
  exit 1
fi

if gh release view "$tag" --repo "$repository" >/dev/null 2>&1; then
  print -u2 "GitHub Release already exists: $tag"
  exit 1
fi

APP_VERSION=$version BUILD_VERSION=${BUILD_VERSION:-$version} \
  "$project_root/Scripts/package-app.sh"

ditto "$archive_path" "$release_root/$archive_name"
print "DeepSeekV4SSD $version" > "$release_root/${archive_name:r}.md"

"$sparkle_tools/generate_appcast" \
  --account deepseek_ssd \
  --download-url-prefix "https://github.com/$repository/releases/download/$tag/" \
  --link "https://github.com/$repository/releases/tag/$tag" \
  --embed-release-notes \
  --maximum-versions 1 \
  "$release_root"

"$sparkle_tools/sign_update" --account deepseek_ssd --verify "$release_root/appcast.xml"

gh release create "$tag" \
  "$release_root/$archive_name" \
  "$release_root/appcast.xml" \
  --repo "$repository" \
  --title "DeepSeekV4SSD $version" \
  --notes "DeepSeekV4SSD $version"

print "Release: https://github.com/$repository/releases/tag/$tag"
