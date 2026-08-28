#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
version=${VERSION:?Set VERSION, for example: make release VERSION=1.1.0}
tag=${TAG:-v$version}
repository=${GITHUB_REPOSITORY:-yanun0323/Whallm}
archive_name=Whallm-macOS-arm64.zip
archive_path=$project_root/dist/$archive_name
sparkle_tools=$project_root/.build/artifacts/sparkle/Sparkle/bin
notes_path=${RELEASE_NOTES_FILE:-$project_root/Packaging/ReleaseNotes/$version.md}
release_root=$(mktemp -d)
download_root=$(mktemp -d)
trap 'rm -rf "$release_root" "$download_root"' EXIT

if ! command -v gh >/dev/null || ! gh auth status >/dev/null 2>&1; then
  print -u2 "GitHub CLI is not signed in."
  exit 1
fi

if [[ -z ${CODE_SIGN_IDENTITY:-} || -z ${NOTARY_PROFILE:-} ]]; then
  print -u2 "CODE_SIGN_IDENTITY and NOTARY_PROFILE are required for a release."
  exit 1
fi

if [[ ! -f $notes_path ]]; then
  print -u2 "Release notes not found: $notes_path"
  exit 1
fi

if gh release view "$tag" --repo "$repository" >/dev/null 2>&1; then
  print -u2 "GitHub Release already exists: $tag"
  exit 1
fi

APP_VERSION=$version BUILD_VERSION=${BUILD_VERSION:-$version} \
  "$project_root/Scripts/package-app.sh"

REQUIRE_NOTARIZATION=1 "$project_root/Scripts/verify-packaged-app.sh" \
  "$project_root/dist/Whallm.app" "$archive_path"

ditto "$archive_path" "$release_root/$archive_name"
ditto "$notes_path" "$release_root/${archive_name:r}.md"

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
  --title "Whallm $version" \
  --notes-file "$notes_path"

gh release download "$tag" \
  --repo "$repository" \
  --pattern "$archive_name" \
  --pattern appcast.xml \
  --dir "$download_root"

[[ -f $download_root/$archive_name && -f $download_root/appcast.xml ]] || {
  print -u2 "GitHub Release does not contain the ZIP and appcast.xml."
  exit 1
}
REQUIRE_NOTARIZATION=1 "$project_root/Scripts/verify-packaged-app.sh" \
  "$project_root/dist/Whallm.app" "$download_root/$archive_name"

release_url=$(gh release view "$tag" --repo "$repository" --json url --jq .url)
print "Release: $release_url"
shasum -a 256 "$download_root/$archive_name" "$download_root/appcast.xml"
