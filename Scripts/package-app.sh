#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
python_executable=${PYTHON_EXECUTABLE:-$project_root/.venv/bin/python}
output_root=$project_root/dist
app_path=$output_root/DeepSeekV4SSD.app
zip_path=$output_root/DeepSeekV4SSD-macOS-arm64.zip

if [[ ! -x $python_executable ]]; then
  print -u2 "Python environment not found: $python_executable"
  print -u2 "Create .venv and install requirements.txt before packaging."
  exit 1
fi

$python_executable -c 'import mlx, numpy, sentencepiece, tiktoken, transformers'

python_version=$($python_executable -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python_framework=$($python_executable -c 'import pathlib, sys; print(pathlib.Path(sys.base_prefix).parents[1])')
site_packages=$($python_executable -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
python_binary=$($python_executable -c 'import os, sys; print(os.path.realpath(sys._base_executable))')

if [[ ! -d $python_framework || ! -d $site_packages ]]; then
  print -u2 "The selected Python environment cannot be bundled."
  exit 1
fi

swift build --package-path "$project_root" -c release --product dsv4-app
binary_path=$(swift build --package-path "$project_root" -c release --show-bin-path)/dsv4-app

rm -rf "$app_path" "$zip_path"
mkdir -p "$app_path/Contents/MacOS" "$app_path/Contents/Resources/python" "$app_path/Contents/Frameworks"
ditto "$binary_path" "$app_path/Contents/MacOS/dsv4-app"
ditto "$project_root/Packaging/Info.plist" "$app_path/Contents/Info.plist"
ditto "$project_root/runtime" "$app_path/Contents/Resources/runtime"
ditto "$site_packages" "$app_path/Contents/Resources/python/site-packages"
ditto "$python_framework" "$app_path/Contents/Frameworks/Python.framework"
ditto "$python_binary" "$app_path/Contents/MacOS/python3"
rm -f "$app_path/Contents/Frameworks/Python.framework/Versions/$python_version/lib/python$python_version/site-packages"

old_python_library=$(otool -L "$app_path/Contents/MacOS/python3" | sed -n '2{s/^[[:space:]]*//;s/ (.*$//;p;}')
install_name_tool -change "$old_python_library" \
  "@executable_path/../Frameworks/Python.framework/Versions/$python_version/Python" \
  "$app_path/Contents/MacOS/python3"

/usr/bin/find "$app_path" -type d -name __pycache__ -prune -exec rm -rf {} +
/usr/bin/find "$app_path" -type f -name '*.pyc' -delete

code_sign_identity=${CODE_SIGN_IDENTITY:--}
code_sign_arguments=(--force --deep --sign "$code_sign_identity")
if [[ $code_sign_identity != - ]]; then
  code_sign_arguments+=(--options runtime --timestamp)
fi
codesign "${code_sign_arguments[@]}" "$app_path"
codesign --verify --deep --strict "$app_path"

PYTHONHOME="$app_path/Contents/Frameworks/Python.framework/Versions/Current" \
PYTHONPATH="$app_path/Contents/Resources/runtime:$app_path/Contents/Resources/python/site-packages" \
PYTHONDONTWRITEBYTECODE=1 \
  "$app_path/Contents/MacOS/python3" -c 'import mlx, deepseek_v4_ssd.server'

ditto -c -k --keepParent "$app_path" "$zip_path"

if [[ -n ${NOTARY_PROFILE:-} ]]; then
  if [[ $code_sign_identity == - ]]; then
    print -u2 "CODE_SIGN_IDENTITY is required when NOTARY_PROFILE is set."
    exit 1
  fi
  xcrun notarytool submit "$zip_path" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$app_path"
  rm -f "$zip_path"
  ditto -c -k --keepParent "$app_path" "$zip_path"
fi

print "App: $app_path"
print "Archive: $zip_path"
