#!/usr/bin/env bash
# Build the experimental Go/C++ runtime on the host architecture.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --app-only ) ]]; then
    echo 'Usage: bash native/build.sh [--app-only]' >&2
    exit 1
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPS="$ROOT/dist/native-deps"
BUILD="$ROOT/native/build"
DEST="$ROOT/dist/native-runtime"
JOBS=${JOBS:-4}
CMAKE=${CMAKE:-cmake}
if ! command -v "$CMAKE" >/dev/null 2>&1; then
    CMAKE="$ROOT/dist/native-build-tools/cmake/data/bin/cmake"
fi
if [[ ! -x "$CMAKE" ]] && ! command -v "$CMAKE" >/dev/null 2>&1; then
    echo 'CMake >= 3.20 is required (build dependency only).' >&2
    exit 1
fi
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
        ORT_PLATFORM=osx-arm64
        LIB_RPATH='@loader_path'
        BIN_RPATH='@executable_path/../lib'
        LAPACK=ON
        ;;
    Linux-x86_64)
        ORT_PLATFORM=linux-x64
        LIB_RPATH='$ORIGIN'
        BIN_RPATH='$ORIGIN/../lib'
        LAPACK=OFF
        ;;
    *) echo 'Supported build hosts: Linux x86_64 and macOS arm64.' >&2; exit 1 ;;
esac
mkdir -p "$DEPS/downloads" "$DEPS/json/nlohmann" "$DEST/bin" "$DEST/lib" "$DEST/licenses" "$BUILD"
sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
fetch() {
    local url=$1 path=$2 expected=$3
    if [[ ! -f "$path" ]]; then
        curl --fail --location --retry 2 --silent --show-error "$url" -o "$path.partial"
        if [[ "$(sha256 "$path.partial")" != "$expected" ]]; then
            echo "Checksum mismatch: $path.partial" >&2; exit 1
        fi
        mv "$path.partial" "$path"
    fi
    [[ "$(sha256 "$path")" == "$expected" ]] || { echo "Checksum mismatch: $path" >&2; exit 1; }
}
ORT="onnxruntime-$ORT_PLATFORM-1.29.0"
if [[ ! -d "$DEPS/$ORT" ]]; then
    python3 "$ROOT/tools/fetch_onnxruntime.py" --platform "$ORT_PLATFORM" \
        --destination "$DEPS/$ORT" --cache "$DEPS/downloads"
fi
fetch 'https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp' "$DEPS/json/nlohmann/json.hpp" aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
fetch 'https://raw.githubusercontent.com/nlohmann/json/v3.12.0/LICENSE.MIT' "$DEPS/json/LICENSE" 46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603
OPENCV_SHA=b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095
fetch 'https://github.com/opencv/opencv/archive/refs/tags/5.0.0.tar.gz' "$DEPS/downloads/opencv-5.0.0.tar.gz" "$OPENCV_SHA"
if [[ ! -d "$DEPS/opencv-5.0.0" ]]; then tar -xzf "$DEPS/downloads/opencv-5.0.0.tar.gz" -C "$DEPS"; fi
if [[ "${1:-}" != --app-only ]]; then
"$CMAKE" -S "$DEPS/opencv-5.0.0" -B "$DEPS/opencv-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$DEPS/opencv" \
    -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH="$LIB_RPATH" \
    -DCMAKE_INSTALL_NAME_DIR=@rpath -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_LIST=core,imgproc,imgcodecs,geometry -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
    -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_TBB=OFF \
    -DWITH_VA=OFF -DWITH_VA_INTEL=OFF -DWITH_GDAL=OFF -DWITH_GDCM=OFF -DWITH_SPNG=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=OFF \
    -DWITH_V4L=OFF -DWITH_OBSENSOR=OFF -DWITH_OPENEXR=OFF -DWITH_TIFF=OFF \
    -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_JPEG=ON -DBUILD_JPEG=ON \
    -DWITH_AVIF=OFF -DWITH_JPEGXL=OFF -DWITH_LAPACK="$LAPACK" \
    -DWITH_PNG=ON -DBUILD_PNG=ON -DWITH_WEBP=ON -DBUILD_WEBP=ON -DBUILD_ZLIB=ON
"$CMAKE" --build "$DEPS/opencv-build" --parallel "$JOBS"
"$CMAKE" --install "$DEPS/opencv-build"
fi
"$CMAKE" -S "$ROOT/native" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DEST" -DOpenCV_DIR="$DEPS/opencv/lib/cmake/opencv5" \
    -DONNXRUNTIME_ROOT="$DEPS/$ORT" -DJSON_INCLUDE_DIR="$DEPS/json"
"$CMAKE" --build "$BUILD" --parallel "$JOBS"
"$CMAKE" --install "$BUILD"
"$CMAKE" --build "$BUILD" --target test
find "$DEPS/opencv/lib" "$DEPS/$ORT/lib" -maxdepth 1 \( -name '*.dylib' -o -name '*.so*' \) -exec cp -P {} "$DEST/lib/" \;
cp "$DEPS/opencv-5.0.0/LICENSE" "$DEST/licenses/OpenCV-LICENSE"
cp "$DEPS/$ORT/LICENSE" "$DEST/licenses/ONNX-Runtime-LICENSE"
cp "$DEPS/$ORT/ThirdPartyNotices.txt" "$DEST/licenses/ONNX-Runtime-ThirdPartyNotices.txt"
cp "$DEPS/$ORT/Privacy.md" "$DEST/licenses/ONNX-Runtime-Privacy.md"
cp "$DEPS/json/LICENSE" "$DEST/licenses/nlohmann-json-LICENSE"
cp "$DEPS/opencv-5.0.0/3rdparty/libjpeg-turbo/LICENSE.md" "$DEST/licenses/libjpeg-turbo-LICENSE.md"
cp "$DEPS/opencv-5.0.0/3rdparty/libjpeg-turbo/README.ijg" "$DEST/licenses/libjpeg-README.ijg"
cp "$DEPS/opencv-5.0.0/3rdparty/libpng/LICENSE" "$DEST/licenses/libpng-LICENSE"
cp "$DEPS/opencv-5.0.0/3rdparty/libwebp/COPYING" "$DEST/licenses/libwebp-COPYING"
cp "$DEPS/opencv-5.0.0/3rdparty/zlib/LICENSE" "$DEST/licenses/zlib-LICENSE"
sed '/^#ifndef/,$d' "$DEPS/opencv-5.0.0/hal/carotene/src/common.hpp" > "$DEST/licenses/carotene-LICENSE"
if [[ -f "$DEPS/opencv-build/3rdparty/kleidicv/kleidicv-26.03/LICENSES/Apache-2.0.txt" ]]; then
    cp "$DEPS/opencv-build/3rdparty/kleidicv/kleidicv-26.03/LICENSES/Apache-2.0.txt" "$DEST/licenses/KleidiCV-LICENSE"
fi
GO_LICENSE="$(go env GOROOT)/LICENSE"
if [[ ! -f "$GO_LICENSE" ]]; then GO_LICENSE="$(go env GOROOT)/../LICENSE"; fi
cp "$GO_LICENSE" "$DEST/licenses/Go-LICENSE"
cp "$ROOT/LICENSE" "$DEST/licenses/Seiyuu-FaceID-LICENSE"
export GOCACHE="$BUILD/go-cache"
export GOTOOLCHAIN=local
export CGO_ENABLED=1
export CGO_CFLAGS="-I$ROOT/native/include"
export CGO_LDFLAGS="-L$DEST/lib -Wl,-rpath,$BIN_RPATH"
cd "$ROOT/native/go"
go test ./...
CGO_LDFLAGS="-L$DEST/lib -Wl,-rpath,$DEST/lib" go test -tags native ./...
go build -tags native -trimpath -o "$DEST/bin/faceid-server" ./cmd/faceid
go build -trimpath -o "$DEST/bin/faceid-bench" ./cmd/faceid-bench
echo "Built $DEST/bin/faceid-server"
