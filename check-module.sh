#!/usr/bin/env bash
# Fast dev loop: syntax-check ONE generated wrapper module (or a single .cpp)
#
# Usage: ./check-module.sh <OrtModule or path-to.cpp>   (default: OrtEnv)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARG="${1:-OrtEnv}"
if [[ "$ARG" == *.cpp ]]; then
    CPPS=("$ARG")
else
    mapfile -t CPPS < <(ls "$ROOT"/src/autowrapper/"$ARG"*.cpp 2>/dev/null)
fi
if [ "${#CPPS[@]}" -eq 0 ]; then
    echo "no generated cpp matched: $ARG" >&2
    exit 2
fi
echo "checking ${#CPPS[@]} translation unit(s): ${CPPS[0]##*/} ..."

ORT="$ROOT/vcpkg/installed/x64-linux/include/onnxruntime"

CXX="${CXX:-c++}"
set -x
"$CXX" -std=gnu++17 -fsyntax-only -fPIC \
    -DDEBUG_ENABLED -DGDEXTENSION \
    -DONNXRuntime_gd_EXPORTS -DTHREADS_ENABLED \
    -include "$ROOT/src/ort_guard.h" \
    -I"$ROOT/src" \
    -isystem "$ROOT/godot-cpp/include" \
    -isystem "$ROOT/src/autowrapper" \
    -isystem "$ORT" \
    "${CPPS[@]}"
