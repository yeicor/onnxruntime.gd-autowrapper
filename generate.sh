#!/usr/bin/env bash
# Regenerate the ONNX Runtime autowrapper bindings (../src/autowrapper) from the ONNX Runtime
# headers installed via vcpkg (or ONNXRUNTIME_INCLUDE_DIR). Used by CI and by validate.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "
import sys
from autogen.ort import find_ort_install
from autogen.cli import PROJECT_ROOT
find_ort_install(PROJECT_ROOT)
" >/dev/null 2>&1; then
    echo "No ONNX Runtime install found (install via vcpkg or set ONNXRUNTIME_INCLUDE_DIR)." >&2
    echo "Skipping autowrapper generation." >&2
    exit 0
fi

if ! "$PYTHON" -c "import clang.cindex" >/dev/null 2>&1; then
    echo "python clang (libclang) bindings not available; cannot generate the autowrapper." >&2
    echo "Install them with: \"$PYTHON\" -m pip install clang" >&2
    exit 1
fi

NPROC="$(nproc 2>/dev/null || echo 4)"
AUTOWRAPPER_JOBS="${AUTOWRAPPER_JOBS:-$(( NPROC > 8 ? 8 : NPROC ))}"

"$PYTHON" -m autogen regenerate --jobs "${AUTOWRAPPER_JOBS}"