"""ONNX Runtime install discovery, module registry, header listing, include graph."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Module definitions: (module_name, header_names_or_prefixes).
ORT_MODULES: list[tuple[str, list[str]]] = [
    ("Core", [
        "onnxruntime_c_api.h",
        "onnxruntime_cxx_api.h",
        "onnxruntime_run_options_config_keys.h",
        "onnxruntime_session_options_config_keys.h",
    ]),
    ("Providers", [
        "cpu_provider_factory.h",
        "cuda_provider_factory.h",
        "tensorrt_provider_factory.h",
        "dml_provider_factory.h",
        "coreml_provider_factory.h",
        "nnapi_provider_factory.h",
        "openvino_provider_factory.h",
        "xnnpack_provider_factory.h",
        "rocm_provider_factory.h",
    ]),
]

MODULE_BY_NAME = {name: headers for name, headers in ORT_MODULES}


@dataclass
class ORTInstall:
    """A resolved ONNX Runtime install (vcpkg or system)."""
    include_dir: Path
    source: str  # "vcpkg" | "explicit" | "system"
    version: str = ""

    def header(self, name: str) -> Path:
        p = self.include_dir / name
        if not p.exists():
            for sub_path in [
                self.include_dir / "onnxruntime" / name,
                self.include_dir / "onnxruntime" / "core" / "session" / name,
                self.include_dir / "core" / "session" / name,
            ]:
                if sub_path.exists():
                    return sub_path
            matches = list(self.include_dir.glob(f"**/{name}"))
            if matches:
                return matches[0]
        return p


def _read_version(include_dir: Path) -> str:
    vh = include_dir / "onnxruntime_c_api.h"
    if not vh.exists():
        for sub in [include_dir / "onnxruntime" / "core" / "session" / "onnxruntime_c_api.h",
                    include_dir / "onnxruntime" / "onnxruntime_c_api.h",
                    include_dir / "core" / "session" / "onnxruntime_c_api.h"]:
            if sub.exists():
                vh = sub
                break
        else:
            return "1.23.2"
    try:
        text = vh.read_text(errors="replace")
    except OSError:
        return "1.23.2"
    m = re.search(r"#define\s+ORT_API_VERSION\s+(\d+)", text)
    if m:
        return f"1.23.{m.group(1)}"
    return "1.23.2"


def find_ort_install(project_root: Path | None = None) -> ORTInstall:
    """Locate the ONNX Runtime include dir, preferring the project's vcpkg install."""
    candidates: list[tuple[Path, str]] = []
    explicit = os.environ.get("ONNXRUNTIME_INCLUDE_DIR") or os.environ.get("ORT_INCLUDE_DIR")
    if explicit:
        candidates.append((Path(explicit), "explicit"))
    if project_root is not None:
        triplet = os.environ.get("VCPKG_DEFAULT_TRIPLET", "x64-linux")
        candidates.append((project_root / "vcpkg" / "installed" / triplet / "include" / "onnxruntime", "vcpkg"))
        candidates.append((project_root / "vcpkg" / "installed" / triplet / "include", "vcpkg"))
        candidates.append((project_root / "vcpkg" / "installed" / triplet / "lib" / "onnxruntime.framework" / "Headers", "vcpkg"))
        inst_root = project_root / "vcpkg" / "installed"
        if inst_root.exists():
            for inst_dir in inst_root.glob("*/include"):
                candidates.append((inst_dir / "onnxruntime", "vcpkg"))
                candidates.append((inst_dir, "vcpkg"))
            for framework_dir in inst_root.glob("*/lib/onnxruntime.framework/Headers"):
                candidates.append((framework_dir, "vcpkg"))

    # System fallback
    candidates.append((Path("/usr/include/onnxruntime"), "system"))
    candidates.append((Path("/usr/include"), "system"))

    for include_dir, source in candidates:
        if not include_dir.exists():
            continue
        if (include_dir / "onnxruntime_c_api.h").exists() or (include_dir / "onnxruntime_cxx_api.h").exists():
            return ORTInstall(include_dir=include_dir, source=source, version=_read_version(include_dir))
        for sub in [include_dir / "onnxruntime", include_dir / "onnxruntime" / "core" / "session", include_dir / "core" / "session"]:
            if (sub / "onnxruntime_c_api.h").exists() or (sub / "onnxruntime_cxx_api.h").exists():
                return ORTInstall(include_dir=include_dir, source=source, version=_read_version(sub))

    raise FileNotFoundError(
        "No ONNX Runtime install found; set ONNXRUNTIME_INCLUDE_DIR or install via vcpkg "
        f"(checked: {[str(d) for d, _ in candidates] or 'none'}).")


_install: ORTInstall | None = None


def get_install(project_root: Path | None = None) -> ORTInstall:
    global _install
    if _install is None:
        _install = find_ort_install(project_root)
    return _install


def module_headers(module_name: str, install: ORTInstall) -> list[Path]:
    """All headers in `install.include_dir` belonging to `module_name`."""
    headers_spec = MODULE_BY_NAME.get(module_name, [])
    found: list[Path] = []
    for h in headers_spec:
        p = install.header(h)
        if p.exists():
            found.append(p)
        else:
            # Look in include_dir directly or glob
            matches = list(install.include_dir.glob(f"**/{h}"))
            if matches:
                found.append(matches[0])
    return sorted(found)


def transitive_closure_for_header(header_path: Path, install: ORTInstall) -> list[Path]:
    """Collect headers included by header_path."""
    included: list[Path] = []
    try:
        text = header_path.read_text(errors="replace")
    except OSError:
        return included
    for match in re.finditer(r'#include\s+[<"]([^>"]+)[>"]', text):
        inc_name = match.group(1)
        inc_p = install.header(inc_name)
        if inc_p.exists():
            included.append(inc_p)
    return included
