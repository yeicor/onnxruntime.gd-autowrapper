"""Command-line entry point for the ONNX Runtime autogen pipeline."""

from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .codegen import generate_all, generate_module
from .classify import classify_module
from .compile_db import (CompileArgs, ensure_occt_args, probe_data_model)
from .ir import load_module
from .ort import ORT_MODULES, find_ort_install
from .scanner import ModuleScanResult, scan_module, to_dict

# The autowrapper submodule lives next to the project root.
SUBMODULE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SUBMODULE_DIR.parent
DEFAULT_COMPILE_DB = PROJECT_ROOT / ".build-autowrapper" / "compile_commands.json"
DATA_DIR = Path(__file__).resolve().parent / "data"
BASELINE_MISSING = DATA_DIR / "skips-missing.txt"
BASELINE_ILLFORMED = DATA_DIR / "skips-illformed.txt"


def _default_jobs() -> int:
    return max(1, min(os.cpu_count() or 4, 8))


def _count_metrics(result: ModuleScanResult) -> None:
    n_methods = sum(len(c.all_methods) for c in result.classes)
    n_wrap = sum(len(c.all_wrappable_methods) for c in result.classes)
    print(f"module         : {result.module}")
    print(f"headers        : {result.headers} (closure-retried: {result.attempts2})")
    print(f"classes        : {len(result.classes)}")
    print(f"enums          : {len(result.enums)}")
    print(f"typedefs       : {len(result.typedefs)}")
    print(f"methods        : {n_methods} (wrappable: {n_wrap})")
    if result.errors:
        print(f"errors         : {len(result.errors)}")
        for h, e in sorted(result.errors.items())[:5]:
            print(f"  - {h}: {e[:120]}")


def cmd_scan(args: argparse.Namespace) -> int:
    if args.module not in {m for m, _ in ORT_MODULES}:
        sys.exit(f"unknown module: {args.module}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    install = find_ort_install(PROJECT_ROOT)
    compile_args = CompileArgs(args.compile_db)
    args_list = ensure_occt_args(compile_args.args, install.include_dir)
    data_model = probe_data_model(args_list)
    result = scan_module(args.module, install, args_list, jobs=args.jobs)
    payload = to_dict(result, data_model)
    payload["ort_version"] = install.version
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=_json_default))
    _count_metrics(result)
    print(f"wrote          : {out}")
    return 0


def cmd_scan_all(args: argparse.Namespace) -> int:
    """Scan every ORT module into `out/ir/*.json`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    install = find_ort_install(PROJECT_ROOT)
    compile_args = CompileArgs(args.compile_db)
    args_list = ensure_occt_args(compile_args.args, install.include_dir)
    data_model = probe_data_model(args_list)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = [name for name, _ in ORT_MODULES]
    results: dict[str, ModuleScanResult] = {}

    def scan_one(name: str) -> tuple[str, ModuleScanResult]:
        return name, scan_module(name, install, args_list, jobs=1)

    workers = max(1, min(args.jobs, len(modules)))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(scan_one, m): m for m in modules}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    _, res = fut.result()
                except Exception as e:
                    continue
                results[name] = res

    for name in modules:
        if name in results and not results[name].errors:
            continue
        try:
            results[name] = scan_module(name, install, args_list, jobs=args.jobs)
        except Exception as e:
            print(f"scan-all: {name} failed: {e}", file=sys.stderr)

    failures = {name: res.errors for name, res in results.items() if res.errors}
    for name, res in results.items():
        payload = to_dict(res, data_model)
        payload["ort_version"] = install.version
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=1, default=_json_default))
        _count_metrics(res)

    if failures:
        print(f"scan-all: {len(failures)} module(s) with scan errors:", file=sys.stderr)
        for name, errs in sorted(failures.items()):
            print(f"  - {name}: {len(errs)} header error(s)", file=sys.stderr)
    print(f"wrote          : {out_dir} ({len(results)} modules)")
    return 0


def _json_default(o):
    if isinstance(o, enum.Enum):
        return o.value
    return str(o)


def cmd_generate(args: argparse.Namespace) -> int:
    src = Path(args.ir)
    module = load_module(src)
    classify_module(module)
    generate_module(module, Path(args.out))
    print(f"wrote          : {args.out}")
    return 0


def cmd_generate_all(args: argparse.Namespace) -> int:
    modules = [load_module(Path(p)) for p in args.irs]
    global_by_name = {cls.name: cls for m in modules for cls in m.classes}
    for module in modules:
        classify_module(module, global_by_name)
    missing = set()
    illformed = set()
    generate_all(modules, Path(args.out), probe_out=args.probe_out,
                 missing=missing, illformed=illformed,
                 module_filter=args.module_filter)
    print(f"wrote          : {args.out} ({len(modules)} modules)")
    return 0


def _run_cli(argv: list[str]) -> int:
    return subprocess.run([sys.executable, "-m", "autogen", *argv],
                          cwd=SUBMODULE_DIR).returncode


def cmd_regenerate(args: argparse.Namespace) -> int:
    from .ort import find_ort_install

    try:
        install = find_ort_install(PROJECT_ROOT)
    except FileNotFoundError as e:
        print(f"regenerate     : {e}", file=sys.stderr)
        print("regenerate     : no ORT install found; skipping generation", file=sys.stderr)
        return 0
    try:
        import clang.cindex
    except ImportError:
        print("regenerate     : python clang (libclang) bindings not available; "
              "cannot generate the autowrapper", file=sys.stderr)
        return 1

    out_dir = Path(args.ir_out)
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = _run_cli(["scan-all", "--jobs", str(args.jobs),
                   "--out", str(out_dir), "--compile-db", str(args.compile_db)])
    if rc != 0:
        return rc
    irs = sorted(str(p) for p in out_dir.glob("*.json"))

    rc = _run_cli(["generate-all", *irs, "--out", str(args.out)])
    if rc != 0:
        return rc

    stamp = Path(args.out) / ".autowrapper-stamp"
    stamp.write_text(f"ort={install.version}\n")
    print(f"regenerate     : stamp written to {stamp}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ONNX Runtime godot-cpp autowrapper pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan_p = sub.add_parser("scan", help="Scan one module to IR JSON")
    scan_p.add_argument("--module", required=True, help="Module name (e.g. Core)")
    scan_p.add_argument("--out", default="out/ir/Core.json", help="Output IR JSON path")
    scan_p.add_argument("--compile-db", default=str(DEFAULT_COMPILE_DB), help="compile_commands.json path")
    scan_p.add_argument("--jobs", type=int, default=_default_jobs(), help="Parallel parse workers")
    scan_p.set_defaults(func=cmd_scan)

    scan_all_p = sub.add_parser("scan-all", help="Scan all modules to out/ir/*.json")
    scan_all_p.add_argument("--out", default="out/ir", help="Output IR directory")
    scan_all_p.add_argument("--compile-db", default=str(DEFAULT_COMPILE_DB), help="compile_commands.json path")
    scan_all_p.add_argument("--jobs", type=int, default=_default_jobs(), help="Parallel parse workers")
    scan_all_p.set_defaults(func=cmd_scan_all)

    gen_p = sub.add_parser("generate", help="Generate wrapper C++ from one IR JSON")
    gen_p.add_argument("ir", help="Input IR JSON path")
    gen_p.add_argument("--out", default="../src/autowrapper", help="Output source directory")
    gen_p.set_defaults(func=cmd_generate)

    gen_all_p = sub.add_parser("generate-all", help="Generate wrapper C++ from all IR JSONs")
    gen_all_p.add_argument("irs", nargs="+", help="Input IR JSON paths")
    gen_all_p.add_argument("--out", default="../src/autowrapper", help="Output source directory")
    gen_all_p.add_argument("--probe-out", default="out/audit", help="Output directory for symbol audit probe")
    gen_all_p.add_argument("--module-filter", default=None, help="Comma-separated module names to generate")
    gen_all_p.set_defaults(func=cmd_generate_all)

    regen_p = sub.add_parser("regenerate", help="Run full scan-all + generate-all pipeline")
    regen_p.add_argument("--jobs", type=int, default=_default_jobs(), help="Parallel parse workers")
    regen_p.add_argument("--compile-db", default=str(DEFAULT_COMPILE_DB), help="compile_commands.json path")
    regen_p.add_argument("--ir-out", default="out/ir", help="Intermediate IR directory")
    regen_p.add_argument("--out", default="../src/autowrapper", help="Output source directory")
    regen_p.set_defaults(func=cmd_regenerate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
