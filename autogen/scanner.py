"""Module-scoped scanning: headers -> libclang -> IR (JSON).

Parsing runs header-parallel across processes. Each header is parsed
standalone; if it is not self-contained (hard parse errors) it is retried with
its include closure pre-included. Declarations are attributed to the header
that defines them via `location.file`.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .extract import HeaderResult, extract_header
from .ort import ORTInstall, module_headers, transitive_closure_for_header
from .parser import ParseError, parse_header

log = logging.getLogger("autogen.scanner")


def _quoted_types(err: str) -> list[str]:
    """Deduplicated identifier(s) quoted in a libclang error."""
    out: list[str] = []
    for q in re.findall(r"'([^']*)'", err):
        q = q.strip().rstrip("*").strip()
        if not q or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:]*", q):
            continue
        q = q.split("::")[-1]
        if q not in out:
            out.append(q)
    return out


def _candidate_idents(err: str) -> list[str]:
    quoted = _quoted_types(err)
    if not quoted:
        return []
    if "static_cast" in err:
        return [quoted[-1]] + quoted[:-1]
    return [quoted[0]]


def _scan_one(header: str, args: list[str], install_dir: str,
              ) -> tuple[str, HeaderResult, int, str]:
    """Worker: parse one header (with closure retry) and extract its decls."""
    from .ort import ORTInstall
    from .parser import parse_header
    from .extract import extract_header

    install = ORTInstall(include_dir=Path(install_dir), source="")
    hpath = Path(header)
    try:
        tu = parse_header(hpath, args)
        return hpath.name, extract_header(hpath, tu), 1, ""
    except ParseError:
        pass

    pre = [p.name for p in transitive_closure_for_header(hpath, install)]
    for _ in range(5):
        try:
            tu = parse_header(hpath, args, pre_headers=pre)
            hr = extract_header(hpath, tu)
            hr.extra_includes = list(pre)
            return hpath.name, hr, 2, ""
        except ParseError as e:
            return hpath.name, HeaderResult(header=hpath.name), 2, str(e)
    return hpath.name, HeaderResult(header=hpath.name), 2, "too many retries"


@dataclass
class ModuleScanResult:
    module: str
    classes: list = field(default_factory=list)
    enums: list = field(default_factory=list)
    typedefs: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    headers: int = 0
    attempts2: int = 0


def _sort_key(name: str) -> str:
    return name.lower()


def scan_module(module: str, install: ORTInstall, args: list[str],
                jobs: int = 8, _reuse_install: bool = True) -> ModuleScanResult:
    headers = module_headers(module, install)
    result = ModuleScanResult(module=module, headers=len(headers))

    if not headers:
        log.warning("module %s: no headers", module)
        return result

    install_dir = str(install.include_dir)
    items = [(str(h), args, install_dir) for h in headers]

    results: list[HeaderResult] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_scan_one, *it): it[0] for it in items}
        for fut in as_completed(futs):
            name, hr, attempts, err = fut.result()
            if attempts > 1:
                result.attempts2 += 1
            if err:
                result.errors[name] = err
                continue
            results.append(hr)

    seen_classes: dict[str, object] = {}
    seen_enums: dict[str, object] = {}
    typedefs: dict[str, str] = {}
    for hr in results:
        for c in hr.classes:
            if c.name not in seen_classes:
                c.module_name = module
                if hr.extra_includes:
                    c.extra_occt_includes = hr.extra_includes
                seen_classes[c.name] = c
        for e in hr.enums:
            if e.name not in seen_enums:
                seen_enums[e.name] = e
        typedefs.update(dict(hr.typedefs))

    result.classes = sorted(seen_classes.values(), key=lambda c: _sort_key(c.name))
    result.enums = sorted(seen_enums.values(), key=lambda e: _sort_key(e.name))
    result.typedefs = dict(sorted(typedefs.items()))
    return result


def to_dict(result: ModuleScanResult,
            data_model: dict[str, int] | None = None) -> dict:
    return {
        "module": result.module,
        "classes": [asdict(c) for c in result.classes],
        "enums": [asdict(e) for e in result.enums],
        "typedefs": result.typedefs,
        "errors": result.errors,
        "headers": result.headers,
        "attempts2": result.attempts2,
        "ort_version": "",
        "data_model": data_model or {},
    }
