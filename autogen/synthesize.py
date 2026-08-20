"""Synthesize a concrete wrapper ClassDecl from a class-template specialization.

All structure comes from libclang (the CLASS_TEMPLATE cursor: members, macros
already expanded, source-text spellings).  Dependent types are resolved by
re-parsing a tiny probe TU whose struct inherits the *instantiated*
specialization, then running the pipeline's existing ``make_type`` on each
alias.  No source is hand-parsed and no class body is reconstructed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

import clang.cindex as C
from clang.cindex import CursorKind

from .compile_db import ensure_occt_args, find_resource_dir
from .extract import _extract_class
from .model import ModuleDecl, occt_name_to_wrapper
from .occt import find_occt_install
from .parser import ParseError, parse_header
from .types import make_type
from .typemap import PRIMITIVE_MAP

# autogen/ -> autowrapper/ -> project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_top_level(s: str) -> list[str]:
    """Split on top-level commas (angle-bracket depth aware)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _chunk_noncopyable(chunk: str, noncopyable: set[str]) -> bool:
    """True if a template-argument chunk is a value (not handle/pointer/ref)
    OCCT class with deleted/broken copy semantics."""
    chunk = chunk.strip()
    if not chunk:
        return False
    if chunk.startswith("opencascade::handle"):
        return False  # handles copy regardless of the pointee
    if chunk.endswith("*") or chunk.endswith("&"):
        return False  # pointers/references copy
    m = re.match(r"^[A-Za-z_]\w*\s*<(.*)>\s*$", chunk, re.S)
    if m:
        # Nested specialization: only the inner VALUE args can be non-copyable.
        return any(_chunk_noncopyable(inner, noncopyable)
                   for inner in _split_top_level(m.group(1)))
    return chunk in noncopyable


def _noncopyable_classes(modules: list[ModuleDecl]) -> set[str]:
    """OCCT value classes whose copy semantics are deleted or broken."""
    return {cls.name for m in modules for cls in m.classes
            if not cls.has_copy_assignment}


def _specialization_broken(spec_name: str, noncopyable: set[str]) -> bool:
    """A container specialization is ill-formed when a value template arg is
    a non-copyable OCCT class (e.g. NCollection_Sequence<CSLib_Class2d>, whose
    Append/Insert copy the item).  handle<X>, pointer and reference args are
    exempt."""
    m = re.match(r"^[A-Za-z_]\w*\s*<(.*)>\s*$", spec_name, re.S)
    if not m:
        return False
    return any(_chunk_noncopyable(inner, noncopyable)
               for inner in _split_top_level(m.group(1)))


def filter_noncopyable(classes: list[object], modules: list[ModuleDecl]) -> list[object]:
    """Drop synthesized specializations over non-copyable value args.

    Applied both when synthesizing fresh and when loading a cached spec list,
    so a stale cache cannot resurrect a specialization whose members do not
    compile (the probe TU instantiates every wrapped member symbol).
    """
    noncopyable = _noncopyable_classes(modules)
    kept = []
    for cls in classes:
        if _specialization_broken(getattr(cls, "name", ""), noncopyable):
            print(f"synth          : SKIP {cls.name} (non-copyable template arg)",
                  file=__import__("sys").stderr)
            continue
        kept.append(cls)
    return kept


def filter_undeclarable(classes: list[object], install: Path,
                        modules: list[ModuleDecl] | None = None) -> list[object]:
    """Drop cached specializations whose args are not publicly nameable.

    Rebuilds the spec map from cached ClassDecls so a stale cache cannot
    resurrect a specialization over private/protected nested types (those
    cannot be referenced from generated wrappers and trip the probe TU).
    """
    specs: dict[str, tuple[str, list[str]]] = {}
    templates = _template_registry(modules, install) if modules else {}
    for cls in classes:
        m = _SPEC_RE.match(getattr(cls, "name", ""))
        if not m:
            continue
        header = getattr(cls, "header_file", "") or templates.get(m.group(1), "")
        args = _split_args(m.group(2))
        if header and args:
            specs[cls.name] = (header, args)
    bad = _undeclarable_specs(specs, install, modules) if specs else set()
    header_map = _build_header_map(modules) if modules else None
    kept = []
    for cls in classes:
        if getattr(cls, "name", "") in bad:
            print(f"synth          : SKIP {cls.name} (template arg not publicly "
                  "nameable)", file=__import__("sys").stderr)
            continue
        m = _SPEC_RE.match(getattr(cls, "name", ""))
        if m and header_map:
            # A stale cache may carry a name-derived header that does not
            # exist (Graphic3d_Attribute.hxx); rebind to the declaring header
            # from the scanned IR so the wrapper compiles.
            args = _split_args(m.group(2))
            own = getattr(cls, "header_file", "") or ""
            cls.extra_occt_includes = [
                i for i in _collect_includes(args, header_map=header_map)
                if i != own]
        kept.append(cls)
    return kept


def filter_unwrappable(classes: list[object], modules: list[ModuleDecl],
                       install: Path) -> list[object]:
    """Drop cached specializations that can never compile as value wrappers:
    nested-template specs (``Root<T>::Nested<U>``) and abstract
    specializations (pure virtual members).  Run on cache load so a stale
    cache heals itself; mirrors ``filter_undeclarable``."""
    specs: dict[str, tuple[str, list[str]]] = {}
    templates = _template_registry(modules, install) if modules else {}
    for cls in classes:
        name = getattr(cls, "name", "")
        m = _SPEC_RE.match(name)
        if not m or _is_nested_spec(name):
            continue
        header = getattr(cls, "header_file", "") or templates.get(m.group(1), "")
        args = _split_args(m.group(2))
        if header and args:
            specs[name] = (header, args)
    abstract = _unwrappable_specs(specs, install, modules) if specs else set()
    kept = []
    for cls in classes:
        name = getattr(cls, "name", "")
        if _is_nested_spec(name):
            print(f"synth          : SKIP {name} (nested-template spec)",
                  file=__import__("sys").stderr)
            continue
        if name in abstract:
            print(f"synth          : SKIP {name} (unwrappable: abstract or no default ctor)", file=__import__("sys").stderr)
            continue
        kept.append(cls)
    return kept


_SPEC_RE = re.compile(r"^([A-Za-z_]\w*)<(.*)>$")

_ORT_RE = re.compile(r"\bOrt[A-Za-z_][A-Za-z0-9_]*\b")

_template_registry_cache: dict[str, str] | None = None

_TEMPLATE_REGISTRY_CACHE_PATH = Path(__file__).resolve().parents[1] \
    / "out" / "synth" / "templates.json"


def _template_registry(modules: list[ModuleDecl],
                       install: Path) -> dict[str, str]:
    """Root class-template name -> declaring header basename, by content.

    No codebase-specific allow-list: every ``Foo<...>`` referenced in the
    scanned API is a synthesis candidate.  Its header is located from the
    scanned IR (``Foo`` scanned as a class), the name-derived ``Foo.hxx``, or a
    text search for an actual ``template <...> class Foo`` declaration across
    the install headers.  The content-search results are cached under
    ``out/synth/templates.json`` so repeat loads stay cheap.
    """
    global _template_registry_cache
    if _template_registry_cache is not None:
        return _template_registry_cache

    roots: set[str] = set()
    for module in modules:
        for cls in module.classes:
            for meth in cls.all_methods:
                for t in [meth.return_type,
                          *(p.type for p in meth.parameters)]:
                    m = _SPEC_RE.match(getattr(t, "base_name", "") or "")
                    if m:
                        roots.add(m.group(1))
            for f in cls.fields:
                if f.is_public:
                    m = _SPEC_RE.match(getattr(f.type, "base_name", "") or "")
                    if m:
                        roots.add(m.group(1))

    header_map = _build_header_map(modules)
    cached: dict[str, str] = {}
    if _TEMPLATE_REGISTRY_CACHE_PATH.exists():
        try:
            cached = json.loads(
                _TEMPLATE_REGISTRY_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}

    reg: dict[str, str] = {}
    for root in sorted(roots):
        h = _template_header(root, header_map, install, cached)
        if h:
            reg[root] = h
    if reg and reg != cached:
        _TEMPLATE_REGISTRY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TEMPLATE_REGISTRY_CACHE_PATH.write_text(
            json.dumps(reg, indent=1), encoding="utf-8")
    _template_registry_cache = reg
    return reg


def _template_header(root: str, header_map: dict[str, str],
                     install: Path, cached: dict[str, str]) -> str | None:
    """Declaring header basename for a class-template root, or None.

    Order: scanned-IR location, name-derived ``Root.hxx``, cached content
    search, fresh content search for an actual ``template <...> class Root``.
    """
    h = header_map.get(root)
    if h:
        return h
    include_dir = install.include_dir
    h = f"{root}.hxx"
    if (include_dir / h).exists():
        return h
    if root in cached:
        return cached[root]
    pat = re.compile(rf"\btemplate\s*<[^;]*?>\s*(?:class|struct)\s+"
                     rf"{re.escape(root)}\b", re.M)
    for f in include_dir.glob("*.hxx"):
        try:
            if pat.search(f.read_text(encoding="utf-8", errors="replace")):
                return f.name
        except OSError:
            continue
    return None


_template_params_memo: dict[str, list[str]] = {}

_TEMPLATE_PARAMS_CACHE_PATH = Path(__file__).resolve().parents[1] \
    / "out" / "synth" / "template_params.json"


def _load_template_params_cache() -> None:
    """Seed ``_template_params_memo`` from disk.

    The parameter names of a class-template root are derived by parsing the
    root's header with libclang and walking the whole TU (~1s per root, first
    time).  Persist the result so the per-invocation ``upgrade_transitive`` /
    placeholder checks skip the parse on later runs, like ``templates.json``.
    """
    global _template_params_memo
    if _template_params_memo:
        return
    try:
        _template_params_memo = {
            k: list(v)
            for k, v in json.loads(
                _TEMPLATE_PARAMS_CACHE_PATH.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        _template_params_memo = {}


def _save_template_params_cache() -> None:
    """Persist new ``_template_params_memo`` entries (skips unchanged files)."""
    _TEMPLATE_PARAMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(
            _TEMPLATE_PARAMS_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if _template_params_memo != old:
        _TEMPLATE_PARAMS_CACHE_PATH.write_text(
            json.dumps(_template_params_memo, indent=1), encoding="utf-8")


def _template_param_names(template_name: str, templates: dict[str, str] | None,
                          install: Path | None) -> list[str]:
    """Declared parameter names of a class template root (type + non-type).

    Memoized in-process and persisted to ``out/synth/template_params.json`` so
    the per-run placeholder checks do not re-parse every template header;
    returns [] when the header/parse cannot be determined.
    """
    _load_template_params_cache()
    if template_name in _template_params_memo:
        return _template_params_memo[template_name]
    out: list[str] = []
    if templates and install is not None:
        header = templates.get(template_name)
        if header:
            args = ensure_occt_args([], install.include_dir)
            rd = find_resource_dir()
            if rd:
                args.append(f"-resource-dir={rd}")
            try:
                tu, _ = _parse_with_undeclared_retry(
                    install.include_dir / header, args, install.include_dir)
                ct = _find_class_template(tu.cursor, template_name)
                if ct is not None:
                    out = [c.spelling for c in ct.get_children()
                           if c.kind in _TEMPLATE_PARAM_KINDS]
            except Exception:  # noqa: BLE001
                out = []
    _template_params_memo[template_name] = out
    _save_template_params_cache()
    return out


def _has_placeholder_args(key: str, known: set[str],
                          templates: dict[str, str] | None = None,
                          install: Path | None = None) -> bool:
    """True if a specialization argument is an unresolved template-parameter
    name rather than a concrete type.

    In-class signatures of a class template spell the self type with the
    template-parameter names (``NCollection_IndexedDataMap<TheKeyType,
    TheItemType, Hasher>``); such spellings are not concrete types and must
    never be synthesized.  A bare identifier is a placeholder exactly when it
    is one of the root template's own parameter names.  When the root's
    parameters cannot be loaded, fall back to: unknown as a class, primitive
    or builtin.  A genuinely different identifier (``BVH_QuadTree``) is a real
    OCCT marker type, not a placeholder, and must be synthesized.
    """
    m = _SPEC_RE.match(key)
    if not m:
        return False
    tname = m.group(1)
    params = _template_param_names(tname, templates, install)
    for a in _split_args(m.group(2)):
        a = a.strip()
        if re.fullmatch(r"[A-Za-z_]\w*", a):
            if params:
                if a in params:
                    return True
            elif a not in known:
                return True
    return False


def _arg_prefix_covered(key: str, known: set[str],
                        templates: dict[str, str] | None = None,
                        install: Path | None = None,
                        header_map: dict[str, str] | None = None) -> bool:
    """True if `key` names the same type as a known spec with fewer args.

    OCCT spells defaulted trailing template parameters in some signatures, so
    ``NCollection_Map<K, NCollection_DefaultHasher<K>>`` is the same type as an
    already-known ``NCollection_Map<K>``.  A longer spelling is *only* the same
    type when its trailing arguments are the known spec's declared defaults
    (verified by expanding the defaults); a different explicit argument
    (``BVH_Tree<double, 3, BVH_QuadTree>`` vs ``BVH_Tree<double, 3>``) is a
    distinct specialization that must be synthesized on its own.
    """
    m = _SPEC_RE.match(key)
    if not m:
        return True
    tname, args = m.group(1), _split_args(m.group(2))
    for k in known:
        km = _SPEC_RE.match(k)
        if not km or km.group(1) != tname:
            continue
        ka = _split_args(km.group(2))
        if len(ka) > len(args) or ka != args[: len(ka)]:
            continue
        if len(ka) == len(args):
            return True
        expanded = _expanded_spec_args(k, templates, install, header_map)
        if len(expanded) == len(args) and expanded == args:
            return True
    return False


_expanded_spec_memo: dict[str, list[str]] = {}


def _expanded_spec_args(key: str, templates: dict[str, str] | None,
                        install: Path | None,
                        header_map: dict[str, str] | None) -> list[str]:
    """`key`'s arguments with the primary template's defaults filled in."""
    if key in _expanded_spec_memo:
        return _expanded_spec_memo[key]
    m = _SPEC_RE.match(key)
    out = _split_args(m.group(2)) if m else []
    if m and templates and install is not None:
        header = templates.get(m.group(1))
        if header:
            args = ensure_occt_args([], install.include_dir)
            rd = find_resource_dir()
            if rd:
                args.append(f"-resource-dir={rd}")
            includes = [header] + _collect_includes(out, header_map=header_map)
            try:
                out = _expand_spec_args(header, m.group(1), out, includes,
                                        args, install.include_dir)
            except Exception:  # noqa: BLE001
                out = _split_args(m.group(2))
    _expanded_spec_memo[key] = out
    return out


def _transitive_extend(classes: list[object], known: set[str], install: Path,
                       modules: list[ModuleDecl],
                       header_map: dict[str, str],
                       noncopyable: set[str],
                       templates: dict[str, str]) -> list[object]:
    """Grow a synthesized-spec list to a transitive fixpoint.

    Specializations reachable only through *other synthesized* classes (e.g.
    ``HSequence<T>::ChangeArray1`` naming ``NCollection_Array1<T>``) never
    appear in the scanned IR, so a single collection pass misses them.  Re-scan
    the synthesized classes' signatures until no new nameable spec appears.
    """
    known_names = (set(header_map) | set(_BUILTINS) | set(PRIMITIVE_MAP))
    cached: dict[str, str] = {}
    if _TEMPLATE_REGISTRY_CACHE_PATH.exists():
        try:
            cached = json.loads(
                _TEMPLATE_REGISTRY_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
    while True:
        module = ModuleDecl(name="NCollection", classes=classes)
        missing: set[str] = set()
        param_specs: set[str] = set()
        collected = _collect_template_specs([module], templates,
                                            missing_out=missing,
                                            param_specs_out=param_specs)
        if missing:
            # A root only reachable through a synthesized class (e.g. a default
            # template argument such as NCollection_DefaultHasher<K> that real
            # OCCT signatures never spell out) is absent from the registry.
            # Resolve its header and retry the collection before filtering.
            grew = False
            for root in sorted(missing):
                if root in templates:
                    continue
                h = _template_header(root, header_map, install, cached)
                if h:
                    templates[root] = h
                    grew = True
            if grew:
                continue
        fresh = {k for k in collected
                 if k not in known
                 and not _arg_prefix_covered(k, known, templates, install,
                                             header_map)
                 and not _has_placeholder_args(k, known_names, templates,
                                               install)}
        dead = {k for k in fresh if k not in param_specs}
        if dead:
            # Return-type-only specializations reachable through synthesized
            # classes (container iterator adapters such as NCollection_StlIterator)
            # are never named as inputs by a bindable method: their returning
            # begin/end methods are skipped as iterator protocol.  Wrapping them
            # would produce empty shells, so they are dropped rather than
            # synthesized.
            for k in sorted(dead):
                print(f"synth          : SKIP {k} (return-only "
                      "iterator/helper type)", file=__import__("sys").stderr)
                known.add(k)
            fresh -= dead
        if not fresh:
            return classes
        specs = {k: v for k, v in collected.items() if k in fresh}
        undeclarable = _undeclarable_specs(specs, install, modules) if specs else set()
        abstract = _unwrappable_specs(specs, install, modules) if specs else set()
        for key in sorted(fresh):
            if key in undeclarable:
                print(f"synth          : SKIP {key} (template arg not publicly "
                      "nameable)", file=__import__("sys").stderr)
                known.add(key)
                continue
            if key in abstract:
                print(f"synth          : SKIP {key} (unwrappable: abstract "
                      "or no default ctor)", file=__import__("sys").stderr)
                known.add(key)
                continue
            header, args = specs[key]
            if _specialization_broken(key, noncopyable):
                print(f"synth          : SKIP {key} (non-copyable template arg)",
                      file=__import__("sys").stderr)
                known.add(key)
                continue
            try:
                cls = synth_template_spec(header, key.split("<", 1)[0], args,
                                          install=install, header_map=header_map)
                cls.name = key
                classes.append(cls)
            except Exception as e:  # noqa: BLE001
                print(f"synth          : SKIP {key}: {e}",
                      file=__import__("sys").stderr)
            known.add(key)
    return classes


def upgrade_transitive(classes: list[object], modules: list[ModuleDecl],
                       install: Path) -> list[object]:
    """Add transitively-reachable specializations to a cached spec list.

    Idempotent: after a cache has been upgraded once, a later load finds no
    new specs and returns the classes unchanged (cheap pure-Python scan).
    """
    templates = _template_registry(modules, install)
    known = {getattr(c, "name", "") for c in classes}
    known |= set(_collect_template_specs(modules, templates))
    return _transitive_extend(classes, known, install, modules,
                              _build_header_map(modules),
                              _noncopyable_classes(modules), templates)


def _is_nested_spec(base_name: str) -> bool:
    """True when ``base_name`` names a template nested in another template
    (e.g. ``NCollection_DynamicArray<T>::DynamicIterator<U>``).

    Only top-level ``Root<args>`` specializations can be synthesized: nested
    ones have no value-wrappable identity of their own, and looking the class
    template up by its qualified name matches the *outer* template instead.
    A ``::`` *inside* a template argument (``opencascade::handle<...>``) is at
    bracket depth > 0 and must not be treated as nesting.
    """
    depth = 0
    for i, ch in enumerate(base_name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0 and ch == ":" and i + 1 < len(base_name) \
                and base_name[i + 1] == ":":
            return True
    return False


def _collect_template_specs(modules: list[ModuleDecl],
                            templates: dict[str, str],
                            missing_out: set[str] | None = None,
                            param_specs_out: set[str] | None = None,
                            ) -> dict[str, tuple[str, list[str]]]:
    """Distinct class-template specializations used in the scanned signatures.

    Returns ``{ "NCollection_Array2<gp_Pnt>": (header, [args...]) }`` for every
    specialization of any discovered class template (see ``_template_registry``)
    that appears in a method, constructor, static method, operator or field of
    the scanned IR.  Roots with no known header are collected in ``missing_out``
    (if given) so a caller can grow the registry and retry.  Specs seen as a
    top-level *parameter* type are added to ``param_specs_out`` (if given); a
    specialization is worth synthesizing when it is an input the bindable API
    names, while a pure return type of an iterator-protocol method is dead.
    """
    specs: dict[str, tuple[str, list[str]]] = {}

    def handle(t) -> None:
        if t is None:
            return
        b = getattr(t, "base_name", "")
        m = _SPEC_RE.match(b)
        if not m or _is_nested_spec(b):
            return
        tname, argstr = m.group(1), m.group(2)
        header = templates.get(tname)
        if header is None:
            if missing_out is not None:
                missing_out.add(tname)
            return
        args = _split_args(argstr)
        if not args:
            return
        key = f"{tname}<{', '.join(args)}>"
        specs.setdefault(key, (header, args))

    def handle_param(t) -> None:
        if param_specs_out is None:
            return
        b = getattr(t, "base_name", "")
        m = _SPEC_RE.match(b)
        if not m or _is_nested_spec(b):
            return
        args = _split_args(m.group(2))
        if args:
            param_specs_out.add(f"{m.group(1)}<{', '.join(args)}>")

    for module in modules:
        for cls in module.classes:
            for m in cls.all_methods:
                handle(m.return_type)
                for p in m.parameters:
                    handle(p.type)
                    handle_param(p.type)
            for f in cls.fields:
                if not f.is_public:
                    # A specialization reachable only through a private field
                    # can never be named by a wrapper; synthesizing it would
                    # only surface unusable methods (and, for private nested
                    # types, probe compile failures).
                    continue
                handle(f.type)
    return specs


def _collect_demo_refs(project_root: Path) -> set[str]:
    """Wrapper names referenced by the demo project's GDScript sources."""
    out: set[str] = set()
    demo = Path(project_root) / "demo"
    if not demo.is_dir():
        return out
    for p in demo.rglob("*.gd*"):
        if p.suffix not in (".gd", ".gd.disabled"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        out.update(_ORT_RE.findall(text))
    return out


def synthesize_used(project_root: Path,
                    modules: list[ModuleDecl]) -> list[object]:
    """Synthesize the class-template specializations the demo code uses.

    Usage-driven: only specializations whose wrapper name appears in the demo
    GDScript sources (and that occur in the scanned OCCT IR) are synthesized,
    keeping the generated surface small and deterministic.
    """
    demo_refs = _collect_demo_refs(project_root)
    if not demo_refs:
        return []
    install = find_occt_install(_default_project_root())
    templates = _template_registry(modules, install)
    specs = _collect_template_specs(modules, templates)
    if not specs:
        return []
    out: list[object] = []
    for key, (header, args) in sorted(specs.items()):
        tname = key.split("<", 1)[0]
        if occt_name_to_wrapper(key, "NCollection") not in demo_refs:
            continue
        try:
            cls = synth_template_spec(header, tname, args, install=install,
                                      header_map=_build_header_map(modules))
            cls.name = key  # exact spelling used in signatures
            out.append(cls)
        except Exception as e:  # noqa: BLE001
            print(f"synthesize    : SKIP {key}: {e}", file=__import__("sys").stderr)
    return out


def _undeclarable_specs(specs: dict[str, tuple[str, list[str]]],
                        install: Path,
                        modules: list[ModuleDecl] | None = None) -> set[str]:
    """Spec keys whose template arguments are not publicly nameable.

    A free-namespace ``using OrtUndeclN = Spec<args>;`` fails to compile when
    any argument names a private/protected nested type (e.g.
    ``NCollection_Array1<Aspect_VKeySet::KeyState>`` where ``KeyState`` is a
    private nested struct).  No wrapper can ever name such a type, so the spec
    must not be synthesized at all; otherwise every one of its methods trips
    the symbol-audit probe.  Member-level breakage (e.g. an ambiguous ``abs``
    in ``NCollection_Vec3<unsigned long>::cwiseAbs``) is *not* caught here --
    that spec is declarable and is handled by the audit's ill-formed-method
    skipping instead.
    """
    args = ensure_occt_args([], install.include_dir)
    rd = find_resource_dir()
    if rd:
        args.append(f"-resource-dir={rd}")
    header_map = _build_header_map(modules) if modules else None
    includes: set[str] = set()
    for key, (header, _) in specs.items():
        includes.add(header)
    for key, (_, as_) in specs.items():
        includes.update(_collect_includes(as_, header_map=header_map))
    # Some specializations carry args whose derived "header" does not exist
    # (e.g. array bounds); a missing #include would abort the whole batch TU,
    # so only pull in headers that are actually present.
    include_dir = install.include_dir
    includes = {i for i in includes if (include_dir / i).exists()}
    lines: list[str] = [f"#include <{i}>" for i in sorted(includes)]
    lines.append("")
    lines.append("namespace ort_undecl {")
    key_lines: dict[str, int] = {}
    for i, key in enumerate(sorted(specs)):
        lines.append(f"using OrtUndecl{i} = {key};")
        key_lines[key] = len(lines)
    lines.append("}")
    src = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        index = C.Index.create()
        tu = index.parse(tmp, args=args + ["-x", "c++", "-I",
                                           str(install.include_dir)])
        out: set[str] = set()
        for d in tu.diagnostics:
            if d.severity >= C.Diagnostic.Error:
                for key, line_no in key_lines.items():
                    if d.location.line == line_no:
                        out.add(key)
                        break
        return out
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


_abstract_memo: tuple[frozenset[str], set[str]] | None = None


def _unwrappable_specs(specs: dict[str, tuple[str, list[str]]], install: Path,
                       modules: list[ModuleDecl] | None = None) -> set[str]:
    """Spec keys that can never compile as a wrapper.

    Only abstract specializations are unwrappable: codegen's abstract-
    constructor skipping relies on ``cls.is_abstract``, which libclang cannot
    evaluate for class templates (they must be detected here, ahead of
    codegen).  Specializations without a public default constructor are *not*
    unwrappable -- codegen falls back to unique_ptr storage with factory
    constructors (the same path used for scanned value classes like
    math_VectorBase<double>).  Detect in one batch TU via ``std::is_abstract``
    so a (failed) static_assert surfaces as a per-line diagnostic.
    """
    global _abstract_memo
    key_set = frozenset(specs)
    if _abstract_memo is not None and _abstract_memo[0] == key_set:
        return _abstract_memo[1]
    args = ensure_occt_args([], install.include_dir)
    rd = find_resource_dir()
    if rd:
        args.append(f"-resource-dir={rd}")
    header_map = _build_header_map(modules) if modules else None
    includes: set[str] = {"type_traits", "Standard_Transient.hxx"}
    for key, (header, _) in specs.items():
        includes.add(header)
    for key, (_, as_) in specs.items():
        includes.update(_collect_includes(as_, header_map=header_map))
    include_dir = install.include_dir
    includes = {i for i in includes if (include_dir / i).exists()}
    lines: list[str] = [f"#include <{i}>" for i in sorted(includes)]
    lines.append("")
    lines.append("namespace ort_abstract {")
    key_lines: dict[str, int] = {}
    for i, key in enumerate(sorted(specs)):
        lines.append(f"static_assert(!std::is_abstract<{key}>::value, "
                     f"\"abstract:{i}\");")
        key_lines[key] = len(lines)
    lines.append("}")
    src = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        index = C.Index.create()
        tu = index.parse(tmp, args=args + ["-x", "c++", "-I",
                                           str(include_dir)])
        out: set[str] = set()
        for d in tu.diagnostics:
            if d.severity >= C.Diagnostic.Error:
                for key, line_no in key_lines.items():
                    if d.location.line == line_no:
                        out.add(key)
                        break
        _abstract_memo = (key_set, out)
        return out
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def synthesize_all(modules: list[ModuleDecl]) -> list[object]:
    """API-driven: synthesize *every* specialization of every discovered class
    template that appears in any scanned signature, not just the ones the demo
    references.  A method is bindable once its container argument types have
    concrete wrapper classes; instantiations become ordinary wrapper classes.

    Specializations that cannot be instantiated cleanly (nested templates,
    args naming skipped classes, ...) are reported in `last_failures` and left
    for the typemap/synthesis generalizations that target them.
    """
    install = find_occt_install(_default_project_root())
    templates = _template_registry(modules, install)
    specs = _collect_template_specs(modules, templates)
    out: list[object] = []
    failures: list[str] = []
    if not specs:
        synthesize_all.last_failures = failures
        return out
    header_map = _build_header_map(modules)
    undeclarable = _undeclarable_specs(specs, install, modules)
    if undeclarable:
        print(f"synth          : dropping {len(undeclarable)}"
              f" specialization(s) with private/protected template args",
              file=__import__("sys").stderr)
    abstract = _unwrappable_specs(specs, install, modules)
    if abstract:
        print(f"synth          : dropping {len(abstract)} unwrappable"
              f" specialization(s) (abstract / no default ctor)",
              file=__import__("sys").stderr)
    noncopyable = _noncopyable_classes(modules)
    for i, (key, (header, args)) in enumerate(sorted(specs.items())):
        tname = key.split("<", 1)[0]
        print(f"synth[{i + 1}/{len(specs)}]    : {key}", flush=True)
        if key in undeclarable:
            print(f"synth          : SKIP {key} (template arg not publicly "
                  "nameable)", flush=True)
            continue
        if key in abstract:
            print(f"synth          : SKIP {key} (unwrappable: abstract or no default ctor)", flush=True)
            continue
        if _specialization_broken(key, noncopyable):
            print(f"synth          : SKIP {key} (non-copyable template arg)",
                  flush=True)
            continue
        try:
            cls = synth_template_spec(header, tname, args, install=install,
                                      header_map=header_map)
            cls.name = key  # exact spelling used in signatures
            out.append(cls)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{key}: {e}")
            print(f"synth          : SKIP {key}: {e}", flush=True)
    out = _transitive_extend(out, set(specs), install, modules, header_map,
                             noncopyable, templates)
    synthesize_all.last_failures = failures
    return out


synthesize_all.last_failures: list[str] = []



def _default_project_root() -> Path:
    """Locate the repo root (with its vcpkg install)."""
    if (_PROJECT_ROOT / "vcpkg" / "installed").exists():
        return _PROJECT_ROOT
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "vcpkg" / "installed").exists():
            return p
    return _PROJECT_ROOT


def _find_class_template(root: C.Cursor, name: str) -> C.Cursor | None:
    """Locate the class template named ``name``, preferring the definition.

    Some headers (e.g. BVH_Builder.hxx) carry a forward-declared class
    template whose cursor has no members; extracting from it yields an empty
    class and breaks storage classification, so the definition must win.
    """
    candidates: list[C.Cursor] = []

    def rec(c: C.Cursor) -> None:
        if c.kind == CursorKind.CLASS_TEMPLATE and c.spelling == name:
            candidates.append(c)
        for ch in c.get_children():
            rec(ch)

    rec(root)
    if not candidates:
        return None
    for c in candidates:
        if c.is_definition():
            return c
    return candidates[0]


_TEMPLATE_PARAM_KINDS = (CursorKind.TEMPLATE_TYPE_PARAMETER,
                         CursorKind.TEMPLATE_NON_TYPE_PARAMETER,
                         CursorKind.TEMPLATE_TEMPLATE_PARAMETER,
                         CursorKind.PARM_DECL)

_MEMBER_KINDS = (CursorKind.CXX_BASE_SPECIFIER, CursorKind.CONSTRUCTOR,
                 CursorKind.CXX_METHOD, CursorKind.FIELD_DECL,
                 CursorKind.ENUM_DECL, CursorKind.VAR_DECL)


def _template_is_degenerate(cursor: C.Cursor) -> bool:
    """True when the primary template defines no members/bases at all.

    Some OCCT templates (BVH_Tree) declare a primary definition that is a
    placeholder ("// Invalid type"); the real classes are partial
    specializations in other headers.  Synthesis must switch to those.
    """
    return not any(c.kind in _MEMBER_KINDS for c in cursor.get_children())


def _partial_specs_in(root: C.Cursor, name: str) -> list[C.Cursor]:
    out: list[C.Cursor] = []

    def rec(c: C.Cursor) -> None:
        if c.kind == CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION \
                and c.spelling == name:
            out.append(c)
        for ch in c.get_children():
            rec(ch)

    rec(root)
    return out


def _partial_matches(ps: C.Cursor, args_list: list[str]) -> bool:
    """True when the partial specialization's template-id matches the expanded
    args, e.g. ``BVH_Tree<T, N, BVH_BinaryTree>`` vs ``[double, 3,
    BVH_BinaryTree]``: each partial-spec argument is either one of its own
    template parameters (bound to the requested arg) or a fixed spelling that
    must equal it."""
    params = {c.spelling for c in ps.get_children()
              if c.kind in _TEMPLATE_PARAM_KINDS}
    m = re.match(r"^[^<]*<(.*)>$", ps.type.spelling, re.S)
    if not m:
        return False
    ps_args = _split_args(m.group(1))
    if len(ps_args) != len(args_list):
        return False
    bindings: dict[str, str] = {}
    for psa, actual in zip(ps_args, args_list):
        psa = psa.strip()
        actual = actual.strip()
        if psa in params:
            if psa in bindings and bindings[psa] != actual:
                return False
            bindings[psa] = actual
        elif psa != actual:
            return False
    return True


_PARTIAL_SPEC_TEXT_RE = re.compile(r"\b(?:class|struct)\s+([A-Za-z_]\w*)\s*<",
                                   re.M)


def _find_matching_partial_spec(template_name: str, args_list: list[str],
                                tu: object, args: list[str],
                                include_dir: Path,
                                header_name: str,
                                ) -> tuple[str, object, C.Cursor, list[str]] | None:
    """Locate a partial specialization of ``template_name`` matching the
    expanded args, first in the current TU then by a content scan of the other
    install headers (the primary's own header cannot reach specializations that
    live in siblings).  Returns ``(header, tu, cursor, extra_includes)``."""
    def search_tu(root: C.Cursor) -> C.Cursor | None:
        for ps in _partial_specs_in(root, template_name):
            if _partial_matches(ps, args_list):
                return ps
        return None

    ps = search_tu(tu.cursor)
    if ps is not None:
        return header_name, tu, ps, []
    for f in include_dir.glob("*.hxx"):
        if f.name == header_name:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not _PARTIAL_SPEC_TEXT_RE.search(txt):
            continue
        try:
            ptu, pextra = _parse_with_undeclared_retry(f, args, include_dir)
        except Exception:  # noqa: BLE001
            continue
        ps = search_tu(ptu.cursor)
        if ps is not None:
            return f.name, ptu, ps, pextra
    return None


def _split_args(argstr: str) -> list[str]:
    args, depth, cur = [], 0, ""
    for ch in argstr:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur)
    return [a.strip() for a in args]


_BUILTINS = {"void", "bool", "char", "short", "int", "long", "float", "double",
             "unsigned", "signed", "size_t", "char16_t", "char32_t", "wchar_t",
             "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
             "unsigned char", "signed char", "unsigned short", "signed short",
             "unsigned int", "signed int", "unsigned long", "signed long",
             "unsigned long long", "signed long long", "long double",
             "long long", "unsigned long long"}


def _build_header_map(modules: list[ModuleDecl]) -> dict[str, str]:
    """OCCT class name -> header basename, from the scanned IR.

    OCCT usually declares one class per header, so ``Graphic3d_Attribute.hxx``
    is the natural include for ``Graphic3d_Attribute``.  Some classes break the
    convention and live in another header (``Graphic3d_Attribute`` is declared
    in ``Graphic3d_Buffer.hxx``); the scanner records the real header.  Passing
    this map to ``_collect_includes`` lets a synthesized spec include the
    declaring header instead of a name-derived one that does not exist.
    """
    out: dict[str, str] = {}
    for module in modules:
        for cls in module.classes:
            hf = getattr(cls, "header_file", "") or ""
            if hf:
                out[cls.name] = Path(hf).name
    return out


def _collect_includes(args_list: list[str],
                      include_dir: Path | None = None,
                      header_map: dict[str, str] | None = None) -> list[str]:
    """OCCT convention: header file name == (outermost) class name.

    A nested class A::B lives in the enclosing class's header A.hxx, the
    std/opencascade namespaces contribute no header of their own (handles and
    std::* types resolve from their template arguments), and a template's
    arguments are recursed into so ``handle<Geom_Curve>`` yields
    ``Geom_Curve.hxx``.  ``header_map`` (see ``_build_header_map``) overrides
    the name-derived header when the class lives in a different header.  When
    ``include_dir`` is given only headers that exist there are returned, so a
    speculative include can never break the probe TU.
    """
    out: list[str] = []

    def rec(arg: str) -> None:
        head = re.sub(r"^(?:const|volatile)\s+", "", arg.strip())
        head = re.sub(r"(?:[*&])$", "", head)
        head = head.split("<")[0].strip()
        parts = [p for p in head.split("::") if p]
        if parts and parts[0] not in ("std", "opencascade") \
                and parts[0] not in _BUILTINS and not parts[0].endswith("_t") \
                and re.match(r"^[A-Za-z_]\w*$", parts[0]):
            out.append(f"{parts[0]}.hxx")
        m = re.match(r"^[^<]*<(.*)>$", arg.strip(), re.S)
        if m:
            for inner in _split_args(m.group(1)):
                rec(inner)

    for a in args_list:
        rec(a)
    if header_map:
        out = [header_map.get(i[:-len(".hxx")] if i.endswith(".hxx") else i, i)
               for i in out]
    if include_dir is not None:
        existing = {p.name for p in include_dir.iterdir()} if include_dir.is_dir() else set()
        out = [i for i in out if i in existing]
    seen: set[str] = set()
    return [i for i in out if not (i in seen or seen.add(i))]


def _substitute(spelling: str, subst: dict[str, str]) -> str:
    out = spelling
    for name, repl in subst.items():
        if not name:
            continue
        out = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])",
                     repl, out)
    return out


_KNOWN_BASIC = {
    "void", "bool", "char", "short", "int", "long", "float", "double",
    "unsigned", "signed", "size_t", "wchar_t", "char16_t", "char32_t",
    "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t",
    "int64_t", "uint64_t", "intptr_t", "uintptr_t",
}


def _finalize_type(t: object) -> object:
    """Make a probe-resolved type usable for self-contained wrapper codegen.

    Nested typedef names must be canonicalized even when they appear inside
    qualifiers (``const Array1Type &`` -> ``const NCollection_Array1<double> &``)
    so codegen stays self-contained.  Basic/Standard_ spellings are kept as-is
    to preserve typemap conventions.
    """
    if t is None:
        return t
    sp = getattr(t, "spelling", "").strip()
    canon = getattr(t, "canonical_spelling", "").strip()
    if canon and canon != sp:
        if sp in _KNOWN_BASIC or sp.startswith("Standard_"):
            return t
        return replace(t, spelling=canon)
    return t


def _resolve_types(spellings: list[str], template_spec: str,
                   includes: list[str], args: list[str],
                   include_dir: Path) -> dict[str, object]:
    """Resolve each spelling to an OCCTType via a scoped probe TU."""
    aliases = "\n".join(f"  using AW_T{i} = {s};"
                        for i, s in enumerate(spellings))
    incs = "\n".join(f"#include <{i}>" for i in includes)
    src = (f"#include <{includes[0]}>\n{incs}\n"
           f"template class {template_spec};\n"
           f"namespace ort_synth {{\n"
           f"struct AW_Scope : public {template_spec} {{\n{aliases}\n}};\n}}\n")
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        index = C.Index.create()
        tu = index.parse(tmp, args=args + ["-x", "c++", "-I", str(include_dir)])
        out: dict[str, object] = {}
        for ns in tu.cursor.get_children():
            if ns.kind == CursorKind.NAMESPACE and ns.spelling == "ort_synth":
                for s in ns.get_children():
                    if s.kind == CursorKind.STRUCT_DECL:
                        for ta in s.get_children():
                            if ta.kind == CursorKind.TYPE_ALIAS_DECL:
                                idx = int(ta.spelling[len("AW_T"):])
                                if 0 <= idx < len(spellings):
                                    try:
                                        out[spellings[idx]] = make_type(ta.underlying_typedef_type)
                                    except Exception:
                                        pass
        return out
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _expand_spec_args(header_name: str, template_name: str,
                      args_list: list[str], includes: list[str],
                      args: list[str], include_dir: Path) -> list[str]:
    """Fill in default template arguments from the specialization itself."""
    incs = "\n".join(f"#include <{i}>" for i in includes)
    spec = f"{template_name}<{', '.join(args_list)}>"
    src = f"#include <{header_name}>\n{incs}\ntemplate class {spec};\n"
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        index = C.Index.create()
        tu = index.parse(tmp, args=args + ["-x", "c++", "-I", str(include_dir)])
        for d in tu.diagnostics:
            if d.severity >= C.Diagnostic.Error:
                return args_list

        def spec_args(cursor: C.Cursor) -> list[str] | None:
            """The template argument spellings of a specialization, or None."""
            try:
                n = cursor.get_num_template_arguments()
            except Exception:
                return None
            out = []
            for i in range(n):
                try:
                    kind = cursor.get_template_argument_kind(i)
                except Exception:
                    kind = None
                if kind == C.TemplateArgumentKind.INTEGRAL:
                    # A non-type argument (e.g. the `3` in BVH_Tree<double, 3>)
                    # has no type spelling; substitute the integral value.
                    try:
                        out.append(str(cursor.get_template_argument_value(i)))
                    except Exception:
                        out.append(cursor.get_template_argument_type(i).spelling)
                else:
                    out.append(cursor.get_template_argument_type(i).spelling)
            return out if out else None

        def find_spec(cursor: C.Cursor):
            # Match the specialization whose arguments equal `args_list`: the
            # TU also contains *other* instantiations of the same template from
            # the pre-included headers (e.g. NCollection_DefaultHasher<bool>
            # via TCollection_AsciiString.hxx), and those must not be mistaken
            # for the spec being synthesized.
            for c in cursor.get_children():
                if c.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) \
                        and c.spelling == template_name and c.is_definition():
                    a = spec_args(c)
                    if a and a == args_list:
                        return c
                r = find_spec(c)
                if r:
                    return r
            return None

        sp = find_spec(tu.cursor)
        if sp is None:
            return args_list
        out = spec_args(sp)
        if out is None:
            return args_list
        return out if len(out) >= len(args_list) else args_list
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


_UNDECL_RE = re.compile(r"undeclared identifier ['\"]([A-Za-z_]\w*)['\"]")

_undecl_scan_memo: dict[frozenset[str], list[str]] = {}


def _headers_declaring(identifiers: frozenset[str], include_dir: Path) -> list[str]:
    """Headers whose content mentions any of the identifiers (word-bounded).

    A header may reference a name defined in another header it does not
    include (relying on transitive include order); pre-including every header
    that mentions the name makes such templates parse standalone.  Including
    mere users of the identifier is harmless -- all OCCT headers compile
    together.
    """
    if identifiers in _undecl_scan_memo:
        return _undecl_scan_memo[identifiers]
    pats = [re.compile(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])")
            for i in sorted(identifiers)]
    hits: set[str] = set()
    for f in include_dir.glob("*.hxx"):
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        for p in pats:
            if p.search(txt):
                hits.add(f.name)
                break
    out = sorted(hits)
    _undecl_scan_memo[identifiers] = out
    return out


def _parse_with_undeclared_retry(header: Path, args: list[str],
                                 include_dir: Path
                                 ) -> tuple[object, list[str]]:
    """Parse ``header``, pre-including headers that resolve any undeclared
    identifier reported by the first parse (some OCCT templates rely on
    transitive includes).  Raises when no new include can fix the error.
    Returns ``(tu, extra_headers)``."""
    extra: list[str] = []
    for _ in range(2):
        try:
            return parse_header(header, args, pre_headers=extra), extra
        except ParseError as e:
            ids = frozenset(_UNDECL_RE.findall(str(e)))
            new = ([h for h in _headers_declaring(ids, include_dir)
                    if h != header.name and h not in extra] if ids else [])
            if not new:
                raise
            extra.extend(new)
    raise AssertionError("unreachable")


def synth_template_spec(header_name: str, template_name: str,
                        args_list: list[str],
                        install: Path | None = None,
                        header_map: dict[str, str] | None = None) -> object:
    """Return a ClassDecl for the specialization ``template_name<args_list>``."""
    if install is None:
        install = find_occt_install(_default_project_root())
    args = ensure_occt_args([], install.include_dir)
    rd = find_resource_dir()
    if rd:
        args.append(f"-resource-dir={rd}")

    header = install.include_dir / header_name
    tu, extra = _parse_with_undeclared_retry(header, args, install.include_dir)
    ct = _find_class_template(tu.cursor, template_name)
    if ct is None:
        raise ValueError(f"no CLASS_TEMPLATE {template_name} in {header_name}")

    params: list[tuple[str, bool]] = []
    for c in ct.get_children():
        if c.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
            params.append((c.spelling, True))
        elif c.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
            params.append((c.spelling, False))
        elif c.kind == CursorKind.PARM_DECL:
            params.append((c.spelling, False))

    subst: dict[str, str] = {}
    for idx, (pname, is_type) in enumerate(params):
        if idx < len(args_list):
            subst[pname] = args_list[idx]
            subst[f"type-parameter-0-{idx}"] = args_list[idx]

    includes = (extra + [header_name]
                + _collect_includes(args_list, header_map=header_map))
    full_args = _expand_spec_args(header_name, template_name, args_list,
                                  includes, args, install.include_dir)
    if full_args != args_list:
        args_list = full_args
        subst = {}
        for idx, (pname, is_type) in enumerate(params):
            if idx < len(args_list):
                subst[pname] = args_list[idx]
                subst[f"type-parameter-0-{idx}"] = args_list[idx]
    template_spec = f"{template_name}<{', '.join(args_list)}>"

    if _template_is_degenerate(ct):
        # The primary is a placeholder ("Invalid type"); the real class is a
        # partial specialization whose template-id matches the (default-filled)
        # args (BVH_Tree<T, N, BVH_BinaryTree> for BVH_Tree<double, 3>).
        match = _find_matching_partial_spec(
            template_name, args_list, tu, args, install.include_dir,
            header_name)
        if match is not None:
            header_name, tu, ct, extra = match
            includes = (extra + [header_name]
                        + _collect_includes(args_list, header_map=header_map))

    cls = _extract_class(ct, header_name, tu.cursor)
    # Full specialization spelling (e.g. "NCollection_Array2<gp_Pnt>") is the
    # class name: it is exactly the spelling the scanner reports for template
    # arguments in other classes' signatures, so build_context registers the
    # specialization -> wrapper mapping and `native` storage emits the full
    # type in the generated header.  Wrapper naming derives Ort-prefixed names
    # from it via occt_name_to_wrapper.
    cls.name = template_spec
    cls.base_classes = [_substitute(b, subst) for b in cls.base_classes]
    # The generated header spells `_native` as the full specialization, so the
    # template argument headers must be available even though no individual
    # method signature mentions them (e.g. TopTools_ShapeMapHasher.hxx for
    # NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>).
    cls.extra_occt_includes = (extra
                               + [i for i in _collect_includes(args_list, header_map=header_map)
                                  if i != header_name])

    to_resolve: dict[str, str] = {}  # substituted spelling -> substituted spelling

    def queue(t: object) -> None:
        s = _substitute(getattr(t, "spelling", ""), subst)
        to_resolve.setdefault(s, s)

    for b in cls.base_classes:
        queue(type("T", (), {"spelling": b})())
    for m in cls.constructors + cls.methods + cls.operators + cls.static_methods:
        if m.return_type is not None:
            queue(m.return_type)
        for p in m.parameters:
            queue(p.type)
    for f in cls.fields:
        queue(f.type)

    resolved = _resolve_types(list(to_resolve), template_spec, includes,
                              args, install.include_dir)

    # Member signatures that self-reference this class are canonicalized by the
    # probe to the default-expanded template-id (``NCollection_DataMap<A, B,
    # NCollection_DefaultHasher<A>>`` resolving from ``NCollection_DataMap<A,
    # B>``).  Rebind those back to the specialization's own spelling so
    # copy/assign take a Ref of this very wrapper rather than a second wrapper
    # of the same native type.
    for k in list(resolved):
        base = k
        for pre in ("const ", "volatile "):
            if base.startswith(pre):
                base = base[len(pre):]
        for suf in ("&&", "&", "*"):
            if base.endswith(suf):
                base = base[: -len(suf)].rstrip()
        if base == template_spec:
            resolved[k] = replace(resolved[k], spelling=k, canonical_spelling=k)

    def rebind(t: object) -> object:
        s = _substitute(getattr(t, "spelling", ""), subst)
        nt = resolved.get(s)
        return _finalize_type(nt) if nt is not None else t

    for m in cls.constructors + cls.methods + cls.operators + cls.static_methods:
        if m.return_type is not None:
            m.return_type = rebind(m.return_type)
        for p in m.parameters:
            p.type = rebind(p.type)
            dflt = getattr(p, "default_value", None)
            if dflt:
                p.default_value = _substitute(dflt, subst)
    for f in cls.fields:
        f.type = rebind(f.type)
    for i, b in enumerate(cls.base_classes):
        nt = rebind(type("T", (), {"spelling": b})())
        if isinstance(nt, object) and hasattr(nt, "spelling"):
            cls.base_classes[i] = nt.spelling

    cls.is_template = False
    return cls


# Fallback specializations used by `autogen synth-check` when no scanned IR is
# available (nothing in the toolchain is specific to any one template family;
# the checks just exercise the synthesis tiers: simple, handle-args, nested
# templates, defaults expansion, macro-free HArray1).
REPRESENTATIVE_SPECS: list[tuple[str, str, list[str]]] = [
    ("NCollection_Vec3.hxx", "NCollection_Vec3", ["float"]),
    ("NCollection_Array2.hxx", "NCollection_Array2", ["gp_Pnt"]),
    ("NCollection_Array1.hxx", "NCollection_Array1", ["gp_Pnt"]),
    ("NCollection_HArray1.hxx", "NCollection_HArray1", ["double"]),
    ("NCollection_Sequence.hxx", "NCollection_Sequence", ["gp_Pnt"]),
    ("NCollection_DataMap.hxx", "NCollection_DataMap",
     ["TCollection_AsciiString", "TCollection_AsciiString"]),
]


def synth_check(verbose: bool = True) -> int:
    """Synthesize the discovered specializations; return 0 when all succeed.

    When the scanned IR is present, one specialization of every discovered
    class template is exercised (fully generic); otherwise a representative
    fallback list is used.
    """
    import sys as _sys
    ir_dir = Path(__file__).resolve().parents[1] / "out" / "ir"
    specs: list[tuple[str, str, list[str]]] = []
    if ir_dir.is_dir():
        from .ir import load_module
        modules = [load_module(p) for p in sorted(ir_dir.glob("*.json"))]
        install = find_occt_install(_default_project_root())
        templates = _template_registry(modules, install)
        collected = _collect_template_specs(modules, templates)
        for key in sorted(collected):
            header, args = collected[key]
            specs.append((header, key.split("<", 1)[0], args))
        if not specs:
            print("synth-check    : no scanned IR specs; using fallback list",
                  file=_sys.stderr)
    if not specs:
        specs = list(REPRESENTATIVE_SPECS)
    failures = 0
    for header, tname, targs in specs:
        label = f"{tname}<{', '.join(targs)}>"
        try:
            cls = synth_template_spec(header, tname, targs)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {label}: {e}")
            failures += 1
            continue
        print(f"{label}: methods={len(cls.methods)} ctors={len(cls.constructors)} "
              f"ops={len(cls.operators)} statics={len(cls.static_methods)} "
              f"fields={len(cls.fields)} bases={cls.base_classes}")
        if verbose:
            for m in (cls.methods + cls.constructors + cls.operators)[:6]:
                ps = ", ".join(p.type.spelling for p in m.parameters)
                ret = m.return_type.spelling if m.return_type else "void"
                print(f"    {m.name}({ps}) -> {ret}")
    return failures, len(specs)
