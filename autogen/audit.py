"""Link-existence audit for generated wrappers.

Some OCCT methods are declared in headers but never defined in the static
libraries (e.g. ``OSD_Path::LocateExecFile`` -- only the free function
``LocateExecFile(OSD_Path&)`` is exported).  A wrapper calling such a method is
a link error that would only surface at the very end of a slow vcpkg rebuild.

The audit catches those at generation time instead:

  * Pass 1 (``generate-all --probe-out``) emits a probe TU with one discarded
    address-of expression per generated wrapper method.  An explicit member /
    function pointer cast disambiguates overloads, and namespace-scope
    variables force g++ to emit the undefined reference.
  * ``audit`` compiles the probe, lists undefined symbols via ``nm -u -C`` and
    keeps the ones absent from the OCCT libraries' defined symbol set (skipping
    non-OCCT noise such as ``std::`` or ``_GLOBAL_OFFSET_TABLE_``).
  * Pass 2 (``generate-all --missing``) regenerates, skipping every method whose
    symbol is in the missing file.
"""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import heapq
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from .model import ClassKind, MethodKind, OCCTType
from .occt import OCCTInstall, include_closure
from . import typemap as tm

# Source spelling of an operator for pointer casts / symbol names.
_OPERATOR_SPELLING = {
    "unary_minus": "-", "unary_plus": "+", "*deref": "*", "call": "()",
}

# Header pairs whose relative order matters but is invisible to ``_hygiene_order``
# (the referenced name is a template / namespace, not a registered class).  Each
# entry forces ``header`` to be included *after* every ``must_follow`` header.
_HEADER_PRECEDENCE: dict[str, frozenset[str]] = {
    # Circular include: NCollection_PackedMap.hxx includes NCollection_PackedMapAlgo.hxx
    # at line ~870, and NCollection_PackedMapAlgo.hxx re-includes NCollection_PackedMap.hxx.
    # If Algo is included first, PackedMap's member-template bodies reference the
    # ``NCollection_PackedMapAlgo`` namespace that the guarded inner include skipped.
    "NCollection_PackedMapAlgo.hxx": frozenset({"NCollection_PackedMap.hxx"}),
}


def _operator_spelling(op: str) -> str:
    return _OPERATOR_SPELLING.get(op, op)


def render_source_type(t: OCCTType) -> str:
    """Source-style type used inside the probe TU's explicit pointer casts.

    Handles are rendered as ``occ::handle<T>`` (the alias OCCT headers use;
    equivalent to ``opencascade::handle<T>``).  Base names are canonical, so
    typedefs such as ``Poly_MeshPurpose``/``unsigned int`` compare by type.
    """
    inner = f"occ::handle<{t.handle_inner}>" if t.is_handle else t.base_name
    # A reference/pointer to a raw pointer (e.g. `const char*&`, `T**`) stores
    # the inner pointer in `base_name`; pointee_pointee_is_const marks a const
    # pointee (`const char*&`), which must be spelled for the probe cast to
    # bind the real member.
    if t.pointee_pointee_is_const and not t.is_handle:
        base = inner.rstrip()
        if base.endswith("*"):
            inner = f"const {base[:-1].rstrip()} *"
    if t.is_pointer:
        s = f"{inner}*"
        if t.pointee_is_const:
            s = f"const {s}"
    elif t.is_ref:
        s = f"{inner}&&" if t.is_rvalue_ref else f"{inner}&"
        if t.is_const:
            s = f"const {s}"
    else:
        s = inner
        if t.is_const:
            s = f"const {s}"
    return s


def render_nm_type(t: OCCTType) -> str:
    """Parameter type the way `nm -C` demangles it (Itanium ABI).

    Top-level ``const`` on a by-value parameter is dropped (it is not part of
    the mangled function type); low-level const (``T const&``, ``char const*``)
    is kept with the ``const`` after the type.  Handles demangle under the real
    ``opencascade::`` namespace, not the ``occ`` alias.  The demangler puts no
    space around ``*``/``&`` (``char const*&``, ``BOPDS_DS*&``), so pointer-typed
    bases are rendered space-free to match `nm` output exactly.
    """
    inner = _nm_base(t)
    if t.is_pointer:
        return f"{inner}*"
    if t.is_ref:
        return f"{inner}&&" if t.is_rvalue_ref else f"{inner}&"
    return inner


_OCCT_TYPEDEF_NM_MAP = {
    "Standard_OStream": "std::basic_ostream<char>",
    "Standard_IStream": "std::basic_istream<char>",
    "Standard_SStream": "std::basic_stringstream<char>",
    "Standard_Boolean": "bool",
    "Standard_Integer": "int",
    "Standard_Real": "double",
    "Standard_ShortReal": "float",
    "Standard_Character": "char",
    "Standard_Byte": "unsigned char",
}


def _nm_base(t: OCCTType) -> str:
    """Space-free base of a parameter type as `nm -C` prints it.

    ``base_name`` keeps a trailing space in a pointer-typed base (``char *``,
    ``BOPDS_DS *``); the demangler instead emits ``char const*``/``BOPDS_DS*``.
    ``pointee_pointee_is_const`` turns ``char*&`` into the const-pointee form
    ``char const*&``.
    """
    if t.is_handle:
        base = f"opencascade::handle<{t.handle_inner}>"
    elif t.base_name in _OCCT_TYPEDEF_NM_MAP:
        base = _OCCT_TYPEDEF_NM_MAP[t.base_name]
    else:
        base = re.sub(r"\s*([*&])", r"\1", t.base_name)
    if t.pointee_pointee_is_const and base.endswith("*"):
        base = f"{base[:-1]} const*"
    if t.is_const and (t.is_ref or t.is_pointer):
        base = f"{base} const"
    return base


def symbol_for_method(cls, method) -> str:
    """Demangled member symbol a wrapper for `method` will reference at link."""
    args = ", ".join(render_nm_type(p.type) for p in method.parameters)
    if method.operator_type is not None:
        name = f"operator{_operator_spelling(method.operator_type.value)}"
    else:
        name = method.name
    symbol = f"{cls.name}::{name}({args})"
    if method.is_const:
        symbol += " const"
    return symbol


def _method_display_name(cls, method) -> str:
    if method.operator_type is not None:
        return f"{cls.name}::operator{_operator_spelling(method.operator_type.value)}"
    return f"{cls.name}::{method.name}"


def _probe_type(t: OCCTType, ctx: tm.TypeContext) -> str:
    """Source spelling of a signature type inside the probe TU's casts.

    Placeholder-spelled self-specializations (``NCollection_IndexedDataMap<
    TheKeyType, TheItemType, Hasher>&`` from an in-class member signature) are
    substituted with the concrete class name; the 2-arg form is the same type
    as the fully-defaulted 3-arg declaration, so the cast resolves.
    """
    return render_source_type(t)


def _field_probe_line(cls, f, index: int) -> str:
    """Probe the generated ``_ort_field_get_/set_`` accessors of a public data
    member: the getter copy-constructs the field value and the setter assigns
    it, so a member type with implicitly deleted copy semantics (e.g. a class
    holding ``std::atomic`` through a template) makes the accessors ill-formed.
    ``std::declval`` cannot be *called* in an evaluated context, so the lambda
    body reaches a reference through ``ort_field_probe_ref`` (a never-executed
    inline helper that dereferences a null pointer, making the copy/assign
    expressions compile or fail here).  Emitted as a function definition (like
    the ctor probes) since expression statements are invalid at namespace
    scope; the ``ort_field_`` function name carries the diagnostic marker."""
    assign = (f"ort_field_probe_ref<_C>().{f.name} = "
              f"ort_field_probe_ref<const _T>(); "
              if not f.is_const else "")
    head = f"void ort_field_{index:05d}() {{ (void)[] {{ using _C = ::{cls.name}; "
    tail = f"using _T = std::remove_reference<decltype(std::declval<_C&>().{f.name})>::type; "
    body = f"_T _v = ort_field_probe_ref<const _C>().{f.name}; {assign}"
    return head + tail + body + "}(); }"


def _probe_line(cls, method, index: int, ctx: tm.TypeContext) -> str:
    def resolve(t):
        base = tm._self_specialization_base(t.base_name, cls.name, ctx)
        if base is not None and base != t.base_name:
            t = replace(t, base_name=base)
        return t

    params = ", ".join(render_source_type(resolve(p.type))
                       for p in method.parameters)
    ret_is_void = (method.return_type is None
                   or (method.return_type.is_void
                       and not method.return_type.is_pointer))
    if ret_is_void:
        ret = "void"
    else:
        ret = render_source_type(resolve(method.return_type))
    if method.operator_type is not None:
        name = f"operator{_operator_spelling(method.operator_type.value)}"
    else:
        name = method.name
    target = f"&::{cls.name}::{name}"
    if method.kind == MethodKind.STATIC_METHOD:
        cast = f"static_cast<{ret} (*)({params})>({target})"
    else:
        const = " const" if method.is_const else ""
        cast = f"static_cast<{ret} (::{cls.name}::*)({params}){const}>({target})"
    return f"auto const ort_sym_{index:05d} = {cast};"


# Constructor probes
# ------------------
# Constructors cannot be named by a member pointer, so the probe references
# their symbols with `new ::Cls(args)` inside a non-static function.  The
# construction expression emits the same complete-object (C1) symbol the
# wrapper's `new Cls(...)` / `make_unique<Cls>(...)` will reference, and an
# external-linkage function cannot be optimized away.  Arguments are only
# default-constructed for types that are certain to compile (primitives,
# enums, handles, and wrapped default-constructible value classes); a ctor
# with any other parameter type is left unprobed rather than risk a false
# ill-formed flag that would wrongly drop a wrappable ctor.

_PRIMITIVE_BASES = frozenset({
    "int", "unsigned int", "long", "unsigned long", "long long",
    "unsigned long long", "short", "unsigned short", "signed char",
    "unsigned char", "char", "wchar_t", "bool", "float", "double",
    "size_t", "void", "unsigned char",
})


def _default_constructible_set(classes) -> set[str]:
    """Wrapped value classes a probe can default-construct as a value."""
    from .codegen import _default_constructible
    out: set[str] = set()
    for cls in classes:
        if cls.kind in (ClassKind.REF_COUNTED, ClassKind.EXCEPTION):
            continue
        if _default_constructible(cls):
            out.add(cls.name)
    return out


def _probe_ctor_arg(t: OCCTType, dc_set: set[str]) -> str | None:
    """A discardable value expression of type `t`, or None if constructing one
    for the probe is not known-safe.

    Arguments are cast to the exact declared parameter type so overload
    resolution is never ambiguous (a bare ``nullptr`` or ``0`` would be an
    ambiguous match when a class overloads on pointer/arithmetic types).

    A non-default-constructible value type is probed through a borrowed
    reference (``ort_field_probe_ref``, a null dereference never executed at
    runtime -- the probe TU is only compiled and nm'd).  That keeps the ctor's
    C1 symbol in the undefined-symbol set even when no value of the parameter
    type can be fabricated, closing a gap where ctors of newly-wrapped classes
    were never audited and their missing symbols only surfaced at wrapper link
    time."""
    if t.is_handle:
        if t.is_pointer:
            return f"static_cast<opencascade::handle<{t.handle_inner}>*>(nullptr)"
        return f"occ::handle<{t.handle_inner}>()"
    if t.is_pointer:
        const = "const " if t.pointee_is_const else ""
        return f"static_cast<{const}{t.base_name}*>(nullptr)"
    if t.is_rvalue_ref:
        return None
    if t.is_ref:
        if not t.is_const:
            return None
        return _probe_ctor_arg(replace(t, is_ref=False, is_rvalue_ref=False,
                                       is_const=False), dc_set)
    if t.is_enum:
        return f"static_cast<{t.base_name}>(0)"
    base = t.base_name
    if base == "bool":
        return f"static_cast<{base}>(false)"
    if base in ("char", "signed char", "unsigned char", "wchar_t"):
        return f"static_cast<{base}>('\\0')"
    if base in _PRIMITIVE_BASES:
        return f"static_cast<{base}>(0)"
    if base in dc_set:
        return f"{base}()"
    return f"ort_field_probe_ref<const {base}>()"


def _ctor_probe_line(cls, ctor, index: int, dc_set: set[str], ctx) -> str:
    """Probe line referencing the native constructor symbol a wrapper ctor
    emits; "" means the ctor is not probed."""
    if cls.is_abstract or cls.kind == ClassKind.EXCEPTION:
        return ""
    from .codegen import _cg
    cg = _cg(cls, ctx)
    if cg.storage == "none":
        return ""
    args = []
    for p in ctor.parameters:
        arg = _probe_ctor_arg(p.type, dc_set)
        if arg is None:
            return ""
        args.append(arg)
    joined = ", ".join(args)
    if cg.storage == "handle":
        # `new Cls(args)` also covers plain unique_ptr storage (make_unique is
        # new underneath) and references the same C1 symbol + class operator new.
        return (f"::{cls.name}* ort_ctor_{index:05d}() "
                f"{{ return new ::{cls.name}({joined}); }}")
    if cls.wrapper_name in ctx.stdalloc:
        # stdalloc wrappers placement-construct on Standard::Allocate memory and
        # never call the class operator new (which allocator-tagged classes hide
        # and protected bases make inaccessible); a discarded prvalue references
        # the same C1 ctor symbol without pulling in operator new.
        return (f"void ort_ctor_{index:05d}() "
                f"{{ (void)::{cls.name}({joined}); }}")
    # unique_ptr storage (plain operator new); the wrapper emits make_unique.
    return (f"void ort_ctor_{index:05d}() "
            f"{{ (void)::{cls.name}({joined}); }}")


def _default_ctor_probe_line(cls, ctx, index: int) -> str:
    """Probe line for the native default-construction a wrapper's own default
    constructor emits (``_native()`` or ``_handle = new Cls()``).

    The default ctor itself is never bound as a factory (see
    ``_default_ctor``), so it is invisible to the ctor probes above; yet a
    value/handle-stored wrapper references its symbol from the member init.
    """
    if cls.kind == ClassKind.EXCEPTION:
        return ""
    from .codegen import _cg
    cg = _cg(cls, ctx)
    if cg.storage == "handle":
        if not (cls.has_public_default_ctor and not cls.is_abstract):
            return ""
        return (f"::{cls.name}* ort_dctor_{index:05d}() "
                f"{{ return new ::{cls.name}(); }}")
    if cg.storage == "native" and not cg.inherited_native:
        return (f"void ort_dctor_{index:05d}() "
                f"{{ (void)::{cls.name}(); }}")
    return ""


def _copy_probe_line(cls, ctx, index: int) -> str:
    """Probe the copy operation a wrapped value/reference return emits (native
    wrappers copy-assign into ``_native``; unique_ptr wrappers copy-construct
    via make_unique).  A rejection means the OCCT type is implicitly
    non-copyable (copy semantics deleted through members/bases the extractor
    cannot see), so methods returning it cannot be bound.

    The operation is wrapped in a *template* helper: when it is ill-formed the
    instantiation happens inside an OCCT template member (e.g. a
    ``NCollection_Map<Cell, Hasher>::operator=`` inlined in a class body), and
    a bare expression in this non-template function would anchor GCC's
    "required from here" inside the OCCT header instead of at this probe line.
    The helper template puts the probe line in the instantiation chain.
    """
    from .codegen import _cg
    cg = _cg(cls, ctx)
    if cg.storage not in ("native", "unique_ptr"):
        return ""
    helper = ("ort_copy_probe_construct<_C>()" if cg.storage == "unique_ptr"
              else "ort_copy_probe_assign<_C>()")
    return (f"void ort_copy_{index:05d}() "
            f"{{ (void)[] {{ using _C = ::{cls.name}; {helper}; }}; }}")


# ---------------------------------------------------------------------------
# Probe TU generation
# ---------------------------------------------------------------------------

def _type_headers(t: OCCTType, ctx: tm.TypeContext) -> list[str]:
    """OCCT header basenames a signature type requires to be complete."""
    if t.is_handle and t.handle_inner in ctx.wrapped:
        return [ctx.occt_headers.get(t.handle_inner, t.handle_inner + ".hxx")]
    key = tm._wrapped_key(t.base_name, ctx)
    if key is not None:
        return [ctx.occt_headers.get(key, key + ".hxx")]
    # A templated type the header map has no key for (e.g. spelled with
    # defaulted template args): the class template header is still required.
    m = re.match(r"^([A-Za-z_]\w*)<", t.base_name)
    if m:
        tname = m.group(1)
        return [ctx.occt_headers.get(tname, tname + ".hxx")]
    return []


def probe_headers(classes, ctx: tm.TypeContext, install: OCCTInstall) -> list[Path]:
    """OCCT headers the probe TU needs, include-closure ordered (deps first).

    The BFS closure only orders headers linked by ``#include``; several OCCT
    headers are not self-contained and rely on a type being declared by an
    earlier include (e.g. ``GeomFill_SimpleBound.hxx`` uses ``Adaptor3d_Curve``
    without including its header).  ``_hygiene_order`` closes that gap so the
    probe compiles deterministically regardless of set iteration order.
    """
    names: set[str] = set()
    for cls in classes:
        if cls.header_file:
            names.add(Path(cls.header_file).name)
        for e in cls.extra_occt_includes:
            if e and (install.include_dir / e).exists():
                names.add(e)
        for base in cls.base_classes:
            if base in ctx.occt_classes:
                names.add(ctx.occt_headers.get(base, base + ".hxx"))
        for method in cls.all_methods:
            for p in method.parameters:
                names.update(_type_headers(p.type, ctx))
            if method.return_type is not None:
                names.update(_type_headers(method.return_type, ctx))
        for f in cls.fields:
            names.update(_type_headers(f.type, ctx))
    paths = [install.include_dir / n for n in names
             if (install.include_dir / n).exists()]
    return _hygiene_order(include_closure(paths, install, include_self=True),
                          ctx)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _hygiene_order(closure: list[Path], ctx: tm.TypeContext) -> list[Path]:
    """Deterministically reorder ``closure`` so each header comes after the
    declaring header of every OCCT class name it references."""
    idx = {h.name: i for i, h in enumerate(closure)}
    exact: dict[str, int] = {}
    prefix: dict[str, list[int]] = {}
    for cls, hdr in ctx.occt_headers.items():
        if not hdr:
            continue
        j = idx.get(hdr)
        if j is None:
            continue
        m = _IDENT_RE.match(cls)
        token = m.group(0) if m else cls
        if token == cls:
            exact[cls] = j
        else:
            prefix.setdefault(token, []).append(j)
    if not exact and not prefix:
        return closure

    deps: list[set[int]] = [set() for _ in closure]
    for i, h in enumerate(closure):
        try:
            text = h.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for tok in _IDENT_RE.findall(text):
            j = exact.get(tok)
            if j is not None:
                if j != i:
                    deps[i].add(j)
                continue
            for j in prefix.get(tok, ()):
                if j != i:
                    deps[i].add(j)
    for name, must_follow in _HEADER_PRECEDENCE.items():
        i = idx.get(name)
        if i is None:
            continue
        for pre in must_follow:
            j = idx.get(pre)
            if j is not None and j != i:
                deps[i].add(j)

    heap = [i for i, d in enumerate(deps) if not d]
    heapq.heapify(heap)
    emitted = [False] * len(closure)
    out: list[Path] = []
    while heap:
        i = heapq.heappop(heap)
        if emitted[i]:
            continue
        emitted[i] = True
        out.append(closure[i])
        for k in range(len(closure)):
            if not emitted[k] and i in deps[k]:
                deps[k].discard(i)
                if not deps[k]:
                    heapq.heappush(heap, k)
    if len(out) != len(closure):
        rest = [i for i in range(len(closure)) if not emitted[i]]
        rest_set = set(rest)
        pred: dict[int, set[int]] = {i: set() for i in rest}
        for name, must_follow in _HEADER_PRECEDENCE.items():
            i = idx.get(name)
            if i is None or i not in rest_set:
                continue
            for pre in must_follow:
                j = idx.get(pre)
                if j is not None and j in rest_set and j != i:
                    pred[i].add(j)
        heap2 = [i for i in rest if not pred[i]]
        heapq.heapify(heap2)
        emitted2: set[int] = set()
        while heap2:
            i = heapq.heappop(heap2)
            if i in emitted2:
                continue
            emitted2.add(i)
            out.append(closure[i])
            for k in rest:
                if k not in emitted2 and i in pred[k]:
                    pred[k].discard(i)
                    if not pred[k]:
                        heapq.heappush(heap2, k)
        for i in rest:
            if i not in emitted2:
                out.append(closure[i])
    return out


def _probe_lines(classes, dc_set: set[str], ctx) -> list[str]:
    """Probe body lines (indented, with ``// Class::method`` comments).

    Shared by ``generate_probe_tu`` and the per-module chunks written by
    ``write_probe_parts`` so the aggregate and the parts stay consistent.
    ``dc_set`` is passed in because it must be computed over *all* wrapped
    classes (a ctor probe may need to value-construct an argument of a type
    that lives in another module).
    """
    lines: list[str] = []
    index = 0
    for cls in classes:
        for method in (cls.methods + cls.operators + cls.static_methods):
            if method.skip or method.is_deleted or method.is_pure_virtual \
                    or method.is_variadic:
                continue
            lines.append(f"    // {_method_display_name(cls, method)}")
            lines.append(f"    {_probe_line(cls, method, index, ctx)}")
            index += 1
        for ctor in cls.constructors:
            if ctor.skip or ctor.is_deleted or ctor.is_pure_virtual:
                continue
            line = _ctor_probe_line(cls, ctor, index, dc_set, ctx)
            if not line:
                continue
            lines.append(f"    // {_method_display_name(cls, ctor)}")
            lines.append(f"    {line}")
            index += 1
        line = _default_ctor_probe_line(cls, ctx, index)
        if line:
            lines.append(f"    // {cls.name}::{cls.name} (default construction)")
            lines.append(f"    {line}")
            index += 1
        for f in cls.fields:
            if f.skip or not f.is_public:
                continue
            # Array members are mapped element-wise by the accessor (not by
            # value copy), so a by-value copy probe would falsely reject them.
            if f.type.spelling.rstrip().endswith("]"):
                continue
            lines.append(f"    // {cls.name}::{f.name} (field accessor)")
            lines.append(f"    {_field_probe_line(cls, f, index)}")
            index += 1
        line = _copy_probe_line(cls, ctx, index)
        if line and cls.returnable:
            lines.append(f"    // {cls.name}::copy (return value)")
            lines.append(f"    {line}")
            index += 1
    return lines


def _probe_preamble(headers) -> list[str]:
    """Shared preamble lines: the header closure plus the probe helper
    templates (ort_field_probe_ref / ort_copy_probe_*)."""
    return (["// Auto-generated symbol audit probe TU -- DO NOT EDIT"]
            + [f"#include <{h.name}>" for h in headers]
            + ["",
               "#include <utility>   // std::declval (field-accessor probes)",
               "#include <type_traits>  // std::remove_reference (field probes)",
               "",
               "// Compile-time-only reference into a hypothetical instance; the",
               "// field probes copy/assign through it without ever constructing.",
               "template <typename T> inline T& ort_field_probe_ref() noexcept {",
               "    T* p = nullptr;",
               "    return *p;",
               "}",
               "",
               "// Template-wrapped copy probes (see _copy_probe_line): keeping the",
               "// copy operation inside a template anchors GCC's instantiation",
               "// chain at the probe line instead of inside an OCCT header.",
               "template <typename T> inline void ort_copy_probe_assign() {",
               "    ort_field_probe_ref<T>() = ort_field_probe_ref<const T>();",
               "}",
               "template <typename T> inline void ort_copy_probe_construct() {",
               "    T a(ort_field_probe_ref<const T>()); (void)a;",
               "}",
               ""])


def _compose_probe(headers, lines: list[str]) -> str:
    """Assemble a probe TU from the ordered header closure and body lines."""
    out = _probe_preamble(headers)
    out.append("// One discarded address-of per generated wrapper method; overloads")
    out.append("// are disambiguated by explicit pointer casts.  The undefined symbols")
    out.append("// of this TU are the member/static symbols the wrappers will emit.")
    if not lines:
        out.append("auto const ort_sym_none = 0;")
    else:
        out.extend(lines)
    return "\n".join(out) + "\n"


def generate_probe_tu(modules, ctx: tm.TypeContext, install: OCCTInstall,
                      headers=None) -> str:
    """A TU referencing every method the wrappers will emit at link time."""
    classes = [cls for m in modules for cls in m.classes if not cls.skip]
    if headers is None:
        headers = probe_headers(classes, ctx, install)
    lines = _probe_lines(classes, _default_constructible_set(classes), ctx)
    return _compose_probe(headers, lines)


# ---------------------------------------------------------------------------
# nm parsing
# ---------------------------------------------------------------------------

def write_probe_parts(probe_path: Path, modules, ctx: tm.TypeContext,
                      install: OCCTInstall,
                      max_parts: int | None = None) -> list[Path]:
    """Write the aggregate probe TU plus per-chunk part TUs; return the parts.

    The audit compiles the parts in parallel (one g++ per part), so a probe
    covering ~350 modules does not collapse to a single-threaded compile.  The
    aggregate ``probe.cpp`` is still written so ``audit --probe probe.cpp`` and
    the regen loop keep working unchanged when no parts exist.

    Every part reuses the *full* header closure of the aggregate probe: a
    signature may reference types whose headers live in another module (e.g. a
    static class like ``IGESSolid`` used from an Interface method), so a part
    cannot get away with only its own modules' headers.  The per-module probe
    bodies are appended under unique namespaces, since each reuses index-local
    symbol names (``ort_sym_00000`` & co.).  Header parsing is thus duplicated
    across parts, but the template-instantiation work (the dominant cost) is
    split across them.
    """
    all_classes = [cls for m in modules for cls in m.classes if not cls.skip]
    if probe_path.is_dir():
        probe_path = probe_path / "probe.cpp"
    dc_set = _default_constructible_set(all_classes)
    headers = probe_headers(all_classes, ctx, install)
    probe_path.write_text(generate_probe_tu(modules, ctx, install, headers))
    preamble = "\n".join(_probe_preamble(headers)) + "\n"
    parts = max_parts or min(os.cpu_count() or 4, 16)
    # Build one probe body per module (cheap: no per-module header closure,
    # the shared preamble already covers every referenced header), then greedily
    # pack modules into chunks by body size so the biggest TUs (the synthesized
    # NCollection module) don't all land in one part.
    per_module = [
        (m.name, _probe_lines(
            [cls for cls in m.classes if not cls.skip], dc_set, ctx))
        for m in modules if any(not c.skip for c in m.classes)]
    body_for = lambda name, lines: (  # noqa: E731
        name, "".join(l + "\n" for l in lines))
    sizes: list[int] = [0] * parts
    chosen: list[list[tuple[str, str]]] = [[] for _ in range(parts)]
    for name, lines in sorted(per_module, key=lambda kv: len(kv[1]), reverse=True):
        i = sizes.index(min(sizes))
        chosen[i].append(body_for(name, lines))
        sizes[i] += sum(len(l) for l in lines)
    out: list[Path] = []
    for i, group in enumerate(chosen):
        if not group:
            continue
        # Every module's body reuses index-local symbol names (ort_sym_00000 &
        # co.), so each is wrapped in a unique namespace to avoid collisions.
        chunks = [f"namespace ort_p{i}_m{k} {{\n{text}\n}}"
                  for k, (_, text) in enumerate(group)]
        part = probe_path.with_name(f"{probe_path.stem}.part{i}.cpp")
        part.write_text(preamble + "\n".join(chunks) + "\n")
        out.append(part)
    return out


def _nm_undefined(obj: Path, nm_tool: str) -> list[tuple[str, str]]:
    out = subprocess.run([nm_tool, "-u", "-C", str(obj)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        out = subprocess.run([nm_tool, "-u", str(obj)], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"nm failed on {obj}: {out.stderr[:1000]}")
    syms: list[tuple[str, str]] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        if line.startswith("(undefined)"):
            m = re.match(r"^\(undefined\)\s+(?:(?:weak\s+|non-)?external\s+)?(.*)$", line)
            if m:
                syms.append(("U", _normalize_symbol(m.group(1).strip())))
                continue
        m = re.match(r"^([A-Za-z])\s+(.*)$", line)
        if m:
            syms.append((m.group(1), _normalize_symbol(m.group(2).strip())))
            continue
        syms.append(("U", _normalize_symbol(line)))
    return syms


def _nm_defined_one(lib: Path, nm_tool: str, dynamic: bool) -> set[str]:
    args = [nm_tool, "-C", "--defined-only"]
    if dynamic:
        args.insert(1, "-D")
    out = subprocess.run(args + [str(lib)], capture_output=True, text=True)
    if out.returncode != 0:
        args = [nm_tool, "-C", "-g", str(lib)]
        out = subprocess.run(args, capture_output=True, text=True)
        if out.returncode != 0:
            args = [nm_tool, "-C", str(lib)]
            out = subprocess.run(args, capture_output=True, text=True)
            if out.returncode != 0:
                return set()
    defined: set[str] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.endswith(":") or line.startswith("(undefined)"):
            continue
        m_darwin = re.match(r"^[0-9a-fA-F]+\s+\([^)]+\)\s+(?:(?:weak\s+|non-)?external\s+)?(.*)$", line)
        if m_darwin:
            defined.add(_normalize_symbol(m_darwin.group(1).strip()))
            continue
        m_gnu = re.match(r"^[0-9a-fA-F]{8,16}\s+([A-Za-z])\s+(.*)$", line)
        if m_gnu:
            letter = m_gnu.group(1)
            if letter not in ("U", "u"):
                defined.add(_normalize_symbol(m_gnu.group(2).strip()))
            continue
    return defined


def _defined_symbols(lib_dir: Path, nm_tool: str,
                     max_workers: int | None = None) -> set[str]:
    static = sorted(lib_dir.glob("*.a"))
    shared = sorted(lib_dir.glob("*.so*")) + sorted(lib_dir.glob("*.dylib")) + sorted(lib_dir.glob("*.dll"))
    libs = static or shared
    if not libs:
        return set()
    # nm over ~60 libraries is pure subprocess I/O: parallelize it.
    workers = max_workers or min(os.cpu_count() or 4, 16)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(lambda lib: _nm_defined_one(lib, nm_tool, not static),
                         libs)
    defined: set[str] = set()
    for part in results:
        defined |= part
    return defined


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------

def _occt_lib_dir(project_root: Path | None, install: OCCTInstall) -> Path | None:
    triplet = os.environ.get("VCPKG_DEFAULT_TRIPLET", "x64-linux")
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / "vcpkg" / "installed" / triplet / "lib")
    candidates.append(install.include_dir.parent.parent / "lib")
    candidates.append(install.include_dir.parent / "lib")
    for d in candidates:
        if d.is_dir() and (list(d.glob("libTKMath.*")) or list(d.glob("TKMath.*"))):
            return d
    return None


def _gcc_args(args: list[str]) -> list[str]:
    """Filter libclang/compile-db flags that g++ would reject or that drag in
    godot-cpp (the `-include occt_guard.hxx` of the real compile DB)."""
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-resource-dir"):
            continue
        if arg in ("-include", "-Xclang", "-x", "-c", "-o"):
            skip_next = True
            continue
        if arg in ("-target", "--target"):
            # Clang-only two-word form (``-target arm-linux-gnueabihf``); the
            # probe compile runs on the host so g++ cannot honour it.
            skip_next = True
            continue
        if arg.startswith(("--target=", "-target=", "--sysroot=")):
            # Clang-only cross-target / sysroot flags; the host probe cannot
            # reproduce the cross ABI (the symbol diff that would need it is
            # skipped separately -- see run_audit).
            continue
        if arg.startswith("-isystem "):
            out += ["-isystem", arg[len("-isystem "):]]
            continue
        if arg.startswith("-MJ") or arg.startswith("-dependency-file"):
            continue
        out.append(arg)
    return out


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write findings as one-per-line, or an empty file when there are none.

    A bare trailing newline (``"\n"``) would make ``test -s`` treat the file
    as non-empty, so an empty result must produce a zero-byte file.
    """
    path.write_text("" if not lines else "\n".join(sorted(lines)) + "\n")


def run_audit(probe_path: Path, work_dir: Path, out_path: Path,
              project_root: Path | None, install: OCCTInstall,
              args: list[str], occt_classes: set[str],
              compiler: str = "g++", nm_tool: str = "nm",
              illformed_path: Path | None = None) -> list[str]:
    """Compile the probe, diff undefined vs library-defined symbols.

    Returns the sorted list of missing member symbols (also written to
    `out_path`).  When the probe fails to compile, the offending members are
    extracted from the compiler diagnostics and written to `illformed_path`
    (default: ``<work_dir>/illformed.txt`` as ``Class::method`` lines) so
    pass-2 regeneration can skip them; the returned list is then empty.
    Raises if the tools/libs are unavailable (caller decides how to degrade).
    """
    if shutil.which(compiler) is None or shutil.which(nm_tool) is None:
        raise FileNotFoundError(
            f"symbol audit needs {compiler!r} and {nm_tool!r} on PATH")
    lib_dir = _occt_lib_dir(project_root, install)
    if lib_dir is None:
        raise FileNotFoundError("no OCCT library directory found for symbol audit")
    # The probe must be compiled with the target's data model so its undefined
    # symbols line up with the target OCCT libraries.  A plain `-m32` (x86 on
    # an x64 host) is honored by g++; a clang-only `--target=` (arm, android,
    # wasm) cannot be reproduced by the host toolchain for the undefined-symbol
    # diff, so that stage is skipped.  The ill-formed-method pass still runs:
    # copy-constructibility / default-constructibility / field-accessor
    # validity are data-model independent, and cross-target wrappers must not
    # emit copies of implicitly-deleted types (e.g. BRepMesh_CircleTool, whose
    # copy is deleted through a move-holding NCollection_Map member the
    # extractor cannot see).
    cross_target = any(a.startswith(("--target=", "-target")) for a in args)
    if cross_target:
        print("audit          : cross target; undefined-symbol diff skipped "
              "(ill-formed pass still runs)", file=sys.stderr)

    work_dir.mkdir(parents=True, exist_ok=True)
    if illformed_path is None:
        illformed_path = work_dir / "illformed.txt"
    # Both outputs are refreshed on every run so a stale file from a previous
    # invocation (e.g. a probe that has since been fixed) cannot leak into
    # pass-2 regeneration.
    out_path.write_text("")
    illformed_path.write_text("")

    def compile_one(tu: Path) -> tuple[Path, subprocess.CompletedProcess]:
        obj = work_dir / (tu.stem + ".o")
        cmd = [compiler, "-std=gnu++17", "-fPIC", "-w",
               "-ftemplate-backtrace-limit=0", "-c",
               str(tu), "-o", str(obj), *_gcc_args(args)]
        return tu, subprocess.run(cmd, capture_output=True, text=True)

    # When generate-all emitted probe.partN.cpp chunks (see write_probe_parts),
    # compile them concurrently -- each is an independent TU -- instead of one
    # giant single-threaded compile.
    parts = sorted(probe_path.parent.glob(f"{probe_path.stem}.part*.cpp"))
    if parts:
        workers = min(len(parts), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            compiled = list(ex.map(compile_one, parts))
    else:
        compiled = [compile_one(probe_path)]

    illformed: set[str] = set()
    failed_unmapped = False
    for tu, res in compiled:
        if res.returncode == 0:
            continue
        # A generated method whose body does not compile (e.g. a synthesized
        # NCollection_Vec3<unsigned long>::cwiseAbs instantiating an ambiguous
        # std::abs) aborts the probe.  Map each failing probe line back to its
        # "// Class::method" comment and skip exactly those methods in pass 2.
        mapped = _extract_illformed(res.stderr, tu)
        if not mapped:
            failed_unmapped = True
            print(f"symbol audit : {tu.name} failed; no methods mapped to errors"
                  f" (tried {tu.name}:NN) -- using pass-1 output",
                  file=sys.stderr)
            print(res.stderr[-2000:], file=sys.stderr, end="")
        else:
            illformed |= mapped
    if failed_unmapped:
        raise RuntimeError("symbol audit probe failed to compile")
    if illformed:
        _write_lines(illformed_path, sorted(illformed))
        # A compile failure gives no object file for nm: keep the missing set
        # from the parts that *did* compile (below) rather than bailing out.

    defined: set[str] = set()
    missing: set[str] = set()
    if not cross_target:  # cross-target undefined symbols cannot match host libs
        defined = _defined_symbols(lib_dir, nm_tool)
        if not defined:
            print("audit          : warning: no defined symbols extracted from OCCT libraries; "
                  "skipping undefined-symbol diff to avoid false positives", file=sys.stderr)
        else:
            for tu, res in compiled:
                if res.returncode != 0:
                    continue
                obj = work_dir / (tu.stem + ".o")
                for letter, name in _nm_undefined(obj, nm_tool):
                    if letter != "U":
                        continue
                    cls = name.split("::")[0]
                    if cls not in occt_classes:
                        continue
                    if name not in defined:
                        missing.add(name)
    _write_lines(out_path, sorted(missing))
    return sorted(missing)


def _extract_illformed(stderr: str, probe_path: Path) -> set[str]:
    """Class::method names of probe lines rejected by the compiler.

    Every probe line is preceded by a ``// Class::method`` comment; GCC/Clang
    diagnostics reference the offending ``ort_sym_*`` line by file:line, so the
    nearest preceding comment names the method whose instantiation failed.
    """
    probe = probe_path.read_text().splitlines()
    line_index: dict[int, str] = {}
    last_comment = ""
    for no, text in enumerate(probe, start=1):
        line = text.strip()
        if line.startswith("// ") and "::" in line:
            last_comment = line[3:].strip()
        elif "ort_sym_" in line or "ort_ctor_" in line or "ort_dctor_" in line \
                or "ort_field_" in line or "ort_copy_" in line:
            line_index[no] = last_comment
    out: set[str] = set()
    for m in re.finditer(rf"{re.escape(probe_path.name)}:(\d+):", stderr):
        target = line_index.get(int(m.group(1)), "")
        if target:
            out.add(target)
    return out


def load_illformed(path: Path) -> set[str]:
    """Read an ill-formed-methods file (one ``Class::method`` per line)."""
    illformed: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        illformed.add(line)
    return illformed


def apply_illformed(modules, illformed: set[str]) -> int:
    """Skip every generated method whose instantiation does not compile.

    These are OCCT template members that are ill-formed for the substituted
    arguments (e.g. ``NCollection_Vec3<unsigned long>::cwiseAbs`` calling an
    ambiguous ``std::abs``); the API itself is unusable there, so the method
    is dropped exactly as if it were unmappable.

    ``Class::Class (default construction)`` entries come from the probe of the
    wrapper's own default constructor (``_native()`` / ``new Cls()``); a
    rejection means ``T()`` does not exist even though the extractor could not
    tell (libclang misses a deleted implicit default ctor, e.g. when the base
    class suppresses it).  The class is pinned not-default-constructible so
    codegen falls back to unique_ptr storage.
    """
    skipped = 0
    for module in modules:
        for cls in module.classes:
            if cls.skip:
                continue
            dctor = f"{cls.name}::{cls.name} (default construction)"
            if dctor in illformed:
                cls.default_constructible = False
                cls.has_public_default_ctor = False
                skipped += 1
                # No `continue` here: a class can be flagged both as not
                # default-constructible AND not returnable (the dctor label
                # must not shadow the copy-return label, or the copy probe is
                # re-emitted next pass and convergence never terminates).
            for method in cls.all_methods:
                if method.skip:
                    continue
                # _extract_illformed records operators as ``Class::operator()``
                # (via _method_display_name), not the raw ``Class::()``; match
                # with the same spelling so operator methods are actually skipped.
                if _method_display_name(cls, method) in illformed:
                    method.skip = True
                    method.skip_reason = ("ill-formed instantiation "
                                          "(OCCT member does not compile for "
                                          "the substituted template args)")
                    skipped += 1
            for f in cls.fields:
                if f.skip:
                    continue
                # ``Class::field (field accessor)`` entries come from the probe
                # of the generated get/set property accessors: a member whose
                # type has implicitly deleted copy semantics (the getter copies
                # it, the setter assigns it) cannot be exposed as a property.
                label = f"{cls.name}::{f.name} (field accessor)"
                if label in illformed:
                    f.skip = True
                    f.skip_reason = ("ill-formed field accessor "
                                     "(field type is not copyable)")
                    skipped += 1
            # ``Class::copy (return value)`` entries come from the probe of the
            # copy operation a wrapped return emits (copy-assign for native
            # storage, copy-construct for unique_ptr storage).  A rejection
            # means the OCCT type is implicitly non-copyable through members or
            # bases; value/reference returns of it cannot be bound.
            if f"{cls.name}::copy (return value)" in illformed:
                cls.returnable = False
                skipped += 1
    return skipped


# ABI-tagged std templates demangle with the `__cxx11` inline namespace and
# the full default template arguments; the IR's `std::basic_*<char>` short
# forms are mapped back so pass-2 symbol matching is independent of libstdc++.
# libstdc++'s demangler also prints the standard typedefs (`std::ostream`,
# `std::string`) where the IR keeps the underlying `std::basic_*<char>` form;
# both spellings must collapse onto the same symbol name.
_STD_TEMPLATE_MAP = {
    # libc++ (Apple / LLVM)
    "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >": "std::basic_string<char>",
    "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>": "std::basic_string<char>",
    "std::__1::basic_stringstream<char, std::__1::char_traits<char>, std::__1::allocator<char> >": "std::basic_stringstream<char>",
    "std::__1::basic_stringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>": "std::basic_stringstream<char>",
    "std::__1::basic_ostream<char, std::__1::char_traits<char> >": "std::basic_ostream<char>",
    "std::__1::basic_ostream<char, std::__1::char_traits<char>>": "std::basic_ostream<char>",
    "std::__1::basic_istream<char, std::__1::char_traits<char> >": "std::basic_istream<char>",
    "std::__1::basic_istream<char, std::__1::char_traits<char>>": "std::basic_istream<char>",
    # libstdc++ (GCC)
    "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >": "std::basic_string<char>",
    "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>": "std::basic_string<char>",
    "std::basic_string<char, std::char_traits<char>, std::allocator<char> >": "std::basic_string<char>",
    "std::basic_string<char, std::char_traits<char>, std::allocator<char>>": "std::basic_string<char>",
    "std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >": "std::basic_stringstream<char>",
    "std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char>>": "std::basic_stringstream<char>",
    "std::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >": "std::basic_stringstream<char>",
    "std::basic_stringstream<char, std::char_traits<char>, std::allocator<char>>": "std::basic_stringstream<char>",
    "std::__cxx11::basic_ostream<char, std::char_traits<char> >": "std::basic_ostream<char>",
    "std::__cxx11::basic_ostream<char, std::char_traits<char>>": "std::basic_ostream<char>",
    "std::basic_ostream<char, std::char_traits<char> >": "std::basic_ostream<char>",
    "std::basic_ostream<char, std::char_traits<char>>": "std::basic_ostream<char>",
    "std::__cxx11::basic_istream<char, std::char_traits<char> >": "std::basic_istream<char>",
    "std::__cxx11::basic_istream<char, std::char_traits<char>>": "std::basic_istream<char>",
    "std::basic_istream<char, std::char_traits<char> >": "std::basic_istream<char>",
    "std::basic_istream<char, std::char_traits<char>>": "std::basic_istream<char>",
    # Standard and OCCT typedefs
    "std::ostream": "std::basic_ostream<char>",
    "std::istream": "std::basic_istream<char>",
    "std::string": "std::basic_string<char>",
    "std::stringstream": "std::basic_stringstream<char>",
    "Standard_OStream": "std::basic_ostream<char>",
    "Standard_IStream": "std::basic_istream<char>",
    "Standard_SStream": "std::basic_stringstream<char>",
    "Standard_Boolean": "bool",
    "Standard_Integer": "int",
    "Standard_Real": "double",
    "Standard_ShortReal": "float",
    "Standard_Character": "char",
    "Standard_Byte": "unsigned char",
}


def _normalize_symbol(name: str) -> str:
    for full, short in _STD_TEMPLATE_MAP.items():
        if full in name:
            name = name.replace(full, short)
    # The Itanium demangler separates nested closing brackets with a space
    # (`handle<NCollection_HArray1<double> >`); our source spellings (and the
    # wrapper's undefined symbols) use the adjacent `>>` form.  Normalize every
    # nesting level so symbol matching is independent of the demangler.
    while "> >" in name:
        name = name.replace("> >", ">>")
    return name


def load_missing(path: Path) -> set[str]:
    """Read a missing-symbols file (one demangled symbol per line)."""
    missing: set[str] = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        missing.add(_normalize_symbol(line))
    return missing


def apply_missing(modules, missing: set[str]) -> int:
    """Skip every generated method whose link symbol is absent from the libs."""
    skipped = 0
    for module in modules:
        for cls in module.classes:
            if cls.skip:
                continue
            for method in cls.all_methods:
                if method.skip:
                    continue
                if _normalize_symbol(symbol_for_method(cls, method)) in missing:
                    method.skip = True
                    method.skip_reason = "missing OCCT symbol (not exported by linked libraries)"
                    if method.kind == MethodKind.CONSTRUCTOR and not method.parameters:
                        # The wrapper's own default ctor constructs the native
                        # object (`_native()` / `_handle = new Cls()`); when the
                        # OCCT default ctor is absent from the libs, fall back to
                        # no-default-construction (unique_ptr / null handle).
                        # NB: only a zero-arg CONSTRUCTOR means this -- a missing
                        # zero-arg regular method (e.g. an unexported accessor)
                        # must not demote the whole class to unique_ptr storage.
                        cls.has_public_default_ctor = False
                    skipped += 1
    return skipped
