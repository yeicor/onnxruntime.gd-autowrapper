"""Generate wrapper C++ (Ort*.hpp/.cpp) + OrtEnums + module.h from IR.

Output contract mirrors the legacy pipeline exactly (method names, factories,
hash suffixes, stream absorption, guard macros, field accessors, enums).
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .model import (ClassDecl, ClassKind, FieldDecl, MethodDecl, MethodKind,
                    ModuleDecl, OCCTType)
from .names import (_type_to_string, get_method_unique_name,
                    group_overloads, to_snake_case)
from . import typemap as tm

# ---------------------------------------------------------------------------
# Fixed source blocks
# ---------------------------------------------------------------------------

GODOT_INCLUDES = """#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/core/class_db.hpp>"""

GCC_CHANGES = """#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wchanges-meaning"
#endif"""

GCC_DEPRECATED = """#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif"""

OPERATOR_SPELLING = {
    "unary_minus": "-", "unary_plus": "+", "*deref": "*", "call": "()",
}


@dataclass
class CgClass:
    cls: ClassDecl
    wrapper_base: str | None      # wrapper name of wrapped OCCT base, or None
    base_occt: str | None         # OCCT name of that base
    storage: str                  # "handle" | "native" | "unique_ptr"
    has_sync: bool = False
    is_aggregate: bool = False
    inherited_native: bool = False  # shares base wrapper's _native via _native_ref()


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _field_breaks_copy_semantics(f: FieldDecl, ctx: tm.TypeContext) -> bool:
    """True if data member `f` makes the enclosing class non-copy-assignable.

    Reference members and const members delete copy assignment by rule; a
    by-value member whose type is itself in ``ctx.noncopyable`` (a wrapped
    OCCT value or a known non-copyable std value, per extract.py) propagates
    it.  Pointer/handle members never do (copying the pointer/refcount is
    fine even when the pointee is non-copyable).
    """
    if f.type.is_ref or f.is_const:
        return True
    return (not f.type.is_pointer and not f.type.is_handle
            and f.type.base_name in ctx.noncopyable)


def build_context(modules: list[ModuleDecl]) -> tm.TypeContext:
    """Build a TypeContext shared across modules (in include-DAG order).

    Wrapped classes, sync bases, enums and unique_ptr storage are accumulated
    so that a module can reference classes/enums of every earlier module.
    """
    ctx = tm.TypeContext(module_name="__all__")
    for module in modules:
        if module.data_model:
            # The parse target's data model (probed at scan time, recorded in
            # the IR): size-sensitive wrapper storage follows it.
            ctx.data_model = module.data_model
            break
    for module in modules:
        ctx.occt_classes |= {cls.name for cls in module.classes}
        for cls in module.classes:
            if cls.header_file:
                ctx.occt_headers[cls.name] = Path(cls.header_file).name
    for module in modules:
        for cls in module.classes:
            if cls.header_file and cls.extra_occt_includes:
                ctx.occt_extras[Path(cls.header_file).name[:-4]] = list(
                    cls.extra_occt_includes)
    for module in modules:
        for cls in module.classes:
            if not cls.skip and cls.name != module.name:
                ctx.wrapped[cls.name] = cls.wrapper_name
    # Empty value-class type tags that derive a wrapped value class share the
    # base wrapper's native storage (accessed via a downcast _native_ref()).
    # Restricted to TopoDS_Shape tags: those classes are provably member-less,
    # so the base storage layout matches and a downcast reference is sound.
    # Other hierarchies (e.g. BRepMesh splitters) carry state/vptrs and would
    # slice or overflow, so they stay standalone RefCounted wrappers.
    for module in modules:
        for cls in module.classes:
            if cls.skip or cls.name == module.name:
                continue
            if cls.kind == ClassKind.REF_COUNTED:
                continue
            if not _default_constructible(cls):
                continue
            base = next((b for b in cls.base_classes if b in ctx.wrapped), None)
            if base != "TopoDS_Shape":
                continue
            if cls.fields or cls.methods or cls.operators or cls.static_methods:
                continue
            if any(not _default_ctor(c) for c in cls.constructors):
                continue
            ctx.inherited_value.add(cls.wrapper_name)
    # Group inherited-storage concrete type tags by their base wrapper so the
    # base class can expose generated downcast factories (cast_<tag>() and a
    # generic cast()).  These wrap the same _native storage as the base, so a
    # static downcast is sound for every such tag, for any hierarchy.
    ctx.inherited_children: dict[str, list[ClassDecl]] = {}
    for module in modules:
        for cls in module.classes:
            if cls.wrapper_name not in ctx.inherited_value:
                continue
            base_occt = next((b for b in cls.base_classes if b in ctx.wrapped), None)
            base_wrapper = ctx.wrapped.get(base_occt) if base_occt else None
            if base_wrapper:
                ctx.inherited_children.setdefault(base_wrapper, []).append(cls)
    for base_wrapper in ctx.inherited_children:
        ctx.inherited_children[base_wrapper].sort(key=lambda c: c.wrapper_name)
    for module in modules:
        for cls in module.classes:
            if cls.skip or cls.name == module.name:
                continue
            if not cls.has_copy_assignment and cls.kind != ClassKind.REF_COUNTED:
                ctx.noncopyable.add(cls.name)
    # Propagate non-copyability through data members: a class whose member is
    # non-copyable is itself non-copyable (implicitly deleted copy semantics).
    # A reference member or a const member deletes copy assignment regardless
    # of the referenced/const-qualified type, and a by-value member whose type
    # is a known non-copyable std value (std::atomic, std::shared_mutex, ...)
    # does too -- mirroring the scan-time structural detection (extract.py).
    changed = True
    while changed:
        changed = False
        for module in modules:
            for cls in module.classes:
                if cls.skip or cls.name == module.name:
                    continue
                if cls.name in ctx.noncopyable:
                    continue
                if cls.kind == ClassKind.REF_COUNTED:
                    continue
                # Pointer and handle members do not delete copy semantics of the
                # enclosing class (copying the pointer/refcount is fine even when
                # the pointee is non-copyable); only by-value members propagate.
                if any(_field_breaks_copy_semantics(f, ctx) for f in cls.fields):
                    ctx.noncopyable.add(cls.name)
                    changed = True
    for module in modules:
        for cls in module.classes:
            if cls.skip or cls.name == module.name:
                continue
            if cls.is_abstract:
                # Abstract classes cannot be instantiated; drop parameterized
                # constructors (the null-storage default ctor stays for
                # Ref.instantiate(), and factory methods take over).
                for ctor in cls.constructors:
                    ctor.skip = True
                    ctor.skip_reason = "abstract class (not instantiable)"
            if cls.kind == ClassKind.REF_COUNTED:
                base = next((b for b in cls.base_classes if b in ctx.wrapped), None)
                if base is not None:
                    ctx.sync_bases.add(cls.wrapper_name)
                ctx.handles.add(cls.wrapper_name)
            elif not _default_constructible(cls):
                ctx.unique_ptr.add(cls.wrapper_name)
            # Mirror _cg()'s "none" storage: EXCEPTION wrappers and pure-static
            # utility classes hold no native object, so they cannot appear as a
            # method parameter or return (the typemap drops such methods).
            if cls.kind == ClassKind.EXCEPTION or (
                    cls.static_methods and not cls.methods and not cls.operators
                    and not cls.fields and not cls.has_any_public_ctor):
                ctx.no_storage.add(cls.wrapper_name)
            if not cls.returnable:
                ctx.no_return.add(cls.wrapper_name)
    # Classes whose heap storage cannot rely on the plain global operator
    # new/delete.  A class (or an inherited base) declaring custom operator
    # new/delete (DEFINE_STANDARD_ALLOC / DEFINE_INC_ALLOC / ...) may expose no
    # plain `operator new(size_t)` at all (allocator-tagged forms hide it), or
    # carry it through a protected/private base where it is inaccessible from
    # outside.  Such unique_ptr-stored wrappers allocate the native via
    # Standard::Allocate placement new and free it through OrtStdAllocDeleter,
    # which never depends on the class's operator new/delete accessibility.
    from .classify import _has_custom_alloc
    stdalloc_by_name = {c.name: c for m in modules for c in m.classes}
    for module in modules:
        for cls in module.classes:
            if cls.skip or cls.name == module.name:
                continue
            if cls.wrapper_name in ctx.unique_ptr \
                    and _has_custom_alloc(cls, stdalloc_by_name, set()):
                ctx.stdalloc.add(cls.wrapper_name)
    for module in modules:
        for enum in module.enums:
            if enum.is_public:
                ctx.enums[enum.name] = enum
    return ctx


def _flatten_sources(cls: ClassDecl, by_name: dict[str, ClassDecl],
                     ctx: tm.TypeContext,
                     _seen: set[str] | None = None) -> list[str]:
    """Unwrapped value-style OCCT bases of ``cls`` (transitively), nearest
    first.  Wrapped bases are walked through (their own wrappers are unrelated
    to the derived wrapper, so they cannot hand methods down) but are not
    themselves sources -- only bases that have no wrapper of their own need
    their methods flattened, because nothing else exposes them on the derived
    wrapper.
    """
    if _seen is None:
        _seen = set()
    if cls.name in _seen:
        return []
    _seen.add(cls.name)
    sources: list[str] = []
    for base in cls.base_classes:
        parent = by_name.get(base)
        if parent is None:
            continue
        if parent.name not in ctx.wrapped and _flattenable_base(parent):
            sources.append(base)
        sources.extend(_flatten_sources(parent, by_name, ctx, _seen))
    return sources


def _flattenable_base(cls: ClassDecl) -> bool:
    """A base whose public methods are safe to bind on every wrapped value
    descendant.  Templates are excluded (methods may depend on substituted
    arguments; synthesized specializations carry their own methods) and so are
    Transient/exception bases, which manage their own wrapper hierarchies."""
    if cls.is_template:
        return False
    if cls.kind in (ClassKind.REF_COUNTED, ClassKind.EXCEPTION):
        return False
    return True


def _flatten_declared_names(cls: ClassDecl) -> set[str]:
    """Every method name the class declares itself (including skipped ones, so
    a flattened base method never collides with a GDScript method name)."""
    return {m.name for m in cls.all_methods}


def _copy_method(method: MethodDecl) -> MethodDecl:
    import copy
    return copy.deepcopy(method)


def flatten_inherited_methods(modules: list[ModuleDecl],
                              ctx: tm.TypeContext) -> None:
    """Bind inherited methods onto wrappers that would otherwise miss them.

    The generator binds only methods declared in each class's own header, so
    anything a wrapper inherits from an unwrapped OCCT base is invisible from
    GDScript -- most notably the result accessors of shape-construction
    builders (BRepBuilderAPI_MakeShape::Shape/IsDone/Modified/Generated/
    IsDeleted and BRepBuilderAPI_Command::IsDone/Check), which is what blocks
    boolean ops and fillet/chamfer/offset/thicken from handing back a result.

    Rather than enumerate those bases (which would miss the next class with
    the same shape), this auto-discovers every unwrapped value-style base in
    the whole codebase and flattens its public methods onto every wrapped
    value descendant that does not already declare the same name, so a derived
    override keeps its own binding.  Applied in-place before codegen and the
    symbol probe run, so flattened methods are bound and audited exactly like
    own-header methods.
    """
    by_name = {decl.name: decl for mod in modules for decl in mod.classes}
    for module in modules:
        for cls in module.classes:
            if cls.skip or cls.name == module.name:
                continue
            if cls.kind == ClassKind.REF_COUNTED:
                # Handle-storage wrappers mirror the OCCT hierarchy through
                # wrapped Transient bases, so nothing is invisible to them.
                continue
            if cls.kind == ClassKind.EXCEPTION:
                continue  # diagnostics-only wrappers, no native storage
            if cls.static_methods and not cls.methods and not cls.operators \
                    and not cls.fields and not cls.has_any_public_ctor:
                continue  # pure-static utility class, no native storage
            declared = _flatten_declared_names(cls)
            for base in _flatten_sources(cls, by_name, ctx):
                base_cls = by_name[base]
                for method in base_cls.methods:
                    if method.name in declared:
                        continue
                    if method.skip or method.is_deleted or method.is_pure_virtual:
                        continue
                    cls.methods.append(_copy_method(method))
                    declared.add(method.name)


def _default_constructible(cls: ClassDecl) -> bool:
    """A wrapper can hold `T _native` iff the class has a public default ctor
    or declares no constructors at all (implicit default ctor).

    `cls.default_constructible` overrides the heuristic when the symbol audit
    proved `T()` ill-formed (the probe compiled `(void)T();` and the compiler
    rejected it); such classes must fall back to unique_ptr storage.
    `cls.has_usable_implicit_default_ctor` is the scan-time structural twin:
    it is False when a base or data member deletes the implicit default ctor
    of a class that declares none (cross-target scans skip the audit, so the
    probe never runs there).
    """
    if cls.default_constructible is not None:
        return cls.default_constructible
    # libclang cannot evaluate abstractness for class templates
    # (cursor.is_abstract_record() is False for them), but the pure-virtual
    # members are still extracted; either signal forbids value storage.
    if cls.is_abstract or cls.has_pure_virtual:
        return False  # cannot value-initialize an abstract type
    return (cls.has_public_default_ctor
            or (not cls.has_any_ctor and cls.has_usable_implicit_default_ctor))


def _uses_stdalloc(cls: ClassDecl, ctx: tm.TypeContext) -> bool:
    """True when the wrapper heap-builds the native on Standard::Allocate
    memory (OrtStdAllocDeleter) because the class's own operator new/delete is
    not usable (allocator-tagged or carried through a protected base)."""
    return cls.wrapper_name in ctx.stdalloc


def _cg(cls: ClassDecl, ctx: tm.TypeContext) -> CgClass:
    base_occt = next((b for b in cls.base_classes if b in ctx.wrapped), None)
    if cls.kind == ClassKind.REF_COUNTED:
        # Only a Transient base shares the handle storage; a value-class base
        # (e.g. NCollection_HArray1<double> derives NCollection_Array1<double>)
        # is value-stored in its own wrapper, so picking it would emit a
        # _sync_base_storage() into a member that does not exist.
        base_occt = next((b for b in cls.base_classes
                          if b in ctx.wrapped and ctx.wrapped[b] in ctx.handles),
                         None)
        wrapper_base = ctx.wrapped.get(base_occt) if base_occt else None
        return CgClass(cls=cls, wrapper_base=wrapper_base,
                       base_occt=base_occt, storage="handle",
                       has_sync=wrapper_base is not None,
                       is_aggregate=cls.name == ctx.module_name)
    if cls.kind == ClassKind.EXCEPTION:
        # Standard_Failure hierarchy: wrapped as a diagnostics-only class chain.
        # No native storage (exceptions never cross the FFI); instance methods
        # read the thread-local last-error state recorded by OCCT_GUARD_CATCH.
        return CgClass(cls=cls,
                       wrapper_base=ctx.wrapped.get(base_occt) if base_occt else None,
                       base_occt=base_occt, storage="none",
                       is_aggregate=cls.name == ctx.module_name)
    storage = "native" if _default_constructible(cls) else "unique_ptr"
    # Pure-static utility classes (e.g. BRep_Tool) hold no native object: their
    # ctors are non-public so storage (and its new/delete requirements) is
    # skipped entirely.
    if cls.static_methods and not cls.methods and not cls.operators \
            and not cls.fields and not cls.has_any_public_ctor:
        return CgClass(cls=cls, wrapper_base=None, base_occt=None, storage="none",
                       is_aggregate=cls.name == ctx.module_name)
    if cls.wrapper_name in ctx.inherited_value:
        return CgClass(cls=cls,
                       wrapper_base=ctx.wrapped.get(base_occt) if base_occt else None,
                       base_occt=base_occt, storage="native",
                       inherited_native=True,
                       is_aggregate=cls.name == ctx.module_name)
    return CgClass(cls=cls, wrapper_base=None, base_occt=None,
                   storage=storage, is_aggregate=cls.name == ctx.module_name)


def _occt_qual(cls: ClassDecl) -> str:
    if cls.cpp_qual_name:
        return cls.cpp_qual_name
    return tm._occt_qual(cls.name)


def _params_decl(method: MethodDecl, ctx: tm.TypeContext,
                 cls=None, is_ctor: bool = False) -> str | None:
    parts = []
    for p in method.parameters:
        conv = tm.cpp_param(p.type, p.name, ctx, cls, is_ctor)
        if conv is None:
            return None
        parts.append(f"{conv.cpp_type} {conv.name}")
    return ", ".join(parts)


def _unique(method: MethodDecl) -> str:
    return get_method_unique_name(method)


def _has_ostream_param(method: MethodDecl) -> bool:
    return any(tm.stream_kind(p.type) == "out" for p in method.parameters)


def _istream_param_name(method: MethodDecl) -> str | None:
    """Safe name of the Standard_IStream& parameter, if the method consumes one."""
    for p in method.parameters:
        if p.type.is_ref and tm.stream_kind(p.type) == "in":
            return tm.safe_param_name(p.name)
    return None


def _uses_streams(cls: ClassDecl) -> bool:
    return any(tm.stream_kind(p.type) is not None
               for m in cls.all_methods for p in m.parameters)


def _uses_fstream(cls: ClassDecl) -> bool:
    """Class has a custom file-I/O body that opens its own std::fstream."""
    for m in cls.all_methods:
        if cls.name == "BRepTools" and m.name in ("Read", "Write"):
            for p in m.parameters:
                if p.type.base_name == "char" and p.type.is_pointer \
                        and p.type.pointee_is_const:
                    return True
    return False


# ---------------------------------------------------------------------------
# Referenced headers / forward declarations
# ---------------------------------------------------------------------------

def _type_occt_header(t: OCCTType, ctx: tm.TypeContext) -> str | None:
    _spec_re = re.compile(r"^([A-Za-z_]\w*)<")

    def header_of(name: str) -> str:
        if name in ctx.occt_headers:
            return ctx.occt_headers[name]
        m = _spec_re.match(name)
        if m:
            return f"{m.group(1)}.hxx"
        return f"{name}.hxx"

    if t.is_handle and t.handle_inner in ctx.wrapped:
        return f"<{header_of(t.handle_inner)}>"
    base = t.base_name
    inner = tm.optional_inner(base)
    if inner is not None:
        base = inner
    if base in ctx.wrapped:
        return f"<{header_of(base)}>"
    if base in ("TCollection_AsciiString", "TCollection_ExtendedString"):
        return f"<{base}.hxx>"
    if base == "std::string" or base.startswith("std::basic_string<char>"):
        return "<string>"
    return None


def _type_wrapper(t: OCCTType, ctx: tm.TypeContext) -> str | None:
    if t.is_handle and t.handle_inner in ctx.wrapped:
        return ctx.wrapped[t.handle_inner]
    base = t.base_name
    inner = tm.optional_inner(base)
    if inner is not None:
        base = inner
    key = tm._wrapped_key(base, ctx)
    if key is None and base.rstrip().endswith("*"):
        key = tm._wrapped_key(base.rstrip()[:-1].rstrip(), ctx)
    if key is not None:
        return ctx.wrapped[key]
    return None


def _referenced_headers(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    headers: set[str] = set()
    if cls.header_file:
        headers.add(f"<{Path(cls.header_file).name}>")
    for base in cls.base_classes:
        if base in ctx.occt_classes:
            headers.add(f"<{ctx.occt_headers.get(base, base + '.hxx')}>")
    for method in cls.all_methods:
        for p in method.parameters:
            h = _type_occt_header(p.type, ctx)
            if h:
                headers.add(h)
        if method.return_type is not None:
            h = _type_occt_header(method.return_type, ctx)
            if h:
                headers.add(h)
    for f in cls.fields:
        h = _type_occt_header(f.type, ctx)
        if h:
            headers.add(h)
    return sorted(headers)


def _referenced_wrappers(cls: ClassDecl, ctx: tm.TypeContext) -> set[str]:
    names: set[str] = set()
    for method in cls.all_methods:
        for p in method.parameters:
            w = _type_wrapper(p.type, ctx)
            if w:
                names.add(w)
            w = tm.base_list_iterator_list_wrapper(p.type, cls, ctx)
            if w:
                names.add(w)
        if method.return_type is not None:
            w = _type_wrapper(method.return_type, ctx)
            if w:
                names.add(w)
    for f in cls.fields:
        w = _type_wrapper(f.type, ctx)
        if w:
            names.add(w)
    return names


def _uses_primitive_wrappers(cls: ClassDecl, ctx: tm.TypeContext) -> bool:
    for method in cls.all_methods:
        for p in method.parameters:
            if p.type.base_name in tm.PRIMITIVE_WRAPPER_MAP:
                return True
        if method.return_type is not None and \
                method.return_type.base_name in tm.PRIMITIVE_WRAPPER_MAP:
            return True
    for f in cls.fields:
        if f.type.base_name in tm.PRIMITIVE_WRAPPER_MAP:
            return True
    return False


def _uses_enum_boxes(cls: ClassDecl, ctx: tm.TypeContext) -> bool:
    """Class has a non-const `Enum&` out-parameter (needs OrtEnumBox classes)."""
    for method in cls.all_methods:
        for p in method.parameters:
            t = p.type
            if t.is_ref and not t.is_const and not t.is_rvalue_ref \
                    and t.is_enum and t.base_name in ctx.enums:
                return True
    return False


def _enum_box_keys_used(modules: list[ModuleDecl],
                        ctx: tm.TypeContext) -> dict[str, object]:
    """Enum name -> EnumDecl for every non-const `Enum&` out-parameter used."""
    used: dict[str, object] = {}
    for module in modules:
        for cls in module.classes:
            if cls.skip:
                continue
            for method in cls.all_methods:
                for p in method.parameters:
                    t = p.type
                    if t.is_ref and not t.is_const and not t.is_rvalue_ref \
                            and t.is_enum and t.base_name in ctx.enums:
                        used[t.base_name] = ctx.enums[t.base_name]
    return used


def _enum_box_class_names(enum_box_decls: dict[str, object]) -> set[str]:
    """Ort<EnumName>Box class names for the given enum out-parameter decls."""
    return {tm._enum_box_class_name(n) for n in enum_box_decls}


def _uses_enums(cls: ClassDecl, ctx: tm.TypeContext) -> bool:
    for method in cls.all_methods:
        for p in method.parameters:
            if p.type.base_name in ctx.enums:
                return True
        if method.return_type is not None and \
                method.return_type.base_name in ctx.enums:
            return True
    for f in cls.fields:
        if f.type.base_name in ctx.enums:
            return True
    return False


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _public_nested_enums(cls: ClassDecl) -> list[object]:
    return [e for e in cls.nested_enums if e.is_public]


def _nested_enum_hpp_lines(cls: ClassDecl) -> list[str]:
    lines: list[str] = []
    for enum in _public_nested_enums(cls):
        path = f"::{cls.name}::{enum.name}"
        lines.append(f"    enum {enum.name} : int64_t {{")
        for v in enum.values:
            lines.append(
                f"        {enum.name}_{v.name} = static_cast<int64_t>({path}::{v.name}),")
        lines.append("    };")
    return lines


def _field_accessor_decls(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    lines: list[str] = []
    for f in cls.fields:
        if not f.is_public or f.skip:
            continue
        snake = to_snake_case(f.name)
        gret = tm.cpp_return(f.type, ctx)
        sconv = tm.cpp_param(f.type, "value", ctx)
        if gret is None or gret.cpp_type == "void":
            continue  # not readable (e.g. void* return)
        lines.append(f"    {gret.cpp_type} _ort_field_get_{snake}() const;")
        if sconv is not None and not f.is_const:
            lines.append(
                f"    void _ort_field_set_{snake}({_field_setter_param(sconv)});")
    return lines


def _field_setter_param(sconv: tm.ParamConv) -> str:
    if sconv.cpp_type == "String":
        return "const ::godot::String& " + sconv.name
    return f"{sconv.cpp_type} {sconv.name}"


def _exception_method_kind(cls: ClassDecl, method: MethodDecl) -> str | None:
    """Diagnostic role of an instance method on an EXCEPTION-kind wrapper.

    Exception wrappers carry no native object; the instance API reads the
    thread-local last-error state recorded by OCCT_GUARD_CATCH.  Returns one
    of "message" / "stack" / "type" / "print", or None if the method has no
    diagnostics mapping (and must be skipped).
    """
    if method.kind == MethodKind.STATIC_METHOD:
        return "static"
    name = method.name
    if name in ("what", "GetMessageString"):
        return "message"
    if name == "GetStackString":
        return "stack"
    if name == "ExceptionType":
        return "type"
    if name == "Print" and _has_ostream_param(method):
        return "print"
    return None


def _exception_method_body(cls: ClassDecl, method: MethodDecl,
                           kind: str, params: str) -> str:
    unique = _unique(method)
    const_suffix = " const" if method.is_const else ""
    if kind == "message":
        body = "return ::godot::String(ort_gd::get_last_error_message());"
    elif kind == "stack":
        body = "return ::godot::String(ort_gd::get_last_error_stack());"
    elif kind == "type":
        body = f'return ::godot::String("{cls.name}");'
    else:  # print: the message, since exceptions have no native to stream.
        body = "return ::godot::String(ort_gd::get_last_error_message());"
    return f"""String {cls.wrapper_name}::{unique}({params}){const_suffix} {{
    {body}
}}"""


# Range-for iterator protocol: begin/end/cbegin/cend return container-internal
# iterator objects that have no GDScript meaning (containers are indexed
# directly).  Their unmappable signature is skipped with a dedicated reason.
_ITERATOR_PROTOCOL_METHODS = frozenset(
    {"begin", "end", "cbegin", "cend", "rbegin", "rend", "crbegin", "crend"})

# NCollection_Vec{2,3,4}<unsigned int/long/long long>::cwiseAbs() is ill-formed:
# its body calls std::abs on the unsigned element type, and no std::abs
# overload exists for unsigned integer types, so the call is ambiguous
# (libc++/libstdc++/MSVC all reject it).  The symbol-audit probe drops it on
# hosts that provide g++ (Linux), but macOS/iOS and Windows CI hosts cannot run
# that probe, so it must be skipped deterministically here -- exactly like the
# IntPolyh_Array::Dump member below.
_VEC_AMBIGUOUS_ABS_RE = re.compile(
    r"^NCollection_Vec[234]<unsigned (?:int|long|long long|__int64)>$")


def _vec_cwise_abs_ambiguous(cls: ClassDecl, method: MethodDecl) -> bool:
    """True when a synthesized NCollection_Vec wrapper's cwiseAbs would not
    compile (ambiguous std::abs on the unsigned element type)."""
    return method.name == "cwiseAbs" and bool(_VEC_AMBIGUOUS_ABS_RE.match(cls.name))


def _first_unmappable(cls: ClassDecl, method: MethodDecl,
                      ctx: tm.TypeContext, is_ctor: bool = False) -> str | None:
    """Spelling of the first type that cannot cross the FFI, or None.

    Mirrors the None return of ``_method_decl_signature`` exactly so skip
    reasons name the actual offending type instead of a blanket label.
    """
    for p in method.parameters:
        conv = tm.cpp_param(p.type, p.name, ctx, cls, is_ctor=is_ctor)
        if conv is None:
            return p.type.spelling
    if method.return_type is None or (method.return_type.is_void
                                      and not method.return_type.is_pointer):
        return None
    has_ostream = _has_ostream_param(method)
    if tm.cpp_return(method.return_type, ctx, has_ostream=has_ostream,
                     cls=cls, stream_in=_istream_param_name(method)) is None:
        return method.return_type.spelling
    return None


def _method_skip_reason(cls: ClassDecl, method: MethodDecl,
                        ctx: tm.TypeContext) -> str:
    """Precise reason a method's signature cannot cross the FFI.

    Mirrors the None return of ``_method_decl_signature``/``_method_body`` so
    the skip registry (autogen/coverage.py) classifies every skip with the
    reason the generator actually emitted, not a blanket ``unmappable type``.
    """
    if cls.kind == ClassKind.EXCEPTION \
            and _exception_method_kind(cls, method) is None:
        return "exception diagnostic method (no native storage)"
    if method.name in _ITERATOR_PROTOCOL_METHODS:
        return "container iterator protocol (begin/end)"
    if cls.name.startswith("IntPolyh_Array") and method.name == "Dump":
        return ("ill-formed instantiation (OCCT member does not compile for "
                "the substituted template args)")
    if _vec_cwise_abs_ambiguous(cls, method):
        return ("ill-formed instantiation (OCCT member does not compile for "
                "the substituted template args)")
    bad = _first_unmappable(cls, method, ctx,
                            is_ctor=method.kind == MethodKind.CONSTRUCTOR)
    if bad is not None:
        return f"unmappable type: {bad}"
    return "unmappable type"


def _method_decl_signature(cls: ClassDecl, method: MethodDecl,
                           ctx: tm.TypeContext) -> str | None:
    if cls.kind == ClassKind.EXCEPTION \
            and _exception_method_kind(cls, method) is None:
        return None
    if cls.name.startswith("IntPolyh_Array") and method.name == "Dump":
        # IntPolyh_Array<T>::Dump() calls (*this)[i].Dump() inside its body, so
        # instantiating it fails when the item type declares no no-argument
        # Dump (IntPolyh_Edge/IntPolyh_Triangle take an int, IntPolyh_PointNormal
        # has none).  Skip it deterministically (the audit probe cannot run on
        # every CI target, and this debug-only dump has no FFI value).
        return None
    if _vec_cwise_abs_ambiguous(cls, method):
        # NCollection_Vec{2,3,4}<unsigned int/long/long long>::cwiseAbs() calls
        # std::abs on the unsigned element type, which is ambiguous on every
        # toolchain.  Skip deterministically (the audit probe cannot run on
        # every CI target; abs of an unsigned value is the identity anyway).
        return None
    params = _params_decl(method, ctx, cls)
    if params is None:
        return None
    has_ostream = _has_ostream_param(method)
    if method.return_type is None or (method.return_type.is_void
                                      and not method.return_type.is_pointer):
        if has_ostream:
            ret = "String"
        else:
            ret = "void"
    else:
        rconv = tm.cpp_return(method.return_type, ctx, has_ostream=has_ostream,
                              cls=cls, stream_in=_istream_param_name(method))
        if rconv is None:
            return None
        ret = rconv.cpp_type
    const_suffix = " const" if method.is_const else ""
    return f"{ret} {_unique(method)}({params}){const_suffix}"


# TopAbs shape-enum names that correspond 1:1 to TopoDS_* value tags.  A
# TopoDS_<tag> stored in an inherited-storage wrapper always reports exactly
# this ShapeType, so the discriminator is derivable from the tag name alone.
_TOPABS_TAG_ENUMS = {
    "TopAbs_COMPOUND", "TopAbs_COMPSOLID", "TopAbs_SOLID", "TopAbs_SHELL",
    "TopAbs_FACE", "TopAbs_WIRE", "TopAbs_EDGE", "TopAbs_VERTEX",
    "TopAbs_SHAPE",
}


def _inherited_cast_name(child: ClassDecl) -> str:
    """GDScript-visible downcast factory name, e.g. ``cast_vertex``."""
    tag = child.name.rsplit("_", 1)[-1]
    return f"cast_{tag.lower()}"


def _inherited_cast_discriminator(child: ClassDecl) -> str | None:
    """TopAbs_* enum value reported by this tag's ShapeType, or None."""
    tag = child.name.rsplit("_", 1)[-1]
    disc = f"TopAbs_{tag.upper()}"
    return disc if disc in _TOPABS_TAG_ENUMS else None


def _inherited_children(cls: ClassDecl, ctx: tm.TypeContext) -> list[ClassDecl]:
    """Sorted inherited-storage children of cls's wrapper (for cast factories)."""
    return ctx.inherited_children.get(cls.wrapper_name, [])


def _inherited_cast_decls(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    """Static downcast factory declarations emitted on the base wrapper class."""
    children = _inherited_children(cls, ctx)
    if not children:
        return []
    decls = []
    for child in children:
        decls.append(f"    static Ref<{child.wrapper_name}> "
                     f"{_inherited_cast_name(child)}(Ref<{cls.wrapper_name}> S);")
    if all(_inherited_cast_discriminator(c) is not None for c in children):
        decls.append(f"    static Ref<{cls.wrapper_name}> "
                     f"cast(Ref<{cls.wrapper_name}> S);")
    return decls


def _inherited_cast_bodies(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    """Implementations of the downcast factories for the base wrapper class."""
    children = _inherited_children(cls, ctx)
    if not children:
        return []
    bodies: list[str] = []
    for child in children:
        tag = child.name.rsplit("_", 1)[-1]
        pkg = child.name.rsplit("_", 1)[0]
        disc = _inherited_cast_discriminator(child)
        guard = (f"        if (S.is_null() || S->_native.IsNull()"
                 f" || S->_native.ShapeType() != {disc}) {{") if disc else \
                (f"        if (S.is_null() || S->_native.IsNull()) {{")
        bodies.append(f"""Ref<{child.wrapper_name}> {cls.wrapper_name}::{_inherited_cast_name(child)}(Ref<{cls.wrapper_name}> S) {{
    try {{
{guard}
            return Ref<{child.wrapper_name}>();
        }}
        Ref<{child.wrapper_name}> wrapper; wrapper.instantiate();
        wrapper->_native_ref() = ::{pkg}::{tag}(S->_native);
        return wrapper;
    }} ORT_GUARD_CATCH({{}});
}}""")
    if all(_inherited_cast_discriminator(c) is not None for c in children):
        cases = "\n".join(
            f"        case {_inherited_cast_discriminator(c)}:\n"
            f"            return {_inherited_cast_name(c)}(S);"
            for c in children)
        bodies.append(f"""Ref<{cls.wrapper_name}> {cls.wrapper_name}::cast(Ref<{cls.wrapper_name}> S) {{
    try {{
        if (S.is_null() || S->_native.IsNull()) {{
            return Ref<{cls.wrapper_name}>();
        }}
        switch (S->_native.ShapeType()) {{
{cases}
        default:
            return S;
        }}
    }} ORT_GUARD_CATCH({{}});
}}""")
    return bodies


def generate_class_hpp(cls: ClassDecl, ctx: tm.TypeContext) -> str:
    _skip_ambiguous_ctor_calls(cls, ctx)
    cg = _cg(cls, ctx)
    base = cg.wrapper_base or "RefCounted"
    out: list[str] = []
    out.append(f"// Auto-generated wrapper for {cls.name} -- DO NOT EDIT")
    out.append("#pragma once")
    out.append("")
    out.append(GODOT_INCLUDES)
    out.append("")
    out.append(GCC_CHANGES)
    out.append("")
    refs = _referenced_headers(cls, ctx)
    own = f"<{Path(cls.header_file).name}>" if cls.header_file else None
    # The class's own header is not self-contained: the scan needed these
    # pre-includes before it would parse, so the wrapper must include them
    # *before* the class header (extra_occt_includes), then the referenced
    # headers.  Dedupe them out of `refs` so they are not emitted again after.
    extras = list(cls.extra_occt_includes)
    # Referenced OCCT headers (e.g. a template element type pulled in by a
    # synthesized specialization) may likewise be non-self-contained; hoist
    # their scanned extra includes before every referenced header so the parse
    # succeeds in the synth wrapper too.
    for h in refs:
        hdr = h.strip("<>").removesuffix(".hxx")
        for e in ctx.occt_extras.get(hdr, []):
            if e not in extras:
                extras.append(e)
    # Emit each non-self-contained header only after its own extra includes
    # (recursively), so e.g. HLRAlgo_PolyHidingData.hxx -- which uses gp_XYZ
    # without including it -- is preceded by gp_XYZ.hxx.
    ordered: list[str] = []
    seen: set[str] = set()
    for e in extras:
        if e in seen:
            continue
        seen.add(e)
        for dep in ctx.occt_extras.get(e.removesuffix(".hxx"), []):
            if dep not in seen and dep != e:
                ordered.append(dep)
                seen.add(dep)
        ordered.append(e)
    extras = [f"<{e}>" for e in ordered if e and f"<{e}>" != own]
    refs = [h for h in refs if h not in extras]
    if own:
        refs = [own] + [h for h in refs if h != own]
    for h in extras + refs:
        out.append(f"#include {h}")
    if cg.wrapper_base:
        out.append(f'#include "{cg.wrapper_base}.hpp"')
    if cg.storage == "unique_ptr":
        out.append("#include <memory>")
        if _uses_stdalloc(cls, ctx):
            out.append('#include "OrtMemory.hpp"')
    if _uses_primitive_wrappers(cls, ctx) or _uses_enum_boxes(cls, ctx):
        out.append('#include "OrtPrimitiveWrappers.hpp"')
    if _uses_streams(cls):
        out.append('#include "OrtCallableStreams.hpp"')
    if _uses_enums(cls, ctx):
        out.append('#include "OrtEnums.hpp"')
    out.append("")
    if _uses_enums(cls, ctx):
        out.append("")
    out.append("namespace godot {")
    out.append("")
    fwd = sorted(_referenced_wrappers(cls, ctx)
                 - {cls.wrapper_name}
                 - {base} if cg.wrapper_base
                 else _referenced_wrappers(cls, ctx) - {cls.wrapper_name})
    cast_children = _inherited_children(cls, ctx)
    for child in cast_children:
        if child.wrapper_name not in fwd:
            fwd.append(child.wrapper_name)
            fwd.sort()
    if fwd:
        out.append("// Forward declarations")
        for w in fwd:
            out.append(f"class {w};")
        out.append("")
    out.append("")
    out.append(f"class {cls.wrapper_name} : public {base} {{")
    out.append(f"    GDCLASS({cls.wrapper_name}, {base})")
    out.append("")
    out.append("public:")
    out.extend(_nested_enum_hpp_lines(cls))
    if _public_nested_enums(cls):
        out.append("")
    qual = _occt_qual(cls)
    if cg.storage == "handle":
        out.append(f"    opencascade::handle<{qual}> _handle;")
    elif cg.storage == "unique_ptr":
        if _uses_stdalloc(cls, ctx):
            out.append(f"    std::unique_ptr<{qual}, "
                       f"ort_gd::OrtStdAllocDeleter<{qual}>> _native = nullptr;")
        else:
            out.append(f"    std::unique_ptr<{qual}> _native = nullptr;")
    elif cg.inherited_native:
        out.append(f"    {qual}& _native_ref() {{ return *static_cast<{qual}*>(&this->_native); }}")
        out.append(f"    const {qual}& _native_ref() const {{ return *static_cast<const {qual}*>(&this->_native); }}")
    elif cg.storage == "native":
        out.append(f"    {qual} _native;")
    out.append("")
    out.append("    static void _bind_methods();")
    cast_decls = _inherited_cast_decls(cls, ctx)
    if cast_decls:
        out.append("")
        out.extend(cast_decls)
    out.append("")
    out.append(f"    {cls.wrapper_name}();")
    if cg.has_sync:
        out.append("")
        out.append("    void _sync_base_storage();")
    out.append("")
    group_overloads(cls)
    emitted = False
    for ctor in cls.constructors:
        if cls.kind == ClassKind.EXCEPTION:
            # Exceptions are diagnostics-only: they are produced by caught
            # OCCT failures, never constructed from GDScript.
            ctor.skip = True
            ctor.skip_reason = "exception class constructor (diagnostics-only)"
            continue
        if _default_ctor(ctor):
            ctor.skip = True
            ctor.skip_reason = "default constructor (native default-construction)"
            continue
        if ctor.skip:
            continue
        params = _params_decl(ctor, ctx, cls, is_ctor=True)
        if params is None:
            ctor.skip = True
            ctor.skip_reason = _method_skip_reason(cls, ctor, ctx)
            continue
        out.append(f"    static Ref<{cls.wrapper_name}> {_unique(ctor)}({params});")
        out.append("")
        emitted = True
    for m in cls.methods + cls.operators:
        if m.skip:
            continue
        sig = _method_decl_signature(cls, m, ctx)
        if sig is None:
            m.skip = True
            m.skip_reason = _method_skip_reason(cls, m, ctx)
            continue
        out.append(f"    {sig};")
        emitted = True
    if cls.static_methods:
        sigs = []
        for m in cls.static_methods:
            if m.skip:
                continue
            sig = _method_decl_signature(cls, m, ctx)
            if sig is None:
                m.skip = True
                m.skip_reason = _method_skip_reason(cls, m, ctx)
                continue
            sigs.append(f"    static {sig};")
        if sigs:
            if cls.methods or cls.operators:
                out.append("")
            out.extend(sigs)
            emitted = True
    if cg.storage == "handle" and not any(m.name == "is_null" for m in cls.all_methods):
        out.append("    bool is_null() const;")
        emitted = True
    field_decls = _field_accessor_decls(cls, ctx)
    if field_decls:
        out.extend(field_decls)
        emitted = True
    if emitted:
        out.append("")
    out.append("};")
    out.append("")
    out.append("} // namespace godot")
    if _public_nested_enums(cls):
        out.append("")
        for enum in _public_nested_enums(cls):
            out.append(f"VARIANT_ENUM_CAST({cls.wrapper_name}::{enum.name});")
    return "\n".join(out) + ("\n\n" if not _public_nested_enums(cls) else "\n")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _occt_call(cls: ClassDecl, method: MethodDecl, args: str,
               ctx: tm.TypeContext) -> str:
    if method.kind == MethodKind.STATIC_METHOD:
        return f"{_occt_qual(cls)}::{method.name}({args})"
    if method.operator_type is not None:
        op = OPERATOR_SPELLING.get(method.operator_type.value,
                                   method.operator_type.value)
        if cls.kind == ClassKind.REF_COUNTED:
            return f"_handle.get()->operator{op}({args})"
        if _cg(cls, ctx).inherited_native:
            return f"_native_ref().operator{op}({args})"
        if _cg(cls, ctx).storage == "unique_ptr":
            return f"_native->operator{op}({args})"
        return f"_native.operator{op}({args})"
    if cls.kind == ClassKind.REF_COUNTED:
        return f"_handle.get()->{method.name}({args})"
    if _cg(cls, ctx).storage == "unique_ptr":
        return f"_native->{method.name}({args})"
    if _cg(cls, ctx).inherited_native:
        return f"_native_ref().{method.name}({args})"
    return f"_native.{method.name}({args})"


def _custom_method_body(cls: ClassDecl, method: MethodDecl,
                        ctx: tm.TypeContext) -> str | None:
    """Hand-written bodies for signatures the generic FFI cannot express.

    BRepTools' file-based Read/Write overloads take a ``const char*`` path and
    open their own std::fstream; TopTools_ShapeSet then runs an
    imbue(std::locale::classic())/restore dance on that stream.  Because
    Godot's binary interposes its own std::locale symbols, that dance drains
    Godot's global locale's reference count and eventually frees it through the
    wrong allocator.  We open the fstream ourselves, move it onto the classic
    locale (leaking the replaced locale so its reference is never dropped) and
    use the stream overload instead, so OCCT's locale dance stays inside a
    single, consistent libstdc++ universe (see OrtCallableStreams.hpp).
    """
    if cls.name == "Standard_Dump" and method.name == "AddValuesSeparator":
        # OCCT's AddValuesSeparator only writes ", " when tellp() > 0, i.e.
        # when called mid-stream between already-dumped values.  The wrapper
        # hands it a fresh OrtCallableOStream every call (the sink Callable is
        # the only durable state), so that check would never trigger and the
        # API would always return an empty string.  Write the separator
        # unconditionally instead.
        params = _params_decl(method, ctx, cls)
        if params is None:
            return None
        return f"""String {cls.wrapper_name}::add_values_separator({params}) {{
    try {{
        ort_gd::OrtCallableOStream ort_os(theOStream);
        ort_os.stream() << ", ";
        ::godot::String ort_text = ::godot::String::utf8(ort_os.str().c_str());
        ort_os.stream().flush();
        return ort_text;
    }} ORT_GUARD_CATCH({{}});
}}"""
    if cls.name != "BRepTools" or method.name not in ("Read", "Write"):
        return None
    file_param = next((p for p in method.parameters
                       if p.type.base_name == "char" and p.type.is_pointer
                       and p.type.pointee_is_const), None)
    if file_param is None:
        return None
    unique = _unique(method)
    params = _params_decl(method, ctx, cls)
    if params is None:
        return None
    arg_exprs = []
    for p in method.parameters:
        if p is file_param:
            arg_exprs.append("ort_fs")
            continue
        conv = tm.cpp_param(p.type, p.name, ctx, cls)
        if conv is None:
            return None
        if conv.prelude:
            return None
        arg_exprs.append(conv.call_expr)
    call = _occt_call(cls, method, ", ".join(arg_exprs), ctx)
    stype = "std::ifstream" if method.name == "Read" else "std::ofstream"
    const_suffix = " const" if method.is_const else ""
    body_lines = [
        f"        {stype} ort_fs({file_param.name}.utf8().get_data());",
        "        if (!ort_fs)",
        "            return false;",
        "        new std::locale(ort_fs.imbue(std::locale::classic()));",
        f"        {call};",
        "        return ort_fs.good();",
    ]
    return f"""bool {cls.wrapper_name}::{unique}({params}){const_suffix} {{
    try {{
{chr(10).join(body_lines)}
    }} ORT_GUARD_CATCH({{}});
}}"""


def _method_body(cls: ClassDecl, method: MethodDecl,
                 ctx: tm.TypeContext) -> str | None:
    unique = _unique(method)
    params = _params_decl(method, ctx, cls)
    if params is None:
        return None
    custom = _custom_method_body(cls, method, ctx)
    if custom is not None:
        return custom
    if cls.kind == ClassKind.EXCEPTION:
        kind = _exception_method_kind(cls, method)
        if kind is None:
            return None
        if kind == "static":
            pass  # static methods take the normal native-call path
        else:
            return _exception_method_body(cls, method, kind, params)
    preludes: list[str] = []
    postludes: list[str] = []
    arg_exprs: list[str] = []
    for p in method.parameters:
        conv = tm.cpp_param(p.type, p.name, ctx, cls)
        if conv is None:
            return None
        if conv.prelude:
            preludes.append(conv.prelude)
        if conv.postlude:
            postludes.append(conv.postlude)
        arg_exprs.append(conv.call_expr)
    args = ", ".join(arg_exprs)
    call = _occt_call(cls, method, args, ctx)
    const_suffix = " const" if method.is_const else ""
    ret_is_void = method.return_type is None or (
        method.return_type.is_void and not method.return_type.is_pointer)
    has_ostream = _has_ostream_param(method)

    if ret_is_void and not has_ostream:
        rconv = tm.RetConv(cpp_type="void", body="{call};")
    else:
        rconv = tm.cpp_return(method.return_type, ctx, has_ostream=has_ostream,
                              cls=cls, stream_in=_istream_param_name(method))
        if rconv is None:
            return None

    guard = ""
    if method.kind != MethodKind.STATIC_METHOD:
        if _cg(cls, ctx).storage == "handle":
            if rconv.cpp_type == "void":
                guard = "        if (!_handle) return;\n"
            else:
                guard = (f"        if (!_handle) return "
                         f"{tm.default_value(rconv.cpp_type)};\n")
        elif _cg(cls, ctx).storage == "unique_ptr":
            if rconv.cpp_type == "void":
                guard = "        if (!_native) return;\n"
            else:
                guard = (f"        if (!_native) return "
                         f"{tm.default_value(rconv.cpp_type)};\n")

    body_lines = [f"        {p}" for p in preludes]
    if postludes:
        body_lines.append(f"        {_inject_postludes(rconv.body, call, postludes)}")
    else:
        body_lines.append(f"        {rconv.body.replace('{call}', call)}")
    catch = ("ORT_GUARD_CATCH_VOID();" if rconv.cpp_type == "void"
             else "ORT_GUARD_CATCH({});")
    return f"""{rconv.cpp_type} {cls.wrapper_name}::{unique}({params}){const_suffix} {{
    try {{
{guard}{chr(10).join(body_lines)}
    }} {catch}
}}"""


def _inject_postludes(body: str, call: str, postludes: list[str]) -> str:
    """Return `body` (a return-body template) with `postludes` run between the
    OCCT call and its return.

    Param postludes write out-parameters back into the caller's argument after
    the call (char*& strings, T*& pointer boxes).  They must run after the
    call but before the value is returned, so the template body is rewritten
    into capture-then-return form when the call is not already a standalone
    statement.
    """
    post = "\n        ".join(postludes)
    stripped = body.lstrip()
    if stripped == "{call};":
        return f"{call};\n        {post}"
    if stripped.startswith("return {call};"):
        return (f"auto ort_post_value = {call};\n"
                f"        {post}\n"
                "        return ort_post_value;")
    m = re.match(r"^(\s*)(auto(?:&)?)\s+(\w+)\s*=\s*\{call\};", body, re.S)
    if m:
        return (f"{m.group(2)} {m.group(3)} = {call};\n"
                f"        {post}"
                f"{body[m.end():]}")
    # Any other shape embeds {call} inside the returned value: capture it into
    # ort_post_value, run the postludes, then evaluate the body against it.
    return (f"auto ort_post_value = {call};\n"
            f"        {post}\n"
            f"        {body.replace('{call}', 'ort_post_value').lstrip()}")


def _ctor_body(cls: ClassDecl, ctor: MethodDecl, ctx: tm.TypeContext) -> str:
    unique = _unique(ctor)
    params = _params_decl(ctor, ctx, cls, is_ctor=True)
    if params is None:
        return ""  # unmappable param; caller marks skip
    preludes: list[str] = []
    postludes: list[str] = []
    arg_exprs: list[str] = []
    for p in ctor.parameters:
        conv = tm.cpp_param(p.type, p.name, ctx, cls, is_ctor=True)
        if conv is None:
            return ""
        if conv.prelude:
            preludes.append(conv.prelude)
        if conv.postlude:
            postludes.append(conv.postlude)
        arg_exprs.append(conv.call_expr)
    args = ", ".join(arg_exprs)
    pre = "\n".join(f"        {p}" for p in preludes) + "\n" if preludes else ""
    post = "\n".join(f"        {p}" for p in postludes)
    cg = _cg(cls, ctx)
    if cg.storage == "unique_ptr":
        new_expr = f"ref->_native = std::make_unique<{_occt_qual(cls)}>({args});"
        tail = f"\n{post}" if postludes else ""
        return f"""Ref<{cls.wrapper_name}> {cls.wrapper_name}::{unique}({params}) {{
    try {{
        Ref<{cls.wrapper_name}> ref; ref.instantiate();
        ort_gd::clear_last_error();
{pre}        {new_expr}{tail}
        return ref;
    }} ORT_GUARD_CATCH({{}});
}}"""
    if cg.storage == "handle":
        sync = "\n        ref->_sync_base_storage();" if cg.has_sync else ""
        tail = f"\n{post}" if postludes else ""
        return f"""Ref<{cls.wrapper_name}> {cls.wrapper_name}::{unique}({params}) {{
    try {{
        Ref<{cls.wrapper_name}> ref; ref.instantiate();
        ort_gd::clear_last_error();
{pre}        ref->_handle = new {_occt_qual(cls)}({args});
{sync}{tail}
        return ref;
    }} ORT_GUARD_CATCH({{}});
}}"""
    if cg.inherited_native:
        tail = f"\n{post}" if postludes else ""
        return f"""Ref<{cls.wrapper_name}> {cls.wrapper_name}::{unique}({params}) {{
    try {{
        Ref<{cls.wrapper_name}> ref; ref.instantiate();
        ort_gd::clear_last_error();
{pre}        ref->_native_ref() = {_occt_qual(cls)}({args});
{tail}
        return ref;
    }} ORT_GUARD_CATCH({{}});
}}"""
    tail = f"\n{post}" if postludes else ""
    return f"""Ref<{cls.wrapper_name}> {cls.wrapper_name}::{unique}({params}) {{
    try {{
        Ref<{cls.wrapper_name}> ref; ref.instantiate();
        ort_gd::clear_last_error();
{pre}        new (&ref->_native) {_occt_qual(cls)}({args});
{tail}
        return ref;
    }} ORT_GUARD_CATCH({{}});
}}"""


def _plain_ctor_body(cls: ClassDecl, ctx: tm.TypeContext) -> str:
    cg = _cg(cls, ctx)
    base_init = f"{cg.wrapper_base}()" if cg.wrapper_base else "RefCounted()"
    if cg.storage == "handle":
        if cls.has_public_default_ctor and not cls.is_abstract:
            sync = "\n        _sync_base_storage();" if cg.has_sync else ""
            return f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} {{
    try {{
        _handle = new {_occt_qual(cls)}();
{sync}
    }} ORT_GUARD_CATCH_CTOR()
}}"""
        return f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} {{
    // No default constructor -- _handle is null; use factory methods
}}"""
    if cg.storage == "unique_ptr":
        return f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} {{
    // No default constructor -- use factory methods
}}"""
    if cg.storage == "none":
        return f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} {{
}}"""
    if cg.inherited_native:
        return f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} {{
}}"""
    if cls.name == "Env":
        return f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} , _native(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime") {{
}}"""
    return (f"""{cls.wrapper_name}::{cls.wrapper_name}() : {base_init} , _native() {{
}}""")


def _sync_body(cls: ClassDecl, ctx: tm.TypeContext) -> str:
    cg = _cg(cls, ctx)
    if not cg.has_sync:
        return ""
    lines = [f"    {cg.wrapper_base}::_handle = opencascade::handle<::{cg.base_occt}>"
             f"(static_cast<::{cg.base_occt}*>(_handle.get()));"]
    # Propagate up the whole inheritance chain: the direct base's own
    # _sync_base_storage() copies its (just-set) handle to the next level, so a
    # method taking e.g. Ref<OrtGeomSurface> sees a valid handle even when the
    # concrete wrapper is OrtGeomBSplineSurface (two levels below).
    if cg.wrapper_base in ctx.sync_bases:
        lines.append(f"    {cg.wrapper_base}::_sync_base_storage();")
    return "\n".join(lines)


def _field_accessor_bodies(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    out: list[str] = []
    cg = _cg(cls, ctx)
    if cg.storage == "handle":
        target = "(*_handle)"
        get_guard_tmpl = "if (!_handle) return {dflt};"
        set_guard = "if (!_handle) return;"
    elif cg.storage == "unique_ptr":
        target = "(*_native)"
        get_guard_tmpl = "if (!_native) return {dflt};"
        set_guard = "if (!_native) return;"
    else:
        target = "_native_ref()" if cg.inherited_native else "_native"
        get_guard_tmpl, set_guard = None, None
    for f in cls.fields:
        if not f.is_public or f.skip:
            continue
        snake = to_snake_case(f.name)
        gret = tm.cpp_return(f.type, ctx)
        sconv = tm.cpp_param(f.type, "value", ctx)
        if gret is None or gret.cpp_type == "void":
            continue
        get_body = gret.body.replace("{call}", f"{target}.{f.name}")
        if get_guard_tmpl is not None:
            guard = get_guard_tmpl.format(dflt=tm.default_value(gret.cpp_type))
            get_body = f"    {guard}\n    {get_body}"
        out.append(f"""{gret.cpp_type} {cls.wrapper_name}::_ort_field_get_{snake}() const {{
{get_body}
}}""")
        out.append("")
        if sconv is not None and not f.is_const:
            pre = f"\n    {sconv.prelude}" if sconv.prelude else ""
            set_body = f"{target}.{f.name} = {sconv.call_expr};"
            if set_guard is not None:
                set_body = f"    {set_guard}\n    {set_body}"
            out.append(f"""void {cls.wrapper_name}::_ort_field_set_{snake}({_field_setter_param(sconv)}) {{{pre}
{set_body}
}}""")
            out.append("")
    return out


def _default_ctor(ctor: MethodDecl) -> bool:
    """True for a no-argument constructor (native default-construction)."""
    return len(ctor.parameters) == 0


def _skip_ambiguous_ctor_calls(cls: ClassDecl, ctx: tm.TypeContext) -> None:
    """Skip ctor bindings whose emitted ``new T(args...)`` is ambiguous.

    A call passing N args is ambiguous when another ctor of arity M > N has
    the same first-N parameter *types* and defaulted trailing params: both are
    viable with identical conversion sequences, and default arguments do not
    participate in overload-resolution tie-breaking (e.g. an ``IntPolyh_Array``
    ``(int)`` binding colliding with ``(int, int = 256)``).  The shorter
    binding is dropped; the longest binding in a collision chain stays
    unambiguous.

    Comparison uses the canonical C++ type of each parameter, not the mapped
    GDScript type: ``const char16_t*`` and ``const char*`` both map to
    ``String`` but are distinct C++ types, so a ``(const char16_t*)`` ctor is
    *not* ambiguous with a ``(const char*, bool = false)`` one.  Idempotent:
    safe to call from both the hpp and cpp paths.
    """
    bound: list[tuple[MethodDecl, list[str]]] = []
    for ctor in cls.constructors:
        if ctor.skip or _default_ctor(ctor):
            continue
        types: list[str] = []
        for p in ctor.parameters:
            conv = tm.cpp_param(p.type, p.name, ctx, cls)
            if conv is None:
                types = []
                break
            types.append(_type_to_string(p.type))
        if types:
            bound.append((ctor, types))
    for ctor, types in bound:
        for other, other_types in bound:
            if other is ctor or len(other_types) <= len(types):
                continue
            if other_types[: len(types)] != types:
                continue
            if all(p.default_value is not None
                   for p in other.parameters[len(types):]):
                ctor.skip = True
                ctor.skip_reason = ("ambiguous constructor call "
                                    "(collides with a defaulted-argument "
                                    "constructor overload)")
                break


_NUMERIC_DEFVAL_TYPES = {
    "bool", "char", "unsigned char", "int", "long", "long long",
    "unsigned long", "unsigned long long", "char16_t", "float", "double",
}

# Mapped cpp types that Variant constructs unambiguously from; defaults for
# numeric params are cast to these to avoid overload ambiguity on targets where
# the OCCT type and its mapping differ (e.g. Standard_Size = unsigned long on
# LP64 Apple vs mapped uint64_t = unsigned long long).
PRIMITIVE_MAP_CXX_TYPES = frozenset(
    {"bool", "int8_t", "int16_t", "int32_t", "int64_t",
     "uint8_t", "uint16_t", "uint32_t", "uint64_t", "float", "double"})

_CXX_KEYWORDS = frozenset({"true", "false", "nullptr"})

# A conservative grammar for the numeric/bool literals godot-cpp's `DEFVAL`
# accepts: optional sign, decimal/hex/binary/octal integers, floats with a
# fractional dot and/or exponent, char literals and the bool keywords.  The
# default-argument recovery (extract._param_default) returns raw *source* text,
# which can be a bare identifier that only resolves as a macro in the OCCT
# header's own context (e.g. `Update`); such a token would not compile inside
# ClassDB::bind_method's _bind_methods, so anything outside this grammar (when
# not a `::`-qualified constant) is dropped instead of emitted.
_NUMERIC_LITERAL_RE = re.compile(
    r"^(?:[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*"
    r"|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?(?:[uUlLfF]+)?"
    r"|'([^'\\]|\\[^'])'|true|false)$")

# OCCT's own bool constants; they resolve inside every wrapper TU, which
# includes the OCCT headers.
_OCCT_BOOL_CONSTS = frozenset({"Standard_True", "Standard_False"})


def _valid_defval_literal(dflt: str) -> bool:
    """True if `dflt` (post ``_clean_numeric_default``) can appear in DEFVAL.

    `::`-qualified constants (class statics, ``Precision::...``) resolve from
    the wrapper's _bind_methods; anything else must be a plain literal.
    """
    if "::" in dflt:
        return True
    return dflt in _OCCT_BOOL_CONSTS or _NUMERIC_LITERAL_RE.match(dflt) is not None


def _qualify_default(cls: ClassDecl, dflt: str) -> str:
    """Qualify a bare-identifier default with its owning OCCT class so it
    resolves from the wrapper's _bind_methods.

    Only identifiers that name a static member of the class itself are
    qualified; global enumerators (e.g. Graphic3d_ZLayerId_UNKNOWN) and
    namespace-qualified expressions are left untouched.
    """
    if dflt.isidentifier() and dflt not in _CXX_KEYWORDS:
        if dflt in cls.static_constants:
            return f"{cls.name}::{dflt}"
    return dflt


def _defval_suffix(cls: ClassDecl, method: MethodDecl, ctx: tm.TypeContext) -> str:
    """DEFVAL(...) clauses for trailing parameters that carry C++ defaults.

    Only defaults that are expressible as a godot-cpp `Variant` (numeric
    primitives, or the synthetic ``Callable()`` sink for out-stream params) are
    emitted; object/enum/string defaults cannot be forwarded through `DEFVAL`,
    so the clause is dropped at the first such parameter.
    """
    parts = []
    for p in reversed(method.parameters):
        if tm.stream_kind(p.type) == "out":
            parts.append("DEFVAL(Callable())")
            continue
        if p.default_value is None:
            break
        if p.type.is_enum or p.type.base_name not in _NUMERIC_DEFVAL_TYPES:
            break
        cleaned = _clean_numeric_default(_qualify_default(cls, p.default_value))
        if not _valid_defval_literal(cleaned):
            # The recovered default is not a self-contained literal (a bare
            # identifier that only made sense in the OCCT header's context).
            # Dropping the clause is compile-safe: the trailing argument simply
            # becomes required.
            break
        # Cast the default to the mapped cpp parameter type.  OCCT's Standard_Size
        # is `unsigned long` on LP64 targets, but the wrapper's parameter is the
        # mapped `uint64_t` (`unsigned long long` on Apple), so a bare
        # `DEFVAL(NCollection_AccAllocator::DefaultBlockSize)` is ambiguous
        # between the int64_t/uint64_t Variant constructors (arm64-ios C2668-
        # style error).  An explicit cast selects exactly one Variant ctor while
        # keeping the same value.
        conv = tm.cpp_param(p.type, p.name, ctx, cls)
        if conv is not None and conv.cpp_type in PRIMITIVE_MAP_CXX_TYPES:
            cleaned = f"static_cast<{conv.cpp_type}>({cleaned})"
        parts.append(f"DEFVAL({cleaned})")
    if not parts:
        return ""
    return ", " + ", ".join(reversed(parts))


def _clean_numeric_default(dflt: str) -> str:
    """Strip a functional-cast type from a numeric default.

    Substituted template defaults read e.g. ``unsigned long(0)`` (from ``T()``
    with ``T`` a multi-word primitive); ``DEFVAL(unsigned long(0))`` does not
    parse, so unwrap the inner literal.  ``true``/``false``/plain literals pass
    through untouched.
    """
    if dflt.isidentifier():
        return dflt
    m = re.match(r"^[A-Za-z_]\w*(?: [A-Za-z_]\w*)*\((.*)\)$", dflt, re.S)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return dflt


def _bind_arg_names(method: MethodDecl, ctx: tm.TypeContext,
                    cls=None) -> str:
    """D_METHOD argument names; callable stream params are exposed by name."""
    names = []
    for p in method.parameters:
        conv = tm.cpp_param(p.type, p.name, ctx, cls)
        if conv is None:
            continue
        names.append(f'"{p.name}"')
    return ", ".join(names)


def _property_getter_candidates(cls: ClassDecl) -> list[MethodDecl]:
    """Zero-arg const instance getters eligible to become Godot properties.

    Restricted to non-overloaded methods whose stable GDScript name equals their
    plain snake_case name: a name that needed a keyword/reserved guard stays
    bound-only (the property would expose e.g. ``reference_`` for
    CDM_Document::Reference).  Overloaded getters are excluded because their
    unique name carries a signature hash and is not a stable property name.
    """
    out: list[MethodDecl] = []
    for m in cls.methods:
        if m.skip or m.is_static or m.is_overload:
            continue
        if m.kind != MethodKind.METHOD or not m.is_const:
            continue
        if len(m.parameters) != 0:
            continue
        if get_method_unique_name(m) != to_snake_case(m.name):
            continue
        out.append(m)
    return out


def _method_property_entries(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    """add_property lines synthesised from OCCT getter/setter method pairs.

    Only hierarchy roots (no wrapped OCCT base) qualify: an accessor overridden
    in a derived class would otherwise re-register the base's property name in
    ClassDB.  BUILDER/EXCEPTION wrappers are excluded (one-shot construction
    state machines / diagnostics-only).

    A const zero-arg getter paired with a one-arg ``set_<name>`` whose Godot
    type matches the getter's return becomes a read-write property; unpaired
    getters become read-only.  ``GetX``/``SetX`` pairs (setter ``set_x``) match
    the ``x`` getter via the ``get_``-stripped stem.  Property names are deduped
    against field-derived properties and each other.
    """
    cg = _cg(cls, ctx)
    if cg.wrapper_base is not None or cg.storage == "none":
        return []
    if cls.kind in (ClassKind.BUILDER, ClassKind.EXCEPTION):
        return []
    candidates = _property_getter_candidates(cls)
    if not candidates:
        return []
    methods = [m for m in cls.methods if not m.skip and not m.is_static
               and m.kind == MethodKind.METHOD]
    by_snake: dict[str, MethodDecl] = {to_snake_case(m.name): m for m in methods}
    field_names = {to_snake_case(f.name) for f in cls.fields
                   if f.is_public and not f.skip}
    prop_names: set[str] = set(field_names)
    entries: list[tuple[str, str, str]] = []  # (name, gd_type, setter_method)
    for getter in candidates:
        rconv = tm.cpp_return(getter.return_type, ctx, cls=cls)
        if rconv is None or rconv.gd_type == "NIL":
            continue
        name = to_snake_case(getter.name)
        if name in prop_names:
            continue
        prop_names.add(name)
        setter: MethodDecl | None = None
        stems = (name,) if not name.startswith("get_") else (name, name[4:])
        for stem in stems:
            cand = by_snake.get(f"set_{stem}")
            if cand is None or cand.is_overload or cand.is_const:
                continue
            if len(cand.parameters) != 1:
                continue
            pconv = tm.cpp_param(cand.parameters[0].type, "v", ctx, cls=cls)
            if pconv is not None and pconv.gd_type == rconv.gd_type:
                setter = cand
                break
        entries.append((name, rconv.gd_type,
                        get_method_unique_name(setter) if setter else ""))
    out: list[str] = []
    for name, gd, setter in entries:
        if not setter:
            continue
        out.append(
            f'    ClassDB::add_property(get_class_static(), '
            f'PropertyInfo(Variant::{gd}, "{name}", PROPERTY_HINT_NONE, "", '
            f'PROPERTY_USAGE_DEFAULT, "{cls.wrapper_name}"), '
            f'"{setter}", "{name}");')
    return out


def _bind_entries(cls: ClassDecl, ctx: tm.TypeContext) -> list[str]:
    out: list[str] = []
    cg = _cg(cls, ctx)
    for ctor in cls.constructors:
        if ctor.skip or _default_ctor(ctor):
            continue
        unique = _unique(ctor)
        args = _bind_arg_names(ctor, ctx, cls)
        out.append(
            f'    ClassDB::bind_static_method("{cls.wrapper_name}", '
            f'D_METHOD("{unique}"{", " + args if args else ""}), '
            f"&{cls.wrapper_name}::{unique}{_defval_suffix(cls, ctor, ctx)});")
    for m in cls.methods + cls.operators + cls.static_methods:
        if m.skip:
            continue
        unique = _unique(m)
        args = _bind_arg_names(m, ctx, cls)
        defv = _defval_suffix(cls, m, ctx)
        if m.kind == MethodKind.STATIC_METHOD:
            out.append(
                f'    ClassDB::bind_static_method("{cls.wrapper_name}", '
                f'D_METHOD("{unique}"{", " + args if args else ""}), '
                f"&{cls.wrapper_name}::{unique}{defv});")
        else:
            out.append(
                f"    ClassDB::bind_method(D_METHOD(\"{unique}\""
                f'{", " + args if args else ""}), '
                f"&{cls.wrapper_name}::{unique}{defv});")
    if cg.storage == "handle" and not any(m.name == "is_null" for m in cls.all_methods):
        out.append(
            f'    ClassDB::bind_method(D_METHOD("is_null"), '
            f"&{cls.wrapper_name}::is_null);")
    for child in _inherited_children(cls, ctx):
        out.append(
            f'    ClassDB::bind_static_method("{cls.wrapper_name}", '
            f'D_METHOD("{_inherited_cast_name(child)}", "S"), '
            f"&{cls.wrapper_name}::{_inherited_cast_name(child)});")
    if (children := _inherited_children(cls, ctx)) and all(
            _inherited_cast_discriminator(c) is not None for c in children):
        out.append(
            f'    ClassDB::bind_static_method("{cls.wrapper_name}", '
            f'D_METHOD("cast", "S"), &{cls.wrapper_name}::cast);')
    for f in cls.fields:
        if not f.is_public or f.skip:
            continue
        snake = to_snake_case(f.name)
        gret = tm.cpp_return(f.type, ctx)
        sconv = tm.cpp_param(f.type, "value", ctx)
        if gret is None or gret.cpp_type == "void":
            continue
        gd = gret.gd_type
        out.append(
            f'    ClassDB::bind_method(D_METHOD("_ort_field_get_{snake}"), '
            f"&{cls.wrapper_name}::_ort_field_get_{snake});")
        if sconv is None or f.is_const:
            continue
        out.append(
            f'    ClassDB::bind_method(D_METHOD("_ort_field_set_{snake}", "value"), '
            f"&{cls.wrapper_name}::_ort_field_set_{snake});")
        out.append(
            f'    ClassDB::add_property(get_class_static(), '
            f'PropertyInfo(Variant::{gd}, "{snake}", PROPERTY_HINT_NONE, "", '
            f'PROPERTY_USAGE_DEFAULT, "{cls.wrapper_name}"), '
            f'"_ort_field_set_{snake}", "_ort_field_get_{snake}");')
    out.extend(_method_property_entries(cls, ctx))
    # Godot's ClassDB keys integer constants per class by constant NAME (the
    # enum name is not part of the key), so enumerators repeated across nested
    # enums of one OCCT class (e.g. GeomFill_Gordon::ResultStatus::NotStarted
    # and GeomFill_Gordon::BuildStage::NotStarted) collide. Bind each name once.
    bound_constants: set[str] = set()
    for enum in _public_nested_enums(cls):
        for v in enum.values:
            if v.name in bound_constants:
                continue
            bound_constants.add(v.name)
            out.append(
                f'    ClassDB::bind_integer_constant(get_class_static(), '
                f'"{enum.name}", "{v.name}", '
                f"static_cast<int64_t>({cls.wrapper_name}::{enum.name}_{v.name}));")
    return out


def generate_class_cpp(cls: ClassDecl, ctx: tm.TypeContext) -> str:
    _skip_ambiguous_ctor_calls(cls, ctx)
    cg = _cg(cls, ctx)
    out: list[str] = []
    out.append(f"// Auto-generated wrapper for {cls.name} -- DO NOT EDIT")
    out.append(f'#include "{cls.wrapper_name}.hpp"')
    out.append("")
    out.append(GCC_DEPRECATED)
    out.append("")
    for w in sorted(_referenced_wrappers(cls, ctx)
                    - {cls.wrapper_name} | {c.wrapper_name
                                            for c in _inherited_children(cls, ctx)}):
        out.append(f'#include "{w}.hpp"')
    for child in sorted({c.name.rsplit("_", 1)[0]
                         for c in _inherited_children(cls, ctx)}):
        out.append(f'#include <{child}.hxx>')
    if _uses_streams(cls):
        out.append("")
        out.append("#include <sstream>")
    if _uses_fstream(cls):
        out.append("")
        out.append("#include <fstream>")
    out.append("")
    out.append("#include <godot_cpp/core/error_macros.hpp>")
    out.append("")
    out.append("namespace godot {")
    out.append("")
    out.append(f"void {cls.wrapper_name}::_bind_methods() {{")
    out.extend(_bind_entries(cls, ctx))
    out.append("}")
    out.append("")
    out.append(_plain_ctor_body(cls, ctx))
    out.append("")
    if cg.has_sync:
        out.append(f"void {cls.wrapper_name}::_sync_base_storage() {{")
        out.append(_sync_body(cls, ctx))
        out.append("}")
        out.append("")
    for ctor in cls.constructors:
        if ctor.skip:
            continue
        out.append(_ctor_body(cls, ctor, ctx))
        out.append("")
        out.append("")
    for m in cls.methods + cls.operators + cls.static_methods:
        if m.skip:
            continue
        body = _method_body(cls, m, ctx)
        if body is None:
            m.skip = True
            m.skip_reason = _method_skip_reason(cls, m, ctx)
            continue
        out.append(body)
        out.append("")
    if cg.storage == "handle" and not any(m.name == "is_null" for m in cls.all_methods):
        out.append(f"bool {cls.wrapper_name}::is_null() const {{")
        out.append("    try {")
        out.append("        return _handle.IsNull();")
        out.append("    } ORT_GUARD_CATCH({});")
        out.append("}")
        out.append("")
    out.extend(_inherited_cast_bodies(cls, ctx))
    out.extend(_field_accessor_bodies(cls, ctx))
    out.append("} // namespace godot")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# OrtEnums
# ---------------------------------------------------------------------------

def _enum_lines(enum) -> list[str]:
    """Lines declaring one standalone enum as a class-scope int64_t enum."""
    lines: list[str] = [f"    enum {enum.name} : int64_t {{"]
    for v in enum.values:
        scope = f"::{enum.name}::" if enum.is_scoped else "::"
        lines.append(
            f"        {enum.name}_{v.name} = static_cast<int64_t>({scope}{v.name}),")
    lines.append("    };")
    return lines


def generate_enums_hpp(modules: list[ModuleDecl]) -> str:
    enums = [e for m in modules for e in m.enums if e.is_public]
    out: list[str] = []
    out.append("// Auto-generated host class for standalone ONNX Runtime enums -- DO NOT EDIT")
    out.append("#pragma once")
    out.append("")
    out.append(GODOT_INCLUDES)
    out.append("#include <onnxruntime_cxx_api.h>")
    out.append("")
    out.append("namespace godot {")
    out.append("")
    out.append("class OrtEnums : public RefCounted {")
    out.append("    GDCLASS(OrtEnums, RefCounted)")
    out.append("")
    out.append("public:")
    out.append("    OrtEnums() = default;")
    out.append("")
    for enum in enums:
        out.extend(_enum_lines(enum))
        out.append("")
    out.append("    static void _bind_methods();")
    out.append("};")
    out.append("")
    out.append("} // namespace godot")
    out.append("")
    for enum in enums:
        out.append(f"VARIANT_ENUM_CAST(OrtEnums::{enum.name});")
    return "\n".join(out) + "\n"


def generate_enums_cpp(modules: list[ModuleDecl]) -> str:
    enums = [e for m in modules for e in m.enums if e.is_public]
    out: list[str] = []
    out.append("// Auto-generated host class for standalone ONNX Runtime enums -- DO NOT EDIT")
    out.append('#include "OrtEnums.hpp"')
    out.append("")
    out.append("namespace godot {")
    out.append("")
    out.append("void OrtEnums::_bind_methods() {")
    for enum in enums:
        for v in enum.values:
            out.append(f'    BIND_ENUM_CONSTANT({enum.name}_{v.name});')
    out.append("}")
    out.append("")
    out.append("} // namespace godot")
    out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# OrtPrimitiveWrappers.hpp
# ---------------------------------------------------------------------------

_BOX_CLASSES = [
    ("bool", ("OrtStandardBoolean", "bool", "BOOL",
             "bool get_value() const { return _native; }",
             "void set_value(bool v) { _native = v; }")),
    ("unsigned char", ("OrtStandardByte", "uint8_t", "INT",
                      "uint8_t get_value() const { return _native; }",
                      "void set_value(uint8_t v) { _native = v; }")),
    ("char", ("OrtStandardCharacter", "char", "INT",
             "int32_t get_value() const { return (int32_t)_native; }",
             "void set_value(int32_t v) { _native = static_cast<char>(v); }")),
    ("char16_t", ("OrtStandardChar16", "char16_t", "INT",
                 "int32_t get_value() const { return (int32_t)_native; }",
                 "void set_value(int32_t v) { _native = static_cast<char16_t>(v); }")),
    ("int", ("OrtStandardInteger", "int32_t", "INT",
            "int32_t get_value() const { return _native; }",
            "void set_value(int32_t v) { _native = v; }")),
    ("long", ("OrtStandardLongInteger", "int64_t", "INT",
             "int64_t get_value() const { return _native; }",
             "void set_value(int64_t v) { _native = static_cast<long>(v); }")),
    ("double", ("OrtStandardReal", "double", "FLOAT",
               "double get_value() const { return _native; }",
               "void set_value(double v) { _native = v; }")),
    ("float", ("OrtStandardShortReal", "float", "FLOAT",
              "float get_value() const { return _native; }",
              "void set_value(float v) { _native = v; }")),
    ("unsigned long", ("OrtStandardULongInteger", "uint64_t", "INT",
                      "uint64_t get_value() const { return _native; }",
                      "void set_value(uint64_t v) { _native = static_cast<unsigned long>(v); }")),
    ("unsigned int", ("OrtStandardUInteger", "uint32_t", "INT",
                     "uint32_t get_value() const { return _native; }",
                     "void set_value(uint32_t v) { _native = v; }")),
]


def _primitive_box_keys_used(modules: list[ModuleDecl],
                             ctx: tm.TypeContext) -> set[str]:
    """Canonical primitive types needing box classes in the scanned API.

    Only primitive types that actually appear as non-const in/out parameters
    (or non-const pointer parameters) anywhere in the scanned modules get a
    box class, so the generated surface tracks the API rather than a hardcoded
    list.
    """
    used: set[str] = set()
    for module in modules:
        for cls in module.classes:
            if cls.skip:
                continue
            for method in cls.all_methods:
                for p in method.parameters:
                    t = p.type
                    if (t.is_ref and not t.is_const) \
                            or (t.is_pointer and not t.pointee_is_const):
                        if t.base_name in tm.PRIMITIVE_WRAPPER_MAP:
                            used.add(t.base_name)
    return used


def generate_primitive_wrappers_hpp(modules: list[ModuleDecl], ctx: tm.TypeContext) -> str:
    enum_boxes = _enum_box_keys_used(modules, ctx)
    box_keys = _primitive_box_keys_used(modules, ctx)
    out: list[str] = []
    out.append("// Auto-generated primitive box classes for in/out parameters -- DO NOT EDIT")
    out.append("#pragma once")
    out.append("")
    out.append(GODOT_INCLUDES)
    out.append("#include <onnxruntime_cxx_api.h>")
    out.append("")
    out.append("namespace godot {")
    out.append("")
    for occt_type, (box_name, native_type, gd_type, getter, setter) in _BOX_CLASSES:
        if occt_type not in box_keys:
            continue
        out.append(f"class {box_name} : public RefCounted {{")
        out.append(f"    GDCLASS({box_name}, RefCounted)")
        out.append("public:")
        out.append(f"    {native_type} _native = {{}};")
        out.append("")
        out.append(f"    {box_name}() = default;")
        out.append(f"    {box_name}({native_type} v) : _native(v) {{}}")
        out.append("")
        out.append(f"    {getter}")
        out.append(f"    {setter}")
        out.append("")
        out.append("    static void _bind_methods() {")
        out.append(f'        ClassDB::bind_method(D_METHOD("get_value"), &{box_name}::get_value);')
        out.append(f'        ClassDB::bind_method(D_METHOD("set_value", "value"), &{box_name}::set_value);')
        out.append(f'        ADD_PROPERTY(PropertyInfo(Variant::{gd_type}, "value"), "set_value", "get_value");')
        out.append("    }")
        out.append("};")
        out.append("")
    for enum_name, (box_name, enum_decl) in sorted(enum_boxes.items()):
        path = tm._enum_occt_path(enum_decl)
        out.append(f"class {box_name} : public RefCounted {{")
        out.append(f"    GDCLASS({box_name}, RefCounted)")
        out.append("public:")
        out.append(f"    int64_t _native = 0;")
        out.append("")
        out.append(f"    {box_name}() = default;")
        out.append(f"    {box_name}(int64_t v) : _native(v) {{}}")
        out.append(f"    {box_name}({path} v) : _native(static_cast<int64_t>(v)) {{}}")
        out.append("")
        out.append("    int64_t get_value() const { return _native; }")
        out.append("    void set_value(int64_t v) { _native = v; }")
        out.append("")
        out.append("    static void _bind_methods() {")
        out.append(f'        ClassDB::bind_method(D_METHOD("get_value"), &{box_name}::get_value);')
        out.append(f'        ClassDB::bind_method(D_METHOD("set_value", "value"), &{box_name}::set_value);')
        out.append('        ADD_PROPERTY(PropertyInfo(Variant::INT, "value"), "set_value", "get_value");')
        out.append("    }")
        out.append("};")
        out.append("")
    out.append("} // namespace godot")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# OrtCallableStreams.hpp
# ---------------------------------------------------------------------------

def generate_callable_streams_hpp() -> str:
    return """// Auto-generated std::iostream shims over Godot Callable -- DO NOT EDIT
#pragma once

#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/string.hpp>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include <istream>
#include <locale>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>

namespace ort_gd {

inline void OrtPinInterposedLocale(const std::locale &p_replaced) {
    const void *const impl = *(const void *const *)&p_replaced;
    int *const refcount = static_cast<int *>(const_cast<void *>(impl));
#if defined(_MSC_VER)
    _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(refcount), 1);
#else
    __atomic_fetch_add(refcount, 1, __ATOMIC_ACQ_REL);
#endif
}

class OrtCallableOStream final : public std::ostream {
public:
    explicit OrtCallableOStream(const ::godot::Callable &p_sink)
        : std::ostream(&myBuffer), mySink(p_sink) {
        OrtPinInterposedLocale(imbue(std::locale::classic()));
    }

    ~OrtCallableOStream() override {
        flush();
    }

    std::string str() const {
        return myBuffer.str();
    }

    std::ostream &stream() {
        return *this;
    }

private:
    class CallableSink final : public std::stringbuf {
    public:
        explicit CallableSink(const ::godot::Callable &p_sink) : mySink(p_sink) {}

        int sync() override {
            std::string text = str();
            if (!text.empty() && mySink.is_valid()) {
                ::godot::String gd_str = ::godot::String::utf8(text.c_str());
                mySink.call(gd_str);
            }
            str("");
            return 0;
        }

    private:
        ::godot::Callable mySink;
    };

    ::godot::Callable mySink;
    CallableSink myBuffer{mySink};
};

class OrtCallableIStream final : public std::istream {
public:
    explicit OrtCallableIStream(const ::godot::Callable &p_source)
        : std::istream(&myBuffer), myBuffer(p_source) {
        OrtPinInterposedLocale(imbue(std::locale::classic()));
    }

    std::istream &stream() {
        return *this;
    }

private:
    class CallableSource final : public std::streambuf {
    public:
        explicit CallableSource(const ::godot::Callable &p_source) : mySource(p_source) {}

    protected:
        int_type underflow() override {
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
            if (myDone || !mySource.is_valid()) {
                return traits_type::eof();
            }
            ::godot::String chunk = mySource.call();
            if (chunk.is_empty()) {
                myDone = true;
                return traits_type::eof();
            }
            ::godot::CharString utf8 = chunk.utf8();
            myChunk.assign(utf8.get_data(), utf8.length());
            if (myChunk.empty()) {
                myDone = true;
                return traits_type::eof();
            }
            char *base = myChunk.data();
            setg(base, base, base + myChunk.size());
            return traits_type::to_int_type(*gptr());
        }

    private:
        ::godot::Callable mySource;
        std::string myChunk;
        bool myDone = false;
    };

    CallableSource myBuffer;
};

} // namespace ort_gd
"""


# ---------------------------------------------------------------------------
# OrtMemory.hpp
# ---------------------------------------------------------------------------

def generate_occt_memory_hpp() -> str:
    return """// Auto-generated memory helpers -- DO NOT EDIT
#pragma once

#include <memory>

namespace ort_gd {
} // namespace ort_gd
"""


# ---------------------------------------------------------------------------
# module.h
# ---------------------------------------------------------------------------

def _registration_order(wrappers: list[ClassDecl]) -> list[str]:
    by_occt = {w.name: w for w in wrappers}
    order: list[str] = []
    seen: set[str] = set()

    def visit(occt_name: str) -> None:
        cls = by_occt.get(occt_name)
        if cls is None:
            return
        if cls.wrapper_name in seen:
            return
        seen.add(cls.wrapper_name)
        for base in cls.base_classes:
            visit(base)
        order.append(cls.wrapper_name)

    for w in sorted(wrappers, key=lambda c: c.name):
        visit(w.name)
    return order


def generate_module_h(module: ModuleDecl, wrappers: list[ClassDecl],
                      primitive_keys: set[str],
                      enum_box_names: set[str] = frozenset()) -> str:
    out: list[str] = []
    out.append("// AUTOGENERATED by OpenCASCADE.gd-autowrapper -- DO NOT EDIT")
    out.append("#ifndef AUTOWRAPPER_MODULE_H")
    out.append("#define AUTOWRAPPER_MODULE_H")
    out.append("")
    out.append("#include <godot_cpp/core/class_db.hpp>")
    out.append("#include <godot_cpp/godot.hpp>")
    out.append("")
    out.append('#include "OrtEnums.hpp"')
    out.append('#include "OrtCallableStreams.hpp"')
    if primitive_keys or enum_box_names:
        out.append('#include "OrtPrimitiveWrappers.hpp"')
    for w in sorted({c.wrapper_name for c in wrappers}):
        out.append(f'#include "{w}.hpp"')
    out.append("")
    out.append("namespace godot {")
    out.append("")
    out.append("inline void gdext_initialize_module_auto(godot::ModuleInitializationLevel p_level) {")
    out.append("    (void)p_level;")
    wrapper_names = {c.wrapper_name for c in wrappers}
    _box_by_type = {t: b[0] for t, b in _BOX_CLASSES}
    for key in sorted(primitive_keys, key=lambda k: _box_by_type[k]):
        wclass = _box_by_type[key]
        # Primitive wrapper names that are also full generated wrappers (e.g.
        # TCollection_AsciiString) are registered by the wrapper loop below.
        if wclass in wrapper_names:
            continue
        out.append(f"    godot::ClassDB::register_class<{wclass}>();")
    for box in sorted(enum_box_names):
        out.append(f"    godot::ClassDB::register_class<{box}>();")
    out.append("    godot::ClassDB::register_class<OrtEnums>();")
    for w in _registration_order(wrappers):
        out.append(f"    godot::ClassDB::register_class<{w}>();")
    out.append("}")
    out.append("")
    out.append("inline void gdext_uninitialize_module_auto(godot::ModuleInitializationLevel p_level) {")
    out.append("    (void)p_level;")
    out.append("}")
    out.append("")
    out.append("} // namespace godot")
    out.append("")
    out.append("#endif // AUTOWRAPPER_MODULE_H")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------

_GEN_POOL_STATE: tuple | None = None


def _gen_pool_worker_init(modules, ctx, out_dir) -> None:
    global _GEN_POOL_STATE
    _GEN_POOL_STATE = (modules, ctx, Path(out_dir))


def _gen_class_worker(task: tuple[int, int]):
    """Generate one class's wrapper files in a fork pool.

    ``modules``/``ctx`` are inherited from the parent via fork (COW), so only
    the tiny ``(module_index, class_index)`` task is pickled.  Generation
    mutates skip flags (unmappable methods, ambiguous ctors), so the pre-fork
    skip state is snapshotted and the changed entries returned for the parent
    to replay on its own copies.
    """
    global _GEN_POOL_STATE
    modules, ctx, out_dir = _GEN_POOL_STATE
    mi, ci = task
    cls = modules[mi].classes[ci]
    before: dict[tuple[str, int], tuple[bool, str | None]] = {}
    for groups in (("c", cls.constructors), ("m", cls.methods),
                   ("o", cls.operators), ("s", cls.static_methods)):
        for i, m in enumerate(groups[1]):
            before[(groups[0], i)] = (m.skip, m.skip_reason)
    hpp = generate_class_hpp(cls, ctx)
    cpp = generate_class_cpp(cls, ctx)
    written: list[Path] = []
    for name, content in ((f"{cls.wrapper_name}.hpp", hpp),
                          (f"{cls.wrapper_name}.cpp", cpp)):
        p = out_dir / name
        if not (p.exists() and p.read_text() == content):
            p.write_text(content)
        written.append(p)
    mutations: list[tuple[str, int, bool, str | None]] = []
    for groups in (("c", cls.constructors), ("m", cls.methods),
                   ("o", cls.operators), ("s", cls.static_methods)):
        for i, m in enumerate(groups[1]):
            if before[(groups[0], i)] != (m.skip, m.skip_reason):
                mutations.append((groups[0], i, m.skip, m.skip_reason))
    return mi, ci, written, mutations


def _generate_class_serial(cls, ctx, out_dir, write) -> None:
    """Single-process class generation (filtered / tiny module sets)."""
    write(f"{cls.wrapper_name}.hpp", generate_class_hpp(cls, ctx))
    write(f"{cls.wrapper_name}.cpp", generate_class_cpp(cls, ctx))


def generate_all(modules: list[ModuleDecl], out_dir: Path,
                 probe_out: Path | None = None,
                 missing: set[str] | None = None,
                 illformed: set[str] | None = None,
                 module_filter: str | None = None) -> list[Path]:
    """Generate all wrapper files for modules (in include-DAG order) into out_dir.

    `missing` (see autogen.audit) marks every generated method whose OCCT symbol
    is absent from the linked libraries as skipped.  `illformed` (same source)
    marks methods whose instantiation does not compile for the substituted
    template arguments.  When `probe_out` is set, a symbol-audit probe TU is
    also written there after all skip decisions are final.

    `module_filter` restricts *writing* to one module's classes (all modules are
    still loaded so the cross-module context stays complete); used by the dev
    loop to rewrap just the module under work without rescanning.  In filtered
    mode the enums/module.h files and the global stale-file cleanup are skipped.
    """
    # Skip decisions (missing symbols / ill-formed instantiations) mutate the
    # classes (e.g. pinning default_constructible / has_public_default_ctor),
    # so they must land BEFORE build_context: the storage of every wrapper (and
    # the typemap's unique_ptr/handle sets) is derived from those flags.
    if missing:
        from .audit import apply_missing
        apply_missing(modules, missing)
    if illformed:
        from .audit import apply_illformed
        apply_illformed(modules, illformed)
    ctx = build_context(modules)
    # Inherited methods of unwrapped value-style bases (result accessors of
    # BRepBuilderAPI_MakeShape & co) are flattened onto their wrapped value
    # descendants before codegen and the symbol probe see them, so
    # shape()/is_done()/modified()/... are bound and audited exactly like
    # own-header methods.
    flatten_inherited_methods(modules, ctx)
    # Flattened methods were unknown to the apply_illformed above, so re-apply
    # it now: a flattened copy can be ill-formed where the own-header probe
    # never saw it (e.g. the 2-arg Blend_AppFunction::Set is hidden by an
    # intermediate class's Set overloads, making the call unresolvable).  Only
    # method/field skips and `returnable` are set here -- never ctor/storage
    # flags -- so running it after build_context is safe.
    if illformed:
        from .audit import apply_illformed
        apply_illformed(modules, illformed)
    wrappers: list[ClassDecl] = []
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, content: str) -> Path:
        p = out_dir / name
        if p.exists() and p.read_text() == content:
            written.append(p)
            return p
        p.write_text(content)
        written.append(p)
        return p

    if module_filter is not None:
        for module in modules:
            if module.name != module_filter:
                continue
            for cls in module.classes:
                if cls.skip:
                    continue
                group_overloads(cls)
                write(f"{cls.wrapper_name}.hpp", generate_class_hpp(cls, ctx))
                write(f"{cls.wrapper_name}.cpp", generate_class_cpp(cls, ctx))
                wrappers.append(cls)
        return written

    # Generate the class wrappers across a fork pool: generation is pure
    # Python (GIL-bound), so separate processes give real parallelism, and
    # fork+COW lets every worker share the (large) modules/ctx inherited from
    # this process instead of pickling them per task.  The workers replay the
    # skip-flag mutations they made so the probe/probe-parts below see the same
    # state a serial pass would produce.
    tasks: list[tuple[int, int]] = []
    for mi, module in enumerate(modules):
        for ci, cls in enumerate(module.classes):
            if cls.skip:
                continue
            group_overloads(cls)
            tasks.append((mi, ci))
            wrappers.append(cls)
    if len(tasks) > 1:
        workers = min(len(tasks), os.cpu_count() or 4, 16)
        # fork is only available on POSIX; Windows (and macOS >= 3.8 defaults)
        # has no fork start method, so use the platform default (spawn) there.
        # spawn pickles modules/ctx once into the worker initializer; if
        # anything turns out unpicklable the pool fails to start and we fall
        # back to serial generation rather than aborting the whole pass.
        start_methods = _mp.get_all_start_methods()
        mp_ctx = _mp.get_context("fork" if "fork" in start_methods else None)
        pool = None
        try:
            pool = mp_ctx.Pool(
                processes=workers, initializer=_gen_pool_worker_init,
                initargs=(modules, ctx, str(out_dir)))
        except Exception:
            pool = None
        if pool is not None:
            try:
                for mi, ci, chunk, mutations in pool.imap_unordered(
                        _gen_class_worker, tasks):
                    written.extend(chunk)
                    cls = modules[mi].classes[ci]
                    for group, i, skip, reason in mutations:
                        if group == "c":
                            m = cls.constructors[i]
                        elif group == "m":
                            m = cls.methods[i]
                        elif group == "o":
                            m = cls.operators[i]
                        else:
                            m = cls.static_methods[i]
                        m.skip = skip
                        m.skip_reason = reason
            except Exception:
                pool.terminate()
                pool.join()
                pool = None
            else:
                pool.close()
                pool.join()
        if pool is None:
            for mi, ci in tasks:
                cls = modules[mi].classes[ci]
                _generate_class_serial(cls, ctx, out_dir, write)
    else:
        for mi, ci in tasks:
            cls = modules[mi].classes[ci]
            _generate_class_serial(cls, ctx, out_dir, write)

    write("OrtEnums.hpp", generate_enums_hpp(modules))
    write("OrtEnums.cpp", generate_enums_cpp(modules))
    write("OrtPrimitiveWrappers.hpp", generate_primitive_wrappers_hpp(modules, ctx))
    write("OrtCallableStreams.hpp", generate_callable_streams_hpp())
    write("OrtMemory.hpp", generate_occt_memory_hpp())
    primitive_keys = _primitive_box_keys_used(modules, ctx)
    enum_boxes = _enum_box_keys_used(modules, ctx)
    write("module.h", generate_module_h(
        modules[0] if modules else ModuleDecl(name="Core"), wrappers,
        primitive_keys, _enum_box_class_names(enum_boxes)))

    # Remove any wrapper files that are no longer generated (e.g. classes that
    # became skippable since the last run).
    generated = {p.name for p in written}
    for stale in list(out_dir.glob("*.hpp")) + list(out_dir.glob("*.cpp")):
        if stale.name not in generated:
            stale.unlink()

    if probe_out:
        from .audit import write_probe_parts
        from .ort import find_ort_install
        project_root = Path(__file__).resolve().parent.parent.parent
        probe = Path(probe_out)
        probe.parent.mkdir(parents=True, exist_ok=True)
        write_probe_parts(probe, modules, ctx, find_ort_install(project_root))
    return written


def generate_module(module: ModuleDecl, out_dir: Path) -> list[Path]:
    """Generate wrapper files for a single module (kept for compat)."""
    return generate_all([module], out_dir)
