"""AST extraction: classes, enums, typedefs, methods, fields from a TU.

Clean extraction driven by the libclang Type API; no source-text
mis-resolution heuristics.  The only source-text reader is token-extent
default-argument recovery (libclang does not expose defaults).  Field access
is tracked from CXX_ACCESS_SPEC_DECL decls, which are exact, because the
field-level access_specifier is unreliable for OCCT class bodies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from clang.cindex import (AccessSpecifier, Cursor, CursorKind, Diagnostic,
                          TranslationUnit, Type, TypeKind)

from .model import (ClassDecl, DocBlock, EnumDecl, EnumValue, FieldDecl,
                    MethodDecl, MethodKind, OperatorType, Parameter)
from .types import make_type

# ---------------------------------------------------------------------------
# Small, data-driven skip tables (nothing type-resolution related)
# ---------------------------------------------------------------------------

# Operators that cannot be represented as a named GDScript method.
UNWRAPPABLE_OPERATORS = {"[]", ",", "->", "->*", "new", "delete", "new[]",
                         "delete[]", "<<", ">>", "~"}

# Methods that are meaningless from GDScript.
SKIP_METHOD_NAMES = {
    "operator new", "operator delete", "operator new[]", "operator delete[]",
    "operator=", "operator[]", "operator<<", "operator>>",
    "ShallowCopy", "ShallowDump",  # stream/IO helpers
    "ObjectIterator",              # returns a typedef libclang cannot follow
}

# Methods/constructors declared Standard_EXPORT in a header but with NO
# definition in the installed OCCT static libs (header/library drift).
# Wrapping them makes the extension .so reference symbols dlopen cannot
# resolve ("undefined symbol" at runtime). Keyed by OCCT class name, value is
# a set of (method name, parameter count); a constructor's name is its class
# name (matching the IR). Verified against OCCT 8.0.1 via
# `nm -D --undefined-only` on the built extension.
SKIP_METHODS_BY_CLASS: dict[str, set[tuple[str, int]]] = {
    "AppDef_MultiLine": {("SetParameter", 2)},
    "BRepFeat_MakeLinearForm": {("TransformShapeFU", 1)},
    "BRepOffsetAPI_FindContigousEdges": {("NbEdges", 0)},
    "BRepOffset_MakeOffset": {("GetAnalyse", 0)},
    "GeomFill_SweepSectionGenerator": {("GeomFill_SweepSectionGenerator", 4),
                                      ("Init", 4)},
    "IntTools_PntOnFace": {("IsValid", 0)},
    "ShapeFix_WireSegment": {("ShapeFix_WireSegment", 2)},
    "TCollection_AsciiString": {("IsEqual", 2)},
}

# Operators we wrap, mapped to stable names.
BINARY_OPERATOR_TYPES = {
    "+": OperatorType.PLUS, "-": OperatorType.MINUS,
    "*": OperatorType.MULTIPLY, "/": OperatorType.DIVIDE,
    "%": OperatorType.MODULO, "^": OperatorType.CROSS,
    "==": OperatorType.EQUALS, "!=": OperatorType.NOT_EQUALS,
    "<": OperatorType.LESS, ">": OperatorType.GREATER,
}
COMPOUND_OPERATOR_TYPES = {
    "+=": OperatorType.PLUS_ASSIGN, "-=": OperatorType.MINUS_ASSIGN,
    "*=": OperatorType.MULTIPLY_ASSIGN, "/=": OperatorType.DIVIDE_ASSIGN,
    "^=": OperatorType.CROSS_ASSIGN,
}
UNARY_OPERATOR_TYPES = {
    "-": OperatorType.UNARY_MINUS, "+": OperatorType.UNARY_PLUS,
    "*": OperatorType.DEREFERENCE,
}


def classify_operator(name: str) -> tuple[OperatorType | None, str]:
    """Map an operator spelling to (OperatorType, wrapper name) or (None, "")."""
    if not name.startswith("operator"):
        return None, ""
    op = name[len("operator"):].strip()
    if not op or op in UNWRAPPABLE_OPERATORS:
        return None, ""
    if op in BINARY_OPERATOR_TYPES:
        return BINARY_OPERATOR_TYPES[op], op
    if op in COMPOUND_OPERATOR_TYPES:
        return COMPOUND_OPERATOR_TYPES[op], op
    if op in UNARY_OPERATOR_TYPES:
        return UNARY_OPERATOR_TYPES[op], f"unary_{op}"
    if op in ("++", "--"):
        return (OperatorType.INCREMENT if op == "++" else OperatorType.DECREMENT), op
    if op == "()":
        return OperatorType.CALL, "()"
    return None, ""


# ---------------------------------------------------------------------------
# Default arguments (token-extent recovery)
# ---------------------------------------------------------------------------

def _param_default(cursor: Cursor) -> str | None:
    """Recover the source text of a parameter default from its token extent."""
    try:
        tokens = list(cursor.get_tokens())
    except Exception:
        return None
    eq_idx = -1
    for i, t in enumerate(tokens):
        if t.spelling == "=":
            eq_idx = i
            break
    if eq_idx < 0:
        return None
    start = tokens[eq_idx].extent.end
    end = cursor.extent.end
    try:
        if start.file is None or end.file is None or start.file.name != end.file.name:
            return None
        if start.offset is None or end.offset is None:
            return None
        with open(start.file.name) as f:
            text = f.read()[start.offset:end.offset]
    except (OSError, IndexError, AttributeError):
        return None
    text = text.strip()
    if not text:
        return None
    while text.endswith((",", ")")):
        if text.endswith(")"):
            if text.count("(") >= text.count(")"):
                break
            text = text[:-1].rstrip()
        else:
            text = text[:-1].rstrip()
    return text or None


def _is_deleted(cursor: Cursor) -> bool:
    """True when the declaration is `= delete` (libclang does not expose it)."""
    try:
        toks = [t.spelling for t in cursor.get_tokens()]
    except Exception:
        return False
    for i, t in enumerate(toks[:-1]):
        if t == "=" and i + 1 < len(toks) and toks[i + 1] == "delete":
            return True
    return False


def _is_copy_ctor(cursor: Cursor, class_name: str) -> bool:
    """Structurally detect copy constructors (single const-ref / handle<Self>)."""
    params = [c for c in cursor.get_children() if c.kind == CursorKind.PARM_DECL]
    if len(params) != 1:
        return False
    t = make_type(params[0].type)
    return (t.is_handle and t.handle_inner == class_name) or (
        t.base_name == class_name and t.is_ref and t.is_const)


def _has_explicit_noncopyable(cursor: Cursor) -> bool:
    """True if the class explicitly deletes its copy ctor or copy/move assignment."""
    for child in cursor.get_children():
        try:
            if child.kind == CursorKind.CXX_METHOD:
                if (child.spelling == "operator=" and child.is_deleted_method()
                        and (child.is_copy_assignment_operator_method()
                             or child.is_move_assignment_operator_method())):
                    return True
            elif (child.kind == CursorKind.CONSTRUCTOR and child.is_copy_constructor()
                  and child.is_deleted_method()):
                return True
        except Exception:
            pass
    return False


# std class templates whose values are never copy-assignable (mutexes,
# atomics, streams, threads, ...).  A by-value member of such a type makes the
# enclosing class's copy assignment implicitly deleted; libclang does not
# expose the implicitly-deleted operator, so the field's canonical spelling is
# matched directly.  ``std::unique_ptr`` is included (the extractor also flags
# it by name below) so every known root stays in one place.
_STD_NONCOPYABLE_ROOTS = frozenset({
    "std::unique_ptr", "std::auto_ptr",
    "std::atomic", "std::atomic_flag", "std::atomic_ref",
    "std::mutex", "std::recursive_mutex", "std::timed_mutex",
    "std::recursive_timed_mutex", "std::shared_mutex",
    "std::shared_timed_mutex",
    "std::condition_variable", "std::condition_variable_any",
    "std::thread", "std::jthread",
    "std::future", "std::shared_future", "std::promise",
    "std::packaged_task",
    "std::ifstream", "std::ofstream", "std::fstream",
    "std::istringstream", "std::ostringstream", "std::stringstream",
    "std::istream", "std::ostream", "std::iostream",
    "std::stop_callback",
    "std::counting_semaphore", "std::binary_semaphore", "std::latch",
    "std::barrier",
})


def _field_has_deleted_copy_assignment(f: FieldDecl) -> bool:
    """A data member that implicitly deletes the class's copy assignment.

    The enclosing class's copy assignment is unusable when a member is a
    reference, is const, or is a value of a known non-copyable std type
    (``std::atomic``, ``std::shared_mutex``, ...) -- the very cases libclang
    keeps implicit, so these are the only structural signals the scan can see.
    """
    if f.type.is_ref:
        return True
    if f.is_const:
        return True
    canonical = f.type.canonical_spelling
    for root in _STD_NONCOPYABLE_ROOTS:
        if canonical == root or canonical.startswith(root + "<"):
            return True
    return False


def _params(cursor: Cursor) -> list[Parameter]:
    out: list[Parameter] = []
    for child in cursor.get_children():
        if child.kind == CursorKind.PARM_DECL:
            name = child.spelling or f"arg{len(out)}"
            out.append(Parameter(type=make_type(child.type), name=name,
                                 default_value=_param_default(child)))
    return out


def _doc(cursor: Cursor) -> DocBlock:
    try:
        brief = cursor.brief_comment or ""
    except Exception:
        brief = ""
    try:
        raw = cursor.raw_comment or ""
    except Exception:
        raw = ""
    return DocBlock(brief=brief, raw=raw)


def _extract_method(cursor: Cursor, class_name: str) -> MethodDecl | None:
    name = cursor.spelling
    if name in SKIP_METHOD_NAMES:
        return None
    params = _params(cursor)
    if (name, len(params)) in SKIP_METHODS_BY_CLASS.get(class_name, ()):
        return None
    if _is_deleted(cursor):
        return None
    if cursor.access_specifier != AccessSpecifier.PUBLIC:
        return None

    return_type = make_type(cursor.result_type)
    op_type, op_name = classify_operator(name)

    is_static = bool(cursor.is_static_method())
    return MethodDecl(
        name=op_name if op_type else name,
        return_type=return_type,
        parameters=params,
        kind=(MethodKind.STATIC_METHOD if is_static and not op_type
              else MethodKind.OPERATOR if op_type else MethodKind.METHOD),
        is_const=bool(cursor.is_const_method()),
        is_virtual=bool(cursor.is_virtual_method()),
        is_static=is_static,
        is_default=bool(cursor.is_default_method()),
        is_pure_virtual=bool(cursor.is_pure_virtual_method()),
        is_variadic=bool(cursor.type.is_function_variadic()),
        operator_type=op_type,
        doc=_doc(cursor),
    )


def _extract_constructor(cursor: Cursor, class_name: str) -> MethodDecl | None:
    if cursor.access_specifier in (AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED):
        return None
    if _is_deleted(cursor):
        return None
    if _is_copy_ctor(cursor, class_name):
        return None
    params = _params(cursor)
    if (class_name, len(params)) in SKIP_METHODS_BY_CLASS.get(class_name, ()):
        return None
    return MethodDecl(
        name=class_name,
        parameters=params,
        kind=MethodKind.CONSTRUCTOR,
        is_default=bool(cursor.is_default_method()),
        doc=_doc(cursor),
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _find_type_def(root: Cursor, name: str,
                   defs: dict[str, Cursor | None] | None = None) -> Cursor | None:
    """Find the definition cursor of the type named `name` in the TU.

    Memoized: the scan runs once per parsed header and the TU is large, so a
    repeated search for the same root name must not re-walk the whole tree.
    """
    if defs is None:
        defs = {}
    if name in defs:
        return defs[name]

    def search(cursor: Cursor) -> Cursor | None:
        if cursor.spelling == name and cursor.kind in (
                CursorKind.CLASS_DECL, CursorKind.CLASS_TEMPLATE,
                CursorKind.STRUCT_DECL,
                CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            if cursor.is_definition():
                return cursor
        for c in cursor.get_children():
            r = search(c)
            if r:
                return r
        return None

    defs[name] = search(root)
    return defs[name]


def _is_transient(cursor: Cursor, defs: dict[str, Cursor | None] | None = None,
                  root: Cursor | None = None) -> bool:
    """True when the class (or any base) derives from Standard_Transient.

    OCCT marks Transient descendants with DEFINE_STANDARD_RTTIEXT.  The base
    chain can cross typedefs (``BVH_PrimitiveSet3d`` is a typedef of the
    template ``BVH_PrimitiveSet<double, 3>``) and class templates whose base
    specifiers are dependent types.  libclang's ``get_definition`` on a
    typedef base yields the TYPEDEF_DECL (no bases to follow) and on a
    dependent base yields the template definition; a type that cannot be
    followed structurally is resolved by root name in the translation unit.
    """
    if cursor is None or not cursor.is_definition():
        return False
    name = cursor.spelling or ""
    if name == "Standard_Transient":
        return True
    if cursor.kind == CursorKind.TYPEDEF_DECL:
        t = cursor.underlying_typedef_type
        if t is None or root is None:
            return False
        root_name = re.sub(r"<.*", "", t.get_canonical().spelling).strip().split("::")[-1]
        found = _find_type_def(root, root_name, defs)
        return found is not None and _is_transient(found, defs, root)
    for child in cursor.get_children():
        if child.kind != CursorKind.CXX_BASE_SPECIFIER:
            continue
        base = child.get_definition()
        if base is None or not base.is_definition():
            # Dependent base (e.g. BVH_Object<T, N> inside a class template):
            # resolve the root template name in the TU and follow its own base.
            if root is None:
                continue
            root_name = re.sub(r"<.*", "", child.type.spelling).strip().split("::")[-1]
            if not root_name or root_name == name:
                continue
            found = _find_type_def(root, root_name, defs)
            if found is not None and _is_transient(found, defs, root):
                return True
            continue
        if _is_transient(base, defs, root):
            return True
    return False


def _base_names(cursor: Cursor) -> list[str]:
    return [c.type.spelling for c in cursor.get_children()
            if c.kind == CursorKind.CXX_BASE_SPECIFIER]


def _ctor_default_usable(cursor: Cursor) -> bool:
    """True when a constructor can be invoked with no arguments (no parameters
    or all parameters defaulted): a usable default ctor."""
    params = [c for c in cursor.get_children() if c.kind == CursorKind.PARM_DECL]
    if not params:
        return True
    return all(_param_default(p) is not None for p in params)


def _record_decl_for_type(t: Type, tu_root: Cursor | None,
                          defs: dict[str, Cursor | None]) -> Cursor | None:
    """Definition cursor of the record type `t`, or None when unresolvable.

    A canonical record type's ``get_declaration()`` returns the (specialization
    or primary) cursor, which may carry no children; the TU-wide name lookup
    used for Transient detection then resolves the primary template definition
    whose constructors actually decide default-constructibility.
    """
    try:
        canon = t.get_canonical()
        if canon.kind != TypeKind.RECORD:
            return None
        decl = canon.get_declaration()
        if decl is not None and decl.is_definition() \
                and list(decl.get_children()):
            return decl
    except Exception:
        return None
    if tu_root is None:
        return None
    try:
        name = re.sub(r"<.*", "", canon.spelling).strip().split("::")[-1]
    except Exception:
        return None
    if not name:
        return None
    return _find_type_def(tu_root, name, defs)


def _record_default_constructible(decl: Cursor | None, tu_root: Cursor | None,
                                  defs: dict[str, Cursor | None],
                                  seen: set[tuple],
                                  as_base: bool = False) -> bool:
    """Structural scan-time probe of a record's default-constructibility.

    The symbol audit proves ``T()`` by compiling it, but only on host builds:
    cross-target scans (``--target=``) skip it, so an implicitly *deleted*
    default ctor leaks into the wrappers there (e.g. ``BRepGraph_FacesOfEdge``,
    whose ``EdgeParentsOf`` base declares no default ctor: its wrapper's
    ``_native()`` member init does not compile on arm-linux).  Mirror the audit
    with a pure structural check: a class that declares no ctor at all relies
    on the implicit default ctor, which C++ deletes when a direct base or a
    data member is itself not default-constructible (or a member is a
    reference).

    ``as_base`` captures the access context of the call: a class's implicit
    default ctor may call the *protected* default ctor of a direct base (OCCT
    entity interfaces like ``IGESData_IGESEntity`` / ``TPrsStd_Driver`` keep a
    protected ctor, so their leaf subclasses remain default-constructible), but
    only *public* ctors of data members.

    Conservative by design: anything unresolvable (dependent types, unions,
    non-record types, cycles) is assumed constructible, so a genuinely
    constructible class is never flipped.
    """
    if decl is None or not decl.is_definition():
        return True
    if decl.kind in (CursorKind.CLASS_DECL, CursorKind.CLASS_TEMPLATE,
                     CursorKind.STRUCT_DECL):
        ctors = [c for c in decl.get_children()
                 if c.kind == CursorKind.CONSTRUCTOR]
        if ctors:
            # The primary template declares the ctors every specialization
            # (apart from explicit specializations, which OCCT does not use
            # here) inherits; a usable default ctor is exactly one that can be
            # called with no arguments.
            for c in ctors:
                if not _ctor_default_usable(c):
                    continue
                if c.access_specifier == AccessSpecifier.PUBLIC:
                    return True
                if as_base and c.access_specifier == AccessSpecifier.PROTECTED:
                    return True
            return False
    else:
        return True  # unions and other kinds: assume constructible
    # No declared ctor: the implicit default ctor exists iff every direct base
    # and every data member keeps it usable.
    try:
        key = (decl.location.file.name, decl.extent.start.offset)
    except Exception:
        key = None
    if key is not None:
        if key in seen:
            return True  # cycle guard: assume constructible
        seen.add(key)
    try:
        for child in decl.get_children():
            if child.kind == CursorKind.CXX_BASE_SPECIFIER:
                if not _record_default_constructible(
                        _record_decl_for_type(child.type, tu_root, defs),
                        tu_root, defs, seen, as_base=True):
                    return False
            elif child.kind == CursorKind.FIELD_DECL:
                if child.type.kind in (TypeKind.LVALUEREF, TypeKind.RVALUEREF):
                    return False
                if not _record_default_constructible(
                        _record_decl_for_type(child.type, tu_root, defs),
                        tu_root, defs, seen, as_base=False):
                    return False
    except Exception:
        return True
    return True


def _class_implicit_default_usable(cursor: Cursor, tu_root: Cursor | None,
                                   defs: dict[str, Cursor | None]) -> bool:
    """True when a class declaring no ctor at all still has a usable implicit
    default ctor (no direct base or data member deletes it)."""
    seen: set[tuple] = set()
    try:
        for child in cursor.get_children():
            if child.kind == CursorKind.CXX_BASE_SPECIFIER:
                if not _record_default_constructible(
                        _record_decl_for_type(child.type, tu_root, defs),
                        tu_root, defs, seen, as_base=True):
                    return False
            elif child.kind == CursorKind.FIELD_DECL:
                if child.type.kind in (TypeKind.LVALUEREF, TypeKind.RVALUEREF):
                    return False
                if not _record_default_constructible(
                        _record_decl_for_type(child.type, tu_root, defs),
                        tu_root, defs, seen, as_base=False):
                    return False
    except Exception:
        return True
    return True


def _extract_enum(cursor: Cursor, header: str, parent: str = "") -> EnumDecl | None:
    name = cursor.spelling or ""
    if not name or name.startswith("("):
        return None  # anonymous enum: no stable name to expose across modules
    values: list[EnumValue] = []
    for c in cursor.get_children():
        if c.kind == CursorKind.ENUM_CONSTANT_DECL:
            try:
                val = c.enum_value
            except Exception:
                val = None
            values.append(EnumValue(name=c.spelling, value=val))
    try:
        # Nested enums carry a real access specifier; file-scope enums report
        # INVALID but are always public.
        is_public = (not parent) or cursor.access_specifier == AccessSpecifier.PUBLIC
    except Exception:
        is_public = True
    return EnumDecl(
        name=name, values=values,
        is_scoped=cursor.kind == CursorKind.ENUM_DECL and "scoped" in str(cursor.type.spelling),
        is_nested=bool(parent), parent_class=parent,
        is_public=is_public,
        header_file=header, doc=_doc(cursor),
    )


def _extract_base_methods(cursor: Cursor, class_name: str, cls: ClassDecl, seen_methods: set[str]) -> None:
    for child in cursor.get_children():
        if child.kind == CursorKind.CXX_BASE_SPECIFIER:
            try:
                base_type = child.type
                base_decl = base_type.get_declaration()
                if base_decl and base_decl.is_definition():
                    for sub in base_decl.get_children():
                        if sub.kind == CursorKind.CXX_METHOD:
                            try:
                                if sub.access_specifier == AccessSpecifier.PUBLIC:
                                    m = _extract_method(sub, class_name)
                                    if m and m.name not in seen_methods:
                                        seen_methods.add(m.name)
                                        if m.kind == MethodKind.STATIC_METHOD:
                                            cls.static_methods.append(m)
                                        elif m.kind == MethodKind.OPERATOR:
                                            cls.operators.append(m)
                                        else:
                                            cls.methods.append(m)
                            except Exception:
                                pass
                    _extract_base_methods(base_decl, class_name, cls, seen_methods)
            except Exception:
                pass


def _extract_class(cursor: Cursor, header: str,
                   tu_root: Cursor | None = None,
                   defs: dict[str, Cursor | None] | None = None) -> ClassDecl:
    name = cursor.spelling
    cls = ClassDecl(
        name=name, base_classes=_base_names(cursor),
        is_transient_descendant=_is_transient(cursor, defs, tu_root),
        is_template=cursor.kind == CursorKind.CLASS_TEMPLATE,
        header_file=header, doc=_doc(cursor),
    )
    # Track the access level from the class's access-specifier cursors.  The
    # field-level `access_specifier` is unreliable (libclang reports e.g.
    # `Bnd_Box::Xmin`, declared under `private:`, as public), so the current
    # region is maintained from CXX_ACCESS_SPEC_DECL decls, which are exact.
    current_access = ("public" if cursor.kind == CursorKind.STRUCT_DECL
                      else "private")

    for child in cursor.get_children():
        kind = child.kind
        if kind == CursorKind.CXX_ACCESS_SPEC_DECL:
            current_access = str(child.access_specifier).replace(
                "AccessSpecifier.", "").lower()
        if kind == CursorKind.DESTRUCTOR:
            if child.access_specifier != AccessSpecifier.PUBLIC:
                cls.has_protected_dtor = True
        if kind == CursorKind.CONSTRUCTOR:
            cls.has_any_ctor = True
            if child.access_specifier != AccessSpecifier.PUBLIC:
                cls.has_any_nonpublic_ctor = True
            else:
                cls.has_any_public_ctor = True
            ctor = _extract_constructor(child, name)
            if ctor:
                cls.constructors.append(ctor)
                if (len(ctor.parameters) == 0
                        or all(p.default_value is not None
                                for p in ctor.parameters)):
                    cls.has_public_default_ctor = True
        elif kind == CursorKind.CXX_METHOD:
            if child.spelling in ("operator new", "operator delete",
                                  "operator new[]", "operator delete[]"):
                cls.has_operator_new_delete = True
            if child.spelling == "operator new":
                cls.has_operator_new = True
                params = _params(child)
                if len(params) == 1 and params[0].type.canonical_spelling in (
                        "unsigned long", "unsigned long long", "size_t", "unsigned int"):
                    cls.has_plain_operator_new = True
            method = _extract_method(child, name)
            if method:
                if method.kind == MethodKind.STATIC_METHOD:
                    cls.static_methods.append(method)
                elif method.kind == MethodKind.OPERATOR:
                    cls.operators.append(method)
                else:
                    cls.methods.append(method)
        elif kind == CursorKind.FIELD_DECL:
            is_public = current_access == "public"
            cls.fields.append(FieldDecl(name=child.spelling,
                                        type=make_type(child.type),
                                        doc=_doc(child), is_public=is_public,
                                        is_const=bool(child.type.is_const_qualified())))
        elif kind == CursorKind.VAR_DECL:
            # A VarDecl directly under a class definition is a static data member.
            cls.static_constants.append(child.spelling)
        elif kind == CursorKind.ENUM_DECL and child.is_definition():
            enum = _extract_enum(child, header, parent=name)
            if enum is not None:
                cls.nested_enums.append(enum)

    seen = {m.name for m in cls.methods + cls.static_methods + cls.operators}
    _extract_base_methods(cursor, name, cls, seen)
    cls.has_pure_virtual = any(m.is_pure_virtual for m in cls.methods)
    cls.has_copy_assignment = not (
        _has_explicit_noncopyable(cursor)
        or any("unique_ptr" in f.type.base_name for f in cls.fields)
        or any(_field_has_deleted_copy_assignment(f) for f in cls.fields))
    if not cls.has_any_ctor and not cls.has_public_default_ctor:
        # No declared ctor: the implicit default ctor is deleted when a direct
        # base or a data member blocks it (libclang reports nothing for a
        # deleted implicit default ctor).  Without this the symbol audit is the
        # only thing catching it, and cross-target scans skip the audit.
        cls.has_usable_implicit_default_ctor = _class_implicit_default_usable(
            cursor, tu_root, defs)
    try:
        cls.is_abstract = bool(cursor.is_abstract_record())
    except Exception:
        pass
    return cls


@dataclass
class HeaderResult:
    header: str
    classes: list[ClassDecl] = field(default_factory=list)
    enums: list[EnumDecl] = field(default_factory=list)
    typedefs: list[tuple[str, str]] = field(default_factory=list)
    # OCCT headers that had to be pre-included for this header to parse at all
    # (closure + retry fixes); wrappers of its classes must include them too.
    extra_includes: list[str] = field(default_factory=list)


def extract_header(header: Path, tu: TranslationUnit) -> HeaderResult:
    """Extract declarations DEFINED in the given header file."""
    header = str(header)
    result = HeaderResult(header=header)

    def is_ort_name(name: str) -> bool:
        if not name or name.startswith("_") or name in ("detail", "Base", "BaseMemoryInfo", "ConstSessionOptions"):
            return False
        return not (name.islower() or name.isdigit())

    def walk(cursor: Cursor, namespace: tuple[str, ...] = ()):
        for child in cursor.get_children():
            if child.kind == CursorKind.NAMESPACE:
                walk(child, namespace + (child.spelling,))
                continue
            try:
                if child.location.file is None or child.location.file.name != header:
                    continue
            except Exception:
                continue
            in_ort_scope = (not namespace) or (namespace == ("Ort",))
            if child.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
                if child.is_definition() and in_ort_scope \
                        and is_ort_name(child.spelling):
                    try:
                        is_specialization = child.specialized_template is not None
                    except Exception:
                        is_specialization = False
                    if is_specialization:
                        continue  # class template specialization
                    cls_decl = _extract_class(child, header, tu.cursor, defs)
                    if namespace == ("Ort",):
                        cls_decl.cpp_qual_name = f"::Ort::{child.spelling}"
                    else:
                        cls_decl.cpp_qual_name = f"::{child.spelling}"
                    result.classes.append(cls_decl)
            elif child.kind == CursorKind.ENUM_DECL and child.is_definition() \
                    and in_ort_scope and is_ort_name(child.spelling):
                enum = _extract_enum(child, header)
                if enum is not None:
                    result.enums.append(enum)
            elif child.kind == CursorKind.TYPEDEF_DECL and in_ort_scope:
                t = make_type(child.underlying_typedef_type)
                if t.is_enum or t.is_handle or t.is_collection or (
                        t.base_name and t.base_name[0].isupper()):
                    result.typedefs.append((child.spelling, t.base_name))

    defs: dict[str, Cursor | None] = {}
    walk(tu.cursor)
    return result
