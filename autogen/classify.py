"""Classify scanned OCCT classes into wrapping strategies.

Structural rules only (no source-text heuristics).  The kinds mirror the
legacy pipeline so the generated API contract stays identical:

1. Standard_Transient/Standard_Persistent descendants        -> REF_COUNTED
2. BRepBuilderAPI_MakeShape/BRep*API_Command descendants     -> BUILDER
3. classes owning an occ::handle<TopoDS_TShape> field        -> TOPODS_SHAPE
4. no bases, not transient                                   -> VALUE
5. everything else                                           -> OTHER (wrapped as value)

Class-level skips mirror the legacy extraction rules (occast/classes.py):
template classes, protected destructors (non-transient), classes with no
public constructors (non-transient), abstract non-transient classes,
Standard_* exception hierarchies, and internal TopoDS_T* implementations.
"""

from __future__ import annotations

import logging

from .model import ClassDecl, ClassKind, ModuleDecl

log = logging.getLogger("autogen.classify")

_BUILDER_BASES = {"BRepBuilderAPI_MakeShape", "BRepBuilderAPI_Command"}
_BUILDER_PREFIXES = ("BRepPrimAPI_", "BRepBuilderAPI_", "BRepFilletAPI_",
                     "BRepOffsetAPI_", "BRepFeat_", "BRepAlgoAPI_")
_EXCEPTION_ROOT = "Standard_Failure"


def classify_kind(cls: ClassDecl) -> ClassKind:
    if cls.is_transient_descendant or cls.name in ("Standard_Transient",
                                                   "Standard_Persistent"):
        return ClassKind.REF_COUNTED
    for base in cls.base_classes:
        if base in _BUILDER_BASES or base.startswith(_BUILDER_PREFIXES):
            return ClassKind.BUILDER
    for f in cls.fields:
        if f.type.is_handle and "TopoDS_TShape" in f.type.handle_inner:
            return ClassKind.TOPODS_SHAPE
    if not cls.base_classes:
        return ClassKind.VALUE
    return ClassKind.OTHER


def _is_failure_descendant(cls: ClassDecl, by_name: dict[str, ClassDecl],
                           seen: set[str]) -> bool:
    if cls.name in seen:
        return False
    seen.add(cls.name)
    for base in cls.base_classes:
        if base == _EXCEPTION_ROOT:
            return True
        parent = by_name.get(base)
        if parent is not None and _is_failure_descendant(parent, by_name, seen):
            return True
    return False


def _has_custom_alloc(cls: ClassDecl, by_name: dict[str, ClassDecl],
                      seen: set[str]) -> bool:
    """True if cls or any (module-local) base declares operator new/delete."""
    if cls.name in seen:
        return False
    seen.add(cls.name)
    if cls.has_operator_new_delete:
        return True
    for base in cls.base_classes:
        b = by_name.get(base)
        if b is not None and _has_custom_alloc(b, by_name, seen):
            return True
    return False


def _is_allocator_managed(cls: ClassDecl, by_name: dict[str, ClassDecl],
                          seen: set[str]) -> bool:
    """True if constructing `new Cls()` resolves to an allocator-tagged
    operator new instead of the plain `operator new(size_t)`.

    DEFINE_INC_ALLOC / DEFINE_NCOLLECTION_ALLOC declare only
    `operator new(size_t, const NCollection_BaseAllocator&)`-style overloads,
    which hide the plain form, so a handle-allocating `new Cls()` is a compile
    error.  If the class itself declares any operator new, base overloads are
    hidden and only its own forms matter; otherwise the nearest base declaration
    applies.
    """
    if cls.name in seen:
        return False
    seen.add(cls.name)
    if cls.has_operator_new:
        return not cls.has_plain_operator_new
    for base in cls.base_classes:
        b = by_name.get(base)
        if b is not None and _is_allocator_managed(b, by_name, seen):
            return True
    return False


def _skip_reason(cls: ClassDecl, by_name: dict[str, ClassDecl]) -> str:
    """Legacy class-level skip rules; "" means the class is wrapped.

    Exception detection is structural: any class whose base chain reaches
    Standard_Failure is classified EXCEPTION (wrapped as a diagnostics-only
    class preserving the hierarchy), checked *before* the constructor/
    allocation rules so the whole OCCT exception hierarchy is classified
    uniformly (Standard_Mutex, whose base happens to contain "Error" as a
    nested type name, is not an exception and stays wrapped).
    """
    if cls.kind == ClassKind.EXCEPTION:
        return ""  # wrapped as a diagnostics-only hierarchy (see codegen)
    if cls.name == cls.module_name:
        return ""  # module aggregate host (e.g. Standard, gp): keep it
    if cls.is_template:
        return "template class"
    if cls.kind != ClassKind.REF_COUNTED:
        if cls.has_pure_virtual:
            return "abstract (pure virtual) class"
        # Classes exposing only static methods carry no native storage, so the
        # destructor/constructor/allocation requirements below do not apply
        # (e.g. BRep_Tool declares no public ctor and custom new/delete).
        if cls.static_methods and not cls.methods and not cls.operators and not cls.fields:
            pass
        elif cls.has_protected_dtor:
            return "protected destructor"
        elif cls.has_any_nonpublic_ctor and not cls.has_any_public_ctor:
            return "no public constructors"
    elif _is_allocator_managed(cls, by_name, set()):
        # Ref-counted classes construct via `new Cls(...)`, which needs the
        # plain `operator new(size_t)`; allocator-tagged-only classes
        # (DEFINE_INC_ALLOC/DEFINE_NCOLLECTION_ALLOC) cannot be handle-built.
        return "custom allocation (operator new/delete)"
    for base in cls.base_classes:
        if base.startswith("TopoDS_T") and base != "TopoDS_TShape":
            return "internal TopoDS shape implementation"
    return ""


def classify_module(module: ModuleDecl,
                    global_by_name: dict[str, ClassDecl] | None = None) -> None:
    """Set kind/wrapper_name/skip for every class in the module, in-place.

    `global_by_name` (name -> ClassDecl across *all* modules) lets exception
    detection and custom-allocation probing follow base classes that live in
    other modules (e.g. Geom_UndefinedDerivative -> Standard_DomainError).
    """
    by_name: dict[str, ClassDecl] = (
        global_by_name if global_by_name is not None
        else {c.name: c for c in module.classes})

    for cls in module.classes:
        cls.kind = classify_kind(cls)
        cls.wrapper_name = occt_wrapper_name(cls.name, cls.module_name)
        if cls.name == _EXCEPTION_ROOT \
                or _is_failure_descendant(cls, by_name, set()):
            cls.kind = ClassKind.EXCEPTION

    reasons: dict[str, str] = {}
    for cls in module.classes:
        reason = _skip_reason(cls, by_name)
        if reason:
            reasons[cls.name] = reason

    for cls in module.classes:
        if cls.name in reasons:
            cls.skip = True
            cls.skip_reason = reasons[cls.name]
            cls.kind = ClassKind.OTHER
            log.info("skip %s: %s", cls.name, reasons[cls.name])


def occt_wrapper_name(occt_name: str, module_name: str) -> str:
    """Ort prefix + camelized name (module aggregate keeps its plain name)."""
    if occt_name == module_name:
        return f"Ort{occt_name}"
    from .model import occt_name_to_wrapper
    return occt_name_to_wrapper(occt_name, module_name)
