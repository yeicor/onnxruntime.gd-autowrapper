"""Data model for parsed OpenCASCADE API declarations.

Every declaration extracted by the AST parser is represented as a dataclass here.
The generator consumes these models to produce godot-cpp wrapper code.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClassKind(enum.Enum):
    """How the OCCT class should be wrapped."""
    VALUE = "value"              # Plain C++ value type (gp_Pnt, gp_Dir, ...)
    REF_COUNTED = "ref_counted"  # Standard_Transient + Handle<T>
    TOPODS_SHAPE = "topods_shape"  # TopoDS_Shape and subtypes
    BUILDER = "builder"          # BRepBuilderAPI_MakeShape descendants
    EXCEPTION = "exception"      # Standard_Failure hierarchy (diagnostics-only)
    OTHER = "other"              # Anything else (opaque, skipped or wrapped minimally)


class MethodKind(enum.Enum):
    CONSTRUCTOR = "constructor"
    METHOD = "method"
    STATIC_METHOD = "static_method"
    OPERATOR = "operator"


class OperatorType(enum.Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    LESS = "<"
    GREATER = ">"
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    CROSS = "^"
    PLUS_ASSIGN = "+="
    MINUS_ASSIGN = "-="
    MULTIPLY_ASSIGN = "*="
    DIVIDE_ASSIGN = "/="
    CROSS_ASSIGN = "^="
    UNARY_MINUS = "unary_minus"
    UNARY_PLUS = "unary_plus"
    DEREFERENCE = "*deref"
    CALL = "call"
    INCREMENT = "++"
    DECREMENT = "--"


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

@dataclass
class DocBlock:
    """Extracted documentation from an OCCT declaration."""
    brief: str = ""                 # From cursor.brief_comment
    raw: str = ""                   # From cursor.raw_comment (full OCCT doc)
    params: dict[str, str] = field(default_factory=dict)  # @param name desc
    returns: str = ""               # @return description
    notes: list[str] = field(default_factory=list)  # @note, warnings, etc.


# ---------------------------------------------------------------------------
# Type representation
# ---------------------------------------------------------------------------

@dataclass
class OCCTType:
    """Represents an OCCT C++ type with full qualification info."""
    spelling: str                   # Raw clang spelling (e.g. "const gp_Pnt &")
    base_name: str                  # Clean base name (e.g. "gp_Pnt")
    canonical_spelling: str = ""    # libclang canonical type (fully desugared)
    is_const: bool = False
    is_ref: bool = False            # &
    is_rvalue_ref: bool = False     # &&
    is_pointer: bool = False        # *
    is_handle: bool = False         # occ::handle<T> / opencascade::handle<T>
    handle_inner: str = ""          # T if is_handle
    is_transient_descendant: bool = False  # Inherits from Standard_Transient
    pointee_is_const: bool = False  # For pointers: is the pointee const-qualified?
                                    # ("const void*" → True; "void* const" → False)
    pointee_pointee_is_const: bool = False  # For ref/pointer-to-pointer: is the
                                    # pointee a pointer and its own pointee const?
                                    # ("const char*&" → True; "char*&" → False)
    is_enum: bool = False           # resolves to an enum type
    template_args: list[str] = field(default_factory=list)  # top-level template args (spellings)

    @property
    def is_void(self) -> bool:
        return self.base_name == "void"

    @property
    def is_primitive(self) -> bool:
        return self.base_name in (
            "void", "bool", "char", "unsigned char", "signed char",
            "short", "unsigned short", "int", "unsigned int",
            "long", "unsigned long", "long long", "unsigned long long",
            "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
            "float", "double", "long double",
            "Standard_Boolean", "Standard_Character", "Standard_Byte",
            "Standard_Integer", "Standard_Real", "Standard_ShortReal",
            "Standard_CString",
        )

    @property
    def is_string(self) -> bool:
        return self.base_name in ("Standard_CString", "TCollection_AsciiString",
                                   "TCollection_ExtendedString", "std::string")

    @property
    def is_collection(self) -> bool:
        """Type is an NCollection_* / TColStd_* / TColgp_* sequence template."""
        return any(self.base_name.startswith(p) for p in
                   ("NCollection_", "TColStd_", "TColgp_"))

    @property
    def unwrappable(self) -> bool:
        """Type that cannot be wrapped in GDScript."""
        return self.base_name in (
            "Standard_OStream", "Standard_IStream", "Standard_SStream",
            "Standard_ProgramAddress",
        ) or self.is_pointer and not self.is_handle


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class EnumValue:
    name: str
    value: int | None = None


@dataclass
class EnumDecl:
    name: str
    values: list[EnumValue] = field(default_factory=list)
    is_scoped: bool = False        # enum class vs enum
    is_nested: bool = False        # declared inside a class
    parent_class: str = ""
    is_public: bool = True         # false for private/protected nested enums
    header_file: str = ""
    doc: DocBlock = field(default_factory=DocBlock)


@dataclass
class Parameter:
    type: OCCTType
    name: str
    default_value: str | None = None


@dataclass
class MethodDecl:
    name: str
    return_type: OCCTType | None = None  # None = void
    parameters: list[Parameter] = field(default_factory=list)
    kind: MethodKind = MethodKind.METHOD
    is_const: bool = False
    is_virtual: bool = False
    is_static: bool = False
    is_default: bool = False       # = default
    is_deleted: bool = False       # = delete
    is_pure_virtual: bool = False  # = 0
    is_variadic: bool = False      # C-style variadic (...)
    is_overload: bool = False      # has same-name sibling
    overload_index: int = 0        # 0-based index among overloads
    overload_suffix: str = ""      # collision-aware hash disambiguator (set by group_overloads)
    operator_type: OperatorType | None = None
    doc: DocBlock = field(default_factory=DocBlock)
    skip: bool = False             # True if unwrappable
    skip_reason: str = ""


@dataclass
class FieldDecl:
    name: str
    type: OCCTType
    doc: DocBlock = field(default_factory=DocBlock)
    is_public: bool = True
    is_const: bool = False          # True for `const` data members (read-only)
    skip: bool = False              # True if the field's accessors cannot be generated
    skip_reason: str = ""           # why (e.g. field type is not copyable)


@dataclass
class ClassDecl:
    name: str
    cpp_qual_name: str = ""        # e.g. "::Ort::Env" or "::OrtEnv"
    wrapper_name: str = ""         # e.g. "OrtEnv" (set during classification)
    module_name: str = ""          # e.g. "Core" (set during scanning)
    base_classes: list[str] = field(default_factory=list)
    kind: ClassKind = ClassKind.OTHER
    is_transient_descendant: bool = False  # anywhere in hierarchy
    constructors: list[MethodDecl] = field(default_factory=list)
    methods: list[MethodDecl] = field(default_factory=list)
    operators: list[MethodDecl] = field(default_factory=list)
    static_methods: list[MethodDecl] = field(default_factory=list)
    fields: list[FieldDecl] = field(default_factory=list)
    static_constants: list[str] = field(default_factory=list)  # static constexpr member names
    nested_enums: list[EnumDecl] = field(default_factory=list)
    header_file: str = ""
    doc: DocBlock = field(default_factory=DocBlock)
    extra_occt_includes: list[str] = field(default_factory=list)  # OCCT headers this class's own header needs but doesn't include
    has_public_default_ctor: bool = False
    # For classes that declare NO ctor at all: True when the *implicit* default
    # ctor is usable (scan-time structural check).  Set False when a direct
    # base or a data member deletes it (e.g. a base template without a default
    # ctor, or a reference member); such classes must fall back to unique_ptr
    # storage on every target, not just the ones the symbol audit probes.
    has_usable_implicit_default_ctor: bool = True
    has_any_ctor: bool = False        # True if the class declares any ctor (even private)
    has_any_public_ctor: bool = False      # True if the class declares a public ctor
    has_any_nonpublic_ctor: bool = False   # True if the class declares a private/protected ctor
    has_protected_dtor: bool = False       # True if the destructor is private/protected
    is_template: bool = False              # True for (primary) template classes
    has_pure_virtual: bool = False
    is_abstract: bool = False              # cursor.is_abstract() (includes inherited pure virtuals)
    has_copy_assignment: bool = True   # False if copy assignment operator is deleted
    has_operator_new_delete: bool = False  # class declares operator new/delete (custom allocation)
    has_operator_new: bool = False         # class declares an operator new (any form)
    has_plain_operator_new: bool = False   # class declares operator new(size_t)
    # None = derive from has_public_default_ctor/has_any_ctor (implicit default
    # ctor).  False = the audit probe proved `T()` is ill-formed (e.g. a base
    # suppresses it); forces unique_ptr storage, never native `_native()`.
    default_constructible: bool | None = None
    # False = the audit probe proved the type is not copy-assignable (native
    # storage) or not copy-constructible (unique_ptr storage) as emitted by the
    # value/reference-return wrappers, e.g. implicitly deleted copy semantics
    # the extractor cannot see.  Methods returning it are dropped.
    returnable: bool = True
    skip: bool = False                 # True if the class cannot be wrapped
    skip_reason: str = ""

    @property
    def all_methods(self) -> list[MethodDecl]:
        return self.constructors + self.methods + self.operators + self.static_methods

    @property
    def all_wrappable_methods(self) -> list[MethodDecl]:
        return [m for m in self.all_methods if not m.skip]


@dataclass
class ModuleDecl:
    """All declarations extracted from a set of headers belonging to one OCCT module."""
    name: str                      # e.g. "gp", "TopoDS", "BRepPrimAPI"
    classes: list[ClassDecl] = field(default_factory=list)
    enums: list[EnumDecl] = field(default_factory=list)
    # Byte sizes of the size-sensitive builtins for the parse target
    # (compile_db.probe_data_model): keyed "long"/"unsigned long"/"pointer".
    # Empty for IRs predating the probe; consumers fall back to LP64 (long=8).
    data_model: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Naming utilities
# ---------------------------------------------------------------------------

def _sanitize_identifier(s: str) -> str:
    """Replace chars invalid in C++ identifiers (<> etc.) with underscores."""
    s = s.replace("<", "_").replace(">", "_").replace(",", "_")
    s = s.replace(" ", "_").replace("*", "_ptr")
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s


# Longest wrapper name that still leaves room for the ".hpp"/".cpp" suffix
# and the MSVC ".obj" output under Windows' 260-byte MAX_PATH. The CI
# object output prefix
# ("D:\a\OpenCASCADE.gd\OpenCASCADE.gd\vcpkg\buildtrees\gdext\arm64-windows-static-dbg\OpenCASCADE.gd.dir\Debug\")
# is 108 chars, so a 192-char name + 4-char extension = 304 bytes already
# exceeds MAX_PATH and makes MSVC report C1083 "cannot open compiler
# generated file" for the object file. Budget: name (incl. 12-char hash)
# must stay <= 148, i.e. the pre-hash base must be <= 136.
_WRAPPER_NAME_MAX = 130


def occt_name_to_wrapper(occt_name: str, module_name: str) -> str:
    """Convert an OCCT class name to a wrapper name with Ort prefix.

    Examples:
        gp_Pnt, gp              -> OrtGpPnt
        TopoDS_Shape, TopoDS    -> OrtTopoDSShape
        BRepPrimAPI_MakeBox, ... -> OrtBRepPrimAPIMakeBox
        NCollection_Array2<gp_Pnt> -> OrtNCollectionArray2_gp_Pnt
        Aspect_DisplayConnection -> OrtAspectDisplayConnection
        Geom_BSplineSurface     -> OrtGeomBSplineSurface
    """
    clean = _sanitize_identifier(occt_name)
    parts = clean.replace("::", "_").split("_")
    camel = "".join(p[:1].upper() + p[1:] if p else "" for p in parts)
    name = camel if camel.startswith("Ort") else f"Ort{camel}"
    # Nested template specializations (e.g. the 8-argument Extrema_GGExtPC)
    # can produce names far beyond the 255-byte filesystem limit; truncate
    # deterministically and disambiguate with a short hash of the full name.
    if len(name) > _WRAPPER_NAME_MAX:
        name = (name[:_WRAPPER_NAME_MAX]
                + hashlib.sha256(occt_name.encode()).hexdigest()[:12])
    return name


def wrapper_name_for_enum(enum_name: str, parent_class: str) -> str:
    """Generate a unique enum name for binding (avoids collision with nested enum names)."""
    if parent_class:
        return f"{parent_class}_{enum_name}"
    return enum_name
