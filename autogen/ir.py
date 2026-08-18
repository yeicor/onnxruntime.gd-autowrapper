"""Reconstruct the dataclass IR from scan JSON (out/ir/*.json)."""

from __future__ import annotations

import json
from pathlib import Path

from . import model as M


def _otype(d: dict | None) -> M.OCCTType | None:
    if not d:
        return None
    return M.OCCTType(**d)


def _param(d: dict) -> M.Parameter:
    return M.Parameter(type=_otype(d.get("type")), name=d.get("name", ""),
                       default_value=d.get("default_value"))


def _mdecl(d: dict) -> M.MethodDecl:
    return M.MethodDecl(
        name=d["name"],
        return_type=_otype(d.get("return_type")),
        parameters=[_param(p) for p in d.get("parameters", [])],
        kind=M.MethodKind(d["kind"]),
        is_const=d.get("is_const", False),
        is_virtual=d.get("is_virtual", False),
        is_static=d.get("is_static", False),
        is_default=d.get("is_default", False),
        is_deleted=d.get("is_deleted", False),
        is_pure_virtual=d.get("is_pure_virtual", False),
        is_variadic=d.get("is_variadic", False),
        is_overload=d.get("is_overload", False),
        overload_index=d.get("overload_index", 0),
        overload_suffix=d.get("overload_suffix", ""),
        operator_type=(M.OperatorType(d["operator_type"])
                       if d.get("operator_type") else None),
        doc=M.DocBlock(**(d.get("doc") or {})),
        skip=d.get("skip", False),
        skip_reason=d.get("skip_reason", ""),
    )


def _enum_value(d: dict) -> M.EnumValue:
    return M.EnumValue(name=d["name"], value=d.get("value"))


def _enum(d: dict) -> M.EnumDecl:
    return M.EnumDecl(
        name=d["name"],
        values=[_enum_value(v) for v in d.get("values", [])],
        is_scoped=d.get("is_scoped", False),
        is_nested=d.get("is_nested", False),
        parent_class=d.get("parent_class", ""),
        is_public=d.get("is_public", True),
        header_file=d.get("header_file", ""),
        doc=M.DocBlock(**(d.get("doc") or {})),
    )


def _field(d: dict) -> M.FieldDecl:
    return M.FieldDecl(name=d["name"], type=_otype(d.get("type")),
                       doc=M.DocBlock(**(d.get("doc") or {})),
                       is_public=d.get("is_public", True),
                       is_const=d.get("is_const", False))


def _class(d: dict) -> M.ClassDecl:
    return M.ClassDecl(
        name=d["name"],
        cpp_qual_name=d.get("cpp_qual_name", ""),
        wrapper_name=d.get("wrapper_name", ""),
        module_name=d.get("module_name", ""),
        base_classes=list(d.get("base_classes", [])),
        kind=M.ClassKind(d.get("kind", "other")),
        is_transient_descendant=d.get("is_transient_descendant", False),
        constructors=[_mdecl(c) for c in d.get("constructors", [])],
        methods=[_mdecl(c) for c in d.get("methods", [])],
        operators=[_mdecl(c) for c in d.get("operators", [])],
        static_methods=[_mdecl(c) for c in d.get("static_methods", [])],
        fields=[_field(c) for c in d.get("fields", [])],
        static_constants=list(d.get("static_constants", [])),
        nested_enums=[_enum(c) for c in d.get("nested_enums", [])],
        header_file=d.get("header_file", ""),
        doc=M.DocBlock(**(d.get("doc") or {})),
        extra_occt_includes=list(d.get("extra_occt_includes", [])),
        has_public_default_ctor=d.get("has_public_default_ctor", False),
        has_usable_implicit_default_ctor=d.get(
            "has_usable_implicit_default_ctor", True),
        has_any_ctor=d.get("has_any_ctor", False),
        has_any_public_ctor=d.get("has_any_public_ctor", False),
        has_any_nonpublic_ctor=d.get("has_any_nonpublic_ctor", False),
        has_protected_dtor=d.get("has_protected_dtor", False),
        is_template=d.get("is_template", False),
        has_pure_virtual=d.get("has_pure_virtual", False),
        is_abstract=d.get("is_abstract", False),
        has_copy_assignment=d.get("has_copy_assignment", True),
        has_operator_new_delete=d.get("has_operator_new_delete", False),
        has_operator_new=d.get("has_operator_new", False),
        has_plain_operator_new=d.get("has_plain_operator_new", False),
        default_constructible=d.get("default_constructible", None),
        returnable=d.get("returnable", True),
        skip=d.get("skip", False),
        skip_reason=d.get("skip_reason", ""),
    )


def load_module(path: Path) -> M.ModuleDecl:
    data = json.loads(path.read_text())
    return M.ModuleDecl(
        name=data.get("module", path.stem),
        classes=[_class(c) for c in data.get("classes", [])],
        enums=[_enum(e) for e in data.get("enums", [])],
        data_model=data.get("data_model") or {},
    )


def _otype_out(t: M.OCCTType) -> dict:
    return {k: v for k, v in t.__dict__.items() if v not in ("", [], None)}


def _param_out(p: M.Parameter) -> dict:
    d = {"name": p.name, "type": _otype_out(p.type)}
    if p.default_value:
        d["default_value"] = p.default_value
    return d


def _mdecl_out(m: M.MethodDecl) -> dict:
    d = {
        "name": m.name,
        "parameters": [_param_out(p) for p in m.parameters],
        "kind": m.kind.value,
        "is_const": m.is_const,
        "is_virtual": m.is_virtual,
        "is_static": m.is_static,
        "is_default": m.is_default,
        "is_deleted": m.is_deleted,
        "is_pure_virtual": m.is_pure_virtual,
        "is_variadic": m.is_variadic,
        "is_overload": m.is_overload,
        "overload_index": m.overload_index,
        "overload_suffix": m.overload_suffix,
    }
    if m.return_type:
        d["return_type"] = _otype_out(m.return_type)
    if m.operator_type:
        d["operator_type"] = m.operator_type.value
    if m.doc.brief or m.doc.raw:
        d["doc"] = {k: v for k, v in m.doc.__dict__.items() if v not in ("", [], {})}
    if m.skip:
        d["skip"] = True
        d["skip_reason"] = m.skip_reason
    return d


def _enum_out(e: M.EnumDecl) -> dict:
    d = {
        "name": e.name,
        "values": [{"name": v.name, "value": v.value} for v in e.values],
        "is_scoped": e.is_scoped,
        "is_nested": e.is_nested,
        "parent_class": e.parent_class,
        "is_public": e.is_public,
        "header_file": e.header_file,
    }
    if e.doc.brief or e.doc.raw:
        d["doc"] = {k: v for k, v in e.doc.__dict__.items() if v not in ("", [], {})}
    return d


def _field_out(f: M.FieldDecl) -> dict:
    d = {"name": f.name, "type": _otype_out(f.type),
         "is_public": f.is_public, "is_const": f.is_const}
    if f.doc.brief or f.doc.raw:
        d["doc"] = {k: v for k, v in f.doc.__dict__.items() if v not in ("", [], {})}
    return d


def _class_out(c: M.ClassDecl) -> dict:
    d = {
        "name": c.name,
        "cpp_qual_name": c.cpp_qual_name,
        "wrapper_name": c.wrapper_name,
        "module_name": c.module_name,
        "base_classes": c.base_classes,
        "kind": c.kind.value,
        "is_transient_descendant": c.is_transient_descendant,
        "constructors": [_mdecl_out(x) for x in c.constructors],
        "methods": [_mdecl_out(x) for x in c.methods],
        "operators": [_mdecl_out(x) for x in c.operators],
        "static_methods": [_mdecl_out(x) for x in c.static_methods],
        "fields": [_field_out(x) for x in c.fields],
        "static_constants": c.static_constants,
        "nested_enums": [_enum_out(x) for x in c.nested_enums],
        "header_file": c.header_file,
        "extra_occt_includes": c.extra_occt_includes,
        "has_public_default_ctor": c.has_public_default_ctor,
        "has_usable_implicit_default_ctor": c.has_usable_implicit_default_ctor,
        "has_any_ctor": c.has_any_ctor,
        "has_any_public_ctor": c.has_any_public_ctor,
        "has_any_nonpublic_ctor": c.has_any_nonpublic_ctor,
        "has_protected_dtor": c.has_protected_dtor,
        "is_template": c.is_template,
        "has_pure_virtual": c.has_pure_virtual,
        "is_abstract": c.is_abstract,
        "has_copy_assignment": c.has_copy_assignment,
        "has_operator_new_delete": c.has_operator_new_delete,
        "has_operator_new": c.has_operator_new,
        "has_plain_operator_new": c.has_plain_operator_new,
        "default_constructible": c.default_constructible,
        "returnable": c.returnable,
        "skip": c.skip,
        "skip_reason": c.skip_reason,
    }
    if c.doc.brief or c.doc.raw:
        d["doc"] = {k: v for k, v in c.doc.__dict__.items() if v not in ("", [], {})}
    return d


def dump_module(module: M.ModuleDecl) -> dict:
    """Serialize a ModuleDecl back to the scan-JSON shape (round-trips with
    ``load_module``).  Used to cache synthesized specializations."""
    return {
        "module": module.name,
        "classes": [_class_out(c) for c in module.classes],
        "enums": [_enum_out(e) for e in module.enums],
        "data_model": module.data_model,
    }
