"""Normalise a Pydantic JSON Schema into a wording-free tree for prompts.

This module owns schema *semantics*: resolving ``$ref``, collapsing nullable
unions, detecting recursion and reuse, and accounting for every keyword.
Turning the tree into English is the job of ``instructions/_schema.md.j2``.

Anything Pydantic can express in JSON Schema is either understood here or
carried verbatim in ``TypeDoc.extra`` so the template can still show it.
Validators and other Python-only behaviour never reach JSON Schema; describe
those in the field's ``description``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

Kind = Literal[
    "text",
    "integer",
    "number",
    "boolean",
    "null",
    "any",
    "enum",
    "const",
    "list",
    "tuple",
    "map",
    "object",
    "union",
    "all_of",
    "recursive",
    "seen",
]

# Keywords that carry no meaning for the person or model filling the target.
IGNORED_KEYWORDS = frozenset({"title", "readOnly", "writeOnly", "$schema", "$id"})

CONSTRAINT_KEYWORDS = {
    "minLength": "min_length",
    "maxLength": "max_length",
    "pattern": "pattern",
    "minimum": "minimum",
    "maximum": "maximum",
    "exclusiveMinimum": "exclusive_minimum",
    "exclusiveMaximum": "exclusive_maximum",
    "multipleOf": "multiple_of",
    "minItems": "min_items",
    "maxItems": "max_items",
    "uniqueItems": "unique_items",
    "minProperties": "min_properties",
    "maxProperties": "max_properties",
}

_MISSING: Any = object()


@dataclass(frozen=True)
class ModelDoc:
    name: str
    description: str | None
    fields: tuple[FieldDoc, ...]
    # None: unspecified (extra input ignored); False: forbidden; TypeDoc: allowed
    extra_fields: TypeDoc | Literal[False] | None = None


@dataclass(frozen=True)
class TypeDoc:
    kind: Kind
    # Name of a named type (model, enum) when the schema provides one.
    name: str | None = None
    # Description of the named type itself (model or enum docstring).
    summary: str | None = None
    # Description attached where this type is used (e.g. a field).
    description: str | None = None
    format: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    values: tuple[Any, ...] = ()
    items: TypeDoc | None = None
    keys: TypeDoc | None = None
    prefix: tuple[TypeDoc, ...] = ()
    options: tuple[TypeDoc, ...] = ()
    discriminator: str | None = None
    model: ModelDoc | None = None
    content: TypeDoc | None = None
    content_media_type: str | None = None
    nullable: bool = False
    default: Any = _MISSING
    examples: tuple[Any, ...] = ()
    deprecated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    @property
    def models(self) -> tuple[ModelDoc, ...]:
        """Models to expand beneath this type, outermost first, without
        descending into another model's own fields."""
        if self.kind == "object" and self.model is not None:
            return (self.model,)
        found: list[ModelDoc] = []
        for child in (
            self.items,
            self.keys,
            self.content,
            *self.prefix,
            *self.options,
        ):
            if child is not None:
                found.extend(child.models)
        return tuple(found)


@dataclass(frozen=True)
class FieldDoc:
    name: str
    required: bool
    type: TypeDoc

    @property
    def description(self) -> str | None:
        return self.type.description


class SchemaDocError(ValueError):
    """Raised when a schema cannot be described."""


def describe_model(model: type[BaseModel]) -> ModelDoc:
    return describe_schema(model.model_json_schema())


def describe_schema(schema: dict[str, Any]) -> ModelDoc:
    root = _Walker(schema.get("$defs", {})).walk(
        {k: v for k, v in schema.items() if k != "$defs"}, stack=()
    )
    if root.kind != "object" or root.model is None:
        raise SchemaDocError("root schema must describe an object with fields")
    from dataclasses import replace

    return replace(root.model, description=root.description)


class _Walker:
    def __init__(self, defs: dict[str, Any]) -> None:
        self.defs = defs
        self.expanded: set[str] = set()

    def walk(self, schema: dict[str, Any] | bool, stack: tuple[str, ...]) -> TypeDoc:
        if schema is True or schema == {}:
            return TypeDoc(kind="any")
        if schema is False:
            raise SchemaDocError("a schema of `false` accepts no value")
        s = dict(schema)
        for key in IGNORED_KEYWORDS & s.keys():
            if key != "title":
                s.pop(key)
        annotations = self._annotations(s)

        if "$ref" in s:
            doc = self._ref(s.pop("$ref"), stack)
        elif "allOf" in s:
            parts = s.pop("allOf")
            if len(parts) == 1:
                doc = self.walk(parts[0], stack)
            else:
                doc = TypeDoc(
                    kind="all_of",
                    options=tuple(self.walk(p, stack) for p in parts),
                )
        elif _is_decimal(s):
            s.pop("anyOf")
            doc = TypeDoc(kind="number", format="decimal")
        elif "anyOf" in s or "oneOf" in s:
            doc = self._union(s, stack)
        elif isinstance(s.get("type"), list):
            types = s.pop("type")
            doc = self._union(
                {"anyOf": [{**s, "type": t} for t in types]}, stack
            )
            s = {}
        elif "const" in s:
            s.pop("type", None)
            doc = TypeDoc(kind="const", values=(s.pop("const"),))
        elif "enum" in s:
            s.pop("type", None)
            doc = TypeDoc(kind="enum", values=tuple(s.pop("enum")))
        else:
            doc = self._typed(s, stack)

        s.pop("title", None)
        constraints = {
            name: s.pop(key) for key, name in CONSTRAINT_KEYWORDS.items() if key in s
        }
        if doc.kind == "tuple":
            n = len(doc.prefix)
            if doc.items is None and constraints.get("min_items") == n:
                constraints.pop("min_items")
            if doc.items is None and constraints.get("max_items") == n:
                constraints.pop("max_items")
        if doc.kind == "text" and doc.format == "uri":
            # Pydantic's URL length bounds are implementation limits, not guidance.
            constraints.pop("min_length", None)
            constraints.pop("max_length", None)

        return _replace(
            doc,
            constraints={**doc.constraints, **constraints},
            extra={**doc.extra, **s},
            **{k: v for k, v in annotations.items() if v is not _MISSING},
        )

    @staticmethod
    def _annotations(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "description": s.pop("description", _MISSING),
            "default": s.pop("default", _MISSING),
            "examples": tuple(s.pop("examples")) if "examples" in s else _MISSING,
            "deprecated": s.pop("deprecated", _MISSING),
        }

    def _ref(self, ref: str, stack: tuple[str, ...]) -> TypeDoc:
        prefix = "#/$defs/"
        if not ref.startswith(prefix) or ref[len(prefix) :] not in self.defs:
            raise SchemaDocError(f"cannot resolve reference {ref!r}")
        name = ref[len(prefix) :]
        target = self.defs[name]
        title = target.get("title", name)
        is_model = target.get("type") == "object" and "properties" in target
        if name in stack:
            return TypeDoc(kind="recursive", name=title)
        if is_model and name in self.expanded:
            return TypeDoc(kind="seen", name=title)
        if is_model:
            self.expanded.add(name)
        doc = self.walk(target, stack + (name,))
        # The definition's description describes the type, not this use of it.
        return _replace(doc, name=title, summary=doc.description, description=None)

    def _union(self, s: dict[str, Any], stack: tuple[str, ...]) -> TypeDoc:
        branches = s.pop("anyOf", None) or s.pop("oneOf")
        discriminator = s.pop("discriminator", None)
        options = [self.walk(b, stack) for b in branches]
        nullable = any(o.kind == "null" for o in options)
        options = [o for o in options if o.kind != "null"]
        if not options:
            return TypeDoc(kind="null")
        if len(options) == 1 and discriminator is None:
            return _replace(options[0], nullable=nullable or options[0].nullable)
        return TypeDoc(
            kind="union",
            options=tuple(options),
            nullable=nullable,
            discriminator=(discriminator or {}).get("propertyName"),
        )

    def _typed(self, s: dict[str, Any], stack: tuple[str, ...]) -> TypeDoc:
        kind = s.pop("type", None)
        if kind is None:
            return TypeDoc(kind="any")
        if kind == "string":
            content = None
            if "contentSchema" in s:
                content = self.walk(s.pop("contentSchema"), stack)
            return TypeDoc(
                kind="text",
                format=s.pop("format", None),
                content=content,
                content_media_type=s.pop("contentMediaType", None),
            )
        if kind in ("integer", "number", "boolean", "null"):
            return TypeDoc(kind=kind, format=s.pop("format", None))
        if kind == "array":
            if "prefixItems" in s:
                prefix = tuple(self.walk(p, stack) for p in s.pop("prefixItems"))
                rest = s.pop("items", None)
                return TypeDoc(
                    kind="tuple",
                    prefix=prefix,
                    items=None if rest in (None, False) else self.walk(rest, stack),
                )
            if "items" not in s and s.get("maxItems") == 0:
                return TypeDoc(kind="tuple")  # `tuple[()]`
            items = s.pop("items", {})
            return TypeDoc(kind="list", items=self.walk(items, stack))
        if kind == "object":
            return self._object(s, stack)
        raise SchemaDocError(f"unknown JSON Schema type {kind!r}")

    def _object(self, s: dict[str, Any], stack: tuple[str, ...]) -> TypeDoc:
        additional = s.pop("additionalProperties", None)
        keys = None
        if "propertyNames" in s:
            keys = self.walk(s.pop("propertyNames"), stack)
        values = (
            None
            if additional in (None, False)
            else self.walk(True if additional is True else additional, stack)
        )
        if "properties" not in s:
            return TypeDoc(
                kind="map",
                keys=keys or TypeDoc(kind="text"),
                items=values or TypeDoc(kind="any"),
            )
        required = set(s.pop("required", ()))
        fields = tuple(
            FieldDoc(name=name, required=name in required, type=self.walk(prop, stack))
            for name, prop in s.pop("properties").items()
        )
        model = ModelDoc(
            name=s.get("title", "object"),
            # Filled in by the caller: `walk` has already lifted the description.
            description=None,
            fields=fields,
            extra_fields=False if additional is False else values,
        )
        return TypeDoc(kind="object", name=model.name, model=model)


_DECIMAL_PATTERN = r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$"


def _is_decimal(s: dict[str, Any]) -> bool:
    """Pydantic renders `Decimal` as "number, or a string of digits"."""
    branches = s.get("anyOf")
    return (
        isinstance(branches, list)
        and len(branches) == 2
        and {"type": "number"} in branches
        and {"type": "string", "pattern": _DECIMAL_PATTERN} in branches
    )


def _replace(doc: TypeDoc, **changes: Any) -> TypeDoc:
    from dataclasses import replace

    if doc.model is not None and ("name" in changes or "summary" in changes):
        model = replace(
            doc.model,
            name=changes.get("name", doc.model.name),
            description=changes.get("summary", doc.model.description),
        )
        changes = {**changes, "model": model}
    return replace(doc, **changes)
