from __future__ import annotations

from pydantic import BaseModel

from .schema_doc import ModelDoc, describe_model


class TargetDefinitionError(ValueError):
    """Raised when an output model cannot describe a collection target."""


def describe_target(model: type[BaseModel]) -> ModelDoc:
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TargetDefinitionError("output_type must be a pydantic BaseModel subclass")

    missing_descriptions = [
        name
        for name, field in model.model_fields.items()
        if not field.description or not field.description.strip()
    ]
    if missing_descriptions:
        missing = ", ".join(missing_descriptions)
        raise TargetDefinitionError(
            f"{model.__name__} fields require descriptions: {missing}"
        )

    return describe_model(model)
