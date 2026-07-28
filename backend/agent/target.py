from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


class TargetDefinitionError(ValueError):
    """Raised when an output model cannot describe a collection target."""


@dataclass(frozen=True)
class TargetField:
    name: str
    description: str
    required: bool


@dataclass(frozen=True)
class TargetContext:
    name: str
    description: str | None
    fields: tuple[TargetField, ...]

    @classmethod
    def from_model(cls, model: type[BaseModel]) -> TargetContext:
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TargetDefinitionError(
                "output_type must be a pydantic BaseModel subclass"
            )

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

        description = model.__doc__.strip() if model.__doc__ else None
        return cls(
            name=model.__name__,
            description=description,
            fields=tuple(
                TargetField(
                    name=name,
                    description=field.description.strip(),
                    required=field.is_required(),
                )
                for name, field in model.model_fields.items()
                if field.description is not None
            ),
        )
