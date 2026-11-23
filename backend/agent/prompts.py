import json
from functools import partial
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptRenderer:
    def __init__(self, directory: Path | None = None) -> None:
        prompt_directory = directory or Path(__file__).with_name("instructions")
        self.environment = Environment(
            loader=FileSystemLoader(prompt_directory),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do"],
        )
        # Jinja `tojson` escapes some characters we want to include
        self.environment.filters["tojson"] = partial(json.dumps, ensure_ascii=False)

    def render(self, template: str, **context: Any) -> str:
        return self.environment.get_template(template).render(**context).strip()

