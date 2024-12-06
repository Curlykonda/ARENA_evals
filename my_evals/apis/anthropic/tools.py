from typing import Type, get_args, get_origin
from pydantic import BaseModel
from inspect import isclass
from dataclasses import dataclass, asdict

from my_evals.mwe.structs import MCQ_Entry, QuestionGeneration

TOOLS = {
    "get_single_mcq_entry": {
        "name": "get_single_mcq_entry",
    },
}


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict


def get_type_schema(field_type) -> dict:
    """Convert Python type hints to JSON schema types, handling nested BaseModels."""
    # Handle None type
    if field_type is None:
        return {"type": "null"}

    # Check if it's a Pydantic model
    if isclass(field_type) and issubclass(field_type, BaseModel):
        return pydantic_to_json_schema(field_type)

    origin = get_origin(field_type)
    if origin is None:
        # Handle basic types
        if field_type == str:
            return {"type": "string"}
        elif field_type == int:
            return {"type": "integer"}
        elif field_type == float:
            return {"type": "number"}
        elif field_type == bool:
            return {"type": "boolean"}
        elif field_type == dict:
            return {"type": "object", "additionalProperties": {"type": "string"}}
        elif field_type == list:
            return {"type": "array", "items": {"type": "string"}}

    else:
        # Handle generic types (List, Dict, etc.)
        if origin == dict:
            key_type, value_type = get_args(field_type)
            return {
                "type": "object",
                "additionalProperties": get_type_schema(value_type),
            }
        elif origin == list:
            item_type = get_args(field_type)[0]
            return {"type": "array", "items": get_type_schema(item_type)}

    return {"type": "string"}  # default fallback


def pydantic_to_json_schema(model_class: Type[BaseModel]) -> dict:
    """Convert a Pydantic model to JSON schema format."""
    properties = {}
    required = []

    for field_name, field in model_class.model_fields.items():
        # Get field schema
        field_schema = get_type_schema(field.annotation)

        # Add description if available
        if field.description:
            field_schema["description"] = field.description

        properties[field_name] = field_schema

        # If field has no default value, it's required
        if field.default is None and not field.default_factory:
            required.append(field_name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "title": model_class.__name__,
    }

    # Add model description if available
    if model_class.__doc__:
        schema["description"] = model_class.__doc__.strip()

    return schema


def get_tools() -> list[dict]:

    input_schema = pydantic_to_json_schema(QuestionGeneration)

    tool = asdict(
        ToolSchema(
            name="get_mc_questions_w_reasoning",
            description=input_schema.pop("description"),
            input_schema=input_schema,
        )
    )

    # schema = pydantic_to_json_schema(MCQ_Entry)
    # schema["name"] = "get_single_mcq_entry"

    return [tool]


def process_tool_call(tool_name: str, tool_input: dict):

    if tool_name == "get_single_mcq_entry":
        return MCQ_Entry(**tool_input)
    elif tool_name == "get_mc_questions_w_reasoning":
        return QuestionGeneration(**tool_input)
    else:
        raise NotImplementedError(f"No tool post-processing for '{tool_name}'")
