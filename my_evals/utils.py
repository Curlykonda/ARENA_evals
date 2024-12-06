from enum import Enum
from os import PathLike
from pydantic import BaseModel, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
    retry_if_exception_type,
)
from anthropic import APITimeoutError, RateLimitError, InternalServerError
import json


class StopReason(Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"

    def __str__(self):
        return self.value


class LLMResponse(BaseModel):
    model_id: str
    completion: str
    stop_reason: StopReason
    usage: dict[str, int]
    api_duration: float | None = None

    @field_validator("stop_reason", mode="before")
    def parse_stop_reason(cls, v):
        if v in ["length", "max_tokens"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop", "stop_sequence"]:
            return StopReason.STOP_SEQUENCE
        raise ValueError(f"Invalid stop reason: {v}")

    @field_validator("usage", mode="before")
    def parse_usage(cls, v):
        default = {"input_tokens": 0, "output_tokens": 0}
        if v is not None:
            return {k: dict(v).get(k, default[k]) for k in default}
        return default

    def to_dict(self):
        return {**self.model_dump(), "stop_reason": str(self.stop_reason)}


@retry(
    stop=stop_after_attempt(8),
    wait=wait_fixed(1) + wait_random(0, 4),
    retry=retry_if_exception_type(
        (APITimeoutError, RateLimitError, InternalServerError)
    ),
)
def retry_on_anthropic_errors(function, *args, **kwargs):
    return function(*args, **kwargs)


def load_jsonl(file_name: PathLike) -> list:
    data = []
    with open(file_name) as f_in:
        for line in f_in.readlines():
            data.append(json.loads(line))

    return data
