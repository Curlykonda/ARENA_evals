import os
import time
from enum import Enum

import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from chapter.apis.anthropic.chat import Claude
from src.utils import get_logger

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
logger = get_logger(__name__)


class BatchStatus(Enum):
    IN_PROGRESS = "in_progress"
    CANCELING = "canceling"
    ENDED = "ended"


class BatchResult(Enum):
    SUCCEEDED = "succeeded"
    ERRORED = "errored"


class BatchHandler:
    def __init__(self):
        self.requests = []
        self.default_request_name = "my-request-{i}"
        self.default_sleep = 120

    def add_request(
        self,
        messages: list[dict[str, str]],
        model_name: str | Claude,
        max_tokens: int = 1024,
        system: str | list[dict[str, str]] = "",
        request_name: str | None = None,
    ) -> str:
        model_name = Claude(model_name)

        if request_name is None:
            request_name = self.default_request_name.format(i=len(self.requests))

        self.requests.append(
            Request(
                custom_id=request_name,
                params=MessageCreateParamsNonStreaming(
                    model=model_name.value,
                    max_tokens=max_tokens,
                    messages=messages,
                    system=system,
                ),
            )
        )
        return request_name

    def submit_batch_message(self) -> str:
        response = client.beta.messages.batches.create(requests=self.requests)
        logger.debug(f"Submitted batch '{response.id}'")
        return response.id

    def clear_requests(self):
        if len(self.requests):
            self.requests = []

    def poll_batch_done(self, batch_id: str) -> BatchStatus:
        response = client.beta.messages.batches.retrieve(batch_id)
        return BatchStatus(response.processing_status)

    def retrieve_batch_results(self, batch_id: str) -> tuple[list[str], list[str]]:
        ids, texts = [], []
        for batch_result in client.beta.messages.batches.results(batch_id):
            if batch_result.result.type == BatchResult.SUCCEEDED.value:
                texts.append(batch_result.result.message.content[0].text)
                ids.append(batch_result.custom_id)
            elif batch_result.result.type == BatchResult.ERRORED.value:
                logger.warning(
                    f"[{batch_id}]: error status for request '{batch_result.custom_id}' with {repr(batch_result.result.error)}"
                )

        return ids, texts

    def wait(self, sleep: int | None = None) -> None:
        if not sleep:
            sleep = self.default_sleep
        time.sleep(sleep)

    def list_batches(self, limit: int = 20) -> list[str]:
        response = client.beta.messages.batches.list(limit=limit)
        return [d.id for d in response.data]
