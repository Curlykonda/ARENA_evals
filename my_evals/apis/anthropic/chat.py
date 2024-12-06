# docs: https://docs.anthropic.com/en/api/getting-started
import os
from enum import Enum

import anthropic
from my_evals.log import get_logger
from my_evals.config_classes import API_Gen_Config


logger = get_logger(__name__)

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


class Claude(Enum):
    SONNET_3_5 = "claude-3-5-sonnet-20241022"
    SONNET_3 = "claude-3-sonnet-20240229"
    HAIKU_3 = "claude-3-haiku-20240307"
    OPUS_3 = "claude-3-opus-20240229"


CLAUDE_MAPPING = {
    "sonnet-3-5": Claude.SONNET_3_5,
    "sonnet-3": Claude.SONNET_3,
    "haiku-3": Claude.HAIKU_3,
    "opus-3": Claude.OPUS_3,
}


def generate_formatted_response(
    config: API_Gen_Config,
    messages: list[dict[str, str]] | None = None,
    user: str | None = None,
    system: str | None = None,
    verbose: bool = False,
) -> tuple[str, dict] | None:
    """
    Generate a response using the Anthropic API.

    Args:
        client (OpenAI): The OpenAI client object.
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        messages (Optional[List[dict]]): A list of message dictionaries with 'role' and 'content' keys.
                                         If provided, this takes precedence over user and system args.
        user (Optional[str]): The user's input message. Used if messages is None.
        system (Optional[str]): The system message to set the context. Used if messages is None.
        temperature (float): Controls randomness in output. Higher values make output more random. Default is 1.
        verbose (bool): If True, prints the input messages before making the API call. Default is False.

    Returns:
        str: The generated response from the OpenAI model.

    Note:
        - If both 'messages' and 'user'/'system' are provided, 'messages' takes precedence.
        - The function uses a retry mechanism with exponential backoff for handling transient errors.
        - The client object should be properly initialized before calling this function.
    """

    if config.model in CLAUDE_MAPPING:
        model = CLAUDE_MAPPING[config.model]
    else:
        model = Claude(config.model)

    if verbose:
        logger.info(messages)

    if messages is None:
        assert user or system, "Provide input to the model!"
        messages = apply_message_format(user, system)
    else:
        if user or system:
            messages.extend(apply_message_format(user))

    response = client.messages.create(
        model=model.value,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=messages,
        tools=config.tools,
        # tool_choice=config.tool_choice,
    )

    if response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")

        return tool_use.name, tool_use.input
    else:
        raise ValueError(f"Tool use failed. Stop reason '{response.stop_reason}'")


def generate_response(
    config: API_Gen_Config,
    messages: list[dict[str, str]] | None = None,
    user: str | None = None,
    system: str | None = None,
    verbose: bool = False,
) -> str | None:
    """
    Generate a response using the Anthropic API.

    Args:
        client (OpenAI): The OpenAI client object.
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        messages (Optional[List[dict]]): A list of message dictionaries with 'role' and 'content' keys.
                                         If provided, this takes precedence over user and system args.
        user (Optional[str]): The user's input message. Used if messages is None.
        system (Optional[str]): The system message to set the context. Used if messages is None.
        temperature (float): Controls randomness in output. Higher values make output more random. Default is 1.
        verbose (bool): If True, prints the input messages before making the API call. Default is False.

    Returns:
        str: The generated response from the OpenAI model.

    Note:
        - If both 'messages' and 'user'/'system' are provided, 'messages' takes precedence.
        - The function uses a retry mechanism with exponential backoff for handling transient errors.
        - The client object should be properly initialized before calling this function.
    """

    if config.model in CLAUDE_MAPPING:
        model = CLAUDE_MAPPING[config.model]
    else:
        model = Claude(config.model)

    if verbose:
        logger.info(messages)

    if messages is None:
        assert user or system, "Provide input to the model!"
        messages = apply_message_format(user, system)
    else:
        if user or system:
            messages.extend(apply_message_format(user))

    response = client.messages.create(
        model=model.value,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=messages,
    )

    return response.content[0].text


def make_api_call(
    messages: list[dict[str, str]],
    model_name: str | Claude,
    max_tokens: int = 1024,
    system_prompt: str = "",
):
    model_name = Claude(model_name)

    response = client.messages.create(
        system=system_prompt,
        model=model_name.value,
        max_tokens=max_tokens,
        messages=messages,
    )

    return response


def make_cached_api_call(
    messages: list[dict[str, str]],
    model_name: str | Claude,
    max_tokens: int = 1024,
    system: str | list[dict[str, str]] = "",
):

    response = client.messages.create(
        system=system,
        model=model_name.value,
        max_tokens=max_tokens,
        messages=messages,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    return response


def apply_system_format(content: str) -> dict:
    return {"role": "system", "content": content}


def apply_user_format(content: str) -> dict:
    return {"role": "user", "content": content}


def apply_assistant_format(content: str) -> dict:
    return {"role": "assistant", "content": content}


def format_choices(answers: dict[str, str]) -> str:
    options_text = "Choices:\n"
    for option, answer in answers.items():
        options_text += f"\n({option}) {answer}"

    return options_text


def apply_message_format(user: str, system: str | None = None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def pprint_messages(messages: list[dict]) -> str:
    return "".join(
        [f"{message['role'].upper()}:\n{message['content']}\n" for message in messages]
    )
