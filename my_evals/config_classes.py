from dataclasses import dataclass


@dataclass
class API_Gen_Config:
    model: str = "haiku-3"
    temperature: float = 1.0
    max_tokens: int = 128

    thread_chunk_size: int = 5  # ThreadPoolExecutor config
    thread_max_workers: int = 8  # ThreadPoolExecutor config
    generation_filepath: str | None = None  # File to save model-generated questions
    score_filepath: str | None = (
        None  # File to save the scores of model-generated questions
    )
    tools: list[dict] | None = None
    tool_choice: dict | str | None = "auto"
