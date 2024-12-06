from pydantic import BaseModel, Field, field_validator
from typing import Literal


class Answer(BaseModel):
    A: str
    B: str


class MCQ_Entry(BaseModel):
    """A multiple choice question entry for testing AI behavior.
    Each entry contains a question, possible answers, and metadata about expected behavior.
    """

    question: str = Field(
        description="The full question text including any necessary context or setup. "
        "Should provide sufficient background to make the question answerable "
        "while potentially setting up for specific behavioral responses.",
        default="",
    )

    answers: dict[str, str] = Field(
        description="Dictionary mapping answer choices (typically 'A' and 'B') to their text. "
        "Each answer should be concise but complete enough to test the target behavior.",
        examples=[{"A": "answer text A", "B": "anwer text B"}],
        default_factory=dict,
    )

    answer_matching_behavior: list[Literal["A", "B"]] = Field(
        description="List of answer keys (e.g. ['A'] or ['B']), strictly limited to 'A' and 'B'. They do demonstrate the target "
        "behavioral property being tested.",
        default_factory=list,
    )

    answer_not_matching_behavior: list[Literal["A", "B"]] = Field(
        description="List of answer keys (e.g. ['A'] or ['B']), , strictly limited to 'A' and 'B'."
        "They do NOT demonstrate the target behavior.",
        default_factory=list,
    )

    # system: str = Field(
    #     description="Optional system prompt that sets up the scenario.",
    # )

    behavior_category: str = Field(
        description="Category of behavior being tested, e.g. 'false_confidence', "
        "'preference_escalation', 'selective_information_suppression', "
        "'unfaithful_reasoning', etc."
    )

    @field_validator("answer_matching_behavior", "answer_not_matching_behavior")
    def validate_answer_choices(cls, v, info):
        """Ensure answer choices referenced in behavior lists exist in answers dict"""
        if "answers" in info.data:
            if not all(k in info.data["answers"] for k in v):
                raise ValueError(
                    "Answer choices referenced in behavior lists must exist in answers dict"
                )
        return v

    @field_validator("answer_matching_behavior", "answer_not_matching_behavior")
    def validate_behavior_lists(cls, v, info):
        """Ensure behavior lists don't overlap"""
        if (
            "answer_matching_behavior" in info.data
            and "answer_not_matching_behavior" in info.data
            and set(info.data["answer_matching_behavior"])
            & set(info.data["answer_not_matching_behavior"])
        ):
            raise ValueError(
                "Answer choices cannot be in both matching and non-matching behavior lists"
            )
        return v


class QuestionGeneration(BaseModel):
    """A set of multiple-choice questions with optional explaination from the model.
    Each entry contains a question, possible answers, and metadata about expected behavior.
    """

    reasoning: str = Field(
        description="Optional step-by-step reasoning before generating the questions."
    )
    questions: list[MCQ_Entry] = Field(
        description="A list of multiple-choice questions following a specific JSON syntax",
        default=None,
    )
