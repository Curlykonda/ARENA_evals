from os import PathLike
import json
from pathlib import Path
import random

from my_evals.utils import load_jsonl
from my_evals.apis.anthropic.chat import (
    generate_response,
    format_choices,
    apply_user_format,
    apply_assistant_format,
)


def _load_json_examples(file_path: PathLike) -> list:
    with open(file_path) as f_in:
        data = json.load(f_in)

    return data


def build_few_shot_history(examples: list) -> list[dict]:
    messages = []
    for ex in examples:
        question_text = ex["question"] + "\n" + format_choices(ex["answers"])
        messages.append(apply_user_format(question_text))
        messages.append(apply_assistant_format(ex["answer_not_matching_behavior"][0]))

    return messages


# anth_data_dir = Path("chapter3_llm_evals/exercises/data/anthropic")
# anthr_dataset_name = "corrigible-more-HHH"
# file_path = anth_data_dir / f"{anthr_dataset_name}.jsonl"
# dataset = load_jsonl(file_path)
few_shot_size = 4
file_path = Path("my_evals/data/mcq_sycophancy_test.json")
dataset = _load_json_examples(file_path)
few_shot_history = build_few_shot_history(dataset[:few_shot_size])
subset = dataset[few_shot_size:]
# subset = random.sample(dataset, subset_size)

print(few_shot_history)

for question in subset:
    question_text = question["question"] + "\n" + format_choices(question["answers"])
    question_text += "\nDirectly reply with your answer choice A or B. Add brief explanation in a new line."
    response = generate_response(
        model="haiku-3", messages=few_shot_history, user=question_text, max_tokens=128
    )
    print(f"Question: {question_text} \nModel Answer: {response}\n")
