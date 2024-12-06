from chapter3_llm_evals.exercises.utils import pretty_print_questions
from my_evals.apis.anthropic.chat import generate_formatted_response
from my_evals.apis.anthropic.tools import process_tool_call, get_tools
from my_evals.config_classes import API_Gen_Config


if __name__ == "__main__":

    config = API_Gen_Config()
    config.max_tokens = 2048
    config.tools = get_tools()
    config.tool_choice = {"type": "tool", "name": "get_mc_questions_w_reasoning"}

    simple_instruction = "Generate 4 factual questions about Germany's culture with answer options limited to A and B."
    tool_name, tool_input = generate_formatted_response(
        config, user=simple_instruction, verbose=True
    )

    print("Raw tool input:\n", tool_input)

    questions = process_tool_call(tool_name, tool_input)

    print(f'\nModel reasoning:\n{tool_input["reasoning"]}')
    print("\nModel generation:\n")
    pretty_print_questions(tool_input["questions"])
