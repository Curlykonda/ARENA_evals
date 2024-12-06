from my_evals.apis.anthropic.chat import generate_formatted_response
from my_evals.apis.anthropic.tools import process_tool_call
from my_evals.log import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":

    # read config

    # load prompts and examples

    # format things

    #

    tool_name, tool_input = generate_formatted_response()

    try:
        mcq = process_tool_call(tool_name, tool_input)
    except NotImplementedError as e:
        logger.error(repr(e))
    except Exception as e:
        logger.error(f"Unexpected error: {repr(e)}")
        # break
