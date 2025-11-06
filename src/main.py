import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ai_utils import ask_openai

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=openai_api_key)


def get_ai_business_area(client: OpenAI, model: str = "gpt-4o") -> str | None:
    """
    Step 1: ask the model to pick a promising business area for Agentic AI opportunities.
    """
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": (
                "Can you pick a business area that might be worth exploring "
                "for an Agentic AI opportunity?"
            ),
        }
    ]
    return ask_openai(client, messages, model)


def get_ai_pain_point(
    client: OpenAI, business_area: str, model: str = "gpt-4o"
) -> str | None:
    """
    Step 2: ask the model to identify a pain point in the given business area.
    """
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": (
                f"Can you present a pain point in the {business_area} industry — "
                "something challenging that might be ripe for an Agentic AI solution?"
            ),
        }
    ]
    return ask_openai(client, messages, model)


def get_ai_solution(
    client: OpenAI, business_area: str, pain_point: str, model: str = "gpt-4o"
) -> str | None:
    """
    Step 3: ask the model to propose a possible Agentic AI solution for the given pain point.
    """
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": (
                "Can you propose an Agentic AI solution? "
                f"The business area is {business_area} and the pain point is {pain_point}."
            ),
        }
    ]
    return ask_openai(client, messages, model)


if __name__ == "__main__":
    model = "gpt-4o"

    print("\n--- Step 1: Business Area ---")
    business_area = get_ai_business_area(client, model)
    if not business_area:
        print("No business area returned.")
        exit(1)
    print(business_area)

    print("\n--- Step 2: Pain Point ---")
    pain_point = get_ai_pain_point(client, business_area, model)
    if not pain_point:
        print("No pain point returned.")
        exit(1)
    print(pain_point)

    solution = get_ai_solution(client, business_area, pain_point, model)
    if not solution:
        print("No solution returned.")
        exit(1)
    print("\n--- Step 3: AI Solution ---")
    print(solution)
