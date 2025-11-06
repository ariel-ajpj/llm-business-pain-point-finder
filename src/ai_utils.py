from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


def ask_openai(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str = "gpt-4o",
) -> str | None:
    try:
        response: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except Exception as e:
        print(f"Error calling model: {e}")
        return None

    if not response or not response.choices:
        print("Warning: no choices returned in response")
        return None

    message = response.choices[0].message
    if message and message.content:
        return message.content.strip()

    print("Warning: response message has no content")
    return None
