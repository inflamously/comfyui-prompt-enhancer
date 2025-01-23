from ollama import Message


def build_message(prompt):
    return Message(
        role="user",
        content=prompt,
    )
