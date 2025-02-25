from ollama import Message, chat


def prompt_model_with_validation(model, prompt, format):
    return chat(model,
                messages=[
                    Message(
                        role="user",
                        content="{}".format(prompt),
                    )
                ],
                format=format)
