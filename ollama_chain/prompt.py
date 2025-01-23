from custom_nodes.comfyui_prompt_enhancer.ollama_facade.message import build_message


def build_chain_prompt_subject(category, keywords):
    return build_message(
        "given word is {}, decorate {} with keywords of 2-5 words. Answer as direct sentence.".format(keywords,
                                                                                                    category))


def build_chain_prompt(category):
    return build_message(
        "give me a 2-5 words for given image generation category: {}".format(category)
    )


def build_context_prompt(prompt):
    return build_message(
        "this is my previous prompt: {}".format(prompt)
    )