from custom_nodes.comfyui_prompt_enhancer.ollama_facade.message import build_message


def build_chain_prompt_subject(category, attribute):
    return build_message(
        "given word is {}, decorate {} with attribute 2-5 words. Answer as direct sentence.".format(attribute,
                                                                                                    category))


def build_chain_prompt(category, encoder_type):
    return build_message(
        "give me a 2-5 words for given category: {} in {} encoder style".format(category, encoder_type)
    )
