import ollama
from pydantic import Field, BaseModel

from comfy.comfy_types import IO
from .categories import CHAIN_NODE_CATEGORIES, CHAIN_NODE_TOOLTIP
from .prompt import build_chain_prompt_subject, build_chain_prompt, build_context_prompt
from ..ollama_facade.instance import check_ollama_instance_running
from ..ollama_facade.model import model_names


class ChainPrompt(BaseModel):
    description: str = Field(max_length=50, min_length=5)


def _chain(model, category, keywords, prompt):
    previous_prompt = None
    if keywords:
        input_prompt = build_chain_prompt_subject(category, keywords)
    else:
        previous_prompt = build_context_prompt(prompt)
        input_prompt = build_chain_prompt(category)

    raw_response = ollama.chat(
        model=model,
        messages=[previous_prompt, input_prompt] if previous_prompt else [input_prompt],
        format=ChainPrompt.model_json_schema())

    json_response = ChainPrompt.model_validate_json(raw_response.message.content)

    final_prompt = (
    "{}, {}, {}".format(keywords if keywords else '', prompt if prompt else '', json_response.description),)

    print(final_prompt)

    return final_prompt


class OllamaChainControl:
    @classmethod
    def INPUT_TYPES(self):
        models = []
        if check_ollama_instance_running():
            models = [*model_names()]

        return {
            "required": {
                "model": (models,),
                "category": (CHAIN_NODE_CATEGORIES,),
                "keywords": (
                    "STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Subject for image generation"}),
            },
            "optional": {
                "prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt attribute"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    OUTPUT_TOOLTIPS = (CHAIN_NODE_TOOLTIP,)
    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "prompt"
    RETURN_NAMES = ("TEXT",)

    @classmethod
    def IS_CHANGED(self, model, category, keywords, prompt, seed):
        return seed

    def prompt(self, model, category, keywords, prompt, seed):
        return _chain(model, category, keywords, prompt)


class OllamaChainRandom:
    @classmethod
    def INPUT_TYPES(self):
        models = []
        if check_ollama_instance_running():
            models = [*model_names()]

        return {
            "required": {
                "model": (models,),
                "category": (CHAIN_NODE_CATEGORIES[1:],),
            },
            "optional": {
                "prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt attribute"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    OUTPUT_TOOLTIPS = (CHAIN_NODE_TOOLTIP,)
    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "prompt"
    RETURN_NAMES = ("TEXT",)

    @classmethod
    def IS_CHANGED(self, model, category, prompt, seed):
        return seed

    def prompt(self, model, category, prompt, seed):
        return _chain(model, category, None, prompt)
