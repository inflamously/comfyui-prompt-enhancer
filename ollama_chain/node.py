import ollama
from pydantic import Field, BaseModel

from comfy.comfy_types import IO
from .categories import CHAIN_NODE_CATEGORIES, CHAIN_NODE_TOOLTIP
from .prompt import build_chain_prompt_subject, build_chain_prompt
from ..ollama_facade.instance import check_ollama_instance_running
from ..ollama_facade.model import model_names


class ChainPrompt(BaseModel):
    description: str = Field(max_length=50, min_length=1)


def _chain(model, category, keyword, prompt, encoder_type):
    check_ollama_instance_running()

    if prompt and encoder_type:
        input_prompt = build_chain_prompt(category, encoder_type)
    elif keyword:
        input_prompt = build_chain_prompt_subject(category, keyword)
    else:
        raise Exception("invalid chain, either subject (OllamaChainSubject) or prompt (OllamaChain) missing")

    raw_response = ollama.chat(
        model=model,
        messages=[input_prompt],
        format=ChainPrompt.model_json_schema())

    json_response = ChainPrompt.model_validate_json(raw_response.message.content)

    final_prompt = ("{}, {}".format(prompt if prompt else '', json_response.description),)

    print(final_prompt)

    return final_prompt


class OllamaChainSubject:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (model_names(),),
                "category": (CHAIN_NODE_CATEGORIES,),
                "subject": (
                    "STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Subject for image generation"}),
            },
            "optional": {
                "encoder_type": (["clip", "t5"], {"default": "clip"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    OUTPUT_TOOLTIPS = (CHAIN_NODE_TOOLTIP,)
    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "prompt"
    RETURN_NAMES = ("TEXT",)

    @classmethod
    def IS_CHANGED(self, model, category, subject, encoder_type, seed):
        return seed

    def prompt(self, model, category, subject, encoder_type, seed):
        return _chain(model, category, subject, None, encoder_type)


class OllamaChain:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (model_names(),),
                "category": (CHAIN_NODE_CATEGORIES[1:],),
            },
            "optional": {
                "prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt attribute"}),
                "encoder_type": (["clip", "t5"], {"default": "clip"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    OUTPUT_TOOLTIPS = (CHAIN_NODE_TOOLTIP,)
    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "conditioning/advanced"
    FUNCTION = "prompt"
    RETURN_NAMES = ("TEXT",)

    @classmethod
    def IS_CHANGED(self, model, category, prompt, encoder_type, seed):
        return seed

    def prompt(self, model, category, prompt, encoder_type, seed):
        return _chain(model, category, None, prompt, encoder_type)
